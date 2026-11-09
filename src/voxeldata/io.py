"""Zarr and NetCDF persistence for VoxelData arrays.

Portions of this file are derived from ConfUSIus, which is licensed under the
BSD-3-Clause License. See `NOTICE` file for details.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from voxeldata._dims import WORLD_DIMS
from voxeldata.geometry import attach_voxel_to_world_index
from voxeldata.validation import ensure_voxeldata

_GEOMETRY_ARRAY_ATTRS = (
    "voxel_to_world",
    "voxel_to_world_named_affines",
    "voxel_to_world_geometry_dims",
)
"""Structured geometry attributes JSON-encoded for NetCDF storage."""


def _get_format(path: str | Path) -> Literal["zarr", "netcdf"]:
    """Determine the storage format from a path suffix.

    Parameters
    ----------
    path : str or pathlib.Path
        Store or file path.

    Returns
    -------
    {"zarr", "netcdf"}
        Storage format.

    Raises
    ------
    ValueError
        If the suffix is unsupported.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".zarr":
        return "zarr"
    if suffix in (".nc", ".netcdf"):
        return "netcdf"
    raise ValueError(
        f"Unsupported file extension in {str(path)!r}. "
        "Supported extensions are .zarr, .nc, and .netcdf."
    )


def _convert_to_json_serializable(value: Any) -> Any:
    """Convert NumPy values recursively to JSON-compatible containers.

    Parameters
    ----------
    value : Any
        Attribute value.

    Returns
    -------
    Any
        Value with NumPy arrays and scalars converted to Python objects.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _convert_to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_convert_to_json_serializable(item) for item in value]
    return value


def _make_attrs_zarr_safe(attrs: dict[Any, Any]) -> dict[Any, Any]:
    """Convert Zarr attributes and warn about values that must be dropped.

    Parameters
    ----------
    attrs : dict
        Array or coordinate attributes.

    Returns
    -------
    dict
        JSON-compatible copy of the attributes.
    """
    safe = {}
    dropped = []
    for key, value in attrs.items():
        converted = _convert_to_json_serializable(value)
        try:
            json.dumps(converted)
        except (TypeError, ValueError):
            dropped.append(key)
        else:
            safe[key] = converted
    if dropped:
        warnings.warn(
            f"Dropping non-JSON-serializable attrs from Zarr store: {dropped}.",
            stacklevel=3,
        )
    return safe


def save(data: xr.DataArray, path: str | Path, **kwargs: Any) -> Any:
    """Save a VoxelData array to Zarr or NetCDF.

    Derived world coordinates are omitted. Geometry and named frames are stored
    as metadata and rebuilt by [load][voxeldata.io.load]. The input is not mutated.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array. Scalar-selected voxel dimensions are restored and
        dimensions are canonicalized before saving.
    path : str or pathlib.Path
        Output path ending in `.zarr`, `.nc`, or `.netcdf`.
    **kwargs : Any
        Arguments forwarded to `DataArray.to_zarr` or `DataArray.to_netcdf`,
        including overwrite modes, encoding, engine, and `compute`.

    Returns
    -------
    Any
        Underlying Xarray writer's result, including its delayed write when
        `compute=False` is requested.

    Raises
    ------
    ValueError
        If the extension is unsupported or the input is not valid VoxelData.
    TypeError
        If the input is not a DataArray or attributes are unsupported by NetCDF.
    ImportError
        If the chosen storage backend is not installed.

    Notes
    -----
    Install `voxeldata[io]` for Zarr and NetCDF backends. Zarr attributes are
    converted to JSON-compatible values; unsupported objects are dropped with a
    warning. NetCDF user attributes must satisfy the selected Xarray backend's
    restrictions. Geometry metadata is JSON-encoded for NetCDF, which cannot
    store nested dictionaries or multidimensional attribute arrays.

    Existing-file behavior follows Xarray: Zarr defaults to non-overwriting
    creation, while NetCDF defaults to overwriting. Pass `mode` explicitly when
    needed. Incremental Zarr writes are unsupported because they do not merge
    store-wide affine stacks and named frames.
    """
    storage_format = _get_format(path)
    if storage_format == "zarr" and (
        kwargs.get("append_dim") is not None
        or kwargs.get("region") is not None
        or kwargs.get("mode") in ("a", "a-", "r+")
    ):
        raise ValueError(
            "Incremental Zarr writes are unsupported; save the complete array "
            "so geometry metadata stays consistent."
        )
    data = ensure_voxeldata(data).copy(deep=False)
    metadata = {
        "voxel_to_world": data.vd.voxel_to_world,
        "voxel_to_world_units": data.vd.units,
        "voxel_to_world_named_affines": data.vd.affines,
        "voxel_to_world_geometry_dims": data.vd.geometry_dims,
    }
    data = data.drop_vars(WORLD_DIMS)
    data.attrs = {**data.attrs, **metadata}
    if storage_format == "netcdf":
        for key in _GEOMETRY_ARRAY_ATTRS:
            data.attrs[key] = json.dumps(_convert_to_json_serializable(data.attrs[key]))
        return data.to_netcdf(path, **kwargs)

    data.attrs = _make_attrs_zarr_safe(data.attrs)
    for coord in data.coords.values():
        coord.attrs = _make_attrs_zarr_safe(coord.attrs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 "
            "specification.",
        )
        return data.to_zarr(path, **kwargs)


def load(path: str | Path, variable: str | None = None, **kwargs: Any) -> xr.DataArray:
    """Open a saved VoxelData array and rebuild its spatial index.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path ending in `.zarr`, `.nc`, or `.netcdf`.
    variable : str, optional
        Data variable to read. Required for datasets containing multiple data
        variables; a single variable is selected automatically.
    **kwargs : Any
        Arguments forwarded to `xarray.open_zarr` or `xarray.open_dataset`,
        including `chunks` and `engine`. The default `chunks=None` uses backend
        lazy indexing without Dask; pass `chunks={}` for Dask-backed arrays.

    Returns
    -------
    xarray.DataArray
        Validated array with a restored `VoxelToWorldIndex`. Signal data remains
        lazy when supported by the backend. Call `.close()` to release resources.

    Raises
    ------
    ValueError
        If the extension, geometry metadata, or restored VoxelData is invalid,
        or a variable cannot be selected unambiguously.
    TypeError
        If persisted geometry metadata has an invalid type.
    KeyError
        If the requested variable is absent.
    ImportError
        If the chosen storage backend is not installed.

    Notes
    -----
    Zarr metadata written by the ConfUSIus VoxelData integration is also accepted.
    Arbitrary Xarray files without voxel-to-world metadata are not VoxelData;
    open them with Xarray and construct their geometry explicitly instead.
    """
    storage_format = _get_format(path)
    kwargs.setdefault("chunks", None)
    dataset = (
        xr.open_zarr(path, **kwargs)
        if storage_format == "zarr"
        else xr.open_dataset(path, **kwargs)
    )
    try:
        if variable is None:
            if len(dataset.data_vars) != 1:
                raise ValueError(
                    "Specify variable when the dataset does not contain exactly "
                    "one data variable."
                )
            variable = next(iter(dataset.data_vars))
        data = dataset[variable].copy(deep=False)
        # Match Xarray's open_dataarray convention for unnamed or conflicting names.
        if data.name == "__xarray_dataarray_variable__":
            data = data.rename(dataset.attrs.get("__xarray_dataarray_name__"))
        if "voxel_to_world" not in data.attrs:
            raise ValueError(
                "Missing attrs['voxel_to_world']; open foreign files with Xarray "
                "and construct VoxelData geometry explicitly."
            )
        if storage_format == "netcdf":
            for key in _GEOMETRY_ARRAY_ATTRS:
                if key in data.attrs:
                    try:
                        data.attrs[key] = json.loads(data.attrs[key])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"Invalid geometry metadata in {key!r}."
                        ) from error
        affine = data.attrs.pop("voxel_to_world")
        units = data.attrs.pop("voxel_to_world_units", "mm")
        frames = data.attrs.pop("voxel_to_world_named_affines", None)
        geometry_dims = data.attrs.pop("voxel_to_world_geometry_dims", None)
        if not isinstance(units, str):
            raise TypeError("voxel_to_world_units must be a string.")
        if frames is not None and not isinstance(frames, dict):
            raise TypeError("voxel_to_world_named_affines must be a mapping.")
        if geometry_dims is not None and (
            not isinstance(geometry_dims, list | tuple)
            or not all(isinstance(dim, str) for dim in geometry_dims)
        ):
            raise TypeError(
                "voxel_to_world_geometry_dims must be an ordered sequence of names."
            )
        data = attach_voxel_to_world_index(
            data,
            affine,
            units=units,
            named_affines=frames,
            geometry_dims=None if geometry_dims is None else tuple(geometry_dims),
        )
        data = ensure_voxeldata(data)
    except BaseException:
        dataset.close()
        raise
    data.set_close(dataset.close)
    return data
