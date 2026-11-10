"""Persistence for VoxelData arrays, datasets, and trees."""

import json
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from voxeldata._dims import VOXEL_DIMS, WORLD_DIMS
from voxeldata.geometry import attach_voxel_to_world_index
from voxeldata.validation import ensure_voxeldata

_GEOMETRY_ARRAY_ATTRS = (
    "voxel_to_world",
    "voxel_to_world_named_affines",
    "voxel_to_world_geometry_dims",
)
"""Structured geometry attributes JSON-encoded for NetCDF storage."""

_TREE_NAME_ATTR = "__voxeldata_tree_name__"
"""Reserved root attribute preserving a DataTree's otherwise unstored name."""


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


def save(
    data: xr.DataArray | xr.Dataset | xr.DataTree, path: str | Path, **kwargs: Any
) -> Any:
    """Save a VoxelData array, dataset, or tree.

    Derived world coordinates are omitted. Geometry and named frames are stored
    as metadata and rebuilt by the explicit loaders. The input is not mutated.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset or xarray.DataTree
        VoxelData array or container. Spatial variables within a dataset share
        one grid; spatial tree nodes must be leaves with local coordinates.
        Nonspatial variables and empty grouping nodes are preserved. Spatial
        variables are canonicalized before saving without mutating the input.
    path : str or pathlib.Path
        Output path ending in `.zarr`, `.nc`, or `.netcdf`.
    **kwargs : Any
        Arguments forwarded to the corresponding Xarray writer, including
        overwrite modes, encoding, engine, and `compute`. Tree encoding is keyed
        by absolute group path, then variable name.

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
        If the input type is unsupported, a container is saved to NetCDF, or
        attributes are unsupported by NetCDF.
    ImportError
        If the chosen storage backend is not installed.

    Notes
    -----
    Install `voxeldata[io]` for Zarr and NetCDF backends. Zarr attributes are
    converted to JSON-compatible values; unsupported objects are dropped with a
    warning. NetCDF user attributes must satisfy the selected Xarray backend's
    restrictions. Geometry metadata is JSON-encoded for NetCDF, which cannot
    store nested dictionaries or multidimensional attribute arrays.

    Containers are supported only for Zarr. All outputs in a container use one
    shared writer, preserving shared Dask tasks without persisting signal data.
    Deferred writes create metadata eagerly and may leave a partial store on
    failure. This is chunked execution, not append/update support.

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
    if not isinstance(data, xr.DataArray | xr.Dataset | xr.DataTree):
        raise TypeError("Expected an xarray.DataArray, Dataset, or DataTree.")
    if storage_format == "netcdf" and not isinstance(data, xr.DataArray):
        raise TypeError("VoxelData dataset and tree I/O supports Zarr only.")
    if isinstance(data, xr.DataArray):
        prepared = _serialize_array(ensure_voxeldata(data), storage_format)
        if storage_format == "netcdf":
            return prepared.to_netcdf(path, **kwargs)
        _sanitize_zarr_attrs(prepared)
    elif isinstance(data, xr.Dataset):
        prepared = _map_dataset(data, restore=False, group="/")
    else:
        if kwargs.get("write_inherited_coords", False):
            raise ValueError("Spatial tree coordinates must remain node-local.")
        prepared = _map_tree(data, restore=False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Consolidated metadata is currently not part in the Zarr format 3 "
            "specification.",
        )
        return prepared.to_zarr(path, **kwargs)


def load_array(
    path: str | Path, variable: str | None = None, **kwargs: Any
) -> xr.DataArray:
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
        including `chunks` and `engine`. By default, use storage-aligned Dask
        chunks (`chunks={}`) when Dask is installed, or backend lazy indexing
        otherwise. Pass `chunks=None` to disable Dask or a chunk mapping to
        override storage chunking.

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
    """
    storage_format = _get_format(path)
    if "chunks" not in kwargs:
        kwargs["chunks"] = {} if find_spec("dask") is not None else None
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
        data = dataset[variable]
        # Match Xarray's open_dataarray convention for unnamed or conflicting names.
        if data.name == "__xarray_dataarray_variable__":
            data = data.rename(dataset.attrs.get("__xarray_dataarray_name__"))
        data = _restore_array(data, storage_format)
    except BaseException:
        dataset.close()
        raise
    data.set_close(dataset.close)
    return data


def load_dataset(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Open a Zarr dataset and restore its node-local VoxelData geometry.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path ending in `.zarr`.
    **kwargs : Any
        Arguments forwarded to `xarray.open_zarr`, including `group` and `chunks`.
        Chunk defaults match [load_array][voxeldata.io.load_array].

    Returns
    -------
    xarray.Dataset
        Dataset whose spatial variables are directly selectable as validated
        VoxelData arrays. Nonspatial variables are preserved. Close when done.

    Raises
    ------
    ValueError
        If the format, spatial variables, or shared geometry is invalid.
    TypeError
        If persisted geometry metadata has an invalid type.
    """
    _require_zarr(path)
    kwargs.setdefault("chunks", {} if find_spec("dask") is not None else None)
    dataset = xr.open_zarr(path, **kwargs)
    try:
        restored = _map_dataset(dataset, restore=True, group=kwargs.get("group", "/"))
    except BaseException:
        dataset.close()
        raise
    restored.set_close(dataset.close)
    return restored


def load_tree(path: str | Path, **kwargs: Any) -> xr.DataTree:
    """Open a complete Zarr tree and restore each spatial leaf's geometry.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path ending in `.zarr`.
    **kwargs : Any
        Arguments forwarded to `xarray.open_datatree` with `engine="zarr"`.
        Chunk defaults match [load_array][voxeldata.io.load_array].

    Returns
    -------
    xarray.DataTree
        Complete hierarchy, including nonspatial variables and empty groups.
        Spatial variables are directly selectable as validated VoxelData arrays.
        Close the tree when done.

    Raises
    ------
    ValueError
        If the format, spatial variables, or tree layout is invalid.
    TypeError
        If persisted geometry metadata has an invalid type.
    """
    _require_zarr(path)
    kwargs.setdefault("chunks", {} if find_spec("dask") is not None else None)
    tree = xr.open_datatree(path, engine="zarr", **kwargs)
    try:
        restored = _map_tree(tree, restore=True)
    except BaseException:
        tree.close()
        raise
    restored.set_close(tree.close)
    return restored


def _require_zarr(path: str | Path) -> None:
    """Require the Zarr format for container I/O.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path.

    Raises
    ------
    ValueError
        If the path does not identify a Zarr store.
    """
    if _get_format(path) != "zarr":
        raise ValueError("VoxelData dataset and tree I/O supports Zarr only.")


def _serialize_array(
    data: xr.DataArray, storage_format: Literal["zarr", "netcdf"]
) -> xr.DataArray:
    """Replace a canonical array's derived coordinates with geometry metadata.

    Parameters
    ----------
    data : xarray.DataArray
        Canonical, validated VoxelData array.
    storage_format : {"zarr", "netcdf"}
        Target format.

    Returns
    -------
    xarray.DataArray
        Shallow copy with serializable geometry and no derived world coordinates.
    """
    metadata = {
        "voxel_to_world": data.vd.voxel_to_world,
        "voxel_to_world_units": data.vd.units,
        "voxel_to_world_named_affines": data.vd.affines,
        "voxel_to_world_geometry_dims": data.vd.geometry_dims,
    }
    data = data.drop_vars(WORLD_DIMS).copy(deep=False)
    data.attrs = {**data.attrs, **metadata}
    if storage_format == "netcdf":
        for key in _GEOMETRY_ARRAY_ATTRS:
            data.attrs[key] = json.dumps(_convert_to_json_serializable(data.attrs[key]))
    return data


def _restore_array(
    data: xr.DataArray, storage_format: Literal["zarr", "netcdf"]
) -> xr.DataArray:
    """Rebuild and validate an array using its persisted geometry metadata.

    Parameters
    ----------
    data : xarray.DataArray
        Stored array.
    storage_format : {"zarr", "netcdf"}
        Source format.

    Returns
    -------
    xarray.DataArray
        Canonical, validated VoxelData array with consumed geometry attributes.

    Raises
    ------
    ValueError
        If geometry metadata is missing or invalid.
    TypeError
        If geometry metadata has an invalid type.
    """
    data = data.copy(deep=False)
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
    return ensure_voxeldata(data)


def _sanitize_zarr_attrs(data: xr.DataArray | xr.Dataset) -> None:
    """Make attributes on a prepared object and its variables Zarr-safe.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Prepared object to update without touching the original input.
    """
    data.attrs = _make_attrs_zarr_safe(data.attrs)
    variables = (
        data.variables.values()
        if isinstance(data, xr.Dataset)
        else data.coords.values()
    )
    for variable in variables:
        variable.attrs = _make_attrs_zarr_safe(variable.attrs)


def _map_dataset(dataset: xr.Dataset, *, restore: bool, group: str) -> xr.Dataset:
    """Serialize or restore a dataset's spatial variables without alignment.

    Parameters
    ----------
    dataset : xarray.Dataset
        Node-local dataset.
    restore : bool
        Whether to restore persisted geometry rather than serialize it.
    group : str
        Group path for validation errors.

    Returns
    -------
    xarray.Dataset
        Shallow copy with transformed spatial variables and untouched diagnostics.

    Raises
    ------
    ValueError
        If a spatial variable is invalid or variables do not share one index.
    TypeError
        If persisted geometry metadata has an invalid type.
    """
    spatial = {}
    reference = None
    for name, data in dataset.data_vars.items():
        if (
            not (set(data.dims) & {*VOXEL_DIMS, *WORLD_DIMS})
            and "voxel_to_world" not in data.attrs
        ):
            continue
        try:
            array = _restore_array(data, "zarr") if restore else ensure_voxeldata(data)
            index = array.vd.index
            if reference is None:
                reference = array
            else:
                first = reference.vd.index
                if (
                    first.units != index.units
                    or not first.equals(index)
                    or first.named_affines.keys() != index.named_affines.keys()
                    or any(
                        first.named_affines[key].shape != index.named_affines[key].shape
                        or not np.allclose(
                            first.named_affines[key], index.named_affines[key]
                        )
                        for key in first.named_affines
                    )
                ):
                    raise ValueError(
                        "Spatial variables must share one grid and named frames."
                    )
            spatial[name] = array if restore else _serialize_array(array, "zarr")
        except (ValueError, TypeError) as error:
            raise type(error)(f"Group {group!r}, variable {name!r}: {error}") from error

    result = dataset.drop_vars(
        [name for name in WORLD_DIMS if name in dataset.coords]
    ).copy(deep=False)
    if reference is not None:
        result = result.assign_coords(
            {dim: reference.coords[dim] for dim in VOXEL_DIMS}
        )
    for name, array in spatial.items():
        # Variable assignment is positional: never align or broadcast diagnostics.
        result[name] = array.variable
    if restore and reference is not None:
        # assign_coords recreates index variables, losing canonical coordinate order.
        coords = xr.Coordinates(
            {**result.coords, **reference.coords},
            indexes={**result.xindexes, **reference.xindexes},
        )
        result = xr.Dataset(
            {name: data.variable for name, data in result.data_vars.items()},
            coords=coords,
            attrs=result.attrs,
        )
    if not restore:
        _sanitize_zarr_attrs(result)
    return result


def _map_tree(tree: xr.DataTree, *, restore: bool) -> xr.DataTree:
    """Transform local datasets while enforcing spatial leaf nodes.

    Parameters
    ----------
    tree : xarray.DataTree
        Source hierarchy.
    restore : bool
        Whether to restore persisted geometry rather than serialize it.

    Returns
    -------
    xarray.DataTree
        Transformed hierarchy retaining empty nodes and node attributes.

    Raises
    ------
    ValueError
        If a non-leaf carries spatial coordinates or a spatial variable is invalid.
    TypeError
        If persisted geometry metadata has an invalid type.
    """
    nodes = {}
    name = tree.name
    for path, node in tree.subtree_with_keys:
        dataset = node.to_dataset(inherit=False).copy(deep=False)
        if node is tree:
            if restore:
                name = dataset.attrs.pop(_TREE_NAME_ATTR, name)
                if name is not None and not isinstance(name, str):
                    raise TypeError("Persisted tree name must be a string.")
            elif _TREE_NAME_ATTR in dataset.attrs:
                raise ValueError(f"Root attribute {_TREE_NAME_ATTR!r} is reserved.")
            elif name is not None:
                dataset.attrs[_TREE_NAME_ATTR] = name
        if node.children and {*VOXEL_DIMS, *WORLD_DIMS} & (
            set(dataset.dims) | set(dataset.coords)
        ):
            raise ValueError(
                f"Group {node.path!r}: spatial grids must be local to leaf nodes; "
                "use grouping ancestors without spatial coordinates."
            )
        nodes[path] = _map_dataset(dataset, restore=restore, group=node.path)
    return xr.DataTree.from_dict(nodes, name=name)
