"""Constructor helpers for building VoxelData arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from voxeldata._dims import CORE_DIMS, GRID_DIM, TIME_DIM, VOXEL_DIMS, WORLD_DIMS
from voxeldata.geometry import attach_voxel_to_world_index
from voxeldata.validation import require_positive_finite, validate_voxeldata

_TIME_UNITS = "s"
"""Physical units attached to the `time` coordinate by default."""


def _require_spacing(dim: str, spacing: float | None) -> float:
    """Return a finite positive coordinate spacing.

    Parameters
    ----------
    dim : str
        Dimension whose spacing is required.
    spacing : float, optional
        Candidate spacing value.

    Returns
    -------
    float
        Validated spacing.

    Raises
    ------
    ValueError
        If `spacing` is not provided or is not positive and finite.
    """
    if spacing is None:
        hint = "dt" if dim == TIME_DIM else f"d{dim}"
        raise ValueError(
            f"Spacing for dimension {dim!r} is required. Provide {hint} or an "
            f"explicit {dim!r} coordinate with enough information to infer spacing."
        )
    return require_positive_finite(spacing, f"Spacing for dimension {dim!r}")


def _coordinate_values_and_attrs(
    dim: str, coords: Mapping[str, npt.ArrayLike | xr.DataArray]
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Return explicit coordinate values and attrs for `dim` when provided.

    Parameters
    ----------
    dim : str
        Coordinate name.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray]
        Explicit coordinate mapping.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any]] or None
        Coordinate values and attributes, or `None` when `dim` is absent.
    """
    if dim not in coords:
        return None
    coord = coords[dim]
    if isinstance(coord, xr.DataArray):
        return np.atleast_1d(np.asarray(coord.values)), coord.attrs.copy()
    return np.atleast_1d(np.asarray(coord)), {}


def _validate_coordinate_shape(dim: str, values: np.ndarray, size: int) -> None:
    """Validate a one-dimensional coordinate length.

    Parameters
    ----------
    dim : str
        Coordinate name.
    values : numpy.ndarray
        Coordinate values.
    size : int
        Expected length.

    Raises
    ------
    ValueError
        If `values` is not one-dimensional with length `size`.
    """
    if values.ndim != 1 or values.size != size:
        raise ValueError(
            f"Coordinate {dim!r} must be 1D with length {size}, got shape "
            f"{values.shape}."
        )


def _coordinate_dataarray(
    dim: str,
    size: int,
    *,
    coords: Mapping[str, npt.ArrayLike | xr.DataArray],
    spacings: Mapping[str, float | None],
    origins: Mapping[str, float | None],
) -> xr.DataArray:
    """Build one non-spatial dimension coordinate.

    Parameters
    ----------
    dim : str
        Dimension name.
    size : int
        Expected coordinate length.
    coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray]
        Explicit coordinates provided by the caller.
    spacings : mapping[str, float or None]
        Per-dimension spacings.
    origins : mapping[str, float or None]
        Per-dimension origins used with `spacings`.

    Returns
    -------
    xarray.DataArray
        One-dimensional dimension coordinate.

    Raises
    ------
    ValueError
        If an explicit coordinate has the wrong shape or a spacing is missing.
    """
    explicit = _coordinate_values_and_attrs(dim, coords)
    if explicit is not None:
        coord_values, attrs = explicit
        _validate_coordinate_shape(dim, coord_values, size)
    elif dim in spacings:
        step = _require_spacing(dim, spacings[dim])
        origin = origins[dim]
        if origin is None:
            raise ValueError(f"Origin for dimension {dim!r} must be provided.")
        coord_values = origin + np.arange(size) * step
        attrs = {}
    else:
        coord_values = np.arange(size)
        attrs = {}

    if dim == TIME_DIM:
        attrs.setdefault("units", _TIME_UNITS)

    return xr.DataArray(coord_values, dims=(dim,), attrs=attrs)


def _validate_spatial_tuple(
    values: Sequence[float] | None, *, name: str
) -> tuple[float, float, float]:
    """Validate a positive finite `z/y/x` tuple argument.

    Parameters
    ----------
    values : sequence[float], optional
        Candidate tuple values in `z/y/x` order.
    name : str
        Argument name used in validation errors.

    Returns
    -------
    tuple[float, float, float]
        Validated tuple.

    Raises
    ------
    ValueError
        If values are missing when required, have the wrong length, or are not positive
        and finite.
    """
    if values is None:
        raise ValueError(f"{name} or voxel_to_world must be provided.")
    if len(values) != len(WORLD_DIMS):
        raise ValueError(f"{name} must have length 3 in z/y/x order.")
    z, y, x = values
    return (
        require_positive_finite(z, name),
        require_positive_finite(y, name),
        require_positive_finite(x, name),
    )


def _resolve_voxel_to_world(
    *,
    spacing: Sequence[float] | None,
    origin: Sequence[float] | None,
    direction: npt.ArrayLike | None,
    voxel_to_world: npt.ArrayLike | None,
) -> npt.NDArray[np.float64]:
    """Resolve constructor geometry to one canonical 4x4 affine.

    Parameters
    ----------
    spacing : sequence[float], optional
        World spacing in `z/y/x` order.
    origin : sequence[float], optional
        World origin in `z/y/x` order. If not provided, the world origin is zero.
    direction : numpy.typing.ArrayLike, optional
        3x3 world direction matrix in `z/y/x` row and `k/j/i` column order.
    voxel_to_world : numpy.typing.ArrayLike, optional
        4x4 homogeneous affine in `z/y/x` row and `k/j/i` column order.

    Returns
    -------
    (4, 4) numpy.ndarray
        Homogeneous voxel-to-world affine.

    Raises
    ------
    ValueError
        If geometry inputs are ambiguous or invalid.
    """
    if voxel_to_world is not None:
        if spacing is not None or origin is not None or direction is not None:
            raise ValueError(
                "voxel_to_world is mutually exclusive with spacing, origin, and "
                "direction."
            )
        affine = np.asarray(voxel_to_world, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError(
                f"voxel_to_world must have shape (4, 4), got {affine.shape}."
            )
        if not np.allclose(affine[-1], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("voxel_to_world must be a homogeneous affine.")
        return affine

    resolved_spacing = _validate_spatial_tuple(spacing, name="spacing")
    if origin is None:
        resolved_origin: tuple[float, ...] = (0.0, 0.0, 0.0)
    else:
        if len(origin) != len(WORLD_DIMS):
            raise ValueError("origin must have length 3 in z/y/x order.")
        resolved_origin = tuple(float(value) for value in origin)
        if not np.all(np.isfinite(resolved_origin)):
            raise ValueError("origin must contain finite values.")

    resolved_direction = (
        np.eye(3, dtype=np.float64)
        if direction is None
        else np.asarray(direction, dtype=np.float64)
    )
    if resolved_direction.shape != (3, 3):
        raise ValueError(
            f"direction must have shape (3, 3), got {resolved_direction.shape}."
        )
    if not np.all(np.isfinite(resolved_direction)):
        raise ValueError("direction must contain finite values.")

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = resolved_direction @ np.diag(resolved_spacing)
    affine[:3, 3] = resolved_origin
    return affine


def create_voxeldata(
    data: npt.ArrayLike,
    *,
    dims: Sequence[str],
    extra_coords: Mapping[str, npt.ArrayLike | xr.DataArray] | None = None,
    time: npt.ArrayLike | xr.DataArray | None = None,
    grid: npt.ArrayLike | xr.DataArray | None = None,
    pose: npt.ArrayLike | xr.DataArray | None = None,
    k: npt.ArrayLike | xr.DataArray | None = None,
    j: npt.ArrayLike | xr.DataArray | None = None,
    i: npt.ArrayLike | xr.DataArray | None = None,
    dt: float | None = None,
    t0: float | npt.ArrayLike = 0.0,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    direction: npt.ArrayLike | None = None,
    voxel_to_world: npt.ArrayLike | None = None,
    geometry_dims: tuple[str, ...] | None = None,
    affines: Mapping[str, npt.ArrayLike] | None = None,
    units: str = "mm",
    name: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Build a VoxelData array from a raw array.

    Parameters
    ----------
    data : numpy.typing.ArrayLike
        Raw array. Any object exposing ``shape`` and usable by xarray is accepted,
        including Dask arrays.
    dims : sequence[str]
        Input dimension names. Native voxel dimensions are `i`/`j`/`k` (always
        present in the result, added as singleton dimensions when missing). `grid`
        and `time` are optional. Any other extra dimensions are allowed. The returned
        DataArray has dimensions `(*nonspatial_dims, k, j, i)`, preserving the input
        order of all nonspatial dimensions.
    extra_coords : mapping[str, numpy.typing.ArrayLike or xarray.DataArray], optional
        Coordinates for non-core dimensions only.
    time : numpy.typing.ArrayLike or xarray.DataArray, optional
        Coordinates for the `time` dimension. A multidimensional DataArray names
        `time` and any additional nonspatial timestamp dependencies, independently
        of `geometry_dims`. Values must increase along `time` for each series.
        A 2D NumPy array retains legacy per-`grid` or per-`pose` interpretation.
    grid : numpy.typing.ArrayLike or xarray.DataArray, optional
        Integer coordinates for the `grid` dimension.
    pose : numpy.typing.ArrayLike or xarray.DataArray, optional
        Legacy probe-pose coordinate labels; `pose` is an ordinary nonspatial dimension.
    k, j, i : numpy.typing.ArrayLike or xarray.DataArray, optional
        Integer native voxel coordinates. If not provided, dense zero-based
        coordinates are generated.
    dt : float, optional
        Time spacing in seconds, used when `time` is not provided.
    t0 : float or numpy.typing.ArrayLike, default: 0.0
        First time coordinate when `dt` is used. A 1D `t0` (one value per grid)
        generates a grid-dependent `(time, grid)` coordinate.
    spacing : sequence[float], optional
        World spacing in `z/y/x` order. Mutually exclusive with `voxel_to_world`.
    origin : sequence[float], optional
        World origin in `z/y/x` order. If not provided, the origin is zero.
    direction : numpy.typing.ArrayLike, optional
        3x3 direction matrix in world `z/y/x` row and voxel `k/j/i` column order.
    voxel_to_world : numpy.typing.ArrayLike, optional
        4x4 homogeneous affine in world `z/y/x` row and voxel `k/j/i` column order,
        or a stack indexed by `geometry_dims`. Mutually
        exclusive with `spacing`, `origin`, and `direction`.
    geometry_dims : tuple[str, ...], optional
        Ordered nonspatial affine dependencies. The affine shape must be
        `(*[sizes[d] for d in geometry_dims], 4, 4)`. A shared affine uses `()`.
        If omitted, 3D stacks infer `grid`, or `pose` when no `grid` is present;
        all other stacked cases require explicit names.
    affines : mapping[str, numpy.typing.ArrayLike], optional
        Named affine frames stored on the voxel-to-world index, each mapping the
        current world frame (`z/y/x`) to a named frame. Values are `(4, 4)` or, for
        grid-dependent geometry, `(ngrid, 4, 4)`.
    units : str, default: "mm"
        Physical unit shared by every derived world coordinate.
    name : str, optional
        DataArray name.
    attrs : dict, optional
        DataArray attributes.

    Returns
    -------
    xarray.DataArray
        VoxelData array with native voxel dimensions and derived world coordinates.

    Raises
    ------
    ValueError
        If dimensions, coordinate dependencies or values, geometry, or model
        validation are invalid. World `z`/`y`/`x` names are not native dimensions;
        affine stacks must match their named geometry dependencies.
    """
    dims = tuple(str(dim) for dim in dims)
    shape = np.shape(data)
    extra_coords = {} if extra_coords is None else dict(extra_coords)

    if len(set(dims)) != len(dims):
        raise ValueError(f"dims must not contain duplicate names, got {dims!r}.")
    if len(dims) != len(shape):
        raise ValueError(
            f"Length of dims {dims!r} ({len(dims)}) must match the number of array "
            f"dimensions ({len(shape)})."
        )
    invalid_spatial = sorted(set(dims) & set(WORLD_DIMS))
    if invalid_spatial:
        raise ValueError(
            f"dims must use native voxel names {VOXEL_DIMS!r}, not world coordinate "
            f"names {invalid_spatial!r}. World z/y/x coordinates are always derived "
            "from the voxel-to-world index, never passed as dims."
        )
    if TIME_DIM not in dims and time is not None:
        raise ValueError("time requires a 'time' dimension.")

    timing_dim = GRID_DIM if GRID_DIM in dims else "pose"
    named_time = (
        time
        if isinstance(time, xr.DataArray)
        and time.ndim > 1
        and time.dims != ("dim_0", "dim_1")
        else None
    )
    if named_time is not None:
        if TIME_DIM not in named_time.dims or any(
            d not in dims or d in VOXEL_DIMS for d in named_time.dims
        ):
            raise ValueError(
                "Multidimensional time must use time and named nonspatial data dimensions."
            )
        named_time = named_time.transpose(TIME_DIM, ...)
        for dim in named_time.dims:
            if named_time.sizes[dim] != shape[dims.index(str(dim))]:
                raise ValueError(
                    f"time dimension {dim!r} has size {named_time.sizes[dim]}, "
                    f"but data has size {shape[dims.index(str(dim))]}."
                )

    dependent_time: xr.DataArray | None = named_time
    if time is not None and named_time is None:
        time_array = time.values if isinstance(time, xr.DataArray) else np.asarray(time)
        if time_array.ndim == 2:
            if timing_dim not in dims:
                raise ValueError(
                    "A 2D time array (one column per grid) requires a 'grid' "
                    "dimension in dims."
                )
            grid_size = shape[dims.index(timing_dim)]
            if time_array.shape[1] != grid_size:
                raise ValueError(
                    f"time has {time_array.shape[1]} grid columns, but the 'grid' "
                    f"dimension size is {grid_size}."
                )
            dependent_time = xr.DataArray(
                np.asarray(time_array, dtype=np.float64),
                dims=(TIME_DIM, timing_dim),
                attrs=dict(time.attrs) if isinstance(time, xr.DataArray) else {},
            )
    elif time is None and TIME_DIM in dims:
        t0_array = np.asarray(t0)
        if t0_array.ndim > 0:
            if timing_dim not in dims:
                raise ValueError(
                    "A 1D t0 array (one value per grid) requires a 'grid' "
                    "dimension in dims."
                )
            if t0_array.ndim != 1:
                raise ValueError(
                    f"t0 must be scalar or 1D, got shape {t0_array.shape}."
                )
            grid_size = shape[dims.index(timing_dim)]
            if t0_array.size != grid_size:
                raise ValueError(
                    f"t0 has length {t0_array.size}, but the 'grid' dimension size "
                    f"is {grid_size}."
                )
            time_size = shape[dims.index(TIME_DIM)]
            dt_value = _require_spacing(TIME_DIM, dt)
            t0_values = np.asarray(t0_array, dtype=np.float64)
            if not np.all(np.isfinite(t0_values)):
                raise ValueError("t0 must contain finite values.")
            dependent_time = xr.DataArray(
                t0_values[None, :]
                + np.arange(time_size, dtype=np.float64)[:, None] * dt_value,
                dims=(TIME_DIM, timing_dim),
            )

    if dependent_time is not None:
        dependent_time = dependent_time.copy(deep=False)
        dependent_time.attrs.setdefault("units", _TIME_UNITS)

    forbidden_extra = set(CORE_DIMS) | set(WORLD_DIMS)
    overlap = sorted(forbidden_extra & set(extra_coords))
    if overlap:
        raise ValueError(
            f"extra_coords must not include core coordinates: {overlap!r}."
        )

    data_array: Any = data if hasattr(data, "shape") else np.asarray(data)
    data_dims = dims
    for dim in VOXEL_DIMS:
        if dim not in data_dims:
            data_array = np.expand_dims(data_array, axis=-1)
            data_dims = (*data_dims, dim)

    spatial_sizes = {voxel_dim: 1 for voxel_dim in VOXEL_DIMS}
    for dim, size in zip(data_dims, data_array.shape, strict=True):
        if dim in VOXEL_DIMS:
            spatial_sizes[dim] = int(size)

    coord_inputs: dict[str, npt.ArrayLike | xr.DataArray] = dict(extra_coords)
    if time is not None:
        coord_inputs[TIME_DIM] = time
    if grid is not None:
        coord_inputs[GRID_DIM] = grid
    if pose is not None:
        coord_inputs["pose"] = pose
    voxel_coord_inputs = {"k": k, "j": j, "i": i}

    scalar_t0: float | None
    if np.asarray(t0).ndim > 0:
        scalar_t0 = None
    else:
        scalar_t0 = float(np.asarray(t0).item())

    data_coords: dict[str, xr.DataArray] = {}
    spacings = {TIME_DIM: dt}
    origins = {TIME_DIM: scalar_t0}
    for dim, size in zip(data_dims, data_array.shape, strict=True):
        if dim in VOXEL_DIMS:
            continue
        if dim == TIME_DIM and dependent_time is not None:
            data_coords[dim] = xr.DataArray(
                dependent_time.data,
                dims=dependent_time.dims,
                attrs=dict(dependent_time.attrs),
            )
        else:
            data_coords[dim] = _coordinate_dataarray(
                dim,
                int(size),
                coords=coord_inputs,
                spacings=spacings,
                origins=origins,
            )

    voxel_to_world_array = (
        None if voxel_to_world is None else np.asarray(voxel_to_world, dtype=np.float64)
    )
    if voxel_to_world_array is not None and voxel_to_world_array.ndim > 2:
        if spacing is not None or origin is not None or direction is not None:
            raise ValueError(
                "voxel_to_world is mutually exclusive with spacing, origin, and direction."
            )
        resolved_voxel_to_world = voxel_to_world_array
    else:
        resolved_voxel_to_world = _resolve_voxel_to_world(
            spacing=spacing,
            origin=origin,
            direction=direction,
            voxel_to_world=voxel_to_world,
        )

    voxel_coords = {}
    for dim in VOXEL_DIMS:
        if dim not in data_dims:
            continue
        explicit = voxel_coord_inputs[dim]
        if explicit is None:
            voxel_coords[dim] = xr.DataArray(np.arange(spatial_sizes[dim]), dims=(dim,))
            continue
        extracted = _coordinate_values_and_attrs(dim, {dim: explicit})
        values, coord_attrs = extracted or (np.array([], dtype=int), {})
        _validate_coordinate_shape(dim, values, spatial_sizes[dim])
        voxel_coords[dim] = xr.DataArray(values, dims=(dim,), attrs=coord_attrs)

    result = xr.DataArray(
        data_array,
        dims=data_dims,
        coords={
            **data_coords,
            **{dim: voxel_coords[dim] for dim in data_dims if dim in VOXEL_DIMS},
        },
        name=name,
        attrs={} if attrs is None else dict(attrs),
    )

    result = attach_voxel_to_world_index(
        result,
        resolved_voxel_to_world,
        units=units,
        named_affines=affines,
        geometry_dims=geometry_dims,
    )

    nonspatial_dims = [dim for dim in result.dims if dim not in VOXEL_DIMS]
    result = result.transpose(*nonspatial_dims, *VOXEL_DIMS)
    if dependent_time is not None:
        # Keep timestamp axes independent of the parent dimension order.
        result = result.assign_coords({TIME_DIM: data_coords[TIME_DIM]})

    explicit_voxel_dims = {
        dim for dim, coord in voxel_coord_inputs.items() if coord is not None
    }
    regular_spacing_dims = tuple(
        dim
        for dim in CORE_DIMS
        if dim in result.dims
        and result.sizes[dim] > 1
        and dim not in explicit_voxel_dims
        and not (dim == TIME_DIM and (time is not None or dependent_time is not None))
    )
    validate_voxeldata(
        result,
        require_regular_spacing=True,
        regular_spacing_dims=regular_spacing_dims,
    )
    return result
