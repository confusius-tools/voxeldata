"""Validation helpers for the VoxelData model."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal, SupportsFloat, SupportsIndex

import numpy as np
import xarray as xr

from voxeldata._coordinates import get_coordinate_spacing_info
from voxeldata._dims import CORE_DIMS, GRID_DIM, TIME_DIM, VOXEL_DIMS
from voxeldata.geometry import (
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spatial_dims,
    has_voxel_to_world_index,
    restore_voxel_to_world_index,
)

RegularSpacingDims = Literal["space", "core", "all"] | str | Sequence[str]
"""Selector for dimensions that must satisfy regular-spacing checks."""


def require_dataarray(data: Any) -> xr.DataArray:
    """Validate that `data` is an xarray DataArray.

    Parameters
    ----------
    data : Any
        Object to validate.

    Returns
    -------
    xarray.DataArray
        The validated DataArray.

    Raises
    ------
    TypeError
        If `data` is not an xarray DataArray.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Expected an xarray.DataArray, got {type(data).__name__}.")
    return data


def require_positive_finite(
    value: str | SupportsFloat | SupportsIndex, name: str
) -> float:
    """Return a finite positive numeric value.

    Parameters
    ----------
    value : str or typing.SupportsFloat or typing.SupportsIndex
        Candidate value.
    name : str
        Name used in the validation error.

    Returns
    -------
    float
        Validated value.

    Raises
    ------
    ValueError
        If `value` is not numeric, finite, and positive.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _get_spatial_dims(da: xr.DataArray) -> tuple[str, ...]:
    """Return present voxel-space spatial dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    tuple[str, ...]
        Present voxel-space spatial dimensions in canonical order.
    """
    return tuple(dim for dim in VOXEL_DIMS if dim in da.dims)


def _validate_voxel_to_world_geometry(da: xr.DataArray) -> None:
    """Validate that a DataArray has voxel-to-world geometry consistent with its dims.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to validate.

    Raises
    ------
    ValueError
        If voxel-to-world geometry is missing or inconsistent.
    """
    if not has_voxel_to_world_index(da):
        raise ValueError(
            "DataArray must use native voxel dimensions `k/j/i` with "
            "VoxelToWorldIndex-backed world coordinates."
        )

    voxel_dims = get_voxel_to_world_spatial_dims(da)
    if set(voxel_dims) != set(VOXEL_DIMS):
        raise ValueError(
            f"Voxel-to-world index must cover native voxel dimensions {VOXEL_DIMS!r}, "
            f"got {voxel_dims!r}."
        )


def _validate_dimension_coordinate(
    da: xr.DataArray, dim: Hashable, *, require_numeric: bool, require_ascending: bool
) -> None:
    """Validate a single dimension coordinate.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinate should be validated.
    dim : hashable
        Dimension name whose matching coordinate is required.
    require_numeric : bool
        Whether the coordinate must be numeric, finite, and strictly monotonic.
    require_ascending : bool
        Whether strict monotonicity must be increasing specifically. Ignored if
        `require_numeric` is `False`. Voxel dims (`k`/`j`/`i`) may run in either
        direction; `time`/`grid` must be increasing.

    Raises
    ------
    ValueError
        If the coordinate is missing, malformed, non-numeric, non-finite, or not
        strictly monotonic (in the required direction, if any).
    """
    if dim not in da.coords:
        if dim in CORE_DIMS:
            raise ValueError(f"Missing required coordinate for dimension {dim!r}.")
        return

    coord = da.coords[dim]
    if (
        dim == TIME_DIM
        and GRID_DIM in da.dims
        and set(coord.dims) == {TIME_DIM, GRID_DIM}
    ):
        # A grid-dependent array may carry each grid's own timestamps directly.
        if not np.issubdtype(coord.dtype, np.number):
            raise ValueError(f"Coordinate {dim!r} must be numeric.")
        values = np.asarray(coord.transpose(TIME_DIM, GRID_DIM).values)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Coordinate {dim!r} contains non-finite numeric values.")
        if values.shape[0] <= 1:
            return
        if not np.all(np.diff(values, axis=0) > 0):
            raise ValueError(
                f"Coordinate {dim!r} must be strictly monotonic-increasing for "
                "every grid."
            )
        return

    if coord.dims != (dim,):
        raise ValueError(
            f"Coordinate {dim!r} must be a 1D dimension coordinate with dims "
            f"({dim!r},), got {coord.dims!r}."
        )

    if not require_numeric:
        return
    if not np.issubdtype(coord.dtype, np.number):
        raise ValueError(f"Coordinate {dim!r} must be numeric.")

    values = np.asarray(coord.values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Coordinate {dim!r} contains non-finite numeric values.")
    if values.size <= 1:
        return
    diffs = np.diff(values)
    if require_ascending:
        if not np.all(diffs > 0):
            raise ValueError(
                f"Coordinate {dim!r} must be strictly monotonic-increasing."
            )
    elif not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(f"Coordinate {dim!r} must be strictly monotonic.")


def _validate_core_dimension_names(da: xr.DataArray, allow_extra_dims: bool) -> None:
    """Validate core dimension names.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimensions should be checked.
    allow_extra_dims : bool
        Whether dimensions outside the core set are allowed.

    Raises
    ------
    ValueError
        If dimensions are not strings or unexpected dimensions are present.
    """
    invalid_dims = [dim for dim in da.dims if not isinstance(dim, str)]
    if invalid_dims:
        raise ValueError(
            f"All dimensions must be strings, got invalid dimensions: {invalid_dims!r}."
        )

    if not allow_extra_dims:
        unexpected_dims = [dim for dim in da.dims if dim not in CORE_DIMS]
        if unexpected_dims:
            raise ValueError(
                f"Unexpected dimensions {unexpected_dims!r}. VoxelData arrays may "
                f"only use dimensions {CORE_DIMS!r}."
            )


def _validate_no_zero_length_dims(da: xr.DataArray) -> None:
    """Validate that no dimension has zero length.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimension sizes should be checked.

    Raises
    ------
    ValueError
        If any dimension has size 0.
    """
    empty_dims = [str(dim) for dim in da.dims if da.sizes[dim] == 0]
    if empty_dims:
        raise ValueError(
            f"DataArray must not have zero-length dimensions, got empty dimensions "
            f"{empty_dims!r} (shape {da.shape!r})."
        )


def _validate_spatial_dim_order(da: xr.DataArray) -> None:
    """Validate that native voxel dimensions form the trailing spatial block.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose dimension order should be checked.

    Raises
    ------
    ValueError
        If native voxel dimensions are not trailing in `k`, `j`, `i` order.
    """
    if tuple(da.dims[-len(VOXEL_DIMS) :]) != VOXEL_DIMS:
        raise ValueError(
            f"VoxelData dimensions must end with {VOXEL_DIMS!r}, got {da.dims!r}."
        )


def _validate_regular_spacing(
    da: xr.DataArray,
    regular_spacing_tolerance: float,
    regular_spacing_dims: RegularSpacingDims,
    spatial_dims: tuple[str, ...],
) -> None:
    """Validate that selected numeric coordinates have regular spacing.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinates should be checked.
    regular_spacing_tolerance : float
        Relative tolerance used to assess regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str]
        Dimensions to validate.
    spatial_dims : tuple[str, ...]
        Present voxel-space spatial dimensions.

    Raises
    ------
    ValueError
        If selected coordinate spacing is missing, invalid, or non-uniform.
    """
    if regular_spacing_dims == "space":
        dims_to_check = list(spatial_dims)
    elif regular_spacing_dims == "core":
        dims_to_check = [dim for dim in CORE_DIMS if dim in da.dims]
    elif regular_spacing_dims == "all":
        dims_to_check = [str(dim) for dim in da.dims]
    elif isinstance(regular_spacing_dims, str):
        dims_to_check = [regular_spacing_dims]
    else:
        dims_to_check = [str(dim) for dim in regular_spacing_dims]

    missing_dims = [dim for dim in dims_to_check if dim not in da.dims]
    if missing_dims:
        raise ValueError(
            "regular_spacing_dims contains dimensions not present in data: "
            f"{missing_dims!r}. Present dims: {tuple(str(dim) for dim in da.dims)!r}."
        )

    voxel_spacing = get_voxel_to_world_index_spacing(da)
    for dim in dims_to_check:
        if dim not in da.coords:
            raise ValueError(
                f"Missing required coordinate for dimension {dim!r} when checking "
                "for regular spacing."
            )
        coord = da.coords[dim]
        if not np.issubdtype(coord.dtype, np.number):
            continue
        spacing_value = (
            voxel_spacing[dim]
            if dim in spatial_dims
            else get_coordinate_spacing_info(dim, da, regular_spacing_tolerance).value
        )
        if spacing_value is None:
            raise ValueError(
                f"Coordinate {dim!r} must have regular spacing, but spacing is "
                "non-uniform or undefined."
            )


def _validate_uniform_time(da: xr.DataArray, tolerance: float) -> None:
    """Validate uniformly spaced `time` values.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose `time` coordinate should be checked.
    tolerance : float
        Maximum relative variation allowed between consecutive time intervals.

    Raises
    ------
    ValueError
        If `time` is missing, has fewer than two values, or is not uniform.
    """
    if TIME_DIM not in da.coords:
        raise ValueError("DataArray must have a 'time' coordinate.")
    values = np.asarray(da.coords[TIME_DIM].values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("A grid-dependent 'time' coordinate must be selected first.")
    if values.size <= 1:
        raise ValueError("DataArray must have more than one time point.")
    diffs = np.diff(values)
    if not np.allclose(diffs, diffs[0], rtol=tolerance, atol=0.0):
        raise ValueError("'time' coordinate must be uniformly spaced.")


def canonicalize_voxeldata(data: xr.DataArray) -> xr.DataArray:
    """Restore scalar-indexed VoxelData dimensions and order them canonically.

    Scalar indexing such as `data.isel(j=0)` removes `j` from the array dimensions
    but retains it as a scalar coordinate. This function restores every missing
    native voxel dimension (`k`, `j`, `i`) as a length-one dimension, rebuilds
    voxel-to-world geometry from the untouched affine, then orders dimensions as
    `(...extra_dims, grid, time, k, j, i)`. Extra dimensions retain their relative
    order. Missing `grid` and `time` dimensions are not restored.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to canonicalize.

    Returns
    -------
    xarray.DataArray
        Canonicalized DataArray with all native voxel dimensions present.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If a native voxel dimension is absent and has no scalar coordinate from which
        to restore it, or its same-named coordinate is not scalar.
    """
    data = require_dataarray(data)

    result = data
    for dim in VOXEL_DIMS:
        if dim in result.dims:
            continue
        if dim not in result.coords:
            raise ValueError(
                f"DataArray is missing voxel dimension {dim!r}, and has no scalar "
                f"coordinate {dim!r} to restore it from."
            )
        if result.coords[dim].shape != ():
            raise ValueError(
                f"DataArray is missing voxel dimension {dim!r}, but coordinate "
                f"{dim!r} is not scalar."
            )
        coord = result.coords[dim]
        attrs = coord.attrs.copy()
        spatial_index = VOXEL_DIMS.index(dim)
        next_dims = [d for d in VOXEL_DIMS[spatial_index + 1 :] if d in result.dims]
        previous_dims = [d for d in VOXEL_DIMS[:spatial_index] if d in result.dims]
        if next_dims:
            axis = result.dims.index(next_dims[0])
        elif previous_dims:
            axis = result.dims.index(previous_dims[-1]) + 1
        else:
            axis = len(result.dims)
        result = result.expand_dims({dim: [coord.item()]}, axis=axis)
        result.coords[dim].attrs.update(attrs)

    result = restore_voxel_to_world_index(result)
    return result.transpose(
        *(dim for dim in result.dims if dim not in CORE_DIMS),
        *(dim for dim in CORE_DIMS if dim in result.dims),
    )


def ensure_voxeldata(data: xr.DataArray, **validate_kwargs: Any) -> xr.DataArray:
    """Return a canonical, validated VoxelData array.

    This restores scalar-indexed native voxel dimensions, orders dimensions as
    `(...extra_dims, grid, time, k, j, i)`, preserving the relative order of extra
    dimensions, then validates the result with
    [validate_voxeldata][voxeldata.validation.validate_voxeldata].

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to canonicalize and validate.
    **validate_kwargs : Any
        Keyword arguments forwarded to
        [validate_voxeldata][voxeldata.validation.validate_voxeldata].

    Returns
    -------
    xarray.DataArray
        Canonicalized VoxelData array that satisfies the requested checks.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If canonicalization or validation fails.
    """
    result = canonicalize_voxeldata(data)
    validate_voxeldata(result, **validate_kwargs)
    return result


def validate_voxeldata(
    data: xr.DataArray,
    *,
    require_time: bool = False,
    require_uniform_time: bool = False,
    uniformity_tolerance: float = 1e-2,
    allow_grid: bool = True,
    allow_extra_dims: bool = True,
    require_regular_spacing: bool = False,
    regular_spacing_tolerance: float = 1e-2,
    regular_spacing_dims: RegularSpacingDims = "space",
    require_dtype: Any | None = None,
) -> None:
    """Validate a DataArray against the VoxelData model without modifying it.

    This requires non-empty native voxel dimensions (`k`, `j`, `i`), their
    coordinates, a matching `VoxelToWorldIndex` carrying world `z`/`y`/`x`
    coordinates, and dimensions ordered as `(*nonspatial_dims, k, j, i)`.
    Nonspatial dimensions, including optional `grid` and `time`, may appear in any
    order. Every dimension coordinate is validated. Optional flags add requirements needed
    by a particular consumer. Use
    [ensure_voxeldata][voxeldata.validation.ensure_voxeldata] when scalar indexing
    may have removed a voxel dimension and the input should be canonicalized first.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to validate.
    require_time : bool, default: False
        Whether to require `time` with more than one coordinate value.
    require_uniform_time : bool, default: False
        Whether to require `time` with more than one uniformly spaced coordinate
        value.
    uniformity_tolerance : float, default: 1e-2
        Maximum relative variation allowed between consecutive time intervals.
    allow_grid : bool, default: True
        Whether to allow a `grid` dimension.
    allow_extra_dims : bool, default: True
        Whether to allow dimensions outside `time`, `grid`, `k`, `j`, and `i`.
    require_regular_spacing : bool, default: False
        Whether to require regular spacing for selected numeric dimension coordinates.
    regular_spacing_tolerance : float, default: 1e-2
        Relative tolerance used to assess coordinate regularity.
    regular_spacing_dims : {"space", "core", "all"} or str or sequence[str], default: "space"
        Dimensions to check for regular spacing. `"space"` checks `k`, `j`, and `i`;
        `"core"` checks present core dimensions; `"all"` checks every dimension.
    require_dtype : Any, optional
        Required data dtype or dtype class, passed to `numpy.issubdtype`.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray` or its dtype does not satisfy
        `require_dtype`.
    ValueError
        If VoxelData geometry, dimensions, coordinates, spacing, or metadata
        validation fails.
    """
    data = require_dataarray(data)
    _validate_no_zero_length_dims(data)
    _validate_core_dimension_names(data, allow_extra_dims=allow_extra_dims)

    spatial_dims = _get_spatial_dims(data)
    if spatial_dims != VOXEL_DIMS:
        raise ValueError(
            f"DataArray must include all native voxel dimensions {VOXEL_DIMS!r}, "
            f"got {spatial_dims!r}."
        )

    _validate_voxel_to_world_geometry(data)

    if require_time or require_uniform_time:
        if TIME_DIM not in data.dims:
            raise ValueError("DataArray must have a 'time' dimension.")
        if data.sizes[TIME_DIM] <= 1:
            raise ValueError("DataArray must have more than one time point.")
    if require_uniform_time:
        _validate_uniform_time(data, uniformity_tolerance)

    if not allow_grid and GRID_DIM in data.dims:
        raise ValueError("DataArray must not have a 'grid' dimension.")

    for dim in data.dims:
        _validate_dimension_coordinate(
            data,
            dim,
            require_numeric=dim in CORE_DIMS,
            require_ascending=dim not in VOXEL_DIMS,
        )

    if require_regular_spacing:
        _validate_regular_spacing(
            data,
            regular_spacing_tolerance,
            regular_spacing_dims,
            spatial_dims,
        )

    _validate_spatial_dim_order(data)

    if require_dtype is not None and not np.issubdtype(data.dtype, require_dtype):
        raise TypeError(
            f"Expected data dtype compatible with {require_dtype}, got {data.dtype}."
        )
