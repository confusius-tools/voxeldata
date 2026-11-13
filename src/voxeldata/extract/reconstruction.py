"""Masked voxel reconstruction."""

import numpy as np
import xarray as xr

from voxeldata.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_units,
)
from voxeldata.masks import check_mask_dtype
from voxeldata.validation import canonicalize_voxeldata


def unmask(
    signals: np.ndarray | xr.DataArray,
    mask: xr.DataArray,
    new_dims: list[str] | None = None,
    new_dims_coords: dict[str, np.ndarray] | None = None,
    attrs: dict | None = None,
    fill_value: float = 0.0,
) -> xr.DataArray:
    """Reconstruct a voxel grid from samples and their original spatial mask.

    Parameters
    ----------
    signals : numpy.ndarray or xarray.DataArray
        Samples shaped `(..., space)`, one value per selected mask voxel. A
        DataArray must have `space` last; its leading dimension coordinates,
        including timestamp attributes, are preserved.
    mask : xarray.DataArray
        Boolean or binary numeric VoxelData mask. Supplies native voxel labels,
        any extra feature axes, units, geometry dependencies, and named frames.
    new_dims : list[str], optional
        Leading dimension names for NumPy input. If not provided, use `dim_0`,
        `dim_1`, and so on. Ignored for DataArray input.
    new_dims_coords : dict[str, numpy.ndarray], optional
        Leading dimension coordinates for NumPy input. Missing coordinates use
        zero-based positions. Ignored for DataArray input.
    attrs : dict, optional
        Attributes to attach to the reconstructed array.
    fill_value : float, default: 0.0
        Value assigned outside the mask, cast to the signal dtype.

    Returns
    -------
    xarray.DataArray
        Samples restored into the mask's complete grid, including extra feature
        axes. Geometry dependencies and named frames are supplied by the mask.

    Raises
    ------
    TypeError
        If signals have an unsupported type or the mask is not boolean/binary.
    ValueError
        If mask geometry or dimensions are invalid, signal count does not match,
        or leading dimension names/coordinates are inconsistent.

    Notes
    -----
    Reconstruction is eager, including when a DataArray holds Dask-backed signals.
    Mask and signal sample ordering must correspond; a `space` MultiIndex is not
    used to realign samples automatically.
    """
    mask = canonicalize_voxeldata(mask, allow_extra_dims=True)
    check_mask_dtype(mask, "mask")
    mask_values = mask.values

    n_voxels_mask = np.int64(np.count_nonzero(mask_values)).item()

    if isinstance(signals, np.ndarray):
        if signals.ndim == 0:
            raise ValueError("signals must have a sample dimension, not a scalar.")
        if signals.shape[-1] != n_voxels_mask:
            raise ValueError(
                f"Last dimension of signals ({signals.shape[-1]}) doesn't match "
                f"number of masked voxels ({n_voxels_mask})"
            )

        n_leading_dims = signals.ndim - 1

        if new_dims is not None:
            if len(new_dims) != n_leading_dims:
                raise ValueError(
                    f"Length of new_dims ({len(new_dims)}) doesn't match "
                    f"number of leading dimensions ({n_leading_dims})"
                )
            dim_names = new_dims
        else:
            dim_names = [f"dim_{i}" for i in range(n_leading_dims)]

        coords = {"space": np.arange(n_voxels_mask)}
        if new_dims_coords is not None:
            for dim in dim_names:
                if dim in new_dims_coords:
                    coord_array = new_dims_coords[dim]
                    dim_idx = dim_names.index(dim)
                    if len(coord_array) != signals.shape[dim_idx]:
                        raise ValueError(
                            f"Coordinate array for '{dim}' has length {len(coord_array)} "
                            f"but dimension has size {signals.shape[dim_idx]}"
                        )
                    coords[dim] = coord_array
                else:
                    dim_idx = dim_names.index(dim)
                    coords[dim] = np.arange(signals.shape[dim_idx])
        else:
            for i, dim in enumerate(dim_names):
                coords[dim] = np.arange(signals.shape[i])

        all_dims = dim_names + ["space"]
        signals = xr.DataArray(signals, dims=all_dims, coords=coords)
    elif isinstance(signals, xr.DataArray):
        if "space" not in signals.dims:
            raise ValueError(
                f"DataArray signals must have 'space' dimension, got {signals.dims}"
            )
        if signals.dims[-1] != "space":
            raise ValueError(
                f"'space' must be the last dimension, got dims={signals.dims}"
            )
        if signals.sizes["space"] != n_voxels_mask:
            raise ValueError(
                f"Size of 'space' dimension ({signals.sizes['space']}) doesn't match "
                f"number of masked voxels ({n_voxels_mask})"
            )
    else:
        raise TypeError(
            f"'signals' must be Numpy array or DataArray, got {type(signals)}"
        )

    spatial_dims = list(mask.dims)
    spatial_shape = tuple(mask.sizes[d] for d in spatial_dims)
    extra_dims = [d for d in signals.dims if d != "space"]

    mask_flat = (mask_values != 0).flatten()

    if extra_dims:
        output_shape = tuple(signals.sizes[d] for d in extra_dims) + spatial_shape
        output_dims = extra_dims + spatial_dims

        output_data = np.full(output_shape, fill_value, dtype=signals.dtype)

        n_extra = np.int64(np.prod([signals.sizes[d] for d in extra_dims])).item()
        output_flat = output_data.reshape(n_extra, -1)
        signals_flat = signals.values.reshape(n_extra, -1)
        output_flat[:, mask_flat] = signals_flat

        coords = {d: mask.coords[d] for d in spatial_dims}
        for d in extra_dims:
            coords[d] = signals.coords[d]
    else:
        output_shape = spatial_shape
        output_dims = spatial_dims

        output_data = np.full(output_shape, fill_value, dtype=signals.dtype)

        output_data.flat[mask_flat] = signals.values

        coords = {d: mask.coords[d] for d in spatial_dims}

    result = xr.DataArray(
        output_data,
        dims=output_dims,
        coords=coords,
        attrs=attrs if attrs is not None else {},
    )
    return attach_voxel_to_world_index(
        result,
        get_voxel_to_world_affine(mask),
        units=get_voxel_to_world_units(mask),
        geometry_dims=mask.vd.geometry_dims,
        named_affines=mask.vd.affines,
    )
