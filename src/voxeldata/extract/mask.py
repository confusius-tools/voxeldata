"""Extraction using binary masks."""

from typing import Literal

import numpy as np
import xarray as xr

from voxeldata.masks import check_mask_dtype, check_spatial_alignment
from voxeldata.validation import canonicalize_voxeldata


def _match_mask_labels(
    data: xr.DataArray, mask: xr.DataArray, join: Literal["exact", "inner"]
) -> tuple[xr.DataArray, xr.DataArray]:
    """Select corresponding samples on each mask dimension in mask-label order.

    Parameters
    ----------
    data : xarray.DataArray
        Canonical signal array.
    mask : xarray.DataArray
        Canonical binary mask.
    join : {"exact", "inner"}
        Label sets to require on shared indexed dimensions.

    Returns
    -------
    data : xarray.DataArray
        Signal data selected in mask-label order.
    mask : xarray.DataArray
        Mask restricted to the matched labels.

    Raises
    ------
    ValueError
        If a mask dimension is missing, or labels cannot be matched unambiguously.
    """
    data_positions = {}
    mask_positions = {}
    for dim in mask.dims:
        if dim not in data.dims:
            raise ValueError(f"Mask dimension {dim!r} is missing from data.")
        if dim not in data.coords or dim not in mask.coords:
            if dim in data.coords or dim in mask.coords:
                raise ValueError(
                    f"Both arrays need labels on {dim!r} to match by label."
                )
            if mask.sizes[dim] != data.sizes[dim]:
                raise ValueError(
                    f"Unlabeled dimension {dim!r} must have the same size."
                )
            continue
        if data.coords[dim].dims != (dim,) or mask.coords[dim].dims != (dim,):
            raise ValueError(
                f"Matching on {dim!r} requires one-dimensional labels; select coordinate dependencies first."
            )
        if dim not in data.xindexes:
            data = data.set_xindex(dim)
        if dim not in mask.xindexes:
            mask = mask.set_xindex(dim)

        source = data.get_index(dim)
        target = mask.get_index(dim)
        if source.equals(target):
            continue
        if not source.is_unique or not target.is_unique:
            raise ValueError(
                f"Matching different indexes on {dim!r} requires unique labels."
            )
        positions = source.get_indexer(target)
        matches = positions >= 0
        if join == "exact" and (len(target) != len(source) or not matches.all()):
            raise ValueError(
                f"Mask and data must have the same labels on {dim!r} for join='exact'."
            )
        if not matches.any():
            raise ValueError(f"Mask and data have no matching labels on {dim!r}.")
        if not matches.all():
            mask_positions[dim] = np.flatnonzero(matches)
        data_positions[dim] = positions[matches]

    # isel slices indexed affines and frames along with samples on geometry axes.
    if data_positions:
        data = data.isel(data_positions)
    if mask_positions:
        mask = mask.isel(mask_positions)
    return data, mask


def extract_with_mask(
    data: xr.DataArray, mask: xr.DataArray, *, join: Literal["exact", "inner"] = "exact"
) -> xr.DataArray:
    """Extract samples over the dimensions of a binary VoxelData mask.

    Mask dimensions are flattened into `space`, retaining other data dimensions
    and their coordinates. Selected mask-axis labels form a MultiIndex, suitable
    for `.unstack("space")` or `unmask` reconstruction.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, optionally with nonspatial dimensions.
    mask : xarray.DataArray
        VoxelData mask on the same physical grid. Must be boolean or binary
        numeric: zero is background and one distinct nonzero value selects samples.
        Extra mask axes, such as view or channel, are flattened along with native
        voxel axes. A shared mask can span geometry dependencies only if their
        affine matrices agree; otherwise include or select those dependencies.
    join : {"exact", "inner"}, default: "exact"
        How to match shared dimension labels in mask order. `"exact"` requires
        equal label sets. `"inner"` keeps their intersection on each indexed
        dimension. Reordering or subsetting by label requires unique labels.
        Dimensions unlabeled on both arrays use equal-size positional correspondence.

    Returns
    -------
    xarray.DataArray
        Samples with mask dimensions replaced by `space`. Signal data
        remain lazy when Dask-backed; mask values are evaluated for selection.

    Raises
    ------
    ValueError
        If geometry, dimension labels, or units do not match.
        Scalar-selected voxel axes are restored first.
    TypeError
        If the mask is not a boolean or binary numeric VoxelData array.
    """
    if join not in ("exact", "inner"):
        raise ValueError("join must be 'exact' or 'inner'.")
    data = canonicalize_voxeldata(data)
    mask = canonicalize_voxeldata(mask)
    check_mask_dtype(mask, "mask")
    data, mask = _match_mask_labels(data, mask, join)
    check_spatial_alignment(mask, data, "mask")

    spatial_dims = list(mask.dims)
    data_flat = data.stack(space=spatial_dims)
    mask_flat = mask.values.astype(bool, copy=False).ravel()
    # Rebuild the space index from the selected voxel coordinates so unstack() uses the
    # reduced grid implied by the extracted mask.
    return data_flat.isel(space=mask_flat).set_xindex(spatial_dims)
