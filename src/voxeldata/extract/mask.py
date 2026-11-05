"""Extraction using binary masks, adapted from ConfUSIus (see NOTICE)."""

import xarray as xr

from voxeldata.masks import ensure_mask
from voxeldata.validation import ensure_voxeldata


def extract_with_mask(data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
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
        Extra mask axes, such as grid or channel, are flattened along with native
        voxel axes. A shared mask can span geometry dependencies only if their
        affine matrices agree; otherwise include or select those dependencies.

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
    data = ensure_voxeldata(data, allow_extra_dims=True)
    mask = ensure_mask(mask, data, "mask")

    # ensure_mask() already checked mask and data share the same voxel grid (same
    # k/j/i coordinates and voxel_to_world affine), so mask.values.ravel() is already
    # positionally aligned with data.stack(space=spatial_dims) below -- both iterate
    # spatial_dims (mask's own dim order) the same way.
    spatial_dims = list(mask.dims)
    data_flat = data.stack(space=spatial_dims)
    mask_flat = mask.values.ravel()
    # Rebuild the space index from the selected voxel coordinates so unstack() uses the
    # reduced grid implied by the extracted mask.
    return data_flat.isel(space=mask_flat).set_xindex(spatial_dims)
