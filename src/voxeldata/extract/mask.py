"""Extraction using binary masks, adapted from ConfUSIus (see NOTICE)."""

import xarray as xr

from voxeldata._dims import VOXEL_DIMS
from voxeldata.masks import ensure_mask
from voxeldata.validation import ensure_voxeldata


def extract_with_mask(data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Extract voxel samples with a spatial binary mask.

    Native voxel dimensions are flattened into `space`, retaining all nonspatial
    dimensions and their coordinates. Selected native voxel labels form a
    MultiIndex, suitable for `.unstack("space")` or `unmask` reconstruction.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, optionally with nonspatial dimensions.
    mask : xarray.DataArray
        Spatial-only VoxelData mask on the same physical grid. Must be boolean or
        binary numeric: zero is background and one distinct nonzero value selects
        voxels. A shared mask can span geometry dependencies only if their affine
        matrices agree. Select a geometry first when its physical grid differs.

    Returns
    -------
    xarray.DataArray
        Samples with the native voxel dimensions replaced by `space`. Signal data
        remain lazy when Dask-backed; mask values are evaluated for selection.

    Raises
    ------
    ValueError
        If geometry, native labels, or units do not match, or the mask has
        nonspatial dimensions. Scalar-selected voxel axes are restored first.
    TypeError
        If the mask is not a boolean or binary numeric VoxelData array.
    """
    data = ensure_voxeldata(data, allow_extra_dims=True)
    mask = ensure_mask(mask, data, "mask")
    if tuple(mask.dims) != VOXEL_DIMS:
        raise ValueError(
            "Extraction requires a spatial-only mask; select nonspatial mask dimensions first."
        )

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
