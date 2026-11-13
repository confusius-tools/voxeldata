"""VoxelData: spatially referenced voxel arrays built on xarray.

VoxelData defines a small, modality-independent data model for arrays gridded in
space. A VoxelData array is an `xarray.DataArray` with native voxel dimensions
`(k, j, i)` (plus any extra leading dimensions) and a single
[VoxelToWorldIndex][voxeldata.geometry.VoxelToWorldIndex] that lazily derives world
coordinates `(z, y, x)` from a voxel-to-world affine.
"""

from voxeldata._dims import CORE_DIMS, TIME_DIM, VOXEL_DIMS, WORLD_DIMS
from voxeldata.accessors import VoxelDataAccessor
from voxeldata.affine import (
    TransformCenter,
    apply_affine,
    apply_flip,
    apply_rotation,
    apply_translation,
    apply_zoom,
    get_bounding_box,
    get_geometry_center,
    get_relative_affine,
    reindex_voxels,
    reindex_voxels_like,
)
from voxeldata.create import create_voxeldata
from voxeldata.extract import extract_with_labels, extract_with_mask, unmask
from voxeldata.geometry import (
    VoxelToWorldIndex,
    VoxelToWorldTransform,
    attach_voxel_to_world_index,
    convert_length_units,
    get_affine_axis_scalings,
    get_affine_direction_matrix,
    get_array_origin,
    get_array_spacing,
    get_voxel_to_world_affine,
    get_voxel_to_world_direction_matrix,
    get_voxel_to_world_index,
    get_voxel_to_world_spatial_dims,
    get_voxel_to_world_units,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
    require_single_geometry_affine,
    restore_voxel_to_world_index,
)
from voxeldata.io import load_array, load_dataset, load_tree, save
from voxeldata.masks import (
    check_mask_dtype,
    check_spatial_alignment,
    ensure_labels,
    ensure_mask,
    validate_labels,
    validate_mask,
)
from voxeldata.validation import (
    canonicalize_voxeldata,
    ensure_voxeldata,
    require_dataarray,
    require_positive_finite,
    validate_voxeldata,
)

__all__ = [
    "CORE_DIMS",
    "TIME_DIM",
    "VOXEL_DIMS",
    "WORLD_DIMS",
    "TransformCenter",
    "VoxelDataAccessor",
    "VoxelToWorldIndex",
    "VoxelToWorldTransform",
    "apply_affine",
    "apply_flip",
    "apply_rotation",
    "apply_translation",
    "apply_zoom",
    "attach_voxel_to_world_index",
    "canonicalize_voxeldata",
    "check_mask_dtype",
    "check_spatial_alignment",
    "convert_length_units",
    "create_voxeldata",
    "ensure_labels",
    "ensure_mask",
    "ensure_voxeldata",
    "extract_with_labels",
    "extract_with_mask",
    "get_affine_axis_scalings",
    "get_affine_direction_matrix",
    "get_array_origin",
    "get_array_spacing",
    "get_bounding_box",
    "get_geometry_center",
    "get_relative_affine",
    "get_voxel_to_world_affine",
    "get_voxel_to_world_direction_matrix",
    "get_voxel_to_world_index",
    "get_voxel_to_world_spatial_dims",
    "get_voxel_to_world_units",
    "has_axis_aligned_voxel_to_world_index",
    "has_voxel_to_world_index",
    "load_array",
    "load_dataset",
    "load_tree",
    "reindex_voxels",
    "reindex_voxels_like",
    "require_dataarray",
    "require_positive_finite",
    "require_single_geometry_affine",
    "restore_voxel_to_world_index",
    "save",
    "unmask",
    "validate_labels",
    "validate_mask",
    "validate_voxeldata",
]
