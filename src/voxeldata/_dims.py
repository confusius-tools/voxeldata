"""VoxelData dimension name conventions.

VoxelData arrays have trailing spatial dimensions and arbitrary leading nonspatial
dimensions. This module names the spatial and special nonspatial dimensions so that
IO, processing, validation, and UI code can share their conventions.
"""

WORLD_DIMS: tuple[str, ...] = ("z", "y", "x")
"""World spatial dimension names in VoxelData order."""

VOXEL_DIMS: tuple[str, ...] = ("k", "j", "i")
"""Native voxel-space dimension names in VoxelData order."""

GRID_DIM: str = "grid"
"""Dimension selecting one voxel-to-world mapping per spatial sampling grid."""

TIME_DIM: str = "time"
"""Temporal sample dimension name."""

CORE_DIMS: tuple[str, ...] = (GRID_DIM, TIME_DIM, *VOXEL_DIMS)
"""Core dimension names in canonical order, not a required nonspatial order."""
