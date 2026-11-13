# Roadmap

## Implemented foundations

Named geometry dependencies and generic extraction are available in VoxelData
`0.0.4` and implemented on `main`.
No outstanding VoxelData feature work is currently identified as necessary for
the tested ConfUSIus integration.

- Affine stacks use explicit, ordered `geometry_dims`, independently of timestamp
  dependencies. Native voxel and nonspatial labels retain ordinary Xarray indexes;
  the world index owns only derived spatial coordinates.
- Geometry and named frames support selection, transpose, concatenation,
  reconstruction, and lazy arrays. Tests cover independent numerical oracles,
  native labels, multiple geometry dependencies, and signal noncomputation.
- Generic mask/label validation, extraction, and reconstruction live in VoxelData.
  Physical sampling geometry controls mask compatibility. Base extraction works
  without flox or Dask; acceleration and lazy arrays are independent extras.
- Zarr and NetCDF persistence are implemented on `main` for VoxelData `0.0.5`,
  preserving geometry dependencies, units, native labels, timestamps, and named
  frames without storing dense world coordinates. Storage backends use the
  optional `io` extra; loading also works without Dask.
- Zarr container persistence is implemented for VoxelData `0.0.6`: datasets share
  one spatial grid, while spatial tree leaves may use different grids. One shared
  writer preserves common Dask tasks across spatial outputs and diagnostics.
  Explicit loaders are `load_array`, `load_dataset`, and `load_tree`; the former
  `load` entry point is removed. Xarray `2026.9.0` or newer is required.
- ConfUSIus [draft PR #569](https://github.com/confusius-tools/confusius/pull/569)
  delegates geometry, extraction, and ordinary Zarr array persistence to
  VoxelData. Probe positions use `pose`;
  acquisition defaults, timing policies, and fUSI analysis remain in ConfUSIus.
  Its unit suite, including napari and Matplotlib visual comparisons, passes
  against the sibling VoxelData checkout.

API contracts and usage belong in the README and `docs/api/`.

## Deferred downstream adoption

- Update ConfUSIus's minimum dependency and lockfile to VoxelData `0.0.5`.
- Resolve the ConfUSIus integration draft's conflicts with its target branch.
- Validate the updated branch against the published dependency, including the
  complete unit, integration, visual-regression, and supported OS/Python matrix.
- Complete review and merge the ConfUSIus integration.

## Optional future work

### Numerical world-unit conversion

Add an explicit operation that converts world-coordinate units while preserving
physical geometry, native voxel labels, and lazy signal data. Keep it distinct
from `.vd.set_world_units()`, which only relabels the unit metadata. Scale both
the linear and translation parts of voxel-to-world matrices; define named-frame
target-unit semantics before converting their matrices. Cover round-trip unit
conversion, shared and stacked geometry, and named-frame consistency.

### Geometry-axis reindexing

Native Xarray `.reindex()` is unsupported on dimensions listed in
`data.vd.geometry_dims`. Use `.isel()` or `.sel()` to select/reorder existing
geometry labels instead. Alignment or coordinate replacement that changes those
labels has the same restriction. Filling signal samples for an unknown label does
not define its geometry.

Xarray's reindexing path does not synchronize the world index's affine metadata
with the independently indexed geometry labels. VoxelData access/validation
detects mismatching indexed labels, but raw world-coordinate access cannot perform
that parent-array check. Any future support must preserve independent indexing
and define geometry for added labels, without monkey-patching Xarray.

This remains an accepted limitation, not a release prerequisite.
`.vd.reindex_voxels()` and `.vd.reindex_voxels_like()` explicitly rebuild geometry
and remain supported.

### Further modality-independent utilities

Assess additional ConfUSIus utilities individually when another consumer needs
them. Move code only when it directly serves the VoxelData model and has a clear
dependency boundary. Keep acquisition policies, modality-specific analysis, atlas
services, and visualization in their appropriate downstream packages. No further
extraction target is currently committed.
