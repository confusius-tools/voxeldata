# Roadmap

## Implemented foundations

Named geometry dependencies and generic extraction are implemented on `main`.
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
- ConfUSIus [draft PR #569](https://github.com/confusius-tools/confusius/pull/569)
  delegates geometry and extraction to VoxelData. Probe positions use `pose`;
  acquisition defaults, timing policies, and fUSI analysis remain in ConfUSIus.
  Its unit suite, including napari and Matplotlib visual comparisons, passes
  against the sibling VoxelData checkout.

API contracts and usage belong in the README and `docs/api/`.

## Deferred release and downstream adoption

- Agree and publish a VoxelData release containing the current API changes.
- Update ConfUSIus's minimum dependency and lockfile to that release.
- Resolve the ConfUSIus integration draft's conflicts with its target branch.
- Validate the updated branch against the published dependency, including the
  complete unit, integration, visual-regression, and supported OS/Python matrix.
- Complete review and merge the ConfUSIus integration.

## Optional future work

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
