# Roadmap

## 1. Finish named-geometry integration

Support named geometry dependencies without coupling ordinary voxel and
nonspatial indexing in ways that break downstream extraction.

### Accepted limitation: geometry-axis reindexing

Native Xarray `.reindex()` is unsupported on dimensions listed in
`data.vd.geometry_dims`. Use `.isel()` or `.sel()` to select/reorder existing
geometry labels instead. Alignment that changes geometry labels has the same
restriction. Filling signal samples for an unknown label does not define its
geometry.

The world index owns `z/y/x`, while ordinary indexes own geometry-axis labels.
Xarray's reindexing path does not synchronize their separate metadata. No Xarray
bug has been established, and no upstream change or monkey-patch is planned for
this iteration. VoxelData access/validation detects mismatching indexed geometry
labels, but raw world-coordinate access cannot perform that parent-array check.

This limitation is distinct from `.vd.reindex_voxels()` and
`.vd.reindex_voxels_like()`, which explicitly rebuild geometry and remain supported.
Native geometry-axis reindexing can be revisited separately; it is not a release
prerequisite.

### Remaining integration work

- Keep affine metadata contiguous with named, labelled batch axes. A dictionary
  keyed by labels alone does not solve synchronization with parent-array indexing.
- Preserve independent native voxel and nonspatial indexing. Reintroducing one
  joint index covering `k/j/i` and probe poses previously caused problems in
  ConfUSIus extraction.
- Cover mask/label extraction, spatial stacking and unstacking, scalar and
  vectorized indexing, transpose, concatenation, and lazy arrays.
- Update the ConfUSIus integration draft
  [PR #569](https://github.com/confusius-tools/confusius/pull/569), including
  constructor/serialization metadata and downstream expectations as needed. Run
  the full ConfUSIus suite, including napari, integration, and visual-regression
  tests, and report any unavailable checks.

## 2. Move generic extraction from ConfUSIus

**In progress: standalone implementation and test port. Downstream delegation remains.**

The functions in `confusius.extract` operate on spatially referenced voxel arrays,
not specifically on functional ultrasound data. The standalone `voxeldata.extract`
module now provides:

- `extract_with_mask`: select voxel samples while retaining all nonspatial axes.
- `extract_with_labels`: aggregate voxel samples by flat label maps or overlapping
  stacked masks, retaining region identifiers and nonspatial axes.
- `unmask`: reconstruct spatial arrays from extracted samples and a mask.

### Scope and acceptance conditions

- Identify and move only the generic mask/label validation and geometry-alignment
  helpers these functions require. Do not import ConfUSIus into VoxelData or bring
  fUSI acquisition-timing requirements into generic validation.
- Specify mask/label compatibility for shared and geometry-dependent grids. Do not
  broadcast a mask across physically different grids without an explicit rule.
- Preserve native voxel labels, nonspatial coordinates and metadata, region
  identifiers, reconstruction geometry, and existing lazy-array behavior.
- Extraction is included out of the box through Xarray's grouped reductions.
  Flox acceleration and Dask laziness are optional; both are dev dependencies for CI.
  Native and accelerated reductions share the same public API and numerical tests.
- Port applicable ConfUSIus tests and add multi-geometry and Dask regressions with
  independent extraction/reconstruction oracles.
- Update ConfUSIus consumers and its `.fusi` extraction layer to delegate to
  VoxelData, removing duplicate implementation rather than creating a second one.
- Public functions are in `voxeldata.extract`, with top-level re-exports. Generic
  validators live in `voxeldata.masks`. No extraction accessor methods or
  compatibility aliases are added.

## Further candidates

After extraction, assess other modality-independent utilities individually. Move
code only when it directly serves the VoxelData model and has a clear dependency
boundary; keep modality-specific analysis, acquisition timing, atlas services, and
visualization in their appropriate downstream packages.
