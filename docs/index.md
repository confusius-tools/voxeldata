---
hide:
    - navigation
    - toc
---

<!-- Suppress the auto-injected page-title h1 from Zensical's content template.
     Visually hidden (not display:none) so screen readers still see the h1. -->
<h1 class="sr-only">VoxelData</h1>

<div class="hero-banner" markdown>

<p class="hero-title">Voxel<span class="hero-accent">Data</span></p>

<p class="hero-tagline">A small, modality-independent data model for spatially referenced voxel arrays, built on xarray.</p>

<div class="hero-badges">
  <a href="https://pypi.org/project/voxeldata/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/pypi/v/voxeldata?style=flat-square&color=d93a54&label=PyPI" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/voxeldata/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/pypi/pyversions/voxeldata?style=flat-square&color=0099e5&logo=python&logoColor=white" alt="Python versions">
  </a>
  <a href="https://github.com/confusius-tools/voxeldata/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/github/license/confusius-tools/voxeldata?style=flat-square&color=3ad9a4" alt="License">
  </a>
  <a href="https://doi.org/10.64898/2026.10.27.762250" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/DOI-10.64898%2F2026.10.27.762250-8b7ff7?style=flat-square" alt="ConfUSIus preprint DOI">
  </a>
  <a href="https://discord.gg/mZd87tgmy2" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Discord-join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

</div>

!!! info "Alpha Status"
    VoxelData is in **alpha** and under active development. APIs may still evolve.

## :lucide-box: The model

A VoxelData array is an `xarray.DataArray` with:

1. dimensions `(*nonspatial_dims, k, j, i)`, with arbitrary leading dimension
   order, optional special dimensions `time` and `grid`, and `k`/`j`/`i` always
   present and always last;
2. a `VoxelToWorldIndex` on the world coordinates `z`/`y`/`x`, deriving them
   lazily from a voxel-to-world affine;
3. optional `time` coordinates carrying a `units` attribute.

`geometry_dims` explicitly names the nonspatial dimensions that select affine
matrices. The required affine shape is
`(*[data.sizes[d] for d in geometry_dims], 4, 4)`. Unrelated nonspatial dimensions
share geometry. A single `(4, 4)` affine has `geometry_dims=()`, regardless of the
array's nonspatial dimensions.

For compatibility, omitted `geometry_dims` on a 3D stack infers `("grid",)`, or
`("pose",)` when only the legacy probe-pose dimension is present. All other stacks
require explicit names; matching dimension lengths are never used to infer axes.
Identical affines are allowed. Inferred legacy stacks retain their equal-scale
validation; explicit geometry dependencies may have varying spatial scalings.
`time` represents temporal samples, independently of geometry dependencies.

Creation preserves the input nonspatial dimension order, so both
`(time, grid, k, j, i)` and `(grid, time, k, j, i)` are valid. Canonicalization and
`ensure_voxeldata` normalize to `(...extra_dims, grid, time, k, j, i)`, preserving
the relative order of extra dimensions and omitting absent `grid`/`time` dimensions.
Scalar-selected spatial dimensions are restored, but a scalar-selected grid is not.

World coordinates remain lazy: representing, indexing, transposing, or rebuilding
the model does not generate dense world arrays. Explicitly requesting coordinate
values, for example `data.z.values`, evaluates the requested geometry/voxel product.
Unrelated dimensions are not included. Native voxel labels are inputs to the affine:
selecting or cropping `[10, 12, 14]` does not silently replace them with `[0, 1, 2]`.

## :lucide-terminal: Quick start

```python
import numpy as np
import voxeldata as vd

data = vd.create_voxeldata(
    np.zeros((3, 64, 128)),
    dims=("k", "j", "i"),
    spacing=(0.5, 0.1, 0.1),
    origin=(0.0, 0.0, 0.0),
    units="mm",
)

data.coords["x"]     # lazily derived world coordinates
data.vd.origin       # {'z': 0.0, 'y': 0.0, 'x': 0.0}
data.vd.spacing      # {'k': 0.5, 'j': 0.1, 'i': 0.1}
data.vd.direction    # (3, 3) world direction matrix
```

Pass `voxel_to_world` instead of `spacing`/`origin`/`direction` for an arbitrary
(including oblique) affine. Time and legacy grid/probe-pose coordinates are supported
with `time`, `grid`, `pose`, `dt`, and `t0`.

## Named geometry dependencies

For moving geometry indexed by time and view, with two channels sharing it:

```python
affines = np.broadcast_to(np.eye(4), (3, 2, 4, 4)).copy()
affines[1:, :, 0, 3] = 10.0
moving = vd.create_voxeldata(
    np.zeros((2, 3, 2, 4, 5, 6)),
    dims=("channel", "time", "view", "k", "j", "i"),
    time=[0.0, 1.0, 2.0],
    extra_coords={"channel": ["a", "b"], "view": [11, 41]},
    voxel_to_world=affines,
    geometry_dims=("time", "view"),
)

assert moving.vd.geometry_dims == ("time", "view")
assert moving.z.dims == ("time", "view", "k", "j", "i")
assert moving.sel(view=41).vd.geometry_dims == ("time",)
assert moving.isel(time=0, view=1).vd.geometry_dims == ()
```

- **Indexing:** scalar geometry selection removes its affine axis. Slices, lists,
  Boolean masks, and Xarray's named vectorized geometry indexers select matching
  matrices and named frames. Native voxel selections preserve their original labels.
  Vectorized voxel indexing that is no longer a Cartesian grid drops the geometry
  index, as before; it must not be interpreted as another regular VoxelData grid.
- **Array order:** `.transpose()` and canonicalization do not reorder affine batch
  axes. `.rename(view="angle")` updates the dependency name. Spatial names remain
  fixed: `(k, j, i)` map to `(z, y, x)`.
- **World lookup:** select a scalar on *every* geometry dimension first, then make a
  separate world-coordinate `.sel()` call. A combined geometry/world lookup cannot
  be resolved by Xarray's independent indexes and raises an error.
- **Frames and composition:** named frames live on the index, not in array attrs.
  Each frame is shared `(4, 4)` or has the complete ordered geometry batch shape.
  `.vd.apply(action, geometry_dims=("view",))` aligns action axes by name and may
  introduce new dependencies. `.vd.to(other, "lab")` aligns frame dependencies by
  name and rejects differing labels/order on shared axes. Returned NumPy batches
  use source dependency order followed by any additional target dependencies.
- **Geometry helpers:** bounding boxes retain all geometry axes and reindexing
  operates directly on affine stacks, without generating dense world coordinates.
  Scalar origin/direction require a selected geometry. Spacing/scalings require
  equal scales across geometries; otherwise select one first. `.vd.spacing` still
  includes ordinary nonspatial coordinate spacing, where defined.
- **Timing:** timestamps may vary per legacy grid/probe pose or, with a named 2D
  time DataArray, per another nonspatial dimension. Their dependency need not match
  `geometry_dims`. Timing values and attrs survive slicing and canonicalization.
  VoxelData does not interpret `slice_time` or supply modality-specific timing defaults.
- **Persistence:** Python pickle round-trips preserve geometry dependencies and
  named frames. VoxelData does not provide a Zarr/NetCDF serializer: external
  serializers must save ordered `geometry_dims`, affine stacks, units, native voxel
  labels, and named frames, and pass `geometry_dims` explicitly when restoring them.

!!! warning "Unsupported geometry-axis reindexing"
    Native Xarray `.reindex()` is unsupported on dimensions listed in
    `data.vd.geometry_dims`. Use `.isel()` or `.sel()` to select/reorder these axes, not `.reindex()` or
    alignment that changes their labels. Xarray may bypass the custom world index
    during these operations, leaving stale geometry. VoxelData access/validation
    rejects mismatching indexed geometry labels; Xarray's raw world coordinates
    cannot perform this parent-array check. Replacing geometry-axis labels directly
    has the same restriction. Concatenation is supported along existing geometry
    axes with matching other coordinates; concatenating native voxel axes is not.
    The explicit `.vd.reindex_voxels()` and `.vd.reindex_voxels_like()` operations
    rebuild the geometry and are not affected by this restriction.

Dask-backed signal data remain lazy through construction, validation, selection,
canonicalization, bounds, and reindexing. Affine stacks and dimension labels are
small, eager metadata; arbitrary `slice_time` coordinates remain ordinary metadata.

## :lucide-layers: Features

<div class="grid cards" markdown>

-   [:lucide-ruler:{ .lg .middle } **Geometry**](api/geometry.md)

    ---

    Lazy world coordinates from a single affine, with exact behavior through
    slicing, cropping, and grid stacking.

-   [:lucide-package-plus:{ .lg .middle } **Creation**](api/create.md)

    ---

    Build arrays from raw data with spacing, origin, direction, or a full
    voxel-to-world affine.

-   [:lucide-shield-check:{ .lg .middle } **Validation**](api/validation.md)

    ---

    Check the data model, canonicalize dimension order, and restore dimensions
    removed by scalar indexing.

-   [:lucide-move-3d:{ .lg .middle } **Affine operations**](api/affine.md)

    ---

    Apply, rebase, and relate world frames, including named affine frames for
    multi-space workflows.

-   [:lucide-mouse-pointer-click:{ .lg .middle } **Accessors**](api/accessors.md)

    ---

    A flat `data.vd` accessor exposing every geometry operation directly on the
    DataArray.

-   [:lucide-list:{ .lg .middle } **Dimensions**](api/dims.md)

    ---

    Spatial and special nonspatial dimension names, with an optional canonical order.

</div>

## :lucide-book-open: API reference

- [Geometry](api/geometry.md)
- [Creation](api/create.md)
- [Validation](api/validation.md)
- [Affine operations](api/affine.md)
- [Accessors](api/accessors.md)
- [Dimensions](api/dims.md)
