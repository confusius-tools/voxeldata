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

The `grid` dimension selects a spatial sampling grid. Its presence requires one
affine per grid, even when there is only one grid. Each affine applies across time
and all other nonspatial dimensions within that grid; different grids may have
identical affines. A single shared affine is only valid with no `grid` dimension.
`time` represents temporal samples and may carry separate timestamps per grid.

Creation preserves the input nonspatial dimension order, so both
`(time, grid, k, j, i)` and `(grid, time, k, j, i)` are valid. Canonicalization and
`ensure_voxeldata` normalize to `(...extra_dims, grid, time, k, j, i)`, preserving
the relative order of extra dimensions and omitting absent `grid`/`time` dimensions.
Scalar-selected spatial dimensions are restored, but a scalar-selected grid is not.

World coordinates are never materialized and stored as data. They are computed from
the affine, so slicing, cropping, and indexing preserve exact physical geometry.

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
(including oblique) affine. Time and grid coordinates are supported with `time`,
`grid`, `dt`, and `t0`.

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
