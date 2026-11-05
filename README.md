# VoxelData

**VoxelData** is a small, modality-independent data model for spatially referenced
voxel arrays, built on [xarray](https://docs.xarray.dev).

A VoxelData array is an `xarray.DataArray` that is always:

- dimensioned `(*nonspatial_dims, k, j, i)` — leading dimensions may appear in any
  order, including optional special dimensions `time` and `grid`; the native voxel
  indices `k`/`j`/`i` are always present and always last;
- backed by a single [`VoxelToWorldIndex`](src/voxeldata/geometry.py) attached to
  the world coordinates `z`/`y`/`x`, which derives them lazily from one affine
  shared by all data or one affine per `grid`.

World coordinates are never stored directly. This keeps cropping, slicing,
registration, and I/O consistent: the affine always describes the whole physical
grid, and indexing updates it exactly.

## Installation

```bash
uv add voxeldata
```

Or with pip:

```bash
pip install voxeldata
```

## Quick start

```python
import numpy as np
import voxeldata as vd

data = vd.create_voxeldata(
    np.zeros((3, 64, 128)),
    dims=("k", "j", "i"),
    spacing=(0.5, 0.1, 0.1),          # z, y, x in millimetres
    origin=(0.0, 0.0, 0.0),
)

data.coords["x"]        # lazily derived world coordinates
data.sel(x=5.0)          # select in world space
vd.validate_voxeldata(data)

data.vd.origin           # {'z': 0.0, 'y': 0.0, 'x': 0.0}
data.vd.spacing          # {'k': 0.5, 'j': 0.1, 'i': 0.1}
data.vd.direction        # (3, 3) world direction matrix
```

Oblique grids work the same way:

```python
data = vd.create_voxeldata(
    np.zeros((3, 64, 128)),
    dims=("k", "j", "i"),
    voxel_to_world=affine,           # (4, 4) homogeneous affine
)
```

The `grid` dimension selects a spatial sampling grid: each grid has one affine,
shared across time and all other nonspatial dimensions. Affines may be identical
across grids. A `grid` dimension requires an `(ngrid, 4, 4)` affine stack, even for
one grid; without it, geometry uses a single `(4, 4)` affine.

Grid-dependent geometry carries one affine per grid:

```python
data = vd.create_voxeldata(
    np.zeros((2, 3, 64, 128)),
    dims=("grid", "k", "j", "i"),
    grid=np.array([0, 1]),
    voxel_to_world=affines,          # (2, 4, 4)
)
```

Creation preserves the supplied nonspatial dimension order. For example, both
`(time, grid, k, j, i)` and `(grid, time, k, j, i)` are valid.
`data.vd.canonicalize()` and `data.vd.ensure()` normalize to
`(...extra_dims, grid, time, k, j, i)`, preserving the relative order of extra
dimensions and omitting absent `grid`/`time` dimensions. Canonicalization also
restores scalar-selected spatial dimensions, but does not restore a selected grid.

## Public API

| Area | Functions |
|---|---|
| Dimensions | `VOXEL_DIMS`, `WORLD_DIMS`, `GRID_DIM`, `TIME_DIM`, `CORE_DIMS` |
| Construction | `create_voxeldata`, `attach_voxel_to_world_index` |
| Accessor | `data.vd` ([`VoxelDataAccessor`](src/voxeldata/accessors.py)) |
| Geometry | `get_voxel_to_world_affine`, `get_voxel_to_world_units`, `get_voxel_to_world_index_origin`, `get_voxel_to_world_index_spacing`, `get_voxel_to_world_direction_matrix`, `convert_length_units` |
| Validation | `validate_voxeldata`, `ensure_voxeldata`, `canonicalize_voxeldata` |
| Affine operations | `apply_affine`, `reindex_voxels`, `reindex_voxels_like`, `get_bounding_box`, `get_relative_affine` |
| Named affines | `data.vd.get_affine`, `set_affine`, `remove_affine`, `apply`, `to` |

The `VoxelToWorldIndex` and `VoxelToWorldTransform` classes implement the xarray
indexing machinery and are exported for advanced use.

## Relationship to ConfUSIus

VoxelData is extracted from [ConfUSIus](https://github.com/confusius-tools/confusius),
which uses it for functional ultrasound imaging data. The model itself makes no
assumption about modality, acquisition hardware, or signal semantics, so it is
usable by any package working with spatially referenced voxel data.

## Development

```bash
uv sync            # install dependencies
just test          # run tests
just pre-commit    # run lint, format, type checks, docstring checks
just docs          # build documentation
```

## License

BSD-3-Clause. Portions derived from ConfUSIus; see `NOTICE`.
