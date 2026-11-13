[![PyPI version](https://img.shields.io/pypi/v/voxeldata?style=flat-square&color=d93a54&label=PyPI)](https://pypi.org/project/voxeldata/)
[![Python versions](https://img.shields.io/pypi/pyversions/voxeldata?style=flat-square&color=0099e5&logo=python&logoColor=white)](https://pypi.org/project/voxeldata/)
[![License](https://img.shields.io/github/license/confusius-tools/voxeldata?style=flat-square&color=3ad9a4)](https://github.com/confusius-tools/voxeldata/blob/main/LICENSE)
[![ConfUSIus preprint DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.10.27.762250-8b7ff7?style=flat-square)](https://doi.org/10.64898/2026.10.27.762250)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/mZd87tgmy2)

# VoxelData

**VoxelData** is a small, modality-independent data model for spatially referenced
voxel arrays, built on [xarray](https://docs.xarray.dev).

A VoxelData array is an `xarray.DataArray` that is always:

- dimensioned `(*nonspatial_dims, k, j, i)` — leading dimensions may appear in any
  order, including the optional temporal dimension `time`; the native voxel
  indices `k`/`j`/`i` are always present and always last;
- backed by a single [`VoxelToWorldIndex`](src/voxeldata/geometry.py) attached to
  the world coordinates `z`/`y`/`x`, which derives them lazily from a shared affine
  or an affine stack indexed by named nonspatial `geometry_dims`.

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

`geometry_dims` names the nonspatial dimensions indexing an affine stack. For
example, `geometry_dims=("time", "view")` requires `(ntime, nview, 4, 4)` matrices;
other axes, such as channels, share geometry. A single `(4, 4)` affine has no
geometry dependencies, even when nonspatial dimensions are present. Timing
coordinates are independent of these geometry dependencies. A named `time`
DataArray may use `(time, view, channel, ...)`, with any number of nonspatial
dependencies. Each timestamp series must increase along `time`. Select all
additional timestamp axes before using a shared-time index or analysis.
Auxiliary coordinates such as `slice_time(time, k)` may be added with
`assign_coords`; their acquisition-specific meaning belongs to the consumer.

All affine stacks require explicit `geometry_dims`; dimension names or matching
lengths never determine their meaning. Coordinates for arbitrary nonspatial axes
are supplied through `extra_coords`:

```python
data = vd.create_voxeldata(
    np.zeros((2, 3, 64, 128)),
    dims=("view", "k", "j", "i"),
    extra_coords={"view": [0, 1]},
    voxel_to_world=affines,          # (2, 4, 4)
    geometry_dims=("view",),
)

assert data.vd.geometry_dims == ("view",)
```

**Limitation:** native Xarray `.reindex()` is unsupported on geometry dimensions;
use `.isel()` or `.sel()` to select/reorder existing labels instead. Alignment that
changes those labels is also unsupported. This does not affect the explicit
`.vd.reindex_voxels()` and `.vd.reindex_voxels_like()` geometry operations.

Creation preserves the supplied nonspatial dimension order. For example, both
`(time, view, k, j, i)` and `(view, time, k, j, i)` are valid.
`data.vd.canonicalize()` and `data.vd.ensure()` normalize to
`(...extra_dims, time, k, j, i)`, preserving the relative order of extra
dimensions and omitting absent `time` dimensions. Canonicalization also
restores scalar-selected spatial dimensions, but does not restore selected nonspatial dimensions.

`get_array_origin(data)` returns the world position of the first sampled voxel;
`get_array_spacing(data)` returns physical step lengths along native voxel axes.
Both account for retained voxel labels after cropping or striding. `.vd.origin`
and `.vd.spacing` expose the same spatial-only values, keyed by `z/y/x` and
`k/j/i`, respectively. `.vd.world_units` and `.vd.set_world_units(...)` describe
world-coordinate units, not signal-value or time units.

Use `.vd.dt` for the time sampling interval, in the time coordinate's units
(normally seconds). It returns `None` for irregular sampling or fewer than two
timestamps; uniformity uses a 1% relative tolerance around the median interval.
Select any additional timestamp dependencies first, even when their intervals
agree. Missing time coordinates raise `ValueError`. For other dimensions, access
their coordinates directly rather than treating them as spatial metadata.

## World-space transforms

```python
shifted = data.vd.translate((0, 0, 1.5))  # World (z, y, x), in data.vd.world_units.
rotated = data.vd.rotate(90, "x")
zoomed = data.vd.zoom((1, 2, 2))
flipped = data.vd.flip("yx")
center = data.vd.geometry_center
```

These operations change geometry only: they do not resample, reorder, or compute
signal data. Native voxel labels and named frames are preserved (frames are
re-expressed relative to the new world frame). They return a new array by default;
`inplace=True` returns the updated original and requires all native voxel dimensions.

Rotation, zoom, and flip default to `center="geometry"`, the midpoint of the
bounding box **enclosing all grids**, so every geometry uses one shared pivot.
`bounding_box` maps each world axis to a `(min, max)` tuple, e.g.
`data.vd.bounding_box["x"]`. `geometry_center` maps each world axis to a float,
e.g. `data.vd.geometry_center["x"]`, matching `.vd.origin`. Values use
`.vd.world_units`. Select geometry dependencies first for grid-specific values,
e.g. `data.sel(view=20).vd.bounding_box` and
`data.sel(view=20).vd.geometry_center`.

Bounds and centers return dictionaries, not DataArrays: replace
`bounding_box.sel(bound="min", component="x")` with `bounding_box["x"][0]`
and `geometry_center.sel(component="x")` with `geometry_center["x"]`.

`center="origin"` uses the first sampled voxel, including after cropping or
reversing native labels, and requires selecting all geometry dependencies first.
An explicit `center=(z, y, x)` or mapping with `z`, `y`, `x` keys also supplies a
shared world pivot.

Rotations follow the right-hand rule in ordered `(z, y, x)` coordinates: +90 degrees
about `x` maps `+z` onto `+y`. Multiple axis names apply left to right about fixed
world axes, e.g. `.vd.rotate((30, 20, 10), "zyx")`; use `degrees=False` for radians.
Zoom factors act along world axes; negative factors reflect, and unequal factors
can shear an oblique grid. A flip changes world geometry, not the sample order.
Irregular voxel labels require an explicit or origin pivot rather than a geometry
center. Scalar-selected voxel dimensions are restored in non-inplace results.

## Public API

| Area | Functions |
|---|---|
| Dimensions | `VOXEL_DIMS`, `WORLD_DIMS`, `TIME_DIM`, `CORE_DIMS` |
| Construction | `create_voxeldata`, `attach_voxel_to_world_index` |
| Accessor | `data.vd` ([`VoxelDataAccessor`](src/voxeldata/accessors.py)) |
| Geometry | `get_voxel_to_world_affine`, `get_voxel_to_world_units`, `get_array_origin`, `get_array_spacing`, `get_voxel_to_world_direction_matrix`, `convert_length_units` |
| Validation | `validate_voxeldata`, `ensure_voxeldata`, `canonicalize_voxeldata` |
| Affine operations | `apply_affine`, `apply_translation`, `apply_rotation`, `apply_zoom`, `apply_flip`, `get_geometry_center`, `reindex_voxels`, `reindex_voxels_like`, `get_bounding_box`, `get_relative_affine` |
| Named affines | `data.vd.get_affine`, `set_affine`, `remove_affine`, `apply`, `to` |
| Extraction | `extract_with_mask`, `extract_with_labels`, `unmask` |
| Persistence | `save`, `load_array`, `load_dataset`, `load_tree` |
| Masks and labels | `ensure_mask`, `validate_mask`, `ensure_labels`, `validate_labels`, `check_spatial_alignment` |

The `VoxelToWorldIndex` and `VoxelToWorldTransform` classes implement the xarray
indexing machinery and are exported for advanced use.

## Extraction

`voxeldata.extract` provides generic mask extraction, region aggregation, and
reconstruction. All three functions work out of the box using Xarray. Install
`voxeldata[accel]` for optional flox group-reduction acceleration,
`voxeldata[lazy]` for Dask arrays, `voxeldata[io]` for storage backends, or
`voxeldata[all]` for all runtime features:

```bash
uv add 'voxeldata[all]'
```

`all` includes optional runtime features, not development tools.

```python
signals = vd.extract_with_mask(data, mask)  # (..., space), selected native labels.
regions = vd.extract_with_labels(data, labels, reduction="mean")  # (..., region).
restored = vd.unmask(signals, mask)         # Rebuild the mask's full spatial grid.
```

Masks/labels must share the physical voxel grid. Select varying geometry before
using a shared spatial mask. Extraction preserves lazy signal data; `unmask` is
eager. See the [extraction guide](docs/api/extract.md) for formats and chunking limits.

## Saving and loading

Install `voxeldata[io]` for Zarr and NetCDF persistence:

```python
vd.save(data, "volume.zarr")  # Or "volume.nc" for NetCDF.
with vd.load_array("volume.zarr") as restored:
    volume = restored.compute()
```

Both formats restore the spatial index, ordered geometry dependencies, units,
native voxel labels, and named frames without storing dense world coordinates.
Loading uses storage-aligned Dask chunks when Dask is installed (for example via
`voxeldata[lazy]`), and backend lazy indexing otherwise. Pass `chunks=None` to opt
out of Dask. NetCDF overwrites existing files by default, following Xarray. See the
[persistence guide](docs/api/io.md) for metadata restrictions and writer options.

Zarr also supports **VoxelData datasets**: ordinary Xarray Datasets containing
spatial VoxelData variables on one shared grid, plus optional nonspatial variables.
A **VoxelData tree** is an ordinary DataTree with these datasets at spatial leaves;
grouping ancestors carry no spatial coordinates. Different leaves may use different
grids. Variable names, diagnostics, and group names remain consumer-defined.

```python
import xarray as xr

results = xr.Dataset({"image": data, "valid": data > 0})
vd.save(results, "results.zarr", compute=False).compute()
with vd.load_dataset("results.zarr") as restored:
    vd.validate_voxeldata(restored["image"])  # Direct selection retains geometry.

# For multiple grids, save a DataTree and restore it with vd.load_tree(...).
```

Each container save uses one shared Dask write graph. Keep shared outputs in one
container rather than making separate saves. Container I/O is Zarr-only.
Xarray 2026.9.0 or newer is required.

## Relationship to ConfUSIus

VoxelData was originally developed as
[ConfUSIus'](https://github.com/confusius-tools/confusius) data model. It is now a
standalone package and offers a general-purpose voxel data model for any spatially
referenced array.

## Development

See the [roadmap](ROADMAP.md) for planned geometry and extraction work.

```bash
uv sync            # install dependencies
just test          # run tests
just pre-commit    # run lint, format, type checks, docstring checks
just docs          # build documentation
```

## License

BSD-3-Clause. See `LICENSE` and `NOTICE`.
