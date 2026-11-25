---
icon: lucide/filter
---

# Extraction and reconstruction

Extraction is modality-independent: it needs voxel geometry, native labels, and a
mask or label map—not fUSI acquisition metadata. The functions are available from
`voxeldata.extract`, directly from `voxeldata`, and through the flat `.vd`
methods `extract_with_mask`, `extract_with_labels`, and `unmask`.

## Installation

All extraction functions are included in a normal installation, using Xarray's
built-in grouped reductions:

```bash
uv add voxeldata
```

Choose optional backends independently:

| Installation | Additional backend |
|---|---|
| `voxeldata` | None; all extraction functions work using Xarray/NumPy |
| `voxeldata[accel]` | flox for automatic group-reduction acceleration |
| `voxeldata[lazy]` | Dask for lazy/chunked signal data |
| `voxeldata[accel,lazy]` or `voxeldata[all]` | Both |

```bash
uv add 'voxeldata[all]'
```

The `all` extra combines optional runtime features; it does not install development
tools.

Neither backend is needed for NumPy-backed extraction, including overlapping label
maps. The `lazy` extra does not make reconstruction lazy: `unmask` remains eager.

## Example

```python
import numpy as np
import voxeldata as vd

values = np.arange(48).reshape(2, 2, 3, 4)
data = vd.create_voxeldata(
    values, dims=("time", "k", "j", "i"), dt=0.5, spacing=(1.0, 1.0, 1.0)
)
mask = vd.create_voxeldata(
    np.ones((2, 3, 4), dtype=bool), dims=("k", "j", "i"),
    voxel_to_world=data.vd.voxel_to_world,
)
signals = data.vd.extract_with_mask(mask)  # (time, space).
restored = signals.vd.unmask(mask)         # (time, k, j, i).
# Label aggregation: data.vd.extract_with_labels(labels, reduction="mean").
```

The accessor delegates to the same module-level functions. Reconstruction is
available on geometry-free signals through `signals.vd.unmask(mask)`;
the mask supplies the output geometry. Extraction still requires VoxelData inputs
and matching physical grids.

- Mask extraction retains selected native voxel labels in a `space` MultiIndex.
  A partial `.unstack("space")` reconstructs their bounding grid; `unmask` restores
  the original mask's complete shape and fills unselected voxels.
- Label aggregation excludes zero/background. Flat maps identify regions by their
  integer ids. Stacked masks use their `mask` coordinate to identify regions;
  overlapping layers and repeated or negative nonzero ids are supported.
- Binary masks flatten all their dimensions into `space`, including any extra
  feature/geometry axes such as channel or view. Other signal dimensions are
  retained. Label maps are spatial-only, optionally with a leading `mask` axis.
  The model's existing canonical leading-axis ordering is applied first.
- A shared spatial mask/map may span geometry dependencies only when the affine
  matrices agree. Grid origins may differ by at most `1e-5` times the shortest
  voxel-axis length in the two affines. Select a geometry first when its physical
  grid differs. Geometry axes retain their ordinary indexes when not included in
  the flattened mask.
- Signal data remain lazy for Dask-backed extraction and aggregation; mask/label
  values are evaluated eagerly. With flox acceleration, median requires each region to fit
  within an individual Dask chunk. Explicitly rechunk spatial axes when necessary;
  VoxelData does not silently gather an entire volume into one chunk.
- Reconstruction is eager and positional in the `space` samples. It preserves the
  mask's native labels, units, affine, and named frames, and the signal table's
  leading dimension coordinates/attributes. It does not realign samples using a
  `space` MultiIndex or copy arbitrary auxiliary coordinates from the signal table.
- No acquisition units/reference/duration defaults are added. Existing timestamp
  attributes are retained where timestamp coordinates are preserved.

## Matching binary masks by label

By default, `extract_with_mask` requires the same set of labels on each mask
dimension. It reads the signal in mask-label order, including when native voxel
labels run in the opposite direction. For a cropped mask or partially overlapping
recordings, request an inner join:

```python
# recording.pose contains [10, 20, 30] and mask.pose contains [20, 40].
signals = recording.vd.extract_with_mask(mask, join="inner")
# Only pose 20 contributes samples; the spatial labels match in this example.
restored = signals.vd.unmask(mask.sel(pose=[20]))
```

The intersection is taken separately along each named mask axis. An empty
intersection raises an error. Reordering or intersecting unequal indexes needs
unique labels. Keep singleton axes with list selection, such as `sel(pose=[20])`,
to restrict extraction to a particular pose. A spatial mask with no `pose`
dimension applies across all poses only if they share a physical grid. Affines
and units must agree on the matched grids.

For positional reconstruction, pass the mask restricted to the same overlap,
retaining its label order. When every mask label was found, pass the original mask.

::: voxeldata.extract.extract_with_mask

::: voxeldata.extract.extract_with_labels

::: voxeldata.extract.unmask
