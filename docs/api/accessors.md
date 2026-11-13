# Accessors

The flat `.vd` accessor exposes the VoxelData model directly on any
`xarray.DataArray`. Importing `voxeldata` registers it.

```python
import numpy as np
import voxeldata as vd

data = vd.create_voxeldata(
    np.zeros((3, 64, 128)),
    dims=("k", "j", "i"),
    spacing=(0.5, 0.1, 0.1),
)

data.vd.origin          # {'z': 0.0, 'y': 0.0, 'x': 0.0}
data.vd.spacing         # {'k': 0.5, 'j': 0.1, 'i': 0.1}
data.vd.apply("lab")    # apply a stored named frame
```

`.vd.origin` returns only the spatial world position of the first sampled voxel
(`z/y/x`), and `.vd.spacing` returns only physical sample spacing along active
native voxel axes (`k/j/i`). Both account for cropping and striding. The spatial
unit is `.vd.world_units`; these properties never include nonspatial coordinates.

Use `.vd.dt` for the interval of a shared time coordinate, in its own units
(normally seconds). It returns `None` for irregular sampling or fewer than two
timestamps, including a scalar-selected time point. Uniformity uses a 1% relative
tolerance around the median interval. Missing time coordinates and invalid
(non-finite, non-real, or non-increasing) timestamps raise `ValueError`. Select
all additional timestamp dependencies first, even when their sampling intervals
agree. Geometry dependencies are independent and need not be selected for `dt`.
For other dimensions, access coordinates directly.

`.vd.canonicalize(**validation_options)` restores scalar-selected voxel axes,
normalizes dimension order, and validates the result. There is no separate ensure
entry point or unchecked canonicalization API. Use `.vd.validate(...)` to check
an array without changing its dimension order.

`.vd.get_relative_affine(other, via=...)` returns the relative matrix through a
shared named frame; it does not transform the array. Read frames with
`.vd.affines[name]`; update them through `set_affine` and `remove_affine`. The
returned frame dictionary is a shallow copy, not a mutation interface. The actual
index and advanced selection state remain accessible through `.vd.index`.

`set_voxel_to_world` is the only method that accepts a plain DataArray without an
existing voxel-to-world index; every other property and method raises `ValueError`
until geometry is attached.

::: voxeldata.accessors
