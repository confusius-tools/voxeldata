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

`set_voxel_to_world` is the only method that accepts a plain DataArray without an
existing voxel-to-world index; every other property and method raises `ValueError`
until geometry is attached.

::: voxeldata.accessors
