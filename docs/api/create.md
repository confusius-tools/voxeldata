# Creation

## Timestamp dependencies

Pass a named DataArray to describe timestamps that vary along nonspatial axes:

```python
import numpy as np
import xarray as xr
import voxeldata as vd

timestamps = xr.DataArray(
    np.arange(3.)[:, None, None]
    + np.array([0., .1])[None, :, None]
    + np.array([0., .3])[None, None, :],
    dims=("time", "view", "channel"),
    attrs={"units": "s"},
)
data = vd.create_voxeldata(
    np.zeros((3, 2, 2, 1, 2, 3)),
    dims=("time", "view", "channel", "k", "j", "i"),
    time=timestamps,
    voxel_to_world=np.eye(4),
)
```

Timestamp dependencies need not match `geometry_dims`: this example has varying
acquisition timestamps but shared geometry. The constructor accepts `time` in any
axis position and normalizes the timestamp coordinate to put `time` first. All
other timestamp axes must be nonspatial data dimensions of the matching sizes.
Each series must be numeric, finite, and strictly increasing along `time`.

Multidimensional timestamps are not an ordinary Xarray time index. Select all
additional timestamp dependencies before time-label lookup or an analysis that
requires a shared time vector. Multidimensional timestamps must be DataArrays
with explicit dimension names; unnamed arrays are rejected. `dt` and scalar `t0`
generate a shared time vector.

## Consumer-defined auxiliary coordinates

Additional timing metadata can use ordinary Xarray coordinates:

```python
data = data.assign_coords(
    slice_time=xr.DataArray(
        np.arange(3.)[:, None], dims=("time", "k"), attrs={"units": "s"}
    )
)
```

VoxelData does not reserve or interpret `slice_time`: its dimensions, acquisition
windows, and correction rules belong to the consumer. Generic creation/validation
does not infer acquisition reference or duration metadata. ConfUSIus retains its
own acquisition defaults and slice-timing validation.

::: voxeldata.create
