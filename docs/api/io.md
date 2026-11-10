# Persistence

Install the optional storage backends:

```bash
uv add 'voxeldata[io]'
```

Use `save` and `load` rather than raw Xarray writers/readers to restore the
`VoxelToWorldIndex` automatically:

```python
import voxeldata as vd

vd.save(data, "volume.zarr")
with vd.load("volume.zarr") as restored:
    volume = restored.isel(k=0).compute()

vd.save(data, "volume.nc")  # .netcdf is also supported.
with vd.load("volume.nc", chunks=None) as restored:
    volume = restored.compute()
```

Both formats preserve signal values, ordinary coordinates (including native voxel
labels, auxiliary coordinates, and multidimensional timestamps), spatial units,
ordered geometry dependencies, and shared or stacked named frames. Saving restores
scalar-selected voxel dimensions and canonicalizes dimension order as
`(...extra, time, k, j, i)` without mutating the input. Loading returns canonical,
validated VoxelData. Shared geometry stays shared regardless of timestamp or other
nonspatial dimensions.

## Storage representation

Derived `z/y/x` coordinates are not written: even oblique world coordinates are
rebuilt lazily from the affine and the stored native voxel labels. These reserved
attributes on the signal variable describe the index:

| Attribute | Content |
| --- | --- |
| `voxel_to_world` | Shared `(4, 4)` matrix or geometry-shaped stack |
| `voxel_to_world_units` | Spatial unit string |
| `voxel_to_world_geometry_dims` | Ordered geometry dimension names; empty for shared geometry |
| `voxel_to_world_named_affines` | Frame name to shared matrix or full geometry-shaped stack |

Zarr uses JSON-native lists and dictionaries, matching the ConfUSIus VoxelData
integration's layout. NetCDF stores the matrix, dimension names, and frame mapping
as JSON strings because its attributes cannot represent these nested structures.
The units remain a plain string in both formats. Loading consumes the reserved
attributes and reconstructs the index; frames live on the index, not in
`attrs["affines"]`. Any legacy `attrs["affines"]` is ordinary metadata and is never
interpreted as named frames.

User attributes follow backend restrictions. For Zarr, NumPy values are converted
to Python containers, and unsupported objects are dropped with a warning (also for
coordinate attributes). For NetCDF, unsupported user attributes raise a backend
error; encode nested dictionaries yourself or use Zarr. No atlas colormaps or other
modality-specific objects are reconstructed.

## Loading, chunking, and writing options

`load` uses storage-aligned Dask chunks (`chunks={}`) by default when Dask is
installed, for example through `voxeldata[lazy]`. Without Dask, it uses backend lazy
indexing, so `voxeldata[io]` still works on its own. Pass `chunks=None` to disable
Dask explicitly, or a chunk mapping to override storage chunking. Backend lazy
indexing defers reads until values are needed, but does not defer arithmetic as
Dask does. Geometry/coordinate validation may read coordinates, but does not
compute signal data. Close loaded arrays when done, or use a context manager as
above.

Additional options pass through to Xarray, including `engine`, `group`, encoding,
and chunking options where supported. `save(..., compute=False)` returns Xarray's
delayed write; compute it to write the signal. Dask is required for deferred writes.

Existing-file behavior follows Xarray: Zarr creates without overwriting by default,
while NetCDF overwrites by default. Set `mode` explicitly as needed. Incremental
Zarr writes (`append_dim`, `region`, or modes `a`, `a-`, `r+`) are rejected because
they would not merge affine stacks and named frames correctly. Save the complete
array instead.

A dataset with multiple signal variables requires `load(..., variable="name")`.
Files without voxel-to-world metadata are rejected; open foreign data with Xarray
and construct VoxelData explicitly. NIfTI and acquisition-specific loaders remain
in ConfUSIus.

::: voxeldata.io
