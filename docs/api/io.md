# Persistence

Install the optional storage backends:

```bash
uv add 'voxeldata[io]'
```

Use `save` and the explicit loaders rather than raw Xarray writers/readers to restore the
`VoxelToWorldIndex` automatically:

```python
import voxeldata as vd

vd.save(data, "volume.zarr")
with vd.load_array("volume.zarr") as restored:
    volume = restored.isel(k=0).compute()

vd.save(data, "volume.nc")  # .netcdf is also supported.
with vd.load_array("volume.nc", chunks=None) as restored:
    volume = restored.compute()
```

Both formats preserve signal values, ordinary coordinates (including native voxel
labels, auxiliary coordinates, and multidimensional timestamps), spatial units,
ordered geometry dependencies, and shared or stacked named frames. Saving restores
scalar-selected voxel dimensions and canonicalizes dimension order as
`(...extra, time, k, j, i)` without mutating the input. Loading returns canonical,
validated VoxelData. Shared geometry stays shared regardless of timestamp or other
nonspatial dimensions.

## Dataset and tree conventions

These are conventions for ordinary Xarray containers, not wrapper classes:

- A **VoxelData dataset** contains zero or more spatial VoxelData variables sharing
  one grid, units, ordered geometry dependencies, and named frames. It may also
  contain arbitrary nonspatial variables, coordinates, and attributes.
- A **VoxelData tree** contains these datasets at spatial leaf nodes. Grouping
  ancestors must not carry native `k/j/i` or derived `z/y/x` coordinates or
  dimensions. Spatial coordinates and geometry dependencies belong locally to
  each spatial leaf, preventing inherited-grid alignment. Different leaves can
  have different grids. Nonspatial nodes and empty grouping nodes are preserved.
- Variables using any spatial dimension must satisfy the array contract; partial
  spatial variables and missing `VoxelToWorldIndex` objects are rejected, not
  treated as diagnostics. Scalar-selected voxel dimensions are restored during
  saving, promoting their shared scalar coordinates to dimension coordinates.
- Selecting any restored spatial variable returns a valid VoxelData DataArray
  directly. No additional index attachment or canonicalization is needed.

Container I/O is **Zarr-only**. Array NetCDF behavior is unchanged. Choose the
return type explicitly; loaders do not guess based on store contents:

| Loader | Return type | Selection |
| --- | --- | --- |
| `load_array` | DataArray | Optional `group` and `variable` |
| `load_dataset` | Dataset | Optional `group`; defaults to root |
| `load_tree` | DataTree | Complete hierarchy |

The former `load` entry point is replaced by `load_array`. A root-only Dataset
store can also be opened with `load_tree`. Variable names, diagnostics, and group
names are consumer-defined; no acquisition or registration conventions are inferred.

```python
import xarray as xr

images = xr.Dataset({"image": image, "valid": validity_mask, "score": score})
results = xr.DataTree.from_dict({
    "/registered": images,
    "/fields/forward": forward_field.to_dataset(name="displacement"),
    "/empty": xr.Dataset(),
})
write = vd.save(results, "results.zarr", compute=False)
write.compute()  # All leaves share one writer and one Dask write graph.

with vd.load_tree("results.zarr") as restored:
    vd.validate_voxeldata(restored["registered"]["image"])

with vd.load_dataset("results.zarr", group="registered") as restored:
    vd.validate_voxeldata(restored["valid"])

with vd.load_array("results.zarr", group="registered", variable="image") as restored:
    volume = restored.isel(time=0).compute()
```

For a single grid, save `images` directly and use `load_dataset` without `group`.
Tree encoding follows Xarray's nested mapping: absolute group path, then variable
name. Writing a tree underneath a `group` prefix is not supported by Xarray.
Xarray 2026.9.0 or newer is required for the shared writer and correct transposition
of lazy world coordinates with multiple geometry dependencies.

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

Zarr uses JSON-native lists and dictionaries. NetCDF stores the matrix, dimension names,
and frame mapping as JSON strings because its attributes cannot represent these nested
structures. The units remain a plain string in both formats. Loading consumes the
reserved attributes and reconstructs the index; frames live on the index, not in
`attrs["affines"]`. Any legacy `attrs["affines"]` is ordinary metadata and is never
interpreted as named frames. Containers reuse the same per-spatial-variable metadata,
with no dense world coordinates. The reserved root attribute `__voxeldata_tree_name__`
preserves a tree's root name and is consumed by `load_tree`; ordinary node names are
Zarr group names.

User attributes follow backend restrictions. For Zarr, NumPy values are converted
to Python containers, and unsupported objects are dropped with a warning (also for
coordinate attributes). For NetCDF, unsupported user attributes raise a backend
error; encode nested dictionaries yourself or use Zarr. No atlas colormaps or other
modality-specific objects are reconstructed.

## Loading, chunking, and writing options

All loaders use storage-aligned Dask chunks (`chunks={}`) by default when Dask is
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

Loading an array from a dataset with multiple variables requires `load_array(...,
variable="name")`. Files without voxel-to-world metadata are rejected; open foreign data
with Xarray and construct VoxelData explicitly.

## Shared execution and limits

Save related outputs in **one** Dataset or DataTree. The container-wide writer
submits all chunk sources and targets together, allowing expensive shared Dask
source tasks to execute once per chunk. Separate deferred saves followed by a
joint `dask.compute` do not provide the same guarantee. Saving does not call
`.persist()` or materialize the complete signal. Both immediate and deferred
container saves use the same shared writer.

`compute=False` leaves signal computation deferred but creates metadata eagerly.
Write failures propagate during computation and may leave a partial store; writes
are not transactional. Default existing-store protection still applies. Append,
region, and update modes remain unsupported; chunked execution of a complete save
is not append support.

Tiny instrumented tests check source execution counts and release of source chunks
under synchronous and two-worker threaded scheduling. They are not full
larger-than-RAM qualification. Actual peak memory depends on chunk sizes, upstream
task working sets, graph dependencies, and scheduler concurrency. Coordinates and
geometry metadata may be read eagerly for validation.

::: voxeldata.io
