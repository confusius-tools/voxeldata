"""Persistence tests through the public load_array/save API."""

import json
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed
from dask.callbacks import Callback
from numpy.testing import assert_allclose, assert_array_equal

import voxeldata as vd


@pytest.fixture(params=["zarr2", "zarr3", "netcdf"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, dict[str, Any]]:
    """Return a path and writer options for each supported storage backend.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Selected backend.
    tmp_path : pathlib.Path
        Temporary directory.

    Returns
    -------
    path : pathlib.Path
        Output store path.
    options : dict
        Xarray writer options.
    """
    if request.param == "netcdf":
        return tmp_path / "data.nc", {"engine": "h5netcdf"}
    return tmp_path / "data.zarr", {"zarr_format": int(request.param[-1])}


@pytest.mark.parametrize(
    "fixture",
    [
        "sample_voxeldata_3d",
        "sample_voxeldata_3dt",
        "sample_voxeldata_3d_oblique",
        "sample_voxeldata_irregular",
        "sample_voxeldata_view_time",
        "multigeometry_data",
    ],
)
def test_roundtrip(request, fixture, store):
    """Values, coordinates, geometry dependencies, and frames survive storage."""
    source = request.getfixturevalue(fixture)
    expected = vd.canonicalize_voxeldata(source)
    path, options = store
    vd.save(source, path, **options)
    with vd.load_array(path, chunks=None) as actual:
        vd.validate_voxeldata(actual)
        xr.testing.assert_identical(actual, expected)
        assert actual.vd.geometry_dims == expected.vd.geometry_dims
        assert actual.vd.world_units == expected.vd.world_units
        assert_allclose(actual.vd.voxel_to_world, expected.vd.voxel_to_world)
        assert actual.vd.affines.keys() == expected.vd.affines.keys()
        for name, frame in expected.vd.affines.items():
            assert_allclose(actual.vd.affines[name], frame)
        assert not any(key.startswith("voxel_to_world") for key in actual.attrs)


def test_ordered_dependencies_native_labels_and_world_coordinates(
    multigeometry_data, store
):
    """Non-data-order dependencies and cropped labels retain their physical meaning."""
    source = multigeometry_data.isel(k=slice(1, None), i=[3, 1])
    frames = {
        name: frame.swapaxes(0, 1) if frame.ndim > 2 else frame
        for name, frame in source.vd.affines.items()
    }
    affine = source.vd.voxel_to_world.swapaxes(0, 1)
    source = vd.attach_voxel_to_world_index(
        source, affine, units="um", geometry_dims=("view", "time"), named_affines=frames
    )
    path, options = store
    vd.save(source, path, **options)
    with vd.load_array(path, chunks=None) as actual:
        assert actual.vd.geometry_dims == ("view", "time")
        assert actual.vd.world_units == "um"
        assert_allclose(actual.vd.voxel_to_world, affine)
        for name, frame in frames.items():
            assert_allclose(actual.vd.affines[name], frame)
        assert_array_equal(actual.i, [11, 7])
        selected = actual.isel(view=1, time=2, channel=0)
        grid = np.stack(np.meshgrid(selected.k, selected.j, selected.i, indexing="ij"))
        world = np.einsum("ab,b...->a...", affine[1, 2, :3, :3], grid)
        world += affine[1, 2, :3, 3, None, None, None]
        for axis, name in enumerate(("z", "y", "x")):
            assert_allclose(selected[name], world[axis])
        assert_array_equal(actual.values, vd.canonicalize_voxeldata(source).values)


def test_shared_geometry_multiaxis_timestamps(multiaxis_timestamps, store):
    """Timestamp dependencies do not become geometry dependencies on reload."""
    source = vd.create_voxeldata(
        np.arange(12).reshape(3, 2, 2, 1, 1, 1),
        dims=("time", "view", "channel", "k", "j", "i"),
        time=multiaxis_timestamps,
        spacing=(1.0, 1.0, 1.0),
    ).assign_coords(slice_time=(("time", "k"), [[0.1], [1.1], [2.1]]))
    path, options = store
    vd.save(source, path, **options)
    with vd.load_array(path, chunks=None) as actual:
        xr.testing.assert_identical(actual, vd.canonicalize_voxeldata(source))
        assert actual.vd.geometry_dims == ()
        assert actual.vd.voxel_to_world.shape == (4, 4)
        assert actual.time.attrs == multiaxis_timestamps.attrs


def test_scalar_selected_voxels_are_restored(sample_voxeldata_3d_oblique, store):
    """Scalar-selected native dimensions are saved as valid singleton dimensions."""
    source = sample_voxeldata_3d_oblique.isel(j=1)
    path, options = store
    vd.save(source, path, **options)
    with vd.load_array(path, chunks=None) as actual:
        xr.testing.assert_identical(actual, vd.canonicalize_voxeldata(source))
        assert_array_equal(actual.j, [1])


def test_no_dense_world_coordinates_or_input_mutation(multigeometry_data, store):
    """Only geometry inputs are persisted, without changing the source metadata."""
    source = multigeometry_data
    source.attrs["description"] = "test volume"
    original = source.copy(deep=True)
    original_frames = source.vd.affines
    path, options = store
    vd.save(source, path, **options)
    opener = xr.open_zarr if path.suffix == ".zarr" else xr.open_dataset
    with opener(path, chunks=None) as dataset:
        assert not {"z", "y", "x"} & set(dataset.variables)
        attrs = next(iter(dataset.data_vars.values())).attrs
        decode = json.loads if path.suffix == ".nc" else lambda value: value
        assert decode(attrs["voxel_to_world_geometry_dims"]) == ["time", "view"]
        assert_allclose(decode(attrs["voxel_to_world"]), source.vd.voxel_to_world)
        frames = decode(attrs["voxel_to_world_named_affines"])
        for name, frame in source.vd.affines.items():
            assert_allclose(frames[name], frame)
    xr.testing.assert_identical(source, original)
    for name, frame in original_frames.items():
        assert_allclose(source.vd.affines[name], frame)


def test_unnamed_array(sample_voxeldata_3d, store):
    """Xarray's unnamed DataArray convention is restored on load."""
    source = sample_voxeldata_3d.rename(None)
    path, options = store
    vd.save(source, path, **options)
    with vd.load_array(path) as actual:
        assert actual.name is None
        assert_array_equal(actual, source)


def test_lazy_load_and_deferred_save(sample_voxeldata_3d, store):
    """A deferred save does not evaluate the signal, and chunked loading stays lazy."""
    calls = []

    @delayed
    def signal():
        calls.append(True)
        return np.arange(24, dtype=float).reshape(2, 3, 4)

    source = sample_voxeldata_3d.copy(
        data=da.from_delayed(signal(), shape=(2, 3, 4), dtype=float)
    )
    path, options = store
    task = vd.save(source, path, compute=False, **options)
    assert calls == []
    task.compute(scheduler="synchronous")
    assert calls == [True]
    with vd.load_array(path, chunks={}) as actual:
        assert isinstance(actual.data, da.Array)
        assert_array_equal(actual.compute(), np.arange(24).reshape(2, 3, 4))


@pytest.mark.parametrize("dask_available", [False, True])
def test_default_loading_uses_available_backend(
    sample_voxeldata_3d, store, monkeypatch, dask_available
):
    """Default loading uses storage chunks with Dask, or backend indexing without it."""
    path, options = store
    encoding_key = "chunks" if path.suffix == ".zarr" else "chunksizes"
    vd.save(
        sample_voxeldata_3d,
        path,
        encoding={"intensity": {encoding_key: (1, 3, 4)}},
        **options,
    )
    if not dask_available:
        monkeypatch.setattr("voxeldata.io.find_spec", lambda name: None)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        actual = vd.load_array(path)
    with actual:
        assert tasks == []
        if dask_available:
            assert isinstance(actual.data, da.Array)
            assert actual.chunks == ((1, 1), (3,), (4,))
        else:
            assert actual.chunks is None
        assert_array_equal(actual.isel(k=1), sample_voxeldata_3d.isel(k=1))


@pytest.mark.parametrize("chunks", [None, {"k": 2}])
def test_explicit_loading_chunks_are_preserved(sample_voxeldata_3d, store, chunks):
    """Explicit chunks opt out of Dask or override automatic storage chunking."""
    path, options = store
    encoding_key = "chunks" if path.suffix == ".zarr" else "chunksizes"
    vd.save(
        sample_voxeldata_3d,
        path,
        encoding={"intensity": {encoding_key: (1, 3, 4)}},
        **options,
    )
    with vd.load_array(path, chunks=chunks) as actual:
        if chunks is None:
            assert actual.chunks is None
        else:
            assert isinstance(actual.data, da.Array)
            assert actual.chunks == ((2,), (3,), (4,))
        assert_array_equal(actual, sample_voxeldata_3d)


def test_zarr_numpy_attrs_and_unsupported_objects(sample_voxeldata_3d, tmp_path):
    """NumPy metadata is converted while unsupported objects trigger a warning."""
    source = sample_voxeldata_3d.assign_attrs(
        nested={"array": np.arange(3), "scalar": np.int64(4)}, unsupported=object()
    )
    source.k.attrs["steps"] = np.array([1, 2])
    path = tmp_path / "attrs.zarr"
    with pytest.warns(UserWarning, match="Dropping.*unsupported"):
        vd.save(source, path)
    with vd.load_array(path, chunks=None) as actual:
        assert actual.attrs["nested"] == {"array": [0, 1, 2], "scalar": 4}
        assert actual.k.attrs["steps"] == [1, 2]
        assert "unsupported" not in actual.attrs
    assert isinstance(source.attrs["nested"]["array"], np.ndarray)
    assert isinstance(source.k.attrs["steps"], np.ndarray)
    assert "unsupported" in source.attrs


@pytest.mark.parametrize("suffix", [".nc", ".netcdf", ".source.zarr"])
def test_extensions(sample_voxeldata_3d, tmp_path, suffix):
    """All documented suffixes dispatch to their backend."""
    path = tmp_path / f"data{suffix}"
    vd.save(sample_voxeldata_3d, path)
    with vd.load_array(path, chunks=None) as actual:
        assert_array_equal(actual, sample_voxeldata_3d)


@pytest.mark.parametrize(
    "options",
    [
        {"append_dim": "time"},
        {"region": "auto"},
        {"mode": "a"},
        {"mode": "a-"},
        {"mode": "r+"},
    ],
)
def test_incremental_zarr_writes_rejected(sample_voxeldata_3d, tmp_path, options):
    """Incremental saves cannot silently replace the full store's geometry."""
    path = tmp_path / "data.zarr"
    vd.save(sample_voxeldata_3d, path)
    changed = sample_voxeldata_3d.vd.set_voxel_to_world(np.eye(4))
    with pytest.raises(ValueError, match="Incremental Zarr writes"):
        vd.save(changed, path, **options)
    with vd.load_array(path) as actual:
        xr.testing.assert_identical(actual, sample_voxeldata_3d)


def test_invalid_inputs(plain_volume, tmp_path):
    """Unsupported formats and plain arrays fail before writing."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        vd.save(plain_volume, tmp_path / "data.nii")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        vd.load_array(tmp_path / "data.nii")
    with pytest.raises(ValueError, match="VoxelToWorldIndex"):
        vd.save(plain_volume, tmp_path / "plain.zarr")
    assert not (tmp_path / "plain.zarr").exists()


def test_foreign_store_rejected(plain_volume, store):
    """Unreferenced Xarray data is not silently treated as VoxelData."""
    path, options = store
    if path.suffix == ".zarr":
        plain_volume.to_zarr(path, consolidated=False, **options)
    else:
        plain_volume.to_netcdf(path, **options)
    load_options: dict[str, Any] = (
        {"consolidated": False} if path.suffix == ".zarr" else {}
    )
    with pytest.raises(ValueError, match="Missing attrs"):
        vd.load_array(path, chunks=None, **load_options)


def test_variable_selection(sample_voxeldata_3d, store):
    """Multiple variables require an explicit choice, not an arbitrary first one."""
    path, options = store
    vd.save(sample_voxeldata_3d, path, **options)
    opener = xr.open_zarr if path.suffix == ".zarr" else xr.open_dataset
    with opener(path, chunks=None) as dataset:
        dataset = dataset.load()
    dataset["other"] = dataset["intensity"] + 100
    dataset["other"].attrs = dataset["intensity"].attrs.copy()
    if path.suffix == ".zarr":
        dataset.to_zarr(path, mode="w", consolidated=False, **options)
    else:
        dataset.to_netcdf(path, **options)
    load_options: dict[str, Any] = (
        {"consolidated": False} if path.suffix == ".zarr" else {}
    )
    with pytest.raises(ValueError, match="Specify variable"):
        vd.load_array(path, chunks=None, **load_options)
    with pytest.raises(KeyError):
        vd.load_array(path, variable="missing", chunks=None, **load_options)
    with vd.load_array(path, variable="other", chunks=None, **load_options) as actual:
        assert_array_equal(actual, sample_voxeldata_3d + 100)


@pytest.mark.parametrize(
    ("key", "value", "error", "message"),
    [
        ("geometry_dims", "not JSON", ValueError, "Invalid geometry metadata"),
        ("geometry_dims", '["missing"]', ValueError, "Geometry dimension"),
        ("geometry_dims", '"time"', TypeError, "ordered sequence"),
        ("geometry_dims", "[1]", TypeError, "ordered sequence"),
        ("named_affines", "[]", TypeError, "must be a mapping"),
        ("named_affines", '{"lab": [[1, 2]]}', ValueError, "Named affine.*shape"),
        ("units", 42, TypeError, "must be a string"),
    ],
)
def test_invalid_geometry_metadata(
    sample_voxeldata_3d, tmp_path, key, value, error, message
):
    """Malformed or inconsistent persisted geometry is rejected on loading."""
    path = tmp_path / "broken.nc"
    vd.save(sample_voxeldata_3d, path)
    with xr.open_dataset(path) as dataset:
        dataset = dataset.load()
    dataset["intensity"].attrs[f"voxel_to_world_{key}"] = value
    dataset.to_netcdf(path)
    with pytest.raises(error, match=message):
        vd.load_array(path)
