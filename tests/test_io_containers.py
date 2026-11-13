"""Container persistence through the public VoxelData I/O API."""

import weakref
from collections import Counter
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed
from dask.callbacks import Callback
from numpy.testing import assert_allclose, assert_array_equal

import voxeldata as vd


@pytest.fixture(params=[2, 3])
def zarr_store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, dict[str, Any]]:
    """Return a temporary Zarr path and format options.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Selected Zarr format.
    tmp_path : pathlib.Path
        Temporary directory.

    Returns
    -------
    path : pathlib.Path
        Store path.
    options : dict
        Writer options.
    """
    return tmp_path / "results.zarr", {"zarr_format": request.param}


def test_explicit_array_loader(sample_voxeldata_3d, zarr_store):
    """The explicit array loader retains array semantics."""
    path, options = zarr_store
    vd.save(sample_voxeldata_3d, path, **options)
    with vd.load_array(path) as actual:
        xr.testing.assert_identical(actual, sample_voxeldata_3d)
    with vd.load_dataset(path) as dataset:
        xr.testing.assert_identical(dataset["intensity"], sample_voxeldata_3d)
    with vd.load_tree(path) as tree:
        xr.testing.assert_identical(tree["intensity"], sample_voxeldata_3d)


@pytest.mark.parametrize("scalar_selection", [False, True])
def test_dataset_roundtrip(multigeometry_data, zarr_store, scalar_selection):
    """A dataset owns directly selectable geometry and preserves diagnostics."""
    image = multigeometry_data.isel(k=slice(1, None), i=[3, 1])
    if scalar_selection:
        image = image.isel(j=1)
    source = xr.Dataset(
        {
            "image": image,
            "valid": image > 20,
            "score": xr.DataArray(
                [1.5, 2.5, 3.5], dims="time", attrs={"units": "a.u."}
            ),
            "iterations": xr.DataArray(np.int16(4)),
        },
        attrs={"description": "shared grid", "nested": {"value": np.int64(5)}},
    )
    source["image"].attrs["description"] = "signal"
    source.coords["quality"] = xr.DataArray(
        np.array([3, 2, 1], dtype="int8"), dims="time"
    )
    original = source.copy(deep=True)
    path, options = zarr_store
    vd.save(source, path, **options)
    xr.testing.assert_identical(source, original)
    with vd.load_dataset(path, chunks=None) as actual:
        assert actual.attrs == {"description": "shared grid", "nested": {"value": 5}}
        for name in ("image", "valid"):
            expected = vd.ensure_voxeldata(source[name])
            vd.validate_voxeldata(actual[name])
            xr.testing.assert_identical(actual[name], expected)
            assert actual[name].dtype == expected.dtype
            assert actual[name].vd.geometry_dims == expected.vd.geometry_dims
            assert actual[name].vd.world_units == expected.vd.world_units
            assert_allclose(actual[name].vd.voxel_to_world, expected.vd.voxel_to_world)
            for frame, matrix in expected.vd.affines.items():
                assert_allclose(actual[name].vd.affines[frame], matrix)
        for name in ("score", "iterations"):
            expected = source[name]
            if scalar_selection:
                # Canonicalizing the shared grid promotes scalar j to a dimension.
                expected = expected.drop_vars("j")
            xr.testing.assert_identical(actual[name], expected)
        assert actual["image"].vd.index is actual["valid"].vd.index
    with vd.load_array(path, variable="valid", chunks=None) as actual:
        xr.testing.assert_identical(actual, vd.ensure_voxeldata(source["valid"]))
    with xr.open_zarr(path, chunks=None) as raw:
        assert not set(vd.WORLD_DIMS) & set(raw.variables)


def test_tree_distinct_grids_and_empty_groups(
    sample_voxeldata_3dt, sample_voxeldata_3d_oblique, named_affines, zarr_store
):
    """Distinct sibling grids never align, broadcast, or inherit spatial indexes."""
    image = sample_voxeldata_3dt
    field = sample_voxeldata_3d_oblique.isel(k=[1], j=[0, 2], i=[1, 3])
    field = vd.attach_voxel_to_world_index(
        field, field.vd.voxel_to_world, units="um", named_affines=named_affines
    ).expand_dims(component=["z", "y", "x"])
    source = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"description": "results"}),
            "/registered": xr.Dataset({"image": image, "valid": image > 30}),
            "/diagnostics": xr.Dataset(
                {"score": ("step", np.array([3, 1], dtype="int16"))}
            ),
            "/fields": xr.Dataset(attrs={"description": "group"}),
            "/fields/forward": field.rename("displacement").to_dataset(),
            "/empty/nested": xr.Dataset(attrs={"description": "empty"}),
        },
        name="registration",
    )
    path, options = zarr_store
    vd.save(source, path, **options)
    with vd.load_tree(path, chunks=None) as actual:
        assert actual.name == source.name
        assert set(actual.groups) == set(source.groups)
        for group in source.groups:
            source_node = source[group]
            actual_node = actual[group]
            assert isinstance(source_node, xr.DataTree)
            assert isinstance(actual_node, xr.DataTree)
            expected = source_node.to_dataset(inherit=False)
            restored = actual_node.to_dataset(inherit=False)
            xr.testing.assert_identical(restored, expected)
        for group, variable in (
            ("registered", "image"),
            ("registered", "valid"),
            ("fields/forward", "displacement"),
        ):
            restored = actual[group][variable]
            expected = source[group][variable]
            assert isinstance(restored, xr.DataArray)
            vd.validate_voxeldata(restored)
            assert restored.vd.world_units == expected.vd.world_units
            assert_allclose(restored.vd.voxel_to_world, expected.vd.voxel_to_world)
            for name, frame in expected.vd.affines.items():
                assert_allclose(restored.vd.affines[name], frame)
        assert not actual.coords
        assert not actual["fields"].coords
    with vd.load_dataset(path, group="registered", chunks=None) as actual:
        xr.testing.assert_identical(actual["image"], image.rename("image"))
    with vd.load_array(
        path, group="fields/forward", variable="displacement", chunks=None
    ) as actual:
        xr.testing.assert_identical(actual, field.rename("displacement"))


@pytest.mark.parametrize("tree", [False, True])
def test_nonspatial_and_empty_containers(tree, zarr_store):
    """Containers need not contain any spatial variables."""
    dataset = xr.Dataset(
        {"score": ("iteration", np.array([8, 5, 2], dtype="int16"))},
        coords={"iteration": [1, 2, 3]},
        attrs={"description": "diagnostics"},
    )
    source = (
        xr.DataTree.from_dict({"/diagnostics": dataset, "/empty": xr.Dataset()})
        if tree
        else dataset
    )
    path, options = zarr_store
    vd.save(source, path, **options)
    if tree:
        with vd.load_tree(path, chunks=None) as actual:
            xr.testing.assert_identical(actual["diagnostics"].to_dataset(), dataset)
            assert "empty" in actual.children
    else:
        with vd.load_dataset(path, chunks=None) as actual:
            xr.testing.assert_identical(actual, dataset)


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_invalid_spatial_variables_rejected(plain_volume, tmp_path, tree, partial):
    """Native spatial dimensions require a valid index, even for partial grids."""
    dataset = (
        plain_volume.isel(j=0, i=0).to_dataset()
        if partial
        else plain_volume.to_dataset()
    )
    source = xr.DataTree.from_dict({"/bad": dataset}) if tree else dataset
    path = tmp_path / "invalid.zarr"
    with pytest.raises(
        ValueError, match="Group.*variable 'intensity'.*VoxelToWorldIndex"
    ):
        vd.save(source, path)
    assert not path.exists()


def test_spatial_ancestors_rejected(sample_voxeldata_3d, tmp_path):
    """Spatial nodes cannot introduce inherited grids into descendants."""
    tree = xr.DataTree.from_dict(
        {"/": sample_voxeldata_3d.to_dataset(), "/child": xr.Dataset()}
    )
    with pytest.raises(ValueError, match="spatial grids must be local to leaf nodes"):
        vd.save(tree, tmp_path / "invalid.zarr")
    with pytest.raises(ValueError, match="node-local"):
        vd.save(xr.DataTree(), tmp_path / "invalid.zarr", write_inherited_coords=True)
    assert not (tmp_path / "invalid.zarr").exists()


@pytest.mark.parametrize(
    "metadata", [None, "affine", "units", "frames", "dependencies"]
)
def test_invalid_stored_dataset_rejected(sample_voxeldata_3dt, tmp_path, metadata):
    """Restoration rejects missing metadata and incompatible per-variable grids."""
    path = tmp_path / "invalid.zarr"
    source = xr.Dataset(
        {"image": sample_voxeldata_3dt, "valid": sample_voxeldata_3dt > 0}
    )
    vd.save(source, path, zarr_format=2)
    with xr.open_zarr(path, chunks=None) as opened:
        raw = opened.load()
    attrs = raw["valid"].attrs
    if metadata is None:
        attrs.pop("voxel_to_world")
    elif metadata == "affine":
        attrs["voxel_to_world"][0][3] += 100
    elif metadata == "units":
        attrs["voxel_to_world_units"] = "um"
    elif metadata == "frames":
        attrs["voxel_to_world_named_affines"] = {"lab": np.eye(4).tolist()}
    else:
        attrs["voxel_to_world"] = np.broadcast_to(
            attrs["voxel_to_world"], (4, 4, 4)
        ).tolist()
        attrs["voxel_to_world_geometry_dims"] = ["time"]
    raw.to_zarr(path, mode="w", zarr_format=2)
    for loader in (vd.load_dataset, vd.load_tree):
        with pytest.raises(
            ValueError, match="Group.*variable 'valid'.*(Missing attrs|share one grid)"
        ):
            loader(path)


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("compute", [False, True])
@pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
def test_shared_chunk_execution_and_streaming(
    sample_voxeldata_3d, zarr_store, tree, compute, scheduler
):
    """One writer shares expensive chunks and releases them during a small stream."""
    calls = Counter()
    live = weakref.WeakValueDictionary()
    peak = []
    count = 12

    @delayed(pure=True)
    def signal(chunk):
        calls[chunk] += 1
        values = np.full((1, 2, 3, 4), chunk, dtype="float32")
        live[chunk] = values
        peak.append(len(live))
        return values

    data = da.concatenate(
        [
            da.from_delayed(signal(chunk), shape=(1, 2, 3, 4), dtype="float32")
            for chunk in range(count)
        ]
    )
    image = sample_voxeldata_3d.expand_dims(time=np.arange(count)).copy(data=data)
    mean = image.mean(("k", "j", "i"))
    if tree:
        source = xr.DataTree.from_dict(
            {
                "/images": xr.Dataset({"image": image + 10}),
                "/masks": xr.Dataset({"valid": image >= 5}),
                "/diagnostics": xr.Dataset({"mean": mean}),
            }
        )
    else:
        source = xr.Dataset({"image": image + 10, "valid": image >= 5, "mean": mean})
    path, options = zarr_store
    with dask.config.set(scheduler=scheduler, num_workers=2):
        write = vd.save(source, path, compute=compute, **options)
        if not compute:
            assert not calls
            assert not peak
            write.compute()
    assert calls == Counter({chunk: 1 for chunk in range(count)})
    # This controlled graph must release source chunks, not retain the recording.
    assert 0 < max(peak) < count
    assert not live
    with vd.load_array(
        path, group="images" if tree else None, variable="image", chunks=None
    ) as actual:
        expected = np.broadcast_to(
            np.arange(count)[:, None, None, None] + 10, image.shape
        )
        assert_array_equal(actual, expected)
    with vd.load_array(
        path, group="masks" if tree else None, variable="valid", chunks=None
    ) as actual:
        assert actual.dtype == bool
        expected = np.broadcast_to(
            (np.arange(count) >= 5)[:, None, None, None], image.shape
        )
        assert_array_equal(actual, expected)
    with vd.load_dataset(
        path, group="diagnostics" if tree else None, chunks=None
    ) as actual:
        assert_array_equal(actual["mean"], np.arange(count))
    tasks = []
    loader = vd.load_tree if tree else vd.load_dataset
    with Callback(pretask=lambda *args: tasks.append(args)):
        actual = loader(path)
    with actual:
        assert not tasks
        restored_image = actual["images"]["image"] if tree else actual["image"]
        assert isinstance(restored_image.data, da.Array)
        assert restored_image.chunks is not None
        assert restored_image.chunks[0] == (1,) * count


@pytest.mark.parametrize("tree", [False, True])
def test_store_protection_and_delayed_errors(sample_voxeldata_3d, zarr_store, tree):
    """Existing stores stay protected, and deferred signal failures propagate."""
    path, options = zarr_store
    source = sample_voxeldata_3d.to_dataset()
    if tree:
        source = xr.DataTree.from_dict({"/images": source})
    vd.save(source, path, **options)
    with pytest.raises(FileExistsError):
        vd.save(source, path, compute=False, **options)
    with vd.load_array(path, group="images" if tree else None) as actual:
        xr.testing.assert_identical(actual, sample_voxeldata_3d)
    for kwargs in ({"mode": "a"}, {"append_dim": "time"}, {"region": "auto"}):
        with pytest.raises(ValueError, match="Incremental Zarr writes"):
            vd.save(source, path, **kwargs)

    @delayed
    def broken():
        raise RuntimeError("synthetic source failure")

    image = sample_voxeldata_3d.copy(
        data=da.from_delayed(broken(), shape=(2, 3, 4), dtype=float)
    )
    source = image.to_dataset()
    if tree:
        source = xr.DataTree.from_dict({"/images": source})
    write = vd.save(source, path, compute=False, mode="w", **options)
    with pytest.raises(RuntimeError, match="synthetic source failure"):
        write.compute(scheduler="synchronous")


def test_tree_root_name_metadata(tmp_path):
    """Reserved root-name metadata cannot overwrite user attributes silently."""
    path = tmp_path / "results.zarr"
    tree = xr.DataTree(xr.Dataset(attrs={"__voxeldata_tree_name__": "user value"}))
    with pytest.raises(ValueError, match="Root attribute.*reserved"):
        vd.save(tree, path)
    assert not path.exists()
    xr.Dataset(attrs={"__voxeldata_tree_name__": 42}).to_zarr(path, zarr_format=2)
    with pytest.raises(TypeError, match="Persisted tree name must be a string"):
        vd.load_tree(path)


def test_container_netcdf_and_unsupported_inputs(sample_voxeldata_3d, tmp_path):
    """Container support does not silently extend NetCDF or unrelated types."""
    for source in (sample_voxeldata_3d.to_dataset(), xr.DataTree()):
        with pytest.raises(TypeError, match="Zarr only"):
            vd.save(source, tmp_path / "results.nc")
    for loader in (vd.load_dataset, vd.load_tree):
        with pytest.raises(ValueError, match="Zarr only"):
            loader(tmp_path / "results.nc")
    with pytest.raises(TypeError, match="DataArray, Dataset, or DataTree"):
        vd.save(np.zeros(3), tmp_path / "results.zarr")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_container_resources_close(plain_volume, tmp_path, monkeypatch, tree, valid):
    """Closing restored containers and failed restoration release the source."""
    dataset = xr.Dataset({"score": 42}) if valid else plain_volume.to_dataset()
    source = xr.DataTree.from_dict({"/data": dataset}) if tree else dataset
    closed = []
    source.set_close(lambda: closed.append(True))
    if tree:
        monkeypatch.setattr(xr, "open_datatree", lambda *args, **kwargs: source)
        loader = vd.load_tree
    else:
        monkeypatch.setattr(xr, "open_zarr", lambda *args, **kwargs: source)
        loader = vd.load_dataset
    path = tmp_path / "results.zarr"
    if valid:
        restored = loader(path)
        assert not closed
        restored.close()
    else:
        with pytest.raises(ValueError, match="Missing attrs"):
            loader(path)
    assert closed == [True]


@pytest.mark.parametrize("tree", [False, True])
def test_container_zarr_attrs(sample_voxeldata_3d, zarr_store, tree):
    """Dataset, diagnostic, and coordinate attributes follow existing Zarr rules."""
    dataset = xr.Dataset(
        {
            "image": sample_voxeldata_3d,
            "score": xr.DataArray(42, attrs={"values": np.arange(2)}),
        },
        attrs={"unsupported": object(), "values": np.arange(3)},
    )
    dataset.k.attrs["steps"] = np.array([1, 2])
    source = xr.DataTree.from_dict({"/data": dataset}) if tree else dataset
    path, options = zarr_store
    with pytest.warns(UserWarning, match="Dropping.*unsupported"):
        vd.save(source, path, **options)
    with vd.load_dataset(path, group="data" if tree else None, chunks=None) as actual:
        assert actual.attrs == {"values": [0, 1, 2]}
        assert actual["score"].attrs == {"values": [0, 1]}
        assert actual.k.attrs["steps"] == [1, 2]
    assert isinstance(dataset.attrs["values"], np.ndarray)
    assert "unsupported" in dataset.attrs
    assert isinstance(dataset["score"].attrs["values"], np.ndarray)
    assert isinstance(dataset.k.attrs["steps"], np.ndarray)
