"""Named geometry dependencies, independent of acquisition timestamp dependencies."""

import itertools
import pickle
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.mark.parametrize("dim", ["pose", "view", "channel", "time"])
def test_arbitrary_geometry_dimension(multigeometry_data, dim):
    """A selected and renamed geometry axis behaves independently of its name."""
    data = multigeometry_data.isel(time=1, channel=0, drop=True).rename(view=dim)
    assert data.vd.geometry_dims == (dim,)
    data.vd.validate()
    selected = data.sel({dim: 41})
    assert selected.vd.geometry_dims == ()
    # Independent world-coordinate oracle: affine times native voxel labels.
    expected = data.vd.index.voxel_to_world[1] @ np.array([4, 6, 9, 1])
    point = selected.isel(k=1, j=1, i=2)
    np.testing.assert_allclose([point[c].item() for c in vd.WORLD_DIMS], expected[:3])
    assert set(data.z.dims) == {dim, "k", "j", "i"}


def test_shared_affine_with_nonspatial_dimensions(plain_volume):
    """Nonspatial dimensions may share one affine without geometry dependencies."""
    raw = plain_volume.expand_dims(channel=[5, 9], pose=[2, 3], view=[11, 41])
    data = raw.vd.set_voxel_to_world(np.eye(4), geometry_dims=())
    assert data.vd.geometry_dims == ()
    assert data.z.dims == ("k", "j", "i")
    assert data.isel(channel=0, pose=1, view=1).vd.geometry_dims == ()


def test_explicit_pose_geometry_and_timestamp_coordinates():
    """Pose geometry and named (time, pose) timestamps remain independent."""
    affines = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    affines[1, 0, 3] = 17
    data = vd.create_voxeldata(
        np.zeros((3, 2, 1, 2, 3)),
        dims=("time", "pose", "k", "j", "i"),
        extra_coords={"pose": [10, 20]},
        time=xr.DataArray([[0, 0.2], [0.5, 0.7], [1, 1.2]], dims=("time", "pose")),
        voxel_to_world=affines,
        geometry_dims=("pose",),
    )
    assert data.vd.geometry_dims == ("pose",)
    assert data.time.dims == ("time", "pose")
    np.testing.assert_allclose(data.time, [[0, 0.2], [0.5, 0.7], [1, 1.2]])
    data.vd.validate()
    canonical = data.vd.canonicalize()
    canonical.vd.validate()
    np.testing.assert_allclose(canonical.time.transpose("time", "pose"), data.time)
    assert data.isel(pose=1).vd.geometry_dims == ()
    np.testing.assert_allclose(data.isel(pose=1).time, [0.2, 0.7, 1.2])


@pytest.mark.parametrize(
    "indexer", [slice(None, None, 2), [2, 0], np.array([False, True, True])]
)
def test_outer_indexing_multiple_geometry_axes(multigeometry_data, indexer):
    """Index each affine axis independently, not with NumPy paired advanced indexing."""
    data = multigeometry_data
    result = data.isel(
        time=indexer, view=[1, 0], k=slice(None, None, -1), i=slice(None, None, 2)
    )
    expected = data.vd.index.voxel_to_world[indexer][:, [1, 0]]
    np.testing.assert_array_equal(result.vd.index.voxel_to_world, expected)
    np.testing.assert_array_equal(
        result.vd.get_affine("lab"), data.vd.get_affine("lab")[indexer][:, [1, 0]]
    )
    for t in range(result.sizes["time"]):
        for v in range(2):
            point = result.isel(time=t, view=v, k=0, j=1, i=1, channel=0)
            world = expected[t, v] @ [4, 6, 9, 1]
            np.testing.assert_allclose(
                [point[d].item() for d in vd.WORLD_DIMS], world[:3]
            )


def test_vectorized_geometry_indexers(multigeometry_data):
    """Xarray pointwise geometry queries select matching affine pairs."""
    data = multigeometry_data
    result = data.isel(
        time=xr.DataArray([2, 0], dims="sample"),
        view=xr.DataArray([1, 0], dims="sample"),
    )
    assert result.vd.geometry_dims == ("sample",)
    np.testing.assert_array_equal(
        result.vd.index.voxel_to_world, data.vd.index.voxel_to_world[[2, 0], [1, 0]]
    )
    np.testing.assert_array_equal(
        result.isel(sample=0).values, data.isel(time=2, view=1).values
    )
    world = data.vd.index.voxel_to_world[2, 1] @ [2, 3, 5, 1]
    point = result.isel(sample=0, k=0, j=0, i=0, channel=0)
    np.testing.assert_allclose([point[d].item() for d in vd.WORLD_DIMS], world[:3])


def test_transpose_rename_copy_and_reconstruction(multigeometry_data):
    """Array order does not change affine-axis order; renaming updates metadata."""
    data = multigeometry_data
    result = (
        data.transpose("view", "channel", "time", "k", "j", "i")
        .rename(view="angle")
        .copy(deep=True)
    )
    assert result.vd.geometry_dims == ("time", "angle")
    result = result.vd.set_world_units("um")
    result = result.isel(j=1).vd.canonicalize()
    assert result.vd.geometry_dims == ("time", "angle")
    assert result.vd.world_units == "um"
    np.testing.assert_array_equal(
        result.vd.get_affine("lab"), data.vd.get_affine("lab")
    )
    for d in vd.WORLD_DIMS:
        np.testing.assert_allclose(
            result[d].transpose("time", "angle", "k", "j", "i"), data.isel(j=[1])[d]
        )


def test_world_selection_requires_all_geometry_dependencies(multigeometry_data):
    """A scalar view alone does not resolve time-dependent geometry."""
    for data in (multigeometry_data, multigeometry_data.isel(view=0)):
        with pytest.raises(ValueError, match="geometry|grid"):
            data.sel(z=10)
    resolved = multigeometry_data.isel(time=1, view=0)
    world = resolved.vd.index.voxel_to_world @ [2, 3, 5, 1]
    query = {
        d: xr.DataArray([value], dims="point")
        for d, value in zip(vd.WORLD_DIMS, world[:3], strict=True)
    }
    np.testing.assert_array_equal(
        resolved.sel(query, method="nearest").values[..., 0],
        resolved.isel(k=0, j=0, i=0).values,
    )


def test_named_composition_with_permuted_dimensions(multigeometry_data, named_affines):
    """Align reference-frame batches by dimension names, not positional lengths."""
    data = multigeometry_data
    other = data.vd.set_voxel_to_world(
        data.vd.index.voxel_to_world.transpose(1, 0, 2, 3),
        geometry_dims=("view", "time"),
        named_affines={"lab": np.broadcast_to(named_affines["atlas"], (2, 3, 4, 4))},
    )
    expected = np.linalg.inv(named_affines["atlas"]) @ data.vd.get_affine("lab")
    np.testing.assert_allclose(data.vd.to(other, "lab"), expected)
    action = np.broadcast_to(named_affines["lab"], (2, 3, 4, 4)).copy()
    action[1, :, 0, 3] += 8
    result = data.vd.apply(action, geometry_dims=("view", "time"))
    ordered = action.transpose(1, 0, 2, 3)
    np.testing.assert_allclose(
        result.vd.index.voxel_to_world, ordered @ data.vd.index.voxel_to_world
    )
    np.testing.assert_allclose(
        result.vd.get_affine("atlas"), named_affines["atlas"] @ np.linalg.inv(ordered)
    )
    rebased = result.vd.apply("lab")
    np.testing.assert_array_equal(rebased.vd.get_affine("lab"), np.eye(4))


def test_composition_adds_named_geometry_dependency(multigeometry_data):
    """A per-channel action broadcasts over unrelated time/view geometry axes."""
    action = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    action[1, 0, 3] = 13
    result = multigeometry_data.vd.apply(action, geometry_dims=("channel",))
    assert result.vd.geometry_dims == ("time", "view", "channel")
    np.testing.assert_allclose(
        result.vd.index.voxel_to_world[:, :, 1],
        action[1] @ multigeometry_data.vd.index.voxel_to_world,
    )
    np.testing.assert_allclose(
        result.vd.get_affine("lab")[:, :, 1],
        multigeometry_data.vd.get_affine("lab") @ np.linalg.inv(action[1]),
    )


def test_bounds_and_reindex_without_dense_world_arrays(multigeometry_data):
    """Bounds and rebasing need only affine stacks and native coordinate vectors."""
    data = multigeometry_data
    with patch.object(
        vd.VoxelToWorldTransform,
        "forward",
        side_effect=AssertionError("dense coordinates evaluated"),
    ):
        bounds = data.vd.bounding_box
        rebased = data.vd.reindex_voxels()
        restored = rebased.vd.reindex_voxels_like(data)
    assert bounds.dims == ("bound", "component")
    corners = np.array(list(itertools.product([1, 5], [1.5, 10.5], [4, 12])))
    all_world = []
    for t in range(3):
        for v in range(2):
            world = (
                data.vd.index.voxel_to_world[t, v]
                @ np.column_stack([corners, np.ones(8)]).T
            )[:3]
            all_world.append(world)
            selected = data.sel(time=data.time.values[t], view=data.view.values[v])
            expected = np.stack([world.min(axis=1), world.max(axis=1)])
            np.testing.assert_allclose(selected.vd.bounding_box, expected)
            np.testing.assert_allclose(
                selected.vd.geometry_center, expected.mean(axis=0)
            )
    world = np.concatenate(all_world, axis=1)
    np.testing.assert_allclose(bounds, np.stack([world.min(axis=1), world.max(axis=1)]))
    np.testing.assert_allclose(
        rebased.vd.index.voxel_to_world[..., :3, 3],
        (data.vd.index.voxel_to_world @ [2, 3, 5, 1])[..., :3],
    )
    np.testing.assert_array_equal(
        restored.vd.index.voxel_to_world, data.vd.index.voxel_to_world
    )
    np.testing.assert_array_equal(
        restored.vd.get_affine("lab"), data.vd.get_affine("lab")
    )


def test_variable_scalings_are_not_silently_taken_from_first_geometry(
    multigeometry_data,
):
    """Bounds/reindex support varying scale; scalar spacing access must be explicit."""
    affines = multigeometry_data.vd.index.voxel_to_world.copy()
    affines[1, 0, :3, :3] *= 2
    data = multigeometry_data.vd.set_voxel_to_world(
        affines, geometry_dims=("time", "view"), named_affines={}
    )
    with pytest.raises(ValueError, match="scalings vary"):
        _ = data.vd.spacing
    for property_name in ("origin", "direction"):
        with pytest.raises(ValueError, match="geometry dimensions"):
            getattr(data.vd, property_name)
    assert data.vd.bounding_box.shape == (2, 3)
    assert data.vd.reindex_voxels().vd.geometry_dims == ("time", "view")


@pytest.mark.parametrize(
    "dims, shape",
    [
        (("unknown",), (2, 4, 4)),
        (("k",), (2, 4, 4)),
        (("view", "view"), (2, 2, 4, 4)),
        (("view",), (3, 4, 4)),
        ((), (2, 4, 4)),
    ],
)
def test_invalid_geometry_dependencies(plain_volume, dims, shape):
    raw = plain_volume.expand_dims(view=[11, 41])
    with pytest.raises(ValueError, match="dimension|shape"):
        raw.vd.set_voxel_to_world(np.broadcast_to(np.eye(4), shape), geometry_dims=dims)


def test_no_inference_from_matching_lengths(plain_volume):
    raw = plain_volume.expand_dims(view=[11, 41], channel=[10, 30])
    with pytest.raises(ValueError, match="explicit geometry_dims"):
        raw.vd.set_voxel_to_world(np.broadcast_to(np.eye(4), (2, 4, 4)))


def test_pickle_roundtrip(multigeometry_data):
    restored = pickle.loads(pickle.dumps(multigeometry_data))
    assert restored.vd.geometry_dims == ("time", "view")
    np.testing.assert_array_equal(
        restored.vd.index.voxel_to_world, multigeometry_data.vd.index.voxel_to_world
    )
    np.testing.assert_array_equal(
        restored.vd.get_affine("lab"), multigeometry_data.vd.get_affine("lab")
    )
    np.testing.assert_allclose(
        restored.isel(time=2, view=1).z, multigeometry_data.isel(time=2, view=1).z
    )


def test_concat_along_second_geometry_axis(multigeometry_data):
    first, second = multigeometry_data.isel(view=[0]), multigeometry_data.isel(view=[1])
    result = xr.concat([first, second], dim="view")
    np.testing.assert_array_equal(
        result.vd.index.voxel_to_world, multigeometry_data.vd.index.voxel_to_world
    )
    np.testing.assert_array_equal(
        result.vd.get_affine("lab"), multigeometry_data.vd.get_affine("lab")
    )


def test_acquisition_timing_independent_of_geometry(multigeometry_data):
    data = multigeometry_data.isel(channel=0, drop=True)
    timestamps = xr.DataArray(
        [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]],
        dims=("time", "view"),
        attrs={
            "units": "s",
            "volume_acquisition_reference": "start",
            "volume_acquisition_duration": 0.1,
        },
    )
    result = vd.create_voxeldata(
        data.data,
        dims=data.dims,
        time=timestamps.transpose("view", "time"),
        extra_coords={"view": [11, 41]},
        voxel_to_world=np.eye(4),
        geometry_dims=(),
    )
    result.vd.validate()
    assert result.vd.geometry_dims == ()
    np.testing.assert_allclose(result.time, timestamps)
    assert result.time.attrs == timestamps.attrs


def test_lazy_geometry_does_not_compute_signal_data():
    da = pytest.importorskip("dask.array")
    delayed = pytest.importorskip("dask").delayed

    @delayed
    def fail():
        raise AssertionError("signal data computed")

    lazy = da.from_delayed(fail(), shape=(1000, 2, 3, 4), dtype=float)
    with patch.object(
        vd.VoxelToWorldTransform,
        "forward",
        side_effect=AssertionError("world coordinates evaluated"),
    ):
        data = vd.create_voxeldata(
            lazy,
            dims=("time", "k", "j", "i"),
            dt=1.0,
            voxel_to_world=np.broadcast_to(np.eye(4), (1000, 4, 4)),
            geometry_dims=("time",),
        )
        data.vd.validate()
        selected = data.isel(
            time=slice(0, 100, 2), i=slice(None, None, 2)
        ).vd.canonicalize()
        assert selected.vd.geometry_dims == ("time",)
        assert isinstance(selected.data, da.Array)
        assert selected.z.shape == (50, 2, 3, 2)
        np.testing.assert_allclose(
            selected.vd.bounding_box, [[-0.5, -0.5, -1], [1.5, 2.5, 3]]
        )
        assert isinstance(selected.vd.reindex_voxels().data, da.Array)
        assert selected.vd.apply(np.eye(4), inplace=True) is selected
        assert isinstance(selected.data, da.Array)


@pytest.mark.parametrize("operation", ["reorder", "missing", "replace"])
def test_stale_geometry_after_unsupported_reindex_is_rejected(
    multigeometry_data, operation
):
    data = multigeometry_data
    if operation == "reorder":
        result = data.reindex(view=[41, 11])
    elif operation == "missing":
        result = data.reindex(view=[11, 99])
    else:
        result = data.assign_coords(view=[10, 20])
    with pytest.raises(ValueError, match="no longer matches the affine index"):
        result.vd.validate()
    with pytest.raises(ValueError, match="Use .isel"):
        _ = result.vd.voxel_to_world


def test_multidimensional_time_lookup_survives_scalar_timestamp_dependency():
    data = vd.create_voxeldata(
        np.zeros((3, 2, 1, 1, 1)),
        dims=("time", "pose", "k", "j", "i"),
        time=xr.DataArray([[0, 0.2], [1, 1.2], [2, 2.2]], dims=("time", "pose")),
        voxel_to_world=np.broadcast_to(np.eye(4), (3, 2, 4, 4)),
        geometry_dims=("time", "pose"),
    )
    selected = data.isel(pose=1, time=slice(1, None))
    assert selected.vd.geometry_dims == ("time",)
    np.testing.assert_allclose(selected.time, [1.2, 2.2])
    selected.vd.validate()


def test_shared_named_frames_stay_shared_after_shared_action(
    multigeometry_data, named_affines
):
    result = multigeometry_data.vd.apply(named_affines["lab"])
    assert result.vd.get_affine("atlas").shape == (4, 4)
    np.testing.assert_allclose(
        result.vd.get_affine("atlas"),
        named_affines["atlas"] @ np.linalg.inv(named_affines["lab"]),
    )


@pytest.mark.parametrize("failure", ["nan", "last_row", "partial_shape"])
def test_multiaxis_named_affine_validation(multigeometry_data, failure):
    frame = np.broadcast_to(np.eye(4), (3, 2, 4, 4)).copy()
    if failure == "nan":
        frame[2, 1, 0, 0] = np.nan
    elif failure == "last_row":
        frame[1, 0, 3, 0] = 1
    else:
        frame = frame[0]
    with pytest.raises(ValueError, match="finite|homogeneous|shape"):
        multigeometry_data.vd.set_affine("invalid", frame)


def test_relative_frame_rejects_mismatched_geometry_labels(multigeometry_data):
    data = multigeometry_data
    other = vd.attach_voxel_to_world_index(
        data.assign_coords(view=[11, 99]),
        data.vd.index.voxel_to_world,
        geometry_dims=data.vd.geometry_dims,
        named_affines=data.vd.affines,
    )
    with pytest.raises(ValueError, match="same geometry coordinates.*view"):
        data.vd.to(other, "lab")


def test_scalar_spatial_selection_retains_full_affines_and_frames(multigeometry_data):
    selected = multigeometry_data.isel(k=1, time=2)
    np.testing.assert_array_equal(
        selected.vd.index.voxel_to_world, multigeometry_data.vd.index.voxel_to_world[2]
    )
    np.testing.assert_array_equal(
        selected.vd.get_affine("lab"), multigeometry_data.vd.get_affine("lab")[2]
    )
    assert selected.vd.fixed_voxel_coords == {"k": 4}
    np.testing.assert_array_equal(selected.vd.canonicalize().k, [4])


@pytest.mark.parametrize("name", ["k", "z"])
def test_spatial_names_cannot_be_reinterpreted_by_rename(multigeometry_data, name):
    with pytest.raises(ValueError, match="coordinate names are fixed"):
        multigeometry_data.rename({name: "other"})


def test_unnamed_multidimensional_time_requires_named_dependencies():
    timestamps = xr.DataArray([[0.0, 0.2], [0.5, 0.7], [1.0, 1.2]])
    for time in (timestamps, timestamps.values):
        with pytest.raises(ValueError, match="named"):
            vd.create_voxeldata(
                np.zeros((3, 2, 1, 1, 1)),
                dims=("time", "view", "k", "j", "i"),
                time=time,
                spacing=(1, 1, 1),
            )


@pytest.mark.parametrize(
    "indexers",
    [{"view": object()}, {"view": np.zeros((2, 2), dtype=int)}, {"j": "bogus"}],
)
def test_unsupported_transform_indexers_return_none(multigeometry_data, indexers):
    transform = vd.VoxelToWorldTransform(
        {d: multigeometry_data[d].values for d in vd.VOXEL_DIMS},
        multigeometry_data.vd.index.voxel_to_world,
        geometry_coords={d: multigeometry_data[d].values for d in ("time", "view")},
    )
    assert transform.isel(indexers) is None


def test_multiaxis_world_and_frame_values_against_input_oracles(
    multigeometry_data, oblique_affine, named_affines
):
    point = multigeometry_data.isel(channel=1, time=2, view=1, k=1, j=2, i=0)
    base_world = (oblique_affine @ [4, 9, 5, 1])[:3]
    expected_world = [-base_world[1] + 20, base_world[0] + 20, base_world[2] + 15]
    np.testing.assert_allclose([point[d].item() for d in vd.WORLD_DIMS], expected_world)
    expected_frame = named_affines["lab"].copy()
    expected_frame[:3, 3] += [6, 7, 33]
    np.testing.assert_array_equal(
        multigeometry_data.isel(time=2, view=1).vd.get_affine("lab"), expected_frame
    )


@pytest.mark.parametrize("dim", ["channel", "pose", "view"])
def test_dimension_names_never_infer_geometry(plain_volume, dim):
    """Affine dependencies are explicit for every nonspatial dimension name."""
    raw = plain_volume.expand_dims({dim: ["left", "right"]})
    affines = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    affines[1, :3, :3] *= 2
    with pytest.raises(ValueError, match="explicit geometry_dims"):
        raw.vd.set_voxel_to_world(affines)
    data = vd.create_voxeldata(
        raw.data,
        dims=raw.dims,
        extra_coords={dim: raw[dim]},
        voxel_to_world=affines,
        geometry_dims=(dim,),
    )
    data.vd.validate()
    point = data.isel({dim: 1, "k": 1, "j": 2, "i": 3})
    np.testing.assert_allclose([point[d].item() for d in vd.WORLD_DIMS], [2, 4, 6])
    shared = vd.create_voxeldata(
        raw.data, dims=raw.dims, extra_coords={dim: raw[dim]}, spacing=(1, 1, 1)
    )
    assert shared.vd.geometry_dims == ()
    np.testing.assert_allclose(shared.isel({dim: 1}).z.isel(j=0, i=0), [0, 1])


def test_time_offsets_require_named_timestamps():
    """Geometry dependencies never imply timestamp dependencies."""
    with pytest.raises(ValueError, match="t0 must be scalar"):
        vd.create_voxeldata(
            np.zeros((3, 2, 1, 1, 1)),
            dims=("time", "view", "k", "j", "i"),
            dt=1.0,
            t0=[0, 0.2],  # ty: ignore[invalid-argument-type]
            spacing=(1, 1, 1),
        )
