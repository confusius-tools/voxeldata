"""Generic timestamp dependencies without acquisition-specific slice semantics."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

import voxeldata as vd


def test_dt_uses_shared_time_sampling_and_stride(sample_voxeldata_3dt):
    data = sample_voxeldata_3dt
    assert data.vd.dt == 0.5
    assert data.isel(time=slice(None, None, 2), k=0).vd.dt == 1.0
    assert data.isel(time=[0]).vd.dt is None
    assert data.isel(time=0).vd.dt is None
    assert data.assign_coords(time=[10.0, 10.5, 11.0, 12.0]).vd.dt is None


@pytest.mark.parametrize(
    "time,expected",
    [
        ([0, 1, 2.005, 3], 1.0),
        ([0, 1, 2.03, 3], None),
        ([0, 1e-10, 2e-10, 3e-10], 1e-10),
        ([0, 1e-10, 3e-10, 6e-10], None),
    ],
)
def test_dt_uses_relative_uniformity_tolerance(sample_voxeldata_3dt, time, expected):
    interval = sample_voxeldata_3dt.assign_coords(time=time).vd.dt
    if expected is None:
        assert interval is None
    else:
        assert interval == pytest.approx(expected)


def test_dt_requires_time_and_voxeldata(sample_voxeldata_3d, plain_volume):
    with pytest.raises(ValueError, match="time.*coordinate"):
        _ = sample_voxeldata_3d.vd.dt
    with pytest.raises(ValueError, match="voxel-to-world index"):
        _ = plain_volume.expand_dims(time=[0.0, 1.0]).vd.dt


@pytest.mark.parametrize(
    "time",
    [
        [0, 1, np.nan, 3],
        [0, 1, np.inf, 3],
        [0, 1, 1, 3],
        [3, 2, 1, 0],
        np.array([3, 2, 1, 0], dtype=np.uint64),
        ["a", "b", "c", "d"],
    ],
)
def test_dt_rejects_invalid_timestamps(sample_voxeldata_3dt, time):
    with pytest.raises(ValueError, match="time.*finite.*increasing"):
        _ = sample_voxeldata_3dt.assign_coords(time=time).vd.dt


@pytest.mark.parametrize(
    "time_dims",
    [
        ("time", "view", "channel"),
        ("channel", "time", "view"),
        ("view", "channel", "time"),
    ],
)
def test_named_timestamp_dependencies_are_independent_of_geometry(
    multigeometry_data, multiaxis_timestamps, time_dims
):
    source = multigeometry_data
    timestamps = multiaxis_timestamps.transpose(*time_dims)
    data = vd.create_voxeldata(
        source.data,
        dims=source.dims,
        time=timestamps,
        extra_coords={"view": source.view, "channel": source.channel},
        k=source.k,
        j=source.j,
        i=source.i,
        voxel_to_world=source.vd.voxel_to_world,
        geometry_dims=source.vd.geometry_dims,
        affines=source.vd.affines,
    )
    assert data.dims == source.dims
    assert data.vd.geometry_dims == ("time", "view")
    np.testing.assert_array_equal(data.time, timestamps.transpose("time", ...))
    assert data.time.attrs == multiaxis_timestamps.attrs
    vd.validate_voxeldata(data)
    with pytest.raises(ValueError, match="Timestamp dependencies must be selected"):
        _ = data.vd.dt
    with pytest.raises(ValueError, match="Timestamp dependencies must be selected"):
        _ = data.isel(view=1).vd.dt
    selected = data.isel(view=1, channel=0)
    assert selected.vd.dt == pytest.approx(1.0)
    np.testing.assert_allclose(selected.time, [0.1, 1.1, 2.1])
    np.testing.assert_array_equal(
        selected.sel(time=1.1), source.isel(view=1, channel=0, time=1)
    )
    point = data.isel(time=2, view=1, channel=1, k=1, j=2, i=0)
    expected = source.vd.voxel_to_world[2, 1] @ np.array(
        [source.k[1].item(), source.j[2].item(), source.i[0].item(), 1.0]
    )
    np.testing.assert_allclose(
        [point.z.item(), point.y.item(), point.x.item()], expected[:3]
    )


def test_four_dimensional_timestamps_and_shared_geometry(multiaxis_timestamps):
    timestamps = multiaxis_timestamps.expand_dims(echo=[0, 1]).copy(deep=True)
    timestamps.data[1] += 0.01
    data = vd.create_voxeldata(
        np.arange(24).reshape(2, 3, 2, 2, 1, 1, 1),
        dims=("echo", "time", "channel", "view", "k", "j", "i"),
        time=timestamps,
        voxel_to_world=np.eye(4),
    )
    assert data.time.dims == ("time", "echo", "view", "channel")
    np.testing.assert_array_equal(data.time, timestamps.transpose("time", ...))
    assert data.vd.geometry_dims == ()
    selected = data.isel(echo=1, view=1, channel=1)
    np.testing.assert_allclose(selected.time, [0.41, 1.41, 2.41])
    np.testing.assert_array_equal(
        selected.sel(time=selected.time.values[1]),
        data.isel(time=1, echo=1, view=1, channel=1),
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0])
def test_invalid_timestamp_in_nonfirst_series_is_rejected(
    multiaxis_timestamps, invalid
):
    timestamps = multiaxis_timestamps.copy(deep=True)
    timestamps.data[1, 1, 1] = invalid
    with pytest.raises(ValueError, match="finite|monotonic-increasing"):
        vd.create_voxeldata(
            np.zeros((3, 2, 2, 1, 1, 1)),
            dims=("time", "view", "channel", "k", "j", "i"),
            time=timestamps,
            voxel_to_world=np.eye(4),
        )


@pytest.mark.parametrize(
    "dims, shape, match",
    [
        (("time", "view", "channel"), (3, 2, 3), "dimension 'channel'.*size"),
        (("time", "view", "missing"), (3, 2, 2), "nonspatial data dimensions"),
        (("view", "channel", "echo"), (2, 2, 2), "nonspatial data dimensions"),
        (("time", "view", "k"), (3, 2, 1), "nonspatial data dimensions"),
    ],
)
def test_timestamp_dependencies_must_match_named_nonspatial_data_axes(
    dims, shape, match
):
    with pytest.raises(ValueError, match=match):
        vd.create_voxeldata(
            np.zeros((3, 2, 2, 1, 1, 1)),
            dims=("time", "view", "channel", "k", "j", "i"),
            time=xr.DataArray(np.zeros(shape), dims=dims),
            voxel_to_world=np.eye(4),
        )


def test_auxiliary_timing_coordinates_remain_consumer_defined(sample_voxeldata_3dt):
    source = sample_voxeldata_3dt
    values = source.time.values[:, None] + np.arange(source.sizes["k"])[None, :] * 0.01
    coordinate = xr.DataArray(
        values,
        dims=("time", "k"),
        attrs={"clock": "consumer", "volume_acquisition_reference": "custom"},
    )
    data = source.assign_coords(slice_time=coordinate)
    vd.validate_voxeldata(data)
    selected = data.isel(k=[1])
    vd.validate_voxeldata(selected)
    xr.testing.assert_equal(selected.slice_time, data.slice_time.isel(k=[1]))
    assert selected.slice_time.attrs == coordinate.attrs
    assert "volume_acquisition_duration" not in data.time.attrs


def test_timestamp_metadata_does_not_compute_signal_data(multiaxis_timestamps):
    @delayed
    def fail():
        """Fail if construction or validation evaluates signal samples."""
        raise AssertionError("signal data computed")

    values = da.from_delayed(fail(), shape=(3, 2, 2, 1, 1, 1), dtype=float)
    data = vd.create_voxeldata(
        values,
        dims=("time", "view", "channel", "k", "j", "i"),
        time=multiaxis_timestamps.chunk({"time": 1}),
        voxel_to_world=np.eye(4),
    )
    vd.validate_voxeldata(data)
    assert isinstance(data.data, da.Array)
    assert data.isel(view=1, channel=0).vd.dt == pytest.approx(1.0)
    np.testing.assert_allclose(data.time, multiaxis_timestamps)
