"""Model-level extraction regressions independent of modality and acquisition timing."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

import voxeldata as vd


@pytest.mark.parametrize(
    "reduction", ["mean", "sum", "median", "min", "max", "var", "std"]
)
def test_chunked_label_reductions_against_numpy(
    sample_voxeldata_3dt, reduction, label_backend
):
    data = sample_voxeldata_3dt.chunk({"time": 2, "k": 1, "j": 2, "i": 2})
    values = np.zeros(data.shape[-3:], dtype=int)
    values[0, :, 1:] = -3
    values[1, :, :3] = 7
    labels = vd.create_voxeldata(
        values, dims=vd.VOXEL_DIMS, voxel_to_world=data.vd.voxel_to_world
    )
    if reduction == "median" and label_backend:
        data = data.chunk({d: -1 for d in vd.VOXEL_DIMS})
    result = vd.extract_with_labels(data, labels, reduction)
    assert isinstance(result.data, da.Array)
    np.testing.assert_array_equal(result.region, [-3, 7])
    for region in (-3, 7):
        expected = getattr(np, reduction)(
            sample_voxeldata_3dt.values[:, values == region], axis=-1
        )
        np.testing.assert_allclose(result.sel(region=region).compute(), expected)


@pytest.mark.parametrize(
    "reduction", ["mean", "sum", "median", "min", "max", "var", "std"]
)
def test_label_reductions_skip_missing_samples(
    sample_voxeldata_3dt, reduction, label_backend
):
    data = sample_voxeldata_3dt.copy(deep=True)
    data.data[0, 0, 0, 0] = np.nan
    membership = np.broadcast_to(np.array([1, 2])[:, None, None], data.shape[-3:])
    labels = data.isel(time=0, drop=True).copy(data=membership)
    result = vd.extract_with_labels(data, labels, reduction)
    for region in (1, 2):
        expected = getattr(np, f"nan{reduction}")(
            data.values[:, membership == region], axis=-1
        )
        np.testing.assert_allclose(result.sel(region=region), expected)


@pytest.mark.parametrize("overlap", [False, True])
def test_negative_stacked_ids_and_repeated_ids_are_layer_identifiers(
    sample_voxeldata_3dt, overlap, label_backend
):
    values = np.zeros((2, *sample_voxeldata_3dt.shape[-3:]), dtype=int)
    values[0, 0] = -7
    values[1, 0 if overlap else 1] = -7
    raw = xr.DataArray(
        values,
        dims=("mask", *vd.VOXEL_DIMS),
        coords={
            "mask": ["left", "right"],
            **{d: sample_voxeldata_3dt[d] for d in vd.VOXEL_DIMS},
        },
    )
    labels = raw.vd.set_voxel_to_world(sample_voxeldata_3dt.vd.voxel_to_world)
    result = vd.extract_with_labels(sample_voxeldata_3dt, labels)
    np.testing.assert_array_equal(result.region, ["left", "right"])
    for index, name in enumerate(("left", "right")):
        expected = sample_voxeldata_3dt.values[:, values[index] != 0].mean(axis=-1)
        np.testing.assert_allclose(result.sel(region=name), expected)


def test_shared_mask_and_labels_preserve_geometry_axes(
    multigeometry_data, label_backend
):
    data = multigeometry_data
    affine = data.vd.index.voxel_to_world[0, 0]
    matrices = np.broadcast_to(affine, data.vd.index.voxel_to_world.shape)
    data = data.vd.set_voxel_to_world(
        matrices, geometry_dims=data.vd.geometry_dims, named_affines=data.vd.affines
    )
    selected = np.zeros(data.shape[-3:], dtype=bool)
    selected[0, 0, 0] = selected[1, 2, 3] = True
    raw = xr.DataArray(
        selected, dims=vd.VOXEL_DIMS, coords={d: data[d] for d in vd.VOXEL_DIMS}
    )
    mask = raw.vd.set_voxel_to_world(affine)
    signals = vd.extract_with_mask(data, mask)
    np.testing.assert_array_equal(
        signals.transpose("channel", "time", "view", "space"),
        data.values[..., selected],
    )
    assert list(signals.indexes["space"]) == [(2, 3, 5), (4, 9, 11)]
    np.testing.assert_array_equal(
        signals.sel(view=41, channel=30).transpose("time", "space"),
        data.sel(view=41, channel=30).values[..., selected],
    )
    labels = mask.astype(int)
    regions = vd.extract_with_labels(data, labels)
    np.testing.assert_allclose(
        regions.transpose("channel", "time", "view", "region").values[..., 0],
        data.values[..., selected].mean(axis=-1),
    )


def test_mask_includes_geometry_axes_and_restores_frame_stacks(multigeometry_data):
    data = multigeometry_data
    mask = data.isel(channel=0, drop=True).copy(
        data=np.zeros(data.shape[1:], dtype=bool)
    )
    mask.data[..., 0, 0, 0] = True
    signals = vd.extract_with_mask(data, mask)
    canonical_mask = mask.vd.canonicalize()
    ordered = data.transpose("channel", *canonical_mask.dims)
    np.testing.assert_array_equal(signals, ordered.values[:, canonical_mask.values])
    restored = vd.unmask(signals, mask)
    restored.vd.validate()
    np.testing.assert_array_equal(
        restored.transpose(*data.dims), np.where(mask.values, data.values, 0)
    )
    assert restored.vd.geometry_dims == data.vd.geometry_dims
    np.testing.assert_array_equal(restored.vd.voxel_to_world, data.vd.voxel_to_world)
    for name, frame in data.vd.affines.items():
        np.testing.assert_array_equal(restored.vd.affines[name], frame)


def test_shared_mask_rejects_varying_geometry(multigeometry_data):
    data = multigeometry_data
    mask = xr.DataArray(
        np.ones(data.shape[-3:], dtype=bool),
        dims=vd.VOXEL_DIMS,
        coords={d: data[d] for d in vd.VOXEL_DIMS},
    ).vd.set_voxel_to_world(data.vd.index.voxel_to_world[0, 0])
    for operation, spatial in [
        (vd.extract_with_mask, mask),
        (vd.extract_with_labels, mask.astype(int)),
    ]:
        with pytest.raises(ValueError, match="geometry varies along"):
            operation(data, spatial)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_numeric_mask_rejected(
    make_sample_voxeldata_mask, sample_voxeldata_3dt, invalid
):
    mask = make_sample_voxeldata_mask(float)
    mask.data.flat[0] = invalid
    with pytest.raises(TypeError, match="finite binary"):
        vd.extract_with_mask(sample_voxeldata_3dt, mask)
    with pytest.raises(TypeError, match="finite binary"):
        vd.unmask(np.array([1.0]), mask)


def test_unmask_preserves_native_labels_named_frames_and_timestamps(
    multigeometry_data, named_affines
):
    data = multigeometry_data.isel(view=0, channel=0, time=0, drop=True)
    mask = xr.DataArray(
        np.ones(data.shape, dtype=bool),
        dims=vd.VOXEL_DIMS,
        coords={d: data[d] for d in vd.VOXEL_DIMS},
    ).vd.set_voxel_to_world(data.vd.voxel_to_world, named_affines=named_affines)
    signals = xr.DataArray(
        np.arange(2 * mask.size).reshape(2, mask.size),
        dims=("time", "space"),
        coords={
            "time": xr.DataArray(
                [1.0, 2.0],
                dims="time",
                attrs={"units": "s", "volume_acquisition_duration": 0.3},
            )
        },
    )
    restored = vd.unmask(signals, mask, attrs={"units": "a.u."})
    restored.vd.validate()
    for dim in vd.VOXEL_DIMS:
        np.testing.assert_array_equal(restored[dim], mask[dim])
    for name, matrix in named_affines.items():
        np.testing.assert_array_equal(restored.vd.affines[name], matrix)
    np.testing.assert_array_equal(restored.values.reshape(2, -1), signals)
    assert restored.time.attrs == signals.time.attrs
    assert restored.attrs == {"units": "a.u."}


def test_extraction_does_not_compute_signal_data(
    make_sample_voxeldata_mask, sample_voxeldata_3dt, label_backend
):
    @delayed
    def fail():
        """Fail if signal samples are evaluated while constructing an extraction."""
        raise AssertionError("signal data computed")

    data = sample_voxeldata_3dt.copy(
        data=da.from_delayed(fail(), shape=sample_voxeldata_3dt.shape, dtype=float)
    )
    mask = make_sample_voxeldata_mask()
    mask.data[0, 0, 0] = True
    assert isinstance(vd.extract_with_mask(data, mask).data, da.Array)
    assert isinstance(vd.extract_with_labels(data, mask.astype(int)).data, da.Array)


def test_zero_selected_voxels_reconstruct_fill_value(
    make_sample_voxeldata_mask, sample_voxeldata_3dt
):
    mask = make_sample_voxeldata_mask()
    signals = vd.extract_with_mask(sample_voxeldata_3dt, mask)
    assert signals.sizes["space"] == 0
    restored = vd.unmask(signals, mask, fill_value=-1)
    np.testing.assert_array_equal(restored, np.full(sample_voxeldata_3dt.shape, -1.0))


def test_all_background_labels_rejected(
    sample_voxeldata_3dt, make_sample_voxeldata_mask
):
    with pytest.raises(ValueError, match="no regions to extract"):
        vd.extract_with_labels(sample_voxeldata_3dt, make_sample_voxeldata_mask(int))


def test_scalar_signal_rejected(make_sample_voxeldata_mask):
    with pytest.raises(ValueError, match="sample dimension"):
        vd.unmask(np.array(1.0), make_sample_voxeldata_mask())


def test_spatial_units_checked(sample_voxeldata_3dt, make_sample_voxeldata_mask):
    mask = make_sample_voxeldata_mask().vd.set_world_units("cm")
    with pytest.raises(ValueError, match="spatial units differ"):
        vd.extract_with_mask(sample_voxeldata_3dt, mask)
