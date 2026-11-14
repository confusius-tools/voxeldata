"""Public extraction accessor contracts and independent numerical oracles."""

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


def test_mask_accessor_selects_samples_and_reconstructs_full_grid(
    sample_voxeldata_3dt: xr.DataArray,
) -> None:
    data = sample_voxeldata_3dt
    mask = data.isel(time=0, drop=True).copy(data=np.zeros(data.shape[-3:], dtype=bool))
    mask.data.flat[[0, 5, 23]] = True
    mask = mask.vd.set_affine("lab", np.diag([2.0, 3.0, 4.0, 1.0]))
    signals = data.vd.extract_with_mask(mask)
    assert signals.dims == ("time", "space")
    np.testing.assert_array_equal(signals, data.values.reshape(4, -1)[:, [0, 5, 23]])
    assert signals.time.attrs == data.time.attrs
    assert not vd.has_voxel_to_world_index(signals)
    restored = signals.vd.unmask(mask, fill_value=-1.0)
    expected = np.full(data.shape, -1.0)
    expected.reshape(4, -1)[:, [0, 5, 23]] = signals.values
    np.testing.assert_array_equal(restored, expected)
    np.testing.assert_array_equal(restored.vd.voxel_to_world, mask.vd.voxel_to_world)
    np.testing.assert_array_equal(restored.vd.affines["lab"], mask.vd.affines["lab"])
    assert restored.time.attrs == data.time.attrs


@pytest.mark.parametrize(
    "reduction", ["mean", "sum", "median", "min", "max", "var", "std"]
)
def test_labels_accessor_matches_numpy(
    sample_voxeldata_3dt: xr.DataArray, label_backend: bool, reduction: str
) -> None:
    data = sample_voxeldata_3dt
    membership = np.zeros(data.shape[-3:], dtype=int)
    membership.flat[[0, 5]] = 2
    membership.flat[[7, 23]] = 7
    labels = data.isel(time=0, drop=True).copy(data=membership)
    result = data.vd.extract_with_labels(labels, reduction=reduction)
    expected = np.stack(
        [
            getattr(np, reduction)(data.values[:, membership == label], axis=-1)
            for label in (2, 7)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(result.region, [2, 7])
    assert result.time.attrs == data.time.attrs


def test_label_accessor_defaults_to_mean(sample_voxeldata_3d: xr.DataArray) -> None:
    labels = sample_voxeldata_3d.copy(
        data=np.ones(sample_voxeldata_3d.shape, dtype=int)
    )
    result = sample_voxeldata_3d.vd.extract_with_labels(labels)
    np.testing.assert_allclose(result, [11.5])


def test_extraction_accessor_does_not_compute_lazy_signal(
    sample_voxeldata_3dt: xr.DataArray,
) -> None:
    from dask.callbacks import Callback

    data = sample_voxeldata_3dt.chunk({"time": 2})
    mask = sample_voxeldata_3dt.isel(time=0, drop=True).copy(
        data=np.ones(data.shape[-3:], dtype=bool)
    )
    with Callback(pretask=lambda *args: pytest.fail("Signal data computed")):
        signals = data.vd.extract_with_mask(mask)
    np.testing.assert_array_equal(signals.compute(), data.values.reshape(4, -1))


def test_extraction_accessor_rejects_invalid_inputs(
    sample_voxeldata_3d: xr.DataArray, plain_volume: xr.DataArray
) -> None:
    mask = sample_voxeldata_3d.copy(data=np.ones(sample_voxeldata_3d.shape, dtype=bool))
    with pytest.raises(ValueError, match="VoxelToWorldIndex"):
        plain_volume.vd.extract_with_mask(mask)
    with pytest.raises(TypeError, match="integer"):
        sample_voxeldata_3d.vd.extract_with_labels(mask)
    with pytest.raises(ValueError, match="number of masked voxels"):
        xr.DataArray([1.0], dims="space").vd.unmask(mask)
