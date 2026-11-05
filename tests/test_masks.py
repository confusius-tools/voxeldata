"""Generic mask and label validation without acquisition-specific metadata."""

import numpy as np
import pytest

import voxeldata as vd


def test_ensure_mask_coercion_is_optional(
    sample_voxeldata_3dt, make_sample_voxeldata_mask
):
    mask = make_sample_voxeldata_mask(np.int32)
    mask.data[0, 1, 2] = 256
    result = vd.ensure_mask(mask, sample_voxeldata_3dt)
    assert result.dtype == bool
    assert result.values.sum() == 1
    assert result.values[0, 1, 2]
    uncoerced = vd.ensure_mask(mask, sample_voxeldata_3dt, coerce_bool=False)
    assert uncoerced.dtype == np.int32
    assert uncoerced.values[0, 1, 2] == 256
    assert mask.dtype == np.int32


@pytest.mark.parametrize("operation", [vd.validate_mask, vd.validate_labels])
def test_spatial_validator_type_error(operation, sample_voxeldata_3d):
    with pytest.raises(TypeError, match="xarray.DataArray"):
        operation(np.zeros((2, 3, 4)), sample_voxeldata_3d)


def test_exact_dimensions_ignore_only_time(
    sample_voxeldata_3dt, make_sample_voxeldata_mask
):
    mask = make_sample_voxeldata_mask()
    vd.validate_mask(mask, sample_voxeldata_3dt, require_exact_dims=True)
    expanded = sample_voxeldata_3dt.expand_dims(channel=[1, 2])
    with pytest.raises(ValueError, match="dimensions must match all non-time"):
        vd.validate_mask(mask, expanded, require_exact_dims=True)


def test_ensure_labels_restores_voxel_axis(
    sample_voxeldata_3dt, make_sample_voxeldata_mask
):
    labels = make_sample_voxeldata_mask(int)
    labels.data[0, 0, 0] = -5
    result = vd.ensure_labels(labels.isel(k=0), sample_voxeldata_3dt.isel(k=0))
    assert result.dims == vd.VOXEL_DIMS
    assert result.sizes["k"] == 1
    assert result.values[0, 0, 0] == -5
    vd.validate_labels(result, sample_voxeldata_3dt.isel(k=[0]))


@pytest.mark.parametrize("dtype", [float, bool, str])
def test_labels_require_integer_dtype(
    sample_voxeldata_3d, make_sample_voxeldata_mask, dtype
):
    with pytest.raises(TypeError, match="integer dtype"):
        vd.validate_labels(make_sample_voxeldata_mask(dtype), sample_voxeldata_3d)


def test_alignment_rejects_same_shape_with_different_voxel_labels(
    sample_voxeldata_3d, make_sample_voxeldata_mask
):
    raw = (
        make_sample_voxeldata_mask()
        .drop_vars(list(vd.WORLD_DIMS))
        .assign_coords(i=[1, 2, 3, 4])
    )
    shifted = raw.vd.set_voxel_to_world(sample_voxeldata_3d.vd.voxel_to_world)
    with pytest.raises(ValueError, match="does not share data's voxel grid"):
        vd.check_spatial_alignment(shifted, sample_voxeldata_3d, "mask")


def test_alignment_handles_transposed_native_axes(
    sample_voxeldata_3d, make_sample_voxeldata_mask
):
    mask = make_sample_voxeldata_mask().transpose("i", "k", "j")
    vd.check_spatial_alignment(mask, sample_voxeldata_3d, "mask")


@pytest.mark.parametrize("operation", [vd.extract_with_mask, vd.unmask])
def test_extraction_and_reconstruction_reject_nonspatial_masks(
    operation, sample_voxeldata_3dt, make_sample_voxeldata_mask
):
    mask = make_sample_voxeldata_mask().expand_dims(channel=[1, 2])
    if operation is vd.unmask:
        args = (np.array([]), mask)
    else:
        args = (sample_voxeldata_3dt, mask)
    with pytest.raises(ValueError, match="spatial-only mask"):
        operation(*args)


def test_labels_reject_uninterpreted_nonspatial_axis(
    sample_voxeldata_3dt, make_sample_voxeldata_mask
):
    labels = make_sample_voxeldata_mask(int).expand_dims(layer=[1, 2])
    with pytest.raises(ValueError, match="optionally preceded by a mask dimension"):
        vd.extract_with_labels(sample_voxeldata_3dt, labels)


def test_no_timing_defaults_added_to_metadata(
    sample_voxeldata_3dt, make_sample_voxeldata_mask
):
    data = sample_voxeldata_3dt
    mask = make_sample_voxeldata_mask()
    mask.data[0, 0, 0] = True
    signals = vd.extract_with_mask(data, mask)
    assert signals.time.attrs == {"units": "s"}
    assert "volume_acquisition_duration" not in signals.time.attrs
    assert "volume_acquisition_reference" not in signals.time.attrs
