"""Tests for mask extraction and reconstruction."""

import numpy as np
import pytest
import xarray as xr

from voxeldata import create_voxeldata
from voxeldata.extract import extract_with_labels, extract_with_mask, unmask
from voxeldata.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_units,
    has_voxel_to_world_index,
)


def test_extract_with_labels_on_base_install(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Label reductions work without optional acceleration or lazy backends."""
    labels = make_sample_voxeldata_mask(dtype=int)
    labels.data[0, 1, 2] = 1
    labels.data[1, 2, 3] = 1
    result = extract_with_labels(sample_voxeldata_3dt, labels)
    np.testing.assert_array_equal(result.region, [1])
    np.testing.assert_allclose(result.values[:, 0], [14.5, 38.5, 62.5, 86.5])


def test_extract_with_mask_selects_expected_voxels(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Extraction retains the masked voxel signals and spatial index."""
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    signals = extract_with_mask(sample_voxeldata_3dt, mask)

    assert signals.dims == ("time", "space")
    assert signals.indexes["space"].names == ["k", "j", "i"]
    assert list(signals.indexes["space"]) == [(0, 1, 2), (1, 2, 3)]
    np.testing.assert_array_equal(
        signals.values,
        sample_voxeldata_3dt.values[:, [0, 1], [1, 2], [2, 3]],
    )


def test_extract_with_mask_restores_scalar_voxel_dim(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """`data` with a scalar-reduced voxel dim (e.g. from `.isel(k=0)`) is canonicalized.

    `.isel(k=0)` collapses `k` to a scalar coordinate, dropping the dim itself, which
    only `canonicalize_voxeldata` restores -- exercises that `extract_with_mask`
    canonicalizes `data` itself rather than relying on `ensure_mask`'s internal
    (and discarded) canonicalization of it.
    """
    single_k = sample_voxeldata_3dt.isel(k=0)
    mask = make_sample_voxeldata_mask().isel(k=[0])
    mask.data[0, 1, 2] = True

    signals = extract_with_mask(single_k, mask)

    assert signals.dims == ("time", "space")
    np.testing.assert_array_equal(
        signals.values, sample_voxeldata_3dt.values[:, 0, [1], [2]]
    )


@pytest.mark.parametrize(
    ("dtype", "label_value"),
    [(np.int32, 7), (np.int32, 256), (np.float64, 5.0), (np.float32, 1.0)],
)
def test_extract_with_mask_accepts_binary_numeric_mask(
    sample_voxeldata_3dt: xr.DataArray,
    make_sample_voxeldata_mask,
    dtype: type[np.generic],
    label_value: float,
) -> None:
    """A binary numeric mask (0 and one non-zero value) selects like a boolean mask.

    Covers both a single-label integer mask (produced by
    `AtlasAccessor.get_masks`) and a float binary mask (produced by tools such as
    FSL/NiBabel that don't support a boolean NIfTI dtype), see #382.
    """
    boolean_mask = make_sample_voxeldata_mask()
    boolean_mask.data[0, 1, 2] = True
    numeric_mask = boolean_mask.astype(dtype) * label_value

    expected = extract_with_mask(sample_voxeldata_3dt, boolean_mask)
    result = extract_with_mask(sample_voxeldata_3dt, numeric_mask)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ("dtype", "values", "message"),
    [
        (np.int32, (1, 2), "2 distinct non-zero"),
        (np.float64, (1.0, 2.0), "2 distinct non-zero"),
        (str, ("a",), "binary numeric dtype"),
    ],
)
def test_extract_with_mask_rejects_invalid_mask_values(
    sample_voxeldata_3dt: xr.DataArray,
    make_sample_voxeldata_mask,
    dtype: type[np.generic] | type[float] | type[str],
    values: tuple[float, ...],
    message: str,
) -> None:
    """Only boolean and binary numeric masks are accepted."""
    mask = make_sample_voxeldata_mask(dtype)
    mask.data.flat[: len(values)] = values

    with pytest.raises(TypeError, match=message):
        extract_with_mask(sample_voxeldata_3dt, mask)


def test_extract_with_mask_rejects_misaligned_coordinates(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Extraction rejects a mask whose affine differs from data's.

    The mask's voxel-space (k/j/i) coordinates match data's exactly; only the
    voxel_to_world affine (origin shifted) differs, so this specifically exercises
    that affine mismatches are caught, not just voxel-coordinate mismatches.
    """
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    shifted_affine = get_voxel_to_world_affine(sample_voxeldata_3dt).copy()
    shifted_affine[:3, 3] += 1.0
    mask = attach_voxel_to_world_index(
        mask.drop_vars(("z", "y", "x")),
        shifted_affine,
        units=get_voxel_to_world_units(sample_voxeldata_3dt),
    )

    with pytest.raises(ValueError, match="does not share data's voxel grid"):
        extract_with_mask(sample_voxeldata_3dt, mask)


def test_extract_with_mask_preserves_dask_laziness(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Dask-backed fUSI data remain lazy after extraction."""
    pytest.importorskip("dask.array")
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    chunked = sample_voxeldata_3dt.chunk({"time": 2})

    mask = mask.isel(i=slice(2, None, -1))
    result = extract_with_mask(chunked, mask, join="inner")

    assert hasattr(result.data, "chunks")
    xr.testing.assert_identical(
        result.compute(), extract_with_mask(sample_voxeldata_3dt, mask, join="inner")
    )


def test_unmask_reconstructs_masked_voxeldata_grid(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Extraction followed by reconstruction preserves selected voxel signals."""
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    signals = extract_with_mask(sample_voxeldata_3dt, mask)
    restored = unmask(signals, mask)

    assert restored.dims == sample_voxeldata_3dt.dims
    assert has_voxel_to_world_index(restored)
    np.testing.assert_allclose(
        get_voxel_to_world_affine(restored), get_voxel_to_world_affine(mask)
    )
    assert get_voxel_to_world_units(restored) == get_voxel_to_world_units(mask)
    np.testing.assert_array_equal(
        restored.coords["time"].values, sample_voxeldata_3dt.coords["time"].values
    )
    np.testing.assert_array_equal(
        restored.values[:, mask.values], sample_voxeldata_3dt.values[:, mask.values]
    )
    assert np.all(restored.values[:, ~mask.values] == 0.0)


def test_unmask_rejects_invalid_mask_dtype(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Reconstruction rejects masks that are neither boolean nor binary numeric."""
    mask = make_sample_voxeldata_mask(str)
    mask.data[0, 1, 2] = "a"

    with pytest.raises(TypeError, match="binary numeric dtype"):
        unmask(np.array([1.0]), mask)


def test_unmask_rejects_non_voxeldata_mask() -> None:
    """unmask requires a VoxelData mask; it always reconstructs a VoxelData array."""
    mask = xr.DataArray(
        [[False, True], [True, False]],
        dims=("row", "column"),
        coords={"row": ["a", "b"], "column": [0, 1]},
    )

    with pytest.raises(ValueError, match="missing voxel dimension"):
        unmask(np.array([3.0, 7.0]), mask)


def test_unmask_validates_signal_space_size(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Reconstruction requires one value for every selected voxel."""
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    with pytest.raises(ValueError, match="doesn't match"):
        unmask(np.array([1.0]), mask)


def test_inner_matches_varying_geometry_and_voxels_in_mask_order():
    frames = np.stack([np.eye(4)] * 3)
    frames[:, 0, 3] = [10, 20, 30]
    data = create_voxeldata(
        np.arange(9).reshape(3, 3),
        dims=("pose", "i"),
        i=[10, 11, 12],
        extra_coords={"pose": [1, 2, 3]},
        voxel_to_world=frames,
        geometry_dims=("pose",),
    )
    mask = create_voxeldata(
        [[1, 1, 1], [1, 1, 0], [1, 0, 1]],
        dims=("pose", "i"),
        i=[13, 12, 11],
        extra_coords={"pose": [4, 3, 1]},
        voxel_to_world=frames[[1, 2, 0]],
        geometry_dims=("pose",),
    )
    with pytest.raises(ValueError, match="same labels"):
        extract_with_mask(data, mask)
    signals = data.vd.extract_with_mask(mask, join="inner")
    np.testing.assert_array_equal(signals, [8, 1])
    assert list(signals.indexes["space"]) == [(3, 0, 0, 12), (1, 0, 0, 11)]
    matched_mask = mask.sel(pose=[3, 1], i=[12, 11])
    restored = signals.vd.unmask(matched_mask, fill_value=-1)
    np.testing.assert_array_equal(restored.values.reshape(2, 2), [[8, -1], [-1, 1]])

    with pytest.raises(ValueError, match="voxel grid"):
        extract_with_mask(data, mask.vd.translate([0, 0, 5]), join="inner")
    with pytest.raises(ValueError, match="units differ"):
        extract_with_mask(data, mask.vd.set_world_units("cm"), join="inner")
    with pytest.raises(ValueError, match="no matching labels"):
        extract_with_mask(data, mask.sel(pose=[4]), join="inner")


def test_exact_matches_reversed_native_voxel_labels(sample_voxeldata_3d):
    data = sample_voxeldata_3d
    mask = (data >= 0).isel(i=slice(None, None, -1))
    signals = extract_with_mask(data, mask)
    expected = data.values[..., ::-1]
    np.testing.assert_array_equal(signals, expected.ravel())
    np.testing.assert_array_equal(unmask(signals, mask), expected)


def test_default_join_rejects_strict_subset_of_data_labels():
    data = create_voxeldata([10, 20, 30], dims=("i",), spacing=(1, 1, 1))
    mask = (data > 0).isel(i=[2, 0])
    with pytest.raises(ValueError, match="same labels on 'i'"):
        extract_with_mask(data, mask)
    np.testing.assert_array_equal(extract_with_mask(data, mask, join="inner"), [30, 10])


def test_inner_validates_mask_values_outside_intersection():
    data = create_voxeldata([10, 20], dims=("i",), spacing=(1, 1, 1))
    mask = create_voxeldata([1, 2], dims=("i",), i=[1, 2], spacing=(1, 1, 1))
    with pytest.raises(TypeError, match="distinct non-zero values"):
        extract_with_mask(data, mask, join="inner")


def test_mask_join_rejects_ambiguous_and_unknown_joins():
    data = create_voxeldata(
        [1, 2, 3],
        dims=("channel",),
        extra_coords={"channel": ["a", "a", "b"]},
        spacing=(1, 1, 1),
    )
    mask = (data > 0).isel(channel=[2, 0, 1])
    with pytest.raises(ValueError, match="unique labels"):
        extract_with_mask(data, mask, join="inner")
    with pytest.raises(ValueError, match="join must be"):
        extract_with_mask(data, mask, join="left")  # ty: ignore[invalid-argument-type]
