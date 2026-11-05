"""Reconstruction oracles for NumPy and DataArray sample tables."""

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"new_dims": ["component", "sample"]},
        {
            "new_dims": ["component", "sample"],
            "new_dims_coords": {"component": np.array([10, 20])},
        },
        {
            "new_dims": ["component", "sample"],
            "new_dims_coords": {
                "component": np.array([10, 20]),
                "sample": np.array([1, 3, 5]),
            },
        },
    ],
)
def test_numpy_reconstruction_leading_dimensions_and_coordinates(
    make_sample_voxeldata_mask, options
):
    mask = make_sample_voxeldata_mask()
    mask.data.flat[[0, 23]] = True
    signals = np.arange(12).reshape(2, 3, 2)
    result = vd.unmask(
        signals, mask, fill_value=-5, attrs={"kind": "components"}, **options
    )
    expected = np.full((2, 3, 2, 3, 4), -5)
    expected[..., 0, 0, 0] = signals[..., 0]
    expected[..., 1, 2, 3] = signals[..., 1]
    np.testing.assert_array_equal(result, expected)
    names = options.get("new_dims", ["dim_0", "dim_1"])
    assert result.dims == (*names, *vd.VOXEL_DIMS)
    for dim, length in zip(names, (2, 3), strict=True):
        values = options.get("new_dims_coords", {}).get(dim, np.arange(length))
        np.testing.assert_array_equal(result[dim], values)
    assert result.attrs == {"kind": "components"}
    result.vd.validate()


def test_numpy_reconstruction_without_leading_dimensions(make_sample_voxeldata_mask):
    mask = make_sample_voxeldata_mask()
    mask.data.flat[[0, 23]] = True
    result = vd.unmask(np.array([3.0, 7.0]), mask, fill_value=np.nan)
    expected = np.full(mask.shape, np.nan)
    expected.flat[[0, 23]] = [3.0, 7.0]
    np.testing.assert_allclose(result, expected)
    assert result.dims == vd.VOXEL_DIMS


@pytest.mark.parametrize(
    "options, message",
    [
        ({"new_dims": []}, "Length of new_dims"),
        (
            {"new_dims": ["component"], "new_dims_coords": {"component": np.arange(3)}},
            "has length",
        ),
    ],
)
def test_inconsistent_leading_coordinates_rejected(
    make_sample_voxeldata_mask, options, message
):
    mask = make_sample_voxeldata_mask()
    mask.data.flat[0] = True
    with pytest.raises(ValueError, match=message):
        vd.unmask(np.ones((2, 1)), mask, **options)


@pytest.mark.parametrize(
    "signals, message",
    [
        (xr.DataArray(np.ones(2), dims="voxel"), "must have 'space' dimension"),
        (xr.DataArray(np.ones((1, 2)), dims=("space", "time")), "must be the last"),
        (xr.DataArray(np.ones(2), dims="space"), "doesn't match"),
    ],
)
def test_invalid_dataarray_sample_axis_rejected(
    make_sample_voxeldata_mask, signals, message
):
    mask = make_sample_voxeldata_mask()
    mask.data.flat[0] = True
    with pytest.raises(ValueError, match=message):
        vd.unmask(signals, mask)


def test_unsupported_signal_type_rejected(make_sample_voxeldata_mask):
    with pytest.raises(TypeError, match="Numpy array or DataArray"):
        vd.unmask([1.0], make_sample_voxeldata_mask())  # ty: ignore[invalid-argument-type]


def test_dataarray_without_leading_dimensions(make_sample_voxeldata_mask):
    mask = make_sample_voxeldata_mask()
    mask.data.flat[0] = True
    result = vd.unmask(xr.DataArray([9.0], dims="space"), mask)
    expected = np.zeros(mask.shape)
    expected.flat[0] = 9.0
    np.testing.assert_array_equal(result, expected)
