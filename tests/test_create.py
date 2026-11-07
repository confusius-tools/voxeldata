"""Constructor cases adapted from ConfUSIus, excluding probe-specific defaults."""

from typing import Any

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.mark.parametrize(
    "dims, shape, expected_shape",
    [(("i",), (4,), (1, 1, 4)), (("j", "i"), (3, 4), (1, 3, 4))],
)
def test_missing_spatial_axes_are_padded(
    dims: tuple[str, ...], shape: tuple[int, ...], expected_shape: tuple[int, ...]
) -> None:
    values = np.arange(np.prod(shape)).reshape(shape)
    result = vd.create_voxeldata(values, dims=dims, spacing=(1, 2, 3))
    assert result.dims == ("k", "j", "i")
    assert result.shape == expected_shape
    np.testing.assert_array_equal(result.values.reshape(-1), values.reshape(-1))
    np.testing.assert_allclose(result.coords["z"], 0)
    np.testing.assert_allclose(result.coords["x"].isel(k=0, j=0), [0, 3, 6, 9])


def test_numpy_input_is_not_copied(plain_volume: xr.DataArray) -> None:
    values = plain_volume.values
    result = vd.create_voxeldata(values, dims=("k", "j", "i"), spacing=(1, 2, 3))
    assert np.shares_memory(result.values, values)


def test_constructor_reorders_data_not_just_dimension_names(
    view_affines: np.ndarray,
) -> None:
    values = np.arange(4 * 2 * 3 * 2 * 2 * 3).reshape(4, 2, 3, 2, 2, 3)
    result = vd.create_voxeldata(
        values,
        dims=("i", "component", "j", "view", "k", "time"),
        extra_coords={"component": ["a", "b"], "view": [10, 20]},
        dt=0.5,
        voxel_to_world=view_affines,
        geometry_dims=("view",),
    )
    assert result.dims == ("component", "view", "time", "k", "j", "i")
    np.testing.assert_array_equal(
        result.values, np.transpose(values, (1, 3, 5, 4, 2, 0))
    )
    assert (
        result.isel(component=1, view=1, time=2, k=1, j=2, i=3).item()
        == values[3, 1, 2, 1, 1, 2]
    )


def test_direction_matrix_composes_with_spacing(plain_volume: xr.DataArray) -> None:
    data = vd.create_voxeldata(
        plain_volume.values,
        dims=("k", "j", "i"),
        spacing=(1, 2, 3),
        origin=(4, 5, 6),
        direction=[[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    )
    np.testing.assert_allclose(
        data.vd.voxel_to_world, [[0, 2, 0, 4], [1, 0, 0, 5], [0, 0, 3, 6], [0, 0, 0, 1]]
    )
    point = data.isel(k=1, j=2, i=3)
    np.testing.assert_allclose(
        [point.coords[d].item() for d in ("z", "y", "x")], [8, 6, 15]
    )


def test_constructor_preserves_coordinate_attrs_and_metadata(
    plain_volume: xr.DataArray, voxel_affine: np.ndarray
) -> None:
    attrs = {"description": "test volume", "units": "a.u."}
    data = vd.create_voxeldata(
        plain_volume.values,
        dims=("k", "j", "i"),
        k=xr.DataArray([2, 4], dims="k", attrs={"long_name": "native index"}),
        voxel_to_world=voxel_affine,
        name="intensity",
        attrs=attrs,
    )
    assert data.name == "intensity"
    assert data.attrs == attrs
    assert data.coords["k"].attrs == {"long_name": "native index"}
    np.testing.assert_allclose(data.coords["z"].isel(j=0, i=0), [12, 14])
    data.attrs["description"] = "changed"
    assert attrs["description"] == "test volume"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "spacing or voxel_to_world"),
        ({"spacing": (1, 2)}, "length 3"),
        ({"spacing": (0, 2, 3)}, "positive and finite"),
        ({"spacing": (1, np.inf, 3)}, "positive and finite"),
        ({"spacing": (1, 2, 3), "origin": (0, np.nan, 0)}, "finite"),
        ({"spacing": (1, 2, 3), "direction": np.eye(2)}, "shape"),
        ({"voxel_to_world": np.eye(4), "spacing": (1, 2, 3)}, "mutually exclusive"),
        ({"voxel_to_world": np.eye(3)}, "shape"),
        ({"voxel_to_world": np.diag([1, 1, 1, 2])}, "homogeneous"),
    ],
)
def test_invalid_constructor_geometry(
    plain_volume: xr.DataArray, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        vd.create_voxeldata(plain_volume.values, dims=("k", "j", "i"), **kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": [0]}, "1D with length"),
        ({"extra_coords": {"time": [0]}}, "must not include core"),
        ({"i": [0, 1, 1, 3]}, "strictly monotonic"),
    ],
)
def test_invalid_constructor_coordinates(
    plain_volume: xr.DataArray, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        vd.create_voxeldata(
            plain_volume.values, dims=("k", "j", "i"), spacing=(1, 2, 3), **kwargs
        )
