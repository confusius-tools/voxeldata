"""Geometry regressions adapted from ConfUSIus's test_utils/test_geometry.py.

Only public behavior is exercised: world lookup, slicing, alignment, and affine
validation. ConfUSIus-specific helpers and implementation-only checks stay upstream.
"""

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.mark.parametrize(
    "world, label, voxel, position, expected",
    [("z", 20, "k", 1, 12), ("y", 6, "j", 2, 8), ("x", 9, "i", 2, 2)],
)
def test_world_lookup_uses_voxel_labels(
    sample_voxeldata_irregular: xr.DataArray,
    world: str,
    label: float,
    voxel: str,
    position: int,
    expected: int,
) -> None:
    data = sample_voxeldata_irregular
    result = data.sel({world: label})
    np.testing.assert_array_equal(result.values, data.isel({voxel: position}).values)
    assert result.values.flat[0] == expected


@pytest.mark.parametrize("label", [5.0, 999.0])
def test_exact_lookup_rejects_between_or_outside_samples(
    sample_voxeldata_irregular: xr.DataArray, label: float
) -> None:
    with pytest.raises(KeyError):
        sample_voxeldata_irregular.sel(z=label)


def test_lookup_tolerance_is_in_world_units(
    sample_voxeldata_irregular: xr.DataArray,
) -> None:
    selected = sample_voxeldata_irregular.sel(z=5, tolerance=6)
    assert selected.coords["k"].item() == 0
    assert selected.values.flat[0] == 0
    with pytest.raises(KeyError):
        sample_voxeldata_irregular.sel(z=5, tolerance=4)


def test_descending_and_nonmonotonic_lookup(plain_volume: xr.DataArray) -> None:
    data = vd.attach_voxel_to_world_index(
        plain_volume.assign_coords(k=[4, 0], j=[0, 3, 1]), np.eye(4)
    )
    np.testing.assert_array_equal(
        data.sel(z=0).values, [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
    )
    np.testing.assert_array_equal(
        data.sel(y=1).values, [[8, 9, 10, 11], [20, 21, 22, 23]]
    )
    with pytest.raises(KeyError):
        data.sel(y=2)


def test_fixed_plane_lookup_validates_label(
    sample_voxeldata_irregular: xr.DataArray,
) -> None:
    fixed = sample_voxeldata_irregular.isel(k=1)
    xr.testing.assert_identical(fixed.sel(z=20), fixed)
    xr.testing.assert_identical(fixed.sel(z=slice(10, 30)), fixed)
    xr.testing.assert_identical(fixed.sel(z=-100, method="nearest"), fixed)
    with pytest.raises(KeyError):
        fixed.sel(z=0)
    with pytest.raises(KeyError):
        fixed.sel(z=slice(0, 10))


@pytest.mark.parametrize(
    "label, expected",
    [
        (slice(0, 21, 2), [0, 3]),
        (slice(21, 0), []),
        (slice(0, 21, -1), []),
        (slice(1000, 2000), []),
    ],
)
def test_world_label_slice_semantics(
    sample_voxeldata_irregular: xr.DataArray, label: slice, expected: list[int]
) -> None:
    selected = sample_voxeldata_irregular.sel(x=label)
    np.testing.assert_array_equal(selected.coords["i"], expected)


def test_descending_world_slices(sample_voxeldata_3d: xr.DataArray) -> None:
    flipped = sample_voxeldata_3d.isel(i=slice(None, None, -1))
    np.testing.assert_array_equal(flipped.sel(x=slice(39, 33)).coords["i"], [3, 2, 1])
    assert flipped.sel(x=slice(33, 39)).sizes["i"] == 0


def test_oblique_point_lookup_has_independent_expected_value(
    sample_voxeldata_3d_oblique: xr.DataArray,
) -> None:
    selected = sample_voxeldata_3d_oblique.sel(
        z=xr.Variable("point", [10.7]),
        y=xr.Variable("point", [20.7]),
        x=xr.Variable("point", [30.85]),
        method="nearest",
    )
    assert selected.item() == 23
    assert selected.coords["k"].item() == 1
    assert selected.coords["j"].item() == 2
    assert selected.coords["i"].item() == 3


def test_oblique_lookup_after_scalar_spatial_selection(
    sample_voxeldata_3d_oblique: xr.DataArray,
) -> None:
    selected = sample_voxeldata_3d_oblique.isel(k=1).sel(
        z=xr.Variable("point", [10.6]),
        y=xr.Variable("point", [20.4]),
        x=xr.Variable("point", [30.55]),
        method="nearest",
    )
    assert selected.item() == 18


def test_decoupled_row_can_select_without_other_world_labels(
    plain_volume: xr.DataArray,
) -> None:
    affine = np.array(
        [[0.4, 0, 0, 10], [0, 0.3, 0.1, 20], [0, 0.05, 0.25, 30], [0, 0, 0, 1]]
    )
    data = vd.attach_voxel_to_world_index(plain_volume, affine)
    selected = data.sel(z=10.4)
    assert selected.dims == ("j", "i")
    np.testing.assert_array_equal(
        selected.values, [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
    )
    with pytest.raises(ValueError, match="missing labels"):
        data.sel(y=20.4)


@pytest.mark.parametrize("combined", [False, True])
def test_view_world_lookup_requires_prior_scalar_selection(
    sample_voxeldata_views: xr.DataArray, combined: bool
) -> None:
    labels = {"view": 20, "z": 110.0} if combined else {"z": 110.0}
    with pytest.raises(ValueError, match="requires reducing `view` to a scalar"):
        sample_voxeldata_views.sel(labels)
    point = sample_voxeldata_views.sel(view=20).sel(z=111, y=4, x=44)
    assert point.item() == 47


def test_rotated_view_coordinates(
    plain_volume: xr.DataArray, voxel_affine: np.ndarray
) -> None:
    rotated = np.array([[0, -2, 0, 100], [1, 0, 0, 200], [0, 0, 3, 300], [0, 0, 0, 1]])
    data = plain_volume.expand_dims(view=[10, 20])
    result = vd.attach_voxel_to_world_index(
        data, np.stack([voxel_affine, rotated]), geometry_dims=("view",)
    )
    point = result.isel(view=1, k=1, j=2, i=3)
    np.testing.assert_allclose(
        [point.coords[d].item() for d in ("z", "y", "x")], [96, 201, 309]
    )
    assert not result.vd.is_axis_aligned
    assert result.vd.spacing == {"view": 10, "k": 1, "j": 2, "i": 3}


@pytest.mark.parametrize("property_name", ["origin", "direction"])
def test_view_scalar_geometry_requirements(
    sample_voxeldata_views: xr.DataArray, property_name: str
) -> None:
    with pytest.raises(ValueError, match="requires a single geometry"):
        getattr(sample_voxeldata_views.vd, property_name)
    assert sample_voxeldata_views.isel(view=1).vd.origin == {"z": 110, "y": 0, "x": 35}


@pytest.mark.parametrize(
    "failure, message",
    [
        ("nonfinite", "finite"),
        ("last_row", "homogeneous"),
    ],
)
def test_view_affine_validation(
    plain_volume: xr.DataArray, view_affines: np.ndarray, failure: str, message: str
) -> None:
    if failure == "nonfinite":
        view_affines[0, 0, 0] = np.nan
    elif failure == "last_row":
        view_affines[0, -1, -1] = 2
    else:
        view_affines[1, 0, 0] = 2
    with pytest.raises(ValueError, match=message):
        vd.attach_voxel_to_world_index(
            plain_volume.expand_dims(view=[10, 20]),
            view_affines,
            geometry_dims=("view",),
        )


@pytest.mark.parametrize(
    "failure, message",
    [
        ("missing_dim", "all native voxel dims"),
        ("missing_coord", "matching 1D coordinate"),
        ("float_coord", "integer dtype"),
    ],
)
def test_attach_rejects_invalid_voxel_axes(
    plain_volume: xr.DataArray, failure: str, message: str
) -> None:
    if failure == "missing_dim":
        data = plain_volume.isel(k=0, drop=True)
    elif failure == "missing_coord":
        data = plain_volume.drop_vars("k")
    else:
        data = plain_volume.assign_coords(k=[0.0, 1.0])
    with pytest.raises((TypeError, ValueError), match=message):
        vd.attach_voxel_to_world_index(data, np.eye(4))


@pytest.mark.parametrize("operation", ["align", "reindex"])
def test_incompatible_geometry_cannot_align(
    sample_voxeldata_3d: xr.DataArray, operation: str
) -> None:
    left = sample_voxeldata_3d
    right = left.vd.set_voxel_to_world(np.eye(4))
    with pytest.raises(ValueError, match="Cannot automatically"):
        if operation == "align":
            _ = left + right
        else:
            left.reindex_like(right)


def test_matching_geometry_aligns_and_index_equality_checks_affine(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    data = sample_voxeldata_3d
    result = data + data.copy(deep=True)
    np.testing.assert_array_equal(result.values, 2 * np.arange(24).reshape(2, 3, 4))
    assert data.vd.index.equals(data.copy(deep=True).vd.index)
    assert not data.vd.index.equals(data.vd.set_voxel_to_world(np.eye(4)).vd.index)
