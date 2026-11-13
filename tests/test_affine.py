"""Affine and bounding-box regressions."""

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.mark.parametrize(
    "fixture", ["sample_voxeldata_3d", "sample_voxeldata_3d_oblique"]
)
def test_bounds_match_full_materialized_edge_grid(
    request: pytest.FixtureRequest, fixture: str
) -> None:
    data = request.getfixturevalue(fixture)
    # Independently enumerate all voxel edges, not just the implementation's corners.
    affine = (
        np.array([[1, 0, 0, 10], [0, 2, 0, 20], [0, 0, 3, 30], [0, 0, 0, 1]])
        if fixture == "sample_voxeldata_3d"
        else np.array(
            [[0.4, 0, 0.1, 10], [0.1, 0.3, 0, 20], [0, 0.05, 0.25, 30], [0, 0, 0, 1]]
        )
    )
    edges = np.stack(
        np.meshgrid(
            np.arange(-0.5, 2.5),
            np.arange(-0.5, 3.5),
            np.arange(-0.5, 4.5),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    world = edges @ affine[:3, :3].T + affine[:3, 3]
    expected = np.stack([world.min(axis=0), world.max(axis=0)])
    bounds = vd.get_bounding_box(data)
    assert isinstance(bounds, dict)
    assert tuple(bounds) == vd.WORLD_DIMS
    for axis, dim in enumerate(vd.WORLD_DIMS):
        assert isinstance(bounds[dim], tuple)
        assert all(isinstance(value, float) for value in bounds[dim])
        np.testing.assert_allclose(bounds[dim], expected[:, axis])
    assert data.vd.bounding_box == bounds


def test_bounds_use_stride_and_descending_coordinate_step(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    data = sample_voxeldata_3d.isel(i=slice(None, None, -2))
    np.testing.assert_allclose(data.vd.bounding_box["x"], [30, 42])


def test_bounds_restore_singleton_axis(sample_voxeldata_3d: xr.DataArray) -> None:
    expected = [[10.5, 19, 28.5], [11.5, 25, 40.5]]
    for data in (sample_voxeldata_3d.isel(k=1), sample_voxeldata_3d.isel(k=[1])):
        bounds = vd.get_bounding_box(data)
        for axis, dim in enumerate(vd.WORLD_DIMS):
            np.testing.assert_allclose(bounds[dim], np.asarray(expected)[:, axis])


def test_irregular_bounds_are_rejected(
    sample_voxeldata_irregular: xr.DataArray,
) -> None:
    with pytest.raises(ValueError, match="irregular"):
        vd.get_bounding_box(sample_voxeldata_irregular)


def test_view_bounds_enclose_all_grids_and_selection_recovers_individual_bounds(
    sample_voxeldata_views: xr.DataArray,
) -> None:
    bounds = sample_voxeldata_views.vd.bounding_box
    assert bounds == {"z": (9.5, 111.5), "y": (-1.0, 25.0), "x": (28.5, 45.5)}
    assert sample_voxeldata_views.vd.geometry_center == {
        "z": 60.5,
        "y": 12.0,
        "x": 37.0,
    }
    for label, expected in [
        (10, [[9.5, 19, 28.5], [11.5, 25, 40.5]]),
        (20, [[109.5, -1, 33.5], [111.5, 5, 45.5]]),
    ]:
        selected = sample_voxeldata_views.sel(view=label)
        for axis, dim in enumerate(vd.WORLD_DIMS):
            np.testing.assert_allclose(
                selected.vd.bounding_box[dim], np.asarray(expected)[:, axis]
            )
            np.testing.assert_allclose(
                selected.vd.geometry_center[dim], np.mean(expected, axis=0)[axis]
            )


@pytest.mark.parametrize("fixture", ["sample_voxeldata_3d", "sample_voxeldata_views"])
@pytest.mark.parametrize("inplace", [False, True])
def test_reindex_preserves_samples_world_and_named_frames(
    request: pytest.FixtureRequest, fixture: str, inplace: bool
) -> None:
    data = request.getfixturevalue(fixture).vd.set_affine("lab", np.eye(4))
    cropped = data.isel(k=slice(1, 2), j=slice(None, None, 2), i=slice(None, None, -2))
    before = cropped.copy(deep=True)
    result = cropped.vd.reindex_voxels(inplace=inplace)
    assert (result is cropped) is inplace
    for dim, expected in [("k", [0]), ("j", [0, 1]), ("i", [0, 1])]:
        np.testing.assert_array_equal(result.coords[dim], expected)
    expected_affine = np.array(
        [[1, 0, 0, 11], [0, 4, 0, 20], [0, 0, -6, 39], [0, 0, 0, 1]]
    )
    if fixture == "sample_voxeldata_views":
        second = expected_affine.copy()
        second[:3, 3] = [111, 0, 44]
        expected_affine = np.stack([expected_affine, second])
    np.testing.assert_allclose(result.vd.voxel_to_world, expected_affine)
    np.testing.assert_array_equal(result.values, before.values)
    np.testing.assert_array_equal(result.vd.get_affine("lab"), np.eye(4))
    for name in ("z", "y", "x"):
        np.testing.assert_allclose(result.coords[name], before.coords[name])
    if not inplace:
        xr.testing.assert_identical(cropped, before)


def test_reindex_rejects_irregular_spacing(
    sample_voxeldata_irregular: xr.DataArray,
) -> None:
    with pytest.raises(ValueError, match="spacing is undefined"):
        sample_voxeldata_irregular.vd.reindex_voxels()


@pytest.mark.parametrize("inplace", [False, True])
def test_reindex_like_matches_reference_labels_without_replacing_frames(
    sample_voxeldata_3d: xr.DataArray, plain_volume: xr.DataArray, inplace: bool
) -> None:
    reference = sample_voxeldata_3d.isel(i=slice(None, None, 2)).vd.set_affine(
        "reference_only", np.eye(4)
    )
    values = np.arange(12).reshape(2, 3, 2)
    source = vd.create_voxeldata(
        values,
        dims=("k", "j", "i"),
        voxel_to_world=np.array(
            [[1, 0, 0, 10], [0, 2, 0, 20], [0, 0, 6, 30], [0, 0, 0, 1]]
        ),
        affines={"source_only": np.eye(4)},
    )
    result = source.vd.reindex_voxels_like(reference, inplace=inplace)
    assert (result is source) is inplace
    np.testing.assert_array_equal(result.coords["i"], [0, 2])
    np.testing.assert_array_equal(result.values, values)
    assert set(result.vd.affines) == {"source_only"}
    np.testing.assert_allclose(result.coords["x"].isel(k=0, j=0), [30, 36])
    with pytest.raises(ValueError, match="same voxel grid shape"):
        vd.reindex_voxels_like(source.isel(i=[0]), reference)
    with pytest.raises(ValueError, match="not aligned in world space"):
        vd.reindex_voxels_like(
            source.vd.set_voxel_to_world(np.eye(4), named_affines={}), reference
        )
    with pytest.raises(ValueError, match="data must have a voxel-to-world index"):
        vd.reindex_voxels_like(plain_volume, reference)
    with pytest.raises(ValueError, match="reference must have a voxel-to-world index"):
        vd.reindex_voxels_like(source, plain_volume)


def test_view_reindex_like_preserves_per_view_world_coordinates(
    sample_voxeldata_views: xr.DataArray,
) -> None:
    reference = sample_voxeldata_views.isel(i=slice(None, None, 2))
    source = reference.vd.reindex_voxels().vd.set_affine("source_only", np.eye(4))
    result = source.vd.reindex_voxels_like(reference)
    assert set(result.vd.affines) == {"source_only"}
    np.testing.assert_array_equal(result.coords["i"], [0, 2])
    np.testing.assert_allclose(result.coords["x"].isel(k=0, j=0), [[30, 36], [35, 41]])
    np.testing.assert_array_equal(result.values, reference.values)


@pytest.mark.parametrize("mismatch", ["labels", "length", "missing"])
def test_view_reindex_like_requires_corresponding_grids(
    sample_voxeldata_views: xr.DataArray, mismatch: str
) -> None:
    reference = sample_voxeldata_views
    if mismatch == "labels":
        data = reference.isel(view=[1, 0])
    elif mismatch == "length":
        data = reference.isel(view=slice(0, 1))
    else:
        data = reference.isel(view=0)
    with pytest.raises(ValueError, match="same geometry coordinates"):
        vd.reindex_voxels_like(data, reference)


@pytest.mark.parametrize("affine", [np.eye(3), np.ones((3, 4, 4))])
def test_apply_rejects_wrong_shapes(
    sample_voxeldata_views: xr.DataArray, affine: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape matching"):
        sample_voxeldata_views.vd.apply(affine)


def test_shared_rotation_applies_to_every_grid(
    sample_voxeldata_views: xr.DataArray, named_affines: dict[str, np.ndarray]
) -> None:
    result = sample_voxeldata_views.vd.apply(named_affines["lab"])
    origins = result.coords["z"].isel(k=0, j=0, i=0)
    np.testing.assert_allclose(origins, [-15, 5])
    np.testing.assert_allclose(result.coords["y"].isel(k=0, j=0, i=0), [17, 117])
    np.testing.assert_array_equal(result.values, sample_voxeldata_views.values)
