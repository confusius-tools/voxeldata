"""Affine and bounding-box regressions adapted from ConfUSIus's xarray tests."""

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
    np.testing.assert_allclose(bounds.values, expected)
    assert bounds.dims == ("bound", "component")
    np.testing.assert_array_equal(bounds.coords["bound"], ["min", "max"])
    np.testing.assert_array_equal(bounds.coords["component"], ["z", "y", "x"])
    assert bounds.attrs["units"] == "mm"
    np.testing.assert_allclose(data.vd.bounding_box.values, expected)


def test_bounds_use_stride_and_descending_coordinate_step(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    data = sample_voxeldata_3d.isel(i=slice(None, None, -2))
    np.testing.assert_allclose(data.vd.bounding_box.sel(component="x"), [30, 42])


def test_bounds_restore_singleton_axis(sample_voxeldata_3d: xr.DataArray) -> None:
    expected = [[10.5, 19, 28.5], [11.5, 25, 40.5]]
    for data in (sample_voxeldata_3d.isel(k=1), sample_voxeldata_3d.isel(k=[1])):
        np.testing.assert_allclose(vd.get_bounding_box(data), expected)


def test_irregular_bounds_are_rejected(
    sample_voxeldata_irregular: xr.DataArray,
) -> None:
    with pytest.raises(ValueError, match="irregular"):
        vd.get_bounding_box(sample_voxeldata_irregular)


def test_grid_bounds_have_independent_expected_values(
    sample_voxeldata_grids: xr.DataArray,
) -> None:
    bounds = sample_voxeldata_grids.vd.bounding_box
    assert bounds.dims == ("grid", "bound", "component")
    np.testing.assert_array_equal(bounds.coords["grid"], [10, 20])
    np.testing.assert_allclose(
        bounds.values,
        [[[9.5, 19, 28.5], [11.5, 25, 40.5]], [[109.5, -1, 33.5], [111.5, 5, 45.5]]],
    )


@pytest.mark.parametrize("fixture", ["sample_voxeldata_3d", "sample_voxeldata_grids"])
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
    if fixture == "sample_voxeldata_grids":
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


def test_grid_reindex_like_preserves_per_grid_world_coordinates(
    sample_voxeldata_grids: xr.DataArray,
) -> None:
    reference = sample_voxeldata_grids.isel(i=slice(None, None, 2))
    source = reference.vd.reindex_voxels().vd.set_affine("source_only", np.eye(4))
    result = source.vd.reindex_voxels_like(reference)
    assert set(result.vd.affines) == {"source_only"}
    np.testing.assert_array_equal(result.coords["i"], [0, 2])
    np.testing.assert_allclose(result.coords["x"].isel(k=0, j=0), [[30, 36], [35, 41]])
    np.testing.assert_array_equal(result.values, reference.values)


@pytest.mark.parametrize("mismatch", ["labels", "length", "missing"])
def test_grid_reindex_like_requires_corresponding_grids(
    sample_voxeldata_grids: xr.DataArray, mismatch: str
) -> None:
    reference = sample_voxeldata_grids
    if mismatch == "labels":
        data = reference.isel(grid=[1, 0])
    elif mismatch == "length":
        data = reference.isel(grid=slice(0, 1))
    else:
        data = reference.isel(grid=0)
    with pytest.raises(ValueError, match="same grid coordinates"):
        vd.reindex_voxels_like(data, reference)


@pytest.mark.parametrize("affine", [np.eye(3), np.ones((3, 4, 4))])
def test_apply_rejects_wrong_shapes(
    sample_voxeldata_grids: xr.DataArray, affine: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape matching"):
        sample_voxeldata_grids.vd.apply(affine)


def test_shared_rotation_applies_to_every_grid(
    sample_voxeldata_grids: xr.DataArray, named_affines: dict[str, np.ndarray]
) -> None:
    result = sample_voxeldata_grids.vd.apply(named_affines["lab"])
    origins = result.coords["z"].isel(k=0, j=0, i=0)
    np.testing.assert_allclose(origins, [-15, 5])
    np.testing.assert_allclose(result.coords["y"].isel(k=0, j=0, i=0), [17, 117])
    np.testing.assert_array_equal(result.values, sample_voxeldata_grids.values)
