"""Tests for the public VoxelData API."""

from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


def test_create_axis_aligned_geometry(plain_volume: xr.DataArray) -> None:
    data = vd.create_voxeldata(
        plain_volume.values,
        dims=("k", "j", "i"),
        spacing=(1.0, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
    )

    assert data.dims == ("k", "j", "i")
    assert vd.has_voxel_to_world_index(data)
    assert vd.get_voxel_to_world_units(data) == "mm"
    np.testing.assert_allclose(
        vd.get_voxel_to_world_affine(data),
        np.diag([1, 2, 3, 1])
        + np.array([[0, 0, 0, 10], [0, 0, 0, 20], [0, 0, 0, 30], [0, 0, 0, 0]]),
    )
    np.testing.assert_allclose(data.coords["z"].isel(k=1, j=2, i=3), 11.0)
    np.testing.assert_allclose(data.coords["y"].isel(k=1, j=2, i=3), 24.0)
    np.testing.assert_allclose(data.coords["x"].isel(k=1, j=2, i=3), 39.0)


def test_create_oblique_geometry_sel_world() -> None:
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    data = vd.create_voxeldata(
        np.arange(4 * 5 * 6).reshape(4, 5, 6),
        dims=("k", "j", "i"),
        voxel_to_world=np.block(
            [[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]
        ),
    )

    np.testing.assert_allclose(
        data.coords["z"].isel(k=1, j=2, i=3), -0.1339745962155614
    )
    np.testing.assert_allclose(data.coords["y"].isel(k=1, j=2, i=3), 2.232050807568877)
    selected = data.sel(
        z=xr.Variable("point", [-0.1339745962155614]),
        y=xr.Variable("point", [2.232050807568877]),
        x=xr.Variable("point", [3.0]),
        method="nearest",
    )
    assert selected.item() == 45


def test_grid_dependent_geometry(sample_voxeldata_grids: xr.DataArray) -> None:
    data = sample_voxeldata_grids

    index = vd.get_voxel_to_world_index(data)
    assert index is not None
    assert index.is_grid_dependent
    np.testing.assert_allclose(data.coords["z"].values[:, 0, 0, 0], [10.0, 110.0])
    selected = data.isel(grid=1)
    selected_index = vd.get_voxel_to_world_index(selected)
    assert selected_index is not None
    assert not selected_index.is_grid_dependent
    np.testing.assert_allclose(selected.coords["z"].isel(k=0, j=0, i=0), 110.0)
    np.testing.assert_array_equal(selected.values[0, 0], [24, 25, 26, 27])


def test_validate_rejects_non_voxeldata(plain_volume: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="VoxelToWorldIndex"):
        vd.validate_voxeldata(plain_volume)

    with pytest.raises(TypeError, match="xarray.DataArray"):
        vd.validate_voxeldata(cast(Any, np.zeros((2, 3, 4))))


def test_validate_rejects_wrong_voxel_dimensions(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    with pytest.raises(ValueError, match="native voxel dimensions"):
        vd.validate_voxeldata(sample_voxeldata_3d.isel(k=0, drop=True))


@pytest.mark.parametrize(
    "dims",
    [
        ("component", "j", "k", "i"),
        ("k", "component", "j", "i"),
        ("k", "j", "i", "component"),
    ],
)
def test_validate_rejects_wrong_spatial_block(
    sample_voxeldata_3d: xr.DataArray, dims: tuple[str, ...]
) -> None:
    data = sample_voxeldata_3d.expand_dims(component=[0, 1])
    with pytest.raises(ValueError, match="dimensions must end with"):
        vd.validate_voxeldata(data.transpose(*dims))


def test_canonicalize_restores_scalar_indexed_dimension(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    sliced = sample_voxeldata_3d.isel(j=2)
    assert "j" not in sliced.dims

    restored = vd.ensure_voxeldata(sliced)
    assert restored.dims == ("k", "j", "i")
    assert restored.sizes["j"] == 1
    np.testing.assert_allclose(restored.coords["y"].isel(k=0, j=0, i=0), 24.0)
    np.testing.assert_array_equal(
        restored.values[:, 0], [[8, 9, 10, 11], [20, 21, 22, 23]]
    )


def test_create_moves_only_spatial_dims() -> None:
    data = vd.create_voxeldata(
        np.zeros((3, 4, 5, 2)),
        dims=("k", "j", "i", "component"),
        spacing=(1.0, 1.0, 1.0),
    )
    assert data.dims == ("component", "k", "j", "i")


def test_apply_affine_shifts_world_coordinates(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    data = sample_voxeldata_3d
    shift = np.eye(4)
    shift[:3, 3] = [10.0, 5.0, -2.0]

    shifted = vd.apply_affine(data, shift)
    np.testing.assert_allclose(shifted.coords["z"].isel(k=0, j=0, i=0), 20.0)
    np.testing.assert_allclose(shifted.coords["y"].isel(k=0, j=0, i=0), 25.0)
    np.testing.assert_allclose(shifted.coords["x"].isel(k=0, j=0, i=0), 28.0)
    np.testing.assert_array_equal(shifted.values, np.arange(24).reshape(2, 3, 4))


def test_reindex_voxels_rebases_coordinates() -> None:
    data = vd.create_voxeldata(
        np.zeros((5, 5)), dims=("j", "i"), voxel_to_world=np.eye(4)
    )
    cropped = data.isel(j=slice(2, 5), i=slice(1, 5))

    reindexed = vd.reindex_voxels(cropped)
    np.testing.assert_array_equal(reindexed.coords["j"].values, np.arange(3))
    np.testing.assert_array_equal(reindexed.coords["i"].values, np.arange(4))
    np.testing.assert_allclose(reindexed.coords["y"].isel(j=0, i=0, k=0), 2.0)


def test_get_bounding_box_includes_half_voxel_extent() -> None:
    data = vd.create_voxeldata(
        np.zeros((2, 3, 4)), dims=("k", "j", "i"), spacing=(1.0, 2.0, 3.0)
    )
    bbox = vd.get_bounding_box(data)

    np.testing.assert_allclose(bbox.sel(bound="min", component="z"), -0.5)
    np.testing.assert_allclose(bbox.sel(bound="max", component="z"), 1.5)
    np.testing.assert_allclose(bbox.sel(bound="max", component="x"), 10.5)


def test_convert_length_units() -> None:
    np.testing.assert_allclose(vd.convert_length_units(1.0, "cm"), 10.0)
    np.testing.assert_allclose(vd.convert_length_units(1000.0, "um", "mm"), 1.0)
    with pytest.raises(ValueError, match="Unrecognized length unit"):
        vd.convert_length_units(1.0, "parsec")


def test_time_coordinates_generated_from_dt() -> None:
    data = vd.create_voxeldata(
        np.zeros((3, 2, 2, 2)),
        dims=("time", "k", "j", "i"),
        dt=0.5,
        t0=1.0,
        spacing=(1.0, 1.0, 1.0),
    )
    np.testing.assert_allclose(data.coords["time"].values, [1.0, 1.5, 2.0])
    assert data.coords["time"].attrs["units"] == "s"


def test_per_grid_time_coordinate() -> None:
    data = vd.create_voxeldata(
        np.zeros((3, 2, 2, 2, 2)),
        dims=("time", "grid", "k", "j", "i"),
        grid=np.array([0, 1]),
        dt=1.0,
        t0=np.array([0.0, 10.0]),
        voxel_to_world=np.stack([np.eye(4), np.eye(4)]),
    )
    assert data.coords["time"].dims == ("time", "grid")
    np.testing.assert_allclose(data.coords["time"].values[:, 1], [10.0, 11.0, 12.0])


@pytest.mark.parametrize(
    "leading_dims",
    [
        ("time", "grid"),
        ("grid", "time"),
        ("grid", "component", "time"),
        ("echo", "time", "component", "grid"),
    ],
)
def test_nonspatial_order_and_grid_geometry(leading_dims: tuple[str, ...]) -> None:
    dims = (*leading_dims, "k", "j", "i")
    sizes = {"grid": 2, "time": 3, "component": 2, "echo": 2, "k": 2, "j": 3, "i": 4}
    values = np.arange(np.prod([sizes[dim] for dim in dims])).reshape(
        tuple(sizes[dim] for dim in dims)
    )
    affines = np.stack([np.eye(4), np.eye(4)])
    affines[1, :3, 3] = [10.0, 20.0, 30.0]
    frames = np.stack([np.eye(4), np.eye(4)])
    frames[1, 0, 3] = 5.0
    data = vd.create_voxeldata(
        values,
        dims=dims,
        dt=0.5,
        t0=np.array([0.0, 10.0]),
        voxel_to_world=affines,
        affines={"reference": frames},
    )

    assert data.dims == dims
    np.testing.assert_array_equal(data.values, values)
    data.vd.validate()
    assert data.coords["time"].dims == ("time", "grid")
    np.testing.assert_allclose(data.coords["time"].values[:, 1], [10.0, 10.5, 11.0])
    np.testing.assert_allclose(data.coords["z"].isel(grid=1, k=1, j=0, i=0), 11.0)
    selected = data.isel(grid=1)
    assert not selected.vd.is_grid_dependent
    np.testing.assert_allclose(selected.vd.voxel_to_world, affines[1])
    np.testing.assert_allclose(selected.vd.get_affine("reference"), frames[1])

    expected_dims = (
        *(dim for dim in leading_dims if dim not in ("grid", "time")),
        "grid",
        "time",
        "k",
        "j",
        "i",
    )
    for canonical in (data.vd.canonicalize(), data.vd.ensure()):
        assert canonical.dims == expected_dims
        np.testing.assert_array_equal(
            canonical.values, data.transpose(*expected_dims).values
        )
        np.testing.assert_allclose(canonical.vd.voxel_to_world, affines)
        np.testing.assert_allclose(canonical.vd.get_affine("reference"), frames)
        canonical.vd.validate()
        invalid_time = canonical.assign_coords(time=-canonical.coords["time"])
        with pytest.raises(ValueError, match="strictly monotonic-increasing"):
            invalid_time.vd.validate()

    restored = data.isel(j=1).vd.ensure()
    assert restored.dims == expected_dims
    assert restored.sizes["j"] == 1
    np.testing.assert_allclose(restored.coords["y"].isel(grid=1, k=0, j=0, i=0), 21.0)

    single_grid = selected.vd.ensure()
    assert "grid" not in single_grid.dims
    assert not single_grid.vd.is_grid_dependent
    np.testing.assert_allclose(single_grid.vd.voxel_to_world, affines[1])
    with pytest.raises(ValueError, match="must not have a 'grid' dimension"):
        data.vd.validate(allow_grid=False)

    combined = xr.concat(
        [data.isel(grid=slice(0, 1)), data.isel(grid=slice(1, 2))], dim="grid"
    )
    assert combined.dims == dims
    np.testing.assert_array_equal(combined.values, values)
    np.testing.assert_allclose(combined.vd.voxel_to_world, affines)
    np.testing.assert_allclose(combined.vd.get_affine("reference"), frames)


@pytest.mark.parametrize("ngrid", [1, 2])
def test_grid_requires_matching_affine_stack(ngrid: int) -> None:
    values = np.zeros((ngrid, 2, 2, 2))
    shared = vd.create_voxeldata(
        values, dims=("grid", "k", "j", "i"), voxel_to_world=np.eye(4)
    )
    assert shared.vd.geometry_dims == ()
    assert shared.z.dims == ("k", "j", "i")
    with pytest.raises(ValueError, match="stack length"):
        vd.create_voxeldata(
            values,
            dims=("grid", "k", "j", "i"),
            voxel_to_world=np.stack([np.eye(4)] * (ngrid + 1)),
        )
    data = vd.create_voxeldata(
        values,
        dims=("grid", "k", "j", "i"),
        voxel_to_world=np.stack([np.eye(4)] * ngrid),
    )
    assert data.vd.is_grid_dependent
    assert data.vd.voxel_to_world.shape == (ngrid, 4, 4)
    data.vd.validate()
