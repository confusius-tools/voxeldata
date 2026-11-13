"""Public accessor contracts and named affine frames, with independent oracles."""

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


def test_properties_expose_index_geometry(sample_voxeldata_3d: xr.DataArray) -> None:
    data = sample_voxeldata_3d
    assert data.vd.voxel_dims == ("k", "j", "i")
    assert data.vd.world_dims == ("z", "y", "x")
    assert data.vd.world_units == "mm"
    assert not data.vd.geometry_dims
    assert data.vd.is_axis_aligned
    assert data.vd.fixed_voxel_coords == {}
    assert data.vd.spacing == {"k": 1.0, "j": 2.0, "i": 3.0}
    assert data.vd.origin == {"z": 10.0, "y": 20.0, "x": 30.0}
    assert data.vd.scalings == {"k": 1.0, "j": 2.0, "i": 3.0}
    np.testing.assert_allclose(data.vd.direction, np.eye(3))
    np.testing.assert_allclose(
        data.vd.voxel_to_world,
        [[1, 0, 0, 10], [0, 2, 0, 20], [0, 0, 3, 30], [0, 0, 0, 1]],
    )


def test_spacing_and_origin_cover_non_spatial_dims(
    sample_voxeldata_3dt: xr.DataArray,
) -> None:
    assert sample_voxeldata_3dt.vd.spacing == {
        "time": 0.5,
        "k": 1.0,
        "j": 2.0,
        "i": 3.0,
    }
    assert sample_voxeldata_3dt.vd.origin == {
        "time": 10.0,
        "z": 10.0,
        "y": 20.0,
        "x": 30.0,
    }


def test_set_voxel_to_world_on_plain_dataarray(plain_volume: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="voxel-to-world index"):
        _ = plain_volume.vd.origin
    result = plain_volume.vd.set_voxel_to_world(
        np.diag([2.0, 3.0, 4.0, 1.0]), units="um"
    )
    assert result.vd.world_units == "um"
    assert result.vd.spacing == {"k": 2.0, "j": 3.0, "i": 4.0}
    assert not vd.has_voxel_to_world_index(plain_volume)
    np.testing.assert_array_equal(result.values, plain_volume.values)


def test_set_voxel_to_world_drops_named_affines_with_warning(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    data = sample_voxeldata_3d.vd.set_affine("lab", np.eye(4))
    with pytest.warns(UserWarning, match="drops existing named affine"):
        result = data.vd.set_voxel_to_world(np.eye(4))
    assert result.vd.affines == {}
    assert set(data.vd.affines) == {"lab"}


def test_set_affine_get_affine_remove_affine(
    sample_voxeldata_3d: xr.DataArray, named_affines: dict[str, np.ndarray]
) -> None:
    data = sample_voxeldata_3d
    with_affine = data.vd.set_affine("lab", named_affines["lab"])
    np.testing.assert_allclose(with_affine.vd.get_affine("lab"), named_affines["lab"])
    assert set(with_affine.vd.affines) == {"lab"}
    assert data.vd.affines == {}
    removed = with_affine.vd.remove_affine("lab")
    assert removed.vd.affines == {}
    assert set(with_affine.vd.affines) == {"lab"}
    with pytest.raises(KeyError, match="lab"):
        removed.vd.get_affine("lab")


@pytest.mark.parametrize("inplace", [False, True])
def test_apply_named_frame_composition(
    sample_voxeldata_3d: xr.DataArray,
    named_affines: dict[str, np.ndarray],
    inplace: bool,
) -> None:
    data = sample_voxeldata_3d.vd.set_affine("lab", named_affines["lab"]).vd.set_affine(
        "atlas", named_affines["atlas"]
    )
    before = data.copy(deep=True)
    result = data.vd.apply("lab", inplace=inplace)
    assert (result is data) is inplace
    np.testing.assert_array_equal(result.vd.get_affine("lab"), np.eye(4))
    np.testing.assert_allclose(
        result.vd.get_affine("atlas"),
        [[0, 2, 0, -1], [-3, 0, 0, 32], [0, 0, 4, -25], [0, 0, 0, 1]],
    )
    point = result.isel(k=1, j=2, i=3)
    np.testing.assert_allclose(
        [point.coords[d].item() for d in ("z", "y", "x")], [-19, 18, 50]
    )
    np.testing.assert_array_equal(result.values, before.values)
    assert result.attrs == before.attrs
    if not inplace:
        xr.testing.assert_identical(data, before)
        np.testing.assert_array_equal(data.vd.get_affine("lab"), named_affines["lab"])


def test_apply_per_view_frame_broadcasts_shared_frames(
    sample_voxeldata_views: xr.DataArray, named_affines: dict[str, np.ndarray]
) -> None:
    frames = np.stack([named_affines["lab"], named_affines["lab"].copy()])
    frames[1, :3, 3] += [10, 20, 30]
    data = sample_voxeldata_views.vd.set_affine("lab", frames).vd.set_affine(
        "atlas", named_affines["atlas"]
    )
    result = data.vd.apply("lab")
    np.testing.assert_array_equal(result.vd.get_affine("lab"), np.eye(4))
    np.testing.assert_allclose(
        result.vd.get_affine("atlas"),
        [
            [[0, 2, 0, -1], [-3, 0, 0, 32], [0, 0, 4, -25], [0, 0, 0, 1]],
            [[0, 2, 0, -41], [-3, 0, 0, 62], [0, 0, 4, -145], [0, 0, 0, 1]],
        ],
    )
    point = result.isel(view=1, k=0, j=0, i=0)
    np.testing.assert_allclose(
        [point.coords[d].item() for d in ("z", "y", "x")], [15, 137, 76]
    )
    np.testing.assert_array_equal(data.vd.get_affine("lab"), frames)


def test_to_uses_both_nonidentity_frames(
    sample_voxeldata_3d: xr.DataArray, named_affines: dict[str, np.ndarray]
) -> None:
    source = sample_voxeldata_3d.vd.set_affine("common", named_affines["lab"])
    target = sample_voxeldata_3d.vd.set_affine("common", named_affines["atlas"])
    expected = [
        [0, -0.5, 0, -4],
        [1 / 3, 0, 0, -10 / 3],
        [0, 0, 0.25, -2],
        [0, 0, 0, 1],
    ]
    np.testing.assert_allclose(source.vd.to(target, "common"), expected)
    np.testing.assert_allclose(
        vd.get_relative_affine(source, target, "common"), expected
    )


@pytest.mark.parametrize("missing_side", ["source", "target"])
def test_to_requires_frame_on_both_sides(
    sample_voxeldata_3d: xr.DataArray, missing_side: str
) -> None:
    framed = sample_voxeldata_3d.vd.set_affine("lab", np.eye(4))
    source, target = (
        (sample_voxeldata_3d, framed)
        if missing_side == "source"
        else (framed, sample_voxeldata_3d)
    )
    with pytest.raises(KeyError, match="lab"):
        source.vd.to(target, "lab")


@pytest.mark.parametrize(
    "indexer, positions",
    [(0, [0]), (slice(1, 2), [1]), ([1, 0], [1, 0]), (np.array([1]), [1])],
)
def test_view_selection_slices_named_affines(
    sample_voxeldata_views: xr.DataArray,
    view_affines: np.ndarray,
    indexer: object,
    positions: list[int],
) -> None:
    data = sample_voxeldata_views.vd.set_affine("lab", view_affines).vd.set_affine(
        "shared", np.eye(4)
    )
    selected = data.isel(view=indexer)
    expected = (
        view_affines[positions[0]]
        if isinstance(indexer, int)
        else view_affines[positions]
    )
    assert bool(selected.vd.geometry_dims) is not isinstance(indexer, int)
    np.testing.assert_array_equal(selected.vd.get_affine("lab"), expected)
    np.testing.assert_array_equal(selected.vd.get_affine("shared"), np.eye(4))
    np.testing.assert_array_equal(data.vd.get_affine("lab"), view_affines)


def test_spatial_selection_preserves_named_affines(
    sample_voxeldata_views: xr.DataArray, view_affines: np.ndarray
) -> None:
    data = sample_voxeldata_views.vd.set_affine("lab", view_affines)
    np.testing.assert_array_equal(data.isel(j=1).vd.get_affine("lab"), view_affines)


def test_concat_merges_named_affines(
    sample_voxeldata_views: xr.DataArray, view_affines: np.ndarray
) -> None:
    data = sample_voxeldata_views.vd.set_affine("lab", view_affines).vd.set_affine(
        "shared", np.eye(4)
    )
    combined = xr.concat(
        [data.isel(view=slice(1, 2)), data.isel(view=slice(0, 1))], dim="view"
    )
    np.testing.assert_array_equal(combined.coords["view"], [20, 10])
    np.testing.assert_array_equal(combined.vd.get_affine("lab"), view_affines[[1, 0]])
    np.testing.assert_array_equal(combined.vd.get_affine("shared"), np.eye(4))
    np.testing.assert_array_equal(combined.values[:, 0, 0, 0], [24, 0])


@pytest.mark.parametrize(
    "case, message",
    [
        ("names", "different named affines"),
        ("shared", "differs"),
        ("mixed", "mixes shared"),
    ],
)
def test_concat_rejects_incompatible_frames(
    sample_voxeldata_views: xr.DataArray, case: str, message: str
) -> None:
    left = sample_voxeldata_views.isel(view=slice(0, 1)).vd.set_affine("lab", np.eye(4))
    right = sample_voxeldata_views.isel(view=slice(1, 2))
    if case == "shared":
        right = right.vd.set_affine("lab", np.diag([2.0, 2.0, 2.0, 1.0]))
    elif case == "mixed":
        right = right.vd.set_affine("lab", np.eye(4)[None])
    with pytest.raises(ValueError, match=message):
        xr.concat([left, right], dim="view")


@pytest.mark.parametrize(
    "affine, message",
    [
        (np.eye(3), "shape"),
        (np.full((4, 4), np.nan), "finite"),
        (np.diag([1.0, 1.0, 1.0, 2.0]), "homogeneous"),
        (np.eye(4)[None], "shape"),
    ],
)
def test_named_affine_validation(
    sample_voxeldata_3d: xr.DataArray, affine: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_voxeldata_3d.vd.set_affine("bad", affine)
    assert sample_voxeldata_3d.vd.affines == {}


def test_named_frames_may_have_different_scales_per_grid(
    sample_voxeldata_views: xr.DataArray,
) -> None:
    frames = np.stack([np.eye(4), np.diag([2.0, 3.0, 4.0, 1.0])])
    data = sample_voxeldata_views.vd.set_affine("lab", frames)
    np.testing.assert_array_equal(data.vd.get_affine("lab"), frames)
    assert data.vd.index.equals(sample_voxeldata_views.vd.index)


@pytest.mark.parametrize("inplace", [False, True])
def test_frame_mutators_and_world_units_preserve_geometry(
    sample_voxeldata_3d: xr.DataArray,
    named_affines: dict[str, np.ndarray],
    inplace: bool,
) -> None:
    data = sample_voxeldata_3d
    before = data.copy(deep=True)
    framed = data.vd.set_affine("lab", named_affines["lab"], inplace=inplace)
    assert (framed is data) is inplace
    changed = framed.vd.set_world_units("um", inplace=inplace)
    assert (changed is framed) is inplace
    assert changed.vd.world_units == "um"
    assert changed.attrs["units"] == "a.u."
    np.testing.assert_array_equal(changed.vd.get_affine("lab"), named_affines["lab"])
    np.testing.assert_array_equal(changed.vd.voxel_to_world, before.vd.voxel_to_world)
    removed = changed.vd.remove_affine("lab", inplace=inplace)
    assert (removed is changed) is inplace
    assert removed.vd.affines == {}
    if not inplace:
        xr.testing.assert_identical(data, before)
        assert framed.vd.world_units == "mm"
        assert set(changed.vd.affines) == {"lab"}


def test_accessor_validation_and_restoration(sample_voxeldata_3d: xr.DataArray) -> None:
    data = sample_voxeldata_3d.isel(j=2)
    with pytest.raises(ValueError, match="native voxel dimensions"):
        data.vd.validate()
    for restored in (data.vd.canonicalize(), data.vd.ensure()):
        assert restored.dims == ("k", "j", "i")
        np.testing.assert_array_equal(restored.coords["j"], [2])
        np.testing.assert_array_equal(
            restored.values[:, 0, :], [[8, 9, 10, 11], [20, 21, 22, 23]]
        )
        np.testing.assert_allclose(restored.coords["y"], 24)
