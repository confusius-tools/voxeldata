"""World-transform API numerical and preservation regressions."""

from typing import Any

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.mark.parametrize(
    "fixture", ["sample_voxeldata_3d_oblique", "sample_voxeldata_views"]
)
def test_geometry_centers_match_sampled_world_extents(
    request: pytest.FixtureRequest, fixture: str
) -> None:
    data = request.getfixturevalue(fixture).isel(i=slice(None, None, -2))
    expected = np.stack(
        [(data.coords[d].min() + data.coords[d].max()) / 2 for d in vd.WORLD_DIMS],
        axis=-1,
    )
    center = vd.get_geometry_center(data)
    np.testing.assert_allclose(center, expected)
    assert center.dims == ("component",)
    np.testing.assert_array_equal(center.component, vd.WORLD_DIMS)
    assert center.attrs["units"] == data.vd.world_units
    xr.testing.assert_identical(center, data.vd.geometry_center)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    ("method", "args", "linear", "offset"),
    [
        ("translate", ([2, -3, 4],), np.eye(3), [2, -3, 4]),
        ("rotate", (90, "x"), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], None),
        ("zoom", ([2, -3, 4],), np.diag([2, -3, 4]), None),
        ("flip", ("yx",), np.diag([1, -1, -1]), None),
    ],
)
def test_transforms_preserve_samples_labels_and_reexpress_frames(
    sample_voxeldata_views: xr.DataArray,
    named_affines: dict[str, np.ndarray],
    inplace: bool,
    method: str,
    args: tuple[Any, ...],
    linear: Any,
    offset: Any,
) -> None:
    data = sample_voxeldata_views.isel(i=slice(None, None, -2))
    frames = {**named_affines, "stacked": np.stack(list(named_affines.values()))}
    for name, frame in frames.items():
        data = data.vd.set_affine(name, frame)
    before = data.copy(deep=True)
    # Midpoint of the enclosing bounds, not the first grid's center.
    pivot = np.array([60.5, 12, 38.5])
    action = np.eye(4)
    action[:3, :3] = linear
    action[:3, 3] = offset if offset is not None else pivot - np.asarray(linear) @ pivot
    result = getattr(data.vd, method)(*args, inplace=inplace)
    assert (result is data) is inplace
    assert result.data is data.data
    np.testing.assert_array_equal(result.values, before.values)
    for dim in data.dims:
        np.testing.assert_array_equal(result.coords[dim], before.coords[dim])
    np.testing.assert_allclose(
        result.vd.voxel_to_world, action @ before.vd.voxel_to_world, atol=1e-13
    )
    for name, frame in frames.items():
        np.testing.assert_allclose(
            result.vd.get_affine(name),
            frame @ np.linalg.inv(action),
            atol=1e-13,
        )
    vd.validate_voxeldata(result)
    if not inplace:
        xr.testing.assert_identical(data, before)


@pytest.mark.parametrize("axis", ["z", "y", "x", [1, 2, 3], [1e300, -1e300, 1e300]])
def test_rotation_matches_independent_quaternion_reference(
    sample_voxeldata_3d_oblique: xr.DataArray, axis: Any
) -> None:
    vector = (
        np.eye(3)[vd.WORLD_DIMS.index(axis)]
        if isinstance(axis, str)
        else np.array(axis, dtype=float)
    )
    vector /= np.max(np.abs(vector))
    vector /= np.linalg.norm(vector)
    w = np.cos(0.37 / 2)
    a, b, c = np.sin(0.37 / 2) * vector
    expected = np.array(
        [
            [1 - 2 * (b * b + c * c), 2 * (a * b - w * c), 2 * (a * c + w * b)],
            [2 * (a * b + w * c), 1 - 2 * (a * a + c * c), 2 * (b * c - w * a)],
            [2 * (a * c - w * b), 2 * (b * c + w * a), 1 - 2 * (a * a + b * b)],
        ]
    )
    action = np.eye(4)
    action[:3, :3] = expected
    result = vd.apply_rotation(
        sample_voxeldata_3d_oblique, 0.37, axis, center=[0, 0, 0], degrees=False
    )
    np.testing.assert_allclose(
        result.vd.voxel_to_world, action @ sample_voxeldata_3d_oblique.vd.voxel_to_world
    )


def test_rotation_sequence_is_fixed_world_axis_order(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    data = sample_voxeldata_3d
    rotation_z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_x = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    action = np.eye(4)
    action[:3, :3] = rotation_x @ rotation_z @ rotation_x
    result = data.vd.rotate([90, 90, 90], "xzx", center=[0, 0, 0])
    np.testing.assert_allclose(
        result.vd.voxel_to_world, action @ data.vd.voxel_to_world, atol=1e-13
    )


@pytest.mark.parametrize(
    "center",
    [[3, 4, 5], {"x": 5, "z": 3, "y": 4, "ignored": 9}, xr.DataArray([3, 4, 5])],
)
def test_explicit_center_is_shared(
    sample_voxeldata_views: xr.DataArray, center: Any
) -> None:
    result = sample_voxeldata_views.vd.zoom(2, center=center)
    action = np.diag([2, 2, 2, 1])
    action[:3, 3] = [-3, -4, -5]
    np.testing.assert_allclose(
        result.vd.voxel_to_world, action @ sample_voxeldata_views.vd.voxel_to_world
    )


def test_origin_pivot_uses_cropped_descending_native_labels(
    sample_voxeldata_views: xr.DataArray,
) -> None:
    data = sample_voxeldata_views.sel(view=20).isel(
        k=slice(1, 2), i=slice(None, None, -1)
    )
    result = data.vd.zoom(2, center="origin")
    first = {"k": 0, "j": 0, "i": 0}
    for name in vd.WORLD_DIMS:
        np.testing.assert_allclose(
            result.coords[name].isel(first), data.coords[name].isel(first)
        )
    expected = data.vd.voxel_to_world.copy()
    expected[..., :3, :3] *= 2
    expected[..., :3, 3] -= [1, 0, 9]
    np.testing.assert_allclose(result.vd.voxel_to_world, expected)


def test_origin_pivot_requires_selecting_geometry_dependencies(
    sample_voxeldata_views: xr.DataArray,
) -> None:
    with pytest.raises(ValueError, match="geometry dimensions"):
        sample_voxeldata_views.vd.zoom(2, center="origin")


@pytest.mark.parametrize("method", ["translate", "rotate", "zoom", "flip"])
def test_scalar_selected_voxel_dims_are_restored(
    sample_voxeldata_3d: xr.DataArray, method: str
) -> None:
    args = {
        "translate": ([1, 2, 3],),
        "rotate": (90, "x"),
        "zoom": (2,),
        "flip": ("x",),
    }[method]
    selected = sample_voxeldata_3d.isel(k=1)
    expected = getattr(sample_voxeldata_3d.isel(k=[1]).vd, method)(*args)
    result = getattr(selected.vd, method)(*args)
    xr.testing.assert_identical(result, expected)
    with pytest.raises(ValueError, match="inplace=True needs all native voxel dims"):
        getattr(selected.vd, method)(*args, inplace=True)


@pytest.mark.parametrize(
    "function,args",
    [
        (vd.apply_translation, ([1, 2, 3],)),
        (vd.apply_rotation, (30, "x")),
        (vd.apply_zoom, (2,)),
        (vd.apply_flip, ("x",)),
    ],
)
def test_transforms_reject_plain_arrays(
    plain_volume: xr.DataArray, function: Any, args: tuple[Any, ...]
) -> None:
    with pytest.raises(ValueError):
        function(plain_volume, *args)


@pytest.mark.parametrize(
    "function,args",
    [
        (vd.apply_translation, ([1, 2],)),
        (vd.apply_translation, ([1, np.nan, 3],)),
        (vd.apply_translation, (["1", "2", "3"],)),
        (vd.apply_translation, ([True] * 3,)),
        (vd.apply_translation, ([1j, 2, 3],)),
        (vd.apply_translation, ([[1], [2, 3]],)),
        (vd.apply_rotation, (np.inf, "x")),
        (vd.apply_rotation, (1, "xy")),
        (vd.apply_rotation, ([1, 2], "x")),
        (vd.apply_rotation, (1, "")),
        (vd.apply_rotation, (1, "k")),
        (vd.apply_rotation, (1, [0, 0, 0])),
        (vd.apply_rotation, (1, [1, 2])),
        (vd.apply_rotation, (1, [np.nan, 0, 0])),
        (vd.apply_zoom, (0,)),
        (vd.apply_zoom, ([1, 0, 2],)),
        (vd.apply_zoom, ([1, 2],)),
        (vd.apply_zoom, (True,)),
        (vd.apply_zoom, ("2",)),
        (vd.apply_zoom, (np.inf,)),
        (vd.apply_flip, ("xx",)),
        (vd.apply_flip, ("",)),
        (vd.apply_flip, ("k",)),
        (vd.apply_flip, (1,)),
    ],
)
def test_invalid_transform_inputs(
    sample_voxeldata_3d: xr.DataArray, function: Any, args: tuple[Any, ...]
) -> None:
    with pytest.raises(ValueError):
        function(sample_voxeldata_3d, *args)


@pytest.mark.parametrize(
    "center", ["invalid", [1, 2], [0, np.inf, 0], {"x": 1}, ["1", "2", "3"]]
)
def test_invalid_centers(sample_voxeldata_3d: xr.DataArray, center: Any) -> None:
    with pytest.raises(ValueError, match="center"):
        sample_voxeldata_3d.vd.zoom(2, center=center)


def test_irregular_geometry_needs_explicit_or_origin_pivot(
    sample_voxeldata_irregular: xr.DataArray,
) -> None:
    data = sample_voxeldata_irregular
    with pytest.raises(ValueError, match="irregular"):
        data.vd.zoom(2)
    result = data.vd.zoom(2, center=[0, 0, 0])
    np.testing.assert_allclose(
        result.vd.voxel_to_world[:3], 2 * data.vd.voxel_to_world[:3]
    )


def test_transforms_keep_lazy_signal_uncomputed(
    sample_voxeldata_views: xr.DataArray,
) -> None:
    import dask.array as da
    from dask.callbacks import Callback

    data = sample_voxeldata_views.chunk({"view": 1})
    with Callback(pretask=lambda *args: pytest.fail("Signal data computed")):
        result = (
            data.vd.translate([1, 2, 3]).vd.rotate(90, "x").vd.zoom(2).vd.flip("yx")
        )
        assert isinstance(result.data, da.Array)
        assert result.data is data.data
    np.testing.assert_array_equal(result.compute().values, data.compute().values)


@pytest.mark.parametrize(
    "function,method,args",
    [
        (vd.apply_translation, "translate", ([1, 2, 3],)),
        (vd.apply_rotation, "rotate", (45, "y")),
        (vd.apply_zoom, "zoom", (2,)),
        (vd.apply_flip, "flip", ("z",)),
    ],
)
def test_function_and_accessor_keep_nonspatial_shared_geometry(
    sample_voxeldata_3dt: xr.DataArray,
    function: Any,
    method: str,
    args: tuple[Any, ...],
) -> None:
    data = sample_voxeldata_3dt
    result = function(data, *args)
    assert result.vd.geometry_dims == ()
    assert result.vd.voxel_to_world.shape == (4, 4)
    xr.testing.assert_identical(result, getattr(data.vd, method)(*args))
    np.testing.assert_array_equal(result.values, data.values)


def test_multiple_geometry_dependencies_and_variable_scales() -> None:
    affines = np.broadcast_to(np.eye(4), (2, 3, 4, 4)).copy()
    affines[..., 0, 0] = [[1, 2, 3], [4, 5, 6]]
    affines[..., :3, 3] = np.arange(18).reshape(2, 3, 3)
    data = vd.create_voxeldata(
        np.arange(48).reshape(3, 2, 2, 2, 2),
        dims=("time", "view", "k", "j", "i"),
        extra_coords={"view": [5, 8]},
        time=[0, 1, 2],
        voxel_to_world=affines,
        geometry_dims=("view", "time"),
    )
    result = data.vd.zoom([2, 3, 4])
    assert result.vd.geometry_dims == ("view", "time")
    np.testing.assert_allclose(result.vd.geometry_center, data.vd.geometry_center)
    np.testing.assert_allclose(
        result.vd.voxel_to_world[..., :3, :3], np.diag([2, 3, 4]) @ affines[..., :3, :3]
    )
    np.testing.assert_array_equal(result.values, data.values)
