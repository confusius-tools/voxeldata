"""Tests for the flat `.vd` accessor and named affine frames."""

from typing import Any

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


def _data(**kwargs: Any) -> xr.DataArray:
    return vd.create_voxeldata(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        spacing=(1.0, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
        **kwargs,
    )


def test_properties_expose_index_geometry() -> None:
    data = _data()

    assert data.vd.voxel_dims == ("k", "j", "i")
    assert data.vd.world_dims == ("z", "y", "x")
    assert data.vd.units == "mm"
    assert not data.vd.is_grid_dependent
    assert data.vd.is_axis_aligned
    assert data.vd.fixed_voxel_coords == {}
    assert data.vd.spacing == {"k": 1.0, "j": 2.0, "i": 3.0}
    assert data.vd.origin == {"z": 10.0, "y": 20.0, "x": 30.0}
    assert data.vd.scalings == {"k": 1.0, "j": 2.0, "i": 3.0}
    np.testing.assert_allclose(data.vd.direction, np.eye(3))
    np.testing.assert_allclose(
        data.vd.voxel_to_world,
        np.diag([1, 2, 3, 1])
        + np.array([[0, 0, 0, 10], [0, 0, 0, 20], [0, 0, 0, 30], [0, 0, 0, 0]]),
    )


def test_spacing_and_origin_cover_non_spatial_dims() -> None:
    data = vd.create_voxeldata(
        np.zeros((4, 2, 3, 4)),
        dims=("time", "k", "j", "i"),
        dt=0.5,
        t0=10.0,
        spacing=(1.0, 2.0, 3.0),
        origin=(10.0, 20.0, 30.0),
    )

    assert data.vd.spacing == {"time": 0.5, "k": 1.0, "j": 2.0, "i": 3.0}
    assert data.vd.origin == {
        "time": 10.0,
        "z": 10.0,
        "y": 20.0,
        "x": 30.0,
    }


def test_set_voxel_to_world_on_plain_dataarray() -> None:
    data = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={"k": np.arange(2), "j": np.arange(3), "i": np.arange(4)},
    )
    with pytest.raises(ValueError, match="voxel-to-world index"):
        _ = data.vd.origin

    result = data.vd.set_voxel_to_world(np.diag([2.0, 3.0, 4.0, 1.0]), units="um")

    assert result.vd.units == "um"
    assert result.vd.spacing == {"k": 2.0, "j": 3.0, "i": 4.0}
    assert not vd.has_voxel_to_world_index(data)


def test_set_voxel_to_world_drops_named_affines_with_warning() -> None:
    data = _data(affines={"lab": np.eye(4)})

    with pytest.warns(UserWarning, match="drops existing named affine"):
        result = data.vd.set_voxel_to_world(np.eye(4))

    assert result.vd.affines == {}
    assert data.vd.affines != {}


def test_set_affine_get_affine_remove_affine() -> None:
    data = _data()
    lab = np.eye(4)
    lab[:3, 3] = [1.0, 2.0, 3.0]

    with_affine = data.vd.set_affine("lab", lab)
    np.testing.assert_allclose(with_affine.vd.get_affine("lab"), lab)
    assert set(with_affine.vd.affines) == {"lab"}
    assert data.vd.affines == {}

    removed = with_affine.vd.remove_affine("lab")
    assert removed.vd.affines == {}
    with pytest.raises(KeyError, match="lab"):
        removed.vd.get_affine("lab")


def test_apply_named_affine_makes_it_identity() -> None:
    lab = np.eye(4)
    lab[:3, 3] = [10.0, 20.0, 30.0]
    data = _data(affines={"lab": lab})

    result = data.vd.apply("lab")

    np.testing.assert_allclose(result.vd.get_affine("lab"), np.eye(4))
    np.testing.assert_allclose(result.vd.origin["z"], 20.0)
    np.testing.assert_allclose(result.vd.origin["y"], 40.0)
    np.testing.assert_allclose(result.vd.origin["x"], 60.0)


def test_apply_re_expresses_other_frames() -> None:
    lab = np.diag([2.0, 2.0, 2.0, 1.0])
    other = np.diag([4.0, 4.0, 4.0, 1.0])
    data = _data(affines={"lab": lab, "other": other})

    result = data.vd.apply("lab")

    np.testing.assert_allclose(result.vd.get_affine("lab"), np.eye(4))
    np.testing.assert_allclose(
        result.vd.get_affine("other"),
        other @ np.linalg.inv(lab),
    )


def test_to_uses_named_frame() -> None:
    lab = np.diag([2.0, 2.0, 2.0, 1.0])
    data = _data(affines={"lab": lab})
    other = vd.create_voxeldata(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        voxel_to_world=np.eye(4),
        affines={"lab": np.eye(4)},
    )

    np.testing.assert_allclose(data.vd.to(other, "lab"), np.linalg.inv(np.eye(4)) @ lab)


def test_attrs_affines_are_ignored() -> None:
    data = _data()
    data.attrs["affines"] = {"lab": np.eye(4)}

    with pytest.raises(KeyError, match="lab"):
        data.vd.apply("lab")
    assert data.vd.affines == {}


def test_grid_selection_reduces_named_affines() -> None:
    first = np.eye(4)
    first[:3, 3] = [1.0, 0.0, 0.0]
    second = np.eye(4)
    second[:3, 3] = [2.0, 0.0, 0.0]
    data = vd.create_voxeldata(
        np.zeros((2, 2, 3, 4)),
        dims=("grid", "k", "j", "i"),
        grid=np.arange(2),
        voxel_to_world=np.stack([first, second]),
        affines={"lab": np.stack([first, second])},
    )

    selected = data.isel(grid=0)

    assert not selected.vd.is_grid_dependent
    np.testing.assert_allclose(selected.vd.get_affine("lab"), first)


def test_concat_merges_named_affines() -> None:
    first = np.eye(4)
    first[:3, 3] = [1.0, 0.0, 0.0]
    second = np.eye(4)
    second[:3, 3] = [2.0, 0.0, 0.0]
    shared = np.diag([2.0, 2.0, 2.0, 1.0])

    def make(affine: np.ndarray, grid: int) -> xr.DataArray:
        return vd.create_voxeldata(
            np.zeros((1, 2, 3, 4)),
            dims=("grid", "k", "j", "i"),
            grid=np.array([grid]),
            voxel_to_world=affine[np.newaxis],
            affines={"lab": affine[np.newaxis], "shared": shared},
        )

    combined = xr.concat([make(first, 0), make(second, 1)], dim="grid")

    np.testing.assert_allclose(combined.vd.get_affine("lab"), np.stack([first, second]))
    np.testing.assert_allclose(combined.vd.get_affine("shared"), shared)


def test_validate_canonicalize_and_ensure_delegate() -> None:
    data = _data()

    assert data.vd.validate() is None
    assert data.vd.canonicalize().dims == ("k", "j", "i")
    assert data.vd.ensure().dims == ("k", "j", "i")

    invalid = data.isel(k=0).drop_indexes(["z", "y", "x"])
    with pytest.raises(ValueError):
        invalid.vd.validate()
