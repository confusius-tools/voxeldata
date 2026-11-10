"""Public geometry helper contracts, independent of acquisition semantics.

These regressions are maintained here rather than in downstream re-export tests.
"""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from voxeldata.geometry import (
    VoxelToWorldTransform,
    attach_voxel_to_world_index,
    get_affine_axis_scalings,
    get_affine_direction_matrix,
    get_voxel_to_world_affine,
    get_voxel_to_world_index_origin,
    get_voxel_to_world_spacings_from_coords,
    get_voxel_to_world_spatial_dims,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
)


def test_affine_geometry_helpers_extract_scalings_and_orientation() -> None:
    """Affine geometry helpers expose the linear part in world-space form."""
    voxel_to_world = np.array(
        [
            [2.0, 1.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    scalings = get_affine_axis_scalings(voxel_to_world, ("k", "j", "i"))
    assert scalings.keys() == {"k", "j", "i"}
    assert_allclose(scalings["k"], 2.0)
    assert_allclose(scalings["j"], np.sqrt(10.0))
    assert_allclose(scalings["i"], 4.0)
    expected_orientation = np.array(
        [
            [1.0, 1.0 / np.sqrt(10.0), 0.0],
            [0.0, 3.0 / np.sqrt(10.0), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert_allclose(get_affine_direction_matrix(voxel_to_world), expected_orientation)


def test_get_world_spacings_singleton_axis_uses_affine_column_norm() -> None:
    """Singleton voxel axes still have a world per-voxel spacing from the affine."""
    voxel_coords = {
        "k": [0],
        "j": [0, 2, 4],
        "i": [0, 1, 2, 3],
    }
    voxel_to_world = np.array(
        [
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_voxel_to_world_spacings_from_coords(voxel_coords, voxel_to_world)

    assert spacing == {"k": 0.4, "j": 6.0, "i": 5.0}


def test_get_world_spacings_returns_none_for_irregular_voxel_axes() -> None:
    """World spacing is undefined when voxel-space sampling is irregular."""
    voxel_coords = {
        "k": [0, 1, 2],
        "j": [0, 2, 4],
        "i": [0, 1, 3, 4],
    }
    voxel_to_world = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    spacing = get_voxel_to_world_spacings_from_coords(voxel_coords, voxel_to_world)

    assert spacing == {"k": 2.0, "j": 6.0, "i": None}


def test_get_voxel_to_world_origin_uses_first_sampled_voxel() -> None:
    """Voxel-to-world origin is the world location of array index zero."""
    data = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={
            "k": [10, 11],
            "j": [5, 7, 9],
            "i": [100, 101, 102, 103],
        },
    )
    data = attach_voxel_to_world_index(
        data,
        np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )

    assert get_voxel_to_world_index_origin(data) == {"z": 30.0, "y": 35.0, "x": 430.0}


def test_voxel_to_world_transform_rejects_invalid_construction() -> None:
    """`VoxelToWorldTransform` validates its constructor arguments."""
    valid_coords = {"k": np.arange(2), "j": np.arange(3), "i": np.arange(4)}

    with pytest.raises(ValueError, match="at least one active dim"):
        VoxelToWorldTransform({}, np.eye(4))

    with pytest.raises(ValueError, match="must exactly cover"):
        VoxelToWorldTransform({"j": np.arange(3), "i": np.arange(4)}, np.eye(3))

    with pytest.raises(ValueError, match="must exactly cover"):
        VoxelToWorldTransform(
            {**valid_coords, "t": np.arange(2.0)},
            np.eye(5),
        )

    with pytest.raises(ValueError, match="must be 1D"):
        VoxelToWorldTransform(
            {"k": np.arange(2), "j": np.arange(3), "i": np.zeros((2, 2))}, np.eye(4)
        )

    with pytest.raises(ValueError, match=r"must have shape \(4, 4\)"):
        VoxelToWorldTransform(valid_coords, np.eye(3))


def test_add_world_coords_validates_voxel_dims_and_coordinates() -> None:
    """`attach_voxel_to_world_index` validates dims and their coordinates."""
    data = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("k", "j", "i"),
        coords={"k": np.arange(2), "j": np.arange(3), "i": np.arange(4)},
    )

    with pytest.raises(ValueError, match="must have a matching 1D coordinate"):
        attach_voxel_to_world_index(data.drop_vars("i"), np.eye(4))

    non_dim_coord = data.assign_coords(
        j=(("j", "i"), np.zeros((3, 4))),
    )
    with pytest.raises(ValueError, match="must be a 1D dimension coordinate"):
        attach_voxel_to_world_index(non_dim_coord, np.eye(4))

    no_voxel_dims = xr.DataArray(np.zeros((3, 4)), dims=("a", "b"))
    with pytest.raises(ValueError, match="must have all native voxel dims"):
        attach_voxel_to_world_index(no_voxel_dims, np.eye(4))


def test_get_voxel_to_world_affine_raises_without_voxel_to_world_geometry() -> None:
    """`get_voxel_to_world_affine` raises for a DataArray without a voxel-to-world index."""
    data = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"))

    with pytest.raises(ValueError, match="must have a voxel-to-world index"):
        get_voxel_to_world_affine(data)


def test_has_voxel_to_world_index_false_without_index() -> None:
    """A plain DataArray without voxel-to-world geometry returns `False`."""
    plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i"))

    assert has_voxel_to_world_index(plain) is False


def test_has_axis_aligned_voxel_to_world_index_false_without_index() -> None:
    """`has_axis_aligned_voxel_to_world_index` is `False` for a plain DataArray with
    no `VoxelToWorldIndex` attached at all, not just for an oblique one.
    """
    plain = xr.DataArray(np.zeros((2, 3, 4)), dims=("k", "j", "i"))

    assert has_axis_aligned_voxel_to_world_index(plain) is False


def test_get_voxel_to_world_spatial_dims_falls_back_without_index() -> None:
    """Without a `VoxelToWorldIndex`, `get_voxel_to_world_spatial_dims` falls back
    to whichever native voxel dims are present on `data.dims`.
    """
    plain = xr.DataArray(np.zeros((2, 4)), dims=("k", "i"))

    assert get_voxel_to_world_spatial_dims(plain) == ("k", "i")
