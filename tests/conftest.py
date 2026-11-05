"""Small, function-scoped geometry fixtures shared by the public API tests.

Adapted from ConfUSIus's unit fixtures. Distinct sample values make incorrect
selection or reordering visible; each test gets fresh mutable arrays.
"""

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.fixture
def voxel_affine() -> np.ndarray:
    """Return an anisotropic affine with a nonzero origin.

    Returns
    -------
    numpy.ndarray
        Matrix mapping `(k, j, i)` to `(k + 10, 2*j + 20, 3*i + 30)`.
    """
    return np.array(
        [[1, 0, 0, 10], [0, 2, 0, 20], [0, 0, 3, 30], [0, 0, 0, 1]], dtype=float
    )


@pytest.fixture
def oblique_affine() -> np.ndarray:
    """Return the sheared affine used by ConfUSIus's oblique-volume fixture.

    Returns
    -------
    numpy.ndarray
        Matrix with off-diagonal terms, unequal scales, and translation.
    """
    return np.array(
        [[0.4, 0.0, 0.1, 10], [0.1, 0.3, 0.0, 20], [0.0, 0.05, 0.25, 30], [0, 0, 0, 1]]
    )


@pytest.fixture
def named_affines() -> dict[str, np.ndarray]:
    """Return noncommuting world frames for composition-order tests.

    Returns
    -------
    dict[str, numpy.ndarray]
        A rotated/translated lab frame and an anisotropically scaled atlas frame.
    """
    return {
        "lab": np.array(
            [[0, -1, 0, 5], [1, 0, 0, 7], [0, 0, 1, 11], [0, 0, 0, 1]], dtype=float
        ),
        "atlas": np.array(
            [[2, 0, 0, 13], [0, 3, 0, 17], [0, 0, 4, 19], [0, 0, 0, 1]], dtype=float
        ),
    }


@pytest.fixture
def plain_volume() -> xr.DataArray:
    """Return a labelled volume without a voxel-to-world index.

    Returns
    -------
    xarray.DataArray
        Distinct samples in a `(2, 3, 4)` volume with dense integer voxel labels.
    """
    return xr.DataArray(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        dims=("k", "j", "i"),
        coords={"k": [0, 1], "j": [0, 1, 2], "i": [0, 1, 2, 3]},
        name="intensity",
        attrs={"units": "a.u."},
    )


@pytest.fixture
def sample_voxeldata_3d(
    plain_volume: xr.DataArray, voxel_affine: np.ndarray
) -> xr.DataArray:
    """Attach anisotropic geometry without exercising the constructor.

    Parameters
    ----------
    plain_volume : xarray.DataArray
        Labelled spatial samples.
    voxel_affine : numpy.ndarray
        Anisotropic voxel-to-world matrix.

    Returns
    -------
    xarray.DataArray
        Volume with origin `(10, 20, 30)` and spacing `(1, 2, 3)`.
    """
    return vd.attach_voxel_to_world_index(plain_volume, voxel_affine)


@pytest.fixture
def sample_voxeldata_3dt(sample_voxeldata_3d: xr.DataArray) -> xr.DataArray:
    """Return four distinguishable time points on the shared spatial grid.

    Parameters
    ----------
    sample_voxeldata_3d : xarray.DataArray
        Spatial volume to repeat across time.

    Returns
    -------
    xarray.DataArray
        Time series sampled every half second starting at ten seconds.
    """
    result = sample_voxeldata_3d.expand_dims(time=[10.0, 10.5, 11.0, 11.5]).copy(
        deep=True
    )
    result.data = np.arange(96, dtype=float).reshape(4, 2, 3, 4)
    result.coords["time"].attrs["units"] = "s"
    return result


@pytest.fixture
def sample_voxeldata_3d_oblique(
    plain_volume: xr.DataArray, oblique_affine: np.ndarray
) -> xr.DataArray:
    """Attach sheared geometry to the shared spatial samples.

    Parameters
    ----------
    plain_volume : xarray.DataArray
        Labelled spatial samples.
    oblique_affine : numpy.ndarray
        Sheared voxel-to-world matrix.

    Returns
    -------
    xarray.DataArray
        Oblique volume with distinguishable samples.
    """
    return vd.attach_voxel_to_world_index(plain_volume, oblique_affine)


@pytest.fixture
def grid_affines(voxel_affine: np.ndarray) -> np.ndarray:
    """Return two anisotropic grids separated by a translation.

    Parameters
    ----------
    voxel_affine : numpy.ndarray
        First grid's matrix.

    Returns
    -------
    numpy.ndarray
        Affine stack with origins `(10, 20, 30)` and `(110, 0, 35)`.
    """
    second = voxel_affine.copy()
    second[:3, 3] += [100, -20, 5]
    return np.stack([voxel_affine, second])


@pytest.fixture
def sample_voxeldata_grids(
    plain_volume: xr.DataArray, grid_affines: np.ndarray
) -> xr.DataArray:
    """Return two labelled grids with separate geometry and sample values.

    Parameters
    ----------
    plain_volume : xarray.DataArray
        Spatial samples and voxel labels.
    grid_affines : numpy.ndarray
        One voxel-to-world matrix per grid.

    Returns
    -------
    xarray.DataArray
        Two grids labelled `10` and `20`, with values from zero to forty-seven.
    """
    data = plain_volume.expand_dims(grid=[10, 20]).copy(deep=True)
    data.data = np.arange(48, dtype=float).reshape(2, 2, 3, 4)
    return vd.attach_voxel_to_world_index(data, grid_affines)


@pytest.fixture
def sample_voxeldata_grid_time(sample_voxeldata_grids: xr.DataArray) -> xr.DataArray:
    """Return grids with independent, regularly sampled timestamps.

    Parameters
    ----------
    sample_voxeldata_grids : xarray.DataArray
        Spatial grid stack.

    Returns
    -------
    xarray.DataArray
        `(time, grid, k, j, i)` samples with a `(time, grid)` time coordinate.
    """
    result = sample_voxeldata_grids.expand_dims(time=3).copy(deep=True)
    result.data = np.arange(144, dtype=float).reshape(3, 2, 2, 3, 4)
    return result.assign_coords(
        time=(("time", "grid"), [[0.0, 10.0], [0.5, 10.5], [1.0, 11.0]], {"units": "s"})
    )


@pytest.fixture
def sample_voxeldata_irregular(plain_volume: xr.DataArray) -> xr.DataArray:
    """Return ConfUSIus's irregular voxel-label selection fixture.

    Parameters
    ----------
    plain_volume : xarray.DataArray
        Spatial samples to relabel before attaching geometry.

    Returns
    -------
    xarray.DataArray
        Irregular integer labels with world scales `(10, 2, 3)`.
    """
    data = plain_volume.assign_coords(k=[0, 2], j=[0, 1, 3], i=[0, 2, 3, 7])
    return vd.attach_voxel_to_world_index(data, np.diag([10.0, 2.0, 3.0, 1.0]))
