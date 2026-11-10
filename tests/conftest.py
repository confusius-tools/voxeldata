"""Small, function-scoped geometry fixtures shared by the public API tests.

Distinct sample values make incorrect selection or reordering visible; each test
gets fresh mutable arrays.
"""

from collections.abc import Iterator

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.fixture(params=[False, True])
def label_backend(request: pytest.FixtureRequest) -> Iterator[bool]:
    """Exercise native and accelerated Xarray group reductions.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Whether to enable flox acceleration for this test.

    Yields
    ------
    bool
        Whether flox is enabled.
    """
    with xr.set_options(use_flox=request.param):
        yield request.param


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
    """Return a sheared affine for oblique volumes.

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
def view_affines(voxel_affine: np.ndarray) -> np.ndarray:
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
def sample_voxeldata_views(
    plain_volume: xr.DataArray, view_affines: np.ndarray
) -> xr.DataArray:
    """Return two labelled grids with separate geometry and sample values.

    Parameters
    ----------
    plain_volume : xarray.DataArray
        Spatial samples and voxel labels.
    view_affines : numpy.ndarray
        One voxel-to-world matrix per grid.

    Returns
    -------
    xarray.DataArray
        Two grids labelled `10` and `20`, with values from zero to forty-seven.
    """
    data = plain_volume.expand_dims(view=[10, 20]).copy(deep=True)
    data.data = np.arange(48, dtype=float).reshape(2, 2, 3, 4)
    return vd.attach_voxel_to_world_index(data, view_affines, geometry_dims=("view",))


@pytest.fixture
def sample_voxeldata_view_time(sample_voxeldata_views: xr.DataArray) -> xr.DataArray:
    """Return grids with independent, regularly sampled timestamps.

    Parameters
    ----------
    sample_voxeldata_views : xarray.DataArray
        Spatial grid stack.

    Returns
    -------
    xarray.DataArray
        `(time, view, k, j, i)` samples with a `(time, view)` time coordinate.
    """
    result = sample_voxeldata_views.expand_dims(time=3).copy(deep=True)
    result.data = np.arange(144, dtype=float).reshape(3, 2, 2, 3, 4)
    return result.assign_coords(
        time=(("time", "view"), [[0.0, 10.0], [0.5, 10.5], [1.0, 11.0]], {"units": "s"})
    )


@pytest.fixture
def sample_voxeldata_irregular(plain_volume: xr.DataArray) -> xr.DataArray:
    """Return an irregular voxel-label selection fixture.

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


@pytest.fixture
def make_sample_voxeldata_mask(sample_voxeldata_3d):
    """Return a factory for empty spatial masks on the shared sample grid.

    Parameters
    ----------
    sample_voxeldata_3d : xarray.DataArray
        Reference native voxel labels, geometry, and units.

    Returns
    -------
    callable
        Factory accepting a dtype and returning a fresh zero-filled mask.
    """

    def make_mask(dtype=bool):
        """Create an empty mask without sharing mutable sample data.

        Parameters
        ----------
        dtype : type, default: bool
            Mask value dtype.

        Returns
        -------
        xarray.DataArray
            Independent mask on the reference voxel grid.
        """
        return sample_voxeldata_3d.copy(
            data=np.zeros(sample_voxeldata_3d.shape, dtype=dtype)
        )

    return make_mask


@pytest.fixture
def multiaxis_timestamps() -> xr.DataArray:
    """Return independently offset timestamps for two nonspatial dependencies.

    Returns
    -------
    xarray.DataArray
        Three time points for each view/channel pair, with clock metadata.
    """
    values = (
        np.arange(3.0)[:, None, None]
        + np.array([0.0, 0.1])[None, :, None]
        + np.array([0.0, 0.3])[None, None, :]
    )
    return xr.DataArray(
        values,
        dims=("time", "view", "channel"),
        attrs={"units": "s", "clock": "device"},
    )


@pytest.fixture
def multigeometry_data(
    oblique_affine: np.ndarray, named_affines: dict[str, np.ndarray]
) -> xr.DataArray:
    """Return oblique geometry indexed by time and view, but not channel.

    Parameters
    ----------
    oblique_affine : numpy.ndarray
        Nonzero, sheared geometry.
    named_affines : dict[str, numpy.ndarray]
        Noncommuting named world frames.

    Returns
    -------
    xarray.DataArray
        Small labelled data with two independent geometry axes and named frames.
    """
    affines = np.broadcast_to(oblique_affine, (3, 2, 4, 4)).copy()
    affines[:, 1, :3, :] = (
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ affines[:, 1, :3, :]
    )
    frames = np.broadcast_to(named_affines["lab"], (3, 2, 4, 4)).copy()
    for time in range(3):
        for view in range(2):
            affines[time, view, :3, 3] += [10 * time, 20 * view, 5 * (time + view)]
            frames[time, view, :3, 3] += [3 * time, 7 * view, 11 * (time + view)]
    return vd.create_voxeldata(
        np.arange(2 * 3 * 2 * 2 * 3 * 4).reshape(2, 3, 2, 2, 3, 4),
        dims=("channel", "time", "view", "k", "j", "i"),
        time=[1.0, 2.0, 4.0],
        extra_coords={"channel": [10, 30], "view": [11, 41]},
        k=[2, 4],
        j=[3, 6, 9],
        i=[5, 7, 9, 11],
        voxel_to_world=affines,
        geometry_dims=("time", "view"),
        affines={
            "lab": frames,
            "atlas": named_affines["atlas"],
        },
    )
