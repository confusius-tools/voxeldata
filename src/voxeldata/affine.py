"""Xarray accessor for affine transform operations."""

import itertools
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from voxeldata._dims import GRID_DIM, WORLD_DIMS
from voxeldata.geometry import (
    attach_voxel_to_world_index,
    get_affine_axis_scalings,
    get_voxel_to_world_affine,
    get_voxel_to_world_direction_matrix,
    get_voxel_to_world_index,
    get_voxel_to_world_index_origin,
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spatial_dims,
    get_voxel_to_world_units,
    has_voxel_to_world_index,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from voxeldata.geometry import VoxelToWorldIndex


def get_bounding_box(data: xr.DataArray) -> xr.DataArray:
    """Return the world-space bounding box of a VoxelData array.

    The box encloses the full extent of every voxel, not just the voxel centers: the
    voxel-coordinate range of each `k`/`j`/`i` dim is extended by half a voxel at
    both ends before mapping through the voxel-to-world affine. The voxel size is
    [`get_voxel_to_world_index_spacing`][voxeldata.geometry.get_voxel_to_world_index_spacing] (so a singleton dim spans
    one voxel, and irregular voxel coordinates are rejected). The result is exact for
    any affine, including oblique ones, without
    materializing the lazily derived world coordinates: each world coordinate is an
    affine function of the voxel coordinates, so its extremes over the grid are
    attained at the corners of the (extended) voxel-coordinate box.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, normalized with
        [ensure_voxeldata][voxeldata.validation.ensure_voxeldata] (scalar-indexed
        voxel dims are restored as singletons before computing the bounds).

    Returns
    -------
    (2, 3) xarray.DataArray
        Bounding box with dims `(bound, component)`, where `bound` is
        `["min", "max"]` and `component` is `["z", "y", "x"]`, and a `units` attr
        from the voxel-to-world index. Grid-dependent geometry adds a leading
        `grid` dim carrying the input's `grid` coordinate, one bounding box per
        grid.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If `data` is not a valid VoxelData array, or if a voxel dim has irregular
        coordinates (no defined voxel extent).

    Examples
    --------
    >>> bbox = get_bounding_box(volume)
    >>> bbox.sel(bound="min", component="z").item()
    """
    from voxeldata.validation import ensure_voxeldata

    data = ensure_voxeldata(data)
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    affine = get_voxel_to_world_affine(data)

    # Half a voxel per dim in voxel-coordinate units: the world spacing (world length per
    # voxel, `None` for irregular coordinates) divided by the affine column norm
    # (world length per voxel-coordinate unit).
    scalings = get_affine_axis_scalings(affine, voxel_dims)
    half_steps: dict[str, float] = {}
    for dim, world_spacing in get_voxel_to_world_index_spacing(data).items():
        if world_spacing is None:
            raise ValueError(
                f"Voxel dim '{dim}' has irregular coordinates; the voxel extent, and "
                "therefore the bounding box, is undefined."
            )
        half_steps[dim] = 0.5 * world_spacing / scalings[dim]

    # The 8 corners of the voxel-coordinate box extended by half a voxel per dim, as
    # homogeneous column vectors.
    corners = np.array(
        list(
            itertools.product(
                *(
                    (
                        data.coords[dim].values.min() - half_steps[dim],
                        data.coords[dim].values.max() + half_steps[dim],
                    )
                    for dim in voxel_dims
                )
            )
        ),
        dtype=np.float64,
    )
    corners_h = np.hstack([corners, np.ones((len(corners), 1))]).T

    # (ngrid, 4, 4) @ (4, 8) broadcasts to one corner set per grid.
    world_corners = (affine @ corners_h)[..., :-1, :]
    bounds = np.stack([world_corners.min(axis=-1), world_corners.max(axis=-1)], axis=-2)

    coords: dict[str, npt.ArrayLike] = {
        "bound": np.array(["min", "max"], dtype=np.str_),
        "component": np.array(WORLD_DIMS, dtype=np.str_),
    }
    dims = ("bound", "component")
    if affine.ndim == 3:
        dims = (GRID_DIM, *dims)
        coords[GRID_DIM] = data.coords[GRID_DIM].values
    return xr.DataArray(
        bounds,
        dims=dims,
        coords=coords,
        attrs={"units": get_voxel_to_world_units(data)},
    )


def get_relative_affine(
    da: xr.DataArray,
    other: xr.DataArray,
    via: str,
) -> "npt.NDArray[np.float64]":
    """Return the affine mapping `da`'s world space into `other`'s.

    Computes `inv(other_named[via]) @ da_named[via]`, giving the transform that
    takes coordinates expressed in `da`'s world frame and expresses them in
    `other`'s world frame. Both arrays must carry a named affine frame `via` on
    their `VoxelToWorldIndex`.

    Parameters
    ----------
    da : xarray.DataArray
        The source scan (origin world space).
    other : xarray.DataArray
        The scan whose world space is the target.
    via : str
        Named affine frame shared by both world spaces (e.g. `"world_to_lab"`).

    Returns
    -------
    (4, 4) numpy.ndarray
        Homogeneous affine matrix mapping `da`'s world coordinates to `other`'s
        world coordinates.

    Raises
    ------
    ValueError
        If either array lacks voxel-to-world geometry.
    KeyError
        If `via` is not a named affine on either array's index.
    """
    self_index = _require_voxel_to_world_index(da)
    other_index = _require_voxel_to_world_index(other)
    if via not in self_index.named_affines:
        raise KeyError(f"'{via}' not found in da's named affines.")
    if via not in other_index.named_affines:
        raise KeyError(f"'{via}' not found in other's named affines.")
    self_affine = np.asarray(self_index.named_affines[via], dtype=np.float64)
    other_affine = np.asarray(other_index.named_affines[via], dtype=np.float64)
    return np.linalg.inv(other_affine) @ self_affine


def _require_voxel_to_world_index(da: xr.DataArray) -> "VoxelToWorldIndex":
    """Return `da`'s voxel-to-world index or raise.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    VoxelToWorldIndex
        The attached index.

    Raises
    ------
    ValueError
        If `da` has no voxel-to-world index.
    """
    index = get_voxel_to_world_index(da)
    if index is None:
        raise ValueError("DataArray must have a voxel-to-world index.")
    return index


def apply_affine(
    da: xr.DataArray,
    affine: "npt.NDArray[np.float64] | str",
    inplace: bool = False,
) -> xr.DataArray:
    """Apply a world-space affine to a DataArray's world coordinates.

    The transform is composed into the DataArray's `VoxelToWorldIndex`, derived
    world coordinates are regenerated, and named affine frames on that index are
    re-expressed against the new world frame. Legacy `attrs["affines"]` entries are
    ignored entirely.

    Parameters
    ----------
    da : xarray.DataArray
        Input scan with voxel-to-world geometry (a `VoxelToWorldIndex`).
    affine : (4, 4) numpy.ndarray or str
        Homogeneous world-space affine matrix to apply. If a string, it is looked up
        among the index's named affine frames.
    inplace : bool, default: False
        Whether to modify the DataArray in-place.

    Returns
    -------
    xarray.DataArray
        `da` with updated spatial coordinates and updated named affine frames. No
        frame is removed: when `affine` is a string, that frame is kept as the
        identity, since the world frame now is that named space.

    Raises
    ------
    ValueError
        If `da` lacks voxel-to-world geometry or if `affine`'s shape does not match
        the DataArray's voxel-to-world affine.
    KeyError
        If `affine` is a string not present among the index's named affines.

    Examples
    --------
    >>> import numpy as np
    >>> from voxeldata import create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((3, 4, 5)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> shift = np.eye(4)
    >>> shift[:3, 3] = [10.0, 5.0, 0.0]
    >>> result = apply_affine(data, shift)
    >>> float(result.coords["x"].isel(k=0, j=0, i=0))
    10.0
    """
    index = _require_voxel_to_world_index(da)
    applied_key = affine if isinstance(affine, str) else None
    if applied_key is not None:
        if applied_key not in index.named_affines:
            raise KeyError(f"'{applied_key}' not found among named affines.")
        affine = index.named_affines[applied_key]
    affine_array = np.asarray(affine, dtype=np.float64)

    voxel_to_world = get_voxel_to_world_affine(da)
    is_grid_stacked = voxel_to_world.ndim == 3
    # A single (4, 4) world-space affine broadcasts over every grid (numpy's `@`
    # already does this correctly: `affine_array @ voxel_to_world[p]` per grid); a
    # grid-dependent affine_array must match the stack length exactly instead.
    valid_shape = (
        affine_array.shape == voxel_to_world.shape[1:]
        or affine_array.shape == voxel_to_world.shape
        if is_grid_stacked
        else affine_array.shape == voxel_to_world.shape
    )
    if not valid_shape:
        expected = (
            f"{voxel_to_world.shape[1:]} or {voxel_to_world.shape}"
            if is_grid_stacked
            else f"{voxel_to_world.shape}"
        )
        raise ValueError(
            "voxel-to-world data requires an affine with shape matching "
            f"voxel_to_world ({expected}), got {affine_array.shape}."
        )

    inv_affine = np.linalg.inv(affine_array)
    new_named: dict[str, npt.NDArray[np.float64]] = {}
    for stored_key, val in index.named_affines.items():
        if stored_key == applied_key:
            # The world frame now is this named space. Store an exact identity
            # instead of `arr @ inv(arr)`, which carries floating-point error.
            new_named[stored_key] = np.eye(affine_array.shape[-1])
            continue
        new_named[stored_key] = np.asarray(val, dtype=np.float64) @ inv_affine

    result = attach_voxel_to_world_index(
        da,
        affine_array @ voxel_to_world,
        units=get_voxel_to_world_units(da),
        named_affines=new_named,
    )
    if inplace:
        da.coords.update(result.coords)
        da.attrs.clear()
        da.attrs.update(result.attrs)
        return da
    return result


def reindex_voxels(da: xr.DataArray) -> xr.DataArray:
    """Rebase voxel coordinates to dense positions without moving world coordinates.

    A VoxelData array's stored `voxel_to_world` affine is defined in
    terms of voxel *coordinate values*, which stay unchanged across cropping or
    striding by design (see
    `VoxelToWorldIndex`).
    Because of this, the affine generally does not describe where voxel *position*
    `(0, ..., 0)` sits in world space, or the world distance between
    consecutive positions, once `da` has been cropped or strided from a larger
    array. This replaces each voxel dimension's coordinate with `0, 1, ..., dim - 1`
    and rebuilds `voxel_to_world` so the resulting affine directly maps those dense
    positions to `da`'s existing world coordinates, producing a DataArray whose
    affine is directly usable by software that assumes dense, zero-based voxel
    indices (e.g. ITK, nilearn).

    Parameters
    ----------
    da : xarray.DataArray
        Input scan with voxel-to-world geometry.

    Returns
    -------
    xarray.DataArray
        `da` with voxel coordinates rebased to `0, 1, ..., dim - 1` and an updated
        `voxel_to_world` affine. World coordinates are unchanged. For grid-dependent
        geometry, `voxel_to_world` stays a per-grid stack: each grid keeps its own
        origin and direction (spacing is shared, per the equal-scale invariant), so
        rebasing is unambiguous per grid and every grid is reindexed at once.

    Raises
    ------
    ValueError
        If `da` lacks voxel-to-world geometry, or if world spacing is undefined for
        any voxel dimension.
    """
    if not has_voxel_to_world_index(da):
        raise ValueError("DataArray must have a voxel-to-world index.")

    voxel_dims = get_voxel_to_world_spatial_dims(da)

    spacing = get_voxel_to_world_index_spacing(da)
    missing_spacing = [dim for dim in voxel_dims if spacing[dim] is None]
    if missing_spacing:
        raise ValueError(
            f"Cannot reindex voxels because spacing is undefined for dimensions "
            f"{missing_spacing!r}."
        )
    spacing_diag = np.diag([spacing[dim] for dim in voxel_dims])
    ndim = len(voxel_dims)

    def _new_grid_affine(grid_da: xr.DataArray) -> "npt.NDArray[np.float64]":
        origin = get_voxel_to_world_index_origin(grid_da)
        # Already expressed in dense-position terms (accounts for a descending
        # voxel coordinate, e.g. after `.isel(dim=slice(None, None, -1))`),
        # matching how `spacing` (a magnitude) and dense array position combine
        # below.
        direction = get_voxel_to_world_direction_matrix(grid_da)
        affine = np.eye(ndim + 1, dtype=np.float64)
        affine[:ndim, :ndim] = direction @ spacing_diag
        affine[:ndim, ndim] = [origin[name] for name in WORLD_DIMS]
        return affine

    if GRID_DIM in da.dims and get_voxel_to_world_affine(da).ndim == 3:
        new_affine = np.stack(
            [
                _new_grid_affine(da.isel({GRID_DIM: grid}))
                for grid in range(da.sizes[GRID_DIM])
            ]
        )
    else:
        new_affine = _new_grid_affine(da)

    reindexed = da.assign_coords({dim: np.arange(da.sizes[dim]) for dim in voxel_dims})
    return attach_voxel_to_world_index(
        reindexed,
        new_affine,
        units=get_voxel_to_world_units(da),
        named_affines=_require_voxel_to_world_index(da).named_affines,
    )


def reindex_voxels_like(
    data: xr.DataArray, reference: xr.DataArray, *, atol: float = 1e-6
) -> xr.DataArray:
    """Rebase voxel coordinates onto `reference`'s voxel labels.

    `data` and `reference`'s `voxel_to_world` affines can differ even when they describe
    the exact same world grid: the affine is defined in terms of voxel *coordinate
    values*, so two arrays occupying identical world positions can still carry
    different affines if their voxel dimensions happen to be labeled differently (e.g.
    `reference` was cropped or strided from a larger array, while `data` was freshly
    built with dense labels). This verifies the two occupy the same world grid, then
    relabels `data`'s voxel coordinates and affine to match `reference`'s exactly, so
    the two become directly alignable (`.sel()`, arithmetic, `xarray.align`, ...) by
    voxel label as well as by world position.

    Parameters
    ----------
    data : xarray.DataArray
        Input scan with voxel-to-world geometry, physically aligned with `reference`.
    reference : xarray.DataArray
        DataArray whose voxel labels and affine `data` should adopt.
    atol : float, default: 1e-6
        Absolute tolerance, in `reference`'s physical units, for the world-coordinate
        alignment check between `data` and `reference`.

    Returns
    -------
    xarray.DataArray
        `data` with voxel coordinates and `voxel_to_world` replaced by `reference`'s.
        World coordinates are unchanged, since `data` and `reference` are verified to
        already occupy the same world grid.

    Raises
    ------
    ValueError
        If `data` or `reference` lacks voxel-to-world geometry, if their voxel
        dimensions or shapes differ, or if their world coordinates do not match
        within `atol`.
    """
    if not has_voxel_to_world_index(data):
        raise ValueError("data must have a voxel-to-world index.")
    if not has_voxel_to_world_index(reference):
        raise ValueError("reference must have a voxel-to-world index.")

    voxel_dims = get_voxel_to_world_spatial_dims(data)
    reference_voxel_dims = get_voxel_to_world_spatial_dims(reference)
    if voxel_dims != reference_voxel_dims:
        raise ValueError(
            f"data and reference must have the same voxel dimensions; got "
            f"{voxel_dims!r} and {reference_voxel_dims!r}."
        )
    shape = tuple(data.sizes[dim] for dim in voxel_dims)
    reference_shape = tuple(reference.sizes[dim] for dim in voxel_dims)
    if shape != reference_shape:
        raise ValueError(
            f"data and reference must have the same voxel grid shape to reindex; "
            f"got {shape!r} and {reference_shape!r}."
        )

    for name in WORLD_DIMS:
        data_values = data.coords[name].transpose(*voxel_dims).values
        reference_values = reference.coords[name].transpose(*voxel_dims).values
        if not np.allclose(data_values, reference_values, atol=atol):
            raise ValueError(
                f"data and reference are not aligned in world space: coordinate "
                f"{name!r} differs by more than {atol}. reindex_voxels_like requires "
                "data to already occupy reference's exact world grid."
            )

    relabeled = data.assign_coords({dim: reference.coords[dim] for dim in voxel_dims})
    return attach_voxel_to_world_index(
        relabeled,
        get_voxel_to_world_affine(reference),
        units=get_voxel_to_world_units(reference),
        named_affines=_require_voxel_to_world_index(reference).named_affines,
    )
