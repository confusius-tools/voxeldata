"""Xarray accessor for affine transform operations."""

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import xarray as xr

from voxeldata._coordinates import get_representative_step
from voxeldata._dims import VOXEL_DIMS, WORLD_DIMS
from voxeldata.geometry import (
    attach_voxel_to_world_index,
    get_array_origin,
    get_voxel_to_world_affine,
    get_voxel_to_world_index,
    get_voxel_to_world_spatial_dims,
    get_voxel_to_world_units,
    has_voxel_to_world_index,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from voxeldata.geometry import VoxelToWorldIndex


def get_bounding_box(data: xr.DataArray) -> dict[str, tuple[float, float]]:
    """Return the world-space bounding box of a VoxelData array.

    The box encloses the full extent of every voxel, not just the voxel centers: the
    voxel-coordinate range of each `k`/`j`/`i` dim is extended by half a voxel at
    both ends before mapping through the voxel-to-world affine. The voxel size is
    [`get_array_spacing`][voxeldata.geometry.get_array_spacing] (so a singleton dim spans
    one voxel, and irregular voxel coordinates are rejected). The result is exact for
    any affine, including oblique ones, without
    materializing the lazily derived world coordinates: each world coordinate is an
    affine function of the voxel coordinates, so its extremes over the grid are
    attained at the corners of the (extended) voxel-coordinate box. For stacked
    geometry, the box encloses all grids. Select geometry dependencies first for
    a grid-specific bounding box.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, normalized with
        [ensure_voxeldata][voxeldata.validation.ensure_voxeldata] (scalar-indexed
        voxel dims are restored as singletons before computing the bounds).

    Returns
    -------
    dict[str, tuple[float, float]]
        World coordinate names `z`, `y`, `x` mapped to `(min, max)` bounds in
        `.vd.world_units`. The bounds enclose all geometries.

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
    >>> bbox["z"][0]
    """
    from voxeldata.validation import ensure_voxeldata

    data = ensure_voxeldata(data)
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    affine = get_voxel_to_world_affine(data)

    # Half a voxel per dim in voxel-coordinate units: the world spacing (world length per
    # voxel, `None` for irregular coordinates) divided by the affine column norm
    # (world length per voxel-coordinate unit).
    half_steps = {}
    for dim in voxel_dims:
        step, approximate = get_representative_step(data.coords[dim].values)
        if approximate:
            raise ValueError(
                f"Voxel dim {dim!r} has irregular coordinates; its bounding box is undefined."
            )
        half_steps[dim] = 0.5 * (abs(step) if step is not None else 1.0)

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

    # (*geometry_shape, 4, 4) @ (4, 8) broadcasts to one corner set per geometry.
    world_corners = (affine @ corners_h)[..., :-1, :]
    # Reduce every geometry axis and the corner axis, retaining world components.
    axes = (*range(world_corners.ndim - 2), -1)
    bounds = np.stack([world_corners.min(axis=axes), world_corners.max(axis=axes)])

    return {
        dim: (float(bounds[0, axis]), float(bounds[1, axis]))
        for axis, dim in enumerate(WORLD_DIMS)
    }


TransformCenter: TypeAlias = (
    Literal["origin", "geometry"] | Sequence[float] | Mapping[str, float] | np.ndarray
)
"""Transform pivot: a named point or an explicit world `(z, y, x)` point."""


def get_geometry_center(data: xr.DataArray) -> dict[str, float]:
    """Return the midpoint of the world-space box enclosing all grids.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, including cropped or scalar-selected voxel dimensions.

    Returns
    -------
    dict[str, float]
        Shared center keyed by `z`, `y`, `x`, in `.vd.world_units`. Select
        geometry dependencies first for a grid-specific center.

    Raises
    ------
    ValueError
        If the array is invalid or voxel coordinates are irregular.
    """
    return {
        dim: (lower + upper) / 2
        for dim, (lower, upper) in get_bounding_box(data).items()
    }


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
    self_dims = self_index.geometry_dims if self_affine.ndim > 2 else ()
    other_dims = other_index.geometry_dims if other_affine.ndim > 2 else ()
    _check_geometry_labels(da, other, set(self_dims) & set(other_dims))
    dims = tuple(dict.fromkeys((*self_dims, *other_dims)))
    return np.linalg.inv(_align_affine(other_affine, other_dims, dims)) @ _align_affine(
        self_affine, self_dims, dims
    )


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
    *,
    geometry_dims: tuple[str, ...] | None = None,
) -> xr.DataArray:
    """Apply a world-space affine to a DataArray's world coordinates.

    The transform is composed into the DataArray's `VoxelToWorldIndex`, derived
    world coordinates are regenerated, and named affine frames on that index are
    re-expressed against the new world frame.

    Parameters
    ----------
    da : xarray.DataArray
        Input scan with voxel-to-world geometry (a `VoxelToWorldIndex`).
    affine : (4, 4) numpy.ndarray or str
        Homogeneous world-space affine matrix to apply. If a string, it is looked up
        among the index's named affine frames.
    inplace : bool, default: False
        Whether to modify the DataArray in-place, requiring all native voxel dims.
    geometry_dims : tuple[str, ...], optional
        Named batch axes of a NumPy action. Omitted stacked actions use the
        existing geometry dimensions; shared actions use `()`. Explicit axes
        may select any nonspatial dimensions and are aligned by name.

    Returns
    -------
    xarray.DataArray
        `da` with updated spatial coordinates and updated named affine frames. No
        frame is removed: when `affine` is a string, that frame is kept as the
        identity, since the world frame now is that named space. Scalar-selected
        voxel dimensions are restored unless modifying in place.

    Raises
    ------
    ValueError
        If `da` lacks voxel-to-world geometry, if `affine` is invalid or has an
        incompatible shape, or if in-place mutation would restore voxel dimensions.
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
    from voxeldata.validation import ensure_voxeldata

    _require_voxel_to_world_index(da)
    # Restore scalar-selected axes before composing square homogeneous matrices.
    if not set(VOXEL_DIMS) <= set(da.dims):
        if inplace:
            raise ValueError(
                "inplace=True needs all native voxel dims; use inplace=False."
            )
        da = ensure_voxeldata(da)
    index = _require_voxel_to_world_index(da)
    applied_key = affine if isinstance(affine, str) else None
    if applied_key is not None:
        if applied_key not in index.named_affines:
            raise KeyError(f"'{applied_key}' not found among named affines.")
        affine = index.named_affines[applied_key]
    affine_array = np.asarray(affine, dtype=np.float64)

    voxel_to_world = index.voxel_to_world
    action_dims = (
        (() if affine_array.ndim == 2 else index.geometry_dims)
        if geometry_dims is None
        else tuple(geometry_dims)
    )
    if applied_key is not None and geometry_dims is not None:
        raise ValueError(
            "Named affines already have geometry dimensions; do not override them."
        )
    if len(set(action_dims)) != len(action_dims) or any(
        d not in da.dims or d in VOXEL_DIMS for d in action_dims
    ):
        raise ValueError(
            "Affine geometry_dims must be unique nonspatial data dimensions."
        )
    expected = tuple(da.sizes[d] for d in action_dims) + (4, 4)
    if affine_array.shape != expected:
        raise ValueError(
            f"World affine requires shape matching geometry dimensions {expected}, got {affine_array.shape}."
        )
    if not np.all(np.isfinite(affine_array)) or not np.allclose(
        affine_array[..., -1, :], [0.0, 0.0, 0.0, 1.0]
    ):
        raise ValueError("World affine must be finite and homogeneous.")
    dims = tuple(dict.fromkeys((*index.geometry_dims, *action_dims)))
    affine_array = _align_affine(affine_array, action_dims, dims)
    voxel_to_world = _align_affine(voxel_to_world, index.geometry_dims, dims)

    inv_affine = np.linalg.inv(affine_array)
    new_named: dict[str, npt.NDArray[np.float64]] = {}
    for stored_key, val in index.named_affines.items():
        if stored_key == applied_key:
            # The world frame now is this named space. Store an exact identity
            # instead of `arr @ inv(arr)`, which carries floating-point error.
            new_named[stored_key] = np.eye(affine_array.shape[-1])
            continue
        stored_dims = index.geometry_dims if val.ndim > 2 else ()
        updated = _align_affine(val, stored_dims, dims) @ inv_affine
        new_named[stored_key] = (
            updated.reshape(4, 4)
            if not stored_dims and not action_dims
            else np.broadcast_to(
                updated, tuple(da.sizes[d] for d in dims) + (4, 4)
            ).copy()
        )

    result = attach_voxel_to_world_index(
        da,
        np.broadcast_to(
            affine_array @ voxel_to_world, tuple(da.sizes[d] for d in dims) + (4, 4)
        ),
        geometry_dims=dims,
        units=get_voxel_to_world_units(da),
        named_affines=new_named,
    )
    if inplace:
        da.coords.update(result.coords)
        da.attrs.clear()
        da.attrs.update(result.attrs)
        return da
    return result


def _as_finite_array(
    value: "npt.ArrayLike", name: str, shape: tuple[int, ...]
) -> "npt.NDArray[np.float64]":
    """Validate a real numeric array without accepting strings or booleans.

    Parameters
    ----------
    value : numpy.typing.ArrayLike
        Candidate values.
    name : str
        Argument name for errors.
    shape : tuple[int, ...]
        Required shape.

    Returns
    -------
    numpy.ndarray
        Finite float64 values.

    Raises
    ------
    ValueError
        If values are not finite real numbers of the required shape.
    """
    message = f"{name} must be finite real numbers with shape {shape}."
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if (
        array.dtype.kind not in "iuf"
        or array.shape != shape
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(message)
    return array.astype(np.float64)


def apply_translation(
    data: xr.DataArray, translation: "npt.ArrayLike", *, inplace: bool = False
) -> xr.DataArray:
    """Translate world geometry without resampling or changing voxel labels.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to transform.
    translation : (3,) array-like
        Translation in world `(z, y, x)` order, in `data.vd.world_units`.
    inplace : bool, default: False
        Whether to modify the original array, requiring all native voxel dims.

    Returns
    -------
    xarray.DataArray
        Translated array, with named frames re-expressed in the new world frame.
        Scalar-selected voxel dims are restored unless modifying in place.

    Raises
    ------
    ValueError
        If translation is not a finite real vector or geometry is invalid.
    """
    matrix = np.eye(4)
    matrix[:3, 3] = _as_finite_array(translation, "translation", (3,))
    return apply_affine(data, matrix, inplace=inplace)


def _resolve_center(data: xr.DataArray, center: TransformCenter) -> np.ndarray:
    """Resolve a shared world-space transform pivot.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to transform.
    center : TransformCenter
        Named pivot, world vector, or mapping containing `z`, `y`, `x`.

    Returns
    -------
    numpy.ndarray
        World point of shape `(3,)`.

    Raises
    ------
    ValueError
        If the center is invalid, the requested geometry is undefined, or an
        origin is requested before selecting all geometry dependencies.
    """
    if isinstance(center, Mapping):
        missing = set(WORLD_DIMS) - center.keys()
        if missing:
            raise ValueError(
                f"center mapping is missing world coordinates: {sorted(missing)}."
            )
        center = [center[dim] for dim in WORLD_DIMS]
    if not isinstance(center, str):
        return _as_finite_array(center, "center", (3,))
    if center == "geometry":
        point = get_geometry_center(data)
        return np.array([point[dim] for dim in WORLD_DIMS])
    if center == "origin":
        from voxeldata.validation import ensure_voxeldata

        point = get_array_origin(ensure_voxeldata(data))
        return np.array([point[dim] for dim in WORLD_DIMS])
    raise ValueError("center must be 'origin', 'geometry' or a (3,) world point.")


def _apply_linear_transform(
    data: xr.DataArray, linear: np.ndarray, center: TransformCenter, inplace: bool
) -> xr.DataArray:
    """Apply a world linear transform about a shared pivot.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to transform.
    linear : (3, 3) numpy.ndarray
        World-space linear action.
    center : TransformCenter
        Pivot specification.
    inplace : bool
        Whether to modify the original array.

    Returns
    -------
    xarray.DataArray
        Transformed array with preserved samples and re-expressed named frames.

    Raises
    ------
    ValueError
        If the pivot or geometry is invalid.
    """
    pivot = _resolve_center(data, center)
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = pivot - linear @ pivot
    return apply_affine(data, matrix, inplace=inplace)


def apply_rotation(
    data: xr.DataArray,
    angle: "npt.ArrayLike",
    axis: "str | npt.ArrayLike",
    *,
    center: TransformCenter = "geometry",
    degrees: bool = True,
    inplace: bool = False,
) -> xr.DataArray:
    """Rotate world geometry without resampling.

    Rotations follow the right-hand rule in ordered world `(z, y, x)` coordinates:
    +90 degrees about `x` maps `+z` onto `+y`.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to transform.
    angle : float or (N,) array-like
        Angle, or one angle per axis name.
    axis : str or (3,) array-like
        World axis names, such as `"x"` or `"zyx"`, or a nonzero world vector.
        Names rotate left to right about fixed world axes and may repeat.
    center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
        Shared pivot in world units. `"geometry"` is the midpoint of the box
        enclosing all grids. `"origin"` is the first sampled voxel and requires
        selecting all geometry dependencies first. An explicit `(z, y, x)` point
        or mapping with `z`, `y`, `x` keys also gives a shared pivot. Extra mapping
        keys are ignored.
    degrees : bool, default: True
        Whether to interpret angles in degrees rather than radians.
    inplace : bool, default: False
        Whether to modify the original array, requiring all native voxel dims.

    Returns
    -------
    xarray.DataArray
        Rotated array with unchanged samples and labels, and re-expressed frames.
        Scalar-selected voxel dims are restored unless modifying in place.

    Raises
    ------
    ValueError
        If angles, axes, center, or geometry are invalid. A geometry center
        requires regular voxel coordinates.
    """
    unit_axes: np.ndarray
    if isinstance(axis, str):
        if not axis or not set(axis) <= set(WORLD_DIMS):
            raise ValueError("axis must be one or more of 'z', 'y', 'x' or a vector.")
        unit_axes = np.eye(3)[[WORLD_DIMS.index(name) for name in axis]]
    else:
        vector = _as_finite_array(axis, "axis", (3,))
        norm = math.hypot(*vector)
        if not 0 < norm < math.inf:
            raise ValueError("axis must have a non-zero, finite length.")
        unit_axes = vector[np.newaxis] / norm
    angles = _as_finite_array(np.atleast_1d(angle), "angle", (len(unit_axes),))
    if degrees:
        angles = np.deg2rad(angles)
    rotation = np.eye(3)
    for index, theta in enumerate(angles):
        a, b, c = unit_axes[index, 0], unit_axes[index, 1], unit_axes[index, 2]
        # Rodrigues' formula keeps the runtime API independent of SciPy.
        cross = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
        step = np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * (cross @ cross)
        rotation = step @ rotation
    return _apply_linear_transform(data, rotation, center, inplace)


def apply_zoom(
    data: xr.DataArray,
    zoom: "npt.ArrayLike",
    *,
    center: TransformCenter = "geometry",
    inplace: bool = False,
) -> xr.DataArray:
    """Rescale world geometry without resampling.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to transform.
    zoom : float or (3,) array-like
        Nonzero finite scale factors in world `(z, y, x)` order. A scalar scales
        every axis. Negative factors reflect; unequal factors can shear an oblique
        voxel grid. Rotation and anisotropic zoom do not generally commute.
    center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
        Shared pivot: the center of the box enclosing all grids, the first sampled
        voxel (`"origin"`, requiring selection of all geometry dependencies), or
        an explicit world point in `(z, y, x)` order. Mappings require `z`, `y`, `x`
        keys; extra keys are ignored. Coordinates use `data.vd.world_units`.
    inplace : bool, default: False
        Whether to modify the original array, requiring all native voxel dims.

    Returns
    -------
    xarray.DataArray
        Rescaled array with unchanged samples and labels, and re-expressed frames.
        Scalar-selected voxel dims are restored unless modifying in place.

    Raises
    ------
    ValueError
        If factors are zero or invalid, or center or geometry is invalid.
        A geometry center requires regular voxel coordinates.
    """
    if np.ndim(zoom) == 0:
        zoom = np.full(3, zoom)
    factors = _as_finite_array(zoom, "zoom", (3,))
    if np.any(factors == 0):
        raise ValueError("zoom factors must be non-zero.")
    return _apply_linear_transform(data, np.diag(factors), center, inplace)


def apply_flip(
    data: xr.DataArray,
    axis: str,
    *,
    center: TransformCenter = "geometry",
    inplace: bool = False,
) -> xr.DataArray:
    """Reflect world geometry without reversing or resampling the voxel array.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to transform.
    axis : str
        Distinct world axis names, such as `"x"` or `"yx"`. Order is irrelevant.
    center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
        Shared pivot: the center of the box enclosing all grids, the first sampled
        voxel (`"origin"`, requiring selection of all geometry dependencies), or
        an explicit world point in `(z, y, x)` order. Mappings require `z`, `y`, `x`
        keys; extra keys are ignored. Coordinates use `data.vd.world_units`.
    inplace : bool, default: False
        Whether to modify the original array, requiring all native voxel dims.

    Returns
    -------
    xarray.DataArray
        Reflected array with unchanged samples and labels, and re-expressed frames.
        Scalar-selected voxel dims are restored unless modifying in place.

    Raises
    ------
    ValueError
        If axes are empty, repeated, or invalid, or center or geometry is invalid.
        A geometry center requires regular voxel coordinates.
    """
    if (
        not isinstance(axis, str)
        or not axis
        or not set(axis) <= set(WORLD_DIMS)
        or len(set(axis)) != len(axis)
    ):
        raise ValueError("axis must contain distinct names among 'z', 'y', 'x'.")
    return apply_zoom(
        data,
        [-1.0 if name in axis else 1.0 for name in WORLD_DIMS],
        center=center,
        inplace=inplace,
    )


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
        `voxel_to_world` affine. World coordinates are unchanged. Each geometry
        retains its origin, direction, and spatial scalings. Rebasing is performed
        for every geometry without evaluating world coordinates.

    Raises
    ------
    ValueError
        If `da` lacks voxel-to-world geometry, or if world spacing is undefined for
        any voxel dimension.
    """
    index = _require_voxel_to_world_index(da)
    if tuple(index.voxel_dims) != VOXEL_DIMS:
        raise ValueError(
            "Canonicalize scalar-selected voxel dimensions before reindexing."
        )
    label_from_position = np.eye(4)
    for axis, dim in enumerate(VOXEL_DIMS):
        labels = np.asarray(da.coords[dim].values)
        step, approximate = get_representative_step(labels)
        if approximate:
            raise ValueError(
                f"Cannot reindex voxels because spacing is undefined for {dim!r}."
            )
        label_from_position[axis, axis] = step if step is not None else 1.0
        label_from_position[axis, -1] = labels[0]
    reindexed = da.assign_coords({dim: np.arange(da.sizes[dim]) for dim in VOXEL_DIMS})
    return attach_voxel_to_world_index(
        reindexed,
        index.voxel_to_world @ label_from_position,
        geometry_dims=index.geometry_dims,
        units=index.units,
        named_affines=index.named_affines,
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
    voxel label as well as by world position. Inputs with geometry dependencies must have
    matching geometry labels in the same order. Named frames remain those of `data`,
    since relabeling voxel coordinates does not change its world frame.

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
    source_index = _require_voxel_to_world_index(data)
    reference_index = _require_voxel_to_world_index(reference)
    if get_voxel_to_world_spatial_dims(data) != get_voxel_to_world_spatial_dims(
        reference
    ) or any(data.sizes[d] != reference.sizes[d] for d in source_index.voxel_dims):
        raise ValueError(
            "data and reference must have the same voxel grid shape and dimensions."
        )
    if set(source_index.geometry_dims) != set(reference_index.geometry_dims):
        raise ValueError(
            "data and reference must have the same geometry coordinates and geometry dimensions."
        )
    _check_geometry_labels(data, reference, set(source_index.geometry_dims))
    if source_index.units != reference_index.units:
        raise ValueError("data and reference must have matching spatial units.")
    dims = reference_index.geometry_dims
    source_affine = _align_affine(
        reindex_voxels(data).vd.index.voxel_to_world, source_index.geometry_dims, dims
    )
    reference_affine = reindex_voxels(reference).vd.index.voxel_to_world
    if not np.allclose(source_affine, reference_affine, atol=atol, rtol=0):
        raise ValueError(
            "data and reference are not aligned in world space; reindex_voxels_like requires the exact same world grid."
        )
    frames = {
        name: _align_affine(value, source_index.geometry_dims, dims)
        if value.ndim > 2
        else value
        for name, value in source_index.named_affines.items()
    }
    relabeled = data.assign_coords(
        {d: reference.coords[d] for d in source_index.voxel_dims}
    )
    return attach_voxel_to_world_index(
        relabeled,
        reference_index.voxel_to_world,
        geometry_dims=dims,
        units=reference_index.units,
        named_affines=frames,
    )


def _align_affine(
    affine: np.ndarray, source_dims: tuple[str, ...], target_dims: tuple[str, ...]
) -> np.ndarray:
    """Align affine batch axes by name, inserting singleton axes for broadcasting.

    Parameters
    ----------
    affine : numpy.ndarray
        Affine array with batch axes in `source_dims` order.
    source_dims : tuple[str, ...]
        Existing batch dimensions.
    target_dims : tuple[str, ...]
        Desired batch dimension order containing all source dimensions.

    Returns
    -------
    numpy.ndarray
        Transposed and reshaped affine view, without evaluating world coordinates.
    """
    present = [d for d in target_dims if d in source_dims]
    order = [source_dims.index(d) for d in present]
    values = affine.transpose(*order, len(source_dims), len(source_dims) + 1)
    shape = tuple(
        affine.shape[source_dims.index(d)] if d in source_dims else 1
        for d in target_dims
    )
    return values.reshape(*shape, *affine.shape[-2:])


def _geometry_labels(data: xr.DataArray, dim: str) -> np.ndarray:
    """Return dimension labels, or positions when timestamps are auxiliary arrays.

    Parameters
    ----------
    data : xarray.DataArray
        Array carrying the dimension.
    dim : str
        Geometry dimension name.

    Returns
    -------
    numpy.ndarray
        One-dimensional labels or positions.
    """
    if dim in data.coords and data.coords[dim].dims == (dim,):
        return data.coords[dim].values
    return np.arange(data.sizes[dim])


def _check_geometry_labels(
    source: xr.DataArray, other: xr.DataArray, dims: set[str]
) -> None:
    """Reject implicit alignment of incompatible affine batch coordinates.

    Parameters
    ----------
    source : xarray.DataArray
        Source array.
    other : xarray.DataArray
        Other array being composed or compared.
    dims : set[str]
        Common batch dimensions to compare.

    Raises
    ------
    ValueError
        If common geometry dimensions have different labels or ordering.
    """
    for dim in dims:
        if not np.array_equal(
            _geometry_labels(source, dim), _geometry_labels(other, dim)
        ):
            raise ValueError(
                f"Arrays must have the same geometry coordinates for geometry dimension {dim!r}."
            )
