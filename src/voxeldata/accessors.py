"""Flat ``.vd`` accessor for the VoxelData model.

This module registers a single root accessor name, ``vd``, on
`xarray.DataArray`. It is imported from the ``voxeldata`` package so importing it
is enough to make ``data.vd`` available.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from voxeldata._coordinates import get_coordinate_origins, get_coordinate_spacing_info
from voxeldata.affine import (
    TransformCenter,
    apply_affine,
    apply_flip,
    apply_rotation,
    apply_translation,
    apply_zoom,
    get_bounding_box,
    get_geometry_center,
    reindex_voxels,
    reindex_voxels_like,
)
from voxeldata.geometry import (
    VoxelToWorldIndex,
    attach_voxel_to_world_index,
    get_affine_axis_scalings,
    get_array_origin,
    get_array_spacing,
    get_voxel_to_world_affine,
    get_voxel_to_world_direction_matrix,
    get_voxel_to_world_index,
    get_voxel_to_world_spatial_dims,
    get_voxel_to_world_units,
    has_axis_aligned_voxel_to_world_index,
)
from voxeldata.validation import (
    canonicalize_voxeldata,
    ensure_voxeldata,
    validate_voxeldata,
)

if TYPE_CHECKING:
    import numpy.typing as npt


@xr.register_dataarray_accessor("vd")
class VoxelDataAccessor:
    """VoxelData operations exposed as ``data.vd.<name>``.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray wrapped by xarray when the accessor is resolved.
    """

    def __init__(self, data: xr.DataArray) -> None:
        self._obj = data

    @property
    def index(self) -> VoxelToWorldIndex:
        """Attached voxel-to-world index.

        Returns
        -------
        VoxelToWorldIndex
            The DataArray's voxel-to-world index.

        Raises
        ------
        ValueError
            If the DataArray carries no voxel-to-world index.
        """
        return _require_index(self._obj)

    @property
    def voxel_dims(self) -> tuple[str, ...]:
        """Active voxel dimensions.

        Returns
        -------
        tuple[str, ...]
            Active voxel dimension names in affine column order.
        """
        return get_voxel_to_world_spatial_dims(self._obj)

    @property
    def world_dims(self) -> tuple[str, ...]:
        """World coordinate names.

        Returns
        -------
        tuple[str, ...]
            World coordinate names in affine row order.
        """
        _require_index(self._obj)
        return ("z", "y", "x")

    @property
    def geometry_dims(self) -> tuple[str, ...]:
        """Named affine dependencies, in affine batch-axis order.

        Returns
        -------
        tuple[str, ...]
            Geometry-dependent nonspatial dimension names.
        """
        return self.index.geometry_dims

    @property
    def is_axis_aligned(self) -> bool:
        """Whether the voxel-to-world affine is axis-aligned.

        Returns
        -------
        bool
            Whether world axes align with voxel axes (allowing signed scaling).
        """
        return has_axis_aligned_voxel_to_world_index(self._obj)

    @property
    def fixed_voxel_coords(self) -> dict[str, float]:
        """Voxel coordinates pinned by a prior scalar selection.

        Returns
        -------
        dict[str, float]
            Pinned voxel-space coordinates keyed by dimension name.
        """
        return _require_index(self._obj).fixed_voxel_coords

    @property
    def world_units(self) -> str:
        """Physical unit of the world coordinates.

        Returns
        -------
        str
            World coordinate unit.
        """
        return get_voxel_to_world_units(self._obj)

    @property
    def voxel_to_world(self) -> np.ndarray:
        """Voxel-to-world affine.

        Returns
        -------
        (4, 4) numpy.ndarray or (*geometry_shape, 4, 4) numpy.ndarray
            Affine mapping active voxel dimensions to world coordinates, shared or
            stacked along `geometry_dims`.
        """
        return get_voxel_to_world_affine(self._obj)

    @property
    def origin(self) -> dict[str, float]:
        """Origin metadata for all dimensions.

        Non-spatial dimensions report their first coordinate value; `z`/`y`/`x`
        report the world location of the first sampled voxel.

        Returns
        -------
        dict[str, float]
            Origin per non-spatial dimension followed by world coordinates.
        """
        index = _require_index(self._obj)
        voxel_dims = set(index.voxel_dims)
        regular_origin = get_coordinate_origins(self._obj)
        return {
            **{
                str(dim): regular_origin[str(dim)]
                for dim in self._obj.dims
                if str(dim) not in voxel_dims
            },
            **get_array_origin(self._obj),
        }

    @property
    def spacing(self) -> dict[str, float | None]:
        """Coordinate spacing for all dimensions.

        Voxel dimensions report their world step length; other dimensions report
        their 1D coordinate step when defined.

        Returns
        -------
        dict[str, float or None]
            Spacing per dimension, or `None` when undefined or non-uniform.
        """
        _require_index(self._obj)
        voxel_spacing = get_array_spacing(self._obj)
        regular_spacing = {
            str(dim): get_coordinate_spacing_info(str(dim), self._obj, 1e-2).value
            for dim in self._obj.dims
            if str(dim) not in voxel_spacing
        }
        return {
            str(dim): voxel_spacing[str(dim)]
            if str(dim) in voxel_spacing
            else regular_spacing[str(dim)]
            for dim in self._obj.dims
        }

    @property
    def scalings(self) -> dict[str, float]:
        """World distance per voxel-space unit along each voxel axis.

        Returns
        -------
        dict[str, float]
            Column norms of the affine's linear part, keyed by voxel dimension.
        """
        index = _require_index(self._obj)
        return get_affine_axis_scalings(self.voxel_to_world, index.voxel_dims)

    @property
    def direction(self) -> np.ndarray:
        """World-space direction matrix.

        Returns
        -------
        (3, 3) numpy.ndarray
            Unit world direction of each voxel axis; identity for axis-aligned
            data.
        """
        return get_voxel_to_world_direction_matrix(self._obj)

    @property
    def bounding_box(self) -> xr.DataArray:
        """World-space bounding box enclosing every voxel across all grids.

        Returns
        -------
        xarray.DataArray
            Bounds with dimensions `(bound, component)` and world units. Select
            geometry dependencies first for grid-specific bounds.
        """
        return get_bounding_box(self._obj)

    @property
    def geometry_center(self) -> xr.DataArray:
        """Midpoint of the world-space bounding box enclosing all grids.

        Returns
        -------
        xarray.DataArray
            Shared center with dimension `component` and world units. Select
            geometry dependencies first for a grid-specific center.
        """
        return get_geometry_center(self._obj)

    @property
    def affines(self) -> dict[str, np.ndarray]:
        """Named affine frames stored on the index.

        Returns
        -------
        dict[str, numpy.ndarray]
            Named frames mapping the current world frame to each named frame.
        """
        return dict(_require_index(self._obj).named_affines)

    def get_affine(self, name: str) -> np.ndarray:
        """Return a named affine frame.

        Parameters
        ----------
        name : str
            Frame name.

        Returns
        -------
        (4, 4) numpy.ndarray or (*geometry_shape, 4, 4) numpy.ndarray
            The named frame.

        Raises
        ------
        KeyError
            If `name` is not a stored frame.
        """
        index = _require_index(self._obj)
        if name not in index.named_affines:
            raise KeyError(f"'{name}' not found among named affines.")
        return index.named_affines[name]

    def set_affine(
        self,
        name: str,
        affine: Any,
        *,
        inplace: bool = False,
    ) -> xr.DataArray:
        """Store a named affine frame on the index.

        Parameters
        ----------
        name : str
            Frame name.
        affine : numpy.typing.ArrayLike
            Homogeneous `(4, 4)` frame, or a full stack in `geometry_dims` order.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray with the new named frame.
        """
        index = _require_index(self._obj)
        named = dict(index.named_affines)
        named[name] = affine
        return self._replace_index(index=index, named_affines=named, inplace=inplace)

    def remove_affine(self, name: str, *, inplace: bool = False) -> xr.DataArray:
        """Remove a named affine frame from the index.

        Parameters
        ----------
        name : str
            Frame name to remove.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray without the named frame.

        Raises
        ------
        KeyError
            If `name` is not a stored frame.
        """
        index = _require_index(self._obj)
        if name not in index.named_affines:
            raise KeyError(f"'{name}' not found among named affines.")
        named = dict(index.named_affines)
        del named[name]
        return self._replace_index(index=index, named_affines=named, inplace=inplace)

    def set_voxel_to_world(
        self,
        voxel_to_world: Any,
        *,
        units: str | None = None,
        geometry_dims: tuple[str, ...] | None = None,
        named_affines: dict[str, Any] | None = None,
        inplace: bool = False,
    ) -> xr.DataArray:
        """Attach a new voxel-to-world affine.

        This is the only operation allowed on a plain DataArray (one without a
        voxel-to-world index). If the DataArray already carries named affine
        frames and `named_affines` is not provided, those frames are dropped with a
        warning: they are expressed relative to the old world frame and cannot be
        carried over automatically. Use `apply` to change the world frame while
        preserving them.

        Parameters
        ----------
        voxel_to_world : numpy.typing.ArrayLike
            Homogeneous `(4, 4)` affine, or a full stack in `geometry_dims` order.
        units : str, optional
            World coordinate unit. If not provided, keeps the existing unit, or
            defaults to `"mm"` for a plain DataArray.
        geometry_dims : tuple[str, ...], optional
            Ordered nonspatial dimensions indexing the affine stack. A shared
            affine uses `()`. Stacks always require explicit names.
        named_affines : mapping[str, numpy.typing.ArrayLike], optional
            Named affine frames for the new geometry.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray carrying the new voxel-to-world geometry.
        """
        existing = get_voxel_to_world_index(self._obj)
        if existing is not None and existing.named_affines and named_affines is None:
            warnings.warn(
                "Replacing voxel-to-world geometry drops existing named affine "
                "frames. Use apply() to change the world frame while preserving "
                "them, or pass named_affines explicitly.",
                stacklevel=2,
            )
        resolved_units = (
            units
            if units is not None
            else (existing.units if existing is not None else "mm")
        )
        result = attach_voxel_to_world_index(
            self._obj,
            voxel_to_world,
            geometry_dims=geometry_dims,
            units=resolved_units,
            named_affines=named_affines,
        )
        return _maybe_inplace(self._obj, result, inplace)

    def set_world_units(self, units: str, *, inplace: bool = False) -> xr.DataArray:
        """Change the world coordinate unit.

        Parameters
        ----------
        units : str
            New world coordinate unit.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray with the new unit.
        """
        index = _require_index(self._obj)
        return self._replace_index(index=index, units=units, inplace=inplace)

    def apply(
        self,
        affine: Any,
        *,
        geometry_dims: tuple[str, ...] | None = None,
        inplace: bool = False,
    ) -> xr.DataArray:
        """Apply a world-space affine to the world coordinates.

        Parameters
        ----------
        affine : (4, 4) numpy.ndarray or str
            World-space affine, or the name of a stored frame. Applying a named
            frame makes it the identity and re-expresses every other frame.
        geometry_dims : tuple[str, ...], optional
            Ordered dependencies of a NumPy action. Defaults to the existing
            geometry dimensions for a stack, or `()` for a shared affine. Named
            frames already carry their dependencies. New action dependencies
            are broadcast by name and added to the geometry.
        inplace : bool, default: False
            Whether to modify the DataArray in-place, requiring all native voxel dims.

        Returns
        -------
        xarray.DataArray
            DataArray with updated world coordinates and named frames.
            Scalar-selected voxel dims are restored unless modifying in place.
        """
        return apply_affine(
            self._obj, affine, inplace=inplace, geometry_dims=geometry_dims
        )

    def translate(
        self, translation: npt.ArrayLike, *, inplace: bool = False
    ) -> xr.DataArray:
        """Translate world geometry without resampling.

        Parameters
        ----------
        translation : (3,) array-like
            World `(z, y, x)` translation in `.vd.world_units`.
        inplace : bool, default: False
            Whether to modify the original array, requiring all native voxel dims.

        Returns
        -------
        xarray.DataArray
            Translated array. See [apply_translation][voxeldata.affine.apply_translation].
        """
        return apply_translation(self._obj, translation, inplace=inplace)

    def rotate(
        self,
        angle: npt.ArrayLike,
        axis: str | npt.ArrayLike,
        *,
        center: TransformCenter = "geometry",
        degrees: bool = True,
        inplace: bool = False,
    ) -> xr.DataArray:
        """Rotate world geometry without resampling.

        Parameters
        ----------
        angle : float or (N,) array-like
            Angle, or one angle per axis name.
        axis : str or (3,) array-like
            World names or vector in `(z, y, x)` order. Names rotate left to right
            about fixed world axes, following the right-hand rule in this order.
        center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
            Shared enclosing-box center or explicit world point in `(z, y, x)`
            order (or a mapping with those keys). `"origin"` uses the first sampled
            voxel and requires selecting all geometry dependencies first.
        degrees : bool, default: True
            Whether to interpret angles in degrees rather than radians.
        inplace : bool, default: False
            Whether to modify the original array, requiring all native voxel dims.

        Returns
        -------
        xarray.DataArray
            Rotated array. See [apply_rotation][voxeldata.affine.apply_rotation].
        """
        return apply_rotation(
            self._obj, angle, axis, center=center, degrees=degrees, inplace=inplace
        )

    def zoom(
        self,
        zoom: npt.ArrayLike,
        *,
        center: TransformCenter = "geometry",
        inplace: bool = False,
    ) -> xr.DataArray:
        """Rescale world geometry without resampling.

        Parameters
        ----------
        zoom : float or (3,) array-like
            Nonzero finite world scale factors. Negative values reflect.
        center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
            Shared enclosing-box center or explicit world point in `(z, y, x)`
            order (or a mapping with those keys). `"origin"` uses the first sampled
            voxel and requires selecting all geometry dependencies first.
        inplace : bool, default: False
            Whether to modify the original array, requiring all native voxel dims.

        Returns
        -------
        xarray.DataArray
            Rescaled array. See [apply_zoom][voxeldata.affine.apply_zoom].
        """
        return apply_zoom(self._obj, zoom, center=center, inplace=inplace)

    def flip(
        self, axis: str, *, center: TransformCenter = "geometry", inplace: bool = False
    ) -> xr.DataArray:
        """Reflect world geometry without reversing the voxel array.

        Parameters
        ----------
        axis : str
            Distinct world axis names, such as `"x"` or `"yx"`.
        center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
            Shared enclosing-box center or explicit world point in `(z, y, x)`
            order (or a mapping with those keys). `"origin"` uses the first sampled
            voxel and requires selecting all geometry dependencies first.
        inplace : bool, default: False
            Whether to modify the original array, requiring all native voxel dims.

        Returns
        -------
        xarray.DataArray
            Reflected array. See [apply_flip][voxeldata.affine.apply_flip].
        """
        return apply_flip(self._obj, axis, center=center, inplace=inplace)

    def reindex_voxels(self, *, inplace: bool = False) -> xr.DataArray:
        """Rebase voxel coordinates to dense positions.

        Parameters
        ----------
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray with voxel coordinates `0, 1, ..., n - 1`.
        """
        return _maybe_inplace(self._obj, reindex_voxels(self._obj), inplace)

    def reindex_voxels_like(
        self,
        reference: xr.DataArray,
        *,
        atol: float = 1e-6,
        inplace: bool = False,
    ) -> xr.DataArray:
        """Rebase voxel coordinates onto `reference`'s voxel labels.

        Parameters
        ----------
        reference : xarray.DataArray
            DataArray whose voxel labels and affine should be adopted.
        atol : float, default: 1e-6
            Absolute tolerance for the world-coordinate alignment check.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray aligned on `reference`'s voxel grid.
        """
        return _maybe_inplace(
            self._obj,
            reindex_voxels_like(self._obj, reference, atol=atol),
            inplace,
        )

    def to(self, other: xr.DataArray, via: str) -> np.ndarray:
        """Return the affine mapping this world frame into `other`'s.

        Parameters
        ----------
        other : xarray.DataArray
            Target VoxelData array.
        via : str
            Shared named affine frame.

        Returns
        -------
        (4, 4) numpy.ndarray
            Affine mapping this DataArray's world coordinates to `other`'s.
        """
        from voxeldata.affine import get_relative_affine

        return get_relative_affine(self._obj, other, via)

    def validate(self, **kwargs: Any) -> None:
        """Validate that this DataArray follows the VoxelData model.

        Parameters
        ----------
        **kwargs : Any
            Forwarded to [validate_voxeldata][voxeldata.validation.validate_voxeldata].

        Returns
        -------
        None
            Validation raises on failure.
        """
        validate_voxeldata(self._obj, **kwargs)

    def canonicalize(self) -> xr.DataArray:
        """Return a canonical VoxelData array.

        Returns
        -------
        xarray.DataArray
            Canonical form with reinstated core dimensions and canonical order.
        """
        return canonicalize_voxeldata(self._obj)

    def ensure(self, **kwargs: Any) -> xr.DataArray:
        """Canonicalize and validate in one step.

        Parameters
        ----------
        **kwargs : Any
            Forwarded to [ensure_voxeldata][voxeldata.validation.ensure_voxeldata].

        Returns
        -------
        xarray.DataArray
            Canonical, validated VoxelData array.
        """
        return ensure_voxeldata(self._obj, **kwargs)

    def _replace_index(
        self,
        *,
        index: VoxelToWorldIndex,
        units: str | None = None,
        named_affines: dict[str, Any] | None = None,
        inplace: bool = False,
    ) -> xr.DataArray:
        """Reattach the index with changed metadata.

        Parameters
        ----------
        index : VoxelToWorldIndex
            Existing index supplying the affine and defaults.
        units : str, optional
            New unit; defaults to the index's current unit.
        named_affines : mapping[str, numpy.typing.ArrayLike], optional
            New named frames; defaults to the index's current frames.
        inplace : bool, default: False
            Whether to modify the DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray with the replaced index.
        """
        result = attach_voxel_to_world_index(
            self._obj,
            index.voxel_to_world,
            geometry_dims=index.geometry_dims,
            units=index.units if units is None else units,
            named_affines=(
                index.named_affines if named_affines is None else named_affines
            ),
        )
        return _maybe_inplace(self._obj, result, inplace)


def _require_index(data: xr.DataArray) -> VoxelToWorldIndex:
    """Return `data`'s voxel-to-world index or raise.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to inspect.

    Returns
    -------
    VoxelToWorldIndex
        The attached index.

    Raises
    ------
    ValueError
        If `data` has no voxel-to-world index.
    """
    index = get_voxel_to_world_index(data)
    if index is None:
        raise ValueError("DataArray must have a voxel-to-world index.")
    return index


def _maybe_inplace(
    original: xr.DataArray, result: xr.DataArray, inplace: bool
) -> xr.DataArray:
    """Return `result`, or copy its coordinates and attributes into `original`.

    Parameters
    ----------
    original : xarray.DataArray
        DataArray to mutate when `inplace` is true.
    result : xarray.DataArray
        Result carrying the new coordinates and attributes.
    inplace : bool
        Whether to mutate `original`.

    Returns
    -------
    xarray.DataArray
        `original` after mutation, or `result`.
    """
    if not inplace:
        return result
    original.coords.update(result.coords)
    original.attrs.clear()
    original.attrs.update(result.attrs)
    return original
