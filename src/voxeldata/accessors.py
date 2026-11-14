"""Flat ``.vd`` accessor for the VoxelData model.

This module registers a single root accessor name, ``vd``, on
`xarray.DataArray`. It is imported from the ``voxeldata`` package so importing it
is enough to make ``data.vd`` available.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from voxeldata._dims import TIME_DIM
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
    get_array_origin,
    get_array_spacing,
    get_voxel_to_world_affine,
    get_voxel_to_world_direction_matrix,
    get_voxel_to_world_index,
    get_voxel_to_world_units,
    has_axis_aligned_voxel_to_world_index,
)
from voxeldata.validation import (
    canonicalize_voxeldata,
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

    def extract_with_mask(self, mask: xr.DataArray) -> xr.DataArray:
        """Flatten selected mask samples into a `space` dimension.

        Parameters
        ----------
        mask : xarray.DataArray
            Boolean or binary numeric VoxelData mask on the same physical grid.
            All mask dimensions are flattened, including extra feature axes.

        Returns
        -------
        xarray.DataArray
            Extracted samples with a `space` MultiIndex of native labels.
            Dask-backed signal data remain lazy.

        Raises
        ------
        TypeError
            If the mask is not boolean or binary numeric.
        ValueError
            If geometry, dimensions, or coordinates are incompatible.

        See Also
        --------
        voxeldata.extract.extract_with_mask : Module-level extraction function.
        """
        from voxeldata.extract import extract_with_mask

        return extract_with_mask(self._obj, mask)

    def extract_with_labels(
        self,
        labels: xr.DataArray,
        reduction: Literal[
            "mean", "sum", "median", "min", "max", "var", "std"
        ] = "mean",
    ) -> xr.DataArray:
        """Aggregate samples by integer regions or overlapping mask layers.

        Parameters
        ----------
        labels : xarray.DataArray
            Spatial integer VoxelData label map, optionally stacked along `mask`.
            Background zero is excluded. Physical grids must agree.
        reduction : {"mean", "sum", "median", "min", "max", "var", "std"}, default: "mean"
            Reduction applied to voxel samples in each region.

        Returns
        -------
        xarray.DataArray
            Samples with voxel dimensions replaced by `region`, retaining
            nonspatial dimensions. Dask-backed signal data remain lazy.

        Raises
        ------
        TypeError
            If labels do not have integer dtype.
        ValueError
            If grids, dimensions, reduction, labels, or backend chunks are invalid.

        See Also
        --------
        voxeldata.extract.extract_with_labels : Module-level aggregation function.
        """
        from voxeldata.extract import extract_with_labels

        return extract_with_labels(self._obj, labels, reduction=reduction)

    def unmask(self, mask: xr.DataArray, fill_value: float = 0.0) -> xr.DataArray:
        """Reconstruct the full mask grid from positional `space` samples.

        Available on geometry-free signals arrays; the mask supplies the output
        geometry. No acquisition metadata defaults or validation are applied.

        Parameters
        ----------
        mask : xarray.DataArray
            Original boolean or binary numeric VoxelData mask. Supplies native
            labels, extra feature axes, units, geometry, and named frames.
        fill_value : float, default: 0.0
            Value outside the mask, cast to the signal dtype.

        Returns
        -------
        xarray.DataArray
            Eager reconstructed VoxelData array. Leading signal coordinates and
            timestamp attributes are preserved; samples are not label-realigned.

        Raises
        ------
        TypeError
            If the mask is not boolean or binary numeric.
        ValueError
            If mask geometry or signal dimensions/count are invalid.

        See Also
        --------
        voxeldata.extract.unmask : Module-level reconstruction function.
        """
        from voxeldata.extract import unmask

        return unmask(self._obj, mask, fill_value=fill_value)

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
        """World position of the first sampled voxel.

        Returns
        -------
        dict[str, float]
            Spatial origin keyed by `z`, `y`, `x`, in `.vd.world_units`.
            Retained voxel labels account for cropping and striding.

        Raises
        ------
        ValueError
            If geometry is missing or geometry dependencies remain unselected.
        """
        _require_index(self._obj)
        return get_array_origin(self._obj)

    @property
    def spacing(self) -> dict[str, float | None]:
        """Physical sample spacing along native voxel axes only.

        Returns
        -------
        dict[str, float or None]
            World-space step lengths keyed by active voxel dimensions `k`, `j`,
            `i`. Irregular voxel labels have undefined spacing (`None`).
            Time sampling is exposed separately through `.vd.dt`.

        Raises
        ------
        ValueError
            If geometry is missing or spatial scalings differ across geometries.
        """
        _require_index(self._obj)
        return get_array_spacing(self._obj)

    @property
    def dt(self) -> float | None:
        """Sampling interval of a shared time coordinate.

        Returns
        -------
        float or None
            Median time interval in the time coordinate's units, normally seconds.
            Sampling is uniform when every interval differs from the median by
            at most 1%. Irregular sampling or fewer than two timestamps returns
            `None`. Signal data is not evaluated.

        Raises
        ------
        ValueError
            If geometry or the time coordinate is missing, additional timestamp
            dependencies remain unselected, or timestamps are not finite, real,
            and strictly increasing.
        """
        _require_index(self._obj)
        if TIME_DIM not in self._obj.coords:
            raise ValueError("DataArray must have a 'time' coordinate.")
        time = self._obj.coords[TIME_DIM]
        if time.dims not in ((), (TIME_DIM,)):
            raise ValueError(
                "Timestamp dependencies must be selected before computing dt."
            )
        values = np.asarray(time.values)
        if (
            values.dtype.kind not in "iuf"
            or not np.all(np.isfinite(values))
            or np.any(values.reshape(-1)[1:] <= values.reshape(-1)[:-1])
        ):
            raise ValueError(
                "'time' coordinate must be finite, real, and strictly increasing."
            )
        if values.size < 2:
            return None
        intervals = np.diff(values.astype(np.float64))
        interval = float(np.median(intervals))
        return (
            interval if np.allclose(intervals, interval, rtol=1e-2, atol=0.0) else None
        )

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
    def bounding_box(self) -> dict[str, tuple[float, float]]:
        """World-space bounding box enclosing every voxel across all grids.

        Returns
        -------
        dict[str, tuple[float, float]]
            World axis names mapped to `(min, max)` bounds in `.vd.world_units`.
            Select geometry dependencies first for grid-specific bounds.
        """
        return get_bounding_box(self._obj)

    @property
    def geometry_center(self) -> dict[str, float]:
        """Midpoint of the world-space bounding box enclosing all grids.

        Returns
        -------
        dict[str, float]
            Shared center keyed by world axis name, in `.vd.world_units`. Select
            geometry dependencies first for a grid-specific center.
        """
        return get_geometry_center(self._obj)

    @property
    def affines(self) -> dict[str, np.ndarray]:
        """Named affine frames stored on the index.

        Returns
        -------
        dict[str, numpy.ndarray]
            Shallow dictionary copy of named frames mapping the current world
            frame to each named frame. Use `set_affine` and `remove_affine` to
            update the index; assigning or deleting dictionary entries only
            changes this copy. Matrix arrays are shared with the index.
        """
        return dict(_require_index(self._obj).named_affines)

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

        This bootstraps geometry on a plain DataArray (one without a
        voxel-to-world index). Reconstruction through `.vd.unmask` also
        accepts geometry-free signals, using the mask's geometry instead.
        If the DataArray already carries named affine
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

    def get_relative_affine(self, other: xr.DataArray, via: str) -> np.ndarray:
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

    def canonicalize(self, **kwargs: Any) -> xr.DataArray:
        """Return a canonical, validated VoxelData array.

        Parameters
        ----------
        **kwargs : Any
            Forwarded to [canonicalize_voxeldata][voxeldata.validation.canonicalize_voxeldata].

        Returns
        -------
        xarray.DataArray
            Canonical, validated VoxelData array.
        """
        return canonicalize_voxeldata(self._obj, **kwargs)

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
