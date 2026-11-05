"""Coordinate spacing helpers shared across voxeldata modules."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import xarray as xr


def get_representative_step(
    values: npt.NDArray[np.floating], uniformity_tolerance: float = 1e-2
) -> tuple[float | None, bool]:
    """Return a representative step size for a 1D coordinate array.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional coordinate values.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed per-interval relative deviation from the median consecutive
        difference. Each interval must satisfy
        `|interval - median| <= uniformity_tolerance * |median|` for the coordinate
        to be considered uniform.

    Returns
    -------
    float or None
        Exact step size when sampling is uniform, the median consecutive difference
        when sampling is non-uniform, or `None` when fewer than two values are
        provided.
    bool
        Whether the returned step is a median approximation derived from non-uniform
        spacing.
    """
    if len(values) < 2:
        return None, False

    diffs = np.diff(values)
    median = float(np.median(diffs))
    if np.isclose(median, 0.0):
        is_uniform = bool(np.isclose(np.max(diffs), np.min(diffs)))
    else:
        is_uniform = bool(
            np.allclose(diffs, median, rtol=uniformity_tolerance, atol=0.0)
        )

    if is_uniform:
        return median, False
    return median, True


class CoordinateSpacingInfo:
    """Result of per-dimension spacing analysis.

    Parameters
    ----------
    value : float or None
        Exact spacing when the coordinate is uniform. `None` otherwise, including for
        single-point coordinates.
    median : float or None
        Median consecutive difference for numeric coordinates with two or more points.
        Available for both uniform and non-uniform coordinates; `None` for missing,
        non-numeric, or single-point coordinates.
    warn_msg : str or None
        Warning message to emit, or `None` if no warning is needed.
    """

    __slots__ = ("median", "value", "warn_msg")

    def __init__(
        self,
        value: float | None,
        median: float | None,
        warn_msg: str | None,
    ) -> None:
        self.value = value
        self.median = median
        self.warn_msg = warn_msg


def get_coordinate_spacing_info(
    dim: str,
    data: xr.DataArray,
    uniformity_tolerance: float,
) -> CoordinateSpacingInfo:
    """Compute coordinate spacing information for a single dimension.

    Parameters
    ----------
    dim : str
        Dimension name.
    data : xarray.DataArray
        DataArray whose coordinate to inspect.
    uniformity_tolerance : float
        Maximum allowed per-interval relative deviation from the median consecutive
        difference.

    Returns
    -------
    CoordinateSpacingInfo
        Spacing result for the dimension.
    """
    if dim not in data.coords:
        return CoordinateSpacingInfo(
            value=None,
            median=None,
            warn_msg=f"Dimension '{dim}' has no coordinate; spacing is undefined.",
        )

    coord = data.coords[dim]
    if not np.issubdtype(coord.dtype, np.integer) and not np.issubdtype(
        coord.dtype, np.floating
    ):
        return CoordinateSpacingInfo(value=None, median=None, warn_msg=None)

    # A coordinate named after its dim is not necessarily 1D (e.g. a grid-dependent
    # "time" coordinate can be (time, grid)-shaped). Compute the step along `dim`
    # independently for every combination of the other dims' values.
    other_dims = [d for d in coord.dims if d != dim]
    dim_axis = coord.dims.index(dim)

    if coord.shape[dim_axis] < 2:
        return CoordinateSpacingInfo(
            value=None,
            median=None,
            warn_msg=(
                f"Dimension '{dim}' has a single coordinate point; spacing is "
                "undefined."
            ),
        )

    values = np.moveaxis(coord.values, dim_axis, 0).reshape(coord.shape[dim_axis], -1)

    steps: list[float] = []
    for column in range(values.shape[1]):
        step, is_approximate = get_representative_step(
            values[:, column], uniformity_tolerance=uniformity_tolerance
        )
        assert step is not None  # values.shape[0] >= 2, checked above.
        if is_approximate:
            return CoordinateSpacingInfo(
                value=None,
                median=float(np.median(steps + [step])),
                warn_msg=(
                    f"Coordinate '{dim}' has non-uniform sampling; spacing is "
                    "undefined."
                ),
            )
        steps.append(step)

    if other_dims and not np.allclose(
        steps, steps[0], rtol=uniformity_tolerance, atol=0.0
    ):
        return CoordinateSpacingInfo(
            value=None,
            median=float(np.median(steps)),
            warn_msg=(
                f"Coordinate '{dim}' spacing along '{dim}' differs across "
                f"{other_dims!r}; spacing is undefined."
            ),
        )

    return CoordinateSpacingInfo(value=steps[0], median=steps[0], warn_msg=None)


def get_coordinate_origins(data: xr.DataArray) -> dict[str, float]:
    """Return the first coordinate value for each dimension.

    For each dimension, returns the first coordinate value. If a coordinate is
    missing, falls back to `0.0` with a warning. If a coordinate is non-numeric
    (e.g. string-based), falls back to `0.0` without a warning.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray whose coordinate origins to compute.

    Returns
    -------
    dict[str, float]
        Origin per dimension in DataArray dimension order.
    """
    result: dict[str, float] = {}
    for dim in (str(d) for d in data.dims):
        if dim not in data.coords:
            warnings.warn(
                f"Dimension '{dim}' has no coordinate; origin defaults to 0.0.",
                stacklevel=2,
            )
            result[dim] = 0.0
        else:
            coord = data.coords[dim]
            if not np.issubdtype(coord.dtype, np.integer) and not np.issubdtype(
                coord.dtype, np.floating
            ):
                result[dim] = 0.0
            else:
                result[dim] = float(coord.values.flat[0])
    return result
