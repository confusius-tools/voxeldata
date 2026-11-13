"""Region aggregation."""

from collections.abc import Hashable
from typing import Literal

import numpy as np
import xarray as xr

from voxeldata._dims import VOXEL_DIMS
from voxeldata.masks import ensure_labels
from voxeldata.validation import canonicalize_voxeldata

_VALID_REDUCTIONS = frozenset({"mean", "sum", "median", "min", "max", "var", "std"})
"""Valid reduction names accepted by `extract_with_labels`."""


def _validate_stacked_mask_layers(labels: xr.DataArray) -> None:
    """Validate that each stacked mask layer has exactly one non-zero value.

    The layer's own non-zero value is never used to key the reduction (a fresh,
    per-layer id is assigned instead, see `extract_with_labels`), so layers may
    reuse the same id across the `mask` dimension—a stacked mask layer is already
    uniquely identified by its position along `mask`.

    Parameters
    ----------
    labels : xarray.DataArray
        Stacked mask array with a leading `mask` dimension.

    Raises
    ------
    ValueError
        If any layer has more or fewer than one unique non-zero value.
    """
    for i in range(labels.sizes["mask"]):
        layer_vals = np.unique(labels.isel(mask=i).values)
        layer_vals = layer_vals[layer_vals != 0]
        if len(layer_vals) != 1:
            raise ValueError(
                f"Stacked mask layer {i} must have exactly one unique non-zero "
                f"value, got {len(layer_vals)}: {layer_vals.tolist()}."
            )


def _grouped_reduce(
    data: xr.DataArray,
    labels: xr.DataArray,
    spatial_dims: list[Hashable],
    reduction: Literal["mean", "sum", "median", "min", "max", "var", "std"],
) -> xr.DataArray:
    """Reduce spatial samples with Xarray's optional flox acceleration.

    Parameters
    ----------
    data : xarray.DataArray
        Input data.
    labels : xarray.DataArray
        Integer label map. May have a leading `mask` dimension for the
        overlapping groups case.
    spatial_dims : list[str]
        Spatial dimensions to reduce over.
    reduction : str
        Aggregation function name.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions replaced by a `region` dimension,
        with integer region coordinates. Background (0) is excluded.
    """
    labels_aligned = labels

    if not (labels_aligned != 0).any():
        raise ValueError("labels contains no non-zero values: no regions to extract.")

    data_stacked = data.stack(space=spatial_dims, create_index=False)
    # Group membership must be known without computing the signal samples.
    labels_stacked = (
        labels_aligned.stack(space=spatial_dims, create_index=False)
        .rename("region")
        .compute()
    )
    # Overlapping layers add a grouping axis absent from the signal data.
    data_stacked = xr.broadcast(data_stacked, labels_stacked)[0]
    result = getattr(data_stacked.groupby(labels_stacked), reduction)()
    return result.isel(region=result.region.values != 0)  # Drop background.


def extract_with_labels(
    data: xr.DataArray,
    labels: xr.DataArray,
    reduction: Literal["mean", "sum", "median", "min", "max", "var", "std"] = "mean",
) -> xr.DataArray:
    """Aggregate voxel samples by integer regions or overlapping mask layers.

    Background label zero is excluded. A flat map identifies regions by its
    nonzero integer values; stacked mask layers are identified by their `mask`
    coordinate, even when layers reuse the same integer id or overlap.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array with arbitrary nonspatial dimensions.
    labels : xarray.DataArray
        Spatial VoxelData label map with dimensions `(k, j, i)`, or stacked masks
        with dimensions `(mask, k, j, i)`. Each stacked layer must contain exactly
        one distinct nonzero value. Geometry, units, and native voxel labels must
        match the data. Select varying data geometry before using a shared map.
    reduction : {"mean", "sum", "median", "min", "max", "var", "std"}, default: "mean"
        Reduction applied across voxel samples belonging to each region.

    Returns
    -------
    xarray.DataArray
        Region-aggregated samples, retaining nonspatial dimensions and replacing
        native voxel dimensions with `region`.

    Raises
    ------
    TypeError
        If labels are not an integer VoxelData array.
    ValueError
        If grids do not match, dimensions, reduction, or backend chunking are
        unsupported, labels are all background, or a stacked layer has zero or
        multiple distinct ids.

    Notes
    -----
    Uses Xarray's grouped reductions, automatically accelerated by flox when
    installed and enabled. Dask-backed signal data remain lazy. With flox, median
    requires each group to fit within a single Dask chunk; explicitly rechunk the
    spatial dimensions when needed. Label metadata are evaluated eagerly to
    determine groups. No modality-specific acquisition metadata are required.
    """
    data = canonicalize_voxeldata(data, allow_extra_dims=True)
    labels = ensure_labels(labels, data, "labels")
    if tuple(labels.dims) not in (VOXEL_DIMS, ("mask", *VOXEL_DIMS)):
        raise ValueError(
            "Labels must have spatial dimensions, optionally preceded by a mask dimension."
        )

    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Must be one of: {sorted(_VALID_REDUCTIONS)}."
        )

    if "mask" in labels.dims:
        spatial_dims = [d for d in labels.dims if d != "mask"]
        _validate_stacked_mask_layers(labels)

        # A stacked mask layer is already uniquely identified by its position along
        # `mask`, so group by a fresh per-layer id instead of the layer's own
        # (possibly repeated) non-zero value.
        mask_names = labels.coords["mask"].values
        layer_ids = xr.DataArray(np.arange(1, labels.sizes["mask"] + 1), dims="mask")
        synthetic_labels = xr.where(labels != 0, layer_ids, 0)

        has_overlap = bool(((labels != 0).sum(dim="mask") > 1).any().values)

        if not has_overlap:
            # Sum across mask dim: no-overlap guarantees each voxel keeps its
            # synthetic id.
            result = _grouped_reduce(
                data, synthetic_labels.sum(dim="mask"), spatial_dims, reduction
            )
        else:
            result = _grouped_reduce(data, synthetic_labels, spatial_dims, reduction)

        region_names = [mask_names[int(r) - 1] for r in result.coords["region"].values]
        return result.assign_coords(region=region_names)
    else:
        return _grouped_reduce(data, labels, list(labels.dims), reduction)
