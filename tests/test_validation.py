"""Modality-independent validation regressions adapted from ConfUSIus.

Probe timing metadata, slice acquisition timing, and fUSI signal attributes are
intentionally excluded from the standalone data model.
"""

import numpy as np
import pytest
import xarray as xr

import voxeldata as vd


@pytest.mark.parametrize("dim", ["k", "time", "component"])
def test_empty_dimensions_are_rejected(
    sample_voxeldata_3dt: xr.DataArray, dim: str
) -> None:
    data = sample_voxeldata_3dt.expand_dims(component=[0])
    with pytest.raises(ValueError, match=f"zero-length dimensions.*'{dim}'"):
        vd.validate_voxeldata(data.isel({dim: slice(0, 0)}))


def test_extra_dimensions_are_optional_and_can_be_forbidden(
    sample_voxeldata_3dt: xr.DataArray,
) -> None:
    data = sample_voxeldata_3dt.expand_dims(region=["roi"]).drop_vars("region")
    vd.validate_voxeldata(data)
    with pytest.raises(ValueError, match="Unexpected dimensions"):
        vd.validate_voxeldata(data, allow_extra_dims=False)


@pytest.mark.parametrize(
    "values, message",
    [
        ([0, 0.5, np.nan, 1.5], "non-finite"),
        ([0, 0.5, np.inf, 1.5], "non-finite"),
        ([0, 0.5, 0.5, 1.5], "monotonic-increasing"),
        ([1.5, 1, 0.5, 0], "monotonic-increasing"),
        (["a", "b", "c", "d"], "numeric"),
    ],
)
def test_invalid_time_coordinates(
    sample_voxeldata_3dt: xr.DataArray, values: list, message: str
) -> None:
    data = sample_voxeldata_3dt.assign_coords(time=values)
    with pytest.raises(ValueError, match=message):
        vd.validate_voxeldata(data)


def test_time_requirement_rejects_absent_and_singleton_axes(
    sample_voxeldata_3dt: xr.DataArray,
) -> None:
    vd.validate_voxeldata(
        sample_voxeldata_3dt, require_time=True, require_uniform_time=True
    )
    with pytest.raises(ValueError, match="must have a 'time' dimension"):
        vd.validate_voxeldata(sample_voxeldata_3dt.isel(time=0), require_time=True)
    with pytest.raises(ValueError, match="more than one time point"):
        vd.validate_voxeldata(sample_voxeldata_3dt.isel(time=[0]), require_time=True)


@pytest.mark.parametrize("selector", ["space", "core", "all", "time", ("time",)])
def test_regular_spacing_selectors(
    sample_voxeldata_3dt: xr.DataArray, selector: str | tuple[str, ...]
) -> None:
    data = sample_voxeldata_3dt.assign_coords(time=[0.0, 0.5, 1.0, 1.8])
    if selector == "space":
        vd.validate_voxeldata(
            data, require_regular_spacing=True, regular_spacing_dims=selector
        )
    else:
        with pytest.raises(ValueError, match="must have regular spacing"):
            vd.validate_voxeldata(
                data, require_regular_spacing=True, regular_spacing_dims=selector
            )
    with pytest.raises(ValueError, match="uniformly spaced"):
        vd.validate_voxeldata(data, require_uniform_time=True)


def test_regular_spacing_rejects_unknown_dimensions(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    with pytest.raises(ValueError, match="not present in data"):
        vd.validate_voxeldata(
            sample_voxeldata_3d,
            require_regular_spacing=True,
            regular_spacing_dims="unknown",
        )


def test_regular_spacing_skips_string_extra_coordinates(
    sample_voxeldata_3dt: xr.DataArray,
) -> None:
    data = sample_voxeldata_3dt.expand_dims(region=["left", "right"])
    vd.validate_voxeldata(
        data, require_regular_spacing=True, regular_spacing_dims="all"
    )


def test_irregular_voxel_labels_are_valid_but_spacing_is_undefined(
    sample_voxeldata_irregular: xr.DataArray,
) -> None:
    vd.validate_voxeldata(sample_voxeldata_irregular)
    with pytest.raises(ValueError, match="must have regular spacing"):
        vd.validate_voxeldata(sample_voxeldata_irregular, require_regular_spacing=True)


def test_descending_spatial_coordinates_are_valid(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    data = sample_voxeldata_3d.isel(i=slice(None, None, -1))
    vd.validate_voxeldata(data)
    assert data.vd.spacing["i"] == 3
    np.testing.assert_allclose(data.vd.direction, np.diag([1, 1, -1]))


@pytest.mark.parametrize(
    "values, message",
    [([0, 2, 1, 3], "strictly monotonic"), (["a", "b", "c", "d"], "numeric")],
)
def test_invalid_voxel_labels(
    sample_voxeldata_3d: xr.DataArray, values: list, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        vd.validate_voxeldata(sample_voxeldata_3d.assign_coords(i=values))


def test_missing_or_misshaped_dimension_coordinate(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    with pytest.raises(ValueError, match="Missing required coordinate"):
        vd.validate_voxeldata(sample_voxeldata_3d.drop_vars("i"))
    with pytest.raises(ValueError, match="1D dimension coordinate"):
        vd.validate_voxeldata(sample_voxeldata_3d.assign_coords(i=("j", [0, 1, 2])))


def test_dimension_names_must_be_strings(plain_volume: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="All dimensions must be strings"):
        vd.validate_voxeldata(xr.DataArray(plain_volume.values, dims=(1, "j", "i")))


@pytest.mark.parametrize("coord_dims", [("time", "grid"), ("grid", "time")])
@pytest.mark.parametrize(
    "failure, message",
    [
        ("string", "numeric"),
        ("nonfinite", "non-finite"),
        ("descending", "monotonic-increasing"),
    ],
)
def test_per_grid_time_validation_uses_named_time_axis(
    sample_voxeldata_grid_time: xr.DataArray,
    coord_dims: tuple[str, str],
    failure: str,
    message: str,
) -> None:
    data = sample_voxeldata_grid_time
    values = np.array([[0.0, 10.0], [0.5, 10.5], [1.0, 11.0]])
    if failure == "string":
        values = values.astype(str)
    elif failure == "nonfinite":
        values[1, 1] = np.nan
    else:
        values[:, 1] = values[::-1, 1]
    coord = xr.DataArray(values, dims=("time", "grid")).transpose(*coord_dims)
    with pytest.raises(ValueError, match=message):
        vd.validate_voxeldata(data.assign_coords(time=coord))


@pytest.mark.parametrize("coord_dims", [("time", "grid"), ("grid", "time")])
def test_singleton_per_grid_time_is_valid(
    sample_voxeldata_grid_time: xr.DataArray, coord_dims: tuple[str, str]
) -> None:
    data = sample_voxeldata_grid_time.isel(time=[0])
    vd.validate_voxeldata(
        data.assign_coords(time=data.coords["time"].transpose(*coord_dims))
    )


@pytest.mark.parametrize(
    "failure, message",
    [("absent", "has no scalar coordinate"), ("nonscalar", "is not scalar")],
)
def test_canonicalize_rejects_unrecoverable_spatial_axis(
    plain_volume: xr.DataArray, failure: str, message: str
) -> None:
    data = plain_volume.isel(j=0, drop=True)
    if failure == "nonscalar":
        data = data.assign_coords(j=("k", [0, 1]))
    with pytest.raises(ValueError, match=message):
        vd.canonicalize_voxeldata(data)


def test_canonicalize_restores_coordinate_attrs_and_values(
    sample_voxeldata_3d: xr.DataArray,
) -> None:
    data = sample_voxeldata_3d
    data.coords["j"].attrs["long_name"] = "native index"
    restored = vd.ensure_voxeldata(data.isel(j=2))
    assert restored.coords["j"].attrs == {"long_name": "native index"}
    np.testing.assert_array_equal(
        restored.values[:, 0], [[8, 9, 10, 11], [20, 21, 22, 23]]
    )
    np.testing.assert_allclose(restored.coords["y"], 24)


def test_dtype_validation(sample_voxeldata_3d: xr.DataArray) -> None:
    vd.validate_voxeldata(sample_voxeldata_3d, require_dtype=np.floating)
    with pytest.raises(TypeError, match="dtype"):
        vd.validate_voxeldata(sample_voxeldata_3d, require_dtype=np.integer)
