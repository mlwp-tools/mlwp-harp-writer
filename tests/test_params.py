"""Tests for mapping CF-described variables to HARP parameters and units."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mlwp_harp_writer import CF_TO_HARP, HarpParam, harp_variables

HEIGHT_ATTRS = {"standard_name": "height", "units": "m"}


def make_obs_like(values: float = 1.0, **attrs) -> xr.DataArray:
    """Create a small ``(valid_time, point_index)`` variable.

    Parameters
    ----------
    values : float, optional
        Value to fill the variable with.
    **attrs
        Attributes of the variable, e.g. ``standard_name`` and ``units``.

    Returns
    -------
    xr.DataArray
        The variable.
    """
    return xr.DataArray(
        np.full((2, 3), values), dims=("valid_time", "point_index"), attrs=attrs
    )


def with_scalar_height(ds_in: xr.Dataset, height: float, units: str = "m"):
    """Add a Dataset-level scalar height coordinate.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset to add the coordinate to.
    height : float
        Height value.
    units : str, optional
        Units of the height.

    Returns
    -------
    xr.Dataset
        The dataset with a scalar ``height`` coordinate.
    """
    return ds_in.assign_coords(
        height=xr.DataArray(height, attrs={"standard_name": "height", "units": units})
    )


def test_scalar_height_maps_near_surface_temperature():
    """``air_temperature`` at a 2 m height coordinate is T2m, height dropped."""
    ds_in = with_scalar_height(
        xr.Dataset(
            {"t": make_obs_like(280.0, standard_name="air_temperature", units="K")}
        ),
        2.0,
    )
    mapped = harp_variables(ds_in)
    assert list(mapped) == ["T2m"]
    assert mapped["T2m"].attrs["units"] == "K"
    assert "height" not in mapped["T2m"].coords
    np.testing.assert_allclose(mapped["T2m"], 280.0)


def test_height_in_km_and_unmapped_height_skipped():
    """Heights are converted to metres; a height without a HARP parameter
    (100 m wind) is skipped."""
    ds_km = with_scalar_height(
        xr.Dataset({"t": make_obs_like(standard_name="air_temperature", units="K")}),
        0.002,
        units="km",
    )
    assert list(harp_variables(ds_km)) == ["T2m"]

    ds_100m = with_scalar_height(
        xr.Dataset({"ws": make_obs_like(standard_name="wind_speed", units="m s-1")}),
        100.0,
    )
    assert harp_variables(ds_100m) == {}


def test_size1_height_dims_for_mixed_heights():
    """Variables at different heights use their own size-1 height dims."""
    ds_in = xr.Dataset(
        {
            "t": make_obs_like(280.0, standard_name="air_temperature", units="K"),
            "ws": make_obs_like(5.0, standard_name="wind_speed", units="m s-1"),
        }
    )
    ds_in["t"] = ds_in["t"].expand_dims(height_2m=[2.0])
    ds_in["ws"] = ds_in["ws"].expand_dims(height_10m=[10.0])
    ds_in["height_2m"].attrs.update(HEIGHT_ATTRS)
    ds_in["height_10m"].attrs.update(HEIGHT_ATTRS)

    mapped = harp_variables(ds_in)
    assert sorted(mapped) == ["S10m", "T2m"]
    assert mapped["T2m"].dims == ("valid_time", "point_index")
    assert mapped["S10m"].attrs["units"] == "m/s"


def test_own_height_dim_wins_over_scalar_height():
    """A variable's own height dimension takes precedence over a scalar one."""
    ds_in = xr.Dataset({"ws": make_obs_like(standard_name="wind_speed", units="m/s")})
    ds_in["ws"] = ds_in["ws"].expand_dims(height_10m=[10.0])
    ds_in["height_10m"].attrs.update(HEIGHT_ATTRS)
    ds_in = with_scalar_height(ds_in, 2.0)
    assert list(harp_variables(ds_in)) == ["S10m"]


def test_missing_height_raises_with_example():
    """Level-dependent standard names need a height coordinate."""
    ds_in = xr.Dataset({"t": make_obs_like(standard_name="air_temperature", units="K")})
    with pytest.raises(ValueError, match="needs a height coordinate") as excinfo:
        harp_variables(ds_in)
    assert "assign_coords" in str(excinfo.value)
    assert "expand_dims" in str(excinfo.value)


def test_height_without_units_raises():
    """A height coordinate must carry units."""
    ds_in = xr.Dataset(
        {"t": make_obs_like(standard_name="air_temperature", units="K")},
        coords={"height": 2.0},
    )
    with pytest.raises(ValueError, match="needs units"):
        harp_variables(ds_in)


def test_ambiguous_height_raises():
    """Two scalar height coordinates on one variable are ambiguous."""
    ds_in = xr.Dataset({"t": make_obs_like(standard_name="air_temperature", units="K")})
    ds_in = ds_in.assign_coords(
        height_a=xr.DataArray(2.0, attrs=HEIGHT_ATTRS),
        height_b=xr.DataArray(10.0, attrs=HEIGHT_ATTRS),
    )
    with pytest.raises(ValueError, match="several height coordinates"):
        harp_variables(ds_in)


@pytest.mark.parametrize(
    "ds_in",
    [
        xr.Dataset(
            {"t": make_obs_like(standard_name="air_temperature", units="K")}
        ).expand_dims(pressure=[850.0, 500.0]),
        xr.Dataset(
            {"t": make_obs_like(standard_name="air_temperature", units="K")}
        ).expand_dims(height=[2.0, 10.0]),
    ],
    ids=["pressure", "several-heights"],
)
def test_upper_air_not_supported(ds_in):
    """Pressure levels and several heights are not supported."""
    if "height" in ds_in.coords:
        ds_in["height"].attrs.update(HEIGHT_ATTRS)
    with pytest.raises(NotImplementedError, match="single-level"):
        harp_variables(ds_in)


def test_level_independent_parameter_needs_no_height():
    """Mean sea level pressure maps without a height, converted to hPa."""
    ds_in = xr.Dataset(
        {
            "msl": make_obs_like(
                101300.0, standard_name="air_pressure_at_mean_sea_level", units="Pa"
            )
        }
    )
    mapped = harp_variables(ds_in)
    assert mapped["Pmsl"].attrs["units"] == "hPa"
    np.testing.assert_allclose(mapped["Pmsl"], 1013.0)


@pytest.mark.parametrize(
    ("standard_name", "units", "value", "harp_name", "expected"),
    [
        ("air_temperature", "degC", 10.0, "T2m", 283.15),
        ("relative_humidity", "1", 0.5, "RH2m", 50.0),
        ("cloud_area_fraction", "1", 0.5, "CCtot", 4.0),
        ("cloud_area_fraction", "%", 50.0, "CCtot", 4.0),
        ("surface_air_pressure", "Pa", 100000.0, "Ps", 1000.0),
    ],
)
def test_unit_conversions(standard_name, units, value, harp_name, expected):
    """Values are converted to the units HARP uses."""
    ds_in = with_scalar_height(
        xr.Dataset(
            {"v": make_obs_like(value, standard_name=standard_name, units=units)}
        ),
        2.0,
    )
    mapped = harp_variables(ds_in)
    np.testing.assert_allclose(mapped[harp_name], expected)
    assert (
        mapped[harp_name].attrs["units"]
        == dict((p.name, p.units) for p in CF_TO_HARP.values())[harp_name]
    )


@pytest.mark.parametrize(
    "attrs", [{"units": "furlong"}, {}], ids=["unknown", "missing"]
)
def test_bad_units_raise(attrs):
    """Unknown or missing units cannot be converted."""
    ds_in = xr.Dataset(
        {"msl": make_obs_like(standard_name="air_pressure_at_mean_sea_level", **attrs)}
    )
    with pytest.raises(ValueError, match="units"):
        harp_variables(ds_in)


def test_missing_standard_name_skipped():
    """Variables without a standard_name are skipped."""
    ds_in = xr.Dataset({"x": make_obs_like(units="K")})
    assert harp_variables(ds_in) == {}


def test_params_override():
    """``params`` maps variables without CF metadata explicitly."""
    ds_in = xr.Dataset(
        {
            "T2m": make_obs_like(10.0, units="degC"),
            "foo": make_obs_like(3.0, units="things"),
        }
    )
    mapped = harp_variables(ds_in, {"T2m": "T2m", "foo": HarpParam("Foo", "")})
    np.testing.assert_allclose(mapped["T2m"], 283.15)
    assert mapped["T2m"].attrs["units"] == "K"
    assert mapped["Foo"].attrs["units"] == "things"


def test_duplicate_harp_names_raise():
    """Two variables mapping to one HARP parameter is an error."""
    ds_in = with_scalar_height(
        xr.Dataset(
            {
                "a": make_obs_like(standard_name="air_temperature", units="K"),
                "b": make_obs_like(standard_name="air_temperature", units="K"),
            }
        ),
        2.0,
    )
    with pytest.raises(ValueError, match="Multiple variables"):
        harp_variables(ds_in)
