"""Synthetic mlwp-data-specs conforming datasets shared by the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
    Space,
    Time,
    Uncertainty,
)

REFERENCE_TIMES = pd.to_datetime(["2026-01-01T00", "2026-01-01T12"])
LEAD_HOURS = np.array([0, 6, 12])
MEMBERS = np.array([0, 1, 2])
STATION_CODES = np.array([6030, 6060, 6180, 6186])
STATION_LATS = np.array([55.2, 56.3, 55.6, 57.1])
STATION_LONS = np.array([9.1, 10.7, 12.6, 11.3])
STATION_ELEVATIONS = np.array([10.0, 45.0, 5.0, 90.0])

# mxalign `refactor/alignment` (81482db): global `align_time` reindexes the
# observations on the flattened (reference_time, lead_time) valid times, which
# contain duplicates whenever forecast cycles overlap (here 00 UTC +12 h ==
# 12 UTC +0 h). Remove this marker once mxalign deduplicates them (e.g. with
# np.unique) - strict=True makes the tests fail as soon as that happens.
# See docs/upstream-issues/mxalign-align-time-overlapping-valid-times.md

MXALIGN_OVERLAP_BUG = pytest.mark.xfail(
    raises=pd.errors.InvalidIndexError,
    strict=True,
    reason=(
        "mxalign refactor/alignment: global align_time fails on overlapping "
        "valid times (reindex on non-unique index)"
    ),
)

# mxalign `refactor/alignment` (81482db): the delaunay interpolator rebuilds
# each variable from a blank map_blocks template, dropping the variable attrs
# (standard_name, units) and non-dimension coords (e.g. a scalar height), so
# the CF-based HARP mapping skips or rejects the variables afterwards.
# See docs/upstream-issues/mxalign-delaunay-drops-variable-metadata.md
MXALIGN_DELAUNAY_METADATA_BUG = pytest.mark.xfail(
    strict=True,
    reason=(
        "mxalign refactor/alignment: delaunay interpolation drops variable "
        "attrs (standard_name/units) and scalar coords"
    ),
)


def linear_field(lat, lon, ref_idx, lead_h, member=0):
    """Return a temperature field that is linear in space.

    Linear fields are reproduced exactly by linear interpolation, so tests
    can compare interpolated values with this analytic expression.

    Parameters
    ----------
    lat, lon : array_like
        Latitudes and longitudes (broadcastable).
    ref_idx : int
        Index of the forecast reference time.
    lead_h : float
        Lead time in hours.
    member : int, optional
        Ensemble member number.

    Returns
    -------
    np.ndarray
        Temperatures in K.
    """
    return 270.0 + 0.5 * lat + 0.2 * lon + ref_idx + lead_h / 6.0 + 0.1 * member


def _set_latlon_attrs(ds_in: xr.Dataset) -> xr.Dataset:
    """Set the CF attributes mlwp-data-specs requires on latitude/longitude.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset with ``latitude`` and ``longitude`` coordinates.

    Returns
    -------
    xr.Dataset
        The same dataset (modified in place) for chaining.
    """
    ds_in["latitude"].attrs.update(standard_name="latitude", units="degrees_north")
    ds_in["longitude"].attrs.update(standard_name="longitude", units="degrees_east")
    return ds_in


def add_t2m_metadata(ds_in: xr.Dataset) -> xr.Dataset:
    """Describe ``2t`` as 2 m air temperature with CF metadata.

    Sets the ``standard_name``/``units`` attributes and adds a scalar ``height``
    coordinate, which is what the HARP parameter mapping needs.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset with a ``2t`` variable in K.

    Returns
    -------
    xr.Dataset
        The dataset with the metadata added.
    """
    ds_in["2t"].attrs.update(standard_name="air_temperature", units="K")
    return ds_in.assign_coords(
        height=xr.DataArray(2.0, attrs={"standard_name": "height", "units": "m"})
    )


def make_grid_forecast(ensemble: bool = False, stacked: bool = False) -> xr.Dataset:
    """Create a regular lat/lon grid forecast of 2 m temperature.

    Lead times are ``timedelta64`` (with ``units`` attrs for mlwp-data-specs),
    as mxalign requires and as the anemoi-inference loader provides.

    Parameters
    ----------
    ensemble : bool, optional
        Add a ``member`` dimension with three members.
    stacked : bool, optional
        Flatten the grid to a trailing ``grid_index`` dimension (with
        ``latitude``/``longitude`` coordinates along it), as needed by the
        mxalign delaunay interpolator and as the anemoi loaders return.

    Returns
    -------
    xr.Dataset
        Forecast with ``2t`` on ``(reference_time, lead_time, [member],
        latitude, longitude)`` (or ``grid_index``) and traits set as
        mlwp-data-loaders does (Enum members).
    """
    lats = np.arange(54.0, 58.01, 0.5)
    lons = np.arange(8.0, 13.01, 0.5)
    ref_idx = np.arange(len(REFERENCE_TIMES))[:, None, None, None, None]
    lead_h = LEAD_HOURS[None, :, None, None, None]
    member = (MEMBERS if ensemble else np.array([0]))[None, None, :, None, None]
    values = linear_field(
        lats[None, None, None, :, None],
        lons[None, None, None, None, :],
        ref_idx,
        lead_h,
        member,
    )
    dims = ["reference_time", "lead_time", "member", "latitude", "longitude"]
    coords = {
        "reference_time": REFERENCE_TIMES.values,
        "lead_time": LEAD_HOURS.astype("timedelta64[h]").astype("timedelta64[ns]"),
        "latitude": lats,
        "longitude": lons,
    }
    if ensemble:
        coords["member"] = MEMBERS
    else:
        values = values[:, :, 0]
        dims.remove("member")

    ds_fcst = xr.Dataset({"2t": (dims, values, {"units": "K"})}, coords=coords)
    if stacked:
        ds_fcst = ds_fcst.stack(grid_index=["latitude", "longitude"]).reset_index(
            "grid_index"
        )
    ds_fcst["reference_time"].attrs["standard_name"] = "forecast_reference_time"
    ds_fcst["lead_time"].attrs.update(standard_name="forecast_period", units="hours")
    if ensemble:
        ds_fcst["member"].attrs["standard_name"] = "realization"
    _set_latlon_attrs(ds_fcst)
    ds_fcst.attrs[TIME_TRAIT_ATTR] = Time.FORECAST
    ds_fcst.attrs[SPACE_TRAIT_ATTR] = Space.GRID
    ds_fcst.attrs[UNCERTAINTY_TRAIT_ATTR] = (
        Uncertainty.ENSEMBLE if ensemble else Uncertainty.DETERMINISTIC
    )
    return add_t2m_metadata(ds_fcst)


def make_point_forecast(ensemble: bool = False) -> xr.Dataset:
    """Create a point forecast at the test stations.

    Parameters
    ----------
    ensemble : bool, optional
        Add a ``member`` dimension with three members.

    Returns
    -------
    xr.Dataset
        Forecast with ``2t`` on ``(reference_time, lead_time, [member],
        point_index)`` and station ``code``/``elevation`` coordinates.
    """
    ref_idx = np.arange(len(REFERENCE_TIMES))[:, None, None, None]
    lead_h = LEAD_HOURS[None, :, None, None]
    member = (MEMBERS if ensemble else np.array([0]))[None, None, :, None]
    values = linear_field(STATION_LATS, STATION_LONS, ref_idx, lead_h, member)
    dims = ["reference_time", "lead_time", "member", "point_index"]
    coords = {
        "reference_time": REFERENCE_TIMES.values,
        "lead_time": LEAD_HOURS,
        "latitude": ("point_index", STATION_LATS),
        "longitude": ("point_index", STATION_LONS),
        "code": ("point_index", STATION_CODES),
        "elevation": ("point_index", STATION_ELEVATIONS),
    }
    if ensemble:
        coords["member"] = MEMBERS
    else:
        values = values[:, :, 0]
        dims.remove("member")

    ds_fcst = xr.Dataset({"2t": (dims, values, {"units": "K"})}, coords=coords)
    ds_fcst["reference_time"].attrs["standard_name"] = "forecast_reference_time"
    ds_fcst["lead_time"].attrs.update(standard_name="forecast_period", units="hours")
    if ensemble:
        ds_fcst["member"].attrs["standard_name"] = "realization"
    _set_latlon_attrs(ds_fcst)
    ds_fcst.attrs[TIME_TRAIT_ATTR] = Time.FORECAST
    ds_fcst.attrs[SPACE_TRAIT_ATTR] = Space.POINT
    ds_fcst.attrs[UNCERTAINTY_TRAIT_ATTR] = (
        Uncertainty.ENSEMBLE if ensemble else Uncertainty.DETERMINISTIC
    )
    return add_t2m_metadata(ds_fcst)


def make_observations(end: str = "2026-01-02T00", altitude: bool = False) -> xr.Dataset:
    """Create hourly point observations of 2 m temperature.

    Parameters
    ----------
    end : str, optional
        Last observation time (inclusive); observations start 2026-01-01T00.
    altitude : bool, optional
        Name the station height coordinate ``altitude`` (as the
        mlwp-data-loaders HARP obstable loader does) instead of ``elevation``.

    Returns
    -------
    xr.Dataset
        Observations with ``2t`` on ``(valid_time, point_index)``.
    """
    valid_times = pd.date_range("2026-01-01T00", end, freq="1h")
    hours = np.arange(len(valid_times))[:, None]
    values = 275.0 + 0.1 * hours + 0.01 * STATION_CODES[None, :] / 100.0
    ds_obs = xr.Dataset(
        {"2t": (["valid_time", "point_index"], values, {"units": "K"})},
        coords={
            "valid_time": valid_times.values,
            "latitude": ("point_index", STATION_LATS),
            "longitude": ("point_index", STATION_LONS),
            "code": ("point_index", STATION_CODES),
            ("altitude" if altitude else "elevation"): (
                "point_index",
                STATION_ELEVATIONS,
            ),
        },
    )
    ds_obs["valid_time"].attrs["standard_name"] = "time"
    _set_latlon_attrs(ds_obs)
    ds_obs.attrs[TIME_TRAIT_ATTR] = Time.OBSERVATION
    ds_obs.attrs[SPACE_TRAIT_ATTR] = Space.POINT
    ds_obs.attrs[UNCERTAINTY_TRAIT_ATTR] = Uncertainty.DETERMINISTIC
    return add_t2m_metadata(ds_obs)


@pytest.fixture
def ds_grid_fcst() -> xr.Dataset:
    """Deterministic grid forecast, see :func:`make_grid_forecast`.

    Returns
    -------
    xr.Dataset
        The forecast dataset.
    """
    return make_grid_forecast()


@pytest.fixture
def ds_grid_ens() -> xr.Dataset:
    """Ensemble grid forecast, see :func:`make_grid_forecast`.

    Returns
    -------
    xr.Dataset
        The forecast dataset.
    """
    return make_grid_forecast(ensemble=True)


@pytest.fixture
def ds_grid_fcst_stacked() -> xr.Dataset:
    """Deterministic grid forecast on ``grid_index``, see :func:`make_grid_forecast`.

    Returns
    -------
    xr.Dataset
        The forecast dataset.
    """
    return make_grid_forecast(stacked=True)


@pytest.fixture
def ds_grid_ens_stacked() -> xr.Dataset:
    """Ensemble grid forecast on ``grid_index``, see :func:`make_grid_forecast`.

    Returns
    -------
    xr.Dataset
        The forecast dataset.
    """
    return make_grid_forecast(ensemble=True, stacked=True)


@pytest.fixture
def ds_point_fcst() -> xr.Dataset:
    """Deterministic point forecast, see :func:`make_point_forecast`.

    Returns
    -------
    xr.Dataset
        The forecast dataset.
    """
    return make_point_forecast()


@pytest.fixture
def ds_point_ens() -> xr.Dataset:
    """Ensemble point forecast, see :func:`make_point_forecast`.

    Returns
    -------
    xr.Dataset
        The forecast dataset.
    """
    return make_point_forecast(ensemble=True)


@pytest.fixture
def ds_obs() -> xr.Dataset:
    """Hourly point observations, see :func:`make_observations`.

    Returns
    -------
    xr.Dataset
        The observation dataset.
    """
    return make_observations()
