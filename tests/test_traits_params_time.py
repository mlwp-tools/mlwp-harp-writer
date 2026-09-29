"""Tests for trait handling, station metadata and time conversions."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from mlwp_data_specs.api import TIME_TRAIT_ATTR

from mlwp_harp_writer import validate
from mlwp_harp_writer._stations import station_elevation, station_ids
from mlwp_harp_writer._time import lead_time_seconds, to_unix_seconds
from mlwp_harp_writer.traits import get_traits, require_traits


def test_get_traits_normalises_enums(ds_grid_ens):
    """Enum trait values (as set by mlwp-data-loaders) become plain strings."""
    assert get_traits(ds_grid_ens) == {
        "time": "forecast",
        "space": "grid",
        "uncertainty": "ensemble",
    }
    assert get_traits(xr.Dataset())["uncertainty"] == "deterministic"


def test_validate_passes_and_fails(ds_obs):
    """Validation passes on conforming data and lists failures otherwise."""
    validate(ds_obs)
    ds_bad = ds_obs.copy()
    ds_bad["valid_time"].attrs = {}
    with pytest.raises(ValueError, match="does not conform"):
        validate(ds_bad)


def test_require_traits(ds_obs):
    """Unexpected trait values raise with a helpful message."""
    require_traits(ds_obs, context="test", time="observation", space={"point"})
    with pytest.raises(ValueError, match="test requires time trait"):
        require_traits(
            ds_obs.assign_attrs({TIME_TRAIT_ATTR: "forecast"}),
            context="test",
            time="observation",
        )


@pytest.mark.parametrize(
    ("values", "attrs", "expected"),
    [
        (np.array([0, 6]), {"units": "hours"}, [0, 21600]),
        (np.array([0, 3600]), {"units": "s"}, [0, 3600]),
        (np.array([0, 90], dtype="timedelta64[m]"), {}, [0, 5400]),
    ],
)
def test_lead_time_seconds(values, attrs, expected):
    """Lead times in hours, seconds or timedelta64 convert to seconds."""
    da_lead_time = xr.DataArray(values, dims="lead_time", attrs=attrs)
    np.testing.assert_array_equal(lead_time_seconds(da_lead_time), expected)


def test_lead_time_seconds_bad_units():
    """Unsupported lead time units raise."""
    da_lead_time = xr.DataArray([0, 1], dims="lead_time", attrs={"units": "days"})
    with pytest.raises(ValueError, match="Unsupported lead_time units"):
        lead_time_seconds(da_lead_time)


def test_to_unix_seconds():
    """Datetimes convert to integer seconds since the epoch."""
    values = np.array(["1970-01-01T00:00:01", "2026-01-01"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(to_unix_seconds(values), [1, 1767225600])


def test_station_ids_and_elevation(ds_obs):
    """SIDs come from ``code`` and elevation from elevation/altitude coords."""
    assert station_ids(ds_obs).dtype == np.int64
    ds_str = ds_obs.assign_coords(code=("point_index", ["a", "b", "c", "d"]))
    assert station_ids(ds_str).tolist() == ["a", "b", "c", "d"]
    ds_alt = ds_obs.rename({"elevation": "altitude"})
    np.testing.assert_array_equal(station_elevation(ds_alt), ds_obs["elevation"])
    assert station_elevation(ds_obs.drop_vars("elevation")) is None
