"""Conversions of time coordinates to the integer seconds HARP expects."""

from __future__ import annotations

import numpy as np
import xarray as xr

_LEAD_TIME_UNITS_TO_SECONDS = {"s": 1, "seconds": 1, "h": 3600, "hours": 3600}


def to_unix_seconds(values) -> np.ndarray:
    """Convert datetime64 values to seconds since the unix epoch.

    Parameters
    ----------
    values : array_like
        Datetime values (anything numpy can cast to ``datetime64[ns]``).

    Returns
    -------
    np.ndarray
        int64 seconds since 1970-01-01T00:00:00.
    """
    values = np.asarray(values, dtype="datetime64[ns]")
    return values.astype("datetime64[s]").astype(np.int64)


def lead_time_seconds(da_lead_time: xr.DataArray) -> np.ndarray:
    """Convert a ``lead_time`` coordinate to seconds.

    Parameters
    ----------
    da_lead_time : xr.DataArray
        Lead times, either decoded ``timedelta64`` values or numeric values
        with a ``units`` attribute (or encoding) of ``s``/``seconds``/``h``/
        ``hours``, the units allowed by mlwp-data-specs.

    Returns
    -------
    np.ndarray
        int64 lead times in seconds.

    Raises
    ------
    ValueError
        If the units are unsupported or the values are not whole seconds.
    """
    values = da_lead_time.values
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[s]").astype(np.int64)

    units = da_lead_time.attrs.get("units", da_lead_time.encoding.get("units"))
    if units not in _LEAD_TIME_UNITS_TO_SECONDS:
        raise ValueError(
            f"Unsupported lead_time units {units!r}; expected one of "
            f"{sorted(_LEAD_TIME_UNITS_TO_SECONDS)} or timedelta64 values"
        )
    seconds = values * _LEAD_TIME_UNITS_TO_SECONDS[units]
    if not np.allclose(seconds, np.round(seconds)):
        raise ValueError("lead_time values are not a whole number of seconds")
    return np.round(seconds).astype(np.int64)


def lead_time_timedelta(da_lead_time: xr.DataArray) -> np.ndarray:
    """Convert a ``lead_time`` coordinate to ``timedelta64[ns]`` values.

    Parameters
    ----------
    da_lead_time : xr.DataArray
        Lead times as accepted by :func:`lead_time_seconds`.

    Returns
    -------
    np.ndarray
        ``timedelta64[ns]`` lead times.
    """
    seconds = lead_time_seconds(da_lead_time)
    return seconds.astype("timedelta64[s]").astype("timedelta64[ns]")
