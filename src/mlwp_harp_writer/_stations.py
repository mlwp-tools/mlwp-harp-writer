"""Station metadata helpers shared by the forecast and observation writers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

# Tolerance (degrees) for treating two point datasets as the same stations
COORD_TOLERANCE = 1e-4

# Point metadata coordinates from mlwp-data-specs (plus "altitude", used by the
# mlwp-data-loaders HARP obstable loader). mxalign interpolation only keeps
# latitude/longitude, so the others are copied from the observations.
STATION_COORDS = (
    "latitude",
    "longitude",
    "code",
    "elevation",
    "altitude",
    "name",
    "country",
)


def station_coords(ds_in: xr.Dataset) -> dict[str, xr.Variable]:
    """Return the per-station coordinates of a point dataset.

    Parameters
    ----------
    ds_in : xr.Dataset
        Point dataset with a ``point_index`` dimension.

    Returns
    -------
    dict[str, xr.Variable]
        The coordinates in :data:`STATION_COORDS` that are defined along
        ``point_index`` only.
    """
    return {
        name: ds_in.coords[name].variable
        for name in STATION_COORDS
        if name in ds_in.coords and ds_in.coords[name].dims == ("point_index",)
    }


def match_stations(ds_points: xr.Dataset, ds_stations: xr.Dataset) -> bool:
    """Check whether two point datasets are located at the same stations.

    Parameters
    ----------
    ds_points : xr.Dataset
        Point dataset, e.g. a forecast interpolated to the stations by mxalign.
    ds_stations : xr.Dataset
        Point dataset defining the stations, e.g. the observations.

    Returns
    -------
    bool
        True if both have the same number of points and matching
        ``latitude``/``longitude`` (within :data:`COORD_TOLERANCE`) per
        ``point_index``.
    """
    if ds_points.sizes.get("point_index") != ds_stations.sizes.get("point_index"):
        return False
    return all(
        np.allclose(ds_points[c].values, ds_stations[c].values, atol=COORD_TOLERANCE)
        for c in ("latitude", "longitude")
    )


def with_station_coords(
    ds_points: xr.Dataset, ds_stations: xr.Dataset | None
) -> xr.Dataset:
    """Add missing station metadata to a point dataset.

    mxalign interpolation keeps only ``latitude``/``longitude`` of the target
    stations, but HARP needs the station ``code`` as ``SID``. If ``ds_points``
    has no ``code`` coordinate, the station coordinates are copied from
    ``ds_stations``.

    Parameters
    ----------
    ds_points : xr.Dataset
        Point dataset to complete.
    ds_stations : xr.Dataset or None
        Point dataset to copy the station coordinates from.

    Returns
    -------
    xr.Dataset
        ``ds_points``, with the station coordinates of ``ds_stations`` if it
        had no ``code`` coordinate.

    Raises
    ------
    ValueError
        If ``ds_points`` and ``ds_stations`` are not at the same stations.
    """
    if ds_stations is None or "code" in ds_points.coords:
        return ds_points
    if not match_stations(ds_points, ds_stations):
        raise ValueError(
            "Cannot copy station metadata: the points are not at the same "
            "stations (point_index/latitude/longitude) as ds_stations"
        )
    return ds_points.assign_coords(station_coords(ds_stations))


def station_ids(ds_in: xr.Dataset) -> np.ndarray:
    """Return HARP station ids (``SID``) for each ``point_index``.

    Parameters
    ----------
    ds_in : xr.Dataset
        Point dataset. The ``code`` coordinate is used when present, otherwise
        ``point_index`` positions are used.

    Returns
    -------
    np.ndarray
        int64 ids when every id is integer-like (as in HARP FCTABLE/OBSTABLE
        files), otherwise string ids.
    """
    if "code" in ds_in.coords:
        codes = ds_in["code"].values
    else:
        logger.warning("No 'code' coordinate found, using point_index as SID")
        codes = np.arange(ds_in.sizes["point_index"])

    as_numeric = pd.to_numeric(pd.Series(codes), errors="coerce")
    if as_numeric.notna().all() and (as_numeric % 1 == 0).all():
        return as_numeric.to_numpy().astype(np.int64)
    logger.warning("Station codes are not all integers, writing SID as strings")
    return np.asarray(codes).astype(str)


def station_elevation(ds_in: xr.Dataset) -> np.ndarray | None:
    """Return the station elevation for each ``point_index``, if available.

    Parameters
    ----------
    ds_in : xr.Dataset
        Point dataset with an ``elevation`` (mlwp-data-specs) or ``altitude``
        (mlwp-data-loaders HARP obstable loader) coordinate.

    Returns
    -------
    np.ndarray or None
        Station elevations, or ``None`` if neither coordinate exists.
    """
    for name in ("elevation", "altitude"):
        if name in ds_in.coords:
            return ds_in[name].values
    return None
