"""Test loader for MEPS forecasts from MET Norway's THREDDS archive.

Follows the mlwp-data-loaders loader contract (a ``load_dataset`` function
returning an mlwp-data-specs conforming ``xr.Dataset`` with traits set), so it
can be passed to ``mlwp_data_loaders.load_and_validate_dataset`` by file path.

The MEPS netCDF files are close to CF already. This loader adds what the
mlwp-data-specs traits, mxalign and the mlwp-harp-writer CF mapping need:

- ``reference_time``/``lead_time`` forecast dims instead of the valid ``time``
- ``xc``/``yc`` projection coordinates and a cartopy CRS (for mxalign's
  xarray interpolator)
- ``degrees_north``/``degrees_east`` spellings on latitude/longitude
- ``standard_name="height"`` on the 2 m height dimension, the CF standard
  name for mean sea level pressure, and no size-1 height dims on quantities
  that don't depend on the level
"""

from __future__ import annotations

import mxalign  # noqa: F401  (registers the ds.mx accessor)
import numpy as np
import xarray as xr
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
    Space,
    Time,
    Uncertainty,
)

THREDDS_URL = "https://thredds.met.no/thredds/dodsC/meps25epsarchive"

# MEPS variable -> fixes applied by the loader:
#   height_dim: size-1 height dimension to mark with standard_name="height"
#   drop_dim: size-1 dimension to squeeze out (level-independent quantities)
#   standard_name: CF standard name to set (MEPS uses an alias)
VARIABLES: dict[str, dict[str, str]] = {
    "air_temperature_2m": {"height_dim": "height1"},
    "relative_humidity_2m": {"height_dim": "height1"},
    "air_pressure_at_sea_level": {
        "drop_dim": "height_above_msl",
        "standard_name": "air_pressure_at_mean_sea_level",
    },
    "surface_air_pressure": {"drop_dim": "height0"},
    "cloud_area_fraction": {},
}


def meps_url(cycle: str) -> str:
    """Return the OPeNDAP URL of a deterministic MEPS forecast.

    Parameters
    ----------
    cycle : str
        Forecast cycle as ``YYYYMMDDTHH``, e.g. ``"20190217T00"``.

    Returns
    -------
    str
        OPeNDAP URL of the MEPS control member post-processed file.
    """
    day = f"{cycle[:4]}/{cycle[4:6]}/{cycle[6:8]}"
    return f"{THREDDS_URL}/{day}/meps_mbr0_pp_2_5km_{cycle}Z.nc"


def load_dataset(
    paths: str | list[str],
    variables: list[str] | None = None,
    lead_hours: tuple[int, ...] = (0, 6, 12, 24),
    stride: int = 5,
    **kwargs,
) -> xr.Dataset:
    """Load MEPS forecasts from THREDDS into an mlwp-data-specs forecast dataset.

    Parameters
    ----------
    paths : str or list of str
        OPeNDAP URLs, one per forecast cycle (see :func:`meps_url`).
    variables : list of str, optional
        MEPS variables to load, defaults to all of :data:`VARIABLES`.
    lead_hours : tuple of int, optional
        Lead times to load, in hours.
    stride : int, optional
        Load every ``stride``-th grid point in x and y, to keep downloads small.
    **kwargs
        Passed to ``xr.open_dataset``.

    Returns
    -------
    xr.Dataset
        Forecast on ``(reference_time, lead_time, yc, xc)`` with the
        ``forecast``/``grid``/``deterministic`` traits and a ``crs`` attribute.
    """
    paths = [paths] if isinstance(paths, str) else list(paths)
    variables = list(VARIABLES) if variables is None else variables
    datasets = [
        _load_cycle(path, variables, lead_hours, stride, **kwargs) for path in paths
    ]
    ds_fcst = xr.concat(datasets, dim="reference_time", combine_attrs="override")

    ds_fcst.coords["reference_time"].attrs["standard_name"] = "forecast_reference_time"
    ds_fcst.coords["lead_time"].attrs.update(
        standard_name="forecast_period", units="hours"
    )
    ds_fcst.coords["latitude"].attrs.update(
        standard_name="latitude", units="degrees_north"
    )
    ds_fcst.coords["longitude"].attrs.update(
        standard_name="longitude", units="degrees_east"
    )
    ds_fcst.attrs[TIME_TRAIT_ATTR] = Time.FORECAST
    ds_fcst.attrs[SPACE_TRAIT_ATTR] = Space.GRID
    ds_fcst.attrs[UNCERTAINTY_TRAIT_ATTR] = Uncertainty.DETERMINISTIC
    return ds_fcst.mx.add_crs(datasets[0].attrs["projection"])


def _load_cycle(
    path: str,
    variables: list[str],
    lead_hours: tuple[int, ...],
    stride: int,
    **kwargs,
) -> xr.Dataset:
    """Load one MEPS forecast cycle and apply the metadata fixes.

    Parameters
    ----------
    path : str
        OPeNDAP URL of the cycle.
    variables : list of str
        MEPS variables to load.
    lead_hours : tuple of int
        Lead times to load, in hours.
    stride : int
        Spatial subsampling in x and y.
    **kwargs
        Passed to ``xr.open_dataset``.

    Returns
    -------
    xr.Dataset
        The cycle on ``(reference_time, lead_time, yc, xc)``, with the
        projection description in ``attrs["projection"]``.
    """
    # No dask: lazy backend indexing sends the subset to the server as a DAP
    # hyperslab. With dask the default chunk is the whole variable (all lead
    # times), which makes every read download hundreds of MB.
    ds_meps = xr.open_dataset(path, engine="netcdf4", chunks=None, **kwargs)
    reference_time = ds_meps["forecast_reference_time"].values
    lead_times = np.array(lead_hours, dtype="timedelta64[h]").astype("timedelta64[ns]")
    projection = _cartopy_projection(ds_meps["projection_lambert"].attrs)

    ds_cycle = (
        ds_meps[variables]
        .drop_vars("forecast_reference_time", errors="ignore")
        .sel(time=reference_time + lead_times)
        .isel(x=slice(None, None, stride), y=slice(None, None, stride))
        .load()
    )
    for var in variables:
        ds_cycle[var] = _fix_variable(ds_cycle[var], **VARIABLES.get(var, {}))
    # size-1 dims squeezed out of every variable remain as dataset coordinates
    unused = [
        dim
        for dim in ds_cycle.dims
        if not any(dim in ds_cycle[var].dims for var in ds_cycle.data_vars)
    ]
    ds_cycle = ds_cycle.drop_vars(unused)

    ds_cycle = (
        ds_cycle.assign_coords(lead_time=("time", lead_times))
        .swap_dims({"time": "lead_time"})
        .drop_vars("time")
        .expand_dims(reference_time=[reference_time])
        .rename({"x": "xc", "y": "yc"})
    )
    ds_cycle.attrs = {"projection": projection}
    return ds_cycle


def _fix_variable(
    da_var: xr.DataArray,
    height_dim: str | None = None,
    drop_dim: str | None = None,
    standard_name: str | None = None,
) -> xr.DataArray:
    """Apply the per-variable metadata fixes.

    Parameters
    ----------
    da_var : xr.DataArray
        MEPS variable.
    height_dim : str, optional
        Size-1 height dimension to mark with ``standard_name="height"``.
    drop_dim : str, optional
        Size-1 dimension to squeeze out.
    standard_name : str, optional
        CF standard name to set.

    Returns
    -------
    xr.DataArray
        The fixed variable.
    """
    if drop_dim is not None:
        da_var = da_var.squeeze(drop_dim, drop=True)
    if height_dim is not None:
        da_var[height_dim].attrs["standard_name"] = "height"
    if standard_name is not None:
        da_var.attrs["standard_name"] = standard_name
    return da_var


def _cartopy_projection(attrs: dict) -> dict:
    """Describe the MEPS Lambert conformal projection for ``ds.mx.add_crs``.

    Parameters
    ----------
    attrs : dict
        Attributes of the MEPS ``projection_lambert`` grid mapping variable.

    Returns
    -------
    dict
        Projection description in the format of
        ``mxalign.utils.projections.create_cartopy_crs``.
    """
    radius = float(attrs["earth_radius"])
    return {
        "projection": "lcc",
        "kws_projection": {
            "central_longitude": float(attrs["longitude_of_central_meridian"]),
            "central_latitude": float(attrs["latitude_of_projection_origin"]),
            "standard_parallels": [float(p) for p in attrs["standard_parallel"]],
        },
        "kws_globe": {"semimajor_axis": radius, "semiminor_axis": radius},
    }
