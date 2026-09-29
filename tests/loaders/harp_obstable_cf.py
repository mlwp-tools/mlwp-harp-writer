"""Test loader: HARP OBSTABLE observations with CF variable metadata.

Wraps the ``harp.obstable`` loader from mlwp-data-loaders, which sets no
metadata on the data variables, and adds the CF ``standard_name``/``units``
the mlwp-harp-writer mapping needs, plus a scalar 2 m ``height`` coordinate
(the only level-dependent quantities loaded here are at 2 m).

Follows the mlwp-data-loaders loader contract, so it can be passed to
``mlwp_data_loaders.load_and_validate_dataset`` by file path.
"""

from __future__ import annotations

import xarray as xr
from mlwp_data_loaders.loaders.harp import obstable

# HARP OBSTABLE parameter -> CF metadata. The units are HARP's own.
CF_ATTRS: dict[str, dict[str, str]] = {
    "T2m": {"standard_name": "air_temperature", "units": "K"},
    "RH2m": {"standard_name": "relative_humidity", "units": "percent"},
    "Pmsl": {"standard_name": "air_pressure_at_mean_sea_level", "units": "hPa"},
    "Ps": {"standard_name": "surface_air_pressure", "units": "hPa"},
    "CCtot": {"standard_name": "cloud_area_fraction", "units": "oktas"},
}


def load_dataset(
    paths: str | list[str], variables: list[str] | None = None, **kwargs
) -> xr.Dataset:
    """Load a HARP OBSTABLE file with CF metadata on the data variables.

    Parameters
    ----------
    paths : str or list of str
        Path to the OBSTABLE SQLite file.
    variables : list of str, optional
        HARP parameters to load, defaults to all of :data:`CF_ATTRS`.
    **kwargs
        Passed to the mlwp-data-loaders ``harp.obstable`` loader.

    Returns
    -------
    xr.Dataset
        Observations on ``(valid_time, point_index)`` with the
        observation/point traits, CF variable attributes and a 2 m ``height``
        coordinate.
    """
    variables = list(CF_ATTRS) if variables is None else variables
    ds_obs = obstable.load_dataset(paths, variables=variables, **kwargs)
    for var in variables:
        ds_obs[var].attrs.update(CF_ATTRS[var])
    return ds_obs.assign_coords(
        height=((), 2.0, {"standard_name": "height", "units": "m"})
    )
