"""Mapping of dataset variable names to HARP parameter names."""

from __future__ import annotations

from typing import NamedTuple

import xarray as xr
from loguru import logger


class HarpParam(NamedTuple):
    """A HARP parameter.

    Attributes
    ----------
    name : str
        HARP parameter name, e.g. ``"T2m"``.
    units : str
        Units of the values written for the parameter.
    """

    name: str
    units: str


_T2M = HarpParam("T2m", "K")
_TD2M = HarpParam("Td2m", "K")
_S10M = HarpParam("S10m", "m/s")
_G10M = HarpParam("G10m", "m/s")
_PMSL = HarpParam("Pmsl", "Pa")
_RH2M = HarpParam("RH2m", "%")
_CCTOT = HarpParam("CCtot", "1")

# Common variable names (ECMWF/anemoi short names and CF-ish aliases) mapped
# to HARP parameter names. Precipitation is deliberately not included since
# HARP expects accumulations (e.g. AccPcp1h) that need explicit handling.
DEFAULT_PARAMS: dict[str, HarpParam] = {
    "2t": _T2M,
    "t2m": _T2M,
    "2d": _TD2M,
    "d2m": _TD2M,
    "10si": _S10M,
    "ws10m": _S10M,
    "10fg": _G10M,
    "msl": _PMSL,
    "2r": _RH2M,
    "tcc": _CCTOT,
}
# Variables that already carry HARP names (e.g. from HARP OBSTABLE files read
# with mlwp_data_loaders.loaders.harp.obstable) map to themselves.
DEFAULT_PARAMS.update({p.name: p for p in list(DEFAULT_PARAMS.values())})


def resolve_params(
    ds_in: xr.Dataset, params: dict[str, str | HarpParam] | None = None
) -> dict[str, HarpParam]:
    """Resolve the HARP parameter name and units for each data variable.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset whose data variables should be mapped.
    params : dict, optional
        Overrides/extensions of :data:`DEFAULT_PARAMS`, mapping a variable name
        to either a HARP parameter name or a :class:`HarpParam`.

    Returns
    -------
    dict[str, HarpParam]
        Mapping of variable name to HARP parameter. Units are taken from the
        variable's ``units`` attribute when present, otherwise from the table.
        Variables that cannot be mapped are skipped with a warning.

    Raises
    ------
    ValueError
        If more than one variable maps to the same HARP parameter.
    """
    table = dict(DEFAULT_PARAMS)
    for var, value in (params or {}).items():
        if isinstance(value, str):
            default = table.get(var)
            value = HarpParam(value, default.units if default else "")
        table[var] = value

    resolved = {}
    for var in ds_in.data_vars:
        if var not in table:
            logger.warning(f"No HARP parameter mapping for variable '{var}', skipping")
            continue
        harp_param = table[var]
        units = ds_in[var].attrs.get("units", harp_param.units)
        resolved[var] = HarpParam(harp_param.name, str(units))

    names = [p.name for p in resolved.values()]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"Multiple variables map to the same HARP parameter(s): {duplicates}"
        )
    return resolved
