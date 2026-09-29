"""Map CF-described variables to HARP parameters in HARP's units.

Variables are identified by their CF ``standard_name`` and, for quantities
that depend on the level (e.g. ``air_temperature``), by the height given by a
``height`` coordinate. Values are converted to the units HARP uses for each
parameter (the ``param_units`` in harpIO's ``harp_params``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import xarray as xr
from loguru import logger


class HarpParam(NamedTuple):
    """A HARP parameter.

    Attributes
    ----------
    name : str
        HARP parameter name, e.g. ``"T2m"``.
    units : str
        Units HARP uses for the parameter, e.g. ``"K"``.
    """

    name: str
    units: str


# (CF standard_name, height in m or None) -> HARP parameter. Entries with a
# height are near-surface quantities that need a height coordinate; entries
# with None don't depend on the level. Precipitation is not included since
# HARP expects accumulations (e.g. AccPcp1h) that need explicit handling.
CF_TO_HARP: dict[tuple[str, float | None], HarpParam] = {
    ("air_temperature", 2.0): HarpParam("T2m", "K"),
    ("dew_point_temperature", 2.0): HarpParam("Td2m", "K"),
    ("relative_humidity", 2.0): HarpParam("RH2m", "percent"),
    ("specific_humidity", 2.0): HarpParam("Q2m", "kg/kg"),
    ("wind_speed", 10.0): HarpParam("S10m", "m/s"),
    ("wind_from_direction", 10.0): HarpParam("D10m", "degrees"),
    ("wind_speed_of_gust", 10.0): HarpParam("G10m", "m/s"),
    ("air_pressure_at_mean_sea_level", None): HarpParam("Pmsl", "hPa"),
    ("surface_air_pressure", None): HarpParam("Ps", "hPa"),
    ("cloud_area_fraction", None): HarpParam("CCtot", "oktas"),
    ("visibility_in_air", None): HarpParam("vis", "m"),
}

# Standard names that need a height coordinate to be mapped
LEVEL_DEPENDENT = {name for name, height in CF_TO_HARP if height is not None}

# HARP units -> {source units: (scale, offset)}, so that
# value_in_harp_units = value * scale + offset
_UNIT_CONVERSIONS: dict[str, dict[str, tuple[float, float]]] = {
    "K": {
        "K": (1.0, 0.0),
        "degC": (1.0, 273.15),
        "celsius": (1.0, 273.15),
        "°C": (1.0, 273.15),
    },
    "hPa": {"Pa": (0.01, 0.0), "hPa": (1.0, 0.0)},
    "percent": {"1": (100.0, 0.0), "%": (1.0, 0.0), "percent": (1.0, 0.0)},
    "oktas": {"1": (8.0, 0.0), "%": (0.08, 0.0), "oktas": (1.0, 0.0)},
    "m/s": {"m s-1": (1.0, 0.0), "m/s": (1.0, 0.0), "m s**-1": (1.0, 0.0)},
    "degrees": {"degree": (1.0, 0.0), "degrees": (1.0, 0.0)},
    "kg/kg": {"1": (1.0, 0.0), "kg kg-1": (1.0, 0.0), "kg/kg": (1.0, 0.0)},
    "m": {"m": (1.0, 0.0)},
}

# Units accepted for height coordinates -> factor to metres
_HEIGHT_UNITS = {"m": 1.0, "metre": 1.0, "meter": 1.0, "metres": 1.0, "km": 1000.0}

# Coordinates that mark upper-air data, which is not supported
_VERTICAL_PRESSURE_NAMES = {"pressure", "air_pressure", "isobaric"}

_HEIGHT_EXAMPLE = """\
Add a height coordinate with units, e.g. for a Dataset where all variables
share one height:

    ds = ds.assign_coords(height=((), 2.0, {"standard_name": "height", "units": "m"}))

or, for variables at different heights, a size-1 height dimension per variable:

    ds["2t"] = ds["2t"].expand_dims(height_2m=[2.0])
    ds["height_2m"].attrs.update(standard_name="height", units="m")"""


def harp_variables(
    ds_in: xr.Dataset, params: dict[str, str | HarpParam] | None = None
) -> dict[str, xr.DataArray]:
    """Map the data variables of a dataset to HARP parameters.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset whose data variables should be mapped. Variables are
        identified by their ``standard_name`` attribute and, for
        level-dependent quantities, a ``height`` coordinate with units.
    params : dict, optional
        Explicit mappings of variable name to HARP parameter name or
        :class:`HarpParam`, used instead of the ``standard_name`` (e.g. for
        variables without CF metadata). A HARP name from :data:`CF_TO_HARP`
        is converted to that parameter's units; any other HARP parameter
        is written with the variable's own ``units``.

    Returns
    -------
    dict[str, xr.DataArray]
        Variables keyed by HARP parameter name, with any height dimension
        squeezed out, values converted to HARP units, and the HARP units in
        ``attrs["units"]``. Variables that cannot be mapped are skipped with
        a warning.

    Raises
    ------
    ValueError
        If a level-dependent variable has no (or an ambiguous) height
        coordinate, a height coordinate has no units, a variable's units
        cannot be converted, or two variables map to the same HARP parameter.
    NotImplementedError
        If a variable is on pressure levels or several heights.
    """
    params = params or {}
    known = {p.name: p for p in CF_TO_HARP.values()}

    mapped: dict[str, xr.DataArray] = {}
    for var in ds_in.data_vars:
        da_var = ds_in[var]
        da_var, height = _squeeze_height(da_var)

        if var in params:
            harp_param = params[var]
            if isinstance(harp_param, str):
                harp_param = known.get(harp_param, HarpParam(harp_param, ""))
        else:
            harp_param = _lookup(da_var, height)
            if harp_param is None:
                continue

        if harp_param.name in mapped:
            raise ValueError(
                f"Multiple variables map to the HARP parameter '{harp_param.name}'"
            )
        mapped[harp_param.name] = _to_harp_units(da_var, harp_param)
    return mapped


def _lookup(da_var: xr.DataArray, height: float | None) -> HarpParam | None:
    """Find the HARP parameter for a variable from its CF metadata.

    Parameters
    ----------
    da_var : xr.DataArray
        Data variable, with any height dimension already squeezed out.
    height : float or None
        Height of the variable in metres, from its height coordinate.

    Returns
    -------
    HarpParam or None
        The matching HARP parameter, or None (with a warning) if there is no
        ``standard_name`` or no mapping.

    Raises
    ------
    ValueError
        If a level-dependent variable has no height coordinate.
    """
    standard_name = da_var.attrs.get("standard_name")
    if standard_name is None:
        logger.warning(
            f"Variable '{da_var.name}' has no standard_name attribute, skipping "
            "(set one, or map it explicitly with params=)"
        )
        return None

    if standard_name in LEVEL_DEPENDENT:
        if height is None:
            raise ValueError(
                f"Variable '{da_var.name}' ({standard_name}) needs a height "
                f"coordinate to be mapped to a HARP parameter.\n{_HEIGHT_EXAMPLE}"
            )
        harp_param = CF_TO_HARP.get((standard_name, height))
    else:
        harp_param = CF_TO_HARP.get((standard_name, None))

    if harp_param is None:
        at_height = f" at {height:g} m" if height is not None else ""
        logger.warning(
            f"No HARP parameter for variable '{da_var.name}' "
            f"({standard_name}{at_height}), skipping"
        )
    return harp_param


def _squeeze_height(da_var: xr.DataArray) -> tuple[xr.DataArray, float | None]:
    """Find the height of a variable and drop its height coordinate.

    Parameters
    ----------
    da_var : xr.DataArray
        Data variable.

    Returns
    -------
    da_out : xr.DataArray
        The variable with any size-1 height dimension squeezed out and the
        height coordinate dropped.
    height : float or None
        Height in metres, or None if the variable has no height coordinate.

    Raises
    ------
    ValueError
        If there is more than one height coordinate, or one without units.
    NotImplementedError
        If the variable is on pressure levels or several heights.
    """
    pressure = [
        name
        for name, coord in da_var.coords.items()
        if name in _VERTICAL_PRESSURE_NAMES
        or coord.attrs.get("standard_name") == "air_pressure"
    ]
    if pressure:
        raise NotImplementedError(
            f"Variable '{da_var.name}' is on pressure levels ({pressure}); "
            "only single-level parameters are supported"
        )

    candidates = [
        name
        for name, coord in da_var.coords.items()
        if name == "height" or coord.attrs.get("standard_name") == "height"
    ]
    if not candidates:
        return da_var, None
    # a height dimension of the variable itself wins over a Dataset-level
    # scalar coordinate
    own = [name for name in candidates if name in da_var.dims]
    candidates = own or candidates
    if len(candidates) > 1:
        raise ValueError(
            f"Variable '{da_var.name}' has several height coordinates "
            f"{candidates}; give each variable exactly one"
        )

    name = candidates[0]
    da_height = da_var[name]
    if da_height.size != 1:
        raise NotImplementedError(
            f"Variable '{da_var.name}' is on several heights ({name}); only "
            "single-level parameters are supported"
        )
    units = da_height.attrs.get("units")
    if units not in _HEIGHT_UNITS:
        raise ValueError(
            f"Height coordinate '{name}' of variable '{da_var.name}' needs "
            f"units in {sorted(_HEIGHT_UNITS)}, got {units!r}.\n{_HEIGHT_EXAMPLE}"
        )
    height = float(np.asarray(da_height.values).item()) * _HEIGHT_UNITS[units]

    if name in da_var.dims:
        da_var = da_var.squeeze(name, drop=True)
    else:
        da_var = da_var.drop_vars(name)
    return da_var, height


def _to_harp_units(da_var: xr.DataArray, harp_param: HarpParam) -> xr.DataArray:
    """Convert a variable to the units HARP uses for a parameter.

    Parameters
    ----------
    da_var : xr.DataArray
        Data variable with a ``units`` attribute.
    harp_param : HarpParam
        Target HARP parameter. If its units are empty (a HARP parameter this
        package has no conversion for), the variable's own units are kept.

    Returns
    -------
    xr.DataArray
        The converted variable with ``attrs["units"]`` set to the HARP units.

    Raises
    ------
    ValueError
        If the variable has no ``units`` or they cannot be converted.
    """
    units = da_var.attrs.get("units")
    if units is None:
        raise ValueError(
            f"Variable '{da_var.name}' has no units attribute; cannot convert "
            f"it to HARP units for {harp_param.name}"
        )
    target = harp_param.units or units
    if target == units:
        scale, offset = 1.0, 0.0
    else:
        conversions = _UNIT_CONVERSIONS.get(target, {})
        if units not in conversions:
            raise ValueError(
                f"Cannot convert variable '{da_var.name}' from {units!r} to "
                f"{target!r} for HARP parameter {harp_param.name}; supported "
                f"units: {sorted(conversions)}"
            )
        scale, offset = conversions[units]

    da_out = da_var * scale + offset if (scale, offset) != (1.0, 0.0) else da_var
    da_out = da_out.copy()
    da_out.attrs = {**da_var.attrs, "units": target}
    da_out.name = da_var.name
    return da_out
