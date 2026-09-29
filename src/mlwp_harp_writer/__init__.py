"""Write mxalign-aligned, mlwp-data-specs conforming datasets to HARP parquet datasets."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from pathlib import Path

import xarray as xr

from .fcparquet import write_fcparquet
from .obsparquet import write_obsparquet
from .params import DEFAULT_PARAMS, HarpParam, resolve_params
from .traits import validate

try:
    __version__ = version("mlwp-harp-writer")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

FCPARQUET_DIR = "FCPARQUET"
OBSPARQUET_DIR = "OBSPARQUET"


def write_harp_parquets(
    forecasts: xr.Dataset | dict[str, xr.Dataset],
    ds_obs: xr.Dataset,
    path: str | PathLike,
    *,
    fcst_model: str | None = None,
    params: dict[str, str | HarpParam] | None = None,
) -> dict[str, list[Path]]:
    """Write mxalign-aligned forecasts and observations as HARP parquet datasets.

    Forecasts are written to ``{path}/FCPARQUET`` and observations to
    ``{path}/OBSPARQUET``. Forecasts without station metadata (``code``, as
    after mxalign interpolation) get it from ``ds_obs``.

    Parameters
    ----------
    forecasts : xr.Dataset or dict[str, xr.Dataset]
        A single point forecast (named with ``fcst_model``), or point
        forecasts keyed by HARP model name, e.g. the dict returned by
        ``mxalign.align_space``/``mxalign.align_time``.
    ds_obs : xr.Dataset
        Point observations at the same stations, either on ``valid_time`` or
        as returned by ``mxalign.align_time`` (see :func:`write_obsparquet`).
    path : str or PathLike
        Root directory to write to.
    fcst_model : str, optional
        HARP model name of ``forecasts`` when it is a single Dataset.
        Required in that case, and not allowed with a dict.
    params : dict, optional
        Variable name to HARP parameter overrides, see :func:`resolve_params`.

    Returns
    -------
    dict[str, list of Path]
        Files written per forecast model name, plus ``"observations"``.

    Raises
    ------
    ValueError
        If ``fcst_model`` is missing for a single Dataset or given with a
        dict.
    """
    if isinstance(forecasts, xr.Dataset):
        if fcst_model is None:
            raise ValueError("fcst_model is required when writing a single forecast")
        forecasts = {fcst_model: forecasts}
    elif fcst_model is not None:
        raise ValueError(
            "fcst_model cannot be combined with a dict of forecasts; the dict "
            "keys are used as model names"
        )

    path = Path(path)
    written = {
        name: write_fcparquet(
            ds_fcst, path / FCPARQUET_DIR, name, params=params, ds_stations=ds_obs
        )
        for name, ds_fcst in forecasts.items()
    }
    written["observations"] = write_obsparquet(
        ds_obs, path / OBSPARQUET_DIR, params=params
    )
    return written


__all__ = [
    "DEFAULT_PARAMS",
    "HarpParam",
    "resolve_params",
    "validate",
    "write_fcparquet",
    "write_harp_parquets",
    "write_obsparquet",
    "__version__",
]
