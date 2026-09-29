"""Write point forecasts as a harpIO ``fcparquet`` dataset.

The layout matches what ``harpIO::read_forecast(output_format_opts =
fcparquet_opts(...))`` writes, so the data can be read with::

    read_point_forecast(
        dttm, fcst_model, parameter, file_path = path,
        file_format = "fcparquet", file_template = "{fcst_model}/{parameter}"
    )

Directory layout::

    {path}/{fcst_model}/{parameter}/fcst_hour=H/fcst_year=Y/fcst_month=M/fcst_day=D/*.parquet
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from loguru import logger

from ._arrow import date_partitions, write_hive_dataset
from ._stations import station_ids, with_station_coords
from ._time import lead_time_seconds, to_unix_seconds
from .params import HarpParam, harp_variables
from .traits import require_traits

PARTITIONING = ["fcst_hour", "fcst_year", "fcst_month", "fcst_day"]
_FORECAST_DIMS = ("reference_time", "lead_time", "point_index")


def write_fcparquet(
    ds_fcst: xr.Dataset,
    path: str | PathLike,
    fcst_model: str,
    *,
    params: dict[str, str | HarpParam] | None = None,
    model_elevation: str | None = None,
    ds_stations: xr.Dataset | None = None,
) -> list[Path]:
    """Write a point forecast dataset as a harpIO fcparquet dataset.

    Parameters
    ----------
    ds_fcst : xr.Dataset
        Point forecast with dims ``(reference_time, lead_time, point_index)``
        and, for ensembles, ``member``, e.g. as returned by
        ``mxalign.align_space``/``mxalign.align_time``. Needs the ``forecast``
        time and ``point`` space traits. Variables are mapped to HARP
        parameters by CF ``standard_name`` and height coordinate, and
        converted to HARP units, see
        :func:`~mlwp_harp_writer.params.harp_variables`.
    path : str or PathLike
        Root of the fcparquet dataset (the ``file_path`` passed to
        ``read_point_forecast``).
    fcst_model : str
        HARP model name, used in the directory layout and member column names.
    params : dict, optional
        Explicit variable name to HARP parameter mappings, see
        :func:`~mlwp_harp_writer.params.harp_variables`.
    model_elevation : str, optional
        Name of a variable or coordinate along ``point_index`` holding the
        model orography at the stations, written as ``model_elevation``.
    ds_stations : xr.Dataset, optional
        Point dataset (typically the observations) to take the station
        metadata from when ``ds_fcst`` has no ``code`` coordinate, which is
        the case after mxalign interpolation. Without it, ``point_index`` is
        used as ``SID``.

    Returns
    -------
    list of Path
        The parquet files written.

    Raises
    ------
    ValueError
        If the dataset has unsupported traits, or is not located at the
        stations of ``ds_stations``.
    NotImplementedError
        If a variable has dims other than the forecast/point/member dims
        (e.g. vertical levels).
    """
    traits = require_traits(
        ds_fcst,
        context="write_fcparquet",
        time="forecast",
        space="point",
        uncertainty={"deterministic", "ensemble"},
    )
    is_ensemble = traits["uncertainty"] == "ensemble"
    ds_fcst = with_station_coords(ds_fcst, ds_stations)
    elevation = None
    ds_values = ds_fcst
    if model_elevation is not None:
        elevation = ds_fcst[model_elevation].values
        if model_elevation in ds_fcst.data_vars:
            ds_values = ds_fcst.drop_vars(model_elevation)
    written = []
    for harp_name, da_param in harp_variables(ds_values, params).items():
        harp_param = HarpParam(harp_name, da_param.attrs["units"])
        df = forecast_dataframe(
            da_param,
            fcst_model=fcst_model,
            harp_param=harp_param,
            is_ensemble=is_ensemble,
            model_elevation=elevation,
        )
        if df.empty:
            logger.warning(f"No non-missing forecasts for {harp_name}, skipping")
            continue
        table = pa.Table.from_pandas(df, preserve_index=False)
        first = pd.Timestamp(int(df["fcst_dttm"].min()), unit="s")
        last = pd.Timestamp(int(df["fcst_dttm"].max()), unit="s")
        base_dir = Path(path) / fcst_model / harp_param.name
        logger.info(f"Writing {harp_param.name} for {fcst_model} to {base_dir}")
        written += write_hive_dataset(
            table,
            base_dir,
            PARTITIONING,
            basename_template=(
                f"{fcst_model}-{first:%Y%m%d%H%M}-{last:%Y%m%d%H%M}-{{i}}.parquet"
            ),
        )
    return written


def forecast_dataframe(
    da_fcst: xr.DataArray,
    *,
    fcst_model: str,
    harp_param: HarpParam,
    is_ensemble: bool,
    model_elevation: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert one forecast variable to a HARP fcparquet table.

    Parameters
    ----------
    da_fcst : xr.DataArray
        Point forecast variable with dims ``(reference_time, lead_time,
        point_index)`` and optionally ``member``.
    fcst_model : str
        HARP model name used for the member column names.
    harp_param : HarpParam
        HARP parameter name and units.
    is_ensemble : bool
        Write one ``{fcst_model}_mbrNNN`` column per member rather than a
        single ``{fcst_model}_det`` column.
    model_elevation : np.ndarray, optional
        Model orography per ``point_index``.

    Returns
    -------
    pd.DataFrame
        One row per forecast time, lead time and station, with the partition
        columns included. Rows where all members are missing are dropped.

    Raises
    ------
    NotImplementedError
        If ``da_fcst`` has unsupported dims.
    """
    member_dims = ("member",) if is_ensemble else ()
    extra_dims = set(da_fcst.dims) - set(_FORECAST_DIMS) - set(member_dims)
    if extra_dims:
        raise NotImplementedError(
            f"Variable '{da_fcst.name}' has unsupported dims {sorted(extra_dims)}; "
            "only surface (single level) parameters are supported"
        )
    da_fcst = da_fcst.transpose(*_FORECAST_DIMS, *member_dims)

    n_ref, n_lead, n_point = (da_fcst.sizes[d] for d in _FORECAST_DIMS)
    ref_idx, lead_idx, point_idx = (
        idx.ravel()
        for idx in np.meshgrid(
            np.arange(n_ref), np.arange(n_lead), np.arange(n_point), indexing="ij"
        )
    )

    fcst_dttm = to_unix_seconds(da_fcst["reference_time"].values)[ref_idx]
    lead_time = lead_time_seconds(da_fcst["lead_time"])[lead_idx]
    columns = {
        "fcst_dttm": fcst_dttm,
        "valid_dttm": fcst_dttm + lead_time,
        "lead_time": lead_time,
        "SID": station_ids(da_fcst)[point_idx],
        "lat": da_fcst["latitude"].values[point_idx],
        "lon": da_fcst["longitude"].values[point_idx],
    }
    if model_elevation is not None:
        columns["model_elevation"] = np.asarray(model_elevation)[point_idx]
    columns["parameter"] = harp_param.name
    columns["units"] = harp_param.units

    values = np.asarray(da_fcst.values).reshape(n_ref * n_lead * n_point, -1)
    member_cols = member_column_names(da_fcst, fcst_model, is_ensemble=is_ensemble)
    df = pd.DataFrame(columns)
    df[member_cols] = values
    df = df[~np.isnan(values).all(axis=1)].reset_index(drop=True)

    partitions = date_partitions(pd.to_datetime(df["fcst_dttm"], unit="s"), "fcst")
    for name, col in partitions.items():
        df[name] = col.to_numpy()
    return df


def member_column_names(
    da_fcst: xr.DataArray, fcst_model: str, *, is_ensemble: bool
) -> list[str]:
    """Return the HARP forecast value column names.

    Parameters
    ----------
    da_fcst : xr.DataArray
        Forecast variable (with a ``member`` coordinate for ensembles).
    fcst_model : str
        HARP model name.
    is_ensemble : bool
        Whether ``da_fcst`` is an ensemble forecast.

    Returns
    -------
    list of str
        ``["{fcst_model}_det"]`` for deterministic forecasts, otherwise
        ``"{fcst_model}_mbrNNN"`` per member. Integer-like member labels are
        used as the member number, otherwise the position along ``member``.
    """
    if not is_ensemble:
        return [f"{fcst_model}_det"]
    labels = pd.to_numeric(pd.Series(da_fcst["member"].values), errors="coerce")
    if labels.notna().all() and (labels % 1 == 0).all():
        numbers = labels.astype(int).tolist()
    else:
        numbers = list(range(da_fcst.sizes["member"]))
    return [f"{fcst_model}_mbr{n:03d}" for n in numbers]
