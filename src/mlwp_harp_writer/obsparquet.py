"""Write point observations as a harpIO ``obsparquet`` dataset.

The layout matches what ``harpIO::read_obs(output_format_opts =
obsparquet_opts(...))`` writes, so the data can be read with::

    read_point_obs(dttm, parameter, obs_path = path, file_format = "obsparquet")

Directory layout::

    {path}/valid_year=Y/valid_month=M/valid_day=D/synop-*.parquet
    {path}/schema/synop-schema.parquet
    {path}/params/params.parquet
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from loguru import logger
from mlwp_data_specs.api import TIME_TRAIT_ATTR

from ._arrow import date_partitions, write_hive_dataset
from ._stations import station_elevation, station_ids
from ._time import to_unix_seconds
from .params import HarpParam, resolve_params
from .traits import require_traits

PARTITIONING = ["valid_year", "valid_month", "valid_day"]
_OBS_DIMS = ("valid_time", "point_index")
# harpIO ignores files starting with "temp" when reading SYNOP data (and vice
# versa), so file and schema names use the table name as prefix.
_TABLE_NAME = "synop"


def write_obsparquet(
    ds_obs: xr.Dataset,
    path: str | PathLike,
    *,
    params: dict[str, str | HarpParam] | None = None,
) -> list[Path]:
    """Write a point observation dataset as a harpIO obsparquet dataset.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Point observations (``point`` space trait), either on
        ``(valid_time, point_index)`` with the ``observation`` time trait, or
        as returned by ``mxalign.align_time`` with an observation reference:
        on ``(reference_time, lead_time, point_index)`` with a 2-D
        ``valid_time`` coordinate and the ``forecast`` time trait.
    path : str or PathLike
        Root of the obsparquet dataset (the ``obs_path`` passed to
        ``read_point_obs``).
    params : dict, optional
        Variable name to HARP parameter overrides, see
        :func:`~mlwp_harp_writer.params.resolve_params`.

    Returns
    -------
    list of Path
        The parquet data files written.

    Raises
    ------
    ValueError
        If the dataset has unsupported traits or no variable can be mapped
        to a HARP parameter.
    """
    ds_obs = obs_on_valid_time(ds_obs)
    params_resolved = resolve_params(ds_obs, params)
    if not params_resolved:
        raise ValueError("No observation variables could be mapped to HARP parameters")

    df = observation_dataframe(ds_obs, params_resolved)
    if df.empty:
        logger.warning("No non-missing observations to write")
        return []

    path = Path(path)
    table = pa.Table.from_pandas(df, preserve_index=False)
    _write_schema(table.schema, path / "schema" / f"{_TABLE_NAME}-schema.parquet")
    _write_params_table(params_resolved.values(), path / "params" / "params.parquet")

    first = pd.Timestamp(int(df["valid_dttm"].min()), unit="s")
    last = pd.Timestamp(int(df["valid_dttm"].max()), unit="s")
    logger.info(f"Writing observations {first} - {last} to {path}")
    return write_hive_dataset(
        table,
        path,
        PARTITIONING,
        basename_template=(
            f"{_TABLE_NAME}-{first:%Y%m%d%H%M}-{last:%Y%m%d%H%M}-{{i}}.parquet"
        ),
    )


def obs_on_valid_time(ds_obs: xr.Dataset) -> xr.Dataset:
    """Return point observations on a 1-D ``valid_time`` dimension.

    ``mxalign.align_time`` with an observation reference returns the
    observations broadcast onto the forecasts' ``(reference_time, lead_time)``
    grid, so the same valid time appears several times (with identical
    values). This flattens that form back to unique valid times.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Point observations on ``valid_time`` (``observation`` time trait) or on
        ``(reference_time, lead_time)`` with a 2-D ``valid_time`` coordinate
        (``forecast`` time trait, as returned by mxalign).

    Returns
    -------
    xr.Dataset
        Observations on ``(valid_time, point_index)`` with the
        ``observation`` time trait, sorted by ``valid_time``.

    Raises
    ------
    ValueError
        If the dataset is not a point dataset in either form.
    """
    traits = require_traits(
        ds_obs,
        context="write_obsparquet",
        time={"observation", "forecast"},
        space="point",
    )
    if traits["time"] == "observation":
        return ds_obs

    if "valid_time" not in ds_obs.coords or ds_obs["valid_time"].dims != (
        "reference_time",
        "lead_time",
    ):
        raise ValueError(
            "write_obsparquet got a forecast-shaped dataset without a 2-D "
            "valid_time coordinate; pass observations on valid_time or the "
            "observations returned by mxalign.align_time"
        )
    ds_flat = (
        ds_obs.stack(time=["reference_time", "lead_time"])
        .reset_index("time")
        .drop_vars(["reference_time", "lead_time"])
        .swap_dims({"time": "valid_time"})
        .drop_duplicates("valid_time")
        .sortby("valid_time")
    )
    ds_flat.attrs[TIME_TRAIT_ATTR] = "observation"
    return ds_flat


def observation_dataframe(
    ds_obs: xr.Dataset, params: dict[str, HarpParam]
) -> pd.DataFrame:
    """Convert observations to a wide HARP obsparquet table.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Point observations with dims ``(valid_time, point_index)``.
    params : dict[str, HarpParam]
        Variables to write and their HARP parameters.

    Returns
    -------
    pd.DataFrame
        One row per valid time and station with ``valid_dttm``, ``SID``,
        ``lat``, ``lon``, ``elev`` (from an ``elevation`` or ``altitude`` coordinate, if any),
        one column per HARP parameter and the partition columns. Rows where
        all parameters are missing are dropped.

    Raises
    ------
    NotImplementedError
        If a variable has dims other than ``(valid_time, point_index)``.
    """
    n_time, n_point = (ds_obs.sizes[d] for d in _OBS_DIMS)
    time_idx, point_idx = (
        idx.ravel()
        for idx in np.meshgrid(np.arange(n_time), np.arange(n_point), indexing="ij")
    )
    columns = {
        "valid_dttm": to_unix_seconds(ds_obs["valid_time"].values)[time_idx],
        "SID": station_ids(ds_obs)[point_idx],
        "lat": ds_obs["latitude"].values[point_idx],
        "lon": ds_obs["longitude"].values[point_idx],
    }
    elevation = station_elevation(ds_obs)
    if elevation is not None:
        columns["elev"] = elevation[point_idx]

    for var, harp_param in params.items():
        da_obs = ds_obs[var]
        if set(da_obs.dims) != set(_OBS_DIMS):
            raise NotImplementedError(
                f"Variable '{var}' has dims {da_obs.dims}; only {_OBS_DIMS} is supported"
            )
        columns[harp_param.name] = da_obs.transpose(*_OBS_DIMS).values.ravel()

    df = pd.DataFrame(columns)
    param_cols = [p.name for p in params.values()]
    df = df[df[param_cols].notna().any(axis=1)].reset_index(drop=True)

    partitions = date_partitions(pd.to_datetime(df["valid_dttm"], unit="s"), "valid")
    for name in PARTITIONING:
        df[name] = partitions[name].to_numpy()
    return df


def _write_schema(schema: pa.Schema, schema_file: Path) -> None:
    """Write (or extend) the dataset schema file harpIO reads the data with.

    Parameters
    ----------
    schema : pa.Schema
        Schema of the data being written.
    schema_file : Path
        Path of the schema parquet file. If it exists its schema is unified
        with ``schema`` so columns from earlier writes are kept.
    """
    schema = schema.remove_metadata()
    if schema_file.exists():
        schema = pa.unify_schemas([pq.read_schema(schema_file), schema])
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(schema.empty_table(), schema_file)


def _write_params_table(params, params_file: Path) -> None:
    """Write (or extend) the parameter units table.

    Parameters
    ----------
    params : iterable of HarpParam
        Parameters being written.
    params_file : Path
        Path of the params parquet file. Entries already in the file are kept.
    """
    new = pd.DataFrame(
        {"parameter": [p.name for p in params], "units": [p.units for p in params]}
    )
    if params_file.exists():
        saved = pd.read_parquet(params_file)
        new = pd.concat([saved, new[~new["parameter"].isin(saved["parameter"])]])
    params_file.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(params_file, index=False)
