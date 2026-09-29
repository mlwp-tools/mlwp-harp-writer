"""The README workflow: mlwp-data-loaders -> mxalign alignment -> HARP parquet."""

from __future__ import annotations

import sqlite3

import mxalign as mx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
from mlwp_data_loaders import load_and_validate_dataset

from mlwp_harp_writer import write_harp_parquets

from .conftest import STATION_CODES, STATION_ELEVATIONS, STATION_LATS, STATION_LONS
from .test_writers import read_hive


def write_anemoi_inference_files(directory, n_steps: int) -> list[str]:
    """Write two small anemoi-inference style netCDF output files.

    Parameters
    ----------
    directory : Path
        Directory to write the files to.
    n_steps : int
        Number of 6-hourly output times per forecast (lead times 0, 6, ... h).

    Returns
    -------
    list of str
        Paths of the files, one per forecast cycle (00 and 12 UTC).
    """
    lat_grid, lon_grid = np.meshgrid(
        np.arange(54.0, 58.01, 0.5), np.arange(8.0, 13.01, 0.5), indexing="ij"
    )
    paths = []
    for cycle in ("2026-01-01T00", "2026-01-01T12"):
        times = pd.date_range(cycle, periods=n_steps, freq="6h")
        values = 270.0 + 0.5 * lat_grid.ravel() + 0.2 * lon_grid.ravel()
        ds_file = xr.Dataset(
            {"2t": (["time", "values"], np.tile(values, (len(times), 1)))},
            coords={"time": times.values},
        )
        ds_file["latitude"] = ("values", lat_grid.ravel())
        ds_file["longitude"] = ("values", lon_grid.ravel())
        path = directory / f"anemoi-inference-{cycle.replace(':', '')}.nc"
        ds_file.to_netcdf(path, engine="h5netcdf")
        paths.append(str(path))
    return paths


def write_obstable(path) -> None:
    """Write a small HARP OBSTABLE SQLite file with hourly T2m observations.

    Parameters
    ----------
    path : Path
        Path of the SQLite file.
    """
    valid_times = pd.date_range("2026-01-01T00", "2026-01-02T00", freq="1h")
    rows = pd.DataFrame(
        [
            {
                "SID": int(sid),
                "lat": lat,
                "lon": lon,
                "elev": elev,
                "validdate": int(vt.timestamp()),
                "T2m": 275.0 + i * 0.1,
            }
            for i, vt in enumerate(valid_times)
            for sid, lat, lon, elev in zip(
                STATION_CODES, STATION_LATS, STATION_LONS, STATION_ELEVATIONS
            )
        ]
    )
    with sqlite3.connect(path) as conn:
        rows.to_sql("SYNOP", conn, index=False)


def test_readme_workflow(tmp_path):
    """Load with mlwp-data-loaders, align with mxalign and write HARP parquet.

    Two 12-hourly cycles with 0, 6 and 12 h lead times, so 00 UTC +12 h and
    12 UTC +0 h share a valid time.
    """
    n_steps = 3
    ds_fcst = load_and_validate_dataset(
        write_anemoi_inference_files(tmp_path, n_steps),
        loader="mlwp_data_loaders.loaders.anemoi.anemoi_inference",
    )
    write_obstable(tmp_path / "OBSTABLE_2026.sqlite")
    ds_obs = load_and_validate_dataset(
        str(tmp_path / "OBSTABLE_2026.sqlite"),
        loader="mlwp_data_loaders.loaders.harp.obstable",
    )

    ds_fcst_points = ds_fcst.mx.align_space_with(ds_obs, method="delaunay")
    ds_fcst_aligned = mx.align_time(ds_fcst_points, reference=ds_obs)
    write_harp_parquets(
        ds_fcst_aligned, ds_obs, tmp_path / "harp", fcst_model="my-ai-model"
    )

    df_fcst = read_hive(
        tmp_path / "harp" / "FCPARQUET" / "my-ai-model" / "T2m",
        ["fcst_hour", "fcst_year", "fcst_month", "fcst_day"],
    ).to_pandas()
    assert len(df_fcst) == 2 * n_steps * len(STATION_CODES)
    assert set(df_fcst["SID"]) == set(STATION_CODES)
    expected = 270.0 + 0.5 * STATION_LATS + 0.2 * STATION_LONS
    df_first = df_fcst[df_fcst["lead_time"] == 0].drop_duplicates("SID")
    np.testing.assert_allclose(
        df_first.sort_values("SID")["my-ai-model_det"],
        expected[np.argsort(STATION_CODES)],
        rtol=1e-6,
    )

    obs_path = tmp_path / "harp" / "OBSPARQUET"
    df_obs = read_hive(obs_path, ["valid_year", "valid_month", "valid_day"]).to_pandas()
    assert {"SID", "elev", "T2m"} <= set(df_obs.columns)
    assert set(df_fcst["valid_dttm"]) <= set(df_obs["valid_dttm"])
    params = pq.read_table(obs_path / "params" / "params.parquet").to_pandas()
    assert params["parameter"].tolist() == ["T2m"]
