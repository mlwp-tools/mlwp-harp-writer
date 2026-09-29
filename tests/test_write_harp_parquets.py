"""End-to-end tests: align with mxalign directly, then write HARP datasets."""

from __future__ import annotations

import mxalign as mx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mlwp_harp_writer import write_fcparquet, write_harp_parquets, write_obsparquet
from mlwp_harp_writer.obsparquet import obs_on_valid_time

from .conftest import (
    LEAD_HOURS,
    MXALIGN_DELAUNAY_METADATA_BUG,
    MXALIGN_OVERLAP_BUG,
    REFERENCE_TIMES,
    STATION_CODES,
    STATION_LATS,
    STATION_LONS,
    linear_field,
)
from .test_writers import read_hive

FC_PARTITIONS = ["fcst_hour", "fcst_year", "fcst_month", "fcst_day"]
OBS_PARTITIONS = ["valid_year", "valid_month", "valid_day"]


def align_with_mxalign(
    forecasts: dict[str, xr.Dataset], ds_obs: xr.Dataset, method: str
) -> dict[str, xr.Dataset]:
    """Align forecasts to the observations with mxalign, as in the README.

    Parameters
    ----------
    forecasts : dict[str, xr.Dataset]
        Grid forecasts keyed by model name.
    ds_obs : xr.Dataset
        Point observations used as the spatial and temporal reference.
    method : str
        mxalign interpolation method.

    Returns
    -------
    dict[str, xr.Dataset]
        Point forecasts on the common ``(reference_time, lead_time)`` grid,
        keyed like ``forecasts``.
    """
    forecasts_points: dict[str, xr.Dataset] = mx.align_space(
        forecasts, reference=ds_obs, method=method
    )
    aligned: dict[str, xr.Dataset] = mx.align_time(forecasts_points, reference=ds_obs)
    return aligned


def select_leads(ds_fcst: xr.Dataset, lead_hours: list[int]) -> xr.Dataset:
    """Select forecast lead times given in hours.

    Parameters
    ----------
    ds_fcst : xr.Dataset
        Forecast with ``timedelta64`` lead times.
    lead_hours : list of int
        Lead times to keep, in hours.

    Returns
    -------
    xr.Dataset
        The forecast at the selected lead times.
    """
    lead_times = np.array(lead_hours, dtype="timedelta64[h]").astype("timedelta64[ns]")
    return ds_fcst.sel(lead_time=lead_times)


@pytest.mark.parametrize(
    ("fixture", "method"),
    [
        pytest.param("ds_grid_fcst", "xarray", id="xarray"),
        pytest.param(
            "ds_grid_fcst_stacked",
            "delaunay",
            marks=MXALIGN_DELAUNAY_METADATA_BUG,
            id="delaunay",
        ),
    ],
)
def test_mxalign_to_harp(tmp_path, request, ds_obs, fixture, method):
    """mxalign output is written with model names from the dict keys and SIDs
    taken from the observations.

    The 12-hourly cycles with 0-12 h lead times share valid times between
    cycles, which mxalign handles when the obs are passed as ``reference``.
    """
    aligned = align_with_mxalign(
        {"my-model": request.getfixturevalue(fixture)}, ds_obs, method
    )
    write_harp_parquets(aligned, ds_obs, tmp_path)

    df_fcst = read_hive(
        tmp_path / "FCPARQUET" / "my-model" / "T2m", FC_PARTITIONS
    ).to_pandas()
    assert len(df_fcst) == len(REFERENCE_TIMES) * len(LEAD_HOURS) * len(STATION_CODES)
    assert set(df_fcst["SID"]) == set(STATION_CODES)

    # linear fields are reproduced exactly by linear interpolation
    df_fcst = df_fcst.sort_values(["fcst_dttm", "lead_time", "SID"])
    order = np.argsort(STATION_CODES)
    expected = linear_field(
        STATION_LATS[order][None, None, :],
        STATION_LONS[order][None, None, :],
        np.arange(len(REFERENCE_TIMES))[:, None, None],
        LEAD_HOURS[None, :, None],
    )
    np.testing.assert_allclose(
        df_fcst["my-model_det"].to_numpy(), expected.ravel(), rtol=1e-6
    )

    # all observation times are written, covering every forecast valid time
    df_obs = read_hive(tmp_path / "OBSPARQUET", OBS_PARTITIONS).to_pandas()
    assert len(df_obs) == ds_obs.sizes["valid_time"] * len(STATION_CODES)
    assert set(df_fcst["valid_dttm"]) <= set(df_obs["valid_dttm"])
    assert set(df_obs["SID"]) == set(STATION_CODES)


def test_single_forecast_via_mx_accessor(tmp_path, ds_grid_fcst, ds_obs):
    """A single forecast aligned as in the README (``ds.mx`` accessor for
    space, ``mx.align_time`` for time) is written under ``fcst_model``."""
    ds_fcst_points = ds_grid_fcst.mx.align_space_with(ds_obs, method="xarray")
    ds_fcst_aligned = mx.align_time(ds_fcst_points, reference=ds_obs)
    write_harp_parquets(ds_fcst_aligned, ds_obs, tmp_path, fcst_model="my-ai-model")

    df_fcst = read_hive(
        tmp_path / "FCPARQUET" / "my-ai-model" / "T2m", FC_PARTITIONS
    ).to_pandas()
    assert len(df_fcst) == len(REFERENCE_TIMES) * len(LEAD_HOURS) * len(STATION_CODES)
    assert set(df_fcst["SID"]) == set(STATION_CODES)
    assert "my-ai-model_det" in df_fcst.columns


def test_write_harp_fcst_model_argument(tmp_path, ds_point_fcst, ds_obs):
    """``fcst_model`` is required for a single Dataset and rejected for a dict."""
    with pytest.raises(ValueError, match="fcst_model is required"):
        write_harp_parquets(ds_point_fcst, ds_obs, tmp_path)
    with pytest.raises(ValueError, match="cannot be combined"):
        write_harp_parquets({"a": ds_point_fcst}, ds_obs, tmp_path, fcst_model="b")


def test_mxalign_ensemble(tmp_path, ds_grid_ens, ds_obs):
    """Ensemble forecasts aligned by mxalign get one column per member."""
    aligned = align_with_mxalign({"ens": ds_grid_ens}, ds_obs, "xarray")
    write_harp_parquets(aligned, ds_obs, tmp_path)
    df_fcst = read_hive(tmp_path / "FCPARQUET" / "ens" / "T2m", FC_PARTITIONS)
    assert [c for c in df_fcst.column_names if "_mbr" in c] == [
        "ens_mbr000",
        "ens_mbr001",
        "ens_mbr002",
    ]


@pytest.mark.parametrize(
    "lead_hours",
    [
        pytest.param([0, 6], id="no-overlap"),
        pytest.param([0, 6, 12], marks=MXALIGN_OVERLAP_BUG, id="overlap"),
    ],
)
def test_obs_in_align_time_dict(tmp_path, ds_grid_fcst, ds_obs, lead_hours):
    """Obs passed through ``mx.align_time`` in the datasets dict come back
    reshaped to ``(reference_time, lead_time)``; they write the same rows as
    the original obs at the forecast valid times."""
    forecasts: dict[str, xr.Dataset] = mx.align_space(
        {"model": select_leads(ds_grid_fcst, lead_hours)},
        reference=ds_obs,
        method="xarray",
    )
    aligned: dict[str, xr.Dataset] = mx.align_time(
        {**forecasts, "obs": ds_obs}, reference="obs"
    )
    ds_obs_aligned = aligned.pop("obs")

    ds_obs_flat = obs_on_valid_time(ds_obs_aligned)
    assert ds_obs_flat["valid_time"].to_index().is_unique
    ds_obs_subset = ds_obs.sel(valid_time=ds_obs_flat["valid_time"])

    write_obsparquet(ds_obs_aligned, tmp_path / "reshaped")
    write_obsparquet(ds_obs_subset, tmp_path / "original")

    def rows(path) -> pd.DataFrame:
        """Read an obsparquet dataset into a sorted DataFrame.

        Parameters
        ----------
        path : Path
            Dataset root.

        Returns
        -------
        pd.DataFrame
            Rows sorted by valid time and station.
        """
        df = read_hive(path, OBS_PARTITIONS).to_pandas()
        return df.sort_values(["valid_dttm", "SID"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        rows(tmp_path / "reshaped"), rows(tmp_path / "original")
    )


def test_forecast_not_at_obs_stations_raises(tmp_path, ds_point_fcst, ds_obs):
    """Station metadata is only copied when the stations match."""
    ds_moved = ds_point_fcst.drop_vars(["code", "elevation"]).assign_coords(
        latitude=("point_index", ds_point_fcst["latitude"].values + 1)
    )
    with pytest.raises(ValueError, match="not at the same stations"):
        write_fcparquet(ds_moved, tmp_path, "model", ds_stations=ds_obs)


def test_obs_without_valid_time_raises(tmp_path, ds_point_fcst):
    """A forecast-shaped dataset without a 2-D valid_time is not accepted as obs."""
    with pytest.raises(ValueError, match="2-D valid_time"):
        write_obsparquet(ds_point_fcst, tmp_path)
