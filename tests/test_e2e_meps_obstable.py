"""End-to-end test on real data: MEPS forecasts + harpData OBSTABLE observations.

Runs the README workflow: load with mlwp-data-loaders (using the test loaders
in ``tests/loaders``), align with mxalign, and write HARP parquet datasets.

- Forecasts: MET Norway MEPS, cycles 2019-02-17 00 and 12 UTC, from the
  THREDDS archive over OPeNDAP (a small subset).
- Observations: harpData's ``OBSTABLE_2019.sqlite`` (2019-02-17 to
  2019-02-20), the observations of the harp tutorial period that MEPS covers.

Needs network access (marker ``network``); skipped if the data can't be
downloaded.
"""

from __future__ import annotations

from pathlib import Path

import mxalign as mx
import numpy as np
import pooch
import pyarrow.parquet as pq
import pytest
from mlwp_data_loaders import load_and_validate_dataset

from mlwp_harp_writer import write_harp_parquets

from .loaders.meps_thredds import meps_url
from .test_writers import read_hive

pytestmark = pytest.mark.network

LOADERS = Path(__file__).parent / "loaders"
OBSTABLE_URL = (
    "https://raw.githubusercontent.com/harphub/harpData/master/inst/OBSTABLE/"
    "OBSTABLE_2019.sqlite"
)
OBSTABLE_HASH = "bdab991c287a41871488456d1a9d697942aa3a612800a88264defa312a9d637b"
CYCLES = ["20190217T00", "20190217T12"]
LEAD_HOURS = (0, 6, 12, 24)
FC_PARTITIONS = ["fcst_hour", "fcst_year", "fcst_month", "fcst_day"]
OBS_PARTITIONS = ["valid_year", "valid_month", "valid_day"]
HARP_UNITS = {
    "T2m": "K",
    "RH2m": "percent",
    "Pmsl": "hPa",
    "Ps": "hPa",
    "CCtot": "oktas",
}


@pytest.fixture(scope="module")
def harp_output(tmp_path_factory) -> Path:
    """Load, align and write the MEPS forecasts and OBSTABLE observations.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for the output directory.

    Returns
    -------
    Path
        Root of the written HARP parquet datasets.
    """
    try:
        obstable_path = pooch.retrieve(url=OBSTABLE_URL, known_hash=OBSTABLE_HASH)
        ds_fcst = load_and_validate_dataset(
            [meps_url(cycle) for cycle in CYCLES],
            loader=str(LOADERS / "meps_thredds.py"),
            lead_hours=LEAD_HOURS,
        )
    except OSError as exc:
        pytest.skip(f"Could not download sample data: {exc}")
    ds_obs = load_and_validate_dataset(
        obstable_path, loader=str(LOADERS / "harp_obstable_cf.py")
    )

    ds_fcst_points = ds_fcst.mx.align_space_with(ds_obs, method="xarray")
    ds_fcst_aligned = mx.align_time(ds_fcst_points, reference=ds_obs)

    path = tmp_path_factory.mktemp("harp")
    write_harp_parquets(ds_fcst_aligned, ds_obs, path, fcst_model="MEPS")
    return path


def read_forecast(harp_output: Path, parameter: str):
    """Read one written forecast parameter.

    Parameters
    ----------
    harp_output : Path
        Root of the written HARP parquet datasets.
    parameter : str
        HARP parameter name.

    Returns
    -------
    pd.DataFrame
        The fcparquet rows of the parameter.
    """
    return read_hive(
        harp_output / "FCPARQUET" / "MEPS" / parameter, FC_PARTITIONS
    ).to_pandas()


def read_observations(harp_output: Path):
    """Read the written observations with their schema, as harpIO does.

    Parameters
    ----------
    harp_output : Path
        Root of the written HARP parquet datasets.

    Returns
    -------
    pd.DataFrame
        The obsparquet rows.
    """
    schema = pq.read_schema(
        harp_output / "OBSPARQUET" / "schema" / "synop-schema.parquet"
    )
    return read_hive(
        harp_output / "OBSPARQUET", OBS_PARTITIONS, schema=schema
    ).to_pandas()


def test_parameters_and_units(harp_output):
    """Every MEPS variable is written as its HARP parameter in HARP units."""
    written = sorted(p.name for p in (harp_output / "FCPARQUET" / "MEPS").iterdir())
    assert written == sorted(HARP_UNITS)
    for parameter, units in HARP_UNITS.items():
        df_fcst = read_forecast(harp_output, parameter)
        assert df_fcst["units"].unique().tolist() == [units]
        assert sorted(df_fcst["lead_time"].unique()) == [h * 3600 for h in LEAD_HOURS]

    df_params = pq.read_table(harp_output / "OBSPARQUET" / "params" / "params.parquet")
    assert dict(zip(*df_params.to_pydict().values())) == HARP_UNITS


def test_stations_in_domain(harp_output):
    """Forecasts are written for the stations inside the MEPS domain only."""
    df_fcst = read_forecast(harp_output, "T2m")
    assert df_fcst["SID"].nunique() > 100
    assert df_fcst["lat"].between(50, 75).all()
    assert df_fcst["lon"].between(-15, 50).all()


@pytest.mark.parametrize(
    ("parameter", "max_rmse"),
    [("T2m", 5.0), ("Pmsl", 5.0), ("RH2m", 25.0)],
)
def test_forecasts_match_observations(harp_output, parameter, max_rmse):
    """Forecasts and observations agree to within a loose bound.

    This checks that stations, valid times and units line up, not model skill.
    """
    df_fcst = read_forecast(harp_output, parameter)
    df_obs = read_observations(harp_output)
    df_joined = df_fcst.merge(
        df_obs[["SID", "valid_dttm", parameter]], on=["SID", "valid_dttm"]
    ).dropna(subset=["MEPS_det", parameter])
    assert len(df_joined) > 100
    rmse = float(np.sqrt(((df_joined["MEPS_det"] - df_joined[parameter]) ** 2).mean()))
    assert rmse < max_rmse, f"{parameter} RMSE {rmse:.2f} {HARP_UNITS[parameter]}"


def test_cloud_cover_in_oktas(harp_output):
    """Total cloud cover (MEPS: percent) is written in oktas."""
    values = read_forecast(harp_output, "CCtot")["MEPS_det"].dropna()
    # MEPS values are float32, so 100 % becomes 8.0000005 oktas
    assert values.between(0, 8 + 1e-5).all()
    assert values.max() > 1
