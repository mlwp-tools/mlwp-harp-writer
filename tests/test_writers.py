"""Tests for the fcparquet and obsparquet writers."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest

from mlwp_harp_writer import write_fcparquet, write_obsparquet

from .conftest import LEAD_HOURS, REFERENCE_TIMES, STATION_CODES, make_observations


def read_hive(path, partitioning):
    """Read a hive-partitioned parquet dataset like harpIO does.

    Parameters
    ----------
    path : Path
        Dataset root.
    partitioning : list of str
        Partition columns; they are read as int32, as harpIO expects.

    Returns
    -------
    pa.Table
        The dataset contents including the partition columns.
    """
    schema = pa.schema([(p, pa.int32()) for p in partitioning])
    dataset = pads.dataset(
        path,
        format="parquet",
        partitioning=pads.partitioning(schema, flavor="hive"),
        exclude_invalid_files=True,
        ignore_prefixes=["schema", "params"],
    )
    return dataset.to_table()


def test_fcparquet_deterministic_layout_and_schema(tmp_path, ds_point_fcst):
    """A deterministic forecast is written in harpIO's fcparquet layout."""
    files = write_fcparquet(ds_point_fcst, tmp_path, "mymodel")

    base = tmp_path / "mymodel" / "T2m"
    assert {f.parent.relative_to(base).as_posix() for f in files} == {
        "fcst_hour=0/fcst_year=2026/fcst_month=1/fcst_day=1",
        "fcst_hour=12/fcst_year=2026/fcst_month=1/fcst_day=1",
    }
    # harpIO opens the fcst_hour=<h> sub directory first to infer the schema
    file_schema = pq.read_schema(files[0])
    assert file_schema.field("fcst_dttm").type == pa.int64()
    assert file_schema.field("valid_dttm").type == pa.int64()
    assert file_schema.field("lead_time").type == pa.int64()
    assert file_schema.field("SID").type == pa.int64()
    assert "fcst_hour" not in file_schema.names

    df = read_hive(
        base, ["fcst_hour", "fcst_year", "fcst_month", "fcst_day"]
    ).to_pandas()
    n_rows = len(REFERENCE_TIMES) * len(LEAD_HOURS) * len(STATION_CODES)
    assert len(df) == n_rows
    assert {"mymodel_det", "parameter", "units", "lat", "lon"} <= set(df.columns)
    assert (df["valid_dttm"] == df["fcst_dttm"] + df["lead_time"]).all()
    assert sorted(df["lead_time"].unique()) == [h * 3600 for h in LEAD_HOURS]
    assert set(df["SID"]) == set(STATION_CODES)
    assert (df["parameter"] == "T2m").all() and (df["units"] == "K").all()


def test_fcparquet_ensemble_member_columns(tmp_path, ds_point_ens):
    """Ensemble members become ``{model}_mbrNNN`` columns."""
    write_fcparquet(ds_point_ens, tmp_path, "ens")
    df = read_hive(
        tmp_path / "ens" / "T2m", ["fcst_hour", "fcst_year", "fcst_month", "fcst_day"]
    ).to_pandas()
    assert [c for c in df.columns if "_mbr" in c] == [
        "ens_mbr000",
        "ens_mbr001",
        "ens_mbr002",
    ]
    np.testing.assert_allclose(df["ens_mbr001"] - df["ens_mbr000"], 0.1)


def test_fcparquet_drops_missing_and_model_elevation(tmp_path, ds_point_fcst):
    """All-NaN rows are dropped and model elevation is written if given."""
    ds_fcst = ds_point_fcst.copy(deep=True)
    ds_fcst["2t"][0, 0, 0] = np.nan
    ds_fcst["orog"] = ("point_index", [1.0, 2.0, 3.0, 4.0])
    write_fcparquet(ds_fcst, tmp_path, "m", model_elevation="orog")
    df = read_hive(
        tmp_path / "m" / "T2m", ["fcst_hour", "fcst_year", "fcst_month", "fcst_day"]
    ).to_pandas()
    assert len(df) == len(REFERENCE_TIMES) * len(LEAD_HOURS) * len(STATION_CODES) - 1
    assert set(df["model_elevation"]) == {1.0, 2.0, 3.0, 4.0}
    assert not (tmp_path / "m" / "orog").exists()


def test_fcparquet_rejects_grid_and_levels(tmp_path, ds_grid_fcst, ds_point_fcst):
    """Grid forecasts and extra (vertical) dims are not supported."""
    with pytest.raises(ValueError, match="space trait"):
        write_fcparquet(ds_grid_fcst, tmp_path, "m")
    ds_levels = ds_point_fcst.expand_dims(pressure=[850, 500])
    with pytest.raises(NotImplementedError, match="unsupported dims"):
        write_fcparquet(ds_levels, tmp_path, "m")


def test_obsparquet_layout_schema_and_params(tmp_path, ds_obs):
    """Observations are written in harpIO's obsparquet layout."""
    files = write_obsparquet(ds_obs, tmp_path)

    assert {f.parent.relative_to(tmp_path).as_posix() for f in files} == {
        "valid_year=2026/valid_month=1/valid_day=1",
        "valid_year=2026/valid_month=1/valid_day=2",
    }
    assert all(f.name.startswith("synop-") for f in files)

    schema = pq.read_schema(tmp_path / "schema" / "synop-schema.parquet")
    assert {"valid_dttm", "SID", "lat", "lon", "elev", "T2m"} <= set(schema.names)
    assert schema.field("valid_dttm").type == pa.int64()

    params = pq.read_table(tmp_path / "params" / "params.parquet").to_pandas()
    assert params.to_dict("records") == [{"parameter": "T2m", "units": "K"}]

    df = read_hive(tmp_path, ["valid_year", "valid_month", "valid_day"]).to_pandas()
    assert len(df) == ds_obs.sizes["valid_time"] * ds_obs.sizes["point_index"]


def test_obsparquet_incremental_writes_merge_metadata(tmp_path):
    """A second write adds data, schema columns and params without losing any."""
    write_obsparquet(make_observations(end="2026-01-01T05"), tmp_path)
    ds_later = make_observations(end="2026-01-01T11", altitude=True).isel(
        valid_time=slice(6, None)
    )
    ds_later["msl"] = ds_later["2t"] * 0 + 101300.0
    ds_later["msl"].attrs["units"] = "Pa"
    write_obsparquet(ds_later, tmp_path)

    df = read_hive(tmp_path, ["valid_year", "valid_month", "valid_day"]).to_pandas()
    assert len(df) == 12 * len(STATION_CODES)
    schema = pq.read_schema(tmp_path / "schema" / "synop-schema.parquet")
    assert {"T2m", "Pmsl"} <= set(schema.names)
    params = pq.read_table(tmp_path / "params" / "params.parquet").to_pandas()
    assert sorted(params["parameter"]) == ["Pmsl", "T2m"]
