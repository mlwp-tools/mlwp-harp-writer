"""Low-level helpers for writing hive-partitioned parquet datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads


def date_partitions(times, prefix: str) -> dict[str, pd.Series]:
    """Return int32 date components used as hive partition columns.

    Parameters
    ----------
    times : array_like
        Date-times to split up (anything accepted by ``pd.DatetimeIndex``).
    prefix : str
        Column name prefix, e.g. ``"fcst"`` gives ``fcst_year`` etc.

    Returns
    -------
    dict[str, pd.Series]
        ``{prefix}_hour``, ``{prefix}_year``, ``{prefix}_month`` and
        ``{prefix}_day`` columns.
    """
    times = pd.DatetimeIndex(times)
    return {
        f"{prefix}_{part}": pd.Series(getattr(times, part), dtype="int32")
        for part in ("hour", "year", "month", "day")
    }


def write_hive_dataset(
    table: pa.Table, base_dir: Path, partitioning: list[str], basename_template: str
) -> list[Path]:
    """Write a table as a hive-partitioned parquet dataset.

    Existing files with the same name are overwritten and other files are
    kept, so data can be added incrementally (e.g. one forecast cycle at a
    time) as long as ``basename_template`` is unique per write.

    Parameters
    ----------
    table : pa.Table
        Data to write, including the partition columns.
    base_dir : Path
        Root directory of the dataset.
    partitioning : list of str
        Partition columns, outermost first.
    basename_template : str
        File name template, must contain ``{i}``.

    Returns
    -------
    list of Path
        The parquet files written.
    """
    written = []
    pads.write_dataset(
        table,
        base_dir,
        format="parquet",
        partitioning=pads.partitioning(
            pa.schema([table.schema.field(p) for p in partitioning]), flavor="hive"
        ),
        basename_template=basename_template,
        existing_data_behavior="overwrite_or_ignore",
        file_visitor=lambda f: written.append(Path(f.path)),
    )
    return written
