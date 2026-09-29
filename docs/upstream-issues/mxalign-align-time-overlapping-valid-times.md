# mxalign: global `align_time` fails on overlapping valid times when observations are in the datasets

- **Upstream:** [mlwp-tools/mxalign](https://github.com/mlwp-tools/mxalign),
  branch `refactor/alignment`, commit `81482db`
- **Status:** not yet reported upstream
- **Affects mlwp-harp-writer:** only the optional "observations in the
  `align_time` dict" path. The README's main path passes the observations as
  `reference=ds_obs` only, and is not affected. The test marked with
  `MXALIGN_OVERLAP_BUG` (`tests/conftest.py`) is a strict `xfail` until this
  is fixed. Remove the marker once mxalign is fixed, and bump the pin in
  `pyproject.toml`.

## Summary

When an observation dataset is one of the datasets being aligned,
`mx.align_time({"fcst": ds_fcst, "obs": ds_obs}, reference="obs")` raises

```
pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects
```

whenever two forecast cycles share a valid time. For example, 12-hourly
cycles with lead times 0, 6 and 12 h: 00 UTC +12 h and 12 UTC +0 h are both
12 UTC. Almost every real forecast archive has such overlaps.

`mx.align_time({"fcst": ds_fcst}, reference=ds_obs)`, with the observations
only as the reference, works. It does not reshape any observations.

## Reproduction

```python
import mxalign as mx  # registers the ds.mx accessor
import numpy as np
import pandas as pd
import xarray as xr

traits = {"mlwp_space_trait": "point", "mlwp_uncertainty_trait": "deterministic"}

ds_fcst = xr.Dataset(
    {"t": (["reference_time", "lead_time", "point_index"], np.zeros((2, 3, 1)))},
    coords={
        "reference_time": pd.to_datetime(["2026-01-01T00", "2026-01-01T12"]).values,
        "lead_time": np.array([0, 6, 12], dtype="timedelta64[h]").astype(
            "timedelta64[ns]"
        ),
    },
    attrs={**traits, "mlwp_time_trait": "forecast"},
)
ds_obs = xr.Dataset(
    {"t": (["valid_time", "point_index"], np.zeros((25, 1)))},
    coords={"valid_time": pd.date_range("2026-01-01T00", periods=25, freq="1h").values},
    attrs={**traits, "mlwp_time_trait": "observation"},
)

mx.align_time({"fcst": ds_fcst, "obs": ds_obs}, reference="obs")
# InvalidIndexError: Reindexing only valid with uniquely valued Index objects
```

With lead times 0 and 6 h only, no valid time repeats and the call succeeds.

## Cause

In `src/mxalign/align/time.py`, `_align_time_global` puts each observation
dataset *among the aligned datasets* onto the common grid with:

```python
out = ds.reindex(valid_time=valid_time_2d.values.ravel()).sel(
    valid_time=valid_time_2d
)
```

`valid_time_2d.values.ravel()` is the flattened `R* + L*` grid, which contains
duplicates when cycles overlap. pandas cannot reindex onto a non-unique index.

The existing global alignment tests pass because their fixtures happen not
to produce overlapping valid times.

## Proposed fix

Deduplicate the target times before reindexing. The vectorised `.sel` then
broadcasts the observations back onto `(reference_time, lead_time)`, so the
shared valid times get the same value in each cycle:

```python
out = ds.reindex(valid_time=np.unique(valid_time_2d.values.ravel())).sel(
    valid_time=valid_time_2d
)
```

I checked this on the example above. It returns observations on
`(reference_time=2, lead_time=3)`, and the shared 12 UTC valid time gets the
same value in both cycles.

A regression test would align two overlapping cycles, as in the
reproduction, against an observation reference. It would then check that
`aligned["obs"]` has dims `(reference_time, lead_time, ...)` and equal values
at the shared valid time.
