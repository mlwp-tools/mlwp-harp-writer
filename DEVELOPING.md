# Developing mlwp-harp-writer

## Setup

```bash
uv sync --all-extras --group dev
uv run pre-commit install
```

## Tests

```bash
uv run python -m pytest
```

- `tests/test_loaders_integration.py` runs the README workflow offline:
  it writes small anemoi-inference netCDF files and a HARP OBSTABLE,
  loads them with mlwp-data-loaders, then aligns and writes them.
- `tests/test_e2e_meps_obstable.py` (marker `network`) runs the whole process
  on real data. It takes about 30 s, mostly downloading, and is skipped if
  the data can't be downloaded.
  - **Data:** MET Norway MEPS forecasts (cycles 2019-02-17 00/12 UTC, a
    small OPeNDAP subset from THREDDS), and harpData's
    `OBSTABLE_2019.sqlite` (the harp tutorial period, which MEPS covers).
    The mlwp-data-loaders sample datasets don't overlap in time.
  - **Loaders:** it uses the test loaders in `tests/loaders/`, which follow
    the mlwp-data-loaders loader contract and add the CF metadata the HARP
    mapping needs.
  - **Checks:** HARP parameters and units, stations in the MEPS domain, and
    loose forecast–observation RMSE bounds (right stations, times and units,
    not model skill).
  - **Skip it** with `uv run python -m pytest -m "not network"`.
- `tests/test_harp_roundtrip.py` (marker `harp`) reads the output back with
  harpIO and verifies it with harpPoint in R. It is skipped unless
  `Rscript` is available with harpIO (parquet support, current master) and
  harpPoint installed:

  ```r
  remotes::install_github("harphub/harpIO")
  remotes::install_github("harphub/harpPoint")
  ```

## Dependencies

`mlwp-data-specs`, `mxalign` and `mlwp-data-loaders` come from git in
`[tool.uv.sources]` in `pyproject.toml`:

- `mxalign` is pinned to a commit on its (unmerged) `refactor/alignment`
  branch.
- `mlwp-data-loaders` uses `branch = "main"`, exactly as mxalign declares it.
- `mlwp-data-specs` uses the revision that mlwp-data-loaders pins.

uv reports conflicting URLs if a pin here differs from the pin in a
dependency's own pyproject. `uv.lock` records the exact commits.

See [AGENTS.md](AGENTS.md) for coding conventions.
