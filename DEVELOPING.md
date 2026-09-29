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

## Trying the output in harp (Docker)

To browse the HARP parquet datasets the end-to-end test writes in harp's web
UI (harpVis' point verification app), run R and harp in Docker. No local R
installation is needed.

`dev/harp/Dockerfile` builds an image with harpCore, harpIO (with parquet
support), harpPoint and harpVis:
- It's based on [r2u](https://github.com/eddelbuettel/r2u), so CRAN
  packages install as prebuilt binaries.
- harp isn't on CRAN or R-universe, and its latest releases (v0.2.x, 2024)
  predate harpIO's parquet support. Each harp package is therefore installed
  from a pinned commit tarball (`dev/harp/install_harp_package.R`), one
  cached layer per package.
- This uses no GitHub API, so there are no rate limits. A cold build takes a
  few minutes.

1. Build the image:

   ```bash
   docker build -t mlwp-harp -f dev/harp/Dockerfile dev/harp
   ```

2. Run the end-to-end test and keep its output. pytest's `--basetemp` keeps
   the temporary directories, and the test writes to `<basetemp>/harp0`:

   ```bash
   uv run python -m pytest tests/test_e2e_meps_obstable.py --basetemp=/tmp/mlwp-harp-e2e
   ```

   This gives `/tmp/mlwp-harp-e2e/harp0/FCPARQUET` (MEPS forecasts) and
   `/tmp/mlwp-harp-e2e/harp0/OBSPARQUET` (observations).

3. Start the web UI and open <http://localhost:3838>:

   ```bash
   docker run --rm -p 3838:3838 -v /tmp/mlwp-harp-e2e/harp0:/data mlwp-harp
   ```

   On first start, the verification is computed and saved to
   `/tmp/mlwp-harp-e2e/harp0/verification`; later starts reuse it:
   - `dev/harp/verify_meps_obstable.R` reads each parameter with
     `read_point_forecast`/`read_point_obs`;
   - joins forecasts and observations with `join_to_fcst`;
   - saves the `det_verify` scores per lead time with `save_point_verif`.

   `dev/harp/shiny_app.R` then starts `harpVis::shiny_plot_point_verif()` on
   those results. Delete the `verification` directory to recompute.

   To only print the scores instead:

   ```bash
   docker run --rm -v /tmp/mlwp-harp-e2e/harp0:/data mlwp-harp /scripts/verify_meps_obstable.R /data
   ```

   Expect realistic MEPS scores for 600–800 stations:
   - T2m RMSE of about 1.2 K at +0 h, rising to 2 K at +24 h;
   - Pmsl RMSE of about 1.3 hPa.

   Ps has a large RMSE (~13 hPa) because nothing corrects surface pressure for
   the difference between station height and model orography. The "removed
   due to gross error check" warnings are harp's own quality control of the
   observations.

4. Or explore interactively in R:

   ```bash
   docker run --rm -it -v /tmp/mlwp-harp-e2e/harp0:/data --entrypoint R mlwp-harp
   ```

   ```r
   library(harpIO)
   library(harpPoint)
   fc <- read_point_forecast(
     dttm = seq_dttm(2019021700, 2019021712, "12h"), fcst_model = "MEPS",
     parameter = "T2m", lead_time = c(0, 6, 12, 24),
     file_path = "/data/FCPARQUET", file_format = "fcparquet",
     file_template = "{fcst_model}/{parameter}"
   )
   obs <- read_point_obs(
     dttm = unique_valid_dttm(fc), parameter = "T2m",
     file_path = "/data/OBSPARQUET", file_format = "obsparquet"
   )
   det_verify(join_to_fcst(fc, obs), T2m)
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
