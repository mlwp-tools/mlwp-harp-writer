# AGENTS

Guidance for agents and contributors working in this repository.

## Project intent

- `mlwp-harp-writer` writes mxalign-aligned, mlwp-data-specs conforming
  `xr.Dataset`s to HARP parquet datasets (harpIO `fcparquet` / `obsparquet`),
  so they can be verified with `harpPoint`.
- It is a **library only** (no CLI, no config files) and does **no
  alignment**. Users load data with `mlwp-data-loaders` and align it by
  calling mxalign directly.
  - For a single forecast, the README uses the `ds.mx` accessor
    (`ds_fcst.mx.align_space_with(ds_obs)`) for space. It uses the
    module-level `mx.align_time(ds, reference=ds_obs)` for time, because the
    accessor's `align_time_with` drops `reference_time`/`lead_time`; this
    should be fixed upstream. It then calls
    `write_harp_parquets(ds_fcst_aligned, ds_obs, path, fcst_model=...)`.
  - For several forecasts, it passes a `dict` through `mx.align_space` and
    `mx.align_time` to `write_harp_parquets`.
  - README examples import names directly, e.g.
    `from mlwp_harp_writer import write_harp_parquets`, with no module alias.
  - Workarounds for mxalign behaviour belong upstream in mxalign, not here.
- Don't use `ds_fcst.mx.align_time_with(ds_obs)` before writing. It collapses
  the forecast onto `valid_time` and drops `reference_time`/`lead_time`; see
  `docs/upstream-issues/mxalign-align-time-with-keep-lead-time.md`.
- It does **not** convert units or compute derived variables. That belongs
  upstream (mxalign transformations).

## Coding conventions

- **`xr.Dataset` objects are always named `ds_<name>` and `xr.DataArray`
  objects `da_<name>`.** This covers variables, function parameters, loop
  variables, fixtures and README/docs examples. Bare `ds`/`da` are not allowed:
  use `ds_fcst`, `ds_obs`, `ds_in` (generic input), `ds_out`,
  `da_lead_time`, and so on.
  - Don't wrap Datasets in custom container classes. Use plain Datasets, or
    tuples/dicts of them, as mxalign does.
  - A `dict[str, xr.Dataset]` is named for its contents, without a prefix
    (e.g. `forecasts`). Annotate it inline wherever it is bound, especially
    results of `mx.align_*`:
    `forecasts: dict[str, xr.Dataset] = mx.align_space(...)`. Use the builtin
    `dict`, not `typing.Dict`.
- **numpy-style docstrings on every module, class, function and method**,
  including private helpers, tests and fixtures. Use `Parameters` / `Returns`
  / `Raises` sections where they apply.
- Formatting: black + isort (`profile = "black"`) + flake8
  (max-line-length 88), run through pre-commit.
- Logging: `loguru.logger`, never `print`.
- Shell commands: use absolute paths or `--directory`/`-C` flags rather than
  `cd <dir> && ...`.
- Edit files directly (not with `perl`/`sed` rewrites). Run ad-hoc Python
  from a script file, never through heredocs.

## Common commands

- Install/sync env: `uv sync --all-extras --group dev`
- Run tests: `uv run python -m pytest`
- Lint: `uv run pre-commit run --all-files`
- R round-trip tests (`-m harp`) run only if `Rscript` with harpIO (parquet
  support, i.e. current master) and harpPoint is on the PATH. Otherwise they
  are skipped.

## Structure

- Public API re-exports and `write_harp_parquets`:
  `src/mlwp_harp_writer/__init__.py`
- Writers: `src/mlwp_harp_writer/fcparquet.py` and `obsparquet.py`, sharing:
  - `_arrow.py`: hive-partitioned writing
  - `_stations.py`: SID, elevation, and copying station metadata from the obs
  - `_time.py`: unix seconds and lead time conversions
- Reading and checking mlwp-data-specs traits: `src/mlwp_harp_writer/traits.py`
- Variable -> HARP parameter table: `src/mlwp_harp_writer/params.py`
- Synthetic test data: `tests/conftest.py`
- Known upstream bugs (cause, reproduction, proposed fix, which tests are
  `xfail`ed): `docs/upstream-issues/`. Add a file there whenever a test is
  marked `xfail` because of a dependency, and delete it once the fix is
  pinned.

## Things that are easy to get wrong

- **mxalign is pinned to the unmerged `refactor/alignment` branch** (a commit
  in `[tool.uv.sources]`). That branch reads and writes the
  `mlwp_*_trait` attributes itself. `mlwp-data-loaders` must use the same
  source spec as in mxalign's pyproject (`branch = "main"`), otherwise uv
  reports conflicting URLs. Move to mxalign main or a release once the
  branch is merged.
- **Loaders store traits as `str` Enum members.** `get_traits` normalises them
  to plain strings, because Enum members don't hash like strings.
- **mxalign interpolation keeps only `latitude`/`longitude`** of the target
  stations. `write_fcparquet(..., ds_stations=ds_obs)`, which `write_harp_parquets`
  calls, copies `code`/`elevation`/`altitude` from the obs after checking
  the locations match. HARP needs `SID` to join forecasts with observations.
- **`mx.align_time` with an obs reference reshapes the obs** onto
  `(reference_time, lead_time)` and gives them the forecast time trait.
  `obs_on_valid_time` flattens them back to unique valid times for
  obsparquet.
- **The writers accept numeric or `timedelta64` lead times.** mxalign itself
  needs `timedelta64`.
- **The harpIO layout is fixed by harpIO's readers**
  (`harpIO/R/parquet.R`). Check against them before changing:
  - partition names and their int32 type
  - `lead_time` in seconds, and `fcst_dttm`/`valid_dttm` as int64 unix seconds
  - member columns `{model}_mbrNNN` / `{model}_det`
  - obs files prefixed `synop-`, plus the `schema/` and `params/` side files
