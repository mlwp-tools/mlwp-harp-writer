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
