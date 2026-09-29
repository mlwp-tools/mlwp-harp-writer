# mlwp-harp-writer

Write [mxalign](https://github.com/mlwp-tools/mxalign)-aligned,
[mlwp-data-specs](https://github.com/mlwp-tools/mlwp-data-specs) conforming
datasets to [HARP](https://harphub.github.io/) parquet datasets.

This lets you verify forecasts with harp's point verification tools
(`harpPoint`). Typical inputs are ML weather models loaded with
[mlwp-data-loaders](https://github.com/mlwp-tools/mlwp-data-loaders).

> [!WARNING]
> Until mxalign `v0.2.0` is released, this package depends on **unmerged
> upstream code**. mxalign is pinned to a commit on its
> [`refactor/alignment`](https://github.com/mlwp-tools/mxalign/tree/refactor/alignment)
> branch, which switches mxalign to mlwp-data-specs traits and
> mlwp-data-loaders. That branch, and so this package's API and behaviour,
> may still change.

## How it fits together

```
mlwp-data-loaders          mxalign                     mlwp-harp-writer
(load + validate)   ──►    (grid → stations,    ──►    (write fcparquet /
xr.Dataset per source       common valid times)         obsparquet datasets)
```

- **Forecasts** (`forecast` time trait, `grid` or `point` space trait, and
  `deterministic` or `ensemble` uncertainty trait) are interpolated to the
  observation stations with mxalign.
- **Observations** (`observation` + `point` traits) define the stations and
  the valid times.
- The result is written in the layout harpIO itself uses for
  `fcparquet`/`obsparquet`, so harp reads it natively.

## Installation

```bash
uv add "mlwp-harp-writer @ git+https://github.com/mlwp-tools/mlwp-harp-writer"
```

This also installs mxalign and mlwp-data-loaders. mxalign is currently pinned
to its
[`refactor/alignment`](https://github.com/mlwp-tools/mxalign/tree/refactor/alignment)
branch, which uses mlwp-data-specs traits.

## Usage

### 1. Load data with mlwp-data-loaders

`load_and_validate_dataset` opens the source files, sets the trait attributes
and validates the result against mlwp-data-specs:

```python
from mlwp_data_loaders import load_and_validate_dataset

ds_fcst = load_and_validate_dataset(
    [
        "/path/to/anemoi-inference-2026010100.nc",
        "/path/to/anemoi-inference-2026010112.nc",
    ],
    loader="mlwp_data_loaders.loaders.anemoi.anemoi_inference",
)

ds_obs = load_and_validate_dataset(
    "/path/to/OBSTABLE_2026.sqlite",
    loader="mlwp_data_loaders.loaders.harp.obstable",
)
```

### 2. Prepare variable metadata

Variables are mapped to HARP parameters by their CF `standard_name`, and
values are converted from their `units` (see
[Parameter names and units](#parameter-names-and-units)). Level-dependent
quantities such as 2 m temperature or 10 m wind also need a **height
coordinate** with units.

Setting this metadata is the loader's job: a loader knows what its variables
are. mlwp-data-loaders doesn't do it for data variables yet, and
[this note](docs/upstream-issues/mlwp-data-loaders-variable-standard-names.md)
proposes it. Until then, add what the loader leaves out to the loaded datasets
**before aligning**, so it travels through mxalign:

```python
ds_fcst["2t"].attrs.update(standard_name="air_temperature", units="K")
ds_fcst = ds_fcst.assign_coords(
    height=((), 2.0, {"standard_name": "height", "units": "m"})
)
```

Do the same for the observations, e.g. for `T2m` in `ds_obs`.

The scalar `height` coordinate, given as a `(dims, value, attrs)` tuple,
applies to every variable in the Dataset. For variables at different heights
(e.g. 2 m temperature and 10 m wind), give each variable its own size-1
height dimension instead, e.g. `height_2m` and `height_10m`.

### 3. Align with mxalign

Importing mxalign registers the `ds.mx` accessor on every `xr.Dataset`:

```python
import mxalign as mx

# grid -> observation stations
ds_fcst_points = ds_fcst.mx.align_space_with(ds_obs, method="xarray")

# keep the forecast cycles within the observation period. This can't use the
# accessor yet: ds_fcst_points.mx.align_time_with(ds_obs) drops
# reference_time and lead_time (see below)
ds_fcst_aligned = mx.align_time(ds_fcst_points, reference=ds_obs)
```

- **Time alignment can't use the accessor yet.**
  - `ds_fcst.mx.align_time_with(ds_obs)` collapses the forecast onto the
    observations' `valid_time`. It keeps one lead time per valid time and
    drops `reference_time` (analysis time) and `lead_time`, which HARP needs.
  - This should be fixed upstream in mxalign, e.g. with an option to keep
    the forecast structure; see
    [this note](docs/upstream-issues/mxalign-align-time-with-keep-lead-time.md).
  - Until then, use the module-level `mx.align_time` with the observations
    as `reference`. With a single Dataset it returns a single Dataset and
    keeps `reference_time`/`lead_time`.
- **Use `method="xarray"`.**
  - mxalign's `method="delaunay"` currently drops the variables'
    `standard_name`/`units` and scalar coordinates such as `height`, so
    nothing can be mapped to HARP parameters afterwards.
  - This should be fixed upstream in mxalign; see
    [this note](docs/upstream-issues/mxalign-delaunay-drops-variable-metadata.md).
- **The xarray method needs `latitude`/`longitude` dimensions**, or projected
  `xc`/`yc` with a CRS: see `ds.mx.add_crs`/`ds.mx.add_grid_mapping` and
  `ds.mx.unstack`.
  - The anemoi loaders return grids on a flattened `grid_index`. For a
    regular lat/lon grid, unstack it first:

    ```python
    ds_fcst = ds_fcst.set_index(grid_index=["latitude", "longitude"]).unstack(
        "grid_index"
    )
    ```
- **Time alignment can't use the accessor yet.**
  - `ds_fcst.mx.align_time_with(ds_obs)` collapses the forecast onto the
    observations' `valid_time`. It keeps one lead time per valid time and
    drops `reference_time` (analysis time) and `lead_time`, which HARP needs.
  - This should be fixed upstream in mxalign, e.g. with an option to keep
    the forecast structure; see
    [this note](docs/upstream-issues/mxalign-align-time-with-keep-lead-time.md).
  - Until then, use the module-level `mx.align_time` with the observations
    as `reference`. With a single Dataset it returns a single Dataset and
    keeps `reference_time`/`lead_time`.
- The time step is optional for HARP, which matches forecasts with
  observations by valid time when it reads them.
- mxalign requires `lead_time` to be `timedelta64`. The anemoi loaders in
  mlwp-data-loaders provide that.
- For several forecasts, pass a dict to `mx.align_space`/`mx.align_time`
  instead; see [Several forecasts](#several-forecasts).

### 4. Write HARP parquet

```python
from mlwp_harp_writer import write_harp_parquets

write_harp_parquets(ds_fcst_aligned, ds_obs, "harp_data/", fcst_model="my-ai-model")
```

- mxalign's interpolation keeps only the stations' `latitude`/`longitude`.
  `write_harp_parquets` therefore copies the station metadata (`code`, used as HARP
  `SID`, and `elevation`/`altitude`) from the observations. It checks first
  that the locations match.
- All observation times in `ds_obs` are written. harp only reads the valid
  times it needs.

This writes:

```
harp_data/
├── FCPARQUET/my-ai-model/T2m/fcst_hour=0/fcst_year=2026/fcst_month=1/fcst_day=1/*.parquet
└── OBSPARQUET/
    ├── valid_year=2026/valid_month=1/valid_day=1/synop-*.parquet
    ├── schema/synop-schema.parquet
    └── params/params.parquet
```

### Several forecasts

To compare several forecasts (e.g. an AI model and an NWP baseline), pass a
dict keyed by model name to mxalign's module-level functions. They return a
dict with the same keys, and `write_harp_parquets` accepts that dict
directly, using the keys as HARP `fcst_model` names:

```python
import mxalign as mx
import xarray as xr

forecasts: dict[str, xr.Dataset] = mx.align_space(
    {"my-ai-model": ds_fcst, "nwp": ds_nwp}, reference=ds_obs, method="xarray"
)
# common (reference_time, lead_time) grid within the observation period
aligned: dict[str, xr.Dataset] = mx.align_time(forecasts, reference=ds_obs)
write_harp_parquets(aligned, ds_obs, "harp_data/")
```

- `mx.align_time` needs `lead_time` to be `timedelta64`.
- To write only the observations at the forecast valid times, include them
  in the dict: `mx.align_time({**forecasts, "obs": ds_obs}, reference="obs")`.
  mxalign then returns them reshaped to `(reference_time, lead_time)`. Pass
  `aligned.pop("obs")` to `write_harp_parquets`, which accepts that shape
  as well.
  Currently this fails when cycles overlap in valid time; see
  [Limitations](#limitations).

### 5. Verify in R with harp

This needs a harpIO version with parquet support (current `master`):

```r
library(harpIO)
library(harpPoint)

fc <- read_point_forecast(
  dttm          = seq_dttm(2026010100, 2026010112, "12h"),
  fcst_model    = "my-ai-model",
  fcst_type     = "det",
  parameter     = "T2m",
  file_path     = "harp_data/FCPARQUET",
  file_format   = "fcparquet",
  file_template = "{fcst_model}/{parameter}"
)
obs <- read_point_obs(
  dttm        = unique_valid_dttm(fc),
  parameter   = "T2m",
  obs_path    = "harp_data/OBSPARQUET",
  file_format = "obsparquet"
)
verif <- det_verify(join_to_fcst(fc, obs), T2m)
```

### Writing forecasts and observations separately

Point datasets can be written on their own, e.g. to add one forecast cycle at
a time. File names include the time range, so earlier writes are kept:

```python
from mlwp_harp_writer import write_fcparquet, write_obsparquet

write_fcparquet(
    ds_fcst_aligned, "harp_data/FCPARQUET", "my-ai-model", ds_stations=ds_obs
)
write_obsparquet(ds_obs, "harp_data/OBSPARQUET")
```

## Parameter names and units

Variables are mapped to HARP parameters by their CF `standard_name` and, for
near-surface quantities, the height of their `height` coordinate
(`mlwp_harp_writer.CF_TO_HARP`). Values are converted to the units HARP uses
for each parameter:

| `standard_name` | height | HARP parameter | HARP units |
|---|---|---|---|
| `air_temperature` | 2 m | `T2m` | K |
| `dew_point_temperature` | 2 m | `Td2m` | K |
| `relative_humidity` | 2 m | `RH2m` | percent |
| `specific_humidity` | 2 m | `Q2m` | kg/kg |
| `wind_speed` | 10 m | `S10m` | m/s |
| `wind_from_direction` | 10 m | `D10m` | degrees |
| `wind_speed_of_gust` | 10 m | `G10m` | m/s |
| `air_pressure_at_mean_sea_level` | – | `Pmsl` | hPa |
| `surface_air_pressure` | – | `Ps` | hPa |
| `cloud_area_fraction` | – | `CCtot` | oktas |
| `visibility_in_air` | – | `vis` | m |

- **Height:**
  - Standard names with a height in the table need a height coordinate:
    either a scalar coord or a size-1 dimension (see
    [step 2](#2-prepare-variable-metadata)), with `standard_name="height"`
    or named `height`, and units `m` or `km`.
  - A missing height, a height coordinate without units, or several height
    coordinates on one variable raise `ValueError`, with an example of how
    to add one.
  - A height with no HARP parameter (e.g. 100 m wind) is skipped with a
    warning.
- **Units:**
  - Values are converted from each variable's `units` attribute, e.g. Pa → hPa,
    degC → K, a cloud fraction `1` → oktas, a relative humidity `1` →
    percent.
  - Missing or unsupported units raise `ValueError`.
  - Conversions work on the `units` attribute. After an mxalign
    transformation such as `mx.transform("kelvin_to_celcius", ...)`, set
    `units` yourself (e.g. `ds_fcst["2t"].attrs["units"] = "degC"`), since the
    transformation doesn't update it.
- **Variables without a `standard_name`** are skipped with a warning.
- **Explicit mappings:** map variables explicitly with `params=`, e.g.
  `params={"T2m": "T2m"}` or `params={"my_var": HarpParam("Foo", "")}`, on
  the `write_*` functions.
  - HARP names from the table are converted to that parameter's units.
  - Other names are written with the variable's own `units`.
- **Not supported:** pressure-level and multi-height variables raise
  `NotImplementedError`.

## Output format details

Forecasts (`fcparquet`, one dataset per `{fcst_model}/{parameter}`):

| column | contents |
|---|---|
| `fcst_dttm`, `valid_dttm` | int64 unix seconds |
| `lead_time` | int64 seconds |
| `SID` | station id, from the `code` coord (copied from the obs if missing; `point_index` as a last resort) |
| `lat`, `lon` | station location |
| `model_elevation` | optional, via `write_fcparquet(..., model_elevation="orog")` |
| `parameter`, `units` | HARP parameter name and units |
| `{model}_det` / `{model}_mbr000`… | forecast values (deterministic / per ensemble member) |

Hive partitions are `fcst_hour/fcst_year/fcst_month/fcst_day`.

Observations (`obsparquet`) form one wide table:
- `valid_dttm`, `SID`, `lat`, `lon`, `elev`, plus one column per parameter.
- Hive partitions are `valid_year/valid_month/valid_day`.
- A schema file and a `params` (units) table sit alongside the data.

## Limitations

- Only single-level (surface) parameters are supported; variables with a
  vertical dimension raise `NotImplementedError`.
- Precipitation accumulations (`AccPcp*h`, CF `precipitation_amount` with
  `cell_methods`) are not mapped.
- **Known mxalign issue:** `method="delaunay"` drops the variables'
  `standard_name`/`units` and scalar coordinates, so use `method="xarray"`.
  See
  [the write-up](docs/upstream-issues/mxalign-delaunay-drops-variable-metadata.md)
  for the cause and a proposed fix. The affected test here is marked `xfail`.
- Point forecasts must be at the observation stations. mxalign does not do
  point-to-point matching yet.
- Quantile forecasts are not supported.
- **Known mxalign issue:** on mxalign's `refactor/alignment` branch,
  `mx.align_time({**forecasts, "obs": ds_obs}, reference="obs")` fails with
  `InvalidIndexError: Reindexing only valid with uniquely valued Index objects`
  when forecast cycles share valid times. For example, 12-hourly cycles
  with a 12 h or longer lead time (00 UTC +12 h = 12 UTC +0 h).
  - Passing the observations as `reference=ds_obs`, without them in the dict,
    is not affected.
  - See
    [the write-up](docs/upstream-issues/mxalign-align-time-overlapping-valid-times.md)
    for the cause and a proposed fix.
  - The affected test here is marked `xfail`.

## Development

See [DEVELOPING.md](DEVELOPING.md).
