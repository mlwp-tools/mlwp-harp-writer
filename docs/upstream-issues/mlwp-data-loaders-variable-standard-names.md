# mlwp-data-loaders: set CF `standard_name`, `units` and height on data variables

- **Upstream:** [mlwp-tools/mlwp-data-loaders](https://github.com/mlwp-tools/mlwp-data-loaders),
  `main` (`a50382f`); possibly also
  [mlwp-tools/mlwp-data-specs](https://github.com/mlwp-tools/mlwp-data-specs)
- **Type:** feature request
- **Status:** not yet raised upstream
- **Affects mlwp-harp-writer:**
  - The writers map data variables to HARP parameters by their CF
    `standard_name` and a `height` coordinate, and convert values from their
    `units`.
  - The loaders currently set `standard_name`/`units` on coordinates only.
    Users therefore have to add the variable metadata themselves; see the
    "Prepare variable metadata" step in the README.

## Current behaviour

| Loader | Data variables | CF metadata on data variables |
|---|---|---|
| `anemoi.anemoi_inference` | ECMWF short names (`2t`, `10u`, `msl`, …) | none |
| `anemoi.anemoi_datasets` | ECMWF short names | none |
| `harp.obstable` | HARP names (`T2m`, `S10m`, `Pmsl`, …) | none |
| `ifs.forecast` | via cfgrib | whatever cfgrib sets (GRIB-derived attrs) |

## Suggestion

Have each loader set `standard_name` and `units` on the data variables it
knows. For near-surface quantities, also set a height coordinate in metres.
A size-1 dimension per variable (e.g. `height_2m`, `height_10m`, each with
`standard_name="height"` and `units="m"`) lets variables at different heights
live in one Dataset. It also survives mxalign interpolation.

For example:

| Loader variable | `standard_name` | height | `units` |
|---|---|---|---|
| anemoi `2t` / obstable `T2m` | `air_temperature` | 2 m | `K` |
| anemoi `2d` / obstable `Td2m` | `dew_point_temperature` | 2 m | `K` |
| anemoi `10si` / obstable `S10m` | `wind_speed` | 10 m | `m s-1` |
| obstable `D10m` | `wind_from_direction` | 10 m | `degree` |
| anemoi `10fg` / obstable `G10m` | `wind_speed_of_gust` | 10 m | `m s-1` |
| obstable `RH2m` | `relative_humidity` | 2 m | `%` |
| anemoi `msl` | `air_pressure_at_mean_sea_level` | – | `Pa` |
| obstable `Pmsl` | `air_pressure_at_mean_sea_level` | – | `hPa` |
| anemoi `sp` / obstable `Ps` | `surface_air_pressure` | – | `Pa` / `hPa` |
| anemoi `tcc` / obstable `CCtot` | `cloud_area_fraction` | – | `1` / `oktas` |

mlwp-data-specs could then add a variable-level check: data variables carry
`standard_name` and `units`, and level-dependent ones carry a vertical
coordinate. mlwp-data-specs#1 (CF-conformance critique) currently covers
coordinates only.
