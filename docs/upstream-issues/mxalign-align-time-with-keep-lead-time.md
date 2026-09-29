# mxalign: option for `align_time_with` to keep `reference_time`/`lead_time`

- **Upstream:** [mlwp-tools/mxalign](https://github.com/mlwp-tools/mxalign),
  branch `refactor/alignment`, commit `81482db`
- **Type:** feature request / API suggestion
- **Status:** not yet raised upstream
- **Affects mlwp-harp-writer:** the README's single-forecast example uses the
  `ds.mx` accessor for spatial alignment (`ds_fcst.mx.align_space_with`).
  There is no accessor-based time alignment that keeps the forecast
  structure HARP needs, so the time step falls back to the module-level
  `mx.align_time(ds_fcst_points, reference=ds_obs)`. With a single Dataset
  that takes the global path and keeps `reference_time`/`lead_time`. Switch
  the README to the accessor once this is fixed.

## Current behaviour

For a forecast and an observation dataset,
`ds_fcst.mx.align_time_with(ds_obs, lead_time="shortest")`
(`align_forecast_to_observation` in `src/mxalign/accessors/time.py`) does
the following:

1. stacks `(reference_time, lead_time)` into `valid_time`;
2. keeps **one** forecast per valid time, chosen by the `lead_time` strategy
   (`"shortest"`, `"longest"`, or specific lead times);
3. drops the `reference_time` and `lead_time` coordinates;
4. reindexes onto the observations' `valid_time` and relabels the result with
   the `observation` time trait.

On our test data (two 12-hourly cycles, lead times 0/6/12 h, 4 stations), the
6 `(reference_time, lead_time)` pairs become 5 valid times. The 00 UTC +12 h
forecast is discarded in favour of 12 UTC +0 h.

That fits "compare the best available forecast with the observation at each
time". It doesn't fit forecast verification by lead time, as done by HARP,
scorecards or anything else that needs `reference_time`/`lead_time`. Those
need every forecast kept.

## Suggestion

Add a keyword argument to `align_time_with` for the forecast→observation
case, e.g. `keep_lead_time=True`, that keeps the forecast structure. It would
return the forecast restricted to `(reference_time, lead_time)` pairs whose
valid time lies in the observation period, with a 2-D `valid_time`
coordinate, and keep the `forecast` time trait. That is what the global
`mx.align_time({...}, reference=ds_obs)` already does for forecasts.

Alternatively, `ds_obs.mx.align_time_with(ds_fcst)` (observation→forecast)
already broadcasts the observations onto the forecast grid. The docs could
point to it as the way to "align" a single forecast and observation pair
without losing lead times.

With either, the README here could show a single-forecast time alignment
through the accessor, e.g.:

```python
ds_fcst_aligned = ds_fcst_points.mx.align_time_with(ds_obs, keep_lead_time=True)
write_harp_parquets(ds_fcst_aligned, ds_obs, "harp_data/", fcst_model="my-ai-model")
```
