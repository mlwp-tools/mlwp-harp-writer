# mxalign: delaunay interpolation drops variable attributes and scalar coordinates

- **Upstream:** [mlwp-tools/mxalign](https://github.com/mlwp-tools/mxalign),
  branch `refactor/alignment`, commit `81482db`
- **Status:** not yet reported upstream
- **Affects mlwp-harp-writer:**
  - The writers map variables to HARP parameters by CF `standard_name`, a
    `height` coordinate and `units`. After delaunay interpolation none of
    these are left, so every variable is skipped or rejected.
  - The README therefore uses `method="xarray"`, which keeps all metadata.
  - The delaunay case of `test_mxalign_to_harp` is a strict `xfail`
    (`MXALIGN_DELAUNAY_METADATA_BUG` in `tests/conftest.py`). Once mxalign is
    fixed, remove the marker, bump the pin in `pyproject.toml`, and mention
    delaunay again in the README.

## Summary

After `ds_fcst.mx.align_space_with(ds_obs, method="delaunay")`:

- **every data variable has empty `attrs`**: `standard_name`, `units`,
  `long_name` etc. are gone;
- **coordinates of the source that are not on `grid_index` are gone**, e.g. a
  scalar `height` coordinate.

  Output variables only carry the target's `latitude`/`longitude`. Those
  bring along the *target's* coordinates, so the target's own scalar coords
  may show up and mask the loss.

Size-1 dimension coordinates (e.g. a `height_2m` dim) survive, because they
are leading dims of the variable. The `xarray` interpolator keeps everything.

## Reproduction

```python
import mxalign  # registers the ds.mx accessor
import numpy as np
import xarray as xr

lat, lon = np.meshgrid(np.arange(54, 58.1, 0.5), np.arange(8, 13.1, 0.5), indexing="ij")
ds_fcst = xr.Dataset(
    {"2t": (["grid_index"], np.full(lat.size, 280.0), {"standard_name": "air_temperature", "units": "K"})},
    coords={
        "latitude": ("grid_index", lat.ravel()),
        "longitude": ("grid_index", lon.ravel()),
        "height": xr.DataArray(2.0, attrs={"standard_name": "height", "units": "m"}),
    },
    attrs={"mlwp_time_trait": "observation", "mlwp_space_trait": "grid"},
)
ds_obs = xr.Dataset(
    coords={"latitude": ("point_index", [55.2, 56.3]), "longitude": ("point_index", [9.1, 10.7])},
    attrs={"mlwp_time_trait": "observation", "mlwp_space_trait": "point"},
)

ds_points = ds_fcst.mx.align_space_with(ds_obs, method="delaunay")
print(ds_points["2t"].attrs)             # {}  (expected standard_name/units)
print("height" in ds_points["2t"].coords)  # False
```

## Cause

In `src/mxalign/interpolations/delaunay.py`:

- `interpolate_da` builds the output with `da_clean.map_blocks(..., template=tmp)`.
  The template `tmp` is a fresh DataArray with no `attrs`, and only the
  coordinates of the leading dims.
- `DelaunayInterpolator._interpolate` then builds
  `xr.Dataset(arrays_out).assign_coords(latitude=..., longitude=...)` from
  those DataArrays. It copies `source_dataset.attrs` (the traits) but no
  variable attrs, and no other source coordinates.

## Proposed fix

1. At the end of `interpolate_da`, keep the variable attributes:

   ```python
   da_interp.attrs = dict(da.attrs)
   ```

2. In `DelaunayInterpolator._interpolate`, keep the source coordinates that
   are not on `grid_index`:

   ```python
   keep = {
       name: coord
       for name, coord in source_dataset.coords.items()
       if "grid_index" not in coord.dims and name not in ds_out.coords
   }
   ds_out = ds_out.assign_coords(keep)
   ```

I applied both at runtime (by wrapping the two functions) on the example
above. `2t` then keeps `{"standard_name": "air_temperature", "units": "K"}`
and its scalar `height` coordinate.
