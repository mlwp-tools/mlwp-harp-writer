"""Read and check mlwp-data-specs trait attributes.

Traits are stored in the ``mlwp_{time,space,uncertainty}_trait`` dataset
attributes by mlwp-data-loaders and kept up to date by mxalign when it changes
a dataset (e.g. from grid to point when interpolating).
"""

from __future__ import annotations

import xarray as xr
from mlwp_data_specs import validate_dataset
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

_TRAIT_ATTRS = {
    "time": TIME_TRAIT_ATTR,
    "space": SPACE_TRAIT_ATTR,
    "uncertainty": UNCERTAINTY_TRAIT_ATTR,
}


def get_traits(ds_in: xr.Dataset) -> dict[str, str | None]:
    """Return the traits of a dataset.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset to read the trait attributes from.

    Returns
    -------
    dict[str, str | None]
        Mapping with keys ``time``, ``space`` and ``uncertainty`` to plain
        string values (Enum members, as set by mlwp-data-loaders, are
        converted to their ``value``). Missing time/space traits are ``None``;
        a missing uncertainty trait defaults to ``"deterministic"``.
    """
    traits = {}
    for name, attr in _TRAIT_ATTRS.items():
        value = ds_in.attrs.get(attr)
        traits[name] = None if value is None else str(getattr(value, "value", value))
    if traits["uncertainty"] is None:
        traits["uncertainty"] = "deterministic"
    return traits


def validate(
    ds_in: xr.Dataset,
    *,
    time: str | None = None,
    space: str | None = None,
    uncertainty: str | None = None,
) -> None:
    """Validate a dataset against mlwp-data-specs and raise if any check fails.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset to validate.
    time, space, uncertainty : str, optional
        Traits to validate against. Traits that are not passed are resolved
        from the dataset attributes.

    Raises
    ------
    ValueError
        If any mlwp-data-specs check fails. The message lists the failures.
    """
    traits = get_traits(ds_in)
    report = validate_dataset(
        ds_in,
        time=time or traits["time"],
        space=space or traits["space"],
        uncertainty=uncertainty or traits["uncertainty"],
    )
    if report.has_fails():
        fails = [
            f"- [{r.section}] {r.requirement}: {r.detail}"
            for r in report.results
            if r.status == "FAIL"
        ]
        raise ValueError(
            "Dataset does not conform to mlwp-data-specs:\n" + "\n".join(fails)
        )


def require_traits(
    ds_in: xr.Dataset, *, context: str, **expected: str | set[str]
) -> dict[str, str | None]:
    """Check that the traits of a dataset have the expected values.

    Parameters
    ----------
    ds_in : xr.Dataset
        Dataset whose traits are checked.
    context : str
        Description of the caller, used in the error message.
    **expected : str or set of str
        Allowed value(s) per trait name, e.g. ``time="forecast"``.

    Returns
    -------
    dict[str, str | None]
        The traits of ``ds_in`` as returned by :func:`get_traits`.

    Raises
    ------
    ValueError
        If a trait does not have one of the allowed values.
    """
    traits = get_traits(ds_in)
    for name, allowed in expected.items():
        allowed = {allowed} if isinstance(allowed, str) else set(allowed)
        if traits[name] not in allowed:
            raise ValueError(
                f"{context} requires {name} trait in {sorted(allowed)}, "
                f"got {traits[name]!r}"
            )
    return traits
