"""Read the written datasets back with harpIO/harpPoint in R.

These tests are skipped unless ``Rscript`` is on the PATH and the harpIO and
harpPoint R packages (with fcparquet/obsparquet support) are installed.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap

import pytest

from mlwp_harp_writer import FCPARQUET_DIR, OBSPARQUET_DIR, write_harp_parquets

from .test_write_harp_parquets import align_with_mxalign

pytestmark = pytest.mark.harp


def _have_harp() -> bool:
    """Check whether R with harpIO (fcparquet support) and harpPoint is available.

    Returns
    -------
    bool
        True if the round-trip tests can run.
    """
    if shutil.which("Rscript") is None:
        return False
    check = (
        "suppressMessages({library(harpIO); library(harpPoint)});"
        "stopifnot(exists('read_fcparquet', envir = asNamespace('harpIO')))"
    )
    result = subprocess.run(["Rscript", "-e", check], capture_output=True)
    return result.returncode == 0


requires_harp = pytest.mark.skipif(
    not _have_harp(), reason="R with harpIO (fcparquet support) and harpPoint needed"
)


def run_r(script: str) -> str:
    """Run an R script and return its stdout.

    Parameters
    ----------
    script : str
        R code to run.

    Returns
    -------
    str
        Standard output of the script.

    Raises
    ------
    AssertionError
        If the script exits with a non-zero status (stderr is included).
    """
    result = subprocess.run(
        ["Rscript", "-e", textwrap.dedent(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@requires_harp
@pytest.mark.parametrize("ensemble", [False, True])
def test_harp_reads_and_verifies(tmp_path, ds_grid_fcst, ds_grid_ens, ds_obs, ensemble):
    """harpIO reads the forecasts and obs, and harpPoint can verify them."""
    ds_fcst = ds_grid_ens if ensemble else ds_grid_fcst
    aligned = align_with_mxalign({"model": ds_fcst}, ds_obs, "xarray")
    write_harp_parquets(aligned, ds_obs, tmp_path)
    verify = "ens_verify" if ensemble else "det_verify"
    out = run_r(
        f"""
        suppressMessages({{library(harpIO); library(harpPoint)}})
        fc <- read_point_forecast(
          dttm = seq_dttm(2026010100, 2026010112, "12h"),
          fcst_model = "model", fcst_type = "{'eps' if ensemble else 'det'}",
          parameter = "T2m", lead_time = seq(0, 12, 6),
          file_path = "{tmp_path / FCPARQUET_DIR}",
          file_format = "fcparquet", file_template = "{{fcst_model}}/{{parameter}}"
        )
        obs <- read_point_obs(
          dttm = unique_valid_dttm(fc), parameter = "T2m",
          obs_path = "{tmp_path / OBSPARQUET_DIR}", file_format = "obsparquet"
        )
        fc <- join_to_fcst(fc, obs)
        stopifnot(nrow(fc[[1]]) == 24)
        v <- {verify}(fc, T2m)
        cat("OK", nrow(fc[[1]]), "\\n")
        """
    )
    assert "OK 24" in out
