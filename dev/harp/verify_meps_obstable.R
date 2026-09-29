# Read and verify the HARP parquet datasets written by mlwp-harp-writer's
# end-to-end test (tests/test_e2e_meps_obstable.py) with harp.
#
# Usage: Rscript verify_meps_obstable.R <harp output dir>
#   where <harp output dir> contains FCPARQUET/ and OBSPARQUET/
#
# The test writes MEPS forecasts for the 2019-02-17 00 and 12 UTC cycles at
# lead times 0, 6, 12 and 24 h, and harpData's OBSTABLE_2019 observations.
#
# The scores are printed and saved with harpPoint::save_point_verif() to
# <harp output dir>/verification, for harpVis' web app (shiny_app.R).

suppressPackageStartupMessages({
  library(harpIO)
  library(harpPoint)
})

args <- commandArgs(trailingOnly = TRUE)
harp_dir <- if (length(args) > 0) args[[1]] else "/data"
verif_dir <- file.path(harp_dir, "verification")
dir.create(verif_dir, showWarnings = FALSE)

for (param in c("T2m", "RH2m", "Pmsl", "Ps", "CCtot")) {
  cat("\n=====", param, "=====\n")
  fc <- read_point_forecast(
    dttm          = seq_dttm(2019021700, 2019021712, "12h"),
    fcst_model    = "MEPS",
    parameter     = param,
    lead_time     = c(0, 6, 12, 24),
    file_path     = file.path(harp_dir, "FCPARQUET"),
    file_format   = "fcparquet",
    file_template = "{fcst_model}/{parameter}"
  )
  obs <- read_point_obs(
    dttm        = unique_valid_dttm(fc),
    parameter   = param,
    file_path   = file.path(harp_dir, "OBSPARQUET"),
    file_format = "obsparquet"
  )
  fc <- join_to_fcst(fc, obs)
  verif <- det_verify(fc, {{param}})
  print(verif)
  save_point_verif(verif, verif_path = verif_dir)
}
cat("\nVerification results saved to", verif_dir, "\n")
