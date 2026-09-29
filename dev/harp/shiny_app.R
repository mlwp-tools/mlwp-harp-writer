# Start harpVis' point verification web app on the HARP parquet datasets
# written by mlwp-harp-writer's end-to-end test.
#
# Usage: Rscript shiny_app.R [<harp output dir>] [<port>]
#   defaults: /data, port 3838
#
# The app shows verification results saved by harpPoint::save_point_verif().
# If <harp output dir>/verification has none yet, verify_meps_obstable.R is
# run first to compute them from the FCPARQUET/OBSPARQUET datasets.
#
# The app listens on all interfaces so it can be reached from outside a
# container: publish the port (docker run -p 3838:3838 ...) and open
# http://localhost:3838 in a browser.

args <- commandArgs(trailingOnly = TRUE)
harp_dir <- if (length(args) > 0) args[[1]] else "/data"
port <- if (length(args) > 1) as.integer(args[[2]]) else 3838L
verif_dir <- file.path(harp_dir, "verification")

if (length(list.files(verif_dir, pattern = "\\.rds$", recursive = TRUE)) == 0) {
  message("No verification results in ", verif_dir, ", computing them first")
  verify_script <- file.path(dirname(sub("^--file=", "", grep(
    "^--file=", commandArgs(), value = TRUE
  ))), "verify_meps_obstable.R")
  system2("Rscript", c(verify_script, harp_dir))
}

message("harpVis point verification app on http://localhost:", port)
harpVis::shiny_plot_point_verif(
  start_dir      = verif_dir,
  host           = "0.0.0.0",
  port           = port,
  launch.browser = FALSE
)
