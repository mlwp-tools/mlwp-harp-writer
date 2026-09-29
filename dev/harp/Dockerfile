# R with harp (harpCore, harpIO, harpPoint, harpVis) for checking the HARP
# parquet datasets written by mlwp-harp-writer, and for browsing the
# verification scores in harpVis' web app. See DEVELOPING.md.
#
# Build:  docker build -t mlwp-harp -f dev/harp/Dockerfile dev/harp
# Web UI: docker run --rm -p 3838:3838 -v <harp output dir>:/data mlwp-harp
#         then open http://localhost:3838
# Verify only (print scores):
#         docker run --rm -v <harp output dir>:/data mlwp-harp /scripts/verify_meps_obstable.R /data

# r2u (https://github.com/eddelbuettel/r2u) installs CRAN packages as prebuilt
# Ubuntu binaries via apt (amd64 and arm64), also through install.packages().
# Compiling them instead (arrow in particular) takes an hour or more.
FROM rocker/r2u:noble

# arrow provides the parquet support harpIO's fcparquet/obsparquet readers use.
# apt-get update and install stay in one layer, otherwise a cached update
# layer goes stale.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libproj-dev r-cran-arrow \
    && rm -rf /var/lib/apt/lists/*
RUN R -q -e 'stopifnot(arrow::arrow_info()$capabilities[["parquet"]])'

# harp isn't on CRAN or R-universe, and its releases (v0.2.x, 2024) predate
# harpIO's parquet support. So each package is installed from a pinned commit
# tarball (a plain download, no GitHub API rate limit), in dependency order,
# one layer each so Docker's cache only reinstalls from the first changed pin.
# CRAN dependencies are installed as r2u binaries.
COPY install_harp_package.R /usr/local/bin/install_harp_package.R
RUN Rscript /usr/local/bin/install_harp_package.R harphub/meteogrid 8e4899f51c60
RUN Rscript /usr/local/bin/install_harp_package.R harphub/harpCore 44b2590c0745
# harpIO with fcparquet/obsparquet support (added in 74b379b, 2025-09)
RUN Rscript /usr/local/bin/install_harp_package.R harphub/harpIO b058d24ef38a
RUN R -q -e 'stopifnot(exists("read_fcparquet", envir = asNamespace("harpIO")))'
RUN Rscript /usr/local/bin/install_harp_package.R harphub/harpPoint 84c18b674c62
RUN Rscript /usr/local/bin/install_harp_package.R harphub/ggsonde dd3883d67ae1
RUN Rscript /usr/local/bin/install_harp_package.R harphub/harpVis bc5a787ee5cb

# scripts last, so editing them doesn't invalidate the package layers
COPY verify_meps_obstable.R shiny_app.R /scripts/

# harpVis' point verification web app
EXPOSE 3838

WORKDIR /data
ENTRYPOINT ["Rscript"]
CMD ["/scripts/shiny_app.R", "/data"]
