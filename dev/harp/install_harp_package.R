# Install an R package from a GitHub commit tarball, without the GitHub API.
#
# Usage: Rscript install_harp_package.R <owner/repo> <commit sha>
#
# Downloads https://github.com/<owner/repo>/archive/<sha>.tar.gz (a plain
# download, not subject to the GitHub API rate limit), installs the CRAN
# dependencies from its DESCRIPTION that are missing (as binaries, via r2u),
# and installs the package from source. GitHub-hosted dependencies (the
# DESCRIPTION's Remotes) are not resolved: install them first, in order.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript install_harp_package.R <owner/repo> <commit sha>", call. = FALSE)
}
repo <- args[[1]]
sha <- args[[2]]

url <- sprintf("https://github.com/%s/archive/%s.tar.gz", repo, sha)
tarball <- tempfile(fileext = ".tar.gz")
download.file(url, tarball, quiet = TRUE)
src_dir <- tempfile()
untar(tarball, exdir = src_dir)
pkg_dir <- list.dirs(src_dir, recursive = FALSE)[[1]]

# dependency package names from Depends/Imports/LinkingTo, without versions
desc <- read.dcf(file.path(pkg_dir, "DESCRIPTION"))
fields <- intersect(c("Depends", "Imports", "LinkingTo"), colnames(desc))
deps <- unlist(strsplit(desc[, fields], ","))
deps <- trimws(sub("\\(.*\\)", "", deps))
deps <- setdiff(deps[nzchar(deps)], c("R", rownames(installed.packages())))
if (length(deps) > 0) {
  message("Installing CRAN dependencies: ", paste(deps, collapse = ", "))
  install.packages(deps)
}

install.packages(pkg_dir, repos = NULL, type = "source")
pkg <- desc[, "Package"]
if (!requireNamespace(pkg, quietly = TRUE)) {
  stop("Failed to install ", pkg, " from ", url, call. = FALSE)
}
message("Installed ", pkg, " ", packageVersion(pkg), " from ", repo, "@", sha)
