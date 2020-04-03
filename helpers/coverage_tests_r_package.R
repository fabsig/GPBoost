# Set working directory to GPBoost main folder

system("Rscript build_r.R")

library(covr)
coverage  <- covr::package_coverage('./gpboost_r', quiet=FALSE)
print(coverage)
covr::report(coverage, browse = TRUE)
