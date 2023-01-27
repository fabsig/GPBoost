# tools::package_native_routine_registration_skeleton('.',character_only = FALSE)

# Comment: need to use old version of roxygen2 (<=6.0.1), since newer versions produces error 
# when trying to compile, but we don't need that. It should simply create Rd files.
# (Error in getDLLRegisteredRoutines.DLLInfo(dll, addNames = FALSE : must specify DLL via a "DLLInfo" object. See getLoadedDLLs())
# (See https://github.com/r-lib/roxygen2/issues/822 and https://github.com/r-lib/roxygen2/issues/771)

###########
#  NEED TO RUN R VERSION 3.3.2 (to run this, since an old version Rcpp runs only on older R version)
###########

# install.packages("remotes")
# remotes::install_version("Rcpp", "0.12.9") ## Need old version of Rcpp for old version of roxygen2
# remotes::install_version("xml2", "1.0.0")
# remotes::install_version("cli", "1.0.0")
# remotes::install_version("magrittr", "1.5")
# remotes::install_version("stringr", "1.1.0")
# remotes::install_version("desc", "1.0.0")
# remotes::install_version("glue", "1.0.0")
# remotes::install_version("rlang", "0.1.1")
# remotes::install_version("vctrs", "0.1.0")
# remotes::install_version("roxygen2", "6.0.1")
## Need old versions of the above packages for old version of Rcpp and roxygen2


setwd("R-package")

library(roxygen2)
roxygen2::roxygenize() ## roclets="rd"





