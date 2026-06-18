# tools::package_native_routine_registration_skeleton('.',character_only = FALSE)

# Comment: need to use old version of roxygen2 (<=6.0.1), since newer versions produces error 
# when trying to compile, but we don't need that. It should simply create Rd files.
# (Error in getDLLRegisteredRoutines.DLLInfo(dll, addNames = FALSE : must specify DLL via a "DLLInfo" object. See getLoadedDLLs())
# (See https://github.com/r-lib/roxygen2/issues/822 and https://github.com/r-lib/roxygen2/issues/771)

###########
#  NEED TO RUN R VERSION 3.3.2 (to run this, since an old version Rcpp runs only on older R version)
###########

install.packages("https://cran.r-project.org/src/contrib/Archive/Rcpp/Rcpp_0.12.9.tar.gz", repos = NULL, type = "source")
install.packages("https://cran.r-project.org/src/contrib/Archive/BH/BH_1.60.0-2.tar.gz", repos = NULL, type = "source")
install.packages("https://cran.r-project.org/src/contrib/Archive/xml2/xml2_1.0.0.tar.gz", repos = NULL, type = "source")
install.packages("https://cran.r-project.org/src/contrib/Archive/assertthat/assertthat_0.2.0.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/crayon/crayon_1.3.4.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/cli/cli_1.0.0.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/magrittr/magrittr_1.5.tar.gz", repos = NULL, type = "source")
install.packages("https://cran.r-project.org/src/contrib/Archive/stringi/stringi_1.1.2.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/stringr/stringr_1.1.0.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/R6/R6_2.2.0.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/desc/desc_1.0.0.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/glue/glue_1.0.0.tar.gz", repos = NULL, type = "source")
install.packages("https://cran.r-project.org/src/contrib/Archive/rlang/rlang_0.1.1.tar.gz", repos = NULL, type = "source")
install.packages("https://cran.r-project.org/src/contrib/Archive/backports/backports_1.1.4.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/digest/digest_0.6.20.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/zeallot/zeallot_0.1.0.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/vctrs/vctrs_0.1.0.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/brew/brew_1.0-6.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/commonmark/commonmark_1.2.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages("https://cran.r-project.org/src/contrib/Archive/roxygen2/roxygen2_6.0.1.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")
install.packages(c("data.table"))
install.packages("https://cran.r-project.org/src/contrib/Archive/data.table/data.table_1.10.4-3.tar.gz", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")


## OLD VERSION
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
# install.packages(c("data.table","RJSONIO")) # not required, but done to avoid warnings

setwd("R-package")

library(roxygen2)
roxygen2::roxygenize() ## roclets="rd"





