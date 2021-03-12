# install.packages("rhub")
library("rhub")
validate_email()

# First run 'sh build-cran-package.sh'
# Set working directory to gpboost_r folder

mycheck <- check(env_vars=c(R_COMPILE_AND_INSTALL_PACKAGES = "always"))
mycheck$cran_summary()

cran_prep <- check_for_cran(env_vars=c(R_COMPILE_AND_INSTALL_PACKAGES = "always"))
cran_prep$cran_summary()
