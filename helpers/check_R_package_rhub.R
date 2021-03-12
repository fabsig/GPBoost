# install.packages("rhub")
library("rhub")
# validate_email()

# First run 'sh build-cran-package.sh'
# Set working directory to gpboost_r folder

mycheck <- check()
mycheck$cran_summary()

cran_prep <- check_for_cran()
cran_prep$cran_summary()
