<img src="https://github.com/fabsig/GPBoost/blob/master/gpboost_sticker.jpg?raw=true"
     alt="GPBoost icon"
     align = "right"
     width="40%" />

GPBoost R Package
==================
     
This is the R package implementation of the GPBoost library. See https://github.com/fabsig/GPBoost for more information on the modeling background and the software implementation.

Examples
--------

* [**GPBoost R and Python demo**](https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html) illustrates how GPBoost can be used in R and Python

* [**More examples**](https://github.com/fabsig/GPBoost/tree/master/R-package/demo):
  * [Gaussian process and other mixed effects model examples (without boosting)](https://github.com/fabsig/GPBoost/blob/master/R-package/demo/Gaussian_process_mixed_effects_models.R)
  * [Boosting functionality (without Gaussian process / random effects)](https://github.com/fabsig/GPBoost/blob/master/R-package/demo/boosting.R)
  * [Combining tree-boosting with Gaussian process and random effects models](https://github.com/fabsig/GPBoost/blob/master/R-package/demo/combined_boosting_GP_random_effects.R)
  * [Cross Validation](https://github.com/fabsig/GPBoost/blob/master/R-package/demo/cross_validation.R)

Installation
------------

The GPBoost package is (hopefully) soon on CRAN. In the meantime, you can follow these instructions for installation. 

In short, the **main steps** for installation are the following ones:

* Install [**git**](https://git-scm.com/downloads)
* Install [**CMake**](https://cmake.org/)
* Install [**Rtools**](https://cran.r-project.org/bin/windows/Rtools/) (**for Windows only**). Choose the option 'add rtools to system PATH'.
* Make sure that you have an appropriate **C++ compiler** (see below for more details). E.g. for Windows, simply download the free [Visual Studio Community Edition](https://visualstudio.microsoft.com/downloads/) and do not forget to select 'Desktop development with C++' when installing it
* **Install the GPBoost package** from the command line using
  * Opening a command line interface: type 'cmd' in the Search or Run line or type 'powershell' in the File Explorer in the address bar
```sh
git clone --recursive https://github.com/fabsig/GPBoost
cd GPBoost
Rscript build_r.R
```

Below is a more complete installation guide.

### Preparation

You need to install [git](https://git-scm.com/downloads) and [CMake](https://cmake.org/) first.

Note: 32-bit R/Rtools is not supported.

#### Windows Preparation

Installing [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is mandatory, and only support the 64-bit version. It requires to add to PATH the Rtools MinGW64 folder, if it was not done automatically during installation.

The default compiler is Visual Studio (or [VS Build Tools](https://visualstudio.microsoft.com/downloads/)) in Windows, with an automatic fallback to Rtools or any [MinGW64](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/) (x86_64-posix-seh) available (this means if you have only Rtools and CMake, it will compile fine). To force the usage of Rtools / MinGW, you can set `use_mingw` to `TRUE` in `R-package/src/install.libs.R`. It is recommended to use *Visual Studio* for its better multi-threading efficiency in Windows.

#### Mac OS Preparation

You can perform installation either with **Apple Clang** or **gcc**. In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in the [Installation Guide](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst#apple-clang)) first and **CMake** version 3.12 or higher is required. In case you prefer **gcc**, you need to install it (details for installation can be found in the [Installation Guide](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst#gcc)) and set some environment variables to tell R to use `gcc` and `g++`. If you install these from Homebrew, your versions of `g++` and `gcc` are most likely in `/usr/local/bin`, as shown below.

```
# replace 8 with version of gcc installed on your machine
export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8
```

### Install

Build and install R-package with the following commands:

```sh
git clone --recursive https://github.com/fabsig/GPBoost
cd GPBoost
Rscript build_r.R
```

The `build_r.R` script builds the package in a temporary directory called `gpboost_r`. It will destroy and recreate that directory each time you run the script.

Note: for the build with Visual Studio/VS Build Tools in Windows, you should use the Windows CMD or Powershell.

Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package). Linux users might require the appropriate user write permissions for packages.

When your package installation is done, you can check quickly if the GPBoost R-package is working by running the following:

```r
#--------------------Zero-mean single-level grouped random effects model (without a fixed effect)----------------
library(gpboost)
n <- 100 # number of samples
m <- 25 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
# Simulate data
sigma2_1 <- 1^2 # random effect variance
sigma2 <- 0.5^2 # error variance
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1) # incidence matrix relating grouped random effects to samples
set.seed(1)
b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
y <- Z1 %*% b1 + sqrt(sigma2) * rnorm(n) # observed data
# Fit model directly using fit.GPModel
gp_model <- fit.GPModel(group_data = group, y = y, std_dev = TRUE)
summary(gp_model)
```
