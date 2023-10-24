<img src="https://github.com/fabsig/GPBoost/blob/master/docs/logo/gpboost_logo.png?raw=true"
     alt="GPBoost icon"
     align = "right"
     width="30%" />

# GPBoost R Package

[![License](https://img.shields.io/badge/Licence-Apache%202.0-green.svg)](https://github.com/fabsig/GPBoost/blob/master/LICENSE)
[![CRAN Version](https://www.r-pkg.org/badges/version/gpboost)](https://cran.r-project.org/package=gpboost)
[![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/gpboost)](https://cran.r-project.org/package=gpboost)

This is the R package implementation of the GPBoost library. See https://github.com/fabsig/GPBoost for more information on the modeling background and the software implementation.

### Table of Contents
* [Examples](#examples)
* [Installation](#installation)
  * [Installation from CRAN](#installation-from-cran)
  * [Installation from source](#installation-from-source)
* [Testing](#testing)
* [Preparing a CRAN package](#preparing-a-cran-package)

## Examples

* [**R examples**](https://github.com/fabsig/GPBoost/tree/master/R-package/demo)
  * [GPBoost / LaGaBoost algorithm](https://github.com/fabsig/GPBoost/blob/master/R-package/demo/GPBoost_algorithm.R) for Gaussian ("regression") and non-Gaussian likelihoods (e.g., "classification", etc.)
  * [Generalized linear Gaussian process and mixed effects models](https://github.com/fabsig/GPBoost/blob/master/R-package/demo/generalized_linear_Gaussian_process_mixed_effects_models.R)

## Installation

### Installation from CRAN

The `gpboost` package is [available on CRAN](https://cran.r-project.org/package=gpboost) and can be installed as follows:

```r
install.packages("gpboost", repos = "https://cran.r-project.org")
```

### Installation from source

It is much easier to install the package from CRAN. However, the package can also be build from source as described in the following. In short, the **main steps** for installation are the following ones:

* Install [**git**](https://git-scm.com/downloads)
* Install [**CMake**](https://cmake.org/)
* Install [**Rtools**](https://cran.r-project.org/bin/windows/Rtools/) (**for Windows only**). Choose the option 'add rtools to system PATH'.
* Make sure that you have an appropriate **C++ compiler** (see below for more details). E.g. for Windows, simply download the free [Visual Studio Community Edition](https://visualstudio.microsoft.com/downloads/) and do not forget to select 'Desktop development with C++' when installing it
* **Install the GPBoost package** from the command line using:
```sh
git clone --recursive https://github.com/fabsig/GPBoost
cd GPBoost
Rscript build_r.R
```

Below is a more complete installation guide.

### Preparation

You need to install [git](https://git-scm.com/downloads) and [CMake](https://cmake.org/) first. Note that 32-bit R/Rtools is not supported for custom installation.

#### Windows Preparation

NOTE: Windows users may need to run with administrator rights (either R or the command prompt, depending on the way you are installing this package).

Installing a 64-bit version of [Rtools](https://cran.r-project.org/bin/windows/Rtools/) is mandatory.

After installing `Rtools` and `CMake`, be sure the following paths are added to the environment variable `PATH`. These may have been automatically added when installing other software.

* `Rtools`
    - If you have `Rtools` 3.x, for example:
        - `C:\Rtools\mingw_64\bin`
    - If you have `Rtools` 4.0, for example:
        - `C:\rtools40\mingw64\bin`
        - `C:\rtools40\usr\bin`
    - If you have Rtools 4.2+, for example:
        - `C:\rtools42\x86_64-w64-mingw32.static.posix\bin`
        - `C:\rtools42\usr\bin`
        - NOTE: this is, e.g., `rtools43\` for R 4.3
   - NOTE: 
      - Two `Rtools` paths are required from `Rtools` 4.0 onwards
      - When installing in R with `install.packages()`, these paths can be added locally in R, e.g., as follows prior to installation: `Sys.setenv(PATH=paste0(Sys.getenv("PATH"),";C:\\Rtools\\mingw_64\\bin\\;C:\\rtools40\\usr\\bin\\"))` 

* `CMake`
    - example: `C:\Program Files\CMake\bin`
* `R`
    - example: `C:\Program Files\R\R-3.6.1\bin`


The default compiler is Visual Studio (or [VS Build Tools](https://visualstudio.microsoft.com/downloads/)) in Windows, with an automatic fallback to MingGW64 (i.e. it is enough to only have Rtools and CMake). To force the usage of MinGW64, you can add the `--use-mingw` (for R 3.x) or `--use-msys2` (for R 4.x) flags (see below).

#### Mac OS Preparation

You can perform installation either with **Apple Clang** or **gcc**. 

* In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in the [Installation Guide](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst#apple-clang)) first and **CMake** version 3.12 or higher is required. Only Apple Clang version 8.1 or higher is supported. 
* In case you prefer **gcc**, you need to install it (details for installation can be found in the [Installation Guide](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst#gcc)) and set some environment variables to tell R to use `gcc` and `g++`. If you install these from Homebrew, your versions of `g++` and `gcc` are most likely in `/usr/local/bin`, as shown below.

```
# replace 8 with version of gcc installed on your machine
export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8
```

### Install

Build and install the R package with the following commands:

```sh
git clone --recursive https://github.com/fabsig/GPBoost
cd GPBoost
Rscript build_r.R
```


The `build_r.R` script builds the package in a temporary directory called `gpboost_r`. It will destroy and recreate that directory each time you run the script. That script supports the following command-line options:

- `--skip-install`: Build the package tarball, but do not install it
- `--use-mingw`: Force the use of MinGW toolchain, regardless of R version
- `--use-msys2`: Force the use of MSYS2 toolchain, regardless of R version
<!-- - `--use-gpu`: Build a GPU-enabled version of the library -->

Note: for the build with Visual Studio/VS Build Tools in Windows, you should use the Windows CMD or PowerShell.

## Testing

There is currently no integration service set up that automatically runs unit tests. However, any contribution needs to pass all unit tests in the `R-package/tests/testthat` directory. These tests can be run using the [run_tests_coverage_R_package.R](https://github.com/fabsig/GPBoost/blob/master/helpers/run_tests_coverage_R_package.R) file. In any case, make sure that you run the full set of tests by speciying the following environment variable
```R
Sys.setenv(GPBOOST_ALL_TESTS = "GPBOOST_ALL_TESTS")
```
before running the tests in the `R-package/tests/testthat` directory.

## Preparing a CRAN package

This section is primarily for maintainers, but may help users and contributors to understand the structure of the R package. Most of `GPBoost` uses `CMake` to handle tasks like setting compiler and linker flags, including header file locations, and linking to other libraries. Because CRAN packages typically do not assume the presence of `CMake`, the R package uses an alternative method that is in the CRAN-supported toolchain for building R packages with C++ code: `Autoconf`. For more information on this approach, see ["Writing R Extensions"](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Configure-and-cleanup).

### Build a CRAN Package

From the root of the repository, run the following:
```shell
sh build-cran-package.sh
```

This will create a file `gpboost_${VERSION}.tar.gz`, where `VERSION` is the version of `GPBoost`.

### Installation of a CRAN package

After building the package, you can install it with the following command:

```shell
R CMD INSTALL gpboost_*.tar.gz
```

### Testing a CRAN package

After building the package, you can test the CRAN package as follows:
```shell
R CMD check --as-cran gpboost_*.tar.gz
```
