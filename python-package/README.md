<img src="https://github.com/fabsig/GPBoost/blob/master/docs/logo/gpboost_logo.png?raw=true"
     alt="GPBoost icon"
     align = "right"
     width="30%" />

# GPBoost Python Package

[![License](https://img.shields.io/badge/Licence-Apache%202.0-green.svg)](https://github.com/fabsig/GPBoost/blob/master/LICENSE)
[<img src="https://img.shields.io/pypi/pyversions/gpboost.svg?logo=python&logoColor=white">](https://pypi.org/project/gpboost)
[<img src="https://img.shields.io/pypi/v/gpboost.svg?logo=pypi&logoColor=white">](https://pypi.org/project/gpboost)
[<img src="https://static.pepy.tech/badge/gpboost">](https://pepy.tech/project/gpboost)

This is the Python package implementation of the GPBoost library. See https://github.com/fabsig/GPBoost for more information on the modeling background and the software implementation.

### Table of Contents
* [Examples and documentation](#examples-and-documentation)
* [Installation](#installation)
  * [Installation from PyPI](#installation-from-pypi-using-precompiled-binaries)
  * [Installation from source](#installation-from-source)


## Examples and documentation

* [**Python examples**](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide)
  * [GPBoost / LaGaBoost algorithm](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/GPBoost_algorithm.py) for Gaussian ("regression") and non-Gaussian likelihoods (e.g., "classification", etc.)
  * [Generalized linear Gaussian process and mixed effects models](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/generalized_linear_Gaussian_process_mixed_effects_models.py)
* The **documentation at [https://gpboost.readthedocs.io](https://gpboost.readthedocs.io/en/latest/Python_package.html)**

## Installation

#### Before you install

* [setuptools](https://pypi.org/project/setuptools) is needed. You can install this using ``pip install setuptools -U``

* 32-bit Python is not supported. Please install the 64-bit version. See [build 32-bit version with 32-bit Python section](#build-32-bit-version-with-32-bit-python).

### Installation from [PyPI](https://pypi.org/project/gpboost) using precompiled binaries

```sh
pip install gpboost -U
```

#### Requirements

* For **Windows** users, [VC runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) is needed if **Visual Studio** (2015 or newer) is not installed.

* For **Linux** users, **glibc** >= 2.14 is required. 

  * If you get an error message ``version `GLIBC_2.27' not found``, you need to [install from source](#installation-from-source).
  
  * In rare cases, when you get the ``OSError: libgomp.so.1: cannot open shared object file: No such file or directory`` error when importing GPBoost, you need to install the OpenMP runtime library separately (use your package manager and search for ``lib[g|i]omp`` for doing this).

* For **macOS** users:

  * The library file in distribution wheels is built by the **Apple Clang** compiler. You need to install the **OpenMP** library. You can install the **OpenMP** library by the following command: ``brew install libomp``. <!-- (Xcode version 12.3 is used starting from GPBoost version 0.3.0) -->
  
  * If you have an **arm64 Apple silicon** processor (e.g., M1 or M2) and experience problems, try the following steps:
    
    * [uninstall homebrew](https://stackoverflow.com/questions/72890277/i-cant-uninstall-brew-on-macos-apple-silicon) (in case you have migrated from an older non-arm64 Mac)
    * [install homebrew](https://treehouse.github.io/installation-guides/mac/homebrew) (to make sure that you have an arm64 version of libomp)
    * install OpenMP (``brew install libomp``)
    * remove existing python environments and install Miniforge (``brew install miniforge`` and ``conda init "$(basename "${SHELL}")"``)

### Installation from source

Installation from source can be either done from PyPI or GitHub.

#### Requirements for installation from source

* Installation from source requires that you have installed [**CMake**](https://cmake.org/).

* For **Linux** users, **glibc** >= 2.14 is required. 

  * In rare cases, you may need to install the OpenMP runtime library separately (use your package manager and search for ``lib[g|i]omp`` for doing this).

* For **macOS** users, you can perform installation either with **Apple Clang** or **gcc**.

  * In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in the [Installation Guide](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst#apple-clang)) first and **CMake** version 3.16 or higher is required. Only Apple Clang version 8.1 or higher is supported.

  * In case you prefer **gcc**, you need to install it (details for installation can be found in the [Installation Guide](https://github.com/fabsig/GPBoost/blob/master/docs/Installation_guide.rst#gcc)) and specify compilers by running ``export CXX=g++-7 CC=gcc-7`` (replace "7" with the version of **gcc** installed on your machine) first.

* For **Windows** users, **Visual Studio** (or [VS Build Tools](https://visualstudio.microsoft.com/downloads/)) is needed. 


#### Installation from source from PyPI

```sh
pip install --no-binary :all: gpboost
```

##### Build with MinGW-w64 on Windows

```sh
pip install gpboost --install-option=--mingw
```

* [CMake](https://cmake.org/) and [MinGW-w64](https://www.mingw-w64.org/) should be installed first.


##### Build 32-bit version with 32-bit Python

```sh
pip install gpboost --install-option=--bit32
```

By default, installation in an environment with 32-bit Python is prohibited. However, you can remove this prohibition on your own risk by passing the ``bit32`` option (not recommended).


#### Installation from source from GitHub

```sh
git clone --recursive https://github.com/fabsig/GPBoost.git
cd GPBoost/python-package
# export CXX=g++-7 CC=gcc-7  # macOS users, if you decided to compile with gcc, don't forget to specify compilers (replace "7" with version of gcc installed on your machine)
python setup.py install
```

Note: ``sudo`` (or administrator rights in **Windows**) may be needed to perform the command.

##### Build with MinGW-w64 on Windows

```sh
python setup.py install --mingw
```
* [CMake](https://cmake.org/) and [MinGW-w64](https://www.mingw-w64.org/) should be installed first.

If you get any errors during installation or due to any other reasons, you may want to build a dynamic library from source by any method you prefer and then just run ``python setup.py install --precompile``.
