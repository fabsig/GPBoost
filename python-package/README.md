<img src="https://github.com/fabsig/GPBoost/blob/master/gpboost_sticker.jpg?raw=true"
     alt="GPBoost icon"
     align = "right"
     width="40%" />

GPBoost Python Package
=======================

[![license](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg)](https://github.com/fabsig/GPBoost/blob/master/LICENSE)
[<img src="https://img.shields.io/pypi/pyversions/gpboost.svg?logo=python&logoColor=white">](https://pypi.org/project/gpboost)
[<img src="https://img.shields.io/pypi/v/gpboost.svg?logo=pypi&logoColor=white">](https://pypi.org/project/gpboost)
[<img src="https://pepy.tech/badge/gpboost">](https://pepy.tech/project/gpboost)

This is the Python package implementation of the GPBoost library. See https://github.com/fabsig/GPBoost for more information on the modeling background and the software implementation.

Examples
--------

- [**GPBoost R and Python demo**](https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html) illustrates how GPBoost can be used in R and Python
- [**Blog post**](https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb) and [corresponding code](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/GPBoost_algorithm_blog_post_example.py) on how to combine tree-boosting with mixed effects models
- [**Detailed Python examples**](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide):
  * [GPBoost algorithm](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/GPBoost_algorithm.py) for combining tree-boosting with Gaussian process and random effects models
  * [GPBoost algorithm for binary classification and other non-Gaussian data](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/classification_non_Gaussian_data.py) (Poisson regression, etc.)
  * [Cross validation](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/cross_validation.py) for parameter tuning
  * [Linear Gaussian process and mixed effects model examples](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/linear_Gaussian_process_mixed_effects_models.py)
  * [Generalized linear Gaussian process and mixed effects model examples](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/generalized_linear_Gaussian_process_mixed_effects_models.py)
  * [Standard boosting functionality (without Gaussian process or random  effects)](https://github.com/fabsig/GPBoost/tree/master/examples/python-guide/boosting.py)

Installation
------------

### Before you install

[setuptools](https://pypi.org/project/setuptools) is needed. You can install this using ``pip install setuptools -U``

32-bit Python is not supported. Please install the 64-bit version. See [build 32-bit version with 32-bit Python section](#build-32-bit-version-with-32-bit-python).

### Install from [PyPI](https://pypi.org/project/gpboost) using ``pip``

In brief, run:

```sh
pip install gpboost -U
```

Below is a more detailed installation guide.

#### Install using precompiled Python wheel (.whl) file

Install [wheel](https://pythonwheels.com) via ``pip install wheel`` first. After that download the wheel file from [whlFiles](https://pypi.org/project/gpboost/#files) and install from the folder where you downloaded the .whl file using:

```sh
pip install gpboost-XXX.whl
```

##### Requirements

- For **Windows** users, [VC runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) is needed if **Visual Studio** (2015 or newer) is not installed.

- For **Linux** users, **glibc** >= 2.14 is required.

- For **macOS** users:

  - The library file in distribution wheels is built by the **Apple Clang** (Xcode_8.3.3 for versions 2.2.1 - 2.3.1, and Xcode_9.4.1 from version 2.3.2) compiler. You need to install the **OpenMP** library. You can install the **OpenMP** library by the following command: ``brew install libomp``.

#### Build from source

```sh
pip install --no-binary :all: gpboost
```

##### Requirements for building from sources

- **Installation from sources requires that you have installed** [CMake](https://cmake.org/).

- For **macOS** users, you can perform installation either with **Apple Clang** or **gcc**.

  - In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in the [Installation Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#apple-clang)) first and **CMake** version 3.16 or higher is required.

  - In case you prefer **gcc**, you need to install it (details for installation can be found in the [Installation Guide](https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc)) and specify compilers by running ``export CXX=g++-7 CC=gcc-7`` (replace "7" with version of **gcc** installed on your machine) first.

- For **Windows** users, **Visual Studio** (or [VS Build Tools](https://visualstudio.microsoft.com/downloads/)) is needed.

##### Build with MinGW-w64 on Windows

```sh
pip install gpboost --install-option=--mingw
```

[CMake](https://cmake.org/) and [MinGW-w64](https://mingw-w64.org/) should be installed first.

It is recommended to use **Visual Studio** for its better multithreading efficiency in **Windows** for many-core systems

##### Build 32-bit version with 32-bit Python

```sh
pip install gpboost --install-option=--bit32
```

By default, installation in environment with 32-bit Python is prohibited. However, you can remove this prohibition on your own risk by passing the ``bit32`` option (**not recommended**).


### Install from GitHub

```sh
git clone --recursive https://github.com/fabsig/GPBoost.git
cd GPBoost/python-package
# export CXX=g++-7 CC=gcc-7  # macOS users, if you decided to compile with gcc, don't forget to specify compilers (replace "7" with version of gcc installed on your machine)
python setup.py install
```

Note: ``sudo`` (or administrator rights in **Windows**) may be needed to perform the command.

If you get any errors during installation or due to any other reasons, you may want to build dynamic library from sources by any method you prefer and then just run ``python setup.py install --precompile``.
