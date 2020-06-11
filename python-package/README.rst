.. image:: https://github.com/fabsig/GPBoost/blob/master/gpboost_sticker.jpg?raw=true
     :alt: GPBoost icon
     :align: right
     :width: 40 %

GPBoost Python Package
=======================

|License| |Python Versions| |PyPI Version| |Downloads|

This is the Python package implementation of the GPBoost library. See https://github.com/fabsig/GPBoost for more information on the modeling background and the software implementation.

Examples
--------

- `GPBoost R and Python demo <https://htmlpreview.github.io/?https://github.com/fabsig/GPBoost/blob/master/examples/GPBoost_demo.html>`_ illustrates how GPBoost can be used in R and Python
- More examples in the `Python guide folder <https://github.com/fabsig/GPBoost/tree/master/examples/python-guide>`_


Installation
------------

Before you install
'''''''''''''''''''

`setuptools <https://pypi.org/project/setuptools>`_ is needed. You can install this using ``pip install setuptools -U``

32-bit Python is not supported. Please install the 64-bit version. See `build 32-bit Version with 32-bit Python section <#build-32-bit-version-with-32-bit-python>`__.

Install from `PyPI <https://pypi.org/project/gpboost>`_ Using ``pip``
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

In brief, run:

.. code:: sh

    pip install gpboost -U

Below is a more detailed installation guide.

Install using precompiled Python wheel (.whl) file
******************************************************

Install `wheel <https://pythonwheels.com>`_ via ``pip install wheel`` first. After that download the wheel file from `whlFiles`_ and install from the folder where you downloaded the .whl file using:

.. code:: sh

    pip install gpboost-XXX.whl

Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- For **Windows** users, `VC runtime <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ is needed if **Visual Studio** (2015 or newer) is not installed.

- For **Linux** users, **glibc** >= 2.14 is required.

- For **macOS** users:

  - The library file in distribution wheels is built by the **Apple Clang** (Xcode_8.3.3 for versions 2.2.1 - 2.3.1, and Xcode_9.4.1 from version 2.3.2) compiler. You need to install the **OpenMP** library. You can install the **OpenMP** library by the following command: ``brew install libomp``.

Build from Sources
******************

.. code:: sh

    pip install --no-binary :all: gpboost

Requirements for building from sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Installation from sources requires that you have installed** `CMake`_.

- For **macOS** users, you can perform installation either with **Apple Clang** or **gcc**.

  - In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in the `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#apple-clang>`__) first and **CMake** version 3.16 or higher is required.

  - In case you prefer **gcc**, you need to install it (details for installation can be found in the `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc>`__) and specify compilers by running ``export CXX=g++-7 CC=gcc-7`` (replace "7" with version of **gcc** installed on your machine) first.

- For **Windows** users, **Visual Studio** (or `VS Build Tools <https://visualstudio.microsoft.com/downloads/>`_) is needed.

Build with MinGW-w64 on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install gpboost --install-option=--mingw

`CMake`_ and `MinGW-w64 <https://mingw-w64.org/>`_ should be installed first.

It is recommended to use **Visual Studio** for its better multithreading efficiency in **Windows** for many-core systems

Build 32-bit Version with 32-bit Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install gpboost --install-option=--bit32

By default, installation in environment with 32-bit Python is prohibited. However, you can remove this prohibition on your own risk by passing the ``bit32`` option (**not recommended**).


Install from GitHub
'''''''''''''''''''

.. code:: sh

    git clone --recursive https://github.com/fabsig/GPBoost.git
    cd GPBoost/python-package
    # export CXX=g++-7 CC=gcc-7  # macOS users, if you decided to compile with gcc, don't forget to specify compilers (replace "7" with version of gcc installed on your machine)
    python setup.py install

Note: ``sudo`` (or administrator rights in **Windows**) may be needed to perform the command.

If you get any errors during installation or due to any other reasons, you may want to build dynamic library from sources by any method you prefer and then just run ``python setup.py install --precompile``.


.. |License| image:: https://img.shields.io/github/license/fabsig/gpboost.svg
   :target: https://github.com/fabsig/GPBoost/blob/master/LICENSE
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/gpboost.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/gpboost
.. |PyPI Version| image:: https://img.shields.io/pypi/v/gpboost.svg?logo=pypi&logoColor=white
   :target: https://pypi.org/project/gpboost
.. |Downloads| image:: https://pepy.tech/badge/gpboost
   :target: https://pepy.tech/project/gpboost
.. _CMake: https://cmake.org/
.. _whlFiles: https://pypi.org/project/gpboost/#files
