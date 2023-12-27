Installation Guide
==================

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

This is the guide for the build of GPBoost command line interface (CLI) version. For building the Python and R packages, please refer to `Python-package`_ and `R-package`_.

All instructions below are aimed for compiling a 64-bit version.
It is worth to compile 32-bit version only in very rare special cases of environmental limitations.
32-bit version is slow and untested, so use it on your own risk and don't forget to adjust some commands in this guide.

Windows
~~~~~~~

On Windows LightGBM can be built using

- **CMake** and **VS Build Tools** or **Visual Studio**

- **CMake** and **MinGW**

Visual Studio (or VS Build Tools)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** (2015 or newer) is already installed).

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/fabsig/GPBoost
     cd GPBoost
     mkdir build
     cd build
     cmake -A x64 ..
     cmake --build . --target ALL_BUILD --config Release

*Note*: sometimes running ``cmake -A x64 ..`` gives an error ``Generator X does not support platform specification, but platform x64 was specified.`` In this case, you need to explicitly provide the generator, for instance: 

   .. code::

     cmake -G "Visual Studio 17 2022" ..

The ``.exe`` and ``.dll`` files will be in the ``GPBoost/Release`` folder.

MinGW-w64
^^^^^^^^^

1. Install `Git for Windows`_, `CMake`_ and `MinGW-w64`_.

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/fabsig/GPBoost
     cd LightGBM
     mkdir build
     cd build
     cmake -G "MinGW Makefiles" ..
     mingw32-make.exe -j4

The ``.exe`` and ``.dll`` files will be in the ``GPBoost/`` folder.

**Note**: You may need to run the ``cmake -G "MinGW Makefiles" ..`` one more time if you encounter the ``sh.exe was found in your PATH`` error.

Linux
~~~~~

On Linux GPBoost can be built using **CMake** and **gcc** or **Clang**.

1. Install `CMake`_.

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/fabsig/GPBoost
     cd GPBoost
     mkdir build
     cd build
     cmake ..
     make -j4

**Note**: glibc >= 2.14 is required.

**Note**: In some rare cases you may need to install OpenMP runtime library separately (use your package manager and search for ``lib[g|i]omp`` for doing this).

macOS
~~~~~

On macOS GPBoost can be built using **CMake** and **Apple Clang** or **gcc**.

Apple Clang
^^^^^^^^^^^

Only **Apple Clang** version 8.1 or higher is supported.

1. Install `CMake`_ (3.16 or higher):

   .. code::

     brew install cmake

2. Install **OpenMP**:

   .. code::

     brew install libomp

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/fabsig/GPBoost
     cd GPBoost
     mkdir build
	 cd build
	 cmake ..
	 make -j4

gcc
^^^

1. Install `CMake`_ (3.2 or higher):

   .. code::

     brew install cmake

2. Install **gcc**:

   .. code::

     brew install gcc

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/fabsig/GPBoost
     cd GPBoost
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     mkdir build
     cd build
     cmake ..
     make -j4


.. _Python-package: https://github.com/fabsig/GPBoost/tree/master/python-package

.. _R-package: https://github.com/fabsig/GPBoost/tree/master/R-package

.. _Visual Studio: https://visualstudio.microsoft.com/downloads/

.. _Git for Windows: https://git-scm.com/download/win

.. _CMake: https://cmake.org/

.. _VS Build Tools: https://visualstudio.microsoft.com/downloads/

.. _MinGW-w64: https://www.mingw-w64.org/downloads/
