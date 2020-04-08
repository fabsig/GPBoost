if(USE_GPU OR APPLE)
  cmake_minimum_required(VERSION 3.2)
else()
  cmake_minimum_required(VERSION 2.8)
endif()

##Only relevant for R-package: Add correrct version (32- or 64-bit) of mingw to path 
set(ENV{PATH} "${MINGW_PATH};$ENV{PATH}")

PROJECT(gpboost)

OPTION(USE_MPI "Enable MPI-based parallel learning" OFF)
OPTION(USE_OPENMP "Enable OpenMP" ON)
OPTION(USE_GPU "Enable GPU-accelerated training" OFF)
OPTION(USE_SWIG "Enable SWIG to generate Java API" OFF)
OPTION(USE_HDFS "Enable HDFS support (EXPERIMENTAL)" OFF)
OPTION(USE_R35 "Set to ON if your R version is not earlier than 3.5" OFF)
OPTION(CMAKE_BUILD_TYPE "Do optimization and no debugging for Eigen" Release)
OPTION(BUILD_FOR_R_REGISTER "Set to ON if building lib_gpboost for use with the R package and registering native routines" OFF)
OPTION(BUILD_FOR_R "Set to ON if building lib_gpboost for use with the R packages but not registering native routines" OFF)
OPTION(BUILD_32BIT_R "Set to ON if building 32-bit lib_gpboost for use with the R packages" OFF)

if(BUILD_FOR_R_REGISTER)
	if(WIN32)
		set(CMAKE_FIND_LIBRARY_PREFIXES "")
		set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib;.dll")
	elseif(VCPKG_CMAKE_SYSTEM_NAME STREQUAL "Linux")
		set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
		set(CMAKE_FIND_LIBRARY_SUFFIXES ".so;.a")
	elseif(VCPKG_CMAKE_SYSTEM_NAME STREQUAL "Darwin")
		set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
		set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib;.so;.a")
	endif()
	if(BUILD_32BIT_R)
	  set(R_LIB_ARCH "i386")
	else()
		set(R_LIB_ARCH "x64")
	endif()


	# The following lines until endif(BUILD_FOR_R_REGISTER) are Copyright (c) 1993-2015 Ken Martin, Will Schroeder, Bill Lorensen
	# source: https://gitlab.kitware.com/vtk/vtk/blob/650db03dff801f73ef8e04649912ffdf104ac671/CMake/FindR.cmake^
	set(TEMP_CMAKE_FIND_APPBUNDLE ${CMAKE_FIND_APPBUNDLE})
	set(CMAKE_FIND_APPBUNDLE "NEVER")
	find_program(R_COMMAND R DOC "R executable.")
	set(CMAKE_FIND_APPBUNDLE ${TEMP_CMAKE_FIND_APPBUNDLE})

	if(R_COMMAND)
	  execute_process(WORKING_DIRECTORY .
					  COMMAND ${R_COMMAND} RHOME
					  OUTPUT_VARIABLE R_ROOT_DIR
					  OUTPUT_STRIP_TRAILING_WHITESPACE)
	  # the following command does nothing currently, but will be used when deprecated code is removed
	  set(R_HOME ${R_ROOT_DIR} CACHE PATH "R home directory obtained from R RHOME")
	

	  find_path(R_INCLUDE_DIR R.h
				HINTS ${R_ROOT_DIR} ${R_ROOT_DIR}/bin/${R_LIB_ARCH}
				PATHS /usr/local/lib /usr/local/lib64 /usr/share
				PATH_SUFFIXES include R/include
				DOC "Path to file R.h")
		
	  find_library(R_LIBRARY_BASE R
				HINTS ${R_ROOT_DIR}/lib ${R_ROOT_DIR}/bin/${R_LIB_ARCH}
				DOC "R library (example libR.a, libR.dylib, etc.).")

	  find_library(R_LIBRARY_BLAS NAMES Rblas blas
				HINTS ${R_ROOT_DIR}/lib ${R_ROOT_DIR}/bin/${R_LIB_ARCH}
				DOC "Rblas library (example libRblas.a, libRblas.dylib, etc.).")

	  find_library(R_LIBRARY_LAPACK NAMES Rlapack lapack
				HINTS ${R_ROOT_DIR}/lib ${R_ROOT_DIR}/bin/${R_LIB_ARCH}
				DOC "Rlapack library (example libRlapack.a, libRlapack.dylib, etc.).")

	  find_library(R_LIBRARY_READLINE readline
				DOC "(Optional) system readline library. Only required if the R libraries were built with readline support.")

	else()
	  message(SEND_ERROR "FindR.cmake requires the following variables to be set: R_COMMAND")
	endif()

	# Note: R_LIBRARY_BASE is added to R_LIBRARIES twice; this may be due to circular linking dependencies; needs further investigation
	set(R_LIBRARIES ${R_LIBRARY_BASE} ${R_LIBRARY_BLAS} ${R_LIBRARY_LAPACK} ${R_LIBRARY_BASE})
	if(R_LIBRARY_READLINE)
	  set(R_LIBRARIES ${R_LIBRARIES} ${R_LIBRARY_READLINE})
	endif()

endif(BUILD_FOR_R_REGISTER)

if(APPLE)
    OPTION(APPLE_OUTPUT_DYLIB "Output dylib shared library" OFF)
endif(APPLE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.2")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.8")
    message(FATAL_ERROR "Insufficient Clang version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1.0")
    message(FATAL_ERROR "Insufficient AppleClang version")
  endif()
  cmake_minimum_required(VERSION 3.12)
elseif(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} doesn't support required C++11 features. Please use a newer MSVC.")
  endif()
  cmake_minimum_required(VERSION 3.8)
endif()

if(USE_SWIG)
  find_package(SWIG REQUIRED)
  find_package(Java REQUIRED)
  find_package(JNI REQUIRED)
  include(UseJava)
  include(UseSWIG)
  set(SWIG_CXX_EXTENSION "cxx")
  set(SWIG_EXTRA_LIBRARIES "")
  set(SWIG_JAVA_EXTRA_FILE_EXTENSIONS ".java" "JNI.java")
  set(SWIG_MODULE_JAVA_LANGUAGE "JAVA")
  set(SWIG_MODULE_JAVA_SWIG_LANGUAGE_FLAG "java")
  set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/java")
  include_directories(Java_INCLUDE_DIRS)
  include_directories(JNI_INCLUDE_DIRS)
  include_directories($ENV{JAVA_HOME}/include)
  if(WIN32)
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/gpboost/windows/x86_64")
      include_directories($ENV{JAVA_HOME}/include/win32)
  elseif(APPLE)
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/gpboost/osx/x86_64")
      include_directories($ENV{JAVA_HOME}/include/darwin)
  else()
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/gpboost/linux/x86_64")
      include_directories($ENV{JAVA_HOME}/include/linux)
  endif()
endif(USE_SWIG)

if(USE_R35)
    ADD_DEFINITIONS(-DR_VER_ABOVE_35)
endif(USE_R35)

if(BUILD_FOR_R_REGISTER)
    ADD_DEFINITIONS(-DGPB_R_BUILD)
endif(BUILD_FOR_R_REGISTER)

if(USE_MPI)
    find_package(MPI REQUIRED)
    ADD_DEFINITIONS(-DUSE_MPI)
    MESSAGE(STATUS "MPI libraries: " ${MPI_LIBRARIES})
    MESSAGE(STATUS "MPI C++ libraries: " ${MPI_CXX_LIBRARIES})
else()
    ADD_DEFINITIONS(-DUSE_SOCKET)
endif(USE_MPI)

if(USE_OPENMP)
  # find_package(OpenMP REQUIRED) does not work for mingw (SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") also does not work ...)
    find_package(OpenMP)
	if(OPENMP_FOUND)
	  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	else()
	  message(STATUS "OpenMP not found")
	endif()
else()
    # Ignore unknown #pragma warning
    if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif(USE_OPENMP)

if(USE_GPU)
    SET(BOOST_COMPUTE_HEADER_DIR ${PROJECT_SOURCE_DIR}/compute/include)
    include_directories(${BOOST_COMPUTE_HEADER_DIR})
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    MESSAGE(STATUS "OpenCL include directory: " ${OpenCL_INCLUDE_DIRS})
    if (WIN32)
        set(Boost_USE_STATIC_LIBS ON)
    endif()
    find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
    if (WIN32)
        # disable autolinking in boost
        add_definitions(-DBOOST_ALL_NO_LIB)
    endif()
    include_directories(${Boost_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DUSE_GPU)
endif(USE_GPU)

if(USE_HDFS)
    find_package(JNI REQUIRED)
    find_path(HDFS_INCLUDE_DIR hdfs.h REQUIRED)
    find_library(HDFS_LIB NAMES hdfs REQUIRED)
    include_directories(${HDFS_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_HDFS)
    SET(HDFS_CXX_LIBRARIES ${HDFS_LIB} ${JAVA_JVM_LIBRARY})
endif(USE_HDFS)

if(UNIX OR MINGW OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3 -Wextra -Wall -Wno-unknown-pragmas -Wno-return-type")
    if(UNIX OR CYGWIN)
	 SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes")
    endif()
    if(USE_SWIG)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
    endif()
endif()

if(WIN32 AND MINGW)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

# if(BUILD_32BIT_R)
	# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
# endif()

if(MSVC)
    SET(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2 /Ob2 /Oi /Ot /Oy /GL /MP")
else()
    if(NOT MINGW)#otherwise CRAN CHECK gives a note
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
endif(MSVC)

SET(gpboost_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${gpboost_HEADER_DIR})

SET(Eigen_HEADER_DIR ${PROJECT_SOURCE_DIR})
include_directories(${Eigen_HEADER_DIR})

SET(CSparse_HEADER_DIR ${PROJECT_SOURCE_DIR}/CSparse/Include)
include_directories(${CSparse_HEADER_DIR})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

if(BUILD_FOR_R_REGISTER)
	include_directories(${R_INCLUDE_DIR})#for R.h
endif(BUILD_FOR_R_REGISTER)

if(APPLE)
  if(APPLE_OUTPUT_DYLIB)
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
  else()
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  endif()
endif(APPLE)

if(USE_MPI)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif(USE_MPI)

file(GLOB SOURCES
    src/LightGBM/application/*.cpp
    src/LightGBM/boosting/*.cpp
    src/LightGBM/io/*.cpp
    src/LightGBM/metric/*.cpp
    src/LightGBM/objective/*.cpp
    src/LightGBM/network/*.cpp
    src/LightGBM/treelearner/*.cpp
    src/GPBoost/*.cpp
    CSparse/Source/*.c
)

add_executable(gpboost src/LightGBM/main.cpp ${SOURCES})
list(APPEND SOURCES "src/LightGBM/c_api.cpp")

# Only build the R part of the library if building for
# use with the R package
if(BUILD_FOR_R OR BUILD_FOR_R_REGISTER)
  list(APPEND SOURCES "src/LightGBM/lightgbm_R.cpp")
endif()

add_library(_gpboost SHARED ${SOURCES})

if(MSVC)
    set_target_properties(_gpboost PROPERTIES OUTPUT_NAME "lib_gpboost")
endif(MSVC)

if(USE_SWIG)
  set_property(SOURCE swig/gpboostlib.i PROPERTY CPLUSPLUS ON)
  LIST(APPEND swig_options -package com.microsoft.ml.gpboost)
  set_property(SOURCE swig/gpboostlib.i PROPERTY SWIG_FLAGS "${swig_options}")
  swig_add_module(_gpboost_swig java swig/gpboostlib.i)
  swig_link_libraries(_gpboost_swig _gpboost)
  # needed to ensure Linux build does not have lib prefix specified twice, e.g. liblib_gpboost_swig
  set_target_properties(_gpboost_swig PROPERTIES PREFIX "")
  # needed in some versions of CMake for VS and MinGW builds to ensure output dll has lib prefix
  set_target_properties(_gpboost_swig PROPERTIES OUTPUT_NAME "lib_gpboost_swig")
  if(WIN32)
    if(MINGW OR CYGWIN)
        add_custom_command(TARGET _gpboost_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PROJECT_SOURCE_DIR}/lib_gpboost.dll" com/microsoft/ml/gpboost/windows/x86_64
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PROJECT_SOURCE_DIR}/lib_gpboost_swig.dll" com/microsoft/ml/gpboost/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf gpboostlib.jar com)
    else()
        add_custom_command(TARGET _gpboost_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND cp "${PROJECT_SOURCE_DIR}/Release/*.dll" com/microsoft/ml/gpboost/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf gpboostlib.jar com)
    endif()
  elseif(APPLE)
    add_custom_command(TARGET _gpboost_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND cp "${PROJECT_SOURCE_DIR}/*.dylib" com/microsoft/ml/gpboost/osx/x86_64
            COMMAND cp "${PROJECT_SOURCE_DIR}/lib_gpboost_swig.jnilib" com/microsoft/ml/gpboost/osx/x86_64/lib_gpboost_swig.dylib
            COMMAND "${Java_JAR_EXECUTABLE}" -cf gpboostlib.jar com)
  else()
    add_custom_command(TARGET _gpboost_swig POST_BUILD
	    COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
	    COMMAND cp "${PROJECT_SOURCE_DIR}/*.so" com/microsoft/ml/gpboost/linux/x86_64
	    COMMAND "${Java_JAR_EXECUTABLE}" -cf gpboostlib.jar com)
  endif()
endif(USE_SWIG)

if(USE_MPI)
  TARGET_LINK_LIBRARIES(gpboost ${MPI_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(_gpboost ${MPI_CXX_LIBRARIES})
endif(USE_MPI)

if(USE_OPENMP)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        TARGET_LINK_LIBRARIES(gpboost OpenMP::OpenMP_CXX)
        TARGET_LINK_LIBRARIES(_gpboost OpenMP::OpenMP_CXX)
    endif()
endif(USE_OPENMP)

if(USE_GPU)
  TARGET_LINK_LIBRARIES(gpboost ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
  TARGET_LINK_LIBRARIES(_gpboost ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
endif(USE_GPU)

if(USE_HDFS)
  TARGET_LINK_LIBRARIES(gpboost ${HDFS_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(_gpboost ${HDFS_CXX_LIBRARIES})
endif(USE_HDFS)

if(WIN32 AND (MINGW OR CYGWIN))
    TARGET_LINK_LIBRARIES(gpboost Ws2_32)
    TARGET_LINK_LIBRARIES(_gpboost Ws2_32)
    TARGET_LINK_LIBRARIES(gpboost IPHLPAPI)
    TARGET_LINK_LIBRARIES(_gpboost IPHLPAPI)
endif()

if(BUILD_FOR_R_REGISTER)
    TARGET_LINK_LIBRARIES(gpboost ${R_LIBRARIES})
    TARGET_LINK_LIBRARIES(_gpboost ${R_LIBRARIES})
endif(BUILD_FOR_R_REGISTER)

install(TARGETS gpboost _gpboost
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY ${gpboost_HEADER_DIR}/gpboost DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
