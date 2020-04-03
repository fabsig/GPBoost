/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
 * Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef GPB_EXPORT_H_
#define GPB_EXPORT_H_

/** Macros for exporting symbols in MSVC/GCC/CLANG **/

#ifdef __cplusplus
#define GPBOOST_EXTERN_C extern "C"
#else
#define GPBOOST_EXTERN_C
#endif


#ifdef _MSC_VER
#define GPBOOST_EXPORT __declspec(dllexport)
#define GPBOOST_C_EXPORT GPBOOST_EXTERN_C __declspec(dllexport)
#else
#define GPBOOST_EXPORT
#define GPBOOST_C_EXPORT GPBOOST_EXTERN_C
#endif

#endif /** GPB_EXPORT_H_ **/
