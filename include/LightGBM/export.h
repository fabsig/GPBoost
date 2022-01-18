/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef GPBOOST_EXPORT_H_
#define GPBOOST_EXPORT_H_

/** Macros for exporting symbols in MSVC/GCC/CLANG **/

#ifdef __cplusplus
#define LIGHTGBM_EXTERN_C extern "C"
#else
#define LIGHTGBM_EXTERN_C
#endif


#ifdef _MSC_VER
#define LIGHTGBM_EXPORT __declspec(dllexport)
#define GPBOOST_C_EXPORT LIGHTGBM_EXTERN_C __declspec(dllexport)
#else
#define LIGHTGBM_EXPORT
#define GPBOOST_C_EXPORT LIGHTGBM_EXTERN_C
#endif

#endif /** GPBOOST_EXPORT_H_ **/
