/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*
* \note
* To avoid type conversion on large data, the most of our exposed interface supports both float32 and float64,
* except the following:
*   1. gradient and Hessian;
*   2. current score for training and validation data.
*   .
* The reason is that they are called frequently, and the type conversion on them may be time-cost.
*/

#ifndef LIGHTGBM_C_API_H_
#define LIGHTGBM_C_API_H_

#include <LightGBM/export.h>

#include <cstdint>
#include <cstdio>
#include <cstring>


typedef void* DatasetHandle;  /*!< \brief Handle of dataset. */
typedef void* BoosterHandle;  /*!< \brief Handle of booster. */
typedef void* FastConfigHandle; /*!< \brief Handle of FastConfig. */
typedef void* REModelHandle;  /*!< \brief Handle of re_model. */

#define C_API_DTYPE_FLOAT32 (0)  /*!< \brief float32 (single precision float). */
#define C_API_DTYPE_FLOAT64 (1)  /*!< \brief float64 (double precision float). */
#define C_API_DTYPE_INT32   (2)  /*!< \brief int32. */
#define C_API_DTYPE_INT64   (3)  /*!< \brief int64. */

#define C_API_PREDICT_NORMAL     (0)  /*!< \brief Normal prediction, with transform (if needed). */
#define C_API_PREDICT_RAW_SCORE  (1)  /*!< \brief Predict raw score. */
#define C_API_PREDICT_LEAF_INDEX (2)  /*!< \brief Predict leaf index. */
#define C_API_PREDICT_CONTRIB    (3)  /*!< \brief Predict feature contributions (SHAP values). */

#define C_API_MATRIX_TYPE_CSR (0)  /*!< \brief CSR sparse matrix type. */
#define C_API_MATRIX_TYPE_CSC (1)  /*!< \brief CSC sparse matrix type. */

#define C_API_FEATURE_IMPORTANCE_SPLIT (0)  /*!< \brief Split type of feature importance. */
#define C_API_FEATURE_IMPORTANCE_GAIN  (1)  /*!< \brief Gain type of feature importance. */

/*!
 * \brief Get string message of the last error.
 * \return Error information
 */
GPBOOST_C_EXPORT const char* LGBM_GetLastError();

/*!
 * \brief Register a callback function for log redirecting.
 * \param callback The callback function to register
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_RegisterLogCallback(void (*callback)(const char*));

// --- start Dataset interface

/*!
 * \brief Load dataset from file (like LightGBM CLI version does).
 * \param filename The name of the file
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out A loaded dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetCreateFromFile(const char* filename,
                                                 const char* parameters,
                                                 const DatasetHandle reference,
                                                 DatasetHandle* out);

/*!
 * \brief Allocate the space for dataset and bucket feature bins according to sampled data.
 * \param sample_data Sampled data, grouped by the column
 * \param sample_indices Indices of sampled data
 * \param ncol Number of columns
 * \param num_per_col Size of each sampling column
 * \param num_sample_row Number of sampled rows
 * \param num_total_row Number of total rows
 * \param parameters Additional parameters
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetCreateFromSampledColumn(double** sample_data,
                                                          int** sample_indices,
                                                          int32_t ncol,
                                                          const int* num_per_col,
                                                          int32_t num_sample_row,
                                                          int32_t num_total_row,
                                                          const char* parameters,
                                                          DatasetHandle* out);

/*!
 * \brief Allocate the space for dataset and bucket feature bins according to reference dataset.
 * \param reference Used to align bin mapper with other dataset
 * \param num_total_row Number of total rows
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetCreateByReference(const DatasetHandle reference,
                                                    int64_t num_total_row,
                                                    DatasetHandle* out);

/*!
 * \brief Push data to existing dataset, if ``nrow + start_row == num_total_row``, will call ``dataset->FinishLoad``.
 * \param dataset Handle of dataset
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nrow Number of rows
 * \param ncol Number of columns
 * \param start_row Row start index
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetPushRows(DatasetHandle dataset,
                                           const void* data,
                                           int data_type,
                                           int32_t nrow,
                                           int32_t ncol,
                                           int32_t start_row);

/*!
 * \brief Push data to existing dataset, if ``nrow + start_row == num_total_row``, will call ``dataset->FinishLoad``.
 * \param dataset Handle of dataset
 * \param indptr Pointer to row headers
 * \param indptr_type Type of ``indptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col Number of columns
 * \param start_row Row start index
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetPushRowsByCSR(DatasetHandle dataset,
                                                const void* indptr,
                                                int indptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t nindptr,
                                                int64_t nelem,
                                                int64_t num_col,
                                                int64_t start_row);

/*!
 * \brief Create a dataset from CSR format.
 * \param indptr Pointer to row headers
 * \param indptr_type Type of ``indptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col Number of columns
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetCreateFromCSR(const void* indptr,
                                                int indptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t nindptr,
                                                int64_t nelem,
                                                int64_t num_col,
                                                const char* parameters,
                                                const DatasetHandle reference,
                                                DatasetHandle* out);

/*!
 * \brief Create a dataset from CSR format through callbacks.
 * \param get_row_funptr Pointer to ``std::function<void(int idx, std::vector<std::pair<int, double>>& ret)>``
 *                       (called for every row and expected to clear and fill ``ret``)
 * \param num_rows Number of rows
 * \param num_col Number of columns
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetCreateFromCSRFunc(void* get_row_funptr,
                                                    int num_rows,
                                                    int64_t num_col,
                                                    const char* parameters,
                                                    const DatasetHandle reference,
                                                    DatasetHandle* out);

/*!
 * \brief Create a dataset from CSC format.
 * \param col_ptr Pointer to column headers
 * \param col_ptr_type Type of ``col_ptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param indices Pointer to row indices
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param ncol_ptr Number of columns in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_row Number of rows
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetCreateFromCSC(const void* col_ptr,
                                                int col_ptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t ncol_ptr,
                                                int64_t nelem,
                                                int64_t num_row,
                                                const char* parameters,
                                                const DatasetHandle reference,
                                                DatasetHandle* out);

/*!
 * \brief Create dataset from dense matrix.
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nrow Number of rows
 * \param ncol Number of columns
 * \param is_row_major 1 for row-major, 0 for column-major
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetCreateFromMat(const void* data,
                                                int data_type,
                                                int32_t nrow,
                                                int32_t ncol,
                                                int is_row_major,
                                                const char* parameters,
                                                const DatasetHandle reference,
                                                DatasetHandle* out);

/*!
 * \brief Create dataset from array of dense matrices.
 * \param nmat Number of dense matrices
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nrow Number of rows
 * \param ncol Number of columns
 * \param is_row_major 1 for row-major, 0 for column-major
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetCreateFromMats(int32_t nmat,
                                                 const void** data,
                                                 int data_type,
                                                 int32_t* nrow,
                                                 int32_t ncol,
                                                 int is_row_major,
                                                 const char* parameters,
                                                 const DatasetHandle reference,
                                                 DatasetHandle* out);

/*!
 * \brief Create subset of a data.
 * \param handle Handle of full dataset
 * \param used_row_indices Indices used in subset
 * \param num_used_row_indices Length of ``used_row_indices``
 * \param parameters Additional parameters
 * \param[out] out Subset of data
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetGetSubset(const DatasetHandle handle,
                                            const int32_t* used_row_indices,
                                            int32_t num_used_row_indices,
                                            const char* parameters,
                                            DatasetHandle* out);

/*!
 * \brief Save feature names to dataset.
 * \param handle Handle of dataset
 * \param feature_names Feature names
 * \param num_feature_names Number of feature names
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetSetFeatureNames(DatasetHandle handle,
                                                  const char** feature_names,
                                                  int num_feature_names);

/*!
 * \brief Get feature names of dataset.
 * \param handle Handle of dataset
 * \param len Number of ``char*`` pointers stored at ``out_strs``.
 *            If smaller than the max size, only this many strings are copied
 * \param[out] num_feature_names Number of feature names
 * \param buffer_len Size of pre-allocated strings.
 *                   Content is copied up to ``buffer_len - 1`` and null-terminated
 * \param[out] out_buffer_len String sizes required to do the full string copies
 * \param[out] feature_names Feature names, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetGetFeatureNames(DatasetHandle handle,
                                                  const int len,
                                                  int* num_feature_names,
                                                  const size_t buffer_len,
                                                  size_t* out_buffer_len,
                                                  char** feature_names);

/*!
 * \brief Free space for dataset.
 * \param handle Handle of dataset to be freed
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetFree(DatasetHandle handle);

/*!
 * \brief Save dataset to binary file.
 * \param handle Handle of dataset
 * \param filename The name of the file
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetSaveBinary(DatasetHandle handle,
                                             const char* filename);

/*!
 * \brief Save dataset to text file, intended for debugging use only.
 * \param handle Handle of dataset
 * \param filename The name of the file
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetDumpText(DatasetHandle handle,
                                           const char* filename);

/*!
 * \brief Set vector to a content in info.
 * \note
 * - \a group only works for ``C_API_DTYPE_INT32``;
 * - \a label and \a weight only work for ``C_API_DTYPE_FLOAT32``;
 * - \a init_score only works for ``C_API_DTYPE_FLOAT64``.
 * \param handle Handle of dataset
 * \param field_name Field name, can be \a label, \a weight, \a init_score, \a group
 * \param field_data Pointer to data vector
 * \param num_element Number of elements in ``field_data``
 * \param type Type of ``field_data`` pointer, can be ``C_API_DTYPE_INT32``, ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetSetField(DatasetHandle handle,
                                           const char* field_name,
                                           const void* field_data,
                                           int num_element,
                                           int type);

/*!
 * \brief Get info vector from dataset.
 * \param handle Handle of dataset
 * \param field_name Field name
 * \param[out] out_len Used to set result length
 * \param[out] out_ptr Pointer to the result
 * \param[out] out_type Type of result pointer, can be ``C_API_DTYPE_INT32``, ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetGetField(DatasetHandle handle,
                                           const char* field_name,
                                           int* out_len,
                                           const void** out_ptr,
                                           int* out_type);

/*!
 * \brief Raise errors for attempts to update dataset parameters.
 * \param old_parameters Current dataset parameters
 * \param new_parameters New dataset parameters
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetUpdateParamChecking(const char* old_parameters,
                                                      const char* new_parameters);

/*!
 * \brief Get number of data points.
 * \param handle Handle of dataset
 * \param[out] out The address to hold number of data points
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetGetNumData(DatasetHandle handle,
                                             int* out);

/*!
 * \brief Get number of features.
 * \param handle Handle of dataset
 * \param[out] out The address to hold number of features
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetGetNumFeature(DatasetHandle handle,
                                                int* out);

/*!
 * \brief Add features from ``source`` to ``target``.
 * \param target The handle of the dataset to add features to
 * \param source The handle of the dataset to take features from
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_DatasetAddFeaturesFrom(DatasetHandle target,
                                                  DatasetHandle source);

// --- start Booster interfaces

/*!
* \brief Get boolean representing whether booster is fitting linear trees.
* \param handle Handle of booster
* \param[out] out The address to hold linear trees indicator
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int LGBM_BoosterGetLinear(BoosterHandle handle, bool* out);

/*!
 * \brief Create a new boosting learner.
 * \param train_data Training dataset
 * \param parameters Parameters in format 'key1=value1 key2=value2'
 * \param[out] out Handle of created booster
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterCreate(const DatasetHandle train_data,
                                         const char* parameters,
                                         BoosterHandle* out);

/*!
 * \brief Create a new boosting learner.
 * \param train_data Training dataset
 * \param parameters Parameters in format 'key1=value1 key2=value2'
 * \param re_model Gaussian process model
 * \param[out] out Handle of created booster
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_GPBoosterCreate(const DatasetHandle train_data,
    const char* parameters,
    const REModelHandle re_model,
    BoosterHandle* out);

/*!
 * \brief Load an existing booster from model file.
 * \param filename Filename of model
 * \param[out] out_num_iterations Number of iterations of this booster
 * \param[out] out Handle of created booster
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterCreateFromModelfile(const char* filename,
                                                      int* out_num_iterations,
                                                      BoosterHandle* out);

/*!
 * \brief Load an existing booster from string.
 * \param model_str Model string
 * \param[out] out_num_iterations Number of iterations of this booster
 * \param[out] out Handle of created booster
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterLoadModelFromString(const char* model_str,
                                                      int* out_num_iterations,
                                                      BoosterHandle* out);

/*!
 * \brief Free space for booster.
 * \param handle Handle of booster to be freed
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterFree(BoosterHandle handle);

/*!
 * \brief Shuffle models.
 * \param handle Handle of booster
 * \param start_iter The first iteration that will be shuffled
 * \param end_iter The last iteration that will be shuffled
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterShuffleModels(BoosterHandle handle,
                                                int start_iter,
                                                int end_iter);

/*!
 * \brief Merge model from ``other_handle`` into ``handle``.
 * \param handle Handle of booster, will merge another booster into this one
 * \param other_handle Other handle of booster
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterMerge(BoosterHandle handle,
                                        BoosterHandle other_handle);

/*!
 * \brief Add new validation data to booster.
 * \param handle Handle of booster
 * \param valid_data Validation dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterAddValidData(BoosterHandle handle,
                                               const DatasetHandle valid_data);

/*!
 * \brief Reset training data for booster.
 * \param handle Handle of booster
 * \param train_data Training dataset
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterResetTrainingData(BoosterHandle handle,
                                                    const DatasetHandle train_data);

/*!
 * \brief Reset config for booster.
 * \param handle Handle of booster
 * \param parameters Parameters in format 'key1=value1 key2=value2'
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterResetParameter(BoosterHandle handle,
                                                 const char* parameters);

/*!
 * \brief Get number of classes.
 * \param handle Handle of booster
 * \param[out] out_len Number of classes
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetNumClasses(BoosterHandle handle,
                                                int* out_len);

/*!
 * \brief Update the model for one iteration.
 * \param handle Handle of booster
 * \param[out] is_finished 1 means the update was successfully finished (cannot split any more), 0 indicates failure
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterUpdateOneIter(BoosterHandle handle,
                                                int* is_finished);

/*!
 * \brief Refit the tree model using the new data (online learning).
 * \param handle Handle of booster
 * \param leaf_preds Pointer to predicted leaf indices
 * \param nrow Number of rows of ``leaf_preds``
 * \param ncol Number of columns of ``leaf_preds``
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterRefit(BoosterHandle handle,
                                        const int32_t* leaf_preds,
                                        int32_t nrow,
                                        int32_t ncol);

/*!
 * \brief Update the model by specifying gradient and Hessian directly
 *        (this can be used to support customized loss functions).
 * \param handle Handle of booster
 * \param grad The first order derivative (gradient) statistics
 * \param hess The second order derivative (Hessian) statistics
 * \param[out] is_finished 1 means the update was successfully finished (cannot split any more), 0 indicates failure
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
                                                      const float* grad,
                                                      const float* hess,
                                                      int* is_finished);

/*!
 * \brief Rollback one iteration.
 * \param handle Handle of booster
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterRollbackOneIter(BoosterHandle handle);

/*!
 * \brief Get index of the current boosting iteration.
 * \param handle Handle of booster
 * \param[out] out_iteration Index of the current boosting iteration
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetCurrentIteration(BoosterHandle handle,
                                                      int* out_iteration);

/*!
 * \brief Get number of trees per iteration.
 * \param handle Handle of booster
 * \param[out] out_tree_per_iteration Number of trees per iteration
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterNumModelPerIteration(BoosterHandle handle,
                                                       int* out_tree_per_iteration);

/*!
 * \brief Get number of weak sub-models.
 * \param handle Handle of booster
 * \param[out] out_models Number of weak sub-models
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterNumberOfTotalModel(BoosterHandle handle,
                                                     int* out_models);

/*!
 * \brief Get number of evaluation datasets.
 * \param handle Handle of booster
 * \param[out] out_len Total number of evaluation datasets
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetEvalCounts(BoosterHandle handle,
                                                int* out_len);

/*!
 * \brief Get names of evaluation datasets.
 * \param handle Handle of booster
 * \param len Number of ``char*`` pointers stored at ``out_strs``.
 *            If smaller than the max size, only this many strings are copied
 * \param[out] out_len Total number of evaluation datasets
 * \param buffer_len Size of pre-allocated strings.
 *                   Content is copied up to ``buffer_len - 1`` and null-terminated
 * \param[out] out_buffer_len String sizes required to do the full string copies
 * \param[out] out_strs Names of evaluation datasets, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetEvalNames(BoosterHandle handle,
                                               const int len,
                                               int* out_len,
                                               const size_t buffer_len,
                                               size_t* out_buffer_len,
                                               char** out_strs);

/*!
 * \brief Get names of features.
 * \param handle Handle of booster
 * \param len Number of ``char*`` pointers stored at ``out_strs``.
 *            If smaller than the max size, only this many strings are copied
 * \param[out] out_len Total number of features
 * \param buffer_len Size of pre-allocated strings.
 *                   Content is copied up to ``buffer_len - 1`` and null-terminated
 * \param[out] out_buffer_len String sizes required to do the full string copies
 * \param[out] out_strs Names of features, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetFeatureNames(BoosterHandle handle,
                                                  const int len,
                                                  int* out_len,
                                                  const size_t buffer_len,
                                                  size_t* out_buffer_len,
                                                  char** out_strs);

/*!
 * \brief Get number of features.
 * \param handle Handle of booster
 * \param[out] out_len Total number of features
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetNumFeature(BoosterHandle handle,
                                                int* out_len);

/*!
 * \brief Get evaluation for training data and validation data.
 * \note
 *   1. You should call ``LGBM_BoosterGetEvalNames`` first to get the names of evaluation datasets.
 *   2. You should pre-allocate memory for ``out_results``, you can get its length by ``LGBM_BoosterGetEvalCounts``.
 * \param handle Handle of booster
 * \param data_idx Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
 * \param[out] out_len Length of output result
 * \param[out] out_results Array with evaluation results
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetEval(BoosterHandle handle,
                                          int data_idx,
                                          int* out_len,
                                          double* out_results);

/*!
 * \brief Get number of predictions for training data and validation data
 *        (this can be used to support customized evaluation functions).
 * \param handle Handle of booster
 * \param data_idx Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
 * \param[out] out_len Number of predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetNumPredict(BoosterHandle handle,
                                                int data_idx,
                                                int64_t* out_len);

/*!
 * \brief Get prediction for training data and validation data.
 * \note
 * You should pre-allocate memory for ``out_result``, its length is equal to ``num_class * num_data``.
 * \param handle Handle of booster
 * \param data_idx Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetPredict(BoosterHandle handle,
                                             int data_idx,
                                             int64_t* out_len,
                                             double* out_result);

/*!
 * \brief Make prediction for file.
 * \param handle Handle of booster
 * \param data_filename Filename of file with data
 * \param data_has_header Whether file has header or not
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param result_filename Filename of result file in which predictions will be written
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForFile(BoosterHandle handle,
                                                 const char* data_filename,
                                                 int data_has_header,
                                                 int predict_type,
                                                 int start_iteration,
                                                 int num_iteration,
                                                 const char* parameter,
                                                 const char* result_filename);

/*!
 * \brief Get number of predictions.
 * \param handle Handle of booster
 * \param num_row Number of rows
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param[out] out_len Length of prediction
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterCalcNumPredict(BoosterHandle handle,
                                                 int num_row,
                                                 int predict_type,
                                                 int start_iteration,
                                                 int num_iteration,
                                                 int64_t* out_len);

/*!
 * \brief Release FastConfig object.
 *
 * \param fastConfig Handle to the FastConfig object acquired with a ``*FastInit()`` method.
 * \return 0 when it succeeds, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_FastConfigFree(FastConfigHandle fastConfig);

/*!
 * \brief Make prediction for a new dataset in CSR format.
 * \note
 * You should pre-allocate memory for ``out_result``:
 *   - for normal and raw score, its length is equal to ``num_class * num_data``;
 *   - for leaf index, its length is equal to ``num_class * num_data * num_iteration``;
 *   - for feature contributions, its length is equal to ``num_class * num_data * (num_feature + 1)``.
 * \param handle Handle of booster
 * \param indptr Pointer to row headers
 * \param indptr_type Type of ``indptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col Number of columns
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForCSR(BoosterHandle handle,
                                                const void* indptr,
                                                int indptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t nindptr,
                                                int64_t nelem,
                                                int64_t num_col,
                                                int predict_type,
                                                int start_iteration,
                                                int num_iteration,
                                                const char* parameter,
                                                int64_t* out_len,
                                                double* out_result);

/*!
 * \brief Make sparse prediction for a new dataset in CSR or CSC format. Currently only used for feature contributions.
 * \note
 * The outputs are pre-allocated, as they can vary for each invocation, but the shape should be the same:
 *   - for feature contributions, the shape of sparse matrix will be ``num_class * num_data * (num_feature + 1)``.
 * The output indptr_type for the sparse matrix will be the same as the given input indptr_type.
 * Call ``LGBM_BoosterFreePredictSparse`` to deallocate resources.
 * \param handle Handle of booster
 * \param indptr Pointer to row headers for CSR or column headers for CSC
 * \param indptr_type Type of ``indptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param indices Pointer to column indices for CSR or row indices for CSC
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col_or_row Number of columns for CSR or number of rows for CSC
 * \param predict_type What should be predicted, only feature contributions supported currently
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param matrix_type Type of matrix input and output, can be ``C_API_MATRIX_TYPE_CSR`` or ``C_API_MATRIX_TYPE_CSC``
 * \param[out] out_len Length of output indices and data
 * \param[out] out_indptr Pointer to output row headers for CSR or column headers for CSC
 * \param[out] out_indices Pointer to sparse column indices for CSR or row indices for CSC
 * \param[out] out_data Pointer to sparse data space
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictSparseOutput(BoosterHandle handle,
                                                      const void* indptr,
                                                      int indptr_type,
                                                      const int32_t* indices,
                                                      const void* data,
                                                      int data_type,
                                                      int64_t nindptr,
                                                      int64_t nelem,
                                                      int64_t num_col_or_row,
                                                      int predict_type,
                                                      int start_iteration,
                                                      int num_iteration,
                                                      const char* parameter,
                                                      int matrix_type,
                                                      int64_t* out_len,
                                                      void** out_indptr,
                                                      int32_t** out_indices,
                                                      void** out_data);

/*!
 * \brief Method corresponding to ``LGBM_BoosterPredictSparseOutput`` to free the allocated data.
 * \param indptr Pointer to output row headers or column headers to be deallocated
 * \param indices Pointer to sparse indices to be deallocated
 * \param data Pointer to sparse data space to be deallocated
 * \param indptr_type Type of ``indptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterFreePredictSparse(void* indptr, int32_t* indices, void* data, int indptr_type, int data_type);

/*!
 * \brief Make prediction for a new dataset in CSR format. This method re-uses the internal predictor structure
 *        from previous calls and is optimized for single row invocation.
 * \note
 * You should pre-allocate memory for ``out_result``:
 *   - for normal and raw score, its length is equal to ``num_class * num_data``;
 *   - for leaf index, its length is equal to ``num_class * num_data * num_iteration``;
 *   - for feature contributions, its length is equal to ``num_class * num_data * (num_feature + 1)``.
 * \param handle Handle of booster
 * \param indptr Pointer to row headers
 * \param indptr_type Type of ``indptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col Number of columns
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForCSRSingleRow(BoosterHandle handle,
                                                         const void* indptr,
                                                         int indptr_type,
                                                         const int32_t* indices,
                                                         const void* data,
                                                         int data_type,
                                                         int64_t nindptr,
                                                         int64_t nelem,
                                                         int64_t num_col,
                                                         int predict_type,
                                                         int start_iteration,
                                                         int num_iteration,
                                                         const char* parameter,
                                                         int64_t* out_len,
                                                         double* out_result);

/*!
 * \brief Initialize and return a ``FastConfigHandle`` for use with ``LGBM_BoosterPredictForCSRSingleRowFast``.
 *
 * Release the ``FastConfig`` by passing its handle to ``LGBM_FastConfigFree`` when no longer needed.
 *
 * \param handle Booster handle
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param num_col Number of columns
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_fastConfig FastConfig object with which you can call ``LGBM_BoosterPredictForCSRSingleRowFast``
 * \return 0 when it succeeds, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForCSRSingleRowFastInit(BoosterHandle handle,
                                                                 const int predict_type,
                                                                 const int start_iteration,
                                                                 const int num_iteration,
                                                                 const int data_type,
                                                                 const int64_t num_col,
                                                                 const char* parameter,
                                                                 FastConfigHandle *out_fastConfig);

/*!
 * \brief Faster variant of ``LGBM_BoosterPredictForCSRSingleRow``.
 *
 * Score single rows after setup with ``LGBM_BoosterPredictForCSRSingleRowFastInit``.
 *
 * By removing the setup steps from this call extra optimizations can be made like
 * initializing the config only once, instead of once per call.
 *
 * \note
 *   Setting up the number of threads is only done once at ``LGBM_BoosterPredictForCSRSingleRowFastInit``
 *   instead of at each prediction.
 *   If you use a different number of threads in other calls, you need to start the setup process over,
 *   or that number of threads will be used for these calls as well.
 *
 * \note
 * You should pre-allocate memory for ``out_result``:
 *   - for normal and raw score, its length is equal to ``num_class * num_data``;
 *   - for leaf index, its length is equal to ``num_class * num_data * num_iteration``;
 *   - for feature contributions, its length is equal to ``num_class * num_data * (num_feature + 1)``.
 *
 * \param fastConfig_handle FastConfig object handle returned by ``LGBM_BoosterPredictForCSRSingleRowFastInit``
 * \param indptr Pointer to row headers
 * \param indptr_type Type of ``indptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForCSRSingleRowFast(FastConfigHandle fastConfig_handle,
                                                             const void* indptr,
                                                             const int indptr_type,
                                                             const int32_t* indices,
                                                             const void* data,
                                                             const int64_t nindptr,
                                                             const int64_t nelem,
                                                             int64_t* out_len,
                                                             double* out_result);

/*!
 * \brief Make prediction for a new dataset in CSC format.
 * \note
 * You should pre-allocate memory for ``out_result``:
 *   - for normal and raw score, its length is equal to ``num_class * num_data``;
 *   - for leaf index, its length is equal to ``num_class * num_data * num_iteration``;
 *   - for feature contributions, its length is equal to ``num_class * num_data * (num_feature + 1)``.
 * \param handle Handle of booster
 * \param col_ptr Pointer to column headers
 * \param col_ptr_type Type of ``col_ptr``, can be ``C_API_DTYPE_INT32`` or ``C_API_DTYPE_INT64``
 * \param indices Pointer to row indices
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param ncol_ptr Number of columns in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_row Number of rows
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iteration for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForCSC(BoosterHandle handle,
                                                const void* col_ptr,
                                                int col_ptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t ncol_ptr,
                                                int64_t nelem,
                                                int64_t num_row,
                                                int predict_type,
                                                int start_iteration,
                                                int num_iteration,
                                                const char* parameter,
                                                int64_t* out_len,
                                                double* out_result);

/*!
 * \brief Make prediction for a new dataset.
 * \note
 * You should pre-allocate memory for ``out_result``:
 *   - for normal and raw score, its length is equal to ``num_class * num_data``;
 *   - for leaf index, its length is equal to ``num_class * num_data * num_iteration``;
 *   - for feature contributions, its length is equal to ``num_class * num_data * (num_feature + 1)``.
 * \param handle Handle of booster
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nrow Number of rows
 * \param ncol Number of columns
 * \param is_row_major 1 for row-major, 0 for column-major
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iteration for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForMat(BoosterHandle handle,
                                                const void* data,
                                                int data_type,
                                                int32_t nrow,
                                                int32_t ncol,
                                                int is_row_major,
                                                int predict_type,
                                                int start_iteration,
                                                int num_iteration,
                                                const char* parameter,
                                                int64_t* out_len,
                                                double* out_result);

/*!
 * \brief Make prediction for a new dataset. This method re-uses the internal predictor structure
 *        from previous calls and is optimized for single row invocation.
 * \note
 * You should pre-allocate memory for ``out_result``:
 *   - for normal and raw score, its length is equal to ``num_class * num_data``;
 *   - for leaf index, its length is equal to ``num_class * num_data * num_iteration``;
 *   - for feature contributions, its length is equal to ``num_class * num_data * (num_feature + 1)``.
 * \param handle Handle of booster
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param ncol Number columns
 * \param is_row_major 1 for row-major, 0 for column-major
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iteration for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForMatSingleRow(BoosterHandle handle,
                                                         const void* data,
                                                         int data_type,
                                                         int ncol,
                                                         int is_row_major,
                                                         int predict_type,
                                                         int start_iteration,
                                                         int num_iteration,
                                                         const char* parameter,
                                                         int64_t* out_len,
                                                         double* out_result);

/*!
 * \brief Initialize and return a ``FastConfigHandle`` for use with ``LGBM_BoosterPredictForMatSingleRowFast``.
 *
 * Release the ``FastConfig`` by passing its handle to ``LGBM_FastConfigFree`` when no longer needed.
 *
 * \param handle Booster handle
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param ncol Number of columns
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_fastConfig FastConfig object with which you can call ``LGBM_BoosterPredictForMatSingleRowFast``
 * \return 0 when it succeeds, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForMatSingleRowFastInit(BoosterHandle handle,
                                                                 const int predict_type,
                                                                 const int start_iteration,
                                                                 const int num_iteration,
                                                                 const int data_type,
                                                                 const int32_t ncol,
                                                                 const char* parameter,
                                                                 FastConfigHandle *out_fastConfig);

/*!
 * \brief Faster variant of ``LGBM_BoosterPredictForMatSingleRow``.
 *
 * Score a single row after setup with ``LGBM_BoosterPredictForMatSingleRowFastInit``.
 *
 * By removing the setup steps from this call extra optimizations can be made like
 * initializing the config only once, instead of once per call.
 *
 * \note
 *   Setting up the number of threads is only done once at ``LGBM_BoosterPredictForMatSingleRowFastInit``
 *   instead of at each prediction.
 *   If you use a different number of threads in other calls, you need to start the setup process over,
 *   or that number of threads will be used for these calls as well.
 *
 * \param fastConfig_handle FastConfig object handle returned by ``LGBM_BoosterPredictForMatSingleRowFastInit``
 * \param data Single-row array data (no other way than row-major form).
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when it succeeds, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForMatSingleRowFast(FastConfigHandle fastConfig_handle,
                                                             const void* data,
                                                             int64_t* out_len,
                                                             double* out_result);

/*!
 * \brief Make prediction for a new dataset presented in a form of array of pointers to rows.
 * \note
 * You should pre-allocate memory for ``out_result``:
 *   - for normal and raw score, its length is equal to ``num_class * num_data``;
 *   - for leaf index, its length is equal to ``num_class * num_data * num_iteration``;
 *   - for feature contributions, its length is equal to ``num_class * num_data * (num_feature + 1)``.
 * \param handle Handle of booster
 * \param data Pointer to the data space
 * \param data_type Type of ``data`` pointer, can be ``C_API_DTYPE_FLOAT32`` or ``C_API_DTYPE_FLOAT64``
 * \param nrow Number of rows
 * \param ncol Number columns
 * \param predict_type What should be predicted
 *   - ``C_API_PREDICT_NORMAL``: normal prediction, with transform (if needed);
 *   - ``C_API_PREDICT_RAW_SCORE``: raw score;
 *   - ``C_API_PREDICT_LEAF_INDEX``: leaf index;
 *   - ``C_API_PREDICT_CONTRIB``: feature contributions (SHAP values)
 * \param start_iteration Start index of the iteration to predict
 * \param num_iteration Number of iteration for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterPredictForMats(BoosterHandle handle,
                                                 const void** data,
                                                 int data_type,
                                                 int32_t nrow,
                                                 int32_t ncol,
                                                 int predict_type,
                                                 int start_iteration,
                                                 int num_iteration,
                                                 const char* parameter,
                                                 int64_t* out_len,
                                                 double* out_result);

/*!
 * \brief Save model into file.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be saved
 * \param num_iteration Index of the iteration that should be saved, <= 0 means save all
 * \param feature_importance_type Type of feature importance, can be ``C_API_FEATURE_IMPORTANCE_SPLIT`` or ``C_API_FEATURE_IMPORTANCE_GAIN``
 * \param filename The name of the file
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterSaveModel(BoosterHandle handle,
                                            int start_iteration,
                                            int num_iteration,
                                            int feature_importance_type,
                                            const char* filename);

/*!
 * \brief Save model to string.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be saved
 * \param num_iteration Index of the iteration that should be saved, <= 0 means save all
 * \param feature_importance_type Type of feature importance, can be ``C_API_FEATURE_IMPORTANCE_SPLIT`` or ``C_API_FEATURE_IMPORTANCE_GAIN``
 * \param buffer_len String buffer length, if ``buffer_len < out_len``, you should re-allocate buffer
 * \param[out] out_len Actual output length
 * \param[out] out_str String of model, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterSaveModelToString(BoosterHandle handle,
                                                    int start_iteration,
                                                    int num_iteration,
                                                    int feature_importance_type,
                                                    int64_t buffer_len,
                                                    int64_t* out_len,
                                                    char* out_str);

/*!
 * \brief Dump model to JSON.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be dumped
 * \param num_iteration Index of the iteration that should be dumped, <= 0 means dump all
 * \param feature_importance_type Type of feature importance, can be ``C_API_FEATURE_IMPORTANCE_SPLIT`` or ``C_API_FEATURE_IMPORTANCE_GAIN``
 * \param buffer_len String buffer length, if ``buffer_len < out_len``, you should re-allocate buffer
 * \param[out] out_len Actual output length
 * \param[out] out_str JSON format string of model, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterDumpModel(BoosterHandle handle,
                                            int start_iteration,
                                            int num_iteration,
                                            int feature_importance_type,
                                            int64_t buffer_len,
                                            int64_t* out_len,
                                            char* out_str);

/*!
 * \brief Get leaf value.
 * \param handle Handle of booster
 * \param tree_idx Index of tree
 * \param leaf_idx Index of leaf
 * \param[out] out_val Output result from the specified leaf
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetLeafValue(BoosterHandle handle,
                                               int tree_idx,
                                               int leaf_idx,
                                               double* out_val);

/*!
 * \brief Set leaf value.
 * \param handle Handle of booster
 * \param tree_idx Index of tree
 * \param leaf_idx Index of leaf
 * \param val Leaf value
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterSetLeafValue(BoosterHandle handle,
                                               int tree_idx,
                                               int leaf_idx,
                                               double val);

/*!
 * \brief Get model feature importance.
 * \param handle Handle of booster
 * \param num_iteration Number of iterations for which feature importance is calculated, <= 0 means use all
 * \param importance_type Method of importance calculation:
 *   - ``C_API_FEATURE_IMPORTANCE_SPLIT``: result contains numbers of times the feature is used in a model;
 *   - ``C_API_FEATURE_IMPORTANCE_GAIN``: result contains total gains of splits which use the feature
 * \param[out] out_results Result array with feature importance
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterFeatureImportance(BoosterHandle handle,
                                                    int num_iteration,
                                                    int importance_type,
                                                    double* out_results);

/*!
 * \brief Get model upper bound value.
 * \param handle Handle of booster
 * \param[out] out_results Result pointing to max value
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetUpperBoundValue(BoosterHandle handle,
                                                     double* out_results);

/*!
 * \brief Get model lower bound value.
 * \param handle Handle of booster
 * \param[out] out_results Result pointing to min value
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_BoosterGetLowerBoundValue(BoosterHandle handle,
                                                     double* out_results);

/*!
 * \brief Initialize the network.
 * \param machines List of machines in format 'ip1:port1,ip2:port2'
 * \param local_listen_port TCP listen port for local machines
 * \param listen_time_out Socket time-out in minutes
 * \param num_machines Total number of machines
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_NetworkInit(const char* machines,
                                       int local_listen_port,
                                       int listen_time_out,
                                       int num_machines);

/*!
 * \brief Finalize the network.
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_NetworkFree();

/*!
 * \brief Initialize the network with external collective functions.
 * \param num_machines Total number of machines
 * \param rank Rank of local machine
 * \param reduce_scatter_ext_fun The external reduce-scatter function
 * \param allgather_ext_fun The external allgather function
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int LGBM_NetworkInitWithFunctions(int num_machines,
                                                    int rank,
                                                    void* reduce_scatter_ext_fun,
                                                    void* allgather_ext_fun);


// ---- start REModel related functions

/*!
* \brief Create REModel
* \param num_data Number of data points
* \param cluster_ids_data IDs / labels indicating independent realizations of random effects / Gaussian processes (same values = same process realization)
* \param re_group_data Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
* \param num_re_group Number of grouped random effects
* \param re_group_rand_coef_data Covariate data for grouped random coefficients
* \param ind_effect_group_rand_coef Indices that relate every random coefficients to a "base" intercept grouped random effect. Counting starts at 1.
* \param num_re_group_rand_coef Number of grouped random coefficients
* \param drop_intercept_group_rand_effect Indicates whether intercept random effects are dropped (only for random coefficients). If drop_intercept_group_rand_effect[k] > 0, the intercept random effect number k is dropped. Only random effects with random slopes can be dropped.
* \param num_gp Number of Gaussian processes (intercept only, random coefficients not counting)
* \param gp_coords_data Coordinates (features) for Gaussian process
* \param dim_gp_coords Dimension of the coordinates (=number of features) for Gaussian process
* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
* \param num_gp_rand_coef Number of Gaussian process random coefficients
* \param cov_fct Type of covariance function for Gaussian process (GP)
* \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. This parameter is irrelevant for some covariance functions such as the exponential or Gaussian
* \param gp_approx Type of GP-approximation for handling large data
* \param cov_fct_taper_range Range parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
* \param cov_fct_taper_shape Shape parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
* \param num_neighbors The number of neighbors used in the Vecchia approximation
* \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
* \param num_ind_points Number of inducing points / knots for, e.g., a predictive process approximation
* \param likelihood Likelihood function for the observed response variable
* \param matrix_inversion_method Method which is used for matrix inversion
* \param seed Seed used for model creation (e.g., random ordering in Vecchia approximation)
* \param[out] out Created REModel
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_CreateREModel(int32_t num_data,
    const int32_t* cluster_ids_data,
    const char* re_group_data,
    int32_t num_re_group,
    const double* re_group_rand_coef_data,
    const int32_t* ind_effect_group_rand_coef,
    int32_t num_re_group_rand_coef,
    const int* drop_intercept_group_rand_effect,
    int32_t num_gp,
    const double* gp_coords_data,
    const int dim_gp_coords,
    const double* gp_rand_coef_data,
    int32_t num_gp_rand_coef,
    const char* cov_fct,
    double cov_fct_shape,
    const char* gp_approx,
    double cov_fct_taper_range,
    double cov_fct_taper_shape,
    int num_neighbors,
    const char* vecchia_ordering,
    const char* vecchia_pred_type,
    int num_neighbors_pred,
    int num_ind_points,
    const char* likelihood,
    const char* matrix_inversion_method,
    int seed,
    REModelHandle* out);

/*!
 * \brief Free space for REModel.
 * \param handle Handle of REModel to be freed
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT int GPB_REModelFree(REModelHandle handle);

/*!
* \brief Set configuration parameters for the optimizer
* \param handle Handle of REModel
* \param init_cov_pars Initial values for covariance parameters of RE components
* \param lr Learning rate for covariance parameters. If lr<= 0, internal default values are used (0.1 for "gradient_descent" and 1. for "fisher_scoring")
* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0)
* \param max_iter Maximal number of iterations
* \param delta_rel_conv Convergence tolerance. The algorithm stops if the relative change in eiher the log-likelihood or the parameters is below this value. For "bfgs", the L2 norm of the gradient is used instead of the relative change in the log-likelihood
* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters (only used for "gradient_descent")
* \param nesterov_schedule_version Which version of Nesterov schedule should be used (only relevant if use_nesterov_acc)
* \param trace If true, the value of the gradient is printed for some iterations
* \param optimizer Optimizer for covariance parameters
* \param momentum_offset Number of iterations for which no mometum is applied in the beginning (only relevant if use_nesterov_acc)
* \param convergence_criterion The convergence criterion used for terminating the optimization algorithm. Options: "relative_change_in_log_likelihood" or "relative_change_in_parameters"
* \param calc_std_dev If true, approximate standard deviations are calculated (= square root of diagonal of the inverse Fisher information for Gaussian likelihoods and square root of diagonal of a numerically approximated inverse Hessian for non-Gaussian likelihoods)
* \param num_covariates Number of covariates
* \param init_coef Initial values for the regression coefficients
* \param lr_coef Learning rate for fixed-effect linear coefficients
* \param acc_rate_coef Acceleration rate for coefficients for Nesterov acceleration (only relevant if nesterov_schedule_version == 0)
* \param optimizer_coef Optimizer for linear regression coefficients
* \param cg_max_num_it Maximal number of iterations for conjugate gradient algorithm
* \param cg_max_num_it_tridiag Maximal number of iterations for conjugate gradient algorithm when being run as Lanczos algorithm for tridiagonalization
* \param cg_delta_conv Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for parameter estimation
* \param num_rand_vec_trace Number of random vectors (e.g. Rademacher) for stochastic approximation of the trace of a matrix
* \param reuse_rand_vec_trace If true, random vectors (e.g. Rademacher) for stochastic approximation of the trace of a matrix are sampled only once at the beginning and then reused in later trace approximations, otherwise they are sampled everytime a trace is calculated
* \param cg_preconditioner_type Type of preconditioner used for the conjugate gradient algorithm
* \param seed_rand_vec_trace Seed number to generate random vectors (e.g. Rademacher) for stochastic approximation of the trace of a matrix
* \param piv_chol_rank Rank of the pivoted cholseky decomposition used as preconditioner of the conjugate gradient algorithm
* \param init_aux_pars Initial values for values for aux_pars_ (e.g., shape parameter of gamma likelihood)
* \param estimate_aux_pars If true, any additional parameters for non-Gaussian likelihoods are also estimated (e.g., shape parameter of gamma likelihood)
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_SetOptimConfig(REModelHandle handle,
    double* init_cov_pars,
    double lr,
    double acc_rate_cov,
    int max_iter,
    double delta_rel_conv,
    bool use_nesterov_acc,
    int nesterov_schedule_version,
    bool trace,
    const char* optimizer,
    int momentum_offset,
    const char* convergence_criterion,
    bool calc_std_dev,
    int num_covariates,
    double* init_coef,
    double lr_coef,
    double acc_rate_coef,
    const char* optimizer_coef,
    int cg_max_num_it,
    int cg_max_num_it_tridiag,
    double cg_delta_conv,
    int num_rand_vec_trace,
    bool reuse_rand_vec_trace,
    const char* cg_preconditioner_type,
    int seed_rand_vec_trace,
    int piv_chol_rank,
    double* init_aux_pars,
    bool estimate_aux_pars);

/*!
* \brief Find parameters that minimize the negative log-ligelihood (=MLE)
* \param handle Handle of REModel
* \param y_data Response variable data
* \param fixed_effects Fixed effects component of location parameter (only used for non-Gaussian data). For Gaussian data, this is ignored
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_OptimCovPar(REModelHandle handle,
    const double* y_data,
    const double* fixed_effects);

/*!
* \brief Find linear regression coefficients and covariance parameters that minimize the negative log-ligelihood (=MLE)
*		 Note: You should pre-allocate memory for optim_pars. Its length equals 1 + number of covariance parameters + number of linear regression coefficients and 1
* \param handle Handle of REModel
* \param y_data Response variable data
* \param covariate_data Covariate (=independent variable, feature) data
* \param num_covariates Number of covariates
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_OptimLinRegrCoefCovPar(REModelHandle handle,
    const double* y_data,
    const double* covariate_data,
    int num_covariates);

/*!
* \brief Calculate the value of the negative log-likelihood
* \param handle Handle of REModel
* \param y_data Response variable data
* \param cov_pars Values for covariance parameters of RE components
* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data). For Gaussian data, this is ignored
* \param[out] negll Negative log-likelihood
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_EvalNegLogLikelihood(REModelHandle handle,
    const double* y_data,
    double* cov_pars,
    const double* fixed_effects,
    double* negll);

/*!
* \brief Get the current value of the negative log-likelihood
* \param handle Handle of REModel
* \param[out] negll Negative log-likelihood
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetCurrentNegLogLikelihood(REModelHandle handle,
    double* negll);

/*!
* \brief Get covariance parameters
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariance parameters (num_cov_pars) or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param[out] optim_cov_pars Optimal covariance parameters
* \param calc_std_dev If true, standard deviations are also exported
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetCovPar(REModelHandle handle,
    double* optim_cov_pars,
    bool calc_std_dev);

/*!
* \brief Get initial values for covariance parameters
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariance parameters (num_cov_pars) or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param[out] init_cov_pars Initial covariance parameters
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetInitCovPar(REModelHandle handle,
    double* init_cov_pars);

/*!
* \brief Get / export regression coefficients
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariates or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param[out] optim_coef Optimal regression coefficients
* \param calc_std_dev If true, standard deviations are also exported
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetCoef(REModelHandle handle,
    double* optim_coef,
    bool calc_std_dev);

/*!
* \brief Get / export the number of iterations until convergence
*   Note: You should pre-allocate memory for num_it (length = 1)
* \param handle Handle of REModel
* \param[out] num_it Number of iterations for convergence
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetNumIt(REModelHandle handle,
    int* num_it);

/*!
* \brief Set the data used for making predictions (useful if the same data is used repeatedly, e.g., in validation of GPBoost)
* \param handle Handle of REModel
* \param num_data_pred Number of data points for which predictions are made
* \param cluster_ids_data_pred IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization) for which predictions are to be made
* \param re_group_data_pred Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
* \param re_group_rand_coef_data_pred Covariate data for grouped random coefficients
* \param gp_coords_data_pred Coordinates (features) for Gaussian process
* \param gp_rand_coef_data_pred Covariate data for Gaussian process random coefficients
* \param covariate_data_pred Covariate data (=independent variables, features) for prediction
* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions (-1 means that the value already set at initialization is used)
* \param cg_delta_conv_pred Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for prediction
*/
GPBOOST_C_EXPORT int GPB_SetPredictionData(REModelHandle handle,
    int32_t num_data_pred,
    const int32_t* cluster_ids_data_pred,
    const char* re_group_data_pred,
    const double* re_group_rand_coef_data_pred,
    double* gp_coords_data_pred,
    const double* gp_rand_coef_data_pred,
    const double* covariate_data_pred,
    const char* vecchia_pred_type,
    int num_neighbors_pred,
    double cg_delta_conv_pred);

/*!
* \brief Make predictions: calculate conditional mean and variances or covariance matrix
*		 Note: You should pre-allocate memory for out_predict
*			   Its length is equal to num_data_pred if only the conditional mean is predicted (predict_cov_mat==false && predict_var==false)
*			   or num_data_pred * (1 + num_data_pred) if the predictive covariance matrix is also calculated (predict_cov_mat==true)
*			   or num_data_pred * 2 if predictive variances are also calculated (predict_var==true)
* \param handle Handle of REModel
* \param y_data Response variable for observed data
* \param num_data_pred Number of data points for which predictions are made
* \param[out] out_predict Predictive mean at prediction points followed by the predictive covariance matrix in column-major format (if predict_cov_mat==true) or the predictive variances (if predict_var==true)
* \param predict_cov_mat If true, the predictive/conditional covariance matrix is calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
* \param predict_var If true, the predictive/conditional variances are calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
* \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects
* \param cluster_ids_data_pred IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization) for which predictions are to be made
* \param re_group_data_pred Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
* \param re_group_rand_coef_data_pred Covariate data for grouped random coefficients
* \param gp_coords_data_pred Coordinates (features) for Gaussian process
* \param gp_rand_coef_data_pred Covariate data for Gaussian process random coefficients
* \param cov_pars Covariance parameters of RE components
* \param covariate_data_pred Covariate data (=independent variables, features) for prediction
* \param use_saved_data If true previusly set data on groups, coordinates, and covariates are used and some arguments of this function are ignored
* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions (-1 means that the value already set at initialization is used)
* \param cg_delta_conv_pred Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for prediction
* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data). For Gaussian data, this is ignored
* \param fixed_effects_pred Fixed effects component of location parameter for predicted data (only used for non-Gaussian data)
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_PredictREModel(REModelHandle handle,
    const double* y_data,
    int32_t num_data_pred,
    double* out_predict,
    bool predict_cov_mat,
    bool predict_var,
    bool predict_response,
    const int32_t* cluster_ids_data_pred,
    const char* re_group_data_pred,
    const double* re_group_rand_coef_data_pred,
    double* gp_coords_data_pred,
    const double* gp_rand_coef_data_pred,
    const double* cov_pars,
    const double* covariate_data_pred,
    bool use_saved_data,
    const char* vecchia_pred_type,
    int num_neighbors_pred,
    double cg_delta_conv_pred,
    const double* fixed_effects,
    const double* fixed_effects_pred);

/*!
* \brief Predict ("estimate") training data random effects
* \param handle Handle of REModel
* \param cov_pars_pred Covariance parameters of components
* \param y_obs Response variable for observed data
* \param[out] out_predict Predicted training data random effects
* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data). For Gaussian data, this is ignored
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_PredictREModelTrainingDataRandomEffects(REModelHandle handle,
    const double* cov_pars_pred,
    const double* y_obs,
    double* out_predict,
    const double* fixed_effects);

/*!
* \brief Get name of likelihood
* \param handle Handle of REModel
* \param[out] out_str Likelihood name
* \param[out] num_char Number of characters
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetLikelihoodName(REModelHandle handle,
    char* out_str,
    int* num_char);

/*!
* \brief Get name of covariance parameter optimizer
* \param handle Handle of REModel
* \param[out] out_str Optimizer name
* \param[out] num_char Number of characters
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetOptimizerCovPars(REModelHandle handle,
    char* out_str,
    int* num_char);

/*!
* \brief Get name of linear regression coefficients optimizer
* \param handle Handle of REModel
* \param[out] out_str Optimizer name
* \param[out] num_char Number of characters
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetOptimizerCoef(REModelHandle handle,
    char* out_str,
    int* num_char);

/*!
* \brief Set the type of likelihood
* \param handle Handle of REModel
* \param likelihood Likelihood name
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_SetLikelihood(REModelHandle handle,
    const char* likelihood);

/*!
* \brief Return (last used) response variable data
* \param handle Handle of REModel
* \param[out] response_data Response variable data (memory needs to be preallocated)
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetResponseData(REModelHandle handle,
    double* response_data);

/*!
* \brief Return covariate data
* \param handle Handle of REModel
* \param[out] covariate_data covariate data
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetCovariateData(REModelHandle handle,
    double* covariate_data);

/*!
* \brief Get additional likelihood parameters (e.g., shape parameter for a gamma likelihood)
* \param handle Handle of REModel
* \param[out] Additional likelihood parameters (aux_pars_). This vector needs to be pre-allocated
* \param[out] out_str Name of the first parameter
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetAuxPars(REModelHandle handle,
    double* aux_pars,
    char* out_str);

/*!
* \brief Get number of additional likelihood parameters (e.g., shape parameter for a gamma likelihood)
* \param handle Handle of booster
* \param[out] num_aux_pars Number of additional likelihood parameters
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetNumAuxPars(BoosterHandle handle,
    int* num_aux_pars);

/*!
* \brief Get initial values for additional likelihood parameters (e.g., shape parameter for a gamma likelihood)
* \param handle Handle of booster
* \param[out] aux_pars Initial values for additional likelihood parameters
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_GetInitAuxPars(REModelHandle handle,
    double* aux_pars);

#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)  /*!< \brief Thread local specifier. */
#else
#define THREAD_LOCAL thread_local  /*!< \brief Thread local specifier. */
#endif

/*!
 * \brief Handle of error message.
 * \return Error message
 */
static char* LastErrorMsg() { static THREAD_LOCAL char err_msg[512] = "Everything is fine"; return err_msg; }

#ifdef _MSC_VER
  #pragma warning(disable : 4996)
#endif
/*!
 * \brief Set string message of the last error.
 * \param msg Error message
 */
inline void LGBM_SetLastError(const char* msg) {
  const int err_buf_len = 512;
  snprintf(LastErrorMsg(), err_buf_len, "%s", msg);
}

#endif  // LIGHTGBM_C_API_H_
