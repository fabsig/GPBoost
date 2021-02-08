/*!
* Original work Copyright (c) 2017 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef GPBOOST_R_H_
#define GPBOOST_R_H_

#include <LightGBM/c_api.h>
#include <GPBoost/gpboost_c_api.h>
#include "R_object_helper.h"

/*!
* \brief get string message of the last error
*  all functions in this file will return 0 on success
*  and -1 when an error occurred
* \return err_msg error information
* \return error information
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_GetLastError_R(
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE err_msg
);

// --- start Dataset interface

/*!
* \brief load data set from file like the command_line LightGBM does
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetCreateFromFile_R(
	LGBM_SE filename,
	LGBM_SE parameters,
	LGBM_SE reference,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief create a dataset from CSC format
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param nindptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetCreateFromCSC_R(
	LGBM_SE indptr,
	LGBM_SE indices,
	LGBM_SE data,
	LGBM_SE nindptr,
	LGBM_SE nelem,
	LGBM_SE num_row,
	LGBM_SE parameters,
	LGBM_SE reference,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief create dataset from dense matrix
* \param data matric data
* \param nrow number of rows
* \param ncol number columns
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetCreateFromMat_R(
	LGBM_SE data,
	LGBM_SE nrow,
	LGBM_SE ncol,
	LGBM_SE parameters,
	LGBM_SE reference,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief Create subset of a data
* \param handle handle of full dataset
* \param used_row_indices Indices used in subset
* \param len_used_row_indices length of Indices used in subset
* \param parameters additional parameters
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetGetSubset_R(
	LGBM_SE handle,
	LGBM_SE used_row_indices,
	LGBM_SE len_used_row_indices,
	LGBM_SE parameters,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetSetFeatureNames_R(
	LGBM_SE handle,
	LGBM_SE feature_names,
	LGBM_SE call_state
);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetGetFeatureNames_R(
	LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE feature_names,
	LGBM_SE call_state
);

/*!
* \brief save dataset to binary file
* \param handle an instance of dataset
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetSaveBinary_R(
	LGBM_SE handle,
	LGBM_SE filename,
	LGBM_SE call_state
);

/*!
* \brief free dataset
* \param handle an instance of dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetFree_R(
	LGBM_SE handle,
	LGBM_SE call_state
);

/*!
* \brief set vector to a content in info
*        Note: group and group_id only work for C_API_DTYPE_INT32
*              label and weight only work for C_API_DTYPE_FLOAT32
* \param handle an instance of dataset
* \param field_name field name, can be label, weight, group, group_id
* \param field_data pointer to vector
* \param num_element number of element in field_data
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetSetField_R(
	LGBM_SE handle,
	LGBM_SE field_name,
	LGBM_SE field_data,
	LGBM_SE num_element,
	LGBM_SE call_state
);

/*!
* \brief get size of info vector from dataset
* \param handle an instance of dataset
* \param field_name field name
* \param out size of info vector from dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetGetFieldSize_R(
	LGBM_SE handle,
	LGBM_SE field_name,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief get info vector from dataset
* \param handle an instance of dataset
* \param field_name field name
* \param field_data pointer to vector
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetGetField_R(
	LGBM_SE handle,
	LGBM_SE field_name,
	LGBM_SE field_data,
	LGBM_SE call_state
);

/*!
 * \brief Raise errors for attempts to update dataset parameters
 * \param old_params Current dataset parameters
 * \param new_params New dataset parameters
 * \return 0 when succeed, -1 when failure happens
 */
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetUpdateParamChecking_R(
	LGBM_SE old_params,
	LGBM_SE new_params,
	LGBM_SE call_state
);

/*!
* \brief get number of data.
* \param handle the handle to the dataset
* \param out The address to hold number of data
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetGetNumData_R(
	LGBM_SE handle,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief get number of features
* \param handle the handle to the dataset
* \param out The output of number of features
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_DatasetGetNumFeature_R(
	LGBM_SE handle,
	LGBM_SE out,
	LGBM_SE call_state
);

// --- start Booster interfaces

/*!
* \brief create a new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterCreate_R(
	LGBM_SE train_data,
	LGBM_SE parameters,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief create a new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \param re_model Gaussian process model
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_GPBoosterCreate_R(
	LGBM_SE train_data,
	LGBM_SE parameters,
	LGBM_SE re_model,
	LGBM_SE out,
	LGBM_SE call_state);

/*!
* \brief free obj in handle
* \param handle handle to be freed
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterFree_R(
	LGBM_SE handle,
	LGBM_SE call_state
);

/*!
* \brief load an existing boosting from model file
* \param filename filename of model
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterCreateFromModelfile_R(
	LGBM_SE filename,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief load an existing boosting from model_str
* \param model_str string containing the model
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterLoadModelFromString_R(
	LGBM_SE model_str,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief Merge model in two boosters to first handle
* \param handle handle, will merge other handle to this
* \param other_handle
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterMerge_R(
	LGBM_SE handle,
	LGBM_SE other_handle,
	LGBM_SE call_state
);

/*!
* \brief Add new validation to booster
* \param handle handle
* \param valid_data validation data set
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterAddValidData_R(
	LGBM_SE handle,
	LGBM_SE valid_data,
	LGBM_SE call_state
);

/*!
* \brief Reset training data for booster
* \param handle handle
* \param train_data training data set
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterResetTrainingData_R(
	LGBM_SE handle,
	LGBM_SE train_data,
	LGBM_SE call_state
);

/*!
* \brief Reset config for current booster
* \param handle handle
* \param parameters format: 'key1=value1 key2=value2'
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterResetParameter_R(
	LGBM_SE handle,
	LGBM_SE parameters,
	LGBM_SE call_state
);

/*!
* \brief Get number of classes
* \param handle handle
* \param out number of classes
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterGetNumClasses_R(
	LGBM_SE handle,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief update the model in one round
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterUpdateOneIter_R(
	LGBM_SE handle,
	LGBM_SE call_state
);

/*!
* \brief update the model, by directly specify gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \param len length of grad/hess
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterUpdateOneIterCustom_R(
	LGBM_SE handle,
	LGBM_SE grad,
	LGBM_SE hess,
	LGBM_SE len,
	LGBM_SE call_state
);

/*!
* \brief Rollback one iteration
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterRollbackOneIter_R(
	LGBM_SE handle,
	LGBM_SE call_state
);

/*!
* \brief Get iteration of current boosting rounds
* \param out iteration of boosting rounds
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterGetCurrentIteration_R(
	LGBM_SE handle,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief Get model upper bound value.
* \param handle Handle of booster
* \param[out] out_results Result pointing to max value
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterGetUpperBoundValue_R(
	LGBM_SE handle,
	LGBM_SE out_result,
	LGBM_SE call_state
);

/*!
* \brief Get model lower bound value.
* \param handle Handle of booster
* \param[out] out_results Result pointing to min value
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterGetLowerBoundValue_R(
	LGBM_SE handle,
	LGBM_SE out_result,
	LGBM_SE call_state
);

/*!
* \brief Get Name of eval
* \param eval_names eval names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterGetEvalNames_R(
	LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE eval_names,
	LGBM_SE call_state
);

/*!
* \brief get evaluation for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result float array contains result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterGetEval_R(
	LGBM_SE handle,
	LGBM_SE data_idx,
	LGBM_SE out_result,
	LGBM_SE call_state
);

/*!
* \brief Get number of prediction for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out size of predict
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterGetNumPredict_R(
	LGBM_SE handle,
	LGBM_SE data_idx,
	LGBM_SE out,
	LGBM_SE call_state
);

/*!
* \brief Get prediction for training data and validation data.
*        This can be used to support customized eval function
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result, used to store predict result, should pre-allocate memory
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterGetPredict_R(
	LGBM_SE handle,
	LGBM_SE data_idx,
	LGBM_SE out_result,
	LGBM_SE call_state
);

/*!
* \brief make prediction for file
* \param handle handle
* \param data_filename filename of data file
* \param data_has_header data file has header or not
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \return 0 when succeed, -1 when failure happens
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterPredictForFile_R(
	LGBM_SE handle,
	LGBM_SE data_filename,
	LGBM_SE data_has_header,
	LGBM_SE is_rawscore,
	LGBM_SE is_leafidx,
	LGBM_SE is_predcontrib,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE parameter,
	LGBM_SE result_filename,
	LGBM_SE call_state
);

/*!
* \brief Get number of prediction
* \param handle handle
* \param num_row
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out_len length of prediction
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterCalcNumPredict_R(
	LGBM_SE handle,
	LGBM_SE num_row,
	LGBM_SE is_rawscore,
	LGBM_SE is_leafidx,
	LGBM_SE is_predcontrib,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE out_len,
	LGBM_SE call_state
);

/*!
* \brief make prediction for a new data set
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param nindptr number of cols in the matrix + 1
* \param nelem number of non-zero elements in the matrix
* \param num_row number of rows
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out prediction result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterPredictForCSC_R(
	LGBM_SE handle,
	LGBM_SE indptr,
	LGBM_SE indices,
	LGBM_SE data,
	LGBM_SE nindptr,
	LGBM_SE nelem,
	LGBM_SE num_row,
	LGBM_SE is_rawscore,
	LGBM_SE is_leafidx,
	LGBM_SE is_predcontrib,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE parameter,
	LGBM_SE out_result,
	LGBM_SE call_state
);

/*!
* \brief make prediction for a new data set
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param data pointer to the data space
* \param nrow number of rows
* \param ncol number columns
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out prediction result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterPredictForMat_R(
	LGBM_SE handle,
	LGBM_SE data,
	LGBM_SE nrow,
	LGBM_SE ncol,
	LGBM_SE is_rawscore,
	LGBM_SE is_leafidx,
	LGBM_SE is_predcontrib,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE parameter,
	LGBM_SE out_result,
	LGBM_SE call_state
);

/*!
* \brief save model into file
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterSaveModel_R(
	LGBM_SE handle,
	LGBM_SE num_iteration,
	LGBM_SE feature_importance_type,
	LGBM_SE filename,
	LGBM_SE call_state
);

/*!
* \brief create string containing model
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param out_str string of model
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterSaveModelToString_R(
	LGBM_SE handle,
	LGBM_SE num_iteration,
	LGBM_SE feature_importance_type,
	LGBM_SE buffer_len,
	LGBM_SE actual_len,
	LGBM_SE out_str,
	LGBM_SE call_state
);

/*!
* \brief dump model to json
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param out_str json format string of model
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE LGBM_BoosterDumpModel_R(
	LGBM_SE handle,
	LGBM_SE num_iteration,
	LGBM_SE feature_importance_type,
	LGBM_SE buffer_len,
	LGBM_SE actual_len,
	LGBM_SE out_str,
	LGBM_SE call_state
);

// Below here are REModel / GPModel related functions

/*!
* \brief Create REModel
* \param ndata Number of data points
* \param cluster_ids_data IDs / labels indicating independent realizations of random effects / Gaussian processes (same values = same process realization)
* \param re_group_data Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
* \param num_re_group Number of grouped random effects
* \param re_group_rand_coef_data Covariate data for grouped random coefficients
* \param ind_effect_group_rand_coef Indices that relate every random coefficients to a "base" intercept grouped random effect. Counting starts at 1.
* \param num_re_group_rand_coef Number of grouped random coefficients
* \param num_gp Number of Gaussian processes (intercept only, random coefficients not counting)
* \param gp_coords_data Coordinates (features) for Gaussian process
* \param dim_gp_coords Dimension of the coordinates (=number of features) for Gaussian process
* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
* \param num_gp_rand_coef Number of Gaussian process random coefficients
* \param cov_fct Type of covariance (kernel) function for Gaussian processes. We follow the notation and parametrization of Diggle and Ribeiro (2007) except for the Matern covariance where we follow Rassmusen and Williams (2006)
* \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern covariance, irrelevant for some covariance functions such as the exponential or Gaussian)
* \param vecchia_approx If true, the Veccia approximation is used for the Gaussian process
* \param num_neighbors The number of neighbors used in the Vecchia approximation
* \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
* \param likelihood Likelihood function for the observed response variable. Default = "gaussian"
* \param out Created REModel
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_CreateREModel_R(
	LGBM_SE ndata,
	LGBM_SE cluster_ids_data,
	LGBM_SE re_group_data,
	LGBM_SE num_re_group,
	LGBM_SE re_group_rand_coef_data,
	LGBM_SE ind_effect_group_rand_coef,
	LGBM_SE num_re_group_rand_coef,
	LGBM_SE num_gp,
	LGBM_SE gp_coords_data,
	LGBM_SE dim_gp_coords,
	LGBM_SE gp_rand_coef_data,
	LGBM_SE num_gp_rand_coef,
	LGBM_SE cov_fct,
	LGBM_SE cov_fct_shape,
	LGBM_SE vecchia_approx,
	LGBM_SE num_neighbors,
	LGBM_SE vecchia_ordering,
	LGBM_SE vecchia_pred_type,
	LGBM_SE num_neighbors_pred,
	LGBM_SE likelihood,
	LGBM_SE out,
	LGBM_SE call_state);

/*!
* \brief free obj in handle
* \param handle handle of REModel
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_REModelFree_R(
	LGBM_SE handle,
	LGBM_SE call_state);

/*!
* \brief Set configuration parameters for the optimizer
* \param handle Handle of REModel
* \param init_cov_pars Initial values for covariance parameters of RE components
* \param lr Learning rate. If <= 0, default values are used. Default value = 0.01 for "gradient_descent" and 1. for "fisher_scoring"
* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
* \param max_iter Maximal number of iterations
* \param delta_rel_conv Convergence criterion: stop iteration if relative change is below this value
* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters. Default = true
* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
* \param trace If true, the value of the gradient is printed for some iterations. Default = false
* \param optimizer Options: "gradient_descent" or "fisher_scoring"
* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
* \param convergence_criterion The convergence criterion used for terminating the optimization algorithm. Options: "relative_change_in_log_likelihood" (default) or "relative_change_in_parameters"
* \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_SetOptimConfig_R(
	LGBM_SE handle,
	LGBM_SE init_cov_pars,
	LGBM_SE lr,
	LGBM_SE acc_rate_cov,
	LGBM_SE max_iter,
	LGBM_SE delta_rel_conv,
	LGBM_SE use_nesterov_acc,
	LGBM_SE nesterov_schedule_version,
	LGBM_SE trace,
	LGBM_SE optimizer,
	LGBM_SE momentum_offset,
	LGBM_SE convergence_criterion,
	LGBM_SE calc_std_dev,
	LGBM_SE call_state);

/*!
* \brief Set configuration parameters for the optimizer for linear regression coefficients
* \param handle Handle of REModel
* \param num_covariates Number of covariates
* \param init_coef Initial values for the regression coefficients
* \param lr_coef Learning rate for fixed-effect linear coefficients
* \param acc_rate_coef Acceleration rate for coefficients for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
* \param optimizer Options: "gradient_descent" or "wls" (coordinate descent using weighted least squares)
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_SetOptimCoefConfig_R(
	LGBM_SE handle,
	LGBM_SE num_covariates,
	LGBM_SE init_coef,
	LGBM_SE lr_coef,
	LGBM_SE acc_rate_coef,
	LGBM_SE optimizer,
	LGBM_SE call_state);

/*!
* \brief Find parameters that minimize the negative log-ligelihood (=MLE)
* \param handle Handle of REModel
* \param y_data Response variable data
* \param fixed_effects Fixed effects component F of location parameter (only used for non-Gaussian data). For Gaussian data, this is ignored
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_OptimCovPar_R(
	LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE fixed_effects,
	LGBM_SE call_state);

/*!
* \brief Find parameters that minimize the negative log-ligelihood (=MLE)
*		 Note: You should pre-allocate memory for optim_pars. Its length equals 1 + number of covariance parameters + number of linear regression coefficients and 1
* \param handle Handle of REModel
* \param y_data Response variable data
* \param covariate_data Covariate (=independent variable, feature) data
* \param num_covariates Number of covariates
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_OptimLinRegrCoefCovPar_R(
	LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE covariate_data,
	LGBM_SE num_covariates,
	LGBM_SE call_state);

/*!
* \brief Calculate the value of the negative log-likelihood
* \param handle Handle of REModel
* \param y_data Response variable data
* \param cov_pars Values for covariance parameters of RE components
* \param[out] negll Negative log-likelihood
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_EvalNegLogLikelihood_R(
	LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE cov_pars,
	LGBM_SE negll,
	LGBM_SE call_state);

/*!
* \brief Get covariance paramters
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariance parameters (num_cov_pars) or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param calc_std_dev If true, standard deviations are also exported
* \param[out] optim_cov_pars Optimal covariance parameters
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_GetCovPar_R(
	LGBM_SE handle,
	LGBM_SE calc_std_dev,
	LGBM_SE optim_cov_pars,
	LGBM_SE call_state);

/*!
* \brief Get initial values for covariance paramters
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariance parameters (num_cov_pars) or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param[out] init_cov_pars Initial covariance parameters
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_GetInitCovPar_R(
	LGBM_SE handle,
	LGBM_SE init_cov_pars,
	LGBM_SE call_state);

/*!
* \brief Get / export regression coefficients
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariates or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param calc_std_dev If true, standard deviations are also exported
* \param[out] optim_coef Optimal regression coefficients
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_GetCoef_R(
	LGBM_SE handle,
	LGBM_SE calc_std_dev,
	LGBM_SE optim_coef,
	LGBM_SE call_state);

/*!
* \brief Get / export the number of iterations until convergence
*   Note: You should pre-allocate memory for num_it (length = 1)
* \param handle Handle of REModel
* \param[out] num_it Number of iterations for convergence
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_GetNumIt_R(
	LGBM_SE handle,
	LGBM_SE num_it,
	LGBM_SE call_state);

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
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_SetPredictionData_R(
	LGBM_SE handle,
	LGBM_SE num_data_pred,
	LGBM_SE cluster_ids_data_pred,
	LGBM_SE re_group_data_pred,
	LGBM_SE re_group_rand_coef_data_pred,
	LGBM_SE gp_coords_data_pred,
	LGBM_SE gp_rand_coef_data_pred,
	LGBM_SE covariate_data_pred,
	LGBM_SE call_state);

/*!
* \brief Make predictions: calculate conditional mean and variances or covariance matrix
*		 Note: You should pre-allocate memory for out_predict
*			   Its length is equal to num_data_pred if only the conditional mean is predicted (predict_cov_mat==false && predict_var==false)
*			   or num_data_pred * (1 + num_data_pred) if the predictive covariance matrix is also calculated (predict_cov_mat==true)
*			   or num_data_pred * 2 if predictive variances are also calculated (predict_var==true)
* \param handle Handle of REModel
* \param y_data Response variable for observed data
* \param num_data_pred Number of data points for which predictions are made
* \param predict_cov_mat If true, the predictive/conditional covariance matrix is calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
* \param predict_var If true, the predictive/conditional variances are calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
* \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects (this is only relevant for non-Gaussian data) (default=false)
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
* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data)
* \param fixed_effects_pred Fixed effects component of location parameter for predicted data (only used for non-Gaussian data)
* \param[out] out_predict Predictive/conditional mean at prediciton points followed by the predictive covariance matrix in column-major format (if predict_cov_mat==true) or the predictive variances (if predict_var==true)
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_PredictREModel_R(
	LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE num_data_pred,
	LGBM_SE predict_cov_mat,
	LGBM_SE predict_var,
	LGBM_SE predict_response,
	LGBM_SE cluster_ids_data_pred,
	LGBM_SE re_group_data_pred,
	LGBM_SE re_group_rand_coef_data_pred,
	LGBM_SE gp_coords_pred,
	LGBM_SE gp_rand_coef_data_pred,
	LGBM_SE cov_pars,
	LGBM_SE covariate_data_pred,
	LGBM_SE use_saved_data,
	LGBM_SE vecchia_pred_type,
	LGBM_SE num_neighbors_pred,
	LGBM_SE fixed_effects,
	LGBM_SE fixed_effects_pred,
	LGBM_SE out_predict,
	LGBM_SE call_state);

/*!
* \brief Get name of likelihood
* \param ll_name Likelihood name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_GetLikelihoodName_R(
	LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE ll_name,
	LGBM_SE call_state);

/*!
* \brief Get name of covariance parameter optimizer
* \param name Optimizer name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_GetOptimizerCovPars_R(
	LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE name,
	LGBM_SE call_state);

/*!
* \brief Get name of linear regression coefficients optimizer
* \param name Optimizer name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_GetOptimizerCoef_R(
	LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE name,
	LGBM_SE call_state);

/*!
* \brief Set the type of likelihood
* \param likelihood Likelihood name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT LGBM_SE GPB_SetLikelihood_R(
	LGBM_SE handle,
	LGBM_SE likelihood,
	LGBM_SE call_state);

#endif  // GPBOOST_R_H_
