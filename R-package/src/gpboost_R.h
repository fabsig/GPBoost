/*!
* Original work Copyright (c) 2017 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef GPBOOST_R_H_
#define GPBOOST_R_H_

#include <LightGBM/c_api.h>

#define R_NO_REMAP
#define R_USE_C99_IN_CXX
#include <Rinternals.h>

inline double* R_REAL_PTR(SEXP x) {
	if (Rf_isNull(x)) {
		return nullptr;
	}
	else {
		return REAL(x);
	}
}

inline int* R_INT_PTR(SEXP x) {
	if (Rf_isNull(x)) {
		return nullptr;
	}
	else {
		return INTEGER(x);
	}
}

inline const char* R_CHAR_PTR_FROM_RAW(SEXP x) {
	if (Rf_isNull(x)) {
		return nullptr;
	}
	else {
		return (reinterpret_cast<const char*>(RAW(x)));
	}
}

/*!
* \brief check if an R external pointer (like a Booster or Dataset handle) is a null pointer
* \param handle handle for a Booster, Dataset, or Predictor
* \return R logical, TRUE if the handle is a null pointer
*/
GPBOOST_C_EXPORT SEXP LGBM_HandleIsNull_R(
	SEXP handle
);

// --- start Dataset interface

/*!
* \brief load data set from file like the command_line LightGBM does
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \return Dataset handle
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetCreateFromFile_R(
	SEXP filename,
	SEXP parameters,
	SEXP reference
);

/*!
* \brief create a dataset from CSC format
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param num_indptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \return Dataset handle
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetCreateFromCSC_R(
	SEXP indptr,
	SEXP indices,
	SEXP data,
	SEXP num_indptr,
	SEXP nelem,
	SEXP num_row,
	SEXP parameters,
	SEXP reference
);

/*!
* \brief create dataset from dense matrix
* \param data matric data
* \param num_row number of rows
* \param ncol number columns
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \return Dataset handle
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetCreateFromMat_R(
	SEXP data,
	SEXP num_row,
	SEXP ncol,
	SEXP parameters,
	SEXP reference
);

/*!
* \brief Create subset of a data
* \param handle handle of full dataset
* \param used_row_indices Indices used in subset
* \param len_used_row_indices length of Indices used in subset
* \param parameters additional parameters
* \return Dataset handle
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetGetSubset_R(
	SEXP handle,
	SEXP used_row_indices,
	SEXP len_used_row_indices,
	SEXP parameters
);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return R character vector of feature names
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetSetFeatureNames_R(
	SEXP handle,
	SEXP feature_names
);

/*!
* \brief get feature names from Dataset
* \param handle Dataset handle
* \return an R character vector with feature names from the Dataset or NULL if no feature names
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetGetFeatureNames_R(
	SEXP handle
);

/*!
* \brief save dataset to binary file
* \param handle an instance of dataset
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetSaveBinary_R(
	SEXP handle,
	SEXP filename
);

/*!
* \brief free dataset
* \param handle an instance of dataset
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetFree_R(
	SEXP handle
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
GPBOOST_C_EXPORT SEXP LGBM_DatasetSetField_R(
	SEXP handle,
	SEXP field_name,
	SEXP field_data,
	SEXP num_element
);

/*!
* \brief get size of info vector from dataset
* \param handle an instance of dataset
* \param field_name field name
* \param out size of info vector from dataset
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetGetFieldSize_R(
	SEXP handle,
	SEXP field_name,
	SEXP out
);

/*!
* \brief get info vector from dataset
* \param handle an instance of dataset
* \param field_name field name
* \param field_data pointer to vector
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetGetField_R(
	SEXP handle,
	SEXP field_name,
	SEXP field_data
);

/*!
 * \brief Raise errors for attempts to update dataset parameters
 * \param old_params Current dataset parameters
 * \param new_params New dataset parameters
 * \return 0 when succeed, -1 when failure happens
 */
GPBOOST_C_EXPORT SEXP LGBM_DatasetUpdateParamChecking_R(
	SEXP old_params,
	SEXP new_params
);

/*!
* \brief get number of data.
* \param handle the handle to the dataset
* \param out The address to hold number of data
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetGetNumData_R(
	SEXP handle,
	SEXP out
);

/*!
* \brief get number of features
* \param handle the handle to the dataset
* \param out The output of number of features
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_DatasetGetNumFeature_R(
	SEXP handle,
	SEXP out
);

// --- start Booster interfaces

/*!
* \brief create a new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \return Booster handle
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterCreate_R(
	SEXP train_data,
	SEXP parameters
);

/*!
* \brief create a new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \param re_model Gaussian process model
* \return Booster handle
*/
GPBOOST_C_EXPORT SEXP LGBM_GPBoosterCreate_R(
	SEXP train_data,
	SEXP parameters,
	SEXP re_model
);

/*!
* \brief free obj in handle
* \param handle handle to be freed
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterFree_R(
	SEXP handle
);

/*!
* \brief load an existing boosting from model file
* \param filename filename of model
* \return Booster handle
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterCreateFromModelfile_R(
	SEXP filename
);

/*!
* \brief load an existing boosting from model_str
* \param model_str string containing the model
* \return Booster handle
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterLoadModelFromString_R(
	SEXP model_str
);

/*!
* \brief Merge model in two boosters to first handle
* \param handle handle, will merge other handle to this
* \param other_handle
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterMerge_R(
	SEXP handle,
	SEXP other_handle
);

/*!
* \brief Add new validation to booster
* \param handle handle
* \param valid_data validation data set
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterAddValidData_R(
	SEXP handle,
	SEXP valid_data
);

/*!
* \brief Reset training data for booster
* \param handle handle
* \param train_data training data set
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterResetTrainingData_R(
	SEXP handle,
	SEXP train_data
);

/*!
* \brief Reset config for current booster
* \param handle handle
* \param parameters format: 'key1=value1 key2=value2'
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterResetParameter_R(
	SEXP handle,
	SEXP parameters
);

/*!
* \brief Get number of classes
* \param handle handle
* \param out number of classes
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterGetNumClasses_R(
	SEXP handle,
	SEXP out
);

/*!
* \brief update the model in one round
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterUpdateOneIter_R(
	SEXP handle
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
GPBOOST_C_EXPORT SEXP LGBM_BoosterUpdateOneIterCustom_R(
	SEXP handle,
	SEXP grad,
	SEXP hess,
	SEXP len
);

/*!
* \brief Rollback one iteration
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterRollbackOneIter_R(
	SEXP handle
);

/*!
* \brief Get iteration of current boosting rounds
* \param out iteration of boosting rounds
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterGetCurrentIteration_R(
	SEXP handle,
	SEXP out
);

/*!
* \brief Get model upper bound value.
* \param handle Handle of booster
* \param[out] out_results Result pointing to max value
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterGetUpperBoundValue_R(
	SEXP handle,
	SEXP out_result
);

/*!
* \brief Get model lower bound value.
* \param handle Handle of booster
* \param[out] out_results Result pointing to min value
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterGetLowerBoundValue_R(
	SEXP handle,
	SEXP out_result
);

/*!
* \brief Get names of eval metrics
* \param handle Handle of booster
* \return R character vector with names of eval metrics
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterGetEvalNames_R(
	SEXP handle
);

/*!
* \brief get evaluation for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result float array contains result
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterGetEval_R(
	SEXP handle,
	SEXP data_idx,
	SEXP out_result
);

/*!
* \brief Get number of prediction for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out size of predict
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterGetNumPredict_R(
	SEXP handle,
	SEXP data_idx,
	SEXP out
);

/*!
* \brief Get prediction for training data and validation data.
*        This can be used to support customized eval function
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result, used to store predict result, should pre-allocate memory
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterGetPredict_R(
	SEXP handle,
	SEXP data_idx,
	SEXP out_result
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
GPBOOST_C_EXPORT SEXP LGBM_BoosterPredictForFile_R(
	SEXP handle,
	SEXP data_filename,
	SEXP data_has_header,
	SEXP is_rawscore,
	SEXP is_leafidx,
	SEXP is_predcontrib,
	SEXP start_iteration,
	SEXP num_iteration,
	SEXP parameter,
	SEXP result_filename
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
GPBOOST_C_EXPORT SEXP LGBM_BoosterCalcNumPredict_R(
	SEXP handle,
	SEXP num_row,
	SEXP is_rawscore,
	SEXP is_leafidx,
	SEXP is_predcontrib,
	SEXP start_iteration,
	SEXP num_iteration,
	SEXP out_len
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
* \param num_indptr number of cols in the matrix + 1
* \param nelem number of non-zero elements in the matrix
* \param num_row number of rows
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out prediction result
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterPredictForCSC_R(
	SEXP handle,
	SEXP indptr,
	SEXP indices,
	SEXP data,
	SEXP num_indptr,
	SEXP nelem,
	SEXP num_row,
	SEXP is_rawscore,
	SEXP is_leafidx,
	SEXP is_predcontrib,
	SEXP start_iteration,
	SEXP num_iteration,
	SEXP parameter,
	SEXP out_result
);

/*!
* \brief make prediction for a new data set
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param data pointer to the data space
* \param num_row number of rows
* \param ncol number columns
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out prediction result
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterPredictForMat_R(
	SEXP handle,
	SEXP data,
	SEXP num_row,
	SEXP ncol,
	SEXP is_rawscore,
	SEXP is_leafidx,
	SEXP is_predcontrib,
	SEXP start_iteration,
	SEXP num_iteration,
	SEXP parameter,
	SEXP out_result
);

/*!
* \brief save model into file
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterSaveModel_R(
	SEXP handle,
	SEXP num_iteration,
	SEXP feature_importance_type,
	SEXP filename
);

/*!
* \brief create string containing model
* \param handle Booster handle
* \param start_iteration Start index of the iteration that should be saved
* \param num_iteration, <= 0 means save all
* \param feature_importance_type type of feature importance, 0: split, 1: gain
* \return R character vector (length=1) with model string
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterSaveModelToString_R(
	SEXP handle,
	SEXP start_iteration,
	SEXP num_iteration,
	SEXP feature_importance_type
);

/*!
* \brief dump model to JSON
* \param handle Booster handle
* \param num_iteration, <= 0 means save all
* \param feature_importance_type type of feature importance, 0: split, 1: gain
* \return R character vector (length=1) with model JSON
*/
GPBOOST_C_EXPORT SEXP LGBM_BoosterDumpModel_R(
	SEXP handle,
	SEXP num_iteration,
	SEXP feature_importance_type
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
* \param drop_intercept_group_rand_effect Indicates whether intercept random effects are dropped (only for random coefficients). If drop_intercept_group_rand_effect[k] > 0, the intercept random effect number k is dropped. Only random effects with random slopes can be dropped.
* \param num_gp Number of Gaussian processes (intercept only, random coefficients not counting)
* \param gp_coords_data Coordinates (features) for Gaussian process
* \param dim_gp_coords Dimension of the coordinates (=number of features) for Gaussian process
* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
* \param num_gp_rand_coef Number of Gaussian process random coefficients
* \param cov_fct Type of covariance (kernel) function for Gaussian processes. We follow the notation and parametrization of Diggle and Ribeiro (2007) except for the Matern covariance where we follow Rassmusen and Williams (2006)
* \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. For the Wendland covariance function, we follow the notation of Bevilacqua et al. (2018)). This parameter is irrelevant for some covariance functions such as the exponential or Gaussian.
* \param cov_fct_taper_range Range parameter of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
* \param vecchia_approx If true, the Veccia approximation is used for the Gaussian process
* \param num_neighbors The number of neighbors used in the Vecchia approximation
* \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
* \param likelihood Likelihood function for the observed response variable. Default = "gaussian"
* \return REModel handle
*/
GPBOOST_C_EXPORT SEXP GPB_CreateREModel_R(
	SEXP ndata,
	SEXP cluster_ids_data,
	SEXP re_group_data,
	SEXP num_re_group,
	SEXP re_group_rand_coef_data,
	SEXP ind_effect_group_rand_coef,
	SEXP num_re_group_rand_coef,
	SEXP drop_intercept_group_rand_effect,
	SEXP num_gp,
	SEXP gp_coords_data,
	SEXP dim_gp_coords,
	SEXP gp_rand_coef_data,
	SEXP num_gp_rand_coef,
	SEXP cov_fct,
	SEXP cov_fct_shape,
	SEXP cov_fct_taper_range,
	SEXP vecchia_approx,
	SEXP num_neighbors,
	SEXP vecchia_ordering,
	SEXP vecchia_pred_type,
	SEXP num_neighbors_pred,
	SEXP likelihood
);

/*!
* \brief free obj in handle
* \param handle handle of REModel
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_REModelFree_R(
	SEXP handle
);

/*!
* \brief Set configuration parameters for the optimizer
* \param handle Handle of REModel
* \param init_cov_pars Initial values for covariance parameters of RE components
* \param lr Learning rate. If <= 0, default values are used. Default value = 0.01 for "gradient_descent" and 1. for "fisher_scoring"
* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
* \param max_iter Maximal number of iterations
* \param delta_rel_conv Convergence tolerance. The algorithm stops if the relative change in eiher the log-likelihood or the parameters is below this value. For "bfgs", the L2 norm of the gradient is used instead of the relative change in the log-likelihood
* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters. Default = true
* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
* \param trace If true, the value of the gradient is printed for some iterations. Default = false
* \param optimizer Options: "gradient_descent" or "fisher_scoring"
* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
* \param convergence_criterion The convergence criterion used for terminating the optimization algorithm. Options: "relative_change_in_log_likelihood" (default) or "relative_change_in_parameters"
* \param calc_std_dev If true, approximate standard deviations are calculated (= square root of diagonal of the inverse Fisher information for Gaussian likelihoods and square root of diagonal of a numerically approximated inverse Hessian for non-Gaussian likelihoods)
* \param num_covariates Number of covariates
* \param init_coef Initial values for the regression coefficients
* \param lr_coef Learning rate for fixed-effect linear coefficients
* \param acc_rate_coef Acceleration rate for coefficients for Nesterov acceleration (only relevant if nesterov_schedule_version == 0)
* \param optimizer_coef Optimizer for linear regression coefficients
* \param matrix_inversion_method Method which is used for matrix inversion
* \param cg_max_num_it Maximal number of iterations for conjugate gradient algorithm
* \param cg_max_num_it_tridiag Maximal number of iterations for conjugate gradient algorithm when being run as Lanczos algorithm for tridiagonalization
* \param cg_delta_conv Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for parameter estimation
* \param num_rand_vec_trace Number of random vectors (e.g. Rademacher) for stochastic approximation of the trace of a matrix
* \param reuse_rand_vec_trace If true, random vectors (e.g. Rademacher) for stochastic approximation of the trace of a matrix are sampled only once at the beginning and then reused in later trace approximations, otherwise they are sampled everytime a trace is calculated
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_SetOptimConfig_R(
	SEXP handle,
	SEXP init_cov_pars,
	SEXP lr,
	SEXP acc_rate_cov,
	SEXP max_iter,
	SEXP delta_rel_conv,
	SEXP use_nesterov_acc,
	SEXP nesterov_schedule_version,
	SEXP trace,
	SEXP optimizer,
	SEXP momentum_offset,
	SEXP convergence_criterion,
	SEXP calc_std_dev,
	SEXP num_covariates,
	SEXP init_coef,
	SEXP lr_coef,
	SEXP acc_rate_coef,
	SEXP optimizer_coef,
	SEXP matrix_inversion_method,
	SEXP cg_max_num_it,
	SEXP cg_max_num_it_tridiag,
	SEXP cg_delta_conv,
	SEXP num_rand_vec_trace,
	SEXP reuse_rand_vec_trace
);

/*!
* \brief Find parameters that minimize the negative log-ligelihood (=MLE)
* \param handle Handle of REModel
* \param y_data Response variable data
* \param fixed_effects Fixed effects component F of location parameter (only used for non-Gaussian data). For Gaussian data, this is ignored
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_OptimCovPar_R(
	SEXP handle,
	SEXP y_data,
	SEXP fixed_effects
);

/*!
* \brief Find parameters that minimize the negative log-ligelihood (=MLE)
*		 Note: You should pre-allocate memory for optim_pars. Its length equals 1 + number of covariance parameters + number of linear regression coefficients and 1
* \param handle Handle of REModel
* \param y_data Response variable data
* \param covariate_data Covariate (=independent variable, feature) data
* \param num_covariates Number of covariates
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_OptimLinRegrCoefCovPar_R(
	SEXP handle,
	SEXP y_data,
	SEXP covariate_data,
	SEXP num_covariates
);

/*!
* \brief Calculate the value of the negative log-likelihood
* \param handle Handle of REModel
* \param y_data Response variable data
* \param cov_pars Values for covariance parameters of RE components
* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data). For Gaussian data, this is ignored
* \param[out] negll Negative log-likelihood
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_EvalNegLogLikelihood_R(
	SEXP handle,
	SEXP y_data,
	SEXP cov_pars,
	SEXP fixed_effects,
	SEXP negll
);

/*!
* \brief Get the current value of the negative log-likelihood
* \param handle Handle of REModel
* \param[out] negll Negative log-likelihood
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_GetCurrentNegLogLikelihood_R(
	SEXP handle,
	SEXP negll
);

/*!
* \brief Get covariance paramters
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariance parameters (num_cov_pars) or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param calc_std_dev If true, standard deviations are also exported
* \param[out] optim_cov_pars Optimal covariance parameters
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_GetCovPar_R(
	SEXP handle,
	SEXP calc_std_dev,
	SEXP optim_cov_pars
);

/*!
* \brief Get initial values for covariance paramters
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariance parameters (num_cov_pars) or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param[out] init_cov_pars Initial covariance parameters
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_GetInitCovPar_R(
	SEXP handle,
	SEXP init_cov_pars
);

/*!
* \brief Get / export regression coefficients
*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariates or twice this if calc_std_dev = true
* \param handle Handle of REModel
* \param calc_std_dev If true, standard deviations are also exported
* \param[out] optim_coef Optimal regression coefficients
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_GetCoef_R(
	SEXP handle,
	SEXP calc_std_dev,
	SEXP optim_coef
);

/*!
* \brief Get / export the number of iterations until convergence
*   Note: You should pre-allocate memory for num_it (length = 1)
* \param handle Handle of REModel
* \param[out] num_it Number of iterations for convergence
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_GetNumIt_R(
	SEXP handle,
	SEXP num_it
);

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
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_SetPredictionData_R(
	SEXP handle,
	SEXP num_data_pred,
	SEXP cluster_ids_data_pred,
	SEXP re_group_data_pred,
	SEXP re_group_rand_coef_data_pred,
	SEXP gp_coords_data_pred,
	SEXP gp_rand_coef_data_pred,
	SEXP covariate_data_pred,
	SEXP vecchia_pred_type,
	SEXP num_neighbors_pred,
	SEXP cg_delta_conv_pred
);

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
* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data)
* \param fixed_effects_pred Fixed effects component of location parameter for predicted data (only used for non-Gaussian data)
* \param[out] out_predict Predictive/conditional mean at prediciton points followed by the predictive covariance matrix in column-major format (if predict_cov_mat==true) or the predictive variances (if predict_var==true)
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_PredictREModel_R(
	SEXP handle,
	SEXP y_data,
	SEXP num_data_pred,
	SEXP predict_cov_mat,
	SEXP predict_var,
	SEXP predict_response,
	SEXP cluster_ids_data_pred,
	SEXP re_group_data_pred,
	SEXP re_group_rand_coef_data_pred,
	SEXP gp_coords_pred,
	SEXP gp_rand_coef_data_pred,
	SEXP cov_pars,
	SEXP covariate_data_pred,
	SEXP use_saved_data,
	SEXP vecchia_pred_type,
	SEXP num_neighbors_pred,
	SEXP cg_delta_conv_pred,
	SEXP fixed_effects,
	SEXP fixed_effects_pred,
	SEXP out_predict
);

/*!
* \brief Predict ("estimate") training data random effects
* \param handle Handle of REModel
* \param cov_pars_pred Covariance parameters of components
* \param y_obs Response variable for observed data
* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data)
* \param[out] out_predict Predicted training data random effects
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_PredictREModelTrainingDataRandomEffects_R(
	SEXP handle,
	SEXP cov_pars,
	SEXP y_obs,
	SEXP fixed_effects,
	SEXP out_predict
);

/*!
* \brief Get name of likelihood
* \return R character vector (length=1) with likelihood name
*/
GPBOOST_C_EXPORT SEXP GPB_GetLikelihoodName_R(
	SEXP handle
);

/*!
* \brief Get name of covariance parameter optimizer
* \return R character vector (length=1) with optimizer name
*/
GPBOOST_C_EXPORT SEXP GPB_GetOptimizerCovPars_R(
	SEXP handle
);

/*!
* \brief Get name of linear regression coefficients optimizer
* \return R character vector (length=1) with optimizer name
*/
GPBOOST_C_EXPORT SEXP GPB_GetOptimizerCoef_R(
	SEXP handle
);

/*!
* \brief Set the type of likelihood
* \param likelihood Likelihood name
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_SetLikelihood_R(
	SEXP handle,
	SEXP likelihood
);

/*!
* \brief Return (last used) response variable data
* \param handle Handle of REModel
* \param[out] response_data Response variable data (memory needs to be preallocated)
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_GetResponseData_R(
	SEXP handle,
	SEXP response_data
);

/*!
* \brief Return covariate data
* \param handle Handle of REModel
* \param[out] covariate_data covariate data
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT SEXP GPB_GetCovariateData_R(
	SEXP handle,
	SEXP covariate_data
);

#endif  // GPBOOST_R_H_
