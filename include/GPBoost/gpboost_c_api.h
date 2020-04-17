/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPBOOST_C_API_H_
#define GPBOOST_C_API_H_

#include <GPBoost/export.h>

#include <cstring>

typedef void* REModelHandle;  /*!< \brief Handle of REmodel. */

#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)  /*!< \brief Thread local specifier. */
#else
#define THREAD_LOCAL thread_local  /*!< \brief Thread local specifier. */
#endif

/*!
 * \brief Handle of error message.
 * \return Error message
 */
static char* GPB_LastErrorMsg() { static THREAD_LOCAL char err_msg[512] = "Everything is fine"; return err_msg; }

#pragma warning(disable : 4996)
/*!
 * \brief Set string message of the last error.
 * \param msg Error message
 */
inline void GPB_SetLastError(const char* msg) {
	std::strcpy(GPB_LastErrorMsg(), msg);
}

/*!
* \brief Create REModel
* \param num_data Number of data points
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
* \param[out] out Created REModel
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_CreateREModel(int num_data,
	const int* cluster_ids_data,
	const char* re_group_data,
	int num_re_group,
	const double* re_group_rand_coef_data,
	const int* ind_effect_group_rand_coef,
	int num_re_group_rand_coef,
	int num_gp,
	const double* gp_coords_data,
	const int dim_gp_coords,
	const double* gp_rand_coef_data,
	int num_gp_rand_coef,
	const char* cov_fct,
	double cov_fct_shape,
	bool vecchia_approx,
	int num_neighbors,
	const char* vecchia_ordering,
	const char* vecchia_pred_type,
	int num_neighbors_pred,
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
* \param lr Learning rate
* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
* \param max_iter Maximal number of iterations
* \param delta_rel_conv Convergence criterion: stop iteration if relative change is below this value
* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters. Default = true
* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
* \param trace If true, the value of the gradient is printed for some iterations. Default = false
* \param optimizer Options: "gradient_descent" or "fisher_scoring"
* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_SetOptimConfig(REModelHandle handle,
	double* init_cov_pars = nullptr,
	double lr = 0.01,
	double acc_rate_cov = 0.5,
	int max_iter = 1000,
	double delta_rel_conv = 1.0e-6,
	bool use_nesterov_acc = true,
	int nesterov_schedule_version = 0,
	bool trace = false,
	const char* optimizer = nullptr,
	int momentum_offset = 2);

/*!
* \brief Set configuration parameters for the optimizer for linear regression coefficients
* \param handle Handle of REModel
* \param num_covariates Number of covariates
* \param init_coef Initial values for the regression coefficients
* \param lr_coef Learning rate for fixed-effect linear coefficients
* \param optimizer Options: "gradient_descent" or "wls" (coordinate descent using weighted least squares)
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_SetOptimCoefConfig(REModelHandle handle,
	int num_covariates = 0,
	double* init_coef = nullptr,
	double lr_coef = 0.001,
	double acc_rate_coef = 0.5,
	const char* optimizer = nullptr);

/*!
* \brief Find parameters that minimize the negative log-ligelihood (=MLE)
* \param handle Handle of REModel
* \param y_data Response variable data
* \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_OptimCovPar(REModelHandle handle,
	const double* y_data,
	bool calc_std_dev = false);

/*!
* \brief Find linear regression coefficients and covariance parameters that minimize the negative log-ligelihood (=MLE)
*		 Note: You should pre-allocate memory for optim_pars. Its length equals 1 + number of covariance parameters + number of linear regression coefficients and 1
* \param handle Handle of REModel
* \param y_data Response variable data
* \param covariate_data Covariate (=independent variable, feature) data
* \param num_covariates Number of covariates
* \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_OptimLinRegrCoefCovPar(REModelHandle handle,
	const double* y_data,
	const double* covariate_data,
	int num_covariates,
	bool calc_std_dev = false);

/*!
* \brief Calculate the value of the negative log-likelihood
* \param handle Handle of REModel
* \param y_data Response variable data
* \param cov_pars Values for covariance parameters of RE components
* \param[out] negll Negative log-likelihood
* \return 0 when succeed, -1 when failure happens
*/
GPBOOST_C_EXPORT int GPB_EvalNegLogLikelihood(REModelHandle handle,
	const double* y_data,
	double* cov_pars,
	double* negll);

	/*!
	* \brief Get / export covariance paramters
	*		 Note: You should pre-allocate memory for optim_cov_pars. Its length equals the number of covariance parameters (num_cov_pars) or twice this if calc_std_dev = true
	* \param handle Handle of REModel
	* \param[out] optim_cov_pars Optimal covariance parameters
	* \param calc_std_dev If true, standard deviations are also exported
	* \return 0 when succeed, -1 when failure happens
	*/
	GPBOOST_C_EXPORT int GPB_GetCovPar(REModelHandle handle,
		double* optim_cov_pars,
		bool calc_std_dev = false);

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
		bool calc_std_dev = false);

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
	*/
	GPBOOST_C_EXPORT int GPB_SetPredictionData(REModelHandle handle,
		int num_data_pred,
		const int* cluster_ids_data_pred = nullptr,
		const char* re_group_data_pred = nullptr,
		const double* re_group_rand_coef_data_pred = nullptr,
		double* gp_coords_data_pred = nullptr,
		const double* gp_rand_coef_data_pred = nullptr,
		const double* covariate_data_pred = nullptr);

	/*!
	* \brief Make predictions: calculate conditional mean and covariance matrix
	*		 Note: You should pre-allocate memory for out_predict
	*			   Its length is equal to num_data_pred if only the conditional mean is predicted (predict_cov_mat=false)
	*			   or num_data_pred * (1 + num_data_pred) if both the conditional mean and covariance matrix are predicted (predict_cov_mat=true)
	* \param handle Handle of REModel
	* \param y_data Response variable for observed data
	* \param num_data_pred Number of data points for which predictions are made
	* \param[out] out_predict Conditional mean at prediciton points (="predicted value") followed by (if predict_cov_mat=true) the conditional covariance matrix at in column-major format
	* \param predict_cov_mat If true, the conditional covariance matrix is calculated (default=false)
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
	* \return 0 when succeed, -1 when failure happens
	*/
	GPBOOST_C_EXPORT int GPB_PredictREModel(REModelHandle handle,
		const double* y_data,
		int num_data_pred,
		double* out_predict,
		bool predict_cov_mat = false,
		const int* cluster_ids_data_pred = nullptr,
		const char* re_group_data_pred = nullptr,
		const double* re_group_rand_coef_data_pred = nullptr,
		double* gp_coords_pred = nullptr,
		const double* gp_rand_coef_data_pred = nullptr,
		const double* cov_pars = nullptr,
		const double* covariate_data_pred = nullptr,
		bool use_saved_data = true,
		const char* vecchia_pred_type = nullptr,
		int num_neighbors_pred = -1);

#endif  // GPBOOST_C_API_H_
