/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_RE_MODEL_H_
#define GPB_RE_MODEL_H_

#include <GPBoost/type_defs.h>
#include <GPBoost/re_model_template.h>
#include <LightGBM/export.h>

#include <memory>

namespace GPBoost {

	/*!
	* \brief This class models the random effects components
	*
	*    Some details:
	*       1. It collects the different random effect components
	*		2. It allows for doing the necessary random effect related computations
	*/

	class REModel {
	public:
		/*! \brief Null costructor */
		LIGHTGBM_EXPORT REModel();

		/*!
		* \brief Costructor
		* \param num_data Number of data points
		* \param cluster_ids_data IDs / labels indicating independent realizations of random effects / Gaussian processes (same values = same process realization)
		* \param re_group_data Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
		* \param num_re_group Number of grouped (intercept) random effects
		* \param re_group_rand_coef_data Covariate data for grouped random coefficients
		* \param ind_effect_group_rand_coef Indices that relate every random coefficients to a "base" intercept grouped random effect. Counting start at 1.
		* \param num_re_group_rand_coef Number of grouped random coefficient
		* \param num_gp Number of (intercept) Gaussian processes
		* \param gp_coords_data Coordinates (features) for Gaussian process
		* \param dim_gp_coords Dimension of the coordinates (=number of features) for Gaussian process
		* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
		* \param num_gp_rand_coef Number of Gaussian process random coefficients
		* \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. For the Wendland covariance function, we follow the notation of Bevilacqua et al. (2018)). This parameter is irrelevant for some covariance functions such as the exponential or Gaussian.
		* \param cov_fct_taper_range Range parameter of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
		* \param vecchia_approx If true, the Veccia approximation is used for the Gaussian process
		* \param num_neighbors The number of neighbors used in the Vecchia approximation
		* \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
		* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
		* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
		* \param likelihood Likelihood function for the observed response variable. Default = "gaussian"
		*/
		LIGHTGBM_EXPORT REModel(data_size_t num_data,
			const gp_id_t* cluster_ids_data,
			const char* re_group_data,
			data_size_t num_re_group,
			const double* re_group_rand_coef_data,
			const int32_t* ind_effect_group_rand_coef,
			data_size_t num_re_group_rand_coef,
			data_size_t num_gp,
			const double* gp_coords_data,
			int dim_gp_coords,
			const double* gp_rand_coef_data,
			data_size_t num_gp_rand_coef,
			const char* cov_fct,
			double cov_fct_shape,
			double cov_fct_taper_range,
			bool vecchia_approx,
			int num_neighbors,
			const char* vecchia_ordering,
			const char* vecchia_pred_type,
			int num_neighbors_pred,
			const char* likelihood);

		/*! \brief Destructor */
		LIGHTGBM_EXPORT ~REModel();

		/*! \brief Disable copy */
		REModel& operator=(const REModel&) = delete;

		/*! \brief Disable copy */
		REModel(const REModel&) = delete;

		/*!
		* \brief Returns true if Gaussian data false otherwise
		* \return true if Gaussian data false otherwise
		*/
		bool GaussLikelihood() const;

		/*!
		* \brief Returns the type of likelihood
		* \return Type of likelihood
		*/
		string_t GetLikelihood() const;

		/*!
		* \brief Set the type of likelihood
		* \param likelihood Likelihood name
		*/
		void SetLikelihood(const string_t& likelihood);

		string_t GetOptimizerCovPars() const;

		string_t GetOptimizerCoef() const;

		/*!
		* \brief Set configuration parameters for the optimizer
		* \param init_cov_pars Initial values for covariance parameters of RE components
		* \param lr Learning rate. If lr<= 0, default values are used. Default value = 0.1 for "gradient_descent" and 1. for "fisher_scoring"
		* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
		* \param max_iter Maximal number of iterations
		* \param delta_rel_conv Convergence criterion: stop iteration if relative change in parameters is below this value
		* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters. Default = true
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
		* \param trace If true, the value of the gradient is printed for some iterations. Default = false
		* \param optimizer Options: "gradient_descent" or "fisher_scoring"
		* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
		* \param convergence_criterion The convergence criterion used for terminating the optimization algorithm. Options: "relative_change_in_log_likelihood" (default) or "relative_change_in_parameters"
		* \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
		*/
		void SetOptimConfig(double* init_cov_pars = nullptr, double lr = -1.,
			double acc_rate_cov = 0.5, int max_iter = 1000, double delta_rel_conv = 1.0e-6,
			bool use_nesterov_acc = true, int nesterov_schedule_version = 0, bool trace = true,
			const char* optimizer = nullptr, int momentum_offset = 2, const char* convergence_criterion = nullptr,
			bool calc_std_dev = false);

		/*!
		* \brief Set configuration parameters for the optimizer for linear regression coefficients
		* \param num_covariates Number of coefficients / covariates
		* \param init_coef Initial values for the regression coefficients
		* \param lr_coef Learning rate for fixed-effect linear coefficients
		* \param acc_rate_coef Acceleration rate for coefficients for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
		* \param optimizer Options: "gradient_descent" or "wls" (coordinate descent using weighted least squares)
		*/
		void SetOptimCoefConfig(int num_covariates = 0, double* init_coef = nullptr,
			double lr_coef = 0.1, double acc_rate_coef = 0.5, const char* optimizer = nullptr);

		/*!
		* \brief Reset cov_pars_ (to their initial values).
		*/
		void ResetCovPars();

		/*!
		* \brief Find parameters that minimize the negative log-ligelihood (=MLE) using (Nesterov accelerated) gradient descent
		* \param y_data Response variable data
		*		For the GPBoost algorithm for Gaussian data, this equals F - y where F is the fitted value of the ensemble at the training data and y the response data.
		*		For the GPBoost algorithm for non-Gaussian data, this is ignored (and can be nullptr) as the response data has been set before.
		* \param fixed_effects Fixed effects component F of location parameter (only used for non-Gaussian data). For Gaussian data, this is ignored
		*/
		void OptimCovPar(const double* y_data, const double* fixed_effects = nullptr);

		/*!
		* \brief Find linear regression coefficients and covariance parameters that minimize the negative log-ligelihood (=MLE) using (Nesterov accelerated) gradient descent
		*		 Note: You should pre-allocate memory for optim_par and num_it. Their length equal the number of covariance parameters + number of linear regression coefficients and 1
		* \param y_data Response variable data
		* \param covariate_data Covariate data (=independent variables, features)
		* \param num_covariates Number of covariates
		*/
		void OptimLinRegrCoefCovPar(const double* y_data, const double* covariate_data, int num_covariates);

		/*!
		* \brief Find constant initial value of ensenmble for boosting (used only for non-Gaussian data). 
		* \param[out] init_score Initial value for boosting ensemble (=initial score in LightGBM)
		*/
		void FindInitialValueBoosting(double* init_score);

		/*!
		* \brief Calculate the value of the negative log-likelihood
		* \param y_data Response variable data
		* \param cov_pars Values for covariance parameters of RE components
		* \param[out] negll Negative log-likelihood
		* \param fixed_effects (only used for non-Gaussian data) Fixed effects component of location parameter
		* \param InitializeModeCovMat (only used for non-Gaussian data) If true, posterior mode is initialized to 0 and the covariance matrix is calculated. Otherwise, existing values are used
		* \param CalcModePostRandEff_already_done (only used for non-Gaussian data) If true, it is assumed that the posterior mode of the random effects has already been calculated
		*/
		void EvalNegLogLikelihood(const double* y_data, double* cov_pars, double& negll,
			const double* fixed_effects = nullptr, bool InitializeModeCovMat = true, bool CalcModePostRandEff_already_done = false);

		/*!
		* \brief Calculate gradient and write on input (for Gaussian data, the gradient is Psi^-1*y (=y_aux))
		* \param[out] y Input response data and output gradient written on it. 
		*		For the GPBoost algorithm for Gaussian data, the input is F - y where F is the fitted value of the ensemble at the training data and y the response data.
		*		For the GPBoost algorithm for non-Gaussian data, this input is ignored as the response data has been set before.
		*		The gradient (Psi^-1*y for Gaussian data) is then written on it as output. y needs to be of length num_data_
		* \param fixed_effects Fixed effects component F of location parameter (only used for non-Gaussian data). For Gaussian data, this is ignored (and can be set to nullptr)
		* \param calc_cov_factor If true, the covariance matrix is factorized, otherwise the existing factorization is used
		*/
		void CalcGradient(double* y, const double* fixed_effects = nullptr, bool calc_cov_factor = true);

		/*!
		* \brief Set response data y
		* \param y Response data
		*/
		void SetY(const double* y) const;

		/*!
		* \brief Set response data y if data is of type floaf (used for GPBoost algorithm since labels are float)
		* \param y Response data
		*/
		void SetY(const float* y) const;

		/*!
		* \brief Return (last used) response variable data
		* \param[out] y Response variable data (memory needs to be preallocated)
		*/
		void GetY(double* y) const;

		/*!
		* \brief Return covariate data
		* \param[out] covariate_data covariate data
		*/
		void GetCovariateData(double* covariate_data) const;

		/*!
		* \brief Get covariance paramters
		* \param[out] cov_par Covariance paramters stored in cov_pars_. This vector needs to be pre-allocated of length number of covariance paramters or twice this if calc_std_dev = true
		* \param calc_std_dev If true, standard deviations are also exported
		*/
		void GetCovPar(double* cov_par, bool calc_std_dev = false) const;

		/*!
		* \brief Get initial values for covariance paramters
		* \param[out] init_cov_par Initial covariance paramters stored in init_cov_pars_. This vector needs to be pre-allocated of length number of covariance paramters or twice this if calc_std_dev = true
		* \param calc_std_dev If true, standard deviations are also exported
		*/
		void GetInitCovPar(double* init_cov_par) const;

		/*!
		* \brief Get regression coefficients
		* \param[out] coef Regression coefficients stored in coef_. This vector needs to be pre-allocated of length number of covariates or twice this if calc_std_dev = true
		* \param calc_std_dev If true, standard deviations are also exported
		*/
		void GetCoef(double* coef, bool calc_std_dev = false) const;

		/*!
		* \brief Set the data used for making predictions (useful if the same data is used repeatedly, e.g., in validation of GPBoost)
		* \param num_data_pred Number of data points for which predictions are made
		* \param cluster_ids_data_pred IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization) for which predictions are to be made
		* \param re_group_data_pred Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
		* \param re_group_rand_coef_data_pred Covariate data for grouped random coefficients
		* \param gp_coords_data_pred Coordinates (features) for Gaussian process
		* \param gp_rand_coef_data_pred Covariate data for Gaussian process random coefficients
		* \param covariate_data_pred Covariate data (=independent variables, features) for prediction
		*/
		void SetPredictionData(data_size_t num_data_pred,
			const gp_id_t* cluster_ids_data_pred = nullptr, const char* re_group_data_pred = nullptr,
			const double* re_group_rand_coef_data_pred = nullptr, double* gp_coords_data_pred = nullptr,
			const double* gp_rand_coef_data_pred = nullptr, const double* covariate_data_pred = nullptr);

		/*!
		* \brief Make predictions: calculate conditional mean and variances or covariance matrix
		*		 Note: You should pre-allocate memory for out_predict
		*			   Its length is equal to num_data_pred if only the conditional mean is predicted (predict_cov_mat==false && predict_var==false)
		*			   or num_data_pred * (1 + num_data_pred) if the predictive covariance matrix is also calculated (predict_cov_mat==true)
		*			   or num_data_pred * 2 if predictive variances are also calculated (predict_var==true)
		* \param y_obs Response variable for observed data
		* \param num_data_pred Number of data points for which predictions are made
		* \param[out] out_predict Predictive/conditional mean at prediciton points followed by the predictive covariance matrix in column-major format (if predict_cov_mat==true) or the predictive variances (if predict_var==true)
		* \param predict_cov_mat If true, the predictive/conditional covariance matrix is calculated (default=false) (predict_var and predict_cov_mat cannot be both true
		* \param predict_var If true, the predictive/conditional variances are calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
		* \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects (this is only relevant for non-Gaussian data) (default=false)
		* \param cluster_ids_data_pred IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization) for which predictions are to be made
		* \param re_group_data_pred Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
		* \param re_group_rand_coef_data_pred Covariate data for grouped random coefficients
		* \param gp_coords_data_pred Coordinates (features) for Gaussian process
		* \param gp_rand_coef_data_pred Covariate data for Gaussian process random coefficients
		* \param cov_pars_pred Covariance parameters of RE components
		* \param covariate_data_pred Covariate data (=independent variables, features) for prediction
		* \param use_saved_data If true previusly set data on groups, coordinates, and covariates are used and some arguments of this function are ignored
		* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
		* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions (-1 means that the value already set at initialization is used)
		* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data)
		* \param fixed_effects_pred Fixed effects component of location parameter for predicted data (only used for non-Gaussian data)
		* \param suppress_calc_cov_factor If true, the covariance matrix of the observed data is not factorized (default=false), otherwise it is dynamically decided whether to factorize or nor
		*/
		void Predict(const double* y_obs, data_size_t num_data_pred, double* out_predict,
			bool predict_cov_mat = false, bool predict_var = false, bool predict_response = false,
			const gp_id_t* cluster_ids_data_pred = nullptr, const char* re_group_data_pred = nullptr, const double* re_group_rand_coef_data_pred = nullptr,
			double* gp_coords_data_pred = nullptr, const double* gp_rand_coef_data_pred = nullptr,
			const double* cov_pars_pred = nullptr, const double* covariate_data_pred = nullptr,
			bool use_saved_data = false, const char* vecchia_pred_type = nullptr, int num_neighbors_pred = -1,
			const double* fixed_effects = nullptr, const double* fixed_effects_pred = nullptr,
			bool suppress_calc_cov_factor = false) const;

		int GetNumIt() const;

		int GetNumData() const;

		/*!
		* \brief Calculate the leaf values when performing a Newton update step after the tree structure has been found in tree-boosting
		*    Note: only used in GPBoost for tree-boosting (this is called from regression_objective). It is assumed that 'CalcGetYAux' has been called before.
		* \param data_leaf_index Leaf index for every data point (array of size num_data)
		* \param num_leaves Number of leaves
		* \param[out] leaf_values Leaf values when performing a Newton update step (array of size num_leaves)
		*/
		void NewtonUpdateLeafValues(const int* data_leaf_index,
			const int num_leaves, double* leaf_values) const;

		/*!
		* \brief If cov_pars_ is is not defined, define them as init_cov_pars_ or if init_cov_pars_ is not given, find "reasonable" default values for the intial values of the covariance parameters
		* \param y_data Response variable data used for finding intial values if cov_pars_ is not defined
		*/
		void InitializeCovParsIfNotDefined(const double* y_data);

	private:

		bool sparse_ = false;
		std::unique_ptr < REModelTemplate<sp_mat_t, chol_sp_mat_t> > re_model_sp_;
		std::unique_ptr < REModelTemplate<den_mat_t, chol_den_mat_t> > re_model_den_;
		vec_t cov_pars_;//covariance paramters
		vec_t init_cov_pars_;//Initial values for covariance parameters
		bool cov_pars_initialized_ = false;// This is true of InitializeCovParsIfNotDefined() has been called
		bool covariance_matrix_has_been_factorized_ = false;//If true, the covariance matrix Psi has been factorized for the cov_pars_ (either through OptimCovPar/OptimLinRegrCoefCovPar or EvalNegLogLikelihood) and will not be factorized anew when making predictions in Predict
		bool init_cov_pars_provided_ = false;
		vec_t std_dev_cov_pars_;
		int num_cov_pars_;
		int num_it_ = 0;
		double lr_cov_ = -1.;
		double acc_rate_cov_ = 0.5;
		int momentum_offset_ = 2;
		int max_iter_ = 1000;
		double delta_rel_conv_ = 1.0e-6;
		bool use_nesterov_acc_ = true;//only used for "gradient_descent"
		int nesterov_schedule_version_ = 0;
		string_t optimizer_cov_pars_ = "gradient_descent";//"gradient_descent", "fisher_scoring", or "nelder_mead" (The default = "fisher_scoring" is changed to "gradient_descent" for non-Gaussian data upon initialization)
		vec_t coef_;//linear regression coefficients for fixed effects (in case there are any)
		bool has_covariates_ = false;
		bool coef_initialized_ = false;
		vec_t std_dev_coef_;
		double lr_coef_ = 0.1;
		double acc_rate_coef_ = 0.5;
		string_t optimizer_coef_ = "wls";//"gradient_descent" or "wls" (The default = "wls" is changed to "gradient_descent" for non-Gaussian data upon initialization)
		string_t convergence_criterion_ = "relative_change_in_log_likelihood";//"relative_change_in_log_likelihood" (default) or "relative_change_in_parameters"
		bool cov_pars_optimizer_hase_been_set_ = false;//true if the function 'SetOptimConfig' has been called and optimizer_cov_pars_ has been set
		bool coef_optimizer_hase_been_set_ = false;//true if the function 'SetOptimCoefConfig' has been called and optimizer_coef_ has been set
		bool calc_std_dev_ = false;
		/*! \brief List of covariance functions wtih compact support */
		const std::set<string_t> COMPACT_SUPPORT_COVS_{ "wendland",
			"exponential_tapered" };
	};

}  // namespace GPBoost

#endif   // GPB_RE_MODEL_H_
