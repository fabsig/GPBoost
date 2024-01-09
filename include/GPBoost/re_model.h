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
		* \param drop_intercept_group_rand_effect Indicates whether intercept random effects are dropped (only for random coefficients). If drop_intercept_group_rand_effect[k] > 0, the intercept random effect number k is dropped. Only random effects with random slopes can be dropped.
		* \param num_gp Number of (intercept) Gaussian processes
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
		* \param num_ind_points Number of inducing points / knots for, e.g., a predictive process approximation
		* \param cover_tree_radius Radius (= "spatial resolution") for the cover tree algorithm
		* \param ind_points_selection Method for choosing inducing points
		* \param likelihood Likelihood function for the observed response variable
		* \param matrix_inversion_method Method which is used for matrix inversion
		* \param seed Seed used for model creation (e.g., random ordering in Vecchia approximation)
		*/
		LIGHTGBM_EXPORT REModel(data_size_t num_data,
			const data_size_t* cluster_ids_data,
			const char* re_group_data,
			data_size_t num_re_group,
			const double* re_group_rand_coef_data,
			const data_size_t* ind_effect_group_rand_coef,
			data_size_t num_re_group_rand_coef,
			const int* drop_intercept_group_rand_effect,
			data_size_t num_gp,
			const double* gp_coords_data,
			int dim_gp_coords,
			const double* gp_rand_coef_data,
			data_size_t num_gp_rand_coef,
			const char* cov_fct,
			double cov_fct_shape,
			const char* gp_approx,
			double cov_fct_taper_range,
			double cov_fct_taper_shape,
			int num_neighbors,
			const char* vecchia_ordering,
			int num_ind_points,
			double cover_tree_radius,
			const char* ind_points_selection,
			const char* likelihood,
			const char* matrix_inversion_method,
			int seed);

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

		string_t GetCGPreconditionerType() const;

		/*!
		* \brief Set configuration parameters for the optimizer
		* \param init_cov_pars Initial values for covariance parameters of RE components
		* \param lr Learning rate for covariance parameters. If lr<= 0, internal default values are used (0.1 for "gradient_descent" and 1. for "fisher_scoring")
		* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
		* \param max_iter Maximal number of iterations
		* \param delta_rel_conv Convergence tolerance. The algorithm stops if the relative change in eiher the log-likelihood or the parameters is below this value. For "bfgs", the L2 norm of the gradient is used instead of the relative change in the log-likelihood
		* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters (only used for "gradient_descent")e
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used (only relevant if use_nesterov_acc)
		* \param trace If true, the value of the gradient is printed for some iterations
		* \param optimizer_cov Optimizer for covariance parameters
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
		*/
		void SetOptimConfig(double* init_cov_pars,
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
		* \brief Reset cov_pars_ (to their initial values).
		*/
		void ResetCovPars();

		/*!
		* \brief Find parameters that minimize the (approximate) negative log-ligelihood
		* \param y_data Response variable data
		*		For the GPBoost algorithm for Gaussian data, this equals F - y where F is the fitted value of the ensemble at the training data and y the response data.
		*		For the GPBoost algorithm for non-Gaussian data, this is ignored (and can be nullptr) as the response data has been set before.
		* \param fixed_effects Fixed effects component F of location parameter (only used for non-Gaussian data). For Gaussian data, this is ignored
		* \param called_in_GPBoost_algorithm If true, this function is called in the GPBoost algorithm, otherwise for the estimation of a GLMM
		*/
		void OptimCovPar(const double* y_data,
			const double* fixed_effects,
			bool called_in_GPBoost_algorithm);

		/*!
		* \brief Find covariance parameters and linear regression coefficients (if there are any) that minimize the (approximate) negative log-ligelihood
		*		 Note: You should pre-allocate memory for optim_par and num_it. Their length equal the number of covariance parameters + number of linear regression coefficients and 1
		* \param y_data Response variable data
		* \param covariate_data Covariate data (=independent variables, features)
		* \param num_covariates Number of covariates
		*/
		void OptimLinRegrCoefCovPar(const double* y_data,
			const double* covariate_data,
			int num_covariates);

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
		void EvalNegLogLikelihood(const double* y_data,
			double* cov_pars,
			double& negll,
			const double* fixed_effects,
			bool InitializeModeCovMat,
			bool CalcModePostRandEff_already_done);

		/*!
		* \brief Get the current value of the negative log-likelihood
		* \param[out] negll Negative log-likelihood
		*/
		void GetCurrentNegLogLikelihood(double& negll);

		/*!
		* \brief Calculate gradient and write on input (for Gaussian data, the gradient is Psi^-1*y (=y_aux))
		* \param[out] y Input response data and output gradient written on it. 
		*		For the GPBoost algorithm for Gaussian data, the input is F - y where F is the fitted value of the ensemble at the training data and y the response data.
		*		For the GPBoost algorithm for non-Gaussian data, this input is ignored as the response data has been set before.
		*		The gradient (Psi^-1*y for Gaussian data) is then written on it as output. y needs to be of length num_data_
		* \param fixed_effects Fixed effects component F of location parameter (only used for non-Gaussian data). For Gaussian data, this is ignored (and can be set to nullptr)
		* \param calc_cov_factor If true, the covariance matrix is factorized, otherwise the existing factorization is used
		*/
		void CalcGradient(double* y, const double* fixed_effects, bool calc_cov_factor);

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
		* \brief Get covariance parameters
		* \param[out] cov_par Covariance parameters stored in cov_pars_. This vector needs to be pre-allocated of length number of covariance parameters or twice this if calc_std_dev = true
		* \param calc_std_dev If true, standard deviations are also exported
		*/
		void GetCovPar(double* cov_par, bool calc_std_dev) const;

		/*!
		* \brief Get initial values for covariance parameters
		* \param[out] init_cov_par Initial covariance parameters stored in init_cov_pars_
		*/
		void GetInitCovPar(double* init_cov_par) const;

		/*!
		* \brief Get regression coefficients
		* \param[out] coef Regression coefficients stored in coef_. This vector needs to be pre-allocated of length number of covariates or twice this if calc_std_dev = true
		* \param calc_std_dev If true, standard deviations are also exported
		*/
		void GetCoef(double* coef, bool calc_std_dev) const;

		/*!
		* \brief Set the data used for making predictions (useful if the same data is used repeatedly, e.g., in validation of GPBoost)
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
		* \param nsim_var_pred Number of samples when simulation is used for calculating predictive variances
		* \param rank_pred_approx_matrix_lanczos Rank of the matrix for approximating predictive covariances obtained using the Lanczos algorithm
		*/
		void SetPredictionData(data_size_t num_data_pred,
			const data_size_t* cluster_ids_data_pred,
			const char* re_group_data_pred,
			const double* re_group_rand_coef_data_pred,
			double* gp_coords_data_pred,
			const double* gp_rand_coef_data_pred,
			const double* covariate_data_pred,
			const char* vecchia_pred_type,
			int num_neighbors_pred,
			double cg_delta_conv_pred,
			int nsim_var_pred,
			int rank_pred_approx_matrix_lanczos);

		/*!
		* \brief Make predictions: calculate conditional mean and variances or covariance matrix
		*		 Note: You should pre-allocate memory for out_predict
		*			   Its length is equal to num_data_pred if only the conditional mean is predicted (predict_cov_mat==false && predict_var==false)
		*			   or num_data_pred * (1 + num_data_pred) if the predictive covariance matrix is also calculated (predict_cov_mat==true)
		*			   or num_data_pred * 2 if predictive variances are also calculated (predict_var==true)
		* \param y_obs Response variable for observed data
		* \param num_data_pred Number of data points for which predictions are made
		* \param[out] out_predict Predictive mean at prediction points followed by the predictive covariance matrix in column-major format (if predict_cov_mat==true) or the predictive variances (if predict_var==true)
		* \param predict_cov_mat If true, the predictive/conditional covariance matrix is calculated (default=false) (predict_var and predict_cov_mat cannot be both true
		* \param predict_var If true, the predictive/conditional variances are calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
		* \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects
		* \param cluster_ids_data_pred IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization) for which predictions are to be made
		* \param re_group_data_pred Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
		* \param re_group_rand_coef_data_pred Covariate data for grouped random coefficients
		* \param gp_coords_data_pred Coordinates (features) for Gaussian process
		* \param gp_rand_coef_data_pred Covariate data for Gaussian process random coefficients
		* \param cov_pars_pred Covariance parameters of RE components
		* \param covariate_data_pred Covariate data (=independent variables, features) for prediction
		* \param use_saved_data If true previusly set data on groups, coordinates, and covariates are used and some arguments of this function are ignored
		* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data)
		* \param fixed_effects_pred Fixed effects component of location parameter for predicted data (only used for non-Gaussian data)
		* \param suppress_calc_cov_factor If true, the covariance matrix of the observed data is not factorized (default=false), otherwise it is dynamically decided whether to factorize or nor
		*/
		void Predict(const double* y_obs,
			data_size_t num_data_pred,
			double* out_predict,
			bool predict_cov_mat,
			bool predict_var,
			bool predict_response,
			const data_size_t* cluster_ids_data_pred,
			const char* re_group_data_pred,
			const double* re_group_rand_coef_data_pred,
			double* gp_coords_data_pred,
			const double* gp_rand_coef_data_pred,
			const double* cov_pars_pred,
			const double* covariate_data_pred,
			bool use_saved_data,
			const double* fixed_effects,
			const double* fixed_effects_pred,
			bool suppress_calc_cov_factor);

		/*!
		* \brief Predict ("estimate") training data random effects
		* \param cov_pars_pred Covariance parameters of components
		* \param y_obs Response variable for observed data
		* \param[out] out_predict Predicted training data random effects and variances if calc_var
		* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian data)
		* \param calc_var If true, variances are also calculated
		*/
		void PredictTrainingDataRandomEffects(const double* cov_pars_pred,
			const double* y_obs,
			double* out_predict,
			const double* fixed_effects,
			bool calc_var) const;

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
			const int num_leaves,
			double* leaf_values) const;

		/*!
		* \brief If cov_pars_ is is not defined, define them as init_cov_pars_ or if init_cov_pars_ is not given, find "reasonable" default values for the intial values of the covariance parameters
		* \param y_data Response variable data used for finding intial values if cov_pars_ is not defined
		*/
		void InitializeCovParsIfNotDefined(const double* y_data);

		/*!
		* \brief Return number of additional likelihood parameters (aux_pars_)
		*/
		int NumAuxPars() const;

		/*!
		* \brief Get additional likelihood parameters (e.g., shape parameter for a gamma likelihood)
		* \param[out] aux_pars Additional likelihood parameters (aux_pars_). This vector needs to be pre-allocated
		* \param[out] name Name of the first parameter
		*/
		void GetAuxPars(double* aux_pars,
			string_t& name) const;

		/*!
		* \brief Set aux_pars_
		* \param aux_pars New values for aux_pars_
		*/
		void SetAuxPars(const double* aux_pars);

		/*!
		* \brief Get initial values for additional likelihood parameters (e.g., shape parameter for a gamma likelihood)
		* \param[out] aux_pars Initial additional likelihood parameters stored in init_aux_pars_
		*/
		void GetInitAuxPars(double* aux_pars) const;

		/*!
		* \brief Calculate test negative log-likelihood using adaptive GH quadrature
		* \param y_test Test response variable
		* \param pred_mean Predictive mean of latent random effects
		* \param pred_var Predictive variances of latent random effects
		* \param num_data Number of data points
		*/
		double TestNegLogLikelihoodAdaptiveGHQuadrature(const label_t* y_test,
			const double* pred_mean,
			const double* pred_var,
			const data_size_t num_data);

	private:

		string_t matrix_format_ = "den_mat_t";//den_mat_t, sp_mat_t, sp_mat_rm_t
		std::unique_ptr<REModelTemplate<sp_mat_t, chol_sp_mat_t>> re_model_sp_;
		std::unique_ptr<REModelTemplate<sp_mat_rm_t, chol_sp_mat_rm_t>> re_model_sp_rm_;
		std::unique_ptr<REModelTemplate<den_mat_t, chol_den_mat_t>> re_model_den_;
		/*! \brief List of covariance functions wtih compact support */
		const std::set<string_t> COMPACT_SUPPORT_COVS_{ "wendland", "exponential_tapered" };
		int num_it_ = 0; //Number of iterations done for covariance and linear regression parameter estimation
		bool calc_std_dev_ = false;
		// Covariance parameters related variables
		vec_t cov_pars_; //Covariance parameters
		vec_t init_cov_pars_; //Initial values for covariance parameters
		bool cov_pars_initialized_ = false; //This is true if InitializeCovParsIfNotDefined() has been called
		bool covariance_matrix_has_been_factorized_ = false; //If true, the covariance matrix Psi has been factorized for the cov_pars_ (either through OptimCovPar/OptimLinRegrCoefCovPar or EvalNegLogLikelihood) and will not be factorized anew when making predictions in Predict
		bool init_cov_pars_provided_ = false;
		bool cov_pars_have_been_provided_for_prediction_ = false; //This is true if Predict() has been called once with cov_pars_pred != nullptr (saved in order to determine whether covariance matrix needs to be factorized again or not)
		vec_t std_dev_cov_pars_;
		int num_cov_pars_;
		// Linear regression coefficients related variables
		vec_t coef_;//linear regression coefficients for fixed effects (in case there are any)
		bool has_covariates_ = false;
		bool init_coef_given_ = false;
		bool coef_given_or_estimated_ = false;
		vec_t std_dev_coef_;
		// Variables for additional parameters for non-Gaussian likelihoods
		vec_t init_aux_pars_; // Additional parameters for non-Gaussian likelihoods
		bool init_aux_pars_given_ = false;
		bool model_has_been_estimated_ = false;
	};

}  // namespace GPBoost

#endif   // GPB_RE_MODEL_H_
