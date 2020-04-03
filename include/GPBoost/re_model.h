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
#include <GPBoost/export.h>
#include <GPBoost/re_model_template.h>

#include <memory>

namespace GPBoost {

	/*!
	* \brief This class models the random effects components
	*
	*    Some details:
	*       1. It collects the different random effect components
	*		    2. It allows for doing the necessary random effect related computations
	*		    3. 
	*/

  class REModel {
  public:
    /*! \brief Null costructor */
    GPBOOST_EXPORT REModel();

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
    * \param cov_fct Type of covariance (kernel) function for Gaussian process. We follow the notation and parametrization of Diggle and Ribeiro (2007) except for the Matern covariance where we follow Rassmusen and Williams (2006)
    * \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern covariance, irrelevant for some covariance functions such as the exponential or Gaussian)
    * \param vecchia_approx If true, the Veccia approximation is used for the Gaussian process
    * \param num_neighbors The number of neighbors used in the Vecchia approximation
    * \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
    * \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
    * \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
    */
    GPBOOST_EXPORT REModel(data_size_t num_data, const gp_id_t* cluster_ids_data = nullptr, const char* re_group_data = nullptr, data_size_t num_re_group = 0,
      const double* re_group_rand_coef_data = nullptr, const int32_t* ind_effect_group_rand_coef = nullptr, data_size_t num_re_group_rand_coef = 0,
      data_size_t num_gp = 0, const double* gp_coords_data = nullptr, int dim_gp_coords = 2, const double* gp_rand_coef_data = nullptr, data_size_t num_gp_rand_coef = 0,
      const char* cov_fct = nullptr, double cov_fct_shape = 0. , bool vecchia_approx = false, int num_neighbors = 30, const char* vecchia_ordering = nullptr,
      const char* vecchia_pred_type = nullptr, int num_neighbors_pred = 30);

    /*! \brief Destructor */
    GPBOOST_EXPORT ~REModel();

    /*! \brief Disable copy */
    REModel& operator=(const REModel&) = delete;

    /*! \brief Disable copy */
    REModel(const REModel&) = delete;

    /*!
    * \brief Set configuration parameters for the optimizer
    * \param init_cov_pars Initial values for covariance parameters of RE components
    * \param lr Learning rate
    * \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
    * \param max_iter Maximal number of iterations
    * \param delta_rel_conv Convergence criterion: stop iteration if relative change in parameters is below this value
    * \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters. Default = true
    * \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
    * \param trace If true, the value of the gradient is printed for some iterations. Default = false
    * \param optimizer Options: "gradient_descent" or "fisher_scoring"
    * \param momentum_offset Number of iterations for which no mometum is applied in the beginning
    */
    void SetOptimConfig(double* init_cov_pars = nullptr, double lr = 0.01,
      double acc_rate_cov = 0.5, int max_iter = 1000, double delta_rel_conv = 1.0e-6,
      bool use_nesterov_acc = true, int nesterov_schedule_version = 0, bool trace = true,
      const char* optimizer = nullptr, int momentum_offset = 2);

    /*!
    * \brief Reset cov_pars_ (to their initial values).
    */
    void ResetCovPars();

    /*!
    * \brief Set configuration parameters for the optimizer for linear regression coefficients
    * \param num_covariates Number of coefficients / covariates
    * \param init_coef Initial values for the regression coefficients
    * \param lr_coef Learning rate for fixed-effect linear coefficients
    * \param acc_rate_coef Acceleration rate for coefficients for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
    * \param optimizer Options: "gradient_descent" or "wls" (coordinate descent using weighted least squares)
    */
    void SetOptimCoefConfig(int num_covariates = 0, double* init_coef = nullptr,
      double lr_coef = 0.001, double acc_rate_coef = 0.5, const char* optimizer = nullptr);

    /*!
    * \brief Find parameters that minimize the negative log-ligelihood (=MLE) using (Nesterov accelerated) gradient descent
    * \param y_data Response variable data
    * \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
    */
    void OptimCovPar(const double* y_data, bool calc_std_dev = false);

    /*!
    * \brief Find linear regression coefficients and covariance parameters that minimize the negative log-ligelihood (=MLE) using (Nesterov accelerated) gradient descent
    *		 Note: You should pre-allocate memory for optim_par and num_it. Their length equal the number of covariance parameters + number of linear regression coefficients and 1
    * \param y_data Response variable data
    * \param covariate_data Covariate data (=independent variables, features)
    * \param num_covariates Number of covariates
    * \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
    */
    void OptimLinRegrCoefCovPar(const double* y_data, const double* covariate_data, int num_covariates, bool calc_std_dev = false);

    /*!
    * \brief Calculate y_aux = Psi^-1*y and write on input
    * \param[out] y Response data. Output Psi^-1*y (=y_aux_) is then written on it. This vector needs to be pre-allocated of length num_data_
    * \param[out] calc_cov_factor If true, the covariance matrix is factorized, otherwise the existing factorization is used
    */
    void CalcGetYAux(double* y, bool calc_cov_factor = true);

    /*!
    * \brief Set response data y
    * \param y Response data
    */
    void SetY(const double* y) const;

    /*!
    * \brief Get / export covariance paramters
    * \param[out] covariance paramters stored in cov_pars_. This vector needs to be pre-allocated of length number of covariance paramters or twice this if calc_std_dev = true
    * \param calc_std_dev If true, standard deviations are also exported
    */
    void GetCovPar(double* cov_par, bool calc_std_dev = false) const;

    /*!
    * \brief Get / export regression coefficients 
    * \param[out] Regression coefficients stored in coef_. This vector needs to be pre-allocated of length number of covariates or twice this if calc_std_dev = true
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
    * \brief Make predictions: calculate conditional mean and covariance matrix
    *		 Note: You should pre-allocate memory for out_predict
    *			   Its length is equal to num_data_pred if only the conditional mean is predicted (predict_cov_mat=false)
    *			   or num_data_pred * (1 + num_data_pred) if both the conditional mean and covariance matrix are predicted (predict_cov_mat=true)
    * \param y_obs Response variable for observed data
    * \param num_data_pred Number of data points for which predictions are made
    * \param[out] out_predict Conditional mean at prediciton points (="predicted value") followed by (if predict_cov_mat=true) the conditional covariance matrix at in column-major format
    * \param predict_cov_mat If true, the conditional covariance matrix is calculated (default=false)
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
    */
    void Predict(const double* y_obs, data_size_t num_data_pred,
      double* out_predict, bool predict_cov_mat = false,
      const gp_id_t* cluster_ids_data_pred = nullptr, const char* re_group_data_pred = nullptr, const double* re_group_rand_coef_data_pred = nullptr,
      double* gp_coords_data_pred = nullptr, const double* gp_rand_coef_data_pred = nullptr,
      const double* cov_pars_pred = nullptr, const double* covariate_data_pred = nullptr,
      bool use_saved_data = false, const char* vecchia_pred_type = nullptr, int num_neighbors_pred = -1) const;

    int GetNumIt() const;

    int GetNumData() const;

    /*!
    * \brief Calculate the leaf values when performing a Newton update step after the tree structure has been found in tree-boosting
    *    Note: only used in GPBoost for tree-boosting (this is called from regression_objective). It is assume that 'CalcGetYAux' has been called before.
    * \param data_leaf_index Leaf index for every data point (array of size num_data)
    * \param num_leaves Number of leaves
    * \param[out] leaf_values Leaf values when performing a Newton update step (array of size num_leaves)
    */
    void NewtonUpdateLeafValues(const int* data_leaf_index,
      const int num_leaves, double* leaf_values) const;

  private:

    bool sparse_ = false;
    std::unique_ptr < REModelTemplate<sp_mat_t, chol_sp_mat_t> > re_model_sp_;
    std::unique_ptr < REModelTemplate<den_mat_t, chol_den_mat_t> > re_model_den_;
    vec_t cov_pars_;
    bool cov_pars_initialized_ = false;
    vec_t init_cov_pars_;
    bool init_cov_pars_provided_ = false;
    vec_t std_dev_cov_pars_;
    int num_cov_pars_;
    int num_it_ = 0;
    double lr_cov_ = 0.01;
    double acc_rate_cov_ = 0.5;
    int momentum_offset_ = 2;
    int max_iter_ = 1000;
    double delta_rel_conv_ = 1.0e-6;
    bool use_nesterov_acc_ = true;
    int nesterov_schedule_version_ = 0;
    bool optim_trace_ = false;
    string_t optimizer_cov_pars_ = "fisher_scoring";//"gradient_descent" or "fisher_scoring"
    vec_t coef_;
    bool has_covariates_ = false;
    bool coef_initialized_ = false;
    vec_t std_dev_coef_;
    double lr_coef_ = 0.001;
    double acc_rate_coef_ = 0.5;
    string_t optimizer_coef_ = "wls";//"gradient_descent" or "wls"

    /*!
    * \brief Check whether cov_pars_ is defined and if not define them as init_cov_pars_ and if init_cov_pars_ is not given, find "reasonable" default values for the intial values of the covariance parameters
    * \param y_data Response variable data
    */
    void CheckCovParsInitialized(const double* y_data);

  };

}  // namespace GPBoost

#endif   // GPB_RE_MODEL_H_
