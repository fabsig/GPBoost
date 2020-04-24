/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_RE_MODEL_TEMPLATE_H_
#define GPB_RE_MODEL_TEMPLATE_H_

#define _USE_MATH_DEFINES // for M_PI
#include <cmath>
#include <GPBoost/log.h>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <GPBoost/sparse_matrix_utils.h>
#include <GPBoost/Vecchia_utils.h>
#include <GPBoost/GP_utils.h>
//#include <Eigen/src/misc/lapack.h>

#include <memory>
#include <mutex>
#include <vector>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
//#include <typeinfo> // Only needed for debugging
//#include <chrono>  // Only needed for debugging
//#include <thread> // Only needed for debugging
//Log::Info("Fine here ");// Only for debugging
//std::this_thread::sleep_for(std::chrono::milliseconds(20));

namespace GPBoost {

	/*!
	* \brief Template class used in the wrapper class REModel
	* The template parameters T1 and T2 can either be <sp_mat_t, chol_sp_mat_t> or <den_mat_t, chol_den_mat_t>
	*/
	template<typename T1, typename T2>
	class REModelTemplate {
	public:
		/*! \brief Null costructor */
		REModelTemplate();

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
		REModelTemplate(data_size_t num_data, const gp_id_t* cluster_ids_data = nullptr, const char* re_group_data = nullptr,
			data_size_t num_re_group = 0, const double* re_group_rand_coef_data = nullptr,
			const int32_t* ind_effect_group_rand_coef = nullptr, data_size_t num_re_group_rand_coef = 0,
			data_size_t num_gp = 0, const double* gp_coords_data = nullptr, int dim_gp_coords = 2,
			const double* gp_rand_coef_data = nullptr, data_size_t num_gp_rand_coef = 0,
			const char* cov_fct = nullptr, double cov_fct_shape = 0., bool vecchia_approx = false, int num_neighbors = 30,
			const char* vecchia_ordering = nullptr, const char* vecchia_pred_type = nullptr, int num_neighbors_pred = 30) {

			num_cov_par_ = 1;
			CHECK(num_data > 0);
			num_data_ = num_data;
			vecchia_approx_ = vecchia_approx;

			//Set up GP IDs
			SetUpGPIds(num_data_, cluster_ids_data, num_data_per_cluster_, data_indices_per_cluster_, unique_clusters_, num_clusters_);
			//Indices of parameters of individual components in joint parameter vector
			ind_par_.push_back(0);//0+1 is starting point of parameter for first component since the first parameter is the nugget effect variance
			num_comps_total_ = 0;

			//Do some checks for grouped RE components and set meta data (number of components etc.)
			std::vector<std::vector<string_t>> re_group_levels;//Matrix with group levels for the grouped random effects (re_group_levels[j] contains the levels for RE number j)
			if (num_re_group > 0) {
				if (vecchia_approx) {
					Log::Fatal("The Veccia approximation cannot be used when there are grouped random effects (in the current implementation).");
				}
				num_re_group_ = num_re_group;
				CHECK(re_group_data != nullptr);
				if (num_re_group_rand_coef > 0) {
					num_re_group_rand_coef_ = num_re_group_rand_coef;
					CHECK(re_group_rand_coef_data != nullptr);
					CHECK(ind_effect_group_rand_coef != nullptr);
					for (int j = 0; j < num_re_group_rand_coef_; ++j) {
						CHECK(0 < ind_effect_group_rand_coef[j] && ind_effect_group_rand_coef[j] <= num_re_group_);
					}
					ind_effect_group_rand_coef_ = std::vector<int>(ind_effect_group_rand_coef, ind_effect_group_rand_coef + num_re_group_rand_coef_);
				}
				num_re_group_total_ = num_re_group_ + num_re_group_rand_coef_;
				num_cov_par_ += num_re_group_total_;
				num_comps_total_ += num_re_group_total_;
				//Add indices of parameters of individual components in joint parameter vector
				for (int j = 0; j < num_re_group_total_; ++j) {
					ind_par_.push_back(1 + j);//end points of parameter indices of components
				}
				// Convert characters in 'const char* re_group_data' to matrix (num_re_group_ x num_data_) with strings of group labels
				re_group_levels = std::vector<std::vector<string_t>>(num_re_group_, std::vector<string_t>(num_data_));
				if (num_re_group_ > 0) {
					ConvertCharToStringGroupLevels(num_data_, num_re_group_, re_group_data, re_group_levels);
				}
			}
			//Do some checks for GP components and set meta data (number of components etc.)
			if (num_gp > 0) {
				if (num_gp > 2) {
					Log::Fatal("num_gp can only be either 0 or 1 in the current implementation");
				}
				num_gp_ = num_gp;
				ind_intercept_gp_ = num_comps_total_;
				CHECK(dim_gp_coords > 0);
				CHECK(gp_coords_data != nullptr);
				CHECK(cov_fct != nullptr);
				dim_gp_coords_ = dim_gp_coords;
				cov_fct_ = std::string(cov_fct);
				cov_fct_shape_ = cov_fct_shape;
				if (vecchia_approx) {
					Log::Info("Starting nearest neighbor search for Vecchia approximation");
					CHECK(num_neighbors > 0);
					num_neighbors_ = num_neighbors;
					CHECK(num_neighbors_pred > 0);
					num_neighbors_pred_ = num_neighbors_pred;
					if (vecchia_ordering == nullptr) {
						vecchia_ordering_ = "none";
					}
					else {
						vecchia_ordering_ = std::string(vecchia_ordering);
						CHECK(vecchia_ordering_ == "none" || vecchia_ordering_ == "random");
					}
					if (vecchia_pred_type == nullptr) {
						vecchia_pred_type_ = "order_obs_first_cond_obs_only";
					}
					else {
						vecchia_pred_type_ = std::string(vecchia_pred_type);
						if (SUPPORTED_VECCHIA_PRED_TYPES_.find(vecchia_pred_type_) == SUPPORTED_VECCHIA_PRED_TYPES_.end()) {
							Log::Fatal("Prediction type '%s' is not supported for the Veccia approximation.", vecchia_pred_type_.c_str());
						}
					}
				}
				if (num_gp_rand_coef > 0) {//Random slopes
					CHECK(gp_rand_coef_data != nullptr);
					num_gp_rand_coef_ = num_gp_rand_coef;
				}
				num_gp_total_ = num_gp_ + num_gp_rand_coef_;
				num_cov_par_ += (2 * num_gp_total_);
				num_comps_total_ += num_gp_total_;
				//Add indices of parameters of individual components in joint parameter vector
				for (int j = 0; j < num_gp_total_; ++j) {
					ind_par_.push_back(ind_par_.back() + 2);//end points of parameter indices of components
				}

				if (vecchia_approx) {
					double num_mem_d = ((double)num_gp_total_) * ((double)num_data_) * ((double)num_neighbors_) * ((double)num_neighbors_);
					int mem_size = (int)(num_mem_d * 8. / 1000000.);
					if (mem_size > 8000) {
						Log::Warning("The current implementation of the Vecchia approximation is not optimized for memory usage. In your case (num. obs. = %d and num. neighbors = %d), at least approximately %d mb of memory is needed. If this is a problem, contact the developer of this package and ask to implement this feature.", num_data_, num_neighbors_, mem_size);
					}
				}

			}

			if (num_re_group_ > 0 && num_gp_total_ == 0) {
				do_symbolic_decomposition_ = true;//Symbolic decompostion is only done if sparse matrices are used
			}
			else {
				do_symbolic_decomposition_ = false;
			}

			//Create RE/GP component models
			for (const auto& cluster_i : unique_clusters_) {
				ConstructI<T1>(cluster_i);//Idendity matrices needed for computing inverses of covariance matrices used in gradient descent
				std::vector<std::shared_ptr<RECompBase<T1>>> re_comps_cluster_i;
				if (vecchia_approx_) {

					std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_data_per_cluster_[cluster_i]);
					std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_data_per_cluster_[cluster_i]);
					std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_data_per_cluster_[cluster_i]);
					std::vector<Triplet_t> entries_init_B_cluster_i;
					std::vector<Triplet_t> entries_init_B_grad_cluster_i;
					std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_data_per_cluster_[cluster_i]);

					CreateREComponentsVecchia(num_data_, data_indices_per_cluster_, cluster_i, num_data_per_cluster_,
						gp_coords_data, dim_gp_coords_, gp_rand_coef_data, num_gp_rand_coef_, cov_fct_, cov_fct_shape_, re_comps_cluster_i,
						nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i,
						entries_init_B_cluster_i, entries_init_B_grad_cluster_i,
						z_outer_z_obs_neighbors_cluster_i, vecchia_ordering_, num_neighbors_);

					nearest_neighbors_.insert({ cluster_i, nearest_neighbors_cluster_i });
					dist_obs_neighbors_.insert({ cluster_i, dist_obs_neighbors_cluster_i });
					dist_between_neighbors_.insert({ cluster_i, dist_between_neighbors_cluster_i });
					entries_init_B_.insert({ cluster_i, entries_init_B_cluster_i });
					entries_init_B_grad_.insert({ cluster_i, entries_init_B_grad_cluster_i });
					z_outer_z_obs_neighbors_.insert({ cluster_i, z_outer_z_obs_neighbors_cluster_i });

					Log::Info("Nearest neighbors for Vecchia approximation found");
				}
				else {

					CreateREComponents(num_data_, num_re_group_, data_indices_per_cluster_, cluster_i, re_group_levels, num_data_per_cluster_,
						num_re_group_rand_coef_, re_group_rand_coef_data, ind_effect_group_rand_coef_, num_gp_, gp_coords_data,
						dim_gp_coords_, gp_rand_coef_data, num_gp_rand_coef_, cov_fct_, cov_fct_shape_, ind_intercept_gp_, re_comps_cluster_i);

				}

				re_comps_.insert({ cluster_i, re_comps_cluster_i });
			}


			////Following only prints stuff for testing. TODO: delete

			//Log::Info("********************** Meta data ********************************");
			//Log::Info("num_data_ : %d", num_data_);
			//Log::Info("num_clusters_ : %d", num_clusters_);
			//Log::Info("num_re_group_ : %d", num_re_group_);
			//Log::Info("num_re_group_rand_coef_ : %d", num_re_group_rand_coef_);
			//Log::Info("num_re_group_total_ : %d", num_re_group_total_);
			//Log::Info("num_gp_rand_coef_ : %d", num_gp_rand_coef_);
			//Log::Info("num_gp_total_ : %d", num_gp_total_);
			//Log::Info("num_cov_par_: %d", num_cov_par_);
			//for (unsigned i = 0; i < ind_par_.size(); i++) { Log::Info("ind_par_[%d]: %d", i, ind_par_[i]); }

			//Log::Info("******************************************************");
			//int ii = 0;
			//for (const auto& cluster_i : unique_clusters_) {
			//	Log::Info("unique_clusters_[%d]: %d", ii, cluster_i);
			//	Log::Info("num_data_per_cluster_[%d]: %d", cluster_i, num_data_per_cluster_[cluster_i]);
			//	//for (int j = 0; j < std::min((int)data_indices_per_cluster_[cluster_i].size(), 10); ++j) { Log::Info("data_indices_per_cluster_[%d][%d]: %d", cluster_i, j, data_indices_per_cluster_[cluster_i][j]); }

			//	if (num_re_group_ > 0) {
			//		Log::Info("*********************** Grouped REs *******************************");
			//		//Log::Info("re_comps_[cluster_i] %s ", typeid(re_comps_[cluster_i]).name());
			//		//Log::Info("re_comps_[cluster_i].size(): %d", re_comps_[cluster_i].size());
			//		//for (const auto& re_comp : re_comps_[cluster_i]) {
			//		for (int j = 0; j < re_comps_[cluster_i].size(); ++j) {
			//			std::shared_ptr<RECompGroup<T1>> re_comp_group = std::dynamic_pointer_cast<RECompGroup<T1>>(re_comps_[cluster_i][j]);
			//			//for (const auto& el : re_comp_group->group_data_) { Log::Info("re_comps_[%d][j].group_data_[i]: %d", cluster_i, el); }
			//			if (!re_comp_group->is_rand_coef_) {
			//				for (int i = 0; i < std::min((int)(*re_comp_group->group_data_).size(), 10); i++) { Log::Info("re_comps_[%d][%d].group_data_[%d]: %s", cluster_i, j, i, (*re_comp_group->group_data_)[i]); }
			//			}
			//			else if (re_comp_group->is_rand_coef_) {
			//				for (int i = 0; i < std::min(num_data_per_cluster_[cluster_i], 10); i++) { Log::Info("re_comps_[%d][%d].group_data_ref_[%d]: %s", cluster_i, j, i, (*re_comp_group->group_data_)[i]); }
			//				for (int i = 0; i < std::min(num_data_per_cluster_[cluster_i], 10); i++) { Log::Info("re_comps_[%d][%d].rand_coef_data_[%d]: %f", cluster_i, j, i, re_comp_group->rand_coef_data_[i]); }
			//			}
			//		}
			//	}
			//	ii++;
			//}

		}

		/*! \brief Destructor */
		~REModelTemplate() {
		}

		/*! \brief Disable copy */
		REModelTemplate& operator=(const REModelTemplate&) = delete;

		/*! \brief Disable copy */
		REModelTemplate(const REModelTemplate&) = delete;

		/*!
		* \brief Find parameters that minimize the negative log-ligelihood (=MLE) using (Nesterov accelerated) gradient descent
		*		 Note: You should pre-allocate memory for optim_cov_pars (length = number of covariance parameters)
		* \param y_data Response variable data
		* \param init_cov_pars Initial values for covariance parameters of RE components
		* \param[out] optim_cov_pars Optimal covariance parameters
		* \param[out] num_it Number of iterations
		* \param lr Learning rate
		* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
		* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
		* \param max_iter Maximal number of iterations
		* \param delta_rel_conv Convergence criterion: stop iteration if relative change in parameters is below this value
		* \param optimizer Options: "gradient_descent" or "fisher_scoring"
		* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters. Default = true
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
		* \param[out] std_dev_cov_par Standard deviations for the covariance parameters
		* \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
		* \param cov_pars_lag_1 Covariance parameters from previous iteration used for Nesterov step (on transformed scale). Default = nullptr
		*/
		void OptimCovPar(const double* y_data, double* init_cov_pars, double* optim_cov_pars,
			int& num_it, double lr = 0.01, double acc_rate_cov = 0.5, int momentum_offset = 2,
			int max_iter = 1000, double delta_rel_conv = 1.0e-6, string_t optimizer = "fisher_scoring",
			bool use_nesterov_acc = true, int nesterov_schedule_version = 0,
			double* std_dev_cov_par = nullptr, bool calc_std_dev = false, double* cov_pars_lag_1 = nullptr) {
			if (SUPPORTED_OPTIM_COV_PAR_.find(optimizer) == SUPPORTED_OPTIM_COV_PAR_.end()) {
				Log::Fatal("Optimizer option '%s' is not supported for covariance parameters.", optimizer.c_str());
			}
			SetY(y_data);
			vec_t cov_pars = Eigen::Map<vec_t>(init_cov_pars, num_cov_par_);
			vec_t cov_pars_lag1 = (cov_pars_lag_1 == nullptr) ? cov_pars : cov_pars_lag1;
			num_it = max_iter;
			Log::Debug("Initial covariance parameters");
			for (int i = 0; i < (int)cov_pars.size(); ++i) { Log::Debug("cov_pars[%d]: %f", i, cov_pars[i]); }
			for (int it = 0; it < max_iter; ++it) {
				ApplyMomentumStep(it, cov_pars, cov_pars_lag1, use_nesterov_acc, acc_rate_cov, nesterov_schedule_version, true, momentum_offset);
				SetCovParsComps(cov_pars);
				CalcCovFactor(vecchia_approx_, true, 1., false);//Create covariance matrix and factorize it (and also calculate derivatives if Vecchia approximation is used)
				CalcYAux();
				if (optimizer == "gradient_descent") {//gradient descent
					UpdateCovParGradOneIter(lr, cov_pars, true);//closed_form_solution_sigma = true: we profile out sigma (=use closed for expression for error / nugget variance) since this is better for gradient descent (the paremeters usually live on different scales and the nugget needs a small learning rate but the others not...)
				}
				else if (optimizer == "fisher_scoring") {//Fisher scoring
					UpdateCovParFisherScoringOneIter(cov_pars, false);//closed_form_solution_sigma = false: we don't profile out sigma (=don't use closed for expression for error / nugget variance) since this is better for Fisher scoring (otherwise much more iterations are needed)
				}
				CheckNaN(cov_pars);

				if (it < 10 || ((it + 1) % 10 == 0 && (it + 1) < 100) || ((it + 1) % 100 == 0 && (it + 1) < 1000) || ((it + 1) % 1000 == 0 && (it + 1) < 10000) || ((it + 1) % 10000 == 0)) {
					Log::Debug("Covariance parameter estimation: iteration number %d", it + 1);
					for (int i = 0; i < (int)cov_pars.size(); ++i) { Log::Debug("cov_pars[%d]: %f", i, cov_pars[i]); }
				}
				if ((cov_pars - cov_pars_lag1).norm() / cov_pars_lag1.norm() < delta_rel_conv) {
					num_it = it + 1;
					break;
				}
			}
			if (num_it == max_iter) {
				Log::Warning("Covariance parameter estimation: no convergence after the maximal number of iterations. If this is a problem, you might consider increasing the number of iterations or using a different learning rate.");
			}
			for (int i = 0; i < num_cov_par_; ++i) {
				optim_cov_pars[i] = cov_pars[i];
			}
			if (calc_std_dev) {
				vec_t std_dev_cov(num_cov_par_);
				CalcStdDevCovPar(cov_pars, std_dev_cov);
				for (int i = 0; i < num_cov_par_; ++i) {
					std_dev_cov_par[i] = std_dev_cov[i];
				}
			}
			has_covariates_ = false;
		}

		/*!
		* \brief Find linear regression coefficients and covariance parameters that minimize the negative log-ligelihood (=MLE) using (Nesterov accelerated) gradient descent
		*		 Note: You should pre-allocate memory for optim_cov_pars and optim_coef. Their length equal the number of covariance parameters and the number of regression coefficients
		*           If calc_std_dev=true, you also need to pre-allocate memory for std_dev_cov_par and std_dev_coef of the same length for the standard deviations
		* \param y_data Response variable data
		* \param covariate_data Covariate data (=independent variables, features)
		* \param num_covariates Number of covariates
		* \param[out] optim_cov_pars Optimal covariance parameters
		* \param[out] optim_coef Optimal regression coefficients
		* \param[out] num_it Number of iterations
		* \param init_cov_pars Initial values for covariance parameters of RE components
		* \param init_coef Initial values for the regression coefficients
		* \param lr_coef Learning rate for fixed-effect linear coefficients
		* \param lr_cov Learning rate for covariance parameters
		* \param acc_rate_coef Acceleration rate for coefficients for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
		* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
		* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
		* \param max_iter Maximal number of iterations
		* \param delta_rel_conv Convergence criterion: stop iteration if relative change in in parameters is below this value
		* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters. Default = true
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
		* \param optimizer_cov Optimizer for covariance parameters. Options: "gradient_descent" or "fisher_scoring"
		* \param optimizer_coef Optimizer for coefficients. Options: "gradient_descent" or "wls" (coordinate descent using weighted least squares)
		* \param[out] std_dev_cov_par Standard deviations for the covariance parameters
		* \param[out] std_dev_coef Standard deviations for the coefficients
		* \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
		*/
		void OptimLinRegrCoefCovPar(const double* y_data, const double* covariate_data, int num_covariates,
			double* optim_cov_pars, double* optim_coef, int& num_it, double* init_cov_pars, double* init_coef = nullptr,
			double lr_coef = 0.01, double lr_cov = 0.01, double acc_rate_coef = 0.1, double acc_rate_cov = 0.5, int momentum_offset = 2,
			int max_iter = 1000, double delta_rel_conv = 1.0e-6, bool use_nesterov_acc = true, int nesterov_schedule_version = 0,
			string_t optimizer_cov = "fisher_scoring", string_t optimizer_coef = "wls", double* std_dev_cov_par = nullptr,
			double* std_dev_coef = nullptr, bool calc_std_dev = false) {
			if (SUPPORTED_OPTIM_COV_PAR_.find(optimizer_cov) == SUPPORTED_OPTIM_COV_PAR_.end()) {
				Log::Fatal("Optimizer option '%s' is not supported for covariance parameters.", optimizer_cov.c_str());
			}
			if (SUPPORTED_OPTIM_COEF_.find(optimizer_coef) == SUPPORTED_OPTIM_COEF_.end()) {
				Log::Fatal("Optimizer option '%s' is not supported for regression coefficients.", optimizer_coef.c_str());
			}
			CHECK(covariate_data != nullptr);
			has_covariates_ = true;
			num_coef_ = num_covariates;
			X_ = Eigen::Map<const den_mat_t>(covariate_data, num_data_, num_coef_);
			//Check whether one of the colums contains only 1's and if not, give out warning
			vec_t vec_ones(num_data_);
			vec_ones.setOnes();
			bool has_intercept = false;
			for (int icol = 0; icol < num_coef_; ++icol) {
				if ((X_.col(icol) - vec_ones).cwiseAbs().sum() < 0.001) {
					has_intercept = true;
					break;
				}
			}
			if (!has_intercept) {
				Log::Warning("The covariate data contains no column of ones. This means that there is no intercept included.");
			}
			y_vec_ = Eigen::Map<const vec_t>(y_data, num_data_);
			vec_t cov_pars = Eigen::Map<const vec_t>(init_cov_pars, num_cov_par_);
			vec_t cov_pars_lag1 = cov_pars;
			vec_t beta(num_covariates);
			if (init_coef == nullptr) {
				beta.setZero();
			}
			else {
				beta = Eigen::Map<const vec_t>(init_coef, num_covariates);
			}
			vec_t beta_lag1 = beta;
			vec_t resid;
			num_it = max_iter;
			for (int it = 0; it < max_iter; ++it) {
				if (it > 0) {
					ApplyMomentumStep(it, cov_pars, cov_pars_lag1, use_nesterov_acc, acc_rate_cov, nesterov_schedule_version, true, momentum_offset);
					if (optimizer_coef == "gradient_descent") {
						ApplyMomentumStep(it, beta, beta_lag1, use_nesterov_acc, acc_rate_coef, nesterov_schedule_version, false, momentum_offset);
					}
				}
				SetCovParsComps(cov_pars);
				CalcCovFactor(vecchia_approx_, true, 1., false);
				//Update linear regression coefficients
				if (optimizer_coef == "gradient_descent") {//one step of gradient descent
					resid = y_vec_ - (X_ * beta);
					SetY(resid.data());
					CalcYAux();
					UpdateCoefGradOneIter(lr_coef, cov_pars[0], X_, beta);
				}
				else if (optimizer_coef == "wls") {//coordinate descent using generalized least squares
					SetY(y_vec_.data());
					CalcYAux();
					beta_lag1 = beta;
					UpdateCoefGLS(X_, beta);
				}
				//Update covariance parameters
				resid = y_vec_ - (X_ * beta);
				SetY(resid.data());
				CalcYAux();
				if (optimizer_cov == "gradient_descent") {//one step of gradient descent
					UpdateCovParGradOneIter(lr_cov, cov_pars, true);//closed_form_solution_sigma = true: we profile out sigma (=use closed for expression for error / nugget variance) since this is better for gradient descent (the paremeters usually live on different scales and the nugget needs a small learning rate but the others not...)
				}
				else if (optimizer_cov == "fisher_scoring") {//one step of Fisher scoring
					UpdateCovParFisherScoringOneIter(cov_pars, false);//closed_form_solution_sigma = false: we don't profile out sigma (=don't use closed for expression for error / nugget variance) since this is better for Fisher scoring (otherwise much more iterations are needed)
				}
				CheckNaN(cov_pars);
				if (it < 10 || ((it + 1) % 10 == 0 && (it + 1) < 100) || ((it + 1) % 100 == 0 && (it + 1) < 1000) || ((it + 1) % 1000 == 0 && (it + 1) < 10000) || ((it + 1) % 10000 == 0)) {
					Log::Debug("Gradient descent iteration number %d", it + 1);
					for (int i = 0; i < (int)cov_pars.size(); ++i) { Log::Debug("cov_pars[%d]: %f", i, cov_pars[i]); }
					for (int i = 0; i < std::min((int)beta.size(), 3); ++i) { Log::Debug("beta[%d]: %f", i, beta[i]); }
				}
				if (((beta - beta_lag1).norm() / beta_lag1.norm() < delta_rel_conv) && ((cov_pars - cov_pars_lag1).norm() / cov_pars_lag1.norm() < delta_rel_conv)) {
					num_it = it + 1;
					break;
				}
			}
			if (num_it == max_iter) {
				Log::Warning("Covariance parameter estimation: no convergence after the maximal number of iterations");
			}
			for (int i = 0; i < num_cov_par_; ++i) {
				optim_cov_pars[i] = cov_pars[i];
			}
			if (calc_std_dev) {
				vec_t std_dev_cov(num_cov_par_);
				CalcStdDevCovPar(cov_pars, std_dev_cov);
				for (int i = 0; i < num_cov_par_; ++i) {
					std_dev_cov_par[i] = std_dev_cov[i];
				}
			}
			for (int i = 0; i < num_covariates; ++i) {
				optim_coef[i] = beta[i];
			}
			if (calc_std_dev) {
				vec_t std_dev_beta(num_covariates);
				CalcStdDevCoef(cov_pars, X_, std_dev_beta);
				for (int i = 0; i < num_covariates; ++i) {
					std_dev_coef[i] = std_dev_beta[i];
				}
			}
		}

		/*!
		* \brief Calculate the value of the negative log-likelihood
		* \param y_data Response variable data
		* \param cov_pars Values for covariance parameters of RE components
		* \param[out] negll Negative log-likelihood
		*/
		void EvalNegLogLikelihood(const double* y_data, double* cov_pars, double& negll) {
			negll = 0.;
			SetY(y_data);
			vec_t cov_pars_vec = Eigen::Map<vec_t>(cov_pars, num_cov_par_);
			SetCovParsComps(cov_pars_vec);
			CalcCovFactor(false, true, 1., false);//Create covariance matrix and factorize it
			//Calculate quadratic form
			double yTPsiInvy = 0.;
			CalcYTPsiIInvY<T1>(yTPsiInvy);
			//Calculate log determinant
			double log_det = 0;
			for (const auto& cluster_i : unique_clusters_) {
				if (vecchia_approx_) {
					log_det -= D_inv_[cluster_i].diagonal().array().log().sum();
				}
				else {
					log_det += (2. * chol_facts_[cluster_i].diagonal().array().log().sum());
				}
			}
			negll = yTPsiInvy / 2. / cov_pars[0] + log_det / 2. + num_data_ / 2. * (std::log(cov_pars[0]) + std::log(2 * M_PI));
		}

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
		void SetPredictionData(int num_data_pred,
			const gp_id_t* cluster_ids_data_pred = nullptr, const char* re_group_data_pred = nullptr,
			const double* re_group_rand_coef_data_pred = nullptr, double* gp_coords_data_pred = nullptr,
			const double* gp_rand_coef_data_pred = nullptr, const double* covariate_data_pred = nullptr) {
			if (cluster_ids_data_pred == nullptr) {
				cluster_ids_data_pred_.clear();
			}
			else {
				cluster_ids_data_pred_ = std::vector<gp_id_t>(cluster_ids_data_pred, cluster_ids_data_pred + num_data_pred);
			}
			if (re_group_data_pred == nullptr) {
				re_group_levels_pred_.clear();
				if (num_re_group_ > 0) {
					Log::Fatal("No group data is provided for making predictions");
				}
			}
			else {
				//For grouped random effecst: create matrix 're_group_levels_pred' (vector of vectors, dimension: num_re_group_ x num_data_) with strings of group levels from characters in 'const char* re_group_data_pred'
				re_group_levels_pred_ = std::vector<std::vector<string_t>>(num_re_group_, std::vector<string_t>(num_data_pred));
				ConvertCharToStringGroupLevels(num_data_pred, num_re_group_, re_group_data_pred, re_group_levels_pred_);
			}
			if (re_group_rand_coef_data_pred == nullptr) {
				re_group_rand_coef_data_pred_.clear();
			}
			else {
				re_group_rand_coef_data_pred_ = std::vector<double>(re_group_rand_coef_data_pred, re_group_rand_coef_data_pred + num_data_pred * num_re_group_rand_coef_);
			}
			if (gp_coords_data_pred == nullptr) {
				gp_coords_data_pred_.clear();
			}
			else {
				gp_coords_data_pred_ = std::vector<double>(gp_coords_data_pred, gp_coords_data_pred + num_data_pred * dim_gp_coords_);
			}
			if (gp_rand_coef_data_pred == nullptr) {
				gp_rand_coef_data_pred_.clear();
			}
			else {
				gp_rand_coef_data_pred_ = std::vector<double>(gp_rand_coef_data_pred, gp_rand_coef_data_pred + num_data_pred * num_gp_rand_coef_);
			}
			if (covariate_data_pred == nullptr) {
				covariate_data_pred_.clear();
			}
			else {
				covariate_data_pred_ = std::vector<double>(covariate_data_pred, covariate_data_pred + num_data_pred * num_coef_);
			}
		}

		/*!
		* \brief Make predictions: calculate conditional mean and covariance matrix
		*		 Note: You should pre-allocate memory for out_predict
		*			   Its length is equal to num_data_pred if only the conditional mean is predicted (predict_cov_mat=false)
		*			   or num_data_pred * (1 + num_data_pred) if both the conditional mean and covariance matrix are predicted (predict_cov_mat=true)
		* \param cov_pars_pred Covariance parameters of components
		* \param y_obs Response variable for observed data
		* \param num_data_pred Number of data points for which predictions are made
		* \param[out] out_predict Conditional mean at prediciton points (="predicted value") followed by (if predict_cov_mat=true) the conditional covariance matrix at in column-major format
		* \param predict_cov_mat If true, the conditional covariance matrix is calculated (default=false)
		* \param covariate_data_pred Covariate data (=independent variables, features) for prediction
		* \param coef_pred Coefficients for linear covariates
		* \param cluster_ids_data_pred IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization) for which predictions are to be made
		* \param re_group_data_pred Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
		* \param re_group_rand_coef_data_pred Covariate data for grouped random coefficients
		* \param gp_coords_data_pred Coordinates (features) for Gaussian process
		* \param gp_rand_coef_data_pred Covariate data for Gaussian process random coefficients
		* \param use_saved_data If true, saved data is used and some arguments are ignored
		* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
		* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions (-1 means that the value already set at initialization is used)
		*/
		void Predict(const double* cov_pars_pred, const double* y_obs, data_size_t num_data_pred,
			double* out_predict, bool predict_cov_mat = false,
			const double* covariate_data_pred = nullptr, const double* coef_pred = nullptr,
			const gp_id_t* cluster_ids_data_pred = nullptr, const char* re_group_data_pred = nullptr,
			const double* re_group_rand_coef_data_pred = nullptr, double* gp_coords_data_pred = nullptr,
			const double* gp_rand_coef_data_pred = nullptr, bool use_saved_data = false,
			const char* vecchia_pred_type = nullptr, int num_neighbors_pred = -1) {
			//Should previously set data be used?
			std::vector<std::vector<string_t>> re_group_levels_pred;//Matrix with group levels for the grouped random effects (re_group_levels_pred[j] contains the levels for RE number j)
			if (use_saved_data) {
				re_group_levels_pred = re_group_levels_pred_;
				if (cluster_ids_data_pred_.empty()) {
					cluster_ids_data_pred = nullptr;
				}
				else {
					cluster_ids_data_pred = cluster_ids_data_pred_.data();
				}
				if (re_group_rand_coef_data_pred_.empty()) {
					re_group_rand_coef_data_pred = nullptr;
				}
				else {
					re_group_rand_coef_data_pred = re_group_rand_coef_data_pred_.data();
				}
				if (gp_coords_data_pred_.empty()) {
					gp_coords_data_pred = nullptr;
				}
				else {
					gp_coords_data_pred = gp_coords_data_pred_.data();
				}
				if (gp_rand_coef_data_pred_.empty()) {
					gp_rand_coef_data_pred = nullptr;
				}
				else {
					gp_rand_coef_data_pred = gp_rand_coef_data_pred_.data();
				}
				if (covariate_data_pred_.empty()) {
					covariate_data_pred = nullptr;
				}
				else {
					covariate_data_pred = covariate_data_pred_.data();
				}
			}
			else {
				if (num_re_group_ > 0) {
					if (re_group_data_pred == nullptr) {
						Log::Fatal("No group data is provided for making predictions");
					}
					else {
						//For grouped random effecst: create matrix 're_group_levels_pred' (vector of vectors, dimension: num_re_group_ x num_data_) with strings of group levels from characters in 'const char* re_group_data_pred'
						re_group_levels_pred = std::vector<std::vector<string_t>>(num_re_group_, std::vector<string_t>(num_data_pred));
						ConvertCharToStringGroupLevels(num_data_pred, num_re_group_, re_group_data_pred, re_group_levels_pred);
					}
				}
			}
			//Some checks
			CHECK(num_data_pred > 0);
			if (has_covariates_) {
				CHECK(covariate_data_pred != nullptr);
				CHECK(coef_pred != nullptr);
			}
			if (y_obs == nullptr) {
				if (y_.empty()) {
					Log::Fatal("Observed data is not provided and has not been set before");
				}
			}
			//Check whether some data is missing
			if (re_group_rand_coef_data_pred == nullptr && num_re_group_rand_coef_ > 0) {
				Log::Fatal("No covariate data for grouped random coefficients is provided for making predictions");
			}
			if (gp_coords_data_pred == nullptr && num_gp_ > 0) {
				Log::Warning("No coordinate data for the Gaussian process is provided for making predictions");
			}
			if (gp_rand_coef_data_pred == nullptr && num_gp_rand_coef_ > 0) {
				Log::Warning("No covariate data for Gaussian process random coefficients is provided for making predictions");
			}
			if (num_data_pred > 10000 && predict_cov_mat) {
				double num_mem_d = ((double)num_data_pred) * ((double)num_data_pred);
				int mem_size = (int)(num_mem_d * 8. / 1000000.);
				Log::Warning("The covariance matrix can be very large for large sample sizes which might lead to memory limitations. In your case (n = %d), the covariance needs at least approximately %d mb of memory. If you only need variances or covariances for linear combinations, contact the developer of this package and ask to implement this feature.", num_data_pred, mem_size);
			}

			if (vecchia_approx_) {
				if (vecchia_pred_type != nullptr) {
					string_t vecchia_pred_type_S = std::string(vecchia_pred_type);
					CHECK(vecchia_pred_type_S == "order_obs_first_cond_obs_only" ||
						vecchia_pred_type_S == "order_obs_first_cond_all" ||
						vecchia_pred_type_S == "order_pred_first" ||
						vecchia_pred_type_S == "latent_order_obs_first_cond_obs_only" ||
						vecchia_pred_type_S == "latent_order_obs_first_cond_all");
					vecchia_pred_type_ = vecchia_pred_type_S;
				}
				if (num_neighbors_pred > 0) {
					num_neighbors_pred_ = num_neighbors_pred;
				}
			}

			vec_t coef;
			if (has_covariates_) {
				coef = Eigen::Map<const vec_t>(coef_pred, num_coef_);
				den_mat_t X_pred = Eigen::Map<const den_mat_t>(covariate_data_pred, num_data_pred, num_coef_);
				vec_t mu = X_pred * coef;
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_pred; ++i) {
					out_predict[i] = mu[i];
				}
			}
			vec_t cov_pars = Eigen::Map<const vec_t>(cov_pars_pred, num_cov_par_);
			//Set up cluster IDs
			std::map<gp_id_t, int> num_data_per_cluster_pred;
			std::map<gp_id_t, std::vector<int>> data_indices_per_cluster_pred;
			std::vector<gp_id_t> unique_clusters_pred;
			data_size_t num_clusters_pred;
			SetUpGPIds(num_data_pred, cluster_ids_data_pred, num_data_per_cluster_pred,
				data_indices_per_cluster_pred, unique_clusters_pred, num_clusters_pred);

			//Check whether predictions are made for existing clusters or if only for new independet clusters predictions are made
			bool pred_for_observed_data = false;
			for (const auto& cluster_i : unique_clusters_pred) {
				if (std::find(unique_clusters_.begin(), unique_clusters_.end(), cluster_i) != unique_clusters_.end()) {
					pred_for_observed_data = true;
					break;
				}
			}

			//Factorize covariance matrix and calculate Psi^{-1}y_obs (if required for prediction)
			if (pred_for_observed_data) {//TODO: this acutally needs to be done only for the GP realizations for which predictions are made (currently it is done for all of them in unique_clusters_pred)
				if (has_covariates_) {
					vec_t resid;
					if (y_obs != nullptr) {
						vec_t y = Eigen::Map<const vec_t>(y_obs, num_data_);
						resid = y - (X_ * coef);
					}
					else {
						resid = y_vec_ - (X_ * coef);
					}
					SetY(resid.data());
				}
				else {
					if (y_obs != nullptr) {
						SetY(y_obs);
					}
				}

				SetCovParsComps(cov_pars);
				if (!vecchia_approx_) {
					CalcCovFactor(false, true, 1., false);//no need to do this for the Vecchia approximation, is done in the prediction steps
					CalcYAux();
				}
			}//end if(pred_for_observed_data)

			//Initialize covariance matrix
			if (predict_cov_mat) {//TODO: avoid unnecessary initialization (only set to 0 for covariances accross different realizations of GPs)
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (num_data_pred * num_data_pred); ++i) {
					out_predict[i + num_data_pred] = 0.;
				}
			}

			for (const auto& cluster_i : unique_clusters_pred) {

				//no data observed for this Gaussian process with ID 'cluster_i'. Thus use prior mean (0) and prior covariance matrix
				if (std::find(unique_clusters_.begin(), unique_clusters_.end(), cluster_i) == unique_clusters_.end()) {

					if (!has_covariates_) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
							out_predict[data_indices_per_cluster_pred[cluster_i][i]] = 0.;
						}
					}

					if (predict_cov_mat) {
						T1 psi;
						std::vector<std::shared_ptr<RECompBase<T1>>> re_comps_cluster_i;

						if (vecchia_approx_) {

							std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_data_per_cluster_pred[cluster_i]);
							std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_data_per_cluster_pred[cluster_i]);
							std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_data_per_cluster_pred[cluster_i]);
							std::vector<Triplet_t> entries_init_B_cluster_i;
							std::vector<Triplet_t> entries_init_B_grad_cluster_i;
							std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_data_per_cluster_pred[cluster_i]);

							CreateREComponentsVecchia(num_data_pred, data_indices_per_cluster_pred, cluster_i, num_data_per_cluster_pred,
								gp_coords_data_pred, dim_gp_coords_, gp_rand_coef_data_pred, num_gp_rand_coef_, cov_fct_, cov_fct_shape_, re_comps_cluster_i,
								nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i,
								entries_init_B_cluster_i, entries_init_B_grad_cluster_i,
								z_outer_z_obs_neighbors_cluster_i, "none", num_neighbors_pred_);//TODO: maybe also use ordering for making predictions? (need to check that there are not errors)

							for (int j = 0; j < num_comps_total_; ++j) {
								const vec_t pars = cov_pars.segment(ind_par_[j] + 1, ind_par_[j + 1] - ind_par_[j]);
								re_comps_cluster_i[j]->SetCovPars(pars);
							}

							sp_mat_t B_cluster_i;
							sp_mat_t D_inv_cluster_i;
							std::vector<sp_mat_t> B_grad_cluster_i;//not used, but needs to be passed to function
							std::vector<sp_mat_t> D_grad_cluster_i;//not used, but needs to be passed to function
							CalcCovFactorVecchia(num_data_per_cluster_pred[cluster_i], false, re_comps_cluster_i,
								nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i,
								entries_init_B_cluster_i, entries_init_B_grad_cluster_i,
								z_outer_z_obs_neighbors_cluster_i,
								B_cluster_i, D_inv_cluster_i, B_grad_cluster_i, D_grad_cluster_i);

							//Calculate Psi
							sp_mat_t D_sqrt(num_data_per_cluster_pred[cluster_i], num_data_per_cluster_pred[cluster_i]);
							D_sqrt.setIdentity();
							D_sqrt.diagonal().array() = D_inv_cluster_i.diagonal().array().pow(-0.5);

							sp_mat_t B_inv_D_sqrt;
							eigen_sp_Lower_sp_RHS_cs_solve(B_cluster_i, D_sqrt, B_inv_D_sqrt, true);
							psi = B_inv_D_sqrt * B_inv_D_sqrt.transpose();

						}//end Vecchia

						else {

							psi.resize(num_data_per_cluster_pred[cluster_i], num_data_per_cluster_pred[cluster_i]);
							psi.setIdentity();
							CreateREComponents(num_data_pred, num_re_group_, data_indices_per_cluster_pred, cluster_i, re_group_levels_pred, num_data_per_cluster_pred,
								num_re_group_rand_coef_, re_group_rand_coef_data_pred, ind_effect_group_rand_coef_, num_gp_, gp_coords_data_pred,
								dim_gp_coords_, gp_rand_coef_data_pred, num_gp_rand_coef_, cov_fct_, cov_fct_shape_, ind_intercept_gp_, re_comps_cluster_i);
							for (int j = 0; j < num_comps_total_; ++j) {
								const vec_t pars = cov_pars.segment(ind_par_[j] + 1, ind_par_[j + 1] - ind_par_[j]);
								re_comps_cluster_i[j]->SetCovPars(pars);
								re_comps_cluster_i[j]->CalcSigma();
								psi += (*(re_comps_cluster_i[j]->GetZSigmaZt().get()));
							}
						}//end not Vecchia

						psi *= cov_pars[0];

						//write on output
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {//column index
							for (int j = 0; j < num_data_per_cluster_pred[cluster_i]; ++j) {//row index
								out_predict[data_indices_per_cluster_pred[cluster_i][i] * num_data_pred + data_indices_per_cluster_pred[cluster_i][j] + num_data_pred] = psi.coeff(j, i);
							}
						}

					}//end predict_cov_mat

				}//end cluster_i with no observed data
				else {//there exists observed data for this cluster_i (= typical case)

					den_mat_t gp_coords_mat_pred;
					if (num_gp_ > 0) {
						std::vector<double> gp_coords_pred;
						for (int j = 0; j < dim_gp_coords_; ++j) {
							for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
								gp_coords_pred.push_back(gp_coords_data_pred[j * num_data_pred + id]);
							}
						}
						gp_coords_mat_pred = Eigen::Map<den_mat_t>(gp_coords_pred.data(), num_data_per_cluster_pred[cluster_i], dim_gp_coords_);
					}

					vec_t mean_pred_id(num_data_per_cluster_pred[cluster_i]);
					T1 cov_mat_pred_id;
					if (predict_cov_mat) {
						cov_mat_pred_id = T1(num_data_per_cluster_pred[cluster_i], num_data_per_cluster_pred[cluster_i]);
					}

					if (vecchia_approx_) {

						std::shared_ptr<RECompGP<T1>> re_comp = std::dynamic_pointer_cast<RECompGP<T1>>(re_comps_[cluster_i][ind_intercept_gp_]);

						int num_data_tot = num_data_per_cluster_[cluster_i] + num_data_per_cluster_pred[cluster_i];
						double num_mem_d = ((double)num_neighbors_pred_) * ((double)num_neighbors_pred_) * (double)(num_data_tot)+(double)(num_neighbors_pred_) * (double)(num_data_tot);
						int mem_size = (int)(num_mem_d * 8. / 1000000.);
						if (mem_size > 4000) {
							Log::Warning("The current implementation of the Vecchia approximation needs a lot of memory if the number of neighbors is large. In your case (nb. of neighbors = %d, nb. of observations = %d, nb. of predictions = %d), this needs at least approximately %d mb of memory. If this is a problem for you, contact the developer of this package and ask to change this.", num_neighbors_pred_, num_data_per_cluster_[cluster_i], num_data_per_cluster_pred[cluster_i], mem_size);
						}

						if (vecchia_pred_type_ == "order_obs_first_cond_obs_only") {
							CalcPredVecchiaObservedFirstOrder(true, cluster_i, num_data_pred, num_data_per_cluster_pred, data_indices_per_cluster_pred,
								re_comp->coords_, gp_coords_mat_pred, gp_rand_coef_data_pred,
								predict_cov_mat, mean_pred_id, cov_mat_pred_id);
						}
						else if (vecchia_pred_type_ == "order_obs_first_cond_all") {
							CalcPredVecchiaObservedFirstOrder(false, cluster_i, num_data_pred, num_data_per_cluster_pred, data_indices_per_cluster_pred,
								re_comp->coords_, gp_coords_mat_pred, gp_rand_coef_data_pred,
								predict_cov_mat, mean_pred_id, cov_mat_pred_id);
						}
						else if (vecchia_pred_type_ == "order_pred_first") {
							CalcPredVecchiaPredictedFirstOrder(cluster_i, num_data_pred, num_data_per_cluster_pred, data_indices_per_cluster_pred,
								re_comp->coords_, gp_coords_mat_pred, gp_rand_coef_data_pred,
								predict_cov_mat, mean_pred_id, cov_mat_pred_id);
						}
						else if (vecchia_pred_type_ == "latent_order_obs_first_cond_obs_only") {
							CalcPredVecchiaLatentObservedFirstOrder(true, cluster_i, num_data_per_cluster_pred,
								re_comp->coords_, gp_coords_mat_pred, predict_cov_mat, mean_pred_id, cov_mat_pred_id);
						}
						else if (vecchia_pred_type_ == "latent_order_obs_first_cond_all") {
							CalcPredVecchiaLatentObservedFirstOrder(false, cluster_i, num_data_per_cluster_pred,
								re_comp->coords_, gp_coords_mat_pred, predict_cov_mat, mean_pred_id, cov_mat_pred_id);
						}

					}//end Vecchia approximation
					else {

						CalcPred(cluster_i, num_data_pred, num_data_per_cluster_pred, data_indices_per_cluster_pred,
							re_group_levels_pred, re_group_rand_coef_data_pred, gp_coords_mat_pred, gp_rand_coef_data_pred,
							predict_cov_mat, mean_pred_id, cov_mat_pred_id);

					}//end not Vecchia approximation

					//write on output
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
						if (has_covariates_) {
							out_predict[data_indices_per_cluster_pred[cluster_i][i]] += mean_pred_id[i];
						}
						else {
							out_predict[data_indices_per_cluster_pred[cluster_i][i]] = mean_pred_id[i];
						}
					}
					if (predict_cov_mat) {
						cov_mat_pred_id *= cov_pars[0];
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {//column index
							for (int j = 0; j < num_data_per_cluster_pred[cluster_i]; ++j) {//row index
								out_predict[data_indices_per_cluster_pred[cluster_i][i] * num_data_pred + data_indices_per_cluster_pred[cluster_i][j] + num_data_pred] = cov_mat_pred_id.coeff(j, i);//cov_mat_pred_id_den(j, i);
							}
						}
					}

				}//end cluster_i with data
			}//end loop over cluster
		}

		/*!
		* \brief Find "reasonable" default values for the intial values of the covariance parameters (on transformed scale)
		*		 Note: You should pre-allocate memory for optim_cov_pars (length = number of covariance parameters)
		* \param y_data Response variable data
		* \param[out] init_cov_pars Initial values for covariance parameters of RE components
		*/
		void FindInitCovPar(const double* y_data, double* init_cov_pars) {
			double mean = 0;
			for (int i = 0; i < num_data_; ++i) {
				mean += y_data[i];
			}
			mean /= num_data_;
			double var = 0;
			for (int i = 0; i < num_data_; ++i) {
				var += (y_data[i] - mean) * (y_data[i] - mean);
			}
			var /= (num_data_ - 1);
			init_cov_pars[0] = var;

			int ind_par = 1;
			for (int j = 0; j < num_comps_total_; ++j) {
				int num_par_j = ind_par_[j + 1] - ind_par_[j];
				vec_t pars = vec_t(num_par_j);
				re_comps_[unique_clusters_[0]][j]->FindInitCovPar(pars);
				for (int jj = 0; jj < num_par_j; ++jj) {
					init_cov_pars[ind_par] = pars[jj];
					ind_par++;
				}
			}
		}

		int num_cov_par() {
			return(num_cov_par_);
		}

		/*!
		* \brief Calculate the leaf values when performing a Newton update step after the tree structure has been found in tree-boosting
		*    Note: only used in GPBoost for tree-boosting (this is called from regression_objective). It is assume that 'CalcYAux' has been called before.
		* \param data_leaf_index Leaf index for every data point (array of size num_data)
		* \param num_leaves Number of leaves
		* \param[out] leaf_values Leaf values when performing a Newton update step (array of size num_leaves)
		* \param marg_variance The marginal variance. Default = 1. Can be used to multiply values by it since Newton updates do not depend on it but 'CalcYAux' might have been called using marg_variance!=1.
		*/
		void NewtonUpdateLeafValues(const int* data_leaf_index,
			const int num_leaves, double* leaf_values, double marg_variance = 1.) {
			CHECK(y_aux_has_been_calculated_);
			den_mat_t HTPsiInvH(num_leaves, num_leaves);
			vec_t HTYAux(num_leaves);
			HTPsiInvH.setZero();
			HTYAux.setZero();

			for (const auto& cluster_i : unique_clusters_) {

				//Entries for matrix H_cluster_i = incidence matrix H that relates tree leaves to observations for cluster_i
				std::vector<Triplet_t> entries_H_cluster_i(num_data_per_cluster_[cluster_i]);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
					entries_H_cluster_i[i] = Triplet_t(i, data_leaf_index[data_indices_per_cluster_[cluster_i][i]], 1.);
				}

				if (vecchia_approx_) {
					sp_mat_t H_cluster_i(num_data_per_cluster_[cluster_i], num_leaves);//row major format is needed for Vecchia approx.
					H_cluster_i.setFromTriplets(entries_H_cluster_i.begin(), entries_H_cluster_i.end());

					HTYAux -= H_cluster_i.transpose() * y_aux_[cluster_i];//minus sign since y_aux_ has been calculated on the gradient = F-y (and not y-F)
					sp_mat_t BH = B_[cluster_i] * H_cluster_i;
					den_mat_t HTPsiInvH_cluster_i = den_mat_t(BH.transpose() * D_inv_[cluster_i] * BH);
					HTPsiInvH += HTPsiInvH_cluster_i;
				}
				else {
					sp_mat_t H_cluster_i(num_data_per_cluster_[cluster_i], num_leaves);
					H_cluster_i.setFromTriplets(entries_H_cluster_i.begin(), entries_H_cluster_i.end());

					HTYAux -= H_cluster_i.transpose() * y_aux_[cluster_i];//minus sign since y_aux_ has been calculated on the gradient = F-y (and not y-F)
					T1 PsiInvSqrtH;
					CalcPsiInvSqrtH(PsiInvSqrtH, H_cluster_i, cluster_i);
					den_mat_t HTPsiInvH_cluster_i = PsiInvSqrtH.transpose() * PsiInvSqrtH;
					HTPsiInvH += HTPsiInvH_cluster_i;
				}
			}

			HTYAux *= marg_variance;
			vec_t new_leaf_values = HTPsiInvH.llt().solve(HTYAux);
			for (int i = 0; i < num_leaves; ++i) {
				leaf_values[i] = new_leaf_values[i];
			}
		}

	private:
		/*! \brief Number of data points */
		data_size_t num_data_;

		/*! \brief Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points */
		std::map<gp_id_t, std::vector<int>> data_indices_per_cluster_;
		/*! \brief Keys: Labels of independent realizations of REs/GPs, values: number of data points per independent realization */
		std::map<gp_id_t, int> num_data_per_cluster_;
		/*! \brief Number of independent realizations of the REs/GPs */
		data_size_t num_clusters_;
		/*! \brief Unique labels of independent realizations */
		std::vector<gp_id_t> unique_clusters_;

		/*! \brief Number of grouped (intercept) random effects */
		data_size_t num_re_group_ = 0;
		/*! \brief Number of grouped random coefficients */
		data_size_t num_re_group_rand_coef_ = 0;
		/*! \brief Indices that relate every random coefficients to a "base" intercept grouped random effect. Counting starts at 1 (and ends at the number of base intercept random effects). Length of vector = num_re_group_rand_coef_. */
		std::vector<int> ind_effect_group_rand_coef_;
		/*! \brief Total number of grouped random effects (random intercepts plus random coefficients (slopes)) */
		data_size_t num_re_group_total_ = 0;

		/*! \brief 1 if there is a Gaussian process 0 otherwise */
		data_size_t num_gp_ = 0;
		/*! \brief Type of GP. 0 = classical (spatial) GP, 1 = spatio-temporal GP */ //TODO: remove?
		int8_t GP_type_ = 0;
		/*! \brief Number of random coefficient GPs */
		data_size_t num_gp_rand_coef_ = 0;
		/*! \brief Total number of GPs (random intercepts plus random coefficients) */
		data_size_t num_gp_total_ = 0;
		/*! \brief Index in the vector of random effect components (in the values of 're_comps_') of the intercept GP associated with the random coefficient GPs */
		int ind_intercept_gp_;
		/*! \brief Dimension of the coordinates (=number of features) for Gaussian process */
		int dim_gp_coords_ = 2;//required to save since it is needed in the Predict() function when predictions are made for new independent realizations of GPs
		/*! \brief Type of covariance(kernel) function for Gaussian processes */
		string_t cov_fct_ = "exponential";//required to also save here since it is needed in the Predict() function when predictions are made for new independent realizations of GPs
	/*! \brief Shape parameter of covariance function (=smoothness parameter for Matern covariance) */
		double cov_fct_shape_ = 0.;

		/*! \brief Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components */
		std::map<gp_id_t, std::vector<std::shared_ptr<RECompBase<T1>>>> re_comps_;
		/*! \brief Indices of parameters of RE components in global parameter vector cov_pars. ind_par_[i] + 1 and ind_par_[i+1] are the indices of the first and last parameter of component number i */
		std::vector<data_size_t> ind_par_;
		/*! \brief Number of covariance parameters */
		data_size_t num_cov_par_;
		/*! \brief Total number of random effect components (grouped REs plus other GPs) */
		data_size_t num_comps_total_ = 0;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Symbolic Cholesky decomposition of Psi matrices */
		std::map<gp_id_t, T2> chol_facts_solve_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Cholesky factors of Psi matrices */ //TODO: above needed or can pattern be saved somewhere else?
		std::map<gp_id_t, T1> chol_facts_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: **** */ //TODO: remove?
		std::map<gp_id_t, T1> Id_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Idendity matrices used for calculation of inverse covariance matrix **** */
		std::map<gp_id_t, cs> Id_cs_;
		/*! \brief Key: labels of independent realizations of REs/GPs, value: data y */
		std::map<gp_id_t, vec_t> y_;
		/*! \brief Key: labels of independent realizations of REs/GPs, value: Psi^-1*y_ (used for various computations) */
		std::map<gp_id_t, vec_t> y_aux_;
		/*! \brief Indicates whether y_aux_ has been calculated */
		bool y_aux_has_been_calculated_ = false;
		/*! \brief Collects inverse covariance matrices Psi^{-1} (usually not saved, but used e.g. in Fisher scoring without the Vecchia approximation) */
		std::map<gp_id_t, T1> psi_inv_;
		/*! \brief Copy of response data (used only in case there are also linear covariates since then y_ is modified during the algorithm) */
		vec_t y_vec_;
		/*! \brief Key: labels of independent realizations of REs/GPs, value: Psi^-1*y_ (used for various computations) */
		bool do_symbolic_decomposition_ = true;

		/*! \brief If true, the model linearly incluses covariates */
		bool has_covariates_ = false;
		/*! \brief Number of covariates */
		int num_coef_;
		/*! \brief Covariate data */
		den_mat_t X_;

		/*! \brief List of supported optimizers for covariance parameters */
		const std::set<string_t> SUPPORTED_OPTIM_COV_PAR_{ "gradient_descent", "fisher_scoring" };
		/*! \brief List of supported optimizers for regression coefficients */
		const std::set<string_t> SUPPORTED_OPTIM_COEF_{ "gradient_descent", "wls" };

		/*! \brief If true, the Veccia approximation is used for the Gaussian process */
		bool vecchia_approx_ = false;
		/*! \brief The number of neighbors used in the Vecchia approximation */
		int num_neighbors_;
		/*! \brief Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering */
		string_t vecchia_ordering_ = "none";
		/*! \brief The number of neighbors used in the Vecchia approximation for making predictions */
		int num_neighbors_pred_;
		/*! \brief Ordering used in the Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions */
		string_t vecchia_pred_type_ = "order_obs_first_cond_obs_only";//This is saved here and not simply set in the prediction function since it needs to be used repeatedly in the GPBoost algorithm when making predictions in "regression_metric.hpp" and the way predictions are done for the Vecchia approximation should be decoupled from the boosting algorithm
		/*! \brief List of supported covariance functions */
		const std::set<string_t> SUPPORTED_VECCHIA_PRED_TYPES_{ "order_obs_first_cond_obs_only",
		  "order_obs_first_cond_all", "order_pred_first",
		  "latent_order_obs_first_cond_obs_only", "latent_order_obs_first_cond_all" };
		/*! \brief Collects indices of nearest neighbors (used for Vecchia approximation) */
		std::map<gp_id_t, std::vector<std::vector<int>>> nearest_neighbors_;
		/*! \brief Distances between locations and their nearest neighbors (this is used only if the Vecchia approximation is used, otherwise the distances are saved directly in the base GP component) */
		std::map<gp_id_t, std::vector<den_mat_t>> dist_obs_neighbors_;
		/*! \brief Distances between nearest neighbors for all locations (this is used only if the Vecchia approximation is used, otherwise the distances are saved directly in the base GP component) */
		std::map<gp_id_t, std::vector<den_mat_t>> dist_between_neighbors_;//TODO: this contains duplicate information (i.e. distances might be saved reduntly several times). But there is a trade-off between storage and computational speed. I currently don't see a way for saving unique distances without copying them when using the^m.
		/*! \brief Outer product of covariate vector at observations and neighbors with itself. First index = cluster, second index = data point i, third index = GP number j (this is used only if the Vecchia approximation is used, this is handled saved directly in the GP component using Z_) */
		std::map<gp_id_t, std::vector<std::vector<den_mat_t>>> z_outer_z_obs_neighbors_;
		/*! \brief Collects matrices B = I - A (=Cholesky factor of inverse covariance) for Vecchia approximation */
		std::map<gp_id_t, sp_mat_t> B_;
		/*! \brief Collects diagonal matrices D^-1 for Vecchia approximation */
		std::map<gp_id_t, sp_mat_t> D_inv_;
		/*! \brief Collects derivatives of matrices B ( = derivative of matrix -A) for Vecchia approximation */
		std::map<gp_id_t, std::vector<sp_mat_t>> B_grad_;
		/*! \brief Collects derivatives of matrices D for Vecchia approximation */
		std::map<gp_id_t, std::vector<sp_mat_t>> D_grad_;
		/*! \brief Triplets for intializing the matrices B */
		std::map<gp_id_t, std::vector<Triplet_t>> entries_init_B_;
		/*! \brief Triplets for intializing the matrices B_grad */
		std::map<gp_id_t, std::vector<Triplet_t>> entries_init_B_grad_;

		/*! \brief Variance of idiosyncratic error term (nugget effect) */
		double sigma2_;

		/*! \brief Cluster IDs for prediction */
		std::vector<gp_id_t> cluster_ids_data_pred_;
		/*! \brief Levels of grouped RE for prediction */
		std::vector<std::vector<string_t>> re_group_levels_pred_;
		/*! \brief Covariate data for grouped random RE for prediction */
		std::vector<double> re_group_rand_coef_data_pred_;
		/*! \brief Coordinates for GP for prediction */
		std::vector<double> gp_coords_data_pred_;
		/*! \brief Covariate data for random GP for prediction */
		std::vector<double> gp_rand_coef_data_pred_;
		/*! \brief Covariate data  for linear regression term */
		std::vector<double> covariate_data_pred_;

		/*! \brief Nesterov schedule */
		double NesterovSchedule(int iter, int momentum_schedule_version = 0,
			double nesterov_acc_rate = 0.5, int momentum_offset = 2) {
			if (iter < momentum_offset) {
				return(0.);
			}
			else {
				if (momentum_schedule_version == 0) {
					return(nesterov_acc_rate);
				}
				else if (momentum_schedule_version == 1) {
					return(1. - (3. / (6. + iter)));
				}
				else {
					return(0.);
				}
			}
		}

		/*! \brief mutex for threading safe call */
		std::mutex mutex_;

		/*! \brief Constructs identity matrices if sparse matrices are used (used for calculating inverse covariance matrix) */
		template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr  >
		void ConstructI(gp_id_t cluster_i) {
			T3 I(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);//identity matrix for calculating precision matrix
			I.setIdentity();
			Id_.insert({ cluster_i, I });
			cs Id_cs = cs();//same for cs type //TODO: construct this independently of Id_ , but then care need to be taken for deleting the pointer objects.
			Id_cs.nzmax = num_data_per_cluster_[cluster_i];
			Id_cs.m = num_data_per_cluster_[cluster_i];
			Id_cs.n = num_data_per_cluster_[cluster_i];
			Id_[cluster_i].makeCompressed();
			Id_cs.p = reinterpret_cast<csi*>(Id_[cluster_i].outerIndexPtr());
			Id_cs.i = reinterpret_cast<csi*>(Id_[cluster_i].innerIndexPtr());
			Id_cs.x = Id_[cluster_i].valuePtr();
			Id_cs.nz = -1;
			Id_cs_.insert({ cluster_i, Id_cs });
		}

		/*! \brief Constructs identity matrices if dense matrices are used (used for calculating inverse covariance matrix) */
		template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr  >
		void ConstructI(gp_id_t cluster_i) {
			T3 I(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);//identity matrix for calculating precision matrix
			I.setIdentity();
			Id_.insert({ cluster_i, I });
		}

		/*!
		* \brief Set response variable data (y_)
		* \param y_data Response variable data
		*/
		void SetY(const double* y_data) {
			if (num_clusters_ == 1 && vecchia_ordering_ == "none") {
				y_[unique_clusters_[0]] = Eigen::Map<const vec_t>(y_data, num_data_);
				//y_[unique_clusters_[0]] = vec_t(num_data_);
				//y_[unique_clusters_[0]].setZero();
			}
			else {
				for (const auto& cluster_i : unique_clusters_) {
					y_[cluster_i] = vec_t(num_data_per_cluster_[cluster_i]);//TODO: Is there a more efficient way that avoids copying?
					for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
						y_[cluster_i][j] = y_data[data_indices_per_cluster_[cluster_i][j]];
					}
				}
			}
		}

		/*!
		* \brief Get y_aux = Psi^-1*y
		* \param[out] y_aux Psi^-1*y (=y_aux_). Array needs to be pre-allocated of length num_data_
		*/
		void GetYAux(double* y_aux) {
			CHECK(y_aux_has_been_calculated_);
			if (num_clusters_ == 1 && vecchia_ordering_ == "none") {
				for (int j = 0; j < num_data_; ++j) {
					y_aux[j] = y_aux_[unique_clusters_[0]][j];
				}
			}
			else {
				for (const auto& cluster_i : unique_clusters_) {
					for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
						y_aux[data_indices_per_cluster_[cluster_i][j]] = y_aux_[cluster_i][j];
					}
				}
			}
		}

		/*!
		* \brief Get y_aux = Psi^-1*y
		* \param[out] y_aux Psi^-1*y (=y_aux_). This vector needs to be pre-allocated of length num_data_
		*/
		void GetYAux(vec_t& y_aux) {
			CHECK(y_aux_has_been_calculated_);
			if (num_clusters_ == 1 && vecchia_ordering_ == "none") {
				y_aux = y_aux_[unique_clusters_[0]];
			}
			else {
				for (const auto& cluster_i : unique_clusters_) {
					y_aux(data_indices_per_cluster_[cluster_i]) = y_aux_[cluster_i];
				}
			}
		}

		/*! \brief Do Cholesky decomposition if sparse matrices are used */
		template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr  >
		void CalcChol(T3& psi, gp_id_t cluster_i, bool analyze_pattern) {
			if (analyze_pattern) {
				chol_facts_solve_[cluster_i].analyzePattern(psi);
			}
			chol_facts_solve_[cluster_i].factorize(psi);
			chol_facts_[cluster_i] = chol_facts_solve_[cluster_i].matrixL();
			chol_facts_[cluster_i].makeCompressed();
		}

		/*! \brief Do Cholesky decomposition if dense matrices are used */
		template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr  >
		void CalcChol(T3& psi, gp_id_t cluster_i, bool analyze_pattern) {
			if (analyze_pattern) {
				Log::Warning("Pattern of Cholesky factor is not analyzed when dense matrices are used.");
			}
			chol_facts_solve_[cluster_i].compute(psi);
			chol_facts_[cluster_i] = chol_facts_solve_[cluster_i].matrixL();
		}

		/*! \brief Caclulate Psi^(-1) if sparse matrices are used */
		template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr  >
		void CalcPsiInv(T3& psi_inv, gp_id_t cluster_i) {
			//Using CSparse function 'cs_spsolve'
			cs L_cs = cs();//Prepare LHS
			L_cs.nzmax = (int)chol_facts_[cluster_i].nonZeros();
			L_cs.m = num_data_per_cluster_[cluster_i];
			L_cs.n = num_data_per_cluster_[cluster_i];
			L_cs.p = reinterpret_cast<csi*>(chol_facts_[cluster_i].outerIndexPtr());
			L_cs.i = reinterpret_cast<csi*>(chol_facts_[cluster_i].innerIndexPtr());
			L_cs.x = chol_facts_[cluster_i].valuePtr();
			L_cs.nz = -1;

			sp_mat_t L_inv;
			sp_Lower_sp_RHS_cs_solve(&L_cs, &Id_cs_[cluster_i], L_inv, true);
			psi_inv = L_inv.transpose() * L_inv;

			////Version 2: doing sparse solving "by hand" but ignoring sparse RHS
			//const double* val = chol_facts_[cluster_i].valuePtr();
			//const int* row_idx = chol_facts_[cluster_i].innerIndexPtr();
			//const int* col_ptr = chol_facts_[cluster_i].outerIndexPtr();
			//den_mat_t L_inv_dens = den_mat_t(Id_[cluster_i]);
			//for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
			//	sp_L_solve(val, row_idx, col_ptr, num_data_per_cluster_[cluster_i], L_inv_dens.data() + j * num_data_per_cluster_[cluster_i]);
			//}
			//const sp_mat_t L_inv = L_inv_dens.sparseView();
			//psi_inv = L_inv.transpose() * L_inv;

			////Version 1: let Eigen do the solving
			//cpsi_inv = chol_facts_solve_[cluster_i].solve(Id_[cluster_i]);
		}

		/*! \brief Caclulate Psi^(-1) if dense matrices are used */
		template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr  >
		void CalcPsiInv(T3& psi_inv, gp_id_t cluster_i) {
			////Version 1
			//psi_inv = chol_facts_solve_[cluster_i].solve(Id_[cluster_i]);

			//Version 2: solving by hand
			T3 L_inv = Id_[cluster_i];
#pragma omp parallel for schedule(static)//TODO: maybe sometimes faster without parallelization?
			for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
				L_solve(chol_facts_[cluster_i].data(), num_data_per_cluster_[cluster_i], L_inv.data() + j * num_data_per_cluster_[cluster_i]);
			}
			//chol_facts_[cluster_i].triangularView<Eigen::Lower>().solveInPlace(L_inv); //slower
			psi_inv = L_inv.transpose() * L_inv;

			// Using dpotri from LAPACK does not work since LAPACK is not installed
			//int info = 0;
			//int n = num_data_per_cluster_[cluster_i];
			//int lda = num_data_per_cluster_[cluster_i];
			//char* uplo = "L";
			//den_mat_t M = chol_facts_[cluster_i];
			//BLASFUNC(dpotri)(uplo, &n, M.data(), &lda, &info);
		}

		/*! \brief Caclulate Psi^(-0.5)H if dense matrices are used. Used in 'NewtonUpdateLeafValues' */
		template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr  >
		void CalcPsiInvSqrtH(T3& PsiInvSqrtH, sp_mat_t& H, gp_id_t cluster_i) {
			PsiInvSqrtH = den_mat_t(H);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < H.cols(); ++j) {
				L_solve(chol_facts_[cluster_i].data(), num_data_per_cluster_[cluster_i], PsiInvSqrtH.data() + j * num_data_per_cluster_[cluster_i]);
			}
		}

		/*! \brief Caclulate Psi^(-0.5)H if sparse matrices are used. Used in 'NewtonUpdateLeafValues' */
		template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr  >
		void CalcPsiInvSqrtH(T3& PsiInvSqrtH, sp_mat_t& H, gp_id_t cluster_i) {
			//Using CSparse function 'cs_spsolve'
			eigen_sp_Lower_sp_RHS_cs_solve(chol_facts_[cluster_i], H, PsiInvSqrtH, true);
		}

		///*!
		//* \brief Caclulate X^TPsi^(-1)X
		//* \param X Covariate data matrix X
		//* \param[out] XT_psi_inv_X X^TPsi^(-1)X
		//*/
		//  template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr  >
		//  void CalcXTPsiInvX(const den_mat_t& X, den_mat_t& XT_psi_inv_X) {
		//    den_mat_t BX;
		//    if (num_clusters_ == 1) {
		//      gp_id_t cluster0 = unique_clusters_[0];
		//      if (vecchia_approx_) {
		//        BX = B_[cluster0] * X;
		//        XT_psi_inv_X = BX.transpose() * D_inv_[cluster0] * BX;
		//      }
		//      else {
		//        BX = X;
		//        #pragma omp parallel for schedule(static)
		//        for (int j = 0; j < num_data_per_cluster_[cluster0]; ++j) {
		//          L_solve(chol_facts_[cluster0].data(), num_data_per_cluster_[cluster0], BX.data() + j * num_data_per_cluster_[cluster0]);
		//        }
		//        XT_psi_inv_X = BX.transpose() * BX;
		//      }
		//    }
		//    else {
		//      XT_psi_inv_X = den_mat_t(X.cols(), X.cols());
		//      XT_psi_inv_X.setZero();
		//      for (const auto& cluster_i : unique_clusters_) {
		//        if (vecchia_approx_) {
		//          BX = B_[cluster_i] * X(data_indices_per_cluster_[cluster_i], Eigen::all);
		//          XT_psi_inv_X += BX.transpose() * D_inv_[cluster_i] * BX;
		//        }
		//        else {
		//          BX = X(data_indices_per_cluster_[cluster_i], Eigen::all);
		//          #pragma omp parallel for schedule(static)
		//          for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
		//            L_solve(chol_facts_[cluster_i].data(), num_data_per_cluster_[cluster_i], BX.data() + j * num_data_per_cluster_[cluster_i]);
		//          }
		//          XT_psi_inv_X += (BX.transpose() * BX);
		//        }
		//      }
		//    }
		//  }
		//  //same for sparse matrices
		//  template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr  >
		//  void CalcXTPsiInvX(const den_mat_t& X, den_mat_t& XT_psi_inv_X) {
		//    den_mat_t BX;
		//    if (num_clusters_ == 1) {
		//      gp_id_t cluster0 = unique_clusters_[0];
		//      if (vecchia_approx_) {
		//        BX = B_[cluster0] * X;
		//        XT_psi_inv_X = BX.transpose() * D_inv_[cluster0] * BX;
		//      }
		//      else {
		//        BX = X;
		//        #pragma omp parallel for schedule(static)
		//        for (int j = 0; j < num_data_per_cluster_[cluster0]; ++j) {
		//          sp_L_solve(chol_facts_[cluster0].valuePtr(), chol_facts_[cluster0].innerIndexPtr(), chol_facts_[cluster0].outerIndexPtr(),
		//            num_data_per_cluster_[cluster0], BX.data() + j * num_data_per_cluster_[cluster0]);
		//        }
		//        XT_psi_inv_X = BX.transpose() * BX;
		//      }
		//    }
		//    else {
		//      XT_psi_inv_X = den_mat_t(X.cols(), X.cols());
		//      XT_psi_inv_X.setZero();
		//      for (const auto& cluster_i : unique_clusters_) {
		//        if (vecchia_approx_) {
		//          BX = B_[cluster_i] * X(data_indices_per_cluster_[cluster_i], Eigen::all);
		//          XT_psi_inv_X += BX.transpose() * D_inv_[cluster_i] * BX;
		//        }
		//        else {
		//          BX = X(data_indices_per_cluster_[cluster_i], Eigen::all);
		//          #pragma omp parallel for schedule(static)
		//          for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
		//            sp_L_solve(chol_facts_[cluster_i].valuePtr(), chol_facts_[cluster_i].innerIndexPtr(), chol_facts_[cluster_i].outerIndexPtr(),
		//              num_data_per_cluster_[cluster_i], BX.data() + j * num_data_per_cluster_[cluster_i]);
		//          }
		//          XT_psi_inv_X += (BX.transpose() * BX);
		//        }
		//      }
		//    }
		//  }

		/*!
		* \brief Caclulate X^TPsi^(-1)X
		* \param X Covariate data matrix X
		* \param[out] XT_psi_inv_X X^TPsi^(-1)X
		*/
		void CalcXTPsiInvX(const den_mat_t& X, den_mat_t& XT_psi_inv_X) {
			if (num_clusters_ == 1 && vecchia_ordering_ == "none") {
				if (vecchia_approx_) {
					den_mat_t BX = B_[unique_clusters_[0]] * X;
					XT_psi_inv_X = BX.transpose() * D_inv_[unique_clusters_[0]] * BX;
				}
				else {
					XT_psi_inv_X = X.transpose() * chol_facts_solve_[unique_clusters_[0]].solve(X);
				}
			}
			else {
				XT_psi_inv_X = den_mat_t(X.cols(), X.cols());
				XT_psi_inv_X.setZero();
				den_mat_t BX;
				for (const auto& cluster_i : unique_clusters_) {
					if (vecchia_approx_) {
						BX = B_[cluster_i] * X(data_indices_per_cluster_[cluster_i], Eigen::all);
						XT_psi_inv_X += BX.transpose() * D_inv_[cluster_i] * BX;
					}
					else {
						XT_psi_inv_X += ((den_mat_t)X(data_indices_per_cluster_[cluster_i], Eigen::all)).transpose() * chol_facts_solve_[cluster_i].solve((den_mat_t)X(data_indices_per_cluster_[cluster_i], Eigen::all));
					}
				}
			}
		}

		/*!
		* \brief Initialize data structures for handling independent realizations of the Gaussian processes. Answers written on arguments.
		* \param num_data Number of data points
		* \param cluster_ids_data IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization)
		* \param[out] num_data_per_cluster Keys: labels of independent clusters, values: number of data points per independent realization
		* \param[out] data_indices_per_cluster Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
		* \param[out] unique_clusters Unique labels of independent realizations
		* \param[out] num_clusters Number of independent clusters
		*/
		void SetUpGPIds(data_size_t num_data, const gp_id_t* cluster_ids_data,
			std::map<gp_id_t, int>& num_data_per_cluster, std::map<gp_id_t, std::vector<int>>& data_indices_per_cluster,
			std::vector<gp_id_t>& unique_clusters, data_size_t& num_clusters) {
			if (cluster_ids_data != nullptr) {
				for (int i = 0; i < num_data; ++i) {
					if (num_data_per_cluster.find(cluster_ids_data[i]) == num_data_per_cluster.end()) {//first occurrence of cluster_ids_data[i]
						unique_clusters.push_back(cluster_ids_data[i]);
						num_data_per_cluster.insert({ cluster_ids_data[i], 1 });
						std::vector<int> id;
						id.push_back(i);
						data_indices_per_cluster.insert({ cluster_ids_data[i], id });
					}
					else {
						num_data_per_cluster[cluster_ids_data[i]] += 1;
						data_indices_per_cluster[cluster_ids_data[i]].push_back(i);
					}
				}
				num_clusters = (data_size_t)unique_clusters.size();
			}
			else {
				unique_clusters.push_back(0);
				num_data_per_cluster.insert({ 0, num_data });
				num_clusters = 1;
				std::vector<int> gp_id_vec(num_data);
				for (int i = 0; i < num_data; ++i) {
					gp_id_vec[i] = i;
				}
				data_indices_per_cluster.insert({ 0, gp_id_vec });
			}
		}

		/*!
		* \brief Convert characters in 'const char* re_group_data' to matrix (num_re_group x num_data) with strings of group labels
		* \param num_data Number of data points
		* \param num_re_group Number of grouped random effects
		* \param re_group_data Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
		* \param[out] Matrix of dimension num_re_group x num_data with strings of group labels for levels of grouped random effects
		*/
		void ConvertCharToStringGroupLevels(data_size_t num_data, data_size_t num_re_group,
			const char* re_group_data, std::vector<std::vector<string_t>>& re_group_levels) {
			int char_start = 0;
			for (int ire = 0; ire < num_re_group; ++ire) {//TODO: catch / report potential error if format of re_group_data is not correct
				for (int id = 0; id < num_data; ++id) {
					int number_chars = 0;
					while (re_group_data[char_start + number_chars] != '\0') {
						number_chars++;
					}
					re_group_levels[ire][id] = std::string(re_group_data + char_start);
					char_start += number_chars + 1;
				}
			}
		}

		/*!
		* \brief Initialize individual component models and collect them in a containter
		* \param num_data Number of data points
		* \param num_re_group Number of grouped random effects
		* \param data_indices_per_cluster Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points
		* \param cluster_i Index / label of the realization of the Gaussian process for which the components should be constructed
		* \param Group levels for every grouped random effect
		* \param num_data_per_cluster Keys: Labels of independent realizations of REs/GPs, values: number of data points per independent realization
		* \param num_re_group_rand_coef Number of grouped random coefficients
		* \param re_group_rand_coef_data Covariate data for grouped random coefficients
		* \param ind_effect_group_rand_coef Indices that relate every random coefficients to a "base" intercept grouped random effect. Counting start at 1.
		* \param num_gp Number of Gaussian processes (intercept only, random coefficients not counting)
		* \param gp_coords_data Coordinates (features) for Gaussian process
		* \param dim_gp_coords Dimension of the coordinates (=number of features) for Gaussian process
		* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
		* \param num_gp_rand_coef Number of Gaussian process random coefficients
		* \param cov_fct Type of covariance (kernel) function for Gaussian processes
	* \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern covariance)
		* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps_') of the intercept GP associated with the random coefficient GPs
		* \param[out] re_comps_cluster_i Container that collects the individual component models
		*/
		void CreateREComponents(data_size_t num_data, data_size_t num_re_group, std::map<gp_id_t, std::vector<int>>& data_indices_per_cluster, gp_id_t cluster_i,
			std::vector<std::vector<string_t>>& re_group_levels, std::map<gp_id_t, int>& num_data_per_cluster, data_size_t num_re_group_rand_coef,
			const double* re_group_rand_coef_data, std::vector<int>& ind_effect_group_rand_coef, data_size_t num_gp, const double* gp_coords_data, int dim_gp_coords,
			const double* gp_rand_coef_data, data_size_t num_gp_rand_coef, const string_t cov_fct, double cov_fct_shape, int ind_intercept_gp,
			std::vector<std::shared_ptr<RECompBase<T1>>>& re_comps_cluster_i) {
			//Grouped REs
			if (num_re_group > 0) {
				for (int j = 0; j < num_re_group; ++j) {
					std::vector<re_group_t> group_data;
					for (const auto& id : data_indices_per_cluster[cluster_i]) {
						group_data.push_back(re_group_levels[j][id]);//group_data_.push_back(std::string(re_group_data[j * num_data_ + id]));
					}
					re_comps_cluster_i.push_back(std::shared_ptr<RECompGroup<T1>>(new RECompGroup<T1>(group_data)));
				}
				//Random slopes
				if (num_re_group_rand_coef > 0) {
					for (int j = 0; j < num_re_group_rand_coef; ++j) {
						std::vector<double> rand_coef_data;
						for (const auto& id : data_indices_per_cluster[cluster_i]) {
							rand_coef_data.push_back(re_group_rand_coef_data[j * num_data + id]);
						}
						std::shared_ptr<RECompGroup<T1>> re_comp = std::dynamic_pointer_cast<RECompGroup<T1>>(re_comps_cluster_i[ind_effect_group_rand_coef[j] - 1]);//Subtract -1 since ind_effect_group_rand_coef[j] starts counting at 1 not 0
						re_comps_cluster_i.push_back(std::shared_ptr<RECompGroup<T1>>(new RECompGroup<T1>(re_comp->group_data_, re_comp->map_group_label_index_, re_comp->num_group_, rand_coef_data)));
					}
				}
			}
			//GPs
			if (num_gp > 0) {
				std::vector<double> gp_coords;
				for (int j = 0; j < dim_gp_coords; ++j) {
					for (const auto& id : data_indices_per_cluster[cluster_i]) {
						gp_coords.push_back(gp_coords_data[j * num_data + id]);
					}
				}
				den_mat_t gp_coords_mat = Eigen::Map<den_mat_t>(gp_coords.data(), num_data_per_cluster[cluster_i], dim_gp_coords);
				re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T1>>(new RECompGP<T1>(gp_coords_mat, cov_fct, cov_fct_shape, true)));

				//Random slopes
				if (num_gp_rand_coef > 0) {
					for (int j = 0; j < num_gp_rand_coef; ++j) {
						std::vector<double> rand_coef_data;
						for (const auto& id : data_indices_per_cluster[cluster_i]) {
							rand_coef_data.push_back(gp_rand_coef_data[j * num_data + id]);
						}
						std::shared_ptr<RECompGP<T1>> re_comp = std::dynamic_pointer_cast<RECompGP<T1>>(re_comps_cluster_i[ind_intercept_gp]);
						re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T1>>(new RECompGP<T1>(re_comp->dist_, re_comp->has_Z_,
							&re_comp->Z_, rand_coef_data, cov_fct, cov_fct_shape)));
					}
				}
			}
		}

		/*!
		* \brief Initialize individual component models and collect them in a containter when the Vecchia approximation is used
		* \param num_data Number of data points
		* \param data_indices_per_cluster Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points
		* \param cluster_i Index / label of the realization of the Gaussian process for which the components should be constructed
		* \param num_data_per_cluster Keys: Labels of independent realizations of REs/GPs, values: number of data points per independent realization
		* \param gp_coords_data Coordinates (features) for Gaussian process
		* \param dim_gp_coords Dimension of the coordinates (=number of features) for Gaussian process
		* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
		* \param num_gp_rand_coef Number of Gaussian process random coefficients
		* \param cov_fct Type of covariance (kernel) function for Gaussian processes
		* \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern covariance)
		* \param[out] re_comps_cluster_i Container that collects the individual component models
		* \param[out] nearest_neighbors_cluster_i Collects indices of nearest neighbors
		* \param[out] dist_obs_neighbors_cluster_i Distances between locations and their nearest neighbors
		* \param[out] dist_between_neighbors_cluster_i Distances between nearest neighbors for all locations
		* \param[out] entries_init_B_cluster_i Triplets for intializing the matrices B
		* \param[out] entries_init_B_grad_cluster_i Triplets for intializing the matrices B_grad
		* \param[out] z_outer_z_obs_neighbors_cluster_i Outer product of covariate vector at observations and neighbors with itself for random coefficients. First index = data point i, second index = GP number j
		* \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
		* \param num_neighbors The number of neighbors used in the Vecchia approximation
		*/
		void CreateREComponentsVecchia(data_size_t num_data, std::map<gp_id_t, std::vector<int>>& data_indices_per_cluster, gp_id_t cluster_i, std::map<gp_id_t, int>& num_data_per_cluster,
			const double* gp_coords_data, int dim_gp_coords, const double* gp_rand_coef_data, data_size_t num_gp_rand_coef, const string_t cov_fct, double cov_fct_shape,
			std::vector<std::shared_ptr<RECompBase<T1>>>& re_comps_cluster_i, std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
			std::vector<den_mat_t>& dist_obs_neighbors_cluster_i, std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
			std::vector<Triplet_t >& entries_init_B_cluster_i, std::vector<Triplet_t >& entries_init_B_grad_cluster_i,
			std::vector<std::vector<den_mat_t>>& z_outer_z_obs_neighbors_cluster_i, string_t vecchia_ordering = "none", int num_neighbors = 30) {

			if (vecchia_ordering == "random") {
				unsigned seed = 0;
				std::shuffle(data_indices_per_cluster[cluster_i].begin(), data_indices_per_cluster[cluster_i].end(), std::default_random_engine(seed));
			}

			std::vector<double> gp_coords;
			for (int j = 0; j < dim_gp_coords; ++j) {
				for (const auto& id : data_indices_per_cluster[cluster_i]) {
					gp_coords.push_back(gp_coords_data[j * num_data + id]);
				}
			}
			den_mat_t gp_coords_mat = Eigen::Map<den_mat_t>(gp_coords.data(), num_data_per_cluster[cluster_i], dim_gp_coords);
			re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T1>>(new RECompGP<T1>(gp_coords_mat, cov_fct, cov_fct_shape, false)));
			find_nearest_neighbors_Veccia_fast(gp_coords_mat, num_data_per_cluster[cluster_i], num_neighbors,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1);

			for (int i = 0; i < num_data_per_cluster[cluster_i]; ++i) {
				for (int j = 0; j < (int)nearest_neighbors_cluster_i[i].size(); ++j) {
					entries_init_B_cluster_i.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.));
					entries_init_B_grad_cluster_i.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.));
				}
				entries_init_B_cluster_i.push_back(Triplet_t(i, i, 1.));//Put 1's on the diagonal since B = I - A
			}

			//Random coefficients
			if (num_gp_rand_coef > 0) {

				for (int j = 0; j < num_gp_rand_coef; ++j) {
					std::vector<double> rand_coef_data;
					for (const auto& id : data_indices_per_cluster[cluster_i]) {
						rand_coef_data.push_back(gp_rand_coef_data[j * num_data + id]);
					}
					re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T1>>(new RECompGP<T1>(rand_coef_data, cov_fct, cov_fct_shape)));

					//save random coefficient data in the form ot outer product matrices
#pragma omp for schedule(static)
					for (int i = 0; i < num_data_per_cluster[cluster_i]; ++i) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef);
						int dim_z = (i == 0) ? 1 : ((int)nearest_neighbors_cluster_i[i].size() + 1);
						vec_t coef_vec(dim_z);
						coef_vec(0) = rand_coef_data[i];
						if (i > 0) {
							for (int ii = 1; ii < dim_z; ++ii) {
								coef_vec(ii) = rand_coef_data[nearest_neighbors_cluster_i[i][ii - 1]];
							}
						}
						z_outer_z_obs_neighbors_cluster_i[i][j] = coef_vec * coef_vec.transpose();
					}
				}
			}
		}


		/*!
		* \brief Set the covariance parameters of the components
		* \param cov_pars Covariance parameters
		*/
		void SetCovParsComps(const vec_t& cov_pars) {
			CHECK(cov_pars.size() == num_cov_par_);
			sigma2_ = cov_pars[0];
			for (const auto& cluster_i : unique_clusters_) {
				for (int j = 0; j < num_comps_total_; ++j) {
					//const std::vector<double> pars = std::vector<double>(cov_pars.begin() + ind_par_[j] + 1, cov_pars.begin() + ind_par_[j + 1] + 1);
					const vec_t pars = cov_pars.segment(ind_par_[j] + 1, ind_par_[j + 1] - ind_par_[j]);
					re_comps_[cluster_i][j]->SetCovPars(pars);
				}
			}
		}

		/*!
		* \brief Transform the covariance parameters to the scake on which the MLE is found
		* \param cov_pars_trans Covariance parameters
		* \param[out] pars_trans Transformed covariance parameters
		*/
		void TransformCovPars(const vec_t& cov_pars, vec_t& cov_pars_trans) {
			CHECK(cov_pars.size() == num_cov_par_);
			cov_pars_trans = vec_t(num_cov_par_);
			cov_pars_trans[0] = cov_pars[0];
			for (int j = 0; j < num_comps_total_; ++j) {
				const vec_t pars = cov_pars.segment(ind_par_[j] + 1, ind_par_[j + 1] - ind_par_[j]);
				vec_t pars_trans = pars;
				re_comps_[unique_clusters_[0]][j]->TransformCovPars(cov_pars[0], pars, pars_trans);
				cov_pars_trans.segment(ind_par_[j] + 1, ind_par_[j + 1] - ind_par_[j]) = pars_trans;
			}
		}

		/*!
		* \brief Back-transform the covariance parameters to the original scale
		* \param cov_pars Covariance parameters
		* \param[out] cov_pars_orig Back-transformed, original covariance parameters
		*/
		void TransformBackCovPars(const vec_t& cov_pars, vec_t& cov_pars_orig) {
			CHECK(cov_pars.size() == num_cov_par_);
			cov_pars_orig = vec_t(num_cov_par_);
			cov_pars_orig[0] = cov_pars[0];
			for (int j = 0; j < num_comps_total_; ++j) {
				const vec_t pars = cov_pars.segment(ind_par_[j] + 1, ind_par_[j + 1] - ind_par_[j]);
				vec_t pars_orig = pars;
				re_comps_[unique_clusters_[0]][j]->TransformBackCovPars(cov_pars[0], pars, pars_orig);
				cov_pars_orig.segment(ind_par_[j] + 1, ind_par_[j + 1] - ind_par_[j]) = pars_orig;
			}
		}

		/*!
		* \brief Calculate covariance matrices of the components
		*/
		void CalcSigmaComps() {
			for (const auto& cluster_i : unique_clusters_) {
				for (int j = 0; j < num_comps_total_; ++j) {
					re_comps_[cluster_i][j]->CalcSigma();
				}
			}
		}

		/*!
		* \brief Calculate matrices A and D_inv as well as their derivatives for the Vecchia approximation for one cluster (independent realization of GP)
		* \param num_data_cluster_i Number of data points
		* \param calc_gradient If true, the gradient also be calculated (only for Vecchia approximation)
		* \param re_comps_cluster_i Container that collects the individual component models
		* \param nearest_neighbors_cluster_i Collects indices of nearest neighbors
		* \param dist_obs_neighbors_cluster_i Distances between locations and their nearest neighbors
		* \param dist_between_neighbors_cluster_i Distances between nearest neighbors for all locations
		* \param entries_init_B_cluster_i Triplets for intializing the matrices B
		* \param entries_init_B_grad_cluster_i Triplets for intializing the matrices B_grad
		* \param z_outer_z_obs_neighbors_cluster_i Outer product of covariate vector at observations and neighbors with itself for random coefficients. First index = data point i, second index = GP number j
		* \param[out] B_cluster_i Matrix A = I - B (= Cholesky factor of inverse covariance) for Vecchia approximation
		* \param[out] D_inv_cluster_i Diagonal matrices D^-1 for Vecchia approximation
		* \param[out] B_grad_cluster_i Derivatives of matrices A ( = derivative of matrix -B) for Vecchia approximation
		* \param[out] D_grad_cluster_i Derivatives of matrices D for Vecchia approximation
		* \param transf_scale If true, the derivatives are taken on the transformed scale otherwise on the original scale. Default = true
		* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back)
		* \param calc_gradient_nugget If true, derivatives are also taken with respect to the nugget / noise variance
		*/
		void CalcCovFactorVecchia(int num_data_cluster_i, bool calc_gradient,//TODO: make arguments const
			std::vector<std::shared_ptr<RECompBase<T1>>>& re_comps_cluster_i, std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
			std::vector<den_mat_t>& dist_obs_neighbors_cluster_i, std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
			std::vector<Triplet_t >& entries_init_B_cluster_i, std::vector<Triplet_t >& entries_init_B_grad_cluster_i,
			std::vector<std::vector<den_mat_t>>& z_outer_z_obs_neighbors_cluster_i,
			sp_mat_t& B_cluster_i, sp_mat_t& D_inv_cluster_i, std::vector<sp_mat_t>& B_grad_cluster_i, std::vector<sp_mat_t>& D_grad_cluster_i,
			bool transf_scale = true, double nugget_var = 1., bool calc_gradient_nugget = false) {

			int num_par_comp = re_comps_cluster_i[ind_intercept_gp_]->num_cov_par_;
			int num_par_gp = num_par_comp * num_gp_total_ + calc_gradient_nugget;

			//Initialize matrices B = I - A and D^-1 as well as their derivatives (in order that the code below can be run in parallel)
			B_cluster_i = sp_mat_t(num_data_cluster_i, num_data_cluster_i);//B = I - A
			B_cluster_i.setFromTriplets(entries_init_B_cluster_i.begin(), entries_init_B_cluster_i.end());//Note: 1's are put on the diagonal
			D_inv_cluster_i = sp_mat_t(num_data_cluster_i, num_data_cluster_i);//D^-1. Note: we first calculate D, and then take the inverse below
			D_inv_cluster_i.setIdentity();//Put 1's on the diagonal for nugget effect (entries are not overriden but added below)
			if (!transf_scale) {
				D_inv_cluster_i.diagonal().array() *= nugget_var;//nugget effect is not 1 if not on transformed scale
			}
			if (calc_gradient) {
				B_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of B = derviateive of (-A)
				D_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of D
				for (int ipar = 0; ipar < num_par_gp; ++ipar) {
					B_grad_cluster_i[ipar] = sp_mat_t(num_data_cluster_i, num_data_cluster_i);
					B_grad_cluster_i[ipar].setFromTriplets(entries_init_B_grad_cluster_i.begin(), entries_init_B_grad_cluster_i.end());
					D_grad_cluster_i[ipar] = sp_mat_t(num_data_cluster_i, num_data_cluster_i);
					D_grad_cluster_i[ipar].setIdentity();//Put 0 on the diagonal
					D_grad_cluster_i[ipar].diagonal().array() = 0.;//TODO: maybe change initialization of this matrix by also using triplets -> faster?
				}
			}//end initialization

#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data_cluster_i; ++i) {
				int num_nn = (int)nearest_neighbors_cluster_i[i].size();

				//calculate covariance matrices between observations and neighbors and among neighbors as well as their derivatives
				den_mat_t cov_mat_obs_neighbors(1, num_nn);
				den_mat_t cov_mat_between_neighbors(num_nn, num_nn);
				std::vector<den_mat_t> cov_grad_mats_obs_neighbors(num_par_gp);//covariance matrix plus derivative wrt to every parameter
				std::vector<den_mat_t> cov_grad_mats_between_neighbors(num_par_gp);

				if (i > 0) {

					for (int j = 0; j < num_gp_total_; ++j) {
						int ind_first_par = j * num_par_comp;//index of first parameter (variance) of component j in gradient vectors

						if (j == 0) {
							re_comps_cluster_i[ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_obs_neighbors_cluster_i[i],//re_comp->
								cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors[ind_first_par], cov_grad_mats_obs_neighbors[ind_first_par + 1],
								calc_gradient, transf_scale, nugget_var);//write on matrices directly for first GP component
							re_comps_cluster_i[ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_between_neighbors_cluster_i[i],
								cov_mat_between_neighbors, cov_grad_mats_between_neighbors[ind_first_par], cov_grad_mats_between_neighbors[ind_first_par + 1],
								calc_gradient, transf_scale, nugget_var);
						}
						else {//random coefficient GPs
							den_mat_t cov_mat_obs_neighbors_j;
							den_mat_t cov_mat_between_neighbors_j;
							re_comps_cluster_i[ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_obs_neighbors_cluster_i[i],
								cov_mat_obs_neighbors_j, cov_grad_mats_obs_neighbors[ind_first_par], cov_grad_mats_obs_neighbors[ind_first_par + 1],
								calc_gradient, transf_scale, nugget_var);
							re_comps_cluster_i[ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_between_neighbors_cluster_i[i],
								cov_mat_between_neighbors_j, cov_grad_mats_between_neighbors[ind_first_par], cov_grad_mats_between_neighbors[ind_first_par + 1],
								calc_gradient, transf_scale, nugget_var);
							//multiply by coefficient matrix
							cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(0, 1, 1, num_nn)).array();//cov_mat_obs_neighbors_j.cwiseProduct()
							cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
							cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
							cov_mat_between_neighbors += cov_mat_between_neighbors_j;

							if (calc_gradient) {
								cov_grad_mats_obs_neighbors[ind_first_par].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(0, 1, 1, num_nn)).array();
								cov_grad_mats_obs_neighbors[ind_first_par + 1].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(0, 1, 1, num_nn)).array();
								cov_grad_mats_between_neighbors[ind_first_par].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
								cov_grad_mats_between_neighbors[ind_first_par + 1].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
							}
						}
					}//end loop over components j
				}//end if(i>1)

				//Calculate matrices B and D as well as their derivatives

				//1. add first summand of matrix D (ZCZ^T_{ii}) and its derivatives
				for (int j = 0; j < num_gp_total_; ++j) {
					double d_comp_j = re_comps_cluster_i[ind_intercept_gp_ + j]->cov_pars_[0];
					if (!transf_scale) {
						d_comp_j *= nugget_var;
					}
					if (j > 0) {//random coefficient
						d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
					}
					D_inv_cluster_i.coeffRef(i, i) += d_comp_j;
					if (calc_gradient) {
						if (transf_scale) {
							D_grad_cluster_i[j * num_par_comp].coeffRef(i, i) = d_comp_j;//derivative of the covariance function wrt the variance. derivative of the covariance function wrt to range is zero on the diagonal
						}
						else {
							D_grad_cluster_i[j * num_par_comp].coeffRef(i, i) = 1.;//1's on the diagonal on the orignal scale
						}
					}
				}

				if (calc_gradient && calc_gradient_nugget) {
					D_grad_cluster_i[num_par_gp - 1].coeffRef(i, i) = 1.;
				}

				//2. remaining terms
				if (i > 0) {

					if (transf_scale) {
						cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
					}
					else {
						cov_mat_between_neighbors.diagonal().array() += nugget_var;
					}

					den_mat_t A_i(1, num_nn);
					den_mat_t cov_mat_between_neighbors_inv;
					den_mat_t A_i_grad_sigma2;
					if (calc_gradient) {
						den_mat_t I(num_nn, num_nn);
						I.setIdentity();
						cov_mat_between_neighbors_inv = cov_mat_between_neighbors.llt().solve(I);
						A_i = cov_mat_obs_neighbors * cov_mat_between_neighbors_inv;
						if (calc_gradient_nugget) {
							A_i_grad_sigma2 = -A_i * cov_mat_between_neighbors_inv;
						}
					}
					else {
						A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors.transpose())).transpose();
					}

					for (int inn = 0; inn < num_nn; ++inn) {
						B_cluster_i.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) = -A_i(0, inn);
					}
					D_inv_cluster_i.coeffRef(i, i) -= (A_i * cov_mat_obs_neighbors.transpose())(0, 0);

					if (calc_gradient) {
						den_mat_t A_i_grad(1, num_nn);
						for (int j = 0; j < num_gp_total_; ++j) {
							int ind_first_par = j * num_par_comp;
							for (int ipar = 0; ipar < num_par_comp; ++ipar) {
								A_i_grad = (cov_grad_mats_obs_neighbors[ind_first_par + ipar] * cov_mat_between_neighbors_inv) -
									(cov_mat_obs_neighbors * cov_mat_between_neighbors_inv *
										cov_grad_mats_between_neighbors[ind_first_par + ipar] * cov_mat_between_neighbors_inv);
								for (int inn = 0; inn < num_nn; ++inn) {
									B_grad_cluster_i[ind_first_par + ipar].coeffRef(i, nearest_neighbors_cluster_i[i][inn]) = -A_i_grad(0, inn);
								}
								if (ipar == 0) {
									D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) -= ((A_i_grad * cov_mat_obs_neighbors.transpose())(0, 0) +
										(A_i * cov_grad_mats_obs_neighbors[ind_first_par + ipar].transpose())(0, 0));//add to derivative of diagonal elements for marginal variance 
								}
								else {
									D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) = -((A_i_grad * cov_mat_obs_neighbors.transpose())(0, 0) +
										(A_i * cov_grad_mats_obs_neighbors[ind_first_par + ipar].transpose())(0, 0));//don't add to existing values since derivative of diagonal is zero for range
								}
							}
						}
						if (calc_gradient_nugget) {
							for (int inn = 0; inn < num_nn; ++inn) {
								B_grad_cluster_i[num_par_gp - 1].coeffRef(i, nearest_neighbors_cluster_i[i][inn]) = -A_i_grad_sigma2(0, inn);
							}
							D_grad_cluster_i[num_par_gp - 1].coeffRef(i, i) -= (A_i_grad_sigma2 * cov_mat_obs_neighbors.transpose())(0, 0);
						}
					}//end calc_gradient

				}//end if i > 0

				D_inv_cluster_i.coeffRef(i, i) = 1. / D_inv_cluster_i.coeffRef(i, i);

			}//end loop over data i

		}

		/*!
		* \brief Create the covariance matrix Psi and factorize it (either calculate a Cholesky factor or the inverse covariance matrix)
		* \param calc_gradient If true, the gradient also be calculated (only for Vecchia approximation)
		* \param transf_scale If true, the derivatives are taken on the transformed scale otherwise on the original scale. Default = true (only for Vecchia approximation)
		* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back, normally this is equal to one, since the variance paramter is modelled separately and factored out)
		* \param calc_gradient_nugget If true, derivatives are also taken with respect to the nugget / noise variance (only for Vecchia approximation)
		*/
		void CalcCovFactor(bool calc_gradient = false, bool transf_scale = true, double nugget_var = 1., bool calc_gradient_nugget = false) {
			if (vecchia_approx_) {
				for (const auto& cluster_i : unique_clusters_) {
					int num_data_cl_i = num_data_per_cluster_[cluster_i];
					CalcCovFactorVecchia(num_data_cl_i, calc_gradient, re_comps_[cluster_i], nearest_neighbors_[cluster_i],
						dist_obs_neighbors_[cluster_i], dist_between_neighbors_[cluster_i],
						entries_init_B_[cluster_i], entries_init_B_grad_[cluster_i], z_outer_z_obs_neighbors_[cluster_i],
						B_[cluster_i], D_inv_[cluster_i], B_grad_[cluster_i], D_grad_[cluster_i], transf_scale, nugget_var, calc_gradient_nugget);
				}
			}
			else {
				CalcSigmaComps();
				for (const auto& cluster_i : unique_clusters_) {
					T1 psi;
					psi.resize(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
					psi.setIdentity();
					for (int j = 0; j < num_comps_total_; ++j) {
						psi += (*(re_comps_[cluster_i][j]->GetZSigmaZt()));
					}
					CalcChol<T1>(psi, cluster_i, do_symbolic_decomposition_);
				}
				do_symbolic_decomposition_ = false;//Symbolic decompostion done only once (if sparse matrices are used)
			}
		}

		/*!
		* \brief Calculate Psi^-1*y (and save in y_aux_)
		* \param marg_variance The marginal variance. Default = 1.
		*/
		void CalcYAux(double marg_variance = 1.) {
			for (const auto& cluster_i : unique_clusters_) {
				if (y_.find(cluster_i) == y_.end()) {
					Log::Fatal("Response variable data (y_) for random effects model has not been set. Call 'SetY' first.");
				}
				if (vecchia_approx_) {
					if (B_.find(cluster_i) == B_.end()) {
						Log::Fatal("Factorisation of covariance matrix has not been done. Call 'CalcCovFactor' first.");
					}
					y_aux_[cluster_i] = B_[cluster_i].transpose() * D_inv_[cluster_i] * B_[cluster_i] * y_[cluster_i];
				}//end Vecchia
				else {
					if (chol_facts_.find(cluster_i) == chol_facts_.end()) {
						Log::Fatal("Factorisation of covariance matrix has not been done. Call 'CalcCovFactor' first.");
					}
					//Version 1: let Eigen do the computation
					y_aux_[cluster_i] = chol_facts_solve_[cluster_i].solve(y_[cluster_i]);
					//// Version 2 'do-it-yourself' (for sparse matrices)
					//y_aux_[cluster_i] = y_[cluster_i];
					//const double* val = chol_facts_[cluster_i].valuePtr();
					//const int* row_idx = chol_facts_[cluster_i].innerIndexPtr();
					//const int* col_ptr = chol_facts_[cluster_i].outerIndexPtr();
					//sp_L_solve(val, row_idx, col_ptr, num_data_per_cluster_[cluster_i], y_aux_[cluster_i].data());
					//sp_L_t_solve(val, row_idx, col_ptr, num_data_per_cluster_[cluster_i], y_aux_[cluster_i].data());

				}//end non-Vecchia
				if (marg_variance != 1.) {
					y_aux_[cluster_i] /= marg_variance;
				}
			}
			y_aux_has_been_calculated_ = true;
		}

		/*!
		* \brief Calculate y^T*Psi^-1*y if sparse matrices are used
		* \param[out] yTPsiInvy y^T*Psi^-1*y
		*/
		template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr  >
		void CalcYTPsiIInvY(double& yTPsiInvy) {
			yTPsiInvy = 0;
			for (const auto& cluster_i : unique_clusters_) {
				if (y_.find(cluster_i) == y_.end()) {
					Log::Fatal("Response variable data (y_) for random effects model has not been set. Call 'SetY' first.");
				}
				if (vecchia_approx_) {
					if (B_.find(cluster_i) == B_.end()) {
						Log::Fatal("Factorisation of covariance matrix has not been done. Call 'CalcCovFactor' first.");
					}
					vec_t y_aux_sqrt = B_[cluster_i] * y_[cluster_i];
					yTPsiInvy += (y_aux_sqrt.transpose() * D_inv_[cluster_i] * y_aux_sqrt)(0, 0);
				}//end Vecchia
				else {
					if (chol_facts_.find(cluster_i) == chol_facts_.end()) {
						Log::Fatal("Factorisation of covariance matrix has not been done. Call 'CalcCovFactor' first.");
					}
					vec_t y_aux_sqrt = y_[cluster_i];
					const double* val = chol_facts_[cluster_i].valuePtr();
					const int* row_idx = chol_facts_[cluster_i].innerIndexPtr();
					const int* col_ptr = chol_facts_[cluster_i].outerIndexPtr();
					sp_L_solve(val, row_idx, col_ptr, num_data_per_cluster_[cluster_i], y_aux_sqrt.data());
					yTPsiInvy += (y_aux_sqrt.transpose() * y_aux_sqrt)(0, 0);
				}//end non-Vecchia
			}
		}

		/*!
		* \brief Calculate y^T*Psi^-1*y if dense matrices are used
		* \param[out] yTPsiInvy y^T*Psi^-1*y
		*/
		template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr  >
		void CalcYTPsiIInvY(double& yTPsiInvy) {
			yTPsiInvy = 0;
			for (const auto& cluster_i : unique_clusters_) {
				if (y_.find(cluster_i) == y_.end()) {
					Log::Fatal("Response variable data (y_) for random effects model has not been set. Call 'SetY' first.");
				}
				if (vecchia_approx_) {
					if (B_.find(cluster_i) == B_.end()) {
						Log::Fatal("Factorisation of covariance matrix has not been done. Call 'CalcCovFactor' first.");
					}
					vec_t y_aux_sqrt = B_[cluster_i] * y_[cluster_i];
					yTPsiInvy += (y_aux_sqrt.transpose() * D_inv_[cluster_i] * y_aux_sqrt)(0, 0);
				}//end Vecchia
				else {
					if (chol_facts_.find(cluster_i) == chol_facts_.end()) {
						Log::Fatal("Factorisation of covariance matrix has not been done. Call 'CalcCovFactor' first.");
					}
					vec_t y_aux_sqrt = y_[cluster_i];
					L_solve(chol_facts_[cluster_i].data(), num_data_per_cluster_[cluster_i], y_aux_sqrt.data());
					yTPsiInvy += (y_aux_sqrt.transpose() * y_aux_sqrt)(0, 0);
				}//end non-Vecchia
			}
		}

		/*!
		* \brief Calculate gradient for covariance parameters
		* \param include_error_var If true, the gradient for the marginal variance parameter (=error, nugget effect) is also calculated, otherwise not (set this to true if the nugget effect is not calculated by using the closed-form solution)
		* \param save_psi_inv If true, the inverse covariance matrix Pis^-1 is saved for reuse later (e.g. when calculating the Fisher information in Fisher scoring). This option is ignored if the Vecchia approximation is used.
		* \return Gradient for covariance parameters
		*/
		vec_t GetCovParGrad(bool include_error_var = false, bool save_psi_inv = false) {
			vec_t cov_grad;
			if (include_error_var) {
				cov_grad = vec_t::Zero(num_cov_par_);
			}
			else {
				cov_grad = vec_t::Zero(num_cov_par_ - 1);
			}
			int first_cov_par = 0;
			if (include_error_var) {
				first_cov_par = 1;
			}
			for (const auto& cluster_i : unique_clusters_) {
				if (vecchia_approx_) {//Vechia approximation
					vec_t u(num_data_per_cluster_[cluster_i]);
					vec_t uk(num_data_per_cluster_[cluster_i]);
					if (include_error_var) {
						u = B_[cluster_i] * y_[cluster_i];
						cov_grad[0] += -1. * ((double)(u.transpose() * D_inv_[cluster_i] * u)) / sigma2_ / 2. + num_data_per_cluster_[cluster_i] / 2.;
						u = D_inv_[cluster_i] * u;
					}
					else {
						u = D_inv_[cluster_i] * B_[cluster_i] * y_[cluster_i];//TODO: this is already calculated in CalcYAux -> save it there and re-use here?
					}
					for (int j = 0; j < num_comps_total_; ++j) {
						int num_par_comp = re_comps_[cluster_i][j]->num_cov_par_;
						for (int ipar = 0; ipar < num_par_comp; ++ipar) {
							uk = B_grad_[cluster_i][num_par_comp * j + ipar] * y_[cluster_i];
							cov_grad[first_cov_par + ind_par_[j] + ipar] += ((uk.dot(u) - 0.5 * u.dot(D_grad_[cluster_i][num_par_comp * j + ipar] * u)) / sigma2_ +
								0.5 * (D_inv_[cluster_i].diagonal()).dot(D_grad_[cluster_i][num_par_comp * j + ipar].diagonal()));
						}
					}
				}//end Vecchia
				else {
					T1 psi_inv;
					CalcPsiInv(psi_inv, cluster_i);
					if (save_psi_inv) {
						psi_inv_[cluster_i] = psi_inv;
					}
					if (include_error_var) {
						cov_grad[0] += -1. * ((double)(y_[cluster_i].transpose() * y_aux_[cluster_i])) / sigma2_ / 2. + num_data_per_cluster_[cluster_i] / 2.;
					}
					for (int j = 0; j < num_comps_total_; ++j) {
						for (int ipar = 0; ipar < re_comps_[cluster_i][j]->num_cov_par_; ++ipar) {
							std::shared_ptr<T1> gradPsi = re_comps_[cluster_i][j]->GetZSigmaZtGrad(ipar, true, 1.);
							cov_grad[first_cov_par + ind_par_[j] + ipar] += -1. * ((double)(y_aux_[cluster_i].transpose() * (*gradPsi) * y_aux_[cluster_i])) / sigma2_ / 2. +
								((double)(((*gradPsi).cwiseProduct(psi_inv)).sum())) / 2.;
						}
					}
				}//end standard (non-Vecchia) calculation
			}// end loop over clusters
			return(cov_grad);
		}

		/*!
		* \brief Apply a momentum step
		* \param it Iteration number
		* \param[out] pars Parameters
		* \param[out] pars_lag1 Parameters from last iteration
		* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters. Default = true
		* \param nesterov_acc_rate Acceleration rate for Nesterov acceleration
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
		* \param exclude_first_log_scale If true, no momentum is applied to the first value and the momentum step is done on the log-scale for the other values. Default = true
		* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
		*/
		void ApplyMomentumStep(int it, vec_t& pars, vec_t& pars_lag1, bool use_nesterov_acc = true,
			double nesterov_acc_rate = 0.5, int nesterov_schedule_version = 0, bool exclude_first_log_scale = true,
			int momentum_offset = 2) {
			if (use_nesterov_acc) {
				double mu = NesterovSchedule(it, nesterov_schedule_version, nesterov_acc_rate, momentum_offset);
				int num_par = (int)pars.size();
				vec_t pars_mom(num_par);//Covariance parameters plus a momentum step
				if (exclude_first_log_scale) {
					pars_mom.segment(1, num_par - 1) = ((mu + 1.) * (pars.segment(1, num_par - 1).array().log()) - mu * (pars_lag1.segment(1, num_par - 1).array().log())).exp().matrix();//Momentum is added on the log scale
					pars_mom[0] = pars[0];
				}
				else {
					pars_mom = (mu + 1) * pars - mu * pars_lag1;
				}
				pars_lag1 = pars;
				pars = pars_mom;
			}
			else {
				pars_lag1 = pars;
			}
		}

		/*!
		* \brief Update covariance parameters doing one gradient descent step (except for the marginal variance which is updated using an explicit solution)
		* \param lr Learning rate
		* \param[out] cov_pars Covariance parameters
		* \param closed_form_solution_sigma If true, the error variance (nugget effect) is calculated exactly using a closed form expression
		*/
		void UpdateCovParGradOneIter(double lr, vec_t& cov_pars, bool closed_form_solution_sigma=true) {
			vec_t grad;
			if (closed_form_solution_sigma) {
				cov_pars[0] = 0.;
				for (const auto& cluster_i : unique_clusters_) {
					cov_pars[0] += (double)(y_[cluster_i].transpose() * y_aux_[cluster_i]);
				}
				cov_pars[0] /= num_data_;
				sigma2_ = cov_pars[0];
				grad = GetCovParGrad(false, false);
				cov_pars.segment(1, num_cov_par_ - 1) = (cov_pars.segment(1, num_cov_par_ - 1).array().log() - lr * grad.array()).exp().matrix();
			}
			else {
				grad = GetCovParGrad(true, false);
				cov_pars = (cov_pars.array().log() - lr * grad.array()).exp().matrix();
			}
			//for (int i = 0; i < (int)grad.size(); ++i) { Log::Debug("grad[%d]: %f", i, grad[i]); }//For debugging only
		}

		/*!
		* \brief Update covariance parameters doing one step of Fisher scoring (except for the marginal variance which is updated using an explicit solution)
		* \param[out] cov_pars Covariance parameters
		* \param closed_form_solution_sigma If true, the error variance (nugget effect) is calculated exactly using a closed form expression
		*/
		void UpdateCovParFisherScoringOneIter(vec_t& cov_pars, bool closed_form_solution_sigma = false) {
			vec_t grad;
			den_mat_t FI;
			if (closed_form_solution_sigma) {
				cov_pars[0] = 0.;
				for (const auto& cluster_i : unique_clusters_) {
					cov_pars[0] += (double)(y_[cluster_i].transpose() * y_aux_[cluster_i]);
				}
				cov_pars[0] /= num_data_;
				sigma2_ = cov_pars[0];

				grad = GetCovParGrad(false, true);
				CalcFisherInformation(cov_pars, FI, true, false, true);
				vec_t update = FI.llt().solve(grad);
				cov_pars.segment(1, num_cov_par_ - 1) = (cov_pars.segment(1, num_cov_par_ - 1).array().log() - update.array()).exp().matrix();//make update on log-scale
			}
			else {
				grad = GetCovParGrad(true, true);
				CalcFisherInformation(cov_pars, FI, true, true, true);
				vec_t update = FI.llt().solve(grad);
				cov_pars = (cov_pars.array().log() - update.array()).exp().matrix();//make update on log-scale
			}
			////For debugging only
			//for (int i = 0; i < (int)grad.size(); ++i) { Log::Debug("grad[%d]: %f", i, grad[i]); }
			////For debugging only
			//if (FI.cols() >= 3) {
			//	for (int i = 0; i < FI.rows(); ++i) { Log::Debug("FI[%d,:]: %f, %f, %f", i, FI.coeffRef(i, 0), FI.coeffRef(i, 1), FI.coeffRef(i, 2)); }
			//}
			//else {
			//	for (int i = 0; i < FI.rows(); ++i) { Log::Debug("FI[%d,:]: %f, %f", i, FI.coeffRef(i, 0), FI.coeffRef(i, 1)); }
			//}		
		}

		/*!
		* \brief Update linear fixed-effect coefficients doing one gradient descent step
		* \param lr Learning rate
		* \param marg_var Marginal variance parameters sigma^2
		* \param X Covariate data for linear fixed-effect
		* \param[out] beta Linear regression coefficients
		*/
		void UpdateCoefGradOneIter(double lr, double marg_var, den_mat_t& X, vec_t& beta) {
			vec_t y_aux(num_data_);
			GetYAux(y_aux);
			beta += lr * (1. / marg_var) * (X.transpose()) * y_aux;
		}

		/*!
		* \brief Update linear fixed-effect coefficients using generalized least squares (GLS)
		* \param X Covariate data for linear fixed-effect
		* \param[out] beta Linear regression coefficients
		*/
		void UpdateCoefGLS(den_mat_t& X, vec_t& beta) {
			vec_t y_aux(num_data_);
			GetYAux(y_aux);
			den_mat_t XT_psi_inv_X;
			CalcXTPsiInvX(X, XT_psi_inv_X);
			beta = XT_psi_inv_X.llt().solve(X.transpose() * y_aux);
		}

		/*!
		* \brief Check whether NaN's are presend
		* \param par Vector of parameters that should be checked
		*/
		void CheckNaN(vec_t& par) {
			if (std::isnan(par[0])) {
				Log::Fatal("NaN occurred. (if gradient descent is used, consider using a smaller learning rate)");
			}
		}

		/*!
		* \brief Calculate the Fisher information for covariance parameters. Note: you need to call CalcCovFactor first
		* \param cov_pars Covariance parameters
		* \param[out] FI Fisher information
		* \param transf_scale If true, the derivative is taken on the transformed scale otherwise on the original scale. Default = true
		* \param include_error_var If true, the marginal variance parameter is also included, otherwise not
		* \param use_saved_psi_inv If false, the inverse covariance matrix Psi^-1 is calculated, otherwise a saved version is used
		*/
		void CalcFisherInformation(const vec_t& cov_pars, den_mat_t& FI, bool transf_scale = true,
			bool include_error_var = false, bool use_saved_psi_inv = false) {
			if (include_error_var) {
				FI = den_mat_t(num_cov_par_, num_cov_par_);
			}
			else {
				FI = den_mat_t(num_cov_par_ - 1, num_cov_par_ - 1);
			}
			FI.setZero();

			for (const auto& cluster_i : unique_clusters_) {
				if (vecchia_approx_) {
					//Note: if transf_scale==false, then all matrices and derivatives have been calculated on the original scale for the Vecchia approximation, that is why there is no adjustment here
					//Calculate auxiliary matrices for use below
					sp_mat_t Identity(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
					Identity.setIdentity();
					sp_mat_t B_inv;
					eigen_sp_Lower_sp_RHS_cs_solve(B_[cluster_i], Identity, B_inv, true);
					sp_mat_t D = sp_mat_t(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
					D.setIdentity();
					D.diagonal().array() = D_inv_[cluster_i].diagonal().array().pow(-1);
					sp_mat_t D_inv_2 = sp_mat_t(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
					D_inv_2.setIdentity();
					D_inv_2.diagonal().array() = D_inv_[cluster_i].diagonal().array().pow(2);
					//Calculate derivative(B) * B^-1
					std::vector<sp_mat_t> B_grad_B_inv(num_cov_par_ - 1);
					for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
						B_grad_B_inv[par_nb] = B_grad_[cluster_i][par_nb] * B_inv;
					}
					//Calculate Fisher information
					int start_cov_pars = include_error_var ? 1 : 0;
					sp_mat_t D_inv_B_grad_B_inv, B_grad_B_inv_D;
					if (include_error_var) {
						//First calculate terms for nugget effect / noise variance parameter
						if (transf_scale) {//Optimization is done on transformed scale (in particular, log-scale)
							//The derivative for the nugget variance on the log scale is the original covariance matrix Psi, i.e. psi_inv_grad_psi_sigma2 is the identity matrix.
							FI(0, 0) += num_data_per_cluster_[cluster_i] / 2.;
							for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
								FI(0, par_nb + 1) += (double)((D_inv_[cluster_i].diagonal().array() * D_grad_[cluster_i][par_nb].diagonal().array()).sum()) / 2.;
							}
						}
						else {//Original scale for asymptotic covariance matrix
							int ind_grad_nugget = num_cov_par_ - 1;
							D_inv_B_grad_B_inv = D_inv_[cluster_i] * B_grad_[cluster_i][ind_grad_nugget] * B_inv;
							B_grad_B_inv_D = B_grad_[cluster_i][ind_grad_nugget] * B_inv * D;
							double diag = (double)((D_inv_2.diagonal().array() * D_grad_[cluster_i][ind_grad_nugget].diagonal().array() * D_grad_[cluster_i][ind_grad_nugget].diagonal().array()).sum());
							FI(0, 0) += ((double)(B_grad_B_inv_D.cwiseProduct(D_inv_B_grad_B_inv)).sum() + diag / 2.);

							for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
								B_grad_B_inv_D = B_grad_B_inv[par_nb] * D;
								diag = (double)((D_inv_2.diagonal().array() * D_grad_[cluster_i][ind_grad_nugget].diagonal().array() * D_grad_[cluster_i][par_nb].diagonal().array()).sum());
								FI(0, par_nb + 1) += ((double)(B_grad_B_inv_D.cwiseProduct(D_inv_B_grad_B_inv)).sum() + diag / 2.);
							}
						}
					}
					//Remaining covariance parameters
					for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
						D_inv_B_grad_B_inv = D_inv_[cluster_i] * B_grad_B_inv[par_nb];
						for (int par_nb_cross = par_nb; par_nb_cross < num_cov_par_ - 1; ++par_nb_cross) {
							B_grad_B_inv_D = B_grad_B_inv[par_nb_cross] * D;
							double diag = (double)((D_inv_2.diagonal().array() * D_grad_[cluster_i][par_nb].diagonal().array() * D_grad_[cluster_i][par_nb_cross].diagonal().array()).sum());
							FI(par_nb + start_cov_pars, par_nb_cross + start_cov_pars) += ((double)(B_grad_B_inv_D.cwiseProduct(D_inv_B_grad_B_inv)).sum() + diag / 2.);
						}
					}
				}//end Vecchia approximation
				else {//not Vecchia approximation
					T1 psi_inv;
					if (use_saved_psi_inv) {
						psi_inv = psi_inv_[cluster_i];
					}
					else {
						CalcPsiInv(psi_inv, cluster_i);
					}
					if (!transf_scale) {
						psi_inv /= cov_pars[0];//psi_inv has been calculated with a transformed parametrization, so we need to divide everything by cov_pars[0] to obtain the covariance matrix
					}
					//Calculate Psi^-1 * derivative(Psi)
					std::vector<T1> psi_inv_deriv_psi(num_cov_par_-1);
					int deriv_par_nb = 0;
					for (int j = 0; j < num_comps_total_; ++j) {//there is currently no possibility to loop over the parameters directly
						for (int jpar = 0; jpar < re_comps_[cluster_i][j]->num_cov_par_; ++jpar) {
							psi_inv_deriv_psi[deriv_par_nb] = psi_inv * *(re_comps_[cluster_i][j]->GetZSigmaZtGrad(jpar, transf_scale, cov_pars[0]));
							deriv_par_nb++;
						}
					}
					//Calculate Fisher information
					int start_cov_pars = include_error_var ? 1 : 0;
					if (include_error_var) {
						//First calculate terms for nugget effect / noise variance parameter
						if (transf_scale) {//Optimization is done on transformed scale (in particular, log-scale)
							//The derivative for the nugget variance on the log scale is the original covariance matrix Psi, i.e. psi_inv_grad_psi_sigma2 is the identity matrix.
							FI(0, 0) += num_data_per_cluster_[cluster_i] / 2.;
							for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
								FI(0, par_nb + 1) += psi_inv_deriv_psi[par_nb].diagonal().sum() / 2.;
							}
						}
						else {//Original scale for asymptotic covariance matrix
							//The derivative for the nugget variance is the identity matrix, i.e. psi_inv_grad_psi_sigma2 = psi_inv.
							FI(0, 0) += ((double)(psi_inv.cwiseProduct(psi_inv)).sum()) / 2.;
							for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
								FI(0, par_nb + 1) += ((double)(psi_inv.cwiseProduct(psi_inv_deriv_psi[par_nb])).sum()) / 2.;
							}
						}
					}
					//Remaining covariance parameters
					for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
						T1 psi_inv_grad_psi_par_nb_T = psi_inv_deriv_psi[par_nb].transpose();
						FI(par_nb + start_cov_pars, par_nb + start_cov_pars) += ((double)(psi_inv_grad_psi_par_nb_T.cwiseProduct(psi_inv_deriv_psi[par_nb])).sum()) / 2.;
						for (int par_nb_cross = par_nb + 1; par_nb_cross < num_cov_par_ - 1; ++par_nb_cross) {
							FI(par_nb + start_cov_pars, par_nb_cross + start_cov_pars) += ((double)(psi_inv_grad_psi_par_nb_T.cwiseProduct(psi_inv_deriv_psi[par_nb_cross])).sum()) / 2.;
						}
						psi_inv_deriv_psi[par_nb].resize(0, 0);//not needed anymore
						psi_inv_grad_psi_par_nb_T.resize(0, 0);
					}
				}//end not Vecchia approximation
			}//end loop over clusters
			FI.triangularView<Eigen::StrictlyLower>() = FI.triangularView<Eigen::StrictlyUpper>().transpose();
			//for (int i = 0; i < (int)FI.rows(); ++i) {//For debugging only
			//    for (int j = i; j < (int)FI.cols(); ++j) {
			//	    Log::Info("FI(%d,%d) %f", i, j, FI(i, j));
			//    }
			//}
		}

		/*!
		* \brief Calculate the standard deviations for the MLE of the covariance parameters as the diagonal of the inverse Fisher information (on the orignal scale and not the transformed scale used in the optimization)
		* \param cov_pars MLE of covariance parameters
		* \param[out] std_dev Standard deviations
		*/
		void CalcStdDevCovPar(const vec_t& cov_pars, vec_t& std_dev) {
			SetCovParsComps(cov_pars);
			CalcCovFactor(true, false, cov_pars[0], true);
			den_mat_t FI;
			CalcFisherInformation(cov_pars, FI, false, true, false);
			std_dev = FI.inverse().diagonal().array().sqrt().matrix();
		}

		/*!
		* \brief Calculate the standard deviations for the MLE of the regression coefficients as the diagonal of the inverse Fisher information
		* \param cov_pars MLE of covariance parameters
		* \param X Covariate data for linear fixed-effect
		* \param[out] std_dev Standard deviations
		*/
		void CalcStdDevCoef(vec_t& cov_pars, const den_mat_t& X, vec_t& std_dev) {
			if ((int)std_dev.size() >= num_data_) {
				Log::Warning("Sample size too small to calculate standard deviations for coefficients");
				for (int i = 0; i < (int)std_dev.size(); ++i) {
					std_dev[i] = std::numeric_limits<double>::quiet_NaN();
				}
			}
			else {
				SetCovParsComps(cov_pars);
				CalcCovFactor(false, true, 1., false);
				den_mat_t FI((int)X.cols(), (int)X.cols());
				CalcXTPsiInvX(X, FI);
				FI /= cov_pars[0];
				std_dev = FI.inverse().diagonal().array().sqrt().matrix();
			}
		}

		/*!
		 * \brief Calculate predictions (conditional mean and covariance matrix) (for one cluster
		 * \param cluster_i Cluster index for which prediction are made
		 * \param num_data_pred Number of prediction locations
		 * \param num_data_per_cluster_pred Keys: Labels of independent realizations of REs/GPs, values: number of prediction locations per independent realization
		 * \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
		 * \param re_group_levels_pred Group levels for the grouped random effects (re_group_levels_pred[j] contains the levels for RE number j)
		 * \param re_group_rand_coef_data_pred Random coefficient data for grouped REs
		 * \param gp_coords_mat_pred Coordinates for prediction locations
		 * \param gp_rand_coef_data_pred Random coefficient data for GPs
		 * \param predict_cov_mat If true, the covariance matrix is also calculated
		 * \param[out] mean_pred_id Predicted mean
		 * \param[out] cov_mat_pred_id Predicted covariance matrix
		 */
		void CalcPred(gp_id_t cluster_i, int num_data_pred,
			std::map<gp_id_t, int>& num_data_per_cluster_pred, std::map<gp_id_t, std::vector<int>>& data_indices_per_cluster_pred,
			const std::vector<std::vector<string_t>>& re_group_levels_pred, const double* re_group_rand_coef_data_pred,
			const den_mat_t& gp_coords_mat_pred, const double* gp_rand_coef_data_pred,
			bool predict_cov_mat, vec_t& mean_pred_id, T1& cov_mat_pred_id) {
			// Vector which contains covariance matrices needed for making predictions in the following order:
			//		0. Ztilde*Sigma*Z^T, 1. Zstar*Sigmatilde^T*Z^T, 2. Ztilde*Sigma*Ztilde^T, 3. Ztilde*Sigmatilde*Zstar^T, 4. Zstar*Sigmastar*Zstar^T
			std::vector<T1> pred_mats(5);
			//Define which covariance matrices are zero ('false') or non-zero ('true')
			std::vector<bool> active_mats{ false, false, false, false, false };
			if (num_re_group_total_ > 0) {
				active_mats[0] = true;
				active_mats[2] = true;
				active_mats[4] = true;
			}
			if (num_gp_total_ > 0) {
				active_mats[1] = true;
				active_mats[4] = true;
			}
			//Initialize covariance matrices
			for (int i = 0; i < 2; ++i) {
				if (active_mats[i]) {
					pred_mats[i].resize(num_data_per_cluster_pred[cluster_i], num_data_per_cluster_[cluster_i]);
					pred_mats[i].setZero();
				}
			}
			if (predict_cov_mat) {
				for (int i = 2; i < 5; ++i) {
					if (active_mats[i]) {
						pred_mats[i].resize(num_data_per_cluster_pred[cluster_i], num_data_per_cluster_pred[cluster_i]);
						pred_mats[i].setZero();
					}
				}
			}
			//Calculate covariance matrices
			int cn = 0;//component number

			if (num_re_group_ > 0) {
				//Grouped random effects
				for (int j = 0; j < num_re_group_; ++j) {
					std::shared_ptr<RECompGroup<T1>> re_comp = std::dynamic_pointer_cast<RECompGroup<T1>>(re_comps_[cluster_i][cn]);
					std::vector<re_group_t> group_data;
					for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
						group_data.push_back(re_group_levels_pred[j][id]);
					}
					re_comp->AddPredCovMatrices(group_data, pred_mats, predict_cov_mat);
					cn += 1;
				}
				if (num_re_group_rand_coef_ > 0) {
					//Random coefficient grouped random effects
					for (int j = 0; j < num_re_group_rand_coef_; ++j) {
						std::shared_ptr<RECompGroup<T1>> re_comp = std::dynamic_pointer_cast<RECompGroup<T1>>(re_comps_[cluster_i][cn]);
						std::vector<re_group_t> group_data;
						std::vector<double> rand_coef_data;
						for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
							rand_coef_data.push_back(re_group_rand_coef_data_pred[j * num_data_pred + id]);
							group_data.push_back(re_group_levels_pred[ind_effect_group_rand_coef_[j] - 1][id]);//subtract 1 since counting starts at one for this index
						}
						re_comp->AddPredCovMatrices(group_data, pred_mats, predict_cov_mat, rand_coef_data.data());
						cn += 1;
					}
				}
			}

			if (num_gp_ > 0) {
				//Gaussian process
				std::shared_ptr<RECompGP<T1>> re_comp_base = std::dynamic_pointer_cast<RECompGP<T1>>(re_comps_[cluster_i][cn]);
				re_comp_base->AddPredCovMatrices(re_comp_base->coords_, gp_coords_mat_pred, pred_mats, predict_cov_mat);
				cn += 1;
				if (num_gp_rand_coef_ > 0) {
					std::shared_ptr<RECompGP<T1>> re_comp;
					//Random coefficient Gaussian processes
					for (int j = 0; j < num_gp_rand_coef_; ++j) {
						re_comp = std::dynamic_pointer_cast<RECompGP<T1>>(re_comps_[cluster_i][cn]);
						std::vector<double> rand_coef_data;
						for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
							rand_coef_data.push_back(gp_rand_coef_data_pred[j * num_data_pred + id]);
						}
						re_comp->AddPredCovMatrices(re_comp_base->coords_, gp_coords_mat_pred, pred_mats, predict_cov_mat, rand_coef_data.data());
						cn += 1;
					}
				}
			}

			T1 M_aux(num_data_per_cluster_pred[cluster_i], num_data_per_cluster_[cluster_i]);//Ztilde*Sigma*Z^T + Zstar*Sigmatilde^T*Z^T
			M_aux.setZero();
			for (int i = 0; i < 2; ++i) {
				if (active_mats[i]) {
					M_aux += pred_mats[i];
				}
			}

			mean_pred_id = M_aux * y_aux_[cluster_i];

			if (predict_cov_mat) {
				cov_mat_pred_id.setIdentity();
				for (int i = 2; i < 5; ++i) {
					if (active_mats[i]) {
						cov_mat_pred_id += pred_mats[i];
						if (i == 3) {//Ztilde*Sigmatilde*Zstar^T
							cov_mat_pred_id += T1(pred_mats[i].transpose());
						}
					}
				}
				cov_mat_pred_id -= (M_aux * (chol_facts_solve_[cluster_i].solve(T1(M_aux.transpose()))));
			}
		}

		/*!
		* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the covariance matrix of the observable process when observed locations appear first in the ordering
		* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data
		* \param cluster_i Cluster index for which prediction are made
		* \param num_data_pred Number of prediction locations
		* \param num_data_per_cluster_pred Keys: Labels of independent realizations of REs/GPs, values: number of prediction locations per independent realization
		* \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
		* \param gp_coords_mat_obs Coordinates for observed locations
		* \param gp_coords_mat_pred Coordinates for prediction locations
		* \param gp_rand_coef_data_pred Random coefficient data for GPs
		* \param predict_cov_mat If true, the covariance matrix is also calculated
		* \param[out] mean_pred_id Predicted mean
		* \param[out] cov_mat_pred_id Predicted covariance matrix
		*/
		void CalcPredVecchiaObservedFirstOrder(bool CondObsOnly, gp_id_t cluster_i, int num_data_pred,
			std::map<gp_id_t, int>& num_data_per_cluster_pred, std::map<gp_id_t, std::vector<int>>& data_indices_per_cluster_pred,
			const den_mat_t& gp_coords_mat_obs, const den_mat_t& gp_coords_mat_pred, const double* gp_rand_coef_data_pred,
			bool predict_cov_mat, vec_t& mean_pred_id, T1& cov_mat_pred_id) {
			int num_data_cli = num_data_per_cluster_[cluster_i];
			int num_data_pred_cli = num_data_per_cluster_pred[cluster_i];
			//Find nearest neighbors
			den_mat_t coords_all(num_data_cli + num_data_pred_cli, dim_gp_coords_);
			coords_all << gp_coords_mat_obs, gp_coords_mat_pred;
			std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_data_pred_cli);
			std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_data_pred_cli);
			std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_data_pred_cli);
			if (CondObsOnly) {
				find_nearest_neighbors_Veccia_fast(coords_all, num_data_cli + num_data_pred_cli, num_neighbors_pred_,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_data_cli, num_data_cli - 1);
			}
			else {//find neighbors among both the observed and prediction locations
				find_nearest_neighbors_Veccia_fast(coords_all, num_data_cli + num_data_pred_cli, num_neighbors_pred_,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_data_cli, -1);
			}


			//Random coefficients
			std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_data_pred_cli);
			if (num_gp_rand_coef_ > 0) {
				for (int j = 0; j < num_gp_rand_coef_; ++j) {
					std::vector<double> rand_coef_data = re_comps_[cluster_i][ind_intercept_gp_ + j + 1]->rand_coef_data_;//First entries are the observed data, then the predicted data
					for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {//TODO: maybe do the following in parallel? (see CalcPredVecchiaPredictedFirstOrder)
						rand_coef_data.push_back(gp_rand_coef_data_pred[j * num_data_pred + id]);
					}
#pragma omp for schedule(static)
					for (int i = 0; i < num_data_pred_cli; ++i) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef_);
						int dim_z = (int)nearest_neighbors_cluster_i[i].size() + 1;
						vec_t coef_vec(dim_z);
						coef_vec(0) = rand_coef_data[num_data_cli + i];
						if ((num_data_cli + i) > 0) {
							for (int ii = 1; ii < dim_z; ++ii) {
								coef_vec(ii) = rand_coef_data[nearest_neighbors_cluster_i[i][ii - 1]];
							}
						}
						z_outer_z_obs_neighbors_cluster_i[i][j] = coef_vec * coef_vec.transpose();
					}
				}
			}

			// Determine Triplet for initializing Bpo and Bp
			std::vector<Triplet_t> entries_init_Bpo, entries_init_Bp;
			for (int i = 0; i < num_data_pred_cli; ++i) {
				entries_init_Bp.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
				for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i].size(); ++inn) {
					if (nearest_neighbors_cluster_i[i][inn] < num_data_cli) {//nearest neighbor belongs to observed data
						entries_init_Bpo.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn], 0.));
					}
					else {//nearest neighbor belongs to predicted data
						entries_init_Bp.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn] - num_data_cli, 0.));
					}
				}
			}
			sp_mat_t Bpo(num_data_pred_cli, num_data_cli);
			sp_mat_t Bp(num_data_pred_cli, num_data_pred_cli);
			Bpo.setFromTriplets(entries_init_Bpo.begin(), entries_init_Bpo.end());//initialize matrices (in order that the code below can be run in parallel)
			Bp.setFromTriplets(entries_init_Bp.begin(), entries_init_Bp.end());
			sp_mat_t Dp(num_data_pred_cli, num_data_pred_cli);
			Dp.setIdentity();//Put 1 on the diagonal (for nugget effect)

#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data_pred_cli; ++i) {

				int num_nn = (int)nearest_neighbors_cluster_i[i].size();
				//define covariance and gradient matrices
				den_mat_t cov_mat_obs_neighbors(1, num_nn);//dim = 1 x nn
				den_mat_t cov_mat_between_neighbors(num_nn, num_nn);//dim = nn x nn
				den_mat_t cov_grad_mats_obs_neighbors, cov_grad_mats_between_neighbors; //not used, just as mock argument for functions below
				for (int j = 0; j < num_gp_total_; ++j) {
					if (j == 0) {
						re_comps_[cluster_i][ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_obs_neighbors_cluster_i[i],
							cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors, cov_grad_mats_obs_neighbors, false);//write on matrices directly for first GP component
						re_comps_[cluster_i][ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_between_neighbors_cluster_i[i],
							cov_mat_between_neighbors, cov_grad_mats_between_neighbors, cov_grad_mats_between_neighbors, false);
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps_[cluster_i][ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_obs_neighbors_cluster_i[i],
							cov_mat_obs_neighbors_j, cov_grad_mats_obs_neighbors, cov_grad_mats_obs_neighbors, false);
						re_comps_[cluster_i][ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_between_neighbors_cluster_i[i],
							cov_mat_between_neighbors_j, cov_grad_mats_between_neighbors, cov_grad_mats_between_neighbors, false);
						//multiply by coefficient matrix
						cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(0, 1, 1, num_nn)).array();
						cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
						cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
						cov_mat_between_neighbors += cov_mat_between_neighbors_j;
					}
				}//end loop over components j

				//Calculate matrices A and D as well as their derivatives

				//1. add first summand of matrix D (ZCZ^T_{ii})
				for (int j = 0; j < num_gp_total_; ++j) {
					double d_comp_j = re_comps_[cluster_i][ind_intercept_gp_ + j]->cov_pars_[0];
					if (j > 0) {//random coefficient
						d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
					}
					Dp.coeffRef(i, i) += d_comp_j;
				}

				//2. remaining terms
				cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
				den_mat_t A_i(1, num_nn);//dim = 1 x nn
				A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors.transpose())).transpose();
				for (int inn = 0; inn < num_nn; ++inn) {
					if (nearest_neighbors_cluster_i[i][inn] < num_data_cli) {//nearest neighbor belongs to observed data
						Bpo.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
					}
					else {
						Bp.coeffRef(i, nearest_neighbors_cluster_i[i][inn] - num_data_cli) -= A_i(0, inn);
					}
				}
				Dp.coeffRef(i, i) -= (A_i * cov_mat_obs_neighbors.transpose())(0, 0);

			}//end loop over data i

			mean_pred_id = -Bpo * y_[cluster_i];
			if (!CondObsOnly) {
				sp_L_solve(Bp.valuePtr(), Bp.innerIndexPtr(), Bp.outerIndexPtr(), num_data_pred_cli, mean_pred_id.data());
			}

			if (predict_cov_mat) {
				if (CondObsOnly) {
					cov_mat_pred_id = Dp;
				}
				else {
					sp_mat_t Identity(num_data_pred_cli, num_data_pred_cli);
					Identity.setIdentity();
					sp_mat_t Bp_inv;
					eigen_sp_Lower_sp_RHS_cs_solve(Bp, Identity, Bp_inv, true);
					cov_mat_pred_id = T1(Bp_inv * Dp * Bp_inv.transpose());
				}
			}
		}

		/*!
		* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the covariance matrix of the observable proces when prediction locations appear first in the ordering
		* \param cluster_i Cluster index for which prediction are made
		* \param num_data_pred Number of prediction locations
		* \param num_data_per_cluster_pred Keys: Labels of independent realizations of REs/GPs, values: number of prediction locations per independent realization
		* \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
		* \param gp_coords_mat_obs Coordinates for observed locations
		* \param gp_coords_mat_pred Coordinates for prediction locations
		* \param gp_rand_coef_data_pred Random coefficient data for GPs
		* \param predict_cov_mat If true, the covariance matrix is also calculated
		* \param[out] mean_pred_id Predicted mean
		* \param[out] cov_mat_pred_id Predicted covariance matrix
		*/
		void CalcPredVecchiaPredictedFirstOrder(gp_id_t cluster_i, int num_data_pred,
			std::map<gp_id_t, int>& num_data_per_cluster_pred, std::map<gp_id_t, std::vector<int>>& data_indices_per_cluster_pred,
			const den_mat_t& gp_coords_mat_obs, const den_mat_t& gp_coords_mat_pred, const double* gp_rand_coef_data_pred,
			bool predict_cov_mat, vec_t& mean_pred_id, T1& cov_mat_pred_id) {
			int num_data_cli = num_data_per_cluster_[cluster_i];
			int num_data_pred_cli = num_data_per_cluster_pred[cluster_i];
			int num_data_tot = num_data_cli + num_data_pred_cli;
			//Find nearest neighbors
			den_mat_t coords_all(num_data_tot, dim_gp_coords_);
			coords_all << gp_coords_mat_pred, gp_coords_mat_obs;
			std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_data_tot);
			std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_data_tot);
			std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_data_tot);
			find_nearest_neighbors_Veccia_fast(coords_all, num_data_tot, num_neighbors_pred_,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1);

			//Prepare data for random coefficients
			std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_data_tot);
			if (num_gp_rand_coef_ > 0) {
				for (int j = 0; j < num_gp_rand_coef_; ++j) {
					std::vector<double> rand_coef_data(num_data_tot);//First entries are the predicted data, then the observed data
#pragma omp for schedule(static)
					for (int i = 0; i < num_data_pred_cli; ++i) {
						rand_coef_data[i] = gp_rand_coef_data_pred[j * num_data_pred + data_indices_per_cluster_pred[cluster_i][i]];
					}
#pragma omp for schedule(static)
					for (int i = 0; i < num_data_cli; ++i) {
						rand_coef_data[num_data_pred_cli + i] = re_comps_[cluster_i][ind_intercept_gp_ + j + 1]->rand_coef_data_[i];
					}
					//re_comps_[cluster_i][ind_intercept_gp_ + j + 1]->rand_coef_data_
					//for (int i = 0; i < rand_coef_data.size(); ++i) {
					//  Log::Info("rand_coef_data[%d]: %f", i, rand_coef_data[i]);
					//}
#pragma omp for schedule(static)
					for (int i = 0; i < num_data_tot; ++i) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef_);
						int dim_z = (int)nearest_neighbors_cluster_i[i].size() + 1;
						vec_t coef_vec(dim_z);
						coef_vec(0) = rand_coef_data[i];
						if (i > 0) {
							for (int ii = 1; ii < dim_z; ++ii) {
								coef_vec(ii) = rand_coef_data[nearest_neighbors_cluster_i[i][ii - 1]];
							}
						}
						z_outer_z_obs_neighbors_cluster_i[i][j] = coef_vec * coef_vec.transpose();
					}
				}
			}

			// Determine Triplet for initializing Bo, Bop, and Bp
			std::vector<Triplet_t> entries_init_Bo, entries_init_Bop, entries_init_Bp;
			for (int i = 0; i < num_data_pred_cli; ++i) {
				entries_init_Bp.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
				for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i].size(); ++inn) {
					entries_init_Bp.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn], 0.));
				}
			}
			for (int i = 0; i < num_data_cli; ++i) {
				entries_init_Bo.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
				for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i + num_data_pred_cli].size(); ++inn) {
					if (nearest_neighbors_cluster_i[i + num_data_pred_cli][inn] < num_data_pred_cli) {//nearest neighbor belongs to predicted data
						entries_init_Bop.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i + num_data_pred_cli][inn], 0.));
					}
					else {//nearest neighbor belongs to predicted data
						entries_init_Bo.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i + num_data_pred_cli][inn] - num_data_pred_cli, 0.));
					}
				}
			}
			sp_mat_t Bo(num_data_cli, num_data_cli);
			sp_mat_t Bop(num_data_cli, num_data_pred_cli);
			sp_mat_t Bp(num_data_pred_cli, num_data_pred_cli);
			Bo.setFromTriplets(entries_init_Bo.begin(), entries_init_Bo.end());//initialize matrices (in order that the code below can be run in parallel)
			Bop.setFromTriplets(entries_init_Bop.begin(), entries_init_Bop.end());
			Bp.setFromTriplets(entries_init_Bp.begin(), entries_init_Bp.end());

			sp_mat_t Do_inv(num_data_cli, num_data_cli);
			sp_mat_t Dp_inv(num_data_pred_cli, num_data_pred_cli);
			Do_inv.setIdentity();//Put 1 on the diagonal (for nugget effect)
			Dp_inv.setIdentity();

#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data_tot; ++i) {

				int num_nn = (int)nearest_neighbors_cluster_i[i].size();
				//define covariance and gradient matrices
				den_mat_t cov_mat_obs_neighbors(1, num_nn);//dim = 1 x nn
				den_mat_t cov_mat_between_neighbors(num_nn, num_nn);//dim = nn x nn
				den_mat_t cov_grad_mats_obs_neighbors, cov_grad_mats_between_neighbors; //not used, just as mock argument for functions below

				if (i > 0) {
					for (int j = 0; j < num_gp_total_; ++j) {
						if (j == 0) {
							re_comps_[cluster_i][ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_obs_neighbors_cluster_i[i],
								cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors, cov_grad_mats_obs_neighbors, false);//write on matrices directly for first GP component
							re_comps_[cluster_i][ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_between_neighbors_cluster_i[i],
								cov_mat_between_neighbors, cov_grad_mats_between_neighbors, cov_grad_mats_between_neighbors, false);
						}
						else {//random coefficient GPs
							den_mat_t cov_mat_obs_neighbors_j;
							den_mat_t cov_mat_between_neighbors_j;
							re_comps_[cluster_i][ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_obs_neighbors_cluster_i[i],
								cov_mat_obs_neighbors_j, cov_grad_mats_obs_neighbors, cov_grad_mats_obs_neighbors, false);
							re_comps_[cluster_i][ind_intercept_gp_ + j]->CalcSigmaAndSigmaGrad(dist_between_neighbors_cluster_i[i],
								cov_mat_between_neighbors_j, cov_grad_mats_between_neighbors, cov_grad_mats_between_neighbors, false);
							//multiply by coefficient matrix
							cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(0, 1, 1, num_nn)).array();
							cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
							cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
							cov_mat_between_neighbors += cov_mat_between_neighbors_j;
						}
					}//end loop over components j
				}

				//Calculate matrices A and D as well as their derivatives

				//1. add first summand of matrix D (ZCZ^T_{ii})
				for (int j = 0; j < num_gp_total_; ++j) {
					double d_comp_j = re_comps_[cluster_i][ind_intercept_gp_ + j]->cov_pars_[0];
					if (j > 0) {//random coefficient
						d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
					}
					if (i < num_data_pred_cli) {
						Dp_inv.coeffRef(i, i) += d_comp_j;
					}
					else {
						Do_inv.coeffRef(i - num_data_pred_cli, i - num_data_pred_cli) += d_comp_j;
					}
				}

				//2. remaining terms
				if (i > 0) {
					cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
					den_mat_t A_i(1, num_nn);//dim = 1 x nn
					A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors.transpose())).transpose();
					for (int inn = 0; inn < num_nn; ++inn) {
						if (i < num_data_pred_cli) {
							Bp.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
						}
						else {
							if (nearest_neighbors_cluster_i[i][inn] < num_data_pred_cli) {//nearest neighbor belongs to predicted data
								Bop.coeffRef(i - num_data_pred_cli, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
							}
							else {
								Bo.coeffRef(i - num_data_pred_cli, nearest_neighbors_cluster_i[i][inn] - num_data_pred_cli) -= A_i(0, inn);
							}
						}
					}

					if (i < num_data_pred_cli) {
						Dp_inv.coeffRef(i, i) -= (A_i * cov_mat_obs_neighbors.transpose())(0, 0);
					}
					else {
						Do_inv.coeffRef(i - num_data_pred_cli, i - num_data_pred_cli) -= (A_i * cov_mat_obs_neighbors.transpose())(0, 0);
					}

				}

				if (i < num_data_pred_cli) {
					Dp_inv.coeffRef(i, i) = 1 / Dp_inv.coeffRef(i, i);
				}
				else {
					Do_inv.coeffRef(i - num_data_pred_cli, i - num_data_pred_cli) = 1 / Do_inv.coeffRef(i - num_data_pred_cli, i - num_data_pred_cli);
				}

			}//end loop over data i

			sp_mat_t cond_prec = Bp.transpose() * Dp_inv * Bp + Bop.transpose() * Do_inv * Bop;
			chol_sp_mat_t CholFact;
			CholFact.compute(cond_prec);

			if (predict_cov_mat) {
				sp_mat_t Identity(num_data_pred_cli, num_data_pred_cli);
				Identity.setIdentity();
				sp_mat_t cond_prec_chol = CholFact.matrixL();
				sp_mat_t cond_prec_chol_inv;
				eigen_sp_Lower_sp_RHS_cs_solve(cond_prec_chol, Identity, cond_prec_chol_inv, true);
				cov_mat_pred_id = T1(cond_prec_chol_inv.transpose() * cond_prec_chol_inv);
				mean_pred_id = -cov_mat_pred_id * Bop.transpose() * Do_inv * Bo * y_[cluster_i];

			}
			else {
				mean_pred_id = -CholFact.solve(Bop.transpose() * Do_inv * Bo * y_[cluster_i]);
			}
		}

		/*!
		 * \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the latent process when observed locations appear first in the ordering
		 * \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data
		 * \param cluster_i Cluster index for which prediction are made
		 * \param num_data_per_cluster_pred Keys: Labels of independent realizations of REs/GPs, values: number of prediction locations per independent realization
		 * \param gp_coords_mat_obs Coordinates for observed locations
		 * \param gp_coords_mat_pred Coordinates for prediction locations
		 * \param predict_cov_mat If true, the covariance matrix is also calculated
		 * \param[out] mean_pred_id Predicted mean
		 * \param[out] cov_mat_pred_id Predicted covariance matrix
		 */
		void CalcPredVecchiaLatentObservedFirstOrder(bool CondObsOnly, gp_id_t cluster_i,
			std::map<gp_id_t, int>& num_data_per_cluster_pred,
			const den_mat_t& gp_coords_mat_obs, const den_mat_t& gp_coords_mat_pred,
			bool predict_cov_mat, vec_t& mean_pred_id, T1& cov_mat_pred_id) {
			if (num_gp_rand_coef_ > 0) {
				Log::Fatal("The Vecchia approximation for the latent process is currently not implemented when having random coefficients");
			}
			int num_data_cli = num_data_per_cluster_[cluster_i];
			int num_data_pred_cli = num_data_per_cluster_pred[cluster_i];
			int num_data_tot = num_data_cli + num_data_pred_cli;
			//Find nearest neighbors
			den_mat_t coords_all(num_data_cli + num_data_pred_cli, dim_gp_coords_);
			coords_all << gp_coords_mat_obs, gp_coords_mat_pred;

			//Determine number of unique observartion locations
			std::vector<int> uniques;//unique points
			std::vector<int> unique_idx;//used for constructing incidence matrix Z_ if there are duplicates
			DetermineUniqueDuplicateCoords(gp_coords_mat_obs, num_data_cli, uniques, unique_idx);
			int num_coord_unique_obs = (int)uniques.size();

			//Determine unique locations (observed and predicted)
			DetermineUniqueDuplicateCoords(coords_all, num_data_tot, uniques, unique_idx);
			int num_coord_unique = (int)uniques.size();
			den_mat_t coords_all_unique;
			if ((int)uniques.size() == num_data_tot) {//no multiple observations at the same locations -> no incidence matrix needed
				coords_all_unique = coords_all;
			}
			else {
				coords_all_unique = coords_all(uniques, Eigen::all);
			}

			//Determine incidence matrices
			sp_mat_t Z_o = sp_mat_t(num_data_cli, uniques.size());
			sp_mat_t Z_p = sp_mat_t(num_data_pred_cli, uniques.size());
			for (int i = 0; i < num_data_tot; ++i) {
				if (i < num_data_cli) {
					Z_o.insert(i, unique_idx[i]) = 1.;
				}
				else {
					Z_p.insert(i - num_data_cli, unique_idx[i]) = 1.;
				}
			}

			std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_coord_unique);
			std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_coord_unique);
			std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_coord_unique);
			if (CondObsOnly) {//find neighbors among both the observed locations only
				find_nearest_neighbors_Veccia_fast(coords_all_unique, num_coord_unique, num_neighbors_pred_,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, num_coord_unique_obs - 1);
			}
			else {//find neighbors among both the observed and prediction locations
				find_nearest_neighbors_Veccia_fast(coords_all_unique, num_coord_unique, num_neighbors_pred_,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1);
			}

			// Determine Triplet for initializing Bpo and Bp
			std::vector<Triplet_t> entries_init_B;
			for (int i = 0; i < num_coord_unique; ++i) {
				entries_init_B.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
				for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i].size(); ++inn) {
					entries_init_B.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn], 0.));
				}
			}
			sp_mat_t B(num_coord_unique, num_coord_unique);
			B.setFromTriplets(entries_init_B.begin(), entries_init_B.end());//initialize matrices (in order that the code below can be run in parallel)
			sp_mat_t D(num_coord_unique, num_coord_unique);
			D.setIdentity();
			D.diagonal().array() = 0.;

#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_coord_unique; ++i) {

				int num_nn = (int)nearest_neighbors_cluster_i[i].size();
				//define covariance and gradient matrices
				den_mat_t cov_mat_obs_neighbors(1, num_nn);//dim = 1 x nn
				den_mat_t cov_mat_between_neighbors(num_nn, num_nn);//dim = nn x nn
				den_mat_t cov_grad_mats_obs_neighbors, cov_grad_mats_between_neighbors; //not used, just as mock argument for functions below

				if (i > 0) {
					re_comps_[cluster_i][ind_intercept_gp_]->CalcSigmaAndSigmaGrad(dist_obs_neighbors_cluster_i[i],
						cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors, cov_grad_mats_obs_neighbors, false);//write on matrices directly for first GP component
					re_comps_[cluster_i][ind_intercept_gp_]->CalcSigmaAndSigmaGrad(dist_between_neighbors_cluster_i[i],
						cov_mat_between_neighbors, cov_grad_mats_between_neighbors, cov_grad_mats_between_neighbors, false);
				}

				//Calculate matrices A and D as well as their derivatives

				//1. add first summand of matrix D (ZCZ^T_{ii})
				D.coeffRef(i, i) = re_comps_[cluster_i][ind_intercept_gp_]->cov_pars_[0];

				//2. remaining terms
				if (i > 0) {
					den_mat_t A_i(1, num_nn);//dim = 1 x nn
					A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors.transpose())).transpose();
					for (int inn = 0; inn < num_nn; ++inn) {
						B.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
					}
					D.coeffRef(i, i) -= (A_i * cov_mat_obs_neighbors.transpose())(0, 0);
				}

			}//end loop over data i

			//Calculate D_inv and B_inv in order to calcualte Sigma and Sigma^-1
			sp_mat_t D_inv(num_coord_unique, num_coord_unique);
			D_inv.setIdentity();
			D_inv.diagonal().array() = D.diagonal().array().pow(-1);

			sp_mat_t Identity_all(num_coord_unique, num_coord_unique);
			Identity_all.setIdentity();
			sp_mat_t B_inv;
			eigen_sp_Lower_sp_RHS_cs_solve(B, Identity_all, B_inv, true);

			//Calculate inverse of covariance matrix for observed data using the Woodbury identity
			sp_mat_t Z_o_T = Z_o.transpose();
			sp_mat_t M_aux_Woodbury = B.transpose() * D_inv * B + Z_o_T * Z_o;
			chol_sp_mat_t CholFac_M_aux_Woodbury;
			CholFac_M_aux_Woodbury.compute(M_aux_Woodbury);

			if (predict_cov_mat) {
				//Using Eigen's solver
				sp_mat_t M_aux_Woodbury2 = CholFac_M_aux_Woodbury.solve(Z_o_T);
				sp_mat_t Identity_obs(num_data_cli, num_data_cli);
				Identity_obs.setIdentity();
				sp_mat_t ZoSigmaZoT_plusI_Inv = -Z_o * M_aux_Woodbury2 + Identity_obs;

				sp_mat_t ZpSigmaZoT = Z_p * B_inv * D * B_inv.transpose() * Z_o_T;

				sp_mat_t M_aux = ZpSigmaZoT * ZoSigmaZoT_plusI_Inv;

				mean_pred_id = M_aux * y_[cluster_i];

				sp_mat_t Identity_pred(num_data_pred_cli, num_data_pred_cli);
				Identity_pred.setIdentity();
				cov_mat_pred_id = T1(Z_p * B_inv * D * B_inv.transpose() * Z_p.transpose() + Identity_pred - M_aux * ZpSigmaZoT.transpose());

			}
			else {
				vec_t resp_aux = Z_o_T * y_[cluster_i];
				vec_t resp_aux2 = CholFac_M_aux_Woodbury.solve(resp_aux);
				resp_aux = y_[cluster_i] - Z_o * resp_aux2;
				mean_pred_id = Z_p * B_inv * D * B_inv.transpose() * Z_o_T * resp_aux;
			}

		}

		friend class REModel;

	};

}  // namespace GPBoost

#endif   // GPB_RE_MODEL_TEMPLATE_H_
