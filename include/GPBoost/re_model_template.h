/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2024 Fabio Sigrist and Tim Gyger. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_RE_MODEL_TEMPLATE_H_
#define GPB_RE_MODEL_TEMPLATE_H_

#define _USE_MATH_DEFINES // for M_PI
#include <cmath>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <GPBoost/sparse_matrix_utils.h>
#include <GPBoost/Vecchia_utils.h>
#include <GPBoost/GP_utils.h>
#include <GPBoost/likelihoods.h>
#include <GPBoost/utils.h>
#include <GPBoost/optim_utils.h>
//#include <Eigen/src/misc/lapack.h>

#include <memory>
#include <mutex>
#include <vector>
#include <algorithm>    // std::shuffle
#include <chrono>  // only for debugging
#include <thread> // only for debugging

#ifndef M_PI
#define M_PI      3.1415926535897932384626433832795029
#endif

#include <LightGBM/utils/log.h>
using LightGBM::Log;
using LightGBM::LogLevelRE;
#include <LightGBM/utils/common.h>
#include <LightGBM/meta.h>
using LightGBM::label_t;

namespace GPBoost {

	/*!
	* \brief Template class used in the wrapper class REModel
	* The template parameters <T_mat, T_chol> can be <den_mat_t, chol_den_mat_t>, <sp_mat_t, chol_sp_mat_t>, <sp_mat_rm_t, chol_sp_mat_rm_t>
	*	depending on whether dense or sparse linear matrix algebra is used
	*/
	template<typename T_mat, typename T_chol>
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
		REModelTemplate(data_size_t num_data,
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
			int seed) {
			CHECK(num_data > 0);
			num_data_ = num_data;
			//Initialize RNG
			CHECK(seed >= 0);
			rng_ = RNG_t(seed);
			//Set up likelihood
			string_t likelihood_strg;
			if (likelihood == nullptr) {
				likelihood_strg = "gaussian";
			}
			else {
				likelihood_strg = Likelihood<T_mat, T_chol>::ParseLikelihoodAlias(std::string(likelihood));
			}
			gauss_likelihood_ = likelihood_strg == "gaussian";
			//Set up GP approximation
			if (gp_approx == nullptr) {
				gp_approx_ = "none";
			}
			else {
				gp_approx_ = std::string(gp_approx);
			}
			if (SUPPORTED_GP_APPROX_.find(gp_approx_) == SUPPORTED_GP_APPROX_.end()) {
				Log::REFatal("GP approximation '%s' is currently not supported ", gp_approx_.c_str());
			}
			//Set up matrix inversion method
			if (matrix_inversion_method != nullptr) {
				matrix_inversion_method_ = std::string(matrix_inversion_method);
				if (SUPPORTED_MATRIX_INVERSION_METHODS_.find(matrix_inversion_method_) == SUPPORTED_MATRIX_INVERSION_METHODS_.end()) {
					Log::REFatal("Matrix inversion method '%s' is not supported.", matrix_inversion_method_.c_str());
				}
			}
			//Set up GP IDs
			SetUpGPIds(num_data_, cluster_ids_data, num_data_per_cluster_, data_indices_per_cluster_, unique_clusters_, num_clusters_);
			num_comps_total_ = 0;
			//Do some checks for grouped RE components and set meta data (number of components etc.)
			std::vector<std::vector<re_group_t>> re_group_levels;//Matrix with group levels for the grouped random effects (re_group_levels[j] contains the levels for RE number j)
			if (num_re_group > 0) {
				if (gp_approx_ != "none") {
					Log::REFatal("The GP approximation '%s' can currently not be used when there are grouped random effects ", gp_approx_.c_str());
				}
				num_re_group_ = num_re_group;
				num_group_variables_ = num_re_group_;
				drop_intercept_group_rand_effect_ = std::vector<bool>(num_re_group_);
				for (int j = 0; j < num_re_group_; ++j) {
					drop_intercept_group_rand_effect_[j] = false;
				}
				CHECK(re_group_data != nullptr);
				if (num_re_group_rand_coef > 0) {
					num_re_group_rand_coef_ = num_re_group_rand_coef;
					CHECK(re_group_rand_coef_data != nullptr);
					CHECK(ind_effect_group_rand_coef != nullptr);
					for (int j = 0; j < num_re_group_rand_coef_; ++j) {
						CHECK(0 < ind_effect_group_rand_coef[j] && ind_effect_group_rand_coef[j] <= num_re_group_);
					}
					ind_effect_group_rand_coef_ = std::vector<int>(ind_effect_group_rand_coef, ind_effect_group_rand_coef + num_re_group_rand_coef_);
					if (drop_intercept_group_rand_effect != nullptr) {
						drop_intercept_group_rand_effect_ = std::vector<bool>(num_re_group_);
						for (int j = 0; j < num_re_group_; ++j) {
							drop_intercept_group_rand_effect_[j] = drop_intercept_group_rand_effect[j] > 0;
						}
						for (int j = 0; j < num_re_group_; ++j) { // check that all dropped intercept random effects have at least on random slope
							if (drop_intercept_group_rand_effect_[j] &&
								std::find(ind_effect_group_rand_coef_.begin(), ind_effect_group_rand_coef_.end(), j) != ind_effect_group_rand_coef_.end()) {
								Log::REFatal("Cannot drop intercept random effect number %d as this random effect has no corresponding random coefficients", j);
							}
						}
					}
				}
				num_re_group_total_ = num_re_group_ + num_re_group_rand_coef_;
				num_comps_total_ += num_re_group_total_;
				// Convert characters in 'const char* re_group_data' to matrix (num_group_variables_ x num_data_) with strings of group labels
				re_group_levels = std::vector<std::vector<re_group_t>>(num_group_variables_, std::vector<re_group_t>(num_data_));
				if (num_group_variables_ > 0) {
					ConvertCharToStringGroupLevels(num_data_, num_group_variables_, re_group_data, re_group_levels);
				}
			}
			//Do some checks for GP components and set meta data (number of components etc.)
			if (num_gp > 0) {
				if (num_gp > 1) {
					Log::REFatal("num_gp can only be either 0 or 1 in the current implementation");
				}
				num_gp_ = num_gp;
				ind_intercept_gp_ = num_comps_total_;
				CHECK(dim_gp_coords > 0);
				CHECK(gp_coords_data != nullptr);
				CHECK(cov_fct != nullptr);
				dim_gp_coords_ = dim_gp_coords;
				cov_fct_ = std::string(cov_fct);
				cov_fct_shape_ = cov_fct_shape;
				cov_fct_taper_range_ = cov_fct_taper_range;
				cov_fct_taper_shape_ = cov_fct_taper_shape;
				if (gp_approx_ == "vecchia") {
					Log::REInfo("Starting nearest neighbor search for Vecchia approximation");
					CHECK(num_neighbors > 0);
					num_neighbors_ = num_neighbors;
					num_neighbors_pred_ = 2 * num_neighbors_;
					if (vecchia_ordering == nullptr) {
						vecchia_ordering_ = "none";
					}
					else {
						vecchia_ordering_ = std::string(vecchia_ordering);
						if (SUPPORTED_VECCHIA_ORDERING_.find(vecchia_ordering_) == SUPPORTED_VECCHIA_ORDERING_.end()) {
							Log::REFatal("Ordering of type '%s' is not supported for the Veccia approximation ", vecchia_ordering_.c_str());
						}
					}
				}//end if gp_approx_ == "vecchia"
				else if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
					CHECK(num_ind_points > 0);
					num_ind_points_ = num_ind_points;
					CHECK(cover_tree_radius > 0);
					cover_tree_radius_ = cover_tree_radius;
					ind_points_selection_ = std::string(ind_points_selection);
					if (SUPPORTED_METHOD_INDUCING_POINTS_.find(ind_points_selection_) == SUPPORTED_METHOD_INDUCING_POINTS_.end()) {
						Log::REFatal("Method '%s' is not supported for choosing the inducing points ", ind_points_selection_.c_str());
					}
				}
				if (num_gp_rand_coef > 0) {//Random slopes
					CHECK(gp_rand_coef_data != nullptr);
					num_gp_rand_coef_ = num_gp_rand_coef;
				}
				num_gp_total_ = num_gp_ + num_gp_rand_coef_;
				num_comps_total_ += num_gp_total_;
			}
			DetermineSpecialCasesModelsEstimationPrediction();
			//Create RE/GP component models
			for (const auto& cluster_i : unique_clusters_) {
				if (gp_approx_ == "vecchia") {
					std::vector<std::shared_ptr<RECompBase<T_mat>>> re_comps_cluster_i;
					std::vector<std::vector<int>> nearest_neighbors_cluster_i;
					std::vector<den_mat_t> dist_obs_neighbors_cluster_i;
					std::vector<den_mat_t> dist_between_neighbors_cluster_i;
					std::vector<Triplet_t> entries_init_B_cluster_i;
					std::vector<Triplet_t> entries_init_B_grad_cluster_i;
					std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i;
					CreateREComponentsVecchia<T_mat>(num_data_, dim_gp_coords_, data_indices_per_cluster_, cluster_i,
						num_data_per_cluster_, gp_coords_data, gp_rand_coef_data,
						re_comps_cluster_i, nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i,
						entries_init_B_cluster_i, entries_init_B_grad_cluster_i, z_outer_z_obs_neighbors_cluster_i, only_one_GP_calculations_on_RE_scale_, has_duplicates_coords_,
						vecchia_ordering_, num_neighbors_, vecchia_neighbor_selection_, true, rng_, num_gp_rand_coef_, num_gp_total_, num_comps_total_, gauss_likelihood_,
						cov_fct_, cov_fct_shape_, cov_fct_taper_range_, cov_fct_taper_shape_, gp_approx_ == "tapering");
					nearest_neighbors_.insert({ cluster_i, nearest_neighbors_cluster_i });
					dist_obs_neighbors_.insert({ cluster_i, dist_obs_neighbors_cluster_i });
					dist_between_neighbors_.insert({ cluster_i, dist_between_neighbors_cluster_i });
					entries_init_B_.insert({ cluster_i, entries_init_B_cluster_i });
					entries_init_B_grad_.insert({ cluster_i, entries_init_B_grad_cluster_i });
					z_outer_z_obs_neighbors_.insert({ cluster_i, z_outer_z_obs_neighbors_cluster_i });
					re_comps_.insert({ cluster_i, re_comps_cluster_i });
				}//end gp_approx_ == "vecchia"
				else if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
					std::vector<std::shared_ptr<RECompGP<den_mat_t>>> re_comps_ip_cluster_i;
					std::vector<std::shared_ptr<RECompGP<den_mat_t>>> re_comps_cross_cov_cluster_i;
					std::vector<std::shared_ptr<RECompGP<T_mat>>> re_comps_resid_cluster_i;
					CreateREComponentsPPFSA(num_data_, data_indices_per_cluster_, cluster_i, gp_coords_data,
						re_comps_ip_cluster_i, re_comps_cross_cov_cluster_i, re_comps_resid_cluster_i);
					re_comps_ip_.insert({ cluster_i, re_comps_ip_cluster_i });
					re_comps_cross_cov_.insert({ cluster_i, re_comps_cross_cov_cluster_i });
					re_comps_resid_.insert({ cluster_i, re_comps_resid_cluster_i });
				}
				else {
					std::vector<std::shared_ptr<RECompBase<T_mat>>> re_comps_cluster_i;
					CreateREComponents(num_data_,
						data_indices_per_cluster_,
						cluster_i,
						re_group_levels,
						num_data_per_cluster_,
						re_group_rand_coef_data,
						gp_coords_data,
						gp_rand_coef_data,
						!only_grouped_REs_use_woodbury_identity_,
						re_comps_cluster_i);
					re_comps_.insert({ cluster_i, re_comps_cluster_i });
				}
			}//end loop over clusters
			//Create matrices Z and ZtZ if Woodbury identity is used (used only if there are only grouped REs and no GPs)
			if (only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_) {
				InitializeMatricesForOnlyGroupedREsUseWoodburyIdentity();
			}
			if (gp_approx_ != "vecchia") {
				InitializeIdentityMatricesForGaussianData();
			}
			if (gp_approx_ == "vecchia") {
				Log::REInfo("Nearest neighbors for Vecchia approximation found");
			}
			InitializeLikelihoods(likelihood_strg);
			DetermineCovarianceParameterIndicesNumCovPars();
			InitializeDefaultSettings();
			CheckCompatibilitySpecialOptions();
			SetMatrixInversionPropertiesLikelihood();
		}//end REModelTemplate

		/*! \brief Destructor */
		~REModelTemplate() {
		}

		/*! \brief Disable copy */
		REModelTemplate& operator=(const REModelTemplate&) = delete;

		/*! \brief Disable copy */
		REModelTemplate(const REModelTemplate&) = delete;

		/*!
		* \brief Returns the type of likelihood
		*/
		string_t GetLikelihood() {
			return(likelihood_[unique_clusters_[0]]->GetLikelihood());
		}

		/*!
		* \brief Set / change the type of likelihood
		* \param likelihood Likelihood name
		*/
		void SetLikelihood(const string_t& likelihood) {
			bool gauss_likelihood_before = gauss_likelihood_;
			bool only_one_grouped_RE_calculations_on_RE_scale_before = only_one_grouped_RE_calculations_on_RE_scale_;
			bool only_one_GP_calculations_on_RE_scale_before = only_one_GP_calculations_on_RE_scale_;
			bool only_grouped_REs_use_woodbury_identity_before = only_grouped_REs_use_woodbury_identity_;
			gauss_likelihood_ = (Likelihood<T_mat, T_chol>::ParseLikelihoodAlias(likelihood) == "gaussian");
			DetermineSpecialCasesModelsEstimationPrediction();
			CheckCompatibilitySpecialOptions();
			//Make adaptions in re_comps_ for special options when switching between Gaussian and non-Gaussian likelihoods
			if (gauss_likelihood_before && !gauss_likelihood_) {
				if (only_one_GP_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_) {
					CHECK(gp_approx_ != "fitc" && gp_approx_ != "full_scale_tapering");
					for (const auto& cluster_i : unique_clusters_) {
						re_comps_[cluster_i][0]->DropZ();
					}
				}
			}
			else if (!gauss_likelihood_before && gauss_likelihood_) {
				if (only_one_GP_calculations_on_RE_scale_before && gp_approx_ == "vecchia") {
					Log::REFatal("Cannot change the likelihood to 'gaussian' when using a Vecchia approximation and having duplicate coordinates");
				}
				if (only_one_GP_calculations_on_RE_scale_before || only_one_grouped_RE_calculations_on_RE_scale_before) {
					CHECK(gp_approx_ != "fitc" && gp_approx_ != "full_scale_tapering");
					for (const auto& cluster_i : unique_clusters_) {
						re_comps_[cluster_i][0]->AddZ();
					}
				}
			}
			//Matrices used when only_grouped_REs_use_woodbury_identity_==true 
			if ((only_grouped_REs_use_woodbury_identity_ && !only_grouped_REs_use_woodbury_identity_before) ||
				(only_grouped_REs_use_woodbury_identity_ && only_one_grouped_RE_calculations_on_RE_scale_before && !only_one_grouped_RE_calculations_on_RE_scale_)) {
				InitializeMatricesForOnlyGroupedREsUseWoodburyIdentity();
			}
			else if (!only_grouped_REs_use_woodbury_identity_) {
				//Delete not required matrices
				Zt_ = std::map<data_size_t, sp_mat_t>();
				P_Zt_ = std::map<data_size_t, sp_mat_t>();
				ZtZ_ = std::map<data_size_t, sp_mat_t>();
				cum_num_rand_eff_ = std::map<data_size_t, std::vector<data_size_t>>();
				Zj_square_sum_ = std::map<data_size_t, std::vector<double>>();
				ZtZj_ = std::map<data_size_t, std::vector<sp_mat_t>>();
				P_ZtZj_ = std::map<data_size_t, std::vector<sp_mat_t>>();
			}
			//Identity matrices for Gaussian data
			if (!gauss_likelihood_before && gauss_likelihood_) {
				InitializeIdentityMatricesForGaussianData();
			}
			else if (gauss_likelihood_before && !gauss_likelihood_) {
				//Delete not required matrices
				Id_ = std::map<data_size_t, T_mat>();
				P_Id_ = std::map<data_size_t, T_mat>();
				if (gp_approx_ == "vecchia" && has_duplicates_coords_) {
					Log::REFatal("Cannot change the likelihood from 'gaussian' to another one when using a Vecchia approximation and having duplicate coordinates");
				}
			}
			InitializeLikelihoods(likelihood);
			DetermineCovarianceParameterIndicesNumCovPars();
			InitializeDefaultSettings();
			CheckPreconditionerType();
			SetMatrixInversionPropertiesLikelihood();
		}//end SetLikelihood

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
			const data_size_t num_data) {
			return(likelihood_[unique_clusters_[0]]->TestNegLogLikelihoodAdaptiveGHQuadrature(y_test, pred_mean, pred_var, num_data));
		}

		/*!
		* \brief Set configuration parameters for the optimizer
		* \param lr Learning rate for covariance parameters. If lr<= 0, internal default values are used (0.1 for "gradient_descent" and 1. for "fisher_scoring")
		* \param acc_rate_cov Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if nesterov_schedule_version == 0).
		* \param max_iter Maximal number of iterations
		* \param delta_rel_conv Convergence tolerance. The algorithm stops if the relative change in eiher the log-likelihood or the parameters is below this value. For "bfgs", the L2 norm of the gradient is used instead of the relative change in the log-likelihood
		* \param use_nesterov_acc Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters (only used for "gradient_descent")e
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used (only relevant if use_nesterov_acc)
		* \param optimizer_cov Optimizer for covariance parameters
		* \param momentum_offset Number of iterations for which no mometum is applied in the beginning (only relevant if use_nesterov_acc)
		* \param convergence_criterion The convergence criterion used for terminating the optimization algorithm. Options: "relative_change_in_log_likelihood" or "relative_change_in_parameters"
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
		* \param estimate_aux_pars If true, any additional parameters for non-Gaussian likelihoods are also estimated (e.g., shape parameter of gamma likelihood)
		*/
		void SetOptimConfig(double lr,
			double acc_rate_cov,
			int max_iter,
			double delta_rel_conv,
			bool use_nesterov_acc,
			int nesterov_schedule_version,
			const char* optimizer,
			int momentum_offset,
			const char* convergence_criterion,
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
			bool estimate_aux_pars) {
			lr_cov_init_ = lr;
			acc_rate_cov_ = acc_rate_cov;
			max_iter_ = max_iter;
			delta_rel_conv_init_ = delta_rel_conv;
			use_nesterov_acc_ = use_nesterov_acc;
			nesterov_schedule_version_ = nesterov_schedule_version;
			if (optimizer != nullptr) {
				optimizer_cov_pars_ = std::string(optimizer);
				if (optimizer_cov_pars_ == "gradient_descent_constant_change" ||
					optimizer_cov_pars_ == "newton_constant_change" ||
					optimizer_cov_pars_ == "fisher_scoring_constant_change") {
					learning_rate_constant_first_order_change_ = true;
				}
				else {
					learning_rate_constant_first_order_change_ = false;
				}
				if (optimizer_cov_pars_ == "gradient_descent_constant_change" ||
					optimizer_cov_pars_ == "gradient_descent_increase_lr" ||
					optimizer_cov_pars_ == "gradient_descent_reset_lr") {
					optimizer_cov_pars_ = "gradient_descent";
				}
				if (optimizer_cov_pars_ == "newt_constant_change") {
					optimizer_cov_pars_ = "newton";
				}
				if (optimizer_cov_pars_ == "fisher_scoring_constant_change") {
					optimizer_cov_pars_ = "fisher_scoring";
				}
				if (optimizer_cov_pars_ == "gradient_descent_increase_lr") {
					increase_learning_rate_again_ = true;
				}
				else {
					increase_learning_rate_again_ = false;
				}
				if (optimizer_cov_pars_ == "gradient_descent_reset_lr") {
					reset_learning_rate_every_iteration_ = true;
				}
				else {
					reset_learning_rate_every_iteration_ = false;
				}
			}
			momentum_offset_ = momentum_offset;
			if (convergence_criterion != nullptr) {
				convergence_criterion_ = std::string(convergence_criterion);
				if (SUPPORTED_CONV_CRIT_.find(convergence_criterion_) == SUPPORTED_CONV_CRIT_.end()) {
					Log::REFatal("Convergence criterion '%s' is not supported.", convergence_criterion_.c_str());
				}
			}
			lr_coef_init_ = lr_coef;
			acc_rate_coef_ = acc_rate_coef;
			if (optimizer_coef != nullptr) {
				optimizer_coef_ = std::string(optimizer_coef);
				coef_optimizer_has_been_set_ = true;
			}
			num_rand_vec_trace_ = num_rand_vec_trace;
			seed_rand_vec_trace_ = seed_rand_vec_trace;
			reuse_rand_vec_trace_ = reuse_rand_vec_trace;
			// Conjugate gradient algorithm related parameters
			if (matrix_inversion_method_ == "iterative") {
				cg_max_num_it_ = cg_max_num_it;
				cg_max_num_it_tridiag_ = cg_max_num_it_tridiag;
				cg_delta_conv_ = cg_delta_conv;
				piv_chol_rank_ = piv_chol_rank;
				if (cg_preconditioner_type != nullptr) {
					if (cg_preconditioner_type_ != std::string(cg_preconditioner_type) &&
						model_has_been_estimated_) {
						Log::REFatal("Cannot change 'cg_preconditioner_type' after a model has been fitted ");
					}
					cg_preconditioner_type_ = std::string(cg_preconditioner_type);
					CheckPreconditionerType();
					cg_preconditioner_type_has_been_set_ = true;
				}
				SetMatrixInversionPropertiesLikelihood();
			}
			estimate_aux_pars_ = estimate_aux_pars;
			if (lr > 0) {
				lr_aux_pars_init_ = lr;
			}
			set_optim_config_has_been_called_ = true;
		}//end SetOptimConfig

		/*!
		* \brief Find covariance parameters and linear regression coefficients (if there are any) that minimize the (approximate) negative log-ligelihood
		*		 Note: You should pre-allocate memory for optim_cov_pars and optim_coef. Their length equal the number of covariance parameters and the number of regression coefficients
		*           If calc_std_dev, you also need to pre-allocate memory for std_dev_cov_par and std_dev_coef of the same length for the standard deviations
		* \param y_data Response variable data
		* \param covariate_data Covariate data (=independent variables, features). Set to nullptr if there is no covariate data
		* \param num_covariates Number of covariates
		* \param[out] optim_cov_pars Optimal covariance parameters
		* \param[out] optim_coef Optimal regression coefficients
		* \param[out] num_it Number of iterations
		* \param init_cov_pars Initial values for covariance parameters of RE components
		* \param init_coef Initial values for the regression coefficients (can be nullptr)
		* \param[out] std_dev_cov_par Standard deviations for the covariance parameters (can be nullptr, used only if calc_std_dev)
		* \param[out] std_dev_coef Standard deviations for the coefficients (can be nullptr, used only if calc_std_dev and if covariate_data is not nullptr)
		* \param calc_std_dev If true, asymptotic standard deviations for the MLE of the covariance parameters are calculated as the diagonal of the inverse Fisher information
		* \param fixed_effects Additional fixed effects that are added to the linear predictor (= offset) (can be nullptr)
		* \param learn_covariance_parameters If true, covariance parameters are estimated
		* \param called_in_GPBoost_algorithm If true, this function is called in the GPBoost algorithm, otherwise for the estimation of a GLMM
		* \param reuse_learning_rates_from_previous_call If true, the learning rates for the covariance and potential auxiliary parameters are kept at the values from the previous call and not re-initialized (can only be set to true if called_in_GPBoost_algorithm is true)
		*/
		void OptimLinRegrCoefCovPar(const double* y_data,
			const double* covariate_data,
			int num_covariates,
			double* optim_cov_pars,
			double* optim_coef,
			int& num_it,
			double* init_cov_pars,
			double* init_coef,
			double* std_dev_cov_par,
			double* std_dev_coef,
			bool calc_std_dev,
			const double* fixed_effects,
			bool learn_covariance_parameters,
			bool called_in_GPBoost_algorithm,
			bool reuse_learning_rates_from_previous_call) {
			if (reuse_learning_rates_from_previous_call) {
				CHECK(called_in_GPBoost_algorithm);
			}
			if (NumAuxPars() == 0) {
				estimate_aux_pars_ = false;
			}
			// Some checks
			if (SUPPORTED_OPTIM_COV_PAR_.find(optimizer_cov_pars_) == SUPPORTED_OPTIM_COV_PAR_.end()) {
				Log::REFatal("Optimizer option '%s' is not supported for covariance parameters ", optimizer_cov_pars_.c_str());
			}
			if (!gauss_likelihood_) {
				if (optimizer_cov_pars_ == "fisher_scoring") {
					Log::REFatal("Optimizer option '%s' is not supported for covariance parameters for non-Gaussian likelihoods ", optimizer_cov_pars_.c_str());
				}
			}
			if (optimizer_cov_pars_ == "fisher_scoring" && estimate_aux_pars_) {
				Log::REFatal("Optimizer option '%s' is not supported when estimating additional auxiliary parameters for non-Gaussian likelihoods ", optimizer_cov_pars_.c_str());
			}
			if (covariate_data != nullptr) {
				if (gauss_likelihood_) {
					if (SUPPORTED_OPTIM_COEF_GAUSS_.find(optimizer_coef_) == SUPPORTED_OPTIM_COEF_GAUSS_.end()) {
						Log::REFatal("Optimizer option '%s' is not supported for linear regression coefficients.", optimizer_coef_.c_str());
					}
				}
				else {
					if (SUPPORTED_OPTIM_COEF_NONGAUSS_.find(optimizer_coef_) == SUPPORTED_OPTIM_COEF_NONGAUSS_.end()) {
						Log::REFatal("Optimizer option '%s' is not supported for linear regression coefficients for non-Gaussian likelihoods ", optimizer_coef_.c_str());
					}
				}
			}
			// Check response variable data
			if (y_data != nullptr) {
				if (LightGBM::Common::HasNAOrInf(y_data, num_data_)) {
					Log::REFatal("NaN or Inf in response variable / label ");
				}
			}
			// Initialization of variables
			OptimParamsSetInitialValues();
			if (reuse_learning_rates_from_previous_call) {
				if (!lr_have_been_initialized_) {//do not reset learning rates to their initial values in every boosting iteration
					InitializeLearningRatesConvTol();
				}
			}
			else {
				InitializeLearningRatesConvTol();
			}
			if (covariate_data == nullptr) {
				has_covariates_ = false;
			}
			else {
				has_covariates_ = true;
			}
			bool use_nesterov_acc = use_nesterov_acc_;
			bool use_nesterov_acc_coef = use_nesterov_acc_;
			//Nesterov acceleration is only used for gradient descent and not for other methods
			if (optimizer_cov_pars_ != "gradient_descent") {
				use_nesterov_acc = false;
			}
			if (optimizer_coef_ != "gradient_descent") {
				use_nesterov_acc_coef = false;
			}
			if (OPTIM_EXTERNAL_.find(optimizer_cov_pars_) != OPTIM_EXTERNAL_.end()) {
				if (coef_optimizer_has_been_set_ && optimizer_coef_ != optimizer_cov_pars_) {
					Log::REWarning("'%s' is also used for estimating regression coefficients (optimizer_coef = '%s' is ignored)",
						optimizer_cov_pars_.c_str(), optimizer_coef_.c_str());
				}
				optimizer_coef_ = optimizer_cov_pars_;
			}
			bool terminate_optim = false;
			learning_rate_decreased_first_time_ = false;
			learning_rate_increased_after_descrease_ = false;
			learning_rate_decreased_after_increase_ = false;
			num_ll_evaluations_ = 0;
			num_iter_ = 0;
			num_it = max_iter_;
			profile_out_marginal_variance_ = gauss_likelihood_ &&
				(optimizer_cov_pars_ == "gradient_descent" || optimizer_cov_pars_ == "nelder_mead" || optimizer_cov_pars_ == "adam" || 
					optimizer_cov_pars_ == "bfgs_v3" || optimizer_cov_pars_ == "lbfgs_linesearch_nocedal_wright");
			if (optimizer_cov_pars_ == "bfgs_v3") {
				optimizer_cov_pars_ = "bfgs_v2";
			}
			// Profiling out sigma (=use closed-form expression for error / nugget variance) is better for gradient descent for Gaussian data 
			//	(the paremeters usually live on different scales and the nugget needs a small learning rate but the others not...)
			bool gradient_contains_error_var = gauss_likelihood_ && !profile_out_marginal_variance_;//If true, the error variance parameter (=nugget effect) is also included in the gradient, otherwise not
			has_intercept_ = false; //If true, the covariates contain an intercept column (only relevant if there are covariates)
			bool only_intercept_for_GPBoost_algo = false;//If true, the covariates contain only an intercept (only relevant if there are covariates)
			intercept_col_ = -1;
			// Check whether one of the columns contains only 1's and if not, make warning
			if (has_covariates_) {
				num_coef_ = num_covariates;
				X_ = Eigen::Map<const den_mat_t>(covariate_data, num_data_, num_coef_);
				for (int icol = 0; icol < num_coef_; ++icol) {
					if ((X_.col(icol).array() - 1.).abs().sum() < EPSILON_VECTORS) {
						has_intercept_ = true;
						intercept_col_ = icol;
						break;
					}
				}
				if (!has_intercept_) {
					Log::REWarning("The covariate data contains no column of ones, i.e., no intercept is included.");
				}
				only_intercept_for_GPBoost_algo = has_intercept_ && num_coef_ == 1 && !learn_covariance_parameters;
				if (only_intercept_for_GPBoost_algo) {
					CHECK(called_in_GPBoost_algorithm);
				}
				if (!only_intercept_for_GPBoost_algo) {
					Eigen::ColPivHouseholderQR<den_mat_t> qr_decomp(X_);
					int rank = (int)qr_decomp.rank();
					// If X_ was a sparse matrix, use the following code:
					//Eigen::SparseQR<sp_mat_t, Eigen::COLAMDOrdering<int>> qr_decomp;
					//qr_decomp.compute(X_);
					if (rank < num_coef_) {
						Log::REWarning("The linear regression covariate data matrix (fixed effect) is rank deficient. "
							"This is not necessarily a problem when using gradient descent. "
							"If this is not desired, consider dropping some columns / covariates.");
					}
				}
			}
			// Assume that this function is only called for initialization of the GPBoost algorithm
			//	when (i) there is only an intercept (and not other covariates) and (ii) the covariance parameters are not learned
			const double* fixed_effects_ptr = fixed_effects;
			// Initialization of covariance parameters related variables as well as additional parameters for likelihood (aux_pars)
			int num_cov_par_estimate = num_cov_par_;
			if (estimate_aux_pars_) {
				num_cov_par_estimate += NumAuxPars();
			}
			vec_t cov_aux_pars = vec_t(num_cov_par_estimate);
			for (int i = 0; i < num_cov_par_; ++i) {
				cov_aux_pars[i] = init_cov_pars[i];
			}
			if (estimate_aux_pars_) {
				// Find initial values for additional likelihood parameters (aux_pars) if they have not been given
				if (!(likelihood_[unique_clusters_[0]]->AuxParsHaveBeenSet())) {
					const double* aux_pars;
					if (y_data == nullptr) {
						vec_t y_aux_temp(num_data_);
						GetY(y_aux_temp.data());
						aux_pars = likelihood_[unique_clusters_[0]]->FindInitialAuxPars(y_aux_temp.data(), num_data_);
						y_aux_temp.resize(0);
					}
					else {
						aux_pars = likelihood_[unique_clusters_[0]]->FindInitialAuxPars(y_data, num_data_);
					}
					SetAuxPars(aux_pars);
				}
				for (int i = 0; i < NumAuxPars(); ++i) {
					cov_aux_pars[num_cov_par_ + i] = GetAuxPars()[i];
				}
			}
			vec_t cov_aux_pars_lag1 = vec_t(num_cov_par_estimate);
			vec_t cov_aux_pars_init = cov_aux_pars;
			vec_t cov_pars_after_grad_aux = cov_aux_pars, cov_aux_pars_after_grad_aux_lag1 = cov_aux_pars;//auxiliary variables used only if use_nesterov_acc == true
			vec_t cov_aux_pars_before_lr_coef_small, aux_pars_before_lr_cov_small, cov_pars_before_lr_aux_pars_small;//auxiliary variables
			// Set response variabla data (if needed). Note: for the GPBoost algorithm this is set a prior by calling SetY. For Gaussian data with covariates, this is set later repeatedly.
			if ((!has_covariates_ || !gauss_likelihood_) && y_data != nullptr) {
				SetY(y_data);
			}
			if (!has_covariates_ || !gauss_likelihood_) {
				CHECK(y_has_been_set_);//response variable data needs to have been set at this point for non-Gaussian likelihoods and for Gaussian data without covariates
			}
			if (gauss_likelihood_) {
				CHECK(y_data != nullptr);
				// Copy of response data (used only for Gaussian data and if there are also linear covariates since then y_ is modified during the optimization algorithm and this contains the original data)
				y_vec_ = Eigen::Map<const vec_t>(y_data, num_data_);
			}
			// Initialization of linear regression coefficients related variables
			vec_t beta, beta_lag1, beta_init, beta_after_grad_aux, beta_after_grad_aux_lag1, beta_before_lr_cov_small, beta_before_lr_aux_pars_small, fixed_effects_vec;
			scale_covariates_ = false;
			if (has_covariates_) {
				scale_covariates_ = (optimizer_coef_ == "gradient_descent" || (optimizer_cov_pars_ == "bfgs" && !gauss_likelihood_) ||
					optimizer_cov_pars_ == "bfgs_v2" || optimizer_cov_pars_ == "lbfgs_linesearch_nocedal_wright") && !only_intercept_for_GPBoost_algo;
				// Scale covariates (in order that the gradient is less sample-size dependent)
				if (scale_covariates_) {
					loc_transf_ = vec_t(num_coef_);
					scale_transf_ = vec_t(num_coef_);
					vec_t col_i_centered;
					for (int icol = 0; icol < num_coef_; ++icol) {
						if (!has_intercept_ || icol != intercept_col_) {
							loc_transf_[icol] = X_.col(icol).mean();
							col_i_centered = X_.col(icol);
							col_i_centered.array() -= loc_transf_[icol];
							scale_transf_[icol] = std::sqrt(col_i_centered.array().square().sum() / num_data_);
							X_.col(icol) = col_i_centered / scale_transf_[icol];
						}
					}
					if (has_intercept_) {
						loc_transf_[intercept_col_] = 0.;
						scale_transf_[intercept_col_] = 1.;
					}
				}
				beta = vec_t(num_coef_);
				if (init_coef == nullptr) {
					beta.setZero();
				}
				else {
					beta = Eigen::Map<const vec_t>(init_coef, num_covariates);
				}
				if (init_coef == nullptr || only_intercept_for_GPBoost_algo) {
					if (has_intercept_) {
						double tot_var = GetTotalVarComps(cov_aux_pars.segment(0, num_cov_par_));
						if (y_data == nullptr) {
							vec_t y_temp(num_data_);
							GetY(y_temp.data());
							beta[intercept_col_] = likelihood_[unique_clusters_[0]]->FindInitialIntercept(y_temp.data(), num_data_, tot_var, fixed_effects);
						}
						else {
							beta[intercept_col_] = likelihood_[unique_clusters_[0]]->FindInitialIntercept(y_data, num_data_, tot_var, fixed_effects);
						}
					}
				}
				else if (scale_covariates_) {
					// transform initial coefficients
					TransformCoef(beta, beta);
				}
				beta_after_grad_aux = beta;
				beta_after_grad_aux_lag1 = beta;
				beta_init = beta;
				UpdateFixedEffects(beta, fixed_effects, fixed_effects_vec);
				if (!gauss_likelihood_) {
					fixed_effects_ptr = fixed_effects_vec.data();
				}
				// Determine constants C_mu and C_sigma2 used for checking whether step sizes for linear regression coefficients are clearly too large
				if (y_data == nullptr) {
					vec_t y_temp(num_data_);
					GetY(y_temp.data());
					likelihood_[unique_clusters_[0]]->FindConstantsCapTooLargeLearningRateCoef(y_temp.data(), num_data_, fixed_effects, C_mu_, C_sigma2_);
				}
				else {
					likelihood_[unique_clusters_[0]]->FindConstantsCapTooLargeLearningRateCoef(y_data, num_data_, fixed_effects, C_mu_, C_sigma2_);
				}
			}//end if has_covariates_
			else if (!called_in_GPBoost_algorithm && fixed_effects == nullptr) {//!has_covariates_ && !called_in_GPBoost_algorithm && fixed_effects == nullptr
				CHECK(y_data != nullptr);
				double tot_var = GetTotalVarComps(cov_aux_pars.segment(0, num_cov_par_));
				if (likelihood_[unique_clusters_[0]]->ShouldHaveIntercept(y_data, num_data_, tot_var, fixed_effects)) {
					Log::REWarning("There is no intercept for modeling a possibly non-zero mean of the random effects. "
						"Consider including an intercept (= a column of 1's) in the covariates 'X' ");
				}
			}
			if (!has_covariates_ && fixed_effects != nullptr && gauss_likelihood_) {
				vec_t resid = y_vec_;
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_; ++i) {
					resid[i] -= fixed_effects[i];
				}
				SetY(resid.data());
			}
			Log::REDebug("GPModel: initial parameters: ");
			PrintTraceParameters(cov_aux_pars.segment(0, num_cov_par_), beta, cov_aux_pars.data() + num_cov_par_);
			// Initialize optimizer:
			// - factorize the covariance matrix (Gaussian data) or calculate the posterior mode of the random effects for use in the Laplace approximation (non-Gaussian likelihoods)
			// - calculate initial value of objective function
			// - Note: initial values of aux_pars (additional parameters of likelihood) are set in likelihoods.h
			if (ShouldRedetermineNearestNeighborsVecchia()) {
				SetCovParsComps(cov_aux_pars.segment(0, num_cov_par_));
				RedetermineNearestNeighborsVecchia();//called only if gp_approx == "vecchia" and neighbors are selected based on correlations and not distances
			}
			CalcCovFactorOrModeAndNegLL(cov_aux_pars.segment(0, num_cov_par_), fixed_effects_ptr);
			// TODO: for likelihood evaluation we don't need y_aux = Psi^-1 * y but only Psi^-0.5 * y. So, if has_covariates_==true, we might skip this step here and save some time
			string_t ll_str;
			if (gauss_likelihood_) {
				ll_str = "negative log-likelihood";
			}
			else {
				ll_str = "approximate negative marginal log-likelihood";
			}
			string_t init_coef_str = "";
			if (has_covariates_) {
				init_coef_str = " and 'init_coef'";
			}
			string_t problem_str = "none";
			if (std::isnan(neg_log_likelihood_)) {
				problem_str = "NaN";
			}
			else if (std::isinf(neg_log_likelihood_)) {
				problem_str = "Inf";
			}
			if (problem_str != "none") {
				Log::REFatal((problem_str + " occurred in initial " + ll_str + ". "
					"Possible solutions: try other initial values ('init_cov_pars'" + init_coef_str + ") "
					"or other tuning parameters in case you apply the GPBoost algorithm (e.g., learning_rate)").c_str());
			}
			if (gauss_likelihood_) {
				Log::REDebug("Initial negative log-likelihood: %g", neg_log_likelihood_);
			}
			else {
				Log::REDebug("Initial approximate negative marginal log-likelihood: %g", neg_log_likelihood_);
			}
			bool na_or_inf_occurred = false;
			if (OPTIM_EXTERNAL_.find(optimizer_cov_pars_) != OPTIM_EXTERNAL_.end()) {
				OptimExternal<T_mat, T_chol>(this, cov_aux_pars, beta, fixed_effects, max_iter_,
					delta_rel_conv_, convergence_criterion_, num_it, learn_covariance_parameters,
					optimizer_cov_pars_, profile_out_marginal_variance_,
					neg_log_likelihood_, num_cov_par_, NumAuxPars(), GetAuxPars(), has_covariates_);
				// Check for NA or Inf
				if (optimizer_cov_pars_ == "bfgs" || optimizer_cov_pars_ == "bfgs_v2" || optimizer_cov_pars_ == "lbfgs_linesearch_nocedal_wright") {
					if (learn_covariance_parameters) {
						for (int i = 0; i < (int)cov_aux_pars.size(); ++i) {
							if (std::isnan(cov_aux_pars[i]) || std::isinf(cov_aux_pars[i])) {
								na_or_inf_occurred = true;
							}
						}
					}
					if (has_covariates_ && !na_or_inf_occurred) {
						for (int i = 0; i < (int)beta.size(); ++i) {
							if (std::isnan(beta[i]) || std::isinf(beta[i])) {
								na_or_inf_occurred = true;
							}
						}
					}
				} // end check for NA or Inf
			} // end use of external optimizer
			else {
				// Start optimization with "gradient_descent" or "fisher_scoring"
				bool lr_cov_is_small = false, lr_aux_pars_is_small = false, lr_coef_is_small = false;
				for (num_iter_ = 0; num_iter_ < max_iter_; ++num_iter_) {
					if (reset_learning_rate_every_iteration_) {
						InitializeLearningRatesConvTol();//reset learning rates to their initial values
					}
					neg_log_likelihood_lag1_ = neg_log_likelihood_;
					cov_aux_pars_lag1 = cov_aux_pars;
					// Update linear regression coefficients using gradient descent or generalized least squares (the latter option only for Gaussian data)
					if (has_covariates_) {
						beta_lag1 = beta;
						if (optimizer_coef_ == "gradient_descent") {// one step of gradient descent
							vec_t grad_beta;
							// Calculate gradient for linear regression coefficients
							CalcGradLinCoef(cov_aux_pars[0], grad_beta, fixed_effects_ptr);
							AvoidTooLargeLearningRateCoef(beta, grad_beta);
							CalcDirDerivArmijoAndLearningRateConstChangeCoef(grad_beta, beta, beta_after_grad_aux, use_nesterov_acc_coef);
							// Update linear regression coefficients, do learning rate backtracking, and recalculate mode for Laplace approx. (only for non-Gaussian likelihoods)
							UpdateLinCoef(beta, grad_beta, cov_aux_pars[0], use_nesterov_acc_coef, num_iter_, beta_after_grad_aux, beta_after_grad_aux_lag1,
								acc_rate_coef_, nesterov_schedule_version_, momentum_offset_, fixed_effects, fixed_effects_vec);
							if (num_iter_ == 0) {
								lr_coef_after_first_iteration_ = lr_coef_;
								lr_is_small_threshold_coef_ = lr_coef_ / 1e4;
							}
							fixed_effects_ptr = fixed_effects_vec.data();
							// In case lr_coef_ is very small, we monitor whether cov_aux_pars continues to change. If it does, we will reset lr_coef_ to its initial value
							if (lr_coef_ < lr_is_small_threshold_coef_ && learn_covariance_parameters && !lr_coef_is_small) {
								if ((beta - beta_lag1).norm() < LR_IS_SMALL_REL_CHANGE_IN_PARS_THRESHOLD_ * beta_lag1.norm()) {//also require that relative change in parameters is small
									lr_coef_is_small = true;
									cov_aux_pars_before_lr_coef_small = cov_aux_pars;
								}
							}
						}
						else if (optimizer_coef_ == "wls") {// coordinate descent using generalized least squares (only for Gaussian data)
							CHECK(gauss_likelihood_);
							SetY(y_vec_.data());
							CalcYAux(1.);
							UpdateCoefGLS(X_, beta);
							// Set resid for updating covariance parameters
							UpdateFixedEffects(beta, fixed_effects, fixed_effects_vec);
							EvalNegLogLikelihoodOnlyUpdateFixedEffects(cov_aux_pars[0], neg_log_likelihood_after_lin_coef_update_);
						}
						// Reset lr_cov_ to its initial values in case beta changes substantially after lr_cov_ is very small
						bool mode_hast_just_been_recalculated = false;
						if (lr_cov_is_small && learn_covariance_parameters) {
							if ((beta - beta_before_lr_cov_small).norm() > MIN_REL_CHANGE_IN_OTHER_PARS_FOR_RESETTING_LR_ * beta_before_lr_cov_small.norm()) {
								lr_cov_ = lr_cov_after_first_iteration_;
								lr_cov_is_small = false;
								RecalculateModeLaplaceApprox(fixed_effects_ptr);
								mode_hast_just_been_recalculated = true;
							}
						}
						// Reset lr_aux_pars_ to its initial values in case beta changes substantially after lr_aux_pars_ is very small
						if (lr_aux_pars_is_small && estimate_aux_pars_ && learn_covariance_parameters) {
							if ((beta - beta_before_lr_cov_small).norm() > MIN_REL_CHANGE_IN_OTHER_PARS_FOR_RESETTING_LR_ * beta_before_lr_cov_small.norm()) {
								lr_aux_pars_ = lr_aux_pars_after_first_iteration_;
								lr_aux_pars_is_small = false;
								if (!mode_hast_just_been_recalculated) {
									RecalculateModeLaplaceApprox(fixed_effects_ptr);
								}
							}
						}
					}//end has_covariates_
					else {
						neg_log_likelihood_after_lin_coef_update_ = neg_log_likelihood_lag1_;
					}// end update regression coefficients
					// Update covariance parameters using one step of gradient descent or Fisher scoring
					if (learn_covariance_parameters) {
						// Calculate gradient or natural gradient = FI^-1 * grad (for Fisher scoring)
						vec_t grad, neg_step_dir; // gradient and negative step direction. E.g., neg_step_dir = grad for gradient descent and neg_step_dir = FI^-1 * grad for Fisher scoring (="natural" gradient)
						den_mat_t approx_Hessian;
						if (profile_out_marginal_variance_) {
							// Profile out sigma2 (=use closed-form expression for error / nugget variance) since this is better for gradient descent (the paremeters usually live on different scales and the nugget needs a small learning rate but the others not...)
							cov_aux_pars[0] = yTPsiInvy_ / num_data_;
						}
						if (optimizer_cov_pars_ == "gradient_descent" || optimizer_cov_pars_ == "newton") {
							CalcGradCovParAuxPars(cov_aux_pars.segment(0, num_cov_par_), grad, gradient_contains_error_var, false, fixed_effects_ptr);
							if (optimizer_cov_pars_ == "gradient_descent") {
								neg_step_dir = grad;
							}
							else if (optimizer_cov_pars_ == "newton") {
								CalcHessianCovParAuxPars(cov_aux_pars, gradient_contains_error_var, fixed_effects_ptr, approx_Hessian);
								neg_step_dir = approx_Hessian.llt().solve(grad);
							}
						}
						else if (optimizer_cov_pars_ == "fisher_scoring") {
							CHECK(gauss_likelihood_);
							// We don't profile out sigma2 since this seems better for Fisher scoring (less iterations)	
							CalcGradCovParAuxPars(cov_aux_pars.segment(0, num_cov_par_), grad, gradient_contains_error_var, true, fixed_effects_ptr);
							CalcFisherInformation(cov_aux_pars.segment(0, num_cov_par_), approx_Hessian, true, gradient_contains_error_var, true);
							neg_step_dir = approx_Hessian.llt().solve(grad);
						}
						if (optimizer_cov_pars_ == "gradient_descent") {
							AvoidTooLargeLearningRatesCovAuxPars(neg_step_dir);// Avoid too large learning rates for covariance parameters and aux_pars (for fisher_scoring and newton, this is done non-permanently in 'UpdateCovAuxPars')
						}
						CalcDirDerivArmijoAndLearningRateConstChangeCovAuxPars(grad, neg_step_dir, cov_aux_pars, cov_pars_after_grad_aux, use_nesterov_acc);
						if (reuse_learning_rates_from_previous_call && cov_pars_have_been_estimated_once_ &&
							optimizer_cov_pars_ == "gradient_descent") {//potentially increase learning rates again
							PotentiallyIncreaseLearningRatesForGPBoostAlgorithm();
						}//end called_in_GPBoost_algorithm / potentially increase learning rates again
						// Update covariance and additional likelihood parameters, do learning rate backtracking, factorize covariance matrix, and calculate new value of objective function
						UpdateCovAuxPars(cov_aux_pars, neg_step_dir, use_nesterov_acc, num_iter_,
							cov_pars_after_grad_aux, cov_aux_pars_after_grad_aux_lag1, acc_rate_cov_, nesterov_schedule_version_, momentum_offset_, fixed_effects_ptr);
						if (num_iter_ == 0) {
							lr_cov_after_first_iteration_ = lr_cov_;
							lr_is_small_threshold_cov_ = lr_cov_ / 1e4;
							if (estimate_aux_pars_) {
								lr_aux_pars_after_first_iteration_ = lr_aux_pars_;
								lr_is_small_threshold_aux_ = lr_aux_pars_ / 1e4;
							}
						}
						// In case lr_cov_ is very small, we monitor whether the other parameters (beta, aux_pars) continue to change. If yes, we will reset lr_cov_ to its initial value
						if (lr_cov_ < lr_is_small_threshold_cov_ && (has_covariates_ || estimate_aux_pars_) && !lr_cov_is_small) {
							if ((cov_aux_pars.segment(0, num_cov_par_) - cov_aux_pars_lag1.segment(0, num_cov_par_)).norm() <
								LR_IS_SMALL_REL_CHANGE_IN_PARS_THRESHOLD_ * cov_aux_pars_lag1.segment(0, num_cov_par_).norm()) {//also require that relative change in parameters is small
								lr_cov_is_small = true;
								if (has_covariates_) {
									beta_before_lr_cov_small = beta;
								}
								if (estimate_aux_pars_) {
									aux_pars_before_lr_cov_small = cov_aux_pars.segment(num_cov_par_, NumAuxPars());
								}
							}
						}
						// In case lr_aux_pars_ is very small, we monitor whether the other parameters (beta, covariance parameters) continue to change. If yes, we will reset lr_aux_pars_ to its initial value
						if (estimate_aux_pars_) {
							if (lr_aux_pars_ < lr_is_small_threshold_aux_ && !lr_cov_is_small) {
								if ((cov_aux_pars.segment(num_cov_par_, NumAuxPars()) - cov_aux_pars_lag1.segment(num_cov_par_, NumAuxPars())).norm() <
									LR_IS_SMALL_REL_CHANGE_IN_PARS_THRESHOLD_ * cov_aux_pars_lag1.segment(num_cov_par_, NumAuxPars()).norm()) {//also require that relative change in parameters is small
									lr_aux_pars_is_small = true;
									if (has_covariates_) {
										beta_before_lr_aux_pars_small = beta;
									}
									cov_pars_before_lr_aux_pars_small = cov_aux_pars.segment(0, num_cov_par_);
								}
							}
						}
						// Reset lr_coef_ to its initial value in case cov_aux_pars changes substantially after lr_coef_ is very small
						bool mode_hast_just_been_recalculated = false;
						if (lr_coef_is_small && has_covariates_) {
							if ((cov_aux_pars - cov_aux_pars_before_lr_coef_small).norm() > MIN_REL_CHANGE_IN_OTHER_PARS_FOR_RESETTING_LR_ * cov_aux_pars_before_lr_coef_small.norm()) {
								lr_coef_ = lr_coef_after_first_iteration_;
								lr_coef_is_small = false;
								RecalculateModeLaplaceApprox(fixed_effects_ptr);
								mode_hast_just_been_recalculated = true;
							}
						}
						// Reset lr_aux_pars_ to its initial values in case covariance paremeters change substantially after lr_aux_pars_ is very small
						if (lr_aux_pars_is_small && estimate_aux_pars_) {
							if ((cov_aux_pars.segment(0, num_cov_par_) - cov_pars_before_lr_aux_pars_small).norm() > MIN_REL_CHANGE_IN_OTHER_PARS_FOR_RESETTING_LR_ * cov_pars_before_lr_aux_pars_small.norm()) {
								lr_aux_pars_ = lr_aux_pars_after_first_iteration_;
								lr_aux_pars_is_small = false;
								if (!mode_hast_just_been_recalculated) {
									RecalculateModeLaplaceApprox(fixed_effects_ptr);
									mode_hast_just_been_recalculated = true;
								}
							}
						}
						// Reset lr_cov_ to its initial values in case aux_pars changes substantially after lr_cov_ is very small
						if (lr_cov_is_small && estimate_aux_pars_) {
							if ((cov_aux_pars.segment(num_cov_par_, NumAuxPars()) - aux_pars_before_lr_cov_small).norm() > MIN_REL_CHANGE_IN_OTHER_PARS_FOR_RESETTING_LR_ * aux_pars_before_lr_cov_small.norm()) {
								lr_cov_ = lr_cov_after_first_iteration_;
								lr_cov_is_small = false;
								if (!mode_hast_just_been_recalculated) {
									RecalculateModeLaplaceApprox(fixed_effects_ptr);
								}
							}
						}
					}//end if (learn_covariance_parameters)
					else {
						neg_log_likelihood_ = neg_log_likelihood_after_lin_coef_update_;
					}// end update covariance parameters
					// Check for NA or Inf
					if (std::isnan(neg_log_likelihood_) || std::isinf(neg_log_likelihood_)) {
						na_or_inf_occurred = true;
						terminate_optim = true;
					}
					else {
						if (learn_covariance_parameters) {
							for (int i = 0; i < (int)cov_aux_pars.size(); ++i) {
								if (std::isnan(cov_aux_pars[i]) || std::isinf(cov_aux_pars[i])) {
									na_or_inf_occurred = true;
									terminate_optim = true;
								}
							}
						}
					}
					if (!na_or_inf_occurred) {
						// Check convergence
						if (convergence_criterion_ == "relative_change_in_parameters") {
							if (has_covariates_) {
								if (((beta - beta_lag1).norm() < delta_rel_conv_ * beta_lag1.norm()) && ((cov_aux_pars - cov_aux_pars_lag1).norm() < delta_rel_conv_ * cov_aux_pars_lag1.norm())) {
									terminate_optim = true;
								}
							}
							else {
								if ((cov_aux_pars - cov_aux_pars_lag1).norm() < delta_rel_conv_ * cov_aux_pars_lag1.norm()) {
									terminate_optim = true;
								}
							}
						}
						else if (convergence_criterion_ == "relative_change_in_log_likelihood") {
							if ((neg_log_likelihood_lag1_ - neg_log_likelihood_) < delta_rel_conv_ * std::max(std::abs(neg_log_likelihood_lag1_), 1.)) {
								terminate_optim = true;
							}
						} // end check convergence
						if (learn_covariance_parameters && ShouldRedetermineNearestNeighborsVecchia()) {
							RedetermineNearestNeighborsVecchia();//called only in certain iterations if gp_approx == "vecchia" and neighbors are selected based on correlations and not distances
							CalcCovFactorOrModeAndNegLL(cov_aux_pars.segment(0, num_cov_par_), fixed_effects_ptr);//recalculate log-likelihood
						}
						// Trace output for convergence monitoring
						if ((num_iter_ < 10 || ((num_iter_ + 1) % 10 == 0 && (num_iter_ + 1) < 100) || ((num_iter_ + 1) % 100 == 0 && (num_iter_ + 1) < 1000) ||
							((num_iter_ + 1) % 1000 == 0 && (num_iter_ + 1) < 10000) || ((num_iter_ + 1) % 10000 == 0)) && (num_iter_ != (max_iter_ - 1)) && !terminate_optim) {
							Log::REDebug("GPModel: parameters after optimization iteration number %d: ", num_iter_ + 1);
							PrintTraceParameters(cov_aux_pars.segment(0, num_cov_par_), beta, cov_aux_pars.data() + num_cov_par_);
							if (gauss_likelihood_) {
								Log::REDebug("Negative log-likelihood: %g", neg_log_likelihood_);
							}
							else {
								Log::REDebug("Approximate negative marginal log-likelihood: %g", neg_log_likelihood_);
							}
						} // end trace output
					}// end not na_or_inf_occurred
					// Check whether to terminate
					if (terminate_optim) {
						num_it = num_iter_ + 1;
						break;
					}
					//increase learning rates again
					else if (increase_learning_rate_again_ && optimizer_cov_pars_ == "gradient_descent" && (num_iter_ + 1) >= 10 &&
						learning_rate_decreased_first_time_ && !learning_rate_decreased_after_increase_ && !na_or_inf_occurred) {
						if ((neg_log_likelihood_lag1_ - neg_log_likelihood_) < INCREASE_LR_CHANGE_LL_THRESHOLD_ * std::max(std::abs(neg_log_likelihood_lag1_), 1.)) {
							if (has_covariates_) {
								lr_coef_ /= LR_SHRINKAGE_FACTOR_;
							}
							if (learn_covariance_parameters) {
								lr_cov_ /= LR_SHRINKAGE_FACTOR_;
							}
							if (estimate_aux_pars_) {
								lr_aux_pars_ /= LR_SHRINKAGE_FACTOR_;
							}
							learning_rate_increased_after_descrease_ = true;
							Log::REDebug("GPModel covariance parameter estimation: The learning rates have been increased in iteration number %d ", num_iter_ + 1);
						}
					}
				}//end for loop for optimization
			}//end "gradient_descent" or "fisher_scoring"
			// redo optimization with "nelder_mead" in case NA or Inf occurred
			if (na_or_inf_occurred && optimizer_cov_pars_ != "nelder_mead") {
				string_t optimizers = "";
				for (auto elem : SUPPORTED_OPTIM_COV_PAR_) {
					if (gauss_likelihood_ || elem != "fisher_scoring") {
						optimizers += " '" + elem + "'";
					}
				}
				Log::REWarning("NaN or Inf occurred in covariance parameter optimization using '%s'. "
					"The optimization will be started a second time using 'nelder_mead'. "
					"If you want to avoid this, try directly using a different optimizer. "
					"If you have used 'gradient_descent', you can also consider using a smaller learning rate ", optimizer_cov_pars_.c_str());
				cov_aux_pars = cov_aux_pars_init;
				if (has_covariates_) {
					beta = beta_init;
				}
				if (!gauss_likelihood_) { // reset the initial modes to 0
					for (const auto& cluster_i : unique_clusters_) {
						likelihood_[cluster_i]->InitializeModeAvec();
					}
				}
				delta_rel_conv_ = delta_rel_conv_init_;
				OptimExternal<T_mat, T_chol>(this, cov_aux_pars, beta, fixed_effects, max_iter_,
					delta_rel_conv_, convergence_criterion_, num_it,
					learn_covariance_parameters, "nelder_mead", profile_out_marginal_variance_,
					neg_log_likelihood_, num_cov_par_, NumAuxPars(), GetAuxPars(), has_covariates_);
			}
			if (num_it == max_iter_) {
				Log::REDebug("GPModel: no convergence after the maximal number of iterations "
					"(%d, nb. likelihood evaluations = %d) ", max_iter_, num_ll_evaluations_);
			}
			else {
				Log::REDebug("GPModel: parameter estimation finished after %d iteration "
					"(nb. likelihood evaluations = %d) ", num_it, num_ll_evaluations_);
			}
			PrintTraceParameters(cov_aux_pars.segment(0, num_cov_par_), beta, cov_aux_pars.data() + num_cov_par_);
			if (gauss_likelihood_) {
				Log::REDebug("Negative log-likelihood: %g", neg_log_likelihood_);
			}
			else {
				Log::REDebug("Approximate negative marginal log-likelihood: %g", neg_log_likelihood_);
			}
			for (int i = 0; i < num_cov_par_; ++i) {
				optim_cov_pars[i] = cov_aux_pars[i];
			}
			if (estimate_aux_pars_) {
				SetAuxPars(cov_aux_pars.data() + num_cov_par_);
			}
			if (has_covariates_) {
				if (scale_covariates_) {
					//// transform coefficients back to original scale
					TransformBackCoef(beta, beta);
					//transform covariates back
					for (int icol = 0; icol < num_coef_; ++icol) {
						if (!has_intercept_ || icol != intercept_col_) {
							X_.col(icol).array() *= scale_transf_[icol];
							X_.col(icol).array() += loc_transf_[icol];
						}
					}
					if (has_intercept_) {
						X_.col(intercept_col_).array() = 1.;
					}
				}
				for (int i = 0; i < num_covariates; ++i) {
					optim_coef[i] = beta[i];
				}
			}
			if (calc_std_dev) {
				vec_t std_dev_cov(num_cov_par_);
				if (gauss_likelihood_) {
					CalcStdDevCovPar(cov_aux_pars.segment(0, num_cov_par_), std_dev_cov);//TODO: maybe another call to CalcCovFactor can be avoided in CalcStdDevCovPar (need to take care of cov_aux_pars[0])
					for (int i = 0; i < num_cov_par_; ++i) {
						std_dev_cov_par[i] = std_dev_cov[i];
					}
				}
				else {
					std_dev_cov.setZero();// Calculation of standard deviations for covariance parameters is not supported for non-Gaussian likelihoods
					if (!has_covariates_) {
						Log::REWarning("Calculation of standard deviations of covariance parameters for non-Gaussian likelihoods is currently not supported.");
					}
				}
				if (has_covariates_) {
					vec_t std_dev_beta(num_covariates);
					if (gauss_likelihood_) {
						CalcStdDevCoef(cov_aux_pars.segment(0, num_cov_par_), X_, std_dev_beta);
					}
					else {
						Log::REDebug("Standard deviations of linear regression coefficients for non-Gaussian likelihoods can be \"very approximative\". ");
						CalcStdDevCoefNonGaussian(num_covariates, beta, cov_aux_pars.segment(0, num_cov_par_), fixed_effects, std_dev_beta);
					}
					for (int i = 0; i < num_covariates; ++i) {
						std_dev_coef[i] = std_dev_beta[i];
					}
				}
			}
			if (has_covariates_) {
				if (only_intercept_for_GPBoost_algo) {
					has_covariates_ = false;
					// When this function is only called for initialization of the GPBoost algorithm, 
					//	we set has_covariates_ to false in order to avoid potential problems when making predictions with the GPBoostOOS algorithm,
					//	since in the second phase of the GPBoostOOS algorithm covariance parameters are not estimated (and thus has_covariates_ is not set to false)
					//	but this function is called for initialization of the GPBoost algorithm.
				}
			}
			model_has_been_estimated_ = true;
			if (learn_covariance_parameters) {
				cov_pars_have_been_estimated_once_ = true;
			}
		}//end OptimLinRegrCoefCovPar

		/*!
		* \brief Calculate gradient wrt the covariance parameters on the log-scale and any additional parameters for the likelihood for non-Gaussian likelihoods
		*	Call 'CalcCovFactorOrModeAndNegLL' first since this function assumes that the covariance matrix has been factorized (by 'CalcCovFactor') and
		*		that y_aux or y_tilde/y_tilde2 (if only_grouped_REs_use_woodbury_identity_) have been calculated (by 'CalcYAux' or 'CalcYtilde')
		* \param cov_pars Covariance parameters
		* \param[out] grad_cov_aux_par Gradient wrt the covariance parameters and any additional parameters for the likelihood for non-Gaussian likelihoods
		* \param include_error_var If true, the gradient with respect to the error variance parameter (=nugget effect) is also calculated, otherwise not (set this to true if the nugget effect is not calculated by using the closed-form solution)
		* \param save_psi_inv_for_FI If true, the inverse covariance matrix Psi^-1 is saved for reuse later (e.g. when calculating the Fisher information in Fisher scoring). This option is ignored if the Vecchia approximation is used.
		* \param fixed_effects Fixed effects component of location parameter (used only for non-Gaussian likelihoods)
		*/
		void CalcGradCovParAuxPars(const vec_t& cov_pars,
			vec_t& grad_cov_aux_par,
			bool include_error_var,
			bool save_psi_inv_for_FI,
			const double* fixed_effects) {
			if (gauss_likelihood_) {//Gaussian data
				if (include_error_var) {
					grad_cov_aux_par = vec_t::Zero(num_cov_par_);
				}
				else {
					grad_cov_aux_par = vec_t::Zero(num_cov_par_ - 1);
				}
				int first_cov_par = include_error_var ? 1 : 0;
				for (const auto& cluster_i : unique_clusters_) {
					if (gp_approx_ == "vecchia") {//Vechia approximation
						vec_t u(num_data_per_cluster_[cluster_i]);
						vec_t uk(num_data_per_cluster_[cluster_i]);
						if (include_error_var) {
							u = B_[cluster_i] * y_[cluster_i];
							grad_cov_aux_par[0] += -1. * ((double)(u.transpose() * D_inv_[cluster_i] * u)) / cov_pars[0] / 2. + num_data_per_cluster_[cluster_i] / 2.;
							u = D_inv_[cluster_i] * u;
						}
						else {
							u = D_inv_[cluster_i] * B_[cluster_i] * y_[cluster_i];//TODO: this is already calculated in CalcYAux -> save it there and re-use here?
						}
						for (int j = 0; j < num_comps_total_; ++j) {
							int num_par_comp = re_comps_[cluster_i][j]->num_cov_par_;
							for (int ipar = 0; ipar < num_par_comp; ++ipar) {
								uk = B_grad_[cluster_i][num_par_comp * j + ipar] * y_[cluster_i];
								grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] += ((uk.dot(u) - 0.5 * u.dot(D_grad_[cluster_i][num_par_comp * j + ipar] * u)) / cov_pars[0] +
									0.5 * (D_inv_[cluster_i].diagonal()).dot(D_grad_[cluster_i][num_par_comp * j + ipar].diagonal()));
							}
						}
					}//end gp_approx_ == "vecchia"
					else {//not gp_approx_ == "vecchia"
						if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
							if (include_error_var) {
								grad_cov_aux_par[0] += -1. * ((double)(y_[cluster_i].transpose() * y_aux_[cluster_i])) / cov_pars[0] / 2. + num_data_per_cluster_[cluster_i] / 2.;
							}
							for (int j = 0; j < num_comps_total_; ++j) {
								int num_par_comp = re_comps_ip_[cluster_i][j]->num_cov_par_;
								// sigma_cross_cov
								std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov_[cluster_i][j]->GetZSigmaZt();
								// sigma_ip^-1 * sigma_cross_cov * sigma^-1 * y
								vec_t sigma_ip_inv_cross_cov_y_aux = chol_fact_sigma_ip_[cluster_i].solve((*cross_cov).transpose() * y_aux_[cluster_i]);
								// Initialize Matrices
								den_mat_t sigma_resid_inv_cross_cov_T;
								std::shared_ptr<T_mat> sigma_resid;
								den_mat_t rand_vec_probe_P_inv;
								if (matrix_inversion_method_ == "cholesky" && gp_approx_ == "full_scale_tapering") {
									// sigma_resid^-1 * t(cross_cov)
									sigma_resid_inv_cross_cov_T = chol_fact_resid_[cluster_i].solve((*cross_cov));
									sigma_resid = re_comps_resid_[cluster_i][j]->GetZSigmaZt();
									// sigma_resid^-1 with sparsity pattern of sigma_resid
									T_mat Identity(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
									Identity.setIdentity();
									T_mat chol_fact_resid_inv;
									TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_resid_[cluster_i], Identity, chol_fact_resid_inv, false);
									CalcLtLGivenSparsityPattern<T_mat>(chol_fact_resid_inv, (*sigma_resid), true);
									chol_fact_resid_inv.resize(0, 0);
								}
								else if (matrix_inversion_method_ == "iterative") {
									if (gp_approx_ == "fitc") {
										Log::REFatal("The iterative methods are not implemented for Predictive Processes. Please use Cholesky.");
									}
									// P^-1 * sample vectors
									if (cg_preconditioner_type_ == "predictive_process_plus_diagonal") {
										den_mat_t diag_sigma_resid_inv_Z = diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal() * rand_vec_probe_[cluster_i];
										rand_vec_probe_P_inv = diag_sigma_resid_inv_Z - (diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal() * ((*cross_cov) * chol_fact_woodbury_preconditioner_[cluster_i].solve((*cross_cov).transpose() * diag_sigma_resid_inv_Z)));
									}
									else {
										rand_vec_probe_P_inv = rand_vec_probe_[cluster_i];
									}
								}
								for (int ipar = 0; ipar < num_par_comp; ++ipar) {
									// Derivative of Components
									std::shared_ptr<den_mat_t> cross_cov_grad = re_comps_cross_cov_[cluster_i][j]->GetZSigmaZtGrad(ipar, true, 0.);
									den_mat_t sigma_ip_stable_grad = *(re_comps_ip_[cluster_i][j]->GetZSigmaZtGrad(ipar, true, 0.));

									// Trace of sigma_ip^-1 * sigma_ip_grad
									if (matrix_inversion_method_ == "cholesky") {
										den_mat_t sigma_ip_inv_sigma_ip_stable_grad;
										sigma_ip_inv_sigma_ip_stable_grad = chol_fact_sigma_ip_[cluster_i].solve(sigma_ip_stable_grad);
										grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] -= 0.5 * sigma_ip_inv_sigma_ip_stable_grad.trace();
									}

									grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] += ((0.5 * sigma_ip_inv_cross_cov_y_aux.dot((sigma_ip_stable_grad)*sigma_ip_inv_cross_cov_y_aux)
										- (((*cross_cov_grad).transpose()) * y_aux_[cluster_i]).dot(sigma_ip_inv_cross_cov_y_aux)) / cov_pars[0]);

									// sigma_woodbury_grad
									den_mat_t sigma_woodbury_grad;
									den_mat_t cross_cov_grad_sigma_resid_inv_cross_cov_T;

									if (gp_approx_ == "full_scale_tapering") {

										// Initialize Residual Process
										re_comps_resid_[cluster_i][j]->CalcSigma();
										std::shared_ptr<T_mat> sigma_resid_grad = re_comps_resid_[cluster_i][j]->GetZSigmaZtGrad(ipar, true, 1.);
										// sigma_ip^-1 * sigma_cross_cov
										den_mat_t sigma_ip_inv_sigma_cross_cov = chol_fact_sigma_ip_[cluster_i].solve((*cross_cov).transpose());
										// Subtract gradient of predictive process covariance
										SubtractProdFromMat<T_mat>(*sigma_resid_grad, -sigma_ip_inv_sigma_cross_cov, sigma_ip_stable_grad * sigma_ip_inv_sigma_cross_cov, true);
										SubtractProdFromMat<T_mat>(*sigma_resid_grad, (*cross_cov_grad).transpose(), sigma_ip_inv_sigma_cross_cov, false);
										SubtractProdFromMat<T_mat>(*sigma_resid_grad, sigma_ip_inv_sigma_cross_cov, (*cross_cov_grad).transpose(), false);

										// Apply taper
										re_comps_resid_[cluster_i][j]->ApplyTaper(*(re_comps_resid_[cluster_i][j]->dist_), *sigma_resid_grad);
										if (matrix_inversion_method_ == "cholesky") {
											// cross_crov_grad *  sigma_resid^-1 * t(cross_cov)
											cross_cov_grad_sigma_resid_inv_cross_cov_T = ((*cross_cov_grad).transpose()) * sigma_resid_inv_cross_cov_T;
											// cross_crov_grad *  sigma_resid^-1 * t(cross_cov) + cross_crov *  sigma_resid^-1 * t(cross_cov_grad) - cross_cov * sigma_resid^-1 * sigma_resid_grad * sigma_resid^-1 * t(cross_cov)
											sigma_woodbury_grad = cross_cov_grad_sigma_resid_inv_cross_cov_T + cross_cov_grad_sigma_resid_inv_cross_cov_T.transpose() -
												sigma_resid_inv_cross_cov_T.transpose() * ((*sigma_resid_grad) * sigma_resid_inv_cross_cov_T);

											grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] += 0.5 * ((double)(((*sigma_resid).cwiseProduct(*sigma_resid_grad)).sum()));
										}
										else {// Conjugate Gradient
											// Derivative of Woodbury preconditioner matrix (Cm + Cmn * diag(Cs)^-1 * Cnm) or (Lambda + t(EVects of Cm) * Cmn * diag(Cs)^-1 * Cnm * EVects of Cm)
											den_mat_t sigma_woodbury_preconditioner_grad;
											vec_t diagonal_approx_grad_preconditioner;
											if (cg_preconditioner_type_ == "predictive_process_plus_diagonal") {
												diagonal_approx_grad_preconditioner = (*sigma_resid_grad).diagonal();
												den_mat_t sigma_cross_cov_diag_sigma_resid_inv = (*cross_cov).transpose() * diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal();
												den_mat_t cross_cov_grad_d_inv_cross_cov = (*cross_cov_grad).transpose() * (diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal() * (*cross_cov));
												sigma_woodbury_preconditioner_grad = sigma_ip_stable_grad + cross_cov_grad_d_inv_cross_cov + cross_cov_grad_d_inv_cross_cov.transpose() - sigma_cross_cov_diag_sigma_resid_inv * ((diagonal_approx_grad_preconditioner.asDiagonal()) * sigma_cross_cov_diag_sigma_resid_inv.transpose());
											}
											den_mat_t sigma_ip_inv_sigma_ip_stable_grad;
											sigma_ip_inv_sigma_ip_stable_grad = chol_fact_sigma_ip_[cluster_i].solve(sigma_ip_stable_grad);
											// (Derivative of Sigma) * P^-1 * sample vectors
											den_mat_t sigma_resid_grad_Z(num_data_per_cluster_[cluster_i], rand_vec_probe_P_inv.cols());
											sigma_resid_grad_Z.setZero();
#pragma omp parallel for schedule(static)   
											for (int i = 0; i < rand_vec_probe_P_inv.cols(); ++i) {
												sigma_resid_grad_Z.col(i) += (*sigma_resid_grad) * rand_vec_probe_P_inv.col(i); //parallelization in for loop is much faster
											}
											den_mat_t sigma_ip_inv_sigma_cross_cov_Z = sigma_ip_inv_sigma_cross_cov * rand_vec_probe_P_inv;
											den_mat_t sigma_grad_Z = sigma_resid_grad_Z + (*cross_cov_grad) * sigma_ip_inv_sigma_cross_cov_Z +
												sigma_ip_inv_sigma_cross_cov.transpose() * ((*cross_cov_grad).transpose() * rand_vec_probe_P_inv) - sigma_ip_inv_sigma_cross_cov.transpose() * (sigma_ip_stable_grad * sigma_ip_inv_sigma_cross_cov_Z);

											// Stochastic Trace
											vec_t sample_Sigma = (solution_for_trace_[cluster_i].cwiseProduct(sigma_grad_Z)).colwise().sum();
											double stochastic_tr = sample_Sigma.mean();

											// Variance Reduction 
											if (cg_preconditioner_type_ == "predictive_process_plus_diagonal") {

												den_mat_t sigma_cross_cov_diag_sigma_resid_inv = (*cross_cov).transpose() * diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal();
												den_mat_t cross_cov_grad_d_inv_cross_cov = (*cross_cov_grad).transpose() * (diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal() * (*cross_cov));
												// (Derivative of P) * P^-1 * sample vectors
												den_mat_t P_G_Z = sigma_grad_Z + (diagonal_approx_grad_preconditioner.asDiagonal()) * rand_vec_probe_P_inv - sigma_resid_grad_Z;
												// Stochastic Trace (Preconditioner)
												vec_t sample_P = (rand_vec_probe_P_inv.cwiseProduct(P_G_Z)).colwise().sum();
												double Tr_P_stoch = sample_P.mean();

												// Exact Trace (Preconditioner)
												double Tr_P = 0;
												Tr_P -= sigma_ip_inv_sigma_ip_stable_grad.trace();
												Tr_P += diagonal_approx_inv_preconditioner_[cluster_i].dot(diagonal_approx_grad_preconditioner);
												den_mat_t sigma_woodbury_inv_sigma_woodbury_stable_grad = chol_fact_woodbury_preconditioner_[cluster_i].solve(sigma_woodbury_preconditioner_grad);
												Tr_P += sigma_woodbury_inv_sigma_woodbury_stable_grad.trace();
												// Calculate optimal c
												double c_opt;
												CalcOptimalC(sample_Sigma, sample_P, stochastic_tr, Tr_P, c_opt);
												// Reduce variance
												stochastic_tr += c_opt * (Tr_P - Tr_P_stoch);
											}
											grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] += 0.5 * stochastic_tr;
										}
										grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] -= 0.5 * y_aux_[cluster_i].dot((*sigma_resid_grad) * y_aux_[cluster_i]) / cov_pars[0];
									}
									else { // fitc
										// Derivative of diagonal part
										vec_t FITC_Diag_grad = vec_t::Zero(num_data_per_cluster_[cluster_i]);
										FITC_Diag_grad = FITC_Diag_grad.array() + sigma_ip_stable_grad.coeffRef(0, 0);
										den_mat_t sigma_ip_inv_sigma_cross_cov = chol_fact_sigma_ip_[cluster_i].solve((*cross_cov).transpose());
										den_mat_t sigma_ip_grad_inv_sigma_cross_cov = sigma_ip_stable_grad * sigma_ip_inv_sigma_cross_cov;
#pragma omp parallel for schedule(static)
										for (int ii = 0; ii < num_data_per_cluster_[cluster_i]; ++ii) {
											FITC_Diag_grad[ii] -= 2 * sigma_ip_inv_sigma_cross_cov.col(ii).dot((*cross_cov_grad).transpose().col(ii))
												- sigma_ip_inv_sigma_cross_cov.col(ii).dot(sigma_ip_grad_inv_sigma_cross_cov.col(ii));
										}
										grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] -= 0.5 * y_aux_[cluster_i].dot(FITC_Diag_grad.asDiagonal() * y_aux_[cluster_i]) / cov_pars[0];
										grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] += 0.5 * FITC_Diag_grad.dot(FITC_Diag_[cluster_i].cwiseInverse());

										// Derivative of Woodbury Matrix
										den_mat_t cross_cov_FITC_inv = (*cross_cov).transpose() * FITC_Diag_[cluster_i].cwiseInverse().asDiagonal();
										cross_cov_grad_sigma_resid_inv_cross_cov_T = cross_cov_FITC_inv * (*cross_cov_grad);
										sigma_woodbury_grad = cross_cov_grad_sigma_resid_inv_cross_cov_T + cross_cov_grad_sigma_resid_inv_cross_cov_T.transpose();
										sigma_woodbury_grad -= (cross_cov_FITC_inv * FITC_Diag_grad.asDiagonal()) * cross_cov_FITC_inv.transpose();

									}
									// sigma_woodbury^-1 * sigma_woodbury_grad
									if (matrix_inversion_method_ == "cholesky") {
										sigma_woodbury_grad += (sigma_ip_stable_grad);
										den_mat_t sigma_woodbury_inv_sigma_woodbury_grad = chol_fact_sigma_woodbury_[cluster_i].solve(sigma_woodbury_grad);
										grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] += 0.5 * ((sigma_woodbury_inv_sigma_woodbury_grad.trace()));
									}
								}
							}
						}
						else {// no GP approximation
							if (only_grouped_REs_use_woodbury_identity_) {
								if (include_error_var) {
									double yTPsiInvy;
									CalcYTPsiIInvY(yTPsiInvy, false, cluster_i, true, true);
									grad_cov_aux_par[0] += -1. * yTPsiInvy / cov_pars[0] / 2. + num_data_per_cluster_[cluster_i] / 2.;
								}
								std::vector<T_mat> LInvZtZj_cluster_i;
								if (save_psi_inv_for_FI) {
									LInvZtZj_[cluster_i].clear();
									LInvZtZj_cluster_i = std::vector<T_mat>(num_comps_total_);
								}
								for (int j = 0; j < num_comps_total_; ++j) {
									sp_mat_t* Z_j = re_comps_[cluster_i][j]->GetZ();
									vec_t y_tilde_j = (*Z_j).transpose() * y_[cluster_i];
									vec_t y_tilde2_j = (*Z_j).transpose() * y_tilde2_[cluster_i];
									double yTPsiIGradPsiPsiIy = y_tilde_j.transpose() * y_tilde_j - 2. * (double)(y_tilde_j.transpose() * y_tilde2_j) + y_tilde2_j.transpose() * y_tilde2_j;
									yTPsiIGradPsiPsiIy *= cov_pars[j + 1];
									T_mat LInvZtZj;
									if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ == ZtZj_ and L_inv are diagonal  
										LInvZtZj = ZtZ_[cluster_i];
										LInvZtZj.diagonal().array() /= sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array();
									}
									else {
										// Note: the following is often the bottleneck (= slower than Cholesky dec.) when there are multiple REs and the number of random effects is large
										if (CholeskyHasPermutation<T_chol>(chol_facts_[cluster_i])) {
											TriangularSolve<T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), P_ZtZj_[cluster_i][j], LInvZtZj, false);
										}
										else {
											TriangularSolve<T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), ZtZj_[cluster_i][j], LInvZtZj, false);
										}
									}
									if (save_psi_inv_for_FI) {//save for latter use when calculating the Fisher information
										LInvZtZj_cluster_i[j] = LInvZtZj;
									}
									double trace_PsiInvGradPsi = Zj_square_sum_[cluster_i][j] - LInvZtZj.squaredNorm();
									trace_PsiInvGradPsi *= cov_pars[j + 1];
									grad_cov_aux_par[first_cov_par + j] += -1. * yTPsiIGradPsiPsiIy / cov_pars[0] / 2. + trace_PsiInvGradPsi / 2.;
								}
								if (save_psi_inv_for_FI) {
									LInvZtZj_[cluster_i] = LInvZtZj_cluster_i;
								}
							}//end only_grouped_REs_use_woodbury_identity_
							else {//not only_grouped_REs_use_woodbury_identity_
								T_mat psi_inv;
								CalcPsiInv(psi_inv, cluster_i, !save_psi_inv_for_FI);
								if (save_psi_inv_for_FI) {//save for latter use when calculating the Fisher information
									psi_inv_[cluster_i] = psi_inv;
								}
								if (include_error_var) {
									grad_cov_aux_par[0] += -1. * ((double)(y_[cluster_i].transpose() * y_aux_[cluster_i])) / cov_pars[0] / 2. + num_data_per_cluster_[cluster_i] / 2.;
								}
								for (int j = 0; j < num_comps_total_; ++j) {
									for (int ipar = 0; ipar < re_comps_[cluster_i][j]->num_cov_par_; ++ipar) {
										std::shared_ptr<T_mat> gradPsi = re_comps_[cluster_i][j]->GetZSigmaZtGrad(ipar, true, 1.);
										grad_cov_aux_par[first_cov_par + ind_par_[j] - 1 + ipar] += -1. * ((double)(y_aux_[cluster_i].transpose() * (*gradPsi) * y_aux_[cluster_i])) / cov_pars[0] / 2. +
											((double)(((*gradPsi).cwiseProduct(psi_inv)).sum())) / 2.;
									}
								}
							}//end not only_grouped_REs_use_woodbury_identity_
						} //end not (GP_approx_ == "fitc" || GP_approx_ == "full_scale_tapering")
					}//end not gp_approx_ == "vecchia"
				}// end loop over clusters
			}//end gauss_likelihood_
			else {//not gauss_likelihood_
				if (include_error_var) {
					Log::REFatal("There is no error variance (nugget effect) for non-Gaussian likelihoods");
				}
				int length_cov_grad = num_cov_par_;
				if (estimate_aux_pars_) {
					length_cov_grad += NumAuxPars();
				}
				grad_cov_aux_par = vec_t::Zero(length_cov_grad);
				vec_t grad_cluster_i(length_cov_grad);
				vec_t empty_unused_vec(0);//placeholder for fixed effects gradient
				const double* fixed_effects_cluster_i_ptr = nullptr;
				vec_t fixed_effects_cluster_i;
				for (const auto& cluster_i : unique_clusters_) {
					//map fixed effects to clusters (if needed)
					vec_t grad_F_cluster_i(num_data_per_cluster_[cluster_i]);
					if (num_clusters_ == 1 && (gp_approx_ != "vecchia" || vecchia_ordering_ == "none")) {//only one cluster / independent realization and order of data does not matter
						fixed_effects_cluster_i_ptr = fixed_effects;
					}
					else if (fixed_effects != nullptr) {//more than one cluster and order of samples matters
						fixed_effects_cluster_i = vec_t(num_data_per_cluster_[cluster_i]);
#pragma omp parallel for schedule(static)
						for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
							fixed_effects_cluster_i[j] = fixed_effects[data_indices_per_cluster_[cluster_i][j]];
						}
						fixed_effects_cluster_i_ptr = fixed_effects_cluster_i.data();
					}
					if (gp_approx_ == "vecchia") {
						likelihood_[cluster_i]->CalcGradNegMargLikelihoodLaplaceApproxVecchia(y_[cluster_i].data(),
							y_int_[cluster_i].data(),
							fixed_effects_cluster_i_ptr,
							B_[cluster_i],
							D_inv_[cluster_i],
							B_grad_[cluster_i],
							D_grad_[cluster_i],
							true,
							false,
							estimate_aux_pars_,
							grad_cluster_i.data(),
							empty_unused_vec,
							grad_cluster_i.data() + num_cov_par_,
							false,
							num_comps_total_);
					}
					else if (only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_) {
						likelihood_[cluster_i]->CalcGradNegMargLikelihoodLaplaceApproxGroupedRE(y_[cluster_i].data(),
							y_int_[cluster_i].data(),
							fixed_effects_cluster_i_ptr,
							num_data_per_cluster_[cluster_i],
							SigmaI_[cluster_i],
							Zt_[cluster_i],
							cum_num_rand_eff_[cluster_i],
							true,
							false,
							estimate_aux_pars_,
							grad_cluster_i.data(),
							empty_unused_vec,
							grad_cluster_i.data() + num_cov_par_,
							false);
					}
					else if (only_one_grouped_RE_calculations_on_RE_scale_) {
						likelihood_[cluster_i]->CalcGradNegMargLikelihoodLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(y_[cluster_i].data(),
							y_int_[cluster_i].data(),
							fixed_effects_cluster_i_ptr,
							num_data_per_cluster_[cluster_i],
							re_comps_[cluster_i][0]->cov_pars_[0],
							re_comps_[cluster_i][0]->random_effects_indices_of_data_.data(),
							true,
							false,
							estimate_aux_pars_,
							grad_cluster_i.data(),
							empty_unused_vec,
							grad_cluster_i.data() + num_cov_par_,
							false);
					}
					else {
						likelihood_[cluster_i]->CalcGradNegMargLikelihoodLaplaceApproxStable(y_[cluster_i].data(),
							y_int_[cluster_i].data(),
							fixed_effects_cluster_i_ptr,
							ZSigmaZt_[cluster_i],
							re_comps_[cluster_i],
							true,
							false,
							estimate_aux_pars_,
							grad_cluster_i.data(),
							empty_unused_vec,
							grad_cluster_i.data() + num_cov_par_,
							false);
					}
					grad_cov_aux_par += grad_cluster_i;
				}// end loop over clusters
			}//end not gauss_likelihood_
		//// For debugging
		//for (int i = 0; i < (int)grad_cov_aux_par.size(); ++i) { Log::REDebug("grad_cov_aux_par[%d]: %g", i, grad_cov_aux_par[i]); }
		}//end CalcGradCovParAuxPars

		/*!
		* \brief Calculate gradient for linear fixed-effect coefficients
		* \param marg_var Marginal variance parameters sigma^2 (only used for Gaussian data)
		* \param[out] grad_beta Gradient for linear regression coefficients
		 * \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian likelihoods)
		*/
		void CalcGradLinCoef(double marg_var,
			vec_t& grad_beta,
			const double* fixed_effects) {
			if (gauss_likelihood_) {
				if (only_grouped_REs_use_woodbury_identity_) {// calculate y_aux = Psi^-1*y (in most cases, this has been already calculated when calling 'CalcCovFactorOrModeAndNegLL' before this)
					CalcYAux(1.);
				}
				vec_t y_aux(num_data_);
				GetYAux(y_aux);
				grad_beta = (-1. / marg_var) * (X_.transpose()) * y_aux;
			}
			else {
				vec_t grad_F(num_data_);
				CalcGradFLaplace(grad_F.data(), fixed_effects);
				grad_beta = (X_.transpose()) * grad_F;
			}
			//// For debugging
			//for (int i = 0; i < (int)grad_beta.size(); ++i) { Log::REDebug("grad_beta[%d]: %g", i, grad_beta[i]); }
		}//end CalcGradLinCoef

		/*!
		* \brief Reset mode to previous value. Used when, e.g., NA or Inf occurred
		*/
		void ResetLaplaceApproxModeToPreviousValue() {
			CHECK(!gauss_likelihood_);
			for (const auto& cluster_i : unique_clusters_) {
				likelihood_[cluster_i]->ResetModeToPreviousValue();
			}
		}

		/*!
		* \brief Profile out sigma2 (=use closed-form expression for error / nugget variance)
		* \return sigma2_
		*/
		double ProfileOutSigma2() {
			sigma2_ = yTPsiInvy_ / num_data_;
			return sigma2_;
		}

		/*!
		* \brief Get sigma2
		* \return sigma2_
		*/
		double Sigma2() {
			return sigma2_;
		}

		/*!
		* \brief Return value of neg_log_likelihood_
		* \return neg_log_likelihood_
		*/
		double GetNegLogLikelihood() {
			return neg_log_likelihood_;
		}

		/*!
		* \brief Return num_cov_par_
		* \return num_cov_par_
		*/
		int GetNumCovPar() {
			return num_cov_par_;
		}

		/*!
		* \brief Return num_coef_
		* \return num_coef_
		*/
		int GetNumCoef() {
			return num_coef_;
		}

		/*!
		* \brief Return has_covariates_
		* \return has_covariates_
		*/
		bool HasCovariates() {
			return has_covariates_;
		}

		/*!
		* \brief Return estimate_aux_pars_
		*/
		bool EstimateAuxPars() const {
			return estimate_aux_pars_;
		}

		/*!
		* \brief Number of auxiliary parameters in likelihoods.h
		*/
		int NumAuxPars() {
			return likelihood_[unique_clusters_[0]]->NumAuxPars();
		}

		/*!
		* \brief Returns a pointer to aux_pars_
		*/
		const double* GetAuxPars() {
			return likelihood_[unique_clusters_[0]]->GetAuxPars();
		}

		/*!
		* \brief Set aux_pars_
		* \param aux_pars New values for aux_pars_
		*/
		void SetAuxPars(const double* aux_pars) {
			for (const auto& cluster_i : unique_clusters_) {
				likelihood_[cluster_i]->SetAuxPars(aux_pars);
			}
		}

		void GetNameFirstAuxPar(string_t& name) {
			likelihood_[unique_clusters_[0]]->GetNameFirstAuxPar(name);
		}

		/*!
		* \brief Set num_iter_. Used by external optimizers from OptimLib
		* \param num_iter New values for num_iter_
		*/
		void SetNumIter(int num_iter) {
			num_iter_ = num_iter;
		}

		/*!
		* \brief Factorize the covariance matrix (Gaussian data) or
		*	calculate the posterior mode of the random effects for use in the Laplace approximation (non-Gaussian likelihoods)
		*	And calculate the negative log-likelihood (Gaussian data) or the negative approx. marginal log-likelihood (non-Gaussian likelihoods)
		* \param cov_pars Covariance parameters
		* \param fixed_effects Fixed effects component of location parameter
		*/
		void CalcCovFactorOrModeAndNegLL(const vec_t& cov_pars,
			const double* fixed_effects) {
			SetCovParsComps(cov_pars);
			if (gauss_likelihood_) {
				CalcCovFactor(gp_approx_ == "vecchia", true, 1., false);//Create covariance matrix and factorize it (and also calculate derivatives if Vecchia approximation is used)
				if (only_grouped_REs_use_woodbury_identity_) {
					CalcYtilde(true);//y_tilde = L^-1 * Z^T * y and y_tilde2 = Z * L^-T * L^-1 * Z^T * y, L = chol(Sigma^-1 + Z^T * Z)
				}
				else {
					CalcYAux(1.);//y_aux = Psi^-1 * y
				}
				EvalNegLogLikelihood(nullptr, cov_pars.data(), nullptr, neg_log_likelihood_, true, true, true, false);
			}//end gauss_likelihood_
			else {//not gauss_likelihood_
				if (gp_approx_ == "vecchia") {
					CalcCovFactor(true, true, 1., false);
				}
				else {
					CalcSigmaComps();
					CalcCovMatrixNonGauss();
				}
				neg_log_likelihood_ = -CalcModePostRandEffCalcMLL(fixed_effects, true);//calculate mode and approximate marginal likelihood
			}//end not gauss_likelihood_
		}//end CalcCovFactorOrModeAndNegLL

		/*!
		* \brief Update fixed effects with new linear regression coefficients
		* \param beta Linear regression coefficients
		* \param fixed_effects Externally provided fixed effects component of location parameter (only used for non-Gaussian likelihoods)
		* \param fixed_effects_vec[out] Vector of fixed effects (used only for non-Gaussian likelihoods)
		*/
		void UpdateFixedEffects(const vec_t& beta,
			const double* fixed_effects,
			vec_t& fixed_effects_vec) {
			if (gauss_likelihood_) {
				vec_t resid = y_vec_ - (X_ * beta);
				if (fixed_effects != nullptr) {//add external fixed effects to linear predictor
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_data_; ++i) {
						resid[i] -= fixed_effects[i];
					}
				}
				SetY(resid.data());
			}
			else {
				fixed_effects_vec = X_ * beta;
				if (fixed_effects != nullptr) {//add external fixed effects to linear predictor
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_data_; ++i) {
						fixed_effects_vec[i] += fixed_effects[i];
					}
				}
			}
		}

		/*!
		* \brief Calculate the value of the negative log-likelihood
		* \param y_data Response variable data
		* \param cov_pars Values for covariance parameters of RE components
		* \param fixed_effects Externally provided fixed effects component of location parameter
		* \param[out] negll Negative log-likelihood
		* \param CalcCovFactor_already_done If true, it is assumed that the covariance matrix has already been factorized
		* \param CalcYAux_already_done If true, it is assumed that y_aux_=Psi^-1y_ has already been calculated (only relevant if not only_grouped_REs_use_woodbury_identity_)
		* \param CalcYtilde_already_done If true, it is assumed that y_tilde = L^-1 * Z^T * y, L = chol(Sigma^-1 + Z^T * Z), has already been calculated (only relevant for only_grouped_REs_use_woodbury_identity_)
		* \param redetermine_neighbors_vecchia If true, 'RedetermineNearestNeighborsVecchia' is called if applicable
		*/
		void EvalNegLogLikelihood(const double* y_data,
			const double* cov_pars,
			const double* fixed_effects,
			double& negll,
			bool CalcCovFactor_already_done,
			bool CalcYAux_already_done,
			bool CalcYtilde_already_done,
			bool redetermine_neighbors_vecchia) {
			CHECK(!(CalcYAux_already_done && !CalcCovFactor_already_done));// CalcYAux_already_done && !CalcCovFactor_already_done makes no sense
			if (fixed_effects != nullptr) {
				if (y_data == nullptr) {
					Log::REFatal("EvalNegLogLikelihood: 'y_data' cannot nullptr when 'fixed_effects' is provided ");
				}
				vec_t y_minus_lp(num_data_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_; ++i) {
					y_minus_lp[i] = y_data[i] - fixed_effects[i];
				}
				SetY(y_minus_lp.data());
			}
			else {
				if (y_data != nullptr) {
					SetY(y_data);
				}
			}
			if (!CalcCovFactor_already_done) {
				const vec_t cov_pars_vec = Eigen::Map<const vec_t>(cov_pars, num_cov_par_);
				SetCovParsComps(cov_pars_vec);
				if (redetermine_neighbors_vecchia) {
					int num_it_temp = num_iter_;//"hack" to make sure that 'RedetermineNearestNeighborsVecchia' is called (if applicable) even if a model has been estimated already
					num_iter_ = 0;
					if (ShouldRedetermineNearestNeighborsVecchia()) {
						RedetermineNearestNeighborsVecchia();//called only if gp_approx == "vecchia" and neighbors are selected based on correlations and not distances
					}
					num_iter_ = num_it_temp;
				}
				CalcCovFactor(false, true, 1., false);//Create covariance matrix and factorize it
			}
			//Calculate quadratic form y^T Psi^-1 y
			CalcYTPsiIInvY(yTPsiInvy_, true, 1, CalcYAux_already_done, CalcYtilde_already_done);
			//Calculate log determinant
			if (matrix_inversion_method_ == "iterative") {
				if (saved_rand_vec_.size() == 0) {
					for (const auto& cluster_i : unique_clusters_) {
						saved_rand_vec_[cluster_i] = false;
					}
				}
			}
			log_det_Psi_ = 0;
			for (const auto& cluster_i : unique_clusters_) {
				if (gp_approx_ == "vecchia") {
					log_det_Psi_ -= D_inv_[cluster_i].diagonal().array().log().sum();
				}
				else {
					if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
						if (matrix_inversion_method_ == "cholesky") {//Cholesky
							log_det_Psi_ -= 2. * (((den_mat_t)chol_fact_sigma_ip_[cluster_i].matrixL()).diagonal().array().log().sum());
							log_det_Psi_ += 2. * (((den_mat_t)chol_fact_sigma_woodbury_[cluster_i].matrixL()).diagonal().array().log().sum());
							if (gp_approx_ == "full_scale_tapering") {
								log_det_Psi_ += 2. * (((T_mat)chol_fact_resid_[cluster_i].matrixL()).diagonal().array().log().sum());
							}
							else {
								log_det_Psi_ += FITC_Diag_[cluster_i].array().log().sum();
							}
						}//end Cholesky
						else if (matrix_inversion_method_ == "iterative") {//Conjugate Gradient
							// Sample probe vectors
							if (!saved_rand_vec_[cluster_i]) {
								if (!cg_generator_seeded_) {
									cg_generator_ = RNG_t(seed_rand_vec_trace_);
									cg_generator_seeded_ = true;
								}
								rand_vec_probe_[cluster_i].resize(num_data_per_cluster_[cluster_i], num_rand_vec_trace_);
								GenRandVecTrace(cg_generator_, rand_vec_probe_[cluster_i]);
								// Sample probe vectors from N(0,P)
								if (cg_preconditioner_type_ == "predictive_process_plus_diagonal") {
									rand_vec_probe_low_rank_[cluster_i].resize(num_ind_points_, num_rand_vec_trace_);
									GenRandVecTrace(cg_generator_, rand_vec_probe_low_rank_[cluster_i]);
									rand_vec_probe_P_[cluster_i] = rand_vec_probe_[cluster_i];
								}
								if (reuse_rand_vec_trace_) {//Use same random vectors for each iteration && cluster_i == end(unique_cluster) Tim
									saved_rand_vec_[cluster_i] = true;
								}
							}
							if (cg_preconditioner_type_ == "predictive_process_plus_diagonal") {
								den_mat_t chol_ip_cross_cov_Z = chol_ip_cross_cov_[cluster_i].transpose() * rand_vec_probe_low_rank_[cluster_i];
								rand_vec_probe_[cluster_i] = chol_ip_cross_cov_Z + diagonal_approx_preconditioner_[cluster_i].cwiseSqrt().asDiagonal() * rand_vec_probe_P_[cluster_i];
							}
							std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov_[cluster_i][0]->GetZSigmaZt();
							std::shared_ptr<T_mat> sigma_resid = re_comps_resid_[cluster_i][0]->GetZSigmaZt();
							// Initialize Solution Sigma^-1 (u_1,...,u_t) 
							solution_for_trace_[cluster_i].resize(num_data_per_cluster_[cluster_i], num_rand_vec_trace_);
							solution_for_trace_[cluster_i].setZero();
							// Initialize tridiagonal Matrices
							std::vector<vec_t> Tdiags_(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag_));
							std::vector<vec_t> Tsubdiags_(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag_ - 1));
							// Conjuagte Gradient with Lanczos
							CGTridiagFSA<T_mat>(*sigma_resid, (*cross_cov), chol_ip_cross_cov_[cluster_i], rand_vec_probe_[cluster_i],
								Tdiags_, Tsubdiags_, solution_for_trace_[cluster_i], NaN_found, num_data_per_cluster_[cluster_i],
								num_rand_vec_trace_, cg_max_num_it_tridiag_, cg_delta_conv_, cg_preconditioner_type_,
								chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
							if (NaN_found) {
								Log::REFatal("There was Nan or Inf value generated in the Conjugate Gradient Method!");
							}
							// LogDet Estimation
							LogDetStochTridiag(Tdiags_, Tsubdiags_, log_det_Psi_, num_data_per_cluster_[cluster_i], num_rand_vec_trace_);
							// Correction for Preconditioner (necessary if using preconditioner)
							if (cg_preconditioner_type_ == "predictive_process_plus_diagonal") {
								log_det_Psi_ -= 2. * (((den_mat_t)chol_fact_sigma_ip_[cluster_i].matrixL()).diagonal().array().log().sum());
								log_det_Psi_ += 2. * (((den_mat_t)chol_fact_woodbury_preconditioner_[cluster_i].matrixL()).diagonal().array().log().sum());
								log_det_Psi_ -= diagonal_approx_inv_preconditioner_[cluster_i].array().log().sum();
							}
						}//end Conjugate Gradient
						else {
							Log::REFatal("Matrix inversion method '%s' is not supported.", matrix_inversion_method_.c_str());
						}
					}//end gp_approx_ == "fitc" or "full_scale_tapering"
					else {
						if (only_grouped_REs_use_woodbury_identity_) {
							if (num_re_group_total_ == 1 && num_comps_total_ == 1) {
								log_det_Psi_ += (2. * sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().log().sum());
							}
							else {
								log_det_Psi_ += (2. * chol_facts_[cluster_i].CholFactMatrix().diagonal().array().log().sum());
							}
							for (int j = 0; j < num_comps_total_; ++j) {
								int num_rand_eff = cum_num_rand_eff_[cluster_i][j + 1] - cum_num_rand_eff_[cluster_i][j];
								log_det_Psi_ += (num_rand_eff * std::log(re_comps_[cluster_i][j]->cov_pars_[0]));
							}
						}
						else {
							log_det_Psi_ += (2. * chol_facts_[cluster_i].CholFactMatrix().diagonal().array().log().sum());
						}
					}
				}
			}
			negll = yTPsiInvy_ / 2. / cov_pars[0] + log_det_Psi_ / 2. + num_data_ / 2. * (std::log(cov_pars[0]) + std::log(2 * M_PI));
		}//end EvalNegLogLikelihood

		/*!
		* \brief Calculate the value of the negative log-likelihood when yTPsiInvy_ and log_det_Psi_ is already known
		* \param sigma2 Nugget / error term variance
		* \param[out] negll Negative log-likelihood
		*/
		void EvalNegLogLikelihoodOnlyUpdateNuggetVariance(const double sigma2,
			double& negll) {
			negll = yTPsiInvy_ / 2. / sigma2 + log_det_Psi_ / 2. + num_data_ / 2. * (std::log(sigma2) + std::log(2 * M_PI));
		}//end EvalNegLogLikelihoodOnlyUpdateNuggetVariance

		/*!
		* \brief Calculate the value of the negative log-likelihood when only the fixed effects part has changed and the covariance matrix has not changed
		*	Note: It is assuzmed that y_ has been set before by calling 'SetY' with the residuals = y - fixed_effcts
		* \param sigma2 Nugget / error term variance
		* \param[out] negll Negative log-likelihood
		*/
		void EvalNegLogLikelihoodOnlyUpdateFixedEffects(const double sigma2,
			double& negll) {
			// Calculate y_aux = Psi^-1 * y (if not only_grouped_REs_use_woodbury_identity_) or y_tilde and y_tilde2 (if only_grouped_REs_use_woodbury_identity_) for covariance parameter update (only for Gaussian data)
			if (only_grouped_REs_use_woodbury_identity_) {
				CalcYtilde(true);//y_tilde = L^-1 * Z^T * y and y_tilde2 = Z * L^-T * L^-1 * Z^T * y, L = chol(Sigma^-1 + Z^T * Z)
			}
			else {
				CalcYAux(1.);//y_aux = Psi^-1 * y
			}
			//Calculate quadratic form y^T Psi^-1 y
			CalcYTPsiIInvY(yTPsiInvy_, true, 1, false, false);
			negll = yTPsiInvy_ / 2. / sigma2 + log_det_Psi_ / 2. + num_data_ / 2. * (std::log(sigma2) + std::log(2 * M_PI));
		}

		/*!
		* \brief Calculate the value of the approximate negative marginal log-likelihood obtained when using the Laplace approximation
		* \param y_data Response variable data
		* \param cov_pars Values for covariance parameters of RE components
		* \param[out] negll Approximate negative marginal log-likelihood
		* \param fixed_effects Fixed effects component of location parameter
		* \param InitializeModeCovMat If true, posterior mode is initialized to 0 and the covariance matrix is calculated. Otherwise, existing values are used
		* \param CalcModePostRandEff_already_done If true, it is assumed that the posterior mode of the random effects has already been calculated
		* \param redetermine_neighbors_vecchia If true, 'RedetermineNearestNeighborsVecchia' is called if applicable
		*/
		void EvalLaplaceApproxNegLogLikelihood(const double* y_data,
			const double* cov_pars,
			double& negll,
			const double* fixed_effects,
			bool InitializeModeCovMat,
			bool CalcModePostRandEff_already_done,
			bool redetermine_neighbors_vecchia) {
			if (y_data != nullptr) {
				SetY(y_data);
			}
			else {
				if (!CalcModePostRandEff_already_done) {
					CHECK(y_has_been_set_);
				}
			}
			if (InitializeModeCovMat) {
				CHECK(cov_pars != nullptr);
			}
			if (CalcModePostRandEff_already_done) {
				negll = neg_log_likelihood_;//Whenever the mode is calculated that likelihood is calculated as well. So we might as well just return the saved neg_log_likelihood_
			}
			else {//not CalcModePostRandEff_already_done
				if (InitializeModeCovMat) {
					//We reset the initial modes to 0. This is done to avoid that different calls to EvalLaplaceApproxNegLogLikelihood lead to (very small) differences.
					for (const auto& cluster_i : unique_clusters_) {
						likelihood_[cluster_i]->InitializeModeAvec();//TODO: maybe ommit this step?
					}
					const vec_t cov_pars_vec = Eigen::Map<const vec_t>(cov_pars, num_cov_par_);
					SetCovParsComps(cov_pars_vec);
					if (redetermine_neighbors_vecchia) {
						int num_it_temp = num_iter_;//"hack" to make sure that 'RedetermineNearestNeighborsVecchia' is called (if applicable) even if a model has been estimated already
						num_iter_ = 0;
						if(ShouldRedetermineNearestNeighborsVecchia()) {
							RedetermineNearestNeighborsVecchia();//called only if gp_approx == "vecchia" and neighbors are selected based on correlations and not distances
						}
						num_iter_ = num_it_temp;
					}
					if (gp_approx_ == "vecchia") {
						CalcCovFactor(true, true, 1., false);
					}
					else {
						CalcSigmaComps();
						CalcCovMatrixNonGauss();
					}
				}//end InitializeModeCovMat
				negll = -CalcModePostRandEffCalcMLL(fixed_effects, true);
			}//end not CalcModePostRandEff_already_done
		}//end EvalLaplaceApproxNegLogLikelihood

		/*!
		* \brief Print out current parameters when trace / logging is activated for convergence monitoring
		* \param cov_pars Covariance parameters on transformed scale
		* \param beta Regression coefficients on transformed scale
		* \param aux_pars Additional parameters for the likelihood
		*/
		void PrintTraceParameters(const vec_t& cov_pars,
			const vec_t& beta,
			const double* aux_pars) {
			vec_t cov_pars_orig, beta_orig;
			if (Log::GetLevelRE() == LogLevelRE::Debug) { // do transformation only if log level Debug is active
				TransformBackCovPars(cov_pars, cov_pars_orig);
				for (int i = 0; i < (int)cov_pars.size(); ++i) {
					Log::REDebug("cov_pars[%d]: %g", i, cov_pars_orig[i]);
				}
				if (has_covariates_) {
					if (scale_covariates_) {
						CHECK(loc_transf_.size() == beta.size());
						CHECK(scale_transf_.size() == beta.size());
						TransformBackCoef(beta, beta_orig);
					}
					else {
						beta_orig = beta;
					}
					for (int i = 0; i < std::min((int)beta.size(), NUM_COEF_PRINT_TRACE_); ++i) {
						Log::REDebug("beta[%d]: %g", i, beta_orig[i]);
					}
					if (has_covariates_ && beta.size() > NUM_COEF_PRINT_TRACE_) {
						Log::REDebug("Note: only the first %d linear regression coefficients are shown ", NUM_COEF_PRINT_TRACE_);
					}
				}
				if (estimate_aux_pars_) {
					for (int i = 0; i < NumAuxPars(); ++i) {
						Log::REDebug("%s: %g", likelihood_[unique_clusters_[0]]->GetNameAuxPars(i), aux_pars[i]);
					}
				}
			}
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
		* \param vecchia_pred_type Type of Vecchia approximation for making predictions. "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points, "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted), "order_pred_first" = predicted data is ordered first for making predictions, "latent_order_obs_first_cond_obs_only"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are only observed points, "latent_order_obs_first_cond_all"  = Vecchia approximation for the latent process and observed data is ordered first and neighbors are selected among all points
		* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions (-1 means that the value already set at initialization is used)
		* \param cg_delta_conv_pred Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for prediction
		* \param nsim_var_pred Number of samples when simulation is used for calculating predictive variances
		* \param rank_pred_approx_matrix_lanczos Rank of the matrix for approximating predictive covariances obtained using the Lanczos algorithm
		*/
		void SetPredictionData(int num_data_pred,
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
			int rank_pred_approx_matrix_lanczos) {
			if (!(gp_coords_data_pred == nullptr && re_group_data_pred == nullptr && re_group_rand_coef_data_pred == nullptr
				&& cluster_ids_data_pred == nullptr && gp_rand_coef_data_pred == nullptr && covariate_data_pred == nullptr)) {
				CHECK(num_data_pred > 0);
				num_data_pred_ = num_data_pred;
			}
			if (cluster_ids_data_pred != nullptr) {
				cluster_ids_data_pred_ = std::vector<data_size_t>(cluster_ids_data_pred, cluster_ids_data_pred + num_data_pred);
			}
			if (re_group_data_pred != nullptr) {
				//For grouped random effecst: create matrix 're_group_levels_pred' (vector of vectors, dimension: num_group_variables_ x num_data_) with strings of group levels from characters in 'const char* re_group_data_pred'
				re_group_levels_pred_ = std::vector<std::vector<re_group_t>>(num_group_variables_, std::vector<re_group_t>(num_data_pred));
				ConvertCharToStringGroupLevels(num_data_pred, num_group_variables_, re_group_data_pred, re_group_levels_pred_);
			}
			if (re_group_rand_coef_data_pred != nullptr) {
				re_group_rand_coef_data_pred_ = std::vector<double>(re_group_rand_coef_data_pred, re_group_rand_coef_data_pred + num_data_pred * num_re_group_rand_coef_);
			}
			if (gp_coords_data_pred != nullptr) {
				gp_coords_data_pred_ = std::vector<double>(gp_coords_data_pred, gp_coords_data_pred + num_data_pred * dim_gp_coords_);
			}
			if (gp_rand_coef_data_pred != nullptr) {
				gp_rand_coef_data_pred_ = std::vector<double>(gp_rand_coef_data_pred, gp_rand_coef_data_pred + num_data_pred * num_gp_rand_coef_);
			}
			if (covariate_data_pred != nullptr) {
				covariate_data_pred_ = std::vector<double>(covariate_data_pred, covariate_data_pred + num_data_pred * num_coef_);
			}
			if (gp_approx_ == "vecchia") {
				if (vecchia_pred_type != nullptr) {
					SetVecchiaPredType(vecchia_pred_type);
				}
				if (num_neighbors_pred > 0) {
					num_neighbors_pred_ = num_neighbors_pred;
				}
			}
			if (matrix_inversion_method_ == "iterative") {
				if (cg_delta_conv_pred > 0) {
					cg_delta_conv_pred_ = cg_delta_conv_pred;
				}
				if (nsim_var_pred > 0) {
					nsim_var_pred_ = nsim_var_pred;
				}
				if (rank_pred_approx_matrix_lanczos > 0) {
					rank_pred_approx_matrix_lanczos_ = rank_pred_approx_matrix_lanczos;
				}
				SetMatrixInversionPropertiesLikelihood();
			}
		}//end SetPredictionData

		/*!
		* \brief Make predictions: calculate conditional mean and variances or covariance matrix
		*		 Note: You should pre-allocate memory for out_predict
		*			   Its length is equal to num_data_pred if only the conditional mean is predicted (predict_cov_mat==false && predict_var==false)
		*			   or num_data_pred * (1 + num_data_pred) if the predictive covariance matrix is also calculated (predict_cov_mat==true)
		*			   or num_data_pred * 2 if predictive variances are also calculated (predict_var==true)
		* \param cov_pars_pred Covariance parameters of components
		* \param y_obs Response variable for observed data
		* \param num_data_pred Number of data points for which predictions are made
		* \param[out] out_predict Predictive/conditional mean at prediciton points followed by the predictive covariance matrix in column-major format (if predict_cov_mat==true) or the predictive variances (if predict_var==true)
		* \param calc_cov_factor If true, the covariance matrix of the observed data is factorized otherwise a previously done factorization is used (default=true)
		* \param predict_cov_mat If true, the predictive/conditional covariance matrix is calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
		* \param predict_var If true, the predictive/conditional variances are calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
		* \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects
		* \param covariate_data_pred Covariate data (=independent variables, features) for prediction
		* \param coef_pred Coefficients for linear covariates
		* \param cluster_ids_data_pred IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization) for which predictions are to be made
		* \param re_group_data_pred Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
		* \param re_group_rand_coef_data_pred Covariate data for grouped random coefficients
		* \param gp_coords_data_pred Coordinates (features) for Gaussian process
		* \param gp_rand_coef_data_pred Covariate data for Gaussian process random coefficients
		* \param use_saved_data If true, saved data is used and some arguments are ignored
		* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian likelihoods)
		* \param fixed_effects_pred Fixed effects component of location parameter for predicted data (only used for non-Gaussian likelihoods)
		*/
		void Predict(const double* cov_pars_pred,
			const double* y_obs,
			data_size_t num_data_pred,
			double* out_predict,
			bool calc_cov_factor,
			bool predict_cov_mat,
			bool predict_var,
			bool predict_response,
			const double* covariate_data_pred,
			const double* coef_pred,
			const data_size_t* cluster_ids_data_pred,
			const char* re_group_data_pred,
			const double* re_group_rand_coef_data_pred,
			double* gp_coords_data_pred,
			const double* gp_rand_coef_data_pred,
			bool use_saved_data,
			const double* fixed_effects,
			const double* fixed_effects_pred) {
			//First check whether previously set data should be used and load it if required
			std::vector<std::vector<re_group_t>> re_group_levels_pred, re_group_levels_pred_orig;//Matrix with group levels for the grouped random effects (re_group_levels_pred[j] contains the levels for RE number j)
			// Note: re_group_levels_pred_orig is only used for the case (only_one_grouped_RE_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_for_prediction_)
			//			since then re_group_levels_pred is over-written for every cluster and the original data thus needs to be saved
			if (use_saved_data) {
				if (num_data_pred > 0) {
					CHECK(num_data_pred == num_data_pred_);
				}
				else {
					num_data_pred = num_data_pred_;
				}
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
			}// end use_saved_data
			else {
				if (re_group_data_pred != nullptr) {
					//For grouped random effects: create matrix 're_group_levels_pred' (vector of vectors, dimension: num_group_variables_ x num_data_) with strings of group levels from characters in 'const char* re_group_data_pred'
					re_group_levels_pred = std::vector<std::vector<re_group_t>>(num_group_variables_, std::vector<re_group_t>(num_data_pred));
					ConvertCharToStringGroupLevels(num_data_pred, num_group_variables_, re_group_data_pred, re_group_levels_pred);
				}
			}
			if (only_one_grouped_RE_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {
				re_group_levels_pred_orig = re_group_levels_pred;
			}
			//Some checks including whether required data is present
			if (gp_approx_ == "vecchia") {
				CHECK(num_neighbors_pred_ > 0);
				CHECK(cg_delta_conv_pred_ > 0.);
				CHECK(nsim_var_pred_ > 0);
				CHECK(rank_pred_approx_matrix_lanczos_ > 0);
			}
			if ((int)re_group_levels_pred.size() == 0 && num_group_variables_ > 0) {
				Log::REFatal("Missing grouping data ('group_data_pred') for grouped random effects for making predictions");
			}
			if (re_group_rand_coef_data_pred == nullptr && num_re_group_rand_coef_ > 0) {
				Log::REFatal("Missing covariate data ('re_group_rand_coef_data_pred') for random coefficients "
					"for grouped random effects for making predictions");
			}
			if (gp_coords_data_pred == nullptr && num_gp_ > 0) {
				Log::REFatal("Missing coordinates data ('gp_coords_pred') for Gaussian process for making predictions");
			}
			if (gp_rand_coef_data_pred == nullptr && num_gp_rand_coef_ > 0) {
				Log::REFatal("Missing covariate data ('gp_rand_coef_data_pred') for random coefficients for Gaussian process for making predictions");
			}
			if (cluster_ids_data_pred == nullptr && num_clusters_ > 1) {
				Log::REFatal("Missing cluster_id data ('cluster_ids_pred') for making predictions");
			}
			CHECK(num_data_pred > 0);
			if (!gauss_likelihood_ && predict_response && predict_cov_mat) {
				Log::REFatal("Calculation of the predictive covariance matrix is not supported "
					"when predicting the response variable (label) for non-Gaussian likelihoods");
			}
			if (predict_cov_mat && predict_var) {
				Log::REFatal("Calculation of both the predictive covariance matrix and variances is not supported. "
					"Choose one of these option (predict_cov_mat or predict_var)");
			}
			if (gp_approx_ == "vecchia" && gauss_likelihood_ && predict_var && num_data_pred > 10000) {
				Log::REWarning("Calculation of (only) predictive variances is currently not optimized for the Vecchia approximation, "
					"and this might takes a lot of time and/or memory.");
			}
			CHECK(cov_pars_pred != nullptr);
			if (has_covariates_) {
				if (covariate_data_pred == nullptr) {
					Log::REFatal("Covariate data 'X_pred' not provided ");
				}
				CHECK(coef_pred != nullptr);
			}
			else {
				if (covariate_data_pred != nullptr) {
					Log::REFatal("Covariate data 'X_pred' provided but model has no linear regresion covariates ");
				}
			}
			if (y_obs == nullptr) {
				if (!y_has_been_set_) {
					Log::REFatal("Response variable data is not provided and has not been set before");
				}
			}
			else {
				// Check response variable data
				if (LightGBM::Common::HasNAOrInf(y_obs, num_data_)) {
					Log::REFatal("NaN or Inf in response variable / label");
				}
			}
			if (num_data_pred > 10000 && predict_cov_mat) {
				double num_mem_d = ((double)num_data_pred) * ((double)num_data_pred);
				int mem_size = (int)(num_mem_d * 8. / 1000000.);
				Log::REWarning("The covariance matrix can be very large for large sample sizes which might lead to memory limitations. "
					"In your case (n = %d), the covariance needs at least approximately %d mb of memory. ", num_data_pred, mem_size);
			}
			// Initialize linear predictor related terms and covariance parameters
			vec_t coef, mu;//mu = linear regression predictor
			if (has_covariates_) {//calculate linear regression term
				coef = Eigen::Map<const vec_t>(coef_pred, num_coef_);
				den_mat_t X_pred = Eigen::Map<const den_mat_t>(covariate_data_pred, num_data_pred, num_coef_);
				mu = X_pred * coef;
			}
			vec_t cov_pars = Eigen::Map<const vec_t>(cov_pars_pred, num_cov_par_);
			//Set up cluster IDs
			std::map<data_size_t, int> num_data_per_cluster_pred;
			std::map<data_size_t, std::vector<int>> data_indices_per_cluster_pred;
			std::vector<data_size_t> unique_clusters_pred;
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
			//Factorize covariance matrix and calculate Psi^{-1}y_obs or calculate Laplace approximation (if required)
			if (pred_for_observed_data) {
				SetYCalcCovCalcYAuxForPred(cov_pars, coef, y_obs, calc_cov_factor, fixed_effects, false);
			}
			// Loop over different clusters to calculate predictions
			for (const auto& cluster_i : unique_clusters_pred) {

				//Case 1: no data observed for this Gaussian process with ID 'cluster_i'
				if (std::find(unique_clusters_.begin(), unique_clusters_.end(), cluster_i) == unique_clusters_.end()) {
					T_mat psi;
					std::vector<std::shared_ptr<RECompBase<T_mat>>> re_comps_cluster_i;
					int num_REs_pred = num_data_per_cluster_pred[cluster_i];
					//Calculate covariance matrix if needed
					if (predict_cov_mat || predict_var || predict_response) {
						if (gp_approx_ == "vecchia") {
							//TODO: move this code out into another function for better readability
							// Initialize RE components
							std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_data_per_cluster_pred[cluster_i]);
							std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_data_per_cluster_pred[cluster_i]);
							std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_data_per_cluster_pred[cluster_i]);
							std::vector<Triplet_t> entries_init_B_cluster_i;
							std::vector<Triplet_t> entries_init_B_grad_cluster_i;
							std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_data_per_cluster_pred[cluster_i]);
							CreateREComponentsVecchia<T_mat>(num_data_pred, dim_gp_coords_, data_indices_per_cluster_pred, cluster_i,
								num_data_per_cluster_pred, gp_coords_data_pred,
								gp_rand_coef_data_pred, re_comps_cluster_i,
								nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i,
								entries_init_B_cluster_i, entries_init_B_grad_cluster_i, z_outer_z_obs_neighbors_cluster_i, only_one_GP_calculations_on_RE_scale_, has_duplicates_coords_,
								"none", num_neighbors_pred_, vecchia_neighbor_selection_, false, rng_, num_gp_rand_coef_, num_gp_total_, num_comps_total_, gauss_likelihood_,
								cov_fct_, cov_fct_shape_, cov_fct_taper_range_, cov_fct_taper_shape_, gp_approx_ == "tapering");//TODO: maybe also use ordering for making predictions? (need to check that there are not errors)
							for (int j = 0; j < num_comps_total_; ++j) {
								const vec_t pars = cov_pars.segment(ind_par_[j], ind_par_[j + 1] - ind_par_[j]);
								re_comps_cluster_i[j]->SetCovPars(pars);
							}
							// Calculate a Cholesky factor
							sp_mat_t B_cluster_i;
							sp_mat_t D_inv_cluster_i;
							std::vector<sp_mat_t> B_grad_cluster_i;//not used, but needs to be passed to function
							std::vector<sp_mat_t> D_grad_cluster_i;//not used, but needs to be passed to function
							CalcCovFactorVecchia<T_mat>(num_data_per_cluster_pred[cluster_i], false, re_comps_cluster_i,
								nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i,
								entries_init_B_cluster_i, entries_init_B_grad_cluster_i,
								z_outer_z_obs_neighbors_cluster_i,
								B_cluster_i, D_inv_cluster_i, B_grad_cluster_i, D_grad_cluster_i,
								true, 1., false, num_gp_total_, ind_intercept_gp_, gauss_likelihood_);
							//Calculate Psi
							sp_mat_t D_sqrt(num_data_per_cluster_pred[cluster_i], num_data_per_cluster_pred[cluster_i]);
							D_sqrt.setIdentity();
							D_sqrt.diagonal().array() = D_inv_cluster_i.diagonal().array().pow(-0.5);
							sp_mat_t B_inv_D_sqrt;
							TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(B_cluster_i, D_sqrt, B_inv_D_sqrt, false);
							psi = B_inv_D_sqrt * B_inv_D_sqrt.transpose();
						}//end gp_approx_ == "vecchia"
						else if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
							psi = T_mat(num_REs_pred, num_REs_pred);
							if (gauss_likelihood_ && predict_response) {
								psi.setIdentity();//nugget effect
							}
							else {
								psi.setZero();
							}
							std::vector<std::shared_ptr<RECompGP<den_mat_t>>> re_comps_ip_cluster_i;
							std::vector<std::shared_ptr<RECompGP<den_mat_t>>> re_comps_cross_cov_cluster_i;
							std::vector<std::shared_ptr<RECompGP<T_mat>>> re_comps_resid_cluster_i;
							CreateREComponentsPPFSA(num_data_pred, data_indices_per_cluster_pred, cluster_i, gp_coords_data_pred,
								re_comps_ip_cluster_i, re_comps_cross_cov_cluster_i, re_comps_resid_cluster_i);
							for (int j = 0; j < num_comps_total_; ++j) {
								const vec_t pars = cov_pars.segment(ind_par_[j], ind_par_[j + 1] - ind_par_[j]);
								re_comps_ip_cluster_i[j]->SetCovPars(pars);
								re_comps_cross_cov_cluster_i[j]->SetCovPars(pars);
								re_comps_ip_cluster_i[j]->CalcSigma();
								re_comps_cross_cov_cluster_i[j]->CalcSigma();
								den_mat_t sigma_ip_stable = *(re_comps_ip_cluster_i[j]->GetZSigmaZt());
								den_mat_t cross_cov = *(re_comps_cross_cov_cluster_i[j]->GetZSigmaZt());
								sigma_ip_stable.diagonal().array() += EPSILON_ADD_COVARIANCE_STABLE;
								chol_den_mat_t chol_fact_sigma_ip;
								chol_fact_sigma_ip.compute(sigma_ip_stable);
								den_mat_t sigma_interim = cross_cov * chol_fact_sigma_ip.solve(cross_cov.transpose());
								ConvertTo_T_mat_FromDense<T_mat>(sigma_interim, psi); // for all T_mat? see ConvertTo_T_mat_FromDense from Pascal
								//psi = cross_cov * chol_fact_sigma_ip.solve(cross_cov.transpose());
								if (gp_approx_ == "full_scale_tapering") {
									re_comps_resid_cluster_i[j]->SetCovPars(pars);
									re_comps_resid_cluster_i[j]->CalcSigma();
									// Subtract predictive process covariance
									re_comps_resid_cluster_i[j]->SubtractMatFromSigmaForResidInFullScale(psi);
									// Apply Taper
									re_comps_resid_cluster_i[j]->ApplyTaper();

									psi += *(re_comps_resid_cluster_i[j]->GetZSigmaZt());
								}
								else {
									vec_t FITC_Diag = vec_t::Zero(cross_cov.rows());
									FITC_Diag = FITC_Diag.array() + sigma_ip_stable.coeffRef(0, 0);
									FITC_Diag -= psi.diagonal();
									psi += FITC_Diag.asDiagonal();
								}
							}
						}//end gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering"
						else if (gp_approx_ == "none") {
							CreateREComponents(num_data_pred, data_indices_per_cluster_pred,
								cluster_i, re_group_levels_pred, num_data_per_cluster_pred,
								re_group_rand_coef_data_pred, gp_coords_data_pred,
								gp_rand_coef_data_pred, true, re_comps_cluster_i);
							if (only_one_GP_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_) {
								num_REs_pred = re_comps_cluster_i[0]->GetNumUniqueREs();
							}
							else {
								num_REs_pred = num_data_per_cluster_pred[cluster_i];
							}
							psi = T_mat(num_REs_pred, num_REs_pred);
							if (gauss_likelihood_ && predict_response) {
								psi.setIdentity();//nugget effect
							}
							else {
								psi.setZero();
							}
							for (int j = 0; j < num_comps_total_; ++j) {
								const vec_t pars = cov_pars.segment(ind_par_[j], ind_par_[j + 1] - ind_par_[j]);
								re_comps_cluster_i[j]->SetCovPars(pars);
								re_comps_cluster_i[j]->CalcSigma();
								psi += (*(re_comps_cluster_i[j]->GetZSigmaZt().get()));
							}

						}//end gp_approx_ == "none"
						if (gauss_likelihood_) {
							psi *= cov_pars[0];//back-transform
						}
					}//end calculation of covariance matrix
					// Add external fixed_effects
					vec_t mean_pred_id = vec_t::Zero(num_data_per_cluster_pred[cluster_i]);
					if (fixed_effects_pred != nullptr) {//add externaly provided fixed effects
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
							mean_pred_id[i] += fixed_effects_pred[data_indices_per_cluster_pred[cluster_i][i]];
						}
					}
					// Add linear regression predictor
					if (has_covariates_) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
							mean_pred_id[i] += mu[data_indices_per_cluster_pred[cluster_i][i]];
						}
					}
					bool predict_var_or_response = predict_var || (predict_response && !gauss_likelihood_);
					vec_t var_pred_id;
					if (predict_var_or_response) {
						var_pred_id = psi.diagonal();
					}
					// Map from predictions from random effects scale b to "data scale" Zb
					if (only_one_GP_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_) {
						if (predict_var_or_response) {
							vec_t var_pred_id_on_RE_scale = var_pred_id;
							var_pred_id = vec_t(num_data_per_cluster_pred[cluster_i]);
#pragma omp parallel for schedule(static)
							for (data_size_t i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
								var_pred_id[i] = var_pred_id_on_RE_scale[(re_comps_cluster_i[0]->random_effects_indices_of_data_)[i]];
							}
						}
						if (predict_cov_mat) {
							T_mat cov_mat_pred_id_on_RE_scale = psi;
							sp_mat_t Zpred(num_data_per_cluster_pred[cluster_i], num_REs_pred);
							std::vector<Triplet_t> triplets(num_data_per_cluster_pred[cluster_i]);
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
								triplets[i] = Triplet_t(i, (re_comps_cluster_i[0]->random_effects_indices_of_data_)[i], 1.);
							}
							Zpred.setFromTriplets(triplets.begin(), triplets.end());
							psi = Zpred * cov_mat_pred_id_on_RE_scale * Zpred.transpose();
						}
					}//end only_one_GP_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_
					// Transform to response scale for non-Gaussian likelihoods if needed
					if (!gauss_likelihood_ && predict_response) {
						likelihood_[unique_clusters_[0]]->PredictResponse(mean_pred_id, var_pred_id, predict_var);
					}
					// Write on output
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
						out_predict[data_indices_per_cluster_pred[cluster_i][i]] = mean_pred_id[i];
					}
					// Write covariance / variance on output
					if (!predict_response || gauss_likelihood_) {//this is not done if predict_response==true for non-Gaussian likelihoods 
						if (predict_cov_mat) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {//column index
								for (int j = 0; j < num_data_per_cluster_pred[cluster_i]; ++j) {//row index
									out_predict[data_indices_per_cluster_pred[cluster_i][i] * num_data_pred + data_indices_per_cluster_pred[cluster_i][j] + num_data_pred] = psi.coeff(j, i);
								}
							}
						}//end predict_cov_mat
						if (predict_var) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_pred[cluster_i][i] + num_data_pred] = var_pred_id[i];
							}
						}//end predict_var
					}//end !predict_response || gauss_likelihood_
					else { // predict_response && !gauss_likelihood_
						if (predict_var) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_pred[cluster_i][i] + num_data_pred] = var_pred_id[i];
							}
						}//end predict_var
					}//end write covariance / variance on output

				}//end cluster_i with no observed data
				else {

					//Case 2: there exists observed data for this cluster_i
					den_mat_t gp_coords_mat_pred;
					std::vector<data_size_t> random_effects_indices_of_data_pred;
					int num_REs_pred = num_data_per_cluster_pred[cluster_i];
					if (num_gp_ > 0) {
						std::vector<double> gp_coords_pred;
						for (int j = 0; j < dim_gp_coords_; ++j) {
							for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
								gp_coords_pred.push_back(gp_coords_data_pred[j * num_data_pred + id]);
							}
						}
						gp_coords_mat_pred = Eigen::Map<den_mat_t>(gp_coords_pred.data(), num_data_per_cluster_pred[cluster_i], dim_gp_coords_);
					}
					if (only_one_grouped_RE_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {
						// Determine unique group levels per cluster and create map which maps every data point per cluster to a group level
						random_effects_indices_of_data_pred = std::vector<data_size_t>(num_data_per_cluster_pred[cluster_i]);
						std::vector<re_group_t> re_group_levels_pred_unique;
						std::map<re_group_t, int> map_group_label_index_pred;
						int num_group_pred = 0;
						int ii = 0;
						for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
							if (map_group_label_index_pred.find(re_group_levels_pred_orig[0][id]) == map_group_label_index_pred.end()) {
								map_group_label_index_pred.insert({ re_group_levels_pred_orig[0][id], num_group_pred });
								re_group_levels_pred_unique.push_back(re_group_levels_pred_orig[0][id]);
								random_effects_indices_of_data_pred[ii] = num_group_pred;
								num_group_pred += 1;
							}
							else {
								random_effects_indices_of_data_pred[ii] = map_group_label_index_pred[re_group_levels_pred_orig[0][id]];
							}
							ii += 1;
						}
						re_group_levels_pred[0] = re_group_levels_pred_unique;
						num_REs_pred = (int)re_group_levels_pred[0].size();
					}//end only_one_grouped_RE_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_for_prediction_
					else if (only_one_GP_calculations_on_RE_scale_) {
						random_effects_indices_of_data_pred = std::vector<data_size_t>(num_data_per_cluster_pred[cluster_i]);
						std::vector<int> uniques;//unique points
						std::vector<int> unique_idx;//used for constructing incidence matrix Z_ if there are duplicates
						DetermineUniqueDuplicateCoordsFast(gp_coords_mat_pred, num_data_per_cluster_pred[cluster_i], uniques, unique_idx);
#pragma omp for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
							random_effects_indices_of_data_pred[i] = unique_idx[i];
						}
						den_mat_t gp_coords_mat_pred_unique = gp_coords_mat_pred(uniques, Eigen::all);
						gp_coords_mat_pred = gp_coords_mat_pred_unique;
						num_REs_pred = (int)gp_coords_mat_pred.rows();
					}//end only_one_GP_calculations_on_RE_scale_
					// Initialize predictive mean and covariance
					vec_t mean_pred_id;
					if (only_one_GP_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_ ||
						only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {
						mean_pred_id = vec_t(num_REs_pred);
					}
					else {
						mean_pred_id = vec_t(num_data_per_cluster_pred[cluster_i]);
					}
					T_mat cov_mat_pred_id;
					vec_t var_pred_id;
					sp_mat_t Bpo, Bp; // used only if gp_approx_ == "vecchia" && !gauss_likelihood_
					vec_t Dp;
					std::shared_ptr<RECompGP<T_mat>> re_comp;
					if (gp_approx_ == "vecchia") {
						re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_[cluster_i][ind_intercept_gp_]);
					}
					bool predict_var_or_response = predict_var || (predict_response && !gauss_likelihood_);//variance needs to be available for response prediction for non-Gaussian likelihoods
					// Calculate predictions
					if (gp_approx_ == "vecchia") {
						if (gauss_likelihood_) {
							int num_data_tot = num_data_per_cluster_[cluster_i] + num_data_per_cluster_pred[cluster_i];
							double num_mem_d = ((double)num_neighbors_pred_) * ((double)num_neighbors_pred_) * (double)(num_data_tot)+(double)(num_neighbors_pred_) * (double)(num_data_tot);
							int mem_size = (int)(num_mem_d * 8. / 1000000.);
							if (mem_size > 4000) {
								Log::REDebug("The current implementation of the Vecchia approximation needs a lot of memory if the number of neighbors is large. "
									"In your case (nb. of neighbors = %d, nb. of observations = %d, nb. of predictions = %d), "
									"this needs at least approximately %d mb of memory.",
									num_neighbors_pred_, num_data_per_cluster_[cluster_i], num_data_per_cluster_pred[cluster_i], mem_size);
							}
							if (vecchia_pred_type_ == "order_obs_first_cond_obs_only") {
								CalcPredVecchiaObservedFirstOrder<T_mat>(true, cluster_i, num_data_pred, data_indices_per_cluster_pred,
									re_comp->coords_, gp_coords_mat_pred, gp_rand_coef_data_pred, num_neighbors_pred_, vecchia_neighbor_selection_,
									re_comps_, ind_intercept_gp_, num_gp_rand_coef_, num_gp_total_, y_[cluster_i], gauss_likelihood_, rng_,
									predict_cov_mat, predict_var, mean_pred_id, cov_mat_pred_id, var_pred_id, Bpo, Bp, Dp);
							}
							else if (vecchia_pred_type_ == "order_obs_first_cond_all") {
								CalcPredVecchiaObservedFirstOrder<T_mat>(false, cluster_i, num_data_pred, data_indices_per_cluster_pred,
									re_comp->coords_, gp_coords_mat_pred, gp_rand_coef_data_pred, num_neighbors_pred_, vecchia_neighbor_selection_,
									re_comps_, ind_intercept_gp_, num_gp_rand_coef_, num_gp_total_, y_[cluster_i], gauss_likelihood_, rng_,
									predict_cov_mat, predict_var, mean_pred_id, cov_mat_pred_id, var_pred_id, Bpo, Bp, Dp);
							}
							else if (vecchia_pred_type_ == "order_pred_first") {
								CalcPredVecchiaPredictedFirstOrder<T_mat>(cluster_i, num_data_pred, data_indices_per_cluster_pred,
									re_comp->coords_, gp_coords_mat_pred, gp_rand_coef_data_pred, num_neighbors_pred_, vecchia_neighbor_selection_,
									re_comps_, ind_intercept_gp_, num_gp_rand_coef_, num_gp_total_, y_[cluster_i], rng_,
									predict_cov_mat, predict_var, mean_pred_id, cov_mat_pred_id, var_pred_id);
							}
							else if (vecchia_pred_type_ == "latent_order_obs_first_cond_obs_only") {
								if (num_gp_rand_coef_ > 0) {
									Log::REFatal("The Vecchia approximation for latent process(es) is currently not implemented when having random coefficients");
								}
								CalcPredVecchiaLatentObservedFirstOrder<T_mat>(true, cluster_i,
									re_comp->coords_, gp_coords_mat_pred, num_neighbors_pred_, vecchia_neighbor_selection_,
									re_comps_, ind_intercept_gp_, y_[cluster_i], rng_,
									predict_cov_mat, predict_var, predict_response, mean_pred_id, cov_mat_pred_id, var_pred_id);
								// Note: we use the function 'CalcPredVecchiaLatentObservedFirstOrder' instead of the function 'CalcPredVecchiaObservedFirstOrder' since 
								//	the current implementation cannot handle duplicate values in gp_coords (coordinates / input features) for Vecchia approximations
								//	for latent processes (as matrices that need to be inverted will be singular due to the duplicate values).
								//	The function 'CalcPredVecchiaLatentObservedFirstOrder' avoids this singularity problem by using incidence matrices Z and 
								//	and applying a Vecchia approximation to the GP of all unique gp_coords
							}
							else if (vecchia_pred_type_ == "latent_order_obs_first_cond_all") {
								if (num_gp_rand_coef_ > 0) {
									Log::REFatal("The Vecchia approximation for latent process(es) is currently not implemented when having random coefficients");
								}
								CalcPredVecchiaLatentObservedFirstOrder<T_mat>(false, cluster_i,
									re_comp->coords_, gp_coords_mat_pred, num_neighbors_pred_, vecchia_neighbor_selection_,
									re_comps_, ind_intercept_gp_, y_[cluster_i], rng_,
									predict_cov_mat, predict_var, predict_response, mean_pred_id, cov_mat_pred_id, var_pred_id);
							}
							else {
								Log::REFatal("Prediction type '%s' is not supported for the Veccia approximation.", vecchia_pred_type_.c_str());
							}
							if (predict_var || predict_cov_mat) {
								// subtract nugget variance in case latent process is predicted
								if (!predict_response && (vecchia_pred_type_ == "order_obs_first_cond_obs_only" ||
									vecchia_pred_type_ == "order_obs_first_cond_all" ||
									vecchia_pred_type_ == "order_pred_first")) {
									if (predict_cov_mat) {
#pragma omp parallel for schedule(static)
										for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
											cov_mat_pred_id.coeffRef(i, i) -= 1.;
										}
									}
									if (predict_var) {
#pragma omp parallel for schedule(static)
										for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
											var_pred_id[i] -= 1.;
										}
									}
								}
							}
						}//end gauss_likelihood_
						else {//not gauss_likelihood_
							const double* fixed_effects_cluster_i_ptr = nullptr;
							// Note that fixed_effects_cluster_i_ptr is not used since calc_mode == false
							// The mode has been calculated already before in the Predict() function above
							// mean_pred_id and cov_mat_pred_id are not calculate in 'CalcPredVecchiaObservedFirstOrder', only Bpo, Bp, and Dp for non-Gaussian likelihoods
							if (vecchia_pred_type_ == "latent_order_obs_first_cond_obs_only") {
								CalcPredVecchiaObservedFirstOrder<T_mat>(true, cluster_i, num_data_pred, data_indices_per_cluster_pred,
									re_comp->coords_, gp_coords_mat_pred, gp_rand_coef_data_pred, num_neighbors_pred_, vecchia_neighbor_selection_,
									re_comps_, ind_intercept_gp_, num_gp_rand_coef_, num_gp_total_, y_[cluster_i], gauss_likelihood_, rng_,
									false, false, mean_pred_id, cov_mat_pred_id, var_pred_id, Bpo, Bp, Dp);
								likelihood_[cluster_i]->PredictLaplaceApproxVecchia(y_[cluster_i].data(), y_int_[cluster_i].data(), fixed_effects_cluster_i_ptr,
									B_[cluster_i], D_inv_[cluster_i], Bpo, Bp, Dp,
									mean_pred_id, cov_mat_pred_id, var_pred_id,
									predict_cov_mat, predict_var_or_response, false, true);
							}
							else if (vecchia_pred_type_ == "latent_order_obs_first_cond_all") {
								CalcPredVecchiaObservedFirstOrder<T_mat>(false, cluster_i, num_data_pred, data_indices_per_cluster_pred,
									re_comp->coords_, gp_coords_mat_pred, gp_rand_coef_data_pred, num_neighbors_pred_, vecchia_neighbor_selection_,
									re_comps_, ind_intercept_gp_, num_gp_rand_coef_, num_gp_total_, y_[cluster_i], gauss_likelihood_, rng_,
									false, false, mean_pred_id, cov_mat_pred_id, var_pred_id, Bpo, Bp, Dp);
								likelihood_[cluster_i]->PredictLaplaceApproxVecchia(y_[cluster_i].data(), y_int_[cluster_i].data(), fixed_effects_cluster_i_ptr,
									B_[cluster_i], D_inv_[cluster_i], Bpo, Bp, Dp,
									mean_pred_id, cov_mat_pred_id, var_pred_id,
									predict_cov_mat, predict_var_or_response, false, false);
							}
							else {
								Log::REFatal("Prediction type '%s' is not supported for the Veccia approximation.", vecchia_pred_type_.c_str());
							}
						}//end not gauss_likelihood_
					}//end gp_approx_ == "vecchia"
					else {// not gp_approx_ == "vecchia"
						if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
							CalcPredPPFSA(cluster_i, num_data_per_cluster_pred, num_data_per_cluster_, gp_coords_mat_pred, predict_cov_mat,
								predict_var, mean_pred_id, cov_mat_pred_id, var_pred_id, nsim_var_pred_, cg_delta_conv_pred_);
						}
						else {
							CalcPred(cluster_i, num_data_pred, num_data_per_cluster_pred, data_indices_per_cluster_pred,
								re_group_levels_pred, re_group_rand_coef_data_pred, gp_coords_mat_pred, gp_rand_coef_data_pred,
								predict_cov_mat, predict_var_or_response, predict_response,
								mean_pred_id, cov_mat_pred_id, var_pred_id);
						}
					}//end not gp_approx_ == "vecchia"
					//map from predictions from random effects scale b to "data scale" Zb
					if (only_one_GP_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_ ||
						only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {
						vec_t mean_pred_id_on_RE_scale = mean_pred_id;
						mean_pred_id = vec_t(num_data_per_cluster_pred[cluster_i]);
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
							mean_pred_id[i] = mean_pred_id_on_RE_scale[random_effects_indices_of_data_pred[i]];
						}
						if (predict_var_or_response) {
							vec_t var_pred_id_on_RE_scale = var_pred_id;
							var_pred_id = vec_t(num_data_per_cluster_pred[cluster_i]);
#pragma omp parallel for schedule(static)
							for (data_size_t i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
								var_pred_id[i] = var_pred_id_on_RE_scale[random_effects_indices_of_data_pred[i]];
							}
						}
						if (predict_cov_mat) {
							T_mat cov_mat_pred_id_on_RE_scale = cov_mat_pred_id;
							sp_mat_t Zpred(num_data_per_cluster_pred[cluster_i], num_REs_pred);
							std::vector<Triplet_t> triplets(num_data_per_cluster_pred[cluster_i]);
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
								triplets[i] = Triplet_t(i, random_effects_indices_of_data_pred[i], 1.);
							}
							Zpred.setFromTriplets(triplets.begin(), triplets.end());
							cov_mat_pred_id = Zpred * cov_mat_pred_id_on_RE_scale * Zpred.transpose();
						}
					}//end only_one_GP_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_for_prediction_
					// Add externaly provided fixed effects
					if (fixed_effects_pred != nullptr) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
							mean_pred_id[i] += fixed_effects_pred[data_indices_per_cluster_pred[cluster_i][i]];
						}
					}
					// Add linear regression predictor
					if (has_covariates_) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
							mean_pred_id[i] += mu[data_indices_per_cluster_pred[cluster_i][i]];
						}
					}
					if (!gauss_likelihood_ && predict_response) {
						likelihood_[unique_clusters_[0]]->PredictResponse(mean_pred_id, var_pred_id, predict_var);
					}
					// Write on output
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
						out_predict[data_indices_per_cluster_pred[cluster_i][i]] = mean_pred_id[i];
					}
					// Write covariance / variance on output
					if (predict_cov_mat) {
						if (gauss_likelihood_) {
							cov_mat_pred_id *= cov_pars[0];
						}
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {//column index
							for (int j = 0; j < num_data_per_cluster_pred[cluster_i]; ++j) {//row index
								out_predict[data_indices_per_cluster_pred[cluster_i][i] * num_data_pred + data_indices_per_cluster_pred[cluster_i][j] + num_data_pred] = cov_mat_pred_id.coeff(j, i);
							}
						}
					}//end predict_cov_mat
					if (predict_var) {
						if (gauss_likelihood_) {
							var_pred_id *= cov_pars[0];
						}
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {
							out_predict[data_indices_per_cluster_pred[cluster_i][i] + num_data_pred] = var_pred_id[i];
						}
					}//end predict_var
					//end write covariance / variance on output
				}//end cluster_i with data
			}//end loop over cluster
			//Set cross-covariances between different independent clusters to 0
			if (predict_cov_mat && unique_clusters_pred.size() > 1 && (!predict_response || gauss_likelihood_)) {
				for (const auto& cluster_i : unique_clusters_pred) {
					for (const auto& cluster_j : unique_clusters_pred) {
						if (cluster_i != cluster_j) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_per_cluster_pred[cluster_i]; ++i) {//column index
								for (int j = 0; j < num_data_per_cluster_pred[cluster_j]; ++j) {//row index
									out_predict[data_indices_per_cluster_pred[cluster_i][i] * num_data_pred + data_indices_per_cluster_pred[cluster_j][j] + num_data_pred] = 0.;
								}
							}
						}
					}
				}
			}
		}//end Predict

		/*!
		* \brief Predict ("estimate") training data random effects
		* \param cov_pars_pred Covariance parameters of components
		* \param coef_pred Coefficients for linear covariates
		* \param y_obs Response variable for observed data
		* \param[out] out_predict Predicted training data random effects and variances if calc_var
		* \param calc_cov_factor If true, the covariance matrix of the observed data is factorized otherwise a previously done factorization is used
		* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian likelihoods)
		* \param calc_var If true, variances are also calculated
		*/
		void PredictTrainingDataRandomEffects(const double* cov_pars_pred,
			const double* coef_pred,
			const double* y_obs,
			double* out_predict,
			bool calc_cov_factor,
			const double* fixed_effects,
			bool calc_var) {
			//Some checks
			CHECK(cov_pars_pred != nullptr);
			if (has_covariates_) {
				CHECK(coef_pred != nullptr);
			}
			if (y_obs == nullptr) {
				if (!y_has_been_set_) {
					Log::REFatal("Response variable data is not provided and has not been set before");
				}
			}
			vec_t cov_pars = Eigen::Map<const vec_t>(cov_pars_pred, num_cov_par_);
			vec_t coef;
			if (has_covariates_) {
				coef = Eigen::Map<const vec_t>(coef_pred, num_coef_);
			}
			if (gauss_likelihood_ && gp_approx_ == "vecchia") {
				calc_cov_factor = true;//recalculate Vecchia approximation since it might have been done (saved in B_) with a different nugget effect if calc_std_dev == true in CalcStdDevCovPar
			}
			SetYCalcCovCalcYAuxForPred(cov_pars, coef, y_obs, calc_cov_factor, fixed_effects, true);
			// Loop over different clusters to calculate predictions
			for (const auto& cluster_i : unique_clusters_) {
				if (gauss_likelihood_) {
					if (gp_approx_ == "vecchia") {
						if (num_comps_total_ > 1) {
							Log::REFatal("PredictTrainingDataRandomEffects() is not implemented for the Vecchia approximation "
								"when having multiple GPs / random coefficient GPs ");
						}
#pragma omp parallel for schedule(static)// Write on output
						for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
							out_predict[data_indices_per_cluster_[cluster_i][i]] = y_[cluster_i][i] - y_aux_[cluster_i][i];
						}
						if (calc_var) {
							sp_mat_t B_inv(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
							sp_mat_t M_aux = B_[cluster_i].cwiseProduct(D_inv_[cluster_i] * B_[cluster_i]);
#pragma omp parallel for schedule(static)// Write on output
							for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * num_comps_total_] = cov_pars[0] * (1. - M_aux.col(i).sum());
							}
						}
					}
					else {// not gp_approx_ == "vecchia"
						int cn = 0;//component number counter
						const vec_t* y_aux = &(y_aux_[cluster_i]);
						vec_t mean_pred_id, var_pred_id;
						if (calc_var) {
							var_pred_id = vec_t(num_data_per_cluster_[cluster_i]);
						}
						//Grouped random effects
						for (int j = 0; j < num_re_group_total_; ++j) {
							double sigma = re_comps_[cluster_i][cn]->cov_pars_[0];
							if (only_grouped_REs_use_woodbury_identity_ && num_re_group_total_ == 1) {
								if (re_comps_[cluster_i][cn]->IsRandCoef()) {
									Log::REFatal("PredictTrainingDataRandomEffects() is not implemented when having only one grouped random coefficient effect ");
								}
								vec_t ZtYAux;
								mean_pred_id = vec_t(num_data_per_cluster_[cluster_i]);
								int num_re = re_comps_[cluster_i][cn]->GetNumUniqueREs();
								CalcZtVGivenIndices(num_data_per_cluster_[cluster_i], num_re,
									re_comps_[cluster_i][cn]->random_effects_indices_of_data_.data(), *y_aux, ZtYAux, true);
#pragma omp parallel for schedule(static)
								for (data_size_t i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
									mean_pred_id[i] = sigma * ZtYAux[(re_comps_[cluster_i][0]->random_effects_indices_of_data_)[i]];
								}
								if (calc_var) {
									vec_t M_aux = (ZtZ_[cluster_i].diagonal().array() / sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array()).matrix();
									vec_t M_aux2 = (ZtZ_[cluster_i].diagonal().array() - M_aux.array().square()).matrix();
									vec_t M_aux3 = (sigma - (sigma * sigma * M_aux2.array())).matrix();
									vec_t M_aux4 = M_aux3.array().sqrt().matrix();
									sp_mat_t M_aux5 = M_aux4.asDiagonal() * Zt_[cluster_i];
#pragma omp parallel for schedule(static)
									for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
										var_pred_id[i] = cov_pars[0] * M_aux5.col(i).squaredNorm();
									}
								}
							}//end only_grouped_REs_use_woodbury_identity_ && num_re_group_total_ == 1
							else {
								sp_mat_t* Z_j = re_comps_[cluster_i][cn]->GetZ();
								sp_mat_t Z_base_j;
								if (re_comps_[cluster_i][cn]->IsRandCoef()) {
									Z_base_j = *Z_j;
#pragma omp parallel for schedule(static)
									for (int k = 0; k < Z_base_j.outerSize(); ++k) {
										for (sp_mat_t::InnerIterator it(Z_base_j, k); it; ++it) {
											it.valueRef() = 1.;
										}
									}
									mean_pred_id = sigma * Z_base_j * ((*Z_j).transpose() * (*y_aux));
								}
								else {
									mean_pred_id = sigma * (*Z_j) * ((*Z_j).transpose() * (*y_aux));
								}
								if (calc_var) {
									sp_mat_t ZjtZj = (*Z_j).transpose() * (*Z_j);
									if (only_grouped_REs_use_woodbury_identity_) {
										T_mat M_aux;
										if (CholeskyHasPermutation<T_chol>(chol_facts_[cluster_i])) {
											TriangularSolve<T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), P_ZtZj_[cluster_i][cn], M_aux, false);
										}
										else {
											TriangularSolve<T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), ZtZj_[cluster_i][cn], M_aux, false);
										}
										T_mat M_aux3 = sigma * sigma * (M_aux.transpose() * M_aux - ZjtZj);
										M_aux3.diagonal().array() += sigma;
										if (re_comps_[cluster_i][cn]->IsRandCoef()) {
											Z_j = &Z_base_j;
										}
										T_mat M_aux4 = (*Z_j) * M_aux3;
#pragma omp parallel for schedule(static)
										for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
											var_pred_id[i] = cov_pars[0] * (*Z_j).row(i).cwiseProduct(M_aux4.row(i)).sum();
										}
									}//end only_grouped_REs_use_woodbury_identity_
									else {//!only_grouped_REs_use_woodbury_identity_
										T_mat M_aux;
										TriangularSolveGivenCholesky<T_chol, T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i], *Z_j, M_aux, false);
										T_mat M_aux2 = (*Z_j) * M_aux.transpose();
										if (re_comps_[cluster_i][cn]->IsRandCoef()) {
											Z_j = &Z_base_j;
										}
#pragma omp parallel for schedule(static)
										for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
											var_pred_id[i] = cov_pars[0] * (sigma - sigma * sigma * M_aux2.row(i).squaredNorm());
										}
									}//end !only_grouped_REs_use_woodbury_identity_
								}//end calc_var
							}//end !(only_grouped_REs_use_woodbury_identity_ && num_re_group_total_ == 1)
#pragma omp parallel for schedule(static)// Write on output
							for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * cn] = mean_pred_id[i];
							}
							if (calc_var) {
#pragma omp parallel for schedule(static)// Write on output
								for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
									out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * cn + num_data_ * num_comps_total_] = var_pred_id[i];
								}
							}
							cn += 1;
						}//end gouped random effects
						//Gaussian process
						if (num_gp_ > 0) {
							std::shared_ptr<RECompGP<T_mat>> re_comp_base = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_[cluster_i][cn]);
							sp_mat_t* Z_j = nullptr, * Z_base_j = nullptr;
							for (int j = 0; j < num_gp_total_; ++j) {
								double sigma = re_comps_[cluster_i][cn]->cov_pars_[0];
								std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_[cluster_i][cn]);
								if (re_comp->IsRandCoef() || re_comp_base->HasZ()) {
									Z_j = re_comp->GetZ();
									if (re_comp_base->HasZ()) {
										Z_base_j = re_comp_base->GetZ();
									}
								}
								if (re_comp_base->HasZ()) {
									mean_pred_id = (*Z_base_j) * ((re_comp->sigma_) * ((*Z_j).transpose() * (*y_aux)));
								}
								else if (re_comp->IsRandCoef()) {
									mean_pred_id = (re_comp->sigma_) * ((*Z_j).transpose() * (*y_aux));
								}
								else {
									mean_pred_id = (re_comp->sigma_) * (*y_aux);
								}
								if (calc_var) {
									T_mat M_aux, M_aux2, M_aux3;
									if (re_comp_base->HasZ()) {
										M_aux = (*Z_j) * (re_comp->sigma_) * (*Z_base_j).transpose();
									}
									else if (re_comp->IsRandCoef()) {
										M_aux = (*Z_j) * (re_comp->sigma_);
									}
									else {
										M_aux = re_comp->sigma_;
									}
									TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_facts_[cluster_i], M_aux, M_aux2, false);
#pragma omp parallel for schedule(static)
									for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
										var_pred_id[i] = cov_pars[0] * (sigma - M_aux2.col(i).squaredNorm());
									}
								}
#pragma omp parallel for schedule(static)// Write on output
								for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
									out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * cn] = mean_pred_id[i];
								}
								if (calc_var) {
#pragma omp parallel for schedule(static)// Write on output
									for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
										out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * cn + num_data_ * num_comps_total_] = var_pred_id[i];
									}
								}
								cn += 1;
							}//end loop over GP components
						}// end Gaussian process
					}// end not gp_approx_ == "vecchia"
				}//end gauss_likelihood_
				else {//not gauss_likelihood_
					const vec_t* mode = likelihood_[cluster_i]->GetMode();
					if (gp_approx_ == "vecchia") {
						if (only_one_GP_calculations_on_RE_scale_) {//there are duplicates
#pragma omp parallel for schedule(static)// Write on output
							for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_[cluster_i][i]] = (*mode)[(re_comps_[cluster_i][0]->random_effects_indices_of_data_)[i]];
							}
						}
						else {//no duplicates
#pragma omp parallel for schedule(static)// Write on output
							for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_[cluster_i][i]] = (*mode)[i];
							}
						}
						if (calc_var) {
							vec_t var_pred_id;
							likelihood_[cluster_i]->CalcVarLaplaceApproxVecchia(var_pred_id);
							if (only_one_GP_calculations_on_RE_scale_) {//there are duplicates
#pragma omp parallel for schedule(static)// Write on output
								for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
									out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_] = var_pred_id[(re_comps_[cluster_i][0]->random_effects_indices_of_data_)[i]];
								}
							}
							else {//no duplicates
#pragma omp parallel for schedule(static)// Write on output
								for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
									out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_] = var_pred_id[i];
								}
							}
						}//end calc_var
					}//end gp_approx_ == "vecchia"
					else if (only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_) {
						vec_t var_pred_all;
						for (int cn = 0; cn < num_re_group_total_; ++cn) {
							vec_t mean_pred_id;
							sp_mat_t* Z_j = re_comps_[cluster_i][cn]->GetZ();
							sp_mat_t Z_base_j;
							if (re_comps_[cluster_i][cn]->IsRandCoef()) {
								Z_base_j = *Z_j;
#pragma omp parallel for schedule(static)
								for (int k = 0; k < Z_base_j.outerSize(); ++k) {
									for (sp_mat_t::InnerIterator it(Z_base_j, k); it; ++it) {
										it.valueRef() = 1.;
									}
								}
								Z_j = &Z_base_j;
							}
							int num_re_comp = cum_num_rand_eff_[cluster_i][cn + 1] - cum_num_rand_eff_[cluster_i][cn];
							int start_re_comp = cum_num_rand_eff_[cluster_i][cn];
							mean_pred_id = (*Z_j) * (*mode).segment(start_re_comp, num_re_comp);
#pragma omp parallel for schedule(static)// Write on output
							for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * cn] = mean_pred_id[i];
							}
							if (calc_var) {
								if (cn == 0) {
									likelihood_[cluster_i]->CalcVarLaplaceApproxGroupedRE(var_pred_all);
								}
								vec_t var_pred_id = (*Z_j) * var_pred_all.segment(start_re_comp, num_re_comp);
#pragma omp parallel for schedule(static)// Write on output
								for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
									out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * cn + num_data_ * num_comps_total_] = var_pred_id[i];
								}
							}
						}
					}//end only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_
					else if (only_one_grouped_RE_calculations_on_RE_scale_ || only_one_GP_calculations_on_RE_scale_) {
#pragma omp parallel for schedule(static)// Write on output
						for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
							out_predict[data_indices_per_cluster_[cluster_i][i]] = (*mode)[(re_comps_[cluster_i][0]->random_effects_indices_of_data_)[i]];
						}
						if (calc_var) {
							vec_t var_pred_id;
							if (only_one_GP_calculations_on_RE_scale_) {
								likelihood_[cluster_i]->CalcVarLaplaceApproxOnlyOneGPCalculationsOnREScale(ZSigmaZt_[cluster_i],
									var_pred_id);
							}
							else if (only_one_grouped_RE_calculations_on_RE_scale_) {
								likelihood_[cluster_i]->CalcVarLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(var_pred_id);
							}
#pragma omp parallel for schedule(static)// Write on output
							for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_] = var_pred_id[(re_comps_[cluster_i][0]->random_effects_indices_of_data_)[i]];
							}
						}
					}//end only_one_grouped_RE_calculations_on_RE_scale_ || only_one_GP_calculations_on_RE_scale_
					else {//at least one GP and additional components
						//Note: use the "general" prediction formula since the mode is calculated on the aggragate scale and not for every component separaretly (mode_.size() == num_data and not num_data * num_comps_total_)
						if (calc_var) {
							Log::REFatal("PredictTrainingDataRandomEffects(): calculating of variances is not implemented when having at least on GP and additional random effects ");
						}
						const vec_t* first_deriv = likelihood_[cluster_i]->GetFirstDerivLL();
						int cn = 0;//component number counter
						vec_t mean_pred_id;
						for (int j = 0; j < num_re_group_total_; ++j) {
							double sigma = re_comps_[cluster_i][cn]->cov_pars_[0];
							if (re_comps_[cluster_i][cn]->IsRandCoef()) {
								sp_mat_t* Z_j = re_comps_[cluster_i][cn]->GetZ();
								sp_mat_t Z_base_j = *Z_j;
#pragma omp parallel for schedule(static)
								for (int k = 0; k < Z_base_j.outerSize(); ++k) {
									for (sp_mat_t::InnerIterator it(Z_base_j, k); it; ++it) {
										it.valueRef() = 1.;
									}
								}
								mean_pred_id = sigma * Z_base_j * (*Z_j).transpose() * (*first_deriv);
							}
							else {
								sp_mat_t* Z_j = re_comps_[cluster_i][cn]->GetZ();
								mean_pred_id = sigma * (*Z_j) * (*Z_j).transpose() * (*first_deriv);
							}
#pragma omp parallel for schedule(static)// Write on output
							for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * cn] = mean_pred_id[i];
							}
							cn += 1;
						}//end loop over grouped RE
						//GPs 
						std::shared_ptr<RECompGP<T_mat>> re_comp_base = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_[cluster_i][cn]);
						sp_mat_t* Z_j = nullptr, * Z_base_j = nullptr;
						for (int j = 0; j < num_gp_total_; ++j) {
							std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_[cluster_i][cn]);
							if (re_comp->IsRandCoef() || re_comp_base->HasZ()) {
								Z_j = re_comp->GetZ();
								if (re_comp_base->HasZ()) {
									Z_base_j = re_comp_base->GetZ();
								}
							}
							if (re_comp_base->HasZ()) {
								mean_pred_id = (*Z_base_j) * (re_comp->sigma_) * (*Z_j).transpose() * (*first_deriv);
							}
							else if (re_comp->IsRandCoef()) {
								mean_pred_id = (re_comp->sigma_) * (*Z_j).transpose() * (*first_deriv);
							}
							else {
								mean_pred_id = (re_comp_base->sigma_) * (*first_deriv);
							}
#pragma omp parallel for schedule(static)// Write on output
							for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
								out_predict[data_indices_per_cluster_[cluster_i][i] + num_data_ * cn] = mean_pred_id[i];
							}
							cn += 1;
						}//end loop over GPs
					}//end at least one GP and additional components
				}//end not gauss_likelihood_
			}//end loop over cluster
		}//end PredictTrainingDataRandomEffects

		/*!
		* \brief Find "reasonable" default values for the initial values of the covariance parameters (on transformed scale)
		*		 Note: You should pre-allocate memory for optim_cov_pars (length = number of covariance parameters)
		* \param y_data Response variable data
		* \param fixed_effects Additional fixed effects that are added to the linear predictor (= offset)
		* \param[out] init_cov_pars Initial values for covariance parameters of RE components
		*/
		void FindInitCovPar(const double* y_data,
			const double* fixed_effects,
			double* init_cov_pars) {
			double mean = 0;
			double var = 0;
			int ind_par;
			double init_marg_var = 1.;
			if (gauss_likelihood_) {
				//determine initial value for nugget effect
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:mean)
					for (int i = 0; i < num_data_; ++i) {
						mean += y_data[i];
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:mean)
					for (int i = 0; i < num_data_; ++i) {
						mean += y_data[i] - fixed_effects[i];
					}
				}
				mean /= num_data_;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:var)
					for (int i = 0; i < num_data_; ++i) {
						var += (y_data[i] - mean) * (y_data[i] - mean);
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:var)
					for (int i = 0; i < num_data_; ++i) {
						var += (y_data[i] - fixed_effects[i] - mean) * (y_data[i] - fixed_effects[i] - mean);
					}
				}
				var /= (num_data_ - 1);
				init_cov_pars[0] = var / 2;
				ind_par = 1;
			}//end Gaussian data
			else {//non-Gaussian likelihoods
				ind_par = 0;
				if (optimizer_cov_pars_ == "nelder_mead") {
					init_marg_var = 0.1;
				}
				//TODO: find better initial values depending on the likelihood (e.g., poisson, gamma, etc.)
			}
			if (gp_approx_ == "vecchia") {//Neither distances nor coordinates are saved for random coefficient GPs in the Vecchia approximation -> cannot find initial parameters -> just copy the ones from the intercept GP
				// find initial values for intercept process
				int num_par_j = ind_par_[1] - ind_par_[0];
				vec_t pars = vec_t(num_par_j);
				re_comps_[unique_clusters_[0]][0]->FindInitCovPar(rng_, pars, init_marg_var);
				for (int jj = 0; jj < num_par_j; ++jj) {
					init_cov_pars[ind_par] = pars[jj];
					ind_par++;
				}
				//set the same values to random coefficient processes
				for (int j = 1; j < num_gp_total_; ++j) {
					num_par_j = ind_par_[j + 1] - ind_par_[j];
					for (int jj = 0; jj < num_par_j; ++jj) {
						init_cov_pars[ind_par] = pars[jj];
						ind_par++;
					}
				}
			}
			else {
				for (int j = 0; j < num_comps_total_; ++j) {
					int num_par_j = ind_par_[j + 1] - ind_par_[j];
					vec_t pars = vec_t(num_par_j);
					if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
						re_comps_ip_[unique_clusters_[0]][j]->FindInitCovPar(rng_, pars, init_marg_var);
					}
					else {
						re_comps_[unique_clusters_[0]][j]->FindInitCovPar(rng_, pars, init_marg_var);
					}
					for (int jj = 0; jj < num_par_j; ++jj) {
						init_cov_pars[ind_par] = pars[jj];
						ind_par++;
					}
				}
			}
		}//end FindInitCovPar

		int num_cov_par() {
			return(num_cov_par_);
		}

		/*!
		* \brief Calculate the leaf values when performing a Newton update step after the tree structure has been found in tree-boosting
		*    Note: only used in GPBoost for combined Gaussian process tree-boosting (this is called from 'objective_function_->NewtonUpdateLeafValues'). It is assumed that 'CalcYAux' has been called before (from 'objective_function_->GetGradients').
		* \param data_leaf_index Leaf index for every data point (array of size num_data)
		* \param num_leaves Number of leaves
		* \param[out] leaf_values Leaf values when performing a Newton update step (array of size num_leaves)
		* \param marg_variance The marginal variance. Default = 1. Can be used to multiply values by it since Newton updates do not depend on it but 'CalcYAux' might have been called using marg_variance!=1.
		*/
		void NewtonUpdateLeafValues(const int* data_leaf_index,
			const int num_leaves,
			double* leaf_values,
			double marg_variance) {
			if (!gauss_likelihood_) {
				Log::REFatal("Newton updates for leaf values is only supported for Gaussian data");
			}
			CHECK(y_aux_has_been_calculated_);//y_aux_ has already been calculated when calculating the gradient for finding the tree structure from 'GetGradients' in 'regression_objetive.hpp'
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
				den_mat_t HTPsiInvH_cluster_i;
				if (gp_approx_ == "vecchia") {
					sp_mat_t H_cluster_i(num_data_per_cluster_[cluster_i], num_leaves);//row major format is needed for Vecchia approx.
					H_cluster_i.setFromTriplets(entries_H_cluster_i.begin(), entries_H_cluster_i.end());
					HTYAux -= H_cluster_i.transpose() * y_aux_[cluster_i];//minus sign since y_aux_ has been calculated on the gradient = F-y (and not y-F)
					sp_mat_t BH = B_[cluster_i] * H_cluster_i;
					HTPsiInvH_cluster_i = den_mat_t(BH.transpose() * D_inv_[cluster_i] * BH);
				}
				else {
					sp_mat_t H_cluster_i(num_data_per_cluster_[cluster_i], num_leaves);
					H_cluster_i.setFromTriplets(entries_H_cluster_i.begin(), entries_H_cluster_i.end());
					HTYAux -= H_cluster_i.transpose() * y_aux_[cluster_i];//minus sign since y_aux_ has been calculated on the gradient = F-y (and not y-F)
					if (only_grouped_REs_use_woodbury_identity_) {
						T_mat MInvSqrtZtH;
						if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
							sp_mat_t ZtH_cluster_i = Zt_[cluster_i] * H_cluster_i;
							MInvSqrtZtH = sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().inverse().matrix().asDiagonal() * ZtH_cluster_i;
						}
						else {
							sp_mat_t ZtH_cluster_i;
							if (CholeskyHasPermutation<T_chol>(chol_facts_[cluster_i])) {
								ZtH_cluster_i = P_Zt_[cluster_i] * H_cluster_i;
							}
							else {
								ZtH_cluster_i = Zt_[cluster_i] * H_cluster_i;
							}
							TriangularSolve<T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), ZtH_cluster_i, MInvSqrtZtH, false);
						}
						HTPsiInvH_cluster_i = H_cluster_i.transpose() * H_cluster_i - MInvSqrtZtH.transpose() * MInvSqrtZtH;
					}
					else {
						T_mat PsiInvSqrtH;
						TriangularSolveGivenCholesky<T_chol, T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i], H_cluster_i, PsiInvSqrtH, false);
						HTPsiInvH_cluster_i = PsiInvSqrtH.transpose() * PsiInvSqrtH;
					}
				}
				HTPsiInvH += HTPsiInvH_cluster_i;
			}
			HTYAux *= marg_variance;
			vec_t new_leaf_values = HTPsiInvH.llt().solve(HTYAux);
			for (int i = 0; i < num_leaves; ++i) {
				leaf_values[i] = new_leaf_values[i];
			}
		}//end NewtonUpdateLeafValues

		/*!
		* \brief Apply a momentum step
		* \param it Iteration number
		* \param pars Parameters
		* \param pars_lag1 Parameters from last iteration
		* \param[out] pars_acc Accelerated parameters
		* \param nesterov_acc_rate Nesterov acceleration speed
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
		* \param exclude_first_log_scale If true, no momentum is applied to the first value and the momentum step is done on the log-scale for the other values. Default = true
		* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
		* \param log_scale If true, the momentum step is done on the log-scale
		*/
		static void ApplyMomentumStep(int it,
			vec_t& pars,
			vec_t& pars_lag1,
			vec_t& pars_acc,
			double nesterov_acc_rate,
			int nesterov_schedule_version,
			bool exclude_first_log_scale,
			int momentum_offset,
			bool log_scale) {
			double mu = NesterovSchedule(it, nesterov_schedule_version, nesterov_acc_rate, momentum_offset);
			int num_par = (int)pars.size();
			if (exclude_first_log_scale) {
				pars_acc[0] = pars[0];
				pars_acc.segment(1, num_par - 1) = ((mu + 1.) * (pars.segment(1, num_par - 1).array().log()) - mu * (pars_lag1.segment(1, num_par - 1).array().log())).exp().matrix();//Momentum is added on the log scale
			}
			else {
				if (log_scale) {
					pars_acc = ((mu + 1.) * (pars.array().log()) - mu * (pars_lag1.array().log())).exp().matrix();
				}
				else {
					pars_acc = (mu + 1) * pars - mu * pars_lag1;
				}
			}
		}// end ApplyMomentumStep

		/*!
		* \brief Indicates whether correlation-based nearest neighbors for Vecchia approximation should be updated
		* \return redetermine_nn True, if nearest neighbors have been redetermined
		*/
		bool ShouldRedetermineNearestNeighborsVecchia() {
			bool redetermine_nn = false;
			if (gp_approx_ == "vecchia") {
				std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_[unique_clusters_[0]][ind_intercept_gp_]);
				if (!(re_comp->ShouldSaveDistances())) {
					if ((((num_iter_ + 1) & num_iter_) == 0) || num_iter_ == 0) {//(num_iter_ + 1) is power of 2 or 0. 
						// Note that convergence of internal optimizers is not checked in iterations with redetermine_nn if convergence_criterion_ == "relative_change_in_log_likelihood"
						redetermine_nn = true;
					}
				}
			}
			return(redetermine_nn);
		}//end RedetermineNearestNeighborsVecchia

		/*!
		* \brief Redetermine correlation-based nearest neighbors for Vecchia approximation
		*/
		void RedetermineNearestNeighborsVecchia() {
			CHECK(ShouldRedetermineNearestNeighborsVecchia());
			for (const auto& cluster_i : unique_clusters_) {
				// redetermine nearest neighbors for models for which neighbors are selected based on correlations / scaled distances
				UpdateNearestNeighbors(re_comps_[cluster_i], nearest_neighbors_[cluster_i],
					entries_init_B_[cluster_i], entries_init_B_grad_[cluster_i],
					num_neighbors_, vecchia_neighbor_selection_, rng_, ind_intercept_gp_);
				if (!gauss_likelihood_) {
					likelihood_[cluster_i]->SetCholFactPatternAnalyzedFalse();
				}
			}
			if (num_ll_evaluations_ > 0) {
				Log::REDebug("Nearest neighbors redetermined after iteration number %d ", num_iter_ + 1);
			}
		}//end RedetermineNearestNeighborsVecchia

		/*!
		* \brief Get the maximal step length along a direction such that relative changes of covariance and auxiliary parameters are not larger than 'MAX_GRADIENT_UPDATE_LOG_SCALE_'
		* \param neg_step_dir Negative step direction for making updates. E.g., neg_step_dir = grad for gradient descent and neg_step_dir = FI^-1 * grad for Fisher scoring (="natural" gradient)
		*/
		double MaximalLearningRateCovAuxPars(const vec_t& neg_step_dir) const {
			double max_abs_neg_step_dir = 0.;
			for (int ip = 0; ip < (int)neg_step_dir.size(); ++ip) {
				if (std::abs(neg_step_dir[ip]) > max_abs_neg_step_dir) {
					max_abs_neg_step_dir = std::abs(neg_step_dir[ip]);
				}
			}
			return(MAX_GRADIENT_UPDATE_LOG_SCALE_ / max_abs_neg_step_dir);
		}//end MaximalLearningRateCovAuxPars

		/*!
		* \brief Get the maximal step length along a direction such that the change in linear regression coefficients is not overly large (monitor relative change in linear predictor)
		* \param beta Current / lag1 value of beta
		* \param neg_step_dir Negative step direction for making updates
		*/
		double MaximalLearningRateCoef(const vec_t& beta,
			const vec_t& neg_step_dir) const {
			vec_t lp_change = X_ * neg_step_dir;
			vec_t lp_lag1 = X_ * beta;
			double mean_lp_change = 0., mean_lp_lag1 = 0., var_lp_change = 0, cov_lp_lag1_lp_change = 0;
#pragma omp parallel for schedule(static) reduction(+:mean_lp_change, mean_lp_lag1, var_lp_change, cov_lp_lag1_lp_change)
			for (data_size_t i = 0; i < num_data_; ++i) {
				mean_lp_change += lp_change[i];
				mean_lp_lag1 += lp_lag1[i];
				var_lp_change += lp_change[i] * lp_change[i];
				cov_lp_lag1_lp_change += lp_change[i] * lp_lag1[i];
			}
			mean_lp_change /= num_data_;
			mean_lp_lag1 /= num_data_;
			var_lp_change /= num_data_;
			cov_lp_lag1_lp_change /= num_data_;
			var_lp_change -= mean_lp_change * mean_lp_change;
			cov_lp_lag1_lp_change -= mean_lp_change * mean_lp_lag1;
			double max_lr_mu = C_mu_ * C_MAX_CHANGE_COEF_ / std::abs(mean_lp_change);
			double max_lr_var = (std::abs(cov_lp_lag1_lp_change) +
				std::sqrt(cov_lp_lag1_lp_change * cov_lp_lag1_lp_change + 4 * var_lp_change * C_sigma2_ * C_MAX_CHANGE_COEF_)) /
				2 / var_lp_change;
			return(std::min({ max_lr_mu , max_lr_var }));
		}//end MaximalLearningRateCoef

	private:

		// RESPONSE DATA
		/*! \brief Number of data points */
		data_size_t num_data_;
		/*! \brief If true, the response variables have a Gaussian likelihood, otherwise not */
		bool gauss_likelihood_ = true;
		/*! \brief Likelihood objects */
		std::map<data_size_t, std::unique_ptr<Likelihood<T_mat, T_chol>>> likelihood_;
		/*! \brief Value of negative log-likelihood or approximate marginal negative log-likelihood for non-Gaussian likelihoods */
		double neg_log_likelihood_;
		/*! \brief Value of negative log-likelihood or approximate marginal negative log-likelihood for non-Gaussian likelihoods of previous iteration in optimization used for convergence checking */
		double neg_log_likelihood_lag1_;
		/*! \brief Value of negative log-likelihood or approximate marginal negative log-likelihood for non-Gaussian likelihoods after linear regression coefficients are update (this equals neg_log_likelihood_lag1_ if there are no regression coefficients). This is used for step-size checking for the covariance parameters */
		double neg_log_likelihood_after_lin_coef_update_;
		/*! \brief Key: labels of independent realizations of REs/GPs, value: data y */
		std::map<data_size_t, vec_t> y_;
		/*! \brief Copy of response data (used only for Gaussian data and if there are also linear covariates since then y_ is modified during the optimization algorithm and this contains the original data) */
		vec_t y_vec_;
		/*! \brief Key: labels of independent realizations of REs/GPs, value: data y of integer type (used only for non-Gaussian likelihood) */
		std::map<data_size_t, vec_int_t> y_int_;
		// Note: the response variable data is saved in y_ / y_int_ (depending on the likelihood type) for Gaussian data with no covariates and for all non-Gaussian likelihoods.
		//			For Gaussian data with covariates, the response variables is saved in y_vec_ and y_ is replaced by y - X * beta during the optimization
		/*! \brief Key: labels of independent realizations of REs/GPs, value: Psi^-1*y_ (used for various computations) */
		std::map<data_size_t, vec_t> y_aux_;
		/*! \brief Key: labels of independent realizations of REs/GPs, value: L^-1 * Z^T * y, L = chol(Sigma^-1 + Z^T * Z) (used for various computations when only_grouped_REs_use_woodbury_identity_==true) */
		std::map<data_size_t, vec_t> y_tilde_;
		/*! \brief Key: labels of independent realizations of REs/GPs, value: Z * L ^ -T * L ^ -1 * Z ^ T * y, L = chol(Sigma^-1 + Z^T * Z) (used for various computations when only_grouped_REs_use_woodbury_identity_==true) */
		std::map<data_size_t, vec_t> y_tilde2_;
		/*! \brief Indicates whether y_aux_ has been calculated */
		bool y_aux_has_been_calculated_ = false;
		/*! \brief If true, the response variable data has been set (otherwise y_ is empty) */
		bool y_has_been_set_ = false;

		// GROUPED RANDOM EFFECTS
		/*! \brief Number of grouped (intercept) random effects components */
		data_size_t num_re_group_ = 0;
		/*! \brief Number of categorical grouping variables that define grouped random effects (can be different from num_re_group_ in case intercept effects are dropped when having random coefficients)  */
		data_size_t num_group_variables_ = 0;
		/*! \brief Number of grouped random coefficients components */
		data_size_t num_re_group_rand_coef_ = 0;
		/*! \brief Indices that relate every random coefficients to a "base" intercept grouped random effect component. Counting starts at 1 (and ends at the number of base intercept random effects). Length of vector = num_re_group_rand_coef_. */
		std::vector<int> ind_effect_group_rand_coef_;
		/*! \brief Indicates whether intercept random effects are dropped (only for random coefficients). If drop_intercept_group_rand_effect_[k] is true, the intercept random effect number k is dropped. Only random effects with random slopes can be dropped. Length of vector = num_re_group_. */
		std::vector<bool> drop_intercept_group_rand_effect_;
		/*! \brief Total number of grouped random effects components (random intercepts plus random coefficients (slopes)) */
		data_size_t num_re_group_total_ = 0;

		// GAUSSIAN PROCESS
		/*! \brief 1 if there is a Gaussian process 0 otherwise */
		data_size_t num_gp_ = 0;
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
		/*! \brief Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. For the Wendland covariance function, we follow the notation of Bevilacqua et al. (2019, AOS)). This parameter is irrelevant for some covariance functions such as the exponential or Gaussian. */
		double cov_fct_shape_ = 0.;
		/*! \brief Range parameter of the Wendland covariance functionand Wendland correlation taper. We follow the notation of Bevilacqua et al. (2019, AOS)) */
		double cov_fct_taper_range_ = 1.;
		/*! \brief Shape parameter of the Wendland correlation taper. We follow the notation of Bevilacqua et al. (2019, AOS) */
		double cov_fct_taper_shape_ = 0.;
		/*! \brief If true, there are duplicates in coords among the neighbors (currently only used for the Vecchia approximation for non-Gaussian likelihoods) */
		bool has_duplicates_coords_ = false;
		/*! \brief Type of GP-approximation for handling large data */
		string_t gp_approx_ = "none";
		/*! \brief List of supported optimizers for covariance parameters */
		const std::set<string_t> SUPPORTED_GP_APPROX_{ "none", "vecchia", "tapering", "fitc", "full_scale_tapering" };

		// RANDOM EFFECT / GP COMPONENTS
		/*! \brief Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components */
		std::map<data_size_t, std::vector<std::shared_ptr<RECompBase<T_mat>>>> re_comps_;
		/*! \brief Indices of parameters of RE components in global parameter vector cov_pars. ind_par_[i] and ind_par_[i+1] -1 are the indices of the first and last parameter of component number i (counting starts at 1) */
		std::vector<data_size_t> ind_par_;
		/*! \brief Number of covariance parameters */
		data_size_t num_cov_par_;
		/*! \brief Total number of random effect components (grouped REs plus other GPs) */
		data_size_t num_comps_total_ = 0;

		// SPECIAL CASES OF RE MODELS FOR FASTER CALCULATIONS
		/*! \brief If true, the Woodbury, Sherman and Morrison matrix inversion formula is used for calculating the inverse of the covariance matrix (only used if there are only grouped REs and no Gaussian processes) */
		bool only_grouped_REs_use_woodbury_identity_ = false;
		/*! \brief True if there is only one grouped random effect component, and (all) calculations are done on the b-scale instead of the Zb-scale (this flag is only used for non-Gaussian likelihoods) */
		bool only_one_grouped_RE_calculations_on_RE_scale_ = false;
		/*! \brief True if there is only one grouped random effect component for Gaussian data, can calculations for predictions (only) are done on the b-scale instead of the Zb-scale (this flag is only used for Gaussian likelihoods) */
		bool only_one_grouped_RE_calculations_on_RE_scale_for_prediction_ = false;
		/*! \brief True if there is only one GP random effect component, and calculations are done on the b-scale instead of the Zb-scale (only for non-Gaussian likelihoods) */
		bool only_one_GP_calculations_on_RE_scale_ = false;

		// COVARIANCE MATRIX AND CHOLESKY FACTORS OF IT
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Cholesky decomposition of covariance matrices */
		std::map<data_size_t, T_chol> chol_facts_;
		/*! \brief  Key: labels of independent realizations of REs/GPs, values: Square root of diagonal of matrix Sigma^-1 + Zt * Z  (used only if there is only one grouped random effect and ZtZ is diagonal) */
		std::map<data_size_t, vec_t> sqrt_diag_SigmaI_plus_ZtZ_;
		/*! \brief Indicates whether the covariance matrix has been factorized or not */
		bool covariance_matrix_has_been_factorized_ = false;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Idendity matrices used for calculation of inverse covariance matrix */
		std::map<data_size_t, T_mat> Id_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Permuted idendity matrices used for calculation of inverse covariance matrix when Cholesky factors have a permutation matrix */
		std::map<data_size_t, T_mat> P_Id_;
		/*! \brief Indicates whether a symbolic decomposition for calculating the Cholesky factor of the covariance matrix has been done or not (only for sparse matrices) */
		bool chol_fact_pattern_analyzed_ = false;
		/*! \brief Collects inverse covariance matrices Psi^{-1} (usually not saved, but used e.g. in Fisher scoring without the Vecchia approximation) */
		std::map<data_size_t, T_mat> psi_inv_;
		/*! \brief Inverse covariance matrices Sigma^-1 of random effects. This is only used if only_grouped_REs_use_woodbury_identity_==true (if there are only grouped REs) */
		std::map<data_size_t, sp_mat_t> SigmaI_;
		/*! \brief Pointer to covariance matrix of the random effects (sum of all components). This is only used for non-Gaussian likelihoods and if only_grouped_REs_use_woodbury_identity_==false. In the Gaussian case this needs not be saved */
		std::map<data_size_t, std::shared_ptr<T_mat>> ZSigmaZt_;

		// COVARIATE DATA FOR LINEAR REGRESSION TERM
		/*! \brief If true, the model linearly incluses covariates */
		bool has_covariates_ = false;
		/*! \brief Number of covariates */
		int num_coef_;
		/*! \brief Covariate data */
		den_mat_t X_;
		/*! \brief Number of coefficients that are printed out when trace / logging is activated */
		const int NUM_COEF_PRINT_TRACE_ = 5;
		/*! \brief True if X_ contains an intercept column */
		bool has_intercept_;
		/*! \brief Index of the intercept column in X_ */
		int intercept_col_;
		/*! \brief If true, X_ and the linear regression covariates are scaled */
		bool scale_covariates_;
		/*! \brief Location transformation if covariate data is scaled */
		vec_t loc_transf_;
		/*! \brief Scale transformation if covariate data is scaled */
		vec_t scale_transf_;

		// OPTIMIZER PROPERTIES
		/*! \brief Optimizer for covariance parameters */
		string_t optimizer_cov_pars_ = "gradient_descent";
		/*! \brief List of supported optimizers for covariance parameters */
		const std::set<string_t> SUPPORTED_OPTIM_COV_PAR_{ "gradient_descent", "fisher_scoring", "newton", "nelder_mead", "bfgs", "adam", "bfgs_v2", "bfgs_v3", "lbfgs_linesearch_nocedal_wright" };
		/*! \brief Convergence criterion for terminating the 'OptimLinRegrCoefCovPar' optimization algorithm */
		string_t convergence_criterion_ = "relative_change_in_log_likelihood";
		/*! \brief List of supported convergence criteria used for terminating the optimization algorithm */
		const std::set<string_t> SUPPORTED_CONV_CRIT_{ "relative_change_in_parameters", "relative_change_in_log_likelihood" };
		/*! \brief Maximal number of iterations for covariance parameter and linear regression parameter estimation */
		int max_iter_ = 1000;
		/*!
		\brief Convergence tolerance for covariance and linear regression coefficient estimation.
			The algorithm stops if the relative change in either the (approximate) log-likelihood or the parameters is below this value.
			For "bfgs", the L2 norm of the gradient is used instead of the relative change in the log-likelihood.
			If delta_rel_conv_init_ < 0, internal default values are set in 'InitializeLearningRatesConvTol'
		*/
		double delta_rel_conv_;
		/*! \brief Initial convergence tolerance (to remember as default values for delta_rel_conv_ are different for 'nelder_mead' vs. other optimizers and the optimization might get restarted) */
		double delta_rel_conv_init_ = -1;
		/*! \brief Learning rate for covariance parameters. If lr_cov_init_ < 0, internal default values are set in 'InitializeLearningRatesConvTol' */
		double lr_cov_;
		/*! \brief Initial learning rate for covariance parameters (to remember as lr_cov_ can be decreased) */
		double lr_cov_init_ = -1;
		/*! \brief True if 'lr_cov_' and other learning rates have been initialized, i.e., if 'InitializeLearningRatesConvTol' has been called */
		bool lr_have_been_initialized_ = false;
		/*! \brief Learning rate for covariance parameters after first iteration (to remember as lr_cov_ can be decreased) */
		double lr_cov_after_first_iteration_ = 0.1;
		/*! \brief Learning rate for auxiliary parameters for non-Gaussian likelihoods (e.g., shape of a gamma likelihood) */
		double lr_aux_pars_;
		/*! \brief Initial learning rate for auxiliary parameters for non-Gaussian likelihoods (e.g., shape of a gamma likelihood) */
		double lr_aux_pars_init_ = 0.1;
		/*! \brief Learning rate for auxiliary parameters after first iteration (to remember as lr_cov_ can be decreased) */
		double lr_aux_pars_after_first_iteration_ = 0.1;
		/*! \brief Indicates whether Nesterov acceleration is used in the gradient descent for finding the covariance parameters (only used for "gradient_descent") */
		bool use_nesterov_acc_ = true;
		/*! \brief Acceleration rate for covariance parameters for Nesterov acceleration (only relevant if use_nesterov_acc and nesterov_schedule_version == 0) */
		double acc_rate_cov_ = 0.5;
		/*! \brief Number of iterations for which no mometum is applied in the beginning (only relevant if use_nesterov_acc) */
		int momentum_offset_ = 2;
		/*! \brief Select Nesterov acceleration schedule 0 or 1 */
		int nesterov_schedule_version_ = 0;
		/*! \brief Optimizer for linear regression coefficients (The default = "wls" is changed to "gradient_descent" for non-Gaussian likelihoods upon initialization). See the constructor REModelTemplate() for the default values which depend on whether the likelihood is Gaussian or not */
		string_t optimizer_coef_;
		/*! \brief List of supported optimizers for regression coefficients for Gaussian likelihoods */
		const std::set<string_t> SUPPORTED_OPTIM_COEF_GAUSS_{ "gradient_descent", "wls", "nelder_mead", "bfgs", "adam", "bfgs_v2", "bfgs_v3", "lbfgs_linesearch_nocedal_wright" };
		/*! \brief List of supported optimizers for regression coefficients for non-Gaussian likelihoods */
		const std::set<string_t> SUPPORTED_OPTIM_COEF_NONGAUSS_{ "gradient_descent", "nelder_mead", "bfgs", "adam", "bfgs_v2", "bfgs_v3", "lbfgs_linesearch_nocedal_wright" };
		/*! \brief Learning rate for fixed-effect linear coefficients */
		double lr_coef_;
		/*! \brief Initial learning rate for fixed-effect linear coefficients (to remember as lr_coef_ can be decreased) */
		double lr_coef_init_ = 0.1;
		/*! \brief Learning rate for fixed-effect linear coefficients after first iteration (to remember as lr_coef_ can be decreased) */
		double lr_coef_after_first_iteration_ = 0.1;
		/*! \brief Acceleration rate for coefficients for Nesterov acceleration (only relevant if use_nesterov_acc and nesterov_schedule_version == 0) */
		double acc_rate_coef_ = 0.5;
		/*! \brief Maximal number of steps for which learning rate shrinkage is done for gradient-based optimization of covariance parameters and regression coefficients */
		int MAX_NUMBER_LR_SHRINKAGE_STEPS_ = 30;
		/*! \brief Learning rate shrinkage factor for gradient-based optimization of covariance parameters and regression coefficients */
		double LR_SHRINKAGE_FACTOR_ = 0.5;
		/*! \brief Threshold value for a learning rate below which a learning rate might be increased again (only in case there are also regression coefficients and for gradient descent optimization of covariance parameters and regression coefficients) */
		double lr_is_small_threshold_cov_ = 1e-6;
		/*! \brief Threshold value for a learning rate below which a learning rate might be increased again (only in case there are also regression coefficients and for gradient descent optimization of covariance parameters and regression coefficients) */
		double lr_is_small_threshold_coef_ = 1e-6;
		/*! \brief Threshold value for a learning rate below which a learning rate might be increased again (only in case there are also regression coefficients and for gradient descent optimization of covariance parameters and regression coefficients) */
		double lr_is_small_threshold_aux_ = 1e-6;
		/*! \brief Threshold value for relative change in parameters below which a learning rate might be increased again (only in case there are also regression coefficients and for gradient descent optimization of covariance parameters and regression coefficients) */
		double LR_IS_SMALL_REL_CHANGE_IN_PARS_THRESHOLD_ = 1e-4;
		/*! \brief Threshold value for relative change in other parameters above which a learning rate is again set to its initial value (only in case there are also regression coefficients and for gradient descent optimization of covariance parameters and regression coefficients) */
		double MIN_REL_CHANGE_IN_OTHER_PARS_FOR_RESETTING_LR_ = 1e-2;
		/*! \brief true if 'optimizer_coef_' has been set */
		bool coef_optimizer_has_been_set_ = false;
		/*! \brief List of optimizers which are externally handled by OptimLib */
		const std::set<string_t> OPTIM_EXTERNAL_{ "nelder_mead", "bfgs", "adam", "bfgs_v2", "lbfgs_linesearch_nocedal_wright" };
		/*! \brief If true, any additional parameters for non-Gaussian likelihoods are also estimated (e.g., shape parameter of gamma likelihood) */
		bool estimate_aux_pars_ = false;
		/*! \brief True if the function 'SetOptimConfig' has been called */
		bool set_optim_config_has_been_called_ = false;
		/*! \brief If true, the covariance parameters or linear coefficients were updated for the first time with gradient descent*/
		bool first_update_ = false;
		/*! \brief Number of likelihood evaluations during optimization */
		int num_ll_evaluations_ = 0;
		/*! \brief Number of iterations during optimization */
		int num_iter_ = 0;
		/*! \brief True, if 'OptimLinRegrCoefCovPar' has been called */
		bool model_has_been_estimated_ = false;
		/*! \brief Maximal relative change for covariance parameters in one iteration */
		int MAX_REL_CHANGE_GRADIENT_UPDATE_ = 100; // allow maximally a change by a factor of 'MAX_REL_CHANGE_GRADIENT_UPDATE_' in one iteration
		/*! \brief Maximal value of gradient updates on log-scale for covariance parameters */
		double MAX_GRADIENT_UPDATE_LOG_SCALE_ = std::log((double)MAX_REL_CHANGE_GRADIENT_UPDATE_);
		/*! \brief Maximal relative change for for auxiliary parameters in one iteration */
		int MAX_REL_CHANGE_GRADIENT_UPDATE_AUX_PARS_ = 100;
		/*! \brief Maximal value of gradient updates on log-scale for auxiliary parameters */
		double MAX_GRADIENT_UPDATE_LOG_SCALE_AUX_PARS_ = std::log((double)MAX_REL_CHANGE_GRADIENT_UPDATE_AUX_PARS_);
		/*! \brief Constant C_mu used for checking whether step sizes for linear regression coefficients are clearly too large */
		double C_mu_;
		/*! \brief Constant C_sigma2_ used for checking whether step sizes for linear regression coefficients are clearly too large */
		double C_sigma2_;
		/*! \brief Constant used for checking whether step sizes for linear regression coefficients are clearly too large */
		double C_MAX_CHANGE_COEF_ = 10.;
		/*! \brief True if 'lr_cov_' and other learning rates have been etimated before in a previous boosting iteration (applies only to the GPBoost algorithm) */
		bool cov_pars_have_been_estimated_once_ = false;
		/*! \brief True if 'lr_cov_' and other learning rates have been doubled in the first optimization iteration (num_iter_ == 0) (applies only to the GPBoost algorithm) */
		bool learning_rates_have_been_doubled_in_first_iteration_ = false;

		/*! \brief If true, Armijo's condition is used to check whether there is sufficient decrease in the negative log-likelighood (otherwise it is only checked for a decrease) */
		bool armijo_condition_ = true;
		/*! \brief Constant c for Armijo's condition. Needs to be in (0,1) */
		double c_armijo_ = 1e-4;
		/*! \brief Constant c for Armijo's condition for the Nesterov momentum part. Needs to be in (0,1) */
		double c_armijo_mom_ = 1e-4;
		/*! \brief Directional derivative wrt covariance parameters for Armijo / Wolfe condition */
		double dir_deriv_armijo_cov_pars_;
		/*! \brief Directional derivativet wrt auxiliary coefficients for Armijo / Wolfe condition */
		double dir_deriv_armijo_aux_pars_;
		/*! \brief Directional derivative linear regression coefficients for Armijo / Wolfe condition */
		double dir_deriv_armijo_coef_;
		/*! \brief Momentum term directional derivative wrt covariance parameters for Armijo / Wolfe condition */
		double mom_dir_deriv_armijo_cov_pars_;
		/*! \brief Momentum term directional derivativet wrt auxiliary coefficients for Armijo / Wolfe condition */
		double mom_dir_deriv_armijo_aux_pars_;
		/*! \brief Momentum term directional derivative linear regression coefficients for Armijo / Wolfe condition */
		double mom_dir_deriv_armijo_coef_;
		/*! \brief If true, the initial learning rates in every iteration are set such that there is a constant first order change */
		bool learning_rate_constant_first_order_change_ = false;
		/*! \brief If true, the learning rates are reset to initial values in every iteration (only for gradient_descent) */
		bool reset_learning_rate_every_iteration_ = false;
		/*! \brief If true, the learning rates can be increased again in latter iterations after they have been decreased (only for gradient_descent) */
		bool increase_learning_rate_again_ = false;
		/*! \brief If true, the learning rates have been descreased (only for gradient_descent) */
		bool learning_rate_decreased_first_time_ = false;
		/*! \brief If true, the learning rates have been increased after they have been descreased (only for gradient_descent) */
		bool learning_rate_increased_after_descrease_ = false;
		/*! \brief If true, the learning rates have been descreased again after they have been increased (only for gradient_descent) */
		bool learning_rate_decreased_after_increase_ = false;
		/*! \brief Threshold value, for relative change in the log-likelihood, below which learning rates are increased again for gradient descent */
		double INCREASE_LR_CHANGE_LL_THRESHOLD_ = 1e-3;
		/*! \brief If true, the nugget effect is profiled out for Gaussian likelihoods (=use closed-form expression for error / nugget variance) */
		bool profile_out_marginal_variance_ = false;

		// MATRIX INVERSION PROPERTIES
		/*! \brief Matrix inversion method */
		string_t matrix_inversion_method_ = "cholesky";
		/*! \brief Supported matrix inversion methods */
		const std::set<string_t> SUPPORTED_MATRIX_INVERSION_METHODS_{ "cholesky", "iterative" };
		/*! \brief Maximal number of iterations for conjugate gradient algorithm */
		int cg_max_num_it_ = 1000;
		/*! \brief Maximal number of iterations for conjugate gradient algorithm when being run as Lanczos algorithm for tridiagonalization */
		int cg_max_num_it_tridiag_ = 1000;
		/*! \brief Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for parameter estimation */
		double cg_delta_conv_ = 1e-2;
		/*! \brief Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for prediction */
		double cg_delta_conv_pred_ = 1e-3;
		/*! \brief Threshold to avoid numerical instability in the CG: If the L1-norm of the rhs is below the defined threshold the CG is not executed and a vector of 0's is returned */
		const double THRESHOLD_ZERO_RHS_CG_ = 1.0e-100;
		/*! \brief Number of samples when simulation is used for calculating predictive variances */
		int nsim_var_pred_ = 1000;
		/*! \brief Number of random vectors (e.g., Rademacher) for stochastic approximation of the trace of a matrix */
		int num_rand_vec_trace_ = 50;
		/*! \brief If true, random vectors (e.g., Rademacher) for stochastic approximation of the trace of a matrix are sampled only once at the beginning of Newton's method for finding the mode in the Laplace approximation and are then reused in later trace approximations, otherwise they are sampled every time a trace is calculated */
		bool reuse_rand_vec_trace_ = true;
		/*! \brief Seed number to generate random vectors (e.g., Rademacher) */
		int seed_rand_vec_trace_ = 1;
		/*! If the seed of the random number generator cg_generator_ is set, cg_generator_seeded_ is set to true*/
		bool cg_generator_seeded_ = false;
		/*! Indicates if we observe a NAN or Inf value in a conjugate gradient iteration */
		bool NaN_found = false;
		/*! Random number generator used generate Rademacher-vectors in GenRandVecTrace() and GenRandVecDiag()*/
		RNG_t cg_generator_;
		/*! Matrix of random Rademacher vectors (u_1,...,u_t), where u_i is of dimension n & Cov(u_i) = I*/
		std::map<data_size_t, den_mat_t> rand_vec_probe_;
		/*! Matrix of random Rademacher vectors (u_1,...,u_t), where u_i is of dimension n & Cov(u_i) = I or Cov(u_i) = P*/
		std::map<data_size_t, den_mat_t> rand_vec_probe_P_;
		/*! Matrix of random Rademacher vectors (u_1,...,u_t), where u_i is of dimension m & Cov(u_i) = I*/
		std::map<data_size_t, den_mat_t> rand_vec_probe_low_rank_;
		/*! Matrix of random Rademacher vectors for Fisher information (u_1,...,u_t), where u_i is of dimension n & Cov(u_i) = I*/
		std::map<data_size_t, den_mat_t> rand_vec_fisher_info_;
		/*! If reuse_rand_vec_trace_ is true and random probe vectors have been generated for the first time, then saved_rand_vec_ is set to true  */
		std::map<data_size_t, bool> saved_rand_vec_;
		/*! If reuse_rand_vec_trace_ is true and random probe vectors have been generated for the first time, then saved_rand_vec_ is set to true  */
		std::map<data_size_t, bool> saved_rand_vec_fisher_info_;
		/*! CG solution for Rademacher vectors:	Sigma^-1 (u_1,...,u_t) */
		std::map<data_size_t, den_mat_t> solution_for_trace_;
		/*! \brief Type of preconditioner used for conjugate gradient algorithms */
		string_t cg_preconditioner_type_;
		/*! \brief List of supported preconditioners for conjugate gradient algorithms for Gaussian likelihood */
		const std::set<string_t> SUPPORTED_CG_PRECONDITIONER_TYPE_GAUSS_{ "none", "predictive_process_plus_diagonal" };
		/*! \brief List of supported preconditioners for conjugate gradient algorithms for non-Gaussian likelihoods */
		const std::set<string_t> SUPPORTED_CG_PRECONDITIONER_TYPE_NONGAUSS_{ "Sigma_inv_plus_BtWB", "piv_chol_on_Sigma" };
		/*! \brief true if 'cg_preconditioner_type_' has been set */
		bool cg_preconditioner_type_has_been_set_ = false;
		/*! \brief Rank of the pivoted Cholesky decomposition used as preconditioner in conjugate gradient algorithms */
		int piv_chol_rank_ = 50;
		/*! \brief Rank of the matrix for approximating predictive covariances obtained using the Lanczos algorithm */
		int rank_pred_approx_matrix_lanczos_ = 1000;

		// WOODBURY IDENTITY FOR GROUPED RANDOM EFFECTS ONLY
		/*! \brief Collects matrices Z^T (only saved when only_grouped_REs_use_woodbury_identity_=true i.e. when there are only grouped random effects, otherwise these matrices are saved only in the indepedent RE components) */
		std::map<data_size_t, sp_mat_t> Zt_;
		/*! \brief Collects matrices Z^TZ (only saved when only_grouped_REs_use_woodbury_identity_=true i.e. when there are only grouped random effects, otherwise these matrices are saved only in the indepedent RE components) */
		std::map<data_size_t, sp_mat_t> ZtZ_;
		/*! \brief Collects vectors Z^Ty (only saved when only_grouped_REs_use_woodbury_identity_=true i.e. when there are only grouped random effects) */
		std::map<data_size_t, vec_t> Zty_;
		/*! \brief Cumulative number of random effects for components (usually not saved, only saved when only_grouped_REs_use_woodbury_identity_=true i.e. when there are only grouped random effects, otherwise these matrices are saved only in the indepedent RE components) */
		std::map<data_size_t, std::vector<data_size_t>> cum_num_rand_eff_;//The random effects of component j start at cum_num_rand_eff_[0][j]+1 and end at cum_num_rand_eff_[0][j+1]
		/*! \brief Sum of squared entries of Z_j for every random effect component (usually not saved, only saved when only_grouped_REs_use_woodbury_identity_=true i.e. when there are only grouped random effects) */
		std::map<data_size_t, std::vector<double>> Zj_square_sum_;
		/*! \brief Collects matrices Z^T * Z_j for every random effect component (usually not saved, only saved when only_grouped_REs_use_woodbury_identity_=true i.e. when there are only grouped random effects) */
		std::map<data_size_t, std::vector<sp_mat_t>> ZtZj_;
		/*! \brief Collects matrices L^-1 * Z^T * Z_j for every random effect component (usually not saved, only saved when only_grouped_REs_use_woodbury_identity_=true i.e. when there are only grouped random effects and when Fisher scoring is done) */
		std::map<data_size_t, std::vector<T_mat>> LInvZtZj_;
		/*! \brief Permuted matrices Zt_ when Cholesky factors have a permutation matrix */
		std::map<data_size_t, sp_mat_t> P_Zt_;
		/*! \brief Permuted matrices ZtZj_ when Cholesky factors have a permutation matrix */
		std::map<data_size_t, std::vector<sp_mat_t>> P_ZtZj_;

		// VECCHIA APPROXIMATION for GP
		/*! \brief If true, a memory optimized version of the Vecchia approximation is used (at the expense of being slightly slower). THiS IS CURRENTLY NOT IMPLEMENTED */
		bool vecchia_approx_optim_memory = false;
		/*! \brief The number of neighbors used in the Vecchia approximation */
		int num_neighbors_;
		/*! \brief Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering */
		string_t vecchia_ordering_ = "random";
		/*! \brief List of supported options for orderings of the Vecchia approximation */
		const std::set<string_t> SUPPORTED_VECCHIA_ORDERING_{ "none", "random", "time", "time_random_space" };
		/*! \brief The way how neighbors are selected */
		string_t vecchia_neighbor_selection_ = "nearest";
		/*! \brief The number of neighbors used in the Vecchia approximation for making predictions */
		int num_neighbors_pred_;
		/*!
		* \brief Ordering used in the Vecchia approximation for making predictions
		* "order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points & Vecchia approximation is done for observable process (only for Gaussian likelihoods)
		* "order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted) & Vecchia approximation is done for observable process (only for Gaussian likelihoods)
		* "order_pred_first" = predicted data is ordered first for making prediction & Vecchia approximation is done for observable process (only for Gaussian likelihoods)
		* "latent_order_obs_first_cond_obs_only" = observed data is ordered first and neighbors are only observed points & Vecchia approximation is done for latent process
		* "latent_order_obs_first_cond_all" = observed data is ordered first and neighbors are selected among all points (observed + predicted) & Vecchia approximation is done for latent process
		* See the constructor REModelTemplate() for the default values which depend on whether the likelihood is Gaussian or not
		*/
		string_t vecchia_pred_type_; //This is saved and not simply set in the prediction function since it needs to be used repeatedly in the GPBoost algorithm when making predictions in "regression_metric.hpp" and the way predictions are done for the Vecchia approximation should be decoupled from the boosting algorithm
		/*! \brief List of supported options for prediction with a Vecchia approximation for Gaussian likelihoods */
		const std::set<string_t> SUPPORTED_VECCHIA_PRED_TYPES_GAUSS_{ "order_obs_first_cond_obs_only",
			"order_obs_first_cond_all", "order_pred_first",
			"latent_order_obs_first_cond_obs_only", "latent_order_obs_first_cond_all" };
		/*! \brief List of supported options for prediction with a Vecchia approximation for non-Gaussian likelihoods */
		const std::set<string_t> SUPPORTED_VECCHIA_PRED_TYPES_NONGAUSS_{ "latent_order_obs_first_cond_obs_only",
			"latent_order_obs_first_cond_all", "order_obs_first_cond_obs_only", "order_obs_first_cond_all" };
		/*! \brief Collects indices of nearest neighbors (used for Vecchia approximation) */
		std::map<data_size_t, std::vector<std::vector<int>>> nearest_neighbors_;
		/*! \brief Distances between locations and their nearest neighbors (this is used only if the Vecchia approximation is used, otherwise the distances are saved directly in the base GP component) */
		std::map<data_size_t, std::vector<den_mat_t>> dist_obs_neighbors_;
		/*! \brief Distances between nearest neighbors for all locations (this is used only if the Vecchia approximation is used, otherwise the distances are saved directly in the base GP component) */
		std::map<data_size_t, std::vector<den_mat_t>> dist_between_neighbors_;//Note: this contains duplicate information (i.e. distances might be saved reduntly several times). But there is a trade-off between storage and computational speed. I currently don't see a way for saving unique distances without copying them when using them.
		/*! \brief Outer product of covariate vector at observations and neighbors with itself. First index = cluster, second index = data point i, third index = GP number j (this is used only if the Vecchia approximation is used, this is handled saved directly in the GP component using Z_) */
		std::map<data_size_t, std::vector<std::vector<den_mat_t>>> z_outer_z_obs_neighbors_;
		/*! \brief Collects matrices B = I - A (=Cholesky factor of inverse covariance) for Vecchia approximation */
		std::map<data_size_t, sp_mat_t> B_;
		/*! \brief Collects diagonal matrices D^-1 for Vecchia approximation */
		std::map<data_size_t, sp_mat_t> D_inv_;
		/*! \brief Collects derivatives of matrices B ( = derivative of matrix -A) for Vecchia approximation */
		std::map<data_size_t, std::vector<sp_mat_t>> B_grad_;
		/*! \brief Collects derivatives of matrices D for Vecchia approximation */
		std::map<data_size_t, std::vector<sp_mat_t>> D_grad_;
		/*! \brief Triplets for initializing the matrices B */
		std::map<data_size_t, std::vector<Triplet_t>> entries_init_B_;
		/*! \brief Triplets for initializing the matrices B_grad */
		std::map<data_size_t, std::vector<Triplet_t>> entries_init_B_grad_;
		/*! \brief If true, the function 'SetVecchiaPredType' has been called and vecchia_pred_type_ has been set */
		bool vecchia_pred_type_has_been_set_ = false;
		/*! \brief If true, a stochastic trace approximation is used to calculate the Fisher information for a Vecchia approximation for Gaussian likelihoods */
		bool use_stochastic_trace_for_Fisher_information_Vecchia_ = true;

		// PREDICTIVE PROCESS AND FULL SCALE APPROXIMATION FOR GP
		/*! \brief Method for choosing inducing points */
		string_t ind_points_selection_ = "kmeans++";
		/*! \brief List of supported inducing point methods*/
		const std::set<string_t> SUPPORTED_METHOD_INDUCING_POINTS_{ "random", "kmeans++", "cover_tree" };
		/*! \brief Number of inducing points */
		int num_ind_points_;
		/*! \brief Radius (= "spatial resolution") for the cover tree algorithm */
		double cover_tree_radius_;
		/*! \brief Keys: labels of independent realizations of REs/GPs, values: vectors with inducing points GP components */
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<den_mat_t>>>> re_comps_ip_;
		/*! \brief Keys: labels of independent realizations of REs/GPs, values: vectors with cross-covariance GP components */
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<den_mat_t>>>> re_comps_cross_cov_;
		/*! \brief Keys: labels of independent realizations of REs/GPs, values: vectors with residual GP components with sparse (tapered) covariances */
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<T_mat>>>> re_comps_resid_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Cholesky decompositions of inducing points matrix sigma_ip */
		std::map<data_size_t, chol_den_mat_t> chol_fact_sigma_ip_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Inverse of Cholesky factor of inducing points matrix sigma_ip times cross-covariance */
		std::map<data_size_t, den_mat_t> chol_ip_cross_cov_;

		std::map<data_size_t, den_mat_t> sigma_inv_sigma_grad_rand_vec_;
		std::map<data_size_t, den_mat_t> sigma_grad_sigma_inv_rand_vec_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Inverse of Cholesky factor of inducing points matrix sigma_ip times cross-covariance */
		std::map<data_size_t, den_mat_t> resid_inv_cross_cov_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: diagonal of fully independent training conditional for predictive process */
		std::map<data_size_t, vec_t> FITC_Diag_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Cholesky decompositions of residual covariance matrix */
		std::map<data_size_t, T_chol> chol_fact_resid_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Cholesky decompositions of matrix sigma_ip + cross_cov^T * sigma_resid^-1 * cross_cov used in Woodbury identity */
		std::map<data_size_t, chol_den_mat_t> chol_fact_sigma_woodbury_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Diagonal of residual covariance matrix (Preconditioner) */
		std::map<data_size_t, vec_t> diagonal_approx_preconditioner_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Inverse of diagonal of residual covariance matrix (Preconditioner) */
		std::map<data_size_t, vec_t> diagonal_approx_inv_preconditioner_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Cholesky decompositions of matrix sigma_ip + cross_cov^T * D^-1 * cross_cov used in Woodbury identity where D is given by the Preconditioner */
		std::map<data_size_t, chol_den_mat_t> chol_fact_woodbury_preconditioner_;


		// CLUSTERs of INDEPENDENT REALIZATIONS
		/*! \brief Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points */
		std::map<data_size_t, std::vector<int>> data_indices_per_cluster_;
		/*! \brief Keys: Labels of independent realizations of REs/GPs, values: number of data points per independent realization */
		std::map<data_size_t, int> num_data_per_cluster_;
		/*! \brief Number of independent realizations of the REs/GPs */
		data_size_t num_clusters_;
		/*! \brief Unique labels of independent realizations */
		std::vector<data_size_t> unique_clusters_;

		/*! \brief Variance of idiosyncratic error term (nugget effect) */
		double sigma2_;
		/*! \brief Quadratic form y^T Psi^-1 y (saved for avoiding double computations when profiling out sigma2 for Gaussian data) */
		double yTPsiInvy_;
		/*! \brief Determinant of Psi (to avoid double computations) */
		double log_det_Psi_;

		// PREDICTION
		/*! \brief Cluster IDs for prediction */
		std::vector<data_size_t> cluster_ids_data_pred_;
		/*! \brief Levels of grouped RE for prediction */
		std::vector<std::vector<re_group_t>> re_group_levels_pred_;
		/*! \brief Covariate data for grouped random RE for prediction */
		std::vector<double> re_group_rand_coef_data_pred_;
		/*! \brief Coordinates for GP for prediction */
		std::vector<double> gp_coords_data_pred_;
		/*! \brief Covariate data for random GP for prediction */
		std::vector<double> gp_rand_coef_data_pred_;
		/*! \brief Covariate data for linear regression term */
		std::vector<double> covariate_data_pred_;
		/*! \brief Number of prediction points */
		data_size_t num_data_pred_;

		/*! Random number generator */
		RNG_t rng_;

		/*! \brief Nesterov schedule */
		static double NesterovSchedule(int iter,
			int momentum_schedule_version,
			double nesterov_acc_rate,
			int momentum_offset) {
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
					Log::REFatal("NesterovSchedule: version = %d is not supported ", momentum_schedule_version);
				}
			}
			return(0.);
		}//end NesterovSchedule

		/*! \brief mutex for threading safe call */
		std::mutex mutex_;

		/*! \brief Constructs identity matrices if sparse matrices are used (used for calculating inverse covariance matrix) */
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
		void ConstructI(data_size_t cluster_i) {
			int dim_I = only_grouped_REs_use_woodbury_identity_ ? cum_num_rand_eff_[cluster_i][num_re_group_total_] : num_data_per_cluster_[cluster_i];
			T_mat I(dim_I, dim_I);//identity matrix for calculating precision matrix
			I.setIdentity();
			I.makeCompressed();
			Id_.insert({ cluster_i, I });
		}

		/*! \brief Constructs identity matrices if dense matrices are used (used for calculating inverse covariance matrix) */
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<den_mat_t, T_aux>::value>::type* = nullptr >
		void ConstructI(data_size_t cluster_i) {
			int dim_I = only_grouped_REs_use_woodbury_identity_ ? cum_num_rand_eff_[cluster_i][num_re_group_total_] : num_data_per_cluster_[cluster_i];
			den_mat_t I(dim_I, dim_I);//identity matrix for calculating precision matrix
			I.setIdentity();
			Id_.insert({ cluster_i, I });
		}

		/*!
		* \brief Set response variable data y_ (and calculate Z^T * y if  only_grouped_REs_use_woodbury_identity_ == true)
		* \param y_data Response variable data
		*/
		void SetY(const double* y_data) {
			if (gauss_likelihood_) {
				if (num_clusters_ == 1 && (gp_approx_ != "vecchia" || vecchia_ordering_ == "none")) {
					y_[unique_clusters_[0]] = Eigen::Map<const vec_t>(y_data, num_data_);
				}
				else {
					for (const auto& cluster_i : unique_clusters_) {
						y_[cluster_i] = vec_t(num_data_per_cluster_[cluster_i]);
						for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
							y_[cluster_i][j] = y_data[data_indices_per_cluster_[cluster_i][j]];
						}
					}
				}
				if (only_grouped_REs_use_woodbury_identity_) {
					CalcZtY();
				}
			}//end gauss_likelihood_
			else {//not gauss_likelihood_
				(*likelihood_[unique_clusters_[0]]).template CheckY<double>(y_data, num_data_);
				if (likelihood_[unique_clusters_[0]]->label_type() == "int") {
					for (const auto& cluster_i : unique_clusters_) {
						y_int_[cluster_i] = vec_int_t(num_data_per_cluster_[cluster_i]);
						for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
							y_int_[cluster_i][j] = static_cast<int>(y_data[data_indices_per_cluster_[cluster_i][j]]);
						}
					}
				}
				else if (likelihood_[unique_clusters_[0]]->label_type() == "double") {
					for (const auto& cluster_i : unique_clusters_) {
						y_[cluster_i] = vec_t(num_data_per_cluster_[cluster_i]);
						for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
							y_[cluster_i][j] = y_data[data_indices_per_cluster_[cluster_i][j]];
						}
					}
				}
			}//end not gauss_likelihood_
			y_has_been_set_ = true;
		}

		/*!
		* \brief Set response variable data y_ if data is of type float (used for GPBoost algorithm since labels are float)
		* \param y_data Response variable data
		*/
		void SetY(const float* y_data) {
			if (gauss_likelihood_) {
				Log::REFatal("SetY is not implemented for Gaussian data and lables of type float (since it is not needed)");
			}//end gauss_likelihood_
			else {//not gauss_likelihood_
				(*likelihood_[unique_clusters_[0]]).template CheckY<float>(y_data, num_data_);
				if (likelihood_[unique_clusters_[0]]->label_type() == "int") {
					for (const auto& cluster_i : unique_clusters_) {
						y_int_[cluster_i] = vec_int_t(num_data_per_cluster_[cluster_i]);
						for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
							y_int_[cluster_i][j] = static_cast<int>(y_data[data_indices_per_cluster_[cluster_i][j]]);
						}
					}
				}
				else if (likelihood_[unique_clusters_[0]]->label_type() == "double") {
					for (const auto& cluster_i : unique_clusters_) {
						y_[cluster_i] = vec_t(num_data_per_cluster_[cluster_i]);
						for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
							y_[cluster_i][j] = static_cast<double>(y_data[data_indices_per_cluster_[cluster_i][j]]);
						}
					}
				}
			}
			y_has_been_set_ = true;
		}

		/*!
		* \brief Return (last used) response variable data
		* \param[out] y Response variable data (memory needs to be preallocated)
		*/
		void GetY(double* y) {
			if (!y_has_been_set_) {
				Log::REFatal("Respone variable data has not been set");
			}
			if (has_covariates_ && gauss_likelihood_) {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_; ++i) {
					y[i] = y_vec_[i];
				}
			}
			else if (likelihood_[unique_clusters_[0]]->label_type() == "double") {
				for (const auto& cluster_i : unique_clusters_) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
						y[data_indices_per_cluster_[cluster_i][i]] = y_[cluster_i][i];
					}
				}
			}
			else if (likelihood_[unique_clusters_[0]]->label_type() == "int") {
				for (const auto& cluster_i : unique_clusters_) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_data_per_cluster_[cluster_i]; ++i) {
						y[data_indices_per_cluster_[cluster_i][i]] = y_int_[cluster_i][i];
					}
				}
			}
		}

		/*!
		* \brief Return covariate data
		* \param[out] covariate_data covariate data
		*/
		void GetCovariateData(double* covariate_data) {
			if (!has_covariates_) {
				Log::REFatal("Model does not have covariates for a linear predictor");
			}
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data_ * num_coef_; ++i) {
				covariate_data[i] = X_.data()[i];
			}
		}

		/*!
		* \brief Calculate Z^T*y (use only when only_grouped_REs_use_woodbury_identity_ == true)
		*/
		void CalcZtY() {
			for (const auto& cluster_i : unique_clusters_) {
				Zty_[cluster_i] = Zt_[cluster_i] * y_[cluster_i];
			}
		}

		/*!
		* \brief Get y_aux = Psi^-1*y
		* \param[out] y_aux Psi^-1*y (=y_aux_). Array needs to be pre-allocated of length num_data_
		*/
		void GetYAux(double* y_aux) {
			CHECK(y_aux_has_been_calculated_);
			if (num_clusters_ == 1 && (gp_approx_ != "vecchia" || vecchia_ordering_ == "none")) {
#pragma omp parallel for schedule(static)
				for (int j = 0; j < num_data_; ++j) {
					y_aux[j] = y_aux_[unique_clusters_[0]][j];
				}
			}
			else {
				for (const auto& cluster_i : unique_clusters_) {
#pragma omp parallel for schedule(static)
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
			if (num_clusters_ == 1 && (gp_approx_ != "vecchia" || vecchia_ordering_ == "none")) {
				y_aux = y_aux_[unique_clusters_[0]];
			}
			else {
				for (const auto& cluster_i : unique_clusters_) {
					y_aux(data_indices_per_cluster_[cluster_i]) = y_aux_[cluster_i];
				}
			}
		}

		/*!
		* \brief Calculate Cholesky decomposition
		* \param psi Covariance matrix for which the Cholesky decomposition is calculated
		* \param cluster_i Cluster index for which the Cholesky factor is calculated
		*/
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
		void CalcChol(const T_mat& psi, data_size_t cluster_i) {
			if (!chol_fact_pattern_analyzed_) {
				chol_facts_[cluster_i].analyzePattern(psi);
				if (cluster_i == unique_clusters_.back()) {
					chol_fact_pattern_analyzed_ = true;
				}
				if (CholeskyHasPermutation<T_chol>(chol_facts_[cluster_i])) {
					P_Id_[cluster_i] = chol_facts_[cluster_i].permutationP() * Id_[cluster_i];
					P_Id_[cluster_i].makeCompressed();
					if (only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_) {
						P_Zt_[cluster_i] = chol_facts_[cluster_i].permutationP() * Zt_[cluster_i];
						std::vector<sp_mat_t> P_ZtZj_cluster_i(num_comps_total_);
						for (int j = 0; j < num_comps_total_; ++j) {
							P_ZtZj_cluster_i[j] = chol_facts_[cluster_i].permutationP() * ZtZj_[cluster_i][j];
						}
						P_ZtZj_[cluster_i] = P_ZtZj_cluster_i;
					}
				}
			}
			chol_facts_[cluster_i].factorize(psi);
		}
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<den_mat_t, T_aux>::value>::type* = nullptr >
		void CalcChol(const den_mat_t& psi, data_size_t cluster_i) {
			chol_facts_[cluster_i].compute(psi);
		}

		/*!
		* \brief Calculate Cholesky decomposition of residual process in full scale approximation
		* \param psi Covariance matrix for which the Cholesky decomposition is calculated
		* \param cluster_i Cluster index for which the Cholesky factor is calculated
		*/
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
		void CalcCholFSAResid(const T_mat& psi, data_size_t cluster_i) {
			if (!chol_fact_pattern_analyzed_) {
				chol_fact_resid_[cluster_i].analyzePattern(psi);
				if (cluster_i == unique_clusters_.back()) {
					chol_fact_pattern_analyzed_ = true;
				}
			}
			chol_fact_resid_[cluster_i].factorize(psi);
		}
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<den_mat_t, T_aux>::value>::type* = nullptr >
		void CalcCholFSAResid(const den_mat_t& psi, data_size_t cluster_i) {
			chol_fact_resid_[cluster_i].compute(psi);
		}

		/*!
		* \brief Caclulate Psi^(-1) if sparse matrices are used
		* \param psi_inv[out] Inverse covariance matrix
		* \param cluster_i Cluster index for which Psi^(-1) is calculated
		* \param only_at_non_zeroes_of_psi If true, psi_inv is calculated only at non-zero entries of psi, e.g., since it is used for calculating gradients afterwards
		*/
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
		void CalcPsiInv(T_mat& psi_inv, data_size_t cluster_i, bool only_at_non_zeros_of_psi) {
			if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
				Log::REFatal("'CalcPsiInv': no implemented for approximation '%s' ", gp_approx_.c_str());
			}
			if (only_grouped_REs_use_woodbury_identity_) {
				sp_mat_t MInvSqrtZt;
				if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
					MInvSqrtZt = sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().inverse().matrix().asDiagonal() * Zt_[cluster_i];
				}
				else {
					T_mat L_inv;
					if (CholeskyHasPermutation<T_chol>(chol_facts_[cluster_i])) {
						TriangularSolve<T_mat, T_mat, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), P_Id_[cluster_i], L_inv, false);
					}
					else {
						TriangularSolve<T_mat, T_mat, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), Id_[cluster_i], L_inv, false);
					}
					MInvSqrtZt = L_inv * Zt_[cluster_i];
				}
				if (only_at_non_zeros_of_psi) {
					CalcZSigmaZt(psi_inv, cluster_i);//find out sparsity pattern where psi_inv is needed for gradient
					CalcLtLGivenSparsityPattern<T_mat>(MInvSqrtZt, psi_inv, true);
					psi_inv *= -1.;
				}
				else {
					psi_inv = -MInvSqrtZt.transpose() * MInvSqrtZt;//this is slow since n can be large (O(n^2*m)) (but its usually not run, only when calculating the Fisher information)
				}
				psi_inv.diagonal().array() += 1.0;
			}
			else {
				T_mat L_inv;
				if (CholeskyHasPermutation<T_chol>(chol_facts_[cluster_i])) {
					TriangularSolve<T_mat, T_mat, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), P_Id_[cluster_i], L_inv, false);
				}
				else {
					TriangularSolve<T_mat, T_mat, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), Id_[cluster_i], L_inv, false);
				}
				if (only_at_non_zeros_of_psi) {
					//find out sparsity pattern where psi_inv is needed for gradient
					if (num_re_group_total_ == 0) {
						std::shared_ptr<T_mat> psi = re_comps_[cluster_i][0]->GetZSigmaZt();
						psi_inv = *psi;
					}
					else {
						CalcZSigmaZt(psi_inv, cluster_i);
					}
					CalcLtLGivenSparsityPattern<T_mat>(L_inv, psi_inv, true);
				}
				else {
					psi_inv = L_inv.transpose() * L_inv;//Note: this is the computational bottleneck for large data when psi=ZSigmaZt and its Cholesky factor is sparse (but its usually not run, only when calculating the Fisher information)
				}
			}
		}// end CalcPsiInv for sparse matrices
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<den_mat_t, T_aux>::value>::type* = nullptr >
		void CalcPsiInv(den_mat_t& psi_inv, data_size_t cluster_i, bool) {
			if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
				Log::REFatal("'CalcPsiInv': no implemented for approximation '%s' ", gp_approx_.c_str());
			}
			if (only_grouped_REs_use_woodbury_identity_) {//typically currently not called as only_grouped_REs_use_woodbury_identity_ is only true for grouped REs only i.e. sparse matrices
				den_mat_t MInvSqrtZt;
				if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
					MInvSqrtZt = sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().inverse().matrix().asDiagonal() * Zt_[cluster_i];
				}
				else {
					TriangularSolve<den_mat_t, sp_mat_t, den_mat_t>(chol_facts_[cluster_i].CholFactMatrix(), Zt_[cluster_i], MInvSqrtZt, false);
				}
				psi_inv = -MInvSqrtZt.transpose() * MInvSqrtZt;
				psi_inv.diagonal().array() += 1.0;
			}
			else {
				den_mat_t L_inv;
				TriangularSolve<den_mat_t, den_mat_t, den_mat_t>(chol_facts_[cluster_i].CholFactMatrix(), Id_[cluster_i], L_inv, false);
				psi_inv = L_inv.transpose() * L_inv;
			}
		}// end CalcPsiInv for dense matrices

		/*!
		* \brief Caclulate X^TPsi^(-1)X
		* \param X Covariate data matrix X
		* \param[out] XT_psi_inv_X X^TPsi^(-1)X
		*/
		void CalcXTPsiInvX(const den_mat_t& X, den_mat_t& XT_psi_inv_X) {
			if (num_clusters_ == 1 && (gp_approx_ != "vecchia" || vecchia_ordering_ == "none") && gp_approx_ != "full_scale_tapering" && gp_approx_ != "fitc") {//only one cluster / idependent GP realization
				if (gp_approx_ == "vecchia") {
					den_mat_t BX = B_[unique_clusters_[0]] * X;
					XT_psi_inv_X = BX.transpose() * D_inv_[unique_clusters_[0]] * BX;
				}
				else {
					if (only_grouped_REs_use_woodbury_identity_) {
						den_mat_t ZtX = Zt_[unique_clusters_[0]] * X;
						den_mat_t MInvSqrtZtX;
						if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
							MInvSqrtZtX = sqrt_diag_SigmaI_plus_ZtZ_[unique_clusters_[0]].array().inverse().matrix().asDiagonal() * ZtX;
						}
						else {
							TriangularSolveGivenCholesky<T_chol, T_mat, den_mat_t, den_mat_t>(chol_facts_[unique_clusters_[0]], ZtX, MInvSqrtZtX, false);
						}
						XT_psi_inv_X = X.transpose() * X - MInvSqrtZtX.transpose() * MInvSqrtZtX;
					}
					else {
						den_mat_t MInvSqrtX;
						TriangularSolveGivenCholesky<T_chol, T_mat, den_mat_t, den_mat_t>(chol_facts_[unique_clusters_[0]], X, MInvSqrtX, false);
						XT_psi_inv_X = MInvSqrtX.transpose() * MInvSqrtX;
					}
				}
			}//end only one cluster / idependent GP realization
			else {//more than one cluster and order of samples matters
				XT_psi_inv_X = den_mat_t(X.cols(), X.cols());
				XT_psi_inv_X.setZero();
				den_mat_t BX, psi_inv_X;;
				for (const auto& cluster_i : unique_clusters_) {
					if (gp_approx_ == "vecchia") {
						BX = B_[cluster_i] * X(data_indices_per_cluster_[cluster_i], Eigen::all);
						XT_psi_inv_X += BX.transpose() * D_inv_[cluster_i] * BX;
					}
					else if (gp_approx_ == "full_scale_tapering" || gp_approx_ == "fitc") {
						std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov_[cluster_i][0]->GetZSigmaZt();
						if (matrix_inversion_method_ == "cholesky") {
							if (gp_approx_ == "fitc") {
								den_mat_t cross_covT_X = (*cross_cov).transpose() * (FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * X);
								den_mat_t sigma_woodbury_I_cross_covT_X = chol_fact_sigma_woodbury_[cluster_i].solve(cross_covT_X);
								cross_covT_X.resize(0, 0);
								den_mat_t cross_cov_sigma_woodbury_I_cross_covT_X = FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * ((*cross_cov) * sigma_woodbury_I_cross_covT_X);
								sigma_woodbury_I_cross_covT_X.resize(0, 0);
								psi_inv_X = FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * X - cross_cov_sigma_woodbury_I_cross_covT_X;
							}
							else if (gp_approx_ == "full_scale_tapering") {
								den_mat_t sigma_resid_I_X = chol_fact_resid_[cluster_i].solve(X);
								den_mat_t cross_covT_sigma_resid_I_X = (*cross_cov).transpose() * sigma_resid_I_X;
								den_mat_t sigma_woodbury_I_cross_covT_sigma_resid_I_X = chol_fact_sigma_woodbury_[cluster_i].solve(cross_covT_sigma_resid_I_X);
								cross_covT_sigma_resid_I_X.resize(0, 0);
								den_mat_t cross_cov_sigma_woodbury_I_cross_covT_sigma_resid_I_X = (*cross_cov) * sigma_woodbury_I_cross_covT_sigma_resid_I_X;
								sigma_woodbury_I_cross_covT_sigma_resid_I_X.resize(0, 0);
								den_mat_t sigma_resid_I_cross_cov_sigma_woodbury_I_cross_covT_sigma_resid_I_X = chol_fact_resid_[cluster_i].solve(cross_cov_sigma_woodbury_I_cross_covT_sigma_resid_I_X);
								psi_inv_X = sigma_resid_I_X - sigma_resid_I_cross_cov_sigma_woodbury_I_cross_covT_sigma_resid_I_X;
							}
						}
						else {
							std::shared_ptr<T_mat> sigma_resid = re_comps_resid_[cluster_i][0]->GetZSigmaZt();
							psi_inv_X.resize(num_data_per_cluster_[cluster_i], X.cols());
							psi_inv_X.setZero();
							CGFSA_MULTI_RHS<T_mat>(*sigma_resid, (*cross_cov), chol_fact_sigma_ip_[cluster_i], X, psi_inv_X,
								NaN_found, num_data_per_cluster_[cluster_i], (int)X.cols(), cg_max_num_it_, cg_delta_conv_,
								cg_preconditioner_type_, chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
							if (NaN_found) {
								Log::REFatal("There was Nan or Inf value generated in the Conjugate Gradient Method!");
							}
						}
						XT_psi_inv_X = X.transpose() * psi_inv_X;
					}
					else {
						if (only_grouped_REs_use_woodbury_identity_) {
							den_mat_t ZtX = Zt_[cluster_i] * (den_mat_t)X(data_indices_per_cluster_[cluster_i], Eigen::all);
							den_mat_t MInvSqrtZtX;
							if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
								MInvSqrtZtX = sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().inverse().matrix().asDiagonal() * ZtX;
							}
							else {
								TriangularSolveGivenCholesky<T_chol, T_mat, den_mat_t, den_mat_t>(chol_facts_[cluster_i], ZtX, MInvSqrtZtX, false);
							}
							XT_psi_inv_X += ((den_mat_t)X(data_indices_per_cluster_[cluster_i], Eigen::all)).transpose() * (den_mat_t)X(data_indices_per_cluster_[cluster_i], Eigen::all) -
								MInvSqrtZtX.transpose() * MInvSqrtZtX;
						}
						else {
							den_mat_t MInvSqrtX;
							TriangularSolveGivenCholesky<T_chol, T_mat, den_mat_t, den_mat_t>(chol_facts_[cluster_i], (den_mat_t)X(data_indices_per_cluster_[cluster_i], Eigen::all), MInvSqrtX, false);
							XT_psi_inv_X += MInvSqrtX.transpose() * MInvSqrtX;
						}
					}
				}
			}//end more than one cluster
		}//end CalcXTPsiInvX

		/*!
		* \brief Initialize data structures for handling independent realizations of the Gaussian processes
		* \param num_data Number of data points
		* \param cluster_ids_data IDs / labels indicating independent realizations of Gaussian processes (same values = same process realization)
		* \param[out] num_data_per_cluster Keys: labels of independent clusters, values: number of data points per independent realization
		* \param[out] data_indices_per_cluster Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
		* \param[out] unique_clusters Unique labels of independent realizations
		* \param[out] num_clusters Number of independent clusters
		*/
		void SetUpGPIds(data_size_t num_data,
			const data_size_t* cluster_ids_data,
			std::map<data_size_t, int>& num_data_per_cluster,
			std::map<data_size_t, std::vector<int>>& data_indices_per_cluster,
			std::vector<data_size_t>& unique_clusters,
			data_size_t& num_clusters) {
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
		}//end SetUpGPIds

		/*!
		* \brief Convert characters in 'const char* re_group_data' to matrix (num_re_group x num_data) with strings of group labels
		* \param num_data Number of data points
		* \param num_re_group Number of grouped random effects
		* \param re_group_data Labels of group levels for the grouped random effects in column-major format (i.e. first the levels for the first effect, then for the second, etc.). Every group label needs to end with the null character '\0'
		* \param[out] Matrix of dimension num_re_group x num_data with strings of group labels for levels of grouped random effects
		*/
		void ConvertCharToStringGroupLevels(data_size_t num_data,
			data_size_t num_re_group,
			const char* re_group_data,
			std::vector<std::vector<re_group_t>>& re_group_levels) {
			int char_start = 0;
			for (int ire = 0; ire < num_re_group; ++ire) {
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
		* \brief Initialize likelihoods
		* \param likelihood Likelihood name
		*/
		void InitializeLikelihoods(const string_t& likelihood) {
			for (const auto& cluster_i : unique_clusters_) {
				if (gp_approx_ == "vecchia") {
					likelihood_[cluster_i] = std::unique_ptr<Likelihood<T_mat, T_chol>>(new Likelihood<T_mat, T_chol>(likelihood,
						num_data_per_cluster_[cluster_i],
						re_comps_[cluster_i][ind_intercept_gp_]->GetNumUniqueREs(),
						false,
						only_one_GP_calculations_on_RE_scale_,
						re_comps_[cluster_i][ind_intercept_gp_]->random_effects_indices_of_data_.data()));
				}
				else if (only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_) {
					likelihood_[cluster_i] = std::unique_ptr<Likelihood<T_mat, T_chol>>(new Likelihood<T_mat, T_chol>(likelihood,
						num_data_per_cluster_[cluster_i],
						cum_num_rand_eff_[cluster_i][num_re_group_total_],
						false,
						false,
						nullptr));
				}
				else if (only_one_grouped_RE_calculations_on_RE_scale_) {
					likelihood_[cluster_i] = std::unique_ptr<Likelihood<T_mat, T_chol>>(new Likelihood<T_mat, T_chol>(likelihood,
						num_data_per_cluster_[cluster_i],
						re_comps_[cluster_i][0]->GetNumUniqueREs(),
						false,
						false,
						nullptr));
				}
				else if (only_one_GP_calculations_on_RE_scale_ && gp_approx_ != "vecchia") {
					likelihood_[cluster_i] = std::unique_ptr<Likelihood<T_mat, T_chol>>(new Likelihood<T_mat, T_chol>(likelihood,
						num_data_per_cluster_[cluster_i],
						re_comps_[cluster_i][0]->GetNumUniqueREs(),
						true,
						true,
						re_comps_[cluster_i][0]->random_effects_indices_of_data_.data()));
				}
				else {
					likelihood_[cluster_i] = std::unique_ptr<Likelihood<T_mat, T_chol>>(new Likelihood<T_mat, T_chol>(likelihood,
						num_data_per_cluster_[cluster_i],
						num_data_per_cluster_[cluster_i],
						true,
						false,
						nullptr));
				}
				if (!gauss_likelihood_) {
					likelihood_[cluster_i]->InitializeModeAvec();
				}
			}
		}//end InitializeLikelihoods

		/*!
		* \brief Function that determines
		*		(i) the indices (in ind_par_) of the covariance parameters of every random effect component in the vector of all covariance parameter
		*		(ii) the total number of covariance parameters
		*/
		void DetermineCovarianceParameterIndicesNumCovPars() {
			// Determine ind_par_ and num_cov_par_
			ind_par_ = std::vector<data_size_t>();
			//First re_comp has either index 0 or 1 (the latter if there is an nugget effect for Gaussian data)
			if (gauss_likelihood_) {
				num_cov_par_ = 1;
				ind_par_.push_back(1);
			}
			else {
				num_cov_par_ = 0;
				ind_par_.push_back(0);
			}
			//Add indices of parameters of individual components in joint parameter vector
			if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
				for (int j = 0; j < (int)re_comps_ip_[unique_clusters_[0]].size(); ++j) {
					ind_par_.push_back(ind_par_.back() + re_comps_ip_[unique_clusters_[0]][j]->NumCovPar());//end points of parameter indices of components
					num_cov_par_ += re_comps_ip_[unique_clusters_[0]][j]->NumCovPar();
				}
			}
			else {
				for (int j = 0; j < (int)re_comps_[unique_clusters_[0]].size(); ++j) {
					ind_par_.push_back(ind_par_.back() + re_comps_[unique_clusters_[0]][j]->NumCovPar());//end points of parameter indices of components
					num_cov_par_ += re_comps_[unique_clusters_[0]][j]->NumCovPar();
				}
			}
		}

		/*!
		* \brief Function that determines whether to use special options for estimation and prediction for certain special cases of random effects models
		*/
		void DetermineSpecialCasesModelsEstimationPrediction() {
			chol_fact_pattern_analyzed_ = false;
			// Decide whether to use the Woodbury identity (i.e. do matrix inversion on the b scale and not the Zb scale) for grouped random effects models only
			if (num_re_group_total_ > 0 && num_gp_total_ == 0) {
				only_grouped_REs_use_woodbury_identity_ = true;//Faster to use Woodbury identity since the dimension of the random effects is typically much smaller than the number of data points
				//Note: the use of the Woodburry identity is currently only implemented for grouped random effects (which is also the only use of it). 
				//		If this should be applied to GPs in the future, adaptions need to be made e.g. in the calculations of the gradient (see y_tilde2_)
			}
			else {
				only_grouped_REs_use_woodbury_identity_ = false;
			}
			// Define options for faster calculations for special cases of RE models (these options depend on the type of likelihood)
			only_one_GP_calculations_on_RE_scale_ = num_gp_total_ == 1 && num_comps_total_ == 1 && !gauss_likelihood_ && gp_approx_ == "none";//If there is only one GP, we do calculations on the b-scale instead of Zb-scale (only for non-Gaussian likelihoods)
			only_one_grouped_RE_calculations_on_RE_scale_ = num_re_group_total_ == 1 && num_comps_total_ == 1 && !gauss_likelihood_;//If there is only one grouped RE, we do (all) calculations on the b-scale instead of the Zb-scale (this flag is only used for non-Gaussian likelihoods)
			only_one_grouped_RE_calculations_on_RE_scale_for_prediction_ = num_re_group_total_ == 1 && num_comps_total_ == 1 && gauss_likelihood_;//If there is only one grouped RE, we do calculations for prediction on the b-scale instead of the Zb-scale (this flag is only used for Gaussian likelihoods)
		}//end DetermineSpecialCasesModelsEstimationPrediction

		/*!
		* \brief Function that set default values for several parameters if they were not initialized
		*/
		void InitializeDefaultSettings() {
			if (!coef_optimizer_has_been_set_) {
				if (gauss_likelihood_) {
					optimizer_coef_ = "wls";
				}
				else {
					optimizer_coef_ = "gradient_descent";
				}
			}
			if (!vecchia_pred_type_has_been_set_) {
				if (gauss_likelihood_) {
					vecchia_pred_type_ = "order_obs_first_cond_obs_only";
				}
				else {
					vecchia_pred_type_ = "latent_order_obs_first_cond_obs_only";
				}
			}
			if (!set_optim_config_has_been_called_ && NumAuxPars() > 0) {
				if (!gauss_likelihood_) {
					estimate_aux_pars_ = true;;
				}
				else {
					estimate_aux_pars_ = false;
				}
			}
			if (!cg_preconditioner_type_has_been_set_) {
				if (gauss_likelihood_) {
					cg_preconditioner_type_ = "predictive_process_plus_diagonal";
				}
				else {
					cg_preconditioner_type_ = "Sigma_inv_plus_BtWB";
				}
				CheckPreconditionerType();
			}
		}//end InitializeDefaultSettings

		/*!
		* \brief Initialize required matrices used when only_grouped_REs_use_woodbury_identity_==true
		*/
		void InitializeMatricesForOnlyGroupedREsUseWoodburyIdentity() {
			CHECK(num_comps_total_ == num_re_group_total_);
			CHECK(only_grouped_REs_use_woodbury_identity_);
			Zt_ = std::map<data_size_t, sp_mat_t>();
			ZtZ_ = std::map<data_size_t, sp_mat_t>();
			cum_num_rand_eff_ = std::map<data_size_t, std::vector<data_size_t>>();
			Zj_square_sum_ = std::map<data_size_t, std::vector<double>>();
			ZtZj_ = std::map<data_size_t, std::vector<sp_mat_t>>();
			for (const auto& cluster_i : unique_clusters_) {
				std::vector<data_size_t> cum_num_rand_eff_cluster_i(num_comps_total_ + 1);
				cum_num_rand_eff_cluster_i[0] = 0;
				//Determine number of rows and non-zero entries of Z
				int non_zeros = 0;
				int ncols = 0;
				for (int j = 0; j < num_comps_total_; ++j) {
					sp_mat_t* Z_j = re_comps_[cluster_i][j]->GetZ();
					ncols += (int)Z_j->cols();
					non_zeros += (int)Z_j->nonZeros();
					cum_num_rand_eff_cluster_i[j + 1] = ncols;
				}
				//Create matrix Z and calculate sum(Z_j^2) = trace(Z_j^T * Z_j)
				std::vector<Triplet_t> triplets;
				triplets.reserve(non_zeros);
				std::vector<double> Zj_square_sum_cluster_i(num_comps_total_);
				int ncol_prev = 0;
				for (int j = 0; j < num_comps_total_; ++j) {
					sp_mat_t* Z_j = re_comps_[cluster_i][j]->GetZ();
					for (int k = 0; k < Z_j->outerSize(); ++k) {
						for (sp_mat_t::InnerIterator it(*Z_j, k); it; ++it) {
							triplets.emplace_back(it.row(), ncol_prev + it.col(), it.value());
						}
					}
					ncol_prev += (int)Z_j->cols();
					Zj_square_sum_cluster_i[j] = Z_j->squaredNorm();
				}
				sp_mat_t Z_cluster_i(num_data_per_cluster_[cluster_i], ncols);
				Z_cluster_i.setFromTriplets(triplets.begin(), triplets.end());
				sp_mat_t Zt_cluster_i = Z_cluster_i.transpose();
				sp_mat_t ZtZ_cluster_i = Zt_cluster_i * Z_cluster_i;
				//Calculate Z^T * Z_j
				std::vector<sp_mat_t> ZtZj_cluster_i(num_comps_total_);
				for (int j = 0; j < num_comps_total_; ++j) {
					sp_mat_t* Z_j = re_comps_[cluster_i][j]->GetZ();
					ZtZj_cluster_i[j] = Zt_cluster_i * (*Z_j);
				}
				//Save all quantities
				Zt_.insert({ cluster_i, Zt_cluster_i });
				ZtZ_.insert({ cluster_i, ZtZ_cluster_i });
				cum_num_rand_eff_.insert({ cluster_i, cum_num_rand_eff_cluster_i });
				Zj_square_sum_.insert({ cluster_i, Zj_square_sum_cluster_i });
				ZtZj_.insert({ cluster_i, ZtZj_cluster_i });
			}
		}

		/*!
		* \brief Initialize identity matrices required for Gaussian data
		*/
		void InitializeIdentityMatricesForGaussianData() {
			if (gauss_likelihood_ && gp_approx_ != "vecchia" && gp_approx_ != "fitc" && gp_approx_ != "full_scale_tapering") {
				for (const auto& cluster_i : unique_clusters_) {
					ConstructI(cluster_i);//Idendity matrices needed for computing inverses of covariance matrices used in gradient descent for Gaussian data
				}
			}
		}

		/*!
		* \brief Function that checks the compatibility of the chosen special options for estimation and prediction for certain special cases of random effects models
		*/
		void CheckCompatibilitySpecialOptions() {
			//Some checks
			if (only_one_GP_calculations_on_RE_scale_ && only_grouped_REs_use_woodbury_identity_) {
				Log::REFatal("Cannot set both 'only_one_GP_calculations_on_RE_scale_' and 'only_grouped_REs_use_woodbury_identity_' to 'true'");
			}
			if (only_one_GP_calculations_on_RE_scale_ && only_one_grouped_RE_calculations_on_RE_scale_) {
				Log::REFatal("Cannot set both 'only_one_GP_calculations_on_RE_scale_' and 'only_one_grouped_RE_calculations_on_RE_scale_' to 'true'");
			}
			if (gp_approx_ != "none") {
				if (num_re_group_total_ > 0) {
					Log::REFatal("The approximation '%s' can currently not be used when there are grouped random effects ", gp_approx_.c_str());
				}
			}
			if (only_one_GP_calculations_on_RE_scale_) {//only_one_GP_calculations_on_RE_scale_
				if (gauss_likelihood_) {
					Log::REFatal("Option 'only_one_GP_calculations_on_RE_scale_' is currently not implemented for Gaussian data");
				}
				if (gp_approx_ != "vecchia" && gp_approx_ != "none") {
					Log::REFatal("Option 'only_one_GP_calculations_on_RE_scale_' is currently not implemented for the approximation '%s' ", gp_approx_.c_str());
				}
				CHECK(num_gp_total_ == 1);
				CHECK(num_comps_total_ == 1);
				CHECK(num_re_group_total_ == 0);
			}
			if (only_one_grouped_RE_calculations_on_RE_scale_) {//only_one_grouped_RE_calculations_on_RE_scale_
				if (gauss_likelihood_) {
					Log::REFatal("Option 'only_one_grouped_RE_calculations_on_RE_scale_' is currently not implemented for Gaussian data");
				}
				CHECK(gp_approx_ == "none");
				CHECK(num_gp_total_ == 0);
				CHECK(num_comps_total_ == 1);
				CHECK(num_re_group_total_ == 1);
			}
			if (only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {//only_one_grouped_RE_calculations_on_RE_scale_for_prediction_
				CHECK(gp_approx_ == "none");
				CHECK(num_gp_total_ == 0);
				CHECK(num_comps_total_ == 1);
				CHECK(num_re_group_total_ == 1);
				if (!gauss_likelihood_) {
					Log::REFatal("Option 'only_one_grouped_RE_calculations_on_RE_scale_for_prediction_' is currently only effective for Gaussian data");
				}
			}
			if (only_grouped_REs_use_woodbury_identity_) {//only_grouped_REs_use_woodbury_identity_
				if (gauss_likelihood_ && only_one_grouped_RE_calculations_on_RE_scale_) {
					Log::REFatal("Cannot enable 'only_one_grouped_RE_calculations_on_RE_scale_' if 'only_grouped_REs_use_woodbury_identity_' is enabled for Gaussian data");
				}
				CHECK(num_gp_total_ == 0);
				CHECK(num_comps_total_ == num_re_group_total_);
			}
			if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
				if (!gauss_likelihood_) {
					Log::REFatal("Approximation '%s' is currently not supported for non-Gaussian likelihoods ", gp_approx_.c_str());
				}
			}
		}

		/*! \brief Check whether preconditioenr is supported */
		void CheckPreconditionerType() const {
			if (gauss_likelihood_) {
				if (SUPPORTED_CG_PRECONDITIONER_TYPE_GAUSS_.find(cg_preconditioner_type_) == SUPPORTED_CG_PRECONDITIONER_TYPE_GAUSS_.end()) {
					Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
				}
			}
			else {
				if (SUPPORTED_CG_PRECONDITIONER_TYPE_NONGAUSS_.find(cg_preconditioner_type_) == SUPPORTED_CG_PRECONDITIONER_TYPE_NONGAUSS_.end()) {
					Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
				}
			}
		}//end CheckPreconditionerType

		/*! \brief Set matrix inversion properties and choices for iterative methods in likelihoods.h */
		void SetMatrixInversionPropertiesLikelihood() {
			if (!gauss_likelihood_) {
				for (const auto& cluster_i : unique_clusters_) {
					likelihood_[cluster_i]->SetMatrixInversionProperties(matrix_inversion_method_,
						cg_max_num_it_, cg_max_num_it_tridiag_, cg_delta_conv_, cg_delta_conv_pred_,
						num_rand_vec_trace_, reuse_rand_vec_trace_, seed_rand_vec_trace_,
						cg_preconditioner_type_, piv_chol_rank_, rank_pred_approx_matrix_lanczos_, nsim_var_pred_);
				}
			}
		}//end SetMatrixInversionPropertiesLikelihood

		/*!
		* \brief Initialize individual component models and collect them in a containter
		* \param num_data Number of data points
		* \param data_indices_per_cluster Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points
		* \param cluster_i Index / label of the realization of the Gaussian process for which the components should be constructed
		* \param Group levels for every grouped random effect
		* \param num_data_per_cluster Keys: Labels of independent realizations of REs/GPs, values: number of data points per independent realization
		* \param re_group_rand_coef_data Covariate data for grouped random coefficients
		* \param gp_coords_data Coordinates (features) for Gaussian process
		* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
		* \param calculateZZt If true, the matrix Z*Z^T is calculated for grouped random effects and saved (usually not needed if Woodbury identity is used)
		* \param[out] re_comps_cluster_i Container that collects the individual component models
		*/
		void CreateREComponents(data_size_t num_data,
			std::map<data_size_t, std::vector<int>>& data_indices_per_cluster,
			data_size_t cluster_i,
			std::vector<std::vector<re_group_t>>& re_group_levels,
			std::map<data_size_t, int>& num_data_per_cluster,
			const double* re_group_rand_coef_data,
			const double* gp_coords_data,
			const double* gp_rand_coef_data,
			bool calculateZZt,
			std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i) {
			//Grouped random effects
			if (num_re_group_ > 0) {
				for (int j = 0; j < num_re_group_; ++j) {
					std::vector<re_group_t> group_data;
					for (const auto& id : data_indices_per_cluster[cluster_i]) {
						group_data.push_back(re_group_levels[j][id]);
					}
					re_comps_cluster_i.push_back(std::shared_ptr<RECompGroup<T_mat>>(new RECompGroup<T_mat>(
						group_data,
						calculateZZt,
						!only_one_grouped_RE_calculations_on_RE_scale_)));
				}
				//Random slope grouped random effects
				if (num_re_group_rand_coef_ > 0) {
					for (int j = 0; j < num_re_group_rand_coef_; ++j) {
						std::vector<double> rand_coef_data;
						for (const auto& id : data_indices_per_cluster[cluster_i]) {
							rand_coef_data.push_back(re_group_rand_coef_data[j * num_data + id]);
						}
						std::shared_ptr<RECompGroup<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGroup<T_mat>>(re_comps_cluster_i[ind_effect_group_rand_coef_[j] - 1]);//Subtract -1 since ind_effect_group_rand_coef[j] starts counting at 1 not 0
						re_comps_cluster_i.push_back(std::shared_ptr<RECompGroup<T_mat>>(new RECompGroup<T_mat>(
							re_comp->random_effects_indices_of_data_.data(),
							re_comp->num_data_,
							re_comp->map_group_label_index_,
							re_comp->num_group_,
							rand_coef_data,
							calculateZZt)));
					}
					// drop some intercept random effects (if specified)
					int num_droped = 0;
					for (int j = 0; j < num_re_group_; ++j) {
						if (drop_intercept_group_rand_effect_[j]) {
							re_comps_cluster_i.erase(re_comps_cluster_i.begin() + j);
							num_droped += 1;
						}
					}
					num_re_group_ -= num_droped;
					num_re_group_total_ -= num_droped;
					num_comps_total_ -= num_droped;
				}
			}
			//GPs
			if (num_gp_ > 0) {
				std::vector<double> gp_coords;
				for (int j = 0; j < dim_gp_coords_; ++j) {
					for (const auto& id : data_indices_per_cluster[cluster_i]) {
						gp_coords.push_back(gp_coords_data[j * num_data + id]);
					}
				}
				den_mat_t gp_coords_mat = Eigen::Map<den_mat_t>(gp_coords.data(), num_data_per_cluster[cluster_i], dim_gp_coords_);
				bool use_Z_for_duplicates = (gp_approx_ == "none");
				re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T_mat>>(new RECompGP<T_mat>(
					gp_coords_mat,
					cov_fct_,
					cov_fct_shape_,
					cov_fct_taper_range_,
					cov_fct_taper_shape_,
					gp_approx_ == "tapering",
					false,
					true,
					use_Z_for_duplicates,
					only_one_GP_calculations_on_RE_scale_)));
				//Random slope GPs
				if (num_gp_rand_coef_ > 0) {
					for (int j = 0; j < num_gp_rand_coef_; ++j) {
						std::vector<double> rand_coef_data;
						for (const auto& id : data_indices_per_cluster[cluster_i]) {
							rand_coef_data.push_back(gp_rand_coef_data[j * num_data + id]);
						}
						std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp_]);
						re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T_mat>>(new RECompGP<T_mat>(
							re_comp->dist_,
							re_comp->has_Z_,
							&re_comp->Z_,
							rand_coef_data,
							cov_fct_,
							cov_fct_shape_,
							cov_fct_taper_range_,
							cov_fct_taper_shape_,
							re_comp->GetTaperMu(),
							gp_approx_ == "tapering",
							false)));
					}
				}
			}
		}//end CreateREComponents

		/*!
		* \brief Initialize individual component models and collect them in a containter
		* \param num_data Number of data points
		* \param data_indices_per_cluster Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points
		* \param cluster_i Index / label of the realization of the Gaussian process for which the components should be constructed
		* \param gp_coords_data Coordinates (features) for Gaussian process
		* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
		* \param[out] re_comps_ip_cluster_i Inducing point GP for predictive process
		* \param[out] re_comps_cross_cov_cluster_i Cross-covariance GP for predictive process
		* \param[out] re_comps_resid_cluster_i Residual GP component for full scale approximation
		*/
		void CreateREComponentsPPFSA(data_size_t num_data,
			std::map<data_size_t, std::vector<int>>& data_indices_per_cluster,
			data_size_t cluster_i,
			const double* gp_coords_data,
			std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
			std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i) {
			if (gp_approx_ == "fitc") {
				if (num_data_per_cluster_[cluster_i] < num_ind_points_) {
					Log::REFatal("Cannot have more inducing points than data points for '%s' approximation ", gp_approx_.c_str());
				}
			}
			else if (gp_approx_ == "full_scale_tapering") {
				if (num_data_per_cluster_[cluster_i] <= num_ind_points_) {
					Log::REFatal("Need to have less inducing points than data points for '%s' approximation ", gp_approx_.c_str());
				}
			}
			CHECK(num_gp_ > 0);
			std::vector<double> gp_coords_all;
			for (int j = 0; j < dim_gp_coords_; ++j) {
				for (const auto& id : data_indices_per_cluster[cluster_i]) {
					gp_coords_all.push_back(gp_coords_data[j * num_data + id]);
				}
			}
			den_mat_t gp_coords_all_mat = Eigen::Map<den_mat_t>(gp_coords_all.data(), num_data_per_cluster_[cluster_i], dim_gp_coords_);
			// Determine inducing points on unique locataions
			den_mat_t gp_coords_all_unique;
			std::vector<int> uniques;//unique points
			std::vector<int> unique_idx;//not used
			DetermineUniqueDuplicateCoordsFast(gp_coords_all_mat, num_data_per_cluster_[cluster_i], uniques, unique_idx);
			if ((data_size_t)uniques.size() == num_data_per_cluster_[cluster_i]) {//no multiple observations at the same locations -> no incidence matrix needed
				gp_coords_all_unique = gp_coords_all_mat;
			}
			else {//there are multiple observations at the same locations
				gp_coords_all_unique = gp_coords_all_mat(uniques, Eigen::all);
				if ((int)gp_coords_all_unique.rows() < num_ind_points_) {
					Log::REFatal("Cannot have more inducing points than unique coordinates for '%s' approximation ", gp_approx_.c_str());
				}
			}
			std::vector<int> indices;
			den_mat_t gp_coords_ip_mat;
			if (ind_points_selection_ == "cover_tree") {
				CoverTree(gp_coords_all_unique, cover_tree_radius_, rng_, gp_coords_ip_mat);
				num_ind_points_ = (int)gp_coords_ip_mat.rows();
			}
			else if (ind_points_selection_ == "random") {
				SampleIntNoReplaceSort((int)gp_coords_all_unique.rows(), num_ind_points_, rng_, indices);
				gp_coords_ip_mat.resize(num_ind_points_, gp_coords_all_mat.cols());
				for (int j = 0; j < num_ind_points_; ++j) {
					gp_coords_ip_mat.row(j) = gp_coords_all_unique.row(indices[j]);
				}
			}
			else if (ind_points_selection_ == "kmeans++") {
				gp_coords_ip_mat.resize(num_ind_points_, gp_coords_all_mat.cols());
				int max_it_kmeans = 1000;
				kmeans_plusplus(gp_coords_all_unique, num_ind_points_, rng_, gp_coords_ip_mat, max_it_kmeans);
			}
			else {
				Log::REFatal("Method '%s' is not supported for finding inducing points ", ind_points_selection_.c_str());
			}
			gp_coords_all_unique.resize(0, 0);
			std::shared_ptr<RECompGP<den_mat_t>> gp_ip(new RECompGP<den_mat_t>(
				gp_coords_ip_mat, cov_fct_, cov_fct_shape_, cov_fct_taper_range_, cov_fct_taper_shape_, false, false, true, false, false));
			if (gp_ip->HasDuplicatedCoords()) {
				Log::REFatal("Duplicates found in inducing points / low-dimensional knots ");
			}
			re_comps_ip_cluster_i.push_back(gp_ip);
			re_comps_cross_cov_cluster_i.push_back(std::shared_ptr<RECompGP<den_mat_t>>(new RECompGP<den_mat_t>(
				gp_coords_all_mat, gp_coords_ip_mat, cov_fct_, cov_fct_shape_, cov_fct_taper_range_, cov_fct_taper_shape_, false, false)));
			if (gp_approx_ == "full_scale_tapering") {
				re_comps_resid_cluster_i.push_back(std::shared_ptr<RECompGP<T_mat>>(new RECompGP<T_mat>(
					gp_coords_all_mat, cov_fct_, cov_fct_shape_, cov_fct_taper_range_, cov_fct_taper_shape_,
					true, true, true, false, false)));
			}
			//Random slope GPs
			if (num_gp_rand_coef_ > 0) {
				Log::REFatal("Random coefficients are currently not supported for '%s' approximation ", ind_points_selection_.c_str());
			}
		}//end CreateREComponentsPPFSA

		/*!
		* \brief Set the covariance parameters of the components
		* \param cov_pars Covariance parameters
		*/
		void SetCovParsComps(const vec_t& cov_pars) {
			CHECK(cov_pars.size() == num_cov_par_);
			for (const auto& cluster_i : unique_clusters_) {
				for (int j = 0; j < num_comps_total_; ++j) {
					const vec_t pars = cov_pars.segment(ind_par_[j], ind_par_[j + 1] - ind_par_[j]);
					if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
						re_comps_ip_[cluster_i][j]->SetCovPars(pars);
						re_comps_cross_cov_[cluster_i][j]->SetCovPars(pars);
						if (gp_approx_ == "full_scale_tapering") {
							re_comps_resid_[cluster_i][j]->SetCovPars(pars);
						}
					}
					else {
						re_comps_[cluster_i][j]->SetCovPars(pars);
					}
				}
			}
		}

		/*!
		* \brief Calculate the total variance of all random effects
		*		Note: for random coefficients processes, we ignore the covariates and simply use the marginal variance for simplicity (this function is used for calling 'FindInitialIntercept' for non-Gaussian likelihoods)
		* \param cov_pars Covariance parameters
		*/
		double GetTotalVarComps(const vec_t& cov_pars) {
			CHECK(cov_pars.size() == num_cov_par_);
			vec_t cov_pars_orig;
			TransformBackCovPars(cov_pars, cov_pars_orig);
			double tot_var = 0.;
			for (int j = 0; j < num_comps_total_; ++j) {
				tot_var += cov_pars_orig[ind_par_[j]];
			}
			if (gauss_likelihood_) {
				tot_var += cov_pars_orig[0];
			}
			return(tot_var);
		}

		/*!
		* \brief Transform the covariance parameters to the scale on which the optimization is done
		* \param cov_pars Covariance parameters on orginal scale
		* \param[out] cov_pars_trans Covariance parameters on transformed scale
		*/
		void TransformCovPars(const vec_t& cov_pars,
			vec_t& cov_pars_trans) {
			CHECK(cov_pars.size() == num_cov_par_);
			cov_pars_trans = vec_t(num_cov_par_);
			if (gauss_likelihood_) {
				cov_pars_trans[0] = cov_pars[0];
			}
			for (int j = 0; j < num_comps_total_; ++j) {
				const vec_t pars = cov_pars.segment(ind_par_[j], ind_par_[j + 1] - ind_par_[j]);
				vec_t pars_trans = pars;
				if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
					if (gauss_likelihood_) {
						re_comps_ip_[unique_clusters_[0]][j]->TransformCovPars(cov_pars[0], pars, pars_trans);
					}
					else {
						re_comps_ip_[unique_clusters_[0]][j]->TransformCovPars(1., pars, pars_trans);
					}
				}
				else {
					if (gauss_likelihood_) {
						re_comps_[unique_clusters_[0]][j]->TransformCovPars(cov_pars[0], pars, pars_trans);
					}
					else {
						re_comps_[unique_clusters_[0]][j]->TransformCovPars(1., pars, pars_trans);
					}
				}
				cov_pars_trans.segment(ind_par_[j], ind_par_[j + 1] - ind_par_[j]) = pars_trans;
			}
		}

		/*!
		* \brief Back-transform the covariance parameters to the original scale
		* \param cov_pars Covariance parameters on transformed scale
		* \param[out] cov_pars_orig Covariance parameters on orginal scale
		*/
		void TransformBackCovPars(const vec_t& cov_pars,
			vec_t& cov_pars_orig) {
			CHECK(cov_pars.size() == num_cov_par_);
			cov_pars_orig = vec_t(num_cov_par_);
			if (gauss_likelihood_) {
				cov_pars_orig[0] = cov_pars[0];
			}
			for (int j = 0; j < num_comps_total_; ++j) {
				const vec_t pars = cov_pars.segment(ind_par_[j], ind_par_[j + 1] - ind_par_[j]);
				vec_t pars_orig = pars;
				if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
					if (gauss_likelihood_) {
						re_comps_ip_[unique_clusters_[0]][j]->TransformBackCovPars(cov_pars[0], pars, pars_orig);
					}
					else {
						re_comps_ip_[unique_clusters_[0]][j]->TransformBackCovPars(1, pars, pars_orig);
					}
				}
				else {
					if (gauss_likelihood_) {
						re_comps_[unique_clusters_[0]][j]->TransformBackCovPars(cov_pars[0], pars, pars_orig);
					}
					else {
						re_comps_[unique_clusters_[0]][j]->TransformBackCovPars(1, pars, pars_orig);
					}
				}
				cov_pars_orig.segment(ind_par_[j], ind_par_[j + 1] - ind_par_[j]) = pars_orig;
			}
		}

		/*!
		* \brief Transform the linear regression coefficients to the scale on which the optimization is done
		* \param beta Regression coefficients on orginal scale
		* \param[out] beta_trans Regression coefficients on transformed scale
		*/
		void TransformCoef(const vec_t& beta,
			vec_t& beta_trans) {
			beta_trans = beta;
			for (int icol = 0; icol < num_coef_; ++icol) {
				if (!has_intercept_ || icol != intercept_col_) {
					if (has_intercept_) {
						beta_trans[intercept_col_] += beta_trans[icol] * loc_transf_[icol];
					}
					beta_trans[icol] *= scale_transf_[icol];
				}
			}
			if (has_intercept_) {
				beta_trans[intercept_col_] *= scale_transf_[intercept_col_];
			}
		}

		/*!
		* \brief Back-transform linear regression coefficients back to original scale
		* \param beta Regression coefficients on transformed scale
		* \param[out] beta_orig Regression coefficients on orginal scale
		*/
		void TransformBackCoef(const vec_t& beta,
			vec_t& beta_orig) {
			beta_orig = beta;
			if (has_intercept_) {
				beta_orig[intercept_col_] /= scale_transf_[intercept_col_];
			}
			for (int icol = 0; icol < num_coef_; ++icol) {
				if (!has_intercept_ || icol != intercept_col_) {
					beta_orig[icol] /= scale_transf_[icol];
					if (has_intercept_) {
						beta_orig[intercept_col_] -= beta_orig[icol] * loc_transf_[icol];
					}
				}
			}
		}

		/*!
		* \brief Calculate covariance matrices of the components
		*/
		void CalcSigmaComps() {
			for (const auto& cluster_i : unique_clusters_) {
				for (int j = 0; j < num_comps_total_; ++j) {
					if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
						re_comps_ip_[cluster_i][j]->CalcSigma();
						re_comps_cross_cov_[cluster_i][j]->CalcSigma();
						den_mat_t sigma_ip_stable = *(re_comps_ip_[cluster_i][j]->GetZSigmaZt());
						sigma_ip_stable.diagonal().array() += EPSILON_ADD_COVARIANCE_STABLE;

						chol_fact_sigma_ip_[cluster_i].compute(sigma_ip_stable);
						if (gp_approx_ == "full_scale_tapering") {

							re_comps_resid_[cluster_i][j]->CalcSigma();

							// Subtract predictive process covariance
							TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_[cluster_i],
								(*(re_comps_cross_cov_[cluster_i][j]->GetZSigmaZt())).transpose(), chol_ip_cross_cov_[cluster_i], false);
							re_comps_resid_[cluster_i][j]->SubtractPredProcFromSigmaForResidInFullScale(chol_ip_cross_cov_[cluster_i], true);
							// Apply Taper
							re_comps_resid_[cluster_i][j]->ApplyTaper();
							if (gauss_likelihood_) {
								re_comps_resid_[cluster_i][j]->AddConstantToDiagonalSigma(1.);//add nugget effect variance
							}
						}
					}//end gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering"
					else {
						re_comps_[cluster_i][j]->CalcSigma();
					}
				}
			}
		}

		/*!
		* \brief Construct covariance matrix Sigma or inverse covariance matrix Sigma^-1 if there are only grouped random effecs (this is then a diagonal matrix)
		* \param[out] SigmaI Covariance matrix or inverse covariance matrix of random effects (a diagonal matrix)
		* \param cluster_i Cluster index for which SigmaI is constructed
		* \param inverse If true, the inverse covariance matrix is calculated
		*/
		void CalcSigmaIGroupedREsOnly(sp_mat_t& SigmaI, data_size_t cluster_i, bool inverse) {
			CHECK(!only_one_grouped_RE_calculations_on_RE_scale_);
			std::vector<Triplet_t> triplets(cum_num_rand_eff_[cluster_i][num_re_group_total_]);
			for (int j = 0; j < num_comps_total_; ++j) {
				double sigmaI = re_comps_[cluster_i][j]->cov_pars_[0];
				if (inverse) {
					sigmaI = 1.0 / sigmaI;
				}
#pragma omp parallel for schedule(static)
				for (int i = cum_num_rand_eff_[cluster_i][j]; i < cum_num_rand_eff_[cluster_i][j + 1]; ++i) {
					triplets[i] = Triplet_t(i, i, sigmaI);
				}
			}
			SigmaI = sp_mat_t(cum_num_rand_eff_[cluster_i][num_re_group_total_], cum_num_rand_eff_[cluster_i][num_re_group_total_]);
			SigmaI.setFromTriplets(triplets.begin(), triplets.end());
		}

		/*!
		* \brief Set initial values for some of the optimizer parameters
		* Internal default values are used if the corresponding parameters have not been set
		*/
		void OptimParamsSetInitialValues() {
			SetInitialValueLRCov();
			SetInitialValueDeltaRelConv();
		}//end OptimParamsSetInitialValues

		/*!
		* \brief Initialitze learning rates and convergence tolerance
		*/
		void InitializeLearningRatesConvTol() {
			lr_coef_ = lr_coef_init_;
			lr_aux_pars_ = lr_aux_pars_init_;
			lr_cov_ = lr_cov_init_;
			delta_rel_conv_ = delta_rel_conv_init_;
			lr_have_been_initialized_ = true;
		}//end InitializeLearningRatesConvTol

		/*! * \brief Set initial values 'lr_cov_init_' for lr_cov_ */
		void SetInitialValueLRCov() {
			if (lr_cov_init_ < 0.) {//A value below 0 indicates that default values should be used
				if (optimizer_cov_pars_ == "gradient_descent") {
					lr_cov_init_ = 0.1;
				}
				else {
					lr_cov_init_ = 1.;
				}
			}
		}//end SetInitialValueLRCov

		/*! * \brief Set initial values 'delta_rel_conv_init_' for delta_rel_conv_ */
		void SetInitialValueDeltaRelConv() {
			if (delta_rel_conv_init_ < 0) {
				if (optimizer_cov_pars_ == "nelder_mead") {
					delta_rel_conv_init_ = 1e-8;
				}
				else {
					delta_rel_conv_init_ = 1e-6;
				}
			}
		}//end SetInitialValueDeltaRelConv

		/*!
		* \brief Avoid too large learning rates for covariance parameters and aux_pars
		* \param neg_step_dir Negative step direction for making updates. E.g., neg_step_dir = grad for gradient descent and neg_step_dir = FI^-1 * grad for Fisher scoring (="natural" gradient)
		*/
		void AvoidTooLargeLearningRatesCovAuxPars(const vec_t& neg_step_dir) {
			int num_grad_cov_par = (int)neg_step_dir.size();
			if (estimate_aux_pars_) {
				num_grad_cov_par -= NumAuxPars();
			}
			double max_lr_cov = MaximalLearningRateCovAuxPars(neg_step_dir.segment(0, num_grad_cov_par));
			if (lr_cov_ > max_lr_cov) {
				lr_cov_ = max_lr_cov;
				Log::REDebug("GPModel: The learning rate for the covariance parameters has been decreased in iteration number %d since "
					"the gradient update on the log-scale would have been too large (change by more than a factor %d). New learning rate = %g",
					num_iter_ + 1, MAX_REL_CHANGE_GRADIENT_UPDATE_, lr_cov_);
			}
			if (estimate_aux_pars_) {
				double max_lr_caux_pars = MaximalLearningRateCovAuxPars(neg_step_dir.segment(num_grad_cov_par, NumAuxPars()));
				if (lr_cov_ > max_lr_caux_pars) {
					lr_aux_pars_ = max_lr_caux_pars;
					Log::REDebug("GPModel: The learning rate for the auxiliary parameters has been decreased in iteration number %d since "
						"the gradient update on the log-scale would have been too large (change by more than a factor %d). New learning rate = %g",
						num_iter_ + 1, MAX_REL_CHANGE_GRADIENT_UPDATE_AUX_PARS_, lr_aux_pars_);
				}
			}
		}//end AvoidTooLargeLearningRatesCovAuxPars

		/*!
		* \brief Avoid too large learning rates for linear regression coefficients
		* \param beta Current / lag1 value of beta
		* \param neg_step_dir Negative step direction for making updates. E.g., neg_step_dir = grad for gradient descent and neg_step_dir = FI^-1 * grad for Fisher scoring (="natural" gradient)
		*/
		void AvoidTooLargeLearningRateCoef(const vec_t& beta,
			const vec_t& neg_step_dir) {
			double max_lr_coef = MaximalLearningRateCoef(beta, neg_step_dir);
			if (lr_coef_ > max_lr_coef) {
				lr_coef_ = max_lr_coef;
				Log::REDebug("GPModel: The learning rate for the regression coefficients has been decreased in iteration number %d since "
					"the current one would have implied a too large change in the mean and variance of the linear predictor relative to the data. New learning rate = %g",
					num_iter_ + 1, lr_coef_);
			}
		}//end AvoidTooLargeLearningRateCoef

		/*!
		* \brief Calculate the directional derivative for the Armijo condition (if armijo_condition_) and
		*		update learning rate such that there is a constant first order change (if learning_rate_constant_first_order_change_)
		* \param grad Gradient of covariance and additional auxiliary likelihood parameters
		* \param neg_step_dir Negative step direction for making updates. E.g., neg_step_dir = grad for gradient descent and neg_step_dir = FI^-1 * grad for Fisher scoring (="natural" gradient)
		* \param cov_aux_pars Covariance and additional auxiliary likelihood parameters
		* \param cov_pars_after_grad_aux Covariance and auxiliary parameters after gradient step and before momentum step (of previous iteration)
		* \param use_nesterov_acc If true, Nesterov acceleration is used
		*/
		void CalcDirDerivArmijoAndLearningRateConstChangeCovAuxPars(const vec_t& grad,
			const vec_t& neg_step_dir,
			const vec_t& cov_aux_pars,
			const vec_t& cov_pars_after_grad_aux,
			bool use_nesterov_acc) {
			if ((learning_rate_constant_first_order_change_ && num_iter_ > 0) || armijo_condition_) {
				CHECK(grad.size() == neg_step_dir.size());
				int num_grad_cov_par = (int)neg_step_dir.size();
				if (estimate_aux_pars_) {
					num_grad_cov_par -= NumAuxPars();
				}
				if (learning_rate_constant_first_order_change_ && num_iter_ > 0) {
					double dir_deriv_armijo_cov_pars_new = -(grad.segment(0, num_grad_cov_par)).dot(neg_step_dir.segment(0, num_grad_cov_par));
					lr_cov_ *= dir_deriv_armijo_cov_pars_ / dir_deriv_armijo_cov_pars_new;
					dir_deriv_armijo_cov_pars_ = dir_deriv_armijo_cov_pars_new;
					if (estimate_aux_pars_) {
						double dir_deriv_armijo_aux_pars_new = -(grad.segment(num_grad_cov_par, NumAuxPars())).dot(neg_step_dir.segment(num_grad_cov_par, NumAuxPars()));
						lr_aux_pars_ *= dir_deriv_armijo_aux_pars_ / dir_deriv_armijo_aux_pars_new;
						dir_deriv_armijo_aux_pars_ = dir_deriv_armijo_aux_pars_new;
					}
				}
				else if (armijo_condition_) {
					dir_deriv_armijo_cov_pars_ = -(grad.segment(0, num_grad_cov_par)).dot(neg_step_dir.segment(0, num_grad_cov_par));
					if (estimate_aux_pars_) {
						dir_deriv_armijo_aux_pars_ = -(grad.segment(num_grad_cov_par, NumAuxPars())).dot(neg_step_dir.segment(num_grad_cov_par, NumAuxPars()));
					}
				}
				if (armijo_condition_ && use_nesterov_acc) {
					vec_t delta_pars = cov_aux_pars.array().log().matrix() - cov_pars_after_grad_aux.array().log().matrix();//gradient steps / update is done on log-scale
					vec_t delta_cov_pars;
					if (profile_out_marginal_variance_) {
						delta_cov_pars = delta_pars.segment(1, num_grad_cov_par);
					}
					else {
						delta_cov_pars = delta_pars.segment(0, num_grad_cov_par);
					}
					mom_dir_deriv_armijo_cov_pars_ = (grad.segment(0, num_grad_cov_par)).dot(delta_cov_pars);
					if (estimate_aux_pars_) {
						vec_t delta_aux_pars = delta_pars.segment(num_cov_par_, NumAuxPars());
						mom_dir_deriv_armijo_aux_pars_ = (grad.segment(num_grad_cov_par, NumAuxPars())).dot(delta_aux_pars);
					}
				}
				else {
					mom_dir_deriv_armijo_cov_pars_ = 0.;
					mom_dir_deriv_armijo_aux_pars_ = 0;
				}
			}
		}//CalcDirDerivArmijoAndLearningRateConstChangeCovAuxPars

		/*!
		* \brief Calculate the directional derivative for the Armijo condition (if armijo_condition_) and
		*			update the learning rate such that there is a constant first order change (if learning_rate_constant_first_order_change_) for regression coefficients
		* \param grad Gradient of linear regression coefficients
		* \param beta Linear regression coefficients
		* \param beta_after_grad_aux Linear regression coefficients after gradient step and before momentum step (of previous iteration)
		* \param use_nesterov_acc If true, Nesterov acceleration is used
		*/
		void CalcDirDerivArmijoAndLearningRateConstChangeCoef(const vec_t& grad,
			const vec_t& beta,
			const vec_t& beta_after_grad_aux,
			bool use_nesterov_acc) {
			if (learning_rate_constant_first_order_change_ && num_iter_ > 0) {
				double dir_deriv_armijo_coef_new = grad.squaredNorm();
				lr_coef_ *= dir_deriv_armijo_coef_ / dir_deriv_armijo_coef_new;
				dir_deriv_armijo_coef_ = dir_deriv_armijo_coef_new;
			}
			else if (armijo_condition_) {
				dir_deriv_armijo_coef_ = grad.squaredNorm();
			}
			if (armijo_condition_ && use_nesterov_acc) {
				mom_dir_deriv_armijo_coef_ = grad.dot(beta - beta_after_grad_aux);
			}
		}//end CalcDirDerivArmijoAndLearningRateConstChangeCoef

		/*!
		* \brief For the GPBoost algorithm, learning rates (lr_cov_ and lr_aux_pars_) are no reset to initial values in every boosting iteration 
		* but rather kept at their previous valzues. This can, however, imply that learning rates might become too small in later boosting iterations.
		* This function checks whether we should increase (double) the learing rates again and does the increase if necessary
		*/
		void PotentiallyIncreaseLearningRatesForGPBoostAlgorithm() {
			bool double_learning_rate = false;
			if (num_iter_ == 0) {
				if (!estimate_aux_pars_) {
					if ((-dir_deriv_armijo_cov_pars_ * lr_cov_) <= (delta_rel_conv_ * std::max(std::abs(neg_log_likelihood_lag1_), 1.))) {
						if ((-dir_deriv_armijo_cov_pars_ * lr_cov_init_) >= std::max(std::abs(neg_log_likelihood_lag1_), 1.)) {
							// Increase the learning again if 
							//		(i) likely no change will be detected in the log-likelihood, i.e., convergence is achieved, (=first "if" condition) 
							//		(ii) but this is due a very small learning rate and for larger learning rates the log-likelihood would still change (=second "if" condition).
							//		In other words, if lr_cov_ is small but the directional derivative of the log-likelihood in the search direction (=first order change of log-likelihood) is large relative to the log-likelihood
							//		Note that (-dir_deriv_armijo_cov_pars_) is approximately equal to (neg_log_likelihood_lag1_ - neg_log_likelihood_) for small lr_cov_
							double_learning_rate = true;
						}
					}
				}
				else {//estimate_aux_pars_
					if ((-dir_deriv_armijo_cov_pars_ * lr_cov_ + -dir_deriv_armijo_aux_pars_ * lr_aux_pars_) <= (delta_rel_conv_ * std::max(std::abs(neg_log_likelihood_lag1_), 1.))) {
						if ((-dir_deriv_armijo_cov_pars_ * lr_cov_init_ + dir_deriv_armijo_aux_pars_ * lr_aux_pars_init_) >= std::max(std::abs(neg_log_likelihood_lag1_), 1.)) {
							double_learning_rate = true;
						}
					}
				}
			}//end num_iter_ == 0
			else if (num_iter_ == 1 && !learning_rates_have_been_doubled_in_first_iteration_) {//always increase the learning rate in the second iteration if more than one iteration is needed 
				double_learning_rate = true;
			}
			if (double_learning_rate) {
				if (2 * lr_cov_ <= lr_cov_init_) {
					lr_cov_ *= 2;
					if (num_iter_ == 0) {
						learning_rates_have_been_doubled_in_first_iteration_ = true;
					}
				}
				if (estimate_aux_pars_) {
					if (2 * lr_aux_pars_ <= lr_aux_pars_init_) {
						lr_aux_pars_ *= 2;
						if (num_iter_ == 0) {
							learning_rates_have_been_doubled_in_first_iteration_ = true;
						}
					}
				}
			}//end double_learning_rate
		}//end PotentiallyIncreaseLearningRatesForGPBoostAlgorithm

		/*!
		* \brief Recaculate mode for Laplace approximation after reseting them to zero
		* \param fixed_effects Fixed effects component of location parameter
		*/
		void RecalculateModeLaplaceApprox(const double* fixed_effects) {
			if (!gauss_likelihood_) {
				//Reset the initial modes to 0. Otherwise, they can get stuck
				for (const auto& cluster_i : unique_clusters_) {
					likelihood_[cluster_i]->InitializeModeAvec();
				}
				CalcModePostRandEffCalcMLL(fixed_effects, false);
			}
		}//end RecalculateModeLaplaceApprox

		/*!
		* \brief Calculate the gradient of the Laplace-approximated negative log-likelihood with respect to the fixed effects F (only used for non-Gaussian likelihoods)
		* \param[out] grad_F Gradient of the Laplace-approximated negative log-likelihood with respect to the fixed effects F. This vector needs to be pre-allocated of length num_data_
		* \param fixed_effects Fixed effects component of location parameter
		*/
		void CalcGradFLaplace(double* grad_F, const double* fixed_effects = nullptr) {
			const double* fixed_effects_cluster_i_ptr = nullptr;
			vec_t fixed_effects_cluster_i;
			for (const auto& cluster_i : unique_clusters_) {
				vec_t grad_F_cluster_i(num_data_per_cluster_[cluster_i]);
				//map fixed effects to clusters (if needed)
				if (num_clusters_ == 1 && (gp_approx_ != "vecchia" || vecchia_ordering_ == "none")) {//only one cluster / independent realization and order of data does not matter
					fixed_effects_cluster_i_ptr = fixed_effects;
				}
				else if (fixed_effects != nullptr) {//more than one cluster and order of samples matters
					fixed_effects_cluster_i = vec_t(num_data_per_cluster_[cluster_i]);
#pragma omp parallel for schedule(static)
					for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
						fixed_effects_cluster_i[j] = fixed_effects[data_indices_per_cluster_[cluster_i][j]];
					}
					fixed_effects_cluster_i_ptr = fixed_effects_cluster_i.data();
				}
				if (gp_approx_ == "vecchia") {
					likelihood_[cluster_i]->CalcGradNegMargLikelihoodLaplaceApproxVecchia(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						B_[cluster_i],
						D_inv_[cluster_i],
						B_grad_[cluster_i],
						D_grad_[cluster_i],
						false,
						true,
						false,
						nullptr,
						grad_F_cluster_i,
						nullptr,
						false,
						num_comps_total_);
				}
				else if (only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_) {
					likelihood_[cluster_i]->CalcGradNegMargLikelihoodLaplaceApproxGroupedRE(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						num_data_per_cluster_[cluster_i],
						SigmaI_[cluster_i],
						Zt_[cluster_i],
						cum_num_rand_eff_[cluster_i],
						false,
						true,
						false,
						nullptr,
						grad_F_cluster_i,
						nullptr,
						false);
				}
				else if (only_one_grouped_RE_calculations_on_RE_scale_) {
					likelihood_[cluster_i]->CalcGradNegMargLikelihoodLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						num_data_per_cluster_[cluster_i],
						re_comps_[cluster_i][0]->cov_pars_[0],
						re_comps_[cluster_i][0]->random_effects_indices_of_data_.data(),
						false,
						true,
						false,
						nullptr,
						grad_F_cluster_i,
						nullptr,
						false);
				}
				else {
					likelihood_[cluster_i]->CalcGradNegMargLikelihoodLaplaceApproxStable(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						ZSigmaZt_[cluster_i],
						re_comps_[cluster_i],
						false,
						true,
						false,
						nullptr,
						grad_F_cluster_i,
						nullptr,
						false);
				}
				//write on output
				if (num_clusters_ == 1 && (gp_approx_ != "vecchia" || vecchia_ordering_ == "none")) {//only one cluster / independent realization and order of data does not matter
#pragma omp parallel for schedule(static)//write on output
					for (int j = 0; j < num_data_; ++j) {
						grad_F[j] = grad_F_cluster_i[j];
					}
				}
				else {//more than one cluster and order of samples matters
#pragma omp parallel for schedule(static)
					for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
						grad_F[data_indices_per_cluster_[cluster_i][j]] = grad_F_cluster_i[j];
					}
				} // end more than one cluster
			}//end loop over cluster
		}//end CalcGradFLaplace

		/*!
		* \brief Update covariance and potential additional likelihood parameters, apply step size safeguard, factorize covariance matrix, and calculate new value of objective function
		* \param[out] cov_pars Covariance and additional auxiliary likelihood parameters
		* \param neg_step_dir Negative step direction for making updates. E.g., neg_step_dir = grad for gradient descent and neg_step_dir = FI^-1 * grad for Fisher scoring (="natural" gradient)
		* \param use_nesterov_acc If true, Nesterov acceleration is used
		* \param it Iteration number
		* \param[out] cov_pars_after_grad_aux Auxiliary variable used only if use_nesterov_acc == true (see the code below for a description)
		* \param[out] cov_pars_after_grad_aux_lag1 Auxiliary variable used only if use_nesterov_acc == true (see the code below for a description)
		* \param acc_rate_cov Nesterov acceleration speed
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
		* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
		* \param fixed_effects Fixed effects component of location parameter
		*/
		void UpdateCovAuxPars(vec_t& cov_pars,
			const vec_t& neg_step_dir,
			bool use_nesterov_acc,
			int it,
			vec_t& cov_pars_after_grad_aux,
			vec_t& cov_pars_after_grad_aux_lag1,
			double acc_rate_cov,
			int nesterov_schedule_version,
			int momentum_offset,
			const double* fixed_effects) {
			if (use_nesterov_acc && nesterov_schedule_version == 1 && armijo_condition_) {
				Log::REFatal("Armijo condition backtracking is not implemented when nesterov_schedule_version = 1");
			}
			vec_t cov_pars_new(num_cov_par_);
			if (profile_out_marginal_variance_) {
				cov_pars_new[0] = cov_pars[0];
			}
			double lr_cov = lr_cov_;
			double lr_aux_pars = lr_aux_pars_;
			bool decrease_found = false;
			bool halving_done = false;
			int num_grad_cov_par = (int)neg_step_dir.size();
			if (estimate_aux_pars_) {
				num_grad_cov_par -= NumAuxPars();
			}
			if (it == 0) {
				first_update_ = true;
			}
			else {
				first_update_ = false;
			}
			for (int ih = 0; ih < MAX_NUMBER_LR_SHRINKAGE_STEPS_; ++ih) {
				vec_t update(neg_step_dir.size());
				update.segment(0, num_grad_cov_par) = lr_cov * neg_step_dir.segment(0, num_grad_cov_par);
				if (estimate_aux_pars_) {
					update.segment(num_grad_cov_par, NumAuxPars()) = lr_aux_pars * neg_step_dir.segment(num_grad_cov_par, NumAuxPars());
				}
				// Avoid to large steps on log-scale: updates on the log-scale in one Fisher scoring step are capped at a certain level
				// This is not done for gradient_descent since the learning rate is already adjusted accordingly in 'AvoidTooLargeLearningRatesCovAuxPars'
				if (optimizer_cov_pars_ == "fisher_scoring" || optimizer_cov_pars_ == "newton") {
					for (int ip = 0; ip < (int)update.size(); ++ip) {
						if (update[ip] > MAX_GRADIENT_UPDATE_LOG_SCALE_) {
							update[ip] = MAX_GRADIENT_UPDATE_LOG_SCALE_;
						}
						else if (update[ip] < -MAX_GRADIENT_UPDATE_LOG_SCALE_) {
							update[ip] = -MAX_GRADIENT_UPDATE_LOG_SCALE_;
						}
					}
				}
				if (profile_out_marginal_variance_) {
					cov_pars_new.segment(1, cov_pars.size() - 1) = (cov_pars.segment(1, cov_pars.size() - 1).array().log() - update.array()).exp().matrix();//make update on log-scale
				}
				else {
					cov_pars_new = (cov_pars.array().log() - update.array()).exp().matrix();//make update on log-scale
				}
				// Apply Nesterov acceleration
				if (use_nesterov_acc) {
					cov_pars_after_grad_aux = cov_pars_new;
					ApplyMomentumStep(it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1, cov_pars_new, acc_rate_cov,
						nesterov_schedule_version, profile_out_marginal_variance_, momentum_offset, true);
					// Note: (i) cov_pars_after_grad_aux and cov_pars_after_grad_aux_lag1 correspond to the parameters obtained after calculating the gradient before applying acceleration
					//		 (ii) cov_pars (below this) are the parameters obtained after applying acceleration (and cov_pars_lag1 is simply the value of the previous iteration)
					// We first apply a gradient step and then an acceleration step (and not the other way aroung) since this is computationally more efficient 
					//		(otherwise the covariance matrix needs to be factored twice: once for the gradient step (accelerated parameters) and once for calculating the
					//		 log-likelihood (non-accelerated parameters after gradient update) when checking for convergence at the end of an iteration. 
					//		However, performing the acceleration before or after the gradient update gives equivalent algorithms
				}
				if (estimate_aux_pars_) {
					SetAuxPars(cov_pars_new.data() + num_cov_par_);
				}
				CalcCovFactorOrModeAndNegLL(cov_pars_new.segment(0, num_cov_par_), fixed_effects);
				// Safeguard against too large steps by halving the learning rate when the objective increases
				if (armijo_condition_) {
					double mu;
					if (use_nesterov_acc) {
						mu = NesterovSchedule(it, nesterov_schedule_version, acc_rate_cov, momentum_offset);
					}
					else {
						mu = 0.;
					}
					if (estimate_aux_pars_) {
						if ((neg_log_likelihood_ <= (neg_log_likelihood_after_lin_coef_update_ +
							c_armijo_ * lr_cov * dir_deriv_armijo_cov_pars_ + c_armijo_mom_ * mu * mom_dir_deriv_armijo_cov_pars_)) &&
							(neg_log_likelihood_ <= (neg_log_likelihood_after_lin_coef_update_ +
								c_armijo_ * lr_aux_pars * dir_deriv_armijo_aux_pars_ + c_armijo_mom_ * mu * mom_dir_deriv_armijo_aux_pars_))) {
							decrease_found = true;
						}
					}
					else {//!estimate_aux_pars_
						if (neg_log_likelihood_ <= (neg_log_likelihood_after_lin_coef_update_ +
							c_armijo_ * lr_cov * dir_deriv_armijo_cov_pars_ + c_armijo_mom_ * mu * mom_dir_deriv_armijo_cov_pars_)) {
							decrease_found = true;
						}
					}
				}
				else {
					if (neg_log_likelihood_ <= neg_log_likelihood_after_lin_coef_update_) {
						decrease_found = true;
					}
				}
				if (decrease_found) {
					break;
				}
				else {
					halving_done = true;
					learning_rate_decreased_first_time_ = true;
					if (learning_rate_increased_after_descrease_) {
						learning_rate_decreased_after_increase_ = true;
					}
					lr_cov *= LR_SHRINKAGE_FACTOR_;
					if (estimate_aux_pars_) {
						lr_aux_pars *= LR_SHRINKAGE_FACTOR_;
					}
					acc_rate_cov *= 0.5;
					if (!gauss_likelihood_) {
						// Reset mode to previous value since also parameters are discarded
						for (const auto& cluster_i : unique_clusters_) {
							likelihood_[cluster_i]->ResetModeToPreviousValue();
						}
					}
				}
			}//end loop over learnig rate halving procedure
			if (halving_done) {
				if ((optimizer_cov_pars_ == "fisher_scoring" || optimizer_cov_pars_ == "newton") &&
					!learning_rate_constant_first_order_change_) {
					Log::REDebug("GPModel covariance parameter estimation: No decrease in the objective function in iteration number %d. "
						"The learning rate has been decreased in this iteration ", it + 1);
				}
				else if (optimizer_cov_pars_ == "gradient_descent") {
					lr_cov_ = lr_cov; //permanently decrease learning rate (for Fisher scoring, this is not done. I.e., step halving is done newly in every iterarion of Fisher scoring)
					if (estimate_aux_pars_) {
						lr_aux_pars_ = lr_aux_pars;
						Log::REDebug("GPModel covariance and auxiliary parameter estimation: Learning rates have been decreased permanently in iteration number %d "
							"since with the previous learning rates, there was no decrease in the objective function. "
							"New learning rates: covariance parameters = %g, auxiliary parameters = %g", it + 1, lr_cov_, lr_aux_pars_);
					}
					else {
						Log::REDebug("GPModel covariance parameter estimation: The learning rate has been decreased permanently in iteration number %d "
							"since with the previous learning rate, there was no decrease in the objective function. New learning rate = %g", it + 1, lr_cov_);
					}
				}
			}
			if (!decrease_found) {
				Log::REDebug("GPModel covariance parameter estimation: No decrease in the objective function in iteration number %d "
					"after the maximal number of halving steps (%d) ", it + 1, MAX_NUMBER_LR_SHRINKAGE_STEPS_);
			}
			if (use_nesterov_acc) {
				cov_pars_after_grad_aux_lag1 = cov_pars_after_grad_aux;
			}
			cov_pars = cov_pars_new;
		}//end UpdateCovAuxPars

		////Alternative version with separate learning rate descreases for cov_pars and aux_pars
		////	-> not necessarily more efficient as more likelihood evaluations are needed
		//void UpdateCovAuxPars(vec_t& cov_pars,
		//	const vec_t& neg_step_dir,
		//	bool profile_out_marginal_variance,
		//	bool use_nesterov_acc,
		//	int it,
		//	vec_t& cov_pars_after_grad_aux,
		//	vec_t& cov_pars_after_grad_aux_lag1,
		//	double acc_rate_cov,
		//	int nesterov_schedule_version,
		//	int momentum_offset,
		//	const double* fixed_effects) {
		//	vec_t cov_pars_new(num_cov_par_);
		//	if (profile_out_marginal_variance) {
		//		cov_pars_new[0] = cov_pars[0];
		//	}
		//	double lr_cov = lr_cov_;
		//	double lr_aux_pars = lr_aux_pars_;
		//	bool decrease_found = false;
		//	bool halving_done = false;
		//	int num_grad_cov_par = (int)neg_step_dir.size();
		//	if (estimate_aux_pars_) {
		//		num_grad_cov_par -= NumAuxPars();
		//	}
		//	for (int ih = 0; ih < MAX_NUMBER_LR_SHRINKAGE_STEPS_; ++ih) {
		//		if (ih > 0) {
		//			halving_done = true;
		//			//learning_rate_decreased_first_time_ = true;
		//			//if (learning_rate_increased_after_descrease_) {
		//			//	learning_rate_decreased_after_increase_ = true;
		//			//}
		//			lr_cov *= LR_SHRINKAGE_FACTOR_;
		//			acc_rate_cov *= 0.5;
		//			if (!gauss_likelihood_) {
		//				// Reset mode to previous value since also parameters are discarded
		//				for (const auto& cluster_i : unique_clusters_) {
		//					likelihood_[cluster_i]->ResetModeToPreviousValue();
		//				}
		//			}
		//		}// end ih > 0
		//		UpdateCovAuxParsInternal(cov_pars, cov_pars_new, neg_step_dir, num_grad_cov_par, lr_cov, lr_aux_pars,
		//			profile_out_marginal_variance, use_nesterov_acc, it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1,
		//			acc_rate_cov, nesterov_schedule_version, momentum_offset);
		//		if (estimate_aux_pars_) {
		//			SetAuxPars(cov_pars_new.data() + num_cov_par_);
		//		}
		//		CalcCovFactorOrModeAndNegLL(cov_pars_new.segment(0, num_cov_par_), fixed_effects);
		//		if (estimate_aux_pars_ && ih > 0) {
		//			// Undo learning rate decrease for cov_pars and decrease learning rate for aux_pars and check whether this leads to a smaller log-likelihood
		//			lr_cov /= LR_SHRINKAGE_FACTOR_;
		//			lr_aux_pars *= LR_SHRINKAGE_FACTOR_;
		//			UpdateCovAuxParsInternal(cov_pars, cov_pars_new, neg_step_dir, num_grad_cov_par, lr_cov, lr_aux_pars,
		//				profile_out_marginal_variance, use_nesterov_acc, it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1,
		//				acc_rate_cov, nesterov_schedule_version, momentum_offset);
		//			SetAuxPars(cov_pars_new.data() + num_cov_par_);
		//			double neg_log_likelihood_cov_par_decrease = neg_log_likelihood_;
		//			if (!gauss_likelihood_) {
		//				for (const auto& cluster_i : unique_clusters_) {
		//					likelihood_[cluster_i]->ResetModeToPreviousValue();
		//				}
		//			}
		//			CalcCovFactorOrModeAndNegLL(cov_pars_new.segment(0, num_cov_par_), fixed_effects);
		//			if (neg_log_likelihood_cov_par_decrease < neg_log_likelihood_) {
		//				lr_cov *= LR_SHRINKAGE_FACTOR_;
		//				lr_aux_pars /= LR_SHRINKAGE_FACTOR_;
		//				UpdateCovAuxParsInternal(cov_pars, cov_pars_new, neg_step_dir, num_grad_cov_par, lr_cov, lr_aux_pars,
		//					profile_out_marginal_variance, use_nesterov_acc, it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1,
		//					acc_rate_cov, nesterov_schedule_version, momentum_offset);
		//				SetAuxPars(cov_pars_new.data() + num_cov_par_);
		//				neg_log_likelihood_ = neg_log_likelihood_cov_par_decrease;
		//				if (neg_log_likelihood_ <= neg_log_likelihood_after_lin_coef_update_) {
		//					// the following needs only be done in case the for loop stops, otherwise it is done at the beginning of the next iteration
		//					if (!gauss_likelihood_) {
		//						for (const auto& cluster_i : unique_clusters_) {
		//							likelihood_[cluster_i]->ResetModeToPreviousValue();
		//						}
		//					}
		//					CalcCovFactorOrModeAndNegLL(cov_pars_new.segment(0, num_cov_par_), fixed_effects);
		//				}
		//			}
		//		}// end estimate_aux_pars_ && ih > 0
		//		// Safeguard against too large steps by halving the learning rate when the objective increases
		//		if (neg_log_likelihood_ <= neg_log_likelihood_after_lin_coef_update_) {
		//			decrease_found = true;
		//			break;
		//		}
		//	}//end loop over learnig rate halving procedure
		//	if (halving_done) {
		//		if (optimizer_cov_pars_ == "fisher_scoring") {
		//			Log::REDebug("GPModel covariance parameter estimation: No decrease in the objective function in iteration number %d. "
		//				"The learning rate has been decreased in this iteration.", it + 1);
		//		}
		//		else if (optimizer_cov_pars_ == "gradient_descent") {
		//			lr_cov_ = lr_cov; //permanently decrease learning rate (for Fisher scoring, this is not done. I.e., step halving is done newly in every iterarion of Fisher scoring)
		//			if (estimate_aux_pars_) {
		//				lr_aux_pars_ = lr_aux_pars;
		//				Log::REDebug("GPModel covariance and auxiliary parameter estimation: Learning rates have been decreased permanently in iteration number %d "
		//					"since with the previous learning rates, there was no decrease in the objective function. "
		//					"New learning rates: covariance parameters = %g, auxiliary parameters = %g", it + 1, lr_cov_, lr_aux_pars_);
		//			}
		//			else {
		//				Log::REDebug("GPModel covariance parameter estimation: The learning rate has been decreased permanently in iteration number %d "
		//					"since with the previous learning rate, there was no decrease in the objective function. New learning rate = %g", it + 1, lr_cov_);
		//			}
		//		}
		//	}
		//	if (!decrease_found) {
		//		Log::REDebug("GPModel covariance parameter estimation: No decrease in the objective function in iteration number %d "
		//			"after the maximal number of halving steps (%d).", it + 1, MAX_NUMBER_LR_SHRINKAGE_STEPS_);
		//	}
		//	if (use_nesterov_acc) {
		//		cov_pars_after_grad_aux_lag1 = cov_pars_after_grad_aux;
		//	}
		//	cov_pars = cov_pars_new;
		//}//end UpdateCovAuxPars

		//void UpdateCovAuxParsInternal(vec_t& cov_pars,
		//	vec_t& cov_pars_new,
		//	const vec_t& neg_step_dir,
		//	int num_grad_cov_par,
		//	double lr_cov,
		//	double lr_aux_pars,
		//	bool profile_out_marginal_variance,
		//	bool use_nesterov_acc,
		//	int it,
		//	vec_t& cov_pars_after_grad_aux,
		//	vec_t& cov_pars_after_grad_aux_lag1,
		//	double acc_rate_cov,
		//	int nesterov_schedule_version,
		//	int momentum_offset) {
		//	vec_t update(neg_step_dir.size());
		//	update.segment(0, num_grad_cov_par) = lr_cov * neg_step_dir.segment(0, num_grad_cov_par);
		//	if (estimate_aux_pars_) {
		//		update.segment(num_grad_cov_par, NumAuxPars()) = lr_aux_pars * neg_step_dir.segment(num_grad_cov_par, NumAuxPars());
		//	}
		//	// Avoid to large steps on log-scale: updates on the log-scale in one Fisher scoring step are capped at a certain level
		//	// This is not done for gradient_descent since the learning rate is already adjusted accordingly in 'AvoidTooLargeLearningRatesCovAuxPars'
		//	if (optimizer_cov_pars_ != "gradient_descent") {
		//		for (int ip = 0; ip < (int)update.size(); ++ip) {
		//			if (update[ip] > MAX_GRADIENT_UPDATE_LOG_SCALE_) {
		//				update[ip] = MAX_GRADIENT_UPDATE_LOG_SCALE_;
		//			}
		//			else if (update[ip] < -MAX_GRADIENT_UPDATE_LOG_SCALE_) {
		//				update[ip] = -MAX_GRADIENT_UPDATE_LOG_SCALE_;
		//			}
		//		}
		//	}
		//	if (profile_out_marginal_variance) {
		//		cov_pars_new.segment(1, cov_pars.size() - 1) = (cov_pars.segment(1, cov_pars.size() - 1).array().log() - update.array()).exp().matrix();//make update on log-scale
		//	}
		//	else {
		//		cov_pars_new = (cov_pars.array().log() - update.array()).exp().matrix();//make update on log-scale
		//	}
		//	// Apply Nesterov acceleration
		//	if (use_nesterov_acc) {
		//		cov_pars_after_grad_aux = cov_pars_new;
		//		ApplyMomentumStep(it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1, cov_pars_new, acc_rate_cov,
		//			nesterov_schedule_version, profile_out_marginal_variance, momentum_offset, true);
		//		// Note: (i) cov_pars_after_grad_aux and cov_pars_after_grad_aux_lag1 correspond to the parameters obtained after calculating the gradient before applying acceleration
		//		//		 (ii) cov_pars (below this) are the parameters obtained after applying acceleration (and cov_pars_lag1 is simply the value of the previous iteration)
		//		// We first apply a gradient step and then an acceleration step (and not the other way aroung) since this is computationally more efficient 
		//		//		(otherwise the covariance matrix needs to be factored twice: once for the gradient step (accelerated parameters) and once for calculating the
		//		//		 log-likelihood (non-accelerated parameters after gradient update) when checking for convergence at the end of an iteration. 
		//		//		However, performing the acceleration before or after the gradient update gives equivalent algorithms
		//	}
		//}//end UpdateCovAuxParsInternal

		////Alternative version where learning rates are decreased until 
		//// (i) a decrease in the log-likelihood compared to the previous one is found and 
		//// (ii) subsequent decreases are only small.
		//// Problem: this does not necessarily lead to less log-likelihood evaluations compared to the above version
		//void UpdateCovAuxPars(vec_t& cov_pars,
		//	const vec_t& neg_step_dir,
		//	bool profile_out_marginal_variance,
		//	bool use_nesterov_acc,
		//	int it,
		//	vec_t& cov_pars_after_grad_aux,
		//	vec_t& cov_pars_after_grad_aux_lag1,
		//	double acc_rate_cov,
		//	int nesterov_schedule_version,
		//	int momentum_offset,
		//	const double* fixed_effects) {
		//	vec_t cov_pars_new(num_cov_par_);
		//	if (profile_out_marginal_variance) {
		//		cov_pars_new[0] = cov_pars[0];
		//	}
		//	double lr_cov = lr_cov_;
		//	double lr_aux_pars = lr_aux_pars_;
		//	bool decrease_found = false;
		//	bool halving_done = false;
		//	double cur_lowest_neg_log_like = neg_log_likelihood_after_lin_coef_update_;// currently lowest negative log-likelihood
		//	bool lr_cov_last_decreased = true;// indicates whether lr_cov or lr_aux_pars has been last descreased
		//	int num_grad_cov_par = (int)neg_step_dir.size();
		//	if (estimate_aux_pars_) {
		//		num_grad_cov_par -= NumAuxPars();
		//	}
		//	for (int ih = 0; ih < MAX_NUMBER_LR_SHRINKAGE_STEPS_; ++ih) {
		//		if (ih > 0) {
		//			learning_rate_decreased_first_time_ = true;
		//			if (learning_rate_increased_after_descrease_) {
		//				learning_rate_decreased_after_increase_ = true;
		//			}
		//			acc_rate_cov *= 0.5;
		//			if (ih == 1 || lr_cov_last_decreased || !decrease_found) {
		//				//lr_cov is decreased until there is a decrease found and subsequently only if if has been decreased last
		//				lr_cov *= LR_SHRINKAGE_FACTOR_;
		//				if (!gauss_likelihood_) {
		//					// Reset mode to previous value since also parameters are discarded
		//					for (const auto& cluster_i : unique_clusters_) {
		//						likelihood_[cluster_i]->ResetModeToPreviousValue();
		//					}
		//				}
		//				lr_cov_last_decreased = true;
		//			}
		//		}//end ih > 0
		//		UpdateCovAuxParsInternal(cov_pars, cov_pars_new, neg_step_dir, num_grad_cov_par, lr_cov, lr_aux_pars,
		//			profile_out_marginal_variance, use_nesterov_acc, it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1,
		//			acc_rate_cov, nesterov_schedule_version, momentum_offset);
		//		if (estimate_aux_pars_) {
		//			SetAuxPars(cov_pars_new.data() + num_cov_par_);
		//		}
		//		CalcCovFactorOrModeAndNegLL(cov_pars_new.segment(0, num_cov_par_), fixed_effects);
		//		if (estimate_aux_pars_ && ih > 0 && (ih == 1 || !lr_cov_last_decreased || !decrease_found)) {
		//			//lr_aux_pars is decreased until there is a decrease found and subsequently only if if has been decreased last
		//			if (lr_cov_last_decreased) {
		//				// Undo decrease for 'lr_cov', decrease 'lr_aux_pars', and check whether this leads to a smaller negative log-likelihood
		//				lr_cov /= LR_SHRINKAGE_FACTOR_;
		//			}
		//			lr_aux_pars *= LR_SHRINKAGE_FACTOR_;
		//			UpdateCovAuxParsInternal(cov_pars, cov_pars_new, neg_step_dir, num_grad_cov_par, lr_cov, lr_aux_pars,
		//				profile_out_marginal_variance, use_nesterov_acc, it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1,
		//				acc_rate_cov, nesterov_schedule_version, momentum_offset);
		//			SetAuxPars(cov_pars_new.data() + num_cov_par_);
		//			double neg_log_likelihood_cov_par_decrease = neg_log_likelihood_;
		//			if (!gauss_likelihood_) {
		//				for (const auto& cluster_i : unique_clusters_) {
		//					likelihood_[cluster_i]->ResetModeToPreviousValue();
		//				}
		//			}
		//			CalcCovFactorOrModeAndNegLL(cov_pars_new.segment(0, num_cov_par_), fixed_effects);
		//			if (lr_cov_last_decreased && neg_log_likelihood_cov_par_decrease < neg_log_likelihood_) {
		//				// Better to decrease 'lr_cov': undo decrease in 'lr_aux_pars'
		//				lr_cov *= LR_SHRINKAGE_FACTOR_;
		//				lr_aux_pars /= LR_SHRINKAGE_FACTOR_;
		//				UpdateCovAuxParsInternal(cov_pars, cov_pars_new, neg_step_dir, num_grad_cov_par, lr_cov, lr_aux_pars,
		//					profile_out_marginal_variance, use_nesterov_acc, it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1,
		//					acc_rate_cov, nesterov_schedule_version, momentum_offset);
		//				SetAuxPars(cov_pars_new.data() + num_cov_par_);
		//				neg_log_likelihood_ = neg_log_likelihood_cov_par_decrease;
		//			}
		//			else {
		//				lr_cov_last_decreased = false;
		//			}
		//		}// end estimate_aux_pars_ && ih > 0
		//		if (neg_log_likelihood_ <= neg_log_likelihood_after_lin_coef_update_) {
		//			decrease_found = true;
		//		}
		//		// Stop trying more decreases when decrease is only small
		//		if (((cur_lowest_neg_log_like - neg_log_likelihood_) <= 2.) && decrease_found) {
		//			if (ih > 1) {
		//				acc_rate_cov /= 0.5;
		//				if (lr_cov_last_decreased) {
		//					// Undo decrease for 'lr_cov'
		//					lr_cov /= LR_SHRINKAGE_FACTOR_;
		//				}
		//				else {
		//					lr_aux_pars /= LR_SHRINKAGE_FACTOR_;
		//					SetAuxPars(cov_pars_new.data() + num_cov_par_);
		//				}
		//				UpdateCovAuxParsInternal(cov_pars, cov_pars_new, neg_step_dir, num_grad_cov_par, lr_cov, lr_aux_pars,
		//					profile_out_marginal_variance, use_nesterov_acc, it, cov_pars_after_grad_aux, cov_pars_after_grad_aux_lag1,
		//					acc_rate_cov, nesterov_schedule_version, momentum_offset);
		//				if (!gauss_likelihood_) {
		//					for (const auto& cluster_i : unique_clusters_) {
		//						likelihood_[cluster_i]->ResetModeToPreviousValue();
		//					}
		//				}
		//				CalcCovFactorOrModeAndNegLL(cov_pars_new.segment(0, num_cov_par_), fixed_effects);
		//				halving_done = true;
		//			}
		//			break;// ll increases again after a decrease -> stop
		//		}
		//		if (neg_log_likelihood_ <= cur_lowest_neg_log_like) {
		//			cur_lowest_neg_log_like = neg_log_likelihood_;
		//		}
		//	}//end loop over learnig rate halving procedure
		//	if (halving_done) {
		//		if (optimizer_cov_pars_ == "fisher_scoring") {
		//			Log::REDebug("GPModel covariance parameter estimation: No decrease in the objective function in iteration number %d. "
		//				"The learning rate has been decreased in this iteration.", it + 1);
		//		}
		//		else if (optimizer_cov_pars_ == "gradient_descent") {
		//			lr_cov_ = lr_cov; //permanently decrease learning rate (for Fisher scoring, this is not done. I.e., step halving is done newly in every iterarion of Fisher scoring)
		//			if (estimate_aux_pars_) {
		//				lr_aux_pars_ = lr_aux_pars;
		//				Log::REDebug("GPModel covariance and auxiliary parameter estimation: Learning rates have been decreased permanently in iteration number %d "
		//					"since with the previous learning rates, there was no decrease in the objective function. "
		//					"New learning rates: covariance parameters = %g, auxiliary parameters = %g", it + 1, lr_cov_, lr_aux_pars_);
		//			}
		//			else {
		//				Log::REDebug("GPModel covariance parameter estimation: The learning rate has been decreased permanently in iteration number %d "
		//					"since with the previous learning rate, there was no decrease in the objective function. New learning rate = %g", it + 1, lr_cov_);
		//			}
		//		}
		//	}
		//	if (!decrease_found) {
		//		Log::REDebug("GPModel covariance parameter estimation: No decrease in the objective function in iteration number %d "
		//			"after the maximal number of halving steps (%d).", it + 1, MAX_NUMBER_LR_SHRINKAGE_STEPS_);
		//	}
		//	if (use_nesterov_acc) {
		//		cov_pars_after_grad_aux_lag1 = cov_pars_after_grad_aux;
		//	}
		//	cov_pars = cov_pars_new;
		//}//end UpdateCovAuxPars

		/*!
		* \brief Update linear regression coefficients and apply step size safeguard
		* \param[out] beta Linear regression coefficients
		* \param grad Gradient
		* \param sigma2 Nugget / error term variance for Gaussian likelihoods
		* \param use_nesterov_acc If true, Nesterov acceleration is used
		* \param it Iteration number
		* \param[out] beta_after_grad_aux Auxiliary variable used only if use_nesterov_acc == true (see the code below for a description)
		* \param[out] beta_after_grad_aux_lag1 Auxiliary variable used only if use_nesterov_acc == true (see the code below for a description)
		* \param acc_rate_coef Nesterov acceleration speed
		* \param nesterov_schedule_version Which version of Nesterov schedule should be used. Default = 0
		* \param momentum_offset Number of iterations for which no mometum is applied in the beginning
		* \param fixed_effects External fixed effects
		* \param[out] fixed_effects_vec Fixed effects component of location parameter as sum of linear predictor and potentiall additional external fixed effects
		*/
		void UpdateLinCoef(vec_t& beta,
			const vec_t& grad,
			const double sigma2,
			bool use_nesterov_acc,
			int it,
			vec_t& beta_after_grad_aux,
			vec_t& beta_after_grad_aux_lag1,
			double acc_rate_coef,
			int nesterov_schedule_version,
			int momentum_offset,
			const double* fixed_effects,
			vec_t& fixed_effects_vec) {
			vec_t beta_new;
			double lr_coef = lr_coef_;
			bool decrease_found = false;
			bool halving_done = false;
			if (it == 0) {
				first_update_ = true;
			}
			else {
				first_update_ = false;
			}
			for (int ih = 0; ih < MAX_NUMBER_LR_SHRINKAGE_STEPS_; ++ih) {
				beta_new = beta - lr_coef * grad;
				// Apply Nesterov acceleration
				if (use_nesterov_acc) {
					beta_after_grad_aux = beta_new;
					ApplyMomentumStep(it, beta_after_grad_aux, beta_after_grad_aux_lag1, beta_new, acc_rate_coef,
						nesterov_schedule_version, false, momentum_offset, false);
					//Note: use same version of Nesterov acceleration as for covariance parameters (see 'UpdateCovAuxPars')
				}
				UpdateFixedEffects(beta_new, fixed_effects, fixed_effects_vec);
				if (gauss_likelihood_) {
					EvalNegLogLikelihoodOnlyUpdateFixedEffects(sigma2, neg_log_likelihood_after_lin_coef_update_);
				}//end if gauss_likelihood_
				else {//non-Gaussian likelihoods
					neg_log_likelihood_after_lin_coef_update_ = -CalcModePostRandEffCalcMLL(fixed_effects_vec.data(), true);//calculate mode and approximate marginal likelihood
				}
				// Safeguard against too large steps by halving the learning rate when the objective increases
				if (armijo_condition_) {
					double mu;
					if (use_nesterov_acc) {
						mu = NesterovSchedule(it, nesterov_schedule_version, acc_rate_coef, momentum_offset);
					}
					else {
						mu = 0.;
					}
					if (neg_log_likelihood_after_lin_coef_update_ <= (neg_log_likelihood_lag1_ +
						c_armijo_ * lr_coef * dir_deriv_armijo_coef_ + c_armijo_mom_ * mu * mom_dir_deriv_armijo_coef_)) {
						decrease_found = true;
					}
				}
				else {
					if (neg_log_likelihood_after_lin_coef_update_ <= neg_log_likelihood_lag1_) {
						decrease_found = true;
					}
				}
				if (decrease_found) {
					break;
				}
				else {
					// Safeguard against too large steps by halving the learning rate
					halving_done = true;
					learning_rate_decreased_first_time_ = true;
					if (learning_rate_increased_after_descrease_) {
						learning_rate_decreased_after_increase_ = true;
					}
					lr_coef *= LR_SHRINKAGE_FACTOR_;
					acc_rate_coef *= 0.5;
					if (!gauss_likelihood_) {
						// Reset mode to previous value since also parameters are discarded
						for (const auto& cluster_i : unique_clusters_) {
							likelihood_[cluster_i]->ResetModeToPreviousValue();
						}
					}
				}
			}
			if (halving_done) {
				lr_coef_ = lr_coef; //permanently decrease learning rate
				Log::REDebug("GPModel: The learning rate for the regression coefficients has been decreased permanently since with the previous learning rate, "
					"there was no decrease in the objective function in iteration number %d. New learning rate = %g", it + 1, lr_coef_);
			}
			if (!decrease_found) {
				Log::REDebug("GPModel linear regression coefficient estimation: No decrease in the objective function "
					"in iteration number %d after the maximal number of halving steps (%d).", it + 1, MAX_NUMBER_LR_SHRINKAGE_STEPS_);
			}
			if (use_nesterov_acc) {
				beta_after_grad_aux_lag1 = beta_after_grad_aux;
			}
			beta = beta_new;
		}//end UpdateLinCoef

		/*!
		* \brief Calculate the covariance matrix ZSigmaZt of the random effects (sum of all components)
		* \param[out] ZSigmaZt Covariance matrix ZSigmaZt
		* \param cluster_i Cluster index for which the covariance matrix is calculated
		*/
		void CalcZSigmaZt(T_mat& ZSigmaZt, data_size_t cluster_i) {
			ZSigmaZt = T_mat(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
			if (gauss_likelihood_) {
				ZSigmaZt.setIdentity();
			}
			else {
				ZSigmaZt.setZero();
			}
			for (int j = 0; j < num_comps_total_; ++j) {
				ZSigmaZt += (*(re_comps_[cluster_i][j]->GetZSigmaZt()));
			}
		}//end CalcZSigmaZt

		/*!
		* \brief Calculate the covariance matrix ZSigmaZt if only_grouped_REs_use_woodbury_identity_==false or the inverse covariance matrix Sigma^-1 if there are only grouped REs i.e. if only_grouped_REs_use_woodbury_identity_==true.
		*		This function is only used for non-Gaussian likelihoods as in the Gaussian case this needs not be saved
		*/
		void CalcCovMatrixNonGauss() {
			if (!only_one_grouped_RE_calculations_on_RE_scale_) {//Nothing to calculate if only_one_grouped_RE_calculations_on_RE_scale_
				if (only_grouped_REs_use_woodbury_identity_) {
					for (const auto& cluster_i : unique_clusters_) {
						CalcSigmaIGroupedREsOnly(SigmaI_[cluster_i], cluster_i, true);
					}
				}
				else {
					for (const auto& cluster_i : unique_clusters_) {
						if (num_comps_total_ == 1) {//no need to sum up different components
							ZSigmaZt_[cluster_i] = re_comps_[cluster_i][0]->GetZSigmaZt();
						}
						else {
							T_mat ZSigmaZt;
							CalcZSigmaZt(ZSigmaZt, cluster_i);
							ZSigmaZt_[cluster_i] = std::make_shared<T_mat>(ZSigmaZt);
						}
					}
				}
			}
		}//end CalcCovMatrixNonGauss

		/*!
		* \brief Calculate the mode of the posterior of the latent random effects and the Laplace-approximated marginal log-likelihood. This function is only used for non-Gaussian likelihoods
		* \param fixed_effects Fixed effects component of location parameter
		* \param calc_mll If true the marginal log-likelihood is also calculated (only relevant for gp_approx_ == "vecchia" && matrix_inversion_method_ == "iterative")
		* \return Approximate marginal log-likelihood evaluated at the mode
		*/
		double CalcModePostRandEffCalcMLL(const double* fixed_effects,
			bool calc_mll) {
			double mll = 0.;
			double mll_cluster_i;
			const double* fixed_effects_cluster_i_ptr = nullptr;
			vec_t fixed_effects_cluster_i;
			for (const auto& cluster_i : unique_clusters_) {
				if (num_clusters_ == 1 && (gp_approx_ != "vecchia" || vecchia_ordering_ == "none")) {//only one cluster / independent realization and order of data does not matter
					fixed_effects_cluster_i_ptr = fixed_effects;
				}
				else if (fixed_effects != nullptr) {//more than one cluster and order of samples matters
					fixed_effects_cluster_i = vec_t(num_data_per_cluster_[cluster_i]);
					//Note: this is quite inefficient as the mapping of the fixed_effects to the different clusters is done repeatedly for the same data. Could be saved if performance is an issue here. 
#pragma omp parallel for schedule(static)
					for (int j = 0; j < num_data_per_cluster_[cluster_i]; ++j) {
						fixed_effects_cluster_i[j] = fixed_effects[data_indices_per_cluster_[cluster_i][j]];
					}
					fixed_effects_cluster_i_ptr = fixed_effects_cluster_i.data();
				}
				if (gp_approx_ == "vecchia") {
					den_mat_t Sigma_L_k;
					if (matrix_inversion_method_ == "iterative" && cg_preconditioner_type_ == "piv_chol_on_Sigma") {
						//Do pivoted Cholesky decomposition for Sigma
						//TODO: only after cov-pars step, not after fixed-effect step
						PivotedCholsekyFactorizationSigma(re_comps_[cluster_i][ind_intercept_gp_].get(), Sigma_L_k, piv_chol_rank_, PIV_CHOL_STOP_TOL);
					}
					likelihood_[cluster_i]->FindModePostRandEffCalcMLLVecchia(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						B_[cluster_i],
						D_inv_[cluster_i],
						first_update_,
						Sigma_L_k,
						calc_mll,
						mll_cluster_i);
				}
				else if (only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_) {
					likelihood_[cluster_i]->FindModePostRandEffCalcMLLGroupedRE(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						num_data_per_cluster_[cluster_i],
						SigmaI_[cluster_i],
						Zt_[cluster_i],
						mll_cluster_i);
				}
				else if (only_one_grouped_RE_calculations_on_RE_scale_) {
					likelihood_[cluster_i]->FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						num_data_per_cluster_[cluster_i],
						re_comps_[cluster_i][0]->cov_pars_[0],
						re_comps_[cluster_i][0]->random_effects_indices_of_data_.data(),
						mll_cluster_i);
				}
				else {
					likelihood_[cluster_i]->FindModePostRandEffCalcMLLStable(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						ZSigmaZt_[cluster_i],
						mll_cluster_i);
					//Note: if(only_one_GP_calculations_on_RE_scale_), ZSigmaZt_[cluster_i] contains Sigma=Cov(b) and not Z*Sigma*Zt since has_Z_==false for this random effects component
				}
				mll += mll_cluster_i;
			}
			num_ll_evaluations_++;
			return(mll);
		}//CalcModePostRandEffCalcMLL

		void CalcCovFactorsPPFSA() {
			for (const auto& cluster_i : unique_clusters_) {
				// factorize matrix used in Woodbury identity
				std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov_[cluster_i][0]->GetZSigmaZt();
				den_mat_t sigma_ip_stable = *(re_comps_ip_[cluster_i][0]->GetZSigmaZt());
				den_mat_t sigma_woodbury;// sigma_woodbury = sigma_ip + cross_cov^T * sigma_resid^-1 * cross_cov or for Preconditioner sigma_ip + cross_cov^T * D^-1 * cross_cov
				if (matrix_inversion_method_ == "iterative") {
					if (gp_approx_ == "fitc") {
						Log::REFatal("The iterative methods are not implemented for Predictive Processes. Please use Cholesky.");
					}
					else if (gp_approx_ == "full_scale_tapering") {
						std::shared_ptr<T_mat> sigma_resid = re_comps_resid_[cluster_i][0]->GetZSigmaZt();
						if (cg_preconditioner_type_ == "predictive_process_plus_diagonal") {
							diagonal_approx_preconditioner_[cluster_i] = (*sigma_resid).diagonal();
							diagonal_approx_inv_preconditioner_[cluster_i] = diagonal_approx_preconditioner_[cluster_i].cwiseInverse();
							sigma_woodbury = (*cross_cov).transpose() * (diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal() * (*cross_cov));
							sigma_woodbury += *(re_comps_ip_[cluster_i][0]->GetZSigmaZt());

							chol_fact_woodbury_preconditioner_[cluster_i].compute(sigma_woodbury);
						}
						else if (cg_preconditioner_type_ != "none") {
							Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
						}

					}
				}
				else if (matrix_inversion_method_ == "cholesky") {
					if (gp_approx_ == "fitc") {
						den_mat_t sigma_ip_Ihalf_sigma_cross_covT;
						TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_[cluster_i],
							(*cross_cov).transpose(), sigma_ip_Ihalf_sigma_cross_covT, false);
						if (gauss_likelihood_) {
							FITC_Diag_[cluster_i] = vec_t::Ones(num_data_per_cluster_[cluster_i]);//add nugget effect variance
						}
						else {
							FITC_Diag_[cluster_i] = vec_t::Zero(num_data_per_cluster_[cluster_i]);
						}
						FITC_Diag_[cluster_i] = FITC_Diag_[cluster_i].array() + sigma_ip_stable.coeffRef(0, 0);
#pragma omp parallel for schedule(static)
						for (int ii = 0; ii < num_data_per_cluster_[cluster_i]; ++ii) {
							FITC_Diag_[cluster_i][ii] -= sigma_ip_Ihalf_sigma_cross_covT.col(ii).array().square().sum();
						}
						sigma_woodbury = ((*cross_cov).transpose() * FITC_Diag_[cluster_i].cwiseInverse().asDiagonal()) * (*cross_cov);
					}
					else if (gp_approx_ == "full_scale_tapering") {
						// factorize residual covariance matrix
						std::shared_ptr<T_mat> sigma_resid = re_comps_resid_[cluster_i][0]->GetZSigmaZt();
						CalcCholFSAResid(*sigma_resid, cluster_i);
						den_mat_t sigma_resid_Ihalf_cross_cov;
						//ApplyPermutationCholeskyFactor<den_mat_t, T_chol>(chol_fact_resid_[cluster_i], *cross_cov, sigma_resid_Ihalf_cross_cov, false);//DELETE_SOLVEINPLACE
						//chol_fact_resid_[cluster_i].matrixL().solveInPlace(sigma_resid_Ihalf_cross_cov);
						TriangularSolveGivenCholesky<T_chol, T_mat, den_mat_t, den_mat_t>(chol_fact_resid_[cluster_i], *cross_cov, sigma_resid_Ihalf_cross_cov, false);
						sigma_woodbury = sigma_resid_Ihalf_cross_cov.transpose() * sigma_resid_Ihalf_cross_cov;
					}
					sigma_woodbury += sigma_ip_stable;
					chol_fact_sigma_woodbury_[cluster_i].compute(sigma_woodbury);
				}
				else {
					Log::REFatal("Matrix inversion method '%s' is not supported.", matrix_inversion_method_.c_str());
				}
			}
		}//end CalcCovFactorsPPFSA

		/*!
		* \brief Create the covariance matrix Psi and factorize it (either calculate a Cholesky factor or the inverse covariance matrix)
		* \param calc_gradient If true, the gradient is also calculated (only for Vecchia approximation)
		* \param transf_scale If true, the derivatives are taken on the transformed scale otherwise on the original scale. Default = true (only for Vecchia approximation)
		* \param nugget_var Nugget effect variance parameter sigma^2 (used only if gp_approx_ == "vecchia" and transf_scale == false to transform back, normally this is equal to one, since the variance paramter is modelled separately and factored out)
		* \param calc_gradient_nugget If true, derivatives are also taken with respect to the nugget / noise variance (only for Vecchia approximation)
		*/
		void CalcCovFactor(bool calc_gradient,
			bool transf_scale,
			double nugget_var,
			bool calc_gradient_nugget) {
			if (gp_approx_ == "vecchia") {
				for (const auto& cluster_i : unique_clusters_) {
					data_size_t num_re_cluster_i = re_comps_[cluster_i][ind_intercept_gp_]->GetNumUniqueREs();
					CalcCovFactorVecchia<T_mat>(num_re_cluster_i, calc_gradient, re_comps_[cluster_i], nearest_neighbors_[cluster_i],
						dist_obs_neighbors_[cluster_i], dist_between_neighbors_[cluster_i],
						entries_init_B_[cluster_i], entries_init_B_grad_[cluster_i], z_outer_z_obs_neighbors_[cluster_i],
						B_[cluster_i], D_inv_[cluster_i], B_grad_[cluster_i], D_grad_[cluster_i], transf_scale, nugget_var,
						calc_gradient_nugget, num_gp_total_, ind_intercept_gp_, gauss_likelihood_);
				}
			}
			else {
				CalcSigmaComps();
				if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
					CalcCovFactorsPPFSA();
				}
				else {
					for (const auto& cluster_i : unique_clusters_) {
						if (only_grouped_REs_use_woodbury_identity_) {//Use Woodburry matrix inversion formula: used only if there are only grouped REs
							if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
								CalcSigmaIGroupedREsOnly(SigmaI_[cluster_i], cluster_i, true);
								sqrt_diag_SigmaI_plus_ZtZ_[cluster_i] = (SigmaI_[cluster_i].diagonal().array() + ZtZ_[cluster_i].diagonal().array()).sqrt().matrix();
							}
							else {
								sp_mat_t SigmaI;
								CalcSigmaIGroupedREsOnly(SigmaI, cluster_i, true);
								T_mat SigmaIplusZtZ = SigmaI + ZtZ_[cluster_i];
								CalcChol(SigmaIplusZtZ, cluster_i);
							}
						}//end only_grouped_REs_use_woodbury_identity_
						else {//not only_grouped_REs_use_woodbury_identity_
							T_mat psi;
							CalcZSigmaZt(psi, cluster_i);
							CalcChol(psi, cluster_i);
						}//end not only_grouped_REs_use_woodbury_identity_
					}
				}
			}
			covariance_matrix_has_been_factorized_ = true;
			num_ll_evaluations_++;
		}//end CalcCovFactor

		/*!
		* \brief Calculate Psi^-1*y (and save in y_aux_)
		* \param marg_variance The marginal variance. Default = 1.
		*/
		void CalcYAux(double marg_variance) {
			for (const auto& cluster_i : unique_clusters_) {
				if (y_.find(cluster_i) == y_.end()) {
					Log::REFatal("Response variable data (y_) for random effects model has not been set. Call 'SetY' first ");
				}
				if (!covariance_matrix_has_been_factorized_) {
					Log::REFatal("Factorisation of covariance matrix has not been done. Call 'CalcCovFactor' first ");
				}
				if (gp_approx_ == "vecchia") {
					y_aux_[cluster_i] = B_[cluster_i].transpose() * D_inv_[cluster_i] * B_[cluster_i] * y_[cluster_i];
				}
				else {//not gp_approx_ == "vecchia"
					if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
						std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov_[cluster_i][0]->GetZSigmaZt();
						if (matrix_inversion_method_ == "cholesky") {
							if (gp_approx_ == "fitc") {
								vec_t cross_covT_y = (*cross_cov).transpose() * (FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * y_[cluster_i]);
								vec_t sigma_woodbury_I_cross_covT_y = chol_fact_sigma_woodbury_[cluster_i].solve(cross_covT_y);
								cross_covT_y.resize(0);
								vec_t cross_cov_sigma_woodbury_I_cross_covT_y = FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * ((*cross_cov) * sigma_woodbury_I_cross_covT_y);
								sigma_woodbury_I_cross_covT_y.resize(0);
								y_aux_[cluster_i] = FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * y_[cluster_i] - cross_cov_sigma_woodbury_I_cross_covT_y;
							}
							else if (gp_approx_ == "full_scale_tapering") {
								vec_t sigma_resid_I_y = chol_fact_resid_[cluster_i].solve(y_[cluster_i]);
								vec_t cross_covT_sigma_resid_I_y = (*cross_cov).transpose() * sigma_resid_I_y;
								vec_t sigma_woodbury_I_cross_covT_sigma_resid_I_y = chol_fact_sigma_woodbury_[cluster_i].solve(cross_covT_sigma_resid_I_y);
								cross_covT_sigma_resid_I_y.resize(0);
								vec_t cross_cov_sigma_woodbury_I_cross_covT_sigma_resid_I_y = (*cross_cov) * sigma_woodbury_I_cross_covT_sigma_resid_I_y;
								sigma_woodbury_I_cross_covT_sigma_resid_I_y.resize(0);
								vec_t sigma_resid_I_cross_cov_sigma_woodbury_I_cross_covT_sigma_resid_I_y = chol_fact_resid_[cluster_i].solve(cross_cov_sigma_woodbury_I_cross_covT_sigma_resid_I_y);
								y_aux_[cluster_i] = sigma_resid_I_y - sigma_resid_I_cross_cov_sigma_woodbury_I_cross_covT_sigma_resid_I_y;
							}
						}
						else {
							std::shared_ptr<T_mat> sigma_resid = re_comps_resid_[cluster_i][0]->GetZSigmaZt();
							y_aux_[cluster_i] = vec_t::Zero(num_data_per_cluster_[cluster_i]);
							CGFSA<T_mat>(*sigma_resid, (*cross_cov), chol_ip_cross_cov_[cluster_i], y_[cluster_i], y_aux_[cluster_i],
								NaN_found, cg_max_num_it_, cg_delta_conv_, THRESHOLD_ZERO_RHS_CG_, cg_preconditioner_type_,
								chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
							if (NaN_found) {
								Log::REFatal("There was Nan or Inf value generated in the Conjugate Gradient Method!");
							}

						}
					}
					else {
						if (only_grouped_REs_use_woodbury_identity_) {
							vec_t MInvZty;
							if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
								MInvZty = (Zty_[cluster_i].array() / sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().square()).matrix();
							}
							else {
								MInvZty = chol_facts_[cluster_i].solve(Zty_[cluster_i]);
							}
							y_aux_[cluster_i] = y_[cluster_i] - Zt_[cluster_i].transpose() * MInvZty;
						}
						else {
							y_aux_[cluster_i] = chol_facts_[cluster_i].solve(y_[cluster_i]);
						}
					}
				}//end non-Vecchia
				if (!TwoNumbersAreEqual<double>(marg_variance, 1.)) {
					y_aux_[cluster_i] /= marg_variance;
				}
			}
			y_aux_has_been_calculated_ = true;
		}

		/*!
		* \brief Calculate y_tilde = L^-1 * Z^T * y, L = chol(Sigma^-1 + Z^T * Z) (and save in y_tilde_)
		* \param also_calculate_ytilde2 If true, y_tilde2 = Z * L^-T * L^-1 * Z^T * y is also calculated
		*/
		void CalcYtilde(bool also_calculate_ytilde2) {
			for (const auto& cluster_i : unique_clusters_) {
				if (y_.find(cluster_i) == y_.end()) {
					Log::REFatal("Response variable data (y_) for random effects model has not been set. Call 'SetY' first ");
				}
				if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
					y_tilde_[cluster_i] = (Zty_[cluster_i].array() / sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array()).matrix();
					if (also_calculate_ytilde2) {
						y_tilde2_[cluster_i] = Zt_[cluster_i].transpose() * ((y_tilde_[cluster_i].array() / sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array()).matrix());
					}
				}
				else {
					TriangularSolveGivenCholesky<T_chol, T_mat, vec_t, vec_t>(chol_facts_[cluster_i], Zty_[cluster_i], y_tilde_[cluster_i], false);
					if (also_calculate_ytilde2) {
						vec_t ytilde_aux;
						TriangularSolveGivenCholesky<T_chol, T_mat, vec_t, vec_t>(chol_facts_[cluster_i], y_tilde_[cluster_i], ytilde_aux, true);
						y_tilde2_[cluster_i] = Zt_[cluster_i].transpose() * ytilde_aux;
					}
				}
			}
		}//end CalcYtilde

		/*!
		* \brief Calculate y^T*Psi^-1*y
		* \param[out] yTPsiInvy y^T*Psi^-1*y
		* \param all_clusters If true, then y^T*Psi^-1*y is calculated for all clusters / data and cluster_ind is ignored
		* \param cluster_ind Cluster index
		* \param CalcYAux_already_done If true, it is assumed that y_aux_=Psi^-1y_ has already been calculated (only relevant for not only_grouped_REs_use_woodbury_identity_)
		* \param CalcYtilde_already_done If true, it is assumed that y_tilde = L^-1 * Z^T * y, L = chol(Sigma^-1 + Z^T * Z), has already been calculated (only relevant for only_grouped_REs_use_woodbury_identity_)
		*/
		void CalcYTPsiIInvY(double& yTPsiInvy,
			bool all_clusters,
			data_size_t cluster_ind,
			bool CalcYAux_already_done,
			bool CalcYtilde_already_done) {
			yTPsiInvy = 0;
			std::vector<data_size_t> clusters_iterate;
			if (all_clusters) {
				clusters_iterate = unique_clusters_;
			}
			else {
				clusters_iterate = std::vector<data_size_t>(1);
				clusters_iterate[0] = cluster_ind;
			}
			for (const auto& cluster_i : clusters_iterate) {
				if (y_.find(cluster_i) == y_.end()) {
					Log::REFatal("Response variable data (y_) for random effects model has not been set. Call 'SetY' first.");
				}
				if (!covariance_matrix_has_been_factorized_) {
					Log::REFatal("Factorisation of covariance matrix has not been done. Call 'CalcCovFactor' first.");
				}
				if (gp_approx_ == "vecchia") {
					if (CalcYAux_already_done) {
						yTPsiInvy += (y_[cluster_i].transpose() * y_aux_[cluster_i])(0, 0);
					}
					else {
						vec_t y_aux_sqrt = B_[cluster_i] * y_[cluster_i];
						yTPsiInvy += (y_aux_sqrt.transpose() * D_inv_[cluster_i] * y_aux_sqrt)(0, 0);
					}
				}//end gp_approx_ == "vecchia"
				else {//not gp_approx_ == "vecchia"
					if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
						if (!CalcYAux_already_done) {
							CalcYAux(1.);//TODO Fabio (low prio): two tringular solves could be saved here, see CalcYAux()
						}
						yTPsiInvy += (y_[cluster_i].transpose() * y_aux_[cluster_i])(0, 0);
					}
					else {
						if (only_grouped_REs_use_woodbury_identity_) {
							if (!CalcYtilde_already_done) {
								CalcYtilde(false);//y_tilde = L^-1 * Z^T * y, L = chol(Sigma^-1 + Z^T * Z)
							}
							else if ((int)y_tilde_[cluster_i].size() != cum_num_rand_eff_[cluster_i][num_re_group_total_]) {
								Log::REFatal("y_tilde = L^-1 * Z^T * y has not the correct number of data points. Call 'CalcYtilde' first.");
							}
							yTPsiInvy += (y_[cluster_i].transpose() * y_[cluster_i])(0, 0) - (y_tilde_[cluster_i].transpose() * y_tilde_[cluster_i])(0, 0);
						}//end only_grouped_REs_use_woodbury_identity_
						else {//not only_grouped_REs_use_woodbury_identity_
							if (CalcYAux_already_done) {
								yTPsiInvy += (y_[cluster_i].transpose() * y_aux_[cluster_i])(0, 0);
							}
							else {
								vec_t y_aux_sqrt;
								TriangularSolveGivenCholesky<T_chol, T_mat, vec_t, vec_t>(chol_facts_[cluster_i], y_[cluster_i], y_aux_sqrt, false);
								yTPsiInvy += (y_aux_sqrt.transpose() * y_aux_sqrt)(0, 0);
							}
						}//end not only_grouped_REs_use_woodbury_identity_
					}
				}//end not gp_approx_ == "vecchia"
			}
		}//end CalcYTPsiIInvY

		/*!
		* \brief Update linear fixed-effect coefficients using generalized least squares (GLS)
		* \param X Covariate data for linear fixed-effect
		* \param[out] beta Linear regression coefficients
		*/
		void UpdateCoefGLS(den_mat_t& X,
			vec_t& beta) {
			vec_t y_aux(num_data_);
			GetYAux(y_aux);
			den_mat_t XT_psi_inv_X;
			CalcXTPsiInvX(X, XT_psi_inv_X);
			beta = XT_psi_inv_X.llt().solve(X.transpose() * y_aux);
		}

		/*!
		* \brief Calculate the Fisher information for covariance parameters. Note: you need to call CalcCovFactor first
		* \param cov_pars Covariance parameters
		* \param[out] FI Fisher information
		* \param transf_scale If true, the derivative is taken on the transformed scale (log-scale) otherwise on the original scale. Default = true
		* \param include_error_var If true, the error variance parameter (=nugget effect) is also included, otherwise not
		* \param use_saved_psi_inv If false, the inverse covariance matrix Psi^-1 is calculated, otherwise a saved version is used
		*/
		void CalcFisherInformation(const vec_t& cov_pars,
			den_mat_t& FI,
			bool transf_scale,
			bool include_error_var,
			bool use_saved_psi_inv) {
			CHECK(gauss_likelihood_);
			if (include_error_var) {
				FI = den_mat_t(num_cov_par_, num_cov_par_);
			}
			else {
				FI = den_mat_t(num_cov_par_ - 1, num_cov_par_ - 1);
			}
			FI.setZero();
			int start_cov_pars = include_error_var ? 1 : 0;

			if (use_stochastic_trace_for_Fisher_information_Vecchia_) {
				if (saved_rand_vec_fisher_info_.size() == 0) {
					for (const auto& cluster_i : unique_clusters_) {
						saved_rand_vec_fisher_info_[cluster_i] = false;
					}
				}
			}

			for (const auto& cluster_i : unique_clusters_) {
				if (gp_approx_ == "vecchia") {
					//Note: if transf_scale==false, then all matrices and derivatives have been calculated on the original scale for the Vecchia approximation, that is why there is no adjustment here

					if (use_stochastic_trace_for_Fisher_information_Vecchia_) {
						// Using Hutchinson's trace estimator
						sp_mat_t D(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
						D.setIdentity();
						D.diagonal().array() = D_inv_[cluster_i].diagonal().array().pow(-1);
						// Sample vectors
						if (!saved_rand_vec_fisher_info_[cluster_i]) {
							if (!cg_generator_seeded_) {
								cg_generator_ = RNG_t(seed_rand_vec_trace_);
								cg_generator_seeded_ = true;
							}
							rand_vec_fisher_info_[cluster_i].resize(num_data_per_cluster_[cluster_i], num_rand_vec_trace_);
							GenRandVecTrace(cg_generator_, rand_vec_fisher_info_[cluster_i]);
							if (reuse_rand_vec_trace_) {//Use same random vectors for each iteration && cluster_i == end(unique_cluster) Tim
								saved_rand_vec_fisher_info_[cluster_i] = true;
							}
						}
						den_mat_t BT_inv_rand_vec;
						TriangularSolve<sp_mat_t, den_mat_t, den_mat_t>(B_[cluster_i], rand_vec_fisher_info_[cluster_i], BT_inv_rand_vec, true);
						den_mat_t D_BT_inv_rand_vec = D * BT_inv_rand_vec;
						den_mat_t Bi_D_BT_inv_rand_vec;
						TriangularSolve<sp_mat_t, den_mat_t, den_mat_t>(B_[cluster_i], D_BT_inv_rand_vec, Bi_D_BT_inv_rand_vec, false);//Bi_D_BT_inv_rand_vec = B^-1 * D * B^-T * rand_vec
						D_BT_inv_rand_vec.resize(0, 0);
						for (int par_nb = 1; par_nb < num_cov_par_; ++par_nb) {
							den_mat_t minus_dB_Bi_D_BT_inv_rand_vec = -B_grad_[cluster_i][par_nb - 1] * Bi_D_BT_inv_rand_vec + D_grad_[cluster_i][par_nb - 1] * BT_inv_rand_vec;//minus_dB_Bi_D_BT_inv_rand_vec = -dBk * B^-1 * D * B^-T * rand_vec + dDk * B^-T * rand_vec
							sigma_inv_sigma_grad_rand_vec_[par_nb] = (B_[cluster_i].transpose() * (D_inv_[cluster_i] * minus_dB_Bi_D_BT_inv_rand_vec)) - (B_grad_[cluster_i][par_nb - 1]).transpose() * BT_inv_rand_vec;
						}
						Bi_D_BT_inv_rand_vec.resize(0, 0);
						BT_inv_rand_vec.resize(0, 0);
						if (include_error_var && !transf_scale) {
							//The derivative for the nugget variance is the identity matrix on the orginal scale, i.e. psi_inv_grad_psi_sigma2 = psi_inv
							sigma_inv_sigma_grad_rand_vec_[0] = B_[cluster_i].transpose() * (D_inv_[cluster_i] * (B_[cluster_i] * rand_vec_fisher_info_[cluster_i]));
						}
						//Calculate Fisher information
						sp_mat_t D_inv_B_grad_B_inv, B_grad_B_inv_D;
						if (include_error_var) {
							//First calculate terms for nugget effect / noise variance parameter
							if (transf_scale) {//Optimization is done on transformed scale (in particular, log-scale)
								//The derivative for the nugget variance on the log scale is the original covariance matrix Psi, i.e. psi_inv_grad_psi_sigma2 is the identity matrix.
								FI(0, 0) += num_data_per_cluster_[cluster_i] / 2.;
								for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
									FI(0, par_nb + 1) += (double)((D_inv_[cluster_i].diagonal().array() * D_grad_[cluster_i][par_nb].diagonal().array()).sum()) / 2.;
								}
								//Remaining covariance parameters
								for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
									for (int par_nb_cross = par_nb; par_nb_cross < num_cov_par_ - 1; ++par_nb_cross) {
										FI(par_nb + start_cov_pars, par_nb_cross + start_cov_pars) += (sigma_inv_sigma_grad_rand_vec_[par_nb + 1]).cwiseProduct(sigma_inv_sigma_grad_rand_vec_[par_nb_cross + 1]).colwise().sum().mean() / 2.;
									}
								}
							}
							else {//Original scale for asymptotic covariance matrix
								for (int par_nb = 0; par_nb < num_cov_par_; ++par_nb) {
									for (int par_nb_cross = par_nb; par_nb_cross < num_cov_par_; ++par_nb_cross) {
										FI(par_nb, par_nb_cross) += (sigma_inv_sigma_grad_rand_vec_[par_nb]).cwiseProduct(sigma_inv_sigma_grad_rand_vec_[par_nb_cross]).colwise().sum().mean() / 2.;
									}
								}
							}
						}//end include_error_var
						else {//!include_error_var
							for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
								for (int par_nb_cross = par_nb; par_nb_cross < num_cov_par_ - 1; ++par_nb_cross) {
									FI(par_nb, par_nb_cross) += (sigma_inv_sigma_grad_rand_vec_[par_nb + 1]).cwiseProduct(sigma_inv_sigma_grad_rand_vec_[par_nb_cross + 1]).colwise().sum().mean() / 2.;
								}
							}
						}
					}//end use_stochastic_trace_for_Fisher_information_Vecchia_
					else {//!use_stochastic_trace_for_Fisher_information_Vecchia_
						//Calculate auxiliary matrices for use below
						sp_mat_t Identity(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
						Identity.setIdentity();
						sp_mat_t B_inv;
						TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(B_[cluster_i], Identity, B_inv, false);//No noticeable difference in (n=500, nn=100/30) compared to using eigen_sp_Lower_sp_RHS_cs_solve()
						//eigen_sp_Lower_sp_RHS_cs_solve(B_[cluster_i], Identity, B_inv, true);
						sp_mat_t D(num_data_per_cluster_[cluster_i], num_data_per_cluster_[cluster_i]);
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
						}//end include_error_var
						//Remaining covariance parameters
						for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
							D_inv_B_grad_B_inv = D_inv_[cluster_i] * B_grad_B_inv[par_nb];
							for (int par_nb_cross = par_nb; par_nb_cross < num_cov_par_ - 1; ++par_nb_cross) {
								B_grad_B_inv_D = B_grad_B_inv[par_nb_cross] * D;
								double diag = (double)((D_inv_2.diagonal().array() * D_grad_[cluster_i][par_nb].diagonal().array() * D_grad_[cluster_i][par_nb_cross].diagonal().array()).sum());
								FI(par_nb + start_cov_pars, par_nb_cross + start_cov_pars) += ((double)(B_grad_B_inv_D.cwiseProduct(D_inv_B_grad_B_inv)).sum() + diag / 2.);
							}
						}
					}//end !use_stochastic_trace_for_Fisher_information_Vecchia_
				}//end gp_approx_ == "vecchia"
				else if (gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering") {
					// Hutchinson's Trace estimator
						// Sample vectors
					if (!saved_rand_vec_fisher_info_[cluster_i]) {
						if (!cg_generator_seeded_) {
							cg_generator_ = RNG_t(seed_rand_vec_trace_);
							cg_generator_seeded_ = true;
						}
						rand_vec_fisher_info_[cluster_i].resize(num_data_per_cluster_[cluster_i], num_rand_vec_trace_);
						GenRandVecTrace(cg_generator_, rand_vec_fisher_info_[cluster_i]);
						if (reuse_rand_vec_trace_) {//Use same random vectors for each iteration && cluster_i == end(unique_cluster) Tim
							saved_rand_vec_fisher_info_[cluster_i] = true;
						}
					}
					std::shared_ptr<T_mat> sigma_resid;
					den_mat_t sigma_inv_rand_vec_nugget;
					int deriv_par_nb = 0;
					for (int j = 0; j < num_comps_total_; ++j) {
						if (gp_approx_ == "full_scale_tapering" && matrix_inversion_method_ == "iterative") {
							re_comps_resid_[cluster_i][j]->CalcSigma();
							// Subtract predictive process covariance
							re_comps_resid_[cluster_i][j]->SubtractPredProcFromSigmaForResidInFullScale(chol_ip_cross_cov_[cluster_i], true);
							// Apply Taper
							re_comps_resid_[cluster_i][j]->ApplyTaper();
							if (gauss_likelihood_) {
								re_comps_resid_[cluster_i][j]->AddConstantToDiagonalSigma(1.);//add nugget effect variance
							}
							sigma_resid = re_comps_resid_[cluster_i][j]->GetZSigmaZt();
						}
						std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov_[cluster_i][j]->GetZSigmaZt();
						den_mat_t sigma_ip_inv_sigma_cross_cov = chol_fact_sigma_ip_[cluster_i].solve((*cross_cov).transpose());
						int num_par_comp = re_comps_ip_[cluster_i][j]->num_cov_par_;
						// Inverse of Sigma residual times cross covariance
						den_mat_t Sigma_inv_cross_cov;
						den_mat_t Sigma_inv_rand_vec;
						if (matrix_inversion_method_ == "cholesky" && gp_approx_ == "full_scale_tapering") {
							Sigma_inv_cross_cov = chol_fact_resid_[cluster_i].solve(*cross_cov);
							Sigma_inv_rand_vec = chol_fact_resid_[cluster_i].solve(rand_vec_fisher_info_[cluster_i]);
						}
						for (int jpar = 0; jpar < num_par_comp; ++jpar) {
							// Derivative of Components
							std::shared_ptr<den_mat_t> cross_cov_grad = re_comps_cross_cov_[cluster_i][j]->GetZSigmaZtGrad(jpar, transf_scale, cov_pars[0]);
							den_mat_t sigma_ip_stable_grad = *(re_comps_ip_[cluster_i][j]->GetZSigmaZtGrad(jpar, transf_scale, cov_pars[0]));
							den_mat_t sigma_ip_grad_inv_sigma_cross_cov = sigma_ip_stable_grad * sigma_ip_inv_sigma_cross_cov;
							if (gp_approx_ == "full_scale_tapering") {
								// Initialize Residual Process
								re_comps_resid_[cluster_i][j]->CalcSigma();
								std::shared_ptr<T_mat> sigma_resid_grad = re_comps_resid_[cluster_i][j]->GetZSigmaZtGrad(jpar, transf_scale, cov_pars[0]);
								// Subtract gradient of predictive process covariance
								SubtractProdFromMat<T_mat>(*sigma_resid_grad, -sigma_ip_inv_sigma_cross_cov, sigma_ip_grad_inv_sigma_cross_cov, true);
								SubtractProdFromMat<T_mat>(*sigma_resid_grad, (*cross_cov_grad).transpose(), sigma_ip_inv_sigma_cross_cov, false);
								SubtractProdFromMat<T_mat>(*sigma_resid_grad, sigma_ip_inv_sigma_cross_cov, (*cross_cov_grad).transpose(), false);
								// Apply taper
								re_comps_resid_[cluster_i][j]->ApplyTaper(*(re_comps_resid_[cluster_i][j]->dist_), *sigma_resid_grad);
								// Inverse times Gradient times Random vectors
								// Gradient times Random vectors
								den_mat_t sigma_resid_grad_rand_vec(num_data_per_cluster_[cluster_i], num_rand_vec_trace_);
								sigma_resid_grad_rand_vec.setZero();
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < num_rand_vec_trace_; ++i) {
									sigma_resid_grad_rand_vec.col(i) += (*sigma_resid_grad) * rand_vec_fisher_info_[cluster_i].col(i);
								}
								sigma_resid_grad_rand_vec += sigma_ip_inv_sigma_cross_cov.transpose() * ((*cross_cov_grad).transpose() * rand_vec_fisher_info_[cluster_i])
									+ (*cross_cov_grad) * (sigma_ip_inv_sigma_cross_cov * rand_vec_fisher_info_[cluster_i])
									- sigma_ip_inv_sigma_cross_cov.transpose() * (sigma_ip_grad_inv_sigma_cross_cov * rand_vec_fisher_info_[cluster_i]);
								// Inverse times Gradient times Random vectors
								den_mat_t sigma_inv_sigma_grad_rand_vec_interim(num_data_per_cluster_[cluster_i], num_rand_vec_trace_);
								if (matrix_inversion_method_ == "cholesky") {
									den_mat_t Sigma_inv_Grad_rand_vec = chol_fact_resid_[cluster_i].solve(sigma_resid_grad_rand_vec);
									sigma_inv_sigma_grad_rand_vec_[deriv_par_nb] = Sigma_inv_Grad_rand_vec - Sigma_inv_cross_cov * (chol_fact_sigma_woodbury_[cluster_i].solve((*cross_cov).transpose() * Sigma_inv_Grad_rand_vec));
								}
								else if (matrix_inversion_method_ == "iterative") {
									CGFSA_MULTI_RHS<T_mat>(*sigma_resid, *cross_cov, chol_fact_sigma_ip_[cluster_i], sigma_resid_grad_rand_vec, sigma_inv_sigma_grad_rand_vec_interim, NaN_found,
										num_data_per_cluster_[cluster_i], num_rand_vec_trace_, cg_max_num_it_tridiag_, cg_delta_conv_, cg_preconditioner_type_,
										chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
									sigma_inv_sigma_grad_rand_vec_[deriv_par_nb] = sigma_inv_sigma_grad_rand_vec_interim;
								}
								// Gradient times Inverse times Random vectors
								// Inverse times Random vectors
								den_mat_t sigma_inv_rand_vec(num_data_per_cluster_[cluster_i], num_rand_vec_trace_);
								if (matrix_inversion_method_ == "cholesky") {
									sigma_inv_rand_vec = Sigma_inv_rand_vec - Sigma_inv_cross_cov * (chol_fact_sigma_woodbury_[cluster_i].solve((*cross_cov).transpose() * Sigma_inv_rand_vec));
									sigma_inv_rand_vec_nugget = sigma_inv_rand_vec;
								}
								else if (matrix_inversion_method_ == "iterative") {
									CGFSA_MULTI_RHS<T_mat>(*sigma_resid, *cross_cov, chol_fact_sigma_ip_[cluster_i], rand_vec_fisher_info_[cluster_i], sigma_inv_rand_vec, NaN_found,
										num_data_per_cluster_[cluster_i], num_rand_vec_trace_, cg_max_num_it_tridiag_, cg_delta_conv_, cg_preconditioner_type_,
										chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
									sigma_inv_rand_vec_nugget = sigma_inv_rand_vec;
								}
								// Gradient times Inverse times Random vectors
								sigma_grad_sigma_inv_rand_vec_[deriv_par_nb] = (*sigma_resid_grad) * sigma_inv_rand_vec;
								sigma_grad_sigma_inv_rand_vec_[deriv_par_nb] += sigma_ip_inv_sigma_cross_cov.transpose() * ((*cross_cov_grad).transpose() * sigma_inv_rand_vec)
									+ (*cross_cov_grad) * (sigma_ip_inv_sigma_cross_cov * sigma_inv_rand_vec)
									- sigma_ip_inv_sigma_cross_cov.transpose() * (sigma_ip_grad_inv_sigma_cross_cov * sigma_inv_rand_vec);
							}
							else if (gp_approx_ == "fitc") {
								den_mat_t sigma_ip_stable_grad_nugget = *(re_comps_ip_[cluster_i][j]->GetZSigmaZtGrad(jpar, transf_scale, 1.));
								vec_t FITC_Diag_grad = vec_t::Zero(num_data_per_cluster_[cluster_i]);
								FITC_Diag_grad = FITC_Diag_grad.array() + sigma_ip_stable_grad_nugget.coeffRef(0, 0);
#pragma omp parallel for schedule(static)
								for (int ii = 0; ii < num_data_per_cluster_[cluster_i]; ++ii) {
									FITC_Diag_grad[ii] -= 2 * sigma_ip_inv_sigma_cross_cov.col(ii).dot((*cross_cov_grad).transpose().col(ii))
										- sigma_ip_inv_sigma_cross_cov.col(ii).dot(sigma_ip_grad_inv_sigma_cross_cov.col(ii));
								}
								// Inverse times Gradient times Random vectors
								// Gradient times Random vectors
								den_mat_t sigma_resid_grad_rand_vec = FITC_Diag_grad.asDiagonal() * rand_vec_fisher_info_[cluster_i];
								sigma_resid_grad_rand_vec += sigma_ip_inv_sigma_cross_cov.transpose() * ((*cross_cov_grad).transpose() * rand_vec_fisher_info_[cluster_i])
									+ (*cross_cov_grad) * (sigma_ip_inv_sigma_cross_cov * rand_vec_fisher_info_[cluster_i])
									- sigma_ip_inv_sigma_cross_cov.transpose() * (sigma_ip_grad_inv_sigma_cross_cov * rand_vec_fisher_info_[cluster_i]);
								// Inverse times Gradient times Random vectors
								den_mat_t FITC_Diag_inv_Grad_rand_vec = FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * sigma_resid_grad_rand_vec;
								den_mat_t FITC_Diag_inv_cross_cov = FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * (*cross_cov);
								sigma_inv_sigma_grad_rand_vec_[deriv_par_nb] = FITC_Diag_inv_Grad_rand_vec - FITC_Diag_inv_cross_cov * (chol_fact_sigma_woodbury_[cluster_i].solve((*cross_cov).transpose() * FITC_Diag_inv_Grad_rand_vec));

								// Gradient times Inverse times Random vectors
								// Inverse times Random vectors
								den_mat_t FITC_Diag_inv_rand_vec = FITC_Diag_[cluster_i].cwiseInverse().asDiagonal() * rand_vec_fisher_info_[cluster_i];
								den_mat_t sigma_inv_rand_vec = FITC_Diag_inv_rand_vec - FITC_Diag_inv_cross_cov * (chol_fact_sigma_woodbury_[cluster_i].solve((*cross_cov).transpose() * FITC_Diag_inv_rand_vec));
								sigma_inv_rand_vec_nugget = sigma_inv_rand_vec;
								// Gradient times Inverse times Random vectors
								sigma_grad_sigma_inv_rand_vec_[deriv_par_nb] = FITC_Diag_grad.asDiagonal() * sigma_inv_rand_vec;
								sigma_grad_sigma_inv_rand_vec_[deriv_par_nb] += sigma_ip_inv_sigma_cross_cov.transpose() * ((*cross_cov_grad).transpose() * sigma_inv_rand_vec)
									+ (*cross_cov_grad) * (sigma_ip_inv_sigma_cross_cov * sigma_inv_rand_vec)
									- sigma_ip_inv_sigma_cross_cov.transpose() * (sigma_ip_grad_inv_sigma_cross_cov * sigma_inv_rand_vec);
							}
							deriv_par_nb += 1;
						}
					}
					if (include_error_var) {
						//First calculate terms for nugget effect / noise variance parameter
						if (transf_scale) {//Optimization is done on transformed scale (error variance factored out and log-scale)
							//The derivative for the nugget variance on the transformed scale is the original covariance matrix Psi, i.e. psi_inv_grad_psi_sigma2 is the identity matrix.
							FI(0, 0) += num_data_per_cluster_[cluster_i] / 2.;
							for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
								FI(0, par_nb + 1) += (rand_vec_fisher_info_[cluster_i]).cwiseProduct(sigma_inv_sigma_grad_rand_vec_[par_nb]).colwise().sum().mean() / 2.;
							}
						}
						else {//Original scale for asymptotic covariance matrix
							//The derivative for the nugget variance is the identity matrix, i.e. psi_inv_grad_psi_sigma2 = psi_inv.
							FI(0, 0) += (sigma_inv_rand_vec_nugget).cwiseProduct(sigma_inv_rand_vec_nugget).colwise().sum().mean() / 2.;
							for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
								FI(0, par_nb + 1) += (sigma_inv_rand_vec_nugget).cwiseProduct(sigma_inv_sigma_grad_rand_vec_[par_nb]).colwise().sum().mean() / 2.;
							}
						}
					}
					//Remaining covariance parameters
					for (int par_nb = 0; par_nb < num_cov_par_ - 1; ++par_nb) {
						for (int par_nb_cross = par_nb; par_nb_cross < num_cov_par_ - 1; ++par_nb_cross) {
							FI(par_nb + start_cov_pars, par_nb_cross + start_cov_pars) += (sigma_grad_sigma_inv_rand_vec_[par_nb]).cwiseProduct(sigma_inv_sigma_grad_rand_vec_[par_nb_cross]).colwise().sum().mean() / 2.;
						}
					}
					if (!transf_scale) {
						FI /= (cov_pars[0] * cov_pars[0]);
					}
				}//end gp_approx_ == "fitc" || gp_approx_ == "full_scale_tapering"
				else {// gp_approx_ == "none"
					if (only_grouped_REs_use_woodbury_identity_) {
						//Notation used below: M = Sigma^-1 + ZtZ, Sigma = cov(b) b=latent random effects, L=chol(M) i.e. M=LLt, MInv = M^-1 = L^-TL^-1
						if (!use_saved_psi_inv) {
							LInvZtZj_[cluster_i] = std::vector<T_mat>(num_comps_total_);
							if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ == ZtZj_ and L_inv are diagonal  
								LInvZtZj_[cluster_i][0] = ZtZ_[cluster_i];
								LInvZtZj_[cluster_i][0].diagonal().array() /= sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array();
							}
							else {
								for (int j = 0; j < num_comps_total_; ++j) {
									if (CholeskyHasPermutation<T_chol>(chol_facts_[cluster_i])) {
										TriangularSolve<T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), P_ZtZj_[cluster_i][j], LInvZtZj_[cluster_i][j], false);
									}
									else {
										TriangularSolve<T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i].CholFactMatrix(), ZtZj_[cluster_i][j], LInvZtZj_[cluster_i][j], false);
									}
								}
							}
						}
						if (include_error_var) {
							if (transf_scale) {//Optimization is done on transformed scale (error variance factored out and log-scale)
								//The derivative for the nugget variance on the transformed scale is the original covariance matrix Psi, i.e. psi_inv_grad_psi_sigma2 is the identity matrix.
								FI(0, 0) += num_data_per_cluster_[cluster_i] / 2.;
								for (int j = 0; j < num_comps_total_; ++j) {
									double trace_PsiInvGradPsi = Zj_square_sum_[cluster_i][j] - LInvZtZj_[cluster_i][j].squaredNorm();
									FI(0, j + 1) += trace_PsiInvGradPsi * cov_pars[j + 1] / 2.;
								}
							}//end transf_scale
							else {//not transf_scale
								T_mat MInv_ZtZ;//=(Sigma_inv + ZtZ)^-1 * ZtZ
								if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ == ZtZj_ and L_inv are diagonal 
									MInv_ZtZ = T_mat(ZtZ_[cluster_i].rows(), ZtZ_[cluster_i].cols());
									MInv_ZtZ.setIdentity();//initialize
									MInv_ZtZ.diagonal().array() = ZtZ_[cluster_i].diagonal().array() / (sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().square());
								}
								else {
									SolveGivenCholesky<T_chol, T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i], ZtZ_[cluster_i], MInv_ZtZ);

								}
								T_mat MInv_ZtZ_t = MInv_ZtZ.transpose();
								FI(0, 0) += (num_data_per_cluster_[cluster_i] - 2. * MInv_ZtZ.diagonal().sum() + (double)(MInv_ZtZ.cwiseProduct(MInv_ZtZ_t)).sum()) / (cov_pars[0] * cov_pars[0] * 2.);
								for (int j = 0; j < num_comps_total_; ++j) {
									T_mat ZjZ_MInv_ZtZ_t = MInv_ZtZ_t * ZtZj_[cluster_i][j];
									double trace_PsiInvGradPsi;
									if (num_comps_total_ > 1) {
										T_mat MInv_ZtZj;
										SolveGivenCholesky<T_chol, T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i], ZtZj_[cluster_i][j], MInv_ZtZj);
										trace_PsiInvGradPsi = Zj_square_sum_[cluster_i][j] - 2. * (double)(LInvZtZj_[cluster_i][j].squaredNorm()) +
											(double)(ZjZ_MInv_ZtZ_t.cwiseProduct(MInv_ZtZj)).sum();
									}
									else {
										trace_PsiInvGradPsi = Zj_square_sum_[cluster_i][j] - 2. * (double)(LInvZtZj_[cluster_i][j].squaredNorm()) +
											(double)(ZjZ_MInv_ZtZ_t.cwiseProduct(MInv_ZtZ)).sum();
									}
									FI(0, j + 1) += trace_PsiInvGradPsi / (cov_pars[0] * cov_pars[0] * 2.);
								}
							}//end not transf_scale
						}//end include_error_var
						//Remaining covariance parameters
						for (int j = 0; j < num_comps_total_; ++j) {
							sp_mat_t* Z_j = re_comps_[cluster_i][j]->GetZ();
							for (int k = j; k < num_comps_total_; ++k) {
								sp_mat_t* Z_k = re_comps_[cluster_i][k]->GetZ();
								T_mat Zjt_Zk = T_mat((*Z_j).transpose() * (*Z_k));
								T_mat LInvZtZj_t_LInvZtZk = LInvZtZj_[cluster_i][j].transpose() * LInvZtZj_[cluster_i][k];
								double FI_jk = Zjt_Zk.squaredNorm() + LInvZtZj_t_LInvZtZk.squaredNorm() - 2. * (double)(Zjt_Zk.cwiseProduct(LInvZtZj_t_LInvZtZk)).sum();
								if (transf_scale) {
									FI_jk *= cov_pars[j + 1] * cov_pars[k + 1];
								}
								else {
									FI_jk /= cov_pars[0] * cov_pars[0];
								}
								FI(j + start_cov_pars, k + start_cov_pars) += FI_jk / 2.;
							}
						}
					}//end only_grouped_REs_use_woodbury_identity_
					else {//gp_approx == "none" and not only_grouped_REs_use_woodbury_identity_
						T_mat psi_inv;
						if (use_saved_psi_inv) {
							psi_inv = psi_inv_[cluster_i];
						}
						else {
							CalcPsiInv(psi_inv, cluster_i, false);
						}
						if (!transf_scale) {
							psi_inv /= cov_pars[0];//psi_inv has been calculated with a transformed parametrization, so we need to divide everything by cov_pars[0] to obtain the covariance matrix
						}
						//Calculate Psi^-1 * derivative(Psi)
						std::vector<T_mat> psi_inv_deriv_psi(num_cov_par_ - 1);
						int deriv_par_nb = 0;
						for (int j = 0; j < num_comps_total_; ++j) {//there is currently no possibility to loop over the parameters directly
							for (int jpar = 0; jpar < re_comps_[cluster_i][j]->num_cov_par_; ++jpar) {
								psi_inv_deriv_psi[deriv_par_nb] = psi_inv * *(re_comps_[cluster_i][j]->GetZSigmaZtGrad(jpar, transf_scale, cov_pars[0]));
								deriv_par_nb++;
							}
						}
						//Calculate Fisher information
						if (include_error_var) {
							//First calculate terms for nugget effect / noise variance parameter
							if (transf_scale) {//Optimization is done on transformed scale (error variance factored out and log-scale)
								//The derivative for the nugget variance on the transformed scale is the original covariance matrix Psi, i.e. psi_inv_grad_psi_sigma2 is the identity matrix.
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
							T_mat psi_inv_grad_psi_par_nb_T = psi_inv_deriv_psi[par_nb].transpose();
							FI(par_nb + start_cov_pars, par_nb + start_cov_pars) += ((double)(psi_inv_grad_psi_par_nb_T.cwiseProduct(psi_inv_deriv_psi[par_nb])).sum()) / 2.;
							for (int par_nb_cross = par_nb + 1; par_nb_cross < num_cov_par_ - 1; ++par_nb_cross) {
								FI(par_nb + start_cov_pars, par_nb_cross + start_cov_pars) += ((double)(psi_inv_grad_psi_par_nb_T.cwiseProduct(psi_inv_deriv_psi[par_nb_cross])).sum()) / 2.;
							}
							psi_inv_deriv_psi[par_nb].resize(0, 0);//not needed anymore
							psi_inv_grad_psi_par_nb_T.resize(0, 0);
						}
					}//end gp_approx == "none" and not only_grouped_REs_use_woodbury_identity_
				}//end gp_approx == "none"
			}//end loop over clusters
			FI.triangularView<Eigen::StrictlyLower>() = FI.triangularView<Eigen::StrictlyUpper>().transpose();
			//for (int i = 0; i < std::min((int)FI.rows(),4); ++i) {//For debugging only
			//    for (int j = i; j < std::min((int)FI.cols(),4); ++j) {
			//	    Log::REInfo("FI(%d,%d) %g", i, j, FI(i, j));
			//    }
			//}
		}//end CalcFisherInformation

		/*!
		* \brief Calculate the standard deviations for the MLE of the covariance parameters as the diagonal of the inverse Fisher information (on the orignal scale and not the transformed scale used in the optimization, for Gaussian data only)
		* \param cov_pars MLE of covariance parameters
		* \param[out] std_dev Standard deviations
		*/
		void CalcStdDevCovPar(const vec_t& cov_pars,
			vec_t& std_dev) {
			SetCovParsComps(cov_pars);
			CalcCovFactor(true, false, cov_pars[0], true);
			den_mat_t FI;
			CalcFisherInformation(cov_pars, FI, false, true, false);
			std_dev = FI.inverse().diagonal().array().sqrt().matrix();
		}

		/*!
		* \brief Calculate standard deviations for the MLE of the regression coefficients as the diagonal of the inverse Fisher information (for Gaussian data only)
		* \param cov_pars MLE of covariance parameters
		* \param X Covariate data for linear fixed-effect
		* \param[out] std_dev Standard deviations
		*/
		void CalcStdDevCoef(const vec_t& cov_pars,
			const den_mat_t& X,
			vec_t& std_dev) {
			if ((int)std_dev.size() >= num_data_) {
				Log::REWarning("Sample size too small to calculate standard deviations for coefficients");
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
		* \brief Calculate standard deviations for the MLE of the regression coefficients as the square root of diagonal of a numerically approximated inverse Hessian
		* \param num_covariates Number of covariates / coefficients
		* \param beta Regression coefficients
		* \param cov_pars Covariance parameters
		* \param fixed_effects Externally provided fixed effects component of location parameter
		* \param[out] std_dev_beta Standard deviations
		*/
		void CalcStdDevCoefNonGaussian(int num_covariates,
			const vec_t& beta,
			const vec_t& cov_pars,
			const double* fixed_effects,
			vec_t& std_dev_beta) {
			den_mat_t H(num_covariates, num_covariates);// Aproximate Hessian calculated as the Jacobian of the gradient
			const double mach_eps = std::numeric_limits<double>::epsilon();
			vec_t delta_step = beta * std::pow(mach_eps, 1.0 / 3.0);// based on https://math.stackexchange.com/questions/1039428/finite-difference-method
			vec_t fixed_effects_vec, beta_change1, beta_change2, grad_beta_change1, grad_beta_change2;
			for (int i = 0; i < num_covariates; ++i) {
				// Beta plus / minus delta
				beta_change1 = beta;
				beta_change2 = beta;
				beta_change1[i] += delta_step[i];
				beta_change2[i] -= delta_step[i];
				// Gradient vector at beta plus / minus delta
				UpdateFixedEffects(beta_change1, fixed_effects, fixed_effects_vec);
				CalcCovFactorOrModeAndNegLL(cov_pars, fixed_effects_vec.data());
				CalcGradLinCoef(1., grad_beta_change1, fixed_effects_vec.data());
				UpdateFixedEffects(beta_change2, fixed_effects, fixed_effects_vec);
				CalcCovFactorOrModeAndNegLL(cov_pars, fixed_effects_vec.data());
				CalcGradLinCoef(1., grad_beta_change2, fixed_effects_vec.data());
				// Approximate gradient of gradient
				H.row(i) = (grad_beta_change1 - grad_beta_change2) / (2. * delta_step[i]);
			}
			den_mat_t Hsym = (H + H.transpose()) / 2.;
			// (Very) approximate standard deviations as square root of diagonal of inverse Hessian
			std_dev_beta = Hsym.inverse().diagonal().array().sqrt().matrix();
		}//end CalcStdDevCoefNonGaussian

		/*!
		* \brief Calculate numerically approximated Hessian for the covariance and auxiliary parameters
		* \param cov_aux_pars Covariance and auxiliary (if present) parameters
		* \param include_error_var If true, the error variance parameter (=nugget effect) is also included, otherwise not
		* \param fixed_effects Externally provided fixed effects component of location parameter
		* \param[out] Hessian matrix obtained as numerical Jacobian of the gradient
		*/
		void CalcHessianCovParAuxPars(const vec_t& cov_aux_pars,
			bool include_error_var,
			const double* fixed_effects,
			den_mat_t& Hessian) {
			if (estimate_aux_pars_) {
				CHECK(cov_aux_pars.size() == num_cov_par_ + NumAuxPars());
			}
			else {
				CHECK(cov_aux_pars.size() == num_cov_par_);
			}
			int length_grad = num_cov_par_;
			int offset = 0;
			if (gauss_likelihood_ && !include_error_var) {
				length_grad -= 1;
				offset = 1;
			}
			if (estimate_aux_pars_) {
				length_grad += NumAuxPars();
			}
			den_mat_t H(length_grad, length_grad);// Aproximate Hessian calculated as the Jacobian of the gradient
			vec_t log_pars = cov_aux_pars.array().log().matrix();
			const double h_eps = std::pow(std::numeric_limits<double>::epsilon(), 1.0 / 3.0);
			vec_t delta_step = log_pars * h_eps;// based on https://math.stackexchange.com/questions/1039428/finite-difference-method
			for (int i = 0; i < (int)delta_step.size(); ++i) {//avoid problems when log_pars is close to 0
				if (delta_step[i] < h_eps) {
					delta_step[i] = h_eps;
				}
			}
			vec_t pars_change1, pars_change2, grad_change1, grad_change2;
			for (int i = 0; i < length_grad; ++i) {
				// cov_aux_pars plus / minus delta
				pars_change1 = cov_aux_pars;
				pars_change2 = cov_aux_pars;
				pars_change1[i + offset] *= std::exp(delta_step[i + offset]);//gradient is taken on log-scale
				pars_change2[i + offset] *= std::exp(-delta_step[i + offset]);
				if (estimate_aux_pars_) {
					SetAuxPars(pars_change1.data() + num_cov_par_);
				}
				CalcCovFactorOrModeAndNegLL(pars_change1.segment(0, num_cov_par_), fixed_effects);
				CalcGradCovParAuxPars(pars_change1.segment(0, num_cov_par_), grad_change1, include_error_var, false, fixed_effects);
				if (estimate_aux_pars_) {
					SetAuxPars(pars_change2.data() + num_cov_par_);
				}
				CalcCovFactorOrModeAndNegLL(pars_change2.segment(0, num_cov_par_), fixed_effects);
				CalcGradCovParAuxPars(pars_change2.segment(0, num_cov_par_), grad_change2, include_error_var, false, fixed_effects);
				// Approximate gradient of gradient
				H.row(i) = (grad_change1 - grad_change2) / (2. * delta_step[i + offset]);
			}
			Hessian = (H + H.transpose()) / 2.;
		}//end CalcHessianCovParAuxPars

		/*!
		 * \brief Prepare for prediction: set respone variable data, factorize covariance matrix and calculate Psi^{-1}y_obs or calculate Laplace approximation (if required)
		* \param cov_pars Covariance parameters of components
		* \param coef Coefficients for linear covariates
		* \param y_obs Response variable for observed data
		* \param calc_cov_factor If true, the covariance matrix of the observed data is factorized otherwise a previously done factorization is used
		* \param fixed_effects Fixed effects component of location parameter for observed data (only used for non-Gaussian likelihoods)
		* \param predict_training_data_random_effects If true, the goal is to predict training data random effects
		 */
		void SetYCalcCovCalcYAuxForPred(const vec_t& cov_pars,
			const vec_t& coef,
			const double* y_obs,
			bool calc_cov_factor,
			const double* fixed_effects,
			bool predict_training_data_random_effects) {
			const double* fixed_effects_ptr = fixed_effects;
			vec_t fixed_effects_vec;
			// Set response data and fixed effects
			if (gauss_likelihood_) {
				if (has_covariates_ || fixed_effects != nullptr) {
					vec_t resid;
					if (y_obs != nullptr) {
						resid = Eigen::Map<const vec_t>(y_obs, num_data_);
					}
					else {
						resid = y_vec_;
					}
					if (has_covariates_) {
						resid -= X_ * coef;
					}
					//add external fixed effects to linear predictor
					if (fixed_effects != nullptr) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_; ++i) {
							resid[i] -= fixed_effects[i];
						}
					}
					SetY(resid.data());
				}//end if has_covariates_
				else {//no covariates
					if (y_obs != nullptr) {
						SetY(y_obs);
					}
				}//end no covariates
			}//end if gauss_likelihood_
			else {//if not gauss_likelihood_
				if (has_covariates_) {
					fixed_effects_vec = X_ * coef;
					//add external fixed effects to linear predictor
					if (fixed_effects != nullptr) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_; ++i) {
							fixed_effects_vec[i] += fixed_effects[i];
						}
					}
					fixed_effects_ptr = fixed_effects_vec.data();
				}
				if (y_obs != nullptr) {
					SetY(y_obs);
				}
			}//end if not gauss_likelihood_
			//TODO (low prio): the factorization needs to be done only for the GP realizations / clusters for which predictions are made (currently it is done for all)
			SetCovParsComps(cov_pars);
			if (!(gp_approx_ == "vecchia" && gauss_likelihood_) || predict_training_data_random_effects) {
				// no need to call CalcCovFactor here for the Vecchia approximation for Gaussian data, this is done in the prediction steps below, 
				//	but when predicting training data random effects, this is required
				if (calc_cov_factor) {
					if (gauss_likelihood_) {
						CalcCovFactor(false, true, 1., false);// Create covariance matrix and factorize it
					}
					else {//not gauss_likelihood_
						//We reset the initial modes to 0. This is done to avoid that different calls to the prediction function lead to (very small) differences
						//	as the mode is calculated from different starting values.
						//	If one is willing to accept these (very) small differences, one could disable this with the advantage of having faster predictions
						//	as the mode does not need to be found anew.
						for (const auto& cluster_i : unique_clusters_) {
							likelihood_[cluster_i]->InitializeModeAvec();
						}
						if (gp_approx_ == "vecchia") {
							CalcCovFactor(false, true, 1., false);
						}
						else {
							CalcSigmaComps();
							CalcCovMatrixNonGauss();
						}
						CalcModePostRandEffCalcMLL(fixed_effects_ptr, false);
					}//end not gauss_likelihood_
				}//end if calc_cov_factor
				if (gauss_likelihood_) {
					CalcYAux(1.);//note: in some cases a call to CalcYAux() could be avoided (e.g. no covariates and not GPBoost algorithm)...
				}
			}//end not (gp_approx_ == "vecchia" && gauss_likelihood_)
		}// end SetYCalcCovCalcYAuxForPred

		/*!
		 * \brief Calculate predictions (conditional mean and covariance matrix) for one cluster
		 * \param cluster_i Cluster index for which prediction are made
		 * \param num_data_pred Total number of prediction locations (over all clusters)
		 * \param num_data_per_cluster_pred Keys: Labels of independent realizations of REs/GPs, values: number of prediction locations per independent realization
		 * \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
		 * \param re_group_levels_pred Group levels for the grouped random effects (re_group_levels_pred[j] contains the levels for RE number j)
		 * \param re_group_rand_coef_data_pred Random coefficient data for grouped REs
		 * \param gp_coords_mat_pred Coordinates for prediction locations
		 * \param gp_rand_coef_data_pred Random coefficient data for GPs
		 * \param predict_cov_mat If true, the predictive/conditional covariance matrix is calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
		 * \param predict_var If true, the predictive/conditional variances are calculated (default=false) (predict_var and predict_cov_mat cannot be both true)
		 * \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects
		 * \param[out] mean_pred_id Predictive mean
		 * \param[out] cov_mat_pred_id Predictive covariance matrix
		 * \param[out] var_pred_id Predictive variances
		 */
		void CalcPred(data_size_t cluster_i,
			int num_data_pred,
			std::map<data_size_t, int>& num_data_per_cluster_pred,
			std::map<data_size_t, std::vector<int>>& data_indices_per_cluster_pred,
			const std::vector<std::vector<re_group_t>>& re_group_levels_pred,
			const double* re_group_rand_coef_data_pred,
			const den_mat_t& gp_coords_mat_pred,
			const double* gp_rand_coef_data_pred,
			bool predict_cov_mat,
			bool predict_var,
			bool predict_response,
			vec_t& mean_pred_id,
			T_mat& cov_mat_pred_id,
			vec_t& var_pred_id) {
			int num_REs_obs, num_REs_pred;
			if (only_one_grouped_RE_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {
				num_REs_pred = (int)re_group_levels_pred[0].size();
				num_REs_obs = re_comps_[cluster_i][0]->GetNumUniqueREs();
			}
			else if (only_one_GP_calculations_on_RE_scale_) {
				num_REs_pred = (int)gp_coords_mat_pred.rows();
				num_REs_obs = re_comps_[cluster_i][0]->GetNumUniqueREs();
			}
			else {
				num_REs_pred = num_data_per_cluster_pred[cluster_i];
				num_REs_obs = num_data_per_cluster_[cluster_i];
			}
			if (predict_var) {
				if (gauss_likelihood_ && predict_response) {
					var_pred_id = vec_t::Ones(num_REs_pred);//nugget effect
				}
				else {
					var_pred_id = vec_t::Zero(num_REs_pred);
				}
			}
			if (predict_cov_mat) {
				cov_mat_pred_id = T_mat(num_REs_pred, num_REs_pred);
				if (gauss_likelihood_ && predict_response) {
					cov_mat_pred_id.setIdentity();//nugget effect
				}
				else {
					cov_mat_pred_id.setZero();
				}
			}
			T_mat cross_cov;//Cross-covariance between prediction and observation points
			sp_mat_t Ztilde;//Matrix which relates existing random effects to prediction samples (used only if only_grouped_REs_use_woodbury_identity_ and not only_one_grouped_RE_calculations_on_RE_scale_)
			sp_mat_t Sigma;//Covariance matrix of random effects (used only if only_grouped_REs_use_woodbury_identity_ and not only_one_grouped_RE_calculations_on_RE_scale_)
			//Calculate (cross-)covariance matrix
			int cn = 0;//component number counter
			bool dont_add_but_overwrite = true;
			if (only_one_grouped_RE_calculations_on_RE_scale_ || only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {
				std::shared_ptr<RECompGroup<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGroup<T_mat>>(re_comps_[cluster_i][0]);
				re_comp->AddPredCovMatrices(re_group_levels_pred[0], cross_cov, cov_mat_pred_id,
					true, predict_cov_mat, true, true, nullptr);
				if (predict_var) {
					re_comp->AddPredUncondVar(var_pred_id.data(), num_REs_pred, nullptr);
				}
			}
			else if (only_grouped_REs_use_woodbury_identity_) {
				Ztilde = sp_mat_t(num_data_per_cluster_pred[cluster_i], cum_num_rand_eff_[cluster_i][num_re_group_total_]);
				bool has_ztilde = false;
				std::vector<Triplet_t> triplets(num_data_per_cluster_pred[cluster_i] * num_re_group_total_);
				for (int j = 0; j < num_group_variables_; ++j) {
					if (!drop_intercept_group_rand_effect_[j]) {
						std::shared_ptr<RECompGroup<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGroup<T_mat>>(re_comps_[cluster_i][cn]);
						std::vector<re_group_t> group_data;
						for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
							group_data.push_back(re_group_levels_pred[j][id]);
						}
						re_comp->CalcInsertZtilde(group_data, nullptr, cum_num_rand_eff_[cluster_i][cn], cn, triplets, has_ztilde);
						if (predict_cov_mat) {
							re_comp->AddPredCovMatrices(group_data, cross_cov, cov_mat_pred_id,
								false, true, dont_add_but_overwrite, false, nullptr);
							dont_add_but_overwrite = false;
						}
						if (predict_var) {
							re_comp->AddPredUncondVar(var_pred_id.data(), num_REs_pred, nullptr);
						}
						cn += 1;
					}
				}
				if (num_re_group_rand_coef_ > 0) {//Random coefficient grouped random effects
					for (int j = 0; j < num_re_group_rand_coef_; ++j) {
						std::shared_ptr<RECompGroup<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGroup<T_mat>>(re_comps_[cluster_i][cn]);
						std::vector<re_group_t> group_data;
						std::vector<double> rand_coef_data;
						for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
							rand_coef_data.push_back(re_group_rand_coef_data_pred[j * num_data_pred + id]);
							group_data.push_back(re_group_levels_pred[ind_effect_group_rand_coef_[j] - 1][id]);//subtract 1 since counting starts at one for this index
						}
						re_comp->CalcInsertZtilde(group_data, rand_coef_data.data(), cum_num_rand_eff_[cluster_i][cn], cn, triplets, has_ztilde);
						if (predict_cov_mat) {
							re_comp->AddPredCovMatrices(group_data, cross_cov, cov_mat_pred_id,
								false, true, dont_add_but_overwrite, false, rand_coef_data.data());
							dont_add_but_overwrite = false;
						}
						if (predict_var) {
							re_comp->AddPredUncondVar(var_pred_id.data(), num_REs_pred, rand_coef_data.data());
						}
						cn += 1;
					}
				}
				if (has_ztilde) {
					Ztilde.setFromTriplets(triplets.begin(), triplets.end());
				}
				CalcSigmaIGroupedREsOnly(Sigma, cluster_i, false);
			}//end only_grouped_REs_use_woodbury_identity_
			else {
				if (num_re_group_ > 0) {//Grouped random effects
					for (int j = 0; j < num_group_variables_; ++j) {
						if (!drop_intercept_group_rand_effect_[j]) {
							std::shared_ptr<RECompGroup<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGroup<T_mat>>(re_comps_[cluster_i][cn]);
							std::vector<re_group_t> group_data;
							for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
								group_data.push_back(re_group_levels_pred[j][id]);
							}
							re_comp->AddPredCovMatrices(group_data, cross_cov, cov_mat_pred_id,
								true, predict_cov_mat, dont_add_but_overwrite, false, nullptr);
							dont_add_but_overwrite = false;
							if (predict_var) {
								re_comp->AddPredUncondVar(var_pred_id.data(), num_REs_pred, nullptr);
							}
							cn += 1;
						}
					}
				}//end grouped random effects
				if (num_re_group_rand_coef_ > 0) { //Random coefficient grouped random effects
					for (int j = 0; j < num_re_group_rand_coef_; ++j) {
						std::shared_ptr<RECompGroup<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGroup<T_mat>>(re_comps_[cluster_i][cn]);
						std::vector<re_group_t> group_data;
						std::vector<double> rand_coef_data;
						for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
							rand_coef_data.push_back(re_group_rand_coef_data_pred[j * num_data_pred + id]);
							group_data.push_back(re_group_levels_pred[ind_effect_group_rand_coef_[j] - 1][id]);//subtract 1 since counting starts at one for this index
						}
						re_comp->AddPredCovMatrices(group_data, cross_cov, cov_mat_pred_id,
							true, predict_cov_mat, false, false, rand_coef_data.data());
						if (predict_var) {
							re_comp->AddPredUncondVar(var_pred_id.data(), num_REs_pred, rand_coef_data.data());
						}
						cn += 1;
					}
				}//end random coefficient grouped random effects
				//Gaussian process
				if (num_gp_ > 0) {
					std::shared_ptr<RECompGP<T_mat>> re_comp_base = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_[cluster_i][cn]);
					T_mat cross_dist; // unused dummy variable
					re_comp_base->AddPredCovMatrices(re_comp_base->coords_, gp_coords_mat_pred, cross_cov,
						cov_mat_pred_id, true, predict_cov_mat, dont_add_but_overwrite, nullptr,
						false, cross_dist);
					dont_add_but_overwrite = false;
					if (predict_var) {
						re_comp_base->AddPredUncondVar(var_pred_id.data(), num_REs_pred, nullptr);
					}
					cn += 1;
					if (num_gp_rand_coef_ > 0) {
						std::shared_ptr<RECompGP<T_mat>> re_comp;
						//Random coefficient Gaussian processes
						for (int j = 0; j < num_gp_rand_coef_; ++j) {
							re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_[cluster_i][cn]);
							std::vector<double> rand_coef_data;
							for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {
								rand_coef_data.push_back(gp_rand_coef_data_pred[j * num_data_pred + id]);
							}
							re_comp->AddPredCovMatrices(re_comp_base->coords_, gp_coords_mat_pred, cross_cov,
								cov_mat_pred_id, true, predict_cov_mat, false, rand_coef_data.data(),
								false, cross_dist);
							if (predict_var) {
								re_comp->AddPredUncondVar(var_pred_id.data(), num_REs_pred, rand_coef_data.data());
							}
							cn += 1;
						}
					}
				}// end Gaussian process
			}//end calculate cross-covariances

			// Calculate predictive means and covariances
			if (gauss_likelihood_) {//Gaussian data
				if (only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {
					vec_t Zt_y_aux;
					CalcZtVGivenIndices(num_data_per_cluster_[cluster_i], num_REs_obs,
						re_comps_[cluster_i][cn]->random_effects_indices_of_data_.data(), y_aux_[cluster_i], Zt_y_aux, true);
					mean_pred_id = cross_cov * Zt_y_aux;
				}//end only_one_grouped_RE_calculations_on_RE_scale_for_prediction_
				else if (only_grouped_REs_use_woodbury_identity_) {
					vec_t v_aux = Zt_[cluster_i] * y_aux_[cluster_i];
					vec_t v_aux2 = Sigma * v_aux;
					mean_pred_id = Ztilde * v_aux2;
				}//end only_grouped_REs_use_woodbury_identity_
				else {
					mean_pred_id = cross_cov * y_aux_[cluster_i];
				}
				if ((predict_cov_mat || predict_var) && only_one_grouped_RE_calculations_on_RE_scale_for_prediction_) {
					sp_mat_t* Z = re_comps_[cluster_i][0]->GetZ();
					T_mat cross_cov_temp = cross_cov;
					cross_cov = cross_cov_temp * (*Z).transpose();
					cross_cov_temp.resize(0, 0);
					//TODO (low-prio): things could be done more efficiently (using random_effects_indices_of_data_) as ZtZ_ is diagonal
				}
				if (predict_cov_mat) {
					if (only_grouped_REs_use_woodbury_identity_) {
						if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
							T_mat ZtM_aux = (T_mat)(Zt_[cluster_i] * cross_cov.transpose());
							ZtM_aux = sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().inverse().matrix().asDiagonal() * ZtM_aux;
							cov_mat_pred_id -= (T_mat)(cross_cov * cross_cov.transpose());
							cov_mat_pred_id += (T_mat)(ZtM_aux.transpose() * ZtM_aux);
						}
						else {
							T_mat M_aux;
							TriangularSolveGivenCholesky<T_chol, T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i], ZtZ_[cluster_i], M_aux, false);
							sp_mat_t ZtildeSigma = Ztilde * Sigma;
							T_mat M_aux2 = M_aux * ZtildeSigma.transpose();
							M_aux.resize(0, 0);
							cov_mat_pred_id -= (T_mat)(ZtildeSigma * ZtZ_[cluster_i] * ZtildeSigma.transpose());
							cov_mat_pred_id += (T_mat)(M_aux2.transpose() * M_aux2);
						}
					}
					else {
						T_mat M_aux;
						TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_facts_[cluster_i], cross_cov.transpose(), M_aux, false);
						cov_mat_pred_id -= (T_mat)(M_aux.transpose() * M_aux);
					}
				}//end predict_cov_mat
				if (predict_var) {
					if (only_grouped_REs_use_woodbury_identity_) {
						if (num_re_group_total_ == 1 && num_comps_total_ == 1) {//only one random effect -> ZtZ_ is diagonal
							T_mat ZtM_aux = T_mat(Zt_[cluster_i] * cross_cov.transpose());
							T_mat M_aux2 = sqrt_diag_SigmaI_plus_ZtZ_[cluster_i].array().inverse().matrix().asDiagonal() * ZtM_aux;
							M_aux2 = M_aux2.cwiseProduct(M_aux2);
							cross_cov = cross_cov.cwiseProduct(cross_cov);
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_REs_pred; ++i) {
								var_pred_id[i] -= cross_cov.row(i).sum() - M_aux2.col(i).sum();
							}
						}
						else {//more than one grouped RE component
							T_mat M_aux;
							TriangularSolveGivenCholesky<T_chol, T_mat, sp_mat_t, T_mat>(chol_facts_[cluster_i], ZtZ_[cluster_i], M_aux, false);
							sp_mat_t ZtildeSigma = Ztilde * Sigma;
							T_mat M_aux2 = M_aux * ZtildeSigma.transpose();
							M_aux.resize(0, 0);
							sp_mat_t SigmaZtilde_ZtZ = ZtildeSigma * ZtZ_[cluster_i];
							sp_mat_t M_aux3 = ZtildeSigma.cwiseProduct(SigmaZtilde_ZtZ);
							M_aux2 = M_aux2.cwiseProduct(M_aux2);
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_REs_pred; ++i) {
								var_pred_id[i] -= M_aux3.row(i).sum() - M_aux2.col(i).sum();
							}
						}
					}//end only_grouped_REs_use_woodbury_identity_
					else {//not only_grouped_REs_use_woodbury_identity_
						T_mat M_aux2;
						TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_facts_[cluster_i], cross_cov.transpose(), M_aux2, false);
						M_aux2 = M_aux2.cwiseProduct(M_aux2);
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_REs_pred; ++i) {
							var_pred_id[i] -= M_aux2.col(i).sum();
						}
					}//end not only_grouped_REs_use_woodbury_identity_
				}//end predict_var
			}//end gauss_likelihood_
			if (!gauss_likelihood_) {//not gauss_likelihood_
				const double* fixed_effects_cluster_i_ptr = nullptr;
				// Note that fixed_effects_cluster_i_ptr is not used since calc_mode == false
				// The mode has been calculated already before in the Predict() function above
				if (only_grouped_REs_use_woodbury_identity_ && !only_one_grouped_RE_calculations_on_RE_scale_) {
					likelihood_[cluster_i]->PredictLaplaceApproxGroupedRE(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						num_data_per_cluster_[cluster_i],
						SigmaI_[cluster_i],
						Zt_[cluster_i],
						Ztilde,
						Sigma,
						mean_pred_id,
						cov_mat_pred_id,
						var_pred_id,
						predict_cov_mat,
						predict_var,
						false);
				}
				else if (only_one_grouped_RE_calculations_on_RE_scale_) {
					likelihood_[cluster_i]->PredictLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						num_data_per_cluster_[cluster_i],
						re_comps_[cluster_i][0]->cov_pars_[0],
						re_comps_[cluster_i][0]->random_effects_indices_of_data_.data(),
						cross_cov,
						mean_pred_id,
						cov_mat_pred_id,
						var_pred_id,
						predict_cov_mat,
						predict_var,
						false);
				}
				else {
					likelihood_[cluster_i]->PredictLaplaceApproxStable(y_[cluster_i].data(),
						y_int_[cluster_i].data(),
						fixed_effects_cluster_i_ptr,
						ZSigmaZt_[cluster_i],
						cross_cov,
						mean_pred_id,
						cov_mat_pred_id,
						var_pred_id,
						predict_cov_mat,
						predict_var,
						false);
				}
			}//end not gauss_likelihood_
		}//end CalcPred

		void SetVecchiaPredType(const char* vecchia_pred_type) {
			vecchia_pred_type_ = std::string(vecchia_pred_type);
			if (gauss_likelihood_) {
				if (SUPPORTED_VECCHIA_PRED_TYPES_GAUSS_.find(vecchia_pred_type_) == SUPPORTED_VECCHIA_PRED_TYPES_GAUSS_.end()) {
					Log::REFatal("Prediction type '%s' is not supported for the Veccia approximation ", vecchia_pred_type_.c_str());
				}
			}
			else {
				if (SUPPORTED_VECCHIA_PRED_TYPES_NONGAUSS_.find(vecchia_pred_type_) == SUPPORTED_VECCHIA_PRED_TYPES_NONGAUSS_.end()) {
					Log::REFatal("Prediction type '%s' is not supported for the Veccia approximation for non-Gaussian likelihoods ", vecchia_pred_type_.c_str());
				}
				if (vecchia_pred_type_ == "order_obs_first_cond_obs_only") {
					vecchia_pred_type_ = "latent_order_obs_first_cond_obs_only";
				}
				if (vecchia_pred_type_ == "order_obs_first_cond_all") {
					vecchia_pred_type_ = "latent_order_obs_first_cond_all";
				}
			}
			vecchia_pred_type_has_been_set_ = true;
		}

		/*!
		* \brief Calculate predictions (conditional mean and covariance matrix) using the PP/FSA approximation
		* \param cluster_i Cluster index for which prediction are made
		* \param num_data_per_cluster_pred Keys: Labels of independent realizations of REs/GPs, values: number of prediction locations per independent realization
		* \param num_data_per_cluster Keys: Labels of independent realizations of REs/GPs, values: number of observed locations per independent realization
		* \param gp_coords_mat_pred Coordinates for prediction locations
		* \param calc_pred_cov If true, the covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param[out] pred_mean Predictive mean (only for Gaussian likelihoods)
		* \param[out] pred_cov Predictive covariance matrix (only for Gaussian likelihoods)
		* \param[out] pred_var Predictive variances (only for Gaussian likelihoods)
		* \param nsim_var_pred Number of random vectors
		* \param cg_delta_conv_pred Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for prediction
		*/
		void CalcPredPPFSA(data_size_t cluster_i,
			std::map<data_size_t, int>& num_data_per_cluster_pred,
			std::map<data_size_t, int>& num_data_per_cluster,
			const den_mat_t& gp_coords_mat_pred,
			bool calc_pred_cov,
			bool calc_pred_var,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			int nsim_var_pred,
			const double cg_delta_conv_pred) {

			int num_data_cli = num_data_per_cluster[cluster_i];
			int num_data_pred_cli = num_data_per_cluster_pred[cluster_i];

			// Initialization of Components C_pm & C_pn & C_pp
			den_mat_t cross_cov_pred_ip;
			den_mat_t cov_mat_pred_id; // unused dummy variable
			den_mat_t cross_dist; // unused dummy variable
			T_mat sigma_resid_pred_obs;
			T_mat cov_mat_pred_obs; // unused dummy variable
			T_mat cross_dist_resid;
			T_mat sigma_resid_pred;
			T_mat cov_mat_pred; // unused dummy variable
			T_mat cross_dist_resid_pred;

			for (int j = 0; j < num_comps_total_; ++j) {
				// Construct components
				std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov_[cluster_i][j]->GetZSigmaZt();
				den_mat_t sigma_ip_stable = *(re_comps_ip_[cluster_i][j]->GetZSigmaZt());
				// Cross-covariance between predictions and inducing points C_pm
				std::shared_ptr<RECompGP<den_mat_t>> re_comp_cross_cov_cluster_i_pred_ip = std::dynamic_pointer_cast<RECompGP<den_mat_t>>(re_comps_cross_cov_[cluster_i][j]);
				re_comp_cross_cov_cluster_i_pred_ip->AddPredCovMatrices(re_comp_cross_cov_cluster_i_pred_ip->coords_ind_point_, gp_coords_mat_pred, cross_cov_pred_ip,
					cov_mat_pred_id, true, false, true, nullptr, false, cross_dist);
				// Calculating predictive mean
				pred_mean = cross_cov_pred_ip * chol_fact_sigma_ip_[cluster_i].solve((*cross_cov).transpose() * y_aux_[cluster_i]);
				den_mat_t chol_ip_cross_cov_ip_pred;
				std::shared_ptr<T_mat> sigma_resid;
				if (gp_approx_ == "full_scale_tapering") {
					// Residual matrix
					sigma_resid = re_comps_resid_[cluster_i][j]->GetZSigmaZt();
					// Cross-covariance between predictions and observations C_pn (tapered)
					std::shared_ptr<RECompGP<T_mat>> re_comps_resid_po_cluster_i = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid_[cluster_i][j]);
					re_comps_resid_po_cluster_i->AddPredCovMatrices(re_comps_resid_po_cluster_i->coords_, gp_coords_mat_pred, sigma_resid_pred_obs,
						cov_mat_pred_obs, true, false, true, nullptr, true, cross_dist_resid);
					// Calculate Cm_inv * C_mn part of predictive process
					den_mat_t sigma_ip_inv_cross_cov_ip_ob = chol_fact_sigma_ip_[cluster_i].solve((*cross_cov).transpose());
					// Residual part
					// Subtract predictive process (prediction) covariance
					SubtractProdFromNonSqMat<T_mat>(sigma_resid_pred_obs, cross_cov_pred_ip.transpose(), sigma_ip_inv_cross_cov_ip_ob);
					// Apply taper
					re_comps_resid_po_cluster_i->ApplyTaper(cross_dist_resid, sigma_resid_pred_obs);

					pred_mean += sigma_resid_pred_obs * y_aux_[cluster_i];
				}
				// Calculating predicitve covariance and variance
				if (calc_pred_cov || calc_pred_var) {
					// Add nugget and autocovariance of predictions 
					if (calc_pred_var) {
						if (gauss_likelihood_) {
							pred_var = vec_t::Ones(num_data_pred_cli);
						}
						else {
							pred_var = vec_t::Zero(num_data_pred_cli);
						}
						re_comp_cross_cov_cluster_i_pred_ip->AddPredUncondVar(pred_var.data(), num_data_pred_cli, nullptr);
					}
					T_mat PP_Part;
					// Add prediction matrix to predictive Covariance
					if (calc_pred_cov) {
						Log::REInfo("The computational complexity and the storage of the predictive covariance heavily depend on the number of prediction location. Therefore, if this number is large we recommend only computing the predictive variances.");
						pred_cov = T_mat(num_data_pred_cli, num_data_pred_cli);
						if (gauss_likelihood_) {
							pred_cov.setIdentity();
						}
						else {
							pred_cov.setZero();
						}
						TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_[cluster_i], cross_cov_pred_ip.transpose(), chol_ip_cross_cov_ip_pred, false);
						ConvertTo_T_mat_FromDense<T_mat>(chol_ip_cross_cov_ip_pred.transpose() * chol_ip_cross_cov_ip_pred, PP_Part);

						pred_cov += PP_Part;

						if (gp_approx_ == "full_scale_tapering") {
							std::shared_ptr<RECompGP<T_mat>> re_comps_resid_pp_cluster_i = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid_[cluster_i][j]);
							re_comps_resid_pp_cluster_i->AddPredCovMatrices(gp_coords_mat_pred, gp_coords_mat_pred, sigma_resid_pred,
								cov_mat_pred, true, false, true, nullptr, true, cross_dist_resid_pred);
							// Subtract predictive process (predict) 
							SubtractInnerProdFromMat<T_mat>(sigma_resid_pred, chol_ip_cross_cov_ip_pred, false);
							// Apply taper
							re_comps_resid_pp_cluster_i->ApplyTaper(cross_dist_resid_pred, sigma_resid_pred);
							pred_cov += sigma_resid_pred;
						}
						else if (gp_approx_ == "fitc") {
							vec_t diagonal_resid(num_data_pred_cli);
							diagonal_resid.setZero();
							diagonal_resid = diagonal_resid.array() + sigma_ip_stable.coeffRef(0, 0);
#pragma omp parallel for schedule(static)
							for (int ii = 0; ii < num_data_pred_cli; ++ii) {
								diagonal_resid[ii] -= chol_ip_cross_cov_ip_pred.col(ii).array().square().sum();
							}
							pred_cov += diagonal_resid.asDiagonal();
						}
					}
					// Calculate remaining part of predictive covariance
					T_mat woodbury_Part;
					T_mat cross_cov_part;
					if (calc_pred_cov) {
						if (gp_approx_ == "full_scale_tapering") {
							// Whole cross-covariance as dense matrix 
							den_mat_t sigma_obs_pred_dense = (*cross_cov) * chol_fact_sigma_ip_[cluster_i].solve(cross_cov_pred_ip.transpose());
							sigma_obs_pred_dense += sigma_resid_pred_obs.transpose();
							if (gp_approx_ == "iterative") {
								den_mat_t sigma_inv_sigma_obs_pred;
								CGFSA_MULTI_RHS<T_mat>(*sigma_resid, *cross_cov, chol_fact_sigma_ip_[cluster_i], sigma_obs_pred_dense, sigma_inv_sigma_obs_pred, NaN_found,
									num_data_cli, num_data_pred_cli, cg_max_num_it_tridiag_, cg_delta_conv_pred, cg_preconditioner_type_,
									chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
								ConvertTo_T_mat_FromDense<T_mat>(sigma_obs_pred_dense.transpose() * sigma_inv_sigma_obs_pred, cross_cov_part);
								pred_cov -= cross_cov_part;
							}
							else if (gp_approx_ == "cholesky") {
								den_mat_t sigma_resid_inv_sigma_obs_pred = chol_fact_resid_[cluster_i].solve(sigma_obs_pred_dense);
								den_mat_t sigma_resid_inv_sigma_obs_pred_cross_cov_pred_ip = sigma_resid_inv_sigma_obs_pred * cross_cov_pred_ip;
								ConvertTo_T_mat_FromDense<T_mat>(sigma_obs_pred_dense.transpose() * sigma_resid_inv_sigma_obs_pred, cross_cov_part);
								pred_cov -= cross_cov_part;
								ConvertTo_T_mat_FromDense<T_mat>(sigma_resid_inv_sigma_obs_pred_cross_cov_pred_ip * chol_fact_sigma_woodbury_[cluster_i].solve(sigma_resid_inv_sigma_obs_pred_cross_cov_pred_ip.transpose()), woodbury_Part);
								pred_cov += woodbury_Part;
							}
						}
						else if (gp_approx_ == "fitc") {
							ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_ip * chol_fact_sigma_ip_[cluster_i].solve(cross_cov_pred_ip.transpose()), cross_cov_part);
							pred_cov -= cross_cov_part;
							ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_ip * chol_fact_sigma_woodbury_[cluster_i].solve(cross_cov_pred_ip.transpose()), woodbury_Part);
							pred_cov += woodbury_Part;
						}
					} // end calc_pred_cov 
					// Calculate remaining part of predictive variances
					if (calc_pred_var) {
						if (gp_approx_ == "full_scale_tapering") {
							if (matrix_inversion_method_ == "iterative") {
								// Stochastic Diagonal
								// Sample vectors
								cg_generator_ = RNG_t(seed_rand_vec_trace_);
								den_mat_t rand_vec_probe_init(num_data_pred_cli, nsim_var_pred);
								GenRandVecDiag(cg_generator_, rand_vec_probe_init);
								den_mat_t rand_vec_probe_pred(num_data_cli, nsim_var_pred);
								rand_vec_probe_pred.setZero();
								// sigma_resid_pred^T * rand_vec_probe_init
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < rand_vec_probe_pred.cols(); ++i) {
									rand_vec_probe_pred.col(i) += sigma_resid_pred_obs.transpose() * rand_vec_probe_init.col(i);
								}
								// sigma_resid^-1 * rand_vec_probe_pred
								den_mat_t sigma_resid_inv_pv(num_data_cli, rand_vec_probe_pred.cols());
								CGFSA_RESID<T_mat>(*sigma_resid, rand_vec_probe_pred, sigma_resid_inv_pv, NaN_found, num_data_cli, (int)rand_vec_probe_pred.cols(),
									cg_max_num_it_tridiag_, cg_delta_conv_pred,
									cg_preconditioner_type_, diagonal_approx_inv_preconditioner_[cluster_i]);
								// sigma_resid_pred * sigma_resid_inv_pv
								den_mat_t rand_vec_probe_final(num_data_pred_cli, sigma_resid_inv_pv.cols());
								rand_vec_probe_final.setZero();
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < rand_vec_probe_final.cols(); ++i) {
									rand_vec_probe_final.col(i) += sigma_resid_pred_obs * sigma_resid_inv_pv.col(i);
								}
								den_mat_t sample_sigma = rand_vec_probe_final.cwiseProduct(rand_vec_probe_init);
								vec_t stoch_diag = sample_sigma.rowwise().mean();

								// Exact Diagonal (Preconditioner)
								vec_t diag_P(num_data_pred_cli);
								T_mat sigma_resid_pred_obs_pred_var = sigma_resid_pred_obs * (diagonal_approx_inv_preconditioner_[cluster_i].cwiseSqrt()).asDiagonal();
								T_mat* R_ptr_2 = &sigma_resid_pred_obs_pred_var;
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < num_data_pred_cli; ++i) {
									diag_P[i] = ((vec_t)(R_ptr_2->row(i))).array().square().sum();
								}

								// Stochastic Diagonal (Preconditioner)
								den_mat_t rand_vec_probe_cv(num_data_pred_cli, rand_vec_probe_init.cols());
								rand_vec_probe_cv.setZero();
								den_mat_t preconditioner_rand_vec_probe = diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal() * rand_vec_probe_pred;
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < preconditioner_rand_vec_probe.cols(); ++i) {
									rand_vec_probe_cv.col(i) += sigma_resid_pred_obs * preconditioner_rand_vec_probe.col(i);
								}
								den_mat_t sample_P = rand_vec_probe_cv.cwiseProduct(rand_vec_probe_init);
								vec_t diag_P_stoch = sample_P.rowwise().mean();

								// Variance Reduction
								// Optimal c
								vec_t c_opt;
								CalcOptimalCVectorized(sample_sigma, sample_P, stoch_diag, diag_P, c_opt);
								stoch_diag += c_opt.cwiseProduct(diag_P - diag_P_stoch);
								pred_var -= stoch_diag;
								// CG: sigma_resid^-1 * cross_cov
								den_mat_t sigma_resid_inv_cross_cov(num_data_cli, (*cross_cov).cols());
								CGFSA_RESID<T_mat>(*sigma_resid, *cross_cov, sigma_resid_inv_cross_cov, NaN_found, num_data_cli, (int)(*cross_cov).cols(),
									cg_max_num_it_tridiag_, cg_delta_conv_pred,
									cg_preconditioner_type_, diagonal_approx_inv_preconditioner_[cluster_i]);
								// CG: sigma^-1 * cross_cov
								den_mat_t sigma_inv_cross_cov(num_data_cli, (*cross_cov).cols());
								CGFSA_MULTI_RHS<T_mat>(*sigma_resid, *cross_cov, chol_fact_sigma_ip_[cluster_i], *cross_cov, sigma_inv_cross_cov, NaN_found,
									num_data_cli, (int)(*cross_cov).cols(), cg_max_num_it_tridiag_, cg_delta_conv_pred, cg_preconditioner_type_,
									chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
								// sigma_ip^-1 * cross_cov_pred
								den_mat_t sigma_ip_inv_cross_cov_pred = chol_fact_sigma_ip_[cluster_i].solve(cross_cov_pred_ip.transpose());
								// cross_cov^T * sigma^-1 * cross_cov
								den_mat_t auto_cross_cov = (*cross_cov).transpose() * sigma_inv_cross_cov;
								// cross_cov^T * sigma^-1 * cross_cov * sigma_ip^-1 * cross_cov_pred
								den_mat_t auto_cross_cov_sigma_ip_inv_cross_cov_pred = auto_cross_cov * sigma_ip_inv_cross_cov_pred;
								// sigma_resid_pred * sigma^-1 * cross_cov
								den_mat_t sigma_resid_pred_obs_sigma_inv_cross_cov(num_data_pred_cli, (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < sigma_resid_pred_obs_sigma_inv_cross_cov.cols(); ++i) {
									sigma_resid_pred_obs_sigma_inv_cross_cov.col(i) = sigma_resid_pred_obs * sigma_inv_cross_cov.col(i);
								}
								// cross_cov^T * sigma_resid^-1 * cross_cov
								den_mat_t cross_cov_sigma_resid_inv_cross_cov = (*cross_cov).transpose() * sigma_resid_inv_cross_cov;
								// Ensure symmetry
								cross_cov_sigma_resid_inv_cross_cov = (cross_cov_sigma_resid_inv_cross_cov + cross_cov_sigma_resid_inv_cross_cov.transpose()) / 2;
								// Woodburry factor
								chol_den_mat_t Woodburry_fact_chol;
								Woodburry_fact_chol.compute(sigma_ip_stable + cross_cov_sigma_resid_inv_cross_cov);
								den_mat_t Woodburry_fact;
								TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(Woodburry_fact_chol, sigma_resid_inv_cross_cov.transpose(), Woodburry_fact, false);
								den_mat_t sigma_resid_pred_obs_WF(num_data_pred_cli, (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < sigma_resid_pred_obs_WF.cols(); ++i) {
									sigma_resid_pred_obs_WF.col(i) = sigma_resid_pred_obs * Woodburry_fact.transpose().col(i);
								}
#pragma omp parallel for schedule(static)
								for (int i = 0; i < num_data_pred_cli; ++i) {
									pred_var[i] -= sigma_ip_inv_cross_cov_pred.col(i).dot(auto_cross_cov_sigma_ip_inv_cross_cov_pred.col(i))
										+ 2 * sigma_ip_inv_cross_cov_pred.col(i).dot(sigma_resid_pred_obs_sigma_inv_cross_cov.transpose().col(i))
										- sigma_resid_pred_obs_WF.transpose().col(i).array().square().sum();
								}
								if ((pred_var.array() < 0.0).any()) {
									Log::REWarning("There are negative estimates for variances. Use more sample vectors to reduce the variability of the stochastic estimate.");
								}
							}
							else {
								// sigma_resid^-1 * cross_cov
								den_mat_t sigma_resid_inv_cross_cov = chol_fact_resid_[cluster_i].solve((*cross_cov));
								// sigma_ip^-1 * cross_cov_pred^T
								den_mat_t sigma_ip_inv_cross_cov_pred = chol_fact_sigma_ip_[cluster_i].solve(cross_cov_pred_ip.transpose());
								// cross_cov^T * sigma_resid^-1 * cross_cov * sigma_ip^-1 * cross_cov_pred
								den_mat_t auto_cross_cov = ((*cross_cov).transpose() * sigma_resid_inv_cross_cov) * sigma_ip_inv_cross_cov_pred;
								// Sigma_resid_pred * sigma_resid^-1 * cross_cov
								den_mat_t sigma_resid_pred_obs_sigma_resid_inv_cross_cov(num_data_pred_cli, (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < sigma_resid_pred_obs_sigma_resid_inv_cross_cov.cols(); ++i) {
									sigma_resid_pred_obs_sigma_resid_inv_cross_cov.col(i) = sigma_resid_pred_obs * sigma_resid_inv_cross_cov.col(i);
								}
#pragma omp parallel for schedule(static)
								for (int i = 0; i < num_data_pred_cli; ++i) {
									pred_var[i] -= 2 * sigma_ip_inv_cross_cov_pred.col(i).dot(sigma_resid_pred_obs_sigma_resid_inv_cross_cov.transpose().col(i))
										+ auto_cross_cov.col(i).dot(sigma_ip_inv_cross_cov_pred.col(i));
								}
								vec_t sigma_resid_inv_sigma_resid_pred_col;
								T_mat* R_ptr = &sigma_resid_pred_obs;
								for (int i = 0; i < num_data_pred_cli; ++i) {
									TriangularSolveGivenCholesky<T_chol, T_mat, vec_t, vec_t>(chol_fact_resid_[cluster_i], ((vec_t)(R_ptr->row(i))).transpose(), sigma_resid_inv_sigma_resid_pred_col, false);
									pred_var[i] -= sigma_resid_inv_sigma_resid_pred_col.array().square().sum();
								}
								// Woodburry matrix part
								den_mat_t Woodburry_fact_sigma_resid_inv_cross_cov;
								TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury_[cluster_i], sigma_resid_inv_cross_cov.transpose(), Woodburry_fact_sigma_resid_inv_cross_cov, false);
								den_mat_t auto_cross_cov_pred = (Woodburry_fact_sigma_resid_inv_cross_cov * (*cross_cov)) * sigma_ip_inv_cross_cov_pred;
								den_mat_t sigma_resid_pred_obs_Woodburry_fact(num_data_pred_cli, (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < sigma_resid_pred_obs_Woodburry_fact.cols(); ++i) {
									sigma_resid_pred_obs_Woodburry_fact.col(i) = sigma_resid_pred_obs * Woodburry_fact_sigma_resid_inv_cross_cov.transpose().col(i);
								}
#pragma omp parallel for schedule(static)
								for (int i = 0; i < num_data_pred_cli; ++i) {
									pred_var[i] += 2 * auto_cross_cov_pred.col(i).dot(sigma_resid_pred_obs_Woodburry_fact.transpose().col(i))
										+ auto_cross_cov_pred.col(i).array().square().sum()
										+ sigma_resid_pred_obs_Woodburry_fact.transpose().col(i).array().square().sum();
								}
							}
						}//end FSA 
						else if (gp_approx_ == "fitc") { // Predictive Process
							den_mat_t sigma_ip_inv_cross_cov_pred = chol_fact_sigma_ip_[cluster_i].solve(cross_cov_pred_ip.transpose());
							den_mat_t Fact_FITC_R = (((*cross_cov).transpose() * FITC_Diag_[cluster_i].cwiseInverse().asDiagonal()) * (*cross_cov)) * sigma_ip_inv_cross_cov_pred;
							den_mat_t Woodburry_fact;
							TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury_[cluster_i], Fact_FITC_R, Woodburry_fact, false);
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_pred_cli; ++i) {
								pred_var[i] -= Fact_FITC_R.col(i).dot(sigma_ip_inv_cross_cov_pred.col(i))
									- Woodburry_fact.col(i).array().square().sum();
							}
						}//end predictive_process
					}//end calc_pred_var
				}//end calc_pred_cov || calc_pred_var
			}
		}//end CalcPredPPFSA

		friend class REModel;

	};

}  // end namespace GPBoost

#endif   // GPB_RE_MODEL_TEMPLATE_H_
