/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2024 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_VECCHIA_H_
#define GPB_VECCHIA_H_
#include <memory>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <GPBoost/utils.h>

namespace GPBoost {

	/*!
	* \brief Distance function
	* \param coord_ind_i Index i
	* \param coords_ind_j Indices j_1,...
	* \param coords Coordinates
	* \param corr_diag The diagonal of Sigma - Sigma_nm Sigma_m^(-1) Sigma_mn
	* \param chol_ip_cross_cov inverse of Cholesky factor of inducing point matrix times cross covariance: Sigma_ip^-1/2 Sigma_cross_cov
	* \param re_comps_vecchia_cluster_i Container that collects the individual component models
	* \param[out] distances
	* \param dist_function distance function
	* \param distances_saved
	*/
	void distances_funct(const int& coord_ind_i,
		const std::vector<int>& coords_ind_j,
		const den_mat_t& coords,
		const vec_t& corr_diag,
		const den_mat_t& chol_ip_cross_cov,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		vec_t& distances,
		string_t dist_function,
		bool distances_saved);

	/*!
	* \brief Construct Cover Tree
	* \param coords_mat Coordinates
	* \param chol_ip_cross_cov inverse of Cholesky factor of inducing point matrix times cross covariance: Sigma_ip^-1/2 Sigma_cross_cov
	* \param corr_diag The diagonal of Sigma - Sigma_nm Sigma_m^(-1) Sigma_mn
	* \param start Start index
	* \param re_comps_vecchia_cluster_i Container that collects the individual component models
	* \param[out] cover_tree
	* \param[out] level depth of tree
	* \param distances_saved
	* \param dist_function distance function
	*/
	void CoverTree_kNN(const den_mat_t& coords_mat,
		const den_mat_t& chol_ip_cross_cov,
		const vec_t& corr_diag,
		const int start,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		std::map<int, std::vector<int>>& cover_tree,
		int& level,
		bool distances_saved,
		string_t dist_function);

	/*!
	* \brief Find kNN
	* \param i Index
	* \param k number of neighbors
	* \param levels depth of tree
	* \param distances_saved
	* \param coord Coordinates
	* \param chol_ip_cross_cov inverse of Cholesky factor of inducing point matrix times cross covariance: Sigma_ip^-1/2 Sigma_cross_cov
	* \param corr_diag The diagonal of Sigma - Sigma_nm Sigma_m^(-1) Sigma_mn
	* \param re_comps_vecchia_cluster_i Container that collects the individual component models
	* \param[out] neighbors_i
	* \param[out] dist_of_neighbors_i
	* \param cover_tree
	* \param dist_function distance function
	*/
	void find_kNN_CoverTree(const int i,
		const int k,
		const int levels,
		const bool distances_saved,
		const den_mat_t& coords,
		const den_mat_t& chol_ip_cross_cov,
		const vec_t& corr_diag,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		std::vector<int>& neighbors_i,
		std::vector<double>& dist_of_neighbors_i,
		std::map<int, std::vector<int>>& cover_tree,
		string_t dist_function);

	/*!
	* \brief Finds the nearest_neighbors among the previous observations
	* \param coords Coordinates of observations
	* \param num_data Number of observations
	* \param num_neighbors Maximal number of neighbors
	* \param chol_ip_cross_cov inverse of Cholesky factor of inducing point matrix times cross covariance: Sigma_ip^-1/2 Sigma_cross_cov
	* \param re_comps_vecchia_cluster_i Container that collects the individual component models
	* \param[out] neighbors Vector with indices of neighbors for every observations (length = num_data - start_at)
	* \param[out] dist_obs_neighbors Distances needed for the Vecchia approximation: distances between locations and their neighbors (length = num_data - start_at)
	* \param[out] dist_between_neighbors Distances needed for the Vecchia approximation: distances between all neighbors (length = num_data - start_at)
	* \param start_at Index of first point for which neighbors should be found (useful for prediction, otherwise = 0)
	* \param end_search_at Index of last point which can be a neighbor (useful for prediction when the neighbors are only to be found among the observed data, otherwise = num_data - 1 (if end_search_at < 0, we set end_search_at = num_data - 1)
	* \param[out] check_has_duplicates If true, it is checked whether there are duplicates in coords among the neighbors (result written on output)
	* \param save_distances If true, distances are saved in dist_obs_neighbors and dist_between_neighbors
	* \param prediction If true find neighbors for prediction locations
	* \param cond_on_all If true condition on all points
	* \param number of observations
	*/
	void find_nearest_neighbors_Vecchia_FSA_fast(const den_mat_t& coords,
		int num_data,
		int num_neighbors,
		const den_mat_t& chol_ip_cross_cov,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		std::vector<std::vector<int>>& neighbors,
		std::vector<den_mat_t>& dist_obs_neighbors,
		std::vector<den_mat_t>& dist_between_neighbors,
		int start_at,
		int end_search_at,
		bool& check_has_duplicates,
		bool save_distances,
		bool prediction,
		bool cond_on_all,
		const int& num_data_obs);

	/*!
	* \brief Finds the nearest_neighbors among the previous observations
	* \param dist Distance between all observations
	* \param num_data Number of observations
	* \param num_neighbors Maximal number of neighbors
	* \param[out] nearest_neighbor Vector with indices of nearest neighbors for every observations
	*/
	void find_nearest_neighbors_Vecchia(den_mat_t& dist,
		int num_data,
		int num_neighbors,
		std::vector<std::vector<int>>& neighbors);

	/*!
	* \brief Finds the nearest_neighbors among the previous observations using the fast mean-distance-ordered nn search by Ra and Kim (1993)
	* \param coords Coordinates of observations
	* \param num_data Number of observations
	* \param num_neighbors Maximal number of neighbors
	* \param[out] neighbors Vector with indices of neighbors for every observations (length = num_data - start_at)
	* \param[out] dist_obs_neighbors Distances needed for the Vecchia approximation: distances between locations and their neighbors (length = num_data - start_at)
	* \param[out] dist_between_neighbors Distances needed for the Vecchia approximation: distances between all neighbors (length = num_data - start_at)
	* \param start_at Index of first point for which neighbors should be found (useful for prediction, otherwise = 0)
	* \param end_search_at Index of last point which can be a neighbor (useful for prediction when the neighbors are only to be found among the observed data, otherwise = num_data - 1 (if end_search_at < 0, we set end_search_at = num_data - 1)
	* \param[out] check_has_duplicates If true, it is checked whether there are duplicates in coords among the neighbors (result written on output)
	* \param neighbor_selection The way how neighbors are selected
	* \param gen RNG
	* \param save_distances If true, distances are saved in dist_obs_neighbors and dist_between_neighbors
	*/
	void find_nearest_neighbors_Vecchia_fast(const den_mat_t& coords,
		int num_data,
		int num_neighbors,
		std::vector<std::vector<int>>& neighbors,
		std::vector<den_mat_t>& dist_obs_neighbors,
		std::vector<den_mat_t>& dist_between_neighbors,
		int start_at,
		int end_search_at,
		bool& check_has_duplicates,
		const string_t& neighbor_selection,
		RNG_t& gen,
		bool save_distances);

	void find_nearest_neighbors_fast_internal(const int i,
		const int num_data,
		const int num_nearest_neighbors,
		const int end_search_at,
		const int dim_coords,
		const den_mat_t& coords,
		const std::vector<int>& sort_sum,
		const std::vector<int>& sort_inv_sum,
		const std::vector<double>& coords_sum,
		std::vector<int>& neighbors_i,
		std::vector<double>& nn_square_dist);

	/*!
	* \brief Initialize individual component models and collect them in a containter when the Vecchia approximation is used
	* \param num_data Number of data points
	* \param dim_gp_coords Dimension of the coordinates (=number of features) for Gaussian process
	* \param data_indices_per_cluster Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points
	* \param cluster_i Index / label of the realization of the Gaussian process for which the components should be constructed
	* \param num_data_per_cluster Keys: Labels of independent realizations of REs/GPs, values: number of data points per independent realization
	* \param gp_coords_data Coordinates (features) for Gaussian process
	* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
	* \param[out] re_comps_vecchia_cluster_i Container that collects the individual component models
	* \param[out] nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param[out] dist_obs_neighbors_cluster_i Distances between locations and their nearest neighbors
	* \param[out] dist_between_neighbors_cluster_i Distances between nearest neighbors for all locations
	* \param[out] entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param[out] z_outer_z_obs_neighbors_cluster_i Outer product of covariate vector at observations and neighbors with itself for random coefficients. First index = data point i, second index = GP number j
	* \param[out] only_one_GP_calculations_on_RE_scale
	* \param[out] has_duplicates_coords If true, there are duplicates in coords among the neighbors (currently only used for the Vecchia approximation for non-Gaussian likelihoods)
	* \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
	* \param num_neighbors The number of neighbors used in the Vecchia approximation
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param check_has_duplicates If true, it is checked whether there are duplicate locations
	* \param rng Random number generator
	* \param num_gp_rand_coef Number of random coefficient GPs
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param num_comps_total Total number of random effect components (grouped REs plus other GPs)
	* \param gauss_likelihood If true, the response variables have a Gaussian likelihood, otherwise not
	* \param cov_fct Type of covariance function
	* \param cov_fct_shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. This parameter is irrelevant for some covariance functions such as the exponential or Gaussian
	* \param cov_fct_taper_range Range parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
	* \param cov_fct_taper_shape Shape parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
	* \param apply_tapering If true, tapering is applied to the covariance function (element-wise multiplication with a compactly supported Wendland correlation function)
	* \param save_distances_isotropic_cov_fct If true, distances among points and neighbors are saved for Vecchia approximations for isotropic covariance functions
	* \param gp_approx Gaussian process approximation
	*/
	void CreateREComponentsVecchia(data_size_t num_data,
		int dim_gp_coords,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster,
		data_size_t cluster_i,
		std::map<data_size_t, int>& num_data_per_cluster,
		const double* gp_coords_data,
		const double* gp_rand_coef_data,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		std::vector<Triplet_t>& entries_init_B_cluster_i,
		std::vector<std::vector<den_mat_t>>& z_outer_z_obs_neighbors_cluster_i,
		bool& only_one_GP_calculations_on_RE_scale,
		bool& has_duplicates_coords,
		string_t vecchia_ordering,
		int num_neighbors,
		const string_t& vecchia_neighbor_selection,
		bool check_has_duplicates,
		RNG_t& rng,
		int num_gp_rand_coef,
		int num_gp_total,
		int num_comps_total,
		bool gauss_likelihood,
		string_t cov_fct,
		double cov_fct_shape,
		double cov_fct_taper_range,
		double cov_fct_taper_shape,
		bool apply_tapering,
		bool save_distances_isotropic_cov_fct,
		string_t& gp_approx);

	/*!
	* \brief Update the nearest neighbors based on scaled coorrdinates
	* \param[out] re_comps_vecchia_cluster_i Container that collects the individual component models
	* \param[out] nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param[out] entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param num_neighbors The number of neighbors used in the Vecchia approximation
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param rng Random number generator
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps_vecchia') of the intercept GP associated with the random coefficient GPs
	* \param[out] has_duplicates_coords If true, there are duplicates in coords among the neighbors (currently only used for the Vecchia approximation for non-Gaussian likelihoods)
	* \param check_has_duplicates If true, it is checked whether there are duplicate locations
	* \param gauss_likelihood If true, the response variables have a Gaussian likelihood, otherwise not
	* \param gp_approx Gaussian process approximation
	* \param chol_ip_cross_cov inverse of Cholesky factor of inducing point matrix times cross covariance : Sigma_ip ^ -1 / 2 Sigma_cross_cov
	* \param[out] dist_obs_neighbors Distances needed for the Vecchia approximation : distances between locations and their neighbors(length = num_data - start_at)
	* \param[out] dist_between_neighbors Distances needed for the Vecchia approximation : distances between all neighbors(length = num_data - start_at)
	* \param save_distances_isotropic_cov_fct If true, distances among points and neighbors are saved for Vecchia approximations for isotropic covariance functions
	*/
	void UpdateNearestNeighbors(std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		std::vector<Triplet_t>& entries_init_B_cluster_i,
		int num_neighbors,
		const string_t& vecchia_neighbor_selection,
		RNG_t& rng,
		int ind_intercept_gp,
		bool& has_duplicates_coords,
		bool check_has_duplicates,
		bool gauss_likelihood,
		string_t& gp_approx,
		const den_mat_t& chol_ip_cross_cov,
		std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		bool save_distances_isotropic_cov_fct);

	/*!
	* \brief Calculate matrices A and D_inv and their derivatives for the Vecchia approximation for one cluster (independent realization of GP)
	* \param num_re_cluster_i Number of random effects
	* \param calc_cov_factor If true, A and D_inv are calculated
	* \param calc_gradient If true, gradient are calculated
	* \param re_comps_vecchia_cluster_i Container that collects the individual component models
	* \param re_comps_cross_cov_cluster_i Container that collects the individual component models
	* \param re_comps_ip_cluster_i Container that collects the individual component models
	* \param Cholesky factor of inducing point matrix
	* \param Sigma_m^(-1/2) Sigma_mn
	* \param nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param dist_obs_neighbors_cluster_i Distances between locations and their nearest neighbors
	* \param dist_between_neighbors_cluster_i Distances between nearest neighbors for all locations
	* \param entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param z_outer_z_obs_neighbors_cluster_i Outer product of covariate vector at observations and neighbors with itself for random coefficients. First index = data point i, second index = GP number j
	* \param[out] B_cluster_i Matrix A = I - B (= Cholesky factor of inverse covariance) for Vecchia approximation
	* \param[out] D_inv_cluster_i Diagonal matrices D^-1 for Vecchia approximation
	* \param[out] B_grad_cluster_i Derivatives of matrices A ( = derivative of matrix -B) for Vecchia approximation
	* \param[out] D_grad_cluster_i Derivatives of matrices D for Vecchia approximation
	* \param Sigma_m^(-1) Sigma_mn
	* \param grad(Sigma_m) Sigma_m^(-1) Sigma_mn
	* \param transf_scale If true, the derivatives are taken on the transformed scale otherwise on the original scale. Default = true
	* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back)
	* \param calc_gradient_nugget If true, derivatives are also taken with respect to the nugget / noise variance
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps_vecchia') of the intercept GP associated with the random coefficient GPs
	* \param gauss_likelihood If true, the response variables have a Gaussian likelihood, otherwise not
	* \param save_distances_isotropic_cov_fct If true, distances among points and neighbors are saved for Vecchia approximations for isotropic covariance functions
	* \param gp_approx Gaussian process approximation
	* \param add_diagonal Vector of (additional) observation specific nugget / error variance added to the diagonal
	*/
	void CalcCovFactorGradientVecchia(data_size_t num_re_cluster_i,
		bool calc_cov_factor,
		bool calc_gradient,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
		const chol_den_mat_t& chol_fact_sigma_ip_cluster_i,
		const den_mat_t& chol_ip_cross_cov_cluster_i,
		const std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		const std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		const std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		const std::vector<Triplet_t>& entries_init_B_cluster_i,
		const std::vector<std::vector<den_mat_t>>& z_outer_z_obs_neighbors_cluster_i,
		sp_mat_t& B_cluster_i,
		sp_mat_t& D_inv_cluster_i,
		std::vector<sp_mat_t>& B_grad_cluster_i,
		std::vector<sp_mat_t>& D_grad_cluster_i,
		den_mat_t& sigma_ip_inv_cross_cov_T_cluster_i,
		std::vector<den_mat_t>& sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i,
		bool transf_scale,
		double nugget_var,
		bool calc_gradient_nugget,
		int num_gp_total,
		int ind_intercept_gp,
		bool gauss_likelihood,
		bool save_distances_isotropic_cov_fct,
		string_t& gp_approx,
		const double* add_diagonal);

	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the covariance matrix of the observable process when observed locations appear first in the ordering
	* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data
	* \param cluster_i Cluster index for which prediction are made
	* \param num_data_pred Total number of prediction locations (over all clusters)
	* \param Cross covariance matrix between inducing points and observations
	* \param Cholesky factor of inducing point matrix
	* \param Cholesky factor of Woodbury matrix
	* \param Cross covariance matrix between inducing points and prediction locations
	* \param Vecchia matrix B
	* \param Vecchia matrix B^T D^(-1)
	* \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
	* \param gp_coords_mat_obs Coordinates for observed locations
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param gp_rand_coef_data_pred Random coefficient data for GPs
	* \param coordinates of inducing points
	* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param re_comps_vecchia Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps_vecchia') of the intercept GP associated with the random coefficient GPs
	* \param num_gp_rand_coef Number of random coefficient GPs
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param y_cluster_i Reponse variable data
	* \param gauss_likelihood If true, the response variables have a Gaussian likelihood, otherwise not
	* \param rng Random number generator
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param[out] pred_mean Predictive mean (only for Gaussian likelihoods)
	* \param[out] pred_cov Predictive covariance matrix (only for Gaussian likelihoods)
	* \param[out] pred_var Predictive variances (only for Gaussian likelihoods)
	* \param[out] Bpo Lower left part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the observed locations (only for non-Gaussian likelihoods)
	* \param[out] Bp Lower right part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the prediction locations (only for non-Gaussian likelihoods)
	* \param[out] Dp Diagonal matrix with lower right part of matrix D in joint Vecchia approximation for observed and prediction locations (only for non-Gaussian likelihoods)
	* \param save_distances_isotropic_cov_fct If true, distances among points and neighbors are saved for Vecchia approximations for isotropic covariance functions
	* \param gp_approx Gaussian process approximation
	*/
	void CalcPredVecchiaObservedFirstOrder(bool CondObsOnly,
		data_size_t cluster_i,
		int num_data_pred,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const chol_den_mat_t& chol_fact_sigma_ip_cluster_i,
		const chol_den_mat_t& chol_fact_sigma_woodbury_cluster_i,
		den_mat_t& cross_cov_pred_ip,
		const sp_mat_rm_t& B_cluster_i,
		const sp_mat_rm_t& Bt_D_inv_cluster_i,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster_pred,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		const double* gp_rand_coef_data_pred,
		const den_mat_t& gp_coords_mat_ip,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia,
		int ind_intercept_gp,
		int num_gp_rand_coef,
		int num_gp_total,
		const vec_t& y_cluster_i,
		bool gauss_likelihood,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		vec_t& pred_mean,
		den_mat_t& pred_cov,
		vec_t& pred_var,
		sp_mat_t& Bpo,
		sp_mat_t& Bp,
		vec_t& Dp,
		bool save_distances_isotropic_cov_fct,
		string_t& gp_approx);

	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the covariance matrix of the observable proces when prediction locations appear first in the ordering
	* \param cluster_i Cluster index for which prediction are made
	* \param num_data_pred Total number of prediction locations (over all clusters)
	* \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
	* \param gp_coords_mat_obs Coordinates for observed locations
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param gp_rand_coef_data_pred Random coefficient data for GPs
	* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param re_comps_vecchia Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps_vecchia') of the intercept GP associated with the random coefficient GPs
	* \param num_gp_rand_coef Number of random coefficient GPs
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param y_cluster_i Reponse variable data
	* \param rng Random number generator
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param[out] pred_mean Predictive mean
	* \param[out] pred_cov Predictive covariance matrix
	* \param[out] pred_var Predictive variances
	* \param save_distances_isotropic_cov_fct If true, distances among points and neighbors are saved for Vecchia approximations for isotropic covariance functions
	*/
	void CalcPredVecchiaPredictedFirstOrder(data_size_t cluster_i,
		int num_data_pred,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster_pred,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		const double* gp_rand_coef_data_pred,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia,
		int ind_intercept_gp,
		int num_gp_rand_coef,
		int num_gp_total,
		const vec_t& y_cluster_i,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		vec_t& pred_mean,
		den_mat_t& pred_cov,
		vec_t& pred_var,
		bool save_distances_isotropic_cov_fct);

	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the latent process when observed locations appear first in the ordering (only for Gaussian likelihoods)
	* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data
	* \param cluster_i Cluster index for which prediction are made
	* \param gp_coords_mat_obs Coordinates for observed locations
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param re_comps_vecchia Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps_vecchia') of the intercept GP associated with the random coefficient GPs
	* \param y_cluster_i Reponse variable data
	* \param rng Random number generator
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects (only has an effect on pred_cov and pred_var)
	* \param[out] pred_mean Predictive mean
	* \param[out] pred_cov Predictive covariance matrix
	* \param[out] pred_var Predictive variances
	* \param save_distances_isotropic_cov_fct If true, distances among points and neighbors are saved for Vecchia approximations for isotropic covariance functions
	 */
	void CalcPredVecchiaLatentObservedFirstOrder(bool CondObsOnly,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia,
		int ind_intercept_gp,
		const vec_t& y_cluster_i,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool predict_response,
		vec_t& pred_mean,
		den_mat_t& pred_cov,
		vec_t& pred_var,
		bool save_distances_isotropic_cov_fct);

}  // namespace GPBoost

#endif   // GPB_VECCHIA_H_
