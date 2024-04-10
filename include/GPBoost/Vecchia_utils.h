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
	* \param[out] re_comps_cluster_i Container that collects the individual component models
	* \param[out] nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param[out] dist_obs_neighbors_cluster_i Distances between locations and their nearest neighbors
	* \param[out] dist_between_neighbors_cluster_i Distances between nearest neighbors for all locations
	* \param[out] entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param[out] entries_init_B_grad_cluster_i Triplets for initializing the matrices B_grad
	* \param[out] z_outer_z_obs_neighbors_cluster_i Outer product of covariate vector at observations and neighbors with itself for random coefficients. First index = data point i, second index = GP number j
	* \param[out] only_one_GP_calculations_on_RE_scale
	* \param[out] has_duplicates_coords If true, there are duplicates in coords among the neighbors (currently only used for the Vecchia approximation for non-Gaussian likelihoods)
	* \param vecchia_ordering Ordering used in the Vecchia approximation. "none" = no ordering, "random" = random ordering
	* \param num_neighbors The number of neighbors used in the Vecchia approximation
	* \param vecchia_neighbor_selection The way how neighbors are selected
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
	*/
	template<typename T_mat>
	void CreateREComponentsVecchia(data_size_t num_data,
		int dim_gp_coords,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster,
		data_size_t cluster_i,
		std::map<data_size_t, int>& num_data_per_cluster,
		const double* gp_coords_data,
		const double* gp_rand_coef_data,
		std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
		std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		std::vector<Triplet_t>& entries_init_B_cluster_i,
		std::vector<Triplet_t>& entries_init_B_grad_cluster_i,
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
		bool apply_tapering) {
		int ind_intercept_gp = (int)re_comps_cluster_i.size();
		if (vecchia_ordering == "random" || vecchia_ordering == "time_random_space") {
			std::shuffle(data_indices_per_cluster[cluster_i].begin(), data_indices_per_cluster[cluster_i].end(), rng);
		}
		std::vector<double> gp_coords;
		for (int j = 0; j < dim_gp_coords; ++j) {
			for (const auto& id : data_indices_per_cluster[cluster_i]) {
				gp_coords.push_back(gp_coords_data[j * num_data + id]);
			}
		}
		den_mat_t gp_coords_mat = Eigen::Map<den_mat_t>(gp_coords.data(), num_data_per_cluster[cluster_i], dim_gp_coords);
		if (vecchia_ordering == "time" || vecchia_ordering == "time_random_space") {
			std::vector<double> coord_time(gp_coords_mat.rows());
#pragma omp for schedule(static)
			for (int i = 0; i < (int)gp_coords_mat.rows(); ++i) {
				coord_time[i] = gp_coords_mat.coeff(i, 0);
			}
			std::vector<int> sort_time;
			SortIndeces<double>(coord_time, sort_time);
			den_mat_t gp_coords_mat_not_sort = gp_coords_mat;
			gp_coords_mat = gp_coords_mat_not_sort(sort_time, Eigen::all);
			gp_coords_mat_not_sort.resize(0, 0);
			std::vector<int> dt_idx_unsorted = data_indices_per_cluster[cluster_i];
#pragma omp parallel for schedule(static)
			for (int i = 0; i < (int)gp_coords_mat.rows(); ++i) {
				data_indices_per_cluster[cluster_i][i] = dt_idx_unsorted[sort_time[i]];
			}
		}
		only_one_GP_calculations_on_RE_scale = num_gp_total == 1 && num_comps_total == 1 && !gauss_likelihood;
		re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T_mat>>(new RECompGP<T_mat>(
			gp_coords_mat,
			cov_fct,
			cov_fct_shape,
			cov_fct_taper_range,
			cov_fct_taper_shape,
			apply_tapering,
			false,
			false,
			only_one_GP_calculations_on_RE_scale,
			only_one_GP_calculations_on_RE_scale)));
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp]);
		if (re_comp->GetNumUniqueREs() == num_data_per_cluster[cluster_i]) {
			only_one_GP_calculations_on_RE_scale = false;
		}
		bool has_duplicates = check_has_duplicates;
		nearest_neighbors_cluster_i = std::vector<std::vector<int>>(re_comp->GetNumUniqueREs());
		dist_obs_neighbors_cluster_i = std::vector<den_mat_t>(re_comp->GetNumUniqueREs());
		dist_between_neighbors_cluster_i = std::vector<den_mat_t>(re_comp->GetNumUniqueREs());
		find_nearest_neighbors_Vecchia_fast(re_comp->GetCoords(), re_comp->GetNumUniqueREs(), num_neighbors,
			nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, has_duplicates,
			vecchia_neighbor_selection, rng, re_comp->ShouldSaveDistances());
		if ((vecchia_ordering == "time" || vecchia_ordering == "time_random_space") && !(re_comp->IsSpaceTimeModel())) {
			Log::REFatal("'vecchia_ordering' is '%s' but the 'cov_function' is not a space-time covariance function ", vecchia_ordering.c_str());
		}
		if (check_has_duplicates) {
			has_duplicates_coords = has_duplicates_coords || has_duplicates;
			if (!gauss_likelihood && has_duplicates_coords) {
				Log::REFatal("Duplicates found in the coordinates for the Gaussian process. "
					"This is currently not supported for the Vecchia approximation for non-Gaussian likelihoods ");
			}
		}
		for (int i = 0; i < re_comp->GetNumUniqueREs(); ++i) {
			for (int j = 0; j < (int)nearest_neighbors_cluster_i[i].size(); ++j) {
				entries_init_B_cluster_i.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.));
				entries_init_B_grad_cluster_i.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.));
			}
			entries_init_B_cluster_i.push_back(Triplet_t(i, i, 1.));//Put 1's on the diagonal since B = I - A
		}
		//Random coefficients
		if (num_gp_rand_coef > 0) {
			if (!(re_comp->ShouldSaveDistances())) {
				Log::REFatal("Random coefficient processes are not supported for covariance functions "
					"for which the neighbors are dynamically determined based on correlations");
			}
			z_outer_z_obs_neighbors_cluster_i = std::vector<std::vector<den_mat_t>>(re_comp->GetNumUniqueREs());
			for (int j = 0; j < num_gp_rand_coef; ++j) {
				std::vector<double> rand_coef_data;
				for (const auto& id : data_indices_per_cluster[cluster_i]) {
					rand_coef_data.push_back(gp_rand_coef_data[j * num_data + id]);
				}
				re_comps_cluster_i.push_back(std::shared_ptr<RECompGP<T_mat>>(new RECompGP<T_mat>(
					rand_coef_data,
					cov_fct,
					cov_fct_shape,
					cov_fct_taper_range,
					cov_fct_taper_shape,
					re_comp->GetTaperMu(),
					apply_tapering,
					false,
					dim_gp_coords)));
				//save random coefficient data in the form ot outer product matrices
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_per_cluster[cluster_i]; ++i) {
					if (j == 0) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef);
					}
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
		}// end random coefficients
	}//end CreateREComponentsVecchia

	/*!
	* \brief Update the nearest neighbors based on scaled coorrdinates
	* \param[out] re_comps_cluster_i Container that collects the individual component models
	* \param[out] nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param[out] entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param[out] entries_init_B_grad_cluster_i Triplets for initializing the matrices B_grad
	* \param num_neighbors The number of neighbors used in the Vecchia approximation
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param rng Random number generator
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	*/
	template<typename T_mat>
	void UpdateNearestNeighbors(std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
		std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		std::vector<Triplet_t>& entries_init_B_cluster_i,
		std::vector<Triplet_t>& entries_init_B_grad_cluster_i,
		int num_neighbors,
		const string_t& vecchia_neighbor_selection,
		RNG_t& rng,
		int ind_intercept_gp) {
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp]);
		CHECK(re_comp->ShouldSaveDistances() == false);
		int num_re = re_comp->GetNumUniqueREs();
		CHECK((int)nearest_neighbors_cluster_i.size() == num_re);
		// Calculate scaled coordinates
		den_mat_t coords_scaled;
		re_comp->GetScaledCoordinates(coords_scaled);
		// find correlation-based nearest neighbors
		std::vector<den_mat_t> dist_dummy;
		bool check_has_duplicates = false;
		find_nearest_neighbors_Vecchia_fast(coords_scaled, num_re, num_neighbors,
			nearest_neighbors_cluster_i, dist_dummy, dist_dummy, 0, -1, check_has_duplicates,
			vecchia_neighbor_selection, rng, false);
		int ctr = 0, ctr_grad = 0;
		for (int i = 0; i < std::min(num_re, num_neighbors); ++i) {
			for (int j = 0; j < (int)nearest_neighbors_cluster_i[i].size(); ++j) {
				entries_init_B_cluster_i[ctr] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
				entries_init_B_grad_cluster_i[ctr_grad] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
				ctr++;
				ctr_grad++;
			}
			entries_init_B_cluster_i[ctr] = Triplet_t(i, i, 1.);//Put 1's on the diagonal since B = I - A
			ctr++;
		}
		if (num_neighbors < num_re) {
#pragma omp parallel for schedule(static)
			for (int i = num_neighbors; i < num_re; ++i) {
				CHECK((int)nearest_neighbors_cluster_i[i].size() == num_neighbors);
				for (int j = 0; j < num_neighbors; ++j) {
					entries_init_B_cluster_i[ctr + (i - num_neighbors) * (num_neighbors + 1) + j] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
					entries_init_B_grad_cluster_i[ctr_grad + (i - num_neighbors) * num_neighbors + j] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
				}
				entries_init_B_cluster_i[ctr + (i - num_neighbors) * (num_neighbors + 1) + num_neighbors] = Triplet_t(i, i, 1.);//Put 1's on the diagonal since B = I - A
			}
		}
	}//end UpdateNearestNeighbors

	/*!
	* \brief Calculate matrices A and D_inv as well as their derivatives for the Vecchia approximation for one cluster (independent realization of GP)
	* \param num_re_cluster_i Number of random effects
	* \param calc_gradient If true, the gradient also be calculated (only for Vecchia approximation)
	* \param re_comps_cluster_i Container that collects the individual component models
	* \param nearest_neighbors_cluster_i Collects indices of nearest neighbors
	* \param dist_obs_neighbors_cluster_i Distances between locations and their nearest neighbors
	* \param dist_between_neighbors_cluster_i Distances between nearest neighbors for all locations
	* \param entries_init_B_cluster_i Triplets for initializing the matrices B
	* \param entries_init_B_grad_cluster_i Triplets for initializing the matrices B_grad
	* \param z_outer_z_obs_neighbors_cluster_i Outer product of covariate vector at observations and neighbors with itself for random coefficients. First index = data point i, second index = GP number j
	* \param[out] B_cluster_i Matrix A = I - B (= Cholesky factor of inverse covariance) for Vecchia approximation
	* \param[out] D_inv_cluster_i Diagonal matrices D^-1 for Vecchia approximation
	* \param[out] B_grad_cluster_i Derivatives of matrices A ( = derivative of matrix -B) for Vecchia approximation
	* \param[out] D_grad_cluster_i Derivatives of matrices D for Vecchia approximation
	* \param transf_scale If true, the derivatives are taken on the transformed scale otherwise on the original scale. Default = true
	* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back)
	* \param calc_gradient_nugget If true, derivatives are also taken with respect to the nugget / noise variance
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	* \param gauss_likelihood If true, the response variables have a Gaussian likelihood, otherwise not
	*/
	template<typename T_mat>
	void CalcCovFactorVecchia(data_size_t num_re_cluster_i,
		bool calc_gradient,
		const std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
		const std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		const std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		const std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		const std::vector<Triplet_t>& entries_init_B_cluster_i,
		const std::vector<Triplet_t>& entries_init_B_grad_cluster_i,
		const std::vector<std::vector<den_mat_t>>& z_outer_z_obs_neighbors_cluster_i,
		sp_mat_t& B_cluster_i,
		sp_mat_t& D_inv_cluster_i,
		std::vector<sp_mat_t>& B_grad_cluster_i,
		std::vector<sp_mat_t>& D_grad_cluster_i,
		bool transf_scale,
		double nugget_var,
		bool calc_gradient_nugget,
		int num_gp_total,
		int ind_intercept_gp,
		bool gauss_likelihood) {
		int num_par_comp = re_comps_cluster_i[ind_intercept_gp]->NumCovPar();
		int num_par_gp = num_par_comp * num_gp_total + calc_gradient_nugget;
		//Initialize matrices B = I - A and D^-1 as well as their derivatives (in order that the code below can be run in parallel)
		B_cluster_i = sp_mat_t(num_re_cluster_i, num_re_cluster_i);//B = I - A
		B_cluster_i.setFromTriplets(entries_init_B_cluster_i.begin(), entries_init_B_cluster_i.end());//Note: 1's are put on the diagonal
		D_inv_cluster_i = sp_mat_t(num_re_cluster_i, num_re_cluster_i);//D^-1. Note: we first calculate D, and then take the inverse below
		D_inv_cluster_i.setIdentity();//Put 1's on the diagonal for nugget effect (entries are not overriden but added below)
		if (!transf_scale && gauss_likelihood) {
			D_inv_cluster_i.diagonal().array() = nugget_var;//nugget effect is not 1 if not on transformed scale
		}
		if (!gauss_likelihood) {
			D_inv_cluster_i.diagonal().array() = 0.;
		}
		bool exclude_marg_var_grad = !gauss_likelihood && (re_comps_cluster_i.size() == 1);//gradient is not needed if there is only one GP for non-Gaussian likelihoods
		if (calc_gradient) {
			B_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of B = derviateive of (-A)
			D_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of D
			for (int ipar = 0; ipar < num_par_gp; ++ipar) {
				if (!(exclude_marg_var_grad && ipar == 0)) {
					B_grad_cluster_i[ipar] = sp_mat_t(num_re_cluster_i, num_re_cluster_i);
					B_grad_cluster_i[ipar].setFromTriplets(entries_init_B_grad_cluster_i.begin(), entries_init_B_grad_cluster_i.end());
					D_grad_cluster_i[ipar] = sp_mat_t(num_re_cluster_i, num_re_cluster_i);
					D_grad_cluster_i[ipar].setIdentity();//Put 0 on the diagonal
					D_grad_cluster_i[ipar].diagonal().array() = 0.;
				}
			}
		}//end initialization
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_cluster_i[ind_intercept_gp]);
		bool distances_saved = re_comp->ShouldSaveDistances();
#pragma omp parallel for schedule(static)
		for (data_size_t i = 0; i < num_re_cluster_i; ++i) {
			int num_nn = (int)nearest_neighbors_cluster_i[i].size();
			//calculate covariance matrices between observations and neighbors and among neighbors as well as their derivatives
			den_mat_t cov_mat_obs_neighbors;
			den_mat_t cov_mat_between_neighbors;
			std::vector<den_mat_t> cov_grad_mats_obs_neighbors(num_par_gp);//covariance matrix plus derivative wrt to every parameter
			std::vector<den_mat_t> cov_grad_mats_between_neighbors(num_par_gp);
			den_mat_t coords_i, coords_nn_i;
			if (i > 0) {
				for (int j = 0; j < num_gp_total; ++j) {
					int ind_first_par = j * num_par_comp;//index of first parameter (variance) of component j in gradient vectors
					if (j == 0) {
						if (!distances_saved) {
							std::vector<int> ind{ i };
							re_comp->GetSubSetCoords(ind, coords_i);
							re_comp->GetSubSetCoords(nearest_neighbors_cluster_i[i], coords_nn_i);
						}
						re_comps_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, false);//write on matrices directly for first GP component
						re_comps_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors, cov_grad_mats_between_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, true);
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors_j, cov_grad_mats_obs_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, false);
						re_comps_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors_j, cov_grad_mats_between_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, true);
						//multiply by coefficient matrix
						cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();//cov_mat_obs_neighbors_j.cwiseProduct()
						//cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(0, 1, 1, num_nn)).array();//cov_mat_obs_neighbors_j.cwiseProduct()//DELETE_FIRST
						cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
						cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
						cov_mat_between_neighbors += cov_mat_between_neighbors_j;
						if (calc_gradient) {
							for (int ipar = 0; ipar < (int)num_par_comp; ++ipar) {
								cov_grad_mats_obs_neighbors[ind_first_par + ipar].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();
								cov_grad_mats_between_neighbors[ind_first_par + ipar].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
							}
						}
					}
				}//end loop over components j
			}//end if(i>1)
			//Calculate matrices B and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii}) and its derivatives
			for (int j = 0; j < num_gp_total; ++j) {
				double d_comp_j = re_comps_cluster_i[ind_intercept_gp + j]->CovPars()[0];
				if (!transf_scale && gauss_likelihood) {
					d_comp_j *= nugget_var;
				}
				if (j > 0) {//random coefficient
					d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
				}
				D_inv_cluster_i.coeffRef(i, i) += d_comp_j;
				if (calc_gradient) {
					if (!(exclude_marg_var_grad && j == 0)) {
						if (transf_scale) {
							D_grad_cluster_i[j * num_par_comp].coeffRef(i, i) = d_comp_j;//derivative of the covariance function wrt the variance. derivative of the covariance function wrt to range is zero on the diagonal
						}
						else {
							if (j == 0) {
								D_grad_cluster_i[j * num_par_comp].coeffRef(i, i) = 1.;//1's on the diagonal on the orignal scale
							}
							else {
								D_grad_cluster_i[j * num_par_comp].coeffRef(i, i) = z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
							}
						}
					}
				}
			}
			if (calc_gradient && calc_gradient_nugget) {
				D_grad_cluster_i[num_par_gp - 1].coeffRef(i, i) = 1.;
			}
			//2. remaining terms
			if (i > 0) {
				if (gauss_likelihood) {
					if (transf_scale) {
						cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
					}
					else {
						cov_mat_between_neighbors.diagonal().array() += nugget_var;
					}
				}
				else {
					cov_mat_between_neighbors.diagonal().array() += EPSILON_ADD_COVARIANCE_STABLE;//Avoid numerical problems when there is no nugget effect
				}
				den_mat_t A_i(1, num_nn);
				den_mat_t A_i_grad_sigma2;
				Eigen::LLT<den_mat_t> chol_fact_between_neighbors = cov_mat_between_neighbors.llt();
				A_i = (chol_fact_between_neighbors.solve(cov_mat_obs_neighbors)).transpose();
				for (int inn = 0; inn < num_nn; ++inn) {
					B_cluster_i.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) = -A_i(0, inn);
				}
				D_inv_cluster_i.coeffRef(i, i) -= (A_i * cov_mat_obs_neighbors)(0, 0);
				if (calc_gradient) {
					if (calc_gradient_nugget) {
						A_i_grad_sigma2 = -(chol_fact_between_neighbors.solve(A_i.transpose())).transpose();
					}
					den_mat_t A_i_grad(1, num_nn);
					for (int j = 0; j < num_gp_total; ++j) {
						int ind_first_par = j * num_par_comp;
						for (int ipar = 0; ipar < num_par_comp; ++ipar) {
							if (!(exclude_marg_var_grad && ipar == 0)) {
								A_i_grad = (chol_fact_between_neighbors.solve(cov_grad_mats_obs_neighbors[ind_first_par + ipar])).transpose() -
									A_i * ((chol_fact_between_neighbors.solve(cov_grad_mats_between_neighbors[ind_first_par + ipar])).transpose());
								for (int inn = 0; inn < num_nn; ++inn) {
									B_grad_cluster_i[ind_first_par + ipar].coeffRef(i, nearest_neighbors_cluster_i[i][inn]) = -A_i_grad(0, inn);
								}
								if (ipar == 0) {
									D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) -= ((A_i_grad * cov_mat_obs_neighbors)(0, 0) +
										(A_i * cov_grad_mats_obs_neighbors[ind_first_par + ipar])(0, 0));//add to derivative of diagonal elements for marginal variance 
								}
								else {
									D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) = -((A_i_grad * cov_mat_obs_neighbors)(0, 0) +
										(A_i * cov_grad_mats_obs_neighbors[ind_first_par + ipar])(0, 0));//don't add to existing values since derivative of diagonal is zero for range
								}
							}
						}
					}
					if (calc_gradient_nugget) {
						for (int inn = 0; inn < num_nn; ++inn) {
							B_grad_cluster_i[num_par_gp - 1].coeffRef(i, nearest_neighbors_cluster_i[i][inn]) = -A_i_grad_sigma2(0, inn);
						}
						D_grad_cluster_i[num_par_gp - 1].coeffRef(i, i) -= (A_i_grad_sigma2 * cov_mat_obs_neighbors)(0, 0);
					}
				}//end calc_gradient
			}//end if i > 0
			D_inv_cluster_i.coeffRef(i, i) = 1. / D_inv_cluster_i.coeffRef(i, i);
		}//end loop over data i
		Eigen::Index minRow, minCol;
		double min_D_inv = D_inv_cluster_i.diagonal().minCoeff(&minRow, &minCol);
		if (min_D_inv <= 0.) {
			const char* min_D_inv_below_zero_msg = "The matrix D in the Vecchia approximation contains negative or zero values. "
				"This likely results from numerical instabilities ";
			if (gauss_likelihood) {
				Log::REWarning(min_D_inv_below_zero_msg);
			}
			else {
				Log::REFatal(min_D_inv_below_zero_msg);
			}
		}
	}//end CalcCovFactorVecchia

	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the covariance matrix of the observable process when observed locations appear first in the ordering
	* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data
	* \param cluster_i Cluster index for which prediction are made
	* \param num_data_pred Total number of prediction locations (over all clusters)
	* \param data_indices_per_cluster_pred Keys: labels of independent clusters, values: vectors with indices for data points that belong to the every cluster
	* \param gp_coords_mat_obs Coordinates for observed locations
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param gp_rand_coef_data_pred Random coefficient data for GPs
	* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param re_comps Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
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
	*/
	template<typename T_mat>
	void CalcPredVecchiaObservedFirstOrder(bool CondObsOnly,
		data_size_t cluster_i,
		int num_data_pred,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster_pred,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		const double* gp_rand_coef_data_pred,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompBase<T_mat>>>>& re_comps,
		int ind_intercept_gp,
		int num_gp_rand_coef,
		int num_gp_total,
		const vec_t& y_cluster_i,
		bool gauss_likelihood,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var,
		sp_mat_t& Bpo,
		sp_mat_t& Bp,
		vec_t& Dp) {
		data_size_t num_re_cli = re_comps[cluster_i][ind_intercept_gp]->GetNumUniqueREs();
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps[cluster_i][ind_intercept_gp]);
		int num_re_pred_cli = (int)gp_coords_mat_pred.rows();
		//Find nearest neighbors
		den_mat_t coords_all(num_re_cli + num_re_pred_cli, gp_coords_mat_obs.cols());
		coords_all << gp_coords_mat_obs, gp_coords_mat_pred;
		std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_re_pred_cli);
		std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_re_pred_cli);
		std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_re_pred_cli);
		bool check_has_duplicates = false;
		bool distances_saved = re_comp->ShouldSaveDistances();
		den_mat_t coords_scaled;
		if (!distances_saved) {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all, coords_scaled);
		}
		if (CondObsOnly) {
			if (distances_saved) {
				find_nearest_neighbors_Vecchia_fast(coords_all, num_re_cli + num_re_pred_cli, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli, num_re_cli - 1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			else {
				find_nearest_neighbors_Vecchia_fast(coords_scaled, num_re_cli + num_re_pred_cli, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli, num_re_cli - 1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
		}
		else {//find neighbors among both the observed and prediction locations
			if (!gauss_likelihood) {
				check_has_duplicates = true;
			}
			if (distances_saved) {
				find_nearest_neighbors_Vecchia_fast(coords_all, num_re_cli + num_re_pred_cli, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli, -1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			else {
				find_nearest_neighbors_Vecchia_fast(coords_scaled, num_re_cli + num_re_pred_cli, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli, -1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			if (check_has_duplicates) {
				Log::REFatal("Duplicates found among training and test coordinates. "
					"This is not supported for predictions with a Vecchia approximation for non-Gaussian likelihoods "
					"when neighbors are selected among both training and test points ('_cond_all') ");
			}
		}
		//Random coefficients
		std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_re_pred_cli);
		if (num_gp_rand_coef > 0) {
			for (int j = 0; j < num_gp_rand_coef; ++j) {
				std::vector<double> rand_coef_data = re_comps[cluster_i][ind_intercept_gp + j + 1]->RandCoefData();//First entries are the observed data, then the predicted data
				for (const auto& id : data_indices_per_cluster_pred[cluster_i]) {//TODO: maybe do the following in parallel? (see CalcPredVecchiaPredictedFirstOrder)
					rand_coef_data.push_back(gp_rand_coef_data_pred[j * num_data_pred + id]);
				}
#pragma omp for schedule(static)
				for (int i = 0; i < num_re_pred_cli; ++i) {
					if (j == 0) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef);
					}
					int dim_z = (int)nearest_neighbors_cluster_i[i].size() + 1;
					vec_t coef_vec(dim_z);
					coef_vec(0) = rand_coef_data[num_re_cli + i];
					if ((num_re_cli + i) > 0) {
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
		for (int i = 0; i < num_re_pred_cli; ++i) {
			entries_init_Bp.push_back(Triplet_t(i, i, 1.));//Put 1 on the diagonal
			for (int inn = 0; inn < (int)nearest_neighbors_cluster_i[i].size(); ++inn) {
				if (nearest_neighbors_cluster_i[i][inn] < num_re_cli) {//nearest neighbor belongs to observed data
					entries_init_Bpo.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn], 0.));
				}
				else {//nearest neighbor belongs to predicted data
					entries_init_Bp.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][inn] - num_re_cli, 0.));
				}
			}
		}
		Bpo = sp_mat_t(num_re_pred_cli, num_re_cli);
		Bp = sp_mat_t(num_re_pred_cli, num_re_pred_cli);
		Dp = vec_t(num_re_pred_cli);
		Bpo.setFromTriplets(entries_init_Bpo.begin(), entries_init_Bpo.end());//initialize matrices (in order that the code below can be run in parallel)
		Bp.setFromTriplets(entries_init_Bp.begin(), entries_init_Bp.end());
		if (gauss_likelihood) {
			Dp.setOnes();//Put 1 on the diagonal (for nugget effect if gauss_likelihood, see comment below on why we add the nugget effect variance irrespective of 'predict_response')
		}
		else {
			Dp.setZero();
		}
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_re_pred_cli; ++i) {
			int num_nn = (int)nearest_neighbors_cluster_i[i].size();
			den_mat_t cov_mat_obs_neighbors, cov_mat_between_neighbors;
			den_mat_t cov_grad_dummy; //not used, just as mock argument for functions below
			den_mat_t coords_i, coords_nn_i;
			for (int j = 0; j < num_gp_total; ++j) {
				if (j == 0) {
					if (!distances_saved) {
						std::vector<int> ind{ num_re_cli + i };
						coords_i = coords_all(ind, Eigen::all);
						coords_nn_i = coords_all(nearest_neighbors_cluster_i[i], Eigen::all);
					}
					re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
						cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
					re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
						cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
				}
				else {//random coefficient GPs
					den_mat_t cov_mat_obs_neighbors_j;
					den_mat_t cov_mat_between_neighbors_j;
					re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
						cov_mat_obs_neighbors_j, &cov_grad_dummy, false, true, 1., false);
					re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
						cov_mat_between_neighbors_j, &cov_grad_dummy, false, true, 1., true);
					//multiply by coefficient matrix
					cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();
					cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
					cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
					cov_mat_between_neighbors += cov_mat_between_neighbors_j;
				}
			}//end loop over components j
			//Calculate matrices A and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii})
			for (int j = 0; j < num_gp_total; ++j) {
				double d_comp_j = re_comps[cluster_i][ind_intercept_gp + j]->CovPars()[0];
				if (j > 0) {//random coefficient
					d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
				}
				Dp[i] += d_comp_j;
			}
			//2. remaining terms
			if (gauss_likelihood) {
				cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
				//Note: we add the nugget effect variance irrespective of 'predict_response' since (i) this is numerically more stable and 
				//	(ii) otherwise we would have to add it only for the neighbors in the observed training data if predict_response == false
				//	If predict_response == false, the nugget variance is simply subtracted from the predictive covariance matrix later again.
			}
			den_mat_t A_i(1, num_nn);//dim = 1 x nn
			A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors)).transpose();
			for (int inn = 0; inn < num_nn; ++inn) {
				if (nearest_neighbors_cluster_i[i][inn] < num_re_cli) {//nearest neighbor belongs to observed data
					Bpo.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
				}
				else {
					Bp.coeffRef(i, nearest_neighbors_cluster_i[i][inn] - num_re_cli) -= A_i(0, inn);
				}
			}
			Dp[i] -= (A_i * cov_mat_obs_neighbors)(0, 0);
		}//end loop over data i
		if (gauss_likelihood) {
			pred_mean = -Bpo * y_cluster_i;
			if (!CondObsOnly) {
				sp_L_solve(Bp.valuePtr(), Bp.innerIndexPtr(), Bp.outerIndexPtr(), num_re_pred_cli, pred_mean.data());
			}
			if (calc_pred_cov || calc_pred_var) {
				if (calc_pred_var) {
					pred_var = vec_t(num_re_pred_cli);
				}
				if (CondObsOnly) {
					if (calc_pred_cov) {
						pred_cov = Dp.asDiagonal();
					}
					if (calc_pred_var) {
						pred_var = Dp;
					}
				}
				else {
					sp_mat_t Bp_inv(num_re_pred_cli, num_re_pred_cli);
					Bp_inv.setIdentity();
					TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
					sp_mat_t Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
					if (calc_pred_cov) {
						pred_cov = T_mat(Bp_inv_Dp * Bp_inv.transpose());
					}
					if (calc_pred_var) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_re_pred_cli; ++i) {
							pred_var[i] = (Bp_inv_Dp.row(i)).dot(Bp_inv.row(i));
						}
					}
				}
			}//end calc_pred_cov || calc_pred_var
			//release matrices that are not needed anymore
			Bpo.resize(0, 0);
			Bp.resize(0, 0);
			Dp.resize(0);
		}//end if gauss_likelihood
	}//end CalcPredVecchiaObservedFirstOrder

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
	* \param re_comps Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	* \param num_gp_rand_coef Number of random coefficient GPs
	* \param num_gp_total Total number of GPs (random intercepts plus random coefficients)
	* \param y_cluster_i Reponse variable data
	* \param rng Random number generator
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param[out] pred_mean Predictive mean
	* \param[out] pred_cov Predictive covariance matrix
	* \param[out] pred_var Predictive variances
	*/
	template<typename T_mat>
	void CalcPredVecchiaPredictedFirstOrder(data_size_t cluster_i,
		int num_data_pred,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster_pred,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		const double* gp_rand_coef_data_pred,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompBase<T_mat>>>>& re_comps,
		int ind_intercept_gp,
		int num_gp_rand_coef,
		int num_gp_total,
		const vec_t& y_cluster_i,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var) {
		int num_data_cli = (int)gp_coords_mat_obs.rows();
		int num_data_pred_cli = (int)gp_coords_mat_pred.rows();
		int num_data_tot = num_data_cli + num_data_pred_cli;
		//Find nearest neighbors
		den_mat_t coords_all(num_data_tot, gp_coords_mat_obs.cols());
		coords_all << gp_coords_mat_pred, gp_coords_mat_obs;
		std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_data_tot);
		std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_data_tot);
		std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_data_tot);
		bool check_has_duplicates = false;
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps[cluster_i][ind_intercept_gp]);
		bool distances_saved = re_comp->ShouldSaveDistances();
		den_mat_t coords_scaled;
		if (distances_saved) {
			find_nearest_neighbors_Vecchia_fast(coords_all, num_data_tot, num_neighbors_pred,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
				vecchia_neighbor_selection, rng, distances_saved);
		}
		else {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all, coords_scaled);
			find_nearest_neighbors_Vecchia_fast(coords_scaled, num_data_tot, num_neighbors_pred,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
				vecchia_neighbor_selection, rng, distances_saved);
		}

		//Prepare data for random coefficients
		std::vector<std::vector<den_mat_t>> z_outer_z_obs_neighbors_cluster_i(num_data_tot);
		if (num_gp_rand_coef > 0) {
			for (int j = 0; j < num_gp_rand_coef; ++j) {
				std::vector<double> rand_coef_data(num_data_tot);//First entries are the predicted data, then the observed data
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_pred_cli; ++i) {
					rand_coef_data[i] = gp_rand_coef_data_pred[j * num_data_pred + data_indices_per_cluster_pred[cluster_i][i]];
				}
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_cli; ++i) {
					rand_coef_data[num_data_pred_cli + i] = re_comps[cluster_i][ind_intercept_gp + j + 1]->RandCoefData()[i];
				}
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_tot; ++i) {
					if (j == 0) {
						z_outer_z_obs_neighbors_cluster_i[i] = std::vector<den_mat_t>(num_gp_rand_coef);
					}
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
		vec_t Do_inv(num_data_cli);
		vec_t Dp_inv(num_data_pred_cli);
		Do_inv.setOnes();//Put 1 on the diagonal (for nugget effect)
		Dp_inv.setOnes();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data_tot; ++i) {
			int num_nn = (int)nearest_neighbors_cluster_i[i].size();
			//define covariance and gradient matrices
			den_mat_t cov_mat_obs_neighbors, cov_mat_between_neighbors;
			den_mat_t cov_grad_dummy; //not used, just as mock argument for functions below
			den_mat_t coords_i, coords_nn_i;
			if (i > 0) {
				for (int j = 0; j < num_gp_total; ++j) {
					if (j == 0) {
						if (!distances_saved) {
							std::vector<int> ind{ i };
							coords_i = coords_all(ind, Eigen::all);
							coords_nn_i = coords_all(nearest_neighbors_cluster_i[i], Eigen::all);
						}
						re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
						re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors_j, &cov_grad_dummy, false, true, 1., false);
						re_comps[cluster_i][ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors_j, &cov_grad_dummy, false, true, 1., true);
						//multiply by coefficient matrix
						cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();
						cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
						cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
						cov_mat_between_neighbors += cov_mat_between_neighbors_j;
					}
				}//end loop over components j
			}
			//Calculate matrices A and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii})
			for (int j = 0; j < num_gp_total; ++j) {
				double d_comp_j = re_comps[cluster_i][ind_intercept_gp + j]->CovPars()[0];
				if (j > 0) {//random coefficient
					d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
				}
				if (i < num_data_pred_cli) {
					Dp_inv[i] += d_comp_j;
				}
				else {
					Do_inv[i - num_data_pred_cli] += d_comp_j;
				}
			}
			//2. remaining terms
			if (i > 0) {
				cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
				den_mat_t A_i(1, num_nn);//dim = 1 x nn
				A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors)).transpose();
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
					Dp_inv[i] -= (A_i * cov_mat_obs_neighbors)(0, 0);
				}
				else {
					Do_inv[i - num_data_pred_cli] -= (A_i * cov_mat_obs_neighbors)(0, 0);
				}
			}
			if (i < num_data_pred_cli) {
				Dp_inv[i] = 1 / Dp_inv[i];
			}
			else {
				Do_inv[i - num_data_pred_cli] = 1 / Do_inv[i - num_data_pred_cli];
			}
		}//end loop over data i
		sp_mat_t cond_prec = Bp.transpose() * Dp_inv.asDiagonal() * Bp + Bop.transpose() * Do_inv.asDiagonal() * Bop;
		chol_sp_mat_t CholFact;
		CholFact.compute(cond_prec);
		vec_t y_aux = Bop.transpose() * (Do_inv.asDiagonal() * (Bo * y_cluster_i));
		pred_mean = -CholFact.solve(y_aux);
		if (calc_pred_cov || calc_pred_var) {
			sp_mat_t cond_prec_chol_inv(num_data_pred_cli, num_data_pred_cli);
			cond_prec_chol_inv.setIdentity();
			TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(CholFact.CholFactMatrix(), cond_prec_chol_inv, cond_prec_chol_inv, false);
			if (calc_pred_cov) {
				pred_cov = T_mat(cond_prec_chol_inv.transpose() * cond_prec_chol_inv);
			}
			if (calc_pred_var) {
				pred_var = vec_t(num_data_pred_cli);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_pred_cli; ++i) {
					pred_var[i] = (cond_prec_chol_inv.col(i)).dot(cond_prec_chol_inv.col(i));
				}
			}
		}//end calc_pred_cov || calc_pred_var
	}//end CalcPredVecchiaPredictedFirstOrder

	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the Vecchia approximation for the latent process when observed locations appear first in the ordering (only for Gaussian likelihoods)
	* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data
	* \param cluster_i Cluster index for which prediction are made
	* \param gp_coords_mat_obs Coordinates for observed locations
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param num_neighbors_pred The number of neighbors used in the Vecchia approximation for making predictions
	* \param vecchia_neighbor_selection The way how neighbors are selected
	* \param re_comps Keys: labels of independent realizations of REs/GPs, values: vectors with individual RE/GP components
	* \param ind_intercept_gp Index in the vector of random effect components (in the values of 're_comps') of the intercept GP associated with the random coefficient GPs
	* \param y_cluster_i Reponse variable data
	* \param rng Random number generator
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param predict_response If true, the response variable (label) is predicted, otherwise the latent random effects (only has an effect on pred_cov and pred_var)
	* \param[out] pred_mean Predictive mean
	* \param[out] pred_cov Predictive covariance matrix
	* \param[out] pred_var Predictive variances
	 */
	template<typename T_mat>
	void CalcPredVecchiaLatentObservedFirstOrder(bool CondObsOnly,
		data_size_t cluster_i,
		const den_mat_t& gp_coords_mat_obs,
		const den_mat_t& gp_coords_mat_pred,
		int num_neighbors_pred,
		const string_t& vecchia_neighbor_selection,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompBase<T_mat>>>>& re_comps,
		int ind_intercept_gp,
		const vec_t& y_cluster_i,
		RNG_t& rng,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool predict_response,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var) {
		int num_data_cli = (int)gp_coords_mat_obs.rows();
		CHECK(num_data_cli == re_comps[cluster_i][ind_intercept_gp]->GetNumUniqueREs());
		int num_data_pred_cli = (int)gp_coords_mat_pred.rows();
		int num_data_tot = num_data_cli + num_data_pred_cli;
		//Find nearest neighbors
		den_mat_t coords_all(num_data_cli + num_data_pred_cli, gp_coords_mat_obs.cols());
		coords_all << gp_coords_mat_obs, gp_coords_mat_pred;
		//Determine number of unique observartion locations
		std::vector<int> uniques;//unique points
		std::vector<int> unique_idx;//used for constructing incidence matrix Z_ if there are duplicates
		DetermineUniqueDuplicateCoordsFast(gp_coords_mat_obs, num_data_cli, uniques, unique_idx);
		int num_coord_unique_obs = (int)uniques.size();
		//Determine unique locations (observed and predicted)
		DetermineUniqueDuplicateCoordsFast(coords_all, num_data_tot, uniques, unique_idx);
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
		std::vector<Triplet_t> entries_Z_o, entries_Z_p;
		for (int i = 0; i < num_data_tot; ++i) {
			if (i < num_data_cli) {
				entries_Z_o.push_back(Triplet_t(i, unique_idx[i], 1.));
			}
			else {
				entries_Z_p.push_back(Triplet_t(i - num_data_cli, unique_idx[i], 1.));
			}
		}
		Z_o.setFromTriplets(entries_Z_o.begin(), entries_Z_o.end());
		Z_p.setFromTriplets(entries_Z_p.begin(), entries_Z_p.end());
		std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_coord_unique);
		std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_coord_unique);
		std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_coord_unique);
		bool check_has_duplicates = true;
		std::shared_ptr<RECompGP<T_mat>> re_comp = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps[cluster_i][ind_intercept_gp]);
		bool distances_saved = re_comp->ShouldSaveDistances();
		den_mat_t coords_scaled;
		if (!distances_saved) {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all_unique, coords_scaled);
		}
		if (CondObsOnly) {//find neighbors among both the observed locations only
			if (distances_saved) {
				find_nearest_neighbors_Vecchia_fast(coords_all_unique, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, num_coord_unique_obs - 1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			else {
				find_nearest_neighbors_Vecchia_fast(coords_scaled, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, num_coord_unique_obs - 1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
		}
		else {//find neighbors among both the observed and prediction locations
			if (distances_saved) {
				find_nearest_neighbors_Vecchia_fast(coords_all_unique, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			else {
				find_nearest_neighbors_Vecchia_fast(coords_scaled, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
		}
		if (check_has_duplicates) {
			Log::REFatal("Duplicates found among training and test coordinates. "
				"This is not supported for predictions with a Vecchia approximation for the latent process ('latent_') ");
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
		vec_t D(num_coord_unique);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_coord_unique; ++i) {
			int num_nn = (int)nearest_neighbors_cluster_i[i].size();
			//define covariance and gradient matrices
			den_mat_t cov_mat_obs_neighbors, cov_mat_between_neighbors;
			den_mat_t cov_grad_dummy; //not used, just as mock argument for functions below
			den_mat_t coords_i, coords_nn_i;
			if (i > 0) {
				if (!distances_saved) {
					std::vector<int> ind{ i };
					coords_i = coords_all_unique(ind, Eigen::all);
					coords_nn_i = coords_all_unique(nearest_neighbors_cluster_i[i], Eigen::all);
				}
				re_comps[cluster_i][ind_intercept_gp]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
					cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
				re_comps[cluster_i][ind_intercept_gp]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
					cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
			}
			//Calculate matrices A and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii})
			D[i] = re_comps[cluster_i][ind_intercept_gp]->CovPars()[0];
			//2. remaining terms
			if (i > 0) {
				den_mat_t A_i(1, num_nn);//dim = 1 x nn
				A_i = (cov_mat_between_neighbors.llt().solve(cov_mat_obs_neighbors)).transpose();
				for (int inn = 0; inn < num_nn; ++inn) {
					B.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) -= A_i(0, inn);
				}
				D[i] -= (A_i * cov_mat_obs_neighbors)(0, 0);
			}
		}//end loop over data i
		//Calculate D_inv and B_inv in order to calcualte Sigma and Sigma^-1
		vec_t D_inv = D.cwiseInverse();
		sp_mat_t B_inv(num_coord_unique, num_coord_unique);
		B_inv.setIdentity();
		TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(B, B_inv, B_inv, false);
		//Calculate inverse of covariance matrix for observed data using the Woodbury identity
		sp_mat_t M_aux_Woodbury = B.transpose() * D_inv.asDiagonal() * B + Z_o.transpose() * Z_o;
		chol_sp_mat_t CholFac_M_aux_Woodbury;
		CholFac_M_aux_Woodbury.compute(M_aux_Woodbury);
		if (calc_pred_cov || calc_pred_var) {
			sp_mat_t Identity_obs(num_data_cli, num_data_cli);
			Identity_obs.setIdentity();
			sp_mat_t MInvSqrtX_Z_o_T;
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(CholFac_M_aux_Woodbury, Z_o.transpose(), MInvSqrtX_Z_o_T, false);
			sp_mat_t ZoSigmaZoT_plusI_Inv = -MInvSqrtX_Z_o_T.transpose() * MInvSqrtX_Z_o_T + Identity_obs;
			sp_mat_t Z_p_B_inv = Z_p * B_inv;
			sp_mat_t Z_p_B_inv_D = Z_p_B_inv * D.asDiagonal();
			sp_mat_t ZpSigmaZoT = Z_p_B_inv_D * (B_inv.transpose() * Z_o.transpose());
			sp_mat_t M_aux = ZpSigmaZoT * ZoSigmaZoT_plusI_Inv;
			pred_mean = M_aux * y_cluster_i;
			if (calc_pred_cov) {
				pred_cov = T_mat(Z_p_B_inv_D * Z_p_B_inv.transpose() - M_aux * ZpSigmaZoT.transpose());
				if (predict_response) {
					pred_cov.diagonal().array() += 1.;
				}
			}
			if (calc_pred_var) {
				pred_var = vec_t(num_data_pred_cli);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_pred_cli; ++i) {
					pred_var[i] = (Z_p_B_inv_D.row(i)).dot(Z_p_B_inv.row(i)) - (M_aux.row(i)).dot(ZpSigmaZoT.row(i));
				}
				if (predict_response) {
					pred_var.array() += 1.;
				}
			}
		}//end calc_pred_cov || calc_pred_var
		else {
			vec_t resp_aux = Z_o.transpose() * y_cluster_i;
			vec_t resp_aux2 = CholFac_M_aux_Woodbury.solve(resp_aux);
			resp_aux = y_cluster_i - Z_o * resp_aux2;
			pred_mean = Z_p * (B_inv * (D.asDiagonal() * (B_inv.transpose() * (Z_o.transpose() * resp_aux))));
		}
	}//end CalcPredVecchiaLatentObservedFirstOrder

}  // namespace GPBoost

#endif   // GPB_VECCHIA_H_
