/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/Vecchia_utils.h>
#include <GPBoost/utils.h>
#include <cmath>
#include <algorithm> // copy
#include <LightGBM/utils/log.h>
using LightGBM::Log;

namespace GPBoost {

	void find_nearest_neighbors_Vecchia(den_mat_t& dist, 
		int num_data,
		int num_neighbors,
		std::vector<std::vector<int>>& neighbors) {
		CHECK((int)neighbors.size() == num_data);
		CHECK((int)dist.rows() == num_data && (int)dist.cols() == num_data);
		for (int i = 0; i < num_data; ++i) {
			if (i > 0 && i <= num_neighbors) {
				neighbors[i].resize(i);
				for (int j = 0; j < i; ++j) {
					neighbors[i][j] = j;
				}
			}
			else if (i > num_neighbors) {
				neighbors[i].resize(num_neighbors);
			}
		}
		if (num_data > num_neighbors) {
#pragma omp parallel for schedule(static)
			for (int i = (num_neighbors + 1); i < num_data; ++i) {
				std::vector<double> nn_dist(num_neighbors);
				for (int j = 0; j < num_neighbors; ++j) {
					nn_dist[j] = std::numeric_limits<double>::infinity();
				}
				for (int j = 0; j < i; ++j) {
					if (dist(i, j) < nn_dist[num_neighbors - 1]) {
						nn_dist[num_neighbors - 1] = dist(i, j);
						neighbors[i][num_neighbors - 1] = j;
						SortVectorsDecreasing<double>(nn_dist.data(), neighbors[i].data(), num_neighbors);
					}
				}
			}
		}
	}//end find_nearest_neighbors_Vecchia

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
		bool save_distances) {
		CHECK((int)neighbors.size() == (num_data - start_at));
		if (save_distances) {
			CHECK((int)dist_obs_neighbors.size() == (num_data - start_at));
			CHECK((int)dist_between_neighbors.size() == (num_data - start_at));
		}
		CHECK((int)coords.rows() == num_data);
		if (end_search_at < 0) {
			end_search_at = num_data - 2;
		}
		if (num_neighbors > end_search_at + 1) {
			Log::REInfo("The number of neighbors (%d) for the Vecchia approximation needs to be smaller than the number of data points (%d). It is set to %d.", num_neighbors, end_search_at + 2, end_search_at + 1);
			num_neighbors = end_search_at + 1;
		}
		int num_nearest_neighbors = num_neighbors;
		int num_non_nearest_neighbors = 0;
		int mult_const_half_random_close_neighbors = 10;//amount of neighbors that are considered as candidate non-nearest but still close neighbors
		int num_close_neighbors = mult_const_half_random_close_neighbors * num_neighbors;
		if (neighbor_selection == "half_random" || neighbor_selection == "half_random_close_neighbors") {
			num_non_nearest_neighbors = num_neighbors / 2;
			num_nearest_neighbors = num_neighbors - num_non_nearest_neighbors;
			CHECK(num_non_nearest_neighbors > 0);
		}
		else if (neighbor_selection != "nearest") {
			Log::REFatal("find_nearest_neighbors_Vecchia_fast: neighbor_selection = '%s' is not supported ", neighbor_selection.c_str());
		}
		bool has_duplicates = false;
		int dim_coords = (int)coords.cols();
		//Sort along the sum of the coordinates
		std::vector<double> coords_sum(num_data);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			coords_sum[i] = coords(i, Eigen::all).sum();
		}
		std::vector<int> sort_sum(num_data);
		SortIndeces<double>(coords_sum, sort_sum);
		std::vector<int> sort_inv_sum(num_data);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			sort_inv_sum[sort_sum[i]] = i;
		}
		//Intialize neighbor vectors
		for (int i = start_at; i < num_data; ++i) {
			if (i > 0 && i <= num_neighbors) {
				neighbors[i - start_at].resize(i);
				if (save_distances) {
					dist_obs_neighbors[i - start_at].resize(i, 1);
				}
				for (int j = 0; j < i; ++j) {
					neighbors[i - start_at][j] = j;
					double dij = 0.;
					if (save_distances || (check_has_duplicates && !has_duplicates)) {
						dij = (coords(j, Eigen::all) - coords(i, Eigen::all)).lpNorm<2>();
					}
					if (save_distances) {
						dist_obs_neighbors[i - start_at](j, 0) = dij;
					}
					if (check_has_duplicates && !has_duplicates) {
						if (dij < EPSILON_NUMBERS) {
							has_duplicates = true;
						}
					}//end check_has_duplicates
				}
			}
			else if (i > num_neighbors) {
				neighbors[i - start_at].resize(num_neighbors);
			}
		}
		//Find neighbors for those points where the conditioning set (=candidate neighbors) is larger than 'num_neighbors'
		if (num_data > num_neighbors) {
			int first_i = (start_at <= num_neighbors) ? (num_neighbors + 1) : start_at;//The first point (first_i) for which the search is done is the point with index (num_neighbors + 1) or start_at
#pragma omp parallel for schedule(static)
			for (int i = first_i; i < num_data; ++i) {
				int num_cand_neighbors = std::min<int>({ i, end_search_at + 1 });
				std::vector<int> neighbors_i;
				std::vector<double> nn_square_dist;
				if (neighbor_selection == "half_random_close_neighbors" && num_cand_neighbors > num_close_neighbors) {
					neighbors_i.resize(num_close_neighbors);
					find_nearest_neighbors_fast_internal(i, num_data, num_close_neighbors, end_search_at,
						dim_coords, coords, sort_sum, sort_inv_sum, coords_sum, neighbors_i, nn_square_dist);
					std::copy(neighbors_i.begin(), neighbors_i.begin() + num_nearest_neighbors, neighbors[i - start_at].begin());
				}
				else {
					find_nearest_neighbors_fast_internal(i, num_data, num_nearest_neighbors, end_search_at,
						dim_coords, coords, sort_sum, sort_inv_sum, coords_sum, neighbors[i - start_at], nn_square_dist);
				}
				//Save distances between points and neighbors
				if (save_distances) {
					dist_obs_neighbors[i - start_at].resize(num_neighbors, 1);
				}
				for (int j = 0; j < num_nearest_neighbors; ++j) {
					double dij = std::sqrt(nn_square_dist[j]);
					if (save_distances) {
						dist_obs_neighbors[i - start_at](j, 0) = dij;
					}
					if (check_has_duplicates && !has_duplicates) {
						if (dij < EPSILON_NUMBERS) {
#pragma omp critical
							{
								has_duplicates = true;
							}
						}
					}//end check_has_duplicates
				}
				//Find non-nearest neighbors
				if (neighbor_selection == "half_random" || neighbor_selection == "half_random_close_neighbors") {
					if (neighbor_selection == "half_random" ||
						(neighbor_selection == "half_random_close_neighbors" && num_cand_neighbors <= num_close_neighbors)) {
						std::vector<int> nearest_neighbors(neighbors[i - start_at].begin(), neighbors[i - start_at].begin() + num_nearest_neighbors);
						std::vector<int> non_nearest_neighbors;
						SampleIntNoReplaceExcludeSomeIndices(num_cand_neighbors, num_non_nearest_neighbors, gen, non_nearest_neighbors, nearest_neighbors);
						std::copy(non_nearest_neighbors.begin(), non_nearest_neighbors.end(), neighbors[i - start_at].begin() + num_nearest_neighbors);
					}
					else if (neighbor_selection == "half_random_close_neighbors" && num_cand_neighbors > num_close_neighbors){
						std::vector<int> ind_non_nearest_neighbors;
						SampleIntNoReplace(num_close_neighbors - num_nearest_neighbors, num_non_nearest_neighbors, gen, ind_non_nearest_neighbors);
						for (int j = 0; j < num_non_nearest_neighbors; ++j) {
							neighbors[i - start_at][num_nearest_neighbors + j] = neighbors_i[ind_non_nearest_neighbors[j] + num_nearest_neighbors];
						}
					}
					//Calculate distances between points and neighbors
					for (int j = 0; j < num_non_nearest_neighbors; ++j) {
						double dij = 0.;
						if (save_distances || (check_has_duplicates && !has_duplicates)) {
							dij = (coords(neighbors[i - start_at][num_nearest_neighbors + j], Eigen::all) - coords(i, Eigen::all)).norm();
						}
						if (save_distances) {
							dist_obs_neighbors[i - start_at](num_nearest_neighbors + j, 0) = dij;
						}
						if (check_has_duplicates && !has_duplicates) {
							if (dij < EPSILON_NUMBERS) {
#pragma omp critical
								{
									has_duplicates = true;
								}
							}
						}//end check_has_duplicates
					}
				}//end selection of non-nearest neighbors
			}//end parallel for loop for finding neighbors
		}
		// Calculate distances among neighbors
		int first_i = (start_at == 0) ? 1 : start_at;
#pragma omp parallel for schedule(static)
		for (int i = first_i; i < num_data; ++i) {
			int nn_i = (int)neighbors[i - start_at].size();
			if (save_distances) {
				dist_between_neighbors[i - start_at].resize(nn_i, nn_i);
			}
			for (int j = 0; j < nn_i; ++j) {
				if (save_distances) {
					dist_between_neighbors[i - start_at](j, j) = 0.;
				}
				for (int k = j + 1; k < nn_i; ++k) {
					double dij = 0.;
					if (save_distances || (check_has_duplicates && !has_duplicates)) {
						dij = (coords(neighbors[i - start_at][j], Eigen::all) - coords(neighbors[i - start_at][k], Eigen::all)).lpNorm<2>();
					}
					if (save_distances) {
						dist_between_neighbors[i - start_at](j, k) = dij;
					}
					if (check_has_duplicates && !has_duplicates) {
						if (dij < EPSILON_NUMBERS) {
#pragma omp critical
							{
								has_duplicates = true;
							}
						}
					}//end check_has_duplicates
				}
			}
			if (save_distances) {
				dist_between_neighbors[i - start_at].triangularView<Eigen::StrictlyLower>() = dist_between_neighbors[i - start_at].triangularView<Eigen::StrictlyUpper>().transpose();
			}
		}
		if (check_has_duplicates) {
			check_has_duplicates = has_duplicates;
		}
	}//end find_nearest_neighbors_Vecchia_fast

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
		std::vector<double>& nn_square_dist) {
		nn_square_dist = std::vector<double>(num_nearest_neighbors);
		for (int j = 0; j < num_nearest_neighbors; ++j) {
			nn_square_dist[j] = std::numeric_limits<double>::infinity();
		}
		bool down = true;
		bool up = true;
		int up_i = sort_inv_sum[i];
		int down_i = sort_inv_sum[i];
		double smd, sed;
		while (up || down) {
			if (down_i == 0) {
				down = false;
			}
			if (up_i == (num_data - 1)) {
				up = false;
			}
			if (down) {
				down_i--;
				//counting is done on the sorted scale, but the index on the orignal scale needs to be (i) smaller than 'i' in order to be a neighbor (ii) and also below or equal the largest potential neighbor 'end_search_at'
				if (sort_sum[down_i] < i && sort_sum[down_i] <= end_search_at) {
					smd = std::pow(coords_sum[sort_sum[down_i]] - coords_sum[i], 2);
					if (smd > dim_coords * nn_square_dist[num_nearest_neighbors - 1]) {
						down = false;
					}
					else {
						sed = (coords(sort_sum[down_i], Eigen::all) - coords(i, Eigen::all)).squaredNorm();
						if (sed < nn_square_dist[num_nearest_neighbors - 1]) {
							nn_square_dist[num_nearest_neighbors - 1] = sed;
							neighbors_i[num_nearest_neighbors - 1] = sort_sum[down_i];
							SortVectorsDecreasing<double>(nn_square_dist.data(), neighbors_i.data(), num_nearest_neighbors);
						}
					}
				}
			}//end down
			if (up) {
				up_i++;
				//counting is done on the sorted scale, but the index on the orignal scale needs to be (i) smaller than 'i' in order to be a neighbor (ii) and also below or equal the largest potential neighbor 'end_search_at'
				if (sort_sum[up_i] < i && sort_sum[up_i] <= end_search_at) {
					smd = std::pow(coords_sum[sort_sum[up_i]] - coords_sum[i], 2);
					if (smd > dim_coords * nn_square_dist[num_nearest_neighbors - 1]) {
						up = false;
					}
					else {
						sed = (coords(sort_sum[up_i], Eigen::all) - coords(i, Eigen::all)).squaredNorm();
						if (sed < nn_square_dist[num_nearest_neighbors - 1]) {
							nn_square_dist[num_nearest_neighbors - 1] = sed;
							neighbors_i[num_nearest_neighbors - 1] = sort_sum[up_i];
							SortVectorsDecreasing<double>(nn_square_dist.data(), neighbors_i.data(), num_nearest_neighbors);
						}
					}
				}
			}//end up
		}//end while (up || down)
	}//end find_nearest_neighbors_fast_internal

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
		bool save_distances_isotropic_cov_fct) {
		int ind_intercept_gp = (int)re_comps_vecchia_cluster_i.size();
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
		re_comps_vecchia_cluster_i.push_back(std::shared_ptr<RECompGP<den_mat_t>>(new RECompGP<den_mat_t>(
			gp_coords_mat, cov_fct, cov_fct_shape, cov_fct_taper_range, cov_fct_taper_shape, apply_tapering,
			false, false, only_one_GP_calculations_on_RE_scale, only_one_GP_calculations_on_RE_scale, save_distances_isotropic_cov_fct)));
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = re_comps_vecchia_cluster_i[ind_intercept_gp];
		if ((vecchia_ordering == "time" || vecchia_ordering == "time_random_space") && !(re_comp->IsSpaceTimeModel())) {
			Log::REFatal("'vecchia_ordering' is '%s' but the 'cov_function' is not a space-time covariance function ", vecchia_ordering.c_str());
		}
		if (re_comp->GetNumUniqueREs() == num_data_per_cluster[cluster_i]) {
			only_one_GP_calculations_on_RE_scale = false;
		}
		bool has_duplicates = check_has_duplicates;
		nearest_neighbors_cluster_i = std::vector<std::vector<int>>(re_comp->GetNumUniqueREs());
		dist_obs_neighbors_cluster_i = std::vector<den_mat_t>(re_comp->GetNumUniqueREs());
		dist_between_neighbors_cluster_i = std::vector<den_mat_t>(re_comp->GetNumUniqueREs());
		if (re_comp->HasIsotropicCovFct()) {
			Log::REInfo("Starting nearest neighbor search for Vecchia approximation");
			find_nearest_neighbors_Vecchia_fast(re_comp->GetCoords(), re_comp->GetNumUniqueREs(), num_neighbors,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, has_duplicates,
				vecchia_neighbor_selection, rng, save_distances_isotropic_cov_fct);
			Log::REInfo("Nearest neighbors for Vecchia approximation found");
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
				}
				entries_init_B_cluster_i.push_back(Triplet_t(i, i, 1.));//Put 1's on the diagonal since B = I - A
			}
		}
		//Random coefficients
		if (num_gp_rand_coef > 0) {
			if (!(re_comp->HasIsotropicCovFct())) {
				Log::REFatal("Random coefficient processes are not supported for covariance functions "
					"for which the neighbors are dynamically determined based on correlations ");
			}
			z_outer_z_obs_neighbors_cluster_i = std::vector<std::vector<den_mat_t>>(re_comp->GetNumUniqueREs());
			for (int j = 0; j < num_gp_rand_coef; ++j) {
				std::vector<double> rand_coef_data;
				for (const auto& id : data_indices_per_cluster[cluster_i]) {
					rand_coef_data.push_back(gp_rand_coef_data[j * num_data + id]);
				}
				re_comps_vecchia_cluster_i.push_back(std::shared_ptr<RECompGP<den_mat_t>>(new RECompGP<den_mat_t>(
					rand_coef_data, cov_fct, cov_fct_shape, cov_fct_taper_range, cov_fct_taper_shape, re_comp->GetTaperMu(),
					apply_tapering, false, dim_gp_coords, save_distances_isotropic_cov_fct)));
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

	void UpdateNearestNeighbors(std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		std::vector<Triplet_t>& entries_init_B_cluster_i,
		int num_neighbors,
		const string_t& vecchia_neighbor_selection,
		RNG_t& rng,
		int ind_intercept_gp,
		bool& has_duplicates_coords,
		bool check_has_duplicates,
		bool gauss_likelihood) {
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = re_comps_vecchia_cluster_i[ind_intercept_gp];
		CHECK(re_comp->HasIsotropicCovFct() == false);
		int num_re = re_comp->GetNumUniqueREs();
		CHECK((int)nearest_neighbors_cluster_i.size() == num_re);
		// Calculate scaled coordinates
		den_mat_t coords_scaled;
		re_comp->GetScaledCoordinates(coords_scaled);
		// find correlation-based nearest neighbors
		std::vector<den_mat_t> dist_dummy;
		bool has_duplicates = check_has_duplicates;
		find_nearest_neighbors_Vecchia_fast(coords_scaled, num_re, num_neighbors,
			nearest_neighbors_cluster_i, dist_dummy, dist_dummy, 0, -1, has_duplicates,
			vecchia_neighbor_selection, rng, false);
		if (check_has_duplicates) {
			has_duplicates_coords = has_duplicates_coords || has_duplicates;
			if (!gauss_likelihood && has_duplicates_coords) {
				Log::REFatal("Duplicates found in the coordinates for the Gaussian process. "
					"This is currently not supported for the Vecchia approximation for non-Gaussian likelihoods ");
			}
		}
		if (entries_init_B_cluster_i.size() == 0) {
			for (int i = 0; i < re_comp->GetNumUniqueREs(); ++i) {
				for (int j = 0; j < (int)nearest_neighbors_cluster_i[i].size(); ++j) {
					entries_init_B_cluster_i.push_back(Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.));
				}
				entries_init_B_cluster_i.push_back(Triplet_t(i, i, 1.));//Put 1's on the diagonal since B = I - A
			}
		}
		else {
			int ctr = 0, ctr_grad = 0;
			for (int i = 0; i < std::min(num_re, num_neighbors); ++i) {
				for (int j = 0; j < (int)nearest_neighbors_cluster_i[i].size(); ++j) {
					entries_init_B_cluster_i[ctr] = Triplet_t(i, nearest_neighbors_cluster_i[i][j], 0.);
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
					}
					entries_init_B_cluster_i[ctr + (i - num_neighbors) * (num_neighbors + 1) + num_neighbors] = Triplet_t(i, i, 1.);//Put 1's on the diagonal since B = I - A
				}
			}
		}
	}//end UpdateNearestNeighbors

	void CalcCovFactorGradientVecchia(data_size_t num_re_cluster_i,
		bool calc_cov_factor,
		bool calc_gradient,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		const std::vector<std::vector<int>>& nearest_neighbors_cluster_i,
		const std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		const std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		const std::vector<Triplet_t>& entries_init_B_cluster_i,
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
		bool gauss_likelihood,
		bool save_distances_isotropic_cov_fct) {
		int num_par_comp = re_comps_vecchia_cluster_i[ind_intercept_gp]->NumCovPar();
		int num_par_gp = num_par_comp * num_gp_total + calc_gradient_nugget;
		//Initialize matrices B = I - A and D^-1 as well as their derivatives (in order that the code below can be run in parallel)
		if (calc_cov_factor) {
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
		}
		bool exclude_marg_var_grad = !gauss_likelihood && (re_comps_vecchia_cluster_i.size() == 1);//gradient is not needed if there is only one GP for non-Gaussian likelihoods
		if (calc_gradient) {
			B_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of B = derviateive of (-A)
			D_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of D
			for (int ipar = 0; ipar < num_par_gp; ++ipar) {
				if (!(exclude_marg_var_grad && ipar == 0)) {
					B_grad_cluster_i[ipar] = sp_mat_t(num_re_cluster_i, num_re_cluster_i);
					B_grad_cluster_i[ipar].setFromTriplets(entries_init_B_cluster_i.begin(), entries_init_B_cluster_i.end());
					B_grad_cluster_i[ipar].diagonal().array() = 0.;
					D_grad_cluster_i[ipar] = sp_mat_t(num_re_cluster_i, num_re_cluster_i);
					D_grad_cluster_i[ipar].setIdentity();//Put 0 on the diagonal
					D_grad_cluster_i[ipar].diagonal().array() = 0.;
				}
			}
		}//end initialization
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = re_comps_vecchia_cluster_i[ind_intercept_gp];
		bool distances_saved = re_comp->HasIsotropicCovFct() && save_distances_isotropic_cov_fct;
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
						re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, false);//write on matrices directly for first GP component
						re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors, cov_grad_mats_between_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, true);
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors_j, cov_grad_mats_obs_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, false);
						re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors_j, cov_grad_mats_between_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, true);
						//multiply by coefficient matrix
						cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();//cov_mat_obs_neighbors_j.cwiseProduct()
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
				double d_comp_j = re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CovPars()[0];
				if (!transf_scale && gauss_likelihood) {
					d_comp_j *= nugget_var;
				}
				if (j > 0) {//random coefficient
					d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
				}
				if (calc_cov_factor) {
					D_inv_cluster_i.coeffRef(i, i) += d_comp_j;
				}
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
					cov_mat_between_neighbors.diagonal().array() *= JITTER_MULT_VECCHIA;//Avoid numerical problems when there is no nugget effect
				}
				den_mat_t A_i(1, num_nn);
				den_mat_t A_i_grad_sigma2;
				Eigen::LLT<den_mat_t> chol_fact_between_neighbors = cov_mat_between_neighbors.llt();
				A_i = (chol_fact_between_neighbors.solve(cov_mat_obs_neighbors)).transpose();
				if (calc_cov_factor) {
					for (int inn = 0; inn < num_nn; ++inn) {
						B_cluster_i.coeffRef(i, nearest_neighbors_cluster_i[i][inn]) = -A_i(0, inn);
					}
					D_inv_cluster_i.coeffRef(i, i) -= (A_i * cov_mat_obs_neighbors)(0, 0);
				}
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
			if (calc_cov_factor) {
				D_inv_cluster_i.coeffRef(i, i) = 1. / D_inv_cluster_i.coeffRef(i, i);
			}
		}//end loop over data i
		if (calc_cov_factor) {
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
		}
	}//end CalcCovFactorGradientVecchia

	void CalcPredVecchiaObservedFirstOrder(bool CondObsOnly,
		data_size_t cluster_i,
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
		bool save_distances_isotropic_cov_fct) {
		data_size_t num_re_cli = re_comps_vecchia[ind_intercept_gp]->GetNumUniqueREs();
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = re_comps_vecchia[ind_intercept_gp];
		int num_re_pred_cli = (int)gp_coords_mat_pred.rows();
		//Find nearest neighbors
		den_mat_t coords_all(num_re_cli + num_re_pred_cli, gp_coords_mat_obs.cols());
		coords_all << gp_coords_mat_obs, gp_coords_mat_pred;
		std::vector<std::vector<int>> nearest_neighbors_cluster_i(num_re_pred_cli);
		std::vector<den_mat_t> dist_obs_neighbors_cluster_i(num_re_pred_cli);
		std::vector<den_mat_t> dist_between_neighbors_cluster_i(num_re_pred_cli);
		bool check_has_duplicates = false;
		bool distances_saved = re_comp->HasIsotropicCovFct() && save_distances_isotropic_cov_fct;
		bool scale_coordinates = !re_comp->HasIsotropicCovFct();
		den_mat_t coords_scaled;
		if (scale_coordinates) {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all, coords_scaled);
		}
		if (CondObsOnly) {
			if (!scale_coordinates) {
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
			if (!scale_coordinates) {
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
				std::vector<double> rand_coef_data = re_comps_vecchia[ind_intercept_gp + j + 1]->RandCoefData();//First entries are the observed data, then the predicted data
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
					re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
						cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
					re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
						cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
				}
				else {//random coefficient GPs
					den_mat_t cov_mat_obs_neighbors_j;
					den_mat_t cov_mat_between_neighbors_j;
					re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
						cov_mat_obs_neighbors_j, &cov_grad_dummy, false, true, 1., false);
					re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
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
				double d_comp_j = re_comps_vecchia[ind_intercept_gp + j]->CovPars()[0];
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
			else {
				cov_mat_between_neighbors.diagonal().array() *= JITTER_MULT_VECCHIA;//Avoid numerical problems when there is no nugget effect
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
						pred_cov = den_mat_t(Bp_inv_Dp * Bp_inv.transpose());
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
		bool save_distances_isotropic_cov_fct) {
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
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = re_comps_vecchia[ind_intercept_gp];
		bool distances_saved = re_comp->HasIsotropicCovFct() && save_distances_isotropic_cov_fct;
		bool scale_coordinates = !re_comp->HasIsotropicCovFct();
		den_mat_t coords_scaled;
		if (scale_coordinates) {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all, coords_scaled);
		}
		if (!scale_coordinates) {
			find_nearest_neighbors_Vecchia_fast(coords_all, num_data_tot, num_neighbors_pred,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
				vecchia_neighbor_selection, rng, distances_saved);
		}
		else {
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
					rand_coef_data[num_data_pred_cli + i] = re_comps_vecchia[ind_intercept_gp + j + 1]->RandCoefData()[i];
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
						re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
						re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors_j, &cov_grad_dummy, false, true, 1., false);
						re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
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
				double d_comp_j = re_comps_vecchia[ind_intercept_gp + j]->CovPars()[0];
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
				pred_cov = den_mat_t(cond_prec_chol_inv.transpose() * cond_prec_chol_inv);
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
		bool save_distances_isotropic_cov_fct) {
		int num_data_cli = (int)gp_coords_mat_obs.rows();
		CHECK(num_data_cli == re_comps_vecchia[ind_intercept_gp]->GetNumUniqueREs());
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
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = re_comps_vecchia[ind_intercept_gp];
		bool distances_saved = re_comp->HasIsotropicCovFct() && save_distances_isotropic_cov_fct;
		bool scale_coordinates = !re_comp->HasIsotropicCovFct();
		den_mat_t coords_unique_scaled;
		if (scale_coordinates) {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all_unique, coords_unique_scaled);
		}
		if (CondObsOnly) {//find neighbors among both the observed locations only
			if (!scale_coordinates) {
				find_nearest_neighbors_Vecchia_fast(coords_all_unique, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, num_coord_unique_obs - 1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			else {
				find_nearest_neighbors_Vecchia_fast(coords_unique_scaled, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, num_coord_unique_obs - 1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
				coords_unique_scaled.resize(0, 0);
			}
		}
		else {//find neighbors among both the observed and prediction locations
			if (!scale_coordinates) {
				find_nearest_neighbors_Vecchia_fast(coords_all_unique, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
			}
			else {
				find_nearest_neighbors_Vecchia_fast(coords_unique_scaled, num_coord_unique, num_neighbors_pred,
					nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, check_has_duplicates,
					vecchia_neighbor_selection, rng, distances_saved);
				coords_unique_scaled.resize(0, 0);
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
				re_comps_vecchia[ind_intercept_gp]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
					cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false);//write on matrices directly for first GP component
				re_comps_vecchia[ind_intercept_gp]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
					cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true);
			}
			//Calculate matrices A and D as well as their derivatives
			//1. add first summand of matrix D (ZCZ^T_{ii})
			D[i] = re_comps_vecchia[ind_intercept_gp]->CovPars()[0];
			//2. remaining terms
			if (i > 0) {
				den_mat_t A_i(1, num_nn);//dim = 1 x nn
				cov_mat_between_neighbors.diagonal().array() *= JITTER_MULT_VECCHIA;//Avoid numerical problems when there is no nugget effect
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
				pred_cov = den_mat_t(Z_p_B_inv_D * Z_p_B_inv.transpose() - M_aux * ZpSigmaZoT.transpose());
				if (predict_response) {
					pred_cov.diagonal().array() += 1.;
				}
			}
			if (calc_pred_var) {
				pred_var = vec_t(num_data_pred_cli);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data_pred_cli; ++i) {
					vec_t v1 = Z_p_B_inv_D.row(i);
					vec_t v2 = Z_p_B_inv.row(i);
					vec_t v3 = M_aux.row(i);
					vec_t v4 = ZpSigmaZoT.row(i);
					pred_var[i] = v1.dot(v2) - (v3.dot(v4));
				}
				// the following code does not run correctly on some compilers
//#pragma omp parallel for schedule(static)
				//for (int i = 0; i < num_data_pred_cli; ++i) {
				//	pred_var[i] = (Z_p_B_inv_D.row(i)).dot(Z_p_B_inv.row(i)) - (M_aux.row(i)).dot(ZpSigmaZoT.row(i));
				//}
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
