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
		RNG_t& gen) {
		CHECK((int)neighbors.size() == (num_data - start_at));
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
				dist_obs_neighbors[i - start_at].resize(1, i);
				for (int j = 0; j < i; ++j) {
					neighbors[i - start_at][j] = j;
					dist_obs_neighbors[i - start_at](0, j) = (coords(j, Eigen::all) - coords(i, Eigen::all)).lpNorm<2>();
					if (check_has_duplicates) {
						if (!has_duplicates) {
							if (dist_obs_neighbors[i - start_at](0, j) < EPSILON_NUMBERS) {
								has_duplicates = true;
							}
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
				dist_obs_neighbors[i - start_at].resize(1, num_neighbors);
				for (int j = 0; j < num_nearest_neighbors; ++j) {
					dist_obs_neighbors[i - start_at](0, j) = std::sqrt(nn_square_dist[j]);
					if (check_has_duplicates) {
						if (!has_duplicates) {
							if (dist_obs_neighbors[i - start_at](0, j) < EPSILON_NUMBERS) {
#pragma omp critical
								{
									has_duplicates = true;
								}
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
						dist_obs_neighbors[i - start_at](0, num_nearest_neighbors + j) = (coords(neighbors[i - start_at][num_nearest_neighbors + j], Eigen::all) - coords(i, Eigen::all)).norm();;
						if (check_has_duplicates) {
							if (!has_duplicates) {
								if (dist_obs_neighbors[i - start_at](0, num_nearest_neighbors + j) < EPSILON_NUMBERS) {
#pragma omp critical
									{
										has_duplicates = true;
									}
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
			dist_between_neighbors[i - start_at].resize(nn_i, nn_i);
			for (int j = 0; j < nn_i; ++j) {
				dist_between_neighbors[i - start_at](j, j) = 0.;
				for (int k = j + 1; k < nn_i; ++k) {
					dist_between_neighbors[i - start_at](j, k) = (coords(neighbors[i - start_at][j], Eigen::all) - coords(neighbors[i - start_at][k], Eigen::all)).lpNorm<2>();
					if (check_has_duplicates) {
						if (!has_duplicates){
							if (dist_between_neighbors[i - start_at](j, k) < EPSILON_NUMBERS) {
#pragma omp critical
								{
									has_duplicates = true;
								}
							}
						}
					}//end check_has_duplicates
				}
			}
			dist_between_neighbors[i - start_at].triangularView<Eigen::StrictlyLower>() = dist_between_neighbors[i - start_at].triangularView<Eigen::StrictlyUpper>().transpose();
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

}  // namespace GPBoost
