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
#include <LightGBM/utils/log.h>
using LightGBM::Log;

namespace GPBoost {

	void find_nearest_neighbors_Vecchia(den_mat_t& dist, 
		int num_data,
		int num_neighbors,
		std::vector<std::vector<int>>& nearest_neighbors) {
		CHECK((int)nearest_neighbors.size() == num_data);
		CHECK((int)dist.rows() == num_data && (int)dist.cols() == num_data);
		for (int i = 0; i < num_data; ++i) {
			if (i > 0 && i <= num_neighbors) {
				nearest_neighbors[i].resize(i);
				for (int j = 0; j < i; ++j) {
					nearest_neighbors[i][j] = j;
				}
			}
			else if (i > num_neighbors) {
				nearest_neighbors[i].resize(num_neighbors);
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
						nearest_neighbors[i][num_neighbors - 1] = j;
						SortVectorsDecreasing<double>(nn_dist.data(), nearest_neighbors[i].data(), num_neighbors);
					}
				}
			}
		}
	}//end find_nearest_neighbors_Vecchia

	void find_nearest_neighbors_Vecchia_fast(const den_mat_t& coords, 
		int num_data, 
		int num_neighbors,
		std::vector<std::vector<int>>& nearest_neighbors, 
		std::vector<den_mat_t>& dist_obs_neighbors,
		std::vector<den_mat_t>& dist_between_neighbors, 
		int start_at, 
		int end_search_at, 
		bool& check_has_duplicates) {
		bool has_duplicates = false;
		if (end_search_at < 0) {
			end_search_at = num_data - 2;
		}
		CHECK((int)nearest_neighbors.size() == (num_data - start_at));
		CHECK((int)coords.rows() == num_data);
		if (num_neighbors > end_search_at + 1) {
			Log::REInfo("The number of neighbors (%d) for the Vecchia approximation needs to be smaller than the number of data points (%d). It is set to %d.", num_neighbors, end_search_at + 2, end_search_at + 1);
			num_neighbors = end_search_at + 1;
		}
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
		//Intialize nearest neighbor vectors
		for (int i = start_at; i < num_data; ++i) {
			if (i > 0 && i <= num_neighbors) {
				nearest_neighbors[i - start_at].resize(i);
				dist_obs_neighbors[i - start_at].resize(1, i);
				for (int j = 0; j < i; ++j) {
					nearest_neighbors[i - start_at][j] = j;
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
				nearest_neighbors[i - start_at].resize(num_neighbors);
			}
		}
		//Find nearest neighbor vectors for those points where the conditioning set (=potential nearest neighbors) is larger than 'num_neighbors'
		if (num_data > num_neighbors) {
			int first_i = (start_at <= num_neighbors) ? (num_neighbors + 1) : start_at;//The first pint (first_i) for which the search is done is the point with index (num_neighbors + 1) or start_at
#pragma omp parallel for schedule(static)
			for (int i = first_i; i < num_data; ++i) {
				std::vector<double> nn_square_dist(num_neighbors);
				for (int j = 0; j < num_neighbors; ++j) {
					nn_square_dist[j] = std::numeric_limits<double>::infinity();
				}
				bool down = true;
				bool up = true;
				int up_i = sort_sum[i];
				int down_i = sort_sum[i];
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
						if (sort_inv_sum[down_i] < i && sort_inv_sum[down_i] <= end_search_at) {
							smd = pow(coords_sum[sort_inv_sum[down_i]] - coords_sum[i], 2);
							if (smd > dim_coords * nn_square_dist[num_neighbors - 1]) {
								down = false;
							}
							else {
								sed = (coords(sort_inv_sum[down_i], Eigen::all) - coords(i, Eigen::all)).squaredNorm();
								if (sed < nn_square_dist[num_neighbors - 1]) {
									nn_square_dist[num_neighbors - 1] = sed;
									nearest_neighbors[i - start_at][num_neighbors - 1] = sort_inv_sum[down_i];
									SortVectorsDecreasing<double>(nn_square_dist.data(), nearest_neighbors[i - start_at].data(), num_neighbors);
								}
							}
						}
					}//end down
					if (up) {
						up_i++;
						//counting is done on the sorted scale, but the index on the orignal scale needs to be (i) smaller than 'i' in order to be a neighbor (ii) and also below or equal the largest potential neighbor 'end_search_at'
						if (sort_inv_sum[up_i] < i && sort_inv_sum[up_i] <= end_search_at) {
							smd = pow(coords_sum[sort_inv_sum[up_i]] - coords_sum[i], 2);
							if (smd > dim_coords * nn_square_dist[num_neighbors - 1]) {
								up = false;
							}
							else {
								sed = (coords(sort_inv_sum[up_i], Eigen::all) - coords(i, Eigen::all)).squaredNorm();
								if (sed < nn_square_dist[num_neighbors - 1]) {
									nn_square_dist[num_neighbors - 1] = sed;
									nearest_neighbors[i - start_at][num_neighbors - 1] = sort_inv_sum[up_i];
									SortVectorsDecreasing<double>(nn_square_dist.data(), nearest_neighbors[i - start_at].data(), num_neighbors);
								}
							}
						}
					}//end up
				}
				//Save distances
				for (int j = 0; j < num_neighbors; ++j) {
					dist_obs_neighbors[i - start_at].resize(1, num_neighbors);
					dist_obs_neighbors[i - start_at](0, j) = sqrt(nn_square_dist[j]);
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
			}//end parallel for loop
		}

		int first_i = (start_at == 0) ? 1 : start_at;
#pragma omp parallel for schedule(static)
		for (int i = first_i; i < num_data; ++i) {
			int nn_i = (int)nearest_neighbors[i - start_at].size();
			dist_between_neighbors[i - start_at].resize(nn_i, nn_i);
			for (int j = 0; j < nn_i; ++j) {
				dist_between_neighbors[i - start_at](j, j) = 0.;
				for (int k = j + 1; k < nn_i; ++k) {
					dist_between_neighbors[i - start_at](j, k) = (coords(nearest_neighbors[i - start_at][j], Eigen::all) - coords(nearest_neighbors[i - start_at][k], Eigen::all)).lpNorm<2>();
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

}  // namespace GPBoost
