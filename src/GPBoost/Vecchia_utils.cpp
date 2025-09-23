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

	void distances_funct(const int& coord_ind_i,
		const std::vector<int>& coords_ind_j,
		const den_mat_t& coords,
		const vec_t& corr_diag,
		const den_mat_t& chol_ip_cross_cov,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		vec_t& distances,
		string_t dist_function,
		bool distances_saved) {

		if (dist_function == "residual_correlation_FSA") {
			vec_t pp_node(coords_ind_j.size());
			vec_t chol_ip_cross_cov_sample = chol_ip_cross_cov.col(coord_ind_i);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < pp_node.size(); j++) {
				pp_node[j] = chol_ip_cross_cov.col(coords_ind_j[j]).dot(chol_ip_cross_cov_sample);
			}
			den_mat_t corr_mat, coords_i, coords_j;
			std::vector<den_mat_t> dummy_mat_grad;
			coords_i = coords(coord_ind_i, Eigen::all);
			coords_j = coords(coords_ind_j, Eigen::all);
			den_mat_t dist_ij;
			if (distances_saved) {
				dist_ij.resize(coords_ind_j.size(), 1);
#pragma omp parallel for schedule(static)
				for (int j = 0; j < (int)coords_ind_j.size(); j++) {
					dist_ij.coeffRef(j, 0) = (coords_j(j, Eigen::all) - coords_i).lpNorm<2>();
				}
			}
			std::vector<int> calc_grad_index_dummy;
			re_comps_vecchia_cluster_i[0]->CalcSigmaAndSigmaGradVecchia(dist_ij, coords_i, coords_j,
				corr_mat, dummy_mat_grad.data(), false, true, 1., false, calc_grad_index_dummy);
			double corr_diag_sample = corr_diag(coord_ind_i);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < (int)coords_ind_j.size(); j++) {
				distances[j] = std::sqrt((1. - std::abs((corr_mat.data()[j] - pp_node[j]) /
					std::sqrt(corr_diag_sample * corr_diag[coords_ind_j[j]]))));
			}
		}
	}

	void CoverTree_kNN(const den_mat_t& coords_mat,
		const den_mat_t& chol_ip_cross_cov,
		const vec_t& corr_diag,
		const int start,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_vecchia_cluster_i,
		std::map<int, std::vector<int>>& cover_tree,
		int& level,
		bool distances_saved,
		string_t dist_function) {
		den_mat_t coords = coords_mat;
		// Distances already computed
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = std::dynamic_pointer_cast<RECompGP<den_mat_t>>(re_comps_vecchia_cluster_i[0]);
		//Select data point with index 0 as root
		int root = start;
		cover_tree.insert({ -1, { root } });
		//max_dist of root
		double R_max = 1.;
		double R_l;
		// Initialize
		std::vector<int> all_indices(coords.rows() - 1);
		std::iota(std::begin(all_indices), std::end(all_indices), 1);
		std::map<int, std::vector<int>> covert_points_old;
		covert_points_old[0] = all_indices;
		level = 0;
		double base = 2.;
		while ((int)(cover_tree.size() - 1) != (int)(coords.rows())) {
			level += 1;
			R_l = R_max / std::pow(base, level);
			std::map<int, std::vector<int>> covert_points;
			for (const auto& key_covert_points_old_i : covert_points_old) {
				int key = key_covert_points_old_i.first;
				std::vector<int> covert_points_old_i = key_covert_points_old_i.second;
				// sample new node
				bool not_all_covered = covert_points_old_i.size() > 0;
				cover_tree.insert({ key + start, { key + start } });
				while (not_all_covered) {
					int sample_ind = covert_points_old_i[0];
					cover_tree[key + start].push_back(sample_ind + start);
					// new covered points per node
					std::vector<int> covert_points_old_i_up;
					for (int j : covert_points_old_i) {
						if (j > sample_ind) {
							covert_points_old_i_up.push_back(j);
						}
					}
					vec_t dist_vect((int)covert_points_old_i_up.size());
					if ((int)covert_points_old_i_up.size() > 0) {
						distances_funct(sample_ind, covert_points_old_i_up, coords, corr_diag, chol_ip_cross_cov,
							re_comps_vecchia_cluster_i, dist_vect, dist_function, distances_saved);
					}
					for (int j = 0; j < (int)dist_vect.size(); j++) {
						if (dist_vect[j] <= R_l) {
							if (covert_points.find(sample_ind) == covert_points.end()) {
								covert_points.insert({ sample_ind, { covert_points_old_i_up[j] } });
							}
							else {
								covert_points[sample_ind].push_back(covert_points_old_i_up[j]);
							}
						}
					}
					std::vector<int> covert_points_vect = covert_points[sample_ind];
					covert_points_old_i.erase(covert_points_old_i.begin());
					std::vector<int> diff_vect;
					std::set_difference(covert_points_old_i.begin(), covert_points_old_i.end(),
						covert_points_vect.begin(), covert_points_vect.end(),
						std::inserter(diff_vect, diff_vect.begin()));
					covert_points_old_i = diff_vect;
					not_all_covered = covert_points_old_i.size() > 0;
				}
			}
			covert_points_old.clear();
			covert_points_old = covert_points;
		}
	}//end CoverTree_kNN

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
		string_t dist_function) {
		// Initialize distance and help matrices
		// query point
		den_mat_t coords_i = coords(i, Eigen::all);
		vec_t chol_ip_cross_cov_sample = chol_ip_cross_cov.col(i);
		// Initialize vectors
		int root = cover_tree[-1][0];
		std::vector<int> Q;
		std::vector<double> Q_dist;
		std::vector<int> diff_rev = { root };
		// threshold distance
		double max_dist = 1.;
		double dist_k_Q_cor = max_dist;
		bool early_stop = false;
		int k_scaled = k;
		int Q_before_size = 1;
		double base = 2.;
		for (int ii = 1; ii < levels; ii++) {
			// build set of children
			std::vector<int> diff_rev_interim;
			if (ii == 1) {
				Q.push_back(root);
				diff_rev_interim.push_back(root);
			}
			for (int j : diff_rev) {
				for (int jj : cover_tree[j]) {
					if (jj < i) {
						if (jj != j) {
							Q.push_back(jj);
							diff_rev_interim.push_back(jj);
						}
					}
					else {
						break;
					}
				}
			}
			diff_rev.clear();
			early_stop = diff_rev_interim.size() == 0 || ii == (levels - 1);
			if ((int)diff_rev_interim.size() > 0) {
				vec_t dist_vect_interim(diff_rev_interim.size());
				distances_funct(i, diff_rev_interim, coords, corr_diag, chol_ip_cross_cov,
					re_comps_vecchia_cluster_i, dist_vect_interim, dist_function, distances_saved);
				int dist_vect_interim_size = (int)(dist_vect_interim.size());
				for (int j = 0; j < dist_vect_interim_size; j++) {
					Q_dist.push_back(dist_vect_interim[j]);
				}
			}
			// Find k-th smallest element
			if (ii > 1) {
				if ((int)Q_dist.size() < k_scaled) {
					dist_k_Q_cor = *std::max_element(Q_dist.begin(), Q_dist.end());
				}
				else {
					std::vector<double> Q_dist_interim(Q_dist.begin(), Q_dist.end());
					std::nth_element(Q_dist_interim.begin(), Q_dist_interim.begin() + k_scaled - 1, Q_dist_interim.end());
					dist_k_Q_cor = Q_dist_interim[k_scaled - 1];
				}
				dist_k_Q_cor += 1 / std::pow(base, ii - 1);
			}
			int count = 0;
			if (dist_k_Q_cor >= max_dist) {
				if (!early_stop) {
					diff_rev = diff_rev_interim;
					if (ii == 1) {
						diff_rev.erase(diff_rev.begin());
					}
				}
			}
			else {
				std::vector<double> Q_dist_interim;
				std::vector<int> Q_interim;
				auto xi = Q_dist.begin();
				auto yi = Q.begin();
				while (xi != Q_dist.end() && yi != Q.end()) {
					if ((double)*xi <= dist_k_Q_cor) {
						Q_dist_interim.push_back((double)*xi);
						Q_interim.push_back((int)*yi);
						if (count >= Q_before_size) {
							diff_rev.push_back(*yi);
						}
					}
					count += 1;
					++xi;
					++yi;
				}
				Q.clear();
				Q_dist.clear();
				Q = Q_interim;
				Q_dist = Q_dist_interim;
			}
			Q_before_size = (int)Q.size();
			if (early_stop) {
				break;
			}
		}
		std::vector<double> nn_dist(k);
		if (Q_before_size >= k) {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < k; ++j) {
				nn_dist[j] = std::numeric_limits<double>::infinity();
			}
			for (int jj = 0; jj < Q_before_size; ++jj) {
				if (Q_dist[jj] < nn_dist[k - 1]) {
					nn_dist[k - 1] = Q_dist[jj];
					neighbors_i[k - 1] = Q[jj];
					SortVectorsDecreasing<double>(nn_dist.data(), neighbors_i.data(), k);
				}
			}
		}
		else {
			vec_t dist_vect(1);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < k; ++j) {
				nn_dist[j] = std::numeric_limits<double>::infinity();
			}
			for (int jj = 0; jj < i; ++jj) {
				std::vector<int> indj{ jj };
				distances_funct(i, indj, coords, corr_diag, chol_ip_cross_cov,
					re_comps_vecchia_cluster_i, dist_vect, dist_function, distances_saved);
				if (dist_vect[0] < nn_dist[k - 1]) {
					nn_dist[k - 1] = dist_vect[0];
					neighbors_i[k - 1] = jj;
					SortVectorsDecreasing<double>(nn_dist.data(), neighbors_i.data(), k);
				}
			}
		}
		dist_of_neighbors_i = nn_dist;
	}//end find_kNN_CoverTree

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
		const int& num_data_obs) {
		string_t dist_function = "residual_correlation_FSA";
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
		bool has_duplicates = false;
		// For correlation matrix
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = std::dynamic_pointer_cast<RECompGP<den_mat_t>>(re_comps_vecchia_cluster_i[0]);
		std::vector<den_mat_t> dummy_mat_grad;//help matrix
		// Variance for residual process
		vec_t corr_diag(num_data);
		den_mat_t dist_ii(1, 1);
		dist_ii(0, 0) = 0.;
		den_mat_t corr_mat_i;
		den_mat_t coords_ii;
		std::vector<int> indii{ 0 };
		coords_ii = coords(indii, Eigen::all);
		std::vector<int> calc_grad_index_dummy;
		re_comps_vecchia_cluster_i[0]->CalcSigmaAndSigmaGradVecchia(dist_ii, coords_ii, coords_ii,
			corr_mat_i, dummy_mat_grad.data(), false, true, 1., false, calc_grad_index_dummy);
		corr_diag.array() = (double)corr_mat_i.value();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < (int)chol_ip_cross_cov.cols(); ++i) {
			corr_diag[i] -= (double)chol_ip_cross_cov.col(i).array().square().sum();
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
					den_mat_t dist_ij(1, 1);
					dist_ij(0, 0) = 0.;
					if (save_distances || (check_has_duplicates && !has_duplicates)) {
						dist_ij(0, 0) = (coords(j, Eigen::all) - coords(i, Eigen::all)).lpNorm<2>();
					}
					if (save_distances) {
						dist_obs_neighbors[i - start_at](j, 0) = dist_ij.value();
					}
					if (check_has_duplicates && !has_duplicates) {
						if (dist_ij.value() < EPSILON_NUMBERS) {
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
			// Brute force kNN search until certain number of data points
			int brute_force_threshold = std::min(num_data, std::max(1000, num_neighbors));
			if (prediction) {
				brute_force_threshold = std::min(num_data, std::max(first_i + 500, num_neighbors));
			}
			int max_ind_nn = num_data_obs;
			if (cond_on_all) {
				max_ind_nn = num_data;
			}
#pragma omp parallel for schedule(static)
			for (int i = first_i; i < brute_force_threshold; ++i) {
				vec_t dist_vect(1);
				std::vector<double> nn_corr(num_neighbors);
#pragma omp parallel for schedule(static)
				for (int j = 0; j < num_neighbors; ++j) {
					nn_corr[j] = std::numeric_limits<double>::infinity();
				}
				for (int jj = 0; jj < (int)std::min(i, max_ind_nn); ++jj) {
					std::vector<int> indj{ jj };
					distances_funct(i, indj, coords, corr_diag, chol_ip_cross_cov,
						re_comps_vecchia_cluster_i, dist_vect, dist_function, save_distances);
					if (dist_vect[0] < nn_corr[num_neighbors - 1]) {
						nn_corr[num_neighbors - 1] = dist_vect[0];
						neighbors[i - start_at][num_neighbors - 1] = jj;
						SortVectorsDecreasing<double>(nn_corr.data(), neighbors[i - start_at].data(), num_neighbors);
					}
				}
				//Save distances between points and neighbors
				if (save_distances) {
					dist_obs_neighbors[i - start_at].resize(num_neighbors, 1);
				}
				for (int jjj = 0; jjj < num_nearest_neighbors; ++jjj) {
					double dij = (coords(i, Eigen::all) - coords(neighbors[i - start_at][jjj], Eigen::all)).lpNorm<2>();
					if (save_distances) {
						dist_obs_neighbors[i - start_at](jjj, 0) = dij;
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
			if (brute_force_threshold < num_data) {
				int level = 0;
				// Build CoverTree
				std::map<int, std::vector<int>> cover_tree;
				std::map<int, std::map<int, std::vector<int>>> cover_trees;
				std::vector<double> dist_dummy;
				// Set number of threads to the maximum available
				int num_threads;
#ifdef _OPENMP
				num_threads = omp_get_max_threads();
#else
				num_threads = 1;
#endif
				std::vector<int> levels_threads(num_threads);
				std::vector<int> segment_start(num_threads);
				std::vector<int> segment_length(num_threads);
				den_mat_t coords_ct;
				if (prediction && !cond_on_all) {
					coords_ct = coords.topRows(num_data_obs);
				}
				else {
					coords_ct = coords;
				}
				for (int i = 0; i < num_threads; ++i) {
					cover_trees[i] = cover_tree;
				}
				if (num_threads != 1) {
					int segment_size = (int)(std::ceil((double)coords_ct.rows() / (double)num_threads));
					if (segment_size < std::max(1000, num_neighbors)) {
						num_threads = (int)(std::floor((double)coords_ct.rows() / (double)std::max(1000, num_neighbors)));
						segment_size = (int)(std::ceil((double)coords_ct.rows() / (double)num_threads));
					}
					int last_segment = (int)(coords_ct.rows()) - (num_threads - 1) * segment_size;
					bool overhead = false;
					if (last_segment != segment_size) {
						num_threads -= 1;
						levels_threads.resize(num_threads);
						segment_start.resize(num_threads);
						segment_length.resize(num_threads);
						overhead = true;
					}
#pragma omp parallel for
					for (int i = 0; i < num_threads; ++i) {
						segment_start[i] = i * segment_size;
						segment_length[i] = segment_size;
						if (i == num_threads - 1 && overhead) {
							segment_length[i] += last_segment;
						}
						CoverTree_kNN(coords_ct.middleRows(segment_start[i], segment_length[i]), chol_ip_cross_cov.middleCols(segment_start[i], segment_length[i]),
							corr_diag.segment(segment_start[i], segment_length[i]), segment_start[i], re_comps_vecchia_cluster_i, cover_trees[i],
							levels_threads[i], save_distances, dist_function);
					}
				}
				else {
					CoverTree_kNN(coords_ct, chol_ip_cross_cov, corr_diag, 0, re_comps_vecchia_cluster_i, cover_trees[0],
						level, save_distances, dist_function);
				}
#pragma omp parallel for schedule(dynamic)
				for (int i = brute_force_threshold; i < num_data; ++i) {
					if (num_threads != 1) {
						std::map<int, std::vector<int>> neighbors_per_tree;
						std::map<int, std::vector<double>> dist_of_neighbors_per_tree;
						for (int ii = 0; ii < num_threads; ++ii) {
							if (segment_start[ii] >= i) {
								break;
							}
							neighbors_per_tree[ii] = neighbors[i - start_at];
							dist_of_neighbors_per_tree[ii] = dist_dummy;
						}
						for (int ii = 0; ii < (int)neighbors_per_tree.size(); ++ii) {
							if ((segment_start[ii] + num_neighbors) < i) {
								find_kNN_CoverTree(i, num_neighbors, levels_threads[ii], save_distances, coords, chol_ip_cross_cov,
									corr_diag, re_comps_vecchia_cluster_i, neighbors_per_tree[ii], dist_of_neighbors_per_tree[ii], cover_trees[ii], dist_function);
							}
							else if (segment_start[ii] < i) {
								vec_t dist_vect(1);
								int size_smaller_k = std::min(i - segment_start[ii], num_neighbors);
								dist_of_neighbors_per_tree[ii].resize(size_smaller_k);
								neighbors_per_tree[ii].resize(size_smaller_k);
								for (int j = 0; j < size_smaller_k; ++j) {
									dist_of_neighbors_per_tree[ii][j] = std::numeric_limits<double>::infinity();
								}
								for (int jj = segment_start[ii]; jj < i; ++jj) {
									std::vector<int> indj{ jj };
									distances_funct(i, indj, coords, corr_diag, chol_ip_cross_cov,
										re_comps_vecchia_cluster_i, dist_vect, dist_function, save_distances);
									if (dist_vect[0] < dist_of_neighbors_per_tree[ii][size_smaller_k - 1]) {
										dist_of_neighbors_per_tree[ii][size_smaller_k - 1] = dist_vect[0];
										neighbors_per_tree[ii][size_smaller_k - 1] = jj;
										SortVectorsDecreasing<double>(dist_of_neighbors_per_tree[ii].data(), neighbors_per_tree[ii].data(), size_smaller_k);
									}
								}
							}
						}
						if ((int)neighbors_per_tree.size() == 1) {
							neighbors[i - start_at] = neighbors_per_tree[0];
						}
						else {
							std::set<std::tuple<double, int, int, int>> set_tuples;
							for (int ii = 0; ii < (int)neighbors_per_tree.size(); ii++) {
								set_tuples.insert({ dist_of_neighbors_per_tree[ii][0], ii, 0, neighbors_per_tree[ii][0] });
							}
							int index_of_vector;
							int index_in_vector;
							for (int ii = 0; ii < num_neighbors; ii++) {
								auto it = set_tuples.begin();
								neighbors[i - start_at][ii] = std::get<3>(*it);
								index_of_vector = std::get<1>(*it);
								index_in_vector = std::get<2>(*it);
								set_tuples.erase(it);
								if (index_in_vector < (int)dist_of_neighbors_per_tree[index_of_vector].size() - 1) {
									set_tuples.insert({ dist_of_neighbors_per_tree[index_of_vector][index_in_vector + 1], index_of_vector, index_in_vector + 1, neighbors_per_tree[index_of_vector][index_in_vector + 1] });
								}
							}
						}
					}
					else {
						find_kNN_CoverTree(i, num_neighbors, level, save_distances,
							coords, chol_ip_cross_cov, corr_diag, re_comps_vecchia_cluster_i,
							neighbors[i - start_at], dist_dummy, cover_trees[0], dist_function);
					}
					//Save distances between points and neighbors
					if (save_distances) {
						dist_obs_neighbors[i - start_at].resize(num_neighbors, 1);
					}
					for (int j = 0; j < num_nearest_neighbors; ++j) {
						double dij = (coords(i, Eigen::all) - coords(neighbors[i - start_at][j], Eigen::all)).lpNorm<2>();
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
				}
			}
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
				den_mat_t coords_i;
				if (!save_distances) {
					std::vector<int> indi{ neighbors[i - start_at][j] };
					coords_i = coords(indi, Eigen::all);
				}
				for (int k = j + 1; k < nn_i; ++k) {
					den_mat_t dist_ij(1, 1);
					dist_ij(0, 0) = 0.;
					den_mat_t coords_j;
					den_mat_t corr_mat;
					if (save_distances || (check_has_duplicates && !has_duplicates)) {
						if (!save_distances) {
							std::vector<int> indj{ neighbors[i - start_at][k] };
							coords_j = coords(indj, Eigen::all);
						}
						dist_ij(0, 0) = (coords(neighbors[i - start_at][j], Eigen::all) - coords(neighbors[i - start_at][k], Eigen::all)).lpNorm<2>();
					}
					if (save_distances) {
						dist_between_neighbors[i - start_at](j, k) = dist_ij.value();
					}
					if (check_has_duplicates && !has_duplicates) {
						if (dist_ij.value() < EPSILON_NUMBERS) {
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
	}//end find_nearest_neighbors_Vecchia_FSA_fast

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
		bool save_distances_isotropic_cov_fct,
		string_t& gp_approx) {
		int ind_intercept_gp = (int)re_comps_vecchia_cluster_i.size();
		if ((vecchia_ordering == "random" || vecchia_ordering == "time_random_space") && gp_approx != "full_scale_vecchia") {
			std::shuffle(data_indices_per_cluster[cluster_i].begin(), data_indices_per_cluster[cluster_i].end(), rng);//Note: shuffling has been already done if gp_approx == "full_scale_vecchia"
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
		if (gauss_likelihood) {
			std::vector<int> uniques, unique_idx_dummy;
			GPBoost::DetermineUniqueDuplicateCoordsFast(gp_coords_mat, num_data_per_cluster[cluster_i], uniques, unique_idx_dummy);
			if (uniques.size() <= num_data_per_cluster[cluster_i] / 5.) {
				Log::REInfo("There are many duplicate input coordinates (%d unique points among n = %d samples). Consider using gp_approx = 'vecchia_latent' as this might run faster in this case ", uniques.size(), num_data_per_cluster[cluster_i]);
			}
		}
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
		if (!(re_comp->RedetermineVecchiaNeighborsInducingPoints()) && vecchia_neighbor_selection != "residual_correlation" && vecchia_neighbor_selection != "correlation") {
			Log::REDebug("Starting nearest neighbor search for Vecchia approximation");
			find_nearest_neighbors_Vecchia_fast(re_comp->GetCoords(), re_comp->GetNumUniqueREs(), num_neighbors,
				nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, has_duplicates,
				vecchia_neighbor_selection, rng, save_distances_isotropic_cov_fct);
			Log::REDebug("Nearest neighbors for Vecchia approximation found");
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
		if (vecchia_neighbor_selection == "residual_correlation" || vecchia_neighbor_selection == "correlation") {
			has_duplicates = false;
			den_mat_t coords = re_comp->GetCoords();
			//Intialize neighbor vectors
			for (int i = 0; i < num_data; ++i) {
				if (i > 0 && i <= num_neighbors) {
					nearest_neighbors_cluster_i[i].resize(i);
					if (save_distances_isotropic_cov_fct) {
						dist_obs_neighbors_cluster_i[i].resize(i, 1);
					}
					for (int j = 0; j < i; ++j) {
						nearest_neighbors_cluster_i[i][j] = j;
						den_mat_t dist_ij(1, 1);
						dist_ij(0, 0) = 0.;
						if (save_distances_isotropic_cov_fct || (check_has_duplicates && !has_duplicates)) {
							dist_ij(0, 0) = (coords(j, Eigen::all) - coords(i, Eigen::all)).lpNorm<2>();
						}
						if (save_distances_isotropic_cov_fct) {
							dist_obs_neighbors_cluster_i[i](j, 0) = dist_ij.value();
						}
						if (check_has_duplicates && !has_duplicates) {
							if (dist_ij.value() < EPSILON_NUMBERS) {
								has_duplicates = true;
							}
						}//end check_has_duplicates
					}
				}
				else if (i > num_neighbors) {
					nearest_neighbors_cluster_i[i].resize(num_neighbors);
				}
			}
		}
		//Random coefficients
		if (num_gp_rand_coef > 0) {
			if (re_comp->RedetermineVecchiaNeighborsInducingPoints()) {
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
		bool gauss_likelihood,
		string_t& gp_approx,
		const den_mat_t& chol_ip_cross_cov,
		std::vector<den_mat_t>& dist_obs_neighbors_cluster_i,
		std::vector<den_mat_t>& dist_between_neighbors_cluster_i,
		bool save_distances_isotropic_cov_fct) {
		std::shared_ptr<RECompGP<den_mat_t>> re_comp = re_comps_vecchia_cluster_i[ind_intercept_gp];
		CHECK(re_comp->RedetermineVecchiaNeighborsInducingPoints() || vecchia_neighbor_selection == "residual_correlation" || vecchia_neighbor_selection == "correlation");
		int num_re = re_comp->GetNumUniqueREs();
		CHECK((int)nearest_neighbors_cluster_i.size() == num_re);
		// find correlation-based nearest neighbors
		std::vector<den_mat_t> dist_dummy;
		bool has_duplicates = check_has_duplicates;
		if ((gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") || vecchia_neighbor_selection == "correlation") {
			find_nearest_neighbors_Vecchia_FSA_fast(re_comp->GetCoords(), num_re, num_neighbors, chol_ip_cross_cov,
				re_comps_vecchia_cluster_i, nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, 0, -1, has_duplicates, save_distances_isotropic_cov_fct,
				false, false, num_re);
		}
		else {
			// Calculate scaled coordinates
			den_mat_t coords_scaled;
			re_comp->GetScaledCoordinates(coords_scaled);
			// find correlation-based nearest neighbors
			find_nearest_neighbors_Vecchia_fast(coords_scaled, num_re, num_neighbors,
				nearest_neighbors_cluster_i, dist_dummy, dist_dummy, 0, -1, has_duplicates,
				vecchia_neighbor_selection, rng, false);
		}
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
		const double* add_diagonal,
		const std::vector<int>& estimate_cov_par_index) {
		int num_par_comp = re_comps_vecchia_cluster_i[ind_intercept_gp]->NumCovPar();
		int num_par_gp = num_par_comp * num_gp_total + calc_gradient_nugget;
		int nugget_offset_ind_est = (gauss_likelihood && !calc_gradient_nugget) ? 1 : 0;
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
			if (add_diagonal != nullptr) {
				if (calc_gradient) {
					Log::REFatal("CalcCovFactorGradientVecchia: 'add_diagonal' can currently not be used when 'calc_gradient' is true ");
				}
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_re_cluster_i; ++i) {
					D_inv_cluster_i.coeffRef(i, i) += add_diagonal[i];
				}
			}
		}
		bool exclude_marg_var_grad = !gauss_likelihood && (re_comps_vecchia_cluster_i.size() == 1) && !(gp_approx == "full_scale_vecchia");//gradient is not needed if there is only one GP for non-Gaussian likelihoods
		if (calc_gradient) {
			B_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of B = derviateive of (-A)
			D_grad_cluster_i = std::vector<sp_mat_t>(num_par_gp);//derivative of D
			for (int ipar = 0; ipar < num_par_gp; ++ipar) {
				if (!(exclude_marg_var_grad && ipar == 0) && estimate_cov_par_index[ipar + nugget_offset_ind_est] > 0) {
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
		// Components for full scale vecchia
		std::vector<den_mat_t> sigma_cross_cov_gradT((int)num_par_comp);
		std::vector<den_mat_t> sigma_ip_grad((int)num_par_comp);
		if (gp_approx == "full_scale_vecchia") {			
			const den_mat_t* sigma_cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
			if (calc_gradient) {
				CHECK(num_gp_total == 1);
				sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i = std::vector<den_mat_t>(num_par_gp);
				sigma_ip_inv_cross_cov_T_cluster_i = chol_fact_sigma_ip_cluster_i.solve((*sigma_cross_cov).transpose());
#pragma omp parallel for schedule(static)
				for (int ipar = 0; ipar < (int)num_par_comp; ++ipar) {
					if (estimate_cov_par_index[ipar + nugget_offset_ind_est] > 0) {
						sigma_ip_grad[ipar] = *(re_comps_ip_cluster_i[0]->GetZSigmaZtGrad(ipar, true, re_comps_ip_cluster_i[0]->CovPars()[0]));
						sigma_cross_cov_gradT[ipar] = (*(re_comps_cross_cov_cluster_i[0]->GetZSigmaZtGrad(ipar, true, re_comps_cross_cov_cluster_i[0]->CovPars()[0]))).transpose();
						sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar] = sigma_ip_grad[ipar] * sigma_ip_inv_cross_cov_T_cluster_i;
					}
				}
			}
		}
#pragma omp parallel for schedule(static)
		for (data_size_t i = 0; i < num_re_cluster_i; ++i) {
			if (gp_approx == "full_scale_vecchia" && calc_cov_factor) {
				D_inv_cluster_i.coeffRef(i, i) -= chol_ip_cross_cov_cluster_i.col(i).array().square().sum();
			}
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
					std::vector<int> calc_grad_index(estimate_cov_par_index.begin() + ind_first_par + nugget_offset_ind_est, estimate_cov_par_index.begin() + ind_first_par + nugget_offset_ind_est + num_par_comp);
					if (j == 0) {
						if (!distances_saved) {
							std::vector<int> ind{ i };
							re_comp->GetSubSetCoords(ind, coords_i);
							re_comp->GetSubSetCoords(nearest_neighbors_cluster_i[i], coords_nn_i);
						}
						re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors, cov_grad_mats_obs_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, false, calc_grad_index);//write on matrices directly for first GP component
						re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors, cov_grad_mats_between_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, true, calc_grad_index);
						if (gp_approx == "full_scale_vecchia") {
							vec_t sigma_ip_Ihalf_sigma_cross_covT_obs = chol_ip_cross_cov_cluster_i.col(i);
							for (int ii = 0; ii < num_nn; ++ii) {
								cov_mat_obs_neighbors.coeffRef(ii, 0) -= chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i[i][ii]).dot(sigma_ip_Ihalf_sigma_cross_covT_obs);
								for (int jj = ii; jj < num_nn; ++jj) {
									if (ii == jj) {
										cov_mat_between_neighbors.coeffRef(ii, jj) -= chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i[i][ii]).array().square().sum();
									}
									else {
										cov_mat_between_neighbors.coeffRef(ii, jj) -= chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i[i][ii]).dot(chol_ip_cross_cov_cluster_i.col(nearest_neighbors_cluster_i[i][jj]));
										cov_mat_between_neighbors.coeffRef(jj, ii) = cov_mat_between_neighbors.coeffRef(ii, jj);
									}
								}
							}
							// Gradient
							if (calc_gradient) {
								vec_t sigma_ip_I_sigma_cross_covT_obs = sigma_ip_inv_cross_cov_T_cluster_i.col(i);
								for (int ipar = 0; ipar < (int)num_par_comp; ++ipar) {
									if (estimate_cov_par_index[ind_first_par + ipar + nugget_offset_ind_est] > 0) {
										vec_t sigma_cross_cov_gradT_obs = sigma_cross_cov_gradT[ipar].col(i);
										vec_t sigma_ip_grad_sigma_ip_inv_cross_cov_T_obs = sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar].col(i);
										for (int ii = 0; ii < num_nn; ++ii) {
											cov_grad_mats_obs_neighbors[ind_first_par + ipar].coeffRef(ii, 0) -= (sigma_cross_cov_gradT[ipar]).col(nearest_neighbors_cluster_i[i][ii]).dot(sigma_ip_I_sigma_cross_covT_obs) +
												sigma_ip_inv_cross_cov_T_cluster_i.col(nearest_neighbors_cluster_i[i][ii]).dot(sigma_cross_cov_gradT_obs - sigma_ip_grad_sigma_ip_inv_cross_cov_T_obs);
											for (int jj = ii; jj < num_nn; ++jj) {
												cov_grad_mats_between_neighbors[ind_first_par + ipar].coeffRef(ii, jj) -= (sigma_cross_cov_gradT[ipar]).col(nearest_neighbors_cluster_i[i][ii]).dot(sigma_ip_inv_cross_cov_T_cluster_i.col(nearest_neighbors_cluster_i[i][jj])) +
													sigma_ip_inv_cross_cov_T_cluster_i.col(nearest_neighbors_cluster_i[i][ii]).dot((sigma_cross_cov_gradT[ipar]).col(nearest_neighbors_cluster_i[i][jj]) -
														sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar].col(nearest_neighbors_cluster_i[i][jj]));
												if (ii != jj) {
													cov_grad_mats_between_neighbors[ind_first_par + ipar].coeffRef(jj, ii) = cov_grad_mats_between_neighbors[ind_first_par + ipar].coeffRef(ii, jj);
												}
											}
										}
									}
								}
							}
						}
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors_j, cov_grad_mats_obs_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, false, calc_grad_index);
						re_comps_vecchia_cluster_i[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors_j, cov_grad_mats_between_neighbors.data() + ind_first_par,
							calc_gradient, transf_scale, nugget_var, true, calc_grad_index);
						//multiply by coefficient matrix
						cov_mat_obs_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();//cov_mat_obs_neighbors_j.cwiseProduct()
						cov_mat_between_neighbors_j.array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
						cov_mat_obs_neighbors += cov_mat_obs_neighbors_j;
						cov_mat_between_neighbors += cov_mat_between_neighbors_j;
						if (calc_gradient) {
							for (int ipar = 0; ipar < (int)num_par_comp; ++ipar) {
								if (estimate_cov_par_index[ind_first_par + ipar + nugget_offset_ind_est] > 0) {
									cov_grad_mats_obs_neighbors[ind_first_par + ipar].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 0, num_nn, 1)).array();
									cov_grad_mats_between_neighbors[ind_first_par + ipar].array() *= (z_outer_z_obs_neighbors_cluster_i[i][j - 1].block(1, 1, num_nn, num_nn)).array();
								}
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
					d_comp_j *= nugget_var;//back-transform
				}
				if (j > 0) {//random coefficient
					d_comp_j *= z_outer_z_obs_neighbors_cluster_i[i][j - 1](0, 0);
				}
				if (calc_cov_factor) {
					D_inv_cluster_i.coeffRef(i, i) += d_comp_j;
				}
				if (calc_gradient && estimate_cov_par_index[j * num_par_comp + nugget_offset_ind_est] > 0) {
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
			if (calc_gradient && calc_gradient_nugget && estimate_cov_par_index[0] > 0) {
				D_grad_cluster_i[num_par_gp - 1].coeffRef(i, i) = 1.;
			}
			//2. remaining terms
			if (i > 0) {
				if (gauss_likelihood) {
					if (transf_scale) {
						cov_mat_between_neighbors.diagonal().array() += 1.;//add nugget effect
					}
					else {
						cov_mat_between_neighbors.diagonal().array() += nugget_var;//back-transform
					}
				}
				else {
					cov_mat_between_neighbors.diagonal().array() *= JITTER_MULT_VECCHIA;//Avoid numerical problems when there is no nugget effect
				}
				if (add_diagonal != nullptr) {
					for (int ii = 0; ii < (int)cov_mat_between_neighbors.rows(); ++ii) {
						cov_mat_between_neighbors(ii, ii) += add_diagonal[nearest_neighbors_cluster_i[i][ii]];
					}
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
					if (calc_gradient_nugget && estimate_cov_par_index[0] > 0) {
						A_i_grad_sigma2 = -(chol_fact_between_neighbors.solve(A_i.transpose())).transpose();
					}
					den_mat_t A_i_grad(1, num_nn);
					for (int j = 0; j < num_gp_total; ++j) {
						int ind_first_par = j * num_par_comp;
						for (int ipar = 0; ipar < num_par_comp; ++ipar) {
							if (!(exclude_marg_var_grad && ipar == 0) && estimate_cov_par_index[ind_first_par + ipar + nugget_offset_ind_est] > 0) {
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
								if (gp_approx == "full_scale_vecchia") {
									D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) -= sigma_ip_inv_cross_cov_T_cluster_i.col(i).dot(2 * sigma_cross_cov_gradT[ipar].col(i) -
										sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar].col(i));
								}
							}
						}
					}
					if (calc_gradient_nugget && estimate_cov_par_index[0] > 0) {
						for (int inn = 0; inn < num_nn; ++inn) {
							B_grad_cluster_i[num_par_gp - 1].coeffRef(i, nearest_neighbors_cluster_i[i][inn]) = -A_i_grad_sigma2(0, inn);
						}
						D_grad_cluster_i[num_par_gp - 1].coeffRef(i, i) -= (A_i_grad_sigma2 * cov_mat_obs_neighbors)(0, 0);
					}
				}//end calc_gradient
			}//end if i > 0;
			if (i == 0 && calc_gradient) {
				if (gp_approx == "full_scale_vecchia") {
					for (int j = 0; j < num_gp_total; ++j) {
						int ind_first_par = j * num_par_comp;
						for (int ipar = 0; ipar < num_par_comp; ++ipar) {
							if (!(exclude_marg_var_grad && ipar == 0) && estimate_cov_par_index[ind_first_par + ipar + nugget_offset_ind_est] > 0) {
								D_grad_cluster_i[ind_first_par + ipar].coeffRef(i, i) -= sigma_ip_inv_cross_cov_T_cluster_i.col(i).dot(2 * sigma_cross_cov_gradT[ipar].col(i) -
									sigma_ip_grad_sigma_ip_inv_cross_cov_T_cluster_i[ipar].col(i));
							}
						}
					}
				}
			}
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
		string_t& gp_approx) {
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
		bool scale_coordinates = re_comp->UseScaledCoordinates();
		den_mat_t coords_scaled;
		if (scale_coordinates) {
			const vec_t pars = re_comp->CovPars();
			re_comp->ScaleCoordinates(pars, coords_all, coords_scaled);
		}
		// Components for full scale vecchia
		den_mat_t chol_ip_cross_cov_pred, chol_ip_cross_cov_obs, chol_ip_cross_cov_obs_pred, sigma_ip_inv_sigma_cross_cov, sigma_ip_inv_sigma_cross_cov_pred;
		// Cross-covariance between predictions and inducing points C_pm
		den_mat_t cov_mat_pred_id, cross_dist;
		std::shared_ptr<RECompGP<den_mat_t>> re_comp_cross_cov_cluster_i_pred_ip;
		// Components for prediction of full scale vecchia
		if (gp_approx == "full_scale_vecchia") {
			const den_mat_t* sigma_cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
			re_comp_cross_cov_cluster_i_pred_ip = re_comps_cross_cov_cluster_i[0];
			re_comp_cross_cov_cluster_i_pred_ip->AddPredCovMatrices(gp_coords_mat_ip, gp_coords_mat_pred, cross_cov_pred_ip,
				cov_mat_pred_id, true, false, true, nullptr, false, cross_dist);
			TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_cluster_i,
				(*sigma_cross_cov).transpose(), chol_ip_cross_cov_obs, false);
			TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip_cluster_i,
				cross_cov_pred_ip.transpose(), chol_ip_cross_cov_pred, false);
			sigma_ip_inv_sigma_cross_cov = chol_fact_sigma_ip_cluster_i.solve((*sigma_cross_cov).transpose());
			sigma_ip_inv_sigma_cross_cov_pred = chol_fact_sigma_ip_cluster_i.solve(cross_cov_pred_ip.transpose());
			if (vecchia_neighbor_selection == "residual_correlation") {
				chol_ip_cross_cov_obs_pred.resize(chol_ip_cross_cov_obs.rows(), chol_ip_cross_cov_obs.cols() + chol_ip_cross_cov_pred.cols());
				chol_ip_cross_cov_obs_pred.leftCols(chol_ip_cross_cov_obs.cols()) = chol_ip_cross_cov_obs;
				chol_ip_cross_cov_obs_pred.rightCols(chol_ip_cross_cov_pred.cols()) = chol_ip_cross_cov_pred;
			}
		}
		if (CondObsOnly) {
			if ((gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") || vecchia_neighbor_selection == "correlation") {
				find_nearest_neighbors_Vecchia_FSA_fast(coords_all, num_re_cli + num_re_pred_cli, num_neighbors_pred, chol_ip_cross_cov_obs_pred,
					re_comps_vecchia, nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli,
					num_re_cli - 1, check_has_duplicates, distances_saved, true, false, (int)num_re_cli);
			}
			else {
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
		}
		else {//find neighbors among both the observed and prediction locations
			if (!gauss_likelihood) {
				check_has_duplicates = true;
			}
			if ((gp_approx == "full_scale_vecchia" && vecchia_neighbor_selection == "residual_correlation") || vecchia_neighbor_selection == "correlation") {
				find_nearest_neighbors_Vecchia_FSA_fast(coords_all, num_re_cli + num_re_pred_cli, num_neighbors_pred, chol_ip_cross_cov_obs_pred,
					re_comps_vecchia, nearest_neighbors_cluster_i, dist_obs_neighbors_cluster_i, dist_between_neighbors_cluster_i, num_re_cli,
					-1, check_has_duplicates, distances_saved, true, true, (int)num_re_cli);
			}
			else {
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
				std::vector<int> calc_grad_index_dummy;
				if (j == 0) {
					if (!distances_saved) {
						std::vector<int> ind{ num_re_cli + i };
						coords_i = coords_all(ind, Eigen::all);
						coords_nn_i = coords_all(nearest_neighbors_cluster_i[i], Eigen::all);
					}
					re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
						cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false, calc_grad_index_dummy);//write on matrices directly for first GP component
					re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
						cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true, calc_grad_index_dummy);
					// Residual process of full-scale Vecchia approximation
					if (gp_approx == "full_scale_vecchia") {
						std::vector<int> ind_pred{ i };
						// Cross-covariance neighbors and inducing points
						den_mat_t sigma_ip_inv_cross_cov_neighbors(chol_ip_cross_cov_obs.rows(), num_nn);
						for (int inn = 0; inn < num_nn; ++inn) {
							if (nearest_neighbors_cluster_i[i][inn] < num_re_cli) {//nearest neighbor belongs to observed data
								sigma_ip_inv_cross_cov_neighbors.col(inn) = chol_ip_cross_cov_obs.col(nearest_neighbors_cluster_i[i][inn]);
							}
							else {
								sigma_ip_inv_cross_cov_neighbors.col(inn) = chol_ip_cross_cov_pred.col(nearest_neighbors_cluster_i[i][inn] - num_re_cli);
							}
						}
						cov_mat_obs_neighbors -= sigma_ip_inv_cross_cov_neighbors.transpose() * chol_ip_cross_cov_pred(Eigen::all, ind_pred);
						cov_mat_between_neighbors -= sigma_ip_inv_cross_cov_neighbors.transpose() * sigma_ip_inv_cross_cov_neighbors;
					}
				}
				else {//random coefficient GPs
					den_mat_t cov_mat_obs_neighbors_j;
					den_mat_t cov_mat_between_neighbors_j;
					re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
						cov_mat_obs_neighbors_j, &cov_grad_dummy, false, true, 1., false, calc_grad_index_dummy);
					re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
						cov_mat_between_neighbors_j, &cov_grad_dummy, false, true, 1., true, calc_grad_index_dummy);
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
			if (gp_approx == "full_scale_vecchia") {
				Dp[i] -= chol_ip_cross_cov_pred.col(i).array().square().sum();
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
		// row-major
		sp_mat_rm_t Bpo_rm = sp_mat_rm_t(Bpo);
		sp_mat_rm_t Bp_rm = sp_mat_rm_t(Bp);
		if (gauss_likelihood) {
			if (gp_approx == "full_scale_vecchia") {
				const den_mat_t* sigma_cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
				vec_t vecchia_y = chol_fact_sigma_woodbury_cluster_i.solve((*sigma_cross_cov).transpose() * (Bt_D_inv_cluster_i * (B_cluster_i * y_cluster_i)));
				pred_mean = -Bpo_rm * (y_cluster_i - (*sigma_cross_cov) * vecchia_y);
				if (!CondObsOnly) {
					sp_L_solve(Bp.valuePtr(), Bp.innerIndexPtr(), Bp.outerIndexPtr(), num_re_pred_cli, pred_mean.data());
				}
				pred_mean += cross_cov_pred_ip * vecchia_y;
				if (calc_pred_cov || calc_pred_var) {
					den_mat_t Vecchia_cross_cov((*sigma_cross_cov).rows(), (*sigma_cross_cov).cols());
#pragma omp parallel for schedule(static)   
					for (int i = 0; i < (*sigma_cross_cov).cols(); ++i) {
						Vecchia_cross_cov.col(i) = Bt_D_inv_cluster_i * (B_cluster_i * (*sigma_cross_cov).col(i));
					}
					den_mat_t cross_cov_PP_Vecchia = chol_ip_cross_cov_pred.transpose() * (chol_ip_cross_cov_obs * Vecchia_cross_cov);
					den_mat_t cross_cov_pred_obs_pred_inv;
					den_mat_t B_po_cross_cov(num_re_pred_cli, (*sigma_cross_cov).cols());
#pragma omp parallel for schedule(static)   
					for (int i = 0; i < (*sigma_cross_cov).cols(); ++i) {
						B_po_cross_cov.col(i) = Bpo_rm * (*sigma_cross_cov).col(i);
					}
					den_mat_t cross_cov_PP_Vecchia_woodbury = chol_fact_sigma_woodbury_cluster_i.solve(cross_cov_PP_Vecchia.transpose());
					sp_mat_t Bp_inv_Dp;
					sp_mat_t Bp_inv(num_re_pred_cli, num_re_pred_cli);
					if (CondObsOnly) {
						if (calc_pred_cov) {
							pred_cov = Dp.asDiagonal();
						}
						if (calc_pred_var) {
							pred_var = Dp;
						}
						cross_cov_pred_obs_pred_inv = B_po_cross_cov;
					}
					else {
						TriangularSolve<sp_mat_t, den_mat_t, den_mat_t>(Bp, B_po_cross_cov, cross_cov_pred_obs_pred_inv, false);
						Bp_inv.setIdentity();
						TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
						Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
						if (calc_pred_cov) {
							pred_cov = den_mat_t(Bp_inv_Dp * Bp_inv.transpose());
						}
						if (calc_pred_var) {
							pred_var.resize(num_re_pred_cli);
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_re_pred_cli; ++i) {
								pred_var[i] = (Bp_inv_Dp.row(i)).dot(Bp_inv.row(i));
							}
						}
					}
					den_mat_t cross_cov_pred_obs_pred_inv_woodbury = chol_fact_sigma_woodbury_cluster_i.solve(cross_cov_pred_obs_pred_inv.transpose());
					if (calc_pred_cov) {
						if (num_re_pred_cli > 10000) {
							Log::REInfo("The computational complexity and the storage of the predictive covariance martix heavily depend on the number of prediction location. "
								"Therefore, if this number is large we recommend only computing the predictive variances ");
						}
						den_mat_t PP_Part = cross_cov_pred_ip * sigma_ip_inv_sigma_cross_cov_pred;
						den_mat_t PP_V_Part = cross_cov_PP_Vecchia * sigma_ip_inv_sigma_cross_cov_pred;
						den_mat_t V_Part = cross_cov_pred_obs_pred_inv * sigma_ip_inv_sigma_cross_cov_pred;
						den_mat_t V_Part_t = V_Part.transpose();
						den_mat_t PP_V_PP_Part = cross_cov_pred_obs_pred_inv * cross_cov_PP_Vecchia_woodbury;
						den_mat_t PP_V_PP_Part_t = PP_V_PP_Part.transpose();
						den_mat_t PP_V_V_Part = cross_cov_PP_Vecchia * cross_cov_PP_Vecchia_woodbury;
						den_mat_t V_V_Part = cross_cov_pred_obs_pred_inv * cross_cov_pred_obs_pred_inv_woodbury;
						pred_cov += PP_Part - PP_V_Part + V_Part + V_Part_t - PP_V_PP_Part + PP_V_V_Part - PP_V_PP_Part_t + V_V_Part;
					}
					if (calc_pred_var) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_re_pred_cli; ++i) {
							pred_var[i] += (cross_cov_pred_ip.row(i) - cross_cov_PP_Vecchia.row(i) +
								2 * cross_cov_pred_obs_pred_inv.row(i)).dot(sigma_ip_inv_sigma_cross_cov_pred.col(i)) +
								(cross_cov_PP_Vecchia.row(i) - 2 * cross_cov_pred_obs_pred_inv.row(i)).dot(cross_cov_PP_Vecchia_woodbury.col(i)) +
								(cross_cov_pred_obs_pred_inv.row(i)).dot(cross_cov_pred_obs_pred_inv_woodbury.col(i));
						}
					}
				}
			} // end FSVA
			else {
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
			}// end Vecchia
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
		bool scale_coordinates = re_comp->UseScaledCoordinates();
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
					std::vector<int> calc_grad_index_dummy;
					if (j == 0) {
						if (!distances_saved) {
							std::vector<int> ind{ i };
							coords_i = coords_all(ind, Eigen::all);
							coords_nn_i = coords_all(nearest_neighbors_cluster_i[i], Eigen::all);
						}
						re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false, calc_grad_index_dummy);//write on matrices directly for first GP component
						re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true, calc_grad_index_dummy);
					}
					else {//random coefficient GPs
						den_mat_t cov_mat_obs_neighbors_j;
						den_mat_t cov_mat_between_neighbors_j;
						re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
							cov_mat_obs_neighbors_j, &cov_grad_dummy, false, true, 1., false, calc_grad_index_dummy);
						re_comps_vecchia[ind_intercept_gp + j]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
							cov_mat_between_neighbors_j, &cov_grad_dummy, false, true, 1., true, calc_grad_index_dummy);
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
		bool scale_coordinates = re_comp->UseScaledCoordinates();
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
			std::vector<int> calc_grad_index_dummy;
			if (i > 0) {
				if (!distances_saved) {
					std::vector<int> ind{ i };
					coords_i = coords_all_unique(ind, Eigen::all);
					coords_nn_i = coords_all_unique(nearest_neighbors_cluster_i[i], Eigen::all);
				}
				re_comps_vecchia[ind_intercept_gp]->CalcSigmaAndSigmaGradVecchia(dist_obs_neighbors_cluster_i[i], coords_i, coords_nn_i,
					cov_mat_obs_neighbors, &cov_grad_dummy, false, true, 1., false, calc_grad_index_dummy);//write on matrices directly for first GP component
				re_comps_vecchia[ind_intercept_gp]->CalcSigmaAndSigmaGradVecchia(dist_between_neighbors_cluster_i[i], coords_nn_i, coords_nn_i,
					cov_mat_between_neighbors, &cov_grad_dummy, false, true, 1., true, calc_grad_index_dummy);
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
