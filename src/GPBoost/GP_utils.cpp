/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/GP_utils.h>
#include <GPBoost/utils.h>
#include <cmath>

namespace GPBoost {

	void DetermineUniqueDuplicateCoords(const den_mat_t& coords,
		data_size_t num_data,
		std::vector<int>& uniques,
		std::vector<int>& unique_idx) {
		uniques = std::vector<int>();
		unique_idx = std::vector<int>();
		uniques.push_back(0);
		unique_idx.push_back(0);
		double EPSILON_NUMBERS_SQUARE = EPSILON_NUMBERS * EPSILON_NUMBERS;
		for (int i = 1; i < num_data; ++i) {//identify duplicates in coordinates
			bool is_duplicate = false;
			for (int j = 0; j < (int)uniques.size(); ++j) {
				if ((coords.row(uniques[j]) - coords.row(i)).squaredNorm() < EPSILON_NUMBERS_SQUARE) {
					unique_idx.push_back(j);
					is_duplicate = true;
					break;
				}
			}
			// parallel version (unclear whether this is faster given that parallelization has to be done in every iteration i)
//			volatile bool is_duplicate = false;
//#pragma omp parallel for shared(is_duplicate)
//			for (int j = 0; j < (int)uniques.size(); ++j) {
//				if (is_duplicate) continue;
//				if ((coords.row(uniques[j]) - coords.row(i)).squaredNorm() < EPSILON_NUMBERS_SQUARE) {
//					unique_idx.push_back(j);
//					is_duplicate = true;
//				}
//			}
			if (!is_duplicate) {
				unique_idx.push_back((int)uniques.size());
				uniques.push_back(i);
			}
		}
	}//end DetermineUniqueDuplicateCoords

	void closest_distance(const den_mat_t& means,
		const den_mat_t& data,
		vec_t& distances) {
#pragma omp parallel for schedule(static)
		for (int i = 0; i < data.rows(); ++i) {
			double distance = (data(i, Eigen::all) - means(0, Eigen::all)).lpNorm<2>();
			if (distances[i] > distance || distances[i] < 0) {
				distances[i] = distance;
			}
		}
	}// end closest_distance

	void random_plusplus(const den_mat_t& data,
		int k,
		RNG_t& gen,
		den_mat_t& means) {

		vec_t distances(data.rows());
		distances.setOnes();
		int v;
		for (int i = 0; i < k; ++i) {
			// Calculate the distance to the closest mean for each data point
			if (i == 1) {
				distances.array() *= -1;
			}
			if (i > 0) {
				closest_distance(means.block(i - 1, 0, 1, means.cols()), data, distances);
			}
			// Pick a random point weighted by the distance from existing means
			v = std::discrete_distribution<>(distances.data(), distances.data() + distances.size())(gen);
			means(i, Eigen::all) = data(v, Eigen::all);
		}
	}// end random_plusplus

	void calculate_means(const den_mat_t& data,
		vec_t& clusters,
		den_mat_t& means,
		vec_t& indices) {
		den_mat_t means_new(means.rows(), means.cols());
		means_new.setZero();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < data.rows(); ++i) {
			clusters[i] = 0;
			double smallest_distance = (data(i, Eigen::all) - means(0, Eigen::all)).lpNorm<2>();
			for (int j = 1; j < means.rows(); ++j) {
				double distance = (data(i, Eigen::all) - means(j, Eigen::all)).lpNorm<2>();
				if (distance < smallest_distance) {
					smallest_distance = distance;
					clusters[i] = j;
				}
			}
		}
#pragma omp parallel for schedule(static)
		for (int i = 0; i < means.rows(); ++i) {
			double smallest_distance = (data(0, Eigen::all) - means(i, Eigen::all)).lpNorm<2>();
			indices[i] = 0;
			int count = 0;
			if (clusters(0) == i) {
				means_new(i, Eigen::all) += data(0, Eigen::all);
				count += 1;
			}
			for (int j = 1; j < data.rows(); ++j) {
				double distance = (data(j, Eigen::all) - means(i, Eigen::all)).lpNorm<2>();
				if (distance < smallest_distance) {
					smallest_distance = distance;
					indices[i] = j;
				}
				if (clusters(j) == i) {
					means_new(i, Eigen::all) += data(j, Eigen::all);
					count += 1;
				}
			}
			if (count > 0) {
				means(i, Eigen::all) = means_new(i, Eigen::all) / count;
			}
		}
	}//end calculate_means

	void kmeans_plusplus(const den_mat_t& data,
		int k,
		RNG_t& gen,
		den_mat_t& means,
		int max_it) {
		// Initialization
		random_plusplus(data, k, gen, means);
		den_mat_t old_means(k, data.cols());
		old_means.setZero();
		den_mat_t old_old_means = old_means;
		vec_t clusters(data.rows());
		vec_t indices_interim(k);
		indices_interim.setZero();
		// Calculate new means until convergence is reached or we hit the maximum iteration count
		int count = 0;
		do {
			old_old_means = old_means;
			old_means = means;
			calculate_means(data, clusters, means, indices_interim);
			count += 1;
		} while ((means != old_means && means != old_old_means)
			&& !(max_it == count));
		Log::REInfo("Kmeans Iterations: %i", count);
	}//end kmeans_plusplus

	void data_in_ball(const den_mat_t& data,
		const std::vector<int>& indices_start,
		double radius,
		const vec_t& mid,
		std::vector<int>& indices) {
		for (int i = 0; i < indices_start.size(); ++i) {
			double distance = (data(indices_start[i], Eigen::all) - (den_mat_t)mid.transpose()).lpNorm<2>();
			if (distance <= radius) {
				indices.push_back(indices_start[i]);
			}
		}
	}//end data_in_ball

	void CoverTree(const den_mat_t& data,
		double eps,
		RNG_t& gen,
		den_mat_t& means) {

		//max distance
		den_mat_t z_0 = data.colwise().mean();
		double max_dist_d = (data(0, Eigen::all) - z_0).lpNorm<2>();
		for (int i = 1; i < data.rows(); ++i) {
			double distance_new = (data(i, Eigen::all) - z_0).lpNorm<2>();
			if (distance_new > max_dist_d) {
				max_dist_d = distance_new;
			}
		}
		//Tree depth
		int L = (int)(ceil(log2(max_dist_d / eps)));
		//number of nodes
		int M_l_minus = 1;
		//radius
		double R_max = pow(2, L) * eps;
		//Initialization
		int M_l, c;
		double R_l;
		std::vector<int> all_indices(data.rows());
		std::iota(std::begin(all_indices), std::end(all_indices), 0);
		// Covert points
		std::map<int, std::vector<int>> covert_points_old;
		covert_points_old[0] = all_indices;
		std::map<int, std::vector<int>> covert_points;
		covert_points[0] = all_indices;
		std::map<int, std::vector<int>> children;
		std::map<int, std::vector<int>> R_neighbors;
		R_neighbors[0].push_back(0);
		means.resize(data.rows(), data.cols());
		int count_ip = 0;
		for (int l = 0; l < L; ++l) {
			count_ip = 0;
			means.setZero();
			//number of nodes
			M_l = 0;
			//child node index
			c = 0;
			//new radius
			R_l = R_max / pow(2, l + 1);
			children.clear();
			covert_points_old.clear();
			covert_points_old = covert_points;
			covert_points.clear();

			for (int p = 0; p < M_l_minus; ++p) {
				children[p].clear();


				do {
					int v = std::uniform_int_distribution<>(0, (int)(covert_points_old[p].size()) - 1)(gen);
					means(c, Eigen::all) = data(covert_points_old[p][v], Eigen::all);
					std::vector<int> indices_ball;
					data_in_ball(data, covert_points_old[p], R_l, means(c, Eigen::all), indices_ball);

					std::vector<int> intersection_vect = indices_ball;
					den_mat_t zeta_opt = data(intersection_vect, Eigen::all).colwise().mean();
					vec_t distance_to_others(children[p].size());
#pragma omp parallel for schedule(static)
					for (int i = 0; i < children[p].size(); ++i) {
						distance_to_others[i] = (means(children[p][i], Eigen::all) - zeta_opt).lpNorm<2>();

					}
					if (distance_to_others.size() > 0) {
						if (distance_to_others.minCoeff() > R_l) {
							means(c, Eigen::all) = zeta_opt;
						}
					}
					else {
						means(c, Eigen::all) = zeta_opt;
					}

					// Remove Covert indices
					for (int ii = 0; ii < R_neighbors[p].size(); ++ii) {
						int index_R_neighbors = R_neighbors[p][ii];
						std::vector<int> indices_ball_c;
						data_in_ball(data, covert_points_old[index_R_neighbors], R_l, means(c, Eigen::all), indices_ball_c);
						std::vector<int> diff_vect;
						std::set_difference(covert_points_old[index_R_neighbors].begin(), covert_points_old[index_R_neighbors].end(),
							indices_ball_c.begin(), indices_ball_c.end(),
							std::inserter(diff_vect, diff_vect.begin()));
						covert_points_old[index_R_neighbors] = diff_vect;
					}


					if (children.find(p) == children.end()) {
						std::vector<int> id_c{ c };
						children.insert({ p, id_c });
					}
					else {
						children[p].push_back(c);
					}

					c += 1;
					count_ip += 1;
					M_l += 1;
				} while (covert_points_old[p].size() != 0);

			}
			// Voroni
			den_mat_t means_c = means.topRows(c + 1);
			for (int ii = 0; ii < data.rows(); ++ii) {
				int i = 0;
				vec_t distances_jj(means_c.rows());
#pragma omp parallel for schedule(static)
				for (int jj = 0; jj < means_c.rows(); ++jj) {
					distances_jj[jj] = (means_c(jj, Eigen::all) - data(ii, Eigen::all)).lpNorm<2>();
				}
				double min_c = distances_jj.minCoeff(&i);
				if (covert_points.find(i) == covert_points.end()) {
					std::vector<int> id_min_c{ ii };
					covert_points.insert({ i, id_min_c });
				}
				else {
					covert_points[i].push_back(ii);
				}
			}
			R_neighbors.clear();
			// R_neighbors
			for (int jj = 0; jj < means_c.rows(); ++jj) {
				for (int ii = 0; ii < means_c.rows(); ++ii) {
					double distance_btw_childs = (means_c(jj, Eigen::all) - means_c(ii, Eigen::all)).lpNorm<2>();
					if (distance_btw_childs <= 4 * (1 - 1 / pow(2, L - l)) * R_l) {
						R_neighbors[jj].push_back(ii);
					}
				}
			}
			M_l_minus = M_l;
		}
		means.conservativeResize(count_ip, means.cols());
		Log::REInfo("Number of inducing points: %i", count_ip);
	}//end CoverTree

}  // namespace GPBoost
