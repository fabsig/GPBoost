/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2024 Fabio Sigrist and Tim Gyger. All rights reserved.
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
		int max_it,
		bool initial_means_provided) {
		CHECK(k <= (int)data.rows());
		// Initialization
		if (!initial_means_provided) {
			random_plusplus(data, k, gen, means);
		}
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
	}//end kmeans_plusplus

	void data_in_ball(const den_mat_t& data,
		const std::vector<int>& indices_start,
		double radius,
		const vec_t& mid,
		std::vector<int>& indices) {
		for (int i = 0; i < (int)indices_start.size(); ++i) {
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
		if (L < 1) {
			L = 1;
		}
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
					if ((int)covert_points_old[p].size() == 0) {
						break;
					}
					int v = std::uniform_int_distribution<>(0, (int)(covert_points_old[p].size()) - 1)(gen);
					means(c, Eigen::all) = data(covert_points_old[p][v], Eigen::all);
					std::vector<int> indices_ball;
					data_in_ball(data, covert_points_old[p], R_l, means(c, Eigen::all), indices_ball);
					
					std::vector<int> intersection_vect = indices_ball;
					den_mat_t zeta_opt = data(intersection_vect, Eigen::all).colwise().mean();
					vec_t distance_to_others(children[p].size());
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)children[p].size(); ++i) {
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
					for (int ii = 0; ii < (int)R_neighbors[p].size(); ++ii) {
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
	}//end CoverTree
  
	void DetermineUniqueDuplicateCoordsFast(const den_mat_t& coords,
		data_size_t num_data,
		std::vector<int>& uniques,
		std::vector<int>& unique_idx) {
		CHECK((data_size_t)coords.rows() == num_data)
		unique_idx = std::vector<int>(num_data);
		double EPSILON_NUMBERS_SQUARE = EPSILON_NUMBERS * EPSILON_NUMBERS;
		std::vector<double> coords_sum(num_data);
		std::vector<int> sort_sum(num_data);
		std::vector<int> uniques_sorted;//index of unique points on sorted mean scale
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			coords_sum[i] = coords(i, Eigen::all).sum();
		}
		SortIndeces<double>(coords_sum, sort_sum);
		for (int i_sort = 0; i_sort < num_data; ++i_sort) {
			//find potential duplicates in coordinates for point i (= points with same mean)
			int i_actual = sort_sum[i_sort];
			uniques_sorted.push_back(i_actual);//assume that 'i_sort' / 'i_actual' is a new unique point -> add it to 'uniques_sorted' (might be changed below)
			unique_idx[i_actual] = (int)uniques_sorted.size() - 1;//'i_sort' / 'i_actual' is a new unique point -> its index stored in 'unique_idx' is its position in 'uniques_sorted', i.e., the last position of 'uniques_sorted' 
			int j_sort;
			//check for potential duplicates: find all points that have the same mean / 'coords_sum' value as 'i_actual'
			for (j_sort = i_sort + 1; j_sort < num_data; ++j_sort) {
				if (NumberIsSmallerThan<double>(coords_sum[i_actual], coords_sum[sort_sum[j_sort]])) {
					break;//the loop end when 'j_sort' > 'i_sort' is for sure not a duplicate
				}
			}
			j_sort--;//'j_sort' = index of last potential duplicate with i_sort
			//identify true duplicates among potential duplicates
			if (j_sort > i_sort) {//more than one potential duplicates
				std::vector<int> uniques_i = std::vector<int>();//index of "local" unique points among the potential duplicates (i_sort,...,j_sort) - i_sort = (0,...,j_sort - i_sort) , length(uniques_i) = number of unique "local" points among (i_sort,...,j_sort) - i_sort
				uniques_i.push_back(0);//'i_sort' / '0' is the first "local" unique point
				std::vector<int> index_data_uniques(j_sort - i_sort + 1);//index that linkes every "local" unique point in 'uniques_i' to a "global" unique point in 'uniques_sorted'
				index_data_uniques[0] = (int)uniques_sorted.size() - 1; // the first "local" unique point is the last one added to 'uniques_sorted'
				for (int jj = 1; jj < (j_sort - i_sort + 1); ++jj) {//loop over all potential duplicates and check whether they are true duplicates
					int jj_actual = sort_sum[i_sort + jj];
					bool is_duplicate = false;
					//check whether 'jj' / 'jj_actual' is a duplicate by comparing it to all true unique points in 'uniques_i'
					for (int ind_uniques_i = 0; ind_uniques_i < (int)uniques_i.size(); ++ind_uniques_i) {
						int ii_actual = sort_sum[i_sort + uniques_i[ind_uniques_i]];
						if ((coords.row(ii_actual) - coords.row(jj_actual)).squaredNorm() < EPSILON_NUMBERS_SQUARE) {
							//make sure that the first appearance of a coordinate is chosen: -> if jj_actual < ii_actual switch them
							if (jj_actual < ii_actual) {
								uniques_sorted[index_data_uniques[uniques_i[ind_uniques_i]]] = jj_actual;
								index_data_uniques[jj] = index_data_uniques[uniques_i[ind_uniques_i]];//the new "local" unique point 'jj' obtains the same index as the one that is replaces ('uniques_i[ind_uniques_i]')
								uniques_i[ind_uniques_i] = jj;//'ind_uniques_i' now longer a unique point but rather 'jj'
							}
							unique_idx[jj_actual] = index_data_uniques[uniques_i[ind_uniques_i]];
							is_duplicate = true;
							break;
						}
					}
					if (!is_duplicate) {
						uniques_i.push_back(jj);
						uniques_sorted.push_back(jj_actual);
						unique_idx[jj_actual] = (int)uniques_sorted.size() - 1;
						index_data_uniques[jj] = (int)uniques_sorted.size() -1 ;
					}
				}
				i_sort = j_sort;
			}//end j_sort > i_sort)
		}
		// sort indices again
		std::vector<int> order_uniques(uniques_sorted.size());
		SortIndeces<int>(uniques_sorted, order_uniques);
		std::vector<int> inv_order_uniques(uniques_sorted.size());
		uniques = std::vector<int>(uniques_sorted.size());
#pragma omp parallel for schedule(static)
		for (int i = 0; i < (int)uniques_sorted.size(); ++i) {
			inv_order_uniques[order_uniques[i]] = i;
			uniques[i] = uniques_sorted[order_uniques[i]];
		}
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			unique_idx[i] = inv_order_uniques[unique_idx[i]];
		}
	}//end DetermineUniqueDuplicateCoordsFast

}  // namespace GPBoost
