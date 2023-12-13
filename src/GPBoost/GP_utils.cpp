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

	void DetermineUniqueDuplicateCoordsFast(const den_mat_t& coords,
		data_size_t num_data,
		std::vector<int>& uniques,
		std::vector<int>& unique_idx) {
		unique_idx = std::vector<int>(num_data);
		double EPSILON_NUMBERS_SQUARE = EPSILON_NUMBERS * EPSILON_NUMBERS;
		std::vector<double> coords_sum(num_data);
		std::vector<int> sort_sum(num_data);
		std::vector<int> uniques_sorted;//index of unique locations on sorted mean scale
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			coords_sum[i] = coords(i, Eigen::all).sum();
		}
		SortIndeces<double>(coords_sum, sort_sum);
		for (int i_sort = 0; i_sort < num_data; ++i_sort) {
			//find potential duplicates in coordinates for point i (= points with same mean)
			int i_actual = sort_sum[i_sort];
			unique_idx[i_actual] = (int)uniques_sorted.size();
			uniques_sorted.push_back(i_actual);
			int j_sort;
			for (j_sort = i_sort + 1; j_sort < num_data; ++j_sort) {
				//if (coords_sum[i_actual] < coords_sum[sort_sum[j_sort]]) {
				if (NumberIsSmallerThan<double>(coords_sum[i_actual], coords_sum[sort_sum[j_sort]])) {
					break;
				}
			}
			j_sort--;
			//identify true duplicates among potential duplicates
			if (j_sort > i_sort) {//more than one potential duplicates
				std::vector<int> index_data_uniques;//index that linkes every unique coordinate / random effect in unique_idx to a random effect
				index_data_uniques.push_back((int)uniques_sorted.size() - 1);
				std::vector<int> uniques_i = std::vector<int>();//index of unique locations among the potential duplicates i_sort,...,j_sort
				uniques_i.push_back(0);
				for (int ii = 1; ii < (j_sort - i_sort + 1); ++ii) {
					int ii_actual = sort_sum[i_sort + ii];
					bool is_duplicate = false;
					for (int jj = 0; jj < (int)uniques_i.size(); ++jj) {
						int jj_actual = sort_sum[i_sort + uniques_i[jj]];
						if ((coords.row(jj_actual) - coords.row(ii_actual)).squaredNorm() < EPSILON_NUMBERS_SQUARE) {
							if (ii_actual < jj_actual) {
								uniques_sorted[index_data_uniques[uniques_i[jj]]] = ii_actual;//make sure that the first appearance of a coordinate is chosen
							}
							unique_idx[ii_actual] = index_data_uniques[uniques_i[jj]];
							is_duplicate = true;
							break;
						}
					}
					if (!is_duplicate) {
						uniques_i.push_back(ii);
						unique_idx[ii_actual] = (int)uniques_sorted.size();
						index_data_uniques.push_back((int)uniques_sorted.size());
						uniques_sorted.push_back(ii_actual);
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
		for (int i = 0; i < uniques_sorted.size(); ++i) {
			inv_order_uniques[order_uniques[i]] = i;
			uniques[i] = uniques_sorted[order_uniques[i]];
		}
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			unique_idx[i] = inv_order_uniques[unique_idx[i]];
		}
	}//end DetermineUniqueDuplicateCoordsFast

}  // namespace GPBoost
