/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_GP_UTIL_H_
#define GPB_GP_UTIL_H_
#include <memory>
#include <GPBoost/type_defs.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;

namespace GPBoost {

	/*!
	* \brief Determine unique locations and duplicates in coordinates
	* \param coords Coordinates
	* \param num_data Number of data points
	* \param[out] uniques Unique coordinates / points
	* \param[out] unique_idx Every point has an index refering to the corresponding unique coordinates / point. Used for constructing incidence matrix Z_ if there are duplicates
	*/
	void DetermineUniqueDuplicateCoords(const den_mat_t& coords,
		data_size_t num_data,
		std::vector<int>& uniques,
		std::vector<int>& unique_idx);

	/*!
	* \brief Calculate distance matrix (dense matrix)
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1=coords1 and dist is a symmetric square matrix
	* \param[out] dist Distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1=coords2
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		den_mat_t& dist) {
		dist = den_mat_t(coords2.rows(), coords1.rows());
		dist.setZero();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < coords2.rows(); ++i) {
			int first_j = 0;
			if (only_one_set_of_coords) {
				dist(i, i) = 0.;
				first_j = i + 1;
			}
			for (int j = first_j; j < coords1.rows(); ++j) {
				dist(i, j) = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
			}
		}
		if (only_one_set_of_coords) {
			dist.triangularView<Eigen::StrictlyLower>() = dist.triangularView<Eigen::StrictlyUpper>().transpose();
		}
	}//end CalculateDistances (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		T_mat& dist) {
		std::vector<Triplet_t> triplets;
		int n_max_entry;
		if (only_one_set_of_coords) {
			n_max_entry = (int)(coords1.rows() - 1) * (int)coords2.rows();
		}
		else {
			n_max_entry = (int)coords1.rows() * (int)coords2.rows();
		}
		triplets.reserve(n_max_entry);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < coords2.rows(); ++i) {
			int first_j = 0;
			if (only_one_set_of_coords) {
#pragma omp critical
				{
					triplets.emplace_back(i, i, 0.);
				}
				first_j = i + 1;
			}
			for (int j = first_j; j < coords1.rows(); ++j) {
				double dist_i_j = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
#pragma omp critical
				{
					triplets.emplace_back(i, j, dist_i_j);
					if (only_one_set_of_coords) {
						triplets.emplace_back(j, i, dist_i_j);
					}
				}
			}
		}
		dist = T_mat(coords2.rows(), coords1.rows());
		dist.setFromTriplets(triplets.begin(), triplets.end());
	}//end CalculateDistances (sparse)

	/*!
	* \brief Calculate distance matrix (dense matrix) when applying is tapering (this is a placeholder which is not used, only here for template compatibility)
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1=coords1 and dist is a symmetric square matrix
	* \param taper_range Range parameter of Wendland covariance function / taper
	* \param show_number_non_zeros If true, the percentage of non-zero values is shown
	* \param[out] dist Distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1=coords2
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void CalculateDistancesTapering(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		double,
		bool,
		den_mat_t& dist) {
		CalculateDistances<T_mat>(coords1, coords2, only_one_set_of_coords, dist);
	}//end CalculateDistancesTapering (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void CalculateDistancesTapering(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		double taper_range,
		bool show_number_non_zeros,
		T_mat& dist) {
		std::vector<Triplet_t> triplets;
		int n_max_entry;
		if (only_one_set_of_coords) {
			n_max_entry = 30 * (int)coords1.rows();
		}
		else {
			n_max_entry = 10 * (int)coords1.rows() + 10 * (int)coords2.rows();
		}
		triplets.reserve(n_max_entry);
		int non_zeros = 0;
#pragma omp parallel for schedule(static) reduction(+:non_zeros)
		for (int i = 0; i < coords2.rows(); ++i) {
			int first_j = 0;
			if (only_one_set_of_coords) {
#pragma omp critical
				{
					triplets.emplace_back(i, i, 0.);
				}
				first_j = i + 1;
			}
			for (int j = first_j; j < coords1.rows(); ++j) {
				double dist_i_j = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
				if (dist_i_j < taper_range) {
					if (show_number_non_zeros) {
						non_zeros += 1;
					}
#pragma omp critical
					{
						triplets.emplace_back(i, j, dist_i_j);
						if (only_one_set_of_coords) {
							triplets.emplace_back(j, i, dist_i_j);
						}
					}
				}
			}
		}
		dist = T_mat(coords2.rows(), coords1.rows());
		dist.setFromTriplets(triplets.begin(), triplets.end());
		dist.makeCompressed();
		if (show_number_non_zeros) {
			double prct_non_zero;
			if (only_one_set_of_coords) {
				prct_non_zero = (2. * non_zeros + coords1.rows()) / coords1.rows() / coords1.rows() * 100.;
				int num_non_zero_row = (2 * non_zeros + (int)coords1.rows()) / (int)coords1.rows();
				Log::REInfo("Average number of non-zero entries per row in covariance matrix: %d (%g %%)", num_non_zero_row, prct_non_zero);
			}
			else {
				int num_non_zero = non_zeros;
				prct_non_zero = non_zeros / coords1.rows() / coords2.rows() * 100.;
				Log::REInfo("Number of non-zero entries in covariance matrix: %d (%g %%)", num_non_zero, prct_non_zero);
			}
		}
	}//end CalculateDistancesTapering (sparse)

}  // namespace GPBoost

#endif   // GPB_GP_UTIL_H_
