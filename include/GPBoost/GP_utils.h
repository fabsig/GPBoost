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
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		den_mat_t& dist);

	/*!
	* \brief Calculate distance matrix (dense matrix) (this is a placeholder which is not used, only here for template compatibility)
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1=coords1 and dist is a symmetric square matrix
	* \param taper_range Range parameter of Wendland covariance function / taper
	* \param show_number_non_zeros If true, the percentage of non-zero values is shown
	* \param[out] dist Distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1=coords2
	*/
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		double,
		bool,
		den_mat_t& dist);

	/*!
	* \brief Calculate distance matrix (sparse matrix)
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1=coords1 and dist is a symmetric square matrix
	* \param[out] dist Distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1=coords2
	*/
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		sp_mat_t& dist);

	/*!
	* \brief Calculate distance matrix (sparse matrix) when applying tapering (i.e. covariance is zero after a ceartain distances and thus the distance is not saved)
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1=coords1 and dist is a symmetric square matrix
	* \param taper_range Range parameter of Wendland covariance function / taper
	* \param show_number_non_zeros If true, the percentage of non-zero values is shown
	* \param[out] dist Distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1=coords2
	*/
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		double taper_range,
		bool show_number_non_zeros,
		sp_mat_t& dist);

}  // namespace GPBoost

#endif   // GPB_GP_UTIL_H_
