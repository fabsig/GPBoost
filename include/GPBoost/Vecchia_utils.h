/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_VECCHIA_H_
#define GPB_VECCHIA_H_
#include <memory>
#include <GPBoost/type_defs.h>

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
		RNG_t& gen);

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

}  // namespace GPBoost

#endif   // GPB_VECCHIA_H_
