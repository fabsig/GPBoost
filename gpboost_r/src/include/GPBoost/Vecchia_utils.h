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
  void find_nearest_neighbors_Veccia(den_mat_t& dist, int num_data, int num_neighbors,
    std::vector<std::vector<int>>& nearest_neighbors);

  /*!
* \brief Finds the nearest_neighbors among the previous observations using the fast mean-distance-ordered nn search by Ra and Kim 1993
* \param coords Coordinates of observations
* \param num_data Number of observations
* \param num_neighbors Maximal number of neighbors
* \param[out] nearest_neighbor Vector with indices of nearest neighbors for every observations (length = num_data - start_at)
* \param[out] dist All distances needed for the Vecchia approxiamtion (distances between locations and their neighbors as well as distances between all neighbors) (length = num_data - start_at)
* \param start_at Index of first point for which nearest neighbors should be found (useful for prediction, otherwise = 0)
* \param end_search_at Index of last point which can be a nearest neighbor (useful for prediction when the nearest neighbors are only to be found among the observed data, otherwise = num_data - 1 (if end_search_at < 0, we set end_search_at = num_data - 1)
*/
  void find_nearest_neighbors_Veccia_fast(const den_mat_t& coords, int num_data, int num_neighbors,
    std::vector<std::vector<int>>& nearest_neighbors, std::vector<den_mat_t>& dist_obs_neighbors,
    std::vector<den_mat_t>& dist_between_neighbors, int start_at = 0, int end_search_at = -1);

/*!
* \brief Sorts vectors a and b of length n based on decreasing values of a (this is taken from the suplementary code of Finley et al. (2019), JASA)
*/
  void sort_vectors_decreasing(double* a, int* b, int n);

}  // namespace GPBoost

#endif   // GPB_VECCHIA_H_
