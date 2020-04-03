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
  void DetermineUniqueDuplicateCoords(const den_mat_t& coords, data_size_t num_data,
    std::vector<int>& uniques, std::vector<int>& unique_idx);

  /*!
  * \brief Calculate distance matrix
  * \param coords Coordinates
  * \param[out] dist Distances between all pairs of coordinates (rows in coords)
  */
  void CalculateDistances(const den_mat_t& coords, den_mat_t& dist);

}  // namespace GPBoost

#endif   // GPB_GP_UTIL_H_
