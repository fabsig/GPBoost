/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/GP_utils.h>
#include <mutex>
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
		for (int i = 1; i < num_data; ++i) {//identify duplicates in coordinates
			bool is_duplicate = false;
			for (int j = 0; j < (int)uniques.size(); ++j) {
				if ((coords.row(uniques[j]) - coords.row(i)).norm() == 0.) {
					unique_idx.push_back(j);
					is_duplicate = true;
					break;
				}
			}
			if (!is_duplicate) {
				unique_idx.push_back((int)uniques.size());
				uniques.push_back(i);
			}
		}
	}//end DetermineUniqueDuplicateCoords

}  // namespace GPBoost
