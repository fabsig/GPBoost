/*!
* Modified work Copyright (c) 2023 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef LIGHTGBM_NESTEROV_BOOSTING_H_
#define LIGHTGBM_NESTEROV_BOOSTING_H_

#include <LightGBM/utils/common.h>

namespace LightGBM {

	/*!
	* \brief Apply a momentum step (for Nesterov accelerated boosting)
	* \param score[out] Current scores on which the momentum is added
	* \param score_lag1[out] Lag1 of scores
	* \param score_size Number of scores
	* \param mu Neterov acceleration rate
	*/
	inline void DoOneMomentumStep(double* score,
		double* score_lag1,
		int64_t score_size,
		double mu) {
		std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>> score_momentum(score_size);
#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < score_size; ++i) {
			score_momentum[i] = (mu + 1.) * score[i] - mu * score_lag1[i];
		}
#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < score_size; ++i) {
			score_lag1[i] = score[i];
		}
#pragma omp parallel for schedule(static)
		for (int64_t i = 0; i < score_size; ++i) {
			score[i] = score_momentum[i];
		}
	}

}  // namespace LightGBM


#endif   // LIGHTGBM_NESTEROV_BOOSTING_H_
