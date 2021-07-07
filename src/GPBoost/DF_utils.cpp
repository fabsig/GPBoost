/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/DF_utils.h>

namespace GPBoost {

	double normalPDF(double value) {
		return std::exp(-value * value / 2.) / M_SQRT2PI;
	}

	double normalLogPDF(double value) {
		return -value * value / 2. - M_LOGSQRT2PI;
	}

	double normalCDF(double value) {
		double x, y, z;
		x = value * M_SQRT1_2;
		z = std::fabs(x);
		if (z < M_SQRT1_2) {
			y = 0.5 + 0.5 * std::erf(x);
		}
		else {
			y = 0.5 * std::erfc(z);
			if (x > 0) {
				y = 1.0 - y;
			}
		}
		return y;
	}

	// Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier
	double normalLogCDF(double value) {
		double log_LHS,		/* we compute the left hand side of the approx (LHS) in one shot */
			last_total = 0,		/* variable used to check for convergence */
			right_hand_side = 1,	/* includes first term from the RHS summation */
			numerator = 1,		/* numerator for RHS summand */
			denom_factor = 1,	/* use reciprocal for denominator to avoid division */
			denom_cons = 1.0 / (value * value);	/* the precomputed division we use to adjust the denominator */
		long sign = 1, i = 0;
		if (value > 6) {
			return -normalCDF(-value);     /* log(1+x) \approx x */
		}
		if (value > -20) {
			return std::log(normalCDF(value));
		}
		log_LHS = -0.5 * value * value - std::log(-value) - M_LOGSQRT2PI;
		while (std::fabs(last_total - right_hand_side) > std::numeric_limits<double>::epsilon()) {
			i += 1;
			last_total = right_hand_side;
			sign = -sign;
			denom_factor *= denom_cons;
			numerator *= 2 * i - 1;
			right_hand_side += sign * numerator * denom_factor;

		}
		return log_LHS + std::log(right_hand_side);
	}

}  // namespace GPBoost
