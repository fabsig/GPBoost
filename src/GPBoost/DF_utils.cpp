/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/DF_utils.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;

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

    double normalQF(double p) {
        CHECK(p > 0.0 && p < 1.0);
        double r, val;
        const double q = p - 0.5;
        if (std::abs(q) <= .425) {
            r = .180625 - q * q;
            val =
                q * (((((((r * 2509.0809287301226727 +
                    33430.575583588128105) * r + 67265.770927008700853) * r +
                    45921.953931549871457) * r + 13731.693765509461125) * r +
                    1971.5909503065514427) * r + 133.14166789178437745) * r +
                    3.387132872796366608)
                / (((((((r * 5226.495278852854561 +
                    28729.085735721942674) * r + 39307.89580009271061) * r +
                    21213.794301586595867) * r + 5394.1960214247511077) * r +
                    687.1870074920579083) * r + 42.313330701600911252) * r + 1);
        }
        else {
            if (q > 0) {
                r = 1 - p;
            }
            else {
                r = p;
            }

            r = std::sqrt(-std::log(r));

            if (r <= 5)
            {
                r += -1.6;
                val = (((((((r * 7.7454501427834140764e-4 +
                    .0227238449892691845833) * r + .24178072517745061177) *
                    r + 1.27045825245236838258) * r +
                    3.64784832476320460504) * r + 5.7694972214606914055) *
                    r + 4.6303378461565452959) * r +
                    1.42343711074968357734)
                    / (((((((r *
                        1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                        r + .0151986665636164571966) * r +
                        .14810397642748007459) * r + .68976733498510000455) *
                        r + 1.6763848301838038494) * r +
                        2.05319162663775882187) * r + 1);
            }
            else { /* very close to  0 or 1 */
                r += -5;
                val = (((((((r * 2.01033439929228813265e-7 +
                    2.71155556874348757815e-5) * r +
                    .0012426609473880784386) * r + .026532189526576123093) *
                    r + .29656057182850489123) * r +
                    1.7848265399172913358) * r + 5.4637849111641143699) *
                    r + 6.6579046435011037772)
                    / (((((((r *
                        2.04426310338993978564e-15 + 1.4215117583164458887e-7) *
                        r + 1.8463183175100546818e-5) * r +
                        7.868691311456132591e-4) * r + .0148753612908506148525)
                        * r + .13692988092273580531) * r +
                        .59983220655588793769) * r + 1);
            }

            if (q < 0.0) {
                val = -val;
            }
        }

        return val;
    }

    double digamma(double x) {
        static double c = 8.5;
        static double euler_mascheroni = 0.57721566490153286060;
        double r;
        double value;
        double x2;
        CHECK(x > 0);
        //
        //  Use approximation for small argument.
        //
        if (x <= 0.000001)
        {
            value = -euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x;
            return value;
        }
        //
        //  Reduce to DIGAMA(X + N).
        //
        value = 0.0;
        x2 = x;
        while (x2 < c)
        {
            value = value - 1.0 / x2;
            x2 = x2 + 1.0;
        }
        //
        //  Use Stirling's (actually de Moivre's) expansion.
        //
        r = 1.0 / x2;
        value = value + log(x2) - 0.5 * r;

        r = r * r;

        value = value
            - r * (1.0 / 12.0
                - r * (1.0 / 120.0
                    - r * (1.0 / 252.0
                        - r * (1.0 / 240.0
                            - r * (1.0 / 132.0)))));

        return value;
    }

}  // namespace GPBoost
