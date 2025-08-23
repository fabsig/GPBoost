/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2025 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_DF_UTIL_H_
#define GPB_DF_UTIL_H_

#define _USE_MATH_DEFINES // for M_SQRT1_2 and M_PI
#include <cmath>
#include <limits>       // std::numeric_limits
#include <algorithm>

//Mathematical constants usually defined in cmath
#ifndef M_PI
#define M_PI      3.141592653589793238462643383279502884 // pi
#endif
#ifndef M_SQRT2
#define M_SQRT2      1.414213562373095048801688724209698079 // sqrt(2)
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2      0.707106781186547524400844362104849039 // 1/sqrt(2)
#endif

namespace GPBoost {

	static const double M_SQRT2PI = std::sqrt(2. * M_PI);
	static const double M_LOGSQRT2PI = 0.5 * std::log(2. * M_PI);//0.91893853320467274178

	inline double sigmoid_stable(double x) {
		if (x >= 0.0) {
			const double t = std::exp(-x);
			return 1.0 / (1.0 + t);
		}
		else {
			const double t = std::exp(x);
			return t / (1.0 + t);
		}
	}

	inline double sigmoid_stable_clamped(double x) {
		double mu = sigmoid_stable(x);
		// clamp
		const double eps = 1e-12;
		if (mu < eps) mu = eps;
		if (mu > 1.0 - eps) mu = 1.0 - eps;
		return mu;
	}

	inline double softplus(double x) {
		const double a = std::fabs(x);
		return std::log1p(std::exp(-a)) + std::max(x, 0.0);
	}

	inline double normalPDF(double x) {
		return std::exp(-x * x / 2.) / M_SQRT2PI;
	}

	inline double normalLogPDF(double x) {
		return -x * x / 2. - M_LOGSQRT2PI;
	}

	inline double normalCDF(double x) {
		return 0.5 * std::erfc(-x * M_SQRT1_2);
	}

	inline double normalLogCDF(double x) {
		if (x < 0.0) {
			// Left tail: Phi(x) = 0.5 * erfc(-x/sqrt(2))
			const double e = std::erfc(-x * M_SQRT1_2);
			if (e > 0.0) return std::log(0.5) + std::log(e);
			// Extreme left tail: asymptotic approximation
			const double u = -x;
			const double u2 = u * u;
			const double series = 1.0 - 1.0 / u2 + 3.0 / (u2 * u2);
			return -0.5 * u2 - std::log(u) - 0.5 * std::log(2 * M_PI) + std::log(series);
		}
		else {
			// Right/center: log(Phi(x)) = log(1 - Q), Q = 1 - Phi(x) = 0.5 * erfc(x/sqrt(2))
			const double Q = 0.5 * std::erfc(x * M_SQRT1_2);
			if (Q == 0.0) return 0.0;
			return std::log1p(-Q);
		}
	}

	inline double InvMillsRatioNormalPhi(double x) {
		const double logphi = normalLogPDF(x);
		const double logPhi = normalLogCDF(x);
		return std::exp(logphi - logPhi);
	}

	inline double InvMillsRatioNormalOneMinusPhi(double x) {
		const double logphi = normalLogPDF(x);
		const double logQ = normalLogCDF(-x);//log(1-Phi(x)) = log(Phi(-x))
		return std::exp(logphi - logQ);
	}

	//// Inverse Mills ratios phi(x) / (1 - Phi(x)) and phi(x) / Phi(x) 
	////    with an asymptotic fallback when denom (= 1-Phi(x) or Phi(x)) underflows (which is the same for both denom = 1-Phi(x) and Phi(x))
	////     Note: this is currently not used
	//inline double InvMillsRatioNormal(double x, double pdf, double denom) {
	//    if (!std::isfinite(denom)) return std::numeric_limits<double>::quiet_NaN();
	//    if (denom > 0.0) return pdf / denom;
	//    const double u = std::fabs(x);
	//    return u + 1.0 / u + 2.0 / (u * u * u); // phi/denom approx u + 1/u + 2/u^3  with u = |x|
	//}

	/*!
	* \brief Quantile function of a normal distribution
	* \param p Probability for which the quantile is calculated
	* source: https://gist.github.com/kmpm/1211922/6b7fcd0155b23c3dc71e6f4969f2c48785371292 and http://www.wilmott.com/messageview.cfm?catid=10&threadid=38771
	*
	*     For small to moderate probabilities, algorithm referenced
	*     below is used to obtain an initial approximation which is
	*     polished with a final Newton step.
	*
	*     For very large arguments, an algorithm of Wichura is used.
	*
	*  REFERENCE
	*
	*     Beasley, J. D. and S. G. Springer (1977).
	*     Algorithm AS 111: The percentage points of the normal distribution,
	*     Applied Statistics, 26, 118-121.
	*
	*      Wichura, M.J. (1988).
	*      Algorithm AS 241: The Percentage Points of the Normal Distribution.
	*      Applied Statistics, 37, 477-484.
	*/
	double normalQF(double p);

	/*!
	* \brief Calculates the digamma function d ( log ( gamma ( x ) ) ) / dx
	* \param x Value at which the digamma function is evaluated
	* \return The value of the digamma function at x
	* source: https://people.math.sc.edu/Burkardt/cpp_src/asa103/asa103.html
	*
	*  Author:
	*
	*    Original FORTRAN77 version by Jose Bernardo.
	*    C++ version by John Burkardt with minor adaptions by Fabio Sigrist
	*
	*  Reference:
	*
	*    Jose Bernardo,
	*    Algorithm AS 103:
	*    Psi ( Digamma ) Function,
	*    Applied Statistics,
	*    Volume 25, Number 3, 1976, pages 315-317.
	*/
	double digamma(double x);

	/*!
	* \brief Calculates the trigamma function trigamma(x) = d**2 log(gamma(x)) / dx**2
	* \param x Value at which the trigamma function is evaluated
	* \return The value of the trigamma function at x
	* source: https://people.math.sc.edu/Burkardt/cpp_src/asa121/asa121.html
	*
	*  Author:
	*
	*    Original FORTRAN77 version by BE Schneider..
	*    C++ version by John Burkardt with minor adaptions by Fabio Sigrist
	*
	*  Reference:
	*
	*    BE Schneider,
	*    Algorithm AS 121:
	*    Trigamma Function,
	*    Applied Statistics,
	*    Volume 27, Number 1, pages 97-99, 1978.
	*/
	double trigamma(double x);

	/*!
	* \brief Calculates the tetragamma function trigamma(x) = d**3 log(gamma(x)) / dx**3
	* \param x Value at which the tetragamma function is evaluated
	* \return The value of the tetragamma function at x
	*/
	double tetragamma(double x);

}  // namespace GPBoost

#endif   // GPB_DF_UTIL_H_
