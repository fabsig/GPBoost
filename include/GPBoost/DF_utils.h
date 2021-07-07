/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_DF_UTIL_H_
#define GPB_DF_UTIL_H_

#define _USE_MATH_DEFINES // for M_SQRT1_2 and M_PI
#include <cmath>
#include <limits>       // std::numeric_limits

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
	static const double M_LOGSQRT2PI = 0.5 * std::log(2. * M_PI);

	double normalPDF(double value);

	double normalLogPDF(double value);

	double normalCDF(double value);

	/*
	* \brief Logarithm of normal CDF
	* Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier
	*
	* For a > -20, use the existing normalCDF and take a log.
	* for a <= -20, we use the Taylor series approximation of erf to compute
	* the log CDF directly. The Taylor series consists of two parts which we will name "left"
	* and "right" accordingly.  The right part involves a summation which we compute until the
	* difference in terms falls below the machine-specific EPSILON.
	*
	* \Phi(z) &=&
	*   \frac{e^{-z^2/2}}{-z\sqrt{2\pi}}  * [1 +  \sum_{n=1}^{N-1}  (-1)^n \frac{(2n-1)!!}{(z^2)^n}]
	*   + O(z^{-2N+2})
	*   = [\mbox{LHS}] * [\mbox{RHS}] + \mbox{error}.
	*
	*/
	double normalLogCDF(double value);

}  // namespace GPBoost

#endif   // GPB_DF_UTIL_H_
