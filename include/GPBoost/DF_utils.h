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

}  // namespace GPBoost

#endif   // GPB_DF_UTIL_H_
