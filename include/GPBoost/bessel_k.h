/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2026 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_BESSEL_K_
#define GPB_BESSEL_K_

#include <cmath>

namespace GPBoost {

	/*!
	* \brief Modified Bessel function of the second kind K_nu(x) for nu >= 0 and x > 0.
	*		With nu = mu + n, |mu| <= 1/2, and an integer n, K_mu(x) and K_{mu+1}(x) are calculated with the series of
	*		Temme (1975, J. Comput. Phys. 19, 324-337) for x <= 2 and with the continued fraction of Steed's method
	*		(Thompson and Barnett, 1987, Comput. Phys. Commun. 47, 245-257) for x > 2, and K_nu(x) follows from the
	*		upward recurrence K_{mu+k+1}(x) = K_{mu+k-1}(x) + 2 (mu + k) / x * K_{mu+k}(x). The number of recurrence steps
	*		grows with nu; the Matern correlation functions use only orders up to 1
	* \param exponentially_scaled If true, exp(x) K_nu(x) is returned, which does not underflow for large x
	*/
	inline double BesselK(double nu,
		double x,
		bool exponentially_scaled) {
		const double pi = 3.14159265358979323846;
		const double eps = 1e-16;
		const int max_iterations = 10000;
		const int n = (int)(nu + 0.5);
		const double mu = nu - n;// in [-1/2, 1/2)
		const double mu_sq = mu * mu;
		double k_mu, k_mu1;// K_mu(x) and K_{mu+1}(x)
		if (x <= 2.) {
			// Taylor coefficients c_0, c_1, ... of 1 / Gamma(1 + t) around t = 0 (computed with mpmath): gamma_1 =
			//	(1 / Gamma(1 - mu) - 1 / Gamma(1 + mu)) / (2 mu) = -(c_1 + c_3 mu^2 + ...) and gamma_2 = (1 / Gamma(1 - mu) +
			//	1 / Gamma(1 + mu)) / 2 = c_0 + c_2 mu^2 + ..., which avoids the cancellation of these differences for small mu
			static const double coef_odd[13] = { 0.57721566490153286061, -0.042002635034095235529, -0.042197734555544336748,
				0.0072189432466630995424, -0.00021524167411495097282, -0.000020134854780788238656, 1.1330272319816958824e-6,
				6.1160951044814158179e-9, -1.1812745704870201446e-9, 7.782263439905071254e-12, 5.100370287454475979e-13,
				-5.3481225394230179824e-15, -1.1812593016974587695e-16 };
			static const double coef_even[13] = { 1., -0.65587807152025388108, 0.1665386113822914895, -0.0096219715278769735621,
				-0.0011651675918590651121, 0.00012805028238811618615, -1.2504934821426706573e-6, -2.0563384169776071035e-7,
				5.0020076444692229301e-9, 1.0434267116911005105e-10, -3.6968056186422057082e-12, -2.0583260535665067832e-14,
				1.2267786282382607902e-15 };
			double gamma_1 = coef_odd[12];
			double gamma_2 = coef_even[12];
			for (int k = 11; k >= 0; --k) {
				gamma_1 = gamma_1 * mu_sq + coef_odd[k];
				gamma_2 = gamma_2 * mu_sq + coef_even[k];
			}
			gamma_1 = -gamma_1;
			const double inv_gamma_plus = gamma_2 - mu * gamma_1;// 1 / Gamma(1 + mu)
			const double inv_gamma_minus = gamma_2 + mu * gamma_1;// 1 / Gamma(1 - mu)
			const double log_2_x = std::log(2. / x);
			const double sigma = mu * log_2_x;
			const double pi_mu = pi * mu;
			const double factor_sin = (pi_mu == 0.) ? 1. : pi_mu / std::sin(pi_mu);
			const double factor_sinh = (sigma == 0.) ? 1. : std::sinh(sigma) / sigma;
			double f = factor_sin * (gamma_1 * std::cosh(sigma) + gamma_2 * factor_sinh * log_2_x);
			const double exp_sigma = std::exp(sigma);// (2 / x)^mu
			double p = 0.5 * exp_sigma / inv_gamma_plus;
			double q = 0.5 / (exp_sigma * inv_gamma_minus);
			double c = 1.;
			const double x_sq_quarter = x * x / 4.;
			double sum = f;
			double sum1 = p;
			for (int k = 1; k < max_iterations; ++k) {
				f = (k * f + p + q) / (k * k - mu_sq);
				c *= x_sq_quarter / k;
				p /= k - mu;
				q /= k + mu;
				const double term = c * f;
				sum += term;
				sum1 += c * (p - k * f);
				if (std::abs(term) < eps * std::abs(sum)) {
					break;
				}
			}
			k_mu = sum;
			k_mu1 = sum1 * 2. / x;
			if (exponentially_scaled) {
				const double exp_x = std::exp(x);
				k_mu *= exp_x;
				k_mu1 *= exp_x;
			}
		}
		else {
			const double a1 = 0.25 - mu_sq;
			double a = -a1;
			double b = 2. * (1. + x);
			double d = 1. / b;
			double delta_h = d;
			double h = d;
			double q1 = 0.;
			double q2 = 1.;
			double q = a1;
			double c = a1;
			double s = 1. + q * delta_h;
			for (int k = 2; k < max_iterations; ++k) {
				a -= 2. * (k - 1);
				c = -a * c / k;
				const double q_new = (q1 - b * q2) / a;
				q1 = q2;
				q2 = q_new;
				q += c * q_new;
				b += 2.;
				d = 1. / (b + a * d);
				delta_h = (b * d - 1.) * delta_h;
				h += delta_h;
				const double delta_s = q * delta_h;
				s += delta_s;
				if (std::abs(delta_s) < eps * std::abs(s)) {
					break;
				}
			}
			double exp_minus_x = 1.;
			if (!exponentially_scaled) {
				exp_minus_x = std::exp(-x);
			}
			k_mu = std::sqrt(pi / (2. * x)) * exp_minus_x / s;
			k_mu1 = k_mu * (mu + x + 0.5 - a1 * h) / x;
		}
		for (int k = 1; k <= n; ++k) {
			const double k_next = (mu + k) * 2. / x * k_mu1 + k_mu;
			k_mu = k_mu1;
			k_mu1 = k_next;
		}
		return(k_mu);
	}//end BesselK

}  // namespace GPBoost

#endif   // GPB_BESSEL_K_
