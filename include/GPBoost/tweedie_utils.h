/*!
* Clean-room numerical utilities for compound Poisson--Gamma Tweedie densities.
* The parameterization is Var(Y | eta) = phi * exp(eta)^p, 1 < p < 2.
* The density is evaluated in the dispersion-model form f(y; mu, phi, p) = a(y, phi, p) * exp(-d(y, mu) / (2 * phi)) with the
* unit deviance d(y, mu) (Jorgensen, 1997, The Theory of Dispersion Models), i.e., a(y, phi, p) = f(y; y, phi, p) is the density
* at mu = y. In the canonical form, the normalizer and the location-dependent part are both of the order of
* y^(2-p) / ((p-1) * phi) and cancel for a concentrated density (small phi or p close to 1), which makes the log-density and, in
* particular, its derivatives inaccurate. Neither part of the dispersion-model form is large where the density is not small.
*/
#ifndef GPB_TWEEDIE_UTILS_
#define GPB_TWEEDIE_UTILS_

#include <GPBoost/DF_utils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace GPBoost {

enum class TweedieDerivativeOrder { kValue = 0, kFirst = 1 };

struct TweediePowerTransform {
	double p;
	double dp_dtheta;
};

/*!
* \brief log(a(y, phi, p)) of the dispersion-model form of the Tweedie density, i.e., the log-density at mu = y, and its derivatives
*	wrt rho = log(phi) and the transformed power theta. 'lower' and 'upper' are the range of the terms of the series
*/
struct TweedieSeriesResult {
	double log_a = 0.;
	double d_rho = 0.;
	double d_theta = 0.;
	int64_t lower = 0;
	int64_t upper = 0;
	bool converged = true;
};

/*! \brief Values f(j) of a function for the contiguous index range j = start, ..., start + values.size() - 1 */
struct TweedieIndexedValues {
	int64_t start = 1;
	std::vector<double> values;

	/*!
	* \brief Makes sure that f(lower), ..., f(upper) are stored. The stored range is extended to cover them, by at
	*	least a quarter of its length so that extending it term by term costs amortized constant time. If the extended
	*	range would hold more than max_values values, it is replaced by [lower, upper], which has to hold at most max_values
	*	values
	*/
	template <typename F>
	void Ensure(int64_t lower, int64_t upper, int64_t max_values, F f) {
		const int64_t end = start + static_cast<int64_t>(values.size());
		if (!values.empty() && lower >= start && upper < end) return;
		if (values.empty() || std::max(upper + 1, end) - std::min(lower, start) > max_values) {
			start = lower;
			values.resize(static_cast<size_t>(upper - lower + 1));
			for (int64_t j = lower; j <= upper; ++j) values[static_cast<size_t>(j - lower)] = f(j);
			return;
		}
		const int64_t growth = std::max<int64_t>(64, static_cast<int64_t>(values.size()) / 4);
		if (lower < start) {
			lower = std::max<int64_t>({ 1, std::min(lower, start - growth), std::max(upper + 1, end) - max_values });
			std::vector<double> front(static_cast<size_t>(start - lower));
			for (int64_t j = lower; j < start; ++j) front[static_cast<size_t>(j - lower)] = f(j);
			values.insert(values.begin(), front.begin(), front.end());
			start = lower;
		}
		if (upper >= end) {
			upper = std::min(std::max(upper, end - 1 + growth), start + max_values - 1);
			for (int64_t j = end; j <= upper; ++j) values.push_back(f(j));
		}
	}

	double operator[](int64_t j) const {
		return values[static_cast<size_t>(j - start)];
	}

	void Clear() {
		start = 1;
		values.clear();
	}
};

/*!
* \brief Error of Stirling's approximation, log(Gamma(x + 1)) - (x * log(x) - x + log(2 * pi * x) / 2), for x > 0. For x > 15, the
*	asymptotic series is used, which has an absolute error of at most about 2e-16 there and avoids the cancellation of the large terms
*	(Loader, 2000, Fast and accurate computation of binomial probabilities)
*/
inline double TweedieStirlingError(double x) {
	if (x <= 15.) return std::lgamma(x + 1.) - x * std::log(x) + x - M_LOGSQRT2PI - 0.5 * std::log(x);
	const double xi2 = 1. / (x * x);
	return (1. / 12. - (1. / 360. - (1. / 1260. - (1. / 1680. - xi2 / 1188.) * xi2) * xi2) * xi2) / x;
}

/*! \brief Derivative of 'TweedieStirlingError' wrt x, digamma(x + 1) - log(x) - 1 / (2 * x) */
inline double TweedieStirlingErrorDeriv(double x) {
	if (x <= 15.) return GPBoost::digamma(x + 1.) - std::log(x) - 0.5 / x;
	const double xi2 = 1. / (x * x);
	return -(1. / 12. - (3. / 360. - (5. / 1260. - (7. / 1680. - 9. * xi2 / 1188.) * xi2) * xi2) * xi2) * xi2;
}

/*!
* \brief The Stirling errors ('TweedieStirlingError') of j and j * alpha, alpha = (2-p) / (p-1), for the terms j of the series of
*	the Tweedie density. They depend on p only, not on y or phi, so that one cache can be used for all observations. Each function
*	is stored for the range of j that is in use, with at most max_values values, and not for all j = 1, ..., upper
*/
struct TweedieSpecialFunctionCache {
	int64_t max_values = 1000000;
	double p = std::numeric_limits<double>::quiet_NaN();
	double alpha = std::numeric_limits<double>::quiet_NaN();
	TweedieIndexedValues stirling_error;
	TweedieIndexedValues stirling_error_alpha;
	TweedieIndexedValues stirling_error_deriv_alpha;

	void Ensure(double power, int64_t lower, int64_t upper, bool calculate_power_derivatives) {
		lower = std::max<int64_t>(1, lower);
		if (upper < lower) return;
		if (power != p) {
			p = power;
			alpha = (2. - p) / (p - 1.);
			stirling_error_alpha.Clear();
			stirling_error_deriv_alpha.Clear();
		}
		const double a = alpha;
		stirling_error.Ensure(lower, upper, max_values, [](int64_t j) { return TweedieStirlingError(static_cast<double>(j)); });
		stirling_error_alpha.Ensure(lower, upper, max_values, [a](int64_t j) { return TweedieStirlingError(static_cast<double>(j) * a); });
		if (calculate_power_derivatives) {
			stirling_error_deriv_alpha.Ensure(lower, upper, max_values, [a](int64_t j) { return TweedieStirlingErrorDeriv(static_cast<double>(j) * a); });
		}
	}
};

inline double TweedieExpFromLog(double log_value) {
	if (log_value > std::log(std::numeric_limits<double>::max())) return std::numeric_limits<double>::infinity();
	if (log_value < std::log(std::numeric_limits<double>::denorm_min())) return 0.;
	return std::exp(log_value);
}

inline double TweedieSignedLogSum(double coefficient_a, double log_a, double coefficient_b, double log_b) {
	const double log_term_a = coefficient_a == 0. ? -std::numeric_limits<double>::infinity() : std::log(std::abs(coefficient_a)) + log_a;
	const double log_term_b = coefficient_b == 0. ? -std::numeric_limits<double>::infinity() : std::log(std::abs(coefficient_b)) + log_b;
	if (log_term_a == -std::numeric_limits<double>::infinity()) return std::copysign(TweedieExpFromLog(log_term_b), coefficient_b);
	if (log_term_b == -std::numeric_limits<double>::infinity()) return std::copysign(TweedieExpFromLog(log_term_a), coefficient_a);
	if (std::signbit(coefficient_a) == std::signbit(coefficient_b)) {
		const double maximum = std::max(log_term_a, log_term_b);
		return std::copysign(TweedieExpFromLog(maximum + std::log1p(std::exp(std::min(log_term_a, log_term_b) - maximum))), coefficient_a);
	}
	if (log_term_a == log_term_b) return 0.;
	const bool a_is_larger = log_term_a > log_term_b;
	const double maximum = a_is_larger ? log_term_a : log_term_b;
	const double minimum = a_is_larger ? log_term_b : log_term_a;
	const double magnitude = TweedieExpFromLog(maximum + std::log1p(-std::exp(minimum - maximum)));
	return std::copysign(magnitude, a_is_larger ? coefficient_a : coefficient_b);
}

/*!
* \brief Derivatives wrt eta of the location-dependent part of the Tweedie log-density ('EvaluateTweedieDevianceTerm') with
*	mu = exp(eta) and phi = exp(rho): kScore: the first derivative, (y - mu) * mu^(1-p) / phi, kInformation: the negative second
*	derivative, kDerivInformationEta: the first derivative of kInformation
*/
enum class TweedieLocationQuantity { kScore, kInformation, kDerivInformationEta };

/*! \brief log(mu^(2-p) / phi) with mu = exp(eta) and phi = exp(rho) */
inline double TweedieLogScaledA(double eta, double rho, double p) {
	return (2. - p) * eta - rho;
}

/*! \brief log(y * mu^(1-p) / phi) with mu = exp(eta) and phi = exp(rho), -inf for y = 0 */
inline double TweedieLogScaledB(double y, double eta, double rho, double p) {
	return (y > 0.) ? std::log(y) + (1. - p) * eta - rho : -std::numeric_limits<double>::infinity();
}

/*! \brief One location-dependent quantity of the Tweedie log-density from 'TweedieLogScaledA' and 'TweedieLogScaledB' */
inline double TweedieLocationQuantityFromLogScaled(TweedieLocationQuantity quantity, double log_scaled_a, double log_scaled_b, double p) {
	switch (quantity) {
	case TweedieLocationQuantity::kScore:
		return TweedieSignedLogSum(-1., log_scaled_a, 1., log_scaled_b);
	case TweedieLocationQuantity::kInformation:
		return TweedieSignedLogSum(2. - p, log_scaled_a, p - 1., log_scaled_b);
	default:// kDerivInformationEta
		return TweedieSignedLogSum((2. - p) * (2. - p), log_scaled_a, -(p - 1.) * (p - 1.), log_scaled_b);
	}
}

/*!
* \brief One location-dependent quantity of the Tweedie log-density. Callers that need several quantities at the same point
*	calculate 'TweedieLogScaledA' and 'TweedieLogScaledB' once and use 'TweedieLocationQuantityFromLogScaled'
*/
inline double EvaluateTweedieLocationQuantity(TweedieLocationQuantity quantity, double y, double eta, double rho, double p) {
	return TweedieLocationQuantityFromLogScaled(quantity, TweedieLogScaledA(eta, rho, p), TweedieLogScaledB(y, eta, rho, p), p);
}

/*! \brief Derivative of expm1(z) / z wrt z, (z * e^z - expm1(z)) / z^2, evaluated with its Taylor series for small |z|, where the terms cancel */
inline double TweedieDerivExpm1OverZ(double z) {
	if (std::abs(z) >= 0.1) return (z * std::exp(z) - std::expm1(z)) / (z * z);
	// sum_{k >= 0} (k + 1) * z^k / (k + 2)!
	double power_over_factorial = 0.5;
	double sum = 0.5;
	for (int k = 1; k < 30; ++k) {
		power_over_factorial *= z / (k + 2.);
		const double sum_next = sum + (k + 1.) * power_over_factorial;
		if (sum_next == sum) break;
		sum = sum_next;
	}
	return sum;
}

/*!
* \brief The location-dependent part -d(y, mu) / (2 * phi) of the Tweedie log-density with mu = exp(eta), phi = exp(rho), and the unit
*	deviance d(y, mu) = 2 * (y^(2-p) / ((1-p) * (2-p)) - y * mu^(1-p) / (1-p) + mu^(2-p) / (2-p)), and its derivative wrt p.
*	Its derivative wrt rho is minus its value, and its derivatives wrt eta are given by 'TweedieLocationQuantity'
*/
struct TweedieDevianceTerm {
	double value = 0.;
	double d_p = 0.;
};

inline TweedieDevianceTerm EvaluateTweedieDevianceTerm(double y, double eta, double rho, double p, bool calculate_power_derivative) {
	TweedieDevianceTerm ans;
	const double scaled_a = TweedieExpFromLog(TweedieLogScaledA(eta, rho, p));// mu^(2-p) / phi
	if (y == 0.) {
		ans.value = -scaled_a / (2. - p);
		if (calculate_power_derivative) ans.d_p = scaled_a * (eta / (2. - p) - 1. / ((2. - p) * (2. - p)));
		return ans;
	}
	const double u = std::log(y) - eta;
	if (std::abs(u) <= 30.) {
		// d(y, mu) / 2 = mu^(2-p) * h(u) with u = log(y / mu) and h(u) = e^u * expm1((1-p) * u) / (1-p) - expm1((2-p) * u) / (2-p),
		//	which does not suffer from the cancellation of the terms of d for y close to mu
		const double exp_u = std::exp(u);
		const double h = exp_u * std::expm1((1. - p) * u) / (1. - p) - std::expm1((2. - p) * u) / (2. - p);
		ans.value = -scaled_a * h;
		if (calculate_power_derivative) {
			const double dh_dp = u * u * (TweedieDerivExpm1OverZ((2. - p) * u) - exp_u * TweedieDerivExpm1OverZ((1. - p) * u));
			ans.d_p = scaled_a * (eta * h - dh_dp);
		}
		return ans;
	}
	// Far from mu, the terms of d do not cancel. They are calculated from their logarithms, as e^u could overflow
	const double log_y = std::log(y);
	const double scaled_y = TweedieExpFromLog((2. - p) * log_y - rho);// y^(2-p) / phi
	const double scaled_b = TweedieExpFromLog(TweedieLogScaledB(y, eta, rho, p));// y * mu^(1-p) / phi
	ans.value = scaled_y / ((p - 1.) * (2. - p)) - scaled_b / (p - 1.) - scaled_a / (2. - p);
	if (calculate_power_derivative) {
		ans.d_p = scaled_y / ((p - 1.) * (2. - p)) * (1. / (2. - p) - 1. / (p - 1.) - log_y) +
			scaled_a * (eta / (2. - p) - 1. / ((2. - p) * (2. - p))) + scaled_b * (eta / (p - 1.) + 1. / ((p - 1.) * (p - 1.)));
	}
	return ans;
}

/*!
* \brief Power p = (lower + upper * q) / (1 + q) and its derivative wrt theta = log(q) for q in [0, inf]. The power is kept strictly
*	between the bounds also where it rounds to one of them (q or 1 / q below the machine epsilon, including q = 0 and q = inf, which
*	an optimizer reaches where the likelihood is flat in theta), since an estimated power has to be valid as the initial value of a
*	new model
*/
inline TweediePowerTransform TransformTweediePowerFromQ(double q, double lower, double upper) {
	TweediePowerTransform ans{ std::numeric_limits<double>::quiet_NaN(), 0. };
	if (!(q >= 0.) || !(lower < upper)) return ans;
	// Written in min(q, 1 / q), which does not overflow
	if (q <= 1.) {
		ans.p = (lower + upper * q) / (1. + q);
		ans.dp_dtheta = (upper - lower) * q / ((1. + q) * (1. + q));
	}
	else {
		const double qi = 1. / q;
		ans.p = (upper + lower * qi) / (1. + qi);
		ans.dp_dtheta = (upper - lower) * qi / ((1. + qi) * (1. + qi));
	}
	ans.p = std::min(std::max(ans.p, std::nextafter(lower, upper)), std::nextafter(upper, lower));
	return ans;
}

/*!
* \brief x * log(x / m) + m - x >= 0 for x > 0 and m = exp(log_m), the deviance term of the Poisson distribution. For x close to m, where
*	the terms cancel, it is evaluated with a series (Loader, 2000, Fast and accurate computation of binomial probabilities)
*/
inline double TweedieBd0(double x, double m, double log_m) {
	if (std::abs(x - m) < 0.1 * (x + m)) {
		// (x - m) * v + 2 * x * sum_{k >= 1} v^(2k+1) / (2k+1) with v = (x - m) / (x + m)
		double v = (x - m) / (x + m);
		double sum = (x - m) * v;
		double term = 2. * x * v;
		v *= v;
		for (int k = 1; k < 1000; ++k) {
			term *= v;
			const double sum_next = sum + term / (2. * k + 1.);
			if (sum_next == sum) break;
			sum = sum_next;
		}
		return sum;
	}
	return x * (std::log(x) - log_m) + m - x;
}

/*!
* \brief Logarithm of term j of the series a(y, phi, p) = sum_j W_j / y of the Tweedie density at mu = y, up to log(alpha) / 2 - log(2 * pi),
*	which is the same for all j. With j0 = y^(2-p) / ((2-p) * phi) and alpha = (2-p) / (p-1), W_j = j0^j / j! * (alpha * j0)^(j * alpha) /
*	Gamma(j * alpha) * exp(-j0 / (p-1)), and log(W_j) = -bd0(j, j0) / (p-1) - delta(j) - delta(j * alpha) + log(alpha) / 2 - log(2 * pi) with
*	the Poisson deviance term bd0 ('TweedieBd0') and the Stirling error delta ('TweedieStirlingError'). Each part is small where W_j is not
*/
inline double TweedieSeriesLogTerm(int64_t j, double j0, double log_j0, double p, TweedieSpecialFunctionCache* cache = nullptr) {
	const double jd = static_cast<double>(j);
	const double log_term = -TweedieBd0(jd, j0, log_j0) / (p - 1.);
	if (cache != nullptr) {
		cache->Ensure(p, j, j, false);
		return log_term - cache->stirling_error[j] - cache->stirling_error_alpha[j];
	}
	return log_term - TweedieStirlingError(jd) - TweedieStirlingError(jd * (2. - p) / (p - 1.));
}

/*! \brief log(alpha) / 2 - log(2 * pi), the part of the logarithms of the terms of the series that does not depend on j (see 'TweedieSeriesLogTerm') */
inline double TweedieSeriesLogConstant(double p) {
	return 0.5 * std::log((2. - p) / (p - 1.)) - 2. * M_LOGSQRT2PI;
}

/*!
* \brief Derivative of the logarithm of a term of the series wrt p (see 'TweedieSeriesLogTerm'), bd0(j, j0) / (p-1)^2 +
*	(j - j0) * (1 / (2-p) - log(y)) / (p-1) + (x * delta'(x) - 1/2) / ((p-1) * (2-p)) with x = j * alpha. It is linear in
*	bd0(j, j0), j - j0, and x * delta'(x), and thus also gives its mean over the terms from the means of these
* \param bd0 bd0(j, j0) or its mean
* \param j_minus_j0 j - j0 or its mean
* \param x_stirling_error_deriv x * delta'(x) with x = j * alpha or its mean
*/
inline double TweedieSeriesLogTermDerivPower(double bd0, double j_minus_j0, double x_stirling_error_deriv, double log_y, double p) {
	return bd0 / ((p - 1.) * (p - 1.)) + j_minus_j0 * (1. / (2. - p) - log_y) / (p - 1.) + (x_stirling_error_deriv - 0.5) / ((p - 1.) * (2. - p));
}

/*!
* \brief log(a(y, phi, p)) of the dispersion-model form of the Tweedie density with phi = exp(rho), i.e., the log-density at mu = y,
*	and its derivatives wrt rho and the transformed power theta (dp/dtheta = 'dp_dtheta'). The series is summed over the terms j
*	that are within eps^2 of the largest one, wherever they are, and the result is not converged if more than max_terms terms,
*	or more than the max_values of the cache, are needed
*/
inline TweedieSeriesResult EvaluateTweedieLogNormalizer(double y, double rho, double p, double dp_dtheta, TweedieDerivativeOrder order,
	bool calculate_power_derivatives, int64_t max_terms = 1000000, TweedieSpecialFunctionCache* cache = nullptr) {
	TweedieSeriesResult ans;
	if (y == 0.) return ans;
	if (!(y > 0.) || !std::isfinite(y) || !std::isfinite(rho) || !(p > 1. && p < 2.)) {
		ans.converged = false;
		return ans;
	}
	const double log_y = std::log(y);
	// The largest term is close to j0 = y^(2-p) / ((2-p) * phi)
	const double log_j0 = (2. - p) * log_y - rho - std::log(2. - p);
	// j has to be an exact double
	if (!std::isfinite(log_j0) || log_j0 > 53. * std::log(2.)) {
		ans.converged = false;
		return ans;
	}
	const double j0 = std::exp(log_j0);
	int64_t mode = std::max<int64_t>(1, static_cast<int64_t>(std::floor(j0)));
	const double log_tol = 2. * std::log(std::numeric_limits<double>::epsilon());
	// Every range that is requested from the cache has at most max_terms values
	if (cache != nullptr) max_terms = std::min(max_terms, cache->max_values);
	// The terms within log_tol of the largest one are within about half_width = sqrt(2 |log_tol| (p - 1) mode) of it,
	// since -d^2 log(W_j) / dj^2 is approximately 1 / ((p - 1) j). This decreases with j, so that the terms above the
	// largest one alone span at least about half_width, and more than max_terms terms are needed if it exceeds max_terms
	const double half_width = std::sqrt(2. * std::abs(log_tol) * (p - 1.) * static_cast<double>(mode));
	if (half_width > static_cast<double>(max_terms)) {
		ans.converged = false;
		return ans;
	}
	if (cache != nullptr) {
		const int64_t prefill_half_width = std::min<int64_t>(static_cast<int64_t>(std::ceil(1.1 * half_width)) + 16, (max_terms - 1) / 2);
		cache->Ensure(p, mode - prefill_half_width, mode + prefill_half_width, false);
	}
	// Move to the discrete maximum. The continuous approximation is normally within a few terms.
	double mode_term = TweedieSeriesLogTerm(mode, j0, log_j0, p, cache);
	while (true) {
		const double next = TweedieSeriesLogTerm(mode + 1, j0, log_j0, p, cache);
		if (next <= mode_term) break;
		mode_term = next;
		++mode;
	}
	while (mode > 1) {
		const double prev = TweedieSeriesLogTerm(mode - 1, j0, log_j0, p, cache);
		if (prev <= mode_term) break;
		mode_term = prev;
		--mode;
	}
	int64_t lower = mode;
	while (lower > 1 && mode - lower < max_terms && TweedieSeriesLogTerm(lower - 1, j0, log_j0, p, cache) - mode_term > log_tol) --lower;
	int64_t upper = mode;
	while (upper - lower < max_terms && TweedieSeriesLogTerm(upper + 1, j0, log_j0, p, cache) - mode_term > log_tol) ++upper;
	if (upper - lower + 1 > max_terms) {
		ans.converged = false;
		return ans;
	}
	ans.lower = lower;
	ans.upper = upper;
	const bool power_derivatives = calculate_power_derivatives && order != TweedieDerivativeOrder::kValue;
	if (cache != nullptr) cache->Ensure(p, lower, upper, power_derivatives);
	const int64_t n_terms = upper - lower + 1;
	std::vector<double> terms(static_cast<size_t>(n_terms));
	double max_log = -std::numeric_limits<double>::infinity();
	for (int64_t k = 0; k < n_terms; ++k) {
		terms[static_cast<size_t>(k)] = TweedieSeriesLogTerm(lower + k, j0, log_j0, p, cache);
		max_log = std::max(max_log, terms[static_cast<size_t>(k)]);
	}
	double sum_w = 0.;
	for (double term : terms) sum_w += std::exp(term - max_log);
	ans.log_a = max_log + std::log(sum_w) + TweedieSeriesLogConstant(p) - log_y;
	if (order == TweedieDerivativeOrder::kValue) return ans;

	// The derivatives are the means of the derivatives of the logarithms of the terms, weighted by the terms
	const double alpha = (2. - p) / (p - 1.);
	double mean_j_minus_j0 = 0., mean_bd0 = 0., mean_x_stirling_error_deriv = 0.;
	for (int64_t k = 0; k < n_terms; ++k) {
		const int64_t j = lower + k;
		const double jd = static_cast<double>(j);
		const double w = std::exp(terms[static_cast<size_t>(k)] - max_log) / sum_w;
		mean_j_minus_j0 += w * (jd - j0);
		if (power_derivatives) {
			const double x = jd * alpha;
			const double stirling_error_deriv = cache == nullptr ? TweedieStirlingErrorDeriv(x) : cache->stirling_error_deriv_alpha[j];
			mean_bd0 += w * TweedieBd0(jd, j0, log_j0);
			mean_x_stirling_error_deriv += w * x * stirling_error_deriv;
		}
	}
	ans.d_rho = -mean_j_minus_j0 / (p - 1.);
	if (power_derivatives) {
		ans.d_theta = dp_dtheta * TweedieSeriesLogTermDerivPower(mean_bd0, mean_j_minus_j0, mean_x_stirling_error_deriv, log_y, p);
	}
	return ans;
}

/*!
* \brief Joint log-density of (Y, N) of the compound Poisson--Gamma representation, N ~ Poisson(mu^(2-p) / (phi * (2-p))),
*	Y | N = n ~ Gamma(n * (2-p) / (p-1), scale = phi * (p-1) * mu^(p-1)), without the location-dependent part, which equals the one
*	of the marginal Tweedie density ('EvaluateTweedieDevianceTerm'). The remaining part is the n-th term of the series of a(y, phi, p)
*	of 'EvaluateTweedieLogNormalizer', log(W_n / y), i.e., the joint log-density at mu = y. For n = 0, y has to be 0 and the result
*	is 0. The derivatives are wrt rho = log(phi) and the transformed power theta (dp/dtheta = 'dp_dtheta'), with the same fields
*	and conventions as 'EvaluateTweedieLogNormalizer'
*/
inline TweedieSeriesResult EvaluateTweedieJointLogNormalizer(double y, double n, double rho, double p,
	double dp_dtheta, TweedieDerivativeOrder order, bool calculate_power_derivatives) {
	TweedieSeriesResult ans;
	ans.lower = ans.upper = static_cast<int64_t>(n);
	if (n == 0.) {
		ans.converged = (y == 0.);
		return ans;
	}
	if (!(y > 0.) || !std::isfinite(y) || !std::isfinite(rho) || !(p > 1. && p < 2.) || !(n > 0.)) {
		ans.converged = false;
		return ans;
	}
	const double log_y = std::log(y);
	const double log_j0 = (2. - p) * log_y - rho - std::log(2. - p);
	const double j0 = std::exp(log_j0);
	if (!std::isfinite(j0)) {
		ans.converged = false;
		return ans;
	}
	ans.log_a = TweedieSeriesLogTerm(static_cast<int64_t>(n), j0, log_j0, p) + TweedieSeriesLogConstant(p) - log_y;
	if (order == TweedieDerivativeOrder::kValue) return ans;
	ans.d_rho = -(n - j0) / (p - 1.);
	if (calculate_power_derivatives) {
		const double x = n * (2. - p) / (p - 1.);
		ans.d_theta = dp_dtheta * TweedieSeriesLogTermDerivPower(TweedieBd0(n, j0, log_j0), n - j0, x * TweedieStirlingErrorDeriv(x), log_y, p);
	}
	return ans;
}

}  // namespace GPBoost

#endif  // GPB_TWEEDIE_UTILS_
