/*!
* Clean-room numerical utilities for compound Poisson--Gamma Tweedie densities.
* The parameterization is Var(Y | eta) = phi * exp(eta)^p, 1 < p < 2.
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

enum class TweedieDerivativeOrder { kValue = 0, kFirst = 1, kSecond = 2 };

struct TweediePowerTransform {
	double p;
	double dp_dtheta;
	double d2p_dtheta2;
};

struct TweedieSeriesResult {
	double log_a = 0.;
	double d_rho = 0.;
	double d_theta = 0.;
	double d2_rho = 0.;
	double d2_rho_theta = 0.;
	double d2_theta = 0.;
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
* \brief The special functions of the terms j of the series of the Tweedie normalizer. They depend on p only, not on y
*	or phi, so that one cache can be used for all observations. Each function is stored for the range of j that is in
*	use, with at most max_values values, and not for all j = 1, ..., upper
*/
struct TweedieSpecialFunctionCache {
	int64_t max_values = 1000000;
	double p = std::numeric_limits<double>::quiet_NaN();
	double alpha = std::numeric_limits<double>::quiet_NaN();
	TweedieIndexedValues log_factorial;
	TweedieIndexedValues log_gamma_alpha;
	TweedieIndexedValues digamma_alpha;
	TweedieIndexedValues trigamma_alpha;

	void Ensure(double power, int64_t lower, int64_t upper, TweedieDerivativeOrder order, bool calculate_power_derivatives) {
		lower = std::max<int64_t>(1, lower);
		if (upper < lower) return;
		if (power != p) {
			p = power;
			alpha = (2. - p) / (p - 1.);
			log_gamma_alpha.Clear();
			digamma_alpha.Clear();
			trigamma_alpha.Clear();
		}
		const double a = alpha;
		log_factorial.Ensure(lower, upper, max_values, [](int64_t j) { return std::lgamma(static_cast<double>(j) + 1.); });
		log_gamma_alpha.Ensure(lower, upper, max_values, [a](int64_t j) { return std::lgamma(static_cast<double>(j) * a); });
		if (calculate_power_derivatives && order != TweedieDerivativeOrder::kValue) {
			digamma_alpha.Ensure(lower, upper, max_values, [a](int64_t j) { return GPBoost::digamma(static_cast<double>(j) * a); });
			if (order == TweedieDerivativeOrder::kSecond) {
				trigamma_alpha.Ensure(lower, upper, max_values, [a](int64_t j) { return GPBoost::trigamma(static_cast<double>(j) * a); });
			}
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
* \brief The location-dependent quantities of the Tweedie log-density with mu = exp(eta) and phi = exp(rho):
*	kCanonical: the part of the log-density that depends on eta, (y * mu^(1-p) / (1-p) - mu^(2-p) / (2-p)) / phi,
*	kScore: its first derivative wrt eta, kInformation: its negative second derivative wrt eta,
*	kDerivInformationEta: the first derivative of kInformation wrt eta
*/
enum class TweedieLocationQuantity { kCanonical, kScore, kInformation, kDerivInformationEta };

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
	case TweedieLocationQuantity::kCanonical:
		return TweedieSignedLogSum(-1. / (2. - p), log_scaled_a, -1. / (p - 1.), log_scaled_b);
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

inline TweediePowerTransform TransformTweediePowerFromQ(double q, double lower, double upper) {
	TweediePowerTransform ans{ std::numeric_limits<double>::quiet_NaN(), 0., 0. };
	if (!(q > 0.) || !std::isfinite(q) || !(lower < upper)) return ans;
	if (q <= 1.) ans.p = (lower + upper * q) / (1. + q);
	else {
		const double qi = 1. / q;
		ans.p = (upper + lower * qi) / (1. + qi);
	}
	ans.dp_dtheta = (ans.p - lower) * (upper - ans.p) / (upper - lower);
	ans.d2p_dtheta2 = ans.dp_dtheta * (lower + upper - 2. * ans.p) / (upper - lower);
	return ans;
}

inline double TweedieSeriesLogTerm(int64_t j, double log_y, double rho, double p, TweedieSpecialFunctionCache* cache = nullptr) {
	const double alpha = (2. - p) / (p - 1.);
	const double jd = static_cast<double>(j);
	if (cache != nullptr) {
		cache->Ensure(p, j, j, TweedieDerivativeOrder::kValue, false);
		return jd * alpha * (log_y - std::log(p - 1.)) - jd * (alpha + 1.) * rho -
			jd * std::log(2. - p) - cache->log_factorial[j] - cache->log_gamma_alpha[j];
	}
	return jd * alpha * (log_y - std::log(p - 1.)) - jd * (alpha + 1.) * rho -
		jd * std::log(2. - p) - std::lgamma(jd + 1.) - std::lgamma(jd * alpha);
}

/*!
* \brief log(a(y, phi, p)) of the Tweedie density with phi = exp(rho), and its derivatives wrt rho and the transformed
*	power theta. The series is summed over the terms j that are within eps^2 of the largest one, wherever they are,
*	and the result is not converged if more than max_terms terms, or more than the max_values of the cache, are needed
*/
inline TweedieSeriesResult EvaluateTweedieLogNormalizer(double y, double rho, double p,
	double dp_dtheta, double d2p_dtheta2, TweedieDerivativeOrder order,
	bool calculate_power_derivatives, int64_t max_terms = 1000000, TweedieSpecialFunctionCache* cache = nullptr) {
	TweedieSeriesResult ans;
	if (y == 0.) return ans;
	if (!(y > 0.) || !std::isfinite(y) || !std::isfinite(rho) || !(p > 1. && p < 2.)) {
		ans.converged = false;
		return ans;
	}
	const double log_y = std::log(y);
	const double log_mode = (2. - p) * log_y - rho - std::log(2. - p);
	// j and j * alpha have to be exact doubles
	if (!std::isfinite(log_mode) || log_mode > 53. * std::log(2.)) {
		ans.converged = false;
		return ans;
	}
	int64_t mode = std::max<int64_t>(1, static_cast<int64_t>(std::floor(std::exp(log_mode))));
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
		cache->Ensure(p, mode - prefill_half_width, mode + prefill_half_width, TweedieDerivativeOrder::kValue, false);
	}
	// Move to the discrete maximum. The continuous approximation is normally within a few terms.
	double mode_term = TweedieSeriesLogTerm(mode, log_y, rho, p, cache);
	while (true) {
		const double next = TweedieSeriesLogTerm(mode + 1, log_y, rho, p, cache);
		if (next <= mode_term) break;
		mode_term = next;
		++mode;
	}
	while (mode > 1) {
		const double prev = TweedieSeriesLogTerm(mode - 1, log_y, rho, p, cache);
		if (prev <= mode_term) break;
		mode_term = prev;
		--mode;
	}
	int64_t lower = mode;
	while (lower > 1 && mode - lower < max_terms && TweedieSeriesLogTerm(lower - 1, log_y, rho, p, cache) - mode_term > log_tol) --lower;
	int64_t upper = mode;
	while (upper - lower < max_terms && TweedieSeriesLogTerm(upper + 1, log_y, rho, p, cache) - mode_term > log_tol) ++upper;
	if (upper - lower + 1 > max_terms) {
		ans.converged = false;
		return ans;
	}
	ans.lower = lower;
	ans.upper = upper;
	if (cache != nullptr) cache->Ensure(p, lower, upper, order, calculate_power_derivatives);
	const int64_t n_terms = upper - lower + 1;
	std::vector<double> terms(static_cast<size_t>(n_terms));
	double max_log = -std::numeric_limits<double>::infinity();
	for (int64_t k = 0; k < n_terms; ++k) {
		terms[static_cast<size_t>(k)] = TweedieSeriesLogTerm(lower + k, log_y, rho, p, cache);
		max_log = std::max(max_log, terms[static_cast<size_t>(k)]);
	}
	double sum_w = 0.;
	for (double term : terms) sum_w += std::exp(term - max_log);
	ans.log_a = max_log + std::log(sum_w) - log_y;
	if (order == TweedieDerivativeOrder::kValue) return ans;

	const double alpha = (2. - p) / (p - 1.);
	const double alpha_p = -1. / ((p - 1.) * (p - 1.));
	const double alpha_pp = 2. / ((p - 1.) * (p - 1.) * (p - 1.));
	double er = 0., et = 0., er2 = 0., ert = 0., et2 = 0., e_second_rt = 0., e_second_t = 0.;
	for (int64_t k = 0; k < n_terms; ++k) {
		const int64_t j = lower + k;
		const double jd = static_cast<double>(j);
		const double w = std::exp(terms[static_cast<size_t>(k)] - max_log) / sum_w;
		const double lr = -jd / (p - 1.);
		double lt = 0., lrp = 0., lpp = 0.;
		if (calculate_power_derivatives) {
			const double digamma_value = cache == nullptr ? GPBoost::digamma(jd * alpha) : cache->digamma_alpha[j];
			const double H = log_y - std::log(p - 1.) - rho - digamma_value;
			const double lp = jd * (alpha_p * H - alpha / (p - 1.) + 1. / (2. - p));
			lt = dp_dtheta * lp;
			lrp = dp_dtheta * jd / ((p - 1.) * (p - 1.));
			e_second_rt += w * lrp;
			if (order == TweedieDerivativeOrder::kSecond) {
				const double trigamma_value = cache == nullptr ? GPBoost::trigamma(jd * alpha) : cache->trigamma_alpha[j];
				lpp = jd * (alpha_pp * H - 2. * alpha_p / (p - 1.) - jd * alpha_p * alpha_p * trigamma_value + alpha / ((p - 1.) * (p - 1.)) + 1. / ((2. - p) * (2. - p)));
				e_second_t += w * (dp_dtheta * dp_dtheta * lpp + d2p_dtheta2 * lp);
			}
		}
		er += w * lr;
		et += w * lt;
		er2 += w * lr * lr;
		ert += w * lr * lt;
		et2 += w * lt * lt;
	}
	ans.d_rho = er;
	ans.d_theta = et;
	if (order == TweedieDerivativeOrder::kSecond) {
		ans.d2_rho = std::max(0., er2 - er * er);
		ans.d2_rho_theta = e_second_rt + ert - er * et;
		ans.d2_theta = e_second_t + et2 - et * et;
	}
	return ans;
}

/*!
* \brief Joint log-density of (Y, N) of the compound Poisson--Gamma representation, N ~ Poisson(mu^(2-p) / (phi * (2-p))),
*	Y | N = n ~ Gamma(n * (2-p) / (p-1), scale = phi * (p-1) * mu^(p-1)), without the part that depends on the location.
*	The location-dependent part equals the one of the marginal Tweedie density ('TweedieLocationQuantity::kCanonical'), and
*	the remaining part is the n-th term of the series of the marginal normalizer, i.e. log(a_n(y, phi, p)) with
*	a(y, phi, p) = sum_n a_n(y, phi, p). For n = 0, y has to be 0 and the result is 0.
*	The derivatives are wrt rho = log(phi) and the transformed power theta (dp/dtheta = 'dp_dtheta'), with the same
*	fields and conventions as 'EvaluateTweedieLogNormalizer'. Since log(a_n) is linear in rho, d2_rho = 0
*/
inline TweedieSeriesResult EvaluateTweedieJointLogNormalizer(double y, double n, double rho, double p,
	double dp_dtheta, double d2p_dtheta2, TweedieDerivativeOrder order, bool calculate_power_derivatives) {
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
	const double alpha = (2. - p) / (p - 1.);
	ans.log_a = TweedieSeriesLogTerm(static_cast<int64_t>(n), log_y, rho, p) - log_y;
	if (order == TweedieDerivativeOrder::kValue) return ans;
	ans.d_rho = -n / (p - 1.);
	if (calculate_power_derivatives) {
		const double alpha_p = -1. / ((p - 1.) * (p - 1.));
		const double H = log_y - std::log(p - 1.) - rho - GPBoost::digamma(n * alpha);
		const double lp = n * (alpha_p * H - alpha / (p - 1.) + 1. / (2. - p));
		ans.d_theta = dp_dtheta * lp;
		if (order == TweedieDerivativeOrder::kSecond) {
			const double alpha_pp = 2. / ((p - 1.) * (p - 1.) * (p - 1.));
			const double lpp = n * (alpha_pp * H - 2. * alpha_p / (p - 1.) - n * alpha_p * alpha_p * GPBoost::trigamma(n * alpha) +
				alpha / ((p - 1.) * (p - 1.)) + 1. / ((2. - p) * (2. - p)));
			ans.d2_rho_theta = dp_dtheta * n / ((p - 1.) * (p - 1.));
			ans.d2_theta = dp_dtheta * dp_dtheta * lpp + d2p_dtheta2 * lp;
		}
	}
	return ans;
}

}  // namespace GPBoost

#endif  // GPB_TWEEDIE_UTILS_
