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

struct TweedieSpecialFunctionCache {
	double p = std::numeric_limits<double>::quiet_NaN();
	double alpha = std::numeric_limits<double>::quiet_NaN();
	std::vector<double> log_factorial{ 0. };
	std::vector<double> log_gamma_alpha{ std::numeric_limits<double>::quiet_NaN() };
	std::vector<double> digamma_alpha{ std::numeric_limits<double>::quiet_NaN() };
	std::vector<double> trigamma_alpha{ std::numeric_limits<double>::quiet_NaN() };

	void Ensure(double power, int64_t upper, TweedieDerivativeOrder order, bool calculate_power_derivatives) {
		if (upper < 1) return;
		const size_t required_size = static_cast<size_t>(upper + 1);
		const size_t old_factorial_size = log_factorial.size();
		if (old_factorial_size < required_size) {
			log_factorial.resize(required_size);
			for (size_t j = old_factorial_size; j < required_size; ++j) log_factorial[j] = std::lgamma(static_cast<double>(j) + 1.);
		}
		if (power != p) {
			p = power;
			alpha = (2. - p) / (p - 1.);
			log_gamma_alpha.assign(1, std::numeric_limits<double>::quiet_NaN());
			digamma_alpha.assign(1, std::numeric_limits<double>::quiet_NaN());
			trigamma_alpha.assign(1, std::numeric_limits<double>::quiet_NaN());
		}
		const size_t old_gamma_size = log_gamma_alpha.size();
		if (old_gamma_size < required_size) {
			log_gamma_alpha.resize(required_size);
			for (size_t j = old_gamma_size; j < required_size; ++j) log_gamma_alpha[j] = std::lgamma(static_cast<double>(j) * alpha);
		}
		if (calculate_power_derivatives && order != TweedieDerivativeOrder::kValue) {
			const size_t old_digamma_size = digamma_alpha.size();
			if (old_digamma_size < required_size) {
				digamma_alpha.resize(required_size);
				for (size_t j = old_digamma_size; j < required_size; ++j) digamma_alpha[j] = GPBoost::digamma(static_cast<double>(j) * alpha);
			}
			if (order == TweedieDerivativeOrder::kSecond) {
				const size_t old_trigamma_size = trigamma_alpha.size();
				if (old_trigamma_size < required_size) {
					trigamma_alpha.resize(required_size);
					for (size_t j = old_trigamma_size; j < required_size; ++j) trigamma_alpha[j] = GPBoost::trigamma(static_cast<double>(j) * alpha);
				}
			}
		}
	}
};

struct TweedieLocationResult {
	double canonical = 0.;
	double score = 0.;
	double information = 0.;
	double deriv_information_eta = 0.;
	double log_scaled_a = 0.;
	double log_scaled_b = -std::numeric_limits<double>::infinity();
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

inline TweedieLocationResult EvaluateTweedieLocation(double y, double eta, double rho, double p) {
	TweedieLocationResult ans;
	ans.log_scaled_a = (2. - p) * eta - rho;
	if (y > 0.) ans.log_scaled_b = std::log(y) + (1. - p) * eta - rho;
	ans.canonical = TweedieSignedLogSum(-1. / (2. - p), ans.log_scaled_a, -1. / (p - 1.), ans.log_scaled_b);
	ans.score = TweedieSignedLogSum(-1., ans.log_scaled_a, 1., ans.log_scaled_b);
	ans.information = TweedieSignedLogSum(2. - p, ans.log_scaled_a, p - 1., ans.log_scaled_b);
	ans.deriv_information_eta = TweedieSignedLogSum((2. - p) * (2. - p), ans.log_scaled_a, -(p - 1.) * (p - 1.), ans.log_scaled_b);
	return ans;
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
		cache->Ensure(p, j, TweedieDerivativeOrder::kValue, false);
		return jd * alpha * (log_y - std::log(p - 1.)) - jd * (alpha + 1.) * rho -
			jd * std::log(2. - p) - cache->log_factorial[static_cast<size_t>(j)] - cache->log_gamma_alpha[static_cast<size_t>(j)];
	}
	return jd * alpha * (log_y - std::log(p - 1.)) - jd * (alpha + 1.) * rho -
		jd * std::log(2. - p) - std::lgamma(jd + 1.) - std::lgamma(jd * alpha);
}

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
	if (!std::isfinite(log_mode) || log_mode > std::log(static_cast<double>(max_terms))) {
		ans.converged = false;
		return ans;
	}
	int64_t mode = std::max<int64_t>(1, static_cast<int64_t>(std::floor(std::exp(log_mode))));
	if (mode > max_terms) {
		ans.converged = false;
		return ans;
	}
	// Move to the discrete maximum. The continuous approximation is normally within a few terms.
	double mode_term = TweedieSeriesLogTerm(mode, log_y, rho, p, cache);
	while (mode < max_terms) {
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
	const double log_tol = 2. * std::log(std::numeric_limits<double>::epsilon());
	int64_t lower = mode;
	while (lower > 1 && TweedieSeriesLogTerm(lower - 1, log_y, rho, p, cache) - mode_term > log_tol) --lower;
	int64_t upper = mode;
	while (upper < max_terms && TweedieSeriesLogTerm(upper + 1, log_y, rho, p, cache) - mode_term > log_tol) ++upper;
	if (upper == max_terms || upper - lower + 1 > max_terms) {
		ans.converged = false;
		return ans;
	}
	ans.lower = lower;
	ans.upper = upper;
	if (cache != nullptr) cache->Ensure(p, upper, order, calculate_power_derivatives);
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
			const double digamma_value = cache == nullptr ? GPBoost::digamma(jd * alpha) : cache->digamma_alpha[static_cast<size_t>(j)];
			const double H = log_y - std::log(p - 1.) - rho - digamma_value;
			const double lp = jd * (alpha_p * H - alpha / (p - 1.) + 1. / (2. - p));
			lt = dp_dtheta * lp;
			lrp = dp_dtheta * jd / ((p - 1.) * (p - 1.));
			e_second_rt += w * lrp;
			if (order == TweedieDerivativeOrder::kSecond) {
				const double trigamma_value = cache == nullptr ? GPBoost::trigamma(jd * alpha) : cache->trigamma_alpha[static_cast<size_t>(j)];
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

}  // namespace GPBoost

#endif  // GPB_TWEEDIE_UTILS_
