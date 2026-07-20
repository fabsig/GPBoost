/*
 * Copyright (c) 2026 by GPBoost contributors
 *
 * Numerically stable building blocks for the generalized Pareto distribution
 * and the continuous Naveau extended generalized Pareto distributions.
 */
#ifndef GPBOOST_EGPD_UTILS_H_
#define GPBOOST_EGPD_UTILS_H_

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

namespace GPBoost {

constexpr int kMaxEGPDAuxPars = 4;

enum class EGPDVariant { kGPD, kPower, kPowerMixture, kBeta, kPowerBeta };
enum class EGPDEvalStatus {
	kValid,
	kInvalidResponse,
	kInvalidShapeDomain,
	kInvalidAuxParameter,
	kOutsideFiniteEndpoint,
	kNonFiniteInput,
	kNumericalOverflow,
	kQuadratureFailure
};

struct EGPDParams {
	double shape_shift = 0.5;
	double kappa = 1.;
	double kappa1 = 1.;
	double delta_kappa = 1.;
	double delta = 1.;
	double odds = 1.;
};

struct EGPDDerivatives {
	double log_likelihood = 0.;
	double d_eta = 0.;
	double d2_eta = 0.;
	double d3_eta = 0.;
	std::array<double, kMaxEGPDAuxPars> d_aux_optimizer{};
	std::array<double, kMaxEGPDAuxPars> d_eta_aux_optimizer{};
	std::array<double, kMaxEGPDAuxPars> d2_eta_aux_optimizer{};
	EGPDEvalStatus status = EGPDEvalStatus::kValid;
};

struct EGPDMoments {
	double mean_unit_scale = std::numeric_limits<double>::quiet_NaN();
	double variance_unit_scale = std::numeric_limits<double>::quiet_NaN();
	bool mean_exists = false;
	bool variance_exists = false;
	EGPDEvalStatus status = EGPDEvalStatus::kValid;
};

// A fixed-size third-order location / first-order auxiliary Taylor jet. The
// coefficients are factorial-scaled in the location direction. This is a
// compact implementation of the analytic chain rule and introduces neither a
// run-time differentiation dependency nor allocation in observation loops.
struct EGPDJet {
	std::array<double, 4> e{};
	std::array<std::array<double, 3>, kMaxEGPDAuxPars> a{};

	EGPDJet() = default;
	explicit EGPDJet(double value) { e[0] = value; }
	static EGPDJet Eta(double value) {
		EGPDJet x(value);
		x.e[1] = 1.;
		return x;
	}
	static EGPDJet Aux(double value, int index, double derivative) {
		EGPDJet x(value);
		x.a[index][0] = derivative;
		return x;
	}
};

inline EGPDJet operator+(const EGPDJet& x, const EGPDJet& y) {
	EGPDJet z;
	for (int i = 0; i < 4; ++i) z.e[i] = x.e[i] + y.e[i];
	for (int j = 0; j < kMaxEGPDAuxPars; ++j) for (int i = 0; i < 3; ++i) z.a[j][i] = x.a[j][i] + y.a[j][i];
	return z;
}
inline EGPDJet operator-(const EGPDJet& x, const EGPDJet& y) {
	EGPDJet z;
	for (int i = 0; i < 4; ++i) z.e[i] = x.e[i] - y.e[i];
	for (int j = 0; j < kMaxEGPDAuxPars; ++j) for (int i = 0; i < 3; ++i) z.a[j][i] = x.a[j][i] - y.a[j][i];
	return z;
}
inline EGPDJet operator-(const EGPDJet& x) { return EGPDJet(0.) - x; }
inline EGPDJet operator*(const EGPDJet& x, const EGPDJet& y) {
	EGPDJet z;
	for (int n = 0; n < 4; ++n) for (int i = 0; i <= n; ++i) z.e[n] += x.e[i] * y.e[n - i];
	for (int j = 0; j < kMaxEGPDAuxPars; ++j) {
		for (int n = 0; n < 3; ++n) for (int i = 0; i <= n; ++i) {
			z.a[j][n] += x.a[j][i] * y.e[n - i] + x.e[i] * y.a[j][n - i];
		}
	}
	return z;
}
inline EGPDJet operator+(const EGPDJet& x, double y) { return x + EGPDJet(y); }
inline EGPDJet operator+(double x, const EGPDJet& y) { return EGPDJet(x) + y; }
inline EGPDJet operator-(const EGPDJet& x, double y) { return x - EGPDJet(y); }
inline EGPDJet operator-(double x, const EGPDJet& y) { return EGPDJet(x) - y; }
inline EGPDJet operator*(const EGPDJet& x, double y) { return x * EGPDJet(y); }
inline EGPDJet operator*(double x, const EGPDJet& y) { return EGPDJet(x) * y; }

inline EGPDJet EGPDInverse(const EGPDJet& x) {
	EGPDJet z;
	z.e[0] = 1. / x.e[0];
	for (int n = 1; n < 4; ++n) {
		for (int i = 1; i <= n; ++i) z.e[n] -= x.e[i] * z.e[n - i];
		z.e[n] /= x.e[0];
	}
	for (int j = 0; j < kMaxEGPDAuxPars; ++j) {
		for (int n = 0; n < 3; ++n) {
			for (int i = 0; i <= n; ++i) z.a[j][n] -= x.a[j][i] * z.e[n - i];
			for (int i = 1; i <= n; ++i) z.a[j][n] -= x.e[i] * z.a[j][n - i];
			z.a[j][n] /= x.e[0];
		}
	}
	return z;
}
inline EGPDJet operator/(const EGPDJet& x, const EGPDJet& y) { return x * EGPDInverse(y); }
inline EGPDJet operator/(const EGPDJet& x, double y) { return x * (1. / y); }
inline EGPDJet operator/(double x, const EGPDJet& y) { return EGPDJet(x) / y; }

inline EGPDJet EGPDExp(const EGPDJet& x) {
	EGPDJet z;
	z.e[0] = std::exp(x.e[0]);
	for (int n = 1; n < 4; ++n) {
		for (int i = 1; i <= n; ++i) z.e[n] += i * x.e[i] * z.e[n - i];
		z.e[n] /= n;
	}
	for (int j = 0; j < kMaxEGPDAuxPars; ++j) {
		for (int n = 0; n < 3; ++n) for (int i = 0; i <= n; ++i) z.a[j][n] += z.e[i] * x.a[j][n - i];
	}
	return z;
}

inline EGPDJet EGPDLog(const EGPDJet& x) {
	EGPDJet z;
	z.e[0] = std::log(x.e[0]);
	const EGPDJet dx_over_x = EGPDInverse(x);
	std::array<double, 3> dx{};
	dx[0] = x.e[1]; dx[1] = 2. * x.e[2]; dx[2] = 3. * x.e[3];
	for (int n = 0; n < 3; ++n) {
		double coef = 0.;
		for (int i = 0; i <= n; ++i) coef += dx[i] * dx_over_x.e[n - i];
		z.e[n + 1] = coef / (n + 1.);
	}
	for (int j = 0; j < kMaxEGPDAuxPars; ++j) {
		for (int n = 0; n < 3; ++n) for (int i = 0; i <= n; ++i) z.a[j][n] += x.a[j][i] * dx_over_x.e[n - i];
	}
	return z;
}

inline EGPDJet EGPDPowInt(EGPDJet x, int power) {
	EGPDJet ans(1.);
	while (power > 0) {
		if (power & 1) ans = ans * x;
		x = x * x;
		power >>= 1;
	}
	return ans;
}

inline EGPDJet EGPDExprel(const EGPDJet& x) {
	if (std::abs(x.e[0]) < 0.1) {
		EGPDJet sum(1.), term(1.);
		double factorial = 1.;
		for (int n = 1; n <= 16; ++n) {
			term = term * x;
			factorial *= (n + 1.);
			sum = sum + term / factorial;
		}
		return sum;
	}
	return (EGPDExp(x) - 1.) / x;
}

inline EGPDJet EGPDLogAddExp(const EGPDJet& x, const EGPDJet& y) {
	const double m = std::max(x.e[0], y.e[0]);
	return m + EGPDLog(EGPDExp(x - m) + EGPDExp(y - m));
}

inline int EGPDNumAuxPars(EGPDVariant variant) {
	if (variant == EGPDVariant::kGPD) return 1;
	if (variant == EGPDVariant::kPower || variant == EGPDVariant::kBeta) return 2;
	if (variant == EGPDVariant::kPowerBeta) return 3;
	return 4;
}

inline EGPDEvalStatus ValidateEGPDParams(const EGPDParams& pars, EGPDVariant variant) {
	// shape_shift = xi + 0.5, so the GPD regularity constraint xi > -0.5 is exactly shape_shift > 0.
	if (!(std::isfinite(pars.shape_shift) && pars.shape_shift > 0.)) return EGPDEvalStatus::kInvalidShapeDomain;
	if (variant == EGPDVariant::kPower && !(std::isfinite(pars.kappa) && pars.kappa > 0.)) return EGPDEvalStatus::kInvalidAuxParameter;
	if (variant == EGPDVariant::kPowerMixture && (!(std::isfinite(pars.kappa1) && pars.kappa1 > 0.) || !(std::isfinite(pars.delta_kappa) && pars.delta_kappa > 0.) || !(std::isfinite(pars.odds) && pars.odds > 0.) || !std::isfinite(pars.kappa1 + pars.delta_kappa))) return EGPDEvalStatus::kInvalidAuxParameter;
	if ((variant == EGPDVariant::kBeta || variant == EGPDVariant::kPowerBeta) && !(std::isfinite(pars.delta) && pars.delta > 0.)) return EGPDEvalStatus::kInvalidAuxParameter;
	if (variant == EGPDVariant::kPowerBeta && !(std::isfinite(pars.kappa) && pars.kappa > 0.)) return EGPDEvalStatus::kInvalidAuxParameter;
	return EGPDEvalStatus::kValid;
}

inline EGPDEvalStatus CalcEGPDLogLikAndDerivatives(double y, double eta, const EGPDParams& pars, EGPDVariant variant, EGPDDerivatives* out) {
	*out = EGPDDerivatives{};
	if (!(std::isfinite(y) && y > 0.)) return out->status = EGPDEvalStatus::kInvalidResponse;
	if (!std::isfinite(eta)) return out->status = EGPDEvalStatus::kNonFiniteInput;
	const EGPDEvalStatus par_status = ValidateEGPDParams(pars, variant);
	if (par_status != EGPDEvalStatus::kValid) return out->status = par_status;

	const double xi_value = pars.shape_shift - 0.5;
	const double log_z = std::log(y) - eta;
	if (xi_value < 0. && std::log(-xi_value) + log_z >= 0.) return out->status = EGPDEvalStatus::kOutsideFiniteEndpoint;

	EGPDJet eta_j = EGPDJet::Eta(eta);
	EGPDJet shape_shift = EGPDJet::Aux(pars.shape_shift, 0, pars.shape_shift);
	EGPDJet xi = shape_shift - 0.5;
	EGPDJet z = y * EGPDExp(-eta_j);
	EGPDJet x = xi * z;
	EGPDJet a;
	if (std::abs(x.e[0]) < 0.05) {
		EGPDJet term = z;
		a = -term;
		for (int n = 1; n <= 18; ++n) {
			term = term * x;
			a = a + ((n & 1) ? 1. : -1.) * term / (n + 1.);
		}
	}
	else {
		EGPDJet t = 1. + x;
		if (!(t.e[0] > 0.)) return out->status = EGPDEvalStatus::kOutsideFiniteEndpoint;
		a = -EGPDLog(t) / xi;
	}
	EGPDJet r = EGPDExp(a);
	EGPDJet u = 1. - r;
	u.e[0] = -std::expm1(a.e[0]);
	if (!(u.e[0] > 0. && u.e[0] < 1.)) return out->status = EGPDEvalStatus::kNumericalOverflow;
	EGPDJet log_u = EGPDLog(u);
	log_u.e[0] = std::log(-std::expm1(a.e[0]));
	EGPDJet log_lik = -eta_j + (1. + xi) * a;

	if (variant == EGPDVariant::kPower) {
		EGPDJet kappa = EGPDJet::Aux(pars.kappa, 1, pars.kappa);
		log_lik = log_lik + EGPDLog(kappa) + (kappa - 1.) * log_u;
	}
	else if (variant == EGPDVariant::kPowerMixture) {
		EGPDJet k1 = EGPDJet::Aux(pars.kappa1, 1, pars.kappa1);
		EGPDJet gap = EGPDJet::Aux(pars.delta_kappa, 2, pars.delta_kappa);
		EGPDJet k2 = k1 + gap;
		const double p_value = pars.odds / (1. + pars.odds);
		EGPDJet p = EGPDJet::Aux(p_value, 3, p_value * (1. - p_value));
		EGPDJet A = EGPDLog(p) + EGPDLog(k1) + (k1 - 1.) * log_u;
		EGPDJet B = EGPDLog(1. - p) + EGPDLog(k2) + (k2 - 1.) * log_u;
		log_lik = log_lik + EGPDLogAddExp(A, B);
	}
	else if (variant == EGPDVariant::kBeta || variant == EGPDVariant::kPowerBeta) {
		EGPDJet delta = EGPDJet::Aux(pars.delta, 1, pars.delta);
		EGPDJet B;
		EGPDJet Bprime;
		if (u.e[0] < 1e-4) {
			EGPDJet u_power = u * u;
			EGPDJet falling = 1. + delta;
			B = falling * u_power / 2.;
			Bprime = (1. + delta) * u;
			for (int m = 3; m <= 10; ++m) {
				falling = falling * (delta - (m - 2.));
				u_power = u_power * u;
				double fact = 1.;
				for (int j = 2; j <= m; ++j) fact *= j;
				const double sign = (m & 1) ? -1. : 1.;
				B = B + sign * falling * u_power / fact;
				Bprime = Bprime + sign * falling * EGPDPowInt(u, m - 1) / (fact / m);
			}
		}
		else {
			EGPDJet exprel = EGPDExprel(delta * a);
			B = u + r * a * exprel;
			Bprime = (1. + delta) * (-a) * exprel;
		}
		if (!(B.e[0] > 0. && B.e[0] < 1.00000000000001 && Bprime.e[0] > 0.)) return out->status = EGPDEvalStatus::kNumericalOverflow;
		if (variant == EGPDVariant::kBeta) {
			log_lik = log_lik + EGPDLog(Bprime);
		}
		else {
			EGPDJet kappa = EGPDJet::Aux(pars.kappa, 2, pars.kappa);
			log_lik = log_lik + EGPDLog(kappa / 2.) + (kappa / 2. - 1.) * EGPDLog(B) + EGPDLog(Bprime);
		}
	}

	out->log_likelihood = log_lik.e[0];
	out->d_eta = log_lik.e[1];
	out->d2_eta = 2. * log_lik.e[2];
	out->d3_eta = 6. * log_lik.e[3];
	for (int j = 0; j < EGPDNumAuxPars(variant); ++j) {
		out->d_aux_optimizer[j] = log_lik.a[j][0];
		out->d_eta_aux_optimizer[j] = log_lik.a[j][1];
		out->d2_eta_aux_optimizer[j] = 2. * log_lik.a[j][2];
	}
	if (!std::isfinite(out->log_likelihood) || !std::isfinite(out->d_eta) || !std::isfinite(out->d2_eta) || !std::isfinite(out->d3_eta)) return out->status = EGPDEvalStatus::kNumericalOverflow;
	return out->status = EGPDEvalStatus::kValid;
}

inline double EGPDCarrierDensity(double u, const EGPDParams& pars, EGPDVariant variant) {
	if (variant == EGPDVariant::kGPD) return 1.;
	if (variant == EGPDVariant::kPower) return pars.kappa * std::pow(u, pars.kappa - 1.);
	if (variant == EGPDVariant::kPowerMixture) {
		const double p = pars.odds / (1. + pars.odds);
		const double k2 = pars.kappa1 + pars.delta_kappa;
		return p * pars.kappa1 * std::pow(u, pars.kappa1 - 1.) + (1. - p) * k2 * std::pow(u, k2 - 1.);
	}
	const double r = 1. - u;
	const double rd = std::pow(r, pars.delta);
	const double B = (pars.delta - (1. + pars.delta) * r + r * rd) / pars.delta;
	const double Bp = (1. + pars.delta) * (1. - rd) / pars.delta;
	if (variant == EGPDVariant::kBeta) return Bp;
	return 0.5 * pars.kappa * std::pow(B, 0.5 * pars.kappa - 1.) * Bp;
}

inline EGPDMoments CalcEGPDUnitScaleMoments(const EGPDParams& pars, EGPDVariant variant) {
	EGPDMoments ans;
	const EGPDEvalStatus par_status = ValidateEGPDParams(pars, variant);
	if (par_status != EGPDEvalStatus::kValid) { ans.status = par_status; return ans; }
	const double xi = pars.shape_shift - 0.5;
	ans.mean_exists = xi < 1.;
	ans.variance_exists = xi < 0.5;
	if (!ans.mean_exists) return ans;
	// Midpoint-rule quadrature under the substitution u = sin(pi v / 2)^2, which
	// regularizes both carrier endpoints (du/dv vanishes at v = 0 and v = 1).
	// Moment calculations are auxiliary-only and occur once per prediction call,
	// never in an observation loop.
	constexpr int n = 160;
	double first = 0., second = 0.;
	for (int i = 0; i < n; ++i) {
		const double v = (i + 0.5) / n;
		const double s = std::sin(0.5 * 3.14159265358979323846 * v);
		const double u = s * s;
		const double du_dv = 3.14159265358979323846 * s * std::cos(0.5 * 3.14159265358979323846 * v);
		const double log_r = std::log1p(-u);
		const double z = xi == 0. ? -log_r : std::expm1(-xi * log_r) / xi;
		const double weight = EGPDCarrierDensity(u, pars, variant) * du_dv / n;
		first += weight * z;
		if (ans.variance_exists) second += weight * z * z;
	}
	ans.mean_unit_scale = first;
	if (ans.variance_exists) ans.variance_unit_scale = std::max(0., second - first * first);
	if (!std::isfinite(ans.mean_unit_scale) || (ans.variance_exists && !std::isfinite(ans.variance_unit_scale))) ans.status = EGPDEvalStatus::kQuadratureFailure;
	return ans;
}

}  // namespace GPBoost

#endif  // GPBOOST_EGPD_UTILS_H_
