/*!
* This file is part of GPBoost a C++ library for combining
*   boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2026 Fabio Sigrist, Tim Gyger, and Pascal Kuendig. All rights reserved.
* 
* 
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*
* 
* Iterative methods (matrix_inversion_method_ == "iterative") are based on the following references:
*	- Kündig and Sigrist, 2025, "Scalable Krylov Subspace Methods for Generalized Mixed Effects Models with Crossed Random Effects", https://arxiv.org/abs/2505.09552
*	- Gyger, Furrer, and Sigrist, SIAM/ASA JUQ 2026, "Iterative Methods for Full-Scale Gaussian Process Approximations for Large Spatial Data", https://epubs.siam.org/doi/full/10.1137/25M1731320
* 
*  EXPLANATIONS ON PARAMETERIZATIONS USED
*
*  The following notation is used below:
*		- location_par = random + fixed effects
*
* For a "gamma" likelihood, the following density is used:
*   f(y) = lambda^gamma / Gamma(gamma) * y^(gamma - 1) * exp(-lambda * y)
*       - mu = mean(y) = exp(location_par), lambda = gamma / mu, gamma \in (0,\infty) (= aux_pars_[0])
*		- Note: var(y) = mu^2 / gamma, lambda = rate, gamma = shape
*
* For "tweedie" and "tweedie_fixed_p", Y follows a compound Poisson--Gamma Tweedie law:
*       - mu = exp(location_par), Var(Y | location_par) = phi * mu^p, 1.01 < p < 1.99
*       - aux_pars_[0] = phi; aux_pars_[1] is a positive transformed power for "tweedie"
*         and the fixed power on its original scale for "tweedie_fixed_p"
*
* For a "lognormal" likelihood, the following density is used:
*   f(y) = 1 / ( y * sqrt(2*pi*sigma2) ) * exp( - ( log(y) - (eta - 0.5*sigma2) )^2 / (2*sigma2) ),  for y > 0
*       - mu = mean(y) = exp(location_par) = exp(eta)
*       - mean(log(y)) = eta - 0.5*sigma2,  var(log(y)) = sigma2 = \in (0,\infty) (= aux_pars_[0])
*       - Note: var(y) = (exp(sigma2) - 1) * mu^2
*
* For a student "t" likelihood, the following density is used:
*   f(y) = Gamma((nu+1)/2) / sigma / sqrt(pi) / sqrt(nu) / Gamma(nu/2) * (1 + (y - mu)^2/nu/sigma^2)^(-(nu+1)/2)
*       - mu = location_par, sigma \in (0,\infty) (= aux_pars_[0]), nu \in (0,\infty) (= aux_pars_[1])
*		- Note: sigma = scale, nu = degrees of freedom
*
* For a "beta" likelihood, the following parametrization of Ferrari and Cribari-Neto (2004) is used:
*	f(y) = Gamma(mu*phi) / Gamma((1−mu)*phi) / Gamma(phi) * ​y^(mu*phi−1) * (1−y)^((1−mu)*phi−1)
*		- mu = mean(y) = 1 / (1 + exp(-location_par)), phi \in (0,\infty) (= aux_pars_[0])
*		- Note: phi = precision
*
* For a "poisson" likelihood, the following density is used:
*   f(y) = mu^y * exp(-mu) / Gamma(y)
*       - mu = mean(y) = exp(location_par)
*       - Note: var(y) = mu
*
* For a "negative_binomial" likelihood, the following density is used:
*   f(y) = Gamma(y + r) / Gamma(y + 1) / Gamma(r) * (1 - p)^y * p^r
*       - mu = mean(y) = exp(location_par), p = r / (mu + r), r \in (0,\infty) (= aux_pars_[0])
*       - Note: var(y) = mu * (mu + r) / r, p = success probability, r = shape (aka size, theta, or "number of successes")
*
* For a "negative_binomial_1" likelihood, the following density is used:
*   f(y) = Gamma(y + r) / Gamma(y + 1) / Gamma(r) * (1 - p)^y * p^r
*       - mu = mean(y) = exp(location_par), r = mu / phi, p = 1 / (1 + phi), phi \in (0,\infty) (= aux_pars_[0])
*       - Note: var(y) = mu * (1 + phi), p = success probability, r = shape, phi = dispersion
*
* For a "tweedie" (compound Poisson-gamma, 1 < p < 2) likelihood, the following density is used:
*   f(y) = a(y,phi,p) * exp( ( y * mu^(1-p)/(1-p) - mu^(2-p)/(2-p) ) / phi ),  with a point mass P(Y=0) = exp(-mu^(2-p)/(phi*(2-p)))
*       - mu = mean(y) = exp(location_par), phi \in (0,\infty) (= aux_pars_[0], dispersion), p \in (1,2) (= aux_pars_[1], variance power)
*       - Note: var(y) = phi * mu^p. "tweedie_fixed_p" fixes p (given via 'likelihood_additional_param') and estimates only phi
*
* For a "beta-binomial" likelihood, the following density is used:
*	f(y) = C(n, y * n) * Beta(y * n + mu * phi, n - y * n + (1-mu) * phi) / Beta(mu * phi, (1-mu) * phi)
*		- y = (number of successes) / (number of trials), n = number of trials
*		- mu = mean(p) = 1 / (1 + exp(-location_par)), phi \in (0,\infty) (= aux_pars_[0])
*		- Note: phi = precision
*
* For a "hurdle_gamma" likelihood, the following density is used:
*   f(y) = p0 * 1_{y=0} + (1-p0) * 1_{y>0} * lambda^gamma / Gamma(gamma) * y^(gamma - 1) * exp(-lambda * y)
*       - mu = mean(y | y > 0) = exp(location_par), lambda = gamma / mu, gamma \in (0,\infty) (= aux_pars_[0]), p0 \in (0,1) (= aux_pars_[1] / (aux_pars_[1] + 1) )
*		- Note: lambda = rate, gamma = shape, p0 = zero-inflation probability
*
* The "hurdle_<base>" likelihoods generalize "hurdle_gamma" to other positive-continuous bases (two-part density: point
*	mass p0 at y=0, base density on y>0; exp(location_par) = base mean or GPD scale): "hurdle_lognormal" (aux: log_variance,
*	p0) and the extreme-value bases "hurdle_gpd" / "hurdle_egpd_power" / "hurdle_egpd_power_mixture" / "hurdle_egpd_beta" /
*	"hurdle_egpd_power_beta" (same aux parameters as the gpd/egpd likelihood below, plus p0).
*
* The "zero_inflated_<base>" count likelihoods (the base can itself generate zeros) use:
*   f(y) = p0 * 1_{y=0} + (1-p0) * f_base(y),   mu = mean of base = exp(location_par),   E(y) = (1-p0)*mu
*       - "zero_inflated_poisson" (aux: p0), "zero_inflated_negative_binomial" (NB2; aux: shape, p0),
*         "zero_inflated_negative_binomial_1" (NB1; aux: dispersion, p0)
*       - The unsuffixed name defaults to combined Fisher-Laplace: exact score and (quasi-)Fisher curvature for mode finding,
*         followed by the observed-Hessian determinant and its derivatives. "_laplace" uses observed curvature throughout;
*         "_fisher_laplace" uses (quasi-)Fisher curvature throughout.
*
* The "hurdle_regression_<base>" and "zero_inflated_regression_<base>" variants replace the constant p0 by a logistic
*	regression pi_i = 1 / (1 + exp(-x_i'alpha)) on a second, fixed-effects-only location-parameter block (zeta = X*alpha); the
*	response predictor (eta) carries the random effects, the structural-zero predictor (zeta) does not.
*
* For the "gpd" and "egpd_*" (generalized / extended generalized Pareto) likelihoods (positive, heavy-tailed):
*   the base is a GPD with density h(y) = 1/sigma * (1 + xi*y/sigma)^(-1/xi - 1), sigma = exp(location_par) (scale),
*   xi = tail shape (= aux "shape"; the mean exists only for xi < 1). The extended families compose a carrier CDF G on [0,1]
*   with the GPD CDF H via f(y) = G'(H(y)) * h(y), adding lower-tail flexibility:
*       - "gpd": G(u) = u (aux: shape);   "egpd_power": G(u) = u^kappa (aux: shape, kappa)
*       - "egpd_power_mixture": G(u) = p*u^kappa1 + (1-p)*u^(kappa1+delta_kappa) (aux: shape, kappa1, delta_kappa, p)
*       - "egpd_beta": incomplete-Beta-type carrier with shape delta (aux: shape, delta)
*       - "egpd_power_beta": power-composed Beta carrier (aux: shape, delta, kappa)
*
* For a "zero_censored_power_transformed_normal" likelihood, the following density is used:
*   f(y) = Phi(a_0) * 1_{y=0} + 1_{y>0} * 1 / sigma * phi((y^(1/lambda) - mu) / sigma) * 1 / lambda * y^(1/lambda - 1)
*       - mu = location_par, a_0 = -mu / sigma, sigma \in (0,\infty) (= aux_pars_[0]), lambda \in (0,\infty) (= aux_pars_[1])
*		- Phi() and phi() denote the standard normal cumulative distribution and density functions
*		- This corresponds to the model Y = max(0,X)^lambda, X ~ N(mu, sigma^2)
*
* For a "zero_censored_power_transformed_normal_heteroscedastic" likelihood, the same density is used, but sigma varies
*	across observations and is modeled by a second location parameter block instead of by an auxiliary parameter:
*   f(y) = Phi(a_0) * 1_{y=0} + 1_{y>0} * 1 / sigma * phi((y^(1/lambda) - mu) / sigma) * 1 / lambda * y^(1/lambda - 1)
*       - mu = location_par (first block), sigma = exp(location_par2) (second block, i.e., log(sigma) = location_par2),
*         a_0 = -mu / sigma, lambda \in (0,\infty) (= aux_pars_[0], the only auxiliary parameter)
*       - mu = random + fixed effects; log(sigma) = fixed effects only (covariates and / or the GPBoost tree-boosting
*         algorithm; no random effects / GPs for the standard deviation)
*       - Note: the second block parametrizes the standard deviation sigma (not the variance sigma^2), matching the
*         auxiliary parameter "sigma" of the homoscedastic variant above
*
* For a "asymmetric_laplace" (aka "quantile_regression") likelihood, the following density is used:
*	f(y) = q * (1-q) * exp((y - location_par) * (I_{y < location_par} - q)
*		- q = quantile, location_par = random + fixed effects
*		- The default approximation is Fisher-Laplace (Fisher information for both mode finding and determinant evaluation)
*		- Enable triangular-kernel curvature with the suffix "_triangular_kernel_curvature" or "_tkc"
* 
* For a "zoctn" likelihood (censored logit-transformed normal) (Qiang and Sigrist, 2026):
*   T ~ N(mu, sigma^2), W = max(min(T,1),0), Y = g(W), g(x) = expit(a + b * logit(x)),  x in (0,1),  a \in R, b > 0
*		- mu = location_par, sigma in (0,inf) (= aux_pars_[0]), a in (-inf,inf) (= log(aux_pars_[1])), b in (0,inf) (= aux_pars_[2])
*		- expit(t) = 1 / (1 + exp(-t)), logit(u) = log(u) - log(1-u)
*   The resulting distribution of Y is a mixture:
*     P(Y=0) = P(Z<=0) = Phi(-mu/sigma)
*     P(Y=1) = P(Z>=1) = 1 - Phi((1-mu)/sigma) = Phi(-(1-mu)/sigma)
*     For 0<y<1, let x = g^{-1}(y) = expit((logit(y) - a)/b). Then the density is
*       f_Y(y) = (1/sigma) * phi((x - mu)/sigma) * x(1-x) / (b * y(1-y)), where Phi and phi are standard normal CDF and PDF
*
* For a "zero_one_censored_transformed_beta" likelihood (Kosmidis and Zeileis, 2025):
*    P(Y = 0)   = F_B( t0 ; a, b ),     t0 = u / (1 + 2u)
*    f_Y(y)     = f_B( t ; a, b ) / (1 + 2u)   for y in (0,1)
*    P(Y = 1)   = 1 - F_B( t1 ; a, b ), t1 = (1 + u) / (1 + 2u)
*        where f_B and F_B are the Beta(a, b) density and CDF, respectively
*    Parameters: aux_pars_[0] = phi > 0  (precision),  aux_pars_[1] = u > 0 (shift), a = mu * phi, b = (1 - mu) * phi, mu = 1 / (1 + exp(-location_par))
*
* For a "zero_one_censored_shifted_gamma" likelihood (Sigrist and Stahel, 2011):
*     p0 = P(Y=0) = P(Z <= xi) = G(k, xi/theta)
*     f(y | 0<y<1) = g(y+xi; k, theta) with g(z;k,theta) = z^(k-1)*exp(-z/theta)/(Gamma(k)*theta^k), z>0
*     p1 = P(Y=1) = P(Z >= 1+xi) = 1 - G(k, (1+xi)/theta)
*   Parameters:  aux_pars_[0] = k > 0 (shape), aux_pars_[1] = xi > 0 (shift), mu = exp(location_par), theta = mu/k
*
* For a "gaussian_heteroscedastic" likelihood, the following density is used:
*   f(y) = 1 / sqrt(2*pi*sigma2) * exp( -(y - mu)^2 / (2*sigma2) )
*       - mu = location_par (first block), sigma2 = exp(location_par2) (second block, i.e., log(sigma2) = location_par2)
*       - mu = random + fixed effects; log(sigma2) = fixed effects only (covariates and / or the GPBoost tree-boosting
*         algorithm; no random effects / GPs for the variance)
*       - Fisher-Laplace is the default and currently the only implemented approximation for both Gaussian heteroscedastic likelihoods
*
*/
#ifndef GPB_LIKELIHOODS_
#define GPB_LIKELIHOODS_

#define _USE_MATH_DEFINES // for M_SQRT1_2 and M_PI
#include <cmath>
#include <limits>

#include <GPBoost/type_defs.h>
#include <GPBoost/sparse_matrix_utils.h>
#include <GPBoost/DF_utils.h>
#include <GPBoost/utils.h>
#include <GPBoost/CG_utils.h>
#include <GPBoost/tweedie_utils.h>
#include <GPBoost/egpd_utils.h>

#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <functional>

// used in 'DetermineGroupsOrderedMode_Inner()'
// can lead to compiler crashes on some compilers
//#if __has_include(<execution>) && defined(__cpp_lib_execution)
//#  include <execution>
//#  if !defined(_LIBCPP_VERSION)
//#    define HAS_PAR_UNSEQ 1
//#  endif
//#endif
//#ifndef HAS_PAR_UNSEQ
//#  define HAS_PAR_UNSEQ 0
//#endif
//#if HAS_PAR_UNSEQ
//#  define EXEC_POLICY std::execution::par_unseq
//#elif defined(__cpp_lib_execution)
//#  define EXEC_POLICY std::execution::par
//#endif
#include <atomic>
#include <numeric>

#include <LightGBM/utils/log.h>
using LightGBM::Log;
#include <LightGBM/meta.h>
using LightGBM::label_t;

//Mathematical constants usually defined in cmath
#ifndef M_SQRT2
#define M_SQRT2      1.414213562373095048801688724209698079 //sqrt(2)
#endif

#include <chrono>  // only for debugging
#include <thread> // only for debugging

namespace GPBoost {

	// Forward declaration
	template<typename T_mat, typename T_chol>
	class REModelTemplate;

	/*!
	* \brief This class implements the likelihoods for the Gaussian proceses
	* The template parameters <T_mat, T_chol> can be <den_mat_t, chol_den_mat_t> , <sp_mat_t, chol_sp_mat_t>, <sp_mat_rm_t, chol_sp_mat_rm_t>
	*/
	template<typename T_mat, typename T_chol>
	class Likelihood {
	public:
		/*! \brief Constructor */
		Likelihood();

		/*!
		* \brief Constructor
		* \param type Type of likelihood
		* \param num_data Number of data points
		* \param num_re Number of random effects
		* \param has_SigmaI_mode Indicates whether the vector SigmaI_mode_ / a = (Z Sigma Zt)^-1 mode is used in the calculation of the mode or not
		* \param use_random_effects_indices_of_data If true, an incidendce matrix Z is used for duplicate locations and calculations are done on the random effects scale with the unique locations (only for Gaussian processes)
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param Zt Transpose Z^T of random effects design matrix that relates latent random effects to observations/likelihoods (used only for multiple level grouped random effects)
		* \param additional_param Additional parameter for the likelihood which cannot be estimated (e.g., degrees of freedom for likelihood = "t")
		* \param has_weights True, if sample weights should be used
		* \param weights Sample weights
		* \param likelihood_learning_rate Likelihood learning rate for generalized Bayesian inference (only non-Gaussian likelihoods)
		* \param only_one_grouped_RE True if there are only a single level grouped random effects
		* \param iid_model True if this is an explicitly requested iid model
		*/
		Likelihood(string_t type,
			data_size_t num_data,
			data_size_t num_re,
			bool has_SigmaI_mode,
			bool use_random_effects_indices_of_data,
			const data_size_t* random_effects_indices_of_data,
			const sp_mat_t* Zt,
			double additional_param,
			bool has_weights,
			const double* weights,
			double likelihood_learning_rate,
			bool only_one_grouped_RE,
			bool iid_model) {
			num_data_ = num_data;
			string_t likelihood = type;
			likelihood = ParseLikelihoodAliasKinkClipping(likelihood);
			if (kink_cliping_) {
				Log::REInfo("kink_clipping activated");
			}
			likelihood = ParseLikelihoodAliasModeRefinement(likelihood);
			likelihood = ParseLikelihoodAliasVarianceCorrection(likelihood);
			likelihood = ParseLikelihoodAliasModeFindingMethod(likelihood);
			likelihood = ParseLikelihoodAliasApproximationType(likelihood);
			likelihood = ParseLikelihoodAliasEstimateAdditionalPars(likelihood);
			likelihood = ParseLikelihoodAlias(likelihood);
			if (SUPPORTED_LIKELIHOODS_.find(likelihood) == SUPPORTED_LIKELIHOODS_.end()) {
				Log::REFatal("Likelihood of type '%s' is not supported ", likelihood.c_str());
			}
			if (mode_refinement_ != ModeRefinement::kNone && likelihood != "asymmetric_laplace") {
				Log::REFatal("The '_ssn_alm' mode refinement is currently only supported for 'likelihood' = 'asymmetric_laplace' "
					"(aliases 'quantile' and 'quantile_regression'), found '%s' ", likelihood.c_str());
			}
			if (LIKELIHOODS_ONLY_LAPLACE_.find(likelihood) != LIKELIHOODS_ONLY_LAPLACE_.end() && approximation_type_ != "laplace") {
				Log::REFatal("'approximation_type' = '%s' is not supported for 'likelihood' = '%s' ", approximation_type_.c_str(), likelihood.c_str());
			}
			if (use_fisher_for_mode_finding_) {
				if (LIKELIHOODS_SUPPORTS_FISHER_MODE_FINDING_.find(likelihood) == LIKELIHOODS_SUPPORTS_FISHER_MODE_FINDING_.end()) {
					Log::REFatal("The Fisher-Laplace approximation for mode finding is not supported for 'likelihood' = '%s' ", likelihood.c_str());
				}
			}
			likelihood_type_ = likelihood;
			CacheLikelihoodTypeDerivedQuantities();
			if (user_defined_approximation_type_ != "none") {
				approximation_type_ = user_defined_approximation_type_;
			}
			if (likelihood_type_ == "gpd") {
				aux_pars_ = { 0.5 };
				names_aux_pars_ = { "shape" };
				num_aux_pars_ = num_aux_pars_estim_ = 1;
			}
			else if (likelihood_type_ == "egpd_power") {
				aux_pars_ = { 0.5, 1. };
				names_aux_pars_ = { "shape", "kappa" };
				num_aux_pars_ = num_aux_pars_estim_ = 2;
				information_ll_can_be_negative_ = true;
			}
			else if (likelihood_type_ == "egpd_power_mixture") {
				aux_pars_ = { 0.5, 1., 1., 1. };
				names_aux_pars_ = { "shape", "kappa1", "delta_kappa", "p" };
				num_aux_pars_ = num_aux_pars_estim_ = 4;
				information_ll_can_be_negative_ = true;
			}
			else if (likelihood_type_ == "egpd_beta") {
				aux_pars_ = { 0.5, 1. };
				names_aux_pars_ = { "shape", "delta" };
				num_aux_pars_ = num_aux_pars_estim_ = 2;
			}
			else if (likelihood_type_ == "egpd_power_beta") {
				aux_pars_ = { 0.5, 1., 1. };
				names_aux_pars_ = { "shape", "delta", "kappa" };
				num_aux_pars_ = num_aux_pars_estim_ = 3;
				information_ll_can_be_negative_ = true;
			}
			else if (likelihood_type_ == "gamma") {
				aux_pars_ = { 1. };//shape parameter
				names_aux_pars_ = { "shape" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
			}//end "gamma"
			else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
				double p = 1.5;
				if (likelihood_type_ == "tweedie_fixed_p") {
					ValidateFixedTweediePower(additional_param);
					p = additional_param;
				}
				aux_pars_ = { 1., likelihood_type_ == "tweedie" ? 1. : p };
				names_aux_pars_ = { "dispersion", "power" };
				num_aux_pars_ = 2;
				num_aux_pars_estim_ = likelihood_type_ == "tweedie" ? 2 : 1;
				information_ll_can_be_exact_zero_ = true;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
			}
			else if (likelihood_type_ == "negative_binomial") {
				aux_pars_ = { 1. };//shape parameter (aka size, theta, or "number of successes")
				names_aux_pars_ = { "shape" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
			}//end "negative_binomial"
			else if (likelihood_type_ == "negative_binomial_1") {
				aux_pars_ = { 0.5 };
				names_aux_pars_ = { "dispersion" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
				// NB1 has NO closed-form exact Fisher information wrt eta, so a QUASI-Fisher (GLM expected information mu/(1+phi))
				// is used. Default: use the quasi-Fisher only for MODE FINDING (positive -> stable, allows iterative mode finding)
				// but the observed HESSIAN for the DETERMINANT (marginal-likelihood approximation), since the quasi-Fisher is not
				// the true expected information ('combined'). Append '_laplace' (pure Hessian) or '_fisher_laplace' to override.
				if (user_defined_approximation_type_ == "none") {
					approximation_type_ = "laplace";
					if (!user_defined_mode_finding_approach_) use_fisher_for_mode_finding_ = true;
				}
				SetCountApproximationTypeFlags();
			}//end "negative_binomial_1"
			else if (likelihood_type_ == "zero_inflated_poisson") {
				// Zero-inflated Poisson (constant structural-zero probability).
				// aux_pars_[0] = transformed structural-zero odds A = p0 / (1 - p0)  (-> p0 = 0.5), optimized on log(A) = logit(p0)
				aux_pars_ = { 1. };
				names_aux_pars_ = { "p0" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
				// Default combined Fisher-Laplace: exact score and Fisher curvature for mode finding, observed Hessian for the determinant.
				if (user_defined_approximation_type_ == "none") {
					approximation_type_ = "laplace";
					if (!user_defined_mode_finding_approach_) use_fisher_for_mode_finding_ = true;
				}
				SetCountApproximationTypeFlags();
			}//end "zero_inflated_poisson"
			else if (likelihood_type_ == "zero_inflated_negative_binomial") {
				aux_pars_ = { 1., 1. };// shape (kappa), transformed p0/(1-p0) (-> p0 = 0.5)
				names_aux_pars_ = { "shape", "p0" };
				num_aux_pars_ = 2;
				num_aux_pars_estim_ = 2;
				// Default combined Fisher-Laplace: exact score and Fisher curvature for mode finding, observed Hessian for the determinant.
				if (user_defined_approximation_type_ == "none") {
					approximation_type_ = "laplace";
					if (!user_defined_mode_finding_approach_) use_fisher_for_mode_finding_ = true;
				}
				SetCountApproximationTypeFlags();
			}//end "zero_inflated_negative_binomial"
			else if (likelihood_type_ == "zero_inflated_negative_binomial_1") {
				aux_pars_ = { 0.5, 1. };// dispersion (phi), transformed p0/(1-p0) (-> p0 = 0.5)
				names_aux_pars_ = { "dispersion", "p0" };
				num_aux_pars_ = 2;
				num_aux_pars_estim_ = 2;
				// The NB1 base has no closed-form exact Fisher information wrt eta, so a QUASI-Fisher (mu/(1+phi) inside the
				// mixture) is used. Default 'combined': quasi-Fisher for MODE FINDING (positive, stable) but the observed
				// HESSIAN for the DETERMINANT. Append '_laplace' (pure Hessian) or '_fisher_laplace' (quasi-Fisher throughout).
				if (user_defined_approximation_type_ == "none") {
					approximation_type_ = "laplace";
					if (!user_defined_mode_finding_approach_) use_fisher_for_mode_finding_ = true;
				}
				SetCountApproximationTypeFlags();
			}//end "zero_inflated_negative_binomial_1"
			else if (likelihood_type_ == "beta") {
				aux_pars_ = { 1. };//precision
				names_aux_pars_ = { "precision" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
			}// end "beta"
			else if (likelihood_type_ == "t") {
				if (user_defined_approximation_type_ == "none") {
					approximation_type_ = "fisher_laplace"; // default approximation
					// approximation_type_ = "laplace"; can crash due to non-concavity / negative Hessian W
					// Some simulations (maybe redo?) found no significant performance difference from ordinary Fisher–Laplace,
					//		while a combined version (approximation_type_ = "laplace" and use_fisher_for_mode_finding_ = true;) tended to be slower and less computationally stable
				}
				if (TwoNumbersAreEqual<double>(additional_param, -999.)) {
					aux_pars_ = { 1., 2. }; // internal default value for df
				}
				else if (additional_param < 0) {
					Log::REFatal("The 'likelihood_additional_param' (df) is not > 0, found = %g ", additional_param);
				}
				else {
					aux_pars_ = { 1., additional_param };
				}
				names_aux_pars_ = { "scale", "df" };
				num_aux_pars_ = 2;
				if (estimate_df_t_) {
					num_aux_pars_estim_ = 2;
				}
				else {
					num_aux_pars_estim_ = 1;
				}
				need_pred_latent_var_for_response_mean_ = false;
				if (approximation_type_ == "laplace") {
					information_ll_can_be_negative_ = true;
				}
				else if (approximation_type_ == "fisher_laplace") {
					information_changes_during_mode_finding_ = false;
					information_changes_after_mode_finding_ = false;
					grad_information_wrt_mode_non_zero_ = false;
				}
				else {
					Log::REFatal("'approximation_type' = '%s' is not supported for 'likelihood' = '%s' ", approximation_type_.c_str(), likelihood_type_.c_str());
				}
				if (use_fisher_for_mode_finding_) {
					information_changes_during_mode_finding_ = false;
				}
			}//end "t"
			else if (likelihood_type_ == "asymmetric_laplace") {
				ValidateAsymmetricLaplaceQuantile(additional_param);
				quantile_ = additional_param;
				can_use_first_deriv_log_like_for_pred_mean_ = false;
				aux_pars_ = { 1. };
				names_aux_pars_ = { "scale" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
				if (user_defined_approximation_type_ == "none") {
					approximation_type_ = "fisher_laplace"; // default approximation
				}
				if (approximation_type_ == "fisher_laplace") {
					information_ll_can_be_negative_ = false;
					information_changes_during_mode_finding_ = false;
					information_changes_after_mode_finding_ = false;
					grad_information_wrt_mode_non_zero_ = false;
				}
				else if (approximation_type_ == "triangular_kernel_curvature") {
					information_ll_can_be_negative_ = true;
					information_changes_during_mode_finding_ = true;
					information_changes_after_mode_finding_ = true;
					grad_information_wrt_mode_non_zero_ = true;
					grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
				}
				else {
					Log::REFatal("'approximation_type' = '%s' is not supported for 'likelihood' = '%s' ", approximation_type_.c_str(), likelihood_type_.c_str());
				}
				if (!user_defined_mode_finding_approach_ && approximation_type_ == "triangular_kernel_curvature") {
					use_fisher_for_mode_finding_ = true;
				}
				if (use_fisher_for_mode_finding_) {
					information_changes_during_mode_finding_ = false;
				}
			}//end "asymmetric_laplace"
			else if (IsGaussianLikelihood()) {
				aux_pars_ = { 1. };
				names_aux_pars_ = { "error_variance" };
				if (use_likelihoods_file_for_gaussian_ || likelihood_type_ == "gaussian_latent") {
					num_aux_pars_ = 1;
					num_aux_pars_estim_ = 1;
				}
				else {
					num_aux_pars_ = 0;
					num_aux_pars_estim_ = 0;
				}
				need_pred_latent_var_for_response_mean_ = false;
				information_changes_during_mode_finding_ = false;
				information_changes_after_mode_finding_ = false;
				grad_information_wrt_mode_non_zero_ = false;
				maxit_mode_newton_ = 1;
				max_number_lr_shrinkage_steps_newton_ = 1;
			}//end "gaussian"
			else if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
				if (user_defined_approximation_type_ != "none" && user_defined_approximation_type_ != "fisher_laplace") {
					Log::REFatal("Only 'fisher_laplace' approximation is implemented for likelihood = %s ", likelihood_type_.c_str());
				}
				approximation_type_ = "fisher_laplace"; // cannot use "laplace" as log-likelihood is not concave in the mean and variance
				num_aux_pars_ = 0;
				num_aux_pars_estim_ = 0;
				num_sets_re_ = 2;
				num_sets_fixed_effects_ = 2;
				need_pred_latent_var_for_response_mean_ = false;
				armijo_condition_ = false;
			}//end "gaussian_heteroscedastic_fixed_and_random"
			else if (likelihood_type_ == "gaussian_heteroscedastic") {
				// Gaussian likelihood where the mean is related to fixed and random effects and the log-error variance is related to fixed effects only (no random effects / GPs for the variance)
				if (user_defined_approximation_type_ != "none" && user_defined_approximation_type_ != "fisher_laplace") {
					Log::REFatal("Only 'fisher_laplace' approximation is implemented for likelihood = %s ", likelihood_type_.c_str());
				}
				approximation_type_ = "fisher_laplace";
				num_aux_pars_ = 0;
				num_aux_pars_estim_ = 0;
				num_sets_re_ = 1;
				num_sets_fixed_effects_ = 2;
				need_pred_latent_var_for_response_mean_ = false;
				information_changes_during_mode_finding_ = false;
				information_changes_after_mode_finding_ = false;
				grad_information_wrt_mode_non_zero_ = false;
				maxit_mode_newton_ = 1;
				max_number_lr_shrinkage_steps_newton_ = 1;
			}//end "gaussian_heteroscedastic" (fixed effects only)
			else if (likelihood_type_ == "lognormal") {
				aux_pars_ = { 0.5 };// variance on the log scale
				names_aux_pars_ = { "log_variance" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
				information_changes_during_mode_finding_ = false;
				information_changes_after_mode_finding_ = false;
				grad_information_wrt_mode_non_zero_ = false;
			}//end "lognormal"
			else if (likelihood_type_ == "beta_binomial") {
				aux_pars_ = { 20.0 };
				names_aux_pars_ = { "precision" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
			}//end "beta_binomial"
			else if (likelihood_type_ == "hurdle_gamma") {
				aux_pars_ = { 1., 1. };//shape and transformed p0/(1-p0) (-> p0 = 0.5)
				names_aux_pars_ = { "shape", "p0" };
				num_aux_pars_ = 2;
				num_aux_pars_estim_ = 2;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
				information_ll_can_be_exact_zero_ = true;
			}//end "hurdle_gamma"
			else if (likelihood_type_ == "hurdle_lognormal") {
				aux_pars_ = { 0.5, 1. };// log-scale variance (sigma2) and transformed p0/(1-p0) (-> p0 = 0.5)
				names_aux_pars_ = { "log_variance", "p0" };
				num_aux_pars_ = 2;
				num_aux_pars_estim_ = 2;
				information_changes_during_mode_finding_ = false;
				information_changes_after_mode_finding_ = false;
				grad_information_wrt_mode_non_zero_ = false;
				information_ll_can_be_exact_zero_ = true;
			}//end "hurdle_lognormal"
			else if (IsHurdleEGPD()) {
				// Base EGPD auxiliary parameters (see the non-hurdle variants) followed by the structural-zero p0 (odds).
				if (likelihood_type_ == "hurdle_gpd") { aux_pars_ = { 0.5, 1. }; names_aux_pars_ = { "shape", "p0" }; }
				else if (likelihood_type_ == "hurdle_egpd_power") { aux_pars_ = { 0.5, 1., 1. }; names_aux_pars_ = { "shape", "kappa", "p0" }; information_ll_can_be_negative_ = true; }
				else if (likelihood_type_ == "hurdle_egpd_power_mixture") { aux_pars_ = { 0.5, 1., 1., 1., 1. }; names_aux_pars_ = { "shape", "kappa1", "delta_kappa", "p", "p0" }; information_ll_can_be_negative_ = true; }
				else if (likelihood_type_ == "hurdle_egpd_beta") { aux_pars_ = { 0.5, 1., 1. }; names_aux_pars_ = { "shape", "delta", "p0" }; }
				else { aux_pars_ = { 0.5, 1., 1., 1. }; names_aux_pars_ = { "shape", "delta", "kappa", "p0" }; information_ll_can_be_negative_ = true; }// hurdle_egpd_power_beta
				num_aux_pars_ = (int)aux_pars_.size();
				num_aux_pars_estim_ = num_aux_pars_;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
				information_ll_can_be_exact_zero_ = true;
			}//end hurdle EGPD variants
			else if (IsHurdleRegression()) {
				// Regression (fixed-effects) structural-zero model: pi_i = logit^{-1}(x_i^T alpha), modeled through a second
				// fixed-effects-only location-parameter block (zeta), so there is no structural-zero auxiliary parameter here.
				// The auxiliary parameters are exactly those of the underlying positive base likelihood. eta (block 0) carries the
				// random effects; zeta (block 1) is fixed effects only. The positive base and the zero model decouple (l_{eta,zeta}=0).
				const string_t base = HurdleRegressionBaseType();
				if (base == "hurdle_gamma") { aux_pars_ = { 1. }; names_aux_pars_ = { "shape" }; }
				else if (base == "hurdle_lognormal") { aux_pars_ = { 0.5 }; names_aux_pars_ = { "log_variance" }; information_changes_during_mode_finding_ = false; information_changes_after_mode_finding_ = false; grad_information_wrt_mode_non_zero_ = false; }
				else if (base == "hurdle_gpd") { aux_pars_ = { 0.5 }; names_aux_pars_ = { "shape" }; }
				else if (base == "hurdle_egpd_power") { aux_pars_ = { 0.5, 1. }; names_aux_pars_ = { "shape", "kappa" }; information_ll_can_be_negative_ = true; }
				else if (base == "hurdle_egpd_power_mixture") { aux_pars_ = { 0.5, 1., 1., 1. }; names_aux_pars_ = { "shape", "kappa1", "delta_kappa", "p" }; information_ll_can_be_negative_ = true; }
				else if (base == "hurdle_egpd_beta") { aux_pars_ = { 0.5, 1. }; names_aux_pars_ = { "shape", "delta" }; }
				else { aux_pars_ = { 0.5, 1., 1. }; names_aux_pars_ = { "shape", "delta", "kappa" }; information_ll_can_be_negative_ = true; }// hurdle_egpd_power_beta
				num_aux_pars_ = (int)aux_pars_.size();
				num_aux_pars_estim_ = num_aux_pars_;
				num_sets_fixed_effects_ = 2;// eta (response) and zeta (structural-zero logit)
				num_sets_re_ = 1;// random effects only on the response predictor eta
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
				information_ll_can_be_exact_zero_ = true;
			}//end hurdle regression variants
			else if (IsZeroInflatedCountRegression()) {
				// Zero-inflated count with a regression structural-zero model (pi_i = logit^{-1}(x_i^T alpha)). The auxiliary parameters are those of the
				// base count component (none for Poisson, shape for NB2, dispersion for NB1). eta (block 0) carries the random effects; zeta (block 1) is
				// fixed effects only. Unlike the hurdle case, eta and zeta COUPLE at zero counts (l_{eta,zeta} != 0, dJ_eta/dzeta != 0).
				const string_t base = ZICountRegressionBaseType();
				if (base == "zero_inflated_negative_binomial") { aux_pars_ = { 1. }; names_aux_pars_ = { "shape" }; }
				else if (base == "zero_inflated_negative_binomial_1") { aux_pars_ = { 0.5 }; names_aux_pars_ = { "dispersion" }; }
				else { aux_pars_ = {}; names_aux_pars_ = {}; }// zero_inflated_poisson: no base auxiliary parameter
				num_aux_pars_ = (int)aux_pars_.size();
				num_aux_pars_estim_ = num_aux_pars_;
				num_sets_fixed_effects_ = 2;
				num_sets_re_ = 1;
				// Default combined (quasi-)Fisher-Laplace: exact score and (quasi-)Fisher eta curvature for mode finding,
				// observed eta curvature and its derivatives for the determinant. Eta and zeta remain coupled at zero observations.
				if (user_defined_approximation_type_ == "none") {
					approximation_type_ = "laplace";
					if (!user_defined_mode_finding_approach_) use_fisher_for_mode_finding_ = true;
				}
				SetCountApproximationTypeFlags();
				// The (Fisher/observed) information's derivative wrt the mode can vanish at isolated eta values (local extrema of
				// the information as a function of eta); enable the zero-guard used by the iterative ratio-trick diagonal
				// SigmaI_plus_W_inv_diag = d_log_det / deriv_information_diag_loc_par (0/0 -> set to 0 instead of NaN).
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
			}//end zero-inflated count regression variants
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
				aux_pars_ = { 1., 1. };//sigma and lambda
				names_aux_pars_ = { "sigma", "lambda" };
				num_aux_pars_ = 2;
				num_aux_pars_estim_ = 2;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
			}//end "zero_censored_power_transformed_normal"
			else if (likelihood_type_ == "zero_censored_power_transformed_normal_heteroscedastic") {
				// Same model as "zero_censored_power_transformed_normal", but log(sigma) is a second location parameter block
				// related to fixed effects only (no random effects / GPs), so lambda is the only auxiliary parameter
				aux_pars_ = { 1. };//lambda
				names_aux_pars_ = { "lambda" };
				num_aux_pars_ = 1;
				num_aux_pars_estim_ = 1;
				num_sets_re_ = 1;
				num_sets_fixed_effects_ = 2;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
			}//end "zero_censored_power_transformed_normal_heteroscedastic"
			else if (likelihood_type_ == "zoctn") {
				aux_pars_ = { 1., 1., 1. };//sigma, transformed exp(a) (-> a = 0), and b
				names_aux_pars_ = { "sigma", "a", "b" };
				num_aux_pars_ = 3;
				num_aux_pars_estim_ = 3;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
			}
			else if (likelihood_type_ == "zero_one_censored_transformed_beta") {
				aux_pars_ = { 20., 0.01 };//phi and u
				names_aux_pars_ = { "precision", "u" };
				num_aux_pars_ = 2;
				num_aux_pars_estim_ = 2;
				information_ll_can_be_exact_zero_ = true;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
			}
			else if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
				aux_pars_ = { 1., 0.1 };//shape and shift
				names_aux_pars_ = { "shape", "xi" };
				num_aux_pars_ = 2;
				num_aux_pars_estim_ = 2;
				information_ll_can_be_exact_zero_ = true;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
			}
			aux_pars_original_ = aux_pars_;
			BackTransformAuxPars(aux_pars_.data(), aux_pars_original_.data());
			has_SigmaI_mode_ = has_SigmaI_mode;
			use_random_effects_indices_of_data_ = use_random_effects_indices_of_data;
			if (use_random_effects_indices_of_data_) {
				CHECK(random_effects_indices_of_data != nullptr);
				CHECK(Zt == nullptr);
				random_effects_indices_of_data_ = random_effects_indices_of_data;
			}
			else {
				if (Zt != nullptr) {
					use_Z_ = true;
					Zt_ = Zt;
				}
			}
			only_one_grouped_RE_ = only_one_grouped_RE;
			if (only_one_grouped_RE_) {
				CHECK(use_random_effects_indices_of_data_);
				CHECK(random_effects_indices_of_data != nullptr);
				CHECK(Zt == nullptr);
				CHECK(!has_SigmaI_mode);
				iid_model_ = iid_model;
			}
			dim_mode_ = num_sets_re_ * num_re;
			dim_mode_per_set_re_ = num_re;
			dim_location_par_ = num_sets_fixed_effects_ * num_data_;
			if (use_Z_) {
				CHECK(!use_random_effects_indices_of_data_);
				dim_deriv_ll_ = num_sets_re_ * num_data_; // grouped random effects models calculate fist_deriv_ll_ and information_ll_ on the data-scale and the apply the Z^T transformation
				dim_deriv_ll_per_set_re_ = num_data_;
			}
			else {
				dim_deriv_ll_ = dim_mode_;
				dim_deriv_ll_per_set_re_ = dim_mode_per_set_re_;
			}
			if (!use_random_effects_indices_of_data_ && !use_Z_) {
				CHECK(num_data_ == num_re);
			}
			DetermineWhetherToCapChangeModeNewton();
			if (SUPPORTED_APPROX_TYPE_.find(approximation_type_) == SUPPORTED_APPROX_TYPE_.end()) {
				Log::REFatal("'approximation_type' = '%s' is not supported ", approximation_type_.c_str());
			}
			if (has_weights) {
				has_weights_ = true;
				weights_ = weights;
			}
			else {
				if (likelihood_type_ == "binomial_probit" || likelihood_type_ == "binomial_logit" ||
					likelihood_type_ == "beta_binomial") {
					Log::REFatal("'weights' are missing. For the likelihood '%s', 'weights' should contain the number of trials n_i (and 'y' the ratios of successes / trials). If you are sure that the weights are all 1, manually set 'weights' as a vector of 1's ", likelihood_type_.c_str());
				}
				has_weights_ = false;
			}
			CHECK(likelihood_learning_rate > 0.);
			likelihood_learning_rate_ = likelihood_learning_rate;
			if (!TwoNumbersAreEqual<double>(likelihood_learning_rate_, 1.) &&
				!(use_variance_correction_for_prediction_ && var_cor_pred_version_ == "learning_rate")) {
				if (has_weights_) {
					weights_learning_rate_ = Eigen::Map<const vec_t>(weights_, num_data_);
					weights_learning_rate_ *= likelihood_learning_rate_;
				}
				else {
					weights_learning_rate_ = vec_t::Constant(num_data_, likelihood_learning_rate_);
					has_weights_ = true;
				}
				weights_ = weights_learning_rate_.data();
			}//end likelihood_learning_rate_ != 1.
			if (HasEGPDBase() && has_weights_) {
				bool any_positive_weight = false;
				for (data_size_t i = 0; i < num_data_; ++i) {
					if (!std::isfinite(weights_[i]) || weights_[i] < 0.) Log::REFatal("For likelihood='%s', all effective weights must be finite and nonnegative, found %g ", likelihood_type_.c_str(), weights_[i]);
					any_positive_weight = any_positive_weight || weights_[i] > 0.;
					information_ll_can_be_exact_zero_ = information_ll_can_be_exact_zero_ || weights_[i] == 0.;
				}
				if (!any_positive_weight) Log::REFatal("For likelihood='%s', at least one effective weight must be strictly positive ", likelihood_type_.c_str());
			}
			has_int_label_ = label_type() == "int";
			if (iid_model_) {
				maxit_mode_newton_ = 0;
				grad_information_wrt_mode_non_zero_ = false;
				information_changes_during_mode_finding_ = false;
			}
			if (kink_cliping_ && likelihood_type_ != "asymmetric_laplace") {
				kink_cliping_ = false;
			}
			if (kink_cliping_) {
				armijo_condition_ = false;// kink-clipping changes the candidate point (projection), so it’s not on the line anymore where the Armijo condition thinks it is
			}
			// Response-scale conversion can be called concurrently without SetAuxPars() having been called
			// first (e.g. when default auxiliary parameters are kept fixed). Populate the cache eagerly so
			// all response-scale consumers are read-only from the start.
			if (HasEGPDBase()) RefreshEGPDMomentsCache();
		}//end constructor

		void ValidateFixedTweediePower(double p) const {
			if (TwoNumbersAreEqual<double>(p, -999.)) {
				Log::REFatal("No value was provided for 'likelihood_additional_param'. For likelihood='tweedie_fixed_p', provide a fixed power p with %g < p < %g ", TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_);
			}
			if (!std::isfinite(p)) {
				Log::REFatal("For likelihood='tweedie_fixed_p', 'likelihood_additional_param' must be a finite power p with %g < p < %g. Found p = %g ", TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_, p);
			}
			if (p <= 1. || p >= 2.) Log::REFatal("For likelihood='tweedie_fixed_p', only the compound Poisson--Gamma family with 1 < p < 2 is supported. Found p = %g ", p);
			if (p <= TWEEDIE_POWER_LOWER_) Log::REFatal("For likelihood='tweedie_fixed_p', p = %g is too close to 1 for stable Tweedie density evaluation. Choose p > %g. Use likelihood='poisson' explicitly if appropriate ", p, TWEEDIE_POWER_LOWER_);
			if (p >= TWEEDIE_POWER_UPPER_) Log::REFatal("For likelihood='tweedie_fixed_p', p = %g is too close to 2 for stable Tweedie density evaluation. Choose p < %g. Use likelihood='gamma' explicitly if appropriate ", p, TWEEDIE_POWER_UPPER_);
		}

		void ValidateAsymmetricLaplaceQuantile(double quantile) const {
			if (TwoNumbersAreEqual<double>(quantile, -999.)) {
				Log::REFatal("No value was provided for 'likelihood_additional_param'. For likelihood='asymmetric_laplace' (aliases 'quantile' and 'quantile_regression'), provide a quantile q with 0 < q < 1 ");
			}
			if (!std::isfinite(quantile)) {
				Log::REFatal("For likelihood='asymmetric_laplace', 'likelihood_additional_param' must be a finite quantile q with 0 < q < 1. Found q = %g ", quantile);
			}
			if (quantile <= 0. || quantile >= 1.) {
				Log::REFatal("For likelihood='asymmetric_laplace', 'likelihood_additional_param' must be a quantile q with 0 < q < 1. Found q = %g ", quantile);
			}
		}

		inline double GetTweediePower() const {
			if (likelihood_type_ == "tweedie_fixed_p") return aux_pars_[1];
			return TransformTweediePowerFromQ(aux_pars_[1], TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_).p;
		}

		void WarnIfTweediePowerAtBoundary() const {
			if (likelihood_type_ != "tweedie" || tweedie_boundary_warning_issued_) return;
			const auto power = TransformTweediePowerFromQ(aux_pars_[1], TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_);
			if (power.p - TWEEDIE_POWER_LOWER_ < 1e-3 || TWEEDIE_POWER_UPPER_ - power.p < 1e-3) {
				Log::REWarning("The Tweedie power optimizer saturated near the configured boundary (%g, %g), p = %g and dp/dtheta = %g. The estimate may be boundary-sensitive. Inspect a 'tweedie_fixed_p' likelihood and, if appropriate, use 'poisson' (if p close to 1) or 'gamma' (if p close to 2) likelihoods ", TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_, power.p, power.dp_dtheta);
				tweedie_boundary_warning_issued_ = true;
			}
		}

		/*!
		* \brief Transform aux_pars such that they are in (0,infty)
		* \param aux_pars_orig Original aux_pars
		* \param aux_pars_trans Transformed aux_pars
		*/
		void TransformAuxPars(const double* aux_pars_orig,
			double* aux_pars_trans);

		/*!
		* \brief Back-transform aux_pars
		* \param aux_pars_trans Transformed aux_pars
		* \param aux_pars_orig Original aux_pars
		*/
		void BackTransformAuxPars(const double* aux_pars_trans,
			double* aux_pars_orig);

		/*! \brief Set properties (matrix inversion properties, choices for iterative methods, etc.). This function is calle from re_model_template.h which also holds these variables */
		void SetPropertiesLikelihood(const string_t& matrix_inversion_method,
			int cg_max_num_it,
			int cg_max_num_it_tridiag,
			double cg_delta_conv,
			double cg_delta_conv_pred,
			int num_rand_vec_trace,
			bool reuse_rand_vec_trace,
			int seed_rand_vec_trace,
			const string_t& cg_preconditioner_type,
			int fitc_piv_chol_preconditioner_rank,
			int rank_pred_approx_matrix_lanczos,
			int nsim_var_pred,
			double delta_conv_mode_finding) {
			matrix_inversion_method_ = matrix_inversion_method;
			cg_max_num_it_ = cg_max_num_it;
			cg_max_num_it_tridiag_ = cg_max_num_it_tridiag;
			cg_delta_conv_ = cg_delta_conv;
			cg_delta_conv_pred_ = cg_delta_conv_pred;
			num_rand_vec_trace_ = num_rand_vec_trace;
			reuse_rand_vec_trace_ = reuse_rand_vec_trace;
			seed_rand_vec_trace_ = seed_rand_vec_trace;
			cg_preconditioner_type_ = cg_preconditioner_type;
			fitc_piv_chol_preconditioner_rank_ = fitc_piv_chol_preconditioner_rank;
			if (cg_preconditioner_type_ == "pivoted_cholesky") {
				if (fitc_piv_chol_preconditioner_rank_ > dim_mode_) {
					Log::REFatal("'fitc_piv_chol_preconditioner_rank' cannot be larger than the dimension of the mode (= number of unique locations) ");
				}
			}
			rank_pred_approx_matrix_lanczos_ = rank_pred_approx_matrix_lanczos;
			nsim_var_pred_ = nsim_var_pred;
			CHECK(delta_conv_mode_finding > 0.);
			delta_conv_mode_finding_ = delta_conv_mode_finding;
			num_rand_vec_sim_post_ = nsim_var_pred;
			reuse_rand_vec_I_sim_post_ = reuse_rand_vec_trace;
		}//end SetPropertiesLikelihood

		/*!
		* \brief Determine cap_change_mode_newton_
		*/
		void DetermineWhetherToCapChangeModeNewton() {
			if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" || IsEGPDLikelihood() ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				likelihood_type_ == "lognormal" || IsHurdlePositive() || IsZeroInflatedCount()) {
				cap_change_mode_newton_ = true;
			}
			else {
				cap_change_mode_newton_ = false;
			}
		}

		/*!
		* \brief Initialize mode vector_ (used in Laplace approximation for non-Gaussian data)
		*/
		void InitializeModeAvec() {
			if (!mode_is_zero_) {
				mode_ = vec_t::Zero(dim_mode_);
				mode_previous_value_ = vec_t::Zero(dim_mode_);
				if (has_SigmaI_mode_) {
					SigmaI_mode_ = vec_t::Zero(dim_mode_);
					SigmaI_mode_previous_value_ = vec_t::Zero(dim_mode_);
				}
				mode_initialized_ = true;
				first_deriv_ll_ = vec_t(dim_deriv_ll_);
				information_ll_ = vec_t(dim_deriv_ll_);
				if (use_random_effects_indices_of_data_) {
					first_deriv_ll_data_scale_ = vec_t(dim_location_par_);
					information_ll_data_scale_ = vec_t(dim_location_par_);
				}
				if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random" && approximation_type_ == "laplace") {
					off_diag_information_ll_ = vec_t(dim_deriv_ll_per_set_re_);
					if (use_random_effects_indices_of_data_) {
						off_diag_information_ll_data_scale_ = vec_t(num_data_);
					}
				}
				mode_has_been_calculated_ = false;
				na_or_inf_during_last_call_to_find_mode_ = false;
				na_or_inf_during_second_last_call_to_find_mode_ = false;
				mode_is_zero_ = true;
			}
		}

		/*!
		* \brief Reset mode to previous value. This is used if too large step-sizes are done which result in increases in the objective function.
		"           The values (covariance parameters and linear coefficients) are then discarded and consequently the mode should also be reset to the previous value)
		*/
		void ResetModeToPreviousValue() {
			CHECK(mode_initialized_);
			mode_ = mode_previous_value_;
			if (has_SigmaI_mode_) {
				SigmaI_mode_ = SigmaI_mode_previous_value_;
			}
			na_or_inf_during_last_call_to_find_mode_ = na_or_inf_during_second_last_call_to_find_mode_;
			// See the comment in 'RestoreModeState()': all quantities that depend on the mode are not
			//	recalculated here, i.e., the mode is only used as starting value for the next call to a
			//	mode finding algorithm, which recalculates all of them
			mode_has_been_calculated_ = false;
		}

		/*!
		* \brief Save the current mode (and 'SigmaI_mode_') so that it can be restored later with 'RestoreModeState()'.
		*		This is used, e.g., for restarts of the optimizer ('max_num_restarts_lbfgs'): mode finding can be
		*		start-dependent for non-smooth likelihoods, and the objective function values of an optimizer are
		*		thus only reproducible if the corresponding modes are restored as well
		* \param[out] mode Current mode
		* \param[out] SigmaI_mode Current 'SigmaI_mode_' (not used if 'has_SigmaI_mode_' is false)
		*/
		void SaveModeState(vec_t& mode,
			vec_t& SigmaI_mode) const {
			mode = mode_;
			if (has_SigmaI_mode_) {
				SigmaI_mode = SigmaI_mode_;
			}
		}

		/*!
		* \brief Restore a mode that has been saved with 'SaveModeState()'. The mode is used as starting value for
		*		the next call of a mode finding algorithm
		* \param mode Mode
		* \param SigmaI_mode 'SigmaI_mode_' (not used if 'has_SigmaI_mode_' is false)
		*/
		void RestoreModeState(const vec_t& mode,
			const vec_t& SigmaI_mode) {
			CHECK((int)mode.size() == dim_mode_);
			mode_ = mode;
			mode_previous_value_ = mode_;
			if (has_SigmaI_mode_) {
				CHECK((int)SigmaI_mode.size() == dim_mode_);
				SigmaI_mode_ = SigmaI_mode;
				SigmaI_mode_previous_value_ = SigmaI_mode_;
			}
			mode_initialized_ = true;
			mode_is_zero_ = false;
			na_or_inf_during_last_call_to_find_mode_ = false;
			na_or_inf_during_second_last_call_to_find_mode_ = false;
			// All quantities that depend on the mode ('first_deriv_ll_', 'information_ll_', the Cholesky factors,
			//	'approx_marginal_ll_', ...) have been calculated for the mode that is discarded here. They are
			//	not recalculated (this cannot be done here since it requires the covariance matrix / its factorization),
			//	i.e., the restored mode is only used as starting value for the next call to a mode finding algorithm,
			//	which recalculates all of them
			mode_has_been_calculated_ = false;
		}

		/*! \brief Destructor */
		~Likelihood() {
		}

		/*!
		* \brief Returns the type of likelihood
		*/
		string_t GetLikelihood() const {
			return(likelihood_type_);
		}

		/*! \brief True if the information (W) used for the determinant can be negative (observed Hessian). False for the
		* fisher_laplace approximation (nonnegative Fisher information), which is required for the iterative matrix-inversion methods. */
		bool InformationLogLikCanBeNegative() const {
			return information_ll_can_be_negative_;
		}

		bool IsEGPDLikelihood() const {
			return likelihood_type_ == "gpd" || likelihood_type_ == "egpd_power" || likelihood_type_ == "egpd_power_mixture" || likelihood_type_ == "egpd_beta" || likelihood_type_ == "egpd_power_beta";
		}

		/*! \brief True for constant-zero hurdle likelihoods whose positive base is a GPD/EGPD family (auxiliary layout: base EGPD parameters followed by p0). */
		bool IsHurdleEGPD() const {
			return likelihood_type_ == "hurdle_gpd" || likelihood_type_ == "hurdle_egpd_power" || likelihood_type_ == "hurdle_egpd_power_mixture" ||
				likelihood_type_ == "hurdle_egpd_beta" || likelihood_type_ == "hurdle_egpd_power_beta";
		}

		/*! \brief True if the positive base is a GPD/EGPD family (covers constant-zero hurdle EGPD, regression hurdle EGPD, and plain EGPD). */
		bool HasEGPDBase() const {
			if (IsEGPDLikelihood() || IsHurdleEGPD()) return true;
			if (IsHurdleRegression()) {
				const string_t b = EGPDBaseType();
				return b == "gpd" || b == "egpd_power" || b == "egpd_power_mixture" || b == "egpd_beta" || b == "egpd_power_beta";
			}
			return false;
		}

		/*! \brief True for hurdle positive likelihoods with a regression (fixed-effects) structural-zero model: the zero
		* probability is modeled as pi_i = logit^{-1}(x_i^T alpha) via a second fixed-effects-only location-parameter block
		* (zeta = X*alpha), instead of a constant p0. The positive base and the zero model fully decouple (l_{eta,zeta} = 0). */
		bool IsHurdleRegression() const {
			return likelihood_type_ == "hurdle_regression_gamma" || likelihood_type_ == "hurdle_regression_lognormal" ||
				likelihood_type_ == "hurdle_regression_gpd" || likelihood_type_ == "hurdle_regression_egpd_power" ||
				likelihood_type_ == "hurdle_regression_egpd_power_mixture" || likelihood_type_ == "hurdle_regression_egpd_beta" ||
				likelihood_type_ == "hurdle_regression_egpd_power_beta";
		}

		/*! \brief True for any regression (fixed-effects) structural-zero model (hurdle positive or zero-inflated count). */
		bool IsRegressionZeroModel() const {
			return IsHurdleRegression() || IsZeroInflatedCountRegression();
		}

		/*! \brief Recover the underlying constant-zero base likelihood name from a regression zero-model name by removing the
		* "regression_" token that follows the family prefix, e.g. "hurdle_regression_gamma" -> "hurdle_gamma" and
		* "zero_inflated_regression_poisson" -> "zero_inflated_poisson". Returns the input unchanged if the token is absent. */
		/*! \brief Cache the quantities that only depend on 'likelihood_type_'. 'StripRegressionInfix' allocates and
		* concatenates strings, and 'EGPDBaseType()' / 'GetEGPDVariant()' were called once PER OBSERVATION via
		* 'EvaluateEGPD' and the hurdle / zero-inflated regression dispatch functions. Must be called whenever
		* 'likelihood_type_' is assigned */
		void CacheLikelihoodTypeDerivedQuantities() {
			regression_base_type_ = StripRegressionInfix(likelihood_type_);
			string_t t = regression_base_type_;
			const string_t prefix = "hurdle_";
			if (t.size() > prefix.size() && t.compare(0, prefix.size(), prefix) == 0) t = t.substr(prefix.size());
			egpd_base_type_ = t;
			if (t == "gpd") egpd_variant_ = EGPDVariant::kGPD;
			else if (t == "egpd_power") egpd_variant_ = EGPDVariant::kPower;
			else if (t == "egpd_power_mixture") egpd_variant_ = EGPDVariant::kPowerMixture;
			else if (t == "egpd_beta") egpd_variant_ = EGPDVariant::kBeta;
			else egpd_variant_ = EGPDVariant::kPowerBeta;
		}

		static string_t StripRegressionInfix(const string_t& name) {
			const string_t infix = "_regression_";
			const size_t pos = name.find(infix);
			if (pos == string_t::npos) return name;
			return name.substr(0, pos) + "_" + name.substr(pos + infix.size());
		}

		/*! \brief For a hurdle regression likelihood, the underlying constant-zero hurdle type (with the "regression_" token removed). */
		const string_t& HurdleRegressionBaseType() const {
			return regression_base_type_;
		}

		/*! \brief True for hurdle likelihoods with a positive continuous base (point mass at zero + positive density; the zero part fully decouples from the response predictor eta). Includes both the constant-zero and the regression (fixed-effects) zero models. */
		bool IsHurdlePositive() const {
			return likelihood_type_ == "hurdle_gamma" || likelihood_type_ == "hurdle_lognormal" || IsHurdleEGPD() || IsHurdleRegression();
		}

		/*! \brief The underlying EGPD base type of a (possibly hurdle / hurdle-regression) EGPD likelihood, i.e. with any "hurdle_" prefix and "regression_" token stripped. */
		const string_t& EGPDBaseType() const {
			return egpd_base_type_;
		}

		/*! \brief True for zero-inflated count likelihoods (constant or regression structural-zero model). The base count component can itself generate zeros. */
		bool IsZeroInflatedCount() const {
			return likelihood_type_ == "zero_inflated_poisson" || likelihood_type_ == "zero_inflated_regression_poisson" ||
				likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_regression_negative_binomial" ||
				likelihood_type_ == "zero_inflated_negative_binomial_1" || likelihood_type_ == "zero_inflated_regression_negative_binomial_1";
		}

		/*! \brief True for zero-inflated COUNT likelihoods with a regression (fixed-effects) structural-zero model: pi_i = logit^{-1}(x_i^T alpha)
		* via a second fixed-effects-only location-parameter block (zeta). Unlike the hurdle case, the count base and the zero model COUPLE at
		* zero counts (l_{eta,zeta} != 0, dJ_eta/dzeta != 0). */
		bool IsZeroInflatedCountRegression() const {
			return likelihood_type_ == "zero_inflated_regression_poisson" || likelihood_type_ == "zero_inflated_regression_negative_binomial" ||
				likelihood_type_ == "zero_inflated_regression_negative_binomial_1";
		}

		/*! \brief The underlying constant-zero count type for a zero-inflated count regression likelihood (with the "regression_" token removed). */
		const string_t& ZICountRegressionBaseType() const {
			return regression_base_type_;
		}

		/*! \brief Stable log(exp(a) + exp(b)) */
		static inline double LogAddExpStable(double a, double b) {
			const double m = a > b ? a : b;
			if (!std::isfinite(m)) return m;
			return m + std::log1p(std::exp(-std::fabs(a - b)));
		}

		EGPDVariant GetEGPDVariant() const {
			return egpd_variant_;
		}

		EGPDParams GetEGPDParams() const {
			const string_t& base = EGPDBaseType();
			EGPDParams pars;
			pars.shape_shift = aux_pars_[0];
			if (base == "egpd_power") pars.kappa = aux_pars_[1];
			else if (base == "egpd_power_mixture") {
				pars.kappa1 = aux_pars_[1]; pars.delta_kappa = aux_pars_[2]; pars.odds = aux_pars_[3];
			}
			else if (base == "egpd_beta") pars.delta = aux_pars_[1];
			else if (base == "egpd_power_beta") { pars.delta = aux_pars_[1]; pars.kappa = aux_pars_[2]; }
			return pars;
		}

		/*! \brief Number of base (non-structural-zero) auxiliary parameters for a hurdle EGPD likelihood. */
		int NumEGPDBaseAuxPars() const {
			const string_t base = EGPDBaseType();
			if (base == "gpd") return 1;
			if (base == "egpd_power" || base == "egpd_beta") return 2;
			if (base == "egpd_power_beta") return 3;
			return 4;// egpd_power_mixture
		}

		EGPDDerivatives EvaluateEGPD(double y, double eta) const {
			EGPDDerivatives result;
			CalcEGPDLogLikAndDerivatives(y, eta, GetEGPDParams(), GetEGPDVariant(), &result);
			return result;
		}

		// The EGPD unit-scale moments depend only on the BASE EGPD auxiliary parameters (those read by
		// GetEGPDParams(), i.e. the leading NumEGPDBaseAuxPars() entries of aux_pars_), not on the location, the
		// data, or a hurdle structural-zero parameter p0. They require numerical quadrature, so they are cached.
		// Note: the cache key must be indexed by NumEGPDBaseAuxPars() and not by num_aux_pars_ -- the latter is 5
		// for 'hurdle_egpd_power_mixture' and would write past the end of the kMaxEGPDAuxPars(=4)-sized array.
		// Refresh this cache only in single-threaded parameter update paths; response-scale consumers can then
		// safely read it concurrently.
		bool EGPDMomentsCacheMatchesAuxPars() const {
			if (!egpd_moments_cache_initialized_) return false;
			const int nb = NumEGPDBaseAuxPars();
			CHECK(nb <= kMaxEGPDAuxPars);
			for (int i = 0; i < nb; ++i) {
				if (egpd_moments_cache_aux_[i] != aux_pars_[i]) return false;
			}
			return true;
		}

		void RefreshEGPDMomentsCache() {
			egpd_moments_cache_ = CalcEGPDUnitScaleMoments(GetEGPDParams(), GetEGPDVariant());
			const int nb = NumEGPDBaseAuxPars();
			CHECK(nb <= kMaxEGPDAuxPars);
			for (int i = 0; i < nb; ++i) egpd_moments_cache_aux_[i] = aux_pars_[i];
			egpd_moments_cache_initialized_ = true;
		}

		const EGPDMoments& GetEGPDMoments() const {
			CHECK(EGPDMomentsCacheMatchesAuxPars());
			return egpd_moments_cache_;
		}

		bool IsGaussianLikelihood() const {
			return(likelihood_type_ == "gaussian" || likelihood_type_ == "gaussian_latent");
		}

		/*!
		* \brief True for both heteroscedastic Gaussian likelihoods ('gaussian_heteroscedastic' and 'gaussian_heteroscedastic_fixed_and_random').
		*		These two likelihoods share the same location parameter layout (location_par[i] = mean, location_par[i + num_data_] = log-error variance)
		*		and thus the same likelihood / gradient / information formulas wrt the location parameter. They differ in how the
		*		log-error variance block is obtained: exclusively from fixed effects for 'gaussian_heteroscedastic' vs. from the sum of fixed
		*		and random effects for 'gaussian_heteroscedastic_fixed_and_random' (num_sets_re_ = 1 vs. 2, respectively)
		*/
		bool IsGaussianHeteroscedastic() const {
			return(likelihood_type_ == "gaussian_heteroscedastic" || likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random");
		}

		/*!
		* \brief True for the heteroscedastic zero-censored power-transformed normal likelihood, where the standard deviation
		*		of the latent normal variable is modeled by a second, fixed-effects-only location parameter block
		*		(location_par[i + num_data_] = log(sigma_i)) instead of by the auxiliary parameter "sigma"
		*/
		bool IsZeroCensPowNormHetero() const {
			return(likelihood_type_ == "zero_censored_power_transformed_normal_heteroscedastic");
		}

		/*! \brief True for both zero-censored power-transformed normal likelihoods (homoscedastic and heteroscedastic) */
		bool IsZeroCensPowNorm() const {
			return(likelihood_type_ == "zero_censored_power_transformed_normal" || IsZeroCensPowNormHetero());
		}

		/*!
		* \brief True for likelihoods whose second, fixed-effects-only location parameter block needs the diagonal of
		*		(Sigma^-1 + W)^-1 for its gradient ('gaussian_heteroscedastic' and
		*		'zero_censored_power_transformed_normal_heteroscedastic'). Used to force the calculation of that diagonal
		*		in the gradient functions also when it is not needed for the mean / eta block
		*/
		bool SecondFEBlockNeedsSigmaIPlusWInvDiag() const {
			return(likelihood_type_ == "gaussian_heteroscedastic" || IsZeroCensPowNormHetero());
		}

		/*!
		* \brief True for likelihoods with a second, fixed-effects-only block of the location parameter (the "zeta" block,
		*		location_par[i + num_data_]): the log-error variance of 'gaussian_heteroscedastic', the log(sigma) of
		*		'zero_censored_power_transformed_normal_heteroscedastic', and the structural-zero predictor of a hurdle /
		*		zero-inflated count regression. The gradient of all of these is calculated by 'CalcSecondFEBlockFixedEffectGrad'.
		*		NOTE: this excludes 'gaussian_heteroscedastic_fixed_and_random', whose second block is a genuine second set of
		*		random effects (num_sets_re_ == 2) and is handled by the loops over 'num_sets_re_'
		*/
		bool HasSecondFEBlock() const {
			return(likelihood_type_ == "gaussian_heteroscedastic" || IsZeroCensPowNormHetero() || IsRegressionZeroModel());
		}

		/*!
		* \brief True if 'CalcSecondFEBlockFixedEffectGrad' reads its 'diag' argument, i.e. if the zeta block has a
		*		log-determinant term. False only for a zero model that contributes nothing but the direct score (a hurdle
		*		regression, which decouples exactly, or a zero-inflated count regression whose coupled terms are dropped).
		*		Lets a caller skip an expensive calculation of that diagonal
		* \param include_coupled_zi_terms As in 'CalcSecondFEBlockFixedEffectGrad'
		*/
		bool SecondFEBlockGradNeedsDiag(bool include_coupled_zi_terms) const {
			if (iid_model_) {
				return false;// no random effect / mode at all, so both correction terms vanish
			}
			return(!IsRegressionZeroModel() || (!IsHurdleRegression() && include_coupled_zi_terms));
		}

		/*!
		* \brief True if 'CalcSecondFEBlockFixedEffectGrad' reads its 'impl' argument, i.e. if the zeta block has an
		*		implicit-through-the-mode term. Additionally false for 'gaussian_heteroscedastic', whose zeta block has no
		*		mode at all (l_eta_zeta = 0)
		* \param include_coupled_zi_terms As in 'CalcSecondFEBlockFixedEffectGrad'
		*/
		bool SecondFEBlockGradNeedsImpl(bool include_coupled_zi_terms) const {
			return(SecondFEBlockGradNeedsDiag(include_coupled_zi_terms) && likelihood_type_ != "gaussian_heteroscedastic");
		}

		/*!
		* \brief Pick the diagonal of (Sigma^-1+W)^-1 that the zeta block's log-determinant term has to use, for the
		*		approximations that calculate a separate estimate of it for that block. The heteroscedastic likelihoods use
		*		the separate estimate, since for them the eta block's version is not usable (dJ_eta/deta vanishes where it
		*		matters, see 'SecondFEBlockNeedsSigmaIPlusWInvDiag'), whereas a zero-model regression uses the eta block's
		*		version (for which no separate estimate is calculated in the first place)
		* \param eta_block_diag The diagonal calculated for the eta block
		* \param second_block_diag The diagonal calculated separately for the zeta block
		*/
		const vec_t& SecondFEBlockZetaDiag(const vec_t& eta_block_diag, const vec_t& second_block_diag) const {
			return SecondFEBlockNeedsSigmaIPlusWInvDiag() ? second_block_diag : eta_block_diag;
		}

		/*!
		* \brief Set the type of likelihood
		* \param type Likelihood name
		*/
		void SetLikelihood(const string_t& type) {
			string_t likelihood = ParseLikelihoodAliasModeRefinement(type);
			likelihood = ParseLikelihoodAlias(likelihood);
			likelihood = ParseLikelihoodAliasModeFindingMethod(likelihood);
			if (SUPPORTED_LIKELIHOODS_.find(likelihood) == SUPPORTED_LIKELIHOODS_.end()) {
				Log::REFatal("Likelihood of type '%s' is not supported.", likelihood.c_str());
			}
			likelihood_type_ = likelihood;
			CacheLikelihoodTypeDerivedQuantities();
			chol_fact_pattern_analyzed_ = false;
			DetermineWhetherToCapChangeModeNewton();
		}

		/*!
		* \brief True if likelihood type is supported
		*/
		bool LikelihoodSupported(const string_t& type) {
			string_t likelihood = ParseLikelihoodAlias(type);
			return(SUPPORTED_LIKELIHOODS_.find(likelihood) != SUPPORTED_LIKELIHOODS_.end());
		}

		bool UseFisherForModeFinding() const {
			return use_fisher_for_mode_finding_ || (approximation_type_  == "fisher_laplace");
		}

		/*!
		* \brief Returns the number of sets of random effects / GPs. This is larger than 1, e.g., heteroscedastic models
		*/
		int GetNumSetsRE() const {
			return(num_sets_re_);
		}

		/*!
		* \brief Returns the number of sets of fixed effects (covariates / boosting scores). This is larger than num_sets_re_
		*		 for likelihoods where some location parameter blocks are related to fixed effects only (no random effects / GPs), e.g., 'gaussian_heteroscedastic'
		*/
		int GetNumSetsFixedEffects() const {
			return(num_sets_fixed_effects_);
		}

		/*!
		* \brief Returns the dimension of the mode per parameter / number of sets of random effects / GPs
		*/
		data_size_t GetDimModePerSetsRE() const {
			return(dim_mode_per_set_re_);
		}

		/*!
		* \brief True if this likelihood requires latent predictive variances for predicting response means
		* \return need_pred_latent_var_for_response_mean_
		*/
		bool NeedPredLatentVarForResponseMean() const {
			return(need_pred_latent_var_for_response_mean_);
		}

		/*!
		* \brief Set chol_fact_pattern_analyzed_ to false
		*/
		void SetCholFactPatternAnalyzedFalse() {
			chol_fact_pattern_analyzed_ = false;
		}

		/*!
		* \brief Returns the type of the response variable (label). Either "double" or "int"
		*/
		string_t label_type() const {
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit" ||
				likelihood_type_ == "poisson" || likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				IsZeroInflatedCount()) {
				return("int");
			}
			else {
				return("double");
			}
		}

		/*!
		* \brief Returns the number of CG steps when the CG method was last run
		*/
		int GetNumCGSteps() const {
			return(num_cg_steps_last_);
		}

		/*!
		* \brief Returns the number of CG steps when the CG method was last run for the SLQ method
		*/
		int GetNumCGStepsTridiag() const {
			return(num_cg_steps_tridiag_last_);
		}

		int GetNumModeFindingSteps() const {
			return(num_it_mode_finding_);
		}

		/*!
		* \brief Returns a pointer to mode_
		*/
		const vec_t* GetMode() const {
			return(&mode_);
		}

		/*!
		* \brief Returns a pointer to first_deriv_ll_
		*/
		const vec_t* GetFirstDerivLL() const {
			return(&first_deriv_ll_);
		}

		/*!
		* \brief Checks whether the response variables (labels) have the correct values
		* \param y_data Response variable data
		* \param num_data Number of data points
		*/
		template <typename T>//T can be double or float
		void CheckY(const T* y_data, data_size_t num_data) const {
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit") {
				//#pragma omp parallel for schedule(static)//problematic with error message below, not worth parallelizing as other overheads (e.g. model construction) dominate
				for (data_size_t i = 0; i < num_data; ++i) {
					if (fabs(y_data[i]) >= EPSILON_NUMBERS && !TwoNumbersAreEqual<T>(y_data[i], 1.)) {
						Log::REFatal("The response variable ('y') needs to be 0 or 1 for likelihood = '%s' ", likelihood_type_.c_str());
					}
				}
			}
			else if (likelihood_type_ == "binomial_probit" || likelihood_type_ == "binomial_logit" || likelihood_type_ == "beta_binomial" || 
				likelihood_type_ == "quasi_bernoulli_probit" || likelihood_type_ == "quasi_bernoulli_logit") {
				for (data_size_t i = 0; i < num_data; ++i) {// not worth parallelizing as other overheads (e.g. model construction) dominate
					if (y_data[i] < 0. || y_data[i] > 1.) {
						Log::REFatal(" Must have 0 <= y <= 1 for the response variable ('y') for likelihood = '%s', found %g. Note that the response variable should be the proportion of successes / trials ", likelihood_type_.c_str(), y_data[i]);
					}
				}
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				IsZeroInflatedCount()) {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] < 0.) {
						Log::REFatal(" Must have y >= 0 for the response variable ('y') for likelihood = '%s', found %g ", likelihood_type_.c_str(), y_data[i]);
					}
					else {
						double intpart;
						if (std::modf(y_data[i], &intpart) != 0.0) {
							Log::REFatal("Found non-integer response variable ('y'). Response variable can only be integer valued for likelihood = '%s' ", likelihood_type_.c_str());
						}
					}
				}
				if (IsZeroInflatedCount()) {
					double sw = 0., avg_zero = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.) avg_zero += w;
						sw += w;
					}
					avg_zero /= sw;
					if (TwoNumbersAreEqual<double>(avg_zero, 1.)) {
						Log::REFatal("Only 0's in the response variable 'y' but used likelihood = '%s' ", likelihood_type_.c_str());
					}
				}
			}
			else if (likelihood_type_ == "gamma" || likelihood_type_ == "lognormal" || IsEGPDLikelihood()) {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (!std::isfinite(y_data[i]) || y_data[i] <= 0.) {
						Log::REFatal(" Must have y > 0 for the response variable ('y') for likelihood = '%s', found %g ", likelihood_type_.c_str(), y_data[i]);
					}
				}
			}
			else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
				bool any_positive = false;
				for (data_size_t i = 0; i < num_data; ++i) {
					if (!std::isfinite(y_data[i]) || y_data[i] < 0.) Log::REFatal("The response variable ('y') must be finite and nonnegative for likelihood = '%s', found %g.", likelihood_type_.c_str(), y_data[i]);
					any_positive = any_positive || y_data[i] > 0.;
				}
				if (!any_positive) Log::REFatal("The response variable ('y') contains only zeros for likelihood = '%s'; at least one positive value is required.", likelihood_type_.c_str());
			}
			else if (IsHurdlePositive() || IsZeroCensPowNorm()) {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (!std::isfinite(y_data[i]) || y_data[i] < 0.) {
						Log::REFatal(" Must have finite y >= 0 for the response variable ('y') for likelihood = '%s', found %g ", likelihood_type_.c_str(), y_data[i]);
					}
				}
				double sw = 0., avg_zero = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					if (y_data[i] <= 0.) {
						avg_zero += w;
					}
					sw += w;
				}
				avg_zero /= sw;
				if (GPBoost::IsZero<double>(avg_zero)) {
					Log::REWarning("No 0's in the response variable 'y' but used likelihood = '%s ", likelihood_type_.c_str());
				}
				if (TwoNumbersAreEqual<double>(avg_zero, 1.)) {
					Log::REFatal("Only 0's in the response variable 'y' but used likelihood = '%s' ", likelihood_type_.c_str());
				}
			}
			else if (likelihood_type_ == "beta") {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] <= 0. || y_data[i] >= 1.) {
						Log::REFatal(" Must have 0 < y < 1 for the response variable ('y') for likelihood = '%s', found %g ", likelihood_type_.c_str(), y_data[i]);
					}
				}
			}
			else if (likelihood_type_ == "zoctn" || likelihood_type_ == "zero_one_censored_transformed_beta" ||
				likelihood_type_ == "zero_one_censored_shifted_gamma") {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] < 0. || y_data[i] > 1.) {
						Log::REFatal(" Must have 0 <= y <= 1 for the response variable ('y') for likelihood = '%s', found %g ", likelihood_type_.c_str(), y_data[i]);
					}
				}
				double sw = 0., avg_zero = 0., avg_one = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero, avg_one)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					if (y_data[i] <= 0.) {
						avg_zero += w;
					}
					if (y_data[i] >= 1.) {
						avg_one += w;
					}
					sw += w;
				}
				avg_zero /= sw;
				avg_one /= sw;
				if (TwoNumbersAreEqual<double>(avg_zero, 1.)) {
					Log::REFatal("Only 0's in the response variable 'y' but used likelihood = '%s' ", likelihood_type_.c_str());
				}
				if (TwoNumbersAreEqual<double>(avg_one, 1.)) {
					Log::REFatal("Only 1's in the response variable 'y' but used likelihood = '%s' ", likelihood_type_.c_str());
				}
				if (GPBoost::IsZero<double>(avg_zero) && GPBoost::IsZero<double>(avg_one)) {
					Log::REWarning("No 0's and 1's in the response variable 'y' but used likelihood = '%s ", likelihood_type_.c_str());
				}
			}
			else if (!IsGaussianLikelihood() && likelihood_type_ != "t" && !IsGaussianHeteroscedastic() &&
				likelihood_type_ != "asymmetric_laplace") {
				NotSupportedForLikelihood(__func__);
			}
		}//end CheckY

		/*!
		* \brief Method-of-moments anchors for the mean and the log standard deviation of the latent normal variable of the
		*		'zero_censored_power_transformed_normal_heteroscedastic' likelihood at a given lambda. With u_i = y_i^(1/lambda)
		*		for the positive observations, the latent X is a normal variable censored at 0, so that the zero fraction
		*		p0 = Phi(a) with a = -mu / sigma anchors the standardized threshold, and the truncated normal identities
		*		Var[X | X > 0] = sigma^2 * (1 + a * tau(a) - tau(a)^2) and E[X | X > 0] = mu + sigma * tau(a), with
		*		tau(a) = phi(a) / (1 - Phi(a)), give sigma and mu from the sample moments of the u_i
		* \param y_data Response variable data
		* \param num_data Number of data points
		* \param weights_ptr Sample weights (can be a nullptr if 'has_weights_' is false)
		* \param lambda Power transformation parameter
		* \param[out] mu_anchor Anchor for the mean of the latent normal variable
		* \param[out] log_sigma_anchor Anchor for the log standard deviation of the latent normal variable
		*/
		void ZeroCensPowNormHeteroAnchors(const double* y_data,
			const data_size_t num_data,
			const double* weights_ptr,
			double lambda,
			double& mu_anchor,
			double& log_sigma_anchor) const;

		/*!
		* \brief Determine initial value for intercept (=constant)
		* \param y_data Response variable data
		* \param num_data Number of data points
		* \param rand_eff_var Variance of random effects
		* \param fixed_effects Additional fixed effects that are added to the linear predictor (= offset)
		* \param ind_set_re Conuter for number of GPs / REs (e.g. for heteroscedastic GPs)
		*/
		double FindInitialIntercept(const double* y_data,
			const data_size_t num_data,
			double rand_eff_var,
			const double* fixed_effects,
			int ind_set_re,
			const double* weights = nullptr) const;

		/*!
		* \brief Should there be an intercept or not (might raise a warning)
		* \param y_data Response variable data
		* \param num_data Number of data points
		* \param rand_eff_var Variance of random effects
		* \param fixed_effects Additional fixed effects that are added to the linear predictor (= offset)
		*/
		bool ShouldHaveIntercept(const double* y_data,
			const data_size_t num_data,
			double rand_eff_var,
			const double* fixed_effects,
			const double* weights = nullptr) const;

		/*!
		* \brief Determine initial value for additional likelihood parameters (e.g., shape for gamma)
		* \param y_data Response variable data
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		*/
		const double* FindInitialAuxPars(const double* y_data,
			const double* fixed_effects,
			const data_size_t num_data);

		/*!
		* \brief Determine constants C_mu and C_sigma2 used for checking whether step sizes for linear regression coefficients are clearly too large
		* \param y_data Response variable data
		* \param num_data Number of data points
		* \param fixed_effects Additional fixed effects that are added to the linear predictor (= offset)
		* \param[out] C_mu
		* \param[out] C_sigma2
		*/
		void FindConstantsCapTooLargeLearningRateCoef(const double* y_data,
			const data_size_t num_data,
			const double* fixed_effects,
			double& C_mu,
			double& C_sigma2,
			const double* weights = nullptr) const;

		/*!
		* \brief Returns the number of additional parameters
		*/
		int NumAuxPars() const {
			return(num_aux_pars_);
		}

		/*!
		* \brief Returns a pointer to aux_pars_
		*/
		const double* GetAuxPars() const {
			return(aux_pars_.data());
		}

		/*!
		* \brief Set aux_pars_
		* \param aux_pars New values for aux_pars_
		*/
		void SetAuxPars(const double* aux_pars);

		const char* GetNameAuxPars(int ind_aux_par) const {
			CHECK(ind_aux_par < num_aux_pars_);
			return(names_aux_pars_[ind_aux_par].c_str());
		}

		void GetNamesAuxPars(string_t& name) const {
			name = names_aux_pars_[0];
			for (int i = 1; i < num_aux_pars_; ++i) {
				name += "_SEP_" + names_aux_pars_[i];
			}
		}

		/*! \brief True if (effective) weights are used */
		bool HasWeights() const {
			return(has_weights_);
		}

		/*! \brief Returns a pointer to the EFFECTIVE weights of this object (i.e. including a potential
		* scaling by 'likelihood_learning_rate_'). The array has length 'num_data_', i.e. it only contains
		* the weights of the cluster of this object. Returns nullptr if no weights are used */
		const double* GetWeights() const {
			return(has_weights_ ? weights_ : nullptr);
		}

		bool AuxParsHaveBeenSet() const {
			return(aux_pars_have_been_set_);
		}

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood..
		*       Calculations are done using a numerically stable variant based on factorizing ("inverting") B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt).
		*       In the notation of the paper: "Sigma = Z*Sigma*Z^T" and "Z = Id".
		*       This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		*       If use_random_effects_indices_of_data_, calculations are done on the random effects (b) scale and not the "data scale" (Zb)
		*       factorizing ("inverting") B = (Id + ZtWZsqrt * Sigma * ZtWZsqrt).
		*       This version (use_random_effects_indices_of_data_ == true) is used for the Laplace approximation when there is only one Gaussian process and
		*       there are multiple observations at the same location, i.e., the dimenion of the random effects b is much smaller than Zb
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param Sigma Covariance matrix of latent random effects ("Sigma = Z*Sigma*Z^T" if !use_random_effects_indices_of_data_)
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<T_mat>& Sigma,
			double& approx_marginal_ll);

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*       Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*       NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*       This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param SigmaI Inverse covariance matrix of latent random effect. Currently, this needs to be a diagonal matrix
		* \param True, if grouped REs are used together with a Vecchia-approximated GP
		* \param B Matrix B in Vecchia approximation with grouped REs such that Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation with grouped REs
		* \param first_update If true, the covariance parameters or linear regression coefficients are updated for the first time and the max. number of iterations for the CG should be decreased
		* \param calc_mll If true the marginal log-likelihood is also calculated (only relevant for matrix_inversion_method_ == "iterative")
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLGroupedRE(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const sp_mat_t& SigmaI,
			bool has_vecchia_gp,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const bool first_update,
			bool calc_mll,
			double& approx_marginal_ll);

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*       Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*       This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param sigma2 Variance of random effects
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const double sigma2,
			double& approx_marginal_ll);

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*       Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*       of Sigma^-1 has previously been calculated using a Full-scale Vecchia approximation.
		*       This version is used for the Laplace approximation when there are only GP random effects and the Full-scale Vecchia approximation is used.
		*       Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param sigma_ip Covariance matrix of inducing point process
		* \param chol_fact_sigma_ip Cholesky factor of 'sigma_ip'
		* \param chol_fact_sigma_woodbury Cholesky factor of 'sigma_ip + sigma_cross_cov_T * sigma_residual^-1 * sigma_cross_cov'
		* \param cross_cov Cross-covariance matrix between inducing points and all data points
		* \param sigma_woodbury Matrix 'sigma_ip + sigma_cross_cov_T * sigma_residual^-1 * sigma_cross_cov'
		* \param B Matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation Sigma^-1 = B^T D^-1 B
		* \param first_update If true, the covariance parameters or linear regression coefficients are updated for the first time and the max. number of iterations for the CG should be decreased
		* \param Sigma_L_k Pivoted Cholseky decomposition of Sigma - Version Habrecht: matrix of dimension nxk with rank(Sigma_L_k_) <= fitc_piv_chol_preconditioner_rank generated in re_model_template.h
		* \param calc_mll If true the marginal log-likelihood is also calculated (only relevant for matrix_inversion_method_ == "iterative")
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLFSVA(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const den_mat_t& sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_woodbury,
			const den_mat_t& chol_ip_cross_cov,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			const den_mat_t& sigma_woodbury,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const den_mat_t& Bt_D_inv_B_cross_cov,
			const den_mat_t& D_inv_B_cross_cov,
			const bool first_update,
			bool calc_mll,
			double& approx_marginal_ll,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_preconditioner_cluster_i,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i,
			const den_mat_t& chol_ip_cross_cov_preconditioner,
			const chol_den_mat_t& chol_fact_sigma_ip_preconditioner,
			bool GPU_use);

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param B Matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation Sigma^-1 = B^T D^-1 B
		* \param first_update If true, the covariance parameters or linear regression coefficients are updated for the first time and the max. number of iterations for the CG should be decreased
		* \param Sigma_L_k Pivoted Cholseky decomposition of Sigma - Version Habrecht: matrix of dimension nxk with rank(Sigma_L_k_) <= fitc_piv_chol_preconditioner_rank generated in re_model_template.h
		* \param calc_mll If true the marginal log-likelihood is also calculated (only relevant for matrix_inversion_method_ == "iterative")
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		* \param re_comps_ip_cluster_i IP component for FITC preconditioner
		* \paramre_comps_cross_cov_cluster_i cross-coariance component for FITC preconditioner
		* \param chol_fact_sigma_ip Cholesky factor of IP component for FITC preconditioner
		* \param cluster_i Cluster index for which this is run
		* \param REModelTemplate REModelTemplate object for calling functions from it
		*/
		void FindModePostRandEffCalcMLLVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			std::map<int, sp_mat_t>& B,
			std::map<int, sp_mat_t>& D_inv,
			const bool first_update,
			const den_mat_t& Sigma_L_k,
			bool calc_mll,
			double& approx_marginal_ll,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			const den_mat_t& chol_ip_cross_cov,
			const chol_den_mat_t& chol_fact_sigma_ip,
			data_size_t cluster_i,
			REModelTemplate<T_mat, T_chol>* re_model);

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and
		*           calculate the approximative marginal log-likelihood when the 'fitc' aproximation is used
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param sigma_ip Covariance matrix of inducing point process
		* \param chol_fact_sigma_ip Cholesky factor of 'sigma_ip'
		* \param cross_cov Cross-covariance matrix between inducing points and all data points
		* \param fitc_resid_diag Diagonal correction of predictive process
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLFITC(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<den_mat_t> sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_ip,
			const den_mat_t* cross_cov,
			const vec_t& fitc_resid_diag,
			double& approx_marginal_ll,
			bool GPU_use);

		/*!
		* \brief Refine the posterior mode of an 'asymmetric_laplace' likelihood with a semismooth Newton method applied to
		*			the subproblems of an augmented Lagrangian method (SSN-ALM). This is the common, approximation-independent
		*			driver: it performs the exact KKT check that gates the refinement, the observation-scale prox and
		*			multiplier updates, the line search on the reduced augmented Lagrangian, and the outer penalty updates.
		*			The only approximation-specific ingredient is the linear solve with the generalized Hessian, which is
		*			passed in by the calling 'FindModePostRandEffCalcMLL*' routine.
		*			On success 'mode_', 'Qmode', 'location_par' / '*location_par_ptr', and 'approx_marginal_ll' (the
		*			log-posterior at the mode, i.e., without the log-determinant term) are updated consistently. The mode is
		*			only accepted if it improves the exact non-smooth MAP objective, so the refinement can never make the
		*			returned mode worse than the one found by the quasi-Newton iteration
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of the location parameter
		* \param solve_H Solves '(Q + Z^T W Z) d = rhs' for 'd', where W is the active-set curvature in the convention of
		*			'information_ll_' (see 'AggregateActiveSetToInformationScale'). Must return false if the solve failed.
		*			It must not overwrite any member matrix or factorization that the final Laplace approximation uses
		* \param apply_Q Applies the prior precision Q to a vector. Along the semismooth Newton steps 'Q b' is maintained
		*			incrementally as 'Q d = -g - Z^T W Z d', which avoids an application of Q per line search step but
		*			accumulates rounding errors; if this callback is not empty, it is used to recompute 'Q b' exactly once
		*			per outer iteration. Pass an empty function on the branches where Q is not available cheaply
		* \param[in,out] Qmode Q * mode_ (the prior precision applied to the mode), maintained by the driver
		* \param[in,out] location_par Location parameter (see 'UpdateLocationParNewMode')
		* \param[in,out] location_par_ptr Pointer to the location parameter
		* \param[in,out] approx_marginal_ll Log-posterior at the mode without the log-determinant term
		* \return True if the mode was changed
		*/
		bool RefineModeAsymLaplaceSSNALM(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::function<bool(const vec_t&, const vec_t&, vec_t&)>& solve_H,
			const std::function<void(const vec_t&, vec_t&)>& apply_Q,
			vec_t& Qmode,
			vec_t& location_par,
			double** location_par_ptr,
			double& approx_marginal_ll);

		/*!
		* \brief Matrix-free preconditioned conjugate gradient for the semismooth Newton system '(Q + Z^T W Z) d = rhs' of
		*			the SSN-ALM refinement, used by the branches with 'matrix_inversion_method_ == "iterative"'.
		*			A Jacobi (diagonal) preconditioner is used since the active-set curvature 'rho * A' contains exact zeros,
		*			which is incompatible with the preconditioners of the Laplace approximation that rely on W^(-1)
		*			(in particular the FITC preconditioner of the Vecchia and full-scale Vecchia branches)
		* \param apply_H Applies the generalized Hessian to a vector
		* \param diag_H Diagonal of the generalized Hessian (must be strictly positive)
		* \param rhs Right-hand side
		* \param[out] sol Solution
		* \return True if the solve succeeded
		*/
		bool SolveSSNALMCG(const std::function<void(const vec_t&, vec_t&)>& apply_H,
			const vec_t& diag_H,
			const vec_t& rhs,
			vec_t& sol) const;

		/*!
		* \brief Stochastic (Hutchinson) estimate of the DATA-scale diagonal of Z (Sigma^-1 + Z^T W Z)^-1 Z^T on the iterative
		*       grouped random effects path. Needed for the log-determinant term of the second (zeta) block's fixed-effect
		*       gradient: the ratio trick used for the eta block is not applicable there since dJ_eta/deta vanishes at all
		*       observations that matter for these likelihoods.
		*       NOTE: 'SigmaI_plus_ZtWZ_inv_RV_' cannot be reused here since it is solved against the preconditioned vectors
		*       'rand_vec_trace_P_' (Cov = P, the preconditioner), whereas an unbiased diagonal estimate requires solving
		*       against the raw vectors 'rand_vec_trace_I_' (Cov = I, generated for the log-determinant's trace estimation)
		* \return The estimated diagonal, of length num_data_
		*/
		vec_t CalcStochDataScaleDiagSigmaIPlusZtWZInv() const;

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt the fixed effects of
		*       the second, fixed-effects-only block of the location parameter (the "zeta" block, see 'HasSecondFEBlock').
		*       All likelihoods with such a block share the same three-term structure
		*           d(-mll)/dzeta_i = -w_i * (dl/dzeta)_i  +  0.5 * w_i * (dJ_eta/dzeta)_i * diag_i  +  w_i * (l_eta_zeta)_i * impl_i,
		*       i.e., direct score + log-determinant term + implicit-through-the-mode term, and they differ only in the three
		*       likelihood-specific quantities (dl/dzeta, dJ_eta/dzeta, l_eta_zeta). The likelihood is dispatched once here,
		*       outside the loops over the data, so that no per-observation type comparison is done.
		*       The eta block of 'fixed_effect_grad' must already have been filled in by the caller.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (both blocks, i.e. of length dim_location_par_)
		* \param information_data_scale Diagonal of W on the DATA scale. Only read for 'gaussian_heteroscedastic', for which
		*       w * dJ_eta/dzeta = -W and l_eta_zeta = 0
		* \param diag Diagonal of (Sigma^-1+W)^-1 (or its Z-projected / stochastic analogue) used by the log-determinant term.
		*       May be left empty if 'SecondFEBlockGradNeedsDiag' is false, in which case it is not read
		* \param impl (Sigma^-1+W)^-1 * dmll/dmode (or its Z-projected analogue) used by the implicit-derivative term.
		*       May be left empty if 'SecondFEBlockGradNeedsImpl' is false, in which case it is not read
		* \param index_map Maps a data index into 'diag' and 'impl'. Pass nullptr if those two are already on the data scale
		* \param include_coupled_zi_terms If false, only the direct score is used for a zero-inflated count regression. This is
		*       needed for the approximations on which the coupled terms would require a data-scale diagonal of (Sigma^-1+W)^-1
		*       that is only available as a stochastic estimate (the gradient is then approximate for ZI counts). Has no effect
		*       for hurdle regressions (which decouple exactly, l_eta_zeta = dJ_eta/dzeta = 0) or the heteroscedastic likelihoods
		* \param[out] fixed_effect_grad Gradient wrt fixed effects. Grown to 'dim_location_par_' if needed (the eta block, which
		*       the caller has already written, is preserved)
		*/
		void CalcSecondFEBlockFixedEffectGrad(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			const vec_t& information_data_scale,
			const vec_t& diag,
			const vec_t& impl,
			const data_size_t* index_map,
			bool include_coupled_zi_terms,
			vec_t& fixed_effect_grad) const;

		/*!
		* \brief Accumulate the two data-scale sums that appear in the gradient wrt an additional likelihood parameter in
		*       every Laplace approximation: the log-determinant term sum_i (dW/daux)_i * ((Sigma^-1+W)^-1)_ii and the
		*       implicit-derivative term sum_i (d^2l / (dlocpar daux))_i * ((Sigma^-1+W)^-1 dmll/dmode)_i.
		*       The mapping from the data scale to the mode scale (Z^T aggregation vs. identity) is resolved internally
		*       via 'use_random_effects_indices_of_data_', and the implicit-derivative term is only accumulated if
		*       'grad_information_wrt_mode_non_zero_'.
		* \param deriv_information_aux_par dW/daux on the data scale
		* \param second_deriv_loc_aux_par d^2l / (dlocpar daux) on the data scale
		* \param SigmaI_plus_W_inv_diag Diagonal of (Sigma^-1+W)^-1 on the mode scale
		* \param SigmaI_plus_W_inv_d_mll_d_mode (Sigma^-1+W)^-1 dmll/dmode on the mode scale
		* \param accumulate_log_det If false, 'd_detmll_d_aux_par' is left untouched (used when the log-determinant term is
		*       obtained from a stochastic trace estimator instead of this exact sum)
		* \param[out] d_detmll_d_aux_par Log-determinant term, accumulated into (not overwritten)
		* \param[out] implicit_derivative Implicit-derivative term, accumulated into (not overwritten)
		*/
		void AccumulateAuxParGradTerms(const vec_t& deriv_information_aux_par,
			const vec_t& second_deriv_loc_aux_par,
			const vec_t& SigmaI_plus_W_inv_diag,
			const vec_t& SigmaI_plus_W_inv_d_mll_d_mode,
			bool accumulate_log_det,
			double& d_detmll_d_aux_par,
			double& implicit_derivative) const;

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt the additional
		*       likelihood parameters for the approximations in which the diagonal of (Sigma^-1+W)^-1 is available exactly
		*       (i.e., all Cholesky-based variants). The stochastic (iterative) variants cannot use this since they obtain
		*       the log-determinant term from a trace estimator; they call 'AccumulateAuxParGradTerms' directly.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter
		* \param SigmaI_plus_W_inv_diag Diagonal of (Sigma^-1+W)^-1 on the mode scale
		* \param SigmaI_plus_W_inv_d_mll_d_mode (Sigma^-1+W)^-1 dmll/dmode on the mode scale
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters (needs to be preallocated of size num_aux_pars_estim_)
		*/
		void CalcAuxParGradLaplaceExactDiag(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			const vec_t& SigmaI_plus_W_inv_diag,
			const vec_t& SigmaI_plus_W_inv_d_mll_d_mode,
			double* aux_par_grad);// not const: 'CalcGradNegLogLikAuxPars' is non-const

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*       fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*       Calculations are done using a numerically stable variant based on factorizing ("inverting") B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt).
		*       In the notation of the paper: "Sigma = Z*Sigma*Z^T" and "Z = Id".
		*       This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		*       If use_random_effects_indices_of_data_, calculations are done on the random effects (b) scale and not the "data scale" (Zb)
		*       factorizing ("inverting") B = (Id + ZtWZsqrt * Sigma * ZtWZsqrt).
		*       This version (use_random_effects_indices_of_data_ == true) is used for the Laplace approximation when there is only one Gaussian process and
		*       there are multiple observations at the same location, i.e., the dimenion of the random effects b is much smaller than Zb
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param Sigma Covariance matrix of latent random effects ("Sigma = Z*Sigma*Z^T" if !use_random_effects_indices_of_data_)
		* \param re_comps_cluster_i Vector with different random effects components. We pass the component pointers to save memory in order to avoid passing a large collection of gardient covariance matrices in memory//TODO: better way than passing this? (relying on all gradients in a vector can lead to large memory consumption)
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		* \param call_for_std_dev_coef If true, the function is called for calculating standard deviations of linear regression coefficients
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<T_mat>& Sigma,
			const std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode,
			bool call_for_std_dev_coef,
			const std::vector<int>& estimate_cov_par_index);

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*       fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*       Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*       NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*       This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param SigmaI Inverse covariance matrix of latent random effect. Currently, this needs to be a diagonal matrix
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		* \param call_for_std_dev_coef If true, the function is called for calculating standard deviations of linear regression coefficients
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxGroupedRE(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const sp_mat_t& SigmaI,
			bool has_vecchia_gp,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const std::vector<sp_mat_t>& B_grad,
			const std::vector<sp_mat_t>& D_grad,
			const std::vector<data_size_t>& cum_num_rand_eff_cluster_i,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode,
			bool call_for_std_dev_coef,
			const std::vector<int>& estimate_cov_par_index);

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*       fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*       Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*       This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param sigma2 Variance of random effects
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		* \param call_for_std_dev_coef If true, the function is called for calculating standard deviations of linear regression coefficients
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const double sigma2,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode,
			bool call_for_std_dev_coef,
			const std::vector<int>& estimate_cov_par_index);

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*		fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Full-scale-Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param sigma_ip Covariance matrix of inducing point process
		* \param chol_fact_sigma_ip Cholesky factor of 'sigma_ip'
		* \param chol_fact_sigma_woodbury Cholesky factor of 'sigma_ip + sigma_cross_cov_T * sigma_residual^-1 * sigma_cross_cov'
		* \param cross_cov Cross-covariance matrix between inducing points and all data points
		* \param sigma_woodbury Matrix 'sigma_ip + sigma_cross_cov_T * sigma_residual^-1 * sigma_cross_cov'
		* \param re_comps_ip_cluster_i
		* \param re_comps_cross_cov_cluster_i
		* \param B Matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation Sigma^-1 = B^T D^-1 B
		* \param B_grad Derivatives of matrices B ( = derivative of matrix -A) for Vecchia approximation
		* \param D_grad Derivatives of matrices D for Vecchia approximation
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		* \param call_for_std_dev_coef If true, the function is called for calculating standard deviations of linear regression coefficients
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxFSVA(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const chol_den_mat_t& chol_fact_sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_woodbury,
			const den_mat_t& chol_ip_cross_cov,
			const den_mat_t& sigma_woodbury,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const den_mat_t& Bt_D_inv_B_cross_cov,
			const den_mat_t& D_inv_B_cross_cov,
			const den_mat_t& sigma_ip_inv_cross_cov_T_cluster_i,
			const std::vector<sp_mat_t>& B_grad,
			const std::vector<sp_mat_t>& D_grad,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode,
			bool call_for_std_dev_coef,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_preconditioner_cluster_i,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i,
			const den_mat_t& chol_ip_cross_cov_preconditioner,
			const chol_den_mat_t& chol_fact_sigma_ip_preconditioner,
			const std::vector<int>& estimate_cov_par_index,
			bool GPU_use);

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*       fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*       Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*       of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*       This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*       Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param B Matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation Sigma^-1 = B^T D^-1 B
		* \param B_grad Derivatives of matrices B ( = derivative of matrix -A) for Vecchia approximation
		* \param D_grad Derivatives of matrices D for Vecchia approximation
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		* \param num_comps_total Total number of random effect components ( = number of GPs)
		* \param call_for_std_dev_coef If true, the function is called for calculating standard deviations of linear regression coefficients
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			std::map<int, sp_mat_t>& B,
			std::map<int, sp_mat_t>& D_inv,
			std::map<int, std::vector<sp_mat_t>>& B_grad,
			std::map<int, std::vector<sp_mat_t>>& D_grad,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode,
			int num_comps_total,
			bool call_for_std_dev_coef,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			const den_mat_t& chol_ip_cross_cov,
			const chol_den_mat_t& chol_fact_sigma_ip,
			data_size_t cluster_i,
			REModelTemplate<T_mat, T_chol>* re_model,
			const std::vector<int>& estimate_cov_par_index,
			bool GPU_use);

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*       fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*       Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*       of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*       This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*       Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param sigma_ip Covariance matrix of inducing point process
		* \param chol_fact_sigma_ip Cholesky factor of 'sigma_ip'
		* \param cross_cov Cross-covariance matrix between inducing points and all data points
		* \param fitc_resid_diag Diagonal correction of predictive process
		* \param re_comps_ip_cluster_i
		* \param re_comps_cross_cov_cluster_i
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		* \param call_for_std_dev_coef If true, the function is called for calculating standard deviations of linear regression coefficients
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxFITC(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<den_mat_t> sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_ip,
			const den_mat_t* cross_cov,
			const vec_t& fitc_resid_diag,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode,
			bool call_for_std_dev_coef,
			const std::vector<int>& estimate_cov_par_index,
			bool GPU_use);

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*       This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param ZSigmaZt Covariance matrix of latent random effects
		* \param Cross_Cov Cross covariance matrix between predicted and observed random effects ("=Cov(y_p,y)")
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		*/
		void PredictLaplaceApproxStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<T_mat>& ZSigmaZt,
			const T_mat& Cross_Cov,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode);

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*       Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*       NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*       This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param SigmaI Inverse covariance matrix of latent random effect. Currently, this needs to be a diagonal matrix
		* \param Ztilde matrix which relates existing random effects to prediction samples
		* \param Sigma Covariance matrix of random effects
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		*/
		void PredictLaplaceApproxGroupedRE(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const sp_mat_t& SigmaI,
			bool has_vecchia_gp,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const sp_mat_t& Bpo,
			sp_mat_t& Bp,
			const vec_t& Dp,
			bool VecchiaCondObsOnly,
			const sp_mat_t& Ztilde,
			const sp_mat_t& Sigma,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode);

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*       Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*       This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param sigma2 Variance of random effects
		* \param random_effects_indices_of_pred Indices that indicate to which training data random effect every prediction point is related. -1 means to none in the training data
		* \param num_data_pred Number of prediction points
		* \param Cross_Cov Cross covariance matrix between predicted and observed random effects ("=Cov(y_p,y)", = Ztilde * Sigma)
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		*/
		void PredictLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const double sigma2,
			const data_size_t* const random_effects_indices_of_pred,
			const data_size_t num_data_pred,
			const T_mat& Cross_Cov,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode);

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Full-Scale-Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the full-scale Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param B Matrix B in Vecchia approximation for observed locations, Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation for observed locations
		* \param Bpo Lower left part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the observed locations
		* \param Bp Lower right part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the prediction locations
		* \param Dp Diagonal matrix with lower right part of matrix D in joint Vecchia approximation for observed and prediction locations
		* \param chol_fact_sigma_ip Cholesky factor of 'sigma_ip'
		* \param chol_fact_sigma_woodbury Cholesky factor of 'sigma_ip + sigma_mn sigma_resid^-1 sigma_mn'
		* \param cross_cov Cross - covariance matrix between inducing points and all data points
		* \param cross_cov_pred_ip Cross covariance matrix between prediction points and inducing points
		* \param sample_posterior If true, posterior samples are generated
		* \param num_post_samples Number of posterior samples
		* \param post_samples_id Sample from posterior without the mean, the mean is added at then end
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data and Bp is an identity matrix
		*/
		void PredictLaplaceApproxFSVA(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const sp_mat_t& Bpo,
			sp_mat_t& Bp,
			const vec_t& Dp,
			const std::shared_ptr<den_mat_t> sigma_ip,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_preconditioner_cluster_i,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i,
			const chol_den_mat_t& chol_fact_sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_ip_preconditioner,
			const den_mat_t& sigma_woodbury,
			const chol_den_mat_t& chol_fact_sigma_woodbury,
			const den_mat_t& chol_ip_cross_cov,
			const den_mat_t& chol_ip_cross_cov_preconditioner,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			const den_mat_t& cross_cov_pred_ip,
			const den_mat_t& Bt_D_inv_B_cross_cov,
			const den_mat_t& D_inv_B_cross_cov,
			bool sample_posterior,
			int num_post_samples,
			den_mat_t& post_samples_id,
			vec_t& pred_mean,
			den_mat_t& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode,
			bool CondObsOnly,
			bool GPU_use);

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*       Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*       of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*       This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*       Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param B Matrix B in Vecchia approximation for observed locations, Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation for observed locations
		* \param Bpo Lower left part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the observed locations
		* \param Bp Lower right part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the prediction locations
		* \param Dp Diagonal matrix with lower right part of matrix D in joint Vecchia approximation for observed and prediction locations
		* \param sample_posterior If true, posterior samples are generated
		* \param num_post_samples Number of posterior samples
		* \param post_samples_id Sample from posterior without the mean, the mean is added at then end
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data and Bp is an identity matrix
		* \param num_gp GP number in case there are multiple parameters with GPs (e.g., heteroscedastic regression)
		*/
		void PredictLaplaceApproxVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			std::map<int, sp_mat_t>& B,
			std::map<int, sp_mat_t>& D_inv,
			const sp_mat_t& Bpo,
			sp_mat_t& Bp,
			const vec_t& Dp,
			bool sample_posterior,
			int num_post_samples,
			den_mat_t& post_samples_id,
			vec_t& pred_mean,
			den_mat_t& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode,
			bool CondObsOnly,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			const den_mat_t& chol_ip_cross_cov,
			const chol_den_mat_t& chol_fact_sigma_ip,
			int num_gp,
			data_size_t cluster_i,
			REModelTemplate<T_mat, T_chol>* re_model);

		/*!
		* \brief Sampling from the Laplace-approximated posterior
		*/
		void Sample_Posterior_LaplaceApprox_Stable(const std::shared_ptr<T_mat>& Sigma);

		/*!
		* \brief Sampling from the Laplace-approximated posterior when there are multiple levels of grouped random effects
		*/
		void Sample_Posterior_LaplaceApprox_GroupedRE(const sp_mat_t& SigmaI,
			bool has_vecchia_gp,
			const sp_mat_t& B,
			const sp_mat_t& D_inv);

			/*!
		* \brief Sampling from the Laplace-approximated posterior when there are only single-level grouped random effects
		*/
		void Sample_Posterior_LaplaceApprox_OnlyOneGroupedRE();

		/*!
		* \brief Sampling from the Laplace-approximated posterior when using a Vecchia approximation
		*/
		void Sample_Posterior_LaplaceApprox_Vecchia(const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i);

		/*!
		* \brief Sampling from the Laplace-approximated posterior when using a Full-scale Vecchia approximation
		*/
		void Sample_Posterior_LaplaceApprox_FSVA(const den_mat_t* cross_cov,
			const den_mat_t& Bt_D_inv_B_cross_cov,
			const den_mat_t& sigma_woodbury,
			const chol_den_mat_t& chol_fact_sigma_woodbury,
			const chol_den_mat_t& chol_fact_sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_woodbury_2,
			const den_mat_t& chol_ip_cross_cov,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i);

		/*!
		* \brief Sampling from the Laplace-approximated posterior when using an FITC approximation
		*/
		void Sample_Posterior_LaplaceApprox_FITC(const den_mat_t* cross_cov,
			const vec_t& fitc_resid_diag);

		void SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean();

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*       This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param sigma_ip Covariance matrix of inducing point process
		* \param chol_fact_sigma_ip Cholesky factor of 'sigma_ip'
		* \param cross_cov Cross - covariance matrix between inducing points and all data points
		* \param fitc_resid_diag Diagonal correction of predictive process
		* \param cross_cov_pred_ip Cross covariance matrix between prediction points and inducing points
		* \param has_fitc_correction If true, there is an 'fitc_resid_pred_obs' otherwise not
		* \param fitc_resid_pred_obs FITC residual "correction" for entries for which the prediction and training coordinates are the same
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and SigmaI_mode_ are used (default=false)
		*/
		void PredictLaplaceApproxFITC(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<den_mat_t> sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_ip,
			const den_mat_t* cross_cov,
			const vec_t& fitc_resid_diag,
			const den_mat_t& cross_cov_pred_ip,
			bool has_fitc_correction,
			const sp_mat_t& fitc_resid_pred_obs,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode,
			bool GPU_use);

//Note: the following is currently not used
//      /*!
//      * \brief Calculate variance of Laplace-approximated posterior
//      * \param ZSigmaZt Covariance matrix of latent random effect
//      * \param[out] pred_var Variance of Laplace-approximated posterior
//      */
//      void CalcVarLaplaceApproxStable(const std::shared_ptr<T_mat>& ZSigmaZt,
//          vec_t& pred_var) {
//          if (na_or_inf_during_last_call_to_find_mode_) {
//              Log::REFatal(NA_OR_INF_ERROR_);
//          }
//          CHECK(mode_has_been_calculated_);
//          pred_var = vec_t(num_re_);
//          vec_t diag_Wsqrt(information_ll_.size());
//          diag_Wsqrt.array() = information_ll_.array().sqrt();
//          T_mat L_inv_W_sqrt_ZSigmaZt = diag_Wsqrt.asDiagonal() * (*ZSigmaZt);
//          TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, L_inv_W_sqrt_ZSigmaZt, L_inv_W_sqrt_ZSigmaZt, false);
//#pragma omp parallel for schedule(static)
//          for (int i = 0; i < num_re_; ++i) {
//              pred_var[i] = (*ZSigmaZt).coeff(i,i) - L_inv_W_sqrt_ZSigmaZt.col(i).squaredNorm();
//          }
//      }//end CalcVarLaplaceApproxStable

		/*!
		* \brief Calculate variance of Laplace-approximated posterior
		* \param Sigma Covariance matrix of latent random effect
		* \param[out] pred_var Variance of Laplace-approximated posterior
		*/
		void CalcVarLaplaceApproxOnlyOneGPCalculationsOnREScale(const std::shared_ptr<T_mat>& Sigma,
			vec_t& pred_var);

		/*!
		* \brief Calculate variance of Laplace-approximated posterior
		* \param[out] pred_var Variance of Laplace-approximated posterior
		*/
		void CalcVarLaplaceApproxGroupedRE(vec_t& pred_var);

		/*!
		* \brief Calculate variance of Laplace-approximated posterior
		* \param[out] pred_var Variance of Laplace-approximated posterior
		*/
		void CalcVarLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(vec_t& pred_var);

		/*!
		* \brief Calculate variance of Laplace-approximated posterior
		* \param[out] pred_var Variance of Laplace-approximated posterior
		*/
		void CalcVarLaplaceApproxVecchia(vec_t& pred_var,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i);

		/*!
		* \brief Make predictions for the response variable (label) based on predictions for the mean and variance of the latent random effects (the latent predictive distribution is marginalized out)
		* \param pred_mean[in & out] Predictive mean of latent random effects for mean. The Predictive mean for the response variables is written on this
		* \param pred_var[in & out] Predictive variances of latent random effects for mean. The predicted variance for the response variables is written on this
		* \param pred_var_mean Predictive mean of latent random effects for variance parameter in heteroscedastic models
		* \param pred_var_var Predictive variances of latent random effects for variance parameter in heteroscedastic models
		* \param predict_var If true, predictive response variances are also calculated
		*/
		void PredictResponse(vec_t& pred_mean,
			vec_t& pred_var,
			const vec_t& pred_var_mean,
			const vec_t& pred_var_var,
			bool predict_var);

		/*!
		* \brief Adaptive GH quadrature to calculate predictive mean of response variable
		* \param latent_mean Predictive mean of latent random effects
		* \param latent_var Predictive variances of latent random effects
		* \param second_moment If true, the second moment E( E(yp|bp)^2 | y) is calculated
		*/
		double RespMeanAdaptiveGHQuadrature(const double latent_mean,
			const double latent_var,
			bool second_moment);

		/*!
		* \brief Adaptive GH quadrature to calculate E( Var(yp | bp) | y), where Var(yp | bp) is variance of the likelihood given the location parameter bp
		* \param latent_mean Predictive mean of latent random effects
		* \param latent_var Predictive variances of latent random effects
		*/
		double ExpectedValueCondRespVarAdaptiveGHQuadrature(const double latent_mean,
			const double latent_var);

		/*!
		* \brief Calculate test negative log-likelihood using adaptive GH quadrature
		* \param y_test Test response variable
		* \param pred_mean Predictive mean of latent random effects
		* \param pred_var Predictive variances of latent random effects
		* \param num_data Number of data points
		*/
		inline double TestNegLogLikelihoodAdaptiveGHQuadrature(const label_t* y_test,
			const double* pred_mean,
			const double* pred_var,
			const data_size_t num_data) const;

		static string_t ParseLikelihoodAlias(const string_t& likelihood) {
			if (likelihood == string_t("binary_probit")) {
				return "bernoulli_probit";
			}
			else if (likelihood == string_t("binary") || likelihood == string_t("binary_logit")) {
				return "bernoulli_logit";
			}
			else if (likelihood == string_t("binomial")) {
				return "binomial_logit";
			}
			else if (likelihood == string_t("quasi_binary_probit")) {
				return "quasi_bernoulli_probit";
			}
			else if (likelihood == string_t("quasi_binary") || likelihood == string_t("quasi_binary_logit")) {
				return "quasi_bernoulli_logit";
			}
			else if (likelihood == string_t("regression")) {
				return "gaussian";
			}
			else if (likelihood == string_t("nbinom2") || likelihood == string_t("negative_binomial_2") || likelihood == string_t("negative_binomial2")) {
				return "negative_binomial";
			}
			else if (likelihood == string_t("nbinom1") || likelihood == string_t("negative_binomial1")) {
				return "negative_binomial_1";
			}
			else if (likelihood == string_t("student_t") || likelihood == string_t("student-t") ||
				likelihood == string_t("t_distribution") || likelihood == string_t("t-distribution")) {
				return "t";
			}
			else if (likelihood == string_t("log-normal") || likelihood == string_t("log_normal")) {
				return "lognormal";
			}
			else if (likelihood == string_t("beta-binomial") || likelihood == string_t("betabinomial")) {
				return "beta_binomial";
			}
			else if (likelihood == string_t("zero-inflated-gamma") || likelihood == string_t("zero_inflated_gamma")) {
				return "hurdle_gamma";
			}
			else if (likelihood == string_t("zero_inflated_lognormal") || likelihood == string_t("zero-inflated-lognormal")) {
				return "hurdle_lognormal";
			}
			else if (likelihood == string_t("hurdle_poisson")) {
				return "zero_inflated_poisson";
			}
			else if (likelihood == string_t("zero_inflated_nbinom2") || likelihood == string_t("zero_inflated_negative_binomial_2") || likelihood == string_t("zero_inflated_nbinom") ||
				likelihood == string_t("hurdle_negative_binomial") || likelihood == string_t("hurdle_nbinom2") || likelihood == string_t("hurdle_negative_binomial_2")) {
				return "zero_inflated_negative_binomial";
			}
			else if (likelihood == string_t("zero_inflated_nbinom1") || likelihood == string_t("hurdle_negative_binomial_1") || likelihood == string_t("hurdle_nbinom1")) {
				return "zero_inflated_negative_binomial_1";
			}
			else if (likelihood == string_t("zero_inflated_gpd")) {
				return "hurdle_gpd";
			}
			else if (likelihood == string_t("zero_inflated_egpd_power")) {
				return "hurdle_egpd_power";
			}
			else if (likelihood == string_t("zero_inflated_egpd_power_mixture")) {
				return "hurdle_egpd_power_mixture";
			}
			else if (likelihood == string_t("zero_inflated_egpd_beta")) {
				return "hurdle_egpd_beta";
			}
			else if (likelihood == string_t("zero_inflated_egpd_power_beta")) {
				return "hurdle_egpd_power_beta";
			}
			// Regression (fixed-effects) structural-zero-model aliases, analogous to the constant ones above (with "regression"
			// inserted after the family prefix). Positive-continuous bases have canonical name "hurdle_regression_<base>"; count
			// bases have canonical name "zero_inflated_regression_<base>"; the other prefix is accepted as an alias.
			else if (likelihood == string_t("zero-inflated-regression-gamma") || likelihood == string_t("zero_inflated_regression_gamma")) {
				return "hurdle_regression_gamma";
			}
			else if (likelihood == string_t("zero_inflated_regression_lognormal") || likelihood == string_t("zero-inflated-regression-lognormal")) {
				return "hurdle_regression_lognormal";
			}
			else if (likelihood == string_t("hurdle_regression_poisson")) {
				return "zero_inflated_regression_poisson";
			}
			else if (likelihood == string_t("zero_inflated_regression_nbinom2") || likelihood == string_t("zero_inflated_regression_negative_binomial_2") || likelihood == string_t("zero_inflated_regression_nbinom") ||
				likelihood == string_t("hurdle_regression_negative_binomial") || likelihood == string_t("hurdle_regression_nbinom2") || likelihood == string_t("hurdle_regression_negative_binomial_2")) {
				return "zero_inflated_regression_negative_binomial";
			}
			else if (likelihood == string_t("zero_inflated_regression_nbinom1") || likelihood == string_t("hurdle_regression_negative_binomial_1") || likelihood == string_t("hurdle_regression_nbinom1")) {
				return "zero_inflated_regression_negative_binomial_1";
			}
			else if (likelihood == string_t("zero_inflated_regression_gpd")) {
				return "hurdle_regression_gpd";
			}
			else if (likelihood == string_t("zero_inflated_regression_egpd_power")) {
				return "hurdle_regression_egpd_power";
			}
			else if (likelihood == string_t("zero_inflated_regression_egpd_power_mixture")) {
				return "hurdle_regression_egpd_power_mixture";
			}
			else if (likelihood == string_t("zero_inflated_regression_egpd_beta")) {
				return "hurdle_regression_egpd_beta";
			}
			else if (likelihood == string_t("zero_inflated_regression_egpd_power_beta")) {
				return "hurdle_regression_egpd_power_beta";
			}
			else if (likelihood == string_t("zero-censored-power-normal")) {
				return "zero_censored_power_transformed_normal";
			}
			else if (likelihood == string_t("zero-censored-power-normal-heteroscedastic") ||
				likelihood == string_t("zero_censored_power_transformed_normal_het")) {
				return "zero_censored_power_transformed_normal_heteroscedastic";
			}
			else if (likelihood == string_t("quantile") || likelihood == string_t("quantile_regression")) {
				return "asymmetric_laplace";
			}
			return likelihood;
		}

		string_t ParseLikelihoodAliasVarianceCorrection(const string_t& likelihood) {
			if (likelihood.size() > 23) {
				if (likelihood.substr(likelihood.size() - 23) == string_t("_var_cor_pred_freq_asym")) {
					use_variance_correction_for_prediction_ = true;
					var_cor_pred_version_ = "freq_asymptotic";
					return likelihood.substr(0, likelihood.size() - 23);
				}
			}
			if (likelihood.size() > 16) {
				if (likelihood.substr(likelihood.size() - 16) == string_t("_var_cor_pred_lr")) {
					use_variance_correction_for_prediction_ = true;
					var_cor_pred_version_ = "learning_rate";
					return likelihood.substr(0, likelihood.size() - 16);
				}
			}
			return likelihood;
		}

		string_t ParseLikelihoodAliasKinkClipping(const string_t& likelihood) {
			if (likelihood.size() > 14) {
				if (likelihood.substr(likelihood.size() - 14) == string_t("_kink_clipping")) {
					kink_cliping_ = true;
					return likelihood.substr(0, likelihood.size() - 14);
				}
			}
			return likelihood;
		}

		/*!
		* \brief Parse the suffix that enables the SSN-ALM (semismooth Newton with an augmented Lagrangian method) refinement
		*			of the posterior mode. '_ssn_alm' runs the refinement only when the exact non-smooth KKT conditions are
		*			violated after the (Fisher) quasi-Newton mode finding, '_ssn_alm_always' runs it unconditionally (this is
		*			mainly meant for testing and for measuring the cost of the refinement)
		*/
		string_t ParseLikelihoodAliasModeRefinement(const string_t& likelihood) {
			if (likelihood.size() > 15) {
				if (likelihood.substr(likelihood.size() - 15) == string_t("_ssn_alm_always")) {
					mode_refinement_ = ModeRefinement::kSSNALMAlways;
					return likelihood.substr(0, likelihood.size() - 15);
				}
			}
			if (likelihood.size() > 8) {
				if (likelihood.substr(likelihood.size() - 8) == string_t("_ssn_alm")) {
					mode_refinement_ = ModeRefinement::kSSNALMIfNeeded;
					return likelihood.substr(0, likelihood.size() - 8);
				}
			}
			return likelihood;
		}

		string_t ParseLikelihoodAliasModeFindingMethod(const string_t& likelihood) {
			if (likelihood.size() > 29) {
				if (likelihood.substr(likelihood.size() - 29) == string_t("_fisher_mode_finding_continue")) {
					use_fisher_for_mode_finding_ = true;
					continue_mode_finding_after_fisher_ = true;
					user_defined_mode_finding_approach_ = true;
					return likelihood.substr(0, likelihood.size() - 29);
				}
			}
			if (likelihood.size() > 24) {
				if (likelihood.substr(likelihood.size() - 24) == string_t("_not_fisher_mode_finding")) {
					use_fisher_for_mode_finding_ = false;
					user_defined_mode_finding_approach_ = true;
					return likelihood.substr(0, likelihood.size() - 24);
				}
			}
			if (likelihood.size() > 20) {
				if (likelihood.substr(likelihood.size() - 20) == string_t("_fisher_mode_finding")) {
					use_fisher_for_mode_finding_ = true;
					user_defined_mode_finding_approach_ = true;
					return likelihood.substr(0, likelihood.size() - 20);
				}
			}
			return likelihood;
		}

		string_t ParseLikelihoodAliasApproximationType(const string_t& likelihood) {
			if (likelihood.size() > 24) {
				if (likelihood.substr(likelihood.size() - 24) == string_t("_fisher_laplace_combined")) {
					approximation_type_ = "laplace";
					user_defined_approximation_type_ = "laplace";
					use_fisher_for_mode_finding_ = true;
					return likelihood.substr(0, likelihood.size() - 24);
				}
			}
			if (likelihood.size() > 15) {
				if (likelihood.substr(likelihood.size() - 15) == string_t("_fisher-laplace") ||
					likelihood.substr(likelihood.size() - 15) == string_t("_fisher_laplace")) {
					approximation_type_ = "fisher_laplace";
					user_defined_approximation_type_ = "fisher_laplace";
					return likelihood.substr(0, likelihood.size() - 15);
				}
			}
			if (likelihood.size() > 19) {
				if (likelihood.substr(likelihood.size() - 19) == string_t("_triangular_kernel_curvature")) {
					approximation_type_ = "triangular_kernel_curvature";
					user_defined_approximation_type_ = "triangular_kernel_curvature";
					return likelihood.substr(0, likelihood.size() - 19);
				}
			}
			if (likelihood.size() > 4) {
				if (likelihood.substr(likelihood.size() - 4) == string_t("_tkc") || likelihood.substr(likelihood.size() - 4) == string_t("_TKC")) {
					approximation_type_ = "triangular_kernel_curvature";
					user_defined_approximation_type_ = "triangular_kernel_curvature";
					return likelihood.substr(0, likelihood.size() - 4);
				}
			}
			if (likelihood.size() > 8) {
				if (likelihood.substr(likelihood.size() - 8) == string_t("_laplace") && likelihood != string_t("asymmetric_laplace")) {
					approximation_type_ = "laplace";
					user_defined_approximation_type_ = "laplace";
					return likelihood.substr(0, likelihood.size() - 8);
				}
			}
			return likelihood;
		}

		string_t ParseLikelihoodAliasEstimateAdditionalPars(const string_t& likelihood) {
			if (likelihood.size() > 16) {
				if (likelihood.substr(likelihood.size() - 16) == string_t("_use_likelihoods")) {
					use_likelihoods_file_for_gaussian_ = true;
					return likelihood.substr(0, likelihood.size() - 16);
				}
			}
			if (likelihood.size() > 7) {
				if (likelihood.substr(likelihood.size() - 7) == string_t("_fix_df")) {
					estimate_df_t_ = false;
					return likelihood.substr(0, likelihood.size() - 7);
				}
			}
			return likelihood;
		}

		/*!
		* \brief Transform from the latent to the response variable scale (often this is the inverse link function) conditional on the random and fixed effects
		*			This is only used by the 'ConvertOutput()' function in regression_objective.hpp
		*/
		double TransformToResponseScale(const double value) const;

	private:

		/*!
		* \brief Apply a per-sample kernel to all data points and write the results, multiplied by the sample weights, to 'out'
		*			This factors out the OpenMP loop that is used by the per-sample log-likelihood, gradient, and information calculations
		* \param[out] out Vector of length at least num_data_
		* \param kernel Callable that takes a sample index and returns the unweighted value for this sample
		*/
		template <class Kernel>
		void ForEachSampleWeighted(vec_t& out, Kernel kernel) const {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.0;
				out[i] = w * kernel(i);
			}
		}//end ForEachSampleWeighted

		/*!
		* \brief Sum a per-sample kernel over all data points, weighted by the sample weights
		* \param kernel Callable that takes a sample index and returns the unweighted value for this sample
		* \return sum_i w_i * kernel(i)
		*/
		template <class Kernel>
		double SumOverSamplesWeighted(Kernel kernel) const {
			double sum_kernel = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:sum_kernel)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.0;
				sum_kernel += w * kernel(i);
			}
			return sum_kernel;
		}//end SumOverSamplesWeighted

		/*!
		* \brief Aggregate a per-sample quantity to the scale of the modes (= Z^T v), separately for every set of random effects
		* \param v_data_scale Vector of length num_data_ * num_sets_re_ on the scale of the data
		* \param[out] v_mode_scale Vector of length dim_mode_per_set_re_ * num_sets_re_ on the scale of the modes
		*/
		void ReduceToModeScale(const vec_t& v_data_scale, vec_t& v_mode_scale) const {
			for (int igp = 0; igp < num_sets_re_; ++igp) {
				CalcZtVGivenIndices(num_data_, dim_mode_per_set_re_, random_effects_indices_of_data_,
					v_data_scale.data() + num_data_ * igp, v_mode_scale.data() + dim_mode_per_set_re_ * igp, true);
			}
		}//end ReduceToModeScale

		/*!
		* \brief Sum of the sample weights (= num_data_ if no weights are used)
		*			The log-likelihood contribution of every sample is multiplied by its weight, so the parts of the
		*			logarithmic normalizing constant that are identical for all samples must be scaled by this sum and
		*			not by the number of samples
		*/
		double SumOfWeights() const {
			if (!has_weights_) {
				return (double)num_data_;
			}
			return SumOverSamplesWeighted([](data_size_t) { return 1.; });
		}//end SumOfWeights

		/*! \brief Report that the calling function does not support the current likelihood */
		void NotSupportedForLikelihood(const char* caller) const {
			Log::REFatal("%s: Likelihood of type '%s' is not supported ", caller, likelihood_type_.c_str());
		}

		/*! \brief Report that the calling function does not support the current likelihood in combination with 'approximation_type' */
		void NotSupportedForLikelihoodAndApproximation(const char* caller, const string_t& approximation_type) const {
			Log::REFatal("%s: Likelihood of type '%s' is not supported for approximation_type = '%s' ", caller, likelihood_type_.c_str(), approximation_type.c_str());
		}

		/*! \brief True for the likelihoods for which the single-sample functions (used only by 'TestNegLogLikelihoodAdaptiveGHQuadrature()') are not implemented */
		bool NotImplementedForOneSample() const {
			return likelihood_type_ == "binomial_probit" || likelihood_type_ == "binomial_logit" ||
				likelihood_type_ == "beta_binomial" || likelihood_type_ == "quasi_bernoulli_probit" || likelihood_type_ == "quasi_bernoulli_logit";
		}

		/*! \brief Report that the calling single-sample function is not implemented for the current likelihood */
		void FatalOneSampleNotImplemented(const char* caller) const {
			Log::REFatal("%s: not implemented for likelihood = '%s'. If this error happened during the GPBoost algorithm, use another 'metric' instead of the (default) 'test_neg_log_likelihood' metric ", caller, likelihood_type_.c_str());
		}



		/*!
		* \brief Calculate the part of the logarithmic normalizing constant of the likelihood that does not depend neither on aux_pars_ nor on location_par
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		*/
		void CalculateAuxQuantLogNormalizingConstant(const double* y_data,
			const int* y_data_int) {
			if (!aux_normalizing_constant_has_been_calculated_) {
				if (likelihood_type_ == "gamma") {
					double log_aux_normalizing_constant = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						log_aux_normalizing_constant += w * AuxQuantLogNormalizingConstantGammaOneSample(y_data[i]);
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (likelihood_type_ == "negative_binomial") {
					double log_aux_normalizing_constant = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						log_aux_normalizing_constant += w * AuxQuantLogNormalizingConstantNegBinOneSample(y_data_int[i]);
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (likelihood_type_ == "negative_binomial_1") {
					double log_aux_normalizing_constant = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						log_aux_normalizing_constant += w * AuxQuantLogNormalizingConstantNegBin1OneSample(y_data_int[i]);
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (IsZeroInflatedCount()) {
					double log_aux_normalizing_constant = 0.;// -sum_i w_i * log(y_i!) (count-component factorial term)
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						log_aux_normalizing_constant += w * (-std::lgamma((double)y_data_int[i] + 1.));
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (likelihood_type_ == "binomial_probit" || likelihood_type_ == "binomial_logit" || likelihood_type_ == "beta_binomial") {
					CHECK(has_weights_);
					double log_aux_normalizing_constant = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = weights_[i];// w = n = y
						const double k = w * y_data[i];// y = k / n
						log_aux_normalizing_constant += std::lgamma(w + 1.) - std::lgamma(k + 1.) - std::lgamma(w - k + 1.);
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (likelihood_type_ == "lognormal") {
					double log_aux_normalizing_constant = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						log_aux_normalizing_constant -= w * std::log(y_data[i]);
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (likelihood_type_ == "hurdle_lognormal") {
					double log_aux_normalizing_constant = 0.;// -sum_{y>0} w * log(y)
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						if (y_data[i] > 0.) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							log_aux_normalizing_constant -= w * std::log(y_data[i]);
						}
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (likelihood_type_ == "hurdle_gamma") {
					double log_aux_normalizing_constant = 0.0;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						if (y_data[i] > 0.0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							log_aux_normalizing_constant += w * std::log(y_data[i]);
						}
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (IsZeroCensPowNorm()) {
					double s_logy_pos = 0.0;
#pragma omp parallel for schedule(static) reduction(+:s_logy_pos)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double yi = y_data[i];
						if (yi > 0.0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							s_logy_pos += w * std::log(yi);
						}
					}
					aux_log_normalizing_constant_ = s_logy_pos;
				}
				else if (likelihood_type_ == "zoctn") {
					double log_aux_normalizing_constant = 0.0;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double yi = y_data[i];
						if (yi > 0.0 && yi < 1.0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							log_aux_normalizing_constant += w * (-std::log(yi) - std::log1p(-yi));
						}
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (IsHurdleRegression()) {
					const string_t base = HurdleRegressionBaseType();
					double c = 0.;
					if (base == "hurdle_gamma" || base == "hurdle_lognormal") {
						const double sgn = (base == "hurdle_gamma") ? 1. : -1.;// +log(y) for gamma, -log(y) for lognormal (over positive observations)
#pragma omp parallel for schedule(static) reduction(+:c)
						for (data_size_t i = 0; i < num_data_; ++i) {
							if (y_data[i] > 0.) { const double w = has_weights_ ? weights_[i] : 1.0; c += sgn * w * std::log(y_data[i]); }
						}
					}
					aux_log_normalizing_constant_ = c;// 0 for EGPD bases (EvaluateEGPD returns the complete density)
				}
				else if (!IsGaussianLikelihood() && !IsGaussianHeteroscedastic() && !IsEGPDLikelihood() && !IsHurdleEGPD() &&
					likelihood_type_ != "bernoulli_probit" && likelihood_type_ != "bernoulli_logit" &&
					likelihood_type_ != "poisson" && likelihood_type_ != "tweedie" && likelihood_type_ != "tweedie_fixed_p" && likelihood_type_ != "t" && likelihood_type_ != "beta" &&
					likelihood_type_ != "zero_one_censored_transformed_beta" && likelihood_type_ != "zero_one_censored_shifted_gamma" &&
					likelihood_type_ != "asymmetric_laplace" && likelihood_type_ != "quasi_bernoulli_probit" && likelihood_type_ != "quasi_bernoulli_logit") {
					NotSupportedForLikelihood(__func__);
				}
				aux_normalizing_constant_has_been_calculated_ = true;
			}
		}//end CalculateAuxQuantLogNormalizingConstant

		inline double AuxQuantLogNormalizingConstantGammaOneSample(const double y) const {
			return(std::log(y));
		}

		inline double AuxQuantLogNormalizingConstantNegBinOneSample(const int y) const {
			return(-std::lgamma(y + 1));
		}

		inline double AuxQuantLogNormalizingConstantNegBin1OneSample(const int y) const {
			return(-std::lgamma(y + 1));
		}

		/*!
		* \brief Calculate the logarithmic normalizing constant of the likelihood (not depending on location_par)
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		*/
		void CalculateLogNormalizingConstant(const double* y_data,
			const int* y_data_int) {
			if (!normalizing_constant_has_been_calculated_) {
				CalculateAuxQuantLogNormalizingConstant(y_data, y_data_int);
				if (IsEGPDLikelihood()) {
					log_normalizing_constant_ = 0.;
				}
				else if (IsHurdleRegression()) {
					// The structural-zero mixture terms log(pi_i)/log(q_i) depend on zeta and are handled per-sample; only the aux-dependent
					// base normalizer over positive observations remains (0 for EGPD bases, whose density is complete in EvaluateEGPD).
					const string_t base = HurdleRegressionBaseType();
					if (base == "hurdle_gamma" || base == "hurdle_lognormal") {
						double w_pos = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:w_pos)
						for (data_size_t i = 0; i < num_data_; ++i) { if (y_data[i] > 0.) w_pos += has_weights_ ? weights_[i] : 1.0; }
						if (base == "hurdle_gamma") log_normalizing_constant_ = w_pos * (aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0])) + (aux_pars_[0] - 1.) * aux_log_normalizing_constant_;
						else log_normalizing_constant_ = w_pos * (-M_LOGSQRT2PI - 0.5 * std::log(aux_pars_[0])) + aux_log_normalizing_constant_;
					}
					else log_normalizing_constant_ = 0.;// EGPD bases
				}
				else if (IsHurdleEGPD()) {
					// The positive EGPD density is fully evaluated per-sample; only the mixture constants remain: w_zero*log(p0) + w_pos*log(q).
					const double p0 = aux_pars_original_[num_aux_pars_ - 1];
					const double log_q = std::log1p(-p0), log_p0 = std::log(p0);
					double w_pos = 0., w_zero = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:w_pos, w_zero)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] > 0.) w_pos += w; else w_zero += w;
					}
					log_normalizing_constant_ = w_zero * log_p0 + w_pos * log_q;
				}
				else if (likelihood_type_ == "poisson") {
					const double aux_const = SumOverSamplesWeighted([&](data_size_t i) { return LogNormalizingConstantPoissonOneSample(y_data_int[i]); });
					log_normalizing_constant_ = aux_const;
				}
				else if (likelihood_type_ == "gamma") {
					log_normalizing_constant_ = LogNormalizingConstantGamma();
				}
				else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
					const double phi = aux_pars_[0];
					const double p = GetTweediePower();
					const double log_phi = std::log(phi);
					const auto transform = likelihood_type_ == "tweedie" ? TransformTweediePowerFromQ(aux_pars_[1], TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_) : TweediePowerTransform{ p, 0., 0. };
					const bool calc_power_deriv = likelihood_type_ == "tweedie";
					double sum_a = 0., sum_rho = 0., sum_theta = 0.;
					// Non-convergence is exceptional; record it and raise the error after the parallel region (Log::REFatal must not throw out of an OpenMP loop).
					bool convergence_failure = false;
					data_size_t fail_index = 0;
					// Each thread uses its own persistent special-function cache (thread_local => reused across calls, e.g. when only phi changes).
#pragma omp parallel for schedule(static) reduction(+:sum_a,sum_rho,sum_theta) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						thread_local TweedieSpecialFunctionCache cache;
						const double w = has_weights_ ? weights_[i] : 1.;
						const auto res = EvaluateTweedieLogNormalizer(y_data[i], log_phi, p, transform.dp_dtheta, transform.d2p_dtheta2, TweedieDerivativeOrder::kFirst, calc_power_deriv, 1000000, &cache);
						if (!res.converged || !std::isfinite(res.log_a) || !std::isfinite(res.d_rho) || !std::isfinite(res.d_theta)) {
#pragma omp critical
							{
								convergence_failure = true;
								fail_index = i;
							}
						}
						else {
							sum_a += w * res.log_a;
							sum_rho += w * res.d_rho;
							sum_theta += w * res.d_theta;
						}
					}
					if (convergence_failure) {
						Log::REFatal("Tweedie density series did not converge for y=%g, phi=%g, p = %g.", y_data[fail_index], phi, p);
					}
					tweedie_sum_d_log_a_rho_ = sum_rho;
					tweedie_sum_d_log_a_theta_ = sum_theta;
					// Snapshot the parameters the cached normalizer aggregates correspond to (verified in CalcGradNegLogLikAuxPars).
					tweedie_cached_phi_ = phi;
					tweedie_cached_p_ = p;
					log_normalizing_constant_ = sum_a;
				}
				else if (likelihood_type_ == "negative_binomial") {
					log_normalizing_constant_ = LogNormalizingConstantNegBin(y_data_int);
				}
				else if (likelihood_type_ == "negative_binomial_1") {
					log_normalizing_constant_ = LogNormalizingConstantNegBin1(y_data_int);
				}
				else if (likelihood_type_ == "zero_inflated_poisson") {
					// Total constant = sum_{y>0} w * log(q) - sum_i w * log(y!). The log(q) factor for the positive-count branch;
					// the y=0 mixture log D = log(pi + q*f0) is location-dependent and handled in the per-sample log-likelihood.
					const double log_q = std::log1p(-aux_pars_original_[0]);
					double w_pos = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:w_pos)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data_int[i] > 0) w_pos += w;
					}
					log_normalizing_constant_ = w_pos * log_q + aux_log_normalizing_constant_;
				}
				else if (IsZeroInflatedCountRegression()) {
					// Base-count normalizer over positive observations only (the per-sample log(pi)/log(q) mixture terms depend on zeta and are in the per-sample log-likelihood).
					const string_t base = ZICountRegressionBaseType();
					if (base == "zero_inflated_negative_binomial") {
						const double kappa = aux_pars_[0];
						const double kappa_term = kappa * std::log(kappa) - std::lgamma(kappa);
						double c = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:c)
						for (data_size_t i = 0; i < num_data_; ++i) {
							if (y_data_int[i] > 0) { const double w = has_weights_ ? weights_[i] : 1.0; c += w * (std::lgamma((double)y_data_int[i] + kappa) - std::lgamma((double)y_data_int[i] + 1.) + kappa_term); }
						}
						log_normalizing_constant_ = c;
					}
					else if (base == "zero_inflated_negative_binomial_1") {
						log_normalizing_constant_ = LogNormalizingConstantNegBin1(y_data_int);// y=0 contributes 0
					}
					else {// zero_inflated_poisson
						log_normalizing_constant_ = aux_log_normalizing_constant_;// -sum_i w * log(y!)
					}
				}
				else if (likelihood_type_ == "zero_inflated_negative_binomial") {
					// Positive-count NB2 normalizer over y>0 only (the y=0 mixture log D is location/shape-dependent, handled per-sample).
					// Note: the base NB2 one-sample normalizer at y=0 equals kappa*log(kappa) != 0, so we must restrict to y>0.
					const double kappa = aux_pars_[0];
					const double log_q = std::log1p(-aux_pars_original_[1]);
					const double kappa_term = kappa * std::log(kappa) - std::lgamma(kappa);
					double c = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:c)
					for (data_size_t i = 0; i < num_data_; ++i) {
						if (y_data_int[i] > 0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							c += w * (std::lgamma((double)y_data_int[i] + kappa) - std::lgamma((double)y_data_int[i] + 1.) + kappa_term + log_q);
						}
					}
					log_normalizing_constant_ = c;
				}
				else if (likelihood_type_ == "zero_inflated_negative_binomial_1") {
					// NB1 one-sample normalizer at y=0 equals 0, so the base sum over all data equals the sum over y>0. Add w_pos*log(q).
					const double log_q = std::log1p(-aux_pars_original_[1]);
					double w_pos = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:w_pos)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data_int[i] > 0) w_pos += w;
					}
					log_normalizing_constant_ = w_pos * log_q + LogNormalizingConstantNegBin1(y_data_int);
				}
				else if (likelihood_type_ == "beta") {
					log_normalizing_constant_ = SumOfWeights() * std::lgamma(aux_pars_[0]);
				}
				else if (likelihood_type_ == "t") {
					log_normalizing_constant_ = SumOfWeights() * (-std::log(aux_pars_[0]) +
						std::lgamma((aux_pars_[1] + 1.) / 2.) - 0.5 * std::log(aux_pars_[1]) -
						std::lgamma(aux_pars_[1] / 2.) - 0.5 * std::log(M_PI));
				}
				else if (IsGaussianLikelihood()) {
					log_normalizing_constant_ = -SumOfWeights() * (M_LOGSQRT2PI + 0.5 * std::log(aux_pars_[0]));
				}
				else if (IsGaussianHeteroscedastic()) {
					log_normalizing_constant_ = -SumOfWeights() * M_LOGSQRT2PI;
				}
				else if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit" || 
					likelihood_type_ == "quasi_bernoulli_probit" || likelihood_type_ == "quasi_bernoulli_logit") {
					log_normalizing_constant_ = 0.;
				}
				else if (likelihood_type_ == "binomial_probit" || likelihood_type_ == "binomial_logit" || likelihood_type_ == "beta_binomial") {
					log_normalizing_constant_ = aux_log_normalizing_constant_;
				}
				else if (likelihood_type_ == "lognormal") {
					log_normalizing_constant_ = aux_log_normalizing_constant_ - SumOfWeights() * (M_LOGSQRT2PI + 0.5 * std::log(aux_pars_[0]));
				}
				else if (likelihood_type_ == "hurdle_lognormal") {
					const double p0 = aux_pars_original_[1];
					const double log_q = std::log1p(-p0), log_p0 = std::log(p0);
					double w_pos = 0., w_zero = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:w_pos, w_zero)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] > 0.) w_pos += w; else w_zero += w;
					}
					log_normalizing_constant_ = w_zero * log_p0 + w_pos * (log_q - M_LOGSQRT2PI - 0.5 * std::log(aux_pars_[0])) + aux_log_normalizing_constant_;
				}
				else if (likelihood_type_ == "hurdle_gamma") {
					const double q = 1 - aux_pars_original_[1];// = 1 - p0
					const double log_q = std::log(q);
					double w_pos = 0.0, w_zero = 0.0;
#pragma omp parallel for schedule(static) reduction(+:w_pos,w_zero)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] > 0.0) w_pos += w;
						else                 w_zero += w;
					}
					log_normalizing_constant_ = w_zero * std::log(1.0 - q) + w_pos * (log_q + aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0])) +
						(aux_pars_[0] - 1.0) * aux_log_normalizing_constant_;
				}
				else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
					double w_pos = 0.0;
#pragma omp parallel for schedule(static) reduction(+:w_pos)
					for (data_size_t i = 0; i < num_data_; ++i) {
						if (y_data[i] > 0.0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							w_pos += w;
						}
					}
					log_normalizing_constant_ = w_pos * (-std::log(aux_pars_[1]) - std::log(aux_pars_[0]) - M_LOGSQRT2PI);// Per positive y: -log(lambda) -log(sigma) - 0.5*log(2*pi)
					log_normalizing_constant_ += ((1.0 / aux_pars_[1]) - 1.0) * aux_log_normalizing_constant_;// Add the Jacobian term: (1/lambda - 1) * sum_{y>0} w * log(y)
				}
				else if (IsZeroCensPowNormHetero()) {
					// As above, but -log(sigma_i) = -location_par2_i depends on the location parameter and is therefore part
					// of the per-sample log-likelihood instead of this constant
					double w_pos = 0.0;
#pragma omp parallel for schedule(static) reduction(+:w_pos)
					for (data_size_t i = 0; i < num_data_; ++i) {
						if (y_data[i] > 0.0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							w_pos += w;
						}
					}
					log_normalizing_constant_ = w_pos * (-std::log(aux_pars_[0]) - M_LOGSQRT2PI);// Per positive y: -log(lambda) - 0.5*log(2*pi)
					log_normalizing_constant_ += ((1.0 / aux_pars_[0]) - 1.0) * aux_log_normalizing_constant_;// Jacobian term: (1/lambda - 1) * sum_{y>0} w * log(y)
				}
				else if (likelihood_type_ == "zoctn") {
					const double sigma = aux_pars_[0];
					const double a = aux_pars_original_[1];
					const double b = aux_pars_[2];
					double csum = 0.0;
					double w_pos = 0.0;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:csum, w_pos)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double yi = y_data[i];
						if (yi > 0.0 && yi < 1.0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							w_pos += w;;
							const double s_arg = (GPBoost::logit(yi) - a) / b;
							const double x = GPBoost::sigmoid_stable(s_arg);
							const double log_x1mx = std::log(x) + std::log1p(-x);
							csum += w * log_x1mx;
						}
					}
					log_normalizing_constant_ = csum + w_pos * (-std::log(sigma) - std::log(b) - M_LOGSQRT2PI) +
						aux_log_normalizing_constant_;
				}
				else if (likelihood_type_ == "zero_one_censored_transformed_beta") {
					double w_int = 0.0;
#pragma omp parallel for schedule(static) reduction(+:w_int)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double yi = y_data[i];
						if (yi > 0.0 && yi < 1.0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							w_int += w;
						}
					}
					log_normalizing_constant_ = -w_int * std::log(1.0 + 2.0 * aux_pars_[1]);
				}
				else if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
					const double k = aux_pars_[0];
					const double xi = aux_pars_[1];
					double s_log_yxi_int = 0.0;
					double w_int = 0.0;
#pragma omp parallel for schedule(static) reduction(+:s_log_yxi_int,w_int)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double yi = y_data[i];
						if (yi > 0.0 && yi < 1.0) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							s_log_yxi_int += w * std::log(yi + xi);
							w_int += w;
						}
					}
					log_normalizing_constant_ = (k - 1.0) * s_log_yxi_int - w_int * std::lgamma(k);
				}
				else if (likelihood_type_ == "asymmetric_laplace") {
					log_normalizing_constant_ = SumOfWeights() * (std::log(quantile_) + std::log(1. - quantile_) - std::log(aux_pars_[0]));
				}
				else {
					NotSupportedForLikelihood(__func__);
				}
				normalizing_constant_has_been_calculated_ = true;
			}
		}//end CalculateLogNormalizingConstant

		inline double LogNormalizingConstantPoissonOneSample(const int y) const {
			if (y > 1) {
				double log_factorial = 0.;
				for (int k = 2; k <= y; ++k) {
					log_factorial += std::log(k);
				}
				return(-log_factorial);
			}
			else {
				return(0.);
			}
		}

		inline double LogNormalizingConstantGamma() {
			CHECK(aux_normalizing_constant_has_been_calculated_);
			if (TwoNumbersAreEqual<double>(aux_pars_[0], 1.)) {
				return(0.);
			}
			else {
				return((aux_pars_[0] - 1.) * aux_log_normalizing_constant_ +
					SumOfWeights() * (aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0])));
			}
		}

		inline double LogNormalizingConstantGammaOneSample(const double y) const {
			if (TwoNumbersAreEqual<double>(aux_pars_[0], 1.)) {
				return(0.);
			}
			else {
				return((aux_pars_[0] - 1.) * AuxQuantLogNormalizingConstantGammaOneSample(y) +
					aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0]));
			}
		}

		inline double LogNormalizingConstantNegBin(const int* y_data_int) {
			CHECK(aux_normalizing_constant_has_been_calculated_);
			const double aux_const = SumOverSamplesWeighted([&](data_size_t i) { return std::lgamma(y_data_int[i] + aux_pars_[0]); });
			double norm_const = aux_const + aux_log_normalizing_constant_ +
				SumOfWeights() * (aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0]));
			return(norm_const);
		}

		inline double LogNormalizingConstantNegBinOneSample(const int y) const {
			double norm_const = std::lgamma(y + aux_pars_[0]) + AuxQuantLogNormalizingConstantNegBinOneSample(y) +
				aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0]);
			return(norm_const);
		}

		inline double LogNormalizingConstantNegBin1(const int* y_data_int) {
			CHECK(aux_normalizing_constant_has_been_calculated_);
			const double log_1_min_p = std::log(aux_pars_[0] / (1.0 + aux_pars_[0]));// log(1‑p)
			const double aux_const = SumOverSamplesWeighted([&](data_size_t i) { return y_data_int[i] * log_1_min_p; });
			double norm_const = aux_const + aux_log_normalizing_constant_;
			return(norm_const);
		}

		inline double LogNormalizingConstantNegBin1OneSample(const int y) const {
			const double log_1_min_p = std::log(aux_pars_[0] / (1.0 + aux_pars_[0]));// log(1‑p)
			double norm_const = y * log_1_min_p + AuxQuantLogNormalizingConstantNegBin1OneSample(y);
			return(norm_const);
		}

		inline double LogLikTweedie(double y, double eta, bool incl_norm_const) const {
			const double phi = aux_pars_[0];
			const double p = GetTweediePower();
			const auto location = EvaluateTweedieLocation(y, eta, std::log(phi), p);
			double ll = location.canonical;
			if (incl_norm_const) {
				const auto transform = likelihood_type_ == "tweedie" ? TransformTweediePowerFromQ(aux_pars_[1], TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_) : TweediePowerTransform{ p, 0., 0. };
				thread_local TweedieSpecialFunctionCache cache;
				const auto res = EvaluateTweedieLogNormalizer(y, std::log(phi), p, transform.dp_dtheta, transform.d2p_dtheta2, TweedieDerivativeOrder::kValue, false, 1000000, &cache);
				if (!res.converged) return std::numeric_limits<double>::quiet_NaN();
				ll += res.log_a;
			}
			return ll;
		}

		inline double FirstDerivLogLikTweedie(double y, double eta) const {
			const double p = GetTweediePower();
			return EvaluateTweedieLocation(y, eta, std::log(aux_pars_[0]), p).score;
		}

		inline double InformationLogLikTweedie(double y, double eta) const {
			const double p = GetTweediePower();
			return EvaluateTweedieLocation(y, eta, std::log(aux_pars_[0]), p).information;
		}

		/*!
		* \brief Dispatch on 'likelihood_type_' and call 'visit' with a kernel that evaluates the log-likelihood of one sample
		*			(without the sample weight). This is the single place where the mapping from a likelihood to its log-likelihood
		*			formula is defined; it is shared by the calculation over all samples ('LogLikelihood') and by the single-sample
		*			version ('LogLikelihoodOneSample'). Only likelihoods with a single location parameter block whose log-likelihood
		*			is a plain function of (y, eta) are covered here; the remaining ones are handled by the callers
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param incl_norm_const If true, the normalizing constant is included in the log-likelihood of every sample
		* \param visit Callable that is invoked with the selected kernel (a callable that takes a sample index)
		* \return True if the current likelihood is covered here, false otherwise
		*/
		template <class Visitor>
		bool VisitLogLikKernel(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			bool incl_norm_const,
			Visitor visit) const {
			if (likelihood_type_ == "bernoulli_probit") visit([&](data_size_t i) { return LogLikBernoulliProbit(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "bernoulli_logit") visit([&](data_size_t i) { return LogLikBernoulliLogit<int>(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "poisson") visit([&](data_size_t i) { return LogLikPoisson(y_data_int[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "gamma") visit([&](data_size_t i) { return LogLikGamma(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") visit([&](data_size_t i) { return LogLikTweedie(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "negative_binomial") visit([&](data_size_t i) { return LogLikNegBin(y_data_int[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "negative_binomial_1") visit([&](data_size_t i) { return LogLikNegBin1(y_data_int[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "zero_inflated_poisson") visit([&](data_size_t i) { return LogLikZeroInflatedPoisson(y_data_int[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") visit([&](data_size_t i) { return LogLikZeroInflatedNegBinFamily(y_data_int[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "beta") visit([&](data_size_t i) { return LogLikBeta(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "t") visit([&](data_size_t i) { return LogLikT(y_data[i], location_par[i], incl_norm_const); });
			else if (IsGaussianLikelihood()) visit([&](data_size_t i) { return LogLikGaussian(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "lognormal") visit([&](data_size_t i) { return LogLikLogNormal(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "hurdle_gamma") visit([&](data_size_t i) { return LogLikGammaZeroInflated(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "hurdle_lognormal") visit([&](data_size_t i) { return LogLikLogNormalZeroInflated(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") visit([&](data_size_t i) { return LogLikZeroCensPowNorm(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "zoctn") visit([&](data_size_t i) { return LogLikZeroOneCensTransfNorm(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "zero_one_censored_transformed_beta") visit([&](data_size_t i) { return LogLikZeroOneCensTransfBeta(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "zero_one_censored_shifted_gamma") visit([&](data_size_t i) { return LogLikZeroOneCensGamma(y_data[i], location_par[i], incl_norm_const); });
			else if (likelihood_type_ == "asymmetric_laplace") visit([&](data_size_t i) { return LogLikAsymLaplace(y_data[i], location_par[i], incl_norm_const); });
			else return false;
			return true;
		}//end VisitLogLikKernel

		/*!
		* \brief Evaluate the log-likelihood conditional on the latent variable (=location_par)
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		double LogLikelihood(const double* y_data,
			const int* y_data_int,
			const double* location_par) {
			CalculateLogNormalizingConstant(y_data, y_data_int);
			double ll = 0.;
			if (IsEGPDLikelihood() || IsHurdleEGPD()) {
				const bool hurdle = IsHurdleEGPD();
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:ll)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.;
					if (w == 0.) continue;
					if (hurdle && y_data[i] <= 0.) continue;// point mass log(p0) is in the normalizing constant
					const auto result = EvaluateEGPD(y_data[i], location_par[i]);
					ll += result.status == EGPDEvalStatus::kValid ? w * result.log_likelihood : -std::numeric_limits<double>::infinity();
				}
			}
			else if (likelihood_type_ == "binomial_probit" || likelihood_type_ == "quasi_bernoulli_probit") {
				ll += SumOverSamplesWeighted([&](data_size_t i) { return LogLikBinomialProbit(y_data[i], location_par[i]); });
			}
			else if (likelihood_type_ == "binomial_logit" || likelihood_type_ == "quasi_bernoulli_logit") {
				ll += SumOverSamplesWeighted([&](data_size_t i) { return LogLikBernoulliLogit<double>(y_data[i], location_par[i]); });
			}
			else if (likelihood_type_ == "beta_binomial") {
				CHECK(has_weights_);
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:ll)
				for (data_size_t i = 0; i < num_data_; ++i) {
					ll += LogLikBetaBinomial(y_data[i], location_par[i], weights_[i]);
				}
			}
			else if (IsGaussianHeteroscedastic()) {
				ll += SumOverSamplesWeighted([&](data_size_t i) { return LogLikGaussianHeteroscedastic(y_data[i], location_par[i], location_par[i + num_data_], false); });
			}
			else if (IsHurdleRegression()) {
				ll += SumOverSamplesWeighted([&](data_size_t i) { return LogLikHurdleRegression(y_data[i], location_par[i], location_par[i + num_data_]); });
			}
			else if (IsZeroInflatedCountRegression()) {
				ll += SumOverSamplesWeighted([&](data_size_t iz) { return LogLikZICountRegression(y_data_int[iz], location_par[iz], location_par[iz + num_data_]); });
			}
			else if (IsZeroCensPowNormHetero()) {
				ll += SumOverSamplesWeighted([&](data_size_t i) { return LogLikZeroCensPowNormHetero(y_data[i], location_par[i], location_par[i + num_data_], false); });
			}
			else if (!VisitLogLikKernel(y_data, y_data_int, location_par, false,
				[&](auto kernel) { ll += SumOverSamplesWeighted(kernel); })) {
				NotSupportedForLikelihood(__func__);
			}
			//Log::REInfo("ll = %g, log_normalizing_constant_ = %g", ll, log_normalizing_constant_);//for debugging
			ll += log_normalizing_constant_;
			return(ll);
		}//end LogLikelihood

		/*!
		* \brief Evaluate the log-likelihood conditional on the latent variable (=location_par) for one sample
		*			Note: this is only used for TestNegLogLikelihoodAdaptiveGHQuadrature
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		inline double LogLikelihoodOneSample(double y_data,
			int y_data_int,
			double location_par) const {
			if (NotImplementedForOneSample()) {
				FatalOneSampleNotImplemented(__func__);
				return(0.);
			}
			if (IsEGPDLikelihood()) {
				const auto result = EvaluateEGPD(y_data, location_par);
				return result.status == EGPDEvalStatus::kValid ? result.log_likelihood : -std::numeric_limits<double>::infinity();
			}
			if (IsHurdleEGPD()) {
				if (y_data <= 0.) return std::log(aux_pars_original_[num_aux_pars_ - 1]);// log(p0)
				const auto result = EvaluateEGPD(y_data, location_par);
				if (result.status != EGPDEvalStatus::kValid) return -std::numeric_limits<double>::infinity();
				return result.log_likelihood + std::log1p(-aux_pars_original_[num_aux_pars_ - 1]);// + log(q)
			}
			double ll = -1e99;
			if (!VisitLogLikKernel(&y_data, &y_data_int, &location_par, true,
				[&](auto kernel) { ll = kernel(0); })) {
				NotSupportedForLikelihood(__func__);
			}
			return(ll);
		}//end LogLikelihoodOneSample

		inline double LogLikBernoulliProbit(int y, double location_par) const {
			if (y == 0) {
				return GPBoost::normalLogCDF(-location_par);//log(1-Phi(x)) = log(Phi(-x))
			}
			else {
				return GPBoost::normalLogCDF(location_par);
			}
		}

		inline double LogLikBinomialProbit(double y, double location_par) const {
			if (y == 0.0) return GPBoost::normalLogCDF(-location_par);//log(1-Phi(x)) = log(Phi(-x))
			if (y == 1.0) return GPBoost::normalLogCDF(location_par);
			return y * GPBoost::normalLogCDF(location_par) + (1.0 - y) * GPBoost::normalLogCDF(-location_par);
		}

		template <typename T>
		inline double LogLikBernoulliLogit(T y, double location_par) const {
			return static_cast<double>(y) * location_par - GPBoost::softplus(location_par);
			// Alternative version (less numerically stable for large location_par
			//return (y * location_par - std::log(1.0 + std::exp(location_par)));
		}

		inline double LogLikPoisson(int y, double location_par, bool incl_norm_const) const {
			double ll = y * location_par - std::exp(location_par);
			if (incl_norm_const) {
				return (ll + LogNormalizingConstantPoissonOneSample(y));
			}
			else {
				return (ll);
			}
		}

		// ---------------------------------------------------------------------------------------------------------
		// Zero-inflated Poisson (constant structural-zero probability p0 = aux_pars_original_[0], pi = p0, q = 1 - p0)
		//   P(Y=0) = pi + q*exp(-mu),  P(Y=y>0) = q * Poisson(y; mu),  mu = exp(location_par)
		// For a zero count define D = pi + q*f0, w = pi/D (posterior structural-zero prob), v = q*f0/D = 1 - w.
		// ---------------------------------------------------------------------------------------------------------
		/*! \brief For a zero count: returns log D and sets w = pi/D (posterior structural-zero probability). b0 = log f0 = -mu. */
		inline double ZIPoissonZeroLogMixture(double mu, double p0, double& w) const {
			const double log_pi = std::log(p0);
			const double log_q = std::log1p(-p0);
			const double b0 = -mu;// log Poisson(0; mu)
			const double log_D = LogAddExpStable(log_pi, log_q + b0);
			w = std::exp(log_pi - log_D);// pi / D in [0,1]
			if (w > 1.) w = 1.;
			if (w < 0.) w = 0.;
			return log_D;
		}

		inline double LogLikZeroInflatedPoisson(int y, double location_par, bool incl_norm_const) const {
			const double p0 = aux_pars_original_[0];
			if (y == 0) {
				const double mu = std::exp(location_par);
				double w;
				return ZIPoissonZeroLogMixture(mu, p0, w);// = log D (fully location-dependent)
			}
			double ll = y * location_par - std::exp(location_par);// Poisson kernel (location-dependent part)
			if (incl_norm_const) {
				ll += std::log1p(-p0) - std::lgamma((double)y + 1.);// log q - log(y!)
			}
			return ll;
		}

		inline double FirstDerivLogLikZeroInflatedPoisson(int y, double location_par) const {
			const double mu = std::exp(location_par);
			if (y > 0) return (double)y - mu;// s_y
			const double p0 = aux_pars_original_[0];
			double w;
			ZIPoissonZeroLogMixture(mu, p0, w);
			const double v = 1. - w;
			return -v * mu;// l_eta = v * s0, s0 = -mu
		}

		/*! \brief Observed information J_eta = -d^2 l / d eta^2 (can be negative for zero counts) */
		inline double SecondDerivNegLogLikZeroInflatedPoisson(int y, double location_par) const {
			const double mu = std::exp(location_par);
			if (y > 0) return mu;// J_eta = -t_y = mu
			const double p0 = aux_pars_original_[0];
			double w;
			ZIPoissonZeroLogMixture(mu, p0, w);
			const double v = 1. - w;
			// J_eta = -(v*t0 + v*w*s0^2) = v*mu - v*w*mu^2  (t0 = -mu, s0^2 = mu^2)
			return v * mu - v * w * mu * mu;
		}

		/*! \brief d J_eta / d eta = -d^3 l / d eta^3 */
		inline double DerivInformationLocParZeroInflatedPoisson(int y, double location_par) const {
			const double mu = std::exp(location_par);
			if (y > 0) return mu;
			const double p0 = aux_pars_original_[0];
			double w;
			ZIPoissonZeroLogMixture(mu, p0, w);
			const double v = 1. - w;
			// dJ/deta = -(v*u0 + 3*v*w*s0*t0 + v*w*(w-v)*s0^3),  s0=t0=u0=-mu
			//         = v*mu - 3*v*w*mu^2 + v*w*(w-v)*mu^3
			return v * mu - 3. * v * w * mu * mu + v * w * (w - v) * mu * mu * mu;
		}

		// ---------------------------------------------------------------------------------------------------------
		// Zero-inflated negative binomial (NB2 and NB1), constant structural-zero probability.
		//   aux_pars_[0] = base shape (NB2 kappa) / dispersion (NB1 phi) (optimized on log-scale),
		//   aux_pars_[1] = transformed structural-zero odds A = p0/(1-p0)  (optimized on log(A) = logit(p0)).
		// At a zero count the base zero-mass f0 depends on both eta and the base aux parameter, so eta, rho and the
		// shape/dispersion all couple. All base zero-mass quantities are evaluated on the optimizer (log) scale for
		// the auxiliary parameter (g0,h0,k0). See plan Sections 5 and 8.
		// ---------------------------------------------------------------------------------------------------------
		struct ZICountZeroMass {
			double b0 = 0.;// log f0
			double s0 = 0., t0 = 0., u0 = 0.;// d/deta, d^2/deta^2, d^3/deta^3 of b0
			double g0 = 0., h0 = 0., k0 = 0.;// d/da, d^2/(deta da), d^3/(deta^2 da) of b0, a = log(aux_pars_[0])
		};

		/*! \brief NB2 zero-mass quantities at y=0, a = log(kappa) */
		inline void FillZeroMassNegBin(double mu, double kappa, ZICountZeroMass& z) const {
			const double kpm = kappa + mu;
			const double L = std::log1p(mu / kappa);
			z.b0 = -kappa * L;
			z.s0 = -kappa * mu / kpm;
			z.t0 = -kappa * kappa * mu / (kpm * kpm);
			z.u0 = -kappa * kappa * mu * (kappa - mu) / (kpm * kpm * kpm);
			z.g0 = kappa * (-L + mu / kpm);
			z.h0 = -kappa * mu * mu / (kpm * kpm);
			z.k0 = -2. * kappa * kappa * mu * mu / (kpm * kpm * kpm);
		}

		/*! \brief NB1 zero-mass quantities at y=0, a = log(phi). s0=t0=u0=b0 and h0=k0=g0. */
		inline void FillZeroMassNegBin1(double mu, double phi, ZICountZeroMass& z) const {
			const double L = std::log1p(phi);
			const double base = -mu * L / phi;
			z.b0 = base; z.s0 = base; z.t0 = base; z.u0 = base;
			const double g = (mu / phi) * (L - phi / (1. + phi));
			z.g0 = g; z.h0 = g; z.k0 = g;
		}

		/*! \brief For a zero count: log D = log(pi + q*f0); sets w = pi/D (posterior structural-zero prob) and v = 1 - w. */
		inline double ZICountZeroLogMixture(double p0, double b0, double& w, double& v) const {
			const double log_pi = std::log(p0);
			const double log_q = std::log1p(-p0);
			const double log_D = LogAddExpStable(log_pi, log_q + b0);
			w = std::exp(log_pi - log_D);
			if (w > 1.) w = 1.;
			if (w < 0.) w = 0.;
			v = 1. - w;
			return log_D;
		}
		// Generic zero-count derivative building blocks (given zero-mass z and mixture weights w,v):
		inline double ZICountZero_dEta(const ZICountZeroMass& z, double v) const { return v * z.s0; }// l_eta
		inline double ZICountZero_Jeta(const ZICountZeroMass& z, double w, double v) const { return -(v * z.t0 + v * w * z.s0 * z.s0); }// J_eta = -l_etaeta
		inline double ZICountZero_dJetadEta(const ZICountZeroMass& z, double w, double v) const {
			return -(v * z.u0 + 3. * v * w * z.s0 * z.t0 + v * w * (w - v) * z.s0 * z.s0 * z.s0);
		}
		inline double ZICountZero_lEtaRho(const ZICountZeroMass& z, double w, double v) const { return -v * w * z.s0; }
		inline double ZICountZero_dJetadRho(const ZICountZeroMass& z, double w, double v) const { return v * w * (z.t0 + (w - v) * z.s0 * z.s0); }
		inline double ZICountZero_lEtaShape(const ZICountZeroMass& z, double w, double v) const { return v * z.h0 + v * w * z.s0 * z.g0; }
		inline double ZICountZero_dJetadShape(const ZICountZeroMass& z, double w, double v) const {
			return -(v * z.k0 + v * w * z.g0 * z.t0 + v * w * (w - v) * z.g0 * z.s0 * z.s0 + 2. * v * w * z.s0 * z.h0);
		}

		inline void FillZeroMassZICount(double mu, ZICountZeroMass& z) const {
			if (likelihood_type_ == "zero_inflated_negative_binomial") FillZeroMassNegBin(mu, aux_pars_[0], z);
			else FillZeroMassNegBin1(mu, aux_pars_[0], z);// zero_inflated_negative_binomial_1
		}

		// -------- Fisher (expected) information wrt eta for the fisher_laplace approximation of count likelihoods. --------
		// The expected information E[(d log L / d eta)^2] does not depend on the realized y, so (unlike the observed Hessian,
		// which is negative at some zero counts) it is guaranteed nonnegative. This makes W >= 0, which stabilizes mode finding
		// and enables the iterative matrix-inversion methods (Vecchia, crossed grouped REs).
		/*! \brief Base kind of a zero-inflated count likelihood: 0 = Poisson, 1 = NB1, 2 = NB2. */
		inline int ZICountBaseKind() const {
			const string_t b = IsZeroInflatedCountRegression() ? ZICountRegressionBaseType() : likelihood_type_;
			if (b == "zero_inflated_negative_binomial") return 2;
			if (b == "zero_inflated_negative_binomial_1") return 1;
			return 0;
		}
		/*! \brief Constant structural-zero probability pi = p0 (untransformed) for a constant zero-inflated count. */
		inline double ZICountConstantP0() const {
			return aux_pars_original_[(likelihood_type_ == "zero_inflated_poisson") ? 0 : 1];
		}
		/*! \brief Fill the base zero-mass quantities for a given base kind (0 = Poisson, 1 = NB1, 2 = NB2). */
		inline void FillZeroMassZICountKind(double mu, ZICountZeroMass& z, int kind) const {
			if (kind == 2) FillZeroMassNegBin(mu, aux_pars_[0], z);
			else if (kind == 1) FillZeroMassNegBin1(mu, aux_pars_[0], z);
			else { z.b0 = -mu; z.s0 = -mu; z.t0 = -mu; z.u0 = -mu; z.g0 = 0.; z.h0 = 0.; z.k0 = 0.; }// Poisson
		}
		/*! \brief Base-count Fisher information wrt eta (mu = exp(eta)). Exact for Poisson (mu) and NB2 (mu*kappa/(kappa+mu)).
		* NB1 has NO closed-form exact Fisher information wrt eta (it is r^2*Var(digamma(Y+r)) with r = mu/phi, an infinite sum
		* over the NB1 pmf), so the GLM/quasi expected information mu^2/Var(Y) = mu/(1+phi) is used instead -- a QUASI-Fisher
		* information. kind: 0 = Poisson, 1 = NB1, 2 = NB2. */
		inline double ZICountBaseFisherInfoEta(double mu, int kind) const {
			if (kind == 2) return mu * aux_pars_[0] / (aux_pars_[0] + mu);// NB2, exact (kappa = aux_pars_[0])
			if (kind == 1) return mu / (1. + aux_pars_[0]);// NB1, quasi-Fisher (phi = aux_pars_[0])
			return mu;// Poisson, exact
		}
		/*! \brief Fisher (expected) information of a zero-inflated count wrt eta:
		*   E[(d log L / d eta)^2] = D * v^2 * s0^2 + (1 - pi) * (I_base - f0 * s0^2)  >= 0,  D = pi + (1-pi) f0, v = (1-pi) f0 / D.
		* Exact for Poisson/NB2; quasi (via the NB1 base) for NB1. */
		inline double ZICountFisherInfoEta(double mu, double pi, const ZICountZeroMass& z, int kind) const {
			const double q = 1. - pi;
			const double f0 = std::exp(z.b0);
			const double D = pi + q * f0;
			const double v = q * f0 / D;
			const double fisher = D * v * v * z.s0 * z.s0 + q * (ZICountBaseFisherInfoEta(mu, kind) - f0 * z.s0 * z.s0);
			return fisher > 0. ? fisher : 0.;// provably >= 0 for Poisson/NB2; guard tiny negatives from the NB1 quasi-Fisher
		}
		/*! \brief Fisher information wrt eta with the base aux parameter (kappa/phi) and pi given explicitly, for numerical
		* derivatives wrt the auxiliary parameters (dFisher/d log(shape) and dFisher/d rho). */
		inline double ZICountFisherInfoEtaExplicit(double mu, double pi, int kind, double base_aux) const {
			ZICountZeroMass z;
			if (kind == 2) FillZeroMassNegBin(mu, base_aux, z);
			else if (kind == 1) FillZeroMassNegBin1(mu, base_aux, z);
			else { z.b0 = -mu; z.s0 = -mu; }
			const double q = 1. - pi, f0 = std::exp(z.b0), D = pi + q * f0, v = q * f0 / D;
			const double I_base = (kind == 2) ? (mu * base_aux / (base_aux + mu)) : (kind == 1 ? mu / (1. + base_aux) : mu);
			const double fisher = D * v * v * z.s0 * z.s0 + q * (I_base - f0 * z.s0 * z.s0);
			return fisher > 0. ? fisher : 0.;
		}
		/*! \brief Set the information/approximation flags of NB1 and zero-inflated count likelihoods from approximation_type_ and use_fisher_for_mode_finding_.
		* fisher_laplace: nonnegative Fisher information for mode finding AND determinant. laplace: observed Hessian for the determinant
		* (can be negative); if use_fisher_for_mode_finding_ is set (the NB1/ZINB1 default), the positive (quasi-)Fisher is used for
		* mode finding only ('combined'). */
		void SetCountApproximationTypeFlags() {
			if (approximation_type_ == "fisher_laplace") {
				information_ll_can_be_negative_ = false;// Fisher (expected) information is nonnegative
				grad_information_wrt_mode_non_zero_ = true;// Fisher depends on mu
				// The Fisher information's derivative wrt the mode can vanish at isolated eta values (local extrema of the
				// information as a function of eta); enable the zero-guard used by the iterative ratio-trick diagonal
				// SigmaI_plus_W_inv_diag = d_log_det / deriv_information_diag_loc_par (0/0 -> set to 0 instead of NaN).
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
				information_changes_during_mode_finding_ = true;
				information_changes_after_mode_finding_ = false;
				use_fisher_for_mode_finding_ = true;
			} else if (approximation_type_ == "laplace") {
				information_ll_can_be_negative_ = true;// observed Hessian can be negative for NB1 and for zero-inflated counts
				grad_information_wrt_mode_non_zero_ = true;
				grad_information_wrt_mode_can_be_zero_for_some_points_ = true;
				information_changes_during_mode_finding_ = true;
				information_ll_can_be_exact_zero_ = true;
				if (use_fisher_for_mode_finding_) information_changes_after_mode_finding_ = true;// (quasi-)Fisher mode finding + Hessian determinant
			} else {
				Log::REFatal("'approximation_type' = '%s' is not supported for likelihood = '%s' ", approximation_type_.c_str(), likelihood_type_.c_str());
			}
		}

		inline double LogLikZeroInflatedNegBinFamily(int y, double location_par, bool incl_norm_const) const {
			const bool nb2 = (likelihood_type_ == "zero_inflated_negative_binomial");
			const double p0 = aux_pars_original_[1];
			if (y == 0) {
				const double mu = std::exp(location_par);
				ZICountZeroMass z; FillZeroMassZICount(mu, z);
				double w, v;
				return ZICountZeroLogMixture(p0, z.b0, w, v);// = log D (location- and shape-dependent)
			}
			double ll = nb2 ? LogLikNegBin(y, location_par, false) : LogLikNegBin1(y, location_par, false);
			if (incl_norm_const) {
				ll += std::log1p(-p0) + (nb2 ? LogNormalizingConstantNegBinOneSample(y) : LogNormalizingConstantNegBin1OneSample(y));
			}
			return ll;
		}

		inline double FirstDerivLogLikZeroInflatedNegBinFamily(int y, double location_par) const {
			const bool nb2 = (likelihood_type_ == "zero_inflated_negative_binomial");
			if (y > 0) return nb2 ? FirstDerivLogLikNegBin(y, location_par) : FirstDerivLogLikNegBin1(y, location_par);
			const double mu = std::exp(location_par);
			ZICountZeroMass z; FillZeroMassZICount(mu, z);
			double w, v; ZICountZeroLogMixture(aux_pars_original_[1], z.b0, w, v);
			return ZICountZero_dEta(z, v);
		}

		inline double SecondDerivNegLogLikZeroInflatedNegBinFamily(int y, double location_par) const {
			const bool nb2 = (likelihood_type_ == "zero_inflated_negative_binomial");
			if (y > 0) return nb2 ? SecondDerivNegLogLikNegBin(y, location_par) : SecondDerivNegLogLikNegBin1(y, location_par);
			const double mu = std::exp(location_par);
			ZICountZeroMass z; FillZeroMassZICount(mu, z);
			double w, v; ZICountZeroLogMixture(aux_pars_original_[1], z.b0, w, v);
			return ZICountZero_Jeta(z, w, v);
		}

		inline double DerivInformationLocParZeroInflatedNegBinFamily(int y, double location_par) const {
			const bool nb2 = (likelihood_type_ == "zero_inflated_negative_binomial");
			if (y > 0) {
				const double mu = std::exp(location_par);
				if (nb2) {
					const double mu_plus_r = mu + aux_pars_[0];
					return -(y + aux_pars_[0]) * mu * aux_pars_[0] * (mu - aux_pars_[0]) / (mu_plus_r * mu_plus_r * mu_plus_r);
				}
				const double r = mu / aux_pars_[0];
				const double dig_diff = GPBoost::digamma(y + r) - GPBoost::digamma(r);
				const double tri_diff = GPBoost::trigamma(y + r) - GPBoost::trigamma(r);
				const double tet_diff = GPBoost::tetragamma(y + r) - GPBoost::tetragamma(r);
				const double C = dig_diff - std::log1p(aux_pars_[0]);
				return -3.0 * r * r * tri_diff - r * r * r * tet_diff - r * C;
			}
			const double mu = std::exp(location_par);
			ZICountZeroMass z; FillZeroMassZICount(mu, z);
			double w, v; ZICountZeroLogMixture(aux_pars_original_[1], z.b0, w, v);
			return ZICountZero_dJetadEta(z, w, v);
		}

		/*! \brief Base NB2/NB1 negative-log-likelihood gradient wrt log(shape/dispersion) for a single positive count. */
		inline double NegLLGradShapeNegBinFamilyPos(int y, double location_par) const {
			const double mu = std::exp(location_par);
			if (likelihood_type_ == "zero_inflated_negative_binomial") {
				const double mu_plus_r = mu + aux_pars_[0];
				const double y_plus_r = y + aux_pars_[0];
				return aux_pars_[0] * (-GPBoost::digamma(y_plus_r) + std::log(mu_plus_r) + y_plus_r / mu_plus_r)
					+ aux_pars_[0] * (GPBoost::digamma(aux_pars_[0]) - std::log(aux_pars_[0]) - 1.);
			}
			const double r = mu / aux_pars_[0];
			const double C = GPBoost::digamma(y + r) - GPBoost::digamma(r) - std::log1p(aux_pars_[0]);
			return r * C + (mu - y) / (1.0 + aux_pars_[0]);
		}

		// ---------------------------------------------------------------------------------------------------------
		// Hurdle lognormal (constant structural-zero probability p0 = aux_pars_original_[1]).
		//   P(Y=0) = p0,  f(y | y>0) = LogNormal(y; location_par, sigma2).  The zero part fully decouples from eta:
		//   all eta-derivatives vanish at y=0, and l_{eta,rho} = 0 everywhere (same structure as hurdle_gamma).
		// ---------------------------------------------------------------------------------------------------------
		inline double LogLikLogNormalZeroInflated(double y, double location_par, bool incl_norm_const) const {
			if (y > 0.) {
				double ll = LogLikLogNormal(y, location_par, false);
				if (incl_norm_const) ll += std::log1p(-aux_pars_original_[1]) - std::log(y) - M_LOGSQRT2PI - 0.5 * std::log(aux_pars_[0]);
				return ll;
			}
			if (incl_norm_const) return std::log(aux_pars_original_[1]);// log p0
			return 0.;
		}

		inline double FirstDerivLogLikLogNormalZeroInflated(double y, double location_par) const {
			if (y <= 0.) return 0.;
			return FirstDerivLogLikLogNormal(y, location_par);
		}

		inline double SecondDerivNegLogLikLogNormalZeroInflated(double y) const {
			if (y <= 0.) return 0.;
			return 1. / aux_pars_[0];// constant Fisher information for positive observations
		}

		/*! \brief Stable softplus log(1 + exp(x)). */
		static inline double SoftplusStable(double x) { return (x > 0. ? x : 0.) + std::log1p(std::exp(-std::fabs(x))); }

		// ---------------------------------------------------------------------------------------------------------
		// Hurdle regression (fixed-effects structural-zero model) two-block helpers.
		//   Block 0 (eta = location_par): the positive base likelihood, evaluated on y > 0 (random effects live here).
		//   Block 1 (zeta = location_par2): a logistic model for the structural zero, pi = logit^{-1}(zeta), on d = 1{y=0}.
		//   The two blocks decouple: l_{eta,zeta} = 0 and dJ_eta/dzeta = 0.
		// ---------------------------------------------------------------------------------------------------------
		inline double HurdleRegressionBaseLogLikLocDep(double y, double loc_eta) const {
			// location-dependent part of the positive base log-likelihood (the aux-dependent normalizer is in log_normalizing_constant_ for
			// gamma/lognormal; for EGPD, EvaluateEGPD already returns the complete density and log_normalizing_constant_ is 0)
			const string_t base = HurdleRegressionBaseType();
			if (base == "hurdle_gamma") return LogLikGammaZeroInflated(y, loc_eta, false);
			if (base == "hurdle_lognormal") return LogLikLogNormalZeroInflated(y, loc_eta, false);
			const auto r = EvaluateEGPD(y, loc_eta);
			return r.status == EGPDEvalStatus::kValid ? r.log_likelihood : -std::numeric_limits<double>::infinity();
		}
		inline double HurdleRegression_dEta(double y, double loc_eta) const {// l_eta (0 at y=0)
			if (y <= 0.) return 0.;
			const string_t base = HurdleRegressionBaseType();
			if (base == "hurdle_gamma") return FirstDerivLogLikGammaZeroInflated(y, loc_eta);
			if (base == "hurdle_lognormal") return FirstDerivLogLikLogNormalZeroInflated(y, loc_eta);
			const auto r = EvaluateEGPD(y, loc_eta);
			return r.status == EGPDEvalStatus::kValid ? r.d_eta : std::numeric_limits<double>::quiet_NaN();
		}
		inline double HurdleRegression_Jeta(double y, double loc_eta) const {// J_eta = -l_etaeta (0 at y=0)
			if (y <= 0.) return 0.;
			const string_t base = HurdleRegressionBaseType();
			if (base == "hurdle_gamma") return SecondDerivNegLogLikGammaZeroInflated(y, loc_eta);
			if (base == "hurdle_lognormal") return SecondDerivNegLogLikLogNormalZeroInflated(y);
			const auto r = EvaluateEGPD(y, loc_eta);
			return r.status == EGPDEvalStatus::kValid ? -r.d2_eta : std::numeric_limits<double>::quiet_NaN();
		}
		inline double HurdleRegression_dJetadEta(double y, double loc_eta) const {// dJ_eta/deta = -l_etaetaeta (0 at y=0)
			if (y <= 0.) return 0.;
			const string_t base = HurdleRegressionBaseType();
			if (base == "hurdle_gamma") return -aux_pars_[0] * y * std::exp(-loc_eta);
			if (base == "hurdle_lognormal") return 0.;
			const auto r = EvaluateEGPD(y, loc_eta);
			return r.status == EGPDEvalStatus::kValid ? -r.d3_eta : std::numeric_limits<double>::quiet_NaN();
		}
		inline double HurdleRegression_dZeta(double y, double loc_zeta) const { return (y <= 0. ? 1. : 0.) - GPBoost::sigmoid_stable(loc_zeta); }// l_zeta = d - pi
		inline double HurdleRegression_Jzeta(double loc_zeta) const { const double p = GPBoost::sigmoid_stable(loc_zeta); return p * (1. - p); }// -l_zetazeta = pi*q
		inline double LogLikHurdleRegression(double y, double loc_eta, double loc_zeta) const {
			if (y <= 0.) return -SoftplusStable(-loc_zeta);// log(pi)
			return HurdleRegressionBaseLogLikLocDep(y, loc_eta) - SoftplusStable(loc_zeta);// base location-dependent part + log(q)
		}

		// ---------------------------------------------------------------------------------------------------------
		// Zero-inflated COUNT regression (coupled two-block) helpers. Block 0 (eta) = count response predictor (with RE);
		// block 1 (zeta) = structural-zero logit, pi_i = logit^{-1}(zeta_i). At a zero count eta and zeta couple.
		// ---------------------------------------------------------------------------------------------------------
		inline void FillZeroMassCountRegression(double mu, ZICountZeroMass& z) const {
			const string_t base = ZICountRegressionBaseType();
			if (base == "zero_inflated_negative_binomial") FillZeroMassNegBin(mu, aux_pars_[0], z);
			else if (base == "zero_inflated_negative_binomial_1") FillZeroMassNegBin1(mu, aux_pars_[0], z);
			else { z.b0 = -mu; z.s0 = -mu; z.t0 = -mu; z.u0 = -mu; z.g0 = 0.; z.h0 = 0.; z.k0 = 0.; }// Poisson
		}
		inline double CountRegBaseDEta(int y, double loc_eta) const {// base count s_y (y>0)
			const string_t base = ZICountRegressionBaseType();
			if (base == "zero_inflated_negative_binomial") return FirstDerivLogLikNegBin(y, loc_eta);
			if (base == "zero_inflated_negative_binomial_1") return FirstDerivLogLikNegBin1(y, loc_eta);
			return FirstDerivLogLikPoisson(y, loc_eta);
		}
		inline double CountRegBaseJeta(int y, double loc_eta) const {// base count -t_y (y>0)
			const string_t base = ZICountRegressionBaseType();
			if (base == "zero_inflated_negative_binomial") return SecondDerivNegLogLikNegBin(y, loc_eta);
			if (base == "zero_inflated_negative_binomial_1") return SecondDerivNegLogLikNegBin1(y, loc_eta);
			return SecondDerivNegLogLikPoisson(loc_eta);
		}
		inline double CountRegBaseDJetadEta(int y, double loc_eta) const {// base count dJ_eta/deta (y>0)
			const string_t base = ZICountRegressionBaseType();
			const double mu = std::exp(loc_eta);
			if (base == "zero_inflated_negative_binomial") { const double mpr = mu + aux_pars_[0]; return -(y + aux_pars_[0]) * mu * aux_pars_[0] * (mu - aux_pars_[0]) / (mpr * mpr * mpr); }
			if (base == "zero_inflated_negative_binomial_1") { const double r = mu / aux_pars_[0]; const double dd = GPBoost::digamma(y + r) - GPBoost::digamma(r); const double td = GPBoost::trigamma(y + r) - GPBoost::trigamma(r); const double tt = GPBoost::tetragamma(y + r) - GPBoost::tetragamma(r); const double C = dd - std::log1p(aux_pars_[0]); return -3. * r * r * td - r * r * r * tt - r * C; }
			return mu;// Poisson
		}
		inline double CountRegBaseLogLikLocDep(int y, double loc_eta) const {// location-dependent base count log-lik (y>0)
			const string_t base = ZICountRegressionBaseType();
			if (base == "zero_inflated_negative_binomial") return LogLikNegBin(y, loc_eta, false);
			if (base == "zero_inflated_negative_binomial_1") return LogLikNegBin1(y, loc_eta, false);
			return y * loc_eta - std::exp(loc_eta);// Poisson
		}
		/*! \brief All two-block derivative quantities of a zero-inflated count regression likelihood at one observation. */
		struct ZICountRegQuant { double dEta = 0., Jeta = 0., dJetadEta = 0., dZeta = 0., Jzeta = 0., lEtaZeta = 0., dJetadZeta = 0., v = 0.; };
		inline void ZICountRegressionQuantities(int y, double loc_eta, double loc_zeta, ZICountRegQuant& o) const {
			const double pi = GPBoost::sigmoid_stable(loc_zeta);
			const double q = 1. - pi;
			if (y > 0) {
				o.dEta = CountRegBaseDEta(y, loc_eta);
				o.Jeta = CountRegBaseJeta(y, loc_eta);
				o.dJetadEta = CountRegBaseDJetadEta(y, loc_eta);
				o.dZeta = -pi;// tau - pi, tau = 0
				o.Jzeta = pi * q;
				o.lEtaZeta = 0.; o.dJetadZeta = 0.; o.v = 1.;
				return;
			}
			const double mu = std::exp(loc_eta);
			ZICountZeroMass z; FillZeroMassCountRegression(mu, z);
			const double log_pi = std::log(pi), log_q = std::log1p(-pi);
			const double log_D = LogAddExpStable(log_pi, log_q + z.b0);
			double w = std::exp(log_pi - log_D); if (w > 1.) w = 1.; if (w < 0.) w = 0.;
			const double v = 1. - w;
			o.v = v;
			o.dEta = v * z.s0;
			o.Jeta = -(v * z.t0 + v * w * z.s0 * z.s0);
			o.dJetadEta = -(v * z.u0 + 3. * v * w * z.s0 * z.t0 + v * w * (w - v) * z.s0 * z.s0 * z.s0);
			o.dZeta = w - pi;// tau - pi, tau = w
			o.Jzeta = pi * q - w * v;
			o.lEtaZeta = -v * w * z.s0;
			o.dJetadZeta = v * w * (z.t0 + (w - v) * z.s0 * z.s0);
		}
		/*! \brief Derivative of the eta-block information W wrt zeta at one zero-inflated count regression observation, for the
		* zeta-block log-determinant gradient term: the observed dJ_eta/dzeta for the Laplace approximation, or d Fisher_eta / d zeta
		* (numerical central difference; Fisher_eta depends on zeta through pi = sigmoid(zeta)) for the fisher_laplace approximation. */
		inline double RegressionZeroModel_dInfodZeta(double loc_eta, double loc_zeta, const ZICountRegQuant& o) const {
			if (approximation_type_ != "fisher_laplace") return o.dJetadZeta;
			const int kind = ZICountBaseKind();
			const double mu = std::exp(loc_eta), h = 1e-5;
			ZICountZeroMass z; FillZeroMassZICountKind(mu, z, kind);
			return (ZICountFisherInfoEta(mu, GPBoost::sigmoid_stable(loc_zeta + h), z, kind)
				- ZICountFisherInfoEta(mu, GPBoost::sigmoid_stable(loc_zeta - h), z, kind)) / (2. * h);
		}
		inline double LogLikZICountRegression(int y, double loc_eta, double loc_zeta) const {
			if (y <= 0.) {// log(pi + q*f0) = log D
				const double mu = std::exp(loc_eta);
				ZICountZeroMass z; FillZeroMassCountRegression(mu, z);
				return LogAddExpStable(-SoftplusStable(-loc_zeta), -SoftplusStable(loc_zeta) + z.b0);// log(pi) , log(q)+b0
			}
			return CountRegBaseLogLikLocDep(y, loc_eta) - SoftplusStable(loc_zeta);// base location-dependent part + log(q)
		}
		/*! \brief Structural-zero-block score l_zeta = tau - pi for a regression zero model (hurdle or count). */
		inline double RegressionZeroModel_dZeta(const double* y_data, const int* y_data_int, data_size_t iz, double loc_eta, double loc_zeta) const {
			if (IsHurdleRegression()) return HurdleRegression_dZeta(y_data[iz], loc_zeta);
			ZICountRegQuant o; ZICountRegressionQuantities(y_data_int[iz], loc_eta, loc_zeta, o);
			return o.dZeta;
		}
		// NOTE: the former 'RegressionZeroModel_dZetaDense' was removed here: all of its callers now go through
		// 'CalcSecondFEBlockFixedEffectGrad', which dispatches on the likelihood once instead of calling
		// 'IsHurdleRegression()' (a chain of string comparisons) once per observation

		inline double LogLikGamma(double y, double location_par, bool incl_norm_const) const {
			double ll = -aux_pars_[0] * (location_par + y * std::exp(-location_par));
			if (incl_norm_const) {
				return (ll + LogNormalizingConstantGammaOneSample(y));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikNegBin(int y, double location_par, bool incl_norm_const) const {
			double ll = y * location_par - (y + aux_pars_[0]) * std::log(std::exp(location_par) + aux_pars_[0]);
			if (incl_norm_const) {
				return (ll + LogNormalizingConstantNegBinOneSample(y));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikNegBin1(int y, double location_par, bool incl_norm_const) const {
			const double r = std::exp(location_par) / aux_pars_[0];
			double ll = std::lgamma(y + r) - std::lgamma(r) - r * std::log1p(aux_pars_[0]);
			if (incl_norm_const) {
				return (ll + LogNormalizingConstantNegBin1OneSample(y));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikBeta(double y, double location_par, bool incl_norm_const) const {
			const double mu = GPBoost::sigmoid_stable_clamped(location_par);
			double ll = -std::lgamma(mu * aux_pars_[0]) - std::lgamma((1. - mu) * aux_pars_[0])
				+ (mu * aux_pars_[0] - 1.) * std::log(y) + ((1. - mu) * aux_pars_[0] - 1.) * std::log1p(-y);
			if (incl_norm_const) {
				return (ll + std::lgamma(aux_pars_[0]));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikT(double y, double location_par, bool incl_norm_const) const {
			double ll = -(aux_pars_[1] + 1.) / 2. * std::log(1. + (y - location_par) * (y - location_par) / (aux_pars_[1] * aux_pars_[0] * aux_pars_[0]));
			if (incl_norm_const) {
				return (ll - std::log(aux_pars_[0]) +
					std::lgamma((aux_pars_[1] + 1.) / 2.) - 0.5 * std::log(aux_pars_[1]) -
					0.5 * std::lgamma(aux_pars_[1] / 2.) - 0.5 * std::log(M_PI));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikGaussian(double y, double location_par, bool incl_norm_const) const {
			double resid = y - location_par;
			double ll = -resid * resid / 2. / aux_pars_[0];
			if (incl_norm_const) {
				return (ll - M_LOGSQRT2PI - 0.5 * std::log(aux_pars_[0]));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikGaussianHeteroscedastic(double y, double location_par,
			double location_par2, bool incl_norm_const) const {
			double resid = y - location_par;
			double ll = -resid * resid * std::exp(-location_par2) / 2. - location_par2 / 2.;
			if (incl_norm_const) {
				return (ll - M_LOGSQRT2PI);
			}
			else {
				return (ll);
			}
		}

		inline double LogLikLogNormal(double y, double location_par, bool incl_norm_const) const {
			const double s2 = aux_pars_[0];
			const double z = std::log(y) - (location_par - 0.5 * s2);//meanlog = location_par - 0.5 * s2
			double ll = -0.5 * z * z / s2;
			if (incl_norm_const) {
				ll += -std::log(y) - M_LOGSQRT2PI - 0.5 * std::log(s2);
			}
			return ll;
		}

		inline double LogLikBetaBinomial(double y_ratio, double location_par, double w) {
			if (w <= 0.0) return 0.0; // degenerate: no info
			const double mu = GPBoost::sigmoid_stable_clamped(location_par);
			const double phi_raw = aux_pars_[0];
			const double phi = (phi_raw > 0.0 && std::isfinite(phi_raw)) ? phi_raw : 1e-16;
			const double a = mu * phi;
			const double b = (1.0 - mu) * phi;
			const double k = y_ratio * w;
			double ll = std::lgamma(k + a) + std::lgamma(w - k + b) - std::lgamma(w + phi)
				- (std::lgamma(a) + std::lgamma(b) - std::lgamma(phi));
			return ll;
		}

		inline double LogLikGammaZeroInflated(double y, double location_par, bool incl_norm_const) const {
			double ll = 0.0;
			if (y > 0.0) {
				ll = -aux_pars_[0] * (location_par + y * std::exp(-location_par));
				if (incl_norm_const) {
					//note: 'aux_pars_original_[1]' (= p0) must only be accessed if the normalizing constant is needed.
					//	This function is also called for 'hurdle_regression_gamma' (via 'HurdleRegressionBaseLogLikLocDep',
					//	always with incl_norm_const = false), where the structural zero is modeled by a second fixed-effects
					//	block instead of by p0, i.e. where 'aux_pars_' only contains the shape parameter and index 1 does not exist
					const double q = 1. - aux_pars_original_[1];// = 1 - p0
					ll += std::log(q) + aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0]) + (aux_pars_[0] - 1.0) * std::log(y);
				}
			}
			else { // y == 0
				if (incl_norm_const) {
					const double p0 = aux_pars_original_[1];
					ll += std::log(p0);
				}
			}
			return ll;
		}

		inline double LogLikZeroCensPowNorm(double y, double location_par, bool incl_norm_const) const {
			const double sigma = aux_pars_[0];
			if (y <= 0.0) {
				const double a0 = -location_par / sigma;
				return GPBoost::normalLogCDF(a0);// log Phi(a0)
			}
			else {
				const double lambda = aux_pars_[1];
				const double u = std::exp((1.0 / lambda) * std::log(y)); // y^(1/lambda)
				const double z = (u - location_par) / sigma;
				double ll = -0.5 * z * z;
				if (incl_norm_const) {
					ll += -std::log(lambda) - std::log(sigma) - M_LOGSQRT2PI + (1.0 / lambda - 1.0) * std::log(y);
				}
				return ll;
			}
		}

		// ---------------------------------------------------------------------------------------------------------
		// Heteroscedastic zero-censored power-transformed normal: Y = max(0,X)^lambda, X ~ N(mu, sigma^2) with
		//   mu = loc_eta (block 0, fixed + random effects) and log(sigma) = loc_zeta (block 1, fixed effects only).
		//   For y = 0:  l = log Phi(a0),                     a0 = -mu / sigma,  r = phi(a0) / Phi(a0)
		//   For y > 0:  l = -log(lambda) - loc_zeta - log(sqrt(2*pi)) + (1/lambda - 1) * log(y) - z^2 / 2,
		//               u = y^(1/lambda),  z = (u - mu) / sigma
		// All eta derivatives coincide with those of the homoscedastic variant with sigma replaced by sigma_i, and all
		// zeta derivatives coincide with the (log-scale) "sigma" auxiliary parameter derivatives of that variant.
		// ---------------------------------------------------------------------------------------------------------
		inline double LogLikZeroCensPowNormHetero(double y, double loc_eta, double loc_zeta, bool incl_norm_const) const {
			const double sigma = std::exp(loc_zeta);
			if (y <= 0.0) {
				return GPBoost::normalLogCDF(-loc_eta / sigma);// log Phi(a0)
			}
			const double lambda = aux_pars_[0];
			const double u = std::exp((1.0 / lambda) * std::log(y));// y^(1/lambda)
			const double z = (u - loc_eta) / sigma;
			double ll = -0.5 * z * z - loc_zeta;
			if (incl_norm_const) {
				ll += -std::log(lambda) - M_LOGSQRT2PI + (1.0 / lambda - 1.0) * std::log(y);
			}
			return ll;
		}

		/*! \brief First derivative of the heteroscedastic zero-censored power-transformed normal log-likelihood wrt eta (= mu) */
		inline double FirstDerivLogLikZeroCensPowNormHetero(double y, double loc_eta, double loc_zeta) const {
			const double sigma = std::exp(loc_zeta);
			if (y <= 0.0) {
				return -(1.0 / sigma) * GPBoost::InvMillsRatioNormalPhi(-loc_eta / sigma);
			}
			const double u = std::exp((1.0 / aux_pars_[0]) * std::log(y));
			return (u - loc_eta) / (sigma * sigma);// = z / sigma
		}

		/*! \brief Observed information (= negative second derivative) of the heteroscedastic zero-censored power-transformed normal log-likelihood wrt eta */
		inline double SecondDerivNegLogLikZeroCensPowNormHetero(double y, double loc_eta, double loc_zeta) const {
			const double sigma = std::exp(loc_zeta);
			if (y <= 0.0) {
				const double a0 = -loc_eta / sigma;
				const double r = GPBoost::InvMillsRatioNormalPhi(a0);
				return (1.0 / (sigma * sigma)) * r * (a0 + r);
			}
			return 1.0 / (sigma * sigma);
		}

		/*! \brief Derivative wrt eta of the eta-block information of the heteroscedastic zero-censored power-transformed normal log-likelihood */
		inline double DerivInformationZeroCensPowNormHetero(double y, double loc_eta, double loc_zeta) const {
			if (y > 0.0) {
				return 0.0;// the information 1 / sigma^2 does not depend on eta for positive observations
			}
			const double sigma = std::exp(loc_zeta);
			const double a0 = -loc_eta / sigma;
			const double r = GPBoost::InvMillsRatioNormalPhi(a0);
			return (r / (sigma * sigma * sigma)) * ((a0 + r) * (a0 + 2.0 * r) - 1.0);
		}

		/*!
		* \brief Quantities of the second, fixed-effects-only location parameter block (zeta = log(sigma)) of the
		*		heteroscedastic zero-censored power-transformed normal likelihood at one observation
		* \param[out] dZeta Score l_zeta = d log f / d zeta
		* \param[out] lEtaZeta Cross derivative d^2 log f / (d eta d zeta)
		* \param[out] dJetadZeta Derivative of the eta-block information wrt zeta
		*/
		inline void ZeroCensPowNormHeteroZetaQuantities(double y, double loc_eta, double loc_zeta,
			double& dZeta, double& lEtaZeta, double& dJetadZeta) const {
			const double sigma = std::exp(loc_zeta);
			if (y <= 0.0) {
				const double a0 = -loc_eta / sigma;
				const double r = GPBoost::InvMillsRatioNormalPhi(a0);
				dZeta = r * loc_eta / sigma;
				lEtaZeta = r * (1.0 + ((a0 + r) * loc_eta) / sigma) / sigma;
				dJetadZeta = r * ((loc_eta * (1.0 - (a0 + r) * (a0 + 2.0 * r))) / (sigma * sigma * sigma) - 2.0 * (a0 + r) / (sigma * sigma));
			}
			else {
				const double u = std::exp((1.0 / aux_pars_[0]) * std::log(y));
				const double z = (u - loc_eta) / sigma;
				dZeta = -1.0 + z * z;
				lEtaZeta = -2.0 * z / sigma;
				dJetadZeta = -2.0 / (sigma * sigma);
			}
		}

		/*!
		* \brief zeta-block (= log(sigma)) gradient of the negative approximate marginal log-likelihood at one observation of the
		*		heteroscedastic zero-censored power-transformed normal likelihood: the direct score, the log-determinant term
		*		(through dJ_eta/dzeta) and the implicit term through the mode (through l_{eta,zeta})
		* \param w Sample weight
		* \param diag Data-scale diagonal entry of (Sigma^-1 + W)^-1 at this observation
		* \param inv_d_mll_d_mode Data-scale entry of (Sigma^-1 + W)^-1 * d_mll_d_mode at this observation
		*/
		inline double ZeroCensPowNormHeteroZetaGrad(double y, double loc_eta, double loc_zeta, double w,
			double diag, double inv_d_mll_d_mode) const {
			double dZeta, lEtaZeta, dJetadZeta;
			ZeroCensPowNormHeteroZetaQuantities(y, loc_eta, loc_zeta, dZeta, lEtaZeta, dJetadZeta);
			return -w * dZeta + 0.5 * (w * dJetadZeta) * diag + (w * lEtaZeta) * inv_d_mll_d_mode;
		}

		inline double LogLikZeroOneCensTransfNorm(double y, double location_par, bool incl_norm_const) const {
			const double sigma = aux_pars_[0];
			if (y <= 0.0) {
				const double a0 = -location_par / sigma;
				return GPBoost::normalLogCDF(a0);// log Phi(a0)
			}
			else if (y >= 1.0) {
				const double v = (1.0 - location_par) / sigma;
				return GPBoost::normalLogCDF(-v); // log Phi(-(1-mu)/sigma) = log P(Z>=1)
			}
			else {
				const double a = aux_pars_original_[1];
				const double b = aux_pars_[2];
				const double s_arg = (GPBoost::logit(y) - a) / b;
				const double x = GPBoost::sigmoid_stable(s_arg);
				const double z = (x - location_par) / sigma;
				double ll = -0.5 * z * z;
				if (incl_norm_const) {
					const double log_x1mx = std::log(x) + std::log1p(-x);
					ll += -std::log(sigma) - M_LOGSQRT2PI + log_x1mx - std::log(b) - std::log(y) - std::log1p(-y);
				}
				return ll;
			}
		}

		inline double LogLikZeroOneCensTransfBeta(double y, double location_par, bool incl_norm_const) const {
			const double phi = std::max(aux_pars_[0], 1e-12);
			const double u = std::max(aux_pars_[1], 1e-12);
			return LogLikZeroOneCensTransfBeta_at(y, location_par, phi, u, incl_norm_const);
		}
		inline double LogLikZeroOneCensTransfBeta_at(double y, double location_par, double phi, double u, bool incl_norm_const) const {
			const double eps_mu = 1e-12;
			const double eps_ab = 1e-12;
			const double eps_t = 1e-15;
			const double eps_u = 1e-12;
			const double uu = std::max(u, eps_u);
			const double onep2u = 1.0 + 2.0 * uu;
			const double mu_raw = GPBoost::sigmoid_stable_clamped(location_par);
			const double mu = std::min(std::max(mu_raw, eps_mu), 1.0 - eps_mu);
			const double a = std::max(mu * phi, eps_ab);
			const double b = std::max((1.0 - mu) * phi, eps_ab);
			if (TwoNumbersAreEqual(y, 0.)) {
				const double t0 = std::min(std::max(uu / onep2u, eps_t), 1.0 - eps_t);
				return GPBoost::log_beta_cdf(t0, a, b);
			}
			else if (TwoNumbersAreEqual(y, 1.)) {
				const double t1 = std::min(std::max((1.0 + uu) / onep2u, eps_t), 1.0 - eps_t);
				return GPBoost::log1m_beta_cdf(t1, a, b);
			}
			else {
				double t = (y + uu) / onep2u;
				t = std::min(std::max(t, eps_t), 1.0 - eps_t);
				double ll = GPBoost::log_beta_pdf(t, a, b);
				if (incl_norm_const) ll += -std::log(onep2u);
				return ll;
			}
		}

		inline double LogLikZeroOneCensGamma(const double y, const double location_par, const bool incl_norm_const) const {
			return LogLikZeroOneCensGamma_at(y, location_par, aux_pars_[0], aux_pars_[1], incl_norm_const);
		}
		inline double LogLikZeroOneCensGamma_at(const double y, const double location_par, const double k, const double xi, const bool incl_norm_const) const {
			const double mu = std::exp(location_par);
			const double th = mu / k;
			const double tiny = 1e-300;
			if (y <= 0.0) {
				if (xi <= 0.0) { return 0.0; }
				const double t0 = xi / th;
				const double G0 = GPBoost::RegLowerGamma(k, t0);
				const double P0 = std::max(G0, tiny);
				return std::log(P0);
			}
			else if (y >= 1.0) {
				const double t1 = (1.0 + xi) / th;
				const double G1 = GPBoost::RegLowerGamma(k, t1);
				const double H1 = std::max(1.0 - G1, tiny);
				return std::log(H1);
				//double G = GPBoost::RegLowerGamma(k, t1); // alternative version, potentially more stable
				//if (!std::isfinite(G)) return std::log(tiny);
				//G = std::min(std::max(G, tiny), 1.0 - tiny);
				//return std::log1p(-G);
			}
			else {
				const double z = y + xi;
				double ll = -k * std::log(th) - z / th;
				if (incl_norm_const) {
					ll += (k - 1.0) * std::log(std::max(z, tiny)) - std::lgamma(k);
				}
				return ll;
			}
		}

		inline double LogLikAsymLaplace(double y, double location_par, bool incl_norm_const) const {
			double indicator = (y <= location_par) ? 1.0 : 0.0;
			double ll = (y - location_par) * (indicator - quantile_) / aux_pars_[0];
			if (incl_norm_const) {
				return (ll + std::log(quantile_) + std::log(1. - quantile_) - std::log(aux_pars_[0]));
			}
			else {
				return (ll);
			}
		}

		/*!
		* \brief Calculate the first derivative of the log-likelihood with respect to the location parameter aggregated per random effect (= Z^T * d/d eta log_lik(y|g(eta)), eta = Zb
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		void CalcFirstDerivLogLik(const double* y_data,
			const int* y_data_int,
			const double* location_par) {
			if (ssn_alm_exact_subgradient_valid_) {
				// After the SSN-ALM refinement the mode is stationary for an interior subgradient of the check loss at the
				// observations that lie exactly on a kink. Recomputing the score with the endpoint convention would break
				// the identity 'Q b = Z^T first_deriv_ll_' on which the gradients of the approximate marginal likelihood
				// and the predictive distributions rely, so the exact subgradient of the refinement is used instead
				SetFirstDerivLogLikFromDataScale(ssn_alm_exact_subgradient_);
				return;
			}
			if (use_random_effects_indices_of_data_) {
				CalcFirstDerivLogLik_PerSample(y_data, y_data_int, location_par, first_deriv_ll_data_scale_);
				ReduceToModeScale(first_deriv_ll_data_scale_, first_deriv_ll_);
			}
			else {//!use_random_effects_indices_of_data_
				CalcFirstDerivLogLik_PerSample(y_data, y_data_int, location_par, first_deriv_ll_);
			}
		}//end CalcFirstDerivLogLik

		/*!
		* \brief Set 'first_deriv_ll_' (and 'first_deriv_ll_data_scale_') from a score that is given on the observation scale
		* \param first_deriv_data_scale Score of length num_data_, already multiplied by the sample weights
		*/
		void SetFirstDerivLogLikFromDataScale(const vec_t& first_deriv_data_scale) {
			if (use_random_effects_indices_of_data_) {
				first_deriv_ll_data_scale_ = first_deriv_data_scale;
				ReduceToModeScale(first_deriv_ll_data_scale_, first_deriv_ll_);
			}
			else {
				first_deriv_ll_ = first_deriv_data_scale;
			}
		}//end SetFirstDerivLogLikFromDataScale

		/*!
		* \brief Dispatch on 'likelihood_type_' and call 'visit' with a kernel that evaluates the first derivative of the
		*			log-likelihood with respect to the location parameter for one sample (without the sample weight).
		*			This is the single place where the mapping from a likelihood to its first-derivative formula is defined;
		*			it is shared by the calculation over all samples ('CalcFirstDerivLogLik_PerSample') and by the single-sample
		*			version ('CalcFirstDerivLogLikOneSample'). Only likelihoods with a single location parameter block whose
		*			derivative is a plain function of (y, eta) are covered here; the remaining ones are handled by the callers
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param visit Callable that is invoked with the selected kernel (a callable that takes a sample index)
		* \return True if the current likelihood is covered here, false otherwise
		*/
		template <class Visitor>
		bool VisitFirstDerivLogLikKernel(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			Visitor visit) const {
			if (likelihood_type_ == "bernoulli_probit") visit([&](data_size_t i) { return FirstDerivLogLikBernoulliProbit(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "bernoulli_logit") visit([&](data_size_t i) { return FirstDerivLogLikBernoulliLogit<int>(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "binomial_probit" || likelihood_type_ == "quasi_bernoulli_probit") visit([&](data_size_t i) { return FirstDerivLogLikBinomialProbit(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "binomial_logit" || likelihood_type_ == "quasi_bernoulli_logit") visit([&](data_size_t i) { return FirstDerivLogLikBernoulliLogit<double>(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "poisson") visit([&](data_size_t i) { return FirstDerivLogLikPoisson(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "gamma") visit([&](data_size_t i) { return FirstDerivLogLikGamma(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") visit([&](data_size_t i) { return FirstDerivLogLikTweedie(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "negative_binomial") visit([&](data_size_t i) { return FirstDerivLogLikNegBin(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "negative_binomial_1") visit([&](data_size_t i) { return FirstDerivLogLikNegBin1(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "zero_inflated_poisson") visit([&](data_size_t i) { return FirstDerivLogLikZeroInflatedPoisson(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") visit([&](data_size_t i) { return FirstDerivLogLikZeroInflatedNegBinFamily(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "beta") visit([&](data_size_t i) { return FirstDerivLogLikBeta(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "t") visit([&](data_size_t i) { return FirstDerivLogLikT(y_data[i], location_par[i]); });
			else if (IsGaussianLikelihood()) visit([&](data_size_t i) { return FirstDerivLogLikGaussian(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "lognormal") visit([&](data_size_t i) { return FirstDerivLogLikLogNormal(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "hurdle_gamma") visit([&](data_size_t i) { return FirstDerivLogLikGammaZeroInflated(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "hurdle_lognormal") visit([&](data_size_t i) { return FirstDerivLogLikLogNormalZeroInflated(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") visit([&](data_size_t i) { return FirstDerivLogLikZeroCensPowNorm(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "zoctn") visit([&](data_size_t i) { return FirstDerivLogLikZeroOneCensTransfNorm(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "zero_one_censored_transformed_beta") visit([&](data_size_t i) { return FirstDerivLogLikZeroOneCensTransfBeta(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "zero_one_censored_shifted_gamma") visit([&](data_size_t i) { return FirstDerivLogLikZeroOneCensGamma(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "asymmetric_laplace") visit([&](data_size_t i) { return FirstDerivLogLikAsymLaplace(y_data[i], location_par[i]); });
			else return false;
			return true;
		}//end VisitFirstDerivLogLikKernel

		/*!
		* \brief Calculate the first derivative of the log-likelihood with respect to the location parameter for every sample
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param[out] first_deriv_ll First derivative of the log-likelihood with respect to the location parameter
		*/
		void CalcFirstDerivLogLik_PerSample(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			vec_t& first_deriv_ll) {
			if (IsEGPDLikelihood() || IsHurdleEGPD()) {
				const bool hurdle = IsHurdleEGPD();
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.;
					if (w == 0. || (hurdle && y_data[i] <= 0.)) first_deriv_ll[i] = 0.;// zero part decouples from eta
					else {
						const auto result = EvaluateEGPD(y_data[i], location_par[i]);
						first_deriv_ll[i] = result.status == EGPDEvalStatus::kValid ? w * result.d_eta : std::numeric_limits<double>::quiet_NaN();
					}
				}
			}
			else if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					FirstDerivLogLikGaussianHeteroscedastic(y_data[i], location_par[i], location_par[i + num_data_],
						first_deriv_ll[i], first_deriv_ll[i + num_data_]);
					if (has_weights_) {
						first_deriv_ll[i] *= weights_[i];
						first_deriv_ll[i + num_data_] *= weights_[i];
					}
				}
			}
			else if (likelihood_type_ == "gaussian_heteroscedastic") {
				// Only the mean is a mode / random effect here; the log-error variance (location_par[i + num_data_]) is a fixed effect
				ForEachSampleWeighted(first_deriv_ll, [&](data_size_t i) { return FirstDerivLogLikGaussianHeteroscedasticMean(y_data[i], location_par[i], location_par[i + num_data_]); });
			}
			else if (IsZeroCensPowNormHetero()) {
				// Only the mean / eta is a mode / random effect here; log(sigma) (location_par[i + num_data_]) is a fixed effect
				ForEachSampleWeighted(first_deriv_ll, [&](data_size_t i) { return FirstDerivLogLikZeroCensPowNormHetero(y_data[i], location_par[i], location_par[i + num_data_]); });
			}
			else if (IsHurdleRegression()) {
				// Random effects live on the response predictor eta (block 0); this is the block-0 score used for mode finding.
				ForEachSampleWeighted(first_deriv_ll, [&](data_size_t i) { return HurdleRegression_dEta(y_data[i], location_par[i]); });
			}
			else if (IsZeroInflatedCountRegression()) {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t iz = 0; iz < num_data_; ++iz) {
					const double wz = has_weights_ ? weights_[iz] : 1.0;
					ZICountRegQuant o; ZICountRegressionQuantities(y_data_int[iz], location_par[iz], location_par[iz + num_data_], o);
					first_deriv_ll[iz] = wz * o.dEta;
				}
			}
			else if (likelihood_type_ == "beta_binomial") {
				CHECK(has_weights_);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					first_deriv_ll[i] = FirstDerivLogLikBetaBinomial(y_data[i], location_par[i], weights_[i]);
				}
			}
			else {
				if (!VisitFirstDerivLogLikKernel(y_data, y_data_int, location_par, [&](auto kernel) { ForEachSampleWeighted(first_deriv_ll, kernel); })) {
					NotSupportedForLikelihood(__func__);
				}
				if (likelihood_type_ == "asymmetric_laplace" && approximation_type_ == "triangular_kernel_curvature") {
					sum_first_deriv_ = 0.;//can be left at 0. since linear differences cancel each other in positive and negative directions in 'GoodnessFit_TKC_approx' and 'NegativeHessian_TKC_Approx_AsymLaplace'
				}
			}
		}//end CalcFirstDerivLogLik_PerSample

		/*!
		* \brief Calculate the first derivative of the log-likelihood with respect to the location parameter
		*			Note: this is only used for 'TestNegLogLikelihoodAdaptiveGHQuadrature()'
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		inline double CalcFirstDerivLogLikOneSample(double y_data,
			int y_data_int,
			double location_par) const {
			if (NotImplementedForOneSample()) {
				FatalOneSampleNotImplemented(__func__);
				return(0.);
			}
			if (IsEGPDLikelihood() || IsHurdleEGPD()) {
				if (IsHurdleEGPD() && y_data <= 0.) return 0.;
				const auto result = EvaluateEGPD(y_data, location_par);
				return result.status == EGPDEvalStatus::kValid ? result.d_eta : std::numeric_limits<double>::quiet_NaN();
			}
			double first_deriv = 0.;
			if (!VisitFirstDerivLogLikKernel(&y_data, &y_data_int, &location_par,
				[&](auto kernel) { first_deriv = kernel(0); })) {
				NotSupportedForLikelihood(__func__);
			}
			return(first_deriv);
		}//end CalcFirstDerivLogLikOneSample

		inline double FirstDerivLogLikBernoulliProbit(int y, double location_par) const {
			if (y == 0) {
				return -GPBoost::InvMillsRatioNormalOneMinusPhi(location_par);//phi(x) / (1 - Phi(x))
			}
			else {
				return GPBoost::InvMillsRatioNormalPhi(location_par);//phi(x) / Phi(x)
			}
		}

		inline double FirstDerivLogLikBinomialProbit(double y, double location_par) const {
			if (y == 0.0) return -GPBoost::InvMillsRatioNormalOneMinusPhi(location_par);//phi(x) / (1 - Phi(x))
			if (y == 1.0) return GPBoost::InvMillsRatioNormalPhi(location_par);//phi(x) / Phi(x)
			const double pdf_div_cdf = GPBoost::InvMillsRatioNormalPhi(location_par);//phi(x) / Phi(x)
			const double pdf_div_omcdf = GPBoost::InvMillsRatioNormalOneMinusPhi(location_par);//phi(x) / (1 - Phi(x))
			return y * pdf_div_cdf + (1.0 - y) * -pdf_div_omcdf;
		}

		template <typename T>
		inline double FirstDerivLogLikBernoulliLogit(T y, double location_par) const {
			return y - GPBoost::sigmoid_stable(location_par);
		}

		inline double FirstDerivLogLikPoisson(int y, double location_par) const {
			return (y - std::exp(location_par));
		}

		inline double FirstDerivLogLikGamma(double y, double location_par) const {
			return (aux_pars_[0] * (y * std::exp(-location_par) - 1.));
		}

		inline double FirstDerivLogLikNegBin(int y, double location_par) const {
			const double mu = std::exp(location_par);
			return (y - (y + aux_pars_[0]) / (mu + aux_pars_[0]) * mu);
		}

		inline double FirstDerivLogLikNegBin1(int y, double location_par) const {
			const double mu = std::exp(location_par);
			const double r = mu / aux_pars_[0];
			const double C = GPBoost::digamma(y + r) - GPBoost::digamma(r) - std::log1p(aux_pars_[0]);
			return ((mu / aux_pars_[0]) * C);
		}

		inline double FirstDerivLogLikBeta(double y, double location_par) const {
			const double mu = GPBoost::sigmoid_stable_clamped(location_par);
			const double logit_y = std::log(y) - std::log1p(-y);
			const double dig1 = GPBoost::digamma((1.0 - mu) * aux_pars_[0]);
			const double dig2 = GPBoost::digamma(mu * aux_pars_[0]);
			return (aux_pars_[0] * mu * (1.0 - mu) * (dig1 - dig2 + logit_y));
		}

		inline double FirstDerivLogLikT(double y, double location_par) const {
			double res = (y - location_par);
			return (aux_pars_[1] + 1.) * res / (aux_pars_[1] * aux_pars_[0] * aux_pars_[0] + res * res);
		}

		inline double FirstDerivLogLikGaussian(double y, double location_par) const {
			return ((y - location_par) / aux_pars_[0]);
		}

		inline void FirstDerivLogLikGaussianHeteroscedastic(double y, double location_par, double location_par2,
			double& first_deriv_mean, double& first_deriv_log_var) const {
			double sigma2_inv = std::exp(-location_par2);
			double resid = y - location_par;
			first_deriv_mean = resid * sigma2_inv;
			first_deriv_log_var = (first_deriv_mean * resid - 1.) / 2.;
		}

		/*!
		* \brief First derivative of the heteroscedastic Gaussian log-likelihood wrt the mean only (location_par). Used for
		*		'gaussian_heteroscedastic' where the log-error variance (location_par2) is a fixed effect and thus not part of the mode / random effect
		*/
		inline double FirstDerivLogLikGaussianHeteroscedasticMean(double y, double location_par, double location_par2) const {
			return ((y - location_par) * std::exp(-location_par2));
		}

		inline double FirstDerivLogLikLogNormal(double y, double location_par) const {
			const double s2 = aux_pars_[0];
			const double z = std::log(y) - (location_par - 0.5 * s2);
			return z / s2;
		}

		inline double FirstDerivLogLikBetaBinomial(double y_ratio, double location_par, double w) {
			if (w <= 0.0) return 0.0;
			const double mu = GPBoost::sigmoid_stable_clamped(location_par);
			const double phi_raw = aux_pars_[0];
			const double phi = (phi_raw > 0.0 && std::isfinite(phi_raw)) ? phi_raw : 1e-16;
			const double a = mu * phi;
			const double b = (1.0 - mu) * phi;
			const double k = y_ratio * w;
			const double Delta = GPBoost::digamma(k + a) - GPBoost::digamma(a)
				- GPBoost::digamma(w - k + b) + GPBoost::digamma(b);
			const double s = mu * (1.0 - mu);
			return phi * s * Delta;
		}

		inline double FirstDerivLogLikGammaZeroInflated(double y, double location_par) const {
			if (y <= 0.) return 0.;
			return aux_pars_[0] * (y * std::exp(-location_par) - 1.);
		}

		inline double FirstDerivLogLikZeroCensPowNorm(double y, double location_par) const {
			const double sigma = aux_pars_[0];
			if (y <= 0.0) {
				// d/dmu log Phi(-mu/s) = -(1/sigma) * phi(a0)/Phi(a0)
				const double a0 = -location_par / sigma;
				return -(1.0 / sigma) * GPBoost::InvMillsRatioNormalPhi(a0);
			}
			else {
				const double lambda = aux_pars_[1];
				const double u = std::exp((1.0 / lambda) * std::log(y));
				const double z = (u - location_par) / sigma;
				return z / sigma;// d/dmu [-0.5*z^2] = z/sigma
			}
		}

		inline double FirstDerivLogLikZeroOneCensTransfNorm(double y, double location_par) const {
			const double sigma = aux_pars_[0];
			if (y <= 0.0) {
				const double a0 = -location_par / sigma;
				return -(1.0 / sigma) * GPBoost::InvMillsRatioNormalPhi(a0);
			}
			else if (y >= 1.0) {
				const double v = (1.0 - location_par) / sigma;
				return (1.0 / sigma) * GPBoost::InvMillsRatioNormalOneMinusPhi(v);
			}
			else {
				const double a = aux_pars_original_[1];
				const double b = aux_pars_[2];
				const double s_arg = (GPBoost::logit(y) - a) / b;
				const double x = GPBoost::sigmoid_stable(s_arg);
				const double z = (x - location_par) / sigma;
				return z / sigma;
			}
		}

		inline double FirstDerivLogLikZeroOneCensTransfBeta(double y, double location_par) const {
			const double phi = std::max(aux_pars_[0], 1e-12);
			const double u = std::max(aux_pars_[1], 1e-12);
			return FirstDerivLogLikZeroOneCensTransfBeta_at(y, location_par, phi, u);
		}
		inline double FirstDerivLogLikZeroOneCensTransfBeta_at(double y, double location_par,
			double phi, double u) const {
			const double eps_mu = 1e-12;
			const double eps_ab = 1e-12;
			const double eps_t = 1e-15;
			const double eps_u = 1e-12;
			const double tinyP = 1e-300;
			if (TwoNumbersAreEqual(y, 0.0) || TwoNumbersAreEqual(y, 1.0)) {
				const double h = 1e-6 * std::max(1.0, std::abs(location_par));
				const double eta_m = location_par - h, eta_p = location_par + h;
				// Use probabilities (not log-likelihood for robustness): P = BetaCDF(t0;a,b) if y=0, and P = 1 - BetaCDF(t1;a,b) if y=1.
				auto prob_at = [&](double eta_arg) {
					const double mu = GPBoost::sigmoid_stable_clamped(eta_arg);
					const double a = std::max(mu * phi, eps_ab);
					const double b = std::max((1.0 - mu) * phi, eps_ab);
					const double uu = std::max(u, eps_u);
					const double onep2u = 1.0 + 2.0 * uu;
					if (TwoNumbersAreEqual(y, 0.0)) {
						const double t0 = std::min(std::max(uu / onep2u, eps_t), 1.0 - eps_t);
						const double llP = GPBoost::log_beta_cdf(t0, a, b);
						return std::exp(std::max(llP, std::log(tinyP)));
					}
					else { // y == 1.0
						const double t1 = std::min(std::max((1.0 + uu) / onep2u, eps_t), 1.0 - eps_t);
						const double llQ = GPBoost::log1m_beta_cdf(t1, a, b);
						return std::exp(std::max(llQ, std::log(tinyP)));
					}
				};
				const double Pm = prob_at(eta_m);
				const double P0 = std::max(prob_at(location_par), tinyP);
				const double Pp = prob_at(eta_p);
				const double dP_deta = (Pp - Pm) / (2.0 * h);
				const double score = dP_deta / P0;
				return std::isfinite(score) ? score : 0.0;
			}
			else { // Interior: analytic score
				const double mu = std::min(std::max(GPBoost::sigmoid_stable_clamped(location_par), eps_mu), 1.0 - eps_mu);
				const double s = mu * (1.0 - mu);
				const double a = std::max(mu * phi, eps_ab);
				const double b = std::max((1.0 - mu) * phi, eps_ab);
				const double c = 1.0 + 2.0 * std::max(u, eps_u);
				double t = (y + u) / c;
				t = std::min(std::max(t, eps_t), 1.0 - eps_t);
				const double G = std::log(t) - std::log1p(-t) - GPBoost::digamma(a) + GPBoost::digamma(b);
				const double score = phi * s * G;
				return std::isfinite(score) ? score : 0.0;
			}
		}

		inline double FirstDerivLogLikZeroOneCensGamma(const double y, const double location_par) const {
			return FirstDerivLogLikZeroOneCensGamma_at(y, location_par, aux_pars_[0], aux_pars_[1]);
		}
		inline double FirstDerivLogLikZeroOneCensGamma_at(const double y, const double location_par, const double k, const double xi) const {
			const double mu = std::exp(location_par);
			const double th = mu / k;
			if (y <= 0.0) {
				if (xi <= 0.0) return 0.0;
				const double x = xi / th; // = k*xi/mu
				const double G = GPBoost::RegLowerGamma(k, x);
				if (G <= 0.0) return 0.0;
				const double p = std::exp(-x + (k - 1.0) * std::log(x) - std::lgamma(k)); // Gamma(k,1) pdf
				return -x * (p / G);
			}
			else if (y >= 1.0) {
				const double a = 1.0 + xi;
				const double x = a / th; // = k*(1+xi)/mu
				const double G = GPBoost::RegLowerGamma(k, x);
				const double H = 1.0 - G;
				if (H <= 0.0) return 0.0;
				const double p = std::exp(-x + (k - 1.0) * std::log(x) - std::lgamma(k));
				return  x * (p / H);
			}
			else {
				const double z = y + xi;
				return z / th - k; // interior score (k*z/mu - k)
			}
		}
		//alternative version, potentially more stable
		//inline double FirstDerivLogLikZeroOneCensGamma_at(
		//  const double y,
		//  const double location_par,
		//  const double k,
		//  const double xi) const {
		//  const double tiny = 1e-300;
		//  const double maxAbsGrad = 1e12;
		//  if (!(k > 0.0) || !std::isfinite(k)) return 0.0;      
		//  const double eta = std::max(std::min(location_par, 700.0), -700.0);// Safe exp(-eta)
		//  const double inv_mu = std::exp(-eta);
		//  auto clipGrad = [&](double g) {
		//    if (!std::isfinite(g)) return 0.0;
		//    if (g > maxAbsGrad) return  maxAbsGrad;
		//    if (g < -maxAbsGrad) return -maxAbsGrad;
		//    return g;
		//  };
		//  if (y <= 0.0) {
		//    if (xi <= 0.0) return 0.0;
		//    const double x = k * xi * inv_mu; // = k*xi/mu
		//    if (!(x > 0.0) || !std::isfinite(x)) return 0.0;
		//    double G = GPBoost::RegLowerGamma(k, x);
		//    if (!std::isfinite(G)) return 0.0;
		//    G = std::min(std::max(G, tiny), 1.0 - tiny);
		//    const double logG = std::log(G);
		//    const double logp = -x + (k - 1.0) * std::log(x) - std::lgamma(k); // log pdf Gamma(k,1)
		//    const double g = -x * std::exp(logp - logG);
		//    return clipGrad(g);
		//  }
		//  else if (y >= 1.0) {
		//    const double a = 1.0 + xi;
		//    if (!(a > 0.0)) return 0.0;
		//    const double x = k * a * inv_mu; // = k*(1+xi)/mu
		//    if (!(x > 0.0) || !std::isfinite(x)) return 0.0;
		//    double G = GPBoost::RegLowerGamma(k, x);
		//    if (!std::isfinite(G)) return 0.0;
		//    G = std::min(std::max(G, tiny), 1.0 - tiny);
		//    const double tail = std::max(1.0 - G, tiny);
		//    const double logTail = std::log(tail);
		//    const double logp = -x + (k - 1.0) * std::log(x) - std::lgamma(k);
		//    const double g = x * std::exp(logp - logTail);
		//    return clipGrad(g);
		//  }
		//  else {
		//    const double z = y + xi;
		//    if (!(z > 0.0)) return 0.0;       
		//    const double g = -k + (k * z * inv_mu);// g = -k + z/theta = -k + k*z/mu = -k + k*z*exp(-eta)
		//    return clipGrad(g);
		//  }
		//}   

		inline double FirstDerivLogLikAsymLaplace(double y, double location_par) const {
			double indicator = (y <= location_par) ? 1.0 : 0.0;//version without finite precision sub-gradient
			return ((quantile_ - indicator) / aux_pars_[0]);
			//Version with finite precision sub-gradient
			//const double resid = y - location_par;		
			//if (resid < -eps_sub_grad_scale_) {
			//	return (quantile_ - 1.) / aux_pars_[0];
			//}
			//else if (resid > eps_sub_grad_scale_) {
			//	return quantile_ / aux_pars_[0];
			//}
			//else {
			//	return (quantile_ - 0.5) / aux_pars_[0];
			//}
		}

		/*!
		* \brief Calculate the information, i.e., either (i) the Hessian of the negative log-likelihood, (ii) the Fisher information (=expected Hessian), or (iii) sometimes an approximate quasi-Fisher information.
		*			This is usually the second derivative of the negative log-likelihood with respect to the location parameter, i.e., the observed FI.
		*			This is usually a diagonal matrix and only its diagonal part is calculated.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param called_during_mode_finding Indicates whether this function is called during the mode finding algorithm or after the mode is found for the final approximation
		*/
		void CalcInformationLogLik(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			bool called_during_mode_finding) {
			if (use_random_effects_indices_of_data_) {
				CalcInformationLogLik_PerSample(y_data, y_data_int, location_par, called_during_mode_finding, information_ll_data_scale_, off_diag_information_ll_data_scale_);
				ReduceToModeScale(information_ll_data_scale_, information_ll_);
				if (information_has_off_diagonal_) {
					CalcZtVGivenIndices(num_data_, dim_mode_per_set_re_, random_effects_indices_of_data_, off_diag_information_ll_data_scale_.data(), off_diag_information_ll_.data(), true);
				}
			}
			else {// !use_random_effects_indices_of_data_
				CalcInformationLogLik_PerSample(y_data, y_data_int, location_par, called_during_mode_finding, information_ll_, off_diag_information_ll_);
			}
			if (information_has_off_diagonal_) {
				CHECK(num_sets_re_ == 2);
				information_ll_mat_ = sp_mat_t(dim_mode_, dim_mode_);
				std::vector<Triplet_t> triplets(dim_mode_per_set_re_ * 4);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < dim_mode_; ++i) {
					triplets[i] = Triplet_t(i, i, information_ll_[i]);
				}
#pragma omp parallel for schedule(static)
				for (int i = 0; i < dim_mode_per_set_re_; ++i) {
					triplets[dim_mode_ + i] = Triplet_t(i, i + dim_mode_per_set_re_, off_diag_information_ll_[i]);
					triplets[dim_mode_ + dim_mode_per_set_re_ + i] = Triplet_t(i + dim_mode_per_set_re_, i, off_diag_information_ll_[i]);
				}
				information_ll_mat_.setFromTriplets(triplets.begin(), triplets.end());
			}
			if (diag_information_variance_correction_for_prediction_) {
				//Log::REInfo("before correction: information_ll_[0:2] = %g, %g, %g, first_deriv_ll_[0:2] = %g, %g, %g ", 
				//	information_ll_[0], information_ll_[1], information_ll_[2], first_deriv_ll_[0], first_deriv_ll_[1], first_deriv_ll_[2]);//for debugging
				if (var_cor_pred_version_ == "freq_asymptotic") {
					if (likelihood_type_ == "asymmetric_laplace") {
						if (!use_random_effects_indices_of_data_) {
							const double FI = FisherInformationOneSampleAsymLaplace();
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
							for (data_size_t i = 0; i < num_data_; ++i) {
								//information_ll_[i] = information_ll_[i] * information_ll_[i] / first_deriv_ll_[i] / first_deriv_ll_[i];//using an empirical Fisher information based on one sample
								information_ll_[i] = information_ll_[i] * information_ll_[i] / FI;
							}
						}
						else {
							const double FI = FisherInformationOneSampleAsymLaplace();
							vec_t FI_data(num_data_);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
							for (data_size_t i = 0; i < num_data_; ++i) {
								//information_ll_data_scale_[i] = information_ll_data_scale_[i] * information_ll_data_scale_[i] / first_deriv_ll_data_scale_[i] / first_deriv_ll_data_scale_[i];//using an empirical Fisher information based on one sample
								information_ll_data_scale_[i] = information_ll_data_scale_[i];
								FI_data[i] = FI;
							}
							vec_t FI_mode(dim_mode_per_set_re_);
							CalcZtVGivenIndices(num_data_, dim_mode_per_set_re_, random_effects_indices_of_data_, information_ll_data_scale_.data(), information_ll_.data(), true);
							CalcZtVGivenIndices(num_data_, dim_mode_per_set_re_, random_effects_indices_of_data_, FI_data.data(), FI_mode.data(), true);
							information_ll_ = (information_ll_.array().square() / FI_mode.array()).matrix();
						}
					}
					else {
						Log::REFatal("var_cor_pred_version_ = 'freq_asymptotic' not implemented for this likelihood ");
					}
				}//end var_cor_pred_version_ == "freq_asymptotic"
				else if (var_cor_pred_version_ == "learning_rate") {
					if (!use_random_effects_indices_of_data_) {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							information_ll_[i] = information_ll_[i] * likelihood_learning_rate_;
						}
					}
					else {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							information_ll_data_scale_[i] = information_ll_data_scale_[i] * likelihood_learning_rate_;
						}
						CalcZtVGivenIndices(num_data_, dim_mode_per_set_re_, random_effects_indices_of_data_, information_ll_data_scale_.data(), information_ll_.data(), true);
					}
				}//end var_cor_pred_version_ == "learning_rate"
				//Log::REInfo("after correction: information_ll_[0:2] = %g, %g, %g ", information_ll_[0], information_ll_[1], information_ll_[2]);//for debugging
			}//end diag_information_variance_correction_for_prediction_
		}//end CalcInformationLogLik

		/*!
		* \brief Dispatch on 'likelihood_type_' and call 'visit' with a kernel that evaluates the OBSERVED information
		*			(the second derivative of the negative log-likelihood with respect to the location parameter) of one sample
		*			(without the sample weight). This is the single place where the mapping from a likelihood to its observed
		*			information formula is defined; it is shared by the calculation over all samples
		*			('CalcInformationLogLik_PerSample') and by the single-sample version ('CalcDiagInformationLogLikOneSample').
		*			Only likelihoods with a single location parameter block whose information is a plain function of (y, eta)
		*			are covered here; the remaining ones are handled by the callers
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param visit Callable that is invoked with the selected kernel (a callable that takes a sample index)
		* \return True if the current likelihood is covered here, false otherwise
		*/
		template <class Visitor>
		bool VisitObservedInformationKernel(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			Visitor visit) const {
			if (likelihood_type_ == "bernoulli_probit") visit([&](data_size_t i) { return SecondDerivNegLogLikBernoulliProbit(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "bernoulli_logit") visit([&](data_size_t i) { return SecondDerivNegLogLikBernoulliLogit(location_par[i]); });
			else if (likelihood_type_ == "poisson") visit([&](data_size_t i) { return SecondDerivNegLogLikPoisson(location_par[i]); });
			else if (likelihood_type_ == "gamma") visit([&](data_size_t i) { return SecondDerivNegLogLikGamma(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") visit([&](data_size_t i) { return InformationLogLikTweedie(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "negative_binomial") visit([&](data_size_t i) { return SecondDerivNegLogLikNegBin(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "negative_binomial_1") visit([&](data_size_t i) { return SecondDerivNegLogLikNegBin1(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "zero_inflated_poisson") visit([&](data_size_t i) { return SecondDerivNegLogLikZeroInflatedPoisson(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") visit([&](data_size_t i) { return SecondDerivNegLogLikZeroInflatedNegBinFamily(y_data_int[i], location_par[i]); });
			else if (likelihood_type_ == "beta") visit([&](data_size_t i) { return SecondDerivNegLogLikBeta(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "t") visit([&](data_size_t i) { return SecondDerivNegLogLikT(y_data[i], location_par[i]); });
			else if (IsGaussianLikelihood()) visit([FI = SecondDerivNegLogLikGaussian()](data_size_t) { return FI; });
			else if (likelihood_type_ == "lognormal") visit([FI = SecondDerivNegLogLikLogNormal()](data_size_t) { return FI; });
			else if (likelihood_type_ == "hurdle_gamma") visit([&](data_size_t i) { return SecondDerivNegLogLikGammaZeroInflated(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "hurdle_lognormal") visit([&](data_size_t i) { return SecondDerivNegLogLikLogNormalZeroInflated(y_data[i]); });
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") visit([&](data_size_t i) { return SecondDerivNegLogLikZeroCensPowNorm(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "zoctn") visit([&](data_size_t i) { return SecondDerivNegLogLikZeroOneCensTransfNorm(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "zero_one_censored_transformed_beta") visit([&](data_size_t i) { return SecondDerivNegLogLikZeroOneCensTransfBeta(y_data[i], location_par[i]); });
			else if (likelihood_type_ == "zero_one_censored_shifted_gamma") visit([&](data_size_t i) { return SecondDerivNegLogLikZeroOneCensGamma(y_data[i], location_par[i]); });
			else return false;
			return true;
		}//end VisitObservedInformationKernel

		/*!
		* \brief Dispatch on 'likelihood_type_' and call 'visit' with a kernel that evaluates the FISHER (expected) information
		*			with respect to the location parameter of one sample (without the sample weight). Shared by
		*			'CalcInformationLogLik_PerSample' and 'CalcDiagInformationLogLikOneSample'. Only likelihoods with a single
		*			location parameter block are covered here; the remaining ones are handled by the callers
		* \param location_par Location parameter (random plus fixed effects)
		* \param visit Callable that is invoked with the selected kernel (a callable that takes a sample index)
		* \return True if the current likelihood is covered here, false otherwise
		*/
		template <class Visitor>
		bool VisitFisherInformationKernel(const double* location_par,
			Visitor visit) const {
			if (likelihood_type_ == "bernoulli_logit") visit([&](data_size_t i) { return SecondDerivNegLogLikBernoulliLogit(location_par[i]); });
			else if (likelihood_type_ == "poisson") visit([&](data_size_t i) { return SecondDerivNegLogLikPoisson(location_par[i]); });
			else if (likelihood_type_ == "t") visit([FI = FisherInformationT()](data_size_t) { return FI; });
			else if (IsGaussianLikelihood()) visit([FI = SecondDerivNegLogLikGaussian()](data_size_t) { return FI; });
			else if (likelihood_type_ == "lognormal") visit([FI = SecondDerivNegLogLikLogNormal()](data_size_t) { return FI; });
			else if (likelihood_type_ == "asymmetric_laplace") visit([FI = FisherInformationOneSampleAsymLaplace()](data_size_t) { return FI; });
			else return false;
			return true;
		}//end VisitFisherInformationKernel

		/*!
		* Calculate the information per sample., i.e., either (i) the Hessian of the negative log-likelihood, (ii) the Fisher information (=expected Hessian), or (iii) sometimes an approximate quasi-Fisher information.
		*			This is usually the second derivative of the negative log-likelihood with respect to the location parameter, i.e., the observed FI.
		*			This is usually a diagonal matrix and only its diagonal part is calculated.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param called_during_mode_finding Indicates whether this function is called during the mode finding algorithm or after the mode is found for the final approximation
		* \param[out] information_ll Diagonal of information
		* \param[out] off_diag_information_ll Off-diagonal of information (if applicable)
		*/
		void CalcInformationLogLik_PerSample(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			bool called_during_mode_finding,
			vec_t& information_ll,
			vec_t& off_diag_information_ll) {
			string_t approximation_type_local;
			if (use_fisher_for_mode_finding_ && called_during_mode_finding) {
				approximation_type_local = "fisher_laplace";
			}
			else {
				approximation_type_local = approximation_type_;
			}
			if (approximation_type_local == "laplace") {
				if (IsEGPDLikelihood() || IsHurdleEGPD()) {
					const bool hurdle = IsHurdleEGPD();
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.;
						if (w == 0. || (hurdle && y_data[i] <= 0.)) information_ll[i] = 0.;
						else {
							const auto result = EvaluateEGPD(y_data[i], location_par[i]);
							information_ll[i] = result.status == EGPDEvalStatus::kValid ? -w * result.d2_eta : std::numeric_limits<double>::quiet_NaN();
						}
					}
				}
				else if (likelihood_type_ == "binomial_logit" || likelihood_type_ == "quasi_bernoulli_logit") {
					ForEachSampleWeighted(information_ll, [&](data_size_t i) { return SecondDerivNegLogLikBernoulliLogit(location_par[i]); });
				}
				else if (likelihood_type_ == "binomial_probit" || likelihood_type_ == "quasi_bernoulli_probit") {
					ForEachSampleWeighted(information_ll, [&](data_size_t i) { return SecondDerivNegLogLikBinomialProbit(y_data[i], location_par[i]); });
				}
				else if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						SecondDerivNegLogLikGaussianHeteroscedastic(y_data[i], location_par[i], location_par[i + num_data_],
							information_ll[i], information_ll[i + num_data_], off_diag_information_ll[i]);
						if (has_weights_) {
							information_ll[i] *= weights_[i];
							information_ll[i + num_data_] *= weights_[i];
							off_diag_information_ll[i] *= weights_[i];
						}
					}
				}
				else if (likelihood_type_ == "beta_binomial") {
					CHECK(has_weights_);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						information_ll[i] = SecondDerivNegLogLikBetaBinomial(y_data[i], location_par[i], weights_[i]);
					}
				}
				else if (IsHurdleRegression()) {
					// Block-0 (eta) observed information used for the Laplace mode / random effects
					ForEachSampleWeighted(information_ll, [&](data_size_t i) { return HurdleRegression_Jeta(y_data[i], location_par[i]); });
				}
				else if (IsZeroInflatedCountRegression()) {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t iz = 0; iz < num_data_; ++iz) {
						const double wz = has_weights_ ? weights_[iz] : 1.0;
						ZICountRegQuant o; ZICountRegressionQuantities(y_data_int[iz], location_par[iz], location_par[iz + num_data_], o);
						information_ll[iz] = wz * o.Jeta;
					}
				}
				else if (IsZeroCensPowNormHetero()) {
					ForEachSampleWeighted(information_ll, [&](data_size_t i) { return SecondDerivNegLogLikZeroCensPowNormHetero(y_data[i], location_par[i], location_par[i + num_data_]); });
				}
				else if (!VisitObservedInformationKernel(y_data, y_data_int, location_par,
					[&](auto kernel) { ForEachSampleWeighted(information_ll, kernel); })) {
					NotSupportedForLikelihood(__func__);
				}
			}//end approximation_type_local == "laplace"
			else if (approximation_type_local == "fisher_laplace") {
				if (likelihood_type_ == "binomial_logit") {
					ForEachSampleWeighted(information_ll, [&](data_size_t i) { return SecondDerivNegLogLikBernoulliLogit(location_par[i]); });
				}
				else if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						FisherInformationGaussianHeteroscedastic(location_par[i + num_data_], information_ll[i], information_ll[i + num_data_]);
						if (has_weights_) {
							information_ll[i] *= weights_[i];
							information_ll[i + num_data_] *= weights_[i];
						}
					}
				}
				else if (likelihood_type_ == "gaussian_heteroscedastic") {
					// Only the mean is a mode / random effect here; the log-error variance (location_par[i + num_data_]) is a fixed effect
					ForEachSampleWeighted(information_ll, [&](data_size_t i) { return FisherInformationGaussianHeteroscedasticMean(location_par[i + num_data_]); });
				}
				else if (IsZeroInflatedCount()) {
					// Fisher (expected) information wrt eta of a zero-inflated count (does not depend on y -> nonnegative).
					const int kind = ZICountBaseKind();
					const bool reg = IsZeroInflatedCountRegression();
					const double pi_const = reg ? 0. : ZICountConstantP0();
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double mu = std::exp(location_par[i]);
						const double pi = reg ? GPBoost::sigmoid_stable(location_par[i + num_data_]) : pi_const;
						ZICountZeroMass z; FillZeroMassZICountKind(mu, z, kind);
						information_ll[i] = w * ZICountFisherInfoEta(mu, pi, z, kind);
					}
				}
				else if (likelihood_type_ == "negative_binomial_1") {
					// Base NB1: QUASI-Fisher information mu/(1+phi) (no closed-form exact Fisher wrt eta). Used for mode finding only;
					// the determinant/gradient uses the observed Hessian (approximation_type_ = "laplace").
					ForEachSampleWeighted(information_ll, [&](data_size_t i) { return std::exp(location_par[i]) / (1. + aux_pars_[0]); });
				}
				else if (!VisitFisherInformationKernel(location_par,
					[&](auto kernel) { ForEachSampleWeighted(information_ll, kernel); })) {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_local);
				}
			}//end approximation_type_local == "fisher_laplace"
			else if (approximation_type_local == "triangular_kernel_curvature") {
				if (likelihood_type_ == "asymmetric_laplace") {
					FindDeltaMode_TKC_Approx(y_data, y_data_int, location_par);
					double neg_curvature = NegativeHessian_TKC_Approx_AsymLaplace(delta_location_par_);
					ForEachSampleWeighted(information_ll, [&](data_size_t) { return neg_curvature; });
				}
				else {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_local);
				}
			}//end approximation_type_local == "triangular_kernel_curvature"
			else {
				Log::REFatal("CalcInformationLogLik_PerSample: approximation_type '%s' is not supported ", approximation_type_local.c_str());
			}
		}// end CalcInformationLogLik_PerSample

		/*!
		* \brief Calculate the diagonal of the Fisher information (=usually the second derivative of the negative (!) log-likelihood with respect to the location parameter, i.e., the observed FI)
		*			Note: this is only used for 'TestNegLogLikelihoodAdaptiveGHQuadrature()'
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		inline double CalcDiagInformationLogLikOneSample(double y_data,
			int y_data_int,
			double location_par) const {
			if (approximation_type_ == "laplace") {
				if (NotImplementedForOneSample()) {
					FatalOneSampleNotImplemented(__func__);
					return(0.);
				}
				if (IsEGPDLikelihood() || IsHurdleEGPD()) {
					if (IsHurdleEGPD() && y_data <= 0.) return 0.;
					const auto result = EvaluateEGPD(y_data, location_par);
					return result.status == EGPDEvalStatus::kValid ? -result.d2_eta : std::numeric_limits<double>::quiet_NaN();
				}
				double information = 1.;
				if (!VisitObservedInformationKernel(&y_data, &y_data_int, &location_par,
					[&](auto kernel) { information = kernel(0); })) {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_);
				}
				return(information);
			}//end approximation_type_ == "laplace"
			else if (approximation_type_ == "fisher_laplace") {
				double information = 1.;
				if (!VisitFisherInformationKernel(&location_par,
					[&](auto kernel) { information = kernel(0); })) {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_);
				}
				return(information);
			}//end approximation_type_ == "fisher_laplace"
			else {
				Log::REFatal("CalcDiagInformationLogLikOneSample: approximation_type '%s' is not supported ", approximation_type_.c_str());
				return(1.);
			}
		}// end CalcDiagInformationLogLikOneSample

		inline double SecondDerivNegLogLikBernoulliProbit(int y, double location_par) const {
			if (y == 0) {
				const double pdf_div_omcdf = GPBoost::InvMillsRatioNormalOneMinusPhi(location_par);//phi(x) / (1 - Phi(x))
				return -pdf_div_omcdf * (location_par - pdf_div_omcdf);
			}
			else {
				const double pdf_div_cdf = GPBoost::InvMillsRatioNormalPhi(location_par);//phi(x) / Phi(x)
				return pdf_div_cdf * (location_par + pdf_div_cdf);
			}
		}

		inline double SecondDerivNegLogLikBinomialProbit(double y, double location_par) const {
			if (y == 0.0) {
				const double pdf_div_omcdf = GPBoost::InvMillsRatioNormalOneMinusPhi(location_par);//phi(x) / (1 - Phi(x))
				return -pdf_div_omcdf * (location_par - pdf_div_omcdf);
			}
			if (y == 1.0) {
				const double pdf_div_cdf = GPBoost::InvMillsRatioNormalPhi(location_par);//phi(x) / Phi(x)
				return pdf_div_cdf * (location_par + pdf_div_cdf);
			}
			const double pdf_div_cdf = GPBoost::InvMillsRatioNormalPhi(location_par);//phi(x) / Phi(x)
			const double pdf_div_omcdf = GPBoost::InvMillsRatioNormalOneMinusPhi(location_par);//phi(x) / (1 - Phi(x))
			return y * pdf_div_cdf * (location_par + pdf_div_cdf) + (1.0 - y) * -pdf_div_omcdf * (location_par - pdf_div_omcdf);
		}

		inline double SecondDerivNegLogLikBernoulliLogit(double location_par) const {
			const double p = GPBoost::sigmoid_stable(location_par);
			return p * (1.0 - p);
			// Alternatice version (less numerically stable)
			//const double exp_loc_i = std::exp(location_par);
			//return (exp_loc_i / ((1. + exp_loc_i) * (1. + exp_loc_i)));
		}

		inline double SecondDerivNegLogLikPoisson(double location_par) const {
			return std::exp(location_par);
		}

		inline double SecondDerivNegLogLikGamma(double y, double location_par) const {
			return (aux_pars_[0] * y * std::exp(-location_par));
		}

		inline double SecondDerivNegLogLikNegBin(int y, double location_par) const {
			const double mu = std::exp(location_par);
			const double mu_plus_r = mu + aux_pars_[0];
			return ((y + aux_pars_[0]) * mu * aux_pars_[0] / (mu_plus_r * mu_plus_r));
		}

		inline double SecondDerivNegLogLikNegBin1(int y, double location_par) const {
			const double mu = std::exp(location_par);
			const double r = mu / aux_pars_[0];
			const double C = GPBoost::digamma(y + r) - GPBoost::digamma(r) - std::log1p(aux_pars_[0]);
			return (-(mu * mu) / (aux_pars_[0] * aux_pars_[0]) * (GPBoost::trigamma(y + r) - GPBoost::trigamma(r)) - (mu / aux_pars_[0]) * C);
		}

		inline double SecondDerivNegLogLikBeta(double y, double location_par) const {
			const double mu = GPBoost::sigmoid_stable_clamped(location_par);
			const double logit_y = std::log(y) - std::log1p(-y);
			const double dig1 = GPBoost::digamma((1.0 - mu) * aux_pars_[0]);
			const double dig2 = GPBoost::digamma(mu * aux_pars_[0]);
			const double tri1 = GPBoost::trigamma((1.0 - mu) * aux_pars_[0]);
			const double tri2 = GPBoost::trigamma(mu * aux_pars_[0]);
			const double h1 = -aux_pars_[0] * aux_pars_[0] * mu * mu * (1.0 - mu) * (1.0 - mu) * (tri1 + tri2);
			const double h2 = aux_pars_[0] * mu * (1.0 - mu) * (1.0 - 2.0 * mu) * (dig1 - dig2 + logit_y);
			return -(h1 + h2);
		}

		inline double SecondDerivNegLogLikGaussian() const {
			return (1 / aux_pars_[0]);
		}

		inline double SecondDerivNegLogLikT(double y, double location_par) const {
			const double res_sq = (y - location_par) * (y - location_par);
			const double nu_sigma2 = aux_pars_[1] * aux_pars_[0] * aux_pars_[0];
			return (-(aux_pars_[1] + 1.) * (res_sq - nu_sigma2) / ((nu_sigma2 + res_sq) * (nu_sigma2 + res_sq)));
		}

		inline double FisherInformationT() const {
			return ((aux_pars_[1] + 1.) / (aux_pars_[1] + 3.) / (aux_pars_[0] * aux_pars_[0]));
		}

		inline void SecondDerivNegLogLikGaussianHeteroscedastic(double y, double location_par, double location_par2,
			double& second_deriv, double& second_deriv2, double& off_diag_second_deriv) const {
			const double sigma2_inv = std::exp(-location_par2);
			const double resid = y - location_par;
			second_deriv = sigma2_inv;
			second_deriv2 = resid * resid * sigma2_inv / 2.;
			off_diag_second_deriv = resid * sigma2_inv;
		}

		inline void FisherInformationGaussianHeteroscedastic(const double location_par_var, double& second_deriv, double& second_deriv2) const {
			second_deriv = std::exp(-location_par_var);
			second_deriv2 = 1 / 2.;
		}

		/*!
		* \brief Fisher information of the heteroscedastic Gaussian log-likelihood wrt the mean only (location_par). Used for
		*		'gaussian_heteroscedastic' where the log-error variance (location_par_var) is a fixed effect and thus not part of the mode / random effect
		*/
		inline double FisherInformationGaussianHeteroscedasticMean(const double location_par_var) const {
			return (std::exp(-location_par_var));
		}

		inline double SecondDerivNegLogLikLogNormal() const {
			return 1 / aux_pars_[0];
		}

		inline double SecondDerivNegLogLikBetaBinomial(double y_ratio, double location_par, double w) {
			if (w <= 0.0) return 0.0;
			const double mu = GPBoost::sigmoid_stable_clamped(location_par);
			const double phi_raw = aux_pars_[0];
			const double phi = (phi_raw > 0.0 && std::isfinite(phi_raw)) ? phi_raw : 1e-16;
			const double a = mu * phi;
			const double b = (1.0 - mu) * phi;
			const double k = y_ratio * w;
			const double Delta = GPBoost::digamma(k + a) - GPBoost::digamma(a)
				- GPBoost::digamma(w - k + b) + GPBoost::digamma(b);
			const double S1 = GPBoost::trigamma(k + a) - GPBoost::trigamma(a);
			const double S2 = GPBoost::trigamma(w - k + b) - GPBoost::trigamma(b);
			const double s = mu * (1.0 - mu);
			return -phi * (s * (1.0 - 2.0 * mu) * Delta + phi * s * s * (S1 + S2));
		}

		inline double SecondDerivNegLogLikGammaZeroInflated(double y, double location_par) const {
			if (y <= 0.) return 0.;
			return aux_pars_[0] * y * std::exp(-location_par);
		}

		inline double SecondDerivNegLogLikZeroCensPowNorm(double y, double location_par) const {
			const double sigma = aux_pars_[0];
			if (y <= 0.0) {
				// Negative second derivative for zeros = - d^2/dmu^2 log Phi(-mu/sigma) = (1/sigma^2) * r * (a0 + r),  r = phi(a0)/Phi(a0), a0 = -mu/sigma
				const double a0 = -location_par / sigma;
				const double r = GPBoost::InvMillsRatioNormalPhi(a0);
				return (1.0 / (sigma * sigma)) * r * (a0 + r);
			}
			else {
				return 1.0 / (sigma * sigma);
			}
		}

		inline double SecondDerivNegLogLikZeroOneCensTransfNorm(double y, double location_par) const {
			const double sigma = aux_pars_[0];
			if (y <= 0.0) {
				const double a0 = -location_par / sigma;
				const double r = GPBoost::InvMillsRatioNormalPhi(a0);
				return (1.0 / (sigma * sigma)) * r * (a0 + r);
			}
			else if (y >= 1.0) {
				const double v = (1.0 - location_par) / sigma;
				const double r2 = GPBoost::InvMillsRatioNormalOneMinusPhi(v);
				return (1.0 / (sigma * sigma)) * r2 * (r2 - v);
			}
			else {
				return 1.0 / (sigma * sigma);
			}
		}

		inline double SecondDerivNegLogLikZeroOneCensTransfBeta(double y, double location_par) const {
			const double phi = std::max(aux_pars_[0], 1e-12);
			const double u = std::max(aux_pars_[1], 1e-12);
			return SecondDerivNegLogLikZeroOneCensTransfBeta_at(y, location_par, phi, u);
		}
		inline double SecondDerivNegLogLikZeroOneCensTransfBeta_at(double y, double location_par,
			double phi, double u) const {
			const double eps_mu = 1e-12;
			const double eps_ab = 1e-12;
			const double eps_t = 1e-15;
			const double eps_u = 1e-12;
			const double tinyP = 1e-300;
			if (TwoNumbersAreEqual(y, 0.0) || TwoNumbersAreEqual(y, 1.0)) {
				//Boundaries: build from P, P', P'' as  H = - ( P''/P - (P'/P)^2 )
				const double h = 1e-5 * std::max(1.0, std::abs(location_par));
				const double eta_m = location_par - h, eta_p = location_par + h;
				auto prob_at = [&](double eta_arg) {
					const double mu = GPBoost::sigmoid_stable_clamped(eta_arg);
					const double a = std::max(mu * phi, eps_ab);
					const double b = std::max((1.0 - mu) * phi, eps_ab);
					const double uu = std::max(u, eps_u);
					const double onep2u = 1.0 + 2.0 * uu;
					if (TwoNumbersAreEqual(y, 0.0)) {
						const double t0 = std::min(std::max(uu / onep2u, eps_t), 1.0 - eps_t);
						const double llP = GPBoost::log_beta_cdf(t0, a, b);
						return std::exp(std::max(llP, std::log(tinyP)));
					}
					else {
						const double t1 = std::min(std::max((1.0 + uu) / onep2u, eps_t), 1.0 - eps_t);
						const double llQ = GPBoost::log1m_beta_cdf(t1, a, b);
						return std::exp(std::max(llQ, std::log(tinyP)));
					}
				};
				const double Pm = prob_at(eta_m);
				const double P0 = std::max(prob_at(location_par), tinyP);
				const double Pp = prob_at(eta_p);
				const double dP = (Pp - Pm) / (2.0 * h);
				const double d2P = (Pp - 2.0 * P0 + Pm) / (h * h);
				const double H = -(d2P / P0 - (dP / P0) * (dP / P0));
				if (!std::isfinite(H) || H < 0.0) return 0.0;
				return H;
			}
			else {// Interior: analytic expression			
				const double mu = std::min(std::max(GPBoost::sigmoid_stable_clamped(location_par), eps_mu), 1.0 - eps_mu);
				const double s = mu * (1.0 - mu);
				const double sp = s * (1.0 - 2.0 * mu);
				const double a = std::max(mu * phi, eps_ab);
				const double b = std::max((1.0 - mu) * phi, eps_ab);
				const double c = 1.0 + 2.0 * std::max(u, eps_u);
				double t = (y + u) / c;
				t = std::min(std::max(t, eps_t), 1.0 - eps_t);
				const double G = std::log(t) - std::log1p(-t) - GPBoost::digamma(a) + GPBoost::digamma(b);
				const double psi1a = GPBoost::trigamma(a);
				const double psi1b = GPBoost::trigamma(b);
				const double G_eta = -phi * s * (psi1a + psi1b);
				const double H = -phi * (sp * G + s * G_eta);
				if (!std::isfinite(H) || H < 0.0) return 0.0;
				return H;
			}
		}

		inline double SecondDerivNegLogLikZeroOneCensGamma(const double y, const double location_par) const {
			return SecondDerivNegLogLikZeroOneCensGamma_at(y, location_par, aux_pars_[0], aux_pars_[1]);
		}
		inline double SecondDerivNegLogLikZeroOneCensGamma_at(const double y, const double location_par, const double k, const double xi) const {
			const double tiny = 1e-300;
			const double maxW = 1e12; // cap huge curvature to avoid Inf 
			if (!(k > 0.0) || !std::isfinite(k)) return 0.0;
			const double eta = std::max(std::min(location_par, 700.0), -700.0);
			const double inv_mu = std::exp(-eta);  // = 1/mu
			auto clipW = [&](double W) {
				if (!std::isfinite(W) || W < 0.0) return 0.0;
				if (W > maxW) return maxW;
				return W;
			};
			if (y <= 0.0) {
				if (xi <= 0.0) return 0.0;
				const double x = k * xi * inv_mu; // = k*xi/mu
				if (!(x > 0.0) || !std::isfinite(x)) return 0.0;
				double G = GPBoost::RegLowerGamma(k, x);
				if (!std::isfinite(G)) return 0.0;
				G = std::min(std::max(G, tiny), 1.0 - tiny);
				const double logG = std::log(G);
				const double logp = -x + (k - 1.0) * std::log(x) - std::lgamma(k); // log pdf Gamma(k,1)
				const double Q = std::exp(logp - logG);
				const double H = x * (x - k) * Q + x * x * Q * Q;
				return clipW(H);
			}
			else if (y >= 1.0) {
				const double a = 1.0 + xi;
				if (!(a > 0.0)) return 0.0;
				const double x = k * a * inv_mu; // = k*(1+xi)/mu
				if (!(x > 0.0) || !std::isfinite(x)) return 0.0;
				double G = GPBoost::RegLowerGamma(k, x);
				if (!std::isfinite(G)) return 0.0;
				G = std::min(std::max(G, tiny), 1.0 - tiny);
				const double tail = std::max(1.0 - G, tiny);// tail = 1 - G, in a numerically safe way
				const double logTail = std::log(tail);
				const double logp = -x + (k - 1.0) * std::log(x) - std::lgamma(k);
				const double Q = std::exp(logp - logTail);
				const double H = x * (k - x) * Q + x * x * Q * Q;
				return clipW(H);
			}
			else {
				const double z = y + xi;
				if (!(z > 0.0)) return 0.0;
				const double H = k * z * inv_mu; // = k*z/mu
				return clipW(H);
			}
		}

		inline double FisherInformationOneSampleAsymLaplace() const {
			return (quantile_ * (1. - quantile_) / (aux_pars_[0] * aux_pars_[0]));
		}

		/*!
		* \brief Approximate negative Hessian in triangular kernel curvature approximation
		* \param double delta_location_par Distance of +- delta_location_par around the mode at which the quadratic approximation should have the same log-likelihood difference as (delta_log_like_up_ + delta_log_like_down_) / 2.
		* \return Negative Hessian / curvature
		*/
		inline double NegativeHessian_TKC_Approx_AsymLaplace(double delta_location_par) const {
			CHECK(likelihood_type_ == "asymmetric_laplace");
			//double neg_curvature_up = 2. * (delta_log_like_up_ + sum_first_deriv_ * delta_location_par) / (num_data_ * delta_location_par * delta_location_par);
			//double neg_curvature_down = 2. * (delta_log_like_down_ - sum_first_deriv_ * delta_location_par) / (num_data_ * delta_location_par * delta_location_par);
			//double neg_curvature = (neg_curvature_down + neg_curvature_up) / 2.;
			double neg_curvature = (delta_log_like_up_ + delta_log_like_down_) / (num_data_ * delta_location_par * delta_location_par);
			if (neg_curvature < 1e-10) {//Note: this is either zero or positive, but due to finite precision arithmetic, it can be slightly below 0. sometimes
				neg_curvature = 1e-10;
			}
			return (neg_curvature);
		}

		/*!
		* \brief Auxiliary function for adding a constant to the location parameter = mode of random effects + fixed effects + delta_location_par
		* \param location_par_ptr Location parameter (random plus fixed effects)
		* \param delta_location_par
		* \param[out] location_par_delta Location parameter
		*/
		void AddConstantToLocationPar(const double* location_par_ptr,
			double delta_location_par,
			vec_t& location_par_delta) {
#pragma omp parallel for schedule(static)
			for (data_size_t i = 0; i < num_data_; ++i) {
				location_par_delta[i] = location_par_ptr[i] + delta_location_par;
			}
		}//end AddConstantToLocationPar

		/*!
		* \brief Auxiliary function for adding a vector to the location parameter = mode of random effects + fixed effects + delta_location_par
		* \param location_par_ptr Location parameter (random plus fixed effects)
		* \param delta_location_par
		* \param[out] location_par_delta Location parameter
		*/
		void AddToLocationPar(const double* location_par_ptr,
			const vec_t& delta_location_par,
			vec_t& location_par_delta) {
#pragma omp parallel for schedule(static)
			for (data_size_t i = 0; i < num_data_; ++i) {
				location_par_delta[i] = location_par_ptr[i] + delta_location_par[i];
			}
		}//end AddToLocationPar

		/*!
		* \brief Find delta_location_par_ for the triangual kernel curvature approximation
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par_ptr Location parameter (random plus fixed effects)
		*/
		void FindDeltaMode_TKC_Approx(const double* y_data,
			const int* y_data_int,
			const double* location_par_ptr) {
			CHECK(likelihood_type_ == "asymmetric_laplace");
			CHECK(approximation_type_ == "triangular_kernel_curvature" || approximation_type_ == "constant_curvature_manual");
			double ll_mode = LogLikelihood(y_data, y_data_int, location_par_ptr);
			int it = 0;
			if (const_delta_location_par_) {
				CHECK(approximation_type_ == "constant_curvature_manual");
				vec_t location_par_delta(num_data_);
				AddConstantToLocationPar(location_par_ptr, delta_location_par_, location_par_delta);
				delta_log_like_up_ = ll_mode - LogLikelihood(y_data, y_data_int, location_par_delta.data());
				AddConstantToLocationPar(location_par_ptr, -delta_location_par_, location_par_delta);
				delta_log_like_down_ = ll_mode - LogLikelihood(y_data, y_data_int, location_par_delta.data());
			}//end const_delta_location_par_
			else {//!const_delta_location_par_
				CHECK(approximation_type_ == "triangular_kernel_curvature");
				//find delta_location_par_ which minimizes the unexplained variance of the approximation
				delta_location_par_ = 1e-6;
				double lower_limit = 0., upper_limit = 0.1;
				// find upper limit
				double unexpl_variance_up_lim = GoodnessFit_TKC_approx(upper_limit, ll_mode, y_data, y_data_int, location_par_ptr, false);
				//Log::REDebug(" ");
				//Log::REDebug("finding upper limit: it = %d, upper_limit = %g, unexpl_variance_up_lim = %g", it, upper_limit, unexpl_variance_up_lim);//for debugging
				for (it = 0; it < 100; ++it) {
					upper_limit *= 2;
					double unexpl_variance_approx = GoodnessFit_TKC_approx(upper_limit, ll_mode, y_data, y_data_int, location_par_ptr, false);
					//Log::REDebug("it = %d, upper_limit = %g, unexpl_variance_up_lim = %g", it, upper_limit, unexpl_variance_approx);//for debugging
					if (((unexpl_variance_approx < GOODNESS_FIT_MIN_DECREASE_LOG_LIKE_NOT_MET_) && (unexpl_variance_approx >= unexpl_variance_up_lim * 0.999)) ||
						std::isnan(unexpl_variance_approx) || std::isinf(unexpl_variance_approx)) {
						break;
					}
					else {
						unexpl_variance_up_lim = unexpl_variance_approx;
					}
				}
				//Log::REDebug("found upper limit, it = %d, upper_limit = %g", it, upper_limit);//for debugging
				//double unexpl_variance_approx = GoodnessFit_TKC_approx(2 * upper_limit, ll_mode, y_data, y_data_int, location_par_ptr, false);//for debugging
				//Log::REDebug("finding upper limit: it = %d, 2*upper_limit = %g, unexpl_variance_up_lim = %g", it, 2*upper_limit, unexpl_variance_approx);//for debugging
				//Log::REDebug(" ");
				//find minimum using bisection
				for (it = 0; it < 100; ++it) {
					double mid1 = lower_limit + (upper_limit - lower_limit) / 3.;
					double mid2 = lower_limit + 2. * (upper_limit - lower_limit) / 3.;
					double unex_var_mid1 = GoodnessFit_TKC_approx(mid1, ll_mode, y_data, y_data_int, location_par_ptr, false);
					double unex_var_mid2 = GoodnessFit_TKC_approx(mid2, ll_mode, y_data, y_data_int, location_par_ptr, false);
					if (TwoNumbersAreEqual<double>(unex_var_mid1, 1.) && TwoNumbersAreEqual<double>(unex_var_mid2, 1.)) {
						lower_limit = mid2;
					}
					else {
						if (unex_var_mid1 < unex_var_mid2 || std::isnan(unex_var_mid2) || std::isinf(unex_var_mid2)) {
							upper_limit = mid2;
						}
						else {
							lower_limit = mid1;
						}
					}
					//Log::REDebug("find minimum: it = %d, lower_limit = %g, upper_limit = %g, mid1 = %g, mid2 = %g, unex_var_mid1 = %g, unex_var_mid2 = %g",
					//	it, lower_limit, upper_limit, mid1, mid2, unex_var_mid1, unex_var_mid2);//for debugging
					if (std::abs(upper_limit - lower_limit) <= 1e-3 * std::abs(lower_limit)) {
						//Log::REDebug("find minimum: it = %d, lower_limit = %g, upper_limit = %g, mid1 = %g, mid2 = %g, unex_var_mid1 = %g, unex_var_mid2 = %g",
						//	it, lower_limit, upper_limit, mid1, mid2, unex_var_mid1, unex_var_mid2);//for debugging
						break;
					}
				}// end loop
				delta_location_par_ = (upper_limit + lower_limit) / 2.;
				GoodnessFit_TKC_approx(delta_location_par_, ll_mode, y_data, y_data_int, location_par_ptr, true);
				//Log::REDebug("FindDeltaMode_TKC_Approx: it = %d, delta_location_par_ = %g ", it, delta_location_par_);//for debugging
				//Log::REInfo("FindDeltaMode_TKC_Approx: it = %d, delta_location_par_ = %g, delta_log_like_up_ = %g, delta_log_like_down_ = %g",
				//	it, delta_location_par_, delta_log_like_up_, delta_log_like_down_);//for debugging
			}//end !const_delta_location_par_
		}//end FindDeltaMode_TKC_Approx

		double GoodnessFit_TKC_approx(double delta_location_par,
			double ll_mode,
			const double* y_data,
			const int* y_data_int,
			const double* location_par_ptr,
			bool only_calculate_delta_up_down) {
			CHECK(approximation_type_ == "triangular_kernel_curvature");
			vec_t location_par_delta(num_data_);//location parameter = mode of random effects + fixed effects + delta_location_par
			double unexpl_variance_approx = 1.;
			// Difference in log-likelihood when going up and down by delta_location_par
			AddConstantToLocationPar(location_par_ptr, delta_location_par, location_par_delta);
			delta_log_like_up_ = ll_mode - LogLikelihood(y_data, y_data_int, location_par_delta.data());
			AddConstantToLocationPar(location_par_ptr, -delta_location_par, location_par_delta);
			delta_log_like_down_ = ll_mode - LogLikelihood(y_data, y_data_int, location_par_delta.data());
			if (!only_calculate_delta_up_down) {
				if (delta_log_like_up_ < TKC_MIN_DECREASE_LOG_LIKE_ || delta_log_like_down_ < TKC_MIN_DECREASE_LOG_LIKE_) {
					unexpl_variance_approx = GOODNESS_FIT_MIN_DECREASE_LOG_LIKE_NOT_MET_;
				}
				else {
					// Difference in log-likelihood when going up and down by 0.5 * delta_location_par
					AddConstantToLocationPar(location_par_ptr, delta_location_par / 2., location_par_delta);
					double delta_log_like_up_half = ll_mode - LogLikelihood(y_data, y_data_int, location_par_delta.data());
					AddConstantToLocationPar(location_par_ptr, -delta_location_par / 2., location_par_delta);
					double delta_log_like_down_half = ll_mode - LogLikelihood(y_data, y_data_int, location_par_delta.data());
					// Difference in approximate quadratic log-likelihood when going up and down by +/- 1/0.5 * delta_location_par
					double neg_curvature = NegativeHessian_TKC_Approx_AsymLaplace(delta_location_par);
					double delta_ll_approx_up = Diff_TKC_Approx_LocationParPlusDelta(-neg_curvature, delta_location_par);
					double delta_ll_approx_up_half = Diff_TKC_Approx_LocationParPlusDelta(-neg_curvature, delta_location_par / 2.);
					double delta_ll_approx_down = Diff_TKC_Approx_LocationParPlusDelta(-neg_curvature, -delta_location_par);
					double delta_ll_approx_down_half = Diff_TKC_Approx_LocationParPlusDelta(-neg_curvature, -delta_location_par / 2.);
					// Difference between correct and approximate log-likelihood
					double delta_log_like_mean = (delta_log_like_up_ + delta_log_like_up_half + delta_log_like_down_ + delta_log_like_down_half) / 4.;
					double SS_res = std::pow(delta_log_like_up_ - delta_ll_approx_up, 2) + std::pow(delta_log_like_up_half - delta_ll_approx_up_half, 2) +
						std::pow(delta_log_like_down_ - delta_ll_approx_down, 2) + std::pow(delta_log_like_down_half - delta_ll_approx_down_half, 2);
					double SS_tot = std::pow(delta_log_like_up_ - delta_log_like_mean, 2) + std::pow(delta_log_like_up_half - delta_log_like_mean, 2) +
						std::pow(delta_log_like_down_ - delta_log_like_mean, 2) + std::pow(delta_log_like_down_half - delta_log_like_mean, 2);
					unexpl_variance_approx = SS_res / SS_tot;
				}
			}//end !only_calculate_delta_up_down
			return(unexpl_variance_approx);
		}//end GoodnessFit_TKC_approx

		/*!
		* \brief Calculate ll_approx(location_par) - ll_approx(location_par + delta_location_par) where ll_approx() is a quadratic approximation with constant curvature
		* \param curvature Curvature (- approximate Hessian)
		* \param delta_location_par step size by which the location_par is increased
		* \return ll_approx_diff
		*/
		double Diff_TKC_Approx_LocationParPlusDelta(double curvature,
			double delta_location_par) {
			return(-(sum_first_deriv_ * delta_location_par + num_data_ * delta_location_par * delta_location_par * curvature / 2.));
		}//end Diff_TKC_Approx_LocationParPlusDelta

		/*!
		* \brief Calculate the first derivative of the diagonal of the Fisher information wrt the location parameter aggregated per random effect.
		*			This is usually the negative third derivative of the log-likelihood wrt the location parameter.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param[out] deriv_information_diag_loc_par First derivative of the diagonal of the Fisher information wrt the location parameter
		* \param[out] deriv_information_diag_loc_par_data_scale First derivative of the diagonal of the Fisher information wrt the location parameter on the data-scale (only used if use_random_effects_indices_of_data_)
		*/
		void CalcFirstDerivInformationLocPar(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			vec_t& deriv_information_diag_loc_par,
			vec_t& deriv_information_diag_loc_par_data_scale) {
			CHECK(grad_information_wrt_mode_non_zero_);
			deriv_information_diag_loc_par = vec_t(dim_mode_per_set_re_);
			if (use_random_effects_indices_of_data_) {
				deriv_information_diag_loc_par_data_scale = vec_t(num_data_);
				CalcFirstDerivInformationLocPar_PerSample(y_data, y_data_int, location_par, deriv_information_diag_loc_par_data_scale);
				CalcZtVGivenIndices(num_data_, dim_mode_per_set_re_, random_effects_indices_of_data_, deriv_information_diag_loc_par_data_scale.data(), deriv_information_diag_loc_par.data(), true);
			}
			else {
				CalcFirstDerivInformationLocPar_PerSample(y_data, y_data_int, location_par, deriv_information_diag_loc_par);
			}
		}//end CalcFirstDerivInformationLocPar

		/*!
		* \brief Calculate the first derivative of the diagonal of the Fisher information wrt the location parameter per sample.
		*			This is usually the negative third derivative of the log-likelihood wrt the location parameter.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param[out] deriv_information_diag_loc_par First derivative of the diagonal of the Fisher information wrt the location parameter
		*/
		void CalcFirstDerivInformationLocPar_PerSample(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			vec_t& deriv_information_diag_loc_par) {
			if (approximation_type_ == "laplace") {
				if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double x = location_par[i];
						const double x2 = x * x;
						if (y_data_int[i] == 0) {
							const double pdf_div_omcdf = GPBoost::InvMillsRatioNormalOneMinusPhi(x);//phi(x) / (1 - Phi(x))
							deriv_information_diag_loc_par[i] = w * (-pdf_div_omcdf * (1.0 - x2 + pdf_div_omcdf * (3.0 * x - 2.0 * pdf_div_omcdf)));
						}
						else {
							const double pdf_div_cdf = GPBoost::InvMillsRatioNormalPhi(x);//phi(x) / Phi(x)
							deriv_information_diag_loc_par[i] = w * (-pdf_div_cdf * (x2 - 1.0 + pdf_div_cdf * (3.0 * x + 2.0 * pdf_div_cdf)));
						}
					}
				}
				else if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit" || likelihood_type_ == "quasi_bernoulli_logit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double p = GPBoost::sigmoid_stable(location_par[i]);
						deriv_information_diag_loc_par[i] = w * (-p * (1.0 - p) * (2.0 * p - 1.0));
						// Alternatice version (less numerically stable)
						//const double exp_loc_i = std::exp(location_par[i]);
						//deriv_information_diag_loc_par[i] = w * exp_loc_i * (1. - exp_loc_i) / std::pow(1 + exp_loc_i, 3);

					}
				}
				else if (likelihood_type_ == "binomial_probit" || likelihood_type_ == "quasi_bernoulli_probit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double x = location_par[i];
						const double x2 = x * x;
						if (y_data[i] == 0.) {
							const double pdf_div_omcdf = GPBoost::InvMillsRatioNormalOneMinusPhi(x);//phi(x) / (1 - Phi(x))
							deriv_information_diag_loc_par[i] = w * (-pdf_div_omcdf * (1.0 - x2 + pdf_div_omcdf * (3.0 * x - 2.0 * pdf_div_omcdf)));
						}
						else if (y_data[i] == 1.) {
							const double pdf_div_cdf = GPBoost::InvMillsRatioNormalPhi(x);//phi(x) / Phi(x)
							deriv_information_diag_loc_par[i] = w * (-pdf_div_cdf * (x2 - 1.0 + pdf_div_cdf * (3.0 * x + 2.0 * pdf_div_cdf)));
						}
						else {
							const double pdf_div_cdf = GPBoost::InvMillsRatioNormalPhi(x);//phi(x) / Phi(x)
							const double pdf_div_omcdf = GPBoost::InvMillsRatioNormalOneMinusPhi(x);//phi(x) / (1 - Phi(x))
							deriv_information_diag_loc_par[i] = w * (y_data[i] * (-pdf_div_cdf * (x2 - 1.0 + pdf_div_cdf * (3.0 * x + 2.0 * pdf_div_cdf))) +
								(1.0 - y_data[i]) * (-pdf_div_omcdf * (1.0 - x2 + pdf_div_omcdf * (3.0 * x - 2 * pdf_div_omcdf))));
						}
					}
				}
				else if (likelihood_type_ == "poisson") {
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return std::exp(location_par[i]); });
				}
				else if (likelihood_type_ == "zero_inflated_poisson") {
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return DerivInformationLocParZeroInflatedPoisson(y_data_int[i], location_par[i]); });
				}
				else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return DerivInformationLocParZeroInflatedNegBinFamily(y_data_int[i], location_par[i]); });
				}
				else if (likelihood_type_ == "gamma") {
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return -aux_pars_[0] * y_data[i] * std::exp(-location_par[i]); });
				}
				else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
					const double p = GetTweediePower();
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return EvaluateTweedieLocation(y_data[i], location_par[i], std::log(aux_pars_[0]), p).deriv_information_eta; });
				}
				else if (IsEGPDLikelihood() || IsHurdleEGPD()) {
					const bool hurdle = IsHurdleEGPD();
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.;
						if (w == 0. || (hurdle && y_data[i] <= 0.)) deriv_information_diag_loc_par[i] = 0.;
						else {
							const auto result = EvaluateEGPD(y_data[i], location_par[i]);
							deriv_information_diag_loc_par[i] = result.status == EGPDEvalStatus::kValid ? -w * result.d3_eta : std::numeric_limits<double>::quiet_NaN();
						}
					}
				}
				else if (likelihood_type_ == "negative_binomial") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double mu = std::exp(location_par[i]);
						const double mu_plus_r = mu + aux_pars_[0];
						deriv_information_diag_loc_par[i] = w * -(y_data_int[i] + aux_pars_[0]) * mu * aux_pars_[0] * (mu - aux_pars_[0]) / (mu_plus_r * mu_plus_r * mu_plus_r);
					}
				}
				else if (likelihood_type_ == "negative_binomial_1") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double mu = std::exp(location_par[i]);
						const double r = mu / aux_pars_[0];
						const double dig_diff = GPBoost::digamma(y_data_int[i] + r) - GPBoost::digamma(r);
						const double tri_diff = GPBoost::trigamma(y_data_int[i] + r) - GPBoost::trigamma(r);
						const double tet_diff = GPBoost::tetragamma(y_data_int[i] + r) - GPBoost::tetragamma(r);
						const double C = dig_diff - std::log1p(aux_pars_[0]);
						deriv_information_diag_loc_par[i] = w * (-3.0 * r * r * tri_diff - r * r * r * tet_diff - r * C);
					}
				}
				else if (likelihood_type_ == "beta") {
					const double phi_raw = aux_pars_[0];
					const double phi = (phi_raw > 0.0 && std::isfinite(phi_raw)) ? phi_raw : 1e-16;
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double mu = GPBoost::sigmoid_stable_clamped(location_par[i]);
						const double d = mu * (1.0 - mu);
						const double y = y_data[i];
						const double logit_y = std::log(y) - std::log1p(-y);
						const double dig1 = GPBoost::digamma((1.0 - mu) * phi);
						const double dig2 = GPBoost::digamma(mu * phi);
						const double tri1 = GPBoost::trigamma((1.0 - mu) * phi);
						const double tri2 = GPBoost::trigamma(mu * phi);
						const double tet1 = GPBoost::tetragamma((1.0 - mu) * phi);
						const double tet2 = GPBoost::tetragamma(mu * phi);
						const double C = dig1 - dig2 + logit_y;
						const double S = tri1 + tri2;
						const double Dlt = tet2 - tet1;
						const double term_trigam = 3.0 * phi * phi * d * d * (1.0 - 2.0 * mu) * S;
						const double term_tetragam = phi * phi * phi * d * d * d * Dlt;
						const double gp = d * ((1.0 - 2.0 * mu) * (1.0 - 2.0 * mu) - 2.0 * d);
						const double term_jacob = -phi * gp * C;
						deriv_information_diag_loc_par[i] = w * (term_trigam + term_tetragam + term_jacob);
					}
				}
				else if (likelihood_type_ == "t") {
					double nu_sigma2 = aux_pars_[1] * aux_pars_[0] * aux_pars_[0];
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double res = y_data[i] - location_par[i];
						const double res_sq = res * res;
						const double denom = nu_sigma2 + res_sq;
						deriv_information_diag_loc_par[i] = w * -2. * (aux_pars_[1] + 1.) * (res_sq - 3. * nu_sigma2) * res / (denom * denom * denom);
					}
				}
				else if (IsGaussianLikelihood() || likelihood_type_ == "lognormal" || likelihood_type_ == "hurdle_lognormal") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						deriv_information_diag_loc_par[i] = 0.;
					}
				}
				else if (IsHurdleRegression()) {
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return HurdleRegression_dJetadEta(y_data[i], location_par[i]); });
				}
				else if (IsZeroInflatedCountRegression()) {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t iz = 0; iz < num_data_; ++iz) {
						const double wz = has_weights_ ? weights_[iz] : 1.0;
						ZICountRegQuant o; ZICountRegressionQuantities(y_data_int[iz], location_par[iz], location_par[iz + num_data_], o);
						deriv_information_diag_loc_par[iz] = wz * o.dJetadEta;
					}
				}
				else if (likelihood_type_ == "beta_binomial") {
					CHECK(has_weights_);
					const double phi_raw = aux_pars_[0];
					const double phi = (phi_raw > 0.0 && std::isfinite(phi_raw)) ? phi_raw : 1e-16;
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = weights_[i];
						if (!(w > 0.0) || !std::isfinite(w)) {
							deriv_information_diag_loc_par[i] = 0.0;
						}
						else {
							const double mu = GPBoost::sigmoid_stable_clamped(location_par[i]);
							const double s = mu * (1.0 - mu);
							const double a = mu * phi;
							const double b = (1.0 - mu) * phi;
							const double k = y_data[i] * w;
							const double Delta = GPBoost::digamma(k + a) - GPBoost::digamma(a) - GPBoost::digamma(w - k + b) + GPBoost::digamma(b);
							const double S1 = GPBoost::trigamma(k + a) - GPBoost::trigamma(a);
							const double S2 = GPBoost::trigamma(w - k + b) - GPBoost::trigamma(b);
							const double T1 = GPBoost::tetragamma(k + a) - GPBoost::tetragamma(a);
							const double T2 = GPBoost::tetragamma(w - k + b) - GPBoost::tetragamma(b);
							deriv_information_diag_loc_par[i] = -(phi * s * (1.0 - 6.0 * mu + 6.0 * mu * mu) * Delta
								+ 3.0 * phi * phi * s * s * (1.0 - 2.0 * mu) * (S1 + S2) + phi * phi * phi * s * s * s * (T1 - T2));
						}
					}
				} // end "beta_binomial"
				else if (likelihood_type_ == "hurdle_gamma") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.0) {
							deriv_information_diag_loc_par[i] = 0.0;
						}
						else {
							deriv_information_diag_loc_par[i] = w * -aux_pars_[0] * y_data[i] * std::exp(-location_par[i]);
						}
					}
				}//end "hurdle_gamma"
				else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double sigma = aux_pars_[0];
						const double mu = location_par[i];
						if (y_data[i] <= 0.0) {
							// info(mu) = (1/sigma^2) * r * (a0 + r),  a0 = -mu/sigma, r = phi(a0)/Phi(a0)
							// d/dmu info(mu) = (r/sigma^3) * ( (a0 + r)*(a0 + 2r) - 1 )
							const double a0 = -mu / sigma;
							const double r = GPBoost::InvMillsRatioNormalPhi(a0);
							deriv_information_diag_loc_par[i] = w * (r / (sigma * sigma * sigma)) * ((a0 + r) * (a0 + 2.0 * r) - 1.0);
						}
						else {
							deriv_information_diag_loc_par[i] = 0.0;
						}
					}
				}//end "zero_censored_power_transformed_normal"
				else if (IsZeroCensPowNormHetero()) {
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return DerivInformationZeroCensPowNormHetero(y_data[i], location_par[i], location_par[i + num_data_]); });
				}//end "zero_censored_power_transformed_normal_heteroscedastic"
				else if (likelihood_type_ == "zoctn") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double s = aux_pars_[0]; // sigma
						const double mu = location_par[i];
						if (y_data[i] <= 0.0) {
							// info(mu) = (1/s^2) * r * (a0 + r),  a0 = -mu/s, r = phi(a0)/Phi(a0)
							// d/dmu info(mu) = (r/s^3) * ( (a0 + r)*(a0 + 2r) - 1 )
							const double a0 = -mu / s;
							const double r = GPBoost::InvMillsRatioNormalPhi(a0);
							deriv_information_diag_loc_par[i] = w * (r / (s * s * s)) * ((a0 + r) * (a0 + 2.0 * r) - 1.0);
						}
						else if (y_data[i] >= 1.0) {
							// info(mu) = (1/s^2) * r2 * (r2 - v),  v = (1-mu)/s, r2 = phi(v)/(1-Phi(v))
							// d/dmu info(mu) = (r2/s^3) * (1 - v^2 + 3*v*r2 - 2*r2^2)
							const double v = (1.0 - mu) / s;
							const double r2 = GPBoost::InvMillsRatioNormalOneMinusPhi(v);
							const double term = 1.0 - v * v + 3.0 * v * r2 - 2.0 * r2 * r2;
							deriv_information_diag_loc_par[i] = w * (r2 / (s * s * s)) * term;
						}
						else {
							deriv_information_diag_loc_par[i] = 0.0;// 0 < y < 1: info(mu) = 1/s^2 (constant in mu) -> derivative 0
						}
					}
				}//end "zoctn"
				else if (likelihood_type_ == "zero_one_censored_transformed_beta") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						const double eta = location_par[i];
						const double h = 1e-4; // numeric derivative
						const double Hm = SecondDerivNegLogLikZeroOneCensTransfBeta(yi, eta - h);
						const double Hp = SecondDerivNegLogLikZeroOneCensTransfBeta(yi, eta + h);
						const double dI = (Hp - Hm) / (2.0 * h);
						deriv_information_diag_loc_par[i] = w * dI;
					}
				}//end "zero_one_censored_transformed_beta"
				else if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						const double eta = location_par[i];
						const double k = aux_pars_[0];
						const double xi = aux_pars_[1];
						const double mu = std::exp(eta);                  // dmu/deta = mu
						const double tiny = 1e-300;
						if (yi <= 0.0) {
							if (xi <= 0.0) {
								deriv_information_diag_loc_par[i] = 0.0;
							}
							else {
								const double a = xi;
								const double t = std::max(tiny, (k * a) / std::max(mu, 1e-12));
								const double G = std::max(GPBoost::RegLowerGamma(k, t), tiny);
								// p = Gamma(k,1) pdf at t
								const double p = std::exp(-t + (k - 1.0) * std::log(t) - std::lgamma(k));
								const double Q = p / G;                         // lower-tail ratio
								const double Qprime = Q * ((k - 1.0) / t - 1.0) - (Q * Q); // dQ/dt for Q=p/G
								// dI/dt (lower mass)
								const double dIdt = (2.0 * t - k) * Q + 2.0 * t * Q * Q + Qprime * (t * (t - k) + 2.0 * t * t * Q);
								const double dIdmu = -(a * k / (mu * mu)) * dIdt;  // dt/dmu = -(k*a)/mu^2
								deriv_information_diag_loc_par[i] = w * (mu * dIdmu); // dI/deta = mu * dI/dmu
							}
						}
						else if (yi >= 1.0) {
							const double a = 1.0 + xi;
							const double t = std::max(tiny, (k * a) / std::max(mu, 1e-12));
							const double G = GPBoost::RegLowerGamma(k, t);
							const double H = std::max(1.0 - G, tiny);
							const double p = std::exp(-t + (k - 1.0) * std::log(t) - std::lgamma(k));
							const double Q = p / H;                           // upper-tail ratio
							const double Qprime = Q * ((k - 1.0) / t - 1.0) + (Q * Q); // dQ/dt for Q=p/H
							// dI/dt (upper mass)
							const double dIdt = (k - 2.0 * t) * Q + 2.0 * t * Q * Q + Qprime * (t * (k - t) + 2.0 * t * t * Q);
							const double dIdmu = -(a * k / (mu * mu)) * dIdt;  // dt/dmu = -(k*a)/mu^2
							deriv_information_diag_loc_par[i] = w * (mu * dIdmu); // dI/deta
						}
						else {// interior: I(eta) = k * z / mu  => dI/dmu = -k*z/mu^2  => dI/deta = -k*z/mu							
							const double z = yi + xi;
							deriv_information_diag_loc_par[i] = w * (-(k * z) / std::max(mu, 1e-12));
						}
					}
				} // end "zero_one_censored_shifted_gamma"
				else {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_);
				}
			}//end approximation_type_ == "laplace"
			else if (approximation_type_ == "fisher_laplace") {
				if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double p = GPBoost::sigmoid_stable(location_par[i]);
						deriv_information_diag_loc_par[i] = w * (-p * (1.0 - p) * (2.0 * p - 1.0));
						// Alternatice version (less numerically stable)
						//const double exp_loc_i = std::exp(location_par[i]);
						//deriv_information_diag_loc_par[i] = w * exp_loc_i * (1. - exp_loc_i) / std::pow(1 + exp_loc_i, 3);

					}
				}
				else if (likelihood_type_ == "poisson") {
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return std::exp(location_par[i]); });
				}
				else if (likelihood_type_ == "t") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						deriv_information_diag_loc_par[i] = 0.;
					}
				}
				else if (IsGaussianLikelihood() || likelihood_type_ == "lognormal" || likelihood_type_ == "asymmetric_laplace") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						deriv_information_diag_loc_par[i] = 0.;
					}
				}
				else if (IsGaussianHeteroscedastic()) {
					ForEachSampleWeighted(deriv_information_diag_loc_par, [&](data_size_t i) { return -std::exp(-location_par[i + num_data_]); });
				}
					else if (IsZeroInflatedCount()) {
						// d Fisher / d eta by central differences (the Fisher information is a smooth function of eta = log mu).
						const int kind = ZICountBaseKind();
						const bool reg = IsZeroInflatedCountRegression();
						const double pi_const = reg ? 0. : ZICountConstantP0();
						const double h = 1e-5;
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							const double eta = location_par[i];
							const double pi = reg ? GPBoost::sigmoid_stable(location_par[i + num_data_]) : pi_const;
							ZICountZeroMass zp, zm;
							const double mup = std::exp(eta + h), mum = std::exp(eta - h);
							FillZeroMassZICountKind(mup, zp, kind); FillZeroMassZICountKind(mum, zm, kind);
							deriv_information_diag_loc_par[i] = w * (ZICountFisherInfoEta(mup, pi, zp, kind) - ZICountFisherInfoEta(mum, pi, zm, kind)) / (2. * h);
						}
					}
				else {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_);
				}
			}// end approximation_type_ == "fisher_laplace"
			else if (approximation_type_ == "triangular_kernel_curvature" || approximation_type_ == "constant_curvature_manual") {
				if (likelihood_type_ == "asymmetric_laplace") {
					double denom = num_data_ * delta_location_par_ * delta_location_par_;
					const double* first_deriv_ptr = use_random_effects_indices_of_data_ ? first_deriv_ll_data_scale_.data() : first_deriv_ll_.data();
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						deriv_information_diag_loc_par[i] = w * (2 * first_deriv_ptr[i] / w -
							FirstDerivLogLikAsymLaplace(y_data[i], location_par[i] + delta_location_par_) -
							FirstDerivLogLikAsymLaplace(y_data[i], location_par[i] - delta_location_par_)) / denom;
					}
				}//end asymmetric_laplace
			}//end approximation_type_ == "triangular_kernel_curvature"
			else {
				Log::REFatal("CalcFirstDerivInformationLocPar_PerSample: approximation_type '%s' is not supported ", approximation_type_.c_str());
			}
			first_deriv_information_loc_par_caluclated_ = true;
		}//end CalcFirstDerivInformationLocPar_PerSample

		/*!
		* \brief Calculates the gradient of the negative log-likelihood with respect to the
		*       additional parameters of the likelihood (e.g., shape for gamma). The gradient is usually calculated on the log-scale.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param[out] grad Gradient
		*/
		void CalcGradNegLogLikAuxPars(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			double* grad);

		/*!
		* \brief Calculates (i) the second derivative of the log-likelihood wrt the location parameter and an additional parameter (aux_pars_) of the likelihood
		*			and (ii) the first derivative of the diagonal of the Fisher information of the likelihood wrt an additional parameter (aux_pars_) of the likelihood.
		*			The latter (ii) is ususally negative third derivative of the log-likelihood wrt twice the location parameter and an additional parameter of the likelihood.
		*			The gradient wrt to the additional parameter (aux_pars_) is usually calculated on the log-scale.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param ind_aux_par Index of aux_pars_ wrt which the gradient is calculated (currently no used as there is only one)
		* \param[out] second_deriv_loc_aux_par Second derivative of the log-likelihood wrt the location parameter and an additional parameter of the likelihood
		* \param[out] deriv_information_aux_par First derivative of the diagonal of the Fisher information of the likelihood wrt an additional parameter of the likelihood
		*/
		void CalcSecondDerivLogLikFirstDerivInformationAuxPar(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			int ind_aux_par,
			double* second_deriv_loc_aux_par,
			double* deriv_information_aux_par) const;

		/*!
		* \brief Calculate the mean of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable ('RespMeanAdaptiveGHQuadrature')
		*/
		inline double CondMeanLikelihood(const double value) const;

		/*!
		* \brief Calculate the first derivative of the logarithm of the mean of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double FirstDerivLogCondMeanLikelihood(const double value) const;

		/*!
		* \brief Calculate the second derivative of the logarithm of the mean of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double SecondDerivLogCondMeanLikelihood(const double value) const;

		/*!
		* \brief Calculate the variance of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double CondVarLikelihood(const double value) const;

		/*!
		* \brief Calculate the first derivative of the logarithm of the variance of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double FirstDerivLogCondVarLikelihood(const double value) const;

		/*!
		* \brief Calculate the second derivative of the logarithm of the variance of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double SecondDerivLogCondVarLikelihood(const double value) const;

		// Gauss-Hermite quadrature for computing E[f(Z)] with Z~N(0,1): 
		//		E[f(Z)] = (1/sqrt(pi)) * sum_j w_j * f(sqrt(2) * x_j), F(Z) = (max(0, m + s * Z))^lambda
		//		This is used for the prediction of the response variable
		inline double TruncPowerNormalMomentGH(const double m, const double s, const double lambda) const;

		// Gauss-Hermite quadrature for likelihood == "zoctn"
		//		This is used for the prediction of the response variable
		inline double ZeroOneCensTransNormalMomentGH(const double m, const double s, bool second_moment) const;

		inline double XB_FirstMoment_(double mu, double phi, double u) const;

		inline double XB_SecondMoment_(double mu, double phi, double u) const;

		inline void ZOCG_MomentsGivenEta_(const double eta,
			const double k,
			const double xi,
			double& Ey,
			double& Ey2,
			const bool need_second) const;

		/*!
		* \brief Do Cholesky decomposition
		* \param[out] chol_fact Cholesky factor
		* \param psi Matrix for which the Cholesky decomposition should be done
		*/
		template <class T_chol_1>
		void CheckCholeskyFactorization(const T_chol_1& chol_fact, const char* caller) const {
			if (chol_fact.info() != Eigen::Success) {
				Log::REFatal("%s: Cholesky factorization failed because the matrix is not positive definite or contains non-finite values ", caller);
			}
		}

		template <class T_mat_1, typename std::enable_if <std::is_same<sp_mat_t, T_mat_1>::value ||
			std::is_same<sp_mat_rm_t, T_mat_1>::value>::type* = nullptr >
		void CalcChol(T_chol& chol_fact, const T_mat_1& psi, bool& chol_fact_pattern_analyzed) {
			if (!chol_fact_pattern_analyzed) {
				chol_fact.analyzePattern(psi);
				chol_fact_pattern_analyzed = true;
			}
			chol_fact.factorize(psi);
			CheckCholeskyFactorization(chol_fact, "CalcChol");
		}
		template <class T_mat_1, typename std::enable_if <std::is_same<den_mat_t, T_mat_1>::value>::type* = nullptr  >
		void CalcChol(T_chol& chol_fact, const T_mat_1& psi, bool&) {
			chol_fact.compute(psi);
			CheckCholeskyFactorization(chol_fact, "CalcChol");
		}

		// Initialize location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
		/*!
		* \brief Auxiliary function for initializinh the location parameter = mode of random effects + fixed effects
		* \param fixed_effects Fixed effects component of location parameter
		* \param[out] location_par Locatioon parameter (used only if use_random_effects_indices_of_data_)
		* \param[out] location_par_ptr Pointer to location parameter
		*/
		void InitializeLocationPar(const double* fixed_effects,
			vec_t& location_par,
			double** location_par_ptr) {
			if (use_random_effects_indices_of_data_ || fixed_effects != nullptr) {
				location_par = vec_t(dim_location_par_);// if !use_random_effects_indices_of_data_ && fixed_effects == nullptr, then location_par is not used and *location_par_ptr = mode_.data()
			}
			UpdateLocationParNewMode(mode_, fixed_effects, location_par, location_par_ptr);
		}// end InitializeLocationPar

		/*!
		* \brief Auxiliary function for updating the location parameter = Z*mode + fixed_effects with a new mode
		* \param mode Mode
		* \param fixed_effects Fixed effects component of location parameter
		* \param[out] location_par Location parameter
		* \param[out] location_par_ptr Pointer to location parameter
		*/
		void UpdateLocationParNewMode(vec_t& mode,
			const double* fixed_effects,
			vec_t& location_par,
			double** location_par_ptr) {
			if (num_sets_fixed_effects_ > num_sets_re_ && fixed_effects == nullptr) {
				Log::REFatal("UpdateLocationParNewMode: No fixed effects (covariates and / or tree-boosting scores) are provided for likelihood = '%s'. "
					"This likelihood requires a fixed effects term (e.g., covariates 'X' and / or the GPBoost tree-boosting algorithm) ",
					likelihood_type_.c_str());
			}
			if (use_random_effects_indices_of_data_) {
				for (int igp = 0; igp < num_sets_re_; ++igp) {
					if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							location_par[i + igp * num_data_] = mode[random_effects_indices_of_data_[i] + igp * dim_mode_per_set_re_];
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							location_par[i + igp * num_data_] = mode[random_effects_indices_of_data_[i] + igp * dim_mode_per_set_re_] + fixed_effects[i + igp * num_data_];
						}
					}
				}
				// Additional location parameter blocks that are related to fixed effects only (no random effect / GP), e.g., the log-error variance for 'gaussian_heteroscedastic'
				for (int igp = num_sets_re_; igp < num_sets_fixed_effects_; ++igp) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						location_par[i + igp * num_data_] = fixed_effects[i + igp * num_data_];
					}
				}
				*location_par_ptr = location_par.data();
			}//end use_random_effects_indices_of_data_
			else if (use_Z_) {
				CHECK(num_sets_re_ == 1);// not yet implemented otherwise
				if (num_sets_fixed_effects_ > num_sets_re_) {
					// location_par needs to hold additional fixed-effects-only blocks, so it cannot be auto-resized
					// to just num_data_ via assignment below (as is done in the simple case)
					location_par = vec_t(num_sets_fixed_effects_ * num_data_);
					location_par.segment(0, num_data_) = (*Zt_).transpose() * mode;
				}
				else {
					location_par = (*Zt_).transpose() * mode;//location parameter = mode of random effects + fixed effects
				}
				if (fixed_effects != nullptr) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						location_par[i] += fixed_effects[i];
					}
				}
				// Additional location parameter blocks that are related to fixed effects only (no random effect / GP), e.g., the log-error variance for 'gaussian_heteroscedastic'
				for (int igp = num_sets_re_; igp < num_sets_fixed_effects_; ++igp) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						location_par[i + igp * num_data_] = fixed_effects[i + igp * num_data_];
					}
				}
				*location_par_ptr = location_par.data();
			}
			else {// !use_random_effects_indices_of_data_ && !use_Z_
				CHECK(dim_mode_ == num_sets_re_ * num_data_);
				if (fixed_effects == nullptr) {
					CHECK(num_sets_fixed_effects_ == num_sets_re_);// not yet implemented otherwise: without any fixed effects, there cannot be additional fixed-effects-only blocks
					*location_par_ptr = mode.data();
				}
				else {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < dim_mode_; ++i) {
						location_par[i] = mode[i] + fixed_effects[i];
					}
					// Additional location parameter blocks that are related to fixed effects only (no random effect / GP), e.g., the log-error variance for 'gaussian_heteroscedastic'
					for (int igp = num_sets_re_; igp < num_sets_fixed_effects_; ++igp) {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							location_par[i + igp * num_data_] = fixed_effects[i + igp * num_data_];
						}
					}
					*location_par_ptr = location_par.data();
				}
			}//end !use_random_effects_indices_of_data_
		}//end UpdateLocationParNewMode

		/*!
		* \brief Partition the data into groups of size group_size_ sorted according to the order of the mode (currently not used)
		*/
		void DetermineGroupsOrderedMode() {
			if (only_one_grouped_RE_) {
				if (!group_indices_data_only_one_grouped_RE_found_) {
					num_groups_partition_data_ = dim_mode_per_set_re_;
					group_indices_data_.resize(num_groups_partition_data_);
					// 1) Number of data points per group
					std::vector<std::atomic<data_size_t>> counts_per_group(num_groups_partition_data_);
					for (data_size_t g = 0; g < num_groups_partition_data_; ++g) counts_per_group[g].store(0, std::memory_order_relaxed);
#pragma omp parallel for
					for (data_size_t i = 0; i < num_data_; ++i) {
						counts_per_group[random_effects_indices_of_data_[i]].fetch_add(1, std::memory_order_relaxed);
					}
					// 2) Allocate exact sizes + prepare write cursors for group_indices_data_
					std::vector<std::atomic<data_size_t>> write_pos(num_groups_partition_data_);
					for (data_size_t g = 0; g < num_groups_partition_data_; ++g) {
						const data_size_t sz = counts_per_group[g].load(std::memory_order_relaxed);
						group_indices_data_[g].resize(sz);
						write_pos[g].store(0, std::memory_order_relaxed);
					}
					// 3) Fill group_indices_data_
#pragma omp parallel for
					for (data_size_t i = 0; i < num_data_; ++i) {
						const data_size_t g = random_effects_indices_of_data_[i];
						const data_size_t pos = write_pos[g].fetch_add(1, std::memory_order_relaxed);
						group_indices_data_[g][pos] = static_cast<data_size_t>(i);
					}
					group_indices_data_only_one_grouped_RE_found_ = true;
				}//end !group_indices_data_only_one_grouped_RE_found_
			}//end only_one_grouped_RE_
			else {//! only_one_grouped_RE_
				if (use_random_effects_indices_of_data_ || use_Z_) {
					vec_t mode_data_scale(num_data_);
					if (use_random_effects_indices_of_data_) {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							mode_data_scale[i] = mode_[random_effects_indices_of_data_[i]];
						}
					}
					else {
						CHECK(use_Z_);
						if (num_sets_re_ == 1) {
							mode_data_scale = (*Zt_).transpose() * mode_;
						}
						else {
							mode_data_scale = (*Zt_).transpose() * (mode_.segment(0, dim_mode_per_set_re_));
						}
					}
					DetermineGroupsOrderedMode_Inner(mode_data_scale);
				}
				else {
					if (num_sets_re_ == 1) {
						DetermineGroupsOrderedMode_Inner(mode_);
					}
					else {
						DetermineGroupsOrderedMode_Inner(mode_.segment(0, num_data_));
					}
				}
			}//end !only_one_grouped_RE_		
		}//end DetermineGroupsOrderedMode
		void DetermineGroupsOrderedMode_Inner(const vec_t& mode) {
			std::vector<data_size_t> idx(num_data_);
			std::iota(idx.begin(), idx.end(), 0);// [0,1,2,…,n-1]       
			auto cmp = [&mode](auto a, auto b) { return mode[a] < mode[b]; };
			//#ifdef EXEC_POLICY
			//			std::sort(EXEC_POLICY, idx.begin(), idx.end(), cmp);
			//#else
			//			std::sort(idx.begin(), idx.end(), cmp);
			//#endif
						// the above can lead to compiler crashes on some compilers
			std::sort(idx.begin(), idx.end(), cmp);
			// Floor division, but always at least one group as long as there is data: any remaining points
			// (in particular all of them when num_data_ < group_size_) are merged into the last group below
			num_groups_partition_data_ = (num_data_ > 0) ? std::max<data_size_t>(1, num_data_ / group_size_) : 0;
			group_indices_data_.resize(num_groups_partition_data_);
#pragma omp parallel for schedule(static)
			for (data_size_t g = 0; g < num_groups_partition_data_; ++g) {
				const data_size_t first = g * group_size_;
				//the last group takes all remaining points, so it can be larger than group_size_
				const data_size_t last = (g == num_groups_partition_data_ - 1) ? num_data_ : (first + group_size_);
				group_indices_data_[g].assign(idx.begin() + first, idx.begin() + last);
			}
		}//end DetermineGroupsOrderedMode_Inner

		/*!
		* \brief Make sure that the mode can only change by 'MAX_CHANGE_MODE_NEWTON_' in Newton's method (cap_change_mode_newton_)
		* \param mode_new New mode after Newton update
		*/
		void CapChangeModeUpdateNewton(vec_t& mode_new) const {
			if (cap_change_mode_newton_) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < dim_mode_; ++i) {
					double abs_change = std::abs(mode_new[i] - mode_[i]);
					if (abs_change > MAX_CHANGE_MODE_NEWTON_) {
						mode_new[i] = mode_[i] + (mode_new[i] - mode_[i]) / abs_change * MAX_CHANGE_MODE_NEWTON_;
					}
				}
			}
		}//end CapChangeModeUpdateNewton

		/*!
		* \brief Checks whether the mode finding algorithm has converged
		* \param it Iteration number
		* \param approx_marginal_ll_new New value of covergence criterion
		* \param[out] approx_marginal_ll Current value of covergence criterion
		* \param[out] terminate_optim If true, the mode finding algorithm is stopped
		* \param[out] has_NA_or_Inf True if approx_marginal_ll_new is NA or Inf
		*/
		void CheckConvergenceModeFinding(int it,
			double approx_marginal_ll_new,
			double& approx_marginal_ll,
			bool& terminate_optim,
			bool& has_NA_or_Inf) {
			if (std::isnan(approx_marginal_ll_new) || std::isinf(approx_marginal_ll_new)) {
				has_NA_or_Inf = true;
				Log::REDebug(NA_OR_INF_WARNING_);
				approx_marginal_ll = approx_marginal_ll_new;
				na_or_inf_during_last_call_to_find_mode_ = true;
				return;
			}
			if (it == 0) {
				if (std::abs(approx_marginal_ll_new - approx_marginal_ll) < delta_conv_mode_finding_ * std::abs(approx_marginal_ll)) { // allow for small decreases in first iteration
					terminate_optim = true;
				}
			}
			else {
				if ((approx_marginal_ll_new - approx_marginal_ll) < delta_conv_mode_finding_ * std::abs(approx_marginal_ll)) {
					terminate_optim = true;
				}
			}
			if (terminate_optim && continue_mode_finding_after_fisher_) {
				if (!mode_finding_fisher_has_been_continued_) {
					terminate_optim = false;
					use_fisher_for_mode_finding_ = false;
					mode_finding_fisher_has_been_continued_ = true;
				}
				else {
					use_fisher_for_mode_finding_ = true;//reset to initial values for next call
					mode_finding_fisher_has_been_continued_ = false;
				}
			}
			if (terminate_optim) {
				if (approx_marginal_ll_new < approx_marginal_ll) {
					Log::REDebug(NO_INCREASE_IN_MLL_WARNING_);
				}
				approx_marginal_ll = approx_marginal_ll_new;
				return;
			}
			else {
				if ((it + 1) == maxit_mode_newton_ && maxit_mode_newton_ > 1) {
					Log::REDebug(NO_CONVERGENCE_WARNING_);
					if (continue_mode_finding_after_fisher_ && mode_finding_fisher_has_been_continued_) {
						use_fisher_for_mode_finding_ = true;//reset to initial values for next call
						mode_finding_fisher_has_been_continued_ = false;
					}
				}
				approx_marginal_ll = approx_marginal_ll_new;
			}
		}//end CheckConvergenceModeFinding

		bool HasNegativeValueInformationLogLik() const {
			if (!information_ll_can_be_negative_) return false;
			return GPBoost::HasNegativeValues<double>(information_ll_.data(), (data_size_t)information_ll_.size());
		}//end HasNegativeValueInformationLogLik

		bool HasNegativeValueInformationLogLikOnDataScale() const {
			if (!information_ll_can_be_negative_) return false;
			if (use_random_effects_indices_of_data_) {
				return GPBoost::HasNegativeValues<double>(information_ll_data_scale_.data(), (data_size_t)information_ll_data_scale_.size());
			}
			return HasNegativeValueInformationLogLik();
		}

		void LogFatalWithPotentialFisherLaplaceHint(const char* caller, const char* message) const {
			// Recommend Fisher-Laplace only as an alternative to a currently selected observed-Hessian Laplace determinant
			bool should_recommend_fisher_laplace = false, should_recommend_fisher_laplace_combined = false;
			if (approximation_type_ == "laplace" && LIKELIHOODS_SUPPORTS_FISHER_MODE_FINDING_.find(likelihood_type_) != LIKELIHOODS_SUPPORTS_FISHER_MODE_FINDING_.end()) {
				if (use_fisher_for_mode_finding_) {
					should_recommend_fisher_laplace = true, should_recommend_fisher_laplace_combined = false;
				}
				else {
					should_recommend_fisher_laplace = false, should_recommend_fisher_laplace_combined = true;
				}
			}
			const bool quasi_fisher = likelihood_type_ == "negative_binomial_1" || likelihood_type_ == "zero_inflated_negative_binomial_1" ||
				likelihood_type_ == "zero_inflated_regression_negative_binomial_1";
			if (should_recommend_fisher_laplace) {
				Log::REFatal("%s: %sTry likelihood '%s_fisher_laplace' to use %s information for both mode finding and log-determinant ",
					caller, message, likelihood_type_.c_str(), quasi_fisher ? "quasi-Fisher" : "Fisher");
			} 
			else if (should_recommend_fisher_laplace_combined) {
				Log::REFatal("%s: %sTry either (i) likelihood '%s_fisher_laplace_combined' to use %s information for mode finding and Hessian for log-determinant or (ii) likelihood '%s_fisher_laplace' to use %s information for both mode finding and log-determinant ",
					caller, message, likelihood_type_.c_str(), quasi_fisher ? "quasi-Fisher" : "Fisher",
					likelihood_type_.c_str(), quasi_fisher ? "quasi-Fisher" : "Fisher");
			}
			else {
				Log::REFatal("%s: %s", caller, message);
			}
		}

		bool HasZeroValueInformationLogLik() const {
			if (!information_ll_can_be_exact_zero_) return false;
			return GPBoost::HasExactZero<double>(information_ll_.data(), (data_size_t)information_ll_.size());
		}//end HasZeroValueInformationLogLik

		/*!
		* \brief Set the gradient wrt to additional likelihood parameters that are not estimated to some default values (usually 0.)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		*/
		void SetGradAuxParsNotEstimated(double* aux_par_grad) const {
			if (likelihood_type_ == "t" && !estimate_df_t_) {
				aux_par_grad[1] = 0.;
			}
		}//end SetGradAuxParsNotEstimated

		void ChecksBeforeModeFinding() const {
			if (continue_mode_finding_after_fisher_) {
				CHECK(use_fisher_for_mode_finding_ && !mode_finding_fisher_has_been_continued_);
			}
		}//end ChecksBeforeModeFinding

		/*!
		* \brief Initialize the mode at the beginning of the mode finding algorithm.
		*			Either the mode is reset to zero (if it has not been initialized yet, which is numerically more
		*			stable), or the current mode and the NA / Inf status are saved so that they can be rolled back
		*			later. Shared by all 'FindModePostRandEffCalcMLL*' variants
		* \return True if the previous mode was kept (and saved), false if the mode was (re-)initialized to zero
		*/
		bool InitializeModeForModeFinding() {
			ssn_alm_exact_subgradient_valid_ = false;//the stored subgradient belongs to the mode of the previous call
			if (!mode_initialized_) {//Better (numerically more stable) to re-initialize mode to zero in every call
				InitializeModeAvec();
				return false;
			}
			mode_previous_value_ = mode_;
			na_or_inf_during_second_last_call_to_find_mode_ = na_or_inf_during_last_call_to_find_mode_;
			return true;
		}//end InitializeModeForModeFinding

		/*!
		* \brief Acceptance test of the backtracking line search of the mode update: the candidate mode is accepted
		*			if the objective function has increased sufficiently (Armijo condition; 'grad_dot_direction' is 0
		*			if 'armijo_condition_' is false, in which case any increase is accepted), otherwise the learning
		*			rate is halved and the line search continues. Shared by all 'FindModePostRandEffCalcMLL*' variants
		* \param approx_marginal_ll_new Objective function at the candidate mode
		* \param approx_marginal_ll Objective function at the current mode
		* \param grad_dot_direction Inner product of the gradient and the search direction
		* \param[out] lr_mode Learning rate for the mode update, halved when the candidate mode is rejected
		* \return True if the candidate mode is accepted and the line search can be stopped
		*/
		bool AcceptModeUpdate(double approx_marginal_ll_new,
			double approx_marginal_ll,
			double grad_dot_direction,
			double& lr_mode) const {
			if (approx_marginal_ll_new < (approx_marginal_ll + c_armijo_ * lr_mode * grad_dot_direction) ||
				std::isnan(approx_marginal_ll_new) || std::isinf(approx_marginal_ll_new)) {
				lr_mode *= 0.5;
				return false;
			}
			return true;//approx_marginal_ll_new >= approx_marginal_ll
		}//end AcceptModeUpdate

		/*!
		* \brief Record the state after the mode finding algorithm has terminated
		* \param num_it Number of iterations of the mode finding algorithm
		*/
		void FinalizeModeFinding(int num_it) {
			mode_is_zero_ = false;
			num_it_mode_finding_ = num_it;
		}//end FinalizeModeFinding

		/*!
		* \brief Calculate log|Sigma W + I| using stochastic trace estimation and variance reduction.
		* \param num_data Number of data points
		* \param cg_max_num_it_tridiag Maximal number of iterations for conjugate gradient algorithm when being run as Lanczos algorithm for tridiagonalization
		* \param chol_fact_sigma_woodbury Cholesky factor of 'sigma_ip + sigma_cross_cov_T * sigma_residual^-1 * sigma_cross_cov'
		* \param chol_fact_sigma_ip Cholesky factor of 'sigma_ip'
		* \param cross_cov Cross-covariance matrix between inducing points and all data points
		* \param chol_fact_sigma_woodbury_woodbury Cholesky factor of 'sigma_ip - sigma_cross_cov_T * B_t * D_inv * B * (W + D_inv)^-1 * B * D_inv * B * sigma_cross_cov'
		* \param W_D_inv Vector (W + D^-1)
		* \param[out] has_NA_or_Inf Is set to TRUE if NA or Inf occured in the conjugate gradient algorithm
		* \param[out] log_det_Sigma_W_plus_I Solution for log|Sigma W + I|
		*/
		void CalcLogDetStochFSVA(const data_size_t& num_data,
			const int& cg_max_num_it_tridiag,
			const chol_den_mat_t& chol_fact_sigma_woodbury,
			const den_mat_t& chol_ip_cross_cov,
			const chol_den_mat_t& chol_fact_sigma_ip,
			const chol_den_mat_t& chol_fact_sigma_ip_preconditioner,
			const den_mat_t* cross_cov,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i,
			const vec_t& W_D_inv_inv,
			const chol_den_mat_t& chol_fact_sigma_woodbury_woodbury,
			const vec_t& W_D_inv,
			bool& has_NA_or_Inf,
			double& log_det_Sigma_W_plus_I);

		/*!
		* \brief Calculate (Sigma^-1 + ZtWZ)^-1 rhs
		*/
		void Inv_SigmaI_plus_ZtWZ_Vecchia_iterative(int cg_max_num_it,
			den_mat_t& I_k_plus_Sigma_L_kt_W_Sigma_L_k,
			const sp_mat_t& SigmaI,
			sp_mat_t& SigmaI_plus_W,
			const sp_mat_t& B,
			bool& has_NA_or_Inf,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			data_size_t cluster_i,
			REModelTemplate<T_mat, T_chol>* re_model,
			const vec_t& rhs,
			vec_t& SigmaI_plus_ZtWZ_inv_rhs,
			bool initialize_to_zero,
			bool calculate_preconditioners);
		// Overload when preconditioners are not calculated
		void Inv_SigmaI_plus_ZtWZ_Vecchia_iterative_given_PC(int cg_max_num_it,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			const vec_t& rhs,
			vec_t& SigmaI_plus_ZtWZ_inv_rhs,
			bool initialize_to_zero,
			bool& has_NA_or_Inf);

		/*!
		* \brief Calculate log|Sigma W + I| using stochastic trace estimation and variance reduction.
		* \param num_data Number of data points
		* \param cg_max_num_it_tridiag Maximal number of iterations for conjugate gradient algorithm when being run as Lanczos algorithm for tridiagonalization
		* \param I_k_plus_Sigma_L_kt_W_Sigma_L_k Preconditioner "piv_chol_on_Sigma": I_k + Sigma_L_k^T W Sigma_L_k
		* \param SigmaI Preconditioner "zero_infill_incomplete_cholesky": Column-major matrix containing B^T D^(-1) B
		* \param SigmaI_plus_W Preconditioner "zero_infill_incomplete_cholesky": Column-major matrix containing B^T D^(-1) B + W (W not yet updated)
		* \param B Preconditioner "zero_infill_incomplete_cholesky": Column-major matrix B in Vecchia approximation
		* \param[out] has_NA_or_Inf Is set to TRUE if NA or Inf occured in the conjugate gradient algorithm
		* \param[out] log_det_Sigma_W_plus_I Solution for log|Sigma W + I|
		* \param cluster_i Cluster index for which this is run
		* \param REModelTemplate REModelTemplate object for calling functions from it
		*/
		void CalcLogDetStochVecchia(const data_size_t& num_data,
			const int& cg_max_num_it_tridiag,
			den_mat_t& I_k_plus_Sigma_L_kt_W_Sigma_L_k,
			const sp_mat_t& SigmaI,
			sp_mat_t& SigmaI_plus_W,
			const sp_mat_t& B,
			bool& has_NA_or_Inf,
			double& log_det_Sigma_W_plus_I,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			data_size_t cluster_i,
			REModelTemplate<T_mat, T_chol>* re_model);

		/*!
		* \brief Calculate dlog|Sigma W + I|/db_i for all i in 1, ..., n using stochastic trace estimation and variance reduction.
		* \param deriv_information_diag_loc_par Derivative of the diagonal of the Fisher information of the likelihood (= usually negative third derivative of the log-likelihood with respect to the mode)
		* \param num_data Number of data points
		* \param d_log_det_Sigma_W_plus_I_d_mode[out] Solution for dlog|Sigma W + I|/db_i for all i in n
		* \param D_inv_plus_W_inv_diag[out] Preconditioner "Sigma_inv_plus_BtWB": diagonal of (D^(-1) + W)^(-1)
		* \param diag_WI[out] Preconditioner "piv_chol_on_Sigma": diagonal of W^(-1)
		* \param PI_Z[out] Preconditioner "Sigma_inv_plus_BtWB": P^(-1) Z
		* \param WI_PI_Z[out] Preconditioner "piv_chol_on_Sigma": W^(-1) P^(-1) Z
		* \param[out] WI_WI_plus_Sigma_inv_Z Preconditioner "piv_chol_on_Sigma": W^(-1) (W^(-1) + Sigma)^(-1) Z
		*/
		void CalcLogDetStochDerivModeVecchia(const vec_t& deriv_information_diag_loc_par,
			const data_size_t& num_data,
			vec_t& d_log_det_Sigma_W_plus_I_d_mode,
			vec_t& D_inv_plus_W_inv_diag,
			vec_t& diag_WI,
			den_mat_t& PI_Z,
			den_mat_t& WI_PI_Z,
			den_mat_t& WI_WI_plus_Sigma_inv_Z,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
			bool GPU_use) const;

		/*!
		* \brief Calculate dlog|Sigma W + I|/dtheta_j, using stochastic trace estimation and variance reduction.
		* \param num_data Number of data points
		* \param num_comps_total Total number of random effect components (= number of GPs)
		* \param j Index of current covariance parameter in vector theta
		* \param SigmaI_deriv_rm Derivative of Sigma^(-1) wrt. theta_j
		* \param B_grad_j Derivatives of matrices B ( = derivative of matrix -A) for Vecchia approximation wrt. theta_j
		* \param D_grad_j Derivatives of matrices D for Vecchia approximation wrt. theta_j
		* \param D_inv_plus_W_inv_diag Preconditioner "Sigma_inv_plus_BtWB": diagonal of (D^(-1) + W)^(-1)
		* \param PI_Z Preconditioner "Sigma_inv_plus_BtWB": P^(-1) Z
		* \param WI_PI_Z Preconditioner "piv_chol_on_Sigma": W^(-1) P^(-1) Z
		* \param[out] d_log_det_Sigma_W_plus_I_d_cov_pars Solution for dlog|Sigma W + I|/dtheta_j
		*/
		void CalcLogDetStochDerivCovParVecchia(const data_size_t& num_data,
			const int& num_comps_total,
			const int& j,
			const sp_mat_rm_t& SigmaI_deriv_rm,
			const sp_mat_t& B_grad_j,
			const sp_mat_t& D_grad_j,
			const vec_t& D_inv_plus_W_inv_diag,
			const den_mat_t& PI_Z,
			const den_mat_t& WI_PI_Z,
			double& d_log_det_Sigma_W_plus_I_d_cov_pars) const;

		/*!
		* \brief Calculate dlog|Sigma W + I|/daux, using stochastic trace estimation and variance reduction.
		* \param deriv_information_aux_par Negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
		* \param D_inv_plus_W_inv_diag Preconditioner "Sigma_inv_plus_BtWB": diagonal of (D^(-1) + W)^(-1)
		* \param diag_WI Preconditioner "piv_chol_on_Sigma": diagonal of W^(-1)
		* \param PI_Z Preconditioner "Sigma_inv_plus_BtWB": P^(-1) Z
		* \param WI_PI_Z Preconditioner "piv_chol_on_Sigma": W^(-1) P^(-1) Z
		* \param WI_WI_plus_Sigma_inv_Z Preconditioner "piv_chol_on_Sigma": W^(-1) (W^(-1) + Sigma)^(-1) Z
		* \param[out] d_detmll_d_aux_par Solution for dlog|Sigma W + I|/daux
		*/
		void CalcLogDetStochDerivAuxParVecchia(const vec_t& deriv_information_aux_par,
			const vec_t& D_inv_plus_W_inv_diag,
			const vec_t& diag_WI,
			const den_mat_t& PI_Z,
			const den_mat_t& WI_PI_Z,
			const den_mat_t& WI_WI_plus_Sigma_inv_Z,
			double& d_detmll_d_aux_par,
			const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i) const;

		// -------------------------------------------------------------------------------------------------------------
		// SSN-ALM helpers. The variables of the augmented Lagrangian (the residual variable 'z', the multiplier
		// 'lambda', and the active set 'A') all live on the observation scale; only the gradient 'Q b + Z^T lambda^+'
		// and the generalized Hessian 'Q + rho Z^T A Z' are formed on the scale of the modes. All of these are O(n)
		// componentwise operations, the linear solve with 'Q + rho Z^T A Z' is provided by the calling
		// 'FindModePostRandEffCalcMLL*' routine
		// -------------------------------------------------------------------------------------------------------------

		/*! \brief True if the SSN-ALM refinement of the mode should be applied after the quasi-Newton mode finding */
		bool UseSSNALMRefinement() const {
			return mode_refinement_ != ModeRefinement::kNone && likelihood_type_ == "asymmetric_laplace" &&
				!iid_model_ && maxit_mode_newton_ > 0 && num_sets_re_ == 1;
		}

		/*! \brief True if the design matrix Z that relates the latent variables to the observations is the identity */
		bool HasIdentityZ() const {
			return !use_random_effects_indices_of_data_ && !use_Z_;
		}

		/*!
		* \brief Bounds of the dual box B = prod_i [l_i, u_i] of the asymmetric Laplace likelihood, i.e., w_i / sigma times
		*			the subdifferential of the check loss rho_tau at 0: l_i = w_i (tau - 1) / sigma, u_i = w_i tau / sigma.
		*			Observations with a zero (effective) weight get the degenerate box {0}, which is correct since they do
		*			not contribute to the likelihood
		* \param[out] lower Lower bounds l, of length num_data_
		* \param[out] upper Upper bounds u, of length num_data_
		*/
		void CalcAsymLaplaceALMDualBounds(vec_t& lower,
			vec_t& upper) const {
			const double inv_scale = 1. / aux_pars_[0];
			lower.resize(num_data_);
			upper.resize(num_data_);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.;
				lower[i] = w * (quantile_ - 1.) * inv_scale;
				upper[i] = w * quantile_ * inv_scale;
			}
		}//end CalcAsymLaplaceALMDualBounds

		/*!
		* \brief Componentwise proximal operator of sum_i w_i / sigma * rho_tau(z_i) with the penalty rho, i.e., the exact
		*			solution of the z-subproblem of the augmented Lagrangian, together with the implied multiplier and the
		*			generalized derivative (the active set).
		*			Note: an observation with w_i = 0 has the identity as its prox and must not enter the active set even
		*			though z_i = 0 whenever v_i = 0, which is why the active set is determined from the thresholds and the
		*			weight and not from a comparison of z_i with 0
		* \param v Argument of the prox, v = r - lambda / rho, of length num_data_
		* \param rho Penalty parameter of the augmented Lagrangian
		* \param[out] z Prox value
		* \param[out] lambda_plus Implied multiplier lambda^+ = rho (z - v)
		* \param[out] active Active set indicator A_i = 1{z_i = 0 and w_i > 0}
		*/
		void CalcAsymLaplaceALMProx(const vec_t& v,
			double rho,
			vec_t& z,
			vec_t& lambda_plus,
			vec_t& active) const {
			const double inv_rho_scale = 1. / (rho * aux_pars_[0]);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.;
				const double a = w * quantile_ * inv_rho_scale, c = w * (1. - quantile_) * inv_rho_scale;
				if (v[i] > a) {
					z[i] = v[i] - a;
					active[i] = 0.;
				}
				else if (v[i] < -c) {
					z[i] = v[i] + c;
					active[i] = 0.;
				}
				else {
					z[i] = 0.;
					active[i] = (w > 0.) ? 1. : 0.;
				}
				lambda_plus[i] = rho * (z[i] - v[i]);
			}
		}//end CalcAsymLaplaceALMProx

		/*!
		* \brief Apply Z^T to a vector on the observation scale (identity, incidence matrix, or a general sparse Z)
		* \param v_data Vector of length num_data_
		* \param[out] Zt_v Vector of length dim_mode_
		*/
		void ApplyZtToDataVector(const vec_t& v_data,
			vec_t& Zt_v) const {
			if (use_random_effects_indices_of_data_) {
				Zt_v.resize(dim_mode_);
				ReduceToModeScale(v_data, Zt_v);
			}
			else if (use_Z_) {
				Zt_v = (*Zt_) * v_data;
			}
			else {
				Zt_v = v_data;
			}
		}//end ApplyZtToDataVector

		/*!
		* \brief Map the observation-scale active set of the semismooth Newton system to the representation that the
		*			'FindModePostRandEffCalcMLL*' routines use for 'information_ll_', i.e., aggregated with Z^T if Z is an
		*			incidence matrix and left on the data scale if a general sparse Z is used. The linear solve of a branch
		*			can then reuse its existing 'Sigma^-1 + Z^T W Z' algebra with W = rho * A
		* \param active Active set indicator of length num_data_
		* \param rho Penalty parameter of the augmented Lagrangian
		* \param[out] W_ssn Active-set curvature of length dim_deriv_ll_
		*/
		void AggregateActiveSetToInformationScale(const vec_t& active,
			double rho,
			vec_t& W_ssn) const {
			if (use_random_effects_indices_of_data_) {
				const vec_t rho_active = rho * active;
				W_ssn.resize(dim_mode_);
				ReduceToModeScale(rho_active, W_ssn);
			}
			else {
				W_ssn = rho * active;// data scale if use_Z_, identical to the scale of the modes otherwise
			}
		}//end AggregateActiveSetToInformationScale

		/*!
		* \brief Apply Z^T W Z to a vector on the scale of the modes, with W in the convention of 'information_ll_'
		* \param W_ssn Active-set curvature as returned by 'AggregateActiveSetToInformationScale'
		* \param x Vector of length dim_mode_
		* \param[out] out Vector of length dim_mode_
		*/
		void ApplyZtWZToModeVector(const vec_t& W_ssn,
			const vec_t& x,
			vec_t& out) const {
			if (use_Z_) {
				out = (*Zt_) * (W_ssn.cwiseProduct((*Zt_).transpose() * x));
			}
			else {// Z^T W Z is diagonal for Z = I and for an incidence matrix Z, for which W_ssn is already aggregated
				out = W_ssn.cwiseProduct(x);
			}
		}//end ApplyZtWZToModeVector

		/*!
		* \brief Residuals r = y - eta on the observation scale
		* \param y_data Response variable data
		* \param location_par Location parameter eta (random plus fixed effects)
		* \param[out] resid Residuals, of length num_data_
		*/
		void CalcAsymLaplaceResidual(const double* y_data,
			const double* location_par,
			vec_t& resid) const {
			resid.resize(num_data_);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
			for (data_size_t i = 0; i < num_data_; ++i) {
				resid[i] = y_data[i] - location_par[i];
			}
		}//end CalcAsymLaplaceResidual

		/*!
		* \brief Normalized KKT residual of the exact non-smooth MAP problem, used as the gate after the quasi-Newton
		*			mode finding. For Z = I this is the exact natural residual ||Qb - Pi_B(Qb + r)|| / (1 + ||Qb||), which
		*			is zero if and only if 'mode_' is an exact MAP. For a general Z, Qb does not determine the
		*			observation-scale dual vector uniquely, and the endpoint subgradient alpha^F_i = u_i if r_i > 0 and
		*			l_i otherwise is used instead. That criterion can over-trigger the refinement when the exact solution
		*			needs an interior subgradient at a kink, but it can never certify a wrong mode
		* \param Qmode Q * mode_, i.e., the gradient of the prior quadratic
		* \param resid Residuals r = y - eta
		* \return Normalized KKT residual
		*/
		double CalcAsymLaplaceKKTResidual(const vec_t& Qmode,
			const vec_t& resid,
			vec_t& alpha) const {
			vec_t lower, upper;
			CalcAsymLaplaceALMDualBounds(lower, upper);
			if (HasIdentityZ()) {
				alpha = (Qmode + resid).cwiseMax(lower).cwiseMin(upper);
			}
			else {
				alpha.resize(num_data_);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					alpha[i] = (resid[i] > 0.) ? upper[i] : lower[i];
				}
			}
			vec_t Zt_alpha;
			ApplyZtToDataVector(alpha, Zt_alpha);
			return (Qmode - Zt_alpha).norm() / (1. + Qmode.norm());
		}//end CalcAsymLaplaceKKTResidual

		/*!
		* \brief Try to certify that the current mode solves the non-smooth MAP problem exactly and, if that succeeds,
		*			store the certifying subgradient so that 'CalcFirstDerivLogLik' returns it instead of the endpoint
		*			convention. Two candidate duals are tried: the one produced by the augmented Lagrangian (which can
		*			certify an interior subgradient for a general Z) and, as a fallback, the projection / endpoint dual of
		*			'CalcAsymLaplaceKKTResidual'. Nothing is cached if neither is within the tolerance, so an unconverged
		*			refinement never advertises an exact score; the mode itself may still be returned, it is then simply
		*			an improved but not certified mode with the ordinary endpoint score
		* \param Qmode Q * mode_
		* \param resid Residuals r = y - eta at the current mode
		* \param alpha_alm Dual alpha = -lambda^+ of the augmented Lagrangian, empty if none is available
		* \return The KKT residual that describes the current mode
		*/
		double CertifyAsymLaplaceSubgradient(const vec_t& Qmode,
			const vec_t& resid,
			const vec_t& alpha_alm) {
			ssn_alm_exact_subgradient_valid_ = false;
			// The residual is recomputed here from the mode and the dual that are actually returned rather than taken
			//	from the iteration that produced them, so that the certification cannot be invalidated by any later
			//	change of the mode
			double kkt_alm = std::numeric_limits<double>::infinity();
			if ((data_size_t)alpha_alm.size() == num_data_) {
				kkt_alm = CalcAsymLaplaceDualKKTResidual(Qmode, resid, alpha_alm);
			}
			if (kkt_alm <= DELTA_CONV_SSN_ALM_) {
				ssn_alm_exact_subgradient_ = alpha_alm;
				ssn_alm_exact_subgradient_valid_ = true;
				SetFirstDerivLogLikFromDataScale(ssn_alm_exact_subgradient_);
				return kkt_alm;
			}
			vec_t alpha;
			const double kkt = CalcAsymLaplaceKKTResidual(Qmode, resid, alpha);
			if (kkt <= DELTA_KKT_GATE_SSN_ALM_) {
				ssn_alm_exact_subgradient_ = alpha;
				ssn_alm_exact_subgradient_valid_ = true;
				SetFirstDerivLogLikFromDataScale(ssn_alm_exact_subgradient_);
				return kkt;
			}
			return std::min(kkt, kkt_alm);
		}//end CertifyAsymLaplaceSubgradient

		/*!
		* \brief Violation of the exact optimality conditions of the non-smooth MAP problem by a given observation-scale
		*			dual: the normalized stationarity residual 'Qb = Z^T alpha' and the normalized complementarity residual
		*			'alpha in w / sigma * d rho_tau(r)'. Unlike 'CalcAsymLaplaceKKTResidual', which has to guess the dual,
		*			this is exact for a general Z and for a dual that is interior at a kink
		* \param Qmode Q * mode_
		* \param resid Residuals r = y - eta
		* \param alpha Observation-scale dual (alpha = -lambda^+ for a dual produced by the augmented Lagrangian)
		* \return max(eta_stationarity, eta_complementarity)
		*/
		double CalcAsymLaplaceDualKKTResidual(const vec_t& Qmode,
			const vec_t& resid,
			const vec_t& alpha) const {
			vec_t lower, upper;
			CalcAsymLaplaceALMDualBounds(lower, upper);
			vec_t Zt_alpha;
			ApplyZtToDataVector(alpha, Zt_alpha);
			const double eta_s = (Qmode - Zt_alpha).norm() / (1. + Qmode.norm());
			const vec_t p = (alpha + resid).cwiseMax(lower).cwiseMin(upper);
			const double eta_c = (alpha - p).norm() / (1. + alpha.norm());
			return std::max(eta_s, eta_c);
		}//end CalcAsymLaplaceDualKKTResidual

		/*!
		* \brief Stopping criterion of the outer augmented Lagrangian iterations: the optimality violation of the dual
		*			alpha = -lambda^+ plus the primal feasibility of the residual variable z of the subproblem. The primal
		*			term is internal to the augmented Lagrangian and is deliberately not part of the certification of the
		*			mode in 'CertifyAsymLaplaceSubgradient'
		* \param Qmode Q * mode_
		* \param resid Residuals r = y - eta
		* \param z Prox value
		* \param lambda_plus Implied multiplier
		* \return max(eta_stationarity, eta_complementarity, eta_primal)
		*/
		double CalcAsymLaplaceKKTResidualDual(const vec_t& Qmode,
			const vec_t& resid,
			const vec_t& z,
			const vec_t& lambda_plus) const {
			const double eta_p = (z - resid).norm() / (1. + resid.norm());
			return std::max(CalcAsymLaplaceDualKKTResidual(Qmode, resid, -lambda_plus), eta_p);
		}//end CalcAsymLaplaceKKTResidualDual

		/*!
		* \brief Reduced augmented Lagrangian phi(b) = 1/2 b^T Q b + sum_i w_i / sigma * rho_tau(z_i) + rho / 2 ||z - v||^2
		*			with v = r - lambda / rho and z = prox(v). This is the objective of the line search of the inner
		*			semismooth Newton iteration; it must not be replaced by the log-posterior of the original problem.
		*			Terms that are constant in b are omitted
		* \param mode Mode b
		* \param Qmode Q * b
		* \param v Argument of the prox
		* \param z Prox value
		* \param rho Penalty parameter of the augmented Lagrangian
		* \return Value of phi
		*/
		double CalcAsymLaplaceReducedALMObjective(const vec_t& mode,
			const vec_t& Qmode,
			const vec_t& v,
			const vec_t& z,
			double rho) const {
			const double inv_scale = 1. / aux_pars_[0];
			double loss = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:loss)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.;
				const double diff = z[i] - v[i];
				loss += w * inv_scale * ((z[i] > 0.) ? quantile_ * z[i] : (quantile_ - 1.) * z[i]) + 0.5 * rho * diff * diff;
			}
			return 0.5 * mode.dot(Qmode) + loss;
		}//end CalcAsymLaplaceReducedALMObjective

		/*!
		* \brief Apply kink-wise clipping for the asymmetric Laplace likelihood.
		* \details For asymmetric Laplace, the log-likelihood is non-smooth at kinks where
		*          \f$y_i = \eta_i\f$ with \f$\eta_i = \text{location\_par}_i\f$.
		*          If a proposed update `mode_try` crosses a kink (indicator \f$1(y_i \le \eta_i)\f$ flips),
		*          we project the affected coordinate(s) to the kink and cross it only by a small \f$\varepsilon\f$.
		*          If `use_random_effects_indices_of_data_` is true, one mode component can correspond to multiple
		*          data points; then we project to the *last crossed* kink (in direction of motion) plus/minus \f$\varepsilon\f$
		* \note Only supports `num_sets_re_ == 1` and does not handle `use_Z_`
		* \param y_data Response vector
		* \param fixed_effects Fixed effects (can be nullptr)
		* \param mode_old Current mode
		* \param[out] mode_try Proposed mode, modified in-place if clipping occurs
		* \return True if any coordinate was clipped
		*/
		bool ApplyKinkClippingAsymLaplace(const double* y_data,
			const double* fixed_effects,
			const vec_t& mode_old,
			vec_t& mode_try) const {
			CHECK(kink_cliping_);
			const double eps_orthant = 1e-10;
			if (likelihood_type_ != "asymmetric_laplace") return false;
			if (use_Z_) return false; // not handled here (multiple random effects relate to the same data point)
			CHECK(num_sets_re_ == 1);
			if (!use_random_effects_indices_of_data_) {
				CHECK(dim_location_par_ == dim_mode_);
				CHECK(dim_mode_ == (int)num_data_);
				bool any_clipped = false;
#pragma omp parallel for schedule(static) if(num_data_ >= 128) reduction(||:any_clipped)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double fe = (fixed_effects != nullptr) ? fixed_effects[i] : 0.0;
					const double eta0 = mode_old[i] + fe;
					const double eta1 = mode_try[i] + fe;
					const bool ind0 = (y_data[i] <= eta0);
					const bool ind1 = (y_data[i] <= eta1);
					if (ind0 != ind1) {
						const double y = y_data[i];
						const double y_up = std::nextafter(y, std::numeric_limits<double>::infinity());
						const double y_dn = std::nextafter(y, -std::numeric_limits<double>::infinity());
						const double ulp = std::max(y_up - y, y - y_dn);
						const double eps = std::max(eps_orthant, ulp);
						const double eta_proj = y + (ind0 ? -eps : +eps);
						mode_try[i] = eta_proj - fe;
						any_clipped = true;
					}
				}
				return any_clipped;
			}
			else {//use_random_effects_indices_of_data_
				const int d = dim_mode_per_set_re_;
				CHECK(d > 0);
				// Build grouping once (CSR: indptr + indices)
				// re_group_indptr_[j]..re_group_indptr_[j+1] gives the data indices i belonging to RE coord j.
				if (!re_grouping_built_) {
					re_group_indptr_.assign((size_t)d + 1, (data_size_t)0);
					// Count
					for (data_size_t i = 0; i < num_data_; ++i) {
						const int j = random_effects_indices_of_data_[i];
						CHECK(j >= 0 && j < d);
						re_group_indptr_[(size_t)j + 1] += 1;
					}
					// Prefix sum
					for (int j = 0; j < d; ++j) {
						re_group_indptr_[(size_t)j + 1] += re_group_indptr_[(size_t)j];
					}
					re_group_indices_.assign((size_t)num_data_, (data_size_t)0);
					std::vector<data_size_t> write_ptr(re_group_indptr_.begin(), re_group_indptr_.end());
					for (data_size_t i = 0; i < num_data_; ++i) {
						const int j = random_effects_indices_of_data_[i];
						const data_size_t pos = write_ptr[(size_t)j]++;
						re_group_indices_[(size_t)pos] = i;
					}
					re_grouping_built_ = true;
				}
				// 1) direction per coordinate
				std::vector<int> dir((size_t)d, 0);
#pragma omp parallel for schedule(static) if(d >= 256)
				for (int j = 0; j < d; ++j) {
					const double m0 = mode_old[j];
					const double m1 = mode_try[j];
					dir[(size_t)j] = (m1 > m0) ? +1 : ((m1 < m0) ? -1 : 0);
				}
				// 2) scan groups per coordinate and project if crossing occurs
				bool any_clipped = false;
#pragma omp parallel for schedule(static) if(d >= 256) reduction(||:any_clipped)
				for (int j = 0; j < d; ++j) {
					const int dj = dir[(size_t)j];
					if (dj == 0) continue;
					const double m0 = mode_old[j];
					const double m1 = mode_try[j];
					const data_size_t start = re_group_indptr_[(size_t)j];
					const data_size_t end = re_group_indptr_[(size_t)j + 1];
					if (start == end) continue; // no data points mapped to this coord
					bool crossed_any = false;
					double t_last = (dj > 0)
						? -std::numeric_limits<double>::infinity()
						: std::numeric_limits<double>::infinity();
					// Scan all data points i that depend on mode[j]
					for (data_size_t p = start; p < end; ++p) {
						const data_size_t i = re_group_indices_[(size_t)p];
						const double fe = (fixed_effects != nullptr) ? fixed_effects[i] : 0.0;
						const double t = y_data[i] - fe; // kink threshold in mode-space
						// Crossing test (consistent with indicator = 1(y <= mode + fe))
						if (dj > 0) { // moving up
							if (m0 < t && m1 >= t) {
								if (!crossed_any || t > t_last) t_last = t;
								crossed_any = true;
							}
						}
						else {      // moving down
							if (m0 >= t && m1 < t) {
								if (!crossed_any || t < t_last) t_last = t;
								crossed_any = true;
							}
						}
					}
					if (!crossed_any) continue;
					// Project to last crossed kink +/- eps
					const double up = std::nextafter(t_last, std::numeric_limits<double>::infinity());
					const double dn = std::nextafter(t_last, -std::numeric_limits<double>::infinity());
					const double ulp = std::max(up - t_last, t_last - dn);
					const double eps = std::max(eps_orthant, ulp);
					mode_try[j] = t_last + (dj > 0 ? eps : -eps);
					any_clipped = true;
				}
				return any_clipped;
			}//end use_random_effects_indices_of_data_
		}//end ApplyKinkClippingAsymLaplace
		// // non-parallel version
		//bool ApplyKinkClippingAsymLaplace(const double* y_data,
		//	const double* fixed_effects,
		//	const vec_t& mode_old,
		//	vec_t& mode_try) const {
		//	CHECK(kink_cliping_);
		//	const double eps_orthant = 1E-10;
		//	if (likelihood_type_ != "asymmetric_laplace") return false;
		//	if (use_Z_) return false; // not handled here (multiple random effects relate to the same data point)
		//	CHECK(num_sets_re_ == 1);
		//	if (!use_random_effects_indices_of_data_) {
		//		CHECK(dim_location_par_ == dim_mode_);
		//		CHECK(dim_mode_ == (int)num_data_);
		//		bool any_clipped = false;
		//		for (data_size_t i = 0; i < num_data_; ++i) {
		//			const double fe = (fixed_effects != nullptr) ? fixed_effects[i] : 0.0;
		//			const double eta0 = mode_old[i] + fe;
		//			const double eta1 = mode_try[i] + fe;
		//			const bool ind0 = (y_data[i] <= eta0);
		//			const bool ind1 = (y_data[i] <= eta1);
		//			if (ind0 != ind1) {
		//				// cross the kink, but only by eps (end on the "new" side)
		//				const double y = y_data[i];
		//				double y_up = std::nextafter(y, std::numeric_limits<double>::infinity());
		//				double y_dn = std::nextafter(y, -std::numeric_limits<double>::infinity());
		//				double ulp = std::max(y_up - y, y - y_dn);
		//				double eps = std::max(eps_orthant, ulp);
		//				double eta_proj = y + (ind0 ? -eps : +eps);
		//				mode_try[i] = eta_proj - fe;
		//				any_clipped = true;
		//			}
		//		}
		//		return any_clipped;
		//	}
		//	else {//use_random_effects_indices_of_data_
		//		bool any_clipped = false;
		//		std::vector<int> dir(dim_mode_per_set_re_, 0);
		//		std::vector<unsigned char> has_crossed(dim_mode_per_set_re_, 0);
		//		std::vector<double> last(dim_mode_per_set_re_, 0.0);
		//		for (int j = 0; j < dim_mode_per_set_re_; ++j) {
		//			const double m0 = mode_old[j];
		//			const double m1 = mode_try[j];
		//			if (m1 > m0) { 
		//				dir[j] = +1; 
		//				last[j] = -std::numeric_limits<double>::infinity(); 
		//			}
		//			else if (m1 < m0) { 
		//				dir[j] = -1; 
		//				last[j] = +std::numeric_limits<double>::infinity(); 
		//			}
		//		}
		//		for (data_size_t i = 0; i < num_data_; ++i) {
		//			const int j = random_effects_indices_of_data_[i];
		//			CHECK(j >= 0 && j < dim_mode_per_set_re_);
		//			const int dj = dir[j];
		//			if (dj == 0) continue;
		//			const double m0 = mode_old[j];
		//			const double m1 = mode_try[j];
		//			const double fe = (fixed_effects != nullptr) ? fixed_effects[i] : 0.0;
		//			const double t = y_data[i] - fe;
		//			if (dj > 0) { // up
		//				if (m0 < t && m1 >= t) {
		//					if (!has_crossed[j] || t > last[j]) last[j] = t;
		//					has_crossed[j] = 1;
		//				}
		//			}
		//			else { // down
		//				if (m0 >= t && m1 < t) {
		//					if (!has_crossed[j] || t < last[j]) last[j] = t;
		//					has_crossed[j] = 1;
		//				}
		//			}
		//		}
		//		for (int j = 0; j < dim_mode_per_set_re_; ++j) {
		//			if (!has_crossed[j]) continue;
		//			const int dj = dir[j];
		//			const double t_last = last[j];
		//			double y_up = std::nextafter(t_last, std::numeric_limits<double>::infinity());
		//			double y_dn = std::nextafter(t_last, -std::numeric_limits<double>::infinity());
		//			double ulp = std::max(y_up - t_last, t_last - y_dn);
		//			double eps = std::max(eps_orthant, ulp);					
		//			double proj = t_last + (dj > 0 ? eps : -eps);// Move across the kink in the direction of travel
		//			mode_try[j] = proj;
		//			any_clipped = true;
		//		}
		//		return any_clipped;
		//	}//end use_random_effects_indices_of_data_
		//}//end ApplyKinkClippingAsymLaplace

		/*! \brief Number of data points */
		data_size_t num_data_;
		/*! \brief Number of sets of random effects / GPs. This is larger than 1, e.g., heteroscedastic models */
		int num_sets_re_ = 1;
		/*! \brief Number of sets of fixed effects (covariates / boosting scores). This is >= num_sets_re_; it is larger than num_sets_re_ for likelihoods
		*		 where some location parameter blocks are related to fixed effects only, without an associated random effect / GP (e.g., 'gaussian_heteroscedastic') */
		int num_sets_fixed_effects_ = 1;
		/*! \brief Dimension (= length) of mode_ */
		data_size_t dim_mode_;
		/*! \brief Dimension (= length) of mode_ per parameter / set of RE / GP */
		data_size_t dim_mode_per_set_re_;
		/*! \brief Dimension (= length) of location par = Z * mode + F(X) */
		data_size_t dim_location_par_;
		/*! \brief Dimension (= length) of first_deriv_ll_ and information_ll_ */
		data_size_t dim_deriv_ll_;
		/*! \brief Dimension (= length) of first_deriv_ll_ and information_ll_ per parameter / set of RE / GP */
		data_size_t dim_deriv_ll_per_set_re_;
		/*! \brief Posterior mode used for Laplace approximation */
		vec_t mode_;
		/*! \brief Saving a previously found value allows for reseting the mode when having a too large step size. */
		vec_t mode_previous_value_;
		/*! \brief Auxiliary variable a = ZSigmaZt^-1 * mode_b used for Laplace approximation */
		vec_t SigmaI_mode_;
		/*! \brief Saving a previously found value allows for reseting the mode when having a too large step size. */
		vec_t SigmaI_mode_previous_value_;
		/*! \brief Indicates whether the vector SigmaI_mode_ / a=ZSigmaZt^-1 is used or not */
		bool has_SigmaI_mode_;
		/*! \brief First derivatives of the log-likelihood. If use_random_effects_indices_of_data_, this corresponds to Z^T * first_deriv_ll, i.e., it length is dim_mode_ */
		vec_t first_deriv_ll_;
		/*! \brief First derivatives of the log-likelihood on the data scale of length num_data_. Auxiliary variable used only if use_random_effects_indices_of_data_ */
		vec_t first_deriv_ll_data_scale_;
		/*! \brief The diagonal of the (observed or expected) Fisher information for the log-likelihood (diagonal of matrix "W"). Usually, this consists of the second derivatives of the negative log-likelihood (= the observed FI). If use_random_effects_indices_of_data_, this corresponds to Z^T * information_ll, i.e., it length is dim_mode_ */
		vec_t information_ll_;
		/*! \brief The diagonal of the (observed or expected) Fisher information for the log-likelihood (diagonal of matrix "W") on the data scale of length num_data_. Usually, this consists of the second derivatives of the negative log-likelihood. This is an auxiliary variable used only if use_random_effects_indices_of_data_ */
		vec_t information_ll_data_scale_;
		/*! \brief The off-diagonal elements (if there are any) of the (observed or expected) Fisher information for the log-likelihood (diagonal of matrix "W"). Usually, this consists of the second derivatives of the negative log-likelihood (= the observed FI). If use_random_effects_indices_of_data_, this corresponds to Z^T * information_ll, i.e., it length is dim_mode_ */
		vec_t off_diag_information_ll_;
		/*! \brief The off-diagonal elements (if there are any) of the (observed or expected) Fisher information for the log-likelihood (diagonal of matrix "W") on the data scale of length num_data_. Usually, this consists of the second derivatives of the negative log-likelihood. This is an auxiliary variable used only if use_random_effects_indices_of_data_ */
		vec_t off_diag_information_ll_data_scale_;
		/*! \brief (used only if information_has_off_diagonal_) The Fisher information for the log-likelihood (diagonal of matrix "W"). Usually, this consists of the second derivatives of the negative log-likelihood (= the observed FI) */
		sp_mat_t information_ll_mat_;
		/*! \brief Diagonal of matrix Sigma^-1 + Zt * W * Z in Laplace approximation (used only in version 'GroupedRE' when there is only one random effect and ZtWZ is diagonal. Otherwise 'diag_SigmaI_plus_ZtWZ_' is used for grouped REs) */
		vec_t diag_SigmaI_plus_ZtWZ_;
		/*! \brief Cholesky factors of matrix Sigma^-1 + Zt * W * Z in Laplace approximation (used only in version'GroupedRE' if there is more than one random effect). */
		chol_sp_mat_t chol_fact_SigmaI_plus_ZtWZ_grouped_;
		/*! \brief Cholesky factors of matrix Sigma^-1 + Zt * W * Z in Laplace approximation (used only in version 'Vecchia') */
		chol_sp_mat_t chol_fact_SigmaI_plus_ZtWZ_vecchia_;
		/*!
		* \brief Cholesky factors of matrix B = I + Wsqrt *  Z * Sigma * Zt * Wsqrt in Laplace approximation (for version 'Stable')
		*       or of matrix B = Id + ZtWZsqrt * Sigma * ZtWZsqrt (for version 'OnlyOneGPCalculationsOnREScale')
		*/
		T_chol chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_;
		/*! \brief Cholesky factor of dense matrix used in Newton's method for finding mode (used in version 'FITC') */
		chol_den_mat_t chol_fact_dense_Newton_;
		/*! \brief If true, the pattern for the Cholesky factor (chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, chol_fact_SigmaI_plus_ZtWZ_grouped_, or chol_fact_SigmaI_plus_ZtWZ_vecchia_) has been analyzed */
		bool chol_fact_pattern_analyzed_ = false;
		/*! \brief If true, the mode has been initialized to 0 */
		bool mode_initialized_ = false;
		/*! \brief If true, the mode has been determined */
		bool mode_has_been_calculated_ = false;
		/*! \brief If true, the mode is currently zero (after initialization) */
		bool mode_is_zero_ = false;
		/*! \brief If true, NA or Inf has occurred during the last call to find mode */
		bool na_or_inf_during_last_call_to_find_mode_ = false;
		/*! \brief If true, NA or Inf has occurred during the second last call to find mode when mode_previous_value_ was calculated */
		bool na_or_inf_during_second_last_call_to_find_mode_ = false;
		/*! \brief Normalizing constant of the log-likelihood (not all likelihoods have one) */
		double log_normalizing_constant_;
		/*! \brief If true, the function 'CalculateLogNormalizingConstant' has been called */
		bool normalizing_constant_has_been_calculated_ = false;
		/*! \brief Auxiliary quantities that do not depend on aux_pars_ for normalizing constant for likelihoods (not all likelihoods have one, for gamma this is sum( log(y) ) ) */
		double aux_log_normalizing_constant_;
		/*! \brief If true, the function 'CalculateAuxQuantLogNormalizingConstant' has been called */
		bool aux_normalizing_constant_has_been_calculated_ = false;
		/*! \brief If true, an incidendce matrix Z is used for duplicate locations and calculations are done on the random effects scale with the unique locations (used, e.g., for Vecchia) */
		bool use_random_effects_indices_of_data_ = false;
		/*! \brief Indices that indicate to which random effect every data point is related */
		const data_size_t* random_effects_indices_of_data_;
		/*! \brief True if Zt_ is used */
		bool use_Z_ = false;
		/*! \brief Zt Transpose Z^ T of random effects design matrix that relates latent random effects to observations / likelihoods(used only for multiple level grouped random effects) */
		const sp_mat_t* Zt_;
		/*! \brief True if there are only a single level grouped random effects */
		bool only_one_grouped_RE_ = false;
		/*! \brief True if there are weights */
		bool has_weights_ = false;
		/*! \brief Sample weights */
		const double* weights_;
		/*! \brief A learning rate for the likelihood for generalized Bayesian inference */
		double likelihood_learning_rate_ = 1.;
		/*! \brief Auxiliary weights for likelihood_learning_rate_ */
		vec_t weights_learning_rate_;
		/*! \brief If true, SigmaI_mode is always calculated  (currently not used) */
		bool save_SigmaI_mode_ = false;
		/*! \brief Indices of data points in grouped when the data is partitioned into groups of size group_size_ sorted according to the order of the mode  (currently not used) */
		std::vector<std::vector<data_size_t>> group_indices_data_;
		/*! \brief Group size if data is partitioned into groups (currently not used) */
		data_size_t group_size_ = 1000;
		/*! \brief Number of groups (currently not used) */
		data_size_t num_groups_partition_data_ = 0;
		/*! \brief True if group_indices_data_ has been determined for only_one_grouped_RE  */
		bool group_indices_data_only_one_grouped_RE_found_ = false;
		/*! \brief For saving fixed_effects pointer */
		const double* fixed_effects_ = nullptr;
		/*! \brief If true, this is an iid model without a random effects / GP component */
		bool iid_model_ = false;

		// -------------------------------------------------------------------------------------------------------------
		// SSN-ALM: exact non-smooth refinement of the posterior mode for the 'asymmetric_laplace' likelihood.
		// The (Fisher) quasi-Newton mode finding uses a smooth fixed-point iteration with the endpoint convention for the
		// score at the kinks of the check loss and can therefore stall at points that do not satisfy the exact
		// (sub-differential) MAP optimality conditions. If enabled, the mode is certified with an exact KKT check after
		// the quasi-Newton loop and, if the check fails, refined with a semismooth Newton method applied to the
		// subproblems of an augmented Lagrangian method (see 'RefineModeAsymLaplaceSSNALM').
		// Note: the active-set curvature 'rho * Z^T A Z' of the semismooth Newton system exists only to solve the
		//		augmented Lagrangian subproblem. It is not used as the curvature of the Laplace approximation.
		// -------------------------------------------------------------------------------------------------------------
		/*! \brief Options for refining the posterior mode after the quasi-Newton mode finding (currently 'asymmetric_laplace' only) */
		enum class ModeRefinement { kNone, kSSNALMIfNeeded, kSSNALMAlways };
		/*! \brief Selected mode refinement, set by 'ParseLikelihoodAliasModeRefinement' from the likelihood name */
		ModeRefinement mode_refinement_ = ModeRefinement::kNone;
		/*! \brief Multiplier of the augmented Lagrangian on the observation scale (length num_data_, not dim_mode_) */
		vec_t ssn_alm_lambda_;
		/*! \brief Tolerance of the normalized KKT residual below which the quasi-Newton mode is certified and returned */
		static constexpr double DELTA_KKT_GATE_SSN_ALM_ = 1e-6;
		/*! \brief Tolerance of the normalized KKT residuals for terminating the outer augmented Lagrangian iterations */
		static constexpr double DELTA_CONV_SSN_ALM_ = 1e-6;
		/*! \brief Maximal number of outer (multiplier update) iterations of the augmented Lagrangian method */
		static constexpr int MAXIT_SSN_ALM_OUTER_ = 10;
		/*! \brief Maximal number of inner semismooth Newton iterations per augmented Lagrangian subproblem */
		static constexpr int MAXIT_SSN_ALM_INNER_ = 20;
		/*! \brief Growth factor of the penalty parameter 'rho' after every outer iteration */
		static constexpr double SSN_ALM_RHO_GROWTH_ = 2.;
		/*! \brief Maximal penalty parameter, as a multiple of its initial value */
		static constexpr double SSN_ALM_RHO_MAX_MULT_ = 1e4;
		/*! \brief Number of outer augmented Lagrangian iterations of the last call (diagnostics) */
		int num_it_mode_finding_ssn_alm_outer_ = 0;
		/*! \brief Total number of accepted semismooth Newton steps of the last call (diagnostics) */
		int num_it_mode_finding_ssn_ = 0;
		/*! \brief True if the multiplier of the last call was warm started from the previous call (diagnostics) */
		bool ssn_alm_lambda_was_warm_started_ = false;
		/*! \brief True if the KKT gate failed and the refinement was run in the last call (diagnostics) */
		bool ssn_alm_was_needed_ = false;
		/*! \brief Normalized KKT residual of the quasi-Newton mode of the last call (diagnostics, -1 if not calculated) */
		double ssn_alm_initial_kkt_resid_ = -1.;
		/*! \brief Normalized KKT residual of the returned mode of the last call (diagnostics, -1 if not calculated) */
		double ssn_alm_final_kkt_resid_ = -1.;
		/*! \brief Exact KKT subgradient alpha = -lambda^+ of the refined mode on the observation scale (see 'CalcFirstDerivLogLik') */
		vec_t ssn_alm_exact_subgradient_;
		/*! \brief True if 'ssn_alm_exact_subgradient_' corresponds to the current mode and must be used as the score */
		bool ssn_alm_exact_subgradient_valid_ = false;

		/*! \brief Type of likelihood  */
		string_t likelihood_type_ = "gaussian";
		/*! \brief Cached quantities derived from 'likelihood_type_' (see CacheLikelihoodTypeDerivedQuantities) */
		string_t regression_base_type_ = "gaussian";
		string_t egpd_base_type_ = "gaussian";
		EGPDVariant egpd_variant_ = EGPDVariant::kGPD;
		static constexpr double TWEEDIE_POWER_LOWER_ = 1.01;
		static constexpr double TWEEDIE_POWER_UPPER_ = 1.99;
		double tweedie_sum_d_log_a_rho_ = 0.;
		double tweedie_sum_d_log_a_theta_ = 0.;
		// Snapshot of (phi, p) that the cached normalizer aggregates above correspond to; used to verify cache freshness in CalcGradNegLogLikAuxPars.
		double tweedie_cached_phi_ = std::numeric_limits<double>::quiet_NaN();
		double tweedie_cached_p_ = std::numeric_limits<double>::quiet_NaN();
		mutable bool tweedie_boundary_warning_issued_ = false;
		/*! \brief Cache of the EGPD unit-scale moments and the aux-parameter snapshot they correspond to (see GetEGPDMoments) */
		EGPDMoments egpd_moments_cache_;
		std::array<double, kMaxEGPDAuxPars> egpd_moments_cache_aux_{};
		bool egpd_moments_cache_initialized_ = false;
		/*! \brief List of supported likelihoods */
		const std::set<string_t> SUPPORTED_LIKELIHOODS_{ "gaussian", "gaussian_latent", "bernoulli_probit", "bernoulli_logit", "binomial_probit", "binomial_logit", "quasi_bernoulli_probit", "quasi_bernoulli_logit",
			"poisson", "gamma", "tweedie", "tweedie_fixed_p", "negative_binomial", "negative_binomial_1", "beta", "t", "gaussian_heteroscedastic", "gaussian_heteroscedastic_fixed_and_random", "lognormal", "beta_binomial",
			"hurdle_gamma", "hurdle_lognormal", "zero_censored_power_transformed_normal", "zero_censored_power_transformed_normal_heteroscedastic",
			"zoctn", "zero_one_censored_transformed_beta", "zero_one_censored_shifted_gamma",
			"asymmetric_laplace", "gpd", "egpd_power", "egpd_power_mixture", "egpd_beta", "egpd_power_beta",
			"zero_inflated_poisson", "zero_inflated_negative_binomial", "zero_inflated_negative_binomial_1",
			"hurdle_gpd", "hurdle_egpd_power", "hurdle_egpd_power_mixture", "hurdle_egpd_beta", "hurdle_egpd_power_beta",
			"hurdle_regression_gamma", "hurdle_regression_lognormal", "hurdle_regression_gpd", "hurdle_regression_egpd_power",
			"hurdle_regression_egpd_power_mixture", "hurdle_regression_egpd_beta", "hurdle_regression_egpd_power_beta",
			"zero_inflated_regression_poisson", "zero_inflated_regression_negative_binomial", "zero_inflated_regression_negative_binomial_1" };
		/*! \brief List of likelihoods that work only for a standard Laplace approximation */
		const std::set<string_t> LIKELIHOODS_ONLY_LAPLACE_{ "binomial_probit", "binomial_logit", "binomial_logit", "quasi_bernoulli_probit", "quasi_bernoulli_logit", "gamma", "negative_binomial",
			"beta", "beta_binomial", "tweedie", "tweedie_fixed_p", "hurdle_gamma", "hurdle_lognormal", "zero_censored_power_transformed_normal",
			"zero_censored_power_transformed_normal_heteroscedastic", "zoctn", "zero_one_censored_transformed_beta", "zero_one_censored_shifted_gamma",
			"gpd", "egpd_power", "egpd_power_mixture", "egpd_beta", "egpd_power_beta",
			"hurdle_gpd", "hurdle_egpd_power", "hurdle_egpd_power_mixture", "hurdle_egpd_beta", "hurdle_egpd_power_beta",
			"hurdle_regression_gamma", "hurdle_regression_lognormal", "hurdle_regression_gpd", "hurdle_regression_egpd_power",
			"hurdle_regression_egpd_power_mixture", "hurdle_regression_egpd_beta", "hurdle_regression_egpd_power_beta" };
		/*! \brief Likelihoods for which the (quasi-)Fisher information may be used for mode finding (use_fisher_for_mode_finding_ = true) */
		const std::set<string_t> LIKELIHOODS_SUPPORTS_FISHER_MODE_FINDING_{ "t", "asymmetric_laplace", "negative_binomial_1",
			"zero_inflated_poisson", "zero_inflated_negative_binomial", "zero_inflated_negative_binomial_1",
			"zero_inflated_regression_poisson", "zero_inflated_regression_negative_binomial", "zero_inflated_regression_negative_binomial_1" };
		/*! \brief True if response variable has int type */
		bool has_int_label_;
		/*! \brief Number of additional parameters for likelihoods */
		int num_aux_pars_ = 0;
		/*! \brief Number of additional parameters for likelihoods that are estimated */
		int num_aux_pars_estim_ = 0;
		/*! \brief Additional parameters for likelihoods. For "gamma", aux_pars_[0] = shape parameter, for gaussian, aux_pars_[0] = 1 / sqrt(variance) */
		std::vector<double> aux_pars_;
		/*! \brief Additional parameters on original scale to avoid recaculating them everytime (aux_pars_ can be on a transformed scale in order that they are in (0,infty) */
		std::vector<double> aux_pars_original_;
		/*! \brief Names of additional parameters for likelihoods */
		std::vector<string_t> names_aux_pars_;
		/*! \brief True, if the function 'SetAuxPars' has been called */
		bool aux_pars_have_been_set_ = false;
		/*! \brief Type of approximation for non-Gaussian likelihoods */
		string_t approximation_type_ = "laplace";
		/*! \brief Type of approximation for non-Gaussian likelihoods defined by user */
		string_t user_defined_approximation_type_ = "none";
		/*! \brief List of supported approximations */
		const std::set<string_t> SUPPORTED_APPROX_TYPE_{ "laplace", "fisher_laplace", "triangular_kernel_curvature" };
		/*! \brief If true, 'information_ll_' could contain negative values */
		bool information_ll_can_be_negative_ = false;
		/*! \brief If true, 'information_ll_' could contain exact zeros */
		bool information_ll_can_be_exact_zero_ = false;
		/*! \brief If true, the (observed or expected) Fisher information ('information_ll_') changes in the mode finding algorithm (usually Newton's method) for the Laplace approximation */
		bool information_changes_during_mode_finding_ = true;
		/*! \brief If true, the (observed or expected) Fisher information ('information_ll_') changes after the mode finding algorithm (e.g., if Fisher-Laplace is used for mode finding but Laplace for the final likelihood calculation) */
		bool information_changes_after_mode_finding_ = true;
		/*! \brief If true, the derivative of the information wrt the mode is non-zero (it is zero, e.g., for a "gaussian" likelihood) */
		bool grad_information_wrt_mode_non_zero_ = true;
		/*! \brief True, if the derivative of the information wrt the mode can be zero for some points even though it is non-zero generally */
		bool grad_information_wrt_mode_can_be_zero_for_some_points_ = false;
		/*! \brief True, if the information has off-diagonal elements */
		bool information_has_off_diagonal_ = false;
		/*! \brief If true, the (expected) Fisher information is used for the mode finding */
		bool use_fisher_for_mode_finding_ = false;
		/*! \brief If true, the mode finding is continued with an (approximae) Hessian after convergence has been achieved with the Fisher information */
		bool continue_mode_finding_after_fisher_ = false;
		/*! \brief True, if the mode finding has been continued with an (approximae) Hessian after convergence has been achieved with the Fisher information */
		bool mode_finding_fisher_has_been_continued_ = false;
		/*! \brief True, if the user explicitly set the mode finding approach in 'ParseLikelihoodAliasModeFindingMethod' */
		bool user_defined_mode_finding_approach_ = false;
		/*! \brief If true, the relationship "D log_lik(b) - Sigma^-1 b = 0" at the mode is used for calculating predictive means */
		bool can_use_first_deriv_log_like_for_pred_mean_ = true;
		/*! \brief If true, the degrees of freedom (df) are also estimated for the "t" likelihood */
		bool estimate_df_t_ = true;
		/*! \brief If true, a Gaussian likelihood is estimated using this file */
		bool use_likelihoods_file_for_gaussian_ = false;
		/*! \brief If true, the function 'CalcFirstDerivInformationLocPar_PerSample' has been called before */
		bool first_deriv_information_loc_par_caluclated_ = false;
		/*! \brief If true, this likelihood requires latent predictive variances for predicting response means */
		bool need_pred_latent_var_for_response_mean_ = true;
		/*! \brief If true, a curvature / variance correction is applied in 'CalcInformationLogLik' when calculating the approximate information for prediction */
		bool diag_information_variance_correction_for_prediction_ = false;
		/*! \brief If true, 'diag_information_variance_correction_for_prediction_' is enabled for prediction, otherwise not */
		bool use_variance_correction_for_prediction_ = false;
		/*! \brief Type of predictive variance correction */
		string_t var_cor_pred_version_ = "freq_asymptotic";

		// PARAMETERS FOR QUANTILE REGRESSION
		/*! \brief Quantile for asymmetric Laplace distribution */
		double quantile_;
		/*! \brief Tolerance level for calculating sub-gradients (currently not used) */
		const double EPSILON_SUB_GRAD_ = 1e-8;
		double eps_sub_grad_scale_ = EPSILON_SUB_GRAD_;
		/*! \brief If true, kink-clipping is applied during mode finding forthe asymmetric_laplace likelihood */
		bool kink_cliping_ = false;
		mutable bool re_grouping_built_ = false;
		mutable std::vector<data_size_t> re_group_indptr_;
		mutable std::vector<data_size_t> re_group_indices_;

		// PARAMETERS FOR THE TKC APPROXIMATION
		/*! \brief Parameter that determines the curvature in the TKC approximation such that the log-likelihood and the quadratic approximation match well in a neighborhood of "size" delta_location_par_ around the location_par F(X) + Zb (= |ll(location_par) - ll(location_par + delta_location_par_)|). */
		double delta_location_par_ = 1e-6;
		/*! \brief If true, the delta_location_par_ is constant */
		bool const_delta_location_par_ = false;
		/*! \brief Minimal decrease in log-likelihood for automatic finding of delta_location_par_ in TKC approximation */
		double TKC_MIN_DECREASE_LOG_LIKE_ = 0.1;
		/*! \brief Value returned by 'GoodnessFit_TKC_approx' if minimal decrease in log-likelihood for automatic finding of delta_location_par_ in TKC approximation is not met */
		double GOODNESS_FIT_MIN_DECREASE_LOG_LIKE_NOT_MET_ = 1e98;
		/*! \brief delta_log_like_up_ = ll(location_par) - ll(location_par + delta_location_par_) (including sign) */
		double delta_log_like_up_;
		/*! \brief delta_log_like_down_ = ll(location_par) - ll(location_par - delta_location_par_) */
		double delta_log_like_down_;
		/*! \brief Sum of first derivatives of the log-likelihood (used only in special cases) */
		double sum_first_deriv_;

		// MODE FINDING PROPERTIES
		/*! \brief Maximal number of iteration done for finding posterior mode with Newton's method */
		int maxit_mode_newton_ = 1000;
		/*! \brief Number of iteration done for finding posterior mode in Laplace approximation */
		int num_it_mode_finding_ = 0;
		/*! \brief Used for checking convergence in mode finding algorithm (terminate if relative change in Laplace approx. is below this value) */
		double delta_conv_mode_finding_ = 1e-8;
		/*! \brief Maximal number of steps for which learning rate shrinkage is done in the ewton method for mode finding in Laplace approximation */
		int max_number_lr_shrinkage_steps_newton_ = 20;
		/*! \brief Maximal number of steps for which learning rate shrinkage is done in the quasi-Newton method for mode finding in Laplace approximation */
		int MAX_NUMBER_LR_SHRINKAGE_STEPS_QUASI_NEWTON_ = 20;
		/*! \brief If true, the mode can only change by 'MAX_CHANGE_MODE_NEWTON_' in Newton's method */
		bool cap_change_mode_newton_ = false;
		/*! \brief Maximally allowed change for mode in Newton's method for those likelihoods where a cap is enforced */
		double MAX_CHANGE_MODE_NEWTON_ = std::log(100.);
		/*! \brief If true, Armijo's condition is used to check whether there is sufficient increase during the mode finding */
		bool armijo_condition_ = true;
		/*! \brief Constant c for Armijo's condition. Needs to be in (0,1) */
		double c_armijo_ = 1e-4;

		// MATRIX INVERSION PROPERTIES
		/*! \brief Matrix inversion method */
		string_t matrix_inversion_method_;
		/*! \brief Maximal number of iterations for conjugate gradient algorithm */
		int cg_max_num_it_;
		/*! \brief Maximal number of iterations for conjugate gradient algorithm when being run as Lanczos algorithm for tridiagonalization */
		int cg_max_num_it_tridiag_;
		/*! \brief Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for parameter estimation */
		double cg_delta_conv_;
		/*! \brief Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for prediction */
		double cg_delta_conv_pred_;
		/*! \brief Number of random vectors (e.g., Rademacher) for stochastic approximation of the trace of a matrix */
		int num_rand_vec_trace_;
		/*! \brief If true, random vectors (e.g., Rademacher) for stochastic approximation of the trace of a matrix are sampled only once at the beginning of Newton's method for finding the mode in the Laplace approximation and are then reused in later trace approximations, otherwise they are sampled every time a trace is calculated */
		bool reuse_rand_vec_trace_;
		/*! \brief Seed number to generate random vectors (e.g., Rademacher) */
		int seed_rand_vec_trace_;
		/*! \brief Type of preconditioner used for conjugate gradient algorithms */
		string_t cg_preconditioner_type_;
		/*! \brief Rank of the FITC and pivoted Cholesky preconditioners in conjugate gradient algorithms */
		int fitc_piv_chol_preconditioner_rank_;
		/*! \brief Rank of the matrix for approximating predictive covariance matrices obtained using the Lanczos algorithm */
		int rank_pred_approx_matrix_lanczos_;
		/*! \brief Number of samples when simulation is used for calculating predictive variances */
		int nsim_var_pred_;
		/*! \brief If true, cg_max_num_it and cg_max_num_it_tridiag are reduced by 2/3 (multiplied by 1/3) for the mode finding of the Laplace approximation in the first gradient step when finding a learning rate that reduces the ll */
		bool reduce_cg_max_num_it_first_optim_step_ = true;
		/*! \brief Number of CG steps when the CG method was last run */
		int num_cg_steps_last_ = 0;
		/*! \brief Number of CG steps when the CG method was last run for SLQ */
		int num_cg_steps_tridiag_last_ = 0;

		//ITERATIVE MATRIX INVERSION + VECCIA APPROXIMATION
		//A) ROW-MAJOR MATRICES OF VECCIA APPROXIMATION
		/*! \brief Row-major matrix of the Veccia-matrix B*/
		sp_mat_rm_t B_rm_;
		/*! \brief Row-major matrix of the Veccia-matrix D_inv*/
		sp_mat_rm_t D_inv_rm_;
		/*! \brief Row-major matrix of B^T D^(-1)*/
		sp_mat_rm_t B_t_D_inv_rm_;

		//ITERATIVE MATRIX INVERSION + RANDOM EFFECTS
		/*! \brief Row-major version of Inverse covariance matrix of latent random effect. */
		sp_mat_rm_t SigmaI_plus_ZtWZ_rm_;
		/*! Matrix to store (Sigma^(-1) + Z^T W Z)^(-1) (z_1, ..., z_t) calculated in CGTridiagRandomEffects() for later use in the stochastic trace approximation when calculating the gradient*/
		den_mat_t SigmaI_plus_ZtWZ_inv_RV_;
		/*! \brief For SSOR preconditioner - lower.triangular(Sigma^-1 + Z^T W Z) times diag(Sigma^-1 + Z^T W Z)^(-0.5)*/
		sp_mat_rm_t P_SSOR_L_D_sqrt_inv_rm_;
		/*! \brief For SSOR preconditioner - diag(Sigma^-1 + Z^T W Z)^(-1)*/
		vec_t P_SSOR_D_inv_;
		/*! \brief For ZIC preconditioner - sparse cholesky factor L of matrix L L^T approx (Sigma^-1 + Z^T W Z)*/
		sp_mat_rm_t L_SigmaI_plus_ZtWZ_rm_;
		/*! \brief For diagonal preconditioner - diag(Sigma^-1 + Z^T W Z)^(-1)*/
		vec_t SigmaI_plus_ZtWZ_inv_diag_;

		//B) RANDOM VECTOR VARIABLES
		/*! Random number generator used to generate rand_vec_trace_I_ */
		RNG_t cg_generator_;
		/*! If the seed of the random number generator cg_generator_ is set, cg_generator_seeded_ is set to true */
		bool cg_generator_seeded_ = false;
		/*! See counter for parallel RNG */
		uint64_t cg_generator_counter_ = 0;
		/*! If reuse_rand_vec_trace_ is true and rand_vec_trace_I_ has been generated for the first time, then saved_rand_vec_trace_ is set to true */
		bool saved_rand_vec_trace_ = false;
		/*! Matrix of random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I, r_i is of dimension num_data, and t = num_rand_vec_trace_ */
		den_mat_t rand_vec_trace_I_;
		/*! Matrix of random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I, r_i is of dimension fitc_piv_chol_preconditioner_rank_, and t = num_rand_vec_trace_. This is used only if cg_preconditioner_type_ == "pivoted_cholesky" */
		den_mat_t rand_vec_trace_I2_;
		/*! Matrix of random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I, r_i is of dimension fitc_piv_chol_preconditioner_rank_, and t = num_rand_vec_trace_. This is used only if cg_preconditioner_type_ == "pivoted_cholesky" */
		den_mat_t rand_vec_trace_I3_;
		/*! Matrix Z of random vectors (z_1, ..., z_t) with Cov(z_i) = P (P being the preconditioner matrix), z_i is of dimension num_data, and t = num_rand_vec_trace_ */
		den_mat_t rand_vec_trace_P_;
		/*! Matrix to store (Sigma^(-1) + W)^(-1) (z_1, ..., z_t) calculated in CGTridiagVecchiaLaplace() for later use in the stochastic trace approximation when calculating the gradient*/
		den_mat_t SigmaI_plus_W_inv_Z_;
		/*! Matrix to store (W^(-1) + Sigma)^(-1) (z_1, ..., z_t) calculated in CGTridiagVecchiaLaplace_Version_SigmaPlusWinv() for later use in the stochastic trace approximation when calculating the gradient*/
		den_mat_t WI_plus_Sigma_inv_Z_;

		//C) PRECONDITIONER VARIABLES
		/*! \brief piv_chol_on_Sigma: matrix of dimension nxk with rank(Sigma_L_k_) <= fitc_piv_chol_preconditioner_rank generated in re_model_template.h*/
		den_mat_t Sigma_L_k_;
		/*! \brief piv_chol_on_Sigma: Factor E of matrix EE^T = (I_k + Sigma_L_k_^T W Sigma_L_k_)*/
		chol_den_mat_t chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_;
		/*! \brief Sigma_inv_plus_BtWB (P = B^T (D^(-1) + W) B): matrix that contains the product (D^(-1) + W) B */
		sp_mat_rm_t D_inv_plus_W_B_rm_;
		/*! \brief zero_infill_incomplete_cholesky (P = L^T L): sparse cholesky factor L of matrix L^T L =  B^T D^(-1) B + W*/
		sp_mat_rm_t L_SigmaI_plus_W_rm_;
		/*! \brief B of vecchia preconditioner */
		sp_mat_rm_t B_vecchia_pc_rm_;
		/*! \brief D_inv of vecchia preconditioner */
		sp_mat_t D_inv_vecchia_pc_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Diagonal of residual covariance matrix (Preconditioner) */
		vec_t diagonal_approx_preconditioner_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Inverse of diagonal of residual covariance matrix (Preconditioner) */
		vec_t diagonal_approx_inv_preconditioner_;
		/*! \brief Key: labels of independent realizations of REs/GPs, values: Cholesky decompositions of matrix sigma_ip + cross_cov^T * D^-1 * cross_cov used in Woodbury identity where D is given by the Preconditioner */
		chol_den_mat_t chol_fact_woodbury_preconditioner_;
		/*! \brief Sigma_ip */
		den_mat_t sigma_ip_stable_;
		/*! \brief Sigma_ip^(-1/2) Sigma_mn */
		den_mat_t chol_ip_cross_cov_;
		/*! \brief Cholesky decompositions of inducing points matrix sigma_ip */
		chol_den_mat_t chol_fact_sigma_ip_;
		/*! \brief Doubled Woodbury */
		den_mat_t sigma_woodbury_woodbury_;
		/*! \brief Cholesky decompositions of doubled Woodbury */
		chol_den_mat_t chol_fact_sigma_woodbury_woodbury_;
		/*! \brief Matrix D^(-1) B Sigma_nm */
		den_mat_t D_inv_B_cross_cov_;
		/*! \brief Row-major matrix D^(-1) B*/
		sp_mat_rm_t D_inv_B_rm_;

		// VARIABLES FOR SAMPLING FROM THE LAPLACE-APPROXIMATED POSTERIOR
		/*! \brief If true, samples are generated from the Laplace-approximated posterior after the mode is found */
		bool sample_from_posterior_after_mode_finding_ = false;
		/*! \brief True if samples have been generated from the Laplace-approximated posterior */
		bool rand_vec_sim_post_calculated_ = false;
		/*! \brief Number of random vectors (e.g., Rademacher) for sampling from the Laplace-approximated posterior */
		int num_rand_vec_sim_post_ = 100;
		/*! Matrix of random vectors (r_1, r_2, r_3, ...) with samples from the Laplace-approximated posterior */
		den_mat_t rand_vec_sim_post_;
		/*! Matrix with iid normal random vectors for sampling from the Laplace-approximated posterior */
		den_mat_t rand_vec_I_sim_post_;
		/*! Second set of iid normal random vectors for sampling from the Laplace-approximated posterior (e.g., for iterative methods) */
		den_mat_t rand_vec_I_2_sim_post_;
		/*! Second set of iid normal random vectors for sampling from the Laplace-approximated posterior (e.g., for iterative methods) */
		den_mat_t rand_vec_I_3_sim_post_;
		/*! If true, the mean is added for sampling from the Laplace-approximated posterior (otherwise the mean of the samples is 0)  */
		bool add_mean_sim_post_ = true;
		/*! Constant by whose square the the covariance of the posterior is multiplied with */
		double c_mult_sim_post_ = 1.;
		/*! Vector of length num_rand_vec_sim_post_ with sums of squared random vectors from the Laplace-approximated posterior minus the mean / mode */
		std::vector<double> sum_sq_rand_vec_sim_post_zero_mean_;
		/*! \brief If true, iid normal random vectors for sampling from the Laplace-approximated posterior are sampled only once and are then reused later */
		bool reuse_rand_vec_I_sim_post_ = true;
		/*! If reuse_rand_vec_I_sim_post_ is true and rand_vec_I_sim_post_ has been generated for the first time, then sampled_rand_vec_I_sim_post_ is set to true */
		bool sampled_rand_vec_I_sim_post_ = false;

		/*! \brief Order of the (adaptive) Gauss-Hermite quadrature */
		int order_GH_ = 30;
		/*!
		\brief Nodes and weights for the Gauss-Hermite quadrature
		Source: https://keisan.casio.com/exec/system/1281195844

		Can also be computed using the following Python code:
		import numpy as np
		from scipy.special import roots_hermite

		N = 30  # Number of quadrature points
		nodes, weights = roots_hermite(N)
		adaptive_weights = weights * np.exp(nodes**2)

		*/
		const std::vector<double> GH_nodes_ = { -6.863345293529891581061,
										-6.138279220123934620395,
										-5.533147151567495725118,
										-4.988918968589943944486,
										-4.48305535709251834189,
										-4.003908603861228815228,
										-3.544443873155349886925,
										-3.099970529586441748689,
										-2.667132124535617200571,
										-2.243391467761504072473,
										-1.826741143603688038836,
										-1.415527800198188511941,
										-1.008338271046723461805,
										-0.6039210586255523077782,
										-0.2011285765488714855458,
										0.2011285765488714855458,
										0.6039210586255523077782,
										1.008338271046723461805,
										1.415527800198188511941,
										1.826741143603688038836,
										2.243391467761504072473,
										2.667132124535617200571,
										3.099970529586441748689,
										3.544443873155349886925,
										4.003908603861228815228,
										4.48305535709251834189,
										4.988918968589943944486,
										5.533147151567495725118,
										6.138279220123934620395,
										6.863345293529891581061 };
		const std::vector<double> GH_weights_ = { 2.908254700131226229411E-21,
										2.8103336027509037088E-17,
										2.87860708054870606219E-14,
										8.106186297463044204E-12,
										9.1785804243785282085E-10,
										5.10852245077594627739E-8,
										1.57909488732471028835E-6,
										2.9387252289229876415E-5,
										3.48310124318685523421E-4,
										0.00273792247306765846299,
										0.0147038297048266835153,
										0.0551441768702342511681,
										0.1467358475408900997517,
										0.2801309308392126674135,
										0.386394889541813862556,
										0.3863948895418138625556,
										0.2801309308392126674135,
										0.1467358475408900997517,
										0.0551441768702342511681,
										0.01470382970482668351528,
										0.002737922473067658462989,
										3.48310124318685523421E-4,
										2.938725228922987641501E-5,
										1.579094887324710288346E-6,
										5.1085224507759462774E-8,
										9.1785804243785282085E-10,
										8.10618629746304420399E-12,
										2.87860708054870606219E-14,
										2.81033360275090370876E-17,
										2.9082547001312262294E-21 };
		const std::vector<double> adaptive_GH_weights_ = { 0.83424747101276179534,
										0.64909798155426670071,
										0.56940269194964050397,
										0.52252568933135454964,
										0.491057995832882696506,
										0.46837481256472881677,
										0.45132103599118862129,
										0.438177022652683703695,
										0.4279180629327437485828,
										0.4198950037368240886418,
										0.413679363611138937184,
										0.4089815750035316024972,
										0.4056051233256844363121,
										0.403419816924804022553,
										0.402346066701902927115,
										0.4023460667019029271154,
										0.4034198169248040225528,
										0.4056051233256844363121,
										0.4089815750035316024972,
										0.413679363611138937184,
										0.4198950037368240886418,
										0.427918062932743748583,
										0.4381770226526837037,
										0.45132103599118862129,
										0.46837481256472881677,
										0.4910579958328826965056,
										0.52252568933135454964,
										0.56940269194964050397,
										0.64909798155426670071,
										0.83424747101276179534 };

		const char* NA_OR_INF_WARNING_ = "Mode finding algorithm for Laplace approximation: NA or Inf occurred. "
			"This is not necessary a problem as it might have been the cause of a too large learning rate which, "
			"consequently, might have been decreased by the optimization algorithm ";
		const char* CANNOT_CALC_STDEV_ERROR_ = "Cannot calculate standard deviations for the regression coefficients since "
			"the marginal likelihood is numerically unstable (NA or Inf) in a neighborhood of the optimal values. "
			"The likely reason for this is that the marginal likelihood is very flat. "
			"If you include an intercept in your model, you can try estimating your model without an intercept (and excluding variables that are almost constant) ";
		const char* NA_OR_INF_ERROR_ = "NA or Inf occurred in the mode finding algorithm for the Laplace approximation ";
		const char* NO_INCREASE_IN_MLL_WARNING_ = "Mode finding algorithm for Laplace approximation: "
			"The convergence criterion (log-likelihood + log-prior) has decreased and the algorithm has been terminated ";
		const char* NO_CONVERGENCE_WARNING_ = "Algorithm for finding mode for Laplace approximation has not "
			"converged after the maximal number of iterations ";
		const char* CG_NA_OR_INF_WARNING_GRADIENT_ = "NA or Inf occured in the conjugate gradient (CG) algorithm when calculating gradients. The CG algorithm was terminated early ";
		const char* CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_ = "NA or Inf occured in the conjugate gradient (CG) algorithm when sampling from the posterior. The CG algorithm was terminated early ";

	};//end class Likelihood

}  // namespace GPBoost

// Definitions of the Laplace approximation member functions declared above
#include <GPBoost/likelihoods_laplace.h>
#include <GPBoost/likelihoods_aux_pars.h>
#include <GPBoost/likelihoods_predict.h>

#endif   // GPB_LIKELIHOODS_
