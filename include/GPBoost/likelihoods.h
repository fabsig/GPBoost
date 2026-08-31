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
			likelihood = ParseLikelihoodAliasVarianceCorrection(likelihood);
			likelihood = ParseLikelihoodAliasModeFindingMethod(likelihood);
			likelihood = ParseLikelihoodAliasApproximationType(likelihood);
			likelihood = ParseLikelihoodAliasEstimateAdditionalPars(likelihood);
			likelihood = ParseLikelihoodAlias(likelihood);
			if (SUPPORTED_LIKELIHOODS_.find(likelihood) == SUPPORTED_LIKELIHOODS_.end()) {
				Log::REFatal("Likelihood of type '%s' is not supported ", likelihood.c_str());
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
			double* aux_pars_trans) {
			for (int i = 0; i < num_aux_pars_; ++i) {
				aux_pars_trans[i] = aux_pars_orig[i];
			}
			if (IsEGPDLikelihood()) {
				if (!(std::isfinite(aux_pars_orig[0]) && aux_pars_orig[0] > -0.5)) Log::REFatal("For likelihood='%s', the 'shape' parameter must be finite and larger than -0.5, found %g ", likelihood_type_.c_str(), aux_pars_orig[0]);
				aux_pars_trans[0] = aux_pars_orig[0] + 0.5;
				for (int i = 1; i < num_aux_pars_; ++i) {
					if (likelihood_type_ == "egpd_power_mixture" && i == 3) continue;
					if (!(std::isfinite(aux_pars_orig[i]) && aux_pars_orig[i] > 0.)) Log::REFatal("For likelihood='%s', the '%s' parameter must be finite and larger than 0, found %g ", likelihood_type_.c_str(), names_aux_pars_[i].c_str(), aux_pars_orig[i]);
				}
				if (likelihood_type_ == "egpd_power_mixture") {
					const double p = aux_pars_orig[3];
					if (!(std::isfinite(p) && p > 0. && p < 1.)) Log::REFatal("For likelihood='egpd_power_mixture', the 'p' parameter must be finite and strictly between 0 and 1, found %g ", p);
					const double log_odds = std::log(p) - std::log1p(-p);
					aux_pars_trans[3] = std::exp(log_odds);
					if (!(std::isfinite(aux_pars_trans[3]) && aux_pars_trans[3] > 0.)) Log::REFatal("For likelihood='egpd_power_mixture', the transformed 'p' parameter is not representable ");
				}
			}
			else if (IsHurdleEGPD()) {
				// Base EGPD transform on the leading NumEGPDBaseAuxPars() parameters, plus the structural-zero p0 (odds) at the end.
				const int nb = NumEGPDBaseAuxPars();
				const int ip0 = num_aux_pars_ - 1;
				if (!(std::isfinite(aux_pars_orig[0]) && aux_pars_orig[0] > -0.5)) Log::REFatal("For likelihood='%s', the 'shape' parameter must be finite and larger than -0.5, found %g ", likelihood_type_.c_str(), aux_pars_orig[0]);
				aux_pars_trans[0] = aux_pars_orig[0] + 0.5;
				for (int i = 1; i < nb; ++i) {
					if (likelihood_type_ == "hurdle_egpd_power_mixture" && i == 3) continue;// the mixture weight 'p' is handled below
					if (!(std::isfinite(aux_pars_orig[i]) && aux_pars_orig[i] > 0.)) Log::REFatal("For likelihood='%s', the '%s' parameter must be finite and larger than 0, found %g ", likelihood_type_.c_str(), names_aux_pars_[i].c_str(), aux_pars_orig[i]);
				}
				if (likelihood_type_ == "hurdle_egpd_power_mixture") {
					const double p = aux_pars_orig[3];
					if (!(std::isfinite(p) && p > 0. && p < 1.)) Log::REFatal("For likelihood='hurdle_egpd_power_mixture', the 'p' parameter must be finite and strictly between 0 and 1, found %g ", p);
					aux_pars_trans[3] = std::exp(std::log(p) - std::log1p(-p));
				}
				if (!(aux_pars_orig[ip0] > 0. && aux_pars_orig[ip0] < 1.)) Log::REFatal("The '%s' parameter (= %g) needs to be larger than 0 and smaller than 1 ", names_aux_pars_[ip0].c_str(), aux_pars_orig[ip0]);
				aux_pars_trans[ip0] = aux_pars_orig[ip0] / (1. - aux_pars_orig[ip0]);
			}//end hurdle EGPD variants
			else if (IsHurdleRegression() && HasEGPDBase()) {
				// Regression hurdle EGPD: the auxiliary parameters are exactly the base EGPD parameters (no structural-zero p0).
				if (!(std::isfinite(aux_pars_orig[0]) && aux_pars_orig[0] > -0.5)) Log::REFatal("For likelihood='%s', the 'shape' parameter must be finite and larger than -0.5, found %g ", likelihood_type_.c_str(), aux_pars_orig[0]);
				aux_pars_trans[0] = aux_pars_orig[0] + 0.5;
				const bool mix = EGPDBaseType() == "egpd_power_mixture";
				for (int i = 1; i < num_aux_pars_; ++i) {
					if (mix && i == 3) continue;
					if (!(std::isfinite(aux_pars_orig[i]) && aux_pars_orig[i] > 0.)) Log::REFatal("For likelihood='%s', the '%s' parameter must be finite and larger than 0, found %g ", likelihood_type_.c_str(), names_aux_pars_[i].c_str(), aux_pars_orig[i]);
				}
				if (mix) {
					const double p = aux_pars_orig[3];
					if (!(std::isfinite(p) && p > 0. && p < 1.)) Log::REFatal("For likelihood='%s', the 'p' parameter must be finite and strictly between 0 and 1, found %g ", likelihood_type_.c_str(), p);
					aux_pars_trans[3] = std::exp(std::log(p) - std::log1p(-p));
				}
			}//end hurdle regression EGPD
			else if (likelihood_type_ == "hurdle_gamma" || likelihood_type_ == "hurdle_lognormal" || likelihood_type_ == "zero_inflated_negative_binomial" ||
				likelihood_type_ == "zero_inflated_negative_binomial_1") {
				if (!(aux_pars_orig[1] > 0. && aux_pars_orig[1] < 1.)) {
					Log::REFatal("The '%s' parameter (= %g) needs to be larger than 0 and smaller than 1 ", names_aux_pars_[1].c_str(), aux_pars_orig[1]);
				}
				aux_pars_trans[1] = aux_pars_orig[1] / (1. - aux_pars_orig[1]);
			}//end hurdle_gamma / hurdle_lognormal / zero_inflated_negative_binomial(_1)
			else if (likelihood_type_ == "zero_inflated_poisson") {
				if (!(aux_pars_orig[0] > 0. && aux_pars_orig[0] < 1.)) {
					Log::REFatal("The '%s' parameter (= %g) needs to be larger than 0 and smaller than 1 ", names_aux_pars_[0].c_str(), aux_pars_orig[0]);
				}
				aux_pars_trans[0] = aux_pars_orig[0] / (1. - aux_pars_orig[0]);
			}//end likelihood_type_ == "zero_inflated_poisson"
			else if (likelihood_type_ == "zoctn") {
				aux_pars_trans[1] = std::exp(aux_pars_orig[1]);
			}
			else if (likelihood_type_ == "tweedie") {
				if (!(aux_pars_orig[1] > TWEEDIE_POWER_LOWER_ && aux_pars_orig[1] < TWEEDIE_POWER_UPPER_) || !std::isfinite(aux_pars_orig[1])) {
					Log::REFatal("For likelihood='tweedie', the initial power must satisfy %g < p < %g. Found p = %g.", TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_, aux_pars_orig[1]);
				}
				aux_pars_trans[1] = (aux_pars_orig[1] - TWEEDIE_POWER_LOWER_) / (TWEEDIE_POWER_UPPER_ - aux_pars_orig[1]);
			}
		}

		/*!
		* \brief Back-transform aux_pars
		* \param aux_pars_trans Transformed aux_pars
		* \param aux_pars_orig Original aux_pars
		*/
		void BackTransformAuxPars(const double* aux_pars_trans,
			double* aux_pars_orig) {
			for (int i = 0; i < num_aux_pars_; ++i) {
				aux_pars_orig[i] = aux_pars_trans[i];
			}
			if (IsEGPDLikelihood()) {
				for (int i = 0; i < num_aux_pars_; ++i) if (!(std::isfinite(aux_pars_trans[i]) && aux_pars_trans[i] > 0.)) Log::REFatal("BackTransformAuxPars: transformed '%s' parameter must be finite and larger than 0, found %g ", names_aux_pars_[i].c_str(), aux_pars_trans[i]);
				aux_pars_orig[0] = aux_pars_trans[0] - 0.5;
				if (!(aux_pars_orig[0] > -0.5)) Log::REFatal("BackTransformAuxPars: transformed 'shape' rounded onto the forbidden -0.5 boundary ");
				if (likelihood_type_ == "egpd_power_mixture") {
					const double odds = aux_pars_trans[3];
					aux_pars_orig[3] = odds >= 1. ? 1. / (1. + 1. / odds) : odds / (1. + odds);
					if (!(aux_pars_orig[3] > 0. && aux_pars_orig[3] < 1.) || !std::isfinite(aux_pars_trans[1] + aux_pars_trans[2])) Log::REFatal("BackTransformAuxPars: EGPD mixture parameters overflowed or rounded onto a boundary ");
				}
			}
			else if (IsHurdleEGPD()) {
				const int ip0 = num_aux_pars_ - 1;
				for (int i = 0; i < num_aux_pars_; ++i) if (!(std::isfinite(aux_pars_trans[i]) && aux_pars_trans[i] > 0.)) Log::REFatal("BackTransformAuxPars: transformed '%s' parameter must be finite and larger than 0, found %g ", names_aux_pars_[i].c_str(), aux_pars_trans[i]);
				aux_pars_orig[0] = aux_pars_trans[0] - 0.5;
				if (!(aux_pars_orig[0] > -0.5)) Log::REFatal("BackTransformAuxPars: transformed 'shape' rounded onto the forbidden -0.5 boundary ");
				if (likelihood_type_ == "hurdle_egpd_power_mixture") {
					const double odds = aux_pars_trans[3];
					aux_pars_orig[3] = odds >= 1. ? 1. / (1. + 1. / odds) : odds / (1. + odds);
					if (!(aux_pars_orig[3] > 0. && aux_pars_orig[3] < 1.)) Log::REFatal("BackTransformAuxPars: EGPD mixture 'p' overflowed or rounded onto a boundary ");
				}
				aux_pars_orig[ip0] = aux_pars_trans[ip0] / (1. + aux_pars_trans[ip0]);// p0 from odds
			}//end hurdle EGPD variants
			else if (IsHurdleRegression() && HasEGPDBase()) {
				for (int i = 0; i < num_aux_pars_; ++i) if (!(std::isfinite(aux_pars_trans[i]) && aux_pars_trans[i] > 0.)) Log::REFatal("BackTransformAuxPars: transformed '%s' parameter must be finite and larger than 0, found %g ", names_aux_pars_[i].c_str(), aux_pars_trans[i]);
				aux_pars_orig[0] = aux_pars_trans[0] - 0.5;
				if (!(aux_pars_orig[0] > -0.5)) Log::REFatal("BackTransformAuxPars: transformed 'shape' rounded onto the forbidden -0.5 boundary ");
				if (EGPDBaseType() == "egpd_power_mixture") {
					const double odds = aux_pars_trans[3];
					aux_pars_orig[3] = odds >= 1. ? 1. / (1. + 1. / odds) : odds / (1. + odds);
					if (!(aux_pars_orig[3] > 0. && aux_pars_orig[3] < 1.)) Log::REFatal("BackTransformAuxPars: EGPD mixture 'p' overflowed or rounded onto a boundary ");
				}
			}//end hurdle regression EGPD
			else if (likelihood_type_ == "hurdle_gamma") {
				if (!(aux_pars_trans[1] > 0.)) {
					Log::REFatal("BackTransformAuxPars: the transformed '%s' parameter (= %g) needs to be larger than 0 ", names_aux_pars_[1].c_str(), aux_pars_trans[1]);
				}
				aux_pars_orig[1] = aux_pars_trans[1] / (1. + aux_pars_trans[1]);
			}//end likelihood_type_ == "hurdle_gamma"
			else if (likelihood_type_ == "hurdle_lognormal" || likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
				if (!(aux_pars_trans[1] > 0.)) {
					Log::REFatal("BackTransformAuxPars: the transformed '%s' parameter (= %g) needs to be larger than 0 ", names_aux_pars_[1].c_str(), aux_pars_trans[1]);
				}
				aux_pars_orig[1] = aux_pars_trans[1] / (1. + aux_pars_trans[1]);
			}//end hurdle_lognormal / zero_inflated_negative_binomial(_1)
			else if (likelihood_type_ == "zero_inflated_poisson") {
				if (!(aux_pars_trans[0] > 0.)) {
					Log::REFatal("BackTransformAuxPars: the transformed '%s' parameter (= %g) needs to be larger than 0 ", names_aux_pars_[0].c_str(), aux_pars_trans[0]);
				}
				aux_pars_orig[0] = aux_pars_trans[0] / (1. + aux_pars_trans[0]);
			}//end likelihood_type_ == "zero_inflated_poisson"
			else if (likelihood_type_ == "zoctn") {
				if (!(aux_pars_trans[1] > 0.)) {
					Log::REFatal("BackTransformAuxPars: the transformed '%s' parameter (= %g) needs to be larger than 0 ", names_aux_pars_[1].c_str(), aux_pars_trans[1]);
				}
				aux_pars_orig[1] = std::log(aux_pars_trans[1]);
			}
			else if (likelihood_type_ == "tweedie") {
				const auto transform = TransformTweediePowerFromQ(aux_pars_trans[1], TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_);
				if (!std::isfinite(transform.p)) Log::REFatal("BackTransformAuxPars: transformed Tweedie power parameter must be finite and > 0, found %g.", aux_pars_trans[1]);
				aux_pars_orig[1] = transform.p;
			}
		}

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
			string_t likelihood = ParseLikelihoodAlias(type);
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
			double& log_sigma_anchor) const {
			const double eps_p = 1e-6;// clipping for probabilities
			double W = 0., W0 = 0., Wpos = 0., sum_u = 0., sum_u_sq = 0.;
#pragma omp parallel for schedule(static) reduction(+:W, W0, Wpos, sum_u, sum_u_sq)
			for (data_size_t i = 0; i < num_data; ++i) {
				const double w = has_weights_ ? weights_ptr[i] : 1.0;
				W += w;
				if (y_data[i] <= 0.) {
					W0 += w;
				}
				else {
					const double u = std::exp((1.0 / lambda) * std::log(y_data[i]));// u = y^(1/lambda) computed stably
					Wpos += w;
					sum_u += w * u;
					sum_u_sq += w * u * u;
				}
			}
			const double p0 = std::min(std::max(W0 / W, eps_p), 1. - eps_p);
			const double a = GPBoost::normalQF(p0);// a = Phi^{-1}(p0) = -mu / sigma
			const double tau = GPBoost::normalPDF(a) / std::max(1. - GPBoost::normalCDF(a), 1e-12);
			const double var_factor = std::max(1. + a * tau - tau * tau, 1e-6);// Var[X | X > 0] / sigma^2
			double sigma = 1., mu = 0.;
			if (Wpos > 0.) {
				const double mean_u = sum_u / Wpos;
				const double var_u = std::max(sum_u_sq / Wpos - mean_u * mean_u, 1e-12);
				sigma = std::sqrt(var_u / var_factor);
				mu = mean_u - sigma * tau;
			}
			if (!(sigma > 0.) || !std::isfinite(sigma)) {
				sigma = 1.;
			}
			if (!std::isfinite(mu)) {
				mu = 0.;
			}
			mu_anchor = mu;
			log_sigma_anchor = std::log(sigma);
		}//end ZeroCensPowNormHeteroAnchors

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
			const double* weights = nullptr) const {
			//'weights' can be used to pass weights that do not correspond to 'weights_' of this object
			//	(e.g. the weights of ALL clusters and not only the ones of the cluster of this object)
			const double* weights_ptr = (weights != nullptr) ? weights : weights_;
			CHECK(rand_eff_var > 0.);
			double init_intercept = 0.;
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit" ||
				likelihood_type_ == "binomial_probit" || likelihood_type_ == "binomial_logit" ||
				likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial" || likelihood_type_ == "zero_one_censored_transformed_beta" || 
				likelihood_type_ == "quasi_bernoulli_probit" || likelihood_type_ == "quasi_bernoulli_logit") {
				double sw = 0.0, swy = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sw,swy)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_ptr[i] : 1.0;
					swy += w * y_data[i];
					sw += w;
				}
				double pavg = (swy > 0.0 && sw > 0.0) ? (swy / sw) : 0.5;
				const double eps = 1e-12;
				pavg = std::min(std::max(pavg, eps), 1.0 - eps);
				if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit" ||
					likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial" || likelihood_type_ == "zero_one_censored_transformed_beta" || 
					likelihood_type_ == "quasi_bernoulli_logit") {
					init_intercept = GPBoost::logit(pavg);
				}
				else if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "binomial_probit" || 
					likelihood_type_ == "quasi_bernoulli_probit") {
					init_intercept = GPBoost::normalQF(pavg);
				}
				else {
					Log::REFatal("FindInitialIntercept not implemented for likelihood = '%s' ", likelihood_type_.c_str());
				}
				init_intercept = std::min(std::max(init_intercept, -3.0), 3.0); // avoid too small / large initial intercepts for better numerical stability
			}
			else if (IsHurdlePositive()) {
				double sw = 0.0, avg = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:avg, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						if (y_data[i] > 0.) {
							const double w = has_weights_ ? weights_ptr[i] : 1.0;
							avg += w * y_data[i];
							sw += w;
						}
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:avg, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						if (y_data[i] > 0.) {
							const double w = has_weights_ ? weights_ptr[i] : 1.0;
							avg += w * y_data[i] / std::exp(fixed_effects[i]);
							sw += w;
						}
					}
				}
				avg /= sw;
				avg = std::max(avg, 1e-12);
				init_intercept = std::log(avg) - 0.5 * rand_eff_var;
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" || IsEGPDLikelihood() ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				likelihood_type_ == "lognormal" || IsZeroInflatedCount()) {
				// For zero-inflated counts, mean(y) = (1 - p0) * mu, so this underestimates the count-component mean mu;
				// this is only a finite starting value and is refined by the optimizer.
				double sw = 0.0, avg = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:avg, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						avg += w * y_data[i];
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:avg, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						avg += w * y_data[i] / std::exp(fixed_effects[i]);
						sw += w;
					}
				}
				avg /= sw;
				avg = std::max(avg, 1e-12);
				init_intercept = std::log(avg) - 0.5 * rand_eff_var; // log-normal distribution: mean of exp(beta_0 + Zb) = exp(beta_0 + 0.5 * sigma^2) => use beta_0 = mean(y) - 0.5 * sigma^2
			}
			else if (likelihood_type_ == "t") {
				//use the median as robust initial estimate
				std::vector<double> y_v;//for calculating the median
				if (fixed_effects == nullptr) {
					y_v.assign(y_data, y_data + num_data);
				}
				else {
					y_v = std::vector<double>(num_data);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						y_v[i] = y_data[i] - fixed_effects[i];
					}
				}
				if (has_weights_) {
					init_intercept = GPBoost::CalculateWeightedQuantile(y_v, weights_ptr, 0.5);//weighted median
				}
				else {
					init_intercept = GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(y_v);
				}
			}//end "t"
			else if (IsGaussianLikelihood() || (IsGaussianHeteroscedastic() && ind_set_re == 0)) {
				double sw = 0.0;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:init_intercept, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						init_intercept += w * y_data[i];
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:init_intercept, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						init_intercept += w * (y_data[i] - fixed_effects[i]);
						sw += w;
					}
				}
				init_intercept /= sw;
			}//end "gaussian"
			else if (IsGaussianHeteroscedastic() && ind_set_re == 1) {
				double sw = 0.0, avg = 0., sum_sq = 0., avg_exp_var_offset = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:avg, sum_sq, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						avg += w * y_data[i];
						sum_sq += w * y_data[i] * y_data[i];
						sw += w;
					}
					avg_exp_var_offset = 1.;
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:avg, sum_sq, sw, avg_exp_var_offset)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						double y_min_FE = y_data[i] - fixed_effects[i];
						avg += w * y_min_FE;
						sum_sq += w * y_min_FE * y_min_FE;
						avg_exp_var_offset += w * std::exp(fixed_effects[i + num_data]);
						sw += w;
					}
					avg_exp_var_offset /= sw;
				}
				avg /= sw;
				double avg_sq = avg * avg;
				double sample_var = std::max((sum_sq - sw * avg_sq) / (sw - 1), 1e-8);
				double sample_error_var = sample_var - rand_eff_var;
				if (sample_error_var < 1e-6) {
					sample_error_var = 1e-6;
				}
				init_intercept = std::log(sample_error_var) - std::log(avg_exp_var_offset);
			}//end gaussian_heteroscedastic
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
				// Strategy:
				//  1) Use fraction of zeros to anchor -mu/sigma via probit inversion.
				//  2) Use positive observations (de-powered by 1/lambda) to anchor the right tail.
				//  3) Do one Newton step using the exact score and information at the blended anchor
				const double sigma = aux_pars_[0];
				const double lambda = aux_pars_[1];
				const double eps_p = 1e-6;// clipping for probabilities
				const double eps_I = 1e-12;// clipping for information denominator
				double W0 = 0.0, Wpos = 0.0, W = 0.0;
				double sum_w_u_pos = 0.0;
#pragma omp parallel for schedule(static) reduction(+:W, W0, Wpos, sum_w_u_pos)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_ptr[i] : 1.0;
					const double yi = y_data[i];
					W += w;
					if (yi <= 0.0) {
						W0 += w;
					}
					else {
						Wpos += w;
						const double u = std::exp((1.0 / lambda) * std::log(yi));// u = y^(1/lambda) computed stably as exp((1/lambda)*log y)
						sum_w_u_pos += w * u;
					}
				}
				// 1) Zero-fraction anchor via probit inversion
				double p0 = W0 / W;
				p0 = std::min(std::max(p0, eps_p), 1.0 - eps_p);
				const double a0_0 = GPBoost::normalQF(p0); // Phi^{-1}(p0)
				const double mu_0 = -sigma * a0_0;
				// 2) Positive-mean anchor using truncated-normal identity
				double mu_1 = mu_0;
				if (Wpos > 0.0) {
					const double mean_u_pos = sum_w_u_pos / Wpos;
					const double Phi_a0 = GPBoost::normalCDF(a0_0);
					const double phi_a0 = GPBoost::normalPDF(a0_0);
					const double one_minus_Phi = std::max(1.0 - Phi_a0, 1e-12);
					const double corr = sigma * (phi_a0 / one_minus_Phi); // sigma * phi(a0)/(1-Phi(a0))
					mu_1 = mean_u_pos - corr;
				}
				// 3) Blend the two anchors
				double mu_tilde;
				if (Wpos == 0.0) {
					mu_tilde = mu_0;
				}
				else if (W0 == 0.0) {
					mu_tilde = mu_1;
				}
				else {
					mu_tilde = (W0 / W) * mu_0 + (Wpos / W) * mu_1;// Weighted blend; gives more weight to the dominant side
				}
				// 4) One Newton step: mu_new = mu_tilde + S/I				
				double S = 0.0, I = 0.0;// S and information I at mu_tilde
				// Terms that only depend on mu via a0 for the zero part
				const double a0 = -mu_tilde / sigma;
				const double Phi_a0 = GPBoost::normalCDF(a0);
				const double phi_a0 = GPBoost::normalPDF(a0);
				const double Phi_clipped = std::max(Phi_a0, 1e-12); // avoid division by zero
				const double r = phi_a0 / Phi_clipped; // r(a0) = phi/Phi
				// Zero-part contributions
				S += W0 * (-r / sigma);// Score: sum w * ( -r / sigma )
				I += W0 * (r * (a0 + r) / (sigma * sigma));// Info : sum w * ( r*(a0 + r) / sigma^2 )
				// Positive-part contributions: Score: sum w * ( (u - mu) / sigma^2 ), Info : sum w * ( 1 / sigma^2 )
#pragma omp parallel for schedule(static) reduction(+:S, I)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_ptr[i] : 1.0;
					const double yi = y_data[i];
					if (yi > 0.0) {
						const double u = std::exp((1.0 / lambda) * std::log(yi));
						S += w * ((u - mu_tilde) / (sigma * sigma));
						I += w * (1.0 / (sigma * sigma));
					}
				}
				if (I < eps_I) {
					init_intercept = mu_tilde;// Degenerate case: keep the blended anchor
				}
				else {
					init_intercept = mu_tilde + S / I;
				}
			}//end "zero_censored_power_transformed_normal"
			else if (IsZeroCensPowNormHetero()) {
				CHECK(ind_set_re == 0 || ind_set_re == 1);
				double mu_anchor, log_sigma_anchor;
				ZeroCensPowNormHeteroAnchors(y_data, num_data, weights_ptr, aux_pars_[0], mu_anchor, log_sigma_anchor);
				// The anchors are on the scale of the total location parameters -> subtract the pooled fixed effects offsets
				double sw = 0., off_mean = 0., off_log_sigma = 0.;
				if (fixed_effects != nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sw, off_mean, off_log_sigma)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						sw += w;
						off_mean += w * fixed_effects[i];
						off_log_sigma += w * fixed_effects[i + num_data];
					}
					off_mean /= sw;
					off_log_sigma /= sw;
				}
				init_intercept = (ind_set_re == 0) ? (mu_anchor - off_mean) : (log_sigma_anchor - off_log_sigma);
			}//end "zero_censored_power_transformed_normal_heteroscedastic"
			else if (likelihood_type_ == "zoctn") {
				const double sigma = aux_pars_[0];
				const double a = aux_pars_original_[1];
				const double b = aux_pars_[2];
				double sw_int = 0.0, sum_x = 0.0;
				double W = 0.0, W0 = 0.0, W1 = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sw_int,sum_x,W,W0,W1)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_ptr[i] : 1.0;
					const double yi = y_data[i];
					W += w;
					if (yi <= 0.0) {
						W0 += w;
					}
					else if (yi >= 1.0) {
						W1 += w;
					}
					else {
						const double logit_y = GPBoost::logit(yi);
						const double t = (logit_y - a) / b;
						const double x = GPBoost::sigmoid_stable(t); // approx latent Z in (0,1)
						if (fixed_effects == nullptr) {
							sum_x += w * x;
						}
						else {
							sum_x += w * (x - fixed_effects[i]);
						}
						sw_int += w;
					}
				}
				if (sw_int > 0.0) {
					init_intercept = sum_x / sw_int;// Use average of pseudo-latent x as initial mu
				}
				else {
					// Fallback: use zero/one fractions to anchor mu from the underlying normal model
					const double eps_p = 1e-6;
					bool have_mu0 = false, have_mu1 = false;
					double mu0 = 0.0, mu1 = 0.0;
					if (W > 0.0 && W0 > 0.0) {
						double p0 = W0 / W;
						if (p0 < eps_p) p0 = eps_p;
						if (p0 > 1.0 - eps_p) p0 = 1.0 - eps_p;
						const double a0 = GPBoost::normalQF(p0); // Phi^{-1}(p0)
						mu0 = -sigma * a0;
						have_mu0 = true;
					}
					if (W > 0.0 && W1 > 0.0) {
						double p1 = W1 / W;
						if (p1 < eps_p) p1 = eps_p;
						if (p1 > 1.0 - eps_p) p1 = 1.0 - eps_p;
						const double v = GPBoost::normalQF(p1); // Phi^{-1}(p1) = (mu-1)/sigma approx
						mu1 = 1.0 + sigma * v;
						have_mu1 = true;
					}
					if (have_mu0 && have_mu1) {
						init_intercept = 0.5 * (mu0 + mu1);
					}
					else if (have_mu0) {
						init_intercept = mu0;
					}
					else if (have_mu1) {
						init_intercept = mu1;
					}
					else {// Completely degenerate fallback						
						init_intercept = 0.0;
					}
				}
			}//end "zoctn"
			else if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
				// Use interior points 0<y<1 when available. If no interior points, fall back to mu ~ xi + 0.5
				const double xi = aux_pars_[1];
				double sumz = 0.0, cnt = 0.0;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sumz,cnt)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						const double yi = y_data[i];
						if (!TwoNumbersAreEqual<double>(yi, 0.) && !TwoNumbersAreEqual<double>(yi, 1.)) {
							sumz += w * (yi + xi);
							cnt += w;
						}
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:sumz,cnt)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						const double yi = y_data[i];
						if (!TwoNumbersAreEqual<double>(yi, 0.) && !TwoNumbersAreEqual<double>(yi, 1.)) {
							sumz += w * (yi + xi) / std::exp(fixed_effects[i]);
							cnt += w;
						}
					}
				}
				double mu = (cnt > 0.) ? (sumz / std::max(1., cnt)) : (xi + 0.5);
				mu = std::max(mu, 1e-8);
				init_intercept = std::log(mu);
			}//end "zero_one_censored_shifted_gamma"
			else if (likelihood_type_ == "asymmetric_laplace") {
				std::vector<double> y_v;//calculate sample quantile
				if (fixed_effects == nullptr) {
					y_v.assign(y_data, y_data + num_data);
				}
				else {
					y_v.resize(num_data);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						y_v[i] = y_data[i] - fixed_effects[i];
					}
				}

				if (has_weights_) {
					init_intercept = GPBoost::CalculateWeightedQuantile(y_v, weights_ptr, quantile_);
				}
				else {
					data_size_t pos_quant =
						static_cast<data_size_t>(std::ceil(quantile_ * num_data)) - 1;
					pos_quant = std::min(pos_quant, num_data - 1);

					std::nth_element(y_v.begin(), y_v.begin() + pos_quant, y_v.end());
					init_intercept = y_v[pos_quant];
				}
			}//end "asymmetric_laplace"
			else {
				NotSupportedForLikelihood(__func__);
			}
			return(init_intercept);
		}//end FindInitialIntercept

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
			const double* weights = nullptr) const {
			bool ret_val = false;
			if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" || IsEGPDLikelihood() ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" || IsZeroInflatedCount() ||
				IsGaussianHeteroscedastic() || likelihood_type_ == "lognormal" || IsHurdlePositive() ||
				IsZeroCensPowNorm() ||
				likelihood_type_ == "zoctn" || likelihood_type_ == "zero_one_censored_transformed_beta" ||
				likelihood_type_ == "zero_one_censored_shifted_gamma" ||
				likelihood_type_ == "asymmetric_laplace") {
				ret_val = true;
			}
			else {
				double beta_zero = FindInitialIntercept(y_data, num_data, rand_eff_var, fixed_effects, 0, weights);
				if (std::abs(beta_zero) > 0.1) {
					ret_val = true;
				}
			}
			return(ret_val);
		}

		/*!
		* \brief Determine initial value for additional likelihood parameters (e.g., shape for gamma)
		* \param y_data Response variable data
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		*/
		const double* FindInitialAuxPars(const double* y_data,
			const double* fixed_effects,
			const data_size_t num_data) {
			double sw = 0.0, avg = 0., avg_sq = 0., sample_var = 1.;
			if (likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				IsGaussianLikelihood()) {
				double sum_sq = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:avg, sum_sq, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						avg += w * y_data[i];
						sum_sq += w * y_data[i] * y_data[i];
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:avg, sum_sq, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						double y_min_FE = y_data[i] / std::exp(fixed_effects[i]);
						avg += w * y_min_FE;
						sum_sq += w * y_min_FE * y_min_FE;
						sw += w;
					}
				}
				avg /= sw;
				avg_sq = avg * avg;
				sample_var = std::max((sum_sq - sw * avg_sq) / (sw - 1), 1e-6);
			}
			if (likelihood_type_ == "gamma") {
				// Use a simple "MLE" approach for the shape parameter ignoring random and fixed effects and 
				//  using the approximation: ln(k) - digamma(k) approx = (1 + 1 / (6k + 1)) / (2k), where k = shape
				//  See https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation (as of 02.03.2023)
				sw = 0.0;
				double log_avg = 0., avg_log = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:log_avg, avg_log, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						log_avg += w * y_data[i];
						avg_log += w * std::log(y_data[i]);
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:log_avg, avg_log, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						log_avg += w * y_data[i] / std::exp(fixed_effects[i]);
						avg_log += w * (std::log(y_data[i]) - fixed_effects[i]);
						sw += w;
					}
				}
				log_avg /= sw;
				log_avg = std::log(log_avg);
				avg_log /= sw;
				const double s = std::max(log_avg - avg_log, 1e-8);
				aux_pars_[0] = (3. - s + std::sqrt((s - 3.) * (s - 3.) + 24. * s)) / (12. * s);
			}
			else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
				const double p = GetTweediePower();
				double sum_y = 0., sum_w = 0.;
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.;
					sum_y += w * y_data[i] / (fixed_effects == nullptr ? 1. : std::exp(fixed_effects[i]));
					sum_w += w;
				}
				const double base_mean = std::max(sum_y / sum_w, 1e-12);
				double pearson = 0.;
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.;
					const double mu = std::max(base_mean * (fixed_effects == nullptr ? 1. : std::exp(fixed_effects[i])), 1e-12);
					const double residual = y_data[i] - mu;
					pearson += w * residual * residual / std::pow(mu, p);
				}
				double phi = pearson / sum_w;
				if (!(phi > 0.) || !std::isfinite(phi)) phi = 1.;
				aux_pars_[0] = std::min(std::max(phi, 1e-6), 1e6);
			}
			else if (IsEGPDLikelihood()) {
				aux_pars_[0] = 0.5; // shape = 0 (exponential GPD base)
			}
			else if (likelihood_type_ == "negative_binomial") {
				// Use a method of moments estimator				
				if (sample_var <= avg) {
					aux_pars_[0] = 100 * avg_sq;//marginally no over-dispersion in data -> set shape parameter to a large value
					Log::REDebug("FindInitialAuxPars: the internally found initial estimate (MoM) for the shape parameter (%g) might be not very good as there is there is marginally no over-disperion in the data ", aux_pars_[0]);
				}
				else {
					aux_pars_[0] = avg_sq / (sample_var - avg);
				}
			}//end "negative_binomial"
			else if (likelihood_type_ == "negative_binomial_1") {
				//  Method‑of‑moments start value for the dispersion phi = (var − mu) / mu since var(y) = mu + phi * mu
				double phi_init = std::max((sample_var - avg) / avg, 1e-3);//clip below
				phi_init = std::min(phi_init, 100.0);//keep in a sane range
				aux_pars_[0] = phi_init;
			}//end negative_binomial_1
			else if (likelihood_type_ == "beta") {
				// method of moment estimator for the Beta precision
				// phi = mu (1–mu) / var – 1   where  mu = E[Y],  var = Var[Y]
				avg = 0.;
				sw = 0.0;
				double sum_sq = 0.;
#pragma omp parallel for schedule(static) reduction(+:avg, sum_sq, sw)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					avg += w * y_data[i];
					sum_sq += w * y_data[i] * y_data[i];
					sw += w;
				}
				avg /= sw;
				avg_sq = avg * avg;
				sample_var = std::max((sum_sq - sw * avg_sq) / (sw - 1), 1e-6);
				double phi = avg * (1.0 - avg) / sample_var - 1.0; // method of moments
				if (std::isnan(phi) || phi <= 0.0)  phi = 1.0; // fall-back
				phi = std::min(std::max(phi, 0.1), 100.0); // clip to a sane range
				aux_pars_[0] = phi;
			}//end "beta"
			else if (likelihood_type_ == "t") {
				//use MAD as robust initial estimate for the scale parameter
				std::vector<double> y_v;//for calculating the median
				if (fixed_effects == nullptr) {
					y_v.assign(y_data, y_data + num_data);
				}
				else {
					y_v = std::vector<double>(num_data);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						y_v[i] = y_data[i] - fixed_effects[i];
					}
				}
				double median = has_weights_ ? GPBoost::CalculateWeightedQuantile(y_v, weights_, 0.5) :
					GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(y_v);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					y_v[i] = std::abs(y_v[i] - median);
				}
				aux_pars_[0] = 1.4826 * (has_weights_ ? GPBoost::CalculateWeightedQuantile(y_v, weights_, 0.5) :
					GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(y_v));//MAD
				if (aux_pars_[0] <= EPSILON_NUMBERS) {
					// use IQR if MAD is zero
					if (fixed_effects == nullptr) {
						y_v.assign(y_data, y_data + num_data);
					}
					else {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data; ++i) {
							y_v[i] = y_data[i] - fixed_effects[i];
						}
					}
					double q25, q75;
					if (has_weights_) {
						q25 = GPBoost::CalculateWeightedQuantile(y_v, weights_, 0.25);
						q75 = GPBoost::CalculateWeightedQuantile(y_v, weights_, 0.75);
					}
					else {
						int pos = (int)(num_data * 0.25);
						std::nth_element(y_v.begin(), y_v.begin() + pos, y_v.end());
						q25 = y_v[pos];
						pos = (int)(num_data * 0.75);
						std::nth_element(y_v.begin(), y_v.begin() + pos, y_v.end());
						q75 = y_v[pos];
					}
					aux_pars_[0] = (q75 - q25) / 1.349;
				}
			}//end "t"
			else if (IsGaussianLikelihood()) {
				aux_pars_[0] = sample_var / 2.;
			}//end "gaussian")
			else if (likelihood_type_ == "lognormal") {
				// moment-based init: var(log y - offset) as log-variance
				sw = 0.0;
				double mean_log = 0., mean_log_sq = 0.;
#pragma omp parallel for schedule(static) reduction(+:mean_log, mean_log_sq, sw)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double z = (fixed_effects == nullptr) ? std::log(y_data[i]) : (std::log(y_data[i]) - fixed_effects[i]);
					mean_log += w * z;
					mean_log_sq += w * z * z;
					sw += w;
				}
				mean_log /= sw;
				mean_log_sq /= sw;
				aux_pars_[0] = std::max(mean_log_sq - mean_log * mean_log, 1e-6);;
			}
			else if (likelihood_type_ == "beta_binomial") {
				// Moment-based init for precision phi using Var(Y) decomposition:
				// Var(Y_i) = mu_i(1-mu_i)/n_i + [mu_i(1-mu_i)*(1 - 1/n_i)] * rho, with rho = 1/(phi+1).
				// Solve rho approx (V_obs - A) / B, where:
				//   V_obs = average_i (y_i - mu_i)^2
				//   A     = average_i mu_i(1-mu_i)/n_i
				//   B     = average_i mu_i(1-mu_i)*(1 - 1/n_i)
				// Then set phi = 1/rho - 1, clipped to a safe range.
				// 1) Build mu_i
				const double eps_mu = 1e-12;
				double pooled_mu = 0.5;
				bool have_fe = (fixed_effects != nullptr);
				if (!have_fe) {
					// Pooled proportion if no fixed effects: mu = (sum n*y) / (sum n)
					sw = 0.0;
					double swy = 0.0;
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						sw += w;
						swy += w * y_data[i];
					}
					pooled_mu = (sw > 0.0) ? (swy / sw) : 0.5;
					// clamp pooled mu
					if (pooled_mu < eps_mu) pooled_mu = eps_mu;
					if (pooled_mu > 1.0 - eps_mu) pooled_mu = 1.0 - eps_mu;
				}
				// 2) Compute V_obs, A, B as simple averages (equal weight per i)
				double V_obs = 0.0, A = 0.0, B = 0.0;
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? std::max(weights_[i], 1.0) : 1.0; // guard n>=1
					double mu_i;
					if (have_fe) {
						mu_i = GPBoost::sigmoid_stable_clamped(fixed_effects[i]);
					}
					else {
						mu_i = pooled_mu;
					}
					const double yi = y_data[i];
					const double s = mu_i * (1.0 - mu_i);
					V_obs += (yi - mu_i) * (yi - mu_i);
					A += s / w;
					B += s * (1.0 - 1.0 / w);
				}
				const double invN = (num_data > 0) ? (1.0 / (double)num_data) : 0.0;
				V_obs *= invN; A *= invN; B *= invN;
				// 3) Solve for rho, then phi = 1/rho - 1
				const double tiny = 1e-12;
				double rho = 0.0;
				if (B > tiny) {
					rho = (V_obs > A) ? ((V_obs - A) / B) : 0.0;  // no negative overdispersion
				}
				else {
					rho = 0.0; // when B approx 0 (e.g., many n=1), fall back to binomial
				}
				// clamp rho to [0, 1 - eps]
				if (rho < 0.0) rho = 0.0;
				if (rho > 1.0 - 1e-8) rho = 1.0 - 1e-8;
				double phi_init;
				if (rho <= 0.0) {
					phi_init = 1e6; // approximate binomial (very small ICC)
				}
				else {
					phi_init = (1.0 / rho) - 1.0;
					if (phi_init < 1e-6) phi_init = 1e-6;
					if (phi_init > 1e12) phi_init = 1e12;
				}
				aux_pars_[0] = phi_init;
			}//end "beta_binomial"
			else if (likelihood_type_ == "hurdle_gamma") {
				sw = 0.0;
				double log_avg = 0., avg_log = 0., avg_zero = 0., sw_pos = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:log_avg, avg_log, sw, avg_zero, sw_pos)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						if (yi <= 0.) {
							avg_zero += w;
						}
						else {
							log_avg += w * yi;
							avg_log += w * std::log(yi);
							sw_pos += w;
						}
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:log_avg, avg_log, sw, avg_zero, sw_pos)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						if (yi <= 0.) {
							avg_zero += w;
						}
						else {
							log_avg += w * yi / std::exp(fixed_effects[i]);
							avg_log += w * (std::log(yi) - fixed_effects[i]);
							sw_pos += w;
						}
						sw += w;
					}
				}
				log_avg /= sw_pos;
				log_avg = std::log(log_avg);
				avg_log /= sw_pos;
				avg_zero /= sw;
				avg_zero = std::min(std::max(avg_zero, 1e-3), 1. - 1e-3);
				double s = log_avg - avg_log;
				aux_pars_[0] = (3. - s + std::sqrt((s - 3.) * (s - 3.) + 24. * s)) / (12. * s);//same as for "gamma"
				aux_pars_[1] = avg_zero / (1. - avg_zero);
			}//end likelihood_type_ == "hurdle_gamma"
			else if (likelihood_type_ == "hurdle_lognormal") {
				// sigma2 = variance of log(y) over positive observations; p0 = weighted zero fraction.
				sw = 0.;
				double sw_pos = 0., avg_zero = 0., mean_log = 0., mean_log_sq = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, sw_pos, avg_zero, mean_log, mean_log_sq)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					if (y_data[i] <= 0.) { avg_zero += w; }
					else {
						const double ly = (fixed_effects == nullptr) ? std::log(y_data[i]) : (std::log(y_data[i]) - fixed_effects[i]);
						mean_log += w * ly;
						mean_log_sq += w * ly * ly;
						sw_pos += w;
					}
					sw += w;
				}
				mean_log /= sw_pos;
				mean_log_sq /= sw_pos;
				avg_zero /= sw;
				avg_zero = std::min(std::max(avg_zero, 1e-3), 1. - 1e-3);
				aux_pars_[0] = std::max(mean_log_sq - mean_log * mean_log, 1e-6);
				aux_pars_[1] = avg_zero / (1. - avg_zero);
			}//end likelihood_type_ == "hurdle_lognormal"
			else if (IsHurdleRegression()) {
				// Initialize only the base auxiliary parameters (there is no structural-zero p0 here). EGPD bases keep their constructor defaults.
				const string_t base = HurdleRegressionBaseType();
				if (base == "hurdle_gamma") {
					double log_avg = 0., avg_log = 0., sw_pos = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_avg, avg_log, sw_pos)
					for (data_size_t i = 0; i < num_data; ++i) {
						if (y_data[i] > 0.) { const double w = has_weights_ ? weights_[i] : 1.0; const double yr = (fixed_effects == nullptr) ? y_data[i] : y_data[i] / std::exp(fixed_effects[i]); log_avg += w * yr; avg_log += w * ((fixed_effects == nullptr) ? std::log(y_data[i]) : (std::log(y_data[i]) - fixed_effects[i])); sw_pos += w; }
					}
					log_avg = std::log(log_avg / sw_pos); avg_log /= sw_pos;
					const double s = std::max(log_avg - avg_log, 1e-8);
					aux_pars_[0] = (3. - s + std::sqrt((s - 3.) * (s - 3.) + 24. * s)) / (12. * s);
				}
				else if (base == "hurdle_lognormal") {
					double mean_log = 0., mean_log_sq = 0., sw_pos = 0.;
#pragma omp parallel for schedule(static) reduction(+:mean_log, mean_log_sq, sw_pos)
					for (data_size_t i = 0; i < num_data; ++i) {
						if (y_data[i] > 0.) { const double w = has_weights_ ? weights_[i] : 1.0; const double ly = (fixed_effects == nullptr) ? std::log(y_data[i]) : (std::log(y_data[i]) - fixed_effects[i]); mean_log += w * ly; mean_log_sq += w * ly * ly; sw_pos += w; }
					}
					mean_log /= sw_pos; mean_log_sq /= sw_pos;
					aux_pars_[0] = std::max(mean_log_sq - mean_log * mean_log, 1e-6);
				}
			}//end hurdle regression
			else if (IsHurdleEGPD()) {
				// Keep the base EGPD auxiliary parameters at their (constructor) defaults; initialize only the structural-zero p0.
				sw = 0.;
				double avg_zero = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					if (y_data[i] <= 0.) avg_zero += w;
					sw += w;
				}
				avg_zero = std::min(std::max(avg_zero / sw, 1e-3), 1. - 1e-3);
				aux_pars_[num_aux_pars_ - 1] = avg_zero / (1. - avg_zero);
			}//end hurdle EGPD variants
			else if (likelihood_type_ == "zero_inflated_poisson") {
				// Rough excess-zero start: p0 approx (dbar - f0bar) / (1 - f0bar), with f0bar approx exp(-mean(y)).
				// mean(y) underestimates the count-component mean, so this is only a finite deterministic start.
				sw = 0.;
				double avg_zero = 0., mean_y = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero, mean_y)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.) avg_zero += w;
						mean_y += w * y_data[i];
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero, mean_y)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.) avg_zero += w;
						mean_y += w * y_data[i] / std::exp(fixed_effects[i]);
						sw += w;
					}
				}
				avg_zero /= sw;
				mean_y = std::max(mean_y / sw, 1e-8);
				const double f0 = std::exp(-mean_y);
				double p0 = (avg_zero - f0) / std::max(1. - f0, 1e-6);
				p0 = std::min(std::max(p0, 1e-3), 1. - 1e-3);
				aux_pars_[0] = p0 / (1. - p0);
			}//end likelihood_type_ == "zero_inflated_poisson"
			else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
				// Moment-based start for shape/dispersion from the full (inflated) sample and an excess-zero start for p0.
				sw = 0.;
				double avg_zero = 0., mean_y = 0., sec_mom = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero, mean_y, sec_mom)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.) avg_zero += w;
						mean_y += w * y_data[i];
						sec_mom += w * y_data[i] * y_data[i];
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero, mean_y, sec_mom)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yr = y_data[i] / std::exp(fixed_effects[i]);
						if (y_data[i] <= 0.) avg_zero += w;
						mean_y += w * yr;
						sec_mom += w * yr * yr;
						sw += w;
					}
				}
				avg_zero /= sw;
				mean_y = std::max(mean_y / sw, 1e-8);
				const double var_y = std::max(sec_mom / sw - mean_y * mean_y, mean_y * 1.0001);// ensure overdispersion
				if (likelihood_type_ == "zero_inflated_negative_binomial") {
					const double kappa = mean_y * mean_y / std::max(var_y - mean_y, 1e-6);
					aux_pars_[0] = std::min(std::max(kappa, 1e-2), 1e6);
				}
				else {
					const double phi = var_y / mean_y - 1.;
					aux_pars_[0] = std::min(std::max(phi, 1e-3), 1e6);
				}
				const double f0 = std::exp(-mean_y);
				double p0 = (avg_zero - f0) / std::max(1. - f0, 1e-6);
				p0 = std::min(std::max(p0, 1e-3), 1. - 1e-3);
				aux_pars_[1] = p0 / (1. - p0);
			}//end zero_inflated_negative_binomial(_1)
			else if (IsZeroInflatedCountRegression()) {
				// Initialize only the base count auxiliary parameter (Poisson has none); moment-based from the full sample.
				const string_t base = ZICountRegressionBaseType();
				if (base != "zero_inflated_poisson") {
					sw = 0.;
					double mean_y = 0., sec_mom = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, mean_y, sec_mom)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0; const double yr = (fixed_effects == nullptr) ? y_data[i] : y_data[i] / std::exp(fixed_effects[i]);
						mean_y += w * yr; sec_mom += w * yr * yr; sw += w;
					}
					mean_y = std::max(mean_y / sw, 1e-8); const double var_y = std::max(sec_mom / sw - mean_y * mean_y, mean_y * 1.0001);
					if (base == "zero_inflated_negative_binomial") aux_pars_[0] = std::min(std::max(mean_y * mean_y / std::max(var_y - mean_y, 1e-6), 1e-2), 1e6);
					else aux_pars_[0] = std::min(std::max(var_y / mean_y - 1., 1e-3), 1e6);
				}
			}//end zero-inflated count regression
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
				//Estimating (sigma, lambda) using two moment conditions that are exact for the latent  X ~ N(mu,sigma^2) truncated at 0
				 // Notation: For y_i > 0, define u_i(lambda) = y_i^(1/lambda) = exp((1/lambda) * log y_i), a_i = -mu_i / sigma, tau(a) = phi(a) / (1 - Phi(a))
				 //   The truncated normal identities:  E[X | X > 0] = mu + sigma * tau(a),  Var[X | X > 0] = sigma^2 * (1 + a * tau(a) - tau(a)^2)
				const double eps_p = 1e-6;// probability clipping
				const double eps_sigma = 1e-6;// lower bound for sigma
				const double eps_var = 1e-12;// variance floor
				double W = 0.0, W0 = 0.0;
				double sw_mu = 0.0; // for pooled mu if fixed_effects != nullptr
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:W,W0)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						W += w;
						if (yi <= 0.0) W0 += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:W,W0,sw_mu)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						W += w;
						if (yi <= 0.0) W0 += w;
						sw_mu += w * fixed_effects[i];
					}
				}
				const double mu_pooled = (fixed_effects == nullptr) ? 0.0 : (sw_mu / W);// Use a pooled mu for the moment-matching step.
				// 1) Initial sigma from zeros if available: p0 = W0 / W ≈ Phi(a),  a = Phi^{-1}(p0) = -mu / sigma  =>  sigma ≈ -mu / a
				double sigma_init = 1.0;
				CHECK(W0 > 0.0 && W0 < W);
				double p0 = std::max(std::min(W0 / W, 1.0 - eps_p), eps_p);
				const double a_p0 = GPBoost::normalQF(p0); // a_p0 = Phi^{-1}(p0)
				if (std::fabs(a_p0) > 1e-6) {
					sigma_init = std::fabs(mu_pooled / a_p0);
				}
				else {
					sigma_init = 1.0; // avoid division by near-zero a_p0
				}
				sigma_init = std::max(sigma_init, eps_sigma);
				// Helpers for tau, E[X|X>0], Var[X|X>0] at given (mu, sigma)
				auto tau_upper = [](double a)->double {
					const double denom = std::max(1.0 - GPBoost::normalCDF(a), 1e-12);
					return GPBoost::normalPDF(a) / denom;// tau(a) = phi(a) / (1 - Phi(a))
				};
				auto vpos = [&](double mu_local, double sigma_local)->double {
					const double a = -mu_local / sigma_local;
					const double t = tau_upper(a);
					return sigma_local * sigma_local * (1.0 + a * t - t * t);
				};
				auto mpos = [&](double mu_local, double sigma_local)->double {
					const double a = -mu_local / sigma_local;
					return mu_local + sigma_local * tau_upper(a);
				};
				// 2) Small lambda grid: pick lambda that makes the positive u_i match mpos/vpos
				double mpos_target, vpos_target;
				double m_sum = 0.0, m2_sum = 0.0, v_sum = 0.0, Wpos_fe = 0.0;
				if (fixed_effects == nullptr) {
					const double sigma_for_grid = sigma_init;
					mpos_target = mpos(mu_pooled, sigma_for_grid);
					vpos_target = std::max(vpos(mu_pooled, sigma_for_grid), 1e-10);
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:m_sum,m2_sum,v_sum,Wpos_fe)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] > 0.0) {
							const double mu_i = fixed_effects[i];
							const double m_i = mpos(mu_i, sigma_init);
							const double v_i = vpos(mu_i, sigma_init);
							Wpos_fe += w;
							m_sum += w * m_i;
							m2_sum += w * m_i * m_i;
							v_sum += w * v_i;
						}
					}
					const double m_mix = m_sum / Wpos_fe;
					const double v_within = v_sum / Wpos_fe;
					const double v_between = std::max(m2_sum / Wpos_fe - m_mix * m_mix, 0.0);
					mpos_target = m_mix;
					vpos_target = std::max(v_within + v_between, 1e-10);
				}
				const double lambda_grid[] = { 0.25, 0.3333333333, 0.5, 0.6666666667, 0.75, 1.0, 1.25, 1.5, 2.0 };
				const int nL = static_cast<int>(sizeof(lambda_grid) / sizeof(lambda_grid[0]));
				auto compute_weighted_moments_u = [&](double lambda, double& mean_u, double& var_u, double& skew_u)->void {
					// Weighted mean/var/skew of u = y^(1/lambda) over positives
					double sumw = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sumw,sum1,sum2,sum3)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double y = y_data[i];
						if (y > 0.0) {
							const double u = std::exp((1.0 / lambda) * std::log(y));
							sumw += w;
							sum1 += w * u;
							sum2 += w * u * u;
							sum3 += w * u * u * u;
						}
					}
					mean_u = sum1 / sumw;
					var_u = std::max(sum2 / sumw - mean_u * mean_u, 0.0);
					const double m3c = (sum3 / sumw) - 3.0 * mean_u * (sum2 / sumw) + 2.0 * mean_u * mean_u * mean_u;
					skew_u = (var_u > 0.0) ? (m3c / std::pow(var_u, 1.5)) : 0.0;
				};
				double best_lambda = 1.0, best_score = std::numeric_limits<double>::infinity();
				for (int k = 0; k < nL; ++k) {
					const double lam = lambda_grid[k];
					double mean_u = 0.0, var_u = 0.0, skew_u = 0.0;
					compute_weighted_moments_u(lam, mean_u, var_u, skew_u);
					// Match (mean_u, var_u) to theoretical (mpos_target, vpos_target)
					const double dm = mean_u - mpos_target;
					const double dv = var_u - vpos_target;
					double score = (dm * dm) / std::max(vpos_target, 1e-12) + (dv * dv) / std::max(vpos_target * vpos_target, 1e-12);
					if (score < best_score) {
						best_score = score;
						best_lambda = lam;
					}
				}
				// 3) Refine sigma by matching the sample var of u to the theoretical positive variance
				double mean_u = 0.0, var_u = 0.0, skew_u = 0.0;
				compute_weighted_moments_u(best_lambda, mean_u, var_u, skew_u);
				var_u = std::max(var_u, eps_var);
				auto vpos_mixture = [&](double sigma_curr)->double {
					// Returns theoretical mixture variance of X|X>0 at current sigma (weights over positives)
					if (fixed_effects == nullptr) {
						return std::max(vpos(mu_pooled, sigma_curr), eps_var);// Homogeneous mu: just vpos at pooled mu
					}
					else {
						double m_sum_it = 0.0, m2_sum_it = 0.0, v_sum_it = 0.0, Wpos_it = 0.0;
#pragma omp parallel for schedule(static) reduction(+:m_sum_it,m2_sum_it,v_sum_it,Wpos_it)
						for (data_size_t i = 0; i < num_data; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							if (y_data[i] > 0.0) {
								const double mu_i = fixed_effects[i];
								const double m_i = mpos(mu_i, sigma_curr);
								const double v_i = vpos(mu_i, sigma_curr);
								Wpos_it += w;
								m_sum_it += w * m_i;
								m2_sum_it += w * m_i * m_i;
								v_sum_it += w * v_i;
							}
						}
						if (Wpos_it <= 0.0) return std::max(vpos(mu_pooled, sigma_curr), eps_var);
						const double m_mix_it = m_sum_it / Wpos_it;
						const double v_within_it = v_sum_it / Wpos_it;
						const double v_between_it = std::max(m2_sum_it / Wpos_it - m_mix_it * m_mix_it, 0.0);
						return std::max(v_within_it + v_between_it, eps_var);
					}
				};
				double sigma = sigma_init;
				for (int it = 0; it < 2; ++it) {
					const double vth = vpos_mixture(sigma);         // theoretical var at current sigma
					sigma = sigma * std::sqrt(var_u / vth);         // fixed-point update
					sigma = std::max(sigma, eps_sigma);
				}
				if (!(sigma > 0.0) || !std::isfinite(sigma))   sigma = 1.0;
				if (!(best_lambda > 0.0) || !std::isfinite(best_lambda)) best_lambda = 1.0;
				aux_pars_[0] = sigma;
				aux_pars_[1] = best_lambda;
			}//end "zero_censored_power_transformed_normal"
			else if (IsZeroCensPowNormHetero()) {
				// lambda is the only auxiliary parameter here. Profile it out over a small grid: for each candidate lambda,
				// the (homoscedastic) method-of-moments anchors give the pooled (mu, sigma) of the total location parameters,
				// and the candidate maximizing the resulting censored log-likelihood is chosen
				const double lambda_grid[] = { 0.25, 0.3333333333, 0.5, 0.6666666667, 0.75, 1.0, 1.25, 1.5, 2.0 };
				const int nL = static_cast<int>(sizeof(lambda_grid) / sizeof(lambda_grid[0]));
				double best_lambda = 1., best_ll = -std::numeric_limits<double>::infinity();
				for (int k = 0; k < nL; ++k) {
					const double lam = lambda_grid[k];
					double mu_anchor, log_sigma_anchor;
					ZeroCensPowNormHeteroAnchors(y_data, num_data, weights_, lam, mu_anchor, log_sigma_anchor);
					const double mu = mu_anchor, sigma = std::exp(log_sigma_anchor);
					double ll = 0.;
#pragma omp parallel for schedule(static) reduction(+:ll)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.) {
							ll += w * GPBoost::normalLogCDF(-mu / sigma);
						}
						else {
							const double logy = std::log(y_data[i]);
							const double z = (std::exp(logy / lam) - mu) / sigma;
							ll += w * (-0.5 * z * z - std::log(lam) - std::log(sigma) - M_LOGSQRT2PI + (1. / lam - 1.) * logy);
						}
					}
					if (std::isfinite(ll) && ll > best_ll) {
						best_ll = ll;
						best_lambda = lam;
					}
				}
				aux_pars_[0] = best_lambda;
			}//end "zero_censored_power_transformed_normal_heteroscedastic"
			else if (likelihood_type_ == "zoctn") {
				// Rough init: sigma from y in (0,1), a=1, b=1
				double sw_int = 0.0, sum_y = 0.0, sum_y_sq = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sw_int,sum_y,sum_y_sq)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double yi = y_data[i];
					if (yi > 0.0 && yi < 1.0) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						sw_int += w;
						sum_y += w * yi;
						sum_y_sq += w * yi * yi;
					}
				}
				double sigma = 1.0;
				if (sw_int > 0.0) {
					const double mean = sum_y / sw_int;
					const double var = std::max(sum_y_sq / sw_int - mean * mean, 1e-6);
					sigma = std::sqrt(var);
					if (!(sigma > 0.0) || !std::isfinite(sigma)) sigma = 1.0;
				}
				aux_pars_[0] = sigma; // sigma
				aux_pars_[1] = 1.0;   // a
				aux_pars_[2] = 1.0;   // b
			}//end "zoctn"
			else if (likelihood_type_ == "zero_one_censored_transformed_beta") {
				double swi = 0.0, sum = 0.0, sumsq = 0.0, W0 = 0.0, W1 = 0.0;
#pragma omp parallel for schedule(static) reduction(+:swi,sum,sumsq,W0,W1)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double yi = y_data[i];
					const double w = has_weights_ ? weights_[i] : 1.0;
					if (yi > 0.0 && yi < 1.0) {
						swi += w; sum += w * yi;
						sumsq += w * yi * yi;
					}
					else if (yi <= 0.0) {
						W0 += w;
					}
					else {
						W1 += w;
					}
				}
				double phi = 20.;
				if (swi > 1.0) {
					double m = sum / swi;
					double v = std::max(sumsq / swi - m * m, 1e-6);
					phi = std::max(m * (1.0 - m) / v - 1.0, 0.1);
					phi = std::min(phi, 100.0);
				}
				double u;
				if (W0 + W1 > 0.0) {
					u = 0.05;
				}
				else {
					u = 1e-3;
				}
				aux_pars_[0] = phi;
				aux_pars_[1] = u;
			}//end "zero_one_censored_transformed_beta"
			else if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
				double W = 0.0, W0 = 0.0, sum_y = 0.0, cnt_interior = 0.0;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:W,W0,sum_y,cnt_interior)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						W += w;
						if (yi <= 0.0) { W0 += w; }
						else if (yi < 1.0) { sum_y += w * yi; cnt_interior += w; }
						// yi >= 1 contributes neither to sum_y nor W0
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:W,W0,sum_y,cnt_interior)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						W += w;
						if (yi <= 0.0) { W0 += w; }
						else if (yi < 1.0) { sum_y += w * yi / std::exp(fixed_effects[i]); cnt_interior += w; }
					}
				}
				// 1) zero fraction
				const double p0 = (W > 0.0) ? std::min(std::max(W0 / W, 1e-12), 1.0 - 1e-12) : 0.1;
				// 2) crude mean of Z from interior Y + small offset (0.5 works as rough prior mass in middle)
				const double mu_crude = (cnt_interior > 0.0) ? std::max(sum_y / cnt_interior, 1e-12) + 0.5 : 1.0;
				// 3) first xi via inverse lower gamma using a provisional k (1.0 is robust) and mu_crude
				double k_init = 1.0;
				double xq = GPBoost::InvRegLowerGamma(k_init, p0);
				double xi_init = std::max(1e-6, std::min(1e6, (mu_crude / k_init) * xq));
				// 4) refine k using z = y + xi_init on interior points
				double sum_z = 0.0, sum_logz = 0.0, cnt_z = 0.0;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_z,sum_logz,cnt_z)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						if (yi > 0.0 && yi < 1.0) {
							const double z = yi + xi_init;
							sum_z += w * z;
							sum_logz += w * std::log(std::max(z, 1e-12));
							cnt_z += w;
						}
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:sum_z,sum_logz,cnt_z)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						if (yi > 0.0 && yi < 1.0) {
							const double z = (yi + xi_init) / std::exp(fixed_effects[i]);
							sum_z += w * z;
							sum_logz += w * std::log(std::max(z, 1e-12));
							cnt_z += w;
						}
					}
				}
				if (cnt_z > 0.0) {
					const double zbar = std::max(sum_z / cnt_z, 1e-12);
					const double s = std::max(std::log(zbar) - (sum_logz / cnt_z), 1e-12);
					// standard log-moment estimator for Gamma shape
					k_init = std::max(0.2, std::min(50.0,
						(3.0 - s + std::sqrt((s - 3.0) * (s - 3.0) + 24.0 * s)) / (12.0 * s)));
					// optionally, one could re-update xi via p0 here; the first xi_init is already reasonable
				}
				aux_pars_[0] = k_init;
				aux_pars_[1] = xi_init;
			}//end "zero_one_censored_shifted_gamma"
			else if (likelihood_type_ == "asymmetric_laplace") {
				// Use MLE for initial scale assuming location_par is zero
				double aux_sum = 0.;
				sw = 0.;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:aux_sum, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						double indicator = (y_data[i] <= 0.) ? 1.0 : 0.0;
						aux_sum += w * y_data[i] * (indicator - quantile_);
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:aux_sum, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						double indicator = (y_data[i] <= fixed_effects[i]) ? 1.0 : 0.0;
						aux_sum += w * (y_data[i] - fixed_effects[i]) * (indicator - quantile_);
						sw += w;
					}
				}
				aux_pars_[0] = -aux_sum / sw;
			}//end "asymmetric_laplace"
			else if (likelihood_type_ != "bernoulli_probit" && likelihood_type_ != "bernoulli_logit" &&
				likelihood_type_ != "binomial_probit" && likelihood_type_ != "binomial_logit" &&
				likelihood_type_ != "poisson" && !IsGaussianHeteroscedastic() && !IsEGPDLikelihood() &&
				likelihood_type_ != "quasi_bernoulli_probit" && likelihood_type_ != "quasi_bernoulli_logit") {
				NotSupportedForLikelihood(__func__);
			}
			aux_pars_original_ = aux_pars_;
			BackTransformAuxPars(aux_pars_.data(), aux_pars_original_.data());
			return(aux_pars_.data());
		}//end FindInitialAuxPars

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
			const double* weights = nullptr) const {
			const double* weights_ptr = (weights != nullptr) ? weights : weights_;
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit" ||
				likelihood_type_ == "binomial_probit" || likelihood_type_ == "binomial_logit" ||
				likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial" || likelihood_type_ == "zoctn" ||
				likelihood_type_ == "zero_one_censored_transformed_beta" || likelihood_type_ == "zero_one_censored_shifted_gamma" || 
				likelihood_type_ == "quasi_bernoulli_probit" || likelihood_type_ == "quasi_bernoulli_logit") {
				C_mu = 1.;
				C_sigma2 = 1.;
			}
			else if (IsHurdlePositive()) {
				double sw = 0.0, mean = 0., sec_mom = 0.;
#pragma omp parallel for schedule(static) reduction(+:mean, sec_mom, sw)
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] > 0.) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						mean += w * y_data[i];
						sec_mom += w * y_data[i] * y_data[i];
						sw += w;
					}
				}
				mean /= sw;
				sec_mom /= sw;
				C_mu = std::abs(SafeLog(mean));
				C_sigma2 = std::abs(SafeLog(sec_mom - mean * mean));
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" || IsEGPDLikelihood() ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				likelihood_type_ == "lognormal" || IsZeroInflatedCount()) {
				double sw = 0.0, mean = 0., sec_mom = 0;
#pragma omp parallel for schedule(static) reduction(+:mean, sec_mom, sw)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_ptr[i] : 1.0;
					mean += w * y_data[i];
					sec_mom += w * y_data[i] * y_data[i];
					sw += w;
				}
				mean /= sw;
				sec_mom /= sw;
				C_mu = std::abs(SafeLog(mean));
				C_sigma2 = std::abs(SafeLog(sec_mom - mean * mean));
			}
			else if (likelihood_type_ == "t") {
				//use the median and MAD^2 as robust location and scale^2 parameters
				std::vector<double> y_v;//for calculating the median
				if (fixed_effects == nullptr) {
					y_v.assign(y_data, y_data + num_data);
				}
				else {
					y_v = std::vector<double>(num_data);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						y_v[i] = y_data[i] - fixed_effects[i];
					}
				}
				C_mu = has_weights_ ? GPBoost::CalculateWeightedQuantile(y_v, weights_ptr, 0.5) :
					GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(y_v);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					y_v[i] = std::abs(y_v[i] - C_mu);
				}
				C_sigma2 = 1.4826 * (has_weights_ ? GPBoost::CalculateWeightedQuantile(y_v, weights_ptr, 0.5) :
					GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(y_v));//MAD
				C_sigma2 = C_sigma2 * C_sigma2;
				if (C_sigma2 <= EPSILON_NUMBERS) {
					// use IQR if MAD is zero
					if (fixed_effects == nullptr) {
						y_v.assign(y_data, y_data + num_data);
					}
					else {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data; ++i) {
							y_v[i] = y_data[i] - fixed_effects[i];
						}
					}
					double q25, q75;
					if (has_weights_) {
						q25 = GPBoost::CalculateWeightedQuantile(y_v, weights_ptr, 0.25);
						q75 = GPBoost::CalculateWeightedQuantile(y_v, weights_ptr, 0.75);
					}
					else {
						int pos = (int)(num_data * 0.25);
						std::nth_element(y_v.begin(), y_v.begin() + pos, y_v.end());
						q25 = y_v[pos];
						pos = (int)(num_data * 0.75);
						std::nth_element(y_v.begin(), y_v.begin() + pos, y_v.end());
						q75 = y_v[pos];
					}
					C_sigma2 = (q75 - q25) / 1.349;
					C_sigma2 = C_sigma2 * C_sigma2;
				}
			}//end "t"
			else if (IsGaussianLikelihood() || likelihood_type_ == "zero_censored_power_transformed_normal" ||
				likelihood_type_ == "asymmetric_laplace") {
				double sw = 0.0, mean = 0., sec_mom = 0;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:mean, sec_mom, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						mean += w * y_data[i];
						sec_mom += w * y_data[i] * y_data[i];
						sw += w;
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:mean, sec_mom, sw)
					for (data_size_t i = 0; i < num_data; ++i) {
						const double w = has_weights_ ? weights_ptr[i] : 1.0;
						mean += w * y_data[i] - fixed_effects[i];
						sec_mom += w * (y_data[i] - fixed_effects[i]) * (y_data[i] - fixed_effects[i]);
						sw += w;
					}
				}
				mean /= sw;
				sec_mom /= sw;
				C_mu = std::abs(mean);
				C_sigma2 = sec_mom - mean * mean;
			}//end "gaussian"|| likelihood_type_ == "zero_censored_power_transformed_normal" || likelihood_type_ == "asymmetric_laplace"
			else if (IsGaussianHeteroscedastic() || IsZeroCensPowNormHetero()) {
				C_mu = 1e99;//not implemented
				C_sigma2 = 1e99;
			}
			else {
				NotSupportedForLikelihood(__func__);
			}
			if (C_mu < 1.) {
				C_mu = 1.;
			}
		}//end FindConstantsCapTooLargeLearningRateCoef

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
		void SetAuxPars(const double* aux_pars) {
			if (likelihood_type_ == "t" && !estimate_df_t_ && !aux_pars_have_been_set_) {
				if (!TwoNumbersAreEqual<double>(aux_pars[1], aux_pars_[1])) {
					Log::REWarning("The '%s' parameter provided in 'init_aux_pars' (= %g) and 'likelihood_additional_param' (= %g) are not equal. "
						"Will use the value provided in 'likelihood_additional_param' ", names_aux_pars_[1].c_str(), aux_pars[1], aux_pars_[1]);
				}
			}
			if (likelihood_type_ == "tweedie_fixed_p" && !aux_pars_have_been_set_ && !TwoNumbersAreEqual<double>(aux_pars[1], aux_pars_[1])) {
				Log::REWarning("The 'power' parameter provided in 'init_aux_pars' (= %g) and 'likelihood_additional_param' (= %g) are not equal. Will use the value provided in 'likelihood_additional_param'.", aux_pars[1], aux_pars_[1]);
			}
			if (IsGaussianLikelihood() || IsEGPDLikelihood() || likelihood_type_ == "gamma" ||
				likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				likelihood_type_ == "beta" || likelihood_type_ == "t" || likelihood_type_ == "lognormal" ||
				likelihood_type_ == "beta_binomial" || IsHurdlePositive() || IsZeroInflatedCount() ||
				IsZeroCensPowNorm() || likelihood_type_ == "zoctn" ||
				likelihood_type_ == "zero_one_censored_transformed_beta" || likelihood_type_ == "zero_one_censored_shifted_gamma" ||
				likelihood_type_ == "asymmetric_laplace") {
				for (int i = 0; i < num_aux_pars_estim_; ++i) {
					if (!(aux_pars[i] > 0.)) {
						Log::REFatal("The '%s' parameter (= %g) is not > 0. This might be due to a problem when estimating the '%s' parameter (e.g., a numerical overflow). "
							"You can try either (i) manually setting a different initial value using the 'init_aux_pars' parameter "
							"or (ii) not estimating the '%s' parameter at all by setting 'estimate_aux_pars' to 'false'. "
							"Both these options can be specified in the 'params' argument by calling, e.g., the 'set_optim_params()' function of a 'GPModel' ",
							names_aux_pars_[i].c_str(), aux_pars[i], names_aux_pars_[i].c_str(), names_aux_pars_[i].c_str());
					}
					aux_pars_[i] = aux_pars[i];
				}
				if (likelihood_type_ == "asymmetric_laplace") {
					eps_sub_grad_scale_ = EPSILON_SUB_GRAD_ * aux_pars_[0];
				}
			}
			aux_pars_original_ = aux_pars_;
			BackTransformAuxPars(aux_pars_.data(), aux_pars_original_.data());
			normalizing_constant_has_been_calculated_ = false;
			if (likelihood_type_ == "tweedie") tweedie_boundary_warning_issued_ = false;
			// Refresh in this single-threaded update path only when the parameters actually changed. Later
			// response-scale transforms (e.g. boosting ConvertOutput) only read the cache.
			if (HasEGPDBase() && !EGPDMomentsCacheMatchesAuxPars()) RefreshEGPDMomentsCache();
			aux_pars_have_been_set_ = true;
		}

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
			bool predict_var) {
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "binomial_probit" || 
				likelihood_type_ == "quasi_bernoulli_probit") {
				CHECK(need_pred_latent_var_for_response_mean_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					pred_mean[i] = GPBoost::normalCDF(pred_mean[i] / std::sqrt(1. + pred_var[i]));
				}
				if (predict_var) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] = pred_mean[i] * (1. - pred_mean[i]);
					}
				}
			}
			else if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit" || 
				likelihood_type_ == "quasi_bernoulli_logit") {
				CHECK(need_pred_latent_var_for_response_mean_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					pred_mean[i] = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i], false);
				}
				if (predict_var) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] = pred_mean[i] * (1. - pred_mean[i]);
					}
				}
			}
			else if (likelihood_type_ == "poisson") {
				CHECK(need_pred_latent_var_for_response_mean_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					double pm = std::exp(pred_mean[i] + 0.5 * pred_var[i]);
					//double pm = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i], false);// alternative version using quadrature
					if (predict_var) {
						pred_var[i] = pm * ((std::exp(pred_var[i]) - 1.) * pm + 1.);
						//double psm = RespMeanAdaptiveGHQuadrature(2 * pred_mean[i], 4 * pred_var[i], false);// alternative version using quadrature
						//pred_var[i] = psm - pm * pm + pm;
					}
					pred_mean[i] = pm;
				}
			}
			else if (likelihood_type_ == "zero_inflated_poisson") {
				// latent eta ~ N(m,v), zeta fixed. E(Y*) = q*A1; Var(Y*) = q*A1 + q*p0*A2 + q^2*(A2 - A1^2)
				CHECK(need_pred_latent_var_for_response_mean_);
				const double p0 = aux_pars_original_[0];
				const double q = 1. - p0;
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					const double A1 = std::exp(m + 0.5 * v);
					const double pm = q * A1;
					if (predict_var) {
						const double A2 = std::exp(2. * m + 2. * v);
						const double V_mu = A2 - A1 * A1;
						pred_var[i] = q * A1 + q * p0 * A2 + q * q * V_mu;
					}
					pred_mean[i] = pm;
				}
			}
			else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
				// E(Y*) = q*A1; Var(Y*) = q*Vfac*A1 + q*p0*A2 + q^2*(A2 - A1^2), with the count-component excess dispersion
				// captured by Vfac (NB1) and the extra shape term (NB2).  A1=exp(m+v/2), A2=exp(2m+2v).
				CHECK(need_pred_latent_var_for_response_mean_);
				const bool nb2 = (likelihood_type_ == "zero_inflated_negative_binomial");
				const double p0 = aux_pars_original_[1];
				const double q = 1. - p0;
				const double shape = aux_pars_[0];
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					const double A1 = std::exp(m + 0.5 * v);
					const double pm = q * A1;
					if (predict_var) {
						const double A2 = std::exp(2. * m + 2. * v);
						const double V_mu = A2 - A1 * A1;
						if (nb2) pred_var[i] = q * A1 + q * (1. / shape + p0) * A2 + q * q * V_mu;// NB2: Var(Y|eta)=mu+mu^2/kappa
						else     pred_var[i] = q * (1. + shape) * A1 + q * p0 * A2 + q * q * V_mu;// NB1: Var(Y|eta)=mu*(1+phi)
					}
					pred_mean[i] = pm;
				}
			}
			else if (likelihood_type_ == "gamma") {
				CHECK(need_pred_latent_var_for_response_mean_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					double pm = std::exp(pred_mean[i] + 0.5 * pred_var[i]);
					//double pm = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i], false);// alternative version using quadrature
					if (predict_var) {
						pred_var[i] = (std::exp(pred_var[i]) - 1.) * pm * pm + std::exp(2 * pred_mean[i] + 2 * pred_var[i]) / aux_pars_[0];
						//double psm = RespMeanAdaptiveGHQuadrature(2 * pred_mean[i], 4 * pred_var[i], false);// alternative version using quadrature
						//pred_var[i] = psm - pm * pm + psm / aux_pars_[0];
					}
					pred_mean[i] = pm;
				}
			}
			else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
				CHECK(need_pred_latent_var_for_response_mean_);
				const double phi = aux_pars_[0];
				const double p = GetTweediePower();
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = pred_var[i];
					const double pm = std::exp(m + 0.5 * v);
					if (predict_var) pred_var[i] = phi * std::exp(p * m + 0.5 * p * p * v) + std::exp(2. * m + v) * std::expm1(v);
					pred_mean[i] = pm;
				}
			}
			else if (IsEGPDLikelihood()) {
				CHECK(need_pred_latent_var_for_response_mean_);
				const auto& moments = GetEGPDMoments();
				if (!moments.mean_exists) Log::REFatal("PredictResponse: the response mean does not exist for likelihood='%s' when shape >= 1 ", likelihood_type_.c_str());
				if (predict_var && !moments.variance_exists) Log::REFatal("PredictResponse: the response variance does not exist for likelihood='%s' when shape >= 0.5 ", likelihood_type_.c_str());
				if (moments.status != EGPDEvalStatus::kValid) Log::REFatal("PredictResponse: failed to calculate EGPD unit-scale moments ");
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double latent_mean = pred_mean[i];
					const double latent_var = pred_var[i];
					if (!(std::isfinite(latent_var) && latent_var >= 0.)) { pred_mean[i] = std::numeric_limits<double>::quiet_NaN(); if (predict_var) pred_var[i] = std::numeric_limits<double>::quiet_NaN(); continue; }
					const double response_mean = moments.mean_unit_scale * std::exp(latent_mean + 0.5 * latent_var);
					if (predict_var) {
						const double unit_second = moments.variance_unit_scale + moments.mean_unit_scale * moments.mean_unit_scale;
						pred_var[i] = unit_second * std::exp(2. * latent_mean + 2. * latent_var) - response_mean * response_mean;
					}
					pred_mean[i] = response_mean;
				}
			}
			else if (IsHurdleEGPD()) {
				// Scale family (M_b = c1*exp(eta), V_b = c2*exp(2*eta)). E(Y*) = q*c1*exp(m+v/2);
				// Var(Y*) = q*(c2 + p0*c1^2)*exp(2m+2v) + q^2*c1^2*exp(2m+v)*(exp(v)-1).
				CHECK(need_pred_latent_var_for_response_mean_);
				const auto& moments = GetEGPDMoments();
				if (!moments.mean_exists) Log::REFatal("PredictResponse: the response mean does not exist for likelihood='%s' when shape >= 1 ", likelihood_type_.c_str());
				if (predict_var && !moments.variance_exists) Log::REFatal("PredictResponse: the response variance does not exist for likelihood='%s' when shape >= 0.5 ", likelihood_type_.c_str());
				if (moments.status != EGPDEvalStatus::kValid) Log::REFatal("PredictResponse: failed to calculate EGPD unit-scale moments ");
				const double p0 = aux_pars_original_[num_aux_pars_ - 1];
				const double q = 1. - p0;
				const double c1 = moments.mean_unit_scale, c2 = moments.variance_unit_scale;
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					const double response_mean = q * c1 * std::exp(m + 0.5 * v);
					if (predict_var) {
						pred_var[i] = q * (c2 + p0 * c1 * c1) * std::exp(2. * m + 2. * v) + q * q * c1 * c1 * std::exp(2. * m + v) * std::expm1(v);
					}
					pred_mean[i] = response_mean;
				}
			}
			else if (likelihood_type_ == "negative_binomial") {
				CHECK(need_pred_latent_var_for_response_mean_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					double pm = std::exp(pred_mean[i] + 0.5 * pred_var[i]);
					if (predict_var) {
						pred_var[i] = std::exp(2 * (pred_mean[i] + pred_var[i])) * (1 + 1 / aux_pars_[0]) + pm * (1 - pm);
					}
					pred_mean[i] = pm;
				}
			}
			else if (likelihood_type_ == "negative_binomial_1") {
				CHECK(need_pred_latent_var_for_response_mean_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					double pm = std::exp(pred_mean[i] + 0.5 * pred_var[i]);
					if (predict_var) {
						pred_var[i] = pm * ((std::exp(pred_var[i]) - 1.) * pm + 1. + aux_pars_[0]);
					}
					pred_mean[i] = pm;
				}
			}
			else if (likelihood_type_ == "beta") {
				CHECK(need_pred_latent_var_for_response_mean_);
				if (predict_var) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						double resp_mean = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i], false);
						double var_E = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i], true) - resp_mean * resp_mean;
						double E_var = ExpectedValueCondRespVarAdaptiveGHQuadrature(pred_mean[i], pred_var[i]);
						pred_mean[i] = resp_mean;
						pred_var[i] = var_E + E_var;
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_mean[i] = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i], false);
					}
				}
			}
			else if (likelihood_type_ == "t") {
				CHECK(!need_pred_latent_var_for_response_mean_);
				if (predict_var) {
					pred_var.array() += aux_pars_[0] * aux_pars_[0];
					Log::REDebug("Response prediction for a 't' likelihood: we simply add the squared 'scale' parameter to the variances of the latent predictions "
						"and do not assume that the 't' distribution is the true likelihood but rather an auxiliary tool for robust regression ");
				}
				//              // Code when assuming that the t-distribution is the true likelihood
				//              if (aux_pars_[1] <= 1.) {
				//                  Log::REFatal("The response mean of a 't' distribution is only defined if the "
				//                      "'%s' parameter (=degrees of freedom) is larger than 1. Currently, it is %g. "
				//                      "You can set this parameter via the 'likelihood_additional_param' parameter ", names_aux_pars_[1].c_str(), aux_pars_[1]);
				//              }
				//              if (predict_var && aux_pars_[1] <= 2.) {
				//                  Log::REFatal("The response mean of a 't' distribution is only defined if the "
				//                      "'%s' parameter (=degrees of freedom) is larger than 2. Currently, it is %g. "
				//                      "You can set this parameter via the 'likelihood_additional_param' parameter ", names_aux_pars_[1].c_str(), aux_pars_[1]);
				//              }
				//              if (predict_var) {
				//                  Log::REWarning("Predicting the response variable for a 't' likelihood: it is assumed that the t-distribution is the true likelihood, and  "
				//                      " predictive variance are calculated accordingly. If you use the 't' likelihood only as an auxiliary tool for robust regression, "
				//                      "consider predicting the latent variable (predict_response = false) (and maybe add the squared scale parameter assuming the true likelihood without contamination is gaussian) ");
				//                  double pred_var_const = aux_pars_[0] * aux_pars_[0] * aux_pars_[1] / (aux_pars_[1] - 2.);
				//#pragma omp parallel for schedule(static)
				//                  for (int i = 0; i < (int)pred_mean.size(); ++i) {
				//                      pred_var[i] = pred_var[i] + pred_var_const;
				//                  }
				//              }
			}//end "t"
			else if (IsGaussianLikelihood()) {
				if (predict_var) {
					pred_var.array() += aux_pars_[0];
				}
			}
			else if (IsGaussianHeteroscedastic()) {
				// For 'gaussian_heteroscedastic' (fixed effects only), the caller sets 'pred_var_var' = 0 since the log-error
				// variance is deterministic given the fixed effects (no random effect / GP posterior uncertainty)
				if (predict_var) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] += std::exp(pred_var_mean[i] + pred_var_var[i] / 2.);
					}
				}
			}
			else if (likelihood_type_ == "lognormal") {
				CHECK(need_pred_latent_var_for_response_mean_);
				const double s2 = aux_pars_[0];
				const double exp_s2_m1 = std::expm1(s2);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = pred_var[i];
					const double pm = std::exp(m + 0.5 * v);
					pred_mean[i] = pm;
					if (predict_var) {
						const double exp_v_m1 = std::expm1(v);
						const double pm2 = pm * pm;
						const double var_of_mean = exp_v_m1 * pm2;
						const double mean_of_var = exp_s2_m1 * pm2 * (exp_v_m1 + 1.);
						//const double var_of_mean = (std::exp(v) - 1.0) * std::exp(2.0 * m + v);
						//const double mean_of_var = (exp_s2 - 1.0) * std::exp(2.0 * m + 2.0 * v);
						pred_var[i] = var_of_mean + mean_of_var;
					}
				}
			}//end "lognormal"
			else if (likelihood_type_ == "beta_binomial") {
				CHECK(need_pred_latent_var_for_response_mean_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					//const double w = 1.;//assume 1 trial
					double mu = GPBoost::sigmoid_stable_clamped(m);
					const double s = mu * (1.0 - mu);
					// Mean on response scale: E[mu] aprox mu + 0.5 mu'' v, with mu'' = s(1-2mu)
					pred_mean[i] = mu + 0.5 * s * (1.0 - 2.0 * mu) * v;
					if (predict_var) {
						// Var(E[Y|eta]|y): Var(mu) approx (dmu/deta)^2 Var(eta) = s^2 v
						const double var_of_mean = (s * s) * v;
						// E(Var(Y|eta)|y) for proportion Y: Var(Y|eta)= mu(1-mu)/w + (1-1/w)mu(1-mu)rho, assume w = 1
						// Use second-order delta for E[mu(1-mu)]:  E[s] approx s + 0.5 s'' v, with s'' = s(1-6mu+6mu^2)
						const double s_dd = s * (1.0 - 6.0 * mu + 6.0 * mu * mu);
						double mean_of_var = s + 0.5 * s_dd * v;//E[s]
						if (mean_of_var < 0.0) mean_of_var = 0.0;
						if (mean_of_var > 0.25) mean_of_var = 0.25;
						pred_var[i] = var_of_mean + mean_of_var;
					}
				}
			}//end "beta_binomial"
			else if (likelihood_type_ == "hurdle_gamma") {
				CHECK(need_pred_latent_var_for_response_mean_);
				const double p0 = aux_pars_original_[1];
				const double q = 1. - p0;
				double k = 0.;
				if (predict_var) {
					k = aux_pars_[0];
				}
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					// E[Y | data] = q * E[mu], where mu = E[Y | Y > 0, eta] = exp(eta)
					const double E_mu = std::exp(m + 0.5 * v);
					double pm = q * E_mu;
					if (predict_var) {
						// Var(E[Y|eta]) = q^2 * Var(mu) and Var(Y|eta) = q * (1/k + p0) * mu^2.
						const double var_of_mean = q * q * (std::exp(v) - 1.0) * E_mu * E_mu;
						const double E_mu2 = std::exp(2.0 * m + 2.0 * v);
						const double mean_of_var = q * (1.0 / k + p0) * E_mu2;
						pred_var[i] = var_of_mean + mean_of_var;
					}
					pred_mean[i] = pm;
				}
			}//end hurdle_gamma
			else if (likelihood_type_ == "hurdle_lognormal") {
				CHECK(need_pred_latent_var_for_response_mean_);
				const double p0 = aux_pars_original_[1];
				const double q = 1. - p0;
				const double exp_s2_m1 = std::expm1(aux_pars_[0]);// e^{sigma2} - 1
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					const double E_mu = std::exp(m + 0.5 * v);// E[mu], mu = exp(eta) = lognormal mean given eta
					double pm = q * E_mu;
					if (predict_var) {
						// Var(Y|eta) = q*(V_b + p0*mu^2) with V_b = (e^{sigma2}-1)*mu^2; Var(E[Y|eta]) = q^2*(e^v-1)*E_mu^2
						const double var_of_mean = q * q * std::expm1(v) * E_mu * E_mu;
						const double E_mu2 = std::exp(2.0 * m + 2.0 * v);
						const double mean_of_var = q * (exp_s2_m1 + p0) * E_mu2;
						pred_var[i] = var_of_mean + mean_of_var;
					}
					pred_mean[i] = pm;
				}
			}//end hurdle_lognormal
			else if (IsHurdleRegression()) {
				// Regression zero model: q_i = sigmoid(-zeta_i) is per observation (zeta = pred_var_mean, a fixed-effects predictor).
				// Scale-family form E[Y|eta,y>0] = c1*exp(eta), V_b = c2*exp(2 eta). alpha is treated as fixed (pred_var_var ignored).
				CHECK(need_pred_latent_var_for_response_mean_);
				const string_t base = HurdleRegressionBaseType();
				double c1 = 1., c2 = 0.;
				if (base == "hurdle_gamma") { c1 = 1.; c2 = 1. / aux_pars_[0]; }// V_b = mu^2 / shape
				else if (base == "hurdle_lognormal") { c1 = 1.; c2 = std::expm1(aux_pars_[0]); }// V_b = (e^{sigma2}-1) mu^2
				else {// EGPD base
					const auto& mom = GetEGPDMoments();
					if (!mom.mean_exists) Log::REFatal("PredictResponse: the response mean does not exist for likelihood='%s' when shape >= 1 ", likelihood_type_.c_str());
					if (predict_var && !mom.variance_exists) Log::REFatal("PredictResponse: the response variance does not exist for likelihood='%s' when shape >= 0.5 ", likelihood_type_.c_str());
					if (mom.status != EGPDEvalStatus::kValid) Log::REFatal("PredictResponse: failed to calculate EGPD unit-scale moments ");
					c1 = mom.mean_unit_scale; c2 = mom.variance_unit_scale;
				}
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					const double q = GPBoost::sigmoid_stable(-pred_var_mean[i]);// 1 - pi_i, pi_i = logit^{-1}(zeta_i)
					const double p0 = 1. - q;
					pred_mean[i] = q * c1 * std::exp(m + 0.5 * v);
					if (predict_var) {
						pred_var[i] = q * (c2 + p0 * c1 * c1) * std::exp(2. * m + 2. * v) + q * q * c1 * c1 * std::exp(2. * m + v) * std::expm1(v);
					}
				}
			}//end hurdle regression
			else if (IsZeroInflatedCountRegression()) {
				// E(Y*) = q_i*A1; Var per Section 9 with per-observation q_i = sigmoid(-zeta_i) (pred_var_mean = zeta). A1=exp(m+v/2), A2=exp(2m+2v).
				CHECK(need_pred_latent_var_for_response_mean_);
				const string_t base = ZICountRegressionBaseType();
				const int kind = (base == "zero_inflated_negative_binomial") ? 2 : ((base == "zero_inflated_negative_binomial_1") ? 1 : 0);
				const double shape = (num_aux_pars_ > 0) ? aux_pars_[0] : 0.;
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i]; const double v = std::max(pred_var[i], 0.0);
					const double q = GPBoost::sigmoid_stable(-pred_var_mean[i]); const double p0 = 1. - q;
					const double A1 = std::exp(m + 0.5 * v); const double A2 = std::exp(2. * m + 2. * v); const double V_mu = A2 - A1 * A1;
					pred_mean[i] = q * A1;
					if (predict_var) {
						if (kind == 2) pred_var[i] = q * A1 + q * (1. / shape + p0) * A2 + q * q * V_mu;
						else if (kind == 1) pred_var[i] = q * (1. + shape) * A1 + q * p0 * A2 + q * q * V_mu;
						else pred_var[i] = q * A1 + q * p0 * A2 + q * q * V_mu;
					}
				}
			}//end zero-inflated count regression
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
				// Model: Y = max(0, X)^lambda,  X | eta ~ N(eta, sigma^2),  eta ~ approx N(m, v)
				// Unconditional: X ~ N(m, v + sigma^2). We compute E[Y] and optionally Var(Y).
				const double sigma = aux_pars_[0]; // > 0
				const double lambda = aux_pars_[1]; // > 0
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = pred_var[i];
					const double s = std::sqrt(std::max(v + sigma * sigma, 0.0));
					const double EY = TruncPowerNormalMomentGH(m, s, lambda);
					if (predict_var) {
						const double EY2 = TruncPowerNormalMomentGH(m, s, 2.0 * lambda);
						pred_var[i] = std::max(0.0, EY2 - EY * EY);
					}
					pred_mean[i] = EY;
				}
			}//end "zero_censored_power_transformed_normal"
			else if (IsZeroCensPowNormHetero()) {
				// As above, but sigma_i = exp(pred_var_mean[i]) is the prediction of the second, fixed-effects-only location
				// parameter block (the caller sets pred_var_var = 0 since that block is deterministic given the fixed effects)
				const double lambda = aux_pars_[0];
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					const double sigma = std::exp(pred_var_mean[i]);
					const double s = std::sqrt(std::max(v + sigma * sigma, 0.0));
					const double EY = TruncPowerNormalMomentGH(m, s, lambda);
					if (predict_var) {
						const double EY2 = TruncPowerNormalMomentGH(m, s, 2.0 * lambda);
						pred_var[i] = std::max(0.0, EY2 - EY * EY);
					}
					pred_mean[i] = EY;
				}
			}//end "zero_censored_power_transformed_normal_heteroscedastic"
			else if (likelihood_type_ == "zoctn") {
				CHECK(need_pred_latent_var_for_response_mean_);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double mu_latent = pred_mean[i];
					const double var_latent = std::max(pred_var[i], 0.0);
					const double sigma = aux_pars_[0];
					double var_Z = var_latent + sigma * sigma;
					if (!(var_Z > 0.0) || !std::isfinite(var_Z)) {
						var_Z = sigma * sigma;
					}
					const double s = std::sqrt(var_Z);
					const double pm = ZeroOneCensTransNormalMomentGH(mu_latent, s, false);
					if (predict_var) {
						const double second_moment = ZeroOneCensTransNormalMomentGH(mu_latent, s, true);
						pred_var[i] = std::max(second_moment - pm * pm, 0.0);
					}
					pred_mean[i] = pm;
				}
			}//end "zoctn"
			else if (likelihood_type_ == "zero_one_censored_transformed_beta") {
				CHECK(need_pred_latent_var_for_response_mean_);
				// Predictive response mean/var via Gauss–Hermite 
				const double phi = aux_pars_[0];
				const double u = aux_pars_[1];
				const double inv_sqrt_pi = 1.0 / std::sqrt(3.14159265358979323846);
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					CHECK(std::isfinite(pred_mean[i]));
					CHECK(std::isfinite(pred_var[i]) && pred_var[i] > 0.0);
				}
				const size_t K = GH_nodes_.size();
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = pred_var[i];
					const double sc = std::sqrt(2.0 * v);
					double Ey = 0.0, Ey2 = 0.0;
					for (size_t k = 0; k < K; ++k) {
						const double wk = GH_weights_[k] * inv_sqrt_pi;
						const double eta_k = m + sc * GH_nodes_[k];
						const double mu_k = GPBoost::sigmoid_stable(eta_k);
						const double m1_k = XB_FirstMoment_(mu_k, phi, u);
						Ey += wk * m1_k;
						if (predict_var) {
							const double m2_k = XB_SecondMoment_(mu_k, phi, u);
							Ey2 += wk * m2_k;
						}
					}
					pred_mean[i] = Ey;
					if (predict_var) {
						pred_var[i] = std::max(0.0, Ey2 - Ey * Ey);
					}
				}
			} // end "zero_one_censored_transformed_beta"
			else if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
				CHECK(need_pred_latent_var_for_response_mean_);
				const double k = aux_pars_[0];
				const double xi = aux_pars_[1];
				const size_t K = GH_nodes_.size();
				const double inv_sqrt_pi = 1.0 / std::sqrt(3.14159265358979323846);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					const double m = pred_mean[i];
					const double v = std::max(pred_var[i], 0.0);
					const double sc = std::sqrt(2.0 * v);
					double Ey = 0.0;
					double Ey2 = 0.0;
					for (size_t j = 0; j < K; ++j) {
						const double wj = GH_weights_[j] * inv_sqrt_pi;
						const double eta_j = m + sc * GH_nodes_[j];
						double m1 = 0.0;
						double m2 = 0.0;
						if (predict_var) {
							ZOCG_MomentsGivenEta_(eta_j, k, xi, m1, m2, true);
							Ey += wj * m1;
							Ey2 += wj * m2;
						}
						else {
							ZOCG_MomentsGivenEta_(eta_j, k, xi, m1, m2, false);
							Ey += wj * m1;
						}
					}
					pred_mean[i] = std::min(1.0, std::max(0.0, Ey));
					if (predict_var) {
						const double varY = std::max(0.0, Ey2 - Ey * Ey);
						pred_var[i] = varY;
					}
				}
			}//end "zero_one_censored_shifted_gamma"
			else if (likelihood_type_ == "asymmetric_laplace") {
				if (predict_var) {
					Log::REFatal("PredictResponse: Predictive variances for likelihood of type '%s' is not supported ", likelihood_type_.c_str());
				}
			}
			else {
				NotSupportedForLikelihood(__func__);
			}
		}//end PredictResponse

		/*!
		* \brief Adaptive GH quadrature to calculate predictive mean of response variable
		* \param latent_mean Predictive mean of latent random effects
		* \param latent_var Predictive variances of latent random effects
		* \param second_moment If true, the second moment E( E(yp|bp)^2 | y) is calculated
		*/
		double RespMeanAdaptiveGHQuadrature(const double latent_mean,
			const double latent_var,
			bool second_moment) {
			// Find mode of integrand
			double mode_integrand_last, update;
			double mode_integrand = 0.;
			double sigma2_inv = 1. / latent_var;
			double sqrt_sigma2_inv = std::sqrt(sigma2_inv);
			double c_mult = second_moment ? 2.0 : 1.0;
			for (int it = 0; it < 100; ++it) {
				mode_integrand_last = mode_integrand;
				update = (c_mult * FirstDerivLogCondMeanLikelihood(mode_integrand) - sigma2_inv * (mode_integrand - latent_mean))
					/ (c_mult * SecondDerivLogCondMeanLikelihood(mode_integrand) - sigma2_inv);
				mode_integrand -= update;
				if (std::abs(update) / std::abs(mode_integrand_last) < delta_conv_mode_finding_) {
					break;
				}
			}
			// Adaptive GH quadrature
			double sqrt2_sigma_hat = M_SQRT2 / std::sqrt(-c_mult * SecondDerivLogCondMeanLikelihood(mode_integrand) + sigma2_inv);
			double x_val;
			double mean_resp = 0.;
			for (int j = 0; j < order_GH_; ++j) {
				x_val = sqrt2_sigma_hat * GH_nodes_[j] + mode_integrand;
				double c_mu = CondMeanLikelihood(x_val);
				if (second_moment) {
					c_mu *= c_mu;
				}
				mean_resp += adaptive_GH_weights_[j] * c_mu * GPBoost::normalPDF(sqrt_sigma2_inv * (x_val - latent_mean));
			}
			mean_resp *= sqrt2_sigma_hat * sqrt_sigma2_inv;
			return mean_resp;
		}//end RespMeanAdaptiveGHQuadrature

		/*!
		* \brief Adaptive GH quadrature to calculate E( Var(yp | bp) | y), where Var(yp | bp) is variance of the likelihood given the location parameter bp
		* \param latent_mean Predictive mean of latent random effects
		* \param latent_var Predictive variances of latent random effects
		*/
		double ExpectedValueCondRespVarAdaptiveGHQuadrature(const double latent_mean,
			const double latent_var) {
			// Find mode of integrand
			double mode_integrand_last, update;
			double mode_integrand = 0.;
			double sigma2_inv = 1. / latent_var;
			double sqrt_sigma2_inv = std::sqrt(sigma2_inv);
			for (int it = 0; it < 100; ++it) {
				mode_integrand_last = mode_integrand;
				update = (FirstDerivLogCondVarLikelihood(mode_integrand) - sigma2_inv * (mode_integrand - latent_mean))
					/ (SecondDerivLogCondVarLikelihood(mode_integrand) - sigma2_inv);
				mode_integrand -= update;
				if (std::abs(update) / std::abs(mode_integrand_last) < delta_conv_mode_finding_) {
					break;
				}
			}
			// Adaptive GH quadrature
			double sqrt2_sigma_hat = M_SQRT2 / std::sqrt(-SecondDerivLogCondVarLikelihood(mode_integrand) + sigma2_inv);
			double x_val;
			double mean_resp = 0.;
			for (int j = 0; j < order_GH_; ++j) {
				x_val = sqrt2_sigma_hat * GH_nodes_[j] + mode_integrand;
				mean_resp += adaptive_GH_weights_[j] * CondVarLikelihood(x_val) * GPBoost::normalPDF(sqrt_sigma2_inv * (x_val - latent_mean));
			}
			mean_resp *= sqrt2_sigma_hat * sqrt_sigma2_inv;
			return mean_resp;
		}//end RespMeanAdaptiveGHQuadrature

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
			const data_size_t num_data) const {
			double ll = 0.;
			bool na_inf_flag_11 = false;
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:ll) reduction(||:na_inf_flag_11)
			for (data_size_t i = 0; i < num_data; ++i) {
				int y_test_int = 1;
				double y_test_d = static_cast<double>(y_test[i]);
				// Note: we need to convert from float to double as label_t is float. Unfortunately, the lightGBM part does not allow for setting the LABEL_T_USE_DOUBLE macro in meta.h (multiple bugs...)
				if (label_type() == "int") {
					y_test_int = static_cast<int>(y_test[i]);
				}
				// Find mode of integrand
				double mode_integrand_last, update;
				double mode_integrand = 0.;
				double sigma2_inv = 1. / pred_var[i];
				double sqrt_sigma2_inv = std::sqrt(sigma2_inv);
				for (int it = 0; it < 100; ++it) {
					mode_integrand_last = mode_integrand;
					update = (CalcFirstDerivLogLikOneSample(y_test_d, y_test_int, mode_integrand) - sigma2_inv * (mode_integrand - pred_mean[i]))
						/ (-CalcDiagInformationLogLikOneSample(y_test_d, y_test_int, mode_integrand) - sigma2_inv);
					mode_integrand -= update;
					if (std::abs(update) / std::abs(mode_integrand_last) < delta_conv_mode_finding_) {
						break;
					}
				}
				// Adaptive GH quadrature
				double sqrt2_sigma_hat = M_SQRT2 / std::sqrt(CalcDiagInformationLogLikOneSample(y_test_d, y_test_int, mode_integrand) + sigma2_inv);
				double x_val;
				double likelihood = 0.;
				double log_normalizer = 0.;
				if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
					const double p = GetTweediePower();
					const auto power = likelihood_type_ == "tweedie" ? TransformTweediePowerFromQ(aux_pars_[1], TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_) : TweediePowerTransform{ p, 0., 0. };
					thread_local TweedieSpecialFunctionCache cache;
					const auto normalizer = EvaluateTweedieLogNormalizer(y_test_d, std::log(aux_pars_[0]), p, power.dp_dtheta, power.d2p_dtheta2, TweedieDerivativeOrder::kValue, false, 1000000, &cache);
					if (!normalizer.converged) na_inf_flag_11 = true;
					log_normalizer = normalizer.log_a;
				}
				for (int j = 0; j < order_GH_; ++j) {
					x_val = sqrt2_sigma_hat * GH_nodes_[j] + mode_integrand;
					const double node_log_likelihood = likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" ? LogLikTweedie(y_test_d, x_val, false) : LogLikelihoodOneSample(y_test_d, y_test_int, x_val);
					likelihood += adaptive_GH_weights_[j] * std::exp(node_log_likelihood) * GPBoost::normalPDF(sqrt_sigma2_inv * (x_val - pred_mean[i]));
				}
				likelihood *= sqrt2_sigma_hat * sqrt_sigma2_inv;
				ll += std::log(likelihood) + log_normalizer;
			}
			if (na_inf_flag_11) { Log::REFatal("Tweedie density series did not converge for at least one test point (phi=%g, p = %g).", aux_pars_[0], GetTweediePower()); }
			return -ll;
		}//end TestNegLogLikelihoodAdaptiveGHQuadrature

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
		double TransformToResponseScale(const double value) const {
			if (IsGaussianLikelihood() || likelihood_type_ == "t") {
				return value;
			}
			else if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "binomial_probit" || likelihood_type_ == "quasi_bernoulli_probit") {
				return GPBoost::normalCDF(value);
			}
			else if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit" ||
				likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial" || likelihood_type_ == "quasi_bernoulli_logit") {
				return GPBoost::sigmoid_stable(value);
			}
			else if (IsEGPDLikelihood()) {
				const auto& moments = GetEGPDMoments();
				if (!moments.mean_exists || moments.status != EGPDEvalStatus::kValid) Log::REFatal("TransformToResponseScale: the EGPD response mean is unavailable ");
				return moments.mean_unit_scale * std::exp(value);
			}
			else if (IsHurdleEGPD()) {
				const auto& moments = GetEGPDMoments();
				if (!moments.mean_exists || moments.status != EGPDEvalStatus::kValid) Log::REFatal("The EGPD response mean is unavailable for likelihood='%s' ", likelihood_type_.c_str());
				return (1. - aux_pars_original_[num_aux_pars_ - 1]) * moments.mean_unit_scale * std::exp(value);
			}
			else if (likelihood_type_ == "hurdle_gamma" || likelihood_type_ == "hurdle_lognormal") {
				return (1. - aux_pars_original_[1]) * std::exp(value);
			}
			else if (likelihood_type_ == "zero_inflated_poisson") {
				return (1. - aux_pars_original_[0]) * std::exp(value);
			}
			else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
				return (1. - aux_pars_original_[1]) * std::exp(value);
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				likelihood_type_ == "lognormal") {
				return std::exp(value);
			}
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
				if (value <= 0.) {
					return 0.;
				}
				else {
					return std::exp(aux_pars_[1] * std::log(value));
				}
			}
			else if (IsZeroCensPowNormHetero()) {
				// Same transformation Y = max(0,X)^lambda as above; lambda is aux_pars_[0] here (sigma is a location parameter block)
				if (value <= 0.) {
					return 0.;
				}
				else {
					return std::exp(aux_pars_[0] * std::log(value));
				}
			}
			else if (likelihood_type_ == "zoctn") {
				if (value <= 0.) {
					return 0.;
				}
				else if (value >= 1.) {
					return 1.;
				}
				else {
					const double a = aux_pars_original_[1];
					const double b = aux_pars_[2];
					return GPBoost::sigmoid_stable(a + b * GPBoost::logit(value));
				}
			}//end "zoctn" 
			else if (likelihood_type_ == "zero_one_censored_transformed_beta") {
				const double p = GPBoost::sigmoid_stable(value);
				const double onep2u = 1. + 2 * aux_pars_[1];
				if (p <= aux_pars_[1] / onep2u) {
					return 0.;
				}
				else if (p >= (1. + aux_pars_[1]) / onep2u) {
					return 1.;
				}
				else {
					return onep2u * p - aux_pars_[1];
				}
			}//"zero_one_censored_transformed_beta"
			if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
				const double mu = std::exp(value);
				if (mu <= aux_pars_[1]) {
					return 0.;
				}
				else if (mu >= 1. + aux_pars_[1]) {
					return 1.;
				}
				else {
					return mu - aux_pars_[1];
				}
			}//end "zero_one_censored_shifted_gamma"
			else {
				NotSupportedForLikelihood(__func__);
				return 0.;
			}
		}//end TransformToResponseScale

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
			if (use_random_effects_indices_of_data_) {
				CalcFirstDerivLogLik_PerSample(y_data, y_data_int, location_par, first_deriv_ll_data_scale_);
				ReduceToModeScale(first_deriv_ll_data_scale_, first_deriv_ll_);
			}
			else {//!use_random_effects_indices_of_data_
				CalcFirstDerivLogLik_PerSample(y_data, y_data_int, location_par, first_deriv_ll_);
			}
		}//end CalcFirstDerivLogLik

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
			double* grad) {
			if (likelihood_type_ == "gamma") {
				CHECK(aux_normalizing_constant_has_been_calculated_);
				//gradient for shape parameter is calculated on the log-scale
				double neg_log_grad = 0.;
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					neg_log_grad += w * (location_par[i] + y_data[i] * std::exp(-location_par[i]));
				}
				neg_log_grad -= SumOfWeights() * (std::log(aux_pars_[0]) + 1. - GPBoost::digamma(aux_pars_[0]));
				neg_log_grad -= aux_log_normalizing_constant_;
				neg_log_grad *= aux_pars_[0];
				grad[0] = neg_log_grad;
			}
			else if (likelihood_type_ == "negative_binomial") {
				//gradient for shape parameter is calculated on the log-scale
				double neg_log_grad = 0.;
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					double mu_plus_r = std::exp(location_par[i]) + aux_pars_[0];
					double y_plus_r = y_data_int[i] + aux_pars_[0];
					neg_log_grad += w * aux_pars_[0] * (-GPBoost::digamma(y_plus_r) + std::log(mu_plus_r) + y_plus_r / mu_plus_r);
				}
				neg_log_grad += SumOfWeights() * aux_pars_[0] * (GPBoost::digamma(aux_pars_[0]) - std::log(aux_pars_[0]) - 1);
				grad[0] = neg_log_grad;
			}
			else if (likelihood_type_ == "negative_binomial_1") {
				//gradient for disperison parameter is calculated on the log-scale
				double neg_log_grad = 0.;
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double mu = std::exp(location_par[i]);
					const double r = mu / aux_pars_[0];
					const double C = GPBoost::digamma(y_data_int[i] + r) - GPBoost::digamma(r) - std::log1p(aux_pars_[0]);
					neg_log_grad += w * (r * C + (mu - y_data_int[i]) / (1.0 + aux_pars_[0]));
				}
				grad[0] = neg_log_grad;
			}
			else if (likelihood_type_ == "beta") {
				double grad_log_phi = 0.0;
#pragma omp parallel for schedule(static) reduction(+:grad_log_phi)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double mu = GPBoost::sigmoid_stable_clamped(location_par[i]);
					const double y = y_data[i];
					grad_log_phi += w * (digamma(aux_pars_[0]) - mu * digamma(mu * aux_pars_[0])
						- (1. - mu) * digamma((1. - mu) * aux_pars_[0]) + mu * std::log(y) + (1. - mu) * std::log1p(-y));
				}
				grad[0] = -aux_pars_[0] * grad_log_phi;   // negative log-likelihood gradient
			}
			else if (likelihood_type_ == "t") {
				//gradients are calculated on the log-scale
				double nu_sigma2 = aux_pars_[1] * aux_pars_[0] * aux_pars_[0];
				double neg_log_grad_scale = 0., neg_log_grad_df = 0.;
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad_scale, neg_log_grad_df)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					double res_sq = (y_data[i] - location_par[i]) * (y_data[i] - location_par[i]);
					neg_log_grad_scale -= w * (aux_pars_[1] + 1.) / (nu_sigma2 / res_sq + 1.);
					if (estimate_df_t_) {
						neg_log_grad_df += w * (-aux_pars_[1] * std::log(1 + res_sq / nu_sigma2) + (aux_pars_[1] + 1.) / (1. + nu_sigma2 / res_sq));
					}
				}
				neg_log_grad_scale += SumOfWeights();
				grad[0] = neg_log_grad_scale;
				if (estimate_df_t_) {
					neg_log_grad_df += SumOfWeights() * (-1. + aux_pars_[1] * (GPBoost::digamma((aux_pars_[1] + 1) / 2.) - GPBoost::digamma(aux_pars_[1] / 2.)));
					neg_log_grad_df /= -2.;
					grad[1] = neg_log_grad_df;
				}
			}//end "t"
			else if (IsGaussianLikelihood()) {
				//gradient for variance parameter is calculated on the log-scale
				double neg_log_grad = 0.;
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					double resid = y_data[i] - location_par[i];
					neg_log_grad += w * resid * resid;
				}
				neg_log_grad *= -0.5 / aux_pars_[0];
				neg_log_grad += 0.5 * SumOfWeights();
				grad[0] = neg_log_grad;
			}//end "gaussian"
			else if (likelihood_type_ == "lognormal") {
				double neg_log_grad = 0.;
				const double s2 = aux_pars_[0];
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double z = std::log(y_data[i]) - (location_par[i] - 0.5 * s2);
					neg_log_grad += w * ((z + 1.0) * 0.5 - (z * z) / (2.0 * s2));
				}
				grad[0] = neg_log_grad;
			}//end lognormal
			else if (likelihood_type_ == "beta_binomial") {
				CHECK(has_weights_);
				double neg_log_grad = 0.0;
				const double phi_raw = aux_pars_[0];
				const double phi = (phi_raw > 0.0 && std::isfinite(phi_raw)) ? phi_raw : 1e-16;
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double mu = GPBoost::sigmoid_stable_clamped(location_par[i]);
					const double a = mu * phi;
					const double b = (1.0 - mu) * phi;
					const double w = weights_[i];
					const double k = y_data[i] * w;
					const double dL_dphi = mu * (GPBoost::digamma(k + a) - GPBoost::digamma(a))
						+ (1.0 - mu) * (GPBoost::digamma(w - k + b) - GPBoost::digamma(b))
						- GPBoost::digamma(w + phi) + GPBoost::digamma(phi);
					neg_log_grad -= phi * dL_dphi;
				}
				grad[0] = neg_log_grad;
			}//end "beta_binomial"
			else if (likelihood_type_ == "hurdle_gamma") {
				CHECK(aux_normalizing_constant_has_been_calculated_);
				const double p0 = aux_pars_original_[1];
				const double q = 1. - aux_pars_original_[1];// = 1 - p0
				double Wpos = 0.0, Wzero = 0.0, sum_for_gamma = 0.0;
#pragma omp parallel for schedule(static) reduction(+:Wpos,Wzero,sum_for_gamma)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					if (y_data[i] > 0.0) {
						Wpos += w;
						sum_for_gamma += w * (location_par[i] + y_data[i] * std::exp(-location_par[i]));
					}
					else {
						Wzero += w;
					}
				}
				double neg_log_grad_gamma = sum_for_gamma - Wpos * (std::log(aux_pars_[0]) + 1. - GPBoost::digamma(aux_pars_[0])) -
					aux_log_normalizing_constant_;
				grad[0] = neg_log_grad_gamma * aux_pars_[0];//grad on log(gamma)
				// Gradient on log odds, log(r) = log(p0 / (1-p0)).
				grad[1] = p0 * Wpos - q * Wzero;
			}//end "hurdle_gamma"
			else if (likelihood_type_ == "hurdle_lognormal") {
				const double p0 = aux_pars_original_[1];
				const double q = 1. - p0;
				const double s2 = aux_pars_[0];
				double Wpos = 0., Wzero = 0., neg_log_grad_s2 = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:Wpos, Wzero, neg_log_grad_s2)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					if (y_data[i] > 0.0) {
						Wpos += w;
						const double z = std::log(y_data[i]) - (location_par[i] - 0.5 * s2);
						neg_log_grad_s2 += w * ((z + 1.0) * 0.5 - (z * z) / (2.0 * s2));// same per-obs term as base lognormal, wrt log(sigma2)
					}
					else Wzero += w;
				}
				grad[0] = neg_log_grad_s2;// already on log(sigma2) scale (matches base lognormal)
				grad[1] = p0 * Wpos - q * Wzero;// wrt rho = logit(p0)
			}//end "hurdle_lognormal"
			else if (IsHurdleRegression()) {
				// Base-likelihood auxiliary gradient over the positive observations only (the structural zero is a separate fixed-effects block).
				const string_t base = HurdleRegressionBaseType();
				if (base == "hurdle_gamma") {
					double Wpos = 0., sum_for_gamma = 0.;
#pragma omp parallel for schedule(static) reduction(+:Wpos, sum_for_gamma)
					for (data_size_t i = 0; i < num_data_; ++i) {
						if (y_data[i] > 0.) { const double w = has_weights_ ? weights_[i] : 1.0; Wpos += w; sum_for_gamma += w * (location_par[i] + y_data[i] * std::exp(-location_par[i])); }
					}
					grad[0] = (sum_for_gamma - Wpos * (std::log(aux_pars_[0]) + 1. - GPBoost::digamma(aux_pars_[0])) - aux_log_normalizing_constant_) * aux_pars_[0];
				}
				else if (base == "hurdle_lognormal") {
					const double s2 = aux_pars_[0];
					double g = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:g)
					for (data_size_t i = 0; i < num_data_; ++i) {
						if (y_data[i] > 0.) { const double w = has_weights_ ? weights_[i] : 1.0; const double z = std::log(y_data[i]) - (location_par[i] - 0.5 * s2); g += w * ((z + 1.0) * 0.5 - (z * z) / (2.0 * s2)); }
					}
					grad[0] = g;
				}
				else {// EGPD base: score over positive observations, on the optimizer scale
					for (int j = 0; j < num_aux_pars_estim_; ++j) {
						double score_sum = 0.;
#pragma omp parallel for schedule(static) reduction(+:score_sum) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							if (y_data[i] <= 0.) continue;
							const double w = has_weights_ ? weights_[i] : 1.;
							const auto r = EvaluateEGPD(y_data[i], location_par[i]);
							if (r.status == EGPDEvalStatus::kValid) score_sum += w * r.d_aux_optimizer[j];
						}
						grad[j] = -score_sum;
					}
				}
			}//end hurdle regression
			else if (likelihood_type_ == "zero_inflated_poisson") {
				// Gradient of the negative log-likelihood wrt rho = log(p0/(1-p0)) = logit(p0): sum_i w_i (pi_i - tau_i),
				// with pi_i = p0 and tau_i = w_i^post (posterior structural-zero prob) for y_i = 0, tau_i = 0 for y_i > 0.
				const double p0 = aux_pars_original_[0];
				double grad_rho = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:grad_rho)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double wt = has_weights_ ? weights_[i] : 1.0;
					if (y_data_int[i] == 0) {
						const double mu = std::exp(location_par[i]);
						double wpost;
						ZIPoissonZeroLogMixture(mu, p0, wpost);
						grad_rho += wt * (p0 - wpost);
					}
					else {
						grad_rho += wt * p0;
					}
				}
				grad[0] = grad_rho;
			}//end "zero_inflated_poisson"
			else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
				// grad[0]: negative-ll gradient wrt log(shape/dispersion); grad[1]: wrt rho = logit(p0).
				const double p0 = aux_pars_original_[1];
				double grad_shape = 0., grad_rho = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:grad_shape, grad_rho)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double wt = has_weights_ ? weights_[i] : 1.0;
					if (y_data_int[i] > 0) {
						grad_shape += wt * NegLLGradShapeNegBinFamilyPos(y_data_int[i], location_par[i]);
						grad_rho += wt * p0;
					}
					else {
						const double mu = std::exp(location_par[i]);
						ZICountZeroMass z; FillZeroMassZICount(mu, z);
						double w, v; ZICountZeroLogMixture(p0, z.b0, w, v);
						grad_shape += wt * (-v * z.g0);// d(-l)/d log(shape) = -v*g0 at a zero count
						grad_rho += wt * (p0 - w);
					}
				}
				grad[0] = grad_shape;
				grad[1] = grad_rho;
			}//end zero_inflated_negative_binomial(_1)
			else if (IsZeroInflatedCountRegression()) {
				// Base-count auxiliary gradient (shape for NB2, dispersion for NB1; Poisson has none). Per-obs pi from zeta.
				const string_t base = ZICountRegressionBaseType();
				if (base != "zero_inflated_poisson") {
					const bool nb2 = (base == "zero_inflated_negative_binomial");
					double grad_shape = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:grad_shape)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double wt = has_weights_ ? weights_[i] : 1.0;
						if (y_data_int[i] > 0) {
							const double mu = std::exp(location_par[i]);
							if (nb2) {
								const double mu_plus_r = mu + aux_pars_[0]; const double y_plus_r = y_data_int[i] + aux_pars_[0];
								grad_shape += wt * (aux_pars_[0] * (-GPBoost::digamma(y_plus_r) + std::log(mu_plus_r) + y_plus_r / mu_plus_r) + aux_pars_[0] * (GPBoost::digamma(aux_pars_[0]) - std::log(aux_pars_[0]) - 1.));
							}
							else {
								const double r = mu / aux_pars_[0]; const double C = GPBoost::digamma(y_data_int[i] + r) - GPBoost::digamma(r) - std::log1p(aux_pars_[0]);
								grad_shape += wt * (r * C + (mu - y_data_int[i]) / (1.0 + aux_pars_[0]));
							}
						}
						else {
							ZICountRegQuant o; ZICountRegressionQuantities(0, location_par[i], location_par[i + num_data_], o);
							ZICountZeroMass z; FillZeroMassCountRegression(std::exp(location_par[i]), z);
							grad_shape += wt * (-o.v * z.g0);
						}
					}
					grad[0] = grad_shape;
				}
			}//end zero-inflated count regression
			else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
				double grad_log_sigma = 0.0, grad_log_lambda = 0.0;
#pragma omp parallel for schedule(static) reduction(+:grad_log_sigma,grad_log_lambda)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double mu = location_par[i];
					const double s = aux_pars_[0];
					const double lambda = aux_pars_[1];
					const double yi = y_data[i];
					if (yi <= 0.0) {
						// d/d log(sigma) log f(0) = r * mu / s, r = phi(a0)/Phi(a0), a0 = -mu/s
						const double a0 = -mu / s;
						const double r = GPBoost::InvMillsRatioNormalPhi(a0);
						grad_log_sigma += w * (r * mu / s);
						// d/d log(lambda) is zero at y=0
					}
					else {
						const double logy = std::log(yi);
						const double u = std::exp((1.0 / lambda) * logy);   // yi^(1/lambda)
						const double z = (u - mu) / s;
						grad_log_sigma += w * (-1.0 + z * z);// d/d log(sigma) log f
						grad_log_lambda += w * (-1.0 - (logy / lambda) + (z * u * logy) / (lambda * s));// d/d log(lambda) log f
					}
				}
				grad[0] = -grad_log_sigma;
				grad[1] = -grad_log_lambda;
			}//end "zero_censored_power_transformed_normal"
			else if (IsZeroCensPowNormHetero()) {
				// lambda is the only auxiliary parameter (sigma is the second location parameter block). Zeros do not depend on lambda
				const double lambda = aux_pars_[0];
				double grad_log_lambda = 0.0;
#pragma omp parallel for schedule(static) reduction(+:grad_log_lambda)
				for (data_size_t i = 0; i < num_data_; ++i) {
					if (y_data[i] > 0.0) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double s = std::exp(location_par[i + num_data_]);
						const double logy = std::log(y_data[i]);
						const double u = std::exp(logy / lambda);
						const double z = (u - location_par[i]) / s;
						grad_log_lambda += w * (-1.0 - (logy / lambda) + (z * u * logy) / (lambda * s));// d/d log(lambda) log f
					}
				}
				grad[0] = -grad_log_lambda;
			}//end "zero_censored_power_transformed_normal_heteroscedastic"
			else if (likelihood_type_ == "zoctn") {
				const double a = aux_pars_original_[1];
				const double b = aux_pars_[2];
				double grad_log_sigma = 0.0, grad_log_a = 0.0, grad_log_b = 0.0;
#pragma omp parallel for schedule(static) reduction(+:grad_log_sigma,grad_log_a,grad_log_b)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double mu = location_par[i];
					const double sigma = aux_pars_[0];
					const double yi = y_data[i];
					if (yi <= 0.0) {
						// d/d log(sigma) log f(0) = r * mu / sigma, r = phi(a0)/Phi(a0), a0 = -mu/sigma
						const double a0 = -mu / sigma;
						const double r0 = GPBoost::InvMillsRatioNormalPhi(a0);
						grad_log_sigma += w * (r0 * mu / sigma);
						// d/d log(a) and d/d log(b) are zero for y <= 0
					}
					else if (yi >= 1.0) {
						// d/d log(sigma) log f(1) = (1 - mu) * r1 / sigma, r1 = phi(v)/(1-Phi(v)), v = (1-mu)/sigma
						const double v = (1.0 - mu) / sigma;
						const double r1 = GPBoost::InvMillsRatioNormalOneMinusPhi(v);
						grad_log_sigma += w * ((1.0 - mu) * r1 / sigma);
						// d/d log(a) and d/d log(b) are zero for y >= 1
					}
					else {// 0 < y < 1: continuous part					
						const double logit_y = GPBoost::logit(yi);
						const double t = (logit_y - a) / b;
						const double x = GPBoost::sigmoid_stable(t);
						const double z = (x - mu) / sigma;
						const double A = (z / sigma) * x * (1.0 - x);
						const double B = 1.0 - 2.0 * x;
						const double C = A - B;
						grad_log_sigma += w * (z * z - 1.0);
						grad_log_a += w * (C / b);
						grad_log_b += w * (C * t - 1.0);
					}
				}
				grad[0] = -grad_log_sigma;
				grad[1] = -grad_log_a;
				grad[2] = -grad_log_b;
			}//end "zoctn"
			else if (likelihood_type_ == "zero_one_censored_transformed_beta") {
				// Interior y in (0,1): use analytic derivative for log-phi.
				// Boundaries (y==0 or y==1): use central differences in log-space.
				// For u: use central differences in log-space everywhere.
				const double phi0 = aux_pars_[0];
				const double u0 = aux_pars_[1];
				const double h_log_phi = 1e-4;// log-space step sizes
				const double h_log_u = 1e-4;
				const double phi_base = std::max(phi0, 1e-300);
				const double u_base = std::max(u0, 1e-300);
				const double phi_minus = phi_base * std::exp(-h_log_phi);
				const double phi_plus = phi_base * std::exp(+h_log_phi);
				const double u_minus = u_base * std::exp(-h_log_u);
				const double u_plus = u_base * std::exp(+h_log_u);
				double dlogL_dlogphi = 0.0;
				double dlogL_dlogu = 0.0;
#pragma omp parallel for schedule(static) reduction(+:dlogL_dlogphi,dlogL_dlogu)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double yi = y_data[i];
					const double eta = location_par[i];
					double d_ll_d_logphi_i = 0.0;
					if (yi > 0.0 && yi < 1.0) { // analytic interior derivative
						const double mu = GPBoost::sigmoid_stable_clamped(eta);
						const double a = mu * phi0;
						const double b = (1.0 - mu) * phi0;
						const double c = 1.0 + 2.0 * u0;
						const double t = (yi + u0) / c;
						const double d_ll_d_phi = GPBoost::digamma(phi0) - mu * GPBoost::digamma(a)
							- (1.0 - mu) * GPBoost::digamma(b) + mu * std::log(t) + (1.0 - mu) * std::log1p(-t);
						d_ll_d_logphi_i = phi0 * d_ll_d_phi;
					}
					else {
						// boundary points: use central FD directly in log(phi)
						const double ll_m = LogLikZeroOneCensTransfBeta_at(yi, eta, phi_minus, u0, true);
						const double ll_p = LogLikZeroOneCensTransfBeta_at(yi, eta, phi_plus, u0, true);
						d_ll_d_logphi_i = (ll_p - ll_m) / (2.0 * h_log_phi);
					}
					dlogL_dlogphi += w * d_ll_d_logphi_i;
					const double ll_u_m = LogLikZeroOneCensTransfBeta_at(yi, eta, phi0, u_minus, true);
					const double ll_u_p = LogLikZeroOneCensTransfBeta_at(yi, eta, phi0, u_plus, true);
					const double d_ll_d_logu_i = (ll_u_p - ll_u_m) / (2.0 * h_log_u);
					dlogL_dlogu += w * d_ll_d_logu_i;
				}
				grad[0] = -dlogL_dlogphi;
				grad[1] = -dlogL_dlogu;
			} // end "zero_one_censored_transformed_beta"
			else if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
				const double k0 = std::max(aux_pars_[0], 1e-12);
				const double xi0 = std::max(aux_pars_[1], 1e-12);
				const double h_log_k = 1e-4;
				const double k_minus = k0 * std::exp(-h_log_k);
				const double k_plus = k0 * std::exp(+h_log_k);
				double dlogL_dlogk = 0.0;
				double dlogL_dlogxi = 0.0;
#pragma omp parallel for schedule(static) reduction(+:dlogL_dlogk,dlogL_dlogxi)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double yi = y_data[i];
					const double eta = location_par[i];
					const double mu = std::exp(eta);
					const double tiny = 1e-300;

					if (yi <= 0.0) {
						// y = 0 : log P with t = k*xi/mu
						const double t = (k0 * xi0) / std::max(mu, 1e-12);
						const double G = std::max(GPBoost::RegLowerGamma(k0, t), tiny);
						const double p = std::exp(-t + (k0 - 1.0) * std::log(std::max(t, tiny)) - std::lgamma(k0));
						const double Q = p / G;                                   // (1/P) dP/dt
						const double dlogL_dlogxi_i = t * Q;                      // dt/dlogxi = t
						const double ll_km = LogLikZeroOneCensGamma_at(yi, eta, k_minus, xi0, true);
						const double ll_kp = LogLikZeroOneCensGamma_at(yi, eta, k_plus, xi0, true);
						const double dlogL_dlogk_i = (ll_kp - ll_km) / (2.0 * h_log_k);
						dlogL_dlogk += w * dlogL_dlogk_i;
						dlogL_dlogxi += w * dlogL_dlogxi_i;

					}
					else if (yi >= 1.0) {
						// y = 1 : log H with t = k*(1+xi)/mu
						const double a = 1.0 + xi0;
						const double t = (k0 * a) / std::max(mu, 1e-12);
						const double G = GPBoost::RegLowerGamma(k0, t);
						const double H = std::max(1.0 - G, tiny);
						const double p = std::exp(-t + (k0 - 1.0) * std::log(std::max(t, tiny)) - std::lgamma(k0));
						const double Q = p / H;                                   // (1/H) dH/dt = -(1/H) dG/dt = -p/H
						const double dlogL_dlogxi_i = -(xi0 / a) * t * Q;         // dt/dlogxi = (xi/(1+xi)) t
						const double ll_km = LogLikZeroOneCensGamma_at(yi, eta, k_minus, xi0, true);
						const double ll_kp = LogLikZeroOneCensGamma_at(yi, eta, k_plus, xi0, true);
						const double dlogL_dlogk_i = (ll_kp - ll_km) / (2.0 * h_log_k);
						dlogL_dlogk += w * dlogL_dlogk_i;
						dlogL_dlogxi += w * dlogL_dlogxi_i;

					}
					else {
						// 0 < y < 1 : continuous density; fully analytic
						const double z = yi + xi0;
						const double dlogf_dk = std::log(std::max(z, tiny))
							- std::log(std::max(mu, 1e-12))
							- (z / std::max(mu, 1e-12))
							+ std::log(k0) + 1.0 - digamma(k0);
						const double dlogf_dlogk_i = k0 * dlogf_dk;
						const double dlogf_dxi = (k0 - 1.0) / std::max(z, tiny) - (k0 / std::max(mu, 1e-12));
						const double dlogf_dlogxi_i = xi0 * dlogf_dxi;
						dlogL_dlogk += w * dlogf_dlogk_i;
						dlogL_dlogxi += w * dlogf_dlogxi_i;
					}
				}
				grad[0] = -dlogL_dlogk;     // gradient of *negative* log-likelihood
				grad[1] = -dlogL_dlogxi;
			} // end "zero_one_censored_shifted_gamma"
			else if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
				// SetAuxPars() invalidates the normalizer cache. Some optimization paths evaluate the auxiliary
				// gradient before reevaluating the objective, so refresh the cache here when necessary.
				CalculateLogNormalizingConstant(y_data, y_data_int);
				const double phi = aux_pars_[0];
				const double p = GetTweediePower();
				const double dp = likelihood_type_ == "tweedie" ? TransformTweediePowerFromQ(aux_pars_[1], TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_).dp_dtheta : 0.;
				// Verify that the normalizer-derivative aggregates correspond to the current auxiliary parameters.
				CHECK(normalizing_constant_has_been_calculated_);
				CHECK(TwoNumbersAreEqual<double>(tweedie_cached_phi_, phi));
				CHECK(TwoNumbersAreEqual<double>(tweedie_cached_p_, p));
				double canonical_sum = 0., power_sum = 0.;
#pragma omp parallel for schedule(static) reduction(+:canonical_sum,power_sum)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.;
					const double eta = location_par[i];
					const auto location = EvaluateTweedieLocation(y_data[i], eta, std::log(phi), p);
					canonical_sum += w * location.canonical;
					if (likelihood_type_ == "tweedie") {
						const double coefficient_a = eta / (2. - p) - 1. / ((2. - p) * (2. - p));
						const double coefficient_b = eta / (p - 1.) + 1. / ((p - 1.) * (p - 1.));
						power_sum += w * dp * TweedieSignedLogSum(coefficient_a, location.log_scaled_a, coefficient_b, location.log_scaled_b);
					}
				}
				grad[0] = -tweedie_sum_d_log_a_rho_ + canonical_sum;
				if (likelihood_type_ == "tweedie") grad[1] = -tweedie_sum_d_log_a_theta_ - power_sum;
			}
			else if (IsEGPDLikelihood()) {
				for (int j = 0; j < num_aux_pars_estim_; ++j) {
					double score_sum = 0.;
#pragma omp parallel for schedule(static) reduction(+:score_sum) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.;
						if (w == 0.) continue;
						const auto result = EvaluateEGPD(y_data[i], location_par[i]);
						if (result.status == EGPDEvalStatus::kValid) score_sum += w * result.d_aux_optimizer[j];
					}
					grad[j] = -score_sum;
				}
			}
			else if (IsHurdleEGPD()) {
				const int ip0 = num_aux_pars_ - 1;
				const double p0 = aux_pars_original_[ip0];
				for (int j = 0; j < ip0; ++j) {// base EGPD auxiliary parameters (score over positive observations)
					double score_sum = 0.;
#pragma omp parallel for schedule(static) reduction(+:score_sum) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.;
						if (w == 0. || y_data[i] <= 0.) continue;
						const auto result = EvaluateEGPD(y_data[i], location_par[i]);
						if (result.status == EGPDEvalStatus::kValid) score_sum += w * result.d_aux_optimizer[j];
					}
					grad[j] = -score_sum;
				}
				double Wpos = 0., Wzero = 0.;
#pragma omp parallel for schedule(static) reduction(+:Wpos, Wzero) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.;
					if (y_data[i] > 0.) Wpos += w; else Wzero += w;
				}
				grad[ip0] = p0 * Wpos - (1. - p0) * Wzero;// gradient wrt rho = logit(p0)
			}//end hurdle EGPD variants
			else if (likelihood_type_ == "asymmetric_laplace") {
				//gradient for scale parameter is calculated on the log-scale
				double neg_log_grad = 0.;
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					double indicator = (y_data[i] <= location_par[i]) ? 1.0 : 0.0;
					neg_log_grad += w * (y_data[i] - location_par[i]) * (indicator - quantile_) / aux_pars_[0];
				}
				neg_log_grad += SumOfWeights();
				grad[0] = neg_log_grad;
			}//end "asymmetric_laplace"
			else if (num_aux_pars_estim_ > 0) {
				NotSupportedForLikelihood(__func__);
			}
		}//end CalcGradNegLogLikAuxPars

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
			double* deriv_information_aux_par) const {
			if (approximation_type_ == "laplace") {
				if (likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p") {
					CHECK(ind_aux_par == 0 || (likelihood_type_ == "tweedie" && ind_aux_par == 1));
					const double dp = likelihood_type_ == "tweedie" ? TransformTweediePowerFromQ(aux_pars_[1], TWEEDIE_POWER_LOWER_, TWEEDIE_POWER_UPPER_).dp_dtheta : 0.;
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.;
						const double s = FirstDerivLogLikTweedie(y_data[i], location_par[i]);
						const double information = InformationLogLikTweedie(y_data[i], location_par[i]);
						if (ind_aux_par == 0) {
							second_deriv_loc_aux_par[i] = -w * s;
							deriv_information_aux_par[i] = -w * information;
						}
						else {
							second_deriv_loc_aux_par[i] = -w * dp * location_par[i] * s;
							deriv_information_aux_par[i] = w * dp * (s - location_par[i] * information);
						}
					}
				}
				else if (IsEGPDLikelihood()) {
					CHECK(ind_aux_par >= 0 && ind_aux_par < num_aux_pars_estim_);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.;
						if (w == 0.) { second_deriv_loc_aux_par[i] = 0.; deriv_information_aux_par[i] = 0.; }
						else {
							const auto result = EvaluateEGPD(y_data[i], location_par[i]);
							if (result.status == EGPDEvalStatus::kValid) {
								second_deriv_loc_aux_par[i] = w * result.d_eta_aux_optimizer[ind_aux_par];
								deriv_information_aux_par[i] = -w * result.d2_eta_aux_optimizer[ind_aux_par];
							}
							else { second_deriv_loc_aux_par[i] = std::numeric_limits<double>::quiet_NaN(); deriv_information_aux_par[i] = std::numeric_limits<double>::quiet_NaN(); }
						}
					}
				}
				else if (IsHurdleEGPD()) {
					CHECK(ind_aux_par >= 0 && ind_aux_par < num_aux_pars_estim_);
					const int ip0 = num_aux_pars_ - 1;// structural-zero parameter index (rho); decouples from eta
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.;
						if (w == 0. || ind_aux_par == ip0 || y_data[i] <= 0.) { second_deriv_loc_aux_par[i] = 0.; deriv_information_aux_par[i] = 0.; }
						else {
							const auto result = EvaluateEGPD(y_data[i], location_par[i]);
							if (result.status == EGPDEvalStatus::kValid) {
								second_deriv_loc_aux_par[i] = w * result.d_eta_aux_optimizer[ind_aux_par];
								deriv_information_aux_par[i] = -w * result.d2_eta_aux_optimizer[ind_aux_par];
							}
							else { second_deriv_loc_aux_par[i] = std::numeric_limits<double>::quiet_NaN(); deriv_information_aux_par[i] = std::numeric_limits<double>::quiet_NaN(); }
						}
					}
				}
				else if (likelihood_type_ == "gamma") {
					//note: gradient wrt to aux_pars_[0] on the log-scale
					CHECK(ind_aux_par == 0);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						second_deriv_loc_aux_par[i] = w * aux_pars_[0] * (y_data[i] * std::exp(-location_par[i]) - 1.);
						deriv_information_aux_par[i] = w * (second_deriv_loc_aux_par[i] + aux_pars_[0]);
					}
				}
				else if (likelihood_type_ == "negative_binomial") {
					//gradient for shape parameter is calculated on the log-scale
					CHECK(ind_aux_par == 0);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double mu = std::exp(location_par[i]);
						const double mu_plus_r = mu + aux_pars_[0];
						const double mu_r_div_mu_plus_r_sqr = mu * aux_pars_[0] / (mu_plus_r * mu_plus_r);
						second_deriv_loc_aux_par[i] = w * mu_r_div_mu_plus_r_sqr * (y_data_int[i] - mu);
						deriv_information_aux_par[i] = w * -mu_r_div_mu_plus_r_sqr * (y_data_int[i] * (aux_pars_[0] - mu) - 2 * aux_pars_[0] * mu) / mu_plus_r;
					}
				}
				else if (likelihood_type_ == "negative_binomial_1") {
					CHECK(ind_aux_par == 0);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const int    y = y_data_int[i];
						const double mu = std::exp(location_par[i]);
						const double r = mu / aux_pars_[0];
						const double dig_diff = GPBoost::digamma(y + r) - GPBoost::digamma(r);
						const double tri_diff = GPBoost::trigamma(y + r) - GPBoost::trigamma(r);
						const double tet_diff = GPBoost::tetragamma(y + r) - GPBoost::tetragamma(r);
						const double C = dig_diff - std::log1p(aux_pars_[0]);
						second_deriv_loc_aux_par[i] = -w * (r * C + r * r * tri_diff + mu / (1.0 + aux_pars_[0]));
						deriv_information_aux_par[i] = w * (3.0 * r * r * tri_diff + r * r * r * tet_diff + r * C + mu / (1.0 + aux_pars_[0]));
					}
				}//end negative_binomial_1
				else if (likelihood_type_ == "beta") {
					CHECK(ind_aux_par == 0);
					const double phi_raw = aux_pars_[0];
					const double phi = (phi_raw > 0.0 && std::isfinite(phi_raw)) ? phi_raw : 1e-16;
#pragma omp parallel for schedule(static)
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
						const double Dlt_tri = (1.0 - mu) * tri1 - mu * tri2;
						const double Dlt_tet = (1.0 - mu) * tet1 + mu * tet2;
						const double cross_deriv = -(phi * d * C + phi * phi * d * Dlt_tri);
						const double term1 = 2.0 * phi * phi * d * d * S;
						const double term2 = phi * phi * phi * d * d * Dlt_tet;
						const double term3 = -phi * d * (1.0 - 2.0 * mu) * C;
						const double term4 = -phi * phi * d * (1.0 - 2.0 * mu) * Dlt_tri;
						second_deriv_loc_aux_par[i] = w * cross_deriv;
						deriv_information_aux_par[i] = w * (term1 + term2 + term3 + term4);
					}
				}
				else if (likelihood_type_ == "t") {
					CHECK(ind_aux_par == 0 || ind_aux_par == 1);
					if (ind_aux_par == 0) {
						//gradient for scale parameter is calculated on the log-scale
						const double sigma2 = aux_pars_[0] * aux_pars_[0];
						const double nu_sigma2 = aux_pars_[1] * sigma2;
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							const double res = y_data[i] - location_par[i];
							const double res_sq = res * res;
							const double denom = nu_sigma2 + res_sq;
							const double denom_sq = denom * denom;
							second_deriv_loc_aux_par[i] = w * -2. * (aux_pars_[1] + 1.) * aux_pars_[1] * res * sigma2 / denom_sq;
							deriv_information_aux_par[i] = w * 2. * (aux_pars_[1] + 1.) * aux_pars_[1] * sigma2 * (3. * res_sq - nu_sigma2) / (denom_sq * denom);
						}
					}
					else if (ind_aux_par == 1) {
						CHECK(estimate_df_t_);
						//gradient for df parameter is calculated on the log-scale
						const double sigma2 = aux_pars_[0] * aux_pars_[0];
						const double nu_sigma2 = aux_pars_[1] * sigma2;
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							const double res = y_data[i] - location_par[i];
							const double res_sq = res * res;
							const double denom = nu_sigma2 + res_sq;
							const double denom_sq = denom * denom;
							second_deriv_loc_aux_par[i] = w * aux_pars_[1] * res * (res_sq - sigma2) / denom_sq;
							deriv_information_aux_par[i] = w * -aux_pars_[1] * (res_sq * res_sq + nu_sigma2 * sigma2 -
								3. * res_sq * sigma2 * (aux_pars_[1] + 1)) / (denom_sq * denom);
						}
					}
				}//end "t"
				else if (IsGaussianLikelihood()) {
					//gradient for variance parameter is calculated on the log-scale
					CHECK(ind_aux_par == 0);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						second_deriv_loc_aux_par[i] = w * (location_par[i] - y_data[i]) / aux_pars_[0];
						deriv_information_aux_par[i] = w * -1. / aux_pars_[0];
					}
				}//end "gaussian"
				else if (likelihood_type_ == "lognormal") {
					CHECK(ind_aux_par == 0);
					const double s2 = aux_pars_[0];
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						second_deriv_loc_aux_par[i] = w * (-std::log(y_data[i]) + location_par[i]) / s2;
						deriv_information_aux_par[i] = w * (-1.0 / s2);
					}
				}//end "lognormal"
				else if (likelihood_type_ == "beta_binomial") {
					CHECK(ind_aux_par == 0);
					CHECK(has_weights_);
					const double phi_raw = aux_pars_[0];
					const double phi = (phi_raw > 0.0 && std::isfinite(phi_raw)) ? phi_raw : 1e-16;
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = weights_[i];
						if (w <= 0.0) {
							second_deriv_loc_aux_par[i] = 0.;
							deriv_information_aux_par[i] = 0.;
						}
						else {
							const double mu = GPBoost::sigmoid_stable_clamped(location_par[i]);
							const double s = mu * (1.0 - mu);
							const double a = mu * phi, b = (1.0 - mu) * phi;
							const double k = y_data[i] * w;
							const double Delta = GPBoost::digamma(k + a) - GPBoost::digamma(a)
								- GPBoost::digamma(w - k + b) + GPBoost::digamma(b);
							const double S1 = GPBoost::trigamma(k + a) - GPBoost::trigamma(a);
							const double S2 = GPBoost::trigamma(w - k + b) - GPBoost::trigamma(b);
							const double T1 = GPBoost::tetragamma(k + a) - GPBoost::tetragamma(a);
							const double T2 = GPBoost::tetragamma(w - k + b) - GPBoost::tetragamma(b);
							const double dDelta_dphi = mu * S1 + (1.0 - mu) * S2;
							const double dSsum_dphi = mu * T1 + (1.0 - mu) * T2;
							const double sp = s * (1.0 - 2.0 * mu);
							second_deriv_loc_aux_par[i] = phi * s * (Delta + phi * dDelta_dphi);
							const double term1 = phi * sp * Delta;
							const double term2 = phi * phi * sp * dDelta_dphi;
							const double term3 = 2.0 * phi * phi * s * s * (S1 + S2);
							const double term4 = phi * phi * phi * s * s * dSsum_dphi;
							deriv_information_aux_par[i] = -(term1 + term2 + term3 + term4);
						}
					}
				}//end "beta_binomial"
				else if (likelihood_type_ == "hurdle_gamma") {
					CHECK(ind_aux_par == 0 || ind_aux_par == 1);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.0) {
							second_deriv_loc_aux_par[i] = 0.0;
							deriv_information_aux_par[i] = 0.0;
							continue;
						}
						const double y_exp_neg_loc = y_data[i] * std::exp(-location_par[i]);
						if (ind_aux_par == 0) { // log gamma
							second_deriv_loc_aux_par[i] = w * aux_pars_[0] * (y_exp_neg_loc - 1.0);
							deriv_information_aux_par[i] = w * aux_pars_[0] * y_exp_neg_loc; // = W_i
						}
						else { // log r
							second_deriv_loc_aux_par[i] = 0.;
							deriv_information_aux_par[i] = 0.;
						}
					}
				}//end "hurdle_gamma"
				else if (likelihood_type_ == "hurdle_lognormal") {
					CHECK(ind_aux_par == 0 || ind_aux_par == 1);
					const double s2 = aux_pars_[0];
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.0 || ind_aux_par == 1) {// zero part decouples; rho (ind 1) has no coupling with eta
							second_deriv_loc_aux_par[i] = 0.0;
							deriv_information_aux_par[i] = 0.0;
							continue;
						}
						// ind_aux_par == 0 (log sigma2), positive observation: base lognormal cross-derivatives
						second_deriv_loc_aux_par[i] = w * (-std::log(y_data[i]) + location_par[i]) / s2;
						deriv_information_aux_par[i] = w * (-1.0 / s2);
					}
				}//end "hurdle_lognormal"
				else if (IsHurdleRegression()) {
					const string_t base = HurdleRegressionBaseType();
					const bool egpd = (base != "hurdle_gamma" && base != "hurdle_lognormal");
					const double s2 = aux_pars_[0];
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data[i] <= 0.0) { second_deriv_loc_aux_par[i] = 0.; deriv_information_aux_par[i] = 0.; continue; }
						if (base == "hurdle_gamma") {
							const double y_exp_neg_loc = y_data[i] * std::exp(-location_par[i]);
							second_deriv_loc_aux_par[i] = w * aux_pars_[0] * (y_exp_neg_loc - 1.0);
							deriv_information_aux_par[i] = w * aux_pars_[0] * y_exp_neg_loc;
						}
						else if (base == "hurdle_lognormal") {
							second_deriv_loc_aux_par[i] = w * (-std::log(y_data[i]) + location_par[i]) / s2;
							deriv_information_aux_par[i] = w * (-1.0 / s2);
						}
						else {// EGPD base
							const auto r = EvaluateEGPD(y_data[i], location_par[i]);
							if (r.status == EGPDEvalStatus::kValid) { second_deriv_loc_aux_par[i] = w * r.d_eta_aux_optimizer[ind_aux_par]; deriv_information_aux_par[i] = -w * r.d2_eta_aux_optimizer[ind_aux_par]; }
							else { second_deriv_loc_aux_par[i] = std::numeric_limits<double>::quiet_NaN(); deriv_information_aux_par[i] = std::numeric_limits<double>::quiet_NaN(); }
						}
					}
					(void)egpd;
				}//end hurdle regression
				else if (likelihood_type_ == "zero_inflated_poisson") {
					// Only aux par is rho = logit(p0) (= zero predictor zeta). Cross-derivatives are nonzero only at zero counts.
					CHECK(ind_aux_par == 0);
					const double p0 = aux_pars_original_[0];
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						if (y_data_int[i] != 0) {
							second_deriv_loc_aux_par[i] = 0.;
							deriv_information_aux_par[i] = 0.;
							continue;
						}
						const double mu = std::exp(location_par[i]);
						double wpost;
						ZIPoissonZeroLogMixture(mu, p0, wpost);
						const double v = 1. - wpost;
						// l_{eta,rho} = -v*w*s0 = v*w*mu  (s0 = -mu)
						second_deriv_loc_aux_par[i] = w * (v * wpost * mu);
						// dJ_eta/drho = v*w*(t0 + (w-v)*s0^2) = v*w*(-mu + (w-v)*mu^2)
						deriv_information_aux_par[i] = w * (v * wpost * (-mu + (wpost - v) * mu * mu));
					}
				}//end "zero_inflated_poisson"
				else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
					CHECK(ind_aux_par == 0 || ind_aux_par == 1);
					const bool nb2 = (likelihood_type_ == "zero_inflated_negative_binomial");
					const double p0 = aux_pars_original_[1];
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double wt = has_weights_ ? weights_[i] : 1.0;
						double sdl = 0., dinfo = 0.;
						if (y_data_int[i] > 0) {
							if (ind_aux_par == 0) {// log(shape/dispersion), base NB cross-derivatives (positive count)
								const double mu = std::exp(location_par[i]);
								if (nb2) {
									const double mu_plus_r = mu + aux_pars_[0];
									const double mrr = mu * aux_pars_[0] / (mu_plus_r * mu_plus_r);
									sdl = mrr * (y_data_int[i] - mu);
									dinfo = -mrr * (y_data_int[i] * (aux_pars_[0] - mu) - 2. * aux_pars_[0] * mu) / mu_plus_r;
								}
								else {
									const double r = mu / aux_pars_[0];
									const double dig_diff = GPBoost::digamma(y_data_int[i] + r) - GPBoost::digamma(r);
									const double tri_diff = GPBoost::trigamma(y_data_int[i] + r) - GPBoost::trigamma(r);
									const double tet_diff = GPBoost::tetragamma(y_data_int[i] + r) - GPBoost::tetragamma(r);
									const double C = dig_diff - std::log1p(aux_pars_[0]);
									sdl = -(r * C + r * r * tri_diff + mu / (1.0 + aux_pars_[0]));
									dinfo = 3.0 * r * r * tri_diff + r * r * r * tet_diff + r * C + mu / (1.0 + aux_pars_[0]);
								}
							}// else ind_aux_par == 1 (rho): both zero for positive counts
						}
						else {// zero count
							const double mu = std::exp(location_par[i]);
							ZICountZeroMass z; FillZeroMassZICount(mu, z);
							double wpost, v; ZICountZeroLogMixture(p0, z.b0, wpost, v);
							if (ind_aux_par == 0) {
								sdl = ZICountZero_lEtaShape(z, wpost, v);
								dinfo = ZICountZero_dJetadShape(z, wpost, v);
							}
							else {
								sdl = ZICountZero_lEtaRho(z, wpost, v);
								dinfo = ZICountZero_dJetadRho(z, wpost, v);
							}
						}
						second_deriv_loc_aux_par[i] = wt * sdl;
						deriv_information_aux_par[i] = wt * dinfo;
					}
				}//end zero_inflated_negative_binomial(_1)
				else if (IsZeroInflatedCountRegression()) {
					// Base-count auxiliary cross-derivatives over positive observations; at zero counts use the mixture-weighted forms (per-obs pi from zeta).
					const string_t base = ZICountRegressionBaseType();
					const bool nb2 = (base == "zero_inflated_negative_binomial");
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double wt = has_weights_ ? weights_[i] : 1.0;
						double sdl = 0., dinfo = 0.;
						if (y_data_int[i] > 0) {
							const double mu = std::exp(location_par[i]);
							if (nb2) {
								const double mu_plus_r = mu + aux_pars_[0]; const double mrr = mu * aux_pars_[0] / (mu_plus_r * mu_plus_r);
								sdl = mrr * (y_data_int[i] - mu);
								dinfo = -mrr * (y_data_int[i] * (aux_pars_[0] - mu) - 2. * aux_pars_[0] * mu) / mu_plus_r;
							}
							else {
								const double r = mu / aux_pars_[0];
								const double dd = GPBoost::digamma(y_data_int[i] + r) - GPBoost::digamma(r); const double td = GPBoost::trigamma(y_data_int[i] + r) - GPBoost::trigamma(r); const double tt = GPBoost::tetragamma(y_data_int[i] + r) - GPBoost::tetragamma(r); const double C = dd - std::log1p(aux_pars_[0]);
								sdl = -(r * C + r * r * td + mu / (1.0 + aux_pars_[0]));
								dinfo = 3.0 * r * r * td + r * r * r * tt + r * C + mu / (1.0 + aux_pars_[0]);
							}
						}
						else {
							const double mu = std::exp(location_par[i]);
							ZICountZeroMass z; FillZeroMassCountRegression(mu, z);
							ZICountRegQuant o; ZICountRegressionQuantities(0, location_par[i], location_par[i + num_data_], o);
							const double v = o.v, w = 1. - v;
							sdl = ZICountZero_lEtaShape(z, w, v);
							dinfo = ZICountZero_dJetadShape(z, w, v);
						}
						second_deriv_loc_aux_par[i] = wt * sdl;
						deriv_information_aux_par[i] = wt * dinfo;
					}
				}//end zero-inflated count regression
				else if (likelihood_type_ == "zero_censored_power_transformed_normal") {
					CHECK(ind_aux_par == 0 || ind_aux_par == 1);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double s = aux_pars_[0];
						const double lambda = aux_pars_[1];
						const double mu = location_par[i];
						const double yi = y_data[i];
						double sdl = 0.0; // second_deriv_loc_aux_par
						double dinfo = 0.0; // deriv_information_aux_par
						if (yi <= 0.0) {
							const double a0 = -mu / s;
							const double r = GPBoost::InvMillsRatioNormalPhi(a0);
							if (ind_aux_par == 0) {
								sdl = r * (1.0 + ((a0 + r) * mu) / s) / s;// d^2/dmu d log(sigma) log f at y=0 								
								dinfo = r * ((mu * (1.0 - (a0 + r) * (a0 + 2.0 * r))) / (s * s * s)
									- 2.0 * (a0 + r) / (s * s));// d/d log(sigma) of info(mu) = (1/s^2)*r*(a0 + r)  = r * [ mu * (1 - (a0 + r)*(a0 + 2r)) / s^3 - 2*(a0 + r) / s^2 ]
							}
							else {// ind_aux_par == 1							
								sdl = 0.0;// log(lambda), no dependence at y=0	
								dinfo = 0.0;
							}
						}
						else {
							const double logy = std::log(yi);
							const double u = std::exp((1.0 / lambda) * logy);
							const double z = (u - mu) / s;
							if (ind_aux_par == 0) {
								sdl = -2.0 * z / s;// d^2/dmu d log(sigma) log f at y>0
								dinfo = -2.0 / (s * s);// d/d log(sigma) info
							}
							else {//ind_aux_par == 1
								sdl = -u * logy / (lambda * s * s);//d^2/dmu d log(lambda) log f at y>0 = -u*log(y)/(lambda*s^2)								
								dinfo = 0.0;// info does not depend on lambda for y>0
							}
						}
						second_deriv_loc_aux_par[i] = w * sdl;
						deriv_information_aux_par[i] = w * dinfo;
					}
				}//end "zero_censored_power_transformed_normal"
				else if (IsZeroCensPowNormHetero()) {
					CHECK(ind_aux_par == 0);// lambda is the only auxiliary parameter
					const double lambda = aux_pars_[0];
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						double sdl = 0.0;// second_deriv_loc_aux_par
						if (y_data[i] > 0.0) {
							const double s = std::exp(location_par[i + num_data_]);
							const double logy = std::log(y_data[i]);
							const double u = std::exp(logy / lambda);
							sdl = -u * logy / (lambda * s * s);//d^2/deta d log(lambda) log f at y>0
						}// zeros do not depend on lambda
						second_deriv_loc_aux_par[i] = w * sdl;
						deriv_information_aux_par[i] = 0.0;// the eta-block information does not depend on lambda
					}
				}//end "zero_censored_power_transformed_normal_heteroscedastic"
				else if (likelihood_type_ == "zoctn") {
					// aux_pars_ = { sigma, a, b }, derivatives on the log-scale
					CHECK(ind_aux_par == 0 || ind_aux_par == 1 || ind_aux_par == 2);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double s = aux_pars_[0];     // sigma
						const double a = aux_pars_original_[1]; // a > 0
						const double b = aux_pars_[2];     // b > 0
						const double mu = location_par[i];
						const double yi = y_data[i];
						double sdl = 0.0;   // second_deriv_loc_aux_par
						double dinfo = 0.0; // deriv_information_aux_par
						if (yi <= 0.0) {
							const double a0 = -mu / s;
							const double r = GPBoost::InvMillsRatioNormalPhi(a0);
							if (ind_aux_par == 0) {
								sdl = r * (1.0 + ((a0 + r) * mu) / s) / s;// d^2/dmu d log(sigma) log f at y=0								
								dinfo = r * ((mu * (1.0 - (a0 + r) * (a0 + 2.0 * r))) / (s * s * s) - 2.0 * (a0 + r) / (s * s));// d/d log(sigma) of info(mu) at y=0
							}
							else {// no dependence on a or b at y=0 
								sdl = 0.0;
								dinfo = 0.0;
							}
						}
						else if (yi >= 1.0) {
							const double v = (1.0 - mu) / s;
							const double r2 = GPBoost::InvMillsRatioNormalOneMinusPhi(v);
							if (ind_aux_par == 0) {
								// d^2/dmu d log(sigma) log f at y=1
								// S_mu = (1/s)*r2, v=(1-mu)/s, r2' = r2*(r2 - v)
								// dS_mu/d log(sigma) = -(r2/s) * (1 + v*(r2 - v))
								const double G = r2 * (r2 - v);
								const double dGdv = r2 * (2.0 * G - 1.0) - v * G;
								sdl = -(r2 / s) * (1.0 + v * (r2 - v));
								dinfo = -(1.0 / (s * s)) * (2.0 * G + v * dGdv);// info(mu) = (1/s^2)*G, d/d log(sigma) info = -(1/s^2)*(2*G + v*dGdv)
							}
							else {// no dependence on a or b at y=1 
								sdl = 0.0;
								dinfo = 0.0;
							}
						}
						else { // 0 < y < 1: continuous part
							const double logit_y = GPBoost::logit(yi);
							const double t = (logit_y - a) / b;
							const double x = GPBoost::sigmoid_stable(t);
							const double z = (x - mu) / s;
							if (ind_aux_par == 0) {
								// sigma: d^2/dmu d log(sigma) log f = -2*z/s
								// info(mu) = 1/s^2 -> d/d log(sigma) info = -2/s^2
								sdl = -2.0 * z / s;
								dinfo = -2.0 / (s * s);
							}
							else if (ind_aux_par == 1) {
								// log(a): x depends on a, info does not
								// d^2/dmu d log(a) log f = -x(1-x)/(b*s^2)
								sdl = -x * (1.0 - x) / (b * s * s);
								dinfo = 0.0;
							}
							else { // ind_aux_par == 2, log(b)
								// d^2/dmu d log(b) log f = -t * x(1-x)/s^2
								sdl = -t * x * (1.0 - x) / (s * s);
								dinfo = 0.0;
							}
						}
						second_deriv_loc_aux_par[i] = w * sdl;
						deriv_information_aux_par[i] = w * dinfo;
					}
				}//end "zoctn"
				else if (likelihood_type_ == "zero_one_censored_transformed_beta") {
					CHECK(ind_aux_par == 0 || ind_aux_par == 1);
					const double phi0 = aux_pars_[0];
					const double u0 = aux_pars_[1];
					const double h_log_phi = 1e-4;
					const double h_log_u = 1e-4;
					const double phi_base = std::max(phi0, 1e-300);
					const double u_base = std::max(u0, 1e-300);
					const double phi_minus = phi_base * std::exp(-h_log_phi);
					const double phi_plus = phi_base * std::exp(+h_log_phi);
					const double u_minus = u_base * std::exp(-h_log_u);
					const double u_plus = u_base * std::exp(+h_log_u);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						const double eta = location_par[i];
						if (ind_aux_par == 0) {// derivatives w.r.t. log(phi)
							if (yi > 0.0 && yi < 1.0) {
								// explicit interior formulas
								const double mu = GPBoost::sigmoid_stable_clamped(eta);
								const double s = mu * (1.0 - mu);
								const double sp = s * (1.0 - 2.0 * mu);
								const double a = mu * phi0, b = (1.0 - mu) * phi0;
								const double c = 1.0 + 2.0 * u0;
								const double t = (yi + u0) / c;
								const double G = std::log(t) - std::log1p(-t) - GPBoost::digamma(a) + GPBoost::digamma(b);
								const double psi1a = GPBoost::trigamma(a), psi1b = GPBoost::trigamma(b);
								const double psi2a = GPBoost::tetragamma(a), psi2b = GPBoost::tetragamma(b);
								const double cross_logphi = -phi0 * s * G - phi0 * phi0 * s * (-mu * psi1a + (1.0 - mu) * psi1b);
								const double H_logphi = -phi0 * sp * G - phi0 * phi0 * sp * (-mu * psi1a + (1.0 - mu) * psi1b)
									+ 2.0 * phi0 * phi0 * s * s * (psi1a + psi1b) + phi0 * phi0 * phi0 * s * s * (mu * psi2a + (1.0 - mu) * psi2b);
								second_deriv_loc_aux_par[i] = w * cross_logphi;
								deriv_information_aux_par[i] = w * H_logphi;
							}
							else {
								// boundaries: central differences in log(phi)
								const double neg_score_m = -FirstDerivLogLikZeroOneCensTransfBeta_at(yi, eta, phi_minus, u0);
								const double neg_score_p = -FirstDerivLogLikZeroOneCensTransfBeta_at(yi, eta, phi_plus, u0);
								const double cross_logphi = (neg_score_p - neg_score_m) / (2.0 * h_log_phi);
								const double H_m = SecondDerivNegLogLikZeroOneCensTransfBeta_at(yi, eta, phi_minus, u0);
								const double H_p = SecondDerivNegLogLikZeroOneCensTransfBeta_at(yi, eta, phi_plus, u0);
								const double dH_dlogphi = (H_p - H_m) / (2.0 * h_log_phi);
								second_deriv_loc_aux_par[i] = w * cross_logphi;
								deriv_information_aux_par[i] = w * dH_dlogphi;
							}
						}
						else {
							const double neg_score_m = -FirstDerivLogLikZeroOneCensTransfBeta_at(yi, eta, phi0, u_minus);
							const double neg_score_p = -FirstDerivLogLikZeroOneCensTransfBeta_at(yi, eta, phi0, u_plus);
							const double cross_logu = (neg_score_p - neg_score_m) / (2.0 * h_log_u);
							const double H_m = SecondDerivNegLogLikZeroOneCensTransfBeta_at(yi, eta, phi0, u_minus);
							const double H_p = SecondDerivNegLogLikZeroOneCensTransfBeta_at(yi, eta, phi0, u_plus);
							const double dH_dlogu = (H_p - H_m) / (2.0 * h_log_u);
							second_deriv_loc_aux_par[i] = w * cross_logu;
							deriv_information_aux_par[i] = w * dH_dlogu;
						}
					}
				} // end "zero_one_censored_transformed_beta"
				else if (likelihood_type_ == "zero_one_censored_shifted_gamma") {
					CHECK(ind_aux_par == 0 || ind_aux_par == 1);
					const double k0 = std::max(aux_pars_[0], 1e-12);
					const double xi0 = std::max(aux_pars_[1], 1e-12);
					const double h_log_k = 1e-4;
					const double h_log_xi = 1e-4;
					const double tiny = 1e-300;
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						const double yi = y_data[i];
						const double eta = location_par[i];
						const double mu = std::exp(eta);
						const double inv_mu = 1.0 / std::max(mu, 1e-12);
						double sdl = 0.0; // d^2(-log L) / d eta d (aux)
						double dinfo = 0.0; // d/d(aux) [ I(eta) ]
						if (yi <= 0.0) {
							if (xi0 <= 0.0) {
								sdl = 0.0; dinfo = 0.0;
							}
							else {
								const double a = xi0;
								const double t = std::max(tiny, (k0 * a) * inv_mu);
								const double G = std::max(GPBoost::RegLowerGamma(k0, t), tiny);
								// pdf p(t;k) = exp(-t + (k-1) log t - lgamma(k))
								const double p = std::exp(-t + (k0 - 1.0) * std::log(t) - std::lgamma(k0));
								const double Q = p / G;                 // lower mass ratio							
								const double Qprime = Q * ((k0 - 1.0) / t - 1.0) - Q * Q;// Q' = Q*((k-1)/t - 1) - Q^2
								if (ind_aux_par == 1) { // log(xi)
									const double dt_dlogxi = t; // dt/d log xi
									// s(eta) = t*Q  => cross = d/d log xi (t Q) = t*Q + t^2*Q'
									sdl = dt_dlogxi * (Q + t * Qprime);
									const double xi_minus = xi0 * std::exp(-h_log_xi);
									const double xi_plus = xi0 * std::exp(+h_log_xi);
									const double Hm = SecondDerivNegLogLikZeroOneCensGamma_at(yi, eta, k0, xi_minus);
									const double Hp = SecondDerivNegLogLikZeroOneCensGamma_at(yi, eta, k0, xi_plus);
									dinfo = (Hp - Hm) / (2.0 * h_log_xi);
								}
								else { // log(k) — numeric at mass
									const double k_minus = k0 * std::exp(-h_log_k);
									const double k_plus = k0 * std::exp(+h_log_k);
									const double s_km = FirstDerivLogLikZeroOneCensGamma_at(yi, eta, k_minus, xi0);
									const double s_kp = FirstDerivLogLikZeroOneCensGamma_at(yi, eta, k_plus, xi0);
									sdl = -(s_kp - s_km) / (2.0 * h_log_k);
									const double Hm = SecondDerivNegLogLikZeroOneCensGamma_at(yi, eta, k_minus, xi0);
									const double Hp = SecondDerivNegLogLikZeroOneCensGamma_at(yi, eta, k_plus, xi0);
									dinfo = (Hp - Hm) / (2.0 * h_log_k);
								}
							}
						}
						else if (yi >= 1.0) {
							const double a = 1.0 + xi0;
							const double t = std::max(tiny, (k0 * a) * inv_mu);
							const double G = GPBoost::RegLowerGamma(k0, t);
							const double H = std::max(1.0 - G, tiny);
							const double p = std::exp(-t + (k0 - 1.0) * std::log(t) - std::lgamma(k0));// pdf p(t;k)
							const double Q = p / H;                 // upper mass ratio
							// Q' = Q*((k-1)/t - 1) + Q^2
							const double Qprime = Q * ((k0 - 1.0) / t - 1.0) + Q * Q;
							if (ind_aux_par == 1) { // log(xi)
								const double dt_dlogxi = (xi0 / a) * t; // dt/d log xi
								// s(eta) = - t*Q  => cross = d/d log xi (-t Q) = - (t*Q + t^2*Q')
								sdl = -dt_dlogxi * (Q + t * Qprime);
								// numeric for dinfo wrt log(xi)
								const double xi_minus = xi0 * std::exp(-h_log_xi);
								const double xi_plus = xi0 * std::exp(+h_log_xi);
								const double Hm = SecondDerivNegLogLikZeroOneCensGamma_at(yi, eta, k0, xi_minus);
								const double Hp = SecondDerivNegLogLikZeroOneCensGamma_at(yi, eta, k0, xi_plus);
								dinfo = (Hp - Hm) / (2.0 * h_log_xi);
							}
							else { // log(k) — numeric at mass
								const double k_minus = k0 * std::exp(-h_log_k);
								const double k_plus = k0 * std::exp(+h_log_k);
								const double s_km = FirstDerivLogLikZeroOneCensGamma_at(yi, eta, k_minus, xi0);
								const double s_kp = FirstDerivLogLikZeroOneCensGamma_at(yi, eta, k_plus, xi0);
								sdl = -(s_kp - s_km) / (2.0 * h_log_k);
								const double Hm = SecondDerivNegLogLikZeroOneCensGamma_at(yi, eta, k_minus, xi0);
								const double Hp = SecondDerivNegLogLikZeroOneCensGamma_at(yi, eta, k_plus, xi0);
								dinfo = (Hp - Hm) / (2.0 * h_log_k);
							}
						}
						else {
							// Interior (0<y<1), z = y + xi0, theta = mu/k0
							const double z = yi + xi0;
							if (ind_aux_par == 1) { // log(xi)								
								sdl = -(xi0 * k0) * inv_mu;// cross(eta, log xi) = - xi / theta = - xi0 * k0 / mu								
								dinfo = (xi0 * k0) * inv_mu;// dI/d log xi where I = k0*z/mu: = xi0 * k0 / mu
							}
							else { // log(k)								
								sdl = -k0 * (z * inv_mu - 1.0);	// cross(eta, log k) = - k * (z/mu - 1)							
								dinfo = k0 * z * inv_mu;// dI/d log k where I = k0*z/mu: = k0 * z / mu
							}
						}
						second_deriv_loc_aux_par[i] = w * sdl;
						deriv_information_aux_par[i] = w * dinfo;
					}
				} // end "zero_one_censored_shifted_gamma"
				else if (num_aux_pars_estim_ > 0) {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_);
				}
			}//end approximation_type_ == "laplace"
			else if (approximation_type_ == "fisher_laplace") {
				if (likelihood_type_ == "t") {
					CHECK(ind_aux_par == 0 || ind_aux_par == 1);
					if (ind_aux_par == 0) {
						//gradient for scale parameter is calculated on the log-scale
						const double sigma2 = aux_pars_[0] * aux_pars_[0];
						const double nu_sigma2 = aux_pars_[1] * sigma2;
						const double deriv_FI = -2. * (aux_pars_[1] + 1.) / (aux_pars_[1] + 3.) / sigma2;
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							const double res = y_data[i] - location_par[i];
							const double denom = nu_sigma2 + res * res;
							const double denom_sq = denom * denom;
							second_deriv_loc_aux_par[i] = w * -2. * (aux_pars_[1] + 1.) * aux_pars_[1] * res * sigma2 / denom_sq;
							deriv_information_aux_par[i] = w * deriv_FI;
						}
					}
					else if (ind_aux_par == 1) {
						CHECK(estimate_df_t_);
						//gradient for df parameter is calculated on the log-scale
						const double sigma2 = aux_pars_[0] * aux_pars_[0];
						const double nu_sigma2 = aux_pars_[1] * sigma2;
						const double deriv_FI = aux_pars_[1] * 2. / sigma2 / (aux_pars_[1] + 3.) / (aux_pars_[1] + 3.);
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							const double res = y_data[i] - location_par[i];
							const double res_sq = res * res;
							const double denom = nu_sigma2 + res_sq;
							const double denom_sq = denom * denom;
							second_deriv_loc_aux_par[i] = w * aux_pars_[1] * res * (res_sq - sigma2) / denom_sq;
							deriv_information_aux_par[i] = w * deriv_FI;
						}
					}
				}//end "t"
				else if (likelihood_type_ == "lognormal") {
					CHECK(ind_aux_par == 0);
					const double s2 = aux_pars_[0];
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						second_deriv_loc_aux_par[i] = w * (-std::log(y_data[i]) + location_par[i]) / s2;
						deriv_information_aux_par[i] = w * (-1.0 / s2);
					}
				}//end "lognormal"
				else if (likelihood_type_ == "asymmetric_laplace") {
					//gradient for scale parameter is calculated on the log-scale
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						const double w = has_weights_ ? weights_[i] : 1.0;
						double indicator = (y_data[i] <= location_par[i]) ? 1.0 : 0.0;
						second_deriv_loc_aux_par[i] = w * -(quantile_ - indicator) / aux_pars_[0];
						deriv_information_aux_par[i] = w * -2. * quantile_ * (1. - quantile_) / (aux_pars_[0] * aux_pars_[0]);
					}
				}// end "asymmetric_laplace"
					else if (likelihood_type_ == "zero_inflated_poisson") {
						// Fisher aux cross-derivative for rho = logit(p0): second_deriv_loc_aux (score cross-deriv) is unchanged; the
						// information derivative uses d Fisher / d rho (numerical). Nonzero only at zero counts for the score cross-deriv.
						CHECK(ind_aux_par == 0);
						const double pi = ZICountConstantP0();
						const double h = 1e-5;
						const double rho = std::log(pi) - std::log1p(-pi);
						const double pi_p = GPBoost::sigmoid_stable(rho + h), pi_m = GPBoost::sigmoid_stable(rho - h);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							const double mu = std::exp(location_par[i]);
							ZICountZeroMass z; FillZeroMassZICountKind(mu, z, 0);
							if (y_data_int[i] != 0) { second_deriv_loc_aux_par[i] = 0.; }
							else { double wpost; ZIPoissonZeroLogMixture(mu, pi, wpost); second_deriv_loc_aux_par[i] = w * ((1. - wpost) * wpost * mu); }
							deriv_information_aux_par[i] = w * (ZICountFisherInfoEta(mu, pi_p, z, 0) - ZICountFisherInfoEta(mu, pi_m, z, 0)) / (2. * h);
						}
					}
					else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
						CHECK(ind_aux_par == 0 || ind_aux_par == 1);
						const int kind = ZICountBaseKind();
						const bool nb2 = (kind == 2);
						const double pi = ZICountConstantP0();
						const double base_aux = aux_pars_[0];
						const double h = 1e-5;
						const double rho = std::log(pi) - std::log1p(-pi);
						const double pi_p = GPBoost::sigmoid_stable(rho + h), pi_m = GPBoost::sigmoid_stable(rho - h);
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							const double mu = std::exp(location_par[i]);
							double sdl = 0.;// second_deriv_loc_aux = d^2 l / d eta d aux (same as the Laplace/observed case)
							if (y_data_int[i] > 0) {
								if (ind_aux_par == 0) {// log(shape/dispersion), positive count
									if (nb2) { const double mpr = mu + base_aux; sdl = mu * base_aux / (mpr * mpr) * (y_data_int[i] - mu); }
									else { const double r = mu / base_aux; const double C = GPBoost::digamma(y_data_int[i] + r) - GPBoost::digamma(r) - std::log1p(base_aux); const double td = GPBoost::trigamma(y_data_int[i] + r) - GPBoost::trigamma(r); sdl = -(r * C + r * r * td + mu / (1.0 + base_aux)); }
								}// ind_aux_par == 1 (rho): sdl = 0 for positive counts
							}
							else {// zero count
								ZICountZeroMass z; FillZeroMassZICount(mu, z);
								double wpost, v; ZICountZeroLogMixture(pi, z.b0, wpost, v);
								sdl = (ind_aux_par == 0) ? ZICountZero_lEtaShape(z, wpost, v) : ZICountZero_lEtaRho(z, wpost, v);
							}
							second_deriv_loc_aux_par[i] = w * sdl;
							double dinfo;// deriv_information_aux = d Fisher / d aux (numerical)
							if (ind_aux_par == 0) { const double a_p = base_aux * std::exp(h), a_m = base_aux * std::exp(-h); dinfo = (ZICountFisherInfoEtaExplicit(mu, pi, kind, a_p) - ZICountFisherInfoEtaExplicit(mu, pi, kind, a_m)) / (2. * h); }
							else { dinfo = (ZICountFisherInfoEtaExplicit(mu, pi_p, kind, base_aux) - ZICountFisherInfoEtaExplicit(mu, pi_m, kind, base_aux)) / (2. * h); }
							deriv_information_aux_par[i] = w * dinfo;
						}
					}
					else if (IsZeroInflatedCountRegression()) {
						// Fisher case: only ZINB regression has an aux (shape). ZIP regression has none; ZINB1 regression is combined (Laplace).
						const int kind = ZICountBaseKind();
						const bool nb2 = (kind == 2);
						const double base_aux = aux_pars_[0];
						const double h = 1e-5;
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							const double w = has_weights_ ? weights_[i] : 1.0;
							const double mu = std::exp(location_par[i]);
							const double pi = GPBoost::sigmoid_stable(location_par[i + num_data_]);
							double sdl = 0.;// score cross-deriv d^2 l / d eta d log(shape) (same as the Laplace case)
							if (y_data_int[i] > 0) {
								if (nb2) { const double mpr = mu + base_aux; sdl = mu * base_aux / (mpr * mpr) * (y_data_int[i] - mu); }
								else { const double r = mu / base_aux; const double C = GPBoost::digamma(y_data_int[i] + r) - GPBoost::digamma(r) - std::log1p(base_aux); const double td = GPBoost::trigamma(y_data_int[i] + r) - GPBoost::trigamma(r); sdl = -(r * C + r * r * td + mu / (1.0 + base_aux)); }
							}
							else {
								ZICountZeroMass z; FillZeroMassCountRegression(mu, z);
								ZICountRegQuant o; ZICountRegressionQuantities(0, location_par[i], location_par[i + num_data_], o);
								const double v = o.v, wp = 1. - v; sdl = ZICountZero_lEtaShape(z, wp, v);
							}
							second_deriv_loc_aux_par[i] = w * sdl;
							const double a_p = base_aux * std::exp(h), a_m = base_aux * std::exp(-h);// d Fisher / d log(shape), per-obs pi from zeta
							deriv_information_aux_par[i] = w * (ZICountFisherInfoEtaExplicit(mu, pi, kind, a_p) - ZICountFisherInfoEtaExplicit(mu, pi, kind, a_m)) / (2. * h);
						}
					}
				else if (num_aux_pars_estim_ > 0) {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_);
				}
			}// end approximation_type_ == "fisher_laplace"
			else if (approximation_type_ == "triangular_kernel_curvature" || approximation_type_ == "constant_curvature_manual") {
				if (likelihood_type_ == "asymmetric_laplace") {
					//gradient for scale parameter is calculated on the log-scale
					if (!use_random_effects_indices_of_data_) {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							second_deriv_loc_aux_par[i] = -first_deriv_ll_[i];
							deriv_information_aux_par[i] = -information_ll_[i];
						}
					}
					else {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
						for (data_size_t i = 0; i < num_data_; ++i) {
							second_deriv_loc_aux_par[i] = -first_deriv_ll_data_scale_[i];
							deriv_information_aux_par[i] = -information_ll_data_scale_[i];
						}
					}
				}// end "asymmetric_laplace"
				else if (num_aux_pars_estim_ > 0) {
					NotSupportedForLikelihoodAndApproximation(__func__, approximation_type_);
				}
			}//end approximation_type_ == "triangular_kernel_curvature"
			else {
				Log::REFatal("CalcSecondDerivLogLikFirstDerivInformationAuxPar: approximation_type '%s' is not supported ", approximation_type_.c_str());
			}
		}//end CalcSecondDerivLogLikFirstDerivInformationAuxPar

		/*!
		* \brief Calculate the mean of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable ('RespMeanAdaptiveGHQuadrature')
		*/
		inline double CondMeanLikelihood(const double value) const {
			if (IsGaussianLikelihood() || likelihood_type_ == "t") {
				return value;
			}
			else if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "binomial_probit" || likelihood_type_ == "quasi_bernoulli_probit") {
				return GPBoost::normalCDF(value);
			}
			else if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit" ||
				likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial" || likelihood_type_ == "quasi_bernoulli_logit") {
				return GPBoost::sigmoid_stable(value);
			}
			else if (IsEGPDLikelihood()) {
				const auto& moments = GetEGPDMoments();
				if (!moments.mean_exists || moments.status != EGPDEvalStatus::kValid) Log::REFatal("CondMeanLikelihood: the EGPD response mean is unavailable ");
				return moments.mean_unit_scale * std::exp(value);
			}
			else if (IsHurdleEGPD()) {
				const auto& moments = GetEGPDMoments();
				if (!moments.mean_exists || moments.status != EGPDEvalStatus::kValid) Log::REFatal("The EGPD response mean is unavailable for likelihood='%s' ", likelihood_type_.c_str());
				return (1. - aux_pars_original_[num_aux_pars_ - 1]) * moments.mean_unit_scale * std::exp(value);
			}
			else if (likelihood_type_ == "hurdle_gamma" || likelihood_type_ == "hurdle_lognormal") {
				return (1. - aux_pars_original_[1]) * std::exp(value);
			}
			else if (likelihood_type_ == "zero_inflated_poisson") {
				return (1. - aux_pars_original_[0]) * std::exp(value);
			}
			else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
				return (1. - aux_pars_original_[1]) * std::exp(value);
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				likelihood_type_ == "lognormal") {
				return std::exp(value);
			}
			else {
				NotSupportedForLikelihood(__func__);
				return 0.;
			}
		}

		/*!
		* \brief Calculate the first derivative of the logarithm of the mean of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double FirstDerivLogCondMeanLikelihood(const double value) const {
			if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit" ||
				likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial") {
				return GPBoost::sigmoid_stable(-value);
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" || IsEGPDLikelihood() ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				likelihood_type_ == "lognormal" || IsHurdlePositive() || IsZeroInflatedCount()) {
				return 1.;
			}
			else if (likelihood_type_ == "t" || IsGaussianLikelihood()) {
				return (1. / value);
			}
			else {
				NotSupportedForLikelihood(__func__);
				return 0.;
			}
		}

		/*!
		* \brief Calculate the second derivative of the logarithm of the mean of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double SecondDerivLogCondMeanLikelihood(const double value) const {
			if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit" ||
				likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial") {
				const double p = GPBoost::sigmoid_stable(value);
				return -p * (1.0 - p);
				//alternative version (less numerically stable)
				//double exp_x = std::exp(value);
				//return -exp_x / ((1. + exp_x) * (1. + exp_x));
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" || IsEGPDLikelihood() ||
				likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
				likelihood_type_ == "lognormal" || IsHurdlePositive() || IsZeroInflatedCount()) {
				return 0.;
			}
			else if (likelihood_type_ == "t" || IsGaussianLikelihood()) {
				return (-1. / (value * value));
			}
			else {
				NotSupportedForLikelihood(__func__);
				return 0.;
			}
		}

		/*!
		* \brief Calculate the variance of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double CondVarLikelihood(const double value) const {
			if (likelihood_type_ == "beta") {
				double exp_min_val = std::exp(-value);
				return exp_min_val / ((1. + exp_min_val) * (1. + exp_min_val)) / (1. + aux_pars_[0]);
			}
			else if (IsEGPDLikelihood()) {
				const auto& moments = GetEGPDMoments();
				if (!moments.variance_exists || moments.status != EGPDEvalStatus::kValid) Log::REFatal("CondVarLikelihood: the EGPD response variance is unavailable ");
				return moments.variance_unit_scale * std::exp(2. * value);
			}
			else {
				NotSupportedForLikelihood(__func__);
				return 0.;
			}
		}

		/*!
		* \brief Calculate the first derivative of the logarithm of the variance of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double FirstDerivLogCondVarLikelihood(const double value) const {
			if (likelihood_type_ == "beta") {
				return (-1. + 2. / (1. + std::exp(value)));
			}
			else if (IsEGPDLikelihood()) {
				return 2.;
			}
			else {
				NotSupportedForLikelihood(__func__);
				return 0.;
			}
		}

		/*!
		* \brief Calculate the second derivative of the logarithm of the variance of the likelihood conditional on the (predicted) latent variable
		*           Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double SecondDerivLogCondVarLikelihood(const double value) const {
			if (likelihood_type_ == "beta") {
				double exp_x = std::exp(value);
				return -2 * exp_x / ((1. + exp_x) * (1. + exp_x));
			}
			else if (IsEGPDLikelihood()) {
				return 0.;
			}
			else {
				NotSupportedForLikelihood(__func__);
				return 0.;
			}
		}

		// Gauss-Hermite quadrature for computing E[f(Z)] with Z~N(0,1): 
		//		E[f(Z)] = (1/sqrt(pi)) * sum_j w_j * f(sqrt(2) * x_j), F(Z) = (max(0, m + s * Z))^lambda
		//		This is used for the prediction of the response variable
		inline double TruncPowerNormalMomentGH(const double m, const double s, const double lambda) const {
			const double sqrt2 = std::sqrt(2.0);
			double sum = 0.0;
			for (int j = 0; j < order_GH_; ++j) {
				const double z = sqrt2 * GH_nodes_[j];
				const double x = m + s * z;
				if (x > 0.0) {
					sum += GH_weights_[j] * std::exp(lambda * std::log(x)); // x^lambda
				}
			}
			return sum / std::sqrt(M_PI);
		}

		// Gauss-Hermite quadrature for likelihood == "zoctn"
		//		This is used for the prediction of the response variable
		inline double ZeroOneCensTransNormalMomentGH(const double m, const double s, bool second_moment) const {
			const double sqrt2 = std::sqrt(2.0);
			const double a = aux_pars_original_[1]; // a>0
			const double b = aux_pars_[2];          // b>0
			const double eps = 1e-12;
			double sum = 0.0;
			for (int j = 0; j < order_GH_; ++j) {
				const double z0 = sqrt2 * GH_nodes_[j];
				const double Z = m + s * z0; // full latent Z (includes both GP variance and sigma^2)
				double y;
				if (Z <= 0.0) {
					y = 0.0;
				}
				else if (Z >= 1.0) {
					y = 1.0;
				}
				else {
					double x = Z;
					if (x < eps) x = eps;
					if (x > 1.0 - eps) x = 1.0 - eps;
					const double logit_x = std::log(x) - std::log1p(-x);
					const double inner = a + b * logit_x;
					y = GPBoost::sigmoid_stable(inner);
				}
				if (second_moment) {
					y *= y;
				}
				sum += GH_weights_[j] * y;
			}
			return sum / std::sqrt(M_PI);
		}

		inline double XB_FirstMoment_(double mu, double phi, double u) const {
			double a = mu * phi;
			double b = (1.0 - mu) * phi;
			// guard against edge a,b
			const double tiny = 1e-12;
			if (!(a > 0.0) || !std::isfinite(a)) a = tiny;
			if (!(b > 0.0) || !std::isfinite(b)) b = tiny;
			const double c = 1.0 + 2.0 * u;
			const double eps = 1e-15;
			double t0 = u / c;
			double t1 = (1.0 + u) / c;
			if (t0 <= eps) t0 = eps;
			if (t0 >= 1.0 - eps) t0 = 1.0 - eps;
			if (t1 <= eps) t1 = eps;
			if (t1 >= 1.0 - eps) t1 = 1.0 - eps;
			// regularized CDFs
			const double F0 = GPBoost::reg_incbeta(a, b, t0);
			const double F1 = GPBoost::reg_incbeta(a, b, t1);
			const double Pmid = F1 - F0;
			const double P1 = 1.0 - F1;
			// E[Z * 1(t0<Z<t1)] = [a/(a+b)] * ( I_{t1}(a+1,b) - I_{t0}(a+1,b) )
			const double Iz1_t1 = GPBoost::reg_incbeta(a + 1.0, b, t1);
			const double Iz1_t0 = GPBoost::reg_incbeta(a + 1.0, b, t0);
			const double Ez1 = (a / (a + b)) * (Iz1_t1 - Iz1_t0);
			// E[Y] = (1+2u) * E[Z 1(mid)] - u * P(mid) + P(Y=1)
			double m1 = (1.0 + 2.0 * u) * Ez1 - u * Pmid + P1;
			if (!(m1 >= 0.0) || !std::isfinite(m1)) m1 = 0.0;
			if (m1 > 1.0) m1 = 1.0;
			return m1;
		}

		inline double XB_SecondMoment_(double mu, double phi, double u) const {
			double a = mu * phi;
			double b = (1.0 - mu) * phi;
			const double tiny = 1e-12;
			if (!(a > 0.0) || !std::isfinite(a)) a = tiny;
			if (!(b > 0.0) || !std::isfinite(b)) b = tiny;
			const double c = 1.0 + 2.0 * u;
			const double eps = 1e-15;
			double t0 = u / c;
			double t1 = (1.0 + u) / c;
			if (t0 <= eps) t0 = eps;
			if (t0 >= 1.0 - eps) t0 = 1.0 - eps;
			if (t1 <= eps) t1 = eps;
			if (t1 >= 1.0 - eps) t1 = 1.0 - eps;
			const double F0 = GPBoost::reg_incbeta(a, b, t0);
			const double F1 = GPBoost::reg_incbeta(a, b, t1);
			const double Pmid = F1 - F0;
			const double P1 = 1.0 - F1;
			// E[Z^2 * 1(t0<Z<t1)] = [a(a+1)/((a+b)(a+b+1))] * ( I_{t1}(a+2,b) - I_{t0}(a+2,b) )
			const double Iz2_t1 = GPBoost::reg_incbeta(a + 2.0, b, t1);
			const double Iz2_t0 = GPBoost::reg_incbeta(a + 2.0, b, t0);
			const double coeff2 = (a * (a + 1.0)) / ((a + b) * (a + b + 1.0));
			const double Ez2 = coeff2 * (Iz2_t1 - Iz2_t0);
			// need E[Z * 1(mid)] for the cross term
			const double Iz1_t1 = GPBoost::reg_incbeta(a + 1.0, b, t1);
			const double Iz1_t0 = GPBoost::reg_incbeta(a + 1.0, b, t0);
			const double Ez1 = (a / (a + b)) * (Iz1_t1 - Iz1_t0);
			const double onep2u = 1.0 + 2.0 * u;
			double term_mid = (onep2u * onep2u) * Ez2 - 2.0 * u * onep2u * Ez1 + u * u * Pmid;
			double m2 = term_mid + P1;
			if (!(m2 >= 0.0) || !std::isfinite(m2)) m2 = 0.0;
			if (m2 > 1.0) m2 = 1.0;
			return m2;
		}

		inline void ZOCG_MomentsGivenEta_(const double eta,
			const double k,
			const double xi,
			double& Ey,
			double& Ey2,
			const bool need_second) const {
			const double mu = std::exp(eta);
			const double th = mu / k;
			const double t0 = xi / th;
			const double t1 = (1.0 + xi) / th;
			const double Gk_t0 = GPBoost::RegLowerGamma(k, t0);
			const double Gk_t1 = GPBoost::RegLowerGamma(k, t1);
			const double Pk_int = Gk_t1 - Gk_t0;
			const double p1 = 1.0 - Gk_t1;
			const double Pk1_t0 = GPBoost::RegLowerGamma(k + 1.0, t0);
			const double Pk1_t1 = GPBoost::RegLowerGamma(k + 1.0, t1);
			const double M1 = (k * th) * (Pk1_t1 - Pk1_t0);
			Ey = p1 + (M1 - xi * Pk_int);
			if (need_second) {
				const double Pk2_t0 = GPBoost::RegLowerGamma(k + 2.0, t0);
				const double Pk2_t1 = GPBoost::RegLowerGamma(k + 2.0, t1);
				const double M2 = (k * (k + 1.0) * th * th) * (Pk2_t1 - Pk2_t0);
				Ey2 = p1 + (M2 - 2.0 * xi * M1 + xi * xi * Pk_int);
			}
			if (!(Ey >= 0.0)) Ey = 0.0;
			if (Ey < 0.0) Ey = 0.0;
			if (Ey > 1.0) Ey = 1.0;
			if (need_second) {
				if (!(Ey2 >= 0.0)) Ey2 = (Ey * Ey);
				if (Ey2 < 0.0) Ey2 = 0.0;
				if (Ey2 > 1.0) Ey2 = 1.0;
			}
		}

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

#endif   // GPB_LIKELIHOODS_
