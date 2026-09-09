/*!
* This file is part of GPBoost a C++ library for combining
*   boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2026 Fabio Sigrist, Tim Gyger, and Pascal Kuendig. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*
* Definitions of the member functions of the 'Likelihood' class that map the latent location parameter
* to the response scale: the predictive mean and variance of the response, the conditional mean and
* variance of the likelihood and their derivatives, and the adaptive Gauss-Hermite quadrature used for
* the predictive moments and for the 'test_neg_log_likelihood' evaluation metric.
* All functions defined here are also declared and documented in 'likelihoods.h'.
*
* NOTE: this file is included at the end of 'likelihoods.h' and cannot be compiled on its own.
*/
#ifndef GPB_LIKELIHOODS_PREDICT_H_
#define GPB_LIKELIHOODS_PREDICT_H_

namespace GPBoost {

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::PredictResponse(vec_t& pred_mean,
		vec_t& pred_var,
		const vec_t& pred_var_mean,
		const vec_t& pred_var_var,
		bool predict_var,
		const vec_t& pred_third_block_mean) {
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
		else if (IsGammaVaryingShape()) {
			// As for "gamma" / "hurdle_gamma", but with the per-observation shape k_i = exp(zeta_i) taken from the last
			// location parameter block (deterministic given the fixed effects, so it carries no posterior uncertainty).
			// Scale-family form E(Y|eta,y>0) = exp(eta), V_b = exp(2*eta) / k_i:
			//   E(Y*) = q*exp(m + v/2),  Var(Y*) = q*(1/k_i + p0)*exp(2m + 2v) + q^2*exp(2m + v)*(exp(v) - 1)
			CHECK(need_pred_latent_var_for_response_mean_);
			const bool regression_zero = (likelihood_type_ == "hurdle_regression_gamma_varying_shape");
			const vec_t& log_shape = regression_zero ? pred_third_block_mean : pred_var_mean;
			CHECK(log_shape.size() == pred_mean.size());
			const double q_const = (likelihood_type_ == "hurdle_gamma_varying_shape") ? (1. - aux_pars_original_[0]) : 1.;
#pragma omp parallel for schedule(static)
			for (int i = 0; i < (int)pred_mean.size(); ++i) {
				const double m = pred_mean[i];
				const double v = std::max(pred_var[i], 0.0);
				const double q = regression_zero ? GPBoost::sigmoid_stable(-pred_var_mean[i]) : q_const;// 1 - pi_i
				const double p0 = 1. - q;
				pred_mean[i] = q * std::exp(m + 0.5 * v);
				if (predict_var) {
					pred_var[i] = q * (std::exp(-log_shape[i]) + p0) * std::exp(2. * m + 2. * v) + q * q * std::exp(2. * m + v) * std::expm1(v);
				}
			}
		}//end gamma varying shape variants
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
		else if (IsZeroCensShiftedGamma()) {
			// The conditional moments of Y = max(Z - xi, 0) given eta are available in closed form; the predictive moments
			// are obtained by integrating them over the (approximately Gaussian) predictive distribution of eta. The shape
			// of the varying-shape variant is deterministic given the fixed effects, so it carries no posterior uncertainty
			CHECK(need_pred_latent_var_for_response_mean_);
			const bool varying_shape = IsZeroCensShiftedGammaVaryingShape();
			const double xi = varying_shape ? aux_pars_[0] : aux_pars_[1];
			const double k_const = varying_shape ? 0. : aux_pars_[0];
			if (varying_shape) CHECK(pred_var_mean.size() == pred_mean.size());
			const size_t K = GH_nodes_.size();
			const double inv_sqrt_pi = 1.0 / std::sqrt(3.14159265358979323846);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < (int)pred_mean.size(); ++i) {
				const double m = pred_mean[i];
				const double v = std::max(pred_var[i], 0.0);
				const double sc = std::sqrt(2.0 * v);
				const double k = varying_shape ? ZeroCensGammaVarShapeShape(pred_var_mean[i]) : k_const;
				double Ey = 0.0, Ey2 = 0.0;
				for (size_t j = 0; j < K; ++j) {
					const double wj = GH_weights_[j] * inv_sqrt_pi;
					double m1 = 0.0, m2 = 0.0;
					ZCG_MomentsGivenEta_(m + sc * GH_nodes_[j], k, xi, m1, m2, predict_var);
					Ey += wj * m1;
					if (predict_var) Ey2 += wj * m2;
				}
				pred_mean[i] = std::max(0.0, Ey);
				if (predict_var) pred_var[i] = std::max(0.0, Ey2 - Ey * Ey);
			}
		}//end zero-censored shifted gamma variants
		else if (likelihood_type_ == "asymmetric_laplace") {
			if (predict_var) {
				Log::REFatal("PredictResponse: Predictive variances for likelihood of type '%s' is not supported ", likelihood_type_.c_str());
			}
		}
		else {
			NotSupportedForLikelihood(__func__);
		}
	}//end PredictResponse

	template <typename T_mat, typename T_chol>
	double Likelihood<T_mat, T_chol>::RespMeanAdaptiveGHQuadrature(const double latent_mean,
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

	template <typename T_mat, typename T_chol>
	double Likelihood<T_mat, T_chol>::ExpectedValueCondRespVarAdaptiveGHQuadrature(const double latent_mean,
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::TestNegLogLikelihoodAdaptiveGHQuadrature(const label_t* y_test,
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

	template <typename T_mat, typename T_chol>
	double Likelihood<T_mat, T_chol>::TransformToResponseScale(const double value) const {
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
		else if (likelihood_type_ == "hurdle_gamma_varying_shape") {
			return (1. - aux_pars_original_[0]) * std::exp(value);// p0 is the only auxiliary parameter here
		}
		else if (likelihood_type_ == "zero_inflated_poisson") {
			return (1. - aux_pars_original_[0]) * std::exp(value);
		}
		else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
			return (1. - aux_pars_original_[1]) * std::exp(value);
		}
		else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "gamma_varying_shape" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" ||
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
		else if (IsZeroCensShiftedGamma()) {
			// As for the other censored likelihoods above, this is the censoring transformation Y = max(Z - xi, 0)
			// applied to the location parameter itself, not the conditional mean E(Y | eta) (which is calculated by
			// 'PredictResponse' and which the single-argument interface cannot provide for the varying-shape variant)
			return std::max(std::exp(value) - (IsZeroCensShiftedGammaVaryingShape() ? aux_pars_[0] : aux_pars_[1]), 0.);
		}//end zero-censored shifted gamma variants
		else {
			NotSupportedForLikelihood(__func__);
			return 0.;
		}
	}//end TransformToResponseScale

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::CondMeanLikelihood(const double value) const {
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
		else if (likelihood_type_ == "hurdle_gamma_varying_shape") {
			return (1. - aux_pars_original_[0]) * std::exp(value);// p0 is the only auxiliary parameter here
		}
		else if (likelihood_type_ == "zero_inflated_poisson") {
			return (1. - aux_pars_original_[0]) * std::exp(value);
		}
		else if (likelihood_type_ == "zero_inflated_negative_binomial" || likelihood_type_ == "zero_inflated_negative_binomial_1") {
			return (1. - aux_pars_original_[1]) * std::exp(value);
		}
		else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "gamma_varying_shape" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" ||
			likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
			likelihood_type_ == "lognormal") {
			return std::exp(value);
		}
		else {
			NotSupportedForLikelihood(__func__);
			return 0.;
		}
	}

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::FirstDerivLogCondMeanLikelihood(const double value) const {
		if (likelihood_type_ == "bernoulli_logit" || likelihood_type_ == "binomial_logit" ||
			likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial") {
			return GPBoost::sigmoid_stable(-value);
		}
		else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" || IsEGPDLikelihood() ||
			likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" ||
			likelihood_type_ == "lognormal" || IsHurdlePositive() || IsZeroInflatedCount() || IsGammaVaryingShape()) {
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::SecondDerivLogCondMeanLikelihood(const double value) const {
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::CondVarLikelihood(const double value) const {
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::FirstDerivLogCondVarLikelihood(const double value) const {
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::SecondDerivLogCondVarLikelihood(const double value) const {
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::TruncPowerNormalMomentGH(const double m, const double s, const double lambda) const {
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::ZeroOneCensTransNormalMomentGH(const double m, const double s, bool second_moment) const {
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::XB_FirstMoment_(double mu, double phi, double u) const {
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

	template <typename T_mat, typename T_chol>
	inline double Likelihood<T_mat, T_chol>::XB_SecondMoment_(double mu, double phi, double u) const {
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

	template <typename T_mat, typename T_chol>
	inline void Likelihood<T_mat, T_chol>::ZOCG_MomentsGivenEta_(const double eta,
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

	template <typename T_mat, typename T_chol>
	inline void Likelihood<T_mat, T_chol>::ZCG_MomentsGivenEta_(const double eta,
		const double k,
		const double xi,
		double& Ey,
		double& Ey2,
		const bool need_second) const {
		// Y = max(Z - xi, 0), Z ~ Gamma(k, theta), theta = mu / k, mu = exp(eta). With S_j = 1 - G(k + j, t0), t0 = xi / theta:
		//   E(Y)   = k * theta * S_1 - xi * S_0
		//   E(Y^2) = k * (k + 1) * theta^2 * S_2 - 2 * xi * k * theta * S_1 + xi^2 * S_0
		const double mu = std::exp(eta);
		const double th = mu / k;
		const double t0 = xi / th;
		const double S0 = 1.0 - GPBoost::RegLowerGamma(k, t0);
		const double S1 = 1.0 - GPBoost::RegLowerGamma(k + 1.0, t0);
		const double M1 = (k * th) * S1;
		Ey = M1 - xi * S0;
		if (!(Ey >= 0.0)) Ey = 0.0;
		if (need_second) {
			const double S2 = 1.0 - GPBoost::RegLowerGamma(k + 2.0, t0);
			const double M2 = (k * (k + 1.0) * th * th) * S2;
			Ey2 = M2 - 2.0 * xi * M1 + xi * xi * S0;
			if (!(Ey2 >= Ey * Ey)) Ey2 = Ey * Ey;
		}
	}
}  // namespace GPBoost

#endif   // GPB_LIKELIHOODS_PREDICT_H_
