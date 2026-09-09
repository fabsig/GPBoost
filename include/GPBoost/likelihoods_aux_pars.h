/*!
* This file is part of GPBoost a C++ library for combining
*   boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2026 Fabio Sigrist, Tim Gyger, and Pascal Kuendig. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*
* Definitions of the member functions of the 'Likelihood' class that handle the auxiliary (likelihood)
* parameters and the initial values: the transformation between the internal and the original parameter
* scale, the data-driven initial values for the intercept and for the auxiliary parameters, the constants
* that cap a too large learning rate, and the gradients of the negative Laplace-approximated marginal
* log-likelihood with respect to the auxiliary parameters.
* All functions defined here are also declared and documented in 'likelihoods.h'.
*
* NOTE: this file is included at the end of 'likelihoods.h' and cannot be compiled on its own.
*/
#ifndef GPB_LIKELIHOODS_AUX_PARS_H_
#define GPB_LIKELIHOODS_AUX_PARS_H_

namespace GPBoost {

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::TransformAuxPars(const double* aux_pars_orig,
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
		else if (likelihood_type_ == "zero_inflated_poisson" || likelihood_type_ == "hurdle_gamma_varying_shape") {
			// p0 is the only auxiliary parameter of these likelihoods (the shape of 'hurdle_gamma_varying_shape' is a location parameter block)
			if (!(aux_pars_orig[0] > 0. && aux_pars_orig[0] < 1.)) {
				Log::REFatal("The '%s' parameter (= %g) needs to be larger than 0 and smaller than 1 ", names_aux_pars_[0].c_str(), aux_pars_orig[0]);
			}
			aux_pars_trans[0] = aux_pars_orig[0] / (1. - aux_pars_orig[0]);
		}//end "zero_inflated_poisson" / "hurdle_gamma_varying_shape"
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

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::BackTransformAuxPars(const double* aux_pars_trans,
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
		else if (likelihood_type_ == "zero_inflated_poisson" || likelihood_type_ == "hurdle_gamma_varying_shape") {
			if (!(aux_pars_trans[0] > 0.)) {
				Log::REFatal("BackTransformAuxPars: the transformed '%s' parameter (= %g) needs to be larger than 0 ", names_aux_pars_[0].c_str(), aux_pars_trans[0]);
			}
			aux_pars_orig[0] = aux_pars_trans[0] / (1. + aux_pars_trans[0]);
		}//end "zero_inflated_poisson" / "hurdle_gamma_varying_shape"
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

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::ZeroCensPowNormHeteroAnchors(const double* y_data,
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

	template <typename T_mat, typename T_chol>
	double Likelihood<T_mat, T_chol>::FindInitialIntercept(const double* y_data,
		const data_size_t num_data,
		double rand_eff_var,
		const double* fixed_effects,
		int ind_set_re,
		const double* weights) const {
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
		else if (IsGammaVaryingShape()) {
			// Block 0 (eta = log(mu)): as for the constant-shape variants, the log of the mean of the positive
			// observations, which are already divided by their block-0 fixed effects offset below.
			// Last block (log(shape)): the approximate marginal gamma shape MLE of those offset-corrected observations
			// (as in 'FindInitialAuxPars' for "gamma"). Block 1 of a hurdle regression (structural-zero logit): the logit
			// of the observed zero fraction. The latter two are anchors on the scale of the TOTAL location parameter of
			// their block, so the pooled fixed effects offset of that block is subtracted from them
			const int ind_shape = num_sets_fixed_effects_ - 1;
			CHECK(ind_set_re >= 0 && ind_set_re < num_sets_fixed_effects_);
			const double eps_p = 1e-6;// clipping for probabilities
			double sw = 0., w_pos = 0., avg = 0., avg_log = 0., off = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, w_pos, avg, avg_log, off)
			for (data_size_t i = 0; i < num_data; ++i) {
				const double w = has_weights_ ? weights_ptr[i] : 1.0;
				sw += w;
				if (fixed_effects != nullptr) off += w * fixed_effects[i + (data_size_t)ind_set_re * num_data];
				if (y_data[i] > 0.) {
					const double y_scaled = fixed_effects == nullptr ? y_data[i] : y_data[i] / std::exp(fixed_effects[i]);
					w_pos += w;
					avg += w * y_scaled;
					avg_log += w * std::log(y_scaled);
				}
			}
			off /= sw;
			if (ind_set_re == 0) {
				// The block-0 offset has already been divided out of every observation, so it must not be subtracted again
				avg = std::max(w_pos > 0. ? avg / w_pos : 1., 1e-12);
				init_intercept = std::log(avg) - 0.5 * rand_eff_var;
			}
			else if (ind_set_re == ind_shape) {
				// ln(k) - digamma(k) approx = (1 + 1 / (6k + 1)) / (2k) with s = log(mean(y)) - mean(log(y)), see 'FindInitialAuxPars'
				double shape = 1.;
				if (w_pos > 0.) {
					const double s = std::max(std::log(std::max(avg / w_pos, 1e-12)) - avg_log / w_pos, 1e-8);
					shape = (3. - s + std::sqrt((s - 3.) * (s - 3.) + 24. * s)) / (12. * s);
				}
				if (!(shape > 0.) || !std::isfinite(shape)) shape = 1.;
				init_intercept = std::log(shape) - off;
			}
			else {// structural-zero logit of "hurdle_regression_gamma_varying_shape"
				const double p0 = std::min(std::max((sw - w_pos) / sw, eps_p), 1. - eps_p);
				init_intercept = GPBoost::logit(p0) - off;
			}
		}//end gamma varying shape variants
		else if (IsZeroCensShiftedGamma()) {
			// Block 0 (eta = log(mu)): the log of the mean of z = y + xi over the positive observations, which are already
			// divided by their block-0 fixed effects offset below; if there is no positive observation at all, mu ~ xi + 0.5.
			// Last block (log(shape)) of the varying-shape variant: the approximate marginal gamma shape MLE of those
			// offset-corrected z (as in 'FindInitialAuxPars' for "gamma"). The latter is an anchor on the scale of the total
			// location parameter of its block, so the pooled fixed effects offset of that block is subtracted from it
			const double xi = IsZeroCensShiftedGammaVaryingShape() ? aux_pars_[0] : aux_pars_[1];
			CHECK(ind_set_re >= 0 && ind_set_re < num_sets_fixed_effects_);
			double sw = 0., w_pos = 0., avg = 0., avg_log = 0., off = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, w_pos, avg, avg_log, off)
			for (data_size_t i = 0; i < num_data; ++i) {
				const double w = has_weights_ ? weights_ptr[i] : 1.0;
				sw += w;
				if (fixed_effects != nullptr) off += w * fixed_effects[i + (data_size_t)ind_set_re * num_data];
				if (y_data[i] > 0.) {
					const double z = fixed_effects == nullptr ? (y_data[i] + xi) : (y_data[i] + xi) / std::exp(fixed_effects[i]);
					w_pos += w;
					avg += w * z;
					avg_log += w * std::log(z);
				}
			}
			off /= sw;
			if (ind_set_re == 0) {
				// The block-0 offset has already been divided out of every observation, so it must not be subtracted again
				init_intercept = std::log(std::max(w_pos > 0. ? avg / w_pos : (xi + 0.5), 1e-12)) - 0.5 * rand_eff_var;
			}
			else {// log(shape) of "zero_censored_shifted_gamma_varying_shape"
				// ln(k) - digamma(k) approx = (1 + 1 / (6k + 1)) / (2k) with s = log(mean(z)) - mean(log(z)), see 'FindInitialAuxPars'
				double shape = 1.;
				if (w_pos > 0.) {
					const double s = std::max(std::log(std::max(avg / w_pos, 1e-12)) - avg_log / w_pos, 1e-8);
					shape = (3. - s + std::sqrt((s - 3.) * (s - 3.) + 24. * s)) / (12. * s);
				}
				if (!(shape > 0.) || !std::isfinite(shape)) shape = 1.;
				init_intercept = std::log(shape) - off;
			}
		}//end zero-censored shifted gamma variants
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

	template <typename T_mat, typename T_chol>
	bool Likelihood<T_mat, T_chol>::ShouldHaveIntercept(const double* y_data,
		const data_size_t num_data,
		double rand_eff_var,
		const double* fixed_effects,
		const double* weights) const {
		bool ret_val = false;
		if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || likelihood_type_ == "tweedie" || likelihood_type_ == "tweedie_fixed_p" || IsEGPDLikelihood() ||
			likelihood_type_ == "negative_binomial" || likelihood_type_ == "negative_binomial_1" || IsZeroInflatedCount() ||
			IsGaussianHeteroscedastic() || likelihood_type_ == "lognormal" || IsHurdlePositive() ||
			IsZeroCensPowNorm() || IsGammaVaryingShape() || IsZeroCensShiftedGamma() ||
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

	template <typename T_mat, typename T_chol>
	const double* Likelihood<T_mat, T_chol>::FindInitialAuxPars(const double* y_data,
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
		else if (likelihood_type_ == "hurdle_gamma_varying_shape") {
			// The shape is a location parameter block here, so only the structural-zero p0 is initialized (weighted zero fraction).
			sw = 0.;
			double avg_zero = 0.;
#pragma omp parallel for schedule(static) reduction(+:sw, avg_zero)
			for (data_size_t i = 0; i < num_data; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.0;
				if (y_data[i] <= 0.) avg_zero += w;
				sw += w;
			}
			avg_zero = std::min(std::max(avg_zero / sw, 1e-3), 1. - 1e-3);
			aux_pars_[0] = avg_zero / (1. - avg_zero);
		}//end "hurdle_gamma_varying_shape"
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
		else if (IsZeroCensShiftedGamma()) {
			// (1) the observed zero fraction estimates p0 = P(Z <= xi). (2) For a provisional shape k = 1 (an exponential Z),
			// E(Y) = E(max(Z - xi, 0)) = mu * exp(-xi / mu) = mu * (1 - p0) by memorylessness, which gives the mean mu of Z
			// from the sample mean of y, and then xi = -mu * log(1 - p0). (3) The shape is refined with the log-moment
			// estimator of the gamma shape applied to z = y + xi over the positive observations (constant-shape variant only)
			double W = 0.0, W0 = 0.0, sum_y = 0.0;
#pragma omp parallel for schedule(static) reduction(+:W,W0,sum_y)
			for (data_size_t i = 0; i < num_data; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.0;
				const double yi = y_data[i];
				W += w;
				if (yi <= 0.0) { W0 += w; }
				else { sum_y += w * (fixed_effects == nullptr ? yi : yi / std::exp(fixed_effects[i])); }
			}
			const double p0 = (W > 0.0) ? std::min(std::max(W0 / W, 1e-12), 1.0 - 1e-6) : 0.1;
			const double y_bar = (W > 0.0) ? std::max(sum_y / W, 1e-8) : 1.0;
			const double mu_init = y_bar / (1.0 - p0);
			const double xi_init = std::min(std::max(-mu_init * std::log1p(-p0), 1e-6), 1e6);
			if (IsZeroCensShiftedGammaVaryingShape()) {
				aux_pars_[0] = xi_init;// the shape is a location parameter block here
			}
			else {
				double sum_z = 0.0, sum_logz = 0.0, cnt_z = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sum_z,sum_logz,cnt_z)
				for (data_size_t i = 0; i < num_data; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					if (y_data[i] > 0.0) {
						const double z = (y_data[i] + xi_init) / (fixed_effects == nullptr ? 1.0 : std::exp(fixed_effects[i]));
						sum_z += w * z;
						sum_logz += w * std::log(z);
						cnt_z += w;
					}
				}
				double k_init = 1.0;
				if (cnt_z > 0.0) {
					const double sd = std::max(std::log(std::max(sum_z / cnt_z, 1e-12)) - sum_logz / cnt_z, 1e-12);
					k_init = std::min(std::max((3.0 - sd + std::sqrt((sd - 3.0) * (sd - 3.0) + 24.0 * sd)) / (12.0 * sd), 0.2), 50.0);
				}
				aux_pars_[0] = k_init;
				aux_pars_[1] = xi_init;
			}
		}//end zero-censored shifted gamma variants
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
			likelihood_type_ != "poisson" && !IsGaussianHeteroscedastic() && !IsEGPDLikelihood() && !IsGammaVaryingShape() &&
			likelihood_type_ != "quasi_bernoulli_probit" && likelihood_type_ != "quasi_bernoulli_logit") {
			NotSupportedForLikelihood(__func__);
		}
		aux_pars_original_ = aux_pars_;
		BackTransformAuxPars(aux_pars_.data(), aux_pars_original_.data());
		return(aux_pars_.data());
	}//end FindInitialAuxPars

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::FindConstantsCapTooLargeLearningRateCoef(const double* y_data,
		const data_size_t num_data,
		const double* fixed_effects,
		double& C_mu,
		double& C_sigma2,
		const double* weights) const {
		const double* weights_ptr = (weights != nullptr) ? weights : weights_;
		if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit" ||
			likelihood_type_ == "binomial_probit" || likelihood_type_ == "binomial_logit" ||
			likelihood_type_ == "beta" || likelihood_type_ == "beta_binomial" || likelihood_type_ == "zoctn" ||
			likelihood_type_ == "zero_one_censored_transformed_beta" || likelihood_type_ == "zero_one_censored_shifted_gamma" || 
			likelihood_type_ == "quasi_bernoulli_probit" || likelihood_type_ == "quasi_bernoulli_logit") {
			C_mu = 1.;
			C_sigma2 = 1.;
		}
		else if (IsGammaVaryingShape() || IsZeroCensShiftedGammaVaryingShape()) {
			C_mu = 1e99;//not implemented (the caps assume a single location parameter block)
			C_sigma2 = 1e99;
		}
		else if (IsHurdlePositive() || likelihood_type_ == "zero_censored_shifted_gamma") {
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

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::SetAuxPars(const double* aux_pars) {
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
			IsZeroCensShiftedGamma() || likelihood_type_ == "asymmetric_laplace") {
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

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcGradNegLogLikAuxPars(const double* y_data,
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
		else if (likelihood_type_ == "hurdle_gamma_varying_shape") {
			// p0 is the only auxiliary parameter and the structural zero decouples from both location parameter blocks:
			// gradient of the negative log-likelihood wrt rho = logit(p0), as for "hurdle_gamma"
			const double p0 = aux_pars_original_[0];
			const double q = 1. - p0;
			double Wpos = 0., Wzero = 0.;
#pragma omp parallel for schedule(static) if (num_data_ >= 128) reduction(+:Wpos, Wzero)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.0;
				if (y_data[i] > 0.) Wpos += w; else Wzero += w;
			}
			grad[0] = p0 * Wpos - q * Wzero;
		}//end "hurdle_gamma_varying_shape"
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
		else if (IsZeroCensShiftedGamma()) {
			const bool varying_shape = IsZeroCensShiftedGammaVaryingShape();
			const double xi0 = varying_shape ? aux_pars_[0] : aux_pars_[1];
			const double k_const = varying_shape ? 0. : aux_pars_[0];
			double dlogL_dlogk = 0.0, dlogL_dlogxi = 0.0;
#pragma omp parallel for schedule(static) reduction(+:dlogL_dlogk,dlogL_dlogxi)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.0;
				const double k = varying_shape ? ZeroCensGammaVarShapeShape(location_par[i + num_data_]) : k_const;
				double dLogXi, lEtaLogXi, dJetadLogXi;
				ZeroCensGammaLogXiQuantities(y_data[i], location_par[i], k, xi0, dLogXi, lEtaLogXi, dJetadLogXi);
				dlogL_dlogxi += w * dLogXi;
				if (!varying_shape) {
					double dLogK, lEtaLogK, dJetadLogK;
					ZeroCensGammaLogShapeQuantities(y_data[i], location_par[i], k, xi0, dLogK, lEtaLogK, dJetadLogK);
					dlogL_dlogk += w * dLogK;
				}
			}
			if (varying_shape) {
				grad[0] = -dlogL_dlogxi;// gradient of the negative log-likelihood; the shape is a location parameter block here
			}
			else {
				grad[0] = -dlogL_dlogk;
				grad[1] = -dlogL_dlogxi;
			}
		} // end zero-censored shifted gamma variants
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

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcSecondDerivLogLikFirstDerivInformationAuxPar(const double* y_data,
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
			else if (likelihood_type_ == "hurdle_gamma_varying_shape") {
				// The only auxiliary parameter is rho = logit(p0), which decouples from the location parameter blocks
				CHECK(ind_aux_par == 0);
				std::fill(second_deriv_loc_aux_par, second_deriv_loc_aux_par + num_data_, 0.);
				std::fill(deriv_information_aux_par, deriv_information_aux_par + num_data_, 0.);
			}//end "hurdle_gamma_varying_shape"
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
					// 'sdl' above is the cross derivative of the negative log-likelihood, whereas this output is the
					// cross derivative of the log-likelihood itself (as for "gamma" / "tweedie" / the EGPD families)
					second_deriv_loc_aux_par[i] = -w * sdl;
					deriv_information_aux_par[i] = w * dinfo;
				}
			} // end "zero_one_censored_shifted_gamma"
			else if (IsZeroCensShiftedGamma()) {
				const bool varying_shape = IsZeroCensShiftedGammaVaryingShape();
				CHECK(ind_aux_par >= 0 && ind_aux_par < num_aux_pars_estim_);
				const double xi0 = varying_shape ? aux_pars_[0] : aux_pars_[1];
				const double k_const = varying_shape ? 0. : aux_pars_[0];
				const bool wrt_log_xi = varying_shape || ind_aux_par == 1;// log(xi) is the only auxiliary parameter of the varying-shape variant
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const double k = varying_shape ? ZeroCensGammaVarShapeShape(location_par[i + num_data_]) : k_const;
					double dAux, lEtaAux, dJetadAux;
					if (wrt_log_xi) ZeroCensGammaLogXiQuantities(y_data[i], location_par[i], k, xi0, dAux, lEtaAux, dJetadAux);
					else ZeroCensGammaLogShapeQuantities(y_data[i], location_par[i], k, xi0, dAux, lEtaAux, dJetadAux);
					second_deriv_loc_aux_par[i] = w * lEtaAux;// d^2 log L / (d eta d aux)
					deriv_information_aux_par[i] = w * dJetadAux;
				}
			} // end zero-censored shifted gamma variants
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
}  // namespace GPBoost

#endif   // GPB_LIKELIHOODS_AUX_PARS_H_
