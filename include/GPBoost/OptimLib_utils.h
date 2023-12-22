/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_OPTIMLIB_UTILS_H_
#define GPB_OPTIMLIB_UTILS_H_

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>// OptimLib
#include <GPBoost/re_model_template.h>
#include <GPBoost/type_defs.h>

namespace GPBoost {

	// Forward declaration
	template<typename T_mat, typename T_chol>
	class REModelTemplate;

	// Auxiliary class for passing data to EvalLLforNMOptimLib for OpimtLib
	template<typename T_mat, typename T_chol>
	class OptDataOptimLib {
	public:
		//Constructor
		OptDataOptimLib(REModelTemplate<T_mat, T_chol>* re_model_templ,
			const double* fixed_effects,
			bool learn_covariance_parameters,
			const vec_t& cov_pars,
			bool profile_out_marginal_variance) {
			re_model_templ_ = re_model_templ;
			fixed_effects_ = fixed_effects;
			learn_covariance_parameters_ = learn_covariance_parameters;
			cov_pars_ = cov_pars;
			profile_out_marginal_variance_ = profile_out_marginal_variance;
		}
		REModelTemplate<T_mat, T_chol>* re_model_templ_;
		const double* fixed_effects_;//Externally provided fixed effects component of location parameter (only used for non-Gaussian likelihoods)
		bool learn_covariance_parameters_;//Indicates whether covariance parameters are optimized or not
		vec_t cov_pars_;//vector of covariance parameters (only used in case the covariance parameters are not estimated)
		bool profile_out_marginal_variance_;// If true, the error variance sigma is profiled out(= use closed - form expression for error / nugget variance)

	};//end EvalLLforOptim class definition

	/*!
	* \brief Auxiliary function for optimization using OptimLib
	* \param pars Parameter vector
	* \param[out] Gradient of function that is optimized
	* \param opt_data additional data passed to the function that is optimized
	*/
	template<typename T_mat, typename T_chol>
	double EvalLLforOptimLib(const vec_t& pars,
		vec_t* gradient,
		void* opt_data) {
		OptDataOptimLib<T_mat, T_chol>* objfn_data = reinterpret_cast<OptDataOptimLib<T_mat, T_chol>*>(opt_data);
		REModelTemplate<T_mat, T_chol>* re_model_templ_ = objfn_data->re_model_templ_;
		double neg_log_likelihood;
		vec_t cov_pars, beta, fixed_effects_vec, aux_pars;
		const double* fixed_effects_ptr;
		bool gradient_contains_error_var = re_model_templ_->GetLikelihood() == "gaussian" && !(objfn_data->profile_out_marginal_variance_);//If true, the error variance parameter (=nugget effect) is also included in the gradient, otherwise not
		bool has_covariates = re_model_templ_->HasCovariates();
		// Determine number of covariance and linear regression coefficient parameters
		int num_cov_pars_optim, num_covariates, num_aux_pars;
		if (objfn_data->learn_covariance_parameters_) {
			num_cov_pars_optim = re_model_templ_->GetNumCovPar();
			if (objfn_data->profile_out_marginal_variance_) {
				num_cov_pars_optim -= 1;
			}
		}
		else {
			num_cov_pars_optim = 0;
		}
		if (has_covariates) {
			num_covariates = re_model_templ_->GetNumCoef();
		}
		else {
			num_covariates = 0;
		}
		if (re_model_templ_->EstimateAuxPars()) {
			num_aux_pars = re_model_templ_->NumAuxPars();
		}
		else {
			num_aux_pars = 0;
		}
		CHECK((int)pars.size() == num_cov_pars_optim + num_covariates + num_aux_pars);
		// Extract covariance parameters, regression coefficients, and additional likelihood parameters from pars vector
		if (objfn_data->learn_covariance_parameters_) {
			if (objfn_data->profile_out_marginal_variance_) {
				cov_pars = vec_t(num_cov_pars_optim + 1);
				cov_pars[0] = 1.;//nugget effect
				cov_pars.segment(1, num_cov_pars_optim) = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			else {
				cov_pars = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
		}
		else {
			cov_pars = objfn_data->cov_pars_;
		}
		if (has_covariates) {
			beta = pars.segment(num_cov_pars_optim, num_covariates);
			re_model_templ_->UpdateFixedEffects(beta, objfn_data->fixed_effects_, fixed_effects_vec);
			fixed_effects_ptr = fixed_effects_vec.data();
		}//end has_covariates
		else {//no covariates
			fixed_effects_ptr = objfn_data->fixed_effects_;
		}
		if (re_model_templ_->EstimateAuxPars()) {
			aux_pars = pars.segment(num_cov_pars_optim + num_covariates, num_aux_pars).array().exp().matrix();
			re_model_templ_->SetAuxPars(aux_pars.data());
		}
		// Calculate objective function
		if (objfn_data->profile_out_marginal_variance_) {
			if (objfn_data->learn_covariance_parameters_) {
				re_model_templ_->CalcCovFactorOrModeAndNegLL(cov_pars, fixed_effects_ptr);
				cov_pars[0] = re_model_templ_->ProfileOutSigma2();
				re_model_templ_->EvalNegLogLikelihoodOnlyUpdateNuggetVariance(cov_pars[0], neg_log_likelihood);
			}
			else {
				re_model_templ_->EvalNegLogLikelihoodOnlyUpdateFixedEffects(cov_pars[0], neg_log_likelihood);
			}
		}
		else {
			re_model_templ_->CalcCovFactorOrModeAndNegLL(cov_pars, fixed_effects_ptr);
			neg_log_likelihood = re_model_templ_->GetNegLogLikelihood();
		}
		// Calculate gradient
		if (gradient) {
			vec_t grad_cov;
			if (objfn_data->learn_covariance_parameters_ || re_model_templ_->EstimateAuxPars()) {
				re_model_templ_->CalcGradCovParAuxPars(cov_pars, grad_cov, gradient_contains_error_var, false, fixed_effects_ptr);
			}
			if (objfn_data->learn_covariance_parameters_) {
				(*gradient).segment(0, num_cov_pars_optim) = grad_cov.segment(0, num_cov_pars_optim);
			}
			if (has_covariates) {
				vec_t grad_beta;
				re_model_templ_->CalcGradLinCoef(cov_pars[0], beta, grad_beta, fixed_effects_ptr);
				(*gradient).segment(num_cov_pars_optim, num_covariates) = grad_beta;
			}
			if (re_model_templ_->EstimateAuxPars()) {
				(*gradient).segment(num_cov_pars_optim + num_covariates, num_aux_pars) = grad_cov.segment(num_cov_pars_optim, num_aux_pars);
			}
		}
		// Check for NA or Inf
		if (re_model_templ_->GetLikelihood() != "gaussian") {
			if (std::isnan(neg_log_likelihood) || std::isinf(neg_log_likelihood)) {
				re_model_templ_->ResetLaplaceApproxModeToPreviousValue();
			}
			else if (gradient) {
				for (int i = 0; i < (int)((*gradient).size()); ++i) {
					if (std::isnan((*gradient)[i]) || std::isinf((*gradient)[i])) {
						re_model_templ_->ResetLaplaceApproxModeToPreviousValue();
						break;
					}
				}
			}
		}
		return neg_log_likelihood;
	} // end EvalLLforOptimLib

	/*!
	* \brief Find minimum for parameters using an external optimization library (OptimLib)
	* \param Pointer to main re_model_template.h object
	* \param cov_pars[out] Covariance parameters (initial values and output written on it). Note: any potential estimated additional likelihood parameters (aux_pars) are also written on this
	* \param beta[out] Linear regression coefficients (if there are any) (initial values and output written on it)
	* \param fixed_effects Externally provided fixed effects component of location parameter (only used for non-Gaussian likelihoods)
	* \param max_iter Maximal number of iterations
	* \param delta_rel_conv Convergence criterion: stop iteration if relative change in in parameters is below this value
	* \param convergence_criterion The convergence criterion used for terminating the optimization algorithm. Options: "relative_change_in_log_likelihood" or "relative_change_in_parameters"
	* \param num_it[out] Number of iterations
	* \param learn_covariance_parameters If true, covariance parameters and additional likelihood parameters (aux_pars) are estimated, otherwise not
	* \param optimizer Optimizer
	* \param profile_out_marginal_variance If true, the error variance sigma is profiled out (=use closed-form expression for error / nugget variance)
	* \param[out] neg_log_likelihood Value of negative log-likelihood or approximate marginal negative log-likelihood for non-Gaussian likelihoods
	* \param num_cov_par Number of covariance parameters
	* \param estim_aux_pars If true, any additional parameters for non-Gaussian likelihoods are also estimated (e.g., shape parameter of gamma likelihood)
	* \param nb_aux_pars Number of auxiliary parameters
	* \param aux_pars Pointer to aux_pars_ in likelihoods.h
	* \param has_covariates If true, the model linearly incluses covariates
	* \param sigma2 Variance of idiosyncratic error term (nugget effect)
	*/
	template<typename T_mat, typename T_chol>
	void OptimExternal(REModelTemplate<T_mat, T_chol>* re_model_templ,
		vec_t& cov_pars,
		vec_t& beta,
		const double* fixed_effects,
		int max_iter,
		double delta_rel_conv,
		string_t convergence_criterion,
		int& num_it,
		bool learn_covariance_parameters,
		string_t optimizer,
		bool profile_out_marginal_variance,
		double& neg_log_likelihood,
		int num_cov_par,
		bool estim_aux_pars,
		int nb_aux_pars,
		const double* aux_pars,
		bool has_covariates) {
		// Some checks
		if (estim_aux_pars) {
			CHECK(num_cov_par + nb_aux_pars == (int)cov_pars.size());
		}
		else {
			CHECK(num_cov_par == (int)cov_pars.size());
		}
		//if (has_covariates) {
		//	CHECK(beta.size() == X_.cols());
		//}
		// Determine number of covariance and linear regression coefficient parameters
		int num_cov_pars_optim, num_covariates, num_aux_pars;
		if (learn_covariance_parameters) {
			num_cov_pars_optim = num_cov_par;
			if (profile_out_marginal_variance) {
				num_cov_pars_optim = num_cov_par - 1;
			}
		}
		else {
			num_cov_pars_optim = 0;
		}
		if (has_covariates) {
			num_covariates = (int)beta.size();
		}
		else {
			num_covariates = 0;
		}
		bool estimate_aux_pars = estim_aux_pars && learn_covariance_parameters;
		if (estimate_aux_pars) {
			num_aux_pars = nb_aux_pars;
		}
		else {
			num_aux_pars = 0;
		}
		// Initialization of parameters
		vec_t pars_init(num_cov_pars_optim + num_covariates + num_aux_pars);
		if (learn_covariance_parameters) {
			if (profile_out_marginal_variance) {
				pars_init.segment(0, num_cov_pars_optim) = cov_pars.segment(1, num_cov_pars_optim).array().log().matrix();//exclude nugget and transform to log-scale
			}
			else {
				pars_init.segment(0, num_cov_pars_optim) = cov_pars.segment(0, num_cov_pars_optim).array().log().matrix();//transform to log-scale
			}
		}
		if (has_covariates) {
			pars_init.segment(num_cov_pars_optim, num_covariates) = beta;//regresion coefficients
		}
		if (estim_aux_pars) {
			for (int i = 0; i < num_aux_pars; ++i) {
				pars_init[num_cov_pars_optim + num_covariates + i] = std::log(aux_pars[i]);//transform to log-scale
			}
		}
		//Do optimization
		OptDataOptimLib<T_mat, T_chol> opt_data = OptDataOptimLib<T_mat, T_chol>(re_model_templ, fixed_effects, learn_covariance_parameters,
			cov_pars.segment(0, num_cov_par), profile_out_marginal_variance);
		optim::algo_settings_t settings;
		settings.iter_max = max_iter;
		if (convergence_criterion == "relative_change_in_parameters") {
			settings.rel_sol_change_tol = delta_rel_conv;
		}
		else if (convergence_criterion == "relative_change_in_log_likelihood") {
			settings.rel_objfn_change_tol = delta_rel_conv;
			settings.grad_err_tol = delta_rel_conv;
		}
		if (optimizer == "nelder_mead") {
			optim::nm(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		}
		else if (optimizer == "bfgs") {
			optim::bfgs(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		}
		else if (optimizer == "adam") {
			settings.gd_settings.method = 6;
			settings.gd_settings.ada_max = false;
			optim::gd(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		}
		//else if (optimizer == "adadelta") {// adadelta currently not supported as default settings do not always work
		//	settings.gd_settings.method = 5;
		//	optim::gd(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		//}
		num_it = (int)settings.opt_iter;
		neg_log_likelihood = settings.opt_fn_value;
		// Transform parameters back for export
		if (learn_covariance_parameters) {
			if (profile_out_marginal_variance) {
				cov_pars[0] = re_model_templ->Sigma2();
				cov_pars.segment(1, num_cov_par - 1) = pars_init.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			else {
				cov_pars.segment(0, num_cov_par) = pars_init.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
		}
		if (has_covariates) {
			beta = pars_init.segment(num_cov_pars_optim, num_covariates);
		}
		if (estimate_aux_pars) {
			for (int i = 0; i < num_aux_pars; ++i) {
				cov_pars[num_cov_par + i] = std::exp(pars_init[num_cov_pars_optim + num_covariates + i]);//back-transform to original scale
			}
		}
	}//end OptimExternal

}

#endif   // GPB_OPTIMLIB_UTILS_H_