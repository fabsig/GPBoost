/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022-2024 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_OPTIMLIB_UTILS_H_
#define GPB_OPTIMLIB_UTILS_H_

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <optim.hpp>// OptimLib
#include <LBFGS.h>// LBFGSpp
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
			bool learn_cov_aux_pars,
			const vec_t& cov_pars,
			bool profile_out_error_variance,
			optim::algo_settings_t* settings,
			string_t optimizer) {
			re_model_templ_ = re_model_templ;
			fixed_effects_ = fixed_effects;
			learn_cov_aux_pars_ = learn_cov_aux_pars;
			cov_pars_ = cov_pars;
			profile_out_error_variance_ = profile_out_error_variance;
			settings_ = settings;
			optimizer_ = optimizer;
		}
		REModelTemplate<T_mat, T_chol>* re_model_templ_;
		const double* fixed_effects_;//Externally provided fixed effects component of location parameter (only used for non-Gaussian likelihoods)
		bool learn_cov_aux_pars_;//Indicates whether covariance parameters are optimized or not
		vec_t cov_pars_;//vector of covariance parameters (only used in case the covariance parameters are not estimated)
		bool profile_out_error_variance_;// If true, the error variance sigma is profiled out(= use closed - form expression for error / nugget variance)
		optim::algo_settings_t* settings_;
		string_t optimizer_;//optimizer name

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
		double neg_log_likelihood = 9999999999;
		bool gradient_contains_error_var = re_model_templ_->IsGaussLikelihood() && !(objfn_data->profile_out_error_variance_);//If true, the error variance parameter (=nugget effect) is also included in the gradient, otherwise not
		bool has_covariates = re_model_templ_->HasCovariates();
		bool should_print_trace = false;
		bool should_redetermine_neighbors_vecchia = false;
		if (gradient != nullptr) {//"hack" for printing nice logging information and redetermining neighbors for the Vecchia approximation
			if ((*gradient).size() == 3 || (*gradient).size() == 2) {
				if ((*gradient)[0] >= -1.00000000002e30 && (*gradient)[0] <= -1e30 && (*gradient)[1] >= 1e30 && (*gradient)[1] <= 1.00000000002e30) {
					should_print_trace = true;
				}
				else if ((*gradient)[0] >= 1e30 && (*gradient)[0] <= 1.00000000002e30 && (*gradient)[1] >= -1.00000000002e30 && (*gradient)[1] <= -1e30) {
					if (objfn_data->learn_cov_aux_pars_) {
						should_redetermine_neighbors_vecchia = true;
					}
				}
			}
		}
		bool calc_likelihood = !should_redetermine_neighbors_vecchia && !should_print_trace;
		// Determine number of covariance and linear regression coefficient parameters
		int num_cov_pars_optim = 0, num_coef = 0, num_aux_pars = 0;
		if (objfn_data->learn_cov_aux_pars_) {
			num_cov_pars_optim = re_model_templ_->GetNumCovPar();
			if (objfn_data->profile_out_error_variance_) {
				num_cov_pars_optim -= 1;
			}
			if (re_model_templ_->EstimateAuxPars()) {
				num_aux_pars = re_model_templ_->NumAuxPars();
			}
		}
		if (has_covariates) {
			num_coef = re_model_templ_->GetNumCoef();
		}
		CHECK((int)pars.size() == num_cov_pars_optim + num_coef + num_aux_pars);
		// Extract covariance parameters, regression coefficients, and additional likelihood parameters from pars vector
		vec_t cov_pars, beta, fixed_effects_vec, aux_pars;
		const double* aux_pars_ptr = nullptr;
		const double* fixed_effects_ptr = nullptr;
		if (objfn_data->learn_cov_aux_pars_) {
			if (objfn_data->profile_out_error_variance_) {
				cov_pars = vec_t(num_cov_pars_optim + 1);
				cov_pars[0] = re_model_templ_->Sigma2();//nugget effect
				cov_pars.segment(1, num_cov_pars_optim) = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			else {
				cov_pars = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			if (re_model_templ_->EstimateAuxPars()) {
				aux_pars = pars.segment(num_cov_pars_optim + num_coef, num_aux_pars).array().exp().matrix();
				aux_pars_ptr = aux_pars.data();
			}
		}
		else {
			cov_pars = objfn_data->cov_pars_;
			if (should_print_trace) {
				aux_pars_ptr = re_model_templ_->GetAuxPars();
			}
		}
		if (has_covariates) {
			if (should_print_trace || calc_likelihood) {
				beta = pars.segment(num_cov_pars_optim, num_coef);
			}
		}
		if (should_print_trace) {//print trace information
			Log::REDebug("GPModel: parameters after optimization iteration number %d: ", (int)objfn_data->settings_->opt_iter + 1);
			re_model_templ_->PrintTraceParameters(cov_pars, beta, aux_pars_ptr, objfn_data->learn_cov_aux_pars_);
			if ((*gradient).size() == 3) {
				if (re_model_templ_->IsGaussLikelihood()) {
					Log::REDebug("Negative log-likelihood: %g", (*gradient)[2]);
				}
				else {
					Log::REDebug("Approximate negative marginal log-likelihood: %g", (*gradient)[2]);
				}
			}
		}//end should_print_trace
		else {//!should_print_trace
			if (should_redetermine_neighbors_vecchia) {
				re_model_templ_->SetNumIter((int)objfn_data->settings_->opt_iter);
				bool force_redermination = false;
				if ((*gradient)[2] >= 1e30 && (*gradient)[2] <= 1.00000000002e30){//hack to indicate that convergece has been achieved and nearest neighbors in Vecchia approximation should potentially been redetermined
					force_redermination = true;
				}
				if (re_model_templ_->ShouldRedetermineNearestNeighborsVecchiaInducingPointsFITC(force_redermination)) {
					re_model_templ_->RedetermineNearestNeighborsVecchiaInducingPointsFITC(force_redermination); // called only in certain iterations if gp_approx == "vecchia" and neighbors are selected based on correlations and not distances
					neg_log_likelihood = 1.00000000001e30;//hack to tell the optimizers that the neighbors have indeed been redetermined
				}
			} //end should_redetermine_neighbors_vecchia
			if (calc_likelihood) {
				if (has_covariates) {
					re_model_templ_->UpdateFixedEffects(beta, objfn_data->fixed_effects_, fixed_effects_vec);
					fixed_effects_ptr = fixed_effects_vec.data();
				}//end has_covariates
				else {//no covariates
					fixed_effects_ptr = objfn_data->fixed_effects_;
				}
				if (objfn_data->learn_cov_aux_pars_ && re_model_templ_->EstimateAuxPars()) {
					re_model_templ_->SetAuxPars(aux_pars_ptr);
				}
				// Calculate objective function
				if (objfn_data->profile_out_error_variance_) {
					if (objfn_data->learn_cov_aux_pars_) {
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
			}//end calc_likelihood
			// Calculate gradient
			if (gradient && !should_redetermine_neighbors_vecchia) {
				vec_t grad_cov, grad_beta;
				re_model_templ_->CalcGradPars(cov_pars, cov_pars[0], objfn_data->learn_cov_aux_pars_, has_covariates,
					grad_cov, grad_beta, gradient_contains_error_var, false, fixed_effects_ptr, false);
				if (objfn_data->learn_cov_aux_pars_) {
					(*gradient).segment(0, num_cov_pars_optim) = grad_cov.segment(0, num_cov_pars_optim);
					if (re_model_templ_->EstimateAuxPars()) {
						(*gradient).segment(num_cov_pars_optim + num_coef, num_aux_pars) = grad_cov.segment(num_cov_pars_optim, num_aux_pars);
					}
				}
				if (has_covariates) {
					(*gradient).segment(num_cov_pars_optim, num_coef) = grad_beta;
				}
			}
			if (calc_likelihood || gradient) {
				// Check for NA or Inf
				if (!(re_model_templ_->IsGaussLikelihood())) {
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
			}//end Check for NA or Inf
		}//end !should_print_trace
		return neg_log_likelihood;
	} // end EvalLLforOptimLib

	template<typename T_mat, typename T_chol>
	class EvalLLforLBFGSpp {
	public:
		REModelTemplate<T_mat, T_chol>* re_model_templ_;
		const double* fixed_effects_;//Externally provided fixed effects component of location parameter (only used for non-Gaussian likelihoods)
		bool learn_cov_aux_pars_;//Indicates whether covariance and auxiliary parameters are optimized or not
		vec_t cov_pars_;//vector of covariance parameters (only used in case the covariance parameters are not estimated)
		bool profile_out_error_variance_;// If true, the error variance sigma is profiled out (= use closed-form expression for error / nugget variance)
		bool profile_out_regression_coef_;// If true, the linear regression coefficients are profiled out (= use closed-form WLS expression)

		EvalLLforLBFGSpp(REModelTemplate<T_mat, T_chol>* re_model_templ,
			const double* fixed_effects,
			bool learn_cov_aux_pars,
			const vec_t& cov_pars,
			bool profile_out_error_variance,
			bool profile_out_regression_coef) {
			re_model_templ_ = re_model_templ;
			fixed_effects_ = fixed_effects;
			learn_cov_aux_pars_ = learn_cov_aux_pars;
			cov_pars_ = cov_pars;
			profile_out_error_variance_ = profile_out_error_variance;
			profile_out_regression_coef_ = profile_out_regression_coef;
			if (profile_out_error_variance_) {
				CHECK(re_model_templ_->IsGaussLikelihood());
			}
			if (profile_out_regression_coef_) {
				CHECK(re_model_templ_->IsGaussLikelihood());
			}
		}
		double operator()(const vec_t& pars,
			vec_t& gradient,
			bool eval_likelihood,
			bool calc_gradient) {
			double neg_log_likelihood = 1e99;
			vec_t cov_pars, beta, fixed_effects_vec, aux_pars;
			const double* fixed_effects_ptr = nullptr;
			bool gradient_contains_error_var = re_model_templ_->IsGaussLikelihood() && !profile_out_error_variance_;//If true, the error variance parameter (=nugget effect) is also included in the gradient, otherwise not
			bool estimate_coef_using_bfgs = re_model_templ_->HasCovariates() && !profile_out_regression_coef_;
			bool estimate_coef_using_wls = re_model_templ_->HasCovariates() && profile_out_regression_coef_;
			// Determine number of covariance and linear regression coefficient parameters
			int num_cov_pars_optim = 0, num_coef = 0, num_aux_pars = 0;
			if (learn_cov_aux_pars_) {
				num_cov_pars_optim = re_model_templ_->GetNumCovPar();
				if (profile_out_error_variance_) {
					num_cov_pars_optim -= 1;
				}
				if (re_model_templ_->EstimateAuxPars()) {
					num_aux_pars = re_model_templ_->NumAuxPars();
				}
			}
			if (estimate_coef_using_bfgs) {
				num_coef = re_model_templ_->GetNumCoef();
			}
			CHECK((int)pars.size() == num_cov_pars_optim + num_coef + num_aux_pars);
			// Extract covariance parameters, regression coefficients, and additional likelihood parameters from pars vector
			if (learn_cov_aux_pars_) {
				if (profile_out_error_variance_) {
					cov_pars = vec_t(num_cov_pars_optim + 1);
					cov_pars[0] = re_model_templ_->Sigma2();//nugget effect
					cov_pars.segment(1, num_cov_pars_optim) = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
				}
				else {
					cov_pars = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
				}
				if (re_model_templ_->EstimateAuxPars()) {
					aux_pars = pars.segment(num_cov_pars_optim + num_coef, num_aux_pars).array().exp().matrix();
					re_model_templ_->SetAuxPars(aux_pars.data());
				}
			}
			else {
				cov_pars = cov_pars_;
			}
			if (!(re_model_templ_->HasCovariates())) {//no covariates
				fixed_effects_ptr = fixed_effects_;
			}
			else if (estimate_coef_using_bfgs) {
				beta = pars.segment(num_cov_pars_optim, num_coef);
				re_model_templ_->UpdateFixedEffects(beta, fixed_effects_, fixed_effects_vec); // set y_ to resid = y - X * beta - fixed_effcts for Gaussian likelihood or fixed_effects_vec = fixed_effects_ + X * beta for non-Gaussian likelihoods
				fixed_effects_ptr = fixed_effects_vec.data();
			}
			// Calculate objective function
			if (eval_likelihood) {
				if (re_model_templ_->IsGaussLikelihood()) {
					if (estimate_coef_using_wls) {
						re_model_templ_->CalcCovFactorOrModeAndNegLL(cov_pars, fixed_effects_ptr);
						re_model_templ_->ProfileOutCoef(fixed_effects_, fixed_effects_vec);//this sets y_ to resid = y - X*beta - fixed_effcts
						fixed_effects_ptr = fixed_effects_vec.data();
					}
					if(learn_cov_aux_pars_) {
						if (profile_out_error_variance_) {
							//calculate yTPsiInvy for profiling out the nugget variance below. Note that the nugget variance used here is irrelvant. The 'neg_log_likelihood' is recalculated below with the correct sigma2
							if (estimate_coef_using_wls) {
								re_model_templ_->EvalNegLogLikelihoodOnlyUpdateFixedEffects(cov_pars[0], neg_log_likelihood);
							}
							else {
								re_model_templ_->CalcCovFactorOrModeAndNegLL(cov_pars, fixed_effects_ptr);
							}
							cov_pars[0] = re_model_templ_->ProfileOutSigma2();
							re_model_templ_->EvalNegLogLikelihoodOnlyUpdateNuggetVariance(cov_pars[0], neg_log_likelihood);
						}//end profile_out_error_variance_
						else {//!profile_out_error_variance_
							if (estimate_coef_using_wls) {
								re_model_templ_->EvalNegLogLikelihoodOnlyUpdateFixedEffects(cov_pars[0], neg_log_likelihood);
							}
							else {
								re_model_templ_->CalcCovFactorOrModeAndNegLL(cov_pars, fixed_effects_ptr);
								neg_log_likelihood = re_model_templ_->GetNegLogLikelihood();
							}
						}
					}//end learn_cov_aux_pars_
					else {// ! learn_cov_aux_pars_
						re_model_templ_->EvalNegLogLikelihoodOnlyUpdateFixedEffects(cov_pars[0], neg_log_likelihood);
					}
				}//end re_model_templ_->IsGaussLikelihood()
				else {// non-Gaussian likelihood
					re_model_templ_->CalcCovFactorOrModeAndNegLL(cov_pars, fixed_effects_ptr);
					neg_log_likelihood = re_model_templ_->GetNegLogLikelihood();
				}
			}//end eval_likelihood
			if (calc_gradient) {
				// Calculate gradient
				vec_t grad_cov, grad_beta;
				bool calc_cov_aux_par_grad = learn_cov_aux_pars_ || re_model_templ_->EstimateAuxPars();
				re_model_templ_->CalcGradPars(cov_pars, cov_pars[0], calc_cov_aux_par_grad, estimate_coef_using_bfgs,
					grad_cov, grad_beta, gradient_contains_error_var, false, fixed_effects_ptr, false);//note: fixed_effects_ptr is only used for non-Gaussian likelihood
				if (learn_cov_aux_pars_) {
					gradient.segment(0, num_cov_pars_optim) = grad_cov.segment(0, num_cov_pars_optim);
				}
				if (estimate_coef_using_bfgs) {
					gradient.segment(num_cov_pars_optim, num_coef) = grad_beta;
				}
				if (re_model_templ_->EstimateAuxPars()) {
					gradient.segment(num_cov_pars_optim + num_coef, num_aux_pars) = grad_cov.segment(num_cov_pars_optim, num_aux_pars);
				}
			}//end calc_gradient
			// Check for NA or Inf
			if (!(re_model_templ_->IsGaussLikelihood())) {
				if (std::isnan(neg_log_likelihood) || std::isinf(neg_log_likelihood)) {
					re_model_templ_->ResetLaplaceApproxModeToPreviousValue();
				}
				else if (calc_gradient) {
					for (int i = 0; i < (int)(gradient.size()); ++i) {
						if (std::isnan(gradient[i]) || std::isinf(gradient[i])) {
							re_model_templ_->ResetLaplaceApproxModeToPreviousValue();
							break;
						}
					}
				}
			}
			return neg_log_likelihood;
		}

		bool HasCovariates() {
			return(re_model_templ_->HasCovariates());
		}

		/*!
		* \brief Set the iteration number in re_model_templ_ (e.g. for correlation-based neighbor selection in Vecchia approximations)
		* \param iter iteration number
		*/
		void SetNumIter(int iter) {
			re_model_templ_->SetNumIter(iter);
		}

		/*!
		* \brief Write the current values of profiled-out variables (if there are any such as nugget effects, regression coefficients) to their lag1 variables
		*/
		void SetLag1ProfiledOutVariables() {
			re_model_templ_->SetLag1ProfiledOutVariables(profile_out_error_variance_, profile_out_regression_coef_);
		}

		/*!
		* \brief Reset the profiled-out variables (if there are any such as nugget effects, regression coefficients) to their lag1 variables
		*/
		void ResetProfiledOutVariablesToLag1() {
			re_model_templ_->ResetProfiledOutVariablesToLag1(profile_out_error_variance_, profile_out_regression_coef_);
		}

		/*!
		* \brief Indicates whether inducing points or/and correlation-based nearest neighbors for Vecchia approximation should be updated
		* \param force_redermination If true, inducing points/neighbors are redetermined if applicaple irrespective of num_iter_
		* \return True, if inducing points/nearest neighbors have been redetermined
		*/
		bool ShouldRedetermineNearestNeighborsVecchiaInducingPointsFITC(bool force_redermination) {
			return(re_model_templ_->ShouldRedetermineNearestNeighborsVecchiaInducingPointsFITC(force_redermination));
		}

		/*!
		* \brief Redetermine inducing points or/and correlation-based nearest neighbors for Vecchia approximation
		* \param force_redermination If true, inducing points/neighbors are redetermined if applicaple irrespective of num_iter_
		*/
		void RedetermineNearestNeighborsVecchiaInducingPointsFITC(bool force_redermination) {
			re_model_templ_->RedetermineNearestNeighborsVecchiaInducingPointsFITC(force_redermination);
		}

		/*!
		* \brief Indicates whether covariance and auxiliary parameters should be estimated
		* \return True, if covariance and auxiliary parameters should be estimated
		*/
		bool LearnCovarianceParameters() {
			return(learn_cov_aux_pars_);
		}

		bool IsGaussLikelihood() {
			return(re_model_templ_->IsGaussLikelihood());
		}

		bool ProfileOutRegreCoef() {
			return(re_model_templ_->IsGaussLikelihood() && re_model_templ_->HasCovariates() && profile_out_regression_coef_);
		}

		bool ProfileOutErrorVariance() {
			return(re_model_templ_->IsGaussLikelihood() && profile_out_error_variance_);
		}		

		/*!
		* \brief Print out trace information
		* \param pars Current parameters
		* \param iter iteration number
		* \param fx current objective value (negative log-lilekihood
		*/
		void Logging(const vec_t& pars,
			int iter,
			double fx) const {
			vec_t cov_pars, beta, aux_pars;
			const double* aux_pars_ptr = nullptr;
			bool estimate_coef_using_bfgs = re_model_templ_->HasCovariates() && !profile_out_regression_coef_;
			bool estimate_coef_using_wls = re_model_templ_->HasCovariates() && profile_out_regression_coef_;
			// Determine number of covariance and linear regression coefficient parameters
			int num_cov_pars_optim = 0, num_coef = 0, num_aux_pars = 0;
			if (learn_cov_aux_pars_) {
				num_cov_pars_optim = re_model_templ_->GetNumCovPar();
				if (profile_out_error_variance_) {
					num_cov_pars_optim -= 1;
				}
				if (re_model_templ_->EstimateAuxPars()) {
					num_aux_pars = re_model_templ_->NumAuxPars();
				}
			}
			if (estimate_coef_using_bfgs) {
				num_coef = re_model_templ_->GetNumCoef();
			}
			CHECK((int)pars.size() == num_cov_pars_optim + num_coef + num_aux_pars);
			// Extract covariance parameters, regression coefficients, and additional likelihood parameters from pars vector
			if (learn_cov_aux_pars_) {
				if (profile_out_error_variance_) {
					cov_pars = vec_t(num_cov_pars_optim + 1);
					cov_pars[0] = re_model_templ_->Sigma2();//nugget effect
					cov_pars.segment(1, num_cov_pars_optim) = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
				}
				else {
					cov_pars = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
				}
				if (re_model_templ_->EstimateAuxPars()) {
					aux_pars = pars.segment(num_cov_pars_optim + num_coef, num_aux_pars).array().exp().matrix();
					aux_pars_ptr = aux_pars.data();
				}
			}
			else {
				cov_pars = cov_pars_;
				aux_pars_ptr = re_model_templ_->GetAuxPars();
			}
			if (estimate_coef_using_bfgs) {
				beta = pars.segment(num_cov_pars_optim, num_coef);
			}
			else if(estimate_coef_using_wls){
				re_model_templ_->GetBeta(beta);
			}
			Log::REDebug("GPModel: parameters after optimization iteration number %d: ", iter);
			re_model_templ_->PrintTraceParameters(cov_pars, beta, aux_pars_ptr, learn_cov_aux_pars_);
			if (re_model_templ_->IsGaussLikelihood()) {
				Log::REDebug("Negative log-likelihood: %g", fx);
			}
			else {
				Log::REDebug("Approximate negative marginal log-likelihood: %g", fx);
			}
		}//end Logging

		/*!
		* \brief Get the maximal step length along a direction such that the change in the parameters is not overly large
		* \param pars Current / lag1 value of pars
		* \param neg_step_dir Negative step direction for making updates
		*/
		double GetMaximalLearningRate(const vec_t& pars,
			vec_t& neg_step_dir) const {
			bool estimate_coef_using_bfgs = re_model_templ_->HasCovariates() && !profile_out_regression_coef_;
			// Determine number of covariance and linear regression coefficient parameters
			int num_cov_pars_optim = 0, num_coef = 0, num_aux_pars = 0;
			if (learn_cov_aux_pars_) {
				num_cov_pars_optim = re_model_templ_->GetNumCovPar();
				if (profile_out_error_variance_) {
					num_cov_pars_optim -= 1;
				}
				if (re_model_templ_->EstimateAuxPars()) {
					num_aux_pars = re_model_templ_->NumAuxPars();
				}
			}
			if (estimate_coef_using_bfgs) {
				num_coef = re_model_templ_->GetNumCoef();
			}
			CHECK((int)pars.size() == num_cov_pars_optim + num_coef + num_aux_pars);
			CHECK((int)neg_step_dir.size() == num_cov_pars_optim + num_coef + num_aux_pars);
			double max_lr = 1e99;
			if (learn_cov_aux_pars_) {
				vec_t neg_step_dir_cov_aux_pars(num_cov_pars_optim + num_aux_pars);
				neg_step_dir_cov_aux_pars.segment(0, num_cov_pars_optim) = neg_step_dir.segment(0, num_cov_pars_optim);
				if (re_model_templ_->EstimateAuxPars()) {
					neg_step_dir_cov_aux_pars.segment(num_cov_pars_optim, num_aux_pars) = neg_step_dir.segment(num_cov_pars_optim + num_coef, num_aux_pars);
				}
				max_lr = re_model_templ_->MaximalLearningRateCovAuxPars(neg_step_dir_cov_aux_pars);
			}
			if (estimate_coef_using_bfgs) {
				vec_t beta = pars.segment(num_cov_pars_optim, num_coef);
				vec_t neg_step_dir_beta = neg_step_dir.segment(num_cov_pars_optim, num_coef);
				double max_lr_beta = re_model_templ_->MaximalLearningRateCoef(beta, neg_step_dir_beta);
				if (max_lr_beta < max_lr) {
					max_lr = max_lr_beta;
				}
			}
			return(max_lr);
		}//end GetMaximalLearningRate

	};//end EvalLLforLBFGSpp

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
	* \param learn_cov_aux_pars If true, covariance parameters and additional likelihood parameters (aux_pars) are estimated, otherwise not
	* \param optimizer Optimizer
	* \param profile_out_error_variance If true, the error variance sigma is profiled out (=use closed-form expression for error / nugget variance)
	* \param profile_out_regression_coef If true, the linear regression coefficients are profiled out (= use closed-form WLS expression). Applies only to lbfgs and Gaussian likelihood
	* \param[out] neg_log_likelihood Value of negative log-likelihood or approximate marginal negative log-likelihood for non-Gaussian likelihoods
	* \param num_cov_par Number of covariance parameters
	* \param nb_aux_pars Number of auxiliary parameters
	* \param aux_pars Pointer to aux_pars_ in likelihoods.h
	* \param has_covariates If true, the model linearly incluses covariates
	* \param initial_step_factor Only for 'lbfgs': The initial step length in the first iteration is this factor divided by the search direction (i.e. gradient)
	* \param reuse_m_bfgs_from_previous_call If true, the approximate Hessian for the LBFGS are kept at the values from a previous call and not re-initialized (applies only to LBFGSSolver)
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
		bool learn_cov_aux_pars,
		string_t optimizer,
		bool profile_out_error_variance,
		bool profile_out_regression_coef,
		double& neg_log_likelihood,
		int num_cov_par,
		int nb_aux_pars,
		const double* aux_pars,
		bool has_covariates,
		double initial_step_factor,
		bool reuse_m_bfgs_from_previous_call) {
		// Some checks
		if (re_model_templ->EstimateAuxPars()) {
			CHECK(num_cov_par + nb_aux_pars == (int)cov_pars.size());
		}
		else {
			CHECK(num_cov_par == (int)cov_pars.size());
		}
		if (profile_out_regression_coef) {
			CHECK(optimizer == "lbfgs" || optimizer == "lbfgs_linesearch_nocedal_wright");
		}
		//if (has_covariates) {
		//	CHECK(beta.size() == X_.cols());
		//}
		// Determine number of covariance and linear regression coefficient parameters
		int num_cov_pars_optim = 0, num_coef = 0, num_aux_pars = 0;
		if (learn_cov_aux_pars) {
			num_cov_pars_optim = num_cov_par;
			if (profile_out_error_variance) {
				num_cov_pars_optim = num_cov_par - 1;
			}
			if (re_model_templ->EstimateAuxPars()) {
				num_aux_pars = nb_aux_pars;
			}
		}
		if (has_covariates && !profile_out_regression_coef) {
			num_coef = (int)beta.size();
		}
		// Initialization of parameters
		vec_t pars_init(num_cov_pars_optim + num_coef + num_aux_pars);//order of parameters: 1. cov_pars, 2. coefs, 3. aux_pars
		if (learn_cov_aux_pars) {
			if (profile_out_error_variance) {
				pars_init.segment(0, num_cov_pars_optim) = cov_pars.segment(1, num_cov_pars_optim).array().log().matrix();//exclude nugget and transform to log-scale
			}
			else {
				pars_init.segment(0, num_cov_pars_optim) = cov_pars.segment(0, num_cov_pars_optim).array().log().matrix();//transform to log-scale
			}
			if (re_model_templ->EstimateAuxPars()) {
				for (int i = 0; i < num_aux_pars; ++i) {
					pars_init[num_cov_pars_optim + num_coef + i] = std::log(aux_pars[i]);//transform to log-scale
				}
			}
		}
		if (has_covariates && !profile_out_regression_coef) {
			pars_init.segment(num_cov_pars_optim, num_coef) = beta;//regresion coefficients
		}
		//Do optimization
		optim::algo_settings_t settings;
		settings.iter_max = max_iter;
		OptDataOptimLib<T_mat, T_chol> opt_data = OptDataOptimLib<T_mat, T_chol>(re_model_templ, fixed_effects, learn_cov_aux_pars,
			cov_pars.segment(0, num_cov_par), profile_out_error_variance, &settings, optimizer);
		if (convergence_criterion == "relative_change_in_parameters") {
			settings.rel_sol_change_tol = delta_rel_conv;
			settings.rel_objfn_change_tol = 1e-20;
			settings.grad_err_tol = 1e-20;
		}
		else if (convergence_criterion == "relative_change_in_log_likelihood") {
			settings.rel_objfn_change_tol = delta_rel_conv;
			settings.grad_err_tol = delta_rel_conv;
			settings.rel_sol_change_tol = 1e-20;
		}
		if (optimizer == "nelder_mead") {
			optim::nm(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		}
		else if (optimizer == "bfgs_optim_lib") {
			optim::bfgs(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		}
		else if (optimizer == "adam") {
			settings.gd_settings.method = 6;
			settings.gd_settings.ada_max = false;
			optim::gd(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		}
		else if (optimizer == "lbfgs" || optimizer == "lbfgs_linesearch_nocedal_wright") {
			LBFGSpp::LBFGSParam<double> param_LBFGSpp;
			param_LBFGSpp.max_iterations = max_iter;
			param_LBFGSpp.past = 1;//convergence should be determined by checking the change in the objective function and not the norm of the gradient
			param_LBFGSpp.delta = delta_rel_conv;//tolerence for relative change in objective function as convergence chec
			param_LBFGSpp.epsilon = 1e-20;//tolerance for norm of gradient as convergence check
			param_LBFGSpp.epsilon_rel = 1e-20;//tolerance for norm of gradient relative to norm of parameters as convergence check
			param_LBFGSpp.max_linesearch = 20;
			param_LBFGSpp.m = 6;
			param_LBFGSpp.initial_step_factor = initial_step_factor;
			EvalLLforLBFGSpp<T_mat, T_chol> ll_fun(re_model_templ, fixed_effects, learn_cov_aux_pars,
				cov_pars.segment(0, num_cov_par), profile_out_error_variance, profile_out_regression_coef);
			if (optimizer == "lbfgs") {
				param_LBFGSpp.linesearch = 1;//LBFGS_LINESEARCH_BACKTRACKING_ARMIJO
				LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBacktracking> solver(param_LBFGSpp);
				num_it = solver.minimize(ll_fun, pars_init, neg_log_likelihood, reuse_m_bfgs_from_previous_call, re_model_templ->GetMBFGS());
			}
			else if (optimizer == "lbfgs_linesearch_nocedal_wright") {
				param_LBFGSpp.linesearch = 3;//LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
				LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchNocedalWright> solver(param_LBFGSpp);
				num_it = solver.minimize(ll_fun, pars_init, neg_log_likelihood, reuse_m_bfgs_from_previous_call, re_model_templ->GetMBFGS());
			}
		}
		//else if (optimizer == "adadelta") {// adadelta currently not supported as default settings do not always work
		//	settings.gd_settings.method = 5;
		//	optim::gd(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		//}
		if (optimizer != "lbfgs" && optimizer != "lbfgs_linesearch_nocedal_wright") {//only for optimizers from OptimLib
			num_it = (int)settings.opt_iter;
			neg_log_likelihood = settings.opt_fn_value;
			if (profile_out_error_variance || profile_out_regression_coef) {
				vec_t* grad_dummy = nullptr;
				EvalLLforOptimLib<T_mat, T_chol>(pars_init, grad_dummy, &opt_data);//re-evaluate log-likelihood to make sure that the profiled-out variables are correct
			}
		}
		// Transform parameters back for export
		if (learn_cov_aux_pars) {
			if (profile_out_error_variance) {
				cov_pars[0] = re_model_templ->Sigma2();
				cov_pars.segment(1, num_cov_par - 1) = pars_init.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			else {
				cov_pars.segment(0, num_cov_par) = pars_init.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			if (re_model_templ->EstimateAuxPars()) {
				for (int i = 0; i < num_aux_pars; ++i) {
					cov_pars[num_cov_par + i] = std::exp(pars_init[num_cov_pars_optim + num_coef + i]);//back-transform to original scale
				}
			}
		}
		if (has_covariates && !profile_out_regression_coef) {
			beta = pars_init.segment(num_cov_pars_optim, num_coef);
		}
	}//end OptimExternal

}

#endif   // GPB_OPTIMLIB_UTILS_H_