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
			bool profile_out_marginal_variance,
			optim::algo_settings_t* settings,
			string_t optimizer) {
			re_model_templ_ = re_model_templ;
			fixed_effects_ = fixed_effects;
			learn_cov_aux_pars_ = learn_cov_aux_pars;
			cov_pars_ = cov_pars;
			profile_out_marginal_variance_ = profile_out_marginal_variance;
			settings_ = settings;
			optimizer_ = optimizer;
		}
		REModelTemplate<T_mat, T_chol>* re_model_templ_;
		const double* fixed_effects_;//Externally provided fixed effects component of location parameter (only used for non-Gaussian likelihoods)
		bool learn_cov_aux_pars_;//Indicates whether covariance parameters are optimized or not
		vec_t cov_pars_;//vector of covariance parameters (only used in case the covariance parameters are not estimated)
		bool profile_out_marginal_variance_;// If true, the error variance sigma is profiled out(= use closed - form expression for error / nugget variance)
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
		bool gradient_contains_error_var = re_model_templ_->GetLikelihood() == "gaussian" && !(objfn_data->profile_out_marginal_variance_);//If true, the error variance parameter (=nugget effect) is also included in the gradient, otherwise not
		bool has_covariates = re_model_templ_->HasCovariates();
		bool should_print_trace = false;
		bool should_redetermine_neighbors_vecchia = false;
		if (gradient != nullptr) {//"hack" for printing nice logging information and redermininig neighbors for the Vecchia approximation
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
		int num_cov_pars_optim = 0, num_covariates = 0, num_aux_pars = 0;
		if (objfn_data->learn_cov_aux_pars_) {
			num_cov_pars_optim = re_model_templ_->GetNumCovPar();
			if (objfn_data->profile_out_marginal_variance_) {
				num_cov_pars_optim -= 1;
			}
			if (re_model_templ_->EstimateAuxPars()) {
				num_aux_pars = re_model_templ_->NumAuxPars();
			}
		}
		if (has_covariates) {
			num_covariates = re_model_templ_->GetNumCoef();
		}
		CHECK((int)pars.size() == num_cov_pars_optim + num_covariates + num_aux_pars);
		// Extract covariance parameters, regression coefficients, and additional likelihood parameters from pars vector
		vec_t cov_pars, beta, fixed_effects_vec, aux_pars;
		const double* aux_pars_ptr = nullptr;
		const double* fixed_effects_ptr = nullptr;
		if (objfn_data->learn_cov_aux_pars_) {
			if (objfn_data->profile_out_marginal_variance_) {
				cov_pars = vec_t(num_cov_pars_optim + 1);
				cov_pars[0] = re_model_templ_->Sigma2();//nugget effect
				cov_pars.segment(1, num_cov_pars_optim) = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			else {
				cov_pars = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			if (re_model_templ_->EstimateAuxPars()) {
				aux_pars = pars.segment(num_cov_pars_optim + num_covariates, num_aux_pars).array().exp().matrix();
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
				beta = pars.segment(num_cov_pars_optim, num_covariates);
			}
		}
		if (should_print_trace) {//print trace information
			Log::REDebug("GPModel: parameters after optimization iteration number %d: ", (int)objfn_data->settings_->opt_iter + 1);
			re_model_templ_->PrintTraceParameters(cov_pars, beta, aux_pars_ptr);
			if ((*gradient).size() == 3) {
				if (re_model_templ_->GetLikelihood() == "gaussian") {
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
				if (re_model_templ_->ShouldRedetermineNearestNeighborsVecchia()) {
					re_model_templ_->RedetermineNearestNeighborsVecchia(); // called only in certain iterations if gp_approx == "vecchia" and neighbors are selected based on correlations and not distances
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
				if (objfn_data->profile_out_marginal_variance_) {
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
				vec_t grad_cov;
				if (objfn_data->learn_cov_aux_pars_) {
					re_model_templ_->CalcGradCovParAuxPars(cov_pars, grad_cov, gradient_contains_error_var, false, fixed_effects_ptr);
					(*gradient).segment(0, num_cov_pars_optim) = grad_cov.segment(0, num_cov_pars_optim);
					if (re_model_templ_->EstimateAuxPars()) {
						(*gradient).segment(num_cov_pars_optim + num_covariates, num_aux_pars) = grad_cov.segment(num_cov_pars_optim, num_aux_pars);
					}
				}
				if (has_covariates) {
					vec_t grad_beta;
					re_model_templ_->CalcGradLinCoef(cov_pars[0], beta, grad_beta, fixed_effects_ptr);
					(*gradient).segment(num_cov_pars_optim, num_covariates) = grad_beta;
				}
			}
			if (calc_likelihood || gradient) {
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
		bool profile_out_marginal_variance_;// If true, the error variance sigma is profiled out(= use closed - form expression for error / nugget variance)

		EvalLLforLBFGSpp(REModelTemplate<T_mat, T_chol>* re_model_templ,
			const double* fixed_effects,
			bool learn_cov_aux_pars,
			const vec_t& cov_pars,
			bool profile_out_marginal_variance) {
			re_model_templ_ = re_model_templ;
			fixed_effects_ = fixed_effects;
			learn_cov_aux_pars_ = learn_cov_aux_pars;
			cov_pars_ = cov_pars;
			profile_out_marginal_variance_ = profile_out_marginal_variance;
		}
		double operator()(const vec_t& pars,
			vec_t& gradient,
			bool calc_gradient = true) {
			double neg_log_likelihood;
			vec_t cov_pars, beta, fixed_effects_vec, aux_pars;
			const double* fixed_effects_ptr;
			bool gradient_contains_error_var = re_model_templ_->GetLikelihood() == "gaussian" && !profile_out_marginal_variance_;//If true, the error variance parameter (=nugget effect) is also included in the gradient, otherwise not
			bool has_covariates = re_model_templ_->HasCovariates();
			// Determine number of covariance and linear regression coefficient parameters
			int num_cov_pars_optim = 0, num_covariates = 0, num_aux_pars = 0;
			if (learn_cov_aux_pars_) {
				num_cov_pars_optim = re_model_templ_->GetNumCovPar();
				if (profile_out_marginal_variance_) {
					num_cov_pars_optim -= 1;
				}
				if (re_model_templ_->EstimateAuxPars()) {
					num_aux_pars = re_model_templ_->NumAuxPars();
				}
			}
			if (has_covariates) {
				num_covariates = re_model_templ_->GetNumCoef();
			}
			CHECK((int)pars.size() == num_cov_pars_optim + num_covariates + num_aux_pars);
			// Extract covariance parameters, regression coefficients, and additional likelihood parameters from pars vector
			if (learn_cov_aux_pars_) {
				if (profile_out_marginal_variance_) {
					cov_pars = vec_t(num_cov_pars_optim + 1);
					cov_pars[0] = re_model_templ_->Sigma2();//nugget effect
					cov_pars.segment(1, num_cov_pars_optim) = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
				}
				else {
					cov_pars = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
				}
				if (re_model_templ_->EstimateAuxPars()) {
					aux_pars = pars.segment(num_cov_pars_optim + num_covariates, num_aux_pars).array().exp().matrix();
					re_model_templ_->SetAuxPars(aux_pars.data());
				}
			}
			else {
				cov_pars = cov_pars_;
			}
			if (has_covariates) {
				beta = pars.segment(num_cov_pars_optim, num_covariates);
				re_model_templ_->UpdateFixedEffects(beta, fixed_effects_, fixed_effects_vec);
				fixed_effects_ptr = fixed_effects_vec.data();
			}//end has_covariates
			else {//no covariates
				fixed_effects_ptr = fixed_effects_;
			}
			// Calculate objective function
			if (profile_out_marginal_variance_) {
				if (learn_cov_aux_pars_) {
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
			if (calc_gradient) {
				// Calculate gradient
				vec_t grad_cov;
				if (learn_cov_aux_pars_ || re_model_templ_->EstimateAuxPars()) {
					re_model_templ_->CalcGradCovParAuxPars(cov_pars, grad_cov, gradient_contains_error_var, false, fixed_effects_ptr);
				}
				if (learn_cov_aux_pars_) {
					gradient.segment(0, num_cov_pars_optim) = grad_cov.segment(0, num_cov_pars_optim);
				}
				if (has_covariates) {
					vec_t grad_beta;
					re_model_templ_->CalcGradLinCoef(cov_pars[0], beta, grad_beta, fixed_effects_ptr);
					gradient.segment(num_cov_pars_optim, num_covariates) = grad_beta;
				}
				if (re_model_templ_->EstimateAuxPars()) {
					gradient.segment(num_cov_pars_optim + num_covariates, num_aux_pars) = grad_cov.segment(num_cov_pars_optim, num_aux_pars);
				}
				// Check for NA or Inf
				if (re_model_templ_->GetLikelihood() != "gaussian") {
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
			}//end calc_gradient
			return neg_log_likelihood;
		}

		/*!
		* \brief Set the iteration number in re_model_templ_ (e.g. for correlation-based neighbor selection in Vecchia approximations)
		* \param iter iteration number
		*/
		void SetNumIter(int iter) {
			re_model_templ_->SetNumIter(iter);
		}

		/*!
		* \brief Indicates whether correlation-based nearest neighbors for Vecchia approximation should be updated
		* \return True, if nearest neighbors have been redetermined
		*/
		bool ShouldRedetermineNearestNeighborsVecchia() {
			return(re_model_templ_->ShouldRedetermineNearestNeighborsVecchia());
		}

		/*!
		* \brief Redetermine correlation-based nearest neighbors for Vecchia approximation
		*/
		void RedetermineNearestNeighborsVecchia() {
			re_model_templ_->RedetermineNearestNeighborsVecchia();
		}

		/*!
		* \brief Indicates whether covariance and auxiliary parameters should be estimated
		* \return True, if covariance and auxiliary parameters should be estimated
		*/
		bool LearnCovarianceParameters() {
			return(learn_cov_aux_pars_);
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
			bool has_covariates = re_model_templ_->HasCovariates();
			// Determine number of covariance and linear regression coefficient parameters
			int num_cov_pars_optim = 0, num_covariates = 0, num_aux_pars = 0;
			if (learn_cov_aux_pars_) {
				num_cov_pars_optim = re_model_templ_->GetNumCovPar();
				if (profile_out_marginal_variance_) {
					num_cov_pars_optim -= 1;
				}
				if (re_model_templ_->EstimateAuxPars()) {
					num_aux_pars = re_model_templ_->NumAuxPars();
				}
			}
			if (has_covariates) {
				num_covariates = re_model_templ_->GetNumCoef();
			}
			CHECK((int)pars.size() == num_cov_pars_optim + num_covariates + num_aux_pars);
			// Extract covariance parameters, regression coefficients, and additional likelihood parameters from pars vector
			if (learn_cov_aux_pars_) {
				if (profile_out_marginal_variance_) {
					cov_pars = vec_t(num_cov_pars_optim + 1);
					cov_pars[0] = re_model_templ_->Sigma2();//nugget effect
					cov_pars.segment(1, num_cov_pars_optim) = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
				}
				else {
					cov_pars = pars.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
				}
				if (re_model_templ_->EstimateAuxPars()) {
					aux_pars = pars.segment(num_cov_pars_optim + num_covariates, num_aux_pars).array().exp().matrix();
					aux_pars_ptr = aux_pars.data();
				}
			}
			else {
				cov_pars = cov_pars_;
				aux_pars_ptr = re_model_templ_->GetAuxPars();
			}
			if (has_covariates) {
				beta = pars.segment(num_cov_pars_optim, num_covariates);
			}
			Log::REDebug("GPModel: parameters after optimization iteration number %d: ", iter);
			re_model_templ_->PrintTraceParameters(cov_pars, beta, aux_pars_ptr);
			if (re_model_templ_->GetLikelihood() == "gaussian") {
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
			bool has_covariates = re_model_templ_->HasCovariates();
			// Determine number of covariance and linear regression coefficient parameters
			int num_cov_pars_optim = 0, num_covariates = 0, num_aux_pars = 0;
			if (learn_cov_aux_pars_) {
				num_cov_pars_optim = re_model_templ_->GetNumCovPar();
				if (profile_out_marginal_variance_) {
					num_cov_pars_optim -= 1;
				}
				if (re_model_templ_->EstimateAuxPars()) {
					num_aux_pars = re_model_templ_->NumAuxPars();
				}
			}
			if (has_covariates) {
				num_covariates = re_model_templ_->GetNumCoef();
			}
			CHECK((int)pars.size() == num_cov_pars_optim + num_covariates + num_aux_pars);
			CHECK((int)neg_step_dir.size() == num_cov_pars_optim + num_covariates + num_aux_pars);
			double max_lr = 1e99;
			if (learn_cov_aux_pars_) {
				vec_t neg_step_dir_cov_aux_pars(num_cov_pars_optim + num_aux_pars);
				neg_step_dir_cov_aux_pars.segment(0, num_cov_pars_optim) = neg_step_dir.segment(0, num_cov_pars_optim);
				if (re_model_templ_->EstimateAuxPars()) {
					neg_step_dir_cov_aux_pars.segment(num_cov_pars_optim, num_aux_pars) = neg_step_dir.segment(num_cov_pars_optim + num_covariates, num_aux_pars);
				}
				max_lr = re_model_templ_->MaximalLearningRateCovAuxPars(neg_step_dir_cov_aux_pars);
			}
			if (has_covariates) {
				vec_t beta = pars.segment(num_cov_pars_optim, num_covariates);
				vec_t neg_step_dir_beta = neg_step_dir.segment(num_cov_pars_optim, num_covariates);
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
	* \param profile_out_marginal_variance If true, the error variance sigma is profiled out (=use closed-form expression for error / nugget variance)
	* \param[out] neg_log_likelihood Value of negative log-likelihood or approximate marginal negative log-likelihood for non-Gaussian likelihoods
	* \param num_cov_par Number of covariance parameters
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
		bool learn_cov_aux_pars,
		string_t optimizer,
		bool profile_out_marginal_variance,
		double& neg_log_likelihood,
		int num_cov_par,
		int nb_aux_pars,
		const double* aux_pars,
		bool has_covariates) {
		// Some checks
		if (re_model_templ->EstimateAuxPars()) {
			CHECK(num_cov_par + nb_aux_pars == (int)cov_pars.size());
		}
		else {
			CHECK(num_cov_par == (int)cov_pars.size());
		}
		//if (has_covariates) {
		//	CHECK(beta.size() == X_.cols());
		//}
		// Determine number of covariance and linear regression coefficient parameters
		int num_cov_pars_optim = 0, num_covariates = 0, num_aux_pars = 0;
		if (learn_cov_aux_pars) {
			num_cov_pars_optim = num_cov_par;
			if (profile_out_marginal_variance) {
				num_cov_pars_optim = num_cov_par - 1;
			}
			if (re_model_templ->EstimateAuxPars()) {
				num_aux_pars = nb_aux_pars;
			}
		}
		if (has_covariates) {
			num_covariates = (int)beta.size();
		}
		// Initialization of parameters
		vec_t pars_init(num_cov_pars_optim + num_covariates + num_aux_pars);//order of parameters: 1. cov_pars, 2. coefs, 3. aux_pars
		if (learn_cov_aux_pars) {
			if (profile_out_marginal_variance) {
				pars_init.segment(0, num_cov_pars_optim) = cov_pars.segment(1, num_cov_pars_optim).array().log().matrix();//exclude nugget and transform to log-scale
			}
			else {
				pars_init.segment(0, num_cov_pars_optim) = cov_pars.segment(0, num_cov_pars_optim).array().log().matrix();//transform to log-scale
			}
			if (re_model_templ->EstimateAuxPars()) {
				for (int i = 0; i < num_aux_pars; ++i) {
					pars_init[num_cov_pars_optim + num_covariates + i] = std::log(aux_pars[i]);//transform to log-scale
				}
			}
		}
		if (has_covariates) {
			pars_init.segment(num_cov_pars_optim, num_covariates) = beta;//regresion coefficients
		}
		//Do optimization optimizer == "bfgs_v2"
		optim::algo_settings_t settings;
		settings.iter_max = max_iter;
		OptDataOptimLib<T_mat, T_chol> 	opt_data = OptDataOptimLib<T_mat, T_chol>(re_model_templ, fixed_effects, learn_cov_aux_pars,
			cov_pars.segment(0, num_cov_par), profile_out_marginal_variance, &settings, optimizer);
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
		else if (optimizer == "bfgs_v2") {
			LBFGSpp::LBFGSParam<double> param_LBFGSpp;
			param_LBFGSpp.max_iterations = max_iter;
			param_LBFGSpp.past = 1;//convergence should be determined by checking the change in the obejctive function and not the norm of the gradient
			param_LBFGSpp.delta = delta_rel_conv;
			param_LBFGSpp.epsilon = 1e-10;
			LBFGSpp::LBFGSSolver<double> solver(param_LBFGSpp);
			EvalLLforLBFGSpp<T_mat, T_chol> ll_fun(re_model_templ, fixed_effects, learn_cov_aux_pars,
				cov_pars.segment(0, num_cov_par), profile_out_marginal_variance);
			num_it = solver.minimize(ll_fun, pars_init, neg_log_likelihood);
		}
		//else if (optimizer == "adadelta") {// adadelta currently not supported as default settings do not always work
		//	settings.gd_settings.method = 5;
		//	optim::gd(pars_init, EvalLLforOptimLib<T_mat, T_chol>, &opt_data, settings);
		//}
		if (optimizer != "bfgs_v2") {
			num_it = (int)settings.opt_iter;
			neg_log_likelihood = settings.opt_fn_value;
		}
		// Transform parameters back for export
		if (learn_cov_aux_pars) {
			if (profile_out_marginal_variance) {
				cov_pars[0] = re_model_templ->Sigma2();
				cov_pars.segment(1, num_cov_par - 1) = pars_init.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			else {
				cov_pars.segment(0, num_cov_par) = pars_init.segment(0, num_cov_pars_optim).array().exp().matrix();//back-transform to original scale
			}
			if (re_model_templ->EstimateAuxPars()) {
				for (int i = 0; i < num_aux_pars; ++i) {
					cov_pars[num_cov_par + i] = std::exp(pars_init[num_cov_pars_optim + num_covariates + i]);//back-transform to original scale
				}
			}
		}
		if (has_covariates) {
			beta = pars_init.segment(num_cov_pars_optim, num_covariates);
		}
	}//end OptimExternal

}

#endif   // GPB_OPTIMLIB_UTILS_H_