/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/re_model.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;
using LightGBM::LogLevelRE;
#include <LightGBM/meta.h>
using LightGBM::label_t;

namespace GPBoost {

	REModel::REModel() {
	}

	REModel::REModel(data_size_t num_data,
		const data_size_t* cluster_ids_data,
		const char* re_group_data,
		data_size_t num_re_group,
		const double* re_group_rand_coef_data,
		const data_size_t* ind_effect_group_rand_coef,
		data_size_t num_re_group_rand_coef,
		const int* drop_intercept_group_rand_effect,
		data_size_t num_gp,
		const double* gp_coords_data,
		int dim_gp_coords,
		const double* gp_rand_coef_data,
		data_size_t num_gp_rand_coef,
		const char* cov_fct,
		double cov_fct_shape,
		const char* gp_approx,
		double cov_fct_taper_range,
		double cov_fct_taper_shape,
		int num_neighbors,
		const char* vecchia_ordering,
		int num_ind_points,
		const char* likelihood,
		const char* matrix_inversion_method,
		int seed) {
		string_t cov_fct_str = "none";
		if (cov_fct != nullptr) {
			cov_fct_str = std::string(cov_fct);
		}
		string_t gp_approx_str = "none";
		if (gp_approx != nullptr) {
			gp_approx_str = std::string(gp_approx);
		}
		string_t matrix_inversion_method_str = "cholesky";
		if (matrix_inversion_method != nullptr) {
			matrix_inversion_method_str = std::string(matrix_inversion_method);
		}
		bool use_sparse_matrices = (num_gp + num_gp_rand_coef) == 0 || (COMPACT_SUPPORT_COVS_.find(cov_fct_str) != COMPACT_SUPPORT_COVS_.end()) || 
			gp_approx_str == "tapering";
		if (use_sparse_matrices) {
			if (matrix_inversion_method_str == "iterative") {
				matrix_format_ = "sp_mat_rm_t";
			}
			else {
				matrix_format_ = "sp_mat_t";
			}
		}
		else {
			matrix_format_ = "den_mat_t";
		}
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_ = std::unique_ptr<REModelTemplate<sp_mat_t, chol_sp_mat_t>>(new REModelTemplate<sp_mat_t, chol_sp_mat_t>(
				num_data,
				cluster_ids_data,
				re_group_data,
				num_re_group,
				re_group_rand_coef_data,
				ind_effect_group_rand_coef,
				num_re_group_rand_coef,
				drop_intercept_group_rand_effect,
				num_gp,
				gp_coords_data,
				dim_gp_coords,
				gp_rand_coef_data,
				num_gp_rand_coef,
				cov_fct,
				cov_fct_shape,
				gp_approx,
				cov_fct_taper_range,
				cov_fct_taper_shape,
				num_neighbors, 
				vecchia_ordering,
				num_ind_points,
				likelihood,
				matrix_inversion_method,
				seed));
			num_cov_pars_ = re_model_sp_->num_cov_par_;
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_ = std::unique_ptr<REModelTemplate<sp_mat_rm_t, chol_sp_mat_rm_t>>(new REModelTemplate<sp_mat_rm_t, chol_sp_mat_rm_t>(
				num_data,
				cluster_ids_data,
				re_group_data,
				num_re_group,
				re_group_rand_coef_data,
				ind_effect_group_rand_coef,
				num_re_group_rand_coef,
				drop_intercept_group_rand_effect,
				num_gp,
				gp_coords_data,
				dim_gp_coords,
				gp_rand_coef_data,
				num_gp_rand_coef,
				cov_fct,
				cov_fct_shape,
				gp_approx,
				cov_fct_taper_range,
				cov_fct_taper_shape,
				num_neighbors,
				vecchia_ordering,
				num_ind_points,
				likelihood,
				matrix_inversion_method,
				seed));
			num_cov_pars_ = re_model_sp_rm_->num_cov_par_;
		}
		else {
			re_model_den_ = std::unique_ptr <REModelTemplate< den_mat_t, chol_den_mat_t>>(new REModelTemplate<den_mat_t, chol_den_mat_t>(
				num_data,
				cluster_ids_data,
				re_group_data,
				num_re_group,
				re_group_rand_coef_data,
				ind_effect_group_rand_coef,
				num_re_group_rand_coef,
				drop_intercept_group_rand_effect,
				num_gp,
				gp_coords_data,
				dim_gp_coords,
				gp_rand_coef_data,
				num_gp_rand_coef,
				cov_fct,
				cov_fct_shape,
				gp_approx,
				cov_fct_taper_range,
				cov_fct_taper_shape,
				num_neighbors,
				vecchia_ordering,
				num_ind_points,
				likelihood,
				matrix_inversion_method,
				seed));
			num_cov_pars_ = re_model_den_->num_cov_par_;
		}
	}

	/*! \brief Destructor */
	REModel::~REModel() {
	}

	bool REModel::GaussLikelihood() const {
		if (matrix_format_ == "sp_mat_t") {
			return(re_model_sp_->gauss_likelihood_);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			return(re_model_sp_rm_->gauss_likelihood_);
		}
		else {
			return(re_model_den_->gauss_likelihood_);
		}
	}

	string_t REModel::GetLikelihood() const {
		if (matrix_format_ == "sp_mat_t") {
			return(re_model_sp_->GetLikelihood());
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			return(re_model_sp_rm_->GetLikelihood());
		}
		else {
			return(re_model_den_->GetLikelihood());
		}
	}

	void REModel::SetLikelihood(const string_t& likelihood) {
		if (model_has_been_estimated_) {
			if (GetLikelihood() != likelihood) {
				Log::REFatal("Cannot change likelihood after a model has been estimated ");
			}
		}
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->SetLikelihood(likelihood);
			num_cov_pars_ = re_model_sp_->num_cov_par_;
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->SetLikelihood(likelihood);
			num_cov_pars_ = re_model_sp_rm_->num_cov_par_;
		}
		else {
			re_model_den_->SetLikelihood(likelihood);
			num_cov_pars_ = re_model_den_->num_cov_par_;
		}
	}

	string_t REModel::GetOptimizerCovPars() const {
		if (matrix_format_ == "sp_mat_t") {
			return(re_model_sp_->optimizer_cov_pars_);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			return(re_model_sp_rm_->optimizer_cov_pars_);
		}
		else {
			return(re_model_den_->optimizer_cov_pars_);
		}
	}

	string_t REModel::GetOptimizerCoef() const {
		if (matrix_format_ == "sp_mat_t") {
			return(re_model_sp_->optimizer_coef_);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			return(re_model_sp_rm_->optimizer_coef_);
		}
		else {
			return(re_model_den_->optimizer_coef_);
		}
	}

	string_t REModel::GetCGPreconditionerType() const {
		if (matrix_format_ == "sp_mat_t") {
			return(re_model_sp_->cg_preconditioner_type_);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			return(re_model_sp_rm_->cg_preconditioner_type_);
		}
		else {
			return(re_model_den_->cg_preconditioner_type_);
		}
	}

	void REModel::SetOptimConfig(double* init_cov_pars,
		double lr,
		double acc_rate_cov,
		int max_iter,
		double delta_rel_conv,
		bool use_nesterov_acc,
		int nesterov_schedule_version,
		bool trace,
		const char* optimizer,
		int momentum_offset,
		const char* convergence_criterion,
		bool calc_std_dev, 
		int num_covariates,
		double* init_coef,
		double lr_coef,
		double acc_rate_coef,
		const char* optimizer_coef,
		int cg_max_num_it,
		int cg_max_num_it_tridiag,
		double cg_delta_conv,
		int num_rand_vec_trace,
		bool reuse_rand_vec_trace,
		const char* cg_preconditioner_type,
		int seed_rand_vec_trace,
		int piv_chol_rank,
		double* init_aux_pars,
		bool estimate_aux_pars) {
		// Initial covariance parameters
		if (init_cov_pars != nullptr) {
			vec_t init_cov_pars_orig = Eigen::Map<const vec_t>(init_cov_pars, num_cov_pars_);
			init_cov_pars_ = vec_t(num_cov_pars_);
			if (matrix_format_ == "sp_mat_t") {
				re_model_sp_->TransformCovPars(init_cov_pars_orig, init_cov_pars_);
			}
			else if (matrix_format_ == "sp_mat_rm_t") {
				re_model_sp_rm_->TransformCovPars(init_cov_pars_orig, init_cov_pars_);
			}
			else {
				re_model_den_->TransformCovPars(init_cov_pars_orig, init_cov_pars_);
			}
			cov_pars_ = init_cov_pars_;
			cov_pars_initialized_ = true;
			init_cov_pars_provided_ = true;
			covariance_matrix_has_been_factorized_ = false;
		}
		// Initial linear regression coefficients
		if (init_coef != nullptr) {
			coef_ = Eigen::Map<const vec_t>(init_coef, num_covariates);
			init_coef_given_ = true;
			coef_given_or_estimated_ = true;
		}
		else {
			init_coef_given_ = false;
		}
		// Initial aux_pars
		if (init_aux_pars != nullptr) {
			init_aux_pars_ = Eigen::Map<const vec_t>(init_aux_pars, NumAuxPars());
			SetAuxPars(init_aux_pars);
			init_aux_pars_given_ = true;
		}
		else {
			init_aux_pars_given_ = false;
		}
		// Logging level
		if (trace) {
			Log::ResetLogLevelRE(LogLevelRE::Debug);
		}
		else {
			Log::ResetLogLevelRE(LogLevelRE::Info);
		}
		calc_std_dev_ = calc_std_dev;
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->SetOptimConfig(lr, acc_rate_cov, max_iter, delta_rel_conv, use_nesterov_acc, nesterov_schedule_version,
				optimizer, momentum_offset, convergence_criterion, lr_coef, acc_rate_coef, optimizer_coef,
				cg_max_num_it, cg_max_num_it_tridiag, cg_delta_conv, num_rand_vec_trace, reuse_rand_vec_trace,
				cg_preconditioner_type, seed_rand_vec_trace, piv_chol_rank, estimate_aux_pars);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->SetOptimConfig(lr, acc_rate_cov, max_iter, delta_rel_conv, use_nesterov_acc, nesterov_schedule_version,
				optimizer, momentum_offset, convergence_criterion, lr_coef, acc_rate_coef, optimizer_coef,
				cg_max_num_it, cg_max_num_it_tridiag, cg_delta_conv, num_rand_vec_trace, reuse_rand_vec_trace,
				cg_preconditioner_type, seed_rand_vec_trace, piv_chol_rank, estimate_aux_pars);
		}
		else {
			re_model_den_->SetOptimConfig(lr, acc_rate_cov, max_iter, delta_rel_conv, use_nesterov_acc, nesterov_schedule_version,
				optimizer, momentum_offset, convergence_criterion, lr_coef, acc_rate_coef, optimizer_coef,
				cg_max_num_it, cg_max_num_it_tridiag, cg_delta_conv, num_rand_vec_trace, reuse_rand_vec_trace,
				cg_preconditioner_type, seed_rand_vec_trace, piv_chol_rank, estimate_aux_pars);
		}
	}

	void REModel::ResetCovPars() {
		cov_pars_ = vec_t(num_cov_pars_);
		cov_pars_initialized_ = false;
	}

	void REModel::OptimCovPar(const double* y_data,
		const double* fixed_effects,
		bool called_in_GPBoost_algorithm) {
		if (y_data != nullptr) {
			InitializeCovParsIfNotDefined(y_data);
			// Note: y_data can be null_ptr for non-Gaussian data. For non-Gaussian data, the function 'InitializeCovParsIfNotDefined' is called in 'SetY'
		}
		CHECK(cov_pars_initialized_);
		double* std_dev_cov_par;
		if (calc_std_dev_) {
			std_dev_cov_pars_ = vec_t(num_cov_pars_);
			std_dev_cov_par = std_dev_cov_pars_.data();
		}
		else {
			std_dev_cov_par = nullptr;
		}
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->OptimLinRegrCoefCovPar(y_data,
				nullptr,
				0,
				cov_pars_.data(),
				nullptr,
				num_it_,
				cov_pars_.data(),
				nullptr,
				std_dev_cov_par,
				nullptr,
				calc_std_dev_,
				fixed_effects,
				true,
				called_in_GPBoost_algorithm);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->OptimLinRegrCoefCovPar(y_data,
				nullptr,
				0,
				cov_pars_.data(),
				nullptr,
				num_it_,
				cov_pars_.data(),
				nullptr,
				std_dev_cov_par,
				nullptr,
				calc_std_dev_,
				fixed_effects,
				true,
				called_in_GPBoost_algorithm);
		}
		else {
			re_model_den_->OptimLinRegrCoefCovPar(y_data,
				nullptr,
				0,
				cov_pars_.data(),
				nullptr,
				num_it_,
				cov_pars_.data(),
				nullptr,
				std_dev_cov_par,
				nullptr,
				calc_std_dev_,
				fixed_effects,
				true,
				called_in_GPBoost_algorithm);
		}
		has_covariates_ = false;
		covariance_matrix_has_been_factorized_ = true;
		model_has_been_estimated_ = true;
	}

	void REModel::OptimLinRegrCoefCovPar(const double* y_data,
		const double* covariate_data,
		int num_covariates) {
		InitializeCovParsIfNotDefined(y_data);
		double* coef_ptr;;
		if (init_coef_given_) {
			coef_ptr = coef_.data();
		}
		else {
			coef_ptr = nullptr;
			coef_ = vec_t(num_covariates);
		}
		double* std_dev_cov_par;
		double* std_dev_coef;
		if (calc_std_dev_) {
			std_dev_cov_pars_ = vec_t(num_cov_pars_);
			std_dev_cov_par = std_dev_cov_pars_.data();
			std_dev_coef_ = vec_t(num_covariates);
			std_dev_coef = std_dev_coef_.data();
		}
		else {
			std_dev_cov_par = nullptr;
			std_dev_coef = nullptr;
		}
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->OptimLinRegrCoefCovPar(y_data,
				covariate_data,
				num_covariates,
				cov_pars_.data(),
				coef_.data(),
				num_it_,
				cov_pars_.data(),
				coef_ptr,
				std_dev_cov_par,
				std_dev_coef,
				calc_std_dev_,
				nullptr,
				true,
				false);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->OptimLinRegrCoefCovPar(y_data,
				covariate_data,
				num_covariates,
				cov_pars_.data(),
				coef_.data(),
				num_it_,
				cov_pars_.data(),
				coef_ptr,
				std_dev_cov_par,
				std_dev_coef,
				calc_std_dev_,
				nullptr,
				true,
				false);
		}
		else {
			re_model_den_->OptimLinRegrCoefCovPar(y_data,
				covariate_data,
				num_covariates,
				cov_pars_.data(),
				coef_.data(),
				num_it_,
				cov_pars_.data(),
				coef_ptr,
				std_dev_cov_par,
				std_dev_coef,
				calc_std_dev_,
				nullptr,
				true,
				false);
		}
		has_covariates_ = true;
		coef_given_or_estimated_ = true;
		covariance_matrix_has_been_factorized_ = true;
		model_has_been_estimated_ = true;
	}

	void REModel::FindInitialValueBoosting(double* init_score) {
		CHECK(cov_pars_initialized_);
		vec_t covariate_data(GetNumData());
		covariate_data.setOnes();
		init_score[0] = 0.;
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->OptimLinRegrCoefCovPar(nullptr,
				covariate_data.data(),
				1,
				cov_pars_.data(),
				init_score,
				num_it_,
				cov_pars_.data(),
				init_score,
				nullptr,
				nullptr,
				false,
				nullptr,
				false,//learn_covariance_parameters=false
				true);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->OptimLinRegrCoefCovPar(nullptr,
				covariate_data.data(),
				1,
				cov_pars_.data(),
				init_score,
				num_it_,
				cov_pars_.data(),
				init_score,
				nullptr,
				nullptr,
				false,
				nullptr,
				false,//learn_covariance_parameters=false
				true);
		}
		else {
			re_model_den_->OptimLinRegrCoefCovPar(nullptr,
				covariate_data.data(),
				1,
				cov_pars_.data(),
				init_score,
				num_it_,
				cov_pars_.data(),
				init_score,
				nullptr,
				nullptr,
				false,
				nullptr,
				false,//learn_covariance_parameters=false
				true);
		}
	}

	void REModel::EvalNegLogLikelihood(const double* y_data,
		double* cov_pars,
		double& negll,
		const double* fixed_effects,
		bool InitializeModeCovMat,
		bool CalcModePostRandEff_already_done) {
		vec_t cov_pars_trafo;
		if (cov_pars == nullptr) {
			if (y_data != nullptr) {
				InitializeCovParsIfNotDefined(y_data);
				// Note: y_data can be null_ptr for non-Gaussian data. For non-Gaussian data, the function 'InitializeCovParsIfNotDefined' is called in 'SetY'
			}
			CHECK(cov_pars_initialized_);
			cov_pars_trafo = cov_pars_;
		}
		else {
			vec_t cov_pars_orig = Eigen::Map<const vec_t>(cov_pars, num_cov_pars_);
			cov_pars_trafo = vec_t(num_cov_pars_);
			if (matrix_format_ == "sp_mat_t") {
				re_model_sp_->TransformCovPars(cov_pars_orig, cov_pars_trafo);
			}
			else if (matrix_format_ == "sp_mat_rm_t") {
				re_model_sp_rm_->TransformCovPars(cov_pars_orig, cov_pars_trafo);
			}
			else {
				re_model_den_->TransformCovPars(cov_pars_orig, cov_pars_trafo);
			}
		}

		if (matrix_format_ == "sp_mat_t") {
			if (re_model_sp_->gauss_likelihood_) {
				re_model_sp_->EvalNegLogLikelihood(y_data, cov_pars_trafo.data(), fixed_effects, 
					negll, false, false, false);
			}
			else {
				re_model_sp_->EvalLaplaceApproxNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, 
					fixed_effects, InitializeModeCovMat, CalcModePostRandEff_already_done);
			}
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			if (re_model_sp_rm_->gauss_likelihood_) {
				re_model_sp_rm_->EvalNegLogLikelihood(y_data, cov_pars_trafo.data(), fixed_effects, 
					negll, false, false, false);
			}
			else {
				re_model_sp_rm_->EvalLaplaceApproxNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, 
					fixed_effects, InitializeModeCovMat, CalcModePostRandEff_already_done);
			}
		}
		else {
			if (re_model_den_->gauss_likelihood_) {
				re_model_den_->EvalNegLogLikelihood(y_data, cov_pars_trafo.data(), fixed_effects, 
					negll, false, false, false);
			}
			else {
				re_model_den_->EvalLaplaceApproxNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, 
					fixed_effects, InitializeModeCovMat, CalcModePostRandEff_already_done);
			}
		}
		covariance_matrix_has_been_factorized_ = false;
		//set to false as otherwise the covariance is not factorized for prediction for Gaussian data and this can lead to problems 
		//(e.g. fitting model with covariates, then calling likelihood without covariates, then making prediction with covariates)
	}

	void REModel::GetCurrentNegLogLikelihood(double& negll) {
		if (matrix_format_ == "sp_mat_t") {
			negll = re_model_sp_->neg_log_likelihood_;
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			negll = re_model_sp_rm_->neg_log_likelihood_;
		}
		else {
			negll = re_model_den_->neg_log_likelihood_;
		}
	}

	void REModel::CalcGradient(double* y, const double* fixed_effects, bool calc_cov_factor) {
		if (y != nullptr) {
			InitializeCovParsIfNotDefined(y);
		}
		CHECK(cov_pars_initialized_);
		if (matrix_format_ == "sp_mat_t") {
			//1. Factorize covariance matrix
			if (calc_cov_factor) {
				re_model_sp_->SetCovParsComps(cov_pars_);
				if (re_model_sp_->gauss_likelihood_) {//Gaussian data
					re_model_sp_->CalcCovFactor(false, true, 1., false);
				}
				else {//not gauss_likelihood_
					if (re_model_sp_->gp_approx_ == "vecchia") {
						re_model_sp_->CalcCovFactor(false, true, 1., false);
					}
					else {
						re_model_sp_->CalcSigmaComps();
						re_model_sp_->CalcCovMatrixNonGauss();
					}
					re_model_sp_->CalcModePostRandEffCalcMLL(fixed_effects, true);
				}//end gauss_likelihood_
			}//end calc_cov_factor
			//2. Calculate gradient
			if (re_model_sp_->gauss_likelihood_) {//Gaussian data
				re_model_sp_->SetY(y);
				re_model_sp_->CalcYAux(cov_pars_[0]);
				re_model_sp_->GetYAux(y);
			}
			else {//not gauss_likelihood_
				re_model_sp_->CalcGradFLaplace(y, fixed_effects);
			}
		}//end matrix_format_ == "sp_mat_t"
		else if (matrix_format_ == "sp_mat_rm_t") {
			//1. Factorize covariance matrix
			if (calc_cov_factor) {
				re_model_sp_rm_->SetCovParsComps(cov_pars_);
				if (re_model_sp_rm_->gauss_likelihood_) {//Gaussian data
					re_model_sp_rm_->CalcCovFactor(false, true, 1., false);
				}
				else {//not gauss_likelihood_
					if (re_model_sp_rm_->gp_approx_ == "vecchia") {
						re_model_sp_rm_->CalcCovFactor(false, true, 1., false);
					}
					else {
						re_model_sp_rm_->CalcSigmaComps();
						re_model_sp_rm_->CalcCovMatrixNonGauss();
					}
					re_model_sp_rm_->CalcModePostRandEffCalcMLL(fixed_effects, true);
				}//end gauss_likelihood_
			}//end calc_cov_factor
			//2. Calculate gradient
			if (re_model_sp_rm_->gauss_likelihood_) {//Gaussian data
				re_model_sp_rm_->SetY(y);
				re_model_sp_rm_->CalcYAux(cov_pars_[0]);
				re_model_sp_rm_->GetYAux(y);
			}
			else {//not gauss_likelihood_
				re_model_sp_rm_->CalcGradFLaplace(y, fixed_effects);
			}
		}//end matrix_format_ == "sp_mat_rm_t"
		else {//matrix_format_ == "den_mat_t"
			//1. Factorize covariance matrix
			if (calc_cov_factor) {
				re_model_den_->SetCovParsComps(cov_pars_);
				if (re_model_den_->gauss_likelihood_) {//Gaussian data
					re_model_den_->CalcCovFactor(false, true, 1., false);
				}
				else {//not gauss_likelihood_
					if (re_model_den_->gp_approx_ == "vecchia") {
						re_model_den_->CalcCovFactor(false, true, 1., false);
					}
					else {
						re_model_den_->CalcSigmaComps();
						re_model_den_->CalcCovMatrixNonGauss();
					}
					re_model_den_->CalcModePostRandEffCalcMLL(fixed_effects, true);
				}//end gauss_likelihood_
			}//end calc_cov_factor
			//2. Calculate gradient
			if (re_model_den_->gauss_likelihood_) {//Gaussian data
				re_model_den_->SetY(y);
				re_model_den_->CalcYAux(cov_pars_[0]);
				re_model_den_->GetYAux(y);
			}//end gauss_likelihood_
			else {//not gauss_likelihood_
				re_model_den_->CalcGradFLaplace(y, fixed_effects);
			}
		}//end not matrix_format_ == "sp_mat_t"
		if (calc_cov_factor) {
			covariance_matrix_has_been_factorized_ = true;
		}
	}

	void REModel::SetY(const double* y) const {
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->SetY(y);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->SetY(y);
		}
		else {
			re_model_den_->SetY(y);
		}
	}

	void REModel::SetY(const float* y) const {
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->SetY(y);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->SetY(y);
		}
		else {
			re_model_den_->SetY(y);
		}
	}

	void REModel::GetY(double* y) const {
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->GetY(y);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->GetY(y);
		}
		else {
			re_model_den_->GetY(y);
		}
	}

	void REModel::GetCovariateData(double* covariate_data) const {
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->GetCovariateData(covariate_data);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->GetCovariateData(covariate_data);
		}
		else {
			re_model_den_->GetCovariateData(covariate_data);
		}
	}

	void REModel::GetCovPar(double* cov_par, bool calc_std_dev) const {
		if (cov_pars_.size() == 0) {
			Log::REFatal("Covariance parameters have not been estimated or set");
		}
		//Transform covariance paramters back to original scale
		vec_t cov_pars_orig(num_cov_pars_);
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->TransformBackCovPars(cov_pars_, cov_pars_orig);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->TransformBackCovPars(cov_pars_, cov_pars_orig);
		}
		else {
			re_model_den_->TransformBackCovPars(cov_pars_, cov_pars_orig);
		}
		for (int j = 0; j < num_cov_pars_; ++j) {
			cov_par[j] = cov_pars_orig[j];
		}
		if (calc_std_dev) {
			for (int j = 0; j < num_cov_pars_; ++j) {
				cov_par[j + num_cov_pars_] = std_dev_cov_pars_[j];
			}
		}
	}

	void REModel::GetInitCovPar(double* init_cov_par) const {
		vec_t init_cov_pars_orig(num_cov_pars_);
		if (init_cov_pars_provided_ || cov_pars_initialized_) {
			if (matrix_format_ == "sp_mat_t") {
				re_model_sp_->TransformBackCovPars(init_cov_pars_, init_cov_pars_orig);
			}
			else if (matrix_format_ == "sp_mat_rm_t") {
				re_model_sp_rm_->TransformBackCovPars(init_cov_pars_, init_cov_pars_orig);
			}
			else {
				re_model_den_->TransformBackCovPars(init_cov_pars_, init_cov_pars_orig);
			}
			for (int j = 0; j < num_cov_pars_; ++j) {
				init_cov_par[j] = init_cov_pars_orig[j];
			}
		}
		else {
			for (int j = 0; j < num_cov_pars_; ++j) {
				init_cov_par[j] = -1.;
			}
		}
	}

	void REModel::GetCoef(double* coef, bool calc_std_dev) const {
		int num_coef = (int)coef_.size();
		for (int j = 0; j < num_coef; ++j) {
			coef[j] = coef_[j];
		}
		if (calc_std_dev) {
			for (int j = 0; j < num_coef; ++j) {
				coef[j + num_coef] = std_dev_coef_[j];
			}
		}
	}

	void REModel::SetPredictionData(data_size_t num_data_pred,
		const data_size_t* cluster_ids_data_pred,
		const char* re_group_data_pred,
		const double* re_group_rand_coef_data_pred,
		double* gp_coords_data_pred,
		const double* gp_rand_coef_data_pred,
		const double* covariate_data_pred,
		const char* vecchia_pred_type,
		int num_neighbors_pred,
		double cg_delta_conv_pred,
		int nsim_var_pred,
		int rank_pred_approx_matrix_lanczos) {
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->SetPredictionData(num_data_pred,
				cluster_ids_data_pred,
				re_group_data_pred,
				re_group_rand_coef_data_pred,
				gp_coords_data_pred,
				gp_rand_coef_data_pred,
				covariate_data_pred,
				vecchia_pred_type,
				num_neighbors_pred,
				cg_delta_conv_pred,
				nsim_var_pred,
				rank_pred_approx_matrix_lanczos);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->SetPredictionData(num_data_pred,
				cluster_ids_data_pred,
				re_group_data_pred,
				re_group_rand_coef_data_pred,
				gp_coords_data_pred,
				gp_rand_coef_data_pred,
				covariate_data_pred,
				vecchia_pred_type,
				num_neighbors_pred,
				cg_delta_conv_pred,
				nsim_var_pred,
				rank_pred_approx_matrix_lanczos);
		}
		else {
			re_model_den_->SetPredictionData(num_data_pred,
				cluster_ids_data_pred,
				re_group_data_pred,
				re_group_rand_coef_data_pred,
				gp_coords_data_pred,
				gp_rand_coef_data_pred,
				covariate_data_pred,
				vecchia_pred_type,
				num_neighbors_pred,
				cg_delta_conv_pred,
				nsim_var_pred,
				rank_pred_approx_matrix_lanczos);
		}
	}

	void REModel::Predict(const double* y_obs,
		data_size_t num_data_pred,
		double* out_predict,
		bool predict_cov_mat,
		bool predict_var,
		bool predict_response,
		const data_size_t* cluster_ids_data_pred,
		const char* re_group_data_pred,
		const double* re_group_rand_coef_data_pred,
		double* gp_coords_data_pred,
		const double* gp_rand_coef_data_pred,
		const double* cov_pars_pred,
		const double* covariate_data_pred,
		bool use_saved_data,
		const double* fixed_effects,
		const double* fixed_effects_pred,
		bool suppress_calc_cov_factor) {
		bool calc_cov_factor = true;
		vec_t cov_pars_pred_trans;
		if (cov_pars_pred != nullptr) {
			vec_t cov_pars_pred_orig = Eigen::Map<const vec_t>(cov_pars_pred, num_cov_pars_);
			cov_pars_pred_trans = vec_t(num_cov_pars_);
			if (matrix_format_ == "sp_mat_t") {
				re_model_sp_->TransformCovPars(cov_pars_pred_orig, cov_pars_pred_trans);
			}
			else if (matrix_format_ == "sp_mat_rm_t") {
				re_model_sp_rm_->TransformCovPars(cov_pars_pred_orig, cov_pars_pred_trans);
			}
			else {
				re_model_den_->TransformCovPars(cov_pars_pred_orig, cov_pars_pred_trans);
			}
			cov_pars_have_been_provided_for_prediction_ = true;
		}//end if cov_pars_pred != nullptr
		else {// use saved cov_pars
			if (!cov_pars_initialized_) {
				Log::REFatal("Covariance parameters have not been estimated or are not given.");
			}
			// Note: cov_pars_initialized_ is set to true by InitializeCovParsIfNotDefined() which is called by OptimCovPar(), OptimLinRegrCoefCovPar(), and EvalNegLogLikelihood().
			//			It is assumed that if one of these three functions has been called, the covariance parameters have been estimated
			cov_pars_pred_trans = cov_pars_;
			if (GaussLikelihood()) {
				// We don't factorize the covariance matrix for Gaussian data in case this has already been done (e.g. at the end of the estimation)
				// If cov_pars_have_been_provided_for_prediction_, we redo the factorization since the saved factorization will likely not correspond to the parameters in cov_pars_
				// For non-Gaussian, we always calculate the Laplace approximation to guarantee that all calls to predict() return the same values
				//	Otherwise, there can be (very) small difference between (i) calculating predictions using an estimated a model and
				//	(ii) loading / initializing an empty a model, providing the same covariance paramters, and thus recalculating the mode.
				//	This is due to very small differences in the mode which are found differently: for an estimated model, the mode has been iteratively found during the estimation,
				//	in particular, in the last optimization round the mode has been initialized at the previous value,
				//	where as when the covariance parameters are provided here, the mode is found using these parameters but initilized from 0.
				calc_cov_factor = !covariance_matrix_has_been_factorized_ || cov_pars_have_been_provided_for_prediction_;
			}
		}// end use saved cov_pars
		if (has_covariates_) {
			CHECK(coef_given_or_estimated_ == true);
		}
		if (suppress_calc_cov_factor) {
			calc_cov_factor = false;
		}
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->Predict(cov_pars_pred_trans.data(),
				y_obs,
				num_data_pred,
				out_predict,
				calc_cov_factor,
				predict_cov_mat,
				predict_var,
				predict_response,
				covariate_data_pred,
				coef_.data(),
				cluster_ids_data_pred,
				re_group_data_pred,
				re_group_rand_coef_data_pred,
				gp_coords_data_pred,
				gp_rand_coef_data_pred,
				use_saved_data,
				fixed_effects,
				fixed_effects_pred);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->Predict(cov_pars_pred_trans.data(),
				y_obs,
				num_data_pred,
				out_predict,
				calc_cov_factor,
				predict_cov_mat,
				predict_var,
				predict_response,
				covariate_data_pred,
				coef_.data(),
				cluster_ids_data_pred,
				re_group_data_pred,
				re_group_rand_coef_data_pred,
				gp_coords_data_pred,
				gp_rand_coef_data_pred,
				use_saved_data,
				fixed_effects,
				fixed_effects_pred);
		}
		else {
			re_model_den_->Predict(cov_pars_pred_trans.data(),
				y_obs,
				num_data_pred,
				out_predict,
				calc_cov_factor,
				predict_cov_mat,
				predict_var,
				predict_response,
				covariate_data_pred,
				coef_.data(),
				cluster_ids_data_pred,
				re_group_data_pred,
				re_group_rand_coef_data_pred,
				gp_coords_data_pred,
				gp_rand_coef_data_pred,
				use_saved_data,
				fixed_effects,
				fixed_effects_pred);
		}
	}//end Predict

	void REModel::PredictTrainingDataRandomEffects(const double* cov_pars_pred,
		const double* y_obs,
		double* out_predict,
		const double* fixed_effects,
		bool calc_var) const {
		bool calc_cov_factor = true;
		vec_t cov_pars_pred_trans;
		if (cov_pars_pred != nullptr) {
			vec_t cov_pars_pred_orig = Eigen::Map<const vec_t>(cov_pars_pred, num_cov_pars_);
			cov_pars_pred_trans = vec_t(num_cov_pars_);
			if (matrix_format_ == "sp_mat_t") {
				re_model_sp_->TransformCovPars(cov_pars_pred_orig, cov_pars_pred_trans);
			}
			else if (matrix_format_ == "sp_mat_rm_t") {
				re_model_sp_rm_->TransformCovPars(cov_pars_pred_orig, cov_pars_pred_trans);
			}
			else {
				re_model_den_->TransformCovPars(cov_pars_pred_orig, cov_pars_pred_trans);
			}
		}//end if cov_pars_pred != nullptr
		else {// use saved cov_pars
			if (!cov_pars_initialized_) {
				Log::REFatal("Covariance parameters have not been estimated or are not given.");
			}
			cov_pars_pred_trans = cov_pars_;
			if (GaussLikelihood()) {
				calc_cov_factor = !covariance_matrix_has_been_factorized_;
			}
		}// end use saved cov_pars
		if (has_covariates_) {
			CHECK(coef_given_or_estimated_ == true);
		}
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->PredictTrainingDataRandomEffects(cov_pars_pred_trans.data(),
				coef_.data(),
				y_obs,
				out_predict, 
				calc_cov_factor,
				fixed_effects,
				calc_var);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->PredictTrainingDataRandomEffects(cov_pars_pred_trans.data(),
				coef_.data(),
				y_obs,
				out_predict,
				calc_cov_factor,
				fixed_effects,
				calc_var);
		}
		else {
			re_model_den_->PredictTrainingDataRandomEffects(cov_pars_pred_trans.data(),
				coef_.data(),
				y_obs,
				out_predict,
				calc_cov_factor,
				fixed_effects,
				calc_var);
		}
	}//end PredictTrainingDataRandomEffects

	int REModel::GetNumIt() const {
		return(num_it_);
	}

	int REModel::GetNumData() const {
		if (matrix_format_ == "sp_mat_t") {
			return(re_model_sp_->num_data_);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			return(re_model_sp_rm_->num_data_);
		}
		else {
			return(re_model_den_->num_data_);
		}
	}

	void REModel::NewtonUpdateLeafValues(const int* data_leaf_index,
		const int num_leaves,
		double* leaf_values) const {
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->NewtonUpdateLeafValues(data_leaf_index, num_leaves, leaf_values, cov_pars_[0]);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->NewtonUpdateLeafValues(data_leaf_index, num_leaves, leaf_values, cov_pars_[0]);
		}
		else {
			re_model_den_->NewtonUpdateLeafValues(data_leaf_index, num_leaves, leaf_values, cov_pars_[0]);
		}
	}

	void REModel::InitializeCovParsIfNotDefined(const double* y_data) {
		if (!cov_pars_initialized_) {
			if (init_cov_pars_provided_) {
				cov_pars_ = init_cov_pars_;
			}
			else {
				cov_pars_ = vec_t(num_cov_pars_);
				if (matrix_format_ == "sp_mat_t") {
					re_model_sp_->FindInitCovPar(y_data, cov_pars_.data());
				}
				else if (matrix_format_ == "sp_mat_rm_t") {
					re_model_sp_rm_->FindInitCovPar(y_data, cov_pars_.data());
				}
				else {
					re_model_den_->FindInitCovPar(y_data, cov_pars_.data());
				}
				covariance_matrix_has_been_factorized_ = false;
				init_cov_pars_ = cov_pars_;
			}
			cov_pars_initialized_ = true;
		}
	}

	int REModel::NumAuxPars() const {
		int num_aux_pars;
		if (matrix_format_ == "sp_mat_t") {
			num_aux_pars = re_model_sp_->NumAuxPars();
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			num_aux_pars = re_model_sp_rm_->NumAuxPars();
		}
		else {
			num_aux_pars = re_model_den_->NumAuxPars();
		}
		return num_aux_pars;
	}

	void REModel::GetAuxPars(double* aux_pars,
		string_t& name) const {
		const double* aux_pars_temp;
		if (matrix_format_ == "sp_mat_t") {
			aux_pars_temp = re_model_sp_->GetAuxPars();
			re_model_sp_->GetNameFirstAuxPar(name);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			aux_pars_temp = re_model_sp_rm_->GetAuxPars();
			re_model_sp_rm_->GetNameFirstAuxPar(name);
		}
		else {
			aux_pars_temp = re_model_den_->GetAuxPars();
			re_model_den_->GetNameFirstAuxPar(name);
		}
		for (int j = 0; j < NumAuxPars(); ++j) {
			aux_pars[j] = aux_pars_temp[j];
		}
	}

	void REModel::SetAuxPars(const double* aux_pars) {
		if (matrix_format_ == "sp_mat_t") {
			re_model_sp_->SetAuxPars(aux_pars);
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			re_model_sp_rm_->SetAuxPars(aux_pars);
		}
		else {
			re_model_den_->SetAuxPars(aux_pars);
		}
	}

	void REModel::GetInitAuxPars(double* aux_pars) const {
		vec_t init_cov_pars_orig(num_cov_pars_);
		if (init_aux_pars_given_) {
			for (int j = 0; j < NumAuxPars(); ++j) {
				aux_pars[j] = init_aux_pars_[j];
			}
		}
		else {
			for (int j = 0; j < NumAuxPars(); ++j) {
				aux_pars[j] = -1.;
			}
		}
	}

	/*!
	* \brief Calculate test log-likelihood using adaptive GH quadrature
	* \param y_test Test response variable
	* \param pred_mean Predictive mean of latent random effects
	* \param pred_var Predictive variances of latent random effects
	* \param num_data Number of data points
	*/
	double REModel::TestNegLogLikelihoodAdaptiveGHQuadrature(const label_t* y_test,
		const double* pred_mean,
		const double* pred_var,
		const data_size_t num_data) {
		if (GetLikelihood() == "gaussian") {
			double aux_par = 1. / (std::sqrt(cov_pars_[0]));
			SetAuxPars(&aux_par);
		}
		if (matrix_format_ == "sp_mat_t") {
			return(re_model_sp_->TestNegLogLikelihoodAdaptiveGHQuadrature(y_test, pred_mean, pred_var, num_data));
		}
		else if (matrix_format_ == "sp_mat_rm_t") {
			return(re_model_sp_rm_->TestNegLogLikelihoodAdaptiveGHQuadrature(y_test, pred_mean, pred_var, num_data));
		}
		else {
			return(re_model_den_->TestNegLogLikelihoodAdaptiveGHQuadrature(y_test, pred_mean, pred_var, num_data));
		}
	}

}  // namespace GPBoost
