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

namespace GPBoost {

	REModel::REModel() {
	}

	REModel::REModel(data_size_t num_data,
		const gp_id_t* cluster_ids_data,
		const char* re_group_data,
		data_size_t num_re_group,
		const double* re_group_rand_coef_data,
		const int32_t* ind_effect_group_rand_coef,
		data_size_t num_re_group_rand_coef,
		data_size_t num_gp,
		const double* gp_coords_data,
		int dim_gp_coords,
		const double* gp_rand_coef_data,
		data_size_t num_gp_rand_coef,
		const char* cov_fct,
		double cov_fct_shape,
		double cov_fct_taper_range,
		bool vecchia_approx,
		int num_neighbors,
		const char* vecchia_ordering,
		const char* vecchia_pred_type,
		int num_neighbors_pred,
		const char* likelihood) {
		string_t cov_fct_str = "none";
		if (cov_fct != nullptr) {
			cov_fct_str = std::string(cov_fct);
		}
		bool use_sparse_matrices = (num_gp + num_gp_rand_coef) == 0 || cov_fct_str == "wendland";
		if (use_sparse_matrices) {
			sparse_ = true;
			re_model_sp_ = std::unique_ptr<REModelTemplate<sp_mat_t, chol_sp_mat_t>>(new REModelTemplate<sp_mat_t, chol_sp_mat_t>(
				num_data,
				cluster_ids_data,
				re_group_data,
				num_re_group,
				re_group_rand_coef_data,
				ind_effect_group_rand_coef,
				num_re_group_rand_coef,
				num_gp,
				gp_coords_data,
				dim_gp_coords,
				gp_rand_coef_data,
				num_gp_rand_coef,
				cov_fct,
				cov_fct_shape,
				cov_fct_taper_range,
				vecchia_approx,
				num_neighbors, 
				vecchia_ordering,
				vecchia_pred_type,
				num_neighbors_pred,
				likelihood));
			num_cov_pars_ = re_model_sp_->num_cov_par_;
		}
		else {
			sparse_ = false;
			re_model_den_ = std::unique_ptr <REModelTemplate< den_mat_t, chol_den_mat_t>>(new REModelTemplate<den_mat_t, chol_den_mat_t>(
				num_data,
				cluster_ids_data,
				re_group_data,
				num_re_group,
				re_group_rand_coef_data,
				ind_effect_group_rand_coef,
				num_re_group_rand_coef,
				num_gp,
				gp_coords_data,
				dim_gp_coords,
				gp_rand_coef_data,
				num_gp_rand_coef,
				cov_fct,
				cov_fct_shape,
				cov_fct_taper_range,
				vecchia_approx,
				num_neighbors,
				vecchia_ordering,
				vecchia_pred_type,
				num_neighbors_pred,
				likelihood));
			num_cov_pars_ = re_model_den_->num_cov_par_;
		}
		if (!GaussLikelihood()) {
			optimizer_cov_pars_ = "gradient_descent";
			optimizer_coef_ = "gradient_descent";
		}
	}

	/*! \brief Destructor */
	REModel::~REModel() {
	}

	bool REModel::GaussLikelihood() const {
		if (sparse_) {
			return(re_model_sp_->gauss_likelihood_);
		}
		else {
			return(re_model_den_->gauss_likelihood_);
		}
	}

	string_t REModel::GetLikelihood() const {
		if (sparse_) {
			return(re_model_sp_->GetLikelihood());
		}
		else {
			return(re_model_den_->GetLikelihood());
		}
	}

	void REModel::SetLikelihood(const string_t& likelihood) {
		if (sparse_) {
			re_model_sp_->SetLikelihood(likelihood);
			num_cov_pars_ = re_model_sp_->num_cov_par_;
		}
		else {
			re_model_den_->SetLikelihood(likelihood);
			num_cov_pars_ = re_model_den_->num_cov_par_;
		}
		if (!GaussLikelihood() && !cov_pars_optimizer_hase_been_set_) {
			optimizer_cov_pars_ = "gradient_descent";
			optimizer_coef_ = "gradient_descent";
		}
	}

	string_t REModel::GetOptimizerCovPars() const {
		return(optimizer_cov_pars_);
	}

	string_t REModel::GetOptimizerCoef() const {
		return(optimizer_coef_);
	}

	void REModel::SetOptimConfig(double* init_cov_pars, double lr,
		double acc_rate_cov, int max_iter, double delta_rel_conv,
		bool use_nesterov_acc, int nesterov_schedule_version, bool trace,
		const char* optimizer, int momentum_offset, const char* convergence_criterion,
		bool calc_std_dev) {
		if (init_cov_pars != nullptr) {
			vec_t init_cov_pars_orig = Eigen::Map<const vec_t>(init_cov_pars, num_cov_pars_);
			init_cov_pars_ = vec_t(num_cov_pars_);
			if (sparse_) {
				re_model_sp_->TransformCovPars(init_cov_pars_orig, init_cov_pars_);
			}
			else {
				re_model_den_->TransformCovPars(init_cov_pars_orig, init_cov_pars_);
			}
			init_cov_pars_provided_ = true;
			covariance_matrix_has_been_factorized_ = false;
		}
		lr_cov_ = lr;
		acc_rate_cov_ = acc_rate_cov;
		max_iter_ = max_iter;
		delta_rel_conv_ = delta_rel_conv;
		use_nesterov_acc_ = use_nesterov_acc;
		nesterov_schedule_version_ = nesterov_schedule_version;
		if (optimizer != nullptr) {
			optimizer_cov_pars_ = std::string(optimizer);
			cov_pars_optimizer_hase_been_set_ = true;
		}
		if (convergence_criterion != nullptr) {
			convergence_criterion_ = std::string(convergence_criterion);
		}
		momentum_offset_ = momentum_offset;
		if (trace) {
			Log::ResetLogLevelRE(LogLevelRE::Debug);
		}
		else {
			Log::ResetLogLevelRE(LogLevelRE::Info);
		}
		calc_std_dev_ = calc_std_dev;
	}

	void REModel::InitializeCovParsIfNotDefined(const double* y_data) {
		if (!cov_pars_initialized_) {
			if (init_cov_pars_provided_) {
				cov_pars_ = init_cov_pars_;
			}
			else {
				cov_pars_ = vec_t(num_cov_pars_);
				if (sparse_) {
					re_model_sp_->FindInitCovPar(y_data, cov_pars_.data());
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

	void REModel::ResetCovPars() {
		cov_pars_ = vec_t(num_cov_pars_);
		cov_pars_initialized_ = false;
	}

	void REModel::SetOptimCoefConfig(int num_covariates, double* init_coef,
		double lr_coef, double acc_rate_coef, const char* optimizer) {
		if (init_coef != nullptr) {
			coef_ = Eigen::Map<const vec_t>(init_coef, num_covariates);
			coef_initialized_ = true;
		}
		else {
			coef_initialized_ = false;
		}
		lr_coef_ = lr_coef;
		acc_rate_coef_ = acc_rate_coef;
		if (optimizer != nullptr) {
			optimizer_coef_ = std::string(optimizer);
		}
	}

	void REModel::OptimCovPar(const double* y_data, const double* fixed_effects) {
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
		if (sparse_) {
			re_model_sp_->OptimLinRegrCoefCovPar(y_data,
				nullptr,
				0,
				cov_pars_.data(),
				nullptr,
				num_it_,
				cov_pars_.data(),
				nullptr,
				1,
				lr_cov_,
				1,
				acc_rate_cov_,
				momentum_offset_,
				max_iter_,
				delta_rel_conv_,
				use_nesterov_acc_,
				nesterov_schedule_version_,
				optimizer_cov_pars_,
				"none",
				std_dev_cov_par,
				nullptr,
				calc_std_dev_,
				convergence_criterion_,
				fixed_effects,
				true);
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
				1,
				lr_cov_,
				1,
				acc_rate_cov_,
				momentum_offset_,
				max_iter_,
				delta_rel_conv_,
				use_nesterov_acc_,
				nesterov_schedule_version_,
				optimizer_cov_pars_,
				"none",
				std_dev_cov_par,
				nullptr,
				calc_std_dev_,
				convergence_criterion_,
				fixed_effects,
				true);
		}
		has_covariates_ = false;
		covariance_matrix_has_been_factorized_ = true;
	}

	void REModel::OptimLinRegrCoefCovPar(const double* y_data, const double* covariate_data, int num_covariates) {
		InitializeCovParsIfNotDefined(y_data);
		if (!coef_initialized_) {
			coef_ = vec_t(num_covariates);
			coef_.setZero();
			coef_initialized_ = true;
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
		if (sparse_) {
			re_model_sp_->OptimLinRegrCoefCovPar(y_data,
				covariate_data,
				num_covariates,
				cov_pars_.data(),
				coef_.data(),
				num_it_,
				cov_pars_.data(),
				coef_.data(),
				lr_coef_,
				lr_cov_,
				acc_rate_coef_,
				acc_rate_cov_,
				momentum_offset_,
				max_iter_,
				delta_rel_conv_,
				use_nesterov_acc_,
				nesterov_schedule_version_,
				optimizer_cov_pars_,
				optimizer_coef_,
				std_dev_cov_par,
				std_dev_coef,
				calc_std_dev_,
				convergence_criterion_,
				nullptr,
				true);
		}
		else {
			re_model_den_->OptimLinRegrCoefCovPar(y_data,
				covariate_data,
				num_covariates,
				cov_pars_.data(),
				coef_.data(),
				num_it_,
				cov_pars_.data(),
				coef_.data(),
				lr_coef_,
				lr_cov_,
				acc_rate_coef_,
				acc_rate_cov_,
				momentum_offset_,
				max_iter_,
				delta_rel_conv_,
				use_nesterov_acc_,
				nesterov_schedule_version_,
				optimizer_cov_pars_,
				optimizer_coef_,
				std_dev_cov_par,
				std_dev_coef,
				calc_std_dev_,
				convergence_criterion_,
				nullptr,
				true);
		}
		has_covariates_ = true;
		covariance_matrix_has_been_factorized_ = true;
	}

	void REModel::FindInitialValueBoosting(double* init_score) {
		CHECK(cov_pars_initialized_);
		vec_t covariate_data(GetNumData());
		covariate_data.setOnes();
		init_score[0] = 0.;
		if (sparse_) {
			re_model_sp_->OptimLinRegrCoefCovPar(nullptr,
				covariate_data.data(),
				1,
				cov_pars_.data(),
				init_score,
				num_it_,
				cov_pars_.data(),
				init_score,
				lr_coef_,
				lr_cov_,
				acc_rate_coef_,
				acc_rate_cov_,
				momentum_offset_,
				max_iter_,
				delta_rel_conv_,
				use_nesterov_acc_,
				nesterov_schedule_version_,
				optimizer_cov_pars_,
				optimizer_coef_,
				nullptr,
				nullptr,
				false,
				convergence_criterion_,
				nullptr,
				false);
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
				lr_coef_,
				lr_cov_,
				acc_rate_coef_,
				acc_rate_cov_,
				momentum_offset_,
				max_iter_,
				delta_rel_conv_,
				use_nesterov_acc_,
				nesterov_schedule_version_,
				optimizer_cov_pars_,
				optimizer_coef_,
				nullptr,
				nullptr,
				false,
				convergence_criterion_,
				nullptr,
				false);
		}
	}

	void REModel::EvalNegLogLikelihood(const double* y_data, double* cov_pars, double& negll,
		const double* fixed_effects, bool InitializeModeCovMat, bool CalcModePostRandEff_already_done) {
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
			if (sparse_) {
				re_model_sp_->TransformCovPars(cov_pars_orig, cov_pars_trafo);
			}
			else {
				re_model_den_->TransformCovPars(cov_pars_orig, cov_pars_trafo);
			}
		}

		if (sparse_) {
			if (re_model_sp_->gauss_likelihood_) {
				re_model_sp_->EvalNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, false, false, false);
			}
			else {
				re_model_sp_->EvalLAApproxNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, fixed_effects, InitializeModeCovMat, CalcModePostRandEff_already_done);
			}
		}
		else {
			if (re_model_den_->gauss_likelihood_) {
				re_model_den_->EvalNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, false, false, false);
			}
			else {
				re_model_den_->EvalLAApproxNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, fixed_effects, InitializeModeCovMat, CalcModePostRandEff_already_done);
			}
		}
		covariance_matrix_has_been_factorized_ = false;
		//set to false as otherwise the covariance is not factorized for prediction for Gaussian data and this can lead to problems 
		//(e.g. fitting model with covariates, then calling likelihood without covariates, then making prediction with covariates)
	}

	void REModel::CalcGradient(double* y, const double* fixed_effects, bool calc_cov_factor) {
		if (y != nullptr) {
			InitializeCovParsIfNotDefined(y);
		}
		CHECK(cov_pars_initialized_);
		if (sparse_) {
			//1. Factorize covariance matrix
			if (calc_cov_factor) {
				re_model_sp_->SetCovParsComps(cov_pars_);
				if (re_model_sp_->gauss_likelihood_) {//Gaussian data
					re_model_sp_->CalcCovFactor(false, true, 1., false);
				}
				else {//not gauss_likelihood_
					if (re_model_sp_->vecchia_approx_) {
						re_model_sp_->CalcCovFactor(false, true, 1., false);
					}
					else {
						re_model_sp_->CalcSigmaComps();
						re_model_sp_->CalcCovMatrixNonGauss();
					}
					re_model_sp_->CalcModePostRandEff(fixed_effects);
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
		}//end sparse_
		else {//not sparse_
			//1. Factorize covariance matrix
			if (calc_cov_factor) {
				re_model_den_->SetCovParsComps(cov_pars_);
				if (re_model_den_->gauss_likelihood_) {//Gaussian data
					re_model_den_->CalcCovFactor(false, true, 1., false);
				}
				else {//not gauss_likelihood_
					if (re_model_den_->vecchia_approx_) {
						re_model_den_->CalcCovFactor(false, true, 1., false);
					}
					else {
						re_model_den_->CalcSigmaComps();
						re_model_den_->CalcCovMatrixNonGauss();
					}
					re_model_den_->CalcModePostRandEff(fixed_effects);
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
		}//end not sparse_
		if (calc_cov_factor) {
			covariance_matrix_has_been_factorized_ = true;
		}
	}

	void REModel::SetY(const double* y) const {
		if (sparse_) {
			re_model_sp_->SetY(y);
		}
		else {
			re_model_den_->SetY(y);
		}
	}

	void REModel::SetY(const float* y) const {
		if (sparse_) {
			re_model_sp_->SetY(y);
		}
		else {
			re_model_den_->SetY(y);
		}
	}

	void REModel::GetY(double* y) const {
		if (sparse_) {
			re_model_sp_->GetY(y);
		}
		else {
			re_model_den_->GetY(y);
		}
	}

	void REModel::GetCovariateData(double* covariate_data) const {
		if (sparse_) {
			re_model_sp_->GetCovariateData(covariate_data);
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
		if (sparse_) {
			re_model_sp_->TransformBackCovPars(cov_pars_, cov_pars_orig);
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
			if (sparse_) {
				re_model_sp_->TransformBackCovPars(init_cov_pars_, init_cov_pars_orig);
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
		const gp_id_t* cluster_ids_data_pred, const char* re_group_data_pred,
		const double* re_group_rand_coef_data_pred, double* gp_coords_data_pred,
		const double* gp_rand_coef_data_pred, const double* covariate_data_pred) {
		if (sparse_) {
			re_model_sp_->SetPredictionData(num_data_pred,
				cluster_ids_data_pred, re_group_data_pred, re_group_rand_coef_data_pred, gp_coords_data_pred,
				gp_rand_coef_data_pred, covariate_data_pred);
		}
		else {
			re_model_den_->SetPredictionData(num_data_pred,
				cluster_ids_data_pred, re_group_data_pred, re_group_rand_coef_data_pred, gp_coords_data_pred,
				gp_rand_coef_data_pred, covariate_data_pred);
		}
	}

	void REModel::Predict(const double* y_obs, data_size_t num_data_pred, double* out_predict,
		bool predict_cov_mat, bool predict_var, bool predict_response,
		const gp_id_t* cluster_ids_data_pred, const char* re_group_data_pred, const double* re_group_rand_coef_data_pred,
		double* gp_coords_data_pred, const double* gp_rand_coef_data_pred,
		const double* cov_pars_pred, const double* covariate_data_pred,
		bool use_saved_data, const char* vecchia_pred_type, int num_neighbors_pred,
		const double* fixed_effects, const double* fixed_effects_pred,
		bool suppress_calc_cov_factor) const {
		bool calc_cov_factor = true;
		vec_t cov_pars_pred_trans;
		if (cov_pars_pred != nullptr) {
			vec_t cov_pars_pred_orig = Eigen::Map<const vec_t>(cov_pars_pred, num_cov_pars_);
			cov_pars_pred_trans = vec_t(num_cov_pars_);
			if (sparse_) {
				re_model_sp_->TransformCovPars(cov_pars_pred_orig, cov_pars_pred_trans);
			}
			else {
				re_model_den_->TransformCovPars(cov_pars_pred_orig, cov_pars_pred_trans);
			}
		}//end if cov_pars_pred != nullptr
		else {// use saved cov_pars
			if (!cov_pars_initialized_) {
				Log::REFatal("Covariance parameters have not been estimated or are not given.");
			}
			// Note: cov_pars_initialized_ is set to true by InitializeCovParsIfNotDefined() which is called by OptimCovPar(), OptimLinRegrCoefCovPar(), and EvalNegLogLikelihood().
			//			It is assume that if one of these three functions has been called, the covariance parameters have been estimated
			cov_pars_pred_trans = cov_pars_;
			if (GaussLikelihood()) {
				// We don't factorize the covariance matrix for Gaussian data in case this has already been done (e.g. at the end of the estimation)
				// For non-Gaussian, we always calculate the Laplace approximation to guarantee that all calls to predict() return the same values
				//	Otherwise, there can be (very) small difference between (i) calculating predictions using an estimated a model and
				//	(ii) loading / initializing an empty a model, providing the same covariance paramters, and thus recalculating the mode.
				//	This is due to very small differences in the mode which are found differently: for an estimated model, the mode has been iteratively found during the estimation,
				//	in particular, in the last optimization round the mode has been initialized at the previous value,
				//	where as when the covariance parameters are provided here, the mode is found using these parameters but initilized from 0.
				calc_cov_factor = !covariance_matrix_has_been_factorized_;
			}
		}// end use saved cov_pars
		if (has_covariates_) {
			CHECK(coef_initialized_ == true);
		}
		if (suppress_calc_cov_factor) {
			calc_cov_factor = false;
		}
		if (sparse_) {
			re_model_sp_->Predict(cov_pars_pred_trans.data(), y_obs, num_data_pred,
				out_predict, calc_cov_factor, predict_cov_mat, predict_var, predict_response,
				covariate_data_pred, coef_.data(),
				cluster_ids_data_pred, re_group_data_pred,
				re_group_rand_coef_data_pred, gp_coords_data_pred,
				gp_rand_coef_data_pred, use_saved_data,
				vecchia_pred_type, num_neighbors_pred,
				fixed_effects, fixed_effects_pred);
		}
		else {
			re_model_den_->Predict(cov_pars_pred_trans.data(), y_obs, num_data_pred,
				out_predict, calc_cov_factor, predict_cov_mat, predict_var, predict_response,
				covariate_data_pred, coef_.data(),
				cluster_ids_data_pred, re_group_data_pred,
				re_group_rand_coef_data_pred, gp_coords_data_pred,
				gp_rand_coef_data_pred, use_saved_data,
				vecchia_pred_type, num_neighbors_pred,
				fixed_effects, fixed_effects_pred);
		}
	}

	int REModel::GetNumIt() const {
		return(num_it_);
	}

	int REModel::GetNumData() const {
		if (sparse_) {
			return(re_model_sp_->num_data_);
		}
		else {
			return(re_model_den_->num_data_);
		}
	}

	void REModel::NewtonUpdateLeafValues(const int* data_leaf_index,
		const int num_leaves, double* leaf_values) const {
		if (sparse_) {
			re_model_sp_->NewtonUpdateLeafValues(data_leaf_index, num_leaves, leaf_values, cov_pars_[0]);
		}
		else {
			re_model_den_->NewtonUpdateLeafValues(data_leaf_index, num_leaves, leaf_values, cov_pars_[0]);
		}
	}

}  // namespace GPBoost
