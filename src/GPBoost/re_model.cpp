/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/re_model.h>
#include <GPBoost/log.h>

namespace GPBoost {

	REModel::REModel() {
	}

	REModel::REModel(data_size_t num_data, const gp_id_t* cluster_ids_data, const char* re_group_data, data_size_t num_re_group,
		const double* re_group_rand_coef_data, const int32_t* ind_effect_group_rand_coef, data_size_t num_re_group_rand_coef,
		data_size_t num_gp, const double* gp_coords_data, int dim_gp_coords, const double* gp_rand_coef_data, data_size_t num_gp_rand_coef,
		const char* cov_fct, double cov_fct_shape, bool vecchia_approx, int num_neighbors, const char* vecchia_ordering,
		const char* vecchia_pred_type, int num_neighbors_pred) {
		if ((num_gp + num_gp_rand_coef) == 0) {
			sparse_ = true;
			re_model_sp_ = std::unique_ptr<REModelTemplate<sp_mat_t, chol_sp_mat_t>>(new REModelTemplate<sp_mat_t, chol_sp_mat_t>(
				num_data, cluster_ids_data, re_group_data, num_re_group,
				re_group_rand_coef_data, ind_effect_group_rand_coef, num_re_group_rand_coef,
				num_gp, gp_coords_data, dim_gp_coords, gp_rand_coef_data, num_gp_rand_coef,
				cov_fct, cov_fct_shape, vecchia_approx, num_neighbors, vecchia_ordering,
				vecchia_pred_type, num_neighbors_pred));
			num_cov_pars_ = re_model_sp_->num_cov_par_;
		}
		else {
			sparse_ = false;
			re_model_den_ = std::unique_ptr <REModelTemplate< den_mat_t, chol_den_mat_t >>(new REModelTemplate<den_mat_t, chol_den_mat_t>(
				num_data, cluster_ids_data, re_group_data, num_re_group,
				re_group_rand_coef_data, ind_effect_group_rand_coef, num_re_group_rand_coef,
				num_gp, gp_coords_data, dim_gp_coords, gp_rand_coef_data, num_gp_rand_coef,
				cov_fct, cov_fct_shape, vecchia_approx, num_neighbors, vecchia_ordering,
				vecchia_pred_type, num_neighbors_pred));
			num_cov_pars_ = re_model_den_->num_cov_par_;
		}
	}

	/*! \brief Destructor */
	REModel::~REModel() {
	}

	void REModel::SetOptimConfig(double* init_cov_pars, double lr,
		double acc_rate_cov, int max_iter, double delta_rel_conv,
		bool use_nesterov_acc, int nesterov_schedule_version, bool trace,
		const char* optimizer, int momentum_offset, const char* convergence_criterion) {
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
		}
		else {
			init_cov_pars_provided_ = false;
		}
		cov_pars_initialized_ = false;
		lr_cov_ = lr;
		acc_rate_cov_ = acc_rate_cov;
		max_iter_ = max_iter;
		delta_rel_conv_ = delta_rel_conv;
		use_nesterov_acc_ = use_nesterov_acc;
		nesterov_schedule_version_ = nesterov_schedule_version;
		if (optimizer != nullptr) {
			optimizer_cov_pars_ = std::string(optimizer);
		}
		if (convergence_criterion != nullptr) {
			convergence_criterion_ = std::string(convergence_criterion);
		}
		momentum_offset_ = momentum_offset;
		if (trace) {
			Log::ResetLogLevel(LogLevel::Debug);
		}
		else {
			Log::ResetLogLevel(LogLevel::Info);
		}
	}

	void REModel::CheckCovParsInitialized(const double* y_data) {
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

	void REModel::OptimCovPar(const double* y_data, bool calc_std_dev) {
		CheckCovParsInitialized(y_data);
		double* std_dev_cov_par;
		if (calc_std_dev) {
			std_dev_cov_pars_ = vec_t(num_cov_pars_);
			std_dev_cov_par = std_dev_cov_pars_.data();
		}
		else {
			std_dev_cov_par = nullptr;
		}
		if (sparse_) {
			re_model_sp_->OptimCovPar(y_data, cov_pars_.data(), cov_pars_.data(), num_it_, lr_cov_, acc_rate_cov_, momentum_offset_,
				max_iter_, delta_rel_conv_, optimizer_cov_pars_, use_nesterov_acc_, nesterov_schedule_version_, std_dev_cov_par, calc_std_dev,
				nullptr, convergence_criterion_);
		}
		else {
			re_model_den_->OptimCovPar(y_data, cov_pars_.data(), cov_pars_.data(), num_it_, lr_cov_, acc_rate_cov_, momentum_offset_,
				max_iter_, delta_rel_conv_, optimizer_cov_pars_, use_nesterov_acc_, nesterov_schedule_version_, std_dev_cov_par, calc_std_dev,
				nullptr, convergence_criterion_);
		}
		has_covariates_ = false;
	}

	void REModel::OptimLinRegrCoefCovPar(const double* y_data, const double* covariate_data, int num_covariates, bool calc_std_dev) {
		CheckCovParsInitialized(y_data);
		if (!coef_initialized_) {
			coef_ = vec_t(num_covariates);
			coef_.setZero();
			coef_initialized_ = true;
		}
		double* std_dev_cov_par;
		double* std_dev_coef;
		if (calc_std_dev) {
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
			re_model_sp_->OptimLinRegrCoefCovPar(y_data, covariate_data, num_covariates,
				cov_pars_.data(), coef_.data(), num_it_, cov_pars_.data(), coef_.data(), lr_coef_, lr_cov_, acc_rate_coef_, acc_rate_cov_, momentum_offset_,
				max_iter_, delta_rel_conv_, use_nesterov_acc_, nesterov_schedule_version_, optimizer_cov_pars_, optimizer_coef_, std_dev_cov_par, std_dev_coef,
				calc_std_dev, convergence_criterion_);
		}
		else {
			re_model_den_->OptimLinRegrCoefCovPar(y_data, covariate_data, num_covariates,
				cov_pars_.data(), coef_.data(), num_it_, cov_pars_.data(), coef_.data(), lr_coef_, lr_cov_, acc_rate_coef_, acc_rate_cov_, momentum_offset_,
				max_iter_, delta_rel_conv_, use_nesterov_acc_, nesterov_schedule_version_, optimizer_cov_pars_, optimizer_coef_, std_dev_cov_par, std_dev_coef,
				calc_std_dev, convergence_criterion_);
		}
		has_covariates_ = true;
	}

	void REModel::EvalNegLogLikelihood(const double* y_data, double* cov_pars, double& negll) {
		vec_t cov_pars_orig = Eigen::Map<const vec_t>(cov_pars, num_cov_pars_);
		vec_t cov_pars_trafo = vec_t(num_cov_pars_);
		if (sparse_) {
			re_model_sp_->TransformCovPars(cov_pars_orig, cov_pars_trafo);
		}
		else {
			re_model_den_->TransformCovPars(cov_pars_orig, cov_pars_trafo);
		}
		if (sparse_) {
			re_model_sp_->EvalNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, true);
		}
		else {
			re_model_den_->EvalNegLogLikelihood(y_data, cov_pars_trafo.data(), negll, true);
		}
	}

	void REModel::CalcGetYAux(double* y, bool calc_cov_factor) {
		CheckCovParsInitialized(y);
		if (sparse_) {
			if (calc_cov_factor) {
				re_model_sp_->SetCovParsComps(cov_pars_);
				re_model_sp_->CalcCovFactor(false);
			}
			re_model_sp_->SetY(y);
			re_model_sp_->CalcYAux(cov_pars_[0]);
			re_model_sp_->GetYAux(y);
		}
		else {
			if (calc_cov_factor) {
				re_model_den_->SetCovParsComps(cov_pars_);
				re_model_den_->CalcCovFactor(false);
			}
			re_model_den_->SetY(y);
			re_model_den_->CalcYAux(cov_pars_[0]);
			re_model_den_->GetYAux(y);
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

	void REModel::GetCovPar(double* cov_par, bool calc_std_dev) const {
		if (cov_pars_.size() == 0) {
			Log::Fatal("Covariance parameters have not been estimated or set");
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

	void REModel::Predict(const double* y_obs, data_size_t num_data_pred,
		double* out_predict, bool predict_cov_mat,
		const gp_id_t* cluster_ids_data_pred, const char* re_group_data_pred, const double* re_group_rand_coef_data_pred,
		double* gp_coords_data_pred, const double* gp_rand_coef_data_pred, const double* cov_pars_pred,
		const double* covariate_data_pred, bool use_saved_data, const char* vecchia_pred_type, int num_neighbors_pred) const {
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
		}
		else {
			if (!cov_pars_initialized_) {
				Log::Fatal("Covariance parameters have not been estimated or set");
			}
			cov_pars_pred_trans = cov_pars_;
		}
		if (has_covariates_) {
			CHECK(coef_initialized_ == true);
		}
		if (sparse_) {
			re_model_sp_->Predict(cov_pars_pred_trans.data(), y_obs, num_data_pred, out_predict, predict_cov_mat,
				covariate_data_pred, coef_.data(),
				cluster_ids_data_pred, re_group_data_pred, re_group_rand_coef_data_pred, gp_coords_data_pred,
				gp_rand_coef_data_pred, use_saved_data, vecchia_pred_type, num_neighbors_pred);
		}
		else {
			re_model_den_->Predict(cov_pars_pred_trans.data(), y_obs, num_data_pred, out_predict, predict_cov_mat,
				covariate_data_pred, coef_.data(),
				cluster_ids_data_pred, re_group_data_pred, re_group_rand_coef_data_pred, gp_coords_data_pred,
				gp_rand_coef_data_pred, use_saved_data, vecchia_pred_type, num_neighbors_pred);
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
