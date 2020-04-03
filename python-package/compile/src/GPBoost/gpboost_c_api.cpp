/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*	This file contains original work (functions 'GPB_APIHandleException', 'API_BEGIN', 'API_END') which are
*		Copyright (c) 2016 Microsoft Corporation. All rights reserved.
*
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#include <GPBoost/re_model.h>
#include <GPBoost/gpboost_c_api.h>

namespace GPBoost {

	inline int GPB_APIHandleException(const std::exception& ex) {
		GPB_SetLastError(ex.what());
		return -1;
	}
	inline int GPB_APIHandleException(const std::string& ex) {
		GPB_SetLastError(ex.c_str());
		return -1;
	}

	#define API_BEGIN() try {
	#define API_END() } \
	catch(std::exception& ex) { return GPB_APIHandleException(ex); } \
	catch(std::string& ex) { return GPB_APIHandleException(ex); } \
	catch(...) { return GPB_APIHandleException("unknown exception"); } \
	return 0;

}

using namespace GPBoost;

int GPB_CreateREModel(int num_data,
	const int* cluster_ids_data,
	const char* re_group_data,
	int num_re_group,
	const double* re_group_rand_coef_data,
	const int* ind_effect_group_rand_coef,
	int num_re_group_rand_coef,
	int num_gp,
	const double* gp_coords_data,
	const int dim_gp_coords,
	const double* gp_rand_coef_data,
	int num_gp_rand_coef,
	const char* cov_fct,
  double cov_fct_shape,
  bool vecchia_approx,
  int num_neighbors,
  const char* vecchia_ordering,
  const char* vecchia_pred_type,
  int num_neighbors_pred,
	REModelHandle* out) {
	API_BEGIN();
  std::unique_ptr<REModel> ret;
  ret.reset(new REModel(num_data, cluster_ids_data, re_group_data, num_re_group,
    re_group_rand_coef_data, ind_effect_group_rand_coef, num_re_group_rand_coef,
    num_gp, gp_coords_data, dim_gp_coords, gp_rand_coef_data, num_gp_rand_coef, cov_fct, cov_fct_shape,
    vecchia_approx, num_neighbors, vecchia_ordering, vecchia_pred_type, num_neighbors_pred));
  *out = ret.release();
	API_END();
}

int GPB_REModelFree(REModelHandle handle) {
	API_BEGIN();
  delete reinterpret_cast<REModel*>(handle);
	API_END();
}

int GPB_SetOptimConfig(REModelHandle handle,
  double* init_cov_pars,
  double lr,
  double acc_rate_cov,
  int max_iter,
  double delta_rel_conv,
  bool use_nesterov_acc,
  int nesterov_schedule_version,
  bool trace,
  const char* optimizer,
  int momentum_offset) {
  API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  ref_remodel->SetOptimConfig(init_cov_pars, lr, acc_rate_cov, max_iter, delta_rel_conv,
    use_nesterov_acc, nesterov_schedule_version, trace, optimizer, momentum_offset);
  API_END();
}

int GPB_SetOptimCoefConfig(REModelHandle handle,
  int num_covariates,
  double* init_coef,
  double lr_coef,
  double acc_rate_coef,
  const char* optimizer) {
  API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  ref_remodel->SetOptimCoefConfig(num_covariates, init_coef, lr_coef, acc_rate_coef, optimizer);
  API_END();
}

int GPB_OptimCovPar(REModelHandle handle,
  const double* y_data,
  bool calc_std_dev) {
  API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  ref_remodel->OptimCovPar(y_data, calc_std_dev);
  API_END();
}

int GPB_OptimLinRegrCoefCovPar(REModelHandle handle,
	const double* y_data,
	const double* covariate_data,
	int num_covariates,
  bool calc_std_dev) {
	API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  ref_remodel->OptimLinRegrCoefCovPar(y_data, covariate_data, num_covariates, calc_std_dev);
	API_END();
}

int GPB_GetCovPar(REModelHandle handle,
  double* optim_cov_pars,
  bool calc_std_dev) {
  API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  ref_remodel->GetCovPar(optim_cov_pars, calc_std_dev);
  API_END();
}

int GPB_GetCoef(REModelHandle handle,
  double* optim_coef,
  bool calc_std_dev) {
  API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  ref_remodel->GetCoef(optim_coef, calc_std_dev);
  API_END();
}

int GPB_GetNumIt(REModelHandle handle,
  int* num_it) {
  API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  num_it[0] = ref_remodel->GetNumIt();
  API_END();
}

int GPB_SetPredictionData(REModelHandle handle,
  int num_data_pred,
  const int* cluster_ids_data_pred,
  const char* re_group_data_pred,
  const double* re_group_rand_coef_data_pred,
  double* gp_coords_data_pred,
  const double* gp_rand_coef_data_pred,
  const double* covariate_data_pred) {
  API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  ref_remodel->SetPredictionData(num_data_pred,
    cluster_ids_data_pred, re_group_data_pred, re_group_rand_coef_data_pred,
    gp_coords_data_pred, gp_rand_coef_data_pred, covariate_data_pred);
  API_END();
}

int GPB_PredictREModel(REModelHandle handle,
	const double* y_data,
	int num_data_pred,
	double* out_predict,
	bool predict_cov_mat,
	const int* cluster_ids_data_pred,
	const char* re_group_data_pred,
	const double* re_group_rand_coef_data_pred,
	double* gp_coords_pred,
	const double* gp_rand_coef_data_pred,
  const double* cov_pars,
  const double* covariate_data_pred,
  bool use_saved_data,
  const char* vecchia_pred_type,
  int num_neighbors_pred) {
	API_BEGIN();
  REModel* ref_remodel = reinterpret_cast<REModel*>(handle);
  ref_remodel->Predict(y_data, num_data_pred, out_predict, predict_cov_mat,
    cluster_ids_data_pred, re_group_data_pred, re_group_rand_coef_data_pred,
    gp_coords_pred, gp_rand_coef_data_pred, cov_pars, covariate_data_pred,
    use_saved_data, vecchia_pred_type, num_neighbors_pred);
	API_END();
}
