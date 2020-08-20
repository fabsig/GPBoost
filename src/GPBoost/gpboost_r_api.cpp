/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*	This file contains original work (functions 'R_API_BEGIN', 'R_API_END', 'CHECK_CALL') which are
*		Copyright (c) 2016 Microsoft Corporation. All rights reserved.
*
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#include <GPBoost/gpboost_r_api.h>
#include <LightGBM/lightgbm_R.h>
#include <string>

#ifdef GPB_R_BUILD
#include <R_ext/Rdynload.h>
#endif

#define R_API_BEGIN() \
  try {
#define R_API_END() } \
  catch(std::exception& ex) { R_INT_PTR(call_state)[0] = -1; GPB_SetLastError(ex.what()); return call_state;} \
  catch(std::string& ex) { R_INT_PTR(call_state)[0] = -1; GPB_SetLastError(ex.c_str()); return call_state; } \
  catch(...) { R_INT_PTR(call_state)[0] = -1; GPB_SetLastError("unknown exception"); return call_state;} \
  return call_state;

#define CHECK_CALL(x) \
  if ((x) != 0) { \
    R_INT_PTR(call_state)[0] = -1;\
    return call_state;\
  }

//using namespace GPBoost;

LGBM_SE GPB_CreateREModel_R(LGBM_SE ndata,
	LGBM_SE cluster_ids_data,
	LGBM_SE re_group_data,
	LGBM_SE num_re_group,
	LGBM_SE re_group_rand_coef_data,
	LGBM_SE ind_effect_group_rand_coef,
	LGBM_SE num_re_group_rand_coef,
	LGBM_SE num_gp,
	LGBM_SE gp_coords_data,
	LGBM_SE dim_gp_coords,
	LGBM_SE gp_rand_coef_data,
	LGBM_SE num_gp_rand_coef,
	LGBM_SE cov_fct,
	LGBM_SE cov_fct_shape,
	LGBM_SE vecchia_approx,
	LGBM_SE num_neighbors,
	LGBM_SE vecchia_ordering,
	LGBM_SE vecchia_pred_type,
	LGBM_SE num_neighbors_pred,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	REModelHandle handle = nullptr;
	//bool va = R_AS_BOOL(vecchia_approx);//FOR DEBUGGING
	//Log::Info("type(va) = %s ", typeid(va).name());
	//Log::Info("va = %d ", va);
	//int va_int_convert = va ? 1 : 0;
	//Log::Info("va_int_convert = %d ", va_int_convert);
	//int va_int = R_AS_INT(vecchia_approx);
	//Log::Info("va_int = %d ", va_int);
	CHECK_CALL(GPB_CreateREModel(R_AS_INT(ndata), R_INT_PTR(cluster_ids_data), R_CHAR_PTR(re_group_data),
		R_AS_INT(num_re_group), R_REAL_PTR(re_group_rand_coef_data), R_INT_PTR(ind_effect_group_rand_coef), R_AS_INT(num_re_group_rand_coef),
		R_AS_INT(num_gp), R_REAL_PTR(gp_coords_data), R_AS_INT(dim_gp_coords), R_REAL_PTR(gp_rand_coef_data), R_AS_INT(num_gp_rand_coef),
		R_CHAR_PTR(cov_fct), R_AS_DOUBLE(cov_fct_shape), R_AS_BOOL(vecchia_approx), R_AS_INT(num_neighbors), R_CHAR_PTR(vecchia_ordering), R_CHAR_PTR(vecchia_pred_type),
		R_AS_INT(num_neighbors_pred), &handle));
	R_SET_PTR(out, handle);
	R_API_END();
}

LGBM_SE GPB_REModelFree_R(LGBM_SE handle,
	LGBM_SE call_state) {
	R_API_BEGIN();
	if (R_GET_PTR(handle) != nullptr) {
		CHECK_CALL(GPB_REModelFree(R_GET_PTR(handle)));
		R_SET_PTR(handle, nullptr);
	}
	R_API_END();
}

LGBM_SE GPB_SetOptimConfig_R(LGBM_SE handle,
	LGBM_SE init_cov_pars,
	LGBM_SE lr,
	LGBM_SE acc_rate_cov,
	LGBM_SE max_iter,
	LGBM_SE delta_rel_conv,
	LGBM_SE use_nesterov_acc,
	LGBM_SE nesterov_schedule_version,
	LGBM_SE trace,
	LGBM_SE optimizer,
	LGBM_SE momentum_offset,
	LGBM_SE convergence_criterion,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_SetOptimConfig(R_GET_PTR(handle), R_REAL_PTR(init_cov_pars),
		R_AS_DOUBLE(lr), R_AS_DOUBLE(acc_rate_cov), R_AS_INT(max_iter),
		R_AS_DOUBLE(delta_rel_conv), R_AS_BOOL(use_nesterov_acc),
		R_AS_INT(nesterov_schedule_version), R_AS_BOOL(trace), R_CHAR_PTR(optimizer),
		R_AS_INT(momentum_offset), R_CHAR_PTR(convergence_criterion)));
	R_API_END();
}

LGBM_SE GPB_SetOptimCoefConfig_R(LGBM_SE handle,
	LGBM_SE num_covariates,
	LGBM_SE init_coef,
	LGBM_SE lr_coef,
	LGBM_SE acc_rate_coef,
	LGBM_SE optimizer,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_SetOptimCoefConfig(R_GET_PTR(handle), R_AS_INT(num_covariates),
		R_REAL_PTR(init_coef), R_AS_DOUBLE(lr_coef), R_AS_DOUBLE(acc_rate_coef), R_CHAR_PTR(optimizer)));
	R_API_END();
}

LGBM_SE GPB_OptimCovPar_R(LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE calc_std_dev,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_OptimCovPar(R_GET_PTR(handle), R_REAL_PTR(y_data), R_AS_BOOL(calc_std_dev)));
	R_API_END();
}

LGBM_SE GPB_OptimLinRegrCoefCovPar_R(LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE covariate_data,
	LGBM_SE num_covariates,
	LGBM_SE calc_std_dev,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_OptimLinRegrCoefCovPar(R_GET_PTR(handle), R_REAL_PTR(y_data), R_REAL_PTR(covariate_data),
		R_AS_INT(num_covariates), R_AS_BOOL(calc_std_dev)));
	R_API_END();
}

LGBM_SE GPB_EvalNegLogLikelihood_R(LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE cov_pars,
	LGBM_SE negll,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_EvalNegLogLikelihood(R_GET_PTR(handle), R_REAL_PTR(y_data), R_REAL_PTR(cov_pars), R_REAL_PTR(negll)));
	R_API_END();
}

LGBM_SE GPB_GetCovPar_R(LGBM_SE handle,
	LGBM_SE calc_std_dev,
	LGBM_SE optim_cov_pars,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_GetCovPar(R_GET_PTR(handle), R_REAL_PTR(optim_cov_pars), R_AS_BOOL(calc_std_dev)));
	R_API_END();
}

LGBM_SE GPB_GetCoef_R(LGBM_SE handle,
	LGBM_SE calc_std_dev,
	LGBM_SE optim_coef,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_GetCoef(R_GET_PTR(handle), R_REAL_PTR(optim_coef), R_AS_BOOL(calc_std_dev)));
	R_API_END();
}

LGBM_SE GPB_GetNumIt_R(LGBM_SE handle,
	LGBM_SE num_it,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_GetNumIt(R_GET_PTR(handle), R_INT_PTR(num_it)));
	R_API_END();
}

LGBM_SE GPB_SetPredictionData_R(LGBM_SE handle,
	LGBM_SE num_data_pred,
	LGBM_SE cluster_ids_data_pred,
	LGBM_SE re_group_data_pred,
	LGBM_SE re_group_rand_coef_data_pred,
	LGBM_SE gp_coords_data_pred,
	LGBM_SE gp_rand_coef_data_pred,
	LGBM_SE covariate_data_pred,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_SetPredictionData(R_GET_PTR(handle),
		R_AS_INT(num_data_pred), R_INT_PTR(cluster_ids_data_pred),
		R_CHAR_PTR(re_group_data_pred), R_REAL_PTR(re_group_rand_coef_data_pred),
		R_REAL_PTR(gp_coords_data_pred), R_REAL_PTR(gp_rand_coef_data_pred), R_REAL_PTR(covariate_data_pred)));
	R_API_END();
}

LGBM_SE GPB_PredictREModel_R(LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE num_data_pred,
	LGBM_SE predict_cov_mat,
	LGBM_SE cluster_ids_data_pred,
	LGBM_SE re_group_data_pred,
	LGBM_SE re_group_rand_coef_data_pred,
	LGBM_SE gp_coords_pred,
	LGBM_SE gp_rand_coef_data_pred,
	LGBM_SE cov_pars,
	LGBM_SE covariate_data_pred,
	LGBM_SE use_saved_data,
	LGBM_SE vecchia_pred_type,
	LGBM_SE num_neighbors_pred,
	LGBM_SE out_predict,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_PredictREModel(R_GET_PTR(handle), R_REAL_PTR(y_data),
		R_AS_INT(num_data_pred), R_REAL_PTR(out_predict),
		R_AS_BOOL(predict_cov_mat), R_INT_PTR(cluster_ids_data_pred),
		R_CHAR_PTR(re_group_data_pred), R_REAL_PTR(re_group_rand_coef_data_pred),
		R_REAL_PTR(gp_coords_pred), R_REAL_PTR(gp_rand_coef_data_pred), R_REAL_PTR(cov_pars),
		R_REAL_PTR(covariate_data_pred), R_AS_BOOL(use_saved_data),
		R_CHAR_PTR(vecchia_pred_type), R_AS_INT(num_neighbors_pred)));
	R_API_END();
}


#ifdef GPB_R_BUILD
// .Call() calls
static const R_CallMethodDef CallEntries[] = {
  {"GPB_CreateREModel_R"              , (DL_FUNC)&GPB_CreateREModel_R              , 21},
  {"GPB_REModelFree_R"                , (DL_FUNC)&GPB_REModelFree_R                , 2},
  {"GPB_SetOptimConfig_R"             , (DL_FUNC)&GPB_SetOptimConfig_R             , 13},
  {"GPB_SetOptimCoefConfig_R"         , (DL_FUNC)&GPB_SetOptimCoefConfig_R         , 7},
  {"GPB_OptimCovPar_R"                , (DL_FUNC)&GPB_OptimCovPar_R                , 4},
  {"GPB_OptimLinRegrCoefCovPar_R"     , (DL_FUNC)&GPB_OptimLinRegrCoefCovPar_R     , 6},
  {"GPB_EvalNegLogLikelihood_R"       , (DL_FUNC)&GPB_EvalNegLogLikelihood_R       , 5},
  {"GPB_GetCovPar_R"                  , (DL_FUNC)&GPB_GetCovPar_R                  , 4},
  {"GPB_GetCoef_R"                    , (DL_FUNC)&GPB_GetCoef_R                    , 4},
  {"GPB_GetNumIt_R"                   , (DL_FUNC)&GPB_GetNumIt_R                   , 3},
  {"GPB_SetPredictionData_R"          , (DL_FUNC)&GPB_SetPredictionData_R          , 9},
  {"GPB_PredictREModel_R"             , (DL_FUNC)&GPB_PredictREModel_R             , 16},
  {"LGBM_GetLastError_R"              , (DL_FUNC)&LGBM_GetLastError_R              , 3},
  {"LGBM_DatasetCreateFromFile_R"     , (DL_FUNC)&LGBM_DatasetCreateFromFile_R     , 5},
  {"LGBM_DatasetCreateFromCSC_R"      , (DL_FUNC)&LGBM_DatasetCreateFromCSC_R      , 10},
  {"LGBM_DatasetCreateFromMat_R"      , (DL_FUNC)&LGBM_DatasetCreateFromMat_R      , 7},
  {"LGBM_DatasetGetSubset_R"          , (DL_FUNC)&LGBM_DatasetGetSubset_R          , 6},
  {"LGBM_DatasetSetFeatureNames_R"    , (DL_FUNC)&LGBM_DatasetSetFeatureNames_R    , 3},
  {"LGBM_DatasetGetFeatureNames_R"    , (DL_FUNC)&LGBM_DatasetGetFeatureNames_R    , 5},
  {"LGBM_DatasetSaveBinary_R"         , (DL_FUNC)&LGBM_DatasetSaveBinary_R         , 3},
  {"LGBM_DatasetFree_R"               , (DL_FUNC)&LGBM_DatasetFree_R               , 2},
  {"LGBM_DatasetSetField_R"           , (DL_FUNC)&LGBM_DatasetSetField_R           , 5},
  {"LGBM_DatasetGetFieldSize_R"       , (DL_FUNC)&LGBM_DatasetGetFieldSize_R       , 4},
  {"LGBM_DatasetGetField_R"           , (DL_FUNC)&LGBM_DatasetGetField_R           , 4},
  {"LGBM_DatasetUpdateParam_R"        , (DL_FUNC)&LGBM_DatasetUpdateParam_R        , 3},
  {"LGBM_DatasetGetNumData_R"         , (DL_FUNC)&LGBM_DatasetGetNumData_R         , 3},
  {"LGBM_DatasetGetNumFeature_R"      , (DL_FUNC)&LGBM_DatasetGetNumFeature_R      , 3},
  {"LGBM_BoosterCreate_R"             , (DL_FUNC)&LGBM_BoosterCreate_R             , 4},
  {"LGBM_GPBoosterCreate_R"           , (DL_FUNC)&LGBM_GPBoosterCreate_R           , 5},
  {"LGBM_BoosterFree_R"               , (DL_FUNC)&LGBM_BoosterFree_R               , 2},
  {"LGBM_BoosterCreateFromModelfile_R", (DL_FUNC)&LGBM_BoosterCreateFromModelfile_R, 3},
  {"LGBM_BoosterLoadModelFromString_R", (DL_FUNC)&LGBM_BoosterLoadModelFromString_R, 3},
  {"LGBM_BoosterMerge_R"              , (DL_FUNC)&LGBM_BoosterMerge_R              , 3},
  {"LGBM_BoosterAddValidData_R"       , (DL_FUNC)&LGBM_BoosterAddValidData_R       , 3},
  {"LGBM_BoosterResetTrainingData_R"  , (DL_FUNC)&LGBM_BoosterResetTrainingData_R  , 3},
  {"LGBM_BoosterResetParameter_R"     , (DL_FUNC)&LGBM_BoosterResetParameter_R     , 3},
  {"LGBM_BoosterGetNumClasses_R"      , (DL_FUNC)&LGBM_BoosterGetNumClasses_R      , 3},
  {"LGBM_BoosterUpdateOneIter_R"      , (DL_FUNC)&LGBM_BoosterUpdateOneIter_R      , 2},
  {"LGBM_BoosterUpdateOneIterCustom_R", (DL_FUNC)&LGBM_BoosterUpdateOneIterCustom_R, 5},
  {"LGBM_BoosterRollbackOneIter_R"    , (DL_FUNC)&LGBM_BoosterRollbackOneIter_R    , 2},
  {"LGBM_BoosterGetCurrentIteration_R", (DL_FUNC)&LGBM_BoosterGetCurrentIteration_R, 3},
  {"LGBM_BoosterGetEvalNames_R"       , (DL_FUNC)&LGBM_BoosterGetEvalNames_R       , 5},
  {"LGBM_BoosterGetEval_R"            , (DL_FUNC)&LGBM_BoosterGetEval_R            , 4},
  {"LGBM_BoosterGetNumPredict_R"      , (DL_FUNC)&LGBM_BoosterGetNumPredict_R      , 4},
  {"LGBM_BoosterGetPredict_R"         , (DL_FUNC)&LGBM_BoosterGetPredict_R         , 4},
  {"LGBM_BoosterPredictForFile_R"     , (DL_FUNC)&LGBM_BoosterPredictForFile_R     , 10},
  {"LGBM_BoosterCalcNumPredict_R"     , (DL_FUNC)&LGBM_BoosterCalcNumPredict_R     , 8},
  {"LGBM_BoosterPredictForCSC_R"      , (DL_FUNC)&LGBM_BoosterPredictForCSC_R      , 14},
  {"LGBM_BoosterPredictForMat_R"      , (DL_FUNC)&LGBM_BoosterPredictForMat_R      , 11},
  {"LGBM_BoosterSaveModel_R"          , (DL_FUNC)&LGBM_BoosterSaveModel_R          , 4},
  {"LGBM_BoosterSaveModelToString_R"  , (DL_FUNC)&LGBM_BoosterSaveModelToString_R  , 6},
  {"LGBM_BoosterDumpModel_R"          , (DL_FUNC)&LGBM_BoosterDumpModel_R          , 6},
  {NULL, NULL, 0}
};

void R_init_gpboost(DllInfo* dll) {
	R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
	R_useDynamicSymbols(dll, FALSE);
}
#endif //end GPB_R_BUILD
