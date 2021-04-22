/*!
 * Original work Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
 * Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
 */

#include "gpboost_R.h"

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>

#include <R_ext/Rdynload.h>

#define R_NO_REMAP
#define R_USE_C99_IN_CXX
#include <R_ext/Error.h>

#include <string>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#define COL_MAJOR (0)

#define R_API_BEGIN() \
  try {
#define R_API_END() } \
  catch(std::exception& ex) { R_INT_PTR(call_state)[0] = -1; LGBM_SetLastError(ex.what()); return call_state;} \
  catch(std::string& ex) { R_INT_PTR(call_state)[0] = -1; LGBM_SetLastError(ex.c_str()); return call_state; } \
  catch(...) { R_INT_PTR(call_state)[0] = -1; LGBM_SetLastError("unknown exception"); return call_state;} \
  return call_state;

#define CHECK_CALL(x) \
  if ((x) != 0) { \
    Rf_error(LGBM_GetLastError()); \
    return call_state;\
  }

using LightGBM::Common::Join;
using LightGBM::Common::Split;
using LightGBM::Log;

LGBM_SE EncodeChar(LGBM_SE dest, const char* src, LGBM_SE buf_len, LGBM_SE actual_len, size_t str_len) {
	if (str_len > INT32_MAX) {
		Log::Fatal("Don't support large string in R-package");
	}
	R_INT_PTR(actual_len)[0] = static_cast<int>(str_len);
	if (R_AS_INT(buf_len) < static_cast<int>(str_len)) {
		return dest;
	}
	auto ptr = R_CHAR_PTR(dest);
	std::memcpy(ptr, src, str_len);
	return dest;
}

LGBM_SE LGBM_GetLastError_R(LGBM_SE buf_len, LGBM_SE actual_len, LGBM_SE err_msg) {
	return EncodeChar(err_msg, LGBM_GetLastError(), buf_len, actual_len, std::strlen(LGBM_GetLastError()) + 1);
}

LGBM_SE LGBM_DatasetCreateFromFile_R(LGBM_SE filename,
	LGBM_SE parameters,
	LGBM_SE reference,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	DatasetHandle handle = nullptr;
	CHECK_CALL(LGBM_DatasetCreateFromFile(R_CHAR_PTR(filename), R_CHAR_PTR(parameters),
		R_GET_PTR(reference), &handle));
	R_SET_PTR(out, handle);
	R_API_END();
}

LGBM_SE LGBM_DatasetCreateFromCSC_R(LGBM_SE indptr,
	LGBM_SE indices,
	LGBM_SE data,
	LGBM_SE num_indptr,
	LGBM_SE nelem,
	LGBM_SE num_row,
	LGBM_SE parameters,
	LGBM_SE reference,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	const int* p_indptr = R_INT_PTR(indptr);
	const int* p_indices = R_INT_PTR(indices);
	const double* p_data = R_REAL_PTR(data);

	int64_t nindptr = static_cast<int64_t>(R_AS_INT(num_indptr));
	int64_t ndata = static_cast<int64_t>(R_AS_INT(nelem));
	int64_t nrow = static_cast<int64_t>(R_AS_INT(num_row));
	DatasetHandle handle = nullptr;
	CHECK_CALL(LGBM_DatasetCreateFromCSC(p_indptr, C_API_DTYPE_INT32, p_indices,
		p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
		nrow, R_CHAR_PTR(parameters), R_GET_PTR(reference), &handle));
	R_SET_PTR(out, handle);
	R_API_END();
}

LGBM_SE LGBM_DatasetCreateFromMat_R(LGBM_SE data,
	LGBM_SE num_row,
	LGBM_SE num_col,
	LGBM_SE parameters,
	LGBM_SE reference,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int32_t nrow = static_cast<int32_t>(R_AS_INT(num_row));
	int32_t ncol = static_cast<int32_t>(R_AS_INT(num_col));
	double* p_mat = R_REAL_PTR(data);
	DatasetHandle handle = nullptr;
	CHECK_CALL(LGBM_DatasetCreateFromMat(p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
		R_CHAR_PTR(parameters), R_GET_PTR(reference), &handle));
	R_SET_PTR(out, handle);
	R_API_END();
}

LGBM_SE LGBM_DatasetGetSubset_R(LGBM_SE handle,
	LGBM_SE used_row_indices,
	LGBM_SE len_used_row_indices,
	LGBM_SE parameters,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int len = R_AS_INT(len_used_row_indices);
	std::vector<int> idxvec(len);
	// convert from one-based to  zero-based index
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
	for (int i = 0; i < len; ++i) {
		idxvec[i] = R_INT_PTR(used_row_indices)[i] - 1;
	}
	DatasetHandle res = nullptr;
	CHECK_CALL(LGBM_DatasetGetSubset(R_GET_PTR(handle),
		idxvec.data(), len, R_CHAR_PTR(parameters),
		&res));
	R_SET_PTR(out, res);
	R_API_END();
}

LGBM_SE LGBM_DatasetSetFeatureNames_R(LGBM_SE handle,
	LGBM_SE feature_names,
	LGBM_SE call_state) {
	R_API_BEGIN();
	auto vec_names = Split(R_CHAR_PTR(feature_names), '\t');
	std::vector<const char*> vec_sptr;
	int len = static_cast<int>(vec_names.size());
	for (int i = 0; i < len; ++i) {
		vec_sptr.push_back(vec_names[i].c_str());
	}
	CHECK_CALL(LGBM_DatasetSetFeatureNames(R_GET_PTR(handle),
		vec_sptr.data(), len));
	R_API_END();
}

LGBM_SE LGBM_DatasetGetFeatureNames_R(LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE feature_names,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int len = 0;
	CHECK_CALL(LGBM_DatasetGetNumFeature(R_GET_PTR(handle), &len));
	const size_t reserved_string_size = 256;
	std::vector<std::vector<char>> names(len);
	std::vector<char*> ptr_names(len);
	for (int i = 0; i < len; ++i) {
		names[i].resize(reserved_string_size);
		ptr_names[i] = names[i].data();
	}
	int out_len;
	size_t required_string_size;
	CHECK_CALL(
		LGBM_DatasetGetFeatureNames(
			R_GET_PTR(handle),
			len, &out_len,
			reserved_string_size, &required_string_size,
			ptr_names.data()));
	CHECK_EQ(len, out_len);
	CHECK_GE(reserved_string_size, required_string_size);
	auto merge_str = Join<char*>(ptr_names, "\t");
	EncodeChar(feature_names, merge_str.c_str(), buf_len, actual_len, merge_str.size() + 1);
	R_API_END();
}

LGBM_SE LGBM_DatasetSaveBinary_R(LGBM_SE handle,
	LGBM_SE filename,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(LGBM_DatasetSaveBinary(R_GET_PTR(handle),
		R_CHAR_PTR(filename)));
	R_API_END();
}

LGBM_SE LGBM_DatasetFree_R(LGBM_SE handle,
	LGBM_SE call_state) {
	R_API_BEGIN();
	if (R_GET_PTR(handle) != nullptr) {
		CHECK_CALL(LGBM_DatasetFree(R_GET_PTR(handle)));
		R_SET_PTR(handle, nullptr);
	}
	R_API_END();
}

LGBM_SE LGBM_DatasetSetField_R(LGBM_SE handle,
	LGBM_SE field_name,
	LGBM_SE field_data,
	LGBM_SE num_element,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int len = static_cast<int>(R_AS_INT(num_element));
	const char* name = R_CHAR_PTR(field_name);
	if (!strcmp("group", name) || !strcmp("query", name)) {
		std::vector<int32_t> vec(len);
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
		for (int i = 0; i < len; ++i) {
			vec[i] = static_cast<int32_t>(R_INT_PTR(field_data)[i]);
		}
		CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, vec.data(), len, C_API_DTYPE_INT32));
	}
	else if (!strcmp("init_score", name)) {
		CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, R_REAL_PTR(field_data), len, C_API_DTYPE_FLOAT64));
	}
	else {
		std::vector<float> vec(len);
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
		for (int i = 0; i < len; ++i) {
			vec[i] = static_cast<float>(R_REAL_PTR(field_data)[i]);
		}
		CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, vec.data(), len, C_API_DTYPE_FLOAT32));
	}
	R_API_END();
}

LGBM_SE LGBM_DatasetGetField_R(LGBM_SE handle,
	LGBM_SE field_name,
	LGBM_SE field_data,
	LGBM_SE call_state) {
	R_API_BEGIN();
	const char* name = R_CHAR_PTR(field_name);
	int out_len = 0;
	int out_type = 0;
	const void* res;
	CHECK_CALL(LGBM_DatasetGetField(R_GET_PTR(handle), name, &out_len, &res, &out_type));

	if (!strcmp("group", name) || !strcmp("query", name)) {
		auto p_data = reinterpret_cast<const int32_t*>(res);
		// convert from boundaries to size
#pragma omp parallel for schedule(static, 512) if (out_len >= 1024)
		for (int i = 0; i < out_len - 1; ++i) {
			R_INT_PTR(field_data)[i] = p_data[i + 1] - p_data[i];
		}
	}
	else if (!strcmp("init_score", name)) {
		auto p_data = reinterpret_cast<const double*>(res);
#pragma omp parallel for schedule(static, 512) if (out_len >= 1024)
		for (int i = 0; i < out_len; ++i) {
			R_REAL_PTR(field_data)[i] = p_data[i];
		}
	}
	else {
		auto p_data = reinterpret_cast<const float*>(res);
#pragma omp parallel for schedule(static, 512) if (out_len >= 1024)
		for (int i = 0; i < out_len; ++i) {
			R_REAL_PTR(field_data)[i] = p_data[i];
		}
	}
	R_API_END();
}

LGBM_SE LGBM_DatasetGetFieldSize_R(LGBM_SE handle,
	LGBM_SE field_name,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	const char* name = R_CHAR_PTR(field_name);
	int out_len = 0;
	int out_type = 0;
	const void* res;
	CHECK_CALL(LGBM_DatasetGetField(R_GET_PTR(handle), name, &out_len, &res, &out_type));
	if (!strcmp("group", name) || !strcmp("query", name)) {
		out_len -= 1;
	}
	R_INT_PTR(out)[0] = static_cast<int>(out_len);
	R_API_END();
}

LGBM_SE LGBM_DatasetUpdateParamChecking_R(LGBM_SE old_params,
	LGBM_SE new_params,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(LGBM_DatasetUpdateParamChecking(R_CHAR_PTR(old_params), R_CHAR_PTR(new_params)));
	R_API_END();
}

LGBM_SE LGBM_DatasetGetNumData_R(LGBM_SE handle, LGBM_SE out,
	LGBM_SE call_state) {
	int nrow;
	R_API_BEGIN();
	CHECK_CALL(LGBM_DatasetGetNumData(R_GET_PTR(handle), &nrow));
	R_INT_PTR(out)[0] = static_cast<int>(nrow);
	R_API_END();
}

LGBM_SE LGBM_DatasetGetNumFeature_R(LGBM_SE handle,
	LGBM_SE out,
	LGBM_SE call_state) {
	int nfeature;
	R_API_BEGIN();
	CHECK_CALL(LGBM_DatasetGetNumFeature(R_GET_PTR(handle), &nfeature));
	R_INT_PTR(out)[0] = static_cast<int>(nfeature);
	R_API_END();
}

// --- start Booster interfaces

LGBM_SE LGBM_BoosterFree_R(LGBM_SE handle,
	LGBM_SE call_state) {
	R_API_BEGIN();
	if (R_GET_PTR(handle) != nullptr) {
		CHECK_CALL(LGBM_BoosterFree(R_GET_PTR(handle)));
		R_SET_PTR(handle, nullptr);
	}
	R_API_END();
}

LGBM_SE LGBM_BoosterCreate_R(LGBM_SE train_data,
	LGBM_SE parameters,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	BoosterHandle handle = nullptr;
	CHECK_CALL(LGBM_BoosterCreate(R_GET_PTR(train_data), R_CHAR_PTR(parameters), &handle));
	R_SET_PTR(out, handle);
	R_API_END();
}

LGBM_SE LGBM_GPBoosterCreate_R(LGBM_SE train_data,
	LGBM_SE parameters,
	LGBM_SE re_model,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	BoosterHandle handle = nullptr;
	CHECK_CALL(LGBM_GPBoosterCreate(R_GET_PTR(train_data), R_CHAR_PTR(parameters), R_GET_PTR(re_model), &handle));
	R_SET_PTR(out, handle);
	R_API_END();
}

LGBM_SE LGBM_BoosterCreateFromModelfile_R(LGBM_SE filename,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int out_num_iterations = 0;
	BoosterHandle handle = nullptr;
	CHECK_CALL(LGBM_BoosterCreateFromModelfile(R_CHAR_PTR(filename), &out_num_iterations, &handle));
	R_SET_PTR(out, handle);
	R_API_END();
}

LGBM_SE LGBM_BoosterLoadModelFromString_R(LGBM_SE model_str,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int out_num_iterations = 0;
	BoosterHandle handle = nullptr;
	CHECK_CALL(LGBM_BoosterLoadModelFromString(R_CHAR_PTR(model_str), &out_num_iterations, &handle));
	R_SET_PTR(out, handle);
	R_API_END();
}

LGBM_SE LGBM_BoosterMerge_R(LGBM_SE handle,
	LGBM_SE other_handle,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterMerge(R_GET_PTR(handle), R_GET_PTR(other_handle)));
	R_API_END();
}

LGBM_SE LGBM_BoosterAddValidData_R(LGBM_SE handle,
	LGBM_SE valid_data,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterAddValidData(R_GET_PTR(handle), R_GET_PTR(valid_data)));
	R_API_END();
}

LGBM_SE LGBM_BoosterResetTrainingData_R(LGBM_SE handle,
	LGBM_SE train_data,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterResetTrainingData(R_GET_PTR(handle), R_GET_PTR(train_data)));
	R_API_END();
}

LGBM_SE LGBM_BoosterResetParameter_R(LGBM_SE handle,
	LGBM_SE parameters,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterResetParameter(R_GET_PTR(handle), R_CHAR_PTR(parameters)));
	R_API_END();
}

LGBM_SE LGBM_BoosterGetNumClasses_R(LGBM_SE handle,
	LGBM_SE out,
	LGBM_SE call_state) {
	int num_class;
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterGetNumClasses(R_GET_PTR(handle), &num_class));
	R_INT_PTR(out)[0] = static_cast<int>(num_class);
	R_API_END();
}

LGBM_SE LGBM_BoosterUpdateOneIter_R(LGBM_SE handle,
	LGBM_SE call_state) {
	int is_finished = 0;
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterUpdateOneIter(R_GET_PTR(handle), &is_finished));
	R_API_END();
}

LGBM_SE LGBM_BoosterUpdateOneIterCustom_R(LGBM_SE handle,
	LGBM_SE grad,
	LGBM_SE hess,
	LGBM_SE len,
	LGBM_SE call_state) {
	int is_finished = 0;
	R_API_BEGIN();
	int int_len = R_AS_INT(len);
	std::vector<float> tgrad(int_len), thess(int_len);
#pragma omp parallel for schedule(static, 512) if (int_len >= 1024)
	for (int j = 0; j < int_len; ++j) {
		tgrad[j] = static_cast<float>(R_REAL_PTR(grad)[j]);
		thess[j] = static_cast<float>(R_REAL_PTR(hess)[j]);
	}
	CHECK_CALL(LGBM_BoosterUpdateOneIterCustom(R_GET_PTR(handle), tgrad.data(), thess.data(), &is_finished));
	R_API_END();
}

LGBM_SE LGBM_BoosterRollbackOneIter_R(LGBM_SE handle,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterRollbackOneIter(R_GET_PTR(handle)));
	R_API_END();
}

LGBM_SE LGBM_BoosterGetCurrentIteration_R(LGBM_SE handle,
	LGBM_SE out,
	LGBM_SE call_state) {
	int out_iteration;
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterGetCurrentIteration(R_GET_PTR(handle), &out_iteration));
	R_INT_PTR(out)[0] = static_cast<int>(out_iteration);
	R_API_END();
}

LGBM_SE LGBM_BoosterGetUpperBoundValue_R(LGBM_SE handle,
	LGBM_SE out_result,
	LGBM_SE call_state) {
	R_API_BEGIN();
	double* ptr_ret = R_REAL_PTR(out_result);
	CHECK_CALL(LGBM_BoosterGetUpperBoundValue(R_GET_PTR(handle), ptr_ret));
	R_API_END();
}

LGBM_SE LGBM_BoosterGetLowerBoundValue_R(LGBM_SE handle,
	LGBM_SE out_result,
	LGBM_SE call_state) {
	R_API_BEGIN();
	double* ptr_ret = R_REAL_PTR(out_result);
	CHECK_CALL(LGBM_BoosterGetLowerBoundValue(R_GET_PTR(handle), ptr_ret));
	R_API_END();
}

LGBM_SE LGBM_BoosterGetEvalNames_R(LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE eval_names,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int len;
	CHECK_CALL(LGBM_BoosterGetEvalCounts(R_GET_PTR(handle), &len));

	const size_t reserved_string_size = 128;
	std::vector<std::vector<char>> names(len);
	std::vector<char*> ptr_names(len);
	for (int i = 0; i < len; ++i) {
		names[i].resize(reserved_string_size);
		ptr_names[i] = names[i].data();
	}

	int out_len;
	size_t required_string_size;
	CHECK_CALL(
		LGBM_BoosterGetEvalNames(
			R_GET_PTR(handle),
			len, &out_len,
			reserved_string_size, &required_string_size,
			ptr_names.data()));
	CHECK_EQ(out_len, len);
	CHECK_GE(reserved_string_size, required_string_size);
	auto merge_names = Join<char*>(ptr_names, "\t");
	EncodeChar(eval_names, merge_names.c_str(), buf_len, actual_len, merge_names.size() + 1);
	R_API_END();
}

LGBM_SE LGBM_BoosterGetEval_R(LGBM_SE handle,
	LGBM_SE data_idx,
	LGBM_SE out_result,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int len;
	CHECK_CALL(LGBM_BoosterGetEvalCounts(R_GET_PTR(handle), &len));
	double* ptr_ret = R_REAL_PTR(out_result);
	int out_len;
	CHECK_CALL(LGBM_BoosterGetEval(R_GET_PTR(handle), R_AS_INT(data_idx), &out_len, ptr_ret));
	CHECK_EQ(out_len, len);
	R_API_END();
}

LGBM_SE LGBM_BoosterGetNumPredict_R(LGBM_SE handle,
	LGBM_SE data_idx,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int64_t len;
	CHECK_CALL(LGBM_BoosterGetNumPredict(R_GET_PTR(handle), R_AS_INT(data_idx), &len));
	R_INT_PTR(out)[0] = static_cast<int>(len);
	R_API_END();
}

LGBM_SE LGBM_BoosterGetPredict_R(LGBM_SE handle,
	LGBM_SE data_idx,
	LGBM_SE out_result,
	LGBM_SE call_state) {
	R_API_BEGIN();
	double* ptr_ret = R_REAL_PTR(out_result);
	int64_t out_len;
	CHECK_CALL(LGBM_BoosterGetPredict(R_GET_PTR(handle), R_AS_INT(data_idx), &out_len, ptr_ret));
	R_API_END();
}

int GetPredictType(LGBM_SE is_rawscore, LGBM_SE is_leafidx, LGBM_SE is_predcontrib) {
	int pred_type = C_API_PREDICT_NORMAL;
	if (R_AS_INT(is_rawscore)) {
		pred_type = C_API_PREDICT_RAW_SCORE;
	}
	if (R_AS_INT(is_leafidx)) {
		pred_type = C_API_PREDICT_LEAF_INDEX;
	}
	if (R_AS_INT(is_predcontrib)) {
		pred_type = C_API_PREDICT_CONTRIB;
	}
	return pred_type;
}

LGBM_SE LGBM_BoosterPredictForFile_R(LGBM_SE handle,
	LGBM_SE data_filename,
	LGBM_SE data_has_header,
	LGBM_SE is_rawscore,
	LGBM_SE is_leafidx,
	LGBM_SE is_predcontrib,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE parameter,
	LGBM_SE result_filename,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
	CHECK_CALL(LGBM_BoosterPredictForFile(R_GET_PTR(handle), R_CHAR_PTR(data_filename),
		R_AS_INT(data_has_header), pred_type, R_AS_INT(start_iteration), R_AS_INT(num_iteration), R_CHAR_PTR(parameter),
		R_CHAR_PTR(result_filename)));
	R_API_END();
}

LGBM_SE LGBM_BoosterCalcNumPredict_R(LGBM_SE handle,
	LGBM_SE num_row,
	LGBM_SE is_rawscore,
	LGBM_SE is_leafidx,
	LGBM_SE is_predcontrib,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE out_len,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
	int64_t len = 0;
	CHECK_CALL(LGBM_BoosterCalcNumPredict(R_GET_PTR(handle), R_AS_INT(num_row),
		pred_type, R_AS_INT(start_iteration), R_AS_INT(num_iteration), &len));
	R_INT_PTR(out_len)[0] = static_cast<int>(len);
	R_API_END();
}

LGBM_SE LGBM_BoosterPredictForCSC_R(LGBM_SE handle,
	LGBM_SE indptr,
	LGBM_SE indices,
	LGBM_SE data,
	LGBM_SE num_indptr,
	LGBM_SE nelem,
	LGBM_SE num_row,
	LGBM_SE is_rawscore,
	LGBM_SE is_leafidx,
	LGBM_SE is_predcontrib,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE parameter,
	LGBM_SE out_result,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);

	const int* p_indptr = R_INT_PTR(indptr);
	const int* p_indices = R_INT_PTR(indices);
	const double* p_data = R_REAL_PTR(data);

	int64_t nindptr = R_AS_INT(num_indptr);
	int64_t ndata = R_AS_INT(nelem);
	int64_t nrow = R_AS_INT(num_row);
	double* ptr_ret = R_REAL_PTR(out_result);
	int64_t out_len;
	CHECK_CALL(LGBM_BoosterPredictForCSC(R_GET_PTR(handle),
		p_indptr, C_API_DTYPE_INT32, p_indices,
		p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
		nrow, pred_type, R_AS_INT(start_iteration), R_AS_INT(num_iteration), R_CHAR_PTR(parameter), &out_len, ptr_ret));
	R_API_END();
}

LGBM_SE LGBM_BoosterPredictForMat_R(LGBM_SE handle,
	LGBM_SE data,
	LGBM_SE num_row,
	LGBM_SE num_col,
	LGBM_SE is_rawscore,
	LGBM_SE is_leafidx,
	LGBM_SE is_predcontrib,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE parameter,
	LGBM_SE out_result,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);

	int32_t nrow = R_AS_INT(num_row);
	int32_t ncol = R_AS_INT(num_col);

	const double* p_mat = R_REAL_PTR(data);
	double* ptr_ret = R_REAL_PTR(out_result);
	int64_t out_len;
	CHECK_CALL(LGBM_BoosterPredictForMat(R_GET_PTR(handle),
		p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
		pred_type, R_AS_INT(start_iteration), R_AS_INT(num_iteration), R_CHAR_PTR(parameter), &out_len, ptr_ret));

	R_API_END();
}

LGBM_SE LGBM_BoosterSaveModel_R(LGBM_SE handle,
	LGBM_SE num_iteration,
	LGBM_SE feature_importance_type,
	LGBM_SE filename,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(LGBM_BoosterSaveModel(R_GET_PTR(handle), 0, R_AS_INT(num_iteration), R_AS_INT(feature_importance_type), R_CHAR_PTR(filename)));
	R_API_END();
}

LGBM_SE LGBM_BoosterSaveModelToString_R(LGBM_SE handle,
	LGBM_SE start_iteration,
	LGBM_SE num_iteration,
	LGBM_SE feature_importance_type,
	LGBM_SE buffer_len,
	LGBM_SE actual_len,
	LGBM_SE out_str,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int64_t out_len = 0;
	std::vector<char> inner_char_buf(R_AS_INT(buffer_len));
	CHECK_CALL(LGBM_BoosterSaveModelToString(R_GET_PTR(handle), R_AS_INT(start_iteration), R_AS_INT(num_iteration), R_AS_INT(feature_importance_type), R_AS_INT(buffer_len), &out_len, inner_char_buf.data()));
	EncodeChar(out_str, inner_char_buf.data(), buffer_len, actual_len, static_cast<size_t>(out_len));
	R_API_END();
}

LGBM_SE LGBM_BoosterDumpModel_R(LGBM_SE handle,
	LGBM_SE num_iteration,
	LGBM_SE feature_importance_type,
	LGBM_SE buffer_len,
	LGBM_SE actual_len,
	LGBM_SE out_str,
	LGBM_SE call_state) {
	R_API_BEGIN();
	int64_t out_len = 0;
	std::vector<char> inner_char_buf(R_AS_INT(buffer_len));
	CHECK_CALL(LGBM_BoosterDumpModel(R_GET_PTR(handle), 0, R_AS_INT(num_iteration), R_AS_INT(feature_importance_type), R_AS_INT(buffer_len), &out_len, inner_char_buf.data()));
	EncodeChar(out_str, inner_char_buf.data(), buffer_len, actual_len, static_cast<size_t>(out_len));
	R_API_END();
}

// Below here are REModel / GPModel related functions

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
	LGBM_SE cov_fct_taper_range,
	LGBM_SE vecchia_approx,
	LGBM_SE num_neighbors,
	LGBM_SE vecchia_ordering,
	LGBM_SE vecchia_pred_type,
	LGBM_SE num_neighbors_pred,
	LGBM_SE likelihood,
	LGBM_SE out,
	LGBM_SE call_state) {
	R_API_BEGIN();
	REModelHandle handle = nullptr;
	CHECK_CALL(GPB_CreateREModel(R_AS_INT(ndata),
		R_INT_PTR(cluster_ids_data),
		R_CHAR_PTR(re_group_data),
		R_AS_INT(num_re_group),
		R_REAL_PTR(re_group_rand_coef_data),
		R_INT_PTR(ind_effect_group_rand_coef),
		R_AS_INT(num_re_group_rand_coef),
		R_AS_INT(num_gp),
		R_REAL_PTR(gp_coords_data),
		R_AS_INT(dim_gp_coords),
		R_REAL_PTR(gp_rand_coef_data),
		R_AS_INT(num_gp_rand_coef),
		R_CHAR_PTR(cov_fct),
		R_AS_DOUBLE(cov_fct_shape),
		R_AS_DOUBLE(cov_fct_taper_range),
		R_AS_BOOL(vecchia_approx),
		R_AS_INT(num_neighbors),
		R_CHAR_PTR(vecchia_ordering),
		R_CHAR_PTR(vecchia_pred_type),
		R_AS_INT(num_neighbors_pred),
		R_CHAR_PTR(likelihood), &handle));
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
	LGBM_SE calc_std_dev,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_SetOptimConfig(R_GET_PTR(handle), R_REAL_PTR(init_cov_pars),
		R_AS_DOUBLE(lr), R_AS_DOUBLE(acc_rate_cov), R_AS_INT(max_iter),
		R_AS_DOUBLE(delta_rel_conv), R_AS_BOOL(use_nesterov_acc),
		R_AS_INT(nesterov_schedule_version), R_AS_BOOL(trace), R_CHAR_PTR(optimizer),
		R_AS_INT(momentum_offset), R_CHAR_PTR(convergence_criterion), R_AS_BOOL(calc_std_dev)));
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
	LGBM_SE fixed_effects,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_OptimCovPar(R_GET_PTR(handle), R_REAL_PTR(y_data),
		R_REAL_PTR(fixed_effects)));
	R_API_END();
}

LGBM_SE GPB_OptimLinRegrCoefCovPar_R(LGBM_SE handle,
	LGBM_SE y_data,
	LGBM_SE covariate_data,
	LGBM_SE num_covariates,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_OptimLinRegrCoefCovPar(R_GET_PTR(handle), R_REAL_PTR(y_data), R_REAL_PTR(covariate_data),
		R_AS_INT(num_covariates)));
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

LGBM_SE GPB_GetInitCovPar_R(LGBM_SE handle,
	LGBM_SE init_cov_pars,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_GetInitCovPar(R_GET_PTR(handle), R_REAL_PTR(init_cov_pars)));
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
	LGBM_SE predict_var,
	LGBM_SE predict_response,
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
	LGBM_SE fixed_effects,
	LGBM_SE fixed_effects_pred,
	LGBM_SE out_predict,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_PredictREModel(R_GET_PTR(handle), R_REAL_PTR(y_data),
		R_AS_INT(num_data_pred), R_REAL_PTR(out_predict),
		R_AS_BOOL(predict_cov_mat), R_AS_BOOL(predict_var), R_AS_BOOL(predict_response), R_INT_PTR(cluster_ids_data_pred),
		R_CHAR_PTR(re_group_data_pred), R_REAL_PTR(re_group_rand_coef_data_pred),
		R_REAL_PTR(gp_coords_pred), R_REAL_PTR(gp_rand_coef_data_pred), R_REAL_PTR(cov_pars),
		R_REAL_PTR(covariate_data_pred), R_AS_BOOL(use_saved_data),
		R_CHAR_PTR(vecchia_pred_type), R_AS_INT(num_neighbors_pred),
		R_REAL_PTR(fixed_effects), R_REAL_PTR(fixed_effects_pred)));
	R_API_END();
}

LGBM_SE GPB_GetLikelihoodName_R(LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE ll_name,
	LGBM_SE call_state) {
	R_API_BEGIN();
	std::vector<char> name(128);
	int num_char;
	CHECK_CALL(GPB_GetLikelihoodName(R_GET_PTR(handle), name.data(), num_char));
	EncodeChar(ll_name, name.data(), buf_len, actual_len, num_char);
	R_API_END();
}

LGBM_SE GPB_GetOptimizerCovPars_R(LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE opt_name,
	LGBM_SE call_state) {
	R_API_BEGIN();
	std::vector<char> name(128);
	int num_char;
	CHECK_CALL(GPB_GetOptimizerCovPars(R_GET_PTR(handle), name.data(), num_char));
	EncodeChar(opt_name, name.data(), buf_len, actual_len, num_char);
	R_API_END();
}

LGBM_SE GPB_GetOptimizerCoef_R(LGBM_SE handle,
	LGBM_SE buf_len,
	LGBM_SE actual_len,
	LGBM_SE opt_name,
	LGBM_SE call_state) {
	R_API_BEGIN();
	std::vector<char> name(128);
	int num_char;
	CHECK_CALL(GPB_GetOptimizerCoef(R_GET_PTR(handle), name.data(), num_char));
	EncodeChar(opt_name, name.data(), buf_len, actual_len, num_char);
	R_API_END();
}

LGBM_SE GPB_SetLikelihood_R(LGBM_SE handle,
	LGBM_SE likelihood,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_SetLikelihood(R_GET_PTR(handle), R_CHAR_PTR(likelihood)));
	R_API_END();
}

LGBM_SE GPB_GetResponseData_R(LGBM_SE handle,
	LGBM_SE response_data,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_GetResponseData(R_GET_PTR(handle), R_REAL_PTR(response_data)));
	R_API_END();
}

LGBM_SE GPB_GetCovariateData_R(LGBM_SE handle,
	LGBM_SE covariate_data,
	LGBM_SE call_state) {
	R_API_BEGIN();
	CHECK_CALL(GPB_GetCovariateData(R_GET_PTR(handle), R_REAL_PTR(covariate_data)));
	R_API_END();
}

// .Call() calls
static const R_CallMethodDef CallEntries[] = {
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
  {"LGBM_DatasetUpdateParamChecking_R", (DL_FUNC)&LGBM_DatasetUpdateParamChecking_R, 3},
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
  {"LGBM_BoosterGetUpperBoundValue_R" , (DL_FUNC)&LGBM_BoosterGetUpperBoundValue_R , 3},
  {"LGBM_BoosterGetLowerBoundValue_R" , (DL_FUNC)&LGBM_BoosterGetLowerBoundValue_R , 3},
  {"LGBM_BoosterGetEvalNames_R"       , (DL_FUNC)&LGBM_BoosterGetEvalNames_R       , 5},
  {"LGBM_BoosterGetEval_R"            , (DL_FUNC)&LGBM_BoosterGetEval_R            , 4},
  {"LGBM_BoosterGetNumPredict_R"      , (DL_FUNC)&LGBM_BoosterGetNumPredict_R      , 4},
  {"LGBM_BoosterGetPredict_R"         , (DL_FUNC)&LGBM_BoosterGetPredict_R         , 4},
  {"LGBM_BoosterPredictForFile_R"     , (DL_FUNC)&LGBM_BoosterPredictForFile_R     , 11},
  {"LGBM_BoosterCalcNumPredict_R"     , (DL_FUNC)&LGBM_BoosterCalcNumPredict_R     , 9},
  {"LGBM_BoosterPredictForCSC_R"      , (DL_FUNC)&LGBM_BoosterPredictForCSC_R      , 15},
  {"LGBM_BoosterPredictForMat_R"      , (DL_FUNC)&LGBM_BoosterPredictForMat_R      , 12},
  {"LGBM_BoosterSaveModel_R"          , (DL_FUNC)&LGBM_BoosterSaveModel_R          , 5},
  {"LGBM_BoosterSaveModelToString_R"  , (DL_FUNC)&LGBM_BoosterSaveModelToString_R  , 8},
  {"LGBM_BoosterDumpModel_R"          , (DL_FUNC)&LGBM_BoosterDumpModel_R          , 7},
  {"GPB_CreateREModel_R"              , (DL_FUNC)&GPB_CreateREModel_R              , 22},
  {"GPB_REModelFree_R"                , (DL_FUNC)&GPB_REModelFree_R                , 2},
  {"GPB_SetOptimConfig_R"             , (DL_FUNC)&GPB_SetOptimConfig_R             , 14},
  {"GPB_SetOptimCoefConfig_R"         , (DL_FUNC)&GPB_SetOptimCoefConfig_R         , 7},
  {"GPB_OptimCovPar_R"                , (DL_FUNC)&GPB_OptimCovPar_R                , 4},
  {"GPB_OptimLinRegrCoefCovPar_R"     , (DL_FUNC)&GPB_OptimLinRegrCoefCovPar_R     , 5},
  {"GPB_EvalNegLogLikelihood_R"       , (DL_FUNC)&GPB_EvalNegLogLikelihood_R       , 5},
  {"GPB_GetCovPar_R"                  , (DL_FUNC)&GPB_GetCovPar_R                  , 4},
  {"GPB_GetInitCovPar_R"              , (DL_FUNC)&GPB_GetInitCovPar_R              , 3},
  {"GPB_GetCoef_R"                    , (DL_FUNC)&GPB_GetCoef_R                    , 4},
  {"GPB_GetNumIt_R"                   , (DL_FUNC)&GPB_GetNumIt_R                   , 3},
  {"GPB_SetPredictionData_R"          , (DL_FUNC)&GPB_SetPredictionData_R          , 9},
  {"GPB_PredictREModel_R"             , (DL_FUNC)&GPB_PredictREModel_R             , 20},
  {"GPB_GetLikelihoodName_R"          , (DL_FUNC)&GPB_GetLikelihoodName_R          , 5},
  {"GPB_GetOptimizerCovPars_R"        , (DL_FUNC)&GPB_GetOptimizerCovPars_R        , 5},
  {"GPB_GetOptimizerCoef_R"           , (DL_FUNC)&GPB_GetOptimizerCoef_R           , 5},
  {"GPB_SetLikelihood_R"              , (DL_FUNC)&GPB_SetLikelihood_R              , 3},
  {"GPB_GetResponseData_R"            , (DL_FUNC)&GPB_GetResponseData_R            , 3},
  {"GPB_GetCovariateData_R"           , (DL_FUNC)&GPB_GetCovariateData_R           , 3},
  {NULL, NULL, 0}
};

void R_init_lightgbm(DllInfo* dll) {
	R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
	R_useDynamicSymbols(dll, FALSE);
}
