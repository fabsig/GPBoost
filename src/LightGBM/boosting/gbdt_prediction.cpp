/*!
* Original work Copyright (c) 2017 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/nesterov_boosting.h>

#include "gbdt.h"

namespace LightGBM {

	void GBDT::PredictRaw(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
		int early_stop_round_counter = 0;
		// set zero
		std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
		std::vector<double> pred_lag1;//used for momentum step
		const int end_iteration_for_pred = start_iteration_for_pred_ + num_iteration_for_pred_;
		for (int i = start_iteration_for_pred_; i < end_iteration_for_pred; ++i) {
			// apply momentum step
			if (use_nesterov_acc_ && i > 0) {
				if (i == 1) {//// initialize lag1 for momentum step
					pred_lag1 = std::vector<double>(num_tree_per_iteration_);
					for (int j = 0; j < num_tree_per_iteration_; ++j) {
						pred_lag1[j] = output[j];
					}
				}
				else {
					double mu = NesterovSchedule(i, momentum_schedule_version_, nesterov_acc_rate_, momentum_offset_);
					DoOneMomentumStep(output, pred_lag1.data(), (int64_t)num_tree_per_iteration_, mu);
				}
			}
			// predict all the trees for one iteration
			for (int k = 0; k < num_tree_per_iteration_; ++k) {
				output[k] += models_[i * num_tree_per_iteration_ + k]->Predict(features);
			}
			// check early stopping
			++early_stop_round_counter;
			if (early_stop->round_period == early_stop_round_counter) {
				if (early_stop->callback_function(output, num_tree_per_iteration_)) {
					return;
				}
				early_stop_round_counter = 0;
			}
		}
	}

	void GBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {
		int early_stop_round_counter = 0;
		// set zero
		std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
		// initialize for momentum step
		std::vector<double> pred_lag1;
		if (use_nesterov_acc_) {
			pred_lag1 = std::vector<double>(num_tree_per_iteration_);
			for (int i = 0; i < num_tree_per_iteration_; ++i) {
				pred_lag1[i] = output[i];
			}
		}
		const int end_iteration_for_pred = start_iteration_for_pred_ + num_iteration_for_pred_;
		for (int i = start_iteration_for_pred_; i < end_iteration_for_pred; ++i) {
			// apply momentum step
			if (use_nesterov_acc_) {
				double mu = NesterovSchedule(i, momentum_schedule_version_, nesterov_acc_rate_, momentum_offset_);
				DoOneMomentumStep(output, pred_lag1.data(), (int64_t)num_tree_per_iteration_, mu);
			}
			// predict all the trees for one iteration
			for (int k = 0; k < num_tree_per_iteration_; ++k) {
				output[k] += models_[i * num_tree_per_iteration_ + k]->PredictByMap(features);
			}
			// check early stopping
			++early_stop_round_counter;
			if (early_stop->round_period == early_stop_round_counter) {
				if (early_stop->callback_function(output, num_tree_per_iteration_)) {
					return;
				}
				early_stop_round_counter = 0;
			}
		}
	}

	void GBDT::Predict(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
		PredictRaw(features, output, early_stop);
		if (average_output_) {
			for (int k = 0; k < num_tree_per_iteration_; ++k) {
				output[k] /= num_iteration_for_pred_;
			}
		}
		if (objective_function_ != nullptr) {
			objective_function_->ConvertOutput(output, output);
		}
	}

	void GBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output, const PredictionEarlyStopInstance* early_stop) const {
		PredictRawByMap(features, output, early_stop);
		if (average_output_) {
			for (int k = 0; k < num_tree_per_iteration_; ++k) {
				output[k] /= num_iteration_for_pred_;
			}
		}
		if (objective_function_ != nullptr) {
			objective_function_->ConvertOutput(output, output);
		}
	}

	void GBDT::PredictLeafIndex(const double* features, double* output) const {
		int start_tree = start_iteration_for_pred_ * num_tree_per_iteration_;
		int num_trees = num_iteration_for_pred_ * num_tree_per_iteration_;
		const auto* models_ptr = models_.data() + start_tree;
		for (int i = 0; i < num_trees; ++i) {
			output[i] = models_ptr[i]->PredictLeafIndex(features);
		}
	}

	void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {
		int start_tree = start_iteration_for_pred_ * num_tree_per_iteration_;
		int num_trees = num_iteration_for_pred_ * num_tree_per_iteration_;
		const auto* models_ptr = models_.data() + start_tree;
		for (int i = 0; i < num_trees; ++i) {
			output[i] = models_ptr[i]->PredictLeafIndexByMap(features);
		}
	}

}  // namespace LightGBM
