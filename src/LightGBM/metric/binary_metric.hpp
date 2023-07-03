/*!
* Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef LIGHTGBM_METRIC_BINARY_METRIC_HPP_
#define LIGHTGBM_METRIC_BINARY_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

namespace LightGBM {

	/*!
	* \brief Metric for binary classification task.
	* Use static class "PointWiseLossCalculator" to calculate loss point-wise
	*/
	template<typename PointWiseLossCalculator>
	class BinaryMetric : public Metric {
	public:
		explicit BinaryMetric(const Config&) {
		}

		virtual ~BinaryMetric() {
		}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			name_.emplace_back(PointWiseLossCalculator::Name());

			num_data_ = num_data;
			// get label
			label_ = metadata.label();

			// get weights
			weights_ = metadata.weights();

			if (weights_ == nullptr) {
				sum_weights_ = static_cast<double>(num_data_);
			}
			else {
				sum_weights_ = 0.0f;
				for (data_size_t i = 0; i < num_data; ++i) {
					sum_weights_ += weights_[i];
				}
			}
		}

		const std::vector<std::string>& GetName() const override {
			return name_;
		}

		double factor_to_bigger_better() const override {
			return -1.0f;
		}

		std::vector<double> Eval(const double* score, const ObjectiveFunction* objective, const double*) const override {
			double sum_loss = 0.0f;
			if (objective == nullptr) {
				if (weights_ == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
					for (data_size_t i = 0; i < num_data_; ++i) {
						// add loss
						sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i]);
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
					for (data_size_t i = 0; i < num_data_; ++i) {
						// add loss
						sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i]) * weights_[i];
					}
				}
			}
			else {
				if (weights_ == nullptr) {
					// Add predictions from re_model (if needed)
					if (objective->HasGPModel() && objective->UseGPModelForValidation()) {
						if (metric_for_train_data_) {
							Log::Fatal("Cannot use the option 'use_gp_model_for_validation = true' for calculating this validation metric on the training data. Either (i) set 'use_gp_model_for_validation = false' or (ii) choose the metric 'neg_log_likelihood' and/or use only the training data as validation data.");
						}
						REModel* re_model = objective->GetGPModel();
						if (re_model->GaussLikelihood()) {//Gaussian data (this is rarely used)
							std::vector<double> minus_gp_pred(num_data_);
							re_model->Predict(nullptr, num_data_, minus_gp_pred.data(), false, false, false,
								nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
								true, nullptr, nullptr, true);//suppress_calc_cov_factor=true as this has been done already at the end of the last boosting update iteration
							// Note that the re_model already has the updated response data score - label = F_t - y 
							//	since 'Boosting()' is called (i.e. gradients are calculated) at the end of TrainOneIter()
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
							for (data_size_t i = 0; i < num_data_; ++i) {
								double pred = score[i] - minus_gp_pred[i];//minus since the re_model uses score - label (= F - y) instead of y - F to make predictions
								sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], pred);
							}
						}//end Gaussian data
						else {//non-Gaussian data
							std::vector<double> gp_pred(num_data_);
							re_model->Predict(nullptr, num_data_, gp_pred.data(), false, false, true,
								nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
								true, nullptr, score, true);//suppress_calc_cov_factor=true as this has been done already at the end of the last boosting update iteration
							// Note that the re_model already has the updated training score (= F_t)
							//	since 'Boosting()' is called (i.e. gradients are calculated) at the end of TrainOneIter()
							//	We thus don't provide this here (see the above nullptr). This also implies
							//	that the Laplace approximation (in particular the mode) is note calculated again
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
							for (data_size_t i = 0; i < num_data_; ++i) {
								sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], gp_pred[i]);
							}
						}//end non-Gaussian data
					}//end if (objective->HasGPModel()) && objective->UseGPModelForValidation())
					else {//re_model inexistent or not used for calculating validation loss
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
						for (data_size_t i = 0; i < num_data_; ++i) {
							double prob = 0;
							objective->ConvertOutput(&score[i], &prob);
							// add loss
							sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob);
						}
					} // end re_model inexistent or not used for calculating validation loss
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
					for (data_size_t i = 0; i < num_data_; ++i) {
						double prob = 0;
						objective->ConvertOutput(&score[i], &prob);
						// add loss
						sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob) * weights_[i];
					}
				}
			}
			double loss = sum_loss / sum_weights_;
			return std::vector<double>(1, loss);
		}

	private:
		/*! \brief Number of data */
		data_size_t num_data_;
		/*! \brief Pointer of label */
		const label_t* label_;
		/*! \brief Pointer of weighs */
		const label_t* weights_;
		/*! \brief Sum weights */
		double sum_weights_;
		/*! \brief Name of test set */
		std::vector<std::string> name_;
	};

	/*!
	* \brief Log loss metric for binary classification task.
	*/
	class BinaryLoglossMetric : public BinaryMetric<BinaryLoglossMetric> {
	public:
		explicit BinaryLoglossMetric(const Config& config) :BinaryMetric<BinaryLoglossMetric>(config) {}

		inline static double LossOnPoint(label_t label, double prob) {
			if (label <= 0) {
				if (1.0f - prob > kEpsilon) {
					return -std::log(1.0f - prob);
				}
			}
			else {
				if (prob > kEpsilon) {
					return -std::log(prob);
				}
			}
			return -std::log(kEpsilon);
		}

		inline static const char* Name() {
			return "binary_logloss";
		}
	};
	/*!
	* \brief Error rate metric for binary classification task.
	*/
	class BinaryErrorMetric : public BinaryMetric<BinaryErrorMetric> {
	public:
		explicit BinaryErrorMetric(const Config& config) :BinaryMetric<BinaryErrorMetric>(config) {}

		inline static double LossOnPoint(label_t label, double prob) {
			if (prob <= 0.5f) {
				return label > 0;
			}
			else {
				return label <= 0;
			}
		}

		inline static const char* Name() {
			return "binary_error";
		}
	};

	/*!
	* \brief Auc Metric for binary classification task.
	*/
	class AUCMetric : public Metric {
	public:
		explicit AUCMetric(const Config&) {
		}

		virtual ~AUCMetric() {
		}

		const std::vector<std::string>& GetName() const override {
			return name_;
		}

		double factor_to_bigger_better() const override {
			return 1.0f;
		}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			name_.emplace_back("auc");

			num_data_ = num_data;
			// get label
			label_ = metadata.label();
			// get weights
			weights_ = metadata.weights();

			if (weights_ == nullptr) {
				sum_weights_ = static_cast<double>(num_data_);
			}
			else {
				sum_weights_ = 0.0f;
				for (data_size_t i = 0; i < num_data; ++i) {
					sum_weights_ += weights_[i];
				}
			}
		}

		std::vector<double> Eval(const double* score, const ObjectiveFunction*, const double*) const override {
			// get indices sorted by score, descent order
			std::vector<data_size_t> sorted_idx;
			for (data_size_t i = 0; i < num_data_; ++i) {
				sorted_idx.emplace_back(i);
			}
			Common::ParallelSort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });
			// temp sum of postive label
			double cur_pos = 0.0f;
			// total sum of postive label
			double sum_pos = 0.0f;
			// accumlate of auc
			double accum = 0.0f;
			// temp sum of negative label
			double cur_neg = 0.0f;
			double threshold = score[sorted_idx[0]];
			if (weights_ == nullptr) {  // no weights
				for (data_size_t i = 0; i < num_data_; ++i) {
					const label_t cur_label = label_[sorted_idx[i]];
					const double cur_score = score[sorted_idx[i]];
					// new threshold
					if (cur_score != threshold) {
						threshold = cur_score;
						// accmulate
						accum += cur_neg * (cur_pos * 0.5f + sum_pos);
						sum_pos += cur_pos;
						// reset
						cur_neg = cur_pos = 0.0f;
					}
					cur_neg += (cur_label <= 0);
					cur_pos += (cur_label > 0);
				}
			}
			else {  // has weights
				for (data_size_t i = 0; i < num_data_; ++i) {
					const label_t cur_label = label_[sorted_idx[i]];
					const double cur_score = score[sorted_idx[i]];
					const label_t cur_weight = weights_[sorted_idx[i]];
					// new threshold
					if (cur_score != threshold) {
						threshold = cur_score;
						// accmulate
						accum += cur_neg * (cur_pos * 0.5f + sum_pos);
						sum_pos += cur_pos;
						// reset
						cur_neg = cur_pos = 0.0f;
					}
					cur_neg += (cur_label <= 0) * cur_weight;
					cur_pos += (cur_label > 0)* cur_weight;
				}
			}
			accum += cur_neg * (cur_pos * 0.5f + sum_pos);
			sum_pos += cur_pos;
			double auc = 1.0f;
			if (sum_pos > 0.0f && sum_pos != sum_weights_) {
				auc = accum / (sum_pos * (sum_weights_ - sum_pos));
			}
			return std::vector<double>(1, auc);
		}

	private:
		/*! \brief Number of data */
		data_size_t num_data_;
		/*! \brief Pointer of label */
		const label_t* label_;
		/*! \brief Pointer of weighs */
		const label_t* weights_;
		/*! \brief Sum weights */
		double sum_weights_;
		/*! \brief Name of test set */
		std::vector<std::string> name_;
	};


	/*!
	* \brief Average Precision Metric for binary classification task.
	*/
	class AveragePrecisionMetric : public Metric {
	public:
		explicit AveragePrecisionMetric(const Config&) {
		}

		virtual ~AveragePrecisionMetric() {
		}

		const std::vector<std::string>& GetName() const override {
			return name_;
		}

		double factor_to_bigger_better() const override {
			return 1.0f;
		}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			name_.emplace_back("average_precision");

			num_data_ = num_data;
			// get label
			label_ = metadata.label();
			// get weights
			weights_ = metadata.weights();

			if (weights_ == nullptr) {
				sum_weights_ = static_cast<double>(num_data_);
			}
			else {
				sum_weights_ = 0.0f;
				for (data_size_t i = 0; i < num_data; ++i) {
					sum_weights_ += weights_[i];
				}
			}
		}

		std::vector<double> Eval(const double* score, const ObjectiveFunction*, const double*) const override {
			// get indices sorted by score, descending order
			std::vector<data_size_t> sorted_idx;
			for (data_size_t i = 0; i < num_data_; ++i) {
				sorted_idx.emplace_back(i);
			}
			Common::ParallelSort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });
			// temp sum of postive label
			double cur_actual_pos = 0.0f;
			// total sum of postive label
			double sum_actual_pos = 0.0f;
			// total sum of predicted positive
			double sum_pred_pos = 0.0f;
			// accumulated precision
			double accum_prec = 1.0f;
			// accumlated pr-auc
			double accum = 0.0f;
			// temp sum of negative label
			double cur_neg = 0.0f;
			double threshold = score[sorted_idx[0]];
			if (weights_ == nullptr) {  // no weights
				for (data_size_t i = 0; i < num_data_; ++i) {
					const label_t cur_label = label_[sorted_idx[i]];
					const double cur_score = score[sorted_idx[i]];
					// new threshold
					if (cur_score != threshold) {
						threshold = cur_score;
						// accumulate
						sum_actual_pos += cur_actual_pos;
						sum_pred_pos += cur_actual_pos + cur_neg;
						accum_prec = sum_actual_pos / sum_pred_pos;
						accum += cur_actual_pos * accum_prec;
						// reset
						cur_neg = cur_actual_pos = 0.0f;
					}
					cur_neg += (cur_label <= 0);
					cur_actual_pos += (cur_label > 0);
				}
			}
			else {  // has weights
				for (data_size_t i = 0; i < num_data_; ++i) {
					const label_t cur_label = label_[sorted_idx[i]];
					const double cur_score = score[sorted_idx[i]];
					const label_t cur_weight = weights_[sorted_idx[i]];
					// new threshold
					if (cur_score != threshold) {
						threshold = cur_score;
						// accmulate
						sum_actual_pos += cur_actual_pos;
						sum_pred_pos += cur_actual_pos + cur_neg;
						accum_prec = sum_actual_pos / sum_pred_pos;
						accum += cur_actual_pos * accum_prec;
						// reset
						cur_neg = cur_actual_pos = 0.0f;
					}
					cur_neg += (cur_label <= 0) * cur_weight;
					cur_actual_pos += (cur_label > 0)* cur_weight;
				}
			}
			sum_actual_pos += cur_actual_pos;
			sum_pred_pos += cur_actual_pos + cur_neg;
			accum_prec = sum_actual_pos / sum_pred_pos;
			accum += cur_actual_pos * accum_prec;
			double ap = 1.0f;
			if (sum_actual_pos > 0.0f && sum_actual_pos != sum_weights_) {
				ap = accum / sum_actual_pos;
			}
			return std::vector<double>(1, ap);
		}

	private:
		/*! \brief Number of data */
		data_size_t num_data_;
		/*! \brief Pointer of label */
		const label_t* label_;
		/*! \brief Pointer of weighs */
		const label_t* weights_;
		/*! \brief Sum weights */
		double sum_weights_;
		/*! \brief Name of test set */
		std::vector<std::string> name_;
	};

}  // namespace LightGBM
#endif   // LightGBM_METRIC_BINARY_METRIC_HPP_
