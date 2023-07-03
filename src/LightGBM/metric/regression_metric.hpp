/*!
* Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef LIGHTGBM_METRIC_REGRESSION_METRIC_HPP_
#define LIGHTGBM_METRIC_REGRESSION_METRIC_HPP_

#include <GPBoost/re_model.h>
#include <LightGBM/metric.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI      3.1415926535897932384626433832795029
#endif

namespace LightGBM {
	/*!
	* \brief Metric for regression task.
	* Use static class "PointWiseLossCalculator" to calculate loss point-wise
	*/
	template<typename PointWiseLossCalculator>
	class RegressionMetric : public Metric {
	public:
		explicit RegressionMetric(const Config& config) :config_(config) {
		}

		virtual ~RegressionMetric() {
		}

		const std::vector<std::string>& GetName() const override {
			return name_;
		}

		double factor_to_bigger_better() const override {
			return -1.0f;
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
				for (data_size_t i = 0; i < num_data_; ++i) {
					sum_weights_ += weights_[i];
				}
			}
			for (data_size_t i = 0; i < num_data_; ++i) {
				PointWiseLossCalculator::CheckLabel(label_[i]);
			}
		}

		std::vector<double> Eval(const double* score, const ObjectiveFunction* objective, const double*) const override {
			double sum_loss = 0.0f;
			if (objective == nullptr) {
				if (weights_ == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
					for (data_size_t i = 0; i < num_data_; ++i) {
						// add loss
						sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i], config_);
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
					for (data_size_t i = 0; i < num_data_; ++i) {
						// add loss
						sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i], config_) * weights_[i];
					}
				}
			}
			else {
				if (weights_ == nullptr) {
					if (objective->HasGPModel() && objective->UseGPModelForValidation()) {
						if (metric_for_train_data_) {
							Log::Fatal("Cannot use the option 'use_gp_model_for_validation = true' for calculating this "
								"validation metric on the training data. If you want a metric on the training data, either (i) set 'use_gp_model_for_validation = false' "
								"or (ii) choose the metric 'neg_log_likelihood' and use only the training data as validation data.");
						}
						REModel* re_model = objective->GetGPModel();
						if (re_model->GaussLikelihood()) {//Gaussian data
							std::vector<double> minus_gp_pred(num_data_);
							re_model->Predict(nullptr, num_data_, minus_gp_pred.data(), false, false,  false, 
								nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
								true, nullptr, nullptr, true);//suppress_calc_cov_factor=true as this has been done already at the end of the last boosting update iteration
							// Note that the re_model already has the updated response data score - label = F_t - y 
							//	since 'Boosting()' is called (i.e. gradients are calculated) at the end of TrainOneIter()
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
							for (data_size_t i = 0; i < num_data_; ++i) {
								double pred = score[i] - minus_gp_pred[i];//minus since the re_model uses score - label (= F - y) instead of y - F to make predictions
								sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], pred, config_);
							}
						}//end Gaussian data
						else {//non-Gaussian data
							std::vector<double> gp_pred(num_data_);
							re_model->Predict(nullptr, num_data_, gp_pred.data(), false, false, true,
								nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
								true, nullptr, score, true);//suppress_calc_cov_factor=true as this has been done already at the end of the last boosting update iteration
							// Note that the re_model already has the updated training score (= F_t)
							//	since 'Boosting()' is called (i.e. gradients are calculated) at the end of TrainOneIter()
							//	We thus dont provide this here (see the above nullptr). This also implies
							//	that the Laplace approximation (in particular the mode) is note calculated again
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
							for (data_size_t i = 0; i < num_data_; ++i) {
								sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], gp_pred[i], config_);
							}
						}//end non-Gaussian data
					}//end if (objective->HasGPModel()) && objective->UseGPModelForValidation())
					else {//re_model inexistent or not used for calculating validation loss
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
						for (data_size_t i = 0; i < num_data_; ++i) {
							// add loss
							double t = 0;
							objective->ConvertOutput(&score[i], &t);
							sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], t, config_);
						}
					}//end re_model inexistent or not used for calculating validation loss
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
					for (data_size_t i = 0; i < num_data_; ++i) {
						// add loss
						double t = 0;
						objective->ConvertOutput(&score[i], &t);
						sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], t, config_) * weights_[i];
					}
				}
			}
			double loss = PointWiseLossCalculator::AverageLoss(sum_loss, sum_weights_);
			return std::vector<double>(1, loss);
		}

		inline static double AverageLoss(double sum_loss, double sum_weights) {
			return sum_loss / sum_weights;
		}

		inline static void CheckLabel(label_t) {
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
		/*! \brief Name of this test set */
		Config config_;
		std::vector<std::string> name_;
	};

	/*! \brief RMSE loss for regression task */
	class RMSEMetric : public RegressionMetric<RMSEMetric> {
	public:
		explicit RMSEMetric(const Config& config) :RegressionMetric<RMSEMetric>(config) {}

		inline static double LossOnPoint(label_t label, double score, const Config&) {
			return (score - label) * (score - label);
		}

		inline static double AverageLoss(double sum_loss, double sum_weights) {
			// need sqrt the result for RMSE loss
			return std::sqrt(sum_loss / sum_weights);
		}

		inline static const char* Name() {
			return "rmse";
		}
	};

	/*! \brief L2 loss for regression task */
	class L2Metric : public RegressionMetric<L2Metric> {
	public:
		explicit L2Metric(const Config& config) :RegressionMetric<L2Metric>(config) {}

		inline static double LossOnPoint(label_t label, double score, const Config&) {
			return (score - label) * (score - label);
		}

		inline static const char* Name() {
			return "l2";
		}
	};

	/*! \brief Quantile loss for regression task */
	class QuantileMetric : public RegressionMetric<QuantileMetric> {
	public:
		explicit QuantileMetric(const Config& config) :RegressionMetric<QuantileMetric>(config) {
		}

		inline static double LossOnPoint(label_t label, double score, const Config& config) {
			double delta = label - score;
			if (delta < 0) {
				return (config.alpha - 1.0f) * delta;
			}
			else {
				return config.alpha * delta;
			}
		}

		inline static const char* Name() {
			return "quantile";
		}
	};


	/*! \brief L1 loss for regression task */
	class L1Metric : public RegressionMetric<L1Metric> {
	public:
		explicit L1Metric(const Config& config) :RegressionMetric<L1Metric>(config) {}

		inline static double LossOnPoint(label_t label, double score, const Config&) {
			return std::fabs(score - label);
		}
		inline static const char* Name() {
			return "l1";
		}
	};

	/*! \brief Huber loss for regression task */
	class HuberLossMetric : public RegressionMetric<HuberLossMetric> {
	public:
		explicit HuberLossMetric(const Config& config) :RegressionMetric<HuberLossMetric>(config) {
		}

		inline static double LossOnPoint(label_t label, double score, const Config& config) {
			const double diff = score - label;
			if (std::abs(diff) <= config.alpha) {
				return 0.5f * diff * diff;
			}
			else {
				return config.alpha * (std::abs(diff) - 0.5f * config.alpha);
			}
		}

		inline static const char* Name() {
			return "huber";
		}
	};

	/*! \brief Fair loss for regression task */
	// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
	class FairLossMetric : public RegressionMetric<FairLossMetric> {
	public:
		explicit FairLossMetric(const Config& config) :RegressionMetric<FairLossMetric>(config) {
		}

		inline static double LossOnPoint(label_t label, double score, const Config& config) {
			const double x = std::fabs(score - label);
			const double c = config.fair_c;
			return c * x - c * c * std::log(1.0f + x / c);
		}

		inline static const char* Name() {
			return "fair";
		}
	};

	/*! \brief Poisson regression loss for regression task */
	class PoissonMetric : public RegressionMetric<PoissonMetric> {
	public:
		explicit PoissonMetric(const Config& config) :RegressionMetric<PoissonMetric>(config) {
		}

		inline static double LossOnPoint(label_t label, double score, const Config&) {
			const double eps = 1e-10f;
			if (score < eps) {
				score = eps;
			}
			return score - label * std::log(score);
		}
		inline static const char* Name() {
			return "poisson";
		}
	};


	/*! \brief Mape regression loss for regression task */
	class MAPEMetric : public RegressionMetric<MAPEMetric> {
	public:
		explicit MAPEMetric(const Config& config) :RegressionMetric<MAPEMetric>(config) {
		}

		inline static double LossOnPoint(label_t label, double score, const Config&) {
			return std::fabs((label - score)) / std::max(1.0f, std::fabs(label));
		}
		inline static const char* Name() {
			return "mape";
		}
	};

	class GammaMetric : public RegressionMetric<GammaMetric> {
	public:
		explicit GammaMetric(const Config& config) :RegressionMetric<GammaMetric>(config) {
		}

		inline static double LossOnPoint(label_t label, double score, const Config&) {
			const double psi = 1.0;
			const double theta = -1.0 / score;
			const double a = psi;
			const double b = -Common::SafeLog(-theta);
			const double c = 1. / psi * Common::SafeLog(label / psi) - Common::SafeLog(label) - 0;  // 0 = std::lgamma(1.0 / psi) = std::lgamma(1.0);
			return -((label * theta - b) / a + c);
		}
		inline static const char* Name() {
			return "gamma";
		}

		inline static void CheckLabel(label_t label) {
			CHECK_GT(label, 0);
		}
	};


	class GammaDevianceMetric : public RegressionMetric<GammaDevianceMetric> {
	public:
		explicit GammaDevianceMetric(const Config& config) :RegressionMetric<GammaDevianceMetric>(config) {
		}

		inline static double LossOnPoint(label_t label, double score, const Config&) {
			const double epsilon = 1.0e-9;
			const double tmp = label / (score + epsilon);
			return tmp - Common::SafeLog(tmp) - 1;
		}
		inline static const char* Name() {
			return "gamma_deviance";
		}
		inline static double AverageLoss(double sum_loss, double) {
			return sum_loss * 2;
		}
		inline static void CheckLabel(label_t label) {
			CHECK_GT(label, 0);
		}
	};

	class TweedieMetric : public RegressionMetric<TweedieMetric> {
	public:
		explicit TweedieMetric(const Config& config) :RegressionMetric<TweedieMetric>(config) {
		}

		inline static double LossOnPoint(label_t label, double score, const Config& config) {
			const double rho = config.tweedie_variance_power;
			const double eps = 1e-10f;
			if (score < eps) {
				score = eps;
			}
			const double a = label * std::exp((1 - rho) * std::log(score)) / (1 - rho);
			const double b = std::exp((2 - rho) * std::log(score)) / (2 - rho);
			return -a + b;
		}
		inline static const char* Name() {
			return "tweedie";
		}
	};

	/*!
	* \brief Metric for test negative log-likelihood
	*/
	class TestNegLogLikelihood : public Metric {
	public:
		explicit TestNegLogLikelihood(const Config& config) :config_(config) {
		}

		virtual ~TestNegLogLikelihood() {
		}

		const std::vector<std::string>& GetName() const override {
			return name_;
		}

		double factor_to_bigger_better() const override {
			return -1.0f;
		}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			num_data_ = num_data;
			label_ = metadata.label();
			weights_ = metadata.weights();
			if (weights_ == nullptr) {
				sum_weights_ = static_cast<double>(num_data_);
			}
			else {
				Log::Fatal("Sample weights can currently not be used for the metric 'test_neg_log_likelihood'");
			}
		}

		std::vector<double> Eval(const double* score, 
			const ObjectiveFunction* objective, 
			const double* residual_variance) const override {
			if (objective == nullptr) {
				Log::Fatal("'objective' cannot be nullptr for the metric 'test_neg_log_likelihood' ");
			}
			if (metric_for_train_data_) {
				Log::Fatal("Cannot use the metric 'test_neg_log_likelihood' on the training data ");
			}
			std::string obj_name = objective->GetName();
			if (!(objective->HasGPModel()) && obj_name != "regression") {
				Log::Fatal("The metric 'test_neg_log_likelihood' can only be used when "
					"having a GPModel / including random effects for non-Gaussian likelihoods ");
			}
			REModel* re_model = nullptr;
			if (objective->HasGPModel()) {
				re_model = objective->GetGPModel();
				if (!(re_model->GaussLikelihood()) && !(objective->UseGPModelForValidation())) {
					Log::Fatal("The metric 'test_neg_log_likelihood' can only be used when "
						"'use_gp_model_for_validation == true' for non-Gaussian likelihoods ");
				}
			}
			double sum_loss = 0.;
			if (objective->HasGPModel() && objective->UseGPModelForValidation()) {
				if (re_model->GaussLikelihood()) {//Gaussian data
					std::vector<double> re_pred(num_data_ * 2); // the first num_data_ are the negative predictive means followed by num_data_ predictive variances
					re_model->Predict(nullptr, num_data_, re_pred.data(), false, true, true,
						nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
						true, nullptr, nullptr, true);//suppress_calc_cov_factor=true as this has been done already at the end of the last boosting update iteration
					// Note that the re_model already has the updated response data score - label = F_t - y 
					//	since 'Boosting()' is called (i.e. gradients are calculated) at the end of TrainOneIter()
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
					for (data_size_t i = 0; i < num_data_; ++i) {
						sum_loss += std::pow(score[i] - re_pred[i] - label_[i], 2) / re_pred[num_data_ + i] + std::log(re_pred[num_data_ + i]);
					}
					sum_loss += num_data_ * LOG_2PI_;
					sum_loss *= 0.5;

					// The following code is equivalent to the above (but slower)
//					re_model->Predict(nullptr, num_data_, re_pred.data(), false, true, false,
//						nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
//						true, nullptr, nullptr, true);//suppress_calc_cov_factor=true as this has been done already at the end of the last boosting update iteration
//					// Note that the re_model already has the updated response data score - label = F_t - y 
//					//	since 'Boosting()' is called (i.e. gradients are calculated) at the end of TrainOneIter()
//#pragma omp parallel for schedule(static)
//					for (data_size_t i = 0; i < num_data_; ++i) {
//						re_pred[i] *= -1;
//						re_pred[i] += score[i];
//					}
//					sum_loss = re_model->TestNegLogLikelihoodAdaptiveGHQuadrature(label_, re_pred.data(), re_pred.data() + num_data_, num_data_);

				}//end Gaussian data
				else {//non-Gaussian data
					std::vector<double> re_pred(num_data_ * 2); // the first num_data_ are the predictive means followed by num_data_ predictive variances
					re_model->Predict(nullptr, num_data_, re_pred.data(), false, true, false,
						nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
						true, nullptr, score, true);//suppress_calc_cov_factor=true as this has been done already at the end of the last boosting update iteration
					// Note that the re_model already has the updated training score (= F_t)
					//	since 'Boosting()' is called (i.e. gradients are calculated) at the end of TrainOneIter()
					//	We thus don't provide this here (see the above nullptr). This also implies
					//	that the Laplace approximation (in particular the mode) is note calculated again
					sum_loss = re_model->TestNegLogLikelihoodAdaptiveGHQuadrature(label_, re_pred.data(), re_pred.data() + num_data_, num_data_);
				}//end non-Gaussian data
			}//end if (objective->HasGPModel()) && objective->UseGPModelForValidation())
			else {//re_model inexistent or not used for calculating validation loss for Gaussian likelihoods
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
				for (data_size_t i = 0; i < num_data_; ++i) {
					sum_loss += std::pow(score[i] - label_[i], 2) / residual_variance[0];
				}
				sum_loss += num_data_ * std::log(residual_variance[0]);
				sum_loss += num_data_ * LOG_2PI_;
				sum_loss *= 0.5;
			}//end re_model inexistent or not used for calculating validation loss
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
		/*! \brief Name of this metric */
		std::vector<std::string> name_ = { "test_neg_log_likelihood" };
		Config config_;
		double LOG_2PI_ = std::log(M_PI) + std::log(2);
	};//end TestNegLogLikelihood

}  // namespace LightGBM
#endif   // LightGBM_METRIC_REGRESSION_METRIC_HPP_
