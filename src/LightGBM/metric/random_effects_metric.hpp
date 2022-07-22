/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef GPBOOST_METRIC_RANDOM_EFFECTS_METRIC_HPP_
#define GPBOOST_METRIC_RANDOM_EFFECTS_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/log.h>
#include <GPBoost/re_model.h>

#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

namespace LightGBM {
	/*!
	* \brief Metric when having a random effects model (re_model) for Gaussian data
	*/
	class NegLogLikelihood : public Metric {
	public:
		explicit NegLogLikelihood(const Config& config) :config_(config) {
		}

		virtual ~NegLogLikelihood() {
		}

		const std::vector<std::string>& GetName() const override {
			return name_;
		}

		double factor_to_bigger_better() const override {
			return -1.0f;
		}

		void Init(const Metadata&, data_size_t) override {
			if (!metric_for_train_data_) {
				Log::Fatal("The metric 'neg_log_likelihood' cannot be used for validation data, it can only be used for training data");
			}
		}

		std::vector<double> Eval(const double*, const ObjectiveFunction* objective, const double*) const override {
			double loss;
			if (metric_for_train_data_) {
				REModel* re_model = objective->GetGPModel();
				re_model->EvalNegLogLikelihood(nullptr, nullptr, loss, nullptr, false, false);
			}
			else {
				//loss = std::numeric_limits<double>::quiet_NaN();//gives an error
				loss = 0;
			}
			return std::vector<double>(1, loss);
		}

	private:
		/*! \brief Name of this metric */
		std::vector<std::string> name_ = { "Negative log-likelihood" };
		Config config_;
	};

	/*!
	* \brief Metric when having a random effects model (re_model) for non-Gaussian data and inference is done using the Laplace approximation
	*/
	class LatenGaussianLaplace : public Metric {
	public:
		explicit LatenGaussianLaplace(const Config& config) :config_(config) {
		}

		virtual ~LatenGaussianLaplace() {
		}

		const std::vector<std::string>& GetName() const override {
			return name_;
		}

		double factor_to_bigger_better() const override {
			return -1.0f;
		}

		void Init(const Metadata&, data_size_t) override {
			if (!metric_for_train_data_) {
				Log::Fatal("The metric 'approx_neg_marginal_log_likelihood' cannot be used for validation data, it can only be used for training data");
			}
		}

		std::vector<double> Eval(const double* score, const ObjectiveFunction* objective, const double*) const override {
			double loss;
			if (metric_for_train_data_) {
				REModel* re_model = objective->GetGPModel();
				re_model->EvalNegLogLikelihood(nullptr, nullptr, loss, score, false, false);
			}
			else {
				//loss = std::numeric_limits<double>::quiet_NaN();//gives an error
				loss = 0;
			}
			return std::vector<double>(1, loss);
		}

	private:
		/*! \brief Name of this metric */
		std::vector<std::string> name_ = { "Approx. negative marginal log-likelihood" };
		Config config_;
	};

}  // namespace LightGBM
#endif   // GPBOOST_METRIC_RANDOM_EFFECTS_METRIC_HPP_
