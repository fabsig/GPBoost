/*!
* Original work Copyright (c) 2017 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef LIGHTGBM_METRIC_H_
#define LIGHTGBM_METRIC_H_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/common.h>

#include <string>
#include <vector>

namespace LightGBM {

	/*!
	* \brief The interface of metric.
	*        Metric is used to calculate metric result
	*/
	class Metric {
	public:
		/*! \brief virtual destructor */
		virtual ~Metric() {}

		/*!
		* \brief Initialize
		* \param test_name Specific name for this metric, will output on log
		* \param metadata Label data
		* \param num_data Number of data
		*/
		virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;

		virtual const std::vector<std::string>& GetName() const = 0;

		virtual double factor_to_bigger_better() const = 0;
		/*!
		* \brief Calculating and printing metric result
		* \param score Current prediction score
		* \param objective Objective function
		* \param residual_variance Current residual variance
		*/
		virtual std::vector<double> Eval(const double* score, const ObjectiveFunction* objective, const double* residual_variance) const = 0;

		Metric() = default;
		/*! \brief Disable copy */
		Metric& operator=(const Metric&) = delete;
		/*! \brief Disable copy */
		Metric(const Metric&) = delete;

		/*!
		* \brief Create object of metrics
		* \param type Specific type of metric
		* \param config Config for metric
		*/
		LIGHTGBM_EXPORT static Metric* CreateMetric(const std::string& type, const Config& config);

		/*! \brief True, if this metric is for the training data (used in latent_gaussian_laplace_metric) */
		bool metric_for_train_data_ = false;
	};

	/*!
	* \brief Static class, used to calculate DCG score
	*/
	class DCGCalculator {
	public:
		static void DefaultEvalAt(std::vector<int>* eval_at);
		static void DefaultLabelGain(std::vector<double>* label_gain);
		/*!
		* \brief Initial logic
		* \param label_gain Gain for labels, default is 2^i - 1
		*/
		static void Init(const std::vector<double>& label_gain);

		/*!
		* \brief Calculate the DCG score at position k
		* \param k The position to evaluate
		* \param label Pointer of label
		* \param score Pointer of score
		* \param num_data Number of data
		* \return The DCG score
		*/
		static double CalDCGAtK(data_size_t k, const label_t* label,
			const double* score, data_size_t num_data);

		/*!
		* \brief Calculate the DCG score at multi position
		* \param ks The positions to evaluate
		* \param label Pointer of label
		* \param score Pointer of score
		* \param num_data Number of data
		* \param out Output result
		*/
		static void CalDCG(const std::vector<data_size_t>& ks,
			const label_t* label, const double* score,
			data_size_t num_data, std::vector<double>* out);

		/*!
		* \brief Calculate the Max DCG score at position k
		* \param k The position want to eval at
		* \param label Pointer of label
		* \param num_data Number of data
		* \return The max DCG score
		*/
		static double CalMaxDCGAtK(data_size_t k,
			const label_t* label, data_size_t num_data);

		/*!
		* \brief Check the label range for NDCG and lambdarank
		* \param label Pointer of label
		* \param num_data Number of data
		*/
		static void CheckLabel(const label_t* label, data_size_t num_data);

		/*!
		* \brief Calculate the Max DCG score at multi position
		* \param ks The positions want to eval at
		* \param label Pointer of label
		* \param num_data Number of data
		* \param out Output result
		*/
		static void CalMaxDCG(const std::vector<data_size_t>& ks,
			const label_t* label, data_size_t num_data, std::vector<double>* out);

		/*!
		* \brief Get discount score of position k
		* \param k The position
		* \return The discount of this position
		*/
		inline static double GetDiscount(data_size_t k) { return discount_[k]; }

	private:
		/*! \brief store gains for different label */
		static std::vector<double> label_gain_;
		/*! \brief store discount score for different position */
		static std::vector<double> discount_;
		/*! \brief max position for eval */
		static const data_size_t kMaxPosition;
	};


}  // namespace LightGBM


#endif   // LightGBM_METRIC_H_
