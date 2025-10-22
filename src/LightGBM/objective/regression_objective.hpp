/*!
* Original work Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 - 2025 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_

#include <GPBoost/re_model.h>
#include <GPBoost/DF_utils.h>

#include <LightGBM/meta.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/array_args.h>

#include <string>
#include <algorithm>
#include <vector>

#include <cmath>
#ifndef M_PI
#define M_PI      3.141592653589793238462643383279502884 // pi
#endif

using GPBoost::REModel;

namespace LightGBM {

#define PercentileFun(T, data_reader, cnt_data, alpha)                    \
  {                                                                       \
    if (cnt_data <= 1) {                                                  \
      return data_reader(0);                                              \
    }                                                                     \
    std::vector<T> ref_data(cnt_data);                                    \
    for (data_size_t i = 0; i < cnt_data; ++i) {                          \
      ref_data[i] = data_reader(i);                                       \
    }                                                                     \
    const double float_pos = (1.0f - alpha) * cnt_data;                   \
    const data_size_t pos = static_cast<data_size_t>(float_pos);          \
    if (pos < 1) {                                                        \
      return ref_data[ArrayArgs<T>::ArgMax(ref_data)];                    \
    } else if (pos >= cnt_data) {                                         \
      return ref_data[ArrayArgs<T>::ArgMin(ref_data)];                    \
    } else {                                                              \
      const double bias = float_pos - pos;                                \
      if (pos > cnt_data / 2) {                                           \
        ArrayArgs<T>::ArgMaxAtK(&ref_data, 0, cnt_data, pos - 1);         \
        T v1 = ref_data[pos - 1];                                         \
        T v2 = ref_data[pos + ArrayArgs<T>::ArgMax(ref_data.data() + pos, \
                                                   cnt_data - pos)];      \
        return static_cast<T>(v1 - (v1 - v2) * bias);                     \
      } else {                                                            \
        ArrayArgs<T>::ArgMaxAtK(&ref_data, 0, cnt_data, pos);             \
        T v2 = ref_data[pos];                                             \
        T v1 = ref_data[ArrayArgs<T>::ArgMin(ref_data.data(), pos)];      \
        return static_cast<T>(v1 - (v1 - v2) * bias);                     \
      }                                                                   \
    }                                                                     \
  }\

#define WeightedPercentileFun(T, data_reader, weight_reader, cnt_data, alpha) \
  {                                                                           \
    if (cnt_data <= 1) {                                                      \
      return data_reader(0);                                                  \
    }                                                                         \
    std::vector<data_size_t> sorted_idx(cnt_data);                            \
    for (data_size_t i = 0; i < cnt_data; ++i) {                              \
      sorted_idx[i] = i;                                                      \
    }                                                                         \
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),                    \
                     [&](data_size_t a, data_size_t b) {                      \
                       return data_reader(a) < data_reader(b);                \
                     });                                                      \
    std::vector<double> weighted_cdf(cnt_data);                               \
    weighted_cdf[0] = weight_reader(sorted_idx[0]);                           \
    for (data_size_t i = 1; i < cnt_data; ++i) {                              \
      weighted_cdf[i] = weighted_cdf[i - 1] + weight_reader(sorted_idx[i]);   \
    }                                                                         \
    double threshold = weighted_cdf[cnt_data - 1] * alpha;                    \
    size_t pos = std::upper_bound(weighted_cdf.begin(), weighted_cdf.end(),   \
                                  threshold) -                                \
                 weighted_cdf.begin();                                        \
    pos = std::min(pos, static_cast<size_t>(cnt_data - 1));                   \
    if (pos == 0 || pos == static_cast<size_t>(cnt_data - 1)) {               \
      return data_reader(sorted_idx[pos]);                                    \
    }                                                                         \
    CHECK_GE(threshold, weighted_cdf[pos - 1]);                               \
    CHECK_LT(threshold, weighted_cdf[pos]);                                   \
    T v1 = data_reader(sorted_idx[pos - 1]);                                  \
    T v2 = data_reader(sorted_idx[pos]);                                      \
    if (weighted_cdf[pos + 1] - weighted_cdf[pos] >= 1.0f) {                  \
      return static_cast<T>((threshold - weighted_cdf[pos]) /                 \
                                (weighted_cdf[pos + 1] - weighted_cdf[pos]) * \
                                (v2 - v1) +                                   \
                            v1);                                              \
    } else {                                                                  \
      return static_cast<T>(v2);                                              \
    }                                                                         \
  }\

	/*!
	* \brief Objective function for regression
	*/
	class RegressionL2loss : public ObjectiveFunction {
	public:

		int NumModelPerIteration() const override { 
			if (has_gp_model_) {
				return(num_sets_fixed_effects_);
			}
			else {
				return 1;
			}		
		}

		explicit RegressionL2loss(const Config& config)
			: deterministic_(config.deterministic) {
			sqrt_ = config.reg_sqrt;
			reuse_learning_rates_gp_model_ = config.reuse_learning_rates_gp_model;
		}

		explicit RegressionL2loss(const std::vector<std::string>& strs)
			: deterministic_(false) {
			sqrt_ = false;
			reuse_learning_rates_gp_model_ = false;
			for (auto str : strs) {
				if (str == std::string("sqrt")) {
					sqrt_ = true;
				}
				else if (str == std::string("reuse_learning_rates_gp_model")) {
					reuse_learning_rates_gp_model_ = true;
				}
			}
		}

		~RegressionL2loss() {
		}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			num_data_ = num_data;
			label_ = metadata.label();
			if (sqrt_) {
				trans_label_.resize(num_data_);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					trans_label_[i] = Common::Sign(label_[i]) * std::sqrt(std::fabs(label_[i]));
				}
				label_ = trans_label_.data();
			}
			weights_ = metadata.weights();
		}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
				if (has_gp_model_) {//GPBoost algorithm
					if (re_model_->GaussLikelihood()) {//Gaussian likelihood
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							gradients[i] = static_cast<score_t>(score[i] - label_[i]);
							hessians[i] = 1.0f;
						}
						if (train_gp_model_cov_pars_) {//also train covariance parameters
							re_model_->OptimCovPar(gradients, nullptr, true, reuse_learning_rates_gp_model_);
							re_model_->CalcGradient(gradients, nullptr, false);//calc_cov_factor = false since this has already been done in OptimCovPar()
						}
						else {//don't train covariance parameters
							re_model_->CalcGradient(gradients, nullptr, calc_cov_factor_);
							calc_cov_factor_ = false;//calculate a covariance factor only once
						}
					}//end Gaussian data
					else {//non-Gaussian data
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_ * num_sets_fixed_effects_; ++i) {
							hessians[i] = 1.0f;
						}
						if (train_gp_model_cov_pars_) {//also train covariance parameters
							re_model_->OptimCovPar(nullptr, score, true, reuse_learning_rates_gp_model_);
							re_model_->CalcGradient(gradients, score, false);//calc_cov_factor = false since this has already been done in OptimCovPar()
						}
						else {//don't train covariance parameters
							re_model_->CalcGradient(gradients, score, true); // calc_cov_factor = true since the "factor" (mode) depends on the value of the score / F_t
						}
					}
				}//end has_gp_model_
				else {//classical boosting without random effects / re_model_
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						gradients[i] = static_cast<score_t>(score[i] - label_[i]);
						hessians[i] = 1.0f;
					}
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					gradients[i] = static_cast<score_t>((score[i] - label_[i]) * weights_[i]);
					hessians[i] = static_cast<score_t>(weights_[i]);
				}
			}
		}//end GetGradients

		const char* GetName() const override {
			return "regression";
		}

		void ConvertOutput(const double* input, double* output) const override {
			if (has_gp_model_) {
				output[0] = re_model_->TransformToReponseScale(input[0]);
			}
			else {
				if (sqrt_) {
					output[0] = Common::Sign(input[0]) * input[0] * input[0];
				}
				else {
					output[0] = input[0];
				}
			}
		}

		std::string ToString() const override {
			std::stringstream str_buf;
			str_buf << GetName();
			if (sqrt_) {
				str_buf << " sqrt";
			}
			return str_buf.str();
		}

		bool IsConstantHessian() const override {
			if (weights_ == nullptr) {
				return true;
			}
			else {
				return false;
			}
		}

		void FindInitScoreGP() const override {
			if (has_gp_model_ && likelihood_type_ != std::string("gaussian")) {
				re_model_->FindInitialValueBoosting();
			}
		}

		double BoostFromScore(int num_tree) const override {
			double suml = 0.0f;
			double sumw = 0.0f;
			double initscore = 0.0f;
			if (weights_ != nullptr) {
#pragma omp parallel for schedule(static) reduction(+:suml, sumw) if (!deterministic_)
				for (data_size_t i = 0; i < num_data_; ++i) {
					suml += label_[i] * weights_[i];
					sumw += weights_[i];
				}
				initscore = suml / sumw;
			}
			else {
				if (has_gp_model_) {
					if (likelihood_type_ == std::string("gaussian")) {
						sumw = static_cast<double>(num_data_);
#pragma omp parallel for schedule(static) reduction(+:suml)
						for (data_size_t i = 0; i < num_data_; ++i) {
							suml += label_[i];
						}
						initscore = suml / sumw;
					}
					else {
						initscore = re_model_->GetInitialValueBoosting(num_tree);
					}
					Log::Info("[GPBoost with %s likelihood]: initscore=%f",
						likelihood_type_.c_str(), initscore);
				}//end has_gp_model_
				else {//no gp_model
					sumw = static_cast<double>(num_data_);
#pragma omp parallel for schedule(static) reduction(+:suml) if (!deterministic_)
					for (data_size_t i = 0; i < num_data_; ++i) {
						suml += label_[i];
					}
					initscore = suml / sumw;
				}
			}
			return initscore;
		}

	protected:
		bool sqrt_;
		/*! \brief Number of data */
		data_size_t num_data_;
		/*! \brief Pointer of label */
		const label_t* label_;
		/*! \brief Pointer of weights */
		const label_t* weights_;
		std::vector<label_t> trans_label_;
		const bool deterministic_;
		/*! \brief Indicates whether the covariance matrix should also be factorized when calling re_model_->CalcGradient(). Only relevant if has_gp_model_ = true and train_gp_model_cov_pars_ = true */
		mutable bool calc_cov_factor_ = true;
		std::function<bool(label_t)> is_pos_ = [](label_t label) { return label > 0; };
	};

	/*!
	* \brief L1 regression loss
	*/
	class RegressionL1loss : public RegressionL2loss {
	public:
		explicit RegressionL1loss(const Config& config) : RegressionL2loss(config) {
		}

		explicit RegressionL1loss(const std::vector<std::string>& strs) : RegressionL2loss(strs) {
		}

		~RegressionL1loss() {}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double diff = score[i] - label_[i];
					gradients[i] = static_cast<score_t>(Common::Sign(diff));
					hessians[i] = 1.0f;
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double diff = score[i] - label_[i];
					gradients[i] = static_cast<score_t>(Common::Sign(diff) * weights_[i]);
					hessians[i] = weights_[i];
				}
			}
		}

		double BoostFromScore(int) const override {
			const double alpha = 0.5;
			if (weights_ != nullptr) {
#define data_reader(i) (label_[i])
#define weight_reader(i) (weights_[i])
				WeightedPercentileFun(label_t, data_reader, weight_reader, num_data_, alpha);
#undef data_reader
#undef weight_reader
			}
			else {
#define data_reader(i) (label_[i])
				PercentileFun(label_t, data_reader, num_data_, alpha);
#undef data_reader
			}
		}

		bool IsRenewTreeOutput() const override { return true; }

		double RenewTreeOutput(double, std::function<double(const label_t*, int)> residual_getter,
			const data_size_t* index_mapper,
			const data_size_t* bagging_mapper,
			data_size_t num_data_in_leaf) const override {
			const double alpha = 0.5;
			if (weights_ == nullptr) {
				if (bagging_mapper == nullptr) {
#define data_reader(i) (residual_getter(label_, index_mapper[i]))
					PercentileFun(double, data_reader, num_data_in_leaf, alpha);
#undef data_reader
				}
				else {
#define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
					PercentileFun(double, data_reader, num_data_in_leaf, alpha);
#undef data_reader
				}
			}
			else {
				if (bagging_mapper == nullptr) {
#define data_reader(i) (residual_getter(label_, index_mapper[i]))
#define weight_reader(i) (weights_[index_mapper[i]])
					WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha);
#undef data_reader
#undef weight_reader
				}
				else {
#define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
#define weight_reader(i) (weights_[bagging_mapper[index_mapper[i]]])
					WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha);
#undef data_reader
#undef weight_reader
				}
			}
		}

		const char* GetName() const override {
			return "regression_l1";
		}
	};

	/*!
	* \brief Huber regression loss
	*/
	class RegressionHuberLoss : public RegressionL2loss {
	public:
		explicit RegressionHuberLoss(const Config& config) : RegressionL2loss(config) {
			alpha_ = static_cast<double>(config.alpha);
			if (sqrt_) {
				Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
				sqrt_ = false;
			}
		}

		explicit RegressionHuberLoss(const std::vector<std::string>& strs) : RegressionL2loss(strs) {
			if (sqrt_) {
				Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
				sqrt_ = false;
			}
		}

		~RegressionHuberLoss() {
		}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double diff = score[i] - label_[i];
					if (std::abs(diff) <= alpha_) {
						gradients[i] = static_cast<score_t>(diff);
					}
					else {
						gradients[i] = static_cast<score_t>(Common::Sign(diff) * alpha_);
					}
					hessians[i] = 1.0f;
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double diff = score[i] - label_[i];
					if (std::abs(diff) <= alpha_) {
						gradients[i] = static_cast<score_t>(diff * weights_[i]);
					}
					else {
						gradients[i] = static_cast<score_t>(Common::Sign(diff) * weights_[i] * alpha_);
					}
					hessians[i] = static_cast<score_t>(weights_[i]);
				}
			}
		}

		const char* GetName() const override {
			return "huber";
		}

	private:
		/*! \brief delta for Huber loss */
		double alpha_;
	};


	// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
	class RegressionFairLoss : public RegressionL2loss {
	public:
		explicit RegressionFairLoss(const Config& config) : RegressionL2loss(config) {
			c_ = static_cast<double>(config.fair_c);
		}

		explicit RegressionFairLoss(const std::vector<std::string>& strs) : RegressionL2loss(strs) {
		}

		~RegressionFairLoss() {}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double x = score[i] - label_[i];
					gradients[i] = static_cast<score_t>(c_ * x / (std::fabs(x) + c_));
					hessians[i] = static_cast<score_t>(c_ * c_ / ((std::fabs(x) + c_) * (std::fabs(x) + c_)));
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double x = score[i] - label_[i];
					gradients[i] = static_cast<score_t>(c_ * x / (std::fabs(x) + c_) * weights_[i]);
					hessians[i] = static_cast<score_t>(c_ * c_ / ((std::fabs(x) + c_) * (std::fabs(x) + c_)) * weights_[i]);
				}
			}
		}

		const char* GetName() const override {
			return "fair";
		}

		bool IsConstantHessian() const override {
			return false;
		}

	private:
		/*! \brief c for Fair loss */
		double c_;
	};


	/*!
	* \brief Objective function for Poisson regression
	*/
	class RegressionPoissonLoss : public RegressionL2loss {
	public:
		explicit RegressionPoissonLoss(const Config& config) : RegressionL2loss(config) {
			max_delta_step_ = static_cast<double>(config.poisson_max_delta_step);
			if (sqrt_) {
				Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
				sqrt_ = false;
			}
		}

		explicit RegressionPoissonLoss(const std::vector<std::string>& strs) : RegressionL2loss(strs) {
		}

		~RegressionPoissonLoss() {}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			if (sqrt_) {
				Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
				sqrt_ = false;
			}
			RegressionL2loss::Init(metadata, num_data);
			// Safety check of labels
			label_t miny;
			double sumy;
			Common::ObtainMinMaxSum(label_, num_data_, &miny, static_cast<label_t*>(nullptr), &sumy);
			if (miny < 0.0f) {
				Log::Fatal("[%s]: at least one target label is negative", GetName());
			}
			if (sumy == 0.0f) {
				Log::Fatal("[%s]: sum of labels is zero", GetName());
			}
		}

		/* Parametrize with unbounded internal score "f"; then
		 *  loss = exp(f) - label * f
		 *  grad = exp(f) - label
		 *  hess = exp(f)
		 *
		 * And the output is exp(f); so the associated metric get s=exp(f)
		 * so that its loss = s - label * log(s); a little awkward maybe.
		 *
		 */
		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					gradients[i] = static_cast<score_t>(std::exp(score[i]) - label_[i]);
					hessians[i] = static_cast<score_t>(std::exp(score[i] + max_delta_step_));
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					gradients[i] = static_cast<score_t>((std::exp(score[i]) - label_[i]) * weights_[i]);
					hessians[i] = static_cast<score_t>(std::exp(score[i] + max_delta_step_) * weights_[i]);
				}
			}
		}

		void ConvertOutput(const double* input, double* output) const override {
			output[0] = std::exp(input[0]);
		}

		const char* GetName() const override {
			return "poisson";
		}

		double BoostFromScore(int) const override {
			return Common::SafeLog(RegressionL2loss::BoostFromScore(0));
		}

		bool IsConstantHessian() const override {
			return false;
		}

	private:
		/*! \brief used to safeguard optimization */
		double max_delta_step_;
	};

	class RegressionQuantileloss : public RegressionL2loss {
	public:
		explicit RegressionQuantileloss(const Config& config) : RegressionL2loss(config) {
			alpha_ = static_cast<score_t>(config.alpha);
			CHECK(alpha_ > 0 && alpha_ < 1);
		}

		explicit RegressionQuantileloss(const std::vector<std::string>& strs) : RegressionL2loss(strs) {
		}

		~RegressionQuantileloss() {}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					score_t delta = static_cast<score_t>(score[i] - label_[i]);
					if (delta >= 0) {
						gradients[i] = (1.0f - alpha_);
					}
					else {
						gradients[i] = -alpha_;
					}
					hessians[i] = 1.0f;
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					score_t delta = static_cast<score_t>(score[i] - label_[i]);
					if (delta >= 0) {
						gradients[i] = static_cast<score_t>((1.0f - alpha_) * weights_[i]);
					}
					else {
						gradients[i] = static_cast<score_t>(-alpha_ * weights_[i]);
					}
					hessians[i] = static_cast<score_t>(weights_[i]);
				}
			}
		}

		const char* GetName() const override {
			return "quantile";
		}

		double BoostFromScore(int) const override {
			if (weights_ != nullptr) {
#define data_reader(i) (label_[i])
#define weight_reader(i) (weights_[i])
				WeightedPercentileFun(label_t, data_reader, weight_reader, num_data_, alpha_);
#undef data_reader
#undef weight_reader
			}
			else {
#define data_reader(i) (label_[i])
				PercentileFun(label_t, data_reader, num_data_, alpha_);
#undef data_reader
			}
		}

		bool IsRenewTreeOutput() const override { return true; }

		double RenewTreeOutput(double, std::function<double(const label_t*, int)> residual_getter,
			const data_size_t* index_mapper,
			const data_size_t* bagging_mapper,
			data_size_t num_data_in_leaf) const override {
			if (weights_ == nullptr) {
				if (bagging_mapper == nullptr) {
#define data_reader(i) (residual_getter(label_, index_mapper[i]))
					PercentileFun(double, data_reader, num_data_in_leaf, alpha_);
#undef data_reader
				}
				else {
#define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
					PercentileFun(double, data_reader, num_data_in_leaf, alpha_);
#undef data_reader
				}
			}
			else {
				if (bagging_mapper == nullptr) {
#define data_reader(i) (residual_getter(label_, index_mapper[i]))
#define weight_reader(i) (weights_[index_mapper[i]])
					WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha_);
#undef data_reader
#undef weight_reader
				}
				else {
#define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
#define weight_reader(i) (weights_[bagging_mapper[index_mapper[i]]])
					WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha_);
#undef data_reader
#undef weight_reader
				}
			}
		}

	private:
		score_t alpha_;
	};


	/*!
	* \brief Mape Regression Loss
	*/
	class RegressionMAPELOSS : public RegressionL1loss {
	public:
		explicit RegressionMAPELOSS(const Config& config) : RegressionL1loss(config) {
		}

		explicit RegressionMAPELOSS(const std::vector<std::string>& strs) : RegressionL1loss(strs) {
		}

		~RegressionMAPELOSS() {}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			RegressionL2loss::Init(metadata, num_data);
			for (data_size_t i = 0; i < num_data_; ++i) {
				if (std::fabs(label_[i]) < 1) {
					Log::Warning(
						"Some label values are < 1 in absolute value. MAPE is unstable with such values, "
						"so LightGBM rounds them to 1.0 when calculating MAPE.");
					break;
				}
			}
			label_weight_.resize(num_data);
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					label_weight_[i] = 1.0f / std::max(1.0f, std::fabs(label_[i]));
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					label_weight_[i] = 1.0f / std::max(1.0f, std::fabs(label_[i])) * weights_[i];
				}
			}
		}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double diff = score[i] - label_[i];
					gradients[i] = static_cast<score_t>(Common::Sign(diff) * label_weight_[i]);
					hessians[i] = 1.0f;
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double diff = score[i] - label_[i];
					gradients[i] = static_cast<score_t>(Common::Sign(diff) * label_weight_[i]);
					hessians[i] = weights_[i];
				}
			}
		}

		double BoostFromScore(int) const override {
			const double alpha = 0.5;
#define data_reader(i) (label_[i])
#define weight_reader(i) (label_weight_[i])
			WeightedPercentileFun(label_t, data_reader, weight_reader, num_data_, alpha);
#undef data_reader
#undef weight_reader
		}

		bool IsRenewTreeOutput() const override { return true; }

		double RenewTreeOutput(double, std::function<double(const label_t*, int)> residual_getter,
			const data_size_t* index_mapper,
			const data_size_t* bagging_mapper,
			data_size_t num_data_in_leaf) const override {
			const double alpha = 0.5;
			if (bagging_mapper == nullptr) {
#define data_reader(i) (residual_getter(label_, index_mapper[i]))
#define weight_reader(i) (label_weight_[index_mapper[i]])
				WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha);
#undef data_reader
#undef weight_reader
			}
			else {
#define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
#define weight_reader(i) (label_weight_[bagging_mapper[index_mapper[i]]])
				WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha);
#undef data_reader
#undef weight_reader
			}
		}

		const char* GetName() const override {
			return "mape";
		}

		bool IsConstantHessian() const override {
			return true;
		}

	private:
		std::vector<label_t> label_weight_;
	};



	/*!
	* \brief Objective function for Gamma regression
	*/
	class RegressionGammaLoss : public RegressionPoissonLoss {
	public:
		explicit RegressionGammaLoss(const Config& config) : RegressionPoissonLoss(config) {
		}

		explicit RegressionGammaLoss(const std::vector<std::string>& strs) : RegressionPoissonLoss(strs) {
		}

		~RegressionGammaLoss() {}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					gradients[i] = static_cast<score_t>(1.0 - label_[i] / std::exp(score[i]));
					hessians[i] = static_cast<score_t>(label_[i] / std::exp(score[i]));
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					gradients[i] = static_cast<score_t>(1.0 - label_[i] / std::exp(score[i]) * weights_[i]);
					hessians[i] = static_cast<score_t>(label_[i] / std::exp(score[i]) * weights_[i]);
				}
			}
		}

		const char* GetName() const override {
			return "gamma";
		}
	};

	/*!
	* \brief Objective function for Tweedie regression
	*/
	class RegressionTweedieLoss : public RegressionPoissonLoss {
	public:
		explicit RegressionTweedieLoss(const Config& config) : RegressionPoissonLoss(config) {
			rho_ = config.tweedie_variance_power;
		}

		explicit RegressionTweedieLoss(const std::vector<std::string>& strs) : RegressionPoissonLoss(strs) {
		}

		~RegressionTweedieLoss() {}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					gradients[i] = static_cast<score_t>(-label_[i] * std::exp((1 - rho_) * score[i]) + std::exp((2 - rho_) * score[i]));
					hessians[i] = static_cast<score_t>(-label_[i] * (1 - rho_) * std::exp((1 - rho_) * score[i]) +
						(2 - rho_) * std::exp((2 - rho_) * score[i]));
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					gradients[i] = static_cast<score_t>((-label_[i] * std::exp((1 - rho_) * score[i]) + std::exp((2 - rho_) * score[i])) * weights_[i]);
					hessians[i] = static_cast<score_t>((-label_[i] * (1 - rho_) * std::exp((1 - rho_) * score[i]) +
						(2 - rho_) * std::exp((2 - rho_) * score[i])) * weights_[i]);
				}
			}
		}

		const char* GetName() const override {
			return "tweedie";
		}

	private:
		double rho_;
	};

	/*!
* \brief Objective function for Tobit model
*	Reference: Sigrist, F., & Hirnschall, C. (2019). Grabit: Gradient Tree Boosted Tobit Models for Default Prediction. Journal of Banking and Finance
*/
	class TobitLoss : public RegressionL2loss {
	public:
		explicit TobitLoss(const Config& config) : RegressionL2loss(config) {
			sigma_ = static_cast<double>(config.sigma);
			yl_ = static_cast<double>(config.yl);
			yu_ = static_cast<double>(config.yu);
			if (sigma_ <= 0.0) {
				Log::Fatal("'sigma' must be greater than zero but was %f", sigma_);
			}
			if (yu_ <= yl_) {
				Log::Fatal("'yl' must be smaller than 'yu'");
			}
		}

		explicit TobitLoss(const std::vector<std::string>& strs) : RegressionL2loss(strs) {
		}

		~TobitLoss() {}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			if (sqrt_) {
				Log::Warning("Cannot use sqrt transform for %s loss, will auto disable it", GetName());
				sqrt_ = false;
			}
			RegressionL2loss::Init(metadata, num_data);
			const_ = 0.5 * std::log(2 * M_PI) + std::log(sigma_);
			sigma2_inverse_ = 1. / (sigma_ * sigma_);
			// Safety check for labels
#pragma omp parallel for schedule(static)
			for (data_size_t i = 0; i < num_data_; ++i) {
				if (label_[i] - yl_ < -1e-6 * std::abs(yl_)) {
					Log::Fatal("Label / response variable (sample nb. =%d, value=%f) must not be smaller than yl (=%f)", i, label_[i], yl_);
				}
				else if (label_[i] - yu_ > 1e-6 * std::abs(yu_)) {
					Log::Fatal("Label / response variable (sample nb. =%d, value=%f) must not be larger than yu (=%f)", i, label_[i], yu_);
				}
			}
		}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double diff = (label_[i] - score[i]) / sigma_;
					if (label_[i] <= yl_) {// lower censoring
						const double logpdf = GPBoost::normalLogPDF(diff);
						const double logcdf = GPBoost::normalLogCDF(diff);
						gradients[i] = static_cast<score_t>(std::exp(logpdf - logcdf) / sigma_);
						hessians[i] = static_cast<score_t>(std::exp(logpdf - logcdf) * sigma2_inverse_ * diff +
							std::exp(2 * logpdf - 2 * logcdf) * sigma2_inverse_);
					}
					else if (label_[i] >= yu_) {// upper censoring
						const double logpdf = GPBoost::normalLogPDF(diff);
						const double logcdf = GPBoost::normalLogCDF(-diff);
						gradients[i] = static_cast<score_t>(-std::exp(logpdf - logcdf) / sigma_);
						hessians[i] = static_cast<score_t>(-std::exp(logpdf - logcdf) * sigma2_inverse_ * diff +
							std::exp(2 * logpdf - 2 * logcdf) * sigma2_inverse_);
					}
					else {// not censored observation
						gradients[i] = static_cast<score_t>(-diff / sigma_);
						hessians[i] = static_cast<score_t>(sigma2_inverse_);
					}
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double diff = (label_[i] - score[i]) / sigma_;
					if (label_[i] <= yl_) {// lower censoring
						const double logpdf = GPBoost::normalLogPDF(diff);
						const double logcdf = GPBoost::normalLogCDF(diff);
						gradients[i] = static_cast<score_t>(std::exp(logpdf - logcdf) / sigma_ * weights_[i]);
						hessians[i] = static_cast<score_t>((std::exp(logpdf - logcdf) * sigma2_inverse_ * diff +
							std::exp(2 * logpdf - 2 * logcdf) * sigma2_inverse_) * weights_[i]);
					}
					else if (label_[i] >= yu_) {// upper censoring
						const double logpdf = GPBoost::normalLogPDF(diff);
						const double logcdf = GPBoost::normalLogCDF(-diff);
						gradients[i] = static_cast<score_t>(-std::exp(logpdf - logcdf) / sigma_ * weights_[i]);
						hessians[i] = static_cast<score_t>((-std::exp(logpdf - logcdf) * sigma2_inverse_ * diff +
							std::exp(2 * logpdf - 2 * logcdf) * sigma2_inverse_) * weights_[i]);
					}
					else {// not censored observation
						gradients[i] = static_cast<score_t>(-diff / sigma_ * weights_[i]);
						hessians[i] = static_cast<score_t>(sigma2_inverse_ * weights_[i]);
					}
				}
			}
		}

		const char* GetName() const override {
			return "tobit";
		}

		bool IsConstantHessian() const override {
			return false;
		}

	private:
		/*! \brief Standard deviation of latent Gaussian variable */
		double sigma_;
		double sigma2_inverse_;
		/*! \brief Lower censoring threshold */
		double yl_;
		/*! \brief Upper censoring threshold */
		double yu_;
		/*! \brief Normalizing constant for (negative) Tobit log-likelihood not depending on data */
		double const_;
	};

	/*!
	* \brief Objective function for mean-scale regression
	*/
	class MeanScaleLoss : public ObjectiveFunction {
	public:

		int NumModelPerIteration() const override {
			return 2;
		}

		explicit MeanScaleLoss(const Config& config)
			: deterministic_(config.deterministic) {
		}

		explicit MeanScaleLoss(const std::vector<std::string>& )
			: deterministic_(false) {
		}

		~MeanScaleLoss() {
		}

		void Init(const Metadata& metadata, data_size_t num_data) override {
			num_data_ = num_data;
			label_ = metadata.label();
			weights_ = metadata.weights();
		}

		void GetGradients(const double* score, score_t* gradients,
			score_t* hessians) const override {
			if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					double inv_var = std::exp(-score[i + num_data_]);
					double neg_resid = score[i] - label_[i];
					gradients[i] = static_cast<score_t>(inv_var * neg_resid);
					hessians[i] = static_cast<score_t>(inv_var);
					hessians[i + num_data_] = static_cast<score_t>(inv_var * neg_resid * neg_resid / 2.);
					gradients[i + num_data_] = -hessians[i + num_data_] + 0.5;
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					double inv_var = std::exp(-score[i + num_data_]);
					double neg_resid = score[i] - label_[i];
					gradients[i] = static_cast<score_t>(neg_resid * inv_var * weights_[i]);
					hessians[i] = static_cast<score_t>(inv_var * weights_[i]);
					hessians[i + num_data_] = static_cast<score_t>(inv_var * neg_resid * neg_resid / 2. * weights_[i]);
					gradients[i + num_data_] = -gradients[i + num_data_] + 0.5;
				}
			}
		}//end GetGradients

		const char* GetName() const override {
			return "mean_scale_regression";
		}

		void ConvertOutput(const double* input, double* output) const override {
			output[0] = input[0];
		}

		std::string ToString() const override {
			std::stringstream str_buf;
			str_buf << GetName();
			return str_buf.str();
		}

		bool IsConstantHessian() const override {
			return false;
		}

		double BoostFromScore(int num_tree) const override {
			double suml = 0.0f;
			double sumw = 0.0f;
			double initscore = 0.0f;
			if (num_tree == 0) {
				if (weights_ != nullptr) {
#pragma omp parallel for schedule(static) reduction(+:suml, sumw) if (!deterministic_)
					for (data_size_t i = 0; i < num_data_; ++i) {
						suml += label_[i] * weights_[i];
						sumw += weights_[i];
					}
					initscore = suml / sumw;
				}
				else {
					sumw = static_cast<double>(num_data_);
#pragma omp parallel for schedule(static) reduction(+:suml) if (!deterministic_)
					for (data_size_t i = 0; i < num_data_; ++i) {
						suml += label_[i];
					}
					initscore = suml / sumw;
				}
			}
			else if (num_tree == 1) {
				double sum_sq = 0.0f;
				if (weights_ != nullptr) {
#pragma omp parallel for schedule(static) reduction(+:suml, sumw) if (!deterministic_)
					for (data_size_t i = 0; i < num_data_; ++i) {
						suml += label_[i] * weights_[i];
						sum_sq += label_[i] * label_[i] * weights_[i];
						sumw += weights_[i];
					}
					initscore = std::log(sum_sq / sumw - suml * suml / sumw / sumw);
				}
				else {
					sumw = static_cast<double>(num_data_);
#pragma omp parallel for schedule(static) reduction(+:suml) if (!deterministic_)
					for (data_size_t i = 0; i < num_data_; ++i) {
						suml += label_[i];
						sum_sq += label_[i] * label_[i];
					}
					initscore = std::log(sum_sq / sumw - suml * suml / sumw / sumw);
				}
			}
			return initscore;
		}

	protected:
		/*! \brief Number of data */
		data_size_t num_data_;
		/*! \brief Pointer of label */
		const label_t* label_;
		/*! \brief Pointer of weights */
		const label_t* weights_;
		const bool deterministic_;
		std::function<bool(label_t)> is_pos_ = [](label_t label) { return label > 0; };
	};

#undef PercentileFun
#undef WeightedPercentileFun

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
