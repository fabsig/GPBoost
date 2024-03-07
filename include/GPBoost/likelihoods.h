/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*
*
*  EXPLANATIONS ON PARAMETERIZATIONS USED
*
* For a "gamma" likelihood, the following density is used:
*	f(y) = lambda^gamma / Gamma(gamma) * y^(gamma - 1) * exp(-lambda * y)
*		- lambda = gamma * exp(-location_par) (i.e., mean(y) = exp(location_par)
*		- lambda = rate parameter, gamma = shape parameter, location_par = random + fixed effects
* 
* For a "negative_binomial" likelihood, the following density is used:
*	f(y) = Gamma(y + r) / Gamma(y + 1) / Gamma(r) * (1 - p)^y * p^r
*		- p = r / (mu + r), where mu = mean(y) = exp(location_par)
*		- p = success probability, r = shape parameter, location_par = random + fixed effects
*
*/
#ifndef GPB_LIKELIHOODS_
#define GPB_LIKELIHOODS_

#define _USE_MATH_DEFINES // for M_SQRT1_2 and M_PI
#include <cmath>

#include <GPBoost/type_defs.h>
#include <GPBoost/sparse_matrix_utils.h>
#include <GPBoost/DF_utils.h>
#include <GPBoost/utils.h>
#include <GPBoost/CG_utils.h>

#include <string>
#include <set>
#include <vector>

#include <LightGBM/utils/log.h>
using LightGBM::Log;
#include <LightGBM/meta.h>
using LightGBM::label_t;

//Mathematical constants usually defined in cmath
#ifndef M_SQRT2
#define M_SQRT2      1.414213562373095048801688724209698079 //sqrt(2)
#endif

#include <chrono>  // only for debugging
#include <thread> // only for debugging

namespace GPBoost {

	// Forward declaration
	template<typename T_mat, typename T_chol>
	class REModelTemplate;

	/*!
	* \brief This class implements the likelihoods for the Gaussian proceses
	* The template parameters <T_mat, T_chol> can be <den_mat_t, chol_den_mat_t> , <sp_mat_t, chol_sp_mat_t>, <sp_mat_rm_t, chol_sp_mat_rm_t>
	*/
	template<typename T_mat, typename T_chol>
	class Likelihood {
	public:
		/*! \brief Constructor */
		Likelihood();

		/*!
		* \brief Constructor
		* \param type Type of likelihood
		* \param num_data Number of data points
		* \param num_re Number of random effects
		* \param has_a_vec Indicates whether the vector a_vec_ / a = (Z Sigma Zt)^-1 mode is used or not
		* \param use_Z_for_duplicates If true, an incidendce matrix Z is used for duplicate locations and calculations are done on the random effects scale with the unique locations (only for Gaussian processes)
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		*/
		Likelihood(string_t type,
			data_size_t num_data,
			data_size_t num_re,
			bool has_a_vec,
			bool use_Z_for_duplicates,
			const data_size_t* random_effects_indices_of_data) {
			string_t likelihood = ParseLikelihoodAlias(type);
			likelihood = ParseLikelihoodAliasGradientDescent(likelihood);
			if (SUPPORTED_LIKELIHOODS_.find(likelihood) == SUPPORTED_LIKELIHOODS_.end()) {
				Log::REFatal("Likelihood of type '%s' is not supported.", likelihood.c_str());
			}
			likelihood_type_ = likelihood;
			num_data_ = num_data;
			num_re_ = num_re;
			num_aux_pars_ = 0;
			if (likelihood_type_ == "gamma") {
				aux_pars_ = { 1. };//shape parameter
				names_aux_pars_ = { "shape" };
				num_aux_pars_ = 1;
			}
			else if (likelihood_type_ == "negative_binomial") {
				aux_pars_ = { 1. };//shape parameter (aka size, theta, or "number of successes")
				names_aux_pars_ = { "shape" };
				num_aux_pars_ = 1;
			}
			else if (likelihood_type_ == "gaussian") {
				aux_pars_ = { 1. };//1 / sqrt(variance)
				names_aux_pars_ = { "inverse_std_dev" };
				num_aux_pars_ = 0;
			} 
			chol_fact_pattern_analyzed_ = false;
			has_a_vec_ = has_a_vec;
			use_Z_for_duplicates_ = use_Z_for_duplicates;
			if (use_Z_for_duplicates_) {
				random_effects_indices_of_data_ = random_effects_indices_of_data;
				dim_mode_ = num_re_;
			}
			else {
				dim_mode_ = num_data_;
			}
			DetermineWhetherToCapChangeModeNewton();
		}

		/*!
		* \brief Determine cap_change_mode_newton_
		*/
		void DetermineWhetherToCapChangeModeNewton() {
			if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" ||
				likelihood_type_ == "negative_binomial") {
				cap_change_mode_newton_ = true;
			}
			else {
				cap_change_mode_newton_ = false;
			}
		}

		/*!
		* \brief Initialize mode vector_ (used in Laplace approximation for non-Gaussian data)
		*/
		void InitializeModeAvec() {
			mode_ = vec_t::Zero(num_re_);
			mode_previous_value_ = vec_t::Zero(num_re_);
			if (has_a_vec_) {
				a_vec_ = vec_t::Zero(num_re_);
				a_vec_previous_value_ = vec_t::Zero(num_re_);
			}
			mode_initialized_ = true;
			first_deriv_ll_ = vec_t(dim_mode_);
			second_deriv_neg_ll_ = vec_t(dim_mode_);
			if (use_Z_for_duplicates_) {
				first_deriv_ll_data_scale_ = vec_t(num_data_);
				second_deriv_neg_ll_data_scale_ = vec_t(num_data_);
			}
			mode_has_been_calculated_ = false;
			na_or_inf_during_last_call_to_find_mode_ = false;
			na_or_inf_during_second_last_call_to_find_mode_ = false;
		}

		/*!
		* \brief Reset mode to previous value. This is used if too large step-sizes are done which result in increases in the objective function.
		"			The values (covariance parameters and linear coefficients) are then discarded and consequently the mode should also be reset to the previous value)
		*/
		void ResetModeToPreviousValue() {
			CHECK(mode_initialized_);
			mode_ = mode_previous_value_;
			if (has_a_vec_) {
				a_vec_ = a_vec_previous_value_;
			}
			na_or_inf_during_last_call_to_find_mode_ = na_or_inf_during_second_last_call_to_find_mode_;
		}

		/*! \brief Destructor */
		~Likelihood() {
		}

		/*!
		* \brief Returns the type of likelihood
		*/
		string_t GetLikelihood() const {
			return(likelihood_type_);
		}

		/*!
		* \brief Set the type of likelihood
		* \param type Likelihood name
		*/
		void SetLikelihood(const string_t& type) {
			string_t likelihood = ParseLikelihoodAlias(type);
			likelihood = ParseLikelihoodAliasGradientDescent(likelihood);
			if (SUPPORTED_LIKELIHOODS_.find(likelihood) == SUPPORTED_LIKELIHOODS_.end()) {
				Log::REFatal("Likelihood of type '%s' is not supported.", likelihood.c_str());
			}
			likelihood_type_ = likelihood;
			chol_fact_pattern_analyzed_ = false;
			DetermineWhetherToCapChangeModeNewton();
		}

		/*!
		* \brief Set chol_fact_pattern_analyzed_ to false
		*/
		void SetCholFactPatternAnalyzedFalse() {
			chol_fact_pattern_analyzed_ = false;
		}

		/*!
		* \brief Returns the type of the response variable (label). Either "double" or "int"
		*/
		string_t label_type() const {
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit" ||
				likelihood_type_ == "poisson" || likelihood_type_ == "negative_binomial") {
				return("int");
			}
			else {
				return("double");
			}
		}

		/*!
		* \brief Returns a pointer to mode_
		*/
		const vec_t* GetMode() const {
			return(&mode_);
		}

		/*!
		* \brief Returns a pointer to first_deriv_ll_
		*/
		const vec_t* GetFirstDerivLL() const {
			return(&first_deriv_ll_);
		}

		/*!
		* \brief Checks whether the response variables (labels) have the correct values
		* \param y_data Response variable data
		* \param num_data Number of data points
		*/
		template <typename T>//T can be double or float
		void CheckY(const T* y_data,
			const data_size_t num_data) const {
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit") {
				//#pragma omp parallel for schedule(static)//problematic with error message below... 
				for (data_size_t i = 0; i < num_data; ++i) {
					if (fabs(y_data[i]) >= EPSILON_NUMBERS && !TwoNumbersAreEqual<T>(y_data[i], 1.)) {
						Log::REFatal("Response variable (label) data needs to be 0 or 1 for likelihood of type '%s' ", likelihood_type_.c_str());
					}
				}
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "negative_binomial") {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] < 0) {
						Log::REFatal("Found negative response variable. Response variable cannot be negative for likelihood of type '%s' ", likelihood_type_.c_str());
					}
					else {
						double intpart;
						if (std::modf(y_data[i], &intpart) != 0.0) {
							Log::REFatal("Found non-integer response variable. Response variable can only be integer valued for likelihood of type '%s' ", likelihood_type_.c_str());
						}
					}
				}
			}
			else if (likelihood_type_ == "gamma") {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] <= 0) {
						Log::REFatal("Found non-positive response variable. Response variable must be positive for likelihood of type '%s' ", likelihood_type_.c_str());
					}
				}
			}
			else {
				Log::REFatal("GPModel: Likelihood of type '%s' is not supported ", likelihood_type_.c_str());
			}
		}//end CheckY

		/*!
		* \brief Determine initial value for intercept (=constant)
		* \param y_data Response variable data
		* \param num_data Number of data points
		* \param rand_eff_var Variance of random effects
		* \param fixed_effects Additional fixed effects that are added to the linear predictor (= offset)
		*/
		double FindInitialIntercept(const double* y_data,
			const data_size_t num_data,
			double rand_eff_var,
			const double* fixed_effects) const {
			CHECK(rand_eff_var > 0.);
			double init_intercept = 0.;
			if (likelihood_type_ == "gaussian") {
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:init_intercept)
					for (data_size_t i = 0; i < num_data; ++i) {
						init_intercept += y_data[i];
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:init_intercept)
					for (data_size_t i = 0; i < num_data; ++i) {
						init_intercept += y_data[i] - fixed_effects[i];
					}
				}
				init_intercept /= num_data;
			}
			else if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit") {
				double pavg = 0.;
#pragma omp parallel for schedule(static) reduction(+:pavg)
				for (data_size_t i = 0; i < num_data; ++i) {
					pavg += bool(y_data[i] > 0);
				}
				pavg /= num_data;
				pavg = std::min(pavg, 1.0 - 1e-15);
				pavg = std::max<double>(pavg, 1e-15);
				if (likelihood_type_ == "bernoulli_logit") {
					init_intercept = std::log(pavg / (1.0 - pavg));
				}
				else {
					init_intercept = normalQF(pavg);
				}
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || 
				likelihood_type_ == "negative_binomial") {
				double avg = 0.;
#pragma omp parallel for schedule(static) reduction(+:avg)
				for (data_size_t i = 0; i < num_data; ++i) {
					avg += y_data[i];
				}
				avg /= num_data;
				init_intercept = SafeLog(avg) - 0.5 * rand_eff_var; // log-normal distribution: mean of exp(beta_0 + Zb) = exp(beta_0 + 0.5 * sigma^2) => use beta_0 = mean(y) - 0.5 * sigma^2
			}
			else {
				Log::REFatal("FindInitialIntercept: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
			return(init_intercept);
		}//end FindInitialIntercept

		/*!
		* \brief Should there be an intercept or not (might raise a warning)
		* \param y_data Response variable data
		* \param num_data Number of data points
		* \param rand_eff_var Variance of random effects
		* \param fixed_effects Additional fixed effects that are added to the linear predictor (= offset)
		*/
		bool ShouldHaveIntercept(const double* y_data,
			const data_size_t num_data,
			double rand_eff_var,
			const double* fixed_effects) const {
			bool ret_val = false;
			if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" || 
				likelihood_type_ == "negative_binomial") {
				ret_val = true;
			}
			else {
				double beta_zero = FindInitialIntercept(y_data, num_data, rand_eff_var, fixed_effects);
				if (std::abs(beta_zero) > 0.1) {
					ret_val = true;
				}
			}
			return(ret_val);
		}

		/*!
		* \brief Determine initial value for additional likelihood parameters (e.g., shape for gamma)
		* \param y_data Response variable data
		* \param num_data Number of data points
		*/
		const double* FindInitialAuxPars(const double* y_data,
			const data_size_t num_data) {
			if (likelihood_type_ == "gamma") {
				// Use a simple "MLE" approach for the shape parameter ignoring random and fixed effects and 
				//	using the approximation: ln(k) - digamma(k) approx = (1 + 1 / (6k + 1)) / (2k), where k = shape
				//	See https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation (as of 02.03.2023)
				double log_avg = 0., avg_log = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_avg, avg_log)
				for (data_size_t i = 0; i < num_data; ++i) {
					log_avg += y_data[i];
					avg_log += std::log(y_data[i]);
				}
				log_avg /= num_data;
				log_avg = std::log(log_avg);
				avg_log /= num_data;
				double s = log_avg - avg_log;
				aux_pars_[0] = (3. - s + std::sqrt((s - 3.) * (s - 3.) + 24. * s)) / (12. * s);
			}
			else if (likelihood_type_ == "negative_binomial") {
				// Use a method of moments estimator
				double avg = 0., sum_sq = 0.;
#pragma omp parallel for schedule(static) reduction(+:avg, sum_sq)
				for (data_size_t i = 0; i < num_data; ++i) {
					avg += y_data[i];
					sum_sq += y_data[i] * y_data[i];
				}
				avg /= num_data;
				double avg_sq = avg * avg;
				double sample_var = (sum_sq - num_data * avg_sq) / (num_data - 1);
				if (sample_var <= avg) {
					aux_pars_[0] = 100 * avg_sq;//marginally no over-disperson in data -> set shape parameter to a large value
					Log::REDebug("FindInitialAuxPars: the internally found initial estimate (MoM) for the shape parameter (%g) might be not very good as there is there is marginally no over-disperion in the data ", aux_pars_[0]);
				}
				else {
					aux_pars_[0] = avg_sq / (sample_var - avg);
				}
			}
			else if (likelihood_type_ != "gaussian" && likelihood_type_ != "bernoulli_probit" &&
				likelihood_type_ != "bernoulli_logit" && likelihood_type_ != "poisson") {
				Log::REFatal("FindInitialAuxPars: Likelihood of type '%s' is not supported ", likelihood_type_.c_str());
			}
			return(aux_pars_.data());
		}//end FindInitialAuxPars

		/*!
		* \brief Determine constants C_mu and C_sigma2 used for checking whether step sizes for linear regression coefficients are clearly too large
		* \param y_data Response variable data
		* \param num_data Number of data points
		* \param fixed_effects Additional fixed effects that are added to the linear predictor (= offset)
		* \param[out] C_mu
		* \param[out] C_sigma2
		*/
		void FindConstantsCapTooLargeLearningRateCoef(const double* y_data,
			const data_size_t num_data,
			const double* fixed_effects,
			double& C_mu,
			double& C_sigma2) const {
			if (likelihood_type_ == "gaussian") {
				double mean = 0., sec_mom = 0;
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:mean, sec_mom)
					for (data_size_t i = 0; i < num_data; ++i) {
						mean += y_data[i];
						sec_mom += y_data[i] * y_data[i];
					}
				}
				else {
#pragma omp parallel for schedule(static) reduction(+:mean, sec_mom)
					for (data_size_t i = 0; i < num_data; ++i) {
						mean += y_data[i] - fixed_effects[i];
						sec_mom += (y_data[i] - fixed_effects[i]) * (y_data[i] - fixed_effects[i]);
					}
				}
				mean /= num_data;
				sec_mom /= num_data;
				C_mu = std::abs(mean);
				C_sigma2 = sec_mom - mean * mean;
			}
			else if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit") {
				C_mu = 1.;
				C_sigma2 = 1.;
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" ||
				likelihood_type_ == "negative_binomial") {
				double mean = 0., sec_mom = 0;
#pragma omp parallel for schedule(static) reduction(+:mean, sec_mom)
				for (data_size_t i = 0; i < num_data; ++i) {
					mean += y_data[i];
					sec_mom += y_data[i] * y_data[i];
				}
				mean /= num_data;
				sec_mom /= num_data;
				C_mu = std::abs(SafeLog(mean));
				C_sigma2 = std::abs(SafeLog(sec_mom - mean * mean));
			}
			else {
				Log::REFatal("FindConstantsCapTooLargeLearningRateCoef: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
			if (C_mu < 1.) {
				C_mu = 1.;
			}
		}//end FindConstantsCapTooLargeLearningRateCoef

		/*!
		* \brief Returns the number of additional parameters
		*/
		int NumAuxPars() const {
			return(num_aux_pars_);
		}

		/*!
		* \brief Returns a pointer to aux_pars_
		*/
		const double* GetAuxPars() const {
			return(aux_pars_.data());
		}

		/*!
		* \brief Set aux_pars_
		* \param aux_pars New values for aux_pars_
		*/
		void SetAuxPars(const double* aux_pars) {
			if (likelihood_type_ == "gaussian" || likelihood_type_ == "gamma" || 
				likelihood_type_ == "negative_binomial") {
				//CHECK(aux_pars[0] > 0);//DELETE
				if (!(aux_pars[0] > 0)) {
					Log::REFatal("The '%s' parameter is not > 0. This might be due to a problem when estimating the '%s' parameter (e.g., a numerical overflow). "
						"You can try either (i) manually setting a different initial value using the 'init_aux_pars' parameter "
						" or (ii) not estimating the '%s' parameter at all by setting 'estimate_aux_pars' to 'false'. "
						"Both these parameters can be specified in the 'params' argument by calling, e.g., the 'set_optim_params' function of a 'GPModel' ",
						names_aux_pars_[0].c_str(), names_aux_pars_[0].c_str(), names_aux_pars_[0].c_str());
				}
				aux_pars_[0] = aux_pars[0];
			}
			normalizing_constant_has_been_calculated_ = false;
			aux_pars_have_been_set_ = true;
		}

		const char* GetNameAuxPars(int ind_aux_par) const {
			CHECK(ind_aux_par < num_aux_pars_);
			return(names_aux_pars_[ind_aux_par].c_str());
		}

		void GetNameFirstAuxPar(string_t& name) const {
			name = names_aux_pars_[0];
		}

		bool AuxParsHaveBeenSet() const {
			return(aux_pars_have_been_set_);
		}

		/*!
		* \brief Calculate the part of the logarithmic normalizing constant of the likelihood that does not depend on aux_pars_ and location_par
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param num_data Number of data points
		*/
		void CalculateAuxQuantLogNormalizingConstant(const double* y_data,
			const int* y_data_int,
			const data_size_t num_data) {
			if (!aux_normalizing_constant_has_been_calculated_) {
				if (likelihood_type_ == "gamma") {
					double log_aux_normalizing_constant = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data; ++i) {
						log_aux_normalizing_constant += AuxQuantLogNormalizingConstantGammaOneSample(y_data[i]);
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (likelihood_type_ == "negative_binomial") {
					double log_aux_normalizing_constant = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_aux_normalizing_constant)
					for (data_size_t i = 0; i < num_data; ++i) {
						log_aux_normalizing_constant += AuxQuantLogNormalizingConstantNegBinOneSample(y_data_int[i]);
					}
					aux_log_normalizing_constant_ = log_aux_normalizing_constant;
				}
				else if (likelihood_type_ != "gaussian" && likelihood_type_ != "bernoulli_probit" &&
					likelihood_type_ != "bernoulli_logit" && likelihood_type_ != "poisson") {
					Log::REFatal("CalculateAuxQuantLogNormalizingConstant: Likelihood of type '%s' is not supported ", likelihood_type_.c_str());
				}
				aux_normalizing_constant_has_been_calculated_ = true;
			}
		}//end CalculateAuxQuantLogNormalizingConstant

		inline double AuxQuantLogNormalizingConstantGammaOneSample(const double y) const {
			return(std::log(y));
		}

		inline double AuxQuantLogNormalizingConstantNegBinOneSample(const int y) const {
			return(-std::lgamma(y + 1));
		}

		/*!
		* \brief Calculate the logarithmic normalizing constant of the likelihood (not depending on location_par)
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param num_data Number of data points
		*/
		void CalculateLogNormalizingConstant(const double* y_data,
			const int* y_data_int,
			const data_size_t num_data) {
			if (!normalizing_constant_has_been_calculated_) {
				if (likelihood_type_ == "poisson") {
					double aux_const = 0.;
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:aux_const)
					for (data_size_t i = 0; i < num_data; ++i) {
						aux_const += LogNormalizingConstantPoissonOneSample(y_data_int[i]);
					}
					log_normalizing_constant_ = aux_const;
				}
				else if (likelihood_type_ == "gamma") {
					log_normalizing_constant_ = LogNormalizingConstantGamma(y_data, y_data_int, num_data);
				}
				else if (likelihood_type_ == "negative_binomial") {
					log_normalizing_constant_ = LogNormalizingConstantNegBin(y_data, y_data_int, num_data);
				}
				else if (likelihood_type_ != "gaussian" && likelihood_type_ != "bernoulli_probit" &&
					likelihood_type_ != "bernoulli_logit") {
					Log::REFatal("CalculateLogNormalizingConstant: Likelihood of type '%s' is not supported ", likelihood_type_.c_str());
				}
				normalizing_constant_has_been_calculated_ = true;
			}
		}//end CalculateLogNormalizingConstant

		inline double LogNormalizingConstantPoissonOneSample(const int y) const {
			if (y > 1) {
				double log_factorial = 0.;
				for (int k = 2; k <= y; ++k) {
					log_factorial += std::log(k);
				}
				return(-log_factorial);
			}
			else {
				return(0.);
			}
		}

		inline double LogNormalizingConstantGamma(const double* y_data, const int* y_data_int, const data_size_t num_data) {
			CalculateAuxQuantLogNormalizingConstant(y_data, y_data_int, num_data);//note: the second argument is not used
			if (TwoNumbersAreEqual<double>(aux_pars_[0], 1.)) {
				return(0.);
			}
			else {
				return((aux_pars_[0] - 1.) * aux_log_normalizing_constant_ +
					num_data * (aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0])));
			}
		}

		inline double LogNormalizingConstantGammaOneSample(const double y) const {
			if (TwoNumbersAreEqual<double>(aux_pars_[0], 1.)) {
				return(0.);
			}
			else {
				return((aux_pars_[0] - 1.) * AuxQuantLogNormalizingConstantGammaOneSample(y) +
					aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0]));
			}
		}

		inline double LogNormalizingConstantNegBin(const double* y_data, const int* y_data_int, const data_size_t num_data) {
			CalculateAuxQuantLogNormalizingConstant(y_data, y_data_int, num_data);//note: the first argument is not used
			double aux_const = 0.;
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:aux_const)
			for (data_size_t i = 0; i < num_data; ++i) {
				aux_const += std::lgamma(y_data_int[i] + aux_pars_[0]);
			}
			double norm_const = aux_const + aux_log_normalizing_constant_ +
				num_data * (aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0]));
			return(norm_const);
		}

		inline double LogNormalizingConstantNegBinOneSample(const int y) const {
			double norm_const = std::lgamma(y + aux_pars_[0]) + AuxQuantLogNormalizingConstantNegBinOneSample(y) + 
				 aux_pars_[0] * std::log(aux_pars_[0]) - std::lgamma(aux_pars_[0]);
			return(norm_const);
		}

		/*!
		* \brief Evaluate the log-likelihood conditional on the latent variable (=location_par)
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param num_data Number of data points
		*/
		double LogLikelihood(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			const data_size_t num_data) {
			CalculateLogNormalizingConstant(y_data, y_data_int, num_data);
			double ll = 0.;
			if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += LogLikBernoulliProbit(y_data_int[i], location_par[i]);
				}
			}
			else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += LogLikBernoulliLogit(y_data_int[i], location_par[i]);
				}
			}
			else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += LogLikPoisson(y_data_int[i], location_par[i], false);
				}
				ll += log_normalizing_constant_;
			}
			else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += LogLikGamma(y_data[i], location_par[i], false);
				}
				ll += log_normalizing_constant_;
			}
			else if (likelihood_type_ == "negative_binomial") {
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += LogLikNegBin(y_data_int[i], location_par[i], false);
				}
				ll += log_normalizing_constant_;
			}
			else if (likelihood_type_ == "gaussian") {
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += LogLikGaussian(y_data[i], location_par[i]);
				}
			}
			else {
				Log::REFatal("LogLikelihood: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
			return(ll);
		}//end LogLikelihood

		/*!
		* \brief Evaluate the log-likelihood conditional on the latent variable (=location_par) for one sample
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		inline double LogLikelihoodOneSample(const double y_data,
			const int y_data_int,
			const double location_par) const {
			if (likelihood_type_ == "bernoulli_probit") {
				return(LogLikBernoulliProbit(y_data_int, location_par));
			}
			else if (likelihood_type_ == "bernoulli_logit") {
				return(LogLikBernoulliLogit(y_data_int, location_par));
			}
			else if (likelihood_type_ == "poisson") {
				return(LogLikPoisson(y_data_int, location_par, true));
			}
			else if (likelihood_type_ == "gamma") {
				return(LogLikGamma(y_data, location_par, true));
			}
			else if (likelihood_type_ == "negative_binomial") {
				return(LogLikNegBin(y_data_int, location_par, true));
			}
			else if (likelihood_type_ == "gaussian") {
				return(LogLikGaussian(y_data, location_par));
			}
			else {
				Log::REFatal("LogLikelihood: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				return(-1e99);
			}
		}//end LogLikelihood

		inline double LogLikBernoulliProbit(const int y, const double location_par) const {
			if (y == 0) {
				return std::log(1 - normalCDF(location_par));
			}
			else {
				return std::log(normalCDF(location_par));
			}
		}

		inline double LogLikBernoulliLogit(const int y, const double location_par) const {
			return (y * location_par - std::log(1 + std::exp(location_par)));
			//Alternative version:
			//if (y == 0) {
			//	ll += std::log(1 - CondMeanLikelihood(location_par));//CondMeanLikelihood = logistic function
			//}
			//else {
			//	ll += std::log(CondMeanLikelihood(location_par));
			//}
		}

		inline double LogLikPoisson(const int y, const double location_par, bool incl_norm_const) const {
			double ll = y * location_par - std::exp(location_par);
			if (incl_norm_const) {
				return (ll + LogNormalizingConstantPoissonOneSample(y));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikGamma(const double y, const double location_par, bool incl_norm_const) const {
			double ll = -aux_pars_[0] * (location_par + y * std::exp(-location_par));
			if (incl_norm_const) {
				return (ll + LogNormalizingConstantGammaOneSample(y));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikNegBin(const int y, const double location_par, bool incl_norm_const) const {
			double ll = y * location_par - (y + aux_pars_[0]) * std::log(std::exp(location_par) + aux_pars_[0]);
			if (incl_norm_const) {
				return (ll + LogNormalizingConstantNegBinOneSample(y));
			}
			else {
				return (ll);
			}
		}

		inline double LogLikGaussian(const double y, const double location_par) const {
			return (std::log(aux_pars_[0]) + normalLogPDF(aux_pars_[0] * (y - location_par)));//aux_pars_[0] = 1. / std::sqrt(variance)
		}

		/*!
		* \brief Calculate the first derivative of the log-likelihood with respect to the location parameter
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		void CalcFirstDerivLogLik(const double* y_data,
			const int* y_data_int,
			const double* location_par) {
			if (!use_Z_for_duplicates_) {
				if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_[i] = FirstDerivLogLikBernoulliProbit(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_[i] = FirstDerivLogLikBernoulliLogit(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_[i] = FirstDerivLogLikPoisson(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_[i] = FirstDerivLogLikGamma(y_data[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "negative_binomial") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_[i] = FirstDerivLogLikNegBin(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "gaussian") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_[i] = FirstDerivLogLikGaussian(y_data[i], location_par[i]);
					}
				}
				else {
					Log::REFatal("CalcFirstDerivLogLik: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				}
			}//end !use_Z_for_duplicates_
			else {//use_Z_for_duplicates_
				if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_data_scale_[i] = FirstDerivLogLikBernoulliProbit(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_data_scale_[i] = FirstDerivLogLikBernoulliLogit(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_data_scale_[i] = FirstDerivLogLikPoisson(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_data_scale_[i] = FirstDerivLogLikGamma(y_data[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "negative_binomial") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_data_scale_[i] = FirstDerivLogLikNegBin(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "gaussian") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						first_deriv_ll_data_scale_[i] = FirstDerivLogLikGaussian(y_data[i], location_par[i]);
					}
				}
				else {
					Log::REFatal("CalcFirstDerivLogLik: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				}
				CalcZtVGivenIndices(num_data_, num_re_, random_effects_indices_of_data_, first_deriv_ll_data_scale_, first_deriv_ll_, true);
			}//end use_Z_for_duplicates_
		}//end CalcFirstDerivLogLik

		/*!
		* \brief Calculate the first derivative of the log-likelihood with respect to the location parameter
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		inline double CalcFirstDerivLogLikOneSample(const double y_data,
			const int y_data_int,
			const double location_par) const {
			if (likelihood_type_ == "bernoulli_probit") {
				return(FirstDerivLogLikBernoulliProbit(y_data_int, location_par));
			}
			else if (likelihood_type_ == "bernoulli_logit") {
				return(FirstDerivLogLikBernoulliLogit(y_data_int, location_par));
			}
			else if (likelihood_type_ == "poisson") {
				return(FirstDerivLogLikPoisson(y_data_int, location_par));
			}
			else if (likelihood_type_ == "gamma") {
				return(FirstDerivLogLikGamma(y_data, location_par));
			}
			else if (likelihood_type_ == "negative_binomial") {
				return(FirstDerivLogLikNegBin(y_data_int, location_par));
			}
			else if (likelihood_type_ == "gaussian") {
				return(FirstDerivLogLikGaussian(y_data, location_par));
			}
			else {
				Log::REFatal("CalcFirstDerivLogLikOneSample: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				return(0.);
			}
		}//end CalcFirstDerivLogLikOneSample

		inline double FirstDerivLogLikBernoulliProbit(const int y, const double location_par) const {
			if (y == 0) {
				return (-normalPDF(location_par) / (1 - normalCDF(location_par)));
			}
			else {
				return (normalPDF(location_par) / normalCDF(location_par));
			}
		}

		inline double FirstDerivLogLikBernoulliLogit(const int y, const double location_par) const {
			return (y - CondMeanLikelihood(location_par));//CondMeanLikelihood = logistic(x)
		}

		inline double FirstDerivLogLikPoisson(const int y, const double location_par) const {
			return (y - std::exp(location_par));
		}

		inline double FirstDerivLogLikGamma(const double y, const double location_par) const {
			return (aux_pars_[0] * (y * std::exp(-location_par) - 1.));
		}

		inline double FirstDerivLogLikNegBin(const int y, const double location_par) const {
			double mu = std::exp(location_par);
			return (y - (y + aux_pars_[0]) / (mu + aux_pars_[0]) * mu);
		}

		inline double FirstDerivLogLikGaussian(const double y, const double location_par) const {
			return (aux_pars_[0] * aux_pars_[0] * (y - location_par));
		}

		/*!
		* \brief Calculate the second derivative of the negative (!) log-likelihood with respect to the location parameter
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		void CalcSecondDerivNegLogLik(const double* y_data,
			const int* y_data_int,
			const double* location_par) {
			if (!use_Z_for_duplicates_) {
				if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_[i] = SecondDerivNegLogLikBernoulliProbit(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_[i] = SecondDerivNegLogLikBernoulliLogit(location_par[i]);
					}
				}
				else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_[i] = SecondDerivNegLogLikPoisson(location_par[i]);
					}
				}
				else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_[i] = SecondDerivNegLogLikGamma(y_data[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "negative_binomial") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_[i] = SecondDerivNegLogLikNegBin(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "gaussian") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_[i] = SecondDerivNegLogLikGaussian();
					}
				}
				else {
					Log::REFatal("CalcSecondDerivNegLogLik: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				}
			}//end !use_Z_for_duplicates_
			else {//use_Z_for_duplicates_
				if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_data_scale_[i] = SecondDerivNegLogLikBernoulliProbit(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_data_scale_[i] = SecondDerivNegLogLikBernoulliLogit(location_par[i]);
					}
				}
				else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_data_scale_[i] = SecondDerivNegLogLikPoisson(location_par[i]);
					}
				}
				else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_data_scale_[i] = SecondDerivNegLogLikGamma(y_data[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "negative_binomial") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_data_scale_[i] = SecondDerivNegLogLikNegBin(y_data_int[i], location_par[i]);
					}
				}
				else if (likelihood_type_ == "gaussian") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
					for (data_size_t i = 0; i < num_data_; ++i) {
						second_deriv_neg_ll_data_scale_[i] = SecondDerivNegLogLikGaussian();
					}
				}
				else {
					Log::REFatal("CalcSecondDerivNegLogLik: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				}
				CalcZtVGivenIndices(num_data_, num_re_, random_effects_indices_of_data_, second_deriv_neg_ll_data_scale_, second_deriv_neg_ll_, true);
			}//end use_Z_for_duplicates_
		}// end CalcSecondDerivNegLogLik

		/*!
		* \brief Calculate the second derivative of the negative (!) log-likelihood with respect to the location parameter
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		*/
		inline double CalcSecondDerivNegLogLikOneSample(const double y_data,
			const int y_data_int,
			const double location_par) const {
			if (likelihood_type_ == "bernoulli_probit") {
				return(SecondDerivNegLogLikBernoulliProbit(y_data_int, location_par));
			}
			else if (likelihood_type_ == "bernoulli_logit") {
				return(SecondDerivNegLogLikBernoulliLogit(location_par));
			}
			else if (likelihood_type_ == "poisson") {
				return(SecondDerivNegLogLikPoisson(location_par));
			}
			else if (likelihood_type_ == "gamma") {
				return(SecondDerivNegLogLikGamma(y_data, location_par));
			}
			else if (likelihood_type_ == "negative_binomial") {
				return(SecondDerivNegLogLikNegBin(y_data_int, location_par));
			}
			else if (likelihood_type_ == "gaussian") {
				return(SecondDerivNegLogLikGaussian());
			}
			else {
				Log::REFatal("CalcSecondDerivNegLogLikOneSample: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				return(1.);
			}
		}// end CalcSecondDerivNegLogLikOneSample

		inline double SecondDerivNegLogLikBernoulliProbit(const int y, const double location_par) const {
			double dnorm = normalPDF(location_par);
			double pnorm = normalCDF(location_par);
			if (y == 0) {
				double dnorm_frac_one_min_pnorm = dnorm / (1. - pnorm);
				return (-dnorm_frac_one_min_pnorm * (location_par - dnorm_frac_one_min_pnorm));
			}
			else {
				double dnorm_frac_pnorm = dnorm / pnorm;
				return(dnorm_frac_pnorm * (location_par + dnorm_frac_pnorm));
			}
		}

		inline double SecondDerivNegLogLikBernoulliLogit(const double location_par) const {
			double exp_loc_i = std::exp(location_par);
			return (exp_loc_i * std::pow(1. + exp_loc_i, -2));
		}

		inline double SecondDerivNegLogLikPoisson(const double location_par) const {
			return std::exp(location_par);
		}

		inline double SecondDerivNegLogLikGamma(const double y, const double location_par) const {
			return (aux_pars_[0] * y * std::exp(-location_par));
		}

		inline double SecondDerivNegLogLikNegBin(const int y, const double location_par) const {
			double mu = std::exp(location_par);
			double mu_plus_r = mu + aux_pars_[0];
			return ((y + aux_pars_[0]) * mu * aux_pars_[0] / (mu_plus_r * mu_plus_r));
		}

		inline double SecondDerivNegLogLikGaussian() const {
			return (aux_pars_[0] * aux_pars_[0]);//aux_pars_[0] = 1 / sqrt(variance)
		}

		/*!
		* \brief Calculate the third derivative of the log-likelihood with respect to the location parameter
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param[out] third_deriv Third derivative of the log-likelihood with respect to the location parameter. Need to pre-allocate memory of size num_data_
		*/
		void CalcThirdDerivLogLik(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			vec_t& third_deriv) const {
			if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					double dnorm = normalPDF(location_par[i]);
					double pnorm = normalCDF(location_par[i]);
					if (y_data_int[i] == 0) {
						double dnorm_frac_one_min_pnorm = dnorm / (1. - pnorm);
						third_deriv[i] = dnorm_frac_one_min_pnorm * (1 - location_par[i] * location_par[i] +
							dnorm_frac_one_min_pnorm * (3 * location_par[i] - 2 * dnorm_frac_one_min_pnorm));
					}
					else {
						double dnorm_frac_pnorm = dnorm / pnorm;
						third_deriv[i] = dnorm_frac_pnorm * (location_par[i] * location_par[i] - 1 +
							dnorm_frac_pnorm * (3 * location_par[i] + 2 * dnorm_frac_pnorm));
					}
				}
			}
			else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					double exp_loc_i = std::exp(location_par[i]);
					third_deriv[i] = -exp_loc_i * (1. - exp_loc_i) * std::pow(1 + exp_loc_i, -3);
				}
			}
			else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					third_deriv[i] = -std::exp(location_par[i]);
				}
			}
			else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					third_deriv[i] = aux_pars_[0] * y_data[i] * std::exp(-location_par[i]);
				}
			}
			else if (likelihood_type_ == "negative_binomial") {
#pragma omp parallel for schedule(static) if (num_data_ >= 128)
				for (data_size_t i = 0; i < num_data_; ++i) {
					double mu = std::exp(location_par[i]);
					double mu_plus_r = mu + aux_pars_[0];
					third_deriv[i] = (y_data_int[i] + aux_pars_[0]) * mu * aux_pars_[0] * (mu - aux_pars_[0]) / (mu_plus_r * mu_plus_r * mu_plus_r);
				}
			}
			else {
				Log::REFatal("CalcThirdDerivLogLik: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
		}//end CalcThirdDerivLogLik

		/*!
		* \brief Calculates the gradient of the negative log-likelihood with respect to the 
		*		additional parameters of the likelihood (e.g., shape for gamma). The gradient is usually calculated on the log-scale.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param num_data Number of data points
		* \param[out] grad Gradient
		*/
		void CalcGradNegLogLikAuxPars(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			const data_size_t num_data,
			double* grad) const {
			if (likelihood_type_ == "gamma") {
				CHECK(aux_normalizing_constant_has_been_calculated_);
				double neg_log_grad = 0.;//gradient for shape parameter is calculated on the log-scale
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad)
				for (data_size_t i = 0; i < num_data; ++i) {
					neg_log_grad += location_par[i] + y_data[i] * std::exp(-location_par[i]);
				}
				neg_log_grad -= num_data * (std::log(aux_pars_[0]) + 1. - digamma(aux_pars_[0]));
				neg_log_grad -= aux_log_normalizing_constant_;
				neg_log_grad *= aux_pars_[0];
				grad[0] = neg_log_grad;
			} 
			else if (likelihood_type_ == "negative_binomial") {
				//gradient for shape parameter is calculated on the log-scale
				double neg_log_grad = 0.;
#pragma omp parallel for schedule(static) reduction(+:neg_log_grad)
				for (data_size_t i = 0; i < num_data; ++i) {
					double mu_plus_r = std::exp(location_par[i]) + aux_pars_[0];
					double y_plus_r = y_data_int[i] + aux_pars_[0];
					neg_log_grad += aux_pars_[0] * (-digamma(y_plus_r) + std::log(mu_plus_r) + y_plus_r / mu_plus_r);
				}
				neg_log_grad += num_data * aux_pars_[0] * (digamma(aux_pars_[0]) - std::log(aux_pars_[0]) - 1);
				grad[0] = neg_log_grad;
			}
			else if (likelihood_type_ != "gaussian" && likelihood_type_ != "bernoulli_probit" &&
				likelihood_type_ != "bernoulli_logit" && likelihood_type_ != "poisson") {
				Log::REFatal("CalcGradNegLogLikAuxPars: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
		}//end CalcGradNegLogLikAuxPars

		/*!
		* \brief Calculates the second and the negative third derivative of the log-likelihood with respect to
		*			(i) once and twice the location parameter and (ii) an additional parameter of the likelihood.
		*			The gradient wrt to the additional parameter is usually calculated on the log-scale.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param location_par Location parameter (random plus fixed effects)
		* \param num_data Number of data points
		* \param ind_aux_par Index of aux_pars_ wrt which the gradient is calculated (currently no used as there is only one)
		* \param[out] second_deriv Second derivative
		* \param[out] neg_third_deriv Negative third derivative
		*/
		void CalcSecondNegThirdDerivLogLikAuxParsLocPar(const double* y_data,
			const int* y_data_int,
			const double* location_par,
			const data_size_t num_data,
			int ind_aux_par,
			double* second_deriv,
			double* neg_third_deriv) const {
			if (likelihood_type_ == "gamma") {
				//note: gradient wrt to aux_pars_[0] on the log-scale
				CHECK(ind_aux_par == 0);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					second_deriv[i] = aux_pars_[0] * (y_data[i] * std::exp(-location_par[i]) - 1.);
					neg_third_deriv[i] = second_deriv[i] + aux_pars_[0];
				}
			}
			else if (likelihood_type_ == "negative_binomial") {
				//gradient for shape parameter is calculated on the log-scale
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					double mu = std::exp(location_par[i]);
					double mu_plus_r = mu + aux_pars_[0];
					double y_plus_r = y_data_int[i] + aux_pars_[0];
					double mu_r_div_mu_plus_r_sqr = mu * aux_pars_[0] / (mu_plus_r * mu_plus_r);
					second_deriv[i] = mu_r_div_mu_plus_r_sqr * (y_data_int[i] - mu);
					neg_third_deriv[i] = -mu_r_div_mu_plus_r_sqr * (y_data_int[i] * (aux_pars_[0] - mu) - 2 * aux_pars_[0] * mu) / y_plus_r;
				}
			}
			else if (likelihood_type_ != "gaussian" && likelihood_type_ != "bernoulli_probit" &&
				likelihood_type_ != "bernoulli_logit" && likelihood_type_ != "poisson") {
				Log::REFatal("CalcSecondDerivNegLogLikAuxParsLocPar: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
		}//end CalcSecondNegThirdDerivLogLikAuxParsLocPar

		/*!
		* \brief Calculate the mean of the likelihood conditional on the (predicted) latent variable
		*			Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double CondMeanLikelihood(const double value) const {
			if (likelihood_type_ == "gaussian") {
				return value;
			}
			else if (likelihood_type_ == "bernoulli_probit") {
				return normalCDF(value);
			}
			else if (likelihood_type_ == "bernoulli_logit") {
				return 1. / (1. + std::exp(-value));
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" ||
				likelihood_type_ == "negative_binomial") {
				return std::exp(value);
			}
			else {
				Log::REFatal("CondMeanLikelihood: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				return 0.;
			}
		}

		/*!
		* \brief Calculate the first derivative of the logarithm of the mean of the likelihood conditional on the (predicted) latent variable
		*			Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double FirstDerivLogCondMeanLikelihood(const double value) const {
			if (likelihood_type_ == "bernoulli_logit") {
				return 1. / (1. + std::exp(value));
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" ||
				likelihood_type_ == "negative_binomial") {
				return 1.;
			}
			else {
				Log::REFatal("FirstDerivLogCondMeanLikelihood: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				return 0.;
			}
		}

		/*!
		* \brief Calculate the second derivative of the logarithm of the mean of the likelihood conditional on the (predicted) latent variable
		*			Used for adaptive Gauss-Hermite quadrature for the prediction of the response variable
		*/
		inline double SecondDerivLogCondMeanLikelihood(const double value) const {
			if (likelihood_type_ == "bernoulli_logit") {
				double exp_x = std::exp(value);
				return -exp_x / ((1. + exp_x) * (1. + exp_x));
			}
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma" ||
				likelihood_type_ == "negative_binomial") {
				return 0.;
			}
			else {
				Log::REFatal("SecondDerivLogCondMeanLikelihood: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
				return 0.;
			}
		}

		/*!
		* \brief Do Cholesky decomposition
		* \param[out] chol_fact Cholesky factor
		* \param psi Matrix for which the Cholesky decomposition should be done
		*/
		template <class T_mat_1, typename std::enable_if <std::is_same<sp_mat_t, T_mat_1>::value ||
			std::is_same<sp_mat_rm_t, T_mat_1>::value>::type* = nullptr >
		void CalcChol(T_chol& chol_fact, const T_mat_1& psi) {
			if (!chol_fact_pattern_analyzed_) {
				chol_fact.analyzePattern(psi);
				chol_fact_pattern_analyzed_ = true;
			}
			chol_fact.factorize(psi);
		}
		template <class T_mat_1, typename std::enable_if <std::is_same<den_mat_t, T_mat_1>::value>::type* = nullptr  >
		void CalcChol(T_chol& chol_fact, const T_mat_1& psi) {
			chol_fact.compute(psi);
		}

		// Initialize location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
		/*!
		* \brief Auxiliary function for initializinh the location parameter = mode of random effects + fixed effects
		* \param fixed_effects Fixed effects component of location parameter
		* \param[out] location_par Locatioon parameter (used only if use_Z_for_duplicates_)
		* \param[out] location_par_ptr Pointer to location parameter
		*/
		void InitializeLocationPar(const double* fixed_effects,
			vec_t& location_par,
			double** location_par_ptr) {
			if (use_Z_for_duplicates_) {
				location_par = vec_t(num_data_);
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						location_par[i] = mode_[random_effects_indices_of_data_[i]];
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						location_par[i] = mode_[random_effects_indices_of_data_[i]] + fixed_effects[i];
					}
				}
				*location_par_ptr = location_par.data();
			}//end use_Z_for_duplicates_
			else {
				if (fixed_effects == nullptr) {
					*location_par_ptr = mode_.data();
				}
				else {
					location_par = vec_t(num_data_);
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						location_par[i] = mode_[i] + fixed_effects[i];
					}
					*location_par_ptr = location_par.data();
				}
			}//end initialize location parameter
		}

		/*!
		* \brief Auxiliary function for updating the location parameter = mode of random effects + fixed effects
		* \param fixed_effects Fixed effects component of location parameter
		* \param[out] location_par Location parameter
		*/
		void UpdateLocationPar(const double* fixed_effects,
			vec_t& location_par) {
			if (use_Z_for_duplicates_) {
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						location_par[i] = mode_[random_effects_indices_of_data_[i]];
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						location_par[i] = mode_[random_effects_indices_of_data_[i]] + fixed_effects[i];
					}
				}
			}//end use_Z_for_duplicates_
			else if (fixed_effects != nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					location_par[i] = mode_[i] + fixed_effects[i];
				}
			}
		}//end UpdateLocationPar

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood..
		*		Calculations are done using a numerically stable variant based on factorizing ("inverting") B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt).
		*		In the notation of the paper: "Sigma = Z*Sigma*Z^T" and "Z = Id".
		*		This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		*		If use_Z_for_duplicates_, calculations are done on the random effects (b) scale and not the "data scale" (Zb) 
		*		factorizing ("inverting") B = (Id + ZtWZsqrt * Sigma * ZtWZsqrt).
		*		This version (use_Z_for_duplicates_ == true) is used for the Laplace approximation when there is only one Gaussian process and
		*		there are multiple observations at the same location, i.e., the dimenion of the random effects b is much smaller than Zb
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param Sigma Covariance matrix of latent random effect ("Sigma = Z*Sigma*Z^T" if !use_Z_for_duplicates_)
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<T_mat> Sigma,
			double& approx_marginal_ll) {
			// Initialize variables
			if (!mode_initialized_) {
				InitializeModeAvec();
			}
			else {
				mode_previous_value_ = mode_;
				a_vec_previous_value_ = a_vec_;
				na_or_inf_during_second_last_call_to_find_mode_ = na_or_inf_during_last_call_to_find_mode_;
			}
			vec_t location_par;//location parameter = mode of random effects + fixed effects
			double* location_par_ptr;
			InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
			// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
			approx_marginal_ll = -0.5 * (a_vec_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par_ptr, num_data_);
			double approx_marginal_ll_new = approx_marginal_ll;
			vec_t rhs(dim_mode_), rhs2(dim_mode_);//auxiliary variables for updating mode
			vec_t diag_Wsqrt(dim_mode_);//diagonal of matrix sqrt(ZtWZ) if use_Z_for_duplicates_ or sqrt(W) if !use_Z_for_duplicates_ with square root of negative second derivatives of log-likelihood
			T_mat Id_plus_Wsqrt_Sigma_Wsqrt(dim_mode_, dim_mode_);// = Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt if use_Z_for_duplicates_ or Id_plus_Wsqrt_ZSigmaZt_Wsqrt if !use_Z_for_duplicates_
			// Start finding mode 
			int it;
			bool terminate_optim = false;
			bool has_NA_or_Inf = false;
			for (it = 0; it < MAXIT_MODE_NEWTON_; ++it) {
				// Calculate first and second derivative of log-likelihood
				CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par_ptr);
				// Calculate right hand side for mode update and sqrt(ZtWZ) / sqrt(W)
				diag_Wsqrt.array() = second_deriv_neg_ll_.array().sqrt();
				rhs.array() = second_deriv_neg_ll_.array() * mode_.array() + first_deriv_ll_.array();
				// Calculate Cholesky factor of matrix B = (Id + ZtWZsqrt * Sigma * ZtWZsqrt) if use_Z_for_duplicates_ or B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt) if !use_Z_for_duplicates_
				Id_plus_Wsqrt_Sigma_Wsqrt.setIdentity();
				Id_plus_Wsqrt_Sigma_Wsqrt += (diag_Wsqrt.asDiagonal() * (*Sigma) * diag_Wsqrt.asDiagonal());
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_Wsqrt_Sigma_Wsqrt);//this is the bottleneck (for large data and sparse matrices)
				// Update mode and a_vec_
				rhs2 = (*Sigma) * rhs;//rhs2 = sqrt(W) * Sigma * rhs
				rhs2.array() *= diag_Wsqrt.array();
				a_vec_ = -chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_.solve(rhs2);//a_vec_ = rhs - sqrt(W) * Id_plus_Wsqrt_Sigma_Wsqrt^-1 * rhs2
				a_vec_.array() *= diag_Wsqrt.array();
				a_vec_.array() += rhs.array();
				mode_ = (*Sigma) * a_vec_;
				UpdateLocationPar(fixed_effects, location_par); // Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
				// Calculate new objective function
				approx_marginal_ll_new = -0.5 * (a_vec_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par_ptr, num_data_);
				if (std::isnan(approx_marginal_ll_new) || std::isinf(approx_marginal_ll_new)) {
					has_NA_or_Inf = true;
					Log::REDebug(NA_OR_INF_WARNING_);
					break;
				}
				// Check convergence
				if (it == 0) {
					if (std::abs(approx_marginal_ll_new - approx_marginal_ll) < DELTA_REL_CONV_ * std::abs(approx_marginal_ll)) { // allow for decreases in first iteration
						terminate_optim = true;
					}
				}
				else {
					if ((approx_marginal_ll_new - approx_marginal_ll) < DELTA_REL_CONV_ * std::abs(approx_marginal_ll)) {
						terminate_optim = true;
					}
				}
				if (terminate_optim) {
					if (approx_marginal_ll_new < approx_marginal_ll) {
						Log::REDebug(NO_INCREASE_IN_MLL_WARNING_);
					}
					approx_marginal_ll = approx_marginal_ll_new;
					break;
				}
				else {
					approx_marginal_ll = approx_marginal_ll_new;
				}
			}
			if (it == MAXIT_MODE_NEWTON_) {
				Log::REDebug(NO_CONVERGENCE_WARNING_);
			}
			if (has_NA_or_Inf) {
				approx_marginal_ll = approx_marginal_ll_new;
				na_or_inf_during_last_call_to_find_mode_ = true;
			}
			else {
				CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par_ptr);
				diag_Wsqrt.array() = second_deriv_neg_ll_.array().sqrt();
				Id_plus_Wsqrt_Sigma_Wsqrt.setIdentity();
				Id_plus_Wsqrt_Sigma_Wsqrt += (diag_Wsqrt.asDiagonal() * (*Sigma) * diag_Wsqrt.asDiagonal());
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_Wsqrt_Sigma_Wsqrt);
				approx_marginal_ll -= ((T_mat)chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_.matrixL()).diagonal().array().log().sum();
				mode_has_been_calculated_ = true;
				na_or_inf_during_last_call_to_find_mode_ = false;
			}
		}//end FindModePostRandEffCalcMLLStable

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*		This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param SigmaI Inverse covariance matrix of latent random effect. Currently, this needs to be a diagonal matrix
		* \param Zt Transpose Z^T of random effect design matrix that relates latent random effects to observations/likelihoods
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLGroupedRE(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const sp_mat_t& SigmaI,
			const sp_mat_t& Zt,
			double& approx_marginal_ll) {
			// Initialize variables
			if (!mode_initialized_) {
				InitializeModeAvec();
			}
			else {
				mode_previous_value_ = mode_;
				na_or_inf_during_second_last_call_to_find_mode_ = na_or_inf_during_last_call_to_find_mode_;
			}
			sp_mat_t Z = Zt.transpose();
			vec_t location_par = Z * mode_;//location parameter = mode of random effects + fixed effects
			if (fixed_effects != nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] += fixed_effects[i];
				}
			}
			// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
			approx_marginal_ll = -0.5 * (mode_.dot(SigmaI * mode_)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
			double approx_marginal_ll_new = approx_marginal_ll;
			sp_mat_t SigmaI_plus_ZtWZ;
			vec_t rhs;
			// Start finding mode 
			int it;
			bool terminate_optim = false;
			bool has_NA_or_Inf = false;
			for (it = 0; it < MAXIT_MODE_NEWTON_; ++it) {
				// Calculate first and second derivative of log-likelihood
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data());
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data());
				// Calculate Cholesky factor and update mode
				rhs = Zt * first_deriv_ll_ - SigmaI * mode_;//right hand side for updating mode
				SigmaI_plus_ZtWZ = SigmaI + Zt * second_deriv_neg_ll_.asDiagonal() * Z;
				SigmaI_plus_ZtWZ.makeCompressed();
				if (!chol_fact_pattern_analyzed_) {
					chol_fact_SigmaI_plus_ZtWZ_grouped_.analyzePattern(SigmaI_plus_ZtWZ);
					chol_fact_pattern_analyzed_ = true;
				}
				chol_fact_SigmaI_plus_ZtWZ_grouped_.factorize(SigmaI_plus_ZtWZ);
				mode_ += chol_fact_SigmaI_plus_ZtWZ_grouped_.solve(rhs);
				// Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
				location_par = Z * mode_;
				if (fixed_effects != nullptr) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						location_par[i] += fixed_effects[i];
					}
				}
				// Calculate new objective function
				approx_marginal_ll_new = -0.5 * (mode_.dot(SigmaI * mode_)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
				if (std::isnan(approx_marginal_ll_new) || std::isinf(approx_marginal_ll_new)) {
					has_NA_or_Inf = true;
					Log::REDebug(NA_OR_INF_WARNING_);
					break;
				}
				// Check convergence
				if (it == 0) {
					if (std::abs(approx_marginal_ll_new - approx_marginal_ll) < DELTA_REL_CONV_ * std::abs(approx_marginal_ll)) { // allow for decreases in first iteration
						terminate_optim = true;
					}
				}
				else {
					if ((approx_marginal_ll_new - approx_marginal_ll) < DELTA_REL_CONV_ * std::abs(approx_marginal_ll)) {
						terminate_optim = true;
					}
				}
				if (terminate_optim) {
					if (approx_marginal_ll_new < approx_marginal_ll) {
						Log::REDebug(NO_INCREASE_IN_MLL_WARNING_);
					}
					approx_marginal_ll = approx_marginal_ll_new;
					break;
				}
				else {
					approx_marginal_ll = approx_marginal_ll_new;
				}
			}//end mode finding algorithm
			if (it == MAXIT_MODE_NEWTON_) {
				Log::REDebug(NO_CONVERGENCE_WARNING_);
			}
			if (has_NA_or_Inf) {
				approx_marginal_ll = approx_marginal_ll_new;
				na_or_inf_during_last_call_to_find_mode_ = true;
			}
			else {
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data());//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data());
				SigmaI_plus_ZtWZ = SigmaI + Zt * second_deriv_neg_ll_.asDiagonal() * Z;
				SigmaI_plus_ZtWZ.makeCompressed();
				chol_fact_SigmaI_plus_ZtWZ_grouped_.factorize(SigmaI_plus_ZtWZ);
				approx_marginal_ll += -((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_grouped_.matrixL()).diagonal().array().log().sum() + 0.5 * SigmaI.diagonal().array().log().sum();
				mode_has_been_calculated_ = true;
				na_or_inf_during_last_call_to_find_mode_ = false;
			}
		}//end FindModePostRandEffCalcMLLGroupedRE

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param sigma2 Variance of random effects
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const double sigma2,
			const data_size_t* const random_effects_indices_of_data,
			double& approx_marginal_ll) {
			// Initialize variables
			if (!mode_initialized_) {
				InitializeModeAvec();
			}
			else {
				mode_previous_value_ = mode_;
				na_or_inf_during_second_last_call_to_find_mode_ = na_or_inf_during_last_call_to_find_mode_;
			}
			vec_t location_par(num_data);//location parameter = mode of random effects + fixed effects
			if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] = mode_[random_effects_indices_of_data[i]];
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] = mode_[random_effects_indices_of_data[i]] + fixed_effects[i];
				}
			}
			// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
			approx_marginal_ll = -0.5 / sigma2 * (mode_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
			double approx_marginal_ll_new = approx_marginal_ll;
			vec_t rhs;
			diag_SigmaI_plus_ZtWZ_ = vec_t(num_re_);
			// Start finding mode 
			int it;
			bool terminate_optim = false;
			bool has_NA_or_Inf = false;
			for (it = 0; it < MAXIT_MODE_NEWTON_; ++it) {
				// Calculate first and second derivative of log-likelihood
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data());
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data());
				// Calculate rhs for mode update
				rhs = -mode_ / sigma2;//right hand side for updating mode
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, first_deriv_ll_, rhs, false);
				// Update mode
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv_neg_ll_, diag_SigmaI_plus_ZtWZ_, true);
				diag_SigmaI_plus_ZtWZ_.array() += 1. / sigma2;
				mode_ += (rhs.array() / diag_SigmaI_plus_ZtWZ_.array()).matrix();
				// Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
				if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						location_par[i] = mode_[random_effects_indices_of_data[i]];
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						location_par[i] = mode_[random_effects_indices_of_data[i]] + fixed_effects[i];
					}
				}
				// Calculate new objective function
				approx_marginal_ll_new = -0.5 / sigma2 * (mode_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
				if (std::isnan(approx_marginal_ll_new) || std::isinf(approx_marginal_ll_new)) {
					has_NA_or_Inf = true;
					Log::REDebug(NA_OR_INF_WARNING_);
					break;
				}
				// Check convergence
				if (it == 0) {
					if (std::abs(approx_marginal_ll_new - approx_marginal_ll) < DELTA_REL_CONV_ * std::abs(approx_marginal_ll)) { // allow for decreases in first iteration
						terminate_optim = true;
					}
				}
				else {
					if ((approx_marginal_ll_new - approx_marginal_ll) < DELTA_REL_CONV_ * std::abs(approx_marginal_ll)) {
						terminate_optim = true;
					}
				}
				if (terminate_optim) {
					if (approx_marginal_ll_new < approx_marginal_ll) {
						Log::REDebug(NO_INCREASE_IN_MLL_WARNING_);
					}
					approx_marginal_ll = approx_marginal_ll_new;
					break;
				}
				else {
					approx_marginal_ll = approx_marginal_ll_new;
				}
			}//end mode finding algorithm
			if (it == MAXIT_MODE_NEWTON_) {
				Log::REDebug(NO_CONVERGENCE_WARNING_);
			}
			if (has_NA_or_Inf) {
				approx_marginal_ll = approx_marginal_ll_new;
				na_or_inf_during_last_call_to_find_mode_ = true;
			}
			else {
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data());//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data());
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv_neg_ll_, diag_SigmaI_plus_ZtWZ_, true);
				diag_SigmaI_plus_ZtWZ_.array() += 1. / sigma2;
				approx_marginal_ll -= 0.5 * diag_SigmaI_plus_ZtWZ_.array().log().sum() + 0.5 * num_re_ * std::log(sigma2);
				mode_has_been_calculated_ = true;
				na_or_inf_during_last_call_to_find_mode_ = false;
			}
		}//end FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param B Matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation Sigma^-1 = B^T D^-1 B
		* \param first_update If true, the covariance parameters or linear regression coefficients are updated for the first time and the max. number of iterations for the CG should be decreased
		* \param Sigma_L_k Pivoted Cholseky decomposition of Sigma - Version Habrecht: matrix of dimension nxk with rank(Sigma_L_k_) <= piv_chol_rank generated in re_model_template.h
		* \param calc_mll If true the marginal log-likelihood is also calculated (only relevant for matrix_inversion_method_ == "iterative")
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const bool first_update,
			const den_mat_t& Sigma_L_k,
			bool calc_mll,
			double& approx_marginal_ll) {
			// Initialize variables
			if (!mode_initialized_) {
				InitializeModeAvec();
			}
			else {
				mode_previous_value_ = mode_;
				na_or_inf_during_second_last_call_to_find_mode_ = na_or_inf_during_last_call_to_find_mode_;
			}
			vec_t location_par;//location parameter = mode of random effects + fixed effects
			double* location_par_ptr;
			vec_t rhs, B_mode, mode_new(dim_mode_);
			// Variables when using Cholesky factorization
			sp_mat_t SigmaI, SigmaI_plus_W;
			if (cg_preconditioner_type_ == "piv_chol_on_Sigma") { //TEMP: incomp_chol_on_SigmaI_plus_W
				cg_preconditioner_type_ = "incomp_chol_on_SigmaI_plus_W"; //TEMP: incomp_chol_on_SigmaI_plus_W
			} //TEMP: incomp_chol_on_SigmaI_plus_W
			vec_t mode_after_grad_aux, mode_after_grad_aux_lag1;//auxiliary variable used only if quasi_newton_for_mode_finding_
			if (quasi_newton_for_mode_finding_) {
				mode_after_grad_aux_lag1 = mode_;
			}
			// Variables when using iterative methods
			int cg_max_num_it = cg_max_num_it_;
			int cg_max_num_it_tridiag = cg_max_num_it_tridiag_;
			den_mat_t I_k_plus_Sigma_L_kt_W_Sigma_L_k;
			InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
			// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
			B_mode = B * mode_;
			approx_marginal_ll = -0.5 * (B_mode.dot(D_inv * B_mode)) + LogLikelihood(y_data, y_data_int, location_par_ptr, num_data_);
			double approx_marginal_ll_new = approx_marginal_ll;
			if (matrix_inversion_method_ == "iterative") {
				//Reduce max. number of iterations for the CG in first update
				if (first_update && reduce_cg_max_num_it_first_optim_step_) {
					cg_max_num_it = (int)round(cg_max_num_it_ / 3);
					cg_max_num_it_tridiag = (int)round(cg_max_num_it_tridiag_ / 3);
				}
				//Convert to row-major for parallelization
				B_rm_ = sp_mat_rm_t(B);
				D_inv_rm_ = sp_mat_rm_t(D_inv);
				B_t_D_inv_rm_ = B_rm_.transpose() * D_inv_rm_;
				if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
					//Store as class variable
					Sigma_L_k_ = Sigma_L_k;
					I_k_plus_Sigma_L_kt_W_Sigma_L_k.resize(Sigma_L_k_.cols(), Sigma_L_k_.cols());
				}
				else if (cg_preconditioner_type_ == "incomp_chol_on_SigmaI_plus_W") {
					SigmaI = B.transpose() * D_inv * B;
				}
			}
			else {
				SigmaI = B.transpose() * D_inv * B;
			}
			// Start finding mode 
			int it;
			bool terminate_optim = false;
			bool has_NA_or_Inf = false;
			double lr_GD = 1.;// learning rate for gradient descent
			for (it = 0; it < MAXIT_MODE_NEWTON_; ++it) {
				// Calculate first and second derivative of log-likelihood
				CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par_ptr);
				if (quasi_newton_for_mode_finding_) {
					vec_t grad = first_deriv_ll_ - B.transpose() * (D_inv * (B * mode_));
					sp_mat_t D_inv_B = D_inv * B;
					sp_mat_t Bt_D_inv_B_aux = B.cwiseProduct(D_inv_B);
					vec_t SigmaI_diag = Bt_D_inv_B_aux.transpose() * vec_t::Ones(Bt_D_inv_B_aux.rows());
					grad.array() /= (second_deriv_neg_ll_.array() + SigmaI_diag.array());
					//// Alternative way approximating W + Sigma^-1 with Bt * (W + D^-1) * B. 
					//// Note: seems to work worse compared to above diagonal approach. Also, better to comment out "nesterov_acc_rate *= 0.5;"
					//vec_t grad_aux = B.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(grad);
					//grad_aux.array() /= (D_inv.diagonal().array() + second_deriv_neg_ll_.array());
					//grad = B.triangularView<Eigen::UpLoType::UnitLower>().solve(grad_aux);
					lr_GD = 1.;
					double nesterov_acc_rate = (1. - (3. / (6. + it)));//Nesterov acceleration factor
					for (int ih = 0; ih < MAX_NUMBER_LR_SHRINKAGE_STEPS_; ++ih) {
						mode_after_grad_aux = mode_ + lr_GD * grad;
						REModelTemplate<T_mat, T_chol>::ApplyMomentumStep(it, mode_after_grad_aux, mode_after_grad_aux_lag1,
							mode_new, nesterov_acc_rate, 0, false, 2, false);
						if (cap_change_mode_newton_) {
#pragma omp parallel for schedule(static)
							for (data_size_t i = 0; i < dim_mode_; ++i) {
								double abs_change = std::abs(mode_new[i] - mode_[i]);
								if (abs_change > MAX_CHANGE_MODE_NEWTON_) {
									mode_new[i] = mode_[i] + (mode_new[i] - mode_[i]) / abs_change * MAX_CHANGE_MODE_NEWTON_;
								}
							}
						}//end cap_change_mode_newton_
						B_mode = B * mode_new;
						UpdateLocationPar(fixed_effects, location_par); // Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
						approx_marginal_ll_new = -0.5 * (B_mode.dot(D_inv * B_mode)) + LogLikelihood(y_data, y_data_int, location_par_ptr, num_data_);
						if (approx_marginal_ll_new < approx_marginal_ll ||
							std::isnan(approx_marginal_ll_new) || std::isinf(approx_marginal_ll_new)) {
							lr_GD *= 0.5;
							nesterov_acc_rate *= 0.5;
						}
						else {//approx_marginal_ll_new >= approx_marginal_ll
							mode_ = mode_new;
							break;
						}
					}// end loop over learnig rate halving procedure
					mode_after_grad_aux_lag1 = mode_after_grad_aux;
				}//end quasi_newton_for_mode_finding_
				else {//Newton's method
					// Calculate Cholesky factor and update mode
					rhs.array() = second_deriv_neg_ll_.array() * mode_.array() + first_deriv_ll_.array();//right hand side for updating mode
					if (matrix_inversion_method_ == "iterative") {
						if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
							if ((second_deriv_neg_ll_.array() > 1e10).any()) {
								has_NA_or_Inf = true;// the inversion of the preconditioner with the Woodbury identity can be numerically unstable when second_deriv_neg_ll_ is very large
							}
							else {
								I_k_plus_Sigma_L_kt_W_Sigma_L_k.setIdentity();
								I_k_plus_Sigma_L_kt_W_Sigma_L_k += Sigma_L_k_.transpose() * second_deriv_neg_ll_.asDiagonal() * Sigma_L_k_;
								chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.compute(I_k_plus_Sigma_L_kt_W_Sigma_L_k);
								CGVecchiaLaplaceVecWinvplusSigma(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_.transpose(), rhs, mode_new, has_NA_or_Inf,
									cg_max_num_it, it, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_);
							}
						}
						else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB") {
							D_inv_plus_W_B_rm_ = (D_inv_rm_.diagonal() + second_deriv_neg_ll_).asDiagonal() * B_rm_;
							CGVecchiaLaplaceVec(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_, rhs, mode_new, has_NA_or_Inf,
								cg_max_num_it, it, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, D_inv_plus_W_B_rm_);
						}
						else if (cg_preconditioner_type_ == "incomp_chol_on_SigmaI_plus_W") {
							std::chrono::steady_clock::time_point begin, end;
							double el_time;
							begin = std::chrono::steady_clock::now();
								//Log::REInfo("SigmaI.nonZeros() = %d", SigmaI.nonZeros());
								SigmaI_plus_W = SigmaI;
								SigmaI_plus_W.diagonal().array() += second_deriv_neg_ll_.array();
								ReverseIncompleteCholeskyFactorization(SigmaI_plus_W, B, L_SigmaI_plus_W_rm_);
							end = std::chrono::steady_clock::now();
							el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
							Log::REInfo("Time ReverseIncompleteCholeskyFactorization: %g", el_time);
							begin = std::chrono::steady_clock::now();
								CGVecchiaLaplaceVecIncompChol(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_, rhs, mode_new, has_NA_or_Inf,
									cg_max_num_it, it, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, L_SigmaI_plus_W_rm_);
							end = std::chrono::steady_clock::now();
							el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
							Log::REInfo("Time CGVecchiaLaplaceVecIncompChol: %g", el_time);
						}
						else {
							Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
						}
						if (has_NA_or_Inf) {
							approx_marginal_ll_new = std::numeric_limits<double>::quiet_NaN();
							Log::REDebug(NA_OR_INF_WARNING_);
							break;
						}
					} //end iterative
					else { // start Cholesky 
						SigmaI_plus_W = SigmaI;
						SigmaI_plus_W.diagonal().array() += second_deriv_neg_ll_.array();
						SigmaI_plus_W.makeCompressed();
						//Calculation of the Cholesky factor is the bottleneck
						if (!chol_fact_pattern_analyzed_) {
							chol_fact_SigmaI_plus_ZtWZ_vecchia_.analyzePattern(SigmaI_plus_W);
							chol_fact_pattern_analyzed_ = true;
						}
						chol_fact_SigmaI_plus_ZtWZ_vecchia_.factorize(SigmaI_plus_W);//This is the bottleneck for large data
						//Log::REInfo("SigmaI_plus_W: number non zeros = %d", (int)SigmaI_plus_W.nonZeros());//only for debugging
						//Log::REInfo("chol_fact_SigmaI_plus_ZtWZ: Number non zeros = %d", (int)((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_vecchia_.matrixL()).nonZeros());//only for debugging
						mode_new = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(rhs);
					} // end Cholesky
					if (cap_change_mode_newton_) {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < dim_mode_; ++i) {
							double abs_change = std::abs(mode_new[i] - mode_[i]);
							if (abs_change > MAX_CHANGE_MODE_NEWTON_) {
								mode_[i] = mode_[i] + (mode_new[i] - mode_[i]) / abs_change * MAX_CHANGE_MODE_NEWTON_;
							}
							else {
								mode_[i] = mode_new[i];
							}
						}
					}//end cap_change_mode_newton_
					else {
						mode_ = mode_new;
					}
					// Calculate new objective function
					UpdateLocationPar(fixed_effects, location_par); // Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
					B_mode = B * mode_;
					approx_marginal_ll_new = -0.5 * (B_mode.dot(D_inv * B_mode)) + LogLikelihood(y_data, y_data_int, location_par_ptr, num_data_);
				}//end Newton's method
				if (std::isnan(approx_marginal_ll_new) || std::isinf(approx_marginal_ll_new)) {
					has_NA_or_Inf = true;
					Log::REDebug(NA_OR_INF_WARNING_);
					break;
				}
				// Check convergence
				if (it == 0) {
					if (std::abs(approx_marginal_ll_new - approx_marginal_ll) < DELTA_REL_CONV_ * std::abs(approx_marginal_ll)) { // allow for decreases in first iteration
						terminate_optim = true;
					}
				}
				else {
					if ((approx_marginal_ll_new - approx_marginal_ll) < DELTA_REL_CONV_ * std::abs(approx_marginal_ll)) {
						terminate_optim = true;
					}
				}
				if (terminate_optim) {
					if (approx_marginal_ll_new < approx_marginal_ll) {
						Log::REDebug(NO_INCREASE_IN_MLL_WARNING_);
					}
					approx_marginal_ll = approx_marginal_ll_new;
					break;
				}
				else {
					approx_marginal_ll = approx_marginal_ll_new;
				}
			} // end loop for mode finding
			if (it == MAXIT_MODE_NEWTON_) {
				Log::REDebug(NO_CONVERGENCE_WARNING_);
			}
			if (has_NA_or_Inf) {
				approx_marginal_ll = approx_marginal_ll_new;
				na_or_inf_during_last_call_to_find_mode_ = true;
			}
			else {
				mode_has_been_calculated_ = true;
				na_or_inf_during_last_call_to_find_mode_ = false;
				CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par_ptr);
				if (matrix_inversion_method_ == "iterative") {
					//Seed Generator
					if (!cg_generator_seeded_) {
						cg_generator_ = RNG_t(seed_rand_vec_trace_);
						cg_generator_seeded_ = true;
					}
					if (calc_mll) {//calculate determinant term for approx_marginal_ll
						//Generate random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I
						if (!saved_rand_vec_trace_) {
							//Dependent on the preconditioner: Generate t (= num_rand_vec_trace_) or 2*t random vectors
							if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
								rand_vec_trace_I_.resize(dim_mode_, num_rand_vec_trace_);
								rand_vec_trace_I2_.resize(std::min(piv_chol_rank_, dim_mode_), num_rand_vec_trace_);
								GenRandVecTrace(cg_generator_, rand_vec_trace_I2_);
								WI_plus_Sigma_inv_Z_.resize(dim_mode_, num_rand_vec_trace_);
							}
							else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB" || cg_preconditioner_type_ == "incomp_chol_on_SigmaI_plus_W") {
								rand_vec_trace_I_.resize(dim_mode_, num_rand_vec_trace_);
								SigmaI_plus_W_inv_Z_.resize(dim_mode_, num_rand_vec_trace_);
							}
							else {
								Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
							}
							GenRandVecTrace(cg_generator_, rand_vec_trace_I_);
							if (reuse_rand_vec_trace_) {
								saved_rand_vec_trace_ = true;
							}
							rand_vec_trace_P_.resize(dim_mode_, num_rand_vec_trace_);
						}
						double log_det_Sigma_W_plus_I;
						CalcLogDetStoch(dim_mode_, cg_max_num_it_tridiag, I_k_plus_Sigma_L_kt_W_Sigma_L_k, has_NA_or_Inf, log_det_Sigma_W_plus_I, SigmaI, B);
						if (has_NA_or_Inf) {
							approx_marginal_ll = std::numeric_limits<double>::quiet_NaN();
							Log::REDebug(NA_OR_INF_WARNING_);
							na_or_inf_during_last_call_to_find_mode_ = true;
						}
						else {
							approx_marginal_ll -= 0.5 * log_det_Sigma_W_plus_I;
						}
					}//end calculate determinant term for approx_marginal_ll
				}//end iterative
				else {
					SigmaI_plus_W = SigmaI;
					SigmaI_plus_W.diagonal().array() += second_deriv_neg_ll_.array();
					SigmaI_plus_W.makeCompressed();
					if (!chol_fact_pattern_analyzed_) {
						chol_fact_SigmaI_plus_ZtWZ_vecchia_.analyzePattern(SigmaI_plus_W);
						chol_fact_pattern_analyzed_ = true;
					}
					chol_fact_SigmaI_plus_ZtWZ_vecchia_.factorize(SigmaI_plus_W);
					approx_marginal_ll += -((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_vecchia_.matrixL()).diagonal().array().log().sum() + 0.5 * D_inv.diagonal().array().log().sum();
				}
			}
			//Log::REInfo("FindModePostRandEffCalcMLLVecchia: finished after %d iterations ", it);//for debugging
		}//end FindModePostRandEffCalcMLLVecchia

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*		fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*		Calculations are done using a numerically stable variant based on factorizing ("inverting") B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt).
		*		In the notation of the paper: "Sigma = Z*Sigma*Z^T" and "Z = Id".
		*		This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		*		If use_Z_for_duplicates_, calculations are done on the random effects (b) scale and not the "data scale" (Zb) 
		*		factorizing ("inverting") B = (Id + ZtWZsqrt * Sigma * ZtWZsqrt).
		*		This version (use_Z_for_duplicates_ == true) is used for the Laplace approximation when there is only one Gaussian process and
		*		there are multiple observations at the same location, i.e., the dimenion of the random effects b is much smaller than Zb
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param Sigma Covariance matrix of latent random effect ("Sigma = Z*Sigma*Z^T" if !use_Z_for_duplicates_)
		* \param re_comps_cluster_i Vector with different random effects components. We pass the component pointers to save memory in order to avoid passing a large collection of gardient covariance matrices in memory//TODO: better way than passing this? (relying on all gradients in a vector can lead to large memory consumption)
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<T_mat> Sigma,
			const std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode) {
			if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * Sigma * Wsqrt) at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLStable(y_data, y_data_int, fixed_effects, Sigma, mll);
			}
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			// Initialize variables
			vec_t location_par;//location parameter = mode of random effects + fixed effects
			double* location_par_ptr;
			InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
			T_mat L_inv_Wsqrt(dim_mode_, dim_mode_);//L_inv_Wsqrt = L\ZtWZsqrt if use_Z_for_duplicates_ or L\Wsqrt if !use_Z_for_duplicates_ where L is a Cholesky factor of Id_plus_Wsqrt_Sigma_Wsqrt
			vec_t third_deriv(dim_mode_);//vector of third derivatives of log-likelihood
			L_inv_Wsqrt.setIdentity();
			L_inv_Wsqrt.diagonal().array() = second_deriv_neg_ll_.array().sqrt();
			vec_t third_deriv_data_scale;//third derivatives on the data-scale (only used if use_Z_for_duplicates_), the vector 'third_deriv' actually contains diag_ZtThirdDerivZ if use_Z_for_duplicates_
			if (use_Z_for_duplicates_) {
				third_deriv_data_scale = vec_t(num_data_);//third derivatives on the data-scale, the vector 'third_deriv' actually contains diag_ZtThirdDerivZ if use_Z_for_duplicates_
				CalcThirdDerivLogLik(y_data, y_data_int, location_par_ptr, third_deriv_data_scale);
				CalcZtVGivenIndices(num_data_, num_re_, random_effects_indices_of_data_, third_deriv_data_scale, third_deriv, true);
			}
			else {
				//L_inv_Wsqrt.diagonal().array() = second_deriv_neg_ll_.array().sqrt();
				CalcThirdDerivLogLik(y_data, y_data_int, location_par_ptr, third_deriv);
			}
			TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, L_inv_Wsqrt, L_inv_Wsqrt, false);//L_inv_Wsqrt = L\Wsqrt
			T_mat L_inv_Wsqrt_Sigma = L_inv_Wsqrt * (*Sigma);
			//Log::REInfo("CalcGradNegMargLikelihoodLaplaceApproxStable: L_inv_ZtWZsqrt: number non zeros = %d", GetNumberNonZeros<T_mat>(L_inv_ZtWZsqrt));//Only for debugging
			//Log::REInfo("CalcGradNegMargLikelihoodLaplaceApproxStable: L_inv_ZtWZsqrt_Sigma: number non zeros = %d", GetNumberNonZeros<T_mat>(L_inv_ZtWZsqrt_Sigma));//Only for debugging
			// Calculate gradient of approx. marginal log-likelihood wrt the mode
			//		Note: use (i) (Sigma^-1 + W)^-1 = Sigma - Sigma*(W^-1 + Sigma)^-1*Sigma = Sigma - L_inv_Wsqrt_Sigma^T * L_inv_Wsqrt_Sigma and (ii) "Z=Id"
			T_mat L_inv_Wsqrt_Sigma_sqr = L_inv_Wsqrt_Sigma.cwiseProduct(L_inv_Wsqrt_Sigma);
			vec_t SigmaI_plus_W_inv_diag = (*Sigma).diagonal() - L_inv_Wsqrt_Sigma_sqr.transpose() * vec_t::Ones(L_inv_Wsqrt_Sigma_sqr.rows());// diagonal of (Sigma^-1 + ZtWZ) ^ -1 if use_Z_for_duplicates_ or of (ZSigmaZt^-1 + W)^-1 if !use_Z_for_duplicates_
			vec_t d_mll_d_mode = (-0.5 * SigmaI_plus_W_inv_diag.array() * third_deriv.array()).matrix();// gradient of approx. marginal likelihood wrt the mode
			// Calculate gradient wrt covariance parameters
			if (calc_cov_grad) {
				T_mat WI_plus_Sigma_inv;//WI_plus_Sigma_inv = ZtWZsqrt * L^T\(L\ZtWZsqrt) = ((ZtWZ)^-1 + Sigma)^-1 if use_Z_for_duplicates_ or Wsqrt * L^T\(L\Wsqrt) = (W^-1 + ZSigmaZt)^-1 if !use_Z_for_duplicates_
				vec_t d_mode_d_par, SigmaDeriv_first_deriv_ll; //auxiliary variable for caclulating d_mode_d_par
				int par_count = 0;
				double explicit_derivative;
				for (int j = 0; j < (int)re_comps_cluster_i.size(); ++j) {
					for (int ipar = 0; ipar < re_comps_cluster_i[j]->NumCovPar(); ++ipar) {
						std::shared_ptr<T_mat> SigmaDeriv = re_comps_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 1.);
						if (ipar == 0) {
							WI_plus_Sigma_inv = *SigmaDeriv;
							CalcLtLGivenSparsityPattern<T_mat>(L_inv_Wsqrt, WI_plus_Sigma_inv, true);
							//TODO (low-prio): calculate WI_plus_Sigma_inv only once for all relevant non-zero entries as in Gaussian case (see 'CalcPsiInv')
							//					This is only relevant for multiple random effects and/or GPs
						}
						// Calculate explicit derivative of approx. mariginal log-likelihood
						explicit_derivative = -0.5 * (double)(a_vec_.transpose() * (*SigmaDeriv) * a_vec_) +
							0.5 * (WI_plus_Sigma_inv.cwiseProduct(*SigmaDeriv)).sum();
						// Calculate implicit derivative (through mode) of approx. mariginal log-likelihood
						SigmaDeriv_first_deriv_ll = (*SigmaDeriv) * first_deriv_ll_;
						d_mode_d_par = SigmaDeriv_first_deriv_ll;//derivative of mode wrt to a covariance parameter
						d_mode_d_par -= ((*Sigma) * (L_inv_Wsqrt.transpose() * (L_inv_Wsqrt * SigmaDeriv_first_deriv_ll)));
						cov_grad[par_count] = explicit_derivative + d_mll_d_mode.dot(d_mode_d_par);
						par_count++;
					}
				}
			}//end calc_cov_grad
			// calculate gradient wrt fixed effects
			vec_t SigmaI_plus_W_inv_d_mll_d_mode;// for implicit derivative
			if (calc_F_grad || calc_aux_par_grad) {
				vec_t L_inv_Wsqrt_Sigma_d_mll_d_mode = L_inv_Wsqrt_Sigma * d_mll_d_mode;// for implicit derivative
				SigmaI_plus_W_inv_d_mll_d_mode = (*Sigma) * d_mll_d_mode - L_inv_Wsqrt_Sigma.transpose() * L_inv_Wsqrt_Sigma_d_mll_d_mode;
			}
			if (calc_F_grad) {
				if (use_Z_for_duplicates_) {
					fixed_effect_grad = -first_deriv_ll_data_scale_;
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						fixed_effect_grad[i] += -0.5 * third_deriv_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
							second_deriv_neg_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
					}
				}
				else {
					vec_t d_mll_d_F_implicit = (SigmaI_plus_W_inv_d_mll_d_mode.array() * second_deriv_neg_ll_.array()).matrix();// implicit derivative
					fixed_effect_grad = -first_deriv_ll_ + d_mll_d_mode - d_mll_d_F_implicit;
				}
			}//end calc_F_grad
			// calculate gradient wrt additional likelihood parameters
			if (calc_aux_par_grad) {
				vec_t neg_likelihood_deriv(num_aux_pars_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
				vec_t second_deriv(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
				vec_t neg_third_deriv(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
				vec_t d_mode_d_aux_par;
				CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par_ptr, num_data_, neg_likelihood_deriv.data());
				for (int ind_ap = 0; ind_ap < num_aux_pars_; ++ind_ap) {
					CalcSecondNegThirdDerivLogLikAuxParsLocPar(y_data, y_data_int, location_par_ptr, num_data_, ind_ap, second_deriv.data(), neg_third_deriv.data());
					double d_detmll_d_aux_par = 0., implicit_derivative = 0.;
					if (use_Z_for_duplicates_) {
#pragma omp parallel for schedule(static) reduction(+:d_detmll_d_aux_par, implicit_derivative)
						for (data_size_t i = 0; i < num_data_; ++i) {
							d_detmll_d_aux_par += neg_third_deriv[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]];
							implicit_derivative += second_deriv[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];
						}
					}
					else {
#pragma omp parallel for schedule(static) reduction(+:d_detmll_d_aux_par, implicit_derivative)
						for (data_size_t i = 0; i < num_data_; ++i) {
							d_detmll_d_aux_par += neg_third_deriv[i] * SigmaI_plus_W_inv_diag[i];
							implicit_derivative += second_deriv[i] * SigmaI_plus_W_inv_d_mll_d_mode[i];
						}
					}
					aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
				}
			}//end calc_aux_par_grad
		}//end CalcGradNegMargLikelihoodLaplaceApproxStable

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*		fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*		This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param SigmaI Inverse covariance matrix of latent random effect. Currently, this needs to be a diagonal matrix
		* \param Zt Transpose Z^T of random effect design matrix that relates latent random effects to observations/likelihoods
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxGroupedRE(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const sp_mat_t& SigmaI,
			const sp_mat_t& Zt,
			std::vector<data_size_t> cum_num_rand_eff_cluster_i,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode) {
			int num_REs = (int)SigmaI.cols();//number of random effect realizations
			int num_comps = (int)cum_num_rand_eff_cluster_i.size() - 1;//number of different random effect components
			if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLGroupedRE(y_data, y_data_int, fixed_effects, num_data, SigmaI, Zt, mll);
			}
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			// Initialize variables
			sp_mat_t Z = Zt.transpose();
			vec_t location_par = Z * mode_;//location parameter = mode of random effects + fixed effects
			if (fixed_effects != nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] += fixed_effects[i];
				}
			}
			vec_t third_deriv(num_data);//vector of third derivatives of log-likelihood
			CalcThirdDerivLogLik(y_data, y_data_int, location_par.data(), third_deriv);
			// Calculate (Sigma^-1 + Zt*W*Z)^-1
			sp_mat_t L_inv(num_REs, num_REs);
			L_inv.setIdentity();
			if (chol_fact_SigmaI_plus_ZtWZ_grouped_.permutationP().size() > 0) {//Permutation is only used when having an ordering
				L_inv = chol_fact_SigmaI_plus_ZtWZ_grouped_.permutationP() * L_inv;
			}
			sp_mat_t L = chol_fact_SigmaI_plus_ZtWZ_grouped_.matrixL();
			TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(L, L_inv, L_inv, false);
			L.resize(0, 0);
			sp_mat_t SigmaI_plus_ZtWZ_inv;
			// calculate gradient of approx. marginal likelihood wrt the mode
			vec_t d_mll_d_mode(num_REs);
			sp_mat_t Zt_third_deriv = Zt * third_deriv.asDiagonal();//every column of Z multiplied elementwise by third_deriv
#pragma omp parallel for schedule(static)
			for (int ire = 0; ire < num_REs; ++ire) {
				//calculate Z^T * diag(diag_d_W_d_mode_i) * Z = Z^T * diag(Z.col(i) * third_deriv) * Z
				d_mll_d_mode[ire] = 0.;
				double entry_ij;
				for (data_size_t i = 0; i < num_data; ++i) {
					entry_ij = Zt_third_deriv.coeff(ire, i);
					if (std::abs(entry_ij) > EPSILON_NUMBERS) {
						vec_t L_inv_Zt_col_i = L_inv * Zt.col(i);
						d_mll_d_mode[ire] += entry_ij * (L_inv_Zt_col_i.squaredNorm());
					}
				}
				d_mll_d_mode[ire] *= -0.5;
			}
			// calculate gradient wrt covariance parameters
			if (calc_cov_grad) {
				sp_mat_t ZtWZ = Zt * second_deriv_neg_ll_.asDiagonal() * Z;
				vec_t d_mode_d_par;//derivative of mode wrt to a covariance parameter
				vec_t v_aux;//auxiliary variable for caclulating d_mode_d_par
				vec_t SigmaI_mode = SigmaI * mode_;
				double explicit_derivative;
				sp_mat_t I_j(num_REs, num_REs);//Diagonal matrix with 1 on the diagonal for all random effects of component j and 0's otherwise
				sp_mat_t I_j_ZtWZ;
				for (int j = 0; j < num_comps; ++j) {
					// calculate explicit derivative of approx. mariginal log-likelihood
					std::vector<Triplet_t> triplets(cum_num_rand_eff_cluster_i[j + 1] - cum_num_rand_eff_cluster_i[j]);//for constructing I_j
					explicit_derivative = 0.;
#pragma omp parallel for schedule(static) reduction(+:explicit_derivative)
					for (int i = cum_num_rand_eff_cluster_i[j]; i < cum_num_rand_eff_cluster_i[j + 1]; ++i) {
						triplets[i - cum_num_rand_eff_cluster_i[j]] = Triplet_t(i, i, 1.);
						explicit_derivative += SigmaI_mode[i] * mode_[i];
					}
					explicit_derivative *= -0.5;
					I_j.setFromTriplets(triplets.begin(), triplets.end());
					I_j_ZtWZ = I_j * ZtWZ;
					SigmaI_plus_ZtWZ_inv = I_j_ZtWZ;
					CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_ZtWZ_inv, false);
					explicit_derivative += 0.5 * (SigmaI_plus_ZtWZ_inv.cwiseProduct(I_j_ZtWZ)).sum();
					// calculate implicit derivative (through mode) of approx. mariginal log-likelihood
					d_mode_d_par = L_inv.transpose() * (L_inv * (I_j * (Zt * first_deriv_ll_)));
					cov_grad[j] = explicit_derivative + d_mll_d_mode.dot(d_mode_d_par);
				}
			}//end calc_cov_grad
			// calculate gradient wrt fixed effects
			if (calc_F_grad) {
				vec_t d_detmll_d_F(num_data);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data; ++i) {
					vec_t L_inv_Zt_col_i = L_inv * Zt.col(i);
					d_detmll_d_F[i] = -0.5 * third_deriv[i] * (L_inv_Zt_col_i.squaredNorm());

				}
				vec_t d_mll_d_modeT_SigmaI_plus_ZtWZ_inv_Zt_W = (((d_mll_d_mode.transpose() * L_inv.transpose()) * L_inv) * Zt) * second_deriv_neg_ll_.asDiagonal();
				fixed_effect_grad = -first_deriv_ll_ + d_detmll_d_F - d_mll_d_modeT_SigmaI_plus_ZtWZ_inv_Zt_W;
			}//end calc_F_grad
			// calculate gradient wrt additional likelihood parameters
			if (calc_aux_par_grad) {
				vec_t neg_likelihood_deriv(num_aux_pars_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
				vec_t second_deriv(num_data);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
				vec_t neg_third_deriv(num_data);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
				vec_t d_mode_d_aux_par;
				CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par.data(), num_data, neg_likelihood_deriv.data());
				for (int ind_ap = 0; ind_ap < num_aux_pars_; ++ind_ap) {
					CalcSecondNegThirdDerivLogLikAuxParsLocPar(y_data, y_data_int, location_par.data(), num_data, ind_ap, second_deriv.data(), neg_third_deriv.data());
					sp_mat_t ZtdWZ = Zt * neg_third_deriv.asDiagonal() * Z;
					SigmaI_plus_ZtWZ_inv = ZtdWZ;
					CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_ZtWZ_inv, false);
					double d_detmll_d_aux_par = (SigmaI_plus_ZtWZ_inv.cwiseProduct(ZtdWZ)).sum();
					d_mode_d_aux_par = L_inv.transpose() * (L_inv * (Zt * second_deriv));
					double implicit_derivative = d_mll_d_mode.dot(d_mode_d_aux_par);
					aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
				}
			}//end calc_aux_par_grad
		}//end CalcGradNegMargLikelihoodLaplaceApproxGroupedRE

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*		fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param sigma2 Variance of random effects
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const double sigma2,
			const data_size_t* const random_effects_indices_of_data,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode) {
			if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(y_data, y_data_int, fixed_effects, num_data,
					sigma2, random_effects_indices_of_data, mll);
			}
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			// Initialize variables
			vec_t location_par(num_data);//location parameter = mode of random effects + fixed effects
			if (fixed_effects == nullptr) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] = mode_[random_effects_indices_of_data[i]];
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] = mode_[random_effects_indices_of_data[i]] + fixed_effects[i];
				}
			}
			vec_t third_deriv(num_data);//vector of third derivatives of log-likelihood
			CalcThirdDerivLogLik(y_data, y_data_int, location_par.data(), third_deriv);
			// calculate gradient of approx. marginal likelihood wrt the mode
			vec_t d_mll_d_mode;
			CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, third_deriv, d_mll_d_mode, true);
			d_mll_d_mode.array() /= -2. * diag_SigmaI_plus_ZtWZ_.array();
			// calculate gradient wrt covariance parameters
			if (calc_cov_grad) {
				vec_t diag_ZtWZ;
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv_neg_ll_, diag_ZtWZ, true);
				double explicit_derivative = -0.5 * (mode_.array() * mode_.array()).sum() / sigma2 +
					0.5 * (diag_ZtWZ.array() / diag_SigmaI_plus_ZtWZ_.array()).sum();
				// calculate implicit derivative (through mode) of approx. mariginal log-likelihood
				vec_t d_mode_d_par;
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, first_deriv_ll_, d_mode_d_par, true);
				d_mode_d_par.array() /= diag_SigmaI_plus_ZtWZ_.array();
				cov_grad[0] = explicit_derivative + d_mll_d_mode.dot(d_mode_d_par);
			}//end calc_cov_grad
			// calculate gradient wrt fixed effects
			if (calc_F_grad) {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data; ++i) {
					fixed_effect_grad[i] = -first_deriv_ll_[i] -
						0.5 * third_deriv[i] / diag_SigmaI_plus_ZtWZ_[random_effects_indices_of_data[i]] - //=d_detmll_d_F
						d_mll_d_mode[random_effects_indices_of_data[i]] * second_deriv_neg_ll_[i] / diag_SigmaI_plus_ZtWZ_[random_effects_indices_of_data[i]];//=implicit derivative = d_mll_d_mode * d_mode_d_F
				}
			}//end calc_F_grad
			// calculate gradient wrt additional likelihood parameters
			if (calc_aux_par_grad) {
				vec_t neg_likelihood_deriv(num_aux_pars_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
				vec_t second_deriv(num_data);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
				vec_t neg_third_deriv(num_data);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
				CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par.data(), num_data, neg_likelihood_deriv.data());
				for (int ind_ap = 0; ind_ap < num_aux_pars_; ++ind_ap) {
					CalcSecondNegThirdDerivLogLikAuxParsLocPar(y_data, y_data_int, location_par.data(), num_data, ind_ap, second_deriv.data(), neg_third_deriv.data());
					double d_detmll_d_aux_par = 0.;
					double implicit_derivative = 0.;// = implicit derivative = d_mll_d_mode * d_mode_d_aux_par
#pragma omp parallel for schedule(static) reduction(+:d_detmll_d_aux_par, implicit_derivative)
					for (int i = 0; i < num_data; ++i) {
						d_detmll_d_aux_par += neg_third_deriv[i] / diag_SigmaI_plus_ZtWZ_[random_effects_indices_of_data[i]];
						implicit_derivative += d_mll_d_mode[random_effects_indices_of_data[i]] * second_deriv[i] / diag_SigmaI_plus_ZtWZ_[random_effects_indices_of_data[i]];
					}
					aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
					//Equivalent code:
					//vec_t Zt_second_deriv, diag_Zt_neg_third_deriv_Z;
					//CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv, Zt_second_deriv, true);
					//CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, neg_third_deriv, diag_Zt_neg_third_deriv_Z, true);
					//double d_detmll_d_aux_par = (diag_Zt_neg_third_deriv_Z.array() / diag_SigmaI_plus_ZtWZ_.array()).sum();
					//double implicit_derivative = (d_mll_d_mode.array() * Zt_second_deriv.array() / diag_SigmaI_plus_ZtWZ_.array()).sum();
				}
			}//end calc_aux_par_grad
		}//end CalcGradNegMargLikelihoodLaplaceApproxOnlyOneGroupedRECalculationsOnREScale

		/*!
		* \brief Calculate the gradient of the negative Laplace-approximated marginal log-likelihood wrt covariance parameters,
		*		fixed effects (e.g., for linear regression coefficients), and additional likelihood-related parameters.
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param B Matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation Sigma^-1 = B^T D^-1 B
		* \param B_grad Derivatives of matrices B ( = derivative of matrix -A) for Vecchia approximation
		* \param D_grad Derivatives of matrices D for Vecchia approximation
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters is calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F is calculated
		* \param calc_aux_par_grad If true, the gradient wrt additional likelihood parameters is calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param[out] aux_par_grad Gradient wrt additional likelihood parameters
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		* \param num_comps_total Total number of random effect components ( = number of GPs)
		*/
		void CalcGradNegMargLikelihoodLaplaceApproxVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const std::vector<sp_mat_t>& B_grad,
			const std::vector<sp_mat_t>& D_grad,
			bool calc_cov_grad,
			bool calc_F_grad,
			bool calc_aux_par_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			double* aux_par_grad,
			bool calc_mode,
			int num_comps_total) {
			if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLVecchia(y_data, y_data_int, fixed_effects, B, D_inv, false, Sigma_L_k_, true, mll);
			}
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			// Initialize variables
			vec_t location_par;//location parameter = mode of random effects + fixed effects
			double* location_par_ptr;
			InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
			vec_t third_deriv(dim_mode_);//vector of third derivatives of log-likelihood
			vec_t third_deriv_data_scale;//third derivatives on the data-scale (only used if use_Z_for_duplicates_), the vector 'third_deriv' actually contains diag_ZtThirdDerivZ if use_Z_for_duplicates_
			if (use_Z_for_duplicates_) {
				third_deriv_data_scale = vec_t(num_data_);//third derivatives on the data-scale, the vector 'third_deriv' actually contains diag_ZtThirdDerivZ if use_Z_for_duplicates_
				CalcThirdDerivLogLik(y_data, y_data_int, location_par_ptr, third_deriv_data_scale);
				CalcZtVGivenIndices(num_data_, num_re_, random_effects_indices_of_data_, third_deriv_data_scale, third_deriv, true);
			}
			else {
				//L_inv_Wsqrt.diagonal().array() = second_deriv_neg_ll_.array().sqrt();
				CalcThirdDerivLogLik(y_data, y_data_int, location_par_ptr, third_deriv);
			}
			if (matrix_inversion_method_ == "iterative") {
				vec_t d_mll_d_mode, d_log_det_Sigma_W_plus_I_d_mode, SigmaI_plus_W_inv_d_mll_d_mode(dim_mode_);
				//Declarations for preconditioner "piv_chol_on_Sigma"
				vec_t diag_WI;
				den_mat_t WI_PI_Z, WI_WI_plus_Sigma_inv_Z;
				//Declarations for preconditioner "Sigma_inv_plus_BtWB"
				vec_t D_inv_plus_W_inv_diag;
				den_mat_t PI_Z;
				//Stochastic Trace: Calculate gradient of approx. marginal likelihood wrt. the mode (and thus also F here if !use_Z_for_duplicates_)
				CalcLogDetStochDerivMode(third_deriv, dim_mode_, d_log_det_Sigma_W_plus_I_d_mode, D_inv_plus_W_inv_diag, diag_WI, PI_Z, WI_PI_Z, WI_WI_plus_Sigma_inv_Z);
				d_mll_d_mode = 0.5 * d_log_det_Sigma_W_plus_I_d_mode;
				//For implicit derivatives: calculate (Sigma^(-1) + W)^(-1) d_mll_d_mode
				bool has_NA_or_Inf = false;
				if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
					CGVecchiaLaplaceVecWinvplusSigma(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_.transpose(), d_mll_d_mode, SigmaI_plus_W_inv_d_mll_d_mode, has_NA_or_Inf,
						cg_max_num_it_, 0, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_);
				}
				else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB") {
					CGVecchiaLaplaceVec(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_, d_mll_d_mode, SigmaI_plus_W_inv_d_mll_d_mode, has_NA_or_Inf,
						cg_max_num_it_, 0, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, D_inv_plus_W_B_rm_);
				}
				else {
					Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
				}
				if (has_NA_or_Inf) {
					Log::REDebug(CG_NA_OR_INF_WARNING_);
				}
				// Calculate gradient wrt covariance parameters
				if (calc_cov_grad) {
					sp_mat_rm_t SigmaI_deriv_rm, Bt_Dinv_Bgrad_rm, B_t_D_inv_D_grad_D_inv_B_rm;
					vec_t SigmaI_deriv_mode;
					double explicit_derivative, d_log_det_Sigma_W_plus_I_d_cov_pars;
					int num_par = (int)B_grad.size();
					for (int j = 0; j < num_par; ++j) {
						// Calculate SigmaI_deriv
						if (num_comps_total == 1 && j == 0) {
							SigmaI_deriv_rm = -B_rm_.transpose() * B_t_D_inv_rm_.transpose();//SigmaI_deriv = -SigmaI for variance parameters if there is only one GP
						}
						else {
							SigmaI_deriv_rm = sp_mat_rm_t(B_grad[j].transpose()) * B_t_D_inv_rm_.transpose();
							Bt_Dinv_Bgrad_rm = SigmaI_deriv_rm.transpose();
							B_t_D_inv_D_grad_D_inv_B_rm = B_t_D_inv_rm_ * sp_mat_rm_t(D_grad[j]) * B_t_D_inv_rm_.transpose();
							SigmaI_deriv_rm += Bt_Dinv_Bgrad_rm - B_t_D_inv_D_grad_D_inv_B_rm;
							Bt_Dinv_Bgrad_rm.resize(0, 0);
						}
						SigmaI_deriv_mode = SigmaI_deriv_rm * mode_;
						CalcLogDetStochDerivCovPar(dim_mode_, num_comps_total, j, SigmaI_deriv_rm, B_grad[j], D_grad[j], D_inv_plus_W_inv_diag, PI_Z, WI_PI_Z, d_log_det_Sigma_W_plus_I_d_cov_pars);
						explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv_mode) + d_log_det_Sigma_W_plus_I_d_cov_pars);
						cov_grad[j] = explicit_derivative - SigmaI_plus_W_inv_d_mll_d_mode.dot(SigmaI_deriv_mode); //add implicit derivative
					}
				}
				//Calculate gradient wrt fixed effects
				vec_t SigmaI_plus_W_inv_diag;
				if (use_Z_for_duplicates_ && (calc_F_grad || calc_aux_par_grad)) {
//					//Stochastic Trace: Calculate diagonal of SigmaI_plus_W_inv for gradient of approx. marginal likelihood wrt. F
					SigmaI_plus_W_inv_diag = d_log_det_Sigma_W_plus_I_d_mode;
					SigmaI_plus_W_inv_diag.array() *= -1. / third_deriv.array();
				}
				if (calc_F_grad) {
					if (use_Z_for_duplicates_) {
						fixed_effect_grad = -first_deriv_ll_data_scale_;
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							fixed_effect_grad[i] += -0.5 * third_deriv_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
								second_deriv_neg_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
						}
					}
					else {
						vec_t d_mll_d_F_implicit = -(SigmaI_plus_W_inv_d_mll_d_mode.array() * second_deriv_neg_ll_.array()).matrix();// implicit derivative
						fixed_effect_grad = -first_deriv_ll_ + d_mll_d_mode + d_mll_d_F_implicit;
					}
				}
				//Calculate gradient wrt additional likelihood parameters
				if (calc_aux_par_grad) {
					vec_t neg_likelihood_deriv(num_aux_pars_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
					vec_t second_deriv(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
					vec_t neg_third_deriv(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
					vec_t d_mode_d_aux_par;
					CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par_ptr, num_data_, neg_likelihood_deriv.data());
					for (int ind_ap = 0; ind_ap < num_aux_pars_; ++ind_ap) {
						CalcSecondNegThirdDerivLogLikAuxParsLocPar(y_data, y_data_int, location_par_ptr, num_data_, ind_ap, second_deriv.data(), neg_third_deriv.data());
						double d_detmll_d_aux_par = 0., implicit_derivative = 0.;
						if (use_Z_for_duplicates_) {
#pragma omp parallel for schedule(static) reduction(+:d_detmll_d_aux_par, implicit_derivative)
							for (data_size_t i = 0; i < num_data_; ++i) {
								d_detmll_d_aux_par += neg_third_deriv[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]];
								implicit_derivative += second_deriv[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];
							}
						}
						else {
							CalcLogDetStochDerivAuxPar(neg_third_deriv, D_inv_plus_W_inv_diag, diag_WI, PI_Z, WI_PI_Z, WI_WI_plus_Sigma_inv_Z, d_detmll_d_aux_par);
#pragma omp parallel for schedule(static) reduction(+:d_detmll_d_aux_par, implicit_derivative)
							for (data_size_t i = 0; i < num_data_; ++i) {
								implicit_derivative += second_deriv[i] * SigmaI_plus_W_inv_d_mll_d_mode[i];
							}
						}
						aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
					}
				}//end calc_aux_par_grad
			}//end iterative
			else {//Cholesky decomposition
				// Calculate (Sigma^-1 + W)^-1
				sp_mat_t L_inv(dim_mode_, dim_mode_);
				L_inv.setIdentity();
				TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, L_inv, L_inv, false);
				vec_t d_mll_d_mode, SigmaI_plus_W_inv_d_mll_d_mode, SigmaI_plus_W_inv_diag;
				sp_mat_t SigmaI_plus_W_inv;
				// Calculate gradient wrt covariance parameters
				if (calc_cov_grad) {
					double explicit_derivative;
					int num_par = (int)B_grad.size();
					sp_mat_t SigmaI_deriv, BgradT_Dinv_B, Bt_Dinv_Bgrad;
					sp_mat_t D_inv_B = D_inv * B;
					for (int j = 0; j < num_par; ++j) {
						// Calculate SigmaI_deriv
						if (num_comps_total == 1 && j == 0) {
							SigmaI_deriv = -B.transpose() * D_inv_B;//SigmaI_deriv = -SigmaI for variance parameters if there is only one GP
						}
						else {
							SigmaI_deriv = B_grad[j].transpose() * D_inv_B;
							Bt_Dinv_Bgrad = SigmaI_deriv.transpose();
							SigmaI_deriv += Bt_Dinv_Bgrad - D_inv_B.transpose() * D_grad[j] * D_inv_B;
							Bt_Dinv_Bgrad.resize(0, 0);
						}
						if (j == 0) {
							// Calculate SigmaI_plus_W_inv = L_inv.transpose() * L_inv at non-zero entries of SigmaI_deriv
							//	Note: fully calculating SigmaI_plus_W_inv = L_inv.transpose() * L_inv is very slow
							SigmaI_plus_W_inv = SigmaI_deriv;
							CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_W_inv, true);
							d_mll_d_mode = -0.5 * (SigmaI_plus_W_inv.diagonal().array() * third_deriv.array()).matrix();
						}//end if j == 0
						SigmaI_plus_W_inv_d_mll_d_mode = L_inv.transpose() * (L_inv * d_mll_d_mode);
						vec_t SigmaI_deriv_mode = SigmaI_deriv * mode_;
						explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv_mode) + (SigmaI_deriv.cwiseProduct(SigmaI_plus_W_inv)).sum());
						if (num_comps_total == 1 && j == 0) {
							explicit_derivative += 0.5 * dim_mode_;
						}
						else {
							explicit_derivative += 0.5 * (D_inv.diagonal().array() * D_grad[j].diagonal().array()).sum();
						}
						cov_grad[j] = explicit_derivative - SigmaI_plus_W_inv_d_mll_d_mode.dot(SigmaI_deriv_mode);//add implicit derivative
					}
				}//end calc_cov_grad
				if (calc_F_grad || calc_aux_par_grad) {
					if (!calc_cov_grad) {
						sp_mat_t L_inv_sqr = L_inv.cwiseProduct(L_inv);
						SigmaI_plus_W_inv_diag = L_inv_sqr.transpose() * vec_t::Ones(L_inv_sqr.rows());// diagonal of (Sigma^-1 + W) ^ -1
						d_mll_d_mode = (-0.5 * SigmaI_plus_W_inv_diag.array() * third_deriv.array()).matrix();// gradient of approx. marginal likelihood wrt the mode and thus also F here
						SigmaI_plus_W_inv_d_mll_d_mode = L_inv.transpose() * (L_inv * d_mll_d_mode);
					}
					else if (calc_aux_par_grad || use_Z_for_duplicates_) {
						SigmaI_plus_W_inv_diag = SigmaI_plus_W_inv.diagonal();
					}
				}
				// Calculate gradient wrt fixed effects
				if (calc_F_grad) {
					if (use_Z_for_duplicates_) {
						fixed_effect_grad = -first_deriv_ll_data_scale_;
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							fixed_effect_grad[i] += -0.5 * third_deriv_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
								second_deriv_neg_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
						}
					}
					else {
						vec_t d_mll_d_F_implicit = -(SigmaI_plus_W_inv_d_mll_d_mode.array() * second_deriv_neg_ll_.array()).matrix();// implicit derivative
						fixed_effect_grad = -first_deriv_ll_ + d_mll_d_mode + d_mll_d_F_implicit;
					}
				}//end calc_F_grad
				// calculate gradient wrt additional likelihood parameters
				if (calc_aux_par_grad) {
					vec_t neg_likelihood_deriv(num_aux_pars_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
					vec_t second_deriv(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
					vec_t neg_third_deriv(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
					vec_t d_mode_d_aux_par;
					CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par_ptr, num_data_, neg_likelihood_deriv.data());
					for (int ind_ap = 0; ind_ap < num_aux_pars_; ++ind_ap) {
						CalcSecondNegThirdDerivLogLikAuxParsLocPar(y_data, y_data_int, location_par_ptr, num_data_, ind_ap, second_deriv.data(), neg_third_deriv.data());
						double d_detmll_d_aux_par = 0., implicit_derivative = 0.;
						if (use_Z_for_duplicates_) {
#pragma omp parallel for schedule(static) reduction(+:d_detmll_d_aux_par, implicit_derivative)
							for (data_size_t i = 0; i < num_data_; ++i) {
								d_detmll_d_aux_par += neg_third_deriv[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]];
								implicit_derivative += second_deriv[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];
							}
						}
						else {
#pragma omp parallel for schedule(static) reduction(+:d_detmll_d_aux_par, implicit_derivative)
							for (data_size_t i = 0; i < num_data_; ++i) {
								d_detmll_d_aux_par += neg_third_deriv[i] * SigmaI_plus_W_inv_diag[i];
								implicit_derivative += second_deriv[i] * SigmaI_plus_W_inv_d_mll_d_mode[i];
							}
						}
						aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
					}
				}//end calc_aux_par_grad
			}//end Cholesky decomposition
		}//end CalcGradNegMargLikelihoodLaplaceApproxVecchia

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param ZSigmaZt Covariance matrix of latent random effect
		* \param Cross_Cov Cross covariance matrix between predicted and observed random effects ("=Cov(y_p,y)")
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void PredictLaplaceApproxStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const std::shared_ptr<T_mat> ZSigmaZt,
			const T_mat& Cross_Cov,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode) {
			if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * ZSigmaZt * Wsqrt) at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLStable(y_data, y_data_int, fixed_effects, ZSigmaZt, mll);
			}
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			pred_mean = Cross_Cov * first_deriv_ll_;
			if (calc_pred_cov || calc_pred_var) {
				vec_t Wsqrt(dim_mode_);//diagonal of matrix sqrt(ZtWZ) if use_Z_for_duplicates_ or sqrt(W) if !use_Z_for_duplicates_
				Wsqrt.array() = second_deriv_neg_ll_.array().sqrt();
				T_mat Maux = Wsqrt.asDiagonal() * Cross_Cov.transpose();
				TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Maux, Maux, false);//Maux = L\(ZtWZsqrt * Cross_Cov^T)
				if (calc_pred_cov) {
					pred_cov -= (T_mat)(Maux.transpose() * Maux);
				}
				if (calc_pred_var) {
					Maux = Maux.cwiseProduct(Maux);
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] -= Maux.col(i).sum();
					}
				}
			}
		}//end PredictLaplaceApproxStable

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*		This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param SigmaI Inverse covariance matrix of latent random effect. Currently, this needs to be a diagonal matrix
		* \param Zt Transpose Z^T of random effect design matrix that relates latent random effects to observations/likelihoods
		* \param Ztilde matrix which relates existing random effects to prediction samples
		* \param Sigma Covariance matrix of random effects
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void PredictLaplaceApproxGroupedRE(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const sp_mat_t& SigmaI,
			const sp_mat_t& Zt,
			const sp_mat_t& Ztilde,
			const sp_mat_t& Sigma,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode) {
			if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * ZSigmaZt * Wsqrt) at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLGroupedRE(y_data, y_data_int, fixed_effects, num_data, SigmaI, Zt, mll);
			}
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			pred_mean = Ztilde * (Sigma * (Zt * first_deriv_ll_));
			if (calc_pred_cov || calc_pred_var) {
				sp_mat_t SigmaI_plus_ZtWZ_I(Sigma.cols(), Sigma.cols());
				SigmaI_plus_ZtWZ_I.setIdentity();
				TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, SigmaI_plus_ZtWZ_I, SigmaI_plus_ZtWZ_I, false);
				TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, SigmaI_plus_ZtWZ_I, SigmaI_plus_ZtWZ_I, true);
				sp_mat_t Sigma_Zt_W_Z_SigmaI_plus_ZtWZ_I = (Sigma * (Zt * second_deriv_neg_ll_.asDiagonal() * Zt.transpose())) * SigmaI_plus_ZtWZ_I;
				if (calc_pred_cov) {
					pred_cov -= (T_mat)(Ztilde * Sigma_Zt_W_Z_SigmaI_plus_ZtWZ_I * Ztilde.transpose());
				}
				if (calc_pred_var) {
					sp_mat_t Maux = Ztilde;
					CalcAtimesBGivenSparsityPattern<sp_mat_t>(Ztilde, Sigma_Zt_W_Z_SigmaI_plus_ZtWZ_I, Maux);
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] -= (Ztilde.row(i)).dot(Maux.row(i));
					}
				}
				//Old code (not used anymore)
//				// calculate Maux = L\(Z^T * second_deriv_neg_ll_.asDiagonal() * Cross_Cov^T)
//				sp_mat_t Cross_Cov = Ztilde * Sigma * Zt;
//				sp_mat_t Maux = Zt * second_deriv_neg_ll_.asDiagonal() * Cross_Cov.transpose();
//				TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, Maux, Maux, false);
//				if (calc_pred_cov) {
//					pred_cov += (T_mat)(Maux.transpose() * Maux);
//					pred_cov -= (T_mat)(Cross_Cov * second_deriv_neg_ll_.asDiagonal() * Cross_Cov.transpose());
//				}
//				if (calc_pred_var) {
//					sp_mat_t Maux3 = Cross_Cov.cwiseProduct(Cross_Cov * second_deriv_neg_ll_.asDiagonal());
//					Maux = Maux.cwiseProduct(Maux);
//#pragma omp parallel for schedule(static)
//					for (int i = 0; i < (int)pred_mean.size(); ++i) {
//						pred_var[i] += Maux.col(i).sum() - Maux3.row(i).sum();
//					}
//				}
			}
		}//end PredictLaplaceApproxGroupedRE

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param sigma2 Variance of random effects
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param Cross_Cov Cross covariance matrix between predicted and observed random effects ("=Cov(y_p,y)")
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void PredictLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const double sigma2,
			const data_size_t* const random_effects_indices_of_data,
			const T_mat& Cross_Cov,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode) {
			if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * ZSigmaZt * Wsqrt) at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(y_data, y_data_int, fixed_effects, num_data,
					sigma2, random_effects_indices_of_data, mll);
			}
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			vec_t ZtFirstDeriv;
			CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, first_deriv_ll_, ZtFirstDeriv, true);
			pred_mean = Cross_Cov * ZtFirstDeriv;
			if (calc_pred_cov || calc_pred_var) {
				vec_t diag_Sigma_plus_ZtWZI(num_re_);
				diag_Sigma_plus_ZtWZI.array() = 1. / diag_SigmaI_plus_ZtWZ_.array();
				diag_Sigma_plus_ZtWZI.array() /= sigma2;
				diag_Sigma_plus_ZtWZI.array() -= 1.;
				diag_Sigma_plus_ZtWZI.array() /= sigma2;
				if (calc_pred_cov) {
					T_mat Maux = Cross_Cov * diag_Sigma_plus_ZtWZI.asDiagonal() * Cross_Cov.transpose();
					pred_cov += Maux;
				}
				if (calc_pred_var) {
					T_mat Maux = Cross_Cov * diag_Sigma_plus_ZtWZI.asDiagonal();
					T_mat Maux2 = Cross_Cov.cwiseProduct(Maux);
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] += Maux2.row(i).sum();
					}
				}
			}
		}//end PredictLaplaceApproxOnlyOneGroupedRECalculationsOnREScale

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued
		* \param fixed_effects Fixed effects component of location parameter
		* \param B Matrix B in Vecchia approximation for observed locations, Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation for observed locations
		* \param Bpo Lower left part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the observed locations
		* \param Bp Lower right part of matrix B in joint Vecchia approximation for observed and prediction locations with non-zero off-diagonal entries corresponding to the nearest neighbors of the prediction locations among the prediction locations
		* \param Dp Diagonal matrix with lower right part of matrix D in joint Vecchia approximation for observed and prediction locations
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		* \param CondObsOnly If true, the nearest neighbors for the predictions are found only among the observed data and Bp is an identity matrix
		*/
		void PredictLaplaceApproxVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const sp_mat_t& Bpo,
			sp_mat_t& Bp,
			const vec_t& Dp,
			vec_t& pred_mean,
			T_mat& pred_cov,
			vec_t& pred_var,
			bool calc_pred_cov,
			bool calc_pred_var,
			bool calc_mode,
			bool CondObsOnly) {
			if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLVecchia(y_data, y_data_int, fixed_effects, B, D_inv, false, Sigma_L_k_, false, mll);
			}
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			int num_pred = (int)Bp.cols();
			CHECK((int)Dp.size() == num_pred);
			if (CondObsOnly) {
				pred_mean = -Bpo * mode_;
			}
			else {
				vec_t Bpo_mode = Bpo * mode_;
				pred_mean = -Bp.triangularView<Eigen::UpLoType::UnitLower>().solve(Bpo_mode);
			}
			if (calc_pred_cov || calc_pred_var) {
				sp_mat_t Bp_inv, Bp_inv_Dp;
				//Version Simulation
				if (matrix_inversion_method_ == "iterative") {
					sp_mat_rm_t Bp_inv_Dp_rm, Bp_inv_rm;
					sp_mat_rm_t Bpo_rm = sp_mat_rm_t(Bpo);
					sp_mat_rm_t Bp_rm;
					sp_mat_rm_t Bp_inv_Bpo_rm; //Bp^(-1) * Bpo 
					if (CondObsOnly) {
						Bp_inv_Bpo_rm = Bpo_rm; //Bp = Id
					}
					else {
						Bp_rm = sp_mat_rm_t(Bp);
						Bp_inv_rm = sp_mat_rm_t(Bp_rm.rows(), Bp_rm.cols());
						Bp_inv_rm.setIdentity();
						TriangularSolve<sp_mat_rm_t, sp_mat_rm_t, sp_mat_rm_t>(Bp_rm, Bp_inv_rm, Bp_inv_rm, false);
						Bp_inv_Bpo_rm = Bp_inv_rm * Bpo_rm;
						Bp_inv_Dp_rm = Bp_inv_rm * Dp.asDiagonal();
					}
					den_mat_t pred_cov_dense;
					if (calc_pred_cov) {
						pred_cov_dense = den_mat_t::Zero(num_pred, num_pred);
					}
					if (calc_pred_var) {
						pred_var = vec_t::Zero(num_pred);
					}
					vec_t W_diag_sqrt = second_deriv_neg_ll_.cwiseSqrt();
					sp_mat_rm_t B_t_D_inv_sqrt_rm = B_rm_.transpose() * D_inv_rm_.cwiseSqrt();
					int num_threads;
#ifdef _OPENMP
					num_threads = omp_get_max_threads();
#else
					num_threads = 1;
#endif
					std::uniform_int_distribution<> unif(0, 2147483646);
					std::vector<RNG_t> parallel_rngs;
					for (int ig = 0; ig < num_threads; ++ig) {
						int seed_local = unif(cg_generator_);
						parallel_rngs.push_back(RNG_t(seed_local));
					}
#pragma omp parallel
					{
#pragma omp for nowait
						for (int i = 0; i < nsim_var_pred_; ++i) {
							//z_i ~ N(0,I)
							int thread_nb;
#ifdef _OPENMP
							thread_nb = omp_get_thread_num();
#else
							thread_nb = 0;
#endif
							std::normal_distribution<double> ndist(0.0, 1.0);
							vec_t rand_vec_pred_I_1(dim_mode_), rand_vec_pred_I_2(dim_mode_);
							for (int j = 0; j < dim_mode_; j++) {
								rand_vec_pred_I_1(j) = ndist(parallel_rngs[thread_nb]);
								rand_vec_pred_I_2(j) = ndist(parallel_rngs[thread_nb]);
							}
							//z_i ~ N(0,(Sigma^{-1} + W))
							vec_t rand_vec_pred_SigmaI_plus_W = B_t_D_inv_sqrt_rm * rand_vec_pred_I_1 + W_diag_sqrt.cwiseProduct(rand_vec_pred_I_2);
							vec_t rand_vec_pred_SigmaI_plus_W_inv(dim_mode_);
							//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
							bool has_NA_or_Inf = false;
							if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
								CGVecchiaLaplaceVecWinvplusSigma(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_.transpose(), rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf,
									cg_max_num_it_, 0, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_);
							}
							else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB") {
								CGVecchiaLaplaceVec(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf,
									cg_max_num_it_, 0, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, D_inv_plus_W_B_rm_);
							}
							else {
								Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
							}
							if (has_NA_or_Inf) {
								Log::REDebug(CG_NA_OR_INF_WARNING_);
							}
							//z_i ~ N(0, Bp^{-1} Bpo (Sigma^{-1} + W)^{-1} Bpo^T Bp^{-1})
							vec_t rand_vec_pred = Bp_inv_Bpo_rm * rand_vec_pred_SigmaI_plus_W_inv;
							if (calc_pred_cov) {
								den_mat_t pred_cov_private = rand_vec_pred * rand_vec_pred.transpose();
#pragma omp critical
								{
									pred_cov_dense += pred_cov_private;
								}
							}
							if (calc_pred_var) {
								vec_t pred_var_private = rand_vec_pred.cwiseProduct(rand_vec_pred);
#pragma omp critical
								{
									pred_var += pred_var_private;
								}
							}
						}

					}
					if (calc_pred_cov) {
						pred_cov_dense /= nsim_var_pred_;
						if (CondObsOnly) {
							pred_cov_dense.diagonal().array() += Dp.array();
						}
						else {
							pred_cov_dense += Bp_inv_Dp_rm * Bp_inv_rm.transpose();
						}
						ConvertTo_T_mat_FromDense<T_mat>(pred_cov_dense, pred_cov);
					}
					if (calc_pred_var) {
						pred_var /= nsim_var_pred_;
						if (CondObsOnly) {
							pred_var += Dp;
						}
						else {
							pred_var += Bp_inv_Dp_rm.cwiseProduct(Bp_inv_rm) * vec_t::Ones(num_pred);
						}
					}
				} //end iterative methods using simulation
				else {//using Cholesky decomposition
					sp_mat_t Maux; //Maux = L\(Bpo^T * Bp^-1), L = Chol(Sigma^-1 + W)
					if (CondObsOnly) {
						Maux = Bpo.transpose();//Bp = Id
					}
					else {
						Bp_inv = sp_mat_t(Bp.rows(), Bp.cols());
						Bp_inv.setIdentity();
						TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
						//Bp.triangularView<Eigen::UpLoType::UnitLower>().solveInPlace(Bp_inv);//much slower
						Maux = Bpo.transpose() * Bp_inv.transpose();
						Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
					}
					TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, Maux, Maux, false);
					if (calc_pred_cov) {
						if (CondObsOnly) {
							pred_cov = Maux.transpose() * Maux;
							pred_cov.diagonal().array() += Dp.array();
						}
						else {
							pred_cov = Bp_inv_Dp * Bp_inv.transpose() + Maux.transpose() * Maux;
						}
					}
					if (calc_pred_var) {
						pred_var = vec_t(num_pred);
						Maux = Maux.cwiseProduct(Maux);
						if (CondObsOnly) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_pred; ++i) {
								pred_var[i] = Dp[i] + Maux.col(i).sum();
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_pred; ++i) {
								pred_var[i] = (Bp_inv_Dp.row(i)).dot(Bp_inv.row(i)) + Maux.col(i).sum();
							}
						}
					}
				}
			}//end calc_pred_cov || calc_pred_var
		}//end PredictLaplaceApproxVecchia

//Note: the following is currently not used
//		/*!
//		* \brief Calculate variance of Laplace-approximated posterior
//		* \param ZSigmaZt Covariance matrix of latent random effect
//		* \param[out] pred_var Variance of Laplace-approximated posterior
//		*/
//		void CalcVarLaplaceApproxStable(const std::shared_ptr<T_mat> ZSigmaZt,
//			vec_t& pred_var) {
//			if (na_or_inf_during_last_call_to_find_mode_) {
//				Log::REFatal(NA_OR_INF_ERROR_);
//			}
//			CHECK(mode_has_been_calculated_);
//			pred_var = vec_t(num_re_);
//			vec_t diag_Wsqrt(second_deriv_neg_ll_.size());
//			diag_Wsqrt.array() = second_deriv_neg_ll_.array().sqrt();
//			T_mat L_inv_W_sqrt_ZSigmaZt = diag_Wsqrt.asDiagonal() * (*ZSigmaZt);
//			TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, L_inv_W_sqrt_ZSigmaZt, L_inv_W_sqrt_ZSigmaZt, false);
//#pragma omp parallel for schedule(static)
//			for (int i = 0; i < num_re_; ++i) {
//				pred_var[i] = (*ZSigmaZt).coeff(i,i) - L_inv_W_sqrt_ZSigmaZt.col(i).squaredNorm();
//			}
//		}//end CalcVarLaplaceApproxStable

		/*!
		* \brief Calculate variance of Laplace-approximated posterior
		* \param Sigma Covariance matrix of latent random effect
		* \param[out] pred_var Variance of Laplace-approximated posterior
		*/
		void CalcVarLaplaceApproxOnlyOneGPCalculationsOnREScale(const std::shared_ptr<T_mat> Sigma,
			vec_t& pred_var) {
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			pred_var = vec_t(num_re_);
			vec_t diag_ZtWZ_sqrt(second_deriv_neg_ll_.size());
			diag_ZtWZ_sqrt.array() = second_deriv_neg_ll_.array().sqrt();
			T_mat L_inv_ZtWZ_sqrt_Sigma = diag_ZtWZ_sqrt.asDiagonal() * (*Sigma);
			TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, L_inv_ZtWZ_sqrt_Sigma, L_inv_ZtWZ_sqrt_Sigma, false);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_re_; ++i) {
				pred_var[i] = (*Sigma).coeff(i, i) - L_inv_ZtWZ_sqrt_Sigma.col(i).squaredNorm();
			}
		}//end CalcVarLaplaceApproxOnlyOneGPCalculationsOnREScale

		/*!
		* \brief Calculate variance of Laplace-approximated posterior
		* \param[out] pred_var Variance of Laplace-approximated posterior
		*/
		void CalcVarLaplaceApproxGroupedRE(vec_t& pred_var) {
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			pred_var = vec_t(num_re_);
			sp_mat_t L_inv(num_re_, num_re_);
			L_inv.setIdentity();
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, L_inv, L_inv, false);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_re_; ++i) {
				pred_var[i] = L_inv.col(i).squaredNorm();
			}
		}//end CalcVarLaplaceApproxGroupedRE

		/*!
		* \brief Calculate variance of Laplace-approximated posterior
		* \param[out] pred_var Variance of Laplace-approximated posterior
		*/
		void CalcVarLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(vec_t& pred_var) {
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			pred_var = vec_t(num_re_);
			pred_var.array() = diag_SigmaI_plus_ZtWZ_.array().inverse();
		}//end CalcVarLaplaceApproxOnlyOneGroupedRECalculationsOnREScale

		/*!
		* \brief Calculate variance of Laplace-approximated posterior
		* \param[out] pred_var Variance of Laplace-approximated posterior
		*/
		void CalcVarLaplaceApproxVecchia(vec_t& pred_var) {
			if (na_or_inf_during_last_call_to_find_mode_) {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
			CHECK(mode_has_been_calculated_);
			pred_var = vec_t(num_re_);
			//Version Simulation
			if (matrix_inversion_method_ == "iterative") {
				pred_var = vec_t::Zero(num_re_);
				vec_t W_diag_sqrt = second_deriv_neg_ll_.cwiseSqrt();
				sp_mat_rm_t B_t_D_inv_sqrt_rm = B_rm_.transpose() * D_inv_rm_.cwiseSqrt();
				int num_threads;
#ifdef _OPENMP
				num_threads = omp_get_max_threads();
#else
				num_threads = 1;
#endif
				std::uniform_int_distribution<> unif(0, 2147483646);
				std::vector<RNG_t> parallel_rngs;
				for (int ig = 0; ig < num_threads; ++ig) {
					int seed_local = unif(cg_generator_);
					parallel_rngs.push_back(RNG_t(seed_local));
				}
#pragma omp parallel
				{
#pragma omp for nowait
					for (int i = 0; i < nsim_var_pred_; ++i) {
						//z_i ~ N(0,I)
						int thread_nb;
#ifdef _OPENMP
						thread_nb = omp_get_thread_num();
#else
						thread_nb = 0;
#endif
						std::normal_distribution<double> ndist(0.0, 1.0);
						vec_t rand_vec_pred_I_1(num_re_), rand_vec_pred_I_2(num_re_);
						for (int j = 0; j < num_re_; j++) {
							rand_vec_pred_I_1(j) = ndist(parallel_rngs[thread_nb]);
							rand_vec_pred_I_2(j) = ndist(parallel_rngs[thread_nb]);
						}
						//z_i ~ N(0,(Sigma^{-1} + W))
						vec_t rand_vec_pred_SigmaI_plus_W = B_t_D_inv_sqrt_rm * rand_vec_pred_I_1 + W_diag_sqrt.cwiseProduct(rand_vec_pred_I_2);
						vec_t rand_vec_pred_SigmaI_plus_W_inv(num_re_);
						//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
						bool has_NA_or_Inf = false;
						if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
							CGVecchiaLaplaceVecWinvplusSigma(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_.transpose(), rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf,
								cg_max_num_it_, 0, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_);
						}
						else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB") {
							CGVecchiaLaplaceVec(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf,
								cg_max_num_it_, 0, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, D_inv_plus_W_B_rm_);
						}
						else {
							Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
						}
						if (has_NA_or_Inf) {
							Log::REDebug(CG_NA_OR_INF_WARNING_);
						}
						vec_t pred_var_private = rand_vec_pred_SigmaI_plus_W_inv.cwiseProduct(rand_vec_pred_SigmaI_plus_W_inv);
#pragma omp critical
						{
							pred_var += pred_var_private;
						}
					}
				}
				pred_var /= nsim_var_pred_;
			} //end Version Simulation
			else {
				sp_mat_t L_inv(num_re_, num_re_);
				L_inv.setIdentity();
				TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, L_inv, L_inv, false);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_re_; ++i) {
					pred_var[i] = L_inv.col(i).squaredNorm();
				}
			}
		}//end CalcVarLaplaceApproxVecchia

		/*!
		* \brief Make predictions for the response variable (label) based on predictions for the mean and variance of the latent random effects
		* \param pred_mean[in & out] Predictive mean of latent random effects. The Predictive mean for the response variables is written on this
		* \param pred_var[in & out] Predictive variances of latent random effects. The predicted variance for the response variables is written on this
		* \param predict_var If true, predictive variances are also calculated
		*/
		void PredictResponse(vec_t& pred_mean,
			vec_t& pred_var,
			bool predict_var) {
			if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					pred_mean[i] = normalCDF(pred_mean[i] / std::sqrt(1. + pred_var[i]));
				}
				if (predict_var) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] = pred_mean[i] * (1. - pred_mean[i]);
					}
				}
			}
			else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					pred_mean[i] = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i]);
				}
				if (predict_var) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] = pred_mean[i] * (1. - pred_mean[i]);
					}
				}
			}
			else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					double pm = std::exp(pred_mean[i] + 0.5 * pred_var[i]);
					//double pm = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i]);// alternative version using quadrature
					if (predict_var) {
						pred_var[i] = pm * ((std::exp(pred_var[i]) - 1.) * pm + 1.);
						//double psm = RespMeanAdaptiveGHQuadrature(2 * pred_mean[i], 4 * pred_var[i]);// alternative version using quadrature
						//pred_var[i] = psm - pm * pm + pm;
					}
					pred_mean[i] = pm;
				}
			}
			else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					double pm = std::exp(pred_mean[i] + 0.5 * pred_var[i]);
					//double pm = RespMeanAdaptiveGHQuadrature(pred_mean[i], pred_var[i]);// alternative version using quadrature
					if (predict_var) {
						pred_var[i] = (std::exp(pred_var[i]) - 1.) * pm * pm + std::exp(2 * pred_mean[i] + 2 * pred_var[i]) / aux_pars_[0];
						//double psm = RespMeanAdaptiveGHQuadrature(2 * pred_mean[i], 4 * pred_var[i]);// alternative version using quadrature
						//pred_var[i] = psm - pm * pm + psm / aux_pars_[0];
					}
					pred_mean[i] = pm;
				}
			}
			else if (likelihood_type_ == "negative_binomial") {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					double pm = std::exp(pred_mean[i] + 0.5 * pred_var[i]);
					if (predict_var) {
						pred_var[i] = std::exp(2 * (pred_mean[i] + pred_var[i])) * (1 + 1 / aux_pars_[0]) + pm * (1 - pm);
					}
					pred_mean[i] = pm;
				}
			}
			else {
				Log::REFatal("PredictResponse: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
		}//end PredictResponse

		/*!
		* \brief Adaptive GH quadrature to calculate predictive mean of response variable
		* \param latent_mean Predictive mean of latent random effects
		* \param latent_var Predictive variances of latent random effects
		*/
		double RespMeanAdaptiveGHQuadrature(const double latent_mean,
			const double latent_var) {
			// Find mode of integrand
			double mode_integrand_last, update;
			double mode_integrand = 0.;
			double sigma2_inv = 1. / latent_var;
			double sqrt_sigma2_inv = std::sqrt(sigma2_inv);
			for (int it = 0; it < 100; ++it) {
				mode_integrand_last = mode_integrand;
				update = (FirstDerivLogCondMeanLikelihood(mode_integrand) - sigma2_inv * (mode_integrand - latent_mean))
					/ (SecondDerivLogCondMeanLikelihood(mode_integrand) - sigma2_inv);
				mode_integrand -= update;
				if (std::abs(update) / std::abs(mode_integrand_last) < DELTA_REL_CONV_) {
					break;
				}
			}
			// Adaptive GH quadrature
			double sqrt2_sigma_hat = M_SQRT2 / std::sqrt(-SecondDerivLogCondMeanLikelihood(mode_integrand) + sigma2_inv);
			double x_val;
			double mean_resp = 0.;
			for (int j = 0; j < order_GH_; ++j) {
				x_val = sqrt2_sigma_hat * GH_nodes_[j] + mode_integrand;
				mean_resp += adaptive_GH_weights_[j] * CondMeanLikelihood(x_val) * normalPDF(sqrt_sigma2_inv * (x_val - latent_mean));
			}
			mean_resp *= sqrt2_sigma_hat * sqrt_sigma2_inv;
			return mean_resp;
		}//end RespMeanAdaptiveGHQuadrature

		/*!
		* \brief Calculate test negative log-likelihood using adaptive GH quadrature
		* \param y_test Test response variable
		* \param pred_mean Predictive mean of latent random effects
		* \param pred_var Predictive variances of latent random effects
		* \param num_data Number of data points
		*/
		inline double TestNegLogLikelihoodAdaptiveGHQuadrature(const label_t* y_test,
			const double* pred_mean,
			const double* pred_var,
			const data_size_t num_data) const {
			double ll = 0.;
#pragma omp parallel for schedule(static) if (num_data >= 128) reduction(+:ll)
			for (data_size_t i = 0; i < num_data; ++i) {
				int y_test_int = 1;
				double y_test_d = static_cast<double>(y_test[i]);
				// Note: we need to convert from float to double as label_t is float. Unfortunately, the lightGBM part does not allow for setting the LABEL_T_USE_DOUBLE macro in meta.h (multiple bugs...)
				if (label_type() == "int") {
					y_test_int = static_cast<int>(y_test[i]);
				}
				// Find mode of integrand
				double mode_integrand_last, update;
				double mode_integrand = 0.;
				double sigma2_inv = 1. / pred_var[i];
				double sqrt_sigma2_inv = std::sqrt(sigma2_inv);
				for (int it = 0; it < 100; ++it) {
					mode_integrand_last = mode_integrand;
					update = (CalcFirstDerivLogLikOneSample(y_test_d, y_test_int, mode_integrand) - sigma2_inv * (mode_integrand - pred_mean[i]))
						/ (-CalcSecondDerivNegLogLikOneSample(y_test_d, y_test_int, mode_integrand) - sigma2_inv);
					mode_integrand -= update;
					if (std::abs(update) / std::abs(mode_integrand_last) < DELTA_REL_CONV_) {
						break;
					}
				}
				// Adaptive GH quadrature
				double sqrt2_sigma_hat = M_SQRT2 / std::sqrt(CalcSecondDerivNegLogLikOneSample(y_test_d, y_test_int, mode_integrand) + sigma2_inv);
				double x_val;
				double likelihood = 0.;
				for (int j = 0; j < order_GH_; ++j) {
					x_val = sqrt2_sigma_hat * GH_nodes_[j] + mode_integrand;
					likelihood += adaptive_GH_weights_[j] * std::exp(LogLikelihoodOneSample(y_test_d, y_test_int, x_val)) * normalPDF(sqrt_sigma2_inv * (x_val - pred_mean[i]));
				}
				likelihood *= sqrt2_sigma_hat * sqrt_sigma2_inv;
				ll += std::log(likelihood);
			}
			return -ll;
		}//end TestNegLogLikelihoodAdaptiveGHQuadrature

		/*! \brief Set matrix inversion properties and choices for iterative methods. This function is calle from re_model_template.h which also holds these variables */
		void SetMatrixInversionProperties(const string_t& matrix_inversion_method,
			int cg_max_num_it,
			int cg_max_num_it_tridiag,
			double cg_delta_conv,
			double cg_delta_conv_pred,
			int num_rand_vec_trace,
			bool reuse_rand_vec_trace,
			int seed_rand_vec_trace,
			const string_t& cg_preconditioner_type,
			int piv_chol_rank,
			int rank_pred_approx_matrix_lanczos,
			int nsim_var_pred) {
			matrix_inversion_method_ = matrix_inversion_method;
			cg_max_num_it_ = cg_max_num_it;
			cg_max_num_it_tridiag_ = cg_max_num_it_tridiag;
			cg_delta_conv_ = cg_delta_conv;
			cg_delta_conv_pred_ = cg_delta_conv_pred;
			num_rand_vec_trace_ = num_rand_vec_trace;
			reuse_rand_vec_trace_ = reuse_rand_vec_trace;
			seed_rand_vec_trace_ = seed_rand_vec_trace;
			cg_preconditioner_type_ = cg_preconditioner_type;
			piv_chol_rank_ = piv_chol_rank;
			rank_pred_approx_matrix_lanczos_ = rank_pred_approx_matrix_lanczos;
			nsim_var_pred_ = nsim_var_pred;
		}//end SetMatrixInversionProperties

		/*!
		* \brief Calculate log|Sigma W + I| using stochastic trace estimation and variance reduction.
		* \param num_data Number of data points
		* \param cg_max_num_it_tridiag Maximal number of iterations for conjugate gradient algorithm when being run as Lanczos algorithm for tridiagonalization
		* \param I_k_plus_Sigma_L_kt_W_Sigma_L_k Preconditioner "piv_chol_on_Sigma": I_k + Sigma_L_k^T W Sigma_L_k
		* \param[out] has_NA_or_Inf Is set to TRUE if NA or Inf occured in the conjugate gradient algorithm
		* \param[out] log_det_Sigma_W_plus_I Solution for log|Sigma W + I|
		*/
		void CalcLogDetStoch(const data_size_t& num_data, 
			const int& cg_max_num_it_tridiag,
			den_mat_t& I_k_plus_Sigma_L_kt_W_Sigma_L_k,
			bool& has_NA_or_Inf,
			double& log_det_Sigma_W_plus_I,
			sp_mat_t& SigmaI,
			const sp_mat_t& B) {
			CHECK(rand_vec_trace_I_.cols() == num_rand_vec_trace_);
			CHECK(rand_vec_trace_P_.cols() == num_rand_vec_trace_);
			if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
				CHECK(rand_vec_trace_I2_.cols() == num_rand_vec_trace_);
				CHECK(rand_vec_trace_I2_.rows() == Sigma_L_k_.cols());
				std::vector<vec_t> Tdiags_PI_WI_plus_Sigma(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag));
				std::vector<vec_t> Tsubdiags_PI_WI_plus_Sigma(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag - 1));
				//Get random vectors (z_1, ..., z_t) with Cov(z_i) = P:
				//For P = W^(-1) + Sigma_L_k Sigma_L_k^T: z_i = W^(-1/2) r_j + Sigma_L_k r_i, where r_i, r_j ~ N(0,I)
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					rand_vec_trace_P_.col(i) = Sigma_L_k_ * rand_vec_trace_I2_.col(i) + ((second_deriv_neg_ll_.cwiseInverse().cwiseSqrt()).array() * rand_vec_trace_I_.col(i).array()).matrix();
				}
				I_k_plus_Sigma_L_kt_W_Sigma_L_k.setIdentity();
				I_k_plus_Sigma_L_kt_W_Sigma_L_k += Sigma_L_k_.transpose() * second_deriv_neg_ll_.asDiagonal() * Sigma_L_k_;
				chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.compute(I_k_plus_Sigma_L_kt_W_Sigma_L_k);
				CGTridiagVecchiaLaplaceWinvplusSigma(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_.transpose(), rand_vec_trace_P_, Tdiags_PI_WI_plus_Sigma, Tsubdiags_PI_WI_plus_Sigma,
					WI_plus_Sigma_inv_Z_, has_NA_or_Inf, num_data, num_rand_vec_trace_, cg_max_num_it_tridiag, cg_delta_conv_, chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_);
				if (!has_NA_or_Inf) {
					double ldet_PI_WI_plus_Sigma;
					LogDetStochTridiag(Tdiags_PI_WI_plus_Sigma, Tsubdiags_PI_WI_plus_Sigma, ldet_PI_WI_plus_Sigma, num_data, num_rand_vec_trace_);
					//log|Sigma W + I| = log|P^(-1) (W^(-1) + Sigma)| + log|W| + log|P|
					//where log|P| = log|I_k + Sigma_L_k^T W Sigma_L_k| + log|W^(-1)| + log|I_k|, log|I_k| = 0
					log_det_Sigma_W_plus_I = ldet_PI_WI_plus_Sigma + second_deriv_neg_ll_.array().log().sum() +
						2 * ((den_mat_t)chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.matrixL()).diagonal().array().log().sum() - second_deriv_neg_ll_.array().log().sum();
				}
			}
			else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB") {
				std::vector<vec_t> Tdiags_PI_SigmaI_plus_W(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag));
				std::vector<vec_t> Tsubdiags_PI_SigmaI_plus_W(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag - 1));
				//Get random vectors (z_1, ..., z_t) with Cov(z_i) = P:
				//For P = B^T (D^(-1) + W) B: z_i = B^T (D^(-1) + W)^0.5 r_i, where r_i ~ N(0,I)
				vec_t D_inv_plus_W_diag = D_inv_rm_.diagonal() + second_deriv_neg_ll_;
				sp_mat_rm_t B_t_D_inv_plus_W_sqrt_rm = B_rm_.transpose() * (D_inv_plus_W_diag).cwiseSqrt().asDiagonal();
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					rand_vec_trace_P_.col(i) = B_t_D_inv_plus_W_sqrt_rm * rand_vec_trace_I_.col(i);
				}
				//rand_vec_trace_P_ = B_rm_.transpose() * ((D_inv_rm_.diagonal() + second_deriv_neg_ll_).cwiseSqrt().asDiagonal() * rand_vec_trace_I_);
				D_inv_plus_W_B_rm_ = (D_inv_plus_W_diag).asDiagonal() * B_rm_;
				CGTridiagVecchiaLaplace(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_, rand_vec_trace_P_, Tdiags_PI_SigmaI_plus_W, Tsubdiags_PI_SigmaI_plus_W,
					SigmaI_plus_W_inv_Z_, has_NA_or_Inf, num_data, num_rand_vec_trace_, cg_max_num_it_tridiag, cg_delta_conv_, D_inv_plus_W_B_rm_);
				if (!has_NA_or_Inf) {
					double ldet_PI_SigmaI_plus_W;
					LogDetStochTridiag(Tdiags_PI_SigmaI_plus_W, Tsubdiags_PI_SigmaI_plus_W, ldet_PI_SigmaI_plus_W, num_data, num_rand_vec_trace_);
					//log|Sigma W + I| = log|P^(-1) (Sigma^(-1) + W)| + log|P| + log|Sigma|
					//where log|P| = log|B^T (D^(-1) + W) B| = log|(D^(-1) + W)|
					log_det_Sigma_W_plus_I = ldet_PI_SigmaI_plus_W + D_inv_plus_W_diag.array().log().sum() - D_inv_rm_.diagonal().array().log().sum();
				}
			}
			else if (cg_preconditioner_type_ == "incomp_chol_on_SigmaI_plus_W") {
				//Update P with latest W
				sp_mat_t SigmaI_plus_W = SigmaI;
				SigmaI_plus_W.diagonal().array() += second_deriv_neg_ll_.array();
				ReverseIncompleteCholeskyFactorization(SigmaI_plus_W, B, L_SigmaI_plus_W_rm_);
				//////////////////////////////////////////////////////////////////////////////
				std::vector<vec_t> Tdiags_PI_SigmaI_plus_W(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag));
				std::vector<vec_t> Tsubdiags_PI_SigmaI_plus_W(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag - 1));
				//Get random vectors (z_1, ..., z_t) with Cov(z_i) = P:
				//For P = L^T L: z_i = L^T r_i, where r_i ~ N(0,I)
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					rand_vec_trace_P_.col(i) = L_SigmaI_plus_W_rm_.transpose() * rand_vec_trace_I_.col(i);
				}
				CGTridiagVecchiaLaplaceIncompChol(second_deriv_neg_ll_, B_rm_, B_t_D_inv_rm_, rand_vec_trace_P_, Tdiags_PI_SigmaI_plus_W, Tsubdiags_PI_SigmaI_plus_W,
					SigmaI_plus_W_inv_Z_, has_NA_or_Inf, num_data, num_rand_vec_trace_, cg_max_num_it_tridiag, cg_delta_conv_, L_SigmaI_plus_W_rm_);
				if (!has_NA_or_Inf) {
					double ldet_PI_SigmaI_plus_W;
					LogDetStochTridiag(Tdiags_PI_SigmaI_plus_W, Tsubdiags_PI_SigmaI_plus_W, ldet_PI_SigmaI_plus_W, num_data, num_rand_vec_trace_);
					//log|Sigma W + I| = log|P^(-1) (Sigma^(-1) + W)| + log|P| + log|Sigma|
					//where log|P| = log|L^T L| = log|(D^(-1) + W)|
					log_det_Sigma_W_plus_I = ldet_PI_SigmaI_plus_W + 2*(L_SigmaI_plus_W_rm_.diagonal().array().log().sum()) - D_inv_rm_.diagonal().array().log().sum();
					//Log::REInfo("ldet_PI_SigmaI_plus_W = %g", ldet_PI_SigmaI_plus_W);
					//Log::REInfo("2*(L_SigmaI_plus_W_rm_.diagonal().array().log().sum()) = %g", (2 * (L_SigmaI_plus_W_rm_.diagonal().array().log().sum())));
					//Log::REInfo("- D_inv_rm_.diagonal().array().log().sum() = %g", (-D_inv_rm_.diagonal().array().log().sum()));
				}
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
			}
			//Log::REInfo("log_det_Sigma_W_plus_I = %g", log_det_Sigma_W_plus_I);
		}//end CalcLogDetStoch

		/*! 
		* \brief Calculate dlog|Sigma W + I|/db_i for all i in 1, ..., n using stochastic trace estimation and variance reduction.
		* \param third_deriv Third derivative of the log-likelihood with respect to the mode.
		* \param num_data Number of data points
		* \param d_log_det_Sigma_W_plus_I_d_mode[out] Solution for dlog|Sigma W + I|/db_i for all i in n
		* \param D_inv_plus_W_inv_diag[out] Preconditioner "Sigma_inv_plus_BtWB": diagonal of (D^(-1) + W)^(-1)
		* \param diag_WI[out] Preconditioner "piv_chol_on_Sigma": diagonal of W^(-1)
		* \param PI_Z[out] Preconditioner "Sigma_inv_plus_BtWB": P^(-1) Z
		* \param WI_PI_Z[out] Preconditioner "piv_chol_on_Sigma": W^(-1) P^(-1) Z
		* \param[out] WI_WI_plus_Sigma_inv_Z Preconditioner "piv_chol_on_Sigma": W^(-1) (W^(-1) + Sigma)^(-1) Z
		*/
		void CalcLogDetStochDerivMode(const vec_t& third_deriv,
			const data_size_t& num_data,
			vec_t& d_log_det_Sigma_W_plus_I_d_mode,
			vec_t& D_inv_plus_W_inv_diag,
			vec_t& diag_WI,
			den_mat_t& PI_Z,
			den_mat_t& WI_PI_Z,
			den_mat_t& WI_WI_plus_Sigma_inv_Z) const{
			
			den_mat_t Z_PI_P_deriv_PI_Z;
			vec_t tr_PI_P_deriv_vec, c_opt;
			den_mat_t W_deriv_rep = third_deriv.replicate(1, num_rand_vec_trace_);
			W_deriv_rep *= -1; //since third_deriv = -dW/db_i
			if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
				//P^(-1) = (W^(-1) + Sigma_L_k Sigma_L_k^T)^(-1)
				//W^(-1) P^(-1) Z = Z - Sigma_L_k (I_k + Sigma_L_k^T W Sigma_L_k)^(-1) Sigma_L_k^T W Z
				den_mat_t Sigma_Lkt_W_Z;
				diag_WI = second_deriv_neg_ll_.cwiseInverse();
				if (Sigma_L_k_.cols() < num_rand_vec_trace_) {
					Sigma_Lkt_W_Z = (Sigma_L_k_.transpose() * second_deriv_neg_ll_.asDiagonal()) * rand_vec_trace_P_;
				}
				else {
					Sigma_Lkt_W_Z = Sigma_L_k_.transpose() * (second_deriv_neg_ll_.asDiagonal() * rand_vec_trace_P_);
				}
				WI_PI_Z = rand_vec_trace_P_ - Sigma_L_k_ * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.solve(Sigma_Lkt_W_Z);
				//tr(W^(-1) dW/db_i) - do not cancel with deterministic part of variance reduction when using optimal c
				vec_t tr_WI_W_deriv = -1 * diag_WI.cwiseProduct(third_deriv);
				//stochastic tr((Sigma + W^(-1))^(-1) dW^(-1)/db_i) 
				WI_WI_plus_Sigma_inv_Z = diag_WI.asDiagonal() * WI_plus_Sigma_inv_Z_;
				den_mat_t Z_WI_plus_Sigma_inv_WI_deriv_PI_Z = -1 * (WI_WI_plus_Sigma_inv_Z.array() * W_deriv_rep.array() * WI_PI_Z.array()).matrix();
				vec_t tr_WI_plus_Sigma_inv_WI_deriv = Z_WI_plus_Sigma_inv_WI_deriv_PI_Z.rowwise().mean();
				d_log_det_Sigma_W_plus_I_d_mode = tr_WI_plus_Sigma_inv_WI_deriv + tr_WI_W_deriv;
				//variance reduction
				//deterministic tr(Sigma_Lk (I_k + Sigma_Lk^T W Sigma_Lk)^(-1) Sigma_Lk^T dW/db_i) + tr(W dW^(-1)/db_i) (= - tr(W^(-1) dW/db_i))
				den_mat_t L_inv_Sigma_L_kt(Sigma_L_k_.cols(), num_data);
				TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_.transpose(), L_inv_Sigma_L_kt, false);
				den_mat_t L_inv_Sigma_L_kt_sqr = L_inv_Sigma_L_kt.cwiseProduct(L_inv_Sigma_L_kt);
				vec_t Sigma_Lk_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_Lkt_diag = L_inv_Sigma_L_kt_sqr.transpose() * vec_t::Ones(L_inv_Sigma_L_kt_sqr.rows()); //diagonal of Sigma_Lk (I_k + Sigma_Lk^T W Sigma_Lk)^(-1) Sigma_Lk^T
				vec_t tr_Sigma_Lk_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_Lkt_W_deriv = -1 * (Sigma_Lk_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_Lkt_diag.array() * third_deriv.array()); //times (-1), since third_deriv = -dW/db_i
				//stochastic tr(P^(-1) dP/db_i), where dP/db_i = - W^(-1) dW/db_i W^(-1)
				Z_PI_P_deriv_PI_Z = -1 * (WI_PI_Z.array() * W_deriv_rep.array() * WI_PI_Z.array()).matrix();
				tr_PI_P_deriv_vec = Z_PI_P_deriv_PI_Z.rowwise().mean();
				//optimal c
				CalcOptimalCVectorized(Z_WI_plus_Sigma_inv_WI_deriv_PI_Z, Z_PI_P_deriv_PI_Z, tr_WI_plus_Sigma_inv_WI_deriv, tr_PI_P_deriv_vec, c_opt);
				d_log_det_Sigma_W_plus_I_d_mode += c_opt.cwiseProduct(tr_Sigma_Lk_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_Lkt_W_deriv - tr_WI_W_deriv) - c_opt.cwiseProduct(tr_PI_P_deriv_vec);
			}
			else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB") {
				//P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
				//P^(-1) Z
				den_mat_t B_invt_Z(num_data, num_rand_vec_trace_);
				PI_Z.resize(num_data, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					B_invt_Z.col(i) = B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>().solve(rand_vec_trace_P_.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					PI_Z.col(i) = D_inv_plus_W_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_Z.col(i));
				}
				//stochastic tr((Sigma^(-1) + W)^(-1) dW/db_i)
				den_mat_t Z_SigmaI_plus_W_inv_W_deriv_PI_Z = (SigmaI_plus_W_inv_Z_.array() * W_deriv_rep.array() * PI_Z.array()).matrix();
				vec_t tr_SigmaI_plus_W_inv_W_deriv = Z_SigmaI_plus_W_inv_W_deriv_PI_Z.rowwise().mean();
				d_log_det_Sigma_W_plus_I_d_mode = tr_SigmaI_plus_W_inv_W_deriv;
				//variance reduction
				//deterministic tr((D^(-1) + W)^(-1) dW/db_i)
				D_inv_plus_W_inv_diag = (D_inv_rm_.diagonal() + second_deriv_neg_ll_).cwiseInverse();
				vec_t tr_D_inv_plus_W_inv_W_deriv = -1 * (D_inv_plus_W_inv_diag.array() * third_deriv.array()); //times (-1), since third_deriv = -dW/db_i
				//stochastic tr(P^(-1) dP/db_i), where dP/db_i = B^T dW/db_i B
				den_mat_t B_PI_Z = B_rm_ * PI_Z;
				Z_PI_P_deriv_PI_Z = (B_PI_Z.array() * W_deriv_rep.array() * B_PI_Z.array()).matrix();
				tr_PI_P_deriv_vec = Z_PI_P_deriv_PI_Z.rowwise().mean();
				//optimal c
				CalcOptimalCVectorized(Z_SigmaI_plus_W_inv_W_deriv_PI_Z, Z_PI_P_deriv_PI_Z, tr_SigmaI_plus_W_inv_W_deriv, tr_PI_P_deriv_vec, c_opt);
				d_log_det_Sigma_W_plus_I_d_mode += c_opt.cwiseProduct(tr_D_inv_plus_W_inv_W_deriv) - c_opt.cwiseProduct(tr_PI_P_deriv_vec);
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
			}
		} //end CalcLogDetStochDerivMode

		/*!
		* \brief Calculate dlog|Sigma W + I|/dtheta_j, using stochastic trace estimation and variance reduction.
		* \param num_data Number of data points
		* \param num_comps_total Total number of random effect components (= number of GPs)
		* \param j Index of current covariance parameter in vector theta
		* \param SigmaI_deriv_rm Derivative of Sigma^(-1) wrt. theta_j
		* \param B_grad_j Derivatives of matrices B ( = derivative of matrix -A) for Vecchia approximation wrt. theta_j
		* \param D_grad_j Derivatives of matrices D for Vecchia approximation wrt. theta_j
		* \param D_inv_plus_W_inv_diag Preconditioner "Sigma_inv_plus_BtWB": diagonal of (D^(-1) + W)^(-1)
		* \param PI_Z Preconditioner "Sigma_inv_plus_BtWB": P^(-1) Z
		* \param WI_PI_Z Preconditioner "piv_chol_on_Sigma": W^(-1) P^(-1) Z
		* \param[out] d_log_det_Sigma_W_plus_I_d_cov_pars Solution for dlog|Sigma W + I|/dtheta_j
		*/
		void CalcLogDetStochDerivCovPar(const data_size_t& num_data,
			const int& num_comps_total,
			const int& j,
			const sp_mat_rm_t& SigmaI_deriv_rm,
			const sp_mat_t& B_grad_j,
			const sp_mat_t& D_grad_j,
			const vec_t& D_inv_plus_W_inv_diag,
			const den_mat_t& PI_Z,
			const den_mat_t& WI_PI_Z,
			double& d_log_det_Sigma_W_plus_I_d_cov_pars) const {

			if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
				den_mat_t B_invt_WI_plus_Sigma_inv_Z(num_data, num_rand_vec_trace_), Sigma_WI_plus_Sigma_inv_Z(num_data, num_rand_vec_trace_);
				den_mat_t B_invt_PI_Z(num_data, num_rand_vec_trace_), Sigma_PI_Z(num_data, num_rand_vec_trace_);
				//Stochastic Trace: Calculate tr((Sigma + W^(-1))^(-1) dSigma/dtheta_j)
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					B_invt_WI_plus_Sigma_inv_Z.col(i) = B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>().solve(WI_plus_Sigma_inv_Z_.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					Sigma_WI_plus_Sigma_inv_Z.col(i) = B_t_D_inv_rm_.transpose().template triangularView<Eigen::UpLoType::Lower>().solve(B_invt_WI_plus_Sigma_inv_Z.col(i));
				}
				den_mat_t PI_Z_local = second_deriv_neg_ll_.asDiagonal() * WI_PI_Z;
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					B_invt_PI_Z.col(i) = B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>().solve(PI_Z_local.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					Sigma_PI_Z.col(i) = B_t_D_inv_rm_.transpose().template triangularView<Eigen::UpLoType::Lower>().solve(B_invt_PI_Z.col(i));
				}
				d_log_det_Sigma_W_plus_I_d_cov_pars = -1 * ((Sigma_WI_plus_Sigma_inv_Z.cwiseProduct(SigmaI_deriv_rm * Sigma_PI_Z)).colwise().sum()).mean();
				//no variance reduction since dSigma_L_k/d_theta_j can't be solved analytically
			}
			else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB") {
				//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dSigma^(-1)/dtheta_j)
				vec_t zt_SigmaI_plus_W_inv_SigmaI_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(SigmaI_deriv_rm * PI_Z)).colwise().sum()).transpose();
				double tr_SigmaI_plus_W_inv_SigmaI_deriv = zt_SigmaI_plus_W_inv_SigmaI_deriv_PI_z.mean();
				d_log_det_Sigma_W_plus_I_d_cov_pars = tr_SigmaI_plus_W_inv_SigmaI_deriv;
				//tr(Sigma^(-1) dSigma/dtheta_j)
				if (num_comps_total == 1 && j == 0) {
					d_log_det_Sigma_W_plus_I_d_cov_pars += num_data;
				}
				else {
					d_log_det_Sigma_W_plus_I_d_cov_pars += (D_inv_rm_.diagonal().array() * D_grad_j.diagonal().array()).sum();
				}
				//variance reduction
				double tr_D_inv_plus_W_inv_D_inv_deriv, tr_PI_P_deriv;
				vec_t zt_PI_P_deriv_PI_z;
				if (num_comps_total == 1 && j == 0) {
					//dD/dsigma2 = D and dB/dsigma2 = 0
					//deterministic tr((D^(-1) + W)^(-1) dD^(-1)/dsigma2), where dD^(-1)/dsigma2 = -D^(-1)
					tr_D_inv_plus_W_inv_D_inv_deriv = -1 * (D_inv_plus_W_inv_diag.array() * D_inv_rm_.diagonal().array()).sum();
					//stochastic tr(P^(-1) dP/dsigma2), where dP/dsigma2 = -Sigma^(-1)
					zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(SigmaI_deriv_rm * PI_Z)).colwise().sum()).transpose();
					tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				}
				else {
					//deterministic tr((D^(-1) + W)^(-1) dD^(-1)/dtheta_j)
					tr_D_inv_plus_W_inv_D_inv_deriv = -1 * (D_inv_plus_W_inv_diag.array() * D_inv_rm_.diagonal().array() * D_grad_j.diagonal().array() * D_inv_rm_.diagonal().array()).sum();
					//stochastic tr(P^(-1) dP/dtheta_j)
					sp_mat_rm_t Bt_W_Bgrad_rm = B_rm_.transpose() * second_deriv_neg_ll_.asDiagonal() * B_grad_j;
					sp_mat_rm_t P_deriv_rm = SigmaI_deriv_rm + sp_mat_rm_t(Bt_W_Bgrad_rm.transpose()) + Bt_W_Bgrad_rm;
					zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(P_deriv_rm * PI_Z)).colwise().sum()).transpose();
					tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				}
				//optimal c
				double c_opt;
				CalcOptimalC(zt_SigmaI_plus_W_inv_SigmaI_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_SigmaI_deriv, tr_PI_P_deriv, c_opt);
				d_log_det_Sigma_W_plus_I_d_cov_pars += c_opt * tr_D_inv_plus_W_inv_D_inv_deriv - c_opt * tr_PI_P_deriv;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
			}
		} //end CalcLogDetStochDerivCovPar

		/*!
		* \brief Calculate dlog|Sigma W + I|/daux, using stochastic trace estimation and variance reduction.
		* \param neg_third_deriv Negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
		* \param D_inv_plus_W_inv_diag Preconditioner "Sigma_inv_plus_BtWB": diagonal of (D^(-1) + W)^(-1)
		* \param diag_WI Preconditioner "piv_chol_on_Sigma": diagonal of W^(-1)
		* \param PI_Z Preconditioner "Sigma_inv_plus_BtWB": P^(-1) Z
		* \param WI_PI_Z Preconditioner "piv_chol_on_Sigma": W^(-1) P^(-1) Z
		* \param WI_WI_plus_Sigma_inv_Z Preconditioner "piv_chol_on_Sigma": W^(-1) (W^(-1) + Sigma)^(-1) Z
		* \param[out] d_detmll_d_aux_par Solution for dlog|Sigma W + I|/daux
		*/
		void CalcLogDetStochDerivAuxPar(const vec_t& neg_third_deriv,
			const vec_t& D_inv_plus_W_inv_diag,
			const vec_t& diag_WI,
			const den_mat_t& PI_Z,
			const den_mat_t& WI_PI_Z,
			const den_mat_t& WI_WI_plus_Sigma_inv_Z,
			double&	d_detmll_d_aux_par) const {

			double tr_PI_P_deriv, c_opt;
			vec_t zt_PI_P_deriv_PI_z;
			if (cg_preconditioner_type_ == "piv_chol_on_Sigma") {
				//tr(W^(-1) dW/daux) - do not cancel with deterministic part of variance reduction when using optimal c
				double tr_WI_W_deriv = (diag_WI.cwiseProduct(neg_third_deriv)).sum();
				//Stochastic Trace: Calculate tr((Sigma + W^(-1))^(-1) dW^(-1)/daux)
				vec_t zt_WI_plus_Sigma_inv_WI_deriv_PI_z = -1 * ((WI_WI_plus_Sigma_inv_Z.cwiseProduct(neg_third_deriv.asDiagonal() * WI_PI_Z)).colwise().sum()).transpose();
				double tr_WI_plus_Sigma_inv_WI_deriv = zt_WI_plus_Sigma_inv_WI_deriv_PI_z.mean();
				d_detmll_d_aux_par = tr_WI_plus_Sigma_inv_WI_deriv + tr_WI_W_deriv;
				//variance reduction
				//deterministic tr((I_k + Sigma_Lk^T W Sigma_Lk)^(-1) Sigma_Lk^T dW/daux Sigma_Lk) + tr(W dW^(-1)/daux) (= - tr(W^(-1) dW/daux))
				den_mat_t Sigma_L_kt_W_deriv_Sigma_L_k = Sigma_L_k_.transpose() * neg_third_deriv.asDiagonal() * Sigma_L_k_;
				double tr_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_L_kt_W_deriv_Sigma_L_k = (chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.solve(Sigma_L_kt_W_deriv_Sigma_L_k)).diagonal().sum();
				//stochastic tr(P^(-1) dP/daux), where dP/daux = - W^(-1) dW/daux W^(-1)
				zt_PI_P_deriv_PI_z = -1 * ((WI_PI_Z.cwiseProduct(neg_third_deriv.asDiagonal() * WI_PI_Z)).colwise().sum()).transpose();
				tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				//optimal c
				CalcOptimalC(zt_WI_plus_Sigma_inv_WI_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_WI_plus_Sigma_inv_WI_deriv, tr_PI_P_deriv, c_opt);
				d_detmll_d_aux_par += c_opt * (tr_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_L_kt_W_deriv_Sigma_L_k - tr_WI_W_deriv) - c_opt * tr_PI_P_deriv;
			}
			else if (cg_preconditioner_type_ == "Sigma_inv_plus_BtWB") {
				//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
				vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(neg_third_deriv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
				double tr_SigmaI_plus_W_inv_W_deriv = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
				d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv;
				//variance reduction
				//deterministic tr((D^(-1) + W)^(-1) dW/daux)
				double tr_D_inv_plus_W_inv_W_deriv = (D_inv_plus_W_inv_diag.array() * neg_third_deriv.array()).sum();
				//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
				sp_mat_rm_t P_deriv_rm = B_rm_.transpose() * neg_third_deriv.asDiagonal() * B_rm_;
				zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(P_deriv_rm * PI_Z)).colwise().sum()).transpose();
				tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				//optimal c
				CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv, tr_PI_P_deriv, c_opt);
				d_detmll_d_aux_par += c_opt * tr_D_inv_plus_W_inv_W_deriv - c_opt * tr_PI_P_deriv;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type_.c_str());
			}
		} //end CalcLogDetStochDerivAuxPar

		static string_t ParseLikelihoodAlias(const string_t& likelihood) {
			if (likelihood == string_t("binary") || likelihood == string_t("bernoulli_probit") || likelihood == string_t("binary_probit")) {
				return "bernoulli_probit";
			}
			else if (likelihood == string_t("bernoulli_logit") || likelihood == string_t("binary_logit")) {
				return "bernoulli_logit";
			}
			else if (likelihood == string_t("gaussian") || likelihood == string_t("regression")) {
				return "gaussian";
			}
			return likelihood;
		}

		string_t ParseLikelihoodAliasGradientDescent(const string_t& likelihood) {
			if (likelihood.size() > 13) {
				if (likelihood.substr(likelihood.size() - 13) == string_t("_quasi-newton")) {
					quasi_newton_for_mode_finding_ = true;
					DELTA_REL_CONV_ = 1e-9;
					return likelihood.substr(0, likelihood.size() - 13);
				}
			}
			return likelihood;
		}

	private:
		/*! \brief Number of data points */
		data_size_t num_data_;
		/*! \brief Number (dimension) of random effects */
		data_size_t num_re_;
		/*! \brief Dimension (= length) of mode_ */
		data_size_t dim_mode_;
		/*! \brief Posterior mode used for Laplace approximation */
		vec_t mode_;
		/*! \brief Saving a previously found value allows for reseting the mode when having a too large step size. */
		vec_t mode_previous_value_;
		/*! \brief Auxiliary variable a=ZSigmaZt^-1 mode_b used for Laplace approximation */
		vec_t a_vec_;
		/*! \brief Saving a previously found value allows for reseting the mode when having a too large step size. */
		vec_t a_vec_previous_value_;
		/*! \brief Indicates whether the vector a_vec_ / a=ZSigmaZt^-1 is used or not */
		bool has_a_vec_;
		/*! \brief First derivatives of the log-likelihood. If use_Z_for_duplicates_, this corresponds to Z^T * first_deriv_ll, i.e., it length is num_re_ */
		vec_t first_deriv_ll_;
		/*! \brief Second derivatives of the negative log-likelihood (diagonal of matrix "W"). If use_Z_for_duplicates_, this corresponds to Z^T * second_deriv_neg_ll, i.e., it length is num_re_ */
		vec_t second_deriv_neg_ll_;
		/*! \brief First derivatives of the log-likelihood on the data scale of length num_data_. Auxiliary variable used only if use_Z_for_duplicates_ */
		vec_t first_deriv_ll_data_scale_;
		/*! \brief Second derivatives of the negative log-likelihood (diagonal of matrix "W") on the data scale of length num_data_. Auxiliary variable used only if use_Z_for_duplicates_ */
		vec_t second_deriv_neg_ll_data_scale_;
		/*! \brief Diagonal of matrix Sigma^-1 + Zt * W * Z in Laplace approximation (used only in version 'GroupedRE' when there is only one random effect and ZtWZ is diagonal. Otherwise 'diag_SigmaI_plus_ZtWZ_' is used for grouped REs) */
		vec_t diag_SigmaI_plus_ZtWZ_;
		/*! \brief Cholesky factors of matrix Sigma^-1 + Zt * W * Z in Laplace approximation (used only in version'GroupedRE' if there is more than one random effect). */
		chol_sp_mat_t chol_fact_SigmaI_plus_ZtWZ_grouped_;
		/*! \brief Cholesky factors of matrix Sigma^-1 + Zt * W * Z in Laplace approximation (used only in version 'Vecchia') */
		chol_sp_mat_t chol_fact_SigmaI_plus_ZtWZ_vecchia_;
		/*!
		* \brief Cholesky factors of matrix B = I + Wsqrt *  Z * Sigma * Zt * Wsqrt in Laplace approximation (for version 'Stable')
		*		or of matrix B = Id + ZtWZsqrt * Sigma * ZtWZsqrt (for version 'OnlyOneGPCalculationsOnREScale')
		*/
		T_chol chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_;
		/*! \brief If true, the pattern for the Cholesky factor (chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, chol_fact_SigmaI_plus_ZtWZ_grouped_, or chol_fact_SigmaI_plus_ZtWZ_vecchia_) has been analyzed */
		bool chol_fact_pattern_analyzed_ = false;
		/*! \brief If true, the mode has been initialized to 0 */
		bool mode_initialized_ = false;
		/*! \brief If true, the mode has been determined */
		bool mode_has_been_calculated_ = false;
		/*! \brief If true, NA or Inf has occurred during the last call to find mode */
		bool na_or_inf_during_last_call_to_find_mode_ = false;
		/*! \brief If true, NA or Inf has occurred during the second last call to find mode when mode_previous_value_ was calculated */
		bool na_or_inf_during_second_last_call_to_find_mode_ = false;
		/*! \brief Normalizing constant of the log-likelihood (not all likelihoods have one) */
		double log_normalizing_constant_;
		/*! \brief If true, the function 'CalculateLogNormalizingConstant' has been called */
		bool normalizing_constant_has_been_calculated_ = false;
		/*! \brief Auxiliary quantities that do not depend on aux_pars_ for normalizing constant for likelihoods (not all likelihoods have one, for gamma this is sum( log(y) ) ) */
		double aux_log_normalizing_constant_;
		/*! \brief If true, the function 'CalculateAuxQuantLogNormalizingConstant' has been called */
		bool aux_normalizing_constant_has_been_calculated_ = false;
		/*! \brief If true, an incidendce matrix Z is used for duplicate locations and calculations are done on the random effects scale with the unique locations (only for Gaussian processes) */
		bool use_Z_for_duplicates_ = false;
		/*! \brief Indices that indicate to which random effect every data point is related */
		const data_size_t* random_effects_indices_of_data_;

		/*! \brief Type of likelihood  */
		string_t likelihood_type_ = "gaussian";
		/*! \brief List of supported covariance likelihoods */
		const std::set<string_t> SUPPORTED_LIKELIHOODS_{ "gaussian", "bernoulli_probit", "bernoulli_logit", "poisson", "gamma", "negative_binomial"};
		/*! \brief Maximal number of iteration done for finding posterior mode with Newton's method */
		int MAXIT_MODE_NEWTON_ = 100000;//1000;
		/*! \brief Used for checking convergence in mode finding algorithm (terminate if relative change in Laplace approx. is below this value) */
		double DELTA_REL_CONV_ = 1e-6;
		/*! \brief If true, a quasi-Newton method instead of Newton's method is used for finding the maximal mode. Only supported for the Vecchia approximation */
		bool quasi_newton_for_mode_finding_ = false;
		/*! \brief Maximal number of steps for which learning rate shrinkage is done in the quasi-Newton method for mode finding in Laplace approximation */
		int MAX_NUMBER_LR_SHRINKAGE_STEPS_ = 30;
		/*! \brief If true, the mode can only change by 'MAX_CHANGE_MODE_NEWTON_' in Newton's method */
		bool cap_change_mode_newton_ = false;
		/*! \brief Maximally allowed change for mode in Newton's method for those likelihoods where a cap is enforced */
		double MAX_CHANGE_MODE_NEWTON_ = std::log(100.);
		/*! \brief Number of additional parameters for likelihoods  */
		int num_aux_pars_;
		/*! \brief Additional parameters for likelihoods. For "gamma", aux_pars_[0] = shape parameter, for gaussian, aux_pars_[0] = 1 / sqrt(variance) */
		std::vector<double> aux_pars_;
		/*! \brief Names of additional parameters for likelihoods */
		std::vector<string_t> names_aux_pars_;
		/*! \brief True, if the function 'SetAuxPars' has been called */
		bool aux_pars_have_been_set_ = false;

		// MATRIX INVERSION PROPERTIES
		/*! \brief Matrix inversion method */
		string_t matrix_inversion_method_;
		/*! \brief Maximal number of iterations for conjugate gradient algorithm */
		int cg_max_num_it_;
		/*! \brief Maximal number of iterations for conjugate gradient algorithm when being run as Lanczos algorithm for tridiagonalization */
		int cg_max_num_it_tridiag_;
		/*! \brief Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for parameter estimation */
		double cg_delta_conv_;
		/*! \brief Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for prediction */
		double cg_delta_conv_pred_;
		/*! \brief Number of random vectors (e.g., Rademacher) for stochastic approximation of the trace of a matrix */
		int num_rand_vec_trace_;
		/*! \brief If true, random vectors (e.g., Rademacher) for stochastic approximation of the trace of a matrix are sampled only once at the beginning of Newton's method for finding the mode in the Laplace approximation and are then reused in later trace approximations, otherwise they are sampled every time a trace is calculated */
		bool reuse_rand_vec_trace_;
		/*! \brief Seed number to generate random vectors (e.g., Rademacher) */
		int seed_rand_vec_trace_;
		/*! \brief Type of preconditioner used for conjugate gradient algorithms */
		string_t cg_preconditioner_type_;
		/*! \brief Rank of the pivoted Cholesky decomposition used as preconditioner in conjugate gradient algorithms */
		int piv_chol_rank_;
		/*! \brief Rank of the matrix for approximating predictive covariance matrices obtained using the Lanczos algorithm */
		int rank_pred_approx_matrix_lanczos_;
		/*! \brief Number of samples when simulation is used for calculating predictive variances */
		int nsim_var_pred_;
		/*! \brief If true, cg_max_num_it and cg_max_num_it_tridiag are reduced by 2/3 (multiplied by 1/3) for the mode finding of the Laplace approximation in the first gradient step when finding a learning rate that reduces the ll */
		bool reduce_cg_max_num_it_first_optim_step_ = true;

		//ITERATIVE MATRIX INVERSION + VECCIA APPROXIMATION
		//A) ROW-MAJOR MATRICES OF VECCIA APPROXIMATION
		/*! \brief Row-major matrix of the Veccia-matrix B*/
		sp_mat_rm_t B_rm_;
		/*! \brief Row-major matrix of the Veccia-matrix D_inv*/
		sp_mat_rm_t D_inv_rm_;
		/*! \brief Row-major matrix of B^T D^(-1)*/
		sp_mat_rm_t B_t_D_inv_rm_;

		//B) RANDOM VECTOR VARIABLES
		/*! Random number generator used to generate rand_vec_trace_I_*/
		RNG_t cg_generator_;
		/*! If the seed of the random number generator cg_generator_ is set, cg_generator_seeded_ is set to true*/
		bool cg_generator_seeded_ = false;
		/*! If reuse_rand_vec_trace_ is true and rand_vec_trace_I_ has been generated for the first time, then saved_rand_vec_trace_ is set to true */
		bool saved_rand_vec_trace_ = false;
		/*! Matrix of random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I, r_i is of dimension num_data, and t = num_rand_vec_trace_ */
		den_mat_t rand_vec_trace_I_;
		/*! Matrix of random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I, r_i is of dimension piv_chol_rank_, and t = num_rand_vec_trace_. This is used only if cg_preconditioner_type_ == "piv_chol_on_Sigma" */
		den_mat_t rand_vec_trace_I2_;
		/*! Matrix Z of random vectors (z_1, ..., z_t) with Cov(z_i) = P (P being the preconditioner matrix), z_i is of dimension num_data, and t = num_rand_vec_trace_ */
		den_mat_t rand_vec_trace_P_;
		/*! Matrix to store (Sigma^(-1) + W)^(-1) (z_1, ..., z_t) calculated in CGTridiagVecchiaLaplace() for later use in the stochastic trace approximation when calculating the gradient*/
		den_mat_t SigmaI_plus_W_inv_Z_;
		/*! Matrix to store (W^(-1) + Sigma)^(-1) (z_1, ..., z_t) calculated in CGTridiagVecchiaLaplaceSigmaplusWinv() for later use in the stochastic trace approximation when calculating the gradient*/
		den_mat_t WI_plus_Sigma_inv_Z_;

		//C) PRECONDITIONER VARIABLES
		/*! \brief piv_chol_on_Sigma: matrix of dimension nxk with rank(Sigma_L_k_) <= piv_chol_rank generated in re_model_template.h*/
		den_mat_t Sigma_L_k_;
		/*! \brief piv_chol_on_Sigma: Factor E of matrix EE^T = (I_k + Sigma_L_k_^T W Sigma_L_k_)*/
		chol_den_mat_t chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_;
		/*! \brief Sigma_inv_plus_BtWB (P = B^T (D^(-1) + W) B): matrix that contains the product (D^(-1) + W) B */
		sp_mat_rm_t D_inv_plus_W_B_rm_;
		sp_mat_rm_t L_SigmaI_plus_W_rm_;

		/*! \brief Order of the Gauss-Hermite quadrature */
		int order_GH_ = 30;
		/*! \brief Nodes and weights for the Gauss-Hermite quadrature */
		// Source: https://keisan.casio.com/exec/system/1281195844
		const std::vector<double> GH_nodes_ = { -6.863345293529891581061,
										-6.138279220123934620395,
										-5.533147151567495725118,
										-4.988918968589943944486,
										-4.48305535709251834189,
										-4.003908603861228815228,
										-3.544443873155349886925,
										-3.099970529586441748689,
										-2.667132124535617200571,
										-2.243391467761504072473,
										-1.826741143603688038836,
										-1.415527800198188511941,
										-1.008338271046723461805,
										-0.6039210586255523077782,
										-0.2011285765488714855458,
										0.2011285765488714855458,
										0.6039210586255523077782,
										1.008338271046723461805,
										1.415527800198188511941,
										1.826741143603688038836,
										2.243391467761504072473,
										2.667132124535617200571,
										3.099970529586441748689,
										3.544443873155349886925,
										4.003908603861228815228,
										4.48305535709251834189,
										4.988918968589943944486,
										5.533147151567495725118,
										6.138279220123934620395,
										6.863345293529891581061 };
		const std::vector<double> GH_weights_ = { 2.908254700131226229411E-21,
										2.8103336027509037088E-17,
										2.87860708054870606219E-14,
										8.106186297463044204E-12,
										9.1785804243785282085E-10,
										5.10852245077594627739E-8,
										1.57909488732471028835E-6,
										2.9387252289229876415E-5,
										3.48310124318685523421E-4,
										0.00273792247306765846299,
										0.0147038297048266835153,
										0.0551441768702342511681,
										0.1467358475408900997517,
										0.2801309308392126674135,
										0.386394889541813862556,
										0.3863948895418138625556,
										0.2801309308392126674135,
										0.1467358475408900997517,
										0.0551441768702342511681,
										0.01470382970482668351528,
										0.002737922473067658462989,
										3.48310124318685523421E-4,
										2.938725228922987641501E-5,
										1.579094887324710288346E-6,
										5.1085224507759462774E-8,
										9.1785804243785282085E-10,
										8.10618629746304420399E-12,
										2.87860708054870606219E-14,
										2.81033360275090370876E-17,
										2.9082547001312262294E-21 };
		const std::vector<double> adaptive_GH_weights_ = { 0.83424747101276179534,
										0.64909798155426670071,
										0.56940269194964050397,
										0.52252568933135454964,
										0.491057995832882696506,
										0.46837481256472881677,
										0.45132103599118862129,
										0.438177022652683703695,
										0.4279180629327437485828,
										0.4198950037368240886418,
										0.413679363611138937184,
										0.4089815750035316024972,
										0.4056051233256844363121,
										0.403419816924804022553,
										0.402346066701902927115,
										0.4023460667019029271154,
										0.4034198169248040225528,
										0.4056051233256844363121,
										0.4089815750035316024972,
										0.413679363611138937184,
										0.4198950037368240886418,
										0.427918062932743748583,
										0.4381770226526837037,
										0.45132103599118862129,
										0.46837481256472881677,
										0.4910579958328826965056,
										0.52252568933135454964,
										0.56940269194964050397,
										0.64909798155426670071,
										0.83424747101276179534 };

		const char* NA_OR_INF_WARNING_ = "Mode finding algorithm for Laplace approximation: NA or Inf occurred. "
			"This is not necessary a problem as it might have been the cause of a too large learning rate which, "
			"consequently, might have been decreased by the optimization algorithm ";
		const char* NA_OR_INF_ERROR_ = "NA or Inf occurred in the mode finding algorithm for the Laplace approximation ";
		const char* NO_INCREASE_IN_MLL_WARNING_ = "Mode finding algorithm for Laplace approximation: "
			"The approximate marginal log-likelihood (=convergence criterion) has decreased and the algorithm has thus been terminated ";
		const char* NO_CONVERGENCE_WARNING_ = "Algorithm for finding mode for Laplace approximation has not "
			"converged after the maximal number of iterations ";
		const char* CG_NA_OR_INF_WARNING_ = "NA or Inf occured in the Conjugate Gradient Algorithm when calculating the gradients.";

	};//end class Likelihood

}  // namespace GPBoost

#endif   // GPB_LIKELIHOODS_
