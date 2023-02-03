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
*		- lambda = gamma * exp(-location_par) (i.e., mean(y) = exp(-location_par)
*		- lambda = rate parameter, gamma = shape parameter, location_par = random plus fixed effects
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

#include <string>
#include <set>
#include <string>
#include <vector>

#include <LightGBM/utils/log.h>
using LightGBM::Log;

//Mathematical constants usually defined in cmath
#ifndef M_SQRT2
#define M_SQRT2      1.414213562373095048801688724209698079 //sqrt(2)
#endif

#include <chrono>  // only for debugging
#include <thread> // only for debugging

namespace GPBoost {

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
		* \param Indicates whether the vector a_vec_ / a=ZSigmaZt^-1 is used or not
		*/
		Likelihood(string_t type,
			data_size_t num_data,
			data_size_t num_re,
			bool has_a_vec) {
			string_t likelihood = ParseLikelihoodAlias(type);
			if (SUPPORTED_LIKELIHOODS_.find(likelihood) == SUPPORTED_LIKELIHOODS_.end()) {
				Log::REFatal("Likelihood of type '%s' is not supported.", likelihood.c_str());
			}
			likelihood_type_ = likelihood;
			num_data_ = num_data;
			num_re_ = num_re;
			if (likelihood_type_ == "gamma") {
				aux_pars_ = { 1. };//shape parameter, TODO: also estimate this parameter
			}
			chol_fact_pattern_analyzed_ = false;
			has_a_vec_ = has_a_vec;
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
			first_deriv_ll_ = vec_t(num_data_);
			second_deriv_neg_ll_ = vec_t(num_data_);
			mode_has_been_calculated_ = false;
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
			if (SUPPORTED_LIKELIHOODS_.find(likelihood) == SUPPORTED_LIKELIHOODS_.end()) {
				Log::REFatal("Likelihood of type '%s' is not supported.", likelihood.c_str());
			}
			likelihood_type_ = likelihood;
			chol_fact_pattern_analyzed_ = false;
		}

		/*!
		* \brief Returns the type of the response variable (label). Either "double" or "int"
		*/
		string_t label_type() const {
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit" ||
				likelihood_type_ == "poisson") {
				return("int");
			}
			else {
				return("double");
			}
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
		void CheckY(const T* y_data, const data_size_t num_data) const {
			if (likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit") {
				//#pragma omp parallel for schedule(static)//problematic with error message below... 
				for (data_size_t i = 0; i < num_data; ++i) {
					if (fabs(y_data[i]) >= EPSILON_NUMBERS && !TwoNumbersAreEqual<T>(y_data[i], 1.)) {
						Log::REFatal("Response variable (label) data needs to be 0 or 1 for likelihood of type '%s'.", likelihood_type_.c_str());
					}
				}
			}
			else if (likelihood_type_ == "poisson") {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] < 0) {
						Log::REFatal("Found negative response variable. Response variable cannot be negative for likelihood of type '%s'.", likelihood_type_.c_str());
					}
					else {
						double intpart;
						if (std::modf(y_data[i], &intpart) != 0.0) {
							Log::REFatal("Found non-integer response variable. Response variable can only be integer valued for likelihood of type '%s'.", likelihood_type_.c_str());
						}
					}
				}
			}
			else if (likelihood_type_ == "gamma") {
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] < 0) {
						Log::REFatal("Found negative response variable. Response variable cannot be negative for likelihood of type '%s'.", likelihood_type_.c_str());
					}
				}
			}
			else {
				Log::REFatal("CheckY: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
		}

		/*!
		* \brief Determine initial value for intercept (=constant)
		* \param y_data Response variable data
		* \param num_data Number of data points
		* param rand_eff_var Variance of random effects
		*/
		double FindInitialIntercept(const double* y_data, 
			const data_size_t num_data,
			double rand_eff_var) const {
			CHECK(rand_eff_var > 0.);
			double init_intercept = 0.;
			if (likelihood_type_ == "gaussian") {
#pragma omp parallel for schedule(static) reduction(+:init_intercept)
				for (data_size_t i = 0; i < num_data; ++i) {
					init_intercept += y_data[i];
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
			else if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma") {
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
		}

		/*!
		* \brief Determine initial value for intercept (=constant)
		* \param y_data Response variable data
		* \param num_data Number of data points
		* \param rand_eff_var Variance of random effects
		*/
		bool ShouldHaveIntercept(const double* y_data, 
			const data_size_t num_data,
			double rand_eff_var) const {
			bool ret_val = false;
			if (likelihood_type_ == "poisson" || likelihood_type_ == "gamma") {
				ret_val = true;
			}
			else {
				double beta_zero = FindInitialIntercept(y_data, num_data, rand_eff_var);
				if (std::abs(beta_zero) > 0.1) {
					ret_val = true;
				}
			}
			return(ret_val);
		}

		/*!
		* \brief Calculate normalizing constant of negative log-likelihood
		* \param y_data Response variable data
		* \param num_data Number of data points
		*/
		template <typename T>//T can be double or int
		void CalculateNormalizingConstant(const T* y_data, const data_size_t num_data) {
			if (likelihood_type_ == "poisson") {
				double log_normalizing_constant = 0.;
#pragma omp parallel for schedule(static) reduction(+:log_normalizing_constant)
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data[i] > 1) {
						double log_factorial = 0.;
						for (int k = 2; k <= y_data[i]; ++k) {
							log_factorial += std::log(k);
						}
						log_normalizing_constant += log_factorial;
					}
				}
				log_normalizing_constant_ = log_normalizing_constant;
			}
			else if (likelihood_type_ == "gamma") {
				//				//Currently not used since aux_pars_[0]==1 and thus log_normalizing_constant_==0
				//				double log_normalizing_constant = 0.;
				//#pragma omp parallel for schedule(static) reduction(+:log_normalizing_constant)
				//				for (data_size_t i = 0; i < num_data; ++i) {
				//					log_normalizing_constant += -(aux_pars_[0] - 1.) * std::log(y_data[i]) - aux_pars_[0] * std::log(aux_pars_[0]) + std::tgamma(aux_pars_[0]);
				//				}
				//				log_normalizing_constant_ = log_normalizing_constant;
				log_normalizing_constant_ = 0. * y_data[0];//y_data[0] is just a trick to avoid compiler warnings complaining about unreferenced parameters...
			}
			else if (likelihood_type_ == "gaussian" || likelihood_type_ == "bernoulli_probit" || likelihood_type_ == "bernoulli_logit") {
				log_normalizing_constant_ = 0. * y_data[0];//y_data[0] is just a trick to avoid compiler warnings complaining about unreferenced parameters...
			}
			else {
				Log::REFatal("CalculateNormalizingConstant: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
			normalizing_constant_has_been_calculated_ = true;
		}

		/*!
		* \brief Evaluate the log-likelihood conditional on the latent variable (=location_par)
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param location_par Location parameter (random plus fixed effects)
		* \param num_data Number of data points
		*/
		double LogLikelihood(const double* y_data, const int* y_data_int,
			const double* location_par, const data_size_t num_data) {
			if (!normalizing_constant_has_been_calculated_) {
				Log::REFatal("The normalizing constant has not been calculated. Call 'CalculateNormalizingConstant' first.");
			}
			double ll = 0.;
			if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data_int[i] == 0) {
						ll += std::log(1 - normalCDF(location_par[i]));
					}
					else {
						ll += std::log(normalCDF(location_par[i]));
					}
				}
			}
			else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += y_data_int[i] * location_par[i] - std::log(1 + std::exp(location_par[i]));
					//Alternative version:
					//if (y_data_int[i] == 0) {
					//	ll += std::log(1 - CondMeanLikelihood(location_par[i]));//CondMeanLikelihood = logistic function
					//}
					//else {
					//	ll += std::log(CondMeanLikelihood(location_par[i]));
					//}
				}
			}
			else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += y_data_int[i] * location_par[i] - std::exp(location_par[i]);
				}
				ll -= log_normalizing_constant_;
			}
			else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static) reduction(+:ll)
				for (data_size_t i = 0; i < num_data; ++i) {
					ll += -aux_pars_[0] * (location_par[i] + y_data[i] * std::exp(-location_par[i]));
				}
				ll -= log_normalizing_constant_;
			}
			else {
				Log::REFatal("LogLikelihood: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
			return(ll);
		}

		/*!
		* \brief Calculate the first derivative of the log-likelihood with respect to the location parameter
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param location_par Location parameter (random plus fixed effects)
		* \param num_data Number of data points
		*/
		void CalcFirstDerivLogLik(const double* y_data, const int* y_data_int,
			const double* location_par, const data_size_t num_data) {
			if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					if (y_data_int[i] == 0) {
						first_deriv_ll_[i] = -normalPDF(location_par[i]) / (1 - normalCDF(location_par[i]));
					}
					else {
						first_deriv_ll_[i] = normalPDF(location_par[i]) / normalCDF(location_par[i]);
					}
				}
			}
			else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					first_deriv_ll_[i] = y_data_int[i] - CondMeanLikelihood(location_par[i]);//CondMeanLikelihood = logistic(x)
				}
			}
			else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					first_deriv_ll_[i] = y_data_int[i] - std::exp(location_par[i]);
				}
			}
			else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					first_deriv_ll_[i] = aux_pars_[0] * (y_data[i] * std::exp(-location_par[i]) - 1.);
				}
			}
			else {
				Log::REFatal("CalcFirstDerivLogLik: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
		}

		/*!
		* \brief Calculate the second derivative of the negative (!) log-likelihood with respect to the location parameter
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param location_par Location parameter (random plus fixed effects)
		* \param num_data Number of data points
		*/
		void CalcSecondDerivNegLogLik(const double* y_data, const int* y_data_int,
			const double* location_par, const data_size_t num_data) {
			if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					double dnorm = normalPDF(location_par[i]);
					double pnorm = normalCDF(location_par[i]);
					if (y_data_int[i] == 0) {
						double dnorm_frac_one_min_pnorm = dnorm / (1. - pnorm);
						second_deriv_neg_ll_[i] = -dnorm_frac_one_min_pnorm * (location_par[i] - dnorm_frac_one_min_pnorm);
					}
					else {
						double dnorm_frac_pnorm = dnorm / pnorm;
						second_deriv_neg_ll_[i] = dnorm_frac_pnorm * (location_par[i] + dnorm_frac_pnorm);
					}
				}
			}
			else if (likelihood_type_ == "bernoulli_logit") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					double exp_loc_i = std::exp(location_par[i]);
					second_deriv_neg_ll_[i] = exp_loc_i * std::pow(1. + exp_loc_i, -2);
				}
			}
			else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					second_deriv_neg_ll_[i] = std::exp(location_par[i]);
				}
			}
			else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					second_deriv_neg_ll_[i] = aux_pars_[0] * y_data[i] * std::exp(-location_par[i]);
				}
			}
			else {
				Log::REFatal("CalcSecondDerivNegLogLik: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
		}

		/*!
		* \brief Calculate the third derivative of the log-likelihood with respect to the location parameter
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param location_par Location parameter (random plus fixed effects)
		* \param num_data Number of data points
		* \param[out] third_deriv Third derivative of the log-likelihood with respect to the location parameter. Need to pre-allocate memory of size num_data
		*/
		void CalcThirdDerivLogLik(const double* y_data, const int* y_data_int,
			const double* location_par, const data_size_t num_data, double* third_deriv) {
			if (likelihood_type_ == "bernoulli_probit") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
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
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					double exp_loc_i = std::exp(location_par[i]);
					third_deriv[i] = -exp_loc_i * (1. - exp_loc_i) * std::pow(1 + exp_loc_i, -3);
				}
			}
			else if (likelihood_type_ == "poisson") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					third_deriv[i] = -std::exp(location_par[i]);
				}
			}
			else if (likelihood_type_ == "gamma") {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					third_deriv[i] = aux_pars_[0] * y_data[i] * std::exp(-location_par[i]);
				}
			}
			else {
				Log::REFatal("CalcThirdDerivLogLik: Likelihood of type '%s' is not supported.", likelihood_type_.c_str());
			}
		}

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
			else if (likelihood_type_ == "poisson") {
				return std::exp(value);
			}
			else if (likelihood_type_ == "gamma") {
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
			else if (likelihood_type_ == "poisson") {
				return 1.;
			}
			else if (likelihood_type_ == "gamma") {
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
			else if (likelihood_type_ == "poisson") {
				return 0.;
			}
			else if (likelihood_type_ == "gamma") {
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
		template <class T_mat_1,  typename std::enable_if <std::is_same<sp_mat_t, T_mat_1>::value ||
			std::is_same<sp_mat_rm_t, T_mat_1>::value>::type * = nullptr >
		void CalcChol(T_chol& chol_fact, const T_mat_1& psi) {
			if (!chol_fact_pattern_analyzed_) {
				chol_fact.analyzePattern(psi);
				chol_fact_pattern_analyzed_ = true;
			}
			chol_fact.factorize(psi);
		}
		template <class T_mat_1, typename std::enable_if <std::is_same<den_mat_t, T_mat_1>::value>::type * = nullptr  >
		void CalcChol(T_chol& chol_fact, const T_mat_1& psi) {
			chol_fact.compute(psi);
		}

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood..
		*		Calculations are done using a numerically stable variant based on factorizing ("inverting") B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt).
		*		In the notation of the paper: "Sigma = Z*Sigma*Z^T" and "Z = Id".
		*		This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param ZSigmaZt Covariance matrix of latent random effect
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const std::shared_ptr<T_mat> ZSigmaZt,
			double& approx_marginal_ll) {
			// Initialize variables
			if (!mode_initialized_) {
				InitializeModeAvec();
			}
			else {
				mode_previous_value_ = mode_;
				a_vec_previous_value_ = a_vec_;
			}
			bool no_fixed_effects = (fixed_effects == nullptr);
			vec_t location_par;
			// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
			if (no_fixed_effects) {
				approx_marginal_ll = -0.5 * (a_vec_.dot(mode_)) + LogLikelihood(y_data, y_data_int, mode_.data(), num_data);
			}
			else {
				location_par = vec_t(num_data);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] = mode_[i] + fixed_effects[i];
				}
				approx_marginal_ll = -0.5 * (a_vec_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
			}
			double approx_marginal_ll_new = approx_marginal_ll;
			vec_t rhs, v_aux;//auxiliary variables
			vec_t Wsqrt(num_data);//diagonal matrix with square root of negative second derivatives on the diagonal (sqrt of negative Hessian of log-likelihood)
			T_mat Id_plus_Wsqrt_ZSigmaZt_Wsqrt(num_data, num_data);
			// Start finding mode 
			int it;
			bool terminate_optim = false;
			bool has_NA_or_Inf = false;
			for (it = 0; it < MAXIT_MODE_NEWTON_; ++it) {
				// Calculate first and second derivative of log-likelihood
				if (no_fixed_effects) {
					CalcFirstDerivLogLik(y_data, y_data_int, mode_.data(), num_data);
					CalcSecondDerivNegLogLik(y_data, y_data_int, mode_.data(), num_data);
				}
				else {
					CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);
					CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				}
				// Calculate Cholesky factor of matrix B = Id + Wsqrt * Z*Sigma*Zt * Wsqrt
				Wsqrt.array() = second_deriv_neg_ll_.array().sqrt();
				Id_plus_Wsqrt_ZSigmaZt_Wsqrt.setIdentity();
				Id_plus_Wsqrt_ZSigmaZt_Wsqrt += (Wsqrt.asDiagonal() * (*ZSigmaZt) * Wsqrt.asDiagonal());
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_Wsqrt_ZSigmaZt_Wsqrt);
				// Update mode and a_vec_
				rhs.array() = second_deriv_neg_ll_.array() * mode_.array() + first_deriv_ll_.array();
				v_aux = Wsqrt.asDiagonal() * (*ZSigmaZt) * rhs;
				a_vec_ = rhs - Wsqrt.asDiagonal() * (chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_.solve(v_aux));
				mode_ = (*ZSigmaZt) * a_vec_;
				// Calculate new objective function
				if (no_fixed_effects) {
					approx_marginal_ll_new = -0.5 * (a_vec_.dot(mode_)) + LogLikelihood(y_data, y_data_int, mode_.data(), num_data);
				}
				else {
					// Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						location_par[i] = mode_[i] + fixed_effects[i];
					}
					approx_marginal_ll_new = -0.5 * (a_vec_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
				}
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
			}
			else {
				if (no_fixed_effects) {
					CalcFirstDerivLogLik(y_data, y_data_int, mode_.data(), num_data);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
					CalcSecondDerivNegLogLik(y_data, y_data_int, mode_.data(), num_data);
				}
				else {
					CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
					CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				}
				Wsqrt.array() = second_deriv_neg_ll_.array().sqrt();
				Id_plus_Wsqrt_ZSigmaZt_Wsqrt.setIdentity();
				Id_plus_Wsqrt_ZSigmaZt_Wsqrt += (Wsqrt.asDiagonal() * (*ZSigmaZt) * Wsqrt.asDiagonal());
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_Wsqrt_ZSigmaZt_Wsqrt);
				approx_marginal_ll -= ((T_mat)chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_.matrixL()).diagonal().array().log().sum();
				mode_has_been_calculated_ = true;
			}
		}//end FindModePostRandEffCalcMLLStable

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*		Calculations are done on the random effects (b) scale and not the "data scale" (Zb) using 
		*		a numerically stable variant based on factorizing ("inverting") B = (Id + ZtWZsqrt * Sigma * ZtWZsqrt).
		*		This version is used for the Laplace approximation when there is only one Gaussian process and
		*		there are a lot of multiple observations at the same location, i.e., the dimenion of the random effects b is much smaller than Zb
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param Sigma Covariance matrix of latent random effect
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLOnlyOneGPCalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const std::shared_ptr<T_mat> Sigma,
			const data_size_t * const random_effects_indices_of_data,
			double& approx_marginal_ll) {
			// Initialize variables
			if (!mode_initialized_) {
				InitializeModeAvec();
			}
			else {
				mode_previous_value_ = mode_;
				a_vec_previous_value_ = a_vec_;
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
			approx_marginal_ll = -0.5 * (a_vec_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
			double approx_marginal_ll_new = approx_marginal_ll;
			vec_t diag_sqrt_ZtWZ(num_re_);//sqrt of diagonal matrix ZtWZ
			T_mat Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt(num_re_, num_re_);
			vec_t rhs, v_aux;
			int it;
			bool terminate_optim = false;
			bool has_NA_or_Inf = false;
			for (it = 0; it < MAXIT_MODE_NEWTON_; ++it) {
				// Calculate first and second derivative of log-likelihood
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				// Calculate right hand side for mode update
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv_neg_ll_, diag_sqrt_ZtWZ, true);
				rhs = (diag_sqrt_ZtWZ.array() * mode_.array()).matrix();//rhs = ZtWZ * mode_ + Zt * first_deriv_ll_ for updating mode
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, first_deriv_ll_, rhs, false);
				// Calculate Cholesky factor of matrix B = Id + ZtWZsqrt * Sigma * ZtWZsqrt
				diag_sqrt_ZtWZ.array() = diag_sqrt_ZtWZ.array().sqrt();
				Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt.setIdentity();
				Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt += diag_sqrt_ZtWZ.asDiagonal() * (*Sigma) * diag_sqrt_ZtWZ.asDiagonal();
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt);//this is the bottleneck (for large data and sparse matrices)
				// Update mode and a_vec_
				v_aux = (*Sigma) * rhs;
				v_aux.array() *= diag_sqrt_ZtWZ.array();
				a_vec_ = -chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_.solve(v_aux);
				a_vec_.array() *= diag_sqrt_ZtWZ.array();
				a_vec_.array() += rhs.array();
				mode_ = (*Sigma) * a_vec_;
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
				approx_marginal_ll_new = -0.5 * (a_vec_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
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
			}//end loop for finding mode
			if (it == MAXIT_MODE_NEWTON_) {
				Log::REDebug(NO_CONVERGENCE_WARNING_);
			}
			if (has_NA_or_Inf) {
				approx_marginal_ll = approx_marginal_ll_new;
			}
			else {
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv_neg_ll_, diag_sqrt_ZtWZ, true);
				diag_sqrt_ZtWZ.array() = diag_sqrt_ZtWZ.array().sqrt();
				Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt.setIdentity();
				Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt += diag_sqrt_ZtWZ.asDiagonal() * (*Sigma) * diag_sqrt_ZtWZ.asDiagonal();
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt);
				approx_marginal_ll -= ((T_mat)chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_.matrixL()).diagonal().array().log().sum();
				mode_has_been_calculated_ = true;
			}
		}//end FindModePostRandEffCalcMLLOnlyOneGPCalculationsOnREScale

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*		This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
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
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
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
			}
			else {
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				SigmaI_plus_ZtWZ = SigmaI + Zt * second_deriv_neg_ll_.asDiagonal() * Z;
				SigmaI_plus_ZtWZ.makeCompressed();
				chol_fact_SigmaI_plus_ZtWZ_grouped_.factorize(SigmaI_plus_ZtWZ);
				approx_marginal_ll += -((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_grouped_.matrixL()).diagonal().array().log().sum() + 0.5 * SigmaI.diagonal().array().log().sum();
				mode_has_been_calculated_ = true;
			}
		}//end FindModePostRandEffCalcMLLGroupedRE

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
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
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				// Calculate rhs for mode update
				rhs = - mode_ / sigma2;//right hand side for updating mode
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
			}
			else {
				CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
				CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv_neg_ll_, diag_SigmaI_plus_ZtWZ_, true);
				diag_SigmaI_plus_ZtWZ_.array() += 1. / sigma2;
				approx_marginal_ll -= 0.5 * diag_SigmaI_plus_ZtWZ_.array().log().sum() + 0.5 * num_re_ * std::log(sigma2);
				mode_has_been_calculated_ = true;
			}
		}//end FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale

		/*!
		* \brief Find the mode of the posterior of the latent random effects using Newton's method and calculate the approximative marginal log-likelihood.
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param B Matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation Sigma^-1 = B^T D^-1 B
		* \param[out] approx_marginal_ll Approximate marginal log-likelihood evaluated at the mode
		*/
		void FindModePostRandEffCalcMLLVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			double& approx_marginal_ll) {
			// Initialize variables
			if (!mode_initialized_) {
				InitializeModeAvec();
			}
			else {
				mode_previous_value_ = mode_;
			}
			bool no_fixed_effects = (fixed_effects == nullptr);
			sp_mat_t SigmaI = B.transpose() * D_inv * B;
			vec_t location_par;//location parameter = mode of random effects + fixed effects
			sp_mat_t SigmaI_plus_W;
			vec_t rhs, B_mode;
			// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
			B_mode = B * mode_;
			if (no_fixed_effects) {
				approx_marginal_ll = -0.5 * (B_mode.dot(D_inv * B_mode)) + LogLikelihood(y_data, y_data_int, mode_.data(), num_data);
			}
			else {
				location_par = vec_t(num_data);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] = mode_[i] + fixed_effects[i];
				}
				approx_marginal_ll = -0.5 * (B_mode.dot(D_inv * B_mode)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
			}
			double approx_marginal_ll_new = approx_marginal_ll;
			// Start finding mode 
			int it;
			bool terminate_optim = false;
			bool has_NA_or_Inf = false;
			for (it = 0; it < MAXIT_MODE_NEWTON_; ++it) {
				// Calculate first and second derivative of log-likelihood
				if (no_fixed_effects) {
					CalcFirstDerivLogLik(y_data, y_data_int, mode_.data(), num_data);
					CalcSecondDerivNegLogLik(y_data, y_data_int, mode_.data(), num_data);
				}
				else {
					CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);
					CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				}
				// Calculate Cholesky factor and update mode
				rhs.array() = second_deriv_neg_ll_.array() * mode_.array() + first_deriv_ll_.array();//right hand side for updating mode
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
				mode_ = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(rhs);
				// Calculate new objective function
				B_mode = B * mode_;
				if (no_fixed_effects) {
					approx_marginal_ll_new = -0.5 * (B_mode.dot(D_inv * B_mode)) + LogLikelihood(y_data, y_data_int, mode_.data(), num_data);
				}
				else {
					// Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data; ++i) {
						location_par[i] = mode_[i] + fixed_effects[i];
					}
					approx_marginal_ll_new = -0.5 * (B_mode.dot(D_inv * B_mode)) + LogLikelihood(y_data, y_data_int, location_par.data(), num_data);
				}
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
			}
			else {
				if (no_fixed_effects) {
					CalcFirstDerivLogLik(y_data, y_data_int, mode_.data(), num_data);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
					CalcSecondDerivNegLogLik(y_data, y_data_int, mode_.data(), num_data);
				}
				else {
					CalcFirstDerivLogLik(y_data, y_data_int, location_par.data(), num_data);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
					CalcSecondDerivNegLogLik(y_data, y_data_int, location_par.data(), num_data);
				}
				SigmaI_plus_W = SigmaI;
				SigmaI_plus_W.diagonal().array() += second_deriv_neg_ll_.array();
				SigmaI_plus_W.makeCompressed();
				chol_fact_SigmaI_plus_ZtWZ_vecchia_.factorize(SigmaI_plus_W);
				approx_marginal_ll += -((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_vecchia_.matrixL()).diagonal().array().log().sum() + 0.5 * D_inv.diagonal().array().log().sum();
				mode_has_been_calculated_ = true;
			}
		}//end FindModePostRandEffCalcMLLVecchia

		/*!
		* \brief Calculate the gradient of the negative Laplace approximated marginal log-likelihood wrt covariance parameters, fixed effects, or linear regression coefficients
		*		Calculations are done using a numerically stable variant based on factorizing ("inverting") B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt).
		*		In the notation of the paper: "Sigma = Z*Sigma*Z^T" and "Z = Id".
		*		This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param ZSigmaZt Covariance matrix of latent random effect
		* \param re_comps_cluster_i Vector with different random effects components. We pass the component pointers to save memory in order to avoid passing a large collection of gardient covariance matrices in memory//TODO: better way than passing this? (relying on all gradients in a vector can lead to large memory consumption)
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters are calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F are calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void CalcGradNegMargLikelihoodLAApproxStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const std::shared_ptr<T_mat> ZSigmaZt,
			const std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
			bool calc_cov_grad,
			bool calc_F_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			bool calc_mode) {
			if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * ZSigmaZt * Wsqrt) at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLStable(y_data, y_data_int, fixed_effects, num_data, ZSigmaZt, mll);
			}
			else {
				CHECK(mode_has_been_calculated_);
			}
			// Initialize variables
			bool no_fixed_effects = (fixed_effects == nullptr);
			vec_t location_par;//location parameter = mode of random effects + fixed effects
			T_mat L_inv_Wsqrt(num_data, num_data);//diagonal matrix with square root of negative second derivatives on the diagonal (sqrt of negative Hessian of log-likelihood)
			L_inv_Wsqrt.setIdentity();
			L_inv_Wsqrt.diagonal().array() = second_deriv_neg_ll_.array().sqrt();
			vec_t third_deriv(num_data);//vector of third derivatives of log-likelihood
			if (no_fixed_effects) {
				CalcThirdDerivLogLik(y_data, y_data_int, mode_.data(), num_data, third_deriv.data());
			}
			else {
				location_par = vec_t(num_data);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] = mode_[i] + fixed_effects[i];
				}
				CalcThirdDerivLogLik(y_data, y_data_int, location_par.data(), num_data, third_deriv.data());
			}
			TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, L_inv_Wsqrt, L_inv_Wsqrt, false);//L_inv_Wsqrt = L\Wsqrt
			T_mat L_inv_Wsqrt_ZSigmaZt = L_inv_Wsqrt * (*ZSigmaZt);
			// Calculate gradient of approx. marginal log-likelihood wrt the mode
			//		Note: use (i) (Sigma^-1 + W)^-1 = Sigma - Sigma*(W^-1 + Sigma)^-1*Sigma = ZSigmaZt - L_inv_Wsqrt_ZSigmaZt^T*L_inv_Wsqrt_ZSigmaZt and (ii) "Z=Id"N
			T_mat L_inv_Wsqrt_ZSigmaZt_sqr = L_inv_Wsqrt_ZSigmaZt.cwiseProduct(L_inv_Wsqrt_ZSigmaZt);
			vec_t ZSigmaZtI_plus_W_inv_diag = (*ZSigmaZt).diagonal() - L_inv_Wsqrt_ZSigmaZt_sqr.transpose() * vec_t::Ones(L_inv_Wsqrt_ZSigmaZt_sqr.rows());// diagonal of (ZSigmaZt^-1 + W) ^ -1
			vec_t d_mll_d_mode = (-0.5 * ZSigmaZtI_plus_W_inv_diag.array() * third_deriv.array()).matrix();// gradient of approx. marginal likelihood wrt the mode and thus also F here
			// calculate gradient wrt covariance parameters
			if (calc_cov_grad) {
				T_mat WI_plus_Sigma_inv;//WI_plus_Sigma_inv = Wsqrt * L^T\(L\Wsqrt) = (W^-1 + Sigma)^-1
				vec_t d_mode_d_par, SigmaDeriv_first_deriv_ll;
				int par_count = 0;
				double explicit_derivative;
				for (int j = 0; j < (int)re_comps_cluster_i.size(); ++j) {
					for (int ipar = 0; ipar < re_comps_cluster_i[j]->NumCovPar(); ++ipar) {
						std::shared_ptr<T_mat> SigmaDeriv = re_comps_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 1.);
						if (ipar == 0) {
							WI_plus_Sigma_inv = *SigmaDeriv;
							CalcLtLGivenSparsityPattern<T_mat>(L_inv_Wsqrt, WI_plus_Sigma_inv);
							//TODO (low-prio): calculate WI_plus_Sigma_inv only once for all relevant non-zero entries as in Gaussian case (see 'CalcPsiInv')
							//					This is only relevant for multiple random effects and/or GPs
						}
						// calculate explicit derivative of approx. mariginal log-likelihood
						explicit_derivative = -0.5 * (double)(a_vec_.transpose() * (*SigmaDeriv) * a_vec_) + 0.5 * (WI_plus_Sigma_inv.cwiseProduct(*SigmaDeriv)).sum();
						// calculate implicit derivative (through mode) of approx. mariginal log-likelihood
						SigmaDeriv_first_deriv_ll = (*SigmaDeriv) * first_deriv_ll_;//auxiliary variable for caclulating d_mode_d_par
						d_mode_d_par = SigmaDeriv_first_deriv_ll;//derivative of mode wrt to a covariance parameter
						d_mode_d_par -= ((*ZSigmaZt) * (L_inv_Wsqrt.transpose() * (L_inv_Wsqrt * SigmaDeriv_first_deriv_ll)));
						cov_grad[par_count] = explicit_derivative + d_mll_d_mode.dot(d_mode_d_par);
						par_count++;
					}
				}
			}//end calc_cov_grad
			// calculate gradient wrt fixed effects
			if (calc_F_grad) {
				vec_t L_inv_Wsqrt_ZSigmaZt_d_mll_d_mode = L_inv_Wsqrt_ZSigmaZt * d_mll_d_mode;// for implicit derivative
				vec_t ZSigmaZtI_plus_W_inv_d_mll_d_mode = (*ZSigmaZt) * d_mll_d_mode - L_inv_Wsqrt_ZSigmaZt.transpose() * L_inv_Wsqrt_ZSigmaZt_d_mll_d_mode;
				vec_t d_mll_d_F_implicit = (ZSigmaZtI_plus_W_inv_d_mll_d_mode.array() * second_deriv_neg_ll_.array()).matrix();// implicit derivative
				fixed_effect_grad = -first_deriv_ll_ + d_mll_d_mode - d_mll_d_F_implicit;
			}//end calc_F_grad
		}//end CalcGradNegMargLikelihoodLAApproxStable

		/*!
		* \brief Calculate the gradient of the negative Laplace approximated marginal log-likelihood wrt covariance parameters, fixed effects, or linear regression coefficients
		*		Calculations are done on the random effects (b) scale and not the "data scale" (Zb) using
		*		a numerically stable variant based on factorizing ("inverting") B = (Id + ZtWZsqrt * Sigma * ZtWZsqrt).
		*		This version is used for the Laplace approximation when there is only one Gaussian process and
		*		there are a lot of multiple observations at the same location, i.e., the dimenion of the random effects b is much smaller than Zb
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param Sigma Covariance matrix of latent random effect
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param re_comps_cluster_i Vector with different random effects components. We pass the component pointers to save memory in order to avoid passing a large collection of gardient covariance matrices in memory//TODO: better way than passing this? (relying on all gradients in a vector can lead to large memory consumption)
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters are calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F are calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void CalcGradNegMargLikelihoodLAApproxOnlyOneGPCalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const std::shared_ptr<T_mat> Sigma,
			const data_size_t* const random_effects_indices_of_data,
			const std::vector<std::shared_ptr<RECompBase<T_mat>>> & re_comps_cluster_i,
			bool calc_cov_grad,
			bool calc_F_grad,
			double* cov_grad,
			vec_t & fixed_effect_grad,
			bool calc_mode) {
			CHECK(re_comps_cluster_i.size() == 1);
			if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * ZSigmaZt * Wsqrt) at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLOnlyOneGPCalculationsOnREScale(y_data, y_data_int, fixed_effects, num_data,
					Sigma, random_effects_indices_of_data, mll);
			}
			else {
				CHECK(mode_has_been_calculated_);
			}
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
			// Matrix ZtWZsqrt
			vec_t diag_ZtWZ;
			CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv_neg_ll_, diag_ZtWZ, true);
			T_mat L_inv_ZtWZsqrt(num_re_, num_re_);//diagonal matrix with square root of diagonal of ZtWZ
			L_inv_ZtWZsqrt.setIdentity();
			L_inv_ZtWZsqrt.diagonal().array() = diag_ZtWZ.array().sqrt();
			vec_t third_deriv(num_data);//vector of third derivatives of log-likelihood
			CalcThirdDerivLogLik(y_data, y_data_int, location_par.data(), num_data, third_deriv.data());
			vec_t diag_ZtThirdDerivZ;
			CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, third_deriv, diag_ZtThirdDerivZ, true);
			TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, L_inv_ZtWZsqrt, L_inv_ZtWZsqrt, false);//L_inv_ZtWZsqrt = L\ZtWZsqrt
			T_mat L_inv_ZtWZsqrt_Sigma = L_inv_ZtWZsqrt * (*Sigma);
			//Log::REInfo("CalcGradNegMargLikelihoodLAApproxOnlyOneGPCalculationsOnREScale: L_inv_ZtWZsqrt: number non zeros = %d", GetNumberNonZeros<T_mat>(L_inv_ZtWZsqrt));//Only for debugging
			//Log::REInfo("CalcGradNegMargLikelihoodLAApproxOnlyOneGPCalculationsOnREScale: L_inv_ZtWZsqrt_Sigma: number non zeros = %d", GetNumberNonZeros<T_mat>(L_inv_ZtWZsqrt_Sigma));//Only for debugging
			// Calculate gradient of approx. marginal log-likelihood wrt the mode
			//		Note: use (i) (Sigma^-1 + W)^-1 = Sigma - Sigma*(W^-1 + Sigma)^-1*Sigma = ZSigmaZt - L_inv_ZtWZsqrt_Sigma^T*L_inv_ZtWZsqrt_Sigma
			T_mat L_inv_ZtWZsqrt_Sigma_sqr = L_inv_ZtWZsqrt_Sigma.cwiseProduct(L_inv_ZtWZsqrt_Sigma);
			vec_t SigmaI_plus_ZtWZ_inv_diag = (*Sigma).diagonal() - L_inv_ZtWZsqrt_Sigma_sqr.transpose() * vec_t::Ones(L_inv_ZtWZsqrt_Sigma_sqr.rows());// diagonal of (Sigma^-1 + ZtWZ) ^ -1
			vec_t d_mll_d_mode = (-0.5 * SigmaI_plus_ZtWZ_inv_diag.array() * diag_ZtThirdDerivZ.array()).matrix();// gradient of approx. marginal likelihood wrt the mode
			// calculate gradient wrt covariance parameters
			if (calc_cov_grad) {
				vec_t ZtFirstDeriv;
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, first_deriv_ll_, ZtFirstDeriv, true);
				T_mat ZtWZI_Sigma_inv;//ZtWZI_Sigma_inv = ZtWZsqrt * L^T\(L\ZtWZsqrt) = ((ZtWZ)^-1 + Sigma)^-1
				vec_t d_mode_d_par, SigmaDeriv_ZtFirstDeriv;
				int par_count = 0;
				double explicit_derivative;
				for (int j = 0; j < (int)re_comps_cluster_i.size(); ++j) {
					for (int ipar = 0; ipar < re_comps_cluster_i[j]->NumCovPar(); ++ipar) {
						std::shared_ptr<T_mat> SigmaDeriv = re_comps_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 1.);
						if (ipar == 0) {
							ZtWZI_Sigma_inv = *SigmaDeriv;
							CalcLtLGivenSparsityPattern<T_mat>(L_inv_ZtWZsqrt, ZtWZI_Sigma_inv);
							//TODO (low-prio): calculate ZtWZI_Sigma_inv only once for all relevant non-zero entries as in Gaussian case (see 'CalcPsiInv')
							//					This is only relevant for multiple random effects and/or GPs
						}
						// calculate explicit derivative of approx. mariginal log-likelihood
						explicit_derivative = -0.5 * (double)(a_vec_.transpose() * (*SigmaDeriv) * a_vec_) +
							0.5 * (ZtWZI_Sigma_inv.cwiseProduct(*SigmaDeriv)).sum();
						// calculate implicit derivative (through mode) of approx. mariginal log-likelihood
						SigmaDeriv_ZtFirstDeriv = (*SigmaDeriv) * ZtFirstDeriv;//auxiliary variable for caclulating d_mode_d_par
						d_mode_d_par = SigmaDeriv_ZtFirstDeriv;//derivative of mode wrt to a covariance parameter
						d_mode_d_par -= ((*Sigma) * (L_inv_ZtWZsqrt.transpose() * (L_inv_ZtWZsqrt * SigmaDeriv_ZtFirstDeriv)));
						cov_grad[par_count] = explicit_derivative + d_mll_d_mode.dot(d_mode_d_par);
						par_count++;
					}
				}
			}//end calc_cov_grad
			// calculate gradient wrt fixed effects
			if (calc_F_grad) {
				vec_t L_inv_ZtWZsqrt_Sigma_d_mll_d_mode = L_inv_ZtWZsqrt_Sigma * d_mll_d_mode;// for implicit derivative
				vec_t SigmaI_plus_ZtWZ_inv_d_mll_d_mode = (*Sigma) * d_mll_d_mode - L_inv_ZtWZsqrt_Sigma.transpose() * L_inv_ZtWZsqrt_Sigma_d_mll_d_mode;
				fixed_effect_grad = -first_deriv_ll_;
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					fixed_effect_grad[i] += -0.5 * third_deriv[i] * SigmaI_plus_ZtWZ_inv_diag[random_effects_indices_of_data[i]] -
						second_deriv_neg_ll_[i] * SigmaI_plus_ZtWZ_inv_d_mll_d_mode[random_effects_indices_of_data[i]];
				}
			}//end calc_F_grad
		}//end CalcGradNegMargLikelihoodLAApproxOnlyOneGPCalculationsOnREScale

		/*!
		* \brief Calculate the gradient of the negative Laplace approximated marginal log-likelihood wrt covariance parameters, fixed effects, or linear regression coefficients
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*		This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param SigmaI Inverse covariance matrix of latent random effect. Currently, this needs to be a diagonal matrix
		* \param Zt Transpose Z^T of random effect design matrix that relates latent random effects to observations/likelihoods
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters are calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F are calculated
		* \param[out] cov_grad Gradient wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void CalcGradNegMargLikelihoodLAApproxGroupedRE(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const sp_mat_t& SigmaI,
			const sp_mat_t& Zt,
			std::vector<data_size_t> cum_num_rand_eff_cluster_i,
			bool calc_cov_grad,
			bool calc_F_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			bool calc_mode) {
			int num_REs = (int)SigmaI.cols();//number of random effect realizations
			int num_comps = (int)cum_num_rand_eff_cluster_i.size() - 1;//number of different random effect components
			if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLGroupedRE(y_data, y_data_int, fixed_effects, num_data, SigmaI, Zt, mll);
			}
			else {
				CHECK(mode_has_been_calculated_);
			}
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
			CalcThirdDerivLogLik(y_data, y_data_int, location_par.data(), num_data, third_deriv.data());
			// Calculate (Sigma^-1 + Zt*W*Z)^-1
			sp_mat_t L_inv(num_REs, num_REs);
			L_inv.setIdentity();
			if (chol_fact_SigmaI_plus_ZtWZ_grouped_.permutationP().size() > 0) {//Permutation is only used when having an ordering
				L_inv = chol_fact_SigmaI_plus_ZtWZ_grouped_.permutationP() * L_inv;
			}
			sp_mat_t L = chol_fact_SigmaI_plus_ZtWZ_grouped_.matrixL();
			TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(L, L_inv, L_inv, false);
			L.resize(0, 0);
			sp_mat_t SigmaI_plus_ZtWZ_inv = L_inv.transpose() * L_inv;
			// calculate gradient of approx. marginal likeligood wrt the mode
			//Note: the calculation of d_mll_d_mode is the bottleneck of this function (corresponding lines below are indicated with * and, in particular, **)
			vec_t d_mll_d_mode(num_REs);
			sp_mat_t Zt_third_deriv = Zt * third_deriv.asDiagonal();//every column of Z multiplied elementwise by third_deriv
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_REs; ++i) {
				vec_t diag_d_W_d_mode_i = Zt_third_deriv.row(i);//*can be slow
				//calculate Z^T * diag(diag_d_W_d_mode_i) * Z = Z^T * diag(Z.col(i) * third_deriv) * Z
				sp_mat_t Zt_d_W_d_mode_i_Z = (Zt * diag_d_W_d_mode_i.asDiagonal() * Z).pruned();//**can be very slow. Note that this is also slow when the middle diagonal matrix is a pruned sparse matrix
				////Variant 2: slower
				//sp_mat_t Zt_third_deriv_diag = sp_mat_t(((vec_t)Zt_third_deriv.row(i)).asDiagonal());
				//sp_mat_t Zt_d_W_d_mode_i_Z = Zt * Zt_third_deriv_diag * Z;//= Z^T * diag(diag_d_W_d_mode_i) * Z = Z^T * diag(Z.col(i) * third_deriv) * Z
				////Variant 3: slower
				//vec_t Z_i = Z.col(i);// column number i of Z
				//vec_t diag_d_W_d_mode_i = (Z_i.array() * third_deriv.array()).matrix();//diagonal of derivative of matrix W wrt random effect number i
				//sp_mat_t Zt_d_W_d_mode_i_Z = Zt * diag_d_W_d_mode_i.asDiagonal() * Z;//= Z^T * diag(diag_d_W_d_mode_i) * Z
				d_mll_d_mode[i] = -0.5 * (Zt_d_W_d_mode_i_Z.cwiseProduct(SigmaI_plus_ZtWZ_inv)).sum();
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
					std::vector<Triplet_t> triplets;//for constructing I_j
					triplets.reserve(cum_num_rand_eff_cluster_i[j + 1] - cum_num_rand_eff_cluster_i[j]);
					explicit_derivative = 0.;
					for (int i = cum_num_rand_eff_cluster_i[j]; i < cum_num_rand_eff_cluster_i[j + 1]; ++i) {
						triplets.emplace_back(i, i, 1.);
						explicit_derivative += SigmaI_mode[i] * mode_[i];
					}
					// Altervative version using parallelization (not faster)
					//#pragma omp parallel
					//					{
					//						std::vector<Triplet_t> triplets_private;
					//						//triplets_private.reserve(cum_num_rand_eff_cluster_i[num_comps]);
					//#pragma omp for nowait reduction(+:explicit_derivative)
					//						for (int i = cum_num_rand_eff_cluster_i[j]; i < cum_num_rand_eff_cluster_i[j + 1]; ++i) {
					//							triplets_private.emplace_back(i, i, 1.);
					//							explicit_derivative += SigmaI_mode[i] * mode_[i];
					//						}
					//#pragma omp critical
					//						triplets.insert(triplets.end(), triplets_private.begin(), triplets_private.end());
					//					}
					//#pragma omp parallel for schedule(static) reduction(+:explicit_derivative)
					//					for (int i = cum_num_rand_eff_cluster_i[j]; i < cum_num_rand_eff_cluster_i[j + 1]; ++i) {
					//						explicit_derivative += SigmaI_mode[i] * mode_[i];
					//					}
					explicit_derivative *= -0.5;
					I_j.setFromTriplets(triplets.begin(), triplets.end());
					I_j_ZtWZ = I_j * ZtWZ;
					explicit_derivative += 0.5 * (SigmaI_plus_ZtWZ_inv.cwiseProduct(I_j_ZtWZ)).sum();
					// calculate implicit derivative (through mode) of approx. mariginal log-likelihood
					d_mode_d_par = SigmaI_plus_ZtWZ_inv * (I_j * (Zt * first_deriv_ll_));
					cov_grad[j] = explicit_derivative + d_mll_d_mode.dot(d_mode_d_par);
				}
			}//end calc_cov_grad
			// calculate gradient wrt fixed effects
			if (calc_F_grad) {
				vec_t d_detmll_d_F(num_data);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_data; ++i) {
					sp_mat_t zi_zit = Zt.col(i) * Z.row(i);//=Z.row(i) * (Z.row(i)).transpose()
					d_detmll_d_F[i] = -0.5 * third_deriv[i] * (SigmaI_plus_ZtWZ_inv.cwiseProduct(zi_zit)).sum();
				}
				vec_t d_mll_d_modeT_SigmaI_plus_ZtWZ_inv_Zt_W = ((d_mll_d_mode.transpose() * SigmaI_plus_ZtWZ_inv) * Zt) * second_deriv_neg_ll_.asDiagonal();
				fixed_effect_grad = -first_deriv_ll_ + d_detmll_d_F - d_mll_d_modeT_SigmaI_plus_ZtWZ_inv_Zt_W;
			}//end calc_F_grad
		}//end CalcGradNegMargLikelihoodLAApproxGroupedRE

		/*!
		* \brief Calculate the gradient of the negative Laplace approximated marginal log-likelihood wrt covariance parameters, fixed effects, or linear regression coefficients
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param sigma2 Variance of random effects
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters are calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F are calculated
		* \param[out] cov_grad Gradient wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void CalcGradNegMargLikelihoodLAApproxOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const double sigma2,
			const data_size_t* const random_effects_indices_of_data,
			bool calc_cov_grad,
			bool calc_F_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			bool calc_mode) {
			if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(y_data, y_data_int, fixed_effects, num_data,
					sigma2, random_effects_indices_of_data, mll);
			}
			else {
				CHECK(mode_has_been_calculated_);
			}
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
			CalcThirdDerivLogLik(y_data, y_data_int, location_par.data(), num_data, third_deriv.data());
			// calculate gradient of approx. marginal likeligood wrt the mode
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
		}//end CalcGradNegMargLikelihoodLAApproxOnlyOneGroupedRECalculationsOnREScale

		/*!
		* \brief Calculate the gradient of the negative Laplace approximated marginal log-likelihood wrt covariance parameters, fixed effects, or linear regression coefficients
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param B Matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
		* \param D_inv Diagonal matrix D^-1 in Vecchia approximation Sigma^-1 = B^T D^-1 B
		* \param B_grad Derivatives of matrices B ( = derivative of matrix -A) for Vecchia approximation
		* \param D_grad Derivatives of matrices D for Vecchia approximation
		* \param calc_cov_grad If true, the gradient wrt the covariance parameters are calculated
		* \param calc_F_grad If true, the gradient wrt the fixed effects mean function F are calculated
		* \param[out] cov_grad Gradient of approximate marginal log-likelihood wrt covariance parameters (needs to be preallocated of size num_cov_par)
		* \param[out] fixed_effect_grad Gradient of approximate marginal log-likelihood wrt fixed effects F (note: this is passed as a Eigen vector in order to avoid the need for copying)
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		* \param num_comps_total Total number of random effect components ( = number of GPs)
		*/
		void CalcGradNegMargLikelihoodLAApproxVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const sp_mat_t& B,
			const sp_mat_t& D_inv,
			const std::vector<sp_mat_t>& B_grad,
			const std::vector<sp_mat_t>& D_grad,
			bool calc_cov_grad,
			bool calc_F_grad,
			double* cov_grad,
			vec_t& fixed_effect_grad,
			bool calc_mode,
			int num_comps_total) {
			if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
				double mll;//approximate marginal likelihood. This is a by-product that is not used here.
				FindModePostRandEffCalcMLLVecchia(y_data, y_data_int, fixed_effects, num_data, B, D_inv, mll);
			}
			else {
				CHECK(mode_has_been_calculated_);
			}
			// Initialize variables
			bool no_fixed_effects = (fixed_effects == nullptr);
			vec_t location_par;//location parameter = mode of random effects + fixed effects
			vec_t third_deriv(num_data);//vector of third derivatives of log-likelihood
			if (no_fixed_effects) {
				CalcThirdDerivLogLik(y_data, y_data_int, mode_.data(), num_data, third_deriv.data());
			}
			else {
				location_par = vec_t(num_data);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data; ++i) {
					location_par[i] = mode_[i] + fixed_effects[i];
				}
				CalcThirdDerivLogLik(y_data, y_data_int, location_par.data(), num_data, third_deriv.data());
			}
			// Calculate (Sigma^-1 + W)^-1
			sp_mat_t L_inv(num_data, num_data);
			L_inv.setIdentity();
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, L_inv, L_inv, false);
			// Calculate gradient wrt covariance parameters
			if (calc_cov_grad) {
				vec_t d_mll_d_mode, d_mode_d_par;
				double explicit_derivative;
				int num_par = (int)B_grad.size();
				sp_mat_t SigmaI_plus_W_inv,  SigmaI_deriv,  BgradT_Dinv_B, Bt_Dinv_Bgrad;
				sp_mat_t D_inv_B = D_inv * B;
				for (int j = 0; j < num_par; ++j) {
					// Calculate SigmaI_deriv
					if (num_comps_total == 1 && j == 0) {
						SigmaI_deriv = - B.transpose() * D_inv_B;//SigmaI_deriv = -SigmaI for variance paramters if there is only one GP
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
						CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_W_inv);
						d_mll_d_mode = -0.5 * (SigmaI_plus_W_inv.diagonal().array() * third_deriv.array()).matrix();
					}//end if j == 0
					d_mode_d_par = -(L_inv.transpose() * (L_inv * (SigmaI_deriv * mode_)));//derivative of mode wrt to a covariance parameter
					explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv * mode_) + (SigmaI_deriv.cwiseProduct(SigmaI_plus_W_inv)).sum());
					if (num_comps_total == 1 && j == 0) {
						explicit_derivative += 0.5 * num_data;
					}
					else {
						explicit_derivative += 0.5 * (D_inv.diagonal().array() * D_grad[j].diagonal().array()).sum();
					}
					cov_grad[j] = explicit_derivative + d_mll_d_mode.dot(d_mode_d_par);
				}
			}//end calc_cov_grad
			// Calculate gradient wrt fixed effects
			if (calc_F_grad) {
				sp_mat_t L_inv_sqr = L_inv.cwiseProduct(L_inv);
				vec_t SigmaI_plus_W_inv_diag = L_inv_sqr.transpose() * vec_t::Ones(L_inv_sqr.rows());// diagonal of (Sigma^-1 + W) ^ -1
				vec_t d_mll_d_mode = (-0.5 * SigmaI_plus_W_inv_diag.array() * third_deriv.array()).matrix();// gradient of approx. marginal likelihood wrt the mode and thus also F here
				vec_t SigmaI_plus_W_inv_d_mll_d_mode = L_inv.transpose() * (L_inv * d_mll_d_mode);
				vec_t d_mll_d_F_implicit = -(SigmaI_plus_W_inv_d_mll_d_mode.array() * second_deriv_neg_ll_.array()).matrix();// implicit derivative
				fixed_effect_grad = -first_deriv_ll_ + d_mll_d_mode + d_mll_d_F_implicit;
			}//end calc_F_grad
		}//end CalcGradNegMargLikelihoodLAApproxVecchia

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done using a numerically stable variant based on factorizing ("inverting") B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt).
		*		In the notation of the paper: "Sigma = Z*Sigma*Z^T" and "Z = Id".
		*		This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param ZSigmaZt Covariance matrix of latent random effect
		* \param Cross_Cov Cross covariance matrix between predicted and obsreved random effects ("=Cov(y_p,y)")
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void PredictLAApproxStable(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
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
				FindModePostRandEffCalcMLLStable(y_data, y_data_int, fixed_effects, num_data, ZSigmaZt, mll);
			}
			else {
				CHECK(mode_has_been_calculated_);
			}
			pred_mean = Cross_Cov * first_deriv_ll_;
			if (calc_pred_cov || calc_pred_var) {
				sp_mat_t Wsqrt(num_data, num_data);//diagonal matrix with square root of negative second derivatives on the diagonal (sqrt of negative Hessian of log-likelihood)
				Wsqrt.setIdentity();
				Wsqrt.diagonal().array() = second_deriv_neg_ll_.array().sqrt();
				T_mat Maux = Wsqrt * Cross_Cov.transpose();
				TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Maux, Maux, false);
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
		}//end PredictLAApproxStable

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done using a numerically stable variant based on factorizing ("inverting") B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt).
		*		In the notation of the paper: "Sigma = Z*Sigma*Z^T" and "Z = Id".
		*		This version is used for the Laplace approximation when dense matrices are used (e.g. GP models).
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param Sigma Covariance matrix of latent random effect
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param Cross_Cov Cross covariance matrix between predicted and obsreved random effects ("=Cov(y_p,y)")
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void PredictLAApproxOnlyOneGPCalculationsOnREScale(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
			const std::shared_ptr<T_mat> Sigma,
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
				FindModePostRandEffCalcMLLOnlyOneGPCalculationsOnREScale(y_data, y_data_int, fixed_effects,
					num_data, Sigma, random_effects_indices_of_data, mll);
			}
			else {
				CHECK(mode_has_been_calculated_);
			}
			vec_t ZtFirstDeriv;
			CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, first_deriv_ll_, ZtFirstDeriv, true);
			pred_mean = Cross_Cov * ZtFirstDeriv;
			if (calc_pred_cov || calc_pred_var) {
				vec_t diag_ZtWZ;
				CalcZtVGivenIndices(num_data, num_re_, random_effects_indices_of_data, second_deriv_neg_ll_, diag_ZtWZ, true);
				sp_mat_t ZtWZsqrt(num_re_, num_re_);//diagonal matrix with square root of diagonal of ZtWZ
				ZtWZsqrt.setIdentity();
				ZtWZsqrt.diagonal().array() = diag_ZtWZ.array().sqrt();
				T_mat Maux = ZtWZsqrt * Cross_Cov.transpose();
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
		}//end PredictLAApproxOnlyOneGPCalculationsOnREScale

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		NOTE: IT IS ASSUMED THAT SIGMA IS A DIAGONAL MATRIX
		*		This version is used for the Laplace approximation when there are only grouped random effects.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
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
		void PredictLAApproxGroupedRE(const double* y_data,
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
			else {
				CHECK(mode_has_been_calculated_);
			}
			vec_t v_aux = Zt * first_deriv_ll_;
			vec_t v_aux2 = Sigma * v_aux;
			pred_mean = Ztilde * v_aux2;
			if (calc_pred_cov || calc_pred_var) {
				// calculate Maux = L\(Z^T * second_deriv_neg_ll_.asDiagonal() * Cross_Cov^T)
				sp_mat_t Cross_Cov = Ztilde * Sigma * Zt;
				sp_mat_t Maux = Zt * second_deriv_neg_ll_.asDiagonal() * Cross_Cov.transpose();
				TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, Maux, Maux, false);
				if (calc_pred_cov) {
					pred_cov += (T_mat)(Maux.transpose() * Maux); 
					pred_cov -= (T_mat)(Cross_Cov * second_deriv_neg_ll_.asDiagonal() * Cross_Cov.transpose());
				}
				if (calc_pred_var) {
					sp_mat_t Maux3 = Cross_Cov.cwiseProduct(Cross_Cov * second_deriv_neg_ll_.asDiagonal());
					Maux = Maux.cwiseProduct(Maux);
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] += Maux.col(i).sum() - Maux3.row(i).sum();
					}
				}
			}
		}//end PredictLAApproxGroupedRE

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done by directly factorizing ("inverting) (Sigma^-1 + Zt*W*Z).
		*		This version is used for the Laplace approximation when there are only grouped random effects with only one grouping variable.
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
		* \param sigma2 Variance of random effects
		* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
		* \param Cross_Cov Cross covariance matrix between predicted and obsreved random effects ("=Cov(y_p,y)")
		* \param pred_mean[out] Predictive mean
		* \param pred_cov[out] Predictive covariance matrix
		* \param pred_var[out] Predictive variances
		* \param calc_pred_cov If true, predictive covariance matrix is also calculated
		* \param calc_pred_var If true, predictive variances are also calculated
		* \param calc_mode If true, the mode of the random effects posterior is calculated otherwise the values in mode and a_vec_ are used (default=false)
		*/
		void PredictLAApproxOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
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
			else {
				CHECK(mode_has_been_calculated_);
			}
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
		}//end PredictLAApproxOnlyOneGroupedRECalculationsOnREScale

		/*!
		* \brief Make predictions for the (latent) random effects when using the Laplace approximation.
		*		Calculations are done by factorizing ("inverting) (Sigma^-1 + W) where it is assumed that an approximate Cholesky factor
		*		of Sigma^-1 has previously been calculated using a Vecchia approximation.
		*		This version is used for the Laplace approximation when there are only GP random effects and the Vecchia approximation is used.
		*		Caveat: Sigma^-1 + W can be not very sparse
		* \param y_data Response variable data if response variable is continuous
		* \param y_data_int Response variable data if response variable is integer-valued (only one of these two is used)
		* \param fixed_effects Fixed effects component of location parameter
		* \param num_data Number of data points
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
		void PredictLAApproxVecchia(const double* y_data,
			const int* y_data_int,
			const double* fixed_effects,
			const data_size_t num_data,
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
				FindModePostRandEffCalcMLLVecchia(y_data, y_data_int, fixed_effects, num_data, B, D_inv, mll);
			}
			else {
				CHECK(mode_has_been_calculated_);
			}
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
			}//end calc_pred_cov || calc_pred_var
		}//end PredictLAApproxVecchia

		/*!
		* \brief Make predictions for the response variable (label) based on predictions for the mean and variance of the latent random effects
		* \param pred_mean[out] Predictive mean of latent random effects. The Predictive mean for the response variables is written on this
		* \param pred_var[out] Predictive variances of latent random effects. The predicted variance for the response variables is written on this
		* \param predict_var If true, predictive variances are also calculated
		*/
		void PredictResponse(vec_t& pred_mean, vec_t& pred_var, bool predict_var) {
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
		}

		/*!
		* \brief Adaptive GH quadrature to calculate predictive mean of response variable
		* \param latent_mean Predictive mean of latent random effects
		* \param latent_var Predictive variances of latent random effects
		*/
		double RespMeanAdaptiveGHQuadrature(const double latent_mean, const double latent_var) {
			// Find mode of integrand
			double mode_integrand, mode_integrand_last, update;
			mode_integrand = 0.;
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
		}

	private:
		/*! \brief Number of data points */
		data_size_t num_data_;
		/*! \brief Number (dimension) of random effects */
		data_size_t num_re_;
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
		/*! \brief First derivatives of the log-likelihood */
		vec_t first_deriv_ll_;
		/*! \brief Second derivatives of the negative log-likelihood (diagonal of matrix "W") */
		vec_t second_deriv_neg_ll_;
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
		/*! \brief If true, the function 'CheckY' has been called */
		bool normalizing_constant_has_been_calculated_ = false;
		/*! \brief Normalizing constant for likelihoods (not all likelihoods have one) */
		double log_normalizing_constant_;

		/*! \brief Type of likelihood  */
		string_t likelihood_type_ = "gaussian";
		/*! \brief List of supported covariance likelihoods */
		const std::set<string_t> SUPPORTED_LIKELIHOODS_{ "gaussian", "bernoulli_probit", "bernoulli_logit", "poisson", "gamma" };
		/*! \brief Maximal number of iteration done for finding posterior mode with Newton's method */
		int MAXIT_MODE_NEWTON_ = 1000;
		/*! \brief Used for checking convergence in mode finding algorithm (terminate if relative change in Laplace approx. is below this value) */
		double DELTA_REL_CONV_ = 1e-6;
		/*! \brief Additional parameters for likelihoods. For gamma, aux_pars_[0] = shape parameter  */
		std::vector<double> aux_pars_;

		string_t ParseLikelihoodAlias(const string_t& likelihood) {
			if (likelihood == string_t("binary") || likelihood == string_t("bernoulli_probit") || likelihood == string_t("binary_probit")) {
				return "bernoulli_probit";
			}
			else if (likelihood == string_t("gaussian") || likelihood == string_t("regression")) {
				return "gaussian";
			}
			return likelihood;
		}

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

		const char* NA_OR_INF_WARNING_ = "Mode finding algorithm for Laplace approximation: NA or Inf occurred. This is not necessary a problem as it might have been the cause of a too large learning rate which, consequently, might have been decreased by the optimization algorithm ";
		const char* NO_INCREASE_IN_MLL_WARNING_ = "Mode finding algorithm for Laplace approximation: The approximate marginal log-likelihood (=convergence criterion) has decreased and the algorithm has thus been terminated ";
		const char* NO_CONVERGENCE_WARNING_ = "Algorithm for finding mode for Laplace approximation has not converged after the maximal number of iterations ";

	};//end class Likelihood

}  // namespace GPBoost

#endif   // GPB_LIKELIHOODS_
