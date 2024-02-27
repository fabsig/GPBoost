/*!
* Original work Copyright (c) 2017 Microsoft Corporation. All rights reserved.
* Modified work Copyright (c) 2020 Fabio Sigrist. All rights reserved.
* Licensed under the Apache License Version 2.0 See LICENSE file in the project root for license information.
*/
#ifndef LIGHTGBM_OBJECTIVE_FUNCTION_H_
#define LIGHTGBM_OBJECTIVE_FUNCTION_H_

#include <GPBoost/re_model.h>

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>

#include <string>
#include <functional>

using GPBoost::REModel;

namespace LightGBM {
	/*!
	* \brief The interface of Objective Function.
	*/
	class ObjectiveFunction {
	public:
		/*! \brief virtual destructor */
		virtual ~ObjectiveFunction() {}

		/*!
		* \brief Initialize
		* \param metadata Label data
		* \param num_data Number of data
		*/
		virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;

		/*!
		* \brief calculating first order derivative of loss function
		* \param score prediction score in this round
		* \gradients Output gradients
		* \hessians Output hessians
		*/
		virtual void GetGradients(const double* score,
			score_t* gradients, score_t* hessians) const = 0;

		virtual const char* GetName() const = 0;

		virtual bool IsConstantHessian() const { return false; }

		virtual bool IsRenewTreeOutput() const { return false; }

		virtual double RenewTreeOutput(double ori_output, std::function<double(const label_t*, int)>,
			const data_size_t*,
			const data_size_t*,
			data_size_t) const {
			return ori_output;
		}

		virtual double BoostFromScore(int /*class_id*/) const { return 0.0; }

		virtual bool ClassNeedTrain(int /*class_id*/) const { return true; }

		virtual bool SkipEmptyClass() const { return false; }

		virtual int NumModelPerIteration() const { return 1; }

		virtual int NumPredictOneRow() const { return 1; }

		/*! \brief The prediction should be accurate or not. True will disable early stopping for prediction. */
		virtual bool NeedAccuratePrediction() const { return true; }

		/*! \brief Return the number of positive samples. Return 0 if no binary classification tasks.*/
		virtual data_size_t NumPositiveData() const { return 0; }

		virtual void ConvertOutput(const double* input, double* output) const {
			output[0] = input[0];
		}

		virtual std::string ToString() const = 0;

		ObjectiveFunction() = default;
		/*! \brief Disable copy */
		ObjectiveFunction& operator=(const ObjectiveFunction&) = delete;
		/*! \brief Disable copy */
		ObjectiveFunction(const ObjectiveFunction&) = delete;

		/*!
		* \brief Create object of objective function
		* \param type Specific type of objective function
		* \param config Config for objective function
		*/
		LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& type,
			const Config& config);

		/*!
		* \brief Load objective function from string object
		*/
		LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& str);

		/*!
		  * \brief Initialization logic for Gaussian process boosting
		  * \param re_model Gaussian process model
		  * \param train_gp_model_cov_pars
		  * \param use_gp_model_for_validation
		  * \param label Label data
		  */
		void InitGPModel(REModel* re_model,
			bool train_gp_model_cov_pars,
			bool use_gp_model_for_validation,
			const label_t* label);

		/*!
		* \brief Returns true if the objective function has a GP model
		*/
		bool HasGPModel() const;

		/*!
		* \brief Returns true if the random effect / GP model should be used for evaluation
		*/
		bool UseGPModelForValidation() const;

		/*!
		* \brief Returns a pointer to the random effect / GP model
		*/
		REModel* GetGPModel() const;

		/*!
		* \brief Calculate the leaf values when performing a Newton update step after the tree structure has been found (only used when has_gp_model_ == true)
		* \param data_leaf_index Leaf index for every data point (array of size num_data)
		* \param num_leaves Number of leaves
		* \param[out] leaf_values Leaf values when performing a Newton update step (array of size num_leaves)
		*/
		void NewtonUpdateLeafValues(const int* data_leaf_index,
			const int num_leaves,
			double* leaf_values) const;//used only for "regression" loss

		/*!
		* \brief Does a line search as, e.g., in Friedman (2001) to find the optimal step length for every boosting update (only used when has_gp_model_ == true)
		* \param score Current score
		* \param new_score Number of leaves
		* \param[out] lr Optimal learning rate
		*/
		virtual void LineSearchLearningRate(const double* score,
			const double* new_score,
			double& lr) const = 0;//used only for "regression" loss

	protected:
		///*! \brief Gaussian process model */
		REModel* re_model_;
		bool has_gp_model_ = false;
		bool train_gp_model_cov_pars_ = true;
		bool use_gp_model_for_validation_ = false;
		std::string likelihood_type_ = "gaussian";//used only for re_model_
		// Note: likelihood_type_ is only used for re_model_ (i.e. if has_gp_model_==true).
		// In this case, a regression objective function is always used (irrespective of the actual likelihood),
		// but this string than keeps track of the likelihood. This is a copy of re_model_->GetLikelihood(). It is saved here for speed-up
		/*! \brief If true, the learning rates for the covariance and potential auxiliary parameters are kept at the values from the previous boosting iteration and not re-initialized when optimizing them */
		bool reuse_learning_rates_gp_model_ = true;//currently only properly initialized for "regression" loss

	};

}  // namespace LightGBM

#endif   // LightGBM_OBJECTIVE_FUNCTION_H_
