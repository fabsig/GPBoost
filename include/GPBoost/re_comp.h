/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_RE_COMP_H_
#define GPB_RE_COMP_H_

#include <GPBoost/type_defs.h>
#include <GPBoost/cov_fcts.h>
#include <GPBoost/GP_utils.h>

#include <memory>
#include <mutex>
#include <vector>
#include <type_traits>
#include <random>

#include <LightGBM/utils/log.h>
using LightGBM::Log;

namespace GPBoost {

	/*!
	* \brief This class models the random effects components
	*
	*        Some details:
	*		 1. The template parameter <T_mat> can be either <den_mat_t> or <sp_mat_t>
	*/
	template<typename T_mat>
	class RECompBase {
	public:
		/*! \brief Virtual destructor */
		virtual ~RECompBase() {};

		/*!
		* \brief Create and adds the matrix Z_
		*			Note: this is currently only used when changing the likelihood in the re_model
		*/
		virtual void AddZ() = 0;

		/*!
		* \brief Drop the matrix Z_
		*			Note: this is currently only used when changing the likelihood in the re_model
		*/
		virtual void DropZ() = 0;

		/*!
		* \brief Function that sets the covariance parameters
		* \param pars Vector with covariance parameters
		*/
		virtual void SetCovPars(const vec_t& pars) = 0;

		/*!
		* \brief Transform the covariance parameters
		* \param sigma2 Marginal variance
		* \param pars Vector with covariance parameters on orignal scale
		* \param[out] pars_trans Transformed covariance parameters
		*/
		virtual void TransformCovPars(const double sigma2, const vec_t& pars, vec_t& pars_trans) = 0;

		/*!
		* \brief Back-transform the covariance parameters to the original scale
		* \param sigma2 Marginal variance
		* \param pars Vector with covariance parameters
		* \param[out] pars_orig Back-transformed, original covariance parameters
		*/
		virtual void TransformBackCovPars(const double sigma2, const vec_t& pars, vec_t& pars_orig) = 0;

		/*!
		* \brief Find "reasonable" default values for the intial values of the covariance parameters (on transformed scale)
		* \param sigma2 Marginal variance
		* \param[out] pars Vector with covariance parameters
		*/
		virtual void FindInitCovPar(vec_t& pars) const = 0;

		/*!
		* \brief Virtual function that calculates Sigma (not needed for grouped REs, at unique locations for GPs)
		*/
		virtual void CalcSigma() = 0;

		/*!
		* \brief Virtual function that calculates the covariance matrix Z*Sigma*Z^T
		* \return Covariance matrix Z*Sigma*Z^T of this component
		*   Note that since sigma_ is saved (since it is used in GetZSigmaZt and GetZSigmaZtGrad) we return a pointer and do not write on an input paramter in order to avoid copying
		*/
		virtual std::shared_ptr<T_mat> GetZSigmaZt() const = 0;

		/*!
		* \brief Virtual function that calculates the derivatives of the covariance matrix Z*Sigma*Z^T
		* \param ind_par Index for parameter
		* \param transf_scale If true, the derivative is taken on the transformed scale otherwise on the original scale. Default = true
		* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back)
		* \return Derivative of covariance matrix Z*Sigma*Z^T with respect to the parameter number ind_par
		*/
		virtual std::shared_ptr<T_mat> GetZSigmaZtGrad(int ind_par = 0, bool transf_scale = true, double = 1.) const = 0;

		/*!
		* \brief Virtual function that returns the matrix Z
		* \return A pointer to the matrix Z
		*/
		virtual sp_mat_t* GetZ() = 0;

		/*!
		* \brief This is only used for the class RECompGP and not for other derived classes. It is here in order that the base class can have this as a virtual method and no conversion needs to be made in the Vecchia approximation calculation (slightly a hack)
		*/
		virtual void CalcSigmaAndSigmaGrad(const den_mat_t& dist, den_mat_t& cov_mat,
			den_mat_t& cov_grad_1, den_mat_t& cov_grad_2,
			bool calc_gradient = false, bool transf_scale = true, double = 1.) const = 0;

		/*!
		* \brief Virtual function that returns the number of unique random effects
		* \return Number of unique random effects
		*/
		virtual data_size_t GetNumUniqueREs() const = 0;

		/*!
		* \brief Returns number of covariance parameters
		* \return Number of covariance parameters
		*/
		int NumCovPar() const {
			return(num_cov_par_);
		}

		/*!
		* \brief Calculate and add unconditional predictive variances
		* \param[out] pred_uncond_var Array of unconditional predictive variances to which the variance of this component is added
		* \param num_data_pred Number of prediction points
		* \param rand_coef_data_pred Covariate data for varying coefficients
		*/
		void AddPredUncondVar(double* pred_uncond_var, int num_data_pred, const double* const rand_coef_data_pred = nullptr) const {
			if (this->is_rand_coef_) {
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_pred; ++i) {
					pred_uncond_var[i] += this->cov_pars_[0] * rand_coef_data_pred[i] * rand_coef_data_pred[i];
				}
			}
			else {
#pragma omp for schedule(static)
				for (int i = 0; i < num_data_pred; ++i) {
					pred_uncond_var[i] += this->cov_pars_[0];
				}
			}
		}

	protected:
		/*! \brief Number of data points */
		data_size_t num_data_;
		/*! \brief Number of parameters */
		int num_cov_par_;
		/*! \brief Incidence matrix Z */
		sp_mat_t Z_;
		/*! \brief Indicates whether the random effect component has a (non-identity) incidence matrix Z */
		bool has_Z_;
		/*! \brief Covariate data for varying coefficients */
		std::vector<double> rand_coef_data_;
		/*! \brief true if this is a random coefficient */
		bool is_rand_coef_;
		/*! \brief Covariance parameters (on transformed scale, but not logarithmic) */
		vec_t cov_pars_;
		/*! \brief Indices that indicate to which random effect every data point is related (random_effects_indices_of_data_[i] is the random effect for observation number i) */
		std::vector<data_size_t> random_effects_indices_of_data_;

		template<typename T_mat_aux, typename T_chol_aux>
		friend class REModelTemplate;
	};

	/*!
	* \brief Class for the grouped random effect components
	*
	*        Some details:
	*/
	template<typename T_mat>
	class RECompGroup : public RECompBase<T_mat> {
	public:
		/*! \brief Constructor */
		RECompGroup();

		/*!
		* \brief Constructor without random coefficient data
		* \param group_data Group data: factorial variable between 1 and the number of different groups
		* \param calculateZZt If true, the matrix Z*Z^T is calculated and saved (not needed if Woodbury identity is used)
		* \param save_Z If true, the matrix Z_ is constructed and saved
		*/
		RECompGroup(std::vector<re_group_t>& group_data,
			bool calculateZZt,
			bool save_Z) {
			this->has_Z_ = save_Z;
			this->num_data_ = (data_size_t)group_data.size();
			this->is_rand_coef_ = false;
			this->num_cov_par_ = 1;
			num_group_ = 0;
			std::map<re_group_t, int> map_group_label_index;
			for (const auto& el : group_data) {
				if (map_group_label_index.find(el) == map_group_label_index.end()) {
					map_group_label_index.insert({ el, num_group_ });
					num_group_ += 1;
				}
			}
			this->random_effects_indices_of_data_ = std::vector<data_size_t>(this->num_data_);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < this->num_data_; ++i) {
				this->random_effects_indices_of_data_[i] = map_group_label_index[group_data[i]];
			}
			if (save_Z) {
				CreateZ();// Create incidence matrix Z
			}
			has_ZZt_ = calculateZZt;
			if (has_ZZt_) {
				ConstructZZt<T_mat>();
			}
			map_group_label_index_ = std::make_shared<std::map<re_group_t, int>>(map_group_label_index);
		}

		/*!
		* \brief Constructor for random coefficient effects
		* \param group_data Reference to group data of random intercept corresponding to this effect
		* \param num_group Number of groups / levels
		* \param rand_coef_data Covariate data for varying coefficients
		* \param calculateZZt If true, the matrix Z*Z^T is calculated and saved (not needed if Woodbury identity is used)
		*/
		RECompGroup(const data_size_t* random_effects_indices_of_data,
			const data_size_t num_data,
			std::shared_ptr<std::map<re_group_t, int>> map_group_label_index,
			data_size_t num_group,
			std::vector<double>& rand_coef_data,
			bool calculateZZt) {
			this->num_data_ = num_data;
			num_group_ = num_group;
			//group_data_ = group_data;
			map_group_label_index_ = map_group_label_index;
			this->rand_coef_data_ = rand_coef_data;
			this->is_rand_coef_ = true;
			this->num_cov_par_ = 1;
			this->Z_ = sp_mat_t(this->num_data_, num_group_);
			std::vector<Triplet_t> triplets(this->num_data_);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < this->num_data_; ++i) {
				triplets[i] = Triplet_t(i, random_effects_indices_of_data[i], this->rand_coef_data_[i]);
			}
			this->Z_.setFromTriplets(triplets.begin(), triplets.end());
			//// Alternative version: inserting elements directly (see constructor above)
			//for (int i = 0; i < this->num_data_; ++i) {
			//	this->Z_.insert(i, (*map_group_label_index_)[(*group_data_)[i]]) = this->rand_coef_data_[i];
			//}
			this->has_Z_ = true;
			has_ZZt_ = calculateZZt;
			if (has_ZZt_) {
				ConstructZZt<T_mat>();
			}
		}

		/*! \brief Destructor */
		~RECompGroup() {
		}

		/*!
		* \brief Create and adds the matrix Z_
		*			Note: this is currently only used when changing the likelihood in the re_model
		*/
		void AddZ() override {
			CHECK(!this->is_rand_coef_);//not intended for random coefficient models
			if (!this->has_Z_) {
				CreateZ();
				if (has_ZZt_) {
					ConstructZZt<T_mat>();
				}
			}
		}

		/*!
		* \brief Drop the matrix Z_
		*			Note: this is currently only used when changing the likelihood in the re_model
		*/
		void DropZ() override {
			CHECK(!this->is_rand_coef_);//not intended for random coefficient models
			if (this->has_Z_) {
				this->Z_.resize(0, 0);
				this->has_Z_ = false;
				if (has_ZZt_) {
					ConstructZZt<T_mat>();
				}
			}
		}

		/*!
		* \brief Create the matrix Z_
		*/
		void CreateZ() {
			CHECK(!this->is_rand_coef_);//not intended for random coefficient models
			this->Z_ = sp_mat_t(this->num_data_, num_group_);
			std::vector<Triplet_t> triplets(this->num_data_);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < this->num_data_; ++i) {
				triplets[i] = Triplet_t(i, this->random_effects_indices_of_data_[i], 1);
			}
			this->Z_.setFromTriplets(triplets.begin(), triplets.end());
			// Alternative version: inserting elements directly
			// Note: compared to using triples, this is much slower when group_data is not ordered (e.g. [1,2,3,1,2,3]), otherwise if group_data is ordered (e.g. [1,1,2,2,3,3]) there is no big difference
			////this->Z_.reserve(Eigen::VectorXi::Constant(this->num_data_, 1));//don't use this, it makes things much slower
			//for (int i = 0; i < this->num_data_; ++i) {
			//	this->Z_.insert(i, this->random_effects_indices_of_data_[i]) = 1.;
			//}
		}

		/*!
		* \brief Function that sets the covariance parameters
		* \param pars Vector of length 1 with variance of the grouped random effect
		*/
		void SetCovPars(const vec_t& pars) override {
			CHECK((int)pars.size() == 1);
			this->cov_pars_ = pars;
		}

		/*!
		* \brief Transform the covariance parameters
		* \param sigma2 Marginal variance
		* \param pars Vector of length 1 with variance of the grouped random effect
		* \param[out] pars_trans Transformed covariance parameters
		*/
		void TransformCovPars(const double sigma2, const vec_t& pars, vec_t& pars_trans) override {
			pars_trans = pars / sigma2;
		}

		/*!
		* \brief Back-transform the covariance parameters to the original scale
		* \param sigma2 Marginal variance
		* \param pars Vector of length 1 with variance of the grouped random effect
		* \param[out] pars_orig Back-transformed, original covariance parameters
		*/
		void TransformBackCovPars(const double sigma2, const vec_t& pars, vec_t& pars_orig) override {
			pars_orig = sigma2 * pars;
		}

		/*!
		* \brief Find "reasonable" default values for the intial values of the covariance parameters (on transformed scale)
		* \param sigma2 Marginal variance
		* \param[out] pars Vector of length 1 with variance of the grouped random effect
		*/
		void FindInitCovPar(vec_t& pars) const override {//TODO: find better initial estimates (as e.g. the variance of the group means)
			pars[0] = 1;
		}

		/*!
		* \brief Calculate covariance matrix Sigma (not needed for grouped REs)
		*/
		void CalcSigma() override {
		}

		/*!
		* \brief Calculate covariance matrix Z*Sigma*Z^T
		* \param pars Vector of length 1 with covariance parameter sigma_j for grouped RE component number j
		* \return Covariance matrix Z*Sigma*Z^T of this component
		*/
		std::shared_ptr<T_mat> GetZSigmaZt() const override {
			if (this->cov_pars_.size() == 0) {
				Log::REFatal("Covariance parameters are not specified. Call 'SetCovPars' first.");
			}
			if (this->ZZt_.cols() == 0) {
				Log::REFatal("Matrix ZZt_ not defined");
			}
			return(std::make_shared<T_mat>(this->cov_pars_[0] * ZZt_));
		}

		/*!
		* \brief Calculate derivative of covariance matrix Z*Sigma*Z^T with respect to the parameter
		* \param ind_par Index for parameter (0=variance, 1=inverse range)
		* \param transf_scale If true, the derivative is taken on the transformed scale otherwise on the original scale. Default = true
		* \param nugget_var Nugget effect variance parameter sigma^2 (not use here)
		* \return Derivative of covariance matrix Z*Sigma*Z^T with respect to the parameter number ind_par
		*/
		std::shared_ptr<T_mat> GetZSigmaZtGrad(int ind_par, bool transf_scale = true, double = 1.) const override {
			if (this->cov_pars_.size() == 0) {
				Log::REFatal("Covariance parameters are not specified. Call 'SetCovPars' first.");
			}
			if (this->ZZt_.cols() == 0) {
				Log::REFatal("Matrix ZZt_ not defined");
			}
			if (ind_par != 0) {
				Log::REFatal("No covariance parameter for index number %d", ind_par);
			}
			double cm = transf_scale ? this->cov_pars_[0] : 1.;
			return(std::make_shared<T_mat>(cm * ZZt_));
		}

		/*!
		* \brief Function that returns the matrix Z
		* \return A pointer to the matrix Z
		*/
		sp_mat_t* GetZ() override {
			CHECK(this->has_Z_);
			return(&(this->Z_));
		}

		/*!
		* \brief Calculate and add covariance matrices from this component for prediction
		* \param group_data_pred Group data for predictions
		* \param[out] cross_cov Cross covariance between prediction and observation points
		* \param[out] uncond_pred_cov Unconditional covariance for prediction points (used only if predict_cov_mat==true)
		* \param predict_cov_mat If true, all matrices are calculated. If false only Ztilde*Sigma*Z^T required for the conditional mean is calculated
		* \param dont_add_but_overwrite If true, the matrices are not added but overwritten. If false, the matrices are added to cross_cov and uncond_pred_cov
		* \param data_duplicates_dropped_for_prediction If true, duplicate groups in group_data (of training data) are dropped for creating prediction matrices (they are added again in re_model_template)
		* \param rand_coef_data_pred Covariate data for varying coefficients (can be nullptr if this is not a random coefficient)
		*/
		void AddPredCovMatrices(const std::vector<re_group_t>& group_data_pred,
			T_mat& cross_cov,
			T_mat& uncond_pred_cov,
			bool predict_cov_mat,
			bool dont_add_but_overwrite,
			bool data_duplicates_dropped_for_prediction,
			double* rand_coef_data_pred) {
			int num_data_pred = (int)group_data_pred.size();
			if (data_duplicates_dropped_for_prediction) {
				T_mat ZtildeZT(num_data_pred, num_group_);
				ZtildeZT.setZero();
				for (int i = 0; i < num_data_pred; ++i) {
					if (map_group_label_index_->find(group_data_pred[i]) != map_group_label_index_->end()) {//Group level 'group_data_pred[i]' exists in observed data
						ZtildeZT.coeffRef(i, (*map_group_label_index_)[group_data_pred[i]]) = 1.;
					}
				}
				if (dont_add_but_overwrite) {
					cross_cov = this->cov_pars_[0] * ZtildeZT;
				}
				else {
					cross_cov += this->cov_pars_[0] * ZtildeZT;
				}
				if (predict_cov_mat) {
					T_mat ZstarZstarT(num_data_pred, num_data_pred);
					ZstarZstarT.setZero();
					T_mat ZtildeZtildeT(num_data_pred, num_data_pred);
					ZtildeZtildeT.setZero();
					for (int i = 0; i < num_data_pred; ++i) {
						if (map_group_label_index_->find(group_data_pred[i]) == map_group_label_index_->end()) {
							ZstarZstarT.coeffRef(i, i) = 1.;
						}
						else {
							ZtildeZtildeT.coeffRef(i, i) = 1.;
						}
					}
					uncond_pred_cov += (this->cov_pars_[0] * ZtildeZtildeT);
					uncond_pred_cov += (this->cov_pars_[0] * ZstarZstarT);
				}//end predict_cov_mat
			}//end data_duplicates_dropped_for_prediction
			else if (this->has_Z_) {
				//Note: Ztilde relates existing random effects to predictionsand Zstar relates new / unobserved random effects to predictions
				sp_mat_t Ztilde(num_data_pred, num_group_);
				Ztilde.setZero();
				for (int i = 0; i < num_data_pred; ++i) {
					if (map_group_label_index_->find(group_data_pred[i]) != map_group_label_index_->end()) {//Group level 'group_data_pred[i]' exists in observed data
						if (this->is_rand_coef_) {
							Ztilde.insert(i, (*map_group_label_index_)[group_data_pred[i]]) = rand_coef_data_pred[i];
						}
						else {
							Ztilde.insert(i, (*map_group_label_index_)[group_data_pred[i]]) = 1.;
						}
					}
				}
				if (dont_add_but_overwrite) {
					CalculateZ1Z2T<T_mat>(Ztilde, this->Z_, cross_cov);
					cross_cov *= this->cov_pars_[0];
				}
				else {
					T_mat ZtildeZT;
					CalculateZ1Z2T<T_mat>(Ztilde, this->Z_, ZtildeZT);
					cross_cov += (this->cov_pars_[0] * ZtildeZT);
				}
				if (predict_cov_mat) {
					//Count number of new group levels (i.e. group levels not in observed data)
					int num_group_pred_new = 0;
					std::map<re_group_t, int> map_group_label_index_pred_new; //Keys: Group labels, values: index number (integer value) for every label  
					for (auto& el : group_data_pred) {
						if (map_group_label_index_->find(el) == map_group_label_index_->end()) {
							if (map_group_label_index_pred_new.find(el) == map_group_label_index_pred_new.end()) {
								map_group_label_index_pred_new.insert({ el, num_group_pred_new });
								num_group_pred_new += 1;
							}
						}
					}
					sp_mat_t Zstar(num_data_pred, num_group_pred_new);
					for (int i = 0; i < num_data_pred; ++i) {
						if (map_group_label_index_->find(group_data_pred[i]) == map_group_label_index_->end()) {
							if (this->is_rand_coef_) {
								Zstar.insert(i, map_group_label_index_pred_new[group_data_pred[i]]) = rand_coef_data_pred[i];
							}
							else {
								Zstar.insert(i, map_group_label_index_pred_new[group_data_pred[i]]) = 1.;
							}
						}
					}
					T_mat ZtildeZtildeT;
					CalculateZ1Z2T<T_mat>(Ztilde, Ztilde, ZtildeZtildeT);
					uncond_pred_cov += (this->cov_pars_[0] * ZtildeZtildeT);
					T_mat ZstarZstarT;
					CalculateZ1Z2T<T_mat>(Zstar, Zstar, ZstarZstarT);
					uncond_pred_cov += (this->cov_pars_[0] * ZstarZstarT);
				}//end predict_cov_mat
			}//end this->has_Z_
			else {
				Log::REFatal("Need to have either 'Z_' or enable 'data_duplicates_dropped_for_prediction' for calling 'AddPredCovMatrices'");
			}
		}

		// Ignore this. This is not used for this class (it is only used for the class RECompGP). It is here in order that the base class can have this as a virtual method and no conversion needs to be made in the Vecchia approximation calculation (slightly a hack)
		void CalcSigmaAndSigmaGrad(const den_mat_t&, den_mat_t&,
			den_mat_t&, den_mat_t&,
			bool = false, bool = true, double = 1.) const override {

		}

		data_size_t GetNumUniqueREs() const override {
			return(num_group_);
		}

	private:
		/*! \brief Number of groups */
		data_size_t num_group_;
		/*! \brief Keys: Group labels, values: index number (integer value) for every group level. I.e., maps string labels to numbers */
		std::shared_ptr<std::map<re_group_t, int>> map_group_label_index_;
		/*! \brief Matrix Z*Z^T */
		T_mat ZZt_;
		/*! \brief Indicates whether component has a matrix ZZt_ */
		bool has_ZZt_;

		/*! \brief Constructs the matrix ZZt_ if sparse matrices are used */
		template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr >
		void ConstructZZt() {
			if (this->has_Z_) {
				ZZt_ = this->Z_ * this->Z_.transpose();
			}
			else {
				ZZt_ = T_mat(num_group_, num_group_);
				ZZt_.setIdentity();
				//Note: If has_Z_==false, ZZt_ is only used for making predictiosn of new independet clusters when only_one_grouped_RE_calculations_on_RE_scale_==true
			}
		}

		/*! \brief Constructs the matrix ZZt_ if dense matrices are used */
		template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr >
		void ConstructZZt() {
			if (this->has_Z_) {
				ZZt_ = den_mat_t(this->Z_ * this->Z_.transpose());
			}
			else {
				ZZt_ = T_mat(num_group_, num_group_);
				ZZt_.setIdentity();
				//Note: If has_Z_==false, ZZt_ is only used for making predictiosn of new independet clusters when only_one_grouped_RE_calculations_on_RE_scale_==true
			}
		}

		/*!
		* \brief Calculates the matrix Z1*Z2^T if sparse matrices are used
		* \param Z1 Matrix
		* \param Z2 Matrix
		* \param[out] Z1Z2T Matrix Z1*Z2^T
		*/
		template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr >
		void CalculateZ1Z2T(sp_mat_t& Z1, sp_mat_t& Z2, T3& Z1Z2T) {
			Z1Z2T = Z1 * Z2.transpose();
		}

		/*!
		* \brief Calculates the matrix Z1*Z2^T if sparse matrices are used
		* \param Z1 Matrix
		* \param Z2 Matrix
		* \param[out] Z1Z2T Matrix Z1*Z2^T
		*/
		template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr >
		void CalculateZ1Z2T(sp_mat_t& Z1, sp_mat_t& Z2, T3& Z1Z2T) {
			Z1Z2T = den_mat_t(Z1 * Z2.transpose());
		}

		template<typename T_mat_aux, typename T_chol_aux>
		friend class REModelTemplate;
	};

	/*!
	* \brief Class for the Gaussian process components
	*
	*        Some details:
	*        ...
	*/
	template<typename T_mat>
	class RECompGP : public RECompBase<T_mat> {
	public:
		/*! \brief Constructor */
		RECompGP();

		/*!
		* \brief Constructor for Gaussian process
		* \param coords Coordinates (features) for Gaussian process
		* \param cov_fct Type of covariance function
		* \param shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. For the Wendland covariance function, we follow the notation of Bevilacqua et al. (2018)). This parameter is irrelevant for some covariance functions such as the exponential or Gaussian.
		* \param taper_range Range parameter of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
		* \param save_dist_use_Z_for_duplicates If true, distances are calculated and saved here, and an incidendce matrix Z_ is used for duplicate locations.
		*           save_dist_use_Z_for_duplicates = false is used for the Vecchia approximation which saves the required distances in the REModel (REModelTemplate)
		* \param save_random_effects_indices_of_data_and_no_Z If true a vector random_effects_indices_of_data_, which relates random effects b to samples Zb, is used (the matrix Z_ is then not constructed)
		*           save_random_effects_indices_of_data_and_no_Z = true is currently only used when doing calculations on the random effects scale b and not on the "data scale" Zb for non-Gaussian data
		*			This option can only be selected when save_dist_use_Z_for_duplicates = true
		*/
		RECompGP(const den_mat_t& coords,
			string_t cov_fct,
			double shape,
			double taper_range,
			bool save_dist_use_Z_for_duplicates,
			bool save_random_effects_indices_of_data_and_no_Z) {
			if (save_random_effects_indices_of_data_and_no_Z && !save_dist_use_Z_for_duplicates) {
				Log::REFatal("'save_dist_use_Z_for_duplicates' cannot be 'false' when 'save_random_effects_indices_of_data_and_no_Z' is 'true'");
			}
			this->num_data_ = (data_size_t)coords.rows();
			this->is_rand_coef_ = false;
			this->has_Z_ = false;
			double taper_mu = 2.;
			if (cov_fct == "wendland" || cov_fct == "exponential_tapered") {
				taper_mu = (1. + coords.cols()) / 2. + shape + 0.5;
				// taper_mu is chosen such that for coords.cols() = 2, the Wendland covariance functions coincide with the ones from Furrer et al. (2006) (Table 1)
			}
			cov_function_ = std::unique_ptr<CovFunction>(new CovFunction(cov_fct, shape, taper_range, taper_mu));
			this->num_cov_par_ = cov_function_->num_cov_par_;
			if (save_dist_use_Z_for_duplicates) {
				std::vector<int> uniques;//unique points
				std::vector<int> unique_idx;//used for constructing incidence matrix Z_ if there are duplicates
				DetermineUniqueDuplicateCoords(coords, this->num_data_, uniques, unique_idx);
				if ((data_size_t)uniques.size() == this->num_data_) {//no multiple observations at the same locations -> no incidence matrix needed
					coords_ = coords;
				}
				else {//there are multiple observations at the same locations
					coords_ = coords(uniques, Eigen::all);
				}
				num_random_effects_ = (data_size_t)coords_.rows();
				if (save_random_effects_indices_of_data_and_no_Z) {// create random_effects_indices_of_data_
					this->random_effects_indices_of_data_ = std::vector<data_size_t>(this->num_data_);
#pragma omp for schedule(static)
					for (int i = 0; i < this->num_data_; ++i) {
						this->random_effects_indices_of_data_[i] = unique_idx[i];
					}
					this->has_Z_ = false;
				}
				else if (num_random_effects_ != this->num_data_) {// create incidence matrix Z_
					this->Z_ = sp_mat_t(this->num_data_, num_random_effects_);
					for (int i = 0; i < this->num_data_; ++i) {
						this->Z_.insert(i, unique_idx[i]) = 1.;
					}
					this->has_Z_ = true;
				}
				//Calculate distances
				T_mat dist;
				if (COMPACT_SUPPORT_COVS_.find(cov_function_->cov_fct_type_) != COMPACT_SUPPORT_COVS_.end()) {//compactly suported covariance
					CalculateDistances(coords_, coords_, true, cov_function_->taper_range_, true, dist);
				}
				else {
					CalculateDistances(coords_, coords_, true, dist);
				}
				dist_ = std::make_shared<T_mat>(dist);
				dist_saved_ = true;
			}//end save_dist_use_Z_for_duplicates
			else {//this option is used for the Vecchia approximation
				coords_ = coords;
				dist_saved_ = false;
				num_random_effects_ = (data_size_t)coords_.rows();
			}
			coord_saved_ = true;
		}

		/*!
		* \brief Constructor for random coefficient Gaussian processes
		* \param dist Pointer to distance matrix of corresponding base intercept GP
		* \param base_effect_has_Z Indicate whether the corresponding base GP has an incidence matrix Z or not
		* \param Z Pointer to incidence matrix Z of corresponding base intercept GP
		* \param rand_coef_data Covariate data for random coefficient
		* \param cov_fct Type of covariance function
		* \param shape Shape parameter of covariance function (=smoothness parameter for Matern covariance, irrelevant for some covariance functions such as the exponential or Gaussian)
		* \param taper_range Range parameter of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
		* \param taper_mu Parameter \mu of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
		*/
		RECompGP(std::shared_ptr<T_mat> dist,
			bool base_effect_has_Z,
			sp_mat_t* Z,
			const std::vector<double>& rand_coef_data,
			string_t cov_fct,
			double shape,
			double taper_range,
			double taper_mu) {
			this->num_data_ = (data_size_t)rand_coef_data.size();
			dist_ = dist;
			dist_saved_ = true;
			this->rand_coef_data_ = rand_coef_data;
			this->is_rand_coef_ = true;
			this->has_Z_ = true;
			cov_function_ = std::unique_ptr<CovFunction>(new CovFunction(cov_fct, shape, taper_range, taper_mu));
			this->num_cov_par_ = cov_function_->num_cov_par_;
			sp_mat_t coef_W(this->num_data_, this->num_data_);
			for (int i = 0; i < this->num_data_; ++i) {
				coef_W.insert(i, i) = this->rand_coef_data_[i];
			}
			if (base_effect_has_Z) {//"Base" intercept GP has a (non-identity) incidence matrix (i.e., there are multiple observations at the same locations)
				this->Z_ = coef_W * *Z;
			}
			else {
				this->Z_ = coef_W;
			}
			coord_saved_ = false;
			num_random_effects_ = (data_size_t)this->Z_.cols();
		}

		/*!
		* \brief Constructor for random coefficient Gaussian process when multiple locations are not modelled using an incidence matrix.
		*		This is used for the Vecchia approximation.
		* \param rand_coef_data Covariate data for random coefficient
		* \param cov_fct Type of covariance function
		* \param shape Shape parameter of covariance function (=smoothness parameter for Matern covariance, irrelevant for some covariance functions such as the exponential or Gaussian)
		* \param taper_range Range parameter of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
		* \param taper_mu Parameter \mu of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
		*/
		RECompGP(const std::vector<double>& rand_coef_data,
			string_t cov_fct,
			double shape,
			double taper_range,
			double taper_mu) {
			this->rand_coef_data_ = rand_coef_data;
			this->is_rand_coef_ = true;
			this->num_data_ = (data_size_t)rand_coef_data.size();
			this->has_Z_ = true;
			cov_function_ = std::unique_ptr<CovFunction>(new CovFunction(cov_fct, shape, taper_range, taper_mu));
			this->num_cov_par_ = cov_function_->num_cov_par_;
			dist_saved_ = false;
			coord_saved_ = false;
			this->Z_ = sp_mat_t(this->num_data_, this->num_data_);
			for (int i = 0; i < this->num_data_; ++i) {
				this->Z_.insert(i, i) = this->rand_coef_data_[i];
			}
			num_random_effects_ = this->num_data_;
		}

		/*! \brief Destructor */
		~RECompGP() {
		}

		/*!
		* \brief Create and adds the matrix Z_
		*			Note: this is currently only used when changing the likelihood in the re_model
		*/
		void AddZ() override {
			CHECK(!this->is_rand_coef_);//not intended for random coefficient models
			if (!this->has_Z_) {
				if (num_random_effects_ != this->num_data_) {// create incidence matrix Z_
					CHECK((data_size_t)(this->random_effects_indices_of_data_.size()) == this->num_data_);
					this->Z_ = sp_mat_t(this->num_data_, num_random_effects_);
					for (int i = 0; i < this->num_data_; ++i) {
						this->Z_.insert(i, this->random_effects_indices_of_data_[i]) = 1.;
					}
					this->has_Z_ = true;
				}
			}
		}

		/*!
		* \brief Drop the matrix Z_
		*			Note: this is currently only used when changing the likelihood in the re_model
		*/
		void DropZ() override {
			CHECK(!this->is_rand_coef_);//not intended for random coefficient models
			if (this->has_Z_) {
				this->random_effects_indices_of_data_ = std::vector<data_size_t>(this->num_data_);
				for (int k = 0; k < this->Z_.outerSize(); ++k) {
					for (sp_mat_t::InnerIterator it(this->Z_, k); it; ++it) {
						this->random_effects_indices_of_data_[(int)it.row()] = (data_size_t)it.col();
					}
				}
				this->has_Z_ = false;
				this->Z_.resize(0, 0);
			}
		}

		/*!
		* \brief Function that sets the covariance parameters
		* \param pars Vector of length 2 with covariance parameters (variance and inverse range)
		*/
		void SetCovPars(const vec_t& pars) override {
			CHECK((int)pars.size() == this->num_cov_par_);
			this->cov_pars_ = pars;
		}

		/*!
		* \brief Transform the covariance parameters
		* \param sigma2 Marginal variance
		* \param pars Vector with covariance parameters on orignal scale
		* \param[out] pars_trans Transformed covariance parameters
		*/
		void TransformCovPars(const double sigma2, const vec_t& pars, vec_t& pars_trans) override {
			cov_function_->TransformCovPars(sigma2, pars, pars_trans);
		}

		/*!
		* \brief Function that sets the covariance parameters
		* \param sigma2 Marginal variance
		* \param pars Vector with covariance parameters
		* \param[out] pars_orig Back-transformed, original covariance parameters
		*/
		void TransformBackCovPars(const double sigma2, const vec_t& pars, vec_t& pars_orig) override {
			cov_function_->TransformBackCovPars(sigma2, pars, pars_orig);
		}

		/*!
		* \brief Find "reasonable" default values for the intial values of the covariance parameters (on transformed scale)
		* \param sigma2 Marginal variance
		* \param[out] pars Vector with covariance parameters
		*/
		void FindInitCovPar(vec_t& pars) const override {
			if (!dist_saved_ && !coord_saved_) {
				Log::REFatal("Cannot determine initial covariance parameters if neither distances nor coordinates are given");
			}
			pars[0] = 1;
			if (cov_function_->cov_fct_type_ != "wendland") {
				int MAX_POINTS_INIT_RANGE = 1000;//limit number of sampels considered to maximally 1000 to save computational time
				int num_coord;
				if (dist_saved_) {
					num_coord = (int)(*dist_).rows();
				}
				else {
					num_coord = (int)coords_.rows();
				}
				int num_data_find_init = (num_coord > MAX_POINTS_INIT_RANGE) ? MAX_POINTS_INIT_RANGE : num_coord;
				std::vector<int> sample_ind;
				bool use_subsamples = num_data_find_init < num_coord;
				if (use_subsamples) {
					std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with 0
					std::uniform_int_distribution<> dis(0, num_coord - 1);
					sample_ind = std::vector<int>(num_data_find_init);
					for (int i = 0; i < num_data_find_init; ++i) {
						sample_ind[i] = dis(gen);
					}
				}
				double mean_dist = 0;
				if (dist_saved_) {
					if (use_subsamples) {
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								mean_dist += (*dist_).coeffRef(sample_ind[i], sample_ind[j]);
							}
						}
					}
					else {
						for (int i = 0; i < (num_coord - 1); ++i) {
							for (int j = i + 1; j < num_coord; ++j) {
								mean_dist += (*dist_).coeffRef(i, j);
							}
						}
					}
				}
				else {
					//Calculate distances (of a subsample) in case they have not been calculated (for the Vecchia approximation)
					den_mat_t dist;
					if (use_subsamples) {
						CalculateDistances(coords_(sample_ind, Eigen::all), coords_(sample_ind, Eigen::all), true, dist);
					}
					else {
						CalculateDistances(coords_, coords_, true, dist);
					}
					for (int i = 0; i < (num_data_find_init - 1); ++i) {
						for (int j = i + 1; j < num_data_find_init; ++j) {
							mean_dist += dist(i, j);
						}
					}
				}
				mean_dist /= (num_data_find_init * (num_data_find_init - 1) / 2.);
				//Set the range parameter such that the correlation is down to 0.05 at the mean distance
				if (cov_function_->cov_fct_type_ == "exponential" || cov_function_->cov_fct_type_ == "matern" ||
					cov_function_->cov_fct_type_ == "exponential_tapered") {//TODO: find better intial values for matern covariance for shape = 1.5 and shape = 2.5
					pars[1] = 3. / mean_dist;//pars[1] = 1/range
				}
				else if (cov_function_->cov_fct_type_ == "gaussian") {
					pars[1] = 3. / std::pow(mean_dist, 2.);//pars[1] = 1/range^2
				}
				else if (cov_function_->cov_fct_type_ == "powered_exponential") {
					pars[1] = 3. / std::pow(mean_dist, cov_function_->shape_);//pars[1] = 1/range^shape
				}
				else {
					Log::REFatal("Finding initial values for covariance paramters for covariance of type '%s' is not supported.", cov_function_->cov_fct_type_.c_str());
				}
			}
		}

		/*!
		* \brief Calculate covariance matrix at unique locations
		*/
		void CalcSigma() override {
			if (this->cov_pars_.size() == 0) { Log::REFatal("Covariance parameters are not specified. Call 'SetCovPars' first."); }
			//(*cov_function_).template GetCovMat(*dist_, this->cov_pars_, sigma_);Old version used when CovFunction was a template class
			cov_function_->GetCovMat(*dist_, this->cov_pars_, sigma_);
			sigma_defined_ = true;
		}

		/*!
		* \brief Calculate covariance matrix
		* \return Covariance matrix Z*Sigma*Z^T of this component
		*/
		std::shared_ptr<T_mat> GetZSigmaZt() const override {
			if (!sigma_defined_) {
				Log::REFatal("Sigma has not been calculated");
			}
			if (this->is_rand_coef_ || this->has_Z_) {
				return(std::make_shared<T_mat>(this->Z_ * sigma_ * this->Z_.transpose()));
			}
			else {
				return(std::make_shared<T_mat>(sigma_));
			}
		}

		/*!
		* \brief Calculate covariance matrix and gradients with respect to covariance parameters (used for Vecchia approx.)
		* \param dist Distance matrix
		* \param[out] cov_mat Covariance matrix Z*Sigma*Z^T
		* \param[out] cov_grad_1 Gradient of covariance matrix with respect to marginal variance parameter
		* \param[out] cov_grad_2 Gradient of covariance matrix with respect to range parameter
		* \param calc_gradient If true, gradients are also calculated, otherwise not
		* \param transf_scale If true, the derivative are calculated on the transformed scale otherwise on the original scale. Default = true
		* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back)
		*/
		void CalcSigmaAndSigmaGrad(const den_mat_t& dist, den_mat_t& cov_mat,
			den_mat_t& cov_grad_1, den_mat_t& cov_grad_2,
			bool calc_gradient = false, bool transf_scale = true, double nugget_var = 1.) const override {
			if (this->cov_pars_.size() == 0) { Log::REFatal("Covariance parameters are not specified. Call 'SetCovPars' first."); }
			cov_function_->GetCovMat(dist, this->cov_pars_, cov_mat);
			if (calc_gradient) {
				//gradient wrt to variance parameter
				cov_grad_1 = cov_mat;
				if (!transf_scale) {
					cov_grad_1 /= this->cov_pars_[0];
				}
				if (cov_function_->cov_fct_type_ != "wendland") {
					//gradient wrt to range parameter
					cov_function_->GetCovMatGradRange(dist, cov_mat, this->cov_pars_, cov_grad_2, transf_scale, nugget_var);
				}
			}
			if (!transf_scale) {
				cov_mat *= nugget_var;//transform back to original scale
			}
		}

		/*!
		* \brief Calculate derivatives of covariance matrix with respect to the parameters
		* \param ind_par Index for parameter (0=variance, 1=inverse range)
		* \param transf_scale If true, the derivative is taken on the transformed scale otherwise on the original scale. Default = true
		* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back)
		* \return Derivative of covariance matrix Z*Sigma*Z^T with respect to the parameter number ind_par
		*/
		std::shared_ptr<T_mat> GetZSigmaZtGrad(int ind_par, bool transf_scale = true, double nugget_var = 1.) const override {
			if (!sigma_defined_) {
				Log::REFatal("Sigma has not been calculated");
			}
			if (ind_par != 0 && ind_par != 1) {
				Log::REFatal("No covariance parameter for index number %d", ind_par);
			}
			if (ind_par == 0) {//variance
				if (transf_scale) {
					return(GetZSigmaZt());
				}
				else {
					double correct = 1. / this->cov_pars_[0];//divide sigma_ by cov_pars_[0]
					if (this->is_rand_coef_ || this->has_Z_) {
						return(std::make_shared<T_mat>(correct * this->Z_ * sigma_ * this->Z_.transpose()));
					}
					else {
						return(std::make_shared<T_mat>(correct * sigma_));
					}
				}
			}
			else {//inverse range (ind_par == 1)
				CHECK(cov_function_->cov_fct_type_ != "wendland");
				T_mat Z_sigma_grad_Zt;
				if (this->has_Z_) {
					T_mat sigma_grad;
					cov_function_->GetCovMatGradRange(*dist_, sigma_, this->cov_pars_, sigma_grad, transf_scale, nugget_var);
					Z_sigma_grad_Zt = this->Z_ * sigma_grad * this->Z_.transpose();
				}
				else {
					cov_function_->GetCovMatGradRange(*dist_, sigma_, this->cov_pars_, Z_sigma_grad_Zt, transf_scale, nugget_var);
				}
				return(std::make_shared<T_mat>(Z_sigma_grad_Zt));
			}
		}

		/*!
		* \brief Function that returns the matrix Z
		* \return A pointer to the matrix Z
		*/
		sp_mat_t* GetZ() override {
			if (!this->has_Z_) {
				Log::REFatal("Gaussian process has no matrix Z");
			}
			return(&(this->Z_));
		}

		/*!
		* \brief Calculate and add covariance matrices from this component for prediction
		* \param coords Coordinates for observed data
		* \param coords_pred Coordinates for predictions
		* \param[out] cross_cov Cross covariance between prediction and observation points
		* \param[out] uncond_pred_cov Unconditional covariance for prediction points (used only if predict_cov_mat==true)
		* \param predict_cov_mat If true, all matrices are calculated. If false only Ztilde*Sigma*Z^T required for the conditional mean is calculated
		* \param dont_add_but_overwrite If true, the matrices are not added but overwritten
		* \param rand_coef_data_pred Covariate data for varying coefficients (can be nullptr if this is not a random coefficient)
		*/
		void AddPredCovMatrices(const den_mat_t& coords,
			const den_mat_t& coords_pred,
			T_mat& cross_cov,
			T_mat& uncond_pred_cov,
			bool predict_cov_mat,
			bool dont_add_but_overwrite,
			double* rand_coef_data_pred) {
			int num_data_pred = (int)coords_pred.rows();
			std::vector<int>  uniques_pred;//unique points
			std::vector<int>  unique_idx_pred;//used for constructing incidence matrix Z_ if there are duplicates
			DetermineUniqueDuplicateCoords(coords_pred, num_data_pred, uniques_pred, unique_idx_pred);
			bool has_duplicates = (int)uniques_pred.size() != num_data_pred;
			bool has_Zstar = has_duplicates || this->is_rand_coef_;
			sp_mat_t Zstar;
			den_mat_t coords_pred_unique;
			//Create matrix Zstar
			if (has_Zstar) {
				// Note: Ztilde relates existing random effects to predictions and Zstar relates new / unobserved random effects to predictions
				Zstar = sp_mat_t(num_data_pred, uniques_pred.size());
				Zstar.setZero();
			}
			if (has_duplicates) {//Only keep unique coordinates if there are multiple observations with the same coordinates
				coords_pred_unique = coords_pred(uniques_pred, Eigen::all);
			}
			if (has_Zstar) {
				for (int i = 0; i < num_data_pred; ++i) {
					if (this->is_rand_coef_) {
						Zstar.insert(i, unique_idx_pred[i]) = rand_coef_data_pred[i];
					}
					else {
						Zstar.insert(i, unique_idx_pred[i]) = 1.;
					}
				}
			}
			//Calculate cross distances between "existing" and "new" points
			T_mat cross_dist;
			if (has_duplicates) {
				if (COMPACT_SUPPORT_COVS_.find(cov_function_->cov_fct_type_) != COMPACT_SUPPORT_COVS_.end()) {//compactly suported covariance
					CalculateDistances(coords, coords_pred_unique, false, cov_function_->taper_range_, false, cross_dist);
				}
				else {
					CalculateDistances(coords, coords_pred_unique, false, cross_dist);
				}
			}
			else {
				if (COMPACT_SUPPORT_COVS_.find(cov_function_->cov_fct_type_) != COMPACT_SUPPORT_COVS_.end()) {//compactly suported covariance
					CalculateDistances(coords, coords_pred, false, cov_function_->taper_range_, false, cross_dist);
				}
				else {
					CalculateDistances(coords, coords_pred, false, cross_dist);
				}
			}
			T_mat ZstarSigmatildeTZT;
			if (has_Zstar || this->has_Z_) {
				T_mat Sigmatilde;
				cov_function_->GetCovMat(cross_dist, this->cov_pars_, Sigmatilde);
				if (has_Zstar && this->has_Z_) {
					ZstarSigmatildeTZT = Zstar * Sigmatilde * this->Z_.transpose();
				}
				else if (has_Zstar && !(this->has_Z_)) {
					ZstarSigmatildeTZT = Zstar * Sigmatilde;
				}
				else if (!has_Zstar && this->has_Z_) {
					ZstarSigmatildeTZT = Sigmatilde * this->Z_.transpose();
				}
			}//end has_Zstar || this->has_Z_
			else { //no Zstar and no Z_
				cov_function_->GetCovMat(cross_dist, this->cov_pars_, ZstarSigmatildeTZT);
			}
			if (dont_add_but_overwrite) {
				cross_cov = ZstarSigmatildeTZT;
			}
			else {
				cross_cov += ZstarSigmatildeTZT;
			}
			if (predict_cov_mat) {
				T_mat dist;
				if (COMPACT_SUPPORT_COVS_.find(cov_function_->cov_fct_type_) != COMPACT_SUPPORT_COVS_.end()) {//compactly suported covariance
					CalculateDistances(coords_pred, coords_pred, true, cov_function_->taper_range_, false, dist);
				}
				else {
					CalculateDistances(coords_pred, coords_pred, true, dist);
				}
				T_mat ZstarSigmastarZstarT;
				if (has_Zstar) {
					T_mat Sigmastar;
					cov_function_->GetCovMat(dist, this->cov_pars_, Sigmastar);
					ZstarSigmastarZstarT = Zstar * Sigmastar * Zstar.transpose();
				}
				else {
					cov_function_->GetCovMat(dist, this->cov_pars_, ZstarSigmastarZstarT);
				}
				uncond_pred_cov += ZstarSigmastarZstarT;
			}
		}

		data_size_t GetNumUniqueREs() const override {
			return(num_random_effects_);
		}

		double GetTaperMu() const {
			return(cov_function_->taper_mu_);
		}

	private:
		/*! \brief Coordinates (=features) */
		den_mat_t coords_;
		/*! \brief Distance matrix (between unique coordinates in coords_) */
		std::shared_ptr<T_mat> dist_;
		/*! \brief If true, the distancess among all observations are calculated and saved here (false for Vecchia approximation) */
		bool dist_saved_ = true;
		/*! \brief If true, the coordinates are saved (false for random coefficients GPs) */
		bool coord_saved_ = true;
		/*! \brief Covariance function */
		std::unique_ptr<CovFunction> cov_function_;
		/*! \brief Covariance matrix (for a certain choice of covariance paramters). This is saved for re-use at two locations in the code: GetZSigmaZt and GetZSigmaZtGrad) */
		T_mat sigma_;
		/*! \brief Indicates whether sigma_ has been defined or not */
		bool sigma_defined_ = false;
		/*! \brief Number of random effects (usually, number of unique random effects except for the Vecchia approximation where unique locations are not separately modelled) */
		data_size_t num_random_effects_;
		/*! \brief List of covariance functions wtih compact support */
		const std::set<string_t> COMPACT_SUPPORT_COVS_{ "wendland",
			"exponential_tapered" };

		template<typename T_mat_aux, typename T_chol_aux>
		friend class REModelTemplate;
	};

}  // namespace GPBoost

#endif   // GPB_RE_COMP_H_
