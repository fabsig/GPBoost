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

#include <GPBoost/log.h>
#include <GPBoost/type_defs.h>
#include <GPBoost/cov_fcts.h>
#include <GPBoost/GP_utils.h>

#include <memory>
#include <mutex>
#include <vector>
#include <type_traits>
#include <random>

namespace GPBoost {

	/*!
	* \brief This class models the random effects components
	*
	*        Some details:
	*		 1. The template parameter T can either be <sp_mat_t> or <den_mat_t>
	*/
	template<typename T>
	class RECompBase {
	public:
		/*! \brief Virtual destructor */
		virtual ~RECompBase() {};

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
		virtual void FindInitCovPar(vec_t& pars) = 0;

		/*!
		* \brief Virtual function that calculates Sigma (not needed for grouped REs, at unique locations for GPs)
		*/
		virtual void CalcSigma() = 0;

		/*!
		* \brief Virtual function that calculates the covariance matrix Z*Sigma*Z^T
		* \return Covariance matrix Z*Sigma*Z^T of this component
		*   Note that since sigma_ is saved (since it is used in GetZSigmaZt and GetZSigmaZtGrad) we return a pointer and do not write on an input paramter in order to avoid copying
		*/
		virtual std::shared_ptr<T> GetZSigmaZt() = 0;

		/*!
		* \brief Virtual function that calculates the derivatives of the covariance matrix Z*Sigma*Z^T
		* \param ind_par Index for parameter
		* \param transf_scale If true, the derivative is taken on the transformed scale otherwise on the original scale. Default = true
		* \param nugget_var Nugget effect variance parameter sigma^2 (used only if transf_scale = false to transform back)
		* \return Derivative of covariance matrix Z*Sigma*Z^T with respect to the parameter number ind_par
		*/
		virtual std::shared_ptr<T> GetZSigmaZtGrad(int ind_par = 0, bool transf_scale = true, double = 1.) = 0;

		/*!
		* \brief Virtual function that returns the matrix Z
		* \return A pointer to the matrix Z
		*/
		virtual sp_mat_t* GetZ() = 0;

		/*!
		* \brief Ignore this. It is only used for the class RECompGP and not for other derived classes. It is here in order that the base class can have this as a virtual method and no conversion needs to be made in the Vecchia approximation calculation (slightly a hack)
		*/
		virtual void CalcSigmaAndSigmaGrad(const den_mat_t& dist, den_mat_t& cov_mat,
			den_mat_t& cov_grad_1, den_mat_t& cov_grad_2,
			bool calc_gradient = false, bool transf_scale = true, double = 1.) = 0;

		/*!
		* \brief Returns number of covariance parameters
		* \return Number of covariance parameters
		*/
		int NumCovPar() {
			return(num_cov_par_);
		}

		/*!
		* \brief Calculate and add unconditional predictive variances
		* \param[out] pred_uncond_var Array of unconditional predictive variances to which the variance of this component is added
		* \param num_data_pred Number of prediction points
		* \param rand_coef_data_pred Covariate data for varying coefficients
		*/
		void AddPredUncondVar(double* pred_uncond_var, int num_data_pred, const double * const rand_coef_data_pred = nullptr) {
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
		/*! \brief Covariate data for varying coefficients */
		std::vector<double> rand_coef_data_;
		/*! \brief true if this is a random coefficient */
		bool is_rand_coef_;
		/*! \brief Covariance parameters (on transformed scale, but not logarithmic) */
		vec_t cov_pars_;

		template<typename T1, typename T2>
		friend class REModelTemplate;
	};

	/*!
	* \brief Class for the grouped random effect components
	*
	*        Some details:
	*/
	template<typename T>
	class RECompGroup : public RECompBase<T> {
	public:
		/*! \brief Constructor */
		RECompGroup();

		/*!
		* \brief Constructor without random coefficient data
		* \param group_data Group data: factorial variable between 1 and the number of different groups
		* \param num_data Number of data points
		* \param calculateZZt If true, the matrix Z*Z^T is calculated and saved (not needed if Woodbury identity is used)
		*/
		RECompGroup(std::vector<re_group_t>& group_data, bool calculateZZt = true) {
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
			// Create incidence matrix Z
			this->Z_.resize(this->num_data_, num_group_);
			std::vector<Triplet_t> triplets(this->num_data_);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < this->num_data_; ++i) {
				triplets[i] = Triplet_t(i, map_group_label_index[group_data[i]], 1);
			}
			this->Z_.setFromTriplets(triplets.begin(), triplets.end());
			// Alternative version: inserting elements directly
			// Note: compare to using triples, this is much slower when group_data is not ordered (e.g. [1,2,3,1,2,3]), otherwise if group_data is ordered (e.g. [1,1,2,2,3,3]) there is no big difference
			////this->Z_.reserve(Eigen::VectorXi::Constant(this->num_data_, 1));//don't use this, it makes things much slower
			//for (int i = 0; i < this->num_data_; ++i) {
			//	this->Z_.insert(i, map_group_label_index[group_data[i]]) = 1.;
			//}
			if (calculateZZt) {
				ConstructZZt<T>();
			}
			group_data_ = std::make_shared<std::vector<re_group_t>>(group_data);
			map_group_label_index_ = std::make_shared<std::map<re_group_t, int>>(map_group_label_index);
		}

		/*!
		* \brief Constructor for random coefficient effects
		* \param group_data Reference to group data of random intercept corresponding to this effect
		* \param num_group Number of groups / levels
		* \param rand_coef_data Covariate data for varying coefficients
		* \param calculateZZt If true, the matrix Z*Z^T is calculated and saved (not needed if Woodbury identity is used)
		*/
		RECompGroup(std::shared_ptr<std::vector<re_group_t>> group_data,
			std::shared_ptr<std::map<re_group_t, int>> map_group_label_index,
			data_size_t num_group, std::vector<double>& rand_coef_data, bool calculateZZt = true) {
			this->num_data_ = (data_size_t)(*group_data).size();
			num_group_ = num_group;
			group_data_ = group_data;
			map_group_label_index_ = map_group_label_index;
			this->rand_coef_data_ = rand_coef_data;
			this->is_rand_coef_ = true;
			this->num_cov_par_ = 1;
			this->Z_.resize(this->num_data_, num_group_);
			std::vector<Triplet_t> triplets(this->num_data_);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < this->num_data_; ++i) {
				triplets[i] = Triplet_t(i, (*map_group_label_index_)[(*group_data_)[i]], this->rand_coef_data_[i]);
			}
			this->Z_.setFromTriplets(triplets.begin(), triplets.end());
			//// Alternative version: inserting elements directly (see constructor above)
			//for (int i = 0; i < this->num_data_; ++i) {
			//	this->Z_.insert(i, (*map_group_label_index_)[(*group_data_)[i]]) = this->rand_coef_data_[i];
			//}
			if (calculateZZt) {
				ConstructZZt<T>();
			}
		}

		/*! \brief Destructor */
		~RECompGroup() {
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
		void FindInitCovPar(vec_t& pars) override {//TODO: find better initial estimates (as e.g. the variance of the group means)
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
		std::shared_ptr<T> GetZSigmaZt() override {
			if (this->cov_pars_.size() == 0) { Log::Fatal("Covariance parameters are not specified. Call 'SetCovPars' first."); }
			if (this->ZZt_.cols() == 0) { Log::Fatal("Matrix ZZt_ not defined"); }
			return(std::make_shared<T>(this->cov_pars_[0] * ZZt_));
		}

		/*!
		* \brief Calculate derivative of covariance matrix Z*Sigma*Z^T with respect to the parameter
		* \param ind_par Index for parameter (0=variance, 1=inverse range)
		* \param transf_scale If true, the derivative is taken on the transformed scale otherwise on the original scale. Default = true
		* \param nugget_var Nugget effect variance parameter sigma^2 (not use here)
		* \return Derivative of covariance matrix Z*Sigma*Z^T with respect to the parameter number ind_par
		*/
		std::shared_ptr<T> GetZSigmaZtGrad(int ind_par, bool transf_scale = true, double = 1.) override {
			if (this->cov_pars_.size() == 0) { Log::Fatal("Covariance parameters are not specified. Call 'SetCovPars' first."); }
			if (this->ZZt_.cols() == 0) { Log::Fatal("Matrix ZZt_ not defined"); }
			if (ind_par == 0) {
				double cm = transf_scale ? this->cov_pars_[0] : 1.;
				return(std::make_shared<T>(cm * ZZt_));
			}
			else {
				Log::Fatal("No covariance parameter for index number %d", ind_par);
			}
		}

		/*!
		* \brief Function that returns the matrix Z
		* \return A pointer to the matrix Z
		*/
		sp_mat_t* GetZ() override {
			return(&(this->Z_));
		}

		/*!
		* \brief Calculate covariance matrices needed for prediction
		* \param group_data_pred Group data for predictions
		* \param[out] pred_mats Add covariance matrices from this component to this parameter which contains covariance matrices needed for making predictions in the following order: 0. Ztilde*Sigma*Z^T, 1. Zstar*Sigmatilde^T*Z^T (=0 for grouped RE), 2. Ztilde*Sigma*Ztilde^T, 3. Ztilde*Sigmatilde*Zstar^T (=0 for grouped RE), 4. Zstar*Sigmastar*Zstar^T.
		* \param predict_cov_mat If true, all matrices are calculated. If false only Ztilde*Sigma*Z^T required for the conditional mean is calculated
		* \param rand_coef_data_pred Covariate data for varying coefficients
		*/
		void AddPredCovMatrices(const std::vector<re_group_t>& group_data_pred, std::vector<T>& pred_mats,
			bool predict_cov_mat = false, double* rand_coef_data_pred = nullptr) {
			int num_data_pred = (int)group_data_pred.size();
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
			T ZtildeZT;
			CalculateZ1Z2T<T>(Ztilde, this->Z_, ZtildeZT);
			pred_mats[0] += (this->cov_pars_[0] * ZtildeZT);

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
				sp_mat_t Zstar;
				Zstar.resize(num_data_pred, num_group_pred_new);
				Zstar.setZero();
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
				T ZtildeZtildeT;
				CalculateZ1Z2T<T>(Ztilde, Ztilde, ZtildeZtildeT);
				pred_mats[2] += (this->cov_pars_[0] * ZtildeZtildeT);
				T ZstarZstarT;
				CalculateZ1Z2T<T>(Zstar, Zstar, ZstarZstarT);
				pred_mats[4] += (this->cov_pars_[0] * ZstarZstarT);
			}
		}

		/*!
		* \brief Ignore this. This is not used for this class (it is only used for the class RECompGP). It is here in order that the base class can have this as a virtual method and no conversion needs to be made in the Vecchia approximation calculation (slightly a hack)
		*/
		void CalcSigmaAndSigmaGrad(const den_mat_t&, den_mat_t&,
			den_mat_t&, den_mat_t&,
			bool = false, bool = true, double = 1.) override {

		}

	private:
		/*! \brief Number of groups */
		data_size_t num_group_;
		/*! \brief Data with group labels / levels */
		std::shared_ptr<std::vector<re_group_t>> group_data_;
		/*! \brief Keys: Group labels, values: index number (integer value) for every group level. I.e., maps string labels to numbers */
		std::shared_ptr<std::map<re_group_t, int>> map_group_label_index_;
		/*! \brief Matrix Z*Z^T */
		T ZZt_;

		/*! \brief Constructs the matrix ZZt_ if sparse matrices are used */
		template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type * = nullptr >
		void ConstructZZt() {
			ZZt_ = this->Z_ * this->Z_.transpose();
		}

		/*! \brief Constructs the matrix ZZt_ if dense matrices are used */
		template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type * = nullptr >
		void ConstructZZt() {
			ZZt_ = den_mat_t(this->Z_ * this->Z_.transpose());
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

		///*! \brief Constructs the matrix Z_ if sparse matrices are used */
		//template <class T3, typename std::enable_if< std::is_same<sp_mat_t, T3>::value>::type* = nullptr  >
		//void ConstructZ() {
		//	std::vector<Triplet_t> entries;
		//	for (int i = 0; i < num_data_; ++i) {
		//		entries.push_back(Triplet_t(i, group_data_[i] - 1, 1.));
		//	}
		//	Z_.setFromTriplets(entries.begin(), entries.end());
		//}
		///*! \brief Constructs the matrix Z_ if dense matrices are used */
		//template <class T3, typename std::enable_if< std::is_same<den_mat_t, T3>::value>::type* = nullptr  >
		//void ConstructZ() {
		//	for (int i = 0; i < num_data_; ++i) {
		//		Z_(i, group_data_[i] - 1) = 1.;
		//	}
		//}

		template<typename T1, typename T2>
		friend class REModelTemplate;

		template<typename T2>
		friend class Likelihood;//for access of map_group_label_index_ and group_data_ in 'CalcGradNegMargLikelihoodLAApproxGroupedRE'
	};

	/*!
	* \brief Class for the Gaussian process components
	*
	*        Some details:
	*        ...
	*/
	template<typename T>
	class RECompGP : public RECompBase<T> {
	public:
		/*! \brief Constructor */
		RECompGP();

		/*!
		* \brief Constructor for Gaussian process
		* \param coords Coordinates (features) for Gaussian process
		* \param cov_fct Type of covariance function
		* \param shape Shape parameter of covariance function (=smoothness parameter for Matern covariance, irrelevant for some covariance functions such as the exponential or Gaussian)
		* \param save_dist_use_Z_for_duplicates If true, distances are calculated and saved here, and an incidendce matrix Z is used for duplicate locations. save_dist_use_Z_for_duplicates = false is used for the Vecchia approximation which saves the required distances in the REModel (REModelTemplate)
		*/
		RECompGP(const den_mat_t& coords, string_t cov_fct = "exponential",
			double shape = 0., bool save_dist_use_Z_for_duplicates = true) {
			this->num_data_ = (data_size_t)coords.rows();
			this->is_rand_coef_ = false;
			has_Z_ = false;
			this->num_cov_par_ = 2;
			cov_function_ = std::unique_ptr<CovFunction<T>>(new CovFunction<T>(cov_fct, shape));
			if (save_dist_use_Z_for_duplicates) {
				std::vector<int> uniques;//unique points
				std::vector<int> unique_idx;//used for constructing incidence matrix Z_ if there are duplicates
				DetermineUniqueDuplicateCoords(coords, this->num_data_, uniques, unique_idx);
				if ((data_size_t)uniques.size() == this->num_data_) {//no multiple observations at the same locations -> no incidence matrix needed
					coords_ = coords;
				}
				else {
					coords_ = coords(uniques, Eigen::all);
					this->Z_.resize(this->num_data_, uniques.size());
					this->Z_.setZero();
					for (int i = 0; i < this->num_data_; ++i) {
						this->Z_.insert(i, unique_idx[i]) = 1.;
					}
					has_Z_ = true;
				}
				//Calculate distances
				den_mat_t dist;
				CalculateDistances(coords_, dist);
				dist_ = std::make_shared<den_mat_t>(dist);
				dist_saved_ = true;
			}
			else {//this option is used for the Vecchia approximation
				coords_ = coords;
				dist_saved_ = false;
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
		*/
		RECompGP(std::shared_ptr<den_mat_t> dist, bool base_effect_has_Z, sp_mat_t* Z,
			const std::vector<double>& rand_coef_data, string_t cov_fct = "exponential", double shape = 0.) {
			this->num_data_ = (data_size_t)rand_coef_data.size();
			dist_ = dist;
			dist_saved_ = true;
			this->rand_coef_data_ = rand_coef_data;
			this->is_rand_coef_ = true;
			has_Z_ = true;
			this->num_cov_par_ = 2;
			cov_function_ = std::unique_ptr<CovFunction<T>>(new CovFunction<T>(cov_fct, shape));
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
		}

		/*!
		* \brief Constructor for random coefficient Gaussian process when multiple locations are not modelled using an incidence matrix.
		*		This is used for the Vecchia approximation.
		* \param rand_coef_data Covariate data for random coefficient
		* \param cov_fct Type of covariance function
		* \param shape Shape parameter of covariance function (=smoothness parameter for Matern covariance, irrelevant for some covariance functions such as the exponential or Gaussian)
		*/
		RECompGP(const std::vector<double>& rand_coef_data, string_t cov_fct = "exponential", double shape = 0.) {
			this->rand_coef_data_ = rand_coef_data;
			this->is_rand_coef_ = true;
			this->num_data_ = (data_size_t)rand_coef_data.size();
			has_Z_ = true;
			this->num_cov_par_ = 2;
			cov_function_ = std::unique_ptr<CovFunction<T>>(new CovFunction<T>(cov_fct, shape));
			dist_saved_ = false;
			coord_saved_ = false;
			this->Z_ = sp_mat_t(this->num_data_, this->num_data_);
			for (int i = 0; i < this->num_data_; ++i) {
				this->Z_.insert(i, i) = this->rand_coef_data_[i];
			}
		}

		/*! \brief Destructor */
		~RECompGP() {
		}

		/*!
		* \brief Function that sets the covariance parameters
		* \param pars Vector of length 2 with covariance parameters (variance and inverse range)
		*/
		void SetCovPars(const vec_t& pars) override {
			CHECK((int)pars.size() == 2);
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
		void FindInitCovPar(vec_t& pars) override {
			if (!dist_saved_ && !coord_saved_) {
				Log::Fatal("Cannot determine initial covariance parameters if neither distances nor coordinates are given");
			}
			pars[0] = 1;
			double mean_dist = 0;
			if (!dist_saved_) {//Calculate distances (of a Bootstrap sample) in case they have not been calculated (for the Vecchia approximation)
				int num_coord = (int)coords_.rows();
				den_mat_t dist;
				int num_data = (num_coord > 1000) ? 1000 : num_coord;//limit to maximally 1000 to save computational time
				if (num_data < num_coord) {
					std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with 0
					std::uniform_int_distribution<> dis(0, num_coord - 1);

					std::vector<int> sample_ind(num_data);
					for (int i = 0; i < num_data; ++i) {
						sample_ind[i] = dis(gen);
					}
					CalculateDistances(coords_(sample_ind, Eigen::all), dist);
				}
				else {
					CalculateDistances(coords_, dist);
				}
				for (int i = 0; i < (num_data - 1); ++i) {
					for (int j = i + 1; j < num_data; ++j) {
						mean_dist += dist(i, j);
					}
				}
				mean_dist /= (num_data * (num_data - 1) / 2.);
			}
			else {
				int num_coord = (int)(*dist_).rows();
				for (int i = 0; i < (num_coord - 1); ++i) {
					for (int j = i + 1; j < num_coord; ++j) {
						mean_dist += (*dist_)(i, j);
					}
				}
				mean_dist /= (num_coord * (num_coord - 1) / 2.);
			}
			//Set the range parameter such that the correlation is down to 0.05 at the mean distance
			if (cov_function_->cov_fct_type_ == "exponential" || cov_function_->cov_fct_type_ == "matern") {//TODO: find better intial values for matern covariance for shape = 1.5 and shape = 2.5
				pars[1] = 3. / mean_dist;//pars[1] = 1/range
			}
			else if (cov_function_->cov_fct_type_ == "gaussian") {
				pars[1] = 3. / std::pow(mean_dist, 2.);//pars[1] = 1/range^2
			}
			else if (cov_function_->cov_fct_type_ == "powered_exponential") {
				pars[1] = 3. / std::pow(mean_dist, cov_function_->shape_);//pars[1] = 1/range^shape
			}
			else {
				Log::Fatal("Finding initial values for covariance paramters for covariance of type '%s' is not supported.", cov_function_->cov_fct_type_.c_str());
			}
		}

		/*!
		* \brief Calculate covariance matrix at unique locations
		*/
		void CalcSigma() override {
			if (this->cov_pars_.size() == 0) { Log::Fatal("Covariance parameters are not specified. Call 'SetCovPars' first."); }
			(*cov_function_).template GetCovMat<T>(*dist_, this->cov_pars_, sigma_);
			//cov_function_->GetCovMat<T>(*dist_, this->cov_pars_, sigma_);//does not work for mingw compiler, thus use code above
			sigma_defined_ = true;
		}

		/*!
		* \brief Calculate covariance matrix
		* \return Covariance matrix Z*Sigma*Z^T of this component
		*/
		std::shared_ptr<T> GetZSigmaZt() override {
			if (!sigma_defined_) {
				Log::Fatal("Sigma has not been calculated");
			}
			if (this->is_rand_coef_ || has_Z_) {
				return(std::make_shared<T>(this->Z_ * sigma_ * this->Z_.transpose()));
			}
			else {
				return(std::make_shared<T>(sigma_));
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
			bool calc_gradient = false, bool transf_scale = true, double nugget_var = 1.) override {
			if (this->cov_pars_.size() == 0) { Log::Fatal("Covariance parameters are not specified. Call 'SetCovPars' first."); }
			(*cov_function_).template GetCovMat<den_mat_t>(dist, this->cov_pars_, cov_mat);
			if (calc_gradient) {
				//gradient wrt to variance parameter
				cov_grad_1 = cov_mat;
				if (!transf_scale) {
					cov_grad_1 /= this->cov_pars_[0];
				}
				//gradient wrt to range parameter
				(*cov_function_).template GetCovMatGradRange<den_mat_t>(dist, cov_mat, this->cov_pars_, cov_grad_2, transf_scale, nugget_var);
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
		std::shared_ptr<T> GetZSigmaZtGrad(int ind_par, bool transf_scale = true, double nugget_var = 1.) override {
			if (!sigma_defined_) {
				Log::Fatal("Sigma has not been calculated");
			}
			if (ind_par == 0) {//variance
				if (transf_scale) {
					return(GetZSigmaZt());
				}
				else {
					double correct = 1. / this->cov_pars_[0];//divide sigma_ by cov_pars_[0]
					if (this->is_rand_coef_ || has_Z_) {
						return(std::make_shared<T>(correct * this->Z_ * sigma_ * this->Z_.transpose()));
					}
					else {
						return(std::make_shared<T>(correct * sigma_));
					}
				}
			}
			else if (ind_par == 1) {//inverse range
				T Z_sigma_grad_Zt;
				if (has_Z_) {
					T sigma_grad;
					(*cov_function_).template GetCovMatGradRange<T>(*dist_, sigma_, this->cov_pars_, sigma_grad, transf_scale, nugget_var);
					Z_sigma_grad_Zt = this->Z_ * sigma_grad * this->Z_.transpose();
				}
				else {
					(*cov_function_).template GetCovMatGradRange<T>(*dist_, sigma_, this->cov_pars_, Z_sigma_grad_Zt, transf_scale, nugget_var);
				}
				return(std::make_shared<T>(Z_sigma_grad_Zt));
			}
			else {
				Log::Fatal("No covariance parameter for index number %d", ind_par);
			}
		}

		/*!
		* \brief Function that returns the matrix Z
		* \return A pointer to the matrix Z
		*/
		sp_mat_t* GetZ() override {
			if (!has_Z_) {
				Log::Fatal("Gaussian process has no matrix Z");
			}
			return(&(this->Z_));
		}

		/*!
		* \brief Calculate covariance matrices needed for prediction
		* \param coords Coordinates for observed data
		* \param coords_pred Coordinates for predictions
		* \param[out] pred_mats Add covariance matrices from this component to this parameter which contains covariance matrices needed for making predictions in the following order: 0. Ztilde*Sigma*Z^T, 1. Zstar*Sigmatilde^T*Z^T (=0 for grouped RE), 2. Ztilde*Sigma*Ztilde^T, 3. Ztilde*Sigmatilde*Zstar^T (=0 for grouped RE), 4. Zstar*Sigmastar*Zstar^T.
		* \param predict_cov_mat If true, all matrices are calculated. If false only Ztilde*Sigma*Z^T required for the conditional mean is calculated
		* \param rand_coef_data_pred Covariate data for varying coefficients
		*/
		void AddPredCovMatrices(const den_mat_t& coords, const den_mat_t& coords_pred, std::vector<T>& pred_mats,
			bool predict_cov_mat = false, double* rand_coef_data_pred = nullptr) {
			int num_data_pred = (int)coords_pred.rows();
			std::vector<int>  uniques_pred;//unique points
			std::vector<int>  unique_idx_pred;//used for constructing incidence matrix Z_ if there are duplicates
			DetermineUniqueDuplicateCoords(coords_pred, num_data_pred, uniques_pred, unique_idx_pred);
			//Create matrix Zstar
			sp_mat_t Zstar(num_data_pred, uniques_pred.size());
			Zstar.setZero();
			den_mat_t coords_pred_unique;
			bool has_duplicates = (int)uniques_pred.size() != num_data_pred;
			if (has_duplicates) {//Only keep unique coordinates if there are multiple observations with the same coordinates
				coords_pred_unique = coords_pred(uniques_pred, Eigen::all);
			}
			for (int i = 0; i < num_data_pred; ++i) {
				if (this->is_rand_coef_) {
					Zstar.insert(i, unique_idx_pred[i]) = rand_coef_data_pred[i];
				}
				else {
					Zstar.insert(i, unique_idx_pred[i]) = 1.;
				}
			}
			//Calculate cross distances between "existing" and "new" points
			den_mat_t cross_dist((int)uniques_pred.size(), coords.rows());
			cross_dist.setZero();
			for (int i = 0; i < coords.rows(); ++i) {
				for (int j = 0; j < (int)uniques_pred.size(); ++j) {
					if (has_duplicates) {
						cross_dist(j, i) = (coords.row(i) - coords_pred_unique.row(j)).lpNorm<2>();
					}
					else {
						cross_dist(j, i) = (coords.row(i) - coords_pred.row(j)).lpNorm<2>();
					}
				}
			}
			T ZstarSigmatildeTZT;
			T Sigmatilde;
			(*cov_function_).template GetCovMat<T>(cross_dist, this->cov_pars_, Sigmatilde);
			if (this->has_Z_) {
				ZstarSigmatildeTZT = Zstar * Sigmatilde * this->Z_.transpose();
			}
			else {
				ZstarSigmatildeTZT = Zstar * Sigmatilde;
			}
			pred_mats[1] += ZstarSigmatildeTZT;
			if (predict_cov_mat) {
				den_mat_t dist;
				CalculateDistances(coords_pred, dist);
				T Sigmastar;
				(*cov_function_).template GetCovMat<T>(dist, this->cov_pars_, Sigmastar);
				T ZstarSigmastarZstarT = Zstar * Sigmastar * Zstar.transpose();
				pred_mats[4] += ZstarSigmastarZstarT;
			}
		}

	private:
		/*! \brief Coordinates (=features) */
		den_mat_t coords_;
		/*! \brief Distance matrix (between unique coordinates in coords_) */
		std::shared_ptr<den_mat_t> dist_;
		/*! \brief If true, the distancess among all observations are calculated and saved here (false for Vecchia approximation) */
		bool dist_saved_ = true;
		/*! \brief If true, the coordinates are saved (false for random coefficients GPs) */
		bool coord_saved_ = true;
		/*! \brief Indicates whether the GP has a non-identity incidence matrix Z */
		bool has_Z_;
		/*! \brief Covariance function */
		std::unique_ptr<CovFunction<T>> cov_function_;
		/*! \brief Covariance matrix (for a certain choice of covariance paramters). This is saved for re-use at two locations in the code: GetZSigmaZt and GetZSigmaZtGrad) */
		T sigma_;
		/*! \brief Indicates whether sigma_ has been defined or not */
		bool sigma_defined_ = false;

		template<typename T1, typename T2>
		friend class REModelTemplate;
	};

}  // namespace GPBoost

#endif   // GPB_RE_COMP_H_
