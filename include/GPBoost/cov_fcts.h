/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_COV_FUNCTIONS_
#define GPB_COV_FUNCTIONS_

#include <GPBoost/type_defs.h>
#include <GPBoost/utils.h>
#include <GPBoost/sparse_matrix_utils.h>
#include <GPBoost/GP_utils.h>

#include <string>
#include <set>
#include <string>
#include <vector>
#include <cmath>

#include <LightGBM/utils/log.h>
using LightGBM::Log;


namespace GPBoost {

	template<typename T_mat>
	class RECompGP;

	/*!
	* \brief This class implements the covariance functions used for the Gaussian proceses
	*/
	class CovFunction {
	public:
		/*! \brief Constructor */
		CovFunction();

		/*!
		* \brief Constructor
		* \param cov_fct_type Type of covariance function
		* \param shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. This parameter is irrelevant for some covariance functions such as the exponential or Gaussian
		* \param taper_range Range parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
		* \param taper_shape Shape parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
		* \param taper_mu Parameter \mu of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
		* \param apply_tapering If true, tapering is applied to the covariance function (element-wise multiplication with a compactly supported Wendland correlation function)
		*/
		CovFunction(string_t cov_fct_type,
			double shape,
			double taper_range,
			double taper_shape,
			double taper_mu,
			bool apply_tapering) {
			if (cov_fct_type == "exponential_tapered") {
				Log::REFatal("Covariance of type 'exponential_tapered' is discontinued. Use the option 'gp_approx = \"tapering\"' instead ");
			}
			if (SUPPORTED_COV_TYPES_.find(cov_fct_type) == SUPPORTED_COV_TYPES_.end()) {
				Log::REFatal("Covariance of type '%s' is not supported ", cov_fct_type.c_str());
			}
			if (cov_fct_type == "matern_space_time") {
				save_distances_ = false;
			}
			else {
				save_distances_ = true;
			}
			if (cov_fct_type == "matern_space_time") {
				num_cov_par_ = 3;
			}
			else if (cov_fct_type == "wendland") {
				num_cov_par_ = 1;
			}
			else {
				num_cov_par_ = 2;
			}
			cov_fct_type_ = cov_fct_type;
			shape_ = shape;
			if (cov_fct_type == "matern") {
				if (!(TwoNumbersAreEqual<double>(shape, 0.5) || TwoNumbersAreEqual<double>(shape, 1.5) || TwoNumbersAreEqual<double>(shape, 2.5))) {
					Log::REFatal("'shape' of %g is not supported for the '%s' covariance function. Only shape / smoothness parameters 0.5, 1.5, and 2.5 are currently implemented ", shape, cov_fct_type.c_str());
				}
			}
			else if (cov_fct_type == "powered_exponential") {
				if (shape <= 0. || shape > 2.) {
					Log::REFatal("'shape' needs to be larger than 0 and smaller or equal than 2 for the '%s' covariance function, found %g ", cov_fct_type.c_str(), shape);
				}
			}
			if (cov_fct_type == "wendland" || apply_tapering) {
				if (!(TwoNumbersAreEqual<double>(taper_shape, 0.0) || TwoNumbersAreEqual<double>(taper_shape, 1.0) || TwoNumbersAreEqual<double>(taper_shape, 2.0))) {
					Log::REFatal("'taper_shape' of %g is not supported for the 'wendland' covariance function or correlation tapering function. Only shape / smoothness parameters 0, 1, and 2 are currently implemented ", taper_shape);
				}
				CHECK(taper_range > 0.);
				CHECK(taper_mu >= 1.);
				taper_range_ = taper_range;
				taper_shape_ = taper_shape;
				taper_mu_ = taper_mu;
				apply_tapering_ = true;
			}
		}

		/*! \brief Destructor */
		~CovFunction() {
		}

		bool ShouldSaveDistances() const {
			return(save_distances_);
		}

		bool IsSpaceTimeModel() const {
			return(cov_fct_type_ == "matern_space_time");
		}

		/*! \brief Dimension of spatial coordinates in for space-time models */
		int GetDimSpace(const den_mat_t& coords) const {
			int dim_space = (int)coords.cols();
			if (cov_fct_type_ == "matern_space_time") {
				dim_space = (int)coords.cols() - 1;
			}
			return(dim_space);
		}

		/*!
		* \brief Transform the covariance parameters
		* \param sigma2 Marginal variance
		* \param pars Vector with covariance parameters on orignal scale
		* \param[out] pars_trans Transformed covariance parameters
		*/
		void TransformCovPars(const double sigma2,
			const vec_t& pars,
			vec_t& pars_trans) const {
			pars_trans = pars;
			pars_trans[0] = pars[0] / sigma2;
			if (cov_fct_type_ == "exponential" ||
				(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
				pars_trans[1] = 1. / pars[1];
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				pars_trans[1] = sqrt(3.) / pars[1];
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				pars_trans[1] = sqrt(5.) / pars[1];
			}
			else if (cov_fct_type_ == "gaussian") {
				pars_trans[1] = 1. / (pars[1] * pars[1]);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				pars_trans[1] = 1. / (std::pow(pars[1], shape_));
			}
			else if (cov_fct_type_ == "matern_space_time") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					pars_trans[1] = 1. / pars[1];
					pars_trans[2] = 1. / pars[2];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					pars_trans[1] = sqrt(3.) / pars[1];
					pars_trans[2] = sqrt(3.) / pars[2];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					pars_trans[1] = sqrt(5.) / pars[1];
					pars_trans[2] = sqrt(5.) / pars[2];
				}
			}
		}

		/*!
		* \brief Function transforms the covariance parameters back to the original scale
		* \param sigma2 Marginal variance
		* \param pars Vector with covariance parameters
		* \param[out] pars_orig Back-transformed, original covariance parameters
		*/
		void TransformBackCovPars(const double sigma2,
			const vec_t& pars,
			vec_t& pars_orig) const {
			pars_orig = pars;
			pars_orig[0] = sigma2 * pars[0];
			if (cov_fct_type_ == "exponential" ||
				(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
				pars_orig[1] = 1. / pars[1];
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				pars_orig[1] = sqrt(3.) / pars[1];
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				pars_orig[1] = sqrt(5.) / pars[1];
			}
			else if (cov_fct_type_ == "gaussian") {
				pars_orig[1] = 1. / std::sqrt(pars[1]);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				pars_orig[1] = 1. / (std::pow(pars[1], 1. / shape_));
			}
			else if (cov_fct_type_ == "matern_space_time") {				
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					pars_orig[1] = 1. / pars[1];
					pars_orig[2] = 1. / pars[2];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					pars_orig[1] = sqrt(3.) / pars[1];
					pars_orig[2] = sqrt(3.) / pars[2];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					pars_orig[1] = sqrt(5.) / pars[1];
					pars_orig[2] = sqrt(5.) / pars[2];
				}
			}
		}

		/*!
		* \brief Calculates covariance matrix
		* \param dist Distance matrix
		* \param coords Coordinate matrix
		* \param coords_pred Second set of coordinates for predictions
		* \param pars Vector with covariance parameters
		* \param[out] sigma Covariance matrix
		* \param is_symmmetric Set to true if dist and sigma are symmetric (e.g., for training data)
		*/
		template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
		void GetCovMat(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const vec_t& pars,
			T_mat& sigma,
			bool is_symmmetric) const {
			CHECK(pars.size() == num_cov_par_);
			if (save_distances_) {
				sigma = T_mat(dist.rows(), dist.cols());
			}
			else {
				if (is_symmmetric) {
					sigma = T_mat(coords.rows(), coords.rows());
				}
				else {
					sigma = T_mat(coords_pred.rows(), coords.rows());
				}
			}
			int dim_space = (int)coords.cols();
			if (cov_fct_type_ == "matern_space_time") {
				dim_space = (int)coords.cols() - 1;
			}
			if (cov_fct_type_ == "exponential" ||
				(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						sigma(i, i) = pars[0];
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) = MaternCovarianceShape0_5(dist(i, j), pars[0], pars[1]);
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) = MaternCovarianceShape0_5(dist(i, j), pars[0], pars[1]);
						}
					}
				}
			}//end cov_fct_type_ == "exponential"
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						sigma(i, i) = pars[0];
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) = MaternCovarianceShape1_5(dist(i, j), pars[0], pars[1]);
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) = MaternCovarianceShape1_5(dist(i, j), pars[0], pars[1]);
						}
					}
				}
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						sigma(i, i) = pars[0];
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) = MaternCovarianceShape2_5(dist(i, j), pars[0], pars[1]);
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) = MaternCovarianceShape2_5(dist(i, j), pars[0], pars[1]);
						}
					}
				}
			}//end cov_fct_type_ == "matern"
			else if (cov_fct_type_ == "gaussian") {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						sigma(i, i) = pars[0];
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) = GaussianCovariance(dist(i, j), pars[0], pars[1]);
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) = GaussianCovariance(dist(i, j), pars[0], pars[1]);
						}
					}
				}
			}//end cov_fct_type_ == "gaussian"
			else if (cov_fct_type_ == "powered_exponential") {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						sigma(i, i) = pars[0];
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) = PoweredExponentialCovariance(dist(i, j), pars[0], pars[1]);
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) = PoweredExponentialCovariance(dist(i, j), pars[0], pars[1]);
						}
					}
				}
			}//end cov_fct_type_ == "powered_exponential"
			else if (cov_fct_type_ == "wendland") {
				// note: this dense matrix version is usually not used
				// initialize Wendland covariance matrix
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)dist.rows(); ++i) {
					for (int j = 1; j < (int)dist.cols(); ++j) {
						if (dist(i, j) >= taper_range_) {
							sigma(i, j) = 0.;
						}
						else {
							sigma(i, j) = pars[0];
						}
					}
				}
				MultiplyWendlandCorrelationTaper<T_mat>(dist, sigma, is_symmmetric);
			}//end cov_fct_type_ == "wendland"
			else if (cov_fct_type_ == "matern_space_time"){
				den_mat_t coords_scaled(coords.rows(), coords.cols()), coords_pred_scaled;
				coords_scaled.col(0) = coords.col(0) * pars[1];
				coords_scaled.rightCols(dim_space) = coords.rightCols(dim_space) * pars[2];
				if (!is_symmmetric) {
					coords_pred_scaled = den_mat_t(coords_pred.rows(), coords_pred.cols());
					coords_pred_scaled.col(0) = coords_pred.col(0) * pars[1];
					coords_pred_scaled.rightCols(dim_space) = coords_pred.rightCols(dim_space) * pars[2];
				}
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					if (is_symmmetric) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords.rows(); ++i) {
							sigma(i, i) = pars[0];
							for (int j = i + 1; j < (int)coords.rows(); ++j) {
								double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								sigma(i, j) = MaternCovarianceShape0_5(dist_ij, pars[0], 1.);
								sigma(j, i) = sigma(i, j);
							}
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords_pred.rows(); ++i) {
							for (int j = 0; j < (int)coords.rows(); ++j) {
								double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								sigma(i, j) = MaternCovarianceShape0_5(dist_ij, pars[0], 1.);
							}
						}
					}
				}//end TwoNumbersAreEqual<double>(shape_, 0.5)
			else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					if (is_symmmetric) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords.rows(); ++i) {
							sigma(i, i) = pars[0];
							for (int j = i + 1; j < (int)coords.rows(); ++j) {
								double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								sigma(i, j) = MaternCovarianceShape1_5(dist_ij, pars[0], 1.);
								sigma(j, i) = sigma(i, j);
							}
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords_pred.rows(); ++i) {
							for (int j = 0; j < (int)coords.rows(); ++j) {
								double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								sigma(i, j) = MaternCovarianceShape1_5(dist_ij, pars[0], 1.);
							}
						}
					}
			}//end TwoNumbersAreEqual<double>(shape_, 1.5)
			else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					if (is_symmmetric) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords.rows(); ++i) {
							sigma(i, i) = pars[0];
							for (int j = i + 1; j < (int)coords.rows(); ++j) {
								double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								sigma(i, j) = MaternCovarianceShape2_5(dist_ij, pars[0], 1.);
								sigma(j, i) = sigma(i, j);
							}
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords_pred.rows(); ++i) {
							for (int j = 0; j < (int)coords.rows(); ++j) {
								double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								sigma(i, j) = MaternCovarianceShape2_5(dist_ij, pars[0], 1.);
							}
						}
					}
				}//end TwoNumbersAreEqual<double>(shape_, 2.5)
			}//end cov_fct_type_ == "matern_space_time"
			else {
				Log::REFatal("Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end GetCovMat (dense)
		template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
		void GetCovMat(const T_mat& dist,
			const den_mat_t&,
			const den_mat_t&,
			const vec_t& pars,
			T_mat& sigma,
			bool is_symmmetric) const {
			if (cov_fct_type_ == "matern_space_time") {
				Log::REFatal("GetCovMat is not implemented when cov_fct_type_ == 'matern_space_time'' and sparse matrices are used ");
			}
			CHECK(pars.size() == num_cov_par_);
			sigma = dist;
			if (cov_fct_type_ == "exponential" ||
				(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = pars[0];
							}
							else if (i < j) {
								it.valueRef() = MaternCovarianceShape0_5(dist.coeff(i, j), pars[0], pars[1]);
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							it.valueRef() = MaternCovarianceShape0_5(dist.coeff(i, j), pars[0], pars[1]);
						}
					}
				}
			}//end cov_fct_type_ == "exponential"
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = pars[0];
							}
							else if (i < j) {
								it.valueRef() = MaternCovarianceShape1_5(dist.coeff(i, j), pars[0], pars[1]);
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							it.valueRef() = MaternCovarianceShape1_5(dist.coeff(i, j), pars[0], pars[1]);
						}
					}
				}
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = pars[0];
							}
							else if (i < j) {
								it.valueRef() = MaternCovarianceShape2_5(dist.coeff(i, j), pars[0], pars[1]);
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							it.valueRef() = MaternCovarianceShape2_5(dist.coeff(i, j), pars[0], pars[1]);
						}
					}
				}
			}//end cov_fct_type_ == "matern"
			else if (cov_fct_type_ == "gaussian") {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = pars[0];
							}
							else if (i < j) {
								it.valueRef() = GaussianCovariance(dist.coeff(i, j), pars[0], pars[1]);
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							it.valueRef() = GaussianCovariance(dist.coeff(i, j), pars[0], pars[1]);
						}
					}
				}
			}//end cov_fct_type_ == "gaussian"
			else if (cov_fct_type_ == "powered_exponential") {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = pars[0];
							}
							else if (i < j) {
								it.valueRef() = PoweredExponentialCovariance(dist.coeff(i, j), pars[0], pars[1]);
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							it.valueRef() = PoweredExponentialCovariance(dist.coeff(i, j), pars[0], pars[1]);
						}
					}
				}
			}//end cov_fct_type_ == "powered_exponential"
			else if (cov_fct_type_ == "wendland") {
				sigma.coeffs() = pars[0];
				MultiplyWendlandCorrelationTaper<T_mat>(dist, sigma, is_symmmetric);
			}
			else {
				Log::REFatal("Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end GetCovMat (sparse)

		/*!
		* \brief Covariance function for one distance value
		* \param dist Distance 
		* \param pars Vector with covariance parameters
		* \param[out] sigma Covariance at dist
		*/
		void GetCovMat(const double& dist,
			const vec_t& pars,
			double& sigma) const {
			if (cov_fct_type_ == "matern_space_time") {
				Log::REFatal("GetCovMat is not implemented for one distance value when cov_fct_type_ == 'matern_space_time'' ");
			}
			CHECK(pars.size() == num_cov_par_);
			if (cov_fct_type_ == "exponential" ||
				(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
				sigma = MaternCovarianceShape0_5(dist, pars[0], pars[1]);
			}//end cov_fct_type_ == "exponential"
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				sigma = MaternCovarianceShape1_5(dist, pars[0], pars[1]);
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				sigma = MaternCovarianceShape2_5(dist, pars[0], pars[1]);
			}//end cov_fct_type_ == "matern"
			else if (cov_fct_type_ == "gaussian") {
				sigma = GaussianCovariance(dist, pars[0], pars[1]);
			}//end cov_fct_type_ == "gaussian"
			else if (cov_fct_type_ == "powered_exponential") {
				sigma = PoweredExponentialCovariance(dist, pars[0], pars[1]);
			}//end cov_fct_type_ == "powered_exponential"
			else if (cov_fct_type_ == "wendland") {
				// note: this dense matrix version is usually not used
				if (dist >= taper_range_) {
					sigma = 0.;
				}
				else {
					sigma = pars[0];
					MultiplyWendlandCorrelationTaper(dist, sigma);
				}
			}//end cov_fct_type_ == "wendland"
			else {
				Log::REFatal("Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end GetCovMat (double)

		/*!
		* \brief Multiply covariance matrix element-wise with Wendland correlation tapering function
		* \param dist Distance matrix
		* \param[out] sigma Covariance matrix
		* \param is_symmmetric Set to true if dist and sigma are symmetric (e.g. for training data)
		*/
		template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
		void MultiplyWendlandCorrelationTaper(const T_mat& dist,
			T_mat& sigma,
			bool is_symmmetric) const {
			CHECK(apply_tapering_);
			if (TwoNumbersAreEqual<double>(taper_shape_, 0.)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) *= WendlandCorrelationShape0(dist(i, j));
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) *= WendlandCorrelationShape0(dist(i, j));
						}
					}
				}
			}
			else if (TwoNumbersAreEqual<double>(taper_shape_, 1.)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) *= WendlandCorrelationShape1(dist(i, j));
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) *= WendlandCorrelationShape1(dist(i, j));
						}
					}
				}
			}
			else if (TwoNumbersAreEqual<double>(taper_shape_, 2.)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) *= WendlandCorrelationShape2(dist(i, j));
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) *= WendlandCorrelationShape2(dist(i, j));
						}
					}
				}
			}
			else {
				Log::REFatal("'taper_shape' of %g is not supported for the 'wendland' covariance function or correlation tapering function. Only shape / smoothness parameters 0, 1, and 2 are currently implemented ", taper_shape_);
			}
		}//end MultiplyWendlandCorrelationTaper (dense)
		template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
		void MultiplyWendlandCorrelationTaper(const T_mat& dist,
			T_mat& sigma,
			bool is_symmmetric) const {
			CHECK(apply_tapering_);
			if (TwoNumbersAreEqual<double>(taper_shape_, 0.)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i < j) {
								it.valueRef() *= WendlandCorrelationShape0(dist.coeff(i, j));
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							it.valueRef() *= WendlandCorrelationShape0(dist.coeff(i, j));
						}
					}
				}
			}
			else if (TwoNumbersAreEqual<double>(taper_shape_, 1.)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i < j) {
								it.valueRef() *= WendlandCorrelationShape1(dist.coeff(i, j));
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							it.valueRef() *= WendlandCorrelationShape1(dist.coeff(i, j));
						}
					}
				}
			}
			else if (TwoNumbersAreEqual<double>(taper_shape_, 2.)) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i < j) {
								it.valueRef() *= WendlandCorrelationShape2(dist.coeff(i, j));
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							it.valueRef() *= WendlandCorrelationShape2(dist.coeff(i, j));
						}
					}
				}
			}
			else {
				Log::REFatal("'taper_shape' of %g is not supported for the 'wendland' covariance function or correlation tapering function. Only shape / smoothness parameters 0, 1, and 2 are currently implemented ", taper_shape_);
			}
		}//end MultiplyWendlandCorrelationTaper (sparse)

		/*!
		* \brief Multiply covariance with Wendland correlation tapering function for one distance value
		* \param dist Distance
		* \param[out] sigma Covariance at dist after applying tapering
		*/
		void MultiplyWendlandCorrelationTaper(const double& dist,
			double& sigma) const {
			CHECK(apply_tapering_);
			if (TwoNumbersAreEqual<double>(taper_shape_, 0.)) {
				sigma *= WendlandCorrelationShape0(dist);
			}
			else if (TwoNumbersAreEqual<double>(taper_shape_, 1.)) {
				sigma *= WendlandCorrelationShape1(dist);
			}
			else if (TwoNumbersAreEqual<double>(taper_shape_, 2.)) {
				sigma *= WendlandCorrelationShape2(dist);
			}
			else {
				Log::REFatal("'taper_shape' of %g is not supported for the 'wendland' covariance function or correlation tapering function. Only shape / smoothness parameters 0, 1, and 2 are currently implemented ", taper_shape_);
			}
		}//end MultiplyWendlandCorrelationTaper (double)

		/*!
		* \brief Calculates derivatives of the covariance matrix with respect to the range parameters
		* \param dist Distance matrix
		* \param coords Coordinate matrix
		* \param coords_pred Second set of coordinates for predictions
		* \param sigma Covariance matrix
		* \param pars Vector with covariance parameters on the transformed scale (no matter whether 'transf_scale' is true or not)
		* \param[out] sigma_grad Derivative of covariance matrix with respect to the range parameter
		* \param transf_scale If true, the derivative is taken on the transformed and logarithmic scale otherwise with respect to the original range parameter (the parameters values pars are always given on the transformed scale, but not logarithmized). Optimiziation is done using transf_scale=true. transf_scale=false is needed, for instance, for calcualting the Fisher information on the original scale.
		* \param marg_var Marginal variance parameters sigma^2 (used only if transf_scale = false to transform back)
		* \param ind_range Which range parameter (if there are multiple)
		* \param is_symmmetric Set to true if dist and sigma are symmetric (e.g., for training data)
		*/
		template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
		void GetCovMatGradRange(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const T_mat& sigma,
			const vec_t& pars,
			T_mat& sigma_grad,
			bool transf_scale,
			double marg_var,
			int ind_range,
			bool is_symmmetric) const {
			CHECK(pars.size() == num_cov_par_);
			if (save_distances_) {
				CHECK(sigma.cols() == dist.cols());
				CHECK(sigma.rows() == dist.rows());
			}
			else {
				if (is_symmmetric) {
					CHECK(sigma.rows() == coords.rows());
					CHECK(sigma.cols() == coords.rows());
				}
				else {
					CHECK(sigma.rows() == coords_pred.rows());
					CHECK(sigma.cols() == coords.rows());
				}
			}
			int dim_space = (int)coords.cols();
			if (cov_fct_type_ == "matern_space_time") {
				dim_space = (int)coords.cols() - 1;
			}
			if (cov_fct_type_ == "exponential" ||
				(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
				double cm = transf_scale ? (-1. * pars[1]) : (marg_var * pars[1] * pars[1]);
				sigma_grad = cm * sigma.cwiseProduct(dist);
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				double cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (marg_var * pars[0] * pars[1] * pars[1] * pars[1]);
				sigma_grad = cm * (dist.array().square() * ((-pars[1] * dist.array()).exp())).matrix();
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				double cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (marg_var * pars[0] * pars[1] * pars[1] * pars[1]);
				sigma_grad = cm * 1. / 3. * (dist.array().square() * (1. + pars[1] * dist.array()) * ((-pars[1] * dist.array()).exp())).matrix();
			}
			else if (cov_fct_type_ == "gaussian") {
				double cm = transf_scale ? (-1. * pars[1]) : (2. * marg_var * std::pow(pars[1], 3. / 2.));
				sigma_grad = cm * sigma.cwiseProduct(dist.array().square().matrix());
			}
			else if (cov_fct_type_ == "powered_exponential") {
				double cm = transf_scale ? (-1. * pars[1]) : (shape_ * marg_var * std::pow(pars[1], (shape_ + 1.) / shape_));
				sigma_grad = cm * sigma.cwiseProduct(dist.array().pow(shape_).matrix());
			}
			else if (cov_fct_type_ == "matern_space_time") {
				sigma_grad = T_mat(sigma.rows(), sigma.cols());
				CHECK(ind_range >= 0 && ind_range <= 1);
				den_mat_t coords_scaled(coords.rows(), coords.cols()), coords_pred_scaled;
				coords_scaled.col(0) = coords.col(0) * pars[1];
				coords_scaled.rightCols(dim_space) = coords.rightCols(dim_space) * pars[2];
				if (!is_symmmetric) {
					coords_pred_scaled = den_mat_t(coords_pred.rows(), coords_pred.cols());
					coords_pred_scaled.col(0) = coords_pred.col(0) * pars[1];
					coords_pred_scaled.rightCols(dim_space) = coords_pred.rightCols(dim_space) * pars[2];
				}
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					if (ind_range == 0) {
						double cm = transf_scale ? -1. : (marg_var * pars[1]);
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_time = (coords_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
									dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
									if (dist_sq_ij_time < EPSILON_NUMBERS) {
										sigma_grad(i, j) = 0.;
									}
									else {
										sigma_grad(i, j) = cm * dist_sq_ij_time / dist_ij * sigma(i, j);
									}
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_time = (coords_pred_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
									dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
									if (dist_sq_ij_time < EPSILON_NUMBERS) {
										sigma_grad(i, j) = 0.;
									}
									else {
										sigma_grad(i, j) = cm * dist_sq_ij_time / dist_ij * sigma(i, j);
									}
									
								}
							}
						}
					}//end ind_range == 0
					else if (ind_range == 1) {
						double cm = transf_scale ? -1. : (marg_var * pars[2]);
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_space = (coords_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
									if (dist_sq_ij_space < EPSILON_NUMBERS) {
										sigma_grad(i, j) = 0.;
									}
									else {
										sigma_grad(i, j) = cm * dist_sq_ij_space / dist_ij * sigma(i, j);
									}
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_space = (coords_pred_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
									if (dist_sq_ij_space < EPSILON_NUMBERS) {
										sigma_grad(i, j) = 0.;
									}
									else {
										sigma_grad(i, j) = cm * dist_sq_ij_space / dist_ij * sigma(i, j);
									}
								}
							}
						}
					}//end ind_range == 1
				}//end shape_ == 0.5
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					if (ind_range == 0) {
						double cm = transf_scale ? (-1. * pars[0]) : (marg_var * pars[0] * pars[1]);
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_time = (coords_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
									dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
									sigma_grad(i, j) = cm * dist_sq_ij_time * std::exp(-dist_ij);
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_time = (coords_pred_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
									dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
									sigma_grad(i, j) = cm * dist_sq_ij_time * std::exp(-dist_ij);
								}
							}
						}
					}//end ind_range == 0
					else if (ind_range == 1) {
						double cm = transf_scale ? (-1. * pars[0]) : (marg_var * pars[0] * pars[2]);
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_space = (coords_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
									sigma_grad(i, j) = cm * dist_sq_ij_space * std::exp(-dist_ij);
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_space = (coords_pred_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
									sigma_grad(i, j) = cm * dist_sq_ij_space * std::exp(-dist_ij);
								}
							}
						}
					}//end ind_range == 0
				}//end shape_ == 1.5
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					if (ind_range == 0) {
						double cm = transf_scale ? (-1. / 3. * pars[0]) : (marg_var / 3. * pars[0] * pars[1]);
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_time = (coords_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
									dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
									sigma_grad(i, j) = cm * dist_sq_ij_time * (1 + dist_ij) * std::exp(-dist_ij);
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_time = (coords_pred_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
									dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
									sigma_grad(i, j) = cm * dist_sq_ij_time * (1 + dist_ij) * std::exp(-dist_ij);
								}
							}
						}
					}//end ind_range == 0
					else if (ind_range == 1) {
						double cm = transf_scale ? (-1. / 3. * pars[0]) : (marg_var / 3. * pars[0] * pars[2]);
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_space = (coords_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
									sigma_grad(i, j) = cm * dist_sq_ij_space * (1 + dist_ij) * std::exp(-dist_ij);
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_space = (coords_pred_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
									sigma_grad(i, j) = cm * dist_sq_ij_space * (1 + dist_ij) * std::exp(-dist_ij);
								}
							}
						}
					}//end ind_range == 1
				}//end shape_ == 2.5
			}//end matern_space_time
			else {
				Log::REFatal("GetCovMatGradRange: Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end GetCovMatGradRange (dense)
		template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
		void GetCovMatGradRange(const T_mat& dist,
			const den_mat_t&,
			const den_mat_t&,
			const T_mat& sigma,
			const vec_t& pars,
			T_mat& sigma_grad,
			bool transf_scale,
			double marg_var,
			int,
			bool) const {
			CHECK(pars.size() == num_cov_par_);
			CHECK(sigma.cols() == sigma.rows());
			if (!save_distances_) {
				Log::REFatal("GetCovMatGradRange is not implemented when save_distances_ == false (e.g., cov_fct_type_ == 'matern_space_time') and sparse matrices are used ");
			}
			if (cov_fct_type_ == "exponential" ||
				(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
				double cm = transf_scale ? (-1. * pars[1]) : (marg_var * pars[1] * pars[1]);
				sigma_grad = cm * sigma.cwiseProduct(dist);
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				double cm = transf_scale ? 1. : (-1. * marg_var * pars[1]);
				sigma_grad = dist;
				sigma_grad.coeffs() = pars[0] * pars[1] * sigma_grad.coeffs() * ((-pars[1] * sigma_grad.coeffs()).exp());
				sigma_grad -= pars[1] * sigma.cwiseProduct(dist);
				sigma_grad *= cm;
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				double cm = transf_scale ? 1. : (-1. * marg_var * pars[1]);
				sigma_grad = dist;
				sigma_grad.coeffs() = pars[0] * (pars[1] * sigma_grad.coeffs() + (2. / 3.) * pars[1] * pars[1] * sigma_grad.coeffs().square()) *
					((-pars[1] * sigma_grad.coeffs()).exp());
				sigma_grad -= pars[1] * sigma.cwiseProduct(dist);
				sigma_grad *= cm;
			}
			else if (cov_fct_type_ == "gaussian") {
				double cm = transf_scale ? (-1. * pars[1]) : (2. * marg_var * std::pow(pars[1], 3. / 2.));
				sigma_grad = dist;
				sigma_grad.coeffs() = sigma_grad.coeffs().square();
				sigma_grad = cm * sigma.cwiseProduct(sigma_grad);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				double cm = transf_scale ? (-1. * pars[1]) : (shape_ * marg_var * std::pow(pars[1], (shape_ + 1.) / shape_));
				sigma_grad = dist;
				sigma_grad.coeffs() = sigma_grad.coeffs().pow(shape_);
				sigma_grad = cm * sigma.cwiseProduct(sigma_grad);
			}
			else {
				Log::REFatal("GetCovMatGradRange: Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end GetCovMatGradRange (sparse)

		/*!
		* \brief Find "reasonable" default values for the intial values of the covariance parameters (on transformed scale)
		* \param dist Distance matrix
		* \param coords Coordinates matrix
		* \param use_distances If true, 'dist' is used, otherwise 'coords' is used
		* \param rng Random number generator
		* \param[out] pars Vector with covariance parameters
		* \param marginal_variance Initial value for marginal variance
		*/
		template <typename T_mat>
		void FindInitCovPar(const T_mat& dist,
			const den_mat_t& coords,
			bool use_distances,
			RNG_t& rng,
			vec_t& pars,
			double marginal_variance) const {
			CHECK(pars.size() == num_cov_par_);
			pars[0] = marginal_variance;// marginal variance
			if (cov_fct_type_ != "wendland") {
				// Range parameters
				int MAX_POINTS_INIT_RANGE = 1000;//limit number of samples considered to save computational time
				int num_coord;
				if (use_distances) {
					num_coord = (int)dist.rows();
				}
				else {
					num_coord = (int)coords.rows();
				}
				// Calculate average distance
				int num_data_find_init = (num_coord > MAX_POINTS_INIT_RANGE) ? MAX_POINTS_INIT_RANGE : num_coord;
				std::vector<int> sample_ind;
				bool use_subsamples = num_data_find_init < num_coord;
				if (use_subsamples) {
					std::uniform_int_distribution<> dis(0, num_coord - 1);
					sample_ind = std::vector<int>(num_data_find_init);
					for (int i = 0; i < num_data_find_init; ++i) {
						sample_ind[i] = dis(rng);
					}
				}
				double mean_dist = 0;
				if (use_distances) {
					if (use_subsamples) {
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								mean_dist += dist.coeff(sample_ind[i], sample_ind[j]);
							}
						}
					}
					else {
						for (int i = 0; i < (num_coord - 1); ++i) {
							for (int j = i + 1; j < num_coord; ++j) {
								mean_dist += dist.coeff(i, j);
							}
						}
					}
				}
				else {
					// Calculate distances (of a subsample) in case they have not been calculated (e.g., for the Vecchia approximation)
					den_mat_t dist_from_coord;
					if (use_subsamples) {
						if (cov_fct_type_ == "matern_space_time") {
							CalculateDistances<den_mat_t>(coords(sample_ind, Eigen::seq(1, Eigen::last)), coords(sample_ind, Eigen::seq(1, Eigen::last)), true, dist_from_coord);
						}
						else {
							CalculateDistances<den_mat_t>(coords(sample_ind, Eigen::all), coords(sample_ind, Eigen::all), true, dist_from_coord);
						}
					}
					else {
						if (cov_fct_type_ == "matern_space_time") {
							CalculateDistances<den_mat_t>(coords(Eigen::all, Eigen::seq(1, Eigen::last)), coords(Eigen::all, Eigen::seq(1, Eigen::last)), true, dist_from_coord);
						}
						else {
							CalculateDistances<den_mat_t>(coords, coords, true, dist_from_coord);
						}
					}
					for (int i = 0; i < (num_data_find_init - 1); ++i) {
						for (int j = i + 1; j < num_data_find_init; ++j) {
							mean_dist += dist_from_coord(i, j);
						}
					}
				}
				mean_dist /= (num_data_find_init * (num_data_find_init - 1) / 2.);
				// Set the range parameters such that the correlation is down to 0.05 at the mean distance
				if (cov_fct_type_ == "exponential" ||
					(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
					pars[1] = 3. / mean_dist;
				}
				else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
					pars[1] = 4.7 / mean_dist;
				}
				else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
					pars[1] = 5.9 / mean_dist;
				}
				else if (cov_fct_type_ == "gaussian") {
					pars[1] = 3. / std::pow(mean_dist, 2.);
				}
				else if (cov_fct_type_ == "powered_exponential") {
					pars[1] = 3. / std::pow(mean_dist, shape_);
				}
				else if (cov_fct_type_ == "matern_space_time") {
					double mean_dist_time = 0;
					if (use_subsamples) {
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								mean_dist_time += std::abs(coords.coeff(sample_ind[i], 0) - coords.coeff(sample_ind[j], 0));;
							}
						}
					}
					else {
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								mean_dist_time += std::abs(coords.coeff(i, 0) - coords.coeff(j, 0));;
							}
						}
					}
					mean_dist_time /= (num_data_find_init * (num_data_find_init - 1) / 2.);
					pars[1] = 3. / mean_dist_time;
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
						pars[2] = 3. / mean_dist;
					}
					else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
						pars[2] = 4.7 / mean_dist;
					}
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
						pars[2] = 5.9 / mean_dist;
					}
				}//end matern_space_time
				else {
					Log::REFatal("Finding initial values for covariance parameters for covariance of type '%s' is not supported ", cov_fct_type_.c_str());
				}
			}//end cov_fct_type_ != "wendland"
		}//end FindInitCovPar

	private:
		/*! \brief Type of covariance function  */
		string_t cov_fct_type_;
		/*! \brief Shape parameter of covariance function (=smoothness parameter for Matern covariance) */
		double shape_;
		/*! \brief Range parameter of the Wendland covariance functionand Wendland correlation taper function.We follow the notation of Bevilacqua et al. (2019, AOS) */
		double taper_range_;
		/*! \brief Shape parameter of the Wendland covariance functionand Wendland correlation taper function.We follow the notation of Bevilacqua et al. (2019, AOS) */
		double taper_shape_;
		/*! \briefParameter \mu of the Wendland covariance functionand Wendland correlation taper function.We follow the notation of Bevilacqua et al. (2019, AOS) */
		double taper_mu_;
		/*! \brief If true, tapering is applied to the covariance function(element - wise multiplication with a compactly supported Wendland correlation function) */
		bool apply_tapering_ = false;
		/*! \brief Number of covariance parameters */
		int num_cov_par_;
		/*! \brief If true, distances should be saved depending on the covariance function (in re_comp.h) */
		bool save_distances_;
		/*! \brief List of supported covariance functions */
		const std::set<string_t> SUPPORTED_COV_TYPES_{ "exponential",
			"gaussian",
			"powered_exponential",
			"matern",
			"wendland",
			"matern_space_time"};

		/*!
		* \brief Calculates Wendland correlation function if taper_shape == 0
		* \param dist Distance
		* \return Wendland correlation
		*/
		inline double WendlandCorrelationShape0(const double dist) const {
			if (dist < EPSILON_NUMBERS) {
				return(1.);
			}
			else {
				return(std::pow((1. - dist / taper_range_), taper_mu_));
			}
		}

		/*!
		* \brief Calculates Wendland correlation function if taper_shape == 1
		* \param dist Distance
		* \return Wendland correlation
		*/
		inline double WendlandCorrelationShape1(const double dist) const {
			if (dist < EPSILON_NUMBERS) {
				return(1.);
			}
			else {
				return(std::pow((1. - dist / taper_range_), taper_mu_ + 1.) * (1. + dist / taper_range_ * (taper_mu_ + 1.)));
			}
		}

		/*!
		* \brief Calculates Wendland correlation function if taper_shape == 2
		* \param dist Distance
		* \return Wendland correlation
		*/
		inline double WendlandCorrelationShape2(const double dist) const {
			if (dist < EPSILON_NUMBERS) {
				return(1.);
			}
			else {
				return(std::pow((1. - dist / taper_range_), taper_mu_ + 2.) *
					(1. + dist / taper_range_ * (taper_mu_ + 2.) + std::pow(dist / taper_range_, 2) * (taper_mu_ * taper_mu_ + 4 * taper_mu_ + 3.) / 3.));
			}
		}

		/*!
		* \brief Calculates Matern covariance function if shape == 0.5
		* \param dist Distance
		* \param var Marginal variance
		* \param range Transformed range parameter
		* \return Covariance
		*/
		static double MaternCovarianceShape0_5(const double dist,
			const double& var,
			const double& range) {
			return(var * std::exp(-range * dist));
		}

		/*!
		* \brief Calculates Matern covariance function if shape == 1.5
		* \param dist Distance
		* \param var Marginal variance
		* \param range Transformed range parameter
		* \return Covariance
		*/
		static double MaternCovarianceShape1_5(const double dist,
			const double& var,
			const double& range) {
			double range_dist = range * dist;
			return(var * (1. + range_dist) * std::exp(-range_dist));
		}

		/*!
		* \brief Calculates Matern covariance function if shape == 2.5
		* \param dist Distance
		* \param var Marginal variance
		* \param range Transformed range parameter
		* \return Covariance
		*/
		static double MaternCovarianceShape2_5(const double dist,
			const double& var,
			const double& range) {
			double range_dist = range * dist;
			return(var * (1. + range_dist + range_dist * range_dist / 3.) * std::exp(-range_dist));
		}

		/*!
		* \brief Calculates Gaussian covariance function
		* \param dist Distance
		* \param var Marginal variance
		* \param range Transformed range parameter
		* \return Covariance
		*/
		static double GaussianCovariance(const double dist,
			const double& var,
			const double& range) {
			return(var * std::exp(-range * dist * dist));
		}

		/*!
		* \brief Calculates powered exponential covariance function
		* \param dist Distance
		* \param var Marginal variance
		* \param range Transformed range parameter
		* \return Covariance
		*/
		inline double PoweredExponentialCovariance(const double dist,
			const double& var,
			const double& range) const {
			return(var * std::exp(-range * std::pow(dist, shape_)));
		}

		template<typename>
		friend class RECompGP;
	};

}  // namespace GPBoost

#endif   // GPB_COV_FUNCTIONS_
