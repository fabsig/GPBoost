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
		* \param dim_coordinates Dimension of input coordinates / features
		*/
		CovFunction(string_t cov_fct_type,
			double shape,
			double taper_range,
			double taper_shape,
			double taper_mu,
			bool apply_tapering,
			int dim_coordinates) {
			if (cov_fct_type == "exponential_tapered") {
				Log::REFatal("Covariance of type 'exponential_tapered' is discontinued. Use the option 'gp_approx = \"tapering\"' instead ");
			}
			ParseCovFunctionAlias(cov_fct_type, shape);
			if (SUPPORTED_COV_TYPES_.find(cov_fct_type) == SUPPORTED_COV_TYPES_.end()) {
				Log::REFatal("Covariance of type '%s' is not supported ", cov_fct_type.c_str());
			}
			if (cov_fct_type == "matern_space_time" || cov_fct_type == "matern_ard" || cov_fct_type == "gaussian_ard") {
				save_distances_ = false;
			}
			else {
				save_distances_ = true;
			}
			if (cov_fct_type == "matern_space_time") {
				num_cov_par_ = 3;
			}
			else if (cov_fct_type == "matern_ard" || cov_fct_type == "gaussian_ard") {
				num_cov_par_ = dim_coordinates + 1;
			}
			else if (cov_fct_type == "wendland") {
				num_cov_par_ = 1;
			}
			else {
				num_cov_par_ = 2;
			}
			cov_fct_type_ = cov_fct_type;
			shape_ = shape;
			if (cov_fct_type == "matern" || cov_fct_type == "matern_space_time" || cov_fct_type == "matern_ard") {
				CHECK(shape > 0.);
				if (!(TwoNumbersAreEqual<double>(shape, 0.5) || TwoNumbersAreEqual<double>(shape, 1.5) || TwoNumbersAreEqual<double>(shape, 2.5))) {
					const_ = std::pow(2., 1 - shape_) / std::tgamma(shape_);
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

		string_t CovFunctionName() const {
			return(cov_fct_type_);
		}

		bool ShouldSaveDistances() const {
			return(save_distances_);
		}

		bool IsSpaceTimeModel() const {
			return(cov_fct_type_ == "matern_space_time");
		}

		bool IsARDModel() const {
			return(cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard");
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
		* \brief Scale / transform coordinates for anisotropic covariance functions
		* \param pars Vector with covariance parameters
		* \param coords Original coordinates
		* \param[out] coords_scaled Scaled coordinates
		*/
		void ScaleCoordinates(const vec_t& pars,
			const den_mat_t& coords,
			den_mat_t& coords_scaled) const {
			coords_scaled = den_mat_t(coords.rows(), coords.cols());
				if (cov_fct_type_ == "matern_space_time") {
					coords_scaled.col(0) = coords.col(0) * pars[1];
					int dim_space = (int)coords.cols() - 1;
					coords_scaled.rightCols(dim_space) = coords.rightCols(dim_space) * pars[2];
				}
				else if (cov_fct_type_ == "matern_ard") {
					for (int i = 0; i < (int)coords.cols(); ++i) {
						coords_scaled.col(i) = coords.col(i) * pars[i + 1];
					}
				}
				else if (cov_fct_type_ == "gaussian_ard") {
					for (int i = 0; i < (int)coords.cols(); ++i) {
						coords_scaled.col(i) = coords.col(i) * std::sqrt(pars[i + 1]);
					}
				}
				else {
					Log::REFatal("'ScaleCoordinates' is called for a model for which this function is not implemented ");
				}
		}//end ScaleCoordinates

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
			else if (cov_fct_type_ == "matern") {
				pars_trans[1] = sqrt(2. * shape_) / pars[1];
			}
			else if (cov_fct_type_ == "gaussian") {
				pars_trans[1] = 1. / (pars[1] * pars[1]);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				pars_trans[1] = 1. / (std::pow(pars[1], shape_));
			}
			else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
				double mult_const = 1.;
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					mult_const = 1.;
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					mult_const = sqrt(3.);
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					mult_const = sqrt(5.);
				}
				else {
					mult_const = sqrt(2. * shape_);
				}
				for (int i = 1; i < num_cov_par_; ++i) {
					pars_trans[i] = mult_const / pars[i];
				}
			}
			else if (cov_fct_type_ == "gaussian_ard") {
				for (int i = 1; i < num_cov_par_; ++i) {
					pars_trans[i] = 1. / (pars[i] * pars[i]);
				}
			}
		}//end TransformCovPars

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
			else if (cov_fct_type_ == "matern") {
				pars_orig[1] = sqrt(2. * shape_) / pars[1];
			}
			else if (cov_fct_type_ == "gaussian") {
				pars_orig[1] = 1. / std::sqrt(pars[1]);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				pars_orig[1] = 1. / (std::pow(pars[1], 1. / shape_));
			}
			else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
				double mult_const = 1.;
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					mult_const = 1.;
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					mult_const = sqrt(3.);
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					mult_const = sqrt(5.);
				}
				else {
					mult_const = sqrt(2. * shape_);
				}
				for (int i = 1; i < num_cov_par_; ++i) {
					pars_orig[i] = mult_const / pars[i];
				}
			}
			else if (cov_fct_type_ == "gaussian_ard") {
				for (int i = 1; i < num_cov_par_; ++i) {
					pars_orig[i] = 1. / std::sqrt(pars[i]);
				}
			}
		}//end TransformBackCovPars

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
			}
			else if (cov_fct_type_ == "matern") {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						sigma(i, i) = pars[0];
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							sigma(i, j) = MaternCovarianceGeneralShape(dist(i, j), pars[0], pars[1]);
							sigma(j, i) = sigma(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							sigma(i, j) = MaternCovarianceGeneralShape(dist(i, j), pars[0], pars[1]);
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
			else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard") {
				den_mat_t coords_scaled, coords_pred_scaled;
				ScaleCoordinates(pars, coords, coords_scaled);
				if (!is_symmmetric) {
					ScaleCoordinates(pars, coords_pred, coords_pred_scaled);
				}
				if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
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
					else {//general shape
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma(i, i) = pars[0];
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									sigma(i, j) = MaternCovarianceGeneralShape(dist_ij, pars[0], 1.);
									sigma(j, i) = sigma(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									sigma(i, j) = MaternCovarianceGeneralShape(dist_ij, pars[0], 1.);
								}
							}
						}
					}
				}//end cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard"
				else {//cov_fct_type_ == "gaussian_ard"
					if (is_symmmetric) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords.rows(); ++i) {
							sigma(i, i) = pars[0];
							for (int j = i + 1; j < (int)coords.rows(); ++j) {
								double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								sigma(i, j) = GaussianCovariance(dist_ij, pars[0], 1.);
								sigma(j, i) = sigma(i, j);
							}
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords_pred.rows(); ++i) {
							for (int j = 0; j < (int)coords.rows(); ++j) {
								double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								sigma(i, j) = GaussianCovariance(dist_ij, pars[0], 1.);
							}
						}
					}
				}
			}//end cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard"
			else {
				Log::REFatal("Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end GetCovMat (dense)
		template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
		void GetCovMat(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const vec_t& pars,
			T_mat& sigma,
			bool is_symmmetric) const {
			CHECK(pars.size() == num_cov_par_);
			sigma = dist;
			sigma.makeCompressed();
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
			}//end cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)
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
			}//end cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)
			else if (cov_fct_type_ == "matern") {
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
								it.valueRef() = MaternCovarianceGeneralShape(dist.coeff(i, j), pars[0], pars[1]);
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
							it.valueRef() = MaternCovarianceGeneralShape(dist.coeff(i, j), pars[0], pars[1]);
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
			else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard") {
				den_mat_t coords_scaled, coords_pred_scaled;
				ScaleCoordinates(pars, coords, coords_scaled);
				if (!is_symmmetric) {
					ScaleCoordinates(pars, coords_pred, coords_pred_scaled);
				}
				if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
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
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										it.valueRef() = MaternCovarianceShape0_5(dist_ij, pars[0], 1.);
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
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									it.valueRef() = MaternCovarianceShape0_5(dist_ij, pars[0], 1.);
								}
							}
						}
					}//end TwoNumbersAreEqual<double>(shape_, 0.5)
					else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
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
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										it.valueRef() = MaternCovarianceShape1_5(dist_ij, pars[0], 1.);
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
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									it.valueRef() = MaternCovarianceShape1_5(dist_ij, pars[0], 1.);
								}
							}
						}
					}//end TwoNumbersAreEqual<double>(shape_, 1.5)
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
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
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										it.valueRef() = MaternCovarianceShape2_5(dist_ij, pars[0], 1.);
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
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									it.valueRef() = MaternCovarianceShape2_5(dist_ij, pars[0], 1.);
								}
							}
						}
					}//end TwoNumbersAreEqual<double>(shape_, 2.5)
					else {//general shape
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
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										it.valueRef() = MaternCovarianceGeneralShape(dist_ij, pars[0], 1.);
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
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									it.valueRef() = MaternCovarianceGeneralShape(dist_ij, pars[0], 1.);
								}
							}
						}
					}
				}//end cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard"
				else {//cov_fct_type_ == "gaussian_ard"
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
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									it.valueRef() = GaussianCovariance(dist_ij, pars[0], 1.);
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
								double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
								it.valueRef() = GaussianCovariance(dist_ij, pars[0], 1.);
							}
						}
					}
				}
			}//end cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard"
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
			if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard") {
				Log::REFatal("'GetCovMat()' is not implemented for one distance when cov_fct_type_ == '%s' ", cov_fct_type_.c_str());
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
			}
			else if (cov_fct_type_ == "matern") {
				sigma = MaternCovarianceGeneralShape(dist, pars[0], pars[1]);
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
		* \param nugget_var Nugget / error variance parameters sigma^2 (used only if transf_scale = false to transform back for gaussian likelihoods since then the nugget is factored out)
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
			double nugget_var,
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
				double cm = transf_scale ? (-1. * pars[1]) : (nugget_var * pars[1] * pars[1]);
				sigma_grad = cm * sigma.cwiseProduct(dist);
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				double cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (nugget_var * pars[0] * std::pow(pars[1], 3) / sqrt(3.));
				sigma_grad = cm * (dist.array().square() * ((-pars[1] * dist.array()).exp())).matrix();
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				double cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (nugget_var * pars[0] * std::pow(pars[1], 3) / sqrt(5.));
				sigma_grad = cm * 1. / 3. * (dist.array().square() * (1. + pars[1] * dist.array()) * ((-pars[1] * dist.array()).exp())).matrix();
			}
			else if (cov_fct_type_ == "matern") {//general shape
				double cm = transf_scale ? 1. : (- nugget_var * pars[1] / std::sqrt(2. * shape_));
				cm *= pars[0] * const_;
				sigma_grad = T_mat(sigma.rows(), sigma.cols());
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						sigma_grad(i, i) = 0.;
						for (int j = i + 1; j < (int)dist.cols(); ++j) {
							double range_dist = pars[1] * dist.coeff(i,j);
							sigma_grad(i, j) = cm * std::pow(range_dist, shape_) * (2. * shape_ * std::cyl_bessel_k(shape_, range_dist) - range_dist * std::cyl_bessel_k(shape_ + 1., range_dist));
							sigma_grad(j, i) = sigma_grad(i, j);
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)dist.rows(); ++i) {
						for (int j = 0; j < (int)dist.cols(); ++j) {
							double range_dist = pars[1] * dist.coeff(i, j);
							sigma_grad(i, j) = cm * std::pow(range_dist, shape_) * (2. * shape_ * std::cyl_bessel_k(shape_, range_dist) - range_dist * std::cyl_bessel_k(shape_ + 1., range_dist));
						}
					}
				}
			}//end matern
			else if (cov_fct_type_ == "gaussian") {
				double cm = transf_scale ? (-1. * pars[1]) : (2. * nugget_var * std::pow(pars[1], 3. / 2.));
				sigma_grad = cm * sigma.cwiseProduct(dist.array().square().matrix());
			}
			else if (cov_fct_type_ == "powered_exponential") {
				double cm = transf_scale ? (-1. * pars[1]) : (shape_ * nugget_var * std::pow(pars[1], (shape_ + 1.) / shape_));
				sigma_grad = cm * sigma.cwiseProduct(dist.array().pow(shape_).matrix());
			}
			else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard") {
				sigma_grad = T_mat(sigma.rows(), sigma.cols());
				den_mat_t coords_scaled, coords_pred_scaled;
				ScaleCoordinates(pars, coords, coords_scaled);
				if (!is_symmmetric) {
					ScaleCoordinates(pars, coords_pred, coords_pred_scaled);
				}
				if (cov_fct_type_ == "matern_space_time") {
					CHECK(ind_range >= 0 && ind_range <= 1);
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {						
						if (ind_range == 0) {
							double cm = transf_scale ? -1. : (nugget_var * pars[1]);
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
							double cm = transf_scale ? -1. : (nugget_var * pars[2]);
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
							double cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[1] / sqrt(3.));
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
							double cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[2] / sqrt(3.));
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
						}//end ind_range == 1
					}//end shape_ == 1.5
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
						if (ind_range == 0) {
							double cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[1] / sqrt(5.));
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
							double cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[2]/sqrt(5.));
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
					else {//general shape
						if (ind_range == 0) {
							double cm = transf_scale ? 1. : (-nugget_var * pars[1] / sqrt(2. * shape_));
							cm *= pars[0] * const_;
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int i = 0; i < (int)coords.rows(); ++i) {
									sigma_grad(i, i) = 0.;
									for (int j = i + 1; j < (int)coords.rows(); ++j) {
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_time = (coords_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
										dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
										sigma_grad(i, j) = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_time;
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
										sigma_grad(i, j) = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_time;
									}
								}
							}
						}//end ind_range == 0
						else if (ind_range == 1) {
							double cm = transf_scale ? 1. : (-nugget_var * pars[2] / sqrt(2. * shape_));
							cm *= pars[0] * const_;
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int i = 0; i < (int)coords.rows(); ++i) {
									sigma_grad(i, i) = 0.;
									for (int j = i + 1; j < (int)coords.rows(); ++j) {
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_space = (coords_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
										sigma_grad(i, j) = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_space;
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
										sigma_grad(i, j) = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_space;
									}
								}
							}
						}//end ind_range == 1
					}//end general shape
				}//end matern_space_time
				else if (cov_fct_type_ == "matern_ard") {
					CHECK(ind_range >= 0 && ind_range < (int)coords.cols());
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
						double cm = transf_scale ? -1. : (nugget_var * pars[ind_range + 1]);
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									if (dist_sq_ij_coord < EPSILON_NUMBERS) {
										sigma_grad(i, j) = 0.;
									}
									else {
										sigma_grad(i, j) = cm * dist_sq_ij_coord / dist_ij * sigma(i, j);
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
									double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									if (dist_sq_ij_coord < EPSILON_NUMBERS) {
										sigma_grad(i, j) = 0.;
									}
									else {
										sigma_grad(i, j) = cm * dist_sq_ij_coord / dist_ij * sigma(i, j);
									}
								}
							}
						}
					}//end shape_ == 0.5
					else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
						double cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[ind_range + 1] / sqrt(3.));
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									sigma_grad(i, j) = cm * dist_sq_ij_coord * std::exp(-dist_ij);
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									sigma_grad(i, j) = cm * dist_sq_ij_coord * std::exp(-dist_ij);
								}
							}
						}
					}//end shape_ == 1.5
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
						double cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[ind_range + 1] / sqrt(5.));
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									sigma_grad(i, j) = cm * dist_sq_ij_coord * (1 + dist_ij) * std::exp(-dist_ij);
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									sigma_grad(i, j) = cm * dist_sq_ij_coord * (1 + dist_ij) * std::exp(-dist_ij);
								}
							}
						}
					}//end shape_ == 2.5
					else {//general shape
						double cm = transf_scale ? 1. : (-nugget_var * pars[ind_range + 1] / sqrt(2. * shape_));
						cm *= pars[0] * const_;
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords.rows(); ++i) {
								sigma_grad(i, i) = 0.;
								for (int j = i + 1; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									sigma_grad(i, j) = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_coord;
									sigma_grad(j, i) = sigma_grad(i, j);
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)coords_pred.rows(); ++i) {
								for (int j = 0; j < (int)coords.rows(); ++j) {
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									sigma_grad(i, j) = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_coord;
								}
							}
						}
					}//end general shape
				}//end matern_ard
				else if (cov_fct_type_ == "gaussian_ard") {
					CHECK(ind_range >= 0 && ind_range < (int)coords.cols());
					double cm = transf_scale ? -1. : (2. * nugget_var * std::sqrt(pars[1]));
					if (is_symmmetric) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords.rows(); ++i) {
							sigma_grad(i, i) = 0.;
							for (int j = i + 1; j < (int)coords.rows(); ++j) {
								double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
								dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
								if (dist_sq_ij_coord < EPSILON_NUMBERS) {
									sigma_grad(i, j) = 0.;
								}
								else {
									sigma_grad(i, j) = cm * dist_sq_ij_coord * sigma(i, j);
								}
								sigma_grad(j, i) = sigma_grad(i, j);
							}
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)coords_pred.rows(); ++i) {
							for (int j = 0; j < (int)coords.rows(); ++j) {
								double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
								dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
								if (dist_sq_ij_coord < EPSILON_NUMBERS) {
									sigma_grad(i, j) = 0.;
								}
								else {
									sigma_grad(i, j) = cm * dist_sq_ij_coord * sigma(i, j);
								}

							}
						}
					}
				}//end gaussian_ard
			}//end cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard"
			else {
				Log::REFatal("GetCovMatGradRange: Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end GetCovMatGradRange (dense)
		template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
		void GetCovMatGradRange(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const T_mat& sigma,
			const vec_t& pars,
			T_mat& sigma_grad,
			bool transf_scale,
			double nugget_var,
			int ind_range,
			bool is_symmmetric) const {
			CHECK(pars.size() == num_cov_par_);
			CHECK(sigma.cols() == sigma.rows());
			int dim_space = (int)coords.cols();
			if (cov_fct_type_ == "matern_space_time") {
				dim_space = (int)coords.cols() - 1;
			}
			if (cov_fct_type_ == "exponential" ||
				(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
				double cm = transf_scale ? (-1. * pars[1]) : (nugget_var * pars[1] * pars[1]);
				sigma_grad = cm * sigma.cwiseProduct(dist);
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
				double cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (nugget_var * pars[0] * std::pow(pars[1], 3) / sqrt(3.));
				sigma_grad = dist;
				sigma_grad.coeffs() = cm * dist.coeffs().square() * ((-pars[1] * dist.coeffs()).exp());
			}
			else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
				double cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (nugget_var * pars[0] * std::pow(pars[1], 3) / sqrt(5.));
				sigma_grad = dist;
				sigma_grad.coeffs() = cm * 1. / 3. * (dist.coeffs().square() * (1. + pars[1] * dist.coeffs()) * ((-pars[1] * dist.coeffs()).exp())).matrix();
			}
			else if (cov_fct_type_ == "matern") {
				double cm = transf_scale ? 1. : (-nugget_var * pars[1] / std::sqrt(2. * shape_));
				cm *= pars[0] * const_;
				sigma_grad = dist;
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma_grad.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = 0.;
							}
							else if (i < j) {
								double range_dist = pars[1] * dist.coeff(i, j);
								it.valueRef() = cm * std::pow(range_dist, shape_) * (2. * shape_ * std::cyl_bessel_k(shape_, range_dist) - range_dist * std::cyl_bessel_k(shape_ + 1., range_dist));
								sigma_grad.coeffRef(j, i) = it.value();
							}
						}
					}
				}
				else {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma_grad.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							double range_dist = pars[1] * dist.coeff(i, j);
							it.valueRef() = cm * std::pow(range_dist, shape_) * (2. * shape_ * std::cyl_bessel_k(shape_, range_dist) - range_dist * std::cyl_bessel_k(shape_ + 1., range_dist));
						}
					}
				}
			}//end matern
			else if (cov_fct_type_ == "gaussian") {
				double cm = transf_scale ? (-1. * pars[1]) : (2. * nugget_var * std::pow(pars[1], 3. / 2.));
				sigma_grad = dist;
				sigma_grad.coeffs() = cm * sigma.coeffs() * (dist.coeffs().square());
			}
			else if (cov_fct_type_ == "powered_exponential") {
				double cm = transf_scale ? (-1. * pars[1]) : (shape_ * nugget_var * std::pow(pars[1], (shape_ + 1.) / shape_));
				sigma_grad = dist;
				sigma_grad.coeffs() = cm * sigma.coeffs() * (dist.coeffs().pow(shape_));
			}
			else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard") {
				sigma_grad = sigma;
				den_mat_t coords_scaled, coords_pred_scaled;
				ScaleCoordinates(pars, coords, coords_scaled);
				if (!is_symmmetric) {
					ScaleCoordinates(pars, coords_pred, coords_pred_scaled);
				}
				if (cov_fct_type_ == "matern_space_time") {
					CHECK(ind_range >= 0 && ind_range <= 1);
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
						if (ind_range == 0) {
							double cm = transf_scale ? -1. : (nugget_var * pars[1]);
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										if (i == j) {
											it.valueRef() = 0.;
										}
										else if (i < j) {
											double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
											double dist_sq_ij_time = (coords_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
											dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
											if (dist_sq_ij_time < EPSILON_NUMBERS) {
												it.valueRef() = 0.;
											}
											else {
												it.valueRef() *= cm * dist_sq_ij_time / dist_ij;
											}
											sigma_grad.coeffRef(j, i) = it.value();
										}
									}
								}
							}
							else {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_time = (coords_pred_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
										dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
										if (dist_sq_ij_time < EPSILON_NUMBERS) {
											it.valueRef() = 0.;
										}
										else {
											it.valueRef() *= cm * dist_sq_ij_time / dist_ij;
										}
									}
								}
							}
						}//end ind_range == 0
						else if (ind_range == 1) {
							double cm = transf_scale ? -1. : (nugget_var * pars[2]);
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										if (i == j) {
											it.valueRef() = 0.;
										}
										else if (i < j) {
											double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
											double dist_sq_ij_space = (coords_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
											if (dist_sq_ij_space < EPSILON_NUMBERS) {
												it.valueRef() = 0.;
											}
											else {
												it.valueRef() *= cm * dist_sq_ij_space / dist_ij;
											}
											sigma_grad.coeffRef(j, i) = it.value();
										}
									}
								}
							}
							else {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_space = (coords_pred_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
										if (dist_sq_ij_space < EPSILON_NUMBERS) {
											it.valueRef() = 0.;
										}
										else {
											it.valueRef() *= cm * dist_sq_ij_space / dist_ij;
										}
									}
								}
							}
						}//end ind_range == 1
					}//end shape_ == 0.5
					else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
						if (ind_range == 0) {
							double cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[1] / sqrt(3.));
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										if (i == j) {
											it.valueRef() = 0.;
										}
										else if (i < j) {
											double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
											double dist_sq_ij_time = (coords_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
											dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
											it.valueRef() = cm * dist_sq_ij_time * std::exp(-dist_ij);
											sigma_grad.coeffRef(j, i) = it.value();
										}
									}
								}
							}
							else {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_time = (coords_pred_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
										dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
										it.valueRef() = cm * dist_sq_ij_time * std::exp(-dist_ij);
									}
								}
							}
						}//end ind_range == 0
						else if (ind_range == 1) {
							double cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[2] / sqrt(3.));
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_space = (coords_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
										it.valueRef() = cm * dist_sq_ij_space * std::exp(-dist_ij);
										sigma_grad.coeffRef(j, i) = it.value();
									}
								}
							}
							else {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_space = (coords_pred_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
										it.valueRef() = cm * dist_sq_ij_space * std::exp(-dist_ij);
									}
								}
							}
						}//end ind_range == 1
					}//end shape_ == 1.5
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
						if (ind_range == 0) {
							double cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[1] / sqrt(5.));
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										if (i == j) {
											it.valueRef() = 0.;
										}
										else if (i < j) {
											double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
											double dist_sq_ij_time = (coords_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
											dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
											it.valueRef() = cm * dist_sq_ij_time * (1 + dist_ij) * std::exp(-dist_ij);
											sigma_grad.coeffRef(j, i) = it.value();
										}
									}
								}
							}
							else {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_time = (coords_pred_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
										dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
										it.valueRef() = cm * dist_sq_ij_time * (1 + dist_ij) * std::exp(-dist_ij);
									}
								}
							}
						}//end ind_range == 0
						else if (ind_range == 1) {
							double cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[2] / sqrt(5.));
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										if (i == j) {
											it.valueRef() = 0.;
										}
										else if (i < j) {
											double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
											double dist_sq_ij_space = (coords_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
											it.valueRef() = cm * dist_sq_ij_space * (1 + dist_ij) * std::exp(-dist_ij);
											sigma_grad.coeffRef(j, i) = it.value();
										}
									}
								}
							}
							else {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_space = (coords_pred_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
										it.valueRef() = cm * dist_sq_ij_space * (1 + dist_ij) * std::exp(-dist_ij);
									}
								}
							}
						}//end ind_range == 1
					}//end shape_ == 2.5
					else {//general shape
						if (ind_range == 0) {
							double cm = transf_scale ? 1. : (-nugget_var * pars[1] / sqrt(2. * shape_));
							cm *= pars[0] * const_;
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										if (i == j) {
											it.valueRef() = 0.;
										}
										else if (i < j) {
											double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
											double dist_sq_ij_time = (coords_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
											dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
											it.valueRef() = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_time;
											sigma_grad.coeffRef(j, i) = it.value();
										}
									}
								}
							}
							else {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_time = (coords_pred_scaled.coeff(i, 0) - coords_scaled.coeff(j, 0));
										dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
										it.valueRef() = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_time;
									}
								}
							}
						}//end ind_range == 0
						else if (ind_range == 1) {
							double cm = transf_scale ? 1. : (-nugget_var * pars[2] / sqrt(2. * shape_));
							cm *= pars[0] * const_;
							if (is_symmmetric) {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										if (i == j) {
											it.valueRef() = 0.;
										}
										else if (i < j) {
											double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
											double dist_sq_ij_space = (coords_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
											it.valueRef() = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_space;
											sigma_grad.coeffRef(j, i) = it.value();
										}
									}
								}
							}
							else {
#pragma omp parallel for schedule(static)
								for (int k = 0; k < sigma_grad.outerSize(); ++k) {
									for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
										int i = (int)it.row();
										int j = (int)it.col();
										double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_space = (coords_pred_scaled.row(i).tail(dim_space) - coords_scaled.row(j).tail(dim_space)).squaredNorm();
										it.valueRef() = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_space;
									}
								}
							}
						}//end ind_range == 1
					}//end general shape
				}//end matern_space_time
				else if (cov_fct_type_ == "matern_ard") {
					CHECK(ind_range >= 0 && ind_range < (int)coords.cols());
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
						double cm = transf_scale ? -1. : (nugget_var * pars[ind_range + 1]);
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int k = 0; k < sigma_grad.outerSize(); ++k) {
								for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
									int i = (int)it.row();
									int j = (int)it.col();
									if (i == j) {
										it.valueRef() = 0.;
									}
									else if (i < j) {
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
										dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
										if (dist_sq_ij_coord < EPSILON_NUMBERS) {
											it.valueRef() = 0.;
										}
										else {
											it.valueRef() *= cm * dist_sq_ij_coord / dist_ij;
										}
										sigma_grad.coeffRef(j, i) = it.value();
									}
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int k = 0; k < sigma_grad.outerSize(); ++k) {
								for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
									int i = (int)it.row();
									int j = (int)it.col();
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									if (dist_sq_ij_coord < EPSILON_NUMBERS) {
										it.valueRef() = 0.;
									}
									else {
										it.valueRef() *= cm * dist_sq_ij_coord / dist_ij;
									}
								}
							}
						}
					}//end shape_ == 0.5
					else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
						double cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[ind_range + 1] / sqrt(3.));
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int k = 0; k < sigma_grad.outerSize(); ++k) {
								for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
									int i = (int)it.row();
									int j = (int)it.col();
									if (i == j) {
										it.valueRef() = 0.;
									}
									else if (i < j) {
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
										dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
										it.valueRef() = cm * dist_sq_ij_coord * std::exp(-dist_ij);
										sigma_grad.coeffRef(j, i) = it.value();
									}
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int k = 0; k < sigma_grad.outerSize(); ++k) {
								for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
									int i = (int)it.row();
									int j = (int)it.col();
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									it.valueRef() = cm * dist_sq_ij_coord * std::exp(-dist_ij);
								}
							}
						}
					}//end shape_ == 1.5
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
						double cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[ind_range + 1] / sqrt(5.));
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int k = 0; k < sigma_grad.outerSize(); ++k) {
								for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
									int i = (int)it.row();
									int j = (int)it.col();
									if (i == j) {
										it.valueRef() = 0.;
									}
									else if (i < j) {
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
										dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
										it.valueRef() = cm * dist_sq_ij_coord * (1 + dist_ij) * std::exp(-dist_ij);
										sigma_grad.coeffRef(j, i) = it.value();
									}
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int k = 0; k < sigma_grad.outerSize(); ++k) {
								for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
									int i = (int)it.row();
									int j = (int)it.col();
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									it.valueRef() = cm * dist_sq_ij_coord * (1 + dist_ij) * std::exp(-dist_ij);
								}
							}
						}
					}//end shape_ == 2.5
					else {//general shape
						double cm = transf_scale ? 1. : (-nugget_var * pars[ind_range + 1] / sqrt(2. * shape_));
						cm *= pars[0] * const_;
						if (is_symmmetric) {
#pragma omp parallel for schedule(static)
							for (int k = 0; k < sigma_grad.outerSize(); ++k) {
								for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
									int i = (int)it.row();
									int j = (int)it.col();
									if (i == j) {
										it.valueRef() = 0.;
									}
									else if (i < j) {
										double dist_ij = (coords_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
										double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
										dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
										it.valueRef() = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_coord;;
										sigma_grad.coeffRef(j, i) = it.value();
									}
								}
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int k = 0; k < sigma_grad.outerSize(); ++k) {
								for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
									int i = (int)it.row();
									int j = (int)it.col();
									double dist_ij = (coords_pred_scaled.row(i) - coords_scaled.row(j)).lpNorm<2>();
									double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									it.valueRef() = cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_coord;;
								}
							}
						}
					}//end general shape
				}//end matern_ard
				else if (cov_fct_type_ == "gaussian_ard") {
					CHECK(ind_range >= 0 && ind_range < (int)coords.cols());
					double cm = transf_scale ? -1. : (2. * nugget_var * std::sqrt(pars[1]));
					if (is_symmmetric) {
#pragma omp parallel for schedule(static)
						for (int k = 0; k < sigma_grad.outerSize(); ++k) {
							for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
								int i = (int)it.row();
								int j = (int)it.col();
								if (i == j) {
									it.valueRef() = 0.;
								}
								else if (i < j) {
									double dist_sq_ij_coord = (coords_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
									dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
									if (dist_sq_ij_coord < EPSILON_NUMBERS) {
										it.valueRef() = 0.;
									}
									else {
										it.valueRef() *= cm * dist_sq_ij_coord;
									}
									sigma_grad.coeffRef(j, i) = it.value();
								}
							}
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (int k = 0; k < sigma_grad.outerSize(); ++k) {
							for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
								int i = (int)it.row();
								int j = (int)it.col();
								double dist_sq_ij_coord = (coords_pred_scaled.coeff(i, ind_range) - coords_scaled.coeff(j, ind_range));
								dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
								if (dist_sq_ij_coord < EPSILON_NUMBERS) {
									it.valueRef() = 0.;
								}
								else {
									it.valueRef() *= cm * dist_sq_ij_coord;
								}

							}
						}
					}
				}//end gaussian_ard
			}//end cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard"
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
				double mean_dist = 0., mean_dist_space = 0., mean_dist_time = 0.;
				std::vector<double> mean_dist_per_coord;//for ard kernels
				if (cov_fct_type_ != "matern_space_time" && cov_fct_type_ != "matern_ard" && cov_fct_type_ != "gaussian_ard") {
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
							CalculateDistances<den_mat_t>(coords(sample_ind, Eigen::all), coords(sample_ind, Eigen::all), true, dist_from_coord);
						}
						else {
							CalculateDistances<den_mat_t>(coords, coords, true, dist_from_coord);
						}
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								mean_dist += dist_from_coord(i, j);
							}
						}
					}
					mean_dist /= (num_data_find_init * (num_data_find_init - 1) / 2.);
					if (mean_dist < EPSILON_NUMBERS) {
						Log::REFatal("Cannot find an initial value for the range parameter since the average distance among coordinates is zero ");
					}
				}//end cov_fct_type_ != "matern_space_time" && cov_fct_type_ != "matern_ard" && cov_fct_type_ != "gaussian_ard"
				else if (cov_fct_type_ == "matern_space_time") {
					den_mat_t dist_from_coord;
					if (use_subsamples) {
						CalculateDistances<den_mat_t>(coords(sample_ind, Eigen::seq(1, Eigen::last)), coords(sample_ind, Eigen::seq(1, Eigen::last)), true, dist_from_coord);
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								mean_dist_space += dist_from_coord(i, j);
								mean_dist_time += std::abs(coords.coeff(sample_ind[i], 0) - coords.coeff(sample_ind[j], 0));;
							}
						}
					}
					else {
						CalculateDistances<den_mat_t>(coords(Eigen::all, Eigen::seq(1, Eigen::last)), coords(Eigen::all, Eigen::seq(1, Eigen::last)), true, dist_from_coord);
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								mean_dist_space += dist_from_coord(i, j);
								mean_dist_time += std::abs(coords.coeff(i, 0) - coords.coeff(j, 0));;
							}
						}
					}
					mean_dist_space /= (num_data_find_init * (num_data_find_init - 1) / 2.);
					mean_dist_time /= (num_data_find_init * (num_data_find_init - 1) / 2.);
					if (mean_dist_space < EPSILON_NUMBERS) {
						Log::REFatal("Cannot find an initial value for the spatial range parameter since the average distance among spatial coordinates is zero ");
					}
					if (mean_dist_time < EPSILON_NUMBERS) {
						Log::REFatal("Cannot find an initial value for the temporal range parameter since the average distance among time points is zero ");
					}
				}//end cov_fct_type_ == "matern_space_time"
				else if (cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard") {
					mean_dist_per_coord = std::vector<double>((int)coords.cols());					
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						double mean_dist_coord_i = 0.;
						if (use_subsamples) {
							for (int i = 0; i < (num_data_find_init - 1); ++i) {
								for (int j = i + 1; j < num_data_find_init; ++j) {
									mean_dist_coord_i += std::abs(coords.coeff(sample_ind[i], ic) - coords.coeff(sample_ind[j], ic));;
								}
							}
						}
						else {
							for (int i = 0; i < (num_data_find_init - 1); ++i) {
								for (int j = i + 1; j < num_data_find_init; ++j) {
									mean_dist_coord_i += std::abs(coords.coeff(i, ic) - coords.coeff(j, ic));;
								}
							}
						}
						mean_dist_coord_i /= (num_data_find_init * (num_data_find_init - 1) / 2.);
						mean_dist_per_coord[ic] = mean_dist_coord_i;
						if (mean_dist_coord_i < EPSILON_NUMBERS) {
							Log::REFatal("Cannot find an initial value for the range parameter for the input feature number %d (counting starts at 1) since this feature is constant ", ic + 1);
						}
					}
				}//end cov_fct_type_ == "matern_ard" && cov_fct_type_ == "gaussian_ard"
				// Set the range parameters such that the correlation is down to 0.05 at half the mean distance
				if (cov_fct_type_ == "exponential" ||
					(cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 0.5))) {
					pars[1] = 2. * 3. / mean_dist;
				}
				else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 1.5)) {
					pars[1] = 2. * 4.7 / mean_dist;
				}
				else if (cov_fct_type_ == "matern" && TwoNumbersAreEqual<double>(shape_, 2.5)) {
					pars[1] = 2. * 5.9 / mean_dist;
				}
				else if (cov_fct_type_ == "matern") {
					if (shape_ <= 1.) {
						pars[1] = 2. * 3. / mean_dist;//same as shape_ = 0.5
					}
					else if (shape_ <= 2.) {
						pars[1] = 2. * 4.7 / mean_dist;//same as shape_ = 1.5
					}
					else {
						pars[1] = 2. * 5.9 / mean_dist;//same as shape_ = 2.5
					}
				}
				else if (cov_fct_type_ == "gaussian") {
					pars[1] = 3. / std::pow(mean_dist / 2., 2.);
				}
				else if (cov_fct_type_ == "powered_exponential") {
					pars[1] = 3. / std::pow(mean_dist / 2., shape_);
				}
				else if (cov_fct_type_ == "matern_space_time") {
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
						pars[1] = 2. * 3. / mean_dist_time;
						pars[2] = 2. * 3. / mean_dist_space;
					}
					else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
						pars[1] = 2. * 4.7 / mean_dist_time;
						pars[2] = 2. * 4.7 / mean_dist_space;
					}
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
						pars[1] = 2. * 5.9 / mean_dist_time;
						pars[2] = 2. * 5.9 / mean_dist_space;
					}
					else {//general shape
						if (shape_ <= 1.) {
							pars[1] = 2. * 3. / mean_dist_time;
							pars[2] = 2. * 3. / mean_dist_space;//same as shape_ = 0.5
						}
						else if (shape_ <= 2.) {
							pars[1] = 2. * 4.7 / mean_dist_time;
							pars[2] = 2. * 4.7 / mean_dist_space;//same as shape_ = 1.5
						}
						else {
							pars[1] = 2. * 5.9 / mean_dist_time;
							pars[2] = 2. * 5.9 / mean_dist_space;//same as shape_ = 2.5
						}
					}
				}//end matern_space_time
				else if (cov_fct_type_ == "matern_ard") {
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
						for (int ic = 0; ic < (int)coords.cols(); ++ic) {
							pars[1 + ic] = 2. * 3. / mean_dist_per_coord[ic];
						}
					}
					else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
						for (int ic = 0; ic < (int)coords.cols(); ++ic) {
							pars[1 + ic] = 2. * 4.7 / mean_dist_per_coord[ic];
						}
					}
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
						for (int ic = 0; ic < (int)coords.cols(); ++ic) {
							pars[1 + ic] = 2. * 5.9 / mean_dist_per_coord[ic];
						}
					}
					else {//general shape
						if (shape_ <= 1.) {
							for (int ic = 0; ic < (int)coords.cols(); ++ic) {
								pars[1 + ic] = 2. * 3. / mean_dist_per_coord[ic];//same as shape_ = 0.5
							}
						}
						else if (shape_ <= 2.) {
							for (int ic = 0; ic < (int)coords.cols(); ++ic) {
								pars[1 + ic] = 2. * 4.7 / mean_dist_per_coord[ic];//same as shape_ = 1.5
							}
						}
						else {
							for (int ic = 0; ic < (int)coords.cols(); ++ic) {
								pars[1 + ic] = 2. * 5.9 / mean_dist_per_coord[ic];//same as shape_ = 2.5
							}
						}
					}
				}//end matern_ard
				else if (cov_fct_type_ == "gaussian_ard") {
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						pars[1 + ic] = 3. / std::pow(mean_dist_per_coord[ic] / 2., 2.);
					}
				}
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
		/*! \brief Constant in covariance function (used only for Matern with general shape) */
		double const_;
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
			"matern_space_time",
			"matern_ard",
			"gaussian_ard" };

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
		* \brief Calculates Matern covariance function for general shape
		* \param dist Distance
		* \param var Marginal variance
		* \param range Transformed range parameter
		* \return Covariance
		*/
		inline double MaternCovarianceGeneralShape(const double dist,
			const double& var,
			const double& range) const {
			double range_dist = range * dist;
			return(var * const_ * std::pow(range_dist, shape_) * std::cyl_bessel_k(shape_, range_dist));
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

		inline void ParseCovFunctionAlias(string_t& likelihood, 
			double& shape) const {
			if (likelihood == string_t("exponential_space_time")) {
				likelihood = "matern_space_time";
				shape = 0.5;
			}
			else if (likelihood == string_t("exponential_ard")) {
				likelihood = "matern_ard";
				shape = 0.5;
			}
		}

		template<typename>
		friend class RECompGP;
	};

}  // namespace GPBoost

#endif   // GPB_COV_FUNCTIONS_
