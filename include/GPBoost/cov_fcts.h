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
#include <GPBoost/DF_utils.h>

#include <string>
#include <set>
#include <string>
#include <vector>
#include <cmath>

#include <LightGBM/utils/log.h>
using LightGBM::Log;

#if (defined(__GNUC__) && !defined(__clang__)) || defined(_MSC_VER)
#define MSVC_OR_GCC_COMPILER 1
#else 
#define MSVC_OR_GCC_COMPILER 0
#endif


namespace GPBoost {

	template<typename T_mat>
	class RECompGP;

	/*!
	* \brief This class implements the covariance functions used for the Gaussian proceses
	*/
	template <class T_mat>
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
		* \param use_precomputed_dist_for_calc_cov If true, precomputed distances ('dist') are used for calculating covariances, otherwise the coordinates are used ('coords' and 'coords_pred')
		*/
		CovFunction(string_t cov_fct_type,
			double shape,
			double taper_range,
			double taper_shape,
			double taper_mu,
			bool apply_tapering,
			int dim_coordinates,
			bool use_precomputed_dist_for_calc_cov) {
			if (cov_fct_type == "exponential_tapered") {
				Log::REFatal("Covariance of type 'exponential_tapered' is discontinued. Use the option 'gp_approx = \"tapering\"' instead ");
			}
			ParseCovFunctionAlias(cov_fct_type, shape);
			if (SUPPORTED_COV_TYPES_.find(cov_fct_type) == SUPPORTED_COV_TYPES_.end()) {
				Log::REFatal("Covariance of type '%s' is not supported ", cov_fct_type.c_str());
			}
			use_precomputed_dist_for_calc_cov_ = use_precomputed_dist_for_calc_cov;
			if (cov_fct_type == "matern_space_time" || cov_fct_type == "matern_ard" || cov_fct_type == "gaussian_ard") {
				is_isotropic_ = false;
				use_precomputed_dist_for_calc_cov_ = false;
			}
			else {
				is_isotropic_ = true;
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
			else if (cov_fct_type == "matern_estimate_shape") {
				num_cov_par_ = 3;
			}
			else {
				num_cov_par_ = 2;
			}
			cov_fct_type_ = cov_fct_type;
			shape_ = shape;
			if (cov_fct_type == "matern" || cov_fct_type == "matern_space_time" || cov_fct_type == "matern_ard") {
				CHECK(shape > 0.);
				if (!(TwoNumbersAreEqual<double>(shape, 0.5) || TwoNumbersAreEqual<double>(shape, 1.5) || TwoNumbersAreEqual<double>(shape, 2.5))) {
#if MSVC_OR_GCC_COMPILER
					const_ = std::pow(2., 1 - shape_) / std::tgamma(shape_);
#else
					// Mathematical special functions are not supported in C++17 by Clang and some other compilers (see https://en.cppreference.com/w/cpp/compiler_support/17#C.2B.2B17_library_features) 
					Log::REFatal("'shape' of %g is not supported for the '%s' covariance function (only 0.5, 1.5, and 2.5) when using this compiler (e.g. Clang on Mac). Use gcc or (a newer version of) MSVC instead. ", shape, cov_fct_type.c_str());
#endif
				}
			}
			else if (cov_fct_type == "powered_exponential") {
				if (shape <= 0. || shape > 2.) {
					Log::REFatal("'shape' needs to be larger than 0 and smaller or equal than 2 for the '%s' covariance function, found %g ", cov_fct_type.c_str(), shape);
				}
			}
			else if (cov_fct_type == "matern_estimate_shape") {
#if !defined(MSVC_OR_GCC_COMPILER)
				// Mathematical special functions are not supported in C++17 by Clang and some other compilers (see https://en.cppreference.com/w/cpp/compiler_support/17#C.2B.2B17_library_features) 
				Log::REFatal("The covariance function '%s' is not supported when using this compiler (e.g. Clang on Mac). Use gcc or (a newer version of) MSVC instead. ", cov_fct_type.c_str());
#endif
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
			InitializeCovFct();
			InitializeCovFctGrad();
			InitializeGetDistanceForCovFct();
			InitializeGetDistanceForGradientCovFct();
		}

		/*! \brief Destructor */
		~CovFunction() {
		}

		string_t CovFunctionName() const {
			return(cov_fct_type_);
		}

		double CovFunctionShape() const {
			return(shape_);
		}

		double CovFunctionTaperRange() const {
			return(taper_range_);
		}

		double CovFunctionTaperShape() const {
			return(taper_shape_);
		}

		bool IsIsotropic() const {
			return(is_isotropic_);
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
		* \brief Define pointers for caclulating distances and potentially scale / transform coordinates for anisotropic covariance functions
		* \param pars Vector with covariance parameters
		* \param coords Original coordinates
		* \param coords_pred Original coordinates
		* \param is_symmmetric If true, dist and sigma are symmetric (e.g., for training data)
		* \param[out] coords_scaled Scaled coordinates
		* \param[out] coords_pred_scaled Scaled coordinates
		* \param[out] coords_ptr
		* \param[out] coords_pred_ptr
		*/
		void DefineCoordsPtrScaleCoords(const vec_t& pars,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			bool is_symmmetric,
			den_mat_t& coords_scaled,
			den_mat_t& coords_pred_scaled,
			const den_mat_t** coords_ptr,
			const den_mat_t** coords_pred_ptr) const {
			if (!is_isotropic_) {// the coordinates are scaled before calculating distances (e.g. for '_ard' or '_space_time' covariance functions)
				ScaleCoordinates(pars, coords, coords_scaled);
				if (!is_symmmetric) {
					ScaleCoordinates(pars, coords_pred, coords_pred_scaled);
				}
			}
			// choose coords to be used (if applicable)
			if (is_symmmetric) {
				if (is_isotropic_) {
					*coords_ptr = &coords;
					*coords_pred_ptr = &coords;
				}
				else {
					*coords_ptr = &coords_scaled;
					*coords_pred_ptr = &coords_scaled;
				}
			}
			else {
				if (is_isotropic_) {
					*coords_ptr = &coords;
					*coords_pred_ptr = &coords_pred;
				}
				else {
					*coords_ptr = &coords_scaled;
					*coords_pred_ptr = &coords_pred_scaled;
				}
			}
		}//end DefineCoordsPtrScaleCoords

		/*!
		* \brief Transform the covariance parameters
		* \param sigma2 Nugget effect / error variance for Gaussian likelihoods
		* \param pars Vector with covariance parameters on orignal scale
		* \param[out] pars_trans Transformed covariance parameters
		*/
		void TransformCovPars(double sigma2,
			const vec_t& pars,
			vec_t& pars_trans) const {
			pars_trans = pars;
			pars_trans[0] = pars[0] / sigma2;
			if (cov_fct_type_ == "matern") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					pars_trans[1] = 1. / pars[1];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					pars_trans[1] = sqrt(3.) / pars[1];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					pars_trans[1] = sqrt(5.) / pars[1];
				}
				else {
					pars_trans[1] = sqrt(2. * shape_) / pars[1];
				}
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
		* \param sigma2 Nugget effect / error variance for Gaussian likelihoods
		* \param pars Vector with covariance parameters
		* \param[out] pars_orig Back-transformed, original covariance parameters
		*/
		void TransformBackCovPars(double sigma2,
			const vec_t& pars,
			vec_t& pars_orig) const {
			pars_orig = pars;
			pars_orig[0] = sigma2 * pars[0];
			if (cov_fct_type_ == "matern") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					pars_orig[1] = 1. / pars[1];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					pars_orig[1] = sqrt(3.) / pars[1];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					pars_orig[1] = sqrt(5.) / pars[1];
				}
				else {
					pars_orig[1] = sqrt(2. * shape_) / pars[1];
				}
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
		* \param is_symmmetric If true, dist and sigma are symmetric (e.g., for training data)
		*/
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<den_mat_t, T_aux>::value>::type* = nullptr >
		void CalculateCovMat(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const vec_t& pars,
			T_mat& sigma,
			bool is_symmmetric) const {
			// some checks
			CHECK(pars.size() == num_cov_par_);
			if (use_precomputed_dist_for_calc_cov_) {
				CHECK(dist.rows() > 0);
				CHECK(dist.cols() > 0);
				if (is_symmmetric) {
					CHECK(dist.rows() == dist.cols());
				}
			}
			else {
				CHECK(coords.rows() > 0);
				CHECK(coords.cols() > 0);
				if (!is_symmmetric) {
					CHECK(coords_pred.rows() > 0);
					CHECK(coords_pred.cols() > 0);
				}
			}
			// define num_rows, num_cols, and sigma
			int num_rows, num_cols;
			if (use_precomputed_dist_for_calc_cov_) {
				num_cols = (int)dist.cols();
				num_rows = (int)dist.rows();
			}
			else {
				num_cols = (int)coords.rows();
				if (is_symmmetric) {
					num_rows = (int)coords.rows();
				}
				else {
					num_rows = (int)coords_pred.rows();
				}
			}
			sigma = T_mat(num_rows, num_cols);
			if (cov_fct_type_ == "wendland") {
				// initialize Wendland covariance matrix. Note: this dense matrix version is usually not used
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)dist.rows(); ++i) {
					for (int j = 1; j < (int)dist.cols(); ++j) {
						if (dist.coeff(i, j) >= taper_range_) {
							sigma(i, j) = 0.;
						}
						else {
							sigma(i, j) = pars[0];
						}
					}
				}
				MultiplyWendlandCorrelationTaper<T_mat>(dist, sigma, is_symmmetric);
			}// end cov_fct_type_ == "wendland"
			else {// cov_fct_type_ != "wendland"
				den_mat_t coords_scaled, coords_pred_scaled;
				const den_mat_t* coords_ptr = nullptr;
				const den_mat_t* coords_pred_ptr = nullptr;
				if (!use_precomputed_dist_for_calc_cov_) {
					DefineCoordsPtrScaleCoords(pars, coords, coords_pred, is_symmmetric, coords_scaled, coords_pred_scaled, &coords_ptr, &coords_pred_ptr);
				}
				double range, shape = 0.;
				range = !is_isotropic_ ? 1. : pars[1];
				if (cov_fct_type_ == "matern_estimate_shape") {
					shape = pars[2];
				}
				// calculate covariances
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						sigma(i, i) = pars[0];
						for (int j = i + 1; j < num_cols; ++j) {
							double dist_ij = GetDistanceForCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							sigma(i, j) = CovFct_(dist_ij, pars[0], range, shape);
							sigma(j, i) = sigma(i, j);
						}// end loop cols
					}// end loop rows
				}// end is_symmmetric
				else {// !is_symmmetric
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						for (int j = 0; j < num_cols; ++j) {
							double dist_ij = GetDistanceForCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							sigma(i, j) = CovFct_(dist_ij, pars[0], range, shape);
						}// end loop cols
					}// end loop rows
				}// end !is_symmmetric
			}// end cov_fct_type_ != "wendland"
		}// end CalculateCovMat (dense)
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
		void CalculateCovMat(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const vec_t& pars,
			T_mat& sigma,
			bool is_symmmetric) const {
			// some checks
			CHECK(pars.size() == num_cov_par_);
			CHECK(dist.rows() > 0);//dist is used to define sigma
			CHECK(dist.cols() > 0);
			if (is_symmmetric) {
				CHECK(dist.rows() == dist.cols());
			}
			if (!use_precomputed_dist_for_calc_cov_) {
				CHECK(coords.rows() > 0);
				CHECK(coords.cols() > 0);
				CHECK(coords.rows() == dist.cols());
				if (is_symmmetric) {
					CHECK(coords.rows() == dist.rows());
				}
				else {
					CHECK(coords_pred.rows() > 0);
					CHECK(coords_pred.cols() > 0);
					CHECK(coords_pred.rows() == dist.rows());
				}
			}
			// define sigma
			sigma = dist;
			sigma.makeCompressed();
			if (cov_fct_type_ == "wendland") {
				sigma.coeffs() = pars[0];
				MultiplyWendlandCorrelationTaper<T_mat>(dist, sigma, is_symmmetric);
			}
			else {// cov_fct_type_ != "wendland"
				den_mat_t coords_scaled, coords_pred_scaled;
				const den_mat_t* coords_ptr = nullptr;
				const den_mat_t* coords_pred_ptr = nullptr;
				if (!use_precomputed_dist_for_calc_cov_) {
					DefineCoordsPtrScaleCoords(pars, coords, coords_pred, is_symmmetric, coords_scaled, coords_pred_scaled, &coords_ptr, &coords_pred_ptr);
				}
				double range, shape = 0.;
				range = !is_isotropic_ ? 1. : pars[1];
				if (cov_fct_type_ == "matern_estimate_shape") {
					shape = pars[2];
				}
				// calculate covariances
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
								double dist_ij = GetDistanceForCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
								it.valueRef() = CovFct_(dist_ij, pars[0], range, shape);
								sigma.coeffRef(j, i) = it.value();
							}
						}
					}
				}// end is_symmmetric
				else {// !is_symmmetric
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							double dist_ij = GetDistanceForCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							it.valueRef() = CovFct_(dist_ij, pars[0], range, shape);
						}
					}
				}// end !is_symmmetric
			}// end cov_fct_type_ != "wendland"
		}//end CalculateCovMat (sparse)

		/*!
		* \brief Covariance function for one distance value
		* \param dist Distance
		* \param pars Vector with covariance parameters
		* \param[out] sigma Covariance at dist
		*/
		void CalculateCovMat(double dist,
			const vec_t& pars,
			double& sigma) const {
			CHECK(pars.size() == num_cov_par_);
			if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard") {
				Log::REFatal("'CalculateCovMat()' is not implemented for one distance when cov_fct_type_ == '%s' ", cov_fct_type_.c_str());
			}
			else if (cov_fct_type_ == "wendland") {
				// note: this is usually not used
				if (dist >= taper_range_) {
					sigma = 0.;
				}
				else {
					sigma = pars[0];
					MultiplyWendlandCorrelationTaper(dist, sigma);
				}
			}
			else {
				double shape = (cov_fct_type_ == "matern_estimate_shape") ? pars[2] : 0.;
				sigma = CovFct_(dist, pars[0], pars[1], shape);
			}
		}//end CalculateCovMat (one single entry)

		/*!
		* \brief Multiply covariance matrix element-wise with Wendland correlation tapering function
		* \param dist Distance matrix
		* \param[out] sigma Covariance matrix
		* \param is_symmmetric Set to true if dist and sigma are symmetric (e.g. for training data)
		*/
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<den_mat_t, T_aux>::value>::type* = nullptr >
		void MultiplyWendlandCorrelationTaper(const T_mat& dist,
			T_mat& sigma,
			bool is_symmmetric) const {
			CHECK(apply_tapering_);
			if (is_symmmetric) {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)dist.rows(); ++i) {
					for (int j = i + 1; j < (int)dist.cols(); ++j) {
						if (TwoNumbersAreEqual<double>(taper_shape_, 0.)) {
							sigma(i, j) *= WendlandCorrelationShape0(dist(i, j));
						}
						else if (TwoNumbersAreEqual<double>(taper_shape_, 1.)) {
							sigma(i, j) *= WendlandCorrelationShape1(dist(i, j));
						}
						else if (TwoNumbersAreEqual<double>(taper_shape_, 2.)) {
							sigma(i, j) *= WendlandCorrelationShape2(dist(i, j));
						}
						else {
							Log::REFatal("MultiplyWendlandCorrelationTaper: 'taper_shape' of %g is not supported for the 'wendland' covariance function ", taper_shape_);
						}
						sigma(j, i) = sigma(i, j);
					}
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)dist.rows(); ++i) {
					for (int j = 0; j < (int)dist.cols(); ++j) {
						if (TwoNumbersAreEqual<double>(taper_shape_, 0.)) {
							sigma(i, j) *= WendlandCorrelationShape0(dist(i, j));
						}
						else if (TwoNumbersAreEqual<double>(taper_shape_, 1.)) {
							sigma(i, j) *= WendlandCorrelationShape1(dist(i, j));
						}
						else if (TwoNumbersAreEqual<double>(taper_shape_, 2.)) {
							sigma(i, j) *= WendlandCorrelationShape2(dist(i, j));
						}
						else {
							Log::REFatal("MultiplyWendlandCorrelationTaper: 'taper_shape' of %g is not supported for the 'wendland' covariance function ", taper_shape_);
						}
					}
				}
			}
		}//end MultiplyWendlandCorrelationTaper (dense)
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
		void MultiplyWendlandCorrelationTaper(const T_mat& dist,
			T_mat& sigma,
			bool is_symmmetric) const {
			CHECK(apply_tapering_);
			if (is_symmmetric) {
#pragma omp parallel for schedule(static)
				for (int k = 0; k < sigma.outerSize(); ++k) {
					for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
						int i = (int)it.row();
						int j = (int)it.col();
						if (i < j) {
							if (TwoNumbersAreEqual<double>(taper_shape_, 0.)) {
								it.valueRef() *= WendlandCorrelationShape0(dist.coeff(i, j));
							}
							else if (TwoNumbersAreEqual<double>(taper_shape_, 1.)) {
								it.valueRef() *= WendlandCorrelationShape1(dist.coeff(i, j));
							}
							else if (TwoNumbersAreEqual<double>(taper_shape_, 2.)) {
								it.valueRef() *= WendlandCorrelationShape2(dist.coeff(i, j));
							}
							else {
								Log::REFatal("MultiplyWendlandCorrelationTaper: 'taper_shape' of %g is not supported for the 'wendland' covariance function ", taper_shape_);
							}
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
						if (TwoNumbersAreEqual<double>(taper_shape_, 0.)) {
							it.valueRef() *= WendlandCorrelationShape0(dist.coeff(i, j));
						}
						else if (TwoNumbersAreEqual<double>(taper_shape_, 1.)) {
							it.valueRef() *= WendlandCorrelationShape1(dist.coeff(i, j));
						}
						else if (TwoNumbersAreEqual<double>(taper_shape_, 2.)) {
							it.valueRef() *= WendlandCorrelationShape2(dist.coeff(i, j));
						}
						else {
							Log::REFatal("MultiplyWendlandCorrelationTaper: 'taper_shape' of %g is not supported for the 'wendland' covariance function ", taper_shape_);
						}
					}
				}
			}
		}//end MultiplyWendlandCorrelationTaper (sparse)

		/*!
		* \brief Multiply covariance with Wendland correlation tapering function for one distance value
		* \param dist Distance
		* \param[out] sigma Covariance at dist after applying tapering
		*/
		void MultiplyWendlandCorrelationTaper(double dist,
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
				Log::REFatal("MultiplyWendlandCorrelationTaper: 'taper_shape' of %g is not supported for the 'wendland' covariance function ", taper_shape_);
			}
		}//end MultiplyWendlandCorrelationTaper (double)

		/*!
		* \brief Calculates derivatives of the covariance matrix with respect to covariance parameters such as range and smoothness parameters (except marginal variance parameters)
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
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<den_mat_t, T_aux>::value>::type* = nullptr >
		void CalculateGradientCovMat(const T_mat& dist,
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
			if (use_precomputed_dist_for_calc_cov_) {
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
			double cm, cm_num_deriv, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape;//constants and auxiliary parameters
			DetermineConstantsForGradient(pars, (int)coords.cols(), transf_scale, nugget_var, ind_range, 
				cm, cm_num_deriv, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape);
			// define num_rows, num_cols, and sigma
			int num_rows, num_cols;
			if (use_precomputed_dist_for_calc_cov_) {
				num_cols = (int)dist.cols();
				num_rows = (int)dist.rows();
			}
			else {
				num_cols = (int)coords.rows();
				if (is_symmmetric) {
					num_rows = (int)coords.rows();
				}
				else {
					num_rows = (int)coords_pred.rows();
				}
			}
			sigma_grad = T_mat(sigma.rows(), sigma.cols());
			den_mat_t coords_scaled, coords_pred_scaled;
			const den_mat_t* coords_ptr = nullptr;
			const den_mat_t* coords_pred_ptr = nullptr;
			if (!use_precomputed_dist_for_calc_cov_) {
				DefineCoordsPtrScaleCoords(pars, coords, coords_pred, is_symmmetric, coords_scaled, coords_pred_scaled, &coords_ptr, &coords_pred_ptr);
			}
			// calculate gradients
			if (is_symmmetric) {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_rows; ++i) {
					sigma_grad(i, i) = 0.;
					for (int j = i + 1; j < num_cols; ++j) {
						double dist_ij = 0.;
						GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr, dist_ij);
						sigma_grad(i, j) = GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
							ind_range, i, j, dist_ij, sigma, coords_ptr, coords_pred_ptr);
						sigma_grad(j, i) = sigma_grad(i, j);
					}// end loop over cols
				}// end loop over rows
			}// end is_symmmetric
			else {// !is_symmmetric
#pragma omp parallel for schedule(static)
				for (int i = 0; i < num_rows; ++i) {
					for (int j = 0; j < num_cols; ++j) {
						double dist_ij = 0.;
						GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr, dist_ij);
						sigma_grad(i, j) = GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
							ind_range, i, j, dist_ij, sigma, coords_ptr, coords_pred_ptr);
					}// end loop over cols
				}// end loop over rows
			}// end !is_symmmetric
		}//end CalculateGradientCovMat (dense)
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
		void CalculateGradientCovMat(const T_mat& dist,
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
			if (use_precomputed_dist_for_calc_cov_) {
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
			double cm, cm_num_deriv, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape;//constants and auxiliary parameters
			DetermineConstantsForGradient(pars, (int)coords.cols(), transf_scale, nugget_var, ind_range,
				cm, cm_num_deriv, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape);
			sigma_grad = T_mat(sigma.rows(), sigma.cols());
			den_mat_t coords_scaled, coords_pred_scaled;
			const den_mat_t* coords_ptr = nullptr;
			const den_mat_t* coords_pred_ptr = nullptr;
			if (!use_precomputed_dist_for_calc_cov_) {
				DefineCoordsPtrScaleCoords(pars, coords, coords_pred, is_symmmetric, coords_scaled, coords_pred_scaled, &coords_ptr, &coords_pred_ptr);
			}
			// calculate gradients
			sigma_grad = sigma;
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
							double dist_ij = 0.;
							GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr, dist_ij);
							it.valueRef() = GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
								ind_range, i, j, dist_ij, sigma, coords_ptr, coords_pred_ptr);
							sigma_grad.coeffRef(j, i) = it.value();
						}
					}// end loop over cols
				}// end loop over rows
			}// end is_symmmetric
			else {// !is_symmmetric
#pragma omp parallel for schedule(static)
				for (int k = 0; k < sigma_grad.outerSize(); ++k) {
					for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
						int i = (int)it.row();
						int j = (int)it.col();
						double dist_ij = 0.;
						GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr, dist_ij);
						it.valueRef() = GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
							ind_range, i, j, dist_ij, sigma, coords_ptr, coords_pred_ptr);
					}// end loop over cols
				}// end loop over rows
			}// end !is_symmmetric
		}//end CalculateGradientCovMat (sparse)

		/*!
		* \brief Find "reasonable" default values for the intial values of the covariance parameters (on transformed scale)
		* \param dist Distance matrix
		* \param coords Coordinates matrix
		* \param use_distances If true, 'dist' is used, otherwise 'coords' is used
		* \param rng Random number generator
		* \param[out] pars Vector with covariance parameters
		* \param marginal_variance Initial value for marginal variance
		*/
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
				if (cov_fct_type_ == "matern") {
					if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
						pars[1] = 2. * 3. / mean_dist;
					}
					else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
						pars[1] = 2. * 4.7 / mean_dist;
					}
					else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
						pars[1] = 2. * 5.9 / mean_dist;
					}
					else {
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
				}
				else if (cov_fct_type_ == "matern_estimate_shape") {
					pars[1] = 2. * 4.7 * mean_dist / std::sqrt(3.);//same as shape_ = 1.5
					pars[2] = 1.5;
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
		/*! \brief If true, the covariance function is isotropic */
		bool is_isotropic_;
		/*! \brief for calculating finite differences  */
		const double delta_step_ = 1e-6;// based on https://math.stackexchange.com/questions/815113/is-there-a-general-formula-for-estimating-the-step-size-h-in-numerical-different/819015#819015
		/*! \brief If true, precomputed distances('dist') are used for calculating covariances, otherwise the coordinates are used('coords' and 'coords_pred') */
		bool use_precomputed_dist_for_calc_cov_;
		/*! \brief List of supported covariance functions */
		const std::set<string_t> SUPPORTED_COV_TYPES_{ "exponential",
			"gaussian",
			"powered_exponential",
			"matern",
			"wendland",
			"matern_space_time",
			"matern_ard",
			"gaussian_ard",
			"matern_estimate_shape" };

		/*!
		* \brief Calculates Wendland correlation function if taper_shape == 0
		* \param dist Distance
		* \return Wendland correlation
		*/
		inline double WendlandCorrelationShape0(double dist) const {
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
		inline double WendlandCorrelationShape1(double dist) const {
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
		inline double WendlandCorrelationShape2(double dist) const {
			if (dist < EPSILON_NUMBERS) {
				return(1.);
			}
			else {
				return(std::pow((1. - dist / taper_range_), taper_mu_ + 2.) *
					(1. + dist / taper_range_ * (taper_mu_ + 2.) + std::pow(dist / taper_range_, 2) * (taper_mu_ * taper_mu_ + 4 * taper_mu_ + 3.) / 3.));
			}
		}

		/*!
		* \brief Generic function for determininig distances for calculating covariances
		*/
		std::function<double(const int i, const int j, const T_mat& /* dist */, 
			const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */)> GetDistanceForCovFct_;

		void InitializeGetDistanceForCovFct() {
			if (use_precomputed_dist_for_calc_cov_) {
				GetDistanceForCovFct_ = [this](const int i, const int j, const T_mat& dist, 
					const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
					return(dist.coeff(i, j));
				};
			}
			else {
				GetDistanceForCovFct_ = [this](const int i, const int j, const T_mat& /* dist */, 
					const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
					return(((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>());
				};
			}
		}//end InitializeGetDistanceForCovFct_

		/*!
		* \brief Generic function for calculating covariances
		* \param dist Distance
		* \param var Marginal variance
		* \param range Transformed range parameter
		* \param shape Smoothness parameter (if applicable)
		* \return Covariance
		*/
		std::function<double(double /* dist_ij */, double /* var */, double /* range */, double /* shape */)> CovFct_;

		void InitializeCovFct() {
			if (cov_fct_type_ == "matern" || cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					CovFct_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape0_5(dist_ij, var, range);
					};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					CovFct_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape1_5(dist_ij, var, range);
					};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					CovFct_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape2_5(dist_ij, var, range);
					};
				}
				else {//general shape
					CovFct_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternGeneralShape(dist_ij, var, range);
					};
				}
			}//end cov_fct_type_ == "matern" || cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard"
			else if (cov_fct_type_ == "matern_estimate_shape") {
				CovFct_ = [this](double dist_ij, double var, double range, double shape) -> double {
					return CovarianceMaternEstimateShape(dist_ij, var, range, shape);
				};
			}
			else if (cov_fct_type_ == "gaussian" || cov_fct_type_ == "gaussian_ard") {
				CovFct_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
					return CovarianceGaussian(dist_ij, var, range);
				};
			}
			else if (cov_fct_type_ == "powered_exponential") {
				CovFct_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
					return CovariancePoweredExponential(dist_ij, var, range);
				};
			}
			else if (cov_fct_type_ != "wendland") {
				Log::REFatal("InitializeCovFct: covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end InitializeCovFct

		inline double CovarianceMaternShape0_5(double dist,
			double var,
			double range) const {
			return(var * std::exp(-range * dist));
		}

		inline double CovarianceMaternShape1_5(double dist,
			double var,
			double range) const {
			double range_dist = range * dist;
			return(var * (1. + range_dist) * std::exp(-range_dist));
		}

		inline double CovarianceMaternShape2_5(double dist,
			double var,
			double range) const {
			double range_dist = range * dist;
			return(var * (1. + range_dist + range_dist * range_dist / 3.) * std::exp(-range_dist));
		}

		inline double CovarianceMaternGeneralShape(double dist,
			double var,
			double range) const {
			double range_dist = range * dist;
			if (range_dist <= 0.) {
				return(var);
			}
			else {
#if MSVC_OR_GCC_COMPILER
				return(var * const_ * std::pow(range_dist, shape_) * std::cyl_bessel_k(shape_, range_dist));
#else
				return(1.);
#endif
			}
		}//end CovarianceMaternGeneralShape

		inline double CovarianceMaternEstimateShape(double dist,
			double var,
			double range,
			double shape) const {
			CHECK(shape > 0.);
			double range_dist = dist * std::sqrt(2. * shape) / range;
			if (range_dist <= 0.) {
				return(var);
			}
			else {
#if MSVC_OR_GCC_COMPILER
				return(var * std::pow(2., 1 - shape) / std::tgamma(shape) * std::pow(range_dist, shape) * std::cyl_bessel_k(shape, range_dist));
#else
				return(1.);
#endif
			}
		}//end CovarianceMaternEstimateShape

		inline double CovarianceGaussian(double dist,
			double var,
			double range) const {
			return(var * std::exp(-range * dist * dist));
		}

		inline double CovariancePoweredExponential(double dist,
			double var,
			double range) const {
			return(var * std::exp(-range * std::pow(dist, shape_)));
		}

		/*!
		* \brief Determine constants required for calculating gradients of covariances
		*/
		inline void DetermineConstantsForGradient(const vec_t& pars,
			int dim_coords,
			bool transf_scale,
			double nugget_var,
			int ind_range,
			double& cm,
			double& cm_num_deriv,
			double& par_aux,
			double& pars_2_up,
			double& pars_2_down,
			double& par_aux_up,
			double& par_aux_down,
			double& shape) const {
			if (cov_fct_type_ == "matern") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					cm = transf_scale ? (-1. * pars[1]) : (nugget_var * pars[1] * pars[1]);
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (nugget_var * pars[0] * std::pow(pars[1], 3) / sqrt(3.));
					par_aux = pars[1];
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (nugget_var * pars[0] * std::pow(pars[1], 3) / sqrt(5.));
					par_aux = pars[1];
				}
				else {//general shape
					cm = transf_scale ? 1. : (-nugget_var * pars[1] / std::sqrt(2. * shape_));
					cm *= pars[0] * const_;
					par_aux = pars[1];
				}
			}
			else if (cov_fct_type_ == "gaussian") {
				cm = transf_scale ? (-1. * pars[1]) : (2. * nugget_var * std::pow(pars[1], 3. / 2.));
			}
			else if (cov_fct_type_ == "powered_exponential") {
				cm = transf_scale ? (-1. * pars[1]) : (shape_ * nugget_var * std::pow(pars[1], (shape_ + 1.) / shape_));
			}
			else if (cov_fct_type_ == "matern_estimate_shape") {
				CHECK(ind_range >= 0 && ind_range <= 1);
				if (ind_range == 0) {
					cm = transf_scale ? 1. : nugget_var / pars[1];
					cm *= -pars[0] * std::pow(2., 1 - pars[2]) / std::tgamma(pars[2]);
					par_aux = std::sqrt(2. * pars[2]) / pars[1];
				}
				else if (ind_range == 1) {//gradient wrt smoothness parameter
					//for calculating finite differences
					cm = transf_scale ? pars[2] : nugget_var;
					cm *= pars[0] * std::pow(2., 1 - pars[2]) / std::tgamma(pars[2]);
					par_aux = std::sqrt(2. * pars[2]) / pars[1];
					if (transf_scale) {
						cm_num_deriv = pars[0] * std::pow(2., 1 - pars[2]) / std::tgamma(pars[2]);
						pars_2_up = std::exp(std::log(pars[2]) + delta_step_);//gradient on log-scale
						pars_2_down = std::exp(std::log(pars[2]) - delta_step_);
					}
					else {
						cm_num_deriv = cm;
						pars_2_up = pars[2] + delta_step_;
						pars_2_down = pars[2] - delta_step_;
						CHECK(pars_2_down > 0.);
					}
					par_aux_up = std::sqrt(2. * pars_2_up) / pars[1];
					par_aux_down = std::sqrt(2. * pars_2_down) / pars[1];
				}
				shape = pars[2];
			}// end cov_fct_type_ == "matern_estimate_shape"
			else if (cov_fct_type_ == "matern_space_time") {
				CHECK(ind_range >= 0 && ind_range <= 1);
				// calculate constants that are the same for all entries
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					cm = transf_scale ? -1. : (nugget_var * pars[ind_range + 1]);
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[ind_range + 1] / sqrt(3.));
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[ind_range + 1] / sqrt(5.));
				}
				else {//general shape
					cm = transf_scale ? 1. : (-nugget_var * pars[ind_range + 1] / sqrt(2. * shape_));
					cm *= pars[0] * const_;
				}
			}// end cov_fct_type_ == "matern_space_time"
			else if (cov_fct_type_ == "matern_ard") {
				CHECK(ind_range >= 0 && ind_range < dim_coords);
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					cm = transf_scale ? -1. : (nugget_var * pars[ind_range + 1]);
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[ind_range + 1] / sqrt(3.));
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[ind_range + 1] / sqrt(5.));
				}
				else {//general shape
					cm = transf_scale ? 1. : (-nugget_var * pars[ind_range + 1] / sqrt(2. * shape_));
					cm *= pars[0] * const_;
				}
			}// end (cov_fct_type_ == "matern_ard"
			else if (cov_fct_type_ == "gaussian_ard") {
				CHECK(ind_range >= 0 && ind_range < dim_coords);
				cm = transf_scale ? -1. : (2. * nugget_var * std::sqrt(pars[1]));
			}
		}//end DetermineConstantsForGradient

		/*!
		* \brief Generic function for determining distances for calculating gradients of covariances
		*/
		std::function<void(const int /* i */, const int /* j */, const T_mat& /* dist */,
			const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */, double& /* dist_ij */)> GetDistanceForGradientCovFct_;

		void InitializeGetDistanceForGradientCovFct() {
			if (is_isotropic_) {
				if (use_precomputed_dist_for_calc_cov_) {
					GetDistanceForGradientCovFct_ = [this](const int i, const int j, const T_mat& dist,
						const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */, double& dist_ij) {
							dist_ij = dist.coeff(i, j);
					};
				}
				else {
					GetDistanceForGradientCovFct_ = [this](const int i, const int j, const T_mat& /* dist */,
						const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr, double& dist_ij) {
							dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
					};
				}
			}
			else {
				GetDistanceForGradientCovFct_ = [this](const int /* i */, const int /* j */, const T_mat& /* dist */,
					const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */, double& /* dist_ij */) {
						return;//do nothing, distances are not calculated here but in the corresponding gradient functions
				};
			}
		}//end InitializeGetDistanceForGradientCovFct

		/*!
		* \brief Generic function for calculating gradients of covariances wrt range and other parameters such as smoothness
		*/
		std::function<double(double /* cm */, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
			double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
			const int /* ind_range */, const int /* i */, const int /* j */,
			const double /* dist_ij */, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */)> GradientCovFct_;

		void InitializeCovFctGrad() {
			if (cov_fct_type_ == "matern") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int /* ind_range */, const int i, const int j,
						const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
							return GradientRangeMaternShape0_5(cm, dist_ij, sigma, i, j);
					};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double par_aux, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int /* ind_range */, const int /* i */, const int /* j */,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
							return GradientRangeMaternShape1_5(cm, dist_ij, par_aux);
					};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double par_aux, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int /* ind_range */, const int /* i */, const int /* j */,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
							return GradientRangeMaternShape2_5(cm, dist_ij, par_aux);
					};
				}
				else {// general shape
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double par_aux, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int /* ind_range */, const int /* i */, const int /* j */,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
							return GradientRangeMaternGeneralShape(cm, dist_ij, par_aux, shape_);
					};
				}
			}//end matern
			else if (cov_fct_type_ == "gaussian") {
				GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int /* ind_range */, const int i, const int j,
					const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientRangeGaussian(cm, dist_ij, sigma, i, j);
				};
			}
			else if (cov_fct_type_ == "powered_exponential") {
				GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int /* ind_range */, const int i, const int j,
					const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientRangePoweredExponential(cm, dist_ij, sigma, i, j);
				};
			}			
			else if (cov_fct_type_ == "matern_space_time") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_range, const int i, const int j,
						const double /* dist_ij */, const T_mat& sigma, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternSpaceTimeShape0_5(cm, sigma, ind_range, i, j, coords_ptr, coords_pred_ptr);
					};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_range, const int i, const int j,
						const double /* dist_ij */, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternSpaceTimeShape1_5(cm, ind_range, i, j, coords_ptr, coords_pred_ptr);
					};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_range, const int i, const int j,
						const double /* dist_ij */, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternSpaceTimeShape2_5(cm, ind_range, i, j, coords_ptr, coords_pred_ptr);
					};
				}
				else {// general shape
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_range, const int i, const int j,
						const double /* dist_ij */, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternSpaceTimeGeneralShape(cm, ind_range, i, j, coords_ptr, coords_pred_ptr);
					};
				}
			}//end matern_space_time
			else if (cov_fct_type_ == "matern_ard") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_range, const int i, const int j,
						const double /* dist_ij */, const T_mat& sigma, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternARDShape0_5(cm, sigma, ind_range, i, j, coords_ptr, coords_pred_ptr);
					};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_range, const int i, const int j,
						const double /* dist_ij */, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternARDShape1_5(cm, ind_range, i, j, coords_ptr, coords_pred_ptr);
					};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_range, const int i, const int j,
						const double /* dist_ij */, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternARDShape2_5(cm, ind_range, i, j, coords_ptr, coords_pred_ptr);
					};
				}
				else {// general shape
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_range, const int i, const int j,
						const double /* dist_ij */, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternARDGeneralShape(cm, ind_range, i, j, coords_ptr, coords_pred_ptr);
					};
				}
			}//end matern_ard
			else if (cov_fct_type_ == "matern_estimate_shape") {
				GradientCovFct_ = [this](double cm, double cm_num_deriv, double par_aux, double shape,
					double par_aux_up, double par_aux_down, double pars_2_up, double pars_2_down,
					const int ind_range, const int /* i */, const int /* j */,
					const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientMaternEstimateShape(cm, cm_num_deriv, dist_ij, par_aux, par_aux_up, par_aux_down, pars_2_up, pars_2_down, shape, ind_range);
				};
			}
			else if (cov_fct_type_ == "gaussian_ard") {
				GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int ind_range, const int i, const int j,
					const double /* dist_ij */, const T_mat& sigma, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
						return GradientRangeGaussianARD(cm, sigma, ind_range, i, j, coords_ptr, coords_pred_ptr);
				};
			}
			else if (cov_fct_type_ != "wendland" && cov_fct_type_ != "powered_exponential" && cov_fct_type_ != "gaussian") {
				Log::REFatal("InitializeCovFctGrad: covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}//end InitializeCovFctGrad

		inline double GradientRangeMaternShape0_5(double cm,
			const double dist_ij,
			const T_mat& sigma,
			const int i,
			const int j) const {
			return(cm * dist_ij * sigma.coeff(i, j));
		}

		inline double GradientRangeMaternShape1_5(double cm,
			const double dist_ij,
			double range) const {
			return(cm * dist_ij * dist_ij * std::exp(-range * dist_ij));
		}

		inline double GradientRangeMaternShape2_5(double cm,
			const double dist_ij,
			double range) const {
			double range_dist = range * dist_ij;
			return(cm / 3. * dist_ij * dist_ij * (1. + range_dist) * std::exp(-range_dist));
		}

		inline double GradientRangeGaussian(double cm,
			const double dist_ij,
			const T_mat& sigma,
			const int i,
			const int j) const {
			return(cm * dist_ij * dist_ij * sigma.coeff(i, j));
		}

		inline double GradientRangePoweredExponential(double cm,
			const double dist_ij,
			const T_mat& sigma,
			const int i,
			const int j) const {
			return(cm * std::pow(dist_ij, shape_) * sigma.coeff(i, j));
		}

		inline double GradientRangeMaternGeneralShape(double cm,
			const double dist_ij,
			double cm_dist,
			double shape) const {
#if MSVC_OR_GCC_COMPILER
			double range_dist = cm_dist * dist_ij;
			return(cm * std::pow(range_dist, shape) * (2. * shape * std::cyl_bessel_k(shape, range_dist) - range_dist * std::cyl_bessel_k(shape + 1., range_dist)));
#else
			return(1.);
#endif
		}//end GradientRangeMaternGeneralShape

		inline double GradientSmoothnessMaternEstimateShapesFiniteDifference(double cm,
			double cm_num_deriv,
			double dist_ij,
			double par_aux,
			double par_aux_up,
			double par_aux_down,
			double pars_2_up,
			double pars_2_down,
			double shape) const {
#if MSVC_OR_GCC_COMPILER
			double z = dist_ij * par_aux;
			double z_up = dist_ij * par_aux_up;
			double z_down = dist_ij * par_aux_down;
			double bessel_num_deriv = (std::cyl_bessel_k(pars_2_up, z_up) - std::cyl_bessel_k(pars_2_down, z_down)) / (2. * delta_step_);
			return (std::pow(z, shape) * (cm * std::cyl_bessel_k(shape, z) * (std::log(z / 2.) + 0.5 - GPBoost::digamma(shape)) + cm_num_deriv * bessel_num_deriv));
#else
			return(1.);
#endif
		}//end GradientSmoothnessMaternEstimateShapesFiniteDifference

		inline double GradientMaternEstimateShape(double cm,
			double cm_num_deriv,
			const double dist_ij,
			double par_aux,
			double par_aux_up,
			double par_aux_down,
			double pars_2_up,
			double pars_2_down,
			double shape,
			const int ind_range) const {
			if (ind_range == 0) {
				return(GradientRangeMaternGeneralShape(cm, dist_ij, par_aux, shape));
			}
			else if (ind_range == 1) {//gradient wrt smoothness parameter
				return(GradientSmoothnessMaternEstimateShapesFiniteDifference(cm, cm_num_deriv, dist_ij, par_aux,
					par_aux_up, par_aux_down, pars_2_up, pars_2_down, shape));
			}
			return(1.);
		}//end GradientMaternEstimateShape

		inline double GradientRangeMaternSpaceTimeShape0_5(double cm,
			const T_mat& sigma,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			int dim_space = (int)(*coords_ptr).cols() - 1;
			double dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
			if (ind_range == 0) {
				double dist_sq_ij_time = ((*coords_pred_ptr).coeff(i, 0) - (*coords_ptr).coeff(j, 0));
				dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
				if (dist_sq_ij_time < EPSILON_NUMBERS) {
					return(0.);
				}
				else {
					return(cm * dist_sq_ij_time / dist_ij * sigma.coeff(i, j));
				}
			}
			else {// ind_range == 1
				double dist_sq_ij_space = ((*coords_pred_ptr).row(i).tail(dim_space) - (*coords_ptr).row(j).tail(dim_space)).squaredNorm();
				if (dist_sq_ij_space < EPSILON_NUMBERS) {
					return(0.);
				}
				else {
					return(cm * dist_sq_ij_space / dist_ij * sigma.coeff(i, j));
				}
			}
		}//end GradientRangeMaternSpaceTimeShape0_5

		inline double GradientRangeMaternSpaceTimeShape1_5(double cm,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			int dim_space = (int)(*coords_ptr).cols() - 1;
			double dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
			if (ind_range == 0) {
				double dist_sq_ij_time = ((*coords_pred_ptr).coeff(i, 0) - (*coords_ptr).coeff(j, 0));
				dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
				return(cm * dist_sq_ij_time * std::exp(-dist_ij));
			}
			else {// ind_range == 1
				double dist_sq_ij_space = ((*coords_pred_ptr).row(i).tail(dim_space) - (*coords_ptr).row(j).tail(dim_space)).squaredNorm();
				return(cm * dist_sq_ij_space * std::exp(-dist_ij));
			}
		}//end GradientRangeMaternSpaceTimeShape1_5

		inline double GradientRangeMaternSpaceTimeShape2_5(double cm,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			int dim_space = (int)(*coords_ptr).cols() - 1;
			double dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
			if (ind_range == 0) {
				double dist_sq_ij_time = ((*coords_pred_ptr).coeff(i, 0) - (*coords_ptr).coeff(j, 0));
				dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
				return(cm * dist_sq_ij_time * (1 + dist_ij) * std::exp(-dist_ij));
			}
			else {// ind_range == 1
				double dist_sq_ij_space = ((*coords_pred_ptr).row(i).tail(dim_space) - (*coords_ptr).row(j).tail(dim_space)).squaredNorm();
				return(cm * dist_sq_ij_space * (1 + dist_ij) * std::exp(-dist_ij));
			}
		}//end GradientRangeMaternSpaceTimeShape2_5

		inline double GradientRangeMaternSpaceTimeGeneralShape(double cm,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
#if MSVC_OR_GCC_COMPILER
			int dim_space = (int)(*coords_ptr).cols() - 1;
			double dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
			if (ind_range == 0) {
				double dist_sq_ij_time = ((*coords_pred_ptr).coeff(i, 0) - (*coords_ptr).coeff(j, 0));
				dist_sq_ij_time = dist_sq_ij_time * dist_sq_ij_time;
				return(cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_time);
			}
			else {// ind_range == 1
				double dist_sq_ij_space = ((*coords_pred_ptr).row(i).tail(dim_space) - (*coords_ptr).row(j).tail(dim_space)).squaredNorm();
				return(cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_space);
			}
#else
			return(1.);
#endif
		}//end GradientRangeMaternSpaceTimeGeneralShape

		inline double GradientRangeMaternARDShape0_5(double cm,
			const T_mat& sigma,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			double dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
			double dist_sq_ij_coord = ((*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range));
			dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
			if (dist_sq_ij_coord < EPSILON_NUMBERS) {
				return(0.);
			}
			else {
				return(cm * dist_sq_ij_coord / dist_ij * sigma.coeff(i, j));
			}
		}//end GradientRangeMaternARDShape0_5

		inline double GradientRangeMaternARDShape1_5(double cm,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			double dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
			double dist_sq_ij_coord = ((*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range));
			dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
			return(cm * dist_sq_ij_coord * std::exp(-dist_ij));
		}//end GradientRangeMaternARDShape1_5

		inline double GradientRangeMaternARDShape2_5(double cm,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			double dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
			double dist_sq_ij_coord = ((*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range));
			dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
			return(cm * dist_sq_ij_coord * (1 + dist_ij) * std::exp(-dist_ij));
		}//end GradientRangeMaternARDShape2_5

		inline double GradientRangeMaternARDGeneralShape(double cm,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			double dist_ij = ((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>();
			double dist_sq_ij_coord = ((*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range));
			dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
#if MSVC_OR_GCC_COMPILER
			return(cm * std::pow(dist_ij, shape_ - 2.) * (2. * shape_ * std::cyl_bessel_k(shape_, dist_ij) - dist_ij * std::cyl_bessel_k(shape_ + 1., dist_ij)) * dist_sq_ij_coord);
#else
			return(1.);
#endif
		}//end GradientRangeMaternARDGeneralShape

		inline double GradientRangeGaussianARD(double cm,
			const T_mat& sigma,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			double dist_sq_ij_coord = ((*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range));
			dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
			if (dist_sq_ij_coord < EPSILON_NUMBERS) {
				return(0.);
			}
			else {
				return(cm * dist_sq_ij_coord * sigma.coeff(i, j));
			}
		}//end GradientRangeGaussianARD

		inline void ParseCovFunctionAlias(string_t & likelihood,
			double& shape) const {
			if (likelihood == string_t("exponential_space_time")) {
				likelihood = "matern_space_time";
				shape = 0.5;
			}
			else if (likelihood == string_t("exponential_ard")) {
				likelihood = "matern_ard";
				shape = 0.5;
			}
			else if (likelihood == string_t("exponential")) {
				likelihood = "matern";
				shape = 0.5;
			}
		}

		template<typename>
		friend class RECompGP;
	};

}  // namespace GPBoost

#endif   // GPB_COV_FUNCTIONS_
