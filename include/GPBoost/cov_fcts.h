/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2025 Fabio Sigrist. All rights reserved.
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
#if __has_include(<version>)
#  include <version>   // collects all feature-test macros
#endif
#include <cmath>

#include <LightGBM/utils/log.h>
using LightGBM::Log;

#if defined(__cpp_lib_math_special_functions) && __cpp_lib_math_special_functions >= 201603L
#  define HAS_STD_CYL_BESSEL_K 1
#else
#  define HAS_STD_CYL_BESSEL_K 0
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

		/*! \brief Copy constructor */
		CovFunction(const CovFunction& other)
			: cov_fct_type_(other.cov_fct_type_),
			shape_(other.shape_),
			const_(other.const_),
			taper_range_(other.taper_range_),
			taper_shape_(other.taper_shape_),
			taper_mu_(other.taper_mu_),
			apply_tapering_(other.apply_tapering_),
			num_cov_par_(other.num_cov_par_),
			is_isotropic_(other.is_isotropic_),
			use_scaled_coordinates_(other.use_scaled_coordinates_),
			redetermine_vecchia_neighbors_in_transformed_space_(other.redetermine_vecchia_neighbors_in_transformed_space_),
			cov_calculated_based_on_coords_(other.cov_calculated_based_on_coords_),
			need_coordinates_for_calculating_covariance_(other.need_coordinates_for_calculating_covariance_),
			variance_on_the_diagonal_(other.variance_on_the_diagonal_),
			use_precomputed_dist_for_calc_cov_(other.use_precomputed_dist_for_calc_cov_)
		{
			// re-initialize the function pointers
			InitializeCovFct();
			InitializeCovFctGrad();
			InitializeGetDistanceForCovFct();
			InitializeGetDistanceForGradientCovFct();
		}

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
			cov_fct_type_ = cov_fct_type;
			use_precomputed_dist_for_calc_cov_ = use_precomputed_dist_for_calc_cov;
			if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" ||
				cov_fct_type_ == "matern_ard_estimate_shape" || cov_fct_type_ == "gaussian_ard" || 
				cov_fct_type_ == "space_time_gneiting" || cov_fct_type_ == "linear" || 
				cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
				is_isotropic_ = false;
				use_precomputed_dist_for_calc_cov_ = false;
				if (cov_fct_type_ == "space_time_gneiting" || cov_fct_type_ == "linear" || cov_fct_type_ == "hurst") {
					use_scaled_coordinates_ = false;
					redetermine_vecchia_neighbors_in_transformed_space_ = false;// by default, neighbors are not redetermined during hyperparameter optimization. But this can be turned on, e.g., by setting gp_approx = "vecchia_correlation"
				}
				else if (cov_fct_type_ == "hurst_ard") {
					use_scaled_coordinates_ = true;
					redetermine_vecchia_neighbors_in_transformed_space_ = false;
				}
				else if (cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard" || 
					cov_fct_type_ == "matern_ard_estimate_shape" || cov_fct_type_ == "matern_space_time"){
					use_scaled_coordinates_ = true;
					redetermine_vecchia_neighbors_in_transformed_space_ = true;
				}
				else {
					Log::REFatal("Missing option for cov_fct_type_ = '%s'", cov_fct_type_.c_str());
				}
			}
			else {
				is_isotropic_ = true;
				use_scaled_coordinates_ = false;
				redetermine_vecchia_neighbors_in_transformed_space_ = false;
			}
			if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_estimate_shape") {
				num_cov_par_ = 3;
			}
			else if (cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard" || cov_fct_type_ == "hurst_ard") {
				num_cov_par_ = dim_coordinates + 1;//"hurst_ard": (dim_coordinates - 1) * ranges + 1 * variance + 1 * H
			}
			else if (cov_fct_type_ == "matern_ard_estimate_shape") {
				num_cov_par_ = dim_coordinates + 2;
			}
			else if (cov_fct_type_ == "wendland" || cov_fct_type_ == "linear") {
				num_cov_par_ = 1;
			}
			else if (cov_fct_type_ == "space_time_gneiting") {
				num_cov_par_ = 7;
			}
			else {//includes "hurst"
				num_cov_par_ = 2;
			}
			if (cov_fct_type_ == "space_time_gneiting" || cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
				cov_calculated_based_on_coords_ = true;
			}
			else {
				cov_calculated_based_on_coords_ = false;
			}
			if (cov_calculated_based_on_coords_ || cov_fct_type_ == "linear" || cov_fct_type_ == "matern_space_time" || 
				cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard" || cov_fct_type_ == "matern_ard_estimate_shape") {
				need_coordinates_for_calculating_covariance_ = true;
			}
			else {
				need_coordinates_for_calculating_covariance_ = false;
			}
			if (cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard" || cov_fct_type_ == "linear") {
				variance_on_the_diagonal_ = false;
			}
			else {
				variance_on_the_diagonal_ = true;
			}
			shape_ = shape;
			if (cov_fct_type_ == "matern" || cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
				CHECK(shape > 0.);
				if (!(TwoNumbersAreEqual<double>(shape, 0.5) || TwoNumbersAreEqual<double>(shape, 1.5) || TwoNumbersAreEqual<double>(shape, 2.5))) {
#if HAS_STD_CYL_BESSEL_K
					const_ = std::pow(2., 1 - shape_) / std::tgamma(shape_);
#else
					// Mathematical special functions are not supported in C++17 by Clang and some other compilers (see https://en.cppreference.com/w/cpp/compiler_support/17#C.2B.2B17_library_features) 
					Log::REFatal("'shape' of %g is not supported for the '%s' covariance function (only 0.5, 1.5, and 2.5) when using this compiler (e.g., Clang on Mac). Use another compiler such as gcc or (a newer version of) MSVC instead. ", shape, cov_fct_type_.c_str());
#endif
				}
				if (shape > LARGE_SHAPE_WARNING_THRESHOLD_) {
					Log::REInfo(LARGE_SHAPE_WARNING_);
				}
			}
			else if (cov_fct_type_ == "powered_exponential") {
				if (shape <= 0. || shape > 2.) {
					Log::REFatal("'shape' needs to be larger than 0 and smaller or equal than 2 for the '%s' covariance function, found %g ", cov_fct_type_.c_str(), shape);
				}
			}
			else if (cov_fct_type_ == "matern_estimate_shape" || cov_fct_type_ == "matern_ard_estimate_shape") {
#if !defined(HAS_STD_CYL_BESSEL_K)
				// Mathematical special functions are not supported in C++17 by Clang and some other compilers (see https://en.cppreference.com/w/cpp/compiler_support/17#C.2B.2B17_library_features) 
				Log::REFatal("The covariance function '%s' is not supported when using this compiler (e.g. Clang on Mac). Use another compiler such as gcc or (a newer version of) MSVC instead. ", cov_fct_type_.c_str());
#endif
			}
			if (cov_fct_type_ == "wendland" || apply_tapering) {
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

		bool UseScaledCoordinates() const {
			return(use_scaled_coordinates_);
		}

		bool RedetermineVecchiaNeighborsInTransformedSpace() const {
			return(redetermine_vecchia_neighbors_in_transformed_space_);
		}

		bool IsSpaceTimeModel() const {
			return(cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "space_time_gneiting");
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
			else if (cov_fct_type_ == "matern_ard_estimate_shape") {
				for (int i = 0; i < (int)coords.cols(); ++i) {
					coords_scaled.col(i) = coords.col(i) / pars[i + 1];
				}
			}
			else if (cov_fct_type_ == "hurst_ard") {
				coords_scaled.col(0) = coords.col(0);
				for (int i = 0; i < (int)coords.cols() - 1; ++i) {
					coords_scaled.col(i + 1) = coords.col(i + 1) / pars[i + 2];
				}
			}
			else {
				Log::REFatal("'ScaleCoordinates' is called for a model for which this function is not implemented ");
			}
		}//end ScaleCoordinates
		// same for vectors
		void ScaleCoordinates_vec(const vec_t& pars,
			const vec_t& coords_vec,
			vec_t& coords_scaled_vec) const {
			coords_scaled_vec = vec_t(coords_vec.size());
			if (cov_fct_type_ == "matern_space_time") {
				coords_scaled_vec[0] = coords_vec[0] * pars[1];
				int dim_space = (int)coords_vec.size() - 1;
				coords_scaled_vec.tail(dim_space) = coords_vec.tail(dim_space) * pars[2];
			}
			else if (cov_fct_type_ == "matern_ard") {
				for (int i = 0; i < (int)coords_vec.size(); ++i) {
					coords_scaled_vec[i] = coords_vec[i] * pars[i + 1];
				}
			}
			else if (cov_fct_type_ == "gaussian_ard") {
				for (int i = 0; i < (int)coords_vec.size(); ++i) {
					coords_scaled_vec[i] = coords_vec[i] * std::sqrt(pars[i + 1]);
				}
			}
			else if (cov_fct_type_ == "matern_ard_estimate_shape") {
				for (int i = 0; i < (int)coords_vec.size(); ++i) {
					coords_scaled_vec[i] = coords_vec[i] / pars[i + 1];
				}
			}
			else if (cov_fct_type_ == "hurst_ard") {
				coords_scaled_vec[0] = coords_vec[0];
				for (int i = 0; i < (int)coords_vec.size() - 1; ++i) {
					coords_scaled_vec[i + 1] = coords_vec[i + 1] / pars[i + 2];
				}
			}
			else {
				Log::REFatal("'ScaleCoordinates' is called for a model for which this function is not implemented ");
			}
		}//end ScaleCoordinates_vec

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
			if (use_scaled_coordinates_) {// the coordinates are scaled before calculating distances (e.g. for '_ard' or '_space_time' covariance functions)
				ScaleCoordinates(pars, coords, coords_scaled);
				if (!is_symmmetric) {
					ScaleCoordinates(pars, coords_pred, coords_pred_scaled);
				}
			}
			// choose coords to be used (if applicable)
			if (is_symmmetric) {
				if (!use_scaled_coordinates_) {
					*coords_ptr = &coords;
					*coords_pred_ptr = &coords;
				}
				else {
					*coords_ptr = &coords_scaled;
					*coords_pred_ptr = &coords_scaled;
				}
			}//end is_symmmetric
			else {//!is_symmmetric
				if (!use_scaled_coordinates_) {
					*coords_ptr = &coords;
					*coords_pred_ptr = &coords_pred;
				}
				else {
					*coords_ptr = &coords_scaled;
					*coords_pred_ptr = &coords_pred_scaled;
				}
			}//end !is_symmmetric
		}//end DefineCoordsPtrScaleCoords

		/*!
		* \brief Check whether covariance parameter are on correct scales
		* \param pars Vector with covariance parameters 
		*/
		void CheckPars(const vec_t& pars) const {
			if (cov_fct_type_ == "space_time_gneiting") {
				for (int i = 0; i < num_cov_par_; ++i) {//sigma2, a, c, alpha, nu, beta, delta
					if (i == 3) {
						CHECK(pars[i] > 0. && pars[i] <= 1.);
					}
					else if (i == 5) {
						CHECK(pars[i] >= 0. && pars[i] <= 1.);
					}
					else if (i == 6) {
						CHECK(pars[i] >= 0.);
					}
					else {
						CHECK(pars[i] > 0.);
					}
				}
			}
			else {
				for (int i = 0; i < num_cov_par_; ++i) {
					CHECK(pars[i] > 0.);
				}
			}
		}//CheckPars

		/*!
		* \brief Make a warning of some parameters are e.g. too large
		* \param cov_pars Covariance parameters (on transformed scale)
		*/
		void CovarianceParameterRangeWarning(const vec_t& pars) const {
			if (cov_fct_type_ == "matern_estimate_shape" || cov_fct_type_ == "matern_ard_estimate_shape") {
				if (pars[num_cov_par_ - 1] > LARGE_SHAPE_WARNING_THRESHOLD_) {
					Log::REInfo(LARGE_SHAPE_WARNING_);
				}
			}
			else if (cov_fct_type_ == "space_time_gneiting") {
				if (pars[4] > LARGE_SHAPE_WARNING_THRESHOLD_) {
					Log::REInfo(LARGE_SHAPE_WARNING_);
				}
			}
		}//CovarianceParameterRangeWarning

		/*!
		* \brief Cap parameters
		* \param pars Vector with covariance parameters
		*/
		void CapPars(vec_t& pars) const {
			if (cov_fct_type_ == "space_time_gneiting") {
				if (pars[3] > 1.) {//alpha
					pars[3] = 1.;
				}
				if (pars[5] > 1.) {//beta
					pars[5] = 1.;
				}
			}
		}//CapPars

		/*!
		* \brief Transform the covariance parameters
		* \param sigma2 Nugget effect / error variance for Gaussian likelihoods
		* \param pars Vector with covariance parameters on original scale
		* \param[out] pars_trans Transformed covariance parameters
		*/
		void TransformCovPars(double sigma2,
			const vec_t& pars,
			vec_t& pars_trans) const {
			pars_trans = pars;
			pars_trans[0] = pars[0] / sigma2;
			if (cov_fct_type_ == "matern") {
				CHECK(pars[1] > 0.);
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
			else if (cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
				if (!(pars[1] > 0. && pars[1] < 1.)) {
					Log::REFatal("The Hurst exponent H must be in (0,1), found %g ", pars[1]);
				}
				pars_trans[1] = -std::log(pars[1]);
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
			else if (cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
				pars_orig[1] = std::exp(-pars[1]);
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
			// define coords pointers
			den_mat_t coords_scaled, coords_pred_scaled;
			const den_mat_t* coords_ptr = nullptr;
			const den_mat_t* coords_pred_ptr = nullptr;
			if (!use_precomputed_dist_for_calc_cov_) {
				DefineCoordsPtrScaleCoords(pars, coords, coords_pred, is_symmmetric, coords_scaled, coords_pred_scaled, &coords_ptr, &coords_pred_ptr);
			}
			// define sigma
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
			else if (cov_calculated_based_on_coords_) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						sigma(i, i) = variance_on_the_diagonal_ ? pars[0] : CovFct_coords_(i, i, coords_ptr, coords_pred_ptr, pars);
						for (int j = i + 1; j < num_cols; ++j) {
							sigma(i, j) = CovFct_coords_(i, j, coords_ptr, coords_pred_ptr, pars);
							sigma(j, i) = sigma(i, j);
						}// end loop cols
					}// end loop rows
				}// end is_symmmetric
				else {// !is_symmmetric
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						for (int j = 0; j < num_cols; ++j) {
							sigma(i, j) = CovFct_coords_(i, j, coords_ptr, coords_pred_ptr, pars);
						}// end loop cols
					}// end loop rows
				}// end !is_symmmetric
			}//end cov_calculated_based_on_coords_
			else if (cov_fct_type_ == "linear") {
				if (is_symmmetric) {
					sigma = pars[0] * coords * coords.transpose();
				}
				else {
					sigma = pars[0] * coords_pred * coords.transpose();
				}
			}//end linear
			else {// cov_fct_type_ != "wendland"
				const double shape = (cov_fct_type_ == "matern_estimate_shape" || cov_fct_type_ == "matern_ard_estimate_shape") ? pars[num_cov_par_ - 1] : 0.;
				const double range = is_isotropic_ ? pars[1] : 1.;
				// calculate covariances
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						sigma(i, i) = variance_on_the_diagonal_ ? pars[0] : GetDistanceForCovFct_(i, i, dist, coords_ptr, coords_pred_ptr);
						for (int j = i + 1; j < num_cols; ++j) {
							double dist_ij = GetDistanceForCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							sigma(i, j) = CovFct_dist_(dist_ij, pars[0], range, shape);
							sigma(j, i) = sigma(i, j);
						}// end loop cols
					}// end loop rows
				}// end is_symmmetric
				else {// !is_symmmetric
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						for (int j = 0; j < num_cols; ++j) {
							double dist_ij = GetDistanceForCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							sigma(i, j) = CovFct_dist_(dist_ij, pars[0], range, shape);
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
			// define coords pointers
			den_mat_t coords_scaled, coords_pred_scaled;
			const den_mat_t* coords_ptr = nullptr;
			const den_mat_t* coords_pred_ptr = nullptr;
			if (!use_precomputed_dist_for_calc_cov_) {
				DefineCoordsPtrScaleCoords(pars, coords, coords_pred, is_symmmetric, coords_scaled, coords_pred_scaled, &coords_ptr, &coords_pred_ptr);
			}
			// define sigma
			sigma = dist;
			sigma.makeCompressed();
			if (cov_fct_type_ == "wendland") {
				sigma.coeffs() = pars[0];
				MultiplyWendlandCorrelationTaper<T_mat>(dist, sigma, is_symmmetric);
			}
			else if (cov_calculated_based_on_coords_) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = variance_on_the_diagonal_ ? pars[0] : CovFct_coords_(i, i, coords_ptr, coords_pred_ptr, pars);
							}
							else if (i < j) {
								it.valueRef() = CovFct_coords_(i, j, coords_ptr, coords_pred_ptr, pars);
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
							it.valueRef() = CovFct_coords_(i, j, coords_ptr, coords_pred_ptr, pars);
						}
					}
				}// end !is_symmmetric
			}//end cov_calculated_based_on_coords_
			else if (cov_fct_type_ == "linear") {
				if (is_symmmetric) {
					sigma = (pars[0] * (coords * coords.transpose())).sparseView();
				}
				else {
					sigma = (pars[0] * (coords_pred * coords.transpose())).sparseView();
				}
			}//end linear
			else {// cov_fct_type_ != "wendland"
				const double shape = (cov_fct_type_ == "matern_estimate_shape" || cov_fct_type_ == "matern_ard_estimate_shape") ? pars[num_cov_par_ - 1] : 0.;
				const double range = is_isotropic_ ? pars[1] : 1.;
				// calculate covariances
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = variance_on_the_diagonal_ ? pars[0] : GetDistanceForCovFct_(i, i, dist, coords_ptr, coords_pred_ptr);
							}
							else if (i < j) {
								double dist_ij = GetDistanceForCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
								it.valueRef() = CovFct_dist_(dist_ij, pars[0], range, shape);
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
							it.valueRef() = CovFct_dist_(dist_ij, pars[0], range, shape);
						}
					}
				}// end !is_symmmetric
			}// end cov_fct_type_ != "wendland"
		}//end CalculateCovMat (sparse)

		/*!
		* \brief Covariance function for one entry
		* \param dist Distance
		* \param coords Coordinates
		* \param coords_pred Coordinates for second point
		* \param pars Vector with covariance parameters
		* \return sigma Covariance at dist
		*/
		double CalculateCovarianceOneEntry(double dist,
			const vec_t& coords,
			const vec_t& coords_pred,
			const vec_t& pars) const {
			CHECK(pars.size() == num_cov_par_);
			if (need_coordinates_for_calculating_covariance_) {
				if (coords.size() == 0 || coords_pred.size() == 0) {
					Log::REFatal("'CalculateCovarianceOneEntry()' is not implemented when 'coords' or 'coords_pred' are empty for cov_fct_type_ == '%s'", cov_fct_type_.c_str());
				}				
			}
			bool coords_provided = false;
			if (coords.size() > 0) {
				CHECK(coords.size() == coords_pred.size());
				coords_provided = true;
			}			
			vec_t coords_scaled, coords_pred_scaled;
			const vec_t* coords_ptr = nullptr;
			const vec_t* coords_pred_ptr = nullptr;
			if (coords_provided) {
				if (use_scaled_coordinates_) {// the coordinates are scaled before calculating distances (e.g. for '_ard' or '_space_time' covariance functions)
					ScaleCoordinates_vec(pars, coords, coords_scaled);
					ScaleCoordinates_vec(pars, coords_pred, coords_pred_scaled);
					coords_ptr = &coords_scaled;
					coords_pred_ptr = &coords_pred_scaled;
				}
				else {
					coords_ptr = &coords;
					coords_pred_ptr = &coords_pred;
				}
			}
			double sigma;
			if (cov_fct_type_ == "wendland") {// note: this option is usually not used
				if (coords_provided) {
					Log::REFatal("This options is not implemented ");
				}
				if (dist >= taper_range_) {
					sigma = 0.;
				}
				else {
					sigma = pars[0];
					MultiplyWendlandCorrelationTaper(dist, sigma);
				}
			}//end wendland
			else if (cov_calculated_based_on_coords_) {
				sigma = CovFct_coords_vec_(*coords_ptr, *coords_pred_ptr, pars);
			}
			else if (cov_fct_type_ == "linear") {
				sigma = pars[0] * (*coords_ptr).cwiseProduct((*coords_pred_ptr)).sum();
			}
			else {
				const double shape = (cov_fct_type_ == "matern_estimate_shape" || cov_fct_type_ == "matern_ard_estimate_shape") ? pars[num_cov_par_ - 1] : 0.;
				const double range = is_isotropic_ ? pars[1] : 1.;
				if (coords_provided) {
					dist = ((*coords_pred_ptr) - (*coords_ptr)).lpNorm<2>();
				}
				sigma = CovFct_dist_(dist, pars[0], range, shape);
			}
			return sigma;
		}//end CalculateCovarianceOneEntry

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
		* \param ind_par Parameter number for which the gradient is calculated.
		*			Note: sigma2 is not included as the gradient is trivial and computed elsewhere.
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
			int ind_par,
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
			// define coords pointers
			den_mat_t coords_scaled, coords_pred_scaled;
			const den_mat_t* coords_ptr = nullptr;
			const den_mat_t* coords_pred_ptr = nullptr;
			if (!use_precomputed_dist_for_calc_cov_) {
				DefineCoordsPtrScaleCoords(pars, coords, coords_pred, is_symmmetric, coords_scaled, coords_pred_scaled, &coords_ptr, &coords_pred_ptr);
			}
			//define sigma_grad
			sigma_grad = T_mat(sigma.rows(), sigma.cols());
			if (cov_fct_type_ == "linear") {
				Log::REFatal("'CalculateGradientCovMat()' is not implemented for cov_fct_type_ == '%s' ", cov_fct_type_.c_str());
			} 
			else if (cov_calculated_based_on_coords_) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						sigma_grad(i, i) = variance_on_the_diagonal_ ? 0. : GradientCovFct_coords_(i, i, coords_ptr, coords_pred_ptr, pars, ind_par, transf_scale, nugget_var);
						for (int j = i + 1; j < num_cols; ++j) {
							sigma_grad(i, j) = GradientCovFct_coords_(i, j, coords_ptr, coords_pred_ptr, pars, ind_par, transf_scale, nugget_var);
							sigma_grad(j, i) = sigma_grad(i, j);
						}// end loop over cols
					}// end loop over rows
				}// end is_symmmetric
				else {// !is_symmmetric
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						for (int j = 0; j < num_cols; ++j) {
							sigma_grad(i, j) = GradientCovFct_coords_(i, j, coords_ptr, coords_pred_ptr, pars, ind_par, transf_scale, nugget_var);
						}// end loop over cols
					}// end loop over rows
				}// end !is_symmmetric
			}//end cov_calculated_based_on_coords_
			else {// not cov_calculated_based_on_coords_
				double cm, cm_num_deriv, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape;//constants and auxiliary parameters
				DetermineConstantsForGradient(pars, (int)coords.cols(), transf_scale, nugget_var, ind_par,
					cm, cm_num_deriv, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape);
				// calculate gradients
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						sigma_grad(i, i) = variance_on_the_diagonal_ ? 0. : GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
							ind_par, i, i, GetDistanceForGradientCovFct_(i, i, dist, coords_ptr, coords_pred_ptr), sigma, coords_ptr, coords_pred_ptr);
						sigma_grad(i, i) = 0.;
						for (int j = i + 1; j < num_cols; ++j) {
							const double dist_ij = GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							sigma_grad(i, j) = GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
								ind_par, i, j, dist_ij, sigma, coords_ptr, coords_pred_ptr);
							sigma_grad(j, i) = sigma_grad(i, j);
						}// end loop over cols
					}// end loop over rows
				}// end is_symmmetric
				else {// !is_symmmetric
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						for (int j = 0; j < num_cols; ++j) {
							const double dist_ij = GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							sigma_grad(i, j) = GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
								ind_par, i, j, dist_ij, sigma, coords_ptr, coords_pred_ptr);
						}// end loop over cols
					}// end loop over rows
				}// end !is_symmmetric
			}// end not cov_calculated_based_on_coords_
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
			int ind_par,
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
			// define coords pointers
			den_mat_t coords_scaled, coords_pred_scaled;
			const den_mat_t* coords_ptr = nullptr;
			const den_mat_t* coords_pred_ptr = nullptr;
			if (!use_precomputed_dist_for_calc_cov_) {
				DefineCoordsPtrScaleCoords(pars, coords, coords_pred, is_symmmetric, coords_scaled, coords_pred_scaled, &coords_ptr, &coords_pred_ptr);
			}
			//define sigma_grad
			sigma_grad = sigma;
			double cm, cm_num_deriv, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape;//constants and auxiliary parameters
			DetermineConstantsForGradient(pars, (int)coords.cols(), transf_scale, nugget_var, ind_par,
				cm, cm_num_deriv, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape);			
			// calculate gradients
			if (cov_fct_type_ == "linear") {
				Log::REFatal("'CalculateGradientCovMat()' is not implemented for cov_fct_type_ == '%s' ", cov_fct_type_.c_str());
			}
			else if (cov_calculated_based_on_coords_) {
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma_grad.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = variance_on_the_diagonal_ ? 0. : GradientCovFct_coords_(i, i, coords_ptr, coords_pred_ptr, pars, ind_par, transf_scale, nugget_var);
							}
							else if (i < j) {
								it.valueRef() = GradientCovFct_coords_(i, j, coords_ptr, coords_pred_ptr, pars, ind_par, transf_scale, nugget_var);
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
							it.valueRef() = GradientCovFct_coords_(i, j, coords_ptr, coords_pred_ptr, pars, ind_par, transf_scale, nugget_var);
						}// end loop over cols
					}// end loop over rows
				}// end !is_symmmetric
			}//end cov_calculated_based_on_coords_
			else {// not cov_calculated_based_on_coords_
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma_grad.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = variance_on_the_diagonal_ ? 0. : GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
									ind_par, i, i, GetDistanceForGradientCovFct_(i, i, dist, coords_ptr, coords_pred_ptr), sigma, coords_ptr, coords_pred_ptr);
							}
							else if (i < j) {
								const double dist_ij = GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
								it.valueRef() = GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
									ind_par, i, j, dist_ij, sigma, coords_ptr, coords_pred_ptr);
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
							const double dist_ij = GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							it.valueRef() = GradientCovFct_(cm, cm_num_deriv, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
								ind_par, i, j, dist_ij, sigma, coords_ptr, coords_pred_ptr);
						}// end loop over cols
					}// end loop over rows
				}// end !is_symmmetric
			}// end not cov_calculated_based_on_coords_
		}//end CalculateGradientCovMat (sparse)

		/*!
		* \brief Gradient of covariance function for one entry
		* \param dist Distance
		* \param coords Coordinates
		* \param coords_pred Coordinates for second point
		* \param pars Vector with covariance parameters
		* \return sigma_grad Gradient
		*/
		double CalculateGradientCovarianceOneEntry(double /* dist */,
			const vec_t& coords,
			const vec_t& coords_pred,
			const vec_t& pars,
			bool transf_scale,
			double nugget_var,
			int ind_par) const {
			CHECK(pars.size() == num_cov_par_);
			if (need_coordinates_for_calculating_covariance_) {
				if (coords.size() == 0 || coords_pred.size() == 0) {
					Log::REFatal("'CalculateGradientCovarianceOneEntry()' is not implemented when 'coords' or 'coords_pred' are empty for cov_fct_type_ == '%s'", cov_fct_type_.c_str());
				}
			}
			bool coords_provided = false;
			if (coords.size() > 0) {
				CHECK(coords.size() == coords_pred.size());
				coords_provided = true;
			}
			vec_t coords_scaled, coords_pred_scaled;
			const vec_t* coords_ptr = nullptr;
			const vec_t* coords_pred_ptr = nullptr;
			if (coords_provided) {
				if (use_scaled_coordinates_) {// the coordinates are scaled before calculating distances (e.g. for '_ard' or '_space_time' covariance functions)
					ScaleCoordinates_vec(pars, coords, coords_scaled);
					ScaleCoordinates_vec(pars, coords_pred, coords_pred_scaled);
					coords_ptr = &coords_scaled;
					coords_pred_ptr = &coords_pred_scaled;
				}
				else {
					coords_ptr = &coords;
					coords_pred_ptr = &coords_pred;
				}
			}
			double sigma_grad = 0.;
			if (cov_fct_type_ == "wendland") {// note: this option is usually not used
				Log::REFatal("'CalculateGradientCovarianceOneEntry' is not implemented for 'wendland' covariance ");
			}//end wendland
			else if (cov_calculated_based_on_coords_) {
				sigma_grad = GradientCovFct_coords_vec_(*coords_ptr, *coords_pred_ptr, pars, ind_par, transf_scale, nugget_var);
			}
			else if (cov_fct_type_ == "linear") {
				Log::REFatal("'CalculateGradientCovarianceOneEntry' is not implemented for 'linear' kernel (there are not other parameters) ");
			}
			else {
				Log::REFatal("'CalculateGradientCovarianceOneEntry' is not implemented for '%s' covariance (cov_calculated_based_on_coords_ == false) ", cov_fct_type_.c_str());
			}
			return sigma_grad;
		}//end CalculateGradientCovarianceOneEntry

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
			if (num_cov_par_ > 1) {
				// Range parameters
				int MAX_POINTS_INIT_RANGE = 1000;//limit number of samples considered to save computational time
				int num_data;
				if (use_distances) {
					num_data = (int)dist.rows();
				}
				else {
					num_data = (int)coords.rows();
				}
				// Calculate average distance
				int num_data_find_init = (num_data > MAX_POINTS_INIT_RANGE) ? MAX_POINTS_INIT_RANGE : num_data;
				std::vector<int> sample_ind;
				bool use_subsamples = num_data_find_init < num_data;
				if (use_subsamples) {
					std::uniform_int_distribution<> dis(0, num_data - 1);
					sample_ind = std::vector<int>(num_data_find_init);
					for (int i = 0; i < num_data_find_init; ++i) {
						sample_ind[i] = dis(rng);
					}
				}
				double med_dist = 0., med_dist_space = 0., med_dist_time = 0.;
				std::vector<double> med_dist_per_coord;//for ard kernels
				string_t add_error_str = use_subsamples ? "on a random sub-sample of size 1000 " : "";

				int num_distances = (int)(num_data_find_init * (num_data_find_init - 1) / 2.);
				std::vector<double> distances(num_distances);
				if (cov_fct_type_ == "exponential" || cov_fct_type_ == "gaussian" || cov_fct_type_ == "powered_exponential" ||
					cov_fct_type_ == "matern" ||  cov_fct_type_ == "matern_estimate_shape") {
					if (use_distances) {
						if (use_subsamples) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (num_data_find_init - 1); ++i) {
								for (int j = i + 1; j < num_data_find_init; ++j) {
									distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = dist.coeff(sample_ind[i], sample_ind[j]);
								}
							}
						}//end use_subsamples
						else { // not use_subsamples
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (num_data - 1); ++i) {
								for (int j = i + 1; j < num_data; ++j) {
									distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = dist.coeff(i, j);
								}
							}
						}//end not use_subsamples
					}
					else {
						// Calculate distances (of a subsample) in case they have not been calculated (e.g., for the Vecchia approximation)
						den_mat_t dist_from_coord;
						if (use_subsamples) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (num_data_find_init - 1); ++i) {
								for (int j = i + 1; j < num_data_find_init; ++j) {
									distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = (coords.row(sample_ind[i]) - coords.row(sample_ind[j])).lpNorm<2>();
								}
							}
						}//end use_subsamples
						else { // not use_subsamples
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (num_data_find_init - 1); ++i) {
								for (int j = i + 1; j < num_data_find_init; ++j) {
									distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = (coords.row(i) - coords.row(j)).lpNorm<2>();
								}
							}
						}//end not use_subsamples
					}
					med_dist = GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(distances);
					if (med_dist < EPSILON_NUMBERS) {
						med_dist = GPBoost::CalculateMean<std::vector<double>>(distances);
					}
					if (med_dist < EPSILON_NUMBERS) {
						Log::REFatal(("Cannot find an initial value for the range parameter "
							"since both the median and the average distances among coordinates are zero " + add_error_str).c_str());
					}
				}//end cov_fct_type_ == "exponential" || cov_fct_type_ == "gaussian" || cov_fct_type_ == "powered_exponential" || cov_fct_type_ == "matern" || cov_fct_type_ == "matern_estimate_shape"
				else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "space_time_gneiting") {
					std::vector<double> distances_time(num_distances);
					if (use_subsamples) {
						den_mat_t coords_space = coords(sample_ind, Eigen::seq(1, Eigen::last));
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = (coords_space.row(i) - coords_space.row(j)).lpNorm<2>();
								distances_time[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = std::abs(coords.coeff(sample_ind[i], 0) - coords.coeff(sample_ind[j], 0));;
							}
						}
					}//end use_subsamples
					else { // not use_subsamples
						den_mat_t coords_space = coords(Eigen::all, Eigen::seq(1, Eigen::last));
						for (int i = 0; i < (num_data_find_init - 1); ++i) {
							for (int j = i + 1; j < num_data_find_init; ++j) {
								distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = (coords_space.row(i) - coords_space.row(j)).lpNorm<2>();
								distances_time[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = std::abs(coords.coeff(i, 0) - coords.coeff(j, 0));;
							}
						}
					}//end not use_subsamples
					med_dist_space = GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(distances);
					if (med_dist_space < EPSILON_NUMBERS) {
						med_dist_space = GPBoost::CalculateMean<std::vector<double>>(distances);
					}
					med_dist_time = GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(distances_time);
					if (med_dist_time < EPSILON_NUMBERS) {
						med_dist_time = GPBoost::CalculateMean<std::vector<double>>(distances_time);
					}
					if (med_dist_space < EPSILON_NUMBERS) {
						Log::REFatal(("Cannot find an initial value for the spatial range parameter "
							"since both the median and the average distances among spatial coordinates are zero " + add_error_str).c_str());
					}
					if (med_dist_time < EPSILON_NUMBERS) {
						Log::REFatal(("Cannot find an initial value for the temporal range parameter "
							"since both the median and the average distances among time points are zero " + add_error_str).c_str());
					}
				}//end cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "space_time_gneiting"
				else if (cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard" || cov_fct_type_ == "matern_ard_estimate_shape") {
					med_dist_per_coord = std::vector<double>((int)coords.cols());
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						vec_t col_i = coords.col(ic);
						int num_unique_values = GPBoost::NumberUniqueValues(col_i, 11);
						bool feature_is_constant = false;
						if (num_unique_values == 1) {
							add_error_str = "";
							feature_is_constant = true;
						}
						else if (num_unique_values <= 10) {
							med_dist_per_coord[ic] = (num_unique_values * num_unique_values - 1) / 3. / num_unique_values; // use average distance among two random points on {1,...,num_unique_values}
						}
						else {// num_unique_values > 10
							double med_dist_coord_i = 0.;
							if (use_subsamples) {
								for (int i = 0; i < (num_data_find_init - 1); ++i) {
									for (int j = i + 1; j < num_data_find_init; ++j) {
										distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = std::abs(coords.coeff(sample_ind[i], ic) - coords.coeff(sample_ind[j], ic));;
									}
								}
							}//end use_subsamples
							else { // not use_subsamples
								for (int i = 0; i < (num_data_find_init - 1); ++i) {
									for (int j = i + 1; j < num_data_find_init; ++j) {
										distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] = std::abs(coords.coeff(i, ic) - coords.coeff(j, ic));;
									}
								}
							}//end not use_subsamples
							med_dist_coord_i = GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(distances);
							if (med_dist_coord_i < EPSILON_NUMBERS) {
								med_dist_coord_i = GPBoost::CalculateMean<std::vector<double>>(distances);
							}
							med_dist_per_coord[ic] = med_dist_coord_i;
							if (med_dist_coord_i < EPSILON_NUMBERS) {
								feature_is_constant = true;
							}
						}// end num_unique_values > 10
						if (feature_is_constant) {
							Log::REFatal(("Cannot find an initial value for the range parameter for the input feature number " + std::to_string(ic + 1) +
								" (counting starts at 1) since this feature is constant " + add_error_str).c_str());
						}
					}// end loop over features
				}//end cov_fct_type_ == "matern_ard" && cov_fct_type_ == "gaussian_ard" || cov_fct_type_ == "matern_ard_estimate_shape"
				if (cov_fct_type_ == "matern") {
					if (shape_ <= 1.) {
						pars[1] = 2. * 3. / med_dist;//includes shape_ = 0.5
					}
					else if (shape_ <= 2.) {
						pars[1] = 2. * 4.7 / med_dist;//includes shape_ = 1.5
					}
					else {
						pars[1] = 2. * 5.9 / med_dist;//includes shape_ = 2.5
					}
				}
				else if (cov_fct_type_ == "matern_estimate_shape") {
					pars[1] = med_dist * std::sqrt(3.) / 2. / 4.7;//same as shape_ = 1.5
					pars[2] = 1.5;
				}
				else if (cov_fct_type_ == "matern_ard_estimate_shape") {
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						pars[1 + ic] = med_dist_per_coord[ic] * std::sqrt(3.) / 2. / 4.7;//same shape_ = 1.5
					}
					pars[num_cov_par_ - 1] = 1.5;
				}
				else if (cov_fct_type_ == "gaussian") {
					pars[1] = 3. / std::pow(med_dist / 2., 2.);
				}
				else if (cov_fct_type_ == "powered_exponential") {
					pars[1] = 3. / std::pow(med_dist / 2., shape_);
				}
				else if (cov_fct_type_ == "matern_space_time") {
					if (shape_ <= 1.) {
						pars[1] = 2. * 3. / med_dist_time;//includes shape_ = 0.5
						pars[2] = 2. * 3. / med_dist_space;
					}
					else if (shape_ <= 2.) {
						pars[1] = 2. * 4.7 / med_dist_time;//includes shape_ = 1.5
						pars[2] = 2. * 4.7 / med_dist_space;
					}
					else {
						pars[1] = 2. * 5.9 / med_dist_time;//includes shape_ = 2.5
						pars[2] = 2. * 5.9 / med_dist_space;
					}
				}//end matern_space_time
				else if (cov_fct_type_ == "space_time_gneiting") {//pars Parameter in the following order : sigma2, a, c, alpha, nu, beta, delta
					int dim_space = (int)coords.cols() - 1;
					pars[1] = (std::pow(20., 2. / dim_space) - 1.) / (med_dist_time * med_dist_time) * 4.;//a, temporal range such that correlation at 0.05 at half the median distance
					pars[2] = 2. * 4.7 / med_dist_space;//c, spatial range such that correlation at 0.05 at half the median distance
					pars[3] = 1.;//alpha
					pars[4] = 1.5;//nu -> matern 1.5
					pars[5] = 1.;//beta
					pars[6] = 1;//delta
				}
				else if (cov_fct_type_ == "matern_ard") {
					if (shape_ <= 1.) {
						for (int ic = 0; ic < (int)coords.cols(); ++ic) {
							pars[1 + ic] = 2. * 3. / med_dist_per_coord[ic];//includes shape_ = 0.5
						}
					}
					else if (shape_ <= 2.) {
						for (int ic = 0; ic < (int)coords.cols(); ++ic) {
							pars[1 + ic] = 2. * 4.7 / med_dist_per_coord[ic];//includes shape_ = 1.5
						}
					}
					else {
						for (int ic = 0; ic < (int)coords.cols(); ++ic) {
							pars[1 + ic] = 2. * 5.9 / med_dist_per_coord[ic];//includes shape_ = 2.5
						}
					}
				}//end matern_ard
				else if (cov_fct_type_ == "gaussian_ard") {
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						pars[1 + ic] = 3. / std::pow(med_dist_per_coord[ic] / 2., 2.);
					}
				}
				else if (cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
					pars[1] = -std::log(0.5);// Brownian motion
					if (cov_fct_type_ == "hurst_ard") {
						for (int ic = 1; ic < (int)coords.cols(); ++ic) {
							pars[1 + ic] = 1.;
						}
					}
				}
				else {
					Log::REFatal("Finding initial values for covariance parameters for covariance of type '%s' is not supported ", cov_fct_type_.c_str());
				}
			}//end num_cov_par_ > 1
		}//end FindInitCovPar

	private:

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
		* \brief Generic function for calculating covariances based on (scaled) distances (cov_calculated_based_on_coords_ == false)
		* \param dist Distance
		* \param var Marginal variance
		* \param range Transformed range parameter
		* \param shape Smoothness parameter (if applicable)
		* \return Covariance
		*/
		std::function<double(double /* dist_ij */, double /* var */, double /* range */, double /* shape */)> CovFct_dist_;

		/*!
		* \brief Generic function for calculating covariances based on (scaled) coordinates (cov_calculated_based_on_coords_ == true)
		* \param i Location for coords_ptr
		* \param j Location for coords_pred_ptr
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Covariance parameters
		* \return Covariance
		*/
		std::function<double(const int i, const int j, const den_mat_t* coords, const den_mat_t* coords_pred, const vec_t& pars)> CovFct_coords_;

		/*!
		* \brief Generic function for calculating covariances between two points based on two vectors with (scaled) coordinates
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Covariance parameters
		* \return Covariance
		*/
		std::function<double(const vec_t& coords, const vec_t& coords_pred, const vec_t& pars)> CovFct_coords_vec_;

		void InitializeCovFct() {
			if (cov_fct_type_ == "matern" || cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape0_5(dist_ij, var, range);
						};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape1_5(dist_ij, var, range);
						};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape2_5(dist_ij, var, range);
						};
				}
				else {//general shape
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternGeneralShape(dist_ij, var, range);
						};
				}
			}//end cov_fct_type_ == "matern" || cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard"
			else if (cov_fct_type_ == "matern_estimate_shape" || cov_fct_type_ == "matern_ard_estimate_shape") {
				CovFct_dist_ = [this](double dist_ij, double var, double range, double shape) -> double {
					return CovarianceMaternEstimateShape(dist_ij, var, range, shape);
					};
			}
			else if (cov_fct_type_ == "gaussian" || cov_fct_type_ == "gaussian_ard") {
				CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
					return CovarianceGaussian(dist_ij, var, range);
					};
			}
			else if (cov_fct_type_ == "powered_exponential") {
				CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
					return CovariancePoweredExponential(dist_ij, var, range);
					};
			}
			else if (cov_fct_type_ == "space_time_gneiting") {
				CovFct_coords_ = [this](const int i, const int j, const den_mat_t* coords, const den_mat_t* coords_pred, const vec_t& pars) -> double {
					return SpaceTimeGneitingCovariance(i, j, coords, coords_pred, pars);
				};
				CovFct_coords_vec_ = [this](const vec_t& coords, const vec_t& coords_pred, const vec_t& pars) -> double {
					return SpaceTimeGneitingCovariance_vec(coords, coords_pred, pars);
				};
			}
			else if (cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
				CovFct_coords_ = [this](const int i, const int j, const den_mat_t* coords, const den_mat_t* coords_pred, const vec_t& pars) -> double {
					return HurstCovariance(i, j, coords, coords_pred, pars);
				};
				CovFct_coords_vec_ = [this](const vec_t& coords, const vec_t& coords_pred, const vec_t& pars) -> double {
					return HurstCovariance_vec(coords, coords_pred, pars);
				};
			}
			else if (cov_fct_type_ != "wendland" && cov_fct_type_ != "linear") {
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
#if HAS_STD_CYL_BESSEL_K
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
#if HAS_STD_CYL_BESSEL_K
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
			int ind_par,
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
				CHECK(ind_par >= 0 && ind_par <= 1);
				if (ind_par == 0) {
					cm = transf_scale ? 1. : nugget_var / pars[1];
					cm *= -pars[0] * std::pow(2., 1 - pars[2]) / std::tgamma(pars[2]);
					par_aux = std::sqrt(2. * pars[2]) / pars[1];
				}
				else if (ind_par == 1) {//gradient wrt smoothness parameter
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
			else if (cov_fct_type_ == "matern_ard_estimate_shape") {
				int ind_shape = num_cov_par_ - 1;
				CHECK(ind_par >= 0 && ind_par <= num_cov_par_ - 2);
				par_aux = std::sqrt(2. * pars[ind_shape]);
				if (ind_par < num_cov_par_ - 2) {//gradient wrt anisotropic ranges 
					cm = transf_scale ? 1. : nugget_var / pars[ind_par + 1];
					cm *= -pars[0] * std::pow(2., 1 - pars[ind_shape]) / std::tgamma(pars[ind_shape]) * 2. * pars[ind_shape];
				}
				else if (ind_par == num_cov_par_ - 2) {//gradient wrt smoothness parameter
					//for calculating finite differences
					cm = transf_scale ? pars[ind_shape] : nugget_var;
					cm *= pars[0] * std::pow(2., 1 - pars[ind_shape]) / std::tgamma(pars[ind_shape]);
					if (transf_scale) {
						cm_num_deriv = pars[0] * std::pow(2., 1 - pars[ind_shape]) / std::tgamma(pars[ind_shape]);
						pars_2_up = std::exp(std::log(pars[ind_shape]) + delta_step_);//gradient on log-scale
						pars_2_down = std::exp(std::log(pars[ind_shape]) - delta_step_);
					}
					else {
						cm_num_deriv = cm;
						pars_2_up = pars[ind_shape] + delta_step_;
						pars_2_down = pars[ind_shape] - delta_step_;
						CHECK(pars_2_down > 0.);
					}
					par_aux_up = std::sqrt(2. * pars_2_up);
					par_aux_down = std::sqrt(2. * pars_2_down);
				}
				shape = pars[ind_shape];
			}// end cov_fct_type_ == "matern_ard_estimate_shape"
			else if (cov_fct_type_ == "matern_space_time") {
				CHECK(ind_par >= 0 && ind_par <= 1);
				// calculate constants that are the same for all entries
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					cm = transf_scale ? -1. : (nugget_var * pars[ind_par + 1]);
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[ind_par + 1] / sqrt(3.));
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[ind_par + 1] / sqrt(5.));
				}
				else {//general shape
					cm = transf_scale ? 1. : (-nugget_var * pars[ind_par + 1] / sqrt(2. * shape_));
					cm *= pars[0] * const_;
				}
			}// end cov_fct_type_ == "matern_space_time"
			else if (cov_fct_type_ == "matern_ard") {
				CHECK(ind_par >= 0 && ind_par < dim_coords);
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					cm = transf_scale ? -1. : (nugget_var * pars[ind_par + 1]);
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[ind_par + 1] / sqrt(3.));
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[ind_par + 1] / sqrt(5.));
				}
				else {//general shape
					cm = transf_scale ? 1. : (-nugget_var * pars[ind_par + 1] / sqrt(2. * shape_));
					cm *= pars[0] * const_;
				}
			}// end (cov_fct_type_ == "matern_ard"
			else if (cov_fct_type_ == "gaussian_ard") {
				CHECK(ind_par >= 0 && ind_par < dim_coords);
				cm = transf_scale ? -1. : (2. * nugget_var * std::sqrt(pars[1]));
			}
		}//end DetermineConstantsForGradient

		/*!
		* \brief Generic function for determining distances for calculating gradients of covariances
		*/
		std::function<double(const int /* i */, const int /* j */, const T_mat& /* dist */,
			const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */)> GetDistanceForGradientCovFct_;

		void InitializeGetDistanceForGradientCovFct() {
			if (use_precomputed_dist_for_calc_cov_) {
				GetDistanceForGradientCovFct_ = [this](const int i, const int j, const T_mat& dist,
					const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) {
						return static_cast<double>(dist.coeff(i, j));
					};
			}
			else {
				GetDistanceForGradientCovFct_ = [this](const int i, const int j, const T_mat& /* dist */,
					const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) {
						return static_cast<double>(((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>());
					};
			}
		}//end InitializeGetDistanceForGradientCovFct

		/*!
		* \brief Generic function for calculating gradients of covariances wrt range and other parameters such as smoothness based on distances (cov_calculated_based_on_coords_ == false)
		*/
		std::function<double(double /* cm */, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
			double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
			const int /* ind_par */, const int /* i */, const int /* j */,
			const double /* dist_ij */, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */)> GradientCovFct_;

		/*!
		* \brief Generic function for calculating gradients of covariances based on (scaled) coordinates (cov_calculated_based_on_coords_ == true)
		* \param i Row index for coords
		* \param j Row index for coords_pred
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Parameter
		* \param ind_par Parameter number for which the gradient is calculated.
		*			Note: sigma2 is not included as the gradient is trivial and computed elsewhere.
		*			ind_par thus starts at 0 for a, but pars[ind_par + 1] gives the corresponding parameter in pars
		* \param transf_scale On transformed  scale or not
		* \param nugget_var Nugget variance
		* \return Gradient of covariance
		*/
		std::function<double(const int i, const int j, const den_mat_t* coords, const den_mat_t* coords_pred,
			const vec_t& pars, const int ind_par, bool transf_scale, double nugget_var)> GradientCovFct_coords_;

		/*!
		* \brief Generic function for calculating gradients of covariances  based on two vectors with (potentially scaled) coordinates
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Parameter
		* \param ind_par Parameter number for which the gradient is calculated.
		*			Note: sigma2 is not included as the gradient is trivial and computed elsewhere.
		*			ind_par thus starts at 0 for a, but pars[ind_par + 1] gives the corresponding parameter in pars
		* \param transf_scale On transformed  scale or not
		* \param nugget_var Nugget variance
		* \return Gradient of covariance
		*/
		std::function<double(const vec_t& coords, const vec_t& coords_pred, const vec_t& pars, 
			const int ind_par, bool transf_scale, double nugget_var)> GradientCovFct_coords_vec_;

		void InitializeCovFctGrad() {
			if (cov_fct_type_ == "matern") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int /* ind_par */, const int i, const int j,
						const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
							return GradientRangeMaternShape0_5(cm, dist_ij, sigma, i, j);
						};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double par_aux, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int /* ind_par */, const int /* i */, const int /* j */,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
							return GradientRangeMaternShape1_5(cm, dist_ij, par_aux);
						};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double par_aux, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int /* ind_par */, const int /* i */, const int /* j */,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
							return GradientRangeMaternShape2_5(cm, dist_ij, par_aux);
						};
				}
				else {// general shape
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double par_aux, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int /* ind_par */, const int /* i */, const int /* j */,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
							return GradientRangeMaternGeneralShape(cm, dist_ij, par_aux, shape_);
						};
				}
			}//end matern
			else if (cov_fct_type_ == "gaussian") {
				GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int /* ind_par */, const int i, const int j,
					const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientRangeGaussian(cm, dist_ij, sigma, i, j);
					};
			}
			else if (cov_fct_type_ == "powered_exponential") {
				GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int /* ind_par */, const int i, const int j,
					const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientRangePoweredExponential(cm, dist_ij, sigma, i, j);
					};
			}
			else if (cov_fct_type_ == "matern_space_time") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_par, const int i, const int j,
						const double dist_ij, const T_mat& sigma, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternSpaceTimeShape0_5(cm, dist_ij, sigma, ind_par, i, j, coords_ptr, coords_pred_ptr);
						};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_par, const int i, const int j,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternSpaceTimeShape1_5(cm, dist_ij, ind_par, i, j, coords_ptr, coords_pred_ptr);
						};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_par, const int i, const int j,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternSpaceTimeShape2_5(cm, dist_ij, ind_par, i, j, coords_ptr, coords_pred_ptr);
						};
				}
				else {// general shape
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_par, const int i, const int j,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternSpaceTimeGeneralShape(cm, dist_ij, ind_par, i, j, coords_ptr, coords_pred_ptr);
						};
				}
			}//end matern_space_time
			else if (cov_fct_type_ == "matern_ard") {
				if (TwoNumbersAreEqual<double>(shape_, 0.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_par, const int i, const int j,
						const double dist_ij, const T_mat& sigma, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternARDShape0_5(cm, dist_ij, sigma, ind_par, i, j, coords_ptr, coords_pred_ptr);
						};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 1.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_par, const int i, const int j,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternARDShape1_5(cm, dist_ij, ind_par, i, j, coords_ptr, coords_pred_ptr);
						};
				}
				else if (TwoNumbersAreEqual<double>(shape_, 2.5)) {
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_par, const int i, const int j,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternARDShape2_5(cm, dist_ij, ind_par, i, j, coords_ptr, coords_pred_ptr);
						};
				}
				else {// general shape
					GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
						double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
						const int ind_par, const int i, const int j,
						const double dist_ij, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
							return GradientRangeMaternARDGeneralShape(cm, dist_ij, 1., ind_par, i, j, coords_ptr, coords_pred_ptr, shape_);
						};
				}
			}//end matern_ard
			else if (cov_fct_type_ == "matern_estimate_shape") {
				GradientCovFct_ = [this](double cm, double cm_num_deriv, double par_aux, double shape,
					double par_aux_up, double par_aux_down, double pars_2_up, double pars_2_down,
					const int ind_par, const int /* i */, const int /* j */,
					const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientMaternEstimateShape(cm, cm_num_deriv, dist_ij, par_aux, par_aux_up, par_aux_down, pars_2_up, pars_2_down, shape, ind_par);
					};
			}
			else if (cov_fct_type_ == "matern_ard_estimate_shape") {
				GradientCovFct_ = [this](double cm, double cm_num_deriv, double par_aux, double shape,
					double par_aux_up, double par_aux_down, double pars_2_up, double pars_2_down,
					const int ind_par, const int i, const int j,
					const double dist_ij, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
						return GradientMaternARDEstimateShape(cm, cm_num_deriv, dist_ij, par_aux, par_aux_up, par_aux_down, pars_2_up, pars_2_down, shape, ind_par, i, j, coords_ptr, coords_pred_ptr);
					};
			}
			else if (cov_fct_type_ == "gaussian_ard") {
				GradientCovFct_ = [this](double cm, double /* cm_num_deriv */, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int ind_par, const int i, const int j,
					const double /* dist_ij */, const T_mat& sigma, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
						return GradientRangeGaussianARD(cm, sigma, ind_par, i, j, coords_ptr, coords_pred_ptr);
					};
			}
			else if (cov_fct_type_ == "space_time_gneiting") {
				GradientCovFct_coords_ = [this](const int i, const int j, const den_mat_t* coords, const den_mat_t* coords_pred,
					const vec_t& pars, const int ind_par, bool transf_scale, double nugget_var) -> double {
					return GradientSpaceTimeGneitingCovariance(i, j, coords, coords_pred, pars, ind_par, transf_scale, nugget_var);
				};
			}
			else if (cov_fct_type_ == "hurst") {
				GradientCovFct_coords_ = [this](const int i, const int j, const den_mat_t* coords, const den_mat_t* coords_pred,
					const vec_t& pars, const int /* ind_par */, bool transf_scale, double nugget_var) -> double {
						return GradientHurstCovariance(i, j, coords, coords_pred, pars, transf_scale, nugget_var);
				};
				GradientCovFct_coords_vec_ = [this](const vec_t& coords, const vec_t& coords_pred, const vec_t& pars, 
					const int /* ind_par */, bool transf_scale, double nugget_var) -> double {
						return GradientHurstCovariance_vec(coords, coords_pred, pars, transf_scale, nugget_var);
				};
			} else if (cov_fct_type_ == "hurst_ard") {
				GradientCovFct_coords_ = [this](const int i, const int j, const den_mat_t* coords, const den_mat_t* coords_pred,
					const vec_t& pars, const int ind_par, bool transf_scale, double nugget_var) -> double {
						return GradientHurstCovarianceARD(i, j, coords, coords_pred, pars, ind_par, transf_scale, nugget_var);
				};
				GradientCovFct_coords_vec_ = [this](const vec_t& coords, const vec_t& coords_pred, const vec_t& pars,
					const int ind_par, bool transf_scale, double nugget_var) -> double {
						return GradientHurstCovarianceARD_vec(coords, coords_pred, pars, ind_par, transf_scale, nugget_var);
				};
			}
			else if (cov_fct_type_ != "wendland" && cov_fct_type_ != "linear") {
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
#if HAS_STD_CYL_BESSEL_K
			double range_dist = cm_dist * dist_ij;
			return(cm * std::pow(range_dist, shape) * (2. * shape * std::cyl_bessel_k(shape, range_dist) - range_dist * std::cyl_bessel_k(shape + 1., range_dist)));
#else
			return(0.);
#endif
		}//end GradientRangeMaternGeneralShape

		inline double GradientSmoothnessMaternEstimateShapeFiniteDifference(double cm,
			double cm_num_deriv,
			double dist_ij,
			double par_aux,
			double par_aux_up,
			double par_aux_down,
			double pars_2_up,
			double pars_2_down,
			double shape) const {
#if HAS_STD_CYL_BESSEL_K
			double z = dist_ij * par_aux;
			double z_up = dist_ij * par_aux_up;
			double z_down = dist_ij * par_aux_down;
			double bessel_num_deriv = (std::cyl_bessel_k(pars_2_up, z_up) - std::cyl_bessel_k(pars_2_down, z_down)) / (2. * delta_step_);

			// for debugging
			//double grad = std::pow(z, shape) * (cm * std::cyl_bessel_k(shape, z) * (std::log(z / 2.) + 0.5 - GPBoost::digamma(shape)) + cm_num_deriv * bessel_num_deriv);
			//Log::REDebug("cm = %g, cm_num_deriv = %g, pars_2_up = %g, par_aux = %g, par_aux_up = %g, shape = %g",
			//	cm, cm_num_deriv, pars_2_up, par_aux, par_aux_up, shape);
			//Log::REDebug("bessel_num_deriv = %g, std::pow(z, shape) = %g, rest = %g, grad = %g",
			//	bessel_num_deriv, std::pow(z, shape), cm * std::cyl_bessel_k(shape, z) * (std::log(z / 2.) + 0.5 - GPBoost::digamma(shape)), grad);
			//std::this_thread::sleep_for(std::chrono::milliseconds(100));
			// doing calculations on log-scale
			//double grad_2 = std::exp(shape * std::log(z) + std::log(cm * std::cyl_bessel_k(shape, z) * (std::log(z / 2.) + 0.5 - GPBoost::digamma(shape)) + cm_num_deriv * bessel_num_deriv));
			//double grad_2 = std::exp(shape * std::log(z) + std::log(std::exp(std::log(cm) + std::log(std::cyl_bessel_k(shape, z)) + std::log(std::log(z / 2.) + 0.5 - GPBoost::digamma(shape))) + std::exp(std::log(cm_num_deriv) + std::log(bessel_num_deriv))));
			// numerical gradient for entire covariance function
			//double grad_3 = (std::pow(2., 1 - pars_2_up) / std::tgamma(pars_2_up) * std::pow(z_up, pars_2_up) * std::cyl_bessel_k(pars_2_up, z_up) -
			//	std::pow(2., 1 - pars_2_down) / std::tgamma(pars_2_down) * std::pow(z_down, pars_2_down) * std::cyl_bessel_k(pars_2_down, z_down)) / (2. * delta_step_);
			//Log::REDebug("grad = %g, grad_2 = %g, grad_3 = %g ", grad, grad_2, grad_3);//for debugging

			return (std::pow(z, shape) * (cm * std::cyl_bessel_k(shape, z) * (std::log(z / 2.) + 0.5 - GPBoost::digamma(shape)) + cm_num_deriv * bessel_num_deriv));
#else
			return(0.);
#endif
		}//end GradientSmoothnessMaternEstimateShapeFiniteDifference

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
				return(GradientSmoothnessMaternEstimateShapeFiniteDifference(cm, cm_num_deriv, dist_ij, par_aux,
					par_aux_up, par_aux_down, pars_2_up, pars_2_down, shape));
			}
			return(1.);
		}//end GradientMaternEstimateShape

		inline double GradientMaternARDEstimateShape(double cm,
			double cm_num_deriv,
			const double dist_ij,
			double par_aux,
			double par_aux_up,
			double par_aux_down,
			double pars_2_up,
			double pars_2_down,
			double shape,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			if (ind_range < num_cov_par_ - 2) {
				return(GradientRangeMaternARDGeneralShape(cm, dist_ij, par_aux, ind_range, i, j, coords_ptr, coords_pred_ptr, shape));
			}
			else if (ind_range == num_cov_par_ - 2) {//gradient wrt smoothness parameter
				return(GradientSmoothnessMaternEstimateShapeFiniteDifference(cm, cm_num_deriv, dist_ij, par_aux,
					par_aux_up, par_aux_down, pars_2_up, pars_2_down, shape));
			}
			return(0.);
		}//end GradientMaternARDEstimateShape

		inline double GradientRangeMaternSpaceTimeShape0_5(double cm,
			const double dist_ij,
			const T_mat& sigma,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			int dim_space = (int)(*coords_ptr).cols() - 1;
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
			const double dist_ij,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			int dim_space = (int)(*coords_ptr).cols() - 1;
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
			const double dist_ij,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			int dim_space = (int)(*coords_ptr).cols() - 1;
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
			const double dist_ij,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
#if HAS_STD_CYL_BESSEL_K
			int dim_space = (int)(*coords_ptr).cols() - 1;
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
			return(0.);
#endif
		}//end GradientRangeMaternSpaceTimeGeneralShape

		inline double GradientRangeMaternARDShape0_5(double cm,
			const double dist_ij,
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
				return(cm * dist_sq_ij_coord / dist_ij * sigma.coeff(i, j));
			}
		}//end GradientRangeMaternARDShape0_5

		inline double GradientRangeMaternARDShape1_5(double cm,
			const double dist_ij,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			double dist_sq_ij_coord = ((*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range));
			dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
			return(cm * dist_sq_ij_coord * std::exp(-dist_ij));
		}//end GradientRangeMaternARDShape1_5

		inline double GradientRangeMaternARDShape2_5(double cm,
			const double dist_ij,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			double dist_sq_ij_coord = ((*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range));
			dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
			return(cm * dist_sq_ij_coord * (1 + dist_ij) * std::exp(-dist_ij));
		}//end GradientRangeMaternARDShape2_5

		inline double GradientRangeMaternARDGeneralShape(double cm,
			const double dist_ij,
			double cm_dist,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr,
			const double shape) const {
#if HAS_STD_CYL_BESSEL_K
			double range_dist = cm_dist * dist_ij;
			double dist_sq_ij_coord = ((*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range));
			dist_sq_ij_coord = dist_sq_ij_coord * dist_sq_ij_coord;
			return(cm * std::pow(range_dist, shape - 2.) * (2. * shape * std::cyl_bessel_k(shape, range_dist) - range_dist * std::cyl_bessel_k(shape + 1., range_dist)) * dist_sq_ij_coord);
#else
			return(0.);
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

		/*!
		* \brief Calculate space-time covariance function in Eq. (16) of Gneiting (2002)
		* \param i Location for coords_ptr
		* \param j Location for coords_pred_ptr
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Parameter in the following order: sigma2, a, c, alpha, nu, beta, delta
		* \return covariance
		*/
		inline double SpaceTimeGneitingCovariance(const int i,
			const int j,
			const den_mat_t* coords,
			const den_mat_t* coords_pred,
			const vec_t& pars) const {
			vec_t coords_vec = (*coords).row(j);
			vec_t coords_pred_vec = (*coords_pred).row(i);
			return SpaceTimeGneitingCovariance_vec(coords_vec, coords_pred_vec, pars);
		}// end SpaceTimeGneitingCovariance
		inline double SpaceTimeGneitingCovariance_vec(const vec_t& coords,
			const vec_t& coords_pred,
			const vec_t& pars) const {
			int dim_space = (int)coords.size() - 1;
			double dist_time = std::abs(coords_pred[0] - coords[0]);
			double dist_space = (coords_pred.tail(dim_space) - coords.tail(dim_space)).norm();
			double d_aux_time = pars[1] * std::pow(dist_time, 2 * pars[3]) + 1.;
			double d_aux = pars[2] * dist_space / (std::pow(d_aux_time, pars[5] / 2.));
			double d_aux2 = pars[0] / (std::pow(d_aux_time, pars[6] + pars[5] * dim_space / 2.));
			if (TwoNumbersAreEqual<double>(pars[4], 0.5)) {
				return(d_aux2 * std::exp(-d_aux));
			}
			else if (TwoNumbersAreEqual<double>(pars[4], 1.5)) {
				return(d_aux2 * (1. + d_aux) * std::exp(-d_aux));
			}
			else if (TwoNumbersAreEqual<double>(pars[4], 2.5)) {
				return(d_aux2 * (1. + d_aux + d_aux * d_aux / 3.) * std::exp(-d_aux));
			}
			else {
#if HAS_STD_CYL_BESSEL_K
				if (d_aux < EPSILON_NUMBERS) {
					return(d_aux2);
				}
				else {
					return(d_aux2 * std::pow(2., 1 - pars[4]) / std::tgamma(pars[4]) * std::pow(d_aux, pars[4]) * std::cyl_bessel_k(pars[4], d_aux));
				}
#else
				Log::REFatal("'shape' of %g is not supported for the '%s' covariance function (only 0.5, 1.5, and 2.5) when using this compiler (e.g. Clang on Mac). Use gcc or (a newer version of) MSVC instead. ", pars[4], cov_fct_type_.c_str());
				return(0.);
#endif
			}
		}// end SpaceTimeGneitingCovariance_vec

		/*!
		* \brief Calculate gradient of space-time covariance function in Eq. (16) of Gneiting (2002)
		* \param i Row index for coords
		* \param j Row index for coords_pred
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Parameter in the following order: sigma2, a, c, alpha, nu, beta, delta
		* \param ind_par Parameter number for which the gradient is calculated, from 0 to 5 for a, c, alpha, nu, beta, delta. 
		*			Note: sigma2 is not included as the gradient is trivial and computed elsewhere. 
		*			ind_par thus starts at 0 for a, but pars[ind_par + 1] gives the corresponding parameter in pars
		* \param transf_scale On transformed  scale or not
		* \param nugget_var Nugget variance
		* \return Gradient of covariance
		*/
		inline double GradientSpaceTimeGneitingCovariance(const int i,
			const int j,
			const den_mat_t* coords,
			const den_mat_t* coords_pred,
			const vec_t& pars,
			const int ind_par,
			bool transf_scale,
			double nugget_var) const {
			CHECK(0 <= ind_par && ind_par <= 5);
			vec_t coords_vec = (*coords).row(j);
			vec_t coords_pred_vec = (*coords_pred).row(i);
			int dim_space = (int)coords->cols() - 1;
			double dist_time = std::abs(coords_pred_vec[0] - coords_vec[0]);
			double dist_space = (coords_pred_vec.tail(dim_space) - coords_vec.tail(dim_space)).norm();
			double grad = transf_scale ? 1. : 0.;
			double d_aux_time = pars[1] * std::pow(dist_time, 2 * pars[3]) + 1.;// = a*u^(2*alpha) + 1
			double d_aux;
			if (dist_space < EPSILON_NUMBERS) {
				d_aux = 0;
			}
			else {
				d_aux = pars[2] * dist_space / (std::pow(d_aux_time, pars[5] / 2.));// = c * |h| / ( (a*u^(2*alpha) + 1)^(beta/2) )
			}
			double d_aux2 = pars[0] / (std::pow(d_aux_time, pars[6] + pars[5] * dim_space / 2.));// = sigma2 / ( (a*u^(2*alpha) + 1)^(delta + beta*d/2) )
			double cm = transf_scale ? pars[ind_par + 1] : nugget_var;// multiplicative constant to get gradient on log-scale or backtransform with nugget variance
			if (ind_par == 0 || ind_par == 1 || ind_par == 2 || ind_par == 4 || ind_par == 5) {//a, c, alpha, beta, delta
				double d_aux_grad = 0., d_aux2_grad = 0.;
				if (ind_par == 0) {//a
					double c_aux = std::pow(dist_time, 2 * pars[3]);
					d_aux_grad = -pars[5] / 2. * pars[2] * dist_space / (std::pow(d_aux_time, pars[5] / 2. + 1.)) * c_aux;
					d_aux2_grad = -(pars[6] + pars[5] * dim_space / 2.) * pars[0] / (std::pow(d_aux_time, pars[6] + pars[5] * dim_space / 2. + 1.)) * c_aux;
				}
				else if (ind_par == 1) {//c
					if (dist_space < EPSILON_NUMBERS) {
						d_aux_grad = 0;
					}
					else {
						d_aux_grad = dist_space / (std::pow(d_aux_time, pars[5] / 2.));
					}
					d_aux2_grad = 0.;
				}
				else if (ind_par == 2) {//alpha
					if (dist_time < EPSILON_NUMBERS) {
						d_aux_grad = 0;
						d_aux2_grad = 0;
					}
					else {
						double c_aux = 2 * pars[1] * std::log(dist_time) * std::pow(dist_time, 2 * pars[3]);// d/dalpha (a u^(2 alpha) + 1), u = dist_time
						if (dist_space < EPSILON_NUMBERS) {
							d_aux_grad = 0;
						}
						else {
							d_aux_grad = -pars[5] / 2. * pars[2] * dist_space / (std::pow(d_aux_time, pars[5] / 2. + 1.)) * c_aux;
						}
						d_aux2_grad = -(pars[6] + pars[5] * dim_space / 2.) * pars[0] / (std::pow(d_aux_time, pars[6] + pars[5] * dim_space / 2. + 1.)) * c_aux;
					}
				}
				else if (ind_par == 4) {//beta
					if (dist_space < EPSILON_NUMBERS) {
						d_aux_grad = 0;
					}
					else {
						d_aux_grad = -pars[2] * dist_space / 2. * std::log(d_aux_time) / (std::pow(d_aux_time, pars[5] / 2.));
					}
					d_aux2_grad = -pars[0] * dim_space / 2. * std::log(d_aux_time) / (std::pow(d_aux_time, pars[6] + pars[5] * dim_space / 2.));
				}
				else if (ind_par == 5) {//delta
					d_aux_grad = 0.;
					d_aux2_grad = -pars[0] * std::log(d_aux_time) / (std::pow(d_aux_time, pars[6] + pars[5] * dim_space / 2.));
				}
				if (TwoNumbersAreEqual<double>(pars[4], 0.5)) {
					grad = (d_aux2_grad - d_aux2 * d_aux_grad) * std::exp(-d_aux);
				}
				else if (TwoNumbersAreEqual<double>(pars[4], 1.5)) {
					grad = (d_aux2_grad * (1. + d_aux) - d_aux2 * d_aux * d_aux_grad) * std::exp(-d_aux);
				}
				else if (TwoNumbersAreEqual<double>(pars[4], 2.5)) {
					grad = (d_aux2_grad * (1. + d_aux + d_aux * d_aux / 3.) -
						d_aux2 * (d_aux + d_aux * d_aux) / 3. * d_aux_grad) * std::exp(-d_aux);
				}
				else {
#if HAS_STD_CYL_BESSEL_K
					if (d_aux < EPSILON_NUMBERS) {
						grad = d_aux2_grad;
					}
					else {
						double bessel = std::cyl_bessel_k(pars[4], d_aux);
						double grad_bessel = pars[4] / d_aux * bessel - std::cyl_bessel_k(pars[4] + 1., d_aux); // d/dz K_nu(z) = nu / z * K_nu(z) - K_(nu+1)(z)
						grad = d_aux2_grad * std::pow(d_aux, pars[4]) * bessel +
							d_aux2 * pars[4] * std::pow(d_aux, pars[4] - 1.) * d_aux_grad * bessel +
							d_aux2 * std::pow(d_aux, pars[4]) * d_aux_grad * grad_bessel;
						grad *= std::pow(2., 1 - pars[4]) / std::tgamma(pars[4]);
					}
#else
					grad = 0.;
#endif
				}
			}//end a, c, alpha, beta, delta
			else if (ind_par == 3) {//nu
#if HAS_STD_CYL_BESSEL_K
				double nu_up, nu_down;
				if (transf_scale) {
					nu_up = std::exp(std::log(pars[4]) + delta_step_);//gradient on log-scale
					nu_down = std::exp(std::log(pars[4]) - delta_step_);
				}
				else {
					nu_up = pars[4] + delta_step_;
					nu_down = pars[4] - delta_step_;
					CHECK(nu_down > 0.);
				}
				if (d_aux < EPSILON_NUMBERS) {
					grad = 0;
				}
				else {
					cm = 1.;
					const double cm_deriv = transf_scale ? pars[4] : nugget_var;
					const double cm_num_deriv = transf_scale ? 1 : nugget_var;//already on log-scale if transf_scale
					const double bessel_num_deriv = (std::cyl_bessel_k(nu_up, d_aux) - std::cyl_bessel_k(nu_down, d_aux)) / (2. * delta_step_);
					const double bessel = std::cyl_bessel_k(pars[4], d_aux);
					grad = std::pow(2., 1 - pars[4]) / std::tgamma(pars[4]) * d_aux2 * std::pow(d_aux, pars[4]) *
						(cm_deriv * bessel * (-std::log(2.) - GPBoost::digamma(pars[4]) + std::log(d_aux)) + // (i) d/dnu 2^{1-v} = -log(2)*2^{1-v}, (ii) d/dnu 1/Gamma(nu) = -digamm(nu) / Gamma(nu), (iii) d/dnu d_aux^v = log(d_aux)*d_aux^v
							cm_num_deriv * bessel_num_deriv);
				}
#else
				grad = 0.;
#endif
			}
			else {
				Log::REFatal("GradientSpaceTimeGneitingCovariance: not yet implemented for ind_par = %d ", ind_par);
			}
			grad *= cm;
			return(grad);
		}// end GradientSpaceTimeGneitingCovariance

		/*!
		* \brief Calculate Hurst covariance function
		* \param i Location for coords_ptr
		* \param j Location for coords_pred_ptr
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Parameter in the following order: sigma2, H
		* \return covariance
		*/
		inline double HurstCovariance(const int i,
			const int j,
			const den_mat_t* coords,
			const den_mat_t* coords_pred,
			const vec_t& pars) const {
			vec_t coords_vec = (*coords).row(j);
			vec_t coords_pred_vec = (*coords_pred).row(i);
			return HurstCovariance_vec(coords_vec, coords_pred_vec, pars);
		}// end SpaceTimeGneitingCovariance
		inline double HurstCovariance_vec(const vec_t& coords_vec,
			const vec_t& coords_pred_vec,
			const vec_t& pars) const {
			const double sqrd_norm_x = coords_vec.squaredNorm();
			const double sqrd_norm_y = coords_pred_vec.squaredNorm();
			const double sqrd_norm_x_min_y = (coords_vec - coords_pred_vec).squaredNorm();
			const double H = std::exp(-pars[1]);
			return (pars[0] / 2.) * (std::pow(sqrd_norm_x, H) + std::pow(sqrd_norm_y, H) - std::pow(sqrd_norm_x_min_y, H));
		}// end HurstCovariance_vec

		/*!
		* \brief Calculate gradient of Hurst covariance
		* \param i Row index for coords
		* \param j Row index for coords_pred
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Parameter in the following order: sigma2, H
		* \param ind_par Parameter number for which the gradient is calculated
		*			Note: sigma2 is not included as the gradient is trivial and computed elsewhere.
		*			ind_par thus starts at 0 for a, but pars[ind_par + 1] gives the corresponding parameter in pars
		* \param transf_scale On transformed  scale or not
		* \param nugget_var Nugget variance
		* \return Gradient of covariance
		*/
		inline double GradientHurstCovariance(const int i,
			const int j,
			const den_mat_t* coords,
			const den_mat_t* coords_pred,
			const vec_t& pars,
			bool transf_scale,
			double nugget_var) const {
			vec_t coords_vec = (*coords).row(j);
			vec_t coords_pred_vec = (*coords_pred).row(i);
			return GradientHurstCovariance_vec(coords_vec, coords_pred_vec, pars, transf_scale, nugget_var);
		}
		inline double GradientHurstCovariance_vec(const vec_t& coords_vec,
			const vec_t& coords_pred_vec,
			const vec_t& pars,
			bool transf_scale,
			double nugget_var) const {
			const double sqrd_norm_x = coords_vec.squaredNorm();
			const double sqrd_norm_y = coords_pred_vec.squaredNorm();
			const double sqrd_norm_x_min_y = (coords_vec - coords_pred_vec).squaredNorm();
			const double H = std::exp(-pars[1]);
			const double cm = transf_scale ? -H * pars[1] : nugget_var;// multiplicative constant to get gradient on log-scale or backtransform with nugget variance
			const double rx_H = std::pow(sqrd_norm_x, H);
			const double ry_H = std::pow(sqrd_norm_y, H);
			const double rxy_H = std::pow(sqrd_norm_x_min_y, H);
			auto safe_rH_times_log_x = [](double x, double rH) {
				return (x > 0.0) ? (rH * std::log(x)) : 0.0;
			};
			const double grad = cm * 0.5 * pars[0] * (safe_rH_times_log_x(sqrd_norm_x, rx_H) + safe_rH_times_log_x(sqrd_norm_y, ry_H) - safe_rH_times_log_x(sqrd_norm_x_min_y, rxy_H));
			return(grad);
		}//end GradientHurstCovariance_vec

		/*!
		* \brief Calculate gradient of ARD Hurst covariance
		* \param i Row index for coords
		* \param j Row index for coords_pred
		* \param coords Coordinates
		* \param coords_pred Coordinates
		* \param pars Parameter in the following order: sigma2, H, r_2, ..., r_p, where r_j denotes the range for coordinate number j
		* \param ind_par Parameter number for which the gradient is calculated
		*			Note: sigma2 is not included as the gradient is trivial and computed elsewhere.
		*			ind_par thus starts at 0 for a, but pars[ind_par + 1] gives the corresponding parameter in pars
		* \param transf_scale On transformed  scale or not
		* \param nugget_var Nugget variance
		* \return Gradient of covariance
		*/
		inline double GradientHurstCovarianceARD(const int i,
			const int j,
			const den_mat_t* coords,
			const den_mat_t* coords_pred,
			const vec_t& pars,
			const int ind_par,
			bool transf_scale,
			double nugget_var) const {
			vec_t coords_vec = (*coords).row(j);
			vec_t coords_pred_vec = (*coords_pred).row(i);
			return GradientHurstCovarianceARD_vec(coords_vec, coords_pred_vec, pars, ind_par, transf_scale, nugget_var);
		}
		inline double GradientHurstCovarianceARD_vec(const vec_t& coords_vec,
			const vec_t& coords_pred_vec,
			const vec_t& pars,
			const int ind_par,
			bool transf_scale,
			double nugget_var) const {
			CHECK(0 <= ind_par && ind_par <= num_cov_par_ - 1); // num_cov_par_ = dim_coords + 1
			const double sqrd_norm_x = coords_vec.squaredNorm();
			const double sqrd_norm_y = coords_pred_vec.squaredNorm();
			const double sqrd_norm_x_min_y = (coords_vec - coords_pred_vec).squaredNorm();
			double grad;
			const double H = std::exp(-pars[1]);
			if (ind_par == 0) {				
				const double cm = transf_scale ? -H * pars[1] : nugget_var;// multiplicative constant to get gradient on log-scale or backtransform with nugget variance
				const double rx_H = std::pow(sqrd_norm_x, H);
				const double ry_H = std::pow(sqrd_norm_y, H);
				const double rxy_H = std::pow(sqrd_norm_x_min_y, H);
				auto safe_rH_times_log_x = [](double x, double rH) {
					return (x > 0.0) ? (rH * std::log(x)) : 0.0;
				};
				grad = cm * 0.5 * pars[0] * (safe_rH_times_log_x(sqrd_norm_x, rx_H) + safe_rH_times_log_x(sqrd_norm_y, ry_H) - safe_rH_times_log_x(sqrd_norm_x_min_y, rxy_H));
			}
			else {// ----- gradient w.r.t. ARD range parameter l_k -----				
				const int k = ind_par;   // coordinate index (1 ... num_range)
				const double l_k = std::max(pars[ind_par + 1], 1e-12);// The range l_k is stored in pars[ind_par + 1]
				auto safe_pow_Hm1 = [&](double x) {// avoids pow(0, negative) and inf*0 -> NaN					
					return (x > EPSILON_NUMBERS) ? std::pow(x, H - 1.0) : 0.0;
				};
				const double r_x_Hm1 = safe_pow_Hm1(sqrd_norm_x);
				const double r_y_Hm1 = safe_pow_Hm1(sqrd_norm_y);
				const double r_xy_Hm1 = safe_pow_Hm1(sqrd_norm_x_min_y);
				const double x_k = coords_vec[k];
				const double y_k = coords_pred_vec[k];
				const double diff_k = x_k - y_k;
				const double dC_dl_k = -pars[0] * H / l_k * (r_x_Hm1 * x_k * x_k + r_y_Hm1 * y_k * y_k - r_xy_Hm1 * diff_k * diff_k);
				const double cm = transf_scale ? l_k : nugget_var;// multiplicative constant to get gradient on log-scale or backtransform with nugget variance
				grad = cm * dC_dl_k;
			}
			return(grad);
		}//end GradientHurstCovarianceARD_vec

		inline void ParseCovFunctionAlias(string_t& likelihood,
			double& shape) const {
			if (likelihood == string_t("exponential_space_time") || likelihood == string_t("Matern_space_time")) {
				likelihood = "matern_space_time";
				shape = 0.5;
			}
			else if (likelihood == string_t("exponential_ard") || likelihood == string_t("Matern_ard")) {
				likelihood = "matern_ard";
				shape = 0.5;
			}
			else if (likelihood == string_t("exponential") || likelihood == string_t("Matern")) {
				likelihood = "matern";
				shape = 0.5;
			}
			else if (likelihood == string_t("Gaussian")) {
				likelihood = "gaussian";
			}
			else if (likelihood == string_t("Hurst")) {
				likelihood = "hurst";
			}
			else if (likelihood == string_t("Hurst_ard")) {
				likelihood = "hurst_ard";
			}
		}

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
		/*! \brief If true, scaled coordinates are used for calculating covariances (e.g., for ARD kernels) and for determining inducing points in low-rank approximations */
		bool use_scaled_coordinates_ = false;
		/*! \brief If true, Vecchia neigbors are redetermined in a transformed space during parameter estimation (e.g., for ARD kernels) */
		bool redetermine_vecchia_neighbors_in_transformed_space_ = false;
		/*! \brief If true, covariances are calculated from coordinates and cannot be calculated from (potentially scaled) distances (except for the 'linear' kernel) */
		bool cov_calculated_based_on_coords_ = false;
		/*! \brief If true, coordinates are required for calculating distances either because the covariance depends on them or because scaled distances are used */
		bool need_coordinates_for_calculating_covariance_ = false;
		/*! \brief If true, the diagonal of the covariance is constant and equal to the marginal variance */
		bool variance_on_the_diagonal_ = true;
		/*! \brief If true, precomputed distances('dist') are used for calculating covariances, otherwise the coordinates are used('coords' and 'coords_pred') */
		bool use_precomputed_dist_for_calc_cov_;
		/*! \brief for calculating finite differences  */
		const double delta_step_ = 1e-6;// based on https://math.stackexchange.com/questions/815113/is-there-a-general-formula-for-estimating-the-step-size-h-in-numerical-different/819015#819015		
		/*! \brief List of supported covariance functions */
		const std::set<string_t> SUPPORTED_COV_TYPES_{ "exponential", "powered_exponential",
			"matern", "matern_ard", "matern_space_time", "matern_estimate_shape", "matern_ard_estimate_shape", 
			"gaussian", "gaussian_ard",
			"space_time_gneiting", "wendland", "linear", "hurst", "hurst_ard"};
		const double LARGE_SHAPE_WARNING_THRESHOLD_ = 50.;
		const char* LARGE_SHAPE_WARNING_ = "The shape parameter is very large, it is recommended to use the 'gausian' covariance funtion ";

		template<typename>
		friend class RECompGP;
	};

}  // namespace GPBoost

#endif   // GPB_COV_FUNCTIONS_
