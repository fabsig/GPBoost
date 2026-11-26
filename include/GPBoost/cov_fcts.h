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
#include <GPBoost/bessel_k.h>

#include <string>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <functional>

#include <LightGBM/utils/log.h>
using LightGBM::Log;


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
			order_(other.order_),
			matern_half_integer_order_(other.matern_half_integer_order_),
			matern_poly_coef_(other.matern_poly_coef_),
			matern_poly_coef_deriv_(other.matern_poly_coef_deriv_),
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
			grad_is_multiple_of_cov_(other.grad_is_multiple_of_cov_),
			use_precomputed_dist_for_calc_cov_(other.use_precomputed_dist_for_calc_cov_),
			is_ar1_multifidelity_(other.is_ar1_multifidelity_),
			base_cov_fct_type_(other.base_cov_fct_type_),
			num_cov_par_base_(other.num_cov_par_base_),
			dim_coordinates_(other.dim_coordinates_)
		{
			if (other.base_cov_function_low_) {
				base_cov_function_low_ = std::make_shared<CovFunction<T_mat>>(*other.base_cov_function_low_);
				base_cov_function_discrepancy_ = std::make_shared<CovFunction<T_mat>>(*other.base_cov_function_discrepancy_);
			}
			// re-initialize the function pointers
			if (!is_ar1_multifidelity_) {
				InitializeCovFct();
				InitializeCovFctGrad();
				InitializeGetDistanceForCovFct();
				InitializeGetDistanceForGradientCovFct();
			}
		}

		/*!
		* \brief Constructor
		* \param cov_fct_type Type of covariance function
		* \param shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. This parameter is irrelevant for some covariance functions such as the exponential or Gaussian
		* \param order Order m of the 'hurst' and 'hurst_ard' covariance functions (m - 1 < H < m). This must be 1 for all other covariance functions
		* \param taper_range Range parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
		* \param taper_shape Shape parameter of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
		* \param taper_mu Parameter \mu of the Wendland covariance function and Wendland correlation taper function. We follow the notation of Bevilacqua et al. (2019, AOS)
		* \param apply_tapering If true, tapering is applied to the covariance function (element-wise multiplication with a compactly supported Wendland correlation function)
		* \param dim_coordinates Dimension of input coordinates / features
		* \param use_precomputed_dist_for_calc_cov If true, precomputed distances ('dist') are used for calculating covariances, otherwise the coordinates are used ('coords' and 'coords_pred')
		*/
		CovFunction(string_t cov_fct_type,
			double shape,
			int order,
			double taper_range,
			double taper_shape,
			double taper_mu,
			bool apply_tapering,
			int dim_coordinates,
			bool use_precomputed_dist_for_calc_cov) {
			const string_t ar1_mf_prefix = "ar1_mf_";
			if (cov_fct_type.rfind(ar1_mf_prefix, 0) == 0) {
				InitializeAR1Multifidelity(cov_fct_type.substr(ar1_mf_prefix.size()), cov_fct_type,
					shape, order, taper_range, taper_shape, taper_mu, apply_tapering, dim_coordinates);
				return;
			}
			if (cov_fct_type == "exponential_tapered") {
				Log::REFatal("Covariance of type 'exponential_tapered' is discontinued. Use the option 'gp_approx = \"tapering\"' instead ");
			}
			ParseCovFunctionAlias(cov_fct_type, shape);
			if (SUPPORTED_COV_TYPES_.find(cov_fct_type) == SUPPORTED_COV_TYPES_.end()) {
				Log::REFatal("Covariance of type '%s' is not supported ", cov_fct_type.c_str());
			}
			if (cov_fct_type == "hurst" || cov_fct_type == "hurst_ard") {
				if (order != 1 && order != 2) {
					Log::REFatal("'cov_fct_order' = %d is not supported for the '%s' covariance function. Currently supported orders are 1 and 2 ",
						order, cov_fct_type.c_str());
				}
			}
			else if (order != 1) {
				Log::REFatal("'cov_fct_order' = %d is not supported for the '%s' covariance function. "
					"An order other than 1 is only supported for the 'hurst' and 'hurst_ard' covariance functions ", order, cov_fct_type.c_str());
			}
			order_ = order;
			cov_fct_type_ = cov_fct_type;
			dim_coordinates_ = dim_coordinates;
			use_precomputed_dist_for_calc_cov_ = use_precomputed_dist_for_calc_cov;
			if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard" ||
				cov_fct_type_ == "matern_ard_estimate_shape" || cov_fct_type_ == "matern_space_time_estimate_shape" ||
				cov_fct_type_ == "gaussian_ard" || cov_fct_type_ == "space_time_gneiting" || cov_fct_type_ == "linear" ||
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
					cov_fct_type_ == "matern_ard_estimate_shape" || cov_fct_type_ == "matern_space_time" ||
					cov_fct_type_ == "matern_space_time_estimate_shape") {
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
			else if (cov_fct_type_ == "matern_space_time_estimate_shape") {
				num_cov_par_ = 4;
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
				cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard" || cov_fct_type_ == "matern_ard_estimate_shape" ||
				cov_fct_type_ == "matern_space_time_estimate_shape") {
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
			if (IsMaternWithFixedShape()) {
				CHECK(shape > 0.);
				matern_half_integer_order_ = MaternHalfIntegerOrder(shape);
				if (matern_half_integer_order_ >= 3) {
					const int p = matern_half_integer_order_;
					matern_poly_coef_ = MaternPolynomialCoefficients(p);
					matern_poly_coef_deriv_ = MaternPolynomialCoefficients(p - 1);
					for (double& coef : matern_poly_coef_deriv_) {
						coef /= 2. * p - 1.;
					}
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

		int CovFunctionOrder() const {
			return(order_);
		}

		/*!
		* \brief Returns true if the variance of the covariance function is zero at the origin of the coordinates (without the
		*		fidelity indicator for 'ar1_mf_' covariance functions), whatever the covariance parameters are. The Gaussian process
		*		of such a covariance function is zero at the origin, and so are its covariances with all other points
		* \param cov_fct_type Type of covariance function
		*/
		static bool HasZeroVarianceAtOrigin(string_t cov_fct_type) {
			const string_t ar1_mf_prefix = "ar1_mf_";
			if (cov_fct_type.rfind(ar1_mf_prefix, 0) == 0) {
				cov_fct_type = cov_fct_type.substr(ar1_mf_prefix.size());
			}
			return(cov_fct_type == "hurst" || cov_fct_type == "hurst_ard" || cov_fct_type == "Hurst" || cov_fct_type == "Hurst_ard" ||
				cov_fct_type == "linear");
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

		/*!
		* \brief True if the gradients with respect to the range parameters are multiples of the entries of the covariance
		*		matrix that is passed to them, so that they contain a taper that has been applied to this matrix
		*/
		bool GradientIsMultipleOfCov() const {
			return(grad_is_multiple_of_cov_);
		}

		bool UseScaledCoordinates() const {
			return(use_scaled_coordinates_);
		}

		bool RedetermineVecchiaNeighborsInTransformedSpace() const {
			return(redetermine_vecchia_neighbors_in_transformed_space_);
		}

		bool IsSpaceTimeModel() const {
			return(cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_space_time_estimate_shape" ||
				cov_fct_type_ == "space_time_gneiting" ||
				(is_ar1_multifidelity_ &&
					(base_cov_fct_type_ == "matern_space_time" || base_cov_fct_type_ == "matern_space_time_estimate_shape" ||
						base_cov_fct_type_ == "exponential_space_time" || base_cov_fct_type_ == "space_time_gneiting")));
		}

		bool IsAR1Multifidelity() const {
			return(is_ar1_multifidelity_);
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
			else if (cov_fct_type_ == "matern_space_time_estimate_shape") {
				coords_scaled.col(0) = coords.col(0) / pars[1];
				int dim_space = (int)coords.cols() - 1;
				coords_scaled.rightCols(dim_space) = coords.rightCols(dim_space) / pars[2];
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
			else if (cov_fct_type_ == "matern_space_time_estimate_shape") {
				coords_scaled_vec[0] = coords_vec[0] / pars[1];
				int dim_space = (int)coords_vec.size() - 1;
				coords_scaled_vec.tail(dim_space) = coords_vec.tail(dim_space) / pars[2];
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
			if (is_ar1_multifidelity_) {
				CHECK(pars.size() == num_cov_par_);
				base_cov_function_low_->CheckPars(pars.segment(0, num_cov_par_base_));
				base_cov_function_discrepancy_->CheckPars(pars.segment(num_cov_par_base_, num_cov_par_base_));
				CHECK(pars[2 * num_cov_par_base_] > 0.);
				return;
			}
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
			if (is_ar1_multifidelity_) {
				base_cov_function_low_->CovarianceParameterRangeWarning(pars.segment(0, num_cov_par_base_));
				base_cov_function_discrepancy_->CovarianceParameterRangeWarning(pars.segment(num_cov_par_base_, num_cov_par_base_));
				return;
			}
			if (IsMaternWithEstimatedShape()) {
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
		* \brief Correlation between the two corners of the bounding box of the coordinates. If it is large, the GP is almost
		*		constant over the data: the marginal variance and the range parameters can then increase together without changing
		*		the likelihood much, and the estimates depend on the initial values
		* \param pars Vector with covariance parameters (on transformed scale)
		* \param coords Coordinates
		* \return Correlation, or -1 if it is not applicable (e.g., tapering, less than two different coordinates)
		*/
		double CorrelationMostDistantCoords(const vec_t& pars,
			const den_mat_t& coords) const {
			if (is_ar1_multifidelity_ || cov_calculated_based_on_coords_ || cov_fct_type_ == "wendland" || cov_fct_type_ == "linear" ||
				apply_tapering_ || num_cov_par_ < 2 || coords.rows() < 2) {
				return -1.;
			}
			const vec_t extent = (coords.colwise().maxCoeff() - coords.colwise().minCoeff()).transpose();
			double dist_max;
			if (is_isotropic_) {
				dist_max = extent.norm();
			}
			else {
				vec_t extent_scaled;
				ScaleCoordinates_vec(pars, extent, extent_scaled);
				dist_max = extent_scaled.norm();
			}
			const double shape = IsMaternWithEstimatedShape() ? pars[num_cov_par_ - 1] : 0.;
			const double range = is_isotropic_ ? pars[1] : 1.;
			if (!(dist_max > 0.)) {
				return -1.;
			}
			return CovFct_dist_(dist_max, 1., range, shape);
		}//end CorrelationMostDistantCoords

		/*!
		* \brief Cap parameters
		* \param pars Vector with covariance parameters
		*/
		void CapPars(vec_t& pars) const {
			if (is_ar1_multifidelity_) {
				vec_t pars_low = pars.segment(0, num_cov_par_base_);
				vec_t pars_discrepancy = pars.segment(num_cov_par_base_, num_cov_par_base_);
				base_cov_function_low_->CapPars(pars_low);
				base_cov_function_discrepancy_->CapPars(pars_discrepancy);
				pars.segment(0, num_cov_par_base_) = pars_low;
				pars.segment(num_cov_par_base_, num_cov_par_base_) = pars_discrepancy;
				return;
			}
			if (cov_fct_type_ == "space_time_gneiting") {
				if (pars[3] > 1.) {//alpha
					pars[3] = 1.;
				}
				if (pars[4] > MAX_ESTIMATED_MATERN_SHAPE_) {//nu
					pars[4] = MAX_ESTIMATED_MATERN_SHAPE_;
				}
				if (pars[5] > 1.) {//beta
					pars[5] = 1.;
				}
			}
			else if (IsMaternWithEstimatedShape()) {
				if (pars[num_cov_par_ - 1] > MAX_ESTIMATED_MATERN_SHAPE_) {
					pars[num_cov_par_ - 1] = MAX_ESTIMATED_MATERN_SHAPE_;
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
			if (is_ar1_multifidelity_) {
				CHECK(pars.size() == num_cov_par_);
				pars_trans = vec_t(num_cov_par_);
				vec_t pars_low_trans, pars_discrepancy_trans;
				base_cov_function_low_->TransformCovPars(sigma2, pars.segment(0, num_cov_par_base_), pars_low_trans);
				base_cov_function_discrepancy_->TransformCovPars(sigma2,
					pars.segment(num_cov_par_base_, num_cov_par_base_), pars_discrepancy_trans);
				pars_trans.segment(0, num_cov_par_base_) = pars_low_trans;
				pars_trans.segment(num_cov_par_base_, num_cov_par_base_) = pars_discrepancy_trans;
				pars_trans[2 * num_cov_par_base_] = std::exp(pars[2 * num_cov_par_base_]);
				return;
			}
			pars_trans = pars;
			pars_trans[0] = pars[0] / sigma2;
			if (IsMaternWithFixedShape()) {
				const double range_scale = MaternRangeScale();
				for (int i = 1; i < num_cov_par_; ++i) {
					CHECK(pars[i] > 0.);
					pars_trans[i] = range_scale / pars[i];
				}
			}
			else if (cov_fct_type_ == "gaussian") {
				pars_trans[1] = 1. / (pars[1] * pars[1]);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				pars_trans[1] = 1. / (std::pow(pars[1], shape_));
			}
			else if (cov_fct_type_ == "gaussian_ard") {
				for (int i = 1; i < num_cov_par_; ++i) {
					pars_trans[i] = 1. / (pars[i] * pars[i]);
				}
			}
			else if (cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
				// p_H = -log(H - (m - 1)) > 0 for m - 1 < H < m
				if (!(pars[1] > order_ - 1. && pars[1] < order_)) {
					Log::REFatal("The Hurst exponent H must be in (%d,%d) for 'cov_fct_order' = %d, found %g ", order_ - 1, order_, order_, pars[1]);
				}
				pars_trans[1] = -std::log(pars[1] - (order_ - 1.));
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
			if (is_ar1_multifidelity_) {
				CHECK(pars.size() == num_cov_par_);
				pars_orig = vec_t(num_cov_par_);
				vec_t pars_low_orig, pars_discrepancy_orig;
				base_cov_function_low_->TransformBackCovPars(sigma2, pars.segment(0, num_cov_par_base_), pars_low_orig);
				base_cov_function_discrepancy_->TransformBackCovPars(sigma2,
					pars.segment(num_cov_par_base_, num_cov_par_base_), pars_discrepancy_orig);
				pars_orig.segment(0, num_cov_par_base_) = pars_low_orig;
				pars_orig.segment(num_cov_par_base_, num_cov_par_base_) = pars_discrepancy_orig;
				pars_orig[2 * num_cov_par_base_] = std::log(pars[2 * num_cov_par_base_]);
				return;
			}
			pars_orig = pars;
			pars_orig[0] = sigma2 * pars[0];
			if (IsMaternWithFixedShape()) {
				const double range_scale = MaternRangeScale();
				for (int i = 1; i < num_cov_par_; ++i) {
					pars_orig[i] = range_scale / pars[i];
				}
			}
			else if (cov_fct_type_ == "gaussian") {
				pars_orig[1] = 1. / std::sqrt(pars[1]);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				pars_orig[1] = 1. / (std::pow(pars[1], 1. / shape_));
			}
			else if (cov_fct_type_ == "gaussian_ard") {
				for (int i = 1; i < num_cov_par_; ++i) {
					pars_orig[i] = 1. / std::sqrt(pars[i]);
				}
			}
			else if (cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
				pars_orig[1] = HurstExponent(pars[1]);
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
			if (is_ar1_multifidelity_) {
				CalculateCovMatAR1Multifidelity(dist, coords, coords_pred, pars, sigma, is_symmmetric);
				return;
			}
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
				const double shape = IsMaternWithEstimatedShape() ? pars[num_cov_par_ - 1] : 0.;
				const double range = is_isotropic_ ? pars[1] : 1.;
				// calculate covariances
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						sigma(i, i) = variance_on_the_diagonal_ ? pars[0] : CovFct_dist_(GetDistanceForCovFct_(i, i, dist, coords_ptr, coords_pred_ptr), pars[0], range, shape);
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
			if (is_ar1_multifidelity_) {
				CalculateCovMatAR1Multifidelity(dist, coords, coords_pred, pars, sigma, is_symmmetric);
				return;
			}
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
				// covariances only for the entries in the sparsity pattern of 'dist' (e.g., the pairs within the taper range)
				const den_mat_t& coords_rows = is_symmmetric ? coords : coords_pred;
#pragma omp parallel for schedule(static)
				for (int k = 0; k < sigma.outerSize(); ++k) {
					for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
						it.valueRef() = pars[0] * coords_rows.row(it.row()).dot(coords.row(it.col()));
					}
				}
			}//end linear
			else {// cov_fct_type_ != "wendland"
				const double shape = IsMaternWithEstimatedShape() ? pars[num_cov_par_ - 1] : 0.;
				const double range = is_isotropic_ ? pars[1] : 1.;
				// calculate covariances
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int k = 0; k < sigma.outerSize(); ++k) {
						for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
							int i = (int)it.row();
							int j = (int)it.col();
							if (i == j) {
								it.valueRef() = variance_on_the_diagonal_ ? pars[0] : CovFct_dist_(GetDistanceForCovFct_(i, i, dist, coords_ptr, coords_pred_ptr), pars[0], range, shape);
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
			if (is_ar1_multifidelity_) {
				return CalculateCovarianceOneEntryAR1Multifidelity(coords, coords_pred, pars);
			}
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
				const double shape = IsMaternWithEstimatedShape() ? pars[num_cov_par_ - 1] : 0.;
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
			CheckWendlandTaperShape();
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
					}
				}
			}
		}//end MultiplyWendlandCorrelationTaper (dense)
		template <class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
		void MultiplyWendlandCorrelationTaper(const T_mat& dist,
			T_mat& sigma,
			bool is_symmmetric) const {
			CHECK(apply_tapering_);
			CheckWendlandTaperShape();
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
					}
				}
			}
		}//end MultiplyWendlandCorrelationTaper (sparse)

		/*! \brief Raise an error if 'taper_shape_' is not supported by 'MultiplyWendlandCorrelationTaper' (0, 1, or 2) */
		void CheckWendlandTaperShape() const {
			if (!TwoNumbersAreEqual<double>(taper_shape_, 0.) && !TwoNumbersAreEqual<double>(taper_shape_, 1.) &&
				!TwoNumbersAreEqual<double>(taper_shape_, 2.)) {
				Log::REFatal("MultiplyWendlandCorrelationTaper: 'taper_shape' of %g is not supported for the 'wendland' covariance function ", taper_shape_);
			}
		}//end CheckWendlandTaperShape

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
		* \brief Calculates the derivative with respect to any covariance parameter, including variance parameters
		*/
		void CalculateGradientCovMatFull(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const T_mat& sigma,
			const vec_t& pars,
			T_mat& sigma_grad,
			bool transf_scale,
			double nugget_var,
			int ind_par,
			bool is_symmmetric) const {
			CHECK(ind_par >= 0 && ind_par < num_cov_par_);
			if (is_ar1_multifidelity_) {
				CalculateGradientCovMatAR1Multifidelity(dist, coords, coords_pred, pars, sigma_grad,
					transf_scale, nugget_var, ind_par, is_symmmetric);
			}
			else if (ind_par == 0) {
				sigma_grad = sigma;
				if (!transf_scale) {
					sigma_grad /= pars[0];
				}
			}
			else {
				CalculateGradientCovMat(dist, coords, coords_pred, sigma, pars, sigma_grad,
					transf_scale, nugget_var, ind_par - 1, is_symmmetric);
			}
		}

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
			CHECK(!is_ar1_multifidelity_);
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
				double cm, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape;//constants and auxiliary parameters
				DetermineConstantsForGradient(pars, (int)coords.cols(), transf_scale, nugget_var, ind_par,
					cm, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape);
				// calculate gradients
				if (is_symmmetric) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_rows; ++i) {
						sigma_grad(i, i) = variance_on_the_diagonal_ ? 0. : GradientCovFct_(cm, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
							ind_par, i, i, GetDistanceForGradientCovFct_(i, i, dist, coords_ptr, coords_pred_ptr), sigma, coords_ptr, coords_pred_ptr);
						for (int j = i + 1; j < num_cols; ++j) {
							const double dist_ij = GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
							sigma_grad(i, j) = GradientCovFct_(cm, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
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
							sigma_grad(i, j) = GradientCovFct_(cm, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
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
			CHECK(!is_ar1_multifidelity_);
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
			double cm, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape;//constants and auxiliary parameters
			DetermineConstantsForGradient(pars, (int)coords.cols(), transf_scale, nugget_var, ind_par,
				cm, par_aux, pars_2_up, pars_2_down, par_aux_up, par_aux_down, shape);			
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
								it.valueRef() = variance_on_the_diagonal_ ? 0. : GradientCovFct_(cm, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
									ind_par, i, i, GetDistanceForGradientCovFct_(i, i, dist, coords_ptr, coords_pred_ptr), sigma, coords_ptr, coords_pred_ptr);
							}
							else if (i < j) {
								const double dist_ij = GetDistanceForGradientCovFct_(i, j, dist, coords_ptr, coords_pred_ptr);
								it.valueRef() = GradientCovFct_(cm, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
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
							it.valueRef() = GradientCovFct_(cm, par_aux, shape, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
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

		double CalculateGradientCovarianceOneEntryFull(const vec_t& coords,
			const vec_t& coords_pred,
			const vec_t& pars,
			bool transf_scale,
			double nugget_var,
			int ind_par) const {
			CHECK(ind_par >= 0 && ind_par < num_cov_par_);
			if (is_ar1_multifidelity_) {
				den_mat_t coords_mat(1, coords.size());
				den_mat_t coords_pred_mat(1, coords_pred.size());
				coords_mat.row(0) = coords.transpose();
				coords_pred_mat.row(0) = coords_pred.transpose();
				T_mat dist_mat(1, 1);
				dist_mat.coeffRef(0, 0) = 0.;
				T_mat sigma, sigma_grad;
				CalculateCovMat(dist_mat, coords_mat, coords_pred_mat, pars, sigma, false);
				CalculateGradientCovMatFull(dist_mat, coords_mat, coords_pred_mat, sigma, pars, sigma_grad,
					transf_scale, nugget_var, ind_par, false);
				return sigma_grad.coeff(0, 0);
			}
			if (ind_par == 0) {
				const double covariance = CalculateCovarianceOneEntry(0., coords, coords_pred, pars);
				return transf_scale ? covariance : covariance / pars[0];
			}
			return CalculateGradientCovarianceOneEntry(0., coords, coords_pred, pars,
				transf_scale, nugget_var, ind_par - 1);
		}

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
			if (is_ar1_multifidelity_) {
				CHECK(coords.cols() == dim_coordinates_);
				const den_mat_t spatial_coords = coords.leftCols(dim_coordinates_ - 1);
				vec_t pars_low(num_cov_par_base_);
				vec_t pars_discrepancy(num_cov_par_base_);
				base_cov_function_low_->FindInitCovPar(dist, spatial_coords, false, rng, pars_low, marginal_variance / 2.);
				base_cov_function_discrepancy_->FindInitCovPar(dist, spatial_coords, false, rng, pars_discrepancy, marginal_variance / 2.);
				pars.segment(0, num_cov_par_base_) = pars_low;
				pars.segment(num_cov_par_base_, num_cov_par_base_) = pars_discrepancy;
				pars[2 * num_cov_par_base_] = std::exp(1.);// q_rho = exp(rho), initialized at rho = 1
				return;
			}
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
				else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_space_time_estimate_shape" ||
					cov_fct_type_ == "space_time_gneiting") {
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
						Log::REFatal(("Cannot find an initial value for the range parameter "
							"since both the median and the average distances among coordinates are zero " + add_error_str).c_str());
					}
					if (med_dist_time < EPSILON_NUMBERS) {
						Log::REFatal(("Cannot find an initial value for the temporal range parameter "
							"since both the median and the average distances among time points are zero " + add_error_str).c_str());
					}
				}//end cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_space_time_estimate_shape" || cov_fct_type_ == "space_time_gneiting"
				else if (cov_fct_type_ == "matern_ard" || cov_fct_type_ == "gaussian_ard" || cov_fct_type_ == "matern_ard_estimate_shape") {
					med_dist_per_coord = std::vector<double>((int)coords.cols());
					std::vector<bool> feature_is_constant((int)coords.cols(), false);
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						vec_t col_i = coords.col(ic);
						int num_unique_values = GPBoost::NumberUniqueValues(col_i, 11);
						if (num_unique_values == 1) {
							feature_is_constant[ic] = true;
						}
						else if (num_unique_values <= 10) {
							// average distance among two random points on {1,...,num_unique_values} times the average spacing of the values,
							//	so that the scale does not depend on how the feature is coded
							med_dist_per_coord[ic] = (num_unique_values * num_unique_values - 1) / 3. / num_unique_values *
								(col_i.maxCoeff() - col_i.minCoeff()) / (num_unique_values - 1);
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
							// the thresholds are relative to the extent of the feature, so that they do not depend on its units
							const double tol_dist = EPSILON_NUMBERS * (col_i.maxCoeff() - col_i.minCoeff());
							med_dist_coord_i = GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(distances);
							if (med_dist_coord_i <= tol_dist) {
								med_dist_coord_i = GPBoost::CalculateMean<std::vector<double>>(distances);
							}
							if (med_dist_coord_i <= tol_dist) {
								// The feature is (almost) constant on the sub-sample but not on all data: the average distance among
								//	all pairs of points is calculated from the sorted values. The gap between the sorted values k - 1
								//	and k occurs in k (n - k) pairs. A sum of these non-negative terms has no cancellations
								std::vector<double> values_sorted(col_i.data(), col_i.data() + col_i.size());
								std::sort(values_sorted.begin(), values_sorted.end());
								const double num_values = (double)values_sorted.size();
								double sum_dist = 0.;
								for (int k = 1; k < (int)values_sorted.size(); ++k) {
									sum_dist += (double)k * (num_values - (double)k) * (values_sorted[k] - values_sorted[k - 1]);
								}
								med_dist_coord_i = 2. * sum_dist / (num_values * (num_values - 1.));
							}
							med_dist_per_coord[ic] = med_dist_coord_i;
						}// end num_unique_values > 10
					}// end loop over features
					const int dim_coords = (int)coords.cols();
					std::vector<int> constant_features;
					double max_med_dist_non_constant = 0.;
					for (int ic = 0; ic < dim_coords; ++ic) {
						if (feature_is_constant[ic]) {
							constant_features.push_back(ic);
						}
						else {
							max_med_dist_non_constant = std::max(max_med_dist_non_constant, med_dist_per_coord[ic]);
						}
					}
					if ((int)constant_features.size() == dim_coords) {
						Log::REFatal("Cannot find initial values for the range parameters since all input features are constant ");
					}
					if (!constant_features.empty()) {
						string_t features_str;
						for (int ic : constant_features) {
							features_str += (features_str.empty() ? "" : ", ") + std::to_string(ic + 1);
						}
						Log::REWarning(("The input feature(s) number " + features_str + " (counting starts at 1) are constant and carry no information. "
							"They are treated as irrelevant: their range parameters are set to very large values, which the estimation does not change ").c_str());
					}
					// The median distances per coordinate are multiplied by the median distance in the coordinates scaled by them.
					//	Otherwise, the distances in the scaled coordinates grow with the number of coordinates, and the initial
					//	correlations are almost zero in higher dimensions. For coordinates with equal median distances, this gives
					//	the initial range of the isotropic covariance function
					vec_t med_dist_per_coord_inv(dim_coords);
					for (int ic = 0; ic < dim_coords; ++ic) {
						med_dist_per_coord_inv[ic] = feature_is_constant[ic] ? 0. : 1. / med_dist_per_coord[ic];
					}
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (num_data_find_init - 1); ++i) {
						const int ii = use_subsamples ? sample_ind[i] : i;
						for (int j = i + 1; j < num_data_find_init; ++j) {
							const int jj = use_subsamples ? sample_ind[j] : j;
							distances[i * (2 * num_data_find_init - i - 1) / 2 + j - (i + 1)] =
								(coords.row(ii) - coords.row(jj)).cwiseProduct(med_dist_per_coord_inv.transpose()).norm();
						}
					}
					double med_dist_scaled = GPBoost::CalculateMedianPartiallySortInput<std::vector<double>>(distances);
					if (med_dist_scaled < EPSILON_NUMBERS) {
						med_dist_scaled = GPBoost::CalculateMean<std::vector<double>>(distances);
					}
					if (med_dist_scaled >= EPSILON_NUMBERS) {
						for (int ic = 0; ic < dim_coords; ++ic) {
							med_dist_per_coord[ic] *= med_dist_scaled;
						}
						max_med_dist_non_constant *= med_dist_scaled;
					}
					for (int ic : constant_features) {
						// The scale of a constant feature is unknown. Its range is so large relative to the scales of all other features
						//	that the feature is irrelevant also for predictions at other values of it, unless the units of the features
						//	differ by many orders of magnitude
						med_dist_per_coord[ic] = 1e30 * max_med_dist_non_constant;
					}
				}//end cov_fct_type_ == "matern_ard" && cov_fct_type_ == "gaussian_ard" || cov_fct_type_ == "matern_ard_estimate_shape"
				if (cov_fct_type_ == "matern") {
					pars[1] = MaternInitialTransformedRange(med_dist);
				}
				else if (cov_fct_type_ == "matern_space_time") {
					pars[1] = MaternInitialTransformedRange(med_dist_time);
					pars[2] = MaternInitialTransformedRange(med_dist_space);
				}
				else if (cov_fct_type_ == "matern_ard") {
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						pars[1 + ic] = MaternInitialTransformedRange(med_dist_per_coord[ic]);
					}
				}
				else if (cov_fct_type_ == "matern_estimate_shape") {
					pars[1] = MaternEstimateShapeInitialRange(med_dist);
					pars[2] = 1.5;
				}
				else if (cov_fct_type_ == "matern_space_time_estimate_shape") {
					pars[1] = MaternEstimateShapeInitialRange(med_dist_time);
					pars[2] = MaternEstimateShapeInitialRange(med_dist_space);
					pars[3] = 1.5;
				}
				else if (cov_fct_type_ == "matern_ard_estimate_shape") {
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						pars[1 + ic] = MaternEstimateShapeInitialRange(med_dist_per_coord[ic]);
					}
					pars[num_cov_par_ - 1] = 1.5;
				}
				else if (cov_fct_type_ == "gaussian") {
					pars[1] = 3. / std::pow(med_dist / 2., 2.);
				}
				else if (cov_fct_type_ == "powered_exponential") {
					pars[1] = 3. / std::pow(med_dist / 2., shape_);
				}
				else if (cov_fct_type_ == "space_time_gneiting") {//pars Parameter in the following order : sigma2, a, c, alpha, nu, beta, delta
					int dim_space = (int)coords.cols() - 1;
					pars[3] = 0.5;//alpha, in the interior since the temporal covariance is analytic for alpha = 1, which can make the covariance matrix almost singular
					pars[1] = (std::pow(20., 2. / dim_space) - 1.) / std::pow(med_dist_time / 2., 2. * pars[3]);//a, temporal range such that 1 + a * |u|^(2 * alpha) = 20^(2 / d) at half the median distance
					pars[2] = 2. * 4.7 / med_dist_space;//c, range such that correlation at 0.05 at half the median distance
					pars[4] = 1.5;//nu -> matern 1.5
					pars[5] = 1.;//beta
					pars[6] = 1;//delta
				}
				else if (cov_fct_type_ == "gaussian_ard") {
					for (int ic = 0; ic < (int)coords.cols(); ++ic) {
						pars[1 + ic] = 3. / std::pow(med_dist_per_coord[ic] / 2., 2.);
					}
				}
				else if (cov_fct_type_ == "hurst" || cov_fct_type_ == "hurst_ard") {
					pars[1] = -std::log(0.5);// H = m - 1/2: Brownian motion for m = 1 and an integrated Wiener process for m = 2
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

		void InitializeAR1Multifidelity(const string_t& base_cov_fct_type,
			const string_t& cov_fct_type,
			double shape,
			int order,
			double taper_range,
			double taper_shape,
			double taper_mu,
			bool apply_tapering,
			int dim_coordinates) {
			if (dim_coordinates < 2) {
				Log::REFatal("Covariance function '%s' requires at least one spatial coordinate and a fidelity indicator in the last coordinate column",
					cov_fct_type.c_str());
			}
			if (base_cov_fct_type.empty() || base_cov_fct_type.rfind("ar1_mf_", 0) == 0) {
				Log::REFatal("Invalid base covariance function in '%s'", cov_fct_type.c_str());
			}
			if (base_cov_fct_type == "wendland") {
				Log::REFatal("Covariance function '%s' is currently not supported with a Wendland base covariance", cov_fct_type.c_str());
			}
			if (apply_tapering) {
				Log::REFatal("Correlation tapering is currently not supported for covariance function '%s'", cov_fct_type.c_str());
			}
			is_ar1_multifidelity_ = true;
			cov_fct_type_ = cov_fct_type;
			base_cov_fct_type_ = base_cov_fct_type;
			dim_coordinates_ = dim_coordinates;
			shape_ = shape;
			order_ = order;
			use_precomputed_dist_for_calc_cov_ = false;
			is_isotropic_ = false;
			use_scaled_coordinates_ = false;
			redetermine_vecchia_neighbors_in_transformed_space_ = false;
			cov_calculated_based_on_coords_ = true;
			need_coordinates_for_calculating_covariance_ = true;
			variance_on_the_diagonal_ = false;
			const int spatial_dim = dim_coordinates - 1;
			base_cov_function_low_ = std::make_shared<CovFunction<T_mat>>(base_cov_fct_type, shape, order,
				taper_range, taper_shape, taper_mu, false, spatial_dim, false);
			base_cov_function_discrepancy_ = std::make_shared<CovFunction<T_mat>>(base_cov_fct_type, shape, order,
				taper_range, taper_shape, taper_mu, false, spatial_dim, false);
			base_cov_fct_type_ = base_cov_function_low_->cov_fct_type_;//store the canonicalized alias
			num_cov_par_base_ = base_cov_function_low_->num_cov_par_;
			num_cov_par_ = 2 * num_cov_par_base_ + 1;
			(void)taper_range;
			(void)taper_shape;
			(void)taper_mu;
		}

		void ValidateAndSplitAR1Coordinates(const den_mat_t& coords,
			den_mat_t& spatial_coords,
			vec_t& fidelity) const {
			CHECK(coords.cols() == dim_coordinates_);
			spatial_coords = coords.leftCols(dim_coordinates_ - 1);
			fidelity = coords.col(dim_coordinates_ - 1);
			for (int i = 0; i < fidelity.size(); ++i) {
				if (!(TwoNumbersAreEqual<double>(fidelity[i], 0.) || TwoNumbersAreEqual<double>(fidelity[i], 1.))) {
					Log::REFatal("The last coordinate column for covariance function '%s' must contain only 0 (low fidelity) and 1 (high fidelity), found %g",
						cov_fct_type_.c_str(), fidelity[i]);
				}
			}
		}

		void ValidateAndSplitAR1Coordinates(const vec_t& coords,
			vec_t& spatial_coords,
			double& fidelity) const {
			CHECK(coords.size() == dim_coordinates_);
			spatial_coords = coords.head(dim_coordinates_ - 1);
			fidelity = coords[dim_coordinates_ - 1];
			if (!(TwoNumbersAreEqual<double>(fidelity, 0.) || TwoNumbersAreEqual<double>(fidelity, 1.))) {
				Log::REFatal("The last coordinate entry for covariance function '%s' must be 0 (low fidelity) or 1 (high fidelity), found %g",
					cov_fct_type_.c_str(), fidelity);
			}
		}

		template <class T_aux = T_mat, typename std::enable_if<std::is_same<den_mat_t, T_aux>::value>::type* = nullptr>
		void CombineAR1CovarianceMatrices(const T_mat& cov_low,
			const T_mat& cov_discrepancy,
			const vec_t& fidelity,
			const vec_t& fidelity_pred,
			double rho,
			T_mat& sigma) const {
			const vec_t loading = vec_t::Ones(fidelity.size()) + fidelity * (rho - 1.);
			const vec_t loading_pred = vec_t::Ones(fidelity_pred.size()) + fidelity_pred * (rho - 1.);
			sigma = cov_low;
#pragma omp parallel for schedule(static)
			for (int i = 0; i < sigma.rows(); ++i) {
				for (int j = 0; j < sigma.cols(); ++j) {
					sigma(i, j) = loading_pred[i] * loading[j] * cov_low(i, j) +
						fidelity_pred[i] * fidelity[j] * cov_discrepancy(i, j);
				}
			}
		}

		template <class T_aux = T_mat, typename std::enable_if<std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr>
		void CombineAR1CovarianceMatrices(const T_mat& cov_low,
			const T_mat& cov_discrepancy,
			const vec_t& fidelity,
			const vec_t& fidelity_pred,
			double rho,
			T_mat& sigma) const {
			const vec_t loading = vec_t::Ones(fidelity.size()) + fidelity * (rho - 1.);
			const vec_t loading_pred = vec_t::Ones(fidelity_pred.size()) + fidelity_pred * (rho - 1.);
			sigma = cov_low + cov_discrepancy;//use the union of both sparse patterns
			for (int k = 0; k < sigma.outerSize(); ++k) {
				for (typename T_mat::InnerIterator it(sigma, k); it; ++it) {
					const int i = (int)it.row();
					const int j = (int)it.col();
					it.valueRef() = loading_pred[i] * loading[j] * cov_low.coeff(i, j) +
						fidelity_pred[i] * fidelity[j] * cov_discrepancy.coeff(i, j);
				}
			}
		}

		void CalculateCovMatAR1Multifidelity(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const vec_t& pars,
			T_mat& sigma,
			bool is_symmetric) const {
			den_mat_t spatial_coords, spatial_coords_pred;
			vec_t fidelity, fidelity_pred;
			ValidateAndSplitAR1Coordinates(coords, spatial_coords, fidelity);
			if (is_symmetric) {
				spatial_coords_pred = spatial_coords;
				fidelity_pred = fidelity;
			}
			else {
				ValidateAndSplitAR1Coordinates(coords_pred, spatial_coords_pred, fidelity_pred);
			}
			const vec_t pars_low = pars.segment(0, num_cov_par_base_);
			const vec_t pars_discrepancy = pars.segment(num_cov_par_base_, num_cov_par_base_);
			const double rho = std::log(pars[2 * num_cov_par_base_]);
			T_mat cov_low, cov_discrepancy;
			base_cov_function_low_->CalculateCovMat(dist, spatial_coords, spatial_coords_pred, pars_low, cov_low, is_symmetric);
			base_cov_function_discrepancy_->CalculateCovMat(dist, spatial_coords, spatial_coords_pred,
				pars_discrepancy, cov_discrepancy, is_symmetric);
			CombineAR1CovarianceMatrices(cov_low, cov_discrepancy, fidelity, fidelity_pred, rho, sigma);
		}

		double CalculateCovarianceOneEntryAR1Multifidelity(const vec_t& coords,
			const vec_t& coords_pred,
			const vec_t& pars) const {
			vec_t spatial_coords, spatial_coords_pred;
			double fidelity, fidelity_pred;
			ValidateAndSplitAR1Coordinates(coords, spatial_coords, fidelity);
			ValidateAndSplitAR1Coordinates(coords_pred, spatial_coords_pred, fidelity_pred);
			const double rho = std::log(pars[2 * num_cov_par_base_]);
			const double loading = 1. + fidelity * (rho - 1.);
			const double loading_pred = 1. + fidelity_pred * (rho - 1.);
			const double cov_low = base_cov_function_low_->CalculateCovarianceOneEntry(0., spatial_coords,
				spatial_coords_pred, pars.segment(0, num_cov_par_base_));
			const double cov_discrepancy = base_cov_function_discrepancy_->CalculateCovarianceOneEntry(0.,
				spatial_coords, spatial_coords_pred, pars.segment(num_cov_par_base_, num_cov_par_base_));
			return loading * loading_pred * cov_low + fidelity * fidelity_pred * cov_discrepancy;
		}

		template <class T_aux = T_mat, typename std::enable_if<std::is_same<den_mat_t, T_aux>::value>::type* = nullptr>
		void ScaleAR1GradientMatrix(const T_mat& gradient_base,
			const vec_t& row_scale,
			const vec_t& col_scale,
			T_mat& sigma_grad) const {
			sigma_grad = gradient_base;
#pragma omp parallel for schedule(static)
			for (int i = 0; i < sigma_grad.rows(); ++i) {
				for (int j = 0; j < sigma_grad.cols(); ++j) {
					sigma_grad(i, j) *= row_scale[i] * col_scale[j];
				}
			}
		}

		template <class T_aux = T_mat, typename std::enable_if<std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr>
		void ScaleAR1GradientMatrix(const T_mat& gradient_base,
			const vec_t& row_scale,
			const vec_t& col_scale,
			T_mat& sigma_grad) const {
			sigma_grad = gradient_base;
			for (int k = 0; k < sigma_grad.outerSize(); ++k) {
				for (typename T_mat::InnerIterator it(sigma_grad, k); it; ++it) {
					it.valueRef() *= row_scale[it.row()] * col_scale[it.col()];
				}
			}
		}

		void CalculateGradientCovMatAR1Multifidelity(const T_mat& dist,
			const den_mat_t& coords,
			const den_mat_t& coords_pred,
			const vec_t& pars,
			T_mat& sigma_grad,
			bool transf_scale,
			double nugget_var,
			int ind_par,
			bool is_symmetric) const {
			den_mat_t spatial_coords, spatial_coords_pred;
			vec_t fidelity, fidelity_pred;
			ValidateAndSplitAR1Coordinates(coords, spatial_coords, fidelity);
			if (is_symmetric) {
				spatial_coords_pred = spatial_coords;
				fidelity_pred = fidelity;
			}
			else {
				ValidateAndSplitAR1Coordinates(coords_pred, spatial_coords_pred, fidelity_pred);
			}
			const vec_t pars_low = pars.segment(0, num_cov_par_base_);
			const vec_t pars_discrepancy = pars.segment(num_cov_par_base_, num_cov_par_base_);
			const double rho = std::log(pars[2 * num_cov_par_base_]);
			const vec_t loading = vec_t::Ones(fidelity.size()) + fidelity * (rho - 1.);
			const vec_t loading_pred = vec_t::Ones(fidelity_pred.size()) + fidelity_pred * (rho - 1.);
			T_mat cov_low, cov_discrepancy;
			base_cov_function_low_->CalculateCovMat(dist, spatial_coords, spatial_coords_pred, pars_low, cov_low, is_symmetric);
			base_cov_function_discrepancy_->CalculateCovMat(dist, spatial_coords, spatial_coords_pred,
				pars_discrepancy, cov_discrepancy, is_symmetric);
			if (ind_par < num_cov_par_base_) {
				T_mat gradient_low;
				base_cov_function_low_->CalculateGradientCovMatFull(dist, spatial_coords, spatial_coords_pred,
					cov_low, pars_low, gradient_low, transf_scale, nugget_var, ind_par, is_symmetric);
				ScaleAR1GradientMatrix(gradient_low, loading_pred, loading, sigma_grad);
			}
			else if (ind_par < 2 * num_cov_par_base_) {
				T_mat gradient_discrepancy;
				base_cov_function_discrepancy_->CalculateGradientCovMatFull(dist, spatial_coords, spatial_coords_pred,
					cov_discrepancy, pars_discrepancy, gradient_discrepancy, transf_scale, nugget_var,
					ind_par - num_cov_par_base_, is_symmetric);
				ScaleAR1GradientMatrix(gradient_discrepancy, fidelity_pred, fidelity, sigma_grad);
			}
			else {
				const vec_t row_scale = fidelity_pred;
				const vec_t col_scale = loading;
				ScaleAR1GradientMatrix(cov_low, row_scale, col_scale, sigma_grad);
				T_mat second_term;
				ScaleAR1GradientMatrix(cov_low, loading_pred, fidelity, second_term);
				sigma_grad += second_term;
				if (!transf_scale) {
					sigma_grad *= nugget_var;
				}
			}
		}

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
				GetDistanceForCovFct_ = [](const int i, const int j, const T_mat& dist,
					const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return(dist.coeff(i, j));
					};
			}
			else {
				GetDistanceForCovFct_ = [](const int i, const int j, const T_mat& /* dist */,
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
			if (IsMaternWithFixedShape()) {
				if (matern_half_integer_order_ == 0) {
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape0_5(dist_ij, var, range);
						};
				}
				else if (matern_half_integer_order_ == 1) {
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape1_5(dist_ij, var, range);
						};
				}
				else if (matern_half_integer_order_ == 2) {
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternShape2_5(dist_ij, var, range);
						};
				}
				else if (matern_half_integer_order_ > 2) {
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternHalfIntegerShape(dist_ij, var, range);
						};
				}
				else {//general shape
					CovFct_dist_ = [this](double dist_ij, double var, double range, double /* shape */) -> double {
						return CovarianceMaternGeneralShape(dist_ij, var, range);
						};
				}
			}//end IsMaternWithFixedShape()
			else if (IsMaternWithEstimatedShape()) {
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

		/*! \brief Matern covariance with a smoothness p + 1/2 for p >= 3 */
		inline double CovarianceMaternHalfIntegerShape(double dist,
			double var,
			double range) const {
			double range_dist = range * dist;
			return(var * EvaluatePolynomial(matern_poly_coef_, range_dist) * std::exp(-range_dist));
		}

		/*!
		* \brief Matern correlation M_nu(z) = 2^(1 - nu) / Gamma(nu) * z^nu * K_nu(z) for a general smoothness nu > 0 and z > 0.
		*		For nu > 2, it is calculated with the recurrence M_{nu+1}(z) = M_nu(z) + z^2 / (4 nu (nu - 1)) * M_{nu-1}(z) from two
		*		orders in (0, 2]: its terms are positive and at most 1, whereas z^nu underflows and K_nu(z) overflows for large nu
		*		and small z, and Gamma(nu) overflows for nu > 171. Bessel functions are thus only evaluated for orders in (0, 2]
		*/
		inline double MaternCorrelationGeneralShape(double z,
			double nu) const {
			if (nu <= 2.) {
				return(std::pow(2., 1. - nu) / std::tgamma(nu) * std::pow(z, nu) * BesselK(nu, z));
			}
			const int num_steps = (int)std::ceil(nu) - 2;
			double order = nu - num_steps;// in (1, 2]
			double corr_lower = MaternCorrelationGeneralShape(z, order - 1.);
			double corr = MaternCorrelationGeneralShape(z, order);
			const double z_sq_quarter = z * z / 4.;
			for (int step = 0; step < num_steps; ++step) {
				const double corr_upper = corr + z_sq_quarter / (order * (order - 1.)) * corr_lower;
				corr_lower = corr;
				corr = corr_upper;
				order += 1.;
			}
			return(corr);
		}//end MaternCorrelationGeneralShape

		/*!
		* \brief Derivative of the Matern correlation M_nu(z) with respect to z (z > 0): -z / (2 (nu - 1)) * M_{nu-1}(z) for nu > 1,
		*		and -2^(1 - nu) / Gamma(nu) * z^nu * K_{1-nu}(z) for nu <= 1
		*/
		inline double MaternCorrelationDerivativeGeneralShape(double z,
			double nu) const {
			if (nu > 1.) {
				return(-z / (2. * (nu - 1.)) * MaternCorrelationGeneralShape(z, nu - 1.));
			}
			return(-std::pow(2., 1. - nu) / std::tgamma(nu) * std::pow(z, nu) * BesselK(1. - nu, z));
		}//end MaternCorrelationDerivativeGeneralShape

		inline double CovarianceMaternGeneralShape(double dist,
			double var,
			double range) const {
			double range_dist = range * dist;
			if (range_dist <= 0.) {
				return(var);
			}
			return(var * MaternCorrelationGeneralShape(range_dist, shape_));
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
			return(var * MaternCorrelationGeneralShape(range_dist, shape));
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
			double& par_aux,
			double& pars_2_up,
			double& pars_2_down,
			double& par_aux_up,
			double& par_aux_down,
			double& shape) const {
			if (cov_fct_type_ == "matern") {
				if (matern_half_integer_order_ == 0) {
					cm = transf_scale ? (-1. * pars[1]) : (nugget_var * pars[1] * pars[1]);
				}
				else if (matern_half_integer_order_ > 0) {
					cm = transf_scale ? (-1. * pars[0] * pars[1] * pars[1]) : (nugget_var * pars[0] * std::pow(pars[1], 3) / MaternRangeScale());
				}
				else {//general shape
					cm = transf_scale ? 1. : (-nugget_var * pars[1] / MaternRangeScale());
					cm *= pars[0];
				}
				par_aux = pars[1];
			}
			else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
				CHECK(ind_par >= 0 && ind_par < num_cov_par_ - 1);
				if (matern_half_integer_order_ == 0) {
					cm = transf_scale ? -1. : (nugget_var * pars[ind_par + 1]);
				}
				else if (matern_half_integer_order_ == 2) {
					// 1 / 3 = 1 / (2p - 1), which is part of 'matern_poly_coef_deriv_' for p > 2
					cm = transf_scale ? (-1. / 3. * pars[0]) : (nugget_var / 3. * pars[0] * pars[ind_par + 1] / MaternRangeScale());
				}
				else if (matern_half_integer_order_ > 0) {
					cm = transf_scale ? (-1. * pars[0]) : (nugget_var * pars[0] * pars[ind_par + 1] / MaternRangeScale());
				}
				else {//general shape
					cm = transf_scale ? 1. : (-nugget_var * pars[ind_par + 1] / MaternRangeScale());
					cm *= pars[0];
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
					cm *= -pars[0];
					par_aux = std::sqrt(2. * pars[2]) / pars[1];
				}
				else if (ind_par == 1) {//gradient wrt smoothness parameter
					cm = transf_scale ? pars[0] : nugget_var * pars[0];
					ShapesForFiniteDifference(pars[2], transf_scale, pars_2_up, pars_2_down);
					par_aux_up = std::sqrt(2. * pars_2_up) / pars[1];
					par_aux_down = std::sqrt(2. * pars_2_down) / pars[1];
				}
				shape = pars[2];
			}// end cov_fct_type_ == "matern_estimate_shape"
			else if (cov_fct_type_ == "matern_ard_estimate_shape" || cov_fct_type_ == "matern_space_time_estimate_shape") {
				int ind_shape = num_cov_par_ - 1;
				CHECK(ind_par >= 0 && ind_par <= num_cov_par_ - 2);
				if (ind_par < num_cov_par_ - 2) {//gradient wrt range parameters
					cm = transf_scale ? 1. : nugget_var / pars[ind_par + 1];
					cm *= -pars[0] * 2. * pars[ind_shape];
					par_aux = std::sqrt(2. * pars[ind_shape]);
				}
				else if (ind_par == num_cov_par_ - 2) {//gradient wrt smoothness parameter
					cm = transf_scale ? pars[0] : nugget_var * pars[0];
					ShapesForFiniteDifference(pars[ind_shape], transf_scale, pars_2_up, pars_2_down);
					par_aux_up = std::sqrt(2. * pars_2_up);
					par_aux_down = std::sqrt(2. * pars_2_down);
				}
				shape = pars[ind_shape];
			}// end cov_fct_type_ == "matern_ard_estimate_shape" || cov_fct_type_ == "matern_space_time_estimate_shape"
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
				GetDistanceForGradientCovFct_ = [](const int i, const int j, const T_mat& dist,
					const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) {
						return static_cast<double>(dist.coeff(i, j));
					};
			}
			else {
				GetDistanceForGradientCovFct_ = [](const int i, const int j, const T_mat& /* dist */,
					const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) {
						return static_cast<double>(((*coords_pred_ptr).row(i) - (*coords_ptr).row(j)).lpNorm<2>());
					};
			}
		}//end InitializeGetDistanceForGradientCovFct

		/*!
		* \brief Generic function for calculating gradients of covariances wrt range and other parameters such as smoothness based on distances (cov_calculated_based_on_coords_ == false)
		*/
		std::function<double(double /* cm */, double /* par_aux */, double /* shape */,
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
				GradientCovFct_ = [this](double cm, double par_aux, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int /* ind_par */, const int i, const int j,
					const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientRangeMatern(cm, dist_ij, par_aux, sigma, i, j);
					};
				grad_is_multiple_of_cov_ = (matern_half_integer_order_ == 0);
			}
			else if (cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard") {
				const bool space_time = cov_fct_type_ == "matern_space_time";
				GradientCovFct_ = [this, space_time](double cm, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int ind_par, const int i, const int j,
					const double dist_ij, const T_mat& sigma, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
						const double dist_sq_coord = SquaredScaledCoordDifference(space_time, ind_par, i, j, coords_ptr, coords_pred_ptr);
						return GradientRangeMaternScaledCoords(cm, dist_ij, dist_sq_coord, sigma, i, j);
					};
				grad_is_multiple_of_cov_ = (matern_half_integer_order_ == 0);
			}
			else if (cov_fct_type_ == "gaussian") {
				GradientCovFct_ = [this](double cm, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int /* ind_par */, const int i, const int j,
					const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientRangeGaussian(cm, dist_ij, sigma, i, j);
					};
				grad_is_multiple_of_cov_ = true;
			}
			else if (cov_fct_type_ == "powered_exponential") {
				GradientCovFct_ = [this](double cm, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int /* ind_par */, const int i, const int j,
					const double dist_ij, const T_mat& sigma, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientRangePoweredExponential(cm, dist_ij, sigma, i, j);
					};
				grad_is_multiple_of_cov_ = true;
			}
			else if (cov_fct_type_ == "matern_estimate_shape") {
				GradientCovFct_ = [this](double cm, double par_aux, double shape,
					double par_aux_up, double par_aux_down, double pars_2_up, double pars_2_down,
					const int ind_par, const int /* i */, const int /* j */,
					const double dist_ij, const T_mat& /* sigma */, const den_mat_t* /* coords_ptr */, const den_mat_t* /* coords_pred_ptr */) -> double {
						return GradientMaternEstimateShape(cm, dist_ij, par_aux, par_aux_up, par_aux_down, pars_2_up, pars_2_down, shape, ind_par);
					};
			}
			else if (cov_fct_type_ == "matern_ard_estimate_shape" || cov_fct_type_ == "matern_space_time_estimate_shape") {
				const bool space_time = cov_fct_type_ == "matern_space_time_estimate_shape";
				GradientCovFct_ = [this, space_time](double cm, double par_aux, double shape,
					double par_aux_up, double par_aux_down, double pars_2_up, double pars_2_down,
					const int ind_par, const int i, const int j,
					const double dist_ij, const T_mat& /* sigma */, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
						return GradientMaternScaledCoordsEstimateShape(cm, dist_ij, par_aux, par_aux_up, par_aux_down, pars_2_up, pars_2_down,
							shape, ind_par, space_time, i, j, coords_ptr, coords_pred_ptr);
					};
			}
			else if (cov_fct_type_ == "gaussian_ard") {
				GradientCovFct_ = [this](double cm, double /* par_aux */, double /* shape */,
					double /* par_aux_up */, double /* par_aux_down */, double /* pars_2_up */, double /* pars_2_down */,
					const int ind_par, const int i, const int j,
					const double /* dist_ij */, const T_mat& sigma, const den_mat_t* coords_ptr, const den_mat_t* coords_pred_ptr) -> double {
						return GradientRangeGaussianARD(cm, sigma, ind_par, i, j, coords_ptr, coords_pred_ptr);
					};
				grad_is_multiple_of_cov_ = true;
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

		/*!
		* \brief Gradient of the 'matern' covariance function with respect to the range parameter
		* \param cm Constant from 'DetermineConstantsForGradient', which contains the transformed range parameter
		* \param dist_ij Distance
		* \param range Transformed range parameter
		* \param sigma Covariance matrix
		*/
		inline double GradientRangeMatern(double cm,
			const double dist_ij,
			double range,
			const T_mat& sigma,
			const int i,
			const int j) const {
			if (matern_half_integer_order_ == 0) {
				return(cm * dist_ij * sigma.coeff(i, j));
			}
			else if (matern_half_integer_order_ == 1) {
				return(cm * dist_ij * dist_ij * std::exp(-range * dist_ij));
			}
			else if (matern_half_integer_order_ == 2) {
				double range_dist = range * dist_ij;
				return(cm / 3. * dist_ij * dist_ij * (1. + range_dist) * std::exp(-range_dist));
			}
			else if (matern_half_integer_order_ > 2) {
				double range_dist = range * dist_ij;
				return(cm * dist_ij * dist_ij * EvaluatePolynomial(matern_poly_coef_deriv_, range_dist) * std::exp(-range_dist));
			}
			return(GradientRangeMaternGeneralShape(cm, dist_ij, range, shape_));
		}//end GradientRangeMatern

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

		/*!
		* \brief Gradient of the 'matern' covariance function with a general smoothness and of the 'matern_estimate_shape' covariance
		*		function with respect to the range parameter: cm * z * dM_shape(z)/dz with z = cm_dist * dist_ij
		*/
		inline double GradientRangeMaternGeneralShape(double cm,
			const double dist_ij,
			double cm_dist,
			double shape) const {
			double range_dist = cm_dist * dist_ij;
			if (range_dist <= 0.) {
				return(0.);// limit at distance zero, where the derivative of the correlation can be infinite
			}
			return(cm * range_dist * MaternCorrelationDerivativeGeneralShape(range_dist, shape));
		}//end GradientRangeMaternGeneralShape

		/*!
		* \brief Gradient of a Matern covariance function with an estimated smoothness with respect to the smoothness: cm times the central
		*		difference of the correlation M_shape(z) at shape_up and shape_down, where z = dist_ij * par_aux also depends on the smoothness
		* \param par_aux_up Factor of dist_ij in z for shape_up
		* \param par_aux_down Factor of dist_ij in z for shape_down
		*/
		inline double GradientSmoothnessMaternEstimateShapeFiniteDifference(double cm,
			double dist_ij,
			double par_aux_up,
			double par_aux_down,
			double shape_up,
			double shape_down) const {
			if (dist_ij <= 0.) {
				return(0.);// the correlation is 1 at distance zero for every smoothness
			}
			return(cm * (MaternCorrelationGeneralShape(dist_ij * par_aux_up, shape_up) -
				MaternCorrelationGeneralShape(dist_ij * par_aux_down, shape_down)) / (2. * delta_step_));
		}//end GradientSmoothnessMaternEstimateShapeFiniteDifference

		inline double GradientMaternEstimateShape(double cm,
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
				return(GradientSmoothnessMaternEstimateShapeFiniteDifference(cm, dist_ij, par_aux_up, par_aux_down, pars_2_up, pars_2_down));
			}
			return(1.);
		}//end GradientMaternEstimateShape

		/*!
		* \brief Gradient of the 'matern_ard_estimate_shape' and 'matern_space_time_estimate_shape' covariance functions with
		*		respect to a range parameter (ind_par < num_cov_par_ - 2) or the smoothness parameter (ind_par = num_cov_par_ - 2)
		* \param space_time If true, the covariance function is 'matern_space_time_estimate_shape'
		*/
		inline double GradientMaternScaledCoordsEstimateShape(double cm,
			const double dist_ij,
			double par_aux,
			double par_aux_up,
			double par_aux_down,
			double pars_2_up,
			double pars_2_down,
			double shape,
			const int ind_par,
			bool space_time,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			if (ind_par < num_cov_par_ - 2) {
				const double dist_sq_coord = SquaredScaledCoordDifference(space_time, ind_par, i, j, coords_ptr, coords_pred_ptr);
				return(GradientRangeMaternScaledCoordsGeneralShape(cm, dist_ij, par_aux, dist_sq_coord, shape));
			}
			else if (ind_par == num_cov_par_ - 2) {//gradient wrt smoothness parameter
				return(GradientSmoothnessMaternEstimateShapeFiniteDifference(cm, dist_ij, par_aux_up, par_aux_down, pars_2_up, pars_2_down));
			}
			return(0.);
		}//end GradientMaternScaledCoordsEstimateShape

		/*!
		* \brief Squared difference of two points in the scaled coordinates that belong to a range parameter of an ARD or space-time
		*		Matern covariance function: the coordinate 'ind_range' for an ARD covariance function, and the time coordinate
		*		(ind_range = 0) or the space coordinates (ind_range = 1) for a space-time covariance function
		* \param space_time If true, the covariance function is a space-time covariance function
		* \param ind_range Number of the range parameter
		* \param i Row index for coords_pred_ptr
		* \param j Row index for coords_ptr
		*/
		inline double SquaredScaledCoordDifference(bool space_time,
			const int ind_range,
			const int i,
			const int j,
			const den_mat_t* coords_ptr,
			const den_mat_t* coords_pred_ptr) const {
			if (space_time && ind_range == 1) {
				int dim_space = (int)(*coords_ptr).cols() - 1;
				return(((*coords_pred_ptr).row(i).tail(dim_space) - (*coords_ptr).row(j).tail(dim_space)).squaredNorm());
			}
			double diff = (*coords_pred_ptr).coeff(i, ind_range) - (*coords_ptr).coeff(j, ind_range);
			return(diff * diff);
		}

		/*!
		* \brief Gradient of the 'matern_ard' and 'matern_space_time' covariance functions with respect to a range parameter
		* \param cm Constant from 'DetermineConstantsForGradient'
		* \param dist_ij Distance between the scaled coordinates
		* \param dist_sq_coord Squared difference of the scaled coordinates that belong to the range parameter, see 'SquaredScaledCoordDifference'
		* \param sigma Covariance matrix
		*/
		inline double GradientRangeMaternScaledCoords(double cm,
			const double dist_ij,
			const double dist_sq_coord,
			const T_mat& sigma,
			const int i,
			const int j) const {
			if (matern_half_integer_order_ == 0) {
				if (dist_sq_coord < EPSILON_NUMBERS) {
					return(0.);
				}
				return(cm * dist_sq_coord / dist_ij * sigma.coeff(i, j));
			}
			else if (matern_half_integer_order_ == 1) {
				return(cm * dist_sq_coord * std::exp(-dist_ij));
			}
			else if (matern_half_integer_order_ == 2) {
				return(cm * dist_sq_coord * (1 + dist_ij) * std::exp(-dist_ij));
			}
			else if (matern_half_integer_order_ > 2) {
				return(cm * dist_sq_coord * EvaluatePolynomial(matern_poly_coef_deriv_, dist_ij) * std::exp(-dist_ij));
			}
			return(GradientRangeMaternScaledCoordsGeneralShape(cm, dist_ij, 1., dist_sq_coord, shape_));
		}//end GradientRangeMaternScaledCoords

		/*!
		* \brief Gradient of an ARD or space-time Matern covariance function with a general smoothness parameter with respect to a range
		*		parameter: cm * dM_shape(z)/dz / z * dist_sq_coord with z = cm_dist * dist_ij
		* \param cm_dist Factor of the distance between the scaled coordinates in the argument z of the correlation
		* \param dist_sq_coord Squared difference of the scaled coordinates that belong to the range parameter, see 'SquaredScaledCoordDifference'
		*/
		inline double GradientRangeMaternScaledCoordsGeneralShape(double cm,
			const double dist_ij,
			double cm_dist,
			const double dist_sq_coord,
			const double shape) const {
			double range_dist = cm_dist * dist_ij;
			if (range_dist <= 0.) {
				return(0.);// limit at distance zero, where the derivative of the correlation can be infinite
			}
			return(cm * MaternCorrelationDerivativeGeneralShape(range_dist, shape) / range_dist * dist_sq_coord);
		}//end GradientRangeMaternScaledCoordsGeneralShape

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
			else if (d_aux < EPSILON_NUMBERS) {
				return(d_aux2);
			}
			return(d_aux2 * MaternCorrelationGeneralShape(d_aux, pars[4]));
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
				else if (d_aux < EPSILON_NUMBERS) {
					grad = d_aux2_grad;
				}
				else {
					grad = d_aux2_grad * MaternCorrelationGeneralShape(d_aux, pars[4]) +
						d_aux2 * MaternCorrelationDerivativeGeneralShape(d_aux, pars[4]) * d_aux_grad;
				}
			}//end a, c, alpha, beta, delta
			else if (ind_par == 3) {//nu
				if (d_aux < EPSILON_NUMBERS) {
					grad = 0.;
				}
				else {
					// central difference of the Matern correlation, on the log scale if transf_scale
					double nu_up, nu_down;
					ShapesForFiniteDifference(pars[4], transf_scale, nu_up, nu_down);
					grad = d_aux2 * (MaternCorrelationGeneralShape(d_aux, nu_up) - MaternCorrelationGeneralShape(d_aux, nu_down)) / (2. * delta_step_);
					cm = transf_scale ? 1. : nugget_var;
				}
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
			return HurstCovariance_vec((*coords).row(j), (*coords_pred).row(i), pars);
		}// end SpaceTimeGneitingCovariance
		template <typename T_x, typename T_y>//T_x and T_y can be vec_t or rows of a den_mat_t (which avoids copying them)
		inline double HurstCovariance_vec(const Eigen::MatrixBase<T_x>& coords_vec,
			const Eigen::MatrixBase<T_y>& coords_pred_vec,
			const vec_t& pars) const {
			const double sqrd_norm_x = coords_vec.squaredNorm();
			const double sqrd_norm_y = coords_pred_vec.squaredNorm();
			const double sqrd_norm_x_min_y = (coords_vec - coords_pred_vec).squaredNorm();
			if (order_ == 2) {
				return HurstCovarianceOrder2(sqrd_norm_x, sqrd_norm_y, sqrd_norm_x_min_y, coords_vec.dot(coords_pred_vec), pars[0], HurstExponent(pars[1]));
			}
			const double H = std::exp(-pars[1]);
			return (pars[0] / 2.) * (std::pow(sqrd_norm_x, H) + std::pow(sqrd_norm_y, H) - std::pow(sqrd_norm_x_min_y, H));
		}// end HurstCovariance_vec

		/*!
		* \brief Hurst exponent H = m - 1 + exp(-p_H) of the order m from the transformed parameter p_H = -log(H - (m - 1)) > 0
		*/
		inline double HurstExponent(double p_H) const {
			return (order_ - 1.) + std::exp(-p_H);
		}

		/*!
		* \brief Second-order (m = 2, 1 < H < 2) Hurst covariance anchored at the origin (X(0) = 0 and grad X(0) = 0):
		*		K(x,y) = sigma2 / (2 (2H - 1)) * B(H), B(H) = u_xy^H - u_x^H - u_y^H + 2 H c (u_x^(H-1) + u_y^(H-1)),
		*		where u_x = ||x||^2, u_y = ||y||^2, u_xy = ||x - y||^2, and c = x^T y. This implies K(x,x) = sigma2 ||x||^(2H)
		*/
		inline double HurstCovarianceOrder2(double u_x,
			double u_y,
			double u_xy,
			double c,
			double sigma2,
			double H) const {
			if (u_x == 0. || u_y == 0.) {
				return 0.;// the process is zero at the anchor, for all parameters
			}
			const double u_x_Hm1 = std::pow(u_x, H - 1.);
			const double u_y_Hm1 = std::pow(u_y, H - 1.);
			const double B = std::pow(u_xy, H) - u_x * u_x_Hm1 - u_y * u_y_Hm1 + 2. * H * c * (u_x_Hm1 + u_y_Hm1);
			return sigma2 * B / (2. * (2. * H - 1.));
		}

		/*!
		* \brief Derivative of the second-order Hurst covariance (see 'HurstCovarianceOrder2') with respect to H,
		*		dK/dH = sigma2 * (B'(H) / (2 (2H - 1)) - B(H) / (2H - 1)^2),
		*		B'(H) = u_xy^H log(u_xy) - u_x^H log(u_x) - u_y^H log(u_y) + 2 c (u_x^(H-1) + u_y^(H-1)) + 2 H c (u_x^(H-1) log(u_x) + u_y^(H-1) log(u_y))
		* \param pars Parameters in the following order: sigma2, p_H (, ranges)
		* \param transf_scale If true, the derivative is taken with respect to log(p_H), where p_H = -log(H - 1), otherwise with respect to H (and multiplied with nugget_var)
		* \param nugget_var Nugget variance
		*/
		inline double GradientHurstCovarianceOrder2WrtH(double u_x,
			double u_y,
			double u_xy,
			double c,
			const vec_t& pars,
			bool transf_scale,
			double nugget_var) const {
			if (u_x == 0. || u_y == 0.) {
				return 0.;// the process is zero at the anchor, for all parameters
			}
			// log(u) is only multiplied with u^a for a > 0 here, and u^a * log(u) tends to 0 for u -> 0
			auto log_or_zero = [](double u) {
				return (u > 0.) ? std::log(u) : 0.;
			};
			const double H_min_m_plus_1 = std::exp(-pars[1]);// H - (m - 1)
			const double H = HurstExponent(pars[1]);
			const double u_x_Hm1 = std::pow(u_x, H - 1.);
			const double u_y_Hm1 = std::pow(u_y, H - 1.);
			const double u_x_H = u_x * u_x_Hm1;
			const double u_y_H = u_y * u_y_Hm1;
			const double u_xy_H = std::pow(u_xy, H);
			const double log_u_x = log_or_zero(u_x);
			const double log_u_y = log_or_zero(u_y);
			const double B = u_xy_H - u_x_H - u_y_H + 2. * H * c * (u_x_Hm1 + u_y_Hm1);
			const double dB_dH = u_xy_H * log_or_zero(u_xy) - u_x_H * log_u_x - u_y_H * log_u_y +
				2. * c * (u_x_Hm1 + u_y_Hm1) + 2. * H * c * (u_x_Hm1 * log_u_x + u_y_Hm1 * log_u_y);
			const double two_H_m1 = 2. * H - 1.;
			const double cm = transf_scale ? -pars[1] * H_min_m_plus_1 : nugget_var;// dH / dlog(p_H) = -p_H * (H - (m - 1))
			return cm * pars[0] * (dB_dH / (2. * two_H_m1) - B / (two_H_m1 * two_H_m1));
		}

		/*!
		* \brief Derivative of the second-order ARD Hurst covariance (see 'HurstCovarianceOrder2') with respect to the range l_k of coordinate k.
		*		All four quantities u_x, u_y, u_xy, and c depend on l_k through the scaled coordinates x_k / l_k and y_k / l_k:
		*		dK/dl_k = -sigma2 H / (l_k (2H - 1)) * [u_xy^(H-1) a_xy - u_x^(H-1) a_x - u_y^(H-1) a_y
		*			+ 2 a_c (u_x^(H-1) + u_y^(H-1)) + 2 (H - 1) c (u_x^(H-2) a_x + u_y^(H-2) a_y)],
		*		where a_x = x_k^2, a_y = y_k^2, a_xy = (x_k - y_k)^2, and a_c = x_k y_k (scaled coordinates)
		* \param coords_vec Scaled coordinates x
		* \param coords_pred_vec Scaled coordinates y
		* \param pars Parameters in the following order: sigma2, p_H, l_2, ..., l_d
		* \param k Coordinate number (k >= 1)
		* \param transf_scale If true, the derivative is taken with respect to log(l_k), otherwise with respect to l_k (and multiplied with nugget_var)
		* \param nugget_var Nugget variance
		*/
		template <typename T_x, typename T_y>//T_x and T_y can be vec_t or rows of a den_mat_t
		inline double GradientHurstCovarianceOrder2WrtRange(const Eigen::MatrixBase<T_x>& coords_vec,
			const Eigen::MatrixBase<T_y>& coords_pred_vec,
			double u_x,
			double u_y,
			double u_xy,
			double c,
			const vec_t& pars,
			int k,
			bool transf_scale,
			double nugget_var) const {
			if (u_x == 0. || u_y == 0.) {
				return 0.;// the process is zero at the anchor, for all parameters
			}
			// u^(H-2) * a for 0 <= a <= u, which tends to 0 for u -> 0 when multiplied with c since |c| <= sqrt(u * u_other) and H > 1.
			//	It is calculated as (a / u) * u^(H-1) to avoid an overflow of u^(H-2) for tiny u
			auto ratio_or_zero = [](double a, double u) {
				return (u > 0.) ? (a / u) : 0.;
			};
			const double H = HurstExponent(pars[1]);
			const double l_k = pars[k + 1];
			const double x_k = coords_vec[k];
			const double y_k = coords_pred_vec[k];
			const double a_x = x_k * x_k;
			const double a_y = y_k * y_k;
			const double a_xy = (x_k - y_k) * (x_k - y_k);
			const double a_c = x_k * y_k;
			const double u_x_Hm1 = std::pow(u_x, H - 1.);
			const double u_y_Hm1 = std::pow(u_y, H - 1.);
			const double bracket = std::pow(u_xy, H - 1.) * a_xy - u_x_Hm1 * a_x - u_y_Hm1 * a_y + 2. * a_c * (u_x_Hm1 + u_y_Hm1) +
				2. * (H - 1.) * c * (ratio_or_zero(a_x, u_x) * u_x_Hm1 + ratio_or_zero(a_y, u_y) * u_y_Hm1);
			const double dK_dl_k = -pars[0] * H / (l_k * (2. * H - 1.)) * bracket;
			const double cm = transf_scale ? l_k : nugget_var;// multiplicative constant to get gradient on log-scale or backtransform with nugget variance
			return cm * dK_dl_k;
		}

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
			return GradientHurstCovariance_vec((*coords).row(j), (*coords_pred).row(i), pars, transf_scale, nugget_var);
		}
		template <typename T_x, typename T_y>//T_x and T_y can be vec_t or rows of a den_mat_t (which avoids copying them)
		inline double GradientHurstCovariance_vec(const Eigen::MatrixBase<T_x>& coords_vec,
			const Eigen::MatrixBase<T_y>& coords_pred_vec,
			const vec_t& pars,
			bool transf_scale,
			double nugget_var) const {
			const double sqrd_norm_x = coords_vec.squaredNorm();
			const double sqrd_norm_y = coords_pred_vec.squaredNorm();
			const double sqrd_norm_x_min_y = (coords_vec - coords_pred_vec).squaredNorm();
			if (order_ == 2) {
				return GradientHurstCovarianceOrder2WrtH(sqrd_norm_x, sqrd_norm_y, sqrd_norm_x_min_y, coords_vec.dot(coords_pred_vec), pars, transf_scale, nugget_var);
			}
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
			return GradientHurstCovarianceARD_vec((*coords).row(j), (*coords_pred).row(i), pars, ind_par, transf_scale, nugget_var);
		}
		template <typename T_x, typename T_y>//T_x and T_y can be vec_t or rows of a den_mat_t (which avoids copying them)
		inline double GradientHurstCovarianceARD_vec(const Eigen::MatrixBase<T_x>& coords_vec,
			const Eigen::MatrixBase<T_y>& coords_pred_vec,
			const vec_t& pars,
			const int ind_par,
			bool transf_scale,
			double nugget_var) const {
			CHECK(0 <= ind_par && ind_par <= num_cov_par_ - 1); // num_cov_par_ = dim_coords + 1
			const double sqrd_norm_x = coords_vec.squaredNorm();
			const double sqrd_norm_y = coords_pred_vec.squaredNorm();
			const double sqrd_norm_x_min_y = (coords_vec - coords_pred_vec).squaredNorm();
			if (order_ == 2) {
				const double c = coords_vec.dot(coords_pred_vec);
				if (ind_par == 0) {
					return GradientHurstCovarianceOrder2WrtH(sqrd_norm_x, sqrd_norm_y, sqrd_norm_x_min_y, c, pars, transf_scale, nugget_var);
				}
				return GradientHurstCovarianceOrder2WrtRange(coords_vec, coords_pred_vec, sqrd_norm_x, sqrd_norm_y, sqrd_norm_x_min_y, c, pars, ind_par, transf_scale, nugget_var);
			}
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
				const double l_k = pars[ind_par + 1];// The range l_k is stored in pars[ind_par + 1]
				// u^(H-1) * a for 0 <= a <= u, which tends to 0 for u -> 0 although u^(H-1) diverges for H < 1.
				//	It is calculated as (a / u) * u^H to avoid the product of a diverging and a vanishing factor
				auto pow_Hm1_times = [H](double u, double a) {
					return (u > 0.) ? ((a / u) * std::pow(u, H)) : 0.;
				};
				const double x_k = coords_vec[k];
				const double y_k = coords_pred_vec[k];
				const double diff_k = x_k - y_k;
				const double dC_dl_k = -pars[0] * H / l_k * (pow_Hm1_times(sqrd_norm_x, x_k * x_k) + pow_Hm1_times(sqrd_norm_y, y_k * y_k) -
					pow_Hm1_times(sqrd_norm_x_min_y, diff_k * diff_k));
				const double cm = transf_scale ? l_k : nugget_var;// multiplicative constant to get gradient on log-scale or backtransform with nugget variance
				grad = cm * dC_dl_k;
			}
			return(grad);
		}//end GradientHurstCovarianceARD_vec

		/*! \brief True for the Matern covariance functions with a fixed smoothness parameter 'shape_' */
		bool IsMaternWithFixedShape() const {
			return(cov_fct_type_ == "matern" || cov_fct_type_ == "matern_space_time" || cov_fct_type_ == "matern_ard");
		}

		/*! \brief True for the Matern covariance functions whose smoothness parameter is estimated (the last covariance parameter) */
		bool IsMaternWithEstimatedShape() const {
			return(cov_fct_type_ == "matern_estimate_shape" || cov_fct_type_ == "matern_ard_estimate_shape" ||
				cov_fct_type_ == "matern_space_time_estimate_shape");
		}

		/*!
		* \brief Returns p if the smoothness 'shape' of a Matern covariance function is a half-integer p + 1/2 for which the
		*		closed form is used, and -1 otherwise
		*/
		static int MaternHalfIntegerOrder(double shape) {
			const double p = std::round(shape - 0.5);
			if (p >= 0. && p <= MAX_ORDER_MATERN_HALF_INTEGER_ && TwoNumbersAreEqual<double>(shape, p + 0.5)) {
				return((int)p);
			}
			return(-1);
		}

		/*!
		* \brief Coefficients a_0, ..., a_p of the polynomial P_p in the Matern correlation exp(-z) P_p(z) with smoothness p + 1/2,
		*		a_i = p! / (2p)! * (2p - i)! / (i! (p - i)!) * 2^i (Rasmussen and Williams, 2006, Eq. (4.16)), calculated recursively
		*/
		static std::vector<double> MaternPolynomialCoefficients(int p) {
			std::vector<double> coef(p + 1);
			coef[0] = 1.;
			for (int i = 0; i < p; ++i) {
				coef[i + 1] = coef[i] * 2. * (p - i) / ((i + 1.) * (2. * p - i));
			}
			return(coef);
		}

		/*! \brief Polynomial with the coefficients 'coef' (a_0 first) evaluated at z with the Horner scheme */
		static inline double EvaluatePolynomial(const std::vector<double>& coef,
			double z) {
			double value = coef.back();
			for (int i = (int)coef.size() - 2; i >= 0; --i) {
				value = value * z + coef[i];
			}
			return(value);
		}

		/*! \brief Smoothness parameters for the central difference with respect to the smoothness (on the log scale if transf_scale) */
		void ShapesForFiniteDifference(double shape,
			bool transf_scale,
			double& shape_up,
			double& shape_down) const {
			if (transf_scale) {
				shape_up = std::exp(std::log(shape) + delta_step_);
				shape_down = std::exp(std::log(shape) - delta_step_);
			}
			else {
				shape_up = shape + delta_step_;
				shape_down = shape - delta_step_;
				CHECK(shape_down > 0.);
			}
		}

		/*! \brief Factor sqrt(2 * shape_) of a Matern covariance function with a fixed smoothness: the transformed range parameter is this factor divided by the range parameter */
		double MaternRangeScale() const {
			if (matern_half_integer_order_ >= 0) {
				return(std::sqrt(2. * matern_half_integer_order_ + 1.));
			}
			return(std::sqrt(2. * shape_));
		}

		/*!
		* \brief Initial value of a transformed range parameter of a Matern covariance function with a fixed smoothness: the correlation is
		*		approximately 0.05 at half the median distance 'med_dist' for the smoothness 0.5, 1.5, and 2.5, and larger for a larger smoothness
		*/
		double MaternInitialTransformedRange(double med_dist) const {
			double range_dist;// scaled distance at which the correlation is approximately 0.05
			if (shape_ <= 1.) {
				range_dist = 3.;
			}
			else if (shape_ <= 2.) {
				range_dist = 4.7;
			}
			else {
				range_dist = 5.9;
			}
			return(2. * range_dist / med_dist);
		}

		/*! \brief Initial value of a range parameter of a Matern covariance function with an estimated smoothness: the one of the smoothness 1.5, the initial smoothness */
		double MaternEstimateShapeInitialRange(double med_dist) const {
			return(med_dist * std::sqrt(3.) / 2. / 4.7);
		}

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
		/*! \brief Order m of the 'hurst' and 'hurst_ard' covariance functions (m - 1 < H < m), 1 for all other covariance functions */
		int order_ = 1;
		/*! \brief p if the fixed smoothness of a Matern covariance function is a half-integer p + 1/2, for which the covariance has a closed form, and -1 otherwise */
		int matern_half_integer_order_ = -1;
		/*! \brief Coefficients a_0, ..., a_p of the polynomial P_p in the correlation exp(-z) P_p(z) of a Matern covariance function with smoothness p + 1/2 (used only for p >= 3) */
		std::vector<double> matern_poly_coef_;
		/*! \brief Coefficients of P_{p-1} / (2p - 1): the derivative of exp(-z) P_p(z) is -z exp(-z) P_{p-1}(z) / (2p - 1) (used only for p >= 3) */
		std::vector<double> matern_poly_coef_deriv_;
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
		/*! \brief If true, the gradients with respect to the range parameters are multiples of the entries of the covariance matrix passed to them, see 'GradientIsMultipleOfCov' */
		bool grad_is_multiple_of_cov_ = false;
		/*! \brief If true, precomputed distances('dist') are used for calculating covariances, otherwise the coordinates are used('coords' and 'coords_pred') */
		bool use_precomputed_dist_for_calc_cov_;
		/*! \brief If true, this is a two-level autoregressive multifidelity covariance */
		bool is_ar1_multifidelity_ = false;
		/*! \brief Base covariance type used by both latent processes in an AR1 multifidelity covariance */
		string_t base_cov_fct_type_;
		/*! \brief Number of parameters of one base covariance in an AR1 multifidelity covariance */
		int num_cov_par_base_ = 0;
		/*! \brief Total coordinate dimension, including the fidelity indicator for an AR1 multifidelity covariance */
		int dim_coordinates_ = 0;
		/*! \brief Covariance functions for the low-fidelity process and the high-fidelity discrepancy */
		std::shared_ptr<CovFunction<T_mat>> base_cov_function_low_;
		std::shared_ptr<CovFunction<T_mat>> base_cov_function_discrepancy_;
		/*! \brief for calculating finite differences  */
		const double delta_step_ = 1e-6;// based on https://math.stackexchange.com/questions/815113/is-there-a-general-formula-for-estimating-the-step-size-h-in-numerical-different/819015#819015		
		/*! \brief List of supported covariance functions */
		const std::set<string_t> SUPPORTED_COV_TYPES_{ "exponential", "powered_exponential",
			"matern", "matern_ard", "matern_space_time", "matern_estimate_shape", "matern_ard_estimate_shape",
			"matern_space_time_estimate_shape", "gaussian", "gaussian_ard",
			"space_time_gneiting", "wendland", "linear", "hurst", "hurst_ard"};
		/*!
		* \brief Largest p for which the closed form of the Matern correlation with smoothness p + 1/2 is used. Its cost grows with p,
		*		and the Matern covariance is practically the Gaussian one for a larger smoothness
		*/
		static constexpr int MAX_ORDER_MATERN_HALF_INTEGER_ = 100;
		const double LARGE_SHAPE_WARNING_THRESHOLD_ = 50.;
		const char* LARGE_SHAPE_WARNING_ = "The shape parameter is very large, it is recommended to use the 'gaussian' covariance function ";
		/*!
		* \brief Upper bound for an estimated smoothness parameter of a Matern covariance function, also for the smoothness nu of the
		*		'space_time_gneiting' covariance function. The likelihood is practically constant in the smoothness for such large values,
		*		for which the Matern covariance is practically the Gaussian one, and the cost of the Matern correlation grows with it
		*/
		const double MAX_ESTIMATED_MATERN_SHAPE_ = 100.;

		template<typename>
		friend class RECompGP;
	};

}  // namespace GPBoost

#endif   // GPB_COV_FUNCTIONS_
