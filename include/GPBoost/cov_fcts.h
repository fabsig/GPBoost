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
	*
	*   Some details:
	*		 1. The template parameter <T_mat> can be either <den_mat_t> or <sp_mat_t>
	*/
	class CovFunction {
	public:
		/*! \brief Constructor */
		CovFunction();

		/*!
		* \brief Constructor
		* \param cov_fct_type Type of covariance function. We follow the notation and parametrization of Diggle and Ribeiro (2007) except for the Matern covariance where we follow Rassmusen and Williams (2006)
		* \param shape Shape parameter of covariance function (=smoothness parameter for Matern and Wendland covariance. For the Wendland covariance function, we follow the notation of Bevilacqua et al. (2018)). This parameter is irrelevant for some covariance functions such as the exponential or Gaussian.
		* \param taper_range Range parameter of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
		* \param taper_mu Parameter \mu of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018)
		*/
		CovFunction(string_t cov_fct_type,
			double shape = 0.,
			double taper_range = 1.,
			double taper_mu = 2.) {
			num_cov_par_ = 2;
			if (SUPPORTED_COV_TYPES_.find(cov_fct_type) == SUPPORTED_COV_TYPES_.end()) {
				Log::REFatal("Covariance of type '%s' is not supported.", cov_fct_type.c_str());
			}
			if (cov_fct_type == "matern") {
				if (!(AreSame(shape, 0.5) || AreSame(shape, 1.5) || AreSame(shape, 2.5))) {
					Log::REFatal("Only shape / smoothness parameters 0.5, 1.5, and 2.5 supported for the Matern covariance function");
				}
			}		
			if (cov_fct_type == "powered_exponential") {
				if (shape <= 0. || shape > 2.) {
					Log::REFatal("Shape needs to be larger than 0 and smaller or equal than 2 for the 'powered_exponential' covariance function");
				}
			}
			if (cov_fct_type == "wendland") {
				if (!(AreSame(shape, 0.0) || AreSame(shape, 1.0) || AreSame(shape, 2.0))) {
					Log::REFatal("Only shape / smoothness parameters 0, 1, and 2 supported for the Wendland covariance function");
				}
				CHECK(taper_range > 0.);
				CHECK(taper_mu >= 1.);
				taper_range_ = taper_range;
				taper_mu_ = taper_mu;
				num_cov_par_ = 1;
			}
			cov_fct_type_ = cov_fct_type;
			shape_ = shape;
		}

		/*! \brief Destructor */
		~CovFunction() {
		}

		/*!
		* \brief Transform the covariance parameters
		* \param sigma2 Marginal variance
		* \param pars Vector with covariance parameters on orignal scale
		* \param[out] pars_trans Transformed covariance parameters
		*/
		void TransformCovPars(const double sigma2, const vec_t& pars, vec_t& pars_trans) {
			pars_trans = pars;
			pars_trans[0] = pars[0] / sigma2;
			if (cov_fct_type_ == "exponential" || (cov_fct_type_ == "matern" && AreSame(shape_, 0.5))) {
				pars_trans[1] = 1. / pars[1];
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 1.5)) {
				pars_trans[1] = sqrt(3.) / pars[1];
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 2.5)) {
				pars_trans[1] = sqrt(5.) / pars[1];
			}
			else if (cov_fct_type_ == "gaussian") {
				pars_trans[1] = 1. / (pars[1] * pars[1]);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				pars_trans[1] = 1. / (std::pow(pars[1], shape_));
			}
		}

		/*!
		* \brief Function transforms the covariance parameters back to the original scale
		* \param sigma2 Marginal variance
		* \param pars Vector with covariance parameters
		* \param[out] pars_orig Back-transformed, original covariance parameters
		*/
		void TransformBackCovPars(const double sigma2, const vec_t& pars, vec_t& pars_orig) {
			pars_orig = pars;
			pars_orig[0] = sigma2 * pars[0];
			if (cov_fct_type_ == "exponential" || (cov_fct_type_ == "matern" && AreSame(shape_, 0.5))) {
				pars_orig[1] = 1. / pars[1];
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 1.5)) {
				pars_orig[1] = sqrt(3.) / pars[1];
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 2.5)) {
				pars_orig[1] = sqrt(5.) / pars[1];
			}
			else if (cov_fct_type_ == "gaussian") {
				pars_orig[1] = 1. / std::sqrt(pars[1]);
			}
			else if (cov_fct_type_ == "powered_exponential") {
				pars_orig[1] = 1. / (std::pow(pars[1], 1. / shape_));
			}
		}

		//TODO: For the training data, this can be done faster as sigma is symetric: add parameter bool dist_is_symetric and do calculations only for upper half?
		/*!
		* \brief Calculates covariance matrix
		*		Note: this is the version for dense matrixes
		* \param dist Distance matrix
		* \param pars Vector with covariance parameters
		* \param[out] sigma Covariance matrix
		*/
		void GetCovMat(const den_mat_t& dist, const vec_t& pars, den_mat_t& sigma) {
			CHECK(pars.size() == num_cov_par_);
			if (cov_fct_type_ == "exponential" || (cov_fct_type_ == "matern" && AreSame(shape_, 0.5))) {
				//den_mat_t sigma(dist.rows(),dist.cols());//TODO: this is not working, check whether this can be done using triangularView? If it works, make dist_ (see re_comp.h) an upper triangular matrix as lower part is not used
				//sigma.triangularView<Eigen::Upper>() = (pars[1] * ((-pars[2] * dist.triangularView<Eigen::Upper>().array()).exp())).matrix();
				//sigma.triangularView<Eigen::StrictlyLower>() = sigma.triangularView<Eigen::StrictlyUpper>().transpose();
				sigma = (pars[0] * ((-pars[1] * dist.array()).exp())).matrix();
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 1.5)) {
				sigma = (pars[0] * (1. + pars[1] * dist.array()) * ((-pars[1] * dist.array()).exp())).matrix();
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 2.5)) {
				sigma = (pars[0] * (1. + pars[1] * dist.array() + pars[1] * pars[1] * dist.array().square() / 3.) * ((-pars[1] * dist.array()).exp())).matrix();
			}
			else if (cov_fct_type_ == "gaussian") {

				sigma = (pars[0] * ((-pars[1] * dist.array().square()).exp())).matrix();
			}
			else if (cov_fct_type_ == "powered_exponential") {
				sigma = (pars[0] * ((-pars[1] * dist.array().pow(shape_)).exp())).matrix();
			}
			else if (cov_fct_type_ == "wendland" && AreSame(shape_, 0.)) {
				sigma = dist / taper_range_;
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)dist.rows(); ++i) {
					for (int j = 0; j < (int)dist.cols(); ++j) {
						if (sigma(i, j) >= 1.) {
							sigma(i, j) = 0.;
						}
						else {
							sigma(i, j) = pars[0] * std::pow((1. - sigma(i, j)), taper_mu_);
						}
					}
				}
			}
			else if (cov_fct_type_ == "wendland" && AreSame(shape_, 1.)) {
				sigma = dist / taper_range_;
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)dist.rows(); ++i) {
					for (int j = 0; j < (int)dist.cols(); ++j) {
						if (sigma(i, j) >= 1.) {
							sigma(i, j) = 0.;
						}
						else {
							sigma(i, j) = pars[0] * std::pow((1. - sigma(i, j)), taper_mu_ + 1.) * (1. + sigma(i, j) * (taper_mu_ + 1.));
						}
					}
				}
			}
			else if (cov_fct_type_ == "wendland" && AreSame(shape_, 2.)) {
				sigma = dist / taper_range_;
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)dist.rows(); ++i) {
					for (int j = 0; j < (int)dist.cols(); ++j) {
						if (sigma(i, j) >= 1.) {
							sigma(i, j) = 0.;
						}
						else {
							sigma(i, j) = pars[0] * std::pow((1. - sigma(i, j)), taper_mu_ + 2.) *
								(1. + sigma(i, j) * (taper_mu_ + 2.) + std::pow(sigma(i, j), 2) * (taper_mu_ * taper_mu_ + 4 * taper_mu_ + 3.) / 3.);
						}
					}
				}
			}
			else {
				Log::REFatal("Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}

		/*!
		* \brief Calculates covariance matrix
		*		Note: this is the version for dense matrixes
		* \param dist Distance matrix
		* \param pars Vector with covariance parameters
		* \param[out] sigma Covariance matrix
		*/
		void GetCovMat(const sp_mat_t& dist, const vec_t& pars, sp_mat_t& sigma) {
			CHECK(pars.size() == num_cov_par_);
			sigma = dist;
			if (cov_fct_type_ == "exponential" || (cov_fct_type_ == "matern" && AreSame(shape_, 0.5))) {
				sigma.coeffs() = pars[0] * ((-pars[1] * sigma.coeffs()).exp());
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 1.5)) {
				sigma.coeffs() = pars[0] * (1. + pars[1] * sigma.coeffs()) * ((-pars[1] * sigma.coeffs()).exp());
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 2.5)) {
				sigma.coeffs() = pars[0] * (1. + pars[1] * sigma.coeffs() + pars[1] * pars[1] * sigma.coeffs().square() / 3.) * ((-pars[1] * sigma.coeffs()).exp());
			}
			else if (cov_fct_type_ == "gaussian") {
				sigma.coeffs() = pars[0] * ((-pars[1] * sigma.coeffs().square()).exp());
			}
			else if (cov_fct_type_ == "powered_exponential") {
				sigma.coeffs() = pars[0] * ((-pars[1] * sigma.coeffs().pow(shape_)).exp());
			}
			else if (cov_fct_type_ == "wendland" && AreSame(shape_, 0.)) {
				sigma.coeffs() = pars[0] * (1. - sigma.coeffs() / taper_range_).pow(taper_mu_);
			}
			else if (cov_fct_type_ == "wendland" && AreSame(shape_, 1.)) {
				sigma.coeffs() = sigma.coeffs() / taper_range_;
				sigma.coeffs() = pars[0] * (1. - sigma.coeffs()).pow(taper_mu_ + 1.) * (1. + sigma.coeffs() * (taper_mu_ + 1.));
			}
			else if (cov_fct_type_ == "wendland" && AreSame(shape_, 2.)) {
				sigma.coeffs() = sigma.coeffs() / taper_range_;
				sigma.coeffs() = pars[0] * (1. - sigma.coeffs()).pow(taper_mu_ + 2.) *
					(1. + sigma.coeffs() * (taper_mu_ + 2.) + sigma.coeffs().pow(2) * (taper_mu_ * taper_mu_ + 4 * taper_mu_ + 3.) / 3.);
			}
			else {
				Log::REFatal("Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}

		/*!
		* \brief Calculates derivatives of the covariance matrix with respect to the inverse range parameter
		*		Note: this is the version for dense matrixes
		* \param dist Distance matrix
		* \param sigma Covariance matrix
		* \param pars Vector with covariance parameters
		* \param[out] sigma_grad Derivative of covariance matrix with respect to the inverse range parameter
		* \param transf_scale If true, the derivative is taken on the transformed scale otherwise with respect to the original range parameter (the parameters values pars are always given on the transformed scale). Optimiziation is done using transf_scale=true. transf_scale=false is needed, for instance, for calcualting the Fisher information on the original scale.
		* \param marg_var Marginal variance parameters sigma^2 (used only if transf_scale = false to transform back)
		*/
		void GetCovMatGradRange(const den_mat_t& dist, const den_mat_t& sigma, const vec_t& pars, den_mat_t& sigma_grad,
			bool transf_scale = true, double marg_var = 1.) {
			CHECK(pars.size() == num_cov_par_);
			if (cov_fct_type_ == "exponential" || (cov_fct_type_ == "matern" && AreSame(shape_, 0.5))) {
				double cm = transf_scale ? (-1. * pars[1]) : (marg_var * pars[1] * pars[1]);
				sigma_grad = cm * sigma.cwiseProduct(dist);
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 1.5)) {
				double cm = transf_scale ? 1. : (-1. * marg_var * pars[1]);
				sigma_grad = cm * ((pars[0] * pars[1] * dist.array() * ((-pars[1] * dist.array()).exp())).matrix() - pars[1] * sigma.cwiseProduct(dist));
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 2.5)) {
				double cm = transf_scale ? 1. : (-1. * marg_var * pars[1]);
				sigma_grad = cm * ((pars[0] * (pars[1] * dist.array() + (2. / 3.) * pars[1] * pars[1] * dist.array().square()) *
					((-pars[1] * dist.array()).exp())).matrix() - pars[1] * sigma.cwiseProduct(dist));
			}
			else if (cov_fct_type_ == "gaussian") {
				double cm = transf_scale ? (-1. * pars[1]) : (2. * marg_var * std::pow(pars[1], 3. / 2.));
				sigma_grad = cm * sigma.cwiseProduct(dist.array().square().matrix());
			}
			else if (cov_fct_type_ == "powered_exponential") {
				double cm = transf_scale ? (-1. * pars[1]) : (shape_ * marg_var * std::pow(pars[1], (shape_ + 1.) / shape_));
				sigma_grad = cm * sigma.cwiseProduct(dist.array().pow(shape_).matrix());
			}
			else {
				Log::REFatal("GetCovMatGradRange: Covariance of type '%s' is not supported.", cov_fct_type_.c_str());
			}
		}

		/*!
		* \brief Calculates derivatives of the covariance matrix with respect to the inverse range parameter
		*		Note: this is the version for sparse matrixes
		* \param dist Distance matrix
		* \param sigma Covariance matrix
		* \param pars Vector with covariance parameters
		* \param[out] sigma_grad Derivative of covariance matrix with respect to the inverse range parameter
		* \param transf_scale If true, the derivative is taken on the transformed scale otherwise with respect to the original range parameter (the parameters values pars are always given on the transformed scale). Optimiziation is done using transf_scale=true. transf_scale=false is needed, for instance, for calcualting the Fisher information on the original scale.
		* \param marg_var Marginal variance parameters sigma^2 (used only if transf_scale = false to transform back)
		*/
		void GetCovMatGradRange(const sp_mat_t& dist, const sp_mat_t& sigma, const vec_t& pars, sp_mat_t& sigma_grad,
			bool transf_scale = true, double marg_var = 1.) {
			CHECK(pars.size() == num_cov_par_);
			if (cov_fct_type_ == "exponential" || (cov_fct_type_ == "matern" && AreSame(shape_, 0.5))) {
				double cm = transf_scale ? (-1. * pars[1]) : (marg_var * pars[1] * pars[1]);
				sigma_grad = cm * sigma.cwiseProduct(dist);
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 1.5)) {
				double cm = transf_scale ? 1. : (-1. * marg_var * pars[1]);
				sigma_grad = dist;
				sigma_grad.coeffs() = pars[0] * pars[1] * sigma_grad.coeffs() * ((-pars[1] * sigma_grad.coeffs()).exp());
				sigma_grad -= pars[1] * sigma.cwiseProduct(dist);
				sigma_grad *= cm;
			}
			else if (cov_fct_type_ == "matern" && AreSame(shape_, 2.5)) {
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
		}

		bool AreSame(double a, double b) {
			const double epsilon = 0.00000001;
			bool are_same = false;
			if (fabs(a) < epsilon) {
				are_same = fabs(b) < epsilon;
			}
			else {
				are_same = fabs(a - b) < a * epsilon;
			}
			return are_same;
		}

	private:
		/*! \brief Type of covariance function  */
		string_t cov_fct_type_;
		/*! \brief Shape parameter of covariance function (=smoothness parameter for Matern covariance) */
		double shape_;
		/*! \brief Range parameter of Wendland covariance function / taper */
		double taper_range_;
		/*! \brief Parameter \mu of Wendland covariance function / taper. We follow the notation of Bevilacqua et al. (2018) */
		double taper_mu_;
		/*! \brief Number of covariance parameters*/
		int num_cov_par_;

		/*! \brief List of supported covariance functions */
		const std::set<string_t> SUPPORTED_COV_TYPES_{ "exponential", "gaussian", "powered_exponential", "matern", "wendland" };

		template<typename>
		friend class RECompGP;
	};

}  // namespace GPBoost

#endif   // GPB_COV_FUNCTIONS_
