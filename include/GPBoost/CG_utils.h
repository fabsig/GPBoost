/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 - 2024 Tim Gyger, Pascal Kuendig, and Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_CG_UTILS_
#define GPB_CG_UTILS_

#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>

#include <LightGBM/utils/log.h>
#include <chrono>
#include <thread> //temp

using LightGBM::Log;

namespace GPBoost {
	/*!
	* \brief Preconditioned conjugate gradient descent to solve A u = rhs when rhs is a vector
	*		 A = (Sigma^-1 + W) is a symmetric matrix of dimension nxn, a Vecchia approximation for Sigma^-1,
	*		 Sigma^-1 = B^T D^-1 B, is given, and W is a diagonal matrix.
	*		 "Sigma_inv_plus_BtWB" (P = B^T (D^-1 + W) B)  or "zero_infill_incomplete_cholesky" (P = L^T L) is used as preconditioner.
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^(-1) B ("=" Cholesky factor)
	* \param B_t_D_inv_rm Row-major matrix that contains the product B^T D^-1. Outsourced in order to reduce the overhead of the function.
	* \param rhs Vector of dimension nx1 on the rhs
	* \param[out] u Approximative solution of the linear system (solution written on input) (must have been declared with the correct n-dimension)
	* \param[out] NA_or_Inf_found Is set to true, if NA or Inf is found in the residual of conjugate gradient algorithm.
	* \param p Maximal number of conjugate gradient steps
	* \param find_mode_it In the first mode-finding iteration (find_mode_it == 0) u is set to zero at the beginning of the algorithm (cold-start).
	* \param delta_conv Tolerance for checking convergence of the algorithm
	* \param THRESHOLD_ZERO_RHS_CG If the L1-norm of the rhs is below this threshold the CG is not executed and a vector u of 0's is returned.
	* \param cg_preconditioner_type Type of preconditioner used.
	* \param D_inv_plus_W_B_rm Row-major matrix that contains the product (D^(-1) + W) B used for the preconditioner "Sigma_inv_plus_BtWB".
	* \param L_SigmaI_plus_W_rm Row-major matrix that contains sparse cholesky factor L of matrix L^T L =  B^T D^(-1) B + W used for the preconditioner "zero_infill_incomplete_cholesky". 
	*/
	void CGVecchiaLaplaceVec(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const string_t cg_preconditioner_type,
		const sp_mat_rm_t& D_inv_plus_W_B_rm,
		const sp_mat_rm_t& L_SigmaI_plus_W_rm);

	/*!
	* \brief Version of CGVecchiaLaplaceVec() that solves (Sigma^-1 + W) u = rhs by u = W^(-1) (W^(-1) + Sigma)^(-1) Sigma rhs where the preconditioned conjugate 
	*		 gradient descent algorithm is used to approximately solve for (W^(-1) + Sigma)^(-1) Sigma rhs. 
	*        P = (W^(-1) + Sigma_L_k Sigma_L_k^T) is used as preconditioner where Sigma_L_k results from a rank(Sigma_L_k) = k (k << n)
	*		 pivoted Cholseky decomposition of the nonapproximated covariance matrix.
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
	* \param D_inv_B_rm Row-major matrix that contains the product D^-1 B. Outsourced in order to reduce the overhead of the function.
	* \param rhs Vector of dimension nx1 on the rhs
	* \param[out] u Approximative solution of the linear system (solution written on input) (must have been declared with the correct n-dimension)
	* \param[out] NA_or_Inf_found Is set to true, if NA or Inf is found in the residual of conjugate gradient algorithm.
	* \param p Maximal number of conjugate gradient steps
	* \param find_mode_it In the first mode-finding iteration (find_mode_it == 0) u is set to zero at the beginning of the algorithm (cold-start).
	* \param delta_conv Tolerance for checking convergence of the algorithm
	* \param THRESHOLD_ZERO_RHS_CG If the L1-norm of the rhs is below this threshold the CG is not executed and a vector u of 0's is returned.
	* \param chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia Cholesky factor E of matrix EE^T = (I_k + Sigma_L_k^T W^(-1) Sigma_L_k)
	* \param Sigma_L_k Matrix of dimension nxk: Pivoted Cholseky decomposition of the nonapproximated covariance matrix, generated in re_model_template.h
	*/
	void CGVecchiaLaplaceVecWinvplusSigma(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const chol_den_mat_t& chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k);


	void CGVecchiaLaplaceVecWinvplusSigma_FITC_P(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const den_mat_t* cross_cov,
		const vec_t& diagonal_approx_inv_preconditioner);


	/*!
	* \brief Preconditioned conjugate gradient descent in combination with the Lanczos algorithm.
	*		 A linear system A U = rhs is solved, where the rhs is a matrix of dimension nxt of t random column-vectors and 
	*		 A = (Sigma^-1 + W) is a symmetric matrix of dimension nxn. 
	*		 "Sigma_inv_plus_BtWB" (P = B^T (D^-1 + W) B)  or "zero_infill_incomplete_cholesky" (P = L^T L) is used as preconditioner.
	*		 Further, a Vecchia approximation for Sigma^-1 = B^T D^-1 B is given, and W is a diagonal matrix.
	*		 The function returns t approximative tridiagonalizations T of the symmetric matrix P^(-0.5) A P^(-0.5) = Q T Q^T in vector form (diagonal + subdiagonal of T)
	*		 and an approximative solution of the linear system.
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
	* \param B_t_D_inv_rm Row-major matrix that contains the product B^T D^-1. Outsourced in order to reduce the overhead of the function.
	* \param rhs Matrix of dimension nxt that contains random vectors z_1, ..., z_t with Cov(z_i) = P
	* \param[out] Tdiags The diagonals of the t approximative tridiagonalizations of P^(-0.5) A P^(-0.5) in vector form (solution written on input)
	* \param[out] Tsubdiags The subdiagonals of the t approximative tridiagonalizations of P^(-0.5) A P^(-0.5) in vector form (solution written on input)
	* \param[out] U Approximative solution of the linear system (solution written on input) (must have been declared with the correct nxt dimensions)
	* \param[out] NA_or_Inf_found Is set to true, if NA or Inf is found in the residual of conjugate gradient algorithm.
	* \param num_data n-Dimension of the linear system
	* \param t t-Dimension of the linear system
	* \param p Maximal number of conjugate gradient steps
	* \param delta_conv Tolerance for checking convergence of the algorithm
	* \param cg_preconditioner_type Type of preconditioner used.
	* \param D_inv_plus_W_B_rm Row-major matrix that contains the product (D^(-1) + W) B used for the preconditioner "Sigma_inv_plus_BtWB".
	* \param L_SigmaI_plus_W_rm Row-major matrix that contains sparse cholesky factor L of matrix L^T L =  B^T D^(-1) B + W used for the preconditioner "zero_infill_incomplete_cholesky".
	*/
	void CGTridiagVecchiaLaplace(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const sp_mat_rm_t& D_inv_plus_W_B_rm,
		const sp_mat_rm_t& L_SigmaI_plus_W_rm);

	/*!
	* \brief Version of CGTridiagVecchiaLaplace() where A = (W^(-1) + Sigma).
	*		 The linear system is solved with P = (W^(-1) + Sigma_L_k Sigma_L_k^T) as preconditioner, where Sigma_L_k results from a
	*        rank(Sigma_L_k) = k (k << n) pivoted Cholseky decomposition of the nonapproximated covariance matrix.
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
	* \param D_inv_B_rm Row-major matrix that contains the product D^-1 B. Outsourced in order to reduce the overhead of the function.
	* \param rhs Matrix of dimension nxt that contains random vectors z_1, ..., z_t with Cov(z_i) = P
	* \param[out] Tdiags The diagonals of the t approximative tridiagonalizations of P^(-0.5) A P^(-0.5) in vector form (solution written on input)
	* \param[out] Tsubdiags The subdiagonals of the t approximative tridiagonalizations of P^(-0.5) A P^(-0.5) in vector form (solution written on input)
	* \param[out] U Approximative solution of the linear system (solution written on input) (must have been declared with the correct nxt dimensions)
	* \param[out] NA_or_Inf_found Is set to true, if NA or Inf is found in the residual of conjugate gradient algorithm.
	* \param num_data n-Dimension of the linear system
	* \param t t-Dimension of the linear system
	* \param p Maximal number of conjugate gradient steps
	* \param delta_conv Tolerance for checking convergence of the algorithm
	* \param chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia Cholesky factor E of matrix EE^T = (I_k + Sigma_L_k^T W^(-1) Sigma_L_k)
	* \param Sigma_L_k Matrix of dimension nxk: Pivoted Cholseky decomposition of the nonapproximated covariance matrix, generated in re_model_template.h
	*/
	void CGTridiagVecchiaLaplaceWinvplusSigma(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const chol_den_mat_t& chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k);

	void CGTridiagVecchiaLaplaceWinvplusSigma_FITC_P(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const den_mat_t* cross_cov,
		const vec_t& diagonal_approx_inv_preconditioner);

	/*!
	* \brief Fills a given matrix with standard normal RV's.
	* \param generator Random number generator
	* \param[out] R Matrix of random vectors (r_1, r_2, r_3, ...), where r_i is of dimension n & Cov(r_i) = I (must have been declared with the correct dimensions)
	*/
	void GenRandVecNormal(RNG_t& generator,
		den_mat_t& R);
	
	/*!
	* \brief Fills a given matrix with Rademacher RV's.
	* \param generator Random number generator
	* \param[out] R Matrix of random vectors (r_1, r_2, r_3, ...), where r_i is of dimension n & Cov(r_i) = I (must have been declared with the correct dimensions)
	*/
	void GenRandVecRademacher(RNG_t& generator,
		den_mat_t& R);

	/*!
	* \brief Stochastic estimation of log(det(A)) given t approximative Lanczos tridiagonalizations T of a symmetric matrix A = Q T Q^T of dimension nxn,
	*		 where T is given in vector form (diagonal + subdiagonal of T).
	*		 Lanczos was previously run t-times based on t random vectors r_1, ..., r_t with r_i ~ N(0,I).
	* \param Tdiags The diagonals of the t approximative tridiagonalizations of A in vector form
	* \param Tsubdiags The subdiagonals of the t approximative tridiagonalizations of A in vector form
	* \param ldet[out] Estimation of log(det(A)) (solution written on input)
	* \param num_data n-Dimension
	* \param t Number of tridiagonalization matrices T
	*/
	void LogDetStochTridiag(const std::vector<vec_t>& Tdiags,
		const std::vector<vec_t>& Tsubdiags,
		double& ldet,
		const data_size_t num_data,
		const int t);

	/*!
	* \brief Calculate c_opt = Cov(z_i^T (A^(-1) dA P^(-1)) z_i, z_i^T (B^(-1) dB P^(-1)) z_i) / Var(z_i^T (B^(-1) dB P^(-1)) z_i) according to StochSim_script_AS20.pdf on p.39.
	*		 c_opt is used to weight the variance reduction when calculating the gradients.
	* \param zt_AI_A_deriv_PI_z Vector of dimension t, where each entry is a quadratic form z_i^T (A^(-1) dA P^(-1)) z_i with a random vectors z_1, ..., z_t with Cov(z_i) = P.
	* \param zt_BI_B_deriv_PI_z Vector of dimension t, where each entry is a quadratic form z_i^T (B^(-1) dB P^(-1)) z_i with a random vectors z_1, ..., z_t with Cov(z_i) = P.
	* \param tr_AI_A_deriv Stochastic tr(A^(-1) dA) which is equal to the mean of the vector zt_AI_A_deriv_PI_z.
	* \param tr_BI_B_deriv Stochastic tr(B^(-1) dB) which is equal to the mean of the vector zt_BI_B_deriv_PI_z.
	* \param c_opt[out] Evaluation of Cov(z_i^T (A^(-1) dA P^(-1)) z_i, z_i^T (B^(-1) dB P^(-1)) z_i) / Var(z_i^T (B^(-1) dB P^(-1)) z_i)
	*/
	void CalcOptimalC(const vec_t& zt_AI_A_deriv_PI_z,
		const vec_t& zt_BI_B_deriv_PI_z,
		const double& tr_AI_A_deriv,
		const double& tr_BI_B_deriv,
		double& c_opt);

	/*!
	* \brief Vectorized version of CalcOptimalC where c_opt is calculated for all dA/db_i, respectively dB/db_i, with i in 1, ..., n.
	* \param Z_AI_A_deriv_PI_Z Matrix of dimension nxt, where each entry is a quadratic form z_i^T (A^(-1) dA/db_i P^(-1)) z_i, where z_i changes columnwise and dA/db_i changes rowwise.
	* \param Z_BI_B_deriv_PI_Z Matrix of dimension nxt, where each entry is a quadratic form z_i^T (B^(-1) dB/db_i P^(-1)) z_i, where z_i changes columnwise and dB/db_i changes rowwise.
	* \param tr_AI_A_deriv Vector of dimension n, where the ith entry contains tr(A^(-1) dA/db_i) (=mean of the row i of the matrix Z_AI_A_deriv_PI_Z).
	* \param tr_BI_B_deriv Vector of dimension n, where the ith entry contains tr(B^(-1) dB/db_i) (=mean of the row i of the matrix Z_BI_B_deriv_PI_Z).
	* \param c_opt[out] Vector of dimension n, that contains the rowwise evaluation of c_opt.
	*/
	void CalcOptimalCVectorized(const den_mat_t& Z_AI_A_deriv_PI_Z,
		const den_mat_t& Z_BI_B_deriv_PI_Z,
		const vec_t& tr_AI_A_deriv,
		const vec_t& tr_BI_B_deriv,
		vec_t& c_opt);

	/*!
	* \brief Reverse incomplete Cholesky factorization L^T L = A under the constrain that L has the same sparcity pattern as A or B.
	* \param A Column-major matrix to factorize.
	* \param B Column-major matrix providing the sparcity pattern.
	* \param L_rm[out] Row-major matrix containing the sparse lower triangular factor L. 
	*/
	void ReverseIncompleteCholeskyFactorization(sp_mat_t& A,
		const sp_mat_t& B, 
		sp_mat_rm_t& L_rm);

	/*!
	* \brief Pivoted Cholesky factorization according to Habrecht et al. (2012) for the original (nonapproximated) covariance matrix (Sigma)
	* \param cov_f Pointer to function which accesses elements of Sigma (see https://www.geeksforgeeks.org/passing-a-function-as-a-parameter-in-cpp)
	* \param var_f Pointer to fuction which accesses the diagonal of Sigma which equals the marginal variance and is the same for all entries (i,i)
	* \param Sigma_L_k[out] Matrix of dimension nxmax_it such that Sigma_L_k Sigma_L_k^T ~ Sigma and rank(Sigma_L_k) <= max_it (solution written on input)
	* \param max_it Max rank of Sigma_L_k
	* \param err_tol Error tolerance - stop the algorithm if the sum of the diagonal elements of the Schur complement is smaller than err_tol
	*/
	template<typename T_mat>
	void PivotedCholsekyFactorizationSigma(
		RECompBase<T_mat>* re_comp,
		den_mat_t& Sigma_L_k,
		int max_it,
		const double err_tol) {

		int m = 0;
		int i, pi_m_old;
		double err, L_jm;
		data_size_t num_re = re_comp->GetNumUniqueREs();//number of random effects, usually this equals num_data, but it can be smaller if there are duplicates
		vec_t diag(num_re), L_row_m;
		vec_int_t pi(num_re);
		max_it = std::min(max_it, num_re);
		Sigma_L_k.resize(num_re, max_it);
		Sigma_L_k.setZero();
		double diag_ii = re_comp->GetZSigmaZtii();

		for (int h = 0; h < num_re; ++h) {
			pi(h) = h;
			//The diagonal of the covariance matrix equals the marginal variance and is the same for all entries (i,i). 
			diag(h) = diag_ii;
		}
		err = diag.lpNorm<1>();

		while (m == 0 || (m < max_it && err > err_tol)) {

			diag(pi.tail(num_re - m)).maxCoeff(&i);
			i += m;

			pi_m_old = pi(m);
			pi(m) = pi(i);
			pi(i) = pi_m_old;

			//L[(m+1):n,m]
			if ((m + 1) < num_re) {

				if (m > 0) {
					L_row_m = Sigma_L_k.row(pi(m)).transpose();
				}

				for (int j = m + 1; j < num_re; ++j) {

					L_jm = re_comp->GetZSigmaZtij(pi(j), pi(m));

					if (m > 0) { //need previous columns
						L_jm -= Sigma_L_k.row(pi(j)).dot(L_row_m);
					}

					if (!(fabs(L_jm) < 1e-12)) {
						L_jm /= sqrt(diag(pi(m)));
						Sigma_L_k(pi(j), m) = L_jm;
					}

					diag(pi(j)) -= L_jm * L_jm;
				}
				err = diag(pi.tail(num_re - (m + 1))).lpNorm<1>();
			}

			//L[m,m] - Update post L[(m+1):n,m] to be able to multiply with L[m,] beforehand
			Sigma_L_k(pi(m), m) = sqrt(diag(pi(m)));
			m = m + 1;
		}
	}//end PivotedCholsekyFactorizationSigma

	/*!
	* \brief Preconditioned conjugate gradient descent to solve Au=rhs when rhs is a vector
	*		 A = (C_s + C_nm*(C_m)^(-1)*C_mn) is a symmetric matrix of dimension nxn and a full-scale-approximation for Sigma
	*		 P = diag(C_s) + C_nm*(C_m)^(-1)*C_mn is used as preconditioner.
	* \param sigma_resid Residual Matrix C_s
	* \param sigma_cross_cov Matrix C_mn in Predictive Process Part C_nm*(C_m)^(-1)*C_mn
	* \param chol_ip_cross_cov Cholesky Factor of C_m, the inducing point matrix, times cross-covariance
	* \param rhs Vector of dimension nx1 on the rhs
	* \param[out] u Approximative solution of the linear system (solution written on input) (must have been declared with the correct n-dimension)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param p Number of conjugate gradient steps
	* \param delta_conv tolerance for checking convergence
	* \param THRESHOLD_ZERO_RHS_CG If the L1-norm of the rhs is below this threshold the CG is not executed and a vector u of 0's is returned
	* \param cg_preconditioner_type Type of preconditoner used for the conjugate gradient algorithm
	* \param chol_fact_woodbury_preconditioner Cholesky factor of Matrix C_m + C_mn*D^(-1)*C_nm
	* \param diagonal_approx_inv_preconditioner Diagonal D of residual Matrix C_s
	*/
	template <class T_mat>
	void CGFSA(const T_mat& sigma_resid,
		const den_mat_t& sigma_cross_cov,
		const den_mat_t& sigma_cross_cov_preconditioner,
		const den_mat_t& chol_ip_cross_cov,
		const vec_t& rhs,
		vec_t& u,
		bool& NaN_found,
		int p,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const string_t cg_preconditioner_type,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const vec_t& diagonal_approx_inv_preconditioner) {

		p = std::min(p, (int)rhs.size());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h;
		vec_t v;

		vec_t diag_sigma_resid_inv_r, sigma_cross_cov_diag_sigma_resid_inv_r, mean_diag_sigma_resid_inv_r, sigma_cross_cov_mean_diag_sigma_resid_inv_r;
		bool early_stop_alg = false;
		double a = 0;
		double b = 1;
		double r_norm;

		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}
		bool is_zero = u.isZero(0);

		if (is_zero) {
			r = rhs;
		}
		else {
			r = rhs - sigma_resid * u - (chol_ip_cross_cov.transpose() * (chol_ip_cross_cov * u));//r = rhs - A * u
		}

		//z = P^(-1) r
		if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
			//D^-1*r
			diag_sigma_resid_inv_r = diagonal_approx_inv_preconditioner.asDiagonal() * r; // ??? cwiseProd (TODO)

			//Cmn*D^-1*r
			sigma_cross_cov_diag_sigma_resid_inv_r = sigma_cross_cov_preconditioner.transpose() * diag_sigma_resid_inv_r;
			//P^-1*r using Woodbury Identity
			z = diag_sigma_resid_inv_r - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov_preconditioner * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_r)));

		}
		else if (cg_preconditioner_type == "none") {
			z = r;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}
		h = z;

		for (int j = 0; j < p; ++j) {

			v = sigma_resid * h + (chol_ip_cross_cov.transpose() * (chol_ip_cross_cov * h));


			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_norm = r.norm();
			if (std::isnan(r_norm) || std::isinf(r_norm)) {
				NaN_found = true;
				return;
			}
			if (r_norm < delta_conv) {
				early_stop_alg = true;
			}

			z_old = z;

			//z = P^(-1) r 
			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				diag_sigma_resid_inv_r = diagonal_approx_inv_preconditioner.asDiagonal() * r; // ??? cwiseProd (TODO)
				sigma_cross_cov_diag_sigma_resid_inv_r = sigma_cross_cov_preconditioner.transpose() * diag_sigma_resid_inv_r;
				z = diag_sigma_resid_inv_r - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov_preconditioner * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_r)));

			}
			else if (cg_preconditioner_type == "none") {
				z = r;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;

			if (early_stop_alg) {

				Log::REInfo("CGFSA stop after %i CG-Iterations.", j + 1);

				return;
			}
		}
		Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise increase 'cg_max_num_it'.", p);
	}// end CGFSA

	/*!
	* \brief Preconditioned conjugate gradient descent in combination with the Lanczos algorithm
	*		 Given the linear system AU=rhs where rhs is a matrix of dimension nxt of t probe column-vectors and
	*		 A = (C_s + C_nm*(C_m)^(-1)*C_mn) is a symmetric matrix of dimension nxn and a full-fcale-approximation for Sigma
	*		 P = diag(C_s) + C_nm*(C_m)^(-1)*C_mn is used as preconditioner.
	*		 The function returns t approximative tridiagonalizations T of the symmetric matrix A=QTQ' in vector form (diagonal + subdiagonal of T).
	* \param sigma_resid Residual Matrix C_s
	* \param sigma_cross_cov Matrix C_mn in Predictive Process Part C_nm*(C_m)^(-1)*C_mn
	* \param chol_ip_cross_cov Cholesky Factor of C_m, the inducing point matrix, times cross-covariance
	* \param rhs Matrix of dimension nxt that contains (column-)probe vectors z_1,...,z_t with Cov[z_i] = P
	* \param[out] Tdiags The diagonals of the t approximative tridiagonalizations of A in vector form (solution written on input)
	* \param[out] Tsubdiags The subdiagonals of the t approximative tridiagonalizations of A in vector form (solution written on input)
	* \param[out] U Approximative solution of the linear system (solution written on input) (must have been declared with the correct nxt dimensions)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param num_data n-Dimension of the linear system
	* \param t t-Dimension of the linear system
	* \param p Number of conjugate gradient steps
	* \param delta_conv Tolerance for checking convergence of the algorithm
	* \param cg_preconditioner_type Type of preconditoner used for the conjugate gradient algorithm
	* \param chol_fact_woodbury_preconditioner Cholesky factor of Matrix C_m + C_mn*D^(-1)*C_nm
	* \param diagonal_approx_inv_preconditioner Diagonal D of residual Matrix C_s
	*/
	template <class T_mat>
	void CGTridiagFSA(const T_mat& sigma_resid,
		const den_mat_t& sigma_cross_cov,
		const den_mat_t& sigma_cross_cov_preconditioner,
		const den_mat_t& chol_ip_cross_cov,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const vec_t& diagonal_approx_inv_preconditioner) {
		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, Z(num_data, t), Z_old, H, V(num_data, t), diag_sigma_resid_inv_R, sigma_cross_cov_diag_sigma_resid_inv_R,
			mean_diag_sigma_resid_inv_R, sigma_cross_cov_mean_diag_sigma_resid_inv_R; 
		vec_t v1(num_data);
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		bool is_zero = U.isZero(0);

		if (is_zero) {
			R = rhs;
		}
		else {
			R = rhs - (chol_ip_cross_cov.transpose() * (chol_ip_cross_cov * U));
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				R.col(i) -= sigma_resid * U.col(i); //parallelization in for loop is much faster
			}
		}
		//Z = P^(-1) R 
		if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
			//D^-1*R
			diag_sigma_resid_inv_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
			//Cmn*D^-1*R
			sigma_cross_cov_diag_sigma_resid_inv_R = sigma_cross_cov_preconditioner.transpose() * diag_sigma_resid_inv_R;
			//P^-1*R using Woodbury Identity
			Z = diag_sigma_resid_inv_R - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov_preconditioner * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_R)));

		}
		else if (cg_preconditioner_type == "none") {
			Z = R;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}

		H = Z;
		for (int j = 0; j < p; ++j) {

			V = (chol_ip_cross_cov.transpose() * (chol_ip_cross_cov * H));

#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) += sigma_resid * H.col(i);
			}
			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();
			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NaN_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
			}

			Z_old = Z;

			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				diag_sigma_resid_inv_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
				//Cmn*D^-1*R
				sigma_cross_cov_diag_sigma_resid_inv_R = sigma_cross_cov_preconditioner.transpose() * diag_sigma_resid_inv_R;
				//P^-1*R using Woodbury Identity
				Z = diag_sigma_resid_inv_R - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov_preconditioner * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_R)));

			}
			else if (cg_preconditioner_type == "none") {
				Z = R;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}

			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();
#pragma omp parallel for schedule(static)
			for (int i = 0; i < t; ++i) {
				Tdiags[i][j] = 1 / a(i) + b_old(i) / a_old(i);
				if (j > 0) {
					Tsubdiags[i][j - 1] = sqrt(b_old(i)) / a_old(i);
				}
			}
			if (early_stop_alg) {
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				//Log::REInfo("CGTridiagFSA stop after %i CG-Iterations.", j + 1);
				return;
			}
		}
		Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise increase 'cg_max_num_it_tridiag'.", p);
	} // end CGTridiagFSA

	/*!
	* \brief Preconditioned conjugate gradient descent to solve Au=rhs when rhs is a Matrix
	*		 A = (C_s + C_nm*(C_m)^(-1)*C_mn) is a symmetric matrix of dimension nxn and a full-scale-approximation for Sigma
	*		 P = diag(C_s) + C_nm*(C_m)^(-1)*C_mn is used as preconditioner.
	* \param sigma_resid Residual Matrix C_s
	* \param sigma_cross_cov Matrix C_mn in Predictive Process Part C_nm*(C_m)^(-1)*C_mn
	* \param chol_ip_cross_cov Cholesky Factor of C_m, the inducing point matrix, times cross-covariance
	* \param rhs Matrix of dimension nx1 on the rhs
	* \param[out] U Approximative solution of the linear system (solution written on input) (must have been declared with the correct n-dimension)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param num_data n-Dimension of the linear system
	* \param t t-Dimension of the linear system
	* \param p Number of conjugate gradient steps
	* \param delta_conv tolerance for checking convergence
	* \param cg_preconditioner_type Type of preconditoner used for the conjugate gradient algorithm
	* \param chol_fact_woodbury_preconditioner Cholesky factor of Matrix C_m + C_mn*D^(-1)*C_nm
	* \param diagonal_approx_inv_preconditioner Diagonal D of residual Matrix C_s
	*/
	template <class T_mat>
	void CGFSA_MULTI_RHS(const T_mat& sigma_resid,
		const den_mat_t& sigma_cross_cov,
		const den_mat_t& sigma_cross_cov_preconditioner,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const den_mat_t& rhs,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const vec_t& diagonal_approx_inv_preconditioner) {
		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, Z(num_data, t), Z_old, H, V(num_data, t), diag_sigma_resid_inv_R, sigma_cross_cov_diag_sigma_resid_inv_R,
			mean_diag_sigma_resid_inv_R, sigma_cross_cov_mean_diag_sigma_resid_inv_R;
		vec_t v1(num_data);
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		bool is_zero = U.isZero(0);

		if (is_zero) {
			R = rhs;
		}
		else {
			R = rhs - (sigma_cross_cov * (chol_fact_sigma_ip.solve(sigma_cross_cov.transpose() * U)));
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				R.col(i) -= sigma_resid * U.col(i); //parallelization in for loop is much faster
			}
		}
		//Z = P^(-1) R 
		if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
			//D^-1*R
			diag_sigma_resid_inv_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
			//Cmn*D^-1*R
			sigma_cross_cov_diag_sigma_resid_inv_R = sigma_cross_cov_preconditioner.transpose() * diag_sigma_resid_inv_R;
			//P^-1*R using Woodbury Identity
			Z = diag_sigma_resid_inv_R - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov_preconditioner * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_R)));

		}
		else if (cg_preconditioner_type == "none") {
			Z = R;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}

		H = Z;

		for (int j = 0; j < p; ++j) {

			V = (sigma_cross_cov * (chol_fact_sigma_ip.solve(sigma_cross_cov.transpose() * H)));

#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) += sigma_resid * H.col(i);
			}
			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();

			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NaN_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
			}

			Z_old = Z;

			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				diag_sigma_resid_inv_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
				//Cmn*D^-1*R
				sigma_cross_cov_diag_sigma_resid_inv_R = sigma_cross_cov_preconditioner.transpose() * diag_sigma_resid_inv_R;
				//P^-1*R using Woodbury Identity
				Z = diag_sigma_resid_inv_R - (diagonal_approx_inv_preconditioner.asDiagonal() * (sigma_cross_cov_preconditioner * chol_fact_woodbury_preconditioner.solve(sigma_cross_cov_diag_sigma_resid_inv_R)));

			}
			else if (cg_preconditioner_type == "none") {
				Z = R;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}

			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

			if (early_stop_alg) {

				return;
			}
		}
		Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise increase 'cg_max_num_it_tridiag'.", p);
	} // end CGFSA_MULTI_RHS

	/*!
	* \brief Preconditioned conjugate gradient descent to solve Au=rhs when rhs is a Matrix
	*		 A = (C_s) is a symmetric matrix of dimension nxn and the residual part of the full-scale-approximation for Sigma
	*		 P = diag(C_s) is used as preconditioner.
	* \param sigma_resid Residual Matrix C_s
	* \param rhs Matrix of dimension nx1 on the rhs
	* \param[out] U Approximative solution of the linear system (solution written on input) (must have been declared with the correct n-dimension)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param num_data n-Dimension of the linear system
	* \param t t-Dimension of the linear system
	* \param p Number of conjugate gradient steps
	* \param delta_conv tolerance for checking convergence
	* \param cg_preconditioner_type Type of preconditoner used for the conjugate gradient algorithm
	* \param diagonal_approx_inv_preconditioner Diagonal D of residual Matrix C_s
	*/
	template <class T_mat>
	void CGFSA_RESID(const T_mat& sigma_resid,
		const den_mat_t& rhs,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const vec_t& diagonal_approx_inv_preconditioner) {
		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, Z(num_data, t), Z_old, H, V(num_data, t), diag_sigma_resid_inv_R, sigma_cross_cov_diag_sigma_resid_inv_R,
			mean_diag_sigma_resid_inv_R, sigma_cross_cov_mean_diag_sigma_resid_inv_R; //NEW V(num_data, t)
		vec_t v1(num_data);
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		bool is_zero = U.isZero(0);

		if (is_zero) {
			R = rhs;
		}
		else {
			R = rhs;
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				R.col(i) -= sigma_resid * U.col(i); //parallelization in for loop is much faster
			}
		}
		//Z = P^(-1) R 
		if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
			//D^-1*R
			Z = diagonal_approx_inv_preconditioner.asDiagonal() * R;

		}
		else if (cg_preconditioner_type == "none") {
			Z = R;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}

		H = Z;

		for (int j = 0; j < p; ++j) {

			V.setZero();

#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) += sigma_resid * H.col(i);
			}
			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();

			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NaN_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
			}

			Z_old = Z;

			if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
				Z = diagonal_approx_inv_preconditioner.asDiagonal() * R;

			}
			else if (cg_preconditioner_type == "none") {
				Z = R;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}

			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

			if (early_stop_alg) {

				return;
			}
		}
		Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise increase 'cg_max_num_it_tridiag'.", p);
	} // end CGFSA_RESID

}
#endif   // GPB_CG_UTILS_