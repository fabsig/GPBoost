/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 Pascal Kuendig and Fabio Sigrist. All rights reserved.
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
	*		 Sigma^-1 = B^T D^-1 B, is given, and W is a diagonal matrix. P = B^T (D^-1 + W) B is used as preconditioner.
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
	* \param D_inv_plus_W_B_rm Row-major matrix that contains the product (D^(-1) + W) B used for the preconditioner "Sigma_inv_plus_BtWB".
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
		const sp_mat_rm_t& D_inv_plus_W_B_rm);

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

	/*!
	* \brief Preconditioned conjugate gradient descent in combination with the Lanczos algorithm.
	*		 A linear system A U = rhs is solved, where the rhs is a matrix of dimension nxt of t random column-vectors and 
	*		 A = (Sigma^-1 + W) is a symmetric matrix of dimension nxn. P = B^T (D^-1 + W) B is used as preconditioner.
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
	* \param D_inv_plus_W_B_rm Row-major matrix that contains the product (D^(-1) + W) B used for the preconditioner "Sigma_inv_plus_BtWB".
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
		const sp_mat_rm_t& D_inv_plus_W_B_rm);

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

	/*!
	* \brief Fills a given matrix with standard normal RV's.
	* \param generator Random number generator
	* \param[out] R Matrix of random vectors (r_1, r_2, r_3, ...), where r_i is of dimension n & Cov(r_i) = I (must have been declared with the correct dimensions)
	*/
	void GenRandVecTrace(RNG_t& generator,
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
	* \brief Pivoted Cholesky factorization according to Habrecht et al. (2012) for the original (nonapproximated) covariance matrix (Sigma)
	* \param cov_f Pointer to function which accesses elements of Sigma (see https://www.geeksforgeeks.org/passing-a-function-as-a-parameter-in-cpp)
	* \param var_f Pointer to fuction which accesses the diagonal of Sigma which equals the marginal variance and is the same for all entries (i,i)
	* \param Sigma_L_k[out] Matrix of dimension nxmax_it such that Sigma_L_k Sigma_L_k^T ~ Sigma and rank(Sigma_L_k) <= max_it (solution written on input)
	* \param max_it Max rank of Sigma_L_k
	* \param num_data n-Dimension
	* \param err_tol Error tolerance - stop the algorithm if the sum of the diagonal elements of the Schur complement is smaller than err_tol
	*/

	template<typename T_mat>
	void PivotedCholsekyFactorizationSigma(
		RECompBase<T_mat>* re_comp,
		den_mat_t& Sigma_L_k,
		int max_it,
		const data_size_t num_data,
		const double err_tol) {

		int m = 0;
		int i, pi_m_old;
		double err, L_jm;
		vec_t diag(num_data), L_row_m;
		vec_int_t pi(num_data);
		max_it = std::min(max_it, num_data);
		Sigma_L_k.resize(num_data, max_it);
		Sigma_L_k.setZero();
		double diag_ii = re_comp->GetZSigmaZtii();

		for (int h = 0; h < num_data; ++h) {
			pi(h) = h;
			//The diagonal of the covariance matrix equals the marginal variance and is the same for all entries (i,i). 
			diag(h) = diag_ii;
		}
		err = diag.lpNorm<1>();

		while (m == 0 || (m < max_it && err > err_tol)) {

			diag(pi.tail(num_data - m)).maxCoeff(&i);
			i += m;

			pi_m_old = pi(m);
			pi(m) = pi(i);
			pi(i) = pi_m_old;

			//L[(m+1):n,m]
			if ((m + 1) < num_data) {

				if (m > 0) {
					L_row_m = Sigma_L_k.row(pi(m)).transpose();
				}

				for (int j = m + 1; j < num_data; ++j) {

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
				err = diag(pi.tail(num_data - (m + 1))).lpNorm<1>();
			}

			//L[m,m] - Update post L[(m+1):n,m] to be able to multiply with L[m,] beforehand
			Sigma_L_k(pi(m), m) = sqrt(diag(pi(m)));
			m = m + 1;
		}
	}
}
#endif   // GPB_CG_UTILS_