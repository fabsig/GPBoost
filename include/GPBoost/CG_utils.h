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

//#include <functional>
//#include <iostream>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>

#include <LightGBM/utils/log.h>
#include <chrono>
#include <thread> //temp

//#include <GPBoost/likelihoods.h>
//#include <cmath>
//using namespace std;
using LightGBM::Log;

namespace GPBoost {
	/*!
	* \brief Preconditioned conjugate gradient descent to solve Au=rhs when rhs is a vector
	*		 A=(Sigma^-1+W) is a symmetric matrix of dimension nxn, a Vecchia approximation for Sigma^-1,
	*		 Sigma^-1 = B^T D^-1 B, is given, and W is a diagonal matrix. Sigma^-1 = B^T D^-1 B is used as preconditioner.
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^(-1) B ("=" Cholesky factor)
	* \param B_t_D_inv_rm Row-major matrix that contains the product B^T D^(-1). Outsourced in order to reduce the overhead of the function.
	* \param rhs Vector of dimension nx1 on the rhs
	* \param[out] u Approximative solution of the linear system (solution written on input) (must have been declared with the correct n-dimension)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param p Number of conjugate gradient steps
	* \param warm_start If false, u is set to zero at the beginning of the algorithm
	* \param find_mode_it In the first mode-finding iteration (find_mode_it==0) u is set to zero at the beginning of the algorithm
	* \param delta_conv tolerance for checking convergence
	* \param THRESHOLD_ZERO_RHS_CG If the L1-norm of the rhs is below this threshold the CG is not executed and a vector u of 0's is returned
	* \param cg_preconditioner_type Type of preconditoner used for the conjugate gradient algorithm
	* \param chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia Cholesky factor E of matrix EE^T = (D_k + B_k W^(-1) B_k^T)
	* \param chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia Cholesky factor E of matrix EE^T = (I_k + L_k^T W^(-1) L_k)
	* \param B_k_rm Matrix of dimension kxn that contains k rows of the Veccia factor B
	* \param L_k_rm Pivoted Cholseky decomposition of Sigma^(-1): matrix of dimension nxk with rank(L_k) <= piv_chol_rank_ generated with PivotedCholsekyFactorization()
	* \param D_inv_plus_W_B_rm Row-major matrix that contains the product (D^(-1) + W) B used for the preconditioner "sigma_with_W".
	*/
	void CGVecchiaLaplaceVec(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NaN_found,
		int p,
		const bool warm_start,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const string_t cg_preconditioner_type,
		const chol_sp_mat_rm_t& chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia,
		const chol_sp_mat_rm_t& chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia,
		const sp_mat_rm_t& B_k_rm,
		const sp_mat_rm_t& L_k_rm,
		const sp_mat_rm_t& D_inv_plus_W_B_rm);

	/*!
	* \brief Version of CGVecchiaLaplaceVec() that solves (Sigma^-1+W)u=rhs by u=W^(-1)(Sigma+W^(-1))^(-1)Sigma rhs where the preconditioned conjugate 
	*		 gradient descent algorithm is used to approximately solve for (Sigma+W^(-1))^(-1)Sigma rhs. As preconditioner P=(Sigma_L_k Sigma_L_k^T + W^(-1)) is used where 
	*		 Sigma_L_k results from a rank(Sigma_L_k)=k pivoted Cholseky decomposition of the nonapproximated covariance matrix Sigma.
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
	* \param D_inv_B_rm Row-major matrix that contains the product D^-1 B. Outsourced in order to reduce the overhead of the function.
	* \param rhs Vector of dimension nx1 on the rhs
	* \param[out] u Approximative solution of the linear system (solution written on input) (must have been declared with the correct n-dimension)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param p Number of conjugate gradient steps
	* \param warm_start If false, u is set to zero at the beginning of the algorithm
	* \param delta_conv Tolerance for checking convergence of the algorithm
	* \param THRESHOLD_ZERO_RHS_CG If the L1-norm of the rhs is below this threshold the CG is not executed and a vector u of 0's is returned
	* \param chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia Cholesky factor E of matrix EE^T = (I_k + Sigma_L_k^T W^(-1) Sigma_L_k)
	* \param Sigma_L_k Pivoted Cholseky decomposition of Sigma: matrix of dimension nxk with rank(Sigma_L_k) <= piv_chol_rank_ generated in re_model_template.h
	*/
	void CGVecchiaLaplaceVecSigmaplusWinv(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NaN_found,
		int p,
		const bool warm_start,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const chol_den_mat_t chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k);

	/*!
	* \brief Preconditioned conjugate gradient descent in combination with the Lanczos algorithm
	*		 Given the linear system AU=rhs where rhs is a matrix of dimension nxt of t probe column-vectors and 
	*		 A=(Sigma^-1+W) is a symmetric matrix of dimension nxn, a Vecchia approximation for Sigma^-1,
	*		 Sigma^-1 = B^T D^-1 B, is given, and W is a diagonal matrix.
	*		 The function returns t approximative tridiagonalizations T of the symmetric matrix A=QTQ' in vector form (diagonal + subdiagonal of T).
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
	* \param B_t_D_inv_rm Row-major matrix that contains the product B^T D^-1. Outsourced in order to reduce the overhead of the function.
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
	* \param chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia Cholesky factor E of matrix EE^T = (D_k + B_k W^(-1) B_k^T)
	* \param chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia Cholesky factor E of matrix EE^T = (I_k + L_k^T W^(-1) L_k)
	* \param B_k_rm Matrix of dimension kxn that contains k rows of the Veccia factor B
	* \param L_k_rm Pivoted Cholseky decomposition of Sigma^(-1): matrix of dimension nxk with rank(L_k) <= piv_chol_rank_ generated with PivotedCholsekyFactorization()
	* \param D_inv_plus_W_B_rm Row-major matrix that contains the product (D^(-1) + W) B used for the preconditioner "sigma_with_W".
	*/
	void CGTridiagVecchiaLaplace(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
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
		const chol_sp_mat_rm_t& chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia,
		const chol_sp_mat_rm_t& chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia,
		const sp_mat_rm_t& B_k_rm,
		const sp_mat_rm_t& L_k_rm,
		const sp_mat_rm_t& D_inv_plus_W_B_rm);

	/*!
	* \brief Version of CGTridiagVecchiaLaplace() that return t approximative tridiagonalizations T of the symmetric matrix 
	*		 A=(Sigma+W^(-1))=QTQ' in vector form (diagonal + subdiagonal). As preconditioner P=(Sigma_L_k Sigma_L_k^T + W^(-1)) is used where 
	*		 Sigma_L_k results from a rank(Sigma_L_k)=k pivoted Cholseky decomposition of the nonapproximated covariance matrix Sigma.
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
	* \param D_inv_B_rm Row-major matrix that contains the product D^-1 B. Outsourced in order to reduce the overhead of the function.
	* \param rhs Matrix of dimension nxt that contains (column-)probe vectors z_1,...,z_t with Cov[z_i] = P
	* \param[out] Tdiags The diagonals of the t approximative tridiagonalizations of A in vector form (solution written on input)
	* \param[out] Tsubdiags The subdiagonals of the t approximative tridiagonalizations of A in vector form (solution written on input)
	* \param[out] U Approximative solution of the linear system (solution written on input) (must have been declared with the correct nxt dimensions)
	* \param[out] NaN_found Is set to true, if NaN is found in the residual of conjugate gradient algorithm
	* \param num_data n-Dimension of the linear system
	* \param t t-Dimension of the linear system
	* \param p Number of conjugate gradient steps
	* \param delta_conv Tolerance for checking convergence of the algorithm
	* \param chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia Cholesky factor E of matrix EE^T = (I_k + Sigma_L_k^T W^(-1) Sigma_L_k)
	* \param Sigma_L_k Pivoted Cholseky decomposition of Sigma: matrix of dimension nxk with rank(Sigma_L_k) <= piv_chol_rank_ generated in re_model_template.h
	*/
	void CGTridiagVecchiaLaplaceSigmaplusWinv(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const chol_den_mat_t chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k);

	/*!
	* \brief Fills a matrix with Rademacher -1/+1 random numbers
	* \param generator Random number generator
	* \param[out] Z Rademacher matrix. Z need to be defined before calling this function
	*/
	void simRademacher(RNG_t& generator,
		den_mat_t& Z);
	
	/*!
	* \brief Stochastic estimation of log(det(A)) given t approximative tridiagonalizations T of a symmetric matrix A=QTQ' of dimension nxn, 
	*		 where T is given in vector form (diagonal + subdiagonal of T).
	* \param Tdiags The diagonals of the t approximative tridiagonalizations of A in vector form
	* \param Tsubdiags The subdiagonals of the t approximative tridiagonalizations of A in vector form
	* \param ldet[out] Estimation of log(det(A)) (solution written on input)
	* \param num_data Number of data points
	* \param t Number of tridiagonalization matrices T
	*/
	void LogDetStochTridiag(const std::vector<vec_t>& Tdiags,
		const  std::vector<vec_t>& Tsubdiags,
		double& ldet,
		const data_size_t num_data,
		const int t);

	/*!
	* \brief Stochastic trace estimation for tr(A*dW/db_i*B) in vectorized form for all i \in n
	*		 For the vectorization it is assumed that the (i,i)th element of dW/db_i is the only non-zero entry in the matrix.
	* \param A_Z Matrix A of dimension nxn multiplied with the matrix Z = (z_1,...,z_t) of dimension nxt that contains the random probe vectors
	* \param third_deriv Vector of all non-zero entries of -dW/db_i for all i \in n
	* \param B_PI_Z Matrix B of dimension nxn multiplied with the inverse of the preconditioner matrix P (=Cov[z_i]) and the probe vector matrix Z
	* \param tr[out] Vector of stochastic traces tr(A*dW/db_i*B) for all i \in n (solution written on input)
	*/
	void StochTraceVecchiaLaplace(const den_mat_t& A_t_Z,
		const vec_t& third_deriv,
		const den_mat_t& B_PI_Z,
		vec_t& tr);

	/*!
	* \brief Lanczos algorithm with full reorthogonalization to approximately factorize the symmetric matrix (Sigma^-1+W) as Q_k T_k Q_k'
	*		 where T_k is a tridiagonal matrix of dimension kxk and Q_k a matrix of dimension nxk. The diagonal and subdiagonal of T_k is returned in vector form.
	*		 A Vecchia approximation for Sigma^-1 = B^T D^-1 B is given and W is a diagonal matrix.
	* \param diag_W Diagonal of matrix W
	* \param B_rm Row-major matrix B in Vecchia approximation Sigma^-1 = B^T D^-1 B ("=" Cholesky factor)
	* \param D_inv_rm Row-major diagonal matrix D^-1 in Vecchia approximation (only used for preconditioning)
	* \param B_t_D_inv_rm Row-major matrix that contains the product B^T D^-1. Outsourced in order to reduce the overhead of the function.
	* \param b_init Inital column-vector of Q_k (after normalization) of dimension nx1.
	* \param num_data n-Dimension
	* \param[out] Tdiag_k The diagonal of the tridiagonal matrix T_k (solution written on input) (must have been declared with the correct kx1 dimension)
	* \param[out] Tsubdiag_k The subdiagonal of the tridiagonal matrix T_k (solution written on input) (must have been declared with the correct (k-1)x1 dimension)
	* \param[out] Q_k Matrix Q_k = [b_init/||b_init||, q_2, q_3, ...] (solution written on input) (must have been declared with the correct nxk dimensions)
	* \param max_it Rank k of the matrix Q_k and T_k
	* \param tol Tolerance to decide wether reorthogonalization is necessary
	* \param preconditioner_type Type of preconditoner (P) used.
	*		 If "symmetric": Factorize P^(-0.5) (Sigma^-1+W) P^(-0.5*T) where P = B^T (D^(-1) + W) B
	*		 If "asymmetric": Factorize (I + Sigma W)
	*/

	void LanczosTridiagVecchiaLaplace(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const vec_t& b_init,
		const data_size_t num_data,
		vec_t& Tdiag_k,
		vec_t& Tsubdiag_k,
		den_mat_t& Q_k,
		int max_it,
		const double tol,
		const string_t preconditioner_type);

	/*!
	* \brief Pivoted Cholesky factorization according to Habrecht et al. (2012) for the original (nonapproximated) covariance matrix (Sigma)
	* \param cov_f Pointer to function which accesses elements of Sigma (see https://www.geeksforgeeks.org/passing-a-function-as-a-parameter-in-cpp)
	* \param var_f Pointer to fuction which accesses the diagonal of Sigma which equals the marginal variance and is the same for all entries (i,i)
	* \param Sigma_L_k[out] Matrix of dimension nxmax_it such that Sigma_L_k Sigma_L_k^T ~ Sigma and rank(Sigma_L_k) <= max_it (solution written on input)
	* \param max_it Max rank of Sigma_L_k
	* \param num_data n-Dimension
	* \param err_tol Error tolarance - stop the algorithm if the sum of the diagonal elements of the Schur complement is smaller than err_tol
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
		//Log::REInfo("Trace original Covariance Matrix: %g", num_data * diag_ii);
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
		//Log::REInfo("Sigma_L_k.size(): %i", Sigma_L_k.size());
		//Log::REInfo("Sigma_L_k.nonZeros(): %i", Sigma_L_k.nonZeros());
		//Log::REInfo("Trace Pivoted Original Covariance Matrix: %g", (Sigma_L_k.cwiseProduct(Sigma_L_k) * vec_t::Ones(Sigma_L_k.cols())).sum());
	}
}
#endif   // GPB_CG_UTILS_