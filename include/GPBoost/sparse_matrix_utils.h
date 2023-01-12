/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_SPARSE_MAT_H_
#define GPB_SPARSE_MAT_H_
#include <memory>
#include <GPBoost/type_defs.h>
#include <LightGBM/utils/log.h>
#include <GPBoost/utils.h>
using LightGBM::Log;

extern "C" {
#include <cs.h>
}

namespace GPBoost {

	/*!
	* \brief Solve equation system with a dense lower triangular matrix as left-hand side (Lx=b)
	* \param val Values of lower triangular matrix L in column-major format
	* \param ncol Number of columns
	* \param[out] x Right-hand side vector (solution written on input)
	*/
	void L_solve(const double* val, const int ncol, double* x);

	/*!
	* \brief Solve equation system with the transpose of a dense lower triangular matrix as left-hand side (L'x=b)
	* \param val Values of lower triangular matrix L in column-major format
	* \param ncol Number of columns
	* \param[out] x Right-hand side vector (solution written on input)
	*/
	void L_t_solve(const double* val, const int ncol, double* x);

	/*!
	* \brief Solve equation system with a sparse lower triangular matrix in column-major format as left-hand side (Lx=b)
	* \param val Values of sparse lower triangular matrix L
	* \param row_idx Row indices corresponding to the values ('InnerIndices' in Eigen)
	* \param col_ptr val indexes where each column starts ('OuterStarts' in Eigen)
	* \param ncol Number of columns
	* \param[out] x Right-hand side vector (solution written on input)
	*/
	void sp_L_solve(const double* val, const int* row_idx, const int* col_ptr, const int ncol, double* x);

	/*!
	* \brief Solve equation system with the transpose of a sparse lower triangular matrix in column-major format as left-hand side: (L'x=b)
	* \param val Values of sparse lower triangular matrix L
	* \param row_idx Row indices corresponding to the values ('InnerIndices' in Eigen)
	* \param col_ptr val indexes where each column starts ('OuterStarts' in Eigen)
	* \param ncol Number of columns
	* \param[out] x Right-hand side vector (solution written on input)
	*/
	void sp_L_t_solve(const double* val, const int* row_idx, const int* col_ptr, const int ncol, double* x);

	/*!
	* \brief Solve equation system with a sparse left-hand side and a sparse right-hand side (Ax=B) using CSparse function cs_spsolve
	* \param A left-hand side
	* \param B right-hand side
	* \param[out] Solution A^(-1)B
	* \param lower true if A is a lower triangular matrix
	*/
	void sp_Lower_sp_RHS_cs_solve(cs* A, cs* B, sp_mat_t& A_inv_B, bool lower);

	/*!
	* \brief Solve equation system with a sparse left-hand side and a sparse right-hand side (Ax=B) using CSparse function cs_spsolve
	* \param A left-hand side. Sparse Eigen matrix is column-major format
	* \param B right-hand side. Sparse Eigen matrix is column-major format
	* \param[out] Solution A^(-1)B
	* \param lower true if A is a lower triangular matrix
	*/
	void eigen_sp_Lower_sp_RHS_cs_solve(sp_mat_t& A, sp_mat_t& B, sp_mat_t& A_inv_B, bool lower);

	/*!
	* \brief Solve equation system with a sparse left-hand side and a sparse right-hand side (Ax=B) (not used, place-holder for compiler)
	* \param A left-hand side. Sparse Eigen matrix is column-major format
	* \param B right-hand side. Sparse Eigen matrix is column-major format
	* \param[out] Solution A^(-1)B
	* \param lower true if A is a lower triangular matrix
	*/
	void eigen_sp_Lower_sp_RHS_cs_solve(sp_mat_rm_t& A, sp_mat_rm_t& B, sp_mat_rm_t& A_inv_B, bool lower);

	/*!
	* \brief Solve equation system with a sparse left-hand side and a sparse right-hand side (Ax=B)
	* \param A left-hand side. Sparse Eigen matrix is column-major format
	* \param B right-hand side. Sparse Eigen matrix is column-major format
	* \param[out] Solution A^(-1)B
	* \param lower true if A is a lower triangular matrix
	*/
	template <class T_mat, class T_mat2, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value>::type* = nullptr >
	void eigen_sp_Lower_sp_RHS_solve(sp_mat_t& A, sp_mat_t& B, sp_mat_t& A_inv_B, bool lower) {
		//Convert Eigen matrices to correct format
		A.makeCompressed();
		B.makeCompressed();
		const double* val = A.valuePtr();
		const int* row_idx = A.innerIndexPtr();
		const int* col_ptr = A.outerIndexPtr();
		if (lower) {
			den_mat_t L_inv_dens = den_mat_t(B);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < B.cols(); ++j) {
				sp_L_solve(val, row_idx, col_ptr, (int)A.cols(), L_inv_dens.data() + j * B.rows());
			}
			A_inv_B = L_inv_dens.sparseView();
		}
		else {
			den_mat_t U_inv_dens = den_mat_t(B);
#pragma omp parallel for schedule(static)
			for (int j = 0; j < B.cols(); ++j) {
				sp_L_t_solve(val, row_idx, col_ptr, (int)A.cols(), U_inv_dens.data() + j * B.rows());
			}
			A_inv_B = U_inv_dens.sparseView();
		}
	}//end eigen_sp_Lower_sp_RHS_solve (sp_mat_t)
	template <class T_mat, class T_mat2, typename std::enable_if <std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void eigen_sp_Lower_sp_RHS_solve(sp_mat_rm_t& A, T_mat2& B, sp_mat_rm_t& A_inv_B, bool lower) {
		sp_mat_t A_cm = sp_mat_t(A);
		sp_mat_t A_inv_B_cm = (sp_mat_t)(B);
		eigen_sp_Lower_sp_RHS_solve<sp_mat_t, sp_mat_t>(A_cm, A_inv_B_cm, A_inv_B_cm, lower);
		A_inv_B = sp_mat_rm_t(A_inv_B_cm);
	}//end eigen_sp_Lower_sp_RHS_solve (sp_mat_rm_t)

//version that avoids creation of a large dense matrix
//	template <class T_mat, class T_mat2, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value>::type* = nullptr >
//	void eigen_sp_Lower_sp_RHS_solve(sp_mat_t& A, sp_mat_t& B, sp_mat_t& A_inv_B, bool lower) {
//		//Convert Eigen matrices to correct format
//		A.makeCompressed();
//		B.makeCompressed();
//		const double* val = A.valuePtr();
//		const int* row_idx = A.innerIndexPtr();
//		const int* col_ptr = A.outerIndexPtr();
//		std::vector<Triplet_t> triplets;
//		triplets.reserve(B.nonZeros() * 10);
//		if (lower) {
//#pragma omp parallel for schedule(static)//NEW
//			for (int j = 0; j < B.cols(); ++j) {
//				vec_t B_j = B.col(j);
//				sp_L_solve(val, row_idx, col_ptr, (int)A.cols(), B_j.data());
//				for (int i = 0; i < B.rows(); ++i) {
//					if (std::abs(B_j[i]) > EPSILON_NUMBERS) {
//#pragma omp critical
//						{
//							triplets.emplace_back(i, j, B_j[i]);
//						}
//					}
//				}
//			}
//			A_inv_B = sp_mat_t(B.rows(), B.cols());
//			A_inv_B.setFromTriplets(triplets.begin(), triplets.end());
//		}
//		else {
//#pragma omp parallel for schedule(static)//NEW
//			for (int j = 0; j < B.cols(); ++j) {
//				vec_t B_j = B.col(j);
//				sp_L_t_solve(val, row_idx, col_ptr, (int)A.cols(), B_j.data());
//				for (int i = 0; i < B.rows(); ++i) {
//					if (std::abs(B_j[i]) > EPSILON_NUMBERS) {
//#pragma omp critical
//						{
//							triplets.emplace_back(i, j, B_j[i]);
//						}
//					}
//				}
//			}
//			A_inv_B = sp_mat_t(B.rows(), B.cols());
//			A_inv_B.setFromTriplets(triplets.begin(), triplets.end());
//		}
//	}//end eigen_sp_Lower_sp_RHS_solve (sp_mat_t)

	/*!
	* \brief Caclulate triangular solve when a lower Cholesky factor is given
	* \param chol Cholesky factor
	* \param[out] H Right-hand side matrix and solution
	* \param lower true if the Cholesky factor is a lower triangular matrix
	*/
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void TriangularSolveInPlaceGivenCholesky(const chol_den_mat_t& chol, den_mat_t& H) {
		CHECK(chol.matrixL().cols() == H.rows());
		chol.matrixL().solveInPlace(H);//Using Eigen's internal solver
//		//Not using Eigen's internal solver
//		const den_mat_t L = chol.matrixL();
//		const double* L_ptr = L.data();
//		double* H_ptr = H.data();
//		int ncols_L = (int)L.cols();
//		int ncols_H = (int)H.cols();
//#pragma omp parallel for schedule(static)
//		for (int j = 0; j < ncols_H; ++j) {
//			L_solve(L_ptr, ncols_L, H_ptr + j * ncols_L);
//			// if L is not lower-triangular, use L_t_solve(L_ptr, ncols_L, H_ptr + j * ncols_L);
//		}
	}
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value>::type* = nullptr >
	void TriangularSolveInPlaceGivenCholesky(const T_chol& chol, sp_mat_t& H) {
		CHECK(chol.matrixL().cols() == H.rows());
		chol.matrixL().solveInPlace(H);//Using Eigen's internal solver
		////Not using Eigen's internal solver
		//sp_mat_t L = (sp_mat_t)(chol.matrixL());
		//eigen_sp_Lower_sp_RHS_solve<sp_mat_t, sp_mat_t>(L, H, H, true);
	}
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void TriangularSolveInPlaceGivenCholesky(const T_chol& chol, sp_mat_rm_t& H) {
		sp_mat_t LInvH_cm = sp_mat_t(H);//need to convert to 'sp_mat_t' as Eigen produces wrong results when having 'sp_mat_rm_t' as RHS in 'solveInPlace'
		TriangularSolveInPlaceGivenCholesky<sp_mat_t, T_chol>(chol, LInvH_cm);
		H = sp_mat_rm_t(LInvH_cm);
	}

	/*!
	* \brief Solve a linear system given a Cholesky factor using Eigen's internal solver
	*	Note: This method is needed for 'sp_mat_rm_t' due to the fact that Eigen does not accept column-major RHS for row-major sparse matrices and corresponding Cholesky factors; see https://forum.kde.org/viewtopic.php?f=74&t=108773 (as of 06.01.2023)
	* \param chol Cholesky factor
	* \param y RHS
	* \param[out] x Solution
	*/
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value || std::is_same<sp_mat_t, T_mat>::value>::type* = nullptr >
	void SolveGivenCholesky(const T_chol& chol, const T_mat& y, T_mat& x) {
		x = chol.solve(y);
	}
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SolveGivenCholesky(const chol_sp_mat_rm_t& chol, const sp_mat_rm_t& y, sp_mat_rm_t& x) {
		sp_mat_t y_cm = sp_mat_t(y);
		sp_mat_t x_cm = chol.solve(y_cm);
		x = sp_mat_rm_t(x_cm);
	}

	/*!
	* \brief Solve a linear system with a sparse (column-major) RHS given a Cholesky factor
	*		Note: this is just a wrapper around 'SolveGivenCholesky' with appropriate matrix-format conversion
	* \param chol Cholesky factor
	* \param y RHS
	* \param[out] x Solution
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SolveWithSparseRHSGivenCholesky(const chol_den_mat_t& chol, const sp_mat_t& y, den_mat_t& x) {
		den_mat_t y_den = (den_mat_t)(y);
		SolveGivenCholesky<den_mat_t, chol_den_mat_t>(chol, y_den, x);
	}
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value>::type* = nullptr >
	void SolveWithSparseRHSGivenCholesky(const chol_sp_mat_t& chol, const sp_mat_t& y, sp_mat_t& x) {
		SolveGivenCholesky<sp_mat_t, chol_sp_mat_t>(chol, y, x);
	}
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SolveWithSparseRHSGivenCholesky(const chol_sp_mat_rm_t& chol, const sp_mat_t& y, sp_mat_rm_t& x) {
		sp_mat_t x_cm;
		SolveGivenCholesky<sp_mat_t, chol_sp_mat_rm_t>(chol, y, x_cm);
		x = sp_mat_rm_t(x_cm);
	}

	/*!
	* \brief Apply permutation matrix of Cholesky factor (if it exists)
	* \param chol Cholesky factor
	* \param M[out] Matrix to which the permutation is applied to
	*/
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value>::type* = nullptr >
	void ApplyPermutationCholeskyFactor(const chol_den_mat_t&, T_mat&, bool) {
	}
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<chol_sp_mat_t, T_chol>::value || std::is_same<chol_sp_mat_rm_t, T_chol>::value>::type* = nullptr >
	void ApplyPermutationCholeskyFactor(const T_chol& chol, T_mat& M, bool transpose) {
		if (chol.permutationP().size() > 0) {
			if (transpose) {
				M = chol.permutationP().transpose() * M;
			}
			else {
				M = chol.permutationP() * M;
			}
		}
	}

	/*!
	* \brief Multiplies a vector v by the (transposed) incidence matrix Zt when only indices that indicate to which random effect every data point is related are given
	* \param num_data Number of data points
	* \param num_re Number of random effects
	* \param random_effects_indices_of_data Indices that indicate to which random effect every data point is related
	* \param v Vector which is to be multiplied by Zt
	* \param[out] ZtV Vector Zt * v
	* \param initialize_zero If true, ZtV is initialized to zero. Otherwise, the result is added to it
	*/
	void CalcZtVGivenIndices(const data_size_t num_data,
		const data_size_t num_re,
		const data_size_t* const random_effects_indices_of_data,
		const vec_t& v,
		vec_t& ZtV,
		bool initialize_zero);

}  // namespace GPBoost

#endif   // GPB_SPARSE_MAT_H_
