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
	* \brief Solve equation system with a sparse triangular left-hand side and a sparse right-hand side (Ax=B) using CSparse function cs_spsolve
	* \param A left-hand side
	* \param B right-hand side
	* \param[out] Solution A^(-1)B
	* \param lower true if A is a lower triangular matrix
	*/
	void sp_Lower_sp_RHS_cs_solve(cs* A, const cs* B, sp_mat_t& A_inv_B, bool lower);

	/*!
	* \brief Solve equation system with a sparse triangular left-hand side and a sparse right-hand side (Ax=B) using CSparse function cs_spsolve
	* \param A left-hand side. Sparse Eigen matrix is column-major format
	* \param B right-hand side. Sparse Eigen matrix is column-major format
	* \param[out] Solution A^(-1)B
	* \param lower true if A is a lower triangular matrix
	*/
	void eigen_sp_Lower_sp_RHS_cs_solve(const sp_mat_t& A_const, const sp_mat_t& B_const, sp_mat_t& A_inv_B, bool lower);

	/*!
	* \brief Solve equation system with a sparse triangular left-hand side and a sparse right-hand side (Ax=B) (not used, place-holder for compiler)
	* \param A left-hand side. Sparse Eigen matrix is column-major format
	* \param B right-hand side. Sparse Eigen matrix is column-major format
	* \param[out] Solution A^(-1)B
	* \param lower true if A is a lower triangular matrix
	*/
	void eigen_sp_Lower_sp_RHS_cs_solve(const sp_mat_rm_t& A, const sp_mat_rm_t& B, sp_mat_rm_t& A_inv_B, bool lower);

	/*!
	* \brief Check whether Cholesky factor has a permutation matrix
	* \param chol Cholesky factor
	* \return true if chol has a permutation matrix, false otherwise
	*/
	template <class T_chol, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value>::type* = nullptr >
	bool CholeskyHasPermutation(const T_chol&) {
		return false;
	}
	template <class T_chol, typename std::enable_if <std::is_same<chol_sp_mat_t, T_chol>::value || std::is_same<chol_sp_mat_rm_t, T_chol>::value>::type* = nullptr >
	bool CholeskyHasPermutation(const T_chol& chol) {
		if (chol.permutationP().size() > 0) {
			return true;
		}
		return false;
	}//end CholeskyHasPermutation

	/*!
	* \brief Apply permutation matrix of a Cholesky factor (if it exists)
	* \param chol Cholesky factor
	* \param M Matrix to which the permutation is applied to
	* \param P_M[out] Permuted matrix
	* \param transpose If true, the permutation matrix is first transposed
	*/
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value>::type* = nullptr >
	void ApplyPermutationCholeskyFactor(const chol_den_mat_t&, const T_mat& M, T_mat& P_M, bool) {
		P_M = M;
	}
	template <class T_mat, class T_chol, typename std::enable_if <std::is_same<chol_sp_mat_t, T_chol>::value || std::is_same<chol_sp_mat_rm_t, T_chol>::value>::type* = nullptr >
	void ApplyPermutationCholeskyFactor(const T_chol& chol, const T_mat& M, T_mat& P_M, bool transpose) {
		if (chol.permutationP().size() > 0) {
			if (transpose) {
				P_M = chol.permutationP().transpose() * M;
			}
			else {
				P_M = chol.permutationP() * M;
			}
		}
		else {
			P_M = M;
		}
	}//end ApplyPermutationCholeskyFactor

	/*!
	* \brief Solve equation L * X = R system with a lower triangular left-hand side L
	* \param L Lower triangular left-hand left-hand side
	* \param R Right-hand side
	* \param[out] X Solution L^(-1)R
	* \param transpose If true, L is first transposed, i.e., L^T * X = R is solved
	* Note: T_mat_L = type of matrix L and solution, T_mat_R = type of RHS R, T_mat_X = type of solution X
	*/
	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<den_mat_t, T_mat_L>::value && !std::is_same<vec_t, T_mat_R>::value>::type* = nullptr >
	void TriangularSolve(const T_mat_L& L, const T_mat_R& R, T_mat_X& X, bool transpose) {
		CHECK(L.cols() == R.rows());
		int ncols_R = (int)R.cols();
		int nrows_R = (int)R.rows();
		X = (T_mat_X)(R);
		double* X_ptr = X.data();
		const double* L_ptr = L.data();
		if (transpose) {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < ncols_R; ++j) {
				L_t_solve(L_ptr, nrows_R, X_ptr + j * nrows_R);
			}
		}
		else {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < ncols_R; ++j) {
				L_solve(L_ptr, nrows_R, X_ptr + j * nrows_R);
			}
		}
		//Note 1: Eigen's interval solver is slower (probably because it does not parallelize)
		//X = (den_mat_t)(R);
		//if (transpose) {
		//	L.triangularView<Eigen::UpLoType::Lower>().adjoint().solveInPlace(X);
		//}
		//else {
		//	L.triangularView<Eigen::UpLoType::Lower>().solveInPlace(X);
		//}
		// Note 2: Using dpotri from LAPACK does not work since LAPACK is not installed
		//int info = 0;
		//char* uplo = "L";
		//den_mat_t M = chol_facts_[cluster_i];
		//BLASFUNC(dpotri)(uplo, &ncols_R, L.data(), &nrows_R, &info);
	}//end TriangularSolve (L = den_mat_t && R != vec_t)
	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<den_mat_t, T_mat_L>::value && std::is_same<vec_t, T_mat_R>::value>::type* = nullptr >
	void TriangularSolve(const T_mat_L& L, const T_mat_R& R, T_mat_X& X, bool transpose) {
		CHECK(L.cols() == R.size());
		X = R;
		double* X_ptr = X.data();
		const double* L_ptr = L.data();
		if (transpose) {
			L_t_solve(L_ptr, (int)L.cols(), X_ptr);
		}
		else {
			L_solve(L_ptr, (int)L.cols(), X_ptr);
		}
	}//end TriangularSolve (L = den_mat_t && R = vec_t)
	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<sp_mat_t, T_mat_L>::value && !std::is_same<vec_t, T_mat_R>::value && !std::is_same<den_mat_t, T_mat_R>::value>::type* = nullptr >
	void TriangularSolve(const T_mat_L& L, const T_mat_R& R, T_mat_X& X, bool transpose) {
		CHECK(L.cols() == R.rows());
		int ncols_R = (int)R.cols();
		int nrows_R = (int)R.rows();
		const T_mat_R* R_ptr = &R;//can be both sp_mat_t and sp_mat_rm_t
		const double* val = L.valuePtr();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		std::vector<Triplet_t> triplets;
		triplets.reserve(R.nonZeros() * 5);
		if (transpose) {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < ncols_R; ++j) {
				vec_t R_j = R_ptr->col(j);
				sp_L_t_solve(val, row_idx, col_ptr, nrows_R, R_j.data());
				for (int i = 0; i < nrows_R; ++i) {
					if (std::abs(R_j[i]) > EPSILON_NUMBERS) {
#pragma omp critical
						{
							triplets.emplace_back(i, j, R_j[i]);
						}
					}
				}
			}
		}
		else {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < ncols_R; ++j) {
				vec_t R_j = R_ptr->col(j);
				sp_L_solve(val, row_idx, col_ptr, nrows_R, R_j.data());
				for (int i = 0; i < nrows_R; ++i) {
					if (std::abs(R_j[i]) > EPSILON_NUMBERS) {
#pragma omp critical
						{
							triplets.emplace_back(i, j, R_j[i]);
						}
					}
				}
			}
		}
		X = T_mat_X(R.rows(), R.cols());//can be both sp_mat_t and sp_mat_rm_t
		X.setFromTriplets(triplets.begin(), triplets.end());
		// Note 1: Eigen's interval solver is much slower (probably because it does not parallelize)
		//			The code below runs only if X is sp_mat_t
		//X = R;
		//if (transpose) {
		//	L.triangularView<Eigen::UpLoType::Lower>().adjoint().solveInPlace(X);
		//}
		//else {
		//	L.triangularView<Eigen::UpLoType::Lower>().solveInPlace(X);
		//}
		// Note 2: Alternative version using 'eigen_sp_Lower_sp_RHS_cs_solve' which is based
		//			on the CSparse function 'cs_spsolve' might be faster.
		//			However, this can cause problems with some compilers / OS's (e.g., gcc on Linux).
	}//end TriangularSolve (L = sp_mat_t && R != vec_t && R != den_mat_t)
	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<sp_mat_t, T_mat_L>::value && std::is_same<den_mat_t, T_mat_R>::value>::type* = nullptr >
	void TriangularSolve(const T_mat_L& L, const T_mat_R& R, T_mat_X& X, bool transpose) {
		CHECK(L.cols() == R.rows());
		X = R;
		double* X_ptr = X.data();
		int ncols_R = (int)R.cols();
		int nrows_R = (int)R.rows();
		const double* val = L.valuePtr();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		if (transpose) {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < ncols_R; ++j) {
				sp_L_t_solve(val, row_idx, col_ptr, nrows_R, X_ptr + j * nrows_R);
			}
		}
		else {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < ncols_R; ++j) {
				sp_L_solve(val, row_idx, col_ptr, nrows_R, X_ptr + j * nrows_R);
			}
		}
	}//end TriangularSolve (L = sp_mat_t && R = den_mat_t)
	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<sp_mat_t, T_mat_L>::value && std::is_same<vec_t, T_mat_R>::value>::type* = nullptr >
	void TriangularSolve(const T_mat_L& L, const T_mat_R& R, T_mat_X& X, bool transpose) {
		CHECK(L.cols() == R.size());
		X = R;
		const double* val = L.valuePtr();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		if (transpose) {
			sp_L_t_solve(val, row_idx, col_ptr, (int)L.cols(), X.data());
		}
		else {
			sp_L_solve(val, row_idx, col_ptr, (int)L.cols(), X.data());
		}
	}//end TriangularSolve (L = sp_mat_t && R = vec_t)
	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<sp_mat_rm_t, T_mat_L>::value>::type* = nullptr >
	void TriangularSolve(const T_mat_L& L, const T_mat_R& R, T_mat_X& X, bool transpose) {
		// Note: T_mat_R can be sp_mat_t or sp_mat_rm_t
		const sp_mat_t L_cm = sp_mat_t(L);
		TriangularSolve<sp_mat_t, T_mat_R, T_mat_X>(L_cm, R, X, transpose);
	}//end TriangularSolve (L = sp_mat_rm_t)

	//old version that creates a large dense matrix and can thus result in memory problems
//	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<sp_mat_t, T_mat_L>::value && !std::is_same<vec_t, T_mat_R>::value>::type* = nullptr >
//	void TriangularSolve(const sp_mat_t& L, const sp_mat_t& R, sp_mat_t& X, bool transpose) {
//		const double* val = L.valuePtr();
//		const int* row_idx = L.innerIndexPtr();
//		const int* col_ptr = L.outerIndexPtr();
//		if (transpose) {
//			den_mat_t U_inv_dens = den_mat_t(R);
//#pragma omp parallel for schedule(static)
//			for (int j = 0; j < R.cols(); ++j) {
//				sp_L_t_solve(val, row_idx, col_ptr, (int)L.cols(), U_inv_dens.data() + j * R.rows());
//			}
//			X = U_inv_dens.sparseView();
//		}
//		else {
//			den_mat_t L_inv_dens = den_mat_t(R);
//#pragma omp parallel for schedule(static)
//			for (int j = 0; j < R.cols(); ++j) {
//				sp_L_solve(val, row_idx, col_ptr, (int)L.cols(), L_inv_dens.data() + j * R.rows());
//			}
//			X = L_inv_dens.sparseView();
//		}
//	}//end TriangularSolve (sp_mat_t)


	/*!
	* \brief Solve a triangular linear system L * X = R given a lower Cholesky factor
	*		Note: In contrast to 'TriangularSolve', this also applies a permutation if 'chol' has one
	* \param chol Lower Cholesky factor
	* \param R RHS
	* \param[out] X Solution L^(-1)R
	* \param transpose If true, the lower Cholesky factor is first transposed, i.e., L^T * X = R is solved
	* Note: T_chol = type of Cholesky decomposition chol, T_chol_mat = type of cholesky factor matrix in chol, T_mat_R = type of RHS R, T_mat_X = type of solution X
	*/
	template <class T_chol, class T_chol_mat, class T_mat_R, class T_mat_X, typename std::enable_if <!std::is_same<chol_den_mat_t, T_chol>::value || !std::is_same<sp_mat_t, T_mat_R>::value>::type* = nullptr >
	void TriangularSolveGivenCholesky(const T_chol& chol, const T_mat_R& R, T_mat_X& X, bool transpose) {
		// Covers all types except if chol is chol_den_mat_t and R is sp_mat_t (it would also cover chol chol_sp_mat_t and R den_mat_t which does not compile, but this is never used)
		if (transpose) {
			TriangularSolve<T_chol_mat, T_mat_R, T_mat_X>(chol.CholFactMatrix(), R, X, true);
			if (CholeskyHasPermutation<T_chol>(chol)) {
				ApplyPermutationCholeskyFactor<T_mat_X, T_chol>(chol, X, X, true);
			}
		}
		else {
			if (CholeskyHasPermutation<T_chol>(chol)) {
				ApplyPermutationCholeskyFactor<T_mat_X, T_chol>(chol, R, X, false);
				TriangularSolve<T_chol_mat, T_mat_R, T_mat_X>(chol.CholFactMatrix(), X, X, false);
			}
			else {
				TriangularSolve<T_chol_mat, T_mat_R, T_mat_X>(chol.CholFactMatrix(), R, X, false);
			}
		}
	}
	template <class T_chol, class T_chol_mat, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value && std::is_same<sp_mat_t, T_mat_R>::value>::type* = nullptr >
	void TriangularSolveGivenCholesky(const T_chol& chol, const T_mat_R& R, T_mat_X& X, bool transpose) {
		// If chol is chol_den_mat_t and the R is sp_mat_t, then the RHS needs to converted to den_mat_t
		den_mat_t R_den = (den_mat_t)(R);
		TriangularSolveGivenCholesky<chol_den_mat_t, T_chol_mat, den_mat_t, den_mat_t>(chol, R_den, X, transpose);
	}

	/*!
	* \brief Solve a linear system L*L^T * X = R given a lower Cholesky factor
	* \param chol Lower Cholesky factor
	* \param R RHS
	* \param[out] X Solution L^(-T)L^(-1)R
	* Note: T_chol = type of Cholesky decomposition chol, T_chol_mat = type of cholesky factor matrix in chol, T_mat_R = type of RHS R, T_mat_X = type of solution X
	*/
	template <class T_chol, class T_chol_mat, class T_mat_R, class T_mat_X, typename std::enable_if <!std::is_same<chol_den_mat_t, T_chol>::value || !std::is_same<sp_mat_t, T_mat_R>::value>::type* = nullptr >
	void SolveGivenCholesky(const T_chol& chol, const T_mat_R& R, T_mat_X& X) {
		// Covers all types except if chol is chol_den_mat_t and R is sp_mat_t (it would also cover chol chol_sp_mat_t and R den_mat_t which does not compile, but this is never used)
		TriangularSolveGivenCholesky<T_chol, T_chol_mat, T_mat_R, T_mat_X>(chol, R, X, false);
		TriangularSolveGivenCholesky<T_chol, T_chol_mat, T_mat_R, T_mat_X>(chol, X, X, true);
	}
	template <class T_chol, class T_chol_mat, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value && std::is_same<sp_mat_t, T_mat_R>::value>::type* = nullptr >
	void SolveGivenCholesky(const T_chol& chol, const T_mat_R& R, T_mat_X& X) {
		// If chol is chol_den_mat_t and the R is sp_mat_t, then the RHS needs to converted to den_mat_t
		den_mat_t R_den = (den_mat_t)(R);
		SolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol, R_den, X);
	}
	// ALternative version using Eigen's interval solver
	//template <class T_mat, class T_chol, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value || std::is_same<sp_mat_t, T_mat>::value>::type* = nullptr >
	//void SolveGivenCholesky(const T_chol& chol, const T_mat& R, T_mat& X) {
	//	X = chol.solve(R);
	//}
	//template <class T_mat, class T_chol, typename std::enable_if <std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	//void SolveGivenCholesky(const chol_sp_mat_rm_t& chol, const sp_mat_rm_t& R, sp_mat_rm_t& X) {
	//	sp_mat_t R_cm = sp_mat_t(R);
	//	sp_mat_t X_cm = chol.solve(R_cm);
	//	X = sp_mat_rm_t(X_cm);
	//}

	/*!
	* \brief Calculate L.transpose() * L only at non-zero entries for a given sparsiy pattern for sparse matrices (for dense matrices, all entries are calculated)
	* \param L Matrix L
	* \param LtL[out] Matrix which contains a sparsity pattern^and on which L.transpose() * L is calculated at the non-zero entries
	*/
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void CalcLtLGivenSparsityPattern(const T_mat& L, T_mat& LtL) {
#pragma omp parallel for schedule(static)
		for (int k = 0; k < LtL.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(LtL, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				if (i == j) {
					it.valueRef() = (L.col(i)).dot(L.col(j));
				}
				else if (i < j) {
					it.valueRef() = (L.col(i)).dot(L.col(j));
					LtL.coeffRef(j, i) = it.value();
				}
			}
		}
	}//end CalcLtLGivenSparsityPattern (sparse)
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void CalcLtLGivenSparsityPattern(const den_mat_t& L, den_mat_t& LtL) {
		LtL = L.transpose() * L;
	}//end CalcLtLGivenSparsityPattern (dense)

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
