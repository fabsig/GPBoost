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

#include <iterator>
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
	* \brief Calculate the diagonal of L^(-1) diag(d) L^(-T) for a sparse lower triangular matrix L without storing L^(-1),
	*		which is usually not sparse. The columns of L^(-1) are calculated one after another, which requires O(n^2) operations
	* \param L Sparse lower triangular matrix (compressed, column-major format, with the diagonal as first entry of every column)
	* \param d Vector d
	* \param[out] diag Diagonal of L^(-1) diag(d) L^(-T)
	*/
	void CalcDiagLInvDLInvT(const sp_mat_t& L, const vec_t& d, vec_t& diag);

	/*!
	* \brief Calculate the squared norms of the columns of L^(-1) R for a sparse lower triangular matrix L and a sparse matrix R
	*		without storing L^(-1) R. The columns are calculated one after another, starting at the first non-zero row of a column of R
	* \param L Sparse lower triangular matrix (compressed, column-major format, with the diagonal as first entry of every column)
	* \param R Sparse right-hand side
	* \param[out] norms Squared norms of the columns of L^(-1) R
	*/
	void CalcColSquaredNormsLInvR(const sp_mat_t& L, const sp_mat_t& R, vec_t& norms);

	/*!
	* \brief Calculate the squared norms of the columns of L^(-1) R for a sparse Cholesky factor L and a sparse matrix R without
	*		storing L^(-1) R. The non-zeros of L^(-1) r are on the paths from the non-zeros of r to the root of the elimination
	*		tree of L, and only these entries are calculated. This requires that the sparsity pattern of L contains the fill-in,
	*		as it is the case for a Cholesky factor (but not, e.g., for an incomplete Cholesky factor)
	* \param L Sparse Cholesky factor (compressed, column-major format, with the diagonal as first entry of every column)
	* \param R Sparse right-hand side
	* \param[out] norms Squared norms of the columns of L^(-1) R
	*/
	void CalcColSquaredNormsCholFactorInvR(const sp_mat_t& L, const sp_mat_t& R, vec_t& norms);

	/*!
	* \brief Selected inversion with the Takahashi equations: calculate the entries of (L L^T)^(-1) on the sparsity pattern of a
	*		lower triangular Cholesky factor L. The pattern of a Cholesky factor contains the fill-in of the factorization,
	*		and the recursion therefore only needs entries on this pattern. The cost is of the order of a numeric factorization
	* \param L Lower triangular Cholesky factor (compressed, column-major format, with sorted row indices and the diagonal as
	*		first entry of every column)
	* \param[out] Z_lower Lower triangle of (L L^T)^(-1) on the sparsity pattern of L
	* \param max_num_rows_dense The entries of the inverse that are needed for a column are gathered into a dense matrix if the
	*		column has at most this many non-zeros below the diagonal
	*/
	void SelectedInversionCholesky(const sp_mat_t& L, sp_mat_t& Z_lower, int max_num_rows_dense = 4096);

	/*!
	* \brief Entries of the inverse of a symmetric positive definite matrix A given a sparse Cholesky factorization
	*		P A P^T = L L^T. The entries on the sparsity pattern of L + L^T, which include the diagonal and all entries
	*		on the sparsity pattern of A, are calculated once with 'SelectedInversionCholesky'. Other entries are
	*		calculated with triangular solves
	*/
	class SparseInverseEntries {
	public:
		/*!
		* \param L Lower triangular Cholesky factor of P A P^T (compressed, column-major format, with the diagonal as
		*		first entry of every column). It has to exist as long as this object is used
		* \param perm_indices Indices of the permutation P, i.e., row i of A is row perm_indices[i] of P A P^T. If it is
		*		empty, there is no permutation
		*/
		SparseInverseEntries(const sp_mat_t& L, const Eigen::VectorXi& perm_indices);
		/*! \brief A temporary factor, e.g. a converted row-major factor, would not exist as long as this object */
		SparseInverseEntries(sp_mat_t&& L, const Eigen::VectorXi& perm_indices) = delete;

		/*! \brief Diagonal of A^(-1) */
		vec_t Diagonal() const;

		/*!
		* \brief Write the entries of A^(-1) on the non-zeros of a sparse matrix M
		* \param[out] M Matrix whose non-zero entries are overwritten
		*/
		void OnSparsityPattern(sp_mat_t& M) const;

		/*!
		* \brief Calculate the diagonal of Z A^(-1) Z^T for a sparse matrix Z
		* \param Zt Transpose of Z
		* \return Diagonal of Z A^(-1) Z^T
		*/
		vec_t DiagZAinvZt(const sp_mat_t& Zt) const;

	private:
		/*! \brief Position of the entry (i,j) of A^(-1) in 'Z_lower_' for indices of A, -1 if it is not on the sparsity pattern of L */
		int Position(int i, int j) const;
		/*!
		* \brief Calculate entries of A^(-1) with triangular solves. The entries are grouped by their column such that every
		*		column of A^(-1) is calculated only once
		* \param rows Row indices of the entries (indices of A)
		* \param cols Column indices of the entries (indices of A)
		* \param[out] values Entries of A^(-1)
		*/
		void EntriesBySolves(const std::vector<int>& rows, const std::vector<int>& cols, std::vector<double>& values) const;

		/*! \brief Cholesky factor L with sorted row indices */
		const sp_mat_t* L_;
		/*! \brief Copy of L with sorted row indices, used only if the row indices of the given factor are not sorted */
		sp_mat_t L_sorted_;
		/*! \brief Lower triangle of (L L^T)^(-1) on the sparsity pattern of L */
		sp_mat_t Z_lower_;
		/*! \brief perm_[i] is the row of P A P^T that corresponds to row i of A */
		std::vector<int> perm_;
	};

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
	* \brief Whether a Cholesky factorization is one of a sparse matrix, i.e. one that reorders
	*		its rows and columns and therefore carries a permutation matrix
	*/
	template <class T_chol>
	struct is_sparse_chol : public std::integral_constant<bool,
		std::is_same<chol_sp_mat_t, T_chol>::value ||
		std::is_same<chol_sp_mat_rm_t, T_chol>::value ||
		std::is_same<chol_cholmod_sp_mat_t, T_chol>::value ||
		std::is_same<chol_cholmod_sp_mat_rm_t, T_chol>::value> {};

	/*!
	* \brief Check whether Cholesky factor has a permutation matrix
	* \param chol Cholesky factor
	* \return true if chol has a permutation matrix, false otherwise
	*/
	template <class T_chol, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value>::type* = nullptr >
	bool CholeskyHasPermutation(const T_chol&) {
		return false;
	}
	template <class T_chol, typename std::enable_if <is_sparse_chol<T_chol>::value>::type* = nullptr >
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
	template <class T_mat, class T_chol, typename std::enable_if <is_sparse_chol<T_chol>::value>::type* = nullptr >
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
	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<sp_mat_t, T_mat_L>::value && !std::is_same<vec_t, T_mat_R>::value && !std::is_same<den_mat_t, T_mat_R>::value && !std::is_same<den_mat_t, T_mat_X>::value>::type* = nullptr >
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
		//every thread collects its triplets in its own vector and they are concatenated afterwards
		int num_threads;
#ifdef _OPENMP
		num_threads = omp_get_max_threads();
#else
		num_threads = 1;
#endif
		std::vector<std::vector<Triplet_t>> triplets_private(num_threads);
#pragma omp parallel
		{
			int thread_nb;
#ifdef _OPENMP
			thread_nb = omp_get_thread_num();
#else
			thread_nb = 0;
#endif
			std::vector<Triplet_t>& triplets_local = triplets_private[thread_nb];
			triplets_local.reserve((size_t)(R.nonZeros() * 5) / (size_t)num_threads + 64);
			if (transpose) {
#pragma omp for schedule(static)
				for (int j = 0; j < ncols_R; ++j) {
					vec_t R_j = R_ptr->col(j);
					sp_L_t_solve(val, row_idx, col_ptr, nrows_R, R_j.data());
					for (int i = 0; i < nrows_R; ++i) {
						if (std::abs(R_j[i]) > EPSILON_NUMBERS) {
							triplets_local.emplace_back(i, j, R_j[i]);
						}
					}
				}
			}
			else {
#pragma omp for schedule(static)
				for (int j = 0; j < ncols_R; ++j) {
					vec_t R_j = R_ptr->col(j);
					sp_L_solve(val, row_idx, col_ptr, nrows_R, R_j.data());
					for (int i = 0; i < nrows_R; ++i) {
						if (std::abs(R_j[i]) > EPSILON_NUMBERS) {
							triplets_local.emplace_back(i, j, R_j[i]);
						}
					}
				}
			}
		}//end #pragma omp parallel
		for (auto& triplets_local : triplets_private) {
			triplets.insert(triplets.end(), std::make_move_iterator(triplets_local.begin()), std::make_move_iterator(triplets_local.end()));
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
	}//end TriangularSolve (L = sp_mat_t && R != vec_t && R != den_mat_t && X != den_mat_t)
	template <class T_mat_L, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<sp_mat_t, T_mat_L>::value && !std::is_same<vec_t, T_mat_R>::value && !std::is_same<den_mat_t, T_mat_R>::value && std::is_same<den_mat_t, T_mat_X>::value>::type* = nullptr >
	void TriangularSolve(const T_mat_L& L, const T_mat_R& R, T_mat_X& X, bool transpose) {
		// Note: this code has not beed tested as often the version below with R == den_mat_t && X == den_mat_t is called since 'ApplyPermutationCholeskyFactor' "converts" the sp_mat_t R already into a den_mat_t
		CHECK(L.cols() == R.rows());
		int ncols_R = (int)R.cols();
		int nrows_R = (int)R.rows();
		const T_mat_R* R_ptr = &R;//can be both sp_mat_t and sp_mat_rm_t
		const double* val = L.valuePtr();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		X = den_mat_t(R.rows(), R.cols());
		if (transpose) {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < ncols_R; ++j) {
				X.col(j) = R_ptr->col(j);
				sp_L_t_solve(val, row_idx, col_ptr, nrows_R, X.data() + j * nrows_R);
			}
		}
		else {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < ncols_R; ++j) {
				X.col(j) = R_ptr->col(j);
				sp_L_solve(val, row_idx, col_ptr, nrows_R, X.data() + j * nrows_R);
			}
		}
	}//end TriangularSolve (L = sp_mat_t && (R != vec_t && R != den_mat_t) && X == den_mat_t). 
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
				TriangularSolve<T_chol_mat, T_mat_X, T_mat_X>(chol.CholFactMatrix(), X, X, false);
			}
			else {
				TriangularSolve<T_chol_mat, T_mat_R, T_mat_X>(chol.CholFactMatrix(), R, X, false);
			}
		}
	}//end TriangularSolveGivenCholesky (chol != chol_den_mat_t || R ! sp_mat_t)
	template <class T_chol, class T_chol_mat, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value && std::is_same<sp_mat_t, T_mat_R>::value>::type* = nullptr >
	void TriangularSolveGivenCholesky(const T_chol& chol, const T_mat_R& R, T_mat_X& X, bool transpose) {
		// If chol is chol_den_mat_t and R is sp_mat_t, then R is to converted to den_mat_t
		den_mat_t R_den = (den_mat_t)(R);
		TriangularSolveGivenCholesky<chol_den_mat_t, T_chol_mat, den_mat_t, den_mat_t>(chol, R_den, X, transpose);
	}//end TriangularSolveGivenCholesky (chol = chol_den_mat_t && R = sp_mat_t)

	/*!
	* \brief Calculate P^T * L * R for a Cholesky factorization P * A * P^T = L * L^T. If R has independent standard normal
	*		entries, the columns of the result have covariance matrix A
	* \param chol Cholesky factorization
	* \param R Right-hand side
	* \param[out] X P^T * L * R
	*/
	template <class T_chol, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value>::type* = nullptr >
	void MultiplyCholeskyFactor(const T_chol& chol, const den_mat_t& R, den_mat_t& X) {
		X = chol.matrixL() * R;
	}
	template <class T_chol, typename std::enable_if <is_sparse_chol<T_chol>::value>::type* = nullptr >
	void MultiplyCholeskyFactor(const T_chol& chol, const den_mat_t& R, den_mat_t& X) {
		const den_mat_t L_R = chol.CholFactMatrix() * R;
		ApplyPermutationCholeskyFactor<den_mat_t, T_chol>(chol, L_R, X, true);
	}//end MultiplyCholeskyFactor

	/*!
	* \brief Calculate the squared norms of the columns of L^(-1) * P * R for a Cholesky factorization P * A * P^T = L * L^T
	*		without storing L^(-1) * P * R for a sparse Cholesky factorization, see 'CalcColSquaredNormsCholFactorInvR'
	* \param chol Cholesky factorization
	* \param R Right-hand side
	* \param[out] norms Squared norms of the columns of L^(-1) * P * R
	*/
	template <class T_chol, class T_mat_R, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value>::type* = nullptr >
	void CalcColSquaredNormsCholFactorInvPR(const T_chol& chol, const T_mat_R& R, vec_t& norms) {
		// A dense L^-1 has no fill-in, and a matrix product is faster than triangular solves for many dense right-hand sides
		den_mat_t L_inv;
		TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol, den_mat_t::Identity(chol.rows(), chol.cols()), L_inv, false);
		den_mat_t L_inv_R;
		L_inv_R.noalias() = L_inv * R;
		norms = L_inv_R.colwise().squaredNorm().transpose();
	}
	template <class T_chol, class T_mat_R, typename std::enable_if <is_sparse_chol<T_chol>::value>::type* = nullptr >
	void CalcColSquaredNormsCholFactorInvPR(const T_chol& chol, const T_mat_R& R, vec_t& norms) {
		const sp_mat_t L = chol.CholFactMatrix();
		sp_mat_t P_R = R;
		if (CholeskyHasPermutation<T_chol>(chol)) {
			const sp_mat_t R_cm = P_R;
			P_R = chol.permutationP() * R_cm;
		}
		CalcColSquaredNormsCholFactorInvR(L, P_R, norms);
	}//end CalcColSquaredNormsCholFactorInvPR

	/*!
	* \brief Calculate the entries of A^(-1) on the non-zeros of a matrix M given a Cholesky factorization of A, see
	*		'SparseInverseEntries'. For a dense matrix, all entries are calculated
	* \param chol Cholesky factorization of A
	* \param[out] M Matrix whose non-zero entries are overwritten
	*/
	template <class T_chol, class T_mat, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value>::type* = nullptr >
	void CalcInverseOnSparsityPattern(const T_chol& chol, T_mat& M) {
		// the triangular solves are parallelized over the columns and the product is a parallel matrix multiplication
		den_mat_t L_inv;
		TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol, den_mat_t::Identity(chol.rows(), chol.cols()), L_inv, false);
		M.noalias() = L_inv.transpose() * L_inv;
	}
	template <class T_chol, class T_mat, typename std::enable_if <is_sparse_chol<T_chol>::value>::type* = nullptr >
	void CalcInverseOnSparsityPattern(const T_chol& chol, T_mat& M) {
		const sp_mat_t L = chol.CholFactMatrix();
		sp_mat_t M_cm = M;
		SparseInverseEntries(L, chol.permutationP().indices()).OnSparsityPattern(M_cm);
		M = M_cm;
	}//end CalcInverseOnSparsityPattern

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
		TriangularSolveGivenCholesky<T_chol, T_chol_mat, T_mat_X, T_mat_X>(chol, X, X, true);
	}//end SolveGivenCholesky (chol != chol_den_mat_t || R ! sp_mat_t)
	template <class T_chol, class T_chol_mat, class T_mat_R, class T_mat_X, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value && std::is_same<sp_mat_t, T_mat_R>::value>::type* = nullptr >
	void SolveGivenCholesky(const T_chol& chol, const T_mat_R& R, T_mat_X& X) {
		// If chol is chol_den_mat_t and the R is sp_mat_t, then the RHS needs to converted to den_mat_t
		den_mat_t R_den = (den_mat_t)(R);
		SolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol, R_den, X);
	}//end SolveGivenCholesky (chol = chol_den_mat_t && R = sp_mat_t)
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
	* \param LtL[out] Matrix which contains a sparsity pattern and on which L.transpose() * L is calculated at the non-zero entries
	* \param symmetric If true, it is assumed that the sparsity pattern given in LtL is symmetric
	*/
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void CalcLtLGivenSparsityPattern(const T_mat& L, T_mat& LtL, bool symmetric) {
		if (symmetric) {
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
		}
		else {// non-symmetric case
#pragma omp parallel for schedule(static)
			for (int k = 0; k < LtL.outerSize(); ++k) {
				for (typename T_mat::InnerIterator it(LtL, k); it; ++it) {
					int i = (int)it.row();
					int j = (int)it.col();
					it.valueRef() = (L.col(i)).dot(L.col(j));
				}
			}
		}
	}//end CalcLtLGivenSparsityPattern (sparse)
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void CalcLtLGivenSparsityPattern(const den_mat_t& L, den_mat_t& LtL, bool) {
		LtL = L.transpose() * L;
	}//end CalcLtLGivenSparsityPattern (dense)

	/*!
	* \brief Calculate A * B only at non-zero entries for a given sparsiy pattern of AB
	* \param A Matrix A
	* \param B Square matrix B
	* \param AB[out] Matrix which contains a sparsity pattern and on which A * B is calculated at the non-zero entries of A
	*/
	//T_mat needs to be a sp_mat_t or sp_mat_rm_t
	template <class T_mat>
	void CalcAtimesBGivenSparsityPattern(const T_mat& A, const T_mat& B, T_mat& AB) {
		CHECK(B.rows() == B.cols());
		CHECK(A.cols() == B.rows());
		CHECK(AB.rows() == A.rows());
		CHECK(AB.cols() == A.cols());
#pragma omp parallel for schedule(static)
			for (int k = 0; k < AB.outerSize(); ++k) {
				for (typename T_mat::InnerIterator it(AB, k); it; ++it) {
					int i = (int)it.row();
					int j = (int)it.col();
					it.valueRef() = (A.row(i)).dot(B.col(j));
				}
			}
	}//end CalcAtimesBGivenSparsityPattern (sparse)

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
		const double* const v,
		double* ZtV,
		bool initialize_zero);

	/*!
	* \brief Create a sparse block diagonal matrix given two block matrices
	* \param A Square matrix A
	* \param B Square matrix B
	* \param BD[out] Sparse block diagonal matrix
	*/
	//T_mat needs to be a sp_mat_t or sp_mat_rm_t
	template <class T_mat>
	void CreatSparseBlockDiagonalMartix(const T_mat& A, const T_mat& B, T_mat& BD) {
		//NOTE: THIS CAN BE MADE FASTER, see the MakeBlockDiag_B_I function below
		data_size_t nrow_A = (data_size_t)A.rows();
		data_size_t ncol_A = (data_size_t)A.cols();
		data_size_t nrow_BD = nrow_A + (data_size_t)B.rows();
		data_size_t ncol_BD = ncol_A + (data_size_t)B.cols();
		int num_threads;
#ifdef _OPENMP
		num_threads = omp_get_max_threads();
#else
		num_threads = 1;
#endif
		std::vector<std::vector<Triplet_t>> threadTriplets(num_threads);
#pragma omp parallel
		{
			int tid;
#ifdef _OPENMP
			tid = omp_get_thread_num();
#else
			tid = 0;
#endif
			auto& localTriplets = threadTriplets[tid];
			const auto approx = static_cast<size_t>(A.nonZeros() + B.nonZeros()) / static_cast<size_t>(num_threads) + 64;
			localTriplets.reserve(approx);
#pragma omp for schedule(static)
			for (int k = 0; k < A.outerSize(); ++k) {
				for (typename T_mat::InnerIterator it(A, k); it; ++it) {
					localTriplets.emplace_back(it.row(), it.col(), it.value());
				}
			}
#pragma omp for schedule(static)
			for (int k = 0; k < B.outerSize(); ++k) {
				for (typename T_mat::InnerIterator it(B, k); it; ++it) {
					localTriplets.emplace_back(it.row() + nrow_A, it.col() + ncol_A, it.value());
				}
			}
		}
		std::vector<Triplet_t> triplets;
		triplets.reserve(A.nonZeros() + B.nonZeros());
		for (auto& local : threadTriplets) {
			triplets.insert(triplets.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
		}
		BD = T_mat(nrow_BD, ncol_BD);
		BD.setFromTriplets(triplets.begin(), triplets.end());
	}//end CreatSparseBlockDiagonalMartix (sparse)

	/*!
	* \brief Create a sparse block diagonal matrix with the identy or 0 matrix of dimension m x m as upper diaognal block and B as lower diagonal block
	* \param B Square matrix B
	* \param m Size of upper block diagonal identity or 0 matrix
	* \param[out] I_B Matrix ((I,0),(0,B)) or ((0,0),(0,B)) 
	* \param zero_instead_of_one If true, the upper part is 0 and not the identity
	*/
	//T_mat needs to be a sp_mat_t or sp_mat_rm_t
	template <class T_mat>
	void MakeBlockDiag_I_B(const T_mat& B, 
		const data_size_t m,
		T_mat& I_B,
		bool zero_instead_of_one) {
		using SIndex = typename T_mat::StorageIndex;
		CHECK(B.rows() == B.cols());
		CHECK(m > 0);
		const data_size_t n = (data_size_t)B.rows();
		const data_size_t outer = (data_size_t)B.outerSize();
		// 1) Count nnz per outer index
		std::vector<data_size_t> nnz_per_outer(outer, 0);
#pragma omp parallel for schedule(static)
		for (data_size_t k = 0; k < outer; ++k) {
			data_size_t c = 0;
			for (typename T_mat::InnerIterator it(B, k); it; ++it) ++c;
			nnz_per_outer[k] = c;
		}
		// 2) Prefix sum offsets
		std::vector<data_size_t> offsets(outer + 1, 0);
		for (data_size_t k = 0; k < outer; ++k) {
			offsets[k + 1] = offsets[k] + nnz_per_outer[k];
		}
		const data_size_t nnzB = offsets[outer];
		const data_size_t triplet_B_offset = zero_instead_of_one ? 0 : m;
		// 3) Allocate triplets: first m for I, then nnzB for shifted B
		std::vector<Triplet_t> triplets(triplet_B_offset + nnzB);
		if (!zero_instead_of_one) {
			// 4) Fill identity in top-left: indices [0, m)
#pragma omp parallel for schedule(static)
			for (data_size_t i = 0; i < m; ++i) {
				triplets[i] = Triplet_t(i, i, 1.);
			}
		}
		// 5) Fill B in bottom-right (shift by m): indices [m, m+nnzB)
#pragma omp parallel for schedule(static)
		for (data_size_t k = 0; k < outer; ++k) {
			data_size_t pos = offsets[k];
			for (typename T_mat::InnerIterator it(B, k); it; ++it) {
				triplets[triplet_B_offset + pos++] = Triplet_t(static_cast<SIndex>(it.row()) + m, static_cast<SIndex>(it.col()) + m, it.value());
			}
		}
		I_B.resize(m + n, m + n);
		I_B.setFromTriplets(triplets.begin(), triplets.end());
	}//end MakeBlockDiag_I_B

	/*!
	* \brief Create a sparse block diagonal matrix with a diagonal matrix D as upper diagonal block and B as lower diagonal block
	* \param D diagonal matrix
	* \param B Square matrix B
	* \param[out] D_B Block diagonal matrix ((D,0),(0,B))
	*/
	//T_mat needs to be a sp_mat_t or sp_mat_rm_t
	template <class T_mat>
	void MakeBlockDiag_D_B(const T_mat& D,
		const T_mat& B, 
		T_mat& D_B) {
		using SIndex = typename T_mat::StorageIndex;
		CHECK(B.rows() == B.cols());
		CHECK(D.rows() == D.cols());
		const data_size_t m = (data_size_t)D.rows();
		const data_size_t n = (data_size_t)B.rows();
		const data_size_t outer = (data_size_t)B.outerSize();
		// 1) Count nnz per outer index
		std::vector<data_size_t> nnz_per_outer(outer, 0);
#pragma omp parallel for schedule(static)
		for (data_size_t k = 0; k < outer; ++k) {
			data_size_t c = 0;
			for (typename T_mat::InnerIterator it(B, k); it; ++it) ++c;
			nnz_per_outer[k] = c;
		}
		// 2) Prefix sum offsets
		std::vector<data_size_t> offsets(outer + 1, 0);
		for (data_size_t k = 0; k < outer; ++k) {
			offsets[k + 1] = offsets[k] + nnz_per_outer[k];
		}
		const data_size_t nnzB = offsets[outer];
		// 3) Allocate triplets: first m for D, then nnzB for shifted B
		std::vector<Triplet_t> triplets(m + nnzB);
		// 4) Fill identity in top-left: indices [0, m)
#pragma omp parallel for schedule(static)
		for (data_size_t i = 0; i < m; ++i) {
			triplets[i] = Triplet_t(i, i, D.coeff(i, i));
		}
		// 5) Fill B in bottom-right (shift by m): indices [m, m+nnzB)
#pragma omp parallel for schedule(static)
		for (data_size_t k = 0; k < outer; ++k) {
			data_size_t pos = offsets[k];
			for (typename T_mat::InnerIterator it(B, k); it; ++it) {
				triplets[m + pos++] = Triplet_t(static_cast<SIndex>(it.row()) + m, static_cast<SIndex>(it.col()) + m, it.value());
			}
		}
		D_B.resize(m + n, m + n);
		D_B.setFromTriplets(triplets.begin(), triplets.end());
	}//end MakeBlockDiag_D_B

	/*!
	* \brief Scalar product of columns i and j of the matrix M
	*/
	double InnerProductTwoColumns(const sp_mat_t& M, int i, int j);

}  // namespace GPBoost

#endif   // GPB_SPARSE_MAT_H_
