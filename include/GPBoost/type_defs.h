/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_TYPE_DEFS_H_
#define GPB_TYPE_DEFS_H_

//#define EIGEN_SUPERLU_SUPPORT
//#define EIGEN_USE_BLAS
//#define EIGEN_USE_LAPACKE
//
//#define LAPACK_COMPLEX_CUSTOM
//#define lapack_complex_float std::complex<float>
//#define lapack_complex_double std::complex<double>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/CholmodSupport> // CHOLMOD is bundled in external_libs/SuiteSparse
#include <random> //for RNG_t
#include <GPBoost/blas_backend.h>
//#ifdef _MSC_VER
#pragma warning( disable : 4127) // Suppress unnecessary warning (conditional expression is constant)
//#endif

namespace GPBoost {

	/*! \brief Type of Eigen matrices */
	typedef Eigen::MatrixXd den_mat_t;
	typedef Eigen::VectorXd vec_t;
	typedef Eigen::VectorXi vec_int_t;
	typedef Eigen::SparseVector<double> sp_vec_t;
	typedef Eigen::SparseMatrix<double> sp_mat_t; // column-major sparse matrix
	typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat_rm_t; // row-major sparse matrix
	typedef Eigen::Triplet<double> Triplet_t;
	class DenseLLT;
	typedef DenseLLT chol_den_mat_t; // see below
	typedef Eigen::SimplicialLLT<sp_mat_t, Eigen::Lower, Eigen::AMDOrdering<int>> chol_sp_mat_t;
	typedef Eigen::SimplicialLLT<sp_mat_rm_t, Eigen::Lower, Eigen::AMDOrdering<int>> chol_sp_mat_rm_t;
	// AMDOrdering is faster than NaturalOrdering for sparse matrices for GPs (e.g. tapering) but slightly slower than no ordering for grouped random effects for Gaussian data
	// COLAMDOrdering is slower than NaturalOrdering or AMDOrdering for both grouped random effects and the Vecchia approximation for non-Gaussian data

	// CHOLMOD counterparts of the two Cholesky factorizations above. CHOLMOD is bundled in
	// external_libs/SuiteSparse together with the AMD and COLAMD orderings it uses and with an
	// Eigen backed BLAS, so neither an external CHOLMOD nor an external BLAS has to be linked.
	// CholmodAutoLLT lets CHOLMOD choose between a supernodal and a simplicial factorization
	// depending on how dense the factor is. It is a drop-in replacement for chol_sp_mat_t: it
	// offers the same compute() / analyzePattern() / factorize() / solve() / info() and, through
	// the GPBoost additions in Eigen/src/CholmodSupport, also CholFactMatrix() and permutationP().
	// Note that unlike Eigen's SimplicialLLT, CHOLMOD keeps the factor in its own format, so
	// CholFactMatrix() has to convert it once per factorization. Where only solve() and
	// logDeterminant() are needed, that conversion never happens and CHOLMOD is fastest.
	typedef Eigen::CholmodAutoLLT<sp_mat_t, Eigen::Lower> chol_cholmod_sp_mat_t;
	typedef Eigen::CholmodAutoLLT<sp_mat_rm_t, Eigen::Lower> chol_cholmod_sp_mat_rm_t;

	// Defined in sparse_matrix_utils.cpp, see the declaration in sparse_matrix_utils.h
	void TriangularSolveInPlace(const den_mat_t& L, den_mat_t& B, bool transpose);

	/*!
	* \brief Cholesky factorization of a dense matrix that is used like Eigen::LLT<den_mat_t, Eigen::Lower>. It
	*		uses OpenBLAS if one is registered, see blas_backend.h, and Eigen otherwise
	*/
	class DenseLLT : public Eigen::LLT<den_mat_t, Eigen::Lower> {
		typedef Eigen::LLT<den_mat_t, Eigen::Lower> Base;
	public:
		DenseLLT() : Base() {}

		template <typename InputType>
		explicit DenseLLT(const Eigen::EigenBase<InputType>& matrix) : Base(matrix.rows()) {
			compute(matrix.derived());
		}

		template <typename InputType>
		DenseLLT& compute(const Eigen::EigenBase<InputType>& matrix) {
			if (!UseBLAS()) {
				Base::compute(matrix);
				return *this;
			}
			const Eigen::Index n = matrix.rows();
			m_matrix = matrix.derived();
			// L1 norm of the symmetric matrix given by its lower triangle, for rcond(), as in Eigen::LLT::compute()
			m_l1_norm = 0.;
			for (Eigen::Index col = 0; col < n; ++col) {
				const double abs_col_sum = m_matrix.col(col).tail(n - col).lpNorm<1>() + m_matrix.row(col).head(col).lpNorm<1>();
				if (abs_col_sum > m_l1_norm) {
					m_l1_norm = abs_col_sum;
				}
			}
			m_isInitialized = true;
			m_info = CholeskyBLAS(m_matrix.data(), (int)n) ? Eigen::Success : Eigen::NumericalIssue;
			return *this;
		}

		/*!
		* \brief Solve A X = B. Contrary to Eigen::LLT::solve(), this returns the solution and not an expression. A
		*		right-hand side with several columns is solved with 'TriangularSolveInPlace', i.e., in parallel or with
		*		OpenBLAS, whereas Eigen solves it in a single thread
		*/
		template <typename Rhs>
		typename Rhs::PlainObject solve(const Eigen::MatrixBase<Rhs>& B) const {
			typename Rhs::PlainObject X = B;
			SolveInPlace(X);
			return X;
		}

	private:
		void SolveInPlace(den_mat_t& X) const {
			if (X.cols() > 1) {
				TriangularSolveInPlace(m_matrix, X, false);
				TriangularSolveInPlace(m_matrix, X, true);
			}
			else {
				Base::solveInPlace(X);
			}
		}

		template <typename T>
		void SolveInPlace(T& X) const {
			Base::solveInPlace(X);
		}
	};

	typedef std::string string_t;

	/*! \brief Type of labels for group levels for grouped random effects */
	typedef string_t re_group_t;

	/*! \brief Type of data size */
	typedef int32_t data_size_t;

	/*! \brief Type of random number generator */
	typedef std::mt19937 RNG_t;

	// For LightGBM Enable following marco to use double for score_t
	// THIS NEEDS TO BE ACTIVATED. Need to use double since the Eigen objects also use double. If float should be used, the Eigen objects below need to be changed.
	 #define SCORE_T_USE_DOUBLE
	/*! \brief Type of score, gradients */
	#ifdef SCORE_T_USE_DOUBLE
	typedef double score_t;
	#else
	typedef float score_t;
	#endif

}  // namespace GPBoost

#endif   // GPB_TYPE_DEFS_H_
