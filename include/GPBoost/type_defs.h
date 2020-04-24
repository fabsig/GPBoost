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

#include <eigen/Sparse>
#include <eigen/Dense>
//#ifdef _MSC_VER
#pragma warning( disable : 4127) // Suppress unnecessary warning (conditional expression is constant)
//#endif

namespace GPBoost {

	/*! \brief Type of Eigen matrices *///TODO: (i) move this inside #ifdef SCORE_T_USE_DOUBLE and also define float types. This requires a modification of the C and R api (change types to float there as for score initilization, see setfield...)
	typedef Eigen::SparseMatrix<double> sp_mat_t; // column-major sparse matrix type of double
  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat_rm_t; // row-major sparse matrix type of double
	typedef Eigen::Triplet<double> Triplet_t;
	typedef Eigen::MatrixXd den_mat_t;
	typedef Eigen::VectorXd vec_t;//typedef Eigen::VectorXf vec_t;
	typedef Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> chol_sp_mat_t; // sparse Cholesky factor. TODO Maybe use other ordering?
	typedef Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> chol_den_mat_t;

	typedef std::string string_t;

	/*! \brief Type of labels for group levels for grouped random effects */
	typedef string_t re_group_t;

	/*! \brief Type of labels indicating independent realizations of the same gaussian process in 'gp_id' */
	typedef int gp_id_t;

	/*! \brief Type of data size */
	typedef int32_t data_size_t;

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
