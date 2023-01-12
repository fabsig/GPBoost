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
#include <random> //for RNG_t
//#ifdef _MSC_VER
#pragma warning( disable : 4127) // Suppress unnecessary warning (conditional expression is constant)
//#endif

namespace GPBoost {

	/*! \brief Type of Eigen matrices */
	typedef Eigen::MatrixXd den_mat_t;
	typedef Eigen::VectorXd vec_t;
	typedef Eigen::VectorXi vec_int_t;
	typedef Eigen::SparseMatrix<double> sp_mat_t; // column-major sparse matrix
	typedef Eigen::SparseMatrix<double, Eigen::RowMajor> sp_mat_rm_t; // row-major sparse matrix
	typedef Eigen::Triplet<double> Triplet_t;
	typedef Eigen::LLT<Eigen::MatrixXd, Eigen::Lower> chol_den_mat_t;
	typedef Eigen::SimplicialLLT<sp_mat_t, Eigen::Lower, Eigen::AMDOrdering<int>> chol_sp_mat_t;
	typedef Eigen::SimplicialLLT<sp_mat_rm_t, Eigen::Lower, Eigen::AMDOrdering<int>> chol_sp_mat_rm_t;
	// AMDOrdering is faster than NaturalOrdering for sparse matrices for GPs (e.g. tapering) but slightly slower than no ordering for grouped random effects for Gaussian data
	// COLAMDOrdering is slower than NaturalOrdering or AMDOrdering for both grouped random effects and the Vecchia approximation for non-Gaussian data

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
