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

extern "C" {
#include <cs.h>
}

namespace GPBoost {

	/*!
	* \brief Solve equation system with a dense lower triangular matrix as left-hand side (Lx=b)
	* \param val Values of matrix in column-major format
	* \param ncol Number of columns
	* \param[out] x Right-hand side vector (solution written on input)
	*/
  void L_solve(const double* val, const int ncol, double* x);

	/*!
	* \brief Solve equation system with a sparse lower triangular matrix as left-hand side (Lx=b)
	* \param val Values of sparse matrix
	* \param row_idx Row indices corresponding to the values ('InnerIndices' in Eigen)
	* \param col_ptr val indexes where each column starts ('OuterStarts' in Eigen)
	* \param ncol Number of columns
	* \param[out] x Right-hand side vector (solution written on input)
	*/
  void sp_L_solve(const double* val, const int* row_idx, const int* col_ptr, const int ncol, double* x);

	/*!
	* \brief Solve equation system with the transpose of a sparse lower triangular matrix as left-hand side: (L'x=b)
	* \param val Values of sparse matrix
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
  void sp_Lower_sp_RHS_cs_solve(cs* A, cs* B, sp_mat_t& A_inv_B, bool lower = true);

  /*!
* \brief Solve equation system with a sparse left-hand side and a sparse right-hand side (Ax=B) using CSparse function cs_spsolve
* \param A left-hand side. Sparse Eigen matrix is column major format (=default)
* \param B right-hand side. Sparse Eigen matrix is column major format (=default)
* \param[out] Solution A^(-1)B
* \param lower true if A is a lower triangular matrix
*/
  void eigen_sp_Lower_sp_RHS_cs_solve(sp_mat_t& A, sp_mat_t& B, sp_mat_t& A_inv_B, bool lower = true);

}  // namespace GPBoost

#endif   // GPB_SPARSE_MAT_H_
