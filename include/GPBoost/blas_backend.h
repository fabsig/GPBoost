/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2026 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_BLAS_BACKEND_H_
#define GPB_BLAS_BACKEND_H_

// Optional use of an OpenBLAS library for dense Cholesky factorizations, inverses, and triangular solves, and for
// the dense kernels of the supernodal factorization of the bundled CHOLMOD. GPBoost does not link a BLAS library:
// the R and Python packages register an OpenBLAS that is already loaded in their process or that the package
// links statically, and GPBoost's own code (Eigen) is used if there is none.

#include <cstddef>
#include <string>

namespace GPBoost {

	/*!
	* \brief Routines of an OpenBLAS library with 32-bit integers. The trailing size_t arguments are the hidden
	*		lengths of the character arguments of the Fortran calling convention: routines that OpenBLAS implements
	*		in C ignore them, and routines that it takes from the Fortran reference LAPACK may need them
	*/
	struct OpenBLASFunctions {
		void (*dgemm)(const char* transa, const char* transb, const int* m, const int* n, const int* k,
			const double* alpha, const double* A, const int* lda, const double* B, const int* ldb,
			const double* beta, double* C, const int* ldc, size_t, size_t) = nullptr;
		void (*dgemv)(const char* trans, const int* m, const int* n, const double* alpha, const double* A,
			const int* lda, const double* x, const int* incx, const double* beta, double* y, const int* incy,
			size_t) = nullptr;
		void (*dsyrk)(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha,
			const double* A, const int* lda, const double* beta, double* C, const int* ldc, size_t, size_t) = nullptr;
		void (*dtrsm)(const char* side, const char* uplo, const char* transa, const char* diag, const int* m,
			const int* n, const double* alpha, const double* A, const int* lda, double* B, const int* ldb,
			size_t, size_t, size_t, size_t) = nullptr;
		void (*dtrsv)(const char* uplo, const char* trans, const char* diag, const int* n, const double* A,
			const int* lda, double* x, const int* incx, size_t, size_t, size_t) = nullptr;
		void (*dpotrf)(const char* uplo, const int* n, double* A, const int* lda, int* info, size_t) = nullptr;
		void (*dpotri)(const char* uplo, const int* n, double* A, const int* lda, int* info, size_t) = nullptr;
		/*! \brief Threading of the library: 0 = sequential, 1 = own threads (pthreads), 2 = OpenMP */
		int (*get_parallel)() = nullptr;
		int (*get_num_threads)() = nullptr;
		void (*set_num_threads)(int num_threads) = nullptr;
		char* (*get_config)() = nullptr;
	};

	/*!
	* \brief Use the OpenBLAS library in a shared library file. Nothing changes if the file cannot be loaded, is not
	*		OpenBLAS, or uses 64-bit integers. The symbols of OpenBLAS may have the prefix 'scipy_', as in the
	*		OpenBLAS of the Python package scipy
	* \param library_path Path of the shared library (UTF-8)
	* \return True if the library is used from now on
	*/
	bool LoadOpenBLAS(const std::string& library_path);

	/*!
	* \brief Use an OpenBLAS library that is linked into GPBoost
	* \param functions Routines of the library, all of which have to be given
	*/
	void SetOpenBLAS(const OpenBLASFunctions& functions);

	/*!
	* \brief Set whether a registered OpenBLAS library is used ("auto") or GPBoost's own code ("builtin")
	*/
	void SetBLAS(const std::string& blas);

	/*!
	* \brief Description of the BLAS library that is used, i.e., the configuration of OpenBLAS, or "builtin" if
	*		GPBoost uses its own code
	*/
	std::string GetBLAS();

	/*!
	* \brief True if the calling thread uses OpenBLAS: one has been registered, it has not been switched off, and
	*		the thread is not in a parallel region of OpenMP. In parallel regions, GPBoost's own code is used, since
	*		the threads of OpenBLAS would compete with the ones of the region
	*/
	bool UseBLAS();

	/*!
	* \brief Cholesky factorization A = L L^T with OpenBLAS (dpotrf). Use only if 'UseBLAS()' is true
	* \param[in,out] A Column-major n x n matrix whose lower triangle is overwritten by L. Its strictly upper
	*		triangle is neither read nor changed
	* \param n Dimension
	* \return True if A is positive definite
	*/
	bool CholeskyBLAS(double* A, int n);

	/*!
	* \brief Inverse (L L^T)^-1 given the lower triangular Cholesky factor L with OpenBLAS (dpotri). Use only if
	*		'UseBLAS()' is true
	* \param[in,out] A Column-major n x n matrix with L in its lower triangle, which is overwritten by the lower
	*		triangle of the inverse. Its strictly upper triangle is neither read nor changed
	* \param n Dimension
	* \return True if L is not singular
	*/
	bool CholeskyInverseBLAS(double* A, int n);

	/*!
	* \brief Solve L X = B or L^T X = B with OpenBLAS (dtrsm) for a lower triangular L. Use only if 'UseBLAS()' is
	*		true
	* \param L Column-major n x n matrix with L in its lower triangle
	* \param n Dimension
	* \param[in,out] B Column-major n x num_cols right-hand side, which is overwritten by the solution X
	* \param num_cols Number of columns of B
	* \param transpose If true, L^T X = B is solved
	*/
	void TriangularSolveBLAS(const double* L, int n, double* B, int num_cols, bool transpose);

	// The BLAS and LAPACK routines that CHOLMOD calls, with its arguments, for the shim in
	// external_libs/SuiteSparse/EigenBLAS. Use only if 'UseBLAS()' is true
	void DgemmBLAS(const char* transa, const char* transb, const int* m, const int* n, const int* k,
		const double* alpha, const double* A, const int* lda, const double* B, const int* ldb,
		const double* beta, double* C, const int* ldc);
	void DgemvBLAS(const char* trans, const int* m, const int* n, const double* alpha, const double* A,
		const int* lda, const double* x, const int* incx, const double* beta, double* y, const int* incy);
	void DsyrkBLAS(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha,
		const double* A, const int* lda, const double* beta, double* C, const int* ldc);
	void DtrsmBLAS(const char* side, const char* uplo, const char* transa, const char* diag, const int* m,
		const int* n, const double* alpha, const double* A, const int* lda, double* B, const int* ldb);
	void DtrsvBLAS(const char* uplo, const char* trans, const char* diag, const int* n, const double* A,
		const int* lda, double* x, const int* incx);
	void DpotrfBLAS(const char* uplo, const int* n, double* A, const int* lda, int* info);

}  // namespace GPBoost

#endif  // GPB_BLAS_BACKEND_H_
