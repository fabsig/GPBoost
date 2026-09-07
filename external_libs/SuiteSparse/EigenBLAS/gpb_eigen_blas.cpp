/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2026 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/

// The BLAS and LAPACK kernels that the bundled CHOLMOD needs, implemented on
// top of Eigen. This lets GPBoost ship a self-contained CHOLMOD, including its
// supernodal factorization, without asking the user to link an external
// BLAS/LAPACK.
//
// CHOLMOD reaches BLAS/LAPACK exclusively through the SUITESPARSE_BLAS_* and
// SUITESPARSE_LAPACK_* macros of SuiteSparse_config.h. When
// GPB_SUITESPARSE_INTERNAL_BLAS is defined, those macros expand to the
// 'gpb_blas_'-prefixed names defined below instead of to the Fortran ones, so
// nothing here can collide with a real BLAS that the host program links
// (R's libRblas, OpenBLAS, MKL, ...).
//
// Only the six kernels CHOLMOD calls are provided - gemm, gemv, syrk/herk,
// trsm, trsv and potrf - each for the four SuiteSparse scalar types. The
// remaining BLAS/LAPACK routines declared by SuiteSparse_config.h belong to
// other SuiteSparse packages that GPBoost does not bundle.

#ifdef GPB_SUITESPARSE_INTERNAL_BLAS

#include <complex>
#include <vector>

#include <Eigen/Dense>

#define SUITESPARSE_BLAS_PROTOTYPES
#include "SuiteSparse_config.h"

namespace {

	typedef SUITESPARSE_BLAS_INT blas_int;

	// BLAS matrices are column-major with a leading dimension
	template <typename T>
	using ConstMat = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Unaligned, Eigen::OuterStride<>>;
	template <typename T>
	using Mat = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Unaligned, Eigen::OuterStride<>>;
	template <typename T>
	using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

	inline char Upper_(const char* c) {
		const char u = *c;
		return (u >= 'a' && u <= 'z') ? (char)(u - 'a' + 'A') : u;
	}

	template <typename T>
	inline ConstMat<T> MapConst(const T* A, blas_int rows, blas_int cols, blas_int ld) {
		return ConstMat<T>(A, rows, cols, Eigen::OuterStride<>(ld));
	}
	template <typename T>
	inline Mat<T> MapMut(T* A, blas_int rows, blas_int cols, blas_int ld) {
		return Mat<T>(A, rows, cols, Eigen::OuterStride<>(ld));
	}

	// BLAS strided vectors: element i sits at X[i*inc] for inc > 0 and at
	// X[(n-1-i)*(-inc)] for inc < 0. Gathering into a contiguous temporary keeps
	// the kernels below free of stride handling; it is O(n) against their O(n^2).
	template <typename T>
	inline Vec<T> Gather(const T* X, blas_int n, blas_int inc) {
		Vec<T> x(n);
		if (inc == 1) {
			for (blas_int i = 0; i < n; ++i) x[i] = X[i];
		}
		else if (inc > 0) {
			for (blas_int i = 0; i < n; ++i) x[i] = X[i * inc];
		}
		else {
			for (blas_int i = 0; i < n; ++i) x[i] = X[(n - 1 - i) * (-inc)];
		}
		return x;
	}
	template <typename T>
	inline void Scatter(const Vec<T>& x, T* X, blas_int n, blas_int inc) {
		if (inc == 1) {
			for (blas_int i = 0; i < n; ++i) X[i] = x[i];
		}
		else if (inc > 0) {
			for (blas_int i = 0; i < n; ++i) X[i * inc] = x[i];
		}
		else {
			for (blas_int i = 0; i < n; ++i) X[(n - 1 - i) * (-inc)] = x[i];
		}
	}

	//--------------------------------------------------------------------------
	// gemm: C = alpha * op(A) * op(B) + beta * C
	//--------------------------------------------------------------------------

	// 'C' (conjugate transpose) and 'T' (transpose) coincide for real scalars,
	// where Eigen's adjoint() is transpose(), so the same code serves both.
	template <typename T, typename ExprA, typename ExprB>
	inline void GemmCore(const ExprA& A, const ExprB& B, T alpha, T beta, Mat<T>& C) {
		if (beta == T(0)) {
			C.setZero();
		}
		else if (beta != T(1)) {
			C *= beta;
		}
		if (alpha != T(0)) {
			C.noalias() += alpha * (A * B);
		}
	}

	template <typename T, typename ExprA>
	inline void GemmDispatchB(const ExprA& A, char transb, const T* B, blas_int ldb,
		blas_int k, blas_int n, T alpha, T beta, Mat<T>& C) {
		if (transb == 'N') {
			GemmCore<T>(A, MapConst(B, k, n, ldb), alpha, beta, C);
		}
		else if (transb == 'T') {
			GemmCore<T>(A, MapConst(B, n, k, ldb).transpose(), alpha, beta, C);
		}
		else {
			GemmCore<T>(A, MapConst(B, n, k, ldb).adjoint(), alpha, beta, C);
		}
	}

	template <typename T>
	void Gemm(const char* transa_, const char* transb_, const blas_int* m_, const blas_int* n_,
		const blas_int* k_, const T* alpha_, const T* A, const blas_int* lda_,
		const T* B, const blas_int* ldb_, const T* beta_, T* Cp, const blas_int* ldc_) {
		const blas_int m = *m_, n = *n_, k = *k_, lda = *lda_, ldb = *ldb_, ldc = *ldc_;
		if (m <= 0 || n <= 0) return;
		const char transa = Upper_(transa_), transb = Upper_(transb_);
		const T alpha = *alpha_, beta = *beta_;
		Mat<T> C = MapMut(Cp, m, n, ldc);
		if (k <= 0) {
			// only the beta scaling of C remains
			if (beta == T(0)) C.setZero();
			else if (beta != T(1)) C *= beta;
			return;
		}
		if (transa == 'N') {
			GemmDispatchB<T>(MapConst(A, m, k, lda), transb, B, ldb, k, n, alpha, beta, C);
		}
		else if (transa == 'T') {
			GemmDispatchB<T>(MapConst(A, k, m, lda).transpose(), transb, B, ldb, k, n, alpha, beta, C);
		}
		else {
			GemmDispatchB<T>(MapConst(A, k, m, lda).adjoint(), transb, B, ldb, k, n, alpha, beta, C);
		}
	}

	//--------------------------------------------------------------------------
	// gemv: y = alpha * op(A) * x + beta * y
	//--------------------------------------------------------------------------

	template <typename T>
	void Gemv(const char* trans_, const blas_int* m_, const blas_int* n_, const T* alpha_,
		const T* A, const blas_int* lda_, const T* X, const blas_int* incx_,
		const T* beta_, T* Y, const blas_int* incy_) {
		const blas_int m = *m_, n = *n_, lda = *lda_, incx = *incx_, incy = *incy_;
		if (m <= 0 || n <= 0) return;
		const char trans = Upper_(trans_);
		const T alpha = *alpha_, beta = *beta_;
		// op(A) is m x n for 'N' and n x m otherwise
		const blas_int lenx = (trans == 'N') ? n : m;
		const blas_int leny = (trans == 'N') ? m : n;
		Vec<T> x = Gather(X, lenx, incx);
		Vec<T> y = Gather(Y, leny, incy);
		if (beta == T(0)) y.setZero();
		else if (beta != T(1)) y *= beta;
		if (alpha != T(0)) {
			ConstMat<T> Am = MapConst(A, m, n, lda);
			if (trans == 'N') y.noalias() += alpha * (Am * x);
			else if (trans == 'T') y.noalias() += alpha * (Am.transpose() * x);
			else y.noalias() += alpha * (Am.adjoint() * x);
		}
		Scatter(y, Y, leny, incy);
	}

	//--------------------------------------------------------------------------
	// syrk / herk: C = alpha * op(A) * op(A)^H + beta * C, one triangle only
	//--------------------------------------------------------------------------

	// alpha and beta are real even in the Hermitian case; SuiteSparse declares
	// them as complex pointers, matching how Fortran herk reads the leading
	// real word of the array it is handed.
	template <typename T>
	inline typename Eigen::NumTraits<T>::Real RealOf(const T* p) {
		return Eigen::numext::real(*p);
	}

	template <typename T>
	void Syrk(const char* uplo_, const char* trans_, const blas_int* n_, const blas_int* k_,
		const T* alpha_, const T* A, const blas_int* lda_, const T* beta_,
		T* Cp, const blas_int* ldc_) {
		typedef typename Eigen::NumTraits<T>::Real Real;
		const blas_int n = *n_, k = *k_, lda = *lda_, ldc = *ldc_;
		if (n <= 0) return;
		const bool lower = (Upper_(uplo_) == 'L');
		const char trans = Upper_(trans_);
		const Real alpha = RealOf(alpha_);
		const Real beta = RealOf(beta_);
		Mat<T> C = MapMut(Cp, n, n, ldc);

		// scale the referenced triangle by beta
		if (beta != Real(1)) {
			for (blas_int j = 0; j < n; ++j) {
				const blas_int i0 = lower ? j : 0;
				const blas_int i1 = lower ? n : (j + 1);
				for (blas_int i = i0; i < i1; ++i) {
					C(i, j) = (beta == Real(0)) ? T(0) : (T)(beta * C(i, j));
				}
			}
		}
		if (Eigen::NumTraits<T>::IsComplex) {
			// a Hermitian matrix has a real diagonal; LAPACK herk enforces this
			for (blas_int j = 0; j < n; ++j) {
				C(j, j) = T(Eigen::numext::real(C(j, j)));
			}
		}
		if (k <= 0 || alpha == Real(0)) return;

		// Eigen's rankUpdate(U, alpha) computes C += alpha * U * U^H, so U is
		// A itself for 'N' and A^H otherwise (A^T for real scalars).
		if (trans == 'N') {
			ConstMat<T> Am = MapConst(A, n, k, lda);
			if (lower) C.template selfadjointView<Eigen::Lower>().rankUpdate(Am, alpha);
			else C.template selfadjointView<Eigen::Upper>().rankUpdate(Am, alpha);
		}
		else {
			ConstMat<T> Am = MapConst(A, k, n, lda);
			if (lower) C.template selfadjointView<Eigen::Lower>().rankUpdate(Am.adjoint(), alpha);
			else C.template selfadjointView<Eigen::Upper>().rankUpdate(Am.adjoint(), alpha);
		}
	}

	//--------------------------------------------------------------------------
	// trsm: solve op(A) * X = alpha * B or X * op(A) = alpha * B
	//--------------------------------------------------------------------------

	template <typename T, typename TriA>
	inline void TrsmSolve(const TriA& A, char transa, bool left, Mat<T>& B) {
		if (transa == 'N') {
			if (left) A.solveInPlace(B);
			else A.template solveInPlace<Eigen::OnTheRight>(B);
		}
		else if (transa == 'T') {
			if (left) A.transpose().solveInPlace(B);
			else A.transpose().template solveInPlace<Eigen::OnTheRight>(B);
		}
		else {
			if (left) A.adjoint().solveInPlace(B);
			else A.adjoint().template solveInPlace<Eigen::OnTheRight>(B);
		}
	}

	template <typename T>
	void Trsm(const char* side_, const char* uplo_, const char* transa_, const char* diag_,
		const blas_int* m_, const blas_int* n_, const T* alpha_, const T* A,
		const blas_int* lda_, T* Bp, const blas_int* ldb_) {
		const blas_int m = *m_, n = *n_, lda = *lda_, ldb = *ldb_;
		if (m <= 0 || n <= 0) return;
		const bool left = (Upper_(side_) == 'L');
		const bool lower = (Upper_(uplo_) == 'L');
		const bool unit = (Upper_(diag_) == 'U');
		const char transa = Upper_(transa_);
		const T alpha = *alpha_;
		Mat<T> B = MapMut(Bp, m, n, ldb);
		if (alpha == T(0)) {
			B.setZero();
			return;
		}
		if (alpha != T(1)) B *= alpha;

		const blas_int na = left ? m : n;
		ConstMat<T> Am = MapConst(A, na, na, lda);
		if (lower) {
			if (unit) TrsmSolve<T>(Am.template triangularView<Eigen::UnitLower>(), transa, left, B);
			else TrsmSolve<T>(Am.template triangularView<Eigen::Lower>(), transa, left, B);
		}
		else {
			if (unit) TrsmSolve<T>(Am.template triangularView<Eigen::UnitUpper>(), transa, left, B);
			else TrsmSolve<T>(Am.template triangularView<Eigen::Upper>(), transa, left, B);
		}
	}

	//--------------------------------------------------------------------------
	// trsv: solve op(A) * x = b
	//--------------------------------------------------------------------------

	template <typename T, typename TriA>
	inline void TrsvSolve(const TriA& A, char transa, Vec<T>& x) {
		if (transa == 'N') A.solveInPlace(x);
		else if (transa == 'T') A.transpose().solveInPlace(x);
		else A.adjoint().solveInPlace(x);
	}

	template <typename T>
	void Trsv(const char* uplo_, const char* trans_, const char* diag_, const blas_int* n_,
		const T* A, const blas_int* lda_, T* X, const blas_int* incx_) {
		const blas_int n = *n_, lda = *lda_, incx = *incx_;
		if (n <= 0) return;
		const bool lower = (Upper_(uplo_) == 'L');
		const bool unit = (Upper_(diag_) == 'U');
		const char trans = Upper_(trans_);
		Vec<T> x = Gather(X, n, incx);
		ConstMat<T> Am = MapConst(A, n, n, lda);
		if (lower) {
			if (unit) TrsvSolve<T>(Am.template triangularView<Eigen::UnitLower>(), trans, x);
			else TrsvSolve<T>(Am.template triangularView<Eigen::Lower>(), trans, x);
		}
		else {
			if (unit) TrsvSolve<T>(Am.template triangularView<Eigen::UnitUpper>(), trans, x);
			else TrsvSolve<T>(Am.template triangularView<Eigen::Upper>(), trans, x);
		}
		Scatter(x, X, n, incx);
	}

	//--------------------------------------------------------------------------
	// potrf: Cholesky factorization of a positive definite matrix
	//--------------------------------------------------------------------------

	// Eigen's public LLT reports only that a factorization failed, whereas
	// CHOLMOD needs the position of the first non-positive pivot to set
	// L->minor. Eigen::internal::llt_inplace returns exactly that index, which
	// is LAPACK's 'info' minus one.
	template <typename T>
	void Potrf(const char* uplo_, const blas_int* n_, T* Ap, const blas_int* lda_, blas_int* info) {
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> PlainMat;
		const blas_int n = *n_, lda = *lda_;
		*info = 0;
		if (n <= 0) return;
		const bool lower = (Upper_(uplo_) == 'L');
		Mat<T> A = MapMut(Ap, n, n, lda);
		// llt_inplace reads only the factorized triangle, so whatever the caller
		// left in the other one may be copied along unexamined.
		PlainMat work = A;
		Eigen::Index bad;
		if (lower) bad = Eigen::internal::llt_inplace<T, Eigen::Lower>::blocked(work);
		else bad = Eigen::internal::llt_inplace<T, Eigen::Upper>::blocked(work);
		if (bad >= 0) {
			*info = (blas_int)(bad + 1);
			return;
		}
		if (lower) A.template triangularView<Eigen::Lower>() = work.template triangularView<Eigen::Lower>();
		else A.template triangularView<Eigen::Upper>() = work.template triangularView<Eigen::Upper>();
	}

}  // namespace

typedef std::complex<float> cfloat_t;
typedef std::complex<double> cdouble_t;

extern "C" {

	//--------------------------------- gemm -----------------------------------

#define GPB_DEFINE_GEMM(NAME, T)                                                       \
	void NAME(const char* transa, const char* transb, const blas_int* m,               \
		const blas_int* n, const blas_int* k, const T* alpha, const T* A,              \
		const blas_int* lda, const T* B, const blas_int* ldb, const T* beta,           \
		T* C, const blas_int* ldc) NOTHROW {                                                   \
		Gemm<T>(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);         \
	}

	GPB_DEFINE_GEMM(SUITESPARSE_BLAS_DGEMM, double)
	GPB_DEFINE_GEMM(SUITESPARSE_BLAS_SGEMM, float)
	GPB_DEFINE_GEMM(SUITESPARSE_BLAS_ZGEMM, cdouble_t)
	GPB_DEFINE_GEMM(SUITESPARSE_BLAS_CGEMM, cfloat_t)

	//--------------------------------- gemv -----------------------------------

#define GPB_DEFINE_GEMV(NAME, T)                                                       \
	void NAME(const char* trans, const blas_int* m, const blas_int* n,                 \
		const T* alpha, const T* A, const blas_int* lda, const T* X,                   \
		const blas_int* incx, const T* beta, T* Y, const blas_int* incy) NOTHROW {             \
		Gemv<T>(trans, m, n, alpha, A, lda, X, incx, beta, Y, incy);                   \
	}

	GPB_DEFINE_GEMV(SUITESPARSE_BLAS_DGEMV, double)
	GPB_DEFINE_GEMV(SUITESPARSE_BLAS_SGEMV, float)
	GPB_DEFINE_GEMV(SUITESPARSE_BLAS_ZGEMV, cdouble_t)
	GPB_DEFINE_GEMV(SUITESPARSE_BLAS_CGEMV, cfloat_t)

	//------------------------------ syrk / herk -------------------------------

#define GPB_DEFINE_SYRK(NAME, T)                                                       \
	void NAME(const char* uplo, const char* trans, const blas_int* n,                  \
		const blas_int* k, const T* alpha, const T* A, const blas_int* lda,            \
		const T* beta, T* C, const blas_int* ldc) NOTHROW {                                    \
		Syrk<T>(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);                       \
	}

	GPB_DEFINE_SYRK(SUITESPARSE_BLAS_DSYRK, double)
	GPB_DEFINE_SYRK(SUITESPARSE_BLAS_SSYRK, float)
	GPB_DEFINE_SYRK(SUITESPARSE_BLAS_ZHERK, cdouble_t)
	GPB_DEFINE_SYRK(SUITESPARSE_BLAS_CHERK, cfloat_t)

	//--------------------------------- trsm -----------------------------------

#define GPB_DEFINE_TRSM(NAME, T)                                                       \
	void NAME(const char* side, const char* uplo, const char* transa,                  \
		const char* diag, const blas_int* m, const blas_int* n, const T* alpha,        \
		const T* A, const blas_int* lda, T* B, const blas_int* ldb) NOTHROW {                  \
		Trsm<T>(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);                \
	}

	GPB_DEFINE_TRSM(SUITESPARSE_BLAS_DTRSM, double)
	GPB_DEFINE_TRSM(SUITESPARSE_BLAS_STRSM, float)
	GPB_DEFINE_TRSM(SUITESPARSE_BLAS_ZTRSM, cdouble_t)
	GPB_DEFINE_TRSM(SUITESPARSE_BLAS_CTRSM, cfloat_t)

	//--------------------------------- trsv -----------------------------------

#define GPB_DEFINE_TRSV(NAME, T)                                                       \
	void NAME(const char* uplo, const char* trans, const char* diag,                   \
		const blas_int* n, const T* A, const blas_int* lda, T* X,                      \
		const blas_int* incx) NOTHROW {                                                        \
		Trsv<T>(uplo, trans, diag, n, A, lda, X, incx);                                \
	}

	GPB_DEFINE_TRSV(SUITESPARSE_BLAS_DTRSV, double)
	GPB_DEFINE_TRSV(SUITESPARSE_BLAS_STRSV, float)
	GPB_DEFINE_TRSV(SUITESPARSE_BLAS_ZTRSV, cdouble_t)
	GPB_DEFINE_TRSV(SUITESPARSE_BLAS_CTRSV, cfloat_t)

	//-------------------------------- potrf -----------------------------------

#define GPB_DEFINE_POTRF(NAME, T)                                                      \
	void NAME(const char* uplo, const blas_int* n, T* A, const blas_int* lda,          \
		blas_int* info) NOTHROW {                                                              \
		Potrf<T>(uplo, n, A, lda, info);                                               \
	}

	GPB_DEFINE_POTRF(SUITESPARSE_LAPACK_DPOTRF, double)
	GPB_DEFINE_POTRF(SUITESPARSE_LAPACK_SPOTRF, float)
	GPB_DEFINE_POTRF(SUITESPARSE_LAPACK_ZPOTRF, cdouble_t)
	GPB_DEFINE_POTRF(SUITESPARSE_LAPACK_CPOTRF, cfloat_t)

}  // extern "C"

#endif  // GPB_SUITESPARSE_INTERNAL_BLAS
