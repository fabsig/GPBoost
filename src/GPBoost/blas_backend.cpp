/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2026 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/blas_backend.h>
#include <GPBoost/utils.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <atomic>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace GPBoost {

	namespace {

		// All routines are nullptr as long as no OpenBLAS library is registered
		OpenBLASFunctions openblas;
		std::string openblas_config;
		int openblas_parallel = 0;
		std::atomic<bool> blas_switched_off(false);

		// Values of openblas_get_parallel()
		const int OPENBLAS_PTHREADS = 1;
		const int OPENBLAS_OPENMP = 2;

		// Calls with fewer floating point operations use a single thread of OpenBLAS, since starting more threads
		// costs more than they gain. For the Cholesky factorization, the break-even point is at about n = 800 on
		// a laptop with six performance cores, both for OpenBLAS with its own threads and with OpenMP
		const double MIN_FLOPS_MULTIPLE_THREADS = 1.5e8;

		// Smallest dimension for which 'CholeskyBuiltin()' uses its tiled algorithm, and the size of the tiles. With six
		//	threads, the tiled algorithm is 2.8 times faster than Eigen::LLT for n = 512 and 5.4 times for n = 2000
		const int MIN_DIM_TILED_CHOLESKY = 512;
		const int CHOLESKY_TILE_SIZE = 128;

		bool InParallelRegion() {
#ifdef _OPENMP
#if _OPENMP >= 200805
			// also an inactive region (a single thread) counts, so that whether OpenBLAS is used does not depend on
			// the number of threads
			return omp_get_level() > 0;
#else
			return omp_in_parallel() != 0;
#endif
#else
			return false;
#endif
		}

		/*!
		* \brief Calls an OpenBLAS routine with a single thread or with the number of threads of OpenMP, depending on
		*		the number of floating point operations
		*/
		template <typename Call>
		void CallWithThreads(double flops, const Call& call) {
			const int num_threads = flops < MIN_FLOPS_MULTIPLE_THREADS ? 1 : omp_get_max_threads();
			if (openblas_parallel == OPENBLAS_PTHREADS) {
				// OpenBLAS's own threads: their number is restored afterwards, so that other users of the library in the
				// process (e.g., scipy) are not affected
				const int num_threads_previous = openblas.get_num_threads();
				if (num_threads != num_threads_previous) {
					openblas.set_num_threads(num_threads);
				}
				call();
				if (num_threads != num_threads_previous) {
					openblas.set_num_threads(num_threads_previous);
				}
			}
			else if (openblas_parallel == OPENBLAS_OPENMP && num_threads < omp_get_max_threads()) {
				// OpenBLAS with OpenMP uses the number of threads of OpenMP (if it uses the same OpenMP runtime as GPBoost)
				ParallelThreadsScope single_thread(num_threads);
				call();
			}
			else {
				call();
			}
		}

		bool IsComplete(const OpenBLASFunctions& functions) {
			return functions.dgemm != nullptr && functions.dgemv != nullptr && functions.dsyrk != nullptr &&
				functions.dtrsm != nullptr && functions.dtrsv != nullptr && functions.dpotrf != nullptr &&
				functions.dpotri != nullptr && functions.get_parallel != nullptr &&
				functions.get_num_threads != nullptr && functions.set_num_threads != nullptr &&
				functions.get_config != nullptr;
		}

#if defined(_WIN32)
		typedef HMODULE LibraryHandle;

		LibraryHandle OpenLibrary(const std::string& path) {
			const int length = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
			if (length <= 0) {
				return nullptr;
			}
			std::wstring path_wide(length, L'\0');
			MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &path_wide[0], length);
			return LoadLibraryW(path_wide.c_str());
		}

		void CloseLibrary(LibraryHandle library) {
			FreeLibrary(library);
		}

		template <typename Function>
		void FindSymbol(LibraryHandle library, const std::string& name, Function& function) {
			FARPROC symbol = GetProcAddress(library, name.c_str());
			// copied since a cast between different function pointer types gives compiler warnings
			static_assert(sizeof(symbol) == sizeof(function), "unexpected size of a function pointer");
			std::memcpy(&function, &symbol, sizeof(function));
		}
#else
		typedef void* LibraryHandle;

		LibraryHandle OpenLibrary(const std::string& path) {
			return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
		}

		void CloseLibrary(LibraryHandle library) {
			dlclose(library);
		}

		template <typename Function>
		void FindSymbol(LibraryHandle library, const std::string& name, Function& function) {
			void* symbol = dlsym(library, name.c_str());
			// copied since a cast between object and function pointers gives compiler warnings
			static_assert(sizeof(symbol) == sizeof(function), "unexpected size of a function pointer");
			std::memcpy(&function, &symbol, sizeof(function));
		}
#endif

	}  // namespace

	bool LoadOpenBLAS(const std::string& library_path) {
		LibraryHandle library = OpenLibrary(library_path);
		if (library == nullptr) {
			return false;
		}
		for (const std::string prefix : { "", "scipy_" }) {
			OpenBLASFunctions functions;
			FindSymbol(library, prefix + "dgemm_", functions.dgemm);
			FindSymbol(library, prefix + "dgemv_", functions.dgemv);
			FindSymbol(library, prefix + "dsyrk_", functions.dsyrk);
			FindSymbol(library, prefix + "dtrsm_", functions.dtrsm);
			FindSymbol(library, prefix + "dtrsv_", functions.dtrsv);
			FindSymbol(library, prefix + "dpotrf_", functions.dpotrf);
			FindSymbol(library, prefix + "dpotri_", functions.dpotri);
			FindSymbol(library, prefix + "openblas_get_parallel", functions.get_parallel);
			FindSymbol(library, prefix + "openblas_get_num_threads", functions.get_num_threads);
			FindSymbol(library, prefix + "openblas_set_num_threads", functions.set_num_threads);
			FindSymbol(library, prefix + "openblas_get_config", functions.get_config);
			if (IsComplete(functions) && std::strstr(functions.get_config(), "USE64BITINT") == nullptr) {
				SetOpenBLAS(functions);
				// the library stays loaded for the rest of the process
				return true;
			}
		}
		CloseLibrary(library);
		return false;
	}

	void SetOpenBLAS(const OpenBLASFunctions& functions) {
		CHECK(IsComplete(functions));
		openblas = functions;
		openblas_config = functions.get_config();
		openblas_parallel = functions.get_parallel();
	}

	void SetBLAS(const std::string& blas) {
		if (blas == "auto") {
			blas_switched_off = false;
		}
		else if (blas == "builtin") {
			blas_switched_off = true;
		}
		else {
			Log::REFatal("BLAS '%s' is not supported. Use 'auto' or 'builtin' ", blas.c_str());
		}
	}

	std::string GetBLAS() {
		if (openblas.dpotrf == nullptr || blas_switched_off) {
			return "builtin";
		}
		return openblas_config;
	}

	bool UseBLAS() {
		return openblas.dpotrf != nullptr && !blas_switched_off && !InParallelRegion();
	}

	bool UseTiledCholesky(int n) {
		return n >= MIN_DIM_TILED_CHOLESKY && !InParallelRegion();
	}

	int CholeskyBuiltin(double* A, int n, int lda) {
		typedef Eigen::Map<den_mat_t, 0, Eigen::OuterStride<>> MapMat;
		if (!UseTiledCholesky(n)) {
			MapMat A_map(A, n, n, Eigen::OuterStride<>(lda));
			return (int)Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(A_map);
		}
		const int num_tiles = (n + CHOLESKY_TILE_SIZE - 1) / CHOLESKY_TILE_SIZE;
		auto tile = [A, n, lda](int i, int j) {
			const int first_row = i * CHOLESKY_TILE_SIZE;
			const int first_col = j * CHOLESKY_TILE_SIZE;
			return MapMat(A + (size_t)first_col * lda + first_row, std::min(CHOLESKY_TILE_SIZE, n - first_row),
				std::min(CHOLESKY_TILE_SIZE, n - first_col), Eigen::OuterStride<>(lda));
		};
		// right-looking: factorize the diagonal tile k, solve for the tiles below it, and update the trailing tiles
		for (int k = 0; k < num_tiles; ++k) {
			MapMat A_kk = tile(k, k);
			const Eigen::Index failed_col = Eigen::internal::llt_inplace<double, Eigen::Lower>::blocked(A_kk);
			if (failed_col >= 0) {
				return k * CHOLESKY_TILE_SIZE + (int)failed_col;
			}
#pragma omp parallel for schedule(dynamic, 1)
			for (int i = k + 1; i < num_tiles; ++i) {
				MapMat A_ik = tile(i, k);
				A_kk.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(A_ik);
			}
			// the tiles (i, j) with k < j <= i, enumerated row by row
			const int num_trailing = num_tiles - k - 1;
			const int num_updates = num_trailing * (num_trailing + 1) / 2;
#pragma omp parallel for schedule(dynamic, 1)
			for (int p = 0; p < num_updates; ++p) {
				int i = k + 1;
				int index_in_row = p;
				while (index_in_row >= i - k) {
					index_in_row -= i - k;
					++i;
				}
				const int j = k + 1 + index_in_row;
				MapMat A_ij = tile(i, j);
				const MapMat A_ik = tile(i, k);
				if (i == j) {
					A_ij.selfadjointView<Eigen::Lower>().rankUpdate(A_ik, -1.);
				}
				else {
					A_ij.noalias() -= A_ik * tile(j, k).transpose();
				}
			}
		}
		return -1;
	}

	bool CholeskyBLAS(double* A, int n) {
		// LAPACK requires a leading dimension of at least one, also for an empty matrix
		const int lda = std::max(n, 1);
		int info = 0;
		CallWithThreads((double)n * n * n / 3., [&]() {
			openblas.dpotrf("L", &n, A, &lda, &info, 1);
		});
		CHECK(info >= 0);
		return info == 0;
	}

	bool CholeskyInverseBLAS(double* A, int n) {
		const int lda = std::max(n, 1);
		int info = 0;
		CallWithThreads(2. * n * n * n / 3., [&]() {
			openblas.dpotri("L", &n, A, &lda, &info, 1);
		});
		CHECK(info >= 0);
		return info == 0;
	}

	void TriangularSolveBLAS(const double* L, int n, double* B, int num_cols, bool transpose) {
		const int lda = std::max(n, 1);
		const double one = 1.;
		const char* transa = transpose ? "T" : "N";
		CallWithThreads((double)n * n * num_cols, [&]() {
			openblas.dtrsm("L", "L", transa, "N", &n, &num_cols, &one, L, &lda, B, &lda, 1, 1, 1, 1);
		});
	}

	void DgemmBLAS(const char* transa, const char* transb, const int* m, const int* n, const int* k,
		const double* alpha, const double* A, const int* lda, const double* B, const int* ldb,
		const double* beta, double* C, const int* ldc) {
		CallWithThreads(2. * (*m) * (*n) * (*k), [&]() {
			openblas.dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, 1);
		});
	}

	void DgemvBLAS(const char* trans, const int* m, const int* n, const double* alpha, const double* A,
		const int* lda, const double* x, const int* incx, const double* beta, double* y, const int* incy) {
		CallWithThreads(2. * (*m) * (*n), [&]() {
			openblas.dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, 1);
		});
	}

	void DsyrkBLAS(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha,
		const double* A, const int* lda, const double* beta, double* C, const int* ldc) {
		CallWithThreads((double)(*n) * (*n) * (*k), [&]() {
			openblas.dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc, 1, 1);
		});
	}

	void DtrsmBLAS(const char* side, const char* uplo, const char* transa, const char* diag, const int* m,
		const int* n, const double* alpha, const double* A, const int* lda, double* B, const int* ldb) {
		// the triangular matrix is m x m if it is on the left and n x n otherwise
		const bool left = (*side == 'L' || *side == 'l');
		const double flops = left ? (double)(*m) * (*m) * (*n) : (double)(*m) * (*n) * (*n);
		CallWithThreads(flops, [&]() {
			openblas.dtrsm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb, 1, 1, 1, 1);
		});
	}

	void DtrsvBLAS(const char* uplo, const char* trans, const char* diag, const int* n, const double* A,
		const int* lda, double* x, const int* incx) {
		CallWithThreads((double)(*n) * (*n), [&]() {
			openblas.dtrsv(uplo, trans, diag, n, A, lda, x, incx, 1, 1, 1);
		});
	}

	void DpotrfBLAS(const char* uplo, const int* n, double* A, const int* lda, int* info) {
		CallWithThreads((double)(*n) * (*n) * (*n) / 3., [&]() {
			openblas.dpotrf(uplo, n, A, lda, info, 1);
		});
	}

}  // namespace GPBoost
