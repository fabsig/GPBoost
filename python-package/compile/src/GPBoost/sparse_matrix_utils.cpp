/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/sparse_matrix_utils.h>
#include <GPBoost/log.h>

//#include <typeinfo> // Only needed for debugging
//#include <chrono>  // Only needed for debugging
//#include <thread> // Only needed for debugging

namespace GPBoost {

	void L_solve(const double* val, const int ncol, double* x) {
		for (int j = 0; j < ncol; ++j) {
			if (x[j] != 0) {
				x[j] /= val[j * ncol + j];
				for (int i = j + 1; i < ncol; ++i) {
					x[i] -= val[j * ncol + i] * x[j];
				}
			}
		}
	}

	void sp_L_solve(const double* val, const int* row_idx, const int* col_ptr, const int ncol, double* x) {
		for (int j = 0; j < ncol; ++j) {
			if (x[j] != 0) {
				x[j] /= val[col_ptr[j]];
				for (int i = col_ptr[j] + 1; i < col_ptr[j + 1]; ++i) {
					x[row_idx[i]] -= val[i] * x[j];
				}
			}
		}
	}

	void sp_L_t_solve(const double* val, const int* row_idx, const int* col_ptr, const int ncol, double* x) {
		for (int j = ncol - 1; j >= 0; --j) {
			for (int i = col_ptr[j] + 1; i < col_ptr[j + 1]; ++i) {
				x[j] -= val[i] * x[row_idx[i]];
			}
			x[j] /= val[col_ptr[j]];
		}
	}

	void sp_Lower_sp_RHS_cs_solve(cs* A, cs* B, sp_mat_t& A_inv_B, bool lower) {
		if (A->m != A->n || B->n < 1 || A->n < 1 || A->n != B->m) {
			Log::Fatal("Dimensions of system to be solved are inconsistent");
		}
		int lo = lower;

		//// Try to do parallel Version 1. (is slow!)
		//std::vector<std::vector<double>> val_cols(B->n);
		//std::vector<int> nnz_cols(A->n);
		//std::vector<std::vector<int>> row_idx_cols(B->n);
		//#pragma omp parallel for schedule(static)
		//for (int k = 0; k < B->n; k++) {
		//	std::vector<int> xi(2 * A->n);//TODO initialize outside of for loop
		//	std::vector<double> x(A->n);//TODO use pointers or arrays?
		//	//std::shared_ptr<int[]> xi(new int[2 * A->n]);
		//	//std::shared_ptr<double[]> x(new double[A->n]);
		//	//int* xi = new int(2 * A->n);// for cs_reach
		//	//double* x = new double(A->n);
		//	int top = cs_spsolve(A, B, k, xi.data(), x.data(), (int*)NULL, lo);
		//	int nz = A->n - top;
		//	//mutex_.lock();
		//	{
		//		std::lock_guard<std::mutex> lock(mutex_);
		//		nnz_cols[k] = nz;
		//		if (lo) {			/* increasing row order */
		//			for (int p = top; p < A->n; p++) {
		//				row_idx_cols[k].push_back(xi[p]);
		//				val_cols[k].push_back(x[xi[p]]);
		//			}
		//		}
		//		else {			/* decreasing order, reverse copy */
		//			for (int p = A->n - 1; p >= top; p--) {
		//				row_idx_cols[k].push_back(xi[p]);
		//				val_cols[k].push_back(x[xi[p]]);
		//			}
		//		}
		//	}
		//	//mutex_.unlock();
		//}
		//std::vector<double> val;
		//std::vector<int> col_ptr(A->n + 1);
		//std::vector<int> row_idx;
		//col_ptr[0] = 0;
		//for (int k = 0; k < B->n; k++) {
		//	col_ptr[k+1] = nnz_cols[k] + col_ptr[k];
		//	val.insert(val.end(), val_cols[k].begin(), val_cols[k].end());
		//	row_idx.insert(row_idx.end(), row_idx_cols[k].begin(), row_idx_cols[k].end());
		//}

		//// Try to do parallel Version 1b. (avoid n-time allocation of xi and x, but is still slow)
		//std::vector<std::vector<double>> val_cols(B->n);
		//std::vector<int> nnz_cols(A->n);
		//std::vector<std::vector<int>> row_idx_cols(B->n);
		//#pragma omp parallel
		//{
		//	std::vector<int> xi(2 * A->n);//TODO initialize outside of for loop
		//	std::vector<double> x(A->n);//TODO use pointers or arrays?
		//	//std::shared_ptr<int[]> xi(new int[2 * A->n]);
		//	//std::shared_ptr<double[]> x(new double[A->n]);
		//	//int* xi = new int(2 * A->n);// for cs_reach
		//	//double* x = new double(A->n);
		//	#pragma omp for nowait schedule(static)
		//	for (int k = 0; k < B->n; k++) {
		//		int top = cs_spsolve(A, B, k, xi.data(), x.data(), (int*)NULL, lo);
		//		int nz = A->n - top;
		//		//mutex_.lock();
		//		{
		//			std::lock_guard<std::mutex> lock(mutex_);
		//			nnz_cols[k] = nz;
		//			if (lo) {			/* increasing row order */
		//				for (int p = top; p < A->n; p++) {
		//					row_idx_cols[k].push_back(xi[p]);
		//					val_cols[k].push_back(x[xi[p]]);
		//				}
		//			}
		//			else {			/* decreasing order, reverse copy */
		//				for (int p = A->n - 1; p >= top; p--) {
		//					row_idx_cols[k].push_back(xi[p]);
		//					val_cols[k].push_back(x[xi[p]]);
		//				}
		//			}
		//		}
		//		//mutex_.unlock();
		//	}
		//}
		//std::vector<double> val;
		//std::vector<int> col_ptr(A->n + 1);
		//std::vector<int> row_idx;
		//col_ptr[0] = 0;
		//for (int k = 0; k < B->n; k++) {
		//	col_ptr[k + 1] = nnz_cols[k] + col_ptr[k];
		//	val.insert(val.end(), val_cols[k].begin(), val_cols[k].end());
		//	row_idx.insert(row_idx.end(), row_idx_cols[k].begin(), row_idx_cols[k].end());
		//}

		//// Try to do parallel Version 2. (not working, crashes)
		//// See https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
		//// See also https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
		//std::vector<double> val;
		//std::vector<int> col_ptr(A->n + 1);
		//std::vector<int> nnz_cols;
		//std::vector<int> row_idx;
		//col_ptr[0] = 0;

		//#pragma omp parallel
		//{
		//	std::vector<double> val_thr;
		//	std::vector<int> nnz_cols_thr;
		//	std::vector<int> row_idx_thr;
		//	std::vector<int> xi(2 * A->n);//TODO use pointers or arrays?
		//	std::vector<double> x(A->n);
		//	#pragma omp for nowait schedule(static)
		//	for (int k = 0; k < B->n; k++) {
		//		int top = cs_spsolve(A, B, k, xi.data(), x.data(), (int*)NULL, lo);
		//		int nz = A->n - top;

		//		nnz_cols_thr.push_back(nz);
		//		if (lo) {			/* increasing row order */
		//			for (int p = top; p < A->n; p++) {
		//				row_idx_thr.push_back(xi[p]);
		//				val_thr.push_back(x[xi[p]]);
		//			}
		//		}
		//		else {			/* decreasing order, reverse copy */
		//			for (int p = A->n - 1; p >= top; p--) {
		//				row_idx_thr.push_back(xi[p]);
		//				val_thr.push_back(x[xi[p]]);
		//			}
		//		}
		//	}
		//	#pragma omp for schedule(static) ordered
		//	for (int i = 0; i < omp_get_num_threads(); i++) {
		//		#pragma omp ordered
		//		val.insert(val.end(), val_thr.begin(), val_thr.end());
		//		row_idx.insert(row_idx.end(), row_idx_thr.begin(), row_idx_thr.end());
		//		nnz_cols.insert(nnz_cols.end(), nnz_cols_thr.begin(), nnz_cols_thr.end());
		//	}
		//	for (int k = 0; k < B->n; k++) {
		//		col_ptr[k+1] = nnz_cols[k] + col_ptr[k];
		//	}
		//}

		std::vector<double> val;
		std::vector<int> col_ptr(A->n + 1);
		std::vector<int> row_idx;
		std::vector<int> xi(2 * A->n);
		std::vector<double> x(A->n);

		col_ptr[0] = 0;
		for (int k = 0; k < B->n; k++) {//TODO make the following in parallel???
			int top = cs_spsolve(A, B, k, xi.data(), x.data(), (int*)NULL, lo);
			int nz = A->n - top;

			col_ptr[k + 1] = nz + col_ptr[k];
			if (lo) {			/* increasing row order */
				for (int p = top; p < A->n; p++) {
					row_idx.push_back(xi[p]);
					val.push_back(x[xi[p]]);
				}
			}
			else {			/* decreasing order, reverse copy */
				for (int p = A->n - 1; p >= top; p--) {
					row_idx.push_back(xi[p]);
					val.push_back(x[xi[p]]);
				}
			}
		}

		//Log::Info("Fine here MAP TEST");//only for debugging
		//std::this_thread::sleep_for(std::chrono::milliseconds(200));

		//Crash can occurr here on Linux (gcc 7.5.0 on Ubuntu 18.04) when row_idx is not increasing for all columns
		//See https://gitlab.com/libeigen/eigen/-/issues/1852
		A_inv_B = Eigen::Map<sp_mat_t>(A->n, B->n, val.size(), col_ptr.data(), row_idx.data(), val.data());

		//Log::Info("Fine here END eigen_sp_Lower_sp_RHS_cs_solve");
		//std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}

	void eigen_sp_Lower_sp_RHS_cs_solve(sp_mat_t& A, sp_mat_t& B, sp_mat_t& A_inv_B, bool lower) {
		//Convert Eigen matrices to correct format
		A.makeCompressed();
		B.makeCompressed();

		 //Prepocessor flag: Workaround since problems can occurr when calling 'sp_Lower_sp_RHS_cs_solve' from Linux (likely also from macOS)
#ifdef _WIN32

		//This is much than the version below (in particular if B is very sparse) but it can crash on Linux
		//Prepare LHS
		cs L_cs = cs();
		L_cs.nzmax = (int)A.nonZeros();
		L_cs.m = (int)A.cols();
		L_cs.n = (int)A.rows();
		L_cs.p = reinterpret_cast<csi*>(A.outerIndexPtr());
		L_cs.i = reinterpret_cast<csi*>(A.innerIndexPtr());
		L_cs.x = A.valuePtr();
		L_cs.nz = -1;
		//Prepare RHS
		cs R_cs = cs();
		R_cs.nzmax = (int)B.nonZeros();
		R_cs.m = (int)B.rows();
		R_cs.n = (int)B.cols();
		R_cs.p = reinterpret_cast<csi*>(B.outerIndexPtr());
		R_cs.i = reinterpret_cast<csi*>(B.innerIndexPtr());
		R_cs.x = B.valuePtr();
		R_cs.nz = -1;

		sp_Lower_sp_RHS_cs_solve(&L_cs, &R_cs, A_inv_B, lower);

#else

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

#endif

	}
}  // namespace GPBoost
