/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/sparse_matrix_utils.h>

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

	void L_t_solve(const double* val, const int ncol, double* x) {
		for (int j = ncol - 1; j >= 0; --j) {
			if (x[j] != 0) {
				x[j] /= val[j * ncol + j];
				for (int i = 0; i < j; ++i) {
					x[i] -= val[i * ncol + j] * x[j];
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

	void sp_Lower_sp_RHS_cs_solve(cs* A, const cs* B, sp_mat_t& A_inv_B, bool lower) {
		if (A->m != A->n || B->n < 1 || A->n < 1 || A->n != B->m) {
			Log::REFatal("Dimensions of system to be solved are inconsistent");
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
		for (int k = 0; k < B->n; k++) {//TODO: make the following in parallel???
			int top = cs_spsolve(A, B, k, xi.data(), x.data(), (int*)NULL, lo);
			int nz = A->n - top;

			col_ptr[k + 1] = nz + col_ptr[k];


			////start to implement sorting of indices to avoid crash but currently not fully implemented (14.04.2021)
			//// see here for more info: https://stackoverflow.com/questions/33406432/how-to-sort-one-array-and-get-corresponding-second-array-in-c and https://stackoverflow.com/questions/1380463/sorting-a-vector-of-custom-objects
			//bool ComnparePairs(std::pair<int, double> p1, std::pair<int, double> p2)
			//{
			//	return (p1.first < p2.first);
			//}
			//std::vector<std::pair<int,double>> aux_vec_sort(nz);
			//for (int p = top; p < A->n; p++) {
			//	aux_vec_sort[p - top] = std::make_pair(xi[p], x[xi[p]]);
			//}
			//std::sort(aux_vec_sort.begin(), aux_vec_sort.end(), ComnparePairs);


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

		//Crash can occur here on Linux (gcc 7.5.0 on Ubuntu 18.04) when row_idx is not increasing for all columns. This seems to be a bug of Eigen
		//Update 23.04.2020: Problems can also occur on Windows
		//Bug report filed: https://gitlab.com/libeigen/eigen/-/issues/1852 and https://gitlab.com/libeigen/eigen/-/issues/2210
		A_inv_B = Eigen::Map<sp_mat_t>(A->n, B->n, val.size(), col_ptr.data(), row_idx.data(), val.data());

	}

	void eigen_sp_Lower_sp_RHS_cs_solve(const sp_mat_t& A_const, const sp_mat_t& B_const, sp_mat_t& A_inv_B, bool lower) {

		 //Prepocessor flag: Workaround since problems can occurr when calling 'sp_Lower_sp_RHS_cs_solve' from certain gcc versions (e.g. gcc 7.5.0 on Ubuntu 18.04); see comment above in sp_Lower_sp_RHS_cs_solve.
#if defined(_WIN32) && !defined(__GNUC__)

		sp_mat_t A = sp_mat_t(A_const);//need to copy since 'A_const' is const because 'CholFactMatrix()' returns const, but L_cs needs to be non-const for 'cs_spsolve' / 'sp_Lower_sp_RHS_cs_solve'
		sp_mat_t B = sp_mat_t(B_const);
		CHECK(A.isCompressed());
		CHECK(B.isCompressed());

		//This is faster than the version below (in particular if B is very sparse) but it can crash on Linux. Update 23.04.2020: Problems can also occur on Windows
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

		TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(A_const, B_const, A_inv_B, !lower);

#endif

	}//end eigen_sp_Lower_sp_RHS_cs_solve

	void eigen_sp_Lower_sp_RHS_cs_solve(const sp_mat_rm_t& A, const sp_mat_rm_t& B, sp_mat_rm_t& A_inv_B, bool lower) {//not used, place-holder for compiler
		sp_mat_t A_inv_B_cm = sp_mat_t(B);
		sp_mat_t A_cm = sp_mat_t(A);
		eigen_sp_Lower_sp_RHS_cs_solve(A_cm, A_inv_B_cm, A_inv_B_cm, lower);
		A_inv_B = sp_mat_rm_t(A_inv_B_cm);
	}

	void CalcZtVGivenIndices(const data_size_t num_data,
		const data_size_t num_re,
		const data_size_t* const random_effects_indices_of_data,
		const vec_t& v,
		vec_t& ZtV,
		bool initialize_zero) {
		if (initialize_zero) {
			ZtV = vec_t::Zero(num_re);
		}
#pragma omp parallel
		{
			vec_t Ztv_private = vec_t::Zero(num_re);
#pragma omp for
			for (data_size_t i = 0; i < num_data; ++i) {
				Ztv_private[random_effects_indices_of_data[i]] += v[i];
			}
#pragma omp critical
			{
				for (data_size_t i_re = 0; i_re < num_re; ++i_re) {
					ZtV[i_re] += Ztv_private[i_re];
				}
			}//end omp critical
		}//end omp parallel
		//Non-parallel version
		//for (data_size_t i = 0; i < num_data; ++i) {
		//	ZtV[random_effects_indices_of_data[i]] += v[i];
		//}
	}

}  // namespace GPBoost
