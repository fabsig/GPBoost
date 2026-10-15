/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/sparse_matrix_utils.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

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

	void CalcDiagLInvDLInvT(const sp_mat_t& L, const vec_t& d, vec_t& diag) {
		CHECK(L.isCompressed());
		CHECK(L.rows() == L.cols());
		CHECK(L.cols() == d.size());
		const int n = (int)L.cols();
		const double* val = L.valuePtr();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		int num_threads;
#ifdef _OPENMP
		num_threads = omp_get_max_threads();
#else
		num_threads = 1;
#endif
		// The partial sums of every thread are added in a fixed order such that the result does not depend on the scheduling
		std::vector<vec_t> diag_private(num_threads);
#pragma omp parallel
		{
			int thread_nb;
#ifdef _OPENMP
			thread_nb = omp_get_thread_num();
#else
			thread_nb = 0;
#endif
			vec_t& diag_local = diag_private[thread_nb];
			diag_local = vec_t::Zero(n);
			// Column j of L^(-1). Its non-zeros are visited in increasing order of their row with a heap, since the pattern is not
			//	known in advance, and every entry is set back to zero once it has been used. No zeros are scanned, so the cost is
			//	proportional to the number of non-zeros of L^(-1) (n for a diagonal L)
			vec_t x = vec_t::Zero(n);
			std::vector<char> in_heap(n, 0);
			std::priority_queue<int, std::vector<int>, std::greater<int>> heap;
			// Cyclic scheduling: every thread gets the same columns in every run, so the result does not depend on timing
#pragma omp for schedule(static, 1)
			for (int j = 0; j < n; ++j) {
				x[j] = 1.;
				heap.push(j);
				in_heap[j] = 1;
				while (!heap.empty()) {
					const int k = heap.top();
					heap.pop();
					in_heap[k] = 0;
					if (x[k] != 0.) {
						const double x_k = x[k] / val[col_ptr[k]];
						x[k] = 0.;
						for (int i = col_ptr[k] + 1; i < col_ptr[k + 1]; ++i) {
							const int r = row_idx[i];
							x[r] -= val[i] * x_k;
							if (!in_heap[r]) {
								heap.push(r);
								in_heap[r] = 1;
							}
						}
						diag_local[k] += d[j] * x_k * x_k;
					}
				}
			}
		}
		diag = vec_t::Zero(n);
		for (const vec_t& diag_local : diag_private) {
			if (diag_local.size() == n) {
				diag += diag_local;
			}
		}
	}//end CalcDiagLInvDLInvT

	void CalcLInvDLInvT(const sp_mat_t& L, const vec_t& d, den_mat_t& M) {
		CHECK(L.isCompressed());
		CHECK(L.rows() == L.cols());
		CHECK(L.cols() == d.size());
		const int n = (int)L.cols();
		// M = L^(-1) diag(d) first and then M = L^(-1) M^T = L^(-1) diag(d) L^(-T), both in place
		M = den_mat_t::Zero(n, n);
		M.diagonal() = d;
		TriangularSolve<sp_mat_t, den_mat_t, den_mat_t>(L, M, M, false);
		M.transposeInPlace();
		TriangularSolve<sp_mat_t, den_mat_t, den_mat_t>(L, M, M, false);
	}//end CalcLInvDLInvT

	void CheckArgumentsCalcColumnRtLInvT(const sp_mat_t* L, const sp_mat_rm_t& R, int num_columns) {
		CHECK(R.isCompressed());
		CHECK(num_columns >= 0 && num_columns <= R.rows());
		if (L != nullptr) {
			CHECK(L->isCompressed());
			CHECK(L->rows() == L->cols());
			CHECK(L->rows() == R.rows());
		}
	}//end CheckArgumentsCalcColumnRtLInvT

	void CalcColumnRtLInvT(const sp_mat_t* L, const sp_mat_rm_t& R, int i, vec_t& column) {
		column = vec_t::Zero(R.cols());
		const int* r_row_ptr = R.outerIndexPtr();
		const int* r_col_idx = R.innerIndexPtr();
		const double* r_val = R.valuePtr();
		if (L == nullptr) {
			for (int q = r_row_ptr[i]; q < r_row_ptr[i + 1]; ++q) {
				column[r_col_idx[q]] = r_val[q];
			}
			return;
		}
		const double* val = L->valuePtr();
		const int* row_idx = L->innerIndexPtr();
		const int* col_ptr = L->outerIndexPtr();
		// x = L^(-T) e_i by backward substitution over the columns i, ..., 0 of L, x is zero below row i
		vec_t x = vec_t::Zero(i + 1);
		x[i] = 1.;
		for (int j = i; j >= 0; --j) {
			double x_j = x[j];
			for (int q = col_ptr[j] + 1; q < col_ptr[j + 1]; ++q) {
				if (row_idx[q] <= i) {
					x_j -= val[q] * x[row_idx[q]];
				}
			}
			x[j] = x_j / val[col_ptr[j]];
		}
		// R^T x as the sum over the rows r <= i of R with x_r != 0
		for (int r = 0; r <= i; ++r) {
			if (x[r] != 0.) {
				for (int q = r_row_ptr[r]; q < r_row_ptr[r + 1]; ++q) {
					column[r_col_idx[q]] += x[r] * r_val[q];
				}
			}
		}
	}//end CalcColumnRtLInvT

	void CalcColSquaredNormsLInvR(const sp_mat_t& L, const sp_mat_t& R, vec_t& norms) {
		CHECK(L.isCompressed());
		CHECK(L.rows() == L.cols());
		CHECK(L.cols() == R.rows());
		const int n = (int)L.cols();
		const int num_cols = (int)R.cols();
		const double* val = L.valuePtr();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		norms = vec_t::Zero(num_cols);
#pragma omp parallel
		{
			// Column of L^(-1) R. Its non-zeros are visited in increasing order of their row with a heap, see 'CalcDiagLInvDLInvT',
			//	and every entry is set back to zero once it has been used
			vec_t x = vec_t::Zero(n);
			std::vector<char> in_heap(n, 0);
			std::priority_queue<int, std::vector<int>, std::greater<int>> heap;
#pragma omp for schedule(dynamic, 16)
			for (int j = 0; j < num_cols; ++j) {
				for (sp_mat_t::InnerIterator it(R, j); it; ++it) {
					const int r = (int)it.row();
					x[r] = it.value();
					if (!in_heap[r]) {
						heap.push(r);
						in_heap[r] = 1;
					}
				}
				double norm = 0.;
				while (!heap.empty()) {
					const int k = heap.top();
					heap.pop();
					in_heap[k] = 0;
					if (x[k] != 0.) {
						const double x_k = x[k] / val[col_ptr[k]];
						x[k] = 0.;
						for (int i = col_ptr[k] + 1; i < col_ptr[k + 1]; ++i) {
							const int r = row_idx[i];
							x[r] -= val[i] * x_k;
							if (!in_heap[r]) {
								heap.push(r);
								in_heap[r] = 1;
							}
						}
						norm += x_k * x_k;
					}
				}
				norms[j] = norm;
			}
		}
	}//end CalcColSquaredNormsLInvR

	void CalcColSquaredNormsLInvRByRows(const sp_mat_rm_t& L, const sp_mat_rm_t& R, vec_t& norms) {
		CHECK(L.isCompressed());
		CHECK(R.isCompressed());
		CHECK(L.rows() == L.cols());
		CHECK(L.rows() == R.rows());
		const int n = (int)L.rows();
		const int num_cols = (int)R.cols();
		const double* val = L.valuePtr();
		const int* col_idx = L.innerIndexPtr();
		const int* row_ptr = L.outerIndexPtr();
		for (int k = 0; k < n; ++k) {
			CHECK(row_ptr[k] < row_ptr[k + 1] && col_idx[row_ptr[k + 1] - 1] == k);
		}
		const double* r_val = R.valuePtr();
		const int* r_col_idx = R.innerIndexPtr();
		const int* r_row_ptr = R.outerIndexPtr();
		int num_threads;
#ifdef _OPENMP
		num_threads = omp_get_max_threads();
#else
		num_threads = 1;
#endif
		// The partial sums of every thread are added in a fixed order such that the result does not depend on the scheduling
		std::vector<vec_t> norms_private(num_threads);
#pragma omp parallel
		{
			int thread_nb;
#ifdef _OPENMP
			thread_nb = omp_get_thread_num();
#else
			thread_nb = 0;
#endif
			vec_t& norms_local = norms_private[thread_nb];
			norms_local = vec_t::Zero(num_cols);
			// x = L^(-T) e_k, i.e., row k of L^(-1), by backward substitution. Its non-zeros are visited in decreasing order of their
			//	row with a heap, and every entry is set back to zero once it has been used
			vec_t x = vec_t::Zero(n);
			std::vector<char> in_heap(n, 0);
			std::priority_queue<int> heap;
			// Row k of L^(-1) R, of which only the non-zero entries are used and reset afterwards
			vec_t row_k = vec_t::Zero(num_cols);
			std::vector<char> is_non_zero(num_cols, 0);
			std::vector<int> non_zeros;
			// Cyclic scheduling: every thread gets the same rows in every run, so the result does not depend on timing
#pragma omp for schedule(static, 16)
			for (int k = 0; k < n; ++k) {
				x[k] = 1.;
				heap.push(k);
				in_heap[k] = 1;
				while (!heap.empty()) {
					const int q = heap.top();
					heap.pop();
					in_heap[q] = 0;
					if (x[q] != 0.) {
						const double x_q = x[q] / val[row_ptr[q + 1] - 1];
						x[q] = 0.;
						for (int p = row_ptr[q]; p < row_ptr[q + 1] - 1; ++p) {
							const int j = col_idx[p];
							x[j] -= val[p] * x_q;
							if (!in_heap[j]) {
								heap.push(j);
								in_heap[j] = 1;
							}
						}
						for (int p = r_row_ptr[q]; p < r_row_ptr[q + 1]; ++p) {
							const int c = r_col_idx[p];
							if (!is_non_zero[c]) {
								is_non_zero[c] = 1;
								non_zeros.push_back(c);
							}
							row_k[c] += x_q * r_val[p];
						}
					}
				}
				for (int c : non_zeros) {
					norms_local[c] += row_k[c] * row_k[c];
					row_k[c] = 0.;
					is_non_zero[c] = 0;
				}
				non_zeros.clear();
			}
		}
		norms = vec_t::Zero(num_cols);
		for (const vec_t& norms_local : norms_private) {
			if (norms_local.size() == num_cols) {
				norms += norms_local;
			}
		}
	}//end CalcColSquaredNormsLInvRByRows

	/*!
	* \brief Elimination tree of a Cholesky factor (with the diagonal as first entry of every column): the parent of node j is
	*		the first row below the diagonal in column j, and it is -1 for a root
	*/
	static void EliminationTreeCholFactor(const sp_mat_t& L, std::vector<int>& parent) {
		const int n = (int)L.cols();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		parent.assign(n, -1);
#pragma omp parallel for schedule(static)
		for (int j = 0; j < n; ++j) {
			int par = -1;
			for (int p = col_ptr[j] + 1; p < col_ptr[j + 1]; ++p) {
				if (par < 0 || row_idx[p] < par) {
					par = row_idx[p];
				}
			}
			parent[j] = par;
		}
	}//end EliminationTreeCholFactor

	void CalcColSquaredNormsCholFactorInvR(const sp_mat_t& L, const sp_mat_t& R, vec_t& norms) {
		CHECK(L.isCompressed());
		CHECK(L.rows() == L.cols());
		CHECK(L.cols() == R.rows());
		const int n = (int)L.cols();
		const int num_cols = (int)R.cols();
		const double* val = L.valuePtr();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		std::vector<int> parent;
		EliminationTreeCholFactor(L, parent);
		norms = vec_t::Zero(num_cols);
#pragma omp parallel
		{
			// Column of L^(-1) R, every entry is set back to zero once it has been used
			vec_t x = vec_t::Zero(n);
			std::vector<char> visited(n, 0);
			// The nodes on the paths to the root are pushed on a stack such that the stack is a topological order:
			//	a path that is found later ends below a node of a path that has been found earlier and is placed above it
			std::vector<int> stack(n), path(n);
#pragma omp for schedule(dynamic, 16)
			for (int j = 0; j < num_cols; ++j) {
				int top = n;
				for (sp_mat_t::InnerIterator it(R, j); it; ++it) {
					x[it.row()] = it.value();
					int len = 0;
					for (int k = (int)it.row(); k >= 0 && !visited[k]; k = parent[k]) {
						visited[k] = 1;
						path[len++] = k;
					}
					while (len > 0) {
						stack[--top] = path[--len];
					}
				}
				double norm = 0.;
				for (int p = top; p < n; ++p) {
					const int k = stack[p];
					if (x[k] != 0.) {
						const double x_k = x[k] / val[col_ptr[k]];
						for (int i = col_ptr[k] + 1; i < col_ptr[k + 1]; ++i) {
							x[row_idx[i]] -= val[i] * x_k;
						}
						norm += x_k * x_k;
						x[k] = 0.;
					}
					visited[k] = 0;
				}
				norms[j] = norm;
			}
		}
	}//end CalcColSquaredNormsCholFactorInvR

	void CalcColSquaredNormsLInvPRColumnwise(const sp_mat_t& L, const Eigen::VectorXi& perm_indices, int num_cols,
		const std::function<void(int, vec_t&)>& calc_column, vec_t& norms) {
		CHECK(L.isCompressed());
		CHECK(L.rows() == L.cols());
		const int n = (int)L.cols();
		const bool has_permutation = perm_indices.size() > 0;
		if (has_permutation) {
			CHECK((int)perm_indices.size() == n);
		}
		const double* val = L.valuePtr();
		const int* row_idx = L.innerIndexPtr();
		const int* col_ptr = L.outerIndexPtr();
		norms.resize(num_cols);
		bool wrong_length = false;
#pragma omp parallel
		{
			vec_t r(n), x(n);
#pragma omp for schedule(dynamic, 16) reduction(||:wrong_length)
			for (int j = 0; j < num_cols; ++j) {
				calc_column(j, r);
				if ((int)r.size() != n) {
					wrong_length = true;
					continue;
				}
				if (has_permutation) {
					for (int i = 0; i < n; ++i) {
						x[perm_indices[i]] = r[i];
					}
					sp_L_solve(val, row_idx, col_ptr, n, x.data());
					norms[j] = x.squaredNorm();
				}
				else {
					sp_L_solve(val, row_idx, col_ptr, n, r.data());
					norms[j] = r.squaredNorm();
				}
			}
		}
		if (wrong_length) {
			Log::REFatal("CalcColSquaredNormsLInvPRColumnwise: a column of the right-hand side does not have the length of L ");
		}
	}//end CalcColSquaredNormsLInvPRColumnwise

	/*!
	* \brief First position in the sorted range [first, last) whose value is not smaller than 'value'. The search starts at
	*		'first' with exponentially increasing steps, which is fast if the position is close to 'first'
	*/
	static inline const int* GallopingLowerBound(const int* first, const int* last, int value) {
		if (first == last || *first >= value) {
			return first;
		}
		const int* lo = first;// *lo < value
		const int* hi;
		size_t step = 1;
		while (true) {
			if ((size_t)(last - lo) <= step) {
				hi = last;
				break;
			}
			hi = lo + step;
			if (*hi >= value) {
				break;
			}
			lo = hi;
			step *= 2;
		}
		return std::lower_bound(lo + 1, hi, value);
	}

	/*!
	* \brief Supernodes of a Cholesky factor (with sorted row indices and the diagonal as first entry of every column): column j + 1
	*		belongs to the supernode of column j if the rows of column j are j and the rows of column j + 1
	* \param L Cholesky factor
	* \param[out] supernode_start First columns of the supernodes, followed by the number of columns
	*/
	static void SupernodesCholFactor(const sp_mat_t& L, std::vector<int>& supernode_start) {
		const int n = (int)L.cols();
		const int* col_ptr = L.outerIndexPtr();
		const int* row_idx = L.innerIndexPtr();
		supernode_start.assign(1, 0);
		if (n == 0) {
			return;
		}
		for (int j = 0; j < n - 1; ++j) {
			const int nnz_j = col_ptr[j + 1] - col_ptr[j];
			bool same_supernode = nnz_j >= 2 && nnz_j == col_ptr[j + 2] - col_ptr[j + 1] + 1 && row_idx[col_ptr[j] + 1] == j + 1;
			if (same_supernode) {
				same_supernode = std::equal(row_idx + col_ptr[j] + 2, row_idx + col_ptr[j + 1], row_idx + col_ptr[j + 1] + 1);
			}
			if (!same_supernode) {
				supernode_start.push_back(j + 1);
			}
		}
		supernode_start.push_back(n);
	}//end SupernodesCholFactor

	/*! \brief A supernode with t columns and r rows below it is processed with parallel matrix-matrix products if this is true */
	static inline bool IsLargeSupernode(double t, double r) {
		return t * (r * r + r * t + t * t) > 1e7;
	}

	void CostsDiagonalCholFactorInverse(const sp_mat_t& L, double& cost_solves, double& cost_selected_inversion) {
		const int n = (int)L.cols();
		const int* col_ptr = L.outerIndexPtr();
		int num_threads;
#ifdef _OPENMP
		num_threads = omp_get_max_threads();
#else
		num_threads = 1;
#endif
		// The constants are measured run times relative to the time of one operation of the sequential solves. The solves are
		//	parallelized over the columns, the selected inversion only for large supernodes, and these are processed with dense
		//	operations, which are faster per operation but do not scale as well with the number of threads. The solves are up to
		//	2.5 times slower per operation if the factor does not fit into the cache, since every solve runs through a large part
		//	of it, while the dense operations work on blocks
		const double cost_per_supernode = 100.;
		const double speedup_large_supernodes = 5. * std::pow((double)num_threads, 0.75);
		const double memory_factor_solves = 1. + 1.5 * std::min(1., std::max(0., ((double)L.nonZeros() - 1.5e6) / 1.5e6));
		// A solve with the column k of the identity reaches all ancestors of k in the elimination tree, i.e. column j is used in the
		//	solves for all nodes in the subtree of j
		std::vector<int> parent;
		EliminationTreeCholFactor(L, parent);
		std::vector<double> subtree_size(n, 1.);
		cost_solves = 0.;
		for (int j = 0; j < n; ++j) {
			cost_solves += subtree_size[j] * (double)(col_ptr[j + 1] - col_ptr[j]);
			if (parent[j] >= 0) {
				subtree_size[parent[j]] += subtree_size[j];
			}
		}
		cost_solves *= memory_factor_solves / (double)num_threads;
		std::vector<int> supernode_start;
		SupernodesCholFactor(L, supernode_start);
		const int num_supernodes = (int)supernode_start.size() - 1;
		cost_selected_inversion = cost_per_supernode * (double)num_supernodes;
		for (int sn = 0; sn < num_supernodes; ++sn) {
			const double t = (double)(supernode_start[sn + 1] - supernode_start[sn]);
			const int last_col = supernode_start[sn + 1] - 1;
			const double r = (double)(col_ptr[last_col + 1] - col_ptr[last_col] - 1);
			const double cost_dense = t * (r * r + r * t + t * t);
			// gathering Z[S, S] and the dense operations
			cost_selected_inversion += r * r + (IsLargeSupernode(t, r) ? cost_dense / speedup_large_supernodes : cost_dense);
		}
	}//end CostsDiagonalCholFactorInverse

	void SelectedInversionCholesky(const sp_mat_t& L, sp_mat_t& Z_lower, int max_num_rows_dense) {
		CHECK(L.isCompressed());
		CHECK(L.rows() == L.cols());
		CHECK(max_num_rows_dense > 0);
		const int* col_ptr = L.outerIndexPtr();
		const int* row_idx = L.innerIndexPtr();
		const double* l_val = L.valuePtr();
		Z_lower = L;
		double* z_val = Z_lower.valuePtr();
		// Z = (L L^T)^(-1) satisfies Z L = L^(-T), which is upper triangular. For a supernode, i.e. consecutive columns J that
		//	have the same rows S below J, this gives
		//		Z[S, J] = -Z[S, S] Y and Z[J, J] = L[J, J]^(-T) L[J, J]^(-1) + Y^T Z[S, S] Y with Y = L[S, J] L[J, J]^(-1).
		//	Z[S, S] is on the sparsity pattern of L, since the pattern of a Cholesky factor contains its fill-in, and it is known
		//	when the supernodes are processed from the last to the first one. Its rows are gathered in panels S_b of at most
		//	max_num_rows_dense^2 entries: Z[S_b, J] = -Z[S_b, S] Y, and Z[J, J] is accumulated as Y[S_b, J]^T (Z[S_b, S] Y)
		bool pattern_not_closed = false;
		std::vector<int> supernode_start;
		SupernodesCholFactor(L, supernode_start);
		den_mat_t Z_panel, Z_panel_Y, L_JJ, L_SJ, Y, L_JJ_inv, Z_JJ;
		for (int sn = (int)supernode_start.size() - 2; sn >= 0; --sn) {
			const int j0 = supernode_start[sn];
			const int t = supernode_start[sn + 1] - j0;
			// The rows S below the supernode are the rows below the diagonal in its last column
			const int* S = row_idx + col_ptr[j0 + t - 1] + 1;
			const int r = col_ptr[j0 + t] - col_ptr[j0 + t - 1] - 1;
			// Parallel computations only pay off for large supernodes, for the many small ones the overhead of starting
			//	the threads dominates
			const bool large_supernode = IsLargeSupernode((double)t, (double)r);
			// Column j0 + p of L has the rows j0 + p, ..., j0 + t - 1 and then the rows S
			L_JJ = den_mat_t::Zero(t, t);
			L_SJ.resize(r, t);
			for (int p = 0; p < t; ++p) {
				const int c = col_ptr[j0 + p];
				for (int q = p; q < t; ++q) {
					L_JJ(q, p) = l_val[c + q - p];
				}
				for (int a = 0; a < r; ++a) {
					L_SJ(a, p) = l_val[c + t - p + a];
				}
			}
			if (large_supernode) {
				// Y and Z[J, J] = L[J, J]^(-T) L[J, J]^(-1) are calculated with triangular solves, which Eigen does not parallelize,
				//	for blocks of rows and columns in parallel. Column p of L[J, J]^(-1) is zero above row p
				const int block_size = 64;
				if (r > 0) {
					Y.swap(L_SJ);
					const int num_blocks_S = (r + block_size - 1) / block_size;
#pragma omp parallel for schedule(dynamic, 1)
					for (int b = 0; b < num_blocks_S; ++b) {
						const int a0 = b * block_size;
						L_JJ.triangularView<Eigen::Lower>().solveInPlace<Eigen::OnTheRight>(Y.middleRows(a0, std::min(block_size, r - a0)));
					}
				}
				Z_JJ.setIdentity(t, t);
				const int num_blocks_J = (t + block_size - 1) / block_size;
#pragma omp parallel for schedule(dynamic, 1)
				for (int b = 0; b < num_blocks_J; ++b) {
					const int p0 = b * block_size;
					const int w = std::min(block_size, t - p0);
					L_JJ.bottomRightCorner(t - p0, t - p0).triangularView<Eigen::Lower>().solveInPlace(Z_JJ.block(p0, p0, t - p0, w));
					L_JJ.triangularView<Eigen::Lower>().transpose().solveInPlace(Z_JJ.middleCols(p0, w));
				}
			}
			else {
				// lazy products are not parallelized
				L_JJ_inv = L_JJ.triangularView<Eigen::Lower>().solve(den_mat_t::Identity(t, t));
				Z_JJ.noalias() = L_JJ_inv.transpose().lazyProduct(L_JJ_inv);
				if (r > 0) {
					Y.noalias() = L_SJ.lazyProduct(L_JJ_inv);
				}
			}
			// Z(S[a], S[c]) is in column S[min(a, c)] at row S[max(a, c)]. For the row a of a panel, the entries with c >= a are
			//	gathered from column S[a], and they are also the entries (c, a) of the panel if c is in the panel. The entries
			//	with c before the panel are gathered from column S[c]. The two loops write disjoint entries of the panel
			const int num_rows_panel = (r <= max_num_rows_dense) ? r :
				std::max(1, (int)((double)max_num_rows_dense * (double)max_num_rows_dense / (double)r));
			for (int a0 = 0; a0 < r; a0 += num_rows_panel) {
				const int nb = std::min(num_rows_panel, r - a0);
				Z_panel.resize(nb, r);
				auto gather_row = [&](int a) -> bool {
					const int* rows_a_end = row_idx + col_ptr[S[a] + 1];
					const int* pos = row_idx + col_ptr[S[a]];
					bool found_all = true;
					for (int c = a; c < r; ++c) {
						pos = GallopingLowerBound(pos, rows_a_end, S[c]);
						double z_ac = 0.;
						if (pos == rows_a_end || *pos != S[c]) {
							found_all = false;
						}
						else {
							z_ac = z_val[pos - row_idx];
						}
						Z_panel(a - a0, c) = z_ac;
						if (c < a0 + nb) {
							Z_panel(c - a0, a) = z_ac;
						}
					}
					return found_all;
				};
				auto gather_column_before_panel = [&](int c) -> bool {
					const int* rows_c_end = row_idx + col_ptr[S[c] + 1];
					const int* pos = std::lower_bound(row_idx + col_ptr[S[c]], rows_c_end, S[a0]);
					bool found_all = true;
					for (int a = a0; a < a0 + nb; ++a) {
						pos = GallopingLowerBound(pos, rows_c_end, S[a]);
						double z_ac = 0.;
						if (pos == rows_c_end || *pos != S[a]) {
							found_all = false;
						}
						else {
							z_ac = z_val[pos - row_idx];
						}
						Z_panel(a - a0, c) = z_ac;
					}
					return found_all;
				};
				if (r > 2048) {
#pragma omp parallel for schedule(dynamic, 8) reduction(||:pattern_not_closed)
					for (int a = a0; a < a0 + nb; ++a) {
						if (!gather_row(a)) {
							pattern_not_closed = true;
						}
					}
#pragma omp parallel for schedule(dynamic, 8) reduction(||:pattern_not_closed)
					for (int c = 0; c < a0; ++c) {
						if (!gather_column_before_panel(c)) {
							pattern_not_closed = true;
						}
					}
				}
				else {
					for (int a = a0; a < a0 + nb; ++a) {
						if (!gather_row(a)) {
							pattern_not_closed = true;
						}
					}
					for (int c = 0; c < a0; ++c) {
						if (!gather_column_before_panel(c)) {
							pattern_not_closed = true;
						}
					}
				}
				if (large_supernode) {
					Z_panel_Y.noalias() = Z_panel * Y;
					Z_JJ.noalias() += Y.middleRows(a0, nb).transpose() * Z_panel_Y;
				}
				else {
					Z_panel_Y.noalias() = Z_panel.lazyProduct(Y);
					Z_JJ.noalias() += Y.middleRows(a0, nb).transpose().lazyProduct(Z_panel_Y);
				}
				// Z[S_b, J] = -Z[S_b, S] Y
				for (int p = 0; p < t; ++p) {
					const int c = col_ptr[j0 + p];
					for (int a = 0; a < nb; ++a) {
						z_val[c + t - p + a0 + a] = -Z_panel_Y(a, p);
					}
				}
			}
			for (int p = 0; p < t; ++p) {
				const int c = col_ptr[j0 + p];
				for (int q = p; q < t; ++q) {
					z_val[c + q - p] = Z_JJ(q, p);
				}
			}
		}
		if (pattern_not_closed) {
			Log::REFatal("SelectedInversionCholesky: the sparsity pattern of the Cholesky factor does not contain its fill-in ");
		}
	}//end SelectedInversionCholesky

	SparseInverseEntries::SparseInverseEntries(const sp_mat_t& L, const Eigen::VectorXi& perm_indices) {
		CHECK(L.isCompressed());
		CHECK(L.rows() == L.cols());
		const int n = (int)L.cols();
		const int* col_ptr = L.outerIndexPtr();
		const int* row_idx = L.innerIndexPtr();
		bool sorted = true;
		for (int j = 0; j < n && sorted; ++j) {
			for (int p = col_ptr[j] + 1; p < col_ptr[j + 1]; ++p) {
				if (row_idx[p] <= row_idx[p - 1]) {
					sorted = false;
					break;
				}
			}
		}
		if (sorted) {
			L_ = &L;
		}
		else {
			// a transposition sorts the indices
			sp_mat_t L_t = L.transpose();
			L_sorted_ = L_t.transpose();
			L_ = &L_sorted_;
		}
		for (int j = 0; j < n; ++j) {
			CHECK(L_->outerIndexPtr()[j] < L_->outerIndexPtr()[j + 1]);
			CHECK(L_->innerIndexPtr()[L_->outerIndexPtr()[j]] == j);
		}
		perm_.resize(n);
		if (perm_indices.size() == 0) {
			for (int i = 0; i < n; ++i) {
				perm_[i] = i;
			}
		}
		else {
			CHECK((int)perm_indices.size() == n);
			for (int i = 0; i < n; ++i) {
				perm_[i] = perm_indices[i];
			}
		}
	}

	void SparseInverseEntries::CalcSelectedInverse() const {
		if (!selected_inverse_calculated_) {
			SelectedInversionCholesky(*L_, Z_lower_);
			selected_inverse_calculated_ = true;
		}
	}

	void SparseInverseEntries::CalcComponents() const {
		if (components_calculated_) {
			return;
		}
		// The parent of a node in the elimination forest is the first row below the diagonal of its column (the row indices are
		//	sorted), and it has a larger index than the node. A pass in reverse order therefore labels every node with the root of its tree
		const int n = (int)perm_.size();
		const int* col_ptr = L_->outerIndexPtr();
		const int* row_idx = L_->innerIndexPtr();
		component_.resize(n);
		for (int j = n - 1; j >= 0; --j) {
			component_[j] = (col_ptr[j] + 1 < col_ptr[j + 1]) ? component_[row_idx[col_ptr[j] + 1]] : j;
		}
		components_calculated_ = true;
	}

	void SparseInverseEntries::CalcPathCosts() const {
		if (path_costs_calculated_) {
			return;
		}
		// The parent of a node in the elimination tree is the first row below the diagonal of its column, and it has a larger index
		const int n = (int)perm_.size();
		const int* col_ptr = L_->outerIndexPtr();
		const int* row_idx = L_->innerIndexPtr();
		path_cost_.resize(n);
		for (int j = n - 1; j >= 0; --j) {
			const double num_non_zeros_col = (double)(col_ptr[j + 1] - col_ptr[j]);
			path_cost_[j] = (col_ptr[j] + 1 < col_ptr[j + 1]) ? num_non_zeros_col + path_cost_[row_idx[col_ptr[j] + 1]] : num_non_zeros_col;
		}
		path_costs_calculated_ = true;
	}

	int SparseInverseEntries::Position(int i, int j) const {
		int r = perm_[i];
		int c = perm_[j];
		if (r < c) {
			std::swap(r, c);
		}
		const int* row_idx = Z_lower_.innerIndexPtr();
		const int* first = row_idx + Z_lower_.outerIndexPtr()[c];
		const int* last = row_idx + Z_lower_.outerIndexPtr()[c + 1];
		const int* pos = std::lower_bound(first, last, r);
		return (pos != last && *pos == r) ? (int)(pos - row_idx) : -1;
	}

	vec_t SparseInverseEntries::Diagonal() const {
		if (selected_inverse_calculated_) {
			return DiagonalBySelectedInversion();
		}
		double cost_solves, cost_selected_inversion;
		CostsDiagonalCholFactorInverse(*L_, cost_solves, cost_selected_inversion);
		if (cost_solves < cost_selected_inversion) {
			return DiagonalBySolves();
		}
		else {
			return DiagonalBySelectedInversion();
		}
	}

	vec_t SparseInverseEntries::DiagonalBySolves() const {
		const int n = (int)perm_.size();
		// the diagonal of (L L^T)^(-1) = L^(-T) L^(-1) are the squared norms of the columns of L^(-1)
		sp_mat_t Id(n, n);
		Id.setIdentity();
		vec_t norms;
		CalcColSquaredNormsCholFactorInvR(*L_, Id, norms);
		vec_t diag(n);
		for (int i = 0; i < n; ++i) {
			diag[i] = norms[perm_[i]];
		}
		return diag;
	}

	vec_t SparseInverseEntries::DiagonalBySelectedInversion() const {
		CalcSelectedInverse();
		const int n = (int)perm_.size();
		vec_t diag(n);
		const int* col_ptr = Z_lower_.outerIndexPtr();
		const double* z_val = Z_lower_.valuePtr();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; ++i) {
			diag[i] = z_val[col_ptr[perm_[i]]];
		}
		return diag;
	}

	void SparseInverseEntries::EntriesBySolves(const std::vector<int>& rows, const std::vector<int>& cols, std::vector<double>& values) const {
		CHECK(rows.size() == cols.size());
		values.resize(rows.size());
		if (rows.empty()) {
			return;
		}
		// A is block-diagonal up to a permutation with one block per connected component of its graph, and so is its inverse:
		//	an entry whose row and column are in different components is zero
		CalcComponents();
		CalcPathCosts();
		const int n = (int)perm_.size();
		// The other entries as pairs (a, b), a <= b, of indices of P A P^T, since entry (i, j) of A^(-1) is entry (P i, P j) of
		//	(L L^T)^(-1), which is symmetric. Every pair is calculated once, also if it is needed for several entries
		std::vector<std::pair<int, int>> pairs;
		std::vector<int> entry_of_pair;
		pairs.reserve(rows.size());
		entry_of_pair.reserve(rows.size());
		for (int k = 0; k < (int)rows.size(); ++k) {
			int a = perm_[rows[k]];
			int b = perm_[cols[k]];
			if (component_[a] != component_[b]) {
				values[k] = 0.;
				continue;
			}
			if (a > b) {
				std::swap(a, b);
			}
			pairs.push_back(std::make_pair(a, b));
			entry_of_pair.push_back(k);
		}
		if (pairs.empty()) {
			return;
		}
		std::vector<int> order(pairs.size());
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(), [&pairs](int s, int t) { return pairs[s] < pairs[t]; });
		std::vector<std::pair<int, int>> unique_pairs;
		std::vector<int> unique_pair_of_pair(pairs.size());
		for (int t = 0; t < (int)order.size(); ++t) {
			if (t == 0 || pairs[order[t]] != pairs[order[t - 1]]) {
				unique_pairs.push_back(pairs[order[t]]);
			}
			unique_pair_of_pair[order[t]] = (int)unique_pairs.size() - 1;
		}
		const int num_unique_pairs = (int)unique_pairs.size();
		// The column of a pair is the one of its two indices that occurs in more pairs, so that few columns are needed
		std::vector<int> num_pairs_of_index(n, 0);
		for (const auto& pair : unique_pairs) {
			num_pairs_of_index[pair.first]++;
			if (pair.second != pair.first) {
				num_pairs_of_index[pair.second]++;
			}
		}
		std::vector<int> col_of_pair(num_unique_pairs), row_of_pair(num_unique_pairs);
		for (int u = 0; u < num_unique_pairs; ++u) {
			const int a = unique_pairs[u].first;
			const int b = unique_pairs[u].second;
			if (num_pairs_of_index[a] >= num_pairs_of_index[b]) {
				col_of_pair[u] = a;
				row_of_pair[u] = b;
			}
			else {
				col_of_pair[u] = b;
				row_of_pair[u] = a;
			}
		}
		std::vector<int> order_unique(num_unique_pairs);
		std::iota(order_unique.begin(), order_unique.end(), 0);
		std::sort(order_unique.begin(), order_unique.end(), [&col_of_pair](int s, int t) { return col_of_pair[s] < col_of_pair[t]; });
		std::vector<int> group_start;
		for (int t = 0; t < num_unique_pairs; ++t) {
			if (t == 0 || col_of_pair[order_unique[t]] != col_of_pair[order_unique[t - 1]]) {
				group_start.push_back(t);
			}
		}
		group_start.push_back(num_unique_pairs);
		const double* val = L_->valuePtr();
		const int* row_idx = L_->innerIndexPtr();
		const int* col_ptr = L_->outerIndexPtr();
		// Columns whose entries are calculated as inner products of columns of L^(-1), and columns that are calculated with two
		//	triangular solves over all of L
		const int num_groups = (int)group_start.size() - 1;
		const double cost_column_full_solves = 2. * (double)n + (double)L_->nonZeros();
		std::vector<int> groups_inner_products, groups_full_solves;
		for (int g = 0; g < num_groups; ++g) {
			double cost_inner_products = path_cost_[col_of_pair[order_unique[group_start[g]]]];
			for (int t = group_start[g]; t < group_start[g + 1]; ++t) {
				cost_inner_products += path_cost_[row_of_pair[order_unique[t]]];
			}
			if (cost_inner_products < cost_column_full_solves) {
				groups_inner_products.push_back(g);
			}
			else {
				groups_full_solves.push_back(g);
			}
		}
		// The columns with full solves are calculated in blocks, so that L is read once per block and not once per column (the
		//	solves are limited by the memory bandwidth if L is large). The blocks of all threads have at most 2^26 entries (512 MB)
		int num_threads;
#ifdef _OPENMP
		num_threads = omp_get_max_threads();
#else
		num_threads = 1;
#endif
		const int num_full_solves = (int)groups_full_solves.size();
		const int max_cols_block_memory = (int)std::max(1., std::floor((double)(1 << 26) / ((double)n * (double)num_threads)));
		const int cols_per_block = std::max(1, std::min(std::min(16, max_cols_block_memory), (num_full_solves + num_threads - 1) / num_threads));
		const int num_blocks = (num_full_solves + cols_per_block - 1) / cols_per_block;
		std::vector<double> values_unique_pairs(num_unique_pairs);
#pragma omp parallel
		{
			vec_t x = vec_t::Zero(n);
			vec_t y = vec_t::Zero(n);
			std::vector<int> path_x, path_y;
			// z = L^(-1) e_k: its non-zeros are on the path from k to the root of the elimination tree, since the rows below the
			//	diagonal of a column of L are ancestors of the column. The next node of the path is the parent, the first of these rows
			auto calc_column_on_path = [&](int k, vec_t& z, std::vector<int>& path) {
				path.clear();
				z[k] = 1.;
				for (int m = k; m >= 0; ) {
					path.push_back(m);
					const double z_m = z[m] / val[col_ptr[m]];
					z[m] = z_m;
					for (int p = col_ptr[m] + 1; p < col_ptr[m + 1]; ++p) {
						z[row_idx[p]] -= val[p] * z_m;
					}
					m = (col_ptr[m] + 1 < col_ptr[m + 1]) ? row_idx[col_ptr[m] + 1] : -1;
				}
			};
#pragma omp for schedule(dynamic, 1) nowait
			for (int i = 0; i < (int)groups_inner_products.size(); ++i) {
				const int g = groups_inner_products[i];
				// entry (r, col) of (L L^T)^(-1) = (L^(-1) e_r)^T (L^(-1) e_col)
				calc_column_on_path(col_of_pair[order_unique[group_start[g]]], x, path_x);
				for (int t = group_start[g]; t < group_start[g + 1]; ++t) {
					const int u = order_unique[t];
					calc_column_on_path(row_of_pair[u], y, path_y);
					double value = 0.;
					for (int m : path_y) {
						value += y[m] * x[m];
						y[m] = 0.;
					}
					values_unique_pairs[u] = value;
				}
				for (int m : path_x) {
					x[m] = 0.;
				}
			}
			// X = (L L^T)^(-1) [e_col_1, ..., e_col_nb], stored row by row: row j of X are the entries j of the columns of the block
			std::vector<double> X;
#pragma omp for schedule(dynamic, 1)
			for (int b = 0; b < num_blocks; ++b) {
				const int i_begin = b * cols_per_block;
				const int nb = std::min(cols_per_block, num_full_solves - i_begin);
				X.assign((size_t)n * (size_t)nb, 0.);
				int first_col = n;
				for (int k = 0; k < nb; ++k) {
					const int col = col_of_pair[order_unique[group_start[groups_full_solves[i_begin + k]]]];
					X[(size_t)col * nb + k] = 1.;
					first_col = std::min(first_col, col);
				}
				// X = L^(-1) X, the rows before the first column of the block are zero, and so are rows that are not on a path
				//	from a column of the block to the root
				for (int j = first_col; j < n; ++j) {
					double* x_j = X.data() + (size_t)j * nb;
					bool is_zero = true;
					for (int k = 0; k < nb; ++k) {
						if (x_j[k] != 0.) {
							is_zero = false;
							break;
						}
					}
					if (is_zero) {
						continue;
					}
					const double l_jj_inv = 1. / val[col_ptr[j]];
					for (int k = 0; k < nb; ++k) {
						x_j[k] *= l_jj_inv;
					}
					for (int p = col_ptr[j] + 1; p < col_ptr[j + 1]; ++p) {
						double* x_r = X.data() + (size_t)row_idx[p] * nb;
						const double l_rj = val[p];
						for (int k = 0; k < nb; ++k) {
							x_r[k] -= l_rj * x_j[k];
						}
					}
				}
				// X = L^(-T) X
				for (int j = n - 1; j >= 0; --j) {
					double* x_j = X.data() + (size_t)j * nb;
					for (int p = col_ptr[j] + 1; p < col_ptr[j + 1]; ++p) {
						const double* x_r = X.data() + (size_t)row_idx[p] * nb;
						const double l_rj = val[p];
						for (int k = 0; k < nb; ++k) {
							x_j[k] -= l_rj * x_r[k];
						}
					}
					const double l_jj_inv = 1. / val[col_ptr[j]];
					for (int k = 0; k < nb; ++k) {
						x_j[k] *= l_jj_inv;
					}
				}
				for (int k = 0; k < nb; ++k) {
					const int g = groups_full_solves[i_begin + k];
					for (int t = group_start[g]; t < group_start[g + 1]; ++t) {
						const int u = order_unique[t];
						values_unique_pairs[u] = X[(size_t)row_of_pair[u] * nb + k];
					}
				}
			}
		}
		for (int s = 0; s < (int)pairs.size(); ++s) {
			values[entry_of_pair[s]] = values_unique_pairs[unique_pair_of_pair[s]];
		}
	}//end EntriesBySolves

	void SparseInverseEntries::OnSparsityPattern(sp_mat_t& M) const {
		CHECK(M.rows() == (int)perm_.size());
		CHECK(M.cols() == (int)perm_.size());
		CalcSelectedInverse();
		M.makeCompressed();
		const double* z_val = Z_lower_.valuePtr();
		double* m_val = M.valuePtr();
		const int* m_row_idx = M.innerIndexPtr();
		const int* m_col_ptr = M.outerIndexPtr();
		std::vector<int> missing_pos;
#pragma omp parallel
		{
			std::vector<int> missing_pos_private;
#pragma omp for schedule(static)
			for (int j = 0; j < (int)M.cols(); ++j) {
				for (int p = m_col_ptr[j]; p < m_col_ptr[j + 1]; ++p) {
					const int pos = Position(m_row_idx[p], j);
					if (pos >= 0) {
						m_val[p] = z_val[pos];
					}
					else {
						missing_pos_private.push_back(p);
					}
				}
			}
#pragma omp critical
			{
				missing_pos.insert(missing_pos.end(), missing_pos_private.begin(), missing_pos_private.end());
			}
		}
		if (!missing_pos.empty()) {
			std::vector<int> rows(missing_pos.size()), cols(missing_pos.size());
			for (int k = 0; k < (int)missing_pos.size(); ++k) {
				rows[k] = m_row_idx[missing_pos[k]];
				cols[k] = (int)(std::upper_bound(m_col_ptr, m_col_ptr + M.cols() + 1, missing_pos[k]) - m_col_ptr) - 1;
			}
			std::vector<double> values;
			EntriesBySolves(rows, cols, values);
			for (int k = 0; k < (int)missing_pos.size(); ++k) {
				m_val[missing_pos[k]] = values[k];
			}
		}
	}//end OnSparsityPattern

	vec_t SparseInverseEntries::DiagZAinvZt(const sp_mat_t& Zt) const {
		CHECK(Zt.rows() == (int)perm_.size());
		CalcSelectedInverse();
		const int num_rows_Z = (int)Zt.cols();
		vec_t diag(num_rows_Z);
		const double* z_val = Z_lower_.valuePtr();
		// Entries that are not on the sparsity pattern of L: row of Z, (i,j) of A^(-1), and the factor of this entry
		std::vector<int> missing_row_Z, missing_i, missing_j;
		std::vector<double> missing_factor;
#pragma omp parallel
		{
			std::vector<int> row_Z_private, i_private, j_private;
			std::vector<double> factor_private;
			std::vector<int> ind;
			std::vector<double> val;
#pragma omp for schedule(static)
			for (int r = 0; r < num_rows_Z; ++r) {
				ind.clear();
				val.clear();
				for (sp_mat_t::InnerIterator it(Zt, r); it; ++it) {
					ind.push_back((int)it.row());
					val.push_back(it.value());
				}
				double d = 0.;
				for (int k = 0; k < (int)ind.size(); ++k) {
					for (int l = 0; l <= k; ++l) {
						const double factor = (k == l) ? val[k] * val[k] : 2. * val[k] * val[l];
						const int pos = Position(ind[k], ind[l]);
						if (pos >= 0) {
							d += factor * z_val[pos];
						}
						else {
							row_Z_private.push_back(r);
							i_private.push_back(ind[k]);
							j_private.push_back(ind[l]);
							factor_private.push_back(factor);
						}
					}
				}
				diag[r] = d;
			}
#pragma omp critical
			{
				missing_row_Z.insert(missing_row_Z.end(), row_Z_private.begin(), row_Z_private.end());
				missing_i.insert(missing_i.end(), i_private.begin(), i_private.end());
				missing_j.insert(missing_j.end(), j_private.begin(), j_private.end());
				missing_factor.insert(missing_factor.end(), factor_private.begin(), factor_private.end());
			}
		}
		if (!missing_row_Z.empty()) {
			std::vector<double> values;
			EntriesBySolves(missing_i, missing_j, values);
			for (int k = 0; k < (int)missing_row_Z.size(); ++k) {
				diag[missing_row_Z[k]] += missing_factor[k] * values[k];
			}
		}
		return diag;
	}//end DiagZAinvZt

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
		const double* const v,
		double* ZtV,
		bool initialize_zero) {
		if (initialize_zero) {
#pragma omp parallel for schedule(static)
			for (data_size_t i = 0; i < num_re; ++i) {
				ZtV[i] = 0.;
			}
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

	double InnerProductTwoColumns(const sp_mat_t& M, int i, int j) {
		double scalar_prod = 0.0;
		sp_mat_t::InnerIterator it_i(M, i);
		sp_mat_t::InnerIterator it_j(M, j);
		while (it_i && it_j) {
			if (it_i.row() == it_j.row()) {
				scalar_prod += it_i.value() * it_j.value();
				++it_i;
				++it_j;
			}
			else if (it_i.row() < it_j.row()) {
				++it_i;
			}
			else {
				++it_j;
			}
		}
		return(scalar_prod);
	}

}  // namespace GPBoost
