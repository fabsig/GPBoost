/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2026 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_PARALLEL_SUM_H_
#define GPB_PARALLEL_SUM_H_

#include <LightGBM/utils/openmp_wrapper.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace GPBoost {

	/*!
	* \brief Sums of the iterations begin, ..., end - 1 of a loop for ParallelSumIf(). The partial sums are parameters that
	*		are passed by value, i.e., local variables that the compiler can keep in registers like the private copies of an
	*		OpenMP reduction, which is faster than accumulating into the elements of a tuple (by up to 8% with MSVC)
	*/
	template <typename T_index, typename T_body, typename... T_sums>
	std::tuple<T_sums...> SumOfBlock(T_index begin,
		T_index end,
		T_body& body,
		T_sums... partial_sums) {
		for (T_index i = begin; i < end; ++i) {
			body(i, partial_sums...);
		}
		return std::make_tuple(partial_sums...);
	}//end SumOfBlock

	/*!
	* \brief Parallel version of 'for (i = 0; i < n; ++i) body(i, sums...);' for sums that gives the same result in every run
	*		with the same number of threads. An OpenMP 'reduction(+:...)' adds the partial sums of the threads in the order in
	*		which the threads finish (e.g., with GCC), which changes the last digits of floating-point sums from run to run.
	*		Here, the indices are split into one contiguous block per thread, as with 'schedule(static)', and the partial sums of
	*		the blocks are added in the order of the blocks.
	* \param in_parallel If false, the loop runs on one thread (like the 'if' clause of an OpenMP directive)
	* \param n Number of iterations
	* \param body Function (i, sums...) that adds the terms of iteration i to its arguments, which are the partial sums of the
	*		block of i, like the private copies of the variables of a reduction clause
	* \param[out] sums Variables to which the sums are added
	*/
	template <typename T_index, typename T_body, typename... T_sums>
	void ParallelSumIf(bool in_parallel,
		T_index n,
		T_body body,
		T_sums&... sums) {
		const int num_blocks = in_parallel ? omp_get_max_threads() : 1;
		std::vector<std::tuple<T_sums...>> block_sums(num_blocks);
#pragma omp parallel for schedule(static) if (num_blocks > 1)
		for (int b = 0; b < num_blocks; ++b) {
			const T_index begin = (T_index)((int64_t)n * b / num_blocks);
			const T_index end = (T_index)((int64_t)n * (b + 1) / num_blocks);
			block_sums[b] = SumOfBlock(begin, end, body, T_sums{}...);
		}
		for (const std::tuple<T_sums...>& block_sum : block_sums) {
			std::apply([&](const T_sums&... partial_sums) { ((sums += partial_sums), ...); }, block_sum);
		}
	}//end ParallelSumIf

	/*! \brief ParallelSumIf(true, n, body, sums...) */
	template <typename T_index, typename T_body, typename... T_sums>
	void ParallelSum(T_index n,
		T_body body,
		T_sums&... sums) {
		ParallelSumIf(true, n, body, sums...);
	}//end ParallelSum

	/*!
	* \brief Adds the partial sums of the threads to a vector or a dense matrix in the order of the threads, such that the
	*		result does not depend on the order in which the threads finish (unlike additions in a '#pragma omp critical'
	*		section). The additions are parallelized over contiguous blocks of entries
	* \param partial_sums Partial sums of the threads, which have the size of 'sum' or are empty (threads that have not been used)
	* \param[out] sum Vector or dense matrix (or a map of one) to which the partial sums are added
	*/
	template <typename T_mat, typename T_sum>
	void AddPartialSumsOfThreads(const std::vector<T_mat>& partial_sums,
		T_sum& sum) {
		double* sum_data = sum.data();
		const int64_t size = (int64_t)sum.size();
		const int num_blocks = omp_get_max_threads();
#pragma omp parallel for schedule(static) if (num_blocks > 1)
		for (int b = 0; b < num_blocks; ++b) {
			const int64_t begin = size * b / num_blocks;
			const int64_t end = size * (b + 1) / num_blocks;
			for (const T_mat& partial_sum : partial_sums) {
				if (partial_sum.size() > 0) {
					const double* partial_sum_data = partial_sum.data();
					for (int64_t k = begin; k < end; ++k) {
						sum_data[k] += partial_sum_data[k];
					}
				}
			}
		}
	}//end AddPartialSumsOfThreads

}  // namespace GPBoost

#endif   // GPB_PARALLEL_SUM_H_
