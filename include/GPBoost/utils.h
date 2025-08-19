/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 - 2025 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_UTILS_H_
#define GPB_UTILS_H_

#include <cmath>
#include <GPBoost/type_defs.h>
#include <algorithm>    // std::max, std::sort
#include <numeric>      // std::iota
#include <unordered_set>
#include <LightGBM/utils/log.h>

using LightGBM::Log;

namespace GPBoost {

	/*! \brief Tolerance level when comparing two numbers for equality */
	const double EPSILON_NUMBERS = 1e-10;

	/*! \brief Tolerance level when comparing two vectors for equality */
	const double EPSILON_VECTORS = 1e-10;

	const double TINY_NUMBER = (std::numeric_limits<double>::has_denorm == std::denorm_present)
		? std::numeric_limits<double>::denorm_min() : std::numeric_limits<double>::min();

	/*! \brief Small numbers by which the diagonals of some matrices are multiplied to make inversion numerically stable */
	const double JITTER_MUL = 1. + 1e-10;

	/*! \brief Small number by which the diagonals of covariance matrices are multiplied with when calculating Vecchia approximations without a nugget effect to make inversions numerically stable */
	const double JITTER_MULT_VECCHIA= 1. + 1e-10;

	/*! \brief Small number by which the diagonal of inducing points matrix in the FITC & full scales approximations is multiplied with (increased) to make inversions numerically stable */
	const double JITTER_MULT_IP_FITC_FSA = 1. + 1e-6;

	/*! \brief Termination criterion for low-rank pivoted Cholesky decomposition */
	const double PIV_CHOL_STOP_TOL = 1e-6;

	/*! \brief Threshold for considering a rhs as zero in conjugate gradient algorithms */
	const double ZERO_RHS_CG_THRESHOLD = 1e-100;

	/*! \brief Threshold for doing reorthogonalization in the Lanczos algorithm */
	const double LANCZOS_REORTHOGONALIZATION_THRESHOLD = 1e-5;

	/*! \brief Comparing two numbers for equality, source: http://realtimecollisiondetection.net/blog/?p=89 */
	template <typename T>//T can be double or float
	inline bool TwoNumbersAreEqual(const T a, const T b) {
		return std::abs(a - b) < EPSILON_NUMBERS * std::max<T>({ 1.0, std::abs(a), std::abs(b) });
	}

	/*! \brief Checking whether a number is zero */
	template <typename T>//T can be double or float
	inline bool IsZero(const T a) {
		return std::abs(a) < EPSILON_NUMBERS;
	}

	/*! \brief Checking whether a vector contains a zero */
	template <typename T>//T can be double or float
	inline bool HasZero(const T* v, data_size_t num_data) {
		int has_zero = 0;
#pragma omp parallel for reduction(|:has_zero) schedule(static)
		for (data_size_t i = 0; i < num_data; ++i) {
			if (IsZero<T>(v[i])) has_zero = 1;
		}
		return has_zero != 0;
	}//end HasZero

	/*! \brief Checking whether a vector contains negative values */
	template <typename T>//T can be double or float
	inline bool HasNegativeValues(const T* v, data_size_t num_data) {
		int has_negative = 0;
#pragma omp parallel for reduction(|:has_negative) schedule(static)
		for (data_size_t i = 0; i < num_data; ++i) {
			if (v[i] < 0.0) has_negative = 1;
		}
		return has_negative != 0;
	}//end HasNegativeValues


	/*! \brief Checking whether a number 'a' is smaller than another number 'b' */
	template <typename T>//T can be double or float
	inline bool NumberIsSmallerThan(const T a, const T b) {
		return (b - a)  > EPSILON_NUMBERS * std::max<T>({ 1.0, std::abs(b) });
	}

	/*! \brief Get number of non-zero entries in a matrix */
	template <class T_mat1, typename std::enable_if <std::is_same<sp_mat_t, T_mat1>::value ||
		std::is_same<sp_mat_rm_t, T_mat1>::value>::type* = nullptr >
	int GetNumberNonZeros(const T_mat1 M) {
		return((int)M.nonZeros());
	};
	template <class T_mat1, typename std::enable_if <std::is_same<den_mat_t, T_mat1>::value>::type* = nullptr >
	int GetNumberNonZeros(const T_mat1 M) {
		return((int)M.cols() * M.rows());
	};

	/*! \brief Calculate logarithm */
	inline double SafeLog(const double x) {
		if (x > 0) {
			return std::log(x);
		}
		else {
			return -INFINITY;
		}
	};

	/*! \brief Determines the number of unique values of a vector up to a certain number (max_unique_values) */
	inline int NumberUniqueValues(const vec_t vec,
		int max_unique_values) {
		std::unordered_set<double> unique_values;
		bool found_more_uniques_than_max = false;
#pragma omp parallel
		{
			std::unordered_set<double> local_set;
#pragma omp for
			for (data_size_t i = 0; i < (data_size_t)vec.size(); ++i) {
				if (found_more_uniques_than_max) {
					continue;
				}
				local_set.insert(vec[i]);
				if ((int)local_set.size() > max_unique_values) {
#pragma omp critical
					{
						found_more_uniques_than_max = true;
					}
				}
			}
#pragma omp critical
			{
				unique_values.insert(local_set.begin(), local_set.end());
			}
		}
		return (int)unique_values.size();
	};//end NumberUniqueValues

	/*!
	* \brief Finds the median of the vector vec
	* \param[out] vec Vector with values (will be partially sorted)
	* \return Median
	*/
	template <typename T>//T can be std::vector<double> or vec_t
	inline double CalculateMedianPartiallySortInput(T& vec) {
		CHECK(vec.size() > 0);
		int num_el = (int)vec.size();
		double median;
		int pos_med = (int)(num_el / 2);
		std::nth_element(vec.begin(), vec.begin() + pos_med, vec.end());
		median = vec[pos_med];
		if (num_el % 2 == 0) {
			std::nth_element(vec.begin(), vec.begin() + pos_med - 1, vec.end());
			median += vec[pos_med - 1];
			median /= 2.;
		}
		return(median);
	};

	/*!
	* \brief Finds the mean of the vector vec
	* \param[out] vec Vector with values 
	* \return Mean
	*/
	template <typename T>//T can be std::vector<double> or vec_t
	inline double CalculateMean(const T& vec) {
		CHECK(vec.size() > 0);
		int num_el = (int)vec.size();
		double mean = 0.;
#pragma omp parallel for schedule(static) reduction(+:mean)
		for (int i = 0; i < num_el; ++i) {
			mean += vec[i];
		}
		mean /= num_el;
		return(mean);
	};

	/*!
	* \brief Finds the sorting index of vector v and saves it in idx
	* \param v Vector with values
	* \param idx Vector where sorting index is written to. idx[k] corresponds to the index of the k-smallest element of v, i.e., v[idx[0]] <= v[idx[1]] <= v[idx[2]] <= ... 
	*/
	template <typename T>
	void SortIndeces(const std::vector<T>& v,
		std::vector<int>& idx) {
		// initialize original index locations
		idx.resize(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		// sort indexes based on comparing values in v
		std::sort(idx.begin(), idx.end(),
			[&v](int i1, int i2) {return v[i1] < v[i2]; });
	};

	/*!
	* \brief Sorts vectors a and b of length n based on decreasing values of a (source: suplementary code of Finley et al., 2019, JASA)
	* \param a Vector which determines sorting order and which is also ordered
	* \param b Vector which is ordered based on order in a
	* \param n Length of vectors
	*/
	template <typename T>
	void SortVectorsDecreasing(T* a, int* b, int n) {
		int j, k, l;
		double v;
		for (j = 1; j <= n - 1; j++) {
			k = j;
			while (k > 0 && a[k] < a[k - 1]) {
				v = a[k]; l = b[k];
				a[k] = a[k - 1]; b[k] = b[k - 1];
				a[k - 1] = v; b[k - 1] = l;
				k--;
			}
		}
	}

	/*!
	* \brief Sample k integers from 0:(N-1) without replacement while excluding some indices
	*		Source: see https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html and https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement
	* \param N Total number of integers from which to sample
	* \param k Size of integer set which is drawn
	* \param gen RNG
	* \param[out] indices Drawn integers
	* \param exclude Excluded integers
	*/
	inline void SampleIntNoReplaceExcludeSomeIndices(int N,
		int k,
		RNG_t& gen,
		std::vector<int>& indices,
		const std::vector<int>& exclude) {
		for (int r = N - k; r < N; ++r) {
			int v = std::uniform_int_distribution<>(0, r)(gen);
			int new_draw;
			if (std::find(indices.begin(), indices.end(), v) == indices.end()) {
				new_draw = v;
			}
			else {
				new_draw = r;
			}
			if (std::find(exclude.begin(), exclude.end(), new_draw) == exclude.end()) {
				indices.push_back(new_draw);
			}
			else {
				r--;
			}
		}
	}//end SampleIntNoReplaceExcludeSomeIndices

	/*!
	* \brief Sample k integers from 0:(N-1) without replacement
	*		Source: see https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html and https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement
	* \param N Total number of integers from which to sample
	* \param k Size of integer set which is drawn
	* \param gen RNG
	* \param[out] indices Drawn integers
	*/
	inline void SampleIntNoReplace(int N,
		int k,
		RNG_t& gen,
		std::vector<int>& indices) {
		for (int r = N - k; r < N; ++r) {
			int v = std::uniform_int_distribution<>(0, r)(gen);
			if (std::find(indices.begin(), indices.end(), v) == indices.end()) {
				indices.push_back(v);
			}
			else {
				indices.push_back(r);
			}
		}
		std::sort(indices.begin(), indices.end());
	}//end SampleIntNoReplace

	/*!
	* \brief Sample k integers from 0:(N-1) without replacement and sort them
	*		Source: see https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html and https://stackoverflow.com/questions/28287138/c-randomly-sample-k-numbers-from-range-0n-1-n-k-without-replacement
	* \param N Total number of integers from which to sample
	* \param k Size of integer set which is drawn
	* \param gen RNG
	* \param[out] indices Drawn integers
	*/
	inline void SampleIntNoReplaceSort(int N,
		int k,
		RNG_t& gen,
		std::vector<int>& indices) {
		for (int r = N - k; r < N; ++r) {
			int v = std::uniform_int_distribution<>(0, r)(gen);
			if (std::find(indices.begin(), indices.end(), v) == indices.end()) {
				indices.push_back(v);
			}
			else {
				indices.push_back(r);
			}
		}
		std::sort(indices.begin(), indices.end());
	}//end SampleIntNoReplaceSort 

	/*! \brief Convert a dense matrix to a matrix of type T_mat (dense or sparse) */
	template <class T_mat1, typename std::enable_if <std::is_same<sp_mat_t, T_mat1>::value ||
		std::is_same<sp_mat_rm_t, T_mat1>::value>::type* = nullptr >
	inline void ConvertTo_T_mat_FromDense(const den_mat_t M, T_mat1& Mout) {
		Mout = M.sparseView();
	};
	template <class T_mat1, typename std::enable_if< std::is_same<den_mat_t, T_mat1>::value>::type* = nullptr  >
	inline void ConvertTo_T_mat_FromDense(const den_mat_t M, T_mat1& Mout) {
		Mout = M;
	};

}  // namespace GPBoost

#endif   // GPB_UTILS_H_
