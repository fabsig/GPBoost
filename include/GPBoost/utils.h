/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_UTILS_H_
#define GPB_UTILS_H_

#include <cmath>
#include <GPBoost/type_defs.h>
#include <algorithm>    // std::max
#include <numeric>      // std::iota

namespace GPBoost {

	/*! \brief Tolerance level when comparing two numbers for equality */
	const double EPSILON_NUMBERS = 1e-10;

	/*! \brief Tolerance level when comparing two vectors for equality */
	const double EPSILON_VECTORS = 1e-10;

	/*! \brief Small number that is added in some cases to covariance matrices to make inversion numerically stable */
	const double EPSILON_ADD_COVARIANCE_STABLE = 1e-10;

	/*! \brief Comparing two numbers for equality, source: http://realtimecollisiondetection.net/blog/?p=89 */
	template <typename T>//T can be double or float
	inline bool TwoNumbersAreEqual(const T a, const T b) {
		return std::abs(a - b) < EPSILON_NUMBERS * std::max<T>({ 1.0, std::abs(a), std::abs(b) });
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

	/*!
	* \brief Finds the sorting index of vector v and saves it in idx
	* \param v Vector with values
	* \param idx Vector where sorting index is written to
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
	}//end SampleIntNoReplace

}  // namespace GPBoost

#endif   // GPB_UTILS_H_
