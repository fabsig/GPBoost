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

namespace GPBoost {

	/*! \brief Tolerance level when comparing two numbers for equality */
	const double EPSILON_NUMBERS = 1e-10;

	/*! \brief Tolerance level when comparing two vectors for equality */
	const double EPSILON_VECTORS = 1e-10;

	/*! \brief Comparing two numbers for equality */
	template <typename T>//T can be double or float
	inline bool TwoNumbersAreEqual(const T a, const T b) {
		return fabs(a - b) < a * EPSILON_NUMBERS;
	}

	/*! \brief Get number of non-zero entries in a matrix */
	template <class T_mat1, typename std::enable_if< std::is_same<sp_mat_t, T_mat1>::value>::type* = nullptr  >
	int GetNumberNonZeros(const T_mat1 M) {
		return((int)M.nonZeros());
	};
	template <class T_mat1, typename std::enable_if< std::is_same<den_mat_t, T_mat1>::value>::type* = nullptr  >
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
	void SortVectorsDecreasing(double* a,
		int* b,
		int n);

}  // namespace GPBoost

#endif   // GPB_UTILS_H_
