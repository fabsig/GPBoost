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

#include <atomic>
#include <cmath>
#include <limits>
#include <GPBoost/type_defs.h>
#include <algorithm>    // std::max, std::sort
#include <numeric>      // std::iota
#include <unordered_set>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

using LightGBM::Log;

namespace GPBoost {

	/*!
	* \brief Number of physical performance cores of the CPU. On hybrid CPUs with cores of different speeds
	*		(e.g., the performance and efficiency cores of recent Intel CPUs or Apple silicon), the fastest ones
	*		are counted, and simultaneous multithreading siblings ('hyperthreads') are counted only once.
	*		Since almost all parallel loops of GPBoost distribute their iterations evenly over all threads and
	*		then wait for the slowest one, threads running on slow cores can make an entire model slower. The
	*		next fastest cores are added if the fastest ones alone would leave only a single thread, and on
	*		Linux a CPU bandwidth limit of a control group is respected as well. Implemented in
	*		'cpu_topology.cpp'
	* \return Number of physical performance cores, or 0 if the topology of the CPU cannot be determined
	*/
	int NumPerformanceCores();

	/*!
	* \brief Determines the default number of parallel threads, see 'DefaultNumParallelThreads()'.
	*		Implemented in 'cpu_topology.cpp'
	* \return Default number of parallel threads
	*/
	int ComputeDefaultNumParallelThreads();

	/*!
	* \brief Number of threads that are used when no number of threads is explicitly requested. This is the number of
	*		physical performance cores, see 'NumPerformanceCores()', limited by the number of threads that OMP uses
	*		when this function is called for the first time (i.e., before any model has changed it), which is usually
	*		determined by the environment variable 'OMP_NUM_THREADS' or the number of cores. If 'OMP_NUM_THREADS' is
	*		set, it is used as is, and the same holds if the topology of the CPU cannot be determined
	* \return Default number of parallel threads
	*/
	inline int DefaultNumParallelThreads() {
		static const int default_num_parallel_threads = ComputeDefaultNumParallelThreads();
		return(default_num_parallel_threads);
	}

	/*!
	* \brief Sets the number of threads that OMP uses, after making sure that the default number of threads has
	*		been determined. Every place in the library that changes the number of threads of the process has to
	*		use this function: the default is determined only once, and it must not be derived from a number of
	*		threads that the library itself has set before, since a single thread set by, e.g., a boosting call
	*		would otherwise become the default of the entire process
	* \param num_threads Number of threads to use
	*/
	inline void SetNumParallelThreads(int num_threads) {
		DefaultNumParallelThreads();
		omp_set_num_threads(num_threads);
	}

	/*!
	* \brief Sets the number of threads used by OMP and Eigen and restores the previously used numbers of threads when
	*		the object goes out of scope again.
	*		Both 'omp_set_num_threads()' and 'Eigen::setNbThreads()' change the number of threads of the entire process
	*		and not only the one of a single model. Every operation of an 'REModel' that does calculations thus creates
	*		such an object, which makes the number of threads a property of the operation: models with different numbers
	*		of threads can be used alongside each other, and a model does not change the number of threads used by other
	*		models or by other libraries in the same process. If no number of threads is requested (i.e., if
	*		'num_threads' is not positive), 'DefaultNumParallelThreads()' is used, i.e., an operation of a model for
	*		which no number of threads has been specified always uses the default number of threads and not, e.g., a
	*		number of threads that has been set by another model or by the boosting part of the library
	*/
	class ParallelThreadsScope {
	public:
		/*!
		* \brief Constructor
		* \param num_threads Number of threads to use. If num_threads <= 0, 'DefaultNumParallelThreads()' is used
		*/
		explicit ParallelThreadsScope(int num_threads) {
			// The default is determined before the number of threads is changed below, and also when it is
			// not needed here: it is determined only once and must not be derived from a number of threads
			// that this or another scope has already set
			const int num_threads_default = DefaultNumParallelThreads();
			int num_threads_used = num_threads > 0 ? num_threads : num_threads_default;
			num_threads_previous_omp_ = omp_get_max_threads();
			num_threads_previous_eigen_ = Eigen::nbThreads();
			if (num_threads_used != num_threads_previous_omp_ || num_threads_used != num_threads_previous_eigen_) {
				omp_set_num_threads(num_threads_used);
				Eigen::setNbThreads(num_threads_used);
				num_threads_have_been_changed_ = true;
			}
		}

		/*! \brief Destructor. Restores the numbers of threads used before */
		~ParallelThreadsScope() {
			if (num_threads_have_been_changed_) {
				omp_set_num_threads(num_threads_previous_omp_);
				Eigen::setNbThreads(num_threads_previous_eigen_);
			}
		}

		ParallelThreadsScope(const ParallelThreadsScope&) = delete;
		ParallelThreadsScope& operator=(const ParallelThreadsScope&) = delete;

	private:
		/*! \brief Number of threads used by OMP before this object has been created */
		int num_threads_previous_omp_;
		/*! \brief Number of threads used by Eigen before this object has been created */
		int num_threads_previous_eigen_;
		/*! \brief True if the numbers of threads have been changed and thus need to be restored */
		bool num_threads_have_been_changed_ = false;
	};

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

	template <typename T>
	inline bool IsExactlyZero(T x) noexcept {
		return x == T(0);
	}

	/*! \brief Checking whether a vector contains a zero */
	template <typename T>//T can be double or float
	inline bool HasZero(const T* v_ptr, data_size_t num_data) {
		if (num_data == 0) return false;
		if (num_data < 50000) {// serial version for small data (often faster)
			return std::any_of(v_ptr, v_ptr + num_data, [](T v) { return IsZero<T>(v); });
		}
		bool has_zero = false;
#pragma omp parallel for schedule(static) reduction(||:has_zero)
		for (data_size_t i = 0; i < num_data; ++i) {
			has_zero = has_zero || (IsZero<T>(v_ptr[i]));
		}
		return has_zero;
	}//end HasZero

	/*! \brief Checking whether a vector contains an exact zero */
	template <typename T>//T can be double or float
	inline bool HasExactZero(const T* v_ptr, data_size_t num_data) {
		if (num_data == 0) return false;
		if (num_data < 50000) {// serial version for small data (often faster)
			return std::any_of(v_ptr, v_ptr + num_data, [](T v) { return v == 0.; });
		}
		bool has_zero = false;
#pragma omp parallel for schedule(static) reduction(||:has_zero)
		for (data_size_t i = 0; i < num_data; ++i) {
			has_zero = has_zero || (v_ptr[i] == 0.);
		}
		return has_zero;
	}//end HasZero

	/*! \brief Checking whether a vector contains only zeros */
	template <typename T>//T can be double or float
	inline bool HasOnlyExactZero(const T* v_ptr, data_size_t num_data) {
		if (num_data == 0) return false;
		if (num_data < 50000) {// serial version for small data (often faster)
			return std::all_of(v_ptr, v_ptr + num_data, [](T v) { return IsExactlyZero<T>(v); });
		}
		bool has_non_zero = false;
#pragma omp parallel for schedule(static) reduction(||:has_non_zero)
		for (data_size_t i = 0; i < num_data; ++i) {
			has_non_zero = has_non_zero || (!IsExactlyZero<T>(v_ptr[i]));
		}
		return !has_non_zero;
	}//end HasOnlyZero

	/*! \brief Checking whether a vector contains negative values */
	template <typename T>//T can be double or float
	inline bool HasNegativeValues(const T* v_ptr, data_size_t num_data) {
		if (num_data == 0) return false;
		if (num_data < 50000) {// serial version for small data (often faster)
			return std::any_of(v_ptr, v_ptr + num_data, [](T v) { return v < 0.; });
		}
		bool has_negative = false;
#pragma omp parallel for schedule(static) reduction(||:has_negative)
		for (data_size_t i = 0; i < num_data; ++i) {
			has_negative = has_negative || (v_ptr[i] < 0.);
		}
		return has_negative;
	}//end HasNegativeValues

	/*! \brief Checking whether a number 'a' is smaller than another number 'b' */
	template <typename T>//T can be double or float
	inline bool NumberIsSmallerThan(const T a, const T b) {
		return (b - a)  > EPSILON_NUMBERS * std::max<T>({ 1.0, std::abs(b) });
	}

	/*! \brief Get number of non-zero entries in a matrix */
	template <class T_mat1, typename std::enable_if <std::is_same<sp_mat_t, T_mat1>::value ||
		std::is_same<sp_mat_rm_t, T_mat1>::value>::type* = nullptr >
	int GetNumberNonZeros(const T_mat1& M) {
		return((int)M.nonZeros());
	};
	template <class T_mat1, typename std::enable_if <std::is_same<den_mat_t, T_mat1>::value>::type* = nullptr >
	int GetNumberNonZeros(const T_mat1& M) {
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

	/*! \brief Determines the number of unique values of a vector up to a certain number (max_unique_values).
	*		Note: once more than 'max_unique_values' unique values have been found, the remaining entries are skipped,
	*		so the returned count is only meaningful as "<= max_unique_values" vs. "more than max_unique_values" */
	inline int NumberUniqueValues(const vec_t& vec,
		int max_unique_values) {
		std::unordered_set<double> unique_values;
		//atomic: the early-exit flag is read by all threads while one of them writes it
		std::atomic<bool> found_more_uniques_than_max(false);
#pragma omp parallel
		{
			std::unordered_set<double> local_set;
#pragma omp for
			for (data_size_t i = 0; i < (data_size_t)vec.size(); ++i) {
				if (found_more_uniques_than_max.load(std::memory_order_relaxed)) {
					continue;
				}
				local_set.insert(vec[i]);
				if ((int)local_set.size() > max_unique_values) {
					found_more_uniques_than_max.store(true, std::memory_order_relaxed);
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
	* \brief Finds the weighted sample quantile of a vector of values: the smallest value for which the
	*		cumulative weight reaches quantile * (total weight). For all weights being equal, this coincides
	*		with the order statistic at position ceil(quantile * n) - 1.
	* \param values Vector with values
	* \param weights Weights (non-negative), one for every entry of 'values'
	* \param quantile Quantile with 0 < quantile < 1
	* \return Weighted quantile
	*/
	inline double CalculateWeightedQuantile(const std::vector<double>& values,
		const double* weights,
		double quantile) {
		CHECK(values.size() > 0);
		CHECK(weights != nullptr);
		int num_el = (int)values.size();
		std::vector<int> idx(num_el);
		std::iota(idx.begin(), idx.end(), 0);
		std::sort(idx.begin(), idx.end(), [&values](int a, int b) { return values[a] < values[b]; });
		double sum_w = 0.;
#pragma omp parallel for schedule(static) reduction(+:sum_w)
		for (int i = 0; i < num_el; ++i) {
			sum_w += weights[i];
		}
		double target = quantile * sum_w, cum_w = 0.;
		double quant = values[idx[num_el - 1]];// fallback in case of numerical inaccuracies
		for (int i = 0; i < num_el; ++i) {
			cum_w += weights[idx[i]];
			if (cum_w >= target) {
				quant = values[idx[i]];
				break;
			}
		}
		return(quant);
	};//end CalculateWeightedQuantile

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
	* \brief Sorts vectors a and b of length n based on increasing values of a, i.e., a[0] <= a[1] <= ... <= a[n-1]
	*		(insertion sort; source: suplementary code of Finley et al., 2019, JASA).
	*		Note: callers rely on the largest value ending up last (e.g. nearest-neighbour searches that keep the k
	*		smallest distances and compare new candidates against a[n-1])
	* \param a Vector which determines sorting order and which is also ordered
	* \param b Vector which is ordered based on order in a
	* \param n Length of vectors
	*/
	template <typename T>
	void SortVectorsIncreasing(T* a, int* b, int n) {
		int j, k, l;
		T v;
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
		//Maximal number of random attempts per position before falling back to a deterministic scan. Without such a
		//bound, the retry below spins forever whenever every admissible value of 0:r is already drawn or excluded
		const int max_random_attempts = 100;
		for (int r = N - k; r < N; ++r) {
			bool drawn = false;
			for (int attempt = 0; attempt < max_random_attempts && !drawn; ++attempt) {
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
					drawn = true;
				}
			}
			if (!drawn) {
				//Deterministic fallback: take the first admissible value. This terminates and still yields a valid
				//sample (distinct and non-excluded); it only gives up the uniformity of this particular draw
				for (int cand = 0; cand <= r && !drawn; ++cand) {
					if (std::find(indices.begin(), indices.end(), cand) == indices.end() &&
						std::find(exclude.begin(), exclude.end(), cand) == exclude.end()) {
						indices.push_back(cand);
						drawn = true;
					}
				}
			}
			if (!drawn) {
				Log::REFatal("SampleIntNoReplaceExcludeSomeIndices: cannot sample %d indices from 0:%d "
					"as too many of them are excluded ", k, N - 1);
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
	inline void ConvertTo_T_mat_FromDense(const den_mat_t& M, T_mat1& Mout) {
		Mout = M.sparseView();
	};
	template <class T_mat1, typename std::enable_if< std::is_same<den_mat_t, T_mat1>::value>::type* = nullptr  >
	inline void ConvertTo_T_mat_FromDense(const den_mat_t& M, T_mat1& Mout) {
		Mout = M;
	};

}  // namespace GPBoost

#endif   // GPB_UTILS_H_
