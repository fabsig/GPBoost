/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_GP_UTIL_H_
#define GPB_GP_UTIL_H_
#include <memory>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <GPBoost/sparse_matrix_utils.h>
#include <GPBoost/utils.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;

namespace GPBoost {

	/*!
	* \brief Determine unique locations and map duplicates in coordinates to first occurance of unique locations
	* \param coords Coordinates
	* \param num_data Number of data points
	* \param[out] uniques Index of unique coordinates / points
	* \param[out] unique_idx Index that indicates for every data point the corresponding random effect / unique coordinates. Used for constructing incidence matrix Z_ if there are duplicates
	*/
	void DetermineUniqueDuplicateCoords(const den_mat_t& coords,
		data_size_t num_data,
		std::vector<int>& uniques,
		std::vector<int>& unique_idx);

	/*!
	* \brief Determine unique locations and map duplicates in coordinates to first occurance of unique locations
	* \param coords Coordinates
	* \param num_data Number of data points
	* \param[out] uniques Index of unique coordinates / points
	* \param[out] unique_idx Index that indicates for every data point the corresponding random effect / unique coordinates. Used for constructing incidence matrix Z_ if there are duplicates
	*/
	void DetermineUniqueDuplicateCoordsFast(const den_mat_t& coords,
		data_size_t num_data,
		std::vector<int>& uniques,
		std::vector<int>& unique_idx);

	/*!
	* \brief Calculate distance matrix (dense matrix)
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1 == coords2, and dist is a symmetric square matrix
	* \param[out] dist Matrix of dimension coords2.rows() x coords1.rows() with distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1 == coords2
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		den_mat_t& dist) {
		dist = den_mat_t(coords2.rows(), coords1.rows());
		dist.setZero();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < coords2.rows(); ++i) {
			int first_j = 0;
			if (only_one_set_of_coords) {
				dist(i, i) = 0.;
				first_j = i + 1;
			}
			for (int j = first_j; j < coords1.rows(); ++j) {
				dist(i, j) = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
			}
		}
		if (only_one_set_of_coords) {
			dist.triangularView<Eigen::StrictlyLower>() = dist.triangularView<Eigen::StrictlyUpper>().transpose();
		}
	}//end CalculateDistances (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		T_mat& dist) {
		std::vector<Triplet_t> triplets;
		int n_max_entry;
		if (only_one_set_of_coords) {
			n_max_entry = (int)(coords1.rows() - 1) * (int)coords2.rows();
		}
		else {
			n_max_entry = (int)coords1.rows() * (int)coords2.rows();
		}
		triplets.reserve(n_max_entry);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < coords2.rows(); ++i) {
			int first_j = 0;
			if (only_one_set_of_coords) {
#pragma omp critical
				{
					triplets.emplace_back(i, i, 0.);
				}
				first_j = i + 1;
			}
			for (int j = first_j; j < coords1.rows(); ++j) {
				double dist_i_j = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
#pragma omp critical
				{
					triplets.emplace_back(i, j, dist_i_j);
					if (only_one_set_of_coords) {
						triplets.emplace_back(j, i, dist_i_j);
					}
				}
			}
		}
		dist = T_mat(coords2.rows(), coords1.rows());
		dist.setFromTriplets(triplets.begin(), triplets.end());
	}//end CalculateDistances (sparse)

	/*!
	* \brief Calculate distance matrix when compactly supported covariance functions are used
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1 == coords2, and dist is a symmetric square matrix
	* \param taper_range Range parameter of Wendland covariance function / taper beyond which the covariance is zero, and distances are thus not needed
	* \param show_number_non_zeros If true, the percentage of non-zero values is shown
	* \param[out] dist Matrix of dimension coords2.rows() x coords1.rows() with distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1 == coords2
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void CalculateDistancesTapering(const den_mat_t& coords1, //(this is a placeholder which is not used, only here for template compatibility)
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		double,
		bool,
		den_mat_t& dist) {
		CalculateDistances<T_mat>(coords1, coords2, only_one_set_of_coords, dist);
	}//end CalculateDistancesTapering (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void CalculateDistancesTapering(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		double taper_range,
		bool show_number_non_zeros,
		T_mat& dist) {
		std::vector<Triplet_t> triplets;
		int n_max_entry;
		if (only_one_set_of_coords) {
			n_max_entry = 30 * (int)coords1.rows();
		}
		else {
			n_max_entry = 10 * (int)coords1.rows() + 10 * (int)coords2.rows();
		}
		triplets.reserve(n_max_entry);
		//Sort along the sum of the coordinates
		int num_data;
		int dim_coords = (int)coords1.cols();
		double taper_range_square = taper_range * taper_range;
		if (only_one_set_of_coords) {
			num_data = (int)coords1.rows();
		}
		else {
			num_data = (int)(coords1.rows() + coords2.rows());
		}
		std::vector<double> coords_sum(num_data);
		std::vector<int> sort_sum(num_data);
		if (only_one_set_of_coords) {
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data; ++i) {
				coords_sum[i] = coords1(i, Eigen::all).sum();
			}
		}
		else {
			den_mat_t coords_all(num_data, dim_coords);
			coords_all << coords2, coords1;
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data; ++i) {
				coords_sum[i] = coords_all(i, Eigen::all).sum();
			}
		}
		SortIndeces<double>(coords_sum, sort_sum);
		std::vector<int> sort_inv_sum(num_data);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			sort_inv_sum[sort_sum[i]] = i;
		}
		// Search for and calculate distances that are smaller than taper_range
		//  using a fast approach based on results of Ra and Kim (1993)
#pragma omp parallel for schedule(static)
		for (int i = 0; i < coords2.rows(); ++i) {
			if (only_one_set_of_coords) {
#pragma omp critical
				{
					triplets.emplace_back(i, i, 0.);
				}
			}
			bool down = true;
			bool up = true;
			int up_i = sort_inv_sum[i];
			int down_i = sort_inv_sum[i];
			double smd, sed;
			while (up || down) {
				if (down_i == 0) {
					down = false;
				}
				if (up_i == (num_data - 1)) {
					up = false;
				}
				if (down) {
					down_i--;
					if ((only_one_set_of_coords && sort_sum[down_i] > i) ||
						(!only_one_set_of_coords && sort_sum[down_i] >= coords2.rows())) {
						smd = std::pow(coords_sum[sort_sum[down_i]] - coords_sum[i], 2);
						if (smd > dim_coords * taper_range_square) {
							down = false;
						}
						else {
							if (only_one_set_of_coords) {
								sed = (coords1(sort_sum[down_i], Eigen::all) - coords1(i, Eigen::all)).squaredNorm();
							}
							else {
								sed = (coords1(sort_sum[down_i] - coords2.rows(), Eigen::all) - coords2(i, Eigen::all)).squaredNorm();
							}
							if (sed < taper_range_square) {
								double dist_i_j = std::sqrt(sed);
#pragma omp critical
								{
									if (only_one_set_of_coords) {
										triplets.emplace_back(i, sort_sum[down_i], dist_i_j);
										triplets.emplace_back(sort_sum[down_i], i, dist_i_j);
									}
									else {
										triplets.emplace_back(i, sort_sum[down_i] - coords2.rows(), dist_i_j);
									}
								}
							}//end sed < taper_range_square
						}//end smd <= dim_coords * taper_range_square
					}
				}//end down
				if (up) {
					up_i++;
					if ((only_one_set_of_coords && sort_sum[up_i] > i) ||
						(!only_one_set_of_coords && sort_sum[up_i] >= coords2.rows())) {
						smd = std::pow(coords_sum[sort_sum[up_i]] - coords_sum[i], 2);
						if (smd > dim_coords * taper_range_square) {
							up = false;
						}
						else {
							if (only_one_set_of_coords) {
								sed = (coords1(sort_sum[up_i], Eigen::all) - coords1(i, Eigen::all)).squaredNorm();
							}
							else {
								sed = (coords1(sort_sum[up_i] - coords2.rows(), Eigen::all) - coords2(i, Eigen::all)).squaredNorm();
							}
							if (sed < taper_range_square) {
								double dist_i_j = std::sqrt(sed);
#pragma omp critical
								{
									if (only_one_set_of_coords) {
										triplets.emplace_back(i, sort_sum[up_i], dist_i_j);
										triplets.emplace_back(sort_sum[up_i], i, dist_i_j);
									}
									else {
										triplets.emplace_back(i, sort_sum[up_i] - coords2.rows(), dist_i_j);
									}
								}
							}//end sed < taper_range_square
						}//end smd <= dim_coords * taper_range_square
					}
				}//end up
			}//end while (up || down)
		}//end loop over data i

// Old, slow version
//#pragma omp parallel for schedule(static)
//		for (int i = 0; i < coords2.rows(); ++i) {
//			int first_j = 0;
//			if (only_one_set_of_coords) {
//#pragma omp critical
//				{
//					triplets.emplace_back(i, i, 0.);
//				}
//				first_j = i + 1;
//			}
//			for (int j = first_j; j < coords1.rows(); ++j) {
//				double dist_i_j = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
//				if (dist_i_j < taper_range) {
//#pragma omp critical
//					{
//						triplets.emplace_back(i, j, dist_i_j);
//						if (only_one_set_of_coords) {
//							triplets.emplace_back(j, i, dist_i_j);
//						}
//					}
//				}
//			}
//		}

		dist = T_mat(coords2.rows(), coords1.rows());
		dist.setFromTriplets(triplets.begin(), triplets.end());
		dist.makeCompressed();
		if (show_number_non_zeros) {
			double prct_non_zero;
			int non_zeros = (int)dist.nonZeros();
			if (only_one_set_of_coords) {
				prct_non_zero = ((double)non_zeros) / coords1.rows() / coords1.rows() * 100.;
				int num_non_zero_row = non_zeros / (int)coords1.rows();
				Log::REInfo("Average number of non-zero entries per row in covariance matrix: %d (%g %%)", num_non_zero_row, prct_non_zero);
			}
			else {
				prct_non_zero = non_zeros / coords1.rows() / coords2.rows() * 100.;
				Log::REInfo("Number of non-zero entries in covariance matrix: %d (%g %%)", non_zeros, prct_non_zero);
			}
		}
	}//end CalculateDistancesTapering (sparse)

	/*!
	* \brief Subtract the inner product M^TM from a matrix Sigma
	* \param[out] Sigma Matrix from which M^TM is subtracted
	* \param M Matrix M
	* \param only_triangular true/false only compute triangular matrix
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SubtractInnerProdFromMat(T_mat& Sigma,
		const den_mat_t& M,
		bool only_triangular) {
		CHECK(Sigma.rows() == M.cols());
		CHECK(Sigma.cols() == M.cols());
#pragma omp parallel for schedule(static)
		for (int i = 0; i < Sigma.rows(); ++i) {
			for (int j = i; j < Sigma.cols(); ++j) {
				Sigma(i, j) -= M.col(i).dot(M.col(j));
				if (!only_triangular) {
					if (j > i) {
						Sigma(j, i) = Sigma(i, j);
					}
				}
			}
		}
	}//end SubtractInnerProdFromMat (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SubtractInnerProdFromMat(T_mat & Sigma,
		const den_mat_t & M,
		bool only_triangular) {
		CHECK(Sigma.rows() == M.cols());
		CHECK(Sigma.cols() == M.cols());
#pragma omp parallel for schedule(static)
		for (int k = 0; k < Sigma.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				if (i <= j) {
					it.valueRef() -= M.col(i).dot(M.col(j));
					if (!only_triangular) {
						if (i < j) {
							Sigma.coeffRef(j, i) = Sigma.coeff(i, j);
						}
					}
				}
			}
		}
	}//end SubtractInnerProdFromMat (sparse)

	/*!
	* \brief Subtract the product M1^T * M2 from a matrix Sigma
	* \param[out] Sigma Matrix from which M1^T * M2 is subtracted
	* \param M1 Matrix M1
	* \param M2 Matrix M2
	* \param only_triangular true/false only compute triangular matrix
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SubtractProdFromMat(T_mat& Sigma,
		const den_mat_t& M1,
		const den_mat_t& M2,
		bool only_triangular) {
		CHECK(Sigma.rows() == M1.cols());
		CHECK(Sigma.cols() == M2.cols());
#pragma omp parallel for schedule(static)
		for (int i = 0; i < Sigma.rows(); ++i) {
			for (int j = i; j < Sigma.cols(); ++j) {
				Sigma(i, j) -= M1.col(i).dot(M2.col(j));
				if (!only_triangular) {
					if (j > i) {
						Sigma(j, i) = Sigma(i, j);
					}
				}
			}
		}
	}//end SubtractProdFromMat (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SubtractProdFromMat(T_mat & Sigma,
		const den_mat_t & M1,
		const den_mat_t & M2,
		bool only_triangular) {
		CHECK(Sigma.rows() == M1.cols());
		CHECK(Sigma.cols() == M2.cols());
#pragma omp parallel for schedule(static)
		for (int k = 0; k < Sigma.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				if (i <= j) {
					it.valueRef() -= M1.col(i).dot(M2.col(j));
					if (!only_triangular) {
						if (i < j) {
							Sigma.coeffRef(j, i) = Sigma.coeff(i, j);
						}
					}
				}
			}
		}
	}//end SubtractProdFromMat (sparse)

	/*!
	* \brief Subtract the product M1^T * M2 from a matrix non square Sigma (prediction)
	* \param[out] Sigma Matrix from which M1^T * M2 is subtracted
	* \param M1 Matrix M1
	* \param M2 Matrix M2
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SubtractProdFromNonSqMat(T_mat& Sigma,
		const den_mat_t& M1,
		const den_mat_t& M2) {
		CHECK(Sigma.rows() == M1.cols());
		CHECK(Sigma.cols() == M2.cols());
#pragma omp parallel for schedule(static)
		for (int i = 0; i < Sigma.rows(); ++i) {
			for (int j = 0; j < Sigma.cols(); ++j) {
				Sigma(i, j) -= M1.col(i).dot(M2.col(j));
			}
		}
	}//end SubtractProdFromNonSqMat (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SubtractProdFromNonSqMat(T_mat & Sigma,
		const den_mat_t & M1,
		const den_mat_t & M2) {
		CHECK(Sigma.rows() == M1.cols());
		CHECK(Sigma.cols() == M2.cols());
#pragma omp parallel for schedule(static)
		for (int k = 0; k < Sigma.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				it.valueRef() -= M1.col(i).dot(M2.col(j));
			}
		}
	}//end SubtractProdFromNonSqMat (sparse)

	/*!
	* \brief Subtract the matrix from a matrix Sigma
	* \param[out] Sigma Matrix from which M is subtracted
	* \param M Matrix
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SubtractMatFromMat(T_mat& Sigma,
		const den_mat_t& M) {
#pragma omp parallel for schedule(static)
		for (int i = 0; i < Sigma.rows(); ++i) {
			for (int j = i; j < Sigma.cols(); ++j) {
				Sigma(i, j) -= M(i, j);
				if (j > i) {
					Sigma(j, i) = Sigma(i, j);
				}
			}
		}
	}//end SubtractMatFromMat (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SubtractMatFromMat(T_mat & Sigma,
		const den_mat_t & M) {
#pragma omp parallel for schedule(static)
		for (int k = 0; k < Sigma.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				if (i <= j) {
					it.valueRef() -= M(i, j);
					if (i < j) {
						Sigma.coeffRef(j, i) = Sigma.coeff(i, j);
					}
				}
			}
		}
	}//end SubtractMatFromMat (sparse)

	/*
	Calculate the smallest distance between each of the data points and any of the input means.
	* \param means data cluster means that determine the inducing points
	* \param data data coordinates
	* \param[out] distances smallest distance between each of the data points and any of the input means
	*/
	void closest_distance(const den_mat_t& means,
		const den_mat_t& data,
		vec_t& distances);

	/*
	This is an alternate initialization method based on the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
	initialization algorithm.
	* \param data data coordinates
	* \param k Size of inducing points
	* \param gen RNG
	* \param[out] means data cluster means that determine the inducing points
	*/
	void random_plusplus(const den_mat_t& data,
		int k,
		RNG_t& gen,
		den_mat_t& means);

	/*
	Calculate means based on data points and their cluster assignments.
	* \param data data coordinates
	* \param  clusters index of the mean each data point is closest to
	* \param[out] means data cluster means that determine the inducing points
	* \param[out] indices indices of closest data points to means
	*/
	void calculate_means(const den_mat_t& data,
		vec_t& clusters,
		den_mat_t& means,
		vec_t& indices);

	/*
	This implementation of k-means uses [Lloyd's Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
	with the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) used for initializing the means.
	* \param data data coordinates
	* \param k Size of inducing points
	* \param gen RNG
	* \param[out] means data cluster means that determine the inducing points
	* \param[out] max_int maximal number of iterations
	*/
	void kmeans_plusplus(const den_mat_t& data,
		int k,
		RNG_t& gen,
		den_mat_t& means,
		int max_it);

	/*
	Determines indices of data which is inside a ball with given radius around given point
	* \param data data coordinates
	* \param indices_start indices of data considered
	* \param radius radius of ball
	* \param mid centre of ball
	* \param[out] indices indices of data points inside ball
	*/
	void data_in_ball(const den_mat_t& data,
		const std::vector<int>& indices_start,
		double radius,
		const vec_t& mid,
		std::vector<int>& indices);

	/*
	CoverTree Algorithmus
	* \param data data coordinates
	* \param eps size of cover part
	* \param gen RNG
	* \param[out] means data cluster means that determine the inducing points
	*/
	void CoverTree(const den_mat_t& data,
		double eps,
		RNG_t& gen,
		den_mat_t& means);

	/*!
	* \brief Calculate Cholesky decomposition of residual process in full scale approximation
	* \param psi Covariance matrix for which the Cholesky decomposition is calculated
	* \param cluster_i Cluster index for which the Cholesky factor is calculated
	* \param unique_clusters 
	* \param chol_fact_resid Cholesky decompositions of residual covariance matrix
	* \param chol_fact_pattern_analyzed Indicates whether a symbolic decomposition for calculating the Cholesky factor of the covariance matrix has been done or not (only for sparse matrices)
	*/
	template <typename T_mat, typename T_chol, class T_aux = T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_aux>::value || std::is_same<sp_mat_rm_t, T_aux>::value>::type* = nullptr >
	void CalcCholFSAResid(const T_mat & psi, 
		data_size_t cluster_i,
		std::vector<data_size_t>& unique_clusters, 
		std::map<data_size_t, T_chol>& chol_fact_resid,
		bool chol_fact_pattern_analyzed) {
		if (!chol_fact_pattern_analyzed) {
			chol_fact_resid[cluster_i].analyzePattern(psi);
			if (cluster_i == unique_clusters.back()) {
				chol_fact_pattern_analyzed = true;
			}
		}
		chol_fact_resid[cluster_i].factorize(psi);
	}
	template <typename T_mat, typename T_chol, class T_aux = T_mat, typename std::enable_if <std::is_same<den_mat_t, T_aux>::value>::type* = nullptr >
	void CalcCholFSAResid(const den_mat_t& psi, 
		data_size_t cluster_i,
		std::vector<data_size_t>& unique_clusters,
		std::map<data_size_t, T_chol>& chol_fact_resid,
		bool chol_fact_pattern_analyzed) {
		chol_fact_resid[cluster_i].compute(psi);
	}

	/*!
	* \brief Calculate matrices C_s, C_nm, C_m
	* \param unique_clusters 
	* \param re_comps_cross_cov Vectors with cross-covariance GP components
	* \param re_comps_ip Vectors with inducing points GP components
	* \param chol_fact_sigma_ip Cholesky decompositions of inducing points matrix sigma_ip
	* \param re_comps_resid Vectors with residual GP components with sparse(tapered) covariances
	* \param chol_fact_resid Cholesky decompositions of residual covariance matrix
	* \param gp_approx Gaussian process approximation method
	* \param cg_preconditioner_type Preconditioner
	* \param matrix_inversion_method Matrix inversion method
	* \param chol_fact_sigma_woodbury Cholesky decompositions of matrix sigma_ip + cross_cov^T * sigma_resid^-1 * cross_cov used in Woodbury identity
	* \param diagonal_approx_preconditioner_ Diagonal of residual covariance matrix (Preconditioner)
	* \param diagonal_approx_inv_preconditioner_ Inverse of diagonal of residual covariance matrix (Preconditioner)
	* \param chol_fact_woodbury_preconditioner_ Cholesky decompositions of matrix sigma_ip + cross_cov^T * D^-1 * cross_cov used in Woodbury identity where D is given by the Preconditioner
	* \param FITC_Diag diagonal of fully independent training conditional for predictive process
	* \param num_data_per_cluster Number of observed locations per independent realization
	* \param chol_fact_pattern_analyzed Indicates whether a symbolic decomposition for calculating the Cholesky factor of the covariance matrix has been done or not (only for sparse matrices)
	* \param gauss_likelihood gauss or not
	*/
	template <typename T_mat, typename T_chol>
	void CalcCovFactorsPPFSA(std::vector<data_size_t>& unique_clusters,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<den_mat_t>>>>& re_comps_cross_cov,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<den_mat_t>>>>& re_comps_ip,
		std::map<data_size_t, chol_den_mat_t>& chol_fact_sigma_ip,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<T_mat>>>>& re_comps_resid, 
		std::map<data_size_t, T_chol>& chol_fact_resid,
		const string_t gp_approx,
		const string_t cg_preconditioner_type,
		const string_t matrix_inversion_method,
		std::map<data_size_t, chol_den_mat_t>& chol_fact_sigma_woodbury,
		std::map<data_size_t, vec_t>& diagonal_approx_preconditioner_,
		std::map<data_size_t, vec_t>& diagonal_approx_inv_preconditioner_,
		std::map<data_size_t, chol_den_mat_t>& chol_fact_woodbury_preconditioner_,
		std::map<data_size_t, vec_t>& FITC_Diag,
		std::map<data_size_t, int>& num_data_per_cluster,
		bool chol_fact_pattern_analyzed,
		bool gauss_likelihood) {
		for (const auto& cluster_i : unique_clusters) {
			// factorize matrix used in Woodbury identity
			std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov[cluster_i][0]->GetZSigmaZt();
			den_mat_t sigma_ip_stable = *(re_comps_ip[cluster_i][0]->GetZSigmaZt());
			den_mat_t sigma_woodbury;// sigma_woodbury = sigma_ip + cross_cov^T * sigma_resid^-1 * cross_cov or for Preconditioner sigma_ip + cross_cov^T * D^-1 * cross_cov
			if (matrix_inversion_method == "iterative") {
				if (gp_approx == "FITC") {
					Log::REFatal("The iterative methods are not implemented for Predictive Processes. Please use Cholesky.");
				}
				else if (gp_approx == "full_scale_tapering") {
					std::shared_ptr<T_mat> sigma_resid = re_comps_resid[cluster_i][0]->GetZSigmaZt();
					if (cg_preconditioner_type == "predictive_process_plus_diagonal") {
						diagonal_approx_preconditioner_[cluster_i] = (*sigma_resid).diagonal();
						diagonal_approx_inv_preconditioner_[cluster_i] = diagonal_approx_preconditioner_[cluster_i].cwiseInverse();
						sigma_woodbury = (*cross_cov).transpose() * (diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal() * (*cross_cov));
						sigma_woodbury += *(re_comps_ip[cluster_i][0]->GetZSigmaZt());

						chol_fact_woodbury_preconditioner_[cluster_i].compute(sigma_woodbury);
					}
					else if (cg_preconditioner_type != "none") {
						Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
					}

				}
			}
			else if (matrix_inversion_method == "cholesky") {
				if (gp_approx == "FITC") {
					den_mat_t sigma_ip_Ihalf_sigma_cross_covT;
					TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip[cluster_i],
						(*cross_cov).transpose(), sigma_ip_Ihalf_sigma_cross_covT, false);
					if (gauss_likelihood) {
						FITC_Diag[cluster_i] = vec_t::Ones(num_data_per_cluster[cluster_i]);//add nugget effect variance
					}
					else {
						FITC_Diag[cluster_i] = vec_t::Zero(num_data_per_cluster[cluster_i]);
					}
					FITC_Diag[cluster_i] = FITC_Diag[cluster_i].array() + sigma_ip_stable.coeffRef(0, 0);
#pragma omp parallel for schedule(static)
					for (int ii = 0; ii < num_data_per_cluster[cluster_i]; ++ii) {
						FITC_Diag[cluster_i][ii] -= sigma_ip_Ihalf_sigma_cross_covT.col(ii).array().square().sum();
					}
					sigma_woodbury = ((*cross_cov).transpose() * FITC_Diag[cluster_i].cwiseInverse().asDiagonal()) * (*cross_cov);
				}
				else if (gp_approx == "full_scale_tapering") {
					// factorize residual covariance matrix
					std::shared_ptr<T_mat> sigma_resid = re_comps_resid[cluster_i][0]->GetZSigmaZt();
					CalcCholFSAResid<T_mat, T_chol>(*sigma_resid, cluster_i, unique_clusters, chol_fact_resid, chol_fact_pattern_analyzed);
					den_mat_t sigma_resid_Ihalf_cross_cov;

					//ApplyPermutationCholeskyFactor<den_mat_t, T_chol>(chol_fact_resid_[cluster_i], *cross_cov, sigma_resid_Ihalf_cross_cov, false);//DELETE_SOLVEINPLACE
					//chol_fact_resid_[cluster_i].matrixL().solveInPlace(sigma_resid_Ihalf_cross_cov);

					TriangularSolveGivenCholesky<T_chol, T_mat, den_mat_t, den_mat_t>(chol_fact_resid[cluster_i], *cross_cov, sigma_resid_Ihalf_cross_cov, false);

					sigma_woodbury = sigma_resid_Ihalf_cross_cov.transpose() * sigma_resid_Ihalf_cross_cov;
				}
				sigma_woodbury += sigma_ip_stable;
				chol_fact_sigma_woodbury[cluster_i].compute(sigma_woodbury);
			}
			else {
				Log::REFatal("Matrix inversion method '%s' is not supported.", matrix_inversion_method.c_str());
			}
		}
	}//end CalcCovFactorsPPFSA

	/*!
	* \brief Initialize individual component models and collect them in a containter
	* \param num_data Number of data points
	* \param data_indices_per_cluster Keys: Labels of independent realizations of REs/GPs, values: vectors with indices for data points
	* \param cluster_i Index / label of the realization of the Gaussian process for which the components should be constructed
	* \param gp_coords_data Coordinates (features) for Gaussian process
	* \param gp_rand_coef_data Covariate data for Gaussian process random coefficients
	* \param[out] re_comps_ip_cluster_i Inducing point GP for predictive process
	* \param[out] re_comps_cross_cov_cluster_i Cross-covariance GP for predictive process
	* \param[out] re_comps_resid_cluster_i Residual GP component for full scale approximation
	* \param gp_approx Gaussian process approximation method
	* \param num_data_per_cluster Number of observed locations per independent realization
	* \param num_ind_points Number of inducing points
	* \param num_gp 1 if there is a Gaussian process 0 otherwise
	* \param dim_gp_coords Dimension of the coordinates(= number of features) for Gaussian process
	* \param gp_coords_obs_mat Coordinates of all observed points
	* \param method_ind_points Method for inducing points
	* \param rng Random Number Generator
	* \param gp_coords_ip_mat Coordinates of inducing points
	* \param cov_fct Type of covariance(kernel) function for Gaussian processes
	* \param cov_fct_shape Shape parameter of covariance function
	* \param cov_fct_taper_range Range parameter of the Wendland covariance functionand Wendland correlation taper
	* \param cov_fct_taper_shape Shape parameter of the Wendland correlation taper
	* \param num_gp_rand_coef Number of random coefficient GPs
	*/
	template <typename T_mat>
	void CreateREComponentsPPFSA(data_size_t num_data,
		std::map<data_size_t, std::vector<int>>& data_indices_per_cluster,
		data_size_t cluster_i,
		const double* gp_coords_data,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
		std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		std::vector<std::shared_ptr<RECompGP<T_mat>>>& re_comps_resid_cluster_i,
		const string_t gp_approx, 
		std::map<data_size_t, int>& num_data_per_cluster,
		int num_ind_points,
		data_size_t num_gp,
		const int dim_gp_coords,
		den_mat_t gp_coords_obs_mat,
		const string_t method_ind_points,
		RNG_t rng,
		den_mat_t gp_coords_ip_mat,
		const string_t cov_fct,
		const double cov_fct_shape,
		const double cov_fct_taper_range,
		const double cov_fct_taper_shape,
		data_size_t num_gp_rand_coef) {
		if (gp_approx == "FITC") {
			if (num_data_per_cluster[cluster_i] < num_ind_points) {
				Log::REFatal("Cannot have more inducing points than data points for '%s' approximation ", gp_approx.c_str());
			}
		}
		else if (gp_approx == "full_scale_tapering") {
			if (num_data_per_cluster[cluster_i] <= num_ind_points) {
				Log::REFatal("Need to have less inducing points than data points for '%s' approximation ", gp_approx.c_str());
			}
		}
		CHECK(num_gp > 0);
		std::vector<double> gp_coords_all;
		for (int j = 0; j < dim_gp_coords; ++j) {
			for (const auto& id : data_indices_per_cluster[cluster_i]) {
				gp_coords_all.push_back(gp_coords_data[j * num_data + id]);
			}
		}
		den_mat_t gp_coords_all_mat = Eigen::Map<den_mat_t>(gp_coords_all.data(), num_data_per_cluster[cluster_i], dim_gp_coords);
		gp_coords_obs_mat = gp_coords_all_mat;
		// Inducing points
		std::vector<int> indices;
		std::vector<double> gp_coords_ip;
		den_mat_t gp_coords_ip_mat_;
		if (method_ind_points == "CoverTree") {
			CoverTree(gp_coords_all_mat, 1 / ((double)num_ind_points), rng, gp_coords_ip_mat_);
			num_ind_points = (int)gp_coords_ip_mat_.rows();
		}
		else if (method_ind_points == "random") {
			SampleIntNoReplaceSort(num_data_per_cluster[cluster_i], num_ind_points, rng, indices);
			for (int j = 0; j < dim_gp_coords; ++j) {
				for (const auto& ind : indices) {
					gp_coords_ip.push_back(gp_coords_data[j * num_data + data_indices_per_cluster[cluster_i][ind]]);
				}
			}
			gp_coords_ip_mat_ = Eigen::Map<den_mat_t>(gp_coords_ip.data(), num_ind_points, dim_gp_coords);
		}
		else if (method_ind_points == "kmeans++") {
			gp_coords_ip_mat_.resize(num_ind_points, gp_coords_all_mat.cols());
			int max_it_kmeans = 1000;
			kmeans_plusplus(gp_coords_all_mat, num_ind_points, rng, gp_coords_ip_mat_, max_it_kmeans);
		}
		else {
			Log::REFatal("Method '%s' is not supported for finding inducing points ", method_ind_points.c_str());
		}
		gp_coords_ip_mat = gp_coords_ip_mat_;

		std::shared_ptr<RECompGP<den_mat_t>> gp_ip(new RECompGP<den_mat_t>(
			gp_coords_ip_mat, cov_fct, cov_fct_shape, cov_fct_taper_range, cov_fct_taper_shape, false, false, true, false, false));
		if (gp_ip->HasDuplicatedCoords()) {
			Log::REFatal("Duplicates found in inducing points / low-dimensional knots ");
		}
		re_comps_ip_cluster_i.push_back(gp_ip);
		re_comps_cross_cov_cluster_i.push_back(std::shared_ptr<RECompGP<den_mat_t>>(new RECompGP<den_mat_t>(
			gp_coords_all_mat, gp_coords_ip_mat, cov_fct, cov_fct_shape, cov_fct_taper_range, cov_fct_taper_shape, false, false)));
		if (gp_approx == "full_scale_tapering") {
			re_comps_resid_cluster_i.push_back(std::shared_ptr<RECompGP<T_mat>>(new RECompGP<T_mat>(
				gp_coords_all_mat, cov_fct, cov_fct_shape, cov_fct_taper_range, cov_fct_taper_shape,
				true, true, true, false, false)));
		}
		//Random slope GPs
		if (num_gp_rand_coef > 0) {
			Log::REFatal("Random coefficients are currently not supported for '%s' approximation ", method_ind_points.c_str());
		}
	}//end CreateREComponentsPPFSA


	/*!
	* \brief Calculate predictions (conditional mean and covariance matrix) using the PP/FSA approximation
	* \param cluster_i Cluster index for which prediction are made
	* \param num_data_per_cluster_pred Number of prediction locations per independent realization
	* \param num_data_per_cluster Number of observed locations per independent realization
	* \param re_comps_cross_cov Vectors with cross-covariance GP components
	* \param re_comps_ip Vectors with inducing points GP components
	* \param chol_fact_sigma_ip Cholesky decompositions of inducing points matrix sigma_ip
	* \param re_comps_resid Vectors with residual GP components with sparse(tapered) covariances
	* \param y_aux Psi^-1*y_
	* \param gp_coords_mat_pred Coordinates for prediction locations
	* \param calc_pred_cov If true, the covariance matrix is also calculated
	* \param calc_pred_var If true, predictive variances are also calculated
	* \param[out] pred_mean Predictive mean (only for Gaussian likelihoods)
	* \param[out] pred_cov Predictive covariance matrix (only for Gaussian likelihoods)
	* \param[out] pred_var Predictive variances (only for Gaussian likelihoods)
	* \param nsim_var_pred Number of random vectors
	* \param cg_delta_conv_pred Tolerance level for L2 norm of residuals for checking convergence in conjugate gradient algorithm when being used for prediction
	* \param gp_approx Gaussian process approximation method
	* \param cg_preconditioner_type Preconditioner
	* \param matrix_inversion_method Matrix inversion method
	* \param chol_fact_woodbury_preconditioner_ Matrix inversion methodCholesky decompositions of matrix sigma_ip + cross_cov^T * D^-1 * cross_cov used in Woodbury identity where D is given by the Preconditioner
	* \param diagonal_approx_inv_preconditioner_ Inverse of diagonal of residual covariance matrix (Preconditioner)
	* \param chol_fact_sigma_woodbury Cholesky decompositions of matrix sigma_ip + cross_cov^T * sigma_resid^-1 * cross_cov used in Woodbury identity
	* \param chol_fact_resid Cholesky decompositions of residual covariance matrix
	* \param FITC_Diag Diagonal of fully independent training conditional for predictive process
	* \param num_comps_total Number of components
	* \param cg_generator Random number generator
	*/
	template <typename T_mat, typename T_chol>
	void CalcPredPPFSA(data_size_t cluster_i,
		std::map<data_size_t, int>& num_data_per_cluster_pred,
		std::map<data_size_t, int>& num_data_per_cluster,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<den_mat_t>>>>& re_comps_cross_cov,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<den_mat_t>>>>& re_comps_ip,
		std::map<data_size_t, chol_den_mat_t>& chol_fact_sigma_ip,
		std::map<data_size_t, std::vector<std::shared_ptr<RECompGP<T_mat>>>>& re_comps_resid,
		std::map<data_size_t, vec_t>& y_aux,
		const den_mat_t& gp_coords_mat_pred,
		bool calc_pred_cov,
		bool calc_pred_var,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var,
		int nsim_var_pred,
		const double cg_delta_conv_pred,
		const string_t gp_approx,
		const string_t cg_preconditioner_type,
		const string_t matrix_inversion_method,
		std::map<data_size_t, chol_den_mat_t>& chol_fact_woodbury_preconditioner_,
		std::map<data_size_t, vec_t>& diagonal_approx_inv_preconditioner_,
		std::map<data_size_t, chol_den_mat_t>& chol_fact_sigma_woodbury,
		std::map<data_size_t, T_chol>& chol_fact_resid,
		std::map<data_size_t, vec_t>& FITC_Diag,
		data_size_t num_comps_total,
		RNG_t& cg_generator) {

		int num_data_cli = num_data_per_cluster[cluster_i];
		int num_data_pred_cli = num_data_per_cluster_pred[cluster_i];

		// Initialization of Components C_pm & C_pn & C_pp
		den_mat_t cross_cov_pred_ip;
		den_mat_t cov_mat_pred_id; // unused dummy variable
		den_mat_t cross_dist; // unused dummy variable
		T_mat sigma_resid_pred_obs;
		T_mat cov_mat_pred_obs; // unused dummy variable
		T_mat cross_dist_resid;
		T_mat sigma_resid_pred;
		T_mat cov_mat_pred; // unused dummy variable
		T_mat cross_dist_resid_pred;
		bool NaN_found = false;

		for (int j = 0; j < num_comps_total; ++j) {
			// Construct components
			std::shared_ptr<den_mat_t> cross_cov = re_comps_cross_cov[cluster_i][j]->GetZSigmaZt();
			den_mat_t sigma_ip_stable = *(re_comps_ip[cluster_i][j]->GetZSigmaZt());
			// Cross-covariance between predictions and inducing points C_pm
			std::shared_ptr<RECompGP<den_mat_t>> re_comp_cross_cov_cluster_i_pred_ip = std::dynamic_pointer_cast<RECompGP<den_mat_t>>(re_comps_cross_cov[cluster_i][j]);
			re_comp_cross_cov_cluster_i_pred_ip->AddPredCovMatrices(re_comp_cross_cov_cluster_i_pred_ip->coords_ind_point_, gp_coords_mat_pred, cross_cov_pred_ip,
				cov_mat_pred_id, true, false, true, nullptr, false, cross_dist);
			// Calculating predictive mean
			pred_mean = cross_cov_pred_ip * chol_fact_sigma_ip[cluster_i].solve((*cross_cov).transpose() * y_aux[cluster_i]);
			den_mat_t chol_ip_cross_cov_ip_pred;
			std::shared_ptr<T_mat> sigma_resid;
			if (gp_approx == "full_scale_tapering") {
				// Residual matrix
				sigma_resid = re_comps_resid[cluster_i][j]->GetZSigmaZt();
				// Cross-covariance between predictions and observations C_pn (tapered)
				std::shared_ptr<RECompGP<T_mat>> re_comps_resid_po_cluster_i = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid[cluster_i][j]);
				re_comps_resid_po_cluster_i->AddPredCovMatrices(re_comps_resid_po_cluster_i->coords_, gp_coords_mat_pred, sigma_resid_pred_obs,
					cov_mat_pred_obs, true, false, true, nullptr, true, cross_dist_resid);
				// Calculate Cm_inv * C_mn part of predictive process
				den_mat_t sigma_ip_inv_cross_cov_ip_ob = chol_fact_sigma_ip[cluster_i].solve((*cross_cov).transpose());
				// Residual part
				// Subtract predictive process (prediction) covariance
				SubtractProdFromNonSqMat<T_mat>(sigma_resid_pred_obs, cross_cov_pred_ip.transpose(), sigma_ip_inv_cross_cov_ip_ob);
				// Apply taper
				re_comps_resid_po_cluster_i->ApplyTaper(cross_dist_resid, sigma_resid_pred_obs);

				pred_mean += sigma_resid_pred_obs * y_aux[cluster_i];
			}
			// Calculating predicitve covariance and variance
			if (calc_pred_cov || calc_pred_var) {
				// Add nugget and autocovariance of predictions 
				if (calc_pred_var) {
					pred_var = vec_t::Ones(num_data_pred_cli);
					re_comp_cross_cov_cluster_i_pred_ip->AddPredUncondVar(pred_var.data(), num_data_pred_cli, nullptr);
				}
				T_mat PP_Part;
				// Add prediction matrix to predictive Covariance
				if (calc_pred_cov) {
					Log::REInfo("The computational complexity and the storage of the predictive covariance heavily depend on the number of prediction location. Therefore, if this number is large we recommend only computing the predictive variances.");
					pred_cov = T_mat(num_data_pred_cli, num_data_pred_cli);
					pred_cov.setIdentity();
					TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip[cluster_i], cross_cov_pred_ip.transpose(), chol_ip_cross_cov_ip_pred, false);
					ConvertTo_T_mat_FromDense<T_mat>(chol_ip_cross_cov_ip_pred.transpose() * chol_ip_cross_cov_ip_pred, PP_Part);

					pred_cov += PP_Part;

					if (gp_approx == "full_scale_tapering") {
						std::shared_ptr<RECompGP<T_mat>> re_comps_resid_pp_cluster_i = std::dynamic_pointer_cast<RECompGP<T_mat>>(re_comps_resid[cluster_i][j]);
						re_comps_resid_pp_cluster_i->AddPredCovMatrices(gp_coords_mat_pred, gp_coords_mat_pred, sigma_resid_pred,
							cov_mat_pred, true, false, true, nullptr, true, cross_dist_resid_pred);
						// Subtract predictive process (predict) 
						SubtractInnerProdFromMat<T_mat>(sigma_resid_pred, chol_ip_cross_cov_ip_pred, false);
						// Apply taper
						re_comps_resid_pp_cluster_i->ApplyTaper(cross_dist_resid_pred, sigma_resid_pred);
						pred_cov += sigma_resid_pred;
					}
					else if (gp_approx == "FITC") {
						vec_t diagonal_resid(num_data_pred_cli);
						diagonal_resid.setZero();
						diagonal_resid = diagonal_resid.array() + sigma_ip_stable.coeffRef(0, 0);
#pragma omp parallel for schedule(static)
						for (int ii = 0; ii < num_data_pred_cli; ++ii) {
							diagonal_resid[ii] -= chol_ip_cross_cov_ip_pred.col(ii).array().square().sum();
						}
						pred_cov += diagonal_resid.asDiagonal();
					}
				}
				// Calculate remaining part of predictive covariance
				T_mat woodbury_Part;
				T_mat cross_cov_part;
				if (calc_pred_cov) {
					if (gp_approx == "full_scale_tapering") {
						// Whole cross-covariance as dense matrix 
						den_mat_t sigma_obs_pred_dense = (*cross_cov) * chol_fact_sigma_ip[cluster_i].solve(cross_cov_pred_ip.transpose());
						sigma_obs_pred_dense += sigma_resid_pred_obs.transpose();
						if (gp_approx == "iterative") {
							den_mat_t sigma_inv_sigma_obs_pred;
							CGFSA_MULTI_RHS<T_mat>(*sigma_resid, *cross_cov, chol_fact_sigma_ip[cluster_i], sigma_obs_pred_dense, sigma_inv_sigma_obs_pred, NaN_found,
								num_data_cli, num_data_pred_cli, 1000, cg_delta_conv_pred, cg_preconditioner_type,
								chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
							ConvertTo_T_mat_FromDense<T_mat>(sigma_obs_pred_dense.transpose() * sigma_inv_sigma_obs_pred, cross_cov_part);
							pred_cov -= cross_cov_part;
						}
						else if (gp_approx == "cholesky") {
							den_mat_t sigma_resid_inv_sigma_obs_pred = chol_fact_resid[cluster_i].solve(sigma_obs_pred_dense);
							den_mat_t sigma_resid_inv_sigma_obs_pred_cross_cov_pred_ip = sigma_resid_inv_sigma_obs_pred * cross_cov_pred_ip;
							ConvertTo_T_mat_FromDense<T_mat>(sigma_obs_pred_dense.transpose() * sigma_resid_inv_sigma_obs_pred, cross_cov_part);
							pred_cov -= cross_cov_part;
							ConvertTo_T_mat_FromDense<T_mat>(sigma_resid_inv_sigma_obs_pred_cross_cov_pred_ip * chol_fact_sigma_woodbury[cluster_i].solve(sigma_resid_inv_sigma_obs_pred_cross_cov_pred_ip.transpose()), woodbury_Part);
							pred_cov += woodbury_Part;
						}
					}
					else if (gp_approx == "FITC") {
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_ip * chol_fact_sigma_ip[cluster_i].solve(cross_cov_pred_ip.transpose()), cross_cov_part);
						pred_cov -= cross_cov_part;
						ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_ip * chol_fact_sigma_woodbury[cluster_i].solve(cross_cov_pred_ip.transpose()), woodbury_Part);
						pred_cov += woodbury_Part;
					}
				} // end calc_pred_cov 
				// Calculate remaining part of predictive variances
				if (calc_pred_var) {
					if (gp_approx == "full_scale_tapering") {
						if (matrix_inversion_method == "iterative") {
							// Stochastic Diagonal
							// Sample vectors
							den_mat_t rand_vec_probe_init(num_data_pred_cli, nsim_var_pred);
							GenRandVecDiag(cg_generator, rand_vec_probe_init);
							den_mat_t rand_vec_probe_pred(num_data_cli, nsim_var_pred);
							rand_vec_probe_pred.setZero();
							// sigma_resid_pred^T * rand_vec_probe_init
#pragma omp parallel for schedule(static)   
							for (int i = 0; i < rand_vec_probe_pred.cols(); ++i) {
								rand_vec_probe_pred.col(i) += sigma_resid_pred_obs.transpose() * rand_vec_probe_init.col(i);
							}
							// sigma_resid^-1 * rand_vec_probe_pred
							den_mat_t sigma_resid_inv_pv(num_data_cli, rand_vec_probe_pred.cols());
							CGFSA_RESID<T_mat>(*sigma_resid, rand_vec_probe_pred, sigma_resid_inv_pv, NaN_found, num_data_cli, (int)rand_vec_probe_pred.cols(),
								1000, cg_delta_conv_pred, cg_preconditioner_type, diagonal_approx_inv_preconditioner_[cluster_i]);
							// sigma_resid_pred * sigma_resid_inv_pv
							den_mat_t rand_vec_probe_final(num_data_pred_cli, sigma_resid_inv_pv.cols());
							rand_vec_probe_final.setZero();
#pragma omp parallel for schedule(static)   
							for (int i = 0; i < rand_vec_probe_final.cols(); ++i) {
								rand_vec_probe_final.col(i) += sigma_resid_pred_obs * sigma_resid_inv_pv.col(i);
							}
							den_mat_t sample_sigma = rand_vec_probe_final.cwiseProduct(rand_vec_probe_init);
							vec_t stoch_diag = sample_sigma.rowwise().mean();

							// Exact Diagonal (Preconditioner)
							vec_t diag_P(num_data_pred_cli);
							T_mat sigma_resid_pred_obs_pred_var = sigma_resid_pred_obs * (diagonal_approx_inv_preconditioner_[cluster_i].cwiseSqrt()).asDiagonal();
							T_mat* R_ptr_2 = &sigma_resid_pred_obs_pred_var;
#pragma omp parallel for schedule(static)   
							for (int i = 0; i < num_data_pred_cli; ++i) {
								diag_P[i] = ((vec_t)(R_ptr_2->row(i))).array().square().sum();
							}

							// Stochastic Diagonal (Preconditioner)
							den_mat_t rand_vec_probe_cv(num_data_pred_cli, rand_vec_probe_init.cols());
							rand_vec_probe_cv.setZero();
							den_mat_t preconditioner_rand_vec_probe = diagonal_approx_inv_preconditioner_[cluster_i].asDiagonal() * rand_vec_probe_pred;
#pragma omp parallel for schedule(static)   
							for (int i = 0; i < preconditioner_rand_vec_probe.cols(); ++i) {
								rand_vec_probe_cv.col(i) += sigma_resid_pred_obs * preconditioner_rand_vec_probe.col(i);
							}
							den_mat_t sample_P = rand_vec_probe_cv.cwiseProduct(rand_vec_probe_init);
							vec_t diag_P_stoch = sample_P.rowwise().mean();

							// Variance Reduction
							// Optimal c
							vec_t c_opt;
							CalcOptimalCVectorized(sample_sigma, sample_P, stoch_diag, diag_P, c_opt);
							stoch_diag += c_opt.cwiseProduct(diag_P - diag_P_stoch);
							pred_var -= stoch_diag;
							// CG: sigma_resid^-1 * cross_cov
							den_mat_t sigma_resid_inv_cross_cov(num_data_cli, (*cross_cov).cols());
							CGFSA_RESID<T_mat>(*sigma_resid, *cross_cov, sigma_resid_inv_cross_cov, NaN_found, num_data_cli, (int)(*cross_cov).cols(),
								1000, cg_delta_conv_pred, cg_preconditioner_type, diagonal_approx_inv_preconditioner_[cluster_i]);
							// CG: sigma^-1 * cross_cov
							den_mat_t sigma_inv_cross_cov(num_data_cli, (*cross_cov).cols());
							CGFSA_MULTI_RHS<T_mat>(*sigma_resid, *cross_cov, chol_fact_sigma_ip[cluster_i], *cross_cov, sigma_inv_cross_cov, NaN_found,
								num_data_cli, (int)(*cross_cov).cols(), 1000, cg_delta_conv_pred, cg_preconditioner_type,
								chol_fact_woodbury_preconditioner_[cluster_i], diagonal_approx_inv_preconditioner_[cluster_i]);
							// sigma_ip^-1 * cross_cov_pred
							den_mat_t sigma_ip_inv_cross_cov_pred = chol_fact_sigma_ip[cluster_i].solve(cross_cov_pred_ip.transpose());
							// cross_cov^T * sigma^-1 * cross_cov
							den_mat_t auto_cross_cov = (*cross_cov).transpose() * sigma_inv_cross_cov;
							// cross_cov^T * sigma^-1 * cross_cov * sigma_ip^-1 * cross_cov_pred
							den_mat_t auto_cross_cov_sigma_ip_inv_cross_cov_pred = auto_cross_cov * sigma_ip_inv_cross_cov_pred;
							// sigma_resid_pred * sigma^-1 * cross_cov
							den_mat_t sigma_resid_pred_obs_sigma_inv_cross_cov(num_data_pred_cli, (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
							for (int i = 0; i < sigma_resid_pred_obs_sigma_inv_cross_cov.cols(); ++i) {
								sigma_resid_pred_obs_sigma_inv_cross_cov.col(i) = sigma_resid_pred_obs * sigma_inv_cross_cov.col(i);
							}
							// cross_cov^T * sigma_resid^-1 * cross_cov
							den_mat_t cross_cov_sigma_resid_inv_cross_cov = (*cross_cov).transpose() * sigma_resid_inv_cross_cov;
							// Ensure symmetry
							cross_cov_sigma_resid_inv_cross_cov = (cross_cov_sigma_resid_inv_cross_cov + cross_cov_sigma_resid_inv_cross_cov.transpose()) / 2;
							// Woodburry factor
							chol_den_mat_t Woodburry_fact_chol;
							Woodburry_fact_chol.compute(sigma_ip_stable + cross_cov_sigma_resid_inv_cross_cov);
							den_mat_t Woodburry_fact;
							TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(Woodburry_fact_chol, sigma_resid_inv_cross_cov.transpose(), Woodburry_fact, false);
							den_mat_t sigma_resid_pred_obs_WF(num_data_pred_cli, (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
							for (int i = 0; i < sigma_resid_pred_obs_WF.cols(); ++i) {
								sigma_resid_pred_obs_WF.col(i) = sigma_resid_pred_obs * Woodburry_fact.transpose().col(i);
							}
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_pred_cli; ++i) {
								pred_var[i] -= sigma_ip_inv_cross_cov_pred.col(i).dot(auto_cross_cov_sigma_ip_inv_cross_cov_pred.col(i))
									+ 2 * sigma_ip_inv_cross_cov_pred.col(i).dot(sigma_resid_pred_obs_sigma_inv_cross_cov.transpose().col(i))
									- sigma_resid_pred_obs_WF.transpose().col(i).array().square().sum();
							}
							if ((pred_var.array() < 0.0).any()) {
								Log::REWarning("There are negative estimates for variances. Use more sample vectors to reduce the variability of the stochastic estimate.");
							}
						}
						else {
							// sigma_resid^-1 * cross_cov
							den_mat_t sigma_resid_inv_cross_cov = chol_fact_resid[cluster_i].solve((*cross_cov));
							// sigma_ip^-1 * cross_cov_pred^T
							den_mat_t sigma_ip_inv_cross_cov_pred = chol_fact_sigma_ip[cluster_i].solve(cross_cov_pred_ip.transpose());
							// cross_cov^T * sigma_resid^-1 * cross_cov * sigma_ip^-1 * cross_cov_pred
							den_mat_t auto_cross_cov = ((*cross_cov).transpose() * sigma_resid_inv_cross_cov) * sigma_ip_inv_cross_cov_pred;
							// Sigma_resid_pred * sigma_resid^-1 * cross_cov
							den_mat_t sigma_resid_pred_obs_sigma_resid_inv_cross_cov(num_data_pred_cli, (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
							for (int i = 0; i < sigma_resid_pred_obs_sigma_resid_inv_cross_cov.cols(); ++i) {
								sigma_resid_pred_obs_sigma_resid_inv_cross_cov.col(i) = sigma_resid_pred_obs * sigma_resid_inv_cross_cov.col(i);
							}
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_pred_cli; ++i) {
								pred_var[i] -= 2 * sigma_ip_inv_cross_cov_pred.col(i).dot(sigma_resid_pred_obs_sigma_resid_inv_cross_cov.transpose().col(i))
									+ auto_cross_cov.col(i).dot(sigma_ip_inv_cross_cov_pred.col(i));
							}
							vec_t sigma_resid_inv_sigma_resid_pred_col;
							T_mat* R_ptr = &sigma_resid_pred_obs;
							for (int i = 0; i < num_data_pred_cli; ++i) {
								TriangularSolveGivenCholesky<T_chol, T_mat, vec_t, vec_t>(chol_fact_resid[cluster_i], ((vec_t)(R_ptr->row(i))).transpose(), sigma_resid_inv_sigma_resid_pred_col, false);
								pred_var[i] -= sigma_resid_inv_sigma_resid_pred_col.array().square().sum();
							}
							// Woodburry matrix part
							den_mat_t Woodburry_fact_sigma_resid_inv_cross_cov;
							TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury[cluster_i], sigma_resid_inv_cross_cov.transpose(), Woodburry_fact_sigma_resid_inv_cross_cov, false);
							den_mat_t auto_cross_cov_pred = (Woodburry_fact_sigma_resid_inv_cross_cov * (*cross_cov)) * sigma_ip_inv_cross_cov_pred;
							den_mat_t sigma_resid_pred_obs_Woodburry_fact(num_data_pred_cli, (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
							for (int i = 0; i < sigma_resid_pred_obs_Woodburry_fact.cols(); ++i) {
								sigma_resid_pred_obs_Woodburry_fact.col(i) = sigma_resid_pred_obs * Woodburry_fact_sigma_resid_inv_cross_cov.transpose().col(i);
							}
#pragma omp parallel for schedule(static)
							for (int i = 0; i < num_data_pred_cli; ++i) {
								pred_var[i] += 2 * auto_cross_cov_pred.col(i).dot(sigma_resid_pred_obs_Woodburry_fact.transpose().col(i))
									+ auto_cross_cov_pred.col(i).array().square().sum()
									+ sigma_resid_pred_obs_Woodburry_fact.transpose().col(i).array().square().sum();
							}
						}
					}//end FSA 
					else if (gp_approx == "FITC") { // Predictive Process
						den_mat_t sigma_ip_inv_cross_cov_pred = chol_fact_sigma_ip[cluster_i].solve(cross_cov_pred_ip.transpose());
						den_mat_t Fact_FITC_R = (((*cross_cov).transpose() * FITC_Diag[cluster_i].cwiseInverse().asDiagonal()) * (*cross_cov)) * sigma_ip_inv_cross_cov_pred;
						den_mat_t Woodburry_fact;
						TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury[cluster_i], Fact_FITC_R, Woodburry_fact, false);
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_data_pred_cli; ++i) {
							pred_var[i] -= Fact_FITC_R.col(i).dot(sigma_ip_inv_cross_cov_pred.col(i))
								- Woodburry_fact.col(i).array().square().sum();
						}
					}//end predictive_process
				}//end calc_pred_var
			}//end calc_pred_cov || calc_pred_var
		}
	}//end CalcPredPPFSA

}  // namespace GPBoost

#endif   // GPB_GP_UTIL_H_
