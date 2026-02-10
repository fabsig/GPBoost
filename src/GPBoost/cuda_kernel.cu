/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifdef USE_CUDA_GP
#include <chrono>  // only for debugging
#include <thread> // only for debugging
#include <cstdio>
#include <math.h>
#include <GPBoost/GP_utils.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <device_launch_parameters.h>
#include <cusolverDn.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;

// Define infinity
#ifndef CUDART_INF
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#endif

// Maximum neighbor size per data point
#define MAX_K 64


namespace GPBoost {

#define CUDA_CHECK(call)                                                     \
{                                                                            \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err));fflush(stdout); \
        return false;                                                        \
    }                                                                        \
}
    __device__ double Matern_GPU_case(double var, double range_dist, int shape) {
        switch (shape) {
        case 5:  return var * exp(-range_dist);
        case 15: return var * (1. + range_dist) * exp(-range_dist);
        case 25: return var * (1. + range_dist + range_dist * range_dist / 3.) * exp(-range_dist);
        default: return 0.0;
        }
    }


    // Device function: compute distance
    __device__ __forceinline__ double distances_funct_device(
        double corr_diag_i,            // index i
        double corr_diag_j,    // indices j
        int num_ip,                      // number of inducing points
        int dim_coords,             // coordinate dimension
        const double* __restrict__ col_i,      // precomputed
        const double* __restrict__ col_j,      // precomputed
        const double* __restrict__ coord_i_ptr,      // precomputed
        const double* __restrict__ coord_j_ptr,      // precomputed
        const double var,
        const int shape,
        const double range,
        double EPSILON_NUMBERS
    ) {
        // Step 1: dot product
        double dot = 0.0;
        for (int d = 0; d < num_ip; d++) {
            dot = fma(col_j[d], col_i[d], dot);
        }
        // Step 2: Euclidean distance
        double sum = 0.0;
        for (int d = 0; d < dim_coords; d++) {
            double diff = coord_j_ptr[d] - coord_i_ptr[d];
            sum = fma(diff, diff, sum);
        }
        double range_dist = range * sqrt(sum);
        double cov = Matern_GPU_case(var, range_dist, shape);
        double num = cov - dot;
        return  corr_diag_i * corr_diag_j / (num * num);
    }


    // Brute-force kNN kernel -----------------
    __global__ void knn_bruteforce_kernel(
        int n, int d, int k,
        const double* coords,              // [n * d], row-major
        const double* corr_diag,           // [n]
        const double* chol_ip_cross_cov,   // [num_ip * n]
        const double* __restrict__ pars,      // precomputed
        int num_ip,
        const int shape,
        const double range,
        double EPSILON_NUMBERS,
        int dist_funct,
        int* knn_idx,   // [n * k], output
        int start_at,
        int end_at,
        int start_dim
    ) {

        if (k > MAX_K) return;

        int i = blockIdx.x + start_at;   // one block per query point
        if (i >= n) return;

        int tid = threadIdx.x;

        extern __shared__ double shmem[];
        double* dist_buf = shmem;          // [blockDim.x * k]
        int* idx_buf = (int*)&dist_buf[blockDim.x * k];
        // local top-k buffers
        double local_dist[MAX_K];
        int local_idx[MAX_K];
        for (int kk = 0; kk < k; kk++) {
            local_dist[kk] = CUDART_INF;
            local_idx[kk] = -1;
        }
        const double* __restrict__ col_i = chol_ip_cross_cov + i * num_ip;
        const double* __restrict__ coord_i_ptr = coords + i * d;
        double corr_diag_i;
        if (dist_funct != 3) {
            corr_diag_i = corr_diag[i];
        }
        // each thread checks candidates j < i
        int end_at_i = min(i, end_at);
        for (int j = tid; j < end_at_i; j += blockDim.x) {
            const double* __restrict__ col_j = chol_ip_cross_cov + j * num_ip;
            const double* __restrict__ coord_j_ptr = coords + j * d;
            double sum = 0.0;
            for (int dd = start_dim; dd < d; dd++) {
                double diff = coord_j_ptr[dd] - coord_i_ptr[dd];
                sum = fma(diff, diff, sum);
            }
            double inv_r = rsqrt(sum);
            double range_dist = 0.;
            double var;
            if (dist_funct != 3) {
                var = pars[0];
            }
            double dot = 0.0;
            for (int dd = 0; dd < num_ip; dd++) {
                dot = fma(col_j[dd], col_i[dd], dot);
            }
            if (dist_funct == 1) {
                range_dist = range / inv_r;
            }
            else if (dist_funct == 2) {
                double dt = fabs(coord_i_ptr[0] - coord_j_ptr[0]);
                double d_aux_time_log = (dt < EPSILON_NUMBERS) ? 0. : log(pars[1] * exp(log(dt) * 2 * pars[3]) + 1.);
                range_dist = pars[2] / (exp(d_aux_time_log * pars[5] * 0.5) * inv_r);
                var = pars[0] / (exp(d_aux_time_log * (pars[6] + pars[5] * (d - 1) * 0.5)));
            }
            double dij;
            if (dist_funct == 3) {
                dij = sum;
            }
            else {
                double cov = Matern_GPU_case(var, range_dist, shape);

                double num = cov - dot;
                dij = corr_diag_i * corr_diag[j] / (num * num);
            }
            // insert into local top-k
            int worst = 0;
            for (int kk = 1; kk < k; kk++) {
                if (local_dist[kk] > local_dist[worst]) worst = kk;
            }
            if (dij < local_dist[worst]) {
                local_dist[worst] = dij;
                local_idx[worst] = j;
            }
        }
        // write local results to shared memory
        for (int kk = 0; kk < k; kk++) {
            dist_buf[tid * k + kk] = local_dist[kk];
            idx_buf[tid * k + kk] = local_idx[kk];
        }
        __syncthreads();

        // block reduction: keep only best k
        if (tid == 0) {
            double final_dist[MAX_K];
            int final_idx[MAX_K];
            for (int kk = 0; kk < k; kk++) {
                final_dist[kk] = CUDART_INF;
                final_idx[kk] = -1;
            }

            int total = blockDim.x * k;
            for (int t = 0; t < total; t++) {
                double dval = dist_buf[t];
                int jval = idx_buf[t];
                if (jval < 0) continue;

                int worst = 0;
                for (int kk = 1; kk < k; kk++) {
                    if (final_dist[kk] > final_dist[worst]) worst = kk;
                }
                if (dval < final_dist[worst]) {
                    final_dist[worst] = dval;
                    final_idx[worst] = jval;
                }
            }

            // insertion sort: sort results ascending (closest first)
            for (int a = 1; a < k; a++) {
                double key_dist = final_dist[a];
                int key_idx = final_idx[a];
                int b = a - 1;
                while (b >= 0 && final_dist[b] > key_dist) {
                    final_dist[b + 1] = final_dist[b];
                    final_idx[b + 1] = final_idx[b];
                    b--;
                }
                final_dist[b + 1] = key_dist;
                final_idx[b + 1] = key_idx;
            }

            // write out
            for (int kk = 0; kk < k; kk++) {
                knn_idx[(i - start_at) * k + kk] = final_idx[kk];
            }
        }
    }

    bool find_nearest_neighbors_bruteforce_GPU(
        const den_mat_t& coords,
        int num_data,
        int num_neighbors,
        const vec_t& pars,
        int start_at,
        int brute_force_threshold,
        int end_at,
        int dim_coords,
        const vec_t& corr_diag,
        const den_mat_t& chol_ip_cross_cov,
        const int shape,
        const double range,
        double EPSILON_NUMBERS,
        int dist_funct,
        std::vector<std::vector<int>>& neighbors,
        int start_dim
    ) {
        if (num_neighbors > MAX_K) return false;
        // --- prepare sizes ---
        int total_threads = num_data - brute_force_threshold;

        // --- allocate device memory ---
        double* d_coords = nullptr;
        double* d_corr_diag = nullptr;
        double* d_chol_ip_cross_cov = nullptr;
        double* d_pars = nullptr;
        int* d_neighbors = nullptr;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coords_row = coords;

        CUDA_CHECK(cudaMalloc(&d_coords, coords_row.size() * sizeof(double)));
        if (dist_funct != 3) {
            CUDA_CHECK(cudaMalloc(&d_corr_diag, corr_diag.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_pars, pars.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_chol_ip_cross_cov, chol_ip_cross_cov.size() * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&d_neighbors, total_threads * num_neighbors * sizeof(int)));

        // --- copy data to device ---
        CUDA_CHECK(cudaMemcpy(d_coords, coords_row.data(), coords_row.size() * sizeof(double), cudaMemcpyHostToDevice));
        if (dist_funct != 3) {
            CUDA_CHECK(cudaMemcpy(d_corr_diag, corr_diag.data(), corr_diag.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pars, pars.data(), pars.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_chol_ip_cross_cov, chol_ip_cross_cov.data(), chol_ip_cross_cov.size() * sizeof(double), cudaMemcpyHostToDevice));
        }
        // --- launch kernel ---
        //int threads = 128;
        //int blocks = total_threads;   // one block per query point
        //size_t shmem_size = threads * num_neighbors * (sizeof(double) + sizeof(int));

        int blocks = total_threads;   // one block per query point

        // --- adapt threads at runtime so launches are valid on any GPU ---
        cudaDeviceProp prop;
        int dev;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);

        int maxThreadsPerBlock = prop.maxThreadsPerBlock;
        size_t maxSharedPerBlock = prop.sharedMemPerBlock;

        // start from the device maximum rounded down to a warp multiple
        int threads = (maxThreadsPerBlock / 32) * 32;
        if (threads < 32) threads = 32;

        // helper to compute shared mem required by a block with `t` threads
        auto shmemNeededForThreads = [&](int t) -> size_t {
            return (size_t)t * num_neighbors * (sizeof(double) + sizeof(int));
            };

        // reduce threads (preserving warp multiples) while shared mem would exceed device limit
        while (threads > 32 && shmemNeededForThreads(threads) > maxSharedPerBlock) {
            threads /= 2; // still a multiple of 32 because we started from a warp-multiple
        }

        // final safety check: even one warp must fit in shared memory
        size_t shmem_size = shmemNeededForThreads(threads);

        Log::REDebug("Launch %i %i %i", blocks, threads, shmem_size);
        knn_bruteforce_kernel << <blocks, threads, shmem_size >> > (
            num_data, dim_coords, num_neighbors,
            d_coords, d_corr_diag, d_chol_ip_cross_cov, d_pars,
            (int)chol_ip_cross_cov.rows(), shape,
            range, EPSILON_NUMBERS, dist_funct,
            d_neighbors, brute_force_threshold, end_at, start_dim
            );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            fprintf(stderr, "kNN kernel launch failed: %s\n", cudaGetErrorString(launchErr)); fflush(stdout);
            return false;
        }
        cudaError_t execErr = cudaDeviceSynchronize();
        if (execErr != cudaSuccess) {
            fprintf(stderr, "kNN kernel execution failed: %s\n", cudaGetErrorString(execErr)); fflush(stdout);
            return false;
        }
        // --- copy back results ---
        std::vector<int> h_neighbors(total_threads * num_neighbors);

        CUDA_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, h_neighbors.size() * sizeof(int), cudaMemcpyDeviceToHost));
        // --- fill results ---
#pragma omp parallel for schedule(static)
        for (int i = brute_force_threshold; i < num_data; i++) {
            for (int j = 0; j < num_neighbors; j++) {
                neighbors[i - start_at][j] = h_neighbors[(i - brute_force_threshold) * num_neighbors + j];
            }
        }
        // --- cleanup ---
        cudaFree(d_coords);
        if (dist_funct != 3) {
            cudaFree(d_corr_diag);
            cudaFree(d_pars);
            cudaFree(d_chol_ip_cross_cov);
        }
        cudaFree(d_neighbors);
        return true;
    }

    __device__ void SortVectorsDecreasing_GPU(double* a, int* b, int n) {
        int j, k, l;
        double v;
        for (j = 1; j <= n - 1; j++) {
            k = j;
            while (k > 0 && a[k] < a[k - 1]) {  // decreasing order!
                v = a[k];
                l = b[k];
                a[k] = a[k - 1];
                b[k] = b[k - 1];
                a[k - 1] = v;
                b[k - 1] = l;
                k--;
            }
        }
    }

    __device__ void find_nearest_neighbors_fast_internal_GPU(
        const int i,
        const int num_data,
        const int num_neighbors,
        const int end_search_at,
        const int dim_coords,
        const double* coords,          // [num_data * dim_coords], row-major
        const int* sort_sum,           // [num_data]
        const int* sort_inv_sum,       // [num_data]
        const double* coords_sum,      // [num_data]
        int* neighbors_i,              // [num_neighbors], output
        double* nn_square_dist         // [num_neighbors], output
    ) {

        bool down = true;
        bool up = true;
        int up_i = sort_inv_sum[i];
        int down_i = sort_inv_sum[i];

        double smd, sed;
        while (up || down) {
            if (down_i == 0) { down = false; }
            if (up_i == (num_data - 1)) { up = false; }

            if (down) {
                down_i--;
                int cand = sort_sum[down_i];
                if (cand < i && cand <= end_search_at) {
                    smd = (coords_sum[cand] - coords_sum[i]) * (coords_sum[cand] - coords_sum[i]);
                    if (smd > dim_coords * nn_square_dist[num_neighbors - 1]) {
                        down = false;
                    }
                    else {
                        // squared Euclidean distance
                        sed = 0.0;
                        for (int d = 0; d < dim_coords; d++) {
                            double diff = coords[cand * dim_coords + d] - coords[i * dim_coords + d];
                            sed += diff * diff;
                        }
                        if (sed < nn_square_dist[num_neighbors - 1]) {
                            nn_square_dist[num_neighbors - 1] = sed;
                            neighbors_i[num_neighbors - 1] = cand;
                            SortVectorsDecreasing_GPU(nn_square_dist, neighbors_i, num_neighbors);
                        }
                    }
                }
            }
            if (up) {
                up_i++;
                int cand = sort_sum[up_i];
                if (cand < i && cand <= end_search_at) {
                    smd = (coords_sum[cand] - coords_sum[i]) * (coords_sum[cand] - coords_sum[i]);
                    if (smd > dim_coords * nn_square_dist[num_neighbors - 1]) {
                        up = false;
                    }
                    else {
                        // squared Euclidean distance
                        sed = 0.0;
                        for (int d = 0; d < dim_coords; d++) {
                            double diff = coords[cand * dim_coords + d] - coords[i * dim_coords + d];
                            sed += diff * diff;
                        }
                        if (sed < nn_square_dist[num_neighbors - 1]) {
                            nn_square_dist[num_neighbors - 1] = sed;
                            neighbors_i[num_neighbors - 1] = cand;
                            SortVectorsDecreasing_GPU(nn_square_dist, neighbors_i, num_neighbors);
                        }
                    }
                }
            }
        }
    }

    // Kernel
    __global__ void find_neighbors_kernel(
        int first_i,
        int num_data,
        int num_neighbors,
        int num_close_neighbors,
        int start_at,
        int end_search_at,
        int dim_coords,
        const double* coords,         // [num_data * dim_coords]
        const int* sort_sum,          // [num_data]
        const int* sort_inv_sum,      // [num_data]
        const double* coords_sum,     // [num_data]
        int* neighbors,               // [(num_data - first_i) * num_neighbors]
        double* dist_obs_neighbors,   // same shape (optional)
        bool save_distances,
        bool check_has_duplicates,
        int* has_duplicates_flag     // global flag (0 or 1)
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int i = first_i + tid;
        if (i >= num_data) return;
        // output pointers for this thread
        int* neighbors_i = &neighbors[(i - first_i) * num_neighbors];
        double* dist_i = nullptr;
        if (save_distances) {
            dist_i = &dist_obs_neighbors[(i - first_i) * num_neighbors];
        }
        double nn_square_dist[MAX_K];

        // sanity checks
        if (num_neighbors > MAX_K) {
            // out-of-bounds risk, just bail
            return;
        }
        // initialize nearest
        for (int j = 0; j < num_neighbors; j++) {
            nn_square_dist[j] = CUDART_INF;
            neighbors_i[j] = -1;
        }
        find_nearest_neighbors_fast_internal_GPU(
            i, num_data, num_neighbors, end_search_at,
            dim_coords, coords, sort_sum, sort_inv_sum, coords_sum,
            neighbors_i, nn_square_dist
        );
        // --- distances & duplicates ---
        if (save_distances || (check_has_duplicates && (*has_duplicates_flag == 0))) {
            for (int j = 0; j < num_neighbors; j++) {
                double dij = sqrt(nn_square_dist[j]);
                if (save_distances) dist_i[j] = dij;
                if (check_has_duplicates && (*has_duplicates_flag == 0) && dij < 1e-12) {
                    atomicExch(has_duplicates_flag, 1);
                }
            }
        }
    }

    bool find_nearest_neighbors_Vecchia_fast_GPU(
        const den_mat_t& coords,
        int num_data,
        int num_neighbors,
        int num_close_neighbors,
        int start_at,
        int end_search_at,
        int dim_coords,
        const std::vector<int>& sort_sum,
        const std::vector<int>& sort_inv_sum,
        const std::vector<double>& coords_sum,
        std::vector<std::vector<int>>& neighbors,
        std::vector<den_mat_t>& dist_obs_neighbors,
        bool save_distances,
        bool& has_duplicates,
        bool check_has_duplicates
    ) {
        int first_i = (start_at <= num_neighbors) ? (num_neighbors + 1) : start_at;
        int total_threads = num_data - first_i;
        // --- allocate device memory ---
        double* d_coords = nullptr;
        int* d_sort_sum = nullptr;
        int* d_sort_inv_sum = nullptr;
        double* d_coords_sum = nullptr;
        int* d_neighbors = nullptr;
        double* d_dist_obs_neighbors = nullptr;
        int* d_has_duplicates = nullptr;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coords_row = coords;
        CUDA_CHECK(cudaMalloc(&d_coords, coords_row.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sort_sum, num_data * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sort_inv_sum, num_data * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_coords_sum, num_data * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighbors, total_threads * num_neighbors * sizeof(int)));
        if (save_distances) {
            CUDA_CHECK(cudaMalloc(&d_dist_obs_neighbors, total_threads * num_neighbors * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&d_has_duplicates, sizeof(int)));
        // --- copy host data to device ---
        CUDA_CHECK(cudaMemcpy(d_coords, coords_row.data(), num_data * dim_coords * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sort_sum, sort_sum.data(), num_data * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sort_inv_sum, sort_inv_sum.data(), num_data * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coords_sum, coords_sum.data(), num_data * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_has_duplicates, 0, sizeof(int)));
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        printf("Launching kernel with %d blocks, %d threads (n=%d)\n",
            threads, blocks, total_threads);
        fflush(stdout);
        // --- run neighbor kernel ---
        find_neighbors_kernel << <blocks, threads >> > (
            first_i,
            num_data,
            num_neighbors,
            num_close_neighbors,
            start_at,
            end_search_at,
            dim_coords,
            d_coords,
            d_sort_sum,
            d_sort_inv_sum,
            d_coords_sum,
            d_neighbors,
            d_dist_obs_neighbors,
            save_distances,
            check_has_duplicates,
            d_has_duplicates
            );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            fprintf(stderr, "Neighbor kernel launch failed: %s\n", cudaGetErrorString(launchErr)); fflush(stdout);
            return false;
        }
        cudaError_t execErr = cudaDeviceSynchronize();
        if (execErr != cudaSuccess) {
            fprintf(stderr, "Neighbor kernel execution failed: %s\n", cudaGetErrorString(execErr)); fflush(stdout);
            return false;
        }

        // --- copy back results ---
        std::vector<int> h_neighbors(total_threads * num_neighbors);
        CUDA_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, h_neighbors.size() * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<double> h_dist;
        if (save_distances) {
            h_dist.resize(total_threads * num_neighbors);
            CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist_obs_neighbors, h_dist.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        int h_has_duplicates = 0;
        if (check_has_duplicates) {
            CUDA_CHECK(cudaMemcpy(&h_has_duplicates, d_has_duplicates, sizeof(int), cudaMemcpyDeviceToHost));
            has_duplicates = (h_has_duplicates == 1);
        }

        // --- fill into neighbors/dist_obs_neighbors ---
        for (int i = first_i; i < num_data; i++) {
            for (int j = 0; j < num_neighbors; j++) {
                neighbors[i - start_at][j] = h_neighbors[(i - first_i) * num_neighbors + j];
            }
            if (save_distances) {
                dist_obs_neighbors[i - start_at].resize(num_neighbors, 1);
                for (int j = 0; j < num_neighbors; j++) {
                    dist_obs_neighbors[i - start_at](j, 0) =
                        h_dist[(i - first_i) * num_neighbors + j];
                }
            }
        }
        // --- cleanup ---
        cudaFree(d_coords);
        cudaFree(d_sort_sum);
        cudaFree(d_sort_inv_sum);
        cudaFree(d_coords_sum);
        cudaFree(d_neighbors);
        if (save_distances) cudaFree(d_dist_obs_neighbors);
        cudaFree(d_has_duplicates);
        return true;
    }

    bool try_matmul_gpu(const den_mat_t& A, const den_mat_t& B, den_mat_t& C) {
        int M = A.rows(), K = A.cols(), N = B.cols();
        if (K != B.rows()) {
            return false;
        }

        C.resize(M, N);

        const double* h_A = A.data();
        const double* h_B = B.data();
        double* h_C = C.data();

        double* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
        cudaError_t cuda_stat;
        cublasStatus_t stat;
        cublasHandle_t handle;

        size_t size_A = M * K * sizeof(double);
        size_t size_B = K * N * sizeof(double);
        size_t size_C = M * N * sizeof(double);

        cuda_stat = cudaMalloc((void**)&d_A, size_A);
        if (cuda_stat != cudaSuccess) return false;
        cuda_stat = cudaMalloc((void**)&d_B, size_B);
        if (cuda_stat != cudaSuccess) {
            cudaFree(d_A);
            return false;
        }

        cuda_stat = cudaMalloc((void**)&d_C, size_C);
        if (cuda_stat != cudaSuccess) {
            cudaFree(d_A); cudaFree(d_B);
            return false;
        }

        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return false;
        }

        const double alpha = 1.0;
        const double beta = 0.0;

        // cuBLAS performs: C = alpha * op(A) * op(B) + beta * C
        // We want: C = A * B
        // A: MxK, B: KxN, C: MxN
        // So op(A) = A, op(B) = B
        stat = cublasDgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
            M, N, K,                   // C is MxN, A is MxK, B is KxN
            &alpha,
            d_A, M,  // lda = leading dim of A = M (since column-major)
            d_B, K,  // ldb = leading dim of B = K
            &beta,
            d_C, M); // ldc = leading dim of C = M

        if (stat != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return false;
        }

        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

        cublasDestroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        //Log::REInfo("[GPU] Matrix multiplication completed with cuBLAS.");
        return true;
    }

    bool try_spmatmul_gpu(const sp_mat_rm_t& A, const sp_mat_rm_t& B, sp_mat_rm_t& C) {
        if (A.cols() != B.rows()) return false;

        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matA = nullptr, matB = nullptr, matC = nullptr;
        cusparseSpGEMMDescr_t spgemmDescr = nullptr;

        int m = A.rows(), k = A.cols(), n = B.cols();
        int A_nnz = A.nonZeros(), B_nnz = B.nonZeros();

        int* d_A_rowPtr = nullptr, * d_A_colInd = nullptr;
        double* d_A_values = nullptr;
        int* d_B_rowPtr = nullptr, * d_B_colInd = nullptr;
        double* d_B_values = nullptr;
        int* d_C_rowPtr = nullptr, * d_C_colInd = nullptr;
        double* d_C_values = nullptr;
        void* dBuffer1 = nullptr, * dBuffer2 = nullptr;

        // Allocate device memory for A
        cudaMalloc(&d_A_rowPtr, (m + 1) * sizeof(int));
        cudaMalloc(&d_A_colInd, A_nnz * sizeof(int));
        cudaMalloc(&d_A_values, A_nnz * sizeof(double));

        // Allocate device memory for B
        cudaMalloc(&d_B_rowPtr, (k + 1) * sizeof(int));
        cudaMalloc(&d_B_colInd, B_nnz * sizeof(int));
        cudaMalloc(&d_B_values, B_nnz * sizeof(double));

        // Copy A and B to device
        cudaMemcpy(d_A_rowPtr, A.outerIndexPtr(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_colInd, A.innerIndexPtr(), A_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_values, A.valuePtr(), A_nnz * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_B_rowPtr, B.outerIndexPtr(), (k + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_colInd, B.innerIndexPtr(), B_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_values, B.valuePtr(), B_nnz * sizeof(double), cudaMemcpyHostToDevice);

        // cuSPARSE setup
        cusparseCreate(&handle);
        //cusparseCreateSpGEMMDescr(&spgemmDesc);
        cusparseSpGEMM_createDescr(&spgemmDescr);
        cusparseCreateCsr(&matA, m, k, A_nnz, d_A_rowPtr, d_A_colInd, d_A_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateCsr(&matB, k, n, B_nnz, d_B_rowPtr, d_B_colInd, d_B_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateCsr(&matC, m, n, 0, nullptr, nullptr, nullptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        double alpha = 1.0, beta = 0.0;
        size_t bufferSize1 = 0, bufferSize2 = 0;

        // Phase 1: Work estimation
        cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, nullptr);
        cudaMalloc(&dBuffer1, bufferSize1);
        cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, dBuffer1);

        // Phase 2: Compute
        cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, nullptr);
        cudaMalloc(&dBuffer2, bufferSize2);
        cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, dBuffer2);

        // Phase 3: Copy to finalize matC
        int64_t C_num_rows, C_num_cols, C_nnz;
        cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz);
        cudaMalloc(&d_C_rowPtr, (m + 1) * sizeof(int));
        cudaMalloc(&d_C_colInd, C_nnz * sizeof(int));
        cudaMalloc(&d_C_values, C_nnz * sizeof(double));

        cusparseCsrSetPointers(matC, d_C_rowPtr, d_C_colInd, d_C_values);
        cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr);

        // Copy result to host
        std::vector<int> h_C_rowPtr(m + 1);
        std::vector<int> h_C_colInd(C_nnz);
        std::vector<double> h_C_values(C_nnz);

        cudaMemcpy(h_C_rowPtr.data(), d_C_rowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_colInd.data(), d_C_colInd, C_nnz * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_values.data(), d_C_values, C_nnz * sizeof(double), cudaMemcpyDeviceToHost);

        // Build result Eigen matrix
        C.resize(m, n);
        C.makeCompressed();
        C.reserve(C_nnz);
        std::copy(h_C_rowPtr.begin(), h_C_rowPtr.end(), C.outerIndexPtr());
        std::copy(h_C_colInd.begin(), h_C_colInd.end(), C.innerIndexPtr());
        std::copy(h_C_values.begin(), h_C_values.end(), C.valuePtr());

        // Cleanup
        cudaFree(d_A_rowPtr); cudaFree(d_A_colInd); cudaFree(d_A_values);
        cudaFree(d_B_rowPtr); cudaFree(d_B_colInd); cudaFree(d_B_values);
        cudaFree(d_C_rowPtr); cudaFree(d_C_colInd); cudaFree(d_C_values);
        cudaFree(dBuffer1); cudaFree(dBuffer2);
        cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
        //cusparseDestroySpGEMMDescr(spgemmDesc);
        cusparseSpGEMM_destroyDescr(spgemmDescr);
        cusparseDestroy(handle);

        return true;
    }

    bool try_solve_lower_triangular_gpu(const chol_den_mat_t& chol, const den_mat_t& R_host, den_mat_t& X_host) {
        den_mat_t L_host = chol.matrixL();
        int n = L_host.rows();
        int m = R_host.cols();
        if (L_host.cols() != n || R_host.rows() != n) {
            return false;
        }
        X_host.resize(n, m);
        // Allocate device memory
        double* d_L = nullptr;
        double* d_X = nullptr;

        cudaMalloc(&d_L, n * n * sizeof(double));
        cudaMalloc(&d_X, n * m * sizeof(double));

        cudaMemcpy(d_L, L_host.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, R_host.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_L); cudaFree(d_X);
            return false;
        }
        const double alpha = 1.0;

        // Solve: L * X = R -> X = L^{-1} * R
        // L is lower-triangular, column-major
        // Left-side, lower-triangular, no transpose, non-unit diagonal
        stat = cublasDtrsm(
            handle,
            CUBLAS_SIDE_LEFT,      // Solve L * X = R
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,           // No transpose
            CUBLAS_DIAG_NON_UNIT,  // Assume general diagonal
            n,                     // number of rows of L and X
            m,                     // number of columns of X
            &alpha,                // Scalar alpha
            d_L, n,                // L, leading dimension n
            d_X, n                 // R becomes X, leading dimension n
        );

        if (stat != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_L); cudaFree(d_X);
            cublasDestroy(handle);
            return false;
        }

        // Copy result back
        cudaMemcpy(X_host.data(), d_X, n * m * sizeof(double), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_L);
        cudaFree(d_X);
        cublasDestroy(handle);

        //Log::REInfo("[GPU] Triangular solve with CUBLAS.");
        return true;
    }

    bool try_solve_cholesky_gpu(const chol_den_mat_t& chol, const den_mat_t& R_host, den_mat_t& X_host) {
        den_mat_t L_host = chol.matrixL();  // L from LL^T
        int n = L_host.rows();
        int m = R_host.cols();

        if (L_host.cols() != n || R_host.rows() != n) {
            return false;
        }
        X_host.resize(n, m);
        // Allocate memory
        double* d_L = nullptr;
        double* d_Y = nullptr;
        double* d_X = nullptr;

        cudaMalloc(&d_L, n * n * sizeof(double));
        cudaMalloc(&d_Y, n * m * sizeof(double));
        cudaMalloc(&d_X, n * m * sizeof(double));

        cudaMemcpy(d_L, L_host.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, R_host.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);  // Start Y = R

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        const double alpha = 1.0;

        // Step 1: Solve L * Y = R
        cublasStatus_t stat1 = cublasDtrsm(
            handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            n, m,
            &alpha,
            d_L, n,
            d_Y, n  // In-place
        );

        if (stat1 != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_L); cudaFree(d_Y); cudaFree(d_X);
            cublasDestroy(handle);
            return false;
        }

        // Step 2: Solve L^T * X = Y
        cudaMemcpy(d_X, d_Y, n * m * sizeof(double), cudaMemcpyDeviceToDevice);  // Copy Y into X

        cublasStatus_t stat2 = cublasDtrsm(
            handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,  // Transpose
            CUBLAS_DIAG_NON_UNIT,
            n, m,
            &alpha,
            d_L, n,
            d_X, n  // In-place
        );

        if (stat2 != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_L); cudaFree(d_Y); cudaFree(d_X);
            cublasDestroy(handle);
            return false;
        }

        // Copy result back
        X_host.resize(n, m);
        cudaMemcpy(X_host.data(), d_X, n * m * sizeof(double), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_L);
        cudaFree(d_Y);
        cudaFree(d_X);
        cublasDestroy(handle);

        //Log::REInfo("[GPU] Full Cholesky solve (Sigma^-1 * R) with cuBLAS.");
        return true;
    }

}  // namespace GPBoost

#endif  // USE_CUDA_GP
