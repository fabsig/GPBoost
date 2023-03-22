/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 Pascal Kuendig and Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/CG_utils.h>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <LightGBM/utils/log.h>

#include <chrono>
#include <thread> //temp
//#include <functional>
//#include <iostream>
//using namespace std;
using LightGBM::Log;

namespace GPBoost {

	void CGVecchiaLaplaceVec(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NaN_found,
		int p,
		const bool warm_start,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const string_t cg_preconditioner_type,
		const chol_sp_mat_rm_t& chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia,
		const chol_sp_mat_rm_t& chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia,
		const sp_mat_rm_t& B_k_rm,
		const sp_mat_rm_t& L_k_rm,
		const sp_mat_rm_t& D_inv_plus_W_B_rm) {

		p = std::min(p, (int)B_rm.cols());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h;
		vec_t v;
		vec_t B_invt_r, L_kt_W_inv_r, B_k_W_inv_r, W_inv_r, diag_SigmaI_plus_W_inv, diag_W_inv;
		//sp_mat_rm_t W_inv_rm, D_inv_plus_W_B_rm;
		bool early_stop_alg = false;
		double a = 0;
		double b = 1;
		double r_squared_norm;

		//std::chrono::steady_clock::time_point begin, end;
		//double el_time;
		//begin = std::chrono::steady_clock::now();
		
		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			//Log::REInfo("0 - return 0");
			u.setZero();
			return;
		}

		if (!warm_start || find_mode_it == 0) {
			Log::REInfo("DOING COLD START");
			u.setZero(); //No warm-start
		}
		else
		{
			Log::REInfo("DOING WARM START");
		}

		//B_t_D_inv_rm = B_rm.transpose() * D_inv_rm;
		r = rhs - ((B_t_D_inv_rm * (B_rm * u)) + diag_W.cwiseProduct(u)); //r = rhs - A * u

		//z = P^(-1) r 
		if (cg_preconditioner_type == "piv_chol_veccia") {
			//Pivoted Cholseky - Version Veccia
			diag_W_inv = diag_W.cwiseInverse();
			//P^(-1) = (B_k^T D_k^(-1) B_k + W)^(-1) = W^(-1) - W^(-1) B_k^T (D_k + B_k W^(-1) B_k^T)^(-1) B_k W^(-1)
			W_inv_r = diag_W_inv.asDiagonal() * r;
			B_k_W_inv_r = B_k_rm * W_inv_r;
			if (B_k_rm.nonZeros() <= r.size()) {
				//Relevant Version, since B_k_rm.nonZeros() = NN * k 
				z = W_inv_r - (diag_W_inv.asDiagonal() * B_k_rm.transpose()) * chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia.solve(B_k_W_inv_r);
			}
			else {
				z = W_inv_r - diag_W_inv.asDiagonal() * (B_k_rm.transpose() * chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia.solve(B_k_W_inv_r));
			}
		}
		else if (cg_preconditioner_type == "piv_chol_habrecht") {
			//Pivoted Cholseky - Version Habrecht 
			diag_W_inv = diag_W.cwiseInverse();
			//P^(-1) = (L_k L_k^T + W)^(-1) = W^(-1) - W^(-1) L_k (I_k + L_k^T W^(-1) L_k)^(-1) L_k^T W^(-1)
			W_inv_r = diag_W_inv.asDiagonal() * r;
			L_kt_W_inv_r = ((sp_mat_rm_t)L_k_rm.transpose()) * W_inv_r;
			if (L_k_rm.nonZeros() <= r.size()) {
				z = W_inv_r - (diag_W_inv.asDiagonal() * L_k_rm) * chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia.solve(L_kt_W_inv_r);
			}
			else {
				z = W_inv_r - diag_W_inv.asDiagonal() * (L_k_rm * chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia.solve(L_kt_W_inv_r));
			}
		}
		else if (cg_preconditioner_type == "sigma") {
			//P^(-1) = \Sigma = B^(-1)DB^(-T)
			//https://stackoverflow.com/questions/24442850/solving-a-sparse-upper-triangular-system-in-eigen
			B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r); //B^(-T)r
			z = B_t_D_inv_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);
		}
		else if (cg_preconditioner_type == "sigma_with_W") {
			//P^(-1) = B^(-1) (D^(-1)+W)^(-1) B^(-T)
			//D_inv_plus_W_B_rm = (D_inv_rm.diagonal() + diag_W).asDiagonal() * B_rm;
			B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r); //B^(-T)r
			z = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);
		}
		else if (cg_preconditioner_type == "diagonal") {
			//P^(-1) = diag(diag(\Sigma^-1 + W)^(-1))
			diag_SigmaI_plus_W_inv = B_t_D_inv_rm.cwiseProduct(((sp_mat_rm_t)B_rm.transpose())) * vec_t::Ones(B_rm.rows()) + diag_W;
			diag_SigmaI_plus_W_inv = diag_SigmaI_plus_W_inv.cwiseInverse();
			z = diag_SigmaI_plus_W_inv.array() * r.array();
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}

		h = z;

		//end = std::chrono::steady_clock::now();
		//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
		//Log::REInfo("Time overhead: %g", el_time);
		for (int j = 0; j < p; ++j) {
			//The following matrix-vector operation is the expensive part of the loop
			//Parentheses are necessery for performance, otherwise EIGEN does the operation wrongly from left to right

			//begin = std::chrono::steady_clock::now();
			v = (B_t_D_inv_rm * (B_rm * h)) + diag_W.cwiseProduct(h);
			//end = std::chrono::steady_clock::now();
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("CGVecchiaLaplaceVec MV %i: %g", j, el_time);
			//begin = std::chrono::steady_clock::now(); //NEW
			
			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_squared_norm = r.squaredNorm();
			//Log::REInfo("r.squaredNorm(): %g | Iteration: %i", r_squared_norm, j);
			if (std::isnan(r_squared_norm) || std::isinf(r_squared_norm)) {
				NaN_found = true;
				return;
			}
			if (r_squared_norm < delta_conv) {
				early_stop_alg = true;
			}

			z_old = z;

			//end = std::chrono::steady_clock::now(); //NEW
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("In between: %g", el_time);
			//begin = std::chrono::steady_clock::now();
			//z = P^(-1) r 
			if (cg_preconditioner_type == "piv_chol_veccia") {
				//Pivoted Cholseky - Version Veccia
				W_inv_r = diag_W_inv.asDiagonal() * r;
				B_k_W_inv_r = B_k_rm * W_inv_r;
				if (B_k_rm.nonZeros() <= r.size()) {
					//Relevant Version, since B_k_rm.nonZeros() = NN * k 
					z = W_inv_r - (diag_W_inv.asDiagonal() * B_k_rm.transpose()) * chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia.solve(B_k_W_inv_r);
				}
				else {
					z = W_inv_r - diag_W_inv.asDiagonal() * (B_k_rm.transpose() * chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia.solve(B_k_W_inv_r));
				}
			}
			else if (cg_preconditioner_type == "piv_chol_habrecht") {
				//Pivoted Cholseky - Version Habrecht 
				W_inv_r = diag_W_inv.asDiagonal() * r;
				L_kt_W_inv_r = ((sp_mat_rm_t)L_k_rm.transpose()) * W_inv_r;
				if (L_k_rm.nonZeros() <= r.size()) {
					z = W_inv_r - (diag_W_inv.asDiagonal() * L_k_rm) * chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia.solve(L_kt_W_inv_r);
				}
				else {
					z = W_inv_r - diag_W_inv.asDiagonal() * (L_k_rm * chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia.solve(L_kt_W_inv_r));
				}
			}
			else if (cg_preconditioner_type == "sigma") {
				//P^(-1) = \Sigma
				B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r); //B^(-T)r
				z = B_t_D_inv_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);
			}
			else if (cg_preconditioner_type == "sigma_with_W") {
				//P^(-1) = B^(-1) (D^(-1)+W)^(-1) B^(-T)
				B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r); //B^(-T)r
				z = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);
			}
			else if (cg_preconditioner_type == "diagonal") {
				//P^(-1) = diag(diag(\Sigma^-1 + W)^(-1))
				z = diag_SigmaI_plus_W_inv.array() * r.array();
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}
			//end = std::chrono::steady_clock::now(); //NEW
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("P^(-1) r: %g", el_time);
			//begin = std::chrono::steady_clock::now();

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;

			if (early_stop_alg) {
				//Log::REInfo("CGVecchiaLaplaceVec stop after %i CG-Iterations.", j);
				//Log::REInfo("Mode"); //temp
				//for (int i = 0; i < 10; ++i) { //temp
				//    Log::REInfo("mode_[%d]: %g", i, u[i]);//temp
				//}
				//std::this_thread::sleep_for(dura); //temp
				return;
			}
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.; //NEW
			//Log::REInfo("Rest: %g", el_time);
			//begin = std::chrono::steady_clock::now();
		}
		Log::REInfo("CGVecchiaLaplaceVec used all %i iterations!", p);
		Log::REInfo("final r.squaredNorm(): %g", r_squared_norm);
		//NaN_found = true; //Hack to end mode-finding if no convergence of the conjugate gradient algorithm
	} // end CGVecchiaLaplaceVec

	void CGVecchiaLaplaceVecSigmaplusWinv(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NaN_found,
		int p,
		const bool warm_start,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const chol_den_mat_t chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k) {

		p = std::min(p, (int)B_rm.cols());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h;
		vec_t v;
		vec_t diag_W_inv, B_invt_u, B_invt_h, B_invt_rhs, Sigma_Lkt_W_r, Sigma_rhs, W_r;
		bool early_stop_alg = false;
		double a = 0;
		double b = 1;
		double r_squared_norm;

		//std::chrono::steady_clock::time_point begin, end;
		//double el_time;

		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			//Log::REInfo("0 - return 0");
			u.setZero();
			return;
		}

		if (!warm_start) {
			//Log::REInfo("DOING COLD START");
			u.setZero(); //No warm-start
		}
		diag_W_inv = diag_W.cwiseInverse();

		//Sigma*rhs, where Sigma = B^(-1)DB^(-T)
		B_invt_rhs = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(rhs); //B^(-T)rhs
		Sigma_rhs = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_rhs);

		//r = Sigma*rhs - (Sigma + W^(-1)) * u
		B_invt_u = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(u); //B^(-T)u
		r = Sigma_rhs - (D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_u) + diag_W_inv.cwiseProduct(u)); //r = rhs - A * u

		//z = P^(-1) r 
		//P^(-1) = (Sigma_L_k Sigma_L_k^T + W^(-1))^(-1) = W - W Sigma_L_k (I_k + Sigma_L_k^T W Sigma_L_k)^(-1) Sigma_L_k^T W
		W_r = diag_W.asDiagonal() * r;
		Sigma_Lkt_W_r = Sigma_L_k.transpose() * W_r;
		//No case distinction for the brackets since Sigma_L_k is dense
		z = W_r - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_r));

		h = z;
		
		//Log::REInfo("Start triangular solve()");
		//for (int j = 0; j < 200; ++j) {
		//	B_invt_h = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(h);
		//	v = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_h) + diag_W_inv.cwiseProduct(h);
		//}

		//Log::REInfo("Start chol solve()");
		//W_r = diag_W.asDiagonal() * r;
		//Sigma_Lkt_W_r = Sigma_L_k.transpose() * W_r;
		//vec_t x;
		//for (int j = 0; j < 10000; ++j) {
		//	x = chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_r);
		//}

		for (int j = 0; j < p; ++j) {
			//The following matrix-vector operation is the expensive part of the loop
			//K=100: matrix-vector operation requires ~80% of Loop-Runtime
			//K=5: matrix-vector operation requires ~97% of Loop-Runtime
			//CPU is nearly 100% utilized 
			//begin = std::chrono::steady_clock::now();
			//(Sigma + W^(-1)) * h
			B_invt_h = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(h);
			v = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_h) + diag_W_inv.cwiseProduct(h);
			//end = std::chrono::steady_clock::now();
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("CGVecchiaLaplaceVecSigmaplusWI MV %i: %g", j, el_time);
			//begin = std::chrono::steady_clock::now();

			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_squared_norm = r.squaredNorm();
			//Log::REInfo("r.squaredNorm(): %g", r_squared_norm);
			if (std::isnan(r_squared_norm) || std::isinf(r_squared_norm)) {
				NaN_found = true;
				return;
			}
			if (r_squared_norm < delta_conv) {
				early_stop_alg = true;
			}

			z_old = z;

			//z = P^(-1) r
			W_r = diag_W.asDiagonal() * r;
			Sigma_Lkt_W_r = Sigma_L_k.transpose() * W_r;
			//No case distinction for the brackets since Sigma_L_k is dense
			z = W_r - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_r));

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;

			if (early_stop_alg || (j + 1) == p) {
				//Log::REInfo("CGVecchiaLaplaceVecSigmaplusWinv stop after %i CG-Iterations.", j);
				//u=W^(-1) u
				u = diag_W_inv.cwiseProduct(u);
				return;
			}
			//end = std::chrono::steady_clock::now();
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("CGVecchiaLaplaceVecSigmaplusWI remaining loop time: %g", el_time);
		}
		Log::REInfo("CGVecchiaLaplaceVecSigmaplusWinv used all %i iterations!", p);
		Log::REInfo("final r.squaredNorm(): %g", r_squared_norm);
	} // end CGVecchiaLaplaceVecSigmaplusWinv

	void CGTridiagVecchiaLaplace(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const chol_sp_mat_rm_t& chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia,
		const chol_sp_mat_rm_t& chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia,
		const sp_mat_rm_t& B_k_rm,
		const sp_mat_rm_t& L_k_rm,
		const sp_mat_rm_t& D_inv_plus_W_B_rm) {

		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, B_invt_R(num_data, t), Z(num_data, t), Z_old, H, V(num_data, t), L_kt_W_inv_R, B_k_W_inv_R, W_inv_R; //NEW V(num_data, t)
		//sp_mat_rm_t D_inv_plus_W_B_rm;
		vec_t v1(num_data), diag_SigmaI_plus_W_inv, diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_squared_R_norm;
		std::chrono::milliseconds timespan(100); //temp

		//std::chrono::steady_clock::time_point begin, end;
		//double el_time;

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//B_t_D_inv_rm = B_rm.transpose() * D_inv_rm;

		//R = rhs - ((B_t_D_inv_rm * (B_rm * U)) + diag_W.replicate(1, t).cwiseProduct(U));
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < t; ++i) {
			R.col(i) = rhs.col(i) - ((B_t_D_inv_rm * (B_rm * U.col(i))) + diag_W.cwiseProduct(U.col(i))); //parallelization in for loop is much faster
		}

		//Z = P^(-1) R 
		if (cg_preconditioner_type == "piv_chol_veccia") {
			//Pivoted Cholseky - Version Veccia
			diag_W_inv = diag_W.cwiseInverse();
			//P^(-1) = (B_k^T D_k^(-1) B_k + W)^(-1) = W^(-1) - W^(-1) B_k^T (D_k + B_k W^(-1) B_k^T)^(-1) B_k W^(-1)
			W_inv_R = diag_W_inv.asDiagonal() * R;
			B_k_W_inv_R = B_k_rm * W_inv_R;
			if (B_k_rm.nonZeros() <= num_data * t) {
				//Relevant Version, since B_k_rm.nonZeros() = NN * k 
				Z = W_inv_R - (diag_W_inv.asDiagonal() * B_k_rm.transpose()) * chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia.solve(B_k_W_inv_R);
			}
			else {
				Z = W_inv_R - diag_W_inv.asDiagonal() * (B_k_rm.transpose() * chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia.solve(B_k_W_inv_R));
			}
		}
		else if (cg_preconditioner_type == "piv_chol_habrecht") {
			//Pivoted Cholseky - Version Habrecht
			diag_W_inv = diag_W.cwiseInverse();
			//P^(-1) = (L_k L_k^T + W)^(-1) = W^(-1) - W^(-1) L_k (I_k + L_k^T W^(-1) L_k)^(-1) L_k^T W^(-1)
			W_inv_R = diag_W_inv.asDiagonal() * R;
			L_kt_W_inv_R = ((sp_mat_rm_t)L_k_rm.transpose()) * W_inv_R;
			if (L_k_rm.nonZeros() <= num_data * t) {
				Z = W_inv_R - (diag_W_inv.asDiagonal() * L_k_rm) * chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia.solve(L_kt_W_inv_R);
			}
			else {
				Z = W_inv_R - diag_W_inv.asDiagonal() * (L_k_rm * chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia.solve(L_kt_W_inv_R));
			}
		}
		else if (cg_preconditioner_type == "sigma") {
			//P^(-1) = \Sigma = B^(-1)DB^(-T)
			//D_inv_B_rm = D_inv_rm * B_rm; //D^(-1)B
			//B_invt_R = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R);
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				B_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
			}
			//Z = B_t_D_inv_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(B_invt_R);
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				Z.col(i) = B_t_D_inv_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(B_invt_R.col(i));
			}
		}
		else if (cg_preconditioner_type == "sigma_with_W") {
			//P^(-1) = B^(-1) (D^(-1)+W)^(-1) B^(-T)
			//D_inv_plus_W_B_rm = (D_inv_rm.diagonal() + diag_W).asDiagonal() * B_rm;
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				B_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				Z.col(i) = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_R.col(i));
			}
		}
		else if (cg_preconditioner_type == "diagonal") {
			//P^(-1) = diag(diag(\Sigma^-1 + W)^(-1))
			diag_SigmaI_plus_W_inv = B_t_D_inv_rm.cwiseProduct(((sp_mat_rm_t)B_rm.transpose())) * vec_t::Ones(B_rm.rows()) + diag_W;
			diag_SigmaI_plus_W_inv = diag_SigmaI_plus_W_inv.cwiseInverse();
			Z = diag_SigmaI_plus_W_inv.asDiagonal() * R;
		}
		else {
			Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
		}

		H = Z;

		for (int j = 0; j < p; ++j) {
			//begin = std::chrono::steady_clock::now();
			//V = (B_t_D_inv_rm * (B_rm * H)) + diag_W.replicate(1, t).cwiseProduct(H); //V = A * H //replicate() is fast
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = (B_t_D_inv_rm * (B_rm * H.col(i))) + diag_W.cwiseProduct(H.col(i)); //parallelization in for loop is much faster
			}
			//end = std::chrono::steady_clock::now();
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("CGTridiagVecchiaLaplace MM %i: %g", j, el_time);
			//begin = std::chrono::steady_clock::now(); //NEW

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_squared_R_norm = 0;

#pragma omp parallel for schedule(static)            
			for (int i = 0; i < t; ++i) {
				mean_squared_R_norm += R.col(i).squaredNorm();
			}
			mean_squared_R_norm /= t;
			//Log::REInfo("mean_squared_R_norm: %g | Iteration: %i", mean_squared_R_norm, j);
			if (std::isnan(mean_squared_R_norm) || std::isinf(mean_squared_R_norm)) {
				NaN_found = true;
				return;
			}
			if (mean_squared_R_norm < delta_conv) {
				early_stop_alg = true;
			}
			//for (int i = 0; i < t; ++i) {
			//    if (R.col(i).squaredNorm() > delta_conv) {
			//        break; //continue till all cols are below delta_conv
			//    }
			//    if (i == t - 1) {
			//        early_stop_alg = true;
			//    }
			//}

			Z_old = Z;

			//end = std::chrono::steady_clock::now(); //NEW
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("In between: %g", el_time);
			//begin = std::chrono::steady_clock::now(); //NEW

			if (cg_preconditioner_type == "piv_chol_veccia") {
				//Pivoted Cholseky - Version Veccia
				W_inv_R = diag_W_inv.asDiagonal() * R;
				B_k_W_inv_R = B_k_rm * W_inv_R;
				if (B_k_rm.nonZeros() <= num_data * t) {
					//Relevant Version, since B_k_rm.nonZeros() = NN * k 
					Z = W_inv_R - (diag_W_inv.asDiagonal() * B_k_rm.transpose()) * chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia.solve(B_k_W_inv_R);
				}
				else {
					Z = W_inv_R - diag_W_inv.asDiagonal() * (B_k_rm.transpose() * chol_fact_D_k_plus_B_k_W_inv_B_kt_rm_vecchia.solve(B_k_W_inv_R));
				}
			}
			else if (cg_preconditioner_type == "piv_chol_habrecht") {
				//Pivoted Cholseky - Version Habrecht
				W_inv_R = diag_W_inv.asDiagonal() * R;
				L_kt_W_inv_R = ((sp_mat_rm_t)L_k_rm.transpose()) * W_inv_R;
				//Parallelization of cholseky-solve() doesn't lead to a relevant performance increase: Runtime of cholseky-solve() is neglectable compared to the runtime to calculate Z
				if (L_k_rm.nonZeros() <= num_data * t) {
					Z = W_inv_R - (diag_W_inv.asDiagonal() * L_k_rm) * chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia.solve(L_kt_W_inv_R);
				}
				else {
					Z = W_inv_R - diag_W_inv.asDiagonal() * (L_k_rm * chol_fact_I_k_plus_L_kt_W_inv_L_k_rm_vecchia.solve(L_kt_W_inv_R));
				}
			}
			else if (cg_preconditioner_type == "sigma") {
				//P^(-1) = \Sigma
				//Parallelization: The running time is reduced to a third.
				//B_invt_R = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R);
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					B_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
				}
				//Z = B_t_D_inv_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(B_invt_R);
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					Z.col(i) = B_t_D_inv_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(B_invt_R.col(i));
				}
			}
			else if (cg_preconditioner_type == "sigma_with_W") {
				//P^(-1) = B^(-1) (D^(-1)+W)^(-1) B^(-T)
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					B_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					Z.col(i) = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_R.col(i));
				}
			}
			else if (cg_preconditioner_type == "diagonal") {
				//P^(-1) = diag(diag(\Sigma^-1 + W)^(-1))
				Z = diag_SigmaI_plus_W_inv.asDiagonal() * R;
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported.", cg_preconditioner_type.c_str());
			}

			//end = std::chrono::steady_clock::now(); //NEW
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("PI R: %g", el_time);
			//begin = std::chrono::steady_clock::now(); //NEW

			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

#pragma omp parallel for schedule(static)
			for (int i = 0; i < t; ++i) {
				Tdiags[i][j] = 1 / a(i) + b_old(i) / a_old(i);
				if (j > 0) {
					Tsubdiags[i][j - 1] = sqrt(b_old(i)) / a_old(i);
				}
			}

			if (early_stop_alg) {
				//Log::REInfo("CGTridiagVecchiaLaplace stop after %i CG-Iterations.", j);
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}

			//end = std::chrono::steady_clock::now(); //NEW
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("rest loop time: %g", el_time);
		}
		Log::REInfo("CGTridiagVecchiaLaplace used all %i iterations!", p);
		Log::REInfo("final mean_squared_R_norm: %g", mean_squared_R_norm);
	} // end CGTridiagVecchiaLaplace

	void CGTridiagVecchiaLaplaceSigmaplusWinv(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NaN_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const chol_den_mat_t chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k) {

		p = std::min(p, (int)num_data);

		den_mat_t B_invt_U(num_data, t), Sigma_Lkt_W_R, B_invt_H(num_data, t), W_R;
		den_mat_t R(num_data, t), R_old, Z, Z_old, H, V(num_data, t);
		vec_t v1(num_data), diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_squared_R_norm;
		std::chrono::milliseconds timespan(100); //temp

		//std::chrono::steady_clock::time_point begin, end;
		//double el_time;

		diag_W_inv = diag_W.cwiseInverse();
		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//R = rhs - (Sigma + W^(-1)) * U
		//D_inv_B_rm = D_inv_rm * B_rm; //D^(-1)B
		
		//For t=100: 40% performance increase through paralellization
		//B_invt_U = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(U); //B^(-T)U
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < t; ++i) {
			B_invt_U.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(U.col(i));
		}
		//R = rhs - (D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_U) + diag_W_inv.replicate(1, t).cwiseProduct(U));
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < t; ++i) {
			R.col(i) = rhs.col(i) - D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_U.col(i));
		}
		R -= diag_W_inv.replicate(1, t).cwiseProduct(U);

		//Z = P^(-1) R 
		//P^(-1) = (Sigma_L_k Sigma_L_k^T + W^(-1))^(-1) = W - W Sigma_L_k (I_k + Sigma_L_k^T W Sigma_L_k)^(-1) Sigma_L_k^T W
		W_R = diag_W.asDiagonal() * R;
		Sigma_Lkt_W_R = Sigma_L_k.transpose() * W_R;
		if (Sigma_L_k.cols() < t) {
			Z = W_R - (diag_W.asDiagonal() * Sigma_L_k) * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R);
		}
		else {
			Z = W_R - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R));
		}

		H = Z;

		//Log::REInfo("Start triangular solve()");
		//for (int j = 0; j < 100; ++j) {
		//	B_invt_H = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(H); //B^(-T)H
		//	V = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_H) + diag_W_inv.replicate(1, t).cwiseProduct(H);
		//}
		//Log::REInfo("Start chol solve()");
		//W_R = diag_W.asDiagonal() * R;
		//Sigma_Lkt_W_R = Sigma_L_k.transpose() * W_R;
		//den_mat_t X;
		//for (int j = 0; j < 1000; ++j) {
		//	if (Sigma_L_k.cols() < t) {
		//		X = chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R);
		//	}
		//	else {
		//		X =  chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R);
		//	}
		//}

		for (int j = 0; j < p; ++j) {
			//The following matrix-vector operation is the expensive part of the loop
			//old: K=100: matrix-vector operation requires ~42% (old:~60%) of Loop-Runtime
			//K=5: matrix-vector operation requires ~45% (old:~70%) of Loop-Runtime
			//CPU usage: Zigzag -> triangularView-solve() is not parallelized for multiple rhs.
			//The larger k the better the CPU-usage -> More MM-Multiplications that can be parallelized (cholseky-solve() doesn't use full CPU-capacity (35%) for multiple rhs, but more than triangularView-solve() (20%)).
			//begin = std::chrono::steady_clock::now();
			//Parallelization leads to 100% CPU utilization and half the runtime
			//B_invt_H = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(H); //B^(-T)H
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				B_invt_H.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(H.col(i));
			}
			//V = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_H) + diag_W_inv.replicate(1, t).cwiseProduct(H); //replicate() is fast
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_H.col(i));
			}
			V += diag_W_inv.replicate(1, t).cwiseProduct(H);
			//end = std::chrono::steady_clock::now();
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("CGTridiagVecchiaLaplaceSigmaplusWinv MM %i: %g", j, el_time);
			//begin = std::chrono::steady_clock::now();

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_squared_R_norm = 0;

#pragma omp parallel for schedule(static)            
			for (int i = 0; i < t; ++i) {
				mean_squared_R_norm += R.col(i).squaredNorm();
			}
			mean_squared_R_norm /= t;
			if (std::isnan(mean_squared_R_norm) || std::isinf(mean_squared_R_norm)) {
				NaN_found = true;
				return;
			}
			if (mean_squared_R_norm < delta_conv) {
				early_stop_alg = true;
			}

			//for (int i = 0; i < t; ++i) {
			//    if (R.col(i).squaredNorm() > delta_conv) {
			//        break; //continue till all cols are below delta_conv
			//    }
			//    if (i == t - 1) {
			//        early_stop_alg = true;
			//    }
			//}

			Z_old = Z;

			//Z = P^(-1) R
			W_R = diag_W.asDiagonal() * R;
			Sigma_Lkt_W_R = Sigma_L_k.transpose() * W_R;
			if (Sigma_L_k.cols() < t) {
				Z = W_R - (diag_W.asDiagonal() * Sigma_L_k) * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R);
			}
			else {
				Z = W_R - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R));
			}
			b_old = b;
			b = (R.cwiseProduct(Z).transpose() * v1).array() * (R_old.cwiseProduct(Z_old).transpose() * v1).array().inverse();

			H = Z + H * b.asDiagonal();

#pragma omp parallel for schedule(static)
			for (int i = 0; i < t; ++i) {
				Tdiags[i][j] = 1 / a(i) + b_old(i) / a_old(i);
				if (j > 0) {
					Tsubdiags[i][j - 1] = sqrt(b_old(i)) / a_old(i);
				}
			}

			if (early_stop_alg) {
				//Log::REInfo("CGTridiagVecchiaLaplaceSigmaplusWinv stop after %i CG-Iterations.", j);
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}
			//end = std::chrono::steady_clock::now();
			//el_time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.;
			//Log::REInfo("CGTridiagVecchiaLaplaceSigmaplusWinv remaining loop time: %g", el_time);
		}
		Log::REInfo("CGTridiagVecchiaLaplaceSigmaplusWinv used all %i iterations!", p);
		Log::REInfo("final mean_squared_R_norm: %g", mean_squared_R_norm);
	} // end CGTridiagVecchiaLaplaceSigmaplusWinv

	void simRademacher(RNG_t& generator, den_mat_t& Z) {

		//std::uniform_real_distribution<double> udist(0.0, 1.0);
		std::normal_distribution<double> ndist(0.0, 1.0);
		//double u;
		
		//0.94s for n=100k + t=100 (parellel: 0.55s)
//#pragma omp parallel for schedule(static) //Do not parallelize! - Despite seed: no longer deterministic
		for (int i = 0; i < Z.rows(); ++i) {
			for (int j = 0; j < Z.cols(); j++) {
				Z(i, j) = ndist(generator);
				//u = udist(generator);
				//if (u > 0.5) {
				//	Z(i, j) = 1.;
				//}
				//else {
				//	Z(i, j) = -1.;
				//}
			}
		}

		//den_mat_t Z_COV = Z.transpose() * Z;
		//Log::REInfo("Z.rows(): %i", Z.rows());
		//Log::REInfo("Z.cols(): %i", Z.cols());
		//Log::REInfo("Z.mean(): %g", Z.mean());
		//Log::REInfo("Z_COV.sum(): %g", Z_COV.sum());
		//Log::REInfo("Z_COV.cwiseAbs().sum(): %g", Z_COV.cwiseAbs().sum());
		//Log::REInfo("Z_COV.diagonal().sum(): %g", Z_COV.diagonal().sum());

		//Old version: rand() is bad
		//double threshold = round(RAND_MAX * 0.5);
		////#pragma omp parallel for schedule(static) //Do not parallelize! - Estimators based on Z get biased.
		//for (int i = 0; i < Z.rows(); ++i) {
		//	for (int j = 0; j < Z.cols(); j++) {
		//		Z(i, j) = (rand() < threshold) ? -1. : 1.;
		//	}
		//}
	}

	void LogDetStochTridiag(const std::vector<vec_t>& Tdiags,
		const  std::vector<vec_t>& Tsubdiags,
		double& ldet,
		const data_size_t num_data,
		const int t) {
		Eigen::SelfAdjointEigenSolver<den_mat_t> es;
		ldet = 0;
		vec_t e1_logLambda_e1;
		//int size_Tdiags, size_Tsubdiags;

		//size_Tdiags = Tdiags[0].rows();
		//size_Tsubdiags = Tsubdiags[0].rows();

		//Log::REInfo("Size of Tdiags: %i", size_Tdiags);
		//Log::REInfo("Size of Tsubdiags: %i", size_Tsubdiags);

		//for (int k = 0; k < t; ++k) {
		//    for (int j = 0; j < size_Tdiags; ++j) {
		//        Log::REInfo("Tdiags[%i][%i]: %g", k, j, Tdiags[k][j]);
		//    }
		//}

		for (int i = 0; i < t; ++i) {

			e1_logLambda_e1.setZero();

			es.computeFromTridiagonal(Tdiags[i], Tsubdiags[i]);

			//e1_logLambda_e1 = es.eigenvectors().row(0).array() * es.eigenvalues().array().log() * es.eigenvectors().row(0).array();
			e1_logLambda_e1 = es.eigenvectors().row(0).transpose().array() * es.eigenvalues().array().log() * es.eigenvectors().row(0).transpose().array();

			ldet += e1_logLambda_e1.sum();
		}

		ldet = ldet * num_data / t;
	} // end LogDetStochTridiag

	void StochTraceVecchiaLaplace(const den_mat_t& A_t_Z,
		const vec_t& third_deriv,
		const den_mat_t& B_PI_Z,
		vec_t& tr) {

		den_mat_t D = third_deriv.replicate(1, B_PI_Z.cols());
		tr = (A_t_Z.array() * D.array() * B_PI_Z.array()).matrix().rowwise().mean();
		tr *= -1; //Since third_deriv = -dW/db_i
	}

	void LanczosTridiagVecchiaLaplace(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const vec_t& b_init,
		const data_size_t num_data,
		vec_t& Tdiag_k,
		vec_t& Tsubdiag_k,
		den_mat_t& Q_k,
		int max_it,
		const double tol,
		const string_t preconditioner_type) {

		bool could_reorthogonalize;
		int final_rank = 1;
		double alpha_0, beta_0, alpha_curr, beta_curr, beta_prev;
		vec_t q_0, q_curr, q_prev, r, inner_products;
		den_mat_t Q_filled;

		max_it = std::min(max_it, num_data);

		//symmetric preconditioning
		vec_t D_inv_plus_W_sqrt_diag, D_inv_plus_W_sqrt_inv_diag, D_inv_plus_W_sqrt_inv_D_inv_D_inv_plus_W_sqrt_inv_diag, P_t_sqrt_inv_q, W_P_t_sqrt_inv_q;
		sp_mat_rm_t B_t_D_inv_plus_W_sqrt_rm;
		if (preconditioner_type == "symmetric") {
			//P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
			D_inv_plus_W_sqrt_diag = (D_inv_rm.diagonal() + diag_W).cwiseSqrt();
			D_inv_plus_W_sqrt_inv_diag = D_inv_plus_W_sqrt_diag.cwiseInverse();
			B_t_D_inv_plus_W_sqrt_rm = B_rm.transpose() * D_inv_plus_W_sqrt_diag.asDiagonal(); //P^(0.5) = B^T (D^(-1) + W)^(0.5)
			D_inv_plus_W_sqrt_inv_D_inv_D_inv_plus_W_sqrt_inv_diag = D_inv_plus_W_sqrt_inv_diag.array() * D_inv_rm.diagonal().array() * D_inv_plus_W_sqrt_inv_diag.array(); //Diagonal of (D^(-1)+W)^(-0.5) * D^(-1) * (D^(-1)+W)^(-0.5)
		}
	
		//asymmetric preconditioning
		vec_t W_q, B_invt_W_q;

		//Inital vector of Q_k: q_0
		q_0 = b_init / b_init.norm();
		Q_k.col(0) = q_0;

		//Initial alpha value: alpha_0
		if (preconditioner_type == "symmetric") {
			//P^(-0.5) (Sigma^-1+W) P^(-0.5*T) q_0 = (D^(-1) + W)^(-0.5) D^(-1) (D^(-1) + W)^(-0.5) q_0 + P^(-0.5) W P^(-0.5*T) q_0
			P_t_sqrt_inv_q = B_t_D_inv_plus_W_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(q_0); //P^(-0.5*T) q_0
			W_P_t_sqrt_inv_q = diag_W.cwiseProduct(P_t_sqrt_inv_q); //W P^(-0.5*T) q_0
			r = B_t_D_inv_plus_W_sqrt_rm.triangularView<Eigen::UpLoType::Upper>().solve(W_P_t_sqrt_inv_q); //P^(-0.5) W P^(-0.5*T) q_0
			r += D_inv_plus_W_sqrt_inv_D_inv_D_inv_plus_W_sqrt_inv_diag.cwiseProduct(q_0); //(D^(-1) + W)^(-0.5) D^(-1) (D^(-1) + W)^(-0.5) q_0
		}
		else if (preconditioner_type == "asymmetric") {
			//(I + Sigma W) q_0
			W_q = diag_W.cwiseProduct(q_0);
			B_invt_W_q = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(W_q);
			r = B_t_D_inv_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(B_invt_W_q);
			r += q_0;
		}
		else {
			//(Sigma^-1+W) q_0
			r = B_t_D_inv_rm * (B_rm * q_0) + diag_W.cwiseProduct(q_0);
		}
		alpha_0 = q_0.dot(r);

		//Initial beta value: beta_0
		r -= alpha_0 * q_0;
		beta_0 = r.norm();

		//Store alpha_0 and beta_0 into T_k
		Tdiag_k(0) = alpha_0;
		Tsubdiag_k(0) = beta_0;

		//Compute next vector of Q_k: q_1
		Q_k.col(1) = r / beta_0;

		//Start the iterations
		for (int k = 1; k < max_it; ++k) {
			//Log::REInfo("k: %i", k);
			//Get previous values
			q_prev = Q_k.col(k-1);
			q_curr = Q_k.col(k);
			beta_prev = Tsubdiag_k(k-1);

			//Compute next alpha value
			if (preconditioner_type == "symmetric") {
				P_t_sqrt_inv_q = B_t_D_inv_plus_W_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(q_curr);
				W_P_t_sqrt_inv_q = diag_W.cwiseProduct(P_t_sqrt_inv_q);
				r = B_t_D_inv_plus_W_sqrt_rm.triangularView<Eigen::UpLoType::Upper>().solve(W_P_t_sqrt_inv_q);
				r += D_inv_plus_W_sqrt_inv_D_inv_D_inv_plus_W_sqrt_inv_diag.cwiseProduct(q_curr);
				r -= beta_prev * q_prev;
			}
			else if (preconditioner_type == "asymmetric") {
				W_q = diag_W.cwiseProduct(q_curr);
				B_invt_W_q = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(W_q);
				r = B_t_D_inv_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(B_invt_W_q);
				r += q_curr;
				r -= beta_prev * q_prev;
			}
			else {
				r = B_t_D_inv_rm * (B_rm * q_curr) + diag_W.cwiseProduct(q_curr) - beta_prev * q_prev;
			}
			alpha_curr = q_curr.dot(r);

			//Store alpha_curr
			Tdiag_k(k) = alpha_curr;
			final_rank += 1;

			if ((k+1) < max_it) {

				//Compute next residual
				r -= alpha_curr * q_curr;

				//Full reorthogonalization: r = r - Q_k*(Q_k' r)
				Q_filled = Q_k(Eigen::all, Eigen::seq(0,k));
				r -= Q_filled * (Q_filled.transpose() * r);
				
				//Compute next beta value
				beta_curr = r.norm();
				Tsubdiag_k(k) = beta_curr;
				
				r /= beta_curr;

				//More reorthogonalizations if necessary
				inner_products = Q_filled.transpose() * r;
				could_reorthogonalize = false;
				for (int l = 0; l < 10; ++l) {
					if ((inner_products.array() < tol).all()) {
						could_reorthogonalize = true;
						break;
					}
					Log::REInfo("Rereorthogonalize");
					r -= Q_filled * (Q_filled.transpose() * r);
					r /= r.norm();
					inner_products = Q_filled.transpose() * r;
				}

				//Store next vector of Q_k
				Q_k.col(k + 1) = r;

				if (abs(beta_curr) < 1e-6 || !could_reorthogonalize) {
					break;
				}
			}
		}
		
		//Resize Q_k, Tdiag_k, Tsubdiag_k
		//Log::REInfo("final rank: %i", final_rank);
		Q_k.conservativeResize(num_data, final_rank);
		Tdiag_k.conservativeResize(final_rank, 1);
		Tsubdiag_k.conservativeResize(final_rank-1, 1);

		if (preconditioner_type == "symmetric") {
			//Adjust for preconditioning with Q_k = P^(-0.5*T) Q_tilde_k
			den_mat_t Q_tilde_k = Q_k;
#pragma omp parallel for schedule(static)   
			for (int m = 0; m < final_rank; ++m) {
				Q_k.col(m) = B_t_D_inv_plus_W_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(Q_tilde_k.col(m));
			}
		}
	}
}