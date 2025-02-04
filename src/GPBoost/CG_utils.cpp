/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2022 - 2024 Tim Gyger, Pascal Kuendig, and Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/CG_utils.h>
#include <GPBoost/type_defs.h>
#include <GPBoost/re_comp.h>
#include <LightGBM/utils/log.h>

#include <chrono>
#include <thread> //temp

using LightGBM::Log;

namespace GPBoost {

	void CGVecchiaLaplaceVec(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const string_t cg_preconditioner_type,
		const sp_mat_rm_t& D_inv_plus_W_B_rm,
		const sp_mat_rm_t& L_SigmaI_plus_W_rm,
		bool run_in_parallel_do_not_report_non_convergence) {

		p = std::min(p, (int)B_rm.cols());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h, v, B_invt_r, L_invt_r;
		double a, b, r_norm;
		
		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}

		//Cold-start in the first iteration of mode finding, otherwise always warm-start (=initalize with mode from previous iteration)
		if (find_mode_it == 0) {
			u.setZero();
			//r = rhs - A * u
			r = rhs; //since u is 0
		}
		else {
			//r = rhs - A * u
			r = rhs - ((B_t_D_inv_rm * (B_rm * u)) + diag_W.cwiseProduct(u));
		}

		if (cg_preconditioner_type == "vadu") {
			//z = P^(-1) r, where P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
			B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r);
			z = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);
		}
		else if (cg_preconditioner_type == "incomplete_cholesky") {
			//z = P^(-1) r, where P^(-1) = L^(-1) L^(-T)
			L_invt_r = L_SigmaI_plus_W_rm.transpose().triangularView<Eigen::UpLoType::Upper>().solve(r);
			z = L_SigmaI_plus_W_rm.triangularView<Eigen::UpLoType::Lower>().solve(L_invt_r);
		}
		else {
			Log::REFatal("CGVecchiaLaplaceVec: Preconditioner type '%s' is not supported ", cg_preconditioner_type.c_str());
		}

		h = z;

		for (int j = 0; j < p; ++j) {
			//Parentheses are necessery for performance, otherwise EIGEN does the operation wrongly from left to right
			v = (B_t_D_inv_rm * (B_rm * h)) + diag_W.cwiseProduct(h);
			
			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_norm = r.norm();
			//Log::REInfo("r.norm(): %g | Iteration: %i", r_norm, j);
			if (std::isnan(r_norm) || std::isinf(r_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (r_norm < delta_conv) {
				//Log::REInfo("Number CG iterations: %i", j + 1);
				return;
			}

			z_old = z;

			if (cg_preconditioner_type == "vadu") {
				//z = P^(-1) r 
				B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r);
				z = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);
			}
			else if (cg_preconditioner_type == "incomplete_cholesky") {
				//z = P^(-1) r, where P^(-1) = L^(-1) L^(-T)
				L_invt_r = L_SigmaI_plus_W_rm.transpose().triangularView<Eigen::UpLoType::Upper>().solve(r);
				z = L_SigmaI_plus_W_rm.triangularView<Eigen::UpLoType::Lower>().solve(L_invt_r);
			}
			else {
				Log::REFatal("CGVecchiaLaplaceVec: Preconditioner type '%s' is not supported ", cg_preconditioner_type.c_str());
			}

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;
		}
		if (!run_in_parallel_do_not_report_non_convergence) {
			Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
				"This could happen if the initial learning rate is too large in a line search phase. Otherwise you might increase 'cg_max_num_it' ", p);
		}
	} // end CGVecchiaLaplaceVec

	void CGVecchiaLaplaceVecWinvplusSigma(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const chol_den_mat_t& chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k,
		bool run_in_parallel_do_not_report_non_convergence) {

		p = std::min(p, (int)B_rm.cols());

		CHECK(Sigma_L_k.rows() == B_rm.cols());
		CHECK(Sigma_L_k.rows() == diag_W.size());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h, v, diag_W_inv, B_invt_u, B_invt_h, B_invt_rhs, Sigma_Lkt_W_r, Sigma_rhs, W_r;
		double a, b, r_norm;

		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}

		diag_W_inv = diag_W.cwiseInverse();

		//Sigma * rhs, where Sigma = B^(-1) D B^(-T)
		B_invt_rhs = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(rhs);
		Sigma_rhs = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_rhs);

		//Cold-start in the first iteration of mode finding, otherwise always warm-start (=initalize with mode from previous iteration)
		if (find_mode_it == 0) {
			u.setZero();
			//r = Sigma * rhs - (W^(-1) + Sigma) * u
			r = Sigma_rhs; //since u is 0
		}
		else {
			//r = Sigma * rhs - (W^(-1) + Sigma) * u
			B_invt_u = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(u);
			r = Sigma_rhs - (D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_u) + diag_W_inv.cwiseProduct(u));
		}

		//z = P^(-1) r 
		//P^(-1) = (W^(-1) + Sigma_L_k Sigma_L_k^T)^(-1) = W - W Sigma_L_k (I_k + Sigma_L_k^T W Sigma_L_k)^(-1) Sigma_L_k^T W
		W_r = diag_W.asDiagonal() * r;
		Sigma_Lkt_W_r = Sigma_L_k.transpose() * W_r;
		//No case distinction for the brackets since Sigma_L_k is dense
		z = W_r - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_r));

		h = z;
		
		for (int j = 0; j < p; ++j) {
			//(W^(-1) + Sigma) * h
			B_invt_h = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(h);
			v = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_h) + diag_W_inv.cwiseProduct(h);

			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_norm = r.norm();
			if (std::isnan(r_norm) || std::isinf(r_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (r_norm < delta_conv || (j + 1) == p) {
				//u = W^(-1) u
				u = diag_W_inv.cwiseProduct(u);
				if ((j + 1) == p) {
					if (!run_in_parallel_do_not_report_non_convergence) {
						Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
							"This could happen if the initial learning rate is too large in a line search phase. Otherwise you might increase 'cg_max_num_it' ", p);
					}
				}
				//Log::REInfo("Number CG iterations: %i", j + 1);//for debugging
				return;
			}

			z_old = z;

			//z = P^(-1) r
			W_r = diag_W.asDiagonal() * r;
			Sigma_Lkt_W_r = Sigma_L_k.transpose() * W_r;
			z = W_r - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_r));

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;
		}
	} // end CGVecchiaLaplaceVecSigmaplusWinv


	void CGVecchiaLaplaceVecWinvplusSigma_FITC_P(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& rhs,
		vec_t& u,
		bool& NA_or_Inf_found,
		int p,
		const int find_mode_it,
		const double delta_conv,
		const double THRESHOLD_ZERO_RHS_CG,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const den_mat_t cross_cov,
		const vec_t& diagonal_approx_inv_preconditioner,
		bool run_in_parallel_do_not_report_non_convergence) {

		p = std::min(p, (int)B_rm.cols());

		CHECK((cross_cov).rows() == B_rm.cols());
		CHECK((cross_cov).rows() == diag_W.size());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h, v, diag_W_inv, B_invt_u, B_invt_h, B_invt_rhs, Sigma_rhs, W_r;
		double a, b, r_norm;

		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}

		diag_W_inv = diag_W.cwiseInverse();

		//Sigma * rhs, where Sigma = B^(-1) D B^(-T)
		B_invt_rhs = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(rhs);
		Sigma_rhs = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_rhs);

		//Cold-start in the first iteration of mode finding, otherwise always warm-start (=initalize with mode from previous iteration)
		if (find_mode_it == 0) {
			u.setZero();
			//r = Sigma * rhs - (W^(-1) + Sigma) * u
			r = Sigma_rhs; //since u is 0
		}
		else {
			//r = Sigma * rhs - (W^(-1) + Sigma) * u
			B_invt_u = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(u);
			r = Sigma_rhs - (D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_u) + diag_W_inv.cwiseProduct(u));
		}

		//z = P^(-1) r 
		W_r = diagonal_approx_inv_preconditioner.asDiagonal() * r;
		//No case distinction for the brackets since Sigma_L_k is dense
		z = W_r - diagonal_approx_inv_preconditioner.asDiagonal() * ((cross_cov) * chol_fact_woodbury_preconditioner.solve((cross_cov).transpose() * W_r));

		h = z;
		
		for (int j = 0; j < p; ++j) {
			//(W^(-1) + Sigma) * h
			B_invt_h = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(h);
			v = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_h) + diag_W_inv.cwiseProduct(h);

			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_norm = r.norm();
			if (std::isnan(r_norm) || std::isinf(r_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (r_norm < delta_conv || (j + 1) == p) {
				//u = W^(-1) u
				u = diag_W_inv.cwiseProduct(u);
				if ((j + 1) == p) {
					if (!run_in_parallel_do_not_report_non_convergence) {
						Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
							"This could happen if the initial learning rate is too large in a line search phase. Otherwise you might increase 'cg_max_num_it' ", p);
					}
				}
				//Log::REInfo("Number CG iterations: %i", j + 1);//for debugging
				return;
			}

			z_old = z;

			//z = P^(-1) r
			W_r = diagonal_approx_inv_preconditioner.asDiagonal() * r;
			//No case distinction for the brackets since Sigma_L_k is dense
			z = W_r - diagonal_approx_inv_preconditioner.asDiagonal() * ((cross_cov) * chol_fact_woodbury_preconditioner.solve((cross_cov).transpose() * W_r));



			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;
		}
	} // end CGVecchiaLaplaceVecWinvplusSigma_FITC_P

	void CGTridiagVecchiaLaplace(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const string_t cg_preconditioner_type,
		const sp_mat_rm_t& D_inv_plus_W_B_rm,
		const sp_mat_rm_t& L_SigmaI_plus_W_rm) {

		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, P_sqrt_invt_R(num_data, t), Z(num_data, t), Z_old, H, V(num_data, t), L_kt_W_inv_R, B_k_W_inv_R, W_inv_R;
		vec_t v1(num_data), diag_SigmaI_plus_W_inv, diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//R = rhs - (W^(-1) + Sigma) * U
		R = rhs; //Since U is 0

		if (cg_preconditioner_type == "vadu") {
			//Z = P^(-1) R 		
			//P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				P_sqrt_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				Z.col(i) = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(P_sqrt_invt_R.col(i));
			}
		}
		else if (cg_preconditioner_type == "incomplete_cholesky") {
			//Z = P^(-1) R 		
			//P^(-1) = L^(-1) L^(-T)
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				P_sqrt_invt_R.col(i) = L_SigmaI_plus_W_rm.transpose().triangularView<Eigen::UpLoType::Upper>().solve(R.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				Z.col(i) = L_SigmaI_plus_W_rm.triangularView<Eigen::UpLoType::Lower>().solve(P_sqrt_invt_R.col(i));
			}
		}
		else {
			Log::REFatal("CGTridiagVecchiaLaplace: Preconditioner type '%s' is not supported ", cg_preconditioner_type.c_str());
		}

		H = Z;

		for (int j = 0; j < p; ++j) {
			//V = (Sigma^(-1) + W) H
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = (B_t_D_inv_rm * (B_rm * H.col(i))) + diag_W.cwiseProduct(H.col(i));
			}

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();

			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
				//Log::REInfo("Number CG-Tridiag iterations: %i", j + 1);
			}

			Z_old = Z;

			//Z = P^(-1) R
			if (cg_preconditioner_type == "vadu") {
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					P_sqrt_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					Z.col(i) = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(P_sqrt_invt_R.col(i));
				}
			}
			else if (cg_preconditioner_type == "incomplete_cholesky") {
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					P_sqrt_invt_R.col(i) = L_SigmaI_plus_W_rm.transpose().triangularView<Eigen::UpLoType::Upper>().solve(R.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < t; ++i) {
					Z.col(i) = L_SigmaI_plus_W_rm.triangularView<Eigen::UpLoType::Lower>().solve(P_sqrt_invt_R.col(i));
				}
			}
			else {
				Log::REFatal("CGTridiagVecchiaLaplace: Preconditioner type '%s' is not supported ", cg_preconditioner_type.c_str());
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
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}
		}
		Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise you might increase 'cg_max_num_it_tridiag' ", p);
	} // end CGTridiagVecchiaLaplace

	void CGTridiagVecchiaLaplaceWinvplusSigma(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const chol_den_mat_t& chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia,
		const den_mat_t& Sigma_L_k) {

		p = std::min(p, (int)num_data);

		den_mat_t B_invt_U(num_data, t), Sigma_Lkt_W_R, B_invt_H(num_data, t), W_R;
		den_mat_t R(num_data, t), R_old, Z, Z_old, H, V(num_data, t);
		vec_t v1(num_data), diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		diag_W_inv = diag_W.cwiseInverse();
		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//R = rhs - (W^(-1) + Sigma) * U
		R = rhs; //Since U is 0

		//Z = P^(-1) R 
		//P^(-1) = (W^(-1) + Sigma_L_k Sigma_L_k^T)^(-1) = W - W Sigma_L_k (I_k + Sigma_L_k^T W Sigma_L_k)^(-1) Sigma_L_k^T W
		W_R = diag_W.asDiagonal() * R;
		Sigma_Lkt_W_R = Sigma_L_k.transpose() * W_R;
		if (Sigma_L_k.cols() < t) {
			Z = W_R - (diag_W.asDiagonal() * Sigma_L_k) * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R);
		}
		else {
			Z = W_R - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_R));
		}

		H = Z;

		for (int j = 0; j < p; ++j) {
			//V = (W^(-1) + Sigma) * H - expensive part of the loop
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				B_invt_H.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(H.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_H.col(i));
			}
			V += diag_W_inv.replicate(1, t).cwiseProduct(H);

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse();

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();

			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
				//Log::REInfo("Number CG-Tridiag iterations: %i", j + 1);
			}

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
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}
		}
		Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise you might increase 'cg_max_num_it_tridiag' ", p);
	} // end CGTridiagVecchiaLaplaceSigmaplusWinv


	void CGTridiagVecchiaLaplaceWinvplusSigma_FITC_P(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const den_mat_t& rhs,
		std::vector<vec_t>& Tdiags,
		std::vector<vec_t>& Tsubdiags,
		den_mat_t& U,
		bool& NA_or_Inf_found,
		const data_size_t num_data,
		const int t,
		int p,
		const double delta_conv,
		const chol_den_mat_t& chol_fact_woodbury_preconditioner,
		const den_mat_t* cross_cov,
		const vec_t& diagonal_approx_inv_preconditioner) {

		p = std::min(p, (int)num_data);

		den_mat_t B_invt_U(num_data, t), Sigma_Lkt_W_R, B_invt_H(num_data, t), W_R;
		den_mat_t R(num_data, t), R_old, Z, Z_old, H, V(num_data, t);
		vec_t v1(num_data), diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_R_norm;

		diag_W_inv = diag_W.cwiseInverse();
		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//R = rhs - (W^(-1) + Sigma) * U
		R = rhs; //Since U is 0

		//Z = P^(-1) R 
		W_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
		//No case distinction for the brackets since Sigma_L_k is dense
		Z = W_R - diagonal_approx_inv_preconditioner.asDiagonal() * ((*cross_cov) * chol_fact_woodbury_preconditioner.solve((*cross_cov).transpose() * W_R));

		H = Z;

		for (int j = 0; j < p; ++j) {
			//V = (W^(-1) + Sigma) * H - expensive part of the loop
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				B_invt_H.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(H.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_H.col(i));
			}
			V += diag_W_inv.replicate(1, t).cwiseProduct(H);

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse();

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_R_norm = R.colwise().norm().mean();

			if (std::isnan(mean_R_norm) || std::isinf(mean_R_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (mean_R_norm < delta_conv) {
				early_stop_alg = true;
				//Log::REInfo("Number CG-Tridiag iterations: %i", j + 1);
			}

			Z_old = Z;

			//Z = P^(-1) R 
			W_R = diagonal_approx_inv_preconditioner.asDiagonal() * R;
			//No case distinction for the brackets since Sigma_L_k is dense
			Z = W_R - diagonal_approx_inv_preconditioner.asDiagonal() * ((*cross_cov) * chol_fact_woodbury_preconditioner.solve((*cross_cov).transpose() * W_R));

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
				for (int i = 0; i < t; ++i) {
					Tdiags[i].conservativeResize(j + 1, 1);
					Tsubdiags[i].conservativeResize(j, 1);
				}
				return;
			}
		}
		Log::REInfo("Conjugate gradient algorithm has not converged after the maximal number of iterations (%i). "
			"This could happen if the initial learning rate is too large. Otherwise you might increase 'cg_max_num_it_tridiag' ", p);
	} // end CGTridiagVecchiaLaplaceWinvplusSigma_FITC_P

	void simProbeVect(RNG_t& generator, den_mat_t& Z, const bool rademacher) {

		double u;

		if (rademacher) {
			std::uniform_real_distribution<double> udist(0.0, 1.0);

			for (int i = 0; i < Z.rows(); ++i) {
				for (int j = 0; j < Z.cols(); j++) {
					u = udist(generator);
					if (u > 0.5) {
						Z(i, j) = 1.;
					}
					else {
						Z(i, j) = -1.;
					}
				}
			}
		}
		else {
			std::normal_distribution<double> ndist(0.0, 1.0);

			for (int i = 0; i < Z.rows(); ++i) {
				for (int j = 0; j < Z.cols(); j++) {
					Z(i, j) = ndist(generator);
				}
			}
		}
	} // end simProbeVect

	void GenRandVecNormal(RNG_t& generator,
		den_mat_t& R) {
		std::normal_distribution<double> ndist(0.0, 1.0);
		//Do not parallelize! - Despite seed: no longer deterministic
		for (int i = 0; i < R.rows(); ++i) {
			for (int j = 0; j < R.cols(); j++) {
				R(i, j) = ndist(generator);
			}
		}
	}

	void GenRandVecRademacher(RNG_t& generator,
		den_mat_t& R) {
		double u;
		std::uniform_real_distribution<double> udist(0.0, 1.0);
		//Do not parallelize! - Despite seed: no longer deterministic
		for (int i = 0; i < R.rows(); ++i) {
			for (int j = 0; j < R.cols(); j++) {
				u = udist(generator);
				if (u > 0.5) {
					R(i, j) = 1.;
				}
				else {
					R(i, j) = -1.;
				}
			}
		}
	}

	void LogDetStochTridiag(const std::vector<vec_t>& Tdiags,
		const  std::vector<vec_t>& Tsubdiags,
		double& ldet,
		const data_size_t num_data,
		const int t) {

		Eigen::SelfAdjointEigenSolver<den_mat_t> es;
		ldet = 0;
		vec_t e1_logLambda_e1;

		for (int i = 0; i < t; ++i) {
			e1_logLambda_e1.setZero();
			es.computeFromTridiagonal(Tdiags[i], Tsubdiags[i]);
			e1_logLambda_e1 = es.eigenvectors().row(0).transpose().array() * es.eigenvalues().array().log() * es.eigenvectors().row(0).transpose().array();
			ldet += e1_logLambda_e1.sum();
		}
		ldet = ldet * num_data / t;
	} // end LogDetStochTridiag

	void CalcOptimalC(const vec_t& zt_AI_A_deriv_PI_z,
		const vec_t& zt_BI_B_deriv_PI_z,
		const double& tr_AI_A_deriv,
		const double& tr_BI_B_deriv, 
		double& c_opt) {

		vec_t centered_zt_AI_A_deriv_PI_z = zt_AI_A_deriv_PI_z.array() - tr_AI_A_deriv;
		vec_t centered_zt_BI_B_deriv_PI_z = zt_BI_B_deriv_PI_z.array() - tr_BI_B_deriv;
		c_opt = (centered_zt_AI_A_deriv_PI_z.cwiseProduct(centered_zt_BI_B_deriv_PI_z)).mean();
		c_opt /= (centered_zt_BI_B_deriv_PI_z.cwiseProduct(centered_zt_BI_B_deriv_PI_z)).mean();
	} // end CalcOptimalC

	void CalcOptimalCVectorized(const den_mat_t& Z_AI_A_deriv_PI_Z,
		const den_mat_t& Z_BI_B_deriv_PI_Z, 
		const vec_t& tr_AI_A_deriv,
		const vec_t& tr_BI_B_deriv,
		vec_t& c_opt) {

		den_mat_t centered_Z_AI_A_deriv_PI_Z = Z_AI_A_deriv_PI_Z.colwise() - tr_AI_A_deriv;
		den_mat_t centered_Z_BI_B_deriv_PI_Z = Z_BI_B_deriv_PI_Z.colwise() - tr_BI_B_deriv;
		vec_t c_cov = (centered_Z_AI_A_deriv_PI_Z.cwiseProduct(centered_Z_BI_B_deriv_PI_Z)).rowwise().mean();
		vec_t c_var = (centered_Z_BI_B_deriv_PI_Z.cwiseProduct(centered_Z_BI_B_deriv_PI_Z)).rowwise().mean();
		c_opt = c_cov.array() / c_var.array();
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < c_opt.size(); ++i) {
			if (c_var.coeffRef(i) == 0) {
				c_opt[i] = 1;
			}
		}
	} // end CalcOptimalCVectorized

	void ReverseIncompleteCholeskyFactorization(sp_mat_t& A,
		const sp_mat_t& /*B*/,
		sp_mat_rm_t& L_rm) {
		
		//Defining sparsity pattern 
		sp_mat_t L = A;
		//sp_mat_t L = B; //alternative version (less stable)
		
		L *= 0.0;

		////Debugging
		//Log::REInfo("L.nonZeros() = %d", L.nonZeros());
		//Log::REInfo("L.cwiseAbs().sum() = %g", L.cwiseAbs().sum());
		//den_mat_t A_dense = den_mat_t(A);
		//for (int c = 0; c < A_dense.cols(); ++c) {
		//	for (int r = 0; r < A_dense.rows(); ++r) {
		//		Log::REInfo("A_dense(%d,%d): %g", r, c, A_dense(r,c));
		//	}
		//}

		for (int i = ((int)L.outerSize() - 1); i > -1; --i) {
			for (Eigen::SparseMatrix<double>::ReverseInnerIterator it(L, i); it; --it) {
				int j = (int)it.row();
				int ii = (int)it.col();
				double s = (L.col(j)).dot(L.col(ii));
				if (ii == j) {
					it.valueRef() = std::sqrt(A.coeffRef(ii, ii) + 1e-10 - s);
				}
				else if (ii < j) {
					it.valueRef() = (A.coeffRef(ii, j) - s) / L.coeffRef(j, j);
				}
				if (std::isnan(it.value()) || std::isinf(it.value())) {
					//Log::REInfo("column i = %d (%d), row j = %d, value = %g", i, ii, j, it.value());
					//Log::REInfo("s = %g", s);
					//Log::REInfo("A(%d, %d): %g", ii, j, A.coeffRef(ii, j));
					Log::REFatal("nan or inf occured in ReverseIncompleteCholeskyFactorization()");
				}
			}
		}

		////Debugging
		//den_mat_t L_dense = den_mat_t(L);
		//for (int c = 0; c < L_dense.cols(); ++c) {
		//	for (int r = 0; r < L_dense.rows(); ++r) {
		//		Log::REInfo("L_dense(%d,%d): %g", r, c, L_dense(r, c));
		//	}
		//}
		//sp_mat_t Lt_L = L.transpose() * L;
		//sp_mat_t diff = A - Lt_L;
		//Log::REInfo("diff.cwiseAbs().sum() = %g", diff.cwiseAbs().sum());
		//den_mat_t Lt_L_dense = den_mat_t(Lt_L);
		//for (int c = 0; c < Lt_L_dense.cols(); ++c) {
		//	for (int r = 0; r < Lt_L_dense.rows(); ++r) {
		//		Log::REInfo("Lt_L_dense(%d,%d): %g", r, c, Lt_L_dense(r, c));
		//	}
		//}

		L_rm = sp_mat_rm_t(L); //Convert to row-major
	} // end ReverseIncompleteCholeskyFactorization
}