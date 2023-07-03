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
		const sp_mat_rm_t& D_inv_plus_W_B_rm) {

		p = std::min(p, (int)B_rm.cols());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h, v, B_invt_r;
		bool early_stop_alg = false;
		double a = 0;
		double b = 1;
		double r_squared_norm;
		
		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}

		//Cold-start in the first iteration of mode finding, otherwise always warm-start (=initalize with mode from previous iteration)
		if (find_mode_it == 0) {
			u.setZero();
		}

		//r = rhs - A * u
		r = rhs - ((B_t_D_inv_rm * (B_rm * u)) + diag_W.cwiseProduct(u));

		//z = P^(-1) r, where P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
		B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r);
		z = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);

		h = z;

		for (int j = 0; j < p; ++j) {
			//Parentheses are necessery for performance, otherwise EIGEN does the operation wrongly from left to right
			v = (B_t_D_inv_rm * (B_rm * h)) + diag_W.cwiseProduct(h);
			
			a = r.transpose() * z;
			a /= h.transpose() * v;

			u += a * h;
			r_old = r;
			r -= a * v;

			r_squared_norm = r.squaredNorm();
			//Log::REInfo("r.squaredNorm(): %g | Iteration: %i", r_squared_norm, j);
			if (std::isnan(r_squared_norm) || std::isinf(r_squared_norm)) {
				Log::REInfo("CGVecchiaLaplaceVec: NA_or_Inf_found");
				NA_or_Inf_found = true;
				return;
			}
			if (r_squared_norm < delta_conv) {
				Log::REInfo("Number CG iterations: %i", j);
				early_stop_alg = true;
			}

			z_old = z;

			//z = P^(-1) r 
			B_invt_r = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(r);
			z = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_r);

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;

			if (early_stop_alg) {
				return;
			}
		}
		Log::REInfo("CG has not converged after the maximal number of iterations. This could happen if the initial learning rate is too large. Otherwise increase 'cg_max_num_it'.");
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
		const den_mat_t& Sigma_L_k) {

		p = std::min(p, (int)B_rm.cols());

		vec_t r, r_old;
		vec_t z, z_old;
		vec_t h, v, diag_W_inv, B_invt_u, B_invt_h, B_invt_rhs, Sigma_Lkt_W_r, Sigma_rhs, W_r;
		bool early_stop_alg = false;
		double a = 0;
		double b = 1;
		double r_squared_norm;

		//Avoid numerical instabilites when rhs is de facto 0
		if (rhs.cwiseAbs().sum() < THRESHOLD_ZERO_RHS_CG) {
			u.setZero();
			return;
		}

		//Cold-start in the first iteration of mode finding, otherwise always warm-start (=initalize with mode from previous iteration)
		if (find_mode_it == 0) {
			u.setZero();
		}

		diag_W_inv = diag_W.cwiseInverse();

		//Sigma * rhs, where Sigma = B^(-1) D B^(-T)
		B_invt_rhs = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(rhs);
		Sigma_rhs = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_rhs);

		//r = Sigma * rhs - (W^(-1) + Sigma) * u
		B_invt_u = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(u);
		r = Sigma_rhs - (D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_u) + diag_W_inv.cwiseProduct(u));

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

			r_squared_norm = r.squaredNorm();
			if (std::isnan(r_squared_norm) || std::isinf(r_squared_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (r_squared_norm < delta_conv) {
				Log::REInfo("Number CG iterations: %i", j);
				early_stop_alg = true;
			}

			z_old = z;

			//z = P^(-1) r
			W_r = diag_W.asDiagonal() * r;
			Sigma_Lkt_W_r = Sigma_L_k.transpose() * W_r;
			z = W_r - diag_W.asDiagonal() * (Sigma_L_k * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia.solve(Sigma_Lkt_W_r));

			b = r.transpose() * z;
			b /= r_old.transpose() * z_old;

			h = z + b * h;

			if (early_stop_alg || (j + 1) == p) {
				//u = W^(-1) u
				u = diag_W_inv.cwiseProduct(u);
				if ((j + 1) == p) {
				 Log::REInfo("CG has not converged after the maximal number of iterations. This could happen if the initial learning rate is too large. Otherwise increase 'cg_max_num_it'.");
				}
				return;
			}
		}
	} // end CGVecchiaLaplaceVecSigmaplusWinv

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
		const sp_mat_rm_t& D_inv_plus_W_B_rm) {

		p = std::min(p, (int)num_data);

		den_mat_t R(num_data, t), R_old, B_invt_R(num_data, t), Z(num_data, t), Z_old, H, V(num_data, t), L_kt_W_inv_R, B_k_W_inv_R, W_inv_R; //NEW V(num_data, t)
		vec_t v1(num_data), diag_SigmaI_plus_W_inv, diag_W_inv;
		vec_t a(t), a_old(t);
		vec_t b(t), b_old(t);
		bool early_stop_alg = false;
		double mean_squared_R_norm;

		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//Parallelization in for loop is much faster
		//R = rhs - (Sigma^(-1) + W) U
//#pragma omp parallel for schedule(static)   
//		for (int i = 0; i < t; ++i) {
//			R.col(i) = rhs.col(i) - ((B_t_D_inv_rm * (B_rm * U.col(i))) + diag_W.cwiseProduct(U.col(i)));
//		}
		R = rhs;

		Log::REInfo("R.col(0).squaredNorm(): %g", R.col(0).squaredNorm());
		Log::REInfo("R.col(1).squaredNorm(): %g", R.col(1).squaredNorm());
		Log::REInfo("R.col(2).squaredNorm(): %g", R.col(2).squaredNorm());

		//Z = P^(-1) R 		
		//P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < t; ++i) {
			B_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
		}
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < t; ++i) {
			Z.col(i) = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_R.col(i));
		}
		Log::REInfo("Z.col(0).squaredNorm(): %g", Z.col(0).squaredNorm());
		Log::REInfo("Z.col(1).squaredNorm(): %g", Z.col(1).squaredNorm());
		Log::REInfo("Z.col(2).squaredNorm(): %g", Z.col(2).squaredNorm());

		H = Z;

		for (int j = 0; j < p; ++j) {
			//V = (Sigma^(-1) + W) H
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				V.col(i) = (B_t_D_inv_rm * (B_rm * H.col(i))) + diag_W.cwiseProduct(H.col(i));
			}

			Log::REInfo("V.col(0).squaredNorm(): %g", V.col(0).squaredNorm());
			Log::REInfo("V.col(1).squaredNorm(): %g", V.col(1).squaredNorm());
			Log::REInfo("V.col(2).squaredNorm(): %g", V.col(2).squaredNorm());

			a_old = a;
			a = (R.cwiseProduct(Z).transpose() * v1).array() * (H.cwiseProduct(V).transpose() * v1).array().inverse(); //cheap

			Log::REInfo("a[0]: %g", a[0]);
			Log::REInfo("a[1]: %g", a[1]);
			Log::REInfo("a[2]: %g", a[2]);
			Log::REInfo("a[3]: %g", a[3]);

			U += H * a.asDiagonal();
			R_old = R;
			R -= V * a.asDiagonal();

			mean_squared_R_norm = 0;        
			for (int i = 0; i < t; ++i) {
				mean_squared_R_norm += R.col(i).squaredNorm();
				Log::REInfo("R.col(i).squaredNorm(): %g", R.col(i).squaredNorm());
			}
			mean_squared_R_norm /= t;
			Log::REInfo("mean_squared_R_norm old: %g", mean_squared_R_norm);
			mean_squared_R_norm = R.colwise().squaredNorm().mean();
			Log::REInfo("mean_squared_R_norm new: %g", mean_squared_R_norm);
			if (std::isnan(mean_squared_R_norm) || std::isinf(mean_squared_R_norm)) {
				Log::REInfo("CGTridiagVecchiaLaplace: NA_or_Inf_found at iteration: %i", j);
				NA_or_Inf_found = true;
				return;
			}
			if (mean_squared_R_norm < delta_conv) {
				early_stop_alg = true;
				Log::REInfo("Number CG-Tridiag iterations: %i", j);
			}

			Z_old = Z;

			//Z = P^(-1) R
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				B_invt_R.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(R.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < t; ++i) {
				Z.col(i) = D_inv_plus_W_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_R.col(i));
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
		Log::REInfo("CG has not converged after the maximal number of iterations. This could happen if the initial learning rate is too large. Otherwise increase 'cg_max_num_it_tridiag'.");
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
		double mean_squared_R_norm;

		diag_W_inv = diag_W.cwiseInverse();
		U.setZero();
		v1.setOnes();
		a.setOnes();
		b.setZero();

		//R = rhs - (W^(-1) + Sigma) * U
		//Drastic performance increase through paralellization
//#pragma omp parallel for schedule(static)   
//		for (int i = 0; i < t; ++i) {
//			B_invt_U.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(U.col(i));
//		}
//#pragma omp parallel for schedule(static)   
//		for (int i = 0; i < t; ++i) {
//			R.col(i) = rhs.col(i) - D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_U.col(i));
//		}
//		R -= diag_W_inv.replicate(1, t).cwiseProduct(U);
		R = rhs;

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

//			mean_squared_R_norm = 0;
//#pragma omp parallel for schedule(static)            
//			for (int i = 0; i < t; ++i) {
//				mean_squared_R_norm += R.col(i).squaredNorm();
//			}
//			mean_squared_R_norm /= t;
			mean_squared_R_norm = R.colwise().squaredNorm().mean();

			if (std::isnan(mean_squared_R_norm) || std::isinf(mean_squared_R_norm)) {
				NA_or_Inf_found = true;
				return;
			}
			if (mean_squared_R_norm < delta_conv) {
				early_stop_alg = true;
				Log::REInfo("Number CG-Tridiag iterations: %i", j);
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
		Log::REInfo("CG has not converged after the maximal number of iterations. This could happen if the initial learning rate is too large. Otherwise increase 'cg_max_num_it_tridiag'.");
	} // end CGTridiagVecchiaLaplaceSigmaplusWinv

	void GenRandVecTrace(RNG_t& generator, 
		den_mat_t& R) {
		
		std::normal_distribution<double> ndist(0.0, 1.0);
		//Do not parallelize! - Despite seed: no longer deterministic
		for (int i = 0; i < R.rows(); ++i) {
			for (int j = 0; j < R.cols(); j++) {
				R(i, j) = ndist(generator);
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
	} // end CalcOptimalCVectorized

	void LanczosTridiagVecchiaLaplace(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_rm,
		const vec_t& b_init,
		const data_size_t num_data,
		vec_t& Tdiag_k,
		vec_t& Tsubdiag_k,
		den_mat_t& P_t_sqrt_inv_Q_k,
		int max_it,
		const double tol) {

		bool could_reorthogonalize;
		int final_rank = 1;
		double alpha_curr, beta_curr, beta_prev;
		vec_t q_curr, q_prev, inner_products;

		max_it = std::min(max_it, num_data);

		//Preconditioning
		vec_t D_inv_plus_W_diag = D_inv_rm.diagonal() + diag_W;
		sp_mat_rm_t B_t_D_inv_plus_W_sqrt_rm = B_rm.transpose() * D_inv_plus_W_diag.cwiseSqrt().asDiagonal(); //B^T (D^(-1) + W)^(0.5)
		vec_t D_inv_plus_W_inv_D_inv_diag = D_inv_plus_W_diag.cwiseInverse().array() * D_inv_rm.diagonal().array(); //diagonal of (D^(-1) + W)^(-1) D^(-1)

		//Inital vector of Q_k: q_0
		den_mat_t Q_k(num_data, max_it);
		vec_t q_0 = b_init / b_init.norm();
		Q_k.col(0) = q_0;

		//Initial alpha value: alpha_0
		//P^(-0.5) (Sigma^-1+W) P^(-0.5*T) q_0 = (D^(-1) + W)^(-1) D^(-1) q_0 + (D^(-1) + W)^(-0.5) B^(-T) W B^(-1) (D^(-1) + W)^(-0.5) q_0
		vec_t B_inv_D_inv_plus_W_sqrt_inv_q = B_t_D_inv_plus_W_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(q_0);
		vec_t W_B_inv_D_inv_plus_W_sqrt_inv_q = diag_W.cwiseProduct(B_inv_D_inv_plus_W_sqrt_inv_q);
		vec_t r = B_t_D_inv_plus_W_sqrt_rm.triangularView<Eigen::UpLoType::Upper>().solve(W_B_inv_D_inv_plus_W_sqrt_inv_q);
		r += D_inv_plus_W_inv_D_inv_diag.cwiseProduct(q_0);
		double alpha_0 = q_0.dot(r);
		
		//Initial beta value: beta_0
		r -= alpha_0 * q_0;
		double beta_0 = r.norm();

		//Store alpha_0 and beta_0 into T_k
		Tdiag_k(0) = alpha_0;
		Tsubdiag_k(0) = beta_0;

		//Compute next vector of Q_k: q_1
		Q_k.col(1) = r / beta_0;

		//Start the iterations
		for (int k = 1; k < max_it; ++k) {

			//Get previous values
			q_prev = Q_k.col(k-1);
			q_curr = Q_k.col(k);
			beta_prev = Tsubdiag_k(k-1);

			//Compute next alpha value
			B_inv_D_inv_plus_W_sqrt_inv_q = B_t_D_inv_plus_W_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(q_curr);
			W_B_inv_D_inv_plus_W_sqrt_inv_q = diag_W.cwiseProduct(B_inv_D_inv_plus_W_sqrt_inv_q);
			r = B_t_D_inv_plus_W_sqrt_rm.triangularView<Eigen::UpLoType::Upper>().solve(W_B_inv_D_inv_plus_W_sqrt_inv_q);
			r += D_inv_plus_W_inv_D_inv_diag.cwiseProduct(q_curr) - beta_prev * q_prev;
			alpha_curr = q_curr.dot(r);

			//Store alpha_curr
			Tdiag_k(k) = alpha_curr;
			final_rank += 1;

			if ((k+1) < max_it) {

				//Compute next residual
				r -= alpha_curr * q_curr;

				//Full reorthogonalization: r = r - Q_k (Q_k' r)
				r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);
				
				//Compute next beta value
				beta_curr = r.norm();
				Tsubdiag_k(k) = beta_curr;
				
				r /= beta_curr;

				//More reorthogonalizations if necessary
				inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				could_reorthogonalize = false;
				for (int l = 0; l < 10; ++l) {
					if ((inner_products.array() < tol).all()) {
						could_reorthogonalize = true;
						break;
					}
					r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);
					r /= r.norm();
					inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				}

				//Store next vector of Q_k
				Q_k.col(k + 1) = r;

				if (abs(beta_curr) < 1e-6 || !could_reorthogonalize) {
					break;
				}
			}
		}
		
		//Resize Q_k, Tdiag_k, Tsubdiag_k
		Q_k.conservativeResize(num_data, final_rank);
		Tdiag_k.conservativeResize(final_rank, 1);
		Tsubdiag_k.conservativeResize(final_rank-1, 1);

		//Adjustment for preconditioning: P^(-0.5*T) Q_k = B^(-1) (D^(-1) + W)^(-0.5) Q_k
		P_t_sqrt_inv_Q_k.resize(num_data, final_rank);
#pragma omp parallel for schedule(static)   
		for (int m = 0; m < final_rank; ++m) {
			P_t_sqrt_inv_Q_k.col(m) = B_t_D_inv_plus_W_sqrt_rm.transpose().triangularView<Eigen::UpLoType::Lower>().solve(Q_k.col(m));
		}
	} // end LanczosTridiagVecchiaLaplace

	void LanczosTridiagVecchiaLaplaceWinvplusSigma(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& D_inv_B_rm,
		const vec_t& b_init,
		const data_size_t num_data,
		vec_t& Tdiag_k,
		vec_t& Tsubdiag_k,
		den_mat_t& W_inv_P_sqrt_inv_Q_k,
		den_mat_t& Sigma_P_sqrt_inv_Q_k,
		int max_it,
		const double tol,
		const double sigma2) {

		bool could_reorthogonalize;
		int final_rank = 1;
		double alpha_curr, beta_curr, beta_prev;
		vec_t diag_W_inv = diag_W.cwiseInverse();
		vec_t q_curr, q_prev, inner_products;

		max_it = std::min(max_it, num_data);

		//Preconditioning: P = diag(W^(-1) + Sigma)
		vec_t P_sqrt_inv = (sigma2 + diag_W_inv.array()).cwiseInverse().cwiseSqrt();

		//Inital vector of Q_k: q_0
		den_mat_t Q_k(num_data, max_it);
		vec_t q_0 = b_init / b_init.norm();
		Q_k.col(0) = q_0;

		//Initial alpha value: alpha_0
		//P^(-0.5) (W^(-1) + Sigma) P^(-0.5) q_0
		vec_t P_sqrt_inv_q = P_sqrt_inv.cwiseProduct(q_0);
		vec_t B_invt_P_sqrt_inv_q = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(P_sqrt_inv_q);
		vec_t W_inv_plus_Sigma_P_sqrt_inv_q = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_P_sqrt_inv_q);
		W_inv_plus_Sigma_P_sqrt_inv_q += diag_W_inv.cwiseProduct(P_sqrt_inv_q);
		vec_t r = P_sqrt_inv.cwiseProduct(W_inv_plus_Sigma_P_sqrt_inv_q);
		double alpha_0 = q_0.dot(r);

		//Initial beta value: beta_0
		r -= alpha_0 * q_0;
		double beta_0 = r.norm();

		//Store alpha_0 and beta_0 into T_k
		Tdiag_k(0) = alpha_0;
		Tsubdiag_k(0) = beta_0;

		//Compute next vector of Q_k: q_1
		Q_k.col(1) = r / beta_0;

		//Start the iterations
		for (int k = 1; k < max_it; ++k) {

			//Get previous values
			q_prev = Q_k.col(k - 1);
			q_curr = Q_k.col(k);
			beta_prev = Tsubdiag_k(k - 1);

			//Compute next alpha value
			P_sqrt_inv_q = P_sqrt_inv.cwiseProduct(q_curr);
			B_invt_P_sqrt_inv_q = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(P_sqrt_inv_q);
			W_inv_plus_Sigma_P_sqrt_inv_q = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_P_sqrt_inv_q);
			W_inv_plus_Sigma_P_sqrt_inv_q += diag_W_inv.cwiseProduct(P_sqrt_inv_q);
			r = P_sqrt_inv.cwiseProduct(W_inv_plus_Sigma_P_sqrt_inv_q);
			r -= beta_prev * q_prev;
			alpha_curr = q_curr.dot(r);

			//Store alpha_curr
			Tdiag_k(k) = alpha_curr;
			final_rank += 1;

			if ((k + 1) < max_it) {

				//Compute next residual
				r -= alpha_curr * q_curr;

				//Full reorthogonalization: r = r - Q_k (Q_k' r)
				r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);

				//Compute next beta value
				beta_curr = r.norm();
				Tsubdiag_k(k) = beta_curr;

				r /= beta_curr;

				//More reorthogonalizations if necessary
				inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				could_reorthogonalize = false;
				for (int l = 0; l < 10; ++l) {
					if ((inner_products.array() < tol).all()) {
						could_reorthogonalize = true;
						break;
					}
					r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);
					r /= r.norm();
					inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				}

				//Store next vector of Q_k
				Q_k.col(k + 1) = r;

				if (abs(beta_curr) < 1e-6 || !could_reorthogonalize) {
					break;
				}
			}
		}

		//Resize Q_k, Tdiag_k, Tsubdiag_k
		Q_k.conservativeResize(num_data, final_rank);
		Tdiag_k.conservativeResize(final_rank, 1);
		Tsubdiag_k.conservativeResize(final_rank - 1, 1);

		//Adjustment: P^(-0.5) Q_k
		Q_k = P_sqrt_inv.asDiagonal() * Q_k;

		//Adjustment: W^(-1) P^(-0.5) Q_k
		W_inv_P_sqrt_inv_Q_k.resize(num_data, final_rank);
		W_inv_P_sqrt_inv_Q_k = diag_W_inv.asDiagonal() * Q_k;

		//Adjustment: Sigma P^(-0.5) Q_k
		Sigma_P_sqrt_inv_Q_k.resize(num_data, final_rank);
		den_mat_t B_invt_P_sqrt_inv_Q_k(num_data, final_rank);
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < final_rank; ++i) {
			B_invt_P_sqrt_inv_Q_k.col(i) = B_rm.transpose().triangularView<Eigen::UpLoType::UnitUpper>().solve(Q_k.col(i));
		}
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < final_rank; ++i) {
			Sigma_P_sqrt_inv_Q_k.col(i) = D_inv_B_rm.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_P_sqrt_inv_Q_k.col(i));
		}
	} // end LanczosTridiagVecchiaLaplaceWinvplusSigma

	void LanczosTridiagVecchiaLaplaceNoPreconditioner(const vec_t& diag_W,
		const sp_mat_rm_t& B_rm,
		const sp_mat_rm_t& B_t_D_inv_rm,
		const vec_t& b_init,
		const data_size_t num_data,
		vec_t& Tdiag_k,
		vec_t& Tsubdiag_k,
		den_mat_t& Q_k,
		int max_it,
		const double tol) {

		bool could_reorthogonalize;
		int final_rank = 1;
		double alpha_curr, beta_curr, beta_prev;
		vec_t q_curr, q_prev, inner_products;

		max_it = std::min(max_it, num_data);

		//Inital vector of Q_k: q_0
		Q_k.resize(num_data, max_it);
		vec_t q_0 = b_init / b_init.norm();
		Q_k.col(0) = q_0;

		//Initial alpha value: alpha_0
		//(Sigma^-1+W) q_0
		vec_t r = B_t_D_inv_rm * (B_rm * q_0) + diag_W.cwiseProduct(q_0);
		double alpha_0 = q_0.dot(r);

		//Initial beta value: beta_0
		r -= alpha_0 * q_0;
		double beta_0 = r.norm();

		//Store alpha_0 and beta_0 into T_k
		Tdiag_k(0) = alpha_0;
		Tsubdiag_k(0) = beta_0;

		//Compute next vector of Q_k: q_1
		Q_k.col(1) = r / beta_0;

		//Start the iterations
		for (int k = 1; k < max_it; ++k) {

			//Get previous values
			q_prev = Q_k.col(k - 1);
			q_curr = Q_k.col(k);
			beta_prev = Tsubdiag_k(k - 1);

			//Compute next alpha value
			r = B_t_D_inv_rm * (B_rm * q_curr) + diag_W.cwiseProduct(q_curr) - beta_prev * q_prev;
			alpha_curr = q_curr.dot(r);

			//Store alpha_curr
			Tdiag_k(k) = alpha_curr;
			final_rank += 1;

			if ((k + 1) < max_it) {

				//Compute next residual
				r -= alpha_curr * q_curr;

				//Full reorthogonalization: r = r - Q_k (Q_k' r)
				r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);

				//Compute next beta value
				beta_curr = r.norm();
				Tsubdiag_k(k) = beta_curr;

				r /= beta_curr;

				//More reorthogonalizations if necessary
				inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				could_reorthogonalize = false;
				for (int l = 0; l < 10; ++l) {
					if ((inner_products.array() < tol).all()) {
						could_reorthogonalize = true;
						break;
					}
					r -= Q_k(Eigen::all, Eigen::seq(0, k)) * (Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r);
					r /= r.norm();
					inner_products = Q_k(Eigen::all, Eigen::seq(0, k)).transpose() * r;
				}

				//Store next vector of Q_k
				Q_k.col(k + 1) = r;

				if (abs(beta_curr) < 1e-6 || !could_reorthogonalize) {
					break;
				}
			}
		}

		//Resize Q_k, Tdiag_k, Tsubdiag_k
		Q_k.conservativeResize(num_data, final_rank);
		Tdiag_k.conservativeResize(final_rank, 1);
		Tsubdiag_k.conservativeResize(final_rank - 1, 1);
	} // end LanczosTridiagVecchiaLaplaceNoPreconditioner
}