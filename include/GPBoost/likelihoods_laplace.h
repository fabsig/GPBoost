/*!
* This file is part of GPBoost a C++ library for combining
*   boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2026 Fabio Sigrist, Tim Gyger, and Pascal Kuendig. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*
* Definitions of the member functions of the 'Likelihood' class that implement the Laplace approximation
* itself: the mode finding, the gradients of the approximate marginal log-likelihood, the predictive
* distributions, the posterior sampling, and the stochastic (iterative) log-determinant estimators, for
* every supported random effects structure / matrix approximation:
*		- "Stable": the numerically stable version of Rasmussen and Williams (2006)
*		- "GroupedRE" / "OnlyOneGroupedRECalculationsOnREScale": grouped random effects
*		- "Vecchia": a Vecchia approximated GP
*		- "FITC": a fully independent training conditional (modified predictive process) approximation
*		- "FSVA": a full-scale approximation
* The likelihood definitions themselves (the per-observation log-likelihood, its derivatives, the
* information, the normalizing constants, and the auxiliary parameters) remain in 'likelihoods.h', where
* all functions defined here are also declared and documented.
*
* NOTE: this file is included at the end of 'likelihoods.h' and cannot be compiled on its own.
*/
#ifndef GPB_LIKELIHOODS_LAPLACE_H_
#define GPB_LIKELIHOODS_LAPLACE_H_

namespace GPBoost {

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::FindModePostRandEffCalcMLLStable(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const std::shared_ptr<T_mat>& Sigma,
		double& approx_marginal_ll) {
		ChecksBeforeModeFinding();
		fixed_effects_ = fixed_effects;
		// Initialize variables
		if (InitializeModeForModeFinding()) {
			SigmaI_mode_previous_value_ = SigmaI_mode_;
			mode_ = (*Sigma) * SigmaI_mode_;//initialize mode with Sigma^(t+1) * a = Sigma^(t+1) * (Sigma^t)^(-1) * mode^t, where t+1 = current iteration. Otherwise the initial approx_marginal_ll is not correct since SigmaI_mode != Sigma^(-1)mode
			// The alternative way of intializing SigmaI_mode_ = Sigma^(-1) mode_ requires an additional linear solve
			//T_mat Sigma_stable = (*Sigma);
			//Sigma_stable.diagonal().array() *= JITTER_MUL;
			//T_chol chol_fact_Sigma;
			//CalcChol<T_mat>(chol_fact_Sigma, Sigma_stable, chol_fact_pattern_analyzed_);
			//SigmaI_mode_ = chol_fact_Sigma.solve(mode_);
		}
		T_chol chol_fact_Sigma;
		if (kink_cliping_) {
			T_mat Sigma_stable = (*Sigma);
			Sigma_stable.diagonal().array() *= JITTER_MUL;
			CalcChol<T_mat>(chol_fact_Sigma, Sigma_stable, chol_fact_pattern_analyzed_);
		}
		vec_t location_par;//location parameter = mode of random effects + fixed effects
		double* location_par_ptr;
		InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
		// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
		approx_marginal_ll = -0.5 * (SigmaI_mode_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par_ptr);
		double approx_marginal_ll_new = approx_marginal_ll;
		vec_t rhs(dim_mode_), rhs2(dim_mode_), mode_new, SigmaI_mode_new, mode_update, SigmaI_mode_update;//auxiliary variables for updating mode
		vec_t diag_Wsqrt(dim_mode_);//diagonal of matrix sqrt(ZtWZ) if use_random_effects_indices_of_data_ or sqrt(W) if !use_random_effects_indices_of_data_ with square root of negative second derivatives of log-likelihood
		T_mat Id_plus_Wsqrt_Sigma_Wsqrt(dim_mode_, dim_mode_);// = Id_plus_ZtWZsqrt_Sigma_ZtWZsqrt if use_random_effects_indices_of_data_ or Id_plus_Wsqrt_ZSigmaZt_Wsqrt if !use_random_effects_indices_of_data_
		// Start finding mode 
		int it;
		bool terminate_optim = false;
		bool has_NA_or_Inf = false;
		for (it = 0; it < maxit_mode_newton_; ++it) {
			// Calculate first and second derivative of log-likelihood
			CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);
			// Calculate Cholesky factor of matrix B = (Id + ZtWZsqrt * Sigma * ZtWZsqrt) if use_random_effects_indices_of_data_ or B = (Id + Wsqrt * Z*Sigma*Zt * Wsqrt) if !use_random_effects_indices_of_data_
			if (it == 0 || information_changes_during_mode_finding_) {
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, true);
				if (HasNegativeValueInformationLogLik()) {
					LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in the (diagonal) Hessian (or Fisher information) "
						"of the negative log-likelihood. Cannot have negative values when using the numerically stable "
						"version of Rasmussen and Williams(2006) for mode finding ");
				}
				diag_Wsqrt.array() = information_ll_.array().sqrt();
				Id_plus_Wsqrt_Sigma_Wsqrt.setIdentity();
				Id_plus_Wsqrt_Sigma_Wsqrt += (diag_Wsqrt.asDiagonal() * (*Sigma) * diag_Wsqrt.asDiagonal());
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_Wsqrt_Sigma_Wsqrt, chol_fact_pattern_analyzed_);//this is the bottleneck (for large data and sparse matrices)
			}
			// Calculate right hand side for mode update
			rhs.array() = information_ll_.array() * mode_.array() + first_deriv_ll_.array();
			// Update mode and SigmaI_mode_
			rhs2 = (*Sigma) * rhs;//rhs2 = sqrt(W) * Sigma * rhs
			rhs2.array() *= diag_Wsqrt.array();
			// Backtracking line search
			SigmaI_mode_update = -chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_.solve(rhs2);//SigmaI_mode_ = rhs - sqrt(W) * Id_plus_Wsqrt_Sigma_Wsqrt^-1 * rhs2
			SigmaI_mode_update.array() *= diag_Wsqrt.array();
			SigmaI_mode_update.array() += rhs.array();
			mode_update = (*Sigma) * SigmaI_mode_update;
			double lr_mode = 1.;
			double grad_dot_direction = 0.;//for Armijo check
			if (armijo_condition_) {
				vec_t direction = mode_update - mode_;//mode_update = "new mode" = (Sigma^-1 + W)^-1 (W * mode + grad p(y|mode)) -> direction = mode_update - mode_
				grad_dot_direction = direction.dot(SigmaI_mode_update - SigmaI_mode_ + information_ll_.asDiagonal() * direction); // gradient = (Sigma^-1 + W) * direction, SigmaI_mode_update - SigmaI_mode_ = Sigma^-1 direction
			}
			for (int ih = 0; ih < max_number_lr_shrinkage_steps_newton_; ++ih) {
				if (ih == 0) {
					SigmaI_mode_new = SigmaI_mode_update;
					mode_new = mode_update;//mode_update = "new mode" = (Sigma^-1 + W)^-1 (W * mode + grad p(y|mode))
				}
				else {
					SigmaI_mode_new = (1 - lr_mode) * SigmaI_mode_ + lr_mode * SigmaI_mode_update;
					mode_new = (1 - lr_mode) * mode_ + lr_mode * mode_update;
				}
				if (kink_cliping_) {
					bool clipped = ApplyKinkClippingAsymLaplace(y_data, fixed_effects, mode_, mode_new);
					if (clipped) SigmaI_mode_new = chol_fact_Sigma.solve(mode_new);// Recompute Sigma^{-1} mode_new so the quadratic term is consistent
				}
				UpdateLocationParNewMode(mode_new, fixed_effects, location_par, &location_par_ptr); // Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
				approx_marginal_ll_new = -0.5 * (SigmaI_mode_new.dot(mode_new)) + LogLikelihood(y_data, y_data_int, location_par_ptr);// Calculate new objective function
				if (AcceptModeUpdate(approx_marginal_ll_new, approx_marginal_ll, grad_dot_direction, lr_mode)) {
					break;
				}
			}// end loop over learnig rate halving procedure
			mode_ = mode_new;
			SigmaI_mode_ = SigmaI_mode_new;
			CheckConvergenceModeFinding(it, approx_marginal_ll_new, approx_marginal_ll, terminate_optim, has_NA_or_Inf);
			//Log::REInfo("it = %d, mode_[0:2] = %g, %g, %g, LogLikelihood = %g", it, mode_[0], mode_[1], mode_[2], LogLikelihood(y_data, y_data_int, location_par_ptr));//for debugging
			if (terminate_optim || has_NA_or_Inf) {
				break;
			}
		}
		if (!has_NA_or_Inf) {//calculate determinant
			if (sample_from_posterior_after_mode_finding_) {
				Sample_Posterior_LaplaceApprox_Stable(Sigma);
			}
			CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
			if (information_changes_after_mode_finding_) {
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, false);
				if (HasNegativeValueInformationLogLik()) {
					LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in the (diagonal) Hessian (or Fisher information) "
						"of the negative log-likelihood. Cannot have negative values when using the numerically stable "
						"version of Rasmussen and Williams (2006) for calculating the log-determinant in the log-marginal likelihood "
						"");
				}
				diag_Wsqrt.array() = information_ll_.array().sqrt();
				Id_plus_Wsqrt_Sigma_Wsqrt.setIdentity();
				Id_plus_Wsqrt_Sigma_Wsqrt += (diag_Wsqrt.asDiagonal() * (*Sigma) * diag_Wsqrt.asDiagonal());
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_Wsqrt_Sigma_Wsqrt, chol_fact_pattern_analyzed_);
			}
			approx_marginal_ll -= ((T_mat)chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_.matrixL()).diagonal().array().log().sum();
			mode_has_been_calculated_ = true;
			na_or_inf_during_last_call_to_find_mode_ = false;
		}
		FinalizeModeFinding(it);
		//Log::REInfo("FindModePostRandEffCalcMLLStable: finished after %d iterations ", it);//for debugging
		//Log::REInfo("mode_[0:2] = %g, %g, %g, LogLikelihood = %g", mode_[0], mode_[1], mode_[2], LogLikelihood(y_data, y_data_int, location_par_ptr));//for debugging
		//Log::REInfo("it = %d, first_deriv_ll_[0:2] = %g, %g, %g, information_ll_[0:2] = %g, %g, %g", it,
		//	first_deriv_ll_[0], first_deriv_ll_[1], first_deriv_ll_[2], information_ll_[0], information_ll_[1], information_ll_[2]);//for debugging
	}//end FindModePostRandEffCalcMLLStable

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::FindModePostRandEffCalcMLLGroupedRE(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const sp_mat_t& SigmaI,
		bool has_vecchia_gp,
		const sp_mat_t& B,
		const sp_mat_t& D_inv,
		const bool first_update,
		bool calc_mll,
		double& approx_marginal_ll) {
		ChecksBeforeModeFinding();
		CHECK(!kink_cliping_);
		fixed_effects_ = fixed_effects;
		// Initialize variables
		InitializeModeForModeFinding();
		vec_t location_par;
		double* location_par_ptr_dummy;//not used
		UpdateLocationParNewMode(mode_, fixed_effects, location_par, &location_par_ptr_dummy);
		const sp_mat_t* SigmaI_ptr = &SigmaI;
		sp_mat_t SigmaI_re_gp;//precision for grouped REs + Vecchia GP
		if (has_vecchia_gp) {
			sp_mat_t SigmaI_gp = B.transpose() * D_inv * B;
			GPBoost::MakeBlockDiag_D_B<sp_mat_t>(SigmaI, SigmaI_gp, SigmaI_re_gp);
			SigmaI_ptr = &SigmaI_re_gp;
		}
		CHECK((*SigmaI_ptr).rows() == dim_mode_);
		// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
		approx_marginal_ll = -0.5 * (mode_.dot((*SigmaI_ptr) * mode_)) + LogLikelihood(y_data, y_data_int, location_par.data());
		double approx_marginal_ll_new = approx_marginal_ll;
		sp_mat_t SigmaI_plus_ZtWZ;
		vec_t rhs, mode_update(dim_mode_), mode_new;
		// Variables when using iterative methods
		int cg_max_num_it = cg_max_num_it_;
		int cg_max_num_it_tridiag = cg_max_num_it_tridiag_;
		//Reduce max. number of iterations for the CG in first update
		if (matrix_inversion_method_ == "iterative" && first_update && reduce_cg_max_num_it_first_optim_step_) {
			cg_max_num_it = (int)round(cg_max_num_it_ / 3);
			cg_max_num_it_tridiag = (int)round(cg_max_num_it_tridiag_ / 3);
		}
		// Start finding mode 
		int it;
		bool terminate_optim = false;
		bool has_NA_or_Inf = false;
		if (save_SigmaI_mode_) {
			SigmaI_mode_ = (*Zt_).transpose() * ((*SigmaI_ptr) * mode_);
		}
		for (it = 0; it < maxit_mode_newton_; ++it) {
			// Calculate first and second derivative of log-likelihood
			CalcFirstDerivLogLik(y_data, y_data_int, location_par.data());
			rhs = (*Zt_) * first_deriv_ll_ - (*SigmaI_ptr) * mode_;//right hand side for updating mode
			if (matrix_inversion_method_ == "iterative") {
				if (it == 0 || information_changes_after_mode_finding_) {
					CalcInformationLogLik(y_data, y_data_int, location_par.data(), true);
					SigmaI_plus_ZtWZ_rm_ = sp_mat_rm_t((*SigmaI_ptr)) + sp_mat_rm_t((*Zt_) * information_ll_.asDiagonal() * (*Zt_).transpose());
					if (cg_preconditioner_type_ == "incomplete_cholesky") {
						ZeroFillInIncompleteCholeskyFactorization(SigmaI_plus_ZtWZ_rm_, L_SigmaI_plus_ZtWZ_rm_);
					}
					else if (cg_preconditioner_type_ == "ssor") {
						P_SSOR_D_inv_ = SigmaI_plus_ZtWZ_rm_.diagonal().cwiseInverse();
						vec_t P_SSOR_D_inv_sqrt = P_SSOR_D_inv_.cwiseSqrt(); //need to store this, otherwise slow!
						sp_mat_rm_t P_SSOR_L_rm = SigmaI_plus_ZtWZ_rm_.triangularView<Eigen::Lower>();
						P_SSOR_L_D_sqrt_inv_rm_ = P_SSOR_L_rm * P_SSOR_D_inv_sqrt.asDiagonal();
					}
					else if (cg_preconditioner_type_ == "diagonal") {
						SigmaI_plus_ZtWZ_inv_diag_ = SigmaI_plus_ZtWZ_rm_.diagonal().cwiseInverse();
					}
				}
				int num_cg_steps;
				CGRandomEffectsVec(SigmaI_plus_ZtWZ_rm_, rhs, mode_update, has_NA_or_Inf,
					cg_max_num_it, cg_delta_conv_, it == 0, ZERO_RHS_CG_THRESHOLD, false, cg_preconditioner_type_,
					L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps);
				if (it == 0) {
					num_cg_steps_last_ = num_cg_steps;
				}
				if (has_NA_or_Inf) {
					approx_marginal_ll_new = std::numeric_limits<double>::quiet_NaN();
					Log::REDebug(NA_OR_INF_WARNING_);
					break;
				}
			} //end iterative
			else if (matrix_inversion_method_ == "cholesky") { // start Cholesky 
				// Calculate Cholesky factor
				if (it == 0 || information_changes_during_mode_finding_) {
					CalcInformationLogLik(y_data, y_data_int, location_par.data(), true);
					SigmaI_plus_ZtWZ = (*SigmaI_ptr) + (sp_mat_t)((*Zt_) * information_ll_.asDiagonal() * (*Zt_).transpose());
					SigmaI_plus_ZtWZ.makeCompressed();
					if (!chol_fact_pattern_analyzed_) {
						chol_fact_SigmaI_plus_ZtWZ_grouped_.analyzePattern(SigmaI_plus_ZtWZ);
						chol_fact_pattern_analyzed_ = true;
					}
					chol_fact_SigmaI_plus_ZtWZ_grouped_.factorize(SigmaI_plus_ZtWZ);
					if (chol_fact_SigmaI_plus_ZtWZ_grouped_.info() != Eigen::Success) {
						LogFatalWithPotentialFisherLaplaceHint(__func__, "Cholesky factorization of Sigma^(-1) + Z^T W Z failed. "
							"The complete mode-finding curvature matrix is not positive definite ");
					}
				}
				// Update mode and do backtracking line search
				mode_update = chol_fact_SigmaI_plus_ZtWZ_grouped_.solve(rhs);
			} // end Cholesky
			double grad_dot_direction = 0.;//for Armijo check
			if (armijo_condition_) {
				grad_dot_direction = mode_update.dot(rhs);//rhs = gradient of objective
			}
			// Backtracking line search
			double lr_mode = 1.;
			for (int ih = 0; ih < max_number_lr_shrinkage_steps_newton_; ++ih) {
				mode_new = mode_ + lr_mode * mode_update;
				// Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
				UpdateLocationParNewMode(mode_new, fixed_effects, location_par, &location_par_ptr_dummy);
				approx_marginal_ll_new = -0.5 * (mode_new.dot((*SigmaI_ptr) * mode_new)) + LogLikelihood(y_data, y_data_int, location_par.data());// Calculate new objective function
				if (AcceptModeUpdate(approx_marginal_ll_new, approx_marginal_ll, grad_dot_direction, lr_mode)) {
					break;
				}
			}// end loop over learnig rate halving procedure
			mode_ = mode_new;
			CheckConvergenceModeFinding(it, approx_marginal_ll_new, approx_marginal_ll, terminate_optim, has_NA_or_Inf);
			if (terminate_optim || has_NA_or_Inf) {
				break;
			}
		}//end mode finding algorithm
		if (!has_NA_or_Inf) {//calculate determinant
			if (sample_from_posterior_after_mode_finding_) {
				Sample_Posterior_LaplaceApprox_GroupedRE(SigmaI, has_vecchia_gp, B, D_inv);
			}
			CalcFirstDerivLogLik(y_data, y_data_int, location_par.data());//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
			if (matrix_inversion_method_ == "iterative") {
				if (calc_mll) {//calculate determinant term for approx_marginal_ll
					//Generate random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I
					if (!saved_rand_vec_trace_) {
						//Generate t (= num_rand_vec_trace_) random vectors
						rand_vec_trace_I_.resize(dim_mode_, num_rand_vec_trace_);
						GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_trace_I_);
						if (reuse_rand_vec_trace_) {
							saved_rand_vec_trace_ = true;
						}
						rand_vec_trace_P_.resize(dim_mode_, num_rand_vec_trace_);
						SigmaI_plus_ZtWZ_inv_RV_.resize(dim_mode_, num_rand_vec_trace_);
					}
					double log_det_SigmaI_plus_ZtWZ;
					//Stochastic Lanczos quadrature
					CHECK(rand_vec_trace_I_.cols() == num_rand_vec_trace_);
					CHECK(rand_vec_trace_P_.cols() == num_rand_vec_trace_);
					CHECK(rand_vec_trace_I_.rows() == dim_mode_);
					CHECK(rand_vec_trace_P_.rows() == dim_mode_);
					if (information_changes_after_mode_finding_) {
						//upadate with latest W
						CalcInformationLogLik(y_data, y_data_int, location_par.data(), false);
						if (HasNegativeValueInformationLogLikOnDataScale()) {
							LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in W (the diagonal Hessian or Fisher "
								"information of the negative log-likelihood). Iterative evaluation of the Laplace determinant requires "
								"nonnegative W ");
						}
						SigmaI_plus_ZtWZ_rm_ = sp_mat_rm_t((*SigmaI_ptr)) + sp_mat_rm_t((*Zt_) * information_ll_.asDiagonal() * (*Zt_).transpose());
						if (cg_preconditioner_type_ == "incomplete_cholesky") {
							ZeroFillInIncompleteCholeskyFactorization(SigmaI_plus_ZtWZ_rm_, L_SigmaI_plus_ZtWZ_rm_);
						}
						else if (cg_preconditioner_type_ == "ssor") {
							P_SSOR_D_inv_ = SigmaI_plus_ZtWZ_rm_.diagonal().cwiseInverse();
							vec_t P_SSOR_D_inv_sqrt = P_SSOR_D_inv_.cwiseSqrt(); //need to store this, otherwise slow!
							sp_mat_rm_t P_SSOR_L_rm = SigmaI_plus_ZtWZ_rm_.triangularView<Eigen::Lower>();
							P_SSOR_L_D_sqrt_inv_rm_ = P_SSOR_L_rm * P_SSOR_D_inv_sqrt.asDiagonal();
						}
						else if (cg_preconditioner_type_ == "diagonal") {
							SigmaI_plus_ZtWZ_inv_diag_ = SigmaI_plus_ZtWZ_rm_.diagonal().cwiseInverse();
						}
					}
					//Get random vectors (z_1, ..., z_t) with Cov(z_i) = P:
					if (cg_preconditioner_type_ == "incomplete_cholesky") {
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < num_rand_vec_trace_; ++i) {
							rand_vec_trace_P_.col(i) = L_SigmaI_plus_ZtWZ_rm_ * rand_vec_trace_I_.col(i);
						}
					}
					else if (cg_preconditioner_type_ == "ssor") {
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < num_rand_vec_trace_; ++i) {
							rand_vec_trace_P_.col(i) = P_SSOR_L_D_sqrt_inv_rm_ * rand_vec_trace_I_.col(i);
						}
					}
					else if (cg_preconditioner_type_ == "diagonal") {
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < num_rand_vec_trace_; ++i) {
							rand_vec_trace_P_.col(i) = SigmaI_plus_ZtWZ_inv_diag_.cwiseInverse().cwiseSqrt().asDiagonal() * rand_vec_trace_I_.col(i);
						}
					}
					else if (cg_preconditioner_type_ == "none") {
						rand_vec_trace_P_ = rand_vec_trace_I_;
					}
					else {
						Log::REFatal("FindModePostRandEffCalcMLLGroupedRE: Preconditioner type '%s' is not supported ", cg_preconditioner_type_.c_str());
					}
					std::vector<vec_t> Tdiags_PI_SigmaI_plus_ZtWZ(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag));
					std::vector<vec_t> Tsubdiags_PI_SigmaI_plus_ZtWZ(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag - 1));
					CGTridiagRandomEffects(SigmaI_plus_ZtWZ_rm_, rand_vec_trace_P_, Tdiags_PI_SigmaI_plus_ZtWZ, Tsubdiags_PI_SigmaI_plus_ZtWZ,
						SigmaI_plus_ZtWZ_inv_RV_, has_NA_or_Inf, dim_mode_, num_rand_vec_trace_, cg_max_num_it_tridiag, cg_delta_conv_, cg_preconditioner_type_,
						L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps_tridiag_last_);
					if (!has_NA_or_Inf) {
						LogDetStochTridiag(Tdiags_PI_SigmaI_plus_ZtWZ, Tsubdiags_PI_SigmaI_plus_ZtWZ, log_det_SigmaI_plus_ZtWZ, dim_mode_, num_rand_vec_trace_);
						approx_marginal_ll += 0.5 * (SigmaI.diagonal().array().log().sum() - log_det_SigmaI_plus_ZtWZ);
						if (has_vecchia_gp) {
							approx_marginal_ll += 0.5 * D_inv.diagonal().array().log().sum();
						}
						// Correction for preconditioner
						if (cg_preconditioner_type_ == "incomplete_cholesky") {
							approx_marginal_ll -= L_SigmaI_plus_ZtWZ_rm_.diagonal().array().log().sum();//log|P| = log|L| + log|L^T|
						}
						else if (cg_preconditioner_type_ == "ssor") {
							approx_marginal_ll -= P_SSOR_L_D_sqrt_inv_rm_.diagonal().array().log().sum();//log|P| = log|L| + log|D^-1| + log|L^T|
						}
						else if (cg_preconditioner_type_ == "diagonal") {
							approx_marginal_ll += 0.5 * SigmaI_plus_ZtWZ_inv_diag_.array().log().sum();//log|P| = - log|diag(Sigma^-1 + Z^T W Z)^(-1)|
						}
					}
					else {
						approx_marginal_ll = std::numeric_limits<double>::quiet_NaN();
						Log::REDebug(NA_OR_INF_WARNING_);
						na_or_inf_during_last_call_to_find_mode_ = true;
					}
				}//end calculate determinant term for approx_marginal_ll               
			}//end iterative
			else {
				if (information_changes_after_mode_finding_) {
					CalcInformationLogLik(y_data, y_data_int, location_par.data(), false);
					SigmaI_plus_ZtWZ = (*SigmaI_ptr) + (sp_mat_t)((*Zt_) * information_ll_.asDiagonal() * (*Zt_).transpose());
					SigmaI_plus_ZtWZ.makeCompressed();
					chol_fact_SigmaI_plus_ZtWZ_grouped_.factorize(SigmaI_plus_ZtWZ);
					if (chol_fact_SigmaI_plus_ZtWZ_grouped_.info() != Eigen::Success) {
						LogFatalWithPotentialFisherLaplaceHint(__func__, "Cholesky factorization of Sigma^(-1) + Z^T W Z failed when "
							"calculating the Laplace determinant. The observed-Hessian matrix is not positive definite (negative "
							"individual entries of W are permitted) ");
					}
				}
				approx_marginal_ll += -((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_grouped_.matrixL()).diagonal().array().log().sum() + 0.5 * SigmaI.diagonal().array().log().sum();
				if (has_vecchia_gp) {
					approx_marginal_ll += 0.5 * D_inv.diagonal().array().log().sum();
				}
			}//end cholesky	
			mode_has_been_calculated_ = true;
			na_or_inf_during_last_call_to_find_mode_ = false;
		}
		FinalizeModeFinding(it);
	}//end FindModePostRandEffCalcMLLGroupedRE

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const double sigma2,
		double& approx_marginal_ll) {
		ChecksBeforeModeFinding();
		fixed_effects_ = fixed_effects;
		// Initialize variables
		InitializeModeForModeFinding();
		vec_t location_par(dim_location_par_);//location parameter = mode of random effects + fixed effects (+ possibly additional fixed-effects-only blocks)
		double* location_par_ptr_dummy;//not used
		UpdateLocationParNewMode(mode_, fixed_effects, location_par, &location_par_ptr_dummy);
		// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
		if (iid_model_) {
			approx_marginal_ll = LogLikelihood(y_data, y_data_int, location_par.data());
		}
		else {
			approx_marginal_ll = -0.5 / sigma2 * (mode_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par.data());
		}
		double approx_marginal_ll_new = approx_marginal_ll;
		vec_t rhs, mode_update, mode_new;
		// Start finding mode 
		int it;
		bool terminate_optim = false;
		bool has_NA_or_Inf = false;
		if (iid_model_) CHECK(maxit_mode_newton_ == 0);
		for (it = 0; it < maxit_mode_newton_; ++it) {
			// Calculate first and second derivative of log-likelihood
			CalcFirstDerivLogLik(y_data, y_data_int, location_par.data());
			if (it == 0 || information_changes_during_mode_finding_) {
				CalcInformationLogLik(y_data, y_data_int, location_par.data(), true);
				diag_SigmaI_plus_ZtWZ_ = (information_ll_.array() + 1. / sigma2).matrix();
			}
			//Log::REInfo("it = %d, first_deriv_ll_[0:2] = %g, %g, %g, information_ll_[0:2] = %g, %g, %g", it, 
			//	first_deriv_ll_[0], first_deriv_ll_[1], first_deriv_ll_[2], information_ll_[0], information_ll_[1], information_ll_[2]);//for debugging
			// Calculate rhs for mode update
			rhs = first_deriv_ll_ - mode_ / sigma2;//right hand side for updating mode
			// Update mode and do backtracking line search
			mode_update = (rhs.array() / diag_SigmaI_plus_ZtWZ_.array()).matrix();
			double grad_dot_direction = 0.;//for Armijo check
			if (armijo_condition_) {
				grad_dot_direction = mode_update.dot(rhs);//rhs = gradient of objective
			}
			double lr_mode = 1.;
			for (int ih = 0; ih < max_number_lr_shrinkage_steps_newton_; ++ih) {
				mode_new = mode_ + lr_mode * mode_update;
				if (kink_cliping_) {
					ApplyKinkClippingAsymLaplace(y_data, fixed_effects, mode_, mode_new);
				}
				UpdateLocationParNewMode(mode_new, fixed_effects, location_par, &location_par_ptr_dummy);
				approx_marginal_ll_new = -0.5 / sigma2 * (mode_new.dot(mode_new)) + LogLikelihood(y_data, y_data_int, location_par.data());// Calculate new objective function
				if (AcceptModeUpdate(approx_marginal_ll_new, approx_marginal_ll, grad_dot_direction, lr_mode)) {
					break;
				}
			}// end loop over learnig rate halving procedure
			mode_ = mode_new;
			CheckConvergenceModeFinding(it, approx_marginal_ll_new, approx_marginal_ll, terminate_optim, has_NA_or_Inf);
			//Log::REInfo("it = %d, mode_[0:2] = %g, %g, %g, LogLikelihood = %g", it, mode_[0], mode_[1], mode_[2], LogLikelihood(y_data, y_data_int, location_par.data()));//for debugging
			if (terminate_optim || has_NA_or_Inf) {
				break;
			}
		}//end mode finding algorithm
		if (!has_NA_or_Inf) {//calculate determinant
			if (sample_from_posterior_after_mode_finding_ && !iid_model_) {
				Sample_Posterior_LaplaceApprox_OnlyOneGroupedRE();
			}
			CalcFirstDerivLogLik(y_data, y_data_int, location_par.data());//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
			if (!iid_model_) {
				if (information_changes_after_mode_finding_) {
					CalcInformationLogLik(y_data, y_data_int, location_par.data(), false);
					diag_SigmaI_plus_ZtWZ_ = (information_ll_.array() + 1. / sigma2).matrix();
					if ((diag_SigmaI_plus_ZtWZ_.array() <= 0.).any()) {
						LogFatalWithPotentialFisherLaplaceHint(__func__, "negative values found in diag_SigmaI_plus_ZtWZ, must be positive for log-determinant ");
					}
				}
				approx_marginal_ll -= 0.5 * diag_SigmaI_plus_ZtWZ_.array().log().sum() + 0.5 * dim_mode_ * std::log(sigma2);
			}
			mode_has_been_calculated_ = true;
			na_or_inf_during_last_call_to_find_mode_ = false;
		}
		FinalizeModeFinding(it);
		//Log::REInfo("FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale: finished after %d iterations ", it);//for debugging
		//Log::REInfo("mode_[0:2] = %g, %g, %g, LogLikelihood = %g", mode_[0], mode_[1], mode_[2], LogLikelihood(y_data, y_data_int, location_par.data()));//for debugging
	}//end FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::FindModePostRandEffCalcMLLFSVA(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const den_mat_t& sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_woodbury,
		const den_mat_t& chol_ip_cross_cov,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const den_mat_t& sigma_woodbury,
		const sp_mat_t& B,
		const sp_mat_t& D_inv,
		const den_mat_t& Bt_D_inv_B_cross_cov,
		const den_mat_t& D_inv_B_cross_cov,
		const bool first_update,
		bool calc_mll,
		double& approx_marginal_ll,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_preconditioner_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i,
		const den_mat_t& chol_ip_cross_cov_preconditioner,
		const chol_den_mat_t& chol_fact_sigma_ip_preconditioner,
		bool GPU_use) {
		ChecksBeforeModeFinding();
		fixed_effects_ = fixed_effects;
		const den_mat_t* cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
		int num_ip = (int)((sigma_ip).rows());
		int num_ip_preconditioner = 0;
		CHECK((int)((*cross_cov).rows()) == dim_mode_);
		CHECK((int)((*cross_cov).cols()) == num_ip);
		den_mat_t sigma_ip_stable = sigma_ip;
		sigma_ip_stable.diagonal().array() *= JITTER_MULT_IP_FITC_FSA;
		// Initialize variables
		InitializeModeForModeFinding();
		vec_t location_par;//location parameter = mode of random effects + fixed effects
		double* location_par_ptr;
		vec_t rhs, B_mode, D_inv_B_mode, B_t_D_inv_B_mode, cross_cov_B_t_D_inv_B_mode,
			wood_inv_cross_cov_B_t_D_inv_B_mode, mode_new, mode_update(dim_mode_), mode_update_part(dim_mode_);
		//Convert to row-major for parallelization
		B_rm_ = sp_mat_rm_t(B);
		D_inv_rm_ = sp_mat_rm_t(D_inv);
		D_inv_B_rm_ = D_inv_rm_ * B_rm_;
		B_t_D_inv_rm_ = D_inv_B_rm_.transpose();
		// Variables when using Cholesky factorization
		sp_mat_t SigmaI, SigmaI_plus_W;
		den_mat_t woodbury_cross_cov_Bt_D_inv_B;
		// Variables when using iterative methods
		int cg_max_num_it = cg_max_num_it_;
		int cg_max_num_it_tridiag = cg_max_num_it_tridiag_;
		den_mat_t sigma_woodbury_woodbury;
		chol_den_mat_t chol_fact_sigma_woodbury_woodbury;
		InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
		// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
		B_mode = B_rm_ * mode_;
		D_inv_B_mode = D_inv_rm_.diagonal().asDiagonal() * B_mode;
		cross_cov_B_t_D_inv_B_mode = Bt_D_inv_B_cross_cov.transpose() * mode_;
		wood_inv_cross_cov_B_t_D_inv_B_mode = chol_fact_sigma_woodbury.solve(cross_cov_B_t_D_inv_B_mode);
		approx_marginal_ll = -0.5 * ((B_mode.dot(D_inv_B_mode)) - cross_cov_B_t_D_inv_B_mode.dot(wood_inv_cross_cov_B_t_D_inv_B_mode)) + LogLikelihood(y_data, y_data_int, location_par_ptr);
		double approx_marginal_ll_new = approx_marginal_ll;
		vec_t W_D_inv, W_D_inv_inv, W_D_inv_sqrt;
		if (matrix_inversion_method_ == "iterative") {
			//Reduce max. number of iterations for the CG in first update
			if (first_update && reduce_cg_max_num_it_first_optim_step_) {
				cg_max_num_it = (int)round(cg_max_num_it_ / 3);
				cg_max_num_it_tridiag = (int)round(cg_max_num_it_tridiag_ / 3);
			}
		}
		if (matrix_inversion_method_ != "iterative") {
			SigmaI = B.transpose() * D_inv * B;
		}
		// Start finding mode 
		int it;
		bool terminate_optim = false;
		bool has_NA_or_Inf = false;
		vec_t rhs_part, rhs_part1, rhs_part2, W_rhs, information_ll_inv(dim_mode_);
		den_mat_t sigma_woodbury_2;
		chol_den_mat_t chol_fact_sigma_woodbury_2;
		den_mat_t sigma_resid_plus_W_inv_cross_cov;
		den_mat_t B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov(dim_mode_, num_ip);
		D_inv_B_cross_cov_ = D_inv_B_cross_cov;
		den_mat_t chol_fact_SigmaI_plus_ZtWZ_vecchia_cross_cov;
		vec_t diagonal_approx_preconditioner_vecchia(dim_mode_);
		den_mat_t sigma_ip_preconditioner;
		if (matrix_inversion_method_ == "iterative") {
			if (cg_preconditioner_type_ == "fitc") {
				diagonal_approx_preconditioner_.resize(dim_mode_);
				sigma_ip_preconditioner = *(re_comps_ip_preconditioner_cluster_i[0]->GetZSigmaZt());
				sigma_ip_preconditioner.diagonal().array() *= JITTER_MULT_IP_FITC_FSA;
				num_ip_preconditioner = (int)((sigma_ip_preconditioner).rows());
#pragma omp parallel for schedule(static)
				for (int j = 0; j < dim_mode_; ++j) {
					diagonal_approx_preconditioner_vecchia[j] = (sigma_ip_preconditioner).coeffRef(0, 0) - chol_ip_cross_cov_preconditioner.col(j).array().square().sum();
				}
			}
		}
		for (it = 0; it < maxit_mode_newton_; ++it) {
			// Calculate first and second derivative of log-likelihood
			CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);
			if (it == 0 || information_changes_during_mode_finding_) {
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, true);
			}
			// Calculate Cholesky factor and update mode
			rhs.array() = information_ll_.array() * mode_.array() + first_deriv_ll_.array();//right hand side for updating mode
			if (matrix_inversion_method_ == "iterative") {
				//Reduce max. number of iterations for the CG in first update
				if (cg_preconditioner_type_ == "vifdu" || cg_preconditioner_type_ == "none") {
					if (it == 0 || information_changes_during_mode_finding_) {
						W_D_inv = (information_ll_ + D_inv_rm_.diagonal());
						W_D_inv_inv = W_D_inv.cwiseInverse();
						W_D_inv_sqrt = W_D_inv_inv.cwiseSqrt();
						if (cg_preconditioner_type_ == "vifdu") {
							B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov = W_D_inv_sqrt.asDiagonal() * D_inv_B_cross_cov_;
							den_mat_t B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov_dot;
							GPBoost::matmul(B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov.transpose(), B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov, B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov_dot, GPU_use);
							sigma_woodbury_woodbury = sigma_woodbury - B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov_dot;
							chol_fact_sigma_woodbury_woodbury.compute(sigma_woodbury_woodbury);
							CheckCholeskyFactorization(chol_fact_sigma_woodbury_woodbury, "FindModePostRandEffCalcMLLFSVA iterative VIF preconditioner");
						}
					}
					if ((information_ll_.array() > 1e10).any()) {
						has_NA_or_Inf = true;// the inversion of the preconditioner with the Woodbury identity can be numerically unstable when information_ll_ is very large
					}
					else {
						CGFVIFLaplaceVec(information_ll_, B_rm_, B_t_D_inv_rm_, chol_fact_sigma_woodbury, cross_cov, W_D_inv_inv,
							chol_fact_sigma_woodbury_woodbury, rhs, mode_update, has_NA_or_Inf, cg_max_num_it, it == 0, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, false);
					}
				}
				else if (cg_preconditioner_type_ == "fitc") {
					const den_mat_t* cross_cov_preconditioner = re_comps_cross_cov_preconditioner_cluster_i[0]->GetSigmaPtr();
					if (it == 0 || information_changes_during_mode_finding_) {
						if (HasNegativeValueInformationLogLik()) {
							LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in W (the diagonal Hessian or Fisher "
								"information of the negative log-likelihood). The iterative FITC preconditioner is based on the "
								"positive-definite W^(-1) formulation. Try another preconditioner or the Cholesky decomposition ");
						}
						if (HasZeroValueInformationLogLik()) {
							Log::REFatal("FindModePostRandEffCalcMLLFSVA: 0's found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
								"The iterative FITC preconditioner requires W to be invertible. Try using another preconditioner or the Cholesky decomposition ");
						}
						information_ll_inv.array() = information_ll_.array().inverse();
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < dim_mode_; ++i) {
							diagonal_approx_preconditioner_[i] = diagonal_approx_preconditioner_vecchia[i] + information_ll_inv[i];
						}
						diagonal_approx_inv_preconditioner_ = diagonal_approx_preconditioner_.cwiseInverse();
						den_mat_t sigma_woodbury_preconditioner;
						GPBoost::matmul((*cross_cov_preconditioner).transpose(), (diagonal_approx_inv_preconditioner_.asDiagonal() * (*cross_cov_preconditioner)), sigma_woodbury_preconditioner, GPU_use);
						//den_mat_t sigma_woodbury_preconditioner = ((*cross_cov_preconditioner).transpose() * diagonal_approx_inv_preconditioner_.asDiagonal()) * (*cross_cov_preconditioner);
						sigma_woodbury_preconditioner += (sigma_ip_preconditioner);
						chol_fact_woodbury_preconditioner_.compute(sigma_woodbury_preconditioner);
						CheckCholeskyFactorization(chol_fact_woodbury_preconditioner_, "FindModePostRandEffCalcMLLFSVA iterative FITC preconditioner");
					}
					rhs_part1 = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(rhs);
					rhs_part = D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve(rhs_part1);
					rhs_part2 = (*cross_cov) * (chol_fact_sigma_ip.solve((*cross_cov).transpose() * rhs));
					rhs = rhs_part + rhs_part2;
					CGVIFLaplace_Version_SigmaPlusWinvVec(information_ll_inv, D_inv_B_rm_, B_rm_, chol_fact_woodbury_preconditioner_,
						chol_ip_cross_cov, cross_cov_preconditioner, diagonal_approx_inv_preconditioner_, rhs, mode_update_part, has_NA_or_Inf,
						cg_max_num_it, it == 0, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, false);
					mode_update = information_ll_inv.asDiagonal() * mode_update_part;
				}
				if (has_NA_or_Inf) {
					//'mode_update' has not (fully) been calculated and must not be used below
					approx_marginal_ll_new = std::numeric_limits<double>::quiet_NaN();
					Log::REDebug(NA_OR_INF_WARNING_);
					break;
				}
			}//end iterative
			else { // start Cholesky
				information_ll_inv.array() = information_ll_.array().inverse();
				if (it == 0 || information_changes_during_mode_finding_) {
					SigmaI_plus_W = SigmaI;
					SigmaI_plus_W.diagonal().array() += information_ll_.array();
					SigmaI_plus_W.makeCompressed();
					//Calculation of the Cholesky factor is the bottleneck
					if (!chol_fact_pattern_analyzed_) {
						chol_fact_SigmaI_plus_ZtWZ_vecchia_.analyzePattern(SigmaI_plus_W);
						chol_fact_pattern_analyzed_ = true;
					}
					chol_fact_SigmaI_plus_ZtWZ_vecchia_.factorize(SigmaI_plus_W);//This is the bottleneck for large data
					CheckCholeskyFactorization(chol_fact_SigmaI_plus_ZtWZ_vecchia_, "FindModePostRandEffCalcMLLFSVA mode update");
				}
				sigma_woodbury_2 = (sigma_woodbury) - Bt_D_inv_B_cross_cov.transpose() * chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(Bt_D_inv_B_cross_cov);
				chol_fact_sigma_woodbury_2.compute(sigma_woodbury_2);
				CheckCholeskyFactorization(chol_fact_sigma_woodbury_2, "FindModePostRandEffCalcMLLFSVA Woodbury mode update");
				vec_t Sigma_I_rhs = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(rhs);
				vec_t Bt_D_inv_B_cross_cov_T_Sigma_I_rhs = Bt_D_inv_B_cross_cov.transpose() * Sigma_I_rhs;
				vec_t woodI_Bt_D_inv_B_cross_cov_T_Sigma_I_rhs = chol_fact_sigma_woodbury_2.solve(Bt_D_inv_B_cross_cov_T_Sigma_I_rhs);
				vec_t Bt_D_inv_B_cross_cov_woodI_Bt_D_inv_B_cross_cov_T_Sigma_I_rhs = Bt_D_inv_B_cross_cov * woodI_Bt_D_inv_B_cross_cov_T_Sigma_I_rhs;
				vec_t SigmaI_Bt_D_inv_B_cross_cov_woodI_Bt_D_inv_B_cross_cov_T_Sigma_I_rhs = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(Bt_D_inv_B_cross_cov_woodI_Bt_D_inv_B_cross_cov_T_Sigma_I_rhs);
				mode_update = Sigma_I_rhs + SigmaI_Bt_D_inv_B_cross_cov_woodI_Bt_D_inv_B_cross_cov_T_Sigma_I_rhs;
			} // end Cholesky
			// Backtracking line search
			double grad_dot_direction = 0.;//for Armijo check
			if (armijo_condition_) {
				if (num_sets_re_ > 1) {
					Log::REFatal("The Armijo condition check is currently not implemented when num_sets_re_ > 1 (=multiple parameters related to GPs) ");
				}
				vec_t direction = mode_update - mode_;//mode_update = "new mode" = (Sigma^-1 + W)^-1 (W * mode + grad p(y|mode))
				vec_t gradient = B.transpose() * (D_inv * (B * direction)) -
					Bt_D_inv_B_cross_cov * (chol_fact_sigma_woodbury.solve(Bt_D_inv_B_cross_cov.transpose() * direction)) +
					information_ll_.asDiagonal() * direction;//gradient = (Sigma^-1 + W) * direction (Sigma^-1 calculated with Woodbury), direction = (Sigma^-1 + W)^-1 * gradient
				grad_dot_direction = direction.dot(gradient);
			}
			double lr_mode = 1.;
			for (int ih = 0; ih < max_number_lr_shrinkage_steps_newton_; ++ih) {
				if (ih == 0) {
					mode_new = mode_update;//mode_update = "new mode" = (Sigma^-1 + W)^-1 (W * mode + grad p(y|mode))
				}
				else {
					mode_new = (1 - lr_mode) * mode_ + lr_mode * mode_update;
				}
				if (kink_cliping_) {
					ApplyKinkClippingAsymLaplace(y_data, fixed_effects, mode_, mode_new);
				}
				CapChangeModeUpdateNewton(mode_new);
				UpdateLocationParNewMode(mode_new, fixed_effects, location_par, &location_par_ptr); // Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
				B_mode = B * mode_new;
				cross_cov_B_t_D_inv_B_mode = Bt_D_inv_B_cross_cov.transpose() * mode_new;
				wood_inv_cross_cov_B_t_D_inv_B_mode = chol_fact_sigma_woodbury.solve(cross_cov_B_t_D_inv_B_mode);
				approx_marginal_ll_new = -0.5 * ((B_mode.dot(D_inv * B_mode)) - cross_cov_B_t_D_inv_B_mode.dot(wood_inv_cross_cov_B_t_D_inv_B_mode)) + LogLikelihood(y_data, y_data_int, location_par_ptr);
				if (AcceptModeUpdate(approx_marginal_ll_new, approx_marginal_ll, grad_dot_direction, lr_mode)) {
					break;
				}
			}// end loop over learnig rate halving procedure
			mode_ = mode_new;
			CheckConvergenceModeFinding(it, approx_marginal_ll_new, approx_marginal_ll, terminate_optim, has_NA_or_Inf);
			if (terminate_optim || has_NA_or_Inf) {
				break;
			}
		} // end loop for mode finding
		if (!has_NA_or_Inf) {//calculate determinant
			mode_has_been_calculated_ = true;
			na_or_inf_during_last_call_to_find_mode_ = false;
			if (sample_from_posterior_after_mode_finding_) {
				Sample_Posterior_LaplaceApprox_FSVA(cross_cov, Bt_D_inv_B_cross_cov, sigma_woodbury, chol_fact_sigma_woodbury,
					chol_fact_sigma_ip, chol_fact_sigma_woodbury_2, chol_ip_cross_cov, re_comps_cross_cov_preconditioner_cluster_i);
			}
			CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
			if (information_changes_after_mode_finding_) {
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, false);
			}
			if (matrix_inversion_method_ == "iterative") {
				if (HasNegativeValueInformationLogLikOnDataScale()) {
					LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in W (the diagonal Hessian or Fisher "
						"information of the negative log-likelihood). Iterative evaluation of the Laplace determinant requires "
						"nonnegative W ");
				}
				if (calc_mll) {//calculate determinant term for approx_marginal_ll
					//Generate random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I
					if (!saved_rand_vec_trace_) {
						rand_vec_trace_I2_.resize(dim_mode_, num_rand_vec_trace_);
						rand_vec_trace_I_.resize(dim_mode_, num_rand_vec_trace_);
						GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_trace_I2_);
						SigmaI_plus_W_inv_Z_.resize(dim_mode_, num_rand_vec_trace_);
						if (cg_preconditioner_type_ == "vifdu") {
							rand_vec_trace_P_.resize(num_ip, num_rand_vec_trace_);
							rand_vec_trace_I3_.resize(dim_mode_, num_rand_vec_trace_);
							GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_trace_P_);
							GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_trace_I3_);
						}
						else if (cg_preconditioner_type_ == "fitc") {
							rand_vec_trace_P_.resize(num_ip_preconditioner, num_rand_vec_trace_);
							GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_trace_P_);
						}
						if (reuse_rand_vec_trace_) {
							saved_rand_vec_trace_ = true;
						}
					}
					if (cg_preconditioner_type_ == "vifdu") {
						if (HasNegativeValueInformationLogLik()) {
							LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in W (the diagonal Hessian or Fisher "
								"information of the negative log-likelihood). Stochastic evaluation of the Laplace determinant with the "
								"'vifdu' preconditioner requires the square root of W ");
						}
						// B^T * W^1/2 * rand_vec
						den_mat_t Bt_W_sqrt_rand_vec_trace(dim_mode_, num_rand_vec_trace_);
						vec_t information_ll_sqrt = information_ll_.cwiseSqrt();
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < num_rand_vec_trace_; ++i) {
							Bt_W_sqrt_rand_vec_trace.col(i) = B_rm_.transpose() * information_ll_sqrt.cwiseProduct(rand_vec_trace_I2_.col(i));
						}
						// Sigma^1/2 * rand_vec
						den_mat_t Sigma_sqrt_rand_vec = chol_ip_cross_cov.transpose() * rand_vec_trace_P_;
						vec_t D_sqrt = D_inv_rm_.diagonal().cwiseInverse().cwiseSqrt();
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < num_rand_vec_trace_; ++i) {
							Sigma_sqrt_rand_vec.col(i) += B_rm_.triangularView<Eigen::UpLoType::UnitLower>().solve(D_sqrt.cwiseProduct(rand_vec_trace_I3_.col(i)));
						}
						// Sigma^-1 * Sigma^1/2 * rand_vec
						den_mat_t Bt_D_inv_Sigma_sqrt_rand_vec(dim_mode_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < num_rand_vec_trace_; ++i) {
							Bt_D_inv_Sigma_sqrt_rand_vec.col(i) = B_t_D_inv_rm_ * B_rm_ * Sigma_sqrt_rand_vec.col(i);
						}
						den_mat_t Sigma_inv_Sigma_sqrt_rand_vec = Bt_D_inv_Sigma_sqrt_rand_vec - Bt_D_inv_B_cross_cov * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * Bt_D_inv_Sigma_sqrt_rand_vec);
						Bt_D_inv_Sigma_sqrt_rand_vec.resize(0, 0);
						// P^1/2 * rand_vec
						rand_vec_trace_I_ = Bt_W_sqrt_rand_vec_trace + Sigma_inv_Sigma_sqrt_rand_vec;
						Bt_W_sqrt_rand_vec_trace.resize(0, 0);
						Sigma_inv_Sigma_sqrt_rand_vec.resize(0, 0);
					}
					else if (cg_preconditioner_type_ == "fitc") {
						rand_vec_trace_I_ = diagonal_approx_preconditioner_.cwiseSqrt().asDiagonal() * rand_vec_trace_I2_ + chol_ip_cross_cov_preconditioner.transpose() * rand_vec_trace_P_;
					}
					else {
						rand_vec_trace_I_ = rand_vec_trace_I2_;
					}
					if (information_changes_after_mode_finding_) {
						W_D_inv = (information_ll_ + D_inv_rm_.diagonal());
						W_D_inv_inv = W_D_inv.cwiseInverse();
						W_D_inv_sqrt = W_D_inv_inv.cwiseSqrt();
						if (cg_preconditioner_type_ == "vifdu") {
							B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov = W_D_inv_sqrt.asDiagonal() * D_inv_B_cross_cov_;
							den_mat_t B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov_dot;
							GPBoost::matmul(B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov.transpose(), B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov, B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov_dot, GPU_use);
							sigma_woodbury_woodbury_ = sigma_woodbury - B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov_dot;
							chol_fact_sigma_woodbury_woodbury_.compute(sigma_woodbury_woodbury_);
							CheckCholeskyFactorization(chol_fact_sigma_woodbury_woodbury_, "FindModePostRandEffCalcMLLFSVA iterative determinant preconditioner");
						}
					}
					double log_det_Sigma_W_plus_I;
					CalcLogDetStochFSVA(dim_mode_, cg_max_num_it_tridiag, chol_fact_sigma_woodbury, chol_ip_cross_cov, chol_fact_sigma_ip, chol_fact_sigma_ip_preconditioner,
						cross_cov, re_comps_cross_cov_preconditioner_cluster_i, W_D_inv_inv, chol_fact_sigma_woodbury_woodbury_, W_D_inv,
						has_NA_or_Inf, log_det_Sigma_W_plus_I);
					if (has_NA_or_Inf) {
						approx_marginal_ll = std::numeric_limits<double>::quiet_NaN();
						Log::REDebug(NA_OR_INF_WARNING_);
						na_or_inf_during_last_call_to_find_mode_ = true;
					}
					else {
						approx_marginal_ll -= 0.5 * log_det_Sigma_W_plus_I;
					}
				}//end calculate determinant term for approx_marginal_ll
			}//end iterative
			else {
				if (information_changes_after_mode_finding_) {
					SigmaI_plus_W = SigmaI;
					SigmaI_plus_W.diagonal().array() += information_ll_.array();
					SigmaI_plus_W.makeCompressed();
					if (!chol_fact_pattern_analyzed_) {
						chol_fact_SigmaI_plus_ZtWZ_vecchia_.analyzePattern(SigmaI_plus_W);
						chol_fact_pattern_analyzed_ = true;
					}
					chol_fact_SigmaI_plus_ZtWZ_vecchia_.factorize(SigmaI_plus_W);
					if (chol_fact_SigmaI_plus_ZtWZ_vecchia_.info() != Eigen::Success) {
						LogFatalWithPotentialFisherLaplaceHint(__func__, "Cholesky factorization failed because the matrix is not positive "
							"definite or contains non-finite values ");
					}
				}
				TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, den_mat_t, den_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_,
					Bt_D_inv_B_cross_cov, chol_fact_SigmaI_plus_ZtWZ_vecchia_cross_cov, false);
				sigma_woodbury_woodbury_ = sigma_woodbury - chol_fact_SigmaI_plus_ZtWZ_vecchia_cross_cov.transpose() * chol_fact_SigmaI_plus_ZtWZ_vecchia_cross_cov;
				chol_fact_sigma_woodbury_woodbury_.compute(sigma_woodbury_woodbury_);
				CheckCholeskyFactorization(chol_fact_sigma_woodbury_woodbury_, "FindModePostRandEffCalcMLLFSVA Woodbury determinant");
				approx_marginal_ll += -((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_vecchia_.matrixL()).diagonal().array().log().sum() + 0.5 * D_inv.diagonal().array().log().sum();
				approx_marginal_ll += ((den_mat_t)chol_fact_sigma_ip.matrixL()).diagonal().array().log().sum();
				approx_marginal_ll -= ((den_mat_t)chol_fact_sigma_woodbury_woodbury_.matrixL()).diagonal().array().log().sum();
			}
		}
		FinalizeModeFinding(it);
		//Log::REInfo("FindModePostRandEffCalcMLLFSVA: finished after %d iterations, mode_[0:2] = %g, %g, %g ", it, mode_[0], mode_[1], mode_[2]);//for debugging
	}//end FindModePostRandEffCalcMLLFSVA

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::FindModePostRandEffCalcMLLVecchia(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		std::map<int, sp_mat_t>& B,
		std::map<int, sp_mat_t>& D_inv,
		const bool first_update,
		const den_mat_t& Sigma_L_k,
		bool calc_mll,
		double& approx_marginal_ll,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const den_mat_t& chol_ip_cross_cov,
		const chol_den_mat_t& chol_fact_sigma_ip,
		data_size_t cluster_i,
		REModelTemplate<T_mat, T_chol>* re_model) {
		ChecksBeforeModeFinding();
		fixed_effects_ = fixed_effects;
		// Initialize variables
		InitializeModeForModeFinding();
		vec_t location_par;//location parameter = mode of random effects + fixed effects
		double* location_par_ptr;
		vec_t rhs, B_mode, mode_new, mode_update(dim_mode_);
		// Variables when using Cholesky factorization
		sp_mat_t SigmaI, SigmaI_plus_W;
		// Variables when using iterative methods
		int cg_max_num_it = cg_max_num_it_;
		int cg_max_num_it_tridiag = cg_max_num_it_tridiag_;
		den_mat_t I_k_plus_Sigma_L_kt_W_Sigma_L_k;
		InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
		// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
		approx_marginal_ll = LogLikelihood(y_data, y_data_int, location_par_ptr);
		if (num_sets_re_ == 1) {
			B_mode = B[0] * mode_;
			approx_marginal_ll += -0.5 * (B_mode.dot(D_inv[0] * B_mode));
		}
		else {
			CHECK((int)B.size() == num_sets_re_);
			CHECK(dim_mode_ == num_sets_re_ * dim_mode_per_set_re_);
			B_mode = vec_t(dim_mode_);
			for (int igp = 0; igp < num_sets_re_; ++igp) {
				B_mode.segment(dim_mode_per_set_re_ * igp, dim_mode_per_set_re_) = B[igp] * (mode_.segment(dim_mode_per_set_re_ * igp, dim_mode_per_set_re_));
				approx_marginal_ll += -0.5 * ((B_mode.segment(dim_mode_per_set_re_ * igp, dim_mode_per_set_re_)).dot(D_inv[igp] * (B_mode.segment(dim_mode_per_set_re_ * igp, dim_mode_per_set_re_))));
			}
		}
		double approx_marginal_ll_new = approx_marginal_ll;
		if (matrix_inversion_method_ == "iterative") {
			if (num_sets_re_ > 1) {
				if (cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc" ||
					cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "vecchia_response") {
					Log::REFatal("'iterative' methods with the '%s' preconditioner are currently not implemented for a '%s' likleihood ",
						cg_preconditioner_type_.c_str(), likelihood_type_.c_str());
				}
			}
			//Reduce max. number of iterations for the CG in first update
			if (first_update && reduce_cg_max_num_it_first_optim_step_) {
				cg_max_num_it = (int)round(cg_max_num_it_ / 3);
				cg_max_num_it_tridiag = (int)round(cg_max_num_it_tridiag_ / 3);
			}
			//Convert to row-major for parallelization
			if (num_sets_re_ == 1) {
				B_rm_ = sp_mat_rm_t(B[0]);
				D_inv_rm_ = sp_mat_rm_t(D_inv[0]);
			}
			else {
				sp_mat_rm_t B_rm_1 = sp_mat_rm_t(B[0]);
				sp_mat_rm_t B_rm_2 = sp_mat_rm_t(B[1]);
				GPBoost::CreatSparseBlockDiagonalMartix<sp_mat_rm_t>(B_rm_1, B_rm_2, B_rm_);
				sp_mat_rm_t D_inv_1 = sp_mat_rm_t(D_inv[0]);
				sp_mat_rm_t D_inv_2 = sp_mat_rm_t(D_inv[1]);
				GPBoost::CreatSparseBlockDiagonalMartix<sp_mat_rm_t>(D_inv_1, D_inv_2, D_inv_rm_);
			}
			B_t_D_inv_rm_ = B_rm_.transpose() * D_inv_rm_;
			if (cg_preconditioner_type_ == "pivoted_cholesky") {
				//Store as class variable
				Sigma_L_k_ = Sigma_L_k;
				I_k_plus_Sigma_L_kt_W_Sigma_L_k.resize(Sigma_L_k_.cols(), Sigma_L_k_.cols());
			}
			else if (cg_preconditioner_type_ == "fitc") {
				chol_fact_sigma_ip_ = chol_fact_sigma_ip;
				chol_ip_cross_cov_ = chol_ip_cross_cov;
				sigma_ip_stable_ = *(re_comps_ip_cluster_i[0]->GetZSigmaZt());
				sigma_ip_stable_.diagonal().array() *= JITTER_MULT_IP_FITC_FSA;
			}
		}
		if (matrix_inversion_method_ != "iterative" ||
			(matrix_inversion_method_ == "iterative" && cg_preconditioner_type_ == "incomplete_cholesky")) {
			if (num_sets_re_ == 1) {
				SigmaI = B[0].transpose() * D_inv[0] * B[0];
			}
			else {
				sp_mat_t SigmaI_1 = B[0].transpose() * D_inv[0] * B[0];
				sp_mat_t SigmaI_2 = B[1].transpose() * D_inv[1] * B[1];
				GPBoost::CreatSparseBlockDiagonalMartix<sp_mat_t>(SigmaI_1, SigmaI_2, SigmaI);
				CHECK(SigmaI.cols() == dim_mode_);
				CHECK(SigmaI.rows() == dim_mode_);
			}
		}
		// Start finding mode 
		int it;
		bool terminate_optim = false;
		bool has_NA_or_Inf = false;
		for (it = 0; it < maxit_mode_newton_; ++it) {
			// Calculate first and second derivative of log-likelihood
			CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);
			if (it == 0 || information_changes_during_mode_finding_) {
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, true);
			}
			// Calculate Cholesky factor and update mode
			if (information_has_off_diagonal_) {
				rhs = information_ll_mat_ * mode_ + first_deriv_ll_;//right hand side for updating mode
			}
			else {
				rhs.array() = information_ll_.array() * mode_.array() + first_deriv_ll_.array();//right hand side for updating mode
			}
			if (matrix_inversion_method_ == "iterative") {
				bool calculate_preconditioners = it == 0 || information_changes_during_mode_finding_;
				Inv_SigmaI_plus_ZtWZ_Vecchia_iterative(cg_max_num_it, I_k_plus_Sigma_L_kt_W_Sigma_L_k, SigmaI, SigmaI_plus_W, B[0], has_NA_or_Inf,
					re_comps_cross_cov_cluster_i, cluster_i, re_model, rhs, mode_update, it == 0, calculate_preconditioners);
				if (has_NA_or_Inf) {
					approx_marginal_ll_new = std::numeric_limits<double>::quiet_NaN();
					Log::REDebug(NA_OR_INF_WARNING_);
					break;
				}
			} //end iterative
			else { // start Cholesky 
				if (it == 0 || information_changes_during_mode_finding_) {
					SigmaI_plus_W = SigmaI;
					if (information_has_off_diagonal_) {
						SigmaI_plus_W += information_ll_mat_;
					}
					else {
						SigmaI_plus_W.diagonal().array() += information_ll_.array();
					}
					SigmaI_plus_W.makeCompressed();
					//Calculation of the Cholesky factor is the bottleneck
					if (!chol_fact_pattern_analyzed_) {
						chol_fact_SigmaI_plus_ZtWZ_vecchia_.analyzePattern(SigmaI_plus_W);
						chol_fact_pattern_analyzed_ = true;
					}
					chol_fact_SigmaI_plus_ZtWZ_vecchia_.factorize(SigmaI_plus_W);//This is the bottleneck for large data
					CheckCholeskyFactorization(chol_fact_SigmaI_plus_ZtWZ_vecchia_, "FindModePostRandEffCalcMLLVecchia mode update");
				}
				//Log::REInfo("SigmaI_plus_W: number non zeros = %d", (int)SigmaI_plus_W.nonZeros());//only for debugging
				//Log::REInfo("chol_fact_SigmaI_plus_ZtWZ: Number non zeros = %d", (int)((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_vecchia_.matrixL()).nonZeros());//only for debugging
				mode_update = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(rhs);
			} // end Cholesky
			// Backtracking line search
			double grad_dot_direction = 0.;//for Armijo check
			if (armijo_condition_) {
				if (num_sets_re_ > 1) {
					Log::REFatal("The Armijo condition check is currently not implemented when num_sets_re_ > 1 (=multiple parameters related to GPs) ");
				}
				vec_t direction = mode_update - mode_;//mode_update = "new mode" = (Sigma^-1 + W)^-1 (W * mode + grad p(y|mode))
				vec_t gradient = B[0].transpose() * (D_inv[0] * (B[0] * direction)) + information_ll_.asDiagonal() * direction;//gradient = (Sigma^-1 + W) * direction, direction = (Sigma^-1 + W)^-1 * gradient
				grad_dot_direction = direction.dot(gradient);
			}
			double lr_mode = 1.;
			for (int ih = 0; ih < max_number_lr_shrinkage_steps_newton_; ++ih) {
				if (ih == 0) {
					mode_new = mode_update;//mode_update = "new mode" = (Sigma^-1 + W)^-1 (W * mode + grad p(y|mode))
				}
				else {
					mode_new = (1 - lr_mode) * mode_ + lr_mode * mode_update;
				}
				if (kink_cliping_) {
					ApplyKinkClippingAsymLaplace(y_data, fixed_effects, mode_, mode_new);
				}
				CapChangeModeUpdateNewton(mode_new);
				UpdateLocationParNewMode(mode_new, fixed_effects, location_par, &location_par_ptr); // Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
				approx_marginal_ll_new = LogLikelihood(y_data, y_data_int, location_par_ptr);
				if (num_sets_re_ == 1) {
					B_mode = B[0] * mode_new;
					approx_marginal_ll_new += -0.5 * (B_mode.dot(D_inv[0] * B_mode));
				}
				else {
					for (int igp = 0; igp < num_sets_re_; ++igp) {
						B_mode.segment(dim_mode_per_set_re_ * igp, dim_mode_per_set_re_) = B[igp] * (mode_new.segment(dim_mode_per_set_re_ * igp, dim_mode_per_set_re_));
						approx_marginal_ll_new += -0.5 * ((B_mode.segment(dim_mode_per_set_re_ * igp, dim_mode_per_set_re_)).dot(D_inv[igp] * (B_mode.segment(dim_mode_per_set_re_ * igp, dim_mode_per_set_re_))));
					}
				}
				if (AcceptModeUpdate(approx_marginal_ll_new, approx_marginal_ll, grad_dot_direction, lr_mode)) {
					break;
				}
			}// end loop over learnig rate halving procedure
			mode_ = mode_new;
			CheckConvergenceModeFinding(it, approx_marginal_ll_new, approx_marginal_ll, terminate_optim, has_NA_or_Inf);
			if (terminate_optim || has_NA_or_Inf) {
				break;
			}
		} // end loop for mode finding
		if (!has_NA_or_Inf) {//calculate determinant
			mode_has_been_calculated_ = true;
			na_or_inf_during_last_call_to_find_mode_ = false;
			if (sample_from_posterior_after_mode_finding_) {
				Sample_Posterior_LaplaceApprox_Vecchia(re_comps_cross_cov_cluster_i);
			}
			CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
			if (information_changes_after_mode_finding_) {
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, false);
			}
			if (matrix_inversion_method_ == "iterative") {
				if (HasNegativeValueInformationLogLikOnDataScale()) {
					LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in W (the diagonal Hessian or Fisher "
						"information of the negative log-likelihood). Iterative evaluation of the Laplace determinant requires "
						"nonnegative W ");
				}
				if (calc_mll) {//calculate determinant term for approx_marginal_ll
					//Generate random vectors (r_1, r_2, r_3, ...) with Cov(r_i) = I
					if (!saved_rand_vec_trace_) {
						//Dependent on the preconditioner: Generate t (= num_rand_vec_trace_) or 2*t random vectors
						if (cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc" || cg_preconditioner_type_ == "vecchia_response") {
							rand_vec_trace_I_.resize(dim_mode_, num_rand_vec_trace_);
							if (cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc") {
								rand_vec_trace_I2_.resize(fitc_piv_chol_preconditioner_rank_, num_rand_vec_trace_);
								GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_trace_I2_);
							}
							WI_plus_Sigma_inv_Z_.resize(dim_mode_, num_rand_vec_trace_);
						}
						else if (cg_preconditioner_type_ == "vadu" || cg_preconditioner_type_ == "incomplete_cholesky") {
							rand_vec_trace_I_.resize(dim_mode_, num_rand_vec_trace_);
							SigmaI_plus_W_inv_Z_.resize(dim_mode_, num_rand_vec_trace_);
						}
						else {
							Log::REFatal("FindModePostRandEffCalcMLLVecchia: Preconditioner type '%s' is not supported ", cg_preconditioner_type_.c_str());
						}
						GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_trace_I_);
						if (reuse_rand_vec_trace_) {
							saved_rand_vec_trace_ = true;
						}
						rand_vec_trace_P_.resize(dim_mode_, num_rand_vec_trace_);
					}
					double log_det_Sigma_W_plus_I;
					CalcLogDetStochVecchia(dim_mode_, cg_max_num_it_tridiag, I_k_plus_Sigma_L_kt_W_Sigma_L_k, SigmaI, SigmaI_plus_W, B[0], has_NA_or_Inf, log_det_Sigma_W_plus_I,
						re_comps_cross_cov_cluster_i, cluster_i, re_model);
					if (has_NA_or_Inf) {
						approx_marginal_ll = std::numeric_limits<double>::quiet_NaN();
						Log::REDebug(NA_OR_INF_WARNING_);
						na_or_inf_during_last_call_to_find_mode_ = true;
					}
					else {
						approx_marginal_ll -= 0.5 * log_det_Sigma_W_plus_I;
					}
				}//end calculate determinant term for approx_marginal_ll
			}//end iterative
			else {
				if (information_changes_after_mode_finding_) {
					SigmaI_plus_W = SigmaI;
					if (information_has_off_diagonal_) {
						SigmaI_plus_W += information_ll_mat_;
					}
					else {
						SigmaI_plus_W.diagonal().array() += information_ll_.array();
					}
					SigmaI_plus_W.makeCompressed();
					if (!chol_fact_pattern_analyzed_) {
						chol_fact_SigmaI_plus_ZtWZ_vecchia_.analyzePattern(SigmaI_plus_W);
						chol_fact_pattern_analyzed_ = true;
					}
					chol_fact_SigmaI_plus_ZtWZ_vecchia_.factorize(SigmaI_plus_W);
					if (chol_fact_SigmaI_plus_ZtWZ_vecchia_.info() != Eigen::Success) {
						LogFatalWithPotentialFisherLaplaceHint(__func__, " (determinant) Cholesky factorization failed because the matrix is "
							"not positive definite or contains non-finite values ");
					}
				}
				approx_marginal_ll += -((sp_mat_t)chol_fact_SigmaI_plus_ZtWZ_vecchia_.matrixL()).diagonal().array().log().sum();
				for (int igp = 0; igp < num_sets_re_; ++igp) {
					approx_marginal_ll += 0.5 * D_inv[igp].diagonal().array().log().sum();
				}
			}
		}
		FinalizeModeFinding(it);
		//Log::REInfo("FindModePostRandEffCalcMLLVecchia: finished after %d iterations ", it);//for debugging
		//Log::REInfo("mode_[0:1,(last-1):last] = %g, %g, %g, %g, LogLikelihood = %g", mode_[0], mode_[1], mode_[dim_mode_ - 2], mode_[dim_mode_-1], LogLikelihood(y_data, y_data_int, location_par_ptr));//for debugging
		//Log::REInfo("it = %d, first_deriv_ll_[0:2] = %g, %g, %g, information_ll_[0:2] = %g, %g, %g", it,
		//	first_deriv_ll_[0], first_deriv_ll_[1], first_deriv_ll_[2], information_ll_[0], information_ll_[1], information_ll_[2]);//for debugging
	}//end FindModePostRandEffCalcMLLVecchia

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::FindModePostRandEffCalcMLLFITC(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const std::shared_ptr<den_mat_t> sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const den_mat_t* cross_cov,
		const vec_t& fitc_resid_diag,
		double& approx_marginal_ll,
		bool GPU_use) {
		ChecksBeforeModeFinding();
		fixed_effects_ = fixed_effects;
		int num_ip = (int)((*sigma_ip).rows());
		CHECK((int)((*cross_cov).rows()) == dim_mode_);
		CHECK((int)((*cross_cov).cols()) == num_ip);
		CHECK((int)fitc_resid_diag.size() == dim_mode_);
		// Initialize variables
		if (InitializeModeForModeFinding()) {
			SigmaI_mode_previous_value_ = SigmaI_mode_;
			vec_t v_aux_mode = chol_fact_sigma_ip.solve((*cross_cov).transpose() * SigmaI_mode_);
			mode_ = ((*cross_cov) * v_aux_mode) + (fitc_resid_diag.asDiagonal() * SigmaI_mode_);//initialize mode with Sigma^(t+1) * a = Sigma^(t+1) * (Sigma^t)^(-1) * mode^t, where t+1 = current iteration. Otherwise the initial approx_marginal_ll is not correct since SigmaI_mode != Sigma^(-1)mode
			// Note: avoid the inversion of Sigma = (cross_cov * sigma_ip^-1 * cross_cov^T + fitc_resid_diag) with the Woodbury formula since fitc_resid_diag can be zero.
			//       This is also the reason why we initilize with mode_ = Sigma * SigmaI_mode_ and not SigmaI_mode_ = Sigma^-1 mode_
		}
		chol_den_mat_t chol_M_woodbury;
		vec_t Dinv;
		if (kink_cliping_) {
			//CHECK((fitc_resid_diag.array() > 0.0).all());
			const double jitter_D = 1e-12 * (1.0 + fitc_resid_diag.cwiseAbs().maxCoeff());
			vec_t D_safe = fitc_resid_diag.array() + jitter_D;
			Dinv = D_safe.array().inverse().matrix();
			den_mat_t M(num_ip, num_ip);
			M = *sigma_ip;
			M.diagonal().array() *= JITTER_MULT_IP_FITC_FSA;
			M.noalias() += (*cross_cov).transpose() * Dinv.asDiagonal() * (*cross_cov);
			chol_M_woodbury.compute(M);
			CheckCholeskyFactorization(chol_M_woodbury, "FindModePostRandEffCalcMLLFITC covariance Woodbury matrix");
		}
		vec_t location_par;//location parameter = mode of random effects + fixed effects
		double* location_par_ptr;
		InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
		// Initialize objective function (LA approx. marginal likelihood) for use as convergence criterion
		approx_marginal_ll = -0.5 * (SigmaI_mode_.dot(mode_)) + LogLikelihood(y_data, y_data_int, location_par_ptr);
		double approx_marginal_ll_new = approx_marginal_ll;
		vec_t Wsqrt_diag(dim_mode_), sigma_ip_inv_cross_cov_T_rhs(num_ip), rhs(dim_mode_), Wsqrt_Sigma_rhs(dim_mode_), vaux(num_ip), vaux2(num_ip), vaux3(dim_mode_),
			mode_new(dim_mode_), SigmaI_mode_new, DW_plus_I_inv_diag(dim_mode_), SigmaI_mode_update, mode_update, W_times_DW_plus_I_inv_diag;//auxiliary variables for updating mode
		den_mat_t M_aux_Woodbury(num_ip, num_ip); // = sigma_ip + (*cross_cov).transpose() * D_plus_WI_inv_diag.asDiagonal() * (*cross_cov)
		// Start finding mode 
		int it;
		bool terminate_optim = false;
		bool has_NA_or_Inf = false;
		for (it = 0; it < maxit_mode_newton_; ++it) {
			// Calculate first and second derivative of log-likelihood
			CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);
			if (it == 0 || information_changes_during_mode_finding_) {
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, true);
				if (HasNegativeValueInformationLogLik()) {
					LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in the (diagonal) Hessian (or Fisher "
						"information) of the negative log-likelihood. Cannot have negative values when using the numerically stable "
						"version of Rasmussen and Williams (2006) for mode finding ");
				}
				Wsqrt_diag.array() = information_ll_.array().sqrt();
				DW_plus_I_inv_diag = (information_ll_.array() * fitc_resid_diag.array() + 1.).matrix().cwiseInverse();
				// Calculate Cholesky factor of sigma_ip + Sigma_nm^T * Wsqrt * DW_plus_I_inv_diag * Wsqrt * Sigma_nm
				M_aux_Woodbury = *sigma_ip;
				M_aux_Woodbury.diagonal().array() *= JITTER_MULT_IP_FITC_FSA;
				W_times_DW_plus_I_inv_diag = Wsqrt_diag;
				W_times_DW_plus_I_inv_diag.array() *= W_times_DW_plus_I_inv_diag.array();
				W_times_DW_plus_I_inv_diag.array() *= DW_plus_I_inv_diag.array();
				//M_aux_Woodbury += (*cross_cov).transpose() * W_times_DW_plus_I_inv_diag.asDiagonal() * (*cross_cov);// = *sigma_ip + (*cross_cov).transpose() * D_plus_WI_inv_diag.asDiagonal() * (*cross_cov)
				den_mat_t W_times_DW_plus_I_inv_diag_cross_cov = W_times_DW_plus_I_inv_diag.asDiagonal() * (*cross_cov);
				den_mat_t cross_cov_t_W_times_DW_plus_I_inv_diag_cross_cov;
				GPBoost::matmul((*cross_cov).transpose(), W_times_DW_plus_I_inv_diag_cross_cov, cross_cov_t_W_times_DW_plus_I_inv_diag_cross_cov, GPU_use);
				M_aux_Woodbury += cross_cov_t_W_times_DW_plus_I_inv_diag_cross_cov;
				chol_fact_dense_Newton_.compute(M_aux_Woodbury);//Cholesky factor of sigma_ip + Sigma_nm^T * Wsqrt * DW_plus_I_inv_diag * Wsqrt * Sigma_nm
				CheckCholeskyFactorization(chol_fact_dense_Newton_, "FindModePostRandEffCalcMLLFITC mode update");
			}
			rhs.array() = information_ll_.array() * mode_.array() + first_deriv_ll_.array();
			// Update mode and SigmaI_mode_
			sigma_ip_inv_cross_cov_T_rhs = chol_fact_sigma_ip.solve((*cross_cov).transpose() * rhs);
			Wsqrt_Sigma_rhs = ((*cross_cov) * sigma_ip_inv_cross_cov_T_rhs) + (fitc_resid_diag.asDiagonal() * rhs);//Sigma * rhs
			vaux = (*cross_cov).transpose() * (W_times_DW_plus_I_inv_diag.asDiagonal() * Wsqrt_Sigma_rhs);
			vaux2 = chol_fact_dense_Newton_.solve(vaux);
			Wsqrt_Sigma_rhs.array() *= Wsqrt_diag.array();//Wsqrt_Sigma_rhs = sqrt(W) * Sigma * rhs
			// Backtracking line search
			SigmaI_mode_update = DW_plus_I_inv_diag.asDiagonal() * (Wsqrt_Sigma_rhs - Wsqrt_diag.asDiagonal() * ((*cross_cov) * vaux2));
			SigmaI_mode_update.array() *= Wsqrt_diag.array();
			SigmaI_mode_update.array() *= -1.;
			SigmaI_mode_update.array() += rhs.array();//SigmaI_mode_ = rhs - sqrt(W) * Id_plus_Wsqrt_Sigma_Wsqrt^-1 * rhs2
			vaux3 = chol_fact_sigma_ip.solve((*cross_cov).transpose() * SigmaI_mode_update);
			mode_update = ((*cross_cov) * vaux3) + (fitc_resid_diag.asDiagonal() * SigmaI_mode_update);//mode_ = Sigma * SigmaI_mode_
			double grad_dot_direction = 0.;//for Armijo check
			if (armijo_condition_) {
				vec_t direction = mode_update - mode_;//mode_update = "new mode" = (Sigma^-1 + W)^-1 (W * mode + grad p(y|mode))
				grad_dot_direction = direction.dot(SigmaI_mode_update - SigmaI_mode_ + information_ll_.asDiagonal() * direction);
			}
			double lr_mode = 1.;
			for (int ih = 0; ih < max_number_lr_shrinkage_steps_newton_; ++ih) {
				if (ih == 0) {
					SigmaI_mode_new = SigmaI_mode_update;
					mode_new = mode_update;//mode_update = "new mode" = (Sigma^-1 + W)^-1 (W * mode + grad p(y|mode))
				}
				else {
					SigmaI_mode_new = (1 - lr_mode) * SigmaI_mode_ + lr_mode * SigmaI_mode_update;
					mode_new = (1 - lr_mode) * mode_ + lr_mode * mode_update;
				}
				if (kink_cliping_) {
					bool clipped = ApplyKinkClippingAsymLaplace(y_data, fixed_effects, mode_, mode_new);
					if (clipped) {// Recompute Sigma^{-1} mode_new so the quadratic term is consistent
						vec_t Dinv_x = Dinv.array() * mode_new.array(); // D^{-1} x							
						vec_t u = (*cross_cov).transpose() * Dinv_x;// u = cross_cov^T * D^{-1} x
						vec_t v = chol_M_woodbury.solve(u);
						SigmaI_mode_new = Dinv_x - (Dinv.asDiagonal() * ((*cross_cov) * v));// Sigma^{-1} x = D^{-1} x - D^{-1} cross_cov v
					}
				}
				//CapChangeModeUpdateNewton(mode_new);//not done since SigmaI_mode would also have to be modified accordingly. TODO: implement this?
				UpdateLocationParNewMode(mode_new, fixed_effects, location_par, &location_par_ptr); // Update location parameter of log-likelihood for calculation of approx. marginal log-likelihood (objective function)
				approx_marginal_ll_new = -0.5 * (SigmaI_mode_new.dot(mode_new)) + LogLikelihood(y_data, y_data_int, location_par_ptr);// Calculate new objective function
				if (AcceptModeUpdate(approx_marginal_ll_new, approx_marginal_ll, grad_dot_direction, lr_mode)) {
					break;
				}
			}// end loop over learnig rate halving procedure
			mode_ = mode_new;
			SigmaI_mode_ = SigmaI_mode_new;
			CheckConvergenceModeFinding(it, approx_marginal_ll_new, approx_marginal_ll, terminate_optim, has_NA_or_Inf);
			if (terminate_optim || has_NA_or_Inf) {
				break;
			}
		}//end for loop Newton's method
		if (!has_NA_or_Inf) {//calculate determinant
			mode_has_been_calculated_ = true;
			na_or_inf_during_last_call_to_find_mode_ = false;
			if (sample_from_posterior_after_mode_finding_) {
				Sample_Posterior_LaplaceApprox_FITC(cross_cov, fitc_resid_diag);
			}
			CalcFirstDerivLogLik(y_data, y_data_int, location_par_ptr);//first derivative is not used here anymore but since it is reused in gradient calculation and in prediction, we calculate it once more
			vec_t D_plus_WI_inv_diag;
			if (information_changes_after_mode_finding_) CalcInformationLogLik(y_data, y_data_int, location_par_ptr, false);
			if (HasNegativeValueInformationLogLikOnDataScale()) {
				LogFatalWithPotentialFisherLaplaceHint(__func__, "Negative values found in W (the diagonal Hessian or Fisher "
					"information of the negative log-likelihood). This is not permitted when using the FITC approximation ");
			}
			if (HasZeroValueInformationLogLik()) {
				Log::REFatal("FindModePostRandEffCalcMLLFITC: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
					"This is not permitted when using the FITC approximation ");
			}
			if (information_changes_after_mode_finding_) {
				D_plus_WI_inv_diag = (fitc_resid_diag + information_ll_.cwiseInverse()).cwiseInverse();
				M_aux_Woodbury = *sigma_ip;
				M_aux_Woodbury.diagonal().array() *= JITTER_MULT_IP_FITC_FSA;
				//M_aux_Woodbury += (*cross_cov).transpose() * D_plus_WI_inv_diag.asDiagonal() * (*cross_cov);
				den_mat_t D_plus_WI_inv_diag_cross_cov = D_plus_WI_inv_diag.asDiagonal() * (*cross_cov);
				den_mat_t cross_cov_t_D_plus_WI_inv_diag_cross_cov;
				GPBoost::matmul((*cross_cov).transpose(), D_plus_WI_inv_diag_cross_cov, cross_cov_t_D_plus_WI_inv_diag_cross_cov, GPU_use);
				M_aux_Woodbury += cross_cov_t_D_plus_WI_inv_diag_cross_cov;
				chol_fact_dense_Newton_.compute(M_aux_Woodbury);//Cholesky factor of (sigma_ip + Sigma_nm^T * D_plus_WI_inv_diag * Sigma_nm)
				CheckCholeskyFactorization(chol_fact_dense_Newton_, "FindModePostRandEffCalcMLLFITC determinant");
			}
			else {
				D_plus_WI_inv_diag = (fitc_resid_diag + information_ll_.cwiseInverse()).cwiseInverse();
			}
			approx_marginal_ll -= ((den_mat_t)chol_fact_dense_Newton_.matrixL()).diagonal().array().log().sum();
			approx_marginal_ll += ((den_mat_t)chol_fact_sigma_ip.matrixL()).diagonal().array().log().sum();
			approx_marginal_ll += 0.5 * D_plus_WI_inv_diag.array().log().sum();
			approx_marginal_ll -= 0.5 * information_ll_.array().log().sum();
		}
		FinalizeModeFinding(it);
	}//end FindModePostRandEffCalcMLLFITC

	template <typename T_mat, typename T_chol>
	vec_t Likelihood<T_mat, T_chol>::CalcStochDataScaleDiagSigmaIPlusZtWZInv() const {
		CHECK(rand_vec_trace_I_.rows() == dim_mode_);
		CHECK(rand_vec_trace_I_.cols() == num_rand_vec_trace_);
		den_mat_t SigmaI_plus_ZtWZ_inv_RV_I(dim_mode_, num_rand_vec_trace_);
		bool has_NA_or_Inf_stoch_diag = false;
		CGRandomEffectsMat(SigmaI_plus_ZtWZ_rm_, rand_vec_trace_I_, SigmaI_plus_ZtWZ_inv_RV_I, has_NA_or_Inf_stoch_diag,
			dim_mode_, num_rand_vec_trace_, cg_max_num_it_, cg_delta_conv_, cg_preconditioner_type_,
			L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_);
		if (has_NA_or_Inf_stoch_diag) {
			Log::REDebug(CG_NA_OR_INF_WARNING_GRADIENT_);
		}
		den_mat_t Z_SigmaI_plus_ZtWZ_inv_RV_I(num_data_, num_rand_vec_trace_), Z_RV_I(num_data_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_rand_vec_trace_; ++i) {
			Z_SigmaI_plus_ZtWZ_inv_RV_I.col(i) = (*Zt_).transpose() * SigmaI_plus_ZtWZ_inv_RV_I.col(i);
			Z_RV_I.col(i) = (*Zt_).transpose() * rand_vec_trace_I_.col(i);
		}
		return (Z_SigmaI_plus_ZtWZ_inv_RV_I.cwiseProduct(Z_RV_I)).rowwise().mean();
	}//end CalcStochDataScaleDiagSigmaIPlusZtWZInv

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcSecondFEBlockFixedEffectGrad(const double* y_data,
		const int* y_data_int,
		const double* location_par,
		const vec_t& information_data_scale,
		const vec_t& diag,
		const vec_t& impl,
		const data_size_t* index_map,
		bool include_coupled_zi_terms,
		vec_t& fixed_effect_grad) const {
		CHECK(HasSecondFEBlock());
		// 'fixed_effect_grad' was sized to num_data_ by the caller's 'fixed_effect_grad = -first_deriv_ll_' (first_deriv_ll_
		// is not aware of the extra fixed-effects-only block); grow it back to dim_location_par_, preserving the eta block
		if (fixed_effect_grad.size() < dim_location_par_) {
			fixed_effect_grad.conservativeResize(dim_location_par_);
		}
		// For an iid model there is no random effect / mode at all, so both the log-determinant and the
		// implicit-through-the-mode term vanish and neither 'diag' / 'impl' nor 'index_map' is read (see
		// 'SecondFEBlockGradNeedsDiag'). 'iid_model_' is false for every approximation other than the
		// only-one-grouped-RE one, so this is a no-op there
		const bool has_mode = !iid_model_;
		if (likelihood_type_ == "gaussian_heteroscedastic") {
			// Gradient wrt the fixed effects of the log-error variance. There is no random effect / mode for this block,
			// so there is no implicit derivative (through the mode) here, i.e. l_eta_zeta = 0
#pragma omp parallel for schedule(static)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.0;
				double dummy_mean_deriv, deriv_log_var;
				FirstDerivLogLikGaussianHeteroscedastic(y_data[i], location_par[i], location_par[i + num_data_], dummy_mean_deriv, deriv_log_var);
				double log_det_term = 0.;
				if (has_mode) {
					log_det_term = 0.5 * information_data_scale[i] * diag[index_map == nullptr ? i : index_map[i]];
				}
				fixed_effect_grad[i + num_data_] = -w * deriv_log_var - log_det_term;
			}
		}
		else if (IsZeroCensPowNormHetero()) {
			// log(sigma) block (zeta): direct score + log-det (dJ_eta/dzeta) + implicit-through-mode (l_eta_zeta)
#pragma omp parallel for schedule(static)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const double w = has_weights_ ? weights_[i] : 1.0;
				double diag_i = 0., impl_i = 0.;
				if (has_mode) {
					const data_size_t idx = index_map == nullptr ? i : index_map[i];
					diag_i = diag[idx];
					impl_i = impl[idx];
				}
				fixed_effect_grad[i + num_data_] = ZeroCensPowNormHeteroZetaGrad(y_data[i], location_par[i], location_par[i + num_data_],
					w, diag_i, impl_i);
			}
		}
		else {//IsRegressionZeroModel()
			// Structural-zero block (zeta). Hurdle decouples from eta (dJ_eta/dzeta = l_eta_zeta = 0) -> direct score only.
			// Zero-inflated counts COUPLE at zero counts, so the log-determinant and implicit terms are added as well
			if (IsHurdleRegression()) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					fixed_effect_grad[i + num_data_] = -w * HurdleRegression_dZeta(y_data[i], location_par[i + num_data_]);
				}
			}
			else if (!include_coupled_zi_terms || !has_mode) {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					ZICountRegQuant o; ZICountRegressionQuantities(y_data_int[i], location_par[i], location_par[i + num_data_], o);
					fixed_effect_grad[i + num_data_] = -w * o.dZeta;
				}
			}
			else {
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					const double w = has_weights_ ? weights_[i] : 1.0;
					const data_size_t idx = index_map == nullptr ? i : index_map[i];
					ZICountRegQuant o; ZICountRegressionQuantities(y_data_int[i], location_par[i], location_par[i + num_data_], o);
					fixed_effect_grad[i + num_data_] = -w * o.dZeta +
						0.5 * (w * RegressionZeroModel_dInfodZeta(location_par[i], location_par[i + num_data_], o)) * diag[idx] +
						(w * o.lEtaZeta) * impl[idx];
				}
			}
		}
	}//end CalcSecondFEBlockFixedEffectGrad

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::AccumulateAuxParGradTerms(const vec_t& deriv_information_aux_par,
		const vec_t& second_deriv_loc_aux_par,
		const vec_t& SigmaI_plus_W_inv_diag,
		const vec_t& SigmaI_plus_W_inv_d_mll_d_mode,
		bool accumulate_log_det,
		double& d_detmll_d_aux_par,
		double& implicit_derivative) const {
		double d_detmll = 0., implicit_deriv = 0.;
		if (use_random_effects_indices_of_data_) {
#pragma omp parallel for schedule(static) reduction(+:d_detmll, implicit_deriv)
			for (data_size_t i = 0; i < num_data_; ++i) {
				const data_size_t idx = random_effects_indices_of_data_[i];
				if (accumulate_log_det) {
					d_detmll += deriv_information_aux_par[i] * SigmaI_plus_W_inv_diag[idx];
				}
				if (grad_information_wrt_mode_non_zero_) {
					implicit_deriv += second_deriv_loc_aux_par[i] * SigmaI_plus_W_inv_d_mll_d_mode[idx];
				}
			}
		}
		else {
#pragma omp parallel for schedule(static) reduction(+:d_detmll, implicit_deriv)
			for (data_size_t i = 0; i < num_data_; ++i) {
				if (accumulate_log_det) {
					d_detmll += deriv_information_aux_par[i] * SigmaI_plus_W_inv_diag[i];
				}
				if (grad_information_wrt_mode_non_zero_) {
					implicit_deriv += second_deriv_loc_aux_par[i] * SigmaI_plus_W_inv_d_mll_d_mode[i];
				}
			}
		}
		d_detmll_d_aux_par += d_detmll;
		implicit_derivative += implicit_deriv;
	}//end AccumulateAuxParGradTerms

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcAuxParGradLaplaceExactDiag(const double* y_data,
		const int* y_data_int,
		const double* location_par,
		const vec_t& SigmaI_plus_W_inv_diag,
		const vec_t& SigmaI_plus_W_inv_d_mll_d_mode,
		double* aux_par_grad) {// not const: 'CalcGradNegLogLikAuxPars' is non-const
		vec_t neg_likelihood_deriv(num_aux_pars_estim_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
		vec_t second_deriv_loc_aux_par(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
		vec_t deriv_information_aux_par(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
		CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par, neg_likelihood_deriv.data());
		for (int ind_ap = 0; ind_ap < num_aux_pars_estim_; ++ind_ap) {
			CalcSecondDerivLogLikFirstDerivInformationAuxPar(y_data, y_data_int, location_par, ind_ap, second_deriv_loc_aux_par.data(), deriv_information_aux_par.data());
			double d_detmll_d_aux_par = 0., implicit_derivative = 0.;
			AccumulateAuxParGradTerms(deriv_information_aux_par, second_deriv_loc_aux_par, SigmaI_plus_W_inv_diag,
				SigmaI_plus_W_inv_d_mll_d_mode, true, d_detmll_d_aux_par, implicit_derivative);
			aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
		}
		SetGradAuxParsNotEstimated(aux_par_grad);
	}//end CalcAuxParGradLaplaceExactDiag

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcGradNegMargLikelihoodLaplaceApproxStable(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const std::shared_ptr<T_mat>& Sigma,
		const std::vector<std::shared_ptr<RECompBase<T_mat>>>& re_comps_cluster_i,
		bool calc_cov_grad,
		bool calc_F_grad,
		bool calc_aux_par_grad,
		double* cov_grad,
		vec_t& fixed_effect_grad,
		double* aux_par_grad,
		bool calc_mode,
		bool call_for_std_dev_coef,
		const std::vector<int>& estimate_cov_par_index) {
		if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * Sigma * Wsqrt) at mode
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLStable(y_data, y_data_int, fixed_effects, Sigma, mll);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			if (call_for_std_dev_coef) {
				Log::REFatal(CANNOT_CALC_STDEV_ERROR_);
			}
			else {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
		}
		CHECK(mode_has_been_calculated_);
		// Initialize variables
		vec_t location_par;//location parameter = mode of random effects + fixed effects
		double* location_par_ptr;
		InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
		vec_t deriv_information_diag_loc_par;//first derivative of the diagonal of the Fisher information wrt the location parameter (= usually negative third derivatives of the log-likelihood wrt the locatin parameter)
		vec_t deriv_information_diag_loc_par_data_scale;//first derivative of the diagonal of the Fisher information wrt the location parameter on the data-scale (only used if use_random_effects_indices_of_data_), the vector 'deriv_information_diag_loc_par' actually contains diag_ZtDerivInformationZ if use_random_effects_indices_of_data_
		CHECK(num_sets_re_ == 1);
		if (grad_information_wrt_mode_non_zero_) {
			CalcFirstDerivInformationLocPar(y_data, y_data_int, location_par_ptr, deriv_information_diag_loc_par, deriv_information_diag_loc_par_data_scale);
		}
		if (HasNegativeValueInformationLogLik()) {
			Log::REFatal("CalcGradNegMargLikelihoodLaplaceApproxStable: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
				"This gradient calculation requires the square root of W ");
		}
		T_mat L_inv_Wsqrt(dim_mode_, dim_mode_);//L_inv_Wsqrt = L\ZtWZsqrt if use_random_effects_indices_of_data_ or L\Wsqrt if !use_random_effects_indices_of_data_ where L is a Cholesky factor of Id_plus_Wsqrt_Sigma_Wsqrt
		L_inv_Wsqrt.setIdentity();
		L_inv_Wsqrt.diagonal().array() = information_ll_.array().sqrt();
		TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, L_inv_Wsqrt, L_inv_Wsqrt, false);//L_inv_Wsqrt = L\Wsqrt
		vec_t SigmaI_plus_W_inv_diag, d_mll_d_mode;
		T_mat L_inv_Wsqrt_Sigma;
		if (grad_information_wrt_mode_non_zero_ || calc_aux_par_grad || (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad)) {
			L_inv_Wsqrt_Sigma = L_inv_Wsqrt * (*Sigma);
			//Log::REInfo("CalcGradNegMargLikelihoodLaplaceApproxStable: L_inv_ZtWZsqrt: number non zeros = %d", GetNumberNonZeros<T_mat>(L_inv_ZtWZsqrt));//Only for debugging
			//Log::REInfo("CalcGradNegMargLikelihoodLaplaceApproxStable: L_inv_ZtWZsqrt_Sigma: number non zeros = %d", GetNumberNonZeros<T_mat>(L_inv_ZtWZsqrt_Sigma));//Only for debugging
			// Calculate gradient of approx. marginal log-likelihood wrt the mode
			//      Note: use (i) (Sigma^-1 + W)^-1 = Sigma - Sigma*(W^-1 + Sigma)^-1*Sigma = Sigma - L_inv_Wsqrt_Sigma^T * L_inv_Wsqrt_Sigma and (ii) "Z=Id"
			T_mat L_inv_Wsqrt_Sigma_sqr = L_inv_Wsqrt_Sigma.cwiseProduct(L_inv_Wsqrt_Sigma);
			SigmaI_plus_W_inv_diag = (*Sigma).diagonal() - L_inv_Wsqrt_Sigma_sqr.transpose() * vec_t::Ones(L_inv_Wsqrt_Sigma_sqr.rows());// diagonal of (Sigma^-1 + ZtWZ) ^ -1 if use_random_effects_indices_of_data_ or of (ZSigmaZt^-1 + W)^-1 if !use_random_effects_indices_of_data_
		}
		if (grad_information_wrt_mode_non_zero_) {
			CHECK(first_deriv_information_loc_par_caluclated_);
			d_mll_d_mode = (0.5 * SigmaI_plus_W_inv_diag.array() * deriv_information_diag_loc_par.array()).matrix();// gradient of approx. marginal likelihood wrt the mode
		}
		// Calculate gradient wrt covariance parameters
		bool some_cov_par_estimated = std::any_of(estimate_cov_par_index.begin(), estimate_cov_par_index.end(), [](int x) { return x > 0; });
		if (calc_cov_grad && some_cov_par_estimated) {
			T_mat WI_plus_Sigma_inv;//WI_plus_Sigma_inv = ZtWZsqrt * L^T\(L\ZtWZsqrt) = ((ZtWZ)^-1 + Sigma)^-1 if use_random_effects_indices_of_data_ or Wsqrt * L^T\(L\Wsqrt) = (W^-1 + ZSigmaZt)^-1 if !use_random_effects_indices_of_data_
			vec_t d_mode_d_par, SigmaDeriv_first_deriv_ll; //auxiliary variable for caclulating d_mode_d_par
			int par_count = 0;
			for (int j = 0; j < (int)re_comps_cluster_i.size(); ++j) {
				for (int ipar = 0; ipar < re_comps_cluster_i[j]->NumCovPar(); ++ipar) {
					std::shared_ptr<T_mat> SigmaDeriv;
					if (estimate_cov_par_index[par_count] > 0 || ipar == 0) {
						SigmaDeriv = re_comps_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 1.);
					}
					if (ipar == 0) {
						WI_plus_Sigma_inv = *SigmaDeriv;
						CalcLtLGivenSparsityPattern<T_mat>(L_inv_Wsqrt, WI_plus_Sigma_inv, true);
						//TODO (low-prio): calculate WI_plus_Sigma_inv only once for all relevant non-zero entries as in Gaussian case (see 'CalcPsiInv')
						//                  This is only relevant for multiple random effects and/or GPs
					}
					if (estimate_cov_par_index[par_count] > 0) {
						// Calculate explicit derivative of approx. mariginal log-likelihood
						double explicit_derivative = -0.5 * (double)(SigmaI_mode_.transpose() * (*SigmaDeriv) * SigmaI_mode_) +
							0.5 * (WI_plus_Sigma_inv.cwiseProduct(*SigmaDeriv)).sum();
						cov_grad[par_count] = explicit_derivative;
						if (grad_information_wrt_mode_non_zero_) {
							// Calculate implicit derivative (through mode) of approx. mariginal log-likelihood
							SigmaDeriv_first_deriv_ll = (*SigmaDeriv) * first_deriv_ll_;
							d_mode_d_par = SigmaDeriv_first_deriv_ll;//derivative of mode wrt to a covariance parameter
							d_mode_d_par -= ((*Sigma) * (L_inv_Wsqrt.transpose() * (L_inv_Wsqrt * SigmaDeriv_first_deriv_ll)));
							cov_grad[par_count] += d_mll_d_mode.dot(d_mode_d_par);
						}
					}
					par_count++;
				}
			}
		}//end calc_cov_grad
		// calculate gradient wrt fixed effects
		vec_t SigmaI_plus_W_inv_d_mll_d_mode;// for implicit derivative
		if (grad_information_wrt_mode_non_zero_ && (calc_F_grad || calc_aux_par_grad)) {
			vec_t L_inv_Wsqrt_Sigma_d_mll_d_mode = L_inv_Wsqrt_Sigma * d_mll_d_mode;// for implicit derivative
			SigmaI_plus_W_inv_d_mll_d_mode = (*Sigma) * d_mll_d_mode - L_inv_Wsqrt_Sigma.transpose() * L_inv_Wsqrt_Sigma_d_mll_d_mode;
		}
		if (calc_F_grad) {
			if (use_random_effects_indices_of_data_) {
				fixed_effect_grad = -first_deriv_ll_data_scale_;
				if (grad_information_wrt_mode_non_zero_) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						fixed_effect_grad[i] += 0.5 * deriv_information_diag_loc_par_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
							information_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
					}
				}
				if (HasSecondFEBlock()) {
					CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_data_scale_,
						SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_d_mll_d_mode, random_effects_indices_of_data_, true, fixed_effect_grad);
				}
			}//end use_random_effects_indices_of_data_
			else {
				fixed_effect_grad = -first_deriv_ll_;
				if (grad_information_wrt_mode_non_zero_) {
					vec_t d_mll_d_F_implicit = (SigmaI_plus_W_inv_d_mll_d_mode.array() * information_ll_.array()).matrix();// implicit derivative
					fixed_effect_grad += d_mll_d_mode - d_mll_d_F_implicit;
				}
				if (HasSecondFEBlock()) {
					CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_,
						SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_d_mll_d_mode, nullptr, true, fixed_effect_grad);
				}
			}//end !use_random_effects_indices_of_data_
		}//end calc_F_grad
		// calculate gradient wrt additional likelihood parameters
		if (calc_aux_par_grad) {
			CalcAuxParGradLaplaceExactDiag(y_data, y_data_int, location_par_ptr, SigmaI_plus_W_inv_diag,
				SigmaI_plus_W_inv_d_mll_d_mode, aux_par_grad);
		}//end calc_aux_par_grad
	}//end CalcGradNegMargLikelihoodLaplaceApproxStable

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcGradNegMargLikelihoodLaplaceApproxGroupedRE(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const sp_mat_t& SigmaI,
		bool has_vecchia_gp,
		const sp_mat_t& B,
		const sp_mat_t& D_inv,
		const std::vector<sp_mat_t>& B_grad,
		const std::vector<sp_mat_t>& D_grad,
		const std::vector<data_size_t>& cum_num_rand_eff_cluster_i,
		bool calc_cov_grad,
		bool calc_F_grad,
		bool calc_aux_par_grad,
		double* cov_grad,
		vec_t& fixed_effect_grad,
		double* aux_par_grad,
		bool calc_mode,
		bool call_for_std_dev_coef,
		const std::vector<int>& estimate_cov_par_index) {
		CHECK(cum_num_rand_eff_cluster_i.back() == dim_mode_);
		const data_size_t num_grouped_RE = has_vecchia_gp ? (data_size_t)cum_num_rand_eff_cluster_i.size() - 2 : (data_size_t)cum_num_rand_eff_cluster_i.size() - 1;//number of different grouped random effect components
		const data_size_t dim_re_group = cum_num_rand_eff_cluster_i[num_grouped_RE];//number of grouped random effects
		CHECK(SigmaI.cols() == dim_re_group);
		data_size_t dim_gp = 0;//number of GP random effects
		if (has_vecchia_gp) {
			dim_gp = cum_num_rand_eff_cluster_i[num_grouped_RE + 1] - cum_num_rand_eff_cluster_i[num_grouped_RE];
			CHECK(B.rows() == dim_gp);
			CHECK(B.cols() == dim_gp);
			CHECK(D_inv.rows() == dim_gp);
			CHECK(dim_gp > 0);
		}
		CHECK(dim_re_group + dim_gp == dim_mode_);
		if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
			double mll_dummy;
			FindModePostRandEffCalcMLLGroupedRE(y_data, y_data_int, fixed_effects, SigmaI, has_vecchia_gp, B, D_inv, false, true, mll_dummy);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			if (call_for_std_dev_coef) {
				Log::REFatal(CANNOT_CALC_STDEV_ERROR_);
			}
			else {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
		}
		CHECK(mode_has_been_calculated_);
		// Initialize variables
		vec_t location_par;
		double* location_par_ptr_dummy;//not used
		UpdateLocationParNewMode(mode_, fixed_effects, location_par, &location_par_ptr_dummy);
		if (matrix_inversion_method_ == "iterative") {
			// calculate P^(-1) RV
			den_mat_t PI_RV(dim_mode_, num_rand_vec_trace_), L_inv_Z, DI_L_plus_D_t_PI_RV;
			if (cg_preconditioner_type_ == "incomplete_cholesky") {
				L_inv_Z.resize(dim_mode_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					L_inv_Z.col(i) = L_SigmaI_plus_ZtWZ_rm_.triangularView<Eigen::Lower>().solve(rand_vec_trace_P_.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					PI_RV.col(i) = (L_SigmaI_plus_ZtWZ_rm_.transpose().template triangularView<Eigen::Upper>()).solve(L_inv_Z.col(i));
				}
			}
			else if (cg_preconditioner_type_ == "ssor") {
				L_inv_Z.resize(dim_mode_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					L_inv_Z.col(i) = P_SSOR_L_D_sqrt_inv_rm_.triangularView<Eigen::Lower>().solve(rand_vec_trace_P_.col(i));
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					PI_RV.col(i) = (P_SSOR_L_D_sqrt_inv_rm_.transpose().template triangularView<Eigen::Upper>()).solve(L_inv_Z.col(i));
				}
				//For variance reduction
				DI_L_plus_D_t_PI_RV.resize(dim_mode_, num_rand_vec_trace_);
				den_mat_t L_plus_D_t_PI_RV(dim_mode_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					L_plus_D_t_PI_RV.col(i) = SigmaI_plus_ZtWZ_rm_.triangularView<Eigen::Upper>() * PI_RV.col(i);
				}
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					DI_L_plus_D_t_PI_RV.col(i) = P_SSOR_D_inv_.asDiagonal() * L_plus_D_t_PI_RV.col(i);
				}
			}
			else {
				Log::REFatal("Preconditioner type '%s' is not supported for calculating gradients ", cg_preconditioner_type_.c_str());
			}
			// calculate Z P^(-1) z_i
			den_mat_t Z_PI_RV(num_data_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < num_rand_vec_trace_; ++i) {
				Z_PI_RV.col(i) = (*Zt_).transpose() * PI_RV.col(i);
			}
			// calculate Z P^(-1) z_i
			CHECK(SigmaI_plus_ZtWZ_inv_RV_.rows() == dim_mode_);
			CHECK(SigmaI_plus_ZtWZ_inv_RV_.cols() == num_rand_vec_trace_);
			den_mat_t Z_SigmaI_plus_ZtWZ_inv_RV(num_data_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < num_rand_vec_trace_; ++i) {
				Z_SigmaI_plus_ZtWZ_inv_RV.col(i) = (*Zt_).transpose() * SigmaI_plus_ZtWZ_inv_RV_.col(i);
			}
			//calculate gradient of approx. marginal likelihood wrt the mode
			vec_t trace_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_loc_par_Z;
			vec_t Z_SigmaI_plus_ZtWZ_inv_d_mll_d_mode;
			vec_t SigmaI_plus_ZtWZ_inv_d_mll_d_mode;
			if (grad_information_wrt_mode_non_zero_) {
				vec_t deriv_information_diag_loc_par(num_data_);//usually vector of negative third derivatives of log-likelihood
				CalcFirstDerivInformationLocPar_PerSample(y_data, y_data_int, location_par.data(), deriv_information_diag_loc_par);
				//Stochastic trace: tr((Sigma^(-1) + Z^T W Z)^(-1) Z^T dW/dloc_par Z)
				den_mat_t W_deriv_loc_par_rep = deriv_information_diag_loc_par.replicate(1, num_rand_vec_trace_);
				den_mat_t RVt_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_loc_par_Z_PI_RV = (Z_SigmaI_plus_ZtWZ_inv_RV.array() * W_deriv_loc_par_rep.array() * Z_PI_RV.array()).matrix();
				trace_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_loc_par_Z = RVt_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_loc_par_Z_PI_RV.rowwise().mean();
				//Stochastic trace: tr((Sigma^(-1) + Z^T W Z)^(-1) Z^T dW/db_j Z)
				vec_t d_mll_d_mode = (*Zt_) * trace_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_loc_par_Z;
				d_mll_d_mode *= 0.5;
				//For implicit derivatives: calculate (Sigma^(-1) + Z^T W Z)^(-1) d_mll_d_mode
				bool has_NA_or_Inf = false;
				SigmaI_plus_ZtWZ_inv_d_mll_d_mode = vec_t(dim_mode_);
				int num_cg_steps_dummy;
				CGRandomEffectsVec(SigmaI_plus_ZtWZ_rm_, d_mll_d_mode, SigmaI_plus_ZtWZ_inv_d_mll_d_mode, has_NA_or_Inf,
					cg_max_num_it_, cg_delta_conv_pred_, true, ZERO_RHS_CG_THRESHOLD, false, cg_preconditioner_type_,
					L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps_dummy);
				if (has_NA_or_Inf) {
					Log::REDebug(CG_NA_OR_INF_WARNING_GRADIENT_);
				}
				if (calc_F_grad || calc_aux_par_grad) {
					Z_SigmaI_plus_ZtWZ_inv_d_mll_d_mode = (*Zt_).transpose() * SigmaI_plus_ZtWZ_inv_d_mll_d_mode;
				}
			}
			// calculate gradient wrt covariance parameters
			bool some_cov_par_estimated = std::any_of(estimate_cov_par_index.begin(), estimate_cov_par_index.end(), [](int x) { return x > 0; });
			if (calc_cov_grad && some_cov_par_estimated) {
				vec_t SigmaI_mode;
				if (has_vecchia_gp) {
					SigmaI_mode = vec_t(dim_mode_);
					SigmaI_mode.segment(0, dim_re_group) = SigmaI * mode_.segment(0, dim_re_group);
					SigmaI_mode.segment(dim_re_group, dim_gp) = B.transpose() * (D_inv * (B * (mode_.segment(dim_re_group, dim_gp))));
				}
				else {
					SigmaI_mode = SigmaI * mode_;
				}
				sp_mat_t I_j(dim_mode_, dim_mode_);
				for (int j = 0; j < num_grouped_RE; ++j) {
					if (estimate_cov_par_index[j] > 0) {
						// calculate explicit derivative of approx. mariginal log-likelihood
						std::vector<Triplet_t> triplets(cum_num_rand_eff_cluster_i[j + 1] - cum_num_rand_eff_cluster_i[j]);
						double explicit_derivative = 0.;
#pragma omp parallel for schedule(static) reduction(+:explicit_derivative)
						for (int i = cum_num_rand_eff_cluster_i[j]; i < cum_num_rand_eff_cluster_i[j + 1]; ++i) {
							triplets[i - cum_num_rand_eff_cluster_i[j]] = Triplet_t(i, i, 1.);
							explicit_derivative += SigmaI_mode[i] * mode_[i];
						}
						explicit_derivative *= -0.5;
						I_j.setFromTriplets(triplets.begin(), triplets.end());
						double cov_par_inv = SigmaI.coeff(cum_num_rand_eff_cluster_i[j], cum_num_rand_eff_cluster_i[j]);
						//Stochastic trace: tr((Sigma^(-1) + Z^T W Z)^(-1) dSigma^(-1)/dtheta_j)
						vec_t RVt_SigmaI_plus_ZtWZ_inv_SigmaI_deriv_PI_RV = -cov_par_inv * ((SigmaI_plus_ZtWZ_inv_RV_.cwiseProduct(I_j * PI_RV)).colwise().sum()).transpose(); //old: -1. * ((SigmaI_plus_ZtWZ_inv_RV_.cwiseProduct((I_j * SigmaI.coeff(cum_num_rand_eff_cluster_i[j], cum_num_rand_eff_cluster_i[j])) * PI_RV)).colwise().sum()).transpose();
						double trace_SigmaI_plus_ZtWZ_inv_SigmaI_deriv = RVt_SigmaI_plus_ZtWZ_inv_SigmaI_deriv_PI_RV.mean();
						if (cg_preconditioner_type_ == "ssor" && !has_vecchia_gp) {//Variance reduction								
							//deterministic tr(D^(-1) dSigma^(-1)/dtheta_j)
							double tr_D_inv_SigmaI_deriv = -cov_par_inv * (P_SSOR_D_inv_.cwiseProduct(I_j.diagonal())).sum();
							//stochastic tr(P^(-1) dP/dtheta_j)
							den_mat_t neg_SigmaI_deriv_DI_L_plus_D_t_PI_RV = cov_par_inv * (I_j * DI_L_plus_D_t_PI_RV);
							vec_t RVt_PI_P_deriv_PI_RV = -2. * ((PI_RV.cwiseProduct(neg_SigmaI_deriv_DI_L_plus_D_t_PI_RV)).colwise().sum()).transpose();
							RVt_PI_P_deriv_PI_RV += ((DI_L_plus_D_t_PI_RV.cwiseProduct(neg_SigmaI_deriv_DI_L_plus_D_t_PI_RV)).colwise().sum()).transpose();
							double tr_PI_P_deriv = RVt_PI_P_deriv_PI_RV.mean();								
							double c_opt;//optimal c
							CalcOptimalC(RVt_SigmaI_plus_ZtWZ_inv_SigmaI_deriv_PI_RV, RVt_PI_P_deriv_PI_RV, trace_SigmaI_plus_ZtWZ_inv_SigmaI_deriv, tr_PI_P_deriv, c_opt);
							trace_SigmaI_plus_ZtWZ_inv_SigmaI_deriv += c_opt * (tr_D_inv_SigmaI_deriv - tr_PI_P_deriv);
						}
						explicit_derivative += 0.5 * (trace_SigmaI_plus_ZtWZ_inv_SigmaI_deriv + cum_num_rand_eff_cluster_i[j + 1] - cum_num_rand_eff_cluster_i[j]);
						cov_grad[j] = explicit_derivative;
						if (grad_information_wrt_mode_non_zero_) {
							// calculate implicit derivative (through mode) of approx. mariginal log-likelihood
							cov_grad[j] += SigmaI_plus_ZtWZ_inv_d_mll_d_mode.dot(I_j * ((*Zt_) * first_deriv_ll_));
						}
					}
				}//end loop j < num_grouped_RE
				if (has_vecchia_gp) {
					CHECK(num_sets_re_ == 1);
					const int num_par_gp = (int)B_grad.size();
					CHECK((int)D_grad.size() == num_par_gp);
					sp_mat_t D_inv_B = D_inv * B;// D_inv_B for GP block
					den_mat_t SigmaI_deriv_PI_RV(dim_mode_, num_rand_vec_trace_);
					SigmaI_deriv_PI_RV.setZero();
					vec_t RVt_trace(num_rand_vec_trace_);// For computing trace estimator: tr(A^{-1} SigmaI_deriv). Hutchinson w/ vectors PI_RV: mean_j ( (A^{-1}PI_RV_j)^T (SigmaI_deriv PI_RV_j) )
					const vec_t mode_gp = mode_.segment(dim_re_group, dim_gp);// For explicit term: mode^T SigmaI_deriv mode (only GP block)
					// For implicit term: (A^{-1} d_mll_d_mode)^T (SigmaI_deriv * (Z^T first_deriv_ll_)),  rhs = SigmaI_deriv * (Zt * first_deriv_ll_) efficiently on GP block
					const vec_t Zt_first = (*Zt_) * first_deriv_ll_;  // dim_mode_
					vec_t rhs(dim_mode_);
					rhs.setZero();
					sp_mat_t SigmaI_deriv_gp;     // dim_gp x dim_gp
					sp_mat_t Bt_Dinv_Bgrad, BgradT_Dinv_B;
					for (int p = 0; p < num_par_gp; ++p) {
						const int cov_ind = (int)num_grouped_RE + p;
						if (cov_ind >= (int)estimate_cov_par_index.size()) {
							Log::REFatal("estimate_cov_par_index too short for Vecchia GP params in CalcGradNegMargLikelihoodLaplaceApproxGroupedRE");
						}
						if (estimate_cov_par_index[cov_ind] <= 0) {
							continue;
						}
						// Build SigmaI_deriv_gp for GP precision: d(B^T D^{-1} B)/dtheta = B_grad^T D^{-1} B + B^T D^{-1} B_grad - (D^{-1}B)^T D_grad (D^{-1}B)
						if (p == 0) {
							SigmaI_deriv_gp = -B.transpose() * D_inv_B;  // Special case: if p==0 is the variance parameter: SigmaI_deriv_gp = -SigmaI_gp
						}
						else {
							BgradT_Dinv_B = B_grad[p].transpose() * D_inv_B;
							Bt_Dinv_Bgrad = BgradT_Dinv_B.transpose();
							SigmaI_deriv_gp = BgradT_Dinv_B + Bt_Dinv_Bgrad - D_inv_B.transpose() * D_grad[p] * D_inv_B;
							Bt_Dinv_Bgrad.resize(0, 0);
							BgradT_Dinv_B.resize(0, 0);
						}
						// Explicit term: 0.5 * mode^T SigmaI_deriv mode
						const vec_t SigmaI_deriv_mode_gp = SigmaI_deriv_gp * mode_gp;
						double explicit_derivative = 0.5 * mode_gp.dot(SigmaI_deriv_mode_gp);
						// Trace term: 0.5 * tr(A^{-1} SigmaI_deriv) via Hutchinson. Compute SigmaI_deriv * PI_RV (only GP block nonzero)
						SigmaI_deriv_PI_RV.setZero();
						SigmaI_deriv_PI_RV.block(dim_re_group, 0, dim_gp, num_rand_vec_trace_) = (SigmaI_deriv_gp * PI_RV.block(dim_re_group, 0, dim_gp, num_rand_vec_trace_));
						// RVt_trace[j] = (A^{-1}PI_RV_j)^T (SigmaI_deriv PI_RV_j)
						RVt_trace = (SigmaI_plus_ZtWZ_inv_RV_.cwiseProduct(SigmaI_deriv_PI_RV)).colwise().sum().transpose();
						const double tr_Ainv_SigmaIderiv = RVt_trace.mean();
						explicit_derivative += 0.5 * tr_Ainv_SigmaIderiv;
						//  Log-determinant term for +0.5 log|Sigma|
						if (p == 0) {
							explicit_derivative += 0.5 * dim_gp;
						}
						else {
							explicit_derivative += 0.5 * (D_inv.diagonal().array() * D_grad[p].diagonal().array()).sum();
						}
						cov_grad[cov_ind] = explicit_derivative;
						// Implicit term through mode
						// Cholesky version uses: d_mll_d_mode^T * A^{-1} * (SigmaI_deriv * (Zt*first_deriv_ll_))
						// Iterative: we already computed A^{-1} d_mll_d_mode as SigmaI_plus_ZtWZ_inv_d_mll_d_mode
						if (grad_information_wrt_mode_non_zero_) {
							rhs.setZero();
							rhs.segment(dim_re_group, dim_gp) = SigmaI_deriv_gp * Zt_first.segment(dim_re_group, dim_gp);
							cov_grad[cov_ind] += SigmaI_plus_ZtWZ_inv_d_mll_d_mode.dot(rhs);
						}
						SigmaI_deriv_gp.resize(0, 0);
					} // end loop over GP params
				} // end has_vecchia_gp
			}//end calc_cov_grad
			// calculate gradient wrt fixed effects
			if (calc_F_grad) {
				fixed_effect_grad = -first_deriv_ll_;
				if (grad_information_wrt_mode_non_zero_) {
					fixed_effect_grad += 0.5 * trace_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_loc_par_Z - Z_SigmaI_plus_ZtWZ_inv_d_mll_d_mode.cwiseProduct(information_ll_);
				}
				// Second (zeta) block on the ITERATIVE grouped-RE path, calculated separately from the eta block above. Its
				// log-determinant term needs the DATA-scale diagonal of (Sigma^-1+ZtWZ)^-1, which is only available as a
				// stochastic estimate here (the ratio trick used for the eta block is not applicable since dJ_eta/deta
				// vanishes at the observations that matter for these likelihoods). Note that 'gaussian_heteroscedastic'
				// only reaches this point with grad_information_wrt_mode_non_zero_ == false (its information does not
				// depend on the mode), which is why the eta-block branch above and this one are mutually exclusive for it
				if (HasSecondFEBlock()) {
					// This branch is entered unconditionally, whereas the zeta block of 'gaussian_heteroscedastic' used to be
					// calculated only when the eta-block branch above was NOT entered. The two are equivalent only as long as
					// that likelihood's information does not depend on the mode, which is asserted here
					CHECK(likelihood_type_ != "gaussian_heteroscedastic" || !grad_information_wrt_mode_non_zero_);
					vec_t diag_data;
					if (SecondFEBlockGradNeedsDiag(grad_information_wrt_mode_non_zero_)) {
						diag_data = CalcStochDataScaleDiagSigmaIPlusZtWZInv();
					}
					CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par.data(), information_ll_, diag_data,
						Z_SigmaI_plus_ZtWZ_inv_d_mll_d_mode, nullptr, grad_information_wrt_mode_non_zero_, fixed_effect_grad);
				}
			}//end calc_F_grad
			// calculate gradient wrt additional likelihood parameters
			if (calc_aux_par_grad) {
				vec_t neg_likelihood_deriv(num_aux_pars_estim_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
				vec_t second_deriv_loc_aux_par(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
				vec_t deriv_information_aux_par(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
				vec_t d_mode_d_aux_par;
				CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par.data(), neg_likelihood_deriv.data());
				for (int ind_ap = 0; ind_ap < num_aux_pars_estim_; ++ind_ap) {
					CalcSecondDerivLogLikFirstDerivInformationAuxPar(y_data, y_data_int, location_par.data(), ind_ap, second_deriv_loc_aux_par.data(), deriv_information_aux_par.data());
					//stochastic tr((Sigma^(-1) + Z^T W Z)^(-1) Z^T dW/daux Z)
					vec_t RVt_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_aux_Z_PI_RV = ((Z_SigmaI_plus_ZtWZ_inv_RV.cwiseProduct(deriv_information_aux_par.asDiagonal() * Z_PI_RV)).colwise().sum()).transpose();
					double tr_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_aux_Z = RVt_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_aux_Z_PI_RV.mean();
					double d_detmll_d_aux_par = tr_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_aux_Z;
					if (cg_preconditioner_type_ == "ssor") {
						//Variance reduction
						sp_mat_t ZtdWZ = (*Zt_) * deriv_information_aux_par.asDiagonal() * (*Zt_).transpose();
						//deterministic tr(D^(-1) diag(Z^T dW/daux Z))
						double tr_D_inv_diag_Zt_W_deriv_aux_Z = (P_SSOR_D_inv_.cwiseProduct(ZtdWZ.diagonal())).sum();
						//stochastic tr(P^(-1) dP/daux)
						den_mat_t Ltriang_Zt_W_deriv_aux_Z_DI_L_plus_D_t_PI_RV = ZtdWZ.triangularView<Eigen::Lower>() * DI_L_plus_D_t_PI_RV;
						vec_t RVt_PI_P_deriv_PI_RV = 2. * ((PI_RV.cwiseProduct(Ltriang_Zt_W_deriv_aux_Z_DI_L_plus_D_t_PI_RV)).colwise().sum()).transpose();
						RVt_PI_P_deriv_PI_RV -= ((DI_L_plus_D_t_PI_RV.cwiseProduct(Ltriang_Zt_W_deriv_aux_Z_DI_L_plus_D_t_PI_RV)).colwise().sum()).transpose();
						double tr_PI_P_deriv = RVt_PI_P_deriv_PI_RV.mean();
						//optimal c
						double c_opt;
						CalcOptimalC(RVt_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_aux_Z_PI_RV, RVt_PI_P_deriv_PI_RV, tr_SigmaI_plus_ZtWZ_inv_Zt_W_deriv_aux_Z, tr_PI_P_deriv, c_opt);
						d_detmll_d_aux_par += c_opt * (tr_D_inv_diag_Zt_W_deriv_aux_Z - tr_PI_P_deriv);
					}
					aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par;
					if (grad_information_wrt_mode_non_zero_) {
						aux_par_grad[ind_ap] += Z_SigmaI_plus_ZtWZ_inv_d_mll_d_mode.dot(second_deriv_loc_aux_par);
					}
				}
				SetGradAuxParsNotEstimated(aux_par_grad);
			}//end calc_aux_par_grad
		}//end iterative
		else {//Cholesky decomposition
			// Calculate (Sigma^-1 + Zt*W*Z)^-1
			sp_mat_t L_inv(dim_mode_, dim_mode_);
			L_inv.setIdentity();
			if (chol_fact_SigmaI_plus_ZtWZ_grouped_.permutationP().size() > 0) {//Permutation is only used when having an ordering
				L_inv = chol_fact_SigmaI_plus_ZtWZ_grouped_.permutationP() * L_inv;
			}
			sp_mat_t L = chol_fact_SigmaI_plus_ZtWZ_grouped_.matrixL();
			TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(L, L_inv, L_inv, false);
			L.resize(0, 0);
			sp_mat_t SigmaI_plus_ZtWZ_inv;
			// calculate gradient of approx. marginal likelihood wrt the mode
			vec_t deriv_information_diag_loc_par;//usually vector of negative third derivatives of log-likelihood
			vec_t d_mll_d_mode;
			if (grad_information_wrt_mode_non_zero_) {
				deriv_information_diag_loc_par = vec_t(num_data_);
				CalcFirstDerivInformationLocPar_PerSample(y_data, y_data_int, location_par.data(), deriv_information_diag_loc_par);
				sp_mat_t Zt_deriv_information_loc_par = (*Zt_) * deriv_information_diag_loc_par.asDiagonal();//every column of Z multiplied elementwise by deriv_information_diag_loc_par
				// Note: Z^T * diag(diag_d_W_d_mode_i) * Z = Z^T * diag(Z.col(i) * deriv_information_diag_loc_par) * Z
				// Precompute || L_inv * Z^T e_i ||^2  (i.e. squared norm of columns of L_inv * Zt)
				vec_t L_inv_Zt_col_squaredNorm(num_data_);
#pragma omp parallel for schedule(static)
				for (data_size_t i = 0; i < num_data_; ++i) {
					vec_t L_inv_Zt_col_i = L_inv * (*Zt_).col(i);
					L_inv_Zt_col_squaredNorm[i] = L_inv_Zt_col_i.squaredNorm();
				}
				d_mll_d_mode = 0.5 * (Zt_deriv_information_loc_par * L_inv_Zt_col_squaredNorm);
			}
			// old equivalent code
//				if (grad_information_wrt_mode_non_zero_) {
//					deriv_information_diag_loc_par = vec_t(num_data_);
//					CalcFirstDerivInformationLocPar_PerSample(y_data, y_data_int, location_par.data(), deriv_information_diag_loc_par);
//					d_mll_d_mode = vec_t(dim_mode_);
//					sp_mat_t Zt_deriv_information_loc_par = (*Zt_) * deriv_information_diag_loc_par.asDiagonal();//every column of Z multiplied elementwise by deriv_information_diag_loc_par
//#pragma omp parallel for schedule(static)
//					for (int ire = 0; ire < dim_mode_; ++ire) {
//						//calculate Z^T * diag(diag_d_W_d_mode_i) * Z = Z^T * diag(Z.col(i) * deriv_information_diag_loc_par) * Z
//						d_mll_d_mode[ire] = 0.;
//						double entry_ij;
//						for (data_size_t i = 0; i < num_data_; ++i) {
//							entry_ij = Zt_deriv_information_loc_par.coeff(ire, i);
//							if (std::abs(entry_ij) > EPSILON_NUMBERS) {
//								vec_t L_inv_Zt_col_i = L_inv * (*Zt_).col(i);
//								d_mll_d_mode[ire] += entry_ij * (L_inv_Zt_col_i.squaredNorm());
//							}
//						}
//						d_mll_d_mode[ire] *= 0.5;
//					}
//				}
			// calculate gradient wrt covariance parameters
			bool some_cov_par_estimated = std::any_of(estimate_cov_par_index.begin(), estimate_cov_par_index.end(), [](int x) { return x > 0; });
			if (calc_cov_grad && some_cov_par_estimated) {
				sp_mat_t ZtWZ = (*Zt_) * information_ll_.asDiagonal() * (*Zt_).transpose();
				vec_t d_mode_d_par;//derivative of mode wrt to a covariance parameter
				vec_t v_aux;//auxiliary variable for caclulating d_mode_d_par
				vec_t SigmaI_mode;
				if (has_vecchia_gp) {
					SigmaI_mode = vec_t(dim_mode_);
					SigmaI_mode.segment(0, dim_re_group) = SigmaI * mode_.segment(0, dim_re_group);
					SigmaI_mode.segment(dim_re_group, dim_gp) = B.transpose() * (D_inv * (B * (mode_.segment(dim_re_group, dim_gp))));
				}
				else {
					SigmaI_mode = SigmaI * mode_;
				}
				sp_mat_t I_j(dim_mode_, dim_mode_);//Diagonal matrix with 1 on the diagonal for all random effects of component j and 0's otherwise
				sp_mat_t I_j_ZtWZ;
				for (int j = 0; j < num_grouped_RE; ++j) {
					if (estimate_cov_par_index[j] > 0) {
						// calculate explicit derivative of approx. mariginal log-likelihood
						std::vector<Triplet_t> triplets(cum_num_rand_eff_cluster_i[j + 1] - cum_num_rand_eff_cluster_i[j]);//for constructing I_j
						double explicit_derivative = 0.;
#pragma omp parallel for schedule(static) reduction(+:explicit_derivative)
						for (int i = cum_num_rand_eff_cluster_i[j]; i < cum_num_rand_eff_cluster_i[j + 1]; ++i) {
							triplets[i - cum_num_rand_eff_cluster_i[j]] = Triplet_t(i, i, 1.);
							explicit_derivative += SigmaI_mode[i] * mode_[i];
						}
						explicit_derivative *= -0.5;
						I_j.setFromTriplets(triplets.begin(), triplets.end());
						I_j_ZtWZ = I_j * ZtWZ;
						SigmaI_plus_ZtWZ_inv = I_j_ZtWZ;
						CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_ZtWZ_inv, false);
						explicit_derivative += 0.5 * (SigmaI_plus_ZtWZ_inv.cwiseProduct(I_j_ZtWZ)).sum();
						SigmaI_plus_ZtWZ_inv.resize(0, 0);
						cov_grad[j] = explicit_derivative;
						if (grad_information_wrt_mode_non_zero_) {
							// calculate implicit derivative (through mode) of approx. mariginal log-likelihood
							d_mode_d_par = L_inv.transpose() * (L_inv * (I_j * ((*Zt_) * first_deriv_ll_)));
							cov_grad[j] += d_mll_d_mode.dot(d_mode_d_par);
						}
					}
				}
				if (has_vecchia_gp) {
					CHECK(num_sets_re_ == 1);
					const int num_par_gp = (int)B_grad.size();
					CHECK((int)D_grad.size() == num_par_gp);						
					sp_mat_t SigmaI_plus_ZtWZ_inv_on_pattern;// (SigmaI_plus_ZtWZ)^-1 on the sparsity pattern of SigmaI_deriv (GP block), computing the full inverse is too expensive
					sp_mat_t D_inv_B = D_inv * B;// D_inv_B for GP block
					sp_mat_t SigmaI_deriv_gp; // dim_gp x dim_gp
					sp_mat_t SigmaI_deriv_full;// dim_mode_ x dim_mode_ (embedded gp block)
					sp_mat_t Bt_Dinv_Bgrad, BgradT_Dinv_B;
					for (int p = 0; p < num_par_gp; ++p) {
						const int cov_ind = num_grouped_RE + p;
						if (cov_ind >= (int)estimate_cov_par_index.size()) {
							Log::REFatal("estimate_cov_par_index too short for Vecchia GP params in CalcGradNegMargLikelihoodLaplaceApproxGroupedRE");
						}
						if (estimate_cov_par_index[cov_ind] <= 0) {
							continue;
						}
						// Build SigmaI_deriv_gp for GP precision: d(B^T D^{-1} B)/dtheta = B_grad^T D^{-1} B + B^T D^{-1} B_grad - (D^{-1}B)^T D_grad (D^{-1}B)
						if (p == 0) {
							SigmaI_deriv_gp = -B.transpose() * D_inv_B;  // Special case: if p==0 is the variance parameter: SigmaI_deriv_gp = -SigmaI_gp
						}
						else {
							BgradT_Dinv_B = B_grad[p].transpose() * D_inv_B;
							Bt_Dinv_Bgrad = BgradT_Dinv_B.transpose();
							SigmaI_deriv_gp = BgradT_Dinv_B + Bt_Dinv_Bgrad - D_inv_B.transpose() * D_grad[p] * D_inv_B;
							Bt_Dinv_Bgrad.resize(0, 0);
							BgradT_Dinv_B.resize(0, 0);
						}
						//  Embed into full precision derivative (block-diagonal): SigmaI_deriv_full has zeros on grouped RE block and SigmaI_deriv_gp on GP block
						GPBoost::MakeBlockDiag_I_B(SigmaI_deriv_gp, dim_re_group, SigmaI_deriv_full, true);
						const vec_t SigmaI_deriv_mode = SigmaI_deriv_full * mode_;
						// Compute inverse-on-pattern for trace term:
						SigmaI_plus_ZtWZ_inv_on_pattern = SigmaI_deriv_full;
						CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_ZtWZ_inv_on_pattern, false);
						double explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv_mode) +
							(SigmaI_deriv_full.cwiseProduct(SigmaI_plus_ZtWZ_inv_on_pattern)).sum());
						// Add derivative of +0.5*log|Sigma| term (equivalently -0.5*log|Sigma^{-1}|)
						if (p == 0) {								
							explicit_derivative += 0.5 * dim_gp;// variance parameter
						}
						else {
							explicit_derivative += 0.5 * (D_inv.diagonal().array() * D_grad[p].diagonal().array()).sum();
						}
						cov_grad[cov_ind] = explicit_derivative;
						// Implicit derivative via mode
						if (grad_information_wrt_mode_non_zero_) {
							vec_t rhs = SigmaI_deriv_full * ((*Zt_) * first_deriv_ll_);
							d_mode_d_par = L_inv.transpose() * (L_inv * rhs);
							cov_grad[cov_ind] += d_mll_d_mode.dot(d_mode_d_par);
						}
						SigmaI_deriv_gp.resize(0, 0);
						SigmaI_deriv_full.resize(0, 0);
						SigmaI_plus_ZtWZ_inv_on_pattern.resize(0, 0);
					} // end loop over GP parameters p
				} // end has_vecchia_gp
			}//end calc_cov_grad
			// calculate gradient wrt fixed effects
			if (calc_F_grad) {
				fixed_effect_grad = -first_deriv_ll_;
				if (grad_information_wrt_mode_non_zero_) {
					CHECK(first_deriv_information_loc_par_caluclated_);
					vec_t d_detmll_d_F(num_data_);
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_data_; ++i) {
						vec_t L_inv_Zt_col_i = L_inv * (*Zt_).col(i);
						d_detmll_d_F[i] = 0.5 * deriv_information_diag_loc_par[i] * (L_inv_Zt_col_i.squaredNorm());

					}
					vec_t d_mll_d_modeT_SigmaI_plus_ZtWZ_inv_Zt_W = (((d_mll_d_mode.transpose() * L_inv.transpose()) * L_inv) * (*Zt_)) * information_ll_.asDiagonal();
					fixed_effect_grad += d_detmll_d_F - d_mll_d_modeT_SigmaI_plus_ZtWZ_inv_Zt_W;
				}//end grad_information_wrt_mode_non_zero_
				if (HasSecondFEBlock()) {
					vec_t diag_data, Z_Ainv_d_mll_d_mode;
					if (SecondFEBlockGradNeedsDiag(true)) {
						// data-scale diagonal of (Sigma^-1 + Zt*W*Z)^-1, i.e. ||L_inv * Zt.col(i)||^2 (see the mean's d_detmll_d_F above)
						diag_data.resize(num_data_);
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							diag_data[i] = (L_inv * (*Zt_).col(i)).squaredNorm();
						}
					}
					if (SecondFEBlockGradNeedsImpl(true)) {
						Z_Ainv_d_mll_d_mode = (*Zt_).transpose() * (L_inv.transpose() * (L_inv * d_mll_d_mode));// = Z (Sigma^-1+ZtWZ)^-1 d_mll_d_mode (data scale)
					}
					CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par.data(), information_ll_, diag_data,
						Z_Ainv_d_mll_d_mode, nullptr, true, fixed_effect_grad);
				}
			}//end calc_F_grad
			// calculate gradient wrt additional likelihood parameters
			if (calc_aux_par_grad) {
				vec_t neg_likelihood_deriv(num_aux_pars_estim_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
				vec_t second_deriv_loc_aux_par(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
				vec_t deriv_information_aux_par(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
				vec_t d_mode_d_aux_par;
				CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par.data(), neg_likelihood_deriv.data());
				for (int ind_ap = 0; ind_ap < num_aux_pars_estim_; ++ind_ap) {
					CalcSecondDerivLogLikFirstDerivInformationAuxPar(y_data, y_data_int, location_par.data(), ind_ap, second_deriv_loc_aux_par.data(), deriv_information_aux_par.data());
					sp_mat_t ZtdWZ = (*Zt_) * deriv_information_aux_par.asDiagonal() * (*Zt_).transpose();
					SigmaI_plus_ZtWZ_inv = ZtdWZ;
					CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_ZtWZ_inv, false);
					double d_detmll_d_aux_par = (SigmaI_plus_ZtWZ_inv.cwiseProduct(ZtdWZ)).sum();
					aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par;
					if (grad_information_wrt_mode_non_zero_) {
						d_mode_d_aux_par = L_inv.transpose() * (L_inv * ((*Zt_) * second_deriv_loc_aux_par));
						aux_par_grad[ind_ap] += d_mll_d_mode.dot(d_mode_d_aux_par);
					}
				}
				SetGradAuxParsNotEstimated(aux_par_grad);
			}//end calc_aux_par_grad
		}//end Cholesky decomposition
	}//end CalcGradNegMargLikelihoodLaplaceApproxGroupedRE

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcGradNegMargLikelihoodLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const double sigma2,
		bool calc_cov_grad,
		bool calc_F_grad,
		bool calc_aux_par_grad,
		double* cov_grad,
		vec_t& fixed_effect_grad,
		double* aux_par_grad,
		bool calc_mode,
		bool call_for_std_dev_coef,
		const std::vector<int>& estimate_cov_par_index) {
		if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(y_data, y_data_int, fixed_effects, sigma2, mll);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			if (call_for_std_dev_coef) {
				Log::REFatal(CANNOT_CALC_STDEV_ERROR_);
			}
			else {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
		}
		CHECK(mode_has_been_calculated_);
		// Initialize variables
		vec_t location_par(dim_location_par_);//location parameter = mode of random effects + fixed effects (+ possibly additional fixed-effects-only blocks)
		double* location_par_ptr_dummy;//not used
		UpdateLocationParNewMode(mode_, fixed_effects, location_par, &location_par_ptr_dummy);
		// calculate gradient of approx. marginal likelihood wrt the mode
		vec_t deriv_information_diag_loc_par;//usually vector of negative third derivatives of log-likelihood
		vec_t d_mll_d_mode;
		if (grad_information_wrt_mode_non_zero_ && !iid_model_) {
			d_mll_d_mode = vec_t(dim_mode_);
			deriv_information_diag_loc_par = vec_t(num_data_);
			CalcFirstDerivInformationLocPar_PerSample(y_data, y_data_int, location_par.data(), deriv_information_diag_loc_par);
			CalcZtVGivenIndices(num_data_, dim_mode_, random_effects_indices_of_data_, deriv_information_diag_loc_par.data(), d_mll_d_mode.data(), true);
			d_mll_d_mode.array() /= 2. * diag_SigmaI_plus_ZtWZ_.array();
		}
		// calculate gradient wrt covariance parameters
		if (calc_cov_grad && estimate_cov_par_index[0]) {
			double explicit_derivative = -0.5 * (mode_.array() * mode_.array()).sum() / sigma2 +
				0.5 * (information_ll_.array() / diag_SigmaI_plus_ZtWZ_.array()).sum();
			cov_grad[0] = explicit_derivative;
			if (grad_information_wrt_mode_non_zero_ && !iid_model_) {
				CHECK(first_deriv_information_loc_par_caluclated_);
				// calculate implicit derivative (through mode) of approx. mariginal log-likelihood
				vec_t d_mode_d_par = first_deriv_ll_;
				d_mode_d_par.array() /= diag_SigmaI_plus_ZtWZ_.array();
				cov_grad[0] += d_mll_d_mode.dot(d_mode_d_par);
			}
		}//end calc_cov_grad
		// calculate gradient wrt fixed effects
		if (calc_F_grad) {
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data_; ++i) {
				//fixed_effect_grad[i] = -first_deriv_ll_[i];
				fixed_effect_grad[i] = -first_deriv_ll_data_scale_[i];
				if (grad_information_wrt_mode_non_zero_ && !iid_model_) {
					fixed_effect_grad[i] += 0.5 * deriv_information_diag_loc_par[i] / diag_SigmaI_plus_ZtWZ_[random_effects_indices_of_data_[i]] - //=d_detmll_d_F
						d_mll_d_mode[random_effects_indices_of_data_[i]] * information_ll_data_scale_[i] / diag_SigmaI_plus_ZtWZ_[random_effects_indices_of_data_[i]];//=implicit derivative = d_mll_d_mode * d_mode_d_F
				}
			}
			if (HasSecondFEBlock()) {
				// NOTE: unlike the other approximations, this one stores the DIAGONAL OF (Sigma^-1+ZtWZ) ITSELF rather than
				// of its inverse, so the reciprocal has to be formed here to match the convention of the gradient function
				vec_t zeta_diag, zeta_impl;
				if (SecondFEBlockGradNeedsDiag(true)) {
					zeta_diag = diag_SigmaI_plus_ZtWZ_.cwiseInverse();
					if (SecondFEBlockGradNeedsImpl(true)) {
						zeta_impl = d_mll_d_mode.cwiseProduct(zeta_diag);
					}
				}
				CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par.data(), information_ll_data_scale_,
					zeta_diag, zeta_impl, random_effects_indices_of_data_, true, fixed_effect_grad);
			}
		}//end calc_F_grad
		// calculate gradient wrt additional likelihood parameters
		if (calc_aux_par_grad) {
			vec_t neg_likelihood_deriv(num_aux_pars_estim_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
			vec_t second_deriv_loc_aux_par(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
			vec_t deriv_information_aux_par(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
			CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par.data(), neg_likelihood_deriv.data());
			for (int ind_ap = 0; ind_ap < num_aux_pars_estim_; ++ind_ap) {
				CalcSecondDerivLogLikFirstDerivInformationAuxPar(y_data, y_data_int, location_par.data(), ind_ap, second_deriv_loc_aux_par.data(), deriv_information_aux_par.data());
				double d_detmll_d_aux_par = 0., implicit_derivative = 0.;// = implicit derivative = d_mll_d_mode * d_mode_d_aux_par
				if (!iid_model_) {
#pragma omp parallel for schedule(static) reduction(+:d_detmll_d_aux_par, implicit_derivative)
					for (int i = 0; i < num_data_; ++i) {
						d_detmll_d_aux_par += deriv_information_aux_par[i] / diag_SigmaI_plus_ZtWZ_[random_effects_indices_of_data_[i]];
						if (grad_information_wrt_mode_non_zero_) {
							implicit_derivative += d_mll_d_mode[random_effects_indices_of_data_[i]] * second_deriv_loc_aux_par[i] / diag_SigmaI_plus_ZtWZ_[random_effects_indices_of_data_[i]];
						}
					}
				}
				aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
				//Equivalent code:
				//vec_t Zt_second_deriv_loc_aux_par, diag_Zt_deriv_information_loc_par_Z;
				//CalcZtVGivenIndices(num_data_, dim_mode_, random_effects_indices_of_data_, second_deriv_loc_aux_par, Zt_second_deriv_loc_aux_par, true);
				//CalcZtVGivenIndices(num_data_, dim_mode_, random_effects_indices_of_data_, deriv_information_aux_par, diag_Zt_deriv_information_loc_par_Z, true);
				//double d_detmll_d_aux_par = (diag_Zt_deriv_information_loc_par_Z.array() / diag_SigmaI_plus_ZtWZ_.array()).sum();
				//double implicit_derivative = (d_mll_d_mode.array() * Zt_second_deriv_loc_aux_par.array() / diag_SigmaI_plus_ZtWZ_.array()).sum();
			}
			SetGradAuxParsNotEstimated(aux_par_grad);
		}//end calc_aux_par_grad
	}//end CalcGradNegMargLikelihoodLaplaceApproxOnlyOneGroupedRECalculationsOnREScale

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcGradNegMargLikelihoodLaplaceApproxFSVA(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_woodbury,
		const den_mat_t& chol_ip_cross_cov,
		const den_mat_t& sigma_woodbury,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const sp_mat_t& B,
		const sp_mat_t& D_inv,
		const den_mat_t& Bt_D_inv_B_cross_cov,
		const den_mat_t& D_inv_B_cross_cov,
		const den_mat_t& sigma_ip_inv_cross_cov_T_cluster_i,
		const std::vector<sp_mat_t>& B_grad,
		const std::vector<sp_mat_t>& D_grad,
		bool calc_cov_grad,
		bool calc_F_grad,
		bool calc_aux_par_grad,
		double* cov_grad,
		vec_t& fixed_effect_grad,
		double* aux_par_grad,
		bool calc_mode,
		bool call_for_std_dev_coef,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_preconditioner_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i,
		const den_mat_t& chol_ip_cross_cov_preconditioner,
		const chol_den_mat_t& chol_fact_sigma_ip_preconditioner,
		const std::vector<int>& estimate_cov_par_index,
		bool GPU_use) {
		const den_mat_t* cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
		den_mat_t sigma_ip = *(re_comps_ip_cluster_i[0]->GetZSigmaZt());
		int num_ip = (int)(sigma_ip.rows());
		CHECK((int)((*cross_cov).rows()) == dim_mode_);
		CHECK((int)((*cross_cov).cols()) == num_ip);
		if (calc_mode) {// Calculate mode and Cholesky factor 
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLFSVA(y_data, y_data_int, fixed_effects, sigma_ip, chol_fact_sigma_ip,
				chol_fact_sigma_woodbury, chol_ip_cross_cov, re_comps_cross_cov_cluster_i, sigma_woodbury, B, D_inv, Bt_D_inv_B_cross_cov, D_inv_B_cross_cov, false, true, mll,
				re_comps_ip_preconditioner_cluster_i, re_comps_cross_cov_preconditioner_cluster_i, chol_ip_cross_cov_preconditioner,
				chol_fact_sigma_ip_preconditioner, GPU_use);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			if (call_for_std_dev_coef) {
				Log::REFatal(CANNOT_CALC_STDEV_ERROR_);
			}
			else {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
		}
		den_mat_t sigma_ip_stable = sigma_ip;
		sigma_ip_stable.diagonal().array() *= JITTER_MULT_IP_FITC_FSA;
		CHECK(mode_has_been_calculated_);
		// Initialize variables
		vec_t location_par;//location parameter = mode of random effects + fixed effects
		double* location_par_ptr;
		InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
		vec_t deriv_information_diag_loc_par;//first derivative of the diagonal of the Fisher information wrt the location parameter (= usually negative third derivatives of the log-likelihood wrt the locatin parameter)
		vec_t deriv_information_diag_loc_par_data_scale;//first derivative of the diagonal of the Fisher information wrt the location parameter on the data-scale (only used if use_random_effects_indices_of_data_), the vector 'deriv_information_diag_loc_par' actually contains diag_ZtDerivInformationZ if use_random_effects_indices_of_data_
		if (grad_information_wrt_mode_non_zero_) {
			CalcFirstDerivInformationLocPar(y_data, y_data_int, location_par_ptr, deriv_information_diag_loc_par, deriv_information_diag_loc_par_data_scale);
		}
		vec_t W_D_inv = information_ll_ + D_inv_rm_.diagonal();
		vec_t W_D_inv_inv = W_D_inv.cwiseInverse();
		vec_t d_mll_d_mode;
		if (HasZeroValueInformationLogLik()) {
			Log::REFatal("CalcGradNegMargLikelihoodLaplaceApproxFSVA: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
				"This is not permitted when using the VIF approximation and gradient-based optimization ");
		}
		if (matrix_inversion_method_ == "iterative") {
			if ((cg_preconditioner_type_ == "vifdu" || cg_preconditioner_type_ == "fitc") && HasNegativeValueInformationLogLik()) {
				Log::REFatal("CalcGradNegMargLikelihoodLaplaceApproxFSVA: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
					"The stochastic gradient calculation with the '%s' preconditioner requires W to be nonnegative ", cg_preconditioner_type_.c_str());
			}
			double c_opt;
			sp_mat_rm_t SigmaI_rm = B_t_D_inv_rm_ * B_rm_;
			vec_t SigmaI_deriv_mode;
			vec_t d_log_det_Sigma_W_plus_I_d_mode, SigmaI_plus_W_inv_d_mll_d_mode(dim_mode_);
			den_mat_t W_deriv_rep;
			vec_t tr_SigmaI_plus_W_inv_W_deriv, tr_PI_P_deriv_vec(dim_mode_), c_opt_vec;
			den_mat_t Z_SigmaI_plus_W_inv_W_deriv_PI_Z, PI_Z(dim_mode_, num_rand_vec_trace_),
				Z_PI_P_deriv_PI_Z;
			if (grad_information_wrt_mode_non_zero_) {
				W_deriv_rep = deriv_information_diag_loc_par.replicate(1, num_rand_vec_trace_);
			}
			vec_t diag_WI = information_ll_.cwiseInverse();
			bool some_cov_par_estimated = std::any_of(estimate_cov_par_index.begin(), estimate_cov_par_index.end(), [](int x) { return x > 0; });
			if (cg_preconditioner_type_ == "fitc") {
				const den_mat_t* cross_cov_preconditioner = re_comps_cross_cov_preconditioner_cluster_i[0]->GetSigmaPtr();
				den_mat_t cross_cov_preconditioner_t = (*cross_cov_preconditioner).transpose();
				// P^-1 rand_vec
				den_mat_t WI_SigmaI_plus_W_inv_Z = diag_WI.asDiagonal() * SigmaI_plus_W_inv_Z_;
				den_mat_t P_diag_inv_rand_vect = diagonal_approx_inv_preconditioner_.asDiagonal() * rand_vec_trace_I_;
				PI_Z = P_diag_inv_rand_vect - diagonal_approx_inv_preconditioner_.asDiagonal() * (*cross_cov_preconditioner) * chol_fact_woodbury_preconditioner_.solve(cross_cov_preconditioner_t * P_diag_inv_rand_vect);
				den_mat_t WI_PI_Z = diag_WI.asDiagonal() * PI_Z;
				if (grad_information_wrt_mode_non_zero_) {
					Z_SigmaI_plus_W_inv_W_deriv_PI_Z = -1 * (WI_SigmaI_plus_W_inv_Z.array() * W_deriv_rep.array() * WI_PI_Z.array()).matrix();
					tr_SigmaI_plus_W_inv_W_deriv = Z_SigmaI_plus_W_inv_W_deriv_PI_Z.rowwise().mean();

					vec_t tr_WI_W_deriv = diag_WI.cwiseProduct(deriv_information_diag_loc_par);
					d_log_det_Sigma_W_plus_I_d_mode = tr_SigmaI_plus_W_inv_W_deriv + tr_WI_W_deriv;
					//variance reduction
					//-tr(W^-1P^-1W^(-1) dW/db_i)
					vec_t tr_WI_DI_WI_W_deriv = diag_WI.cwiseProduct(tr_WI_W_deriv.cwiseProduct(diagonal_approx_inv_preconditioner_));
					vec_t tr_WI_DI_WI_DI_W_deriv = diagonal_approx_inv_preconditioner_.cwiseProduct(tr_WI_DI_WI_W_deriv);
					den_mat_t chol_wood_cross_cov((*cross_cov_preconditioner).cols(), dim_mode_);
					//TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_woodbury_preconditioner_, cross_cov_preconditioner_t, chol_wood_cross_cov, false);
					GPBoost::solve_lower_triangular(chol_fact_woodbury_preconditioner_, cross_cov_preconditioner_t, chol_wood_cross_cov, GPU_use);
#pragma omp parallel for schedule(static)  
					for (int i = 0; i < dim_mode_; ++i) {
						tr_PI_P_deriv_vec[i] = chol_wood_cross_cov.col(i).array().square().sum() * tr_WI_DI_WI_DI_W_deriv[i];
					}
					tr_PI_P_deriv_vec -= tr_WI_DI_WI_W_deriv;
					//stochastic tr(P^(-1) dP/db_i), where dP/db_i = - W^(-1) dW/db_i W^(-1)
					Z_PI_P_deriv_PI_Z = -1 * (WI_PI_Z.array() * W_deriv_rep.array() * WI_PI_Z.array()).matrix();
					vec_t tr_PI_inv_W_deriv = Z_PI_P_deriv_PI_Z.rowwise().mean();
					//optimal c
					CalcOptimalCVectorized(Z_SigmaI_plus_W_inv_W_deriv_PI_Z, Z_PI_P_deriv_PI_Z, tr_SigmaI_plus_W_inv_W_deriv, tr_PI_P_deriv_vec, c_opt_vec);
					d_log_det_Sigma_W_plus_I_d_mode += c_opt_vec.cwiseProduct(tr_PI_P_deriv_vec - tr_PI_inv_W_deriv);
				}
				//For implicit derivatives: calculate (Sigma^(-1) + W)^(-1) d_mll_d_mode
				bool has_NA_or_Inf = false;
				if (grad_information_wrt_mode_non_zero_) {
					d_mll_d_mode = 0.5 * d_log_det_Sigma_W_plus_I_d_mode;
					vec_t Sigma_d_mll_d_mode = D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve((B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(d_mll_d_mode)) +
						(*cross_cov) * (chol_fact_sigma_ip.solve((*cross_cov).transpose() * d_mll_d_mode));
					vec_t W_SigmaI_plus_W_inv_d_mll_d_mode(dim_mode_);
					CGVIFLaplace_Version_SigmaPlusWinvVec(information_ll_.cwiseInverse(), D_inv_B_rm_, B_rm_, chol_fact_woodbury_preconditioner_,
						chol_ip_cross_cov, cross_cov_preconditioner, diagonal_approx_inv_preconditioner_, Sigma_d_mll_d_mode, W_SigmaI_plus_W_inv_d_mll_d_mode, has_NA_or_Inf,
						cg_max_num_it_, true, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, false);
					SigmaI_plus_W_inv_d_mll_d_mode = information_ll_.cwiseInverse().asDiagonal() * W_SigmaI_plus_W_inv_d_mll_d_mode;
					if (has_NA_or_Inf) {
						Log::REDebug(CG_NA_OR_INF_WARNING_GRADIENT_);
					}
				}
				// Calculate gradient wrt covariance parameters
				if (calc_cov_grad && some_cov_par_estimated) {
					sp_mat_rm_t SigmaI_deriv_rm, Bt_Dinv_Bgrad_rm, B_t_D_inv_D_grad_D_inv_B_rm;
					double explicit_derivative, d_log_det_Sigma_W_plus_I_d_cov_pars;
					int num_par = (int)B_grad.size();
					//den_mat_t sigma_ip_inv_sigma_cross_cov_preconditioner = chol_fact_sigma_ip_preconditioner.solve((*cross_cov_preconditioner).transpose());
					den_mat_t sigma_ip_inv_sigma_cross_cov_preconditioner;
					GPBoost::solve_linear_sys(chol_fact_sigma_ip_preconditioner, cross_cov_preconditioner_t, sigma_ip_inv_sigma_cross_cov_preconditioner, GPU_use);
					den_mat_t sigma_ip_inv_cross_cov_preconditioner_PI_Z = sigma_ip_inv_sigma_cross_cov_preconditioner * PI_Z;
					den_mat_t sigma_ip_inv_cross_cov_PI_Z = sigma_ip_inv_cross_cov_T_cluster_i * PI_Z;
					CHECK(re_comps_ip_cluster_i.size() == 1);
					for (int j = 0; j < (int)re_comps_ip_cluster_i.size(); ++j) {
						for (int ipar = 0; ipar < num_par; ++ipar) {
							if (estimate_cov_par_index[ipar] > 0) {
								std::shared_ptr<den_mat_t> cross_cov_grad = re_comps_cross_cov_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.);
								den_mat_t sigma_ip_grad = *(re_comps_ip_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.));
								if (ipar == 0) {
									SigmaI_deriv_rm = -B_rm_.transpose() * B_t_D_inv_rm_.transpose();//SigmaI_deriv = -SigmaI for variance parameters if there is only one GP
								}
								else {
									//SigmaI_deriv_rm = sp_mat_rm_t(B_grad[ipar].transpose()) * B_t_D_inv_rm_.transpose();
									sp_mat_rm_t B_t_D_inv_rm_t = sp_mat_rm_t(B_t_D_inv_rm_.transpose());
									GPBoost::spmatmul(sp_mat_rm_t(B_grad[ipar].transpose()), B_t_D_inv_rm_t, SigmaI_deriv_rm, GPU_use);
									Bt_Dinv_Bgrad_rm = SigmaI_deriv_rm.transpose();
									//B_t_D_inv_D_grad_D_inv_B_rm = B_t_D_inv_rm_ * sp_mat_rm_t(D_grad[ipar]) * B_t_D_inv_rm_.transpose();
									sp_mat_rm_t B_t_D_inv_D_grad_D_inv_B_rm_inter;
									GPBoost::spmatmul(sp_mat_rm_t(D_grad[ipar]), B_t_D_inv_rm_t, B_t_D_inv_D_grad_D_inv_B_rm_inter, GPU_use);
									GPBoost::spmatmul(B_t_D_inv_rm_, B_t_D_inv_D_grad_D_inv_B_rm_inter, B_t_D_inv_D_grad_D_inv_B_rm, GPU_use);
									SigmaI_deriv_rm += Bt_Dinv_Bgrad_rm - B_t_D_inv_D_grad_D_inv_B_rm;
									Bt_Dinv_Bgrad_rm.resize(0, 0);
								}
								// Derivative of Woodbury matrix
								den_mat_t sigma_woodbury_grad = sigma_ip_grad;
								den_mat_t SigmaI_deriv_rm_cross_cov(dim_mode_, num_ip);
#pragma omp parallel for schedule(static)  
								for (int ii = 0; ii < num_ip; ii++) {
									SigmaI_deriv_rm_cross_cov.col(ii) = SigmaI_deriv_rm * (*cross_cov).col(ii);
								}
								den_mat_t cross_cov_dot;
								GPBoost::matmul((*cross_cov).transpose(), SigmaI_deriv_rm_cross_cov, cross_cov_dot, GPU_use);
								sigma_woodbury_grad += cross_cov_dot;
								//den_mat_t cross_cov_Bt_D_inv_B_cross_cov_grad = Bt_D_inv_B_cross_cov.transpose() * (*cross_cov_grad);
								den_mat_t cross_cov_Bt_D_inv_B_cross_cov_grad;
								GPBoost::matmul(Bt_D_inv_B_cross_cov.transpose(), (*cross_cov_grad), cross_cov_Bt_D_inv_B_cross_cov_grad, GPU_use);
								sigma_woodbury_grad += cross_cov_Bt_D_inv_B_cross_cov_grad + cross_cov_Bt_D_inv_B_cross_cov_grad.transpose();

								vec_t SigmaI_deriv_mode_part = SigmaI_deriv_rm * mode_;
								vec_t SigmaI_mode = SigmaI_rm * mode_;
								SigmaI_deriv_mode = SigmaI_deriv_mode_part - SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_deriv_mode_part)) -
									SigmaI_deriv_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)) -
									SigmaI_rm * ((*cross_cov_grad) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)) -
									SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov_grad).transpose() * SigmaI_mode)) +
									SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve(sigma_woodbury_grad * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)));
								explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv_mode));
								den_mat_t PP_deriv_sample_vec = (*cross_cov_grad) * sigma_ip_inv_cross_cov_PI_Z + sigma_ip_inv_cross_cov_T_cluster_i.transpose() * ((*cross_cov_grad).transpose() * PI_Z) -
									sigma_ip_inv_cross_cov_T_cluster_i.transpose() * (sigma_ip_grad * sigma_ip_inv_cross_cov_PI_Z);

								den_mat_t SigmaI_deriv_sample_vec = PP_deriv_sample_vec;
#pragma omp parallel for schedule(static)  
								for (int ii = 0; ii < num_rand_vec_trace_; ii++) {
									SigmaI_deriv_sample_vec.col(ii) -= D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve((B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(SigmaI_deriv_rm * D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve((B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(PI_Z.col(ii)))));
								}
								vec_t sample_Sigma = (SigmaI_plus_W_inv_Z_.cwiseProduct(SigmaI_deriv_sample_vec)).colwise().sum();
								double stoch_tr = sample_Sigma.mean();
								d_log_det_Sigma_W_plus_I_d_cov_pars = stoch_tr;

								std::shared_ptr<den_mat_t>  cross_cov_preconditioner_grad = re_comps_cross_cov_preconditioner_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.);
								den_mat_t sigma_ip_preconditioner_grad = *(re_comps_ip_preconditioner_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.));
								// Variance reduction
								vec_t D_grad_diagonal = D_grad[ipar].diagonal();
								vec_t D_diagonal = D_inv_rm_.diagonal().cwiseInverse();
								den_mat_t P_grad_PI_Z = (*cross_cov_preconditioner_grad) * sigma_ip_inv_cross_cov_preconditioner_PI_Z +
									sigma_ip_inv_sigma_cross_cov_preconditioner.transpose() * ((*cross_cov_preconditioner_grad).transpose() * PI_Z) -
									sigma_ip_inv_sigma_cross_cov_preconditioner.transpose() * (sigma_ip_preconditioner_grad * sigma_ip_inv_cross_cov_preconditioner_PI_Z);
								vec_t diagonal_approx_preconditioner_grad_ = vec_t::Zero(dim_mode_);
								diagonal_approx_preconditioner_grad_.array() += sigma_ip_preconditioner_grad.coeffRef(0, 0);
								//den_mat_t sigma_ip_grad_inv_sigma_cross_cov_preconditioner = sigma_ip_preconditioner_grad * sigma_ip_inv_sigma_cross_cov_preconditioner;
								den_mat_t sigma_ip_grad_inv_sigma_cross_cov_preconditioner;
								GPBoost::matmul(sigma_ip_preconditioner_grad, sigma_ip_inv_sigma_cross_cov_preconditioner, sigma_ip_grad_inv_sigma_cross_cov_preconditioner, GPU_use);
#pragma omp parallel for schedule(static)
								for (int ii = 0; ii < dim_mode_; ++ii) {
									diagonal_approx_preconditioner_grad_[ii] -= 2 * sigma_ip_inv_sigma_cross_cov_preconditioner.col(ii).dot((*cross_cov_preconditioner_grad).row(ii))
										- sigma_ip_inv_sigma_cross_cov_preconditioner.col(ii).dot(sigma_ip_grad_inv_sigma_cross_cov_preconditioner.col(ii));
								}
								P_grad_PI_Z += diagonal_approx_preconditioner_grad_.asDiagonal() * PI_Z;
								double tr_PI_P_grad = (diagonal_approx_preconditioner_grad_.array() * diagonal_approx_inv_preconditioner_.array()).sum();
								tr_PI_P_grad -= (chol_fact_sigma_ip_preconditioner.solve(sigma_ip_preconditioner_grad)).trace();
								// Derivative of Woodbury matrix
								den_mat_t sigma_woodbury_grad_preconditioner = sigma_ip_preconditioner_grad;
								den_mat_t D_inv_cross_cov = diagonal_approx_inv_preconditioner_.asDiagonal() * (*cross_cov_preconditioner);
								//den_mat_t cross_cov_grad_D_inv_cross_cov = (*cross_cov_preconditioner_grad).transpose() * D_inv_cross_cov;
								den_mat_t cross_cov_grad_D_inv_cross_cov;
								GPBoost::matmul((*cross_cov_preconditioner_grad).transpose(), D_inv_cross_cov, cross_cov_grad_D_inv_cross_cov, GPU_use);
								sigma_woodbury_grad_preconditioner += cross_cov_grad_D_inv_cross_cov + cross_cov_grad_D_inv_cross_cov.transpose();
								den_mat_t cross_cov_D_inv_cross_cov;
								GPBoost::matmul(D_inv_cross_cov.transpose(), (diagonal_approx_preconditioner_grad_.asDiagonal() * D_inv_cross_cov), cross_cov_D_inv_cross_cov, GPU_use);
								sigma_woodbury_grad_preconditioner -= cross_cov_D_inv_cross_cov;
								tr_PI_P_grad += (chol_fact_woodbury_preconditioner_.solve(sigma_woodbury_grad_preconditioner)).trace();
								vec_t sample_P = (PI_Z.cwiseProduct(P_grad_PI_Z)).colwise().sum();
								CalcOptimalC(sample_Sigma, sample_P, stoch_tr, tr_PI_P_grad, c_opt);
								d_log_det_Sigma_W_plus_I_d_cov_pars -= c_opt * (sample_P.mean() - tr_PI_P_grad);

								//Log::REInfo("tr final %g", d_log_det_Sigma_W_plus_I_d_cov_pars);
								explicit_derivative += 0.5 * d_log_det_Sigma_W_plus_I_d_cov_pars;
								//Log::REInfo("explicit_derivative %g", explicit_derivative);
								cov_grad[ipar] = explicit_derivative;
								if (grad_information_wrt_mode_non_zero_) {
									cov_grad[ipar] -= SigmaI_plus_W_inv_d_mll_d_mode.dot(SigmaI_deriv_mode);
								}
								//Log::REInfo("SigmaI_plus_W_inv_d_mll_d_mode %g", SigmaI_plus_W_inv_d_mll_d_mode.dot(SigmaI_deriv_mode));
							}//end estimate_cov_par_index[ipar] > 0
						}//end loop ipar
					}//end loop j
				}//end calc_cov_grad
				//Calculate gradient wrt fixed effects
				vec_t SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_diag_2nd_block;
				if (grad_information_wrt_mode_non_zero_ && ((use_random_effects_indices_of_data_ && calc_F_grad) || calc_aux_par_grad)) {
					//Stochastic Trace: Calculate diagonal of SigmaI_plus_W_inv for gradient of approx. marginal likelihood wrt. F
					SigmaI_plus_W_inv_diag = d_log_det_Sigma_W_plus_I_d_mode;
					SigmaI_plus_W_inv_diag.array() /= deriv_information_diag_loc_par.array();
					if (grad_information_wrt_mode_can_be_zero_for_some_points_) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)SigmaI_plus_W_inv_diag.size(); ++i) {
							if (GPBoost::IsZero<double>(deriv_information_diag_loc_par[i])) {
								SigmaI_plus_W_inv_diag[i] = 0.;//set to 0 for safety, but this is actually not needed
							}
						}
					}//end grad_information_wrt_mode_can_be_zero_for_some_points_
				}
				if (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad) {
					// Stochastic (Hutchinson) estimate of diag((Sigma^-1+W)^-1), needed for the second, fixed-effects-only
					// block's gradient below. The ratio trick used just above is not applicable here: for
					// 'gaussian_heteroscedastic' deriv_information_diag_loc_par is identically zero, and for
					// 'zero_censored_power_transformed_normal_heteroscedastic' it vanishes at every positive observation.
					// Note: rand_vec_trace_I_ is Cov = P (the 'fitc' preconditioner) here, not Cov = I (see
					// FindModePostRandEffCalcMLLFSVA); the raw (Cov = I) vectors are rand_vec_trace_I2_. Solve
					// (Sigma^-1+W) x_k = r_k for each column r_k via the push-through identity
					// (Sigma^-1+W)^-1 = W^-1 (Sigma+W^-1)^-1 Sigma (same recipe as the implicit-derivative solve above),
					// then diag ~= mean_k(x_k * r_k)
					CHECK(num_sets_re_ == 1);
					den_mat_t SigmaI_plus_W_inv_RV_I(dim_mode_, num_rand_vec_trace_);
					bool has_NA_or_Inf_stoch_diag = false;
					for (int k = 0; k < num_rand_vec_trace_; ++k) {
						vec_t Sigma_rv = D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve((B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(rand_vec_trace_I2_.col(k))) +
							(*cross_cov) * (chol_fact_sigma_ip.solve((*cross_cov).transpose() * rand_vec_trace_I2_.col(k)));
						vec_t W_result(dim_mode_);
						bool has_NA_or_Inf_k = false;
						CGVIFLaplace_Version_SigmaPlusWinvVec(information_ll_.cwiseInverse(), D_inv_B_rm_, B_rm_, chol_fact_woodbury_preconditioner_,
							chol_ip_cross_cov, cross_cov_preconditioner, diagonal_approx_inv_preconditioner_, Sigma_rv, W_result, has_NA_or_Inf_k,
							cg_max_num_it_, true, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, false);
						SigmaI_plus_W_inv_RV_I.col(k) = information_ll_.cwiseInverse().cwiseProduct(W_result);
						if (has_NA_or_Inf_k) {
							has_NA_or_Inf_stoch_diag = true;
						}
					}
					if (has_NA_or_Inf_stoch_diag) {
						Log::REDebug(CG_NA_OR_INF_WARNING_GRADIENT_);
					}
					SigmaI_plus_W_inv_diag_2nd_block = (SigmaI_plus_W_inv_RV_I.cwiseProduct(rand_vec_trace_I2_)).rowwise().mean();
					if (likelihood_type_ == "gaussian_heteroscedastic") {
						SigmaI_plus_W_inv_diag = SigmaI_plus_W_inv_diag_2nd_block;// there is no eta-block version for this likelihood
					}
				}
				if (calc_F_grad) {
					if (use_random_effects_indices_of_data_) {
						fixed_effect_grad = -first_deriv_ll_data_scale_;
						if (grad_information_wrt_mode_non_zero_) {
#pragma omp parallel for schedule(static)
							for (data_size_t i = 0; i < num_data_; ++i) {
								fixed_effect_grad[i] += 0.5 * deriv_information_diag_loc_par_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
									information_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
							}
						}
						if (HasSecondFEBlock()) {
							// 'include_coupled_zi_terms' is false: for a zero-inflated count regression the coupled log-determinant
							// term would need the data-scale diagonal of (Sigma^-1+W)^-1, which is only a stochastic estimate here,
							// so those terms are omitted -> the alpha gradient is approximate for ZI counts on this approximation.
							// 'SigmaI_plus_W_inv_diag_2nd_block' is the stochastic diagonal estimated above; for
							// 'gaussian_heteroscedastic' it is the same vector as 'SigmaI_plus_W_inv_diag'
							CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_data_scale_,
								SigmaI_plus_W_inv_diag_2nd_block, SigmaI_plus_W_inv_d_mll_d_mode, random_effects_indices_of_data_, false, fixed_effect_grad);
						}
					}
					else {
						fixed_effect_grad = -first_deriv_ll_;
						if (grad_information_wrt_mode_non_zero_) {
							vec_t d_mll_d_F_implicit = -(SigmaI_plus_W_inv_d_mll_d_mode.array() * information_ll_.array()).matrix();// implicit derivative
							fixed_effect_grad += d_mll_d_mode + d_mll_d_F_implicit;
						}
						if (HasSecondFEBlock()) {
							CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_,
								SigmaI_plus_W_inv_diag_2nd_block, SigmaI_plus_W_inv_d_mll_d_mode, nullptr, false, fixed_effect_grad);
						}
					}
				}
				//Calculate gradient wrt additional likelihood parameters
				if (calc_aux_par_grad) {
					vec_t neg_likelihood_deriv(num_aux_pars_estim_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
					vec_t second_deriv_loc_aux_par(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
					vec_t deriv_information_aux_par(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
					vec_t d_mode_d_aux_par;
					CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par_ptr, neg_likelihood_deriv.data());
					den_mat_t Preconditioner_PP_inv;
					//TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_woodbury_preconditioner_,
					//	(diagonal_approx_inv_preconditioner_.asDiagonal() * (*cross_cov_preconditioner)).transpose(), Preconditioner_PP_inv, false);
					GPBoost::solve_lower_triangular(chol_fact_woodbury_preconditioner_,
						(diagonal_approx_inv_preconditioner_.asDiagonal() * (*cross_cov_preconditioner)).transpose(), Preconditioner_PP_inv, GPU_use);
					if (grad_information_wrt_mode_non_zero_) {
						tr_PI_P_deriv_vec = -diagonal_approx_inv_preconditioner_.cwiseProduct(W_deriv_rep.col(0));
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < dim_mode_; ++i) {
							tr_PI_P_deriv_vec[i] += Preconditioner_PP_inv.col(i).array().square().sum() * W_deriv_rep.col(0)[i];
						}
					}
					for (int ind_ap = 0; ind_ap < num_aux_pars_; ++ind_ap) {
						CalcSecondDerivLogLikFirstDerivInformationAuxPar(y_data, y_data_int, location_par_ptr, ind_ap, second_deriv_loc_aux_par.data(), deriv_information_aux_par.data());
						double d_detmll_d_aux_par = 0., implicit_derivative = 0.;
						if (grad_information_wrt_mode_non_zero_) {
							bool deriv_information_loc_par_has_zero = false;
							if (grad_information_wrt_mode_can_be_zero_for_some_points_) {
								deriv_information_loc_par_has_zero = GPBoost::HasZero<double>(deriv_information_diag_loc_par.data(), (data_size_t)deriv_information_diag_loc_par.size());
							}
							if (deriv_information_loc_par_has_zero) {//deriv_information_diag_loc_par has some zeros
								if (use_random_effects_indices_of_data_) {
									vec_t Zt_deriv_information_aux_par(dim_mode_);
									CalcZtVGivenIndices(num_data_, dim_mode_, random_effects_indices_of_data_, deriv_information_aux_par.data(), Zt_deriv_information_aux_par.data(), true);
									//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
									vec_t W_inv_d_W_W_inv = -1. * Zt_deriv_information_aux_par.cwiseProduct((information_ll_.cwiseInverse().cwiseProduct(information_ll_.cwiseInverse())));
									vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(W_inv_d_W_W_inv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
									double tr_SigmaI_plus_W_inv_W_deriv_d = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
									d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv_d;
									//variance reduction
									//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
									double tr_D_inv_plus_W_inv_W_deriv = (diagonal_approx_inv_preconditioner_.array() * W_inv_d_W_W_inv.array()).sum();
									for (int ii = 0; ii < dim_mode_; ii++) {
										tr_D_inv_plus_W_inv_W_deriv -= Preconditioner_PP_inv.col(ii).array().square().sum() * W_inv_d_W_W_inv[ii];
									}
									vec_t zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(W_inv_d_W_W_inv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
									double tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
									//optimal 
									CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv_d, tr_D_inv_plus_W_inv_W_deriv, c_opt);
									d_detmll_d_aux_par -= c_opt * (tr_PI_P_deriv - tr_D_inv_plus_W_inv_W_deriv);
									d_detmll_d_aux_par += (Zt_deriv_information_aux_par.array() * information_ll_.cwiseInverse().array()).sum();
								}
								else {
									//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
									vec_t W_inv_d_W_W_inv = -1. * deriv_information_aux_par.cwiseProduct((information_ll_.cwiseInverse().cwiseProduct(information_ll_.cwiseInverse())));
									vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(W_inv_d_W_W_inv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
									double tr_SigmaI_plus_W_inv_W_deriv_d = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
									d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv_d;
									//variance reduction
									//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
									double tr_D_inv_plus_W_inv_W_deriv = (diagonal_approx_inv_preconditioner_.array() * W_inv_d_W_W_inv.array()).sum();
									for (int ii = 0; ii < dim_mode_; ii++) {
										tr_D_inv_plus_W_inv_W_deriv -= Preconditioner_PP_inv.col(ii).array().square().sum() * W_inv_d_W_W_inv[ii];
									}
									vec_t zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(W_inv_d_W_W_inv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
									double tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
									//optimal 
									CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv_d, tr_D_inv_plus_W_inv_W_deriv, c_opt);
									d_detmll_d_aux_par -= c_opt * (tr_PI_P_deriv - tr_D_inv_plus_W_inv_W_deriv);
									d_detmll_d_aux_par += (deriv_information_aux_par.array() * information_ll_.cwiseInverse().array()).sum();
								}
								// the log-determinant term is obtained from the stochastic trace estimator above, only the implicit derivative is summed here
								AccumulateAuxParGradTerms(deriv_information_aux_par, second_deriv_loc_aux_par, SigmaI_plus_W_inv_diag,
									SigmaI_plus_W_inv_d_mll_d_mode, false, d_detmll_d_aux_par, implicit_derivative);
							}
							else {//deriv_information_diag_loc_par is non-zero everywhere (!deriv_information_loc_par_has_zero )
								AccumulateAuxParGradTerms(deriv_information_aux_par, second_deriv_loc_aux_par, SigmaI_plus_W_inv_diag,
									SigmaI_plus_W_inv_d_mll_d_mode, true, d_detmll_d_aux_par, implicit_derivative);
							}
						}//end if grad_information_wrt_mode_non_zero_
						else {// grad_information_wrt_mode is zero
							if (use_random_effects_indices_of_data_) {
								vec_t Zt_deriv_information_aux_par;
								CalcZtVGivenIndices(num_data_, dim_mode_, random_effects_indices_of_data_, deriv_information_aux_par.data(), Zt_deriv_information_aux_par.data(), true);
								vec_t W_inv_d_W_W_inv = -1. * Zt_deriv_information_aux_par.cwiseProduct((information_ll_.cwiseInverse().cwiseProduct(information_ll_.cwiseInverse())));
								//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
								vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(W_inv_d_W_W_inv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
								double tr_SigmaI_plus_W_inv_W_deriv_d = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
								d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv_d;
								//variance reduction
								//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
								double tr_D_inv_plus_W_inv_W_deriv = (diagonal_approx_inv_preconditioner_.array() * W_inv_d_W_W_inv.array()).sum();
								for (int ii = 0; ii < dim_mode_; ii++) {
									tr_D_inv_plus_W_inv_W_deriv -= Preconditioner_PP_inv.col(ii).array().square().sum() * W_inv_d_W_W_inv[ii];
								}
								vec_t zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(W_inv_d_W_W_inv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
								double tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
								//optimal 
								CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv_d, tr_D_inv_plus_W_inv_W_deriv, c_opt);
								d_detmll_d_aux_par -= c_opt * (tr_PI_P_deriv - tr_D_inv_plus_W_inv_W_deriv);
								d_detmll_d_aux_par += (Zt_deriv_information_aux_par.array() * information_ll_.cwiseInverse().array()).sum();
							}
							else {
								//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
								vec_t W_inv_d_W_W_inv = -1. * deriv_information_aux_par.cwiseProduct((information_ll_.cwiseInverse().cwiseProduct(information_ll_.cwiseInverse())));
								vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(W_inv_d_W_W_inv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
								double tr_SigmaI_plus_W_inv_W_deriv_d = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
								d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv_d;
								//variance reduction
								//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
								double tr_D_inv_plus_W_inv_W_deriv = (diagonal_approx_inv_preconditioner_.array() * W_inv_d_W_W_inv.array()).sum();
								for (int ii = 0; ii < dim_mode_; ii++) {
									tr_D_inv_plus_W_inv_W_deriv -= Preconditioner_PP_inv.col(ii).array().square().sum() * W_inv_d_W_W_inv[ii];
								}
								vec_t zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(W_inv_d_W_W_inv.asDiagonal() * PI_Z)).colwise().sum()).transpose();
								double tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
								//optimal 
								CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv_d, tr_D_inv_plus_W_inv_W_deriv, c_opt);
								d_detmll_d_aux_par -= c_opt * (tr_PI_P_deriv - tr_D_inv_plus_W_inv_W_deriv);
								d_detmll_d_aux_par += (deriv_information_aux_par.array() * information_ll_.cwiseInverse().array()).sum();
							}
						}
						aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
					}
					SetGradAuxParsNotEstimated(aux_par_grad);
				}//end calc_aux_par_grad
			}//end cg_preconditioner_type_ == "fitc"
			else {//cg_preconditioner_type_ != "fitc"
				if (cg_preconditioner_type_ == "vifdu") {
					den_mat_t W_D_inv_inv_B_invt_rand_vec_trace_I(dim_mode_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
					for (int i = 0; i < num_rand_vec_trace_; ++i) {
						W_D_inv_inv_B_invt_rand_vec_trace_I.col(i) = W_D_inv_inv.cwiseProduct((B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(rand_vec_trace_I_.col(i)));
					}
					den_mat_t sigma_woodbury_woodbury_cross_cov_B_t_D_inv_W_D_inv_inv_B_invt_rand_vec_trace_I = (chol_fact_sigma_woodbury_woodbury_.solve(D_inv_B_cross_cov.transpose() * W_D_inv_inv_B_invt_rand_vec_trace_I));
					den_mat_t vecchia_cross_cov_sigma_woodbury_woodbury_inv_cross_cov_vecchia(dim_mode_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
					for (int i = 0; i < num_rand_vec_trace_; ++i) {
						vecchia_cross_cov_sigma_woodbury_woodbury_inv_cross_cov_vecchia.col(i) = W_D_inv_inv.cwiseProduct(D_inv_B_cross_cov * sigma_woodbury_woodbury_cross_cov_B_t_D_inv_W_D_inv_inv_B_invt_rand_vec_trace_I.col(i));
					}
					den_mat_t W_D_inv_inv_plus_vecchia_woodbury_woodbury_B_invt_rand_vec_trace_I = W_D_inv_inv_B_invt_rand_vec_trace_I + vecchia_cross_cov_sigma_woodbury_woodbury_inv_cross_cov_vecchia;
#pragma omp parallel for schedule(static)   
					for (int i = 0; i < num_rand_vec_trace_; ++i) {
						PI_Z.col(i) = B_rm_.triangularView<Eigen::UpLoType::UnitLower>().solve(W_D_inv_inv_plus_vecchia_woodbury_woodbury_B_invt_rand_vec_trace_I.col(i));
					}
					if (grad_information_wrt_mode_non_zero_) {
						//Z_SigmaI_plus_W_inv_W_deriv_PI_Z = -1 * (SigmaI_plus_W_inv_Z_.cwiseProduct(PI_Z)).cwiseProduct(W_deriv_rep);
						Z_SigmaI_plus_W_inv_W_deriv_PI_Z = (SigmaI_plus_W_inv_Z_.cwiseProduct(PI_Z)).cwiseProduct(W_deriv_rep);
						tr_SigmaI_plus_W_inv_W_deriv = Z_SigmaI_plus_W_inv_W_deriv_PI_Z.rowwise().mean();

						den_mat_t B_PI_Z(dim_mode_, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < num_rand_vec_trace_; ++i) {
							B_PI_Z.col(i) = B_rm_ * PI_Z.col(i);
						}
						//den_mat_t B_PI_Z = B_rm_ * PI_Z;
						Z_PI_P_deriv_PI_Z = (B_PI_Z.array() * W_deriv_rep.array() * B_PI_Z.array()).matrix();
						//Z_PI_P_deriv_PI_Z = -1 *(B_PI_Z.cwiseProduct(B_PI_Z)).cwiseProduct(W_deriv_rep);
						vec_t tr_PI_inv_W_deriv = Z_PI_P_deriv_PI_Z.rowwise().mean();
						d_log_det_Sigma_W_plus_I_d_mode = tr_SigmaI_plus_W_inv_W_deriv;
						//tr_PI_P_deriv_vec = -1. * W_D_inv_inv.cwiseProduct(deriv_information_diag_loc_par);
						tr_PI_P_deriv_vec = W_D_inv_inv.cwiseProduct(deriv_information_diag_loc_par);
						den_mat_t chol_fact_sigma_woodbury_woodbury_D_inv_B_cross_cov;
						//TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury_woodbury_,
						//	D_inv_B_cross_cov.transpose() * W_D_inv_inv.asDiagonal(), chol_fact_sigma_woodbury_woodbury_D_inv_B_cross_cov, false);
						GPBoost::solve_lower_triangular(chol_fact_sigma_woodbury_woodbury_,
							D_inv_B_cross_cov.transpose() * W_D_inv_inv.asDiagonal(), chol_fact_sigma_woodbury_woodbury_D_inv_B_cross_cov, GPU_use);
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < dim_mode_; ++i) {
							tr_PI_P_deriv_vec[i] += chol_fact_sigma_woodbury_woodbury_D_inv_B_cross_cov.col(i).array().square().sum() * deriv_information_diag_loc_par[i];
						}
						CalcOptimalCVectorized(Z_SigmaI_plus_W_inv_W_deriv_PI_Z, Z_PI_P_deriv_PI_Z, tr_SigmaI_plus_W_inv_W_deriv, tr_PI_P_deriv_vec, c_opt_vec);
						d_log_det_Sigma_W_plus_I_d_mode += c_opt_vec.cwiseProduct(tr_PI_P_deriv_vec - tr_PI_inv_W_deriv);
					}
				}
				else {
					if (grad_information_wrt_mode_non_zero_) {
						Z_SigmaI_plus_W_inv_W_deriv_PI_Z = SigmaI_plus_W_inv_Z_.cwiseProduct(rand_vec_trace_I_);
						d_log_det_Sigma_W_plus_I_d_mode = -1. * Z_SigmaI_plus_W_inv_W_deriv_PI_Z.rowwise().mean().cwiseProduct(deriv_information_diag_loc_par);
					}
				}
				//For implicit derivatives: calculate (Sigma^(-1) + W)^(-1) d_mll_d_mode
				bool has_NA_or_Inf = false;
				if (grad_information_wrt_mode_non_zero_) {
					d_mll_d_mode = 0.5 * d_log_det_Sigma_W_plus_I_d_mode;
					//For implicit derivatives: calculate (Sigma^(-1) + W)^(-1) d_mll_d_mode
					CGFVIFLaplaceVec(information_ll_, B_rm_, B_t_D_inv_rm_, chol_fact_sigma_woodbury, cross_cov,
						W_D_inv_inv, chol_fact_sigma_woodbury_woodbury_, d_mll_d_mode, SigmaI_plus_W_inv_d_mll_d_mode, has_NA_or_Inf,
						cg_max_num_it_, true, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, false);
					if (has_NA_or_Inf) {
						Log::REDebug(CG_NA_OR_INF_WARNING_GRADIENT_);
					}
				}
				// Calculate gradient wrt covariance parameters
				if (calc_cov_grad && some_cov_par_estimated) {
					sp_mat_rm_t SigmaI_deriv_rm, Bt_Dinv_Bgrad_rm, B_t_D_inv_D_grad_D_inv_B_rm;
					double explicit_derivative, d_log_det_Sigma_W_plus_I_d_cov_pars;
					int num_par = (int)B_grad.size();
					CHECK(re_comps_ip_cluster_i.size() == 1);
					for (int j = 0; j < (int)re_comps_ip_cluster_i.size(); ++j) {
						for (int ipar = 0; ipar < num_par; ++ipar) {
							if (estimate_cov_par_index[ipar] > 0) {
								std::shared_ptr<den_mat_t> cross_cov_grad = re_comps_cross_cov_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.);
								den_mat_t sigma_ip_grad = *(re_comps_ip_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.));
								den_mat_t sigma_ip_inv_sigma_ip_grad = chol_fact_sigma_ip.solve(sigma_ip_grad);
								if (ipar == 0) {
									SigmaI_deriv_rm = -B_rm_.transpose() * B_t_D_inv_rm_.transpose();//SigmaI_deriv = -SigmaI for variance parameters if there is only one GP
								}
								else {
									SigmaI_deriv_rm = sp_mat_rm_t(B_grad[ipar].transpose()) * B_t_D_inv_rm_.transpose();
									Bt_Dinv_Bgrad_rm = SigmaI_deriv_rm.transpose();
									B_t_D_inv_D_grad_D_inv_B_rm = B_t_D_inv_rm_ * sp_mat_rm_t(D_grad[ipar]) * B_t_D_inv_rm_.transpose();
									SigmaI_deriv_rm += Bt_Dinv_Bgrad_rm - B_t_D_inv_D_grad_D_inv_B_rm;
									Bt_Dinv_Bgrad_rm.resize(0, 0);
								}
								// Derivative of Woodbury matrix
								den_mat_t sigma_woodbury_grad = sigma_ip_grad;
								den_mat_t SigmaI_deriv_rm_cross_cov(dim_mode_, num_ip);
#pragma omp parallel for schedule(static)  
								for (int ii = 0; ii < num_ip; ii++) {
									SigmaI_deriv_rm_cross_cov.col(ii) = SigmaI_deriv_rm * (*cross_cov).col(ii);
								}
								den_mat_t cross_cov_SigmaI_deriv_rm_cross_cov;
								GPBoost::matmul((*cross_cov).transpose(), SigmaI_deriv_rm_cross_cov, cross_cov_SigmaI_deriv_rm_cross_cov, GPU_use);
								sigma_woodbury_grad += cross_cov_SigmaI_deriv_rm_cross_cov;
								//den_mat_t cross_cov_Bt_D_inv_B_cross_cov_grad = Bt_D_inv_B_cross_cov.transpose() * (*cross_cov_grad);
								den_mat_t cross_cov_Bt_D_inv_B_cross_cov_grad;
								GPBoost::matmul(Bt_D_inv_B_cross_cov.transpose(), (*cross_cov_grad), cross_cov_Bt_D_inv_B_cross_cov_grad, GPU_use);
								sigma_woodbury_grad += cross_cov_Bt_D_inv_B_cross_cov_grad + cross_cov_Bt_D_inv_B_cross_cov_grad.transpose();

								vec_t SigmaI_deriv_mode_part = SigmaI_deriv_rm * mode_;
								vec_t SigmaI_mode = SigmaI_rm * mode_;
								SigmaI_deriv_mode = SigmaI_deriv_mode_part - SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_deriv_mode_part)) -
									SigmaI_deriv_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)) -
									SigmaI_rm * ((*cross_cov_grad) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)) -
									SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov_grad).transpose() * SigmaI_mode)) +
									SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve(sigma_woodbury_grad * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)));
								explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv_mode));
								d_log_det_Sigma_W_plus_I_d_cov_pars = 0;
								//if (num_comps_total == 1 && ipar == 0) {
								//	d_log_det_Sigma_W_plus_I_d_cov_pars += dim_mode_;
								//}
								//else {
								d_log_det_Sigma_W_plus_I_d_cov_pars += (D_inv.diagonal().array() * D_grad[ipar].diagonal().array()).sum();
								//}
								d_log_det_Sigma_W_plus_I_d_cov_pars -= sigma_ip_inv_sigma_ip_grad.trace();
								d_log_det_Sigma_W_plus_I_d_cov_pars += (chol_fact_sigma_woodbury.solve(sigma_woodbury_grad)).trace();
								den_mat_t SigmaI_deriv_sample_vec_part(dim_mode_, num_rand_vec_trace_),
									SigmaI_sample_vec(dim_mode_, num_rand_vec_trace_), SigmaI_deriv_sample_vec(dim_mode_, num_rand_vec_trace_);
								den_mat_t sample_vec_final;
								if (cg_preconditioner_type_ == "vifdu") {
									sample_vec_final = PI_Z;
								}
								else {
									sample_vec_final = rand_vec_trace_I_;
								}
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < num_rand_vec_trace_; ++i) {
									SigmaI_deriv_sample_vec_part.col(i) = SigmaI_deriv_rm * sample_vec_final.col(i);
								}
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < num_rand_vec_trace_; ++i) {
									SigmaI_sample_vec.col(i) = SigmaI_rm * sample_vec_final.col(i);
								}
#pragma omp parallel for schedule(static)   
								for (int i = 0; i < num_rand_vec_trace_; ++i) {
									SigmaI_deriv_sample_vec.col(i) = SigmaI_deriv_sample_vec_part.col(i) - SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_deriv_sample_vec_part.col(i))) -
										SigmaI_deriv_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_sample_vec.col(i))) -
										SigmaI_rm * ((*cross_cov_grad) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_sample_vec.col(i))) -
										SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov_grad).transpose() * SigmaI_sample_vec.col(i))) +
										SigmaI_rm * ((*cross_cov) * chol_fact_sigma_woodbury.solve(sigma_woodbury_grad * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_sample_vec.col(i))));
								}
								vec_t sample_Sigma = (SigmaI_plus_W_inv_Z_.cwiseProduct(SigmaI_deriv_sample_vec)).colwise().sum();
								double stoch_tr = sample_Sigma.mean();
								d_log_det_Sigma_W_plus_I_d_cov_pars += stoch_tr;
								//Log::REInfo("stoch_tr %g", stoch_tr);
								//Log::REInfo("d_log_det_Sigma_W_plus_I_d_cov_pars %g", d_log_det_Sigma_W_plus_I_d_cov_pars);
								if (cg_preconditioner_type_ == "vifdu") {
									sp_mat_rm_t B_grad_rm = sp_mat_rm_t(B_grad[ipar]);
									den_mat_t P_grad_PI_Z = SigmaI_deriv_sample_vec;
									//if (!(num_comps_total == 1 && ipar == 0)) {
#pragma omp parallel for schedule(static)  
									for (int ii = 0; ii < num_rand_vec_trace_; ii++) {
										P_grad_PI_Z.col(ii) += B_grad_rm.transpose() * (information_ll_.cwiseProduct(B_rm_ * PI_Z.col(ii))) +
											B_rm_.transpose() * (information_ll_.cwiseProduct(B_grad_rm * PI_Z.col(ii)));
									}
									//}
									den_mat_t D_inv_B_cross_cov_grad(dim_mode_, num_ip);
#pragma omp parallel for schedule(static)  
									for (int ii = 0; ii < num_ip; ii++) {
										D_inv_B_cross_cov_grad.col(ii) = D_inv_rm_ * (B_rm_ * (*cross_cov_grad).col(ii));
									}
									den_mat_t D_inv_B_grad_cross_cov(dim_mode_, num_ip);
									//if (num_comps_total == 1 && ipar == 0) {
									//	D_inv_B_grad_cross_cov.setZero();
									//}
									//else {
#pragma omp parallel for schedule(static)  
									for (int ii = 0; ii < num_ip; ii++) {
										D_inv_B_grad_cross_cov.col(ii) = D_inv_rm_ * (B_grad_rm * (*cross_cov).col(ii));
									}
									//}
									den_mat_t D_inv_grad_B_cross_cov(dim_mode_, num_ip);
									vec_t D_inv_D_grad_D_inv;
									//if (num_comps_total == 1 && ipar == 0) {
									//	D_inv_D_grad_D_inv = -D_inv_rm_.diagonal();
									//}
									//else {
									D_inv_D_grad_D_inv = (D_inv_rm_.diagonal().array().square() * D_grad[ipar].diagonal().array()).matrix();
									//}
#pragma omp parallel for schedule(static)  
									for (int ii = 0; ii < num_ip; ii++) {
										D_inv_grad_B_cross_cov.col(ii) = -D_inv_D_grad_D_inv.cwiseProduct(B_rm_ * (*cross_cov).col(ii));
									}
									//den_mat_t D_inv_B_cross_cov_D_inv_B_cross_cov_grad = D_inv_B_cross_cov_.transpose() * (W_D_inv_inv.asDiagonal() * D_inv_B_cross_cov_grad);
									den_mat_t D_inv_B_cross_cov_t = D_inv_B_cross_cov_.transpose();
									den_mat_t W_D_inv_inv_D_inv_B_cross_cov_grad = W_D_inv_inv.asDiagonal() * D_inv_B_cross_cov_grad;
									den_mat_t D_inv_B_cross_cov_D_inv_B_cross_cov_grad;
									GPBoost::matmul(D_inv_B_cross_cov_t, W_D_inv_inv_D_inv_B_cross_cov_grad, D_inv_B_cross_cov_D_inv_B_cross_cov_grad, GPU_use);
									//den_mat_t D_inv_B_cross_cov_D_inv_B_cross_grad_cov = D_inv_B_cross_cov_.transpose() * (W_D_inv_inv.asDiagonal() * D_inv_B_grad_cross_cov);
									den_mat_t W_D_inv_inv_D_inv_B_grad_cross_cov = W_D_inv_inv.asDiagonal() * D_inv_B_grad_cross_cov;
									den_mat_t D_inv_B_cross_cov_D_inv_B_cross_grad_cov;
									GPBoost::matmul(D_inv_B_cross_cov_t, W_D_inv_inv_D_inv_B_grad_cross_cov, D_inv_B_cross_cov_D_inv_B_cross_grad_cov, GPU_use);
									//den_mat_t D_inv_grad_B_cross_cov_D_inv_B_cross_cov = D_inv_B_cross_cov_.transpose() * (W_D_inv_inv.asDiagonal() * D_inv_grad_B_cross_cov);
									den_mat_t W_D_inv_inv_D_inv_grad_B_cross_cov = W_D_inv_inv.asDiagonal() * D_inv_grad_B_cross_cov;
									den_mat_t D_inv_grad_B_cross_cov_D_inv_B_cross_cov;
									GPBoost::matmul(D_inv_B_cross_cov_t, W_D_inv_inv_D_inv_grad_B_cross_cov, D_inv_grad_B_cross_cov_D_inv_B_cross_cov, GPU_use);
									den_mat_t W_D_inv_inv_D_inv_D_grad_D_inv_D_inv_B_cross_cov = ((vec_t)((W_D_inv_inv.array().square() * D_inv_D_grad_D_inv.array()).matrix())).asDiagonal() * D_inv_B_cross_cov_;
									den_mat_t D_inv_B_cross_cov_t_dDiag_D_inv_B_cross_cov;
									GPBoost::matmul(D_inv_B_cross_cov_t, W_D_inv_inv_D_inv_D_grad_D_inv_D_inv_B_cross_cov, D_inv_B_cross_cov_t_dDiag_D_inv_B_cross_cov, GPU_use);
									den_mat_t sigma_woodbury_woodbury_grad = sigma_woodbury_grad -
										D_inv_B_cross_cov_D_inv_B_cross_cov_grad -
										D_inv_B_cross_cov_D_inv_B_cross_cov_grad.transpose() -
										D_inv_B_cross_cov_D_inv_B_cross_grad_cov -
										D_inv_B_cross_cov_D_inv_B_cross_grad_cov.transpose() -
										D_inv_grad_B_cross_cov_D_inv_B_cross_cov -
										D_inv_grad_B_cross_cov_D_inv_B_cross_cov.transpose() -
										D_inv_B_cross_cov_t_dDiag_D_inv_B_cross_cov;
									double tr_PI_P_grad = -(W_D_inv_inv.array() * D_inv_D_grad_D_inv.array()).sum() -
										(chol_fact_sigma_woodbury.solve(sigma_woodbury_grad)).trace() +
										(chol_fact_sigma_woodbury_woodbury_.solve(sigma_woodbury_woodbury_grad)).trace();
									vec_t sample_P = (PI_Z.cwiseProduct(P_grad_PI_Z)).colwise().sum();
									CalcOptimalC(sample_Sigma, sample_P, stoch_tr, tr_PI_P_grad, c_opt);
									d_log_det_Sigma_W_plus_I_d_cov_pars -= c_opt * (sample_P.mean() - tr_PI_P_grad);
								}
								explicit_derivative += 0.5 * d_log_det_Sigma_W_plus_I_d_cov_pars;
								cov_grad[ipar] = explicit_derivative;
								if (grad_information_wrt_mode_non_zero_) {
									cov_grad[ipar] -= SigmaI_plus_W_inv_d_mll_d_mode.dot(SigmaI_deriv_mode); //add implicit derivative
								}
							}//end estimate_cov_par_index[ipar] > 0
						}//end loop ipar
					}//end loop j
				}//end calc_cov_grad
				//Calculate gradient wrt fixed effects
				vec_t SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_diag_2nd_block;
				if (grad_information_wrt_mode_non_zero_ && ((use_random_effects_indices_of_data_ && calc_F_grad) || calc_aux_par_grad)) {
					//Stochastic Trace: Calculate diagonal of SigmaI_plus_W_inv for gradient of approx. marginal likelihood wrt. F
					SigmaI_plus_W_inv_diag = d_log_det_Sigma_W_plus_I_d_mode;
					SigmaI_plus_W_inv_diag.array() *= -1. / deriv_information_diag_loc_par.array();
					if (grad_information_wrt_mode_can_be_zero_for_some_points_) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)SigmaI_plus_W_inv_diag.size(); ++i) {
							if (GPBoost::IsZero<double>(deriv_information_diag_loc_par[i])) {
								SigmaI_plus_W_inv_diag[i] = 0.;//set to 0 for safety, but this is actually not needed
							}
						}
					} //end grad_information_wrt_mode_can_be_zero_for_some_points_
				}
				if (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad) {
					// Stochastic (Hutchinson) estimate of diag((Sigma^-1+W)^-1) for the second, fixed-effects-only block, using the raw (Cov = I) random vectors
					// rand_vec_trace_I2_ (for 'vifdu', rand_vec_trace_I_ is Cov = P, not I; for 'none' the two coincide,
					// see FindModePostRandEffCalcMLLFSVA). CGFVIFLaplaceVec solves (Sigma^-1+W) directly here (no
					// push-through needed, unlike the 'fitc' preconditioner case above)
					CHECK(num_sets_re_ == 1);
					den_mat_t SigmaI_plus_W_inv_RV_I(dim_mode_, num_rand_vec_trace_);
					bool has_NA_or_Inf_stoch_diag = false;
					for (int k = 0; k < num_rand_vec_trace_; ++k) {
						vec_t SigmaI_plus_W_inv_RV_I_col(dim_mode_);
						bool has_NA_or_Inf_k = false;
						CGFVIFLaplaceVec(information_ll_, B_rm_, B_t_D_inv_rm_, chol_fact_sigma_woodbury, cross_cov,
							W_D_inv_inv, chol_fact_sigma_woodbury_woodbury_, rand_vec_trace_I2_.col(k), SigmaI_plus_W_inv_RV_I_col, has_NA_or_Inf_k,
							cg_max_num_it_, true, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, false);
						SigmaI_plus_W_inv_RV_I.col(k) = SigmaI_plus_W_inv_RV_I_col;
						if (has_NA_or_Inf_k) {
							has_NA_or_Inf_stoch_diag = true;
						}
					}
					if (has_NA_or_Inf_stoch_diag) {
						Log::REDebug(CG_NA_OR_INF_WARNING_GRADIENT_);
					}
					SigmaI_plus_W_inv_diag_2nd_block = (SigmaI_plus_W_inv_RV_I.cwiseProduct(rand_vec_trace_I2_)).rowwise().mean();
					if (likelihood_type_ == "gaussian_heteroscedastic") {
						SigmaI_plus_W_inv_diag = SigmaI_plus_W_inv_diag_2nd_block;// there is no eta-block version for this likelihood
					}
				}
				if (calc_F_grad) {
					if (use_random_effects_indices_of_data_) {
						fixed_effect_grad = -first_deriv_ll_data_scale_;
						if (grad_information_wrt_mode_non_zero_) {
#pragma omp parallel for schedule(static)
							for (data_size_t i = 0; i < num_data_; ++i) {
								fixed_effect_grad[i] += 0.5 * deriv_information_diag_loc_par_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
									information_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
							}
						}
						if (HasSecondFEBlock()) {
							// 'include_coupled_zi_terms' is false: for a zero-inflated count regression the coupled log-determinant
							// term would need the data-scale diagonal of (Sigma^-1+W)^-1, which is only a stochastic estimate here,
							// so those terms are omitted -> the alpha gradient is approximate for ZI counts on this approximation.
							// 'SigmaI_plus_W_inv_diag_2nd_block' is the stochastic diagonal estimated above; for
							// 'gaussian_heteroscedastic' it is the same vector as 'SigmaI_plus_W_inv_diag'
							CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_data_scale_,
								SigmaI_plus_W_inv_diag_2nd_block, SigmaI_plus_W_inv_d_mll_d_mode, random_effects_indices_of_data_, false, fixed_effect_grad);
						}
					}
					else {
						fixed_effect_grad = -first_deriv_ll_;
						if (grad_information_wrt_mode_non_zero_) {
							vec_t d_mll_d_F_implicit = -(SigmaI_plus_W_inv_d_mll_d_mode.array() * information_ll_.array()).matrix();// implicit derivative
							fixed_effect_grad += d_mll_d_mode + d_mll_d_F_implicit;
						}
						if (HasSecondFEBlock()) {
							CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_,
								SigmaI_plus_W_inv_diag_2nd_block, SigmaI_plus_W_inv_d_mll_d_mode, nullptr, false, fixed_effect_grad);
						}
					}
				}
				//Calculate gradient wrt additional likelihood parameters
				if (calc_aux_par_grad) {
					vec_t neg_likelihood_deriv(num_aux_pars_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
					vec_t second_deriv_loc_aux_par(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
					vec_t deriv_information_aux_par(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
					vec_t d_mode_d_aux_par;
					CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par_ptr, neg_likelihood_deriv.data());
					for (int ind_ap = 0; ind_ap < num_aux_pars_; ++ind_ap) {
						CalcSecondDerivLogLikFirstDerivInformationAuxPar(y_data, y_data_int, location_par_ptr, ind_ap, second_deriv_loc_aux_par.data(), deriv_information_aux_par.data());
						double d_detmll_d_aux_par = 0., implicit_derivative = 0.;
						if (grad_information_wrt_mode_non_zero_) {
							bool deriv_information_loc_par_has_zero = false;
							if (grad_information_wrt_mode_can_be_zero_for_some_points_) {
								deriv_information_loc_par_has_zero = GPBoost::HasZero<double>(deriv_information_diag_loc_par.data(), (data_size_t)deriv_information_diag_loc_par.size());
							}
							if (deriv_information_loc_par_has_zero) {//deriv_information_diag_loc_par has some zeros
								if (use_random_effects_indices_of_data_) {
									vec_t Zt_deriv_information_aux_par(dim_mode_);
									CalcZtVGivenIndices(num_data_, dim_mode_, random_effects_indices_of_data_, deriv_information_aux_par.data(), Zt_deriv_information_aux_par.data(), true);
									if (cg_preconditioner_type_ == "vifdu") {
										//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
										vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(Zt_deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum()).transpose();
										double tr_SigmaI_plus_W_inv_W_deriv_d = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
										d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv_d;
										//variance reduction
										//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
										sp_mat_rm_t P_deriv_rm = B_rm_.transpose() * Zt_deriv_information_aux_par.asDiagonal() * B_rm_;
										vec_t W_D_inv_inv_neg_third_deriv_W_D_inv_inv = (W_D_inv_inv.array().square() * Zt_deriv_information_aux_par.array()).matrix();
										double tr_D_inv_plus_W_inv_W_deriv = (W_D_inv_inv.cwiseProduct(Zt_deriv_information_aux_par)).sum() +
											(chol_fact_sigma_woodbury_woodbury_.solve(D_inv_B_cross_cov_.transpose() * (W_D_inv_inv_neg_third_deriv_W_D_inv_inv.asDiagonal() * D_inv_B_cross_cov_))).trace();
										vec_t zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(P_deriv_rm * PI_Z)).colwise().sum()).transpose();
										double tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
										//optimal 
										CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv_d, tr_D_inv_plus_W_inv_W_deriv, c_opt);
										d_detmll_d_aux_par -= c_opt * (tr_PI_P_deriv - tr_D_inv_plus_W_inv_W_deriv);
									}
									else {
										d_detmll_d_aux_par = (SigmaI_plus_W_inv_Z_.cwiseProduct(Zt_deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum().mean();
									}
								}
								else {
									if (cg_preconditioner_type_ == "vifdu") {
										//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
										vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum()).transpose();
										double tr_SigmaI_plus_W_inv_W_deriv_d = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
										d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv_d;
										//variance reduction
										//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
										sp_mat_rm_t P_deriv_rm = B_rm_.transpose() * deriv_information_aux_par.asDiagonal() * B_rm_;
										vec_t W_D_inv_inv_neg_third_deriv_W_D_inv_inv = (W_D_inv_inv.array().square() * deriv_information_aux_par.array()).matrix();
										double tr_D_inv_plus_W_inv_W_deriv = (W_D_inv_inv.cwiseProduct(deriv_information_aux_par)).sum() +
											(chol_fact_sigma_woodbury_woodbury_.solve(D_inv_B_cross_cov_.transpose() * (W_D_inv_inv_neg_third_deriv_W_D_inv_inv.asDiagonal() * D_inv_B_cross_cov_))).trace();
										vec_t zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(P_deriv_rm * PI_Z)).colwise().sum()).transpose();
										double tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
										//optimal 
										CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv_d, tr_D_inv_plus_W_inv_W_deriv, c_opt);
										d_detmll_d_aux_par -= c_opt * (tr_PI_P_deriv - tr_D_inv_plus_W_inv_W_deriv);
									}
									else {
										d_detmll_d_aux_par = (SigmaI_plus_W_inv_Z_.cwiseProduct(deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum().mean();
									}
								}
								// the log-determinant term is obtained from the stochastic trace estimator above, only the implicit derivative is summed here
								AccumulateAuxParGradTerms(deriv_information_aux_par, second_deriv_loc_aux_par, SigmaI_plus_W_inv_diag,
									SigmaI_plus_W_inv_d_mll_d_mode, false, d_detmll_d_aux_par, implicit_derivative);
							}
							else {//deriv_information_diag_loc_par is non-zero everywhere (!deriv_information_loc_par_has_zero )
								AccumulateAuxParGradTerms(deriv_information_aux_par, second_deriv_loc_aux_par, SigmaI_plus_W_inv_diag,
									SigmaI_plus_W_inv_d_mll_d_mode, true, d_detmll_d_aux_par, implicit_derivative);
							}
						}//end if grad_information_wrt_mode_non_zero_
						else {// grad_information_wrt_mode is zero
							if (use_random_effects_indices_of_data_) {
								if (cg_preconditioner_type_ == "vifdu") {
									//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
									vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum()).transpose();
									double tr_SigmaI_plus_W_inv_W_deriv_d = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
									d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv_d;
									//variance reduction
									//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
									sp_mat_rm_t P_deriv_rm = B_rm_.transpose() * deriv_information_aux_par.asDiagonal() * B_rm_;
									vec_t W_D_inv_inv_neg_third_deriv_W_D_inv_inv = (W_D_inv_inv.array().square() * deriv_information_aux_par.array()).matrix();
									double tr_D_inv_plus_W_inv_W_deriv = (W_D_inv_inv.cwiseProduct(deriv_information_aux_par)).sum() +
										(chol_fact_sigma_woodbury_woodbury_.solve(D_inv_B_cross_cov_.transpose() * (W_D_inv_inv_neg_third_deriv_W_D_inv_inv.asDiagonal() * D_inv_B_cross_cov_))).trace();
									vec_t zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(P_deriv_rm * PI_Z)).colwise().sum()).transpose();
									double tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
									//optimal 
									CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv_d, tr_D_inv_plus_W_inv_W_deriv, c_opt);
									d_detmll_d_aux_par -= c_opt * (tr_PI_P_deriv - tr_D_inv_plus_W_inv_W_deriv);
								}
								else {
									d_detmll_d_aux_par = (SigmaI_plus_W_inv_Z_.cwiseProduct(deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum().mean();
								}
							}
							else {
								if (cg_preconditioner_type_ == "vifdu") {
									//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
									vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum()).transpose();
									double tr_SigmaI_plus_W_inv_W_deriv_d = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
									d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv_d;
									//variance reduction
									//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
									sp_mat_rm_t P_deriv_rm = B_rm_.transpose() * deriv_information_aux_par.asDiagonal() * B_rm_;
									vec_t W_D_inv_inv_neg_third_deriv_W_D_inv_inv = (W_D_inv_inv.array().square() * deriv_information_aux_par.array()).matrix();
									double tr_D_inv_plus_W_inv_W_deriv = (W_D_inv_inv.cwiseProduct(deriv_information_aux_par)).sum() +
										(chol_fact_sigma_woodbury_woodbury_.solve(D_inv_B_cross_cov_.transpose() * (W_D_inv_inv_neg_third_deriv_W_D_inv_inv.asDiagonal() * D_inv_B_cross_cov_))).trace();
									vec_t zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(P_deriv_rm * PI_Z)).colwise().sum()).transpose();
									double tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
									//optimal 
									CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv_d, tr_D_inv_plus_W_inv_W_deriv, c_opt);
									d_detmll_d_aux_par -= c_opt * (tr_PI_P_deriv - tr_D_inv_plus_W_inv_W_deriv);
								}
								else {
									d_detmll_d_aux_par = (SigmaI_plus_W_inv_Z_.cwiseProduct(deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum().mean();
								}
							}
						}
						aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
					}
					SetGradAuxParsNotEstimated(aux_par_grad);
				}//end calc_aux_par_grad
			}//end cg_preconditioner_type_ != "fitc"
		}//end matrix_inversion_method_ == "iterative"
		else {// matrix_inversion_method_ == "cholesky"
			// Calculate (Sigma^-1 + W)^-1
			sp_mat_t L_inv(dim_mode_, dim_mode_);
			L_inv.setIdentity();
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, L_inv, L_inv, false);
			vec_t SigmaI_plus_W_inv_d_mll_d_mode, SigmaI_plus_W_inv_diag;
			sp_mat_t SigmaI_plus_W_inv, SigmaI;
			// Calculate gradient wrt covariance parameters
			bool some_cov_par_estimated = std::any_of(estimate_cov_par_index.begin(), estimate_cov_par_index.end(), [](int x) { return x > 0; });
			bool calc_cov_grad_internal = calc_cov_grad && some_cov_par_estimated;
			if (calc_cov_grad_internal) {
				double explicit_derivative;
				sp_mat_t SigmaI_deriv, BgradT_Dinv_B, Bt_Dinv_Bgrad;
				sp_mat_t D_inv_B;
				D_inv_B = D_inv * B;
				SigmaI = B.transpose() * D_inv * B;
				int par_count = 0;
				den_mat_t sigma_resid_plus_W_inv_cross_cov = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(information_ll_.asDiagonal() * (*cross_cov));
				den_mat_t sigma_resid_inv_sigma_resid_plus_W_inv_cross_cov(dim_mode_, num_ip);
#pragma omp parallel for schedule(static)   
				for (int ii = 0; ii < num_ip; ++ii) {
					sigma_resid_inv_sigma_resid_plus_W_inv_cross_cov.col(ii) = B_t_D_inv_rm_ * (B_rm_ * sigma_resid_plus_W_inv_cross_cov.col(ii));
				}
				den_mat_t sigma_woodbury_2 = (sigma_ip_stable)+(*cross_cov).transpose() * sigma_resid_inv_sigma_resid_plus_W_inv_cross_cov;
				chol_den_mat_t chol_fact_sigma_woodbury_2;
				chol_fact_sigma_woodbury_2.compute(sigma_woodbury_2);
				CheckCholeskyFactorization(chol_fact_sigma_woodbury_2, "Laplace gradient Woodbury matrix");
				int num_par = (int)B_grad.size();
				CHECK(re_comps_ip_cluster_i.size() == 1);
				for (int j = 0; j < (int)re_comps_ip_cluster_i.size(); ++j) {
					for (int ipar = 0; ipar < num_par; ++ipar) {
						std::shared_ptr<den_mat_t> cross_cov_grad;
						den_mat_t sigma_woodbury_grad, sigma_ip_grad, sigma_ip_inv_sigma_ip_grad, SigmaI_deriv_rm_cross_cov, cross_cov_Bt_D_inv_B_cross_cov_grad;
						sp_mat_rm_t SigmaI_deriv_rm;
						if (estimate_cov_par_index[par_count] > 0 || ipar == 0) {
							cross_cov_grad = re_comps_cross_cov_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.);
							sigma_ip_grad = *(re_comps_ip_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.));
							sigma_ip_inv_sigma_ip_grad = chol_fact_sigma_ip.solve(sigma_ip_grad);
							// Calculate SigmaI_deriv
							if (ipar == 0) {
								SigmaI_deriv = -B.transpose() * D_inv_B;//SigmaI_deriv = -SigmaI for variance parameters if there is only one GP
							}
							else {
								SigmaI_deriv = B_grad[ipar].transpose() * D_inv_B;
								Bt_Dinv_Bgrad = SigmaI_deriv.transpose();
								SigmaI_deriv += Bt_Dinv_Bgrad - D_inv_B.transpose() * D_grad[ipar] * D_inv_B;
								Bt_Dinv_Bgrad.resize(0, 0);
							}
							// Derivative of Woodbury matrix
							sigma_woodbury_grad = sigma_ip_grad;
							SigmaI_deriv_rm = sp_mat_rm_t(SigmaI_deriv);
							SigmaI_deriv_rm_cross_cov = den_mat_t(dim_mode_, num_ip);
#pragma omp parallel for schedule(static)  
							for (int ii = 0; ii < num_ip; ii++) {
								SigmaI_deriv_rm_cross_cov.col(ii) = SigmaI_deriv_rm * (*cross_cov).col(ii);
							}
							sigma_woodbury_grad += (*cross_cov).transpose() * SigmaI_deriv_rm_cross_cov;
							cross_cov_Bt_D_inv_B_cross_cov_grad = Bt_D_inv_B_cross_cov.transpose() * (*cross_cov_grad);
							sigma_woodbury_grad += cross_cov_Bt_D_inv_B_cross_cov_grad + cross_cov_Bt_D_inv_B_cross_cov_grad.transpose();
						}
						if (ipar == 0) {
							// Calculate SigmaI_plus_W_inv = L_inv.transpose() * L_inv at non-zero entries of SigmaI_deriv
							//	Note: fully calculating SigmaI_plus_W_inv = L_inv.transpose() * L_inv is very slow
							SigmaI_plus_W_inv = SigmaI_deriv;
							CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_W_inv, true);
							den_mat_t SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(Bt_D_inv_B_cross_cov);
							den_mat_t woodbury_SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov_t;
							TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury_woodbury_,
								SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov.transpose(), woodbury_SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov_t, false);
							vec_t SigmaI_plus_W_inv_diag_part = (woodbury_SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov_t.cwiseProduct(woodbury_SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov_t)).colwise().sum();
							SigmaI_plus_W_inv_diag = (SigmaI_plus_W_inv.diagonal().array() + SigmaI_plus_W_inv_diag_part.array()).matrix();
							if (grad_information_wrt_mode_non_zero_) {
								d_mll_d_mode = 0.5 * (SigmaI_plus_W_inv_diag.array() * deriv_information_diag_loc_par.array()).matrix();
								vec_t Sigma_d_mll_d_mode_part = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(d_mll_d_mode);
								vec_t Sigma_d_mll_d_mode_part1 = D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve(Sigma_d_mll_d_mode_part);
								vec_t Sigma_d_mll_d_mode = Sigma_d_mll_d_mode_part1 + (*cross_cov) * (chol_fact_sigma_ip.solve((*cross_cov).transpose() * d_mll_d_mode));
								vec_t W_Sigma_d_mll_d_mode = information_ll_.asDiagonal() * Sigma_d_mll_d_mode;
								vec_t SigmaI_plus_W_inv_d_mll_d_mode_part = B_t_D_inv_rm_ * (B_rm_ * chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(W_Sigma_d_mll_d_mode));
								SigmaI_plus_W_inv_d_mll_d_mode = information_ll_.cwiseInverse().asDiagonal() * (SigmaI_plus_W_inv_d_mll_d_mode_part - sigma_resid_inv_sigma_resid_plus_W_inv_cross_cov * chol_fact_sigma_woodbury_2.solve((*cross_cov).transpose() * SigmaI_plus_W_inv_d_mll_d_mode_part));
							}
						}//end if ipar == 0
						if (estimate_cov_par_index[par_count] > 0) {
							vec_t SigmaI_deriv_mode_part = SigmaI_deriv * mode_;
							vec_t SigmaI_mode = SigmaI * mode_;
							vec_t SigmaI_deriv_mode = SigmaI_deriv_mode_part - SigmaI * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_deriv_mode_part)) -
								SigmaI_deriv * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)) -
								SigmaI * ((*cross_cov_grad) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)) -
								SigmaI * ((*cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov_grad).transpose() * SigmaI_mode)) +
								SigmaI * ((*cross_cov) * chol_fact_sigma_woodbury.solve(sigma_woodbury_grad * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * SigmaI_mode)));
							explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv_mode) +
								(SigmaI_deriv.cwiseProduct(SigmaI_plus_W_inv)).sum());
							explicit_derivative += 0.5 * (D_inv.diagonal().array() * D_grad[ipar].diagonal().array()).sum();
							explicit_derivative -= 0.5 * sigma_ip_inv_sigma_ip_grad.trace();
							den_mat_t sigma_woodbury_woodbury_grad = sigma_woodbury_grad;
							den_mat_t Bt_D_inv_B_cross_cov_grad(dim_mode_, num_ip);
#pragma omp parallel for schedule(static)  
							for (int ii = 0; ii < num_ip; ii++) {
								Bt_D_inv_B_cross_cov_grad.col(ii) = B_t_D_inv_rm_ * (B_rm_ * (*cross_cov_grad).col(ii));
							}
							den_mat_t Sigma_I_cross_cov_SigmaI_plus_W_inv_Sigma_I_cross_cov_grad = Bt_D_inv_B_cross_cov.transpose() * (chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(Bt_D_inv_B_cross_cov_grad));
							sigma_woodbury_woodbury_grad -= Sigma_I_cross_cov_SigmaI_plus_W_inv_Sigma_I_cross_cov_grad + Sigma_I_cross_cov_SigmaI_plus_W_inv_Sigma_I_cross_cov_grad.transpose();
							den_mat_t Sigma_I_cross_cov_SigmaI_plus_W_invSigmaI_deriv_cross_cov = Bt_D_inv_B_cross_cov.transpose() * chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(SigmaI_deriv_rm_cross_cov);
							sigma_woodbury_woodbury_grad -= Sigma_I_cross_cov_SigmaI_plus_W_invSigmaI_deriv_cross_cov + Sigma_I_cross_cov_SigmaI_plus_W_invSigmaI_deriv_cross_cov.transpose();
							den_mat_t SigmaI_plus_W_invSigmaI_cross_cov(dim_mode_, num_ip);
#pragma omp parallel for schedule(static)  
							for (int ii = 0; ii < num_ip; ii++) {
								SigmaI_plus_W_invSigmaI_cross_cov.col(ii) = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(SigmaI_deriv_rm * chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(Bt_D_inv_B_cross_cov.col(ii)));
							}
							sigma_woodbury_woodbury_grad += Bt_D_inv_B_cross_cov.transpose() * SigmaI_plus_W_invSigmaI_cross_cov;
							explicit_derivative += 0.5 * (chol_fact_sigma_woodbury_woodbury_.solve(sigma_woodbury_woodbury_grad)).trace();
							cov_grad[par_count] = explicit_derivative;
							if (grad_information_wrt_mode_non_zero_) {
								cov_grad[par_count] -= SigmaI_plus_W_inv_d_mll_d_mode.dot(SigmaI_deriv_mode);//add implicit derivative
							}
						}//end estimate_cov_par_index[ipar]
						par_count++;
					}//end loop ipar
				}//end loop j
			}//end calc_cov_grad_internal
			// Calcul
			if (calc_F_grad || calc_aux_par_grad) {
				if (!calc_cov_grad_internal) {
					if (calc_aux_par_grad || grad_information_wrt_mode_non_zero_ || (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad)) {
						SigmaI_plus_W_inv = D_inv;
						CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_W_inv, true);
						den_mat_t SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(Bt_D_inv_B_cross_cov);
						den_mat_t woodbury_SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov_t;
						TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury_woodbury_,
							SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov.transpose(), woodbury_SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov_t, false);
						SigmaI_plus_W_inv_diag = (woodbury_SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov_t.cwiseProduct(woodbury_SigmaI_plus_W_inv_Bt_D_inv_B_cross_cov_t)).colwise().sum();
						SigmaI_plus_W_inv_diag = (SigmaI_plus_W_inv.diagonal().array() + SigmaI_plus_W_inv_diag.array()).matrix();
					}
					if (grad_information_wrt_mode_non_zero_) {
						d_mll_d_mode = 0.5 * (SigmaI_plus_W_inv_diag.array() * deriv_information_diag_loc_par.array()).matrix();

						den_mat_t sigma_resid_plus_W_inv_cross_cov = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(information_ll_.asDiagonal() * (*cross_cov));
						den_mat_t sigma_resid_inv_sigma_resid_plus_W_inv_cross_cov(dim_mode_, num_ip);
#pragma omp parallel for schedule(static)   
						for (int ii = 0; ii < num_ip; ++ii) {
							sigma_resid_inv_sigma_resid_plus_W_inv_cross_cov.col(ii) = B_t_D_inv_rm_ * (B_rm_ * sigma_resid_plus_W_inv_cross_cov.col(ii));
						}
						den_mat_t sigma_woodbury_2 = (sigma_ip_stable)+(*cross_cov).transpose() * sigma_resid_inv_sigma_resid_plus_W_inv_cross_cov;
						chol_den_mat_t chol_fact_sigma_woodbury_2;
						chol_fact_sigma_woodbury_2.compute(sigma_woodbury_2);
						CheckCholeskyFactorization(chol_fact_sigma_woodbury_2, "Laplace gradient Woodbury matrix");

						vec_t Sigma_d_mll_d_mode_part = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(d_mll_d_mode);
						vec_t Sigma_d_mll_d_mode_part1 = D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve(Sigma_d_mll_d_mode_part);
						vec_t Sigma_d_mll_d_mode = Sigma_d_mll_d_mode_part1 + (*cross_cov) * (chol_fact_sigma_ip.solve((*cross_cov).transpose() * d_mll_d_mode));
						vec_t W_Sigma_d_mll_d_mode = information_ll_.asDiagonal() * Sigma_d_mll_d_mode;
						vec_t SigmaI_plus_W_inv_d_mll_d_mode_part = B_t_D_inv_rm_ * (B_rm_ * chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(W_Sigma_d_mll_d_mode));
						SigmaI_plus_W_inv_d_mll_d_mode = information_ll_.cwiseInverse().asDiagonal() * (SigmaI_plus_W_inv_d_mll_d_mode_part - sigma_resid_inv_sigma_resid_plus_W_inv_cross_cov * chol_fact_sigma_woodbury_2.solve((*cross_cov).transpose() * SigmaI_plus_W_inv_d_mll_d_mode_part));
					}
				}
				else if (calc_aux_par_grad || (use_random_effects_indices_of_data_ && grad_information_wrt_mode_non_zero_) || (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad)) {
					SigmaI_plus_W_inv_diag = (SigmaI_plus_W_inv.diagonal().array() + SigmaI_plus_W_inv_diag.array()).matrix();
				}
			}
			// Calculate gradient wrt fixed effects
			if (calc_F_grad) {
				if (use_random_effects_indices_of_data_) {
					fixed_effect_grad = -first_deriv_ll_data_scale_;
					if (grad_information_wrt_mode_non_zero_) {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							fixed_effect_grad[i] += 0.5 * deriv_information_diag_loc_par_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
								information_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
						}
					}
					if (HasSecondFEBlock()) {
						// 'include_coupled_zi_terms' is false: for a zero-inflated count regression the coupled log-determinant
						// term would need the FSVA data-scale diagonal of (Sigma^-1+W)^-1, which is only a stochastic estimate
						// here, so those terms are omitted -> the alpha gradient is approximate for ZI counts on FSVA
						CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_data_scale_,
							SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_d_mll_d_mode, random_effects_indices_of_data_, false, fixed_effect_grad);
					}
				}
				else {
					fixed_effect_grad = -first_deriv_ll_;
					if (grad_information_wrt_mode_non_zero_) {
						vec_t d_mll_d_F_implicit = -(SigmaI_plus_W_inv_d_mll_d_mode.array() * information_ll_.array()).matrix();// implicit derivative
						fixed_effect_grad += d_mll_d_mode + d_mll_d_F_implicit;
					}
					if (HasSecondFEBlock()) {
						CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_,
							SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_d_mll_d_mode, nullptr, false, fixed_effect_grad);
					}
				}
			}//end calc_F_grad
			// calculate gradient wrt additional likelihood parameters
			if (calc_aux_par_grad) {
				CalcAuxParGradLaplaceExactDiag(y_data, y_data_int, location_par_ptr, SigmaI_plus_W_inv_diag,
					SigmaI_plus_W_inv_d_mll_d_mode, aux_par_grad);
			}//end calc_aux_par_grad
		}
	}//end CalcGradNegMargLikelihoodLaplaceApproxFSVA

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcGradNegMargLikelihoodLaplaceApproxVecchia(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		std::map<int, sp_mat_t>& B,
		std::map<int, sp_mat_t>& D_inv,
		std::map<int, std::vector<sp_mat_t>>& B_grad,
		std::map<int, std::vector<sp_mat_t>>& D_grad,
		bool calc_cov_grad,
		bool calc_F_grad,
		bool calc_aux_par_grad,
		double* cov_grad,
		vec_t& fixed_effect_grad,
		double* aux_par_grad,
		bool calc_mode,
		int num_comps_total,
		bool call_for_std_dev_coef,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const den_mat_t& chol_ip_cross_cov,
		const chol_den_mat_t& chol_fact_sigma_ip,
		data_size_t cluster_i,
		REModelTemplate<T_mat, T_chol>* re_model,
		const std::vector<int>& estimate_cov_par_index,
		bool GPU_use) {
		if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLVecchia(y_data, y_data_int, fixed_effects, B, D_inv, false, Sigma_L_k_, true, mll,
				re_comps_ip_cluster_i, re_comps_cross_cov_cluster_i, chol_ip_cross_cov, chol_fact_sigma_ip, cluster_i, re_model);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			if (call_for_std_dev_coef) {
				Log::REFatal(CANNOT_CALC_STDEV_ERROR_);
			}
			else {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
		}
		CHECK(mode_has_been_calculated_);
		// Initialize variables
		vec_t location_par;//location parameter = mode of random effects + fixed effects
		double* location_par_ptr;
		InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
		vec_t deriv_information_diag_loc_par;//first derivative of the diagonal of the Fisher information wrt the location parameter (= usually negative third derivatives of the log-likelihood wrt the locatin parameter)
		vec_t deriv_information_diag_loc_par_data_scale;//first derivative of the diagonal of the Fisher information wrt the location parameter on the data-scale (only used if use_random_effects_indices_of_data_), the vector 'deriv_information_diag_loc_par' actually contains diag_ZtDerivInformationZ if use_random_effects_indices_of_data_
		if (grad_information_wrt_mode_non_zero_) {
			CalcFirstDerivInformationLocPar(y_data, y_data_int, location_par_ptr, deriv_information_diag_loc_par, deriv_information_diag_loc_par_data_scale);
		}
		vec_t d_mll_d_mode, SigmaI_plus_W_inv_d_mll_d_mode, SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_diag_2nd_block, SigmaI_plus_W_inv_off_diag;
		if (matrix_inversion_method_ == "iterative") {
			if (cg_preconditioner_type_ == "vecchia_response") {
				Log::REFatal("Calculation of gradients is currently not correctly implemented for the '%s' preconditioner ", cg_preconditioner_type_.c_str());
			}
			if ((cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc") && HasNegativeValueInformationLogLik()) {
				Log::REFatal("CalcGradNegMargLikelihoodLaplaceApproxVecchia: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
					"The stochastic gradient calculation with the '%s' preconditioner requires W to be nonnegative ", cg_preconditioner_type_.c_str());
			}
			vec_t d_log_det_Sigma_W_plus_I_d_mode;
			//Declarations for preconditioner "piv_chol_on_Sigma"
			vec_t diag_WI;
			den_mat_t WI_PI_Z, WI_WI_plus_Sigma_inv_Z;
			//Declarations for preconditioner "Sigma_inv_plus_BtWB"
			vec_t D_inv_plus_W_inv_diag;
			den_mat_t PI_Z; //also used for preconditioner "zero_infill_incomplete_cholesky"
			//Stochastic Trace: Calculate gradient of approx. marginal likelihood wrt. the mode (and thus also F here if !use_random_effects_indices_of_data_)
			if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
				vec_t deriv_information_diag_loc_par_all = vec_t::Zero(dim_mode_);
				deriv_information_diag_loc_par_all.segment(0, dim_mode_per_set_re_) = deriv_information_diag_loc_par;
				vec_t d_log_det_Sigma_W_plus_I_d_mode_temp;
				CalcLogDetStochDerivModeVecchia(deriv_information_diag_loc_par_all, dim_mode_, d_log_det_Sigma_W_plus_I_d_mode_temp, D_inv_plus_W_inv_diag, diag_WI,
					PI_Z, WI_PI_Z, WI_WI_plus_Sigma_inv_Z, re_comps_cross_cov_cluster_i, GPU_use);
				d_log_det_Sigma_W_plus_I_d_mode = vec_t::Zero(dim_mode_);
				d_log_det_Sigma_W_plus_I_d_mode.segment(dim_mode_per_set_re_, dim_mode_per_set_re_) =
					d_log_det_Sigma_W_plus_I_d_mode_temp.segment(0, dim_mode_per_set_re_);
			}
			else {
				CalcLogDetStochDerivModeVecchia(deriv_information_diag_loc_par, dim_mode_, d_log_det_Sigma_W_plus_I_d_mode, D_inv_plus_W_inv_diag, diag_WI, PI_Z, WI_PI_Z,
					WI_WI_plus_Sigma_inv_Z, re_comps_cross_cov_cluster_i, GPU_use);
			}
			//For implicit derivatives: calculate (Sigma^(-1) + W)^(-1) d_mll_d_mode
			if (grad_information_wrt_mode_non_zero_) {
				d_mll_d_mode = 0.5 * d_log_det_Sigma_W_plus_I_d_mode;
				SigmaI_plus_W_inv_d_mll_d_mode = vec_t(dim_mode_);
				bool has_NA_or_Inf = false;
				Inv_SigmaI_plus_ZtWZ_Vecchia_iterative_given_PC(cg_max_num_it_, re_comps_cross_cov_cluster_i, d_mll_d_mode, SigmaI_plus_W_inv_d_mll_d_mode, true, has_NA_or_Inf);
				if (has_NA_or_Inf) {
					Log::REDebug(CG_NA_OR_INF_WARNING_GRADIENT_);
				}
			}
			// Calculate gradient wrt covariance parameters
			bool some_cov_par_estimated = std::any_of(estimate_cov_par_index.begin(), estimate_cov_par_index.end(), [](int x) { return x > 0; });
			if (calc_cov_grad && some_cov_par_estimated) {
				sp_mat_rm_t SigmaI_deriv_rm, Bt_Dinv_Bgrad_rm, B_t_D_inv_D_grad_D_inv_B_rm;
				vec_t SigmaI_deriv_mode;
				double explicit_derivative, d_log_det_Sigma_W_plus_I_d_cov_pars;
				int num_par = (int)B_grad[0].size();
				for (int igp = 0; igp < num_sets_re_; ++igp) {
					sp_mat_t D_inv_B;
					if (num_sets_re_ > 1) {
						D_inv_B = D_inv[igp] * B[igp];
					}
					for (int j = 0; j < num_par; ++j) {
						if (estimate_cov_par_index[j + igp * num_par] > 0) {
							// Calculate SigmaI_deriv
							if (num_sets_re_ == 1) {
								if (num_comps_total == 1 && j == 0) {
									SigmaI_deriv_rm = -B_rm_.transpose() * B_t_D_inv_rm_.transpose();//SigmaI_deriv = -SigmaI for variance parameters if there is only one GP
								}
								else {
									//SigmaI_deriv_rm = sp_mat_rm_t(B_grad[0][j].transpose()) * B_t_D_inv_rm_.transpose();
									sp_mat_rm_t B_t_D_inv_rm_t = sp_mat_rm_t(B_t_D_inv_rm_.transpose());
									GPBoost::spmatmul(sp_mat_rm_t(B_grad[0][j].transpose()), B_t_D_inv_rm_t, SigmaI_deriv_rm, GPU_use);
									Bt_Dinv_Bgrad_rm = SigmaI_deriv_rm.transpose();
									//B_t_D_inv_D_grad_D_inv_B_rm = B_t_D_inv_rm_ * sp_mat_rm_t(D_grad[0][j]) * B_t_D_inv_rm_.transpose();
									sp_mat_rm_t B_t_D_inv_D_grad_D_inv_B_rm_inter;
									GPBoost::spmatmul(sp_mat_rm_t(D_grad[0][j]), B_t_D_inv_rm_t, B_t_D_inv_D_grad_D_inv_B_rm_inter, GPU_use);
									GPBoost::spmatmul(B_t_D_inv_rm_, B_t_D_inv_D_grad_D_inv_B_rm_inter, B_t_D_inv_D_grad_D_inv_B_rm, GPU_use);
									SigmaI_deriv_rm += Bt_Dinv_Bgrad_rm - B_t_D_inv_D_grad_D_inv_B_rm;
									Bt_Dinv_Bgrad_rm.resize(0, 0);
								}
								CalcLogDetStochDerivCovParVecchia(dim_mode_, num_comps_total, j, SigmaI_deriv_rm, B_grad[0][j], D_grad[0][j], D_inv_plus_W_inv_diag, PI_Z, WI_PI_Z, d_log_det_Sigma_W_plus_I_d_cov_pars);
							}
							else {
								CHECK(num_sets_re_ == 2);
								if (num_comps_total == 1 && j == 0) {
									SigmaI_deriv_rm = sp_mat_rm_t(-B[igp].transpose() * D_inv_B);//SigmaI_deriv = -SigmaI for variance parameters if there is only one GP
								}
								else {
									SigmaI_deriv_rm = sp_mat_rm_t(B_grad[igp][j].transpose() * D_inv_B);
									Bt_Dinv_Bgrad_rm = SigmaI_deriv_rm.transpose();
									SigmaI_deriv_rm += Bt_Dinv_Bgrad_rm - sp_mat_rm_t(D_inv_B.transpose() * D_grad[igp][j] * D_inv_B);
									Bt_Dinv_Bgrad_rm.resize(0, 0);
								}
								sp_mat_rm_t SigmaI_deriv_1(dim_mode_per_set_re_, dim_mode_per_set_re_), SigmaI_deriv_2(dim_mode_per_set_re_, dim_mode_per_set_re_);
								if (igp == 0) {
									SigmaI_deriv_1 = SigmaI_deriv_rm;
								}
								else {
									SigmaI_deriv_2 = SigmaI_deriv_rm;
								}
								GPBoost::CreatSparseBlockDiagonalMartix<sp_mat_rm_t>(SigmaI_deriv_1, SigmaI_deriv_2, SigmaI_deriv_rm);
								sp_mat_t grad_1(dim_mode_per_set_re_, dim_mode_per_set_re_), grad_2(dim_mode_per_set_re_, dim_mode_per_set_re_), B_grad_all, D_grad_all;
								if (igp == 0) {
									grad_1 = B_grad[0][j];
								}
								else {
									grad_2 = B_grad[1][j];
								}
								GPBoost::CreatSparseBlockDiagonalMartix<sp_mat_t>(grad_1, grad_2, B_grad_all);
								if (igp == 0) {
									grad_1 = D_grad[0][j];
								}
								else {
									grad_2 = D_grad[1][j];
								}
								GPBoost::CreatSparseBlockDiagonalMartix<sp_mat_t>(grad_1, grad_2, D_grad_all);
								CalcLogDetStochDerivCovParVecchia(dim_mode_, num_comps_total, j, SigmaI_deriv_rm, B_grad_all, D_grad_all, D_inv_plus_W_inv_diag,
									PI_Z, WI_PI_Z, d_log_det_Sigma_W_plus_I_d_cov_pars);
							}//end num_sets_re_ > 1
							SigmaI_deriv_mode = SigmaI_deriv_rm * mode_;
							explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv_mode) + d_log_det_Sigma_W_plus_I_d_cov_pars);
							cov_grad[j + igp * num_par] = explicit_derivative;
							if (grad_information_wrt_mode_non_zero_) {
								cov_grad[j + igp * num_par] -= SigmaI_plus_W_inv_d_mll_d_mode.dot(SigmaI_deriv_mode); //add implicit derivative
							}
						}//end estimate_cov_par_index[j + igp * num_par] > 0)
					}//end loop parameters j
				}//end loop num_sets_re_
			}
			//Calculate gradient wrt fixed effects
			if (grad_information_wrt_mode_non_zero_ && ((use_random_effects_indices_of_data_ && calc_F_grad) || calc_aux_par_grad ||
				(IsZeroInflatedCountRegression() && calc_F_grad))) {// a zero-inflated count regression needs diag((Sigma^-1+W)^-1) for its zeta-block log-det term whether or not use_random_effects_indices_of_data_
				if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
					vec_t ones = vec_t::Ones(dim_mode_);
					vec_t diag_WI_dummy, D_inv_plus_W_inv_dia_dummy;
					den_mat_t PI_Z_dummy, WI_PI_Z_dummy, WI_WI_plus_Sigma_inv_Z_dummy;
					CalcLogDetStochDerivModeVecchia(ones, dim_mode_, SigmaI_plus_W_inv_diag, D_inv_plus_W_inv_dia_dummy, diag_WI_dummy, PI_Z_dummy, WI_PI_Z_dummy,
						WI_WI_plus_Sigma_inv_Z_dummy, re_comps_cross_cov_cluster_i, GPU_use);
				}
				else {
					CHECK(num_sets_re_ == 1);
					//Stochastic Trace: Calculate diagonal of SigmaI_plus_W_inv for gradient of approx. marginal likelihood wrt. F
					SigmaI_plus_W_inv_diag = d_log_det_Sigma_W_plus_I_d_mode;
					SigmaI_plus_W_inv_diag.array() /= deriv_information_diag_loc_par.array();
					if (grad_information_wrt_mode_can_be_zero_for_some_points_) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)SigmaI_plus_W_inv_diag.size(); ++i) {
							if (GPBoost::IsZero<double>(deriv_information_diag_loc_par[i])) {
								SigmaI_plus_W_inv_diag[i] = 0.;//set to 0 for safety, but this is actually not needed
							}
						}
					}//end grad_information_wrt_mode_can_be_zero_for_some_points_
				}
			}
			if (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad) {
				// Stochastic (Hutchinson) estimate of diag((Sigma^-1+W)^-1) (RE/mode-scale, dimension dim_mode_), needed for the
				// second, fixed-effects-only block's gradient below. The ratio trick used just above
				// (d_log_det_Sigma_W_plus_I_d_mode / deriv_information_diag_loc_par) is not applicable here: for
				// 'gaussian_heteroscedastic' deriv_information_diag_loc_par is identically zero (the mean's Fisher information
				// exp(-log-error-variance) does not depend on the mode), and for
				// 'zero_censored_power_transformed_normal_heteroscedastic' it vanishes at every positive observation. Instead, solve
				// (Sigma^-1+W) x_k = r_k for each column r_k of the raw (Cov = I) random vectors rand_vec_trace_I_ (already
				// generated above for the log-determinant's stochastic trace estimation) using the existing,
				// preconditioner-agnostic single-vector solver (it internally performs the required push-through for
				// preconditioners where (Sigma^-1+W) is not solved for directly), then diag ~= mean_k(x_k * r_k)
				CHECK(num_sets_re_ == 1);
				den_mat_t SigmaI_plus_W_inv_RV_I(dim_mode_, num_rand_vec_trace_);
				bool has_NA_or_Inf_stoch_diag = false;
				for (int k = 0; k < num_rand_vec_trace_; ++k) {
					vec_t SigmaI_plus_W_inv_RV_I_col(dim_mode_);
					bool has_NA_or_Inf_k = false;
					Inv_SigmaI_plus_ZtWZ_Vecchia_iterative_given_PC(cg_max_num_it_, re_comps_cross_cov_cluster_i,
						rand_vec_trace_I_.col(k), SigmaI_plus_W_inv_RV_I_col, true, has_NA_or_Inf_k);
					SigmaI_plus_W_inv_RV_I.col(k) = SigmaI_plus_W_inv_RV_I_col;
					if (has_NA_or_Inf_k) {
						has_NA_or_Inf_stoch_diag = true;
					}
				}
				if (has_NA_or_Inf_stoch_diag) {
					Log::REDebug(CG_NA_OR_INF_WARNING_GRADIENT_);
				}
				SigmaI_plus_W_inv_diag_2nd_block = (SigmaI_plus_W_inv_RV_I.cwiseProduct(rand_vec_trace_I_)).rowwise().mean();
				if (likelihood_type_ == "gaussian_heteroscedastic") {
					SigmaI_plus_W_inv_diag = SigmaI_plus_W_inv_diag_2nd_block;// there is no eta-block version for this likelihood
				}
			}
			//Calculate gradient wrt additional likelihood parameters
			if (calc_aux_par_grad) {
				vec_t neg_likelihood_deriv(num_aux_pars_estim_);//derivative of the negative log-likelihood wrt additional parameters of the likelihood
				vec_t second_deriv_loc_aux_par(num_data_);//second derivative of the log-likelihood with respect to (i) the location parameter and (ii) an additional parameter of the likelihood
				vec_t deriv_information_aux_par(num_data_);//negative third derivative of the log-likelihood with respect to (i) two times the location parameter and (ii) an additional parameter of the likelihood
				vec_t d_mode_d_aux_par;
				CalcGradNegLogLikAuxPars(y_data, y_data_int, location_par_ptr, neg_likelihood_deriv.data());
				for (int ind_ap = 0; ind_ap < num_aux_pars_estim_; ++ind_ap) {
					CalcSecondDerivLogLikFirstDerivInformationAuxPar(y_data, y_data_int, location_par_ptr, ind_ap, second_deriv_loc_aux_par.data(), deriv_information_aux_par.data());
					double d_detmll_d_aux_par = 0., implicit_derivative = 0.;
					if (grad_information_wrt_mode_non_zero_) {
						bool deriv_information_loc_par_has_zero = false;
						if (grad_information_wrt_mode_can_be_zero_for_some_points_) {
							deriv_information_loc_par_has_zero = GPBoost::HasZero<double>(deriv_information_diag_loc_par.data(), (data_size_t)deriv_information_diag_loc_par.size());
						}
						if (deriv_information_loc_par_has_zero) {//deriv_information_diag_loc_par has some zeros
							if (use_random_effects_indices_of_data_) {
								vec_t Zt_deriv_information_aux_par(dim_mode_);
								CalcZtVGivenIndices(num_data_, dim_mode_, random_effects_indices_of_data_, deriv_information_aux_par.data(), Zt_deriv_information_aux_par.data(), true);
								CalcLogDetStochDerivAuxParVecchia(Zt_deriv_information_aux_par, D_inv_plus_W_inv_diag, diag_WI, PI_Z, WI_PI_Z, WI_WI_plus_Sigma_inv_Z, d_detmll_d_aux_par, re_comps_cross_cov_cluster_i);
							}
							else {
								CalcLogDetStochDerivAuxParVecchia(deriv_information_aux_par, D_inv_plus_W_inv_diag, diag_WI, PI_Z, WI_PI_Z, WI_WI_plus_Sigma_inv_Z, d_detmll_d_aux_par, re_comps_cross_cov_cluster_i);
							}
							// the log-determinant term is obtained from the stochastic trace estimator above, only the implicit derivative is summed here
							AccumulateAuxParGradTerms(deriv_information_aux_par, second_deriv_loc_aux_par, SigmaI_plus_W_inv_diag,
								SigmaI_plus_W_inv_d_mll_d_mode, false, d_detmll_d_aux_par, implicit_derivative);
						}
						else {//deriv_information_diag_loc_par is non-zero everywhere (!deriv_information_loc_par_has_zero )
							AccumulateAuxParGradTerms(deriv_information_aux_par, second_deriv_loc_aux_par, SigmaI_plus_W_inv_diag,
								SigmaI_plus_W_inv_d_mll_d_mode, true, d_detmll_d_aux_par, implicit_derivative);
						}
					}//end if grad_information_wrt_mode_non_zero_
					else {// grad_information_wrt_mode is zero
						if (use_random_effects_indices_of_data_) {
							vec_t Zt_deriv_information_aux_par(dim_mode_);
							CalcZtVGivenIndices(num_data_, dim_mode_, random_effects_indices_of_data_, deriv_information_aux_par.data(), Zt_deriv_information_aux_par.data(), true);
							CalcLogDetStochDerivAuxParVecchia(Zt_deriv_information_aux_par, D_inv_plus_W_inv_diag, diag_WI, PI_Z, WI_PI_Z, WI_WI_plus_Sigma_inv_Z, d_detmll_d_aux_par, re_comps_cross_cov_cluster_i);
						}
						else {
							CalcLogDetStochDerivAuxParVecchia(deriv_information_aux_par, D_inv_plus_W_inv_diag, diag_WI, PI_Z, WI_PI_Z, WI_WI_plus_Sigma_inv_Z, d_detmll_d_aux_par, re_comps_cross_cov_cluster_i);
						}
					}
					aux_par_grad[ind_ap] = neg_likelihood_deriv[ind_ap] + 0.5 * d_detmll_d_aux_par + implicit_derivative;
				}
				SetGradAuxParsNotEstimated(aux_par_grad);
			}//end calc_aux_par_grad
		}//end iterative
		else {//Cholesky decomposition
			// Calculate (Sigma^-1 + W)^-1
			sp_mat_t L_inv(dim_mode_, dim_mode_);
			L_inv.setIdentity();
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, L_inv, L_inv, false);
			sp_mat_t SigmaI_plus_W_inv;
			// Calculate gradient wrt covariance parameters
			bool some_cov_par_estimated = std::any_of(estimate_cov_par_index.begin(), estimate_cov_par_index.end(), [](int x) { return x > 0; });
			bool calc_cov_grad_internal = calc_cov_grad && some_cov_par_estimated;
			if (calc_cov_grad_internal) {
				double explicit_derivative;
				int num_par = (int)B_grad[0].size();
				sp_mat_t SigmaI_deriv, BgradT_Dinv_B, Bt_Dinv_Bgrad;
				for (int igp = 0; igp < num_sets_re_; ++igp) {
					sp_mat_t D_inv_B = D_inv[igp] * B[igp];
					for (int j = 0; j < num_par; ++j) {
						// Calculate SigmaI_deriv
						if (num_comps_total == 1 && j == 0) {
							SigmaI_deriv = -B[igp].transpose() * D_inv_B;//SigmaI_deriv = -SigmaI for variance parameters if there is only one GP
						}
						else if (estimate_cov_par_index[j + igp * num_par] > 0) {
							SigmaI_deriv = B_grad[igp][j].transpose() * D_inv_B;
							Bt_Dinv_Bgrad = SigmaI_deriv.transpose();
							SigmaI_deriv += Bt_Dinv_Bgrad - D_inv_B.transpose() * D_grad[igp][j] * D_inv_B;
							Bt_Dinv_Bgrad.resize(0, 0);
						}
						if (num_sets_re_ > 1) {
							CHECK(num_sets_re_ == 2);
							sp_mat_t SigmaI_deriv_1(dim_mode_per_set_re_, dim_mode_per_set_re_), SigmaI_deriv_2(dim_mode_per_set_re_, dim_mode_per_set_re_);
							if (igp == 0) {
								SigmaI_deriv_1 = SigmaI_deriv;
							}
							else {
								SigmaI_deriv_2 = SigmaI_deriv;
							}
							GPBoost::CreatSparseBlockDiagonalMartix<sp_mat_t>(SigmaI_deriv_1, SigmaI_deriv_2, SigmaI_deriv);
						}
						if (j == 0) {
							// Calculate SigmaI_plus_W_inv = L_inv.transpose() * L_inv at non-zero entries of SigmaI_deriv
								//  Note: fully calculating SigmaI_plus_W_inv = L_inv.transpose() * L_inv is very slow
							SigmaI_plus_W_inv = SigmaI_deriv;
							CalcLtLGivenSparsityPattern<sp_mat_t>(L_inv, SigmaI_plus_W_inv, true);
							if (grad_information_wrt_mode_non_zero_ && igp == 0) {
								CHECK(first_deriv_information_loc_par_caluclated_);
								if (num_sets_re_ > 1) {
									CHECK(likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random");
								}
								if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
									d_mll_d_mode = vec_t::Zero(dim_mode_);
									d_mll_d_mode.segment(dim_mode_per_set_re_, dim_mode_per_set_re_) = 0.5 * (SigmaI_plus_W_inv.diagonal().segment(0, dim_mode_per_set_re_).array() * deriv_information_diag_loc_par.array()).matrix();
								}
								else {
									d_mll_d_mode = 0.5 * (SigmaI_plus_W_inv.diagonal().array() * deriv_information_diag_loc_par.array()).matrix();
								}
								SigmaI_plus_W_inv_d_mll_d_mode = L_inv.transpose() * (L_inv * d_mll_d_mode);
							}
						}//end if j == 0
						if (estimate_cov_par_index[j + igp * num_par] > 0) {
							vec_t SigmaI_deriv_mode = SigmaI_deriv * mode_;
							explicit_derivative = 0.5 * (mode_.dot(SigmaI_deriv_mode) + (SigmaI_deriv.cwiseProduct(SigmaI_plus_W_inv)).sum());
							if (num_comps_total == 1 && j == 0) {
								explicit_derivative += 0.5 * dim_mode_per_set_re_;
							}
							else {
								explicit_derivative += 0.5 * (D_inv[igp].diagonal().array() * D_grad[igp][j].diagonal().array()).sum();
							}
							cov_grad[j + igp * num_par] = explicit_derivative;
							if (grad_information_wrt_mode_non_zero_) {
								cov_grad[j + igp * num_par] -= SigmaI_plus_W_inv_d_mll_d_mode.dot(SigmaI_deriv_mode);//add implicit derivative
							}
						}
					}//end loop over num_par
				}// end loop over num_sets_re_
			}//end calc_cov_grad_internal
			if (calc_F_grad || calc_aux_par_grad) {
				if (!calc_cov_grad_internal) {
					if (calc_aux_par_grad || grad_information_wrt_mode_non_zero_ || (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad)) {
						sp_mat_t L_inv_sqr = L_inv.cwiseProduct(L_inv);
						SigmaI_plus_W_inv_diag = L_inv_sqr.transpose() * vec_t::Ones(L_inv_sqr.rows());// diagonal of (Sigma^-1 + W) ^ -1
					SigmaI_plus_W_inv_diag_2nd_block = SigmaI_plus_W_inv_diag;
					}
					if (grad_information_wrt_mode_non_zero_) {
						if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
							d_mll_d_mode = vec_t::Zero(dim_mode_);
							d_mll_d_mode.segment(dim_mode_per_set_re_, dim_mode_per_set_re_) = (0.5 * SigmaI_plus_W_inv_diag.segment(0, dim_mode_per_set_re_).array() * deriv_information_diag_loc_par.array()).matrix();// gradient of approx. marginal likelihood wrt the mode and thus also F here
							// note: deriv_information_diag_loc_par is of length dim_mode_per_set_re_ here since only the non-zero derivatives are saved
						}
						else {
							d_mll_d_mode = (0.5 * SigmaI_plus_W_inv_diag.array() * deriv_information_diag_loc_par.array()).matrix();// gradient of approx. marginal likelihood wrt the mode and thus also F here
						}
						SigmaI_plus_W_inv_d_mll_d_mode = L_inv.transpose() * (L_inv * d_mll_d_mode);
					}
				}
				else if (calc_aux_par_grad || (use_random_effects_indices_of_data_ && grad_information_wrt_mode_non_zero_) || (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad) ||
					(IsZeroInflatedCountRegression() && grad_information_wrt_mode_non_zero_ && calc_F_grad)) {// the zeta-block log-det term of a zero-inflated count regression needs the diagonal of (Sigma^-1+W)^-1
					SigmaI_plus_W_inv_diag = SigmaI_plus_W_inv.diagonal();
				SigmaI_plus_W_inv_diag_2nd_block = SigmaI_plus_W_inv_diag;
				}
			}//end calc_F_grad || calc_aux_par_grad
			// calculate gradient wrt additional likelihood parameters
			if (calc_aux_par_grad) {
				CHECK(num_sets_re_ == 1);
				CalcAuxParGradLaplaceExactDiag(y_data, y_data_int, location_par_ptr, SigmaI_plus_W_inv_diag,
					SigmaI_plus_W_inv_d_mll_d_mode, aux_par_grad);
			}//end calc_aux_par_grad
		}//end Cholesky decomposition
		// Calculate gradient wrt fixed effects
		if (calc_F_grad) {
			if (use_random_effects_indices_of_data_) {
				fixed_effect_grad = -first_deriv_ll_data_scale_;
				if (grad_information_wrt_mode_non_zero_) {
					if (likelihood_type_ == "gaussian_heteroscedastic_fixed_and_random") {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							fixed_effect_grad[i] -= information_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
						}
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							fixed_effect_grad[i + num_data_] += 0.5 * deriv_information_diag_loc_par_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
								information_ll_data_scale_[i + num_data_] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i] + dim_mode_per_set_re_];// implicit derivative
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < num_data_; ++i) {
							fixed_effect_grad[i] += 0.5 * deriv_information_diag_loc_par_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
								information_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
						}
					}
				}
				if (HasSecondFEBlock()) {
					CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_data_scale_,
						SecondFEBlockZetaDiag(SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_diag_2nd_block),
						SigmaI_plus_W_inv_d_mll_d_mode, random_effects_indices_of_data_, true, fixed_effect_grad);
				}
			}
			else {
				fixed_effect_grad = -first_deriv_ll_;
				if (grad_information_wrt_mode_non_zero_) {
					vec_t d_mll_d_F_implicit = -(SigmaI_plus_W_inv_d_mll_d_mode.array() * information_ll_.array()).matrix();// implicit derivative
					fixed_effect_grad += d_mll_d_mode + d_mll_d_F_implicit;
				}
				if (HasSecondFEBlock()) {
					CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_,
						SecondFEBlockZetaDiag(SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_diag_2nd_block),
						SigmaI_plus_W_inv_d_mll_d_mode, nullptr, true, fixed_effect_grad);
				}
			}
		}//end calc_F_grad
	}//end CalcGradNegMargLikelihoodLaplaceApproxVecchia

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcGradNegMargLikelihoodLaplaceApproxFITC(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const std::shared_ptr<den_mat_t> sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const den_mat_t* cross_cov,
		const vec_t& fitc_resid_diag,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		bool calc_cov_grad,
		bool calc_F_grad,
		bool calc_aux_par_grad,
		double* cov_grad,
		vec_t& fixed_effect_grad,
		double* aux_par_grad,
		bool calc_mode,
		bool call_for_std_dev_coef,
		const std::vector<int>& estimate_cov_par_index,
		bool GPU_use) {
		int num_ip = (int)((*sigma_ip).rows());
		CHECK((int)((*cross_cov).rows()) == dim_mode_);
		CHECK((int)((*cross_cov).cols()) == num_ip);
		CHECK((int)fitc_resid_diag.size() == dim_mode_);
		if (calc_mode) {// Calculate mode and Cholesky factor 
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLFITC(y_data, y_data_int, fixed_effects, sigma_ip, chol_fact_sigma_ip,
				cross_cov, fitc_resid_diag, mll, GPU_use);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			if (call_for_std_dev_coef) {
				Log::REFatal(CANNOT_CALC_STDEV_ERROR_);
			}
			else {
				Log::REFatal(NA_OR_INF_ERROR_);
			}
		}
		CHECK(mode_has_been_calculated_);
		// Initialize variables
		vec_t location_par;//location parameter = mode of random effects + fixed effects
		double* location_par_ptr;
		InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
		vec_t deriv_information_diag_loc_par;//first derivative of the diagonal of the Fisher information wrt the location parameter (= usually negative third derivatives of the log-likelihood wrt the locatin parameter)
		vec_t deriv_information_diag_loc_par_data_scale;//first derivative of the diagonal of the Fisher information wrt the location parameter on the data-scale (only used if use_random_effects_indices_of_data_), the vector 'deriv_information_diag_loc_par' actually contains diag_ZtDerivInformationZ if use_random_effects_indices_of_data_
		CHECK(num_sets_re_ == 1);
		if (grad_information_wrt_mode_non_zero_) {
			CalcFirstDerivInformationLocPar(y_data, y_data_int, location_par_ptr, deriv_information_diag_loc_par, deriv_information_diag_loc_par_data_scale);
		}
		if (HasZeroValueInformationLogLik()) {
			Log::REFatal("CalcGradNegMargLikelihoodLaplaceApproxFITC: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
				"This is not permitted when using the FITC approximation and gradient-based optimization ");
		}
		vec_t WI = information_ll_.cwiseInverse();
		vec_t DW_plus_I_inv_diag, SigmaI_plus_W_inv_diag, d_mll_d_mode;
		den_mat_t L_inv_cross_cov_T_DW_plus_I_inv;
		if (grad_information_wrt_mode_non_zero_ || calc_aux_par_grad || (SecondFEBlockNeedsSigmaIPlusWInvDiag() && calc_F_grad)) {
			DW_plus_I_inv_diag = (information_ll_.array() * fitc_resid_diag.array() + 1.).matrix().cwiseInverse();
			L_inv_cross_cov_T_DW_plus_I_inv = (*cross_cov).transpose() * (DW_plus_I_inv_diag.asDiagonal());
			TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_dense_Newton_, L_inv_cross_cov_T_DW_plus_I_inv, L_inv_cross_cov_T_DW_plus_I_inv, false);
			SigmaI_plus_W_inv_diag = (L_inv_cross_cov_T_DW_plus_I_inv.cwiseProduct(L_inv_cross_cov_T_DW_plus_I_inv)).colwise().sum();// SigmaI_plus_W_inv_diag = diagonal of (Sigma^-1 + ZtWZ)^-1
			if (!calc_F_grad && !calc_aux_par_grad) {
				L_inv_cross_cov_T_DW_plus_I_inv.resize(0, 0);
			}
			SigmaI_plus_W_inv_diag += WI;
			SigmaI_plus_W_inv_diag.array() -= (DW_plus_I_inv_diag.array() * WI.array());
		}
		if (grad_information_wrt_mode_non_zero_) {
			CHECK(first_deriv_information_loc_par_caluclated_);
			d_mll_d_mode = (0.5 * SigmaI_plus_W_inv_diag.array() * deriv_information_diag_loc_par.array()).matrix();// gradient of approx. marginal likelihood wrt the mode
		}
		// Calculate gradient wrt covariance parameters
		bool some_cov_par_estimated = std::any_of(estimate_cov_par_index.begin(), estimate_cov_par_index.end(), [](int x) { return x > 0; });
		if (calc_cov_grad && some_cov_par_estimated) {
			vec_t sigma_ip_inv_cross_cov_T_SigmaI_mode = chol_fact_sigma_ip.solve((*cross_cov).transpose() * SigmaI_mode_);// sigma_ip^-1 * cross_cov^T * sigma^-1 * mode
			vec_t D_plus_WI_inv_diag = (fitc_resid_diag + WI).cwiseInverse();
			int par_count = 0;
			double explicit_derivative;
			for (int j = 0; j < (int)re_comps_ip_cluster_i.size(); ++j) {
				for (int ipar = 0; ipar < re_comps_ip_cluster_i[j]->NumCovPar(); ++ipar) {
					if (estimate_cov_par_index[par_count] > 0) {
						std::shared_ptr<den_mat_t> cross_cov_grad = re_comps_cross_cov_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.);
						den_mat_t sigma_ip_grad = *(re_comps_ip_cluster_i[j]->GetZSigmaZtGrad(ipar, true, 0.));
						den_mat_t sigma_ip_inv_sigma_ip_grad = chol_fact_sigma_ip.solve(sigma_ip_grad);
						vec_t fitc_diag_grad = vec_t::Zero(dim_mode_);
						fitc_diag_grad.array() += sigma_ip_grad.coeffRef(0, 0);
						den_mat_t sigma_ip_inv_cross_cov_T = chol_fact_sigma_ip.solve((*cross_cov).transpose());
						den_mat_t sigma_ip_grad_sigma_ip_inv_cross_cov_T = sigma_ip_grad * sigma_ip_inv_cross_cov_T;
						fitc_diag_grad -= 2 * (sigma_ip_inv_cross_cov_T.cwiseProduct((*cross_cov_grad).transpose())).colwise().sum();
						fitc_diag_grad += (sigma_ip_inv_cross_cov_T.cwiseProduct(sigma_ip_grad_sigma_ip_inv_cross_cov_T)).colwise().sum();
						// Derivative of Woodbury matrix
						den_mat_t sigma_woodbury_grad = sigma_ip_grad;
						den_mat_t cross_cov_T_fitc_diag_plus_WI_inv_cross_cov_grad = (*cross_cov).transpose() * D_plus_WI_inv_diag.asDiagonal() * (*cross_cov_grad);
						sigma_woodbury_grad += cross_cov_T_fitc_diag_plus_WI_inv_cross_cov_grad + cross_cov_T_fitc_diag_plus_WI_inv_cross_cov_grad.transpose();
						cross_cov_T_fitc_diag_plus_WI_inv_cross_cov_grad.resize(0, 0);
						vec_t v_aux_grad = D_plus_WI_inv_diag;
						v_aux_grad.array() *= v_aux_grad.array();
						v_aux_grad.array() *= fitc_diag_grad.array();
						sigma_woodbury_grad -= (*cross_cov).transpose() * v_aux_grad.asDiagonal() * (*cross_cov);
						den_mat_t sigma_woodbury_inv_sigma_woodbury_grad = chol_fact_dense_Newton_.solve(sigma_woodbury_grad);
						// Calculate explicit derivative of approx. mariginal log-likelihood
						explicit_derivative = -((*cross_cov_grad).transpose() * SigmaI_mode_).dot(sigma_ip_inv_cross_cov_T_SigmaI_mode) +
							0.5 * sigma_ip_inv_cross_cov_T_SigmaI_mode.dot(sigma_ip_grad * sigma_ip_inv_cross_cov_T_SigmaI_mode) -
							0.5 * SigmaI_mode_.dot(fitc_diag_grad.asDiagonal() * SigmaI_mode_);//derivative of mode^T Sigma^-1 mode
						explicit_derivative += 0.5 * sigma_woodbury_inv_sigma_woodbury_grad.trace() -
							0.5 * sigma_ip_inv_sigma_ip_grad.trace() +
							0.5 * fitc_diag_grad.dot(D_plus_WI_inv_diag);//derivative of log determinant
						cov_grad[par_count] = explicit_derivative;
						if (grad_information_wrt_mode_non_zero_) {
							// Calculate implicit derivative (through mode) of approx. mariginal log-likelihood
							vec_t SigmaDeriv_first_deriv_ll = (*cross_cov_grad) * (sigma_ip_inv_cross_cov_T * first_deriv_ll_);
							SigmaDeriv_first_deriv_ll += sigma_ip_inv_cross_cov_T.transpose() * ((*cross_cov_grad).transpose() * first_deriv_ll_);
							SigmaDeriv_first_deriv_ll -= sigma_ip_inv_cross_cov_T.transpose() * (sigma_ip_grad_sigma_ip_inv_cross_cov_T * first_deriv_ll_);
							SigmaDeriv_first_deriv_ll += fitc_diag_grad.asDiagonal() * first_deriv_ll_;
							vec_t rhs = (*cross_cov).transpose() * (D_plus_WI_inv_diag.asDiagonal() * SigmaDeriv_first_deriv_ll);
							vec_t vaux = chol_fact_dense_Newton_.solve(rhs);
							vec_t d_mode_d_par = WI.asDiagonal() *
								(D_plus_WI_inv_diag.asDiagonal() * SigmaDeriv_first_deriv_ll - D_plus_WI_inv_diag.asDiagonal() * ((*cross_cov) * vaux));
							cov_grad[par_count] += d_mll_d_mode.dot(d_mode_d_par);
							////for debugging
							//if (ipar == 0) {
							//  Log::REInfo("mode_[0:4] = %g, %g, %g, %g, %g ", mode_[0], mode_[1], mode_[2], mode_[3], mode_[4]);
							//  Log::REInfo("SigmaI_mode_[0:4] = %g, %g, %g, %g, %g ", SigmaI_mode_[0], SigmaI_mode_[1], SigmaI_mode_[2], SigmaI_mode_[3], SigmaI_mode_[4]);
							//  Log::REInfo("d_mll_d_mode[0:2] = %g, %g, %g ", d_mll_d_mode[0], d_mll_d_mode[1], d_mll_d_mode[2]);
							//}
							//Log::REInfo("d_mode_d_par[0:2] = %g, %g, %g ", d_mode_d_par[0], d_mode_d_par[1], d_mode_d_par[2]);
							//double ed1 = -((*cross_cov_grad).transpose() * SigmaI_mode_).dot(sigma_ip_inv_cross_cov_T_SigmaI_mode);
							//ed1 += 0.5 * sigma_ip_inv_cross_cov_T_SigmaI_mode.dot(sigma_ip_grad * sigma_ip_inv_cross_cov_T_SigmaI_mode);
							//ed1 -= 0.5 * SigmaI_mode_.dot(fitc_diag_grad.asDiagonal() * SigmaI_mode_);
							//double ed2 = 0.5 * sigma_woodbury_inv_sigma_woodbury_grad.trace() -
							//  0.5 * sigma_ip_inv_sigma_ip_grad.trace() +
							//  0.5 * fitc_diag_grad.dot(D_plus_WI_inv_diag);
							//Log::REInfo("explicit_derivative = %g (%g + %g), d_mll_d_mode.dot(d_mode_d_par) = %g, cov_grad = %g ", 
							//  explicit_derivative, ed1, ed2, d_mll_d_mode.dot(d_mode_d_par), cov_grad[par_count]);
						}//end grad_information_wrt_mode_non_zero_
					}//end estimate_cov_par_index[par_count] > 0
					par_count++;
				}//end loop over ipar
			}//end loop over j
		}//end calc_cov_grad
		// calculate gradient wrt fixed effects
		vec_t SigmaI_plus_W_inv_d_mll_d_mode;// for implicit derivative
		if (grad_information_wrt_mode_non_zero_ && (calc_F_grad || calc_aux_par_grad)) {
			// (Sigma^-1 + W)^-1 = W^-1 - W^-1*A^-1*W^-1 + W^-1*A^-1*Sigma_nm*(sigma_ip + Sigma_nm^T*A^-1*Sigma_nm)^-1*Sigma_nm^T*A^-1*W^-1
			// with A = fitc_resid_diag + W^-1, so that W^-1*A^-1 = DW_plus_I_inv_diag = 1 / (W*fitc_resid_diag + 1).
			// The middle factor must therefore be DW_plus_I_inv_diag itself (as in the diagonal above), not its inverse
			SigmaI_plus_W_inv_d_mll_d_mode = WI.asDiagonal() * d_mll_d_mode -
				DW_plus_I_inv_diag.asDiagonal() * (WI.asDiagonal() * d_mll_d_mode) +
				L_inv_cross_cov_T_DW_plus_I_inv.transpose() * (L_inv_cross_cov_T_DW_plus_I_inv * d_mll_d_mode);
		}
		if (calc_F_grad) {
			if (use_random_effects_indices_of_data_) {
				fixed_effect_grad = -first_deriv_ll_data_scale_;
				if (grad_information_wrt_mode_non_zero_) {
#pragma omp parallel for schedule(static)
					for (data_size_t i = 0; i < num_data_; ++i) {
						fixed_effect_grad[i] += 0.5 * deriv_information_diag_loc_par_data_scale[i] * SigmaI_plus_W_inv_diag[random_effects_indices_of_data_[i]] -
							information_ll_data_scale_[i] * SigmaI_plus_W_inv_d_mll_d_mode[random_effects_indices_of_data_[i]];// implicit derivative
					}
				}
				if (HasSecondFEBlock()) {
					CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_data_scale_,
						SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_d_mll_d_mode, random_effects_indices_of_data_, true, fixed_effect_grad);
				}
			}
			else {
				fixed_effect_grad = -first_deriv_ll_;
				if (grad_information_wrt_mode_non_zero_) {
					vec_t d_mll_d_F_implicit = (SigmaI_plus_W_inv_d_mll_d_mode.array() * information_ll_.array()).matrix();// implicit derivative
					fixed_effect_grad += d_mll_d_mode - d_mll_d_F_implicit;
				}
				if (HasSecondFEBlock()) {
					CalcSecondFEBlockFixedEffectGrad(y_data, y_data_int, location_par_ptr, information_ll_,
						SigmaI_plus_W_inv_diag, SigmaI_plus_W_inv_d_mll_d_mode, nullptr, true, fixed_effect_grad);
				}
			}
		}//end calc_F_grad
		// calculate gradient wrt additional likelihood parameters
		if (calc_aux_par_grad) {
			CalcAuxParGradLaplaceExactDiag(y_data, y_data_int, location_par_ptr, SigmaI_plus_W_inv_diag,
				SigmaI_plus_W_inv_d_mll_d_mode, aux_par_grad);
		}//end calc_aux_par_grad
	}//end CalcGradNegMargLikelihoodLaplaceApproxFITC

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::PredictLaplaceApproxStable(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const std::shared_ptr<T_mat>& ZSigmaZt,
		const T_mat& Cross_Cov,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool calc_mode) {
		if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * ZSigmaZt * Wsqrt) at mode
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLStable(y_data, y_data_int, fixed_effects, ZSigmaZt, mll);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		if (can_use_first_deriv_log_like_for_pred_mean_) {
			pred_mean = Cross_Cov * first_deriv_ll_;
		}
		else {
			T_mat ZSigmaZt_stable = (*ZSigmaZt);
			ZSigmaZt_stable.diagonal().array() *= JITTER_MUL;
			T_chol chol_fact_ZSigmaZt;
			bool chol_fact_pattern_analyzed = false;
			CalcChol<T_mat>(chol_fact_ZSigmaZt, ZSigmaZt_stable, chol_fact_pattern_analyzed);
			vec_t SigmaI_mode = chol_fact_ZSigmaZt.solve(mode_);
			pred_mean = Cross_Cov * SigmaI_mode;
		}
		if (calc_pred_cov || calc_pred_var) {
			vec_t Wsqrt(dim_mode_);//diagonal of matrix sqrt(ZtWZ) if use_random_effects_indices_of_data_ or sqrt(W) if !use_random_effects_indices_of_data_
			if (use_variance_correction_for_prediction_) {
				diag_information_variance_correction_for_prediction_ = true;
				vec_t location_par;//location parameter = mode of random effects + fixed effects
				double* location_par_ptr;
				InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, true);
				if (HasNegativeValueInformationLogLik()) {
					Log::REFatal("PredictLaplaceApproxStable: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
						"Predictive covariance and variance calculations in the stable Laplace approximation require the square root of W ");
				}
				Wsqrt.array() = information_ll_.array().sqrt();
				T_mat Id_plus_Wsqrt_Sigma_Wsqrt(dim_mode_, dim_mode_);
				Id_plus_Wsqrt_Sigma_Wsqrt.setIdentity();
				Id_plus_Wsqrt_Sigma_Wsqrt += (Wsqrt.asDiagonal() * (*ZSigmaZt) * Wsqrt.asDiagonal());
				CalcChol<T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Id_plus_Wsqrt_Sigma_Wsqrt, chol_fact_pattern_analyzed_);//this is the bottleneck (for large data and sparse matrices)
				diag_information_variance_correction_for_prediction_ = false;
			}
			else {
				if (HasNegativeValueInformationLogLik()) {
					Log::REFatal("PredictLaplaceApproxStable: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
						"Predictive covariance and variance calculations in the stable Laplace approximation require the square root of W ");
				}
				Wsqrt.array() = information_ll_.array().sqrt();
			}
			T_mat Maux = Wsqrt.asDiagonal() * Cross_Cov.transpose();
			TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, Maux, Maux, false);//Maux = L\(ZtWZsqrt * Cross_Cov^T)
			if (calc_pred_cov) {
				pred_cov -= (T_mat)(Maux.transpose() * Maux);
			}
			if (calc_pred_var) {
				Maux = Maux.cwiseProduct(Maux);
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					pred_var[i] -= Maux.col(i).sum();
				}
			}
		}
	}//end PredictLaplaceApproxStable

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::PredictLaplaceApproxGroupedRE(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const sp_mat_t& SigmaI,
		bool has_vecchia_gp,
		const sp_mat_t& B,
		const sp_mat_t& D_inv,
		const sp_mat_t& Bpo,
		sp_mat_t& Bp,
		const vec_t& Dp,
		bool VecchiaCondObsOnly,
		const sp_mat_t& Ztilde,
		const sp_mat_t& Sigma,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool calc_mode) {
		if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * ZSigmaZt * Wsqrt) at mode
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLGroupedRE(y_data, y_data_int, fixed_effects, SigmaI, has_vecchia_gp, B, D_inv, false, false, mll);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		const data_size_t dim_re_group = (data_size_t)SigmaI.cols();
		CHECK(SigmaI.cols() == SigmaI.rows());
		CHECK(Ztilde.cols() == dim_re_group);
		CHECK(Ztilde.rows() == pred_mean.size());
		if (calc_pred_var) {
			CHECK(Ztilde.rows() == pred_var.size());
		}
		const data_size_t dim_gp = has_vecchia_gp ? (data_size_t)B.rows() : 0;
		CHECK(dim_gp + dim_re_group == dim_mode_);
		if (has_vecchia_gp) {
			CHECK(dim_gp > 0);
			CHECK(Ztilde.rows() == Bpo.rows());
			CHECK(Bpo.cols() == dim_gp);
			pred_mean = Ztilde * mode_.segment(0, dim_re_group);
			if (VecchiaCondObsOnly) {
				pred_mean += -Bpo * mode_.segment(dim_re_group, dim_gp);
			}
			else {
				vec_t Bpo_mode = Bpo * mode_.segment(dim_re_group, dim_gp);
				pred_mean -= Bp.triangularView<Eigen::UpLoType::UnitLower>().solve(Bpo_mode);
			}
		}
		else {
			CHECK(dim_gp == 0);
			pred_mean = Ztilde * mode_;
			//pred_mean = Ztilde * (Sigma * (Zt * first_deriv_ll_));//equivalent version
		}
		if (calc_pred_cov || calc_pred_var) {
			if (use_variance_correction_for_prediction_) {
				Log::REFatal("The variance correction is not yet implemented when having multiple grouped random effects ");
			}
			if (matrix_inversion_method_ == "iterative") {
				if (calc_pred_var) {
					int n_pred = (int)pred_mean.size();
					vec_t pred_var_global = vec_t::Zero(n_pred);
					//Variance reduction
					sp_mat_rm_t Ztilde_P_sqrt_invt_rm;
					vec_t varred_global, c_cov, c_var;
					if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
						varred_global = vec_t::Zero(n_pred);
						c_cov = vec_t::Zero(n_pred);
						c_var = vec_t::Zero(n_pred);
						//Calculate P^(-0.5) explicitly
						sp_mat_rm_t Identity_rm(dim_mode_, dim_re_group);
						std::vector<Triplet_t> triplets(dim_re_group);
						#pragma omp parallel for schedule(static)
						for (data_size_t i = 0; i < dim_re_group; ++i) {
							triplets[i] = Triplet_t(i, i, 1.);
						}
						Identity_rm.setFromTriplets(triplets.begin(), triplets.end());
						sp_mat_rm_t P_sqrt_invt_rm;
						if (cg_preconditioner_type_ == "incomplete_cholesky") {
							TriangularSolve<sp_mat_rm_t, sp_mat_rm_t, sp_mat_rm_t>(L_SigmaI_plus_ZtWZ_rm_, Identity_rm, P_sqrt_invt_rm, true);
						}
						else {
							TriangularSolve<sp_mat_rm_t, sp_mat_rm_t, sp_mat_rm_t>(P_SSOR_L_D_sqrt_inv_rm_, Identity_rm, P_sqrt_invt_rm, true);
						}
						//Z_po P^(-T/2)
						if (has_vecchia_gp) {
							Ztilde_P_sqrt_invt_rm = Ztilde * P_sqrt_invt_rm.topRows(dim_re_group);
						}
						else {
							Ztilde_P_sqrt_invt_rm = Ztilde * P_sqrt_invt_rm;
						}					
					}
					int num_threads;
#ifdef _OPENMP
					num_threads = omp_get_max_threads();
#else
					num_threads = 1;
#endif
					std::uniform_int_distribution<> unif(0, 2147483646);
					std::vector<RNG_t> parallel_rngs;
					for (int ig = 0; ig < num_threads; ++ig) {
						int seed_local = unif(cg_generator_);
						parallel_rngs.push_back(RNG_t(seed_local));
					}
					bool na_inf_flag_1 = false;
#pragma omp parallel
					{
						int thread_nb;
#ifdef _OPENMP
						thread_nb = omp_get_thread_num();
#else
						thread_nb = 0;
#endif
						RNG_t rng_local = parallel_rngs[thread_nb];
						vec_t pred_var_private = vec_t::Zero(n_pred);
						vec_t varred_private;
						vec_t c_cov_private;
						vec_t c_var_private;
						//Variance reduction
						if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
							varred_private = vec_t::Zero(n_pred);
							c_cov_private = vec_t::Zero(n_pred);
							c_var_private = vec_t::Zero(n_pred);
						}
#pragma omp for reduction(||:na_inf_flag_1)
						for (int i = 0; i < nsim_var_pred_; ++i) {
							//RV - Rademacher
							std::uniform_real_distribution<double> udist(0.0, 1.0);
							vec_t rand_vec_init(n_pred);
							double u;
							for (int j = 0; j < n_pred; j++) {
								u = udist(rng_local);
								if (u > 0.5) {
									rand_vec_init(j) = 1.;
								}
								else {
									rand_vec_init(j) = -1.;
								}
							}
							//Z_po^T RV
							vec_t Z_tilde_t_RV = Ztilde.transpose() * rand_vec_init;
							if (has_vecchia_gp) {
								Z_tilde_t_RV.conservativeResize(dim_re_group + dim_gp);
								Z_tilde_t_RV.tail(dim_gp).setZero();
							}
							//Part 2: (Sigma^(-1) + Z^T W Z)^(-1) Z_po^T RV
							vec_t MInv_Ztilde_t_RV(dim_mode_);
							bool has_NA_or_Inf = false;
							int num_cg_steps_dummy;
							CGRandomEffectsVec(SigmaI_plus_ZtWZ_rm_, Z_tilde_t_RV, MInv_Ztilde_t_RV, has_NA_or_Inf,
								cg_max_num_it_, cg_delta_conv_pred_, true, ZERO_RHS_CG_THRESHOLD, true, cg_preconditioner_type_,
								L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps_dummy);
							if (has_vecchia_gp) {
								MInv_Ztilde_t_RV.conservativeResize(dim_re_group);
							}
							if (has_NA_or_Inf) {
								na_inf_flag_1 = true;
							}
							//Part 2: Z_po (Sigma^(-1) + Z^T W Z)^(-1) Z_po^T RV
							vec_t rand_vec_final = Ztilde * MInv_Ztilde_t_RV;
							vec_t pred_var_iter = rand_vec_final.cwiseProduct(rand_vec_init);
							pred_var_private += pred_var_iter;
							//Variance reduction
							if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
								//Stochastic: Z_po P^(-0.5T) P^(-0.5) Z_po^T RV
								vec_t P_sqrt_inv_Ztilde_t_RV = Ztilde_P_sqrt_invt_rm.transpose() * rand_vec_init;
								vec_t rand_vec_varred = Ztilde_P_sqrt_invt_rm * P_sqrt_inv_Ztilde_t_RV;
								vec_t varred_iter = rand_vec_varred.cwiseProduct(rand_vec_init);
								varred_private += varred_iter;
								c_cov_private += varred_iter.cwiseProduct(pred_var_iter);
								c_var_private += varred_iter.cwiseProduct(varred_iter);
							}
						} //end for loop
#pragma omp critical
						{
							pred_var_global += pred_var_private;
							//Variance reduction
							if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
								varred_global += varred_private;
								c_cov += c_cov_private;
								c_var += c_var_private;
							}
						}
					} //end #pragma omp parallel
					if (na_inf_flag_1) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
					pred_var_global /= nsim_var_pred_;
					pred_var += pred_var_global;
					//Variance reduction
					if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
						varred_global /= nsim_var_pred_;
						c_cov /= nsim_var_pred_;
						c_var /= nsim_var_pred_;
						//Deterministic: diag(Z_po P^(-0.5T) P^(-0.5) Z_po^T)
						vec_t varred_determ = Ztilde_P_sqrt_invt_rm.cwiseProduct(Ztilde_P_sqrt_invt_rm) * vec_t::Ones(dim_re_group);
						//optimal c
						c_cov -= varred_global.cwiseProduct(pred_var_global);
						c_var -= varred_global.cwiseProduct(varred_global);
						vec_t c_opt = c_cov.array() / c_var.array();
#pragma omp parallel for schedule(static)   
						for (int i = 0; i < c_opt.size(); ++i) {
							if (c_var.coeffRef(i) == 0) {
								c_opt[i] = 1;
							}
						}
						pred_var += c_opt.cwiseProduct(varred_determ - varred_global);
					}
					if (has_vecchia_gp) {//Add GP part
						vec_t pred_var_gp = vec_t::Zero(n_pred);
						sp_mat_t Bp_inv_Dp, Bp_inv, Bp_inv_Bpo;//Bp^(-1) * Bpo
						if (VecchiaCondObsOnly) {
							Bp_inv_Bpo = Bpo; //Bp = Id
						}
						else {
							Bp_inv = sp_mat_t(Bp.rows(), Bp.cols());
							Bp_inv.setIdentity();
							TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
							Bp_inv_Bpo = Bp_inv * Bpo;
							Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
						}
						if (HasNegativeValueInformationLogLik()) {
							Log::REFatal("PredictLaplaceApproxVecchia: Negative values found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
								"Cannot have negative values when using 'iterative' methods for predictive variances in Vecchia-Laplace approximations ");
						}
						vec_t W_diag_sqrt = information_ll_.cwiseSqrt();
						CHECK(W_diag_sqrt.size() == num_data_);
						sp_mat_t B_t_D_inv_sqrt = B.transpose() * (D_inv.cwiseSqrt());
						sp_mat_t SigmaI_sqrt = SigmaI.cwiseSqrt();
						GPBoost::MakeBlockDiag_D_B<sp_mat_t>(SigmaI_sqrt, B_t_D_inv_sqrt, B_t_D_inv_sqrt);
						CHECK(B_t_D_inv_sqrt.cols() == dim_mode_);
						bool na_inf_flag_2 = false;
#pragma omp parallel
						{
							int thread_nb;
#ifdef _OPENMP
							thread_nb = omp_get_thread_num();
#else
							thread_nb = 0;
#endif
							RNG_t rng_local = parallel_rngs[thread_nb];
							vec_t pred_var_private = vec_t::Zero(n_pred);
#pragma omp for reduction(||:na_inf_flag_2)
							for (int i = 0; i < nsim_var_pred_; ++i) {
								//z_i ~ N(0,I)
								std::normal_distribution<double> ndist(0.0, 1.0);
								vec_t rand_vec_pred_I_1(dim_mode_), rand_vec_pred_I_2(num_data_);
								for (int j = 0; j < dim_mode_; j++) {
									rand_vec_pred_I_1(j) = ndist(rng_local);
								}
								for (int j = 0; j < num_data_; j++) {
									rand_vec_pred_I_2(j) = ndist(rng_local);
								}
								//z_i ~ N(0,(Sigma^{-1} + W))
								vec_t rand_vec_pred_SigmaI_plus_W = B_t_D_inv_sqrt * rand_vec_pred_I_1 + (*Zt_) * (W_diag_sqrt.cwiseProduct(rand_vec_pred_I_2));
								vec_t rand_vec_pred_SigmaI_plus_W_inv(dim_mode_);
								//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
								bool has_NA_or_Inf = false;
								int num_cg_steps_dummy;
								CGRandomEffectsVec(SigmaI_plus_ZtWZ_rm_, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf,
									cg_max_num_it_, cg_delta_conv_pred_, true, ZERO_RHS_CG_THRESHOLD, true, cg_preconditioner_type_,
									L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps_dummy);
								rand_vec_pred_SigmaI_plus_W_inv = rand_vec_pred_SigmaI_plus_W_inv.tail(dim_gp).eval();
								if (has_NA_or_Inf) {
									na_inf_flag_2 = true;
								}
								//z_i ~ N(0, Bp^{-1} Bpo (Sigma^{-1} + W)^{-1} Bpo^T Bp^{-1})
								vec_t rand_vec_pred = Bp_inv_Bpo * rand_vec_pred_SigmaI_plus_W_inv;
								pred_var_private += rand_vec_pred.cwiseProduct(rand_vec_pred);
							}//end for loop
#pragma omp critical
							{
								pred_var_gp += pred_var_private;
							}
						} // end #pragma omp parallel
						if (na_inf_flag_2) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
						pred_var_gp /= nsim_var_pred_;
						if (VecchiaCondObsOnly) {
							pred_var_gp += Dp;
						}
						else {
							pred_var_gp += Bp_inv_Dp.cwiseProduct(Bp_inv) * vec_t::Ones(n_pred);
						}
						pred_var += pred_var_gp;
					}//end has_vecchia_gp adding GP variance
				} //end calc_pred_var
				else if (calc_pred_cov) {
					if (has_vecchia_gp) {
						Log::REFatal("Predictive covariances are not implemented for grouped random effects and a Vecchia-approximated GP ");
					}
					int n_pred = (int)pred_mean.size();
					den_mat_t pred_cov_global = den_mat_t::Zero(n_pred, n_pred);
					vec_t SigmaI_diag_sqrt = SigmaI.diagonal().cwiseSqrt();
					sp_mat_rm_t Zt_W_sqrt_rm = sp_mat_rm_t((*Zt_) * information_ll_.cwiseSqrt().asDiagonal());
					if (!cg_generator_seeded_) {
						cg_generator_ = RNG_t(seed_rand_vec_trace_);
						cg_generator_seeded_ = true;
					}
					int num_threads;
#ifdef _OPENMP
					num_threads = omp_get_max_threads();
#else
					num_threads = 1;
#endif
					std::uniform_int_distribution<> unif(0, 2147483646);
					std::vector<RNG_t> parallel_rngs;
					for (int ig = 0; ig < num_threads; ++ig) {
						int seed_local = unif(cg_generator_);
						parallel_rngs.push_back(RNG_t(seed_local));
					}
					bool na_inf_flag_3 = false;
#pragma omp parallel
					{
						int thread_nb;
#ifdef _OPENMP
						thread_nb = omp_get_thread_num();
#else
						thread_nb = 0;
#endif
						RNG_t rng_local = parallel_rngs[thread_nb];
						den_mat_t pred_cov_private = den_mat_t::Zero(n_pred, n_pred);
#pragma omp for reduction(||:na_inf_flag_3)
						for (int i = 0; i < nsim_var_pred_; ++i) {
							//z_i ~ N(0,I)
							std::normal_distribution<double> ndist(0.0, 1.0);
							vec_t rand_vec_pred_I_1(dim_mode_), rand_vec_pred_I_2(num_data_);
							for (int j = 0; j < dim_mode_; j++) {
								rand_vec_pred_I_1(j) = ndist(rng_local);
							}
							for (int j = 0; j < num_data_; j++) {
								rand_vec_pred_I_2(j) = ndist(rng_local);
							}
							//z_i ~ N(0,(Sigma^(-1) + Z^T W Z))
							vec_t rand_vec_pred_SigmaI_plus_ZtWZ = SigmaI_diag_sqrt.asDiagonal() * rand_vec_pred_I_1 + Zt_W_sqrt_rm * rand_vec_pred_I_2;
							vec_t rand_vec_pred_SigmaI_plus_ZtWZ_inv(dim_mode_);
							//z_i ~ N(0,(Sigma^(-1) + Z^T W Z)^(-1))
							bool has_NA_or_Inf = false;
							int num_cg_steps_dummy;
							CGRandomEffectsVec(SigmaI_plus_ZtWZ_rm_, rand_vec_pred_SigmaI_plus_ZtWZ, rand_vec_pred_SigmaI_plus_ZtWZ_inv, has_NA_or_Inf, cg_max_num_it_, cg_delta_conv_pred_,
								true, ZERO_RHS_CG_THRESHOLD, true, cg_preconditioner_type_, L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps_dummy);
							if (has_NA_or_Inf) {
								na_inf_flag_3 = true;
							}
							//z_i ~ N(0, Z_p (Sigma^(-1) + Z^T W Z)^(-1) Z_p^T)
							vec_t rand_vec_pred = Ztilde * rand_vec_pred_SigmaI_plus_ZtWZ_inv;
							pred_cov_private += rand_vec_pred * rand_vec_pred.transpose();
						} //end for loop
#pragma omp critical
						{
							pred_cov_global += pred_cov_private;
						}
					} //end #pragma omp parallel
					if (na_inf_flag_3) { Log::REFatal("There was Nan or Inf value generated in the Conjugate Gradient Method!"); }
					pred_cov_global /= nsim_var_pred_;
					T_mat pred_cov_T_mat;
					ConvertTo_T_mat_FromDense<T_mat>(pred_cov_global, pred_cov_T_mat);
					pred_cov -= (T_mat)(Ztilde * Sigma * Ztilde.transpose()); //TODO: create and call AddPredCovMatrices only for new groups and remove this line.
					pred_cov += pred_cov_T_mat;
				} //end calc_pred_cov
			}
			else if (matrix_inversion_method_ == "cholesky") { //begin cholesky
				if (has_vecchia_gp) {
					if (calc_pred_cov) {
						Log::REFatal("Predictive covariances are not implemented for grouped random effects and a Vecchia-approximated GP ");
						// Note: the code below is correct, but the corresponding code in re_model_template would have to be changed to add only uconditional covariancs
						//		for new groups and not for all groups (see re_comp->AddPredCovMatrices)
					}
					// Grouped random effects part:  pred_cov = Z_pp * Σ_p * Z_pp^T + Z_po * (SigmaI_plus_ZtWZ)^-1 * Z_po^T, Z_po = Ztilde
					sp_mat_t SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde = Ztilde.transpose();
					sp_mat_t Mfull_group(dim_mode_, dim_re_group);
					std::vector<Triplet_t> triplets;
					triplets.reserve((size_t)SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde.nonZeros());
					for (int k = 0; k < SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde.outerSize(); ++k) {
						for (sp_mat_t::InnerIterator it(SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde, k); it; ++it) {
							triplets.emplace_back(it.row(), it.col(), it.value());
						}
					}
					Mfull_group.setFromTriplets(triplets.begin(), triplets.end());
					TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, Mfull_group, Mfull_group, false);
					SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde = Mfull_group.topRows((int)dim_re_group);
					Mfull_group.resize(0, 0);
					//Alternative approach where SigmaI_plus_ZtWZ_I_group_cols is first calculated
//						sp_mat_t SigmaI_plus_ZtWZ_I_group_cols(dim_mode_, dim_re_group);
//						std::vector<Triplet_t> triplets(dim_re_group);
//#pragma omp parallel for schedule(static)
//						for (data_size_t i = 0; i < dim_re_group; ++i) {
//							triplets[i] = Triplet_t(i, i, 1.);
//						}
//						SigmaI_plus_ZtWZ_I_group_cols.setFromTriplets(triplets.begin(), triplets.end());//dimension dim_mode_ (=dim_re_group + dim_gp) x dim_re_group with identity on the upper part
//						TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, SigmaI_plus_ZtWZ_I_group_cols, SigmaI_plus_ZtWZ_I_group_cols, false);
//						sp_mat_t SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde = SigmaI_plus_ZtWZ_I_group_cols.topRows((int)dim_re_group) * Ztilde.transpose();
//						SigmaI_plus_ZtWZ_I_group_cols.resize(0, 0);
					if (calc_pred_cov) {
						pred_cov += (T_mat)(SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde.transpose() * SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde);
					}
					if (calc_pred_var) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)pred_mean.size(); ++i) {
							pred_var[i] += (SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde.col(i)).dot(SigmaI_plus_ZtWZ_I_group_sqrt_Ztilde.col(i));
						}
					}
					// GP part
					sp_mat_t SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT; //SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT = L\(Bpo^T * Bp^-1), L = Chol(Sigma^-1 + W)
					sp_mat_t Bp_inv, Bp_inv_Dp;
					if (VecchiaCondObsOnly) {
						SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT = Bpo.transpose();//Bp = Id
					}
					else {
						Bp_inv = sp_mat_t(Bp.rows(), Bp.cols());
						Bp_inv.setIdentity();
						TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
						SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT = Bpo.transpose() * Bp_inv.transpose();
						Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
					}
					sp_mat_t Mfull_gp(dim_mode_, dim_gp);
					std::vector<Triplet_t> trips;
					trips.reserve((size_t)SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT.nonZeros());
					for (int k = 0; k < SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT.outerSize(); ++k) {
						for (sp_mat_t::InnerIterator it(SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT, k); it; ++it) {
							trips.emplace_back(it.row() + (int)dim_re_group, it.col(), it.value());
						}
					}
					Mfull_gp.setFromTriplets(trips.begin(), trips.end());
					TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, Mfull_gp, Mfull_gp, false);
					SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT = Mfull_gp.bottomRows((int)dim_gp);
					Mfull_gp.resize(0, 0);
					if (calc_pred_cov) {
						if (VecchiaCondObsOnly) {
							pred_cov += (T_mat)(SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT.transpose() * SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT);
							pred_cov.diagonal().array() += Dp.array();
						}
						else {
							pred_cov += (T_mat)(Bp_inv_Dp * Bp_inv.transpose() + SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT.transpose() * SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT);
						}
					}//end calc_pred_cov
					if (calc_pred_var) {
						SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT = SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT.cwiseProduct(SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT);
						if (VecchiaCondObsOnly) {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)pred_mean.size(); ++i) {
								pred_var[i] += Dp[i] + SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT.col(i).sum();
							}
						}
						else {
#pragma omp parallel for schedule(static)
							for (int i = 0; i < (int)pred_mean.size(); ++i) {
								pred_var[i] += (Bp_inv_Dp.row(i)).dot(Bp_inv.row(i)) + SigmaI_plus_ZtWZ_I_gp_sqrt_BpoT_BpInvT.col(i).sum();
							}
						}
					}//end calc_pred_var					
				}//end has_vecchia_gp
				else {//!has_vecchia_gp
					//VERSION 1: pred_cov = Z_po * Σ * Z_po^T + Z_pp * Σ_p * Z_pp^T - Z_po * Σ * Z_po^T * (SigmaI_plus_ZtWZ)^-1 * Z * Σ * Z_po^T, Z_po = Ztilde
					// This VERSION 1 seems slightly faster than VERSION 2 below for predictive variances (two randomly crossed REs, m = 1000, n = 10 * m) (24.02.2026)
					sp_mat_t SigmaI_plus_ZtWZ_I(dim_re_group, dim_re_group);
					SigmaI_plus_ZtWZ_I.setIdentity();
					TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, SigmaI_plus_ZtWZ_I, SigmaI_plus_ZtWZ_I, false);
					TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, SigmaI_plus_ZtWZ_I, SigmaI_plus_ZtWZ_I, true);
					sp_mat_t Sigma_Zt_W_Z_SigmaI_plus_ZtWZ_I = (Sigma * ((*Zt_) * information_ll_.asDiagonal() * (*Zt_).transpose())) * SigmaI_plus_ZtWZ_I;
					if (calc_pred_cov) {
						pred_cov -= (T_mat)(Ztilde * Sigma_Zt_W_Z_SigmaI_plus_ZtWZ_I * Ztilde.transpose());
					}
					if (calc_pred_var) {
						sp_mat_t Maux = Ztilde;
						CalcAtimesBGivenSparsityPattern<sp_mat_t>(Ztilde, Sigma_Zt_W_Z_SigmaI_plus_ZtWZ_I, Maux);
#pragma omp parallel for schedule(static)
						for (int i = 0; i < (int)pred_mean.size(); ++i) {
							pred_var[i] -= (Ztilde.row(i)).dot(Maux.row(i));
						}
					}
					//					//VERSION 2: pred_cov = Z_pp * Σ_p * Z_pp^T + Z_po * (SigmaI_plus_ZtWZ)^-1 * Z_po^T, Z_po = Ztilde
					//					//sp_mat_t SigmaI_plus_ZtWZ_I_Ztilde(dim_mode_, dim_re_group);
					//					sp_mat_t SigmaI_plus_ZtWZ_Isqrt_Ztilde = Ztilde.transpose();
					//					TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, SigmaI_plus_ZtWZ_Isqrt_Ztilde, SigmaI_plus_ZtWZ_Isqrt_Ztilde, false);
					//					if (calc_pred_cov) {
					//						pred_cov += (T_mat)(SigmaI_plus_ZtWZ_Isqrt_Ztilde.transpose() * SigmaI_plus_ZtWZ_Isqrt_Ztilde);
					//					}
					//					if (calc_pred_var) {
					//#pragma omp parallel for schedule(static)
					//						for (int i = 0; i < (int)pred_mean.size(); ++i) {
					//							pred_var[i] += (SigmaI_plus_ZtWZ_Isqrt_Ztilde.col(i)).dot(SigmaI_plus_ZtWZ_Isqrt_Ztilde.col(i));
					//						}
					//					}
										//Old code for VERSION 1(not used anymore)
						//              // calculate Maux = L\(Z^T * information_ll_.asDiagonal() * Cross_Cov^T)
						//              sp_mat_t Cross_Cov = Ztilde * Sigma * (*Zt_);
						//              sp_mat_t Maux = (*Zt_) * information_ll_.asDiagonal() * Cross_Cov.transpose();
						//              TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, Maux, Maux, false);
						//              if (calc_pred_cov) {
						//                  pred_cov += (T_mat)(Maux.transpose() * Maux);
						//                  pred_cov -= (T_mat)(Cross_Cov * information_ll_.asDiagonal() * Cross_Cov.transpose());
						//              }
						//              if (calc_pred_var) {
						//                  sp_mat_t Maux3 = Cross_Cov.cwiseProduct(Cross_Cov * information_ll_.asDiagonal());
						//                  Maux = Maux.cwiseProduct(Maux);
						//#pragma omp parallel for schedule(static)
						//                  for (int i = 0; i < (int)pred_mean.size(); ++i) {
						//                      pred_var[i] += Maux.col(i).sum() - Maux3.row(i).sum();
						//                  }
						//              }
				}//end !has_vecchia_gp			
			} //end cholesky
		}//end calc_pred_cov || calc_pred_var
	}//end PredictLaplaceApproxGroupedRE

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::PredictLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const double sigma2,
		const data_size_t* const random_effects_indices_of_pred,
		const data_size_t num_data_pred,
		const T_mat& Cross_Cov,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool calc_mode) {
		if (calc_mode) {// Calculate mode and Cholesky factor of B = (Id + Wsqrt * ZSigmaZt * Wsqrt) at mode
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLOnlyOneGroupedRECalculationsOnREScale(y_data, y_data_int, fixed_effects, sigma2, mll);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		pred_mean = vec_t::Zero(num_data_pred);
		if (!iid_model_) {
#pragma omp parallel for schedule(static)
			for (int i = 0; i < (int)pred_mean.size(); ++i) {
				if (random_effects_indices_of_pred[i] >= 0) {
					pred_mean[i] = mode_[random_effects_indices_of_pred[i]];
				}
			}
			if (calc_pred_cov || calc_pred_var) {
				if (use_variance_correction_for_prediction_) {
					diag_information_variance_correction_for_prediction_ = true;
					vec_t location_par(dim_location_par_);//location parameter = mode of random effects + fixed effects (+ possibly additional fixed-effects-only blocks)
					double* location_par_ptr_dummy;//not used
					UpdateLocationParNewMode(mode_, fixed_effects, location_par, &location_par_ptr_dummy);
					CalcInformationLogLik(y_data, y_data_int, location_par.data(), true);
					diag_SigmaI_plus_ZtWZ_ = (information_ll_.array() + 1. / sigma2).matrix();
					diag_information_variance_correction_for_prediction_ = false;
				}
				vec_t minus_diag_Sigma_plus_ZtWZI_inv(dim_mode_);
				minus_diag_Sigma_plus_ZtWZI_inv.array() = 1. / diag_SigmaI_plus_ZtWZ_.array();
				minus_diag_Sigma_plus_ZtWZI_inv.array() /= sigma2;
				minus_diag_Sigma_plus_ZtWZI_inv.array() -= 1.;
				minus_diag_Sigma_plus_ZtWZI_inv.array() /= sigma2;
				if (calc_pred_cov) {
					T_mat Maux = Cross_Cov * minus_diag_Sigma_plus_ZtWZI_inv.asDiagonal() * Cross_Cov.transpose();
					pred_cov += Maux;
				}
				if (calc_pred_var) {
					double sigma4 = sigma2 * sigma2;
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						if (random_effects_indices_of_pred[i] >= 0) {
							pred_var[i] += sigma4 * minus_diag_Sigma_plus_ZtWZI_inv[random_effects_indices_of_pred[i]];
						}
					}
				}
			}//end calc_pred_cov || calc_pred_var
		}//end !iid_model_)
	}//end PredictLaplaceApproxOnlyOneGroupedRECalculationsOnREScale

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::PredictLaplaceApproxFSVA(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const sp_mat_t& B,
		const sp_mat_t& D_inv,
		const sp_mat_t& Bpo,
		sp_mat_t& Bp,
		const vec_t& Dp,
		const std::shared_ptr<den_mat_t> sigma_ip,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_preconditioner_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_ip_preconditioner,
		const den_mat_t& sigma_woodbury,
		const chol_den_mat_t& chol_fact_sigma_woodbury,
		const den_mat_t& chol_ip_cross_cov,
		const den_mat_t& chol_ip_cross_cov_preconditioner,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const den_mat_t& cross_cov_pred_ip,
		const den_mat_t& Bt_D_inv_B_cross_cov,
		const den_mat_t& D_inv_B_cross_cov,
		bool sample_posterior,
		int num_post_samples,
		den_mat_t& post_samples_id,
		vec_t& pred_mean,
		den_mat_t& pred_cov,
		vec_t& pred_var,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool calc_mode,
		bool CondObsOnly,
		bool GPU_use) {
		const den_mat_t* cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
		if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLFSVA(y_data, y_data_int, fixed_effects, *sigma_ip, chol_fact_sigma_ip,
				chol_fact_sigma_woodbury, chol_ip_cross_cov, re_comps_cross_cov_cluster_i, sigma_woodbury, B, D_inv, Bt_D_inv_B_cross_cov,
				D_inv_B_cross_cov, false, false, mll, re_comps_ip_preconditioner_cluster_i, re_comps_cross_cov_preconditioner_cluster_i,
				chol_ip_cross_cov_preconditioner, chol_fact_sigma_ip_preconditioner, GPU_use);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		// Compute predictive mean
		den_mat_t sigma_ip_stable = *sigma_ip;
		sigma_ip_stable.diagonal().array() *= JITTER_MULT_IP_FITC_FSA;
		CHECK(mode_has_been_calculated_);
		int num_pred = (int)Bp.cols();
		CHECK((int)Dp.size() == num_pred);
		sp_mat_t Bt_D_inv = B.transpose() * D_inv;
		vec_t sigma_inv_mode = mode_ - (*cross_cov) * chol_fact_sigma_woodbury.solve(Bt_D_inv_B_cross_cov.transpose() * mode_);
		if (CondObsOnly) {
			pred_mean = -Bpo * sigma_inv_mode;
		}
		else {
			vec_t Bpo_mode = Bpo * sigma_inv_mode;
			pred_mean = -Bp.triangularView<Eigen::UpLoType::UnitLower>().solve(Bpo_mode);
		}
		pred_mean += cross_cov_pred_ip * chol_fact_sigma_ip.solve(Bt_D_inv_B_cross_cov.transpose() * sigma_inv_mode);
		// Compute predictive (co-)variances
		if (calc_pred_cov || calc_pred_var || sample_posterior) {
			if (use_variance_correction_for_prediction_) {
				Log::REFatal("PredictLaplaceApproxFSVA: The variance correction is not yet implemented ");
			}
			den_mat_t chol_ip_cross_cov_pred;
			den_mat_t sigma_ip_inv_sigma_cross_cov_pred = chol_fact_sigma_ip.solve(cross_cov_pred_ip.transpose());
			//TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_ip,
			//	cross_cov_pred_ip.transpose(), chol_ip_cross_cov_pred, false);
			GPBoost::solve_lower_triangular(chol_fact_sigma_ip, cross_cov_pred_ip.transpose(), chol_ip_cross_cov_pred, GPU_use);
			sp_mat_rm_t Bpo_rm = sp_mat_rm_t(Bpo);
			sp_mat_t Bp_inv_Dp;
			sp_mat_t Bp_inv(Bp.rows(), Bp.cols());
			//Version Simulation
			if (matrix_inversion_method_ == "iterative") {
				if (!sample_posterior) {
					num_post_samples = 0;
				}
				if (!calc_pred_cov && !calc_pred_var) {
					nsim_var_pred_ = 0;
				}
				den_mat_t cross_cov_PP_Vecchia = chol_ip_cross_cov_pred.transpose() * (chol_ip_cross_cov * Bt_D_inv_B_cross_cov);
				den_mat_t cross_cov_pred_obs_pred_inv;
				den_mat_t B_po_cross_cov(pred_mean.size(), (*cross_cov).cols());
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < (*cross_cov).cols(); ++i) {
					B_po_cross_cov.col(i) = Bpo_rm * (*cross_cov).col(i);
				}
				den_mat_t cross_cov_PP_Vecchia_woodbury = chol_fact_sigma_woodbury.solve(cross_cov_PP_Vecchia.transpose());
				sp_mat_rm_t Bp_inv_Dp_rm, Bp_inv_rm, Bp_rm, Bp_inv_Bpo_rm;//Bp^(-1) * Bpo 
				if (CondObsOnly) {
					Bp_inv_Bpo_rm = Bpo_rm; //Bp = Id
				}
				else {
					Bp_rm = sp_mat_rm_t(Bp);
					Bp_inv_rm = sp_mat_rm_t(Bp_rm.rows(), Bp_rm.cols());
					Bp_inv_rm.setIdentity();
					TriangularSolve<sp_mat_rm_t, sp_mat_rm_t, sp_mat_rm_t>(Bp_rm, Bp_inv_rm, Bp_inv_rm, false);
					Bp_inv_Bpo_rm = Bp_inv_rm * Bpo_rm;
				}
				if (calc_pred_cov) {
					pred_cov = den_mat_t::Zero(num_pred, num_pred);
				}
				vec_t pred_var_prec;
				vec_t pred_var_prec_sq;
				vec_t pred_var_prec_diff;
				vec_t pred_var_prec_prod;
				if (calc_pred_var) {
					pred_var = vec_t::Zero(num_pred);
					pred_var_prec = vec_t::Zero(num_pred);
					pred_var_prec_sq = vec_t::Zero(num_pred);
					pred_var_prec_diff = vec_t::Zero(num_pred);
					pred_var_prec_prod = vec_t::Zero(num_pred);
				}
				if (HasNegativeValueInformationLogLik()) {
					Log::REFatal("PredictLaplaceApproxFSVA: Negative values found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
						"Cannot have negative values when using 'iterative' methods for predictive variances in Vecchia-Laplace approximations ");
				}
				if (HasZeroValueInformationLogLik()) {
					Log::REFatal("PredictLaplaceApproxFSVA: 0's found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
						"Predictive variance calculations with iterative methods require W to be invertible. Try using the Cholesky decomposition ");
				}
				const den_mat_t* cross_cov_preconditioner;
				vec_t information_ll_inv;
				information_ll_inv.resize(dim_mode_);
				information_ll_inv.array() = information_ll_.array().inverse();
				if (cg_preconditioner_type_ == "fitc") {
					cross_cov_preconditioner = re_comps_cross_cov_preconditioner_cluster_i[0]->GetSigmaPtr();
				}
				else {
					cross_cov_preconditioner = nullptr;
				}
				vec_t W_diag_sqrt = information_ll_.cwiseSqrt();
				sp_mat_rm_t B_t_D_inv_sqrt_rm = B_rm_.transpose() * D_inv_rm_.cwiseSqrt();
				if (CondObsOnly) {
					cross_cov_pred_obs_pred_inv = B_po_cross_cov;
				}
				else {
					TriangularSolve<sp_mat_t, den_mat_t, den_mat_t>(Bp, B_po_cross_cov, cross_cov_pred_obs_pred_inv, false);
				}
				den_mat_t cross_cov_pred_obs_pred_inv_woodbury = chol_fact_sigma_woodbury.solve(cross_cov_pred_obs_pred_inv.transpose());

				vec_t W_D_inv, W_D_inv_inv, information_ll_inv_pluss_Diag_I_I;
				den_mat_t chol_wood_diagonal_cross_cov;
				// Implementation for Bekas approach
				/*vec_t pred_var_prec_ex;
				if (calc_pred_var) {
					information_ll_inv_pluss_Diag_I_I.resize(dim_mode_);
					den_mat_t woodbury_mat;
					if (cg_preconditioner_type_ == "fitc") {
						vec_t diagonal_approx_inv_preconditioner_vecchia = (diagonal_approx_inv_preconditioner_.cwiseInverse() - information_ll_inv).cwiseInverse();
						information_ll_inv_pluss_Diag_I_I.array() = (diagonal_approx_inv_preconditioner_vecchia + information_ll_).array().inverse();
						den_mat_t diagonal_with_cross_cov_preconditioner = diagonal_approx_inv_preconditioner_vecchia.asDiagonal() * (*cross_cov_preconditioner);
						den_mat_t diagonal_cross_cov_preconditioner = information_ll_inv_pluss_Diag_I_I.asDiagonal() * diagonal_with_cross_cov_preconditioner;
						den_mat_t sigma_ip_preconditioner = *(re_comps_ip_preconditioner_cluster_i[0]->GetZSigmaZt());

						den_mat_t sigma_woodbury_preconditioner = sigma_ip_preconditioner +
							((*cross_cov_preconditioner).transpose() * diagonal_approx_inv_preconditioner_vecchia.asDiagonal()) * (*cross_cov_preconditioner);
						woodbury_mat = sigma_woodbury_preconditioner - diagonal_with_cross_cov_preconditioner.transpose() * diagonal_cross_cov_preconditioner;
						chol_den_mat_t chol_fact_woodbury_mat;
						chol_fact_woodbury_mat.compute(woodbury_mat);
						CheckCholeskyFactorization(chol_fact_woodbury_mat, "Laplace prediction Woodbury matrix");
						chol_wood_diagonal_cross_cov.resize((*cross_cov_preconditioner).cols(), dim_mode_);
						TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_woodbury_mat, diagonal_cross_cov_preconditioner.transpose(), chol_wood_diagonal_cross_cov, false);
					}
					else {
						W_D_inv = (information_ll_ + D_inv_rm_.diagonal());
						W_D_inv_inv = W_D_inv.cwiseInverse();
						information_ll_inv_pluss_Diag_I_I = W_D_inv_inv;
						chol_wood_diagonal_cross_cov.resize((*cross_cov).cols(), dim_mode_);
						den_mat_t B_invt_cross_cov = B_rm_.triangularView<Eigen::UpLoType::UnitLower>().solve((((*cross_cov).transpose() * B_t_D_inv_rm_) * W_D_inv_inv.asDiagonal()).transpose());
						den_mat_t B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov = W_D_inv_inv.cwiseSqrt().asDiagonal() * D_inv_B_cross_cov_;
						sigma_woodbury_woodbury_ = sigma_woodbury - B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov.transpose() * B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov;
						chol_fact_sigma_woodbury_woodbury_.compute(sigma_woodbury_woodbury_);
						CheckCholeskyFactorization(chol_fact_sigma_woodbury_woodbury_, "Laplace prediction VIF Woodbury matrix");
						TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury_woodbury_, B_invt_cross_cov.transpose(), chol_wood_diagonal_cross_cov, false);
					}
					den_mat_t Bt_D_inv_B_cross_cov_T_WI = Bt_D_inv_B_cross_cov.transpose() * information_ll_inv_pluss_Diag_I_I.asDiagonal();
					den_mat_t Bt_D_inv_B_cross_cov_T_WI_Bt_D_inv_B_cross_cov = Bt_D_inv_B_cross_cov_T_WI * Bt_D_inv_B_cross_cov;
					den_mat_t Bt_D_inv_B_cross_cov_T_WI_Bp_inv_Bpo_rm_T = Bt_D_inv_B_cross_cov_T_WI * Bp_inv_Bpo_rm.transpose();
					den_mat_t Bt_D_inv_B_cross_cov_T_WI_cross_cov_pred_obs_pred_inv_woodbury = Bt_D_inv_B_cross_cov_T_WI_Bt_D_inv_B_cross_cov * cross_cov_pred_obs_pred_inv_woodbury;
					den_mat_t Bt_D_inv_B_cross_cov_T_WI_sigma_ip_inv_sigma_cross_cov_pred = Bt_D_inv_B_cross_cov_T_WI_Bt_D_inv_B_cross_cov * sigma_ip_inv_sigma_cross_cov_pred;
					den_mat_t Bt_D_inv_B_cross_cov_T_WI_cross_cov_PP_Vecchia_woodbury = Bt_D_inv_B_cross_cov_T_WI_Bt_D_inv_B_cross_cov * cross_cov_PP_Vecchia_woodbury;
					pred_var_prec_ex = vec_t::Zero(num_pred);
					den_mat_t Bp_inv_Bpo_wood_T = (Bp_inv_Bpo_rm * information_ll_inv_pluss_Diag_I_I.cwiseSqrt().asDiagonal()).transpose();
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_pred; ++i) {
						pred_var_prec_ex[i] += sigma_ip_inv_sigma_cross_cov_pred.col(i).dot(Bt_D_inv_B_cross_cov_T_WI_sigma_ip_inv_sigma_cross_cov_pred.col(i) -
							2 * Bt_D_inv_B_cross_cov_T_WI_cross_cov_PP_Vecchia_woodbury.col(i) - 2 * Bt_D_inv_B_cross_cov_T_WI_Bp_inv_Bpo_rm_T.col(i) +
							2 * Bt_D_inv_B_cross_cov_T_WI_cross_cov_pred_obs_pred_inv_woodbury.col(i)) +
							cross_cov_PP_Vecchia_woodbury.col(i).dot(Bt_D_inv_B_cross_cov_T_WI_cross_cov_PP_Vecchia_woodbury.col(i) -
								2 * Bt_D_inv_B_cross_cov_T_WI_cross_cov_pred_obs_pred_inv_woodbury.col(i) +
								2 * Bt_D_inv_B_cross_cov_T_WI_Bp_inv_Bpo_rm_T.col(i)) +
							cross_cov_pred_obs_pred_inv_woodbury.col(i).dot(Bt_D_inv_B_cross_cov_T_WI_cross_cov_pred_obs_pred_inv_woodbury.col(i)) -
							2 * Bt_D_inv_B_cross_cov_T_WI_Bp_inv_Bpo_rm_T.col(i).dot(cross_cov_pred_obs_pred_inv_woodbury.col(i)) +
							Bp_inv_Bpo_wood_T.col(i).array().square().sum();
					}
					Bt_D_inv_B_cross_cov_T_WI = (Bt_D_inv_B_cross_cov.transpose() * chol_wood_diagonal_cross_cov.transpose()) * chol_wood_diagonal_cross_cov;
					Bt_D_inv_B_cross_cov_T_WI_Bt_D_inv_B_cross_cov = Bt_D_inv_B_cross_cov_T_WI * Bt_D_inv_B_cross_cov;
					Bt_D_inv_B_cross_cov_T_WI_Bp_inv_Bpo_rm_T = Bt_D_inv_B_cross_cov_T_WI * Bp_inv_Bpo_rm.transpose();
					Bt_D_inv_B_cross_cov_T_WI_cross_cov_pred_obs_pred_inv_woodbury = Bt_D_inv_B_cross_cov_T_WI_Bt_D_inv_B_cross_cov * cross_cov_pred_obs_pred_inv_woodbury;
					Bt_D_inv_B_cross_cov_T_WI_sigma_ip_inv_sigma_cross_cov_pred = Bt_D_inv_B_cross_cov_T_WI_Bt_D_inv_B_cross_cov * sigma_ip_inv_sigma_cross_cov_pred;
					Bt_D_inv_B_cross_cov_T_WI_cross_cov_PP_Vecchia_woodbury = Bt_D_inv_B_cross_cov_T_WI_Bt_D_inv_B_cross_cov * cross_cov_PP_Vecchia_woodbury;
					Bp_inv_Bpo_wood_T = (Bp_inv_Bpo_rm * chol_wood_diagonal_cross_cov.transpose()).transpose();
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_pred; ++i) {
						pred_var_prec_ex[i] += sigma_ip_inv_sigma_cross_cov_pred.col(i).dot(Bt_D_inv_B_cross_cov_T_WI_sigma_ip_inv_sigma_cross_cov_pred.col(i) -
							2 * Bt_D_inv_B_cross_cov_T_WI_cross_cov_PP_Vecchia_woodbury.col(i) - 2 * Bt_D_inv_B_cross_cov_T_WI_Bp_inv_Bpo_rm_T.col(i) +
							2 * Bt_D_inv_B_cross_cov_T_WI_cross_cov_pred_obs_pred_inv_woodbury.col(i)) +
							cross_cov_PP_Vecchia_woodbury.col(i).dot(Bt_D_inv_B_cross_cov_T_WI_cross_cov_PP_Vecchia_woodbury.col(i) -
								2 * Bt_D_inv_B_cross_cov_T_WI_cross_cov_pred_obs_pred_inv_woodbury.col(i) +
								2 * Bt_D_inv_B_cross_cov_T_WI_Bp_inv_Bpo_rm_T.col(i)) +
							cross_cov_pred_obs_pred_inv_woodbury.col(i).dot(Bt_D_inv_B_cross_cov_T_WI_cross_cov_pred_obs_pred_inv_woodbury.col(i)) -
							2 * Bt_D_inv_B_cross_cov_T_WI_Bp_inv_Bpo_rm_T.col(i).dot(cross_cov_pred_obs_pred_inv_woodbury.col(i)) +
							Bp_inv_Bpo_wood_T.col(i).array().square().sum();
					}
				}*/
				// Hoisted out of the stochastic prediction-variance sampling loop below: the VIF Woodbury factorization
				// used by the 'vifdu'/'none' preconditioner is identical for every sample. Computing it inside the
				// parallel loop wrote to the shared members W_D_inv(_inv)/sigma_woodbury_woodbury_/
				// chol_fact_sigma_woodbury_woodbury_ from all threads (a data race that intermittently corrupted the
				// factorization -> Cholesky failure/segfault) and called the non-thread-safe Log::REFatal().
				if ((cg_preconditioner_type_ == "vifdu" || cg_preconditioner_type_ == "none") && std::max(nsim_var_pred_, num_post_samples) > 0) {
					W_D_inv = (information_ll_ + D_inv_rm_.diagonal());
					W_D_inv_inv = W_D_inv.cwiseInverse();
					den_mat_t B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov = W_D_inv_inv.cwiseSqrt().asDiagonal() * D_inv_B_cross_cov_;
					sigma_woodbury_woodbury_ = sigma_woodbury - B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov.transpose() * B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov;
					chol_fact_sigma_woodbury_woodbury_.compute(sigma_woodbury_woodbury_);
					CheckCholeskyFactorization(chol_fact_sigma_woodbury_woodbury_, "Laplace prediction VIF Woodbury matrix");
				}
				int num_threads;
#ifdef _OPENMP
				num_threads = omp_get_max_threads();
#else
				num_threads = 1;
#endif
				std::uniform_int_distribution<> unif(0, 2147483646);
				std::vector<RNG_t> parallel_rngs;
				for (int ig = 0; ig < num_threads; ++ig) {
					int seed_local = unif(cg_generator_);
					parallel_rngs.push_back(RNG_t(seed_local));
				}
				bool na_inf_flag_4 = false;
#pragma omp parallel
				{
					int thread_nb;
#ifdef _OPENMP
					thread_nb = omp_get_thread_num();
#else
					thread_nb = 0;
#endif
					RNG_t rng_local = parallel_rngs[thread_nb];
#pragma omp for reduction(||:na_inf_flag_4)
					for (int i = 0; i < std::max(nsim_var_pred_, num_post_samples); ++i) {
						//z_i ~ N(0,I)
						vec_t rand_vec_pred_I_1(dim_mode_), rand_vec_pred_I_2(dim_mode_), rand_vec_pred_I_3((*cross_cov).cols());
						std::normal_distribution<double> ndist(0.0, 1.0);
						for (int j = 0; j < dim_mode_; j++) {
							rand_vec_pred_I_1(j) = ndist(rng_local);
							rand_vec_pred_I_2(j) = ndist(rng_local);
						}
						for (int j = 0; j < (*cross_cov).cols(); j++) {
							rand_vec_pred_I_3(j) = ndist(rng_local);
						}

						vec_t rand_vec_pred_SigmaI_plus_W_inv(dim_mode_);
						bool has_NA_or_Inf = false;
						//z_i ~ N(0,Sigma) (not possible to sample directly from Sigma^{-1})
						vec_t Sigma_sqrt_rand_vec = chol_ip_cross_cov.transpose() * rand_vec_pred_I_3;
						vec_t D_sqrt = D_inv_rm_.diagonal().cwiseInverse().cwiseSqrt();
						Sigma_sqrt_rand_vec += B_rm_.triangularView<Eigen::UpLoType::UnitLower>().solve(D_sqrt.cwiseProduct(rand_vec_pred_I_1));
						//z_i ~ N(0,Sigma^{-1})
						vec_t Bt_D_inv_Sigma_sqrt_rand_vec = B_t_D_inv_rm_ * (B_rm_ * Sigma_sqrt_rand_vec);
						vec_t Sigma_inv_Sigma_sqrt_rand_vec = Bt_D_inv_Sigma_sqrt_rand_vec - Bt_D_inv_B_cross_cov * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * Bt_D_inv_Sigma_sqrt_rand_vec);
						//z_i ~ N(0,(Sigma^{-1} + W))
						vec_t rand_vec_pred_SigmaI_plus_W = Sigma_inv_Sigma_sqrt_rand_vec + W_diag_sqrt.cwiseProduct(rand_vec_pred_I_2);
						//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
						if (cg_preconditioner_type_ == "vifdu" || cg_preconditioner_type_ == "none") {
							// W_D_inv_inv, sigma_woodbury_woodbury_ and chol_fact_sigma_woodbury_woodbury_ are loop-invariant and are
							// computed once BEFORE this parallel region (hoisted block above). They must not be recomputed here:
							// writing to these shared members from every thread is a data race, and CheckCholeskyFactorization()
							// calls Log::REFatal(), which is not thread-safe (intermittently segfaulted / printed the 'VIF Woodbury
							// matrix' Cholesky error twice).
							CGFVIFLaplaceVec(information_ll_, B_rm_, B_t_D_inv_rm_, chol_fact_sigma_woodbury, cross_cov, W_D_inv_inv,
								chol_fact_sigma_woodbury_woodbury_, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf, cg_max_num_it_,
								true, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, true);
						}
						else if (cg_preconditioner_type_ == "fitc") {
							vec_t rand_vec_pred_SigmaI_plus_W_inv_interim(dim_mode_);
							vec_t rhs_part, rhs_part1, rhs_part2;
							rhs_part1 = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(rand_vec_pred_SigmaI_plus_W);
							rhs_part = D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve(rhs_part1);
							rhs_part2 = (*cross_cov) * (chol_fact_sigma_ip.solve((*cross_cov).transpose() * rand_vec_pred_SigmaI_plus_W));
							rand_vec_pred_SigmaI_plus_W = rhs_part + rhs_part2;
							CGVIFLaplace_Version_SigmaPlusWinvVec(information_ll_inv, D_inv_B_rm_, B_rm_, chol_fact_woodbury_preconditioner_,
								chol_ip_cross_cov, cross_cov_preconditioner, diagonal_approx_inv_preconditioner_, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv_interim, has_NA_or_Inf,
								cg_max_num_it_, true, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, true);
							rand_vec_pred_SigmaI_plus_W_inv = information_ll_inv.asDiagonal() * rand_vec_pred_SigmaI_plus_W_inv_interim;
						}
						if (has_NA_or_Inf) {
							na_inf_flag_4 = true;
						}
						vec_t sigma_woodbury_vec = (*cross_cov) * chol_fact_sigma_woodbury.solve(Bt_D_inv_B_cross_cov.transpose() * rand_vec_pred_SigmaI_plus_W_inv);
						//z_i ~ N(0, (Sigma_pm Sigma_ip^-1 Sigma_mn - B_p^-1 B_po (B_o^T D_o^-1 B_o)^-1) Sigma^{-1} (Sigma^{-1} + W)^{-1} Sigma^{-1} (Sigma_nm Sigma_ip^-1 Sigma_mp - (B_o^T D_o^-1 B_o)^-1 B_po^T B_p^-1 ))
						vec_t sigma_pred_sigma_inv_vec = cross_cov_pred_ip * chol_fact_sigma_ip.solve(Bt_D_inv_B_cross_cov.transpose() * (rand_vec_pred_SigmaI_plus_W_inv - sigma_woodbury_vec));
						vec_t sigmSigmaI_modechiSigmaI_mode = Bp_inv_Bpo_rm * (rand_vec_pred_SigmaI_plus_W_inv - sigma_woodbury_vec);
						vec_t rand_vec_pred = sigma_pred_sigma_inv_vec - sigmSigmaI_modechiSigmaI_mode;
						if (calc_pred_cov) {
							if (i < nsim_var_pred_) {
								den_mat_t pred_cov_private = rand_vec_pred * rand_vec_pred.transpose();
#pragma omp critical
								{
									pred_cov += pred_cov_private;
								}
							}
						}
						if (calc_pred_var) {
							if (i < nsim_var_pred_) {
								vec_t pred_var_private = rand_vec_pred.cwiseProduct(rand_vec_pred);
#pragma omp critical
								{
									pred_var += pred_var_private;
								}
							}
						}
						if (sample_posterior) {
							if (i < num_post_samples) {
								post_samples_id.col(i) += rand_vec_pred;
							}
						}
						// Implementation for Bekas approach
//							rand_vec_pred_I_1.resize(num_pred);
//							std::uniform_real_distribution<double> udist(0.0, 1.0);
//							for (int j = 0; j < num_pred; j++) {
//								// Map uniform [0,1) to Rademacher -1 or 1
//								rand_vec_pred_I_1(j) = (udist(parallel_rngs[thread_nb]) < 0.5) ? -1.0 : 1.0;
//							}
//							vec_t vecchia_rand_vec_pred_I_1 = Bp_inv_Bpo_rm.transpose()* rand_vec_pred_I_1;
//							vec_t pred_proc_rand_vec_pred_I_1 = (Bt_D_inv_B_cross_cov)*chol_fact_sigma_ip.solve((cross_cov_pred_ip).transpose() * rand_vec_pred_I_1);
//							vec_t sigma_woodbury_vec = (Bt_D_inv_B_cross_cov) * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * (vecchia_rand_vec_pred_I_1 - pred_proc_rand_vec_pred_I_1));
//							vec_t rand_vec_pred_interim = pred_proc_rand_vec_pred_I_1 - vecchia_rand_vec_pred_I_1 + sigma_woodbury_vec;
//							vec_t rand_vec_pred, rand_vec_pred_prec, sigma_pred_sigma_inv_vec, sigmSigmaI_modechiSigmaI_mode;
//							vec_t WI_rand_vec_pred_prec_interim = information_ll_inv_pluss_Diag_I_I.asDiagonal() * rand_vec_pred_interim + chol_wood_diagonal_cross_cov.transpose() * (chol_wood_diagonal_cross_cov * rand_vec_pred_interim);
//							if (cg_preconditioner_type_ == "vifdu" || cg_preconditioner_type_ == "none") {
//								vec_t W_D_inv = (information_ll_ + D_inv_rm_.diagonal());
//								vec_t W_D_inv_inv = W_D_inv.cwiseInverse();
//								CGFVIFLaplaceVec(information_ll_, B_rm_, B_t_D_inv_rm_, chol_fact_sigma_woodbury, cross_cov, W_D_inv_inv,
//									chol_fact_sigma_woodbury_woodbury_, rand_vec_pred_interim, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf, cg_max_num_it_,
//									true, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, true);
//
//								sigma_woodbury_vec = (*cross_cov) * chol_fact_sigma_woodbury.solve(Bt_D_inv_B_cross_cov.transpose() * rand_vec_pred_SigmaI_plus_W_inv);
//								sigma_pred_sigma_inv_vec = cross_cov_pred_ip * chol_fact_sigma_ip.solve(Bt_D_inv_B_cross_cov.transpose() * (rand_vec_pred_SigmaI_plus_W_inv - sigma_woodbury_vec));
//								sigmSigmaI_modechiSigmaI_mode = Bp_inv_Bpo_rm * (rand_vec_pred_SigmaI_plus_W_inv - sigma_woodbury_vec);
//								rand_vec_pred = sigma_pred_sigma_inv_vec - sigmSigmaI_modechiSigmaI_mode;
//							}
//							else if (cg_preconditioner_type_ == "fitc") {
//								vec_t WI_rand_vec_pred_interim = information_ll_inv.asDiagonal() * rand_vec_pred_interim;
//								CGVIFLaplace_Version_SigmaPlusWinvVec(information_ll_inv, D_inv_B_rm_, B_rm_, chol_fact_woodbury_preconditioner_,
//									chol_ip_cross_cov, cross_cov_preconditioner, diagonal_approx_inv_preconditioner_, WI_rand_vec_pred_interim, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf,
//									cg_max_num_it_, true, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, true);
//								vec_t rhs_part1 = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(rand_vec_pred_SigmaI_plus_W_inv);
//								vec_t rhs_part = D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve(rhs_part1);
//								rand_vec_pred = cross_cov_pred_ip * chol_fact_sigma_ip.solve((*cross_cov).transpose() * rand_vec_pred_SigmaI_plus_W_inv) - Bp_inv_Bpo_rm * rhs_part;
//							}
//							sigma_woodbury_vec = (*cross_cov) * chol_fact_sigma_woodbury.solve(Bt_D_inv_B_cross_cov.transpose() * WI_rand_vec_pred_prec_interim);
//							sigma_pred_sigma_inv_vec = cross_cov_pred_ip * chol_fact_sigma_ip.solve(Bt_D_inv_B_cross_cov.transpose() * (WI_rand_vec_pred_prec_interim - sigma_woodbury_vec));
//							sigmSigmaI_modechiSigmaI_mode = Bp_inv_Bpo_rm * (WI_rand_vec_pred_prec_interim - sigma_woodbury_vec);
//							rand_vec_pred_prec = sigma_pred_sigma_inv_vec - sigmSigmaI_modechiSigmaI_mode;
//							if (calc_pred_cov) {
//								den_mat_t pred_cov_private = rand_vec_pred_I_1 * rand_vec_pred.transpose();
//#pragma omp critical			
//								{
//									pred_cov += pred_cov_private;
//								}
//							}
//							if (calc_pred_var) {
//								vec_t pred_var_private = rand_vec_pred_I_1.cwiseProduct(rand_vec_pred);
//								vec_t pred_var_private_prec = rand_vec_pred_I_1.cwiseProduct(rand_vec_pred_prec);
//#pragma omp critical
//								{
//									pred_var += pred_var_private;
//									pred_var_prec += pred_var_private_prec;
//									pred_var_prec_diff += (pred_var_private_prec - pred_var_prec_ex);
//									pred_var_prec_sq.array() += (pred_var_private_prec- pred_var_prec_ex).array().square();
//									pred_var_prec_prod += (pred_var_private_prec - pred_var_prec_ex).cwiseProduct(pred_var_private);
//								}
//							}
					}
				}
				if (na_inf_flag_4) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
				if (calc_pred_cov) {
					pred_cov /= nsim_var_pred_;
					// Deterministic part
					if (CondObsOnly) {
						pred_cov.diagonal().array() += Dp.array();
					}
					else {
						pred_cov += Bp_inv_Dp_rm * Bp_inv_rm.transpose();
					}
					if (pred_mean.size() > 10000) {
						Log::REInfo("The computational complexity and the storage of the predictive covariance martix heavily depend on the number of prediction location. "
							"Therefore, if this number is large we recommend only computing the predictive variances ");
					}
					T_mat PP_Part;
					ConvertTo_T_mat_FromDense<T_mat>(chol_ip_cross_cov_pred.transpose() * chol_ip_cross_cov_pred, PP_Part);
					T_mat PP_V_Part;
					ConvertTo_T_mat_FromDense<T_mat>(cross_cov_PP_Vecchia * sigma_ip_inv_sigma_cross_cov_pred, PP_V_Part);
					T_mat V_Part;
					ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_obs_pred_inv * sigma_ip_inv_sigma_cross_cov_pred, V_Part);
					T_mat V_Part_t;
					ConvertTo_T_mat_FromDense<T_mat>(sigma_ip_inv_sigma_cross_cov_pred.transpose() * cross_cov_pred_obs_pred_inv.transpose(), V_Part_t);
					T_mat PP_V_PP_Part;
					ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_obs_pred_inv * cross_cov_PP_Vecchia_woodbury, PP_V_PP_Part);
					T_mat PP_V_PP_Part_t;
					ConvertTo_T_mat_FromDense<T_mat>(cross_cov_PP_Vecchia_woodbury.transpose() * cross_cov_pred_obs_pred_inv.transpose(), PP_V_PP_Part_t);
					T_mat PP_V_V_Part;
					ConvertTo_T_mat_FromDense<T_mat>(cross_cov_PP_Vecchia * cross_cov_PP_Vecchia_woodbury, PP_V_V_Part);
					T_mat V_V_Part;
					ConvertTo_T_mat_FromDense<T_mat>(cross_cov_pred_obs_pred_inv * cross_cov_pred_obs_pred_inv_woodbury, V_V_Part);
					pred_cov += PP_Part - PP_V_Part + V_Part + V_Part_t - PP_V_PP_Part + PP_V_V_Part - PP_V_PP_Part_t + V_V_Part;
				}
				if (calc_pred_var) {
					pred_var /= nsim_var_pred_;
					// Implementation for Bekas approach
//						pred_var_prec /= nsim_var_pred_;
//						pred_var_prec_prod /= nsim_var_pred_;
//						pred_var_prec_sq /= nsim_var_pred_;
//						pred_var_prec_diff /= nsim_var_pred_;
//
//						vec_t c_cov = pred_var_prec_prod - pred_var.cwiseProduct(pred_var_prec_diff);
//						// Optimal c
//						vec_t c_opt = c_cov.array() / pred_var_prec_sq.array();
//						// Correction if c_opt_i = inf
//#pragma omp parallel for schedule(static)   
//						for (int i = 0; i < c_opt.size(); ++i) {
//							if (pred_var_prec_sq.coeffRef(i) == 0) {
//								c_opt[i] = 1;
//							}
//						}
//						pred_var += c_opt.cwiseProduct(pred_var_prec_ex - pred_var_prec);

					// Deterministic part
					if (CondObsOnly) {
						pred_var += Dp;
					}
					else {
						pred_var += Bp_inv_Dp_rm.cwiseProduct(Bp_inv_rm) * vec_t::Ones(num_pred);
					}
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_pred; ++i) {
						pred_var[i] += (cross_cov_pred_ip.row(i) - cross_cov_PP_Vecchia.row(i) +
							2 * cross_cov_pred_obs_pred_inv.row(i)).dot(sigma_ip_inv_sigma_cross_cov_pred.col(i)) +
							(cross_cov_PP_Vecchia.row(i) - 2 * cross_cov_pred_obs_pred_inv.row(i)).dot(cross_cov_PP_Vecchia_woodbury.col(i)) +
							(cross_cov_pred_obs_pred_inv.row(i)).dot(cross_cov_pred_obs_pred_inv_woodbury.col(i));
					}
				}
			} //end iterative methods using simulation
			else {//using Cholesky decomposition
				if (sample_posterior) {
					Log::REFatal("Posterior sampling is not implemented for the VIF approximation using Cholesky decomposition.");
				}
				den_mat_t sigma_resid_plus_W_inv_cross_cov = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(information_ll_.asDiagonal() * (*cross_cov));
				den_mat_t sigma_woodbury_2 = (sigma_ip_stable)+Bt_D_inv_B_cross_cov.transpose() * sigma_resid_plus_W_inv_cross_cov;
				chol_den_mat_t chol_fact_sigma_woodbury_2;
				chol_fact_sigma_woodbury_2.compute(sigma_woodbury_2);
				CheckCholeskyFactorization(chol_fact_sigma_woodbury_2, "Laplace prediction Woodbury matrix");

				den_mat_t M_aux_1 = sigma_ip_inv_sigma_cross_cov_pred.transpose() * (Bt_D_inv_B_cross_cov.transpose() * sigma_resid_plus_W_inv_cross_cov);
				den_mat_t M_aux_2 = chol_fact_sigma_woodbury_2.solve(M_aux_1.transpose());
				sp_mat_t Maux; //Maux = L\(Bpo^T * Bp^-1), L = Chol(Sigma^-1 + W)
				den_mat_t M_aux_3(pred_mean.size(), (*cross_cov).cols());
				if (CondObsOnly) {
					Maux = Bpo.transpose();//Bp = Id
#pragma omp parallel for schedule(static)   
					for (int i = 0; i < (*cross_cov).cols(); ++i) {
						M_aux_3.col(i) = Bpo_rm * sigma_resid_plus_W_inv_cross_cov.col(i);
					}
				}
				else {
					Bp_inv = sp_mat_t(Bp.rows(), Bp.cols());
					Bp_inv.setIdentity();
					TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
					//Bp.triangularView<Eigen::UpLoType::UnitLower>().solveInPlace(Bp_inv);//much slower
					Maux = Bpo.transpose() * Bp_inv.transpose();
					Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
					M_aux_3 = Maux.transpose() * sigma_resid_plus_W_inv_cross_cov;
				}
				den_mat_t M_aux_4 = chol_fact_sigma_woodbury_2.solve(M_aux_3.transpose());
				TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, Maux, Maux, false);
				if (calc_pred_cov) {
					if (CondObsOnly) {
						pred_cov = Maux.transpose() * Maux;
						pred_cov.diagonal().array() += Dp.array();
					}
					else {
						pred_cov = Bp_inv_Dp * Bp_inv.transpose() + Maux.transpose() * Maux;
					}
					T_mat PP_Part, PP_Part1, PP_Part2, PP_Part3, PP_Part3_t, PP_Part4, PP_Part4_t, PP_Part5;
					ConvertTo_T_mat_FromDense<T_mat>(chol_ip_cross_cov_pred.transpose() * chol_ip_cross_cov_pred, PP_Part);
					ConvertTo_T_mat_FromDense<T_mat>(M_aux_1 * sigma_ip_inv_sigma_cross_cov_pred, PP_Part1);
					ConvertTo_T_mat_FromDense<T_mat>(M_aux_1 * M_aux_2, PP_Part2);
					ConvertTo_T_mat_FromDense<T_mat>(M_aux_3 * sigma_ip_inv_sigma_cross_cov_pred, PP_Part3);
					ConvertTo_T_mat_FromDense<T_mat>(sigma_ip_inv_sigma_cross_cov_pred.transpose() * M_aux_3.transpose(), PP_Part3_t);
					ConvertTo_T_mat_FromDense<T_mat>(M_aux_3 * M_aux_2, PP_Part4);
					ConvertTo_T_mat_FromDense<T_mat>(M_aux_2.transpose() * M_aux_3.transpose(), PP_Part4_t);
					ConvertTo_T_mat_FromDense<T_mat>(M_aux_3 * M_aux_4, PP_Part5);
					ConvertTo_T_mat_FromDense<T_mat>(M_aux_3 * M_aux_4, PP_Part5);
					pred_cov += PP_Part - PP_Part1 + PP_Part2 + PP_Part3 + PP_Part3_t - PP_Part4 - PP_Part4_t + PP_Part5;
				}
				if (calc_pred_var) {
					pred_var = vec_t(num_pred);
					Maux = Maux.cwiseProduct(Maux);
					if (CondObsOnly) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_pred; ++i) {
							pred_var[i] = Dp[i];
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_pred; ++i) {
							pred_var[i] = (Bp_inv_Dp.row(i)).dot(Bp_inv.row(i));
						}
					}
#pragma omp parallel for schedule(static)
					for (int i = 0; i < num_pred; ++i) {
						pred_var[i] += Maux.col(i).sum() + chol_ip_cross_cov_pred.col(i).array().square().sum() - sigma_ip_inv_sigma_cross_cov_pred.col(i).dot(M_aux_1.row(i)) +
							M_aux_2.col(i).dot(M_aux_1.row(i)) + 2 * sigma_ip_inv_sigma_cross_cov_pred.col(i).dot(M_aux_3.row(i)) - 2 * M_aux_2.col(i).dot(M_aux_3.row(i)) +
							M_aux_4.col(i).dot(M_aux_3.row(i));
					}
				}
			}
		}//end calc_pred_cov || calc_pred_var || sample_posterior
		//if (sample_posterior) {
		//	Log::REInfo("Test %g %g %g %g %g", post_samples_id.mean(), post_samples_id.coeffRef(0, 0), post_samples_id.coeffRef(1, 0),
		//		post_samples_id.coeffRef(0, 1), post_samples_id.coeffRef(1, 1));
		//}
	}//end PredictLaplaceApproxFSVA

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::PredictLaplaceApproxVecchia(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		std::map<int, sp_mat_t>& B,
		std::map<int, sp_mat_t>& D_inv,
		const sp_mat_t& Bpo,
		sp_mat_t& Bp,
		const vec_t& Dp,
		bool sample_posterior,
		int num_post_samples,
		den_mat_t& post_samples_id,
		vec_t& pred_mean,
		den_mat_t& pred_cov,
		vec_t& pred_var,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool calc_mode,
		bool CondObsOnly,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_ip_cluster_i,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const den_mat_t& chol_ip_cross_cov,
		const chol_den_mat_t& chol_fact_sigma_ip,
		int num_gp,
		data_size_t cluster_i,
		REModelTemplate<T_mat, T_chol>* re_model) {
		CHECK(num_gp <= num_sets_re_);
		if (calc_mode) {// Calculate mode and Cholesky factor of Sigma^-1 + W at mode
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLVecchia(y_data, y_data_int, fixed_effects, B, D_inv, false, Sigma_L_k_, false, mll,
				re_comps_ip_cluster_i, re_comps_cross_cov_cluster_i, chol_ip_cross_cov, chol_fact_sigma_ip, cluster_i, re_model);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		int num_pred = (int)Bp.cols();
		CHECK((int)Dp.size() == num_pred);
		if (CondObsOnly) {
			pred_mean = -Bpo * mode_.segment(num_gp * dim_mode_per_set_re_, dim_mode_per_set_re_);
		}
		else {
			vec_t Bpo_mode = Bpo * mode_.segment(num_gp * dim_mode_per_set_re_, dim_mode_per_set_re_);
			pred_mean = -Bp.triangularView<Eigen::UpLoType::UnitLower>().solve(Bpo_mode);
		}
		if (calc_pred_cov || calc_pred_var || sample_posterior) {
			if (use_variance_correction_for_prediction_) {
				CHECK(num_sets_re_ == 1);
				diag_information_variance_correction_for_prediction_ = true;
				vec_t location_par;//location parameter = mode of random effects + fixed effects
				double* location_par_ptr;
				InitializeLocationPar(fixed_effects, location_par, &location_par_ptr);
				CalcInformationLogLik(y_data, y_data_int, location_par_ptr, true);
				if (matrix_inversion_method_ != "iterative") {
					sp_mat_t SigmaI_plus_W = B[0].transpose() * D_inv[0] * B[0];
					SigmaI_plus_W.diagonal().array() += information_ll_.array();
					SigmaI_plus_W.makeCompressed();
					chol_fact_SigmaI_plus_ZtWZ_vecchia_.factorize(SigmaI_plus_W);//This is the bottleneck for large data
					CheckCholeskyFactorization(chol_fact_SigmaI_plus_ZtWZ_vecchia_, "PredictLaplaceApproxVecchia");
				}
				diag_information_variance_correction_for_prediction_ = false;
			}
			sp_mat_t Bp_inv, Bp_inv_Dp;
			//Version Simulation
			if (matrix_inversion_method_ == "iterative") {
				if (!sample_posterior) {
					num_post_samples = 0;
				}
				if (!calc_pred_cov && !calc_pred_var) {
					nsim_var_pred_ = 0;
				}
				sp_mat_rm_t Bp_inv_Dp_rm, Bp_inv_rm;
				sp_mat_rm_t Bpo_rm = sp_mat_rm_t(Bpo);
				sp_mat_rm_t Bp_rm;
				sp_mat_rm_t Bp_inv_Bpo_rm; //Bp^(-1) * Bpo 
				if (CondObsOnly) {
					Bp_inv_Bpo_rm = Bpo_rm; //Bp = Id
				}
				else {
					Bp_rm = sp_mat_rm_t(Bp);
					Bp_inv_rm = sp_mat_rm_t(Bp_rm.rows(), Bp_rm.cols());
					Bp_inv_rm.setIdentity();
					TriangularSolve<sp_mat_rm_t, sp_mat_rm_t, sp_mat_rm_t>(Bp_rm, Bp_inv_rm, Bp_inv_rm, false);
					Bp_inv_Bpo_rm = Bp_inv_rm * Bpo_rm;
					Bp_inv_Dp_rm = Bp_inv_rm * Dp.asDiagonal();
				}
				if (calc_pred_cov) {
					pred_cov = den_mat_t::Zero(num_pred, num_pred);
				}
				if (calc_pred_var) {
					pred_var = vec_t::Zero(num_pred);
				}
				if (HasNegativeValueInformationLogLik()) {
					Log::REFatal("PredictLaplaceApproxVecchia: Negative values found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
						"Cannot have negative values when using 'iterative' methods for predictive variances in Vecchia-Laplace approximations ");
				}
				vec_t W_diag_sqrt = information_ll_.cwiseSqrt();
				sp_mat_rm_t B_t_D_inv_sqrt_rm = B_rm_.transpose() * (D_inv_rm_.cwiseSqrt());
				int num_threads;
#ifdef _OPENMP
				num_threads = omp_get_max_threads();
#else
				num_threads = 1;
#endif
				std::uniform_int_distribution<> unif(0, 2147483646);
				std::vector<RNG_t> parallel_rngs;
				for (int ig = 0; ig < num_threads; ++ig) {
					int seed_local = unif(cg_generator_);
					parallel_rngs.push_back(RNG_t(seed_local));
				}
				bool na_inf_flag_5 = false;
#pragma omp parallel
				{
					int thread_nb;
#ifdef _OPENMP
					thread_nb = omp_get_thread_num();
#else
					thread_nb = 0;
#endif
					RNG_t rng_local = parallel_rngs[thread_nb];
					den_mat_t pred_cov_private;
					if (calc_pred_cov) {
						pred_cov_private = den_mat_t::Zero(num_pred, num_pred);
					}
					vec_t pred_var_private;
					if (calc_pred_var) {
						pred_var_private = vec_t::Zero(num_pred);
					}
#pragma omp for reduction(||:na_inf_flag_5)
					for (int i = 0; i < std::max(nsim_var_pred_, num_post_samples); ++i) {
						//z_i ~ N(0,I)
						std::normal_distribution<double> ndist(0.0, 1.0);
						vec_t rand_vec_pred_I_1(dim_mode_), rand_vec_pred_I_2(dim_mode_);
						for (int j = 0; j < dim_mode_; j++) {
							rand_vec_pred_I_1(j) = ndist(rng_local);
							rand_vec_pred_I_2(j) = ndist(rng_local);
						}
						//z_i ~ N(0,(Sigma^{-1} + W))
						vec_t rand_vec_pred_SigmaI_plus_W = B_t_D_inv_sqrt_rm * rand_vec_pred_I_1 + W_diag_sqrt.cwiseProduct(rand_vec_pred_I_2);
						vec_t rand_vec_pred_SigmaI_plus_W_inv(dim_mode_);
						//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
						bool has_NA_or_Inf = false;
						Inv_SigmaI_plus_ZtWZ_Vecchia_iterative_given_PC(cg_max_num_it_, re_comps_cross_cov_cluster_i, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, true, has_NA_or_Inf);
						if (has_NA_or_Inf) {
							na_inf_flag_5 = true;
						}
						if (num_sets_re_ > 1) {
							rand_vec_pred_SigmaI_plus_W_inv = rand_vec_pred_SigmaI_plus_W_inv.segment(num_gp * dim_mode_per_set_re_, dim_mode_per_set_re_);// this could be done much more efficiently avoiding double calculations...
						}
						//z_i ~ N(0, Bp^{-1} Bpo (Sigma^{-1} + W)^{-1} Bpo^T Bp^{-1})
						vec_t rand_vec_pred = Bp_inv_Bpo_rm * rand_vec_pred_SigmaI_plus_W_inv;
						if (calc_pred_cov) {
							if (i < nsim_var_pred_) {
								pred_cov_private += rand_vec_pred * rand_vec_pred.transpose();
							}
						}
						if (calc_pred_var) {
							if (i < nsim_var_pred_) {
								pred_var_private += rand_vec_pred.cwiseProduct(rand_vec_pred);
							}
						}
						if (sample_posterior) {
							if (i < num_post_samples) {
								post_samples_id.col(i) += rand_vec_pred;
							}
						}
					}//end for loop
#pragma omp critical
					{
						if (calc_pred_cov) {
							pred_cov += pred_cov_private;
						}
						if (calc_pred_var) {
							pred_var += pred_var_private;
						}
					}
				} // end #pragma omp parallel
				if (na_inf_flag_5) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
				if (calc_pred_cov) {
					pred_cov /= nsim_var_pred_;
					if (CondObsOnly) {
						pred_cov.diagonal().array() += Dp.array();
					}
					else {
						pred_cov += Bp_inv_Dp_rm * Bp_inv_rm.transpose();
					}
				}
				if (calc_pred_var) {
					pred_var /= nsim_var_pred_;
					if (CondObsOnly) {
						pred_var += Dp;
					}
					else {
						pred_var += Bp_inv_Dp_rm.cwiseProduct(Bp_inv_rm) * vec_t::Ones(num_pred);
					}
				}
			} //end iterative methods using simulation
			else {//using Cholesky decomposition
				if (sample_posterior) {
					Log::REFatal("Posterior sampling is not implemented for the Vecchia approximation using Cholesky decomposition.");
				}
				sp_mat_t Maux; //Maux = L\(Bpo^T * Bp^-1), L = Chol(Sigma^-1 + W)
				if (CondObsOnly) {
					Maux = Bpo.transpose();//Bp = Id
				}
				else {
					Bp_inv = sp_mat_t(Bp.rows(), Bp.cols());
					Bp_inv.setIdentity();
					TriangularSolve<sp_mat_t, sp_mat_t, sp_mat_t>(Bp, Bp_inv, Bp_inv, false);
					//Bp.triangularView<Eigen::UpLoType::UnitLower>().solveInPlace(Bp_inv);//much slower
					Maux = Bpo.transpose() * Bp_inv.transpose();
					Bp_inv_Dp = Bp_inv * Dp.asDiagonal();
				}
				if (num_sets_re_ == 1) {
					TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, Maux, Maux, false);
				}
				else {
					CHECK(num_sets_re_ == 2);
					sp_mat_t Maux_1, Maux_2, Maux_all;
					if (num_gp == 0) {
						Maux_1 = Maux;
						Maux_2 = sp_mat_t(dim_mode_per_set_re_, num_pred);
					}
					else {
						Maux_1 = sp_mat_t(dim_mode_per_set_re_, num_pred);
						Maux_2 = Maux;
					}
					GPBoost::CreatSparseBlockDiagonalMartix<sp_mat_t>(Maux_1, Maux_2, Maux_all);
					Maux_1.resize(0, 0);
					Maux_2.resize(0, 0);
					CHECK(Maux_all.rows() == dim_mode_);
					CHECK(Maux_all.cols() == 2 * num_pred);
					TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, Maux_all, Maux_all, false);
					Maux = Maux_all.block(num_gp * dim_mode_per_set_re_, num_gp * num_pred, dim_mode_per_set_re_, num_pred);
				}
				if (calc_pred_cov) {
					if (CondObsOnly) {
						pred_cov = Maux.transpose() * Maux;
						pred_cov.diagonal().array() += Dp.array();
					}
					else {
						pred_cov = Bp_inv_Dp * Bp_inv.transpose() + Maux.transpose() * Maux;
					}
				}
				if (calc_pred_var) {
					pred_var = vec_t(num_pred);
					Maux = Maux.cwiseProduct(Maux);
					if (CondObsOnly) {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_pred; ++i) {
							pred_var[i] = Dp[i] + Maux.col(i).sum();
						}
					}
					else {
#pragma omp parallel for schedule(static)
						for (int i = 0; i < num_pred; ++i) {
							pred_var[i] = (Bp_inv_Dp.row(i)).dot(Bp_inv.row(i)) + Maux.col(i).sum();
						}
					}
				}
			}
		}//end calc_pred_cov || calc_pred_var || sample_posterior
	}//end PredictLaplaceApproxVecchia

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::Sample_Posterior_LaplaceApprox_Stable(const std::shared_ptr<T_mat>& Sigma) {
		CHECK(num_sets_re_ == 1);
		CHECK(Sigma != nullptr);
		T_chol chol_fact_SigmaI_plus_W;
		{
			T_mat Sigma_stable = (*Sigma);
			Sigma_stable.diagonal().array() *= JITTER_MUL;
			T_chol chol_fact_Sigma;
			bool chol_fact_pattern_analyzed = false;
			CalcChol<T_mat>(chol_fact_Sigma, Sigma_stable, chol_fact_pattern_analyzed);
			T_mat SigmaI_plus_W(Sigma_stable.rows(), Sigma_stable.cols());
			SigmaI_plus_W.setIdentity();
			SolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Sigma, SigmaI_plus_W, SigmaI_plus_W);
			SigmaI_plus_W += information_ll_.asDiagonal();
			chol_fact_pattern_analyzed = false;
			CalcChol<T_mat>(chol_fact_SigmaI_plus_W, SigmaI_plus_W, chol_fact_pattern_analyzed);
		} // Sigma_stable, chol_fact_Sigma, and SigmaI_plus_W are destroyed here
		//sample iid normal random vectors
		if (!sampled_rand_vec_I_sim_post_) {
			rand_vec_I_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_sim_post_);
			rand_vec_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			if (reuse_rand_vec_I_sim_post_) {
				sampled_rand_vec_I_sim_post_ = true;
			}
		}//end !sampled_rand_vec_I_sim_post_
		CHECK(rand_vec_I_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_I_sim_post_.rows() == dim_mode_);
		CHECK(rand_vec_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_sim_post_.rows() == dim_mode_);
		TriangularSolveGivenCholesky<T_chol, T_mat, den_mat_t, den_mat_t>(chol_fact_SigmaI_plus_W, rand_vec_I_sim_post_, rand_vec_sim_post_, true);
		SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean();
		rand_vec_sim_post_calculated_ = true;
	}//end Sample_Posterior_LaplaceApprox_Stable

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::Sample_Posterior_LaplaceApprox_GroupedRE(const sp_mat_t& SigmaI,
		bool has_vecchia_gp,
		const sp_mat_t& B,
		const sp_mat_t& D_inv) {
		CHECK(num_sets_re_ == 1);
		const data_size_t dim_re_group = (data_size_t)SigmaI.cols();
		data_size_t dim_gp = 0;
		if (has_vecchia_gp){
			dim_gp = (data_size_t)B.cols();
		}
		CHECK(dim_gp + dim_re_group == dim_mode_);
		//sample iid normal random vectors
		if (!sampled_rand_vec_I_sim_post_) {
			rand_vec_I_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_sim_post_);
			rand_vec_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			if (matrix_inversion_method_ == "iterative") {
				rand_vec_I_2_sim_post_.resize(num_data_, num_rand_vec_sim_post_);
				GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_2_sim_post_);
			}
			if (reuse_rand_vec_I_sim_post_) {
				sampled_rand_vec_I_sim_post_ = true;
			}
		}//end !sampled_rand_vec_I_sim_post_
		CHECK(rand_vec_I_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_I_sim_post_.rows() == dim_mode_);
		CHECK(rand_vec_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_sim_post_.rows() == dim_mode_);
		if (matrix_inversion_method_ == "cholesky") {
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, den_mat_t, den_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, rand_vec_I_sim_post_, rand_vec_sim_post_, true);
		}
		else if (matrix_inversion_method_ == "iterative") {
			CHECK(rand_vec_I_2_sim_post_.cols() == num_rand_vec_sim_post_);
			CHECK(rand_vec_I_2_sim_post_.rows() == num_data_);
			if (HasNegativeValueInformationLogLik()) {
				Log::REFatal("Sample_Posterior_LaplaceApprox_GroupedRE: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
					"Posterior sampling with iterative matrix inversion requires the square root of W ");
			}
			vec_t SigmaI_diag_sqrt;
			sp_mat_t B_t_D_inv_sqrt;
			if (has_vecchia_gp) {
				B_t_D_inv_sqrt = B.transpose() * (D_inv.cwiseSqrt());
			}
			else {
				SigmaI_diag_sqrt = SigmaI.diagonal().cwiseSqrt();
			}
			sp_mat_rm_t Zt_W_sqrt_rm = sp_mat_rm_t((*Zt_) * information_ll_.cwiseSqrt().asDiagonal());
			bool na_inf_flag_6 = false;
#pragma omp parallel
			{
				vec_t rand_vec_pred_SigmaI_plus_ZtWZ(dim_mode_); // allocated once per thread
				vec_t rand_vec_pred_SigmaI_plus_ZtWZ_inv(dim_mode_);
#pragma omp for schedule(static) reduction(||:na_inf_flag_6)
				for (int i = 0; i < num_rand_vec_sim_post_; ++i) {
					//z_i ~ N(0,(Sigma^(-1) + Z^T W Z))
					if (has_vecchia_gp) {
						rand_vec_pred_SigmaI_plus_ZtWZ.segment(0, dim_re_group) = SigmaI_diag_sqrt.asDiagonal() * rand_vec_I_sim_post_.col(i).segment(0, dim_re_group);
						rand_vec_pred_SigmaI_plus_ZtWZ.segment(dim_re_group, dim_gp) = B_t_D_inv_sqrt * rand_vec_I_sim_post_.col(i).segment(dim_re_group, dim_gp);
						rand_vec_pred_SigmaI_plus_ZtWZ.noalias() += Zt_W_sqrt_rm * rand_vec_I_2_sim_post_.col(i);
					}
					else {
						rand_vec_pred_SigmaI_plus_ZtWZ = SigmaI_diag_sqrt.asDiagonal() * rand_vec_I_sim_post_.col(i) + Zt_W_sqrt_rm * rand_vec_I_2_sim_post_.col(i);
					}
					if (rand_vec_sim_post_calculated_ && reuse_rand_vec_I_sim_post_) {
						rand_vec_pred_SigmaI_plus_ZtWZ_inv = rand_vec_sim_post_.col(i);
					}
					else {
						rand_vec_pred_SigmaI_plus_ZtWZ_inv.setZero();
					}
					//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
					bool has_NA_or_Inf = false;
					int num_cg_steps_dummy;
					CGRandomEffectsVec(SigmaI_plus_ZtWZ_rm_, rand_vec_pred_SigmaI_plus_ZtWZ, rand_vec_pred_SigmaI_plus_ZtWZ_inv, has_NA_or_Inf, cg_max_num_it_, cg_delta_conv_pred_,
						true, ZERO_RHS_CG_THRESHOLD, false, cg_preconditioner_type_, L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps_dummy);
					if (has_NA_or_Inf) {
						na_inf_flag_6 = true;
					}
					rand_vec_sim_post_.col(i) = rand_vec_pred_SigmaI_plus_ZtWZ_inv;
				}//end parallel loop
			}
			if (na_inf_flag_6) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
//				//alernative version with multiple memory allocation
//#pragma omp parallel for schedule(static)
//				for (int i = 0; i < num_rand_vec_sim_post_; ++i) {
//					//z_i ~ N(0,(Sigma^(-1) + Z^T W Z))
//					vec_t rand_vec_pred_SigmaI_plus_ZtWZ;
//					if (has_vecchia_gp) {
//						rand_vec_pred_SigmaI_plus_ZtWZ = vec_t(dim_mode_);
//						rand_vec_pred_SigmaI_plus_ZtWZ.segment(0, dim_re_group) = SigmaI_diag_sqrt.asDiagonal() * rand_vec_I_sim_post_.col(i).segment(0, dim_re_group);
//						rand_vec_pred_SigmaI_plus_ZtWZ.segment(dim_re_group, dim_gp) = B_t_D_inv_sqrt * rand_vec_I_sim_post_.col(i).segment(dim_re_group, dim_gp);
//						rand_vec_pred_SigmaI_plus_ZtWZ.noalias() += Zt_W_sqrt_rm * rand_vec_I_2_sim_post_.col(i);
//					}
//					else {
//						rand_vec_pred_SigmaI_plus_ZtWZ = SigmaI_diag_sqrt.asDiagonal() * rand_vec_I_sim_post_.col(i) + Zt_W_sqrt_rm * rand_vec_I_2_sim_post_.col(i);
//					}
//					vec_t rand_vec_pred_SigmaI_plus_ZtWZ_inv;
//					if (rand_vec_sim_post_calculated_ && reuse_rand_vec_I_sim_post_) {
//						rand_vec_pred_SigmaI_plus_ZtWZ_inv = rand_vec_sim_post_.col(i);
//					}
//					else {
//						rand_vec_pred_SigmaI_plus_ZtWZ_inv = vec_t(dim_mode_);
//						rand_vec_pred_SigmaI_plus_ZtWZ_inv.setZero();
//					}
//					//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
//					bool has_NA_or_Inf = false;
//					int num_cg_steps_dummy;
//					CGRandomEffectsVec(SigmaI_plus_ZtWZ_rm_, rand_vec_pred_SigmaI_plus_ZtWZ, rand_vec_pred_SigmaI_plus_ZtWZ_inv, has_NA_or_Inf, cg_max_num_it_, cg_delta_conv_pred_,
//						true, ZERO_RHS_CG_THRESHOLD, false, cg_preconditioner_type_, L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps_dummy);
//					if (has_NA_or_Inf) {
//						Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_);
//					}
//					rand_vec_sim_post_.col(i) = rand_vec_pred_SigmaI_plus_ZtWZ_inv;
//				}//end parallel loop
		}//end matrix_inversion_method_ == "iterative"
		SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean();
		rand_vec_sim_post_calculated_ = true;
	}//end Sample_Posterior_LaplaceApprox_GroupedRE

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::Sample_Posterior_LaplaceApprox_OnlyOneGroupedRE() {
		CHECK(num_sets_re_ == 1);
		//sample iid normal random vectors
		if (!sampled_rand_vec_I_sim_post_) {
			rand_vec_I_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_sim_post_);
			rand_vec_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			if (reuse_rand_vec_I_sim_post_) {
				sampled_rand_vec_I_sim_post_ = true;
			}
		}//end !sampled_rand_vec_I_sim_post_
		CHECK(rand_vec_I_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_I_sim_post_.rows() == dim_mode_);
		CHECK(rand_vec_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_sim_post_.rows() == dim_mode_);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_rand_vec_sim_post_; ++i) {
			rand_vec_sim_post_.col(i) = (rand_vec_I_sim_post_.col(i).array() / diag_SigmaI_plus_ZtWZ_.array().sqrt()).matrix();
		}
		SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean();
		rand_vec_sim_post_calculated_ = true;
	}//end Sample_Posterior_LaplaceApprox_OnlyOneGroupedRE

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::Sample_Posterior_LaplaceApprox_Vecchia(const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i) {
		CHECK(num_sets_re_ == 1);
		//sample iid normal random vectors
		if (!sampled_rand_vec_I_sim_post_) {
			rand_vec_I_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_sim_post_);
			rand_vec_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			if (matrix_inversion_method_ == "iterative") {
				rand_vec_I_2_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
				GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_2_sim_post_);
			}
			if (reuse_rand_vec_I_sim_post_) {
				sampled_rand_vec_I_sim_post_ = true;
			}
		}//end !sampled_rand_vec_I_sim_post_
		CHECK(rand_vec_I_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_I_sim_post_.rows() == dim_mode_);
		CHECK(rand_vec_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_sim_post_.rows() == dim_mode_);
		if (matrix_inversion_method_ == "cholesky") {
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, den_mat_t, den_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, rand_vec_I_sim_post_, rand_vec_sim_post_, true);
		}
		else if (matrix_inversion_method_ == "iterative") {
			CHECK(rand_vec_I_2_sim_post_.cols() == num_rand_vec_sim_post_);
			CHECK(rand_vec_I_2_sim_post_.rows() == dim_mode_);
			if (HasNegativeValueInformationLogLik()) {
				Log::REFatal("Sample_Posterior_LaplaceApprox_Vecchia: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
					"Posterior sampling with iterative matrix inversion requires the square root of W ");
			}
			vec_t W_diag_sqrt = information_ll_.cwiseSqrt();
			sp_mat_rm_t B_t_D_inv_sqrt_rm = B_rm_.transpose() * (D_inv_rm_.cwiseSqrt());
			bool na_inf_flag_7 = false;
#pragma omp parallel for schedule(static) reduction(||:na_inf_flag_7)
			for (int i = 0; i < num_rand_vec_sim_post_; ++i) {
				//z_i ~ N(0,(Sigma^{-1} + W))
				vec_t rand_vec_pred_SigmaI_plus_W = B_t_D_inv_sqrt_rm * rand_vec_I_sim_post_.col(i) + W_diag_sqrt.cwiseProduct(rand_vec_I_2_sim_post_.col(i));
				vec_t rand_vec_pred_SigmaI_plus_W_inv;
				if (rand_vec_sim_post_calculated_ && reuse_rand_vec_I_sim_post_) {
					rand_vec_pred_SigmaI_plus_W_inv = rand_vec_sim_post_.col(i);
				}
				else {
					rand_vec_pred_SigmaI_plus_W_inv = vec_t(dim_mode_);
					rand_vec_pred_SigmaI_plus_W_inv.setZero();
				}
				//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
				bool has_NA_or_Inf = false;
				Inv_SigmaI_plus_ZtWZ_Vecchia_iterative_given_PC(cg_max_num_it_, re_comps_cross_cov_cluster_i, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, false, has_NA_or_Inf);
				if (has_NA_or_Inf) {
					na_inf_flag_7 = true;
				}
				rand_vec_sim_post_.col(i) = rand_vec_pred_SigmaI_plus_W_inv;
			}//end parallel loop
			if (na_inf_flag_7) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
		}//end matrix_inversion_method_ == "iterative"
		SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean();
		rand_vec_sim_post_calculated_ = true;
	}//end SamplePosterior_LaplaceApprox_Vecchia

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::Sample_Posterior_LaplaceApprox_FSVA(const den_mat_t* cross_cov,
		const den_mat_t& Bt_D_inv_B_cross_cov,
		const den_mat_t& sigma_woodbury,
		const chol_den_mat_t& chol_fact_sigma_woodbury,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_woodbury_2,
		const den_mat_t& chol_ip_cross_cov,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i) {
		CHECK(num_sets_re_ == 1);
		int num_ip = (int)(*cross_cov).cols();
		//sample iid normal random vectors
		if (!sampled_rand_vec_I_sim_post_) {
			rand_vec_I_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_sim_post_);
			rand_vec_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			rand_vec_I_2_sim_post_.resize(num_ip, num_rand_vec_sim_post_);
			GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_2_sim_post_);
			if (matrix_inversion_method_ == "iterative") {
				rand_vec_I_3_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
				GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_3_sim_post_);
			}
			if (reuse_rand_vec_I_sim_post_) {
				sampled_rand_vec_I_sim_post_ = true;
			}
		}//end !sampled_rand_vec_I_sim_post_
		CHECK(rand_vec_I_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_I_sim_post_.rows() == dim_mode_);
		CHECK(rand_vec_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_sim_post_.rows() == dim_mode_);
		CHECK(rand_vec_I_2_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_I_2_sim_post_.rows() == num_ip);
		if (matrix_inversion_method_ == "cholesky") {
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, den_mat_t, den_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, rand_vec_I_sim_post_, rand_vec_sim_post_, true);
			den_mat_t rand_vec_aux(num_ip, num_rand_vec_sim_post_);
			TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_sigma_woodbury_2, rand_vec_I_2_sim_post_, rand_vec_aux, true);
			den_mat_t rand_vec_aux_2 = Bt_D_inv_B_cross_cov * rand_vec_aux;
			den_mat_t rand_vec_aux_3 = chol_fact_SigmaI_plus_ZtWZ_vecchia_.solve(rand_vec_aux_2);
			rand_vec_sim_post_ += rand_vec_aux_3;
		}
		else if (matrix_inversion_method_ == "iterative") {
			CHECK(rand_vec_I_3_sim_post_.cols() == num_rand_vec_sim_post_);
			CHECK(rand_vec_I_3_sim_post_.rows() == dim_mode_);
			if (HasNegativeValueInformationLogLik()) {
				Log::REFatal("Sample_Posterior_LaplaceApprox_FSVA: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
					"Posterior sampling with iterative matrix inversion requires the square root of W ");
			}
			vec_t W_diag_sqrt = information_ll_.cwiseSqrt();
			vec_t D_sqrt = D_inv_rm_.diagonal().cwiseInverse().cwiseSqrt();
			vec_t W_D_inv, W_D_inv_inv, information_ll_inv;
			const den_mat_t* cross_cov_preconditioner = nullptr;
			if (cg_preconditioner_type_ == "vifdu" || cg_preconditioner_type_ == "none") {
				W_D_inv = (information_ll_ + D_inv_rm_.diagonal());
				W_D_inv_inv = W_D_inv.cwiseInverse();
				den_mat_t B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov = W_D_inv_inv.cwiseSqrt().asDiagonal() * D_inv_B_cross_cov_;
				sigma_woodbury_woodbury_ = sigma_woodbury - B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov.transpose() * B_t_D_inv_W_D_inv_inv_D_inv_B_cross_cov;
				chol_fact_sigma_woodbury_woodbury_.compute(sigma_woodbury_woodbury_);
				CheckCholeskyFactorization(chol_fact_sigma_woodbury_woodbury_, "Laplace posterior-sampling Woodbury matrix");
				cross_cov_preconditioner = nullptr;
			}
			else if (cg_preconditioner_type_ == "fitc") {
				if (HasZeroValueInformationLogLik()) {
					Log::REFatal("Sample_Posterior_LaplaceApprox_FSVA: 0's found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
						"Posterior sampling with the iterative FITC preconditioner requires W to be invertible. Try using another preconditioner or the Cholesky decomposition ");
				}
				information_ll_inv.resize(dim_mode_);
				information_ll_inv.array() = information_ll_.array().inverse();
				cross_cov_preconditioner = re_comps_cross_cov_preconditioner_cluster_i[0]->GetSigmaPtr();
			}
			bool na_inf_flag_8 = false;
#pragma omp parallel for schedule(static) reduction(||:na_inf_flag_8)
			for (int i = 0; i < num_rand_vec_sim_post_; ++i) {
				vec_t rand_vec_pred_SigmaI_plus_W_inv;
				//z_i ~ N(0,Sigma) (not possible to sample directly from Sigma^{-1})
				vec_t Sigma_sqrt_rand_vec = chol_ip_cross_cov.transpose() * rand_vec_I_2_sim_post_.col(i);
				Sigma_sqrt_rand_vec += B_rm_.triangularView<Eigen::UpLoType::UnitLower>().solve(D_sqrt.cwiseProduct(rand_vec_I_sim_post_.col(i)));
				//z_i ~ N(0,Sigma^{-1})
				vec_t Bt_D_inv_Sigma_sqrt_rand_vec = B_t_D_inv_rm_ * (B_rm_ * Sigma_sqrt_rand_vec);
				vec_t Sigma_inv_Sigma_sqrt_rand_vec = Bt_D_inv_Sigma_sqrt_rand_vec - Bt_D_inv_B_cross_cov * chol_fact_sigma_woodbury.solve((*cross_cov).transpose() * Bt_D_inv_Sigma_sqrt_rand_vec);
				//z_i ~ N(0,(Sigma^{-1} + W))
				vec_t rand_vec_pred_SigmaI_plus_W = Sigma_inv_Sigma_sqrt_rand_vec + W_diag_sqrt.cwiseProduct(rand_vec_I_3_sim_post_.col(i));
				//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
				bool has_NA_or_Inf = false;
				if (cg_preconditioner_type_ == "vifdu" || cg_preconditioner_type_ == "none") {
					CGFVIFLaplaceVec(information_ll_, B_rm_, B_t_D_inv_rm_, chol_fact_sigma_woodbury, cross_cov, W_D_inv_inv,
						chol_fact_sigma_woodbury_woodbury_, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, has_NA_or_Inf, cg_max_num_it_,
						true, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, true);
				}
				else if (cg_preconditioner_type_ == "fitc") {
					vec_t rand_vec_pred_SigmaI_plus_W_inv_interim(dim_mode_);
					vec_t rhs_part1 = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(rand_vec_pred_SigmaI_plus_W);
					vec_t rhs_part = D_inv_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve(rhs_part1);
					vec_t rhs_part2 = (*cross_cov) * (chol_fact_sigma_ip.solve((*cross_cov).transpose() * rand_vec_pred_SigmaI_plus_W));
					rand_vec_pred_SigmaI_plus_W = rhs_part + rhs_part2;
					CGVIFLaplace_Version_SigmaPlusWinvVec(information_ll_inv, D_inv_B_rm_, B_rm_, chol_fact_woodbury_preconditioner_,
						chol_ip_cross_cov, cross_cov_preconditioner, diagonal_approx_inv_preconditioner_, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv_interim, has_NA_or_Inf,
						cg_max_num_it_, true, cg_delta_conv_pred_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, true);
					rand_vec_pred_SigmaI_plus_W_inv = information_ll_inv.asDiagonal() * rand_vec_pred_SigmaI_plus_W_inv_interim;
				}
				if (has_NA_or_Inf) {
					na_inf_flag_8 = true;
				}
				rand_vec_sim_post_.col(i) = rand_vec_pred_SigmaI_plus_W_inv;
			}//end parallel loop
			if (na_inf_flag_8) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
		}//end matrix_inversion_method_ == "iterative"
		SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean();
		rand_vec_sim_post_calculated_ = true;
	}//end SamplePosterior_LaplaceApprox_FSVA

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::Sample_Posterior_LaplaceApprox_FITC(const den_mat_t* cross_cov,
		const vec_t& fitc_resid_diag) {
		CHECK(num_sets_re_ == 1);
		vec_t DW_plus_I_inv_diag = (information_ll_.array() * fitc_resid_diag.array() + 1.).matrix().cwiseInverse();
		vec_t D_div_DW_plus_I_sqrt = (DW_plus_I_inv_diag.array() * fitc_resid_diag.array()).sqrt().matrix();// = W^-1*(1-1/(DW+1))
		int num_ip = (int)(*cross_cov).cols();
		//sample iid normal random vectors
		if (!sampled_rand_vec_I_sim_post_) {
			rand_vec_I_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_sim_post_);
			rand_vec_I_2_sim_post_.resize(num_ip, num_rand_vec_sim_post_);
			GenRandVecNormalParallel(seed_rand_vec_trace_, cg_generator_counter_, rand_vec_I_2_sim_post_);
			rand_vec_sim_post_.resize(dim_mode_, num_rand_vec_sim_post_);
			if (reuse_rand_vec_I_sim_post_) {
				sampled_rand_vec_I_sim_post_ = true;
			}
		}//end !sampled_rand_vec_I_sim_post_
		CHECK(rand_vec_I_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_I_sim_post_.rows() == dim_mode_);
		CHECK(rand_vec_I_2_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_I_2_sim_post_.rows() == num_ip);
		CHECK(rand_vec_sim_post_.cols() == num_rand_vec_sim_post_);
		CHECK(rand_vec_sim_post_.rows() == dim_mode_);
		den_mat_t rand_vec_aux(num_ip, num_rand_vec_sim_post_);
		TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_dense_Newton_, rand_vec_I_2_sim_post_, rand_vec_aux, true);// ~ sigma_ip + Sigma_nm^T * D_plus_WI_inv_diag * Sigma_nm
		rand_vec_sim_post_ = D_div_DW_plus_I_sqrt.asDiagonal() * rand_vec_I_sim_post_ + DW_plus_I_inv_diag.asDiagonal() * ((*cross_cov) * rand_vec_aux);
		SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean();
		rand_vec_sim_post_calculated_ = true;
	}//end Sample_Posterior_LaplaceApprox_FITC

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean() {
		if (!TwoNumbersAreEqual<double>(c_mult_sim_post_, 1.)) {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < num_rand_vec_sim_post_; ++j) {
				rand_vec_sim_post_.col(j) *= c_mult_sim_post_;
			}
		}
		// Add mean
		if (add_mean_sim_post_) {
#pragma omp parallel for schedule(static)
			for (int j = 0; j < num_rand_vec_sim_post_; ++j) {
				rand_vec_sim_post_.col(j) += mode_;
			}
		}
	}//end SamplePosterior_LaplaceApprox_ScaleCovariance_AddMean

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::PredictLaplaceApproxFITC(const double* y_data,
		const int* y_data_int,
		const double* fixed_effects,
		const std::shared_ptr<den_mat_t> sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const den_mat_t* cross_cov,
		const vec_t& fitc_resid_diag,
		const den_mat_t& cross_cov_pred_ip,
		bool has_fitc_correction,
		const sp_mat_t& fitc_resid_pred_obs,
		vec_t& pred_mean,
		T_mat& pred_cov,
		vec_t& pred_var,
		bool calc_pred_cov,
		bool calc_pred_var,
		bool calc_mode,
		bool GPU_use) {
		if (calc_mode) {// Calculate mode and Cholesky factor 
			double mll;//approximate marginal likelihood. This is a by-product that is not used here.
			FindModePostRandEffCalcMLLFITC(y_data, y_data_int, fixed_effects, sigma_ip, chol_fact_sigma_ip,
				cross_cov, fitc_resid_diag, mll, GPU_use);
		}
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		if (can_use_first_deriv_log_like_for_pred_mean_) {
			pred_mean = cross_cov_pred_ip * (chol_fact_sigma_ip.solve((*cross_cov).transpose() * first_deriv_ll_));
			if (has_fitc_correction) {
				pred_mean += fitc_resid_pred_obs * first_deriv_ll_;
			}
		}
		else {
			Log::REFatal("PredictLaplaceApproxFITC: prediction is not yet implemented for the 'fitc' approximation for the likelihood '%s' ", likelihood_type_.c_str());
		}

		if (calc_pred_cov || calc_pred_var) {
			if (use_variance_correction_for_prediction_) {
				Log::REFatal("PredictLaplaceApproxFITC: The variance correction is not yet implemented ");
			}
			den_mat_t woodburry_part_sqrt = cross_cov_pred_ip.transpose();
			sp_mat_t resid_obs_inv_resid_pred_obs_t;
			if (has_fitc_correction) {
				if (HasZeroValueInformationLogLik()) {
					Log::REFatal("PredictLaplaceApproxFITC: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood ");
				}
				vec_t D_plus_WI_inv_diag = (fitc_resid_diag + information_ll_.cwiseInverse()).cwiseInverse();
				resid_obs_inv_resid_pred_obs_t = D_plus_WI_inv_diag.asDiagonal() * (fitc_resid_pred_obs.transpose());
				woodburry_part_sqrt -= (*cross_cov).transpose() * resid_obs_inv_resid_pred_obs_t;
			}
			TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_dense_Newton_, woodburry_part_sqrt, woodburry_part_sqrt, false);
			if (calc_pred_cov) {
				T_mat Maux;
				ConvertTo_T_mat_FromDense<T_mat>(woodburry_part_sqrt.transpose() * woodburry_part_sqrt, Maux);
				pred_cov += Maux;
				if (has_fitc_correction) {
					den_mat_t diag_correction = fitc_resid_pred_obs * resid_obs_inv_resid_pred_obs_t;
					T_mat diag_correction_T_mat;
					ConvertTo_T_mat_FromDense<T_mat>(diag_correction, diag_correction_T_mat);
					pred_cov -= diag_correction_T_mat;
				}
			}//end calc_pred_cov
			if (calc_pred_var) {
#pragma omp parallel for schedule(static)
				for (int i = 0; i < (int)pred_mean.size(); ++i) {
					pred_var[i] += woodburry_part_sqrt.col(i).array().square().sum();
				}
				if (has_fitc_correction) {
#pragma omp parallel for schedule(static)
					for (int i = 0; i < (int)pred_mean.size(); ++i) {
						pred_var[i] -= fitc_resid_pred_obs.row(i).dot(resid_obs_inv_resid_pred_obs_t.col(i));
					}
				}
			}//end calc_pred_var
		}//end calc_pred_cov || calc_pred_var
	}//end PredictLaplaceApproxFITC

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcVarLaplaceApproxOnlyOneGPCalculationsOnREScale(const std::shared_ptr<T_mat>& Sigma,
		vec_t& pred_var) {
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		pred_var = vec_t(dim_mode_);
		vec_t diag_ZtWZ_sqrt(information_ll_.size());
		if (HasNegativeValueInformationLogLik()) {
			Log::REFatal("CalcVarLaplaceApproxOnlyOneGPCalculationsOnREScale: Negative values found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
				"Cannot have negative values when using the numerically stable version of Rasmussen and Williams (2006) for mode finding ");
		}
		diag_ZtWZ_sqrt.array() = information_ll_.array().sqrt();
		T_mat L_inv_ZtWZ_sqrt_Sigma = diag_ZtWZ_sqrt.asDiagonal() * (*Sigma);
		TriangularSolveGivenCholesky<T_chol, T_mat, T_mat, T_mat>(chol_fact_Id_plus_Wsqrt_Sigma_Wsqrt_, L_inv_ZtWZ_sqrt_Sigma, L_inv_ZtWZ_sqrt_Sigma, false);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < dim_mode_; ++i) {
			pred_var[i] = (*Sigma).coeff(i, i) - L_inv_ZtWZ_sqrt_Sigma.col(i).squaredNorm();
		}
	}//end CalcVarLaplaceApproxOnlyOneGPCalculationsOnREScale

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcVarLaplaceApproxGroupedRE(vec_t& pred_var) {
		CHECK(num_sets_re_ == 1);
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		pred_var = vec_t(dim_mode_);
		if (matrix_inversion_method_ == "iterative") {
			pred_var = vec_t::Zero(dim_mode_);
			//Variance reduction
			sp_mat_rm_t P_sqrt_invt_rm;
			vec_t varred_global, c_cov, c_var;
			if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
				varred_global = vec_t::Zero(dim_mode_);
				c_cov = vec_t::Zero(dim_mode_);
				c_var = vec_t::Zero(dim_mode_);
				//Calculate P^(-0.5) explicitly
				sp_mat_rm_t Identity_rm(dim_mode_, dim_mode_);
				Identity_rm.setIdentity();
				if (cg_preconditioner_type_ == "incomplete_cholesky") {
					TriangularSolve<sp_mat_rm_t, sp_mat_rm_t, sp_mat_rm_t>(L_SigmaI_plus_ZtWZ_rm_, Identity_rm, P_sqrt_invt_rm, true);
				}
				else {
					TriangularSolve<sp_mat_rm_t, sp_mat_rm_t, sp_mat_rm_t>(P_SSOR_L_D_sqrt_inv_rm_, Identity_rm, P_sqrt_invt_rm, true);
				}
			}
			int num_threads;
#ifdef _OPENMP
			num_threads = omp_get_max_threads();
#else
			num_threads = 1;
#endif
			std::uniform_int_distribution<> unif(0, 2147483646);
			std::vector<RNG_t> parallel_rngs;
			for (int ig = 0; ig < num_threads; ++ig) {
				int seed_local = unif(cg_generator_);
				parallel_rngs.push_back(RNG_t(seed_local));
			}
			bool na_inf_flag_9 = false;
#pragma omp parallel
			{
				int thread_nb;
#ifdef _OPENMP
				thread_nb = omp_get_thread_num();
#else
				thread_nb = 0;
#endif
				RNG_t rng_local = parallel_rngs[thread_nb];
				vec_t pred_var_private = vec_t::Zero(dim_mode_);
				vec_t varred_private;
				vec_t c_cov_private;
				vec_t c_var_private;
				//Variance reduction
				if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
					varred_private = vec_t::Zero(dim_mode_);
					c_cov_private = vec_t::Zero(dim_mode_);
					c_var_private = vec_t::Zero(dim_mode_);
				}
#pragma omp for reduction(||:na_inf_flag_9)
				for (int i = 0; i < nsim_var_pred_; ++i) {
					//RV - Rademacher
					std::uniform_real_distribution<double> udist(0.0, 1.0);
					vec_t rand_vec_init(dim_mode_);
					double u;
					for (int j = 0; j < dim_mode_; j++) {
						u = udist(rng_local);
						if (u > 0.5) {
							rand_vec_init(j) = 1.;
						}
						else {
							rand_vec_init(j) = -1.;
						}
					}
					//Part 2: (Sigma^(-1) + Z^T W Z)^(-1) RV
					vec_t MInv_RV(dim_mode_);
					bool has_NA_or_Inf = false;
					int num_cg_steps_dummy;
					CGRandomEffectsVec(SigmaI_plus_ZtWZ_rm_, rand_vec_init, MInv_RV, has_NA_or_Inf,
						cg_max_num_it_, cg_delta_conv_pred_, true, ZERO_RHS_CG_THRESHOLD, true, cg_preconditioner_type_,
						L_SigmaI_plus_ZtWZ_rm_, P_SSOR_L_D_sqrt_inv_rm_, SigmaI_plus_ZtWZ_inv_diag_, num_cg_steps_dummy);
					if (has_NA_or_Inf) {
						na_inf_flag_9 = true;
					}
					//Part 2: RV o (Sigma^(-1) + Z^T W Z)^(-1) RV
					pred_var_private += MInv_RV.cwiseProduct(rand_vec_init);
					//Variance reduction
					if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
						//Stochastic: P^(-0.5T) P^(-0.5) RV
						vec_t P_sqrt_inv_RV = P_sqrt_invt_rm.transpose() * rand_vec_init;
						vec_t rand_vec_varred = P_sqrt_invt_rm * P_sqrt_inv_RV;
						varred_private += rand_vec_varred.cwiseProduct(rand_vec_init);
						c_cov_private += varred_private.cwiseProduct(pred_var_private);
						c_var_private += varred_private.cwiseProduct(varred_private);
					}
				} //end for loop
#pragma omp critical
				{
					pred_var += pred_var_private;
					//Variance reduction
					if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
						varred_global += varred_private;
						c_cov += c_cov_private;
						c_var += c_var_private;
					}
				}
			} //end #pragma omp parallel
			if (na_inf_flag_9) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
			pred_var /= nsim_var_pred_;
			//Variance reduction
			if (cg_preconditioner_type_ == "incomplete_cholesky" || cg_preconditioner_type_ == "ssor") {
				varred_global /= nsim_var_pred_;
				c_cov /= nsim_var_pred_;
				c_var /= nsim_var_pred_;
				//Deterministic: diag(P^(-0.5T) P^(-0.5))
				vec_t varred_determ = P_sqrt_invt_rm.cwiseProduct(P_sqrt_invt_rm) * vec_t::Ones(dim_mode_);
				//optimal c
				c_cov -= varred_global.cwiseProduct(pred_var);
				c_var -= varred_global.cwiseProduct(varred_global);
				vec_t c_opt = c_cov.array() / c_var.array();
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < c_opt.size(); ++i) {
					if (c_var.coeffRef(i) == 0) {
						c_opt[i] = 1;
					}
				}
				pred_var += c_opt.cwiseProduct(varred_determ - varred_global);
			}
		} //end iterative
		else { //begin Cholesky
			sp_mat_t L_inv(dim_mode_, dim_mode_);
			L_inv.setIdentity();
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_grouped_, L_inv, L_inv, false);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < dim_mode_; ++i) {
				pred_var[i] = L_inv.col(i).squaredNorm();
			}
		} //end Cholesky
	}//end CalcVarLaplaceApproxGroupedRE

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcVarLaplaceApproxOnlyOneGroupedRECalculationsOnREScale(vec_t& pred_var) {
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		pred_var = vec_t(dim_mode_);
		pred_var.array() = diag_SigmaI_plus_ZtWZ_.array().inverse();
	}//end CalcVarLaplaceApproxOnlyOneGroupedRECalculationsOnREScale

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcVarLaplaceApproxVecchia(vec_t& pred_var,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i) {
		if (na_or_inf_during_last_call_to_find_mode_) {
			Log::REFatal(NA_OR_INF_ERROR_);
		}
		CHECK(mode_has_been_calculated_);
		pred_var = vec_t(dim_mode_);
		//Version Simulation
		if (matrix_inversion_method_ == "iterative") {
			CHECK(num_sets_re_ == 1);
			pred_var = vec_t::Zero(dim_mode_);
			if (HasNegativeValueInformationLogLik()) {
				Log::REFatal("CalcVarLaplaceApproxVecchia: Negative values found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
					"Cannot have negative values when using 'iterative' methods for predictive variances in Vecchia-Laplace approximations ");
			}
			vec_t W_diag_sqrt = information_ll_.cwiseSqrt();
			sp_mat_rm_t B_t_D_inv_sqrt_rm = B_rm_.transpose() * (D_inv_rm_.cwiseSqrt());
			int num_threads;
#ifdef _OPENMP
			num_threads = omp_get_max_threads();
#else
			num_threads = 1;
#endif
			std::uniform_int_distribution<> unif(0, 2147483646);
			std::vector<RNG_t> parallel_rngs;
			for (int ig = 0; ig < num_threads; ++ig) {
				int seed_local = unif(cg_generator_);
				parallel_rngs.push_back(RNG_t(seed_local));
			}
			bool na_inf_flag_10 = false;
#pragma omp parallel
			{
				int thread_nb;
#ifdef _OPENMP
				thread_nb = omp_get_thread_num();
#else
				thread_nb = 0;
#endif
				RNG_t rng_local = parallel_rngs[thread_nb];
				vec_t pred_var_private = vec_t::Zero(dim_mode_);
#pragma omp for reduction(||:na_inf_flag_10)
				for (int i = 0; i < nsim_var_pred_; ++i) {
					//z_i ~ N(0,I)
					std::normal_distribution<double> ndist(0.0, 1.0);
					vec_t rand_vec_pred_I_1(dim_mode_), rand_vec_pred_I_2(dim_mode_);
					for (int j = 0; j < dim_mode_; j++) {
						rand_vec_pred_I_1(j) = ndist(rng_local);
						rand_vec_pred_I_2(j) = ndist(rng_local);
					}
					//z_i ~ N(0,(Sigma^{-1} + W))
					vec_t rand_vec_pred_SigmaI_plus_W = B_t_D_inv_sqrt_rm * rand_vec_pred_I_1 + W_diag_sqrt.cwiseProduct(rand_vec_pred_I_2);
					vec_t rand_vec_pred_SigmaI_plus_W_inv(dim_mode_);
					//z_i ~ N(0,(Sigma^{-1} + W)^{-1})
					bool has_NA_or_Inf = false;
					Inv_SigmaI_plus_ZtWZ_Vecchia_iterative_given_PC(cg_max_num_it_, re_comps_cross_cov_cluster_i, rand_vec_pred_SigmaI_plus_W, rand_vec_pred_SigmaI_plus_W_inv, true, has_NA_or_Inf);
					if (has_NA_or_Inf) {
						na_inf_flag_10 = true;
					}
					pred_var_private += rand_vec_pred_SigmaI_plus_W_inv.cwiseProduct(rand_vec_pred_SigmaI_plus_W_inv);
				}// end for loop
#pragma omp critical
				{
					pred_var += pred_var_private;
				}
			}// end #pragma omp parallel
			if (na_inf_flag_10) { Log::REDebug(CG_NA_OR_INF_WARNING_SAMPLE_POSTERIOR_); }
			pred_var /= nsim_var_pred_;
		} //end Version Simulation
		else {
			sp_mat_t L_inv(dim_mode_, dim_mode_);
			L_inv.setIdentity();
			TriangularSolveGivenCholesky<chol_sp_mat_t, sp_mat_t, sp_mat_t, sp_mat_t>(chol_fact_SigmaI_plus_ZtWZ_vecchia_, L_inv, L_inv, false);
#pragma omp parallel for schedule(static)
			for (int i = 0; i < dim_mode_; ++i) {
				pred_var[i] = L_inv.col(i).squaredNorm();
			}
		}
	}//end CalcVarLaplaceApproxVecchia

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcLogDetStochFSVA(const data_size_t& num_data,
		const int& cg_max_num_it_tridiag,
		const chol_den_mat_t& chol_fact_sigma_woodbury,
		const den_mat_t& chol_ip_cross_cov,
		const chol_den_mat_t& chol_fact_sigma_ip,
		const chol_den_mat_t& chol_fact_sigma_ip_preconditioner,
		const den_mat_t* cross_cov,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_preconditioner_cluster_i,
		const vec_t& W_D_inv_inv,
		const chol_den_mat_t& chol_fact_sigma_woodbury_woodbury,
		const vec_t& W_D_inv,
		bool& has_NA_or_Inf,
		double& log_det_Sigma_W_plus_I) {
		log_det_Sigma_W_plus_I = 0.;
		CHECK(rand_vec_trace_I_.cols() == num_rand_vec_trace_);
		std::vector<vec_t> Tdiags_W_SigmaI(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag));
		std::vector<vec_t> Tsubdiags_W_SigmaI(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag - 1));
		if (cg_preconditioner_type_ == "fitc") {
			const den_mat_t* cross_cov_preconditioner = re_comps_cross_cov_preconditioner_cluster_i[0]->GetSigmaPtr();
			if (HasNegativeValueInformationLogLik()) {
				Log::REFatal("CalcLogDetStochFSVA: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
					"Stochastic evaluation of the Laplace determinant with the 'fitc' preconditioner requires log(W) ");
			}
			if (HasZeroValueInformationLogLik()) {
				Log::REFatal("CalcLogDetStochFSVA: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
					"This is not permitted for the VIF approximation and iterative methods with the '%s' preconditioner. Try using the 'vifdu' preconditioner or the Cholesky decomposition ", cg_preconditioner_type_.c_str());
			}
			CGTridiagVIFLaplace_Version_SigmaPlusWinv(information_ll_.cwiseInverse(), D_inv_B_rm_, B_rm_, chol_fact_woodbury_preconditioner_,
				chol_ip_cross_cov, cross_cov_preconditioner, diagonal_approx_inv_preconditioner_, rand_vec_trace_I_, Tdiags_W_SigmaI, Tsubdiags_W_SigmaI, SigmaI_plus_W_inv_Z_,
				has_NA_or_Inf, num_data, num_rand_vec_trace_, cg_max_num_it_tridiag, cg_delta_conv_, cg_preconditioner_type_);
		}
		else {
			CGTridiagVIFLaplace(information_ll_, B_rm_, B_t_D_inv_rm_, chol_fact_sigma_woodbury, cross_cov, W_D_inv_inv, chol_fact_sigma_woodbury_woodbury,
				rand_vec_trace_I_, Tdiags_W_SigmaI, Tsubdiags_W_SigmaI, SigmaI_plus_W_inv_Z_, has_NA_or_Inf, num_data, num_rand_vec_trace_, cg_max_num_it_tridiag, cg_delta_conv_,
				cg_preconditioner_type_);
		}
		//'Tdiags_W_SigmaI' / 'Tsubdiags_W_SigmaI' are only fully valid if the CG did not find an
		//	NA or Inf. The caller sets the log-likelihood to NA in that case anyway
		if (!has_NA_or_Inf) {
			LogDetStochTridiag(Tdiags_W_SigmaI, Tsubdiags_W_SigmaI, log_det_Sigma_W_plus_I, num_data, num_rand_vec_trace_);
			if (cg_preconditioner_type_ == "fitc") {
				log_det_Sigma_W_plus_I -= 2. * (((den_mat_t)chol_fact_sigma_ip_preconditioner.matrixL()).diagonal().array().log().sum());
				log_det_Sigma_W_plus_I += information_ll_.array().log().sum();
				log_det_Sigma_W_plus_I += 2. * ((den_mat_t)chol_fact_woodbury_preconditioner_.matrixL()).diagonal().array().log().sum();
				log_det_Sigma_W_plus_I += diagonal_approx_preconditioner_.array().log().sum();
			}
			else {
				log_det_Sigma_W_plus_I -= 2. * (((den_mat_t)chol_fact_sigma_ip.matrixL()).diagonal().array().log().sum()) + D_inv_rm_.diagonal().array().log().sum();
				if (cg_preconditioner_type_ == "vifdu") {
					log_det_Sigma_W_plus_I += W_D_inv.array().log().sum() + 2. * ((den_mat_t)chol_fact_sigma_woodbury_woodbury.matrixL()).diagonal().array().log().sum();
				}
				else {
					log_det_Sigma_W_plus_I += 2. * ((den_mat_t)chol_fact_sigma_woodbury.matrixL()).diagonal().array().log().sum();
				}
			}
		}
	}//end CalcLogDetStochFSVA

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::Inv_SigmaI_plus_ZtWZ_Vecchia_iterative(int cg_max_num_it,
		den_mat_t& I_k_plus_Sigma_L_kt_W_Sigma_L_k,
		const sp_mat_t& SigmaI,
		sp_mat_t& SigmaI_plus_W,
		const sp_mat_t& B,
		bool& has_NA_or_Inf,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		data_size_t cluster_i,
		REModelTemplate<T_mat, T_chol>* re_model,
		const vec_t& rhs,
		vec_t& SigmaI_plus_ZtWZ_inv_rhs,
		bool initialize_to_zero,
		bool calculate_preconditioners) {
		if (cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc" || cg_preconditioner_type_ == "vecchia_response") {
			if (calculate_preconditioners && HasNegativeValueInformationLogLik()) {
				Log::REFatal("Inv_SigmaI_plus_ZtWZ_Vecchia_iterative: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
					"The '%s' preconditioner is based on a positive-definite W^(-1) formulation. Try another preconditioner (e.g. 'vadu') or the Cholesky decomposition ", cg_preconditioner_type_.c_str());
			}
			if ((information_ll_.array() > 1e10).any() && calculate_preconditioners) {
				has_NA_or_Inf = true;// the inversion of the preconditioner with the Woodbury identity can be numerically unstable when information_ll_ is very large
			}
			else {
				const den_mat_t* cross_cov = nullptr;
				if (cg_preconditioner_type_ == "fitc") {
					cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
				}
				if (calculate_preconditioners) {
					if (cg_preconditioner_type_ == "pivoted_cholesky") {
						I_k_plus_Sigma_L_kt_W_Sigma_L_k.setIdentity();
						I_k_plus_Sigma_L_kt_W_Sigma_L_k += Sigma_L_k_.transpose() * information_ll_.asDiagonal() * Sigma_L_k_;
						chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.compute(I_k_plus_Sigma_L_kt_W_Sigma_L_k);
						CheckCholeskyFactorization(chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, "Vecchia iterative pivoted-Cholesky preconditioner");
					}
					else if (cg_preconditioner_type_ == "fitc") {
						if (HasZeroValueInformationLogLik()) {
							Log::REFatal("Inv_SigmaI_plus_ZtWZ_Vecchia_iterative: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
								"This is not permitted for the Vecchia approximation and iterative methods with the '%s' preconditioner. Try using another preconditioner (e.g. 'vadu') or the Cholesky decomposition ", cg_preconditioner_type_.c_str());
						}
						diagonal_approx_preconditioner_ = information_ll_.cwiseInverse();
						diagonal_approx_preconditioner_.array() += sigma_ip_stable_.coeffRef(0, 0);
#pragma omp parallel for schedule(static)
						for (int ii = 0; ii < diagonal_approx_preconditioner_.size(); ++ii) {
							diagonal_approx_preconditioner_[ii] -= chol_ip_cross_cov_.col(ii).array().square().sum();
						}
						diagonal_approx_inv_preconditioner_ = diagonal_approx_preconditioner_.cwiseInverse();
						den_mat_t sigma_woodbury;
						sigma_woodbury = (*cross_cov).transpose() * (diagonal_approx_inv_preconditioner_.asDiagonal() * (*cross_cov));
						sigma_woodbury += sigma_ip_stable_;
						chol_fact_woodbury_preconditioner_.compute(sigma_woodbury);
						CheckCholeskyFactorization(chol_fact_woodbury_preconditioner_, "Vecchia iterative FITC preconditioner");
					}
					else if (cg_preconditioner_type_ == "vecchia_response") {
						sp_mat_t B_vecchia;
						if (HasZeroValueInformationLogLik()) {
							Log::REFatal("Inv_SigmaI_plus_ZtWZ_Vecchia_iterative: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
								"This is not permitted for the Vecchia approximation and iterative methods with the '%s' preconditioner. Try using another preconditioner (e.g. 'vadu') or the Cholesky decomposition ", cg_preconditioner_type_.c_str());
						}
						vec_t pseudo_nugget = information_ll_.cwiseInverse();
						re_model->CalcVecchiaApproxLatentAddDiagonal(cluster_i, B_vecchia, D_inv_vecchia_pc_, pseudo_nugget.data());
						B_vecchia_pc_rm_ = sp_mat_rm_t(B_vecchia);
					}
				}//end calculate_preconditioners
				CGVecchiaLaplace_Version_SigmaPlusWinvVec(information_ll_, B_rm_, B_t_D_inv_rm_.transpose(), rhs, SigmaI_plus_ZtWZ_inv_rhs, has_NA_or_Inf,
					cg_max_num_it, initialize_to_zero, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_,
					chol_fact_woodbury_preconditioner_, cross_cov, diagonal_approx_inv_preconditioner_, B_vecchia_pc_rm_, D_inv_vecchia_pc_, false);
			}
		}//end cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc"
		else if (cg_preconditioner_type_ == "vadu" || cg_preconditioner_type_ == "incomplete_cholesky") {
			if (calculate_preconditioners) {
				if (cg_preconditioner_type_ == "vadu") {
					D_inv_plus_W_B_rm_ = (D_inv_rm_.diagonal() + information_ll_).asDiagonal() * B_rm_;
				}
				else if (cg_preconditioner_type_ == "incomplete_cholesky") {
					SigmaI_plus_W = SigmaI;
					SigmaI_plus_W.diagonal().array() += information_ll_.array();
					ReverseIncompleteCholeskyFactorization(SigmaI_plus_W, B, L_SigmaI_plus_W_rm_);
				}
			}//end calculate_preconditioners
			CGVecchiaLaplaceVec(information_ll_, B_rm_, B_t_D_inv_rm_, rhs, SigmaI_plus_ZtWZ_inv_rhs, has_NA_or_Inf,
				cg_max_num_it, initialize_to_zero, cg_delta_conv_, ZERO_RHS_CG_THRESHOLD, cg_preconditioner_type_, D_inv_plus_W_B_rm_, L_SigmaI_plus_W_rm_, false);
		}
		else {
			Log::REFatal("Inv_SigmaI_plus_ZtWZ_Vecchia_iterative: Preconditioner type '%s' is not supported ", cg_preconditioner_type_.c_str());
		}
	}//end Inv_SigmaI_plus_ZtWZ_Vecchia_iterative

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::Inv_SigmaI_plus_ZtWZ_Vecchia_iterative_given_PC(int cg_max_num_it,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		const vec_t& rhs,
		vec_t& SigmaI_plus_ZtWZ_inv_rhs,
		bool initialize_to_zero,
		bool& has_NA_or_Inf) {
		den_mat_t d1{};
		sp_mat_t  d2{}, d3{}, d4{};
		REModelTemplate<T_mat, T_chol>* model = nullptr;
		Inv_SigmaI_plus_ZtWZ_Vecchia_iterative(cg_max_num_it, d1, d2, d3, d4, has_NA_or_Inf,
			re_comps_cross_cov_cluster_i, 0, model, rhs, SigmaI_plus_ZtWZ_inv_rhs, initialize_to_zero, false);
	}//end Inv_SigmaI_plus_ZtWZ_Vecchia_iterative_given_PC

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcLogDetStochVecchia(const data_size_t& num_data,
		const int& cg_max_num_it_tridiag,
		den_mat_t& I_k_plus_Sigma_L_kt_W_Sigma_L_k,
		const sp_mat_t& SigmaI,
		sp_mat_t& SigmaI_plus_W,
		const sp_mat_t& B,
		bool& has_NA_or_Inf,
		double& log_det_Sigma_W_plus_I,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		data_size_t cluster_i,
		REModelTemplate<T_mat, T_chol>* re_model) {
		CHECK(rand_vec_trace_I_.cols() == num_rand_vec_trace_);
		CHECK(rand_vec_trace_P_.cols() == num_rand_vec_trace_);
		if (cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc" || cg_preconditioner_type_ == "vecchia_response") {
			if (HasNegativeValueInformationLogLik()) {
				Log::REFatal("CalcLogDetStochVecchia: Negative values found in W (the diagonal Hessian or Fisher information of the negative log-likelihood). "
					"Stochastic evaluation of the Laplace determinant with the '%s' preconditioner requires W to be positive ", cg_preconditioner_type_.c_str());
			}
			if (HasZeroValueInformationLogLik()) {
				Log::REFatal("CalcLogDetStochVecchia: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
					"This is not permitted when using the Vecchia approximation and iterative methods with the '%s' preconditioner. Try using another preconditioner (e.g. 'vadu') or the Cholesky decomposition ", cg_preconditioner_type_.c_str());
			}
			const den_mat_t* cross_cov = nullptr;
			std::vector<vec_t> Tdiags_PI_WI_plus_Sigma(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag));
			std::vector<vec_t> Tsubdiags_PI_WI_plus_Sigma(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag - 1));
			if (cg_preconditioner_type_ == "pivoted_cholesky") {
				CHECK(rand_vec_trace_I2_.cols() == num_rand_vec_trace_);
				CHECK(rand_vec_trace_I2_.rows() == Sigma_L_k_.cols());
				//Get random vectors (z_1, ..., z_t) with Cov(z_i) = P:
				//For P = W^(-1) + Sigma_L_k Sigma_L_k^T: z_i = W^(-1/2) r_j + Sigma_L_k r_i, where r_i, r_j ~ N(0,I)
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					rand_vec_trace_P_.col(i) = Sigma_L_k_ * rand_vec_trace_I2_.col(i) + ((information_ll_.cwiseInverse().cwiseSqrt()).array() * rand_vec_trace_I_.col(i).array()).matrix();
				}
				if (information_changes_after_mode_finding_) {
					I_k_plus_Sigma_L_kt_W_Sigma_L_k.setIdentity();
					I_k_plus_Sigma_L_kt_W_Sigma_L_k += Sigma_L_k_.transpose() * information_ll_.asDiagonal() * Sigma_L_k_;
					chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.compute(I_k_plus_Sigma_L_kt_W_Sigma_L_k);
					CheckCholeskyFactorization(chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, "Vecchia iterative pivoted-Cholesky determinant preconditioner");
				}
			}
			else if (cg_preconditioner_type_ == "fitc") {
				CHECK(rand_vec_trace_I2_.cols() == num_rand_vec_trace_);
				CHECK(rand_vec_trace_I2_.rows() == chol_ip_cross_cov_.rows());
				cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
				//Get random vectors (z_1, ..., z_t) with Cov(z_i) = P:
				//For P = W^(-1) + chol_ip_cross_cov^T chol_ip_cross_cov: z_i = W^(-1/2) r_j + chol_ip_cross_cov^T r_i, where r_i, r_j ~ N(0,I)
				if (information_changes_after_mode_finding_) {
					diagonal_approx_preconditioner_ = information_ll_.cwiseInverse();
					diagonal_approx_preconditioner_.array() += sigma_ip_stable_.coeffRef(0, 0);
#pragma omp parallel for schedule(static)
					for (int ii = 0; ii < diagonal_approx_preconditioner_.size(); ++ii) {
						diagonal_approx_preconditioner_[ii] -= chol_ip_cross_cov_.col(ii).array().square().sum();
					}
					diagonal_approx_inv_preconditioner_ = diagonal_approx_preconditioner_.cwiseInverse();
					den_mat_t sigma_woodbury;
					sigma_woodbury = (*cross_cov).transpose() * (diagonal_approx_inv_preconditioner_.asDiagonal() * (*cross_cov));
					sigma_woodbury += sigma_ip_stable_;
					chol_fact_woodbury_preconditioner_.compute(sigma_woodbury);
					CheckCholeskyFactorization(chol_fact_woodbury_preconditioner_, "Vecchia iterative FITC determinant preconditioner");
				}
				rand_vec_trace_P_ = chol_ip_cross_cov_.transpose() * rand_vec_trace_I2_ + diagonal_approx_preconditioner_.cwiseSqrt().asDiagonal() * rand_vec_trace_I_;
			}
			else if (cg_preconditioner_type_ == "vecchia_response") {
				//For P = B^T D^(-1) B: z_i = B^T D^(-0.5) r_i, where r_i ~ N(0,I)					
				if (information_changes_after_mode_finding_) {
					sp_mat_t B_vecchia;
					vec_t pseudo_nugget = information_ll_.cwiseInverse();
					re_model->CalcVecchiaApproxLatentAddDiagonal(cluster_i, B_vecchia, D_inv_vecchia_pc_, pseudo_nugget.data());
					B_vecchia_pc_rm_ = sp_mat_rm_t(B_vecchia);
				}
				vec_t D_sqrt_vecchia_pc = D_inv_vecchia_pc_.diagonal().cwiseInverse().cwiseSqrt();
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					rand_vec_trace_P_.col(i) = (B_vecchia_pc_rm_.template triangularView<Eigen::UpLoType::UnitLower>()).solve(D_sqrt_vecchia_pc.asDiagonal() * (rand_vec_trace_I_.col(i)));
				}
			}
			CGTridiagVecchiaLaplace_Version_SigmaPlusWinv(information_ll_, B_rm_, B_t_D_inv_rm_.transpose(), rand_vec_trace_P_, Tdiags_PI_WI_plus_Sigma, Tsubdiags_PI_WI_plus_Sigma,
				WI_plus_Sigma_inv_Z_, has_NA_or_Inf, num_data, num_rand_vec_trace_, cg_max_num_it_tridiag, cg_delta_conv_, cg_preconditioner_type_,
				chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_, chol_fact_woodbury_preconditioner_, cross_cov, diagonal_approx_inv_preconditioner_, B_vecchia_pc_rm_, D_inv_vecchia_pc_);
			if (!has_NA_or_Inf) {
				double ldet_PI_WI_plus_Sigma;
				LogDetStochTridiag(Tdiags_PI_WI_plus_Sigma, Tsubdiags_PI_WI_plus_Sigma, ldet_PI_WI_plus_Sigma, num_data, num_rand_vec_trace_);
				//log|Sigma W + I| = log|P^(-1) (W^(-1) + Sigma)| + log|W| + log|P|
				log_det_Sigma_W_plus_I = ldet_PI_WI_plus_Sigma + information_ll_.array().log().sum();
				if (cg_preconditioner_type_ == "pivoted_cholesky") {
					// log|P| = log|I_k + Sigma_L_k^T W Sigma_L_k| + log|W^(-1)| + log|I_k|, log|I_k| = 0
					log_det_Sigma_W_plus_I += 2 * ((den_mat_t)chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.matrixL()).diagonal().array().log().sum() - information_ll_.array().log().sum();
				}
				else if (cg_preconditioner_type_ == "fitc") {
					// log|P| = log|Woodburry| - log|Sigma_m| - log|D^-1|
					log_det_Sigma_W_plus_I += 2. * ((den_mat_t)chol_fact_woodbury_preconditioner_.matrixL()).diagonal().array().log().sum() -
						2. * (((den_mat_t)chol_fact_sigma_ip_.matrixL()).diagonal().array().log().sum()) -
						diagonal_approx_inv_preconditioner_.array().log().sum();
				}
				else if (cg_preconditioner_type_ == "vecchia_response") {
					log_det_Sigma_W_plus_I -= D_inv_vecchia_pc_.diagonal().array().log().sum();
				}
			}
		}//end cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc" || cg_preconditioner_type_ == "vecchia_response"
		else if (cg_preconditioner_type_ == "vadu" || cg_preconditioner_type_ == "incomplete_cholesky") {
			vec_t D_inv_plus_W_diag;
			std::vector<vec_t> Tdiags_PI_SigmaI_plus_W(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag));
			std::vector<vec_t> Tsubdiags_PI_SigmaI_plus_W(num_rand_vec_trace_, vec_t(cg_max_num_it_tridiag - 1));
			//Get random vectors (z_1, ..., z_t) with Cov(z_i) = P:
			if (cg_preconditioner_type_ == "vadu") {
				//For P = B^T (D^(-1) + W) B: z_i = B^T (D^(-1) + W)^0.5 r_i, where r_i ~ N(0,I)
				D_inv_plus_W_diag = D_inv_rm_.diagonal() + information_ll_;
				sp_mat_rm_t B_t_D_inv_plus_W_sqrt_rm = B_rm_.transpose() * (D_inv_plus_W_diag).cwiseSqrt().asDiagonal();
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					rand_vec_trace_P_.col(i) = B_t_D_inv_plus_W_sqrt_rm * rand_vec_trace_I_.col(i);
				}
				//rand_vec_trace_P_ = B_rm_.transpose() * ((D_inv_rm_.diagonal() + information_ll_).cwiseSqrt().asDiagonal() * rand_vec_trace_I_);
				D_inv_plus_W_B_rm_ = (D_inv_plus_W_diag).asDiagonal() * B_rm_;
			}
			else if (cg_preconditioner_type_ == "incomplete_cholesky") {
				//Update P with latest W
				if (information_changes_after_mode_finding_) {
					SigmaI_plus_W = SigmaI;
					SigmaI_plus_W.diagonal().array() += information_ll_.array();
					ReverseIncompleteCholeskyFactorization(SigmaI_plus_W, B, L_SigmaI_plus_W_rm_);
				}
				//For P = L^T L: z_i = L^T r_i, where r_i ~ N(0,I)
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					rand_vec_trace_P_.col(i) = L_SigmaI_plus_W_rm_.transpose() * rand_vec_trace_I_.col(i);
				}
			}
			CGTridiagVecchiaLaplace(information_ll_, B_rm_, B_t_D_inv_rm_, rand_vec_trace_P_, Tdiags_PI_SigmaI_plus_W, Tsubdiags_PI_SigmaI_plus_W,
				SigmaI_plus_W_inv_Z_, has_NA_or_Inf, num_data, num_rand_vec_trace_, cg_max_num_it_tridiag, cg_delta_conv_, cg_preconditioner_type_, D_inv_plus_W_B_rm_, L_SigmaI_plus_W_rm_);
			if (!has_NA_or_Inf) {
				double ldet_PI_SigmaI_plus_W;
				LogDetStochTridiag(Tdiags_PI_SigmaI_plus_W, Tsubdiags_PI_SigmaI_plus_W, ldet_PI_SigmaI_plus_W, num_data, num_rand_vec_trace_);
				//log|Sigma W + I| = log|P^(-1) (Sigma^(-1) + W)| + log|P| + log|Sigma|
				log_det_Sigma_W_plus_I = ldet_PI_SigmaI_plus_W - D_inv_rm_.diagonal().array().log().sum();
				if (cg_preconditioner_type_ == "vadu") {
					//log|P| = log|B^T (D^(-1) + W) B| = log|(D^(-1) + W)|
					log_det_Sigma_W_plus_I += D_inv_plus_W_diag.array().log().sum();
				}
				else if (cg_preconditioner_type_ == "incomplete_cholesky") {
					//log|P| = log|L^T L|
					log_det_Sigma_W_plus_I += 2 * (L_SigmaI_plus_W_rm_.diagonal().array().log().sum());
				}
			}
		}
		else {
			Log::REFatal("CalcLogDetStochVecchia: Preconditioner type '%s' is not supported ", cg_preconditioner_type_.c_str());
		}
	}//end CalcLogDetStochVecchia

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcLogDetStochDerivModeVecchia(const vec_t& deriv_information_diag_loc_par,
		const data_size_t& num_data,
		vec_t& d_log_det_Sigma_W_plus_I_d_mode,
		vec_t& D_inv_plus_W_inv_diag,
		vec_t& diag_WI,
		den_mat_t& PI_Z,
		den_mat_t& WI_PI_Z,
		den_mat_t& WI_WI_plus_Sigma_inv_Z,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i,
		bool GPU_use) const {
		den_mat_t Z_PI_P_deriv_PI_Z;
		vec_t tr_PI_P_deriv_vec, c_opt;
		den_mat_t W_deriv_rep;
		if (grad_information_wrt_mode_non_zero_) {
			W_deriv_rep = deriv_information_diag_loc_par.replicate(1, num_rand_vec_trace_);
		}
		if (cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc" || cg_preconditioner_type_ == "vecchia_response") {
			const den_mat_t* cross_cov = nullptr;
			den_mat_t Z_WI_plus_Sigma_inv_WI_deriv_PI_Z;
			vec_t tr_WI_plus_Sigma_inv_WI_deriv;
			if (HasZeroValueInformationLogLik()) {
				Log::REFatal("CalcLogDetStochDerivModeVecchia: 0's found in the (diagonal) Hessian (or Fisher information) of the negative log-likelihood. "
					"This is not permitted when using the Vecchia approximation and iterative methods with the '%s' preconditioner. Try using another preconditioner (e.g. 'vadu') or the Cholesky decomposition ", cg_preconditioner_type_.c_str());
			}
			diag_WI = information_ll_.cwiseInverse();
			WI_WI_plus_Sigma_inv_Z = diag_WI.asDiagonal() * WI_plus_Sigma_inv_Z_;
			if (cg_preconditioner_type_ == "pivoted_cholesky") {
				//P^(-1) = (W^(-1) + Sigma_L_k Sigma_L_k^T)^(-1)
				//W^(-1) P^(-1) Z = Z - Sigma_L_k (I_k + Sigma_L_k^T W Sigma_L_k)^(-1) Sigma_L_k^T W Z
				den_mat_t Sigma_Lkt_W_Z;
				if (Sigma_L_k_.cols() < num_rand_vec_trace_) {
					Sigma_Lkt_W_Z = (Sigma_L_k_.transpose() * information_ll_.asDiagonal()) * rand_vec_trace_P_;
				}
				else {
					Sigma_Lkt_W_Z = Sigma_L_k_.transpose() * (information_ll_.asDiagonal() * rand_vec_trace_P_);
				}
				WI_PI_Z = rand_vec_trace_P_ - Sigma_L_k_ * chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.solve(Sigma_Lkt_W_Z);
			}
			else if (cg_preconditioner_type_ == "fitc") {
				//P^(-1) = (D + Sigma_nm Sigma_m^-1 Sigma_mn)^(-1)
				//W^(-1) P^(-1) Z
				cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
				den_mat_t D_rand_vec = diagonal_approx_inv_preconditioner_.asDiagonal() * rand_vec_trace_P_;
				WI_PI_Z = diag_WI.asDiagonal() * D_rand_vec -
					diag_WI.asDiagonal() * (diagonal_approx_inv_preconditioner_.asDiagonal() * ((*cross_cov) *
						chol_fact_woodbury_preconditioner_.solve((*cross_cov).transpose() * D_rand_vec)));
			}
			else if (cg_preconditioner_type_ == "vecchia_response") {
				WI_PI_Z = diag_WI.asDiagonal() * (B_vecchia_pc_rm_.transpose() * (D_inv_vecchia_pc_ * (B_vecchia_pc_rm_ * rand_vec_trace_P_)));
				//variance reduction currently not implemented
			}
			if (grad_information_wrt_mode_non_zero_) {
				CHECK(first_deriv_information_loc_par_caluclated_);
				Z_WI_plus_Sigma_inv_WI_deriv_PI_Z = -1 * (WI_WI_plus_Sigma_inv_Z.array() * W_deriv_rep.array() * WI_PI_Z.array()).matrix();
				tr_WI_plus_Sigma_inv_WI_deriv = Z_WI_plus_Sigma_inv_WI_deriv_PI_Z.rowwise().mean();
				d_log_det_Sigma_W_plus_I_d_mode = tr_WI_plus_Sigma_inv_WI_deriv;
			}
			//variance reduction
			if (cg_preconditioner_type_ == "pivoted_cholesky") {
				if (grad_information_wrt_mode_non_zero_) {
					//tr(W^(-1) dW/db_i) - do not cancel with deterministic part of variance reduction when using optimal c
					vec_t tr_WI_W_deriv = diag_WI.cwiseProduct(deriv_information_diag_loc_par);
					d_log_det_Sigma_W_plus_I_d_mode += tr_WI_W_deriv;
					//deterministic tr(Sigma_Lk (I_k + Sigma_Lk^T W Sigma_Lk)^(-1) Sigma_Lk^T dW/db_i) + tr(W dW^(-1)/db_i) (= - tr(W^(-1) dW/db_i))
					den_mat_t L_inv_Sigma_L_kt(Sigma_L_k_.cols(), num_data);
					TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_, Sigma_L_k_.transpose(), L_inv_Sigma_L_kt, false);
					den_mat_t L_inv_Sigma_L_kt_sqr = L_inv_Sigma_L_kt.cwiseProduct(L_inv_Sigma_L_kt);
					vec_t Sigma_Lk_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_Lkt_diag = L_inv_Sigma_L_kt_sqr.transpose() * vec_t::Ones(L_inv_Sigma_L_kt_sqr.rows()); //diagonal of Sigma_Lk (I_k + Sigma_Lk^T W Sigma_Lk)^(-1) Sigma_Lk^T
					vec_t tr_Sigma_Lk_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_Lkt_W_deriv = Sigma_Lk_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_Lkt_diag.array() * deriv_information_diag_loc_par.array();
					//stochastic tr(P^(-1) dP/db_i), where dP/db_i = - W^(-1) dW/db_i W^(-1)
					Z_PI_P_deriv_PI_Z = -1 * (WI_PI_Z.array() * W_deriv_rep.array() * WI_PI_Z.array()).matrix();
					tr_PI_P_deriv_vec = Z_PI_P_deriv_PI_Z.rowwise().mean();
					//optimal c
					CalcOptimalCVectorized(Z_WI_plus_Sigma_inv_WI_deriv_PI_Z, Z_PI_P_deriv_PI_Z, tr_WI_plus_Sigma_inv_WI_deriv, tr_PI_P_deriv_vec, c_opt);
					d_log_det_Sigma_W_plus_I_d_mode += c_opt.cwiseProduct(tr_Sigma_Lk_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_Lkt_W_deriv - tr_WI_W_deriv) - c_opt.cwiseProduct(tr_PI_P_deriv_vec);
				}
			}
			else if (cg_preconditioner_type_ == "fitc") {
				if (grad_information_wrt_mode_non_zero_) {
					//tr(W^(-1) dW/db_i) - do not cancel with deterministic part of variance reduction when using optimal c
					vec_t tr_WI_W_deriv = diag_WI.cwiseProduct(deriv_information_diag_loc_par);
					d_log_det_Sigma_W_plus_I_d_mode += tr_WI_W_deriv;
					//-tr(W^-1P^-1W^(-1) dW/db_i)
					vec_t tr_WI_DI_WI_W_deriv = diag_WI.cwiseProduct(tr_WI_W_deriv.cwiseProduct(diagonal_approx_inv_preconditioner_));
					vec_t tr_WI_DI_WI_DI_W_deriv = diagonal_approx_inv_preconditioner_.cwiseProduct(tr_WI_DI_WI_W_deriv);
					den_mat_t chol_wood_cross_cov((*cross_cov).cols(), num_data);
					//TriangularSolveGivenCholesky<chol_den_mat_t, den_mat_t, den_mat_t, den_mat_t>(chol_fact_woodbury_preconditioner_, (*cross_cov).transpose(), chol_wood_cross_cov, false);
					GPBoost::solve_lower_triangular(chol_fact_woodbury_preconditioner_, (*cross_cov).transpose(), chol_wood_cross_cov, GPU_use);
					vec_t tr_WI_PI_WI_W_deriv(num_data);
#pragma omp parallel for schedule(static)  
					for (int i = 0; i < num_data; ++i) {
						tr_WI_PI_WI_W_deriv[i] = chol_wood_cross_cov.col(i).array().square().sum() * tr_WI_DI_WI_DI_W_deriv[i];
					}
					//stochastic tr(P^(-1) dP/db_i), where dP/db_i = - W^(-1) dW/db_i W^(-1)
					Z_PI_P_deriv_PI_Z = -1 * (WI_PI_Z.array() * W_deriv_rep.array() * WI_PI_Z.array()).matrix();
					tr_PI_P_deriv_vec = Z_PI_P_deriv_PI_Z.rowwise().mean();
					//optimal c
					CalcOptimalCVectorized(Z_WI_plus_Sigma_inv_WI_deriv_PI_Z, Z_PI_P_deriv_PI_Z, tr_WI_plus_Sigma_inv_WI_deriv, tr_PI_P_deriv_vec, c_opt);
					d_log_det_Sigma_W_plus_I_d_mode += c_opt.cwiseProduct(tr_WI_PI_WI_W_deriv - tr_WI_DI_WI_W_deriv) - c_opt.cwiseProduct(tr_PI_P_deriv_vec);
				}
			}
		}
		else if (cg_preconditioner_type_ == "vadu" || cg_preconditioner_type_ == "incomplete_cholesky") {
			//P^(-1) Z
			if (cg_preconditioner_type_ == "vadu") {
				//P^(-1) = B^(-1) (D^(-1) + W)^(-1) B^(-T)
				PI_Z.resize(num_data, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)  
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					vec_t B_invt_Z = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(rand_vec_trace_P_.col(i));
					PI_Z.col(i) = D_inv_plus_W_B_rm_.triangularView<Eigen::UpLoType::Lower>().solve(B_invt_Z);
				}
				//den_mat_t B_invt_Z(num_data, num_rand_vec_trace_);
				//TriangularSolve<sp_mat_rm_t, den_mat_t, den_mat_t>(B_rm_, rand_vec_trace_P_, B_invt_Z, true);//it seems that this is not faster (21.11.2024)
				//TriangularSolve<sp_mat_rm_t, den_mat_t, den_mat_t>(D_inv_plus_W_B_rm_, B_invt_Z, PI_Z, false);
			}
			else if (cg_preconditioner_type_ == "incomplete_cholesky") {
				//P^(-1) = L^(-1) L^(-T)
				PI_Z.resize(num_data, num_rand_vec_trace_);
#pragma omp parallel for schedule(static)   
				for (int i = 0; i < num_rand_vec_trace_; ++i) {
					vec_t L_invt_Z = (L_SigmaI_plus_W_rm_.transpose().template triangularView<Eigen::UpLoType::Upper>()).solve(rand_vec_trace_P_.col(i));
					PI_Z.col(i) = L_SigmaI_plus_W_rm_.triangularView<Eigen::UpLoType::Lower>().solve(L_invt_Z);
				}
			}
			den_mat_t Z_SigmaI_plus_W_inv_W_deriv_PI_Z;
			vec_t tr_SigmaI_plus_W_inv_W_deriv;
			if (grad_information_wrt_mode_non_zero_) {
				CHECK(first_deriv_information_loc_par_caluclated_);
				//stochastic tr((Sigma^(-1) + W)^(-1) dW/db_i)
				Z_SigmaI_plus_W_inv_W_deriv_PI_Z = (SigmaI_plus_W_inv_Z_.array() * W_deriv_rep.array() * PI_Z.array()).matrix();
				tr_SigmaI_plus_W_inv_W_deriv = Z_SigmaI_plus_W_inv_W_deriv_PI_Z.rowwise().mean();
				d_log_det_Sigma_W_plus_I_d_mode = tr_SigmaI_plus_W_inv_W_deriv;
			}
			if (cg_preconditioner_type_ == "vadu") {
				//variance reduction
				//deterministic tr((D^(-1) + W)^(-1) dW/db_i)
				D_inv_plus_W_inv_diag = (D_inv_rm_.diagonal() + information_ll_).cwiseInverse();
				if (grad_information_wrt_mode_non_zero_) {
					vec_t tr_D_inv_plus_W_inv_W_deriv = D_inv_plus_W_inv_diag.array() * deriv_information_diag_loc_par.array();
					//stochastic tr(P^(-1) dP/db_i), where dP/db_i = B^T dW/db_i B
					den_mat_t B_PI_Z = B_rm_ * PI_Z;
					Z_PI_P_deriv_PI_Z = (B_PI_Z.array() * W_deriv_rep.array() * B_PI_Z.array()).matrix();
					tr_PI_P_deriv_vec = Z_PI_P_deriv_PI_Z.rowwise().mean();
					//optimal c
					CalcOptimalCVectorized(Z_SigmaI_plus_W_inv_W_deriv_PI_Z, Z_PI_P_deriv_PI_Z, tr_SigmaI_plus_W_inv_W_deriv, tr_PI_P_deriv_vec, c_opt);
					d_log_det_Sigma_W_plus_I_d_mode += c_opt.cwiseProduct(tr_D_inv_plus_W_inv_W_deriv) - c_opt.cwiseProduct(tr_PI_P_deriv_vec);
				}
			}
		}
		else {
			Log::REFatal("CalcLogDetStochDerivMode: Preconditioner type '%s' is not supported ", cg_preconditioner_type_.c_str());
		}
	} //end CalcLogDetStochDerivModeVecchia

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcLogDetStochDerivCovParVecchia(const data_size_t& num_data,
		const int& num_comps_total,
		const int& j,
		const sp_mat_rm_t& SigmaI_deriv_rm,
		const sp_mat_t& B_grad_j,
		const sp_mat_t& D_grad_j,
		const vec_t& D_inv_plus_W_inv_diag,
		const den_mat_t& PI_Z,
		const den_mat_t& WI_PI_Z,
		double& d_log_det_Sigma_W_plus_I_d_cov_pars) const {
		if (cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc" || cg_preconditioner_type_ == "vecchia_response") {
			den_mat_t Sigma_WI_plus_Sigma_inv_Z(num_data, num_rand_vec_trace_);
			den_mat_t B_invt_PI_Z(num_data, num_rand_vec_trace_), Sigma_PI_Z(num_data, num_rand_vec_trace_);
			//Stochastic Trace: Calculate tr((Sigma + W^(-1))^(-1) dSigma/dtheta_j)
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < num_rand_vec_trace_; ++i) {
				vec_t B_invt_WI_plus_Sigma_inv_Z = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(WI_plus_Sigma_inv_Z_.col(i));
				Sigma_WI_plus_Sigma_inv_Z.col(i) = (B_t_D_inv_rm_.transpose().template triangularView<Eigen::UpLoType::Lower>()).solve(B_invt_WI_plus_Sigma_inv_Z);
			}
			den_mat_t PI_Z_local = information_ll_.asDiagonal() * WI_PI_Z;
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < num_rand_vec_trace_; ++i) {
				B_invt_PI_Z.col(i) = (B_rm_.transpose().template triangularView<Eigen::UpLoType::UnitUpper>()).solve(PI_Z_local.col(i));
			}
#pragma omp parallel for schedule(static)   
			for (int i = 0; i < num_rand_vec_trace_; ++i) {
				Sigma_PI_Z.col(i) = (B_t_D_inv_rm_.transpose().template triangularView<Eigen::UpLoType::Lower>()).solve(B_invt_PI_Z.col(i));
			}
			d_log_det_Sigma_W_plus_I_d_cov_pars = -1 * ((Sigma_WI_plus_Sigma_inv_Z.cwiseProduct(SigmaI_deriv_rm * Sigma_PI_Z)).colwise().sum()).mean();
		}
		else if (cg_preconditioner_type_ == "vadu" || cg_preconditioner_type_ == "incomplete_cholesky") {
			//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dSigma^(-1)/dtheta_j)
			vec_t zt_SigmaI_plus_W_inv_SigmaI_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(SigmaI_deriv_rm * PI_Z)).colwise().sum()).transpose();
			double tr_SigmaI_plus_W_inv_SigmaI_deriv = zt_SigmaI_plus_W_inv_SigmaI_deriv_PI_z.mean();
			d_log_det_Sigma_W_plus_I_d_cov_pars = tr_SigmaI_plus_W_inv_SigmaI_deriv;
			//tr(Sigma^(-1) dSigma/dtheta_j)
			if (num_comps_total == 1 && j == 0) {
				d_log_det_Sigma_W_plus_I_d_cov_pars += num_data;
			}
			else {
				d_log_det_Sigma_W_plus_I_d_cov_pars += (D_inv_rm_.diagonal().array() * D_grad_j.diagonal().array()).sum();
			}
			if (cg_preconditioner_type_ == "vadu") {
				//variance reduction
				double tr_D_inv_plus_W_inv_D_inv_deriv = 0., tr_PI_P_deriv = 0.;
				vec_t zt_PI_P_deriv_PI_z;
				if (num_comps_total == 1 && j == 0) {
					//dD/dsigma2 = D and dB/dsigma2 = 0
					//deterministic tr((D^(-1) + W)^(-1) dD^(-1)/dsigma2), where dD^(-1)/dsigma2 = -D^(-1)
					tr_D_inv_plus_W_inv_D_inv_deriv = -1 * (D_inv_plus_W_inv_diag.array() * D_inv_rm_.diagonal().array()).sum();
					//stochastic tr(P^(-1) dP/dsigma2), where dP/dsigma2 = -Sigma^(-1)
					zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(SigmaI_deriv_rm * PI_Z)).colwise().sum()).transpose();
					tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				}
				else {
					//deterministic tr((D^(-1) + W)^(-1) dD^(-1)/dtheta_j)
					tr_D_inv_plus_W_inv_D_inv_deriv = -1 * (D_inv_plus_W_inv_diag.array() * D_inv_rm_.diagonal().array() * D_grad_j.diagonal().array() * D_inv_rm_.diagonal().array()).sum();
					//stochastic tr(P^(-1) dP/dtheta_j)
					sp_mat_rm_t Bt_W_Bgrad_rm = B_rm_.transpose() * information_ll_.asDiagonal() * B_grad_j;
					sp_mat_rm_t P_deriv_rm = SigmaI_deriv_rm + sp_mat_rm_t(Bt_W_Bgrad_rm.transpose()) + Bt_W_Bgrad_rm;
					zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(P_deriv_rm * PI_Z)).colwise().sum()).transpose();
					tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				}
				//optimal c
				double c_opt;
				CalcOptimalC(zt_SigmaI_plus_W_inv_SigmaI_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_SigmaI_deriv, tr_PI_P_deriv, c_opt);
				d_log_det_Sigma_W_plus_I_d_cov_pars += c_opt * tr_D_inv_plus_W_inv_D_inv_deriv - c_opt * tr_PI_P_deriv;
			}
		}
		else {
			Log::REFatal("CalcLogDetStochDerivCovPar: Preconditioner type '%s' is not supported ", cg_preconditioner_type_.c_str());
		}
	} //end CalcLogDetStochDerivCovParVecchia

	template <typename T_mat, typename T_chol>
	void Likelihood<T_mat, T_chol>::CalcLogDetStochDerivAuxParVecchia(const vec_t& deriv_information_aux_par,
		const vec_t& D_inv_plus_W_inv_diag,
		const vec_t& diag_WI,
		const den_mat_t& PI_Z,
		const den_mat_t& WI_PI_Z,
		const den_mat_t& WI_WI_plus_Sigma_inv_Z,
		double& d_detmll_d_aux_par,
		const std::vector<std::shared_ptr<RECompGP<den_mat_t>>>& re_comps_cross_cov_cluster_i) const {
		double tr_PI_P_deriv, c_opt;
		vec_t zt_PI_P_deriv_PI_z;
		if (cg_preconditioner_type_ == "pivoted_cholesky" || cg_preconditioner_type_ == "fitc" || cg_preconditioner_type_ == "vecchia_response") {
			//Stochastic Trace: Calculate tr((Sigma + W^(-1))^(-1) dW^(-1)/daux)
			vec_t zt_WI_plus_Sigma_inv_WI_deriv_PI_z = -1 * ((WI_WI_plus_Sigma_inv_Z.cwiseProduct(deriv_information_aux_par.asDiagonal() * WI_PI_Z)).colwise().sum()).transpose();
			double tr_WI_plus_Sigma_inv_WI_deriv = zt_WI_plus_Sigma_inv_WI_deriv_PI_z.mean();
			d_detmll_d_aux_par = tr_WI_plus_Sigma_inv_WI_deriv;
			//variance reduction
			if (cg_preconditioner_type_ == "pivoted_cholesky") {
				//tr(W^(-1) dW/daux) - do not cancel with deterministic part of variance reduction when using optimal c
				double tr_WI_W_deriv = (diag_WI.cwiseProduct(deriv_information_aux_par)).sum();
				d_detmll_d_aux_par += tr_WI_W_deriv;
				//variance reduction
				//deterministic tr((I_k + Sigma_Lk^T W Sigma_Lk)^(-1) Sigma_Lk^T dW/daux Sigma_Lk) + tr(W dW^(-1)/daux) (= - tr(W^(-1) dW/daux))
				den_mat_t Sigma_L_kt_W_deriv_Sigma_L_k = Sigma_L_k_.transpose() * deriv_information_aux_par.asDiagonal() * Sigma_L_k_;
				double tr_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_L_kt_W_deriv_Sigma_L_k = (chol_fact_I_k_plus_Sigma_L_kt_W_Sigma_L_k_vecchia_.solve(Sigma_L_kt_W_deriv_Sigma_L_k)).diagonal().sum();
				//stochastic tr(P^(-1) dP/daux), where dP/daux = - W^(-1) dW/daux W^(-1)
				zt_PI_P_deriv_PI_z = -1 * ((WI_PI_Z.cwiseProduct(deriv_information_aux_par.asDiagonal() * WI_PI_Z)).colwise().sum()).transpose();
				tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				//optimal c
				CalcOptimalC(zt_WI_plus_Sigma_inv_WI_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_WI_plus_Sigma_inv_WI_deriv, tr_PI_P_deriv, c_opt);
				d_detmll_d_aux_par += c_opt * (tr_I_k_plus_Sigma_L_kt_W_Sigma_L_k_inv_Sigma_L_kt_W_deriv_Sigma_L_k - tr_WI_W_deriv) - c_opt * tr_PI_P_deriv;
			}
			else if (cg_preconditioner_type_ == "fitc") {
				const den_mat_t* cross_cov = re_comps_cross_cov_cluster_i[0]->GetSigmaPtr();
				//tr(W^(-1) dW/daux) - do not cancel with deterministic part of variance reduction when using optimal c
				vec_t tr_WI_W_deriv_vec = diag_WI.cwiseProduct(deriv_information_aux_par);
				double tr_WI_W_deriv = tr_WI_W_deriv_vec.sum();
				d_detmll_d_aux_par += tr_WI_W_deriv;
				//-tr(W^-1P^-1W^(-1) dW/daux)
				vec_t tr_WI_DI_WI_W_deriv_vec = diag_WI.cwiseProduct(tr_WI_W_deriv_vec.cwiseProduct(diagonal_approx_inv_preconditioner_));
				double tr_WI_DI_WI_W_deriv = tr_WI_DI_WI_W_deriv_vec.sum();
				vec_t tr_WI_DI_WI_DI_W_deriv_vec = diagonal_approx_inv_preconditioner_.cwiseProduct(tr_WI_DI_WI_W_deriv_vec);
				den_mat_t woodI_cross_covT_WI_DI_WI_DI_W_deri_cross_cov = chol_fact_woodbury_preconditioner_.solve((*cross_cov).transpose() * (tr_WI_DI_WI_DI_W_deriv_vec.asDiagonal() * (*cross_cov)));
				double tr_WI_PI_WI_W_deriv = woodI_cross_covT_WI_DI_WI_DI_W_deri_cross_cov.diagonal().sum();
				//stochastic tr(P^(-1) dP/db_i), where dP/db_i = - W^(-1) dW/db_i W^(-1)
				zt_PI_P_deriv_PI_z = -1 * ((WI_PI_Z.cwiseProduct(deriv_information_aux_par.asDiagonal() * WI_PI_Z)).colwise().sum()).transpose();
				tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				//optimal c
				CalcOptimalC(zt_WI_plus_Sigma_inv_WI_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_WI_plus_Sigma_inv_WI_deriv, tr_PI_P_deriv, c_opt);
				d_detmll_d_aux_par += c_opt * (tr_WI_PI_WI_W_deriv - tr_WI_DI_WI_W_deriv) - c_opt * tr_PI_P_deriv;
			}
		}
		else if (cg_preconditioner_type_ == "vadu" || cg_preconditioner_type_ == "incomplete_cholesky") {
			//Stochastic Trace: Calculate tr((Sigma^(-1) + W)^(-1) dW/daux)
			vec_t zt_SigmaI_plus_W_inv_W_deriv_PI_z = ((SigmaI_plus_W_inv_Z_.cwiseProduct(deriv_information_aux_par.asDiagonal() * PI_Z)).colwise().sum()).transpose();
			double tr_SigmaI_plus_W_inv_W_deriv = zt_SigmaI_plus_W_inv_W_deriv_PI_z.mean();
			d_detmll_d_aux_par = tr_SigmaI_plus_W_inv_W_deriv;
			if (cg_preconditioner_type_ == "vadu") {
				//variance reduction
				//deterministic tr((D^(-1) + W)^(-1) dW/daux)
				double tr_D_inv_plus_W_inv_W_deriv = (D_inv_plus_W_inv_diag.array() * deriv_information_aux_par.array()).sum();
				//stochastic tr(P^(-1) dP/daux), where dP/daux = B^T dW/daux B
				sp_mat_rm_t P_deriv_rm = B_rm_.transpose() * deriv_information_aux_par.asDiagonal() * B_rm_;
				zt_PI_P_deriv_PI_z = ((PI_Z.cwiseProduct(P_deriv_rm * PI_Z)).colwise().sum()).transpose();
				tr_PI_P_deriv = zt_PI_P_deriv_PI_z.mean();
				//optimal c
				CalcOptimalC(zt_SigmaI_plus_W_inv_W_deriv_PI_z, zt_PI_P_deriv_PI_z, tr_SigmaI_plus_W_inv_W_deriv, tr_PI_P_deriv, c_opt);
				d_detmll_d_aux_par += c_opt * tr_D_inv_plus_W_inv_W_deriv - c_opt * tr_PI_P_deriv;
			}
		}
		else {
			Log::REFatal("CalcLogDetStochDerivAuxPar: Preconditioner type '%s' is not supported ", cg_preconditioner_type_.c_str());
		}
	} //end CalcLogDetStochDerivAuxParVecchia
}  // namespace GPBoost

#endif   // GPB_LIKELIHOODS_LAPLACE_H_
