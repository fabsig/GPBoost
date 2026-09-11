// Checks for the conjugate gradient stopping rules in 'CG_utils'.
//
// These cover cases that cannot be constructed through the R or Python interface, because they need
// a specific matrix or right-hand side: a column that is solved exactly while others still iterate,
// a right-hand side whose norm overflows or underflows, and a tolerance that the arithmetic cannot
// deliver. Build and run with
//
//   sh helpers/run_cg_convergence_check.sh
//
// Exits with a non-zero status if any check fails.
#include <GPBoost/CG_utils.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using namespace GPBoost;

static int failures = 0;

static void Check(bool ok, const char* what) {
	std::printf("%-78s %s\n", what, ok ? "ok" : "FAILED");
	if (!ok) {
		failures += 1;
	}
}

// A diagonally dominant symmetric matrix, so that it is SPD and the CG converges quickly
static sp_mat_rm_t BuildA(int n) {
	std::vector<Eigen::Triplet<double>> tr;
	for (int i = 0; i < n; ++i) {
		tr.emplace_back(i, i, 4. + 0.01 * i);
		if (i + 1 < n) {
			tr.emplace_back(i, i + 1, -1.);
			tr.emplace_back(i + 1, i, -1.);
		}
		if (i + 7 < n) {
			tr.emplace_back(i, i + 7, -0.5);
			tr.emplace_back(i + 7, i, -0.5);
		}
	}
	sp_mat_rm_t A(n, n);
	A.setFromTriplets(tr.begin(), tr.end());
	return A;
}

static den_mat_t RademacherLike(int n, int t) {
	den_mat_t Z(n, t);
	unsigned int state = 12345u;
	for (int j = 0; j < t; ++j) {
		for (int i = 0; i < n; ++i) {
			state = 1664525u * state + 1013904223u;
			Z(i, j) = ((state >> 16) & 1u) ? 1. : -1.;
		}
	}
	return Z;
}

int main() {
	const int n = 200, t = 12;
	const sp_mat_rm_t A = BuildA(n);
	const den_mat_t A_dense = den_mat_t(A);
	const vec_t diag_inv = A_dense.diagonal().cwiseInverse();
	sp_mat_rm_t unused(n, n);
	const den_mat_t rhs_mat = RademacherLike(n, t);
	const vec_t rhs_vec = vec_t::LinSpaced(n, -1., 2.);
	const vec_t exact_vec = A_dense.llt().solve(rhs_vec);

	auto solve_vec = [&](const vec_t& b, const CGConvergenceParams& cp, vec_t& u, bool initialize_to_zero,
		bool* nan_out, int* steps) {
		*nan_out = false;
		*steps = 0;
		CGRandomEffectsVec(A, b, u, *nan_out, n, 1e-10, initialize_to_zero, 1e-100, false, "diagonal",
			unused, unused, diag_inv, *steps, cp);
	};
	auto run_tridiag = [&](const sp_mat_rm_t& A_in, const vec_t& diag_inv_in, const den_mat_t& B,
		const CGConvergenceParams& cp, double* ldet, int* steps, std::vector<int>* depths, bool* nan_out) {
		const int tb = (int)B.cols();
		std::vector<vec_t> Td(tb, vec_t(n)), Ts(tb, vec_t(n - 1));
		den_mat_t U(n, tb);
		*nan_out = false;
		*steps = 0;
		CGTridiagRandomEffects(A_in, B, Td, Ts, U, *nan_out, n, tb, n, 1e-12, "diagonal",
			unused, unused, diag_inv_in, *steps, cp);
		LogDetStochTridiag(Td, Ts, *ldet, n, tb);
		depths->clear();
		for (int i = 0; i < tb; ++i) {
			depths->push_back((int)Td[i].size());
		}
		return U;
	};

	// ------------------------------------------------- a column that is solved exactly
	// A is block diagonal: the first block is 4*I, so with the "diagonal" preconditioner the
	// preconditioned operator is the identity there, the step size is exactly 1 and the residual
	// becomes exactly 0 after one step. Such a column has to stop being updated under every
	// aggregation rule, otherwise its next step size is a 0/0
	{
		const int nb = 8, tb = 3;
		std::vector<Eigen::Triplet<double>> tr;
		for (int i = 0; i < nb; ++i) {
			tr.emplace_back(i, i, 4.);//a power of two, so the division and multiplication are exact
		}
		for (int i = nb; i < n; ++i) {
			tr.emplace_back(i, i, 4.25);
			if (i + 1 < n) {
				tr.emplace_back(i, i + 1, -1.);
				tr.emplace_back(i + 1, i, -1.);
			}
		}
		sp_mat_rm_t A2(n, n);
		A2.setFromTriplets(tr.begin(), tr.end());
		const vec_t diag_inv2 = den_mat_t(A2).diagonal().cwiseInverse();
		den_mat_t rhs2(n, tb);
		rhs2.setZero();
		for (int i = 0; i < nb; ++i) {
			rhs2(i, 0) = (i % 2) ? 1. : -1.;
		}
		for (int i = nb; i < n; ++i) {
			rhs2(i, 1) = 1. + 0.01 * i;
			rhs2(i, 2) = (i % 3) ? 1. : -2.;
		}
		const char* rules[3] = { "average", "max", "per_rhs" };
		for (int r = 0; r < 3; ++r) {
			CGConvergenceParams cp;
			cp.delta_conv = 1e-12;
			cp.multi_rhs_convergence = rules[r];
			double ldet = 0.;
			int steps = 0;
			bool nan = false;
			std::vector<int> depths;
			const den_mat_t U = run_tridiag(A2, diag_inv2, rhs2, cp, &ldet, &steps, &depths, &nan);
			char msg[200];
			std::snprintf(msg, sizeof(msg), "a column solved exactly, '%s': no NaN", rules[r]);
			Check(!nan && U.allFinite() && std::isfinite(ldet), msg);
		}
	}

	// -------------------------------------------------------- non-finite right-hand sides
	{
		CGConvergenceParams cp;
		cp.criterion = "relative";
		cp.rel_tol = 1e-8;
		cp.abs_tol = 1e-8;
		vec_t u(n);
		bool nan = false;
		int steps = 0;
		vec_t b = rhs_vec;
		b(5) = std::numeric_limits<double>::quiet_NaN();
		solve_vec(b, cp, u, true, &nan, &steps);
		Check(nan, "a NaN rhs is reported instead of being answered with a zero solution");
		b = rhs_vec;
		b(5) = std::numeric_limits<double>::infinity();
		solve_vec(b, cp, u, true, &nan, &steps);
		Check(nan, "an infinite rhs is reported instead of being answered with a zero solution");
		//entries whose squares overflow: the plain norm is Inf although the true norm is finite
		b = vec_t::Constant(n, 1e200);
		solve_vec(b, cp, u, true, &nan, &steps);
		Check(std::isfinite(SafeNorm(b)), "a norm that overflows the sum of squares is computed safely");
	}

	// ------------------------------------------- very small right-hand sides and tolerances
	{
		//entries whose squares underflow: the plain norm is 0 although the vector is not zero
		const vec_t tiny = vec_t::Constant(n, 1e-200);
		Check(SafeNorm(tiny) > 0., "a norm that underflows the sum of squares is computed safely");

		//small but solvable, and the zero vector does not satisfy the tolerance: has to be solved
		CGConvergenceParams cp;
		cp.criterion = "relative";
		cp.rel_tol = 1e-8;
		cp.abs_tol = 1e-80;
		const vec_t b = vec_t::Constant(n, 1e-60);
		vec_t u(n);
		bool nan = false;
		int steps = 0;
		solve_vec(b, cp, u, true, &nan, &steps);
		const vec_t exact_small = A_dense.llt().solve(b);
		Check(!nan && steps > 0 && (u - exact_small).norm() / exact_small.norm() < 1e-6,
			"a very small rhs is solved instead of being written off as negligible");

		//the same, but now the arithmetic cannot deliver the requested tolerance: has to be reported
		cp.abs_tol = 1e-300;
		const vec_t b2 = vec_t::Constant(n, 1e-160);
		solve_vec(b2, cp, u, true, &nan, &steps);
		Check(nan, "a tolerance the arithmetic cannot deliver is reported, not called converged");

		//and the zero vector really being accurate enough is still a legitimate early return
		cp.abs_tol = 1e-8;
		solve_vec(b2, cp, u, true, &nan, &steps);
		Check(!nan && u.isZero(0) && steps == 0,
			"a rhs below the tolerance returns the zero solution without iterating");
	}

	// ------------- a residual below the cutoff is only a success when the tolerance is met
	{
		//the zero solution leaves a residual of ||b||, which here is far above the requested
		//	tolerance, so giving up on the recursion has to be reported rather than called converged
		CGConvergenceParams cp;
		cp.criterion = "relative";
		cp.rel_tol = 1e-8;
		cp.abs_tol = 1e-300;
		const vec_t b = vec_t::Constant(n, 1e-155);
		vec_t u = vec_t::Zero(n);
		bool nan = false;
		int steps = 0;
		solve_vec(b, cp, u, false, &nan, &steps);
		Check(nan, "a residual below the cutoff with an unmet tolerance is reported, not called converged");
		//the same residual with a tolerance that it does satisfy is a legitimate convergence
		cp.abs_tol = 1e-8;
		u = vec_t::Zero(n);
		solve_vec(b, cp, u, false, &nan, &steps);
		Check(!nan, "and it is accepted when the tolerance is met");
	}

	// ------------- "absolute" + "average" must not skip a column by its own norm
	{
		//A = 2I, so every column is solved in one step. One column has a norm below 'cg_delta_conv'
		//	while the block average stays above it: the historic rule solves that column, it does not
		//	replace it by zero
		const int tb = 2;
		std::vector<Eigen::Triplet<double>> tr;
		for (int i = 0; i < n; ++i) {
			tr.emplace_back(i, i, 2.);
		}
		sp_mat_rm_t A2(n, n);
		A2.setFromTriplets(tr.begin(), tr.end());
		const vec_t diag_inv2 = vec_t::Constant(n, 0.5);
		den_mat_t rhs2(n, tb);
		rhs2.setZero();
		rhs2(0, 0) = 5e-4;//below cg_delta_conv
		rhs2(1, 1) = 1.;//keeps the block average above it
		CGConvergenceParams cp;
		cp.delta_conv = 1e-3;
		den_mat_t U2(n, tb);
		bool nan = false;
		CGRandomEffectsMat(A2, rhs2, U2, nan, n, tb, n, cp.delta_conv, "incomplete_cholesky",
			[&]() { sp_mat_rm_t A_copy = A2, L; ZeroFillInIncompleteCholeskyFactorization(A_copy, L); return L; }(),
			unused, cp);
		Check(!nan && std::fabs(U2(0, 0) - 2.5e-4) < 1e-12,
			"absolute + average solves a small column instead of zeroing it");
	}

	// ------------- a rhs column that is exactly zero under the default rule
	{
		//'CalcXTPsiInvX' passes ZtX here, so a covariate that is identically zero within a cluster
		//	gives a zero column. The zero solution satisfies an absolute tolerance, so this is a
		//	result, not a failure
		const int tb = 4;
		std::vector<Eigen::Triplet<double>> tr;
		for (int i = 0; i < n; ++i) {
			tr.emplace_back(i, i, 4.);
			if (i + 1 < n) {
				tr.emplace_back(i, i + 1, -1.);
				tr.emplace_back(i + 1, i, -1.);
			}
		}
		sp_mat_rm_t A2(n, n);
		A2.setFromTriplets(tr.begin(), tr.end());
		sp_mat_rm_t A_copy = A2, L_ic;
		ZeroFillInIncompleteCholeskyFactorization(A_copy, L_ic);
		den_mat_t rhs2(n, tb);
		for (int j = 0; j < tb; ++j) {
			for (int i = 0; i < n; ++i) {
				rhs2(i, j) = ((i + j) % 3) ? 1. : -1.;
			}
		}
		rhs2.col(2).setZero();
		CGConvergenceParams cp;
		cp.delta_conv = 1e-2;
		den_mat_t U2(n, tb);
		bool nan = false;
		CGRandomEffectsMat(A2, rhs2, U2, nan, n, tb, 100, cp.delta_conv, "incomplete_cholesky",
			L_ic, unused, cp);
		Check(!nan && U2.allFinite() && U2.col(2).isZero(0),
			"a zero rhs column gives a zero solution and is not reported as a failure");
	}

	// ------------------------------------------------------------------ warm starts
	{
		CGConvergenceParams cp;
		cp.criterion = "relative";
		cp.rel_tol = 1e-12;
		cp.abs_tol = 1e-30;
		vec_t u = exact_vec;
		bool nan = false;
		int steps = 0;
		solve_vec(rhs_vec, cp, u, false, &nan, &steps);
		Check(!nan && u.allFinite() && steps == 0,
			"an exact warm start returns immediately without a 0/0");
	}

	// ------------------------------------------- the behaviour of the rules is unchanged
	{
		CGConvergenceParams tight_abs;
		tight_abs.delta_conv = 1e-10;
		double ldet_ref = 0.;
		int steps = 0;
		bool nan = false;
		std::vector<int> depths;
		den_mat_t U = run_tridiag(A, diag_inv, rhs_mat, tight_abs, &ldet_ref, &steps, &depths, &nan);
		const den_mat_t exact_mat = A_dense.llt().solve(rhs_mat);
		Check(!nan && (U - exact_mat).norm() / exact_mat.norm() < 1e-8,
			"multi rhs, absolute: solves the systems");
		const char* rules[3] = { "average", "max", "per_rhs" };
		for (int r = 0; r < 3; ++r) {
			CGConvergenceParams cp;
			cp.criterion = "relative";
			cp.rel_tol = 1e-12;
			cp.abs_tol = 1e-30;
			cp.multi_rhs_convergence = rules[r];
			double ldet = 0.;
			U = run_tridiag(A, diag_inv, rhs_mat, cp, &ldet, &steps, &depths, &nan);
			char msg[200];
			std::snprintf(msg, sizeof(msg), "relative + '%s': log-determinant matches", rules[r]);
			Check((U - exact_mat).norm() / exact_mat.norm() < 1e-7 &&
				std::fabs(ldet - ldet_ref) / std::fabs(ldet_ref) < 1e-4, msg);
		}
		//a Lanczos probe that satisfies the tolerance at iteration 0 is a probe of a stochastic
		//	estimator and still gets a step, instead of being dropped from the average
		CGConvergenceParams cp;
		cp.criterion = "relative";
		cp.rel_tol = 1e-8;
		cp.abs_tol = 1e6;
		double ldet = 0.;
		run_tridiag(A, diag_inv, rhs_mat, cp, &ldet, &steps, &depths, &nan);
		int min_depth = depths[0];
		for (int i = 0; i < t; ++i) {
			min_depth = std::min(min_depth, depths[i]);
		}
		Check(min_depth >= 1 && std::isfinite(ldet),
			"a tolerance above every probe norm still keeps all probes");
	}

	// ---------------------------------------------------- 'per_rhs' stops columns early
	{
		den_mat_t rhs_mixed = rhs_mat;
		for (int j = 0; j < t; ++j) {
			if (j % 3 != 0) {
				rhs_mixed.col(j) = A_dense * den_mat_t::Identity(n, n).col(j * (n / t));
			}
		}
		CGConvergenceParams cp;
		cp.criterion = "relative";
		cp.rel_tol = 1e-8;
		cp.abs_tol = 1e-30;
		cp.multi_rhs_convergence = "per_rhs";
		double ldet = 0.;
		int steps = 0;
		bool nan = false;
		std::vector<int> depths;
		const den_mat_t U = run_tridiag(A, diag_inv, rhs_mixed, cp, &ldet, &steps, &depths, &nan);
		const den_mat_t exact_mixed = A_dense.llt().solve(rhs_mixed);
		const double err = (U - exact_mixed).norm() / exact_mixed.norm();
		int mn = depths[0], mx = depths[0];
		for (int i = 0; i < t; ++i) {
			mn = std::min(mn, depths[i]);
			mx = std::max(mx, depths[i]);
		}
		std::printf("   per_rhs Lanczos depth in [%d, %d], relative error %.2e\n", mn, mx, err);
		Check(mn < mx && err < 1e-6, "'per_rhs' stops columns early and stays accurate");
	}

	std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures);
	return failures == 0 ? 0 : 1;
}
