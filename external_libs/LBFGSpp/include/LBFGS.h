// Copyright (C) 2016-2023 Yixuan Qiu <yixuan.qiu@cos.name>
// Modified work Copyright (c) 2024 Fabio Sigrist. All rights reserved.
// Under MIT license

#ifndef LBFGSPP_LBFGS_H
#define LBFGSPP_LBFGS_H

#include <Eigen/Core>
#include "LBFGSpp/Param.h"
#include "LBFGSpp/BFGSMat.h"
#include "LBFGSpp/LineSearchBacktracking.h"
#include "LBFGSpp/LineSearchBracketing.h"
#include "LBFGSpp/LineSearchNocedalWright.h"
#include "LBFGSpp/LineSearchMoreThuente.h"

namespace LBFGSpp {

///
/// L-BFGS solver for unconstrained numerical optimization
///
template <typename Scalar,
          template <class> class LineSearch = LineSearchBacktracking>
class LBFGSSolver
{
private:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using MapVec = Eigen::Map<Vector>;

    const LBFGSParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar> m_bfgs;             // Approximation to the Hessian matrix
    Vector m_fx;                        // History of the objective function values
    Vector m_xp;                        // Old x
    Vector m_grad;                      // New gradient
    Scalar m_gnorm;                     // Norm of the gradient
    Vector m_gradp;                     // Old gradient
    Vector m_drt;                       // Moving direction

    // Reset internal variables
    // n: dimension of the vector to be optimized
    // reuse_m_bfgs_from_previous_call: If true, m_bfgs is not initialized
    inline void reset(int n, bool reuse_m_bfgs_from_previous_call)
    {
        const int m = m_param.m;
        if (!reuse_m_bfgs_from_previous_call)
        {
            m_bfgs.reset(n, m);
        }
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
        if (m_param.past > 0)
            m_fx.resize(m_param.past);
    }

public:
    ///
    /// Constructor for the L-BFGS solver.
    ///
    /// \param param An object of \ref LBFGSParam to store parameters for the
    ///        algorithm
    ///
    LBFGSSolver(const LBFGSParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function using the L-BFGS algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f  A function object such that `f(x, grad)` returns the
    ///           objective function value at `x`, and overwrites `grad` with
    ///           the gradient.
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    /// \param reuse_m_bfgs_from_previous_call If true, a given m_bfgs matrix is used (provided in m_bfgs_out)
    /// \param m_bfgs_given 
    ///
    /// \return Number of iterations used.
    ///
    template <typename Foo>
    inline int minimize(Foo& f, Vector& x, Scalar& fx, bool reuse_m_bfgs_from_previous_call, BFGSMat<Scalar>& m_bfgs_given)
    {
        using std::abs;

        // Dimension of the vector
        const int n = (int)x.size();
        reset(n, reuse_m_bfgs_from_previous_call);

        // The length of lag for objective function value to test convergence
        const int fpast = m_param.past;

        // Evaluate function and compute gradient
        fx = f(x, m_grad, true, true);// ChangedForGPBoost

        std::string init_coef_str = "";
        if (f.HasCovariates())
        {
            init_coef_str = " and 'init_coef'";
        }
        std::string problem_str = "none";
        if (std::isnan(fx))
        {
            problem_str = "NaN";
        }
        else if (std::isinf(fx))
        {
            problem_str = "Inf";
        }
        if (problem_str != "none")
        {
            Log::REFatal((problem_str + " occurred in initial approximate negative marginal log-likelihood. "
                "Possible solutions: try other initial values ('init_cov_pars'" + init_coef_str + ") "
                "or other tuning parameters in case you apply the GPBoost algorithm (e.g., learning_rate)").c_str());
        }
        Log::REDebug("Initial approximate negative marginal log-likelihood: %g", fx);

        m_gnorm = m_grad.norm();
        if (fpast > 0)
            m_fx[0] = fx;

        // std::cout << "x0 = " << x.transpose() << std::endl;
        // std::cout << "f(x0) = " << fx << ", ||grad|| = " << m_gnorm << std::endl << std::endl;

        // Early exit if the initial x is already a minimizer
        if (m_gnorm <= m_param.epsilon || m_gnorm <= m_param.epsilon_rel * x.norm())
        {
            return 1;
        }

        // Initial step size
        Scalar step;
        bool really_reuse_m_bfgs_from_previous_call = reuse_m_bfgs_from_previous_call;
        if (reuse_m_bfgs_from_previous_call)
        {
            really_reuse_m_bfgs_from_previous_call = reuse_m_bfgs_from_previous_call && (m_bfgs_given.get_m_ncorr() > 0) && ((int) x.size() == m_bfgs_given.get_dim_param());
        }
        if (really_reuse_m_bfgs_from_previous_call)
        {
            CHECK(m_bfgs_given.get_m_ncorr() > 0);
            m_bfgs = m_bfgs_given;
            step = 1.;
            m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);
        }
        else
        {
            // Initial direction
            m_drt.noalias() = -m_grad;
            step = Scalar(m_param.initial_step_factor) / m_drt.norm();  // ChangedForGPBoost
        }
        

        // Tolerance for s'y >= eps * (y'y)
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
        // s and y vectors
        Vector vecs(n), vecy(n);

        // Number of iterations used
        int k = 1;
        for (;;)
        {

            // std::cout << "Iter " << k << " begins" << std::endl << std::endl;

            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;
            Scalar dg = m_grad.dot(m_drt);
            const Scalar step_max = m_param.max_step;

            // ChangedForGPBoost
            Vector neg_mdrt = -m_drt;
            double max_lr = f.GetMaximalLearningRate(x, neg_mdrt);
            if (max_lr < step)
            {
                step = max_lr;
            }
            // Line search to update x, fx and gradient
            LineSearch<Scalar>::LineSearch(f, m_param, m_xp, m_drt, step_max, step, fx, m_grad, dg, x);

            // New gradient norm
            m_gnorm = m_grad.norm();

            // std::cout << "Iter " << k << " finished line search" << std::endl;
            // std::cout << "   x = " << x.transpose() << std::endl;
            // std::cout << "   f(x) = " << fx << ", ||grad|| = " << m_gnorm << std::endl << std::endl;

            // Convergence test -- gradient
            if (m_gnorm <= m_param.epsilon || m_gnorm <= m_param.epsilon_rel * x.norm())
            {
                m_bfgs_given = m_bfgs;
                return k;
            }
            // Convergence test -- objective function value
            if (fpast > 0)
            {
                const Scalar fxd = m_fx[k % fpast];

                // ChangedForGPBoost
                if (k >= fpast && (fxd - fx) <= m_param.delta * std::max(abs(fxd), Scalar(1)))
                {
                    m_bfgs_given = m_bfgs;
                    return k;
                }                   

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if (m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                m_bfgs_given = m_bfgs;
                return k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            vecs.noalias() = x - m_xp;
            vecy.noalias() = m_grad - m_gradp;
            if (vecs.dot(vecy) > eps * vecy.squaredNorm())
                m_bfgs.add_correction(vecs, vecy);

            // Recursive formula to compute d = -H * g
            m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);

            // Reset step = 1.0 as initial guess for the next line search
            step = Scalar(1);

            // ChangedForGPBoost
            f.SetNumIter(k - 1);
            f.SetLag1ProfiledOutVariables();
            if (f.LearnCovarianceParameters() && f.ShouldRedetermineNearestNeighborsVecchia())
            {
                f.RedetermineNearestNeighborsVecchia();  // called only in certain iterations if gp_approx == "vecchia" and neighbors are selected based on correlations and not distances
                fx = f(x, m_grad, true, false);
                m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);
                if (fpast > 0)
                {
                    m_fx[k % fpast] = fx;
                }
            }
            if ((k < 10 || (k % 10 == 0 && k < 100) || (k % 100 == 0 && k < 1000) ||
                 (k % 1000 == 0 && k < 10000) || (k % 10000 == 0)))
            {

                f.Logging(x,k,fx);
            }

            k++;
        }

        m_bfgs_given = m_bfgs;
        return k;
    }

    ///
    /// Returning the gradient vector on the last iterate.
    /// Typically used to debug and test convergence.
    /// Should only be called after the `minimize()` function.
    ///
    /// \return A const reference to the gradient vector.
    ///
    const Vector& final_grad() const { return m_grad; }

    ///
    /// Returning the Euclidean norm of the final gradient.
    ///
    Scalar final_grad_norm() const { return m_gnorm; }
};

}  // namespace LBFGSpp

#endif  // LBFGSPP_LBFGS_H
