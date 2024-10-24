// Copyright (C) 2016-2023 Yixuan Qiu <yixuan.qiu@cos.name>
// Modified work Copyright (c) 2024 Fabio Sigrist. All rights reserved.
// Under MIT license

#ifndef LBFGSPP_LINE_SEARCH_BACKTRACKING_H
#define LBFGSPP_LINE_SEARCH_BACKTRACKING_H

#include <Eigen/Core>
#include <stdexcept>  // std::runtime_error

namespace LBFGSpp {

///
/// The backtracking line search algorithm for L-BFGS. Mainly for internal use.
///
template <typename Scalar>
class LineSearchBacktracking
{
private:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

public:
    ///
    /// Line search by backtracking.
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    Parameters for the L-BFGS algorithm.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
    ///                 Can be ignored for the L-BFGS solver.
    /// \param step     In: The initial step length.
    ///                 Out: The calculated step length.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param grad     In: The current gradient vector.
    ///                 Out: The gradient at the new point.
    /// \param dg       In: The inner product between drt and grad.
    ///                 Out: The inner product between drt and the new gradient.
    /// \param x        Out: The new point moved to.
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, const LBFGSParam<Scalar>& param,
                           const Vector& xp, const Vector& drt, const Scalar& /*step_max*/,
                           Scalar& step, Scalar& fx, Vector& grad, Scalar& dg, Vector& x)
    {
        // Decreasing and increasing factors
        const Scalar dec = 0.5;
        const Scalar inc = 2.1;

        // Check the value of step
        if (step <= Scalar(0))
            Log::REFatal("GPModel lbfgs: 'step' must be positive");

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if (dg_init > 0)
            Log::REFatal("GPModel lbfgs: the moving direction increases the objective function value");

        const Scalar test_decr = param.ftol * dg_init;
        Scalar width;

        int iter;
        for (iter = 0; iter < param.max_linesearch; iter++)
        {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;
            // Evaluate this candidate
            fx = f(x, grad, true, false);  // ChangedForGPBoost

            //Log::REInfo("LineSearch: iter = %d, fx = %g, step = %g, fx_init = %g", iter, fx, step, fx_init);  // for debugging

            if (fx > fx_init + step * test_decr || (fx != fx))
            {
                // ChangedForGPBoost
                if ((fx - fx_init) > 2. * std::max(abs(fx_init), Scalar(1)))
                {
                    width = dec / 16.;  // make step size much smaller for very large increases to avoid too many backtracking steps
                }
                else
                {
                    width = dec;
                }
            }
            else
            {
                dg = grad.dot(drt);

                // Armijo condition is met
                if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
                    break;

                if (dg < param.wolfe * dg_init)
                {
                    width = inc;
                }
                else
                {
                    // Regular Wolfe condition is met
                    if (param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE)
                        break;

                    if (dg > -param.wolfe * dg_init)
                    {
                        width = dec;
                    }
                    else
                    {
                        // Strong Wolfe condition is met
                        break;
                    }
                }
            }

            if (step < param.min_step)
                Log::REDebug("GPModel lbfgs: the line search step became smaller than the minimum value allowed");

            if (step > param.max_step)
                Log::REDebug("GPModel lbfgs: the line search step became larger than the maximum value allowed");

            step *= width;
        }

        // ChangedForGPBoost
        if (iter >= param.max_linesearch)
        {
            x.noalias() = xp;
            f.ResetProfiledOutVariablesToLag1();
            fx = fx_init;
            step = 0.;
            Log::REDebug("GPModel lbfgs: the line search routine reached the maximum number of iterations");
        }
        else if (iter > 0)
        {
            Log::REDebug("LineSearch for 'lbfgs' finished after %d iterations, step length = %g", iter, step);
        }
         
    }
};

}  // namespace LBFGSpp

#endif  // LBFGSPP_LINE_SEARCH_BACKTRACKING_H
