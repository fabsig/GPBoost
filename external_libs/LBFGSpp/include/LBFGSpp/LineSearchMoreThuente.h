// Copyright (C) 2020-2023 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LINE_SEARCH_MORE_THUENTE_H
#define LBFGSPP_LINE_SEARCH_MORE_THUENTE_H

#include <stdexcept>  // std::invalid_argument, std::runtime_error
#include <Eigen/Core>
#include "LBFGSpp/Param.h"

namespace LBFGSpp {

///
/// The line search algorithm by Moré and Thuente (1994), currently used for the L-BFGS-B algorithm.
///
/// The target of this line search algorithm is to find a step size \f$\alpha\f$ that satisfies the strong Wolfe condition
/// \f$f(x+\alpha d) \le f(x) + \alpha\mu g(x)^T d\f$ and \f$|g(x+\alpha d)^T d| \le \eta|g(x)^T d|\f$.
/// Our implementation is a simplified version of the algorithm in [1]. We assume that \f$0<\mu<\eta<1\f$, while in [1]
/// they do not assume \f$\eta>\mu\f$. As a result, the algorithm in [1] has two stages, but in our implementation we
/// only need the first stage to guarantee the convergence.
///
/// Reference:
/// [1] Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease.
///
template <typename Scalar>
class LineSearchMoreThuente
{
private:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
    // that interpolates fa, ga, and fb, assuming the minimizer exists
    // For case I: fb >= fa and ga * (b - a) < 0
    static Scalar quadratic_minimizer(const Scalar& a, const Scalar& b, const Scalar& fa, const Scalar& ga, const Scalar& fb)
    {
        const Scalar ba = b - a;
        const Scalar w = Scalar(0.5) * ba * ga / (fa - fb + ba * ga);
        return a + w * ba;
    }

    // Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
    // that interpolates fa, ga and gb, assuming the minimizer exists
    // The result actually does not depend on fa
    // For case II: ga * (b - a) < 0, ga * gb < 0
    // For case III: ga * (b - a) < 0, ga * ga >= 0, |gb| <= |ga|
    static Scalar quadratic_minimizer(const Scalar& a, const Scalar& b, const Scalar& ga, const Scalar& gb)
    {
        const Scalar w = ga / (ga - gb);
        return a + w * (b - a);
    }

    // Local minimizer of a cubic function q(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3
    // that interpolates fa, ga, fb and gb, assuming a != b
    // Also sets a flag indicating whether the minimizer exists
    static Scalar cubic_minimizer(const Scalar& a, const Scalar& b, const Scalar& fa, const Scalar& fb,
                                  const Scalar& ga, const Scalar& gb, bool& exists)
    {
        using std::abs;
        using std::sqrt;

        const Scalar apb = a + b;
        const Scalar ba = b - a;
        const Scalar ba2 = ba * ba;
        const Scalar fba = fb - fa;
        const Scalar gba = gb - ga;
        // z3 = c3 * (b-a)^3, z2 = c2 * (b-a)^3, z1 = c1 * (b-a)^3
        const Scalar z3 = (ga + gb) * ba - Scalar(2) * fba;
        const Scalar z2 = Scalar(0.5) * (gba * ba2 - Scalar(3) * apb * z3);
        const Scalar z1 = fba * ba2 - apb * z2 - (a * apb + b * b) * z3;
        // std::cout << "z1 = " << z1 << ", z2 = " << z2 << ", z3 = " << z3 << std::endl;

        // If c3 = z/(b-a)^3 == 0, reduce to quadratic problem
        const Scalar eps = std::numeric_limits<Scalar>::epsilon();
        if (abs(z3) < eps * abs(z2) || abs(z3) < eps * abs(z1))
        {
            // Minimizer exists if c2 > 0
            exists = (z2 * ba > Scalar(0));
            // Return the end point if the minimizer does not exist
            return exists ? (-Scalar(0.5) * z1 / z2) : b;
        }

        // Now we can assume z3 > 0
        // The minimizer is a solution to the equation c1 + 2*c2 * x + 3*c3 * x^2 = 0
        // roots = -(z2/z3) / 3 (+-) sqrt((z2/z3)^2 - 3 * (z1/z3)) / 3
        //
        // Let u = z2/(3z3) and v = z1/z2
        // The minimizer exists if v/u <= 1
        const Scalar u = z2 / (Scalar(3) * z3), v = z1 / z2;
        const Scalar vu = v / u;
        exists = (vu <= Scalar(1));
        if (!exists)
            return b;

        // We need to find a numerically stable way to compute the roots, as z3 may still be small
        //
        // If |u| >= |v|, let w = 1 + sqrt(1-v/u), and then
        // r1 = -u * w, r2 = -v / w, r1 does not need to be the smaller one
        //
        // If |u| < |v|, we must have uv <= 0, and then
        // r = -u (+-) sqrt(delta), where
        // sqrt(delta) = sqrt(|u|) * sqrt(|v|) * sqrt(1-u/v)
        Scalar r1 = Scalar(0), r2 = Scalar(0);
        if (abs(u) >= abs(v))
        {
            const Scalar w = Scalar(1) + sqrt(Scalar(1) - vu);
            r1 = -u * w;
            r2 = -v / w;
        }
        else
        {
            const Scalar sqrtd = sqrt(abs(u)) * sqrt(abs(v)) * sqrt(1 - u / v);
            r1 = -u - sqrtd;
            r2 = -u + sqrtd;
        }
        return (z3 * ba > Scalar(0)) ? ((std::max)(r1, r2)) : ((std::min)(r1, r2));
    }

    // Select the next step size according to the current step sizes,
    // function values, and derivatives
    static Scalar step_selection(
        const Scalar& al, const Scalar& au, const Scalar& at,
        const Scalar& fl, const Scalar& fu, const Scalar& ft,
        const Scalar& gl, const Scalar& gu, const Scalar& gt)
    {
        using std::abs;

        if (al == au)
            return al;

        // If ft = Inf or gt = Inf, we return the middle point of al and at
        if (!std::isfinite(ft) || !std::isfinite(gt))
            return (al + at) / Scalar(2);

        // ac: cubic interpolation of fl, ft, gl, gt
        // aq: quadratic interpolation of fl, gl, ft
        bool ac_exists;
        // std::cout << "al = " << al << ", at = " << at << ", fl = " << fl << ", ft = " << ft << ", gl = " << gl << ", gt = " << gt << std::endl;
        const Scalar ac = cubic_minimizer(al, at, fl, ft, gl, gt, ac_exists);
        const Scalar aq = quadratic_minimizer(al, at, fl, gl, ft);
        // std::cout << "ac = " << ac << ", aq = " << aq << std::endl;
        // Case 1: ft > fl
        if (ft > fl)
        {
            // This should not happen if ft > fl, but just to be safe
            if (!ac_exists)
                return aq;
            // Then use the scheme described in the paper
            return (abs(ac - al) < abs(aq - al)) ? ac : ((aq + ac) / Scalar(2));
        }

        // as: quadratic interpolation of gl and gt
        const Scalar as = quadratic_minimizer(al, at, gl, gt);
        // Case 2: ft <= fl, gt * gl < 0
        if (gt * gl < Scalar(0))
            return (abs(ac - at) >= abs(as - at)) ? ac : as;

        // Case 3: ft <= fl, gt * gl >= 0, |gt| < |gl|
        const Scalar deltal = Scalar(1.1), deltau = Scalar(0.66);
        if (abs(gt) < abs(gl))
        {
            // We choose either ac or as
            // The case for ac: 1. It exists, and
            //                  2. ac is farther than at from al, and
            //                  3. ac is closer to at than as
            // Cases for as: otherwise
            const Scalar res = (ac_exists &&
                                (ac - at) * (at - al) > Scalar(0) &&
                                abs(ac - at) < abs(as - at)) ?
                ac :
                as;
            // Postprocessing the chosen step
            return (at > al) ?
                std::min(at + deltau * (au - at), res) :
                std::max(at + deltau * (au - at), res);
        }

        // Simple extrapolation if au, fu, or gu is infinity
        if ((!std::isfinite(au)) || (!std::isfinite(fu)) || (!std::isfinite(gu)))
            return at + deltal * (at - al);

        // ae: cubic interpolation of ft, fu, gt, gu
        bool ae_exists;
        const Scalar ae = cubic_minimizer(at, au, ft, fu, gt, gu, ae_exists);
        // Case 4: ft <= fl, gt * gl >= 0, |gt| >= |gl|
        // The following is not used in the paper, but it seems to be a reasonable safeguard
        return (at > al) ?
            std::min(at + deltau * (au - at), ae) :
            std::max(at + deltau * (au - at), ae);
    }

public:
    ///
    /// Line search by Moré and Thuente (1994).
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    An `LBFGSParam` or `LBFGSBParam` object that stores the
    ///                 parameters of the solver.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
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
    template <typename Foo, typename SolverParam>
    static void LineSearch(Foo& f, const SolverParam& param,
                           const Vector& xp, const Vector& drt, const Scalar& step_max,
                           Scalar& step, Scalar& fx, Vector& grad, Scalar& dg, Vector& x)
    {
        using std::abs;
        // std::cout << "========================= Entering line search =========================\n\n";

        // Check the value of step
        if (step <= Scalar(0))
            throw std::invalid_argument("'step' must be positive");
        if (step > step_max)
            throw std::invalid_argument("'step' exceeds 'step_max'");

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = dg;

        // std::cout << "fx_init = " << fx_init << ", dg_init = " << dg_init << std::endl << std::endl;

        // Make sure d points to a descent direction
        if (dg_init >= Scalar(0))
            throw std::logic_error("the moving direction does not decrease the objective function value");

        // Tolerance for convergence test
        // Sufficient decrease
        const Scalar test_decr = param.ftol * dg_init;
        // Curvature
        const Scalar test_curv = -param.wolfe * dg_init;

        // The bracketing interval
        Scalar I_lo = Scalar(0), I_hi = std::numeric_limits<Scalar>::infinity();
        Scalar fI_lo = Scalar(0), fI_hi = std::numeric_limits<Scalar>::infinity();
        Scalar gI_lo = (Scalar(1) - param.ftol) * dg_init, gI_hi = std::numeric_limits<Scalar>::infinity();
        // We also need to save x and grad for step=I_lo, since we want to return the best
        // step size along the path when strong Wolfe condition is not met
        Vector x_lo = xp, grad_lo = grad;
        Scalar fx_lo = fx_init, dg_lo = dg_init;

        // Function value and gradient at the current step size
        x.noalias() = xp + step * drt;
        fx = f(x, grad);
        dg = grad.dot(drt);

        // std::cout << "max_step = " << step_max << ", step = " << step << ", fx = " << fx << ", dg = " << dg << std::endl;

        // Convergence test
        if (fx <= fx_init + step * test_decr && abs(dg) <= test_curv)
        {
            // std::cout << "** Criteria met\n\n";
            // std::cout << "========================= Leaving line search =========================\n\n";
            return;
        }

        // Extrapolation factor
        const Scalar delta = Scalar(1.1);
        int iter;
        for (iter = 0; iter < param.max_linesearch; iter++)
        {
            // ft = psi(step) = f(xp + step * drt) - f(xp) - step * test_decr
            // gt = psi'(step) = dg - mu * dg_init
            // mu = param.ftol
            const Scalar ft = fx - fx_init - step * test_decr;
            const Scalar gt = dg - param.ftol * dg_init;

            // Update step size and bracketing interval
            Scalar new_step;
            if (ft > fI_lo)
            {
                // Case 1: ft > fl
                new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
                // Sanity check: if the computed new_step is too small, typically due to
                // extremely large value of ft, switch to the middle point
                if (new_step <= param.min_step)
                    new_step = (I_lo + step) / Scalar(2);

                I_hi = step;
                fI_hi = ft;
                gI_hi = gt;

                // std::cout << "Case 1: new step = " << new_step << std::endl;
            }
            else if (gt * (I_lo - step) > Scalar(0))
            {
                // Case 2: ft <= fl, gt * (al - at) > 0
                //
                // Page 291 of Moré and Thuente (1994) suggests that
                // newat = min(at + delta * (at - al), amax), delta in [1.1, 4]
                new_step = std::min(step_max, step + delta * (step - I_lo));

                // We can also consider the following scheme:
                // First let step_selection() decide a value, and then project to the range above
                //
                // new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
                // const Scalar delta2 = Scalar(4)
                // const Scalar t1 = step + delta * (step - I_lo);
                // const Scalar t2 = step + delta2 * (step - I_lo);
                // const Scalar tl = std::min(t1, t2), tu = std::max(t1, t2);
                // new_step = std::min(tu, std::max(tl, new_step));
                // new_step = std::min(step_max, new_step);

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                // Move x and grad to x_lo and grad_lo, respectively
                x_lo.swap(x);
                grad_lo.swap(grad);
                fx_lo = fx;
                dg_lo = dg;

                // std::cout << "Case 2: new step = " << new_step << std::endl;
            }
            else
            {
                // Case 3: ft <= fl, gt * (al - at) <= 0
                new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);

                I_hi = I_lo;
                fI_hi = fI_lo;
                gI_hi = gI_lo;

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                // Move x and grad to x_lo and grad_lo, respectively
                x_lo.swap(x);
                grad_lo.swap(grad);
                fx_lo = fx;
                dg_lo = dg;

                // std::cout << "Case 3: new step = " << new_step << std::endl;
            }

            // Case 1 and 3 are interpolations, whereas Case 2 is extrapolation
            // This means that Case 2 may return new_step = step_max,
            // and we need to decide whether to accept this value
            // 1. If both step and new_step equal to step_max, it means
            //    step will have no further change, so we accept it
            // 2. Otherwise, we need to test the function value and gradient
            //    on step_max, and decide later

            // In case step, new_step, and step_max are equal, directly return the computed x and fx
            if (step == step_max && new_step >= step_max)
            {
                // std::cout << "** Maximum step size reached\n\n";
                // std::cout << "========================= Leaving line search =========================\n\n";

                // Move {x, grad}_lo back before returning
                x.swap(x_lo);
                grad.swap(grad_lo);
                return;
            }
            // Otherwise, recompute x and fx based on new_step
            step = new_step;

            if (step < param.min_step)
                throw std::runtime_error("the line search step became smaller than the minimum value allowed");

            if (step > param.max_step)
                throw std::runtime_error("the line search step became larger than the maximum value allowed");

            // Update parameter, function value, and gradient
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
            dg = grad.dot(drt);

            // std::cout << "step = " << step << ", fx = " << fx << ", dg = " << dg << std::endl;

            // Convergence test
            if (fx <= fx_init + step * test_decr && abs(dg) <= test_curv)
            {
                // std::cout << "** Criteria met\n\n";
                // std::cout << "========================= Leaving line search =========================\n\n";
                return;
            }

            // Now assume step = step_max, and we need to decide whether to
            // exit the line search (see the comments above regarding step_max)
            // If we reach here, it means this step size does not pass the convergence
            // test, so either the sufficient decrease condition or the curvature
            // condition is not met yet
            //
            // Typically the curvature condition is harder to meet, and it is
            // possible that no step size in [0, step_max] satisfies the condition
            //
            // But we need to make sure that its psi function value is smaller than
            // the best one so far. If not, go to the next iteration and find a better one
            if (step >= step_max)
            {
                const Scalar ft_bound = fx - fx_init - step * test_decr;
                if (ft_bound <= fI_lo)
                {
                    // std::cout << "** Maximum step size reached\n\n";
                    // std::cout << "========================= Leaving line search =========================\n\n";
                    return;
                }
            }
        }

        // If we have used up all line search iterations, then the strong Wolfe condition
        // is not met. We choose not to raise an exception (unless no step satisfying
        // sufficient decrease is found), but to return the best step size so far
        if (iter >= param.max_linesearch)
        {
            // throw std::runtime_error("the line search routine reached the maximum number of iterations");

            // First test whether the last step is better than I_lo
            // If yes, return the last step
            const Scalar ft = fx - fx_init - step * test_decr;
            if (ft <= fI_lo)
                return;

            // If not, then the best step size so far is I_lo, but it needs to be positive
            if (I_lo <= Scalar(0))
                throw std::runtime_error("the line search routine is unable to sufficiently decrease the function value");

            // Return everything with _lo
            step = I_lo;
            fx = fx_lo;
            dg = dg_lo;
            // Move {x, grad}_lo back
            x.swap(x_lo);
            grad.swap(grad_lo);
            return;
        }
    }
};

}  // namespace LBFGSpp

#endif  // LBFGSPP_LINE_SEARCH_MORE_THUENTE_H
