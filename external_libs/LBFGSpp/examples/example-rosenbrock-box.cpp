#include <Eigen/Core>
#include <iostream>
#include <LBFGSB.h>

using namespace LBFGSpp;

typedef double Scalar;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

// Example from the roptim R package
// f(x) = (x[0] - 1)^2 + 4 * (x[1] - x[0]^2)^2 + ... + 4 * (x[end] - x[end - 1]^2)^2
class Rosenbrock
{
private:
    int n;
public:
    Rosenbrock(int n_) : n(n_) {}
    Scalar operator()(const Vector& x, Vector& grad)
    {
        Scalar fx = (x[0] - 1.0) * (x[0] - 1.0);
        grad[0] = 2 * (x[0] - 1) + 16 * (x[0] * x[0] - x[1]) * x[0];
        for(int i = 1; i < n; i++)
        {
            fx += 4 * std::pow(x[i] - x[i - 1] * x[i - 1], 2);
            if(i == n - 1)
            {
                grad[i] = 8 * (x[i] - x[i - 1] * x[i - 1]);
            } else {
                grad[i] = 8 * (x[i] - x[i - 1] * x[i - 1]) + 16 * (x[i] * x[i] - x[i + 1]) * x[i];
            }
        }
        return fx;
    }
};

int main()
{
    const int n = 25;
    LBFGSBParam<Scalar> param;
    LBFGSBSolver<Scalar> solver(param);
    Rosenbrock fun(n);

    // Variable bounds
    Vector lb = Vector::Constant(n, 2.0);
    Vector ub = Vector::Constant(n, 4.0);
    // The third variable is unbounded
    lb[2] = -std::numeric_limits<Scalar>::infinity();
    ub[2] = std::numeric_limits<Scalar>::infinity();
    // Initial values
    Vector x = Vector::Constant(n, 3.0);
    // Make some initial values at the bounds
    x[0] = x[1] = 2.0;
    x[5] = x[7] = 4.0;

    Scalar fx;
    int niter = solver.minimize(fun, x, fx, lb, ub);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
    std::cout << "grad = " << solver.final_grad().transpose() << std::endl;
    std::cout << "projected grad norm = " << solver.final_grad_norm() << std::endl;

    return 0;
}
