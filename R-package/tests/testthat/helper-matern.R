# Negative log-likelihood of a Gaussian process with a Matern covariance function and a Gaussian likelihood,
# calculated with the Bessel function of R ('dist_scaled' are the distances divided by the range parameters)
nll_matern_R <- function(dist_scaled, error_var, gp_var, shape, y) {
  range_dist <- sqrt(2 * shape) * dist_scaled
  corr <- 2^(1 - shape) / gamma(shape) * range_dist^shape * besselK(range_dist, shape)
  corr[range_dist == 0] <- 1
  L <- t(chol(gp_var * corr + error_var * diag(length(y))))
  as.numeric(0.5 * sum(forwardsolve(L, y)^2) + sum(log(diag(L))) + length(y) / 2 * log(2 * pi))
}
