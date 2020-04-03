library(gpboost)

#--------------------Grouped random effects model: single-level random effect----------------
n <- 100 # number of samples
m <- 25 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
# Create random effects model
gp_model <- GPModel(group_data = group)

# Simulate data
sigma2_1 <- 1^2 # random effect variance
sigma2 <- 0.5^2 # error variance
# incidence matrix relating grouped random effects to samples
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
set.seed(1)
b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
eps <- Z1 %*% b1
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- eps + xi # observed data
# Fit model
fit(gp_model, y = y, std_dev = TRUE)
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = group, y = y, std_dev = TRUE)
summary(gp_model)

# Make predictions
group_test <- 1:m
pred <- predict(gp_model, group_data_pred = group_test)
# Compare true and predicted random effects
plot(b1, pred$mu, xlab="truth", ylab="predicted",
     main="Comparison of true and predicted random effects")
abline(a=0,b=1)


#--------------------Two crossed random effects and a random slope----------------
# NOTE: run the above example first to create the first random effect
set.seed(1)
x <- runif(n) # covariate data for random slope
n_obs_gr <- n/m # number of sampels per group
group2 <- rep(1,n) # grouping variable for second crossed random effect
for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
# Create random effects model
gp_model <- GPModel(group_data = cbind(group,group2),
                    group_rand_coef_data = x,
                    ind_effect_group_rand_coef = 1)# the random slope is for the first random effect

# Simulate data
sigma2_2 <- 0.5^2 # variance of second random effect
sigma2_3 <- 0.75^2 # variance of random slope for first random effect
Z2 <- model.matrix(rep(1,n)~factor(group2)-1) # incidence matrix for second random effect
Z3 <- diag(x) %*% Z1 # incidence matrix for random slope for first random effect
b2 <- sqrt(sigma2_2) * rnorm(n_obs_gr) # second random effect
b3 <- sqrt(sigma2_3) * rnorm(m) # random slope for first random effect
eps2 <- Z1%*%b1 + Z2%*%b2 + Z3%*%b3 # sum of all random effects
y <- eps2 + xi # observed data
# Fit model
fit(gp_model, y = y, std_dev = TRUE)
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = cbind(group,group2),
                        group_rand_coef_data = x,
                        ind_effect_group_rand_coef = 1,
                        y = y, std_dev = TRUE)
summary(gp_model)


#--------------------Mixed effects model: random effects and linear fixed effects----------------
# NOTE: run the above example first to create the random effects part
set.seed(1)
X <- cbind(rep(1,n),runif(n)) # desing matrix / covariate data for fixed effect
beta <- c(3,3) # regression coefficents
y <- eps2 + xi + X%*%beta # add fixed effect to observed data
# Create random effects model
gp_model <- GPModel(group_data = cbind(group,group2),
                    group_rand_coef_data = x,
                    ind_effect_group_rand_coef = 1)
# Fit model
fit(gp_model, y = y, X = X, std_dev = TRUE)
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = cbind(group,group2),
                        group_rand_coef_data = x,
                        ind_effect_group_rand_coef = 1,
                        y = y, X = X, std_dev = TRUE)
summary(gp_model)


#--------------------Gaussian process model----------------
library(gpboost)
n <- 200 # number of samples
set.seed(1)
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
# Create Gaussian process model
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential")
## Other covariance functions:
# gp_model <- GPModel(gp_coords = coords, cov_function = "gaussian")
# gp_model <- GPModel(gp_coords = coords,
#                     cov_function = "matern", cov_fct_shape=1.5)
# gp_model <- GPModel(gp_coords = coords,
#                     cov_function = "powered_exponential", cov_fct_shape=1.1)
# Simulate data
sigma2_1 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
sigma2 <- 0.5^2 # error variance
D <- as.matrix(dist(coords))
Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
# Sigma = sigma2_1*exp(-(D/rho)^2)+diag(1E-20,n)
C = t(chol(Sigma))
b_1=rnorm(n) # simulate random effect
eps <- C %*% b_1
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- eps + xi
# Fit model
fit(gp_model, y = y, std_dev = TRUE,
    params = list(optimizer_cov = "gradient_descent",
                  lr_cov = 0.1))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                        y = y, std_dev = TRUE,
                        params = list(optimizer_cov = "gradient_descent",
                                      lr_cov = 0.1))
summary(gp_model)

# Make predictions
set.seed(1)
ntest <- 5
# prediction locations (=features) for Gaussian process
coords_test <- cbind(runif(ntest),runif(ntest))/10
pred <- predict(gp_model, gp_coords_pred = coords_test,
                predict_cov_mat = TRUE)
print("Predicted (posterior/conditional) mean of GP")
pred$mu
print("Predicted (posterior/conditional) covariance matrix of GP")
pred$cov


#--------------------Gaussian process model with Vecchia approximation----------------
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    vecchia_approx = TRUE, num_neighbors = 30)
# Fit model
fit(gp_model, y = y,
    params = list(optimizer_cov = "gradient_descent",
                  lr_cov = 0.1, maxit=200))
summary(gp_model)

# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                        vecchia_approx = TRUE, num_neighbors = 30,
                        y = y, params = list(optimizer_cov = "gradient_descent",
                                             lr_cov = 0.1, maxit=200))
summary(gp_model)


#--------------------Gaussian process model with random coefficents----------------
n <- 500 # number of samples
set.seed(1)
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
X_SVC=cbind(runif(n),runif(n)) # covariate data for random coeffients
colnames(X_SVC) = c("var1","var2")
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    gp_rand_coef_data = X_SVC)

# Simulate data
sigma2_1 <- 1^2 # marginal variance of GP (for simplicity, all GPs have the same parameters)
rho <- 0.1 # range parameter
sigma2 <- 0.5^2 # error variance
D <- as.matrix(dist(coords))
Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
C = t(chol(Sigma))
b_1=rnorm(n) # simulate random effect
b_2=rnorm(n)
b_3=rnorm(n)
eps <- C %*% b_1 + X_SVC[,1] * C %*% b_2 + X_SVC[,2] * C %*% b_3
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- eps + xi
# Fit model (takes a few seconds)
fit(gp_model, y = y, std_dev = TRUE,
    params = list(optimizer_cov = "gradient_descent",
                  lr_cov = 0.05, use_nesterov_acc = TRUE,
                  acc_rate_cov = 0.5))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                        gp_rand_coef_data = X_SVC,
                        y = y, std_dev = TRUE,
                        params = list(optimizer_cov = "gradient_descent",
                                      lr_cov = 0.05,
                                      use_nesterov_acc = TRUE,
                                      acc_rate_cov = 0.5))
summary(gp_model)


#--------------------GP model with two independent observations of the GP----------------
n <- 200 # number of samples
set.seed(1)
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
coords <- rbind(coords,coords) # locations for second observation of GP (same locations)
# indices that indicate the GP sample to which an observations belong
cluster_ids <- c(rep(1,n),rep(2,n))
# Create Gaussian process model
gp_model <- GPModel(gp_coords = coords, cov_function = "exponential",
                    cluster_ids = cluster_ids)
# Simulate data
sigma2_1 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
sigma2 <- 0.5^2 # error variance
D <- as.matrix(dist(coords[1:n,]))
Sigma = sigma2_1*exp(-D/rho)+diag(1E-20,n)
C = t(chol(Sigma))
b_1=rnorm(2 * n) # simulate random effect
eps <- c(C %*% b_1[1:n], C %*% b_1[1:n + n])
xi <- sqrt(sigma2) * rnorm(2 * n) # simulate error term
y <- eps + xi
# Fit model
fit(gp_model, y = y, std_dev = TRUE,
    params = list(optimizer_cov = "gradient_descent",
                  lr_cov = 0.05))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(gp_coords = coords, cov_function = "exponential",
                        cluster_ids = cluster_ids,
                        y = y, std_dev = TRUE,
                        params = list(optimizer_cov = "gradient_descent",
                                      lr_cov = 0.05))
summary(gp_model)


#--------------------Combine Gaussian process with grouped random effects----------------
n <- 200 # number of samples
m <- 25 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
set.seed(1)
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
# Create Gaussian process model
gp_model <- GPModel(group_data = group,
                    gp_coords = coords, cov_function = "exponential")

# Simulate data
sigma2_1 <- 1^2 # random effect variance
sigma2_2 <- 1^2 # marginal variance of GP
rho <- 0.1 # range parameter
sigma2 <- 0.5^2 # error variance
# incidence matrix relating grouped random effects to samples
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
set.seed(156)
b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
D <- as.matrix(dist(coords))
Sigma = sigma2_2*exp(-D/rho)+diag(1E-20,n)
C = t(chol(Sigma))
b_2=rnorm(n) # simulate random effect
eps <- Z1 %*% b1 + C %*% b_2
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- eps + xi
# Fit model
fit(gp_model, y = y, std_dev = TRUE,
    params = list(optimizer_cov = "gradient_descent",
                  lr_cov = 0.05, use_nesterov_acc = TRUE))
summary(gp_model)
# Alternatively, define and fit model directly using fitGPModel
gp_model <- fitGPModel(group_data = group,
                        gp_coords = coords, cov_function = "exponential",
                        y = y, std_dev = TRUE,
                        params = list(optimizer_cov = "gradient_descent",
                                      lr_cov = 0.05, use_nesterov_acc = TRUE))
summary(gp_model)
