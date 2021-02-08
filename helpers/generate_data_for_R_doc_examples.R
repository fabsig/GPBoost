# Simulate example data (GPBoost_data.RData)

ntrain <- 500 # number of samples
ntest <- 5
n <- ntrain + ntest
set.seed(20)
# Simulate fixed effects
f1d <- function(x) 1.5*(1/(1+exp(-(x-0.5)*20))+0.75*x)-1.3
sim_non_lin_f <- function(n){
  X <- matrix(runif(2*n),ncol=2)
  f <- f1d(X[,1])
  return(list(X=X,f=f))
}
sim_FE <- sim_non_lin_f(n=n)
f <- sim_FE$f
X <- sim_FE$X
# Simulate 3 grouped random effects
m <- 50 # number of groups
group1 <- rep(1,ntrain) # first grouping variable
for(i in 1:m) group1[((i-1)*ntrain/m+1):(i*ntrain/m)] <- i
n_obs_gr <- ntrain/m # number of sampels per group
group2 <- rep(1,ntrain) # grouping variable for second crossed random effect
for(i in 1:m) group2[(1:n_obs_gr)+n_obs_gr*(i-1)] <- 1:n_obs_gr
group_data <- cbind(group1,group2)
group_test <- cbind(c(1,1,2,m+1,m+1),c(1,1,2,n_obs_gr+1,n_obs_gr+1))
group_data <- rbind(group_data,group_test)
eps_group1 <- sqrt(0.5) * rnorm(length(unique(group_data[,1])))[group_data[,1]]
eps_group2 <- sqrt(0.25) * rnorm(length(unique(group_data[,2])))[group_data[,2]]
eps_group3 <- X[,2] * sqrt(0.5) * rnorm(length(unique(group_data[,1])))[group_data[,1]] # Random coefficient effect
eps_group <- eps_group1 + eps_group2 + eps_group3
# Simulate spatial random effects
coords <- cbind(runif(n),runif(n)) # locations (=features) for Gaussian process
D <- as.matrix(dist(coords))
Sigma <- 1*exp(-D/0.3)+diag(1E-20,n)
C <- t(chol(Sigma))
eps_spatial1 <- as.vector(C %*% rnorm(n))
eps_spatial_SVC <- X[,2] * as.vector(C %*% rnorm(n))
eps_spatial <- eps_spatial1 + eps_spatial_SVC
# Sum all random effects and mean-center them
eps <- eps_group + eps_spatial
eps <- eps - mean(eps)
# Simulate error term
xi <- sqrt(0.1) * rnorm(n) 
# Observed data
y <- f + eps + xi
# Split in training and test data
y_test <- y[1:ntest+ntrain]
group_data_test <- group_data[1:ntest+ntrain,]
coords_test <- coords[1:ntest+ntrain,]
X_test <- X[1:ntest+ntrain,]
y <- y[1:ntrain]
group_data <- group_data[1:ntrain,]
coords <- coords[1:ntrain,]
X <- X[1:ntrain,]

# save(y, X, group_data, coords, X_test, group_data_test, coords_test, file = "GPBoost_data.RData")

# var(f)
# var(eps_group)
# var(eps_spatial)








