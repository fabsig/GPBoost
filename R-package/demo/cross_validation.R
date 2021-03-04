## Examples on how cross-validation can be done for finding the number of 
## boosting iterations

## Author: Fabio Sigrist

library(gpboost)

#--------------------Cross-validation for tree-boosting without GP or random effects----------------
# Non-linear function for simulation
f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
x <- seq(from=0,to=1,length.out=200)
plot(x,f1d(x),type="l",lwd=2,col="red",main="Mean function")
# Function that simulates data. Two covariates of which only one has an effect
sim_data <- function(n){
  X=matrix(runif(2*n),ncol=2)
  # mean function plus noise
  y=f1d(X[,1])+rnorm(n,sd=0.1)
  return(list(X=X,y=y))
}
# Simulate data
n <- 1000
set.seed(1)
data <- sim_data(2 * n)
dtrain <- gpb.Dataset(data$X[1:n,], label = data$y[1:n])
nrounds <- 100
params <- list(learning_rate = 0.1,
              max_depth = 6,
              min_data_in_leaf = 5,
              objective = "regression_l2")

print("Running cross validation with mean squared error")
bst <- gpb.cv(params = params,
              data = dtrain,
              nrounds = nrounds,
              nfold = 10,
              eval = "l2",
              early_stopping_rounds = 5)
print(paste0("Optimal number of iterations: ", bst$best_iter))

print("Running cross validation with mean absolute error")
bst <- gpb.cv(params = params,
              data = dtrain,
              nrounds = nrounds,
              nfold = 10,
              eval = "l1",
              early_stopping_rounds = 5)
print(paste0("Optimal number of iterations: ", bst$best_iter))


#--------------------Custom loss function----------------
# Cross validation can also be done with a cutomized loss function
# Define custom loss (quantile loss)
quantile_loss <- function(preds, dtrain) {
  alpha <- 0.95
  labels <- getinfo(dtrain, "label")
  y_diff <- as.numeric(labels-preds)
  dummy <- ifelse(y_diff<0,1,0)
  quantloss <- mean((alpha-dummy)*y_diff)
  return(list(name = "quant_loss", value = quantloss, higher_better = FALSE))
}

print("Running cross validation, with cutomsized loss function (quantile loss)")
bst <- gpb.cv(params = params,
              data = dtrain,
              nrounds = nrounds,
              nfold = 10,
              eval = quantile_loss,
              early_stopping_rounds = 5)
print(paste0("Optimal number of iterations: ", bst$best_iter))


#--------------------Combine tree-boosting and grouped random effects model----------------
# Simulate data
f1d <- function(x) 1.7*(1/(1+exp(-(x-0.5)*20))+0.75*x)
set.seed(1)
n <- 1000 # number of samples
X <- matrix(runif(2*n),ncol=2)
y <- f1d(X[,1]) # mean
# Add grouped random effects
m <- 25 # number of categories / levels for grouping variable
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
sigma2_1 <- 1^2 # random effect variance
sigma2 <- 0.1^2 # error variance
# incidence matrix relating grouped random effects to samples
Z1 <- model.matrix(rep(1,n) ~ factor(group) - 1)
b1 <- sqrt(sigma2_1) * rnorm(m) # simulate random effects
eps <- Z1 %*% b1
xi <- sqrt(sigma2) * rnorm(n) # simulate error term
y <- y + eps + xi # add random effects and error to data

# Create random effects model and dataset
gp_model <- GPModel(group_data = group)
dtrain <- gpb.Dataset(X, label = y)
params <- list(learning_rate = 0.05,
               max_depth = 6,
               min_data_in_leaf = 5,
               objective = "regression_l2",
               leaves_newton_update = FALSE)

print("Running cross validation for GPBoost model")
bst <- gpb.cv(params = params,
              data = dtrain,
              gp_model = gp_model,
              nrounds = 100,
              nfold = 10,
              eval = "l2",
              early_stopping_rounds = 5)
print(paste0("Optimal number of iterations: ", bst$best_iter))

# Include random effect predictions for validation (observe the lower test error)
gp_model <- GPModel(group_data = group)
print("Running cross validation for GPBoost model and use_gp_model_for_validation = TRUE")
bst <- gpb.cv(params = params,
              data = dtrain,
              gp_model = gp_model,
              use_gp_model_for_validation = TRUE,
              nrounds = 100,
              nfold = 10,
              eval = "l2",
              early_stopping_rounds = 5)
print(paste0("Optimal number of iterations: ", bst$best_iter))
