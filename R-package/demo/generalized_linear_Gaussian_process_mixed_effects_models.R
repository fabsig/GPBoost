## Examples of generalized linear Gaussian process and random effects models
##    for several non-Gaussian likelihoods
## Author: Fabio Sigrist

library(gpboost)

## Choose likelihood: either "bernoulli_probit" (=default for binary data), 
##"                     bernoulli_logit", "poisson", or "gamma"
likelihood <- "bernoulli_probit"

#--------------------Grouped random effects model----------------
# Simulate data
n <- 5000 # number of samples
m <- 500 # number of groups
set.seed(1)
group <- rep(1,n) # grouping variable
for(i in 1:m) group[((i-1)*n/m+1):(i*n/m)] <- i
b1 <- rnorm(m)
eps <- b1[group]
eps <- eps - mean(eps)
X <- cbind(rep(1,n),runif(n)-0.5) # desing matrix / covariate data for fixed effects
beta <- c(0,3) # regression coefficents
f <- X%*%beta # fixed effects

if (likelihood == "bernoulli_probit") {
  probs <- pnorm(f+eps)
  y <- as.numeric(runif(n) < probs)
} else if (likelihood == "bernoulli_logit") {
  probs <- 1/(1+exp(-(f+eps)))
  y <- as.numeric(runif(n) < probs)
} else if (likelihood == "poisson") {
  mu <- exp(f+eps)
  y <- qpois(runif(n), lambda = mu)
} else if (likelihood == "gamma") {
  mu <- exp(f+eps)
  y <- qgamma(runif(n), scale = mu, shape = 1)
}
hist(y,breaks=50)# visualize response variable

# Train model
gp_model <- fitGPModel(group_data = group, likelihood = likelihood, y = y, X = X)
summary(gp_model)

# Make predictions
group_test <- 1:m
X_test <- cbind(rep(1,m),rep(0,m))
# Predict latent variable
pred <- predict(gp_model, X_pred = X_test, group_data_pred = group_test,
                    predict_var = TRUE, predict_response = FALSE)
pred$mu[1:5] # Predicted latent mean
pred$var[1:5] # Predicted latent variance
# Predict response variable
pred_resp <- predict(gp_model, X_pred = X_test, group_data_pred = group_test,
                         predict_var = TRUE, predict_response = TRUE)
pred_resp$mu[1:5] # Predicted response variable (label)
pred_resp$var[1:5] # Predicted varianec of response


#--------------------Gaussian process model----------------
# Simulate data
ntrain <- 500
set.seed(1)
# training locations (exlcude upper right rectangle)
coords_train <- matrix(runif(2)/2,ncol=2)
while (dim(coords_train)[1]<ntrain) {
  coord_i <- runif(2) 
  if (!(coord_i[1]>=0.7 & coord_i[2]>=0.7)) {
    coords_train <- rbind(coords_train,coord_i)
  }
}
nx <- 30 # test data: number of grid points on each axis
x2 <- x1 <- rep((1:nx)/nx,nx)
for(i in 1:nx) x2[((i-1)*nx+1):(i*nx)]=i/nx
coords_test <- cbind(x1,x2)
coords <- rbind(coords_train, coords_test)
ntest <- nx * nx
n <- ntrain + ntest
# Simulate spatial Gaussian process
sigma2_1 <- 1 # marginal variance of GP
rho <- 0.1 # range parameter
D <- as.matrix(dist(coords))
Sigma <- sigma2_1 * exp(-D/rho) + diag(1E-20,n)
C <- t(chol(Sigma))
b_1 <- rnorm(n=n)
eps <- as.vector(C %*% b_1)
eps <- eps - mean(eps)
# Observed data
if (likelihood == "bernoulli_probit") {
  probs <- pnorm(eps)
  y <- as.numeric(runif(n) < probs)
} else if (likelihood == "bernoulli_logit") {
  probs <- 1/(1+exp(-eps))
  y <- as.numeric(runif(n) < probs)
} else if (likelihood == "poisson") {
  mu <- exp(eps)
  y <- qpois(runif(n), lambda = mu)
} else if (likelihood == "gamma") {
  mu <- exp(eps)
  y <- qgamma(runif(n), scale = mu, shape = 1)
}
# Split into training and test data
y_train <- y[1:ntrain]
y_test <- y[1:ntest+ntrain]
eps_test <- eps[1:ntest+ntrain]
hist(y_train,breaks=50)# visualize response variable

# Train model
gp_model <- fitGPModel(gp_coords = coords_train, cov_function = "exponential",
                       likelihood = likelihood, y = y_train)
summary(gp_model)

# Prediction of latent variable
pred <- predict(gp_model, gp_coords_pred = coords_test,
                predict_var = TRUE, predict_response = FALSE)
# Predict response variable (label)
pred_resp <- predict(gp_model, gp_coords_pred = coords_test,
                     predict_var = TRUE, predict_response = TRUE)
if (likelihood %in% c("bernoulli_probit","bernoulli_logit")) {
  print("Test error:")
  mean(as.numeric(pred_resp$mu>0.5) != y_test)
} else {
  print("Test root mean square error:")
  sqrt(mean((pred_resp$mu - y_test)^2))
}

# Visualize predictions and compare to true values
library(ggplot2)
library(viridis)
library(gridExtra)
plot1 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=eps_test),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + ggtitle("True latent GP and training locations") + 
  geom_point(data = data.frame(s_1=coords_train[,1],s_2=coords_train[,2],y=y_train),aes(x=s_1,y=s_2),size=3, col="white", alpha=1, shape=43)
plot2 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=pred$mu),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + ggtitle("Predicted latent GP mean")
plot3 <- ggplot(data = data.frame(s_1=coords_test[,1],s_2=coords_test[,2],b=sqrt(pred$var)),aes(x=s_1,y=s_2,color=b)) +
  geom_point(size=4, shape=15) + scale_color_viridis(option = "B") + labs(title="Predicted latent GP standard deviation", subtitle=" = prediction uncertainty")
grid.arrange(plot1, plot2, plot3, ncol=2)

