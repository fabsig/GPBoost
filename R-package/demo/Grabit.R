## Examples on how to use GPBoost for the Grabit model of Sigrist and Hirnschall (2019)
## Author: Fabio Sigrist

library(gpboost)

# Function for non-linear mean
sim_friedman3 <- function(n, n_irrelevant=5){
        X <- matrix(runif(4*n),ncol=4)
        X[,1] <- 100*X[,1]
        X[,2] <- X[,2]*pi*(560-40)+40*pi
        X[,4] <- X[,4]*10+1
        f <- sqrt(10)*atan((X[,2]*X[,3]-1/(X[,2]*X[,4]))/X[,1])
        X <- cbind(rep(1,n),X)
        if(n_irrelevant>0) X <- cbind(X,matrix(runif(n_irrelevant*n),ncol=n_irrelevant))
        return(list(X=X,f=f))
}

# simulate data
n <- 10000
set.seed(1)
sim_train <- sim_friedman3(n=n)
sim_test <- sim_friedman3(n=n)
X <- sim_train$X
lp <- sim_train$f
X_test <- sim_test$X
lp_test <- sim_test$f
y <- rnorm(n,mean=lp,sd=1)
y_test <- rnorm(n,mean=lp_test,sd=1)
# apply censoring
yu <- 5
yl <- 3.5
y[y>=yu] <- yu
y[y<=yl] <- yl
# censoring fractions
sum(y==yu) / n
sum(y==yl) / n

# train model and make predictions
dtrain <- gpb.Dataset(data = X, label = y)
bst <- gpb.train(data = dtrain, nrounds = 100, objective = "tobit",
                 verbose = 0, yl = yl, yu = yu)
y_pred <- predict(bst, data = X_test)
# mean square error (approx. 1.0 for n=10'000)
print(paste0("Test error of Grabit: ", mean((y_pred - y_test)^2)))
# compare to standard least squares gradient boosting (approx. 1.5 for n=10'000)
bst <- gpb.train(data = dtrain, nrounds = 100, objective = "regression_l2",
                 verbose = 0)
y_pred_ls <- predict(bst, data = X_test)
print(paste0("Test error of standard least squares gradient boosting: ", mean((y_pred_ls - y_test)^2)))

