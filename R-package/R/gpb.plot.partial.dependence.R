#' @name gpb.plot.partial.dependence
#' @title Plot partial dependence plots
#' @description Plot partial dependence plots
#' @author Fabio Sigrist (adapted from a version by Michael Mayer)
#' @param model A \code{gpb.Booster} model object
#' @param data A \code{matrix} with data for creating partial dependence plots
#' @param variable A \code{string} with a name of the column or an \code{integer} 
#' with an index of the column in \code{data} for which a dependence plot is created
#' @param n.pt Evaluation grid size (used only if x is not discrete)
#' @param discrete.x A \code{boolean}. If TRUE, the evaluation grid is set to the unique values of x
#' @param which.class An \code{integer} indicating the class in multi-class classification (value from 0 to num_class - 1)
#' @param xlab Parameter passed to \code{plot}
#' @param ylab Parameter passed to \code{plot}
#' @param main Parameter passed to \code{plot}
#' @param type Parameter passed to \code{plot}
#' @param ... Additional parameters passed to \code{plot}
#' @param return_plot_data A \code{boolean}. If TRUE, the data for creating the partial dependence  plot is returned
#'
#' @return A two-dimensional \code{matrix} with data for creating the partial dependence plot.
#' This is only returned if \code{return_plot_data==TRUE}
#' 
#' @examples
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#'
#' gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
#' gpboost_model <- gpboost(data = X,
#'                          label = y,
#'                          gp_model = gp_model,
#'                          nrounds = 16,
#'                          learning_rate = 0.05,
#'                          max_depth = 6,
#'                          min_data_in_leaf = 5,
#'                          objective = "regression_l2",
#'                          verbose = 0)
#' gpb.plot.partial.dependence(gpboost_model, X, variable = 1)
#' @export
gpb.plot.partial.dependence <- function(model, data, variable, n.pt = 100,
                                        discrete.x = FALSE, which.class = NULL,
                                        xlab = deparse(substitute(variable)), 
                                        ylab = "", type = if (discrete.x) "p" else "b",
                                        main = "", return_plot_data = FALSE, ...) {
  stopifnot(dim(data) >= 1)
  
  if (discrete.x) {
    x <- unique(data[, variable])
  } else {
    x <- quantile(data[, variable], seq(0.01, 0.99, length.out = n.pt), names = FALSE)
  }
  y <- numeric(length(x))
  
  for (i in seq_along(x)) {
    data[, variable] <- x[i]
    
    if (!is.null(which.class)) {
      preds <- model$predict(data = data, reshape = TRUE, ignore_gp_model=TRUE)[, which.class + 1] 
    } else {
      preds <- model$predict(data = data, ignore_gp_model=TRUE)
    }
    
    y[i] <- mean(preds)
  }
  
  plot(x, y, xlab = xlab, ylab = ylab, main = main, type = type, ...)
  
  if (return_plot_data) {
    return(cbind(x = x, y = y))
  }
  
}
