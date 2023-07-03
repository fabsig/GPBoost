#' @name gpb.plot.partial.dependence
#' @title Plot partial dependence plots
#' @description Plot partial dependence plots
#' @author Fabio Sigrist (adapted from a version by Michael Mayer)
#' @param model A \code{gpb.Booster} model object
#' @param data A \code{matrix} with data for creating partial dependence plots
#' @param variable A \code{string} with a name of the column or an \code{integer} 
#' with an index of the column in \code{data} for which a dependence plot is created
#' @param subsample Fraction of random samples in \code{data} to be used for calculating the partial dependence plot
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
#' \donttest{
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
#'                          verbose = 0)
#' gpb.plot.partial.dependence(gpboost_model, X, variable = 1)
#' }
#' @export
gpb.plot.partial.dependence <- function(model, data, variable, n.pt = 100,
                                        subsample = pmin(1, n.pt * 100 / nrow(data)), 
                                        discrete.x = FALSE, which.class = NULL,
                                        xlab = deparse(substitute(variable)), 
                                        ylab = "", type = if (discrete.x) "p" else "b",
                                        main = "", return_plot_data = FALSE, ...) {
  stopifnot(dim(data) >= 1)
  
  if (subsample < 1) {
    n <- nrow(data)
    data <- data[sample(n, trunc(subsample * n)), , drop = FALSE]
  }
  
  if (discrete.x) {
    x <- sort(unique(data[, variable]))
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

#' @name gpb.plot.part.dep.interact
#' @title Plot interaction partial dependence plots
#' @description Plot interaction partial dependence plots
#' @author Fabio Sigrist
#' @importFrom graphics filled.contour
#' @importFrom graphics contour
#' @param model A \code{gpb.Booster} model object
#' @param data A \code{matrix} with data for creating partial dependence plots
#' @param variables A \code{vector} of length two of type \code{string} with 
#' names of the columns or \code{integer} with indices of the columns in 
#' \code{data} for which an interaction dependence plot is created
#' @param subsample Fraction of random samples in \code{data} to be used for calculating the partial dependence plot
#' @param n.pt.per.var Number of grid points per variable (used only if a variable is not discrete)
#' For continuous variables, the two-dimensional grid for the interaction plot 
#' has dimension c(n.pt.per.var, n.pt.per.var)
#' @param discrete.variables A \code{vector} of length two of type \code{boolean}. 
#' If an entry is TRUE, the evaluation grid of the corresponding variable is set to the unique values of the variable
#' @param which.class An \code{integer} indicating the class in multi-class 
#' classification (value from 0 to num_class - 1)
#' @param type A \code{character} string indicating the type of the plot. 
#' Supported values: "filled.contour" and "contour"
#' @param nlevels Parameter passed to the \code{filled.contour} or \code{contour} function
#' @param xlab Parameter passed to the \code{filled.contour} or \code{contour} function
#' @param ylab Parameter passed to the \code{filled.contour} or \code{contour} function
#' @param zlab Parameter passed to the \code{filled.contour} or \code{contour} function
#' @param main Parameter passed to the \code{filled.contour} or \code{contour} function
#' @param return_plot_data A \code{boolean}. If TRUE, the data for creating the partial dependence  plot is returned
#' @param ... Additional parameters passed to the \code{filled.contour} or \code{contour} function
#'
#' @return A \code{list} with three entries for creating the partial dependence plot: 
#' the first two entries are \code{vector}s with x and y coordinates. 
#' The third is a two-dimensional \code{matrix} of dimension c(length(x), length(y)) 
#' with z-coordinates. This is only returned if \code{return_plot_data==TRUE}
#' 
#' @examples
#' \donttest{
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' gp_model <- GPModel(group_data = group_data[,1], likelihood = "gaussian")
#' gpboost_model <- gpboost(data = X,
#'                         label = y,
#'                         gp_model = gp_model,
#'                         nrounds = 16,
#'                         learning_rate = 0.05,
#'                         max_depth = 6,
#'                         min_data_in_leaf = 5,
#'                         verbose = 0)
#' gpb.plot.part.dep.interact(gpboost_model, X, variables = c(1,2))
#' }
#' @export
gpb.plot.part.dep.interact <- function(model, data, variables, n.pt.per.var = 20,
                                       subsample = pmin(1, n.pt.per.var^2 * 100 / nrow(data)), 
                                       discrete.variables = c(FALSE, FALSE), which.class = NULL,
                                       type = "filled.contour", nlevels = 20, 
                                       xlab = variables[1], ylab = variables[2],
                                       zlab = "", main = "", 
                                       return_plot_data = FALSE, ...) {
  stopifnot(dim(data) >= 1)
  if (length(variables) != 2) {
    stop("gpb.plot.part.dep.interaction: Number of variables is not 2")
  }
  if (!(type %in% c("filled.contour","contour"))) {
    stop(paste0("gpb.plot.part.dep.interaction: type '",type,"' is not supported"))
  }
  
  if (subsample < 1) {
    n <- nrow(data)
    data <- data[sample(n, trunc(subsample * n)), , drop = FALSE]
  }
  
  if (discrete.variables[1]) {
    x <- sort(unique(data[, variables[1]]))
  } else {
    x <- quantile(data[, variables[1]],
                  seq(0.01, 0.99, length.out = n.pt.per.var), names = FALSE)
    x <- unique(x)
  }
  if (discrete.variables[2]) {
    y <- sort(unique(data[, variables[2]]))
  } else {
    y <- quantile(data[, variables[2]],
                  seq(0.01, 0.99, length.out = n.pt.per.var), names = FALSE)
    y <- unique(y)
  }
  z <- matrix(nrow = length(x), ncol = length(y))
  
  for (i in seq_along(x)) {
    for (j in seq_along(y)) {
      
      data[, variables[1]] <- x[i]
      data[, variables[2]] <- y[j]
      
      if (!is.null(which.class)) {
        preds <- model$predict(data = data, reshape = TRUE, ignore_gp_model=TRUE)[, which.class + 1] 
      } else {
        preds <- model$predict(data = data, ignore_gp_model=TRUE)
      }
      
      z[i,j] <- mean(preds)
    }
  }
  
  if (type == "filled.contour") {
    filled.contour(x, y, z, xlab = xlab, ylab = ylab, main = main, nlevels = nlevels, ...)
  } else if (type == "contour") {
    contour(x, y, z, xlab = xlab, ylab = ylab, main = main, nlevels = nlevels, ...)
  }
  
  if (return_plot_data) {
    return(list(x = x, y = y, z = z))
  }
  
}

