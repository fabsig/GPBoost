#' Compute feature importance in a model
#'
#' Creates a \code{data.table} of feature importances in a model.
#'
#' @param model object of class \code{gpb.Booster}.
#' @param percentage whether to show importance in relative percentage.
#'
#' @return
#'
#' For a tree model, a \code{data.table} with the following columns:
#' \itemize{
#'   \item \code{Feature} Feature names in the model.
#'   \item \code{Gain} The total gain of this feature's splits.
#'   \item \code{Cover} The number of observation related to this feature.
#'   \item \code{Frequency} The number of times a feature splited in trees.
#' }
#'
#' @examples
#' \dontrun{
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#'
#' params <- list(objective = "binary",
#'                learning_rate = 0.01, num_leaves = 63, max_depth = -1,
#'                min_data_in_leaf = 1, min_sum_hessian_in_leaf = 1)
#' model <- gpb.train(params = params, data = dtrain, nrounds = 10)
#'
#' tree_imp1 <- feature.importance(model, percentage = TRUE)
#' tree_imp2 <- feature.importance(model, percentage = FALSE)
#' }
#' 
#' @importFrom data.table := setnames setorderv
#' @export
feature.importance <- function(model, percentage = TRUE) {

  # Check if model is a gpboost model
  if (!inherits(model, "gpb.Booster")) {
    stop("'model' has to be an object of class gpb.Booster")
  }

  # Setup importance
  tree_dt <- boost.model.dt.tree(model)

  # Extract elements
  tree_imp_dt <- tree_dt[
    !is.na(split_index)
    , .(Gain = sum(split_gain), Cover = sum(internal_count), Frequency = .N)
    , by = "split_feature"
  ]

  data.table::setnames(
    tree_imp_dt
    , old = "split_feature"
    , new = "Feature"
  )

  # Sort features by Gain
  data.table::setorderv(
    x = tree_imp_dt
    , cols = c("Gain")
    , order = -1
  )

  # Check if relative values are requested
  if (percentage) {
    tree_imp_dt[, ":="(Gain = Gain / sum(Gain),
                    Cover = Cover / sum(Cover),
                    Frequency = Frequency / sum(Frequency))]
  }

  # Return importance table
  return(tree_imp_dt)

}

#' Plot feature importance as a bar graph
#'
#' Plot previously calculated feature importance: Gain, Cover and Frequency, as a bar graph.
#'
#' @param tree_imp a \code{data.table} returned by \code{\link{feature.importance}}.
#' @param top_n maximal number of top features to include into the plot.
#' @param measure the name of importance measure to plot, can be "Gain", "Cover" or "Frequency".
#' @param left_margin (base R barplot) allows to adjust the left margin size to fit feature names.
#' @param cex (base R barplot) passed as \code{cex.names} parameter to \code{barplot}.
#'
#' @details
#' The graph represents each feature as a horizontal bar of length proportional to the defined importance of a feature.
#' Features are shown ranked in a decreasing importance order.
#'
#' @return
#' The \code{plotImportance} function creates a \code{barplot}
#' and silently returns a processed data.table with \code{top_n} features sorted by defined importance.
#'
#' @examples
#' \dontrun{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#'
#' params <- list(
#'     objective = "binary"
#'     , learning_rate = 0.01
#'     , num_leaves = 63
#'     , max_depth = -1
#'     , min_data_in_leaf = 1
#'     , min_sum_hessian_in_leaf = 1
#' )
#'
#' model <- gpb.train(params = params, data = dtrain, nrounds = 10)
#'
#' tree_imp <- feature.importance(model, percentage = TRUE)
#' plotImportance(tree_imp, top_n = 10, measure = "Gain")
#' }
#' @importFrom graphics barplot par
#' @export
plotImportance <- function(tree_imp,
                            top_n = 10,
                            measure = "Gain",
                            left_margin = 10,
                            cex = NULL) {
  
  # Check for measurement (column names) correctness
  measure <- match.arg(measure, choices = c("Gain", "Cover", "Frequency"), several.ok = FALSE)
  
  # Get top N importance (defaults to 10)
  top_n <- min(top_n, nrow(tree_imp))
  
  # Parse importance
  tree_imp <- tree_imp[order(abs(get(measure)), decreasing = TRUE),][seq_len(top_n),]
  
  # Attempt to setup a correct cex
  if (is.null(cex)) {
    cex <- 2.5 / log2(1 + top_n)
  }
  
  # Refresh plot
  op <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(op))
  
  graphics::par(
    mar = c(
      op$mar[1]
      , left_margin
      , op$mar[3]
      , op$mar[4]
    )
  )
  
  # Do plot
  tree_imp[.N:1,
           graphics::barplot(
             height = get(measure),
             names.arg = Feature,
             horiz = TRUE,
             border = NA,
             main = "Feature Importance",
             xlab = measure,
             cex.names = cex,
             las = 1
           )]
  
  # Return invisibly
  invisible(tree_imp)
  
}
