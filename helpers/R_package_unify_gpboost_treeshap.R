#' Unify GPBoost model for treeshap
#'
#' Convert your GPBoost model into a standardized representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param gpb_model A gpboost model - object of class \code{gpb.Booster}
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model.
#' @param recalculate logical indicating if covers should be recalculated according to the dataset given in data. Keep it \code{FALSE} if training data are used.
#'
#' @return a unified model representation - a \code{\link{model_unified.object}} object
#'
#' @export
#'
#' @import data.table
#'
#' @examples
#' \donttest{
#' library(gpboost)
#' data(GPBoost_data, package = "gpboost")
#' gpb_bst <- gpboost(data = X, label = y, gp_model = gp_model, nrounds = 16,
#'                    learning_rate = 0.05, max_depth = 6, min_data_in_leaf = 5,
#'                    verbose = 0)
#' unified_model <- gpboost_unify_treeshap(gpb_bst, X)
#' shaps <- treeshap(unified_model, X)
#' plot_contribution(shaps, obs = 1)
#' # Interaction values
#' interactions_bst <- treeshap::treeshap(unified_model, X, interactions = T, verbose = 0)
#' shap_bst <- shapviz(interactions_bst)
#' top4 <- names(head(sv_importance(shap_bst, kind = "no"), 4))
#' sv_interaction(shap_bst[1:1000, top4])
#' sv_dependence(shap_bst, v = "Covariate_1", color_var = top4, interactions = TRUE)
#' }
gpboost_unify_treeshap <- function(gpb_model, data, recalculate = FALSE) {
  if (!requireNamespace("gpboost", quietly = TRUE)) {
    stop("Package \"gpboost\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  df <- gpboost::gpb.model.dt.tree(gpb_model)
  stopifnot(c("split_index", "split_feature", "node_parent", "leaf_index", "leaf_parent", "internal_value",
              "internal_count", "leaf_value", "leaf_count", "decision_type") %in% colnames(df))
  df <- data.table::as.data.table(df)
  #convert node_parent and leaf_parent into one parent column
  df[is.na(df$node_parent), "node_parent"] <- df[is.na(df$node_parent), "leaf_parent"]
  #convert values into one column...
  df[is.na(df$internal_value), "internal_value"] <- df[!is.na(df$leaf_value), "leaf_value"]
  #...and counts
  df[is.na(df$internal_count), "internal_count"] <- df[!is.na(df$leaf_count), "leaf_count"]
  df[["internal_count"]] <- as.numeric(df[["internal_count"]])
  #convert split_index and leaf_index into one column:
  max_split_index <- df[, max(split_index, na.rm = TRUE), tree_index]
  rep_max_split <- rep(max_split_index$V1, times = as.numeric(table(df$tree_index)))
  new_leaf_index <- rep_max_split + df[, "leaf_index"] + 1
  df[is.na(df$split_index), "split_index"] <- new_leaf_index[!is.na(new_leaf_index[["leaf_index"]]), 'leaf_index']
  df[is.na(df$split_gain), "split_gain"] <- df[is.na(df$split_gain), "leaf_value"]
  # On the basis of column 'Parent', create columns with childs: 'Yes', 'No' and 'Missing' like in the xgboost df:
  ret.first <- function(x) x[1]
  ret.second <- function(x) x[2]
  tmp <- data.table::merge.data.table(df[, .(node_parent, tree_index, split_index)], df[, .(tree_index, split_index, default_left, decision_type)],
                                      by.x = c("tree_index", "node_parent"), by.y = c("tree_index", "split_index"))
  y_n_m <- unique(tmp[, .(Yes = ifelse(decision_type %in% c("<=", "<"), ret.first(split_index),
                                       ifelse(decision_type %in% c(">=", ">"), ret.second(split_index), stop("Unknown decision_type"))),
                          No = ifelse(decision_type %in% c(">=", ">"), ret.first(split_index),
                                      ifelse(decision_type %in% c("<=", "<"), ret.second(split_index), stop("Unknown decision_type"))),
                          Missing = ifelse(default_left, ret.first(split_index),ret.second(split_index)),
                          decision_type = decision_type),
                      .(tree_index, node_parent)])
  df <- data.table::merge.data.table(df[, c("tree_index", "depth", "split_index", "split_feature", "node_parent", "split_gain",
                                            "threshold", "internal_value", "internal_count")],
                                     y_n_m, by.x = c("tree_index", "split_index"),
                                     by.y = c("tree_index", "node_parent"), all.x = TRUE)
  df[decision_type == ">=", decision_type := "<"]
  df[decision_type == ">", decision_type := "<="]
  df$Decision.type <- factor(x = df$decision_type, levels = c("<=", "<"))
  df[is.na(split_index), Decision.type := NA]
  df <- df[, c("tree_index", "split_index", "split_feature", "Decision.type", "threshold", "Yes", "No", "Missing", "split_gain", "internal_count")]
  colnames(df) <- c("Tree", "Node", "Feature", "Decision.type", "Split", "Yes", "No", "Missing", "Prediction", "Cover")
  attr(df, "sorted") <- NULL
  
  ID <- paste0(df$Node, "-", df$Tree)
  df$Yes <- match(paste0(df$Yes, "-", df$Tree), ID)
  df$No <- match(paste0(df$No, "-", df$Tree), ID)
  df$Missing <- match(paste0(df$Missing, "-", df$Tree), ID)
  
  # Here we lose "Quality" information
  df$Prediction[!is.na(df$Feature)] <- NA
  
  feature_names <- jsonlite::fromJSON(gpb_model$dump_model())$feature_names
  data <- data[,colnames(data) %in% feature_names]
  
  ret <- list(model = as.data.frame(df), data = as.data.frame(data), feature_names = feature_names)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- TRUE
  attr(ret, "model") <- "gpboost"
  
  if (recalculate) {
    ret <- set_reference_dataset(ret, as.data.frame(data))
  }
  
  return(ret)
}
