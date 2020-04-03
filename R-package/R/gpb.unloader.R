#' gpboost unloading error fix
#'
#' Attempts to unload gpboost packages so you can remove objects cleanly without having to restart R. This is useful for instance if an object becomes stuck for no apparent reason and you do not want to restart R to fix the lost object.
#'
#' @param restore Whether to reload \code{gpboost} immediately after detaching from R. Defaults to \code{TRUE} which means automatically reload \code{gpboost} once unloading is performed.
#' @param wipe Whether to wipe all \code{gpb.Dataset} and \code{gpb.Booster} from the global environment. Defaults to \code{FALSE} which means to not remove them.
#' @param envir The environment to perform wiping on if \code{wipe == TRUE}. Defaults to \code{.GlobalEnv} which is the global environment.
#'
#' @return NULL invisibly.
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- gpb.train(params = params,
#'                    data = dtrain,
#'                    nrounds = 10,
#'                    valids = valids,
#'                    min_data = 1,
#'                    learning_rate = 1,
#'                    early_stopping_rounds = 5)
#'
#' \dontrun{
#' gpb.unloader(restore = FALSE, wipe = FALSE, envir = .GlobalEnv)
#' rm(model, dtrain, dtest) # Not needed if wipe = TRUE
#' gc() # Not needed if wipe = TRUE
#'
#' library(gpboost)
#' # Do whatever you want again with gpboost without object clashing
#' }
#'
#' @export
gpb.unloader <- function(restore = TRUE, wipe = FALSE, envir = .GlobalEnv) {

  # Unload package
  try(detach("package:gpboost", unload = TRUE), silent = TRUE)

  # Should we wipe variables? (gpb.Booster, gpb.Dataset)
  if (wipe) {
    boosters <- Filter(function(x) inherits(get(x, envir = envir), "gpb.Booster"), ls(envir = envir))
    datasets <- Filter(function(x) inherits(get(x, envir = envir), "gpb.Dataset"), ls(envir = envir))
    rm(list = c(boosters, datasets), envir = envir)
    gc(verbose = FALSE)
  }

  # Load package back?
  if (restore) {
    library(gpboost)
  }

  invisible()

}
