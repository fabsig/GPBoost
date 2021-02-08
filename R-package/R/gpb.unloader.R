#' @name gpb.unloader
#' @title Remove gpboost and its objects from an environment
#' @description Attempts to unload GPBoost packages so you can remove objects cleanly without
#'              having to restart R. This is useful for instance if an object becomes stuck for no
#'              apparent reason and you do not want to restart R to fix the lost object.
#' @param restore Whether to reload \code{GPBoost} immediately after detaching from R.
#'                Defaults to \code{TRUE} which means automatically reload \code{GPBoost} once
#'                unloading is performed.
#' @param wipe Whether to wipe all \code{gpb.Dataset} and \code{gpb.Booster} from the global
#'             environment. Defaults to \code{FALSE} which means to not remove them.
#' @param envir The environment to perform wiping on if \code{wipe == TRUE}. Defaults to
#'              \code{.GlobalEnv} which is the global environment.
#'
#' @return NULL invisibly.
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(objective = "regression", metric = "l2")
#' valids <- list(test = dtest)
#' model <- gpb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#'   , min_data = 1L
#'   , learning_rate = 1.0
#' )
#'
#' gpb.unloader(restore = FALSE, wipe = FALSE, envir = .GlobalEnv)
#' rm(model, dtrain, dtest) # Not needed if wipe = TRUE
#' gc() # Not needed if wipe = TRUE
#'
#' library(gpboost)
#' # Do whatever you want again with GPBoost without object clashing
#' }
#' @export
gpb.unloader <- function(restore = TRUE, wipe = FALSE, envir = .GlobalEnv) {

  # Unload package
  try(detach("package:gpboost", unload = TRUE), silent = TRUE)

  # Should we wipe variables? (gpb.Booster, gpb.Dataset)
  if (wipe) {
    boosters <- Filter(
      f = function(x) {
        inherits(get(x, envir = envir), "gpb.Booster")
      }
      , x = ls(envir = envir)
    )
    datasets <- Filter(
      f = function(x) {
        inherits(get(x, envir = envir), "gpb.Dataset")
      }
      , x = ls(envir = envir)
    )
    rm(list = c(boosters, datasets), envir = envir)
    gc(verbose = FALSE)
  }

  # Load package back?
  if (restore) {
    library(gpboost)
  }

  return(invisible(NULL))

}
