#' @importFrom methods is
#' @importFrom R6 R6Class
Dataset <- R6::R6Class(

  classname = "gpb.Dataset",
  cloneable = FALSE,
  public = list(

    # Finalize will free up the handles
    finalize = function() {

      # Check the need for freeing handle
      if (!gpb.is.null.handle(private$handle)) {

        # Freeing up handle
        gpb.call("LGBM_DatasetFree_R", ret = NULL, private$handle)
        private$handle <- NULL

      }

    },

    # Initialize will create a starter dataset
    initialize = function(data,
                          params = list(),
                          reference = NULL,
                          colnames = NULL,
                          categorical_feature = NULL,
                          predictor = NULL,
                          free_raw_data = FALSE,
                          used_indices = NULL,
                          info = list(),
                          ...) {

      # Check for additional parameters
      additional_params <- list(...)

      # Create known attributes list
      INFO_KEYS <- c("label", "weight", "init_score", "group")

      # Check if attribute key is in the known attribute list
      for (key in names(additional_params)) {

        # Key existing
        if (key %in% INFO_KEYS) {

          # Store as info
          info[[key]] <- additional_params[[key]]

        } else {

          # Store as param
          params[[key]] <- additional_params[[key]]

        }

      }

      # Check for dataset reference
      if (!is.null(reference)) {
        if (!gpb.check.r6.class(reference, "gpb.Dataset")) {
          stop("gpb.Dataset: Can only use ", sQuote("gpb.Dataset"), " as reference")
        }
      }

      # Check for predictor reference
      if (!is.null(predictor)) {
        if (!gpb.check.r6.class(predictor, "Predictor")) {
          stop("gpb.Dataset: Only can use ", sQuote("Predictor"), " as predictor")
        }
      }

      # Check for matrix format
      if (is.matrix(data)) {
        # Check whether matrix is the correct type first ("double")
        if (storage.mode(data) != "double") {
          storage.mode(data) <- "double"
        }
      }

      # Setup private attributes
      private$raw_data <- data
      private$params <- params
      private$reference <- reference
      private$colnames <- colnames

      private$categorical_feature <- categorical_feature
      private$predictor <- predictor
      private$free_raw_data <- free_raw_data
      if (!is.null(used_indices)) {
        private$used_indices <- sort(used_indices, decreasing = FALSE)
      }
      private$info <- info

    },

    create_valid = function(data,
                            info = list(),
                            ...) {

      # Create new dataset
      ret <- Dataset$new(data,
                         private$params,
                         self,
                         private$colnames,
                         private$categorical_feature,
                         private$predictor,
                         private$free_raw_data,
                         NULL,
                         info,
                         ...)

      # Return ret
      return(invisible(ret))

    },

    # Dataset constructor
    construct = function() {

      # Check for handle null
      if (!gpb.is.null.handle(private$handle)) {
        return(invisible(self))
      }

      # Get feature names
      cnames <- NULL
      if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {
        cnames <- colnames(private$raw_data)
      }

      # set feature names if not exist
      if (is.null(private$colnames) && !is.null(cnames)) {
        private$colnames <- as.character(cnames)
      }

      # Get categorical feature index
      if (!is.null(private$categorical_feature)) {

        # Check for character name
        if (is.character(private$categorical_feature)) {

            cate_indices <- as.list(match(private$categorical_feature, private$colnames) - 1)

            # Provided indices, but some indices are not existing?
            if (sum(is.na(cate_indices)) > 0) {
              stop("gpb.self.get.handle: supplied an unknown feature in categorical_feature: ", sQuote(private$categorical_feature[is.na(cate_indices)]))
            }

          } else {

            # Check if more categorical features were output over the feature space
            if (max(private$categorical_feature) > length(private$colnames)) {
              stop("gpb.self.get.handle: supplied a too large value in categorical_feature: ", max(private$categorical_feature), " but only ", length(private$colnames), " features")
            }

            # Store indices as [0, n-1] indexed instead of [1, n] indexed
            cate_indices <- as.list(private$categorical_feature - 1)

          }

        # Store indices for categorical features
        private$params$categorical_feature <- cate_indices

      }

      # Check has header or not
      has_header <- FALSE
      if (!is.null(private$params$has_header) || !is.null(private$params$header)) {
        if (tolower(as.character(private$params$has_header)) == "true" || tolower(as.character(private$params$header)) == "true") {
          has_header <- TRUE
        }
      }

      # Generate parameter str
      params_str <- gpb.params2str(private$params)

      # Get handle of reference dataset
      ref_handle <- NULL
      if (!is.null(private$reference)) {
        ref_handle <- private$reference$.__enclos_env__$private$get_handle()
      }
      handle <- NA_real_

      # Not subsetting
      if (is.null(private$used_indices)) {

        # Are we using a data file?
        if (is.character(private$raw_data)) {

          handle <- gpb.call("LGBM_DatasetCreateFromFile_R",
                             ret = handle,
                             gpb.c_str(private$raw_data),
                             params_str,
                             ref_handle)

        } else if (is.matrix(private$raw_data)) {

          # Are we using a matrix?
          handle <- gpb.call("LGBM_DatasetCreateFromMat_R",
                             ret = handle,
                             private$raw_data,
                             nrow(private$raw_data),
                             ncol(private$raw_data),
                             params_str,
                             ref_handle)

        } else if (methods::is(private$raw_data, "dgCMatrix")) {
          if (length(private$raw_data@p) > 2147483647) {
            stop("Cannot support large CSC matrix")
          }
          # Are we using a dgCMatrix (sparsed matrix column compressed)
          handle <- gpb.call("LGBM_DatasetCreateFromCSC_R",
                             ret = handle,
                             private$raw_data@p,
                             private$raw_data@i,
                             private$raw_data@x,
                             length(private$raw_data@p),
                             length(private$raw_data@x),
                             nrow(private$raw_data),
                             params_str,
                             ref_handle)

        } else {

          # Unknown data type
          stop("gpb.Dataset.construct: does not support constructing from ", sQuote(class(private$raw_data)))

        }

      } else {

        # Reference is empty
        if (is.null(private$reference)) {
          stop("gpb.Dataset.construct: reference cannot be NULL for constructing data subset")
        }

        # Construct subset
        handle <- gpb.call("LGBM_DatasetGetSubset_R",
                           ret = handle,
                           ref_handle,
                           c(private$used_indices), # Adding c() fixes issue in R v3.5
                           length(private$used_indices),
                           params_str)

      }
      if (gpb.is.null.handle(handle)) {
        stop("gpb.Dataset.construct: cannot create Dataset handle")
      }
      # Setup class and private type
      class(handle) <- "gpb.Dataset.handle"
      private$handle <- handle

      # Set feature names
      if (!is.null(private$colnames)) {
        self$set_colnames(private$colnames)
      }

      # Load init score if requested
      if (!is.null(private$predictor) && is.null(private$used_indices)) {

        # Setup initial scores
        init_score <- private$predictor$predict(private$raw_data, rawscore = TRUE, reshape = TRUE)

        # Not needed to transpose, for is col_marjor
        init_score <- as.vector(init_score)
        private$info$init_score <- init_score

      }

      # Should we free raw data?
      if (isTRUE(private$free_raw_data)) {
        private$raw_data <- NULL
      }

      # Get private information
      if (length(private$info) > 0) {

        # Set infos
        for (i in seq_along(private$info)) {

          p <- private$info[i]
          self$setinfo(names(p), p[[1]])

        }

      }

      # Get label information existence
      if (is.null(self$getinfo("label"))) {
        stop("gpb.Dataset.construct: label should be set")
      }

      # Return self
      return(invisible(self))

    },

    # Dimension function
    dim = function() {

      # Check for handle
      if (!gpb.is.null.handle(private$handle)) {

        num_row <- 0L
        num_col <- 0L

        # Get numeric data and numeric features
        c(gpb.call("LGBM_DatasetGetNumData_R", ret = num_row, private$handle),
          gpb.call("LGBM_DatasetGetNumFeature_R", ret = num_col, private$handle))

      } else if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {

        # Check if dgCMatrix (sparse matrix column compressed)
        # NOTE: requires Matrix package
        dim(private$raw_data)

      } else {

        # Trying to work with unknown dimensions is not possible
        stop("dim: cannot get dimensions before dataset has been constructed, please call gpb.Dataset.construct explicitly")

      }

    },

    # Get column names
    get_colnames = function() {

      # Check for handle
      if (!gpb.is.null.handle(private$handle)) {

        # Get feature names and write them
        cnames <- gpb.call.return.str("LGBM_DatasetGetFeatureNames_R", private$handle)
        private$colnames <- as.character(base::strsplit(cnames, "\t")[[1]])
        private$colnames

      } else if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {

        # Check if dgCMatrix (sparse matrix column compressed)
        colnames(private$raw_data)

      } else {

        # Trying to work with unknown dimensions is not possible
        stop("dim: cannot get dimensions before dataset has been constructed, please call gpb.Dataset.construct explicitly")

      }

    },

    # Set column names
    set_colnames = function(colnames) {

      # Check column names non-existence
      if (is.null(colnames)) {
        return(invisible(self))
      }

      # Check empty column names
      colnames <- as.character(colnames)
      if (length(colnames) == 0) {
        return(invisible(self))
      }

      # Write column names
      private$colnames <- colnames
      if (!gpb.is.null.handle(private$handle)) {

        # Merge names with tab separation
        merged_name <- paste0(as.list(private$colnames), collapse = "\t")
        gpb.call("LGBM_DatasetSetFeatureNames_R",
                 ret = NULL,
                 private$handle,
                 gpb.c_str(merged_name))

      }

      # Return self
      return(invisible(self))

    },

    # Get information
    getinfo = function(name) {

      # Create known attributes list
      INFONAMES <- c("label", "weight", "init_score", "group")

      # Check if attribute key is in the known attribute list
      if (!is.character(name) || length(name) != 1 || !name %in% INFONAMES) {
        stop("getinfo: name must one of the following: ", paste0(sQuote(INFONAMES), collapse = ", "))
      }

      # Check for info name and handle
      if (is.null(private$info[[name]])) {

        if (gpb.is.null.handle(private$handle)){
          stop("Cannot perform getinfo before constructing Dataset.")
        }

        # Get field size of info
        info_len <- 0L
        info_len <- gpb.call("LGBM_DatasetGetFieldSize_R",
                             ret = info_len,
                             private$handle,
                             gpb.c_str(name))

        # Check if info is not empty
        if (info_len > 0) {

          # Get back fields
          ret <- NULL
          ret <- if (name == "group") {
            integer(info_len) # Integer
          } else {
            numeric(info_len) # Numeric
          }

          ret <- gpb.call("LGBM_DatasetGetField_R",
                          ret = ret,
                          private$handle,
                          gpb.c_str(name))

          private$info[[name]] <- ret

        }
      }

      private$info[[name]]

    },

    # Set information
    setinfo = function(name, info) {

      # Create known attributes list
      INFONAMES <- c("label", "weight", "init_score", "group")

      # Check if attribute key is in the known attribute list
      if (!is.character(name) || length(name) != 1 || !name %in% INFONAMES) {
        stop("setinfo: name must one of the following: ", paste0(sQuote(INFONAMES), collapse = ", "))
      }

      # Check for type of information
      info <- if (name == "group") {
        as.integer(info) # Integer
      } else {
        as.numeric(info) # Numeric
      }

      # Store information privately
      private$info[[name]] <- info

      if (!gpb.is.null.handle(private$handle) && !is.null(info)) {

        if (length(info) > 0) {

          gpb.call("LGBM_DatasetSetField_R",
                   ret = NULL,
                   private$handle,
                   gpb.c_str(name),
                   info,
                   length(info))

        }

      }

      # Return self
      return(invisible(self))

    },

    # Slice dataset
    slice = function(idxset, ...) {

      # Perform slicing
      Dataset$new(NULL,
                  private$params,
                  self,
                  private$colnames,
                  private$categorical_feature,
                  private$predictor,
                  private$free_raw_data,
                  sort(idxset, decreasing = FALSE),
                  NULL,
                  ...)

    },

    # Update parameters
    update_params = function(params) {

      # Parameter updating
      if (!gpb.is.null.handle(private$handle)) {
        gpb.call("LGBM_DatasetUpdateParam_R", ret = NULL, private$handle, gpb.params2str(params))
        return(invisible(self))
      }
      private$params <- modifyList(private$params, params)
      return(invisible(self))

    },

    # Set categorical feature parameter
    set_categorical_feature = function(categorical_feature) {

      # Check for identical input
      if (identical(private$categorical_feature, categorical_feature)) {
        return(invisible(self))
      }

      # Check for empty data
      if (is.null(private$raw_data)) {
        stop("set_categorical_feature: cannot set categorical feature after freeing raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct gpb.Dataset")
      }

      # Overwrite categorical features
      private$categorical_feature <- categorical_feature

      # Finalize and return self
      self$finalize()
      return(invisible(self))

    },

    # Set reference
    set_reference = function(reference) {

      # Set known references
      self$set_categorical_feature(reference$.__enclos_env__$private$categorical_feature)
      self$set_colnames(reference$get_colnames())
      private$set_predictor(reference$.__enclos_env__$private$predictor)

      # Check for identical references
      if (identical(private$reference, reference)) {
        return(invisible(self))
      }

      # Check for empty data
      if (is.null(private$raw_data)) {

        stop("set_reference: cannot set reference after freeing raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct gpb.Dataset")

      }

      # Check for non-existing reference
      if (!is.null(reference)) {

        # Reference is unknown
        if (!gpb.check.r6.class(reference, "gpb.Dataset")) {
          stop("set_reference: Can only use gpb.Dataset as a reference")
        }

      }

      # Store reference
      private$reference <- reference

      # Finalize and return self
      self$finalize()
      return(invisible(self))

    },

    # Save binary model
    save_binary = function(fname) {

      # Store binary data
      self$construct()
      gpb.call("LGBM_DatasetSaveBinary_R",
               ret = NULL,
               private$handle,
               gpb.c_str(fname))
      return(invisible(self))
    }

  ),
  private = list(
    handle = NULL,
    raw_data = NULL,
    params = list(),
    reference = NULL,
    colnames = NULL,
    categorical_feature = NULL,
    predictor = NULL,
    free_raw_data = TRUE,
    used_indices = NULL,
    info = NULL,

    # Get handle
    get_handle = function() {

      # Get handle and construct if needed
      if (gpb.is.null.handle(private$handle)) {
        self$construct()
      }
      private$handle

    },

    # Set predictor
    set_predictor = function(predictor) {

      # Return self is identical predictor
      if (identical(private$predictor, predictor)) {
        return(invisible(self))
      }

      # Check for empty data
      if (is.null(private$raw_data)) {
        stop("set_predictor: cannot set predictor after free raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct gpb.Dataset")
      }

      # Check for empty predictor
      if (!is.null(predictor)) {

        # Predictor is unknown
        if (!gpb.check.r6.class(predictor, "Predictor")) {
          stop("set_predictor: Can only use 'Predictor' as predictor")
        }

      }

      # Store predictor
      private$predictor <- predictor

      # Finalize and return self
      self$finalize()
      return(invisible(self))

    }

  )
)

#' Construct \code{gpb.Dataset} object
#'
#' Construct \code{gpb.Dataset} object from dense matrix, sparse matrix
#' or local file (that was created previously by saving an \code{gpb.Dataset}).
#'
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param params a list of parameters
#' @param reference reference dataset
#' @param colnames names of columns
#' @param categorical_feature categorical features
#' @param free_raw_data TRUE for need to free raw data after construct
#' @param info a list of information of the \code{gpb.Dataset} object
#' @param ... other information to pass to \code{info} or parameters pass to \code{params}
#'
#' @return constructed dataset
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.save(dtrain, "gpb.Dataset.data")
#' dtrain <- gpb.Dataset("gpb.Dataset.data")
#' gpb.Dataset.construct(dtrain)
#'
#' @export
gpb.Dataset <- function(data,
                        params = list(),
                        reference = NULL,
                        colnames = NULL,
                        categorical_feature = NULL,
                        free_raw_data = FALSE,
                        info = list(),
                        ...) {

  # Create new dataset
  invisible(Dataset$new(data,
              params,
              reference,
              colnames,
              categorical_feature,
              NULL,
              free_raw_data,
              NULL,
              info,
              ...))

}

#' Construct validation data
#'
#' Construct validation data according to training data
#'
#' @param dataset \code{gpb.Dataset} object, training data
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param info a list of information of the \code{gpb.Dataset} object
#' @param ... other information to pass to \code{info}.
#'
#' @return constructed dataset
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#'
#' @export
gpb.Dataset.create.valid <- function(dataset, data, info = list(), ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(dataset)) {
    stop("gpb.Dataset.create.valid: input data should be an gpb.Dataset object")
  }

  # Create validation dataset
  invisible(dataset$create_valid(data, info, ...))

}

#' Construct Dataset explicitly
#'
#' @param dataset Object of class \code{gpb.Dataset}
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.construct(dtrain)
#'
#' @export
gpb.Dataset.construct <- function(dataset) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(dataset)) {
    stop("gpb.Dataset.construct: input data should be an gpb.Dataset object")
  }

  # Construct the dataset
  invisible(dataset$construct())

}

#' Dimensions of an \code{gpb.Dataset}
#'
#' Returns a vector of numbers of rows and of columns in an \code{gpb.Dataset}.
#' @param x Object of class \code{gpb.Dataset}
#' @param ... other parameters
#'
#' @return a vector of numbers of rows and of columns
#'
#' @details
#' Note: since \code{nrow} and \code{ncol} internally use \code{dim}, they can also
#' be directly used with an \code{gpb.Dataset} object.
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#'
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' stopifnot(ncol(dtrain) == ncol(train$data))
#' stopifnot(all(dim(dtrain) == dim(train$data)))
#'
#' @rdname dim
#' @export
dim.gpb.Dataset <- function(x, ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x)) {
    stop("dim.gpb.Dataset: input data should be an gpb.Dataset object")
  }

  # Return dimensions
  x$dim()

}

#' Handling of column names of \code{gpb.Dataset}
#'
#' Only column names are supported for \code{gpb.Dataset}, thus setting of
#' row names would have no effect and returned row names would be NULL.
#'
#' @param x object of class \code{gpb.Dataset}
#' @param value a list of two elements: the first one is ignored
#'        and the second one is column names
#'
#' @details
#' Generic \code{dimnames} methods are used by \code{colnames}.
#' Since row names are irrelevant, it is recommended to use \code{colnames} directly.
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.construct(dtrain)
#' dimnames(dtrain)
#' colnames(dtrain)
#' colnames(dtrain) <- make.names(1:ncol(train$data))
#' print(dtrain, verbose = TRUE)
#'
#' @rdname dimnames.gpb.Dataset
#' @export
dimnames.gpb.Dataset <- function(x) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x)) {
    stop("dimnames.gpb.Dataset: input data should be an gpb.Dataset object")
  }

  # Return dimension names
  list(NULL, x$get_colnames())

}

#' @rdname dimnames.gpb.Dataset
#' @export
`dimnames<-.gpb.Dataset` <- function(x, value) {

  # Check if invalid element list
  if (!is.list(value) || length(value) != 2L) {
    stop("invalid ", sQuote("value"), " given: must be a list of two elements")
  }

  # Check for unknown row names
  if (!is.null(value[[1L]])) {
    stop("gpb.Dataset does not have rownames")
  }

  # Check for second value missing
  if (is.null(value[[2]])) {

    # No column names
    x$set_colnames(NULL)
    return(x)

  }

  # Check for unmatching column size
  if (ncol(x) != length(value[[2]])) {
    stop("can't assign ", sQuote(length(value[[2]])), " colnames to an gpb.Dataset with ", sQuote(ncol(x)), " columns")
  }

  # Set column names properly, and return
  x$set_colnames(value[[2]])
  x

}

#' Slice a dataset
#'
#' Get a new \code{gpb.Dataset} containing the specified rows of
#' original \code{gpb.Dataset} object
#'
#' @param dataset Object of class \code{gpb.Dataset}
#' @param idxset an integer vector of indices of rows needed
#' @param ... other parameters (currently not used)
#' @return constructed sub dataset
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#'
#' dsub <- gpboost::slice(dtrain, 1:42)
#' gpb.Dataset.construct(dsub)
#' labels <- gpboost::getinfo(dsub, "label")
#'
#' @export
slice <- function(dataset, ...) {
  UseMethod("slice")
}

#' @rdname slice
#' @export
slice.gpb.Dataset <- function(dataset, idxset, ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(dataset)) {
    stop("slice.gpb.Dataset: input dataset should be an gpb.Dataset object")
  }

  # Return sliced set
  invisible(dataset$slice(idxset, ...))

}

#' Get information of an \code{gpb.Dataset} object
#'
#' @param dataset Object of class \code{gpb.Dataset}
#' @param name the name of the information field to get (see details)
#' @param ... other parameters
#' @return info data
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item \code{label}: label / response variable;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{group}: group size ;
#'     \item \code{init_score}: initial score is the base prediction.
#' }
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.construct(dtrain)
#'
#' labels <- gpboost::getinfo(dtrain, "label")
#' gpboost::setinfo(dtrain, "label", 1 - labels)
#'
#' labels2 <- gpboost::getinfo(dtrain, "label")
#' stopifnot(all(labels2 == 1 - labels))
#'
#' @export
getinfo <- function(dataset, ...) {
  UseMethod("getinfo")
}

#' @rdname getinfo
#' @export
getinfo.gpb.Dataset <- function(dataset, name, ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(dataset)) {
    stop("getinfo.gpb.Dataset: input dataset should be an gpb.Dataset object")
  }

  # Return information
  dataset$getinfo(name)

}

#' Set information of an \code{gpb.Dataset} object
#'
#' @param dataset Object of class \code{gpb.Dataset}
#' @param name the name of the field to get
#' @param info the specific field of information to set
#' @param ... other parameters
#' @return passed object
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item \code{label}: label / response variable;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{init_score}: initial score is the base prediction :
#'     \item \code{group}.
#' }
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.construct(dtrain)
#'
#' labels <- gpboost::getinfo(dtrain, "label")
#' gpboost::setinfo(dtrain, "label", 1 - labels)
#'
#' labels2 <- gpboost::getinfo(dtrain, "label")
#' stopifnot(all.equal(labels2, 1 - labels))
#'
#' @export
setinfo <- function(dataset, ...) {
  UseMethod("setinfo")
}

#' @rdname setinfo
#' @export
setinfo.gpb.Dataset <- function(dataset, name, info, ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(dataset)) {
    stop("setinfo.gpb.Dataset: input dataset should be an gpb.Dataset object")
  }

  # Set information
  invisible(dataset$setinfo(name, info))
}

#' Set categorical feature of \code{gpb.Dataset}
#'
#' @param dataset object of class \code{gpb.Dataset}
#' @param categorical_feature categorical features
#'
#' @return passed dataset
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.save(dtrain, "gpb.Dataset.data")
#' dtrain <- gpb.Dataset("gpb.Dataset.data")
#' gpb.Dataset.set.categorical(dtrain, 1:2)
#'
#' @rdname gpb.Dataset.set.categorical
#' @export
gpb.Dataset.set.categorical <- function(dataset, categorical_feature) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(dataset)) {
    stop("gpb.Dataset.set.categorical: input dataset should be an gpb.Dataset object")
  }

  # Set categoricals
  invisible(dataset$set_categorical_feature(categorical_feature))

}

#' Set reference of \code{gpb.Dataset}
#'
#' If you want to use validation data, you should set reference to training data
#'
#' @param dataset object of class \code{gpb.Dataset}
#' @param reference object of class \code{gpb.Dataset}
#'
#' @return passed dataset
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package ="gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset(test$data, test = train$label)
#' gpb.Dataset.set.reference(dtest, dtrain)
#'
#' @rdname gpb.Dataset.set.reference
#' @export
gpb.Dataset.set.reference <- function(dataset, reference) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(dataset)) {
    stop("gpb.Dataset.set.reference: input dataset should be an gpb.Dataset object")
  }

  # Set reference
  invisible(dataset$set_reference(reference))
}

#' Save \code{gpb.Dataset} to a binary file
#'
#' @param dataset object of class \code{gpb.Dataset}
#' @param fname object filename of output file
#'
#' @return passed dataset
#'
#' @examples
#' library(gpboost)
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.save(dtrain, "data.bin")
#'
#' @rdname gpb.Dataset.save
#' @export
gpb.Dataset.save <- function(dataset, fname) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(dataset)) {
    stop("gpb.Dataset.set: input dataset should be an gpb.Dataset object")
  }

  # File-type is not matching
  if (!is.character(fname)) {
    stop("gpb.Dataset.set: fname should be a character or a file connection")
  }

  # Store binary
  invisible(dataset$save_binary(fname))
}
