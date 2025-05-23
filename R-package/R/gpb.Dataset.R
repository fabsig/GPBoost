#' @importFrom methods is
#' @importFrom R6 R6Class
Dataset <- R6::R6Class(

  classname = "gpb.Dataset",
  cloneable = FALSE,
  public = list(

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

      # validate inputs early to avoid unnecessary computation
      if (!(is.null(reference) || gpb.check.r6.class(object = reference, name = "gpb.Dataset"))) {
          stop("gpb.Dataset: If provided, reference must be a ", sQuote("gpb.Dataset"))
      }
      if (!(is.null(predictor) || gpb.check.r6.class(object = predictor, name = "gpb.Predictor"))) {
          stop("gpb.Dataset: If provided, predictor must be a ", sQuote("gpb.Predictor"))
      }

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

      # Check for matrix format
      if (is.matrix(data)) {
        # Check whether matrix is the correct type first ("double")
        if (storage.mode(data) != "double") {
          storage.mode(data) <- "double"
        }
        # Make sure that data set is stored correctly
        # This is to avoid problems when dimnames / colnames are changed after initilization 
        data <- matrix(as.vector(data), ncol=ncol(data), dimnames=dimnames(data))
      }

      # Setup private attributes
      private$raw_data <- data
      private$params <- params
      private$reference <- reference
      private$colnames <- colnames

      private$categorical_feature <- categorical_feature
      private$predictor <- predictor
      private$free_raw_data <- free_raw_data
      private$used_indices <- sort(used_indices, decreasing = FALSE)
      private$info <- info
      private$version <- 0L

      return(invisible(NULL))

    },

    create_valid = function(data,
                            info = list(),
                            ...) {

      # Create new dataset
      ret <- Dataset$new(
        data = data
        , params = private$params
        , reference = self
        , colnames = private$colnames
        , categorical_feature = private$categorical_feature
        , predictor = private$predictor
        , free_raw_data = private$free_raw_data
        , used_indices = NULL
        , info = info
        , ...
      )

      return(invisible(ret))

    },

    # Dataset constructor
    construct = function() {

      # Check for handle null
      if (!gpb.is.null.handle(x = private$handle)) {
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

            cate_indices <- as.list(match(private$categorical_feature, private$colnames) - 1L)

            # Provided indices, but some indices are not existing?
            if (sum(is.na(cate_indices)) > 0L) {
              stop(
                "gpb.self.get.handle: supplied an unknown feature in categorical_feature: "
                , sQuote(private$categorical_feature[is.na(cate_indices)])
              )
            }

          } else {

            # Check if more categorical features were output over the feature space
            if (max(private$categorical_feature) > length(private$colnames)) {
              stop(
                "gpb.self.get.handle: supplied a too large value in categorical_feature: "
                , max(private$categorical_feature)
                , " but only "
                , length(private$colnames)
                , " features"
              )
            }

            # Store indices as [0, n-1] indexed instead of [1, n] indexed
            cate_indices <- as.list(private$categorical_feature - 1L)

          }

        # Store indices for categorical features
        private$params$categorical_feature <- cate_indices

      }

      # Check has header or not
      has_header <- FALSE
      if (!is.null(private$params$has_header) || !is.null(private$params$header)) {
        params_has_header <- tolower(as.character(private$params$has_header)) == "true"
        params_header <- tolower(as.character(private$params$header)) == "true"
        if (params_has_header || params_header) {
          has_header <- TRUE
        }
      }

      # Generate parameter str
      params_str <- gpb.params2str(params = private$params)

      # Get handle of reference dataset
      ref_handle <- NULL
      if (!is.null(private$reference)) {
        ref_handle <- private$reference$.__enclos_env__$private$get_handle()
      }
      handle <- NULL

      # Not subsetting
      if (is.null(private$used_indices)) {

        # Are we using a data file?
        if (is.character(private$raw_data)) {

          handle <- .Call(
            LGBM_DatasetCreateFromFile_R
            , private$raw_data
            , params_str
            , ref_handle
          )
          private$free_raw_data <- TRUE

        } else if (is.matrix(private$raw_data)) {

          # Are we using a matrix?
          handle <- .Call(
            LGBM_DatasetCreateFromMat_R
            , private$raw_data
            , nrow(private$raw_data)
            , ncol(private$raw_data)
            , params_str
            , ref_handle
          )

        } else if (methods::is(private$raw_data, "dgCMatrix")) {
          if (length(private$raw_data@p) > 2147483647L) {
            stop("Cannot support large CSC matrix")
          }
          # Are we using a dgCMatrix (sparsed matrix column compressed)
          handle <- .Call(
            LGBM_DatasetCreateFromCSC_R
            , private$raw_data@p
            , private$raw_data@i
            , private$raw_data@x
            , length(private$raw_data@p)
            , length(private$raw_data@x)
            , nrow(private$raw_data)
            , params_str
            , ref_handle
          )

        } else {

          # Unknown data type
          stop(
            "gpb.Dataset.construct: does not support constructing from "
            , sQuote(class(private$raw_data))
          )

        }

      } else {

        # Reference is empty
        if (is.null(private$reference)) {
          stop("gpb.Dataset.construct: reference cannot be NULL for constructing data subset")
        }

        # Construct subset
        handle <- .Call(
          LGBM_DatasetGetSubset_R
          , ref_handle
          , c(private$used_indices) # Adding c() fixes issue in R v3.5
          , length(private$used_indices)
          , params_str
        )

      }
      if (gpb.is.null.handle(x = handle)) {
        stop("gpb.Dataset.construct: cannot create Dataset handle")
      }
      # Setup class and private type
      class(handle) <- "gpb.Dataset.handle"
      private$handle <- handle

      # Set feature names
      if (!is.null(private$colnames)) {
        self$set_colnames(colnames = private$colnames)
      }

      # Load init score if requested
      if (!is.null(private$predictor) && is.null(private$used_indices)) {

        # Setup initial scores
        init_score <- private$predictor$predict(
          data = private$raw_data
          , rawscore = TRUE
          , reshape = TRUE
        )

        # Not needed to transpose, for is col_marjor
        init_score <- as.vector(init_score)
        private$info$init_score <- init_score

      }

      # Should we free raw data?
      if (isTRUE(private$free_raw_data)) {
        private$raw_data <- NULL
      }

      # Get private information
      if (length(private$info) > 0L) {

        # Set infos
        for (i in seq_along(private$info)) {

          p <- private$info[i]
          self$setinfo(name = names(p), info = p[[1L]])

        }

      }

      # Get label information existence
      if (is.null(self$getinfo(name = "label"))) {
        stop("gpb.Dataset.construct: label should be set")
      }

      return(invisible(self))

    },

    # Dimension function
    dim = function() {

      # Check for handle
      if (!gpb.is.null.handle(x = private$handle)) {

        num_row <- 0L
        num_col <- 0L

        # Get numeric data and numeric features
        .Call(
          LGBM_DatasetGetNumData_R
          , private$handle
          , num_row
        )
        .Call(
          LGBM_DatasetGetNumFeature_R
          , private$handle
          , num_col
        )
        return(c(num_row, num_col))

      } else if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {

        # Check if dgCMatrix (sparse matrix column compressed)
        # NOTE: requires Matrix package
        return(dim(private$raw_data))

      } else {

        # Trying to work with unknown dimensions is not possible
        stop(
          "dim: cannot get dimensions before dataset has been constructed, "
          , "please call gpb.Dataset.construct explicitly"
        )

      }

    },

    # Get column names
    get_colnames = function() {

      # Check for handle
      if (!gpb.is.null.handle(x = private$handle)) {

        # Get feature names and write them
        private$colnames <- .Call(
          LGBM_DatasetGetFeatureNames_R
          , private$handle
        )
        return(private$colnames)

      } else if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {

        # Check if dgCMatrix (sparse matrix column compressed)
        return(colnames(private$raw_data))

      } else {

        stop(
          "dim: cannot get colnames before dataset has been constructed, please call "
          , "gpb.Dataset.construct explicitly"
        )

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
      if (length(colnames) == 0L || sum(colnames == "") > 0) {
        return(invisible(self))
      }

      # Write column names
      private$colnames <- colnames
      if (!gpb.is.null.handle(x = private$handle)) {

        # Merge names with tab separation
        merged_name <- paste0(as.list(private$colnames), collapse = "\t")
        .Call(
          LGBM_DatasetSetFeatureNames_R
          , private$handle
          , merged_name
        )

      }

      return(invisible(self))

    },

    # Get information
    getinfo = function(name) {

      # Create known attributes list
      INFONAMES <- c("label", "weight", "init_score", "group")

      # Check if attribute key is in the known attribute list
      if (!is.character(name) || length(name) != 1L || !name %in% INFONAMES) {
        stop("getinfo: name must one of the following: ", paste0(sQuote(INFONAMES), collapse = ", "))
      }

      # Check for info name and handle
      if (is.null(private$info[[name]])) {

        if (gpb.is.null.handle(x = private$handle)) {
          stop("Cannot perform getinfo before constructing Dataset.")
        }

        # Get field size of info
        info_len <- 0L
        .Call(
          LGBM_DatasetGetFieldSize_R
          , private$handle
          , name
          , info_len
        )

        # Check if info is not empty
        if (info_len > 0L) {

          # Get back fields
          ret <- NULL
          ret <- if (name == "group") {
            integer(info_len) # Integer
          } else {
            numeric(info_len) # Numeric
          }

          .Call(
            LGBM_DatasetGetField_R
            , private$handle
            , name
            , ret
          )

          private$info[[name]] <- ret

        }
      }

      return(private$info[[name]])

    },

    # Set information
    setinfo = function(name, info) {

      # Create known attributes list
      INFONAMES <- c("label", "weight", "init_score", "group")

      # Check if attribute key is in the known attribute list
      if (!is.character(name) || length(name) != 1L || !name %in% INFONAMES) {
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

      if (!gpb.is.null.handle(x = private$handle) && !is.null(info)) {

        if (length(info) > 0L) {

          .Call(
            LGBM_DatasetSetField_R
            , private$handle
            , name
            , info
            , length(info)
          )

          private$version <- private$version + 1L

        }

      }

      return(invisible(self))

    },

    # Slice dataset
    slice = function(idxset, ...) {

      # Perform slicing
      return(
        Dataset$new(
          data = NULL
          , params = private$params
          , reference = self
          , colnames = private$colnames
          , categorical_feature = private$categorical_feature
          , predictor = private$predictor
          , free_raw_data = private$free_raw_data
          , used_indices = sort(idxset, decreasing = FALSE)
          , info = NULL
          , ...
        )
      )

    },

    # [description] Update Dataset parameters. If it has not been constructed yet,
    #               this operation just happens on the R side (updating private$params).
    #               If it has been constructed, parameters will be updated on the C++ side
    update_params = function(params) {
      if (length(params) == 0L) {
        return(invisible(self))
      }
      if (gpb.is.null.handle(x = private$handle)) {
        private$params <- modifyList(private$params, params)
      } else {
        tryCatch({
          .Call(
            LGBM_DatasetUpdateParamChecking_R
            , gpb.params2str(params = private$params)
            , gpb.params2str(params = params)
          )
        }, error = function(e) {
          # If updating failed but raw data is not available, raise an error because
          # achieving what the user asked for is not possible
          if (is.null(private$raw_data)) {
            stop(e)
          }
          
          # If updating failed but raw data is available, modify the params
          # on the R side and re-set ("deconstruct") the Dataset
          private$params <- modifyList(private$params, params)
          private$finalize()
        })
      }
      return(invisible(self))

    },

    get_params = function() {
      dataset_params <- unname(unlist(.DATASET_PARAMETERS()))
      ret <- list()
      for (param_key in names(private$params)) {
        if (param_key %in% dataset_params) {
          ret[[param_key]] <- private$params[[param_key]]
        }
      }
      return(ret)
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
      private$finalize()
      return(invisible(self))

    },

    # Set reference
    set_reference = function(reference) {

      # Set known references
      self$set_categorical_feature(categorical_feature = reference$.__enclos_env__$private$categorical_feature)
      self$set_colnames(colnames = reference$get_colnames())
      private$set_predictor(predictor = reference$.__enclos_env__$private$predictor)

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
        if (!gpb.check.r6.class(object = reference, name = "gpb.Dataset")) {
          stop("set_reference: Can only use gpb.Dataset as a reference")
        }

      }

      # Store reference
      private$reference <- reference

      # Finalize and return self
      private$finalize()
      return(invisible(self))

    },

    # Save binary model
    save_binary = function(fname) {

      # Store binary data
      self$construct()
      .Call(
        LGBM_DatasetSaveBinary_R
        , private$handle
        , fname
      )
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
    free_raw_data = FALSE,
    used_indices = NULL,
    info = NULL,
    version = 0L,
    
    # Finalize will free up the handles
    finalize = function() {
      .Call(
        LGBM_DatasetFree_R
        , private$handle
      )
      private$handle <- NULL
      return(invisible(NULL))
    },
    
    # Get handle
    get_handle = function() {

      # Get handle and construct if needed
      if (gpb.is.null.handle(x = private$handle)) {
        self$construct()
      }
      return(private$handle)

    },

    # Set predictor
    set_predictor = function(predictor) {

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
        if (!gpb.check.r6.class(object = predictor, name = "gpb.Predictor")) {
          stop("set_predictor: Can only use gpb.Predictor as predictor")
        }

      }

      # Store predictor
      private$predictor <- predictor

      # Finalize and return self
      private$finalize()
      return(invisible(self))

    }

  )
)

#' @title Construct \code{gpb.Dataset} object
#' @description Construct \code{gpb.Dataset} object from dense matrix, sparse matrix
#'              or local file (that was created previously by saving an \code{gpb.Dataset}).
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param params a list of parameters. See
#'               \href{https://github.com/fabsig/GPBoost/blob/master/docs/Parameters.rst#dataset-parameters}{
#'               the "Dataset Parameters" section of the parameter documentation} for a list of parameters
#'               and valid values.
#' @param reference reference dataset. When GPBoost creates a Dataset, it does some preprocessing like binning
#'                  continuous features into histograms. If you want to apply the same bin boundaries from an existing
#'                  dataset to new \code{data}, pass that existing Dataset to this argument.
#' @param colnames names of columns
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#' @param free_raw_data GPBoost constructs its data format, called a "Dataset", from tabular data.
#'                      By default, this Dataset object on the R side does keep a copy of the raw data.
#'                      If you set \code{free_raw_data = TRUE}, no copy of the raw data is kept (this reduces memory usage)
#' @param info a list of information of the \code{gpb.Dataset} object
#' @param ... other information to pass to \code{info} or parameters pass to \code{params}
#'
#' @return constructed dataset
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data_file <- tempfile(fileext = ".data")
#' gpb.Dataset.save(dtrain, data_file)
#' dtrain <- gpb.Dataset(data_file)
#' gpb.Dataset.construct(dtrain)
#' }
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
  return(
    invisible(Dataset$new(
      data = data
      , params = params
      , reference = reference
      , colnames = colnames
      , categorical_feature = categorical_feature
      , predictor = NULL
      , free_raw_data = free_raw_data
      , used_indices = NULL
      , info = info
      , ...
    ))
  )

}

#' @name gpb.Dataset.create.valid
#' @title Construct validation data
#' @description Construct validation data according to training data
#' @param dataset \code{gpb.Dataset} object, training data
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param info a list of information of the \code{gpb.Dataset} object
#' @param ... other information to pass to \code{info}.
#'
#' @return constructed dataset
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' }
#' @export
gpb.Dataset.create.valid <- function(dataset, data, info = list(), ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x = dataset)) {
    stop("gpb.Dataset.create.valid: input data should be an gpb.Dataset object")
  }

  # Create validation dataset
  return(invisible(dataset$create_valid(data = data, info = info, ...)))

}

#' @name gpb.Dataset.construct
#' @title Construct Dataset explicitly
#' @description Construct Dataset explicitly
#' @param dataset Object of class \code{gpb.Dataset}
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.construct(dtrain)
#' }
#' @return constructed dataset
#' @export
gpb.Dataset.construct <- function(dataset) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x = dataset)) {
    stop("gpb.Dataset.construct: input data should be an gpb.Dataset object")
  }

  # Construct the dataset
  return(invisible(dataset$construct()))

}

#' @title Dimensions of an \code{gpb.Dataset}
#' @description Returns a vector of numbers of rows and of columns in an \code{gpb.Dataset}.
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
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#'
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' stopifnot(ncol(dtrain) == ncol(train$data))
#' stopifnot(all(dim(dtrain) == dim(train$data)))
#' }
#' @rdname dim
#' @export
dim.gpb.Dataset <- function(x, ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x = x)) {
    stop("dim.gpb.Dataset: input data should be an gpb.Dataset object")
  }

  return(x$dim())

}

#' @title Handling of column names of \code{gpb.Dataset}
#' @description Only column names are supported for \code{gpb.Dataset}, thus setting of
#'              row names would have no effect and returned row names would be NULL.
#' @param x object of class \code{gpb.Dataset}
#' @param value a list of two elements: the first one is ignored
#'              and the second one is column names
#'
#' @details
#' Generic \code{dimnames} methods are used by \code{colnames}.
#' Since row names are irrelevant, it is recommended to use \code{colnames} directly.
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.construct(dtrain)
#' dimnames(dtrain)
#' colnames(dtrain)
#' colnames(dtrain) <- make.names(seq_len(ncol(train$data)))
#' print(dtrain, verbose = TRUE)
#' }
#' @rdname dimnames.gpb.Dataset
#' @return A list with the dimension names of the dataset
#' @export
dimnames.gpb.Dataset <- function(x) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x = x)) {
    stop("dimnames.gpb.Dataset: input data should be an gpb.Dataset object")
  }

  # Return dimension names
  return(list(NULL, x$get_colnames()))

}

#' @rdname dimnames.gpb.Dataset
#' @return A list with the dimension names of the dataset
#' @export
`dimnames<-.gpb.Dataset` <- function(x, value) {

  # Check if invalid element list
  if (!identical(class(value), "list") || length(value) != 2L) {
    stop("invalid ", sQuote("value"), " given: must be a list of two elements")
  }

  # Check for unknown row names
  if (!is.null(value[[1L]])) {
    stop("gpb.Dataset does not have rownames")
  }

  if (is.null(value[[2L]])) {

    x$set_colnames(colnames = NULL)
    return(x)

  }

  # Check for unmatching column size
  if (ncol(x) != length(value[[2L]])) {
    stop(
      "can't assign "
      , sQuote(length(value[[2L]]))
      , " colnames to an gpb.Dataset with "
      , sQuote(ncol(x))
      , " columns"
    )
  }

  # Set column names properly, and return
  x$set_colnames(colnames = value[[2L]])
  return(x)

}

#' @title Slice a dataset
#' @description Get a new \code{gpb.Dataset} containing the specified rows of
#'              original \code{gpb.Dataset} object
#' @param dataset Object of class \code{gpb.Dataset}
#' @param idxset an integer vector of indices of rows needed
#' @param ... other parameters (currently not used)
#' @return constructed sub dataset
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#'
#' dsub <- gpboost::slice(dtrain, seq_len(42L))
#' gpb.Dataset.construct(dsub)
#' labels <- gpboost::getinfo(dsub, "label")
#' }
#' @export
slice <- function(dataset, ...) {
  UseMethod("slice")
}

#' @rdname slice
#' @export
slice.gpb.Dataset <- function(dataset, idxset, ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x = dataset)) {
    stop("slice.gpb.Dataset: input dataset should be an gpb.Dataset object")
  }

  # Return sliced set
  return(invisible(dataset$slice(idxset = idxset, ...)))

}

#' @name getinfo
#' @title Get information of an \code{gpb.Dataset} object
#' @description Get one attribute of a \code{gpb.Dataset}
#' @param dataset Object of class \code{gpb.Dataset}
#' @param name the name of the information field to get (see details)
#' @param ... other parameters
#' @return info data
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item \code{label}: label gpboost learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item{\code{group}: used for learning-to-rank tasks. An integer vector describing how to
#'         group rows together as ordered results from the same set of candidate results to be ranked.
#'         For example, if you have a 100-document dataset with \code{group = c(10, 20, 40, 10, 10, 10)},
#'         that means that you have 6 groups, where the first 10 records are in the first group,
#'         records 11-30 are in the second group, etc.}
#'     \item \code{init_score}: initial score is the base prediction gpboost will boost from.
#' }
#'
#' @examples
#' \donttest{
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
#' }
#' @export
getinfo <- function(dataset, ...) {
  UseMethod("getinfo")
}

#' @rdname getinfo
#' @return info data
#' @export
getinfo.gpb.Dataset <- function(dataset, name, ...) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x = dataset)) {
    stop("getinfo.gpb.Dataset: input dataset should be an gpb.Dataset object")
  }

  return(dataset$getinfo(name = name))

}

#' @name setinfo
#' @title Set information of an \code{gpb.Dataset} object
#' @description Set one attribute of a \code{gpb.Dataset}
#' @param dataset Object of class \code{gpb.Dataset}
#' @param name the name of the field to get
#' @param info the specific field of information to set
#' @param ... other parameters
#' @return the dataset you passed in
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item{\code{label}: vector of labels to use as the target variable}
#'     \item{\code{weight}: to do a weight rescale}
#'     \item{\code{init_score}: initial score is the base prediction gpboost will boost from}
#'     \item{\code{group}: used for learning-to-rank tasks. An integer vector describing how to
#'         group rows together as ordered results from the same set of candidate results to be ranked.
#'         For example, if you have a 100-document dataset with \code{group = c(10, 20, 40, 10, 10, 10)},
#'         that means that you have 6 groups, where the first 10 records are in the first group,
#'         records 11-30 are in the second group, etc.}
#' }
#'
#' @examples
#' \donttest{
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
#' }
#' @export
setinfo <- function(dataset, ...) {
  UseMethod("setinfo")
}

#' @rdname setinfo
#' @return the dataset you passed in
#' @export
setinfo.gpb.Dataset <- function(dataset, name, info, ...) {

  if (!gpb.is.Dataset(x = dataset)) {
    stop("setinfo.gpb.Dataset: input dataset should be an gpb.Dataset object")
  }

  # Set information
  return(invisible(dataset$setinfo(name = name, info = info)))
}

#' @name gpb.Dataset.set.categorical
#' @title Set categorical feature of \code{gpb.Dataset}
#' @description Set the categorical features of an \code{gpb.Dataset} object. Use this function
#'              to tell GPBoost which features should be treated as categorical.
#' @param dataset object of class \code{gpb.Dataset}
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data_file <- tempfile(fileext = ".data")
#' gpb.Dataset.save(dtrain, data_file)
#' dtrain <- gpb.Dataset(data_file)
#' gpb.Dataset.set.categorical(dtrain, 1L:2L)
#' }
#' @rdname gpb.Dataset.set.categorical
#' @export
gpb.Dataset.set.categorical <- function(dataset, categorical_feature) {

  if (!gpb.is.Dataset(x = dataset)) {
    stop("gpb.Dataset.set.categorical: input dataset should be an gpb.Dataset object")
  }

  # Set categoricals
  return(invisible(dataset$set_categorical_feature(categorical_feature = categorical_feature)))

}

#' @name gpb.Dataset.set.reference
#' @title Set reference of \code{gpb.Dataset}
#' @description If you want to use validation data, you should set reference to training data
#' @param dataset object of class \code{gpb.Dataset}
#' @param reference object of class \code{gpb.Dataset}
#'
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package ="gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "gpboost")
#' test <- agaricus.test
#' dtest <- gpb.Dataset(test$data, test = train$label)
#' gpb.Dataset.set.reference(dtest, dtrain)
#' }
#' @rdname gpb.Dataset.set.reference
#' @export
gpb.Dataset.set.reference <- function(dataset, reference) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x = dataset)) {
    stop("gpb.Dataset.set.reference: input dataset should be an gpb.Dataset object")
  }

  # Set reference
  return(invisible(dataset$set_reference(reference = reference)))
}

#' @name gpb.Dataset.save
#' @title Save \code{gpb.Dataset} to a binary file
#' @description Please note that \code{init_score} is not saved in binary file.
#'              If you need it, please set it again after loading Dataset.
#' @param dataset object of class \code{gpb.Dataset}
#' @param fname object filename of output file
#'
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "gpboost")
#' train <- agaricus.train
#' dtrain <- gpb.Dataset(train$data, label = train$label)
#' gpb.Dataset.save(dtrain, tempfile(fileext = ".bin"))
#' }
#' @export
gpb.Dataset.save <- function(dataset, fname) {

  # Check if dataset is not a dataset
  if (!gpb.is.Dataset(x = dataset)) {
    stop("gpb.Dataset.set: input dataset should be an gpb.Dataset object")
  }

  # File-type is not matching
  if (!is.character(fname)) {
    stop("gpb.Dataset.set: fname should be a character or a file connection")
  }

  # Store binary
  return(invisible(dataset$save_binary(fname = fname)))
}
