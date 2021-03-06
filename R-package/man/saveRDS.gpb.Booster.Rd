% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/saveRDS.gpb.Booster.R
\name{saveRDS.gpb.Booster}
\alias{saveRDS.gpb.Booster}
\title{saveRDS for \code{gpb.Booster} models}
\usage{
saveRDS.gpb.Booster(object, file, ascii = FALSE, version = NULL,
  compress = TRUE, refhook = NULL, raw = TRUE)
}
\arguments{
\item{object}{R object to serialize.}

\item{file}{a connection or the name of the file where the R object is saved to or read from.}

\item{ascii}{a logical. If TRUE or NA, an ASCII representation is written; otherwise (default),
a binary one is used. See the comments in the help for save.}

\item{version}{the workspace format version to use. \code{NULL} specifies the current default
version (2). Versions prior to 2 are not supported, so this will only be relevant
when there are later versions.}

\item{compress}{a logical specifying whether saving to a named file is to use "gzip" compression,
or one of \code{"gzip"}, \code{"bzip2"} or \code{"xz"} to indicate the type of
compression to be used. Ignored if file is a connection.}

\item{refhook}{a hook function for handling reference objects.}

\item{raw}{whether to save the model in a raw variable or not, recommended to leave it to \code{TRUE}.}
}
\value{
NULL invisibly.
}
\description{
Attempts to save a model using RDS. Has an additional parameter (\code{raw})
             which decides whether to save the raw model or not.
}
\examples{
\donttest{
library(gpboost)
data(agaricus.train, package = "gpboost")
train <- agaricus.train
dtrain <- gpb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "gpboost")
test <- agaricus.test
dtest <- gpb.Dataset.create.valid(dtrain, test$data, label = test$label)
params <- list(objective = "regression", metric = "l2")
valids <- list(test = dtest)
model <- gpb.train(
    params = params
    , data = dtrain
    , nrounds = 10L
    , valids = valids
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 5L
)
model_file <- tempfile(fileext = ".rds")
saveRDS.gpb.Booster(model, model_file)
}
}
