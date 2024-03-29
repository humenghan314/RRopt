% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{RidgeRegression_ADMM}
\alias{RidgeRegression_ADMM}
\title{Solve ridge regression using ADMM.}
\usage{
RidgeRegression_ADMM(
  X,
  y,
  lambda,
  rho = 0.1,
  max_iter = 1000L,
  eps_abs = 1e-05,
  eps_rel = 1e-05
)
}
\arguments{
\item{X}{the independent variable X (size n×p)}

\item{y}{the response variable y (size n×1)}

\item{lambda}{the penalty factor (size 1×1)}

\item{rho}{the lagrange multiplier (size 1×1)}

\item{max_iter}{the maximum number of iterations}

\item{eps_abs}{the absolute convergence threshold}

\item{eps_rel}{the relative convergence threshold}
}
\value{
Estimated value of beta(last and historical), iterations and loss
}
\description{
Solve ridge regression using ADMM.
}
