% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CLUS.R
\name{CLUS}
\alias{CLUS}
\title{Clustering-based Undersampling Method}
\usage{
CLUS(x, y, k = 3, m = 1.5)
}
\arguments{
\item{x}{A data frame of the predictors from training data.}

\item{y}{A vector of response variable from training data.}

\item{k}{Number of clusters.}

\item{m}{Imbalanced ratio in output dataset.}
}
\value{
\item{newdata}{Data frame of the undersampled data using CLUS method.}
}
\description{
This function implements CLUS sampling (clustering-based undersampling), which selects the representative data for training data to improve the classification accuracy for minority class.
}
\examples{
library(caret)

data(Korean)
sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
trainset <- Korean[sub,]
testset <- Korean[-sub,]
x <- trainset[, -11]
y <- trainset[, 11]
newData<- CLUS(x, y, m=2)
}
\references{
Yen, S.-J. and Y.-S. Lee (2009). \emph{"Cluster-based under-sampling approaches for imbalanced data distributions."} Expert Systems with Applications 36(3): 5718-5727.
}
