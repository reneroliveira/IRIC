#' RandomSampling: Implementation of Random Sampling Algorithm
#' @description This function implements random undersampling and oversampling algorithm.
#' @param x  A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param percOver Oversampling percentage.
#' @param percUnder Undersampling percentage.
#' @return (newData) A data frame of the random oversampled/undersampled data.
#' @examples
#' library(caret)
#'
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' newData<- RandomSampling(x, y)
#' @export
RandomSampling <-
    function(x, y, percOver = 0, percUnder = 6.8)
    {
        if (percUnder > 100)
            stop("percUnder must be less than 100")

        # the column where the target variable is
        data <- data.frame(x, y)
        tgt <- length(data)
        classTable <- table(data[, tgt])

        # find the minirty and majority class
        minCl <- names(which.min(classTable))
        majCl <- names(which.max(classTable))

        # get the cases of the minority and majority class
        indexMin <- which(data[, tgt] == minCl)
        numMin  <- length(indexMin)
        indexMaj <- which(data[, tgt] == majCl)
        numMaj  <- length(indexMaj)

        # get the number of instances after sampling
        numMajUnder <- round(percUnder*numMaj/100)
        numMinOver  <- round(percOver*numMin/100)
        if (numMinOver + numMin > numMajUnder)
            warning("More minority instances than majority instances ")

        indexMajUnder <- sample(indexMaj, numMajUnder)
        indexMinOver  <- sample(indexMin, numMinOver, replace = TRUE)
        newIndex  <- c(indexMajUnder, indexMinOver, indexMin)
        data[newIndex, ]
    }






