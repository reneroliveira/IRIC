#' RandomSampling: Implementation of Random Sampling Algorithm
#' @description This function implements random undersampling and oversampling algorithm.
#' @param form A model formula.
#' @param data A data frame of training data.
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
#' newData<- RandomSampling(Churn ~., trainset)
#' @export
RandomSampling <-
    function(form, data, percOver = 0, percUnder = 6.8)
    {
        if (percUnder > 100)
            stop("percUnder must be less than 100")

        # # the column where the target variable is
        # data <- data.frame(x, y)
        # tgt <- length(data)
        # classTable <- table(data[, tgt])

        # find the class variable
        tgt <- which(names(data) == as.character(form[[2]]))
        classTable<- table(data[, tgt])

        # find the minority and majority class
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






