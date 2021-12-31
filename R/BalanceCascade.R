#Copyright (C) 2018 Bing Zhu
#================================================================
# BalanceCascade
#========================================================================
# Reference
# Xu-Ying, L., W. Jianxin, et al. (2009). "Exploratory # Undersampling
# for Class-Imbalance Learning."  Systems, Man, and Cybernetics, Part B:
# Cybernetics, IEEE Transactions on 39(2): 539-550.
#========================================================================

#' BalanceCascade
#' @description This function implements BalanceCascade algorithm for binary class imbalance classification.
#' @param x A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param iter Number of iterations for base classifiers training.
#' @param ... Arguments to be passed to methods (see below).
#' @return An object of class BalanceCascade, which is a list with the following components:
#' \item{call}{Function call}
#' \item{iter}{Number of iterations for base classifiers training.}
#' \item{classLabels}{Names of class labels.}
#' \item{base}{Types of base learner.}
#' \item{alphas}{Weights of base learners.}
#' \item{fits}{Fitted ensembled model.}
#' \item{thresh}{Threshold dor classification.}
#' @references X. Y. Liu, J. Wu and Z. H. Zhou. April 2009. \emph{Exploratory Undersampling for Class-Imbalance Learning.} IEEE Transactions on Systems, Man, and Cybernetics, Part B: Cybernetics, 2009, 39(2), pp. 539-550.
#' @examples
#' library(caret)
#'
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' model <- BalanceCascade(x, y, iter = 4)
#' output<- predict(model, x)
#' @export
BalanceCascade <-
    function(x, ...)
        UseMethod("BalanceCascade")

#' @export
#' @rdname BalanceCascade
BalanceCascade.data.frame  <-
    function (x, y, iter = 4, ...)
    {
        # Input:
        #        x: A data frame of the predictors from training data
        #        y: A vector of response variable from training data
        #     iter: Iterations to train base classifiers
        # allowParallel: A logical number to control the parallel computing. If allowParallel =TRUE, the function is run using parallel techniques

        # source("code/Ensemble-based level/BalanceBoost.R")
        funcCall <- match.call(expand.dots = FALSE)
        data <- data.frame(x, y)
        tgt <- length(data)
        classTable   <- table(data[, tgt])
        classTable   <- sort(classTable, decreasing = TRUE)
        classLabels  <- names(classTable)
        indexMaj <- which(data[, tgt] == classLabels[1])
        indexMin <- which(data[, tgt] == classLabels[2])
        numMin <- length(indexMin)
        numMaj <- length(indexMaj)
        FP <- (numMin/numMaj)^(1/(iter-1))

        #initialization
        x.nam <- names(x)
        form <- as.formula(paste("y ~ ", paste(x.nam, collapse = "+")))
        H      <- list()
        thresh <- rep(NA, iter)

        for (i in seq(iter)){
            if (length(indexMaj) < numMin)
                numMin  <- length(indexMaj)
            indexMajSampling <- sample(indexMaj, numMin)
            dataCurrent <- data[c(indexMin, indexMajSampling),]
            # H[[i]] <- bboost.data.frame(dataCurrent[, -tgt], dataCurrent[,tgt], type = "AdaBoost")
            H[[i]] <- bboost(dataCurrent[, -tgt], dataCurrent[,tgt], type = "AdaBoost")
            pred   <- predict.bboost(H[[i]], data[c(indexMaj), -tgt], type ="probability")
            sortIndex   <- order(pred[, 2], decreasing = TRUE)
            numkeep     <- round(length(indexMaj)*FP)
            thresh[i]   <- pred[sortIndex[numkeep],2]*sum(H[[i]]$alpha)
            indexMaj    <- indexMaj[sortIndex[1:numkeep]]
        }

        iter   <- sum(sapply(H,"[[", 5))
        fits   <- unlist(lapply(H,"[[", 6), recursive = FALSE)
        alphas <- unlist(lapply(H,"[[", 7))

        structure(
            list( call        = funcCall   ,
                  iter        = iter       ,
                  classLabels = classLabels,
                  base        = H[[1]]$base,
                  alphas      = alphas      ,
                  fits        = fits       ,
                  thresh      = sum(thresh))  ,
            class = "BalanceCascade")

    }

#' Predict Method for BalanceCascade Object
#' @description Predicting instances in test set using BalanceCascade object.
#' @param object An object of BalanceCascde class.
#' @param x A data frame of the predictors from testing data.
#' @param type Types of output, which can be **probability** and **class** (predicted label). Default is probability.
#' @param ... Additional arguments for predict method.
#' @return Two types of output can be selected:
#' \item{probability}{Estimated probability of being a minority instance. The probability is averaged by using an equal-weight majority vote by all weak learners.}
#' \item{class}{Predicted class of the instance. Instances of probability larger than 0.5 are predicted as 1, otherwise 0.}
#' @examples
#' library(caret)
#'
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' model <- BalanceCascade(x, y, allowParallel=TRUE)
#' output <- predict(model, x)
#' @export
predict.BalanceCascade<-
    function(object, x,  type = "class", ...)
    {

        #  input
        #     obj: Output from BalanceCascade.data.frame
        #       x: A data frame of the predictors from testing data

        if(is.null(x)) stop("please provide predictors for prediction")
        if (!type %in% c("class", "probability"))
            stop("wrong setting with type")
        data <- x
        classLabels <- object$classLabels
        numClass    <- length(classLabels)
        numIns      <- dim(data)[1]
        weight      <- object$alphas
        btPred      <- sapply(object$fits, object$base$pred, data = data, type ="class")
        classfinal  <- matrix(0, ncol = numClass, nrow = numIns)
        colnames(classfinal) <- classLabels
        for (i in 1:numClass){
            classfinal[, i] <- matrix(as.numeric(btPred == classLabels[i]), nrow = numIns)%*%weight
        }
        if (type == "class")
        {
            classfinal <- classfinal - object$thresh
            out <- factor(classLabels[apply(classfinal, 1, which.max)], levels = classLabels)

        } else {
            out <- data.frame(classfinal/rowSums(classfinal))

        }
        out
    }








