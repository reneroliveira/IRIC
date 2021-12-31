#Copyright (C) 2018 Bing Zhu
#========================================================================
# EasyEnsemble
#========================================================================
# Reference
# Xu-Ying, L., W. Jianxin, et al. (2009). "Exploratory # Undersampling
# for Class-Imbalance Learning."  Systems, Man, and Cybernetics, Part B:
# Cybernetics, IEEE Transactions on 39(2): 539-550.
#========================================================================
#' EasyEnsemble
#' @description This function implements EasyEnsemble algorithm for binary imbalance classification.
#' @param x A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param iter Number of iterations for base classifiers training.
#' @param allowParallel A logical number to control the parallel computing. If allowParallel = TRUE, the function is run using parallel techniques.
#' @param ... Arguments to be passed to methods (see below).
#' @return An object of class EasyEnsemble, which is a list with the following components:
#' \item{call}{Function call.}
#' \item{iter}{Number of iterations for base classifiers training.}
#' \item{fits}{Fitted ensembled model.}
#' \item{base}{Types of base learner.}
#' \item{alphas}{Weights of base learners.}
#' \item{classlabels}{names of class labels.}
#' @references X. Y. Liu, J. Wu and Z. H. Zhou. \emph{Exploratory Undersampling for Class-Imbalance Learning.} IEEE Transactions on Systems, Man, and Cybernetics, Part B: Cybernetics, 2009, 39(2), pp. 539-550.
#' @examples
#' library(caret)
#'
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' model <- EasyEnsemble(x, y, allowParallel=TRUE)
#' output <- predict (model, x)
#' @import doParallel
#' @import foreach
#' @export
EasyEnsemble <-
    function(x, ...)
        UseMethod("EasyEnsemble")

#' @export
#' @rdname EasyEnsemble
EasyEnsemble.data.frame <-
    function(x, y, iter = 4, allowParallel = FALSE, ...)
    {

        # library(foreach)
        # if (allowParallel) library(doParallel)

        funcCall <- match.call(expand.dots = FALSE)
        data <- data.frame(x, y)
        tgt <- length(data)
        #tgt <- which(names(data) == as.character(form[[2]]))
        classTable   <- table(data[, tgt])
        classTable   <- sort(classTable, decreasing = TRUE)
        classLabels  <- names(classTable)
        indexMaj <- which(data[, tgt] == classLabels[1])
        indexMin <- which(data[, tgt] == classLabels[2])
        numMin <- length(indexMin)
        numMaj <- length(indexMaj)

        #x.nam <- names(x)
        #form <- as.formula(paste("y~ ", paste(x.nam, collapse = "+")))
        H      <- list()

        fitter <- function(tgt, data, indexMaj, numMin, indexMin)
        {
            # source("code/Ensemble-based level/BalanceBoost.R")
            indexMajCurrent <- sample(indexMaj, numMin)
            dataCurrent <- data[c(indexMin, indexMajCurrent),]
            out <- bboost.data.frame(dataCurrent[, -tgt], dataCurrent[,tgt], type = "AdaBoost")
        }
        if (allowParallel) {
            `%op%` <- `%dopar%`
            cl <- makeCluster(2)
            registerDoParallel(cl)
        } else {
            `%op%` <- `%do%`
        }
        H  <- foreach(i = seq(1:iter),
                      .verbose = FALSE,
                      .errorhandling = "stop") %op% fitter(tgt, data , indexMaj, numMin, indexMin)

        if (allowParallel) stopCluster(cl)

        iter   <- sum(sapply(H,"[[", 5))
        fits   <- unlist(lapply(H,"[[", 6), recursive = FALSE)
        alphas <- unlist(lapply(H,"[[", 7))
        structure(
            list(call       = funcCall    ,
                 iter       = iter        ,
                 fits       = fits        ,
                 base       = H[[1]]$base ,
                 alphas     = alphas      ,
                 classLabels = classLabels),
            class = "EasyEnsemble")
    }


#' Predict Method for EasyEnsemble Object
#' @description Predicting instances in test set using EasyEnsemble object.
#' @param object An object of EasyEnsemble class.
#' @param x A data frame of the predictors from testing data.
#' @param type Types of output, which can be **probability** and **class** (predicted label). Default is **probability**.
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
#' model <- EasyEnsemble(x, y, allowParallel=TRUE)
#' output <- predict (model, x, type = "probability") # return probability estimation
#' output <- predict (model, x, type = "class") # return predicted class
#' @export
predict.EasyEnsemble <-
    function(object, x, type = "class",...)
    {

        #  input
        #     obj: Output from bboost.formula
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
            out <- factor(classLabels[apply(classfinal, 1, which.max)], levels = classLabels)
        } else {
            out <- data.frame(classfinal/rowSums(classfinal))
        }
        out

    }




