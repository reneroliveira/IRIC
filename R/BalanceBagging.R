#========================================================================
# BalanceBagging: Bagging based algorithm to deal with class imbalance
#========================================================================
# Bagging, RUSBagging,  SMOTEBagging, Rougly Balanced Bagging
# Currently it only can be used to binary classification task
#=======================================================================

#' BalanceBagging - Bagging based algorithms to deal with class imbalance
#'
#' @description This function implements bagging-based algorithm for imbalance classification. Four algorithms can be found in the current version: SMOTEBagging, RUSBagging, RBBagging and ROSBagging. Currently it only can be used to binary classification task
#' @param x A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param numBag Number of bag.
#' @param base Base learner
#' @param type Type of bagging-based algorithm, including "SMOTEBagging","RUSBagging","RBBagging" and "ROSBagging".
#' @param allowParallel A logical number to control the parallel computing. If allowParallel = TRUE, the function is run using parallel techniques.
#' @param ... Arguments to be passed to methods (see below).
#' @return An object of class bbag, which is a list with the following components:
#' \item{call}{Function call.}
#' \item{base}{Types of base learner.}
#' \item{type}{Type of bagging-based algorithm.}
#' \item{fits}{Fitted bagging-based model.}
#' @examples
#' library(caret)
#'
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn, p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' model <- bbagging(x, y, type = "SMOTEBagging", allowParallel=TRUE)
#' output <- predict (model, x)
#' @import rpart
#' @import parallel
#' @importFrom stats as.formula rnbinom
#' @importFrom iterators iter
#' @export
#' @references S. Hido, H. Kashima, Y. Takahashi. \emph{Roughly balanced bagging for imbalanced data.} Statistical Analysis & Data Mining, 2009, 2(5-6), pp.412-426.
#'
#' S. Wang, X. Yao. 2009. \emph{Diversity analysis on imbalanced data sets by using ensemble models.} IEEE Symposium on Computational Intelligence, 2009, pp. 324-331.
bbagging <-
    function(x, ...)
        UseMethod("bbagging")


#' @export
#' @import foreach
#' @import doParallel
#' @rdname bbagging
bbagging.data.frame <-
    function(x, y, numBag = 40, base = treeBag, type = "SMOTEBagging", allowParallel = FALSE, ...)
    {
        # library(foreach)
        # if (allowParallel) library(doParallel)

        funcCall <- match.call(expand.dots = FALSE)
        if (!type %in% c( "RUSBagging", "ROSBagging", "SMOTEBagging", "RBBagging"))
            stop("wrong setting with method type")

        data <- data.frame(x,y)
        tgt <- length(data)
        x.nam <- names(x)
        form <- as.formula(paste("y~ ", paste(x.nam, collapse = "+")))
        classTable  <- table(data[, tgt])
        classTable  <- sort(classTable, decreasing = TRUE)
        classLabels  <- names(classTable)

        CreateResample <- function(data, tgt, classLabels, type, numBag, iter,...)
        {
            indexMaj <- which(data[, tgt] == classLabels[1])
            indexMin <- which(data[, tgt] == classLabels[2])
            numMin   <- length(indexMin)
            numMaj   <- length(indexMaj)

            #do RUSBagging
            if (type  == "RUSBagging"){
                indexMajSampled <- sample(indexMaj, numMin, replace = FALSE)
                indexMinSampled <- sample(indexMin, numMin, replace = TRUE)
                indexNew <- c(indexMajSampled, indexMinSampled)
                newData  <- data[indexNew, ]
            }

            #do ROSBagging
            if (type == "ROSBagging"){
                indexMajSampled <- sample(indexMaj, numMaj, replace = TRUE)
                numMinsampled   <- numMaj - numMin
                indexMinSampled <- sample(indexMin, numMinsampled, replace = TRUE)
                indexNew <- c(indexMajSampled, indexMinSampled, indexMin)
                newData  <- data[indexNew,]
            }
            #do RBBaging
            if (type == "RBBagging"){
                numMajSampled   <- rnbinom(1, numMin, 0.5)
                indexMajSampled <- sample(indexMaj, numMajSampled, replace = FALSE)
                indexMinSampled <- sample(indexMin, numMin, replace = TRUE)
                indexNew <- c(indexMajSampled, indexMinSampled)
                newData  <- data[indexNew,]
            }
            #do smotebagging
            if (type == "SMOTEBagging"){
                # source("code/Data level/SmoteExs.R")
                numCol <- dim(data)[2]
                n <- (iter-1) %/% (numBag/10) + 1
                numMinSampled   <- round(numMaj * n/10)
                indexMinSampled <- sample(indexMin, numMinSampled, replace = TRUE)
                indexMajSampled <- sample(indexMaj, numMaj, replace = TRUE)
                indexNew <- c(indexMajSampled, indexMinSampled)
                dataROS  <- data[indexNew, ]
                perOver  <- round((numMaj - numMinSampled)/numMin)*100
                if (perOver > 0){
                    if (tgt < numCol)
                    {
                        cols <- 1:numCol
                        cols[c(tgt, numCol)] <- cols[c(numCol, tgt)]
                        data <- data[, cols]
                    }
                    newExs <- SmoteExs(data[indexMin, ], perOver, k = 5)
                    if (tgt < numCol)
                    {
                        newExs <- newExs[, cols]
                        data   <- data[, cols]
                    }
                    newData <- rbind(dataROS, newExs)

                } else {
                    newData <- dataROS
                }
            }
            return(newData)
        }
        fitter <- function(form, data, tgt, classLabels, type, base, numBag, iter, ...)
        {
            dataSampled <- CreateResample(data, tgt, classLabels, type, numBag, iter,...)
            model <- base$fit(form, dataSampled)
        }

        if (allowParallel) {
            `%op%` <- `%dopar%`
            cl <- makeCluster(detectCores()/2)
            registerDoParallel(cl)
        } else {
            `%op%` <- `%do%`
        }

        btFits <- foreach(iter = seq(1:numBag),
                          .verbose = FALSE,
                          .errorhandling = "stop") %op% fitter(form, data, tgt, classLabels, type, base, numBag, iter, ...)

        if (allowParallel) stopCluster(cl)

        structure(
            list(call         = funcCall,
                 base         = base    ,
                 type         = type    ,
                 numBag       = numBag  ,
                 classLabels  = classLabels ,
                 fits         = btFits) ,
            class = "bbag")
    }

#' Predict Method for bbagging object
#' @description Predicting instances in test set using bbagging object.
#' @param object An object of bbaging class.
#' @param x A data frame of the predictors from testing data.
#' @param type Types of output, which can be probability and class (predicted label). Default is probability.
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
#' x<- trainset[, -11]
#' y<- trainset[, 11]
#' model <- bbagging(x, y, type = "SMOTEBagging", allowParallel=TRUE)
#' output <- predict(model, x, type = "probability") # return probability estimation
#' output <- predict(model, x, type = "class") # return predicted class
#' @import rpart
#' @export
predict.bbag<-
    function(object, x, type = "class",...)
    {

        if(is.null(x)) stop("please provide predictors for prediction")
        data <- x
        btPred <- sapply(object$fits, object$base$pred, data = data)
        object$base$aggregate(btPred, object$classLabels, type)
    }


#' Basic tree for Bagging
#' @description Bagging Base learner
#' @export
#' @keywords internal
treeBag <- list(
    fit = function(form, data)
    {
        #options(java.parameters="-Xmx8048m")
        # library("RWeka")
        # library(rpart)
        # out<- J48(form, data)
        out<-rpart(form,data)
        return(out)
    },

    pred = function(object, data)
    {
        out <- predict(object, data, type = "class")
    },

    aggregate = function(x, classLabels, type)
    {
        if (!type %in% c("class", "probability"))
            stop("wrong setting with type")
        numClass   <- length(classLabels)
        numIns     <- dim(x)[1]
        numBag     <- dim(x)[2]
        classfinal <- matrix(0, ncol = numClass, nrow = numIns)
        colnames(classfinal) <- classLabels
        for (i in 1:numClass){
            classfinal[,i] <- matrix(as.numeric(x == classLabels[i]), nrow = numIns)%*%rep(1, numBag)
        }

        if(type == "class"){
            out <- factor(classLabels[apply(classfinal, 1, which.max)], levels = classLabels )
        } else {
            out <- data.frame(classfinal/numBag)
        }
        out
    })









