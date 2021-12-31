#Copyright (C) 2018 Bing Zhu
#=====================================================================
# BalanceBoost: Boosting based algorithm to deal with class imbalance
#=====================================================================
# SMOTEBoost, RUSBoost  AdaBoost, AdaC2
# Currently it only can be used to binary classification task
#====================================================================

#' BalanceBoost: Implementation of Boost-based Algorithms
#' @description This function implements boost-based algorithm for imbalance classification. Four algorithms can be found in the current version: Adaboost, SMOTEboost, RUSBoost, AdaC2.
#' @param x A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param iter Number of iterations for base classifiers training.
#' @param base Types of base learner.
#' @param type Type of boosting-based algorithm, including "AdaBoost", "SMOTEBoost", "RUSBoost", "AdaC2"
#' @param costRatio Cost Ratio between Majority class and Minority class.
#' @param ... Arguments to be passed to methods (see below).
#' @return An object of class bboost, which is a list with the following components:
#' \item{call}{Function call.}
#' \item{type}{Type of boosting-based algorithm.}
#' \item{base}{Types of base learner.}
#' \item{classLabels}{Names of class labels.}
#' \item{fits}{Fitted boosting-based model.}
#' \item{alpha}{Weights of base learners.}
#' @references N. Chawla, A. Lazarevic, L. Hall, K.W. Bowyer. \emph{SMOTEBoost: improving prediction of the minority class in boosting}. Proceeding of PKDD, 2003, pp. 107-119.
#'
#' Y. Sun, M.S. Kamel, A.K. Wong, Y. Wang. 2007. \emph{Cost-sensitive boosting for classification of imbalanced data}, Pattern Recognition, 2007, 40 (12), pp. 3358-3378.
#' @examples
#' library(caret)
#'
#' data(Korean)
#' sub <- createDataPartition (Korean$Churn, p=0.75, list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' model <- bboost(x, y, base = treeBoost, type = "AdaBoost")
#' output <- predict (model, x)
#' @export
bboost <-
    function(x, ...)
        UseMethod("bboost")

#' @export
#' @rdname bboost
bboost.data.frame <-
    function(x, y, iter = 40, base =  treeBoost, type = "AdaBoost", costRatio = 56/11, ...)
    {
        if (!type %in% c("AdaBoost", "SMOTEBoost","RUSBoost", "AdaC2"))
            stop("type must be AdaBoost, SMOTEBOost, RUSBoost or AdaC2")
        funcCall <- match.call(expand.dots = FALSE)


        # find the majority and minority class
        data <- data.frame(x, y)
        tgt <- length(data)
        classTable  <- table(data[, tgt])
        classTable  <- sort(classTable, decreasing = TRUE)
        classLabels <- names(classTable)
        indexMaj <- which(data[, tgt] == classLabels[1])
        indexMin <- which(data[, tgt] == classLabels[2])
        numMin <- length(indexMin)
        numMaj <- length(indexMaj)
        numRow <- dim(data)[1]

        #initialization
        x.nam <- names(x)
        form <- as.formula(paste("y~ ", paste(x.nam, collapse = "+")))
        H      <- list()
        alpha  <- rep(0, iter)
        oldWeight <- rep(1/numRow, numRow)
        newWeight <- rep(NA, numRow)
        count <- 0
        t <- 0
        earlyStop <- FALSE

        if (type == "AdaC2")
        {
            cost <- rep(1, numRow)
            cost[indexMin] <- costRatio
        }

        while (t < iter) {
            t <- t + 1
            #data preparation
            if (type == "AdaBoost" | type == "AdaC2"){
                indexBootstrap <- sample(1:numRow, replace = TRUE, prob = oldWeight)
                dataResample   <- data[indexBootstrap, ]
            }

            if (type == "SMOTEBoost") {
                # source("code/Data level/SMOTE.R")
                perOver  <- ((numMaj - numMin)/numMin)*100
                dataSmoteSample  <- SMOTE(form, data, perOver)
                numNew <- dim(dataSmoteSample)[1]
                resampleWeight <- rep(NA, numNew)
                resampleWeight[1:numRow] <- oldWeight
                resampleWeight[(numRow+1):numNew] <- 1/numNew
                indexBootstrap <- sample(1:numNew, replace = TRUE, prob = resampleWeight)
                dataResample   <- dataSmoteSample[indexBootstrap, ]
            }

            if (type == "RUSBoost") {
                indexMajRUS <- sample(1:numMaj, numMin)
                indexNew    <- c(indexMaj[indexMajRUS], indexMin)
                resampleWeight <- oldWeight[indexNew]
                indexBootstrap <- sample(1:(2*numMin), replace = TRUE, prob = resampleWeight)
                dataResample <- data[indexNew[indexBootstrap], ]
            }

            # build classifier with resampling
            H[[t]] <- base$fit(form, dataResample)

            # Computing the (pseudo) loss of hypothesis
            if (type == "AdaBoost")
            {
                weakPrediction <- base$pred(H[[t]], data, type = "class")
                ind  <- data[, tgt]== weakPrediction
                loss <- sum(oldWeight * as.numeric(!ind))
                beta <- loss/(1-loss)
                alpha[t] <- log(1/beta)
            }

            if (type == "RUSBoost" | type == "SMOTEBoost")
            {
                weakPrediction <- base$pred(H[[t]], data, type = "probability")
                loss <- sum(oldWeight * abs(weakPrediction[, 2] - as.numeric(data[, tgt]) + 1))
                beta <- loss/(1-loss)
                alpha[t]  <- log(1/beta)
            }

            if (type == "AdaC2")
            {
                weakPrediction <- base$pred(H[[t]], data, type = "class")
                ind  <- data[, tgt]== weakPrediction
                alpha[t]<- 0.5*log(sum(oldWeight[ind]* cost[ind])/sum(oldWeight[!ind]*cost[!ind]))
            }

            if ( alpha[t] < 0){
                count <- count + 1
                t <- t - 1
                if (count > 5){
                    earlyStop <- TRUE
                    warning("stop with too many big errors")
                    break
                } else {
                    next
                }
            } else {
                count <- 1
            }

            if (type == "AdaBoost")
            {
                newWeight[ind]   <- oldWeight[ind]*beta
                newWeight[!ind]  <- oldWeight[!ind ]
            }

            if (type == "RUSBoost" | type == "SMOTEBoost")
            {
                newWeight <- oldWeight*beta^(1-abs(weakPrediction[, 2] - as.numeric(data[, tgt]) + 1))
            }

            if (type == "AdaC2")
            {
                newWeight[ind]   <- oldWeight[ind]*exp(-alpha[t]) * cost[ind]
                newWeight[!ind]  <- oldWeight[!ind]*exp(alpha[t]) * cost[!ind]
            }

            newWeight <- newWeight / sum(newWeight)
            oldWeight <- newWeight
        }
        if (earlyStop) {
            iter <-  t
            alpha <- alpha[1:iter]
            H <- H[1:iter]
        }

        structure(
            list(call        = funcCall,
                 type        = type,
                 base        = base,
                 classLabels = classLabels,
                 iter        = iter,
                 fits        = H   ,
                 alpha       = alpha),
            class = "bboost")
    }

#' Predict method for bboost method
#' @description Predicting instances in test set using bboost object.
#' @param object An object of bboost class.
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
#' x<- trainset[, -11]
#' y<- trainset[, 11]
#' model <- bboost(x, y, base = treeBoost, type = "AdaC2")
#' output <- predict (model, x, type = "probability") # return probability estimation
#' output <- predict (model, x, type = "class") # return predicted class
#' @export
predict.bboost<-
    function(object, x, type = "class",...)
    {
        #  input
        #     obj: Output from bboost.formula
        #       x: A data frame of the predictors from testing data

        if(is.null(x)) stop("please provide predictors for prediction")
        data <- x
        btPred <- sapply(object$fits, object$base$pred, data = data)
        object$base$aggregate(btPred, object$alpha, object$classLabels, type=type)
    }

#' Basic tree for Boosting
#' @description Boosting Base learner
#' @export
treeBoost <- list(
    fit = function(form, data)
    {
        library(rpart)
        out<-rpart(form,data)
        return(out)
    },

    pred = function(object, data, type="class")
    {
        out <- predict(object, data,  type=type)
    },

    aggregate = function(x, weight, classLabels, type = "class")
    {
        if (!type %in% c("class", "probability"))
            stop("wrong setting with type")
        numClass   <- length(classLabels)
        numIns     <- dim(x)[1]
        iter       <- dim(x)[2]
        classfinal <- matrix(0, ncol = numClass, nrow = numIns)
        colnames(classfinal) <- classLabels
        for (i in 1:numClass){
            classfinal[,i] <- matrix(as.numeric(x == classLabels[i]), nrow = numIns)%*%weight
        }
        if(type == "class")
        {
            out <- factor(classLabels[apply(classfinal, 1, which.max)], levels = classLabels )
        } else {
            out <-  classfinal/rowSums(classfinal)
        }
        out
    })




