#Copyright (C) 2018 Bing Zhu
# ===================================================
# SmoteENN: Smote+ENN
# ===================================================

#' SmoteENN - Implementation of SmoteENN Algorithm
#' @description This function implements SmoteENN algorithm, which combined SMOTE and data cleaning techniques ENN(Edited Nearest Neighbor).
#' @param x A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param percOver Percent of new instance generated for each minority instance.
#' @param k1 Number of the nearest neighbors.
#' @param k2 Number of neighbours for ENN.
#' @param allowParallel A logical number to control the parallel computing. If allowParallel = TRUE, the function is run using parallel techniques.
#' @return
#' \item{newData}{A data frame after the application of SmoteENN.}
#' @importFrom RANN nn2
#' @importFrom parallel makeCluster stopCluster parLapply parSapply parApply
#' @references G. E. Batista, R. C. Prati, M. C. Monard. A study of the behavior of several methods for balancing machine learning training data. ACM SIGKDD Explorations Newsletter , 6 (1) pp. 20 - 29.
#' @examples data(Korean)
#' library(caret)
#'
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' newData<- SmoteENN(x, y, percOver =1400 , allowParallel= TRUE)
#' @export
SmoteENN<-
    function(x, y, percOver = 1400, k1 = 5, k2 = 3, allowParallel= TRUE)
    {
        # source("code/Data level/SMOTE.R")
        newData <- SMOTE(x, y, percOver, k1)
        tgt <- length(newData)
        indexENN  <- ENN(tgt, newData, k2,allowParallel)
        newDataRemoved <- newData[!indexENN, ]
        return(newDataRemoved)
    }


# ===================================================
#  ENN: using ENN rule to find the noisy instances
# ===================================================

ENN <-
    function(tgt, data, k, allowParallel)
    {
        # find column of the target
        numRow  <- dim(data)[1]
        indexENN <- rep(FALSE, numRow)

        # transform the nominal data into  binary
        # source("code/Data level/Numeralize.R")
        dataTransformed <- Numeralize(data[, -tgt])
        classMode<-matrix(nrow=numRow)
        # library("RANN")
        indexOrder <- nn2(dataTransformed, dataTransformed, k+1)$nn.idx
        if  (allowParallel) {

            classMetrix <- matrix(data[indexOrder[,2:(k+1)], tgt], nrow = numRow)
            # library("parallel")
            cl <- makeCluster(2)
            classTable   <- parApply (cl, classMetrix, 1, table)
            modeColumn   <- parLapply(cl, classTable, which.max)
            classMode    <- parSapply(cl, modeColumn, names)
            stopCluster(cl)
            indexENN[data[, tgt]!= classMode] <- TRUE
        } else {

            for (i in 1:numRow)
            {
                classTable    <- table(data[indexOrder[i, ], tgt])
                classMode[i]  <- names(which.max(classTable))
            }
        }
        indexENN[data[, tgt]!= classMode] <- TRUE
        return(indexENN)
    }

