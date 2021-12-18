#Copyright (C) 2018 Bing Zhu
#' SmoteTL - Smote sampling + TomekLinks
#' @description This function implements SmoteTL, which performs over-sampling with SMOTE and clean data with Tomek Links.
#' @param x A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param percOver Number of new instance generated for each minority instance.
#' @param k Number of nearest neighbors used in Smote.
#' @return
#' \item{newData}{A data frame after the application of SmoteTL.}
#' @references G. E. Batista, R. C. Prati, M. C. Monard. A study of the behavior of several methods for balancing machine learning training data. ACM SIGKDD Explorations Newsletter , 6 (1) pp. 20 - 29.
#' @examples
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' newData<- SmoteTL(x, y, percOver = 1400)
#' @export
SmoteTL <-
    function(x, y, percOver = 1400, k = 5)
    {
        # source("code/Data level/SMOTE.R")
        newData <- SMOTE(x, y, percOver, k)
        tgt <- length(newData)
        indexTL <- TomekLink(tgt, newData)
        newDataRemoved <- newData[!indexTL, ]
        return(newDataRemoved)
    }

#' TomekLink
#' @description Function that finds the TomekLink
#' @param tgt Target columns index.
#' @param data Dataset.
#' @return
#' \item{indexTomek}{Logical vector indicating whether a instance is in TomekLinks.}
#' @importFrom RANN nn2
#' @export
TomekLink <-
    function(tgt, data)
    {

        indexTomek <- rep(FALSE, nrow(data))

        # find the column of class variable
        classTable <- table(data[, tgt])

        # seperate the group
        majCl <- names(which.max(classTable))
        minCl <- names(which.min(classTable))

        # get the instances of the larger group
        indexMin <- which(data[, tgt] == minCl)
        #numMin  <- length(indexMin)


        # convert dataset in numeric matrix
        # source("code/Data level/Numeralize.R")
        dataTransformed <- Numeralize(data[, -tgt])

        # generate indicator matrix
        # require("RANN")
        indexOrder1  <- nn2(dataTransformed, dataTransformed[indexMin, ], k = 2)$nn.idx
        indexTomekCa <- data[indexOrder1[, 2], tgt] == majCl
        if (sum(indexTomekCa) > 0)
        {
            TomekCa <- cbind(indexMin[indexTomekCa],indexOrder1[indexTomekCa, 2])

            # find nearest neighbour of potential majority instance
            indexOrder2 <- nn2(dataTransformed, dataTransformed[TomekCa[, 2], ], k = 2)$nn.idx
            indexPaired <- indexOrder2[ ,2] == TomekCa[, 1]
            if (sum(indexPaired) > 0)
            {
                indexTomek[TomekCa[indexPaired, 1]] <- TRUE
                indexTomek[TomekCa[indexPaired, 2]] <- TRUE
            }
        }
        return(indexTomek)
    }







