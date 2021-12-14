# =============================================================================
#  CLUS: clustering-based undersampling method
# =============================================================================
# Yen, S.-J. and Y.-S. Lee (2009).
# "Cluster-based under-sampling approaches for imbalanced data distributions."
# Expert Systems with Applications 36(3): 5718-5727.
#-----------------------------------------------------------------------------

#' CLUS: Clustering-based Undersampling Method
#' @description This function implements CLUS sampling (clustering-based undersampling), which selects the representative data for training data to improve the classification accuracy for minority class.
#' @param x A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param k Number of clusters.
#' @param m Imbalanced ratio in output dataset.
#' @return \item{newdata}{Data frame of the undersampled data using CLUS method.}
#' @importFrom caret createDataPartition
#' @examples
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' newData<- CLUS(x, y, m=2)
#' @references Yen, S.-J. and Y.-S. Lee (2009).
#' "Cluster-based under-sampling approaches for imbalanced data distributions."
#' Expert Systems with Applications 36(3): 5718-5727.
#' @export
CLUS <-
    function(x, y, k = 3, m = 1.5)
    {
        # find the majority and minority instances
        data <- data.frame(x, y)
        tgt <- length(data)
        classTable <- table(data[, tgt])
        numRow <- dim(data)[1]

        # find the minirty and majority
        minCl    <- names(which.min(classTable))
        indexMin <- which(data[, tgt] == minCl)
        numMin   <- length(indexMin)
        majCl    <- names(which.max(classTable))
        #indexMaj <- which(data[, tgt] == majCl)
        numMajFinal <- m*numMin

        # source("code/Data level/Numeralize.R")
        dataX            <- Numeralize(data[, -tgt])
        clusteringMoldel <- kmeans(dataX, k)
        mebership        <- clusteringMoldel$cluster
        indexGrouping    <- split(1:numRow, mebership)
        ratio         <- rep(0, k)
        MajCluster   <- list()
        numMajcluster <- rep(0, k)

        for (i in 1:k)
        {
            indexMajLocal    <- which(data[indexGrouping[[i]], tgt] == majCl)
            indexMajGlobal   <- indexGrouping[[i]][indexMajLocal]
            numMincluster    <- sum(data[indexGrouping[[i]], tgt] == minCl)
            MajCluster[[i]]  <- indexMajGlobal
            numMajcluster[i] <- length(indexMajGlobal)
            if (numMincluster == 0)
                numMincluster <- 1
            ratio[i] <- numMajcluster[i]/numMincluster
        }

        #control the imbalance ratio in the output datasets
        ratio <- ratio/sum(ratio)
        indexMajFinal <- c()
        for (i in 1:k)
        {
            if  (ratio[i]!=0)
            {
                numMajUnder   <- round(numMajFinal*ratio[i])
                indexSelection    <- sample(1:numMajcluster[i], numMajUnder, replace = TRUE)
                indexMajSelection <- MajCluster[[i]][indexSelection]
                indexMajFinal     <- c(indexMajFinal, indexMajSelection)
            }
        }

        newData   <- rbind(data[indexMin, ], data[indexMajFinal, ])
        return(newData)

    }






