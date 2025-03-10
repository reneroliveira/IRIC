#' Majority Weighted Minority Oversampling Technique (MWMOTE)
#' @description This function implements MWMOTE sampling (Majority Weighted Minority Oversampling Technique).
#' @param form A model formula.
#' @param data A data frame of training data.
#' @param percOver Percent of new instance generated for each minority instance. percOver/100 is the number of new instances per minority point.
#' @param k1 Number of neighbours for filtering.
#' @param k2 Number of neighbours for selecting majority instances.
#' @param CThresh Threshold to determine the number of clusters.
#' @importFrom RANN nn2
#' @importFrom stats cutree dist hclust
#' @return
#' \item{newData}{A data frame of the oversampled data using MWMOTE.}
#' @examples
#' library(caret)
#'
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' newData<- MWMOTE(Churn ~., trainset)
#' @references Barua, S., M. Islam, et al. (2014).\emph{"MWMOTE--Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning."} IEEE Transactions on Knowledge and Data Engineering, 26(2): 405-425.
#' @export
MWMOTE<-
    function(form, data, percOver = 1400, k1 = 5, k2 = 5, CThresh = 3)
    {
        # #numRow <- dim(data)[1]
        # data<-data.frame(x,y)
        # numCol <- dim(data)[2]
        #
        # # find the majority and minority instances
        # tgt <- length(data)
        # classTable <- table(data[, tgt])

        # find the class variable
        tgt <- which(names(data) == as.character(form[[2]]))
        classTable<- table(data[, tgt])
        numCol <- dim(data)[2]

        # find the minority and majority instances
        minCl   <- names(which.min(classTable))
        indexMin<- which(data[, tgt] == minCl)
        numMin  <- length(indexMin)
        majCl   <- names(which.max(classTable))
        indexMaj<- which(data[, tgt] == majCl)
        numMaj  <- length(indexMaj)
        k3      <- round(numMin/2)

        # source("code/Data level/Numeralize.R")
        dataTransformed  <- Numeralize(data[, -tgt])

        # find the nearest k1 borderline majority sets after filtering
        # require("RANN")
        indexOrder <- nn2(dataTransformed, dataTransformed[indexMin, ], k1+1)$nn.idx
        indexOrderMaj <- nn2(dataTransformed[indexMaj, ], dataTransformed[indexMin, ], k2+1)$nn.idx
        indexbmaj <- rep(FALSE, numMaj)
        indexMinf <- rep(FALSE, numMin)
        for (i in 1:numMin)
        {
            kNNsMaj <- which(data[indexOrder[i, 2:(k1+1)], tgt] == majCl)
            if (length(kNNsMaj) < k1)
            {
                IndexNNsMaj   <- indexOrderMaj[i, 2:(k2+1)]
                indexMinf[i] <- TRUE
                indexbmaj[IndexNNsMaj] <- TRUE
            }
        }
        bmaj <- dataTransformed[indexMaj[indexbmaj], ]

        numbmaj <- dim(bmaj)[1]
        weightMin <- rep(0, numMin)

        # find nearest k3 borderline minority sets
        kNNsMin<- nn2(dataTransformed[indexMin, ], bmaj, k3+1)

        Cfthresh <- 5
        Cfmax <- 2
        for (i in 1:numbmaj)
        {
            dl <- kNNsMin$nn.dists[i, 2:(k3+1)]
            dl <- dl^2/dim(bmaj)[2]
            cf <- 1/dl
            cf[cf > Cfthresh] <- Cfthresh
            cf <- cf*Cfmax/Cfthresh
            dl <- cf/(sum(cf))
            weightMin[kNNsMin$nn.idx[i, 2:(k3+1)]] <- dl*cf
        }
        selectProb <- weightMin/sum(weightMin)

        # moving the class attribute the last column
        if (tgt < numCol)
        {
            cols <- 1:numCol
            cols[c(tgt, numCol)] <- cols[c(numCol, tgt)]
            data <- data[, cols]
        }

        # transform factors into integer
        dataMin <- data[indexMin, ]
        nomatr <- c()
        for (col in 1:(numCol-1))
        {
            if (class(data[, col]) == "factor")
            {
                dataMin[, col] <- as.integer(dataMin[, col])
                nomatr <- c(nomatr, col)
            } else {
                dataMin[, col] <- dataMin[, col]
            }
        }
        dataMin<- data.matrix(dataMin)
        # clustering of minority instances

        distMetrixMinf <- nn2(dataTransformed[indexMin[indexMinf],], dataTransformed[indexMin[indexMinf], ], k=2)$nn.dists
        avgDists <- sum(distMetrixMinf[,2]^2)/sum(indexMinf)
        thresh   <- sqrt(avgDists*CThresh)
        distMetrixMin <- dist(dataTransformed[indexMin, ], method = "euclidean")
        model_cluster <- hclust(distMetrixMin, method = "average")
        membership <- cutree(model_cluster, h = thresh)
        numCluster  <- max(membership)

        # Generation of new instances
        numExs <- round(percOver*numMin/100)
        indexInstSelected <- sample(1:numMin, size = numExs, replace = TRUE, prob = selectProb)
        membershipSampled <- membership[indexInstSelected ]
        newExs <- matrix(nrow = numExs, ncol = numCol-1)
        numCumExs <- 0
        for (i in 1:numCluster)
        {
            numMincluster <- sum(membership == i)
            numMinclusterSampled <- sum(membershipSampled == i)

            if (numMincluster == 1 && numMinclusterSampled > 0)
            {
                indexRep  <- rep(1, numMinclusterSampled)
                InsMin  <- dataMin[membership == i, ,drop=FALSE]
                newIns  <- InsMin[indexRep, -numCol]
            }
            if (numMincluster > 1 && numMinclusterSampled > 0)
            {
                indexMinselected <- sample(which(membership == i), numMinclusterSampled, replace = TRUE)
                alfa <- runif(numMinclusterSampled)
                newIns  <- alfa*dataMin[indexInstSelected[membershipSampled == i], -numCol,drop = FALSE]+(1-alfa)*dataMin[indexMinselected, -numCol, drop = FALSE]
                for (j in nomatr)
                {
                    newIns[, j] <- dataMin[indexInstSelected[membershipSampled == i], j]
                    indexChange   <- runif(numMinclusterSampled) < 0.5
                    newIns[indexChange, j] <- dataMin[indexMinselected[indexChange], j]
                }
            }
            if (numMinclusterSampled > 0)
            {
                newExs[(numCumExs + 1):(numCumExs + numMinclusterSampled), ] <- newIns
                numCumExs <- numCumExs + numMinclusterSampled
            }
        }

        newExs <- data.frame(newExs, row.names = NULL)
        for(i in nomatr)
        {
            newExs[, i] <- factor(newExs[, i], levels=1:nlevels(data[, i]), labels = levels(data[, i]))
        }
        newExs[, numCol]  <- factor(rep(minCl, nrow(newExs)), levels=levels(data[, numCol]))
        colnames(newExs) <- colnames(data)

        if (tgt < numCol)
        {
            newExs <- newExs[, cols]
            data <- data[, cols]
        }
        newData <- rbind(data, newExs)
        return(newData)
    }


