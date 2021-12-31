# IRIC - Utility Functions -------------------------

#' Numeralize
#' @description Convert dataset into numeric matrix.
#' @param data A two-dimensional dataset.
#' @param form Model formula.
#' @return
#' \item{numerMatrix}{Numerilized Matrix.}
#' @export
Numeralize <-
  function(data, form = NULL)
  {
    if (!is.null(form))
    {
      tgt    <- which(names(data) == as.character(form[[2]]))
      dataY <- data[drop = FALSE,, tgt]
      dataX <- data[, -tgt]
    } else {
      dataX <- data
    }
    numRow      <- dim(dataX)[1]
    #numCol      <- dim(dataX)[2]
    indexOrder      <- sapply(dataX, is.ordered)
    indexMultiValue <- sapply(dataX, nlevels)>2
    indexNominal    <- !indexOrder & indexMultiValue
    numerMatrixNames<- NULL
    if (all(indexNominal))
    {
      numerMatrix   <- NULL
    } else {
      numerMatrix      <- dataX[drop = FALSE, ,!indexNominal]
      numerMatrixNames <- colnames(numerMatrix)
      numerMatrix      <- data.matrix(numerMatrix)
      Min              <- apply(numerMatrix, 2, min)
      range            <- apply(numerMatrix, 2, max)-Min
      numerMatrix      <- scale(numerMatrix, Min, range)[, ]
    }

    if (any(indexNominal))
    {

      BiNames     <- NULL
      dataNominal <- dataX[drop = FALSE, ,indexNominal]
      numNominal  <- sum(indexNominal)
      if (numNominal>1)
      {
        dimEx <- sum(sapply(dataX[,indexNominal], nlevels))
      } else {
        dimEx <- nlevels(dataX[, indexNominal])
      }
      dataBinary  <- matrix(nrow = numRow, ncol = dimEx )
      cl <- 0
      for (i in 1:numNominal)
      {
        numCat <- nlevels(dataNominal[, i])
        for (j in 1:numCat)
        {
          value <- levels(dataNominal[, i])[j]
          ind  <- (dataNominal[,i] == value)
          dataBinary[, cl+1] <- as.integer(ind)
          BiNames[cl+1]   <- paste(names(dataNominal)[i], "_", value, sep="")
          cl <- cl+1
        }
      }
      numerMatrix  <- cbind(numerMatrix, dataBinary)
      colnames(numerMatrix) <- c(numerMatrixNames, BiNames)
    }

    if (!is.null(form))
    {
      numerMatrix <- data.frame(numerMatrix)
      numerMatrix <- cbind(numerMatrix, dataY)
    }
    return(numerMatrix)
  }

#' InsExs
#' @description Generate Synthetic instances from nearest neighborhood
#' @param instance Selected instance
#' @param dataknns nearest instance set
#' @param numExs number of new instances generated for each instance
#' @param nomAtt indicators of factor variables
#' @return
#' \item{newIns}{Matrix of new instances.}
#' @importFrom stats runif
#' @export
InsExs <-
  function(instance, dataknns, numExs, nomAtt)
  {
    numRow  <- dim(dataknns)[1]
    numCol  <- dim(dataknns)[2]
    newIns <- matrix (nrow = numExs, ncol = numCol)
    neig   <- sample(1:numRow, size = numExs, replace = TRUE)

    # generated  attribute values
    insRep  <- matrix(rep(instance, numExs), nrow = numExs, byrow = TRUE)
    diffs   <- dataknns[neig,] - insRep
    newIns  <- insRep + runif(1)*diffs
    # randomly change nominal attribute
    for (j in nomAtt)
    {
      newIns[, j]   <- dataknns[neig, j]
      indexChange   <- runif(numExs) < 0.5
      newIns[indexChange, j] <- insRep[indexChange, j]
    }
    return(newIns)
  }


#' SmoteExs
#' @description Obtain Smote instances for minority instances.
#' @param data Dataset of the minority instances.
#' @param percOver Percentage of oversampling.
#' @param k number of nearest neighours.
#' @importFrom RANN nn2
#' @return
#' \item{newExs}{Dataframe with new instances.}
#' @export
SmoteExs<-
  function(data, percOver, k)
  {
    # transform factors into integer
    nomAtt  <- c()
    numRow  <- dim(data)[1]
    numCol  <- dim(data)[2]
    dataX   <- data[ ,-numCol]
    dataTransformed <- matrix(nrow = numRow, ncol = numCol-1)
    for (col in 1:(numCol-1))
    {
      if (is.factor(data[, col]))
      {
        dataTransformed[, col] <- as.integer(data[, col])
        nomAtt <- c(nomAtt , col)
      } else {
        dataTransformed[, col] <- data[, col]
      }
    }
    numExs  <-  round(percOver/100) # this is the number of artificial instances generated
    newExs  <-  matrix(ncol = numCol-1, nrow = numRow*numExs)

    indexDiff <- sapply(dataX, function(x) length(unique(x)) > 1)
    # source("code/Data level/Numeralize.R")
    numerMatrix <- Numeralize(dataX[ ,indexDiff])
    # require("RANN")
    id_order <- nn2(numerMatrix, numerMatrix, k+1)$nn.idx
    for(i in 1:numRow)
    {
      kNNs   <- id_order[i, 2:(k+1)]
      newIns <- InsExs(dataTransformed[i, ], dataTransformed[kNNs, ], numExs, nomAtt)
      newExs[((i-1)*numExs+1):(i*numExs), ] <- newIns
    }

    # get factors as in the original data.
    newExs <- data.frame(newExs)
    for(i in nomAtt)
    {
      newExs[, i] <- factor(newExs[, i], levels = 1:nlevels(data[, i]), labels = levels(data[, i]))
    }
    newExs[, numCol] <- factor(rep(data[1, numCol], nrow(newExs)), levels=levels(data[, numCol]))
    colnames(newExs) <- colnames(data)
    return(newExs)
  }

FindLabel <-
  function(label){
    out <- list()
    classTable  <- table(label)
    classTable  <- sort(classTable, decreasing = TRUE)
    classLabels <- names(classTable)
    negLabel  <- classLabels[1]
    posLabel  <- classLabels[2]
    out$neg <-  negLabel
    out$pos <-  posLabel
    out
  }
