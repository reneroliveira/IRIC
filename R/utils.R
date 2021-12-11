# IRIC - Utility Functions -------------------------

#' Numeralize
#' @description Convert dataset into numeric matrix.
#' @param data A two-dimensional dataset.
#' @param form Model formula.
#' @return (numerMatrix) Numerilized Matrix
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
#' @return (newIns) matrix of new instances
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
