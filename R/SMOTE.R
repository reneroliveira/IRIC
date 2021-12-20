# =====================================================
#  SMOTE sampling
# =====================================================

#' Synthetic Minority Oversampling Technique (SMOTE)
#' @description This function implements SMOTE sampling (Synthetic Minority Oversampling Technique).
#' @param x A data frame of the predictors from training data.
#' @param y A vector of response variable from training data.
#' @param percOver Percent of new instance generated for each minority instance.
#' @param  k Number of nearest neighbours.
#' @return
#' \item{newData}{A data frame of the oversampled data using SMOTE.}
#' @examples
#' library(caret)
#'
#' data(Korean)
#' sub <- createDataPartition(Korean$Churn,p=0.75,list=FALSE)
#' trainset <- Korean[sub,]
#' testset <- Korean[-sub,]
#' x <- trainset[, -11]
#' y <- trainset[, 11]
#' newData<- SMOTE(x, y)
#' @references Chawla, N., Bowyer, K., Hall, L. and Kegelmeyer, W. SMOTE: Synthetic minority oversampling technique. Journal of Artificial Intelligence Research, 2002, 16(3), pp. 321-357.
#' @export
SMOTE <-
    function(x, y, percOver = 1400, k = 5)
    {

        # find the class variable
        data <- data.frame(x,y)
        classTable   <- table(y)
        numCol       <- dim(data)[2]
        tgt <- length(data)

        # find the minority and majority instances
        minClass  <- names(which.min(classTable))
        indexMin  <- which(data[, tgt] == minClass)
        numMin    <- length(indexMin)
        majClass  <- names(which.max(classTable))
        indexMaj  <- which(data[, tgt] == majClass)
        numMaj    <- length(indexMaj)

        # move the class variable to the last column

        #if (tgt < numCol)
        #{
        #   cols <- 1:numCol
        #   cols[c(tgt, numCol)] <- cols[c(numCol, tgt)]
        #   data <- data[, cols]
        #}
        # generate synthetic minority instances
        # source("code/Data level/SmoteExs.R")
        if (percOver < 100)
        {
            indexMinSelect <- sample(1:numMin, round(numMin*percOver/100))
            dataMinSelect  <- data[indexMin[indexMinSelect], ]
            percOver <- 100
        } else {
            dataMinSelect <- data[indexMin, ]
        }

        newExs <- SmoteExs(dataMinSelect, percOver, k)

        # move the class variable back to original position
        #if (tgt < numCol)
        #{
        #   newExs <- newExs[, cols]
        #   data   <- data[, cols]
        #}

        # unsample for the majority intances
        newData <- rbind(data, newExs)

        return(newData)
    }

