# Better yet, lets work with 200+100 architecture instead of 400+200+50, because all the general trends that work 
# for smaller network will most likely apply to the bigger one as well, and it will be faster to test many parameters 
# with a smaller network. With that starting architecture, try different optimizers one at a time, and decide which one 
# or two work the best. Next keep the optimizer fixed and test different initialization functions. 
# When you find the best initializer, fix the two previous parameters and test different activation functions. 
# Lastly, start with smaller dropouts between layers and increase them gradually until the network is converging 
# too fast because there is not enough data for training. Then go back to dropouts that were still allowing the network 
# to learn well without overfitting against validation data, and plug all the values back into your 
# starting 400+200+50 network and see if it does better. Note that doing more than 5 folds or more than 5 bags is 
# almost guaranteed to give you slight improvement even with identical parameters, but it will take a while to run that.

library(cvTools)
library(ggplot2)
library(data.table)
library(dplyr)
library(Matrix)
library(mxnet)
library(parallel)
library(readr)

ID <- 'id'
TARGET <- 'loss'
TARGET_SHIFT <- 200
SEED <- 2016

TRAIN_FILE <- "train_final.csv"
TEST_FILE <- "test_final.csv"
SUBMISSION_FILE <- "sample_submission.csv"


dfRawTrain <- read_csv(file = TRAIN_FILE)
dfRawTest <- read_csv(file = TEST_FILE)
dfRawTest$loss <- NA

ntrain <- nrow(dfRawTrain)
dfCombined <- rbind(dfRawTrain, dfRawTest)
rm(dfRawTrain) ; rm(dfRawTest)

vFeatures <- names(dfCombined)
vFeaturesOrdinal <- c("cat74","cat78","cat79","cat85","cat87","cat90","cat101","cat102","cat103","cat105","cat111")

# Convert ordinal categorical features to lexical encoding
for(f in vFeaturesOrdinal) {
    dfCombined[[f]] <- as.factor(dfCombined[[f]])
    vlevels <- levels(dfCombined[[f]])
    
    if("Other" %in% vlevels) {
        idx <- which(vlevels == "Other")
        vlevels <- c(vlevels[-idx],"Other")
    }
    dfCombined[[f]] <- factor(dfCombined[[f]], levels = vlevels)
    vlevels <- levels(dfCombined[[f]])
    
    print(paste("Ordered levels in", f, ":", paste(vlevels, collapse = ",")))
    
    # convert to integers and then lexical encoding
    dfCombined[[f]] <- as.integer(dfCombined[[f]]) - 1
    dfCombined[[f]] <- dfCombined[[f]]/max(dfCombined[[f]])
}

# convert other factors to lexical encoding
for (f in vFeatures) {
    if (class(dfCombined[[f]])=="character") {
        levels <- sort(unique(dfCombined[[f]]))
        # convert to integers and then lexical encoding
        dfCombined[[f]] <- as.integer(factor(dfCombined[[f]], levels=levels)) - 1
        dfCombined[[f]] <- dfCombined[[f]]/max(dfCombined[[f]])
    }
}

dfTrain <- dfCombined %>% dplyr::filter(!is.na(loss))
dfTest <- dfCombined %>% dplyr::filter(is.na(loss))
rm(dfCombined)

# Custom function to evaluate mean absolute error
custom.metric.mae <- mx.metric.custom("mae", function(label, pred) {
    res <- Metrics::mae(exp(label),exp(pred))
    return(res)
})

# Cross-validation function
crossValidate.mxnet <- function(dfTrain, dfTest, lParams) {
    inp <- mx.symbol.Variable('data')
    l1 <- mx.symbol.FullyConnected(inp, name = "l1", num.hidden = 200)
    a1 <- mx.symbol.Activation(l1, name = "a1", act_type = 'relu')
    d1 <- mx.symbol.Dropout(a1, name = 'd1', p = 0.2)
    l2 <- mx.symbol.FullyConnected(d1, name = "l2", num.hidden = 100)
    a2 <- mx.symbol.Activation(l2, name = "a2", act_type = 'relu')
    d2 <- mx.symbol.Dropout(a2, name = 'd2', p = 0.1)
    # l3 <- mx.symbol.FullyConnected(d2, name = "l3", num.hidden = 50)
    # a3 <- mx.symbol.Activation(l3, name = "a3", act_type = 'relu')
    # d3 <- mx.symbol.Dropout(a3, name = 'd3', p = 0.1)
    l4 <- mx.symbol.FullyConnected(d2, name = "l4", num.hidden = 1)
    outp <- mx.symbol.LinearRegressionOutput(l4, name = "outp")
    
    x_train <- data.matrix(t(dfTrain %>% select(-loss, -id)))
    x_test <- data.matrix(t(dfTest %>% select(-loss, -id)))
    y_train <- log(dfTrain$loss + TARGET_SHIFT)
    y_label <- log(dfTest$loss + TARGET_SHIFT)
    
    m <- mx.model.FeedForward.create(outp, 
                                     X = x_train, 
                                     y = as.array(y_train),
                                     ctx = mx.cpu(),
                                     eval.data = list(data = x_test, label = as.array(y_label)),
                                     eval.metric=custom.metric.mae,
                                     optimizer = "sgd",
                                     initializer = mx.init.uniform(0.01),
                                     array.layout = 'colmajor',
                                     learning.rate = lParams$learning.rate,
                                     momentum = lParams$momentum,
                                     wd = lParams$wd,
                                     array.batch.size = lParams$batch.size,
                                     num.round = lParams$num.round)
    
    pred_train <- predict(m, x_train, array.layout = 'colmajor', array.batch.size = lParams$batch.size, ctx = mx.cpu())
    pred_test <- predict(m, x_test, array.layout = 'colmajor', array.batch.size = lParams$batch.size, ctx = mx.cpu())
    
    MAE_train <- Metrics::mae(exp(y_train), exp(pred_train))
    MAE_test <- Metrics::mae(exp(y_label), exp(pred_test))
    print(paste("MAE Train:", MAE_train))
    print(paste("MAE Test:", MAE_test))
    
    return(list(model = m, mae_train = MAE_train, mae_test = MAE_test))
}


# Parameters for cross-validation
lParams <- list(learning.rate = 1e-3,
                momentum = 0.9,
                batch.size = 100,
                wd = 0,
                num.round = 100)

# Generate folds for cross-validation
set.seed(SEED)
numfolds <- 5
cv_folds <- cvFolds(ntrain, K = numfolds, type = "random")

cv_results <- list()
for(i in 1:numfolds) {
    print(paste("Tuning", i, "of", numfolds, "folds"))
    idx_test <- cv_folds$which == i
    dfTest_CV <- dfTrain[idx_test,]
    dfTrain_CV <- dfTrain[!idx_test,]
    
    cv_results[[i]] <- crossValidate.mxnet(dfTrain_CV, dfTest_CV, lParams)
}


dfCV <- data.frame()
for(i in 1:numfolds) {
    MAE_train <- cv_results[[i]][[2]]
    MAE_test <- cv_results[[i]][[3]]
    dfCV <- rbind(dfCV, data.frame(mae_train = MAE_train, mae_test = MAE_test))
}

# Write mean mae values to file
cvFileName <- "cv_mxnet.csv"
dfCV_Means <- t(data.frame(colMeans(dfCV)))
write.table(dfCV_Means, file = cvFileName, col.names = F, row.names = F, quote = F, append = T, sep = ",")

