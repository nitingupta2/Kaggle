library(tidyverse)
library(magrittr)
library(cvTools)
library(ggplot2)
library(data.table)
library(dplyr)
library(Matrix)
library(mxnet)
library(parallel)
library(readr)
library(xgboost)

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
ntest <- nrow(dfRawTest)
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

# Split combined data set into final training and testing sets
dfTrain <- dfCombined %>% dplyr::filter(!is.na(loss))
dfTest <- dfCombined %>% dplyr::filter(is.na(loss))
rm(dfCombined)


# Custom function to evaluate mean absolute error in mxnet
custom.metric.mae <- mx.metric.custom("mae", function(label, pred) {
    res <- Metrics::mae(exp(label),exp(pred))
    return(res)
})

# Tuned Parameters for mxnet
lParams_mxnet <- list(learning.rate = 1e-3,
                      momentum = 0.9,
                      batch.size = 100,
                      wd = 0,
                      num.round = 120)

# Run mxnet with tuned hyper-parameters and K-folds
model_mxnet <- function(dfTrain_fold, dfTest_fold) {
    
    x_train_fold <- data.matrix(t(dfTrain_fold %>% select(-loss, -id)))
    x_test_fold <- data.matrix(t(dfTest_fold %>% select(-loss, -id)))
    y_train_fold <- log(dfTrain_fold$loss + TARGET_SHIFT)
    x_test <- data.matrix(t(dfTest %>% select(-loss, -id)))
    
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
    
    mxnet_fit <- mx.model.FeedForward.create(outp, 
                                             X = x_train_fold, 
                                             y = as.array(y_train_fold),
                                             ctx = mx.cpu(),
                                             eval.data = NULL,
                                             eval.metric=custom.metric.mae,
                                             optimizer = "sgd",
                                             initializer = mx.init.uniform(0.01),
                                             array.layout = 'colmajor',
                                             learning.rate = lParams_mxnet$learning.rate,
                                             momentum = lParams_mxnet$momentum,
                                             wd = lParams_mxnet$wd,
                                             array.batch.size = lParams_mxnet$batch.size,
                                             num.round = lParams_mxnet$num.round)
    
    pred_oof <- predict(mxnet_fit, x_test_fold, array.layout = 'colmajor', 
                        array.batch.size = lParams_mxnet$batch.size, ctx = mx.cpu())
    
    pred_test <- predict(mxnet_fit, x_test, array.layout = 'colmajor', 
                        array.batch.size = lParams_mxnet$batch.size, ctx = mx.cpu())
    
    return(list(id_oof = dfTest_fold[["id"]], 
                pred_oof = (exp(pred_oof) - TARGET_SHIFT), 
                pred_test = (exp(pred_test) - TARGET_SHIFT)))
}

#########################################################################################################################
xgb_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= ModelMetrics::mae(exp(y),exp(yhat))
    return (list(metric = "mae", value = err))
}

xgb_eval_obj <- function(preds, dtrain) { 
    con <- 2 
    y <- getinfo(dtrain, "label") 
    x <- preds - y 
    grad <- con * x / (abs(x) + con) 
    hess <- con^2 / ((abs(x) + con)^2) 
    return(list(grad = grad, hess = hess)) 
}

lParams_xgboost <- list(eta = 0.01,
                        max_depth = 12,
                        min_child_weight = 1,
                        gamma = 2,
                        subsample = 0.8,
                        colsample_bytree = 0.5,
                        alpha = 1,
                        best_nrounds = 4857,
                        nthreads = 4)

# lParams_xgboost <- list(eta = 0.3,
#                         max_depth = 4,
#                         min_child_weight = 1,
#                         gamma = 0,
#                         subsample = 1,
#                         colsample_bytree = 1,
#                         alpha = 0,
#                         best_nrounds = 381,
#                         nthreads = 4)

model_xgboost <- function(dfTrain_fold, dfTest_fold, numFolds, seedForFolds) {
    
    y_train_fold <- log(dfTrain_fold$loss + TARGET_SHIFT)
    x_train_fold <- xgb.DMatrix(as.matrix(dfTrain_fold %>% select(-loss, -id)), label=y_train_fold)
    x_test_fold <- xgb.DMatrix(as.matrix(dfTest_fold %>% select(-loss, -id)))
    x_test <- xgb.DMatrix(as.matrix(dfTest %>% select(-loss, -id)))
    
    set.seed(seedForFolds)
    xgb_fit <- xgb.train(lParams_xgboost,
                         x_train_fold,
                         feval=xgb_eval_mae,
                         obj = xgb_eval_obj,
                         nrounds = as.integer(lParams_xgboost$best_nrounds * (numFolds/(numFolds-1))),
                         print_every_n = 50,
                         maximize = FALSE)
    
    pred_oof <- predict(xgb_fit, x_test_fold)
    pred_test <- predict(xgb_fit, x_test)
    
    return(list(id_oof = dfTest_fold[["id"]], 
                pred_oof = (exp(pred_oof) - TARGET_SHIFT), 
                pred_test = (exp(pred_test) - TARGET_SHIFT)))
}

#########################################################################################################################

# Create a list of models
vModels <- c("xgboost", "mxnet")

# Set number of rounds for out of fold predictions
numRoundsForFolds <- 1
numFolds <- 5

# Set primary seed for generating other seeds
set.seed(SEED)
# Generate seeds for creating folds
vSeeds <- sample(10000, numRoundsForFolds)

lPred_train <- list()
lPred_test <- list()

for(m in seq_along(vModels)) {
    modelName <- vModels[m]
    vTrain <- rep(0, ntrain) ; names(vTrain) <- dfTrain[["id"]]
    vTest <- rep(0, ntest) ; names(vTest) <- dfTest[["id"]]
    vTestIDs <- as.character(dfTest[["id"]])
    
    # Generate folds for different seeds
    for(seedForFolds in vSeeds) {
        # Generate folds for cross-validation
        set.seed(seedForFolds)
        lFolds <- caret::createFolds(dfTrain$loss, k = numFolds)
        print(lapply(lFolds, summary))
        print(lapply(lFolds, head, 15))
        
        for(i in 1:numFolds) {
            print(paste("Training", i, "of", numFolds, "folds for model", modelName, "using seed", seedForFolds))
            idx_test <- lFolds[[i]]
            dfTest_fold <- dfTrain[idx_test,]
            dfTrain_fold <- dfTrain[-idx_test,]
            
            print(paste("Training fold observations:", nrow(dfTrain_fold)))
            print(paste("Test fold observations:", nrow(dfTest_fold)))
            
            if(modelName == "mxnet") {
                lResults <- model_mxnet(dfTrain_fold, dfTest_fold)
            }
            else if(modelName == "xgboost") {
                lResults <- model_xgboost(dfTrain_fold, dfTest_fold, numFolds, seedForFolds)
            }
            vTrainIDs <- as.character(lResults$id_oof)
            vTrain[vTrainIDs] <- vTrain[vTrainIDs] + lResults$pred_oof
            vTest[vTestIDs] <- vTest[vTestIDs] + lResults$pred_test
        }
    }
    
    lPred_model <- list(train_loss = vTrain/numRoundsForFolds, test_loss = vTest/(numRoundsForFolds * numFolds))
    
    lPred_train[[m]] <- lPred_model$train_loss
    lPred_test[[m]] <- lPred_model$test_loss
}


dfPred_train <- do.call(cbind.data.frame, lPred_train)
dfPred_train <- dfPred_train %>% 
                set_colnames(paste("loss", vModels, sep = "_")) %>% 
                rownames_to_column(var = "id") %>% 
                mutate(loss = dfTrain$loss)
write.csv(dfPred_train, file = "ensemble_train.csv", row.names = F, quote = F)

dfPred_test <- as.data.frame(lPred_test)
dfPred_test <- dfPred_test %>% 
                set_colnames(paste("loss", vModels, sep = "_")) %>% 
                rownames_to_column(var = "id")
write.csv(dfPred_test, file = "ensemble_test.csv", row.names = F, quote = F)

#####################################################################################################################
################################################# ENSEMBLING ########################################################
#####################################################################################################################

dfPred_train <- read_csv(file = "ensemble_train_xgb_mxnet.csv")
dfPred_test <- read_csv(file = "ensemble_test_xgb_mxnet.csv")

print(paste("xgboost CV:", Metrics::mae(dfPred_train$loss, dfPred_train$loss_xgboost)))
print(paste("mxnet CV:", Metrics::mae(dfPred_train$loss, dfPred_train$loss_mxnet)))

y_train <- dfPred_train$loss
x_train <- xgb.DMatrix(as.matrix(dfPred_train %>% select(-id, -loss)), label=y_train)
x_test <- xgb.DMatrix(as.matrix(dfPred_test %>% select(-id)))

lParams_ensemble_xgb <- list(eta = 0.001,
                             max_depth = 4,
                             min_child_weight = 1,
                             subsample = 0.8,
                             colsample_bytree = 0.5,
                             objective = "reg:linear",
                             eval_metric = "mae",
                             nthreads = 4)

set.seed(SEED)
cv_res <- xgb.cv(lParams_ensemble_xgb,
                 x_train,
                 nrounds=15000,
                 nfold=4,
                 early_stopping_rounds=150,
                 print_every_n = 50,
                 verbose= 1,
                 maximize=FALSE)

best_nrounds <- cv_res$best_iteration
cv_mean <- cv_res$evaluation_log$test_mae_mean[best_nrounds]
cv_stdev <- cv_res$evaluation_log$test_mae_std[best_nrounds]

print(paste("Best_nrounds:", best_nrounds))
print(paste("CV_Mean:", cv_mean))
print(paste("CV_Stdev:", cv_stdev))

set.seed(SEED)
xgb_fit <- xgb.train(lParams_ensemble_xgb,
                     x_train,
                     nrounds = best_nrounds/0.75,
                     maximize = FALSE)

dfSubmission <- read_csv(file = SUBMISSION_FILE)
dfSubmission$loss <- predict(xgb_fit,x_test)
write.csv(dfSubmission, "submission_ensemble.csv", row.names = F, quote = F)
