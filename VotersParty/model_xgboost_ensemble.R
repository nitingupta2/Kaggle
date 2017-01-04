
library(Matrix)
library(magrittr)
library(tidyverse)
library(ggplot2)
library(xgboost)
library(foreach)
library(doParallel)
registerDoParallel(cores = 4)

preProcessFuncName <- "ensemble"

ID_VAR <- "USER_ID"
TARGET_VAR <- "Party"

if(preProcessFuncName == "ensemble") {
    lParams_xgboost <- list(eta = 0.02,
                            max_depth = 3,
                            min_child_weight = 10,
                            subsample = 0.6,
                            colsample_bytree = 0.7,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 6,
                            nthreads = 4
    )
} else if(preProcessFuncName == "preProcess1") {
    lParams_xgboost <- list(eta = 0.6,
                            max_depth = 1,
                            min_child_weight = 100,
                            subsample = 1,
                            colsample_bytree = 0.5,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 26,
                            nthreads = 4
    )
} else if(preProcessFuncName == "preProcess2") {
    lParams_xgboost <- list(eta = 0.1,
                            max_depth = 5,
                            min_child_weight = 100,
                            subsample = 0.7,
                            colsample_bytree = 0.9,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 133,
                            nthreads = 4
    )
} else if(preProcessFuncName == "preProcess3") {
    lParams_xgboost <- list(eta = 0.5,
                            max_depth = 3,
                            min_child_weight = 100,
                            subsample = 1.0,
                            colsample_bytree = 1.0,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 32,
                            nthreads = 4
    )
} else if(preProcessFuncName == "preProcess4") {
    lParams_xgboost <- list(eta = 0.1,
                            max_depth = 4,
                            min_child_weight = 100,
                            subsample = 0.7,
                            colsample_bytree = 1.0,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 173,
                            nthreads = 4
    )
} else if(preProcessFuncName == "preProcess5") {
    lParams_xgboost <- list(eta = 0.2,
                            max_depth = 4,
                            min_child_weight = 100,
                            subsample = 0.7,
                            colsample_bytree = 0.7,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 72,
                            nthreads = 4
    )
} else if(preProcessFuncName == "preProcess6") {
    lParams_xgboost <- list(eta = 0.03,
                            max_depth = 3,
                            min_child_weight = 100,
                            subsample = 0.9,
                            colsample_bytree = 0.5,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 357,
                            nthreads = 4
    )
} else if(preProcessFuncName == "preProcess7") {
    lParams_xgboost <- list(eta = 0.4,
                            max_depth = 1,
                            min_child_weight = 100,
                            subsample = 0.9,
                            colsample_bytree = 0.6,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 78,
                            nthreads = 4
    )
} else if(preProcessFuncName == "preProcess8") {
    lParams_xgboost <- list(eta = 0.4,
                            max_depth = 3,
                            min_child_weight = 100,
                            subsample = 1,
                            colsample_bytree = 1,
                            gamma = 1,
                            alpha = 1,
                            best_nrounds = 25,
                            nthreads = 4
    )
}


custom_eval_func <- function (yhat, x_train) {
    y <- getinfo(x_train, "label")
    y_pred <- as.integer(yhat > 0.5)
    
    # Accuracy = 1 - classification error
    acc <- 1 - ModelMetrics::ce(y, y_pred)
    return (list(metric = "accuracy", value = acc))
}

model_xgboost <- function(dfTrain_fold, dfTest_fold, dfTest, seedForFolds) {

    formula1 <- as.formula(paste(TARGET_VAR, ". -1", sep = " ~ "))    
    smm_train_fold <- sparse.model.matrix(formula1, data = dfTrain_fold %>% select(-starts_with(ID_VAR)))
    y_train_fold <- as.integer(as.factor(dfTrain_fold[[TARGET_VAR]])) - 1
    x_train_fold <- xgb.DMatrix(data = smm_train_fold, label = y_train_fold)
    
    smm_test_fold <- sparse.model.matrix(formula1, data = dfTest_fold %>% select(-starts_with(ID_VAR)))
    x_test_fold <- xgb.DMatrix(data = smm_test_fold)
    
    smm_test <- sparse.model.matrix(formula1, data = dfTest %>% select(-starts_with(ID_VAR)))
    x_test <- xgb.DMatrix(data = smm_test)
    
    set.seed(seedForFolds)
    xgb_fit <- xgb.train(lParams_xgboost,
                         x_train_fold,
                         feval=custom_eval_func,
                         objective = "binary:logistic",
                         nrounds = as.integer(lParams_xgboost$best_nrounds),
                         print_every_n = 50)
    
    pred_oof <- predict(xgb_fit, x_test_fold)
    pred_test <- predict(xgb_fit, x_test)
    
    return(list(id_oof = dfTest_fold[[ID_VAR]], 
                pred_oof = pred_oof, 
                pred_test = pred_test))
}

#########################################################################################################################

dfTrain <- read.table(file = paste0("train_", preProcessFuncName, ".tsv"), header = T, sep = "")
dfTest <- read.table(file = paste0("test_", preProcessFuncName, ".tsv"), header = T, sep = "")

ntrain <- nrow(dfTrain)
ntest <- nrow(dfTest)

SEED <- 2016

# Set number of rounds for out of fold predictions
numRoundsForFolds <- 5
numFolds <- 10

# Set primary seed for generating other seeds
set.seed(SEED)
# Generate seeds for creating folds
vSeeds <- sample(10000, numRoundsForFolds)

print("Parameters used:")
print(as.data.frame(lParams_xgboost))

# Generate folds for different seeds
lPred_model <- foreach(v = seq_along(vSeeds)) %dopar% {
    vTrain <- rep(0, ntrain) ; names(vTrain) <- dfTrain[[ID_VAR]]
    vTest <- rep(0, ntest) ; names(vTest) <- dfTest[[ID_VAR]]
    vTestIDs <- as.character(dfTest[[ID_VAR]])
    
    # Generate folds for cross-validation
    seedForFolds <- vSeeds[v]
    set.seed(seedForFolds)
    lFolds <- caret::createFolds(dfTrain[[TARGET_VAR]], k = numFolds)

    for(i in 1:numFolds) {
        library(Matrix)
        library(magrittr)
        library(tidyverse)
        library(xgboost)
        
        print(paste("Training", i, "of", numFolds, "folds using seed", seedForFolds))
        
        idx_test <- lFolds[[i]]
        dfTest_fold <- dfTrain[idx_test,]
        dfTrain_fold <- dfTrain[-idx_test,]
        
        print(paste("Training fold observations:", nrow(dfTrain_fold)))
        print(paste("Test fold observations:", nrow(dfTest_fold)))
        
        lResults <- model_xgboost(dfTrain_fold, dfTest_fold, dfTest, seedForFolds)
        
        vTrainIDs <- as.character(lResults$id_oof)
        vTrain[vTrainIDs] <- vTrain[vTrainIDs] + lResults$pred_oof
        vTest[vTestIDs] <- vTest[vTestIDs] + lResults$pred_test
    }
    
    list(train_prob = vTrain, test_prob = vTest/numFolds)
}

lPred_train <- lapply(lPred_model, function(x) x$train_prob)
lPred_test <- lapply(lPred_model, function(x) x$test_prob)

dfPred_train <- do.call(cbind.data.frame, lPred_train)
dfPred_train <- dfPred_train %>% 
                set_colnames(paste("xgboost", preProcessFuncName, vSeeds, sep = "_")) %>% 
                rownames_to_column(var = ID_VAR)

dfPred_test <- do.call(cbind.data.frame, lPred_test)
dfPred_test <- dfPred_test %>% 
                set_colnames(paste("xgboost", preProcessFuncName, vSeeds, sep = "_")) %>% 
                rownames_to_column(var = ID_VAR)

if(preProcessFuncName != "ensemble") {
    write.csv(dfPred_train, file = paste0("ensemble_xgboost_train_", preProcessFuncName, ".csv"), row.names = F, quote = F)
    write.csv(dfPred_test, file = paste0("ensemble_xgboost_test_", preProcessFuncName, ".csv"), row.names = F, quote = F)
}

#########################################################################################################################
# Average test predictions
#########################################################################################################################
dfPred_test$Party_prob <- rowMeans(dfPred_test[-1])
dfPred_test <- dfPred_test %>% 
                mutate(PREDICTIONS = ifelse(Party_prob <= 0.5, "Democrat", "Republican"))
write.csv(dfPred_test[c(ID_VAR, "PREDICTIONS")], 
          file = paste0("submission_xgboost_avg_", preProcessFuncName, ".csv"), row.names = F, quote = F)
