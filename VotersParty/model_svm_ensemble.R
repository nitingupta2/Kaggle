
library(Matrix)
library(magrittr)
library(tidyverse)
library(e1071)
library(foreach)
library(doParallel)
registerDoParallel(cores = 4)

preProcessFuncName <- "preProcess8"

ID_VAR <- "USER_ID"
TARGET_VAR <- "Party"


if(preProcessFuncName == "preProcess8") {
    lParams_svm <- list(cost = 1, gamma = 0.01)
} else if(preProcessFuncName == "preProcess3") {
    lParams_svm <- list(cost = 1, gamma = 0.01)
}

model_svm <- function(dfTrain_fold, dfTest_fold, seedForFolds) {
    
    set.seed(seedForFolds)
    formula1 <- as.formula(paste(TARGET_VAR, ".", sep = " ~ "))
    fit_svm <- svm(formula = formula1, 
                   data = dfTrain_fold %>% select(-starts_with(ID_VAR)), 
                   cost = lParams_svm$cost,
                   gamma = lParams_svm$gamma)
    
    pred_oof <- predict(fit_svm, newdata = dfTest_fold %>% select(-starts_with(ID_VAR)))
    pred_test <- predict(fit_svm, newdata = dfTest %>% select(-starts_with(ID_VAR)))
    
    return(list(id_oof = dfTest_fold[[ID_VAR]], 
                pred_oof = as.integer(pred_oof) - 1, 
                pred_test = as.integer(pred_test) - 1))
}

#########################################################################################################################

dfTrain <- read.table(file = paste0("train_", preProcessFuncName, ".tsv"), header = T, sep = "")
dfTest <- read.table(file = paste0("test_", preProcessFuncName, ".tsv"), header = T, sep = "")

ntrain <- nrow(dfTrain)
ntest <- nrow(dfTest)

SEED <- 2016

# Set number of rounds for out of fold predictions
numRoundsForFolds <- 4
numFolds <- 10

# Set primary seed for generating other seeds
set.seed(SEED)
# Generate seeds for creating folds
vSeeds <- sample(10000, numRoundsForFolds)

print("Parameters used:")
print(as.data.frame(lParams_svm))

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
        library(magrittr)
        library(tidyverse)
        library(e1071)
        
        print(paste("Training", i, "of", numFolds, "folds using seed", seedForFolds))
        
        idx_test <- lFolds[[i]]
        dfTest_fold <- dfTrain[idx_test,]
        dfTrain_fold <- dfTrain[-idx_test,]
        
        print(paste("Training fold observations:", nrow(dfTrain_fold)))
        print(paste("Test fold observations:", nrow(dfTest_fold)))
        
        lResults <- model_svm(dfTrain_fold, dfTest_fold, seedForFolds)
        
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
    set_colnames(paste("svm", preProcessFuncName, vSeeds, sep = "_")) %>% 
    rownames_to_column(var = ID_VAR)
write.csv(dfPred_train, file = paste0("ensemble_svm_train_", preProcessFuncName, ".csv"), row.names = F, quote = F)

dfPred_test <- do.call(cbind.data.frame, lPred_test)
dfPred_test <- dfPred_test %>% 
    set_colnames(paste("svm", preProcessFuncName, vSeeds, sep = "_")) %>% 
    rownames_to_column(var = ID_VAR)
write.csv(dfPred_test, file = paste0("ensemble_svm_test_", preProcessFuncName, ".csv"), row.names = F, quote = F)

#########################################################################################################################
# Average test predictions
#########################################################################################################################
dfPred_test$Party_prob <- rowMeans(dfPred_test[-1])
dfPred_test <- dfPred_test %>% 
    mutate(PREDICTIONS = ifelse(Party_prob <= 0.5, "Democrat", "Republican"))
write.csv(dfPred_test[c(ID_VAR, "PREDICTIONS")], 
          file = paste0("submission_svm_avg_", preProcessFuncName, ".csv"), row.names = F, quote = F)
