library(foreach)
library(doParallel)
library(e1071)

source("preProcessing_VotersParty.R")

getAccuracy <- function(yactual, ypred) {
    yactual <- as.factor(as.character(yactual))
    ypred <- as.factor(as.character(ypred))
    return(1 - ModelMetrics::ce(yactual, ypred))
}

########################################################################################################################
# Parameter tuning by 10 fold CV
########################################################################################################################
def_cost <- 1
def_gamma <- 1

dfParams <- expand.grid(cost=c(1, 10, 100))
print(dfParams)

# Set number of rounds for out of fold predictions
numRoundsForFolds <- 1
numFolds <- 10

# Set primary seed for generating other seeds
SEED <- 2016
set.seed(SEED)
# Generate seeds for creating folds
vSeeds <- sample(10000, numRoundsForFolds)

# Cross validation file name
cvFileName <- paste0("cv_svm_", preProcessFuncName , ".csv")

for(p in 1:nrow(dfParams)) {
    print("Training Parameters used:")
    print(dfParams[p,])
    print(paste("Doing", numFolds, "fold CV using seeds:", paste(vSeeds, collapse = ",")))
    
    cost <- dfParams$cost[p]

    start_time <- Sys.time()
    
    # Generate folds for different seeds
    lPred_train <- list()
    for(v in seq_along(vSeeds)) {
        vTrain <- rep(NA_character_, nrow(dfTrain)) ; names(vTrain) <- dfTrain[[ID_VAR]]
        
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
            dfTest_fold <- dfTrain[idx_test,] %>% select(-starts_with(ID_VAR))
            dfTrain_fold <- dfTrain[-idx_test,] %>% select(-starts_with(ID_VAR))
            
            print(paste("Training fold observations:", nrow(dfTrain_fold)))
            print(paste("Test fold observations:", nrow(dfTest_fold)))
            
            formula1 <- as.formula(paste(TARGET_VAR, ".", sep = " ~ "))
            fit_svm <- svm(formula = formula1, data = dfTrain_fold, cost = cost)
            pred_fold <- predict(fit_svm, newdata = dfTest_fold)
            
            vTestIDs <- dfTrain[idx_test,]$USER_ID %>% as.character()
            vTrain <- replace(vTrain, which(names(vTrain) %in% vTestIDs), as.character(pred_fold))
        }
        
        lPred_train[[v]] <- vTrain
    }
    vAccuracy <- sapply(lPred_train, function(x) getAccuracy(dfTrain[[TARGET_VAR]], x))
    
    cv_mean <- mean(vAccuracy)
    cv_std <- sd(vAccuracy)
    
    end_time <- Sys.time()
    min_taken <- as.numeric(difftime(end_time, start_time, units = "mins"))
    
    dfCV <- data.frame(cv_mean = formatC(cv_mean, digits = 5, drop0trailing = F), 
                       cv_std = formatC(cv_std, digits = 5, drop0trailing = F), 
                       min_taken=formatC(min_taken, digits = 5, drop0trailing = F),
                       cost = cost)
    
    write.table(dfCV, file = cvFileName, col.names = (p == 1 & !file.exists(cvFileName)), 
                row.names = F, quote = F, append = T, sep = "\t")
}

# Show best CV values
dfCV <- read.table(file = cvFileName, header = T)
print(tail(dfCV %>% arrange(cv_mean)))

# Write sorted CV data back to file
write.table(dfCV %>% arrange(desc(cv_mean)), file = cvFileName, row.names = F, quote = F, sep = "\t")
