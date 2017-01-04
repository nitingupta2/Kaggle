library(xgboost)

preProcessFuncName <- "bestEnsemble"
commandArgs <- function() {return(preProcessFuncName)}

cvFileName <- paste0("cv_xgboost_", preProcessFuncName , ".csv")

source("preProcessing_VotersParty.R")

########################################################################################################################
# Parameter tuning by 10 fold CV
########################################################################################################################

formula1 <- as.formula(paste(TARGET_VAR, ". -1", sep = " ~ "))
smm_train <- sparse.model.matrix(formula1, data = dfTrain %>% select(-starts_with(ID_VAR)))
smm_test <- sparse.model.matrix(formula1, data = dfTest %>% select(-starts_with(ID_VAR)))
y_train <- as.integer(as.factor(dfTrain[[TARGET_VAR]])) - 1
x_train <- xgb.DMatrix(data = smm_train, label = y_train)

custom_eval_func <- function (yhat, x_train) {
    y <- getinfo(x_train, "label")
    y_pred <- as.integer(yhat > 0.5)
    
    # Accuracy = 1 - classification error
    acc <- 1 - ModelMetrics::ce(y, y_pred)
    return (list(metric = "accuracy", value = acc))
}

# Parameter tuning
def_eta <- 0.3
def_max_depth <- 1
def_min_child_weight <- 1
def_gamma <- 0
def_subsample <- 1
def_colsample_bytree <- 1
def_alpha <- 0

dfParams <- expand.grid(eta=0.11, max_depth=3, min_child_weight=10,
                        gamma=1, subsample=seq(0.6,1,0.1), colsample_bytree=seq(0.5,0.8,0.1), 
                        alpha=1)
print(dfParams)

for(i in 1:nrow(dfParams)) {
    print(paste("Tuning parameter combination", i, "of", nrow(dfParams)))
    
    eta = dfParams$eta[i]
    max_depth = dfParams$max_depth[i]
    min_child_weight = dfParams$min_child_weight[i]
    gamma = dfParams$gamma[i]
    subsample = dfParams$subsample[i]
    colsample_bytree = dfParams$colsample_bytree[i]
    alpha = dfParams$alpha[i]
    
    xgb_params = list(
        colsample_bytree = colsample_bytree,
        subsample = subsample,
        eta = eta,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        alpha = alpha,
        gamma = gamma,
        nthreads = 4
    )
    
    start_time <- Sys.time()
    
    set.seed(SEED)
    cv_res <- xgb.cv(xgb_params,
                     x_train,
                     nrounds=15000,
                     nfold=10,
                     early_stopping_rounds=150,
                     print_every_n = 50,
                     verbose= 1,
                     feval = custom_eval_func,
                     objective = "binary:logistic",
                     maximize=TRUE)
    
    best_nrounds <- cv_res$best_iteration
    cv_mean <- cv_res$evaluation_log$test_accuracy_mean[best_nrounds]
    cv_std <- cv_res$evaluation_log$test_accuracy_std[best_nrounds]
    
    end_time <- Sys.time()
    min_taken <- as.numeric(difftime(end_time, start_time, units = "mins"))
    
    dfCV <- data.frame(cv_mean = formatC(cv_mean, digits = 5, drop0trailing = F), 
                       cv_std = formatC(cv_std, digits = 5, drop0trailing = F), 
                       min_taken=formatC(min_taken, digits = 5, drop0trailing = F),
                       best_iteration=best_nrounds, eta=eta, max_depth=max_depth, min_child_weight=min_child_weight,
                       gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
                       alpha=alpha)
    
    write.table(dfCV, file = cvFileName, col.names = (i == 1 & !file.exists(cvFileName)), 
                row.names = F, quote = F, append = T, sep = "\t")
}

# Show best CV values
dfCV <- read.table(file = cvFileName, header = T)
print(tail(dfCV %>% arrange(cv_mean)))

# Write sorted CV data back to file
write.table(dfCV %>% arrange(desc(cv_mean)), file = cvFileName, row.names = F, quote = F, sep = "\t")
