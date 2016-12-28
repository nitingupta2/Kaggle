# # Install xgboost 0.6 from github
# xgb_version <- packageVersion("xgboost")
# if(xgb_version<"0.6.0") source("install_xgboost.R")

if(!require(Metrics)) install.packages("Metrics")
if(!require(ModelMetrics)) install.packages("ModelMetrics")

library(caret)
library(ggplot2)
library(data.table)
library(dplyr)
library(Matrix)
library(Metrics)
library(xgboost)
library(parallel)

ID <- 'id'
TARGET <- 'loss'
TARGET_SHIFT <- 200
SEED <- 2016

TRAIN_FILE <- "train_final.csv"
TEST_FILE <- "test_final.csv"
SUBMISSION_FILE <- "sample_submission.csv"


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train = log(train[,TARGET, with = FALSE] + TARGET_SHIFT)[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

ntrain <- nrow(train)
train_test <- rbind(train, test)
rm(train) ; rm(test)

vFeatures <- names(train_test)
vFeaturesOrdinal <- c("cat74","cat78","cat79","cat85","cat87","cat90","cat101","cat102","cat103","cat105","cat111")

# Convert ordinal categorical features to integers
for(f in vFeaturesOrdinal) {
    train_test[[f]] <- as.factor(train_test[[f]])
    vlevels <- levels(train_test[[f]])
    
    if("Other" %in% vlevels) {
        idx <- which(vlevels == "Other")
        vlevels <- c(vlevels[-idx],"Other")
    }
    train_test[[f]] <- factor(train_test[[f]], levels = vlevels)
    vlevels <- levels(train_test[[f]])

    print(paste("Ordered levels in", f, ":", paste(vlevels, collapse = ",")))
    
    # convert to integers
    train_test[[f]] <- as.integer(train_test[[f]])
}

# convert other factors to integers
for (f in vFeatures) {
    if (class(train_test[[f]])=="character") {
        levels <- sort(unique(train_test[[f]]))
        train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
    }
}

x_train <- train_test[1:ntrain,]
x_test <- train_test[(ntrain+1):nrow(train_test),]

dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(as.matrix(x_test))


xg_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= ModelMetrics::mae(exp(y),exp(yhat))
    return (list(metric = "mae", value = err))
}

xg_eval_obj <- function(preds, dtrain) { 
    fair_constant <- 2 
    y <- getinfo(dtrain, "label") 
    x <- preds - y 
    grad <- fair_constant * x / (abs(x) + fair_constant) 
    hess <- fair_constant^2 / ((abs(x) + fair_constant)^2) 
    return(list(grad = grad, hess = hess)) 
}


def_eta <- 0.3
def_max_depth <- 6
def_min_child_weight <- 1
def_gamma <- 0
def_subsample <- 1
def_colsample_bytree <- 1
def_alpha <- 0

dfParams <- expand.grid(eta=0.03, max_depth=12, min_child_weight=100,
                        gamma=def_gamma, subsample=0.7, colsample_bytree=0.7,
                        alpha=def_alpha)

# dfParams <- expand.grid(eta=def_eta, max_depth=c(3,4,6,8,10,12), min_child_weight=def_min_child_weight,
#                         gamma=def_gamma, subsample=def_subsample, colsample_bytree=def_colsample_bytree,
#                         alpha=def_alpha)
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
                     dtrain,
                     nrounds=15000,
                     nfold=10,
                     early_stopping_rounds=ifelse(eta < 0.01,150,50),
                     print_every_n = 50,
                     verbose= 1,
                     feval=xg_eval_mae,
                     obj=xg_eval_obj,
                     maximize=FALSE)
    
    best_nrounds <- cv_res$best_iteration
    cv_mean <- cv_res$evaluation_log$test_mae_mean[best_nrounds]
    cv_std <- cv_res$evaluation_log$test_mae_std[best_nrounds]
    
    end_time <- Sys.time()
    
    dfCV <- data.frame(cv_mean=cv_mean, cv_std=cv_std, min_taken=as.numeric(difftime(end_time, start_time, units = "min")),
                       best_iteration=best_nrounds, eta=eta, max_depth=max_depth, min_child_weight=min_child_weight,
                       gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
                       alpha=alpha)
    
    cvFileName <- "cv_xgboost.csv"
    if(file.exists(cvFileName)) {
        dfCV_all <- read.csv(cvFileName, header = T)
        dfCV <- rbind(dfCV_all, dfCV)
    }
    write.csv(dfCV, file = cvFileName, row.names = F)
}

print(dfCV %>% arrange(desc(cv_mean)))

# set.seed(SEED)
# xgb_fit <- xgb.train(xgb_params, 
#                      dtrain, 
#                      feval=xg_eval_mae,
#                      obj = xg_eval_obj,
#                      nrounds = best_nrounds/0.75,
#                      maximize = FALSE)
# 
# submission <- fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
# submission$loss <- exp(predict(xgb_fit,dtest)) - TARGET_SHIFT
# write.csv(submission,"submission_xgboost.csv",row.names = FALSE)
