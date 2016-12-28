# # Install xgboost 0.6 from github
# xgb_version <- packageVersion("xgboost")
# if(xgb_version<"0.6.0") source("install_xgboost.R")

library(caret)
library(ggplot2)
library(data.table)
library(dplyr)
library(Matrix)
library(Metrics)
library(xgboost)
library(parallel)

ID = 'id'
TARGET = 'loss'
SEED = 0
SHIFT = 200
TRAIN_FILE = "train_final.csv"
TEST_FILE = "test_final.csv"
SUBMISSION_FILE = "sample_submission.csv"

train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)
y_train = log(train[,TARGET, with = FALSE] + SHIFT)[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]
ntrain = nrow(train)

train_test = rbind(train, test)
rm(train) ; rm(test)

features = names(train_test)
features_ordinal <- c("cat74","cat78","cat79","cat85","cat87","cat90","cat101","cat102","cat103","cat105","cat111")

# Convert ordinal categorical features to integers
for(f in features_ordinal) {
    train_test[[f]] <- as.factor(train_test[[f]])
    vlevels <- levels(train_test[[f]])
    
    if("Other" %in% vlevels) {
        idx <- which(vlevels == "Other")
        vlevels <- c(vlevels[-idx],"Other")
    }
    train_test[[f]] <- factor(train_test[[f]], levels = vlevels)
    vlevels <- levels(train_test[[f]])
    
    print(paste("Ordered levels in", f, ":", paste(vlevels, collapse = ",")))
    
    # convert to integers and then lexical encoding
    train_test[[f]] <- as.integer(train_test[[f]]) - 1
    train_test[[f]] <- train_test[[f]]/max(train_test[[f]])
}

for (f in features) {
    if (class(train_test[[f]])=="character") {
        levels <- sort(unique(train_test[[f]]))
        train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels)) - 1
        train_test[[f]] <- train_test[[f]]/max(train_test[[f]])
    }
}

x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]
dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(as.matrix(x_test))

xgb_params = list(
    colsample_bytree = 0.5,
    subsample = 0.8,
    eta = 0.01,
    objective = 'reg:linear',
    max_depth = 12,
    alpha = 1,
    gamma = 2,
    min_child_weight = 1,
    base_score = 7.76
)
xg_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= mae(exp(y),exp(yhat) )
    return (list(metric = "error", value = err))
}

set.seed(SEED)
res = xgb.cv(xgb_params,
             dtrain,
             nrounds=15000,
             nfold=5,
             early_stopping_rounds=50,
             print_every_n = 50,
             verbose= 2,
             feval=xg_eval_mae,
             maximize=FALSE)

best_nrounds = res$best_iteration # for xgboost v0.6 users 
cv_mean = res$evaluation_log$test_error_mean[best_nrounds]
cv_std = res$evaluation_log$test_error_std[best_nrounds]
print(paste("CV-Mean:",cv_mean))
print(paste("CV-Stdev:",cv_std))
print(paste("Best_nrounds:",best_nrounds))

set.seed(SEED)
gbdt = xgb.train(xgb_params, dtrain, nrounds = as.integer(best_nrounds/0.8))

submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission$loss = exp(predict(gbdt,dtest)) - SHIFT
write.csv(submission,"submission_xgboost_3.csv",row.names = FALSE)
