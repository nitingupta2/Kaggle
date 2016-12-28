
library(readr)
library(ggplot2)
library(caret)
library(Matrix)
library(xgboost)

# read training and testing datasets
TRAIN <- read.csv("../input/train2016.csv")
TEST <- read.csv("../input/test2016.csv")

# separate multi-level, categorical response variable ("Party") from the remaining predictor variables in the training dataset ("TRAIN")
OUTCOME <- TRAIN$Party
TRAIN$Party <- NULL

# find and remove vectors that are linear combinations of other vectors
LINCOMB <- findLinearCombos(TRAIN)
TRAIN <- TRAIN[, -LINCOMB$remove]
TEST <- TEST[, -LINCOMB$remove]

# find and remove vectors with near-zero variance
NZV <- nearZeroVar(TRAIN, saveMetrics = TRUE)
TRAIN <- TRAIN[, -which(NZV[1:nrow(NZV),]$nzv == TRUE)]
TEST <- TEST[, -which(NZV[1:nrow(NZV),]$nzv == TRUE)]

# re-attach response variable ("OUTCOME") to training dataset ("TRAIN")
TRAIN$OUTCOME <- OUTCOME

# define xgb.train parameters
PARAM <- list(
    # General Parameters
    booster            = "gbtree",          # default
    silent             = 0,                 # default
    # Booster Parameters
    eta                = 0.05,              # default = 0.30
    gamma              = 0,                 # default
    max_depth          = 5,                 # default = 6
    min_child_weight   = 1,                 # default
    subsample          = 0.70,              # default = 1
    colsample_bytree   = 0.95,              # default = 1
    num_parallel_tree  = 1,                 # default
    lambda             = 0,                 # default
    lambda_bias        = 0,                 # default
    alpha              = 0,                 # default
    # Task Parameters
    objective          = "multi:softmax",   # default = "reg:linear"
    num_class          = 10,                # default = 0
    base_score         = 0.5,               # default
    eval_metric        = "merror"           # default = "rmes"
)

# convert TRAIN dataframe into a design matrix
TRAIN_SMM <- sparse.model.matrix(OUTCOME ~ ., data = TRAIN)
TRAIN_XGB <- xgb.DMatrix(data = TRAIN_SMM, Party = OUTCOME)

# set seed
set.seed(1)

# train xgb model
MODEL <- xgb.train(params      = PARAM, 
                   data        = TRAIN_XGB, 
                   nrounds     = 50, # change this to 400
                   verbose     = 2,
                   watchlist   = list(TRAIN_SMM = TRAIN_XGB)
)

# attach a predictions vector to the test dataset
TEST$OUTCOME <- 0

# use the trained xgb model ("MODEL") on the test data ("TEST") to predict the response variable ("OUTCOME")
TEST_SMM <- sparse.model.matrix(OUTCOME ~ ., data = TEST)
PRED <- predict(MODEL, TEST_SMM)
