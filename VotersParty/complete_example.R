train = read.csv("train2016.csv", na.strings = "", check.names=TRUE, stringsAsFactors = FALSE)
test = read.csv("test2016.csv", na.strings = "", check.names=TRUE, stringsAsFactors = FALSE)

imputeVars <- c('YOB','Party')
doNotImputeVars <- setdiff(names(train), imputeVars)

train[,doNotImputeVars][is.na(train[,doNotImputeVars])] <- "Missing"
test[,doNotImputeVars][is.na(test[,doNotImputeVars])] <- "Missing"


write.csv(train, "train_FillMissing.csv", row.names=FALSE)
write.csv(test, "test_FillMissing.csv", row.names=FALSE)


train = read.csv("train_FillMissing.csv")  ## converts chars to factors
test = read.csv("test_FillMissing.csv")

train$YOB <- as.numeric(as.character(train$YOB))
test$YOB <- as.numeric(as.character(test$YOB))

#### impute missing values -------------------

xNames <- setdiff(names(train), c("Party","USER_ID"))
trainImpute <- train[, xNames]
testImpute <- test[, xNames]

allImpute <- rbind(trainImpute, testImpute)

library(mice)

set.seed(999)
imputed = complete(mice(allImpute, m=5, MaxNWts=2000, pred=quickpred(allImpute)))

### bin the YOB variable -----------------

imputed$YOB_bin <- cut(imputed$YOB, breaks=quantile(imputed$YOB, probs=seq(0,1,0.2)), include.lowest=TRUE)
train$YOB_bin <- NA
test$YOB_bin <- NA

xNames <- c(names(imputed)[1:5], 'YOB_bin')

train[, xNames] <- imputed[1:nrow(trainImpute), xNames]
test[, xNames] <- imputed[((nrow(trainImpute)+1):nrow(allImpute)), xNames]



write.csv(train, "train_Imputed.csv", row.names=FALSE)
write.csv(test, "test_Imputed.csv", row.names=FALSE)


train <- read.csv("train_Imputed.csv")
test <- read.csv("test_Imputed.csv")

##### par processing -------------------

library(doParallel)
registerDoParallel(cores=4)



##### caret tuning -------------------

library(caret)

### hyperparms -------------------

cv.folds <- 5
cv.repeats <- 5
tuneLength.set <- 5

### set seed values -------------------
set.seed(321)
seeds <- vector(mode = "list", length = (cv.folds*cv.repeats +1))
for(i in 1:(cv.folds*cv.repeats)) seeds[[i]] <- sample.int(100000, tuneLength.set)
seeds[[cv.folds*cv.repeats +1]] <- 456                                 ### final model

### Use repeated CV -------------------
ctrl <- trainControl(method = "repeatedcv",
                     number = cv.folds,
                     repeats = cv.repeats,
                     seeds = seeds,
                     classProbs = TRUE,
                     allowParallel = TRUE)



##### models -------------------

set.seed(12345)
glm.mod <- train(form=Party ~ . - USER_ID, data=train, trControl=ctrl, method="glm", family="binomial")

set.seed(12345)
glmnet.mod <- train(form=Party ~ . - USER_ID, data=train, trControl=ctrl, method="glmnet", family="binomial", tuneLength=tuneLength.set)


##### out-of-sample accuracy  -------------------

cv.perf <- resamples(list(glm=glm.mod, glmnet=glmnet.mod))


summary(cv.perf)

dotplot(cv.perf, metric="Accuracy")
