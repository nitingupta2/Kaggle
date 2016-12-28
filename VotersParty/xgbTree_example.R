# Analytics Edge Kaggle Competition Spring 2016
# caret xgbTree v2

# 0.65374 on Private Leaderboard nrounds = 150, max_depth = 2, eta = 0.1, colsample_bytree = 0.7
# 0.65661 on Private Leaderboard nrounds = 50, max_depth = 2, eta = 0.1, colsample_bytree = 0.7

set.seed(1000)
rm(list=ls(all=T))

library(caret)
library(randomForest)
#library(doParallel)
library(plyr)
library(dplyr)

#cl<-makeCluster(2)
#registerDoParallel(cl)
strt<-Sys.time()

train <- read.csv("train2016.csv", na.strings = c("","NA"),stringsAsFactors = FALSE)
test <- read.csv("test2016.csv", na.strings = c("","NA"),stringsAsFactors = FALSE)

train$USER_ID <- NULL

# impute using median or most common value
train$YOB[is.na(train$YOB)] <- 1983
test$YOB[is.na(test$YOB)] <- 1983
train$Gender[is.na(train$Gender)] <- as.character("Female")
test$Gender[is.na(test$Gender)] <- as.character("Female")
train$Income[is.na(train$Income)] <- as.character("$75,000 - $100,000")
test$Income[is.na(test$Income)] <- as.character("$75,000 - $100,000")
train$HouseholdStatus[is.na(train$HouseholdStatus)] <- as.character("Single (no kids)")
test$HouseholdStatus[is.na(test$HouseholdStatus)] <- as.character("Single (no kids)")
train$EducationLevel[is.na(train$EducationLevel)] <- as.character("Bachelor's Degree")
test$EducationLevel[is.na(test$EducationLevel)] <- as.character("Bachelor's Degree")

# Remove outliers
train <- train[train$YOB >= 1939 & train$YOB <= 2000,]

# Set missing to "not_provided"
train[is.na(train)] <- as.character('not_provided')
test[is.na(test)] <- as.character('not_provided')

fitControl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary, method="cv", number=10)

xgbGrid <-  expand.grid(max_depth = c(1, 2, 3),
                        nrounds = (1:5)*50,
                        eta = c(0.1,0.2,0.3),
                        gamma = 0,
                        colsample_bytree = 0.7,
                        min_child_weight = 1,
                        subsample = 1)

modelFit=train(factor(Party)~.,
               data = train,
               method = 'xgbTree',
               metric = 'ROC',
               trControl = fitControl,
               tuneGrid = xgbGrid)


modelFit
plot(modelFit)
varImp(modelFit)
plot(varImp(modelFit),25)


predictions <- predict(modelFit, newdata = test)
head(predictions,25)

MySubmission = data.frame(USER_ID = test$USER_ID, Predictions = predictions)
write.csv(MySubmission, "caret_xgbTree_v2.csv", row.names=FALSE)

summary(predictions)
confusionMatrix(modelFit)

print(Sys.time()-strt)
#stopCluster(cl)