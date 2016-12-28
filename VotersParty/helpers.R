# Caio 
###################################################################################
library(caret)
library(dismo)

set.seed(42)
k = kfold(dfTrain, 8)
accuracies = c()

for (i in 1:max(k)) {
    train = dfTrain[k != i,]
    test = dfTrain[k == i,]
    
    # train models and make predictions here
    
    accuracies = c(accuracies, confusionMatrix(predictions, test$Party)$overall[1])
}
mean(accuracies)

###################################################################################
# @Wynnie

crossValidate <- function(df, nfolds, modeler,alpha=0) {
    cv.acc <- vector(mode="numeric", length=nfolds)
    set.seed(113341)
    folds <- sample(rep(1:nfolds,length=nrow(df)))
    for(k in 1:nfolds) {
        pred <- modeler(train=df[folds!=k,],test=df[folds==k,],alpha=alpha)
        tab <- table(df[folds == k,]$Party,pred>0.5)
        cv.acc[k] <- sum(diag(tab))/sum(tab)
        print(paste0("Finished fold ",k,"/",nfolds))
    } 
    avgAcc <- mean(cv.acc)
    return (avgAcc)
}


rfModeler <- function(train,test,alpha=2601) {
    print(paste0("Running rfModeler"))
    rfMod6 <- randomForest(Party ~ . -USER_ID-submit, data=train, ntree=alpha)
    rfTestPred <- predict(rfMod6, newdata=test,type="prob")[,2]
    return (rfTestPred)
}


rfTestAccuracy1 <- crossValidate(df=train,nfolds=5,alpha=2601,modeler=rfModeler)


