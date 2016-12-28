# This script is loosely ported from Python script by modkzs, Misfyre and others
# https://www.kaggle.com/misfyre/allstate-claims-severity/encoding-feature-comb-modkzs-1108-72665
# However it gets a little better results maybe due to
# different realisations of scaling and Box Cox transformations in R and Python
#
# Please run on your local to get high scores

# ----------------------------------- load libraries ------------------------------
#data
library(data.table)
#statistics
library(Metrics)
library(scales)
library(Hmisc)
library(forecast)
#data science
library(caret)
library(xgboost)
library(e1071)
#strings
library(stringr)

# shift applying to the dependent variable ('loss')

SHIFT <- 200

# ----------------------------------- tools & variables ----------------------------

# fair objective 2 for XGBoost

amo.fairobj2 <- function(preds, dtrain) {
    
    labels <- getinfo(dtrain, "label")
    con <- 2
    x <- preds - labels
    grad <- con * x / (abs(x) + con)
    hess <- con ^ 2 / (abs(x) + con) ^ 2
    
    return(list(grad = grad, hess = hess))
    
}

# MAE Metric for XGBoost

amm_mae <- function(preds, dtrain) {
    labels <- xgboost::getinfo(dtrain, "label")
    elab <- as.numeric(labels)
    epreds <- as.numeric(preds)
    err <- mae(exp(elab), exp(epreds))
    
    return(list(metric = "amm_mae", value = round(err,4)))
}

# names of categorical features for feature engineering

dtCat <- c("cat80","cat87","cat57","cat12","cat79","cat10","cat7","cat89","cat2","cat72",
           "cat81","cat11","cat1","cat13","cat9","cat3","cat16","cat90","cat23","cat36",
           "cat73","cat103","cat40","cat28","cat111","cat6","cat76","cat50","cat5",
           "cat4","cat14","cat38","cat24","cat82","cat25")

dtCat <- merge(data.frame(f1 = dtCat), data.frame(f2 = dtCat))
dtCat <- data.table(dtCat)
dtCat <- dtCat[as.integer(str_extract_all(f1, "[0-9]+")) >
                               as.integer(str_extract_all(f2, "[0-9]+"))]

dtCat[, f1 := as.character(f1)]
dtCat[, f2 := as.character(f2)]


#-------------------------------------- load data -----------------------------------------

# load data
dtTrain <- fread("train.csv", showProgress = TRUE)
dtTest <- fread("test.csv", showProgress = TRUE)

# merge to single data set
dtMerged <- rbind(dtTrain, dtTest, fill=TRUE)
remove(dtTrain)
remove(dtTest)

#-------------------------------------- feature engineering -------------------------------

# new categorical features
# a bit straightforward approach so you can try to improve it

for (f in 1:nrow(dtCat)) {
    
    f1 <- dtCat[f, f1]
    f2 <- dtCat[f, f2]
    
    vrb <- paste(f1, f2, sep = "_") # removed as.name
    dtMerged[, eval(vrb) := paste0(dtMerged[[f1]], dtMerged[[f2]])] # simplified with double brackets
}

# categorical features to range ones
# was very slow - must be much faster and much R-approach now

utf.A <- utf8ToInt("A")

for (f in colnames(dtMerged)[colnames(dtMerged) %like% "^cat"]) {
    
    dtMerged[, eval(f) := mapply(function(x, id) { 
        
        if (id == 1) print(f)
        
        x <- utf8ToInt(x)
        ln <- length(x)
        
        x <- (x - utf.A + 1) * 26 ^ (ln - 1:ln - 1)
        x <- sum(x)
        x
        
    }, eval(f), .I)]
    
}

# remove skewness

for (f in colnames(dtMerged)[colnames(dtMerged) %like% "^cont"]) {
    
    tst <- e1071::skewness(dtMerged[[f]])
    if (tst > .25) {
        if (is.na(dtMerged[, BoxCoxTrans(dtMerged[[f]])$lambda])) next
        dtMerged[, eval(f) := BoxCox(dtMerged[[f]], BoxCoxTrans(dtMerged[[f]])$lambda)]
    }
}

# scale

for (f in colnames(dtMerged)[colnames(dtMerged) %like% "^cont"]) {
    dtMerged[, eval(f) := scale(dtMerged[[f]])]
}

# save

save(dtMerged, file="dtMerged.pipe.Rda")



#-------------------------------------- convert to matrices -------------------------------

# data table to matrix
# you can do it different ways
# sometimes it is not neccessary though...

dtMerged.m <- model.matrix(object=~ ., data=model.frame(formula=~ ., data=dtMerged[, !c("id", "loss"), with=FALSE]
                                                   , na.action="na.pass"))
dtMerged.m <- as.matrix(dtMerged.m)

dtTrain.label <- dtMerged[!is.na(loss), loss]
dtTest.key <- dtMerged[is.na(loss), id]
dtTrain.key <- dtMerged[!is.na(loss), id]

dtTrain.m<-dtMerged.m[which(dtMerged[,loss] %in% dtTrain.label),]
dtTest.m<-dtMerged.m[which(!(dtMerged[,loss] %in% dtTrain.label)),]

# remove garbage

remove(dtMerged, dtMerged.m)

# sure we can (and should) save labels and keys as well
# however in this script I save only new data matrices

save(dtTrain.m, file="dtTrain.m.pipe.Rda")
save(dtTest.m, file="dtTest.m.pipe.Rda")



#------------------------------------------ prepare model ---------------------------------

# additional variables

n_folds <- 5
early_stopping <- 50
print.every <- 10

preds <- list()

# split data

set.seed(1)
numFolds <- createFolds(1:nrow(dtTrain.m), k = n_folds)

# parameters

xgb.params <- list(booster = "gbtree"
                   , objective = amo.fairobj2
                   , subsample = 0.7
                   , max_depth = 12
                   , colsample_bytree = 0.7
                   , eta = 0.03
                   , min_child_weight = 1)


# training function

xgb.train.am <- function (ds.x, ds.ev.x, ds.label, ev.label, ds.dtTest.x, params, it = 0, e.stop = 50, print.ev = 100) {
    
    dtTrain.m.xgb <- xgb.DMatrix(ds.x, label=ds.label, missing=NA)
    dtTrain.ev.m.xgb <- xgb.DMatrix(ds.ev.x, label=ev.label, missing=NA)
    dtTest.m.xgb <- xgb.DMatrix(ds.dtTest.x, missing=NA)
    
    print(paste("[", it, "] training xgboost begin ",sep=""," : ",Sys.time()))
    set.seed(1)
    xgb <- xgb.train(params = params
                     , data = dtTrain.m.xgb
                     , nrounds = 10000
                     , verbose = 1
                     , print_every_n = print.ev
                     , feval = amm_mae
                     , watchlist = list(train = dtTrain.m.xgb, eval = dtTrain.ev.m.xgb)
                     , early_stop_round = e.stop
                     , maximize = FALSE)
    
    pred_tst <- predict(xgb, dtTrain.ev.m.xgb)
    
    print(paste("[", it, "] training xgboost complete with score: ", 
                mae(exp(ev.label), exp(pred_tst)), 
                sep="", " : ", 
                Sys.time()))
    
    pred <- predict(xgb, dtTest.m.xgb)
    return(pred)
}



#-------------------------------------------- run and get 1107.23181 ----------------------------------

for (i in 1:n_folds) {
    
    preds[[i]] <- xgb.train.am(ds.x = dtTrain.m[-numFolds[i][[1]], ]
                               , ds.ev.x = dtTrain.m[numFolds[i][[1]], ]
                               , ds.label = log(dtTrain.label[-numFolds[i][[1]]] + SHIFT)
                               , ev.label = log(dtTrain.label[numFolds[i][[1]]] + SHIFT)
                               , ds.dtTest.x = dtTest.m
                               , params = xgb.params
                               , it = i
                               , print.ev = print.every
                               , e.stop = early_stopping)
    
}

# average

preds.t <- as.data.table(preds)
preds.t[, loss := rowMeans(.SD)]

# return to normal condition and write

xgb.sub <- data.table(id = dtTest.key, loss = exp(preds.t[, loss]) - SHIFT)
write.csv(xgb.sub, paste("submission_xgboost_",as.character(Sys.Date()),"_1.csv",sep="")
          , row.names=FALSE, quote=FALSE)
