
library(magrittr)
library(tidyverse)
library(ggplot2)

vModels <- c("xgboost", "kNN", "svm")

ID_VAR <- "USER_ID"
TARGET_VAR <- "Party"

getEnsembleFeatures <- function(fileType = c("train", "test")) {
    vFiles <- unlist(lapply(vModels, function(x) list.files(pattern = paste0("ensemble_", x, "_", fileType, "*"))))
    
    lFeatures <- lapply(vFiles, function(x) read.csv(x))
    dfFeatures <- reshape::merge_recurse(lFeatures, by.x = ID_VAR, by.y = ID_VAR)
    return(dfFeatures)
}

dfRawTrain <- read.csv("train2016.csv", na.strings =  c("", NA))

dfTrain <- getEnsembleFeatures("train")
dfTrain[[TARGET_VAR]] <- dfRawTrain[[TARGET_VAR]]

dfTest <- getEnsembleFeatures("test")
dfTest[[TARGET_VAR]] <- NA

preProcessFuncName <- "ensemble"
