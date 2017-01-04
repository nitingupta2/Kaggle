
library(Matrix)
library(magrittr)
library(tidyverse)
library(ggplot2)
library(RWeka)

args <- commandArgs()
preProcessFuncName <- as.character(args[1])

# Create a data frame of ensemble features
preProcessEnsemble <- function(dfMerged, bestEnsembleOnly = FALSE) {
    vModels <- c("xgboost", "kNN", "svm")
    
    vFiles <- unlist(lapply(vModels, function(x) list.files(pattern=paste0("ensemble_", x, "_train*"))))
    if(bestEnsembleOnly) {
        vFiles <- unlist(lapply(vModels, function(x) list.files(pattern=paste0("ensemble_", x, "_train_preProcess8.csv"))))
    }
    lFeatures <- lapply(vFiles, function(x) read.csv(x))
    dfTrain <- reshape::merge_recurse(lFeatures, by.x = ID_VAR, by.y = ID_VAR)
    dfTrain[[TARGET_VAR]] <- dfMerged[[TARGET_VAR]][!is.na(dfMerged[[TARGET_VAR]])]
    
    vFiles <- unlist(lapply(vModels, function(x) list.files(pattern=paste0("ensemble_", x, "_test*"))))
    if(bestEnsembleOnly) {
        vFiles <- unlist(lapply(vModels, function(x) list.files(pattern=paste0("ensemble_", x, "_test_preProcess8.csv"))))
    }
    lFeatures <- lapply(vFiles, function(x) read.csv(x))
    dfTest <- reshape::merge_recurse(lFeatures, by.x = ID_VAR, by.y = ID_VAR)
    dfTest[[TARGET_VAR]] <- NA
    
    lOutput <- list(dfMerged = rbind(dfTrain, dfTest))
    return(lOutput)
}
    
# Pre-process data - First way
########################################################################################################################
preProcess1 <- function(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues) {
    # Set median age in missing and outlier values
    medianYOB <- median(dfMerged$YOB, na.rm = T)
    dfMerged <- dfMerged %>% 
        mutate(YOB = ifelse(is.na(YOB) | YOB <= 1930 | YOB >= 2003, medianYOB, YOB))
    
    # Assign 'dnr' to unanswered survey questions and convert to factors
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.character)
    dfMerged[vFeaturesSurveyQues][is.na(dfMerged[vFeaturesSurveyQues])] <- "dnr"
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.factor)
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], relevel, "dnr")
    
    # Set the maximum factor level where NA
    for(feature in vFeaturesDemographic) {
        featureClass <- class(dfMerged[[feature]])
        if(featureClass == "character" | featureClass == "factor") {
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- names(which.max(table(dfMerged[[feature]])))
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
        }
    }
    
    lOutput <- list(dfMerged = dfMerged)
    return(lOutput)
}

# Pre-process data - Second way
########################################################################################################################
preProcess2 <- function(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues) {
    # Since many respondents have not provided demographic and survey questions, # of valid responses need to be recorded
    dfMerged$NumSurveyAns <- apply(dfMerged[vFeaturesSurveyQues], 1, function(Z) sum(!is.na(Z)))
    
    # Set median age in missing and outlier values
    medianYOB <- median(dfMerged$YOB, na.rm = T)
    dfMerged <- dfMerged %>% 
        mutate(YOB = ifelse(is.na(YOB) | YOB <= 1930 | YOB >= 2003, medianYOB, YOB))
    
    # Assign 'dnr' to unanswered survey questions and convert to factors
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.character)
    dfMerged[vFeaturesSurveyQues][is.na(dfMerged[vFeaturesSurveyQues])] <- "dnr"
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.factor)
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], relevel, "dnr")
    
    # Set 'dnr' to factor level where NA
    for(feature in vFeaturesDemographic) {
        featureClass <- class(dfMerged[[feature]])
        if(featureClass == "character" | featureClass == "factor") {
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- "dnr"
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
            dfMerged[[feature]] <- relevel(dfMerged[[feature]], "dnr")
        }
    }
    
    lOutput <- list(dfMerged = dfMerged)
    return(lOutput)
}

# Pre-process data - Third way
#########################################################################################################################
preProcess3 <- function(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues,
                        ID_VAR, TARGET_VAR, chisq_threshold_pvalue = 0.05) {
    # Since many respondents have not provided demographic and survey questions, # of valid responses need to be recorded
    dfMerged$NumSurveyAns <- apply(dfMerged[vFeaturesSurveyQues], 1, function(Z) sum(!is.na(Z)))
    
    # Set median age in missing and outlier values
    medianYOB <- median(dfMerged$YOB, na.rm = T)
    dfMerged <- dfMerged %>% 
        mutate(YOB = ifelse(is.na(YOB) | YOB <= 1930 | YOB >= 2003, medianYOB, YOB))
    
    # Assign 'dnr' to unanswered survey questions and convert to factors
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.character)
    dfMerged[vFeaturesSurveyQues][is.na(dfMerged[vFeaturesSurveyQues])] <- "dnr"
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.factor)
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], relevel, "dnr")
    
    # Set 'dnr' to factor level where NA
    for(feature in vFeaturesDemographic) {
        featureClass <- class(dfMerged[[feature]])
        if(featureClass == "character" | featureClass == "factor") {
            dfMerged[[feature]] <- as.character(dfMerged[[feature]])
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- "dnr"
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
            if("dnr" %in% levels(dfMerged[[feature]])) {
                dfMerged[[feature]] <- relevel(dfMerged[[feature]], "dnr")
            }
        }
    }
    
    # With chisq goodness of fit determine survey questions whose probability distribution 
    # is very different from Party distribution in the data set
    dfXsq <- data.frame()
    indexTarget <- which(names(dfMerged) == TARGET_VAR)
    for(i in seq_along(vFeaturesSurveyQues)) {
        indexQ <- which(names(dfMerged) == vFeaturesSurveyQues[i])
        tbl <- table(unlist(dfMerged[,indexTarget]), unlist(dfMerged[,indexQ]))
        Xsq <- chisq.test(tbl)
        dfXsq <- rbind(dfXsq, data.frame(surveyQues = vFeaturesSurveyQues[i], pvalue = Xsq$p.value))
    }

    dfXsq <- dfXsq %>% dplyr::filter(pvalue < chisq_threshold_pvalue) %>% arrange(pvalue)
    vFeaturesSurveyInterest <- as.character(dfXsq$surveyQues)
    
    # reduce survey questions in the final data set
    dfMerged <- dfMerged[c(ID_VAR, TARGET_VAR, "NumSurveyAns", vFeaturesDemographic, vFeaturesSurveyInterest)]
    
    lOutput <- list(dfMerged = dfMerged)
    return(lOutput)
}

# Pre-process data - Fourth & Fifth way
#########################################################################################################################
preProcess4 <- function(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues,
                        ID_VAR, TARGET_VAR, chisq_threshold_pvalue = 0.05) {
    # Since many respondents have not provided demographic and survey questions, # of valid responses need to be recorded
    dfMerged$NumSurveyAns <- apply(dfMerged[vFeaturesSurveyQues], 1, function(Z) sum(!is.na(Z)))
    
    # Set median age in missing and outlier values
    medianYOB <- median(dfMerged$YOB, na.rm = T)
    dfMerged <- dfMerged %>% 
        mutate(YOB = ifelse(is.na(YOB) | YOB <= 1930 | YOB >= 2003, medianYOB, YOB)) %>% 
        mutate(Age = 2016 - YOB) %>% 
        mutate(Age_group = cut(Age, c(0, 18, 44, 64, 90)))
    
    # Assign 'dnr' to unanswered survey questions and convert to factors
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.character)
    dfMerged[vFeaturesSurveyQues][is.na(dfMerged[vFeaturesSurveyQues])] <- "dnr"
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.factor)
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], relevel, "dnr")
    
    # Set 'dnr' to factor level where NA
    for(feature in vFeaturesDemographic) {
        featureClass <- class(dfMerged[[feature]])
        if(featureClass == "character" | featureClass == "factor") {
            dfMerged[[feature]] <- as.character(dfMerged[[feature]])
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- "dnr"
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
            if("dnr" %in% levels(dfMerged[[feature]])) {
                dfMerged[[feature]] <- relevel(dfMerged[[feature]], "dnr")
            }
        }
    }
    
    # With chisq goodness of fit determine survey questions whose probability distribution 
    # is very different from Party distribution in the data set
    dfXsq <- data.frame()
    indexTarget <- which(names(dfMerged) == TARGET_VAR)
    for(i in seq_along(vFeaturesSurveyQues)) {
        indexQ <- which(names(dfMerged) == vFeaturesSurveyQues[i])
        tbl <- table(unlist(dfMerged[,indexTarget]), unlist(dfMerged[,indexQ]))
        Xsq <- chisq.test(tbl)
        dfXsq <- rbind(dfXsq, data.frame(surveyQues = vFeaturesSurveyQues[i], pvalue = Xsq$p.value))
    }
    dfXsq <- dfXsq %>% dplyr::filter(pvalue < chisq_threshold_pvalue) %>% arrange(pvalue)
    vFeaturesSurveyInterest <- as.character(dfXsq$surveyQues)
    
    # reduce survey questions in the final data set
    vFeaturesAdditional <- c(ID_VAR, TARGET_VAR, "Age_group", "NumSurveyAns")
    dfMerged <- dfMerged[c(vFeaturesAdditional, vFeaturesDemographic, vFeaturesSurveyInterest)]
    dfMerged <- dfMerged %>% select(-YOB)
    
    lOutput <- list(dfMerged = dfMerged)
    return(lOutput)
}

# Pre-process data - Sixth & Seventh way
#########################################################################################################################
preProcess6 <- function(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues,
                        ID_VAR, TARGET_VAR, chisq_threshold_pvalue = 0.05) {
    # Since many respondents have not provided demographic and survey questions, # of valid responses need to be recorded
    dfMerged$NumSurveyAns <- apply(dfMerged[vFeaturesSurveyQues], 1, function(Z) sum(!is.na(Z)))
    
    # Set median age in missing and outlier values
    medianYOB <- median(dfMerged$YOB, na.rm = T)
    dfMerged <- dfMerged %>% 
        mutate(YOB = ifelse(is.na(YOB) | YOB <= 1930 | YOB >= 2003, medianYOB, YOB)) %>% 
        mutate(Age = 2016 - YOB) %>% 
        mutate(Age_group = cut(Age, c(0, 18, 44, 64, 90)))
    
    # With chisq goodness of fit determine survey questions whose probability distribution 
    # is very different from Party distribution in the data set
    dfXsq <- data.frame()
    for(surveyQues in vFeaturesSurveyQues) {
        tbl <- table(unlist(dfMerged[[TARGET_VAR]]), unlist(dfMerged[[surveyQues]]))
        Xsq <- chisq.test(tbl)
        dfXsq <- rbind(dfXsq, data.frame(surveyQues = surveyQues, pvalue = Xsq$p.value))
    }
    dfXsq <- dfXsq %>% dplyr::filter(pvalue < chisq_threshold_pvalue) %>% arrange(pvalue)
    vFeaturesSurveyInterest <- as.character(dfXsq$surveyQues)
    
    # Valid responses to interesting questions
    dfMerged$NumSurveyAnsInterest <- apply(dfMerged[vFeaturesSurveyInterest], 1, function(Z) sum(!is.na(Z)))
    
    # Assign 'dnr' to unanswered survey questions and convert to factors
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.character)
    dfMerged[vFeaturesSurveyQues][is.na(dfMerged[vFeaturesSurveyQues])] <- "dnr"
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.factor)
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], relevel, "dnr")
    
    # Set 'dnr' to factor level where NA
    for(feature in vFeaturesDemographic) {
        featureClass <- class(dfMerged[[feature]])
        if(featureClass == "character" | featureClass == "factor") {
            dfMerged[[feature]] <- as.character(dfMerged[[feature]])
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- "dnr"
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
            if("dnr" %in% levels(dfMerged[[feature]])) {
                dfMerged[[feature]] <- relevel(dfMerged[[feature]], "dnr")
            }
        }
    }
    
    # Create a feature 'IsLiberal' from the answers to the survey questions of interest
    dfMerged <- dfMerged %>% 
        mutate(IsLiberal = case_when(.$Q109244 == "Yes" ~ "Yes",
                                     .$Q115611 == "Yes" ~ "No",
                                     .$Q113181 == "Yes" ~ "No",
                                     .$Q113181 == "No" & .$Q115611 == "No" & .$Q109244 == "No" ~ "Yes")) %>% 
        mutate(IsLiberal = ifelse(is.na(IsLiberal), "unknown", IsLiberal))
    
    # reduce survey questions in the final data set
    dfMerged <- dfMerged[c(ID_VAR, TARGET_VAR, "Age_group", "NumSurveyAnsInterest", "IsLiberal",
                           vFeaturesDemographic, vFeaturesSurveyInterest)]
    dfMerged <- dfMerged %>% select(-YOB)
    
    lOutput <- list(dfMerged = dfMerged)
    return(lOutput)
}

# Pre-process data - Eighth way
########################################################################################################################
preProcess8 <- function(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues,
                        ID_VAR, TARGET_VAR, chisq_threshold_pvalue = 0.05) {
    
    # Since many respondents have not provided demographic and survey questions, # of valid responses need to be recorded
    dfMerged$NumSurveyAns <- apply(dfMerged[vFeaturesSurveyQues], 1, function(Z) sum(!is.na(Z)))
    dfMerged$NumDemogrAns <- apply(dfMerged[vFeaturesDemographic], 1, function(Z) sum(!is.na(Z)))
    
    # Set missing EducationLevel to 'dropout'
    dfMerged <- dfMerged %>% 
                mutate(EducationLevel = as.character(EducationLevel),
                       EducationLevel = ifelse(is.na(EducationLevel), "dropout", EducationLevel),
                       EducationLevel = ordered(EducationLevel, levels = c("dropout",
                                                                           "Current K-12", 
                                                                           "Current Undergraduate",
                                                                           "High School Diploma",
                                                                           "Associate's Degree",
                                                                           "Bachelor's Degree",
                                                                           "Master's Degree",
                                                                           "Doctoral Degree")))

    # Determine median YOB by EducationLevel    
    medianYOB <- tapply(dfMerged$YOB, dfMerged$EducationLevel, median, na.rm = T)

    dfMerged <- dfMerged %>% 
                mutate(YOB = ifelse((is.na(YOB) | YOB < 1930 | YOB > 2000) & EducationLevel == "dropout", 
                                    as.integer(medianYOB["dropout"]), YOB)) %>% 
                mutate(YOB = ifelse((is.na(YOB) | YOB < 1930) & EducationLevel == "Current K-12", 
                                    as.integer(medianYOB["Current K-12"]), YOB)) %>% 
                mutate(YOB = ifelse((is.na(YOB) | YOB < 1930) & EducationLevel == "Current Undergraduate",
                                    as.integer(medianYOB["Current Undergraduate"]), YOB)) %>% 
                mutate(YOB = ifelse((is.na(YOB) | YOB < 1930) & EducationLevel == "High School Diploma",
                                    as.integer(medianYOB["High School Diploma"]), YOB)) %>% 
                mutate(YOB = ifelse((is.na(YOB) | YOB < 1930 | YOB > 2000) & EducationLevel == "Associate's Degree", 
                                    as.integer(medianYOB["Associate's Degree"]), YOB)) %>% 
                mutate(YOB = ifelse((is.na(YOB) | YOB < 1930 | YOB > 2000) & EducationLevel == "Bachelor's Degree", 
                                    as.integer(medianYOB["Bachelor's Degree"]), YOB)) %>% 
                mutate(YOB = ifelse((is.na(YOB) | YOB < 1930 | YOB > 2000) & EducationLevel == "Master's Degree", 
                                    as.integer(medianYOB["Master's Degree"]), YOB)) %>% 
                mutate(YOB = ifelse((is.na(YOB) | YOB < 1930 | YOB > 2000) & EducationLevel == "Doctoral Degree", 
                                    as.integer(medianYOB["Doctoral Degree"]), YOB))
    
    # Determine median YOB by Income
    medianYOB <- tapply(dfMerged$YOB, dfMerged$Income, median, na.rm = T)
    for(inRange in levels(dfMerged$Income)) {
        dfMerged <- dfMerged %>% 
                    mutate(YOB = ifelse((is.na(YOB)) & Income == inRange, 
                                        as.integer(medianYOB[inRange]), YOB))
    }
    
    # Determine median YOB by HouseholdStatus
    medianYOB <- tapply(dfMerged$YOB, dfMerged$HouseholdStatus, median, na.rm = T)
    for(hhStatus in levels(dfMerged$HouseholdStatus)) {
        dfMerged <- dfMerged %>% 
                    mutate(YOB = ifelse((is.na(YOB)) & HouseholdStatus == hhStatus, 
                                        as.integer(medianYOB[hhStatus]), YOB))
    }
    
    # Determine median YOB by Gender
    medianYOB <- tapply(dfMerged$YOB, dfMerged$Gender, median, na.rm = T)
    for(sexLevel in levels(dfMerged$Gender)) {
        dfMerged <- dfMerged %>% 
                    mutate(YOB = ifelse((is.na(YOB)) & Gender == sexLevel, 
                                        as.integer(medianYOB[sexLevel]), YOB))
    }
    
    # Set Age, Age_group and reorder factors
    dfMerged <- dfMerged %>% 
                mutate(Age = 2013 - YOB,
                       Age_group = cut(Age, breaks = c(0, 18, 34, 50, 64, 90))) %>% 
                mutate(Income = ordered(Income, levels = c("under $25,000", "$25,001 - $50,000", 
                                                           "$50,000 - $74,999", "$75,000 - $100,000", 
                                                           "$100,001 - $150,000", "over $150,000"))) 
    
    # Impute missing HouseholdStatus by EducationLevel & Age
    for(edlevel in levels(dfMerged$EducationLevel)) {
        vSub <- which(dfMerged$EducationLevel == edlevel)
        TargetVar <- "HouseholdStatus"
        vPredictors <- c("Age")
        lImpute <- imputeMissingValues(dfMerged[vSub,], TargetVar, vPredictors)
        print(paste("Accuracy in determining", TargetVar, "by:", edlevel, "+",
                    paste(vPredictors, collapse = " + "), "=", lImpute$Accuracy))
        dfMerged[vSub,] <- lImpute$dfSub
    }
    
    # Impute missing Income by EducationLevel, HouseholdStatus, Gender & Age
    for(edlevel in levels(dfMerged$EducationLevel)) {
        vSub <- which(dfMerged$EducationLevel == edlevel)
        TargetVar <- "Income"
        vPredictors <- c("HouseholdStatus", "Gender", "Age")
        lImpute <- imputeMissingValues(dfMerged[vSub,], TargetVar, vPredictors)
        print(paste("Accuracy in determining", TargetVar, "by:", edlevel, "+",
                    paste(vPredictors, collapse = " + "), "=", lImpute$Accuracy))
        dfMerged[vSub,] <- lImpute$dfSub
    }
    
    # Impute other missing Income by EducationLevel, HouseholdStatus, Age
    for(edlevel in levels(dfMerged$EducationLevel)) {
        vSub <- which(dfMerged$EducationLevel == edlevel)
        TargetVar <- "Income"
        vPredictors <- c("HouseholdStatus", "Age")
        lImpute <- imputeMissingValues(dfMerged[vSub,], TargetVar, vPredictors)
        print(paste("Accuracy in determining", TargetVar, "by:", edlevel, "+",
                    paste(vPredictors, collapse = " + "), "=", lImpute$Accuracy))
        dfMerged[vSub,] <- lImpute$dfSub
    }
    
    # Set NA values in Gender to 'Male'
    dfMerged <- dfMerged %>% 
                mutate(Gender = as.character(Gender),
                       Gender = ifelse(is.na(Gender), "Male", Gender),
                       Gender = as.factor(Gender))
    
    # With chisq goodness of fit determine survey questions whose probability distribution 
    # is very different from Party distribution in the data set
    dfXsq <- data.frame()
    for(surveyQues in vFeaturesSurveyQues) {
        tbl <- table(unlist(dfMerged[[TARGET_VAR]]), unlist(dfMerged[[surveyQues]]))
        Xsq <- chisq.test(tbl)
        dfXsq <- rbind(dfXsq, data.frame(surveyQues = surveyQues, pvalue = Xsq$p.value))
    }
    dfXsq <- dfXsq %>% dplyr::filter(pvalue < chisq_threshold_pvalue) %>% arrange(pvalue)
    vFeaturesSurveyInterest <- as.character(dfXsq$surveyQues)
    
    # Valid responses to interesting questions
    dfMerged$NumSurveyAnsInterest <- apply(dfMerged[vFeaturesSurveyInterest], 1, function(Z) sum(!is.na(Z)))
    
    # Assign 'dnr' to unanswered survey questions and convert to factors
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.character)
    dfMerged[vFeaturesSurveyQues][is.na(dfMerged[vFeaturesSurveyQues])] <- "dnr"
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], as.factor)
    dfMerged[vFeaturesSurveyQues] <- lapply(dfMerged[vFeaturesSurveyQues], relevel, "dnr")

    # Create a feature 'IsLiberal' from the answers to the survey questions of interest
    dfMerged <- dfMerged %>% 
        mutate(IsLiberal = case_when(.$Q109244 == "Yes" ~ "Yes",
                                     .$Q115611 == "Yes" ~ "No",
                                     .$Q113181 == "Yes" ~ "No",
                                     .$Q113181 == "No" & .$Q115611 == "No" & .$Q109244 == "No" ~ "Yes")) %>% 
        mutate(IsLiberal = ifelse(is.na(IsLiberal), "unknown", IsLiberal),
               IsLiberal = as.factor(IsLiberal),
               IsLiberal = relevel(IsLiberal, "unknown"))
    
    # reduce survey questions in the final data set
    vFeaturesAdditional <- c("IsLiberal", "Age", "NumSurveyAns", "NumDemogrAns")

    vFeaturesDemographic <- c(vFeaturesDemographic[!vFeaturesDemographic %in% "YOB"], vFeaturesAdditional)
    dfMerged <- dfMerged[c(ID_VAR, TARGET_VAR, vFeaturesDemographic, vFeaturesSurveyInterest)]
    
    lOutput <- list(dfMerged = dfMerged, vFeaturesDemographic = vFeaturesDemographic)
    return(lOutput)
}


# Impute missing values by kNN using RWeka
########################################################################################################################
imputeMissingValues <- function(dfSub, TargetVar, vPredictors) {
    cc <- complete.cases(dfSub[vPredictors])
    dfModel <- dfSub[cc,]
    if(sum(is.na(dfModel[[TargetVar]])) > 0) {
        formula_knn <- paste(TargetVar, paste(vPredictors, collapse = " + "), sep = " ~ ")
        classifier <- IBk(formula = formula_knn, data = dfModel)
        pred_Weka <- predict(classifier, newdata = dfModel)
        
        confMatrix <- caret::confusionMatrix(dfModel[[TargetVar]], pred_Weka)
        vMissing <- which(is.na(dfModel[[TargetVar]]))
        dfModel[[TargetVar]][vMissing] <- pred_Weka[vMissing]
        dfSub[cc,] <- dfModel
    }   
    
    return(list(dfSub = dfSub, Accuracy = as.numeric(confMatrix$overall["Accuracy"])))
}


########################################################################################################################
# Process raw data
########################################################################################################################
SEED <- 2016
ID_VAR <- "USER_ID"
TARGET_VAR <- "Party"

dfQuestions <- read_delim("questions.tsv", delim = "\t") %>% set_colnames(c("ID", "QnA"))
dfQuestions$ID <- paste0("Q", dfQuestions$ID)

dfRawTrain <- read.csv("train2016.csv", na.strings =  c("", NA))
dfRawTest <- read.csv("test2016.csv", na.strings =  c("", NA))
dfRawTest[[TARGET_VAR]] <- NA

names(dfRawTrain)
vFeaturesDemographic <- c("YOB", "Gender", "Income", "HouseholdStatus", "EducationLevel")
vFeaturesSurveyQues <- names(dfRawTrain)[!(names(dfRawTrain) %in% vFeaturesDemographic)]
# exclude USER_ID from survey questions
vFeaturesSurveyQues <- vFeaturesSurveyQues[!(vFeaturesSurveyQues %in% c(ID_VAR, TARGET_VAR))]

########################################################################################################################
# Select Preprocessing function
########################################################################################################################

# Combine raw training and test sets
dfMerged <- rbind(dfRawTrain, dfRawTest)
rm(dfRawTrain) ; rm(dfRawTest)

if(preProcessFuncName == "ensemble") {
    lOutput <- preProcessEnsemble(dfMerged)
} else if(preProcessFuncName == "bestEnsemble") {
    lOutput <- preProcessEnsemble(dfMerged, TRUE)
} else if(preProcessFuncName == "preProcess1") {
    lOutput <- preProcess1(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues)
} else if(preProcessFuncName == "preProcess2") {
    lOutput <- preProcess2(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues)
} else if(preProcessFuncName == "preProcess3") {
    lOutput <- preProcess3(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues, ID_VAR, TARGET_VAR)
} else if(preProcessFuncName == "preProcess4") {
    lOutput <- preProcess4(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues, ID_VAR, TARGET_VAR)
} else if(preProcessFuncName == "preProcess5") {
    lOutput <- preProcess4(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues, ID_VAR, TARGET_VAR, 
                           chisq_threshold_pvalue = 0.001)
} else if(preProcessFuncName == "preProcess6") {
    lOutput <- preProcess6(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues, ID_VAR, TARGET_VAR, 
                           chisq_threshold_pvalue = 0.001)
} else if(preProcessFuncName == "preProcess7") {
    lOutput <- preProcess6(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues, ID_VAR, TARGET_VAR, 
                           chisq_threshold_pvalue = 0.05)
} else if(preProcessFuncName == "preProcess8") {
    lOutput <- preProcess8(dfMerged, vFeaturesDemographic, vFeaturesSurveyQues, ID_VAR, TARGET_VAR, 
                           chisq_threshold_pvalue = 0.05)
}

dfMerged <- lOutput$dfMerged

# Split merged data into final training and test sets
dfTrain <- dfMerged %>% subset(!is.na(get(TARGET_VAR)))
dfTest <- dfMerged %>% subset(is.na(get(TARGET_VAR))) 
dfTest[[TARGET_VAR]] <- "dummy"

# Write preprocessed data to file
write.table(dfTrain, file = paste0("train_", preProcessFuncName, ".tsv"), row.names = F, sep = "\t")
write.table(dfTest, file = paste0("test_", preProcessFuncName, ".tsv"), row.names = F, sep = "\t")
