
library(Matrix)
library(magrittr)
library(tidyverse)
library(ggplot2)
library(xgboost)

SEED <- 2016
ID_VAR <- "USER_ID"
TARGET_VAR <- "Party"

dfQuestions <- read_delim("questions.tsv", delim = "\t") %>% set_colnames(c("ID", "QnA"))
dfQuestions$ID <- paste0("Q", dfQuestions$ID)

dfRawTrain <- read_csv("train2016.csv", na = c("", NA))
dfRawTest <- read_csv("test2016.csv", na = c("", NA))
dfRawTest[[TARGET_VAR]] <- NA

names(dfRawTrain)
vFeaturesDemographic <- c("YOB", "Gender", "Income", "HouseholdStatus", "EducationLevel")
vFeaturesSurveyQues <- names(dfRawTrain)[!(names(dfRawTrain) %in% vFeaturesDemographic)]
# exclude USER_ID from survey questions
vFeaturesSurveyQues <- vFeaturesSurveyQues[!(vFeaturesSurveyQues %in% c(ID_VAR, TARGET_VAR))]

# Pre-process data - First way
########################################################################################################################
preProcess1 <- function(dfMerged) {
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
        if(class(dfMerged[[feature]]) == "character") {
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- names(which.max(table(dfMerged[[feature]])))
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
        }
    }
    
    return(dfMerged)
}

# Pre-process data - Second way
########################################################################################################################
preProcess2 <- function(dfMerged) {
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
        if(class(dfMerged[[feature]]) == "character") {
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- "dnr"
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
            dfMerged[[feature]] <- relevel(dfMerged[[feature]], "dnr")
        }
    }
    
    return(dfMerged)
}

# Pre-process data - Third way
#########################################################################################################################
preProcess3 <- function(dfMerged) {
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
        if(class(dfMerged[[feature]]) == "character") {
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- "dnr"
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
            dfMerged[[feature]] <- relevel(dfMerged[[feature]], "dnr")
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
    chisq_threshold_pvalue <- 0.05
    dfXsq <- dfXsq %>% dplyr::filter(pvalue < chisq_threshold_pvalue) %>% arrange(pvalue)
    vFeaturesSurveyInterest <- as.character(dfXsq$surveyQues)
    
    # reduce survey questions in the final data set
    return(dfMerged[c(ID_VAR, TARGET_VAR, "NumSurveyAns", vFeaturesDemographic, vFeaturesSurveyInterest)])
}

# Pre-process data - Fourth & Fifth way
#########################################################################################################################
preProcess4 <- function(dfMerged, chisq_threshold_pvalue = 0.05) {
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
        if(class(dfMerged[[feature]]) == "character") {
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- "dnr"
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
            dfMerged[[feature]] <- relevel(dfMerged[[feature]], "dnr")
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
    dfMerged <- dfMerged[c(ID_VAR, TARGET_VAR, "Age_group", "NumSurveyAns", vFeaturesDemographic, vFeaturesSurveyInterest)]
    dfMerged <- dfMerged %>% select(-YOB)
    return(dfMerged)
}

# Pre-process data - Sixth & Seventh way
#########################################################################################################################
preProcess6 <- function(dfMerged, chisq_threshold_pvalue = 0.05) {
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
        if(class(dfMerged[[feature]]) == "character") {
            dfMerged[[feature]][is.na(dfMerged[[feature]])] <- "dnr"
            dfMerged[[feature]] <- as.factor(dfMerged[[feature]])
            dfMerged[[feature]] <- relevel(dfMerged[[feature]], "dnr")
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
    return(dfMerged)
}

########################################################################################################################
# Select Preprocessing function
########################################################################################################################

# Combine raw training and test sets
dfMerged <- rbind(dfRawTrain, dfRawTest)
summary(dfMerged)

preProcessFuncName <- "preProcess7"

if(preProcessFuncName == "preProcess1") {
    dfMerged <- preProcess1(dfMerged)
} else if(preProcessFuncName == "preProcess2") {
    dfMerged <- preProcess2(dfMerged)
} else if(preProcessFuncName == "preProcess3") {
    dfMerged <- preProcess3(dfMerged)
} else if(preProcessFuncName == "preProcess4") {
    dfMerged <- preProcess4(dfMerged)
} else if(preProcessFuncName == "preProcess5") {
    dfMerged <- preProcess4(dfMerged, chisq_threshold_pvalue = 0.001)
} else if(preProcessFuncName == "preProcess6") {
    dfMerged <- preProcess6(dfMerged, chisq_threshold_pvalue = 0.001)
} else if(preProcessFuncName == "preProcess7") {
    dfMerged <- preProcess6(dfMerged, chisq_threshold_pvalue = 0.05)
}

names(dfMerged)
table(dfMerged$IsLiberal, dfMerged$Party)

# Split merged data into final training and test sets
dfTrain <- dfMerged %>% subset(!is.na(get(TARGET_VAR)))
dfTest <- dfMerged %>% subset(is.na(get(TARGET_VAR))) 
dfTest[[TARGET_VAR]] <- "dummy"

# Write preprocessed data to file
write.table(dfTrain, file = paste0("train_", preProcessFuncName, ".tsv"), row.names = F, sep = "\t")
write.table(dfTest, file = paste0("test_", preProcessFuncName, ".tsv"), row.names = F, sep = "\t")

########################################################################################################################
# Parameter tuning by 10 fold CV
########################################################################################################################

cvFileName <- paste0("cv_xgboost_", preProcessFuncName , ".csv")

smm_train <- sparse.model.matrix(Party ~ . -1, data = dfTrain %>% select(-starts_with(ID_VAR)))
smm_test <- sparse.model.matrix(Party ~ . -1, data = dfTest %>% select(-starts_with(ID_VAR)))
y_train <- as.integer(as.factor(dfTrain[[TARGET_VAR]])) - 1
x_train <- xgb.DMatrix(data = smm_train, label = y_train)

custom_eval_func <- function (yhat, x_train) {
    y = getinfo(x_train, "label")
    y_pred = as.integer(yhat > 0.5)
    err= ModelMetrics::auc(y, y_pred)
    return (list(metric = "auc", value = err))
}

# Parameter tuning
def_eta <- 0.3
def_max_depth <- 1
def_min_child_weight <- 1
def_gamma <- 0
def_subsample <- 1
def_colsample_bytree <- 1
def_alpha <- 0

dfParams <- expand.grid(eta=seq(0.3,0.5,0.1), max_depth=c(1:3), min_child_weight=100,
                        gamma=1, subsample=seq(0.6,1,0.1), colsample_bytree=seq(0.5,0.9,0.1), 
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
    cv_mean <- cv_res$evaluation_log$test_auc_mean[best_nrounds]
    cv_std <- cv_res$evaluation_log$test_auc_std[best_nrounds]
    
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
