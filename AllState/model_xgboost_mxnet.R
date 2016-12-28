library(data.table)
library(xgboost)
library(Matrix)
library(MASS)
library(mxnet)

# define columns for categorical variables
take_columns <- function(file, test_file) {
    f1 <- fread(file)
    f2 <- fread(test_file)
    
    data <- rbind(f1, f2, fill = TRUE)
    data <- data[,grep('(cat|id)', colnames(data)), with = FALSE]
    
    data <- melt(data, id.vars = 'id') # 2D -> 1D
    data <- unique(data[,.(variable, value)]) # take unique columns
    data[,variable_value := paste(variable, value, sep = "_")] #name columns like cat2_A
    setorder(data, variable, value)
    data[,n_var := .N, by = variable] # check number of values for each variable
    data[n_var > 2,column := 1:.N] # those with number of values will be binary coded
    data[,n_var := NULL]
    data[,lin_val := (0:(.N-1))/(.N-1), by = variable] # all variables will be also lex coded (A - 0, B - 0.5, C - 1)
    
    return(data)
}

# read data
load_data <- function(file, columns) {
    data <- fread(file)
    
    # split variables between categorical and numerical
    cn <- colnames(data)
    c_cat <- c("id", cn[grep("cat", cn)]) 
    c_num <- c(cn[-grep("cat", cn)])
    
    cat <- data[,c_cat, with = F]
    num <- data[,c_num, with = F]
    
    cat <- melt(cat, id.vars = "id", measure.vars = c_cat[-1]) # 2D -> 1D
    
    rows <- cat[, .(id = unique(id))] #remember row numers for all id's
    rows[, row := 1:.N]
    
    cat <- columns[cat,,on = c("variable", "value")]
    
    ### assign lex coding values
    lin_cat <- dcast(cat[,.(id, variable, lin_val)], id ~ variable, value.var = 'lin_val', fill = 0)
    lin_cat <- lin_cat[rows[,.(id)],,on = 'id']
    lin_cat <- Matrix(as.matrix(lin_cat[,-'id',with = FALSE]), sparse = TRUE)
    ###
    
    ### assign binary coding
    cat <- cat[!is.na(column), ]
    cat <- rows[cat,,on = "id"]
    
    ### sparse matrix
    cat_mat <- sparseMatrix(i = cat[,row], j = cat[,column], x = 1)
    colnames(cat_mat) <- columns[!is.na(column),variable_value]
    
    num <- Matrix(as.matrix(num[,-'id',with=FALSE]), sparse = TRUE)
    
    ### bind all variables
    data <- cBind(num, cat_mat, lin_cat)
    print("Data loaded")
    return(list(data = data, rows = rows, columns = columns))
}

get_train_sample <- function(frac = 0.8, seed = 123) {
    set.seed(seed)
    which(runif(nrow(data)) < frac)
}

generate_folds = function(n_fold = 5, seed = 123, train_obs = 1:nrow(data)) {
    set.seed(seed)
    rnd <- sample(1:n_fold, length(train_obs), TRUE)
    lapply(1:n_fold, function(i) which(rnd == i))
}

# box-cox
transform <- function(data, lambda) {
    (data ^ lambda - 1) / lambda
}

# reverse box-cox
detransform <- function(data, lambda) {
    if (is.null(lambda))
        return(data)
    (lambda * data + 1) ^ (1 / lambda)
}

# for xgb
MAE <- function(pred, dtrain) {
    pred <- detransform(pred, lambda)
    real <- detransform(getinfo(dtrain, "label"), lambda)
    return(list(metric = "mae", value = mean(abs(pred - real))))
}

# train one xgb model
xgb_model <- function(train_obs, test_obs, params) {
    train <- xgb.DMatrix(data[train_obs,-y], label = transform(data[train_obs,y], lambda))
    test <- xgb.DMatrix(data[test_obs,-y], label = transform(data[test_obs,y], lambda))
    
    mod <- xgb.train(data = train,
                     watchlist = list(test = test),
                     params = params,
                     nrounds = params$nrounds,
                     early.stop.round = 50,
                     print.every.n = 100,
                     feval = MAE,
                     maximize = FALSE
    )
    
    pred <- detransform(predict(mod, test), lambda)
    pred_train <- detransform(predict(mod, train), lambda)
    
    MAE_test <- mean(abs(pred - data[test_obs, y]))
    MAE_train <- mean(abs(pred_train - data[train_obs, y]))
    
    return(list(MAE_test = MAE_test, MAE_train = MAE_train, model = mod))
}

#train one net
nn_model <- function(train_obs, test_obs, params) {
    inp <- mx.symbol.Variable('data')
    l1 <- mx.symbol.FullyConnected(inp, name = "l1", num.hidden = 400)
    a1 <- mx.symbol.Activation(l1, name = "a1", act_type = 'relu')
    d1 <- mx.symbol.Dropout(a1, name = 'd1', p = 0.4)
    l2 <- mx.symbol.FullyConnected(d1, name = "l2", num.hidden = 200)
    a2 <- mx.symbol.Activation(l2, name = "a2", act_type = 'relu')
    d2 <- mx.symbol.Dropout(a2, name = 'd2', p = 0.2)
    l3 <- mx.symbol.FullyConnected(d2, name = "l3", num.hidden = 1)
    outp <- mx.symbol.MAERegressionOutput(l3, name = "outp")
    
    m <- mx.model.FeedForward.create(outp, 
                                     X = as.array(t(data[train_obs, -y])), 
                                     y = as.array(data[train_obs, y]),
                                     eval.data =
                                         list(data = as.array(t(data[test_obs, -y])),
                                              label = as.array(data[test_obs, y])),
                                     array.layout = 'colmajor',
                                     eval.metric=mx.metric.mae,
                                     learning.rate = params$learning.rate,
                                     momentum = params$momentum,
                                     wd = params$wd,
                                     array.batch.size = params$batch.size,
                                     num.round = params$num.round)
    
    pred <- predict(m, as.array(t(data[test_obs, -y])), array.layout = 'colmajor')
    pred_train <- predict(m, as.array(t(data[train_obs, -y])), array.layout = 'colmajor')
    
    MAE_test <- mean(abs(pred - data[test_obs, y]))
    MAE_train <- mean(abs(pred_train - data[train_obs, y]))
    
    return(list(model = m, MAE_test = MAE_test, MAE_train = MAE_train))
}

# cross validation
cv <- function(params = list(), param = NULL, values = NULL, folds, model = 'xgb') {
    if (is.null(param)) {
        param = 'evaluation!'
        values = c(0)
    }
    cat("\n\nModel: ", model, "\n\n")
    train_obs <- do.call(c, folds)
    out <- lapply(values, function(value) {
        par <- params
        par[[param]] <- value
        cat(param, " : ", value, '\n')
        iter <- lapply(folds, function(fold) {
            
            if (model == 'xgb')
                ret <- xgb_model(setdiff(train_obs, fold), fold, par)
            
            if (model == 'nn')
                ret <- nn_model(setdiff(train_obs, fold), fold, par)
            
            cat('\nMAE test: ', ret$MAE_test, 'train: ', ret$MAE_train,'\n')
            
            return(ret)
        })
        cat('\nMAE mean test: ', mean(sapply(iter, '[[', 'MAE_test')), 'train: ',
            mean(sapply(iter, '[[', 'MAE_train')),"\n\n")
        
        return(iter)
        
    })
    
    v <- sapply(values, rep, length(folds))
    test_data <- sapply(out, sapply, '[[', 'MAE_test')
    train_data <- sapply(out, sapply, '[[', 'MAE_train')
    
    #plot something (not useful in case of evaluation)
    plot(v, train_data, xlab = param, ylab = 'MAE train', col = 'red')
    lines(values, sapply(out, function(l) mean(sapply(l, '[[', 'MAE_train'))), col = 'red')
    
    plot(v, test_data, xlab = param, ylab = 'MAE test', col = 'blue')
    lines(values, sapply(out, function(l) mean(sapply(l, '[[', 'MAE_test'))), col = 'blue')
    
    
    return(out)
}

# params for xgb
params <- list(
    eta = 0.05,
    max_depth = 8,
    min_child_weight = 110,
    subsample = 0.8,
    nrounds = 1000
)

# params for nn
params_nn <- list(
    learning.rate = 3e-4,
    momentum = 0.9,
    batch.size = 128,
    wd = 0,
    num.round = 60
)

#load data
columns <- take_columns('train_final.csv', 'test_final.csv')
data_list <- load_data("train_final.csv", columns)
data <- data_list$data

# take index of response
y <- which(colnames(data) == 'loss')
# set cox-box parameter
lambda = 0.3

# take train sample
train_obs <- get_train_sample(frac = 0.8)

# train models
# folds <- generate_folds(train_obs = train_obs)
# out <- cv(params = params, folds = folds, model = 'xgb')
folds_nn <- generate_folds(train_obs = train_obs, n_fold = 3)
out_nn <- cv(params = params_nn, folds = folds_nn, model = 'nn')


merge_prediction <- function(data, obs) {
    models <- lapply(out[[1]], '[[', 'model')
    test <- xgb.DMatrix(data[obs,-y])
    pred <- rowMeans(sapply(models, function(m) {
        detransform(predict(m, test), lambda)
    }))
    
    return(pred)
}

merge_prediction_nn <- function(data, obs) {
    models_nn <- lapply(out_nn[[1]], '[[', 'model')
    pred_nn <- rowMeans(sapply(models_nn, function(m) {
        predict(m, as.array(t(data[obs, -y])), array.layout = 'colmajor')
    }))
    return(pred_nn)
}
prediction_both <- function(p) {
    mean(abs(pred * p + pred_nn *(1 - p) - data[-train_obs, y]))
}

# # predict
# pred <- merge_prediction(data, -train_obs)
pred_nn <- merge_prediction_nn(data, -train_obs)

# # see how combined results perform
# grid <- seq(0,1,0.01)
# mae_both <- sapply(grid, prediction_both)
# 
# p <- grid[which.min(mae_both)]
# 
# plot(grid, mae_both, xlab = 'p', ylab = 'Combined MAE')
