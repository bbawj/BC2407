

#### Loading in package and data####
library(xgboost)
library(data.table)
library(stringr)
library(caret)
library(car)
library(caTools)
library(Matrix)
library(ROSE)
library(lubridate)

clean_path = "clean.csv"
setwd("C:/Users/shuny/Documents/UNI/BC2407 Analytics II/project/kkbox")
df = fread(clean_path, stringsAsFactors = T, encoding="UTF-8")

df$msno = NULL
df$is_churn = factor(df$is_churn, levels= c(0,1), labels=c("renewal","churn"))
df$payment_method_id = factor(df$payment_method_id)
df$is_auto_renew = factor(df$is_auto_renew, levels= c(0,1), labels=c("non-auto renewal","auto-renewal"))
df$transaction_date = ymd(df$transaction_date)
df$membership_expire_date = ymd(df$membership_expire_date)
df$is_cancel = factor(df$is_cancel, levels= c(0,1), labels=c("non-cancellation","cancellation"))
df$city = factor(df$city)
df$gender = factor(df$gender, levels=c("male","female"), labels =c("female","male"))
df$registered_via = factor(df$registered_via)
df$registration_init_time = ymd(df$registration_init_time)
df$date = ymd(df$date)
df$corrected_churn = NULL
df = df[complete.cases(df),] ## removed non complete cases for xgboost
df = df[,c("payment_method_id", "plan_list_price", "actual_amount_paid", "is_auto_renew", "is_cancel", "bd", "gender", "city", "registered_via","is_churn")]
summary(df)
str(df)

#### Train Test Split####
df_split <- sample.split(Y = df$is_churn, SplitRatio = 0.7)
trainset.naive <- subset(df, df_split == T)
trainset.rose = ROSE(is_churn~., data=trainset.naive, seed=2407)$data
setDT(trainset.rose)
testset  <- subset(df, df_split == F)


train.naive_target <- trainset.naive$is_churn
train.rose_target <- trainset.rose$is_churn
test_target <- testset$is_churn
# convert factor types to numeric: churn is 1, renewal is 0
train.naive_target = as.numeric(train.naive_target)-1
train.rose_target = as.numeric(train.rose_target)-1
test_target = as.numeric(test_target)-1


#### Further processing: One hot encoding, DMatrix ####
# convert to matrix representation
train.naive_Matrix = sparse.model.matrix(~.+0, data= trainset.naive[,-c("is_churn"), with=F])
train.rose_Matrix = sparse.model.matrix(~.+0, data= trainset.rose[,-c("is_churn"), with=F])
testMatrix = sparse.model.matrix(~.+0, data= testset[,-c("is_churn"), with=F])

# xgb matrix
dtrain.naive = xgb.DMatrix(data = train.naive_Matrix, label = train.naive_target)
dtrain.rose = xgb.DMatrix(data = train.rose_Matrix, label = train.rose_target)
dtest = xgb.DMatrix(data = testMatrix, label = test_target)



#### XGBoost ####

### NAIVE
params.naive1 = list(booster = "gbtree",
               objective = "binary:logistic",
               eta=0.1, # aka learning rate, how much each tree contributes:
                        #low val -> more robust to overfitting, longer compute time
               gamma=0, # loss reduction required to further partition (denominator)
                        # larger -> more conservative (less splits)
               max_depth=6,
               min_child_weight=1, # larger -> more conservative
               subsample=1,
               colsample_bytree=1)

# find best number of iterations
xgbcv.naive1 = xgb.cv(params = params.naive1,
                data = dtrain.naive,
                nrounds = 100,
                nfold = 5,
                showsd = T,
                stratified = T,
                silent = F,
                early_stopping_rounds = 20,
                maximize = T,
                eval_metric = "aucpr"
)
xgb.naive1 <- xgb.train(
    params = params.naive1,
    data = dtrain.naive,
    nrounds = 300, #xgbcv.naive1$best_iteration,
    watchlist = list(val=dtest,train=dtrain.naive),
    print_every_n = 5,
    early_stopping_rounds = 10,
    maximize = T ,
    eval_metric = "aucpr"
)

### ROSE
params.rose1 = list(booster = "gbtree",
                     objective = "binary:logistic",
                     eta=0.1, # aka learning rate, how much each tree contributes:
                     #low val -> more robust to overfitting, longer compute time
                     gamma=0, # loss reduction required to further partition (denominator)
                     # larger -> more conservative (less splits)
                     max_depth=6,
                     min_child_weight=1, # larger -> more conservative
                     subsample=1,
                     colsample_bytree=1)

# find best number of iterations
xgbcv.rose1 = xgb.cv(params = params.rose1,
                      data = dtrain.rose,
                      nrounds = 500,
                      nfold = 5,
                      showsd = T,
                      stratified = T,
                      silent = F,
                      early_stopping_rounds = 20,
                      maximize = T,
                      eval_metric = "aucpr"
)
xgb.rose1 <- xgb.train(
    params = params.rose1,
    data = dtrain.rose,
    nrounds = 300, #xgbcv.rose1$best_iteration,
    watchlist = list(val=dtest,train=dtrain.rose),
    print_every_n = 5,
    early_stopping_rounds = 10,
    maximize = T ,
    eval_metric = "aucpr"
)
xgb.rose1


# use cut off 0.5,
CUT_OFF_HINGE_VAL = 0.5

xgb.naive1_pred <- predict(object= xgb.naive1,
                           newdata= dtest, type = "response")
xgb.naive1_pred <- ifelse(xgb.naive1_pred>CUT_OFF_HINGE_VAL, 1, 0)

cm.naive1 = confusionMatrix(factor(xgb.naive1_pred), factor(test_target), positive ="1", mode="everything")
cm.naive1
cat("Precision [TP/(TP+FP)]: ",cm.naive1[["byClass"]]["Pos Pred Value"])
cat("Recall    [TP/(TP+FN)]: ",cm.naive1[["byClass"]]["Sensitivity"])
xgb.plot.importance(importance_matrix = xgb.importance(feature_names = colnames(train.naive_Matrix),
                                                       model = xgb.naive1)[1:20])

#precision


xgb.rose1_pred <- predict(object = xgb.rose1,
                          newdata = dtest, type = "response")
xgb.rose1_pred <- ifelse(xgb.rose1_pred>CUT_OFF_HINGE_VAL, 1, 0)

cm.rose1 = confusionMatrix(factor(xgb.rose1_pred), factor(test_target), positive="1",mode="everything")
cm.rose1
cat("Precision [TP/(TP+FP)]: ",cm.rose1[["byClass"]]["Pos Pred Value"])
cat("Recall    [TP/(TP+FN)]: ",cm.rose1[["byClass"]]["Sensitivity"])
xgb.plot.importance(importance_matrix = xgb.importance(feature_names = colnames(train.rose_Matrix),
                                                       model = xgb.rose1)[1:20])



####References####
##### code for this: https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/