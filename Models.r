setwd(getwd())

#### DATA CLEANING ####
library(data.table)
cleandf = read.csv("clean.csv", stringsAsFactors=T)
cleandf$transaction_date = NULL
cleandf$membership_expire_date = NULL
cleandf$registration_init_time = NULL
cleandf$date = NULL
cleandf$corrected_churn = NULL
cleandf$msno = NULL
cleandf$is_churn = factor(cleandf$is_churn, levels= c(0,1), labels=c("not churn","churn"))
cleandf$is_auto_renew = factor(cleandf$is_auto_renew, levels= c(0,1), labels=c("non-auto renewal","auto-renewal"))
cleandf$is_cancel = factor(cleandf$is_cancel, levels= c(0,1), labels=c("non-cancellation","cancellation"))
cleandf$city = factor(cleandf$city)
cleandf$registered_via = factor(cleandf$registered_via)
cleandf$payment_plan_days = factor(cleandf$payment_plan_days)
cleandf$payment_method_id = factor(cleandf$payment_method_id)
cleandf$gender = factor(cleandf$gender, levels=c("male", "female"), labels=c("male","female"))
cleandf = na.omit(cleandf)
summary(cleandf)
sum(is.na(cleandf))

#filter out relevant variables
library(caret)
library(Information)
library(InformationValue)
options(scipen = 999, digits=5)
#obtain the IV values
cleandf$is_churn = as.numeric(cleandf$is_churn)-1
ivs = create_infotables(cleandf,y="is_churn")
ivs
cleandf$is_churn = factor(cleandf$is_churn, levels=c(0,1), labels=c("not churn", "churn"))

#choose only higher IV variables
df = dummyVars(~is_churn+payment_method_id+is_auto_renew+payment_plan_days+actual_amount_paid
               +plan_list_price+registered_via+bd+is_cancel+city+gender, data=cleandf)
df = data.frame(predict(df, newdata = cleandf))

### Correlation analysis ###
library(dplyr)
library(caret)

cor = cor(df,use="pairwise.complete.obs")

corTable = as.data.frame(as.table(cor))
corTable = subset(corTable, abs(Freq) > 0.7 & abs(Freq) != 1)
corTable[!duplicated(corTable[,c('Freq')]),]

cor = findCorrelation(cor, cutoff=0.7)
#remove correlated columns
cor
colnames(df)[cor]
finaldf = df[,-c(42, 35, 64, 34, 15, 39, 71, 94, 38)]
finaldf[, -c(63,57)] = lapply(finaldf[,-c(63,57)], function (x) factor(x))

#### END Data Cleaning ####

#train-test split
library(caTools)
library(caret)
library(ROSE)
library(pROC)
library(PRROC)
set.seed(2407)

colnames(finaldf)[1] = "is_churn"

# dashtrain <- sample.split(Y = cleandf$is_churn, SplitRatio = 0.7)
# dashtest = subset(cleandf, train == F)

train <- sample.split(Y = finaldf$is_churn, SplitRatio = 0.7)
trainset.naive <- subset(finaldf, train == T)
trainset.rose = ROSE(is_churn~., data=trainset.naive, seed=2407)$data
testset <- subset(finaldf, train == F)
summary(trainset.naive$is_churn)
summary(trainset.rose$is_churn)
summary(testset$is_churn)

write.csv(trainset.naive, "trainset_naive.csv")
write.csv(trainset.rose, "trainset_rose.csv")
write.csv(testset, "testset.csv")

#### BEGIN LOGISTIC REGRESSION ####
log.naive = glm(is_churn ~ ., family = binomial, data=trainset.naive)

log.rose = glm(is_churn ~ ., family = binomial, data=trainset.rose) 

threshold = 0.5

# Confusion Matrix Naive
log.predict.naive <- predict(log.naive, newdata = testset, type = 'response')
log.class.naive <- ifelse(log.predict.naive > threshold, 1, 0)
caret::confusionMatrix(factor(log.class.naive), reference=factor(testset$is_churn), positive="1", mode="prec_recall")
auc(testset$is_churn, log.predict.naive)
pr.curve(log.predict.naive[testset$is_churn == 1], log.predict.naive[testset$is_churn == 0])

# Confusion Matrix ROSE
log.predict.rose <- predict(log.rose, newdata = testset, type = 'response')
log.class.rose <- ifelse(log.predict.rose > threshold, 1, 0)
caret::confusionMatrix(as.factor(log.class.rose), reference=factor(testset$is_churn), positive="1", mode="prec_recall")
auc(testset$is_churn, log.predict.rose)
pr.curve(log.predict.rose[testset$is_churn == 1], log.predict.rose[testset$is_churn == 0])

#get importance
coeffs = data.frame(varImp(log.rose))
rownames(coeffs)[rownames(coeffs) == "bd"] = "bd1"
rownames(coeffs)[rownames(coeffs) == "actual_amount_paid"] = "actual_amount_paid1"
rownames(coeffs) = lapply(rownames(coeffs), function(x) substr(x,1,nchar(x)-1))

# dashboard = cbind(dashtest, PR=log.predict.rose)
# write.csv(dashboard, "dashboard.csv", row.names = F)

#### END LOGISTIC REGRESSION ####

#### BEGIN CART ####
library(rpart)
library(rpart.plot)

# Naive CART 
cart <- rpart(is_churn ~ ., data = trainset.naive, method = 'class', control = rpart.control(minsplit = 2, cp=0))
CVerror.cap <- cart$cptable[which.min(cart$cptable[,"xerror"]), "xerror"] + cart$cptable[which.min(cart$cptable[,"xerror"]), "xstd"]
i <- 1; j<- 4

while (cart$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}
cart.cp = ifelse(i > 1, sqrt(cart$cptable[i,1] * cart$cptable[i-1,1]), 1)
cart.naive = prune(cart, cp = cart.cp)

#proportion of importance
#round(100*cart.naive$variable.importance/sum(cart.naive$variable.importance))
#plotcp(cart.naive)
#rpart.plot(cart.naive, nn = T) #-> uncomment to see plot of CART model (commenting out for smooth running of code)

# ROSE cart
cart <- rpart(is_churn ~ ., data = trainset.rose, method = 'class', control = rpart.control(minsplit = 2, cp=0))
CVerror.cap <- cart$cptable[which.min(cart$cptable[,"xerror"]), "xerror"] + cart$cptable[which.min(cart$cptable[,"xerror"]), "xstd"]
i <- 1; j<- 4

while (cart$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}
cart.cp = ifelse(i > 1, sqrt(cart$cptable[i,1] * cart$cptable[i-1,1]), 1)
cart.rose = prune(cart, cp = cart.cp)
# cart.rose$variable.importance
round(100*cart.rose$variable.importance/sum(cart.rose$variable.importance))

# confusion matrix naive
cart.predict.naive = predict(cart.naive, newdata = testset, type="prob")[,2]
cart.class.naive = ifelse(cart.predict.naive > threshold, 1, 0)
caret::confusionMatrix(as.factor(cart.class.naive), reference=testset$is_churn, positive="1", mode="prec_recall")
auc(testset$is_churn, cart.predict.naive)
pr.curve(cart.predict.naive[testset$is_churn == "1"], cart.predict.naive[testset$is_churn == "0"])

# confusion matrix rose
cart.predict.rose = predict(cart.rose, newdata = testset, type="prob")[,2]
cart.class.rose = ifelse(cart.predict.rose > threshold, 1, 0)
caret::confusionMatrix(as.factor(cart.class.rose), reference=testset$is_churn, positive="1", mode="prec_recall")
auc(testset$is_churn, cart.predict.rose)
pr.curve(cart.predict.rose[testset$is_churn == "1"], cart.predict.rose[testset$is_churn == "0"])
#### END OF CART ####

#### BEGIN random forest ####
library(randomForest)
set.seed(2407)

rf.naive = randomForest(is_churn ~ ., data = trainset.naive, importance=T,
                        ntree=100)
rf.predict.naive = predict(rf.naive, newdata = testset, type="prob")
rf.class.naive = ifelse(rf.predict.naive[,"1"] > threshold, 1, 0)
caret::confusionMatrix(as.factor(rf.class.naive), reference=testset$is_churn, positive="1", mode="prec_recall")
auc(testset$is_churn, rf.predict.naive[,"1"])
pr.curve(rf.predict.naive[testset$is_churn == "1", "1"], rf.predict.naive[testset$is_churn == "0", "1"])

rf.rose = randomForest(is_churn ~ ., data = trainset.rose, importance=T,
                       ntree=100)
rf.predict.rose = predict(rf.rose, newdata = testset, type="prob")[,"1"]

rf.class.rose = ifelse(rf.predict.rose > threshold, 1, 0)
caret::confusionMatrix(as.factor(rf.class.rose), reference=testset$is_churn, positive="1", mode="everything")
auc(testset$is_churn, rf.predict.rose)
pr.curve(rf.predict.rose[testset$is_churn == "1"], rf.predict.rose[testset$is_churn == "0"], curve=T)

plot(roc(testset$is_churn, rf.predict.rose, plot=T))
plot(rf.pr.rose)

#precision
850/(850+154)
850/(850+7607)

#precision and recall
112418/(112418+8524)
112418/(112418+5945)
rf.naive
rf.rose
rf.predict.rose
plot(rf.rose)
importance(rf.rose)
varImpPlot(rf.rose)




