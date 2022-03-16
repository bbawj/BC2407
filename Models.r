setwd(getwd())

#### GENERATING SYNTHETIC BALANCED DATA WITH ROSE ALGORITHM ####
# library(data.table)
# clean = read.csv("clean.csv", stringsAsFactors=T)
# clean$msno = factor(clean$msno)
# clean$gender = factor(clean$gender, levels=c("male","female"), labels =c("male","female"))
# clean$transaction_date = NULL
# clean$membership_expire_date = NULL
# clean$registration_init_time = NULL
# clean$date = NULL
# clean$corrected_churn = NULL
# clean$is_churn = factor(clean$is_churn, levels= c(0,1), labels=c("not churn","churn"))
# clean$is_auto_renew = factor(clean$is_auto_renew, levels= c(0,1), labels=c("non-auto renewal","auto-renewal"))
# clean$is_cancel = factor(clean$is_cancel, levels= c(0,1), labels=c("non-cancellation","cancellation"))
# clean$city = factor(clean$city)
# clean$registered_via = factor(clean$registered_via)
# clean$payment_plan_days = factor(clean$payment_plan_days)
# clean$payment_method_id = factor(clean$payment_method_id)
# clean = na.omit(clean)

# 
# library(ROSE)
# rose = ROSE(is_churn~., data=clean, seed=2407)$data
# summary(rose$is_churn)
# write.csv(rose, "rose.csv", row.names = FALSE)
#### END ROSE ####

### TRY SMOTE ###

# library(smotefamily)
# library(performanceEstimation)
# cleandf = read.csv("clean.csv", stringsAsFactors=T)
# cleandf$transaction_date = NULL
# cleandf$membership_expire_date = NULL
# cleandf$registration_init_time = NULL
# cleandf$date = NULL
# cleandf$corrected_churn = NULL
# cleandf$msno = NULL
# cleandf$gender = factor(cleandf$gender, levels=c("male", "female"), labels=c(0,1))
# cleandf$gender = as.numeric(cleandf$gender)
# cleandf = na.omit(cleandf)
# 
# smote = SMOTE(cleandf[,-18], cleandf$is_churn)$data
# smote2 = smote(is_churn ~ ., cleandf, perc.over=1)
# 
# write.csv(smote, "smote.csv", row.names = FALSE)
# write.csv(smote2, "smote2.csv", row.names = FALSE)

#### END SMOTE ####

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
cleandf = na.omit(cleandf)
summary(cleandf)
sum(is.na(cleandf))

### Correlation analysis ###
library(ggcorrplot)
library(dplyr)
cor = model.matrix(~0+., data=cleandf) %>% 
  cor(use="pairwise.complete.obs")

cor = as.data.frame(as.table(cor))
corTable = subset(cor, abs(Freq) > 0.5 & abs(Freq) != 1)
corTable[!duplicated(corTable[,c('Freq')]),]

cor = findCorrelation(cor, cutoff=0.5)

library(Information)
library(InformationValue)
options(scipen = 999, digits=5)
create_infotables(as.data.frame(model.matrix(~0+., data=cleandf)),y="is_churnchurn")

#### OR LOAD SMOTE ####
# df = read.csv("smote.csv", stringsAsFactors = T)
# df$gender = factor(df$gender, levels=c(1,2), labels =c("male","female"))
# df$is_churn = factor(df$is_churn, levels= c(0,1), labels=c("not churn","churn"))
# df$is_auto_renew = factor(df$is_auto_renew, levels= c(0,1), labels=c("non-auto renewal","auto-renewal"))
# df$is_cancel = factor(df$is_cancel, levels= c(0,1), labels=c("non-cancellation","cancellation"))
# df$city = factor(df$city)
# df$registered_via = factor(df$registered_via)
# df$payment_plan_days = factor(df$payment_plan_days)
# df$payment_method_id = factor(df$payment_method_id)
# df = na.omit(df)
# summary(df)

#### Data Cleaning ####

#train-test split
library(caTools)
library(caret)
library(ROSE)
library(pROC)
library(PRROC)
set.seed(2407)

df = as.data.frame(model.matrix(~0+., data=cleandf))
colnames(df)[1] = "is_churn"
#remove correlated columns
df = df[,-c(cor)]
cleandf=df

train <- sample.split(Y = cleandf$is_churn, SplitRatio = 0.7)
trainset.naive <- subset(cleandf, train == T)
trainset.rose = ROSE(is_churn~., data=trainset.naive, seed=2407)$data
testset <- subset(cleandf, train == F)
summary(trainset.naive$is_churn)
summary(trainset.rose$is_churn)
summary(testset$is_churn)


#### BEGIN LOGISTIC REGRESSION ####
log.naive = glm(is_churn ~ ., family = binomial, data=trainset.naive)

log.rose = glm(is_churn ~ ., family = binomial, data=trainset.rose) 

summary(log.rose)
threshold = 0.5

# Confusion Matrix Naive
log.predict.naive <- predict(log.naive, newdata = testset, type = 'response')
log.class.naive <- ifelse(log.predict.naive > threshold, 1, 0)
caret::confusionMatrix(factor(log.class.naive), reference=factor(testset$is_churn), positive="1", mode="prec_recall")
auc(testset$is_churn, log.predict.naive)
pr.curve(log.predict.naive[testset$is_churn == 1], log.predict.naive[testset$is_churn == 0])

#get coefficients
exp(coef(log.naive))

# Confusion Matrix ROSE
log.predict.rose <- predict(log.rose, newdata = testset, type = 'response')
log.class.rose <- ifelse(log.predict.rose > threshold, 1, 0)
caret::confusionMatrix(as.factor(log.class.rose), reference=factor(testset$is_churn), positive="1", mode="prec_recall")
auc(testset$is_churn, log.predict.rose)
pr.curve(log.predict.rose[testset$is_churn == 1], log.predict.rose[testset$is_churn == 0])

#### END LOGISTIC REGRESSION ####

#### BEGIN CART ####
library(rpart)
library(rpart.plot)

# Naive CART 
cart <- rpart(is_churn ~ payment_method_id + plan_list_price + actual_amount_paid + is_auto_renew + 
                is_cancel + bd + gender + city + registered_via, data = trainset.naive, method = 'class', control = rpart.control(minsplit = 2, cp=0))
CVerror.cap <- cart$cptable[which.min(cart$cptable[,"xerror"]), "xerror"] + cart$cptable[which.min(cart$cptable[,"xerror"]), "xstd"]
i <- 1; j<- 4

while (cart$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}
cart.cp = ifelse(i > 1, sqrt(cart$cptable[i,1] * cart$cptable[i-1,1]), 1)
cart.naive = prune(cart, cp = cart.cp)
cart.naive$variable.importance

#proportion of importance
#round(100*cart.naive$variable.importance/sum(cart.naive$variable.importance))
#plotcp(cart.naive)
#rpart.plot(cart.naive, nn = T) #-> uncomment to see plot of CART model (commenting out for smooth running of code)

# ROSE cart
cart <- rpart(is_churn ~ payment_method_id + plan_list_price + actual_amount_paid + is_auto_renew + 
                is_cancel + bd + gender + city + registered_via, data = trainset.rose, method = 'class', control = rpart.control(minsplit = 2, cp=0))
CVerror.cap <- cart$cptable[which.min(cart$cptable[,"xerror"]), "xerror"] + cart$cptable[which.min(cart$cptable[,"xerror"]), "xstd"]
i <- 1; j<- 4

while (cart$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}
cart.cp = ifelse(i > 1, sqrt(cart$cptable[i,1] * cart$cptable[i-1,1]), 1)
cart.rose = prune(cart, cp = cart.cp)
cart.rose$variable.importance
round(100*cart.rose$variable.importance/sum(cart.rose$variable.importance))

# confusion matrix naive
cart.predict.naive = predict(cart.naive, newdata = testset, type="prob")[,2]
cart.class.naive = ifelse(cart.predict.naive > threshold, "churn", "not churn")
confusionMatrix(as.factor(cart.class.naive), reference=testset$is_churn, positive="churn", mode="prec_recall")
auc(testset$is_churn, cart.predict.naive)
pr.curve(cart.predict.naive[testset$is_churn == "churn"], cart.predict.naive[testset$is_churn == "not churn"])

# confusion matrix rose
cart.predict.rose = predict(cart.rose, newdata = testset, type="prob")[,2]
cart.class.rose = ifelse(cart.predict.rose > threshold, "churn", "not churn")
confusionMatrix(as.factor(cart.class.rose), reference=testset$is_churn, positive="churn", mode="prec_recall")
auc(testset$is_churn, cart.predict.rose)
pr.curve(cart.predict.rose[testset$is_churn == "churn"], cart.predict.rose[testset$is_churn == "not churn"])
#### END OF CART ####

#### BEGIN NEURAL NET ####
library(nnet)
library(neuralnet)
set.seed(2407)
temp = trainset
temp$is_churn = ifelse(temp$is_churn == "churn", 1, 0)
temp$is_auto_renew = ifelse(temp$is_auto_renew == "auto-renewal", 1, 0)
temp$is_cancel = ifelse(temp$is_cancel == "non-cancellation", 0, 1)
temp$payment_method_id = as.numeric(temp$payment_method_id)
temp$payment_plan_days = as.numeric(temp$payment_plan_days)

temp_test = testset
temp_test$is_churn = ifelse(temp_test$is_churn == "churn", 1, 0)
temp_test$is_auto_renew = ifelse(temp_test$is_auto_renew == "auto-renewal", 1, 0)
temp_test$is_cancel = ifelse(temp_test$is_cancel == "non-cancellation", 0, 1)
temp_test$payment_method_id = as.numeric(temp_test$payment_method_id)
temp_test$payment_plan_days = as.numeric(temp_test$payment_plan_days)


nnmodel = neuralnet(is_churn ~ is_auto_renew + actual_amount_paid + payment_method_id +
                      plan_list_price + payment_plan_days + is_cancel, data = temp,hidden=2, err.fct="ce", linear.output=FALSE )

nn.predict <- ifelse(unlist(nnmodel$net.result) > 0.5, 1, 0)
cat('Trainset Confusion Matrix with neuralnet (1 hidden layer, 2 hidden nodes, Unscaled X):')
table(temp$is_churn, nn.predict)
# Overall Accuracy
mean(nn.predict == temp$is_churn)

#Test the resulting output
nn.results <- compute(nnmodel, temp_test)
results <- data.frame(actual = temp_test$is_churn, prediction = nn.results$net.result)
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)
roundedresultsdf
mean(roundedresultsdf$actual == roundedresultsdf$prediction)
#### END NEURAL NETWORK ####

#### BEGIN random forest ####
library(randomForest)
set.seed(2407)

rf.naive = randomForest(is_churn ~ ., data = trainset.naive, importance=T,
                        ntree=250)
rf.predict.naive = predict(rf.naive, newdata = testset, type="prob")
pr.curve(rf.predict.naive[testset$is_churn == 1, 1], rf.predict.naive[testset$is_churn == 0, 1])
auc(testset$is_churn, rf.predict.naive[,"churn"])

rf.rose = randomForest(is_churn ~ is_auto_renew + actual_amount_paid + payment_method_id +
                         plan_list_price + payment_plan_days + is_cancel, data = trainset.rose, importance=T,
                       ntree=250)
rf.predict.rose = predict(rf.rose, newdata = testset, type="prob")
pr.curve(scores.class0 = rf.predict.rose[testset$is_churn == "churn", "churn"], scores.class1 = rf.predict.rose[testset$is_churn == "not churn", "churn"], curve=T)
auc(testset$is_churn, rf.predict.rose[,"churn"])
plot(rf.pr.rose)

#precision
850/(850+154)
850/(850+7607)

#precision and recall
112418/(112418+8524)
112418/(112418+5945)
rf.predict.rose
plot(rf.rose)
importance(rf.rose)





