setwd("C:/Users/bawj/Downloads/School/Year 2/BC2407/Project")

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
set.seed(2407)

train <- sample.split(Y = cleandf$is_churn, SplitRatio = 0.7)
trainset.naive <- subset(cleandf, train == T)
trainset.rose = ROSE(is_churn~., data=trainset, seed=2407)$data
summary(trainset.rose$is_churn)
summary(testset$is_churn)
testset <- subset(cleandf, train == F)


#### BEGIN LOGISTIC REGRESSION ####
log.naive = glm(is_churn ~ payment_method_id + plan_list_price + actual_amount_paid + is_auto_renew + 
                  is_cancel + bd + gender + city + registered_via, family = binomial, data=trainset.naive)

log.rose = glm(is_churn ~ payment_method_id + plan_list_price + actual_amount_paid + is_auto_renew + 
           is_cancel + bd + gender + city + registered_via, family = binomial, data=trainset.rose) 

summary(log.rose)
threshold = 0.5

# Confusion Matrix Naive
prob.test.naive <- predict(log.naive, newdata = testset, type = 'response')
log.predict.test.naive <- ifelse(prob.test.naive > threshold, "churn", "not churn")
confusionMatrix(as.factor(log.predict.test.naive), reference=testset$is_churn, positive="churn", mode="prec_recall")

# Confusion Matrix ROSE
prob.test <- predict(log.rose, newdata = testset, type = 'response')
log.rose.predict.test <- ifelse(prob.test > threshold, "churn", "not churn")
confusionMatrix(as.factor(log.rose.predict.test), reference=testset$is_churn, positive="churn", mode="prec_recall")

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
round(100*cart.naive$variable.importance/sum(cart.naive$variable.importance))
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
cart.predict.naive = predict(cart.naive, newdata = testset, type="class")
confusionMatrix(as.factor(cart.predict.naive), reference=testset$is_churn, positive="churn", mode="prec_recall")

# confusion matrix naive
cart.predict.rose = predict(cart.rose, newdata = testset, type="class")
confusionMatrix(as.factor(cart.predict.rose), reference=testset$is_churn, positive="churn", mode="prec_recall")

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

rf.naive = randomForest(is_churn ~ is_auto_renew + actual_amount_paid + payment_method_id +
                         plan_list_price + payment_plan_days + is_cancel, data = trainset.naive, importance=T,
                       ntree=250)
rf.rose = randomForest(is_churn ~ is_auto_renew + actual_amount_paid + payment_method_id +
                         plan_list_price + payment_plan_days + is_cancel, data = trainset.rose, importance=T,
                       ntree=250)
rf.naive
#precision
850/(850+154)
850/(850+7607)
rf.rose
#precision and recall
112418/(112418+8524)
112418/(112418+5945)
plot(rf.rose)
importance(rfmodel)





