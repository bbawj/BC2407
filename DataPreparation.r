library(data.table)
library(plyr)
library(dplyr)
library(tidyr)

path.data = "C:/Users/bawj/Downloads/School/Year 2/BC2407/Project/train.csv"
trans.data = "C:/Users/bawj/Downloads/School/Year 2/BC2407/Project/transactions_v2.csv"
members.data = "C:/Users/bawj/Downloads/School/Year 2/BC2407/Project/members_v3.csv"
user.data = "C:/Users/bawj/Downloads/School/Year 2/BC2407/Project/user_logs_v2.csv"

df = fread(path.data, stringsAsFactors = T)
trans = fread(trans.data, stringsAsFactors = T)
members = fread(members.data, stringsAsFactors = T)
user = fread(user.data, stringsAsFactors = T)

head(members)
head(trans)
head(user)

final = join_all(list(df, trans, members, user), by="msno", match="first", type = "left")
final = na.omit(final)
summary(final$transaction_date)

#remove wonky birthdays 
summary(final$bd)
clean = final[final$bd > 10 & final$bd < 90, ]

summary(clean$membership_expire_date)
head(clean)

#we want to check if is_churn is properly set
temp = join(df, trans, by = "msno", type = "left", match = "all")
temp = na.omit(temp)
#get the max expiry date for each user
temp1 = temp %>% group_by(msno) %>% summarise(max_expiry = max(membership_expire_date))
cancelled_at_max_expiry = left_join(temp1 ,temp, by=c("msno"="msno", "max_expiry"="membership_expire_date"), match="all")
#if max expiry is greater than 30th april 2017, we set as not churned
notchurn = cancelled_at_max_expiry[cancelled_at_max_expiry$max_expiry > 20170430 & cancelled_at_max_expiry$is_cancel==0, ]
#6k people incorrectly labeled as churned
temp2 = clean[clean$is_churn == 1 & clean$msno %in% notchurn$msno, ]
a = notchurn[notchurn$msno=="XaPhtGLk/5UvvOYHcONTwsnH97P4eGECeq+BARGItRw=",]

clean[, corrected_churn:= ifelse(clean$msno %in% notchurn$msno, 0, 1)]
length(unique(cancelled_at_max_expiry$msno))
#check for dataset balance
clean$corrected_churn = as.factor(clean$corrected_churn)
clean$is_churn = as.factor(clean$is_churn)
summary(clean$is_churn)
summary(clean$corrected_churn)

library(ROSE)
corrected.balanced <- ovun.sample(corrected_churn~., data=clean, 
                                  p=0.5, seed=2407, 
                                  method="under")$data
original.balanced<- ovun.sample(is_churn~., data=clean, 
                                p=0.5, seed=2407, 
                                method="under")$data
#remove extra columns?
#clean = subset(clean, select = -c(registration_init_time, date, transaction_date, membership_expire_date))

#train-test split
library(caTools)
train <- sample.split(Y =corrected.balanced$is_churn, SplitRatio = 0.7)
trainset <- subset(corrected.balanced, train == T)
testset <- subset(corrected.balanced, train == F)

# new model without income
m2 = glm(is_churn ~ gender + bd + is_auto_renew + plan_list_price + total_secs, , family = binomial, data=trainset) 
#train set predictor
prob <- predict(m2, type = 'response')

# Confusion Matrix on Trainset
prob.train <- predict(m2, type = 'response')
m2.predict.train <- ifelse(prob.train > 0.5, 1, 0)
table <- table(Trainset.Actual = trainset$is_churn, m2.predict.train, deparse.level = 2)
table
round(prop.table(table),3)
# Overall Accuracy
mean(m2.predict.train == trainset$is_churn)

#train-test split
library(caTools)
train <- sample.split(Y =original.balanced$is_churn, SplitRatio = 0.7)
trainset <- subset(original.balanced, train == T)
testset <- subset(original.balanced, train == F)

# new model without income
m2 = glm(is_churn ~ gender + bd + is_auto_renew + plan_list_price + total_secs, , family = binomial, data=trainset) 
#train set predictor
prob <- predict(m2, type = 'response')

# Confusion Matrix on Trainset
prob.train <- predict(m2, type = 'response')
m2.predict.train <- ifelse(prob.train > 0.5, 1, 0)
table <- table(Trainset.Actual = trainset$is_churn, m2.predict.train, deparse.level = 2)
table
round(prop.table(table),3)
# Overall Accuracy
mean(m2.predict.train == trainset$is_churn)

write.csv(clean,"clean.csv", row.names = FALSE)
write.csv(original.balanced,"origninal_balanced.csv", row.names = FALSE)
write.csv(corrected.balanced,"corrected_balanced.csv", row.names = FALSE)
