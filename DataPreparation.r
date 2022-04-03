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

#remove invalid birthdays 
summary(final$bd)
clean = final[final$bd > 10 & final$bd < 90, ]

head(clean)
#check for dataset balance
clean$is_churn = as.factor(clean$is_churn)
summary(clean$is_churn)

#output clean.csv for use in models
write.csv(clean,"clean.csv", row.names = FALSE)
