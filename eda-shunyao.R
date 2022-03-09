#### Dependencies, Imports ####
library(ggplot2)
library(data.table)
library(lubridate)
clean_path = "clean.csv"
df = fread(clean_path, stringsAsFactors = F, encoding="UTF-8")



#### Data Cleaning ####
summary(df)
str(df)
df$msno = factor(df$msno)
df$gender = factor(df$gender, levels=c("male","female"), labels =c("female","male"))
df$transaction_date = ymd(df$transaction_date)
df$membership_expire_date = ymd(df$membership_expire_date)
df$registration_init_time = ymd(df$registration_init_time)
df$date = ymd(df$date)
df$is_churn = factor(df$is_churn, levels= c(0,1), labels=c("renewal","churn"))
df$is_auto_renew = factor(df$is_auto_renew, levels= c(0,1), labels=c("non-auto renewal","auto-renewal"))
df$is_cancel = factor(df$is_cancel, levels= c(0,1), labels=c("non-cancellation","cancellation"))
df$corrected_churn = NULL
summary(df)
str(df)


#### Check for NA ####
summary(is.na(df)) ## none



#### Find relationships for `plan_list_price`, `actual_amount_paid` ####
ggplot(df, aes(actual_amount_paid)) + geom_histogram(binwidth=5)
# the distribution looks like it falls within certain categories
actualPrice_factored = factor(df$actual_amount_paid)
planPrice_factored = factor(df$plan_list_price)
summary(actualPrice_factored)
summary(planPrice_factored)
# are the NTD 2,000 prices actually real? ~S$50
# is the plan_list_price significantly different from actual_amount_paid?
df_diffprice = df[df$actual_amount_paid!=df$plan_list_price]
df_diffprice[,disc_pct := (plan_list_price-actual_amount_paid)/plan_list_price]
summary(df_diffprice$disc_pct)
quantile(df_diffprice$disc_pct, 0.33)
# two-thirds of those who got a discount didn't actually pay anything
# could this be predictive?
ggplot(df_diffprice, aes(disc_pct, is_churn)) +
    geom_violin()
# looks promising, check main df
df[,disc_pct := (plan_list_price-actual_amount_paid)/plan_list_price]
ggplot(df, aes(disc_pct, is_churn)) +
    geom_violin()
# too hard to see like this, check for paid amount = 0
df[,free:=fifelse(actual_amount_paid==0, "free", fifelse(actual_amount_paid<plan_list_price, "discounted", "full-price"))]
df$free = factor(df$free)
table1 = table(df$free, df$is_churn, deparse.level = 2)
percentage_stay_by_disc = prop.table(table1,margin=1)[,1]
percentage_stay_by_disc
cbind(table1, percentage_stay_by_disc)
ggplot(data.table(price = c("discounted","free","full-price"),
                  pct_stayed = 100*percentage_stay_by_disc),
       aes(price, pct_stayed)) +
    geom_col(fill =c("springgreen3","firebrick2","steelblue3")) + theme_minimal() +
    geom_text(aes(label = round(pct_stayed,2)), hjust=1.5, colour='white') +
    labs(y = "Percentage of users that stayed (did not churn)",
         x = "Payment Plan",
         title = "Relationship between users who stayed and their price plan") +
    coord_flip()
cat("32% of customers on the free plan churned compared to <1% on the discounted plan.
This is as opposed to ~3% of customers staying and paying full price.")



#### Find relationships for `is_auto_renew`####
table2 = table(df$is_auto_renew, df$is_churn, deparse.level = 2)
percentage_stay_by_renewal = prop.table(table2,margin=1)[,1]
cbind(table2, percentage_stay_by_renewal)
ggplot(data.table(renewal_status = c("non-auto-renewal","auto-renewal"),
                  pct_stayed = 100*percentage_stay_by_renewal),
       aes(renewal_status, pct_stayed)) +
    geom_col(fill =c("firebrick2","springgreen3")) + theme_minimal() +
    geom_text(aes(label = round(pct_stayed,2)), vjust=2, colour='white') +
    labs(y = "Percentage of users that stayed (did not churn)",
         x = "Renewal Option",
         title = "Relationship between users who stayed and their renewal option") +
    coord_flip()
cat("99.42% of customers with auto-renewal enabled stayed.
This is as opposed to only 81.25% of customers staying.")



#### Find relationships for `bd`, `gender` ####
ggplot(df[gender!=""], aes(x = gender, y = bd, fill = gender)) +
    geom_violin() +
    facet_grid(.~is_churn) +
    coord_flip()
# gender does not seem to have any impact on renewal, rather age seems to be difference
ggplot(df, aes(bd, fill = is_churn)) +
    geom_histogram(binwidth = 5) +
    facet_grid(is_churn~., scales="free_y") +
    scale_fill_manual(values = c("springgreen3", "firebrick2"))
cat("Those younger than 25 are significantly more likely to churn.")
# could this be due to their low/lack of income? in other words, they are more price sensitive?
ggplot(df, aes(bd, fill = is_churn)) +
    geom_histogram(binwidth=5, position = "stack", color = "white", boundary=0) +
    facet_grid(df$free, scales="free_y") +
    scale_x_continuous(breaks = rep(1:18)*5, minor_breaks = waiver()) +
    labs(x = "Age",
         y = "Number of users",
         title = "Number of users that churn, classified by age and price plan")
cat("Statistically significant numbers of churn are only found in the age range of 15-40.")
# investigate this specific range
ggplot(df, aes(bd, fill = is_churn)) +
    geom_histogram(binwidth=5, position = "fill", color = "white", boundary=0) +
    facet_grid(df$free, scales="free_y") +
    scale_x_continuous(breaks = rep(1:18)*5,limits = c(15,40)) +
    labs(x = "Age",
         y = "Proportion of users",
         title = "Proportion of users that churn, classified by age and price plan")
cat("Churn proportion remains largely constant for each category of users,
with the largest group belonging to those on the free plan.")
# no, it seems like everyone, regardless of whether you are a student or not, is not willing to pay


#### Find relationships for users dataset i.e. num_25 etc. ####
#distribution across percentage of songs played a % duration
long = melt(df, id.vars = c("msno", "is_churn"), measure.vars = c("num_25", "num_50", "num_75", "num_985", "num_100"))
ggplot(long, aes(value, fill=is_churn))+
    geom_density(alpha=0.4) +
    facet_grid(long$variable, scales="free_y")+
    scale_x_continuous(breaks = rep(1:18)*5,limits = c(0,40))+
    labs(x = "Number of songs",
         y = "Proportion of users",
         title = "Proportion of users that churn, classified by number of songs played a % duration")
cat("There is not much difference in the distribution of churn against the number of songs 
    played for each % of duration.")

#number of unique songs
ggplot(df, aes(num_unq, fill=is_churn))+
    geom_density(alpha=0.4) +
    labs(x = "Number of unique songs",
         y = "Proportion of users",
         title = "Proportion of users that churn, classified by number of unique songs played")

#number of unique songs
ggplot(df, aes(total_secs, fill=is_churn))+
    geom_density(alpha=0.4) +
    labs(x = "Total seconds listened",
         y = "Proportion of users",
         title = "Proportion of users that churn, classified by total seconds listened")
cat("The distribution of total seconds listened is similar between those who renew and those who churn")