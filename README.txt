A description of the file names and purpose is provided here for ease of reference

CSV Files
1. clean.csv: This is the output from DataPreparation.r
2. train.csv: The train set, containing the user ids and whether they have churned. Obtained from Kaggle.
3. transactions_v2.csv: Transactions of users. Obtained from Kaggle.
4. user_logs_v2.csv: Daily user logs describing listening behaviors of a user. Obtained from Kaggle.
5. members_v3.csv: User information. Obtained from Kaggle

R Scripts
1. DataPreparation.R: R script used to merge Kaggle datasets into clean.csv
2. Models.R: R script used to generate model and results for Logistic Regression, CART and Random Forest
3. EDA.R: R script used to do exploratary data analysis
4. XGBoost.R: R script used to generate model and results for XGBoost