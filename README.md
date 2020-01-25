# **Loan Prediction – Machine Learning in Python**
Project files are available [here](https://drive.google.com/drive/folders/1L5Sk9Y2J-xP_1sF-C7KZfhckRplT5FUq?usp=sharing).
# 1.	**Data**
Data is provided by Home Credit (source: Kaggle.com).  Home Credit is an international consumer finance provider and they help people with little or no credit history to obtain loans.
In this project I am using four data frames:
1)	application_train
2)	application_test
3)	bureau
4)	bureau_balance

First two are data with information about each loan application at Home Credit. Application data contains features as gender, number of children and income. One row represents one loan which is identified by feature: SK_ID_CURR. Train is pulling together with Target. It is indicated by 0-will repay loan on time or 1-will have difficulty repaying loan. The application_train dataset has 122 variables from 307511 loans.

I merged data with extra information (3,4) to improve model's performance. Bureau data includes all client’s previous credits provided by other financial institution. For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date. Bureau_balance provides information about monthly balances of previous credits in Credit Bureau. In this table each has one row for each month of every previous credit.
HomeCredit_columns_description.csv contains definitions of all columns that are used as a data frame in this project
# 2.	Purpose

The main goal of this project is to predict if the loan would be repaid (0) or the loan would not be repaid on time (1).
Project is divided into three parts:

A)	**Pre-processing**

To prepare the data I have used Pandas.  This process is divided into eight parts:
-	Part I - read data
-	Part II - aggregate statistics for numeric variables
-	Part III - aggregate statistics for categorical variables
-	Part IV - merge data
-	Part V - missing values function: Remove columns contain more than 60% missing data. For low number of missing values (lower than 60%) - apply the mean for float64 objects and "Other" for string columns.
-	Part VI - calculate correlation between variable [matrix]
- Part VII - apply thresholds for removing collinear variables: Threshold is set to 0.9
-	Part VIII - save prepared data to csv

I have prepared several functions to pre-process the data, because it is much easier to apply small changes e.g.: try different thresholds by changing only one variable. Another plus to having function is that I could apply those ones to another data sets without no or small changes.

This was the most time-consuming part, but it allows to have efficient analysis.
After all pre-processing steps the data shape is:
Train: (307511, 162)
Test: (48744, 161)

B)	**Encoding categorical variables**

To build the machine learning model the categorical variables need to be encoded. In this case I have used Label Encoding (Scikit-Learn), which assigns each unique categorical variable to an integer without additional columns and also One-hot encoding, which creates a new column for each unique category in categorical variable (Pandas - get_dummies(df)). In this part I have used ‘ohe' for one-hot encoding and 'le' for integer label encoding.

C)	**Train model**

The model is built with common machine learning algorithms: Logistic Regression and Random Forest Classifier. I have compared results of both models using classification_report. The obtained accuracy for Logistic Regression is 0.616 and for Random Forest Classifier: 0.720.
Then, I have built the model using LightGBM. According to lightgbm.readthedocs.io, LightGBM has a lot of advantages: it is fast, achieves better accuracy that any other boosting algorithm and is compatible with large data sets.
In model parameters, I have set Use small learning_rate (0.01) with large num_iterations (5000), num_leaves to 48 to achieve better accuracy (minus – calculation was slower). I have tested different boosting types: “gbdt”, “goss” and “dart”. Default method 'gbdt' gave good result at a reasonable speed (AUC:0.767). “Dart” provided comparable results (AUC: 0.767) but was much slower. The best result here was obtained with “goss” option (AUC: 0.768), so I have chosen this method.
I calculated predictions for test data and saved results to csv file as Submission.csv where:
1 - the loan will be not repaid on time,
0 - the loan will be repaid.

The analysis showed that only 0.203% of clients would have difficulties with repaying the loan. This result confirms that Home Credit is good with estimating the amount of the credit accordingly to the client’s status.

# 3.	**Conclusion**

I have selected seven variables with highest correlation (but lower than 90%) with TARGET to compare the data for clients who will or will not have difficulties with repaying the loan.
-	An amount credited is estimated appropriate to customer income (AMT_INCOME_TOTAL)
-	It is harder to repay smaller loans (AMT_CREDIT) with lower loan annuity (AMT_ANNUITY)
-	The number of days before the application that person had started current employment (DAYS_EMPLOYED) has a huge impact. People who have stable position less often have problem to repay loan than people who change job often.
-	People who apply for credit near the deadline will have problem to repay it (DAYS_CREDIT)
-	The more days are left to pay back the previous loan, the higher the probability that the client will have difficulties to repay the requested loan (DAYS_CREDIT_ENDDATE)
-	Clients with difficulties to repay have a tendency to change ID documents before issuing application (DAYS_ID_PUBLISH)

# 4.	**Next steps:**
-	Predict, whether the client can repay the loan or not, by using voting assemble techniques of combining the predictions from multiple machine learning algorithms.
-	Add more variables. Home Credit provided four more .csv files on Kaggle.com which can improve model. This data contains information such as credit card balance.
-	Apply different methods for missing data e.g.: Multiple Imputation using MICE or Bayesian Imputation using a Gaussian model and check whether this technique improves model.
