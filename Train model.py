import numpy as np
import lightgbm as lgb
import pandas 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#%%
X_train = pandas.read_csv('X_train.csv')
X_test = pandas.read_csv('X_test.csv')
train = pandas.read_csv('train_prepared.csv')
SK_ID_CURR=train['SK_ID_CURR']
Y_train = train['TARGET']


random_state = 42
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=random_state)
#%%
# Test LogisticRegression and LGBMClassifier
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_train),
                                                 Y_train)
class_weights = {
    0: class_weights[0],
    1: class_weights[1]
}
clf_log = LogisticRegression(solver="liblinear", class_weight=class_weights)
clf_log.fit(x_train, y_train)

y_pred = clf_log.predict(x_val)

print(classification_report(y_val, y_pred, digits=4))  


from lightgbm import LGBMClassifier
clf_lgbmc = LGBMClassifier(objective='binary', metric='auc', class_weight=class_weights, n_estimators=1000, max_depth=20)

clf_lgbmc.fit(x_train, y_train,
              eval_set = [(x_val, y_val)],
              early_stopping_rounds=50,
              verbose=0)

y_pred = clf_lgbmc.predict(x_val, num_iteration=clf_lgbmc.best_iteration_)

print(classification_report(y_val, y_pred, digits=4))   


#%% Best model

lgb_train = lgb.Dataset(data=x_train, label=y_train)
lgb_eval = lgb.Dataset(data=x_val, label=y_val)
params_gbdt = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':0.8, 'subsample':0.9, 'max_depth':7, 'reg_alpha':0.1, 'reg_lambda':0.1, 
          'min_split_gain':.01, 'min_child_weight':1}

params_goss = {'task': 'train', 'boosting_type': 'goss', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':0.8, 'subsample':0.9, 'max_depth':7, 'reg_alpha':0.1, 'reg_lambda':0.1, 
          'min_split_gain':0.01, 'min_child_weight':1}

params_dart = {'task': 'train', 'boosting_type': 'dart', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':0.8, 'subsample':0.9, 'max_depth':7, 'reg_alpha':0.1, 'reg_lambda':0.1, 
          'min_split_gain':0.01, 'min_child_weight':1}

#model_gbdt = lgb.train(params_gbdt, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
model_goss = lgb.train(params_goss, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
#model_dart = lgb.train(params_dart, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
#%%
train_pred = model_goss.predict(x_val)
test_pred = model_goss.predict(X_test)

#%%
# Prepare data to submission
def pred_function(data):
    for i in range(0,len(data)):
        if data[i]>=0.5:
            data[i]=1
        else:
            data[i]=0
    return data

train_pred_result = pred_function(train_pred)
test_pred_result = pred_function(test_pred)
train_pred_result = pandas.DataFrame(train_pred_result, columns = ['Prediction'])
test_pred_result = pandas.DataFrame(test_pred_result, columns = ['Prediction'])
sum_1 = test_pred_result.sum()
percent = ((sum_1/len(test_pred_result))*100).round(3)
print("% of clients who will have difficulty repaying loan: ", percent.to_string(index=False))
test_pred_result = test_pred_result.join(SK_ID_CURR, how = 'left')
test_pred_result  = test_pred_result.reindex(columns=['SK_ID_CURR','Prediction'])
test_pred_result.to_csv('Submission.csv', index = False, header = True)
#%%
# Prepare data for conclusion
import pandas
import numpy as np
test_pred_result = pandas.read_csv('C:\\podyplomowe\\P\\Submission.csv')

X_test_pred_result = X_test.join(test_pred_result, how = 'left')

# Analysis who will have difficulties to repay the loan, split data into repay and no repay 
Client_no_repay = X_test_pred_result.loc[X_test_pred_result['Prediction'] == 1]
Client_repay = X_test_pred_result.loc[X_test_pred_result['Prediction'] == 0]
print(X_test_pred_result.loc[X_test_pred_result['Prediction'] == 1])
# Choose variables, calculate means for each column
Client_no_repay_mean = Client_no_repay[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'bureau_DAYS_CREDIT_max', 'bureau_DAYS_CREDIT_ENDDATE_max', 'DAYS_ID_PUBLISH']]
Client_no_repay_mean= Client_no_repay_mean.mean().round(2)
Client_no_repay_mean= pandas.DataFrame(Client_no_repay_mean, columns = ['no repay'])
print(Client_no_repay_mean)

Client_repay_mean = Client_repay[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'bureau_DAYS_CREDIT_max', 'bureau_DAYS_CREDIT_ENDDATE_max', 'DAYS_ID_PUBLISH']]
Client_repay_mean= Client_repay_mean.mean().round(2)
Client_repay_mean= pandas.DataFrame(Client_repay_mean, columns = ['repay'])
print(Client_repay_mean)

Client_mean= Client_repay_mean.join(Client_no_repay_mean, how='left')

# Compare 7 most correlated variables with TARGET
Client_mean['results'] = np.where((Client_mean['repay'] <= Client_mean['no repay']), 'higer','lower')

