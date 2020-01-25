# Part I - reading data
import pandas 
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# Read data frame
train = pandas.read_csv('C:\\podyplomowe\\P\\application_train.csv')
test = pandas.read_csv('C:\\podyplomowe\\P\\application_test.csv')
app_test = pandas.read_csv('C:\\podyplomowe\\P\\application_test.csv')
bureau = pandas.read_csv('C:\\podyplomowe\\P\\bureau.csv')
bureau_balance = pandas.read_csv('C:\\podyplomowe\\P\\bureau_balance.csv')

# Group by the client id (SK_ID_CURR), count the number of previous loans, and rename the column: num_previous_loan
num_previous_loan = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'num_previous_loan'})
num_previous_loan.head()

#%%
# Part II - aggregate statistics for numeric variables
def agg_num(df, group_var, df_name):
     # Remove id variables
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

#%%
# Part III - aggregate statistics for categorical variables
def count_cat(df, group_var, df_name):
# Select the categorical columns
    categorical = pandas.get_dummies(df.select_dtypes('object'))
    categorical[group_var] = df[group_var]
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for v in categorical.columns.levels[0]:
        for s in ['count', 'count_norm']:
            column_names.append('%s_%s_%s' % (df_name, v, s))
    
    categorical.columns = column_names
    
    return categorical

#%%

# Part IV - merging data
bureau_counts = count_cat(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
#bureau_counts.head()
bureau_aggregated = agg_num(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_balance_counts = count_cat(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg = agg_num(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')

# Dataframe grouped by the loan
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

# Merge to include the SK_ID_CURR
bureau_by_loan = bureau_by_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on = 'SK_ID_BUREAU', how = 'left')

# Aggregate the statistics for each client
bureau_balance_by_client = agg_num(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
original = list(train.columns)
print("Original number of features: ", len(original))

# Prepare train data
train = train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
train = train.merge(bureau_aggregated, on = 'SK_ID_CURR', how = 'left')

# Merge with the monthly information grouped by client
train = train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
new_train_features = list(train.columns)
print("Train data frame number of features: ", len(new_train_features))

# Prepare test data
test = test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_aggregated, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

#%%
# Part V - missing values function. If in columns in train or test data is more than 60% missing data, remove those columns
def missing_values(df):
        missing_val = df.isnull().sum()
        
        missing_val_percent = 100 * df.isnull().sum() / len(df)
        # Concat sum and percent of missing values
        missing_table = pandas.concat([missing_val, missing_val_percent], axis=1)
        missing_tabel_names = missing_table.rename(columns = {0 : 'Missing sum', 1 : '% of Missing'})
        return missing_tabel_names
missing_test = missing_values(test)
# For high quantity (above 60%) of missing values - drop those columns
missing_test = missing_test.sort_values(by=['% of Missing'], ascending=False).round()
missing_test_variables = list(missing_test.index[missing_test['% of Missing'] > 60])   
missing_train = missing_values(train)
missing_train = missing_train.sort_values(by=['% of Missing'], ascending=False).round()
missing_train_variables = list(missing_train.index[missing_train['% of Missing'] > 60])


missing_columns = list(set(missing_test_variables + missing_train_variables))
train = train.drop(columns = missing_columns)
test = test.drop(columns = missing_columns)
missing_train_check = missing_values(train)
missing_test_check = missing_values(test)

# For low number of missing values (lower than 60%) - apply the mean for float64 objects and "Other" for string columns
missing_train = missing_values(train)
missing_test = missing_values(test)
low_missing_train_variables = list(missing_train.index[missing_train['% of Missing'] < 60])
low_missing_test_variables = list(missing_test.index[missing_test['% of Missing'] < 60])
low_missing_variables_columns= list(set(low_missing_train_variables + low_missing_test_variables))
low_missing_variables_columns.remove('TARGET')
for col in low_missing_variables_columns:
    if train[col].dtype == "object":
        train[col].fillna("Other", inplace=True)
    elif train[col].dtype == "float64":
        train[col].fillna(train[col].mean(), inplace=True)
    else:
        print("{} has a different dtype.".format(col))
    print("{}:".format(col), train[col].dtype)
    
for col in low_missing_variables_columns:
    if test[col].dtype == "object":
        test[col].fillna("Other", inplace=True)
    elif test[col].dtype == "float64":
        test[col].fillna(test[col].mean(), inplace=True)
    else:
        print("{} has a different dtype.".format(col))
    print("{}:".format(col), test[col].dtype)
    
missing_test_check = missing_values(test)
#%%
# Check if the columns in test and train is the same,remove the 'TARGET' column
train_labels = train['TARGET']
train, test_p = train.align(test, join = 'inner', axis = 1)
train['TARGET'] = train_labels


#%%
# Part VI - calculate correlation between variable [matrix]
corrs = train.corr()
corrs_tar = corrs["TARGET"]
corrs = corrs.sort_values('TARGET', ascending = False)
corrs.head(10)
#%% 
# Part VII - apply thresholds for removing collinear variables
threshold_max = 0.9

# Empty dictionary to hold correlated variables
above_threshold_variables = {}
below_threshold_variables = {}
for col in corrs:
    above_threshold_variables[col] = list(corrs.index[corrs[col] > threshold_max])
# Create empyty list to save data
columns_to_remove = []
columns_seen = []
columns_to_remove_pair = []

# Iterate through columns and correlated columns
for key, value in above_threshold_variables.items():
    columns_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            if x not in columns_seen:
                columns_to_remove.append(x)
                columns_to_remove_pair.append(key)
            
columns_to_remove = list(set(columns_to_remove))
print('Number of columns to remove: ', len(columns_to_remove))  

train_prepared = train.drop(columns = columns_to_remove)
test_prepared = test.drop(columns = columns_to_remove)

print('Training prepared data frame shape: ', train_prepared.shape)
print('Testing parepated data frame shape: ', test_prepared.shape)

#%%
# Part VIII - saving prepared data to csv
train_prepared.to_csv('train_prepared.csv', index = False)
test_prepared.to_csv('test_prepared.csv', index = False)
