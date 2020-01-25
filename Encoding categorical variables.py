import pandas 
import numpy as np
train = pandas.read_csv('train_prepared.csv')
test = pandas.read_csv('test_prepared.csv')
labels = train['TARGET']
SK_ID_CURR = test['SK_ID_CURR']

#%%
from sklearn.preprocessing import LabelEncoder

def encoding(X_train, Y_test, encoding = 'ohe'): 
    X_train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
    Y_train = train['TARGET']
    X_test = test.drop(columns = ['SK_ID_CURR'])
    # One Hot Encoding
    if encoding == 'ohe':
        X_train = pandas.get_dummies(X_train)
        X_test = pandas.get_dummies(X_test)
        
        # Align the dataframes by the columns
        X_train, X_test = X_train.align(X_test, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for string categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(X_train):
            if X_train[col].dtype == 'object':
                # Map the categorical features to integers
                X_train[col] = label_encoder.fit_transform(np.array(X_train[col].astype(str)).reshape((-1,)))
                X_test[col] = label_encoder.transform(np.array(X_test[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Show error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be 'ohe' or 'le'")

    
train, test = encoding(train, test) 
train.to_csv('X_train.csv', index = False, header = True)
test.to_csv('X_test.csv', index = False, header = True)

   