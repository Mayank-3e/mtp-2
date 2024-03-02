import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_percentage_error
import numpy as np

def train_val_test_split(X:pd.DataFrame, y:pd.DataFrame, poly_features=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    if(poly_features):
        X_train=poly_features.fit_transform(X_train)
        X_val = poly_features.transform(X_val)
        X_test = poly_features.transform(X_test)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test

def partsMetrics(df: pd.DataFrame, reg, poly_features=None):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X,y,poly_features)
    reg.fit(X_train,y_train)

    print('train rmse:',root_mean_squared_error(y_train,reg.predict(X_train)))
    print('val rmse:',root_mean_squared_error(y_val,reg.predict(X_val)))
    print('test rmse:',root_mean_squared_error(y_test,reg.predict(X_test)))
    print()
    print('train si:',root_mean_squared_error(y_train,reg.predict(X_train))/np.mean(y_train))
    print('val si:',root_mean_squared_error(y_val,reg.predict(X_val))/np.mean(y_val))
    print('test si:',root_mean_squared_error(y_test,reg.predict(X_test))/np.mean(y_test))
    print()
    print('train r2:',r2_score(y_train,reg.predict(X_train)))
    print('val r2:',r2_score(y_val,reg.predict(X_val)))
    print('test r2:',r2_score(y_test,reg.predict(X_test)))
    print()

    # for MAPE
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X,y.replace(0,1e-3),poly_features)
    reg.fit(X_train,y_train)
    print('train mape:',mean_absolute_percentage_error(y_train,reg.predict(X_train)))
    print('val mape:',mean_absolute_percentage_error(y_val,reg.predict(X_val)))
    print('test mape:',mean_absolute_percentage_error(y_test,reg.predict(X_test)))

def allMetrics(df: pd.DataFrame, reg, poly_features=None):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    
    if(poly_features): X=poly_features.fit_transform(X)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    reg.fit(X,y)

    print('all rmse:',root_mean_squared_error(y,reg.predict(X)))
    print('all si:',root_mean_squared_error(y,reg.predict(X))/np.mean(y))
    print('all r2:',r2_score(y,reg.predict(X)))

    y=df.iloc[:,-1].replace(0,1e-3)
    reg.fit(X,y)
    print('all mape:',mean_absolute_percentage_error(y,reg.predict(X)))