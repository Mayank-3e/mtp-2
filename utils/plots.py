import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def exp_vs_predict(df: pd.DataFrame, reg, title: str, poly_features=None):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    if(poly_features): X_train=poly_features.fit_transform(X_train)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    reg.fit(X_train,y_train)

    y_pred=y
    if(poly_features): y_pred=reg.predict(sc.transform(poly_features.transform(X)))
    else: y_pred=reg.predict(sc.transform(X))
    
    plt.figure(dpi=200)
    plt.scatter(y,y_pred,edgecolors='black')
    plt.plot(y,y,'r');
    plt.annotate(f'$R^2 = {r2_score(y,y_pred):.3f}$', xy=(0.05, 0.85), xycoords='axes fraction');
    plt.xlabel('Experimental carbonation depth (mm)');
    plt.ylabel('Predicted carbonation depth (mm)');
    plt.title(title);