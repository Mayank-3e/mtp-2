import pandas as pd
import numpy as np
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

def variation_with_co2(df: pd.DataFrame, reg):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    reg.fit(X_train,y_train)

    pts=10
    CO2_pts=np.linspace(0,50,pts)
    def X_pred(root_T:int):
        return sc.transform(pd.DataFrame({
            'B (kg/m3)': np.ones(pts)*350,
            'FA (%)': np.ones(pts)*10.13986,
            'w/b': np.ones(pts)*.49,
            'CO2 (%)': CO2_pts,
            'RH (%)': np.ones(pts)*65,
            'root t': np.ones(pts)*root_T
        }))

    plt.figure(dpi=200)
    plt.plot(CO2_pts,reg.predict(X_pred(2)),'b-*')
    plt.plot(CO2_pts,reg.predict(X_pred(7)),'r-*')
    plt.plot(CO2_pts,reg.predict(X_pred(12)),'y-*')
    plt.plot(CO2_pts,reg.predict(X_pred(17)),'m-*')
    plt.plot(CO2_pts,reg.predict(X_pred(22)),'g-*')
    plt.legend(['t=2 ($\sqrt{days}$)',
                't=7 ($\sqrt{days}$)',
                't=12 ($\sqrt{days}$)',
                't=17 ($\sqrt{days}$)',
                't=22 ($\sqrt{days}$)'],bbox_to_anchor=(0,1),loc='lower left',ncol=3)
    plt.xlabel('CO$_2$ (%)')
    plt.ylabel('Carbonation depth (mm)')

def variation_with_wb(df: pd.DataFrame, reg):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    reg.fit(X_train,y_train)

    pts=10
    wb_pts=np.linspace(.4,.55,pts)
    def X_pred(root_T:int):
        return sc.transform(pd.DataFrame({
            'B (kg/m3)': np.ones(pts)*350,
            'FA (%)': np.ones(pts)*10.13986,
            'w/b': wb_pts,
            'CO2 (%)': np.ones(pts)*6.5,
            'RH (%)': np.ones(pts)*65,
            'root t': np.ones(pts)*root_T
        }))

    plt.figure(dpi=200)
    plt.plot(wb_pts,reg.predict(X_pred(2)),'b-*')
    plt.plot(wb_pts,reg.predict(X_pred(7)),'r-*')
    plt.plot(wb_pts,reg.predict(X_pred(12)),'y-*')
    plt.plot(wb_pts,reg.predict(X_pred(17)),'m-*')
    plt.plot(wb_pts,reg.predict(X_pred(22)),'g-*')
    plt.legend(['t=2 ($\sqrt{days}$)',
                't=7 ($\sqrt{days}$)',
                't=12 ($\sqrt{days}$)',
                't=17 ($\sqrt{days}$)',
                't=22 ($\sqrt{days}$)'],bbox_to_anchor=(0,1),loc='lower left',ncol=3)
    plt.xlabel('w/b')
    plt.ylabel('Carbonation depth (mm)')

def variation_with_FA(df: pd.DataFrame, reg):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    reg.fit(X_train,y_train)

    pts=10
    FA_pts=np.linspace(0,50,pts)
    def X_pred(root_T:int):
        return sc.transform(pd.DataFrame({
            'B (kg/m3)': np.ones(pts)*350,
            'FA (%)': FA_pts,
            'w/b': np.ones(pts)*.49,
            'CO2 (%)': np.ones(pts)*6.5,
            'RH (%)': np.ones(pts)*65,
            'root t': np.ones(pts)*root_T
        }))

    plt.figure(dpi=200)
    plt.plot(FA_pts,reg.predict(X_pred(2)),'b-*')
    plt.plot(FA_pts,reg.predict(X_pred(7)),'r-*')
    plt.plot(FA_pts,reg.predict(X_pred(12)),'y-*')
    plt.plot(FA_pts,reg.predict(X_pred(17)),'m-*')
    plt.plot(FA_pts,reg.predict(X_pred(22)),'g-*')
    plt.legend(['t=2 ($\sqrt{days}$)',
                't=7 ($\sqrt{days}$)',
                't=12 ($\sqrt{days}$)',
                't=17 ($\sqrt{days}$)',
                't=22 ($\sqrt{days}$)'],bbox_to_anchor=(0,1),loc='lower left',ncol=3)
    plt.xlabel('FA (%)')
    plt.ylabel('Carbonation depth (mm)')

def variation_with_RH(df: pd.DataFrame, reg):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    reg.fit(X_train,y_train)

    pts=10
    RH_pts=np.linspace(50,90,pts)
    def X_pred(root_T:int):
        return sc.transform(pd.DataFrame({
            'B (kg/m3)': np.ones(pts)*350,
            'FA (%)': np.ones(pts)*10.13986,
            'w/b': np.ones(pts)*.49,
            'CO2 (%)': np.ones(pts)*6.5,
            'RH (%)': RH_pts,
            'root t': np.ones(pts)*root_T
        }))

    plt.figure(dpi=200)
    plt.plot(RH_pts,reg.predict(X_pred(2)),'b-*')
    plt.plot(RH_pts,reg.predict(X_pred(7)),'r-*')
    plt.plot(RH_pts,reg.predict(X_pred(12)),'y-*')
    plt.plot(RH_pts,reg.predict(X_pred(17)),'m-*')
    plt.plot(RH_pts,reg.predict(X_pred(22)),'g-*')
    plt.legend(['t=2 ($\sqrt{days}$)',
                't=7 ($\sqrt{days}$)',
                't=12 ($\sqrt{days}$)',
                't=17 ($\sqrt{days}$)',
                't=22 ($\sqrt{days}$)'],bbox_to_anchor=(0,1),loc='lower left',ncol=3)
    plt.xlabel('RH (%)')
    plt.ylabel('Carbonation depth (mm)')

def variation_with_T(df: pd.DataFrame, reg):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    reg.fit(X_train,y_train)

    pts=10
    t_pts=np.linspace(0,25,pts)
    def X_pred(fa:int):
        return sc.transform(pd.DataFrame({
            'B (kg/m3)': np.ones(pts)*350,
            'FA (%)': np.ones(pts)*fa,
            'w/b': np.ones(pts)*.49,
            'CO2 (%)': np.ones(pts)*6.5,
            'RH (%)': np.ones(pts)*65,
            'root t': t_pts
        }))

    plt.figure(dpi=200)
    plt.plot(t_pts,reg.predict(X_pred(0)),'b-*')
    plt.plot(t_pts,reg.predict(X_pred(10)),'r-*')
    plt.plot(t_pts,reg.predict(X_pred(20)),'y-*')
    plt.plot(t_pts,reg.predict(X_pred(30)),'m-*')
    plt.plot(t_pts,reg.predict(X_pred(40)),'g-*')
    plt.plot(t_pts,reg.predict(X_pred(50)),'k-*')
    plt.legend(['FA=0%',
                'FA=10%',
                'FA=20%',
                'FA=30%',
                'FA=40%',
                'FA=50%'],bbox_to_anchor=(0,1),loc='lower left',ncol=3)
    plt.xlabel('Carbonation duration ($\sqrt{days}$)')
    plt.ylabel('Carbonation depth (mm)')