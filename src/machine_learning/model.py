import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def perform_pcr(df_main_features, features, test_min_date, test_max_date,
                scale=True, pca_var=None):
    
    train_max_date = test_min_date - dt.timedelta(days=7*12)
    
    df_train = pd.DataFrame(df_main_features[df_main_features['target_date'] <= train_max_date]).reset_index(drop=True)
    df_test = pd.DataFrame(df_main_features[
        (df_main_features['target_date'] > test_min_date)&(df_main_features['target_date'] <= test_max_date)
    ]).reset_index(drop=True)
    
    X_train, y_train = df_train[features].copy(), df_train['target'].copy()
    X_test, y_test = df_test[features].copy(), df_test['target'].copy()
    
    if scale:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)   
    if pca_var:
        pca_components = get_pca_component(X_train, pca_var) 
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    test_pred = model.predict(X_test)
    df_test['predict'] = test_pred.copy()
    
    #print(mean_absolute_percentage_error(df_test['target'], df_test['predict']))
    if pca_var:
        return df_train, df_test, model, pca, scaler
    else:
        return df_train, df_test, model

def get_pca_component(X, pca_variance):
    pca = PCA()
    pca.fit(X)
    #X_pca = pca.fit_transform(X)
    #df_pca = pd.DataFrame(X_pca, columns=[f'pc_{i}' for i in range(1,X.shape[1]+1)])
    #
    #selected_pc = []

    for i in range(1,X.shape[1]+1):
        #print(i, np.sum(pca.explained_variance_ratio_[:i]))
        if np.sum(pca.explained_variance_ratio_[:i]) > pca_variance:
            break
    return i 