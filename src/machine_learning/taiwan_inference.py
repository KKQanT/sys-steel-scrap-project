import pandas as pd
import datetime as dt
import numpy as np
import re
import ast
import pickle

import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

from timeseries_features_engineering import *


if __name__ == '__main__':

    matplotlib.rc('font', **{'size':15})

    TAIWAN_PREP_PATH = '../../data/preprocessed/taiwan_prep.csv'
    WINDOW = 24
    SAVE_MODEL_PATH = '../../model/machine_learning/executing/'
    SAVE_FILE_PATH = '../../output/'
    PLOT = False

    df = pd.read_csv(TAIWAN_PREP_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])
    base_features = df.columns.tolist()[3:]

    PRICE_COLUMNS = ['Iron ore Ocean freight Australia to China ', 'Crude oil USD bbl ',
        'Domestics price SM ', 'Billet from Russia CIS', 'HMS Bulk to Turkey',
        'HMS Bulk to South Korea', 'Container Taiwan',
        'Japanese H2 export price Yen ', 'Iron ore', 'China Iron ore 62 Fe',
        'Australia Coking coal', 'Turkey Ferrous scrap HMS 1 2 80 20 ',
        'Japan Ferrous scrap HMS 2', 'China Square billet 150 mm',
        'Ukraine Square billet 125 150 mm', 'Turkey Square billet 125 150 mm',
        'Turkey Rebar 12 mm', 'Turkey Rebar 8 32 mm', 'Germany Rebar 12 32 mm',
        'China Wire rod 6 5 mm', 'China HRC 3 12 mm', 'Germany HRC base',]

    PROD_COLUMNS = ['product_Bar HR', 'product_Wire_rod', 'product_Total', 'product_Import',
        'product_Export', 'consump_Bar HR 1', 'consump_Wire_rod 1',
        'consump_Total 1', 'consump_long', 'crudsteel_Production',
        'crudsteel_Import 1', 'crudsteel_Export 1', 'crudsteel_Consumption',
        'import_stat_Import_scrap', 'import_stat_Import_Billet']


    ECON_FEATURES = ['TWII', 'TSM','SCHN', 'X', 'avg_econ_factors']

    for col in PROD_COLUMNS:
        df[col] = df[col].fillna(method = 'ffill')

    featuresEngClass = [   
        GenMean(),
        GenLag(2,12),
        GenAutocorr(4),
        GenMax(),
        GenMin(),
        GenNumberCrossingCurr(),
        GenNumberCrossingMean(),
        GenPermutationEntropy(1, 4),
        GenPermutationEntropy(1, 3),
        GenPermutationEntropy(1, 2),
        GenReoccuring(),
        GenSampleEntropy(2),
    ]

    df_ = df.copy()
    df_['time_idx'] = df_.index.copy()

    window_time_feature = WindowTimeFeatures(df_, base_features, WINDOW, featuresEngClass)
    df_main_features, engineered_features = window_time_feature.gen_features()

    with open(SAVE_MODEL_PATH + 'taiwan_highest_corr_features.pkl', 'rb') as highest_corr_features_file:
        highest_corr_features = pickle.load(highest_corr_features_file)

    with open(SAVE_MODEL_PATH + 'taiwan_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(SAVE_MODEL_PATH + 'taiwan_pca.pkl', 'rb') as pca_file:
        pca = pickle.load(pca_file)

    with open(SAVE_MODEL_PATH + 'taiwan_regression.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    idx = df_main_features[highest_corr_features].dropna().index
    df_main_features = df_main_features[df_main_features.index.isin(idx)]
    df_main_features = pd.DataFrame(
        df_main_features[['date','target_date','target']+highest_corr_features].reset_index(drop=True)
    )

    X_infer = df_main_features[highest_corr_features].copy()
    X_infer = scaler.transform(X_infer)
    X_infer = pca.transform(X_infer)
    prediction = model.predict(X_infer)
    df_main_features['prediction'] = prediction

    f,ax = plt.subplots(figsize=(13, 5))
    plt.plot(df_main_features['target_date'], df_main_features['target'],'x-', color='#138D75', label='actual')
    plt.plot(df_main_features['target_date'], df_main_features['prediction'], 'x-', color='#8E44AD', label='predict' )

    df = pd.merge(df, df_main_features[['target_date', 'prediction']], on = ['target_date'], how='left')
    features = [col for col in df.columns if col not in ['date', 'target_date', 'target', 'prediction']]
    for col in features:
      df = df.rename(columns = {col : f't_ML_month3_{col}'})
    
    infer_date = df.loc[df['target'].isna() == False]['target_date'].max()
    df.loc[df['target_date'] > infer_date, 't_ML_month3_pred'] = df['prediction']
    df.loc[df['target_date'] <= infer_date, 't_ML_month3'] = df['prediction']
    df = df.rename(columns = {'target':'t_target'})
    df = df.drop(columns = ['prediction'])

    df.to_csv(SAVE_FILE_PATH + 't_ML.csv', index=False)
    if PLOT == True:
        plt.show()