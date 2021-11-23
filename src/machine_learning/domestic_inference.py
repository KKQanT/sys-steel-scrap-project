import pandas as pd
import re
import ast
import pickle

import matplotlib
import matplotlib.pyplot as plt

import pandas_ta as pta

from timeseries_features_engineering import *


if __name__ == '__main__':

    matplotlib.rc('font', **{'size':15})
    DOMESTIC_PREP_PATH = '../../data/preprocessed/domestic_prep.csv'
    WINDOW = 24
    SAVE_MODEL_PATH = '../../model/machine_learning/executing/'
    SVAE_FILE_PATH = '../../output/'
    PLOT = False

    df = pd.read_csv(DOMESTIC_PREP_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    df['rsi_DomesticspriceSM'] = pta.rsi(df['DomesticspriceSM'], length = WINDOW)
    df['rsi_DomesticspriceSM'] = df['rsi_DomesticspriceSM'].fillna(method='bfill')
    base_features = df.columns.tolist()[3:]

    PROD_COLUMNS = ['product_BarHR',
        'product_Wire_rod', 'product_Total', 'product_Import', 'product_Export',
        'consump_BarHR1', 'consump_Wire_rod1', 'consump_Total1', 'consump_long',
        'crudsteel_Production', 'crudsteel_Import1', 'crudsteel_Export1',
        'crudsteel_Consumption', 'import_stat_Import_scrap',
        'import_stat_Import_Billet',]

    for col in PROD_COLUMNS:
        df[col] = df[col].fillna(method='bfill')

    with open(SAVE_MODEL_PATH + 'domestic_selected_paired_base_features.pkl', 'rb') as selected_paired_base_features_file:
      selected_paired_base_features = pickle.load(selected_paired_base_features_file)

    with open(SAVE_MODEL_PATH + 'domestic_highest_corr_features.pkl', 'rb') as highest_corr_features_file:
      highest_corr_features = pickle.load(highest_corr_features_file)

    with open(SAVE_MODEL_PATH + 'domestic_scaler.pkl', 'rb') as scaler_file:
      scaler = pickle.load(scaler_file)

    with open(SAVE_MODEL_PATH + 'domestic_pca.pkl', 'rb') as pca_file:
      pca = pickle.load(pca_file)

    with open(SAVE_MODEL_PATH + 'domestic_regression.pkl', 'rb') as model_file:
      model = pickle.load(model_file)

    for paired_col in selected_paired_base_features:
        col1, col2 = ast.literal_eval(paired_col)
        df[str([col1, col2])] = df[col1]*df[col2]

    featuresEngClass = [   
        GenMean(),
        GenLag(2,12),
        GenAutocorr(4),
        GenMax(),
        GenMin(),
        GenNumberCrossingCurr(),
        GenNumberCrossingMean(),
        GenPermutationEntropy(1, 6),
        GenPermutationEntropy(1, 4),
        GenPermutationEntropy(1, 3),
        GenPermutationEntropy(1, 2),
        GenReoccuring(),
        GenSampleEntropy(4),
        GenCountAboveMean(),
        GenSlope(),
        GenMeanChange(),
        GenAbsMeanChange(),
        GenStd(),
        GenLongestStrikeAboveMean(),
        GenPartialAutoCorr(4),
        GenCurrToMin(),
        GenCurrToMax(),
        GenMeanRatioChange()
    ]

    df_ = df.copy()
    df_['time_idx'] = df_.index.copy()

    window_time_feature = WindowTimeFeatures(df_, selected_paired_base_features, WINDOW, featuresEngClass)
    df_main_features, engineered_features = window_time_feature.gen_features()

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

    df_main_features.to_csv('../../output/test_result/d_ML.csv', index=False)

    f,ax = plt.subplots(figsize=(13, 5))
    plt.plot(df_main_features['target_date'], df_main_features['target'],'x-', color='#138D75', label='actual')
    plt.plot(df_main_features['target_date'], df_main_features['prediction'], 'x-', color='#8E44AD', label='predict' )

    df = pd.merge(df, df_main_features[['target_date', 'prediction']], on = ['target_date'], how='left')
    features = [col for col in df.columns if col not in ['date', 'target_date', 'target', 'prediction']]
    for col in features:
      df = df.rename(columns = {col : f'd_ML_month3_{col}'})
    
    infer_date = df.loc[df['target'].isna() == False]['target_date'].max()
    df.loc[df['target_date'] > infer_date, 'd_ML_month3_pred'] = df['prediction']
    df.loc[df['target_date'] <= infer_date, 'd_ML_month3'] = df['prediction']
    df = df.rename(columns = {'target':'d_target'})
    df = df.drop(columns = ['prediction'])

    df.to_csv(SVAE_FILE_PATH + 'd_ML.csv', index=False)

    if PLOT == True:
      plt.show()
        

