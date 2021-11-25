import pandas as pd
import re
import ast
import matplotlib
import matplotlib.pyplot as plt
import pickle

import pandas_ta as pta

from validation import get_val_test_date, walking_forward_fold
from timeseries_features_engineering import *
from features_selection import selected_highest_corr
from model import perform_pcr
import configparser


class NullTestSet(Exception):
    pass

if __name__ == '__main__':

    matplotlib.rc('font', **{'size':15})

    DOMESTIC_PREP_PATH = '../../data/preprocessed/domestic_prep.csv'
    #SPLIT_PCT = 20
    #WINDOW = 24
#
    #THRESHOLD = 0.7
    #STD = 0.1
    #VAR = 0.97

    SAVE_MODEL_PATH = '../../model/machine_learning/experiment/'

    config = configparser.ConfigParser()
    config.read('model_config.ini')

    SPLIT_PCT = float(config['domestic'.upper()]['split_pct'])
    WINDOW = int(config['domestic'.upper()]['window'])
    THRESHOLD = float(config['domestic'.upper()]['threshold'])
    STD = float(config['domestic'.upper()]['std'])
    VAR = float(config['domestic'.upper()]['var'])


    df = pd.read_csv(DOMESTIC_PREP_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    _, external_test_date = get_val_test_date(DOMESTIC_PREP_PATH, SPLIT_PCT)

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

    #####################
    selected_features = {}
    for col1 in base_features:
        
        df_ = pd.DataFrame(df[df['target_date'] < external_test_date])
        paired_features = []
        
        for col2 in base_features:
            df_[str([col1, col2])] = df_[col1]*df_[col2]
            paired_features.append(str([col1, col2]))
            
        df_corr = df_[[col1] + paired_features + ['target']].corr()['target'].abs().sort_values(ascending=False).reset_index()
        df_corr = df_corr[df_corr['index'] != 'target']
        
        selected_features[col1] = df_corr['index'].tolist()[0]

    selected_paired_base_features = list(set([v for _, v in selected_features.items()]))
    selected_paired_base_features = [ast.literal_eval(x) for x in selected_paired_base_features]

    for (col1, col2) in selected_paired_base_features:
        df[str([col1, col2])] = df[[col1, col2]].apply(lambda x : x[col1]*x[col2], axis=1)
    selected_paired_base_features = [str(x) for x in selected_paired_base_features]

    ######################

    df_ = pd.DataFrame(df[df['target_date'] < external_test_date])

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

    highest_corr_features_map = selected_highest_corr(
        df_, selected_paired_base_features, WINDOW, featuresEngClass, THRESHOLD, STD, False
    )

    highest_corr_features = [v for _,v in highest_corr_features_map.items()]

    #########################
    df_ = df.copy()
    df_['time_idx'] = df_.index.copy()

    window_time_feature = WindowTimeFeatures(df_, selected_paired_base_features, WINDOW, featuresEngClass)
    df_main_features, engineered_features = window_time_feature.gen_features()

    idx = df_main_features[highest_corr_features].dropna().index
    df_main_features = df_main_features[df_main_features.index.isin(idx)]
    df_main_features = pd.DataFrame(
        df_main_features[['date','target_date','target']+highest_corr_features].reset_index(drop=True)
    )


    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime(external_test_date)

    test_fold = walking_forward_fold(start_date, end_date, steps=7*12)

    df_test_all = pd.DataFrame()
    for fold, (test_min_date, test_max_date) in test_fold.items():
        try:
            df_train, df_test, _, _, _ = perform_pcr(df_main_features, highest_corr_features, test_min_date, test_max_date,
                                    scale=True, pca_var=VAR)
            df_test_all = df_test_all.append(df_test,  ignore_index=True)
        except NullTestSet:
            continue

    df_train, df_test, model, pca, scaler = perform_pcr(df_main_features, highest_corr_features, 
                                    pd.to_datetime(external_test_date), 
                                    df['target_date'].max(),
                                    scale=True, pca_var=VAR)
    
    df_test.to_csv('../../output/test_result/d_ML_test.csv', index=False)


    with open(SAVE_MODEL_PATH + 'domestic_regression.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open(SAVE_MODEL_PATH + 'domestic_pca.pkl', 'wb') as pca_file:
        pickle.dump(pca, pca_file)

    with open(SAVE_MODEL_PATH + 'domestic_selected_paired_base_features.pkl', 'wb') as selected_paired_base_features_file:
        pickle.dump(selected_paired_base_features, selected_paired_base_features_file)

    with open(SAVE_MODEL_PATH + 'domestic_highest_corr_features.pkl', 'wb') as highest_corr_features_file:
        pickle.dump(highest_corr_features, highest_corr_features_file)

    with open(SAVE_MODEL_PATH + 'domestic_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    ##################
    f,ax = plt.subplots(figsize=(15, 3))
    plt.plot(df['target_date'], df['target'],'x-', color='#138D75', label='actual')
    plt.plot(df_test_all['target_date'], df_test_all['predict'], 'x-', color='#8E44AD', label='predict' )
    for _, (fold_date, _) in test_fold.items():
        plt.axvline(fold_date, linestyle='dashed', alpha=0.5, color='#D68910')
    plt.legend()
    plt.plot(df_test['target_date'], df_test['predict'], 'x-', color='#8E44AD', label='predict' )
    plt.axvline(pd.to_datetime(external_test_date), linestyle='dashed', alpha=1, color='red')
    plt.title('Domestic')
    #plt.show()

    df_test_all.to_csv('../../output/d_ml_test_all.csv', index=False)
    df_test.to_csv('../../output/d_ml_test.csv', index=False)