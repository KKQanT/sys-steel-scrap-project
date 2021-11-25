import pandas as pd
import matplotlib.pyplot as plt

import pickle

from validation import get_val_test_date, walking_forward_fold
from timeseries_features_engineering import *
from features_selection import selected_highest_corr
from model import perform_pcr

import configparser

if __name__ == '__main__':

    TAIWAN_PREP_PATH = '../../data/preprocessed/taiwan_prep.csv'
    #SPLIT_PCT = 20
    #WINDOW = 24
#
    #THRESHOLD = 0.7
    #STD = 0.1
    #VAR = 0.95

    SAVE_MODEL_PATH = '../../model/machine_learning/experiment/'

    config = configparser.ConfigParser()
    config.read('model_config.ini')

    SPLIT_PCT = float(config['taiwan'.upper()]['split_pct'])
    WINDOW = int(config['taiwan'.upper()]['window'])
    THRESHOLD = float(config['taiwan'.upper()]['threshold'])
    STD = float(config['taiwan'.upper()]['std'])
    VAR = float(config['taiwan'.upper()]['var'])


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

    _, external_test_date = get_val_test_date(TAIWAN_PREP_PATH, SPLIT_PCT)

    df_ = pd.DataFrame(df[df['target_date']<=pd.to_datetime(external_test_date)]).reset_index(drop=True)

    #####################

    highest_corr_price_features_map = selected_highest_corr(
        df_, PRICE_COLUMNS, WINDOW, featuresEngClass, THRESHOLD, STD, False
    )

    highest_corr_prod_features_map = selected_highest_corr(
        df_.dropna().reset_index(drop=True), PROD_COLUMNS, WINDOW, featuresEngClass, THRESHOLD, STD, False
    )

    highest_corr_econ_features_map = selected_highest_corr(
        df_, ECON_FEATURES, WINDOW, featuresEngClass, THRESHOLD, STD, False
    )
    #########################

    highest_corr_features_map = {
    **highest_corr_price_features_map,
    **highest_corr_econ_features_map
    }

    highest_corr_features = [v for _,v in highest_corr_features_map.items()]

    df_ = df.copy()
    df_['time_idx'] = df_.index.copy()

    window_time_feature = WindowTimeFeatures(df_, base_features, WINDOW, featuresEngClass)
    df_main_features, engineered_features = window_time_feature.gen_features()

    idx = df_main_features[highest_corr_features].dropna().index
    df_main_features = df_main_features[df_main_features.index.isin(idx)]
    df_main_features = pd.DataFrame(
        df_main_features[['date','target_date','target']+highest_corr_features].reset_index(drop=True)
    )

    #########################

    start_date = pd.to_datetime('2019-01-01')
    end_date = pd.to_datetime(external_test_date)

    test_fold = walking_forward_fold(start_date, end_date)

    df_test_all = pd.DataFrame()
    for fold, (test_min_date, test_max_date) in test_fold.items():
        df_train, df_test, _, _, _ = perform_pcr(df_main_features, highest_corr_features, test_min_date, test_max_date,
                                    scale=True, pca_var=VAR)
        df_test_all = df_test_all.append(df_test,  ignore_index=True)


    #f,ax = plt.subplots(figsize=(40, 10))
    #plt.plot(df['target_date'], df['target'],'x-', color='#138D75', label='actual')
    #plt.plot(df_test_all['target_date'], df_test_all['predict'], 'x-', color='#8E44AD', label='predict' )
    #for _, (fold_date, _) in test_fold.items():
    #    plt.axvline(fold_date, linestyle='dashed', alpha=0.5, color='#D68910')
    #plt.legend()
    #
    if 'Container Taiwan' not in df_test_all.columns.tolist():
        df_test_all = pd.merge(df_test_all, df[['date', 'Container Taiwan']], on=['date'], how='left')
    df_test_all.loc[df_test_all['Container Taiwan'] < df_test_all['target'], 'label'] = 1
    df_test_all.loc[df_test_all['Container Taiwan'] >= df_test_all['target'], 'label'] = 0
    df_test_all.loc[df_test_all['Container Taiwan'] < df_test_all['predict'], 'predicted_label'] = 1
    df_test_all.loc[df_test_all['Container Taiwan'] >= df_test_all['predict'], 'predicted_label'] = 0


    df_train, df_test, model, pca, scaler = perform_pcr(df_main_features, highest_corr_features, 
                                    pd.to_datetime(external_test_date), 
                                    df['target_date'].max(),
                                    scale=True, pca_var=VAR)
    df_test.to_csv('../../output/test_result/t_ML_test.csv', index=False)

    if 'Container Taiwan' not in df_test.columns.tolist():
        df_test = pd.merge(df_test, df[['date', 'Container Taiwan']], on=['date'], how='left')
    df_test.loc[df_test['Container Taiwan'] < df_test['target'], 'label'] = 1
    df_test.loc[df_test['Container Taiwan'] >= df_test['target'], 'label'] = 0
    df_test.loc[df_test['Container Taiwan'] < df_test['predict'], 'predicted_label'] = 1
    df_test.loc[df_test['Container Taiwan'] >= df_test['predict'], 'predicted_label'] = 0

    #test_mape = mean_absolute_percentage_error(df_test['target'], df_test['predict'])
    #test_acc = accuracy_score(df_test['label'], df_test['predicted_label'])
    #test_f1 = f1_score(df_test['label'], df_test['predicted_label'])
    df_test_external = df_test[['target_date','target', 'predict','label','predicted_label']].dropna()

    with open(SAVE_MODEL_PATH + 'taiwan_regression.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open(SAVE_MODEL_PATH + 'taiwan_pca.pkl', 'wb') as pca_file:
        pickle.dump(pca, pca_file)

    with open(SAVE_MODEL_PATH + 'taiwan_highest_corr_features.pkl', 'wb') as highest_corr_features_file:
        pickle.dump(highest_corr_features, highest_corr_features_file)

    with open(SAVE_MODEL_PATH + 'taiwan_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    f,ax = plt.subplots(figsize=(15, 5))
    plt.plot(df['target_date'], df['target'],'x-', color='#138D75', label='actual')
    plt.plot(df_test_all['target_date'], df_test_all['predict'], 'x-', color='#8E44AD', label='predict' )
    plt.plot(df_test['target_date'], df_test['predict'], 'x-', color='#8E44AD', label='predict' )
    plt.axvline(pd.to_datetime(external_test_date), linestyle='dashed', alpha=0.5, color='#D68910')
    plt.title('Taiwan')
    for _, (fold_date, _) in test_fold.items():
        plt.axvline(fold_date, linestyle='dashed', alpha=0.5, color='#D68910')
    plt.axvline(pd.to_datetime(external_test_date), linestyle='dashed', alpha=1, color='red')
    #plt.show()

    df_test_all.to_csv('../../output/taiwan_ml_test_all.csv', index=False)
    df_test.to_csv('../../output/taiwan_ml_test.csv', index=False)
