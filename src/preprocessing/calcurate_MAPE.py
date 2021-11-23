import pandas as pd
import datetime as dt
import glob
import pickle
import ast

from sklearn.metrics import mean_absolute_percentage_error

if __name__ == "__main__":
    SAVE_TEST_PATH = '../../output/test_result/'
    SAVE_MODEL_PATH = '../../model/deep_learning/executing/'

    df_result = pd.DataFrame()

    for MODEL_NAME, MODEL in [
        ('domestic_transformerv1_avgsel', 'd_DL1'),
        ('domestic_bigru_avg', 'd_DL2'),
        ('domestic_baseline_gru_avg', 'd_DL3'),
        ('taiwan_small_bigru_avgadj2', 't_DL1'),
        ('taiwan_gru_baseline_avg', 't_DL2'),
        ('domestic_transformerv1_avgsel_1week', 'd_DL1_week1')
    ]:
        if MODEL == 'd_DL1_week1':
            df_DL = pd.read_csv(SAVE_TEST_PATH + f'{MODEL}.csv')
        else:
            df_DL = pd.read_csv(SAVE_TEST_PATH + f'{MODEL}_month3.csv')
            
        df_DL['target_date'] = pd.to_datetime(df_DL['target_date'])
        df_DL = df_DL[['target_date', 'target', 'predict']]
        
        with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_val_date.pkl', 'r') as val_date_file:
            val_date = dt.datetime.strptime(val_date_file.read(), "%d-%b-%Y (%H:%M:%S.%f)")
            val_date = pd.to_datetime(val_date)
        df_DL = df_DL[df_DL['target_date'] >= val_date].dropna()
        MAPE = mean_absolute_percentage_error(df_DL['target'], df_DL['predict'])
        
        df_result = df_result.append(
            {'MAPE':MAPE,
            'model':MODEL}, ignore_index=True
        )

    for MODEL_NAME, MODEL in [
        ('domestic_transformerv1_avgsel_week1_to_4', 'd_DL1_week1_to_4'),
        ('domestic_transformerv1_avgsel_week1_to_12', 'd_DL1_week1_to_12')
    ]:
        df_DL = pd.read_csv(SAVE_TEST_PATH+f'{MODEL}.csv').rename(columns = {'target_date_1':'target_date'})
        df_DL['target_date'] = pd.to_datetime(df_DL['target_date'])
        df_DL = df_DL[['target_date', 'target', 'predict']]
        df_DL = df_DL.dropna()
        df_DL['target'] = df_DL['target'].apply(lambda x : ast.literal_eval(x)[0])
        
        with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_val_date.pkl', 'r') as val_date_file:
            val_date = dt.datetime.strptime(val_date_file.read(), "%d-%b-%Y (%H:%M:%S.%f)")
            val_date = pd.to_datetime(val_date)
        
        df_DL = df_DL[df_DL['target_date'] >= val_date].dropna()
        MAPE = mean_absolute_percentage_error(df_DL['target'], df_DL['predict'])
        
        df_result = df_result.append(
            {'MAPE':MAPE,
            'model':MODEL}, ignore_index=True
        )

    df_ML = pd.read_csv(SAVE_TEST_PATH + 'd_ML.csv')
    df_ML['target_date'] = pd.to_datetime(df_ML['target_date'])
    df_ML = df_ML[['target_date', 'target', 'prediction']]

    with open(SAVE_MODEL_PATH + f'domestic_transformerv1_avgsel_val_date.pkl', 'r') as val_date_file:
        val_date = dt.datetime.strptime(val_date_file.read(), "%d-%b-%Y (%H:%M:%S.%f)")
        val_date = pd.to_datetime(val_date)
        
    df_ML = df_ML[df_ML['target_date'] > val_date]
    df_ML = df_ML.dropna()

    MAPE = mean_absolute_percentage_error(df_ML['target'], df_ML['prediction'])

    df_result = df_result.append(
            {'MAPE':MAPE,
            'model':'d_ML'}, ignore_index=True
        )

    df_ML = pd.read_csv(SAVE_TEST_PATH + 't_ML.csv')
    df_ML['target_date'] = pd.to_datetime(df_ML['target_date'])
    df_ML = df_ML[['target_date', 'target', 'prediction']]

    with open(SAVE_MODEL_PATH + f'taiwan_small_bigru_avgadj2_val_date.pkl', 'r') as val_date_file:
        val_date = dt.datetime.strptime(val_date_file.read(), "%d-%b-%Y (%H:%M:%S.%f)")
        val_date = pd.to_datetime(val_date)

    df_ML = df_ML[df_ML['target_date'] > val_date]
    df_ML = df_ML.dropna()

    MAPE = mean_absolute_percentage_error(df_ML['target'], df_ML['prediction'])

    df_result = df_result.append(
            {'MAPE':MAPE,
            'model':'t_ML'}, ignore_index=True
        )

    df_result['MAPE']*=100
    df_result.to_csv('../../output/test_result.csv', index=False)