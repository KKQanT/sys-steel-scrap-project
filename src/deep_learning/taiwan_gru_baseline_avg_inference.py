import datetime as dt
import pandas as pd
import pickle
import tensorflow as tf
from configparser import ConfigParser

import matplotlib
import matplotlib.pyplot as plt

from util import make_weight_avg, window_sliding_X

if __name__ == "__main__":
    TAIWAN_PREP_PATH = '../../data/preprocessed/taiwan_gru_baseline_avg_prep.csv'
    MODEL_NAME = 'taiwan_gru_baseline_avg'
    SAVE_MODEL_PATH = '../../model/deep_learning/executing/'
    BASE_FEATURES = ['adjusted_avg_factors']
    #WINDOW = 168
    SAVE_PREDICTION_PATH = '../../output/'
    PLOT = False

    config = ConfigParser()
    config.read('infer_model_config.ini')
    WINDOW = int(config[MODEL_NAME.upper()]['window'])

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_val_date.pkl', 'r') as val_date_file:
        val_date = dt.datetime.strptime(val_date_file.read(), "%d-%b-%Y (%H:%M:%S.%f)")
        val_date = pd.to_datetime(val_date)

    df = pd.read_csv(TAIWAN_PREP_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])

    adjusted_avg_factors = ['^TWII', 'TSM', 'SCHN', 'X']
    df = make_weight_avg(df, adjusted_avg_factors, val_date - dt.timedelta(7*4*3), name='adjusted_avg_factors')

    valid_date = df[df['target'].isna() == False]['target_date'].min() - dt.timedelta(WINDOW)
    df_valid = pd.DataFrame(df[df['target_date'] >= valid_date]).reset_index(drop=True)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_scaler_X.pkl', 'rb') as scaler_X_file:
      scaler_X = pickle.load(scaler_X_file)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_scaler_y.pkl', 'rb') as scaler_y_file:
      scaler_y = pickle.load(scaler_y_file)

    X_valid, y_valid = df_valid[BASE_FEATURES].values, df_valid[['target']]
    X_valid = scaler_X.transform(X_valid)

    infer_idx = df_valid[df_valid.index >= WINDOW].index.tolist()
    df_infer = pd.DataFrame(df_valid[df_valid.index.isin(infer_idx)].reset_index(drop=True))
    X_infer = window_sliding_X(X_valid, infer_idx, WINDOW)

    model = tf.keras.models.load_model(SAVE_MODEL_PATH+f'{MODEL_NAME}.h5')
    y_predict = model.predict(X_infer)
    y_predict = scaler_y.inverse_transform(y_predict)
    df_infer['predict'] = y_predict
    
    df_infer.to_csv(SAVE_PREDICTION_PATH + 'test_result/t_DL2_month3.csv', index=False)

    matplotlib.rc('font', **{'size':30})
    f,ax = plt.subplots(figsize=(40, 10))
    plt.plot(df_infer['target_date'], df_infer['target'], 'x-', color='#16A085', label='actual', linewidth=3)  
    plt.plot(df_infer['target_date'], df_infer['predict'], 'x-', color='#7D3C98', label='predict', linewidth=3)  
    plt.legend()

    features = [col for col in df_infer.columns.tolist() if col not in ['date', 'target_date', 'target', 'predict']]
    for col in features:
      df_infer = df_infer.rename(columns = {col : f't_DL2_month3_{col}'})

    infer_date = df_infer.loc[df_infer['target'].isna() == False]['target_date'].max()
    df_infer.loc[df_infer['target_date'] > infer_date, 't_DL2_month3_pred'] = df_infer['predict']
    df_infer.loc[df_infer['target_date'] <= infer_date, 't_DL2_month3'] = df_infer['predict']
    df_infer = df_infer.drop(columns=['predict', 'target','target_date'])

    df_infer.to_csv(SAVE_PREDICTION_PATH + 't_DL2_month3.csv', index=False)

    if PLOT:
      plt.show()