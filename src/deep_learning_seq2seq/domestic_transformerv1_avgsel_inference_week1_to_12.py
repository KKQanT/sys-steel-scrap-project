import pandas as pd
import datetime as dt
from util import preprocess_target, make_weight_avg, window_sliding_X, TargetMinMaxScaler
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tensorflow_addons.layers import MultiHeadAttention
import tensorflow as tf

if __name__ == '__main__':
    PREP_DATA_PATH = '../../data/preprocessed/domestic_transformerv1_avgsel_week1_to_12.csv'
    BASE_FEATURES = ['adjusted_avg_selected_manualy']
    MODEL_NAME = 'domestic_transformerv1_avgsel_week1_to_12'
    SAVE_MODEL_PATH = '../../model/deep_learning/executing/'
    WINDOW = 168
    SAVE_PREDICTION_PATH = '../../output/'
    PLOT = True

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_val_date.pkl', 'r') as val_date_file:
        val_date = dt.datetime.strptime(val_date_file.read(), "%d-%b-%Y (%H:%M:%S.%f)")
        val_date = pd.to_datetime(val_date)

    df = pd.read_csv(PREP_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_date_1'] = pd.to_datetime(df['target_date_1'])
    df['target'] = df['target'].apply(lambda x : preprocess_target(x))

    selected_manualy = ['000932.SZ', '601899.SS','AH.BK', '600019.SS', 'MT', 'steel_scrap_lme']
    df = make_weight_avg(df, selected_manualy, val_date - dt.timedelta(7*4*3), name='adjusted_avg_selected_manualy')

    valid_date = df[df['target'].isna() == False]['target_date_1'].min() - dt.timedelta(WINDOW)
    df_valid = pd.DataFrame(df[df['target_date_1'] >= valid_date]).reset_index(drop=True)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_scaler_X.pkl', 'rb') as scaler_X_file:
      scaler_X = pickle.load(scaler_X_file)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_scaler_y.pkl', 'rb') as scaler_y_file:
      scaler_y = pickle.load(scaler_y_file)

    X_valid, y_valid = df_valid[BASE_FEATURES].values, df_valid[['target']]
    X_valid = scaler_X.transform(X_valid)

    infer_idx = df_valid[df_valid.index >= WINDOW].index.tolist()
    df_infer = pd.DataFrame(df_valid[df_valid.index.isin(infer_idx)].reset_index(drop=True))
    X_infer = window_sliding_X(X_valid, infer_idx, WINDOW)

    model = tf.keras.models.load_model(SAVE_MODEL_PATH+f'{MODEL_NAME}.h5', custom_objects={'MultiHeadAttention':MultiHeadAttention})
    y_predict = model.predict(X_infer)
    y_predict = scaler_y.inverse_transform(y_predict)
    df_infer['predict'] = y_predict[:,0]

    df_infer.to_csv(SAVE_PREDICTION_PATH + 'test_result/d_DL1_week1_to_12.csv', index=False)

    matplotlib.rc('font', **{'size':10})
    f,ax = plt.subplots(figsize=(10, 4))
    plt.plot(df_infer['date'], df_infer['Domestics price (SM)'], 'x-', color='#16A085', label='actual', linewidth=3)  
    plt.plot(df_infer.iloc[:-1, :]['target_date_1'], df_infer.iloc[:-1, :]['predict'], 'x-', color='#7D3C98', label='predict', linewidth=3)  

    df_extend = pd.DataFrame(
        {
        'target_date':pd.date_range(df_infer['target_date_1'].max()+dt.timedelta(days=7*0) , df_infer['target_date_1'].max() + dt.timedelta(days=7*11), freq="7d"),
        'predict':y_predict[-1][0:] 
            }

        )

    plt.plot(df_extend['target_date'], df_extend['predict'], 'x', color='#E74C3C', markersize=10)  

    plt.legend()

    df_extend['date'] = df_infer['date'].max()
    infer_date = df_infer.loc[df_infer['Domestics price (SM)'].isna() == False]['target_date_1'].max() - dt.timedelta(days=1)
    #infer_date = df_infer.loc[df_infer['Domestics price (SM)'].isna() == False]['target_date_1'].max()
    features = [col for col in df_infer.columns.tolist() if col not in ['date', 'target_date_1', 'target', 'predict']]
    for col in features:
      df_infer = df_infer.rename(columns = {col : f'd_DL1_week1_to_month3_{col}'})
    df_infer.loc[df_infer['target_date_1'] > infer_date, 'd_DL1_week1_to_month3_pred'] = df_infer['predict']
    df_infer.loc[df_infer['target_date_1'] <= infer_date, 'd_DL1_week1_to_month3'] = df_infer['predict']
    df_infer = df_infer.drop(columns=['predict',])

    df_infer.to_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1_to_12.csv', index=False)

    df_extend.to_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1_to_12_extended.csv', index=False)

    if PLOT:
        plt.show()
