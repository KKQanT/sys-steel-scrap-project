import pandas as pd
import pickle
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import datetime as dt
import configparser


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L

from validation import get_val_test_date, get_valid_target_date
from util import make_weight_avg, windowlized
from modeling import train_model

if __name__ == "__main__":
    PREP_DATA_PATH = '../../data/preprocessed/domestic_baseline_gru_avg.csv'
    #SPLIT_PCT = 20
    #WINDOW = 168
    MODEL_NAME = 'domestic_baseline_gru_avg'
    BASE_FEATURES = ['adjusted_avg_factors']
    #SEED = 0
    SAVE_MODEL_PATH = '../../model/deep_learning/experiment/'
    #N_UNITS = 2
    #EPOCHS = 50

    config = configparser.ConfigParser()
    config.read('model_config.ini')

    SPLIT_PCT = float(config[MODEL_NAME.upper()['split_pct']])
    SEED = int(config[MODEL_NAME.upper()]['seed'])
    WINDOW = int(config[MODEL_NAME.upper()]['window'])
    N_UNITS = int(config[MODEL_NAME.upper()]['n_units'])
    EPOCHS = int(config[MODEL_NAME.upper()]['epochs'])

    df = pd.read_csv(PREP_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])

    val_date, test_date = get_val_test_date(PREP_DATA_PATH, SPLIT_PCT)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_val_date.pkl', 'w') as val_date_file:
      val_date_file.write(val_date.strftime("%d-%b-%Y (%H:%M:%S.%f)"))
    
    valid_target_date = get_valid_target_date(PREP_DATA_PATH)

    avg_factors = ['000932.SZ', '601899.SS','AH.BK', '600019.SS', '000333.SZ']
    df = make_weight_avg(df, avg_factors, val_date - dt.timedelta(7*4*3), name='adjusted_avg_factors')

    (df_train, df_val, df_test), (X_train, y_train), (X_val, y_val), (X_test, y_test), (scaler_X, scaler_y) = windowlized(df, BASE_FEATURES, val_date, test_date, valid_target_date, WINDOW)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_scaler_X.pkl', 'wb') as scaler_X_file:
      pickle.dump(scaler_X, scaler_X_file)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_scaler_y.pkl', 'wb') as scaler_y_file:
      pickle.dump(scaler_y, scaler_y_file)

    tf.random.set_seed(SEED)
    model = Sequential()
    model.add(L.GRU(N_UNITS, input_shape=(WINDOW, len(BASE_FEATURES),), 
                    return_sequences = False,
                    stateful=False,
                    go_backwards=False))

    model.add(L.Dense(units=1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.01),
          loss='mean_squared_error', )  

    model.summary()

    train_model(X_train, y_train, X_val, y_val, model, MODEL_NAME, epochs=EPOCHS, batch_size=32, save_path=SAVE_MODEL_PATH, 
    save_best_only=True)

    model = tf.keras.models.load_model(SAVE_MODEL_PATH+f'{MODEL_NAME}.h5')

    val_predict = model.predict(X_val)

    test_predict = model.predict(X_test)

    color='#7D3C98'

    val_predict = scaler_y.inverse_transform(val_predict)
    test_predict = scaler_y.inverse_transform(test_predict)
    train_predict = model.predict(X_train)
    train_predict = scaler_y.inverse_transform(train_predict)
    df_train['predict'] = train_predict
    df_val['predict'] = val_predict
    df_test['predict'] = test_predict

    df_val.to_csv('../../output/domestic_baseline_gru_avg_val.csv', index=False)
    df_test.to_csv('../../output/domestic_baseline_gru_avg_test.csv', index=False)

    #val_mape = np.round(mean_absolute_percentage_error(df_val['target'], df_val['predict'])*100, decimals=1)
    #test_mape = np.round(mean_absolute_percentage_error(df_test['target'], df_test['predict'])*100, decimals=1)
#
    #matplotlib.rc('font', **{'size':30})
#
    #f,ax = plt.subplots(figsize=(13, 5))
    #plt.plot(df_train['target_date'], df_train['target'], 'x-', color='#16A085', label='actual', linewidth=3)  
    #plt.plot(df_train['target_date'], df_train['predict'], 'x-', color=color,  linewidth=1, alpha=0.3)
#
    #plt.plot(df_val['target_date'], df_val['target'], 'x-' , color='#16A085', linewidth=3)
    #plt.plot(df_val['target_date'], df_val['predict'], 'x-', color=color,  linewidth=3)
#
    #plt.plot(df_test['target_date'], df_test['target'], 'x-', color='#16A085', linewidth=3)
    #plt.plot(df_test['target_date'], df_test['predict'], 'x-', color=color, label=f'predicted val mape: {val_mape}%, test mape: {test_mape}%', linewidth=3)
    #plt.legend()
    #plt.axvline(val_date, linestyle='dashed', color='#21618C')
    #plt.axvline(test_date, linestyle='dashed', color='#8E44AD')

    #plt.show()
