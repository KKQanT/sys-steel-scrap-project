import pandas as pd
import tensorflow as tf
from validation import get_val_test_date, get_valid_target_date
from util import make_weight_avg, preprocess_target, windowlized
import pickle
import datetime as dt
from modeling import train_model, build_transformerv1_model
from tensorflow_addons.layers import MultiHeadAttention
import configparser
import ast

if __name__ == "__main__":
    PREP_DATA_PATH = '../../data/preprocessed/domestic_transformerv1_avgsel_week1_to_12.csv'
    #SPLIT_PCT = 20
    #WINDOW = 168
    BASE_FEATURES = ['adjusted_avg_selected_manualy']
    MODEL_NAME = 'domestic_transformerv1_avgsel_week1_to_12'
    SAVE_MODEL_PATH = '../../model/deep_learning/experiment/'

    #HEAD_SIZE = 256
    #NUM_HEADS = 4
    #FF_DIM = 4
    #NUM_TRANSFORMER_BLOCKS = 4
    #MLP_UNITS = [32]
    #DROPOUT = 0.2
    #MLP_DROPOUT = 0.4
    #SEED = 5

    config = configparser.ConfigParser()
    config.read('model_config.ini')
    
    SPLIT_PCT = float(config[MODEL_NAME.upper()]['split_pct'])
    SEED = int(config[MODEL_NAME.upper()]['seed'])
    WINDOW = int(config[MODEL_NAME.upper()]['window'])
    DROPOUT = float(config[MODEL_NAME.upper()]['dropout'])
    EPOCHS = int(config[MODEL_NAME.upper()]['epochs'])

    HEAD_SIZE = int(config[MODEL_NAME.upper()]['head_size'])
    NUM_HEADS = int(config[MODEL_NAME.upper()]['num_heads'])
    FF_DIM = int(config[MODEL_NAME.upper()]['ff_dim'])
    NUM_TRANSFORMER_BLOCKS = int(config[MODEL_NAME.upper()]['num_transformer_heads'])
    MLP_UNITS = ast.literal_eval(config[MODEL_NAME.upper()]['mlp_units'])
    MLP_DROPOUT = float(config[MODEL_NAME.upper()]['mlp_dropout'])

    SAVE_BEST_ONLY = False

    df = pd.read_csv(PREP_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_date_1'] = pd.to_datetime(df['target_date_1'])
    df['target'] = df['target'].apply(lambda x : preprocess_target(x))

    val_date, test_date = get_val_test_date(PREP_DATA_PATH, SPLIT_PCT)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_val_date.pkl', 'w') as val_date_file:
      val_date_file.write(val_date.strftime("%d-%b-%Y (%H:%M:%S.%f)"))

    valid_target_date = get_valid_target_date(PREP_DATA_PATH)

    selected_manualy = ['000932.SZ', '601899.SS','AH.BK', '600019.SS', 'MT', 'steel_scrap_lme']
    df = make_weight_avg(df, selected_manualy, val_date - dt.timedelta(7*4*3), name='adjusted_avg_selected_manualy')

    (df_train, df_val, df_test), (X_train, y_train), (X_val, y_val), (X_test, y_test), (scaler_X, scaler_y) = windowlized(df, BASE_FEATURES, val_date, test_date, valid_target_date, WINDOW, n_seq=12)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_scaler_X.pkl', 'wb') as scaler_X_file:
      pickle.dump(scaler_X, scaler_X_file)

    with open(SAVE_MODEL_PATH + f'{MODEL_NAME}_scaler_y.pkl', 'wb') as scaler_y_file:
      pickle.dump(scaler_y, scaler_y_file)

    tf.random.set_seed(SEED)
    model = build_transformerv1_model(
          input_shape = (WINDOW, len(BASE_FEATURES)),
          head_size=HEAD_SIZE,
          num_heads=NUM_HEADS,
          ff_dim=FF_DIM,
          num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
          mlp_units=MLP_UNITS,
          dropout=DROPOUT,
          mlp_dropout=MLP_DROPOUT,
          output_seq=12
      )

    model.summary()

    train_model(X_train, y_train, X_val, y_val, model, MODEL_NAME, epochs=EPOCHS, batch_size=32, save_path=SAVE_MODEL_PATH, save_best_only=SAVE_BEST_ONLY)

    model = tf.keras.models.load_model(SAVE_MODEL_PATH+f'{MODEL_NAME}.h5', custom_objects={'MultiHeadAttention':MultiHeadAttention})

    val_predict = model.predict(X_val)

    test_predict = model.predict(X_test)

    val_predict = scaler_y.inverse_transform(val_predict)
    test_predict = scaler_y.inverse_transform(test_predict)

    df_val['predict'] = val_predict[:,0]
    df_test['predict'] = test_predict[:,0]

    df_val['target'] = df_val['target'].apply(lambda x : x[0])
    df_test['target'] = df_test['target'].apply(lambda x : x[0])

    df_val = df_val.rename(columns={'target_date_1':'target_date'})
    df_test = df_test.rename(columns={'target_date_1':'target_date'})

    df_val.to_csv('../../output/domestic_transformerv1_avgsel_week1_to_12_val.csv', index=False)
    df_test.to_csv('../../output/domestic_transformerv1_avgsel_week1_to_12_test.csv', index=False)

    