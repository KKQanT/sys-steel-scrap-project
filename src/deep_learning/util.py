import glob

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.preprocessing import MinMaxScaler


import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def make_weight_avg(df, selected_cols, date_max, name = 'adjusted_avg_factors'):
  df_copy = df.copy()
  df_copy = pd.DataFrame(df_copy[df_copy['target_date'] < date_max]).reset_index()
  all_min_values = dict(df_copy.min())
  all_max_values = dict(df_copy.max())

  min_values = {k:all_min_values[k] for k in selected_cols}
  max_values = {k:all_max_values[k] for k in selected_cols}

  df_temp = df[selected_cols].copy()
  for col in selected_cols:
    df_temp[col] = (df_temp[col] - min_values[col])/(max_values[col] - min_values[col])
  df_temp['adjusted_avg_factors'] = df_temp[selected_cols].mean(axis=1)
  df[name] = df_temp['adjusted_avg_factors']

  return df

def window_sliding(X, y, idxs, window):
  X_windows = []
  y_windows = []
  for i in idxs:
    X_t = X[i-window:i].copy()
    y_t = y[i].copy()
    X_windows.append(X_t)
    y_windows.append(y_t)
    
  X_windows = np.array(X_windows)
  y_windows = np.array(y_windows)

  return X_windows, y_windows

def windowlized(df, cols, val_date, test_date, valid_target_date, window):

  valid_date = df[df['target'].isna() == False]['target_date'].min() - dt.timedelta(window)
  df_valid = pd.DataFrame(df[df['target_date'] >= valid_date]).reset_index(drop=True)
  df_valid = pd.DataFrame(df_valid[df_valid['target_date'] <= valid_target_date]).reset_index(drop=True)

  X_org = df[cols].values
  scaler_X = MinMaxScaler()
  scaler_X.fit(X_org)

  y_org = df[['target']].values
  scaler_y = MinMaxScaler()
  scaler_y.fit(y_org)

  train_date = val_date - dt.timedelta(7*12)

  X_valid, y_valid = df_valid[cols].values, df_valid[['target']]
  X_valid = scaler_X.transform(X_valid)
  y_valid = scaler_y.transform(y_valid)

  train_idx = df_valid[df_valid['target_date'] <= train_date].dropna().index.tolist()
  val_idx = df_valid[(df_valid['target_date'] >= val_date)&(df_valid['target_date'] < test_date)].index.tolist() 
  test_idx = df_valid[df_valid['target_date'] >= test_date].index.tolist() 

  X_train, y_train = window_sliding(X_valid, y_valid, train_idx, window)
  X_val, y_val = window_sliding(X_valid, y_valid, val_idx, window)
  X_test, y_test = window_sliding(X_valid, y_valid, test_idx, window)

  df_train = df_valid[df_valid['target_date'] <= train_date].dropna().reset_index(drop=True)
  df_val = df_valid[(df_valid['target_date'] >= val_date)&(df_valid['target_date'] < test_date)].reset_index(drop=True)
  df_test = df_valid[df_valid['target_date'] >= test_date].reset_index(drop=True)

  return  (df_train, df_val, df_test), (X_train, y_train), (X_val, y_val), (X_test, y_test), (scaler_X, scaler_y)

def window_sliding_X(X, idxs, window):
  X_windows = []
  for i in idxs:
    X_t = X[i-window:i].copy()
    X_windows.append(X_t)
    
  X_windows = np.array(X_windows)

  return X_windows