import pandas as pd
import numpy as np
import datetime as dt
import os
from sklearn.preprocessing import MinMaxScaler
import ast

def replace_nan_in_nan_vector(x):
  output = x
  for e in x:
    if np.isnan(e):
      output = np.nan
  return output

def preprocess_data_price(price_path):
  df_price = pd.read_excel(price_path).rename(columns = {'Date':'date'})
  selected_columns = ['Iron ore Ocean freight (Australia to China)',
        'Crude oil (USD/bbl)', 'Domestics price (SM)', 'Billet from Russia/CIS',
        'HMS Bulk to Turkey', 'HMS Bulk to South Korea', 'Container Taiwan',
        'Japanese H2  export price (Yen)', 'Iron ore',]

  first_date = df_price['date'].min()
  end_date = df_price['date'].max() + dt.timedelta(days=7)
  pdate_range = pd.date_range(first_date, end_date)

  df_price_filled = pd.DataFrame()
  df_price_filled['date'] = pdate_range.copy()
  df_price_filled = pd.merge(df_price_filled, df_price[['date']+selected_columns], on=['date'], how='left')

  for col in selected_columns:
      df_price_filled[col] = df_price_filled[col].fillna(method = 'ffill')

  first_date = df_price['date'].min()
  end_date = df_price['date'].max()+ dt.timedelta(days=7)
  pdate_range = pd.date_range(first_date, end_date, freq='7d')

  df_price_filled = pd.DataFrame(df_price_filled[df_price_filled['date'].isin(pdate_range) == True]).reset_index(drop=True)

  df_price = df_price_filled.copy()

  df_price['date'] = df_price['date'] - dt.timedelta(days=2)

  return df_price

def read_steel_lme(steel_lme_path):
  df_steel = pd.read_csv(steel_lme_path).rename(columns = {'Date':'date', 'Close':'steel_scrap_lme'})
  df_steel['date'] = pd.to_datetime(df_steel['date'])
  df_steel = pd.DataFrame(df_steel[['date', 'steel_scrap_lme']])

  return df_steel

def read_yahoo(yahoo_path, yahoo_files):
  df_filled = pd.DataFrame()
  yahoo_file_paths = {yahoo_file:os.path.join(yahoo_path, yahoo_file) for yahoo_file in yahoo_files}
  for i, path in enumerate(list(yahoo_file_paths.values())):
    df_ = pd.read_csv(path)
    df_['date'] = pd.to_datetime(df_['Date'])
    if i == 0:
      date_min_ = df_['date'].min()
      date_max_ = df_['date'].max()
    else:
      if date_min_ > df_['date'].min():
        date_min_ = df_['date'].min()
      if date_max_ < df_['date'].max():
        date_max_ = df_['date'].max()

  df_filled['date'] = pd.date_range(date_min_, date_max_)

  for col_name, path in yahoo_file_paths.items():
    df_ = pd.read_csv(path)
    df_['date'] = pd.to_datetime(df_['Date'])
    df_ = pd.DataFrame(df_[['date', 'Close']])
    col_name = col_name.replace('.csv', '')
    df_ = df_.rename(columns = {'Close':col_name})
    df_filled = pd.merge(df_filled, df_, on = ['date'], how='left')
    df_filled[col_name] = df_filled[col_name].fillna(method='ffill')

  return df_filled

class TargetMinMaxScaler:

  def __init__(self):
    self.min_ = 0
    self.max_ = 100

  def fit(self, y):
    self.min_ = np.min(y)
    self.max_ = np.max(y)

  def transform(self, y):
    return (y - self.min_)/(self.max_ - self.min_)

  def inverse_transform(self, y):
    return y*(self.max_ - self.min_) + self.min_
  

def make_weight_avg(df, selected_cols, date_max, name = 'adjusted_avg_factors'):
  df_copy = df.copy()
  df_copy = pd.DataFrame(df_copy[df_copy['target_date_1'] < date_max]).reset_index()
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

def windowlized(df, cols, val_date, test_date, valid_target_date, window, n_seq):

  valid_date = df[df['target'].isna() == False]['target_date_1'].min() - dt.timedelta(window)
  df_valid = pd.DataFrame(df[df['target_date_1'] >= valid_date]).reset_index(drop=True)
  df_valid = pd.DataFrame(df_valid[df_valid['target_date_1'] <= valid_target_date]).reset_index(drop=True)
  df_valid['target'] = df_valid['target']

  X_org = df[cols].values
  scaler_X = MinMaxScaler()
  scaler_X.fit(X_org)

  y_org = np.array(df['target'].dropna().apply(lambda x : np.array(x)).tolist())
  scaler_y = TargetMinMaxScaler()
  scaler_y.fit(y_org)

  train_date = val_date - dt.timedelta(7*1)

  df_valid.loc[df_valid['target'].isna(), 'target'] = df_valid.loc[df_valid['target'].isna()]['target'].apply(lambda x : [9999 for i in range(n_seq)])

  X_valid, y_valid = df_valid[cols].values, np.array(df_valid['target'].dropna().apply(lambda x : np.array(x)).tolist())
  X_valid = scaler_X.transform(X_valid)
  y_valid = scaler_y.transform(y_valid)

  train_idx = df_valid[df_valid['target_date_1'] <= train_date].dropna().index.tolist()
  val_idx = df_valid[(df_valid['target_date_1'] >= val_date)&(df_valid['target_date_1'] < test_date)].index.tolist() 
  test_idx = df_valid[df_valid['target_date_1'] >= test_date].index.tolist() 

  X_train, y_train = window_sliding(X_valid, y_valid, train_idx, window)
  X_val, y_val = window_sliding(X_valid, y_valid, val_idx, window)
  X_test, y_test = window_sliding(X_valid, y_valid, test_idx, window)
  df_train = df_valid[df_valid['target_date_1'] <= train_date].dropna().reset_index(drop=True)
  df_val = df_valid[(df_valid['target_date_1'] >= val_date)&(df_valid['target_date_1'] < test_date)].reset_index(drop=True)
  df_test = df_valid[df_valid['target_date_1'] >= test_date].reset_index(drop=True)

  return  (df_train, df_val, df_test), (X_train, y_train), (X_val, y_val), (X_test, y_test), (scaler_X, scaler_y)

def window_sliding_X(X, idxs, window):
  X_windows = []
  for i in idxs:
    X_t = X[i-window:i].copy()
    X_windows.append(X_t)
    
  X_windows = np.array(X_windows)

  return X_windows

def preprocess_target(target):
  output = np.nan
  try:
    output = ast.literal_eval(target)
    return output
  except ValueError:
    return output