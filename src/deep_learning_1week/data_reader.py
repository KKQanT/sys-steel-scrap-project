import pandas as pd

import os

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