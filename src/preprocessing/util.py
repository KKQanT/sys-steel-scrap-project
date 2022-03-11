import pandas as pd
import numpy as np
import datetime as dt
import os

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

def cleanedFloat(x):
  try:
    if x == '-':
      return np.nan
    elif x.find('-') > 0:
      x = float(x.split('-')[0])
    else:
      return float(x)
  except ValueError:
    #print(x)
    return float(x.split(',')[0])
  except AttributeError:
    return float(x)

def preprocess_daily_price(daily_price_path, daily_temp_path, df_main):

  df = pd.read_excel(daily_price_path)

  first_row = df.T.reset_index().iloc[1:3].T[1].tolist()
  second_row = df.T.reset_index().iloc[:1].T[0].tolist()
  created_columns = [first_row[i]+ ' ' +second_row[i] for i in range(20)]

  df_prep = pd.DataFrame(df.T.reset_index().iloc[3:,:20].copy())
  df_prep.columns = created_columns
  df_prep = df_prep.reset_index(drop=True)
  df_prep = df_prep.rename(columns = {'Country Commodity':'date'})

  df_prep.to_csv(daily_temp_path, index=False)

  df_daily =  pd.read_csv(daily_temp_path)
  df_daily['date'] = pd.to_datetime(df_daily['date'])
  object_cols = df_daily.select_dtypes('object').columns.tolist()
  for col in object_cols:
      #print(col)
      df_daily[col] = df_daily[col].apply(lambda x : cleanedFloat(x))
  lessNullColumns = df_daily.columns[df_daily.isnull().mean() < 0.5].tolist()
  df_daily = pd.DataFrame(df_daily[lessNullColumns])

  date_min = df_main['date'].min()
  date_max = df_main['date'].max()
  pdate_range = pd.date_range(date_min, date_max)

  daily_price=['China Iron ore, 62% Fe', 'Australia Coking coal',
        'Turkey Ferrous scrap, HMS 1&2 (80:20)', 'Japan Ferrous scrap, HMS 2',
        'China Square billet, 150 mm', 'Ukraine Square billet, 125-150 mm',
        'Turkey Square billet, 125-150 mm', 'Turkey Rebar, 12 mm',
        'Turkey Rebar, 8-32 mm', 'Germany Rebar, 12, 32 mm',
        'China Wire rod, 6.5 mm', 'China HRC, 3-12 mm', 'Germany HRC, base']

  df_daily_fill = pd.DataFrame()
  df_daily_fill['date'] = pdate_range.copy()
  df_daily_fill = pd.merge(df_daily_fill, df_daily, on=['date'], how='left')

  for col in daily_price:
      df_daily_fill[col] = df_daily_fill[col].astype('float')
      df_daily_fill[col] = df_daily_fill[col].fillna(method='ffill')
      df_daily_fill[col] = df_daily_fill[col].fillna(method='bfill')

  return df_daily_fill

def preprocess_mprod(mprod_path, df_main):
  mprod_data = pd.read_excel(mprod_path, header=1)
  cols = ['date']

  tmp_cols = ['_'.join(['product'] + col.split()) for col in mprod_data.columns[1:6]]
  cols += tmp_cols

  tmp_cols = ['_'.join(['consump'] + col.split()) for col in mprod_data.columns[6:9]] + ['consump_long']
  cols += tmp_cols + ['sys_sale', 'pro_sale', 'etc']

  tmp_cols = ['_'.join(['crudsteel'] + col.split()) for col in mprod_data.columns[13:17]]
  cols += tmp_cols + ['null0']

  tmp_cols = ['_'.join(['import_stat'] + col.split()) for col in mprod_data.columns[18:21]]
  cols += tmp_cols + ['null1']

  tmp_cols = ['_'.join(['netcrudst'] + col.split()) for col in mprod_data.columns[22:24]]
  cols += tmp_cols + ['null2', 'null3']

  tmp_cols = ['_'.join(['syssale'] + col.split()) for col in mprod_data.columns[26:32]]
  cols += tmp_cols + ['null4', 'HR+Wire-domsale', 'null5', 'Bar+HR-domsale']

  mprod_data.columns = cols
  mprod_data = mprod_data.iloc[:-2]

  df_mprod = mprod_data.copy()
  df_mprod = df_mprod[['date','product_Bar+HR', 'product_Wire_rod', 'product_Total',
    'product_Import', 'product_Export', 'consump_Bar+HR.1',
    'consump_Wire_rod.1', 'consump_Total.1', 'consump_long',
      'crudsteel_Production', 'crudsteel_Import.1',
    'crudsteel_Export.1', 'crudsteel_Consumption',
      'import_stat_Import_scrap', 'import_stat_Import_Billet',]].copy()

  date_min = df_main['date'].min()
  date_max = df_main['date'].max()
  pdate_range = pd.date_range(date_min, date_max)
  df_mprod_fill = pd.DataFrame()
  df_mprod_fill['date'] = pdate_range.copy()
  df_mprod_fill = pd.merge(df_mprod_fill, df_mprod, on=['date'], how='left')

  mprod_cols = df_mprod_fill.columns.tolist()[1:]
  for col in df_mprod_fill.columns.tolist()[1:]:
      df_mprod_fill[col] = df_mprod_fill[col].astype('float')
      df_mprod_fill[col] = df_mprod_fill[col].fillna(method='ffill')
      df_mprod_fill[col] = df_mprod_fill[col].fillna(method='bfill')

  df_mprod_fill['next_2_week_date'] = df_mprod_fill['date'] + dt.timedelta(days=7*8)
  df_mprod_fill = df_mprod_fill.drop(columns = ['date'])
  df_mprod_fill = df_mprod_fill.rename(columns = {'next_2_week_date':'date'})

  return df_mprod_fill

def convert_float(x):
  try:
    x = float(x)
    return x
  except:
    return np.nan

def prep_df(df, col):
  df['date'] = pd.to_datetime(df['date'])
  first_date = df['date'].min()
  end_date = df['date'].max()
  pdate_range = pd.date_range(first_date, end_date)
  df_dummy = pd.DataFrame({'date':pdate_range})
  df_dummy = pd.merge(df_dummy, df, on = ['date'], how='left')
  df_dummy[col] = df_dummy[col].fillna(method='ffill')
  col = df.columns.tolist()[1]
  df_dummy[col] = df_dummy[col].apply(lambda x : convert_float(x))
  return df_dummy

def load_yahoo(df_main, yahoo_path, files):
  for filename in files:
    col = filename.replace('.csv','').replace('.','')
    file_path = os.path.join(yahoo_path, filename)
    df_ = pd.read_csv(file_path)
    df_ = df_.rename(columns = {'Date':'date', 'Close':col})
    df_ = pd.DataFrame(df_[['date', col]])
    df_['date'] = pd.to_datetime(df_['date'])
    df = prep_df(df_, col)
    df_main = pd.merge(df_main, df, on = ['date'], how='left')
  return df_main