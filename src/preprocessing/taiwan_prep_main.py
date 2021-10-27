import pandas as pd
import datetime as dt
import re

from util import preprocess_data_price, preprocess_daily_price, preprocess_mprod, load_yahoo

def generate_taiwan_target_variable(df_price):
  
  df_target = pd.DataFrame(df_price[['date', 'Container Taiwan']]).copy().rename(
          columns = {'date':'target_date', 'Container Taiwan':'target'}
      )
  
  df_main = pd.DataFrame(df_price[['date']]).copy()
  df_main['target_date'] = df_main['date'] + dt.timedelta(days = 7*12)
  df_main = pd.merge(df_main, df_target, on = ['target_date'], how='left')

  return df_main

if __name__ == '__main__':

    PRICE_PATH = '../../data/sys/Data Price.xlsx'
    DAILY_PRICE_PATH = '../../data/sys/Daily price assessments.xlsx'
    MPROD_PATH = '../../data/sys/ข้อมูลการผลิตเหล็กของไทย.xlsx'
    DAILY_TEMP_PATH = '../../data/preprocessed/daily_temp.csv'

    YAHOO_PATH = '../../data/yahoo/'

    SAVE_PATH = '../../data/preprocessed/taiwan_prep.csv'

    df_price = preprocess_data_price(PRICE_PATH)
    df_main = generate_taiwan_target_variable(df_price)
    df_daily_fill = preprocess_daily_price(DAILY_PRICE_PATH, DAILY_TEMP_PATH, df_main)
    df_mprod_fill = preprocess_mprod(MPROD_PATH, df_main)

    TAIWAN_ML_STOCK_FILES = ['^TWII.csv', 'TSM.csv','SCHN.csv','X.csv']
    TAIWAN_ML_STOCK_COLS = ['TWII', 'TSM', 'SCHN', 'X']

    df_yahoo = load_yahoo(df_main, YAHOO_PATH, TAIWAN_ML_STOCK_FILES)
    df_yahoo = df_yahoo.rename(columns={'^TWII':'TWII'})
    df_yahoo = pd.DataFrame(df_yahoo[['date']+TAIWAN_ML_STOCK_COLS])
    scale_map = {'TWII':10000, 'TSM':10,  'SCHN':10, 'X':10}
    for col, scale in scale_map.items():
        df_yahoo[col] = df_yahoo[col]/scale
    df_yahoo['avg_econ_factors'] = df_yahoo[TAIWAN_ML_STOCK_COLS].mean(axis=1)

    df = pd.merge(df_main, df_price, on = ['date'], how='left')
    df = pd.merge(df, df_daily_fill, on = ['date'], how='left')
    df = pd.merge(df, df_mprod_fill, on = ['date'], how='left')
    df = pd.merge(df, df_yahoo, on = ['date'], how='left')
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', ' ', x))

    df.to_csv(SAVE_PATH, index=False)
