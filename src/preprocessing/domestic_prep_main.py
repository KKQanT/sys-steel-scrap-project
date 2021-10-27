import pandas as pd
import datetime as dt

from util import preprocess_data_price, preprocess_daily_price, preprocess_mprod, load_yahoo

def generate_domestics_target_variable(df_price):
  df = df_price.copy()

  df_target = pd.DataFrame(df_price[['date', 'Domestics price (SM)']]).copy().rename(
      columns = {'date':'target_date', 'Domestics price (SM)':'target'}
  )

  df_main = pd.DataFrame(df[['date','Domestics price (SM)']]).copy()
  df_main['target_date'] = df_main['date'] + dt.timedelta(days = 7*12)
  df_main = pd.merge(df_main, df_target, on = ['target_date'], how='left')

  return df_main

if __name__ == '__main__':

    PRICE_PATH = '../../data/sys/Data Price.xlsx'
    DAILY_PRICE_PATH = '../../data/sys/Daily price assessments.xlsx'
    MPROD_PATH = '../../data/sys/ข้อมูลการผลิตเหล็กของไทย.xlsx'
    DAILY_TEMP_PATH = '../../data/preprocessed/daily_temp.csv'

    YAHOO_PATH = '../../data/yahoo/'

    SAVE_PATH = '../../data/preprocessed/domestic_prep.csv'

    df_price = preprocess_data_price(PRICE_PATH)
    df_main = generate_domestics_target_variable(df_price)
    df_daily_fill = preprocess_daily_price(DAILY_PRICE_PATH, DAILY_TEMP_PATH, df_main)
    df_mprod_fill = preprocess_mprod(MPROD_PATH, df_main)

    df_main = df_main.drop(columns= ['Domestics price (SM)'])
    df = pd.merge(df_main, df_price, on = ['date'], how='left')
    df = pd.merge(df, df_daily_fill, on = ['date'], how='left')
    df = pd.merge(df, df_mprod_fill, on = ['date'], how='left')

    DOMES_ML_STOCK_FILES = ['600019.SS.csv', 'AH.BK.csv', '601899.SS.csv','000932.SZ.csv', '000333.SZ.csv']
    DOMES_ML_STOCK_COLS = ['600019SS','AHBK','601899SS','000932SZ','000333SZ']
    df_yahoo = load_yahoo(df_main, yahoo_path=YAHOO_PATH, files=DOMES_ML_STOCK_FILES)
    df_yahoo['avg_factors'] = df_yahoo[DOMES_ML_STOCK_COLS].mean(axis=1)
    df_yahoo = pd.DataFrame(df_yahoo[['date']+DOMES_ML_STOCK_COLS+['avg_factors']]).copy()

    df = pd.merge(df, df_yahoo, on = ['date'], how='left')

    df.to_csv(SAVE_PATH, index=False)



