from util import preprocess_data_price, read_yahoo, replace_nan_in_nan_vector
import pandas as pd
import datetime as dt
import numpy as np

if __name__ == "__main__":

    PRICE_PATH = '../../data/sys/Data Price.xlsx'

    YAHOO_PATH = '../../data/yahoo/'
    DOMES_YAHOO_FILES = ['000932.SZ.csv', '601899.SS.csv','AH.BK.csv', '600019.SS.csv', 'MT.csv',]
    STEEL_LME_PATH = '../../data/stooq/SteelScrapLME.csv'

    SAVE_PATH = '../../data/preprocessed/domestic_transformerv1_avgsel_week1_to_4.csv'
    MAX_WINDOW = 7*4*6

    df_price = preprocess_data_price(PRICE_PATH)
    df_yahoo = read_yahoo(YAHOO_PATH, DOMES_YAHOO_FILES)
    df = pd.merge(df_price[['date', 'Domestics price (SM)']], df_yahoo, how='right', on = ['date'])

    min_idx = df[df['Domestics price (SM)'].isna() == False].index.min()
    df = pd.DataFrame(df[df.index >= min_idx - MAX_WINDOW])
    df = df.reset_index(drop=True)

    valid_date = df_price['date'].max()

    df['Domestics price (SM)'] = df['Domestics price (SM)'].fillna(method='ffill')

    df.loc[df['date'] > valid_date, 'Domestics price (SM)'] = np.nan

    df_steel = pd.read_csv(STEEL_LME_PATH).rename(columns = {'Date':'date', 'Close':'steel_scrap_lme'})
    df_steel['date'] = pd.to_datetime(df_steel['date'])
    df_steel = pd.DataFrame(df_steel[['date', 'steel_scrap_lme']])
    df = pd.merge(df, df_steel, on = ['date'], how='left')
    df['steel_scrap_lme'] = df['steel_scrap_lme'].fillna(method='ffill')
    df['steel_scrap_lme'] = df['steel_scrap_lme'].fillna(method='bfill')

    df_main = pd.DataFrame(df[['date','Domestics price (SM)']]).copy()

    for week in range(1, 5):
      df_main[f'target_date_{week}'] = df_main['date'] + dt.timedelta(days = 7*week)
      df_target = pd.DataFrame(df[['date', 'Domestics price (SM)']]).copy().rename(
        columns = {'date':f'target_date_{week}', 'Domestics price (SM)':f'target_{week}'})
      df_main = pd.merge(df_main, df_target, on = [f'target_date_{week}'], how='left')

    df_main['target'] = df_main.apply(lambda x : [x['target_1'], x['target_2'], x['target_3'], x['target_4']], axis=1)
    df_main['target'] = df_main['target'].apply(lambda x : replace_nan_in_nan_vector(x))
    df_main = pd.DataFrame(df_main[['date',  'target_date_1', 'target']])
    df = pd.merge(df, df_main, on = ['date'], how='left')

    df.to_csv(SAVE_PATH, index=False)


