
import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt

from data_reader import read_yahoo

if __name__ == "__main__":
    DOMESTIC_PREP_PATH = '../../data/preprocessed/domestic_prep.csv'
    YAHOO_PATH = '../../data/yahoo/'
    DOMES_YAHOO_FILES = ['000932.SZ.csv', '601899.SS.csv','AH.BK.csv', '600019.SS.csv','000333.SZ.csv']

    SAVE_PATH = '../../data/preprocessed/domestic_bigru_avg.csv'
    MAX_WINDOW = 7*4*6

    df_target = pd.read_csv(DOMESTIC_PREP_PATH)
    df_target['date'] = pd.to_datetime(df_target['date'])
    df_target['target_date'] = pd.to_datetime(df_target['target_date'])

    df_yahoo = read_yahoo(YAHOO_PATH, DOMES_YAHOO_FILES)

    df = pd.merge(df_target[['date', 'target_date','target','Domestics price (SM)']], df_yahoo, how='right', on = ['date'])

    min_idx = df[df['target'].isna() == False].index.min()
    df = pd.DataFrame(df[df.index >= min_idx - MAX_WINDOW])
    df = df.reset_index(drop=True)

    valid_target_date = df_target[df_target['target'].isna() == False]['target_date'].max()

    df['target'] = df['target'].fillna(method='ffill')
    df['Domestics price (SM)'] = df['Domestics price (SM)'].fillna(method='ffill')
    df['target_date'] = df['date'] + dt.timedelta(days = 7*12)
    df.loc[df['target_date'] > valid_target_date, 'target'] = np.nan

    df.to_csv(SAVE_PATH, index=False)

    f,ax = plt.subplots(figsize=(13,5))
    plt.plot(df['target_date'], df['Domestics price (SM)'])
    plt.plot(df['target_date'], df['target'])
    plt.show()