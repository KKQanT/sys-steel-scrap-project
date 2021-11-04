import numpy as np
import pandas as pd
from selenium.webdriver import Chrome
import os
import time

def to_int(x):
    try:
        y = int(x)
        return y
    except:
        return np.nan

if __name__ == "__main__":
    
    SAVE_PATH = '../../data/stooq/'

    df_old = pd.read_csv(SAVE_PATH + 'SteelScrapLME.csv')
    df_old['Date'] = pd.to_datetime(df_old['Date'])
    recent_date = df_old['Date'].max()
    stop = False

    driver = Chrome()
    driver.get('https://stooq.com/q/d/?s=c-.f')

    time.sleep(np.random.rand()*5)
    while True:
        try:
            df_stooq = pd.DataFrame()
            df = pd.read_html(driver.page_source)
            df_ = df[0]
            df_['No.'] = df_['No.'].apply(lambda x : to_int(x))
            df_ = pd.DataFrame(df_[df_['No.'].isna()==False]).reset_index(drop=True)
            df_stooq = df_stooq.append(df_, ignore_index=True)
            break
        except KeyError:
            time.sleep(1)
            print('solve captcha manually')
            continue

    df_stooq['Date'] = pd.to_datetime(df_stooq['Date'])
    minimum_date = df_stooq['Date'].min()

    if minimum_date < recent_date:
        stop = True

    if stop == True:
        df = pd.concat((df_old, df_stooq), axis=0, ignore_index=True)
        df = df.sort_values('Date', ascending=True).drop_duplicates(keep='first', subset=['Date']).reset_index(drop=True)
        df.to_csv(SAVE_PATH + 'SteelScrapLME.csv', index=False)
    
    for i in range(2, 99999):
        try:
            driver.get(f'https://stooq.com/q/d/?s=c-.f&i=d&l={i}')
            time.sleep(np.random.rand()*5)
            df = pd.read_html(driver.page_source)
            df_ = df[0]
            df_['No.'] = df_['No.'].apply(lambda x : to_int(x))
            df_ = pd.DataFrame(df_[df_['No.'].isna()==False]).reset_index(drop=True)
            df_['Date'] = pd.to_datetime(df_['Date'])
            df_stooq = df_stooq.append(df_, ignore_index=True)
            minimum_date = df_stooq['Date'].min()
            print(minimum_date)
            if minimum_date < recent_date:
                print('reached recent date')
                break
        except KeyError:
            break
    
    df = pd.concat((df_old, df_stooq), axis=0, ignore_index=True)
    df = df.sort_values('Date', ascending=True).drop_duplicates(keep='first', subset=['Date']).reset_index(drop=True)
    df.to_csv(SAVE_PATH + 'SteelScrapLME.csv', index=False)
    driver.close()
    