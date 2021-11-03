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

    exist_files = os.listdir(os.path.realpath(SAVE_PATH))

    for file in exist_files:
        path_todel = os.path.join(os.path.realpath(SAVE_PATH), file)
        os.remove(path_todel)

    driver = Chrome()
    driver.get('https://stooq.com/q/d/?s=c-.f')

    time.sleep(np.random.rand()*5)
    df_stooq = pd.DataFrame()
    df = pd.read_html(driver.page_source)
    df_ = df[0]
    df_['No.'] = df_['No.'].apply(lambda x : to_int(x))
    df_ = pd.DataFrame(df_[df_['No.'].isna()==False]).reset_index(drop=True)
    df_stooq = df_stooq.append(df_, ignore_index=True)

    for i in range(2, 99999):
        try:
            driver.get(f'https://stooq.com/q/d/?s=c-.f&i=d&l={i}')
            time.sleep(np.random.rand()*5)
            df = pd.read_html(driver.page_source)
            df_ = df[0]
            df_['No.'] = df_['No.'].apply(lambda x : to_int(x))
            df_ = pd.DataFrame(df_[df_['No.'].isna()==False]).reset_index(drop=True)
            df_stooq = df_stooq.append(df_, ignore_index=True)
        except KeyError:
            break

    df_stooq.to_csv(os.path.join(SAVE_PATH, 'SteelScrapLME.csv'), index=False)
    
    driver.close()