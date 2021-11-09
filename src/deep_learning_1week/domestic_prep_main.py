import pandas as pd
import matplotlib.pyplot as plt

from util import preprocess_daily_price, preprocess_data_price, preprocess_mprod, generate_domestics_target_variable, load_yahoo

if __name__ == "__main__":
    PRICE_PATH = '../../data/sys/Data Price.xlsx'
    DAILY_PRICE_PATH = '../../data/sys/Daily price assessments.xlsx'
    MPROD_PATH = '../../data/sys/ข้อมูลการผลิตเหล็กของไทย.xlsx'
    DAILY_TEMP_PATH = '../../data/preprocessed/daily_temp.csv'

    YAHOO_PATH = '../../data/yahoo/'

    SAVE_PATH = '../../data/preprocessed/domestic_prep_1week.csv'

    PLOT = False

    df_price = preprocess_data_price(PRICE_PATH)
    df_main = generate_domestics_target_variable(df_price, step=7*1)
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

    f,ax = plt.subplots(figsize = (15, 7))
    plt.plot(df['date'], df['Domestics price (SM)'])
    plt.plot(df['date'], df['target'])

    df.to_csv(SAVE_PATH, index=False)

    if PLOT:
        plt.show()