import pandas as pd
import datetime as dt

if __name__ == "__main__":
    SAVE_PREDICTION_PATH = '../../output/'

    df_main = pd.read_csv(SAVE_PREDICTION_PATH + 'd_ML.csv')
    df_main['date'] = pd.to_datetime(df_main['date'])
    df_main['target_date'] = pd.to_datetime(df_main['target_date'])

    df = pd.read_csv(SAVE_PREDICTION_PATH + 't_ML.csv').drop(columns =['target_date'])
    df['date'] = pd.to_datetime(df['date'])
    df_main = pd.merge(df_main, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_month3.csv')
    df['date'] = pd.to_datetime(df['date'])

    df_main = pd.merge(df_main, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL2_month3.csv')
    df['date'] = pd.to_datetime(df['date'])

    df_main = pd.merge(df_main, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL3_month3.csv')
    df['date'] = pd.to_datetime(df['date'])

    df_main = pd.merge(df_main, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 't_DL1_month3.csv')
    df['date'] = pd.to_datetime(df['date'])

    df_main = pd.merge(df_main, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 't_DL2_month3.csv')
    df['date'] = pd.to_datetime(df['date'])

    df_main = pd.merge(df_main, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1.csv').drop(columns = ['target'])
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])

    df_f = df.drop(columns = ['target_date', 'd_DL1_week1','d_DL1_week1_pred'])
    df_t = df[['target_date', 'd_DL1_week1','d_DL1_week1_pred']]
    df_main = pd.merge(df_main, df_f, on = ['date'], how='left')
    df_main = pd.merge(df_main, df_t, on = ['target_date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1_to_4.csv').drop(columns = ['target']).rename(columns = {'target_date_1':'target_date'})
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])

    df_f = df.drop(columns = ['target_date', 'd_DL1_week1_to_week4','d_DL1_week1_to_week4_pred'])
    df_t = df[['target_date', 'd_DL1_week1_to_week4','d_DL1_week1_to_week4_pred']]
    df_main = pd.merge(df_main, df_f, on = ['date'], how='left')
    df_main = pd.merge(df_main, df_t, on = ['target_date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1_to_12.csv').drop(columns = ['target']).rename(columns = {'target_date_1':'target_date'})
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])

    df_f = df.drop(columns = ['target_date', 'd_DL1_week1_to_month3','d_DL1_week1_to_month3_pred'])
    df_t = df[['target_date', 'd_DL1_week1_to_month3','d_DL1_week1_to_month3_pred']]
    df_main = pd.merge(df_main, df_f, on = ['date'], how='left')
    df_main = pd.merge(df_main, df_t, on = ['target_date'], how='left')

    d_DL = ['d_DL1_month3', 'd_DL2_month3', 'd_DL3_month3',]
    d_DL_pred = ['d_DL1_month3_pred', 'd_DL2_month3_pred', 'd_DL3_month3_pred',]
    t_DL = ['t_DL1_month3', 't_DL2_month3']
    t_DL_pred = ['t_DL1_month3_pred', 't_DL2_month3_pred', ]

    df_main['d_DLavg_month3'] = df_main[d_DL].mean(axis=1)
    df_main['d_DLavg_month3_pred'] = df_main[d_DL_pred].mean(axis=1)
    df_main['t_DLavg_month3'] = df_main[t_DL].mean(axis=1)
    df_main['t_DLavg_month3_pred'] = df_main[t_DL_pred].mean(axis=1)

    df_melt = pd.melt(df_main, id_vars=['target_date'], value_vars =df_main.columns[2:])
    df_main = pd.merge(df_main[['date','target_date']], df_melt, how='right', on='target_date').sort_values('target_date', ascending=False)

    df_extend = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1_to_4_extended.csv')
    df_extend['date'] = pd.to_datetime(df_extend['date'])
    df_extend['target_date'] = pd.to_datetime(df_extend['target_date'])

    df_extend_filled = pd.DataFrame({'target_date':pd.date_range(df_extend['target_date'].min(), df_extend['target_date'].max())})
    df_extend_filled = pd.merge(df_extend_filled, df_extend, on = ['target_date'], how='left')
    df_extend_filled['predict'] = df_extend_filled['predict'].fillna(method='ffill')
    df_extend_filled['date'] = df_extend_filled['date'].fillna(method='ffill')

    df_extend_filled = df_extend_filled[df_extend_filled['target_date'].isin(df_main['target_date'])]

    df_extend_filled = pd.melt(df_extend_filled, id_vars=['date', 'target_date'])
    df_extend_filled['variable'] = 'd_DL1_week1_to_week4_pred'

    df_main = df_main.append(df_extend_filled, ignore_index=True)

    df_extend = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1_to_12_extended.csv')
    df_extend['date'] = pd.to_datetime(df_extend['date'])
    df_extend['target_date'] = pd.to_datetime(df_extend['target_date'])

    df_extend_filled = pd.DataFrame({'target_date':pd.date_range(df_extend['target_date'].min(), df_extend['target_date'].max())})
    df_extend_filled = pd.merge(df_extend_filled, df_extend, on = ['target_date'], how='left')
    df_extend_filled['predict'] = df_extend_filled['predict'].fillna(method='ffill')
    df_extend_filled['date'] = df_extend_filled['date'].fillna(method='ffill')

    df_extend_filled = df_extend_filled[df_extend_filled['target_date'].isin(df_main['target_date'])]
    
    df_extend_filled = pd.melt(df_extend_filled, id_vars=['date', 'target_date'])
    df_extend_filled['variable'] = 'd_DL1_week1_to_month3_pred'
    
    df_main = df_main.append(df_extend_filled, ignore_index=True)

    #df_main = pd.DataFrame(df_main[df_main['date'] < dt.datetime.now()]).reset_index(drop=True)

    rename_dict = {'000333.SZ':'Midea_Group', '000932.SZ':'Hunan_Valin_Steel', 
    '601899.SS': 'Zijin_Mining', '600019.SS': 'Baoshan_Iron&Steel', 
    'AH.BK':'AAPICO_Hitech','MT':'ArcelorMittal','^TWII':'TSEC',
    'SCHN':'Schnitzer_Steel', 'TSM':'Taiwan_Semiconductor', 'X':'US_Steel'}

    for key, value in rename_dict.items():
        df_main['variable'] = df_main['variable'].str.replace(key, value)

    df_main.to_csv(SAVE_PREDICTION_PATH + 'all_prediction.csv', index=False)