import pandas as pd
import datetime as dt

if __name__ == "__main__":

    SAVE_PREDICTION_PATH = '../../output/'

    df_exchange = pd.read_excel('../../data/sys/Weekly scrap Price.xlsx').rename(columns = {'Date':'date'})
    df_exchange = pd.DataFrame(df_exchange[['date', 'F/X.1']])
    df_exchange['date'] = pd.to_datetime(df_exchange['date'])

    df_main_ML = pd.read_csv(SAVE_PREDICTION_PATH + 'd_ML.csv')
    df_main_ML['date'] = pd.to_datetime(df_main_ML['date'])
    df_main_ML['target_date'] = pd.to_datetime(df_main_ML['target_date'])

    df = pd.read_csv(SAVE_PREDICTION_PATH + 't_ML.csv').drop(columns =['target_date'])
    df['date'] = pd.to_datetime(df['date'])

    df = pd.merge(df, df_exchange, on = ['date'], how='left')
    df['F/X.1'] = df['F/X.1'].fillna(method = 'ffill')
    df['t_ML_month3_baht'] = df['t_ML_month3']*df['F/X.1'] + 900
    df['t_ML_month3_pred_baht'] = df['t_ML_month3_pred']*df['F/X.1'] + 900

    df_main_ML = pd.merge(df_main_ML, df, on = ['date'], how='left')

    df_main_DL = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_month3.csv')
    df_main_DL['date'] = pd.to_datetime(df_main_DL['date'])
    df_main_DL['target_date'] = df_main_DL['date'] + dt.timedelta(days=7*3*4)

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL2_month3.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_main_DL = pd.merge(df_main_DL, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL3_month3.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_main_DL = pd.merge(df_main_DL, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 't_DL1_month3.csv')
    df['date'] = pd.to_datetime(df['date'])

    df = pd.merge(df, df_exchange, on = ['date'], how='left')
    df['F/X.1'] = df['F/X.1'].fillna(method = 'ffill')
    df['t_DL1_month3_baht'] = df['t_DL1_month3']*df['F/X.1'] + 900
    df['t_DL1_month3_pred_baht'] = df['t_DL1_month3_pred']*df['F/X.1'] + 900

    df_main_DL = pd.merge(df_main_DL, df, on = ['date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 't_DL2_month3.csv')
    df['date'] = pd.to_datetime(df['date'])

    df = pd.merge(df, df_exchange, on = ['date'], how='left')
    df['F/X.1'] = df['F/X.1'].fillna(method = 'ffill')
    df['t_DL2_month3_baht'] = df['t_DL2_month3']*df['F/X.1'] + 900
    df['t_DL2_month3_pred_baht'] = df['t_DL2_month3_pred']*df['F/X.1'] + 900

    df_main_DL = pd.merge(df_main_DL, df, on = ['date'], how='left')

    df_main_DL_filled = pd.DataFrame()
    df_main_DL_filled['date'] = pd.date_range(df_main_ML['date'].min(), df_main_ML['date'].max())
    df_main_DL_filled['target_date'] = pd.date_range(df_main_ML['target_date'].min(), df_main_ML['target_date'].max())
    df_main_DL_filled = pd.merge(df_main_DL_filled, df_main_DL, on = ['date', 'target_date'], how='left')

    for col in [
        'd_DL1_month3_pred',
        'd_DL2_month3_pred',
        'd_DL3_month3_pred',
        't_DL1_month3_pred',
        't_DL2_month3_pred',
        't_DL1_month3_pred_baht',
        't_DL2_month3_pred_baht',
        
    ]:
        df_main_DL_filled[col] = df_main_DL_filled[col].fillna(method='ffill')

    df_main = pd.merge(df_main_ML, df_main_DL_filled, on = ['date', 'target_date'], how='left')

    df_price = pd.read_excel('../../data/sys/Data Price.xlsx')
    df_price['Date'] = pd.to_datetime(df_price['Date'])

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1.csv').drop(columns = ['target','d_DL1_week1_Domestics price (SM)'])
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])
    df_filled = pd.DataFrame()
    df_filled['date'] = pd.date_range(df_price['Date'].min(), df_price['Date'].max())
    df_filled['target_date'] = df_filled['date'] + dt.timedelta(days=7)
    df_filled = pd.merge(df_filled, df, on = ['date', 'target_date'], how='left')
    df_filled['d_DL1_week1_pred'] = df_filled['d_DL1_week1_pred'].fillna(method='ffill')

    df_f = df_filled.drop(columns = ['target_date', 'd_DL1_week1','d_DL1_week1_pred'])
    df_t = df_filled[['target_date', 'd_DL1_week1','d_DL1_week1_pred']]

    df_main = pd.merge(df_main, df_f, on = ['date'], how='left')
    df_main = pd.merge(df_main, df_t, on = ['target_date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1_to_4.csv').drop(columns = ['target','d_DL1_week1_to_week4_Domestics price (SM)'])
    df = df.rename(columns = {'target_date_1':'target_date'})
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])

    df_filled = pd.DataFrame()
    df_filled['date'] = pd.date_range(df_price['Date'].min(), df_price['Date'].max())
    df_filled['target_date'] = df_filled['date'] + dt.timedelta(days=7)
    df_filled = pd.merge(df_filled, df, on = ['date', 'target_date'], how='left')
    df_filled['d_DL1_week1_to_week4_pred'] = df_filled['d_DL1_week1_to_week4_pred'].fillna(method='ffill')

    df_f = df_filled.drop(columns = ['target_date', 'd_DL1_week1_to_week4','d_DL1_week1_to_week4_pred'])
    df_t = df_filled[['target_date', 'd_DL1_week1_to_week4','d_DL1_week1_to_week4_pred']]

    df_main = pd.merge(df_main, df_f, on = ['date'], how='left')
    df_main = pd.merge(df_main, df_t, on = ['target_date'], how='left')

    df = pd.read_csv(SAVE_PREDICTION_PATH + 'd_DL1_week1_to_12.csv').drop(columns = ['target','d_DL1_week1_to_month3_Domestics price (SM)'])
    df = df.rename(columns = {'target_date_1':'target_date'})
    df['date'] = pd.to_datetime(df['date'])
    df['target_date'] = pd.to_datetime(df['target_date'])

    df_filled = pd.DataFrame()
    df_filled['date'] = pd.date_range(df_price['Date'].min(), df_price['Date'].max())
    df_filled['target_date'] = df_filled['date'] + dt.timedelta(days=7)
    df_filled = pd.merge(df_filled, df, on = ['date', 'target_date'], how='left')
    df_filled['d_DL1_week1_to_month3_pred'] = df_filled['d_DL1_week1_to_month3_pred'].fillna(method='ffill')

    df_f = df_filled.drop(columns = ['target_date', 'd_DL1_week1_to_month3','d_DL1_week1_to_month3_pred'])
    df_t = df_filled[['target_date', 'd_DL1_week1_to_month3','d_DL1_week1_to_month3_pred']]

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
    delta = df_price['Date'].max() - df_extend['date'].min()
    df_extend['date'] = df_extend['date'] + delta
    df_extend['target_date'] = df_extend['target_date'] + delta

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
    delta = df_price['Date'].max() - df_extend['date'].min()
    df_extend['date'] = df_extend['date'] + delta
    df_extend['target_date'] = df_extend['target_date'] + delta

    df_extend_filled = pd.DataFrame({'target_date':pd.date_range(df_extend['target_date'].min(), df_extend['target_date'].max())})
    df_extend_filled = pd.merge(df_extend_filled, df_extend, on = ['target_date'], how='left')
    df_extend_filled['predict'] = df_extend_filled['predict'].fillna(method='ffill')
    df_extend_filled['date'] = df_extend_filled['date'].fillna(method='ffill')

    df_extend_filled = df_extend_filled[df_extend_filled['target_date'].isin(df_main['target_date'])]

    df_extend_filled = pd.melt(df_extend_filled, id_vars=['date', 'target_date'])
    df_extend_filled['variable'] = 'd_DL1_week1_to_month3_pred'

    df_main = df_main.append(df_extend_filled, ignore_index=True)

    rename_dict = {'000333.SZ':'Midea_Group', '000932.SZ':'Hunan_Valin_Steel', 
    '601899.SS': 'Zijin_Mining', '600019.SS': 'Baoshan_Iron&Steel', 
    'AH.BK':'AAPICO_Hitech','MT':'ArcelorMittal','^TWII':'TSEC',
    'SCHN':'Schnitzer_Steel', 'TSM':'Taiwan_Semiconductor', 'X':'US_Steel',
              '000333.SZ':'Midea_Group', '000932.SZ':'Hunan_Valin_Steel', 
               '601899.SS': 'Zijin_Mining', '600019.SS': 'Baoshan_Iron&Steel', 
               'AH.BK':'AAPICO_Hitech', '000333SZ':'Midea_Group', '000932SZ':'Hunan_Valin_Steel', 
               '601899SS': 'Zijin_Mining', '600019SS': 'Baoshan_Iron&Steel', 'AHBK':'AAPICO_Hitech',
               'MT':'ArcelorMittal','^TWII':'TSEC','SCHN':'Schnitzer_Steel', 'TSM':'Taiwan_Semiconductor', 'X':'US_Steel'}

    for key, value in rename_dict.items():
        df_main['variable'] = df_main['variable'].str.replace(key, value)

    map_ = {'d_DL1_month3_pred':'d_DL1',
    'd_DL2_month3_pred':'d_DL2',
    'd_DL3_month3_pred':'d_DL3',
    't_DL1_month3_pred':'t_DL1',
    't_DL2_month3_pred':'t_DL2',
    'd_ML_month3_pred':'d_ML',
    't_ML_month3_pred':'t_ML',
            
    'd_DL1_week1_pred':'d_DL1_week1',
    'd_DL1_week1_to_week4_pred':'d_DL1_week1_to_4',
    'd_DL1_week1_to_month3_pred':'d_DL1_week1_to_12',
    
    'd_DL1_month3':'d_DL1',
    'd_DL2_month3':'d_DL2',
    'd_DL3_month3':'d_DL3',
    't_DL1_month3':'t_DL1',
    't_DL2_month3':'t_DL2',
    'd_ML_month3':'d_ML',
    't_ML_month3':'t_ML',
        
    'd_DL1_week1':'d_DL1_week1',
    'd_DL1_week1_to_week4':'d_DL1_week1_to_4',
    'd_DL1_week1_to_month3':'d_DL1_week1_to_12',}

    df_main['model'] = df_main['variable'].map(map_)

    df_result = pd.read_csv(SAVE_PREDICTION_PATH + 'test_result.csv')

    df_main = pd.merge(df_main, df_result, on = ['model'], how='left')

    df_main = df_main.drop_duplicates(subset=['target_date', 'variable'], keep='first')

    df_main.to_csv(SAVE_PREDICTION_PATH + 'all_prediction_current.csv', index=False)