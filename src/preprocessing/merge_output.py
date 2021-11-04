import pandas as pd

SAVE_PREDICTION_PATH = '../../output/'

if __name__ == "__main__":
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

    df_main.to_csv(SAVE_PREDICTION_PATH + 'all_prediction.csv', index=False)