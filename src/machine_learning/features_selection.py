import pandas as pd
import datetime as dt
import numpy as np

from timeseries_features_engineering import WindowTimeFeatures

def correlationing(df, features, max_date=None):
    if max_date is not None:
        max_date = pd.to_datetime(max_date)
    else:
        max_date = df['date'].max()
    df_ = pd.DataFrame(df[df['date'] <= max_date])
    df_corr = df_[features+['target']].corr()['target'].reset_index()
    df_corr = df_corr.rename(columns = {'index':'feature', 'target':'corr'})
    df_corr['abs_corr'] = df_corr['corr'].abs()
    df_corr = pd.DataFrame(df_corr[df_corr['feature']!='target'])
    df_corr = df_corr.sort_values('abs_corr', ascending=False)
    df_corr = df_corr.reset_index(drop=True)
    return df_corr

def selected_from_corr(df_corr, avg_cutoff, std_cutoff, sort=False, return_many=False):
    selected_feature=[]
    if sort:
        df_corr = df_corr.sort_values('avg_corr', ascending=False)
    df_corr = pd.DataFrame(df_corr[
        (df_corr['avg_corr'] > avg_cutoff)&(df_corr['std_corr'] < std_cutoff)]).reset_index(drop=True)
    if return_many == False:
        if len(df_corr) != 0:
            selected_feature = df_corr['feature'][0]
            return selected_feature
        else:
            return None
    elif return_many == True:
        if len(df_corr) != 0:
            selected_feature = df_corr['feature'].tolist()
            return selected_feature
        else:
            return None
    return selected_feature

def selected_highest_corr(df, base_features, window, featuresEngClass, avg_cutoff, std_cutoff, return_many=False):
    
    highest_corr_features_map = {}
    
    for base_feature in base_features:
        df_ = df.copy()
        df_['time_idx'] = df_.index.copy()

        window_time_feature = WindowTimeFeatures(df_, [base_feature], window, featuresEngClass)
        df_main_features, engineered_features = window_time_feature.gen_features()

        start_date = '2019-01-01'
        start_date = pd.to_datetime(start_date)
        end_date = df['date'].max()
        curr_date = start_date
        while curr_date < end_date:
            if curr_date == start_date:
                df_corr = correlationing(df_main_features, [base_feature]+engineered_features, curr_date)
                df_corr = df_corr.drop(columns = ['abs_corr'])
                df_corr = df_corr.rename(columns = {'corr':f'corr_{dt.datetime.strftime(curr_date, "%Y-%m")}'})
            else:
                df_ = correlationing(df_main_features, [base_feature]+engineered_features, curr_date)
                df_ = df_.drop(columns = ['abs_corr'])
                df_ = df_.rename(columns = {'corr':f'corr_{dt.datetime.strftime(curr_date, "%Y-%m")}'})
                df_corr = pd.merge(df_corr, df_, on = ['feature'], how='left')
            curr_date+=dt.timedelta(days=7*12)
        avg_corr = df_corr.mean(axis=1)
        std_corr = df_corr.std(axis=1)
        df_corr['avg_corr']=np.abs(avg_corr)
        df_corr['std_corr']=std_corr
        df_corr = df_corr.sort_values('avg_corr', ascending=False).reset_index(drop=True)
        
        selected_feature = selected_from_corr(df_corr, avg_cutoff, std_cutoff, return_many=return_many)
        if selected_feature:
            highest_corr_features_map[base_feature] = selected_feature
    return highest_corr_features_map