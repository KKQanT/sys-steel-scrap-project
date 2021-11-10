import pandas as pd
import datetime as dt

def get_val_test_date(prep_path, pct=20):

  df_target = pd.read_csv(prep_path)
  df_target['date'] = pd.to_datetime(df_target['date'])
  df_target['target_date_1'] = pd.to_datetime(df_target['target_date_1'])

  min_target_date = df_target[df_target['target'].isna() == False]['target_date_1'].min()
  max_target_date = df_target[df_target['target'].isna() == False]['target_date_1'].max()

  len_day = max_target_date - min_target_date

  split = int(pct/100*len_day.days)

  val_date = max_target_date - dt.timedelta(days=split)
  test_date = max_target_date - dt.timedelta(days=split//2)

  return (val_date, test_date)

def get_valid_target_date(prep_path):
  df_target = pd.read_csv(prep_path)
  df_target['target_date_1'] = pd.to_datetime(df_target['target_date_1'])

  valid_target_date = df_target[df_target['target'].isna() == False]['target_date_1'].max()

  return valid_target_date