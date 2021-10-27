import pandas as pd
import numpy as np

from tsfresh.feature_extraction.feature_calculators import (
    linear_trend, 
    autocorrelation, 
    count_above_mean,
    first_location_of_maximum,
    first_location_of_minimum,
    last_location_of_maximum,
    last_location_of_minimum,
    kurtosis,
    maximum,
    minimum,
    mean,
    mean_change,
    mean_abs_change,
    mean_second_derivative_central,
    number_crossing_m,
    ratio_beyond_r_sigma,
    sample_entropy,
    standard_deviation,
    percentage_of_reoccurring_datapoints_to_all_datapoints,
    fourier_entropy,
    longest_strike_above_mean, 
    permutation_entropy, 
    fourier_entropy, 
    partial_autocorrelation
)

from scipy.signal import argrelextrema

class WindowTimeFeatures:
    
    def __init__(self, df_main, base_features, window, featuresEngClass):
        self.df_main = df_main.copy()
        self.base_features = base_features
        self.window = window
        self.featuresEngClass = featuresEngClass
        self.engineered_features = []
        
    def gen_features(self):
        for b_feature in self.base_features:
            df_feature, vec_b_feature = self.create_vector(self.df_main, b_feature, self.window)
            for FeatureClass in self.featuresEngClass:
                self.df_feature = df_feature
                df_feature, new_features = FeatureClass.generate(df_feature, vec_b_feature)
                self.engineered_features.extend(new_features)
            self.df_main = pd.merge(self.df_main, df_feature, on = ['time_idx'], how='left')
        return self.df_main, self.engineered_features
        
    def create_vector(self, df_main, col, win):
        as_strided = np.lib.stride_tricks.as_strided

        v = as_strided(df_main[col], (len(df_main) - (win - 1), win)
                   , (df_main[col].values.strides * 2))

        df_feature = pd.DataFrame(df_main[df_main['time_idx'] >= win - 1]).reset_index(drop=True)
        df_feature[f'vector_{col}'] = v.tolist()
        df_feature = pd.DataFrame(df_feature[['time_idx', f'vector_{col}']])
        return df_feature, f'vector_{col}'
    
def select_engineered_features_corr(df, base_features, featuresEngClass, window, max_date):
    highest_col_features = []
    map_ = {}
    max_date = pd.to_datetime(max_date)
    for col in base_features:
        df_ = pd.DataFrame(df[df['target_date'] < max_date]).reset_index(drop=True)
        df_ = pd.DataFrame(df_[['date', 'target',col]])
        df_['time_idx'] = df_.index.copy()
        window_time_feature = WindowTimeFeatures(df_, [col], window, featuresEngClass)
        df_main_features, engineered_features = window_time_feature.gen_features()
        for engineered_col in engineered_features:
            null_count = df_main_features[engineered_col].isna().sum()
            if null_count >= window:
                print(engineered_col)
                df_main_features = df_main_features.drop(columns = [engineered_col])
        df_main_features=df_main_features.drop(columns = ['time_idx'])
        good_feat = df_main_features.corr()['target'].abs().sort_values(ascending=False).reset_index().dropna()['index'][1]
        highest_col_features.append(good_feat)
        map_[good_feat] = col
    return highest_col_features, map_
    
class GenLag:
    
    def __init__(self, step, nums):
        self.step = step
        self.nums = nums
    
    def generate(self, df_feature, vec_col):
        step = self.step
        nums = self.nums
        N = len(df_feature[vec_col][0])
        list_ = [i for i in reversed(range(0,N,step))]
        list_ = list_[:nums]
        features = []
        for i in list_:
            df_feature[f'{i}th_{vec_col}'] = df_feature[vec_col].apply(lambda x : x[i])
            features.append(f'{i}th_{vec_col}')

        return df_feature, features
    
class GenAutocorr:
    
    def __init__(self, lag):
        self.lag=lag
    
    def generate(self, df_feature, vec_col):
        lag = self.lag
        df_feature[f'{vec_col}_autocorr_{lag}'] = df_feature[vec_col].apply(lambda x : autocorrelation(x, lag=lag))
        return df_feature, [f'{vec_col}_autocorr_{lag}']
    
class GenCountAboveMean:
    
    def generate(self, df_feature, vec_col):
        N = len(df_feature[vec_col][0])
        df_feature[f'cam_{vec_col}'] = df_feature[vec_col].apply(lambda x : count_above_mean(x)) 
        return df_feature, [f'cam_{vec_col}']

    
class GenFirstMaxLoc:

    def generate(self, df_feature, vec_col):
        df_feature[f'first_max_loc_{vec_col}'] = df_feature[vec_col].apply(lambda x : first_location_of_maximum(x))
        return df_feature, [f'first_max_loc_{vec_col}']
    
class GenFirstMinLoc:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'first_min_loc_{vec_col}'] = df_feature[vec_col].apply(lambda x : first_location_of_minimum(x))
        return df_feature, [f'first_min_loc_{vec_col}']
    
class GenKurtosis:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'kurtosis_{vec_col}'] = df_feature[vec_col].apply(lambda x : kurtosis(x))
        return df_feature, [f'kurtosis_{vec_col}']

class GenLastMaxLoc:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'last_max_loc_{vec_col}'] = df_feature[vec_col].apply(lambda x : last_location_of_maximum(x))
        return df_feature, [f'last_max_loc_{vec_col}']

class GenLastMinLoc:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'last_min_loc_{vec_col}'] = df_feature[vec_col].apply(lambda x : last_location_of_minimum(x))
        return df_feature, [f'last_min_loc_{vec_col}']

class GenSlope:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'trend_{vec_col}'] = df_feature[vec_col].apply(lambda x :linear_trend(x, [{"attr": 'slope'}])[0][1])  
        return df_feature, [f'trend_{vec_col}']
    
class GenMax:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'max_{vec_col}'] = df_feature[vec_col].apply(lambda x : maximum(x))
        return df_feature, [f'max_{vec_col}']

class GenMin:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'min_{vec_col}'] = df_feature[vec_col].apply(lambda x : minimum(x))
        return df_feature, [f'min_{vec_col}']

class GenMean:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'mean_{vec_col}'] = df_feature[vec_col].apply(lambda x : mean(x))
        return df_feature, [f'mean_{vec_col}']

class GenMeanChange:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'mean_chng_{vec_col}'] = df_feature[vec_col].apply(lambda x : mean_change(x))
        return df_feature, [f'mean_chng_{vec_col}']

class GenAbsMeanChange:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'mean_abs_change_{vec_col}'] = df_feature[vec_col].apply(lambda x : mean_abs_change(x))
        return df_feature, [f'mean_abs_change_{vec_col}']
    
class GenMeansSecDeriveCentral:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'mean_second_derivative_{vec_col}'] = df_feature[vec_col].apply(lambda x : mean_second_derivative_central(x))
        return df_feature, [f'mean_second_derivative_{vec_col}']
    
class GenNumberCrossingCurr:
    
    def generate(self, df_feature, vec_col):
    
        def _num_cross_curr(x):
            curr_ = x[-1]
            num_cross_curr = number_crossing_m(x, curr_)
            return num_cross_curr

        df_feature[f'number_crossing_curr_{vec_col}'] = df_feature[vec_col].apply(lambda x : _num_cross_curr(x))

        return df_feature, [f'number_crossing_curr_{vec_col}']
    
class GenNumberCrossingMean:
    
    def generate(self, df_feature, vec_col):
        
        def _num_cross_mean(x):
            mean = np.mean(x)
            num_cross_mean = number_crossing_m(x, mean)
            return num_cross_mean

        df_feature[f'number_crossing_mean_{vec_col}'] = df_feature[vec_col].apply(lambda x : _num_cross_mean(x))

        return df_feature, [f'number_crossing_mean_{vec_col}']
    
class GenRatioBeyondSigma:
    
    def __init__(self, r):
        self.r = r
        
    def generate(self, df_feature, vec_col):
        r = self.r
        df_feature[f'ratio_beyond_{r}_sigma_{vec_col}'] = df_feature[vec_col].apply(lambda x : ratio_beyond_r_sigma(x, r))
        return df_feature, [f'ratio_beyond_{r}_sigma_{vec_col}']
    
class GenSampleEntropy:
    
    def __init__(self, max_val):
        self.max_val = max_val
    
    def generate(self, df_feature, vec_col):
        max_val = self.max_val
        df_feature[f'sample_entropy_{vec_col}'] = df_feature[vec_col].apply(lambda x : sample_entropy(x))
        df_feature.loc[df_feature[f'sample_entropy_{vec_col}'] > max_val, f'sample_entropy_{vec_col}'] = max_val
        return df_feature, [f'sample_entropy_{vec_col}']

class GenStd:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'std_{vec_col}'] = df_feature[vec_col].apply(lambda x : standard_deviation(x))
        return df_feature, [f'std_{vec_col}']

class GenReoccuring:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'pct_reoccuring_{vec_col}'] = (
            df_feature[vec_col].apply(lambda x: percentage_of_reoccurring_datapoints_to_all_datapoints(x)))
        return df_feature, [f'pct_reoccuring_{vec_col}']
    
class GenFourierEntropy:
    
    def __init__(self, bins):
        self.bins = bins
    
    def generate(self, df_feature, vec_col):
        bins = self.bins
        df_feature[f'fourier_entropy_{bins}_{vec_col}'] = df_feature[vec_col].apply(lambda x : fourier_entropy(x, bins))
        return df_feature, [f'fourier_entropy_{bins}_{vec_col}']
    
class GenPermutationEntropy:
    
    def __init__(self, tau, dim):
        self.tau = tau
        self.dim = dim
    
    def generate(self, df_feature, vec_col):
        tau = self.tau
        dim = self.dim
        df_feature[f'permutation_entropy_{tau}_{dim}_{vec_col}'] = df_feature[vec_col].apply(lambda x : permutation_entropy(x, tau, dim))
        return df_feature, [f'permutation_entropy_{tau}_{dim}_{vec_col}']
    
class GenLongestStrikeAboveMean:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'longest_strike_above_mean_{vec_col}'] = df_feature[vec_col].apply(lambda x : longest_strike_above_mean(x))
        return df_feature, [f'longest_strike_above_mean_{vec_col}']

class GenPartialAutoCorr:
    
    def __init__(self, lag):
        self.lag=lag
    
    def generate(self, df_feature, vec_col):
        lag = self.lag
        df_feature[f'partial_auto_corr_{lag}_{vec_col}'] = df_feature[vec_col].apply(lambda x : partial_autocorrelation(x, [{'lag':lag}])[0][1])
        return df_feature, [f'partial_auto_corr_{lag}_{vec_col}']

class GenCurrToFirst:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'curr_to_first_{vec_col}'] = df_feature[vec_col].apply(lambda x : x[-1]/(x[0]+0.0001))
        return df_feature, [f'curr_to_first_{vec_col}']
    
class GenCurrToMin:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'curr_to_min_{vec_col}'] = df_feature[vec_col].apply(lambda x : x[-1]/(min(x)+0.0001))
        return df_feature, [f'curr_to_min_{vec_col}']
    
class GenCurrToMax:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'curr_to_max_{vec_col}'] = df_feature[vec_col].apply(lambda x : x[-1]/(max(x)+0.0001))
        return df_feature, [f'curr_to_max_{vec_col}']
    
class GenMeanRatioChange:
    
    def generate(self, df_feature, vec_col):
        df_feature[f'mean_ratio_change_{vec_col}'] = df_feature[vec_col].apply(lambda x:self.genRatioChange(x))
        return df_feature, [f'mean_ratio_change_{vec_col}']
        
    def genRatioChange(self, x):
        y = []
        for i in range(len(x)-1):
            y.append(x[i+1]/(x[i]+0.0001))
        return np.mean(y)
    
class GenDiffLocalMin:
    
    def get_diff_local_mins(self, a):
        a = np.array(a)
        local_mins = argrelextrema(a, np.less)[0]
        if len(local_mins) == 0:
            return (0, 0)
        elif len(local_mins) == 1:
            return (0, 1)
        else:
            return (a[local_mins[1]] - a[local_mins[0]], len(local_mins))
        
    def generate(self, df_feature, vec_col):
        df_feature[f'diff_local_mins_{vec_col}'] = df_feature[vec_col].apply(lambda x : self.get_diff_local_mins(x)[0])
        df_feature[f'num_local_mins_{vec_col}'] = df_feature[vec_col].apply(lambda x : self.get_diff_local_mins(x)[1])
        return df_feature, [f'diff_local_mins_{vec_col}', f'num_local_mins_{vec_col}']

class GenDiffLocalMax:
    
    def get_diff_local_max(self, a):
        a = np.array(a)
        local_max = argrelextrema(a, np.greater)[0]
        if len(local_max) == 0:
            return (0, 0)
        elif len(local_max) == 1:
            return (0, 1)
        else:
            return (a[local_max[1]] - a[local_max[0]], len(local_max))
        
    def generate(self, df_feature, vec_col):
        df_feature[f'diff_local_max_{vec_col}'] = df_feature[vec_col].apply(lambda x : self.get_diff_local_max(x)[0])
        df_feature[f'num_local_max_{vec_col}'] = df_feature[vec_col].apply(lambda x : self.get_diff_local_max(x)[1])
        return df_feature, [f'diff_local_max_{vec_col}', f'num_local_max_{vec_col}']
            
