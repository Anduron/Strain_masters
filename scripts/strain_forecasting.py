import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
import shap
import statistics as stat

np.random.seed(42)

def get_experiment_data(folder, filename, experiment, features, target, scale=True):
    """
    IMPORTS STRAIN DATASET FOR THE ROCK EXPERIMENTS FROM FOLDER AND
    CREATES DATAFRAME WITH FORMAT BASED ON TYPE OF TESTING.

    folder: The folder of the data
    experiment: Name of the experiment file
    features: Dataset features contained within the file
    target: String indicating which feature to predict (poor name)
    scale: Determines if time array should be rescaled
    """

    DF = pd.read_csv(folder+filename, delim_whitespace=True)
    DF = DF.dropna()

    scaler = RobustScaler()# works better with outliers
    DF[features] = scaler.fit_transform(DF[features].values)

    if scale == True:
        times = DF[target].values
        times = (times-min(times))/(max(times)-min(times))
        DF[target] = times
        return DF, times

    else:
        times = DF[target].values
        return DF, times


def autoregression():
    """
    Want model to return a probability forecast based on previous time points
    We take model output at time k-- to determine output at next time. Model will
    learn from its mistakes during the span of the autoregression and eventually
    will predict that the chance of epsilon max is high. Hopefully at the right moment
    We can use a recurrent neural network, a time dependent conv neural network and
    finally test xgboost. There is a chance that we might need to keep the data untouched
    for the models to not learn that the max epsilon is allways 1 (which would render the model
    limited in use). We will use pytorch + sklearn to make models. 
    """

    return


def model_selector():

    return


def xgboost():

    return


def RNN():

    return


if __name__ == "__main__":
    #LISTS
    folder = '../strain_data_MS/' #rename experiment_folder -> add result_folder
    experiments = ['FBL01', 'FBL02', 'ETNA01', 'ETNA02', 'MONZ03', 'MONZ04', 'MONZ05', 'WG01', 'WG02', 'WG04', 'GRS02', 'GRS03', 'ANS02', 'ANS03', 'ANS04', 'ANS05']
    num_exps = len(experiments)
    features = ['dp_p90', 'dp_p75', 'dp_p50', 'dp_mean', 'dp_p25', 'dp_p10', 'dp_std', 'dp_num', 'dp_sum']
    rad = '5'
    filenames = ['strains_curr_'+experiment+'_g'+rad+'0.txt' for experiment in experiments]
    target = "ep" #"sigD2" #save as strain if ep, stress elif sigD2, else dont save
