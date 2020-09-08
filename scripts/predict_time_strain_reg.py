"""
BASED ON: predict_time_strain_classXG_SHAP_rock_type.py by mcbeck
"""

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

from sklearn.neural_network import MLPRegressor

np.random.seed(42)


def config():
    """
    CONFIG FUNCTION CONTAINING PARAMETERS AND OTHER MODEL DETAILS.
    """

    conf = {}

    conf['parameters'] = {'colsample_bytree':[0.9], 'alpha':[5], 'lambda':[0.1], 'learning_rate': [0.1],
                  'n_estimators': [50, 100, 200], 'max_depth':[3, 5, 7]}
    #conf['parameters'] = {'colsample_bytree':[0.5], 'alpha':[2], 'lambda':[0.3], 'learning_rate': [0.2],
    #              'n_estimators': [64, 128, 192, 256], 'max_depth':[2, 4, 6, 8]}

    #conf['parameters'] = {'colsample_bytree':[0.7], 'alpha':[3], 'lambda':[0.2], 'learning_rate': [0.08],
    #              'n_estimators': [24, 48, 96, 192], 'max_depth':[3, 5, 7, 9]}

    #conf['parameters'] = {'colsample_bytree':[0.3,0.5,0.7,0.9], 'alpha':[1,3,5], 'lambda':[0.1,0.2,0.3], 'learning_rate': [0.05,0.07,0.1,0.13,0.2],
    #              'n_estimators': [25, 50, 100, 150, 200], 'max_depth':[3, 4, 5, 6, 7, 9]}

    conf['model_details'] = {'objective' : "reg:squarederror",
                            'test_size' : 0.2}

    return conf


def get_experiment_data(folder, filename, experiment, features, target, scale=True):
    """
    IMPORTS STRAIN DATASET FOR THE ROCK EXPERIMENTS FROM FOLDER AND
    CREATES DATAFRAME WITH FORMAT BASED ON TYPE OF TESTING.

    folder: The folder of the data
    filename: string with full name of file or list with string filenames
    experiment: Name of the experiment file             (not needed)
    features: Dataset features contained within the file
    target: String indicating which feature to predict
    scale: Determines if time array should be rescaled
    """

    if isinstance(filename,str):
        DF = pd.read_csv(folder+filename, delim_whitespace=True)
        DF = DF.dropna()

    elif isinstance(filename,list):
        #Scaling cannot be an option if you have multiple datasets
        DF = pd.read_csv(folder+filename[0], delim_whitespace=True)
        DF = DF.dropna()

        scaler = RobustScaler()
        DF[features] = scaler.fit_transform(DF[features].values)
        times = DF[target].values
        times = (times-min(times))/(max(times)-min(times))
        DF[target] = times

        for i in range(1,len(filename)):
            df = pd.read_csv(folder+filename[i], delim_whitespace=True)
            df = df.dropna()

            df[features] = scaler.fit_transform(df[features].values)
            times = df[target].values
            times = (times-min(times))/(max(times)-min(times))
            df[target] = times

            DF = DF.append(df)

    else:
        print("filename variable must be valid type, string or list of strings")

    scaler = RobustScaler()# works better with outliers
    #test other scalers?
    DF[features] = scaler.fit_transform(DF[features].values)


    if scale == True:
        times = DF[target].values
        times = (times-min(times))/(max(times)-min(times))
        DF[target] = times
        return DF, times

    else:
        times = DF[target].values
        return DF, times


def remove_outliers(dataset, features, threshold=0.95):
    """

    """
    
    return


def plot_data_time_evolution(DataFrame, feature, plot_strings):
    """
    PLOTS THE GIVEN FEATURE AGAINST THE NUMBER OF DATAPOINTS OF THE FEATURE
    DataFrame:
    feature:
    plot_strings:
    """
    x = np.linspace(0,len(DataFrame)-1,len(DataFrame))
    y = DataFrame[feature]
    plt.plot(x , y, 'r') #plt.plot(x , y, 'ro')
    plt.ylabel(plot_strings[2])
    plt.xlabel(plot_strings[1])
    plt.title(plot_strings[0])
    plt.show()
    return


def train_xgb_model(dataframe, target, features, config):
    """
    TRAINS AN XGBOOST MODEL ON A (PORTION OF) DATAFRAME
    dataframe: the data to train on
    target: String indicating which feature to predict
    features: Dataset features for the model to train on
    config: Dictionary that contains parameters, test_size and objective
    """
    xgb_model = xgb.XGBRegressor(objective=config['model_details']['objective'])  #regression
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=config['parameters'], cv=10, n_jobs=-1)

    grid_search.fit(dataframe[features], dataframe[target])
    final_model = grid_search.best_estimator_

    return final_model


def other_model(dataframe, target, features):
    """
    SOME OTHER MODEL TO TRAIN AND COMPARE
    """

    NN_model = MLPRegressor(max_iter=500)  #regression
    #grid_search = GridSearchCV(estimator=NN_model, )
    #grid_search.fit(dataframe[features], dataframe[target])
    #final_model = grid_search.best_estimator_
    final_model = NN_model.fit(dataframe[features], dataframe[target])

    return final_model


def test_regression_model(model,features,target, dataset):
    """
    FUNCTION TESTS MODEL ON TEST SET
    model: Trained model to test on test set
    features: Features/Predictors in the dataset
    target: String indicating which feature to predict
    dataset: The data to test on (can send in either train, test or different set)
    """

    predicts = model.predict(dataset[features])
    rmse = np.sqrt(mean_squared_error(dataset[target], predicts))
    r2 = r2_score(dataset[target], predicts)

    scores = [len(dataset[target]), rmse, r2]
    return scores


def represent_model_results(model,dataset,features,target,scores):
    """
    STORES/PLOTS/PRINTS THE RESULTS OF THE MODEL ON TEST SET
    """
        #vals = ['p90', 'p75', 'p50', 'mean', 'p25', 'p10', target'std', 'num', 'sum'] #does this do anything?

        #STRINGS (MAY BE UNNECESSARY)
        #score_str = "exp lim num_train num_test rmse_train r2_train rmse_test r2_test \n";
        #imp_str = "feat_str exp lim feat_num feat_imp \n";
        #shp_str = "feat_str exp lim feat_num mean_shap \n";
        #r2_txt = '../strain_pred_res/predict_strain_comb_'+pred_str+'_micro_accurs_r'+rad+'_XGc'+str(clnum)+feat_str+'.txt'
        #r2_str = "exp1 exp2 time_classes accur rand_chance prec_cl0 rec_cl0 prec_cl1 rec_cl1 prec_cl2 rec_cl2 prec_cl3 rec_cl3\n"
        #imp_txt = accur_txt.replace('r2', 'imp')
        #shap_txt = accur_txt.replace('r2', 'shap')

    shap_vals = shap.TreeExplainer(model).shap_values( dataset[features] ) # train[features]
    shap.summary_plot(shap_vals, dataset[features], plot_type="bar")
    #shap_comb = shap_vals.transpose()


    #f= open(accur_txt, "w")
    #f.write(score_str)
    #f.close()

    #print(imp_str)

    #f= open(imp_txt, "w")
    #f.write(imp_str)
    #f.close()


    #print(shp_str)
    #f= open(shap_txt, "w")
    #f.write(shp_str)
    #f.close()

    return


def plot_sns_score_matrix(score_matrix, train_list, test_list, plot_strings, diagonal=True):
    """
    SEABORN CALLS ORDERED FOR PLOTTING SCORE MATRIX
    Takes the score_matrix and plots it with a title and
    the experiment_list as axes.
    """
    fig, ax = plt.subplots(figsize = (10, 10))
    g = sns.heatmap(score_matrix, annot=True, ax=ax, cmap="viridis", fmt=".2f")
    ax.set_title(plot_strings[0])
    g.set_xticklabels(test_list, rotation=45, horizontalalignment='right')
    g.set_yticklabels(train_list, rotation=45, horizontalalignment='right')
    ax.set_ylabel(plot_strings[2])
    ax.set_xlabel(plot_strings[1])

    if diagonal == True:
        [ax.add_patch(Rectangle((i,i),1,1,fill=False,edgecolor='red',lw=2)) for i in range(len(train_list))]
    plt.show()

    return


def plot_observed_vs_predicted(dataset, target, model, features, plot_strings):
    """
    PLOTS THE OBSERVED TARGET VS THE PREDICTION OF THE MODEL
    dataset:
    target:
    model:
    features:
    plot_strings: list containing strings for title 0, and labels 1, 2
    """
    preds = model.predict(dataset[features])

    plt.plot(dataset[target], preds, 'ko')
    plt.plot([0, max(dataset[target])], [0, max(preds)], 'r-')
    plt.ylabel(plot_strings[2])
    plt.xlabel(plot_strings[1])
    plt.title(plot_strings[0])
    plt.show()


if __name__ == "__main__":
    #____________________________________________________________________________________________#

    #LISTS
    folder = '../strain_data_MS/' #rename experiment_folder -> add result_folder
    experiments = ['FBL01', 'FBL02', 'ETNA01', 'ETNA02', 'MONZ03', 'MONZ04', 'MONZ05', 'WG01', 'WG02', 'WG04', 'GRS02', 'GRS03', 'ANS02', 'ANS03', 'ANS04', 'ANS05']
    num_exps = len(experiments)
    #features = ['dp_p90', 'dp_p75', 'dp_p50', 'dp_mean', 'dp_p25', 'dp_p10', 'dp_std', 'dp_num', 'dp_sum']
    features = ['dn_p90', 'dn_p75', 'dn_p50', 'dn_mean', 'dn_p25', 'dn_p10', 'dn_std', 'dn_num', 'dn_sum', 'dp_p90', 'dp_p75', 'dp_p50', 'dp_mean', 'dp_p25', 'dp_p10', 'dp_std', 'dp_num', 'dp_sum', 'cur_p90', 'cur_p75', 'cur_p50', 'cur_mean', 'cur_p25', 'cur_p10', 'cur_std', 'cur_num', 'cur_sum']
    rad = '5'
    filenames = ['strains_curr_'+experiment+'_g'+rad+'0.txt' for experiment in experiments]
    target = "ep" #"sigD2" #save as strain if ep, stress elif sigD2, else dont save

    conf = config()

    models = []
    r2_train_vector = np.zeros(num_exps)
    r2_test_matrix = np.zeros((num_exps,num_exps))

    #____________________________________________________________________________________________#

    print(f"\nTraining models on {num_exps} experiments, testing on same dataset. \nStoring models for later transfer learning:\n")

    for i in range(num_exps):

        print("Current experiment: %s, Completion: %d%%" %(filenames[i],(100*(i+1)/num_exps)))

        dataframe, timespan = get_experiment_data(folder, filenames[i], experiments[i], features, target)
        df_train, df_test = train_test_split(dataframe, test_size=conf['model_details']['test_size'])

        plot_data_time_evolution(dataframe, features[11], ['Evolution of '+features[11],'Timestep',features[11]]) #This is a dumb way to do it: features[2] if we use the small featurelist

        model = train_xgb_model(df_train, target, features, conf) #other_model(df_train,target,features)
        models.append(model)


        train_scores = test_regression_model(model,features,target,df_train)
        test_scores = test_regression_model(model,features,target,df_test)


        r2_train_vector[i] = train_scores[-1]
        r2_test_matrix[i,i] = test_scores[-1]

        print(f"Train: {train_scores}", f"\nTest: {test_scores}")

        plot_observed_vs_predicted(df_test, target, model, features, ['Testing','Observed','Predicted'])
        represent_model_results(model,df_train,features,target,train_scores)

        plot_data_time_evolution(dataframe, 'ep', ['Evolution of '+'ep','Timestep','ep'])
        plt.plot(np.linspace(0,len(dataframe)-1,len(dataframe)), model.predict(dataframe[features]), 'r')
        plt.show()

    #____________________________________________________________________________________________#

    print("\nPerforming tranfser learning, reprinting r2 scores \nand plotting score martix:\n")
    for i in range(len(models)):
        for j in range(num_exps):
            if j != i:
                dataframe, timespan = get_experiment_data(folder, filenames[j], experiments[j], features, target)
                scores = test_regression_model(models[i], features, target,dataframe)
                r2_test_matrix[i,j] = scores[-1]


        print(f"Train score on experiment {experiments[i]}: r2 = {r2_train_vector[i]}\nTest score: r2 = {r2_test_matrix[i,i]}")

    plot_sns_score_matrix(r2_test_matrix,experiments,experiments,["Test R2 Score", "Tested", "Trained"])

    #____________________________________________________________________________________________#

    print("\nCombining datasets to evaluate effect on model and performance")
    filelist = [ [ filenames[0], filenames[5], filenames[11] ] , [ filenames[1], filenames[3], filenames[10] ], [ filenames[2], filenames[6], filenames[8] ], [ filenames[4], filenames[7], filenames[9] ] ] #THIS IS INEFFICIENT, BUT SUFFICIENT FOR TESTING
    num_train_sets = len(filelist)

    #fbl-etna
    #monz-wg
    #grs-ans

    multi_models = []

    r2_multi_test_matrix = np.zeros((num_train_sets,num_exps))

    for i in range(num_train_sets):
        print(f"\nDatasets used for training: {filelist[i]}")
        dataframe, timespan = get_experiment_data(folder, filelist[i], experiments, features, target)

        df_train, df_test = train_test_split(dataframe, test_size=conf['model_details']['test_size'])

        multi_model = train_xgb_model(df_train, target, features, conf) #other_model(df_train,target,features)
        multi_models.append(multi_model)

        train_scores = test_regression_model(multi_model,features,target,df_train)
        test_scores = test_regression_model(multi_model,features,target,df_test)

        print(f"Train: {train_scores}", f"\nTest: {test_scores}")


        plot_data_time_evolution(dataframe, 'ep', ['Evolution of '+'ep','Timestep','ep'])

        plt.plot(np.linspace(0,len(dataframe)-1,len(dataframe)), multi_model.predict(dataframe[features]), 'r')
        plt.show()

        for j in range(num_exps):
            dataframe, timespan = get_experiment_data(folder, filenames[j], experiments[j], features, target)
            scores = test_regression_model(multi_models[i], features, target,dataframe)
            r2_multi_test_matrix[i,j] = scores[-1]

    plot_sns_score_matrix(r2_multi_test_matrix,filelist,experiments,["Test R2 Score", "Tested", "Trained"], diagonal=False)

    #____________________________________________________________________________________________#
