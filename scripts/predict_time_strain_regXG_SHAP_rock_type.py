#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:51:45 2019

@author: mcbeck
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
import shap
#import statistics
import statistics as stat

folder = '../strain_data_MS/'
np.random.seed(0)

# get importance of features or not
get_imp = 1
# experiments
#exps = ['FBL01', 'FBL02'] #, 'ETNA01', 'ETNA02']
# two rocks per experiment
exps = ['FBL01', 'FBL02', 'ETNA01', 'ETNA02', 'MONZ04', 'MONZ05', 'WG02', 'WG04', 'GRS02', 'GRS03', 'ANS02', 'ANS04']
# all experiments
#exps = ['FBL01', 'FBL02', 'ETNA01', 'ETNA02', 'MONZ04', 'MONZ05', 'WG01', 'WG02', 'WG04', 'GRS02', 'GRS03', 'ANS02', 'ANS03', 'ANS04', 'ANS05']

exp2s = exps.copy()

# strain features
#sigD2 ep x y z p_dn_mean p_dn_sum p_dp_mean p_dp_sum p_cur_mean p_cur_sum binp_dn_mean binp_dn_sum binp_dp_mean binp_dp_sum binp_cur_mean binp_cur_sum dn_p90 dn_p75 dn_p50 dn_mean dn_p25 dn_p10 dn_std dn_num dn_sum dp_p90 dp_p75 dp_p50 dp_mean dp_p25 dp_p10 dp_std dp_num dp_sum cur_p90 cur_p75 cur_p50 cur_mean cur_p25 cur_p10 cur_std cur_num cur_sum
# all the strain components
#features = ['dn_p90', 'dn_p75', 'dn_p50', 'dn_mean', 'dn_p25', 'dn_p10', 'dn_std', 'dn_num', 'dn_sum', 'dp_p90', 'dp_p75', 'dp_p50', 'dp_mean', 'dp_p25', 'dp_p10', 'dp_std', 'dp_num', 'dp_sum', 'cur_p90', 'cur_p75', 'cur_p50', 'cur_mean', 'cur_p25', 'cur_p10', 'cur_std', 'cur_num', 'cur_sum']
#feat_str = ''

# only dilation and shear strain
#features = ['dp_p90', 'dp_p75', 'dp_p50', 'dp_mean', 'dp_p25', 'dp_p10', 'dp_std', 'dp_num', 'dp_sum', 'cur_p90', 'cur_p75', 'cur_p50', 'cur_mean', 'cur_p25', 'cur_p10', 'cur_std', 'cur_num', 'cur_sum']

# only dilation
features = ['dp_p90', 'dp_p75', 'dp_p50', 'dp_mean', 'dp_p25', 'dp_p10', 'dp_std', 'dp_num', 'dp_sum']
feat_str = '_dil'

#        0     1       2      3       4      5      6       7      8
vals = ['p90', 'p75', 'p50', 'mean', 'p25', 'p10', 'std', 'num', 'sum']

# potential predictisons
#pred_str = 'distep'
pred_str = 'distsigD2'

numf = len(features)
lim = 1
score_str = "exp lim num_train num_test rmse_train r2_train rmse_test r2_test \n";
imp_str = "feat_str exp lim feat_num feat_imp \n";
shp_str = "feat_str exp lim feat_num mean_shap \n";

rad = '5'
clnum = 2 # 2 or 4
accur_txt = '../strain_pred_res/predict_strain_comb_'+pred_str+'_micro_accurs_r'+rad+'_XGc'+str(clnum)+feat_str+'.txt'
print(accur_txt)
accur_str = "exp1 exp2 time_classes accur rand_chance prec_cl0 rec_cl0 prec_cl1 rec_cl1 prec_cl2 rec_cl2 prec_cl3 rec_cl3\n"
imp_txt = accur_txt.replace('accur', 'imp')
shap_txt = accur_txt.replace('accur', 'shap')

comps = []
ei=1
# accumulate all data into training and testing data set
for exp in exps:
    print('experiment 1:', exp)

    df_big = pd.DataFrame()
    ei1 = ei

    # find all the other experiments to accumulate data into (by rock type)
    ei2 = 1
    for exp2 in exp2s:
        if exp[0:2] in exp2 and (exp[0:2] not in comps):
            print('experiment 2:', exp2)
            ei2max = ei2
            # if the first two letters of the exp name are the same
            filen = 'strains_curr_'+exp2+'_g'+rad+'0.txt'


            df = pd.read_csv(folder+filen, delim_whitespace=True)
            df = df.dropna().copy()

            # make classes of time
            time_str = pred_str.replace('dist', '')
            times = df[time_str].values

            normtime = [100*round((max(times)-t)/max(times), 2) for t in times]


            # divide times into this many bins
            timebins = np.linspace(0,100,101) #[0, 50, 100]
            if clnum==4:
                timebins = np.linspace(0,100,1001) #[0, 25, 50, 75, 100]

            norms = set(normtime)
            outcomes = []
            oi=0
            for normt in normtime:

                outcome = max(timebins)
                ti=0
                while ti<len(timebins)-1:
                    tim1 = timebins[ti]
                    tim2 = timebins[ti+1]
                    if normt < tim2 and normt >= tim1:
                        outcome = int(tim1)
                        ti=len(timebins)   # exit the loop

                    ti=ti+1

                outcomes.append(outcome)
                oi=oi+1


            # predict these classes of outcomes
            df[pred_str] = outcomes
            df_big = df_big.append(df, ignore_index = True)

        ei2=ei2+1

    if df_big.empty == False:
        # completed experiments
        comps.append(exp[0:2])
        # scale the data
        df_scale = df_big.copy()
        trans = RobustScaler()# works better with outliers
        df_scale[features] = trans.fit_transform(df_scale[features].values)


        # split into training and testing
        df_scale['is_train'] = np.random.uniform(0, 1, len(df_scale)) <= .80
        # split into training and testing
        train, test = df_scale[df_scale['is_train']==True], df_scale[df_scale['is_train']==False]
        n_train = len(train[pred_str])
        n_test = len(test[pred_str])
        #rfor = RandomForestClassifier(n_jobs=2, random_state=0, max_depth = 5, max_features = 10, n_estimators = 200)
        #rfor.fit(train[features], train[pred_str])

        #model = xgb.XGBClassifier() #classification
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror')  #regression
        parameters = {'colsample_bytree':[0.9], 'alpha':[5], 'learning_rate': [0.1],
                      'n_estimators': [50, 100, 200], 'max_depth':[3, 5, 7]}
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=parameters, cv=10, n_jobs=-1)

        grid_search.fit(train[features], train[pred_str])
        xg_reg = grid_search.best_estimator_

        preds = xg_reg.predict(test[features])
        preds_train = xg_reg.predict(train[features])

        rmse_train = np.sqrt(mean_squared_error(train[pred_str], preds_train))
        rmse_test = np.sqrt(mean_squared_error(test[pred_str], preds))

        r2_test = r2_score(test[pred_str], preds)
        r2_train = r2_score(train[pred_str], preds_train)

        curr_str = "%.0f %.2f %.0f %.0f %.2f %.2f %.2f %.2f \n" % (ei, lim, n_train, n_test, rmse_train, r2_train, rmse_test, r2_test)
        score_str = score_str+curr_str
        print(score_str)

        # plot predicted vs observed
        plt.figure(1)
        plt.plot(test[pred_str], preds, 'ko')
        plt.plot([0, max(test[pred_str])], [0, max(preds)], 'r-')
        plt.ylabel('predicted')
        plt.xlabel('observed')
        plt.title('testing')
        plt.show()

        plt.figure(2)
        plt.plot(train[pred_str], preds_train, 'ko')
        plt.plot([0, max(train[pred_str])], [0, max(preds_train)], 'r-')
        plt.ylabel('predicted')
        plt.xlabel('obsernumfved')
        plt.title('training')
        plt.show()

        imps = xg_reg.feature_importances_
        imp = list(zip(features, imps, range(1, numf+1)))
        imp.sort(key=lambda tup: tup[1], reverse=True)

    #    plt.figure(3)
    #    plt.barh(range(len(imps)), imps.sort(key=lambda tup: tup[0], reverse=True), color='b', align='center')
    #    plt.yticks(range(len(imps)), [features[i] for i in imps])
    #    plt.xlabel('Relative Importance')
    #    plt.show()

        fts = ""
        for ft in imp:
            fts = fts+ft[0]+" "+str(ei)+" "+str(lim)+" "+str(ft[2])+" "+str(round(ft[1], 4))+"\n"

        print(fts)
        imp_str = imp_str+fts

        shap_vals = shap.TreeExplainer(xg_reg).shap_values(train[features])

        shap.summary_plot(shap_vals, train[features], plot_type="bar")
        shap_comb = shap_vals.transpose()

        shap_mean = []
        num_f = len(shap_comb)
        for fi in range(len(shap_comb)):
            vabs = abs(shap_comb[fi])
            v_mean = stat.mean(vabs)
            shap_mean.append(v_mean)

        shapl = list(zip(features, shap_mean, range(1, numf+1)))
        shapl.sort(key=lambda tup: tup[1], reverse=True)

        shps = ""
        for ft in shapl:
            shps = shps+ft[0]+" "+str(ei)+" "+str(lim)+" "+str(ft[2])+" "+str(round(ft[1], 4))+"\n"

        print(shps)
        shp_str = shp_str+fts


        ei=ei+1

    print(score_str)

    f= open(accur_txt, "w")
    f.write(score_str)
    f.close()

    print(imp_str)

    f= open(imp_txt, "w")
    f.write(imp_str)
    f.close()


    print(shp_str)
    f= open(shap_txt, "w")
    f.write(shp_str)
    f.close()

    print(imp_txt)
    print(accur_txt)
    print(shap_txt)
