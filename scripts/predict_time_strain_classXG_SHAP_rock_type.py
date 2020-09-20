#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:51:45 2019

@author: mcbeck
"""

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
import shap
import statistics

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

rad = '5'
clnum = 2 # 2 or 4
accur_txt = '../txts/predict_strain_comb_'+pred_str+'_micro_accurs_r'+rad+'_XGc'+str(clnum)+feat_str+'.txt'
print(accur_txt)
accur_str = "exp1 exp2 time_classes accur rand_chance prec_cl0 rec_cl0 prec_cl1 rec_cl1 prec_cl2 rec_cl2 prec_cl3 rec_cl3\n"

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
            timebins = [0, 50, 100]
            if clnum==4:
                timebins = [0, 25, 50, 75, 100]

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

        #rfor = RandomForestClassifier(n_jobs=2, random_state=0, max_depth = 5, max_features = 10, n_estimators = 200)
        #rfor.fit(train[features], train[pred_str])

        clf = xgb.XGBClassifier()
        parameters = {
             "eta"    : [0.3] ,
             "max_depth"        : [3, 4, 5],
             "min_child_weight" : [3, 5],
             "gamma"            : [0.1, 0.2],
             "colsample_bytree" : [0.4, 0.5]
             }

        grid_search = GridSearchCV(clf,
                            parameters, n_jobs=-1,
                            scoring="neg_log_loss",
                            cv=3)

        grid_search.fit(train[features], train[pred_str])
        rfor = grid_search.best_estimator_

        preds_for = rfor.predict(test[features])

        cm = pd.crosstab(test[pred_str], preds_for, rownames=['Actual'], colnames=['Predicted'])
        print(cm)

        mets = metrics.classification_report(test[pred_str], preds_for, digits=3, output_dict=True)
        print(mets)

        cl_txt = ''
        # list of rec_cl0 rec_cl0
        cli=0
        for cl in mets:
            print(cl, mets[cl])
            if cli<clnum or (clnum==4 and ('GRS' in exp) and cli<clnum-1):
                scor = mets[cl]
                prec = scor['precision']
                rec = scor['recall']
                cl_txt = cl_txt + str(round(prec, 4)) + ' ' + str(round(rec, 4)) + ' '
            cli=cli+1


        # calculate model success, test, predictions
        accur = metrics.accuracy_score(test[pred_str], preds_for)
        print("Accuracy:", accur)

        outcomes = df_scale[pred_str].values.tolist()
        outs = set(outcomes)
        tot = len(outcomes)
        rats = []
        for out in outs:
            num = outcomes.count(out)
            rats.append(num/tot)

        rand = sum(rats)/len(rats)

        accur_str = accur_str+str(ei1)+' '+str(ei2max)+' '+str(clnum)+' '+str(round(accur, 4))+' '+str(round(rand, 4))+' '+cl_txt+"\n"
        print(accur_str)

        if get_imp:
            imps_str = "property_name property component importance confidence\n"
            save_txt = accur_txt.replace('accurs', exp[0:2]+'_imp')

            shap_str = "property_name feat_num property component SHAP\n"
            shap_txt = accur_txt.replace('accurs', exp[0:2]+'_SHAP')

            shap_vals = shap.TreeExplainer(rfor).shap_values(train[features])

            shap.summary_plot(shap_vals, train[features], plot_type="bar")
            shap_comb = shap_vals[0].copy()

            if clnum==2 or len(shap_vals)==len(features):
                print("only one value reported per class")
                shap_comb = shap_vals.transpose()

            else:
                if len(shap_vals)==3:
                    shap0 = shap_vals[0]
                    shap1 = shap_vals[1]
                    shap2 = shap_vals[2]

                    for i in range(len(shap0)):
                        for j in range(len(shap0[0])):
                            shap_comb[i][j] = abs(shap0[i][j])+abs(shap1[i][j])+abs(shap2[i][j])

                    shap_comb = shap_comb.transpose()
                else:
                    shap0 = shap_vals[0]
                    shap1 = shap_vals[1]
                    shap2 = shap_vals[2]
                    shap3 = shap_vals[3]

                    for i in range(len(shap0)):
                        for j in range(len(shap0[0])):
                            shap_comb[i][j] = abs(shap0[i][j])+abs(shap1[i][j])+abs(shap2[i][j])+abs(shap3[i][j])

                    shap_comb = shap_comb.transpose()

            #if isinstance(shap_comb[0], list):
                # row=sample
                # col=features

            # originally samples x features, and now features x samples

            shap_mean = []

            num_f = len(shap_comb)
            print("should be 9: ", num_f)
            for fi in range(len(shap_comb)):
                vabs = abs(shap_comb[fi])
                v_mean = statistics.mean(vabs)
                shap_mean.append(v_mean)



            shapl = list(zip(features, shap_mean))
            #recur = list(zip(features, recur_bin, recur_rank))
            shapl.sort(key=lambda tup: tup[1], reverse=True)

            featn = len(features)
            imp = list(zip(features, rfor.feature_importances_))
            imp.sort(key=lambda tup: tup[1], reverse=True)

            strc= ''
            for ft in imp:
                fil = ft[0]
                lis = fil.split('_')


                if len(lis)==2:
                    comp = lis[0]
                    val = lis[1]

                    vs = 0
                    i=0
                    while i < len(vals):
                        if vals[i] in val:
                            vs = str(i)
                            i=len(vals)
                        i=i+1

                    if "dn" in comp:
                        cms= '1'
                    elif "dp" in comp:
                        cms = '2'
                    else:
                        cms = '3'
                else:
                    if "ep" in fil:
                        vs = '-2'
                        cms = '0'
                        if "delep" in fil:
                            cms = '-1'
                    else:
                        vs = '-1'
                        cms = '0'
                        if "delsig" in fil:
                            cms = '-1'


                props = vs+' '+cms

                strc = strc+ft[0]+' '+props+' '+str(ft[1])+"\n"

            imps_str = imps_str+strc

            print(save_txt)
            print(imps_str)
            f= open(save_txt, "w")
            f.write(imps_str)
            f.close()

            strc= ''
            for ft in shapl:
                fil = ft[0]
                lis = fil.split('_')


                if len(lis)==2:
                    comp = lis[0]
                    val = lis[1]

                    vs = 0
                    i=0
                    while i < len(vals):
                        if vals[i] in val:
                            vs = str(i)
                            i=len(vals)
                        i=i+1

                    if "dn" in comp:
                        cms= '1'
                    elif "dp" in comp:
                        cms = '2'
                    else:
                        cms = '3'
                else:
                    if "ep" in fil:
                        vs = '-2'
                        cms = '0'
                        if "delep" in fil:
                            cms = '-1'
                    else:
                        vs = '-1'
                        cms = '0'
                        if "delsig" in fil:
                            cms = '-1'

                props = vs+' '+cms

                strc = strc+ft[0]+' '+props+' '+str(ft[1])+"\n"

            shap_str = shap_str+strc
            print(shap_txt)
            print(shap_str)
            f= open(shap_txt, "w")
            f.write(shap_str)
            f.close()


    ei=ei+1

#print("Not writing score file.")
print(accur_str)
print(accur_txt)

f= open(accur_txt, "w")
f.write(accur_str)
f.close()
