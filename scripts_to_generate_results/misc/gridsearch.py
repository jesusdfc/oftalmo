import pandas as pd
import numpy as np
from collections import Counter 
import seaborn as sns
import os
import copy
import sklearn
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import linear_model
from  sklearn.linear_model import LogisticRegression as LR
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import re
import functools as f
import pickle
import tqdm
import multiprocessing as mlp
from itertools import chain
import matplotlib
from matplotlib import pyplot as plt
from itertools import combinations as c
from itertools import product as p
import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer,balanced_accuracy_score
from tqdm import tqdm
import sys


WORKING_VAR = sys.argv[1]
print(f'Gridding over {WORKING_VAR}')

#RUN FROM oftal_gridding DIRECTORY
def load_data(drop_dangerous=True,drop_previous_ill=True,
              drop_progress=True,drop_extravars=False,drop_liquids=False,
              drop_foveolar=False,drop_vascular=False,drop_cristalin=False,drop_ETDRS = False,y_var='healthy_ill'):
    #
    clinic = pd.read_csv(os.getcwd()+'/inputs/FIS_after_correction/clinic_raw.txt',sep='\t')
    #Get Y
    Y = clinic[['Atrophy_36m','Fibrosis_36m']]
    #Add the healthy/iill and the softmax problem
    Y.insert(2,'healthy_ill',Y['Atrophy_36m']|Y['Fibrosis_36m'])
    Y.insert(2,'softmax', Y['Atrophy_36m']+Y['Fibrosis_36m'])
    #
    extravars = ['Tabaquismo','Sexo_femenino','Hipertension','Suplementos_vitaminicos','Hipercolesterolemia']
    liquids = ['Intraretinal_fluid_b','Subretinal_fluid_b','Intraretinal_fluid_V4','Subretinal_fluid_V4']
    dangerous_vars = ['Fibrosis_b','Fibrosis_V4','Atrophy_b','Atrophy_V4']
    
    foveolar_vars = ['grosor_foveolar_b','grosor_foveolar_V4']
    vascular_vars =['membrana_neovascular_b','membrana_neovascular_V4']
    cristalin_vars = ['estado_cristalino_b','estado_cristalino_V4','estado_cristalino_12m']

    ETDRs_vars = ['Edad','ETDRS_b','ETDRS_V4','ETDRS_12m']
    #      
    #Define drop vars
    progress_bars = ['V4 -Basal','Buen respondedor V4','Mal respondedor V4',
                 'Buen respondedor 12','Mal respondedor 12','12 meses -Basal']
    drop_vars = ['aparicion_fibrosis_V4','aparicion_atrofia_V4', 'progresion_atrofia_V4',
                 'progresion_fibrosis_V4','Dry_36m','INYECCIONES_36m','Atrophy_36m',
                 'Fibrosis_36m','ETDRS_36m']
    #
    #Atrophy_V4 in case we need it
    Atrophy_V4 = clinic[['Atrophy_V4']]
    Fibrosis_V4 = clinic[['Fibrosis_V4']]
    healthy_ill_drop = Atrophy_V4|Fibrosis_V4

    drop_dict = {'Atrophy_36m':Atrophy_V4,'Fibrosis_36m':Fibrosis_V4,'healthy_ill':healthy_ill_drop}
    #
    if drop_dangerous:
        drop_vars += dangerous_vars
    if drop_progress:
        drop_vars += progress_bars
    if drop_extravars:
        drop_vars += extravars
    if drop_liquids:
        drop_vars += liquids
    if drop_foveolar:
        drop_vars += foveolar_vars
    if drop_vascular:
        drop_vars += vascular_vars
    if drop_cristalin:
        drop_vars += cristalin_vars
    if drop_ETDRS:
        drop_vars += ETDRs_vars
    #
    clinic.drop(drop_vars,axis=1,inplace=True)
    #
    #Drop samples that previously had
    if drop_previous_ill:
        prev_v4 = list(np.where(drop_dict[y_var]==1)[0])
        clinic.drop(prev_v4,axis=0,inplace=True)
        Y = Y.drop(prev_v4,axis=0)
    #
    #sx,sy = clinic.shape
    X = clinic.copy()
    #

    #reset index
    X.reset_index(drop=True,inplace=True)
    Y.reset_index(drop=True,inplace=True)
    
    #check NANs
    final_nans = np.unique(np.where(X.isna())[0])
    X.drop(final_nans,inplace=True)
    Y.drop(final_nans,inplace=True)
    
    #reset index again
    X.reset_index(drop=True,inplace=True)
    Y.reset_index(drop=True,inplace=True)

    return X,Y[y_var]

#PARALLELIZATION
def cvloop_rf(train_index,test_index):

    rf = RandomForestClassifier(n_estimators=comb[0],max_features=comb[1],max_depth=comb[2],
    min_samples_split=comb[3],min_samples_leaf=comb[4],bootstrap=comb[5])

    #train test split
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #fit and predict
    rf.fit(X_train, y_train)
    return rf.predict(X_test),y_test

#PARALLELIZATION
def cvloop_svm(train_index,test_index):

    svm_model = svm.SVC(C=comb[0],gamma=comb[1],kernel=comb[2],class_weight=comb[3])

    #train test split
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #fit and predict
    svm_model.fit(X_train, y_train)
    return svm_model.predict(X_test),y_test

#PARALLELIZATION
def cvloop_xgb(train_index,test_index):

    xgb_model = xgb.XGBClassifier(use_label_encoder=False,verbosity=0,nthread=1,
                                min_child_weight=comb[0], gamma=comb[1], subsample=comb[2], colsample_bytree=comb[3], max_depth=comb[4], 
                                learning_rate=comb[5], n_estimators=comb[6], reg_alpha=comb[7], reg_lambda=comb[8])

    #train test split
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #fit and predict
    xgb_model.fit(X_train, y_train)
    return xgb_model.predict(X_test),y_test

def gridsearch_rf(X,y):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 50, num = 3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 5]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    #
    pred_v = []
    real_v = []
    #create comb dict
    comb_dict = dict()
    #
    global comb
    for comb in tqdm(list(p(n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf,bootstrap))):

        #Define the cv
        cv = KFold(n_splits=len(y),random_state=5,shuffle=True)

        with mlp.Pool(processes=64) as pool:
            results = pool.starmap(cvloop_rf, cv.split(X,y))

        v = np.array(results)[:,:,-1]
        pred_v,real_v = v[:,0],v[:,1]
            
        comb_dict[comb] = balanced_accuracy_score(real_v,pred_v)

    return comb_dict
    #max_key
    # max_key = max(comb_dict, key=comb_dict.get)
    # return max_key,comb_dict[max_key]

def gridsearch_svm(X,y):
    #Define the grid
    C = [0.1, 0.5, 1]
    gamma =  [1, 0.01]
    kernel = ['rbf', 'linear','sigmoid']
    class_weight = [None,'balanced']
    #
    pred_v = []
    real_v = []
    #create comb dict
    comb_dict = dict()
    #
    global comb
    for comb in tqdm(list(p(C,gamma,kernel,class_weight))):

        #Define the cv
        cv = KFold(n_splits=len(y),random_state=5,shuffle=True)

        with mlp.Pool(processes=64) as pool:
            results = pool.starmap(cvloop_svm, cv.split(X,y))

        v = np.array(results)[:,:,-1]
        pred_v,real_v = v[:,0],v[:,1]
            
        comb_dict[comb] = balanced_accuracy_score(real_v,pred_v)

    return comb_dict
    #max_key
    # max_key = max(comb_dict, key=comb_dict.get)
    # return max_key,comb_dict[max_key]

def gridsearch_xgb(X,y):
    #
    min_child_weight = [0.1, 5]
    gamma = [0.5, 1.5]
    subsample = [0.6, 1.0]
    colsample_bytree = [0.6, 1.0]
    max_depth = [int(x) for x in np.linspace(10, 50, num = 3)]
    learning_rate = [0.0001, 0.1]
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 5)]
    reg_alpha = [0.0001, 0.1 ]
    reg_lambda = [0.0001, 0.1]
    #Define the grid
    #
    pred_v = []
    real_v = []
    #create comb dict
    comb_dict = dict()
    #
    global comb
    for comb in tqdm(list(p(min_child_weight, gamma, subsample, colsample_bytree, max_depth, learning_rate, n_estimators, reg_alpha, reg_lambda))):

        #Define the cv
        cv = KFold(n_splits=len(y),random_state=5,shuffle=True)
        #
        #
        with mlp.Pool(processes=64) as pool:
            results = pool.starmap(cvloop_xgb, cv.split(X,y))

        #
        v = np.array(results)[:,:,-1]
        pred_v,real_v = v[:,0],v[:,1]
            
        comb_dict[comb] = balanced_accuracy_score(real_v,pred_v)

    return comb_dict
    #max_key
    # max_key = max(comb_dict, key=comb_dict.get)
    # return max_key,comb_dict[max_key]

#y_var='healthy_ill'
def grid_models_data():
    #
    #generate combination of these 4
    extravars_c,liquids_c,foveolar_vars_c,vascular_vars_c,cristalin_vars_c,ETDRs_vars_c = [True,False],[True,False],[True,False],[True,False],[True,False],[True,False]
    #
    #generate all the combinations (DATA)
    comb_v = p(extravars_c,liquids_c,foveolar_vars_c,vascular_vars_c,cristalin_vars_c,ETDRs_vars_c)
    #drop the first one (empty dataset)
    next(comb_v)
    for extravars,liquids,foveolar_vars,vascular_vars,cristalin_vars,ETDRs_vars in tqdm(list(comb_v)):
        #
        save_p = '_'.join([x+'_'+str(y) for x,y in zip(['extravars','liquids','foveolar','vascular','cristalin','ETDRS'],[extravars,liquids,foveolar_vars,vascular_vars,cristalin_vars,ETDRs_vars])])
        #print(extravars,liquids,foveolar_vars,ETDRs_vars)
        #
        global X,y
        X,y = load_data(drop_extravars=extravars,drop_liquids=liquids,
              drop_foveolar=foveolar_vars,drop_vascular=vascular_vars,drop_cristalin=cristalin_vars,
              drop_ETDRS = ETDRs_vars,y_var=WORKING_VAR)
        #check if it exists
        if not os.path.isfile(os.getcwd()+f'/outputs/{WORKING_VAR}/rf/'+save_p+'.pickle'):
            #Do the RF grid
            best_params_rf = gridsearch_rf(X,y)
            #
            #RF
            with open(os.getcwd()+f'/outputs/{WORKING_VAR}/rf/'+save_p+'.pickle', 'wb') as handle:
                pickle.dump(best_params_rf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        #print('rf')
        if not os.path.isfile(os.getcwd()+f'/outputs/{WORKING_VAR}/xgboost/'+save_p+'.pickle'):
            #Do the XGB grid
            best_params_xgb = gridsearch_xgb(X,y)
            #
            #XGB
            with open(os.getcwd()+f'/outputs/{WORKING_VAR}/xgboost/'+save_p+'.pickle', 'wb') as handle:
                pickle.dump(best_params_xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        if not os.path.isfile(os.getcwd()+f'/outputs/{WORKING_VAR}/svm/'+save_p+'.pickle'):
            #print('xgb')
            #Do the SVM grid
            best_params_svm = gridsearch_svm(X,y)
            #
            #SVM
            with open(os.getcwd()+f'/outputs/{WORKING_VAR}/svm/'+save_p+'.pickle', 'wb') as handle:
                pickle.dump(best_params_svm, handle, protocol=pickle.HIGHEST_PROTOCOL)

#try
with open(os.getcwd()+f'/outputs/{WORKING_VAR}/'+'try'+'.pickle', 'wb') as handle:
            pickle.dump([1], handle, protocol=pickle.HIGHEST_PROTOCOL)
grid_models_data()