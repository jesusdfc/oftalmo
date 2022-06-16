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

WORKING_VAR = list(map(float,sys.argv[1:]))
#n_estimators and max depth needs to be an int
WORKING_VAR[-3] = int(WORKING_VAR[-3])
WORKING_VAR[4] = int(WORKING_VAR[4])

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
    extravars = ['Edad','Tabaquismo','Sexo_femenino','Hipertension','Suplementos_vitaminicos','Hipercolesterolemia']
    liquids = ['Intraretinal_fluid_b','Subretinal_fluid_b','Intraretinal_fluid_V4','Subretinal_fluid_V4']
    dangerous_vars = ['Fibrosis_b','Fibrosis_V4','Atrophy_b','Atrophy_V4']
    
    foveolar_vars = ['grosor_foveolar_b','grosor_foveolar_V4']
    vascular_vars =['membrana_neovascular_b','membrana_neovascular_V4']
    cristalin_vars = ['estado_cristalino_b','estado_cristalino_V4','estado_cristalino_12m']

    ETDRs_vars = ['ETDRS_b','ETDRS_V4','ETDRS_12m']
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

#extravars_True_liquids_False_foveolar_True_vascular_False_cristalin_True_ETDRS_False
def trymodel(conf,plot=False):

    #
    X,y = load_data(drop_extravars=True,drop_liquids=False,drop_foveolar=False,
                    drop_vascular=True,drop_cristalin=False,drop_ETDRS=False)
    print(f'X shape is ',X.shape[0],'by ',X.shape[1])
    #
    #Leave-one-out
    #Initialize Kfold
    cv = KFold(n_splits=len(y), random_state=5, shuffle=True)
    # with mlp.Pool(processes=64) as pool:
    #     results = pool.starmap(cvloop, [cv.split(X,y),model,X,y])
    tlist = []
    iter=0
    for train_index,test_index in cv.split(X):
        print('iter num',iter)
        iter+=1
        #Define the model
        model = xgb.XGBClassifier(use_label_encoder=False,verbosity=0,nthread=1,
                                min_child_weight=conf[0],gamma=conf[1],subsample=conf[2],colsample_bytree=conf[3],
                                max_depth=conf[4],learning_rate=conf[5],n_estimators=conf[6],
                                reg_alpha=conf[7],reg_lambda=conf[8])

        #Split
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        
        #Compute probs
        probs = model.predict_proba(X_test)

        tlist.append({'solpred':model.predict(X_test),'soltrue':np.array(y_test),'probs':probs})
    #
    solpred = [y[0] for y in map(lambda x: x['solpred'],tlist)]
    soltrue = [y[0] for y in map(lambda x: x['soltrue'],tlist)]
    #imp_features = [y for y in map(lambda x: x['fi'],tlist)]
    #
    print(confusion_matrix(solpred, soltrue))
    #print(f'Counter of labels train: {Counter(y_train)}')
    #print(f'Counter of labels test: {Counter(y_test)}')
    print(f'Balanced: {sklearn.metrics.balanced_accuracy_score(soltrue,solpred)}')
    print(f'Unbalanced: {sklearn.metrics.accuracy_score(soltrue,solpred)}')
    print()
    if plot:
        feature_importance = pd.DataFrame(imp_features)
        feature_importance.columns = X.columns
        sorted_fi = feature_importance.apply(np.max).sort_values(ascending=False)._index
        #
        fig, ax = plt.subplots(figsize=(15,15))
        plt.xticks(rotation=90)
        ax = sns.boxplot(data=feature_importance[sorted_fi],ax=ax)
        plt.savefig('oftal/'+mode+'.pdf',dpi=300)


print(f'Evaluating {WORKING_VAR}')
trymodel(conf=WORKING_VAR)