import numpy as np
from itertools import product as p
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
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
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score,recall_score,balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import sys

def plot_feature_distribution(variables_p,yvar,chosen_metric):

    #init var for plot
    no_feature_imp = False

    #RUN FROM oftal_gridding DIRECTORY
    def load_data(drop_previous_ill=True, drop_extravars=False, drop_liquids=False,
                drop_foveolar=False, drop_vascular=False, drop_cristalin=False,
                drop_ETDRS = False, drop_SNPs=False, y_var='healthy_ill'):
        #
        clinic = pd.read_csv(os.getcwd()+'/inputs/FIS_dataset.txt',sep='\t')
        
        #Make group of variables to drop
        extravars = ['Edad','Tabaquismo', 'Sexo_femenino', 'Hipertension', 'Suplementos_vitaminicos', 'Hipercolesterolemia']
        liquids = ['Intraretinal_fluid_b', 'Subretinal_fluid_b', 'Intraretinal_fluid_V4', 'Subretinal_fluid_V4']
        foveolar_vars = ['grosor_foveolar_b','grosor_foveolar_V4']
        vascular_vars =['membrana_neovascular_b','membrana_neovascular_V4']
        cristalin_vars = ['estado_cristalino_b','estado_cristalino_V4','estado_cristalino_12m']
        ETDRs_vars = ['ETDRS_b','ETDRS_V4','ETDRS_12m']
        SNPs_vars = ['ARMS2', 'CFH', 'VEGFR']
        PrevAtrophFibr = ['Atrophy_b','Fibrosis_b','Atrophy_V4','Fibrosis_V4'] #This is always dropped
        Predicted = ['Atrophy_36m','Fibrosis_36m']

        #Get Y
        Y = clinic[Predicted]

        #Add the healthy/iill and the softmax problem
        Y.insert(2,'healthy_ill',Y['Atrophy_36m']|Y['Fibrosis_36m'])
        Y.insert(2,'softmax', Y['Atrophy_36m']+Y['Fibrosis_36m'])

        #Define drop vars
        drop_vars = []
        #
        #Atrophy_V4 in case we need it
        Atrophy_V4 = clinic['Atrophy_V4']
        Fibrosis_V4 = clinic['Fibrosis_V4']
        healthy_ill_drop = Atrophy_V4|Fibrosis_V4

        #Define the drop dictionary
        drop_dict = {'Atrophy_36m':Atrophy_V4,
                    'Fibrosis_36m':Fibrosis_V4,
                    'healthy_ill':healthy_ill_drop}

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
        if drop_SNPs:
            drop_vars += SNPs_vars

        ##Drop also Atrophy and Fibrosis from previous time and current time (predicted variables).##
        drop_vars += PrevAtrophFibr
        drop_vars += Predicted

        #Drop variables and reset index
        clinic.drop(drop_vars,axis=1,inplace=True)
        clinic.reset_index(drop=True,inplace=True)
        #
        #Drop samples that previously had depending on the predicted variabel (ATROPHY,FIBROSIS,HEALTHY-ILL)
        if drop_previous_ill:
            prev_v4 = list(np.where(drop_dict[y_var]==1)[0])
            X = clinic.drop(prev_v4,axis=0)
            Y = Y.drop(prev_v4,axis=0)  

        #reset index
        X.reset_index(drop=True,inplace=True)
        Y.reset_index(drop=True,inplace=True)

        return X,Y[y_var]

    def choose_variables(variables_p):

        #RETURN, FROM THE PATH, THE ACTUAL VARIABLES THAT WE HAVE TO SELECT TO USE LOAD_DATA():

        #create dict
        variables_d = {}

       #define var names
        var_names = ['extravars','liquids','foveolar_vars','vascular_vars','cristalin_vars','ETDRS_vars','SNPs_vars']

        #fill it with True
        for x in var_names:
            variables_d[x] = True

        for x in variables_p.split('_'):
            print(x)
            variables_d[var_names[np.argmax([x[0:3] in y for y in var_names])]] = False

        return variables_d

    def retrieve_file_path(variables_p):
        fp = ''
        for var in ['extravars','liquids','foveolar','vascular','cristalin','ETDRS','SNPs']:
            fp+='_'
            if var[0:3] in variables_p:
                fp+=var+'_False'
            else:
                fp+=var+'_True'
        return fp[1:]+'.pickle'

    def choose_model(model_params,model_type):

        comb = model_params.split('_')
        print(f'Before -> {comb}')
        #transform to int and 
        for i,_ in enumerate(comb):
            #check numeric
            if comb[i].replace('.','').isnumeric():
                #float
                if '.' in comb[i]:
                    comb[i] = float(comb[i])
                #int
                else:
                    comb[i] = int(comb[i])
            #check boolean
            elif 'True' == comb[i] or 'False' == comb[i]:
                comb[i] = bool(comb[i])

        print(f'After -> {comb}')
        if model_type=='rf':
            #Generate model
            model_orig = RandomForestClassifier(n_estimators=comb[0],max_features=comb[1],max_depth=comb[2],
                                        min_samples_split=comb[3],min_samples_leaf=comb[4],bootstrap=comb[5],
                                        random_state=1)
        elif model_type=='svm':
            #Generate model
            model_orig = svm.SVC(C=comb[0], gamma=comb[1], kernel=comb[2], class_weight=comb[3], probability=True,
                                random_state=1)

        elif model_type=='xgboost':
            #Generate model
            model_orig = xgb.XGBClassifier(use_label_encoder=False,verbosity=0,nthread=1,
                                            min_child_weight=comb[0], gamma=comb[1], subsample=comb[2], 
                                            colsample_bytree=comb[3], max_depth=comb[4], 
                                            learning_rate=comb[5], n_estimators=comb[6], reg_alpha=comb[7], reg_lambda=comb[8],
                                            random_state = 1)

        return model_orig,comb

    #PLOT DISTRIBUTION OF PREDICTIONS 
    def plot_distribution_of_predictions(ytest,ypred,title,conf_mat):

        #order by ypred
        ytest_ordered = ytest[np.argsort(ypred)[::-1]]
        ypred_ordered = ypred[np.argsort(ypred)[::-1]]

        #compute XOR to see the differences (and negate to have 1-> RIGHT, 0 -> WRONG)
        nxor = [int(not int(x)^int(y)) for x,y in zip(ytest_ordered,np.round(ypred_ordered))]

        print(f'Plotting {title}')
        #Plot
        fig, ax = plt.subplots(figsize=(15,15))
        ax = sns.scatterplot(x = range(len(ypred_ordered)), y = ypred_ordered, hue = nxor ,ax=ax, palette = 'deep')
        handles, labels  =  ax.get_legend_handles_labels()
        ax.legend(handles, ['Wrong', 'Right'], loc='upper right')
        ax.table(conf_mat,loc='upper right',colWidths = [0.02, 0.02], bbox = [0.8, 0.925, 0.05, 0.055])
        plt.plot([0, len(ypred_ordered)],[0.5, 0.5], 'k--', color = 'black')
        plt.savefig(f'{os.getcwd()}/figures/{yvar}/{variables_p}/{title}_pred_distribution_{yvar}_{variables_p}.pdf')
        
    #generate new balanced accuracy
    def bal_accuracy(ytest,ypred):
        return balanced_accuracy_score(ytest,np.round(ypred))

    def confmat(ytest,ypred):
        return confusion_matrix(ytest,np.round(ypred))

    N_SPLITS_INNER, N_SPLITS_OUTER = 6,6

    # Example (Select your combination of variables)
    #variables_p = 'foveolar_vascular_ETDRs_SNPs'
    variables_d = choose_variables(variables_p)

    #Retrieve the data
    X,y = load_data(drop_extravars=variables_d['extravars'],drop_liquids=variables_d['liquids'],
              drop_foveolar=variables_d['foveolar_vars'],drop_vascular=variables_d['vascular_vars'],
              drop_cristalin=variables_d['cristalin_vars'], drop_ETDRS = variables_d['ETDRS_vars'],
              drop_SNPs = variables_d['SNPs_vars'], y_var=yvar)

    print(f'Selected variables are {X.columns}')

    #Load the best conf
    best_conf = pd.read_csv(os.getcwd()+f'/best_conf/nested_cv/{yvar}_best_conf_{N_SPLITS_INNER}_{N_SPLITS_OUTER}.tsv',sep='\t')

    #Select the variables
    best_conf_parameters = best_conf[best_conf['file_name'].values == retrieve_file_path(variables_p)].copy()

    ##--->Select the model<---##
    #OPTION 1 -> SORT BY -> MAXIMUM TEST BALACC
    best_conf_parameters_conditioned_mtb = best_conf_parameters.sort_values(by='test_balacc',ascending=False).iloc[0,:]
    print(f"When sorting by test balacc, the best configuration has a test balacc of {best_conf_parameters_conditioned_mtb['test_balacc']}\
          and a score of {best_conf_parameters_conditioned_mtb['score']}")

    #OPTION 2 -> SORT BY -> MAXIMUM SCORE
    best_conf_parameters_conditioned_mts = best_conf_parameters.sort_values(by='score',ascending=False).iloc[0,:]
    print(f"When sorting by score, the best configuration has a test balacc of {best_conf_parameters_conditioned_mts['test_balacc']}\
          and a score of {best_conf_parameters_conditioned_mts['score']}")

    option_dict = {'MaxTestBalacc':best_conf_parameters_conditioned_mtb, 'MaxScore':best_conf_parameters_conditioned_mts}

    #for title, best_conf_params_cond in zip(['MaxTestBalacc','MaxScore'],[best_conf_parameters_conditioned_mtb,best_conf_parameters_conditioned_mts]):

    title = chosen_metric
    best_conf_params_cond = option_dict[title]

    #Select the configuration
    model_params,model = best_conf_params_cond['specs'],best_conf_params_cond['method']

    #retrieve the model
    selected_model,comb = choose_model(model_params,model)

    #TRAIN (This fold would be the Train/Validation, as we have already selected the model)
    CV = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

    #Initialize variables vector
    variables_imp = []
    variables_imp_d = {}

    #Initialize ytest and ypred vectors
    ytest_v = []
    ypred_v = []

    for train_index, test_index in CV.split(X,y):
        #
        Xtrain, Xtest = X.iloc[train_index,:].to_numpy(), X.iloc[test_index,:].to_numpy()
        ytrain, ytest = y[train_index].to_numpy(), y[test_index].to_numpy()

        #FIT
        _ = selected_model.fit(Xtrain,ytrain)

        if model=='svm':
            if comb[2] == 'linear':
                variables_imp.append(selected_model.coef_.flatten().tolist())
            else:
                print('No linear kernel, so no plot of feature importance')
                no_feature_imp = True
        else:
            variables_imp.append(selected_model.feature_importances_.tolist())

        #PREDICT
        ytest_v.append(ytest)
        ypred_v.append([1-x if x>y else y for x,y in selected_model.predict_proba(Xtest)])

    print(f'Validation Balanced Accuracy in {title} is {bal_accuracy(np.hstack(ytest_v),np.hstack(ypred_v))}')

    #Plot distribution
    #plot_distribution_of_predictions(np.hstack(ytest_v),np.hstack(ypred_v),title=title, conf_mat = confmat(np.hstack(ytest_v),np.hstack(ypred_v)))

    #Fill the dict
    if not no_feature_imp:
        for i,var in enumerate(X.columns):
            variables_imp_d[var] = [x[i] for x in variables_imp]
        #
        print('Generating Figure')
        #Plot
        fig, ax = plt.subplots(figsize=(12,3))
        #plt.xticks(rotation=45)
        df = pd.DataFrame(variables_imp_d)[pd.DataFrame(variables_imp_d).mean(axis=0).sort_values(ascending=False).index]
        #reformat

        #ax = sns.boxplot(data=df,ax=ax)
        palett = sns.color_palette("tab10",len(X.columns.tolist()))
        #palett = sns.color_palette("hls", )
        ax = sns.violinplot(x='variable',y='value',data=pd.melt(df),ax=ax,scale='width',inner = 'points', palette = palett)
        xlabels = pd.DataFrame(variables_imp_d).mean(axis=0).sort_values(ascending=False).index
        #
        ax.set_xticklabels(xlabels, rotation=45, ha='right',rotation_mode='anchor')
        #sns.rugplot(y = "value", hue = 'variable', data = pd.melt(df))
        #ax.set_ylim([0,pd.melt(df).value.values.max()+0.08])
        plt.savefig(f'{os.getcwd()}/paper_stuff/figure1/{title}_feature_imp_{yvar}_{variables_p}.pdf')


#Generating plots!
for vars,yvar,chmetric in zip(['foveolar_vascular_ETDRS_SNPs','extravars_liquids_vascular','cristalin_ETDRS_SNPs'],
                              ['healthy_ill','Atrophy_36m','Fibrosis_36m'],
                              ['MaxTestBalacc','MaxScore','MaxScore']):

    print('Generating figure for {yvar}, using {vars} and selecting {chmetric} as metric!')
    #plot feature distribution
    plot_feature_distribution(vars,yvar,chmetric)