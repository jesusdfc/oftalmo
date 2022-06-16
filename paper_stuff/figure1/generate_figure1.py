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
import sys
import pandas as pd
import random

def choose_variables(variables_p):

    variables_p = variables_p[:-7]
    #RETURN, FROM THE PATH, THE ACTUAL VARIABLES THAT WE HAVE TO SELECT TO USE LOAD_DATA():
    #create dict
    variables_d = {}
    #define var names
    var_names = ['extravars','liquids','foveolar','vascular','cristalin','ETDRS','SNPs']
    #fill it with True
    for x in var_names:
        variables_d[x] = True

    variables_split = variables_p.split('_')
    variables_d = dict(zip(variables_split[0:len(variables_split):2],variables_split[1:len(variables_split):2]))

    vars = []
    for x in variables_d:
        if variables_d[x] =='False': #this mean it has not been dropped, so it has been included
            vars.append(x)
    return sorted(vars)

def plot_test_balacc_from_table(method='Atrophy_36m'):

    #Select method
    excel_df = pd.read_csv(os.getcwd()+'/best_conf/nested_cv/'+method+'_best_conf_6_6.tsv', sep='\t')

    #--------------------------------------PLOT 1, SCORE AS A FUNCTION OF THE METHOD (RF, SVM, XGBOOST)---------------------------------------------#
    plot1 = excel_df[['score','method']]
    generate_violin(plot1,'method','score',f'{method}_method_score')

    #--------------------------------------PLOT 2, SCORE AS A FUNCTION OF THE VARIABLE (RF, SVM, XGBOOST)-------------------------------------------#
    plot2 = excel_df[['score','file_name','method','specs']]
    file_name_parsed = [choose_variables(x) for x in plot2['file_name']]
    plot2['file_name'] = file_name_parsed

    #store in dict
    valdd = dict_value(plot2,'score')

    ##Parse it
    plot2_parsed = compute_variable_aportation(plot2,valdd)
    plot2_parsed.Score = plot2_parsed.Score.astype('float')
    generate_violin(plot2_parsed,'Variables', 'Score', f'{method}_variable_score')
    reducedplot2 = reduce_points(plot2_parsed, th=0.01, sp=0.005)

    #add color
    generate_swarmplot_on_top_of_violin(reducedplot2, 'Variables', 'Score', 'Method', name=f'{method}_variable_score')

    #Generate the other way around
    generate_swarmplot_on_top_of_violin(reducedplot2, 'Method', 'Score', 'Variables', name=f'{method}_variable_score_switched', scale=False)

    #--------------------------------------PLOT 3, BALANCED ACCURACY AS A FUNCTION OF THE VARIABLE (RF, SVM, XGBOOST)-------------------------------------------#

    plot3 = excel_df[['test_balacc','file_name','method','specs']]
    file_name_parsed = [choose_variables(x) for x in plot3['file_name']]
    plot3['file_name'] = file_name_parsed

    #store in dict
    valdd = dict_value(plot3,'test_balacc')

    ##Parse it
    plot3_parsed = compute_variable_aportation(plot3,valdd,var='test_balacc',savevar='Test Balanced Accuracy')
    plot3_parsed['Test Balanced Accuracy'] = plot3_parsed['Test Balanced Accuracy'].astype('float')
    generate_violin(plot3_parsed,'Variables','Test Balanced Accuracy',f'{method}variable_balanced_accuracy')
    reducedplot3 = reduce_points(plot3_parsed,th=0.01,sp=0.005, feature = 'Test Balanced Accuracy')

    #add color
    generate_swarmplot_on_top_of_violin(reducedplot3,'Variables','Test Balanced Accuracy','Method',name=f'{method}variable_balanced_accuracy')

    #Generate the other way around
    generate_swarmplot_on_top_of_violin(reducedplot3,'Method','Test Balanced Accuracy','Variables',name=f'{method}variable_balanced_accuracy_switched', scale=False)

def reduce_points(df,th=0.05,sp=0.05, feature='Score'):

    newdf = []
    accumulated = []

    for method in ['rf','svm','xgboost']:
        for i,row in enumerate(df[df.Method == method].sort_values(by=feature,ascending=False).values):

            value = row[0]
            var = row[1]
            #append values
            accumulated+=[[value,var,method]]

            if i==0:
                ref = value    
                newdf+=[[value,var,method]]
                accumulated = []
            else:
                # print(ref - value)
                if abs(ref - value) > th:
                    ref = value
                    newdf+= random.sample(accumulated,int(len(accumulated)*sp))
                    accumulated = []

    return pd.DataFrame(newdf, columns = df.columns)

def dict_value(unparsed_df,var):

    ##
    valdd = dict()
    for row_i in range(unparsed_df.shape[0]):

        if not row_i%25000:
            print(row_i)

        method = unparsed_df.loc[row_i,'method']
        specs = unparsed_df.loc[row_i,'specs']
        key = method.upper()+'_'+'_'.join(unparsed_df.loc[row_i,'file_name'])+'_'+specs
        value = unparsed_df.loc[row_i,var]

        if key in valdd:
            valdd[key] += [value]
        else:
            valdd[key] = [value]

    return valdd

def load_pickle():

    bb = 'extravars_False_liquids_False_foveolar_False_vascular_False_cristalin_False_ETDRS_False_SNPs_False.pickle'
    path = '/home/cslcollab-gserranos/oftal/oftal_gridding/outputs/nested_cv_6_6/healthy_ill/rf/'

    hh = pd.read_pickle(path+bb)

def compute_variable_aportation(df,valdd,var='score',savevar='Score'):

    file_name_splitted = []
    for row_i in range(df.shape[0]):
        ##
        if not row_i % 25000:
            print(row_i)
        #
        score = df.loc[row_i, var]
        method = df.loc[row_i,'method']
        specs = df.loc[row_i,'specs']
        variables = df.loc[row_i,'file_name']
        #unbalanced first
        if len(variables) == 1:
            file_name_splitted.append([score,variables[0],method])
        else: #>1
            for feature in variables:
                old_score = valdd[method.upper() +'_'+ '_'.join(sorted(set(variables).difference(set([feature])))) + '_' + specs]
                file_name_splitted.append([score-old_score,feature,method]) ##add the difference

    return pd.DataFrame(file_name_splitted,columns=[savevar,'Variables','Method'])
        
def generate_violin(df,x,y,name):

    fig, ax = plt.subplots(figsize=(12,3))
    #palett = sns.color_palette("tab10",nvars)
    #ax = sns.violinplot(x=x,y=y,data=df,ax=ax, scale='width', inner=None, palette = palett)
    ax = sns.violinplot(x=x,y=y,data=df,ax=ax, scale='width', inner=None)
    #xlabels = pd.DataFrame(variables_imp_d).mean(axis=0).sort_values(ascending=False).index
    #
    #ax.set_xticklabels(xlabels, rotation=45, ha='right',rotation_mode='anchor')
    #
    plt.savefig(f'{os.getcwd()}/paper_stuff/figure1/violin_{name}.pdf')

def generate_swarmplot_on_top_of_violin(df,x,y,col,name, scale=True):

    fig, ax = plt.subplots(figsize=(24,6))
    if scale:
        ax = sns.violinplot(x=x, y=y, data=df, ax=ax, scale='width', inner=None, color = 'white')
    else:
        ax = sns.violinplot(x=x, y=y, data=df, ax=ax, inner=None, color = 'white')
    for coll in ax.collections:
        coll.set_edgecolor('#404040')
    ax = sns.swarmplot(x=x, y=y, data=df, edgecolor="black",size=1.5, hue=col, palette = sns.color_palette("Set2"))
    plt.legend(loc = 'upper right')
    ax.grid(alpha=0.4, linestyle='--',linewidth=0.1)
    plt.savefig(f'{os.getcwd()}/paper_stuff/figure1/swarm_{name}.pdf')


## GENERATE FIGURES FOR ATROPHY 36m
plot_test_balacc_from_table(method='Fibrosis_36m')