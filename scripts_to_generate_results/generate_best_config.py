import numpy as np
import pandas as pd
import os
import pickle
import sys
from itertools import permutations
# pd.set_option('display.max_rows', 500)
pd.set_option("display.max_rows", 500, "display.max_columns", 500)
pd.set_option('display.max_colwidth', -1)
#
dataset_s = sys.argv[1]
best_df = pd.DataFrame()

#for folder in ['svm','xgboost','rf']:
#for folder in ['xgboost','rf']:
for folder in ['svm']:
    print(folder)
    for file in os.listdir(os.getcwd()+f'/outputs/nested_cv_6_6/{dataset_s}/'+folder):
        #
        folder_file_d = pd.read_pickle(os.getcwd()+f'/outputs/nested_cv_6_6/{dataset_s}/'+folder+'/'+file)
        #insert filename and method
        folder_file_d.insert(loc=0, column='file_name', value=file)
        folder_file_d.insert(loc=0, column='method', value=folder_file_d.shape[0] * [folder]) 
        #
        #insert score s =  (TEST BALANCED ACCURACY + TEST AREA ACCURACY - MIN TEST AREA REJECTION)/(COMBINATION OF VARIABLES)
        sdf = folder_file_d['test_balacc'] + folder_file_d['test_area_acc'] - folder_file_d['test_area_reject_rate'].values.tolist()
        folder_file_d.insert(loc=0, column='score', value = sdf)
        #
        #test_balacc,test_area_balacc,test_area_reject_rate
        best_df = pd.concat((best_df,folder_file_d),axis=0)

best_df.insert(loc=1,column='specs',value=best_df.index) 

#define the final df
#final_df = pd.DataFrame()

#sort by 
# for perm,asc in zip(permutations(['test_balacc','test_area_acc','test_area_reject_rate']),permutations([False,False,True])):
#     #concatenate
#     final_df = pd.concat((final_df,best_df.sort_values(by=list(perm),ascending=list(asc))))

# best_df = pd.DataFrame(best_l,columns=['method','variables','configuration','balanced_acc']).sort_values(by='balanced_acc',ascending=False)
# rbf_sigmoid_bool = [False if len(x)==4 and (x[2]=='rbf' or x[2]=='sigmoid') else True for x in best_df['configuration'].values]
# best_df = best_df.loc[rbf_sigmoid_bool,:]
print(best_df)
best_df.to_csv(os.getcwd()+f'/best_conf/nested_cv/{dataset_s}_best_conf_6_6.tsv',sep='\t',index=False)
