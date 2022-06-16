import pandas as pd
import numpy as np
import copy
from collections import Counter
import os
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

#Lets load data and do a small preprocessing prior to select variables
FIS = pd.read_excel(os.getcwd()+'/Data/genetica FIS.xlsx', engine='openpyxl',sheet_name='Base total')

total_vars = ['Edad', 'Sexo_femenino', 'Tabaquismo', 'Hipertension', 'Suplementos_vitaminicos', 'Hipercolesterolemia',
             'ETDRS_b', 'ETDRS_V4', 'ETDRS_12m', 'ETDRS_36m', 'Intraretinal_fluid_b', 'Subretinal_fluid_b',
             'Intraretinal_fluid_V4', 'Subretinal_fluid_V4', 'Intraretinal_fluid_36m','Subretinal_fluid_36m',
             'Intraretinal_fluid_any_visit', 'Subretinal_fluid_any_visit', 'Dry_36m', 'Atrophy_b', 'Fibrosis_b',
             'Atrophy_V4', 'Fibrosis_V4', 'Atrophy_36m', 'Fibrosis_36m', 'INYECCIONES_36m','ARMS2', 'CFB', 'CFB.1', 
             'CFH', 'CFH.1', 'SERPINF1', 'VEGFR', 'CFI', 'CFI.1', 'CFI.2', 'SMAD7', 'TGFb1', 'TNF', 'TNF.1',
             'membrana_neovascular_b','membrana_neovascular_V4','grosor_foveolar_b','grosor_foveolar_V4',
             'estado_cristalino_b', 'estado_cristalino_V4', 'estado_cristalino_12m']
             
SNPs_vars = ['ARMS2', 'CFB', 'CFB.1', 'CFH', 'CFH.1', 'SERPINF1', 'VEGFR', 'CFI', 'CFI.1', 'CFI.2', 
             'SMAD7', 'TGFb1', 'TNF', 'TNF.1']

#SELECTED VARIABLES
clinic_num_vars = ['Edad', 'ETDRS_b','ETDRS_V4','ETDRS_12m','grosor_foveolar_b', 'grosor_foveolar_V4']
clinic_cat_vars = ['Sexo_femenino','Tabaquismo','Hipertension','Suplementos_vitaminicos','Hipercolesterolemia',
                    'membrana_neovascular_b','membrana_neovascular_V4','estado_cristalino_b','estado_cristalino_V4',
                    'estado_cristalino_12m','Intraretinal_fluid_b', 'Subretinal_fluid_b', 'Intraretinal_fluid_V4', 
                    'Subretinal_fluid_V4', 'Atrophy_b','Fibrosis_b','Atrophy_V4','Fibrosis_V4']
# geneticvars = ['ARMS2', 'CFB', 'CFB.1', 'CFH', 'CFH.1', 'SERPINF1', 'VEGFR', 'CFI', 
#                'CFI.1', 'CFI.2', 'SMAD7', 'TGFb1', 'TNF', 'TNF.1']
genetic_imp_vars = ['ARMS2', 'CFH', 'VEGFR'] #Here go the variables that Sergio is going to provide me
predvars = ['Atrophy_36m','Fibrosis_36m']

FIS_preprocessed = FIS.copy()
FIS_preprocessed = FIS_preprocessed[clinic_num_vars+clinic_cat_vars+genetic_imp_vars+predvars]

#Drop those samples that contain NANs at predict variables (Atrophy_36m and Fibrosis_36m)
#get NAs in Atrophy_36m and Fibrosis_36m
A36m_NAs = np.where(FIS_preprocessed['Atrophy_36m'].isna().values)[0]
F36m_NAs = np.where(FIS_preprocessed['Fibrosis_36m'].isna().values)[0]
#get the union and drop it
A36m_F36m_NAs_union = list(set(A36m_NAs).union(set(F36m_NAs)))
FIS_preprocessed_na = FIS_preprocessed.drop(A36m_F36m_NAs_union)
FIS_preprocessed_na_index = set(FIS_preprocessed_na.index.values)
#Substitute
FIS_preprocessed = FIS_preprocessed_na.copy()


#------>NUMERICAL VARIABLES<-----#
##Drop the samples containing numerical variables with NAs
NA_index = set(FIS_preprocessed[clinic_num_vars].dropna().index.values)
print(f'Dropping {len(NA_index)} samples from numerical NAs')
##compute difference and drop
FIS_preprocessed.drop(list(FIS_preprocessed_na_index.difference(NA_index)),inplace=True)

print(f'Before imputing categorical variables .... -> {FIS_preprocessed.info()}')

#------>CATEGORICAL VARIABLES<-------#
#Encode and fillna with extra category
print('Encoding with LabelEncoder the categorical variabless!')
for categ_var in clinic_cat_vars+genetic_imp_vars+predvars:
    FIS_preprocessed[categ_var] = LabelEncoder().fit_transform(FIS_preprocessed[categ_var])

#Check there is no NAs
print(f'After imputing categorical variables .... -> {FIS_preprocessed.info()}')

print('Info about numerical variables')
FIS_preprocessed[clinic_num_vars].describe()

print('Info about categorical variables')
for categ_var in clinic_cat_vars+genetic_imp_vars+predvars:
    print(f'{categ_var} -> {Counter(FIS_preprocessed[categ_var])}')

##Save the preprocessed file
FIS_preprocessed.to_csv(os.getcwd()+'/Data/ProyectoFIS/DATA_IN/FIS_dataset.txt',sep='\t',index=False)