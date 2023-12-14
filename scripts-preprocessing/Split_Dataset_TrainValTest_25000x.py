#!/usr/bin/env python


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

import os

# Goal:
# Generate training validation and test CSV files, using stratification strategy by acquistion
# Split: 80% - 10% - 10%
# Method: 2 splits - trainval & test, then train & val

# Parameters
CSV_InputFile = './Dataset_TargetClass_Overlap-9Blocks.csv'
CSV_OutputFile = './Dataset_TargetClass_Overlap-9Blocks_Acquisition_25000xOnly.csv'
CSV_OutputFile_shuffle = './Dataset_TargetClass_Overlap-9Blocks_25000xOnly_shuffled.csv'
CSV_OutputFile_train = './Dataset_TargetClass_Overlap-9Blocks_25000xOnly_shuffled_train.csv'
CSV_OutputFile_val = './Dataset_TargetClass_Overlap-9Blocks_25000xOnly_shuffled_val.csv'
CSV_OutputFile_test = './Dataset_TargetClass_Overlap-9Blocks_25000xOnly_shuffled_test.csv'


# Read CSV file as dataframe
df1 = pd.read_csv(CSV_InputFile)
df1 = df1[['Location','Material','StartingMaterial','Magnification']]
print('df1')
print(df1.columns)
print(df1.shape)
#print(df1.head())
# shape: (1576,2)

# Generate Label
df1['Label'] = df1['Material'] + 'from' + df1['StartingMaterial']
print('Label: ',df1.iloc[0]['Label'])

# Keep Location with only parent folder
def UpdateLocation(x):
	# Split initial location by folders
	Folders = x.split('/',-1)[-4:]
	# Joing last 3 parent folders and basename
	y = '/'.join(Folders)
	return y

df1['Location'] = df1['Location'].apply(lambda x: UpdateLocation(x))

# Generate Acquisition Info (remove sample information)
df1['Acquisition'] = df1['Location'].apply(lambda x: x[:-13])
df1['Acquisition'] = df1['Acquisition'].apply(lambda x: os.path.basename(x))
print('Acquisition: ',df1.iloc[0]['Acquisition'])

# Data filtering
df1 = df1[df1['Magnification'] == '25000x']
print('Magnification: ',df1.iloc[0]['Magnification'])
print(df1.head())
print(df1.shape)

List_Acquisitions = df1['Acquisition'].unique()
print('List_Acquisitions')
print('Length: ',len(List_Acquisitions))
#print(List_Acquisitions[:10])
# unique acquisitions: 359 (as ADU is oversampled)

# Save CSV file
df1 = df1[['Location','Label','Acquisition']]
df1 = df1.rename(columns={'Location': 'File'})
df1.to_csv(CSV_OutputFile, index=False, na_rep = 'NA', header=True)

# Data shuffling
print('data shuffling...')
df1 = df1.sample(frac=1, random_state=42).reset_index(drop=True)
print('df1 shuffled')
print(df1.shape)
#print(df1.head())

# Save shuffled file
df1.to_csv(CSV_OutputFile_shuffle, index=False, na_rep = 'NA', header=True)

List_Acquisitions = df1['Acquisition'].unique()
print('List_Acquisitions')
print('Length: ',len(List_Acquisitions))
#print(List_Acquisitions[:10])
# unique acquisitions: 359 (as ADU is oversampled)

# First split to define train+val & test dataset (generate 90% - 10% fold)
print("-" * 20)
print("First data split!")
groups = df1['Acquisition']
group_kfold = GroupKFold(n_splits=10)
KFold_Iteration = 0
for trainval_index, test_index in group_kfold.split(df1['File'],df1['Label'],groups):
    #print("\n\nKFold_Iteration", KFold_Iteration)
    #print("test_index",test_index[:10])
    #print("Length test dataset: ", len(test_index))
    
    KFold_Iteration += 1

print("test_index",test_index[:10])

df_trainval = df1.iloc[trainval_index,:]
df_test = df1.iloc[test_index,:]

# WARNING: Reset indices
df_trainval = df_trainval.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


print('df_trainval')
print(df_trainval.shape)
print('List unique acquisitions - df_trainval: ',len(df_trainval['Acquisition'].unique()))
print(df_trainval[:10])
print('df_test')
print('List unique acquisitions - df_test: ',len(df_test['Acquisition'].unique()))
print(df_test.shape)
print(df_test[:10])


# Second split (9 splits for 10%)
print("-" * 20)
print("Second data split!")

groups2 = df_trainval['Acquisition']
group_kfold2 = GroupKFold(n_splits=9)
KFold_Iteration = 0
for train_index, val_index in group_kfold2.split(df_trainval['File'],df_trainval['Label'],groups2):
    #print("\n\nKFold_Iteration", KFold_Iteration)
    #print("val_index",val_index[:10])
    #print("Length validation dataset: ", len(val_index))
    
    KFold_Iteration += 1

print("val_index",val_index[:10])

df_train = df_trainval.iloc[train_index,:]
df_val = df_trainval.iloc[val_index,:]



print('df_train')
print(df_train.shape)
print('List unique acquisitions - df_train: ',len(df_train['Acquisition'].unique()))
print('df_val')
print(df_val.shape)
print('List unique acquisitions - df_val: ',len(df_val['Acquisition'].unique()))
print('df_test')
print(df_test.shape)
print('List unique acquisitions - df_test: ',len(df_test['Acquisition'].unique()))


#print(df_train[:10])
#print(df_val[:10])
#print(df_test[:10])


# Save CSV files
df_train.to_csv(CSV_OutputFile_train, index=False, na_rep = 'NA', header=True)
df_val.to_csv(CSV_OutputFile_val, index=False, na_rep = 'NA', header=True)
df_test.to_csv(CSV_OutputFile_test, index=False, na_rep = 'NA', header=True)



