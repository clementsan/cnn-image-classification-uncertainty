#!/usr/bin/env python

#-------------
# Libraries
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import scipy
import os
import pandas as pd
from shutil import copy

from utils import *

#-------------
# Parameters

CSV_InputFile = './Database_Query_TargetClass.csv'
CSV_OutputFile = './Dataset_TargetClass_Overlap-9Blocks.csv'


# Warning: full path needed
Database_Dir = '/path_database_SEM/'
DataDir = '~/Project_SEM/Project_TargetClass/data/'

Nb_Samples = 20
ROI_Width = 512
ROI_Height = 440
Overlap_Width = int(ROI_Width / 2)
Overlap_Height = int(ROI_Height / 2)

#-------------
# Processing

# Create sample by mosaicing dataset


# Read CSV file
df = pd.read_csv(CSV_InputFile)

# Create new dataframe for analysis
columns = ['Location','Material','Magnification','Resolution','StartingMaterial','AcquisitionDate']
df2 = pd.DataFrame(columns=columns)

# Iterate over rows
for index, row in df.iterrows():
	FileName = row['Location']
	FileName_Head, FileName_Tail = os.path.split(FileName)
	Material = row['Material']
	StartingMaterial = row['StartingMaterial']
	Magnification = row['Magnification']
	#print(row)
	#print('\t ImageName: ',FileName)

	DataDir_Material = os.path.join(DataDir,Material)
	DataDir_StartingMaterial = os.path.join(DataDir_Material,StartingMaterial)
	DataDir_Magnification = os.path.join(DataDir_StartingMaterial,Magnification)
	if not os.path.exists(DataDir_Magnification):
		os.makedirs(DataDir_Magnification)

	img = read_img(FileName)
	img_gray = rgb2gray(img)

	Database_Folder = Material + '-' + StartingMaterial + '/'

	# Generate all block samples
	pimg_gray_Array = mosaicData_Overlap(img_gray, w = ROI_Width, h = ROI_Height, Overlap_w = Overlap_Width, Overlap_h = Overlap_Height)
	Nb_Samples = len(pimg_gray_Array)

	# Generate images
	for i in range(Nb_Samples):
		pimg_gray = pimg_gray_Array[i]

		FileName_RandomSample = os.path.join(DataDir_Magnification,FileName_Tail)
		FileName_RandomSample = FileName_RandomSample.replace('.tif.tif','.tif')
		FileName_RandomSample = FileName_RandomSample.replace('.tif','_sample_' + str(i+1) + '.tif')
		#print(FileName_RandomSample)

		# Save random sample
		scipy.misc.imsave(FileName_RandomSample, pimg_gray.astype('uint8'))

		row_sample = row.copy()
		row_sample['Location'] = FileName_RandomSample
		#print(row_sample)

		df2 = df2.append(row_sample, ignore_index = True )

# Fill NA values
df2.fillna('NA')

# Display output dataframe
print('\n Output data frame:')
print(df2.head())

# Save new data frame into CSV file
df2.to_csv(CSV_OutputFile, index=False, na_rep = 'NA')
