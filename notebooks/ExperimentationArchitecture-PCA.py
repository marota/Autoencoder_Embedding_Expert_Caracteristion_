# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Notebook with PCA model and no conditionning

# ## Loading Libraries 

#import external libraries
import sys
import os
import datetime
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import seaborn as sn
from scipy import stats


# +
#paths in git

#root git folder 
path_main_folder = '/home/marotant/dev/Autoencoder_Embedding_Expert_Caracteristion_'

#add  to path root git folder 
sys.path.append(path_main_folder)
#add  to path source code folder
sys.path.append(path_main_folder+'/src')

# +
#import class and methods from src
from keras import backend as K
from CVAE.callbacks import NEpochLogger,callbackWeightLoss
#from CVAE.cvae import compile_cvae, run_cvae
from CVAE.cvae_model import CVAE, CVAE_emb, CAE
from conso.load_shape_data import *  

import Visualisation.buildProjector
from Visualisation.buildProjector import *
from FeaturesScore.scoring import *
#from conso.load_shape_data import get_x_conso_autoencoder
from conso.conso_helpers import plot_latent_space_projection, pyplot_latent_space_projection_temp, pyplot_latent_space_projection_error
from sklearn.manifold import TSNE

# +
#directories to store trained model and the related projector

log_dir_projector=path_main_folder+"/notebooks/logs/Expe1/PCA/projector"
log_dir_model=path_main_folder+"/notebooks/logs/Expe1/PCA/model"
if not(os.path.isdir(log_dir_projector)):
    os.makedirs(log_dir_projector)
if not(os.path.isdir(log_dir_model)):
    os.makedirs(log_dir_model)
# -

# # Table of contents:
# - Load Data
# - Make Training Set
# - Define and Train Model
# - Build Projector
# - Compute Feature Scores in latent space
# - Study reconstruction Error
# - Study Holidays prediction
# - Detect atypical events
# - Conclusion

# # Load dataset 

# Load dataframe
path_data = os.path.join(path_main_folder, 'data')
dataset_csv = os.path.join(path_data, "dataset.csv")
x_conso = pd.read_csv(dataset_csv, sep=",",)
x_conso.ds = pd.to_datetime(x_conso.ds)

#drop indices column
x_conso=x_conso.drop(columns=x_conso.columns[0],axis=1)


#Visualize data frame head
x_conso.head(5)

# # Make training set of daily electrical consumption profiles and conditions 
# In this experiment there is no condition to pass. This is not something we can do with a PCA anyway

name_set_plot = 'train'
version = '-v1'
nPoints=1830

dict_xconso = {'train': x_conso}

# Normalize input variables
type_scaler = 's'
dict_xconso, _ = normalize_xconso(dict_xconso, type_scaler = 'standard')

dataset = get_dataset_autoencoder(dict_xconso=dict_xconso)

calendar_info = pd.DataFrame(dataset[name_set_plot]['ds'])
calendar_info['month'] = calendar_info.ds.dt.month
calendar_info['weekday'] = calendar_info.ds.dt.weekday
calendar_info['is_weekday'] = (calendar_info.weekday < 5).apply(lambda x:int(x))
calendar_info = pd.merge(calendar_info, x_conso[['ds', 'is_holiday_day']], on='ds', how ='left')
calendar_info.loc[calendar_info['is_holiday_day'].isna(),'is_holiday_day'] = 0

# ### Build and learn PCA model

# +
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

# -

x = dataset['train']['x'][0]

# +
from sklearn.model_selection import KFold # import KFold
import sklearn as sk
kf = KFold(n_splits=5) # Define the split - into 2 folds 
kf.get_n_splits(x) # returns the number of splitting iterations in the cross-validator

for train_index, test_index in kf.split(x):
    #print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = x[train_index], x[test_index]
    principalComponents = pca.fit_transform(X_train)
    X_test_pca=pca.transform(X_test)
    x_hat = pca.inverse_transform(X_test_pca)
    print("mae loss:")
    print(sk.metrics.mean_absolute_error(X_test,x_hat))
    print("mse loss:")
    print(sk.metrics.mean_squared_error(X_test,x_hat))
   # y_train, y_test = y[train_index], y[test_index]
# -

pca.explained_variance_ratio_

filename = log_dir_model+'/pca.sav'
pickle.dump(pca, open(filename, 'wb'))

x_encoded=pca.transform(x)
x_hat = pca.inverse_transform(x_encoded)

import sklearn as sk
print("mae loss:")
sk.metrics.mean_absolute_error(x,x_hat)


# # Analysis of the latent space with the construction of a tensorboard projector

# +

nPoints=1500 #if you want to visualize images of consumption profiles and its recontruction in tensorboard, there is a maximum size that can be handle for a sprite image. 1830 is  
import os,cv2
x_encoded_reduced=x_encoded[0:nPoints,]
images=createLoadProfileImages(x,x_hat,nPoints)

# +

sprites=images_to_sprite(images)
cv2.imwrite(os.path.join(log_dir_projector, 'sprite_4_classes.png'), sprites)

# +

writeMetaData(log_dir_projector,x_conso,calendar_info,nPoints,has_Odd=False)
buildProjector(x_encoded_reduced,images=images, log_dir=log_dir_projector)
# -

log_dir_projector

# # Predict Features in latent space

df_noCond_PCA=predictFeaturesInLatentSPace(x_conso,calendar_info,x_encoded)

# # reconstruction error analysis

error=np.sum(np.abs((x - x_hat)),axis=1)/48

#make a histogram over residuals
import seaborn as sn
sn.distplot(error, kde=False, fit=stats.norm, bins=100)

# Check the day with errors above a threshold

# +
ErrorThreshold=0.15
idxMaxError=[i for i in range(0,nPoints) if error[i]>=ErrorThreshold]
calender_error=calendar_info.loc[idxMaxError]
calender_error['error']=error[idxMaxError]

calender_error
# -

# Check the first n days with highest errors 

# +
nDays=30

decreasingOrderIdx=np.argsort(-error)
calendar_Error_Highest=calendar_info.loc[decreasingOrderIdx[0:nDays]]
calendar_Error_Highest['error']=error[decreasingOrderIdx[0:nDays]]
calendar_Error_Highest
# -

# Visualize the reconstruction error over a specific day

indice=1185 #1185 is the changing hour day end of march
fig = plt.figure(dpi=100,figsize=(3,3))
#set(gca,'Color','k')
plt.plot(x[indice,:])
plt.plot(x_hat[indice,:])

# Visualize the reconstruction error over the days with highest error

nPlots=10#len(idxMaxError)
nCols=5
nRows=int(nPlots/nCols)+1
fig = plt.figure(dpi=100,figsize=(10,10))
for i in range(1, nPlots):
    plt.subplot(nRows, nCols, i)
    fig.subplots_adjust(hspace=.5)
    indice=decreasingOrderIdx[i-1]
    plt.plot(x[indice,:])
    plt.plot(x_hat[indice,:])
    plt.title( calendar_Error_Highest.ds.dt.date.iloc[i-1])

# # Study of holiday predictions

#preparation des features d'interet
yHd=calendar_info['is_holiday_day'].astype(int)
indicesHd=np.array([i for i in range(0, nPoints) if yHd[i] == 1])
yHd_only=yHd[yHd==1]
x_encoded_Hd=x_encoded[indicesHd,]

# +
results_hd=scoreKnnResults(x_encoded,yHd,type='classifier',k=5,cv=10)


# -

# ## holidays well predicted

results_hd_only=[results_hd['predP'][i] for i in indicesHd ]
indices_Hd_predict=[i for i in indicesHd if  results_hd['predP'][i]>=0.5]
indices_Hd_not_predicted=[i for i in indicesHd if  results_hd['predP'][i]<0.5]
calendar_info.loc[indices_Hd_predict]

# +
yWeekday=calendar_info['is_weekday']
results_wk=scoreKnnResults(x_encoded,yWeekday,type='classifier',k=10,cv=10)


# -

weekdays_predicted_as_weekend=[i for i in range(0,1830) if  results_wk['predP'][i]<=0.5 and yWeekday[i]==1]
calendar_info.loc[weekdays_predicted_as_weekend]

len(weekdays_predicted_as_weekend)

# We find out that holidays actually look alike weekends even if they are happening during weekdays

# # Holidays & nearest neighbors

# +
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(10)
neigh.fit(x_encoded)

[distance_knn,kneighbors]=neigh.kneighbors(x_encoded, 2, return_distance=True)
nearest=distance_knn[:,1]
fig = plt.figure(dpi=100,figsize=(3,3))
plt.hist(nearest,bins=100)
plt.show

# +
from scipy import stats

stats.describe(nearest)
# -

fig = plt.figure(dpi=100,figsize=(3,3))
plt.hist(nearest[indicesHd],bins=100)
plt.show

stats.describe(nearest[indicesHd])

calendar_info.loc[np.where(nearest>=1)]

# 2013-01-19 and 2017-01-21 were big snowy events in France. 2013-03-03 and 2014-12-01 are harder to interpret

indicesNear=[i for i in range(0,len(nearest)) if nearest[i]>=1]
nearest[np.where(nearest>=1)]

nPlots=len(indicesNear)#len(idxMaxError)
nCols=5
nRows=int(nPlots/nCols)+1
fig = plt.figure(dpi=100,figsize=(10,3))
for i in range(1, nPlots+1):
    plt.subplot(nRows, nCols, i)
    fig.subplots_adjust(hspace=.5)
    indice=indicesNear[i-1]
    plt.plot(x[indice,:])
    plt.plot(x_hat[indice,:])
    plt.title( calendar_info.ds.dt.date.iloc[indice])
fig.show

# # Conclusions
# - 3 dimensions covers most of the information for the variety of daily load curves 
# - We recovered with this simple linear model the two main features that caracterizes electrical consumption: weekday and temperature
# - Holidays are not yet well predicted and represented, although we know they are an important atypical factor.
# - We however detect that holidays all look alike weekend days
# - We discover some first interpretable events.
#
#


