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

# # Notebook with CVAE model and conditionning over day of the week, month and temperature

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
#path_main_folder = '/home/marotant/dev/Autoencoder_Embedding_Expert_Caracteristion_'
path_main_folder = '/home/jovyan'#specify the root folder of the git repo

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

log_dir_projector=path_main_folder+"/notebooks/logs/Expe3/CVAE_W_M_T/projector"
log_dir_model=path_main_folder+"/notebooks/logs/Expe3/CVAE_W_M_T//model"
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
# In this experiment, we use some conditions from expert knowledge we previosuly recovered (temperature, month, days of the week) to learn a new residual latent space.  

name_set_plot = 'train'
version = '-v1'
nPoints=1830

dict_xconso = {'train': x_conso}

# Normalize input variables
type_scaler = 's'
dict_xconso, _ = normalize_xconso(dict_xconso, type_scaler = 'standard')

# Give the features on which to condition

type_x = ['conso']
type_cond = ['day','month','temperature']
dataset = get_dataset_autoencoder(dict_xconso=dict_xconso, type_x=type_x, type_cond=type_cond)

# +

nPoints=dataset['train']['x'][1].shape[0]

#get conditions in array
days_emb =dataset['train']['x'][1][:,0:7]
month_emb =dataset['train']['x'][1][:,7:19]
temp_emb=dataset['train']['x'][1][:,19:]
to_emb=dataset['train']['x'][1]

x = dataset['train']['x'][0]

dataset['train']['x'] = [x,days_emb,month_emb,temp_emb]
# -

np.shape(temp_emb)

calendar_info = pd.DataFrame(dataset[name_set_plot]['ds'])
calendar_info['month'] = calendar_info.ds.dt.month
calendar_info['weekday'] = calendar_info.ds.dt.weekday
calendar_info['is_weekday'] = (calendar_info.weekday < 5).apply(lambda x:int(x))
calendar_info = pd.merge(calendar_info, x_conso[['ds', 'is_holiday_day']], on='ds', how ='left')
calendar_info.loc[calendar_info['is_holiday_day'].isna(),'is_holiday_day'] = 0

# # Build and train model CVAE

#on sauvegarde le dataset
path_out = log_dir_model


# Parameters for autoencoder
e_dims=[48,35,24,12]#encoder dim
d_dims=[48,35,24,12]#decoder dim. Dense Blocks in skip connections can make the dimensions bigger when layers are concatenated with the previous one
to_emb_dim=[7,12,48] #input dimensions for conditions
cond_pre_dim = 0
input_dim = dataset['train']['x'][0].shape[1]
z_dim= 4
lambda_val = 0.5 #hyper-parameter which value was selected after cross-validation

name_model = 'cvae_conso-W_M_T_30min-journalier'
#name_model = 'cvae_classification'

# +
#if needs to relaod model classes after modification wothout restarting the kernel

import CVAE.cvae_model
import CVAE.callbacks
import importlib
importlib.reload(CVAE.cvae_model)
importlib.reload(CVAE.callbacks)

# %load_ext autoreload
# %autoreload
# -

Lambda = K.variable(lambda_val, dtype='float32')
model = CVAE.cvae_model.CVAE_emb(input_dim=input_dim,
                  e_dims=e_dims, 
                  d_dims=d_dims, 
                  cond_pre_dim=cond_pre_dim,
                  z_dim=z_dim, 
                  beta=Lambda,
                  name=name_model, 
                  output=path_out,
                 to_emb_dim=to_emb_dim,
                 emb_dims=[[5,3],[6,3],[12,4]], emb_to_z_dim=[5,5],is_L2_Loss=False,has_BN=2)#these dimensions define the dimension layer of the conitional network

# ### Trainning model 

# +
from keras.callbacks import TensorBoard
from time import time

#embeddingsMetadata = {'dec_dense_0': 'metadata.tsv'}
tensorboard = TensorBoard(log_dir="logs/{}".format(name_model +str(time())),write_graph=True)#,write_images=True,embeddings_freq=10, embeddings_layer_names=['dec_dense_0'],embeddings_metadata= embeddingsMetadata)

# -

# We start a pre-training phase with a constant and high lambda to properly learn the conditional embedding

# +
#model.main_train(dataset, training_epochs=200, batch_size=20, verbose=False,callbacks=[tensorboard])
import warnings
warnings.filterwarnings('ignore')

lambda_decreaseRate=0.0
lambda_min=0.01 #p

out_batch = NEpochLogger(x_train_data=dataset['train']['x'], display=100,x_conso=x_conso,calendar_info=calendar_info)
weightLoss=callbackWeightLoss(lambda_val,lambda_decreaseRate,lambda_min)
#model.main_train(dataset, training_epochs=1500, batch_size=40, verbose=False,callbacks=[tensorboard,out_batch])#,weightLoss])
model.main_train(dataset, training_epochs=1500, batch_size=32, verbose=0,callbacks=[tensorboard,out_batch],validation_split=0.1)

#visualizer = LatentSpaceVisualizer(model_folder_path=model_path, dataset_path=labellisation_data_folder + 'sequences_dataset/sequences_et_labels.npz')
 #   visualizer.visualize_embedding_after_training()
# -

# We now continue with a learning phase for better reconstruction and refine the similarities between instances. It can be seen as a diffusion phase. We smoothly decese the lambda after each epoch.

# +
lambda_decreaseRate=0.001 #parameter by default
weightLoss=callbackWeightLoss(lambda_val,lambda_decreaseRate,lambda_min)
#model.main_train(dataset, training_epochs=1500, batch_size=40, verbose=False,callbacks=[tensorboard,out_batch])#,weightLoss])
model.main_train(dataset, training_epochs=2000, batch_size=40, verbose=0,callbacks=[tensorboard,out_batch,weightLoss],validation_split=0.1)

#visualizer = LatentSpaceVisualizer(model_folder_path=model_path, dataset_path=labellisation_data_folder + 'sequences_dataset/sequences_et_labels.npz')
 #   visualizer.visualize_embedding_after_training()
# -

# DimsImportance=[1394.4407  1343.6127    44.23878 1616.7899 ] Only 3 dimensions are significant here (each term is the sum of absolute values in each direction for the all the datapoints.
# There is no significant overfitting when comparing training error to validation error. This will be confimed later on specific examples.

with open(os.path.join(log_dir_model,name_model,"config.txt"),'w') as file: 
    file.write(str(cond_pre_dim) + '\n')
    #file.write(str(emb_dims) + '\n')
    file.write(str(e_dims) + '\n') 
    file.write(str(d_dims) + '\n') 
    file.write(str(z_dim) + '\n')
    file.write(str(Lambda) + '\n')

# +
#sauvegarde du dataset associé
name_dataset = 'dataset.pickle'

with open( os.path.join(log_dir_model,name_model, name_dataset), "wb" ) as file:
    pickle.dump( dataset, file )
# -

# ## Loading model 

model.load_model(os.path.join(path_out, name_model, 'models'))

# +
emb_inputs=[days_emb,month_emb,temp_emb]
emb_ouputs = model.embedding_enc.predict(emb_inputs)

#cond_pre=day_emb
#cond = np.concatenate((cond_pre, emb_ouputs), axis=1)
cond  = emb_ouputs
x_input = dataset['train']['x'][0]

input_encoder = [x_input,cond]
# -

x_encoded = model.encoder.predict(input_encoder)[0]
x_hat = model.cvae.predict(x=dataset['train']['x'])[0]

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

# It is possible to explore the resulting latent projection with tensorboard. You should easily visualize holidays and similar non-working days: like the Christmas Week or days happening before or after holidays

# # Predict Features in latent space

df_noCond_VAE=predictFeaturesInLatentSPace(x_conso,calendar_info,x_encoded)

# We can notice that for the conditions passed (W-M-T), their recovery score in the residual latent space is closed to random. On the other hand, the recovery score of holidays is now high. So the residual latent space is now much more dependant over this feature: holidays are now well-represented in this representation.  

# # reconstruction error analysis

error=np.sum(np.abs((x - x_hat)),axis=1)/48

#make a histogram over residuals
import seaborn as sn
sn.distplot(error, kde=False, fit=stats.norm, bins=100)

# Check the day with errors above a threshold

# +
ErrorThreshold=0.08
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

# 2013-03-31 is the day with a missing hour because of changing day time and the consumption value is set to 0. It is hence normal that it is not well predicted and a good indicator that the model does not tend to overfit.

# # Study of holiday predictions

#preparation des features d'interet
yHd=calendar_info['is_holiday_day'].astype(int)
indicesHd=np.array([i for i in range(0, nPoints) if yHd[i] == 1])
yHd_only=yHd[yHd==1]
x_encoded_Hd=x_encoded[indicesHd,]

# +
results_hd=scoreKnnResults(x_encoded,yHd,type='classifier',k=5,cv=10)


# -

# ## holidays well predicted and not

results_hd_only=[results_hd['predP'][i] for i in indicesHd ]
indices_Hd_predict=[i for i in indicesHd if  results_hd['predP'][i]>=0.5]
indices_Hd_not_predicted=[i for i in indicesHd if  results_hd['predP'][i]<0.5]
calendar_info.loc[indices_Hd_predict]

len(indices_Hd_predict)


# +

calendar_info.loc[indices_Hd_not_predicted]
# -

len(results_hd_only_NotPredicted)

# Days not predicted are mostly days on weekends except for 2013-11-01, 2015-05-08, 2014-05-08 and 2013-04-01 which are similar to some holidays but also to non-working days. They could hence be predicted as non-working days which is fine

nPlots=10#len(idxMaxError)
nCols=5
nRows=int(nPlots/nCols)+1
fig = plt.figure(dpi=100,figsize=(10,10))
for i in range(1, nPlots):
    plt.subplot(nRows, nCols, i)
    fig.subplots_adjust(hspace=.5)
    indice=indices_Hd_not_predicted[i-1]
    plt.plot(x[indice,:])
    plt.plot(x_hat[indice,:])
    plt.title( calendar_info.ds.dt.date.iloc[indice])

# # Detect Events as local outliers

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

calendar_info.loc[np.where(nearest>=0.35)]

# 2013-01-11 and 2017-01-26 were big snowy events in France. 2013-04-14 was a punctual summer day in early spring. 2014-02-06 was a stormy day. Other days are either holidays or similar non-working days which can be analysed in the projector.

indicesNear=[i for i in range(0,len(nearest)) if nearest[i]>=1]
nearest[np.where(nearest>=1)]

nPlots=len(indicesNear)#len(idxMaxError)
nCols=5
nRows=int(nPlots/nCols)+1
fig = plt.figure(dpi=100,figsize=(10,10))
for i in range(1, nPlots+1):
    plt.subplot(nRows, nCols, i)
    fig.subplots_adjust(hspace=.5)
    indice=indicesNear[i-1]
    plt.plot(x[indice,:])
    plt.plot(x_hat[indice,:])
    plt.title( calendar_info.ds.dt.date.iloc[indice])
fig.show

# # Conclusions
# - We managed to properly condition on the expert feature passed as the residual latent space is almost independant from them
# - We recover holidays which are well-represented and can be predicted
# - We can use the latent projection to discover other non-working days to label such as days near holidays or Christmas week
# - When looking at local outliers, we discover first weather events to further analyze in a more appropriate conditional representation were they are better represented.
#
#
#




