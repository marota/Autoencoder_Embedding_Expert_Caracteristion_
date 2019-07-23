import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor as reg
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_validate

class KL_distance():
    def __init__(self, prior='Gaussian'):
        assert prior == 'Gaussian' or prior == 'Laplace'
        self.prior = prior

    def pairwise(self,vect1, vect2):
        z_dim = int(len(vect1)/2)
        mean1 = vect1[:z_dim]
        mean2 = vect2[:z_dim]
        var1 = vect1[z_dim:]
        var2 = vect2[z_dim:]

        if self.prior =='Gaussian':
            return 0.5 * np.sum((var1 + (mean2 - mean1)**2) / var2 - 1 - np.log(var1/var2) )
        elif self.prior == 'Laplace':
            return np.sum((np.abs(mean2 - mean1) + var1*np.exp(-np.abs(mean2 - mean1)/var1))/var2 - 1. - np.log(var1/var2))


def coordinates_latent(search, factorDesc, factorMatrix, latent):
    col=[k in search.keys() for k in factorDesc.keys()]
    training_fact = factorMatrix[:,col]

    estimator = reg(n_estimators=100)
    cv_results = cross_validate(estimator, training_fact, latent, cv=3, return_estimator=True)
    cv_predict = np.concatenate([esti.predict(np.asarray([v for v in search.values()]).reshape(1,-1)).reshape(1,-1) for esti in cv_results['estimator']], axis=0)

    target = np.mean(cv_predict, axis=0)

    return target

def nearest_profiles(search, factorDesc, factorMatrix, ds, mu, variances=None, metric='minkowski', n_neighbors=5, return_center=False,radius=None):
    if variances is None:
        latent = mu
    else:
        latent = np.c_[mu, variances]

    if 'date' in search.keys():
        center_idx = np.where(ds==search['date'])[0]
        if center_idx is None: print('date not found')
        center = latent[center_idx, :][0]
    else:
        center = coordinates_latent(search, factorDesc, factorMatrix, latent)
    center = center.reshape(1,-1)
    assert center.shape[1] == latent.shape[1]

    if metric == 'KL_distance': assert variances is not None
    neigh = NearestNeighbors(n_neighbors = n_neighbors, metric=metric)
    neigh.fit(latent)

    k_neighbors = neigh.kneighbors(center)
    #radius limitation if the greatest distances of neighbors are too far (radius x min_{u} dist (u0,u))
    if radius is not None:
        r_lim = radius * np.min([k_neighbors[0][0][k] for k in range(len(k_neighbors[0][0])) if (k_neighbors[0][0]-1e-7)[k]>0 ])
        id_lim = [k for k in range(len(k_neighbors[0][0])) if k_neighbors[0][0][k] <= r_lim]
        idx_neighbors=k_neighbors[1][0][id_lim]
        print('{} out of {} neighbors whithin the radius limitation'.format(len(id_lim), n_neighbors))
    else:
        idx_neighbors=k_neighbors[1][0]

    if return_center:
        return idx_neighbors, center
    else:
        return idx_neighbors

def generative_estimation(search, model, factorDesc, factorMatrix, dataset, mu, variance, prior, metric='minkowski', n_neighbors=5, trust=0.01, x_truth = None, mean_sc = None, std_sc = None, radius=None):
    #parameters to rescale the loads profiles
    if mean_sc is None: mean_sc = 0
    if std_sc is None: std_sc = 1

    #search for the nearest coordinates and the nearest profiles
    idx_neighbors, latent_center = nearest_profiles(search=search, factorDesc=factorDesc, factorMatrix=factorMatrix, ds=dataset['train']['ds'], mu=mu, variances=variance, metric=metric, n_neighbors=n_neighbors, return_center=True, radius=radius)
    
    idx_neighbors = idx_neighbors
    z_dim = mu.shape[1]
    latent_center.reshape(1,-1)
    mu_c = latent_center[:,:z_dim]
    var_c = latent_center[:,z_dim:]

    #simulation of the profiles according to stochastic parametrization of the latent space
    estimations=[]
    n= 1000
    for k in range(n):
        if prior=='Gaussian':
            eps = np.random.normal(size=z_dim)
            z_sim = eps * np.sqrt(var_c) + mu_c
        elif prior=='Laplace':
            U = np.random.uniform(size=z_dim)
            V = np.random.uniform(size=z_dim)
            Rad_sample = 2.*(V>=0.5) - 1. 
            Expon_sample = -var_c*np.log(1-U)
            z_sim = mu_c + Rad_sample*Expon_sample
        input_decoder = [z_sim] + [dataset['train']['x'][nb] for nb in np.arange(1,len(dataset['train']['x']), step=1, dtype=int)]
        pred = model.decoder.predict(input_decoder)
        estimations.append(pred.reshape(1,-1))

    estimations = np.concatenate(estimations, axis=0)
    mean_estimation = np.mean(estimations, axis=0)

    x = dataset['train']['x'][0]
    x_hat = model.cvae.predict(dataset['train']['x'])[0]
    mae_error_std = np.std(np.abs(x - x_hat)[idx_neighbors,:], axis=0)
    q = stats.norm.cdf(1-trust)
    first_quantile = mean_estimation - q* mae_error_std
    last_quantile = mean_estimation + q* mae_error_std

    #display profiles
    plt.figure(dpi=100,figsize=(5,5))
    if x_truth is not None:
        plt.plot(x_truth*std_sc+ mean_sc, label = 'truth', color='royalblue', lw=3)
    plt.plot(mean_estimation*std_sc+ mean_sc, label = 'load estimation', color='darkorange')
    plt.plot(first_quantile*std_sc+ mean_sc, '--', label = 'error of neighbors at {}'.format((1-trust)*100), color='peru')
    plt.plot(last_quantile*std_sc+ mean_sc, '--', color='peru')

    if x_truth is None:
        plt.plot(dataset['train']['x'][0][idx_neighbors[0],:]*std_sc+ mean_sc, label = 'nearest profile', color='indigo')

    plt.title('Daily load estimation (MW)')
    plt.xlabel('hours')
    plt.xlim((0,48))
    plt.xticks([5,11,17,23,29,35,41], [3,6,9,12,15,18,21])
    plt.legend(loc='lower right')
    plt.grid();
    
    if x_truth is not None:
        plt.figure(dpi=100,figsize=(5,5))
        gap = np.fmin(0, x_truth - first_quantile) + np.fmax(0, x_truth - last_quantile)
        plt.plot(gap*std_sc, color='red')
        plt.title('Gap between truth and estimation boundaries (MW)')
        plt.xlabel('hours')
        plt.xlim((0,48))
        plt.xticks([5,11,17,23,29,35,41], [3,6,9,12,15,18,21])
        plt.axhline(0, color='black')
    
    plt.subplots_adjust(wspace=0.1)    
    plt.show()
    #return estimators
    return mean_estimation

