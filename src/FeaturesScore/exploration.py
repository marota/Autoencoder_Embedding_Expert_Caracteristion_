import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor as reg
from sklearn.preprocessing import OneHotEncoder as hot_encoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_validate

class KL_distance():

    """Metric distance based on Kullback-Leibler divergence

    """

    def __init__(self, pdf_model='Gaussian'):
        assert pdf_model == 'Gaussian' or pdf_model == 'Laplace'
        self.pdf_model = pdf_model

    def pairwise(self,vect1, vect2):
        z_dim = int(len(vect1)/2)
        mean1 = vect1[:z_dim]
        mean2 = vect2[:z_dim]
        var1 = vect1[z_dim:]
        var2 = vect2[z_dim:]

        if self.pdf_model =='Gaussian':
            return 0.5 * np.sum((var1 + (mean2 - mean1)**2) / var2 - 1 - np.log(var1/var2) )
        elif self.pdf_model == 'Laplace':
            return np.sum((np.abs(mean2 - mean1) + var1*np.exp(-np.abs(mean2 - mean1)/var1))/var2 - 1. - np.log(var1/var2))


def coordinates_latent(search, factorDesc, factorMatrix, latent):

    """ Estimates the coordinates of a set of conditions which can be retrieved in a latent representation (made out of a dataset of points)

    params:
    search -- dict of conditions to look on of the form {name:value}
    factorDesc -- dict of conditions names and types 
    factorMatrix -- array-like, containing conditions values for the representation (columns in the keys order of factorDesc)
    latent -- array-like, consisting of representation coordinates for the dataset of points. 
    
    :return: target -- array-like, array of the predicted coordinates in the representation according to passed conditions
    """
    
    nPoints = factorMatrix.shape[0]
    training_fact = np.zeros((nPoints,1))

    for i,k in enumerate(factorDesc.keys()):
        if k in search.keys():
            target = factorMatrix[:,i]
            n_v = len(np.unique(target))
            if ((n_v>2) & (factorDesc[k]=='category')):
                target_bis = pd.get_dummies(data=target).values
                target = target_bis
                target_search = np.zeros((1,n_v))
                target_search[:,int(search[k]-1)] = 1
                search.update({k:target_search})
            training_fact = np.concatenate((training_fact,target.reshape(nPoints,-1)), axis=1)

    training_fact = training_fact[:,1:]

    estimator = reg(n_estimators=100)
    cv_results = cross_validate(estimator, training_fact, latent, cv=3, return_estimator=True)
    cv_predict = np.concatenate([esti.predict(np.concatenate([v.reshape(1,-1) for v in search.values()],axis=1).reshape(1,-1)).reshape(1,-1) for esti in cv_results['estimator']], axis=0)

    target = np.mean(cv_predict, axis=0)

    return target

def nearest_profiles(center, latent, metric='minkowski', n_neighbors=5, return_center=False,radius=None ):

    """ Return the index and the distances of nearest profiles of a point in a latent representation

    params:
    center -- array-like, coordinates of the query point
    latent -- array-lie, coordinates of the points of the representation
    metric -- string caller or function, the distance metric to evaluate the distance between points
    n_neighbors -- int, the number of nearest neighbors to output
    return_center -- Boolean, whether to return the coordinates of the query point
    radius -- float, masks the nearest neighbors of the query point if the distance between them is larger than the radius times the distance of the nearest neighbor

    :return: idx_neighbors -- array-like, array of index of nearest neighbors
             dist_neighbors -- array-like, array of distances between points and query point
    """

    neigh = NearestNeighbors(n_neighbors = n_neighbors, metric=metric)
    neigh.fit(latent)

    dist_nn, k_neighbors = neigh.kneighbors(center, return_distance = True)
    #radius limitation if the greatest distances of neighbors are too far (radius x min_{u} dist (u0,u))
    if radius is not None:
        r_lim = radius * np.min([dist_nn[0][k] for k in range(len(dist_nn[0])) if (dist_nn[0]-1e-7)[k]>0 ])
        id_lim = [k for k in range(len(dist_nn[0])) if dist_nn[0][k] <= r_lim]
        idx_neighbors= k_neighbors[0][id_lim]
        dist_neighbors = dist_nn[0][id_lim]
        print('{} out of {} neighbors whithin the radius limitation'.format(len(id_lim), n_neighbors))
    else:
        idx_neighbors = k_neighbors
        dist_neighbors = dist_nn

    if return_center:
        return idx_neighbors, dist_neighbors, center
    else:
        return idx_neighbors, dist_neighbors

def search_nearest_profiles(search, factorDesc, factorMatrix, mu, variances=None, metric='minkowski', n_neighbors=5, return_center=False,radius=None):
    """ Return the index and the distances of nearest profiles of a point defined by a set of conditions wich can be retrieved in a latent representation
    
    params
    search -- dict, dict of conditions to look on of the form {name:value}
    factorDesc -- dict, dict of conditions names and types 
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    mu -- array-like, means of profiles for the projection obtained with a (CV)AE
    variances -- array-like, variances of profiles for the projection obtained with a (CV)AE
    metric -- string caller or function, the distance metric to evaluate the distance between points
    n_neighbors -- int, the number of nearest neighbors to output
    return_center -- Boolean, whether to return the coordinates of the query point
    radius -- float, masks the nearest neighbors of the query point if the distance between them is larger than the radius times the distance of the nearest neighbor

    """


    if variances is None:
        latent = mu
    else:
        latent = np.c_[mu, variances]

    center = coordinates_latent(search, factorDesc, factorMatrix, latent).reshape(1,-1)

    assert center.shape[1] == latent.shape[1]

    if metric == 'KL_distance': assert variances is not None
    
    return nearest_profiles(center, latent, metric, n_neighbors, return_center, radius) 

