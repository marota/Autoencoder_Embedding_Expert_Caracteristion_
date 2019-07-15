import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor as reg
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_validate

def KL_distance(vect1, vect2):
	z_dim = vect1.shape[1]/2
	mean1 = vect1[:z_dim]
	mean2 = vect2[:z_dim]
	var1 = vect1[z_dim:]
	var2 = vect2[z_dim:]

	return 0.5 * np.sum((var1 + (mean2 - mean1)**2) / var2 - 1 - np.log(var2) + np.log(var1))


def coordinates_latent(search, factorDesc, factorMatrix, latent):
	col=[k in search.keys() for k in factorDesc.keys()]
	training_fact = factorMatrix[:,col]

	estimator = reg(n_estimators=100)
	cv_results = cross_validate(estimator, training_fact, latent, cv=3, return_estimator=True)
	cv_predict = np.concatenate([esti.predict(np.asarray([v for v in search.values()]).reshape(1,-1)).reshape(1,-1) for esti in cv_results['estimator']], axis=0)

	target = np.mean(cv_predict, axis=0)

	return target

def nearest_profils(search, factorDesc, factorMatrix, calendar_info, mu, variances=None, metric='minkowski', n_neighbors=5):
	if variances is None:
		latent = mu
	else:
		latent = np.c_[mu, variances]

		if 'date' in search.keys():
			center_idx = np.where(calendar_info.ds==search['date'])[0]
			center = latent[center_idx, :]
		else:
			center = coordinates_latent(search, factorDesc, factorMatrix, latent)

			assert center.shape[1] == latent.shape[1]

			neigh = NearestNeighbors(n_neighbors = n_neighbors, metric=metric)
			neigh.fit(latent)

			idx_neighbors = neigh.kneighbors(center)[1][0]

			return idx_neighbors
