from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_validate 

from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from datetime import datetime

def build():
   print("building")

def scoreKnnResults(x,y,type='classifier',k=5,cv=10):
    knn = KNeighborsRegressor(n_neighbors=k, weights = 'distance')
    if(type=='classifier'):
        knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=cv)# scoring='accuracy'
    F1=[]
    probScore=[]
    if(type=='classifier'):
        F1 = cross_val_score(knn, x, y, cv=cv, scoring='f1_macro')
        proba = np.array(cross_val_predict(knn, x, y, cv=cv, method='predict_proba'))
        probScore=[proba[i][y[i]] for i in range(0,len(y)) ]
    #print(np.mean(F1))
    #print(np.mean(scores))
    #print(np.mean(probScore))
    #print(np.std(probScore))
    #cv_scores.append(scores.mean())
    return({'F1':np.mean(F1),'predD':np.mean(scores),'predP':probScore})
    

def predictFeaturesInLatentSPace(xconso,calendar_info,x_reduced,k=5,cv=10):

    columns_x = xconso.columns
    conso_idx = np.argmax(['consumption' in c for c in xconso.columns])
    temp_idx = np.argmax(['temperature' in c for c in xconso.columns])
    
    preditionDetermistic=[]
    preditionProbabilistic=[]
    predictionStd=[]
    predictionRandom=[]
    F1score=[]
    #on va predire nos different features dans l espace latent avec un plus proche voisin
    nPoints=calendar_info.shape[0]
    
    #preparation des features d'interet
    yHd=calendar_info['is_holiday_day'].astype(int)
    indicesHd=np.array([i for i in range(0, nPoints) if yHd[i] == 1])
    yHd_only=yHd[yHd==1]
    x_reduced_Hd=x_reduced[indicesHd,]

    
    yWeekday=calendar_info['is_weekday']
    yMonth=calendar_info['month']
    #yMonth=calendar_info['weekday']
    yWkday=calendar_info['weekday']
    
    dates = np.unique(xconso['ds'].dt.date)
    
    #temperatureMax=[max(xconso[columns_x[temp_idx]].iloc[np.where(xconso['ds'].dt.date==dates[k])]) for k in range(dates.shape[0])]
    temperatureMean=[np.mean(xconso[columns_x[temp_idx]].iloc[np.where(xconso['ds'].dt.date==dates[k])]) for k in range(dates.shape[0])]
    yTemp=temperatureMean
    
    #preparation des classifiers knn
    results_wd=scoreKnnResults(x_reduced,yWeekday,type='classifier',k=k,cv=cv)
    results_day=scoreKnnResults(x_reduced,yWkday,type='classifier',k=k,cv=cv)
    results_month=scoreKnnResults(x_reduced,(yMonth-1),type='classifier',k=k,cv=cv)#variable needs to start at 0
    results_hd=scoreKnnResults(x_reduced,yHd,type='classifier',k=k,cv=cv)
    results_temp=scoreKnnResults(x_reduced,yTemp,type='regressor',k=k,cv=cv)
    
    #preditionDetermistic
    preditionDetermistic.append(results_wd['predD'])
    preditionDetermistic.append(results_day['predD'])
    preditionDetermistic.append(results_month['predD'])
    preditionDetermistic.append(np.mean(np.array(results_hd['predP'])[indicesHd]))#score uniquement pour les jours feriés, pas pour les jours normaux
    preditionDetermistic.append(results_temp['predD'])
    
    #F1 score
    F1score.append(results_wd['F1'])
    F1score.append(results_day['F1'])
    F1score.append(results_month['F1'])
    F1score.append(results_hd['F1'])
    F1score.append(-99999)
    
    #probScore
    probScore_wd=results_wd['predP']
    probScore_day=results_day['predP']
    probScore_month=results_month['predP']
    probScore_Hd=np.array(results_hd['predP'])#[indicesHd]
    
    #preditionProbabilistic
    preditionProbabilistic.append(np.mean(probScore_wd))
    preditionProbabilistic.append(np.mean(probScore_day))
    preditionProbabilistic.append(np.mean(probScore_month))
    preditionProbabilistic.append(np.mean(probScore_Hd))
    preditionProbabilistic.append(results_temp['predD'])
    
    #anomalies dans les predictions
    indicesOddWeekdays=np.array([i for i in range(0, nPoints) if probScore_wd[i] <=0.3])
    oddWeekdays=[]
    if(len(indicesOddWeekdays)>=1):
        oddWeekdays=calendar_info['ds'][indicesOddWeekdays]
        
    #indicesOddHolidays=np.array([i for i in range(0, len(probScore_Hd)) if probScore_Hd[i] <=0.3])
    indicesOddHolidays=np.array([i for i in range(0, nPoints) if probScore_Hd[i] <=0.3])
    oddHolidays=[]
    if(len(indicesOddHolidays)>=1):
        #oddHolidays=calendar_info['ds'][indicesHd[indicesOddHolidays]]
        oddHolidays=calendar_info['ds'][indicesOddHolidays]
    
    knn_temp = KNeighborsRegressor(n_neighbors=k)   
    predictions = cross_val_predict(knn_temp, x_reduced, yTemp, cv=10)
    error=np.abs(predictions-yTemp)
    stdPercentile=np.percentile(error, 95, axis=0)
    print(stdPercentile)
    indicesTemp=np.array([i for i in range(0, nPoints) if error[i] >=stdPercentile])
    oddTemp=[]
    if(len(indicesTemp)>=1):
        oddTemp=calendar_info['ds'][indicesTemp]
        
    #prediction std
    predictionStd.append(np.std(probScore_wd))
    predictionStd.append(np.std(probScore_day))
    predictionStd.append(np.std(probScore_month))
    predictionStd.append(np.std(probScore_Hd))
    predictionStd.append(0)#predictionStd.append(np.std(np.abs(yTemp-knn_temp.score(x_reduced, yTemp))))
    
    #predictionRandom
    knn_random = KNeighborsClassifier(n_neighbors=k)
    x_reduced_random=np.random.rand(nPoints,1)
    predictionRandom.append(np.mean(cross_val_score(knn_random, x_reduced_random, yWeekday, cv=cv,scoring='f1_macro')))
    
    predictionRandom.append(np.mean(cross_val_score(knn_random, x_reduced_random, yWkday, cv=cv)))
    
    predictionRandom.append(np.mean(cross_val_score(knn_random, x_reduced_random, yMonth, cv=cv)))
    
    proba = np.array(cross_val_predict(knn_random, x_reduced_random, yHd, cv=cv, method='predict_proba'))
    probScore=[proba[i][yHd[i]] for i in range(0,len(yHd)) ]
    probScoreHd=np.array(probScore)[indicesHd]
    predictionRandom.append(np.mean(probScoreHd))
    
    knn_random = KNeighborsRegressor(n_neighbors=k)
    predictionRandom.append(np.mean(cross_val_score(knn_random, x_reduced_random, yTemp, cv=cv)))
       
    
    #creation d'une dataFrame pour les résultats
    modelScores=preditionDetermistic
    modelScores[0]=results_wd['F1'] #F1 score for is weekday
    df = pd.DataFrame(data=np.array([modelScores]),columns=['is_weekday','weekday','month','is_holiday_day','temperature'])
    randomScores=predictionRandom
    df.loc[1]=np.array(randomScores)
    #df.append(list(preditionProbabilistic), ignore_index=True)
    #df=df.rename(index={0:'latent deterministic',1:'F1 score',2:'latent probabilistic', 3:'std',4:'random'})
    df=df.rename(index={0:'score model',1:'random model'})
    
    print(df)
    return({'dataFrame':df,'oddWeekdays':oddWeekdays,'oddHolidays':oddHolidays,'oddTemp':oddTemp})




def disentanglement_quantification(x_reduced, factorMatrix, factorDesc, algorithm='RandomForest', cv=3, normalize_information=False):
    """criteria based on "A Framework for the Quantitative Evaluation of Disentangled Representations", Eastwood and Williams (2018)
    
    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of conditions names and types 
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    algorithm -- the kind of estimator to make predictions with
    cv -- int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
    normalize_information -- Boolean, whether to normalize informativeness results with the minimum obtained with a random projection
    
    :return: final_evaluation -- dict, dict of metrics values
             importance_matrix -- array-like, importance matrix for latent dimensions (rows) to predict factors (columns)
    """
    assert algorithm == 'RandomForest' or algorithm == 'GradientBoosting'
    if algorithm == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier as clf
        from sklearn.ensemble import RandomForestRegressor as reg
    else:
        from sklearn.ensemble import GradientBoostingClassifier as clf
        from sklearn.ensemble import GradientBoostingRegressor as reg

    z_dim = x_reduced.shape[1]
    n_factors = factorMatrix.shape[1]
    evaluation = {}
    evaluation['informativeness']={}
    evaluation['importance_variable']={}
    final_evaluation = {}
    #estimation of importance of the latent code variables for each factor using random forest attribut of feature importances
    for i,name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if(factor_type=='category'):
            estimator = clf(n_estimators=100)
            cv_results = cross_validate(estimator, x_reduced, factorMatrix[:,i], cv=cv, return_estimator=True, scoring='f1_macro')
        else:
            estimator = reg(n_estimators=100)
            cv_results = cross_validate(estimator, x_reduced, factorMatrix[:,i], cv=cv, return_estimator=True, scoring='r2')

        if normalize_information:
            x_reduced_random=np.random.rand(x_reduced.shape[0],1)
            if(factor_type=='category'):
                estimator_random = clf(n_estimators=100)
                cv_results_random = cross_validate(estimator_random, x_reduced_random, factorMatrix[:,i], cv=cv, return_estimator=True, scoring='f1_macro')
            else:
                estimator_random = reg(n_estimators=100)
                cv_results_random = cross_validate(estimator_random, x_reduced_random, factorMatrix[:,i], cv=cv, return_estimator=True, scoring='r2')

        if normalize_information:
            min_info = np.mean(cv_results_random['test_score'])
            results_info = (np.mean(cv_results['test_score']) - min_info) / (1 - min_info)
        else:
            results_info = np.mean(cv_results['test_score'])

        evaluation['informativeness'][name]= max(results_info, 0)
        importance_P = np.concatenate([esti.feature_importances_.reshape(-1,1) for esti in cv_results['estimator']], axis=1)
        evaluation['importance_variable'][name]=np.mean(importance_P, axis=1)

    final_evaluation['informativeness'] = np.asarray([evaluation['informativeness'][name] for name in factorDesc.keys()])

    importance_matrix = np.concatenate([evaluation['importance_variable'][name].reshape(-1,1) for name in factorDesc.keys()], axis=1)
    importance_matrix_norm = np.apply_along_axis(lambda x:x/np.sum(x), 1, importance_matrix)
    disentangled_measures = 1 + np.sum(importance_matrix_norm * np.log(importance_matrix_norm+1e-10)/np.log(n_factors),axis=1)
    compactness_measures = 1 + np.sum(importance_matrix * np.log(importance_matrix+1e-10)/np.log(z_dim),axis=0)

    weighted_disentanglement = sum(disentangled_measures*np.sum(importance_matrix_norm, axis=1)/np.sum(importance_matrix_norm))

    final_evaluation['disentanglement'] = disentangled_measures.ravel()
    final_evaluation['compactness'] = compactness_measures.ravel()
    final_evaluation['mean_disentanglement'] = weighted_disentanglement

    return final_evaluation, importance_matrix

def compute_mig(x_reduced, factorMatrix, factorDesc, batch=None):

    """criterion Mutual Information Gap implementation based on "Isolating Sources of Disentanglement in Variational Autoencoders", Chen (2018); 
       inspiration from disentanglement_lib of Olivier Bachem.

    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of conditions names and types 
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    batch -- whether to compute the MIG on a sliced part of the latent representation
    
    :return: mig -- float, MIG average value across the factors
    """

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression

    if batch is None:
        train_size = x_reduced.shape[0]
    else:
        train_size = batch
    sample_index = np.random.choice(range(train_size), size=train_size, replace=False)
    latent = x_reduced[sample_index,  :]
    ys = factorMatrix[sample_index, :]

    m = np.zeros((x_reduced.shape[1], factorMatrix.shape[1]))
    entropy=np.zeros(ys.shape[1])
    for j,name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if(factor_type=='category'):
            m[:,j] = mutual_info_classif(latent, ys[:,j]).T
            entropy[j] = mutual_info_classif(ys[:,j].reshape(-1,1), ys[:,j]).ravel()
        else:
            m[:,j] = mutual_info_regression(latent, ys[:,j]).T
            entropy[j] = mutual_info_regression(ys[:,j].reshape(-1,1), ys[:,j]).ravel()

    sorted_m = np.sort(m, axis=0)[::-1]
    mig = np.mean(np.divide(sorted_m[0,:]-sorted_m[1,:], entropy))

    return mig

def compute_modularity(x_reduced, factorMatrix, factorDesc, batch=None):
    """criterion Modularity based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss", Ridgeway and Mozer (2018); 
        inspiration from disentanglement_lib of Olivier Bachem.
    
    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of conditions names and types 
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    batch -- whether to compute the MIG on a sliced part of the latent representation
    
    :return: modularity -- float, modularity score for the representation
    """

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression

    if batch is None:
        train_size = x_reduced.shape[0]
    else:
        train_size = batch
    sample_index = np.random.choice(range(train_size), size=train_size, replace=False)
    latent = x_reduced[sample_index,  :]
    ys = factorMatrix[sample_index, :]

    m = np.zeros((x_reduced.shape[1], factorMatrix.shape[1]))
    entropy=np.zeros(ys.shape[1])
    for j,name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if(factor_type=='category'):
            m[:,j] = mutual_info_classif(latent, ys[:,j]).T
        else:
            m[:,j] = mutual_info_regression(latent, ys[:,j]).T

    sorted_m = np.r_[[np.eye(1, m.shape[1],k).ravel() for k in np.argmax(m, axis=1)]]
    t_i = m * sorted_m

    d_i = np.sum(np.square(m-t_i), axis=1) / np.square(np.max(m, axis=1)) / (factorMatrix.shape[1] -1)

    return 1 - d_i


def evaluateLatentCode(x_reduced, factorMatrix, factorDesc, algorithm='RandomForest', cv=3, orthogonalize = True, normalize_information=False):
    """ function to return a dict of implemented metrics which are informativeness, compactness, disentanglement, MIG and modularity
    
    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of conditions names and types 
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    algorithm -- the kind of estimator to make predictions with
    cv -- int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
    orthogonalize -- Boolean, whether to fix the explicative axes of the representation on the coordinates dimensions
    normalize_information -- Boolean, whether to normalize informtiveness results with the minimum obtained with a random projection
    
    :return: final_evaluation -- dict, dict of metrics values
             importance_matrix -- array-like, importance matrix for latent dimensions (rows) to predict factors (columns)
    """
    if orthogonalize:
        from sklearn.decomposition import PCA
        ortho_proj = PCA(x_reduced.shape[1])
        x = ortho_proj.fit_transform(x_reduced)
    else:
        x = x_reduced

    final_evaluation, importance_matrix = disentanglement_quantification(x, factorMatrix, factorDesc, cv=3, normalize_information=normalize_information)
    final_evaluation['mig'] = compute_mig(x, factorMatrix, factorDesc)
    final_evaluation['modularity'] = compute_modularity(x, factorMatrix, factorDesc)

    return final_evaluation, importance_matrix





