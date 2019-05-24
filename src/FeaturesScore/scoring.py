from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def build():
   print("building")

def scoreKnnResults(x,y,type='classifier',k=5,cv=10):
    knn = KNeighborsRegressor(n_neighbors=k)
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
    
    temperatureMax=[max(xconso.loc[np.where(xconso['ds'].dt.date==dates[k]),columns_x[temp_idx]]) for k in range(dates.shape[0])]
    temperatureMean=[np.mean(xconso.loc[np.where(xconso['ds'].dt.date==dates[k]),columns_x[temp_idx]]) for k in range(dates.shape[0])]
    
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

