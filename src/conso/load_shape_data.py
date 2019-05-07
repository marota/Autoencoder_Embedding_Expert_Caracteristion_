import os
import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def change_granularity(data_conso_df, granularity = "1H"):

    if granularity not in ["1H", "15min", "30min"]:
        print('"granularity" must be in ["1H", "15min", "30min"]')
        return

    minutes = np.array(data_conso_df.ds.dt.minute)

    if granularity == "1H":
        mask = np.where(minutes == 0)[0]
    if granularity == "30min":
        mask = np.where((minutes == 30) | (minutes == 0))[0]
    if granularity == "15min":
        mask = np.array(data_conso_df.index)

    data_conso_new_granu_df = data_conso_df.loc[mask].copy()
    data_conso_new_granu_df = data_conso_new_granu_df.reset_index(drop=True)

    return data_conso_new_granu_df


def get_x_conso(data_conso_df, dict_colnames_conso):

    # Get one hot encoding of calendar informations (hour, day, month)
    timeserie = data_conso_df.ds
    weekday = timeserie.dt.weekday
    month = timeserie.dt.month
    hour = timeserie.dt.hour
    minute = timeserie.dt.minute

    calendar_ds = pd.DataFrame({'month': month, 'weekday': weekday, 'hour': hour, 'minute': minute, 'ds': timeserie})

    # One hot encoding
    encoded_weekday = pd.get_dummies(calendar_ds['weekday'], prefix="weekday")
    encoded_month = pd.get_dummies(calendar_ds['month'], prefix="month")
    encoded_hour = pd.get_dummies(calendar_ds['hour'], prefix="hour")
    encoded_minute = pd.get_dummies(calendar_ds['minute'], prefix="minute")

    # Check time_step
    timedelta = (timeserie[1] - timeserie[0]).seconds / (60 * 15)
    nb_columns_encoded_minute = encoded_minute.shape[1]

    expected_dim = {4: 1, 2: 2, 1: 4}
    assert expected_dim[nb_columns_encoded_minute] == timedelta

    if nb_columns_encoded_minute == 1:
        calendar_encoded_ds = pd.concat([encoded_weekday, encoded_month, encoded_hour, timeserie], axis=1)
    else:
        calendar_encoded_ds = pd.concat([encoded_weekday, encoded_month, encoded_hour, encoded_minute, timeserie],
                                        axis=1)

    dict_colnames_conso['calendar'] = [el for el in calendar_encoded_ds.columns if el != 'ds']

    # Merge conso and meteo
    x_conso = pd.merge(data_conso_df, calendar_encoded_ds, on='ds', how='left')
    x_conso = x_conso.drop('type_tempo', axis=1)

    return x_conso, dict_colnames_conso


def select_variables(x_conso, dict_colnames_conso, list_variable):
    assert set(list_variable).issubset(set(dict_colnames_conso.keys()))

    mask = ['ds']
    for variable in list_variable:
        mask_variable = [el for el in x_conso.columns if el.startswith(tuple(dict_colnames_conso[variable]))]
        mask += mask_variable

    sorted_mask = [el for el in x_conso if el in mask]

    x_conso_selected_variables = x_conso[sorted_mask].copy()

    return x_conso_selected_variables


def get_x_conso_autoencoder(data_conso_df, dict_colnames_conso):

    x_conso, dict_colnames_conso = get_x_conso(data_conso_df, dict_colnames_conso)

    list_variables = ['conso', 'meteo','holiday_days']
    x_conso = select_variables(x_conso, dict_colnames_conso, list_variables)

    # Keep only average temperature
    x_conso = x_conso.drop([el for el in x_conso.columns if 'Th+0' in el[:8]], axis=1)

    return x_conso


def get_train_test_x_conso(x_conso, date_test_start, date_test_end):
    """
    split the data set in train and test set

    :param x_conso: dataframe
    :param y_conso: dataframe
    :param date_test_start: timestamp of the first day of the test set
    :param date_test_end: timestamp of the last day of the test set
    :return: dataset: dictionary containing the train and test set (x and y)
             dict_ds: dictionary containing the time series of the train and test set
    """

    mask_test = (x_conso.ds >= date_test_start) & (x_conso.ds < date_test_end + datetime.timedelta(days=1))

    x_test = x_conso[mask_test]
    x_train = x_conso[np.invert(mask_test)]

    x_test = x_test.reset_index(drop=True)
    x_train = x_train.reset_index(drop=True)

    dict_ds = {'train': x_train.ds, 'test': x_test.ds}

    dict_xconso = {}
    dict_xconso['train'] = x_train
    dict_xconso['test'] = x_test

    return dict_xconso


def normalize_xconso(dict_xconso, type_scaler = 'standard'):
    """
    Normalization of the needed columns

    :param x_conso:
    :param dict_colnames_conso:
    :return: dataset_scaled
    """

    x_test = None
    dict_xconso_scaled = {}

    if type(dict_xconso) == dict:
        x_train = dict_xconso['train']
        if 'test' in dict_xconso.keys():
            x_test = dict_xconso['test']
    else:
        x_train = dict_xconso

    # Getting columns to normalized
    x_train.columns
    mask_conso = [el for el in x_train.columns if el.startswith('consumption')]
    print(mask_conso)
    mask_meteo = [el for el in x_train.columns if el.startswith('temperature')]

    cols_to_normalized = mask_conso + mask_meteo

    # Fitting scaler on train
    if type_scaler == 'standard':
        scaler = StandardScaler(with_mean=True, with_std=True)
    elif type_scaler == 'minmax':
        scaler = MinMaxScaler()

    scalerfit = scaler.fit(x_train[cols_to_normalized])

    # Applying filter on train
    cols_normalized = scalerfit.transform(x_train[cols_to_normalized])

    x_train_scaled = x_train.copy()
    for i, col_name in enumerate(cols_to_normalized):
        x_train_scaled[col_name] = cols_normalized[:, i]

    dict_xconso_scaled['train'] = x_train_scaled

    if x_test is not None:
    # Applying filter on test
        cols_normalized = scalerfit.transform(x_test[cols_to_normalized])

        x_test_scaled = x_test.copy()
        for i, col_name in enumerate(cols_to_normalized):
            x_test_scaled[col_name] = cols_normalized[:, i]

        dict_xconso_scaled['test'] = x_test_scaled

    return dict_xconso_scaled, scalerfit


def get_x_cond_autoencoder(x_conso, type_x = ['conso'], type_cond = ['month', 'weekday'], data_conso_df = None,slidingWindowSize=0):

    ### X
    x_ds = x_conso.copy()

    # Enumerate days
    x_ds['day'] = (x_ds['ds'] - x_ds['ds'][0]).apply(lambda td: td.days)
    x_ds['minute'] = x_ds['ds'].dt.hour * 100 + x_ds['ds'].dt.minute

    nb_day = len(x_ds['ds'].dt.normalize().unique())

    x_ae = np.zeros((nb_day, 0))
    if not (slidingWindowSize == 0):
        x_ae = np.zeros((x_conso.shape[0] - slidingWindowSize, 0))

    if 'conso' in type_x:
        # pandas pivot
        if (slidingWindowSize==0):
            x = x_ds[['consumption_France', 'day', 'minute']].pivot('day', 'minute')['consumption_France']
        else:
            x = x_ds[['consumption_France']]
            for i in range(1, slidingWindowSize):
                x['consumption_France_shift_' + str(i)] = x['consumption_France'].shift(i)
            x = x.loc[slidingWindowSize:]
            x = x.reset_index(drop=True)

        # Replacing missing values due to the change of hour in march
        # TODO: interpolation for the hour of the given days
        x[x.isna()] = x.values.mean(axis=0)[7]

        # Converting to np.array
        x = x.values

        x_ae = np.concatenate((x_ae, x), axis=1)

    if 'temperature' in type_x:
        # pandas pivot
        if (slidingWindowSize == 0):
            x = x_ds[['temperature_France', 'day', 'minute']].pivot('day', 'minute')['temperature_France']
        else:
            x = x_ds[['temperature_France']]
            for i in range(1, slidingWindowSize):
                x['temperature_France_shift_' + str(i)] = x['temperature_France'].shift(i)
            x = x.loc[slidingWindowSize:]
            x = x.reset_index(drop=True)
        # Replacing missing values due to the change of hour in march
        # TODO: interpolation for the hour of the given days
        x[x.isna()] = x.values.mean(axis=0)[7]

        # Converting to np.array
        x = x.values

        x_ae = np.concatenate((x_ae, x), axis=1)

    # Getting corresponding date of each row
    if (slidingWindowSize == 0):
        ds = x_conso[(x_conso['ds'].dt.hour == 0) & (x_conso['ds'].dt.minute == 0)]['ds']
        ds = ds.reset_index(drop=True)
    else:
        ds = x_conso['ds']
        ds = ds.loc[slidingWindowSize:]
        ds = ds.reset_index(drop=True)

    ### Cond

    cond = get_cond_autoencoder(x_conso, ds, type_cond, data_conso_df)

    assert x_ae.shape[0] == cond.shape[0]

    return x_ae, cond, ds


def get_cond_autoencoder(x_conso, ds, type_cond=['month', 'weekday'], data_conso_df=None):

    # get calendar info
    calendar_info = pd.DataFrame(ds)
    calendar_info['month'] = calendar_info.ds.dt.month
    calendar_info['weekday'] = calendar_info.ds.dt.weekday
    calendar_info['is_weekday'] = (calendar_info.weekday < 5).apply(lambda x: int(x))
    
    # get conditional variables

    list_one_hot = list()

    if 'month' in type_cond:
        # month
        one_hot_month = pd.get_dummies(calendar_info.month, prefix='month')
        list_one_hot.append(one_hot_month)

    if 'weekday' in type_cond:#on considere ici is-weekday
        # weekday
        #one_hot_weekday = pd.get_dummies(calendar_info.is_weekday, prefix='weekday')
        #list_one_hot.append(one_hot_weekday)
        list_one_hot.append(calendar_info.is_weekday)

    if 'day' in type_cond:#on considere ici is-weekday
        # weekday
        one_hot_weekday = pd.get_dummies(calendar_info.weekday, prefix='weekday')
        list_one_hot.append(one_hot_weekday)
    
    if 'holidays' in type_cond:#on considere ici is-weekday
        # weekday
        #one_hot_weekday = pd.get_dummies(calendar_info.is_weekday, prefix='weekday')
        #list_one_hot.append(one_hot_weekday)
        holidays_df= x_conso[['ds', 'is_holiday_day']].copy()
        day_count = (holidays_df['ds'] - holidays_df['ds'][0]).apply(lambda td: td.days)
        holidays_df['day'] = day_count
        daily_holidays__df = pd.DataFrame(holidays_df.groupby(['day']).max())
        list_one_hot.append(daily_holidays__df.is_holiday_day)

    # Continious variable representing the avarage temperature of the day
    if 'temp' in type_cond:
        meteo_nat_df = x_conso[['ds', 'temperature_France']].copy()
        day_count = (meteo_nat_df['ds'] - meteo_nat_df['ds'][0]).apply(lambda td: td.days)
        meteo_nat_df['day'] = day_count

        mean_meteo_nat_df = pd.DataFrame(meteo_nat_df.groupby(['day']).mean())

        scaler = MinMaxScaler()
        scalerfit = scaler.fit(np.array(mean_meteo_nat_df['temperature_France']).reshape(-1, 1))
        cond_temp = scalerfit.transform(np.array(mean_meteo_nat_df['temperature_France']).reshape(-1, 1))
        cond_temp = pd.DataFrame(cond_temp)

        list_one_hot.append(cond_temp)

    # Full temperature profile
    if 'temperature' in type_cond:
        x_ds = x_conso.copy()

        # Enumerate days
        x_ds['day'] = (x_ds['ds'] - x_ds['ds'][0]).apply(lambda td: td.days)
        x_ds['minute'] = x_ds['ds'].dt.hour * 100 + x_ds['ds'].dt.minute

        # pandas pivot
        cond_temp = x_ds[['temperature_France', 'day', 'minute']].pivot('day', 'minute')['temperature_France']

        # Replacing missing values due to the change of hour in march
        # TODO: interpolation for the hour of the given days
        cond_temp[cond_temp.isna()] = cond_temp.values.mean(axis=0)[7]

        list_one_hot.append(cond_temp)

    # get conditional matrix
    cond = pd.concat(list_one_hot , axis=1)
    cond = cond.values
    print(cond.shape)

    return cond

def get_y_autoencoder(x_conso,slidingWindowSize=0):



    ### X
    y_ds = x_conso.copy()
    slidingWindowSize=0

    # Enumerate days
    y_ds['day'] = (y_ds['ds'] - y_ds['ds'][0]).apply(lambda td: td.days)
    y_ds['minute'] = y_ds['ds'].dt.hour * 100 + y_ds['ds'].dt.minute

    nb_day = len(y_ds['ds'].dt.normalize().unique())

    y_ae = np.zeros((nb_day, 0))
    if not (slidingWindowSize == 0):
        y_ae = np.zeros((x_conso.shape[0] - slidingWindowSize, 0))

    # pandas pivot
    if (slidingWindowSize==0):
        y = y_ds[['consumption_France', 'day', 'minute']].pivot('day', 'minute')['consumption_France']
    else:
        y = y_ds[['consumption_France']]
        for i in range(1, slidingWindowSize):
            x['consumption_France_t0_shift_' + str(i)] = x['consumption_France'].shift(i)
        y = y.loc[slidingWindowSize:]
        y = y.reset_index(drop=True)

    # Replacing missing values due to the change of hour in march
    # TODO: interpolation for the hour of the given days
    y[y.isna()] = y.values.mean(axis=0)[7]

    # Converting to np.array
    y = y.values

    y_ae = np.concatenate((y_ae, y), axis=1)
        
    return y_ae
    

def get_dataset_autoencoder(dict_xconso, type_x=['conso'],type_cond=['month', 'weekday'],slidingWindowSize=0, isYNormalized=True,dict_xconso_unormalized=None):

    dataset = {}
    
    
    for key, x_conso_normalized in dict_xconso.items():
        x, cond, cvae_ds = get_x_cond_autoencoder(x_conso=x_conso_normalized, type_x=type_x, type_cond=type_cond,slidingWindowSize=slidingWindowSize)
        
        if(isYNormalized):
            dataset[key] = {'x': [x, cond], 'y': x, 'ds': cvae_ds}
        else:
            x_conso_non_normalized=dict_xconso_unormalized[key]
            y =  get_y_autoencoder(x_conso_non_normalized,slidingWindowSize=0)
            dataset[key] = {'x': [x, cond], 'y': y, 'ds': cvae_ds}

    return dataset
