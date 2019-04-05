import os
import datetime
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_raw_data_conso(path_data_folder):
    """
    Shape the raw consumption data, gather them in a python dictionary and save it as a pickle file on disk.

    :param path_data_folder: Path of the folder containing all the raw files:
                -conso_Y.csv
                -meteoX_T.csv
                -joursFeries.csv
                -Tempo_history_merged.csv
    :return:
    """

    #path_data_folder = os.path.join("/local/home/antorosi/Documents/AutoEncoder/data")

    # CONSUMPTION
    conso_csv = os.path.join(path_data_folder, "conso_Y.csv")
    conso_df = pd.read_csv(conso_csv, sep=";", engine='c', header=0)
    del conso_csv

    conso_df['ds'] = pd.to_datetime(conso_df['date'] + ' ' + conso_df['time'])

    # get only national observation
    conso_df = conso_df[['ds', 'Consommation NAT t0']]
    conso_df.columns = ['ds','conso_nat_t0']

    print('Consumption data loaded')

    # TEMPERATURE
    meteo_csv = os.path.join(path_data_folder, "meteoX_T.csv")
    meteo_df = pd.read_csv(meteo_csv, sep=";", engine='c', header=0)
    meteo_df['ds'] = pd.to_datetime(meteo_df['date'] + ' ' + meteo_df['time'])
    del meteo_csv

    # USELESS NOW
    # time_decay = (conso_df.ds.iloc [-1] - meteo_df.ds.iloc[-1]).seconds/(60*15)
    #
    # if time_decay > 10:
    #     print("meteo time series length does't math the conso one")
    #     return
    # else:
    #     ref = meteo_df.iloc[-1]
    #     for i in range(int(time_decay)):
    #         meteo_df.append(ref)

    # Drop the duplicates (likely due to the change of hour)

    # Correct the last two timestamp manually (like manually manually)
    # ts_ref = meteo_df.ds.iloc[-3]
    # ts_2 = ts_ref + datetime.timedelta(minutes=15)
    # ts_1 = ts_2 + datetime.timedelta(minutes=15)
    # meteo_df.loc[meteo_df.index.values[-2],'ds'] = ts_2
    # meteo_df.loc[meteo_df.index.values[-1],'ds'] = ts_1

    meteo_df = meteo_df.drop_duplicates(subset='ds',keep='last')

    # get observation only
    meteo_df = meteo_df[list(meteo_df.columns[meteo_df.columns.str.endswith('Th+0')]) + ['ds']]
    stationColumns = meteo_df.columns[list(meteo_df.columns.str.endswith('Th+0'))]
    meteo_df['meteo_natTh+0'] = meteo_df[stationColumns].mean(axis=1)

    print('Meteo data loaded')

    # HOLIDAY DAYS
    holiday_days_csv = os.path.join(path_data_folder, "joursFeries.csv")
    holiday_days_df = pd.read_csv(holiday_days_csv, sep=";")
    holiday_days_df.ds = pd.to_datetime(holiday_days_df.ds)

    # Putting dayly label to hourly label
    # TODO: Find a better, vectorized solution
    day = holiday_days_df.ds[0]
    start_day = day
    end_day = day + pd.DateOffset(hour=23) + pd.DateOffset(minute=45)
    day_hourly = pd.date_range(start_day, end_day, freq='15min')

    for day in holiday_days_df.ds[1:]:
        start_day = day
        end_day = day + pd.DateOffset(hour=23) + pd.DateOffset(minute=45)
        day_hourly = day_hourly.append(pd.date_range(start_day, end_day, freq='15min'))

    day_hourly.name = 'ds'
    holiday_days_df = holiday_days_df.set_index('ds')
    holiday_days_df = holiday_days_df.reindex(day_hourly, method="ffill")
    holiday_days_df = holiday_days_df.reset_index()

    holiday_days_df.columns = ['ds', 'type_holiday']

    print('Holiday Days data loaded')

    # TEMPO DAYS
    tempo_csv = os.path.join(path_data_folder, "Tempo_history_merged.csv")
    tempo_df = pd.read_csv(tempo_csv, sep=";")
    tempo_df['ds'] = pd.to_datetime(tempo_df.Date)
    tempo_df.drop(['Date'], axis=1, inplace=True)

    # Putting dayly label to hourly label
    start_day = min(tempo_df.ds)
    end_day = max(tempo_df.ds) + pd.DateOffset(hour=23) + pd.DateOffset(minute=45)
    hourly_tf = pd.date_range(start_day,end_day,freq='15min')

    hourly_tf.name='ds'
    tempo_df = tempo_df.set_index('ds')
    tempo_df = tempo_df.reindex(hourly_tf, method='ffill')
    tempo_df = tempo_df.reset_index()

    tempo_df.columns =  ['ds', 'type_tempo']

    print('Tempo data loaded')

    # Gathering data into dictionnary
    dict_data_conso = {'conso':conso_df, 'meteo':meteo_df, 'holiday_days':holiday_days_df, 'tempo':tempo_df}

    # Saving dict
    with open(os.path.join(path_data_folder, 'dict_data_conso.pickle'), 'wb') as file:
        pickle.dump(dict_data_conso, file, protocol=pickle.HIGHEST_PROTOCOL)

    print('Shaped data saved in {}'.format(os.path.join(path_data_folder, 'dict_data_conso.pickle')))


def load_data_conso(path_data_folder):
    """
    Load a dictionnary containing all the needed data related to consumption prediction

    :param path_data_folder: path of the folder containing the data
    :return: python dictoinnary
    """

    # Checking if the dict containing conso data already exists

    if not os.path.exists(os.path.join(path_data_folder, 'dict_data_conso.pickle')):
        load_raw_data_conso(path_data_folder)

    with open(os.path.join(path_data_folder, 'dict_data_conso.pickle'), 'rb') as f:
        dict_data_conso = pickle.load(f)

    return dict_data_conso


def get_uniformed_data_conso(dict_data_conso):
    """
    Put the data from dict_data_conso in the same dataframe.
    Allows to 'uniform' the data as depending on the source some days or hours are skipped (mostly du to the change of hours).
    The taken reference is the time series from the consumption.

    :param dict_data_conso:
    :return:
    """

    dict_colnames_conso = {}

    for key, df in dict_data_conso.items():
        dict_colnames_conso[key] = [el for el in df.columns if el!='ds']

    data_conso_df = dict_data_conso['conso']. copy()
    data_conso_df = pd.merge(data_conso_df, dict_data_conso['meteo'], on='ds', how='left')
    data_conso_df = pd.merge(data_conso_df, dict_data_conso['tempo'], on='ds', how='left')

    # formating holiday days to be boolean
    hd_ds = dict_data_conso['holiday_days'].copy()
    hd_ds['is_holiday_day'] = np.array(hd_ds['type_holiday']).astype('bool').astype('int')
    data_conso_df = pd.merge(data_conso_df, hd_ds[['ds','is_holiday_day']], on='ds', how='left')
    pd.set_option('chained_assignment', None) # To avoid message about chaine assignment, necessary here
    data_conso_df['is_holiday_day'].loc[data_conso_df['is_holiday_day'].isna()] = 0
    pd.set_option('chained_assignment', 'warn')

    dict_colnames_conso['holiday_days'] = ['is_holiday_day']

    if 'atypical_events' in dict_data_conso.keys():
        ae_ds = dict_data_conso['atypical_events'].copy()
        data_conso_df = pd.merge(data_conso_df, ae_ds, on='ds', how='left')
        pd.set_option('chained_assignment', None)  # To avoid message about chained assignment, necessary here
        data_conso_df['is_atypical'].loc[data_conso_df['is_atypical'].isna()] = 0
        pd.set_option('chained_assignment', 'warn')

        dict_colnames_conso['atypical_events'] = ['is_atypical']

    return data_conso_df, dict_colnames_conso


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


def normalize_xconso(dict_xconso, dict_colnames_conso, type_scaler = 'standard'):
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
    mask_conso = [el for el in x_train.columns if el.startswith(tuple(dict_colnames_conso['conso']))]
    print(mask_conso)
    mask_meteo = [el for el in x_train.columns if el.startswith(tuple(dict_colnames_conso['meteo']))]

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
            x = x_ds[['conso_nat_t0', 'day', 'minute']].pivot('day', 'minute')['conso_nat_t0']
        else:
            x = x_ds[['conso_nat_t0']]
            for i in range(1, slidingWindowSize):
                x['conso_nat_t0_shift_' + str(i)] = x['conso_nat_t0'].shift(i)
            x = x.loc[slidingWindowSize:]
            x = x.reset_index(drop=True)

        # Replacing missing values due to the change of hour in march
        # TODO: interpolation for the hour of the given days
        x[x.isna()] = x.as_matrix().mean(axis=0)[7]

        # Converting to np.array
        x = x.as_matrix()

        x_ae = np.concatenate((x_ae, x), axis=1)

    if 'temperature' in type_x:
        # pandas pivot
        if (slidingWindowSize == 0):
            x = x_ds[['meteo_natTh+0', 'day', 'minute']].pivot('day', 'minute')['meteo_natTh+0']
        else:
            x = x_ds[['meteo_natTh+0']]
            for i in range(1, slidingWindowSize):
                x['meteo_natTh+0_shift_' + str(i)] = x['meteo_natTh+0'].shift(i)
            x = x.loc[slidingWindowSize:]
            x = x.reset_index(drop=True)
        # Replacing missing values due to the change of hour in march
        # TODO: interpolation for the hour of the given days
        x[x.isna()] = x.as_matrix().mean(axis=0)[7]

        # Converting to np.array
        x = x.as_matrix()

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
        meteo_nat_df = x_conso[['ds', 'meteo_natTh+0']].copy()
        day_count = (meteo_nat_df['ds'] - meteo_nat_df['ds'][0]).apply(lambda td: td.days)
        meteo_nat_df['day'] = day_count

        mean_meteo_nat_df = pd.DataFrame(meteo_nat_df.groupby(['day']).mean())

        scaler = MinMaxScaler()
        scalerfit = scaler.fit(np.array(mean_meteo_nat_df['meteo_natTh+0']).reshape(-1, 1))
        cond_temp = scalerfit.transform(np.array(mean_meteo_nat_df['meteo_natTh+0']).reshape(-1, 1))
        cond_temp = pd.DataFrame(cond_temp)

        list_one_hot.append(cond_temp)

    # Full temperature profile
    if 'temperature' in type_cond:
        x_ds = x_conso.copy()

        # Enumerate days
        x_ds['day'] = (x_ds['ds'] - x_ds['ds'][0]).apply(lambda td: td.days)
        x_ds['minute'] = x_ds['ds'].dt.hour * 100 + x_ds['ds'].dt.minute

        # pandas pivot
        cond_temp = x_ds[['meteo_natTh+0', 'day', 'minute']].pivot('day', 'minute')['meteo_natTh+0']

        # Replacing missing values due to the change of hour in march
        # TODO: interpolation for the hour of the given days
        cond_temp[cond_temp.isna()] = cond_temp.as_matrix().mean(axis=0)[7]

        list_one_hot.append(cond_temp)

    # get conditional matrix
    cond = pd.concat(list_one_hot , axis=1)
    cond = cond.as_matrix()
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
        y = y_ds[['conso_nat_t0', 'day', 'minute']].pivot('day', 'minute')['conso_nat_t0']
    else:
        y = y_ds[['conso_nat_t0']]
        for i in range(1, slidingWindowSize):
            x['conso_nat_t0_shift_' + str(i)] = x['conso_nat_t0'].shift(i)
        y = y.loc[slidingWindowSize:]
        y = y.reset_index(drop=True)

    # Replacing missing values due to the change of hour in march
    # TODO: interpolation for the hour of the given days
    y[y.isna()] = y.as_matrix().mean(axis=0)[7]

    # Converting to np.array
    y = y.as_matrix()

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
