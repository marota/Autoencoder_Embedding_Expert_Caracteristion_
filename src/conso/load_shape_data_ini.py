import os
import datetime
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt

def enumerate_days(ds):
    """
    :param df: dataframe containing a columns ds with the time series
    :return: The same data frame with a new columns 'day' containing the day indice
    """

    diff = (ds - ds[0]).apply(lambda td: td.days)

    return diff

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def conso_ds_to_array(Xinput_ds):

    # Adding neccessary new columns
    X_ds = Xinput_ds.copy()
    X_ds['day'] = enumerate_days(X_ds['ds'])
    X_ds['minute'] = X_ds.ds.dt.hour * 100 + Xinput_ds.ds.dt.minute

    # pandas pivot
    X = X_ds[['Consommation NAT t0', 'day', 'minute']].pivot('day', 'minute')['Consommation NAT t0']

    # Replacing missing values due to the change of hour in march
    X[X.isna()] = X.as_matrix().mean(axis=0)[7]

    # Converting to np.array
    X = X.as_matrix()

    # Getting corresponding date of each row
    ds = Xinput_ds[(Xinput_ds.ds.dt.hour == 0) & (Xinput_ds.ds.dt.minute == 0)].ds
    ds = ds.reset_index(drop=True)

    return X, ds

def plot_conso_day(date, X, ds):

    minute_dict = {1: 60, 2: 30, 4: 15}
    date_dt = datetime.datetime(date.year, date.month, date.day)

    # Get conso
    ind = int(np.where(ds == date_dt)[0])
    conso_day = X[ind, :]

    # Get timestamp
    time_step = len(conso_day) / 24
    time_step = datetime.timedelta(minutes=minute_dict[time_step])

    date_day_ds = [dt for dt in datetime_range(date_dt, date_dt + datetime.timedelta(days=1), time_step)]

    plt.plot(date_day_ds, conso_day)
    plt.show()

if __name__ == "__main__":

    # data path
    path_data_folder = os.path.join("/local/home/antorosi/Documents/AutoEncoder/data")

    Xinput_loaded = False

    if os.path.exists(os.path.join(path_data_folder, "Xinput.pickle")):
        print("Loading existing Xinput files")
        with open(os.path.join(path_data_folder, "Xinput.pickle"), "rb") as f:
            Xinput = pickle.load(f)
        Xinput_loaded = True


    if not Xinput_loaded:
        #####################################################################################
        # DATA LOADING
        #####################################################################################
        beg_date = datetime.datetime.now()
        print("Loading begins at {:%Y-%m-%d %H:%M:%S}".format(beg_date))

        # CONSUMPTION
        conso_csv = os.path.join(path_data_folder, "conso_Y.csv")
        conso_df = pd.read_csv(conso_csv, sep=";", engine='c', header=0)

        conso_df['ds'] = pd.to_datetime(conso_df['date'] + ' ' + conso_df['time'])

        # get only national observation
        consoFrance_df = conso_df[['ds', 'Consommation NAT t0']].copy()


        # CALENDAR VARIABLES
        xCalendaire_csv = os.path.join(path_data_folder, "variablesCalendaires.csv")
        xCalendaire_df = pd.read_csv(xCalendaire_csv, sep=";")

        xCalendaire_df['ds'] = pd.to_datetime(xCalendaire_df.date)
        xCalendaire_df.drop(['date'], axis=1, inplace=True)

        print("Calendar data loaded")

        # HOLIDAY DAYS
        jours_feries_csv = os.path.join(path_data_folder, "joursFeries.csv")
        jours_feries_df = pd.read_csv(jours_feries_csv, sep=";")
        jours_feries_df.ds = pd.to_datetime(jours_feries_df.ds)

        # Putting dayly label to hourly label
        # TODO: Find a better, vectorized solution
        day = jours_feries_df.ds[0]
        start_day = day
        end_day = day + pd.DateOffset(hour=23)
        day_hourly = pd.date_range(start_day, end_day, freq='H')

        for day in jours_feries_df.ds[1:]:
            start_day = day
            end_day = day + pd.DateOffset(hour=23)
            day_hourly = day_hourly.append(pd.date_range(start_day, end_day, freq='H'))

        day_hourly.name = 'ds'
        jours_feries_df = jours_feries_df.set_index('ds')
        jours_feries_df = jours_feries_df.reindex(day_hourly, method="ffill")
        jours_feries_df = jours_feries_df.reset_index()

        print("Holiday days loaded")
        #
        # # TEMPO DAYS
        # tempo_csv = os.path.join(path_data_folder, "Tempo_history_merged.csv")
        # tempo_df = pd.read_csv(tempo_csv, sep=";")
        # tempo_df['ds'] = pd.to_datetime(tempo_df.Date)
        # tempo_df.drop(['Date'], axis=1, inplace=True)
        #
        # # Putting dayly label to hourly label
        # start_day = min(tempo_df.ds)
        # end_day = max(tempo_df.ds) + pd.DateOffset(hour=23)
        # hourly_tf = pd.date_range(start_day,end_day,freq='H')
        #
        # hourly_tf.name='ds'
        # tempo_df=tempo_df.set_index('ds')
        # tempo_df = tempo_df.reindex(hourly_tf, method='ffill')
        # tempo_df = tempo_df.reset_index()
        #
        # print("TEMPO days loaded")
        #
        # end_date = datetime.datetime.now()
        # print("All data loaded at {:%Y-%m-%d %H:%M:%S}".format(end_date))
        # print("It took {:.0f}s to load the data".format((end_date-beg_date).total_seconds()))

        #####################################################################################
        # DATA SHAPING
        #####################################################################################

        print("Begin data shaping")

        # TO DO: check data consistency: ts of last meteo observations are not consistent

        # Remark: ordering of the operations is important in order to keep the time consistency in the df
        Xinput = consoFrance_df.copy()

        # Creation of conso J-1
        Xinput['Conso_J_1'] = Xinput['Consommation NAT t0'].shift(96)

        # Creation of variables hour, week
        #  day and month
        time = consoFrance_df.ds
        weekday = time.dt.weekday
        month = time.dt.month

        Xcalendar = pd.DataFrame({'month': month, 'weekday': weekday, 'ds':time})

        # One hot encoding
        encodedWeekDay = pd.get_dummies(Xcalendar['weekday'], prefix="weekday")
        encodedMonth = pd.get_dummies(Xcalendar['month'], prefix="month")
        Xcalendar_oneHot = pd.concat([encodedMonth, encodedWeekDay, Xcalendar['ds']], axis=1)

        Xinput = pd.merge(Xinput, Xcalendar_oneHot, on='ds', how='inner')

        # Creation of the Holidays day
        holidayDays = pd.merge(consoFrance_df, jours_feries_df, how="left", on="ds")
        encodeHolidayDays = pd.get_dummies(holidayDays['holiday'], prefix="HD")
        encodeHolidayDays['HD_HolidayDay'] = encodeHolidayDays.sum(axis=1)
        encodeHolidayDays['ds'] = consoFrance_df.ds
        #
        # # Addition of the Holiday Day with one day lag
        # encodeHolidayDays_J_1 = encodeHolidayDays.copy()
        # encodeHolidayDays_J_1.columns = [el+"_J_1" for el in encodeHolidayDays.columns]
        # encodeHolidayDays_J_1 = encodeHolidayDays_J_1.rename(columns={'ds_J_1': 'ds'})
        # mask_ColHD = [el for el in encodeHolidayDays_J_1 if 'HD' in el]
        # encodeHolidayDays_J_1[mask_ColHD] = encodeHolidayDays_J_1[mask_ColHD].shift(24)
        # encodeHolidayDays = pd.merge(encodeHolidayDays, encodeHolidayDays_J_1, on = 'ds', how = 'left')
        XholidayDays = encodeHolidayDays[['ds', 'HD_HolidayDay']]

        # Merge all input in Xinput
        Xinput = pd.merge(Xinput, XholidayDays, on='ds', how='inner')

        end_date = datetime.datetime.now()

        save = True
        if save:
            with open(os.path.join(path_data_folder, 'Xinput.pickle'), 'wb') as file:
                pickle.dump(Xinput, file, protocol=pickle.HIGHEST_PROTOCOL)
