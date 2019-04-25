import os
import numpy as np
from matplotlib import pyplot as plt
import datetime

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
from plotly import tools


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



def pyplot_latent_space_projection(x_proj, calendar_info, path_folder_out, name=None, size_fig=(17,15)):
    """

    :param x_proj:
    :param calendar_info:
    :param path_folder_out:
    :param name:
    :return:
    """

    #Different possible colormap: nipy_spectral, plasma, viridis

    month = np.array(calendar_info.month)

    mask_isweekday = False
    mask_ishd = False

    if 'is_weekday' in calendar_info.columns:
        mask_isweekday = calendar_info.is_weekday.astype('bool')


    if 'is_hd' in calendar_info.columns:
        mask_ishd = calendar_info.is_hd.astype('bool')

    plt.figure(figsize=size_fig)
    plt.scatter(x_proj[mask_isweekday, 0], x_proj[mask_isweekday, 1], marker='.', lw=2,
                c=month[mask_isweekday], cmap=plt.cm.get_cmap('nipy_spectral', 12), label='Week days')
    plt.scatter(x_proj[np.invert(mask_isweekday), 0], x_proj[np.invert(mask_isweekday), 1], marker='+', lw=2,
                c=month[np.invert(mask_isweekday)], cmap=plt.cm.get_cmap('nipy_spectral', 12), label='Weekend')

    plt.colorbar(ticks=range(0, 12), label='Month')
    plt.clim(-0.5, 11.5)
    plt.scatter(x_proj[mask_ishd, 0], x_proj[mask_ishd, 1], marker='x', lw=2,
                c='black', label='Holiday Days')

    plt.legend()
    plt.title('Projection on the latent space')

    if name is None:
        name = 'latent_space_proj'

    if path_folder_out is not None:
        plt.savefig(os.path.join(path_folder_out, name + '.png'))

    plt.show()


def pyplot_latent_space_projection_temp(x_proj, calendar_info, temp, path_folder_out=None, name=None):
    """

    :param x_proj:
    :param calendar_info:
    :param path_folder_out:
    :param name:
    :return:
    """

    #Different possible colormap: seismic

    mask_isweekday = False
    mask_ishd = False

    if 'is_weekday' in calendar_info.columns:
        mask_isweekday = calendar_info.is_weekday.astype('bool')

    if 'is_hd' in calendar_info.columns:
        mask_ishd = calendar_info.is_hd.astype('bool')

    plt.figure(figsize=(17, 15))
    plt.scatter(x_proj[mask_isweekday, 0], x_proj[mask_isweekday, 1], marker='.', lw=2,
                c=temp[mask_isweekday], cmap=plt.cm.get_cmap('seismic'), label='Week days')
    plt.scatter(x_proj[np.invert(mask_isweekday), 0], x_proj[np.invert(mask_isweekday), 1], marker='+', lw=2,
                c=temp[np.invert(mask_isweekday)], cmap=plt.cm.get_cmap('seismic'), label='Weekend')

    plt.colorbar(label='Temp')
    #plt.clim(-0.5, 11.5)
    plt.scatter(x_proj[mask_ishd, 0], x_proj[mask_ishd, 1], marker='x', lw=2,
                c='black', label='Holiday Days')

    plt.legend()
    plt.title('Projection on the latent space')

    if name is None:
        name = 'latent_space_proj'

    if path_folder_out is not None:
        plt.savefig(os.path.join(path_folder_out, name + '.png'))

    plt.show()


def pyplot_latent_space_projection_error(x_proj, calendar_info, error, color=None, path_folder_out=None, name=None):
    """

    :param x_proj:
    :param calendar_info:
    :param path_folder_out:
    :param name:
    :return:
    """

    #Different possible colormap: seismic

    mask_isweekday = False
    mask_ishd = False

    if color is None:
        cmap_color = 'seismic'
    else:
        cmap_color = color

    if 'is_weekday' in calendar_info.columns:
        mask_isweekday = calendar_info.is_weekday.astype('bool')

    if 'is_hd' in calendar_info.columns:
        mask_ishd = calendar_info.is_hd.astype('bool')

    plt.figure(figsize=(17, 15))
    plt.scatter(x_proj[mask_isweekday, 0], x_proj[mask_isweekday, 1], marker='.', lw=2,
                c=error[mask_isweekday], cmap=plt.cm.get_cmap(cmap_color), label='Week days')
    plt.scatter(x_proj[np.invert(mask_isweekday), 0], x_proj[np.invert(mask_isweekday), 1], marker='+', lw=2,
                c=error[np.invert(mask_isweekday)], cmap=plt.cm.get_cmap(cmap_color), label='Weekend')

    plt.colorbar(label='Error')
    #plt.clim(-0.5, 11.5)
    plt.scatter(x_proj[mask_ishd, 0], x_proj[mask_ishd, 1], marker='x', lw=3,
                c=error[mask_ishd], cmap=plt.cm.get_cmap(cmap_color), label='Holiday Days')

    plt.legend()
    plt.title('Projection on the latent space')

    if name is None:
        name = 'latent_space_proj'

    if path_folder_out is not None:
        plt.savefig(os.path.join(path_folder_out, name + '.png'))

    plt.show()


def plotly_latent_space_projection(x_proj, calendar_info, path_folder_out, name=None):
    """

    :param x_proj:
    :param calendar_info:
    :param path_folder_out:
    :param name:
    :return:
    """

    month = np.array(calendar_info.month)

    mask_isweekday = calendar_info.is_weekday.astype('bool')
    mask_ishd = calendar_info.is_hd.astype('bool')

    mask = mask_isweekday & np.invert(mask_ishd)
    trace_weekday = Scatter(x=x_proj[mask, 0],
                            y=x_proj[mask, 1],
                            name='Week Days',
                            mode='markers',
                            text=calendar_info.ds[mask],
                            hoverinfo='text',
                            marker=Marker(size=5,
                                          symbol=27,
                                          color=month[mask],
                                          colorbar=ColorBar(title='Colorbar'),
                                          colorscale='Viridis'
                                          )
                            )

    mask = np.invert(mask_isweekday) & np.invert(mask_ishd)
    trace_weekend = Scatter(x=x_proj[mask, 0],
                            y=x_proj[mask, 1],
                            name='Week Ends',
                            mode='markers',
                            text=calendar_info.ds[mask],
                            hoverinfo='text',
                            marker=Marker(size=5,
                                          symbol=4,
                                          color=month[mask],
                                          colorbar=ColorBar(title='Colorbar'),
                                          colorscale='Viridis'
                                          )
                            )

    trace_hd = Scatter(x=x_proj[mask_ishd, 0],
                       y=x_proj[mask_ishd, 1],
                       name='Holiday Days',
                       mode='markers',
                       text=calendar_info.ds[mask_ishd],
                       hoverinfo='text',
                       marker=Marker(size=7,
                                     symbol=3,
                                     color='black'
                                     )
                       )

    data = [trace_weekday, trace_weekend, trace_hd]

    layout = dict(title='Projection on the latent space',
                  yaxis=dict(zeroline=False),
                  xaxis=dict(zeroline=False),
                  legend=dict(x=-.1, y=1.2)
                  )

    fig = dict(data=data, layout=layout)

    if name is None:
        name = 'latent_space_proj'

    plot(fig, filename=os.path.join(path_folder_out, name + '.html'))


def plot_latent_space_projection(x_proj, calendar_info, path_folder_out, name=None, pyplot=True, plotly=False):
    """

    :param x_proj:
    :param calendar_info:
    :param path_folder_out:
    :param name:
    :param pyplot:
    :param plotly:
    :return:
    """
    if pyplot:
        pyplot_latent_space_projection(x_proj=x_proj, calendar_info=calendar_info,
                                       path_folder_out=path_folder_out, name=name)

    if plotly:
        plotly_latent_space_projection(x_proj=x_proj, calendar_info=calendar_info,
                                       path_folder_out=path_folder_out, name=name)