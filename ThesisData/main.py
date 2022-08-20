import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import plotly
import os
from scipy import stats

pd.set_option('display.max_columns', None)


def import_war():
    df_war = pd.read_excel(os.path.join('input', 'War.xlsx'))
    df_war['Date Time'] = pd.to_datetime(df_war['Date Time'], format='%m/%d/%Y %H:%M')
    df_war = df_war.sort_values(by='Date Time', ignore_index=True)
    print('Imported war calendar')
    return df_war


def import_min_data(instrument):
    df = pd.DataFrame()
    for filename in os.listdir('input/'):
        if filename.startswith(instrument + '_min'):
            df_file = pd.read_excel(os.path.join('input', filename), skiprows=16)
            df_file = df_file.dropna(axis=1, how='all')
            df_file = df_file.dropna(subset=['Close', 'Volume'])
            df_file = df_file.drop(columns=['Local Date'])
            df = pd.concat([df, df_file], ignore_index=True)
    df = df[df['Volume'] >= 0]
    df = df.sort_values('Local Time', ignore_index=True)
    df.to_excel(os.path.join('output', instrument + '_min_combined.xlsx'))
    # raise Exception
    print(f'Imported minute data of {instrument}')
    return df


def import_day_data(instrument):
    df = pd.read_excel(os.path.join('input', instrument + '_day.xlsx'), skiprows=16)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=['Close', 'Volume'])
    df = df.sort_values('Exchange Date', ignore_index=True)
    df.to_excel(os.path.join('output', instrument + '_day_processed.xlsx'))
    print(f'Imported daily data of {instrument}')
    return df


def generate_stat(df: pd.DataFrame, instrument):
    df_stat = df[['%Chg', 'Volume']].quantile([.01, .25, .50, .75, .99])
    df1 = pd.DataFrame([[df['%Chg'].mean(), df['Volume'].mean()], [df['%Chg'].min(), df['Volume'].min()]],
                       columns=['%Chg', 'Volume'],
                       index=['Mean', 'Min'],
                       )
    df_stat = pd.concat([df1, df_stat])
    df_2 = pd.DataFrame([[df['%Chg'].max(), df['Volume'].max()]], columns=['%Chg', 'Volume'], index=['Max'])
    df_stat = pd.concat([df_stat, df_2])
    df_stat['%Chg'] = df_stat['%Chg'] * 100
    df_stat['Volume'] = df_stat['Volume'] / 1000
    df_stat = df_stat.transpose()
    df_stat.columns = ['Mean', 'Min', '1%', '25%', '50%', '75%', '99%', 'Max']
    df_stat.index = ['Return(percent)', 'Volume(1,000 shares)']

    df_stat.to_excel(os.path.join('output', instrument + '_stat.xlsx'))
    # raise Exception
    print(f'Finished generating volatility and volume statistics')
    return


def generate_distribution(df: pd.DataFrame, instrument):
    # fig1 = ff.create_distplot([df['%Chg'].dropna() * 100], ['Return'], bin_size=.05, histnorm='probability')
    # fig2 = ff.create_distplot([df['Volume'].dropna() / 1000], ['Volume'], bin_size=5000, histnorm='probability')
    df['% Return'] = df['%Chg'] * 100
    df['Volume (1000 Shares)'] = df['Volume'] / 1000
    fig1 = px.histogram(df, x='% Return')
    fig2 = px.histogram(df, x='Volume (1000 Shares)')
    plotly.offline.plot(fig1, filename=os.path.join('output', instrument + '_return_dist.html'))
    plotly.offline.plot(fig2, filename=os.path.join('output', instrument + '_volume_dist.html'))
    print(f'Generated distribution of minute volume and volatility of {instrument}')
    return


def generate_plotting(df: pd.DataFrame, instrument):
    # daily log realized vola (sum of squared one min returns)
    df['Log Realized Vola'] = df['%Chg'] ** 2
    df['Date String'] = df['Local Time'].dt.strftime('%Y-%m-%d')
    df_day = df.groupby(['Date String'])[['Log Realized Vola', 'Volume']].sum()
    df_day['Log Realized Vola'] = np.log10(((df_day['Log Realized Vola'] + 1) ** 365 - 1) * 100)
    df_day['Volume'] = np.log10(df_day['Volume'] / 1000000)
    df_day = df_day.reset_index()

    # create breaks to skip time gap on graphs
    dt_all = pd.date_range(start=datetime.strptime(df_day.loc[0, 'Date String'], '%Y-%m-%d'),
                           end=datetime.strptime(df_day.loc[len(df_day) - 1, 'Date String'], '%Y-%m-%d'))
    dt_obs = df['Date String'].tolist()
    dt_breaks = [d for d in dt_all.strftime('%Y-%m-%d').tolist() if d not in dt_obs]
    dt_breaks = pd.to_datetime(dt_breaks)

    # plotting
    fig1 = go.Figure(data=[go.Scatter(
        x=df_day['Date String'], y=df_day['Log Realized Vola'], mode='lines'
    )])
    fig1.update_xaxes(rangebreaks=[dict(dvalue=24 * 60 * 60 * 1000, values=dt_breaks)])
    fig2 = go.Figure(data=[go.Scatter(
        x=df_day['Date String'], y=df_day['Volume'], mode='lines'
    )])
    fig2.update_xaxes(rangebreaks=[dict(dvalue=24 * 60 * 60 * 1000, values=dt_breaks)])

    plotly.offline.plot(fig1, filename=os.path.join('output', instrument + '_vola_plot.html'))
    plotly.offline.plot(fig2, filename=os.path.join('output', instrument + '_volume_plot.html'))

    print(f'Finished plotting volume and volatility for {instrument}')
    return


def import_news(instrument):
    df_ec = pd.read_excel(os.path.join('input', 'ECO21-22-Eurozone.xlsx'))
    df_us = pd.read_excel(os.path.join('input', 'ECO21-22-US.xlsx'))

    ec_list = ['ECB Main Refinancing Rate', 'CPI YoY', 'GDP SA QoQ', 'S&P Global Eurozone Manufacturing PMI']
    us_list = ['Initial Jobless Claims', 'ISM Manufacturing', 'Change in Nonfarm Payrolls',
               'CPI MoM', 'FOMC Rate Decision (Upper Bound)', 'GDP Annualized QoQ']
    fgbl_list = ['S&P Global/BME Germany Manufacturing PMI', 'CPI YoY', 'ZEW Survey Expectations',
                 'IFO Business Climate']

    df_filtered_ec = df_ec[(df_ec['Region'] == 'EC') & (df_ec['Relevance'] >= 90)]
    df_filtered_ec = df_filtered_ec[df_filtered_ec['Event'].isin(ec_list)]
    df_filtered_us = df_us[(df_us['Region'] == 'US') & (df_us['Relevance'] >= 90)]
    df_filtered_us = df_filtered_us[df_filtered_us['Event'].isin(us_list)]
    df_fgbl = df_ec[(df_ec['Region'] == 'GE') & (df_ec['Relevance'] >= 90)]
    df_fgbl = df_fgbl[df_fgbl['Event'].isin(fgbl_list)]

    if instrument == 'fgbl':
        df_ann = pd.concat([df_filtered_ec, df_filtered_us, df_fgbl])
    else:
        df_ann = pd.concat([df_filtered_ec, df_filtered_us])

    df_ann['Date Time'] = pd.to_datetime(df_ann['Date Time'], format='%m/%d/%Y %H:%M')
    df_ann = df_ann.sort_values(by='Date Time', ignore_index=True)
    df_ann['Date Time'] = df_ann['Date Time'].dt.tz_localize('US/Eastern')
    df_ann['Date Time'] = df_ann['Date Time'].dt.tz_convert('Europe/Berlin')
    df_ann['Date Time'] = df_ann['Date Time'].dt.tz_localize(None)

    df_ann.to_excel(os.path.join('output', instrument + '_news_announcements.xlsx'), index=False)
    print(f'Generated and processed news announcement data for {instrument}')
    return df_ann


def generate_intraday_pattern(df, df_ann, instrument):
    df['Date String'] = df['Local Time'].dt.strftime('%Y-%m-%d')
    df_ann['Date String'] = df_ann['Date Time'].dt.strftime('%Y-%m-%d')
    df_intraday = df[~df['Date String'].isin(df_ann['Date String'])].copy()
    df_intraday['Minute'] = df['Local Time'].dt.strftime('%H:%M')
    df_intraday['Intra Vola'] = np.sqrt(df_intraday['%Chg'].pow(2))
    df_intraday = df_intraday.groupby('Minute')[['Intra Vola', 'Volume']].mean()
    df_intraday = df_intraday.dropna()
    df_intraday['Intra Vola'] = ((df_intraday['Intra Vola'] + 1).pow(365) - 1) * 100
    df_intraday = df_intraday.reset_index()

    df_intraday_vola = df_intraday[['Minute', 'Intra Vola']]
    df_intraday_vola = df_intraday_vola[(np.abs(stats.zscore(df_intraday_vola['Intra Vola'])) < 1.2)]
    df_intraday_vola['Smooth'] = np.polyval(np.polyfit(df_intraday_vola.index, df_intraday_vola['Intra Vola'], 10),
                                            df_intraday_vola.index)

    df_intraday_volume = df_intraday[['Minute', 'Volume']]
    df_intraday_volume = df_intraday_volume[(np.abs(stats.zscore(df_intraday_volume['Volume'])) < 1.2)]
    df_intraday_volume['Smooth'] = np.polyval(np.polyfit(df_intraday_volume.index, df_intraday_volume['Volume'], 10),
                                              df_intraday_volume.index)

    fig1 = px.scatter(df_intraday_vola, x='Minute', y='Intra Vola')
    fig1.add_trace(go.Scatter(x=df_intraday_vola['Minute'], y=df_intraday_vola['Smooth'], name=None), secondary_y=False)
    fig2 = px.scatter(df_intraday_volume, x='Minute', y='Volume')
    fig2.add_trace(go.Scatter(x=df_intraday_volume['Minute'], y=df_intraday_volume['Smooth'], name=None),
                   secondary_y=False)

    plotly.offline.plot(fig1, filename=os.path.join('output', instrument + '_intraday_vola.html'))
    plotly.offline.plot(fig2, filename=os.path.join('output', instrument + '_intraday_volume.html'))

    print(f'Finished generating intraday pattern on non-announcement days for {instrument}')
    return


def generate_ann_pattern(df, df_news, instrument, war_flag=False):
    if war_flag:
        region_list = [['EC']]
    elif instrument == 'stoxx':
        region_list = [['EC'], ['US'], ['EC', 'US']]
    else:
        region_list = [['GE'], ['EC', 'US', 'GE']]

    for region in region_list:
        region_str = '_'.join(region)
        df_news_region = df_news[df_news['Region'].isin(region)]
        pattern_vola_data = []
        pattern_volume_data = []
        for i, row in df_news_region.iterrows():
            min_range = list(range(-15, 16))
            news_vola_line = []
            news_volume_line = []
            for minute in min_range:
                dt = row['Date Time'] + timedelta(minutes=minute)
                try:
                    news_vola_line.append(df.loc[df['Local Time'] == dt, '%Chg'].iloc[0])
                    news_volume_line.append(df.loc[df['Local Time'] == dt, 'Volume'].iloc[0])
                except IndexError:
                    news_vola_line.append(np.nan)
                    news_volume_line.append(np.nan)

            pattern_vola_data.append(news_vola_line)
            pattern_volume_data.append(news_volume_line)
        columns = list(range(-15, 16))

        df_pattern_vola = pd.DataFrame(pattern_vola_data, columns=columns)
        df_pattern_vola = df_pattern_vola.dropna(how='all').pow(2)
        df_pattern_volume = pd.DataFrame(pattern_volume_data, columns=columns)
        df_pattern_volume = df_pattern_volume.dropna(how='all')

        pattern_vola = df_pattern_vola.mean()
        pattern_vola = pattern_vola.rename('Volatility')
        pattern_vola = ((np.sqrt(pattern_vola) + 1) ** 365 - 1) * 100
        pattern_volume = df_pattern_volume.mean() / 100000
        pattern_volume = pattern_volume.rename('Volume')

        fig1 = px.scatter(pattern_vola, labels=dict(index='Time', value='Spot Volatility'))
        fig1.update_traces(marker={'size': 15})
        fig1.add_vline(x=0, line_dash="dash")
        fig2 = px.scatter(pattern_volume, labels=dict(index='Time', value='Volume Intensity'))
        fig2.update_traces(marker={'size': 15})
        fig2.add_vline(x=0, line_dash="dash")

        name = instrument + '_war_' + region_str if war_flag else instrument + '_' + region_str
        plotly.offline.plot(fig1, filename=os.path.join('output', name + '_news_vola.html'))
        plotly.offline.plot(fig2, filename=os.path.join('output', name + '_news_volume.html'))

    if war_flag:
        print(f'Plotted the pattern before and after the war announcements for {instrument}')
    else:
        print(f'Plotted the pattern before and after the news announcements for {instrument}')
    return


def generate_ann_plot(df, df_news, instrument, war_flag=False):
    if war_flag:
        region = ['EC']
    elif instrument == 'stoxx':
        region = ['EC', 'US']
    else:
        region = ['EC', 'US', 'GE']
    df_news_region = df_news[df_news['Region'].isin(region)]
    plot_data = []
    for i, row in df_news_region.iterrows():
        min_range_pre = list(range(-14, 1))
        min_range_post = list(range(2, 17))
        pre_dt_list = [row['Date Time'] + timedelta(minutes=minute) for minute in min_range_pre]
        post_dt_list = [row['Date Time'] + timedelta(minutes=minute) for minute in min_range_post]
        df_pre = df[df['Local Time'].isin(pre_dt_list)].copy()
        df_post = df[df['Local Time'].isin(post_dt_list)].copy()
        df_pre['Volatility'] = df_pre['%Chg'].pow(2)
        df_post['Volatility'] = df_post['%Chg'].pow(2)

        if df_pre.empty and df_post.empty:
            continue
        else:
            line_data = [row['Date Time'], np.log10((((np.sqrt(df_pre['Volatility'].mean())) + 1) ** 365 - 1) * 100),
                         np.log10(df_pre['Volume'].mean()),
                         np.log10((((np.sqrt(df_post['Volatility'].mean())) + 1) ** 365 - 1) * 100),
                         np.log10(df_post['Volume'].mean())]
            plot_data.append(line_data)

    columns = ['Date Time', 'Pre Vola', 'Pre Volume', 'Post Vola', 'Post Volume']
    df_plot = pd.DataFrame(plot_data, columns=columns)
    fig1_trace1 = go.Scatter(x=df_plot['Date Time'], y=df_plot['Pre Vola'], mode='lines+markers', name='Pre-event')
    fig1_trace2 = go.Scatter(x=df_plot['Date Time'], y=df_plot['Post Vola'], mode='lines+markers', name='Post-event')
    fig1 = plotly.subplots.make_subplots(specs=[[{"secondary_y": False}]])
    fig1.add_trace(fig1_trace1)
    fig1.add_trace(fig1_trace2, secondary_y=False)
    fig1.update_layout({'title': go.layout.Title(text='Log Spot Volatility')})

    fig2_trace1 = go.Scatter(x=df_plot['Date Time'], y=df_plot['Pre Volume'], mode='lines+markers', name='Pre-event')
    fig2_trace2 = go.Scatter(x=df_plot['Date Time'], y=df_plot['Post Volume'], mode='lines+markers', name='Post-event')
    fig2 = plotly.subplots.make_subplots(specs=[[{"secondary_y": False}]])
    fig2.add_trace(fig2_trace1)
    fig2.add_trace(fig2_trace2, secondary_y=False)
    fig2.update_layout({'title': go.layout.Title(text='Log Volume Intensity')})

    name = instrument + '_war_' if war_flag else instrument
    plotly.offline.plot(fig1, filename=os.path.join('output', name + '_news_vola_plot.html'))
    plotly.offline.plot(fig2, filename=os.path.join('output', name + '_news_volume_plot.html'))

    if war_flag:
        print(f'Plotted the volume and vola change around the war events for {instrument}')
    else:
        print(f'Plotted the volume and vola change around the announcements for {instrument}')
    return


def generate_normalized_return(df, df_news, instrument, war_flag=False):
    if war_flag:
        region = ['EC']
    elif instrument == 'stoxx':
        region = ['EC', 'US']
    else:
        region = ['EC', 'US', 'GE']
    df_news_region = df_news[df_news['Region'].isin(region)]
    df_norm_list = []
    for i, row in df_news_region.iterrows():
        min_range_pre = list(range(-14, 1))
        min_range_post = list(range(2, 17))
        pre_dt_list = [row['Date Time'] + timedelta(minutes=minute) for minute in min_range_pre]
        post_dt_list = [row['Date Time'] + timedelta(minutes=minute) for minute in min_range_post]
        event_dt = row['Date Time'] + timedelta(minutes=1)
        df_pre = df[df['Local Time'].isin(pre_dt_list)].copy()
        df_post = df[df['Local Time'].isin(post_dt_list)].copy()
        df_pre['Volatility'] = df_pre['%Chg'].pow(2)
        df_post['Volatility'] = df_post['%Chg'].pow(2)
        series_event = df.loc[df['Local Time'] == event_dt, '%Chg'].reset_index(drop=True)

        if df_pre.empty or df_post.empty or series_event.empty:
            continue
        else:
            line_data = [event_dt, series_event[0],
                         np.sqrt(df_pre['Volatility'].mean()),
                         df_pre['Volume'].mean() / 10000,
                         np.sqrt(df_post['Volatility'].mean()),
                         df_post['Volume'].mean() / 10000]
            df_norm_list.append(line_data)

    columns = ['Date Time', 'Minute Return', 'Pre Vola', 'Pre Volume', 'Post Vola', 'Post Volume']
    df_norm = pd.DataFrame(df_norm_list, columns=columns)
    # df_norm['Pre Norm Return'] = df_norm['Minute Return'] / df_norm['Pre Vola'] - 1
    df_norm['Norm Return'] = df_norm['Minute Return'] / df_norm['Post Vola'] - 1
    df_norm = df_norm.sort_values(by='Norm Return', ignore_index=True)

    trace_pre = go.Scatter(x=df_norm['Norm Return'], y=df_norm['Pre Volume'], mode='lines+markers', name='Pre-event')
    trace_post = go.Scatter(x=df_norm['Norm Return'], y=df_norm['Post Volume'], mode='lines+markers', name='Post-event')
    fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(trace_pre)
    fig.add_trace(trace_post, secondary_y=False)
    fig.update_xaxes(title_text='Normalized Return')
    fig.update_yaxes(title_text='Average Volume     10,000 shares')
    #fig.update_xaxes(tickvals=df_norm.index, ticktext=df_norm['Norm Return'], tickmode='array')

    name = instrument + '_war_' if war_flag else instrument
    plotly.offline.plot(fig, filename=os.path.join('output', name + '_normalized_return.html'))
    if war_flag:
        print(f'Generated normalized return chart around war events for {instrument}')
    else:
        print(f'Generated normalized return chart around announcements for {instrument}')
    return


def generate_jump_regression(df, df_news, instrument, war_flag=False):
    if war_flag:
        region = ['EC']
    elif instrument == 'stoxx':
        region = ['EC', 'US']
    else:
        region = ['EC', 'US', 'GE']
    df_news_region = df_news[df_news['Region'].isin(region)]
    df_jump_list = []
    for i, row in df_news_region.iterrows():
        min_range_pre = list(range(-14, 1))
        min_range_post = list(range(2, 17))
        pre_dt_list = [row['Date Time'] + timedelta(minutes=minute) for minute in min_range_pre]
        post_dt_list = [row['Date Time'] + timedelta(minutes=minute) for minute in min_range_post]
        df_pre = df[df['Local Time'].isin(pre_dt_list)].copy()
        df_post = df[df['Local Time'].isin(post_dt_list)].copy()
        df_pre['Volatility'] = df_pre['%Chg'].pow(2)
        df_post['Volatility'] = df_post['%Chg'].pow(2)

        if df_pre.empty or df_post.empty:
            continue
        else:
            line_data = [np.log10(np.sqrt(df_post['Volatility'].mean())) - np.log10(np.sqrt(df_pre['Volatility'].mean())),
                         np.log10(df_post['Volume'].mean()) - np.log10(df_pre['Volume'].mean())]
            df_jump_list.append(line_data)
    columns = ['Log Volatility Jump', 'Log Volume Intensity Jump']
    df_jump = pd.DataFrame(df_jump_list, columns=columns)

    fig = px.scatter(df_jump, x='Log Volatility Jump', y='Log Volume Intensity Jump', trendline='ols')
    name = instrument + '_war_' if war_flag else instrument
    plotly.offline.plot(fig, filename=os.path.join('output', name + '_vola_jump.html'))
    if war_flag:
        print(f'Plotted vola jump of war events for {instrument}')
        print(f'Average log volatility jump of war events for {instrument}: {df_jump["Log Volatility Jump"].mean()}')
        print(f'Average log volume intensity jump of war events for {instrument}: {df_jump["Log Volume Intensity Jump"].mean()}')
    else:
        print(f'Plotted vola jump of announcements for {instrument}')
        print(f'Average log volatility jump of announcements for {instrument}: {df_jump["Log Volatility Jump"].mean()}')
        print(f'Average log volume intensity jump of announcements for {instrument}: {df_jump["Log Volume Intensity Jump"].mean()}')
    results = px.get_trendline_results(fig)
    print(results.px_fit_results.iloc[0].summary())

    return


def generate_ann_stat(df, df_news, instrument):
    if instrument == 'stoxx':
        region = ['EC', 'US']
    else:
        region = ['EC', 'US', 'GE']
    df_news_region = df_news[df_news['Region'].isin(region)]
    event_list = df_news_region['Event'].unique()
    event_list = np.insert(event_list, 0, 'All')
    stat_data = []
    for event in event_list:
        if event == 'All':
            df_event = df_news
        else:
            df_event = df_news[df_news['Event'] == event].reset_index(drop=True)
        df_jump_list = []
        for i, row in df_event.iterrows():
            min_range_pre = list(range(-14, 1))
            min_range_post = list(range(2, 17))
            pre_dt_list = [row['Date Time'] + timedelta(minutes=minute) for minute in min_range_pre]
            post_dt_list = [row['Date Time'] + timedelta(minutes=minute) for minute in min_range_post]
            df_pre = df[df['Local Time'].isin(pre_dt_list)].copy()
            df_post = df[df['Local Time'].isin(post_dt_list)].copy()
            df_pre['Volatility'] = df_pre['%Chg'].pow(2)
            df_post['Volatility'] = df_post['%Chg'].pow(2)

            if df_pre.empty or df_post.empty:
                continue
            else:
                line_data = [
                    np.log10(np.sqrt(df_post['Volatility'].mean())) - np.log10(np.sqrt(df_pre['Volatility'].mean())),
                    np.log10(df_post['Volume'].mean()) - np.log10(df_pre['Volume'].mean())]
                df_jump_list.append(line_data)
        columns = ['Log Volatility Jump', 'Log Volume Intensity Jump']
        df_jump = pd.DataFrame(df_jump_list, columns=columns)

        region = df_event.loc[0, 'Region']
        num_obs = len(df_event)
        avg_vola = df_jump['Log Volatility Jump'].mean()
        avg_volume = df_jump['Log Volume Intensity Jump'].mean()
        coef = df_jump['Log Volume Intensity Jump'].corr(df_jump['Log Volatility Jump'])
        stat_data.append([event, region, num_obs, avg_vola, avg_volume, coef])

    columns = ['Event', 'Region', 'No. of Obs.', 'Log Volatility', 'Log Volume', 'Coefficient']
    df_stat = pd.DataFrame(stat_data, columns=columns).sort_values(by='Region')
    df_stat.to_excel(os.path.join('output', instrument + '_announcements_stat.xlsx'), index=False)
    print(f'Output announcement statistics spread sheet of {instrument}')
    return


def main():
    start_time = time.time()
    df_war = import_war()
    for instrument in ['stoxx', 'fgbl']:
        df_min = import_min_data(instrument)
        # df_day = import_day_data(instrument)
        print(f'{len(df_min)} data points imported for {instrument}')

        '''
        1. Summary statistics of return (use % change column ) and volume can be seen in table 1,
        and also need a distribution graph.
        '''
        # Summary statistics table
        generate_stat(df_min, instrument)
        generate_distribution(df_min, instrument)

        '''
        2. Both daily logarithmic volume (daily volume) and volatilities (sum of the squared one
        minute returns of each day) can be seen in figure 2.
        '''
        generate_plotting(df_min, instrument)

        '''
        3. New announcements table, before and after filter comparisons
        4. News filter: after filtering the below news announces, we see them as my “announcements days”
        '''
        # ALWAYS SET ACTIVE
        df_ann = import_news(instrument)

        '''
        5. By observing the intraday pattern, I plot the average volatility over each minute of the day. 
            First, only select non announcement days. Then, (on each day, on this minute, the squartroot
            of the average of squared returns) (and the average trading volume across each minute) 
            to do: adjust the degree of polynomial fitting & x-axis scale
        '''
        generate_intraday_pattern(df_min, df_ann, instrument)

        '''
        6. Plot the average volume intensity and volatilities changes before and after the news announcements.
            stoxx: EU; US; EU + US
            FGBL: GE; EU + US + GE
        '''
        generate_ann_pattern(df_min, df_ann, instrument)
        generate_ann_pattern(df_min, df_war, instrument, True)

        ''' 
        7. Plot a figure depicts each events volume and vola change across our data set. 
        '''
        generate_ann_plot(df_min, df_ann, instrument)
        generate_ann_plot(df_min, df_war, instrument, True)

        '''
        8. Normalize the one-minute return and plot the pre and post volume.
        '''
        generate_normalized_return(df_min, df_ann, instrument)
        generate_normalized_return(df_min, df_war, instrument, True)

        '''
        9.	Plot the vola and volume elasticity (use log)
            Vola :log(στ )≡log(στ )−log(στ−) and
            Volume:log(mτ )≡log(mτ )−log(mτ−)
        '''
        generate_jump_regression(df_min, df_ann, instrument)
        generate_jump_regression(df_min, df_war, instrument, True)

        '''
        10. Create a spread sheet grouping the data by event type, including the following data:
            number of observations; log volatility; lgo volume; coefficient
        '''
        generate_ann_stat(df_min, df_ann, instrument)

    print(f'Program finished in {time.time() - start_time}s')
    exit(0)


if __name__ == '__main__':
    main()
