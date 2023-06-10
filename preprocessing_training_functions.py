import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable, plasma_r
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import statistics
from scipy.signal import savgol_filter

def scrape_bios(download_file=False, end_year=2023):
    start_year = 2007

    running_df = None
    for year in range(start_year, end_year):
        url = f'https://www.naturalstattrick.com/playerteams.php?fromseason={year}{year+1}&thruseason={year}{year+1}&stype=2&sit=all&score=all&stdoi=bio&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL'
        
        response = ''
        while response == '':
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')

                table = soup.find('table')

                rows = table.findAll('tr')
                headers = [th.text.strip() for th in rows[0].findAll('th')]
                data = []
                for row in rows[1:]:
                    data.append([td.text.strip() for td in row.findAll('td')])

                temp_df = pd.DataFrame(data, columns=headers)
                temp_df = temp_df.drop([''], axis=1)

                running_df = pd.concat([temp_df,running_df]).drop_duplicates(['Player']).reset_index(drop=True)

                if download_file == True:
                    filename = f'player_bios'
                    if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
                        os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
                    running_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
                    print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

                break
            except requests.exceptions.ConnectionError:
                print('Connection failed. Periodic request quota exceeded. Trying again in 5 seconds.')
                time.sleep(5)

    return running_df

def prune_bios(player_bio_df):
    stat_df = player_bio_df.drop(['Team', 'Birth City', 'Birth State/Province', 'Birth Country', 'Nationality', 'Draft Team', 'Draft Round', 'Round Pick'], axis=1)
    return stat_df

def scrape_statistics(stat_df, situation='ev', stat_type='std', download_file=False):
    # situation = (ev, 5v5, pp, pk ...)
    # stat_type = (std, oi)
    start_year = 2007
    end_year = 2023
    try:
        stat_df = stat_df.set_index('Player')
    except KeyError:
        pass
    situation_reassignment = {'ev': 'EV', '5v5': '5v5', 'pp': 'PP', 'pk': 'PK'}
    name_reassignment = {'Jani Hakanpää': 'Jani Hakanpaa', 'Tommy Novak': 'Thomas Novak'}
    stat_type_reassignment = {'std': 'Standard', 'oi': 'On-Ice'}

    print(f'\nNow Scraping: {situation_reassignment[situation]} {stat_type_reassignment[stat_type].lower()} statistics from {start_year}-{end_year}')

    for year in range(start_year, end_year):
        url = f'https://www.naturalstattrick.com/playerteams.php?fromseason={year}{year+1}&thruseason={year}{year+1}&stype=2&sit={situation}&score=all&stdoi={stat_type}&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL'

        response = ''
        while response == '':
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')

                table = soup.find('table')

                rows = table.findAll('tr')
                headers = [th.text.strip() for th in rows[0].findAll('th')]
                data = []
                for row in rows[1:]:
                    data.append([td.text.strip() for td in row.findAll('td')])

                temp_df = pd.DataFrame(data, columns=headers)

                for index, row in temp_df.iterrows():

                    # Change players whose names are different in the bios and statistics
                    if row['Player'] in name_reassignment.keys():
                        player_name = name_reassignment[row['Player']]
                    else:
                        player_name = row['Player']

                    if stat_type == 'std':
                        stat_df.loc[player_name, f'{year+1} GP'] = row['GP']
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} ATOI'] = round(float(row['TOI'])/int(row['GP']),4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} G/60'] = round(float(row['Goals'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} A1/60'] = round(float(row['First Assists'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} A2/60'] = round(float(row['Second Assists'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} ixG/60'] = round(float(row['ixG'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} Shots/60'] = round(float(row['Shots'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} iCF/60'] = round(float(row['iCF'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} Rush Attempts/60'] = round(float(row['Rush Attempts'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} Rebounds Created/60'] = round(float(row['Rebounds Created'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} PIM/60'] = round(float(row['PIM'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} HIT/60'] = round(float(row['Hits'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} BLK/60'] = round(float(row['Shots Blocked'])/float(row['TOI'])*60,4)

                    elif stat_type == 'oi':
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} oiCF/60'] = round(float(row['CF'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} oiSF/60'] = round(float(row['SF'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} oiGF/60'] = round(float(row['GF'])/float(row['TOI'])*60,4)
                        stat_df.loc[player_name, f'{year+1} {situation_reassignment[situation]} oixGF/60'] = round(float(row['xGF'])/float(row['TOI'])*60,4)

                    stat_df = stat_df.copy() # De-fragment the dataframe to improve performance

                print(f'{year}-{year+1}: Scraped. Dimensions = {stat_df.shape[0]}x{stat_df.shape[1]}')
            except requests.exceptions.ConnectionError:
                print('Connection failed. Periodic request quota exceeded. Trying again in 5 seconds.')
                time.sleep(5)

    if download_file == True:
        filename = f'historical_player_statistics'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stat_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return stat_df

def scrape_player_statistics(existing_csv=False):
    if existing_csv == True:
        stat_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/historical_player_statistics.csv")
    elif existing_csv == False:
        player_bio_df = scrape_bios(False, 2023)
        player_bio_df = player_bio_df.drop(player_bio_df.columns[0], axis=1)

        stat_df = prune_bios(player_bio_df)
        stat_df = scrape_statistics(stat_df, 'ev', 'std', True)
        stat_df = scrape_statistics(stat_df, 'pp', 'std', True)
        stat_df = scrape_statistics(stat_df, 'pk', 'std', True)
        stat_df = scrape_statistics(stat_df, 'ev', 'oi', True)
        stat_df = scrape_statistics(stat_df, 'pp', 'oi', True)
        stat_df = scrape_statistics(stat_df, 'pk', 'oi', True)

        shooting_talent_df = calc_shooting_talent(stat_df, True)
        stat_df = pd.merge(stat_df, shooting_talent_df[['Player', 'Shooting Talent']], on='Player', how='left')

        ixg_columns = [col for col in stat_df.columns if 'ixG' in col and 'oi' not in col]
        stat_df[ixg_columns] = round(stat_df[ixg_columns].mul(stat_df['Shooting Talent'] + 1, axis=0),4)

        stat_df.set_index('Player', inplace=True)

        filename = f'historical_player_statistics'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stat_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return stat_df

def aggregate_stats_all_situations(evrate, evatoi, pprate, ppatoi, pkrate, pkatoi, gp, float=False):
    if float == True:
        return round((evrate/60 * evatoi + pprate/60 * ppatoi + pkrate/60 * pkatoi) * gp,2)
    else:
        return int(round((evrate/60 * evatoi + pprate/60 * ppatoi + pkrate/60 * pkatoi) * gp))
    
def calc_shooting_talent(stat_df, download_file=False):
    stat_df = stat_df.fillna(0)
    shooting_talent_cols = ['Player', 'Position', 'Age']

    start_year = 2007
    end_year = 2023

    for year in range(start_year, end_year):
        shooting_talent_cols.append(f'{year+1} Goals')
        shooting_talent_cols.append(f'{year+1} xGoals')
        shooting_talent_cols.append(f'{year+1} Shots')
        stat_df[f'{year+1} Goals'] = stat_df.apply(lambda row: aggregate_stats_all_situations(row[f'{year+1} EV G/60'], row[f'{year+1} EV ATOI'], row[f'{year+1} PP G/60'], row[f'{year+1} PP ATOI'], row[f'{year+1} PK G/60'], row[f'{year+1} PK ATOI'], row[f'{year+1} GP']), axis=1)
        stat_df[f'{year+1} xGoals'] = stat_df.apply(lambda row: aggregate_stats_all_situations(row[f'{year+1} EV ixG/60'], row[f'{year+1} EV ATOI'], row[f'{year+1} PP ixG/60'], row[f'{year+1} PP ATOI'], row[f'{year+1} PK ixG/60'], row[f'{year+1} PK ATOI'], row[f'{year+1} GP'], True), axis=1)
        stat_df[f'{year+1} Shots'] = stat_df.apply(lambda row: aggregate_stats_all_situations(row[f'{year+1} EV Shots/60'], row[f'{year+1} EV ATOI'], row[f'{year+1} PP Shots/60'], row[f'{year+1} PP ATOI'], row[f'{year+1} PK Shots/60'], row[f'{year+1} PK ATOI'], row[f'{year+1} GP']), axis=1)

    shooting_talent_df = stat_df[shooting_talent_cols]
    # shooting_talent_df['Goals'] = shooting_talent_df.filter(like=' Goals').sum(axis=1)
    # shooting_talent_df['xGoals'] = shooting_talent_df.filter(like='xGoals').sum(axis=1)
    # shooting_talent_df['Shots'] = shooting_talent_df.filter(like='Shots').sum(axis=1)

    for index, row in shooting_talent_df.iterrows():
        relevant_shots, relevant_xgoals, relevant_goals = 0, 0, 0
        for year in range(end_year, start_year, -1):
            if relevant_shots < 1000:
                relevant_shots += shooting_talent_df.at[index, f'{year} Shots']
                relevant_xgoals += shooting_talent_df.at[index, f'{year} xGoals']
                relevant_goals += shooting_talent_df.at[index, f'{year} Goals']
        
        shooting_talent_df.at[index, 'Relevant Shots'] = int(relevant_shots)
        shooting_talent_df.at[index, 'Relevant xGoals'] = relevant_xgoals
        shooting_talent_df.at[index, 'Relevant Goals'] = int(relevant_goals)

    avg_xg_per_shot = shooting_talent_df['Relevant xGoals'].sum() / shooting_talent_df['Relevant Shots'].sum()
    avg_g_per_shot = shooting_talent_df['Relevant Goals'].sum() / shooting_talent_df['Relevant Shots'].sum()

    for index, row in shooting_talent_df.iterrows():
        if shooting_talent_df.loc[index, f'Relevant Shots'] < 1000:
            pseudoshots = 1000 - shooting_talent_df.loc[index, f'Relevant Shots']
            shooting_talent_df.at[index, 'Relevant Shots'] = 1000
            shooting_talent_df.at[index, 'Relevant xGoals'] = round(avg_xg_per_shot * pseudoshots, 2)
            shooting_talent_df.at[index, 'Relevant Goals'] = round(avg_g_per_shot * pseudoshots, 2)

    shooting_talent_df['xG/Shot'] = shooting_talent_df['Relevant xGoals'] / shooting_talent_df['Relevant Shots']
    shooting_talent_df['Gax'] = shooting_talent_df['Relevant Goals'] - shooting_talent_df['Relevant xGoals']
    shooting_talent_df['Gax/Shot'] = shooting_talent_df['Gax'] / shooting_talent_df['Relevant Shots']
    shooting_talent_df['Shooting Talent'] = (shooting_talent_df['Relevant Goals'] - shooting_talent_df['Relevant xGoals']) / shooting_talent_df['Relevant xGoals']

    shooting_talent_df = shooting_talent_df.sort_values(by='Shooting Talent', ascending=False)
    shooting_talent_df = shooting_talent_df.reset_index(drop=True)
    
    if download_file == True:
        filename = f'shooting_talent'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        shooting_talent_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return shooting_talent_df

def create_instance_df(dependent_variable, columns, stat_df, download_file=False):
    # str, list, df, bool

    start_year = 2007
    end_year = 2023

    instance_df = pd.DataFrame(columns=columns)

    for index, row in stat_df.iterrows():
        for year in range(start_year+1, end_year):
            if dependent_variable == 'forward_GP':
                # filter out:
                    # defence
                    # players with < 50 GP in Y5
                    # players with no GP in Y4

                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 50:
                    pass
                else:
                    prev_gps = [fetch_data(row, year, 1, None, 'GP'), fetch_data(row, year, 2, None, 'GP'), fetch_data(row, year, 3, None, 'GP'), fetch_data(row, year, 4, None, 'GP')]
                    prev_avg = statistics.mean([x for x in prev_gps if not np.isnan(x)])
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP') - prev_avg
                    ]
            elif dependent_variable == 'defence_GP':
                # filter out:
                    # forwards
                    # players with < 50 GP in Y5
                    # players with no GP in Y4

                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 50:
                    pass
                else:
                    prev_gps = [fetch_data(row, year, 1, None, 'GP'), fetch_data(row, year, 2, None, 'GP'), fetch_data(row, year, 3, None, 'GP'), fetch_data(row, year, 4, None, 'GP')]
                    prev_avg = statistics.mean([x for x in prev_gps if not np.isnan(x)])
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP') - prev_avg
                    ]
            elif dependent_variable == 'forward_EV_ATOI':
                # filter out:
                    # defence
                    # players with < 40 GP in Y5
                    # players with < 20 GP in Y4

                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 40 or fetch_data(row, year, 4, None, 'GP') < 20:
                    pass
                else:
                    prev_ev_atoi = [fetch_data(row, year, 1, 'ev', 'ATOI'), fetch_data(row, year, 2, 'ev', 'ATOI'), fetch_data(row, year, 3, 'ev', 'ATOI'), fetch_data(row, year, 4, 'ev', 'ATOI')]
                    prev_avg = statistics.mean([x for x in prev_ev_atoi if not np.isnan(x)])
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'ev', 'ATOI'),
                        fetch_data(row, year, 2, 'ev', 'ATOI'),
                        fetch_data(row, year, 3, 'ev', 'ATOI'),
                        fetch_data(row, year, 4, 'ev', 'ATOI'),
                        fetch_data(row, year, 5, 'ev', 'ATOI'),
                        fetch_data(row, year, 5, 'ev', 'ATOI') - prev_avg,
                    ]
            elif dependent_variable == 'defence_EV_ATOI':
                # filter out:
                    # forwards
                    # players with < 40 GP in Y5
                    # players with < 20 GP in Y4

                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 40 or fetch_data(row, year, 4, None, 'GP') < 20:
                    pass
                else:
                    prev_ev_atoi = [fetch_data(row, year, 1, 'ev', 'ATOI'), fetch_data(row, year, 2, 'ev', 'ATOI'), fetch_data(row, year, 3, 'ev', 'ATOI'), fetch_data(row, year, 4, 'ev', 'ATOI')]
                    prev_avg = statistics.mean([x for x in prev_ev_atoi if not np.isnan(x)])
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'ev', 'ATOI'),
                        fetch_data(row, year, 2, 'ev', 'ATOI'),
                        fetch_data(row, year, 3, 'ev', 'ATOI'),
                        fetch_data(row, year, 4, 'ev', 'ATOI'),
                        fetch_data(row, year, 5, 'ev', 'ATOI'),
                        fetch_data(row, year, 5, 'ev', 'ATOI') - prev_avg,
                    ]
            elif dependent_variable == 'forward_PP_ATOI':
                # filter out:
                    # defence
                    # players with < 40 GP in Y5
                    # players with < 20 GP in Y4

                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 40 or fetch_data(row, year, 4, None, 'GP') < 20:
                    pass
                else:
                    prev_pp_atoi = [fetch_data(row, year, 1, 'pp', 'ATOI'), fetch_data(row, year, 2, 'pp', 'ATOI'), fetch_data(row, year, 3, 'pp', 'ATOI'), fetch_data(row, year, 4, 'pp', 'ATOI')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_pp_atoi if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0

                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'pp', 'ATOI'),
                        fetch_data(row, year, 2, 'pp', 'ATOI'),
                        fetch_data(row, year, 3, 'pp', 'ATOI'),
                        fetch_data(row, year, 4, 'pp', 'ATOI'),
                        fetch_data(row, year, 5, 'pp', 'ATOI'),
                        fetch_data(row, year, 5, 'pp', 'ATOI') - prev_avg,
                    ]
            elif dependent_variable == 'defence_PP_ATOI':
                # filter out:
                    # forwards
                    # players with < 40 GP in Y5
                    # players with < 20 GP in Y4

                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 40 or fetch_data(row, year, 4, None, 'GP') < 20:
                    pass
                else:
                    prev_pp_atoi = [fetch_data(row, year, 1, 'pp', 'ATOI'), fetch_data(row, year, 2, 'pp', 'ATOI'), fetch_data(row, year, 3, 'pp', 'ATOI'), fetch_data(row, year, 4, 'pp', 'ATOI')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_pp_atoi if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'pp', 'ATOI'),
                        fetch_data(row, year, 2, 'pp', 'ATOI'),
                        fetch_data(row, year, 3, 'pp', 'ATOI'),
                        fetch_data(row, year, 4, 'pp', 'ATOI'),
                        fetch_data(row, year, 5, 'pp', 'ATOI'),
                        fetch_data(row, year, 5, 'pp', 'ATOI') - prev_avg,
                    ]
            elif dependent_variable == 'forward_PK_ATOI':
                # filter out:
                    # defence
                    # players with < 40 GP in Y5
                    # players with < 20 GP in Y4

                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 40 or fetch_data(row, year, 4, None, 'GP') < 20:
                    pass
                else:
                    prev_pp_atoi = [fetch_data(row, year, 1, 'pk', 'ATOI'), fetch_data(row, year, 2, 'pk', 'ATOI'), fetch_data(row, year, 3, 'pk', 'ATOI'), fetch_data(row, year, 4, 'pk', 'ATOI')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_pp_atoi if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'pk', 'ATOI'),
                        fetch_data(row, year, 2, 'pk', 'ATOI'),
                        fetch_data(row, year, 3, 'pk', 'ATOI'),
                        fetch_data(row, year, 4, 'pk', 'ATOI'),
                        fetch_data(row, year, 5, 'pk', 'ATOI'),
                        fetch_data(row, year, 5, 'pk', 'ATOI') - prev_avg,
                    ]
            elif dependent_variable == 'defence_PK_ATOI':
                # filter out:
                    # forwards
                    # players with < 40 GP in Y5
                    # players with < 20 GP in Y4

                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 40 or fetch_data(row, year, 4, None, 'GP') < 20:
                    pass
                else:
                    prev_pp_atoi = [fetch_data(row, year, 1, 'pk', 'ATOI'), fetch_data(row, year, 2, 'pk', 'ATOI'), fetch_data(row, year, 3, 'pk', 'ATOI'), fetch_data(row, year, 4, 'pk', 'ATOI')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_pp_atoi if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'pk', 'ATOI'),
                        fetch_data(row, year, 2, 'pk', 'ATOI'),
                        fetch_data(row, year, 3, 'pk', 'ATOI'),
                        fetch_data(row, year, 4, 'pk', 'ATOI'),
                        fetch_data(row, year, 5, 'pk', 'ATOI'),
                        fetch_data(row, year, 5, 'pk', 'ATOI') - prev_avg,
                    ]
            elif dependent_variable == 'forward_EV_Gper60':
                # filter out:
                    # defence
                    # players with < 50 GP in Y5
                    # players with < 50 GP in Y4

                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
                    prev_ev_gper60 = [fetch_data(row, year, 1, 'ev', 'G/60'), fetch_data(row, year, 2, 'ev', 'G/60'), fetch_data(row, year, 3, 'ev', 'G/60'), fetch_data(row, year, 4, 'ev', 'G/60')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_ev_gper60 if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'ev', 'ATOI'),
                        fetch_data(row, year, 2, 'ev', 'ATOI'),
                        fetch_data(row, year, 3, 'ev', 'ATOI'),
                        fetch_data(row, year, 4, 'ev', 'ATOI'),
                        fetch_data(row, year, 5, 'ev', 'ATOI'),
                        fetch_data(row, year, 1, 'ev', 'G/60'),
                        fetch_data(row, year, 2, 'ev', 'G/60'),
                        fetch_data(row, year, 3, 'ev', 'G/60'),
                        fetch_data(row, year, 4, 'ev', 'G/60'),
                        fetch_data(row, year, 5, 'ev', 'G/60'),
                        fetch_data(row, year, 5, 'ev', 'G/60') - prev_avg,
                        fetch_data(row, year, 1, 'ev', 'ixG/60'),
                        fetch_data(row, year, 2, 'ev', 'ixG/60'),
                        fetch_data(row, year, 3, 'ev', 'ixG/60'),
                        fetch_data(row, year, 4, 'ev', 'ixG/60'),
                        fetch_data(row, year, 5, 'ev', 'ixG/60')
                    ]
            elif dependent_variable == 'defence_EV_Gper60':
                # filter out:
                    # forwards
                    # players with < 50 GP in Y5
                    # players with < 50 GP in Y4

                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
                    prev_ev_gper60 = [fetch_data(row, year, 1, 'ev', 'G/60'), fetch_data(row, year, 2, 'ev', 'G/60'), fetch_data(row, year, 3, 'ev', 'G/60'), fetch_data(row, year, 4, 'ev', 'G/60')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_ev_gper60 if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'ev', 'ATOI'),
                        fetch_data(row, year, 2, 'ev', 'ATOI'),
                        fetch_data(row, year, 3, 'ev', 'ATOI'),
                        fetch_data(row, year, 4, 'ev', 'ATOI'),
                        fetch_data(row, year, 5, 'ev', 'ATOI'),
                        fetch_data(row, year, 1, 'ev', 'G/60'),
                        fetch_data(row, year, 2, 'ev', 'G/60'),
                        fetch_data(row, year, 3, 'ev', 'G/60'),
                        fetch_data(row, year, 4, 'ev', 'G/60'),
                        fetch_data(row, year, 5, 'ev', 'G/60'),
                        fetch_data(row, year, 5, 'ev', 'G/60') - prev_avg,
                        fetch_data(row, year, 1, 'ev', 'ixG/60'),
                        fetch_data(row, year, 2, 'ev', 'ixG/60'),
                        fetch_data(row, year, 3, 'ev', 'ixG/60'),
                        fetch_data(row, year, 4, 'ev', 'ixG/60'),
                        fetch_data(row, year, 5, 'ev', 'ixG/60')
                    ]
            elif dependent_variable == 'forward_PP_Gper60':
                # filter out:
                    # defence
                    # players with < 50 PPTOI in Y5
                    # players with < 50 PPTOI in Y4

                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, 'pp', 'ATOI')*fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, 'pp', 'ATOI')*fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
                    prev_pp_gper60 = [fetch_data(row, year, 1, 'pp', 'G/60'), fetch_data(row, year, 2, 'pp', 'G/60'), fetch_data(row, year, 3, 'pp', 'G/60'), fetch_data(row, year, 4, 'pp', 'G/60')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_pp_gper60 if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'pp', 'ATOI'),
                        fetch_data(row, year, 2, 'pp', 'ATOI'),
                        fetch_data(row, year, 3, 'pp', 'ATOI'),
                        fetch_data(row, year, 4, 'pp', 'ATOI'),
                        fetch_data(row, year, 5, 'pp', 'ATOI'),
                        fetch_data(row, year, 1, 'pp', 'G/60'),
                        fetch_data(row, year, 2, 'pp', 'G/60'),
                        fetch_data(row, year, 3, 'pp', 'G/60'),
                        fetch_data(row, year, 4, 'pp', 'G/60'),
                        fetch_data(row, year, 5, 'pp', 'G/60'),
                        fetch_data(row, year, 5, 'pp', 'G/60') - prev_avg,
                        fetch_data(row, year, 1, 'pp', 'ixG/60'),
                        fetch_data(row, year, 2, 'pp', 'ixG/60'),
                        fetch_data(row, year, 3, 'pp', 'ixG/60'),
                        fetch_data(row, year, 4, 'pp', 'ixG/60'),
                        fetch_data(row, year, 5, 'pp', 'ixG/60'),
                        fetch_data(row, year, 1, 'ev', 'G/60'),
                        fetch_data(row, year, 2, 'ev', 'G/60'),
                        fetch_data(row, year, 3, 'ev', 'G/60'),
                        fetch_data(row, year, 4, 'ev', 'G/60'),
                        fetch_data(row, year, 5, 'ev', 'G/60'),
                        fetch_data(row, year, 1, 'ev', 'ixG/60'),
                        fetch_data(row, year, 2, 'ev', 'ixG/60'),
                        fetch_data(row, year, 3, 'ev', 'ixG/60'),
                        fetch_data(row, year, 4, 'ev', 'ixG/60'),
                        fetch_data(row, year, 5, 'ev', 'ixG/60')
                    ]
            elif dependent_variable == 'defence_PP_Gper60':
                # filter out:
                    # forwards
                    # players with < 50 PPTOI in Y5
                    # players with < 50 PPTOI in Y4

                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, 'pp', 'ATOI')*fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, 'pp', 'ATOI')*fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
                    prev_pp_gper60 = [fetch_data(row, year, 1, 'pp', 'G/60'), fetch_data(row, year, 2, 'pp', 'G/60'), fetch_data(row, year, 3, 'pp', 'G/60'), fetch_data(row, year, 4, 'pp', 'G/60')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_pp_gper60 if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'pp', 'ATOI'),
                        fetch_data(row, year, 2, 'pp', 'ATOI'),
                        fetch_data(row, year, 3, 'pp', 'ATOI'),
                        fetch_data(row, year, 4, 'pp', 'ATOI'),
                        fetch_data(row, year, 5, 'pp', 'ATOI'),
                        fetch_data(row, year, 1, 'pp', 'G/60'),
                        fetch_data(row, year, 2, 'pp', 'G/60'),
                        fetch_data(row, year, 3, 'pp', 'G/60'),
                        fetch_data(row, year, 4, 'pp', 'G/60'),
                        fetch_data(row, year, 5, 'pp', 'G/60'),
                        fetch_data(row, year, 5, 'pp', 'G/60') - prev_avg,
                        fetch_data(row, year, 1, 'pp', 'ixG/60'),
                        fetch_data(row, year, 2, 'pp', 'ixG/60'),
                        fetch_data(row, year, 3, 'pp', 'ixG/60'),
                        fetch_data(row, year, 4, 'pp', 'ixG/60'),
                        fetch_data(row, year, 5, 'pp', 'ixG/60'),
                        fetch_data(row, year, 1, 'ev', 'G/60'),
                        fetch_data(row, year, 2, 'ev', 'G/60'),
                        fetch_data(row, year, 3, 'ev', 'G/60'),
                        fetch_data(row, year, 4, 'ev', 'G/60'),
                        fetch_data(row, year, 5, 'ev', 'G/60'),
                        fetch_data(row, year, 1, 'ev', 'ixG/60'),
                        fetch_data(row, year, 2, 'ev', 'ixG/60'),
                        fetch_data(row, year, 3, 'ev', 'ixG/60'),
                        fetch_data(row, year, 4, 'ev', 'ixG/60'),
                        fetch_data(row, year, 5, 'ev', 'ixG/60')
                    ]
            elif dependent_variable == 'forward_PK_Gper60':
                # filter out:
                    # defence
                    # players with < 50 PPTOI in Y5
                    # players with < 50 PPTOI in Y4

                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, 'pk', 'ATOI')*fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, 'pk', 'ATOI')*fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
                    prev_pk_gper60 = [fetch_data(row, year, 1, 'pk', 'G/60'), fetch_data(row, year, 2, 'pk', 'G/60'), fetch_data(row, year, 3, 'pk', 'G/60'), fetch_data(row, year, 4, 'pk', 'G/60')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_pk_gper60 if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'pk', 'ATOI'),
                        fetch_data(row, year, 2, 'pk', 'ATOI'),
                        fetch_data(row, year, 3, 'pk', 'ATOI'),
                        fetch_data(row, year, 4, 'pk', 'ATOI'),
                        fetch_data(row, year, 5, 'pk', 'ATOI'),
                        fetch_data(row, year, 1, 'pk', 'G/60'),
                        fetch_data(row, year, 2, 'pk', 'G/60'),
                        fetch_data(row, year, 3, 'pk', 'G/60'),
                        fetch_data(row, year, 4, 'pk', 'G/60'),
                        fetch_data(row, year, 5, 'pk', 'G/60'),
                        fetch_data(row, year, 5, 'pk', 'G/60') - prev_avg,
                        fetch_data(row, year, 1, 'pk', 'ixG/60'),
                        fetch_data(row, year, 2, 'pk', 'ixG/60'),
                        fetch_data(row, year, 3, 'pk', 'ixG/60'),
                        fetch_data(row, year, 4, 'pk', 'ixG/60'),
                        fetch_data(row, year, 5, 'pk', 'ixG/60'),
                    ]
            elif dependent_variable == 'defence_PK_Gper60':
                # filter out:
                    # forwards
                    # players with < 50 PPTOI in Y5
                    # players with < 50 PPTOI in Y4

                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, 'pk', 'ATOI')*fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, 'pk', 'ATOI')*fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
                    prev_pk_gper60 = [fetch_data(row, year, 1, 'pk', 'G/60'), fetch_data(row, year, 2, 'pk', 'G/60'), fetch_data(row, year, 3, 'pk', 'G/60'), fetch_data(row, year, 4, 'pk', 'G/60')]
                    try:
                        prev_avg = statistics.mean([x for x in prev_pk_gper60 if not np.isnan(x)])
                    except statistics.StatisticsError:
                        prev_avg = 0
                    
                    instance_df.loc[f"{row['Player']} {year+1}"] = [
                        row['Player'], 
                        year+1, row['Position'],
                        calc_age(row['Date of Birth'], year), 
                        row['Height (in)'], 
                        row['Weight (lbs)'],
                        fetch_data(row, year, 1, None, 'GP'),
                        fetch_data(row, year, 2, None, 'GP'),
                        fetch_data(row, year, 3, None, 'GP'),
                        fetch_data(row, year, 4, None, 'GP'),
                        fetch_data(row, year, 5, None, 'GP'),
                        fetch_data(row, year, 1, 'pk', 'ATOI'),
                        fetch_data(row, year, 2, 'pk', 'ATOI'),
                        fetch_data(row, year, 3, 'pk', 'ATOI'),
                        fetch_data(row, year, 4, 'pk', 'ATOI'),
                        fetch_data(row, year, 5, 'pk', 'ATOI'),
                        fetch_data(row, year, 1, 'pk', 'G/60'),
                        fetch_data(row, year, 2, 'pk', 'G/60'),
                        fetch_data(row, year, 3, 'pk', 'G/60'),
                        fetch_data(row, year, 4, 'pk', 'G/60'),
                        fetch_data(row, year, 5, 'pk', 'G/60'),
                        fetch_data(row, year, 5, 'pk', 'G/60') - prev_avg,
                        fetch_data(row, year, 1, 'pk', 'ixG/60'),
                        fetch_data(row, year, 2, 'pk', 'ixG/60'),
                        fetch_data(row, year, 3, 'pk', 'ixG/60'),
                        fetch_data(row, year, 4, 'pk', 'ixG/60'),
                        fetch_data(row, year, 5, 'pk', 'ixG/60'),
                    ]

    if download_file == True:
        filename = f'{dependent_variable}_instance_training_data_{end_year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        instance_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return instance_df

def create_year_restricted_instance_df(proj_stat, position, prev_years, situation, year=2023, download_file=True):
    if proj_stat == 'GP':
        instance_df = create_instance_df(f'{position}_GP', ['Player', 'Year', 'Position', 'Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 'Y5 dGP'], scrape_player_statistics(True), True)
        if prev_years == 4:
            instance_df = instance_df.loc[(instance_df['Y1 GP'] >= 60) & (instance_df['Y2 GP'] >= 60) & (instance_df['Y3 GP'] >= 60) & (instance_df['Y4 GP'] >= 60)]
            input_shape = (7,)
        elif prev_years == 3:
            instance_df = instance_df.loc[(instance_df['Y2 GP'] >= 60) & (instance_df['Y3 GP'] >= 60) & (instance_df['Y4 GP'] >= 60)]
            input_shape = (6,)
        elif prev_years == 2:
            instance_df = instance_df.loc[(instance_df['Y3 GP'] >= 60) & (instance_df['Y4 GP'] >= 60)]
            input_shape = (5,)
        elif prev_years == 1:
            instance_df = instance_df.loc[(instance_df['Y4 GP'] >= 40)]
            input_shape = (4,)
        else:
            print('Invalid prev_years parameter.')

    elif proj_stat == 'ATOI':
        instance_df = create_instance_df(f'{position}_{situation}_ATOI', ['Player', 'Year', 'Position', 'Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI'], scrape_player_statistics(True), True)
        if prev_years == 4:
            instance_df = instance_df.loc[(instance_df['Y1 GP'] >= 40) & (instance_df['Y2 GP'] >= 40) & (instance_df['Y3 GP'] >= 40) & (instance_df['Y4 GP'] >= 40)]
            input_shape = (7,)
        elif prev_years == 3:
            instance_df = instance_df.loc[(instance_df['Y2 GP'] >= 40) & (instance_df['Y3 GP'] >= 40) & (instance_df['Y4 GP'] >= 40)]
            input_shape = (6,)
        elif prev_years == 2:
            instance_df = instance_df.loc[(instance_df['Y3 GP'] >= 40) & (instance_df['Y4 GP'] >= 40)]
            input_shape = (5,)
        elif prev_years == 1:
            instance_df = instance_df.loc[(instance_df['Y4 GP'] >= 40)]
            input_shape = (4,)
        else:
            print('Invalid prev_years parameter.')

    elif proj_stat == 'Gper60':
        if situation == 'EV':
            if position == 'forward':
                instance_df = create_instance_df(f'{position}_{situation}_Gper60', [
                    'Player', 'Year', 'Position', 'Age', 'Height', 'Weight',
                    'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ], scrape_player_statistics(True), True)      
                if prev_years == 4:
                    instance_df = instance_df.loc[(instance_df['Y1 GP'] >= 50) & (instance_df['Y2 GP'] >= 50) & (instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                    input_shape = (11,)
                elif prev_years == 3:
                    instance_df = instance_df.loc[(instance_df['Y2 GP'] >= 50) & (instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                    input_shape = (9,)
                elif prev_years == 2:
                    instance_df = instance_df.loc[(instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                    input_shape = (7,)
                elif prev_years == 1:
                    instance_df = instance_df.loc[(instance_df['Y4 GP'] >= 50)]
                    input_shape = (5,)
                else:
                    print('Invalid prev_years parameter.')
            elif position == 'defence':
                instance_df = create_instance_df(f'{position}_{situation}_Gper60', [
                    'Player', 'Year', 'Position', 'Age', 'Height', 'Weight',
                    'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ], scrape_player_statistics(True), True)        
                if prev_years == 4:
                    instance_df = instance_df.loc[(instance_df['Y1 GP'] >= 50) & (instance_df['Y2 GP'] >= 50) & (instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                    input_shape = (11,)
                elif prev_years == 3:
                    instance_df = instance_df.loc[(instance_df['Y2 GP'] >= 50) & (instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                    input_shape = (9,)
                elif prev_years == 2:
                    instance_df = instance_df.loc[(instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                    input_shape = (7,)
                elif prev_years == 1:
                    instance_df = instance_df.loc[(instance_df['Y4 GP'] >= 50)]
                    input_shape = (5,)
                else:
                    print('Invalid prev_years parameter.')            
            else:
                print('Position Error')

        elif situation == 'PP':
            instance_df = create_instance_df(f'{position}_{situation}_Gper60', [
                'Player', 'Year', 'Position', 'Age', 'Height', 'Weight',
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 PP ATOI', f'Y2 PP ATOI', f'Y3 PP ATOI', f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y1 PP G/60', f'Y2 PP G/60', f'Y3 PP G/60', f'Y4 PP G/60', f'Y5 PP G/60', f'Y5 dPP G/60',
                f'Y1 PP ixG/60', f'Y2 PP ixG/60', f'Y3 PP ixG/60', f'Y4 PP ixG/60', f'Y5 PP ixG/60',
                f'Y1 EV G/60', f'Y2 EV G/60', f'Y3 EV G/60', f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y1 EV ixG/60', f'Y2 EV ixG/60', f'Y3 EV ixG/60', f'Y4 EV ixG/60', f'Y5 EV ixG/60'], 
                scrape_player_statistics(True), True)   
            if prev_years == 4:
                instance_df = instance_df.loc[(instance_df['Y1 PP ATOI']*instance_df['Y1 GP'] >= 50) & (instance_df['Y2 PP ATOI']*instance_df['Y2 GP'] >= 50) & (instance_df['Y3 PP ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PP ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (19,)
            elif prev_years == 3:
                instance_df = instance_df.loc[(instance_df['Y2 PP ATOI']*instance_df['Y2 GP'] >= 50) & (instance_df['Y3 PP ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PP ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (15,)
            elif prev_years == 2:
                instance_df = instance_df.loc[(instance_df['Y3 PP ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PP ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (11,)
            elif prev_years == 1:
                instance_df = instance_df.loc[(instance_df['Y4 PP ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (7,)
            else:
                print('Invalid prev_years parameter.')  

        elif situation == 'PK':
            instance_df = create_instance_df(f'{position}_{situation}_Gper60', [
                'Player', 'Year', 'Position', 'Age', 'Height', 'Weight',
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 PK ATOI', f'Y2 PK ATOI', f'Y3 PK ATOI', f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y1 PK G/60', f'Y2 PK G/60', f'Y3 PK G/60', f'Y4 PK G/60', f'Y5 PK G/60', f'Y5 dPK G/60',
                f'Y1 PK ixG/60', f'Y2 PK ixG/60', f'Y3 PK ixG/60', f'Y4 PK ixG/60', f'Y5 PK ixG/60'], 
                scrape_player_statistics(True), True)   
            if prev_years == 4:
                instance_df = instance_df.loc[(instance_df['Y1 PK ATOI']*instance_df['Y1 GP'] >= 50) & (instance_df['Y2 PK ATOI']*instance_df['Y2 GP'] >= 50) & (instance_df['Y3 PK ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PK ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (7,)
            elif prev_years == 3:
                instance_df = instance_df.loc[(instance_df['Y2 PK ATOI']*instance_df['Y2 GP'] >= 50) & (instance_df['Y3 PK ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PK ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (6,)
            elif prev_years == 2:
                instance_df = instance_df.loc[(instance_df['Y3 PK ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PK ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (5,)
            elif prev_years == 1:
                instance_df = instance_df.loc[(instance_df['Y4 PK ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (4,)
            else:
                print('Invalid prev_years parameter.')  

        else:
            print('Situation Error')

    if download_file == True:
        if situation == None:
            filename = f'{position}_{proj_stat}_{prev_years}year_instance_training_data_{year}'
        else:
            filename = f'{position}_{situation}_{proj_stat}_{prev_years}year_instance_training_data_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        instance_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return instance_df, input_shape

def extract_instance_data(instance_df, proj_stat, prev_years, situation, position=None):
    # position = None would indicate that same features are considered for forward and defence neural network

    X = []
    y = []

    if proj_stat == 'GP':
        if prev_years == 4:
            for index, row in instance_df.iterrows():
                X.append([row['Age'], row['Height'], row['Weight'], row['Y1 GP'], row['Y2 GP'], row['Y3 GP'], row['Y4 GP']]) # features
                y.append(row['Y5 dGP']) # target
        elif prev_years == 3:
            for index, row in instance_df.iterrows():
                X.append([row['Age'], row['Height'], row['Weight'], row['Y2 GP'], row['Y3 GP'], row['Y4 GP']]) # features
                y.append(row['Y5 dGP']) # target
        elif prev_years == 2:
            for index, row in instance_df.iterrows():
                X.append([row['Age'], row['Height'], row['Weight'], row['Y3 GP'], row['Y4 GP']]) # features
                y.append(row['Y5 dGP']) # target
        elif prev_years == 1:
            for index, row in instance_df.iterrows():
                X.append([row['Age'], row['Height'], row['Weight'], row['Y4 GP']]) # features
                y.append(row['Y5 dGP']) # target
        else:
            print('Invalid prev_years parameter.')

    elif proj_stat == 'ATOI':
        if prev_years == 4:
            instance_df[[f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI']] = instance_df[[f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI']].fillna(0)
            for index, row in instance_df.iterrows():
                X.append([row['Age'], row['Height'], row['Weight'], row[f'Y1 {situation} ATOI'], row[f'Y2 {situation} ATOI'], row[f'Y3 {situation} ATOI'], row[f'Y4 {situation} ATOI']]) # features
                y.append(row[f'Y5 d{situation} ATOI']) # target
        elif prev_years == 3:
            instance_df[[f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI']] = instance_df[[f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI']].fillna(0)
            for index, row in instance_df.iterrows():
                X.append([row['Age'], row['Height'], row['Weight'], row[f'Y2 {situation} ATOI'], row[f'Y3 {situation} ATOI'], row[f'Y4 {situation} ATOI']]) # features
                y.append(row[f'Y5 d{situation} ATOI']) # target
        elif prev_years == 2:
            instance_df[[f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI']] = instance_df[[f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI']].fillna(0)
            for index, row in instance_df.iterrows():
                X.append([row['Age'], row['Height'], row['Weight'], row[f'Y3 {situation} ATOI'], row[f'Y4 {situation} ATOI']]) # features
                y.append(row[f'Y5 d{situation} ATOI']) # target
        elif prev_years == 1:
            instance_df[[f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI']] = instance_df[[f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', f'Y5 d{situation} ATOI']].fillna(0)
            for index, row in instance_df.iterrows():
                X.append([row['Age'], row['Height'], row['Weight'], row[f'Y4 {situation} ATOI']]) # features
                y.append(row[f'Y5 d{situation} ATOI']) # target
        else:
            print('Invalid prev_years parameter.')

    elif proj_stat == 'Gper60':
        if situation == 'EV':
            if position == 'forward':
                if prev_years == 4:
                    instance_df[[
                    'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                    ]] = instance_df[[
                    'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                    ]].fillna(0)
                    for index, row in instance_df.iterrows():
                        X.append([row['Age'], row['Height'], row['Weight'],
                                row[f'Y1 {situation} G/60'], row[f'Y2 {situation} G/60'], row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                                row[f'Y1 {situation} ixG/60'], row[f'Y2 {situation} ixG/60'], row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60']
                                ]) # features
                        y.append(row[f'Y5 {situation} G/60']) # target
                elif prev_years == 3:
                    instance_df[[
                    'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                    ]] = instance_df[[
                    'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                    ]].fillna(0)
                    for index, row in instance_df.iterrows():
                        X.append([row['Age'], row['Height'], row['Weight'],
                                row[f'Y2 {situation} G/60'], row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                                row[f'Y2 {situation} ixG/60'], row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60']
                                ]) # features
                        y.append(row[f'Y5 {situation} G/60']) # target
                elif prev_years == 2:
                    instance_df[[
                    'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                    ]] = instance_df[[
                    'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                    ]].fillna(0)
                    for index, row in instance_df.iterrows():
                        X.append([row['Age'], row['Height'], row['Weight'],
                                row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                                row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60']
                                ]) # features
                        y.append(row[f'Y5 {situation} G/60']) # target
                elif prev_years == 1:
                    instance_df[[
                    'Y4 GP', 'Y5 GP', 
                    f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                    ]] = instance_df[[
                    'Y4 GP', 'Y5 GP', 
                    f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                    ]].fillna(0)
                    for index, row in instance_df.iterrows():
                        X.append([row['Age'], row['Height'], row['Weight'],
                                row[f'Y4 {situation} G/60'],
                                row[f'Y4 {situation} ixG/60']
                                ]) # features
                        y.append(row[f'Y5 {situation} G/60']) # target
                else:
                    print('Invalid prev_years parameter.')

            elif position == 'defence':
                if prev_years == 4:
                    instance_df[[
                    'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ]] = instance_df[[
                    'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ]].fillna(0)
                    for index, row in instance_df.iterrows():
                        X.append([row['Age'], row['Height'], row['Weight'],
                                row[f'Y1 {situation} G/60'], row[f'Y2 {situation} G/60'], row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                                row[f'Y1 {situation} ixG/60'], row[f'Y2 {situation} ixG/60'], row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60']
                                ]) # features
                        y.append(row[f'Y5 {situation} G/60']) # target
                elif prev_years == 3:
                    instance_df[[
                    'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ]] = instance_df[[
                    'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ]].fillna(0)
                    for index, row in instance_df.iterrows():
                        X.append([row['Age'], row['Height'], row['Weight'],
                                row[f'Y2 {situation} G/60'], row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                                row[f'Y2 {situation} ixG/60'], row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60'],
                                ]) # features
                        y.append(row[f'Y5 {situation} G/60']) # target
                elif prev_years == 2:
                    instance_df[[
                    'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ]] = instance_df[[
                    'Y3 GP', 'Y4 GP', 'Y5 GP', 
                    f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ]].fillna(0)
                    for index, row in instance_df.iterrows():
                        X.append([row['Age'], row['Height'], row['Weight'],
                                row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                                row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60'],
                                ]) # features
                        y.append(row[f'Y5 {situation} G/60']) # target
                elif prev_years == 1:
                    instance_df[[
                    'Y4 GP', 'Y5 GP', 
                    f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ]] = instance_df[[
                    'Y4 GP', 'Y5 GP', 
                    f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                    f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                    f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                    ]].fillna(0)
                    for index, row in instance_df.iterrows():
                        X.append([row['Age'], row['Height'], row['Weight'],
                                row[f'Y4 {situation} G/60'],
                                row[f'Y4 {situation} ixG/60'],
                                ]) # features
                        y.append(row[f'Y5 {situation} G/60']) # target
                else:
                    print('Invalid prev_years parameter.')      

        elif situation == 'PP':
            # features don't depend on the position of the player
            if prev_years == 4:
                instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                f'Y1 EV G/60', f'Y2 EV G/60', f'Y3 EV G/60', f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y1 EV ixG/60', f'Y2 EV ixG/60', f'Y3 EV ixG/60', f'Y4 EV ixG/60', f'Y5 EV ixG/60'
                ]] = instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                f'Y1 EV G/60', f'Y2 EV G/60', f'Y3 EV G/60', f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y1 EV ixG/60', f'Y2 EV ixG/60', f'Y3 EV ixG/60', f'Y4 EV ixG/60', f'Y5 EV ixG/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y1 {situation} G/60'], row[f'Y2 {situation} G/60'], row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                            row[f'Y1 {situation} ixG/60'], row[f'Y2 {situation} ixG/60'], row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60'],
                            row[f'Y1 EV G/60'], row[f'Y2 EV G/60'], row[f'Y3 EV G/60'], row[f'Y4 EV G/60'],
                            row[f'Y1 EV ixG/60'], row[f'Y2 EV ixG/60'], row[f'Y3 EV ixG/60'], row[f'Y4 EV ixG/60']
                            ]) # features
                    y.append(row[f'Y5 {situation} G/60']) # target
            elif prev_years == 3:
                instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                f'Y2 EV G/60', f'Y3 EV G/60', f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y2 EV ixG/60', f'Y3 EV ixG/60', f'Y4 EV ixG/60', f'Y5 EV ixG/60'
                ]] = instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                f'Y2 EV G/60', f'Y3 EV G/60', f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y2 EV ixG/60', f'Y3 EV ixG/60', f'Y4 EV ixG/60', f'Y5 EV ixG/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y2 {situation} G/60'], row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                            row[f'Y2 {situation} ixG/60'], row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60'],
                            row[f'Y2 EV G/60'], row[f'Y3 EV G/60'], row[f'Y4 EV G/60'],
                            row[f'Y2 EV ixG/60'], row[f'Y3 EV ixG/60'], row[f'Y4 EV ixG/60']
                            ]) # features
                    y.append(row[f'Y5 {situation} G/60']) # target
            elif prev_years == 2:
                instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                f'Y3 EV G/60', f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y3 EV ixG/60', f'Y4 EV ixG/60', f'Y5 EV ixG/60'
                ]] = instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                f'Y3 EV G/60', f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y3 EV ixG/60', f'Y4 EV ixG/60', f'Y5 EV ixG/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y3 {situation} G/60'], row[f'Y4 {situation} G/60'],
                            row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60'],
                            row[f'Y3 EV G/60'], row[f'Y4 EV G/60'],
                            row[f'Y3 EV ixG/60'], row[f'Y4 EV ixG/60']
                            ]) # features
                    y.append(row[f'Y5 {situation} G/60']) # target
            elif prev_years == 1:
                instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y4 EV ixG/60', f'Y5 EV ixG/60'
                ]] = instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60',
                f'Y4 EV G/60', f'Y5 EV G/60',
                f'Y4 EV ixG/60', f'Y5 EV ixG/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y4 {situation} G/60'],
                            row[f'Y4 {situation} ixG/60'],
                            row[f'Y4 EV G/60'],
                            row[f'Y4 EV ixG/60']
                            ]) # features
                    y.append(row[f'Y5 {situation} G/60']) # target
            else:
                print('Invalid prev_years parameter.')

        elif situation == 'PK':
            # features don't depend on the position of the player
            if prev_years == 4:
                instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                ]] = instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y1 {situation} G/60', f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y1 {situation} ixG/60', f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y1 {situation} ixG/60'], row[f'Y2 {situation} ixG/60'], row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60']
                            ]) # features
                    y.append(row[f'Y5 {situation} G/60']) # target
            elif prev_years == 3:
                instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                ]] = instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y2 {situation} G/60', f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y2 {situation} ixG/60', f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y2 {situation} ixG/60'], row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60']
                            ]) # features
                    y.append(row[f'Y5 {situation} G/60']) # target
            elif prev_years == 2:
                instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                ]] = instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y3 {situation} G/60', f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y3 {situation} ixG/60', f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y3 {situation} ixG/60'], row[f'Y4 {situation} ixG/60']
                            ]) # features
                    y.append(row[f'Y5 {situation} G/60']) # target
            elif prev_years == 1:
                instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                ]] = instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y4 {situation} G/60', f'Y5 {situation} G/60', f'Y5 d{situation} G/60',
                f'Y4 {situation} ixG/60', f'Y5 {situation} ixG/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y4 {situation} ixG/60']
                            ]) # features
                    y.append(row[f'Y5 {situation} G/60']) # target
            else:
                print('Invalid prev_years parameter.')
            
    X = np.array(X)
    y = np.array(y)

    return X, y

def calc_age(dob, year):
    dob = date(int(dob.split('-')[0]), int(dob.split('-')[1]), int(dob.split('-')[2])) #year, month, date
    target_date = date(year, 10, 1) # Age calculated as of October 1st of the season.
    delta_days = target_date - dob
    age = round(delta_days.days/365.24,3)

    return age

def fetch_data(row, year, yX, situation, stat):
    situation_reassignment = {'ev': 'EV', '5v5': '5v5', 'pp': 'PP', 'pk': 'PK'}
    try:
        if situation == None:
            result = row[f'{year+yX-4} {stat}']
            if stat.strip() == 'GP':
                if year+yX-4 == 2021:
                    result = result/56*82
                elif stat == 'GP' and year+yX-4 == 2020:
                    result = result/69.5*82
                elif stat == 'GP' and year+yX-4 == 2013:
                    result = result/48*82
                try:
                    result = int(result)
                except ValueError:
                    pass
        else:
            result = row[f'{year+yX-4} {situation_reassignment[situation]} {stat}']
    except KeyError:
        result = np.nan
    return result

def permutation_feature_importance(model, X_scaled, y, features, scoring='neg_mean_absolute_error'):
    # Compute permutation importances
    result = permutation_importance(model, X_scaled, y, scoring=scoring)
    sorted_idx = result.importances_mean.argsort()

    # Define color map and normalization
    cmap = cm.get_cmap('seismic_r')
    normalize = plt.Normalize(result.importances_mean[sorted_idx].min(), result.importances_mean[sorted_idx].max())

    # Create a scalar mappable
    scalar_mappable = ScalarMappable(norm=normalize, cmap=cmap)

    # Plot permutation importances
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = scalar_mappable.to_rgba(result.importances_mean[sorted_idx])
    ax.barh(range(X_scaled.shape[1]), result.importances_mean[sorted_idx], color=bar_colors)
    ax.set_yticks(range(X_scaled.shape[1]))
    ax.set_yticklabels(np.array(features)[sorted_idx])
    ax.set_title("Permutation Feature Importance Analysis", weight='bold', fontsize=15, pad=20)
    ax.text(0.5, 1.02, 'Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but how important this feature is for a particular model.', ha='center', va='center', transform=ax.transAxes, fontsize=7, fontstyle='italic')
    ax.set_xlabel("Importance Score", weight='bold')
    ax.set_ylabel("Features", weight='bold')
    ax.tick_params(length=0)
    plt.box(True) # False to hide box
    # plt.tight_layout()
    plt.show()

def make_projection_df(stat_df, year=2024):
    projection_df = pd.DataFrame(columns=['Player', 'Position', 'Age', 'Height', 'Weight'])

    for index, row in stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)].iterrows():
        projection_df.loc[index] = [
            row['Player'], 
            row['Position'],
            round(calc_age(row['Date of Birth'], year-1),1),
            row['Height (in)'], 
            row['Weight (lbs)']]

    projection_df = projection_df.sort_values('Player')
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    return projection_df

def make_forward_gp_projections(stat_df, projection_df, download_file, year=2024):
    # Forwards with 4 seasons of > 50 GP: Parent model 1 (126-42-14-6-1), 5 epochs, standard scaler
    # Forwards with 3 seasons of > 50 GP: Parent model 12 (8-1), 50 epochs, standard scaler
    # Forwards with 2 seasons of > 50 GP: Parent model 6 (32-16-8-1), 50 epochs, minmax scaler
    # Forwards with 1 seasons of > 50 GP: Parent model 6 (32-16-8-1), 100 epochs, minmax scaler

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(126, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(42, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('GP', 'forward', 4, None)
    X_4, y_4 = extract_instance_data(instance_df_y4, 'GP', 4, None)
    instance_df_y3, _ = create_year_restricted_instance_df('GP', 'forward', 3, None)
    X_3, y_3 = extract_instance_data(instance_df_y3, 'GP', 3, None)
    instance_df_y2, _ = create_year_restricted_instance_df('GP', 'forward', 2, None)
    X_2, y_2 = extract_instance_data(instance_df_y2, 'GP', 2, None)
    instance_df_y1, _ = create_year_restricted_instance_df('GP', 'forward', 1, None)
    X_1, y_1 = extract_instance_data(instance_df_y1, 'GP', 1, None)

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=5, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=50, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=50, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=100, verbose=1)

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] != 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 50 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 50 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 50 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].iloc[0]*gp_adjustment_factor[year-4],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
        
    for player in yr1_group:
        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'GP'

    # Assuring that games played is <= 82
    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = min(proj_y_4[index][0] + statistics.mean(statline[-4:]), 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = min(proj_y_3[index][0] + statistics.mean(statline[-3:]), 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = min(proj_y_2[index][0] + statistics.mean(statline[-2:]), 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = min(proj_y_1[index][0] + statistics.mean(statline[-1:]), 82) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_defence_gp_projections(stat_df, projection_df, download_file, year=2024):
    # Defence with 4 seasons of > 50 GP: Parent model 5 (64-28-12-1), 30 epochs, standard scaler
    # Defence with 3 seasons of > 50 GP: Parent model 2 (64-32-16-8-1), 30 epochs, minmax scaler
    # Defence with 2 seasons of > 50 GP: Parent model 10 (16-4-1), 10 epochs, standard scaler
    # Forwards with 1 season            : Parent model 7 (128-64-1), 50 epochs, minmax scaler

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('GP', 'defence', 4, None)
    X_4, y_4 = extract_instance_data(instance_df_y4, 'GP', 4, None)
    instance_df_y3, _ = create_year_restricted_instance_df('GP', 'defence', 3, None)
    X_3, y_3 = extract_instance_data(instance_df_y3, 'GP', 3, None)
    instance_df_y2, _ = create_year_restricted_instance_df('GP', 'defence', 2, None)
    X_2, y_2 = extract_instance_data(instance_df_y2, 'GP', 2, None)
    instance_df_y1, _ = create_year_restricted_instance_df('GP', 'defence', 1, None)
    X_1, y_1 = extract_instance_data(instance_df_y1, 'GP', 1, None)

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=30, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=30, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=50, verbose=1)

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] != 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 50 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 50 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 50 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].iloc[0]*gp_adjustment_factor[year-4],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
        
    for player in yr1_group:
        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'GP'

    # Assuring that games played is <= 82
    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = min(proj_y_4[index][0] + statistics.mean(statline[-4:]), 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = min(proj_y_3[index][0] + statistics.mean(statline[-3:]), 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = min(proj_y_2[index][0] + statistics.mean(statline[-2:]), 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = min(proj_y_1[index][0] + statistics.mean(statline[-1:]), 82) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_forward_ev_atoi_projections(stat_df, projection_df, download_file, year=2024):
    # Forwards with 4 seasons of > 40 GP: Parent model 5 (64-28-12-1), 10 epochs, standard scaler
    # Forwards with 3 seasons of > 40 GP: Parent model 11 (24-1), 10 epochs, standard scaler
    # Forwards with 2 seasons of > 40 GP: Parent model 3 (48-24-12-6-1), 10 epochs, standard scaler
    # Forwards with 1 season            : Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(48, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(48, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('ATOI', 'forward', 4, 'EV')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'ATOI', 4, 'EV')
    instance_df_y3, _ = create_year_restricted_instance_df('ATOI', 'forward', 3, 'EV')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV')
    instance_df_y2, _ = create_year_restricted_instance_df('ATOI', 'forward', 2, 'EV')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'ATOI', 2, 'EV')
    instance_df_y1, _ = create_year_restricted_instance_df('ATOI', 'forward', 1, 'EV')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'ATOI', 1, 'EV')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=30, verbose=1)

    # permutation_feature_importance(yr4_model, X_4_scaled, y_4, ['Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP'])

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] != 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 40 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ATOI'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 45 games.
        # Once you reach 45 games, find the ATOI accross these seasons.
        # If they haven't played 45 games in their past 4 seasons, fill the rest of the 45 games with the -1st z-score of the stat.
        if y1_gp >= 45:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 45:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 45:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 45:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 EV ATOI'].mean() - instance_df_y1['Y4 EV ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 45-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0] + statistics.mean(statline[-4:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0] + statistics.mean(statline[-3:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0] + statistics.mean(statline[-2:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0] + statistics.mean(statline[-1:]), 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_defence_ev_atoi_projections(stat_df, projection_df, download_file, year=2024):
    # Defence with 4 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
    # Defence with 3 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
    # Defence with 2 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
    # Defence with 1 season            : Parent model 11 (24-1), 50 epochs, standard scaler

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('ATOI', 'defence', 4, 'EV')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'ATOI', 4, 'EV')
    instance_df_y3, _ = create_year_restricted_instance_df('ATOI', 'defence', 3, 'EV')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV')
    instance_df_y2, _ = create_year_restricted_instance_df('ATOI', 'defence', 2, 'EV')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'ATOI', 2, 'EV')
    instance_df_y1, _ = create_year_restricted_instance_df('ATOI', 'defence', 1, 'EV')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'ATOI', 1, 'EV')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=50, verbose=1)

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] == 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 40 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ATOI'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 30 games.
        # Once you reach 30 games, find the ATOI accross these seasons.
        # If they haven't played 30 games in their past 4 seasons, fill the rest of the 30 games with the -1st z-score of the stat.
        if y1_gp >= 30:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 30:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 30:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 30:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 EV ATOI'].mean() - instance_df_y1['Y4 EV ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 30-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0] + statistics.mean(statline[-4:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0] + statistics.mean(statline[-3:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0] + statistics.mean(statline[-2:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0] + statistics.mean(statline[-1:]), 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_forward_pp_atoi_projections(stat_df, projection_df, download_file, year=2024):
    # Forwards with 4 seasons of > 40 GP: Parent model 5 (64-28-12-1), 30 epochs, minmax scaler
    # Forwards with 3 seasons of > 40 GP: Parent model 5 (64-28-12-1), 30 epochs, minmax scaler
    # Forwards with 2 seasons of > 40 GP: Parent model 5 (64-28-12-1), 30 epochs, minmax scaler
    # Forwards with 1 season            : Parent model 5 (64-28-12-1), 30 epochs, minmax scaler

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('ATOI', 'forward', 4, 'PP')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'ATOI', 4, 'PP')
    instance_df_y3, _ = create_year_restricted_instance_df('ATOI', 'forward', 3, 'PP')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP')
    instance_df_y2, _ = create_year_restricted_instance_df('ATOI', 'forward', 2, 'PP')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'ATOI', 2, 'PP')
    instance_df_y1, _ = create_year_restricted_instance_df('ATOI', 'forward', 1, 'PP')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'ATOI', 1, 'PP')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=30, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=30, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=30, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=30, verbose=1)

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] != 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 40 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 20 games.
        # Once you reach 20 games, find the ATOI accross these seasons.
        # If they haven't played 20 games in their past 4 seasons, fill the rest of the 20 games with the -1st z-score of the stat.
        if y1_gp >= 20:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 PP ATOI'].mean() - instance_df_y1['Y4 PP ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 20-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0] + statistics.mean(statline[-4:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0] + statistics.mean(statline[-3:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0] + statistics.mean(statline[-2:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0] + statistics.mean(statline[-1:]), 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_defence_pp_atoi_projections(stat_df, projection_df, download_file, year=2024):
    # Defence with 4 seasons of > 40 GP: Parent model 5 (64-28-12-1), 30 epochs, minmax scaler
    # Defence with 3 seasons of > 40 GP: Parent model 3 (48-24-12-6-1), 100 epochs, minmax scaler
    # Defence with 2 seasons of > 40 GP: Parent model 1 (126-42-14-6-1), 30 epochs, standard scaler
    # Defence with 1 seasons of > 40 GP: Parent model 10 (16-4-1), 50 epochs, standard scaler

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(48, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(126, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(42, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('ATOI', 'defence', 4, 'PP')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'ATOI', 4, 'PP')
    instance_df_y3, _ = create_year_restricted_instance_df('ATOI', 'defence', 3, 'PP')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP')
    instance_df_y2, _ = create_year_restricted_instance_df('ATOI', 'defence', 2, 'PP')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'ATOI', 2, 'PP')
    instance_df_y1, _ = create_year_restricted_instance_df('ATOI', 'defence', 1, 'PP')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'ATOI', 1, 'PP')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=30, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=100, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=30, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=50, verbose=1)

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] == 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 40 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 20 games.
        # Once you reach 20 games, find the ATOI accross these seasons.
        # If they haven't played 20 games in their past 4 seasons, fill the rest of the 20 games with the -1st z-score of the stat.
        if y1_gp >= 20:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 PP ATOI'].mean() - instance_df_y1['Y4 PP ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 20-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])


    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0] + statistics.mean(statline[-4:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0] + statistics.mean(statline[-3:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0] + statistics.mean(statline[-2:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0] + statistics.mean(statline[-1:]), 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_forward_pk_atoi_projections(stat_df, projection_df, download_file, year=2024):
    # Forwards with 4 seasons of > 40 GP: Parent model 3 (48-24-12-6-1), 10 epochs, minmax scaler
    # Forwards with 3 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
    # Forwards with 2 seasons of > 40 GP: Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler
    # Forwards with 1 season            : Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(48, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(48, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(48, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('ATOI', 'forward', 4, 'PK')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'ATOI', 4, 'PK')
    instance_df_y3, _ = create_year_restricted_instance_df('ATOI', 'forward', 3, 'PK')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK')
    instance_df_y2, _ = create_year_restricted_instance_df('ATOI', 'forward', 2, 'PK')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'ATOI', 2, 'PK')
    instance_df_y1, _ = create_year_restricted_instance_df('ATOI', 'forward', 1, 'PK')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'ATOI', 1, 'PK')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=30, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=30, verbose=1)

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] != 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 40 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 20 games.
        # Once you reach 20 games, find the ATOI accross these seasons.
        # If they haven't played 20 games in their past 4 seasons, fill the rest of the 20 games with the -1st z-score of the stat.
        if y1_gp >= 20:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 PK ATOI'].mean() - instance_df_y1['Y4 PK ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 20-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0] + statistics.mean(statline[-4:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0] + statistics.mean(statline[-3:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0] + statistics.mean(statline[-2:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0] + statistics.mean(statline[-1:]), 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_defence_pk_atoi_projections(stat_df, projection_df, download_file, year=2024):
    # Defence with 4 seasons of > 40 GP: Parent model 7 (128-64-1), 5 epochs, standard scaler
    # Defence with 3 seasons of > 40 GP: Parent model 7 (128-64-1), 5 epochs, minmax scaler
    # Defence with 2 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
    # Defence with 1 season            : Parent model 9 (36-12-1), 5 epochs, minmax scaler

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(36, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('ATOI', 'defence', 4, 'PK')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'ATOI', 4, 'PK')
    instance_df_y3, _ = create_year_restricted_instance_df('ATOI', 'defence', 3, 'PK')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK')
    instance_df_y2, _ = create_year_restricted_instance_df('ATOI', 'defence', 2, 'PK')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'ATOI', 2, 'PK')
    instance_df_y1, _ = create_year_restricted_instance_df('ATOI', 'defence', 1, 'PK')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'ATOI', 1, 'PK')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=5, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=5, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=5, verbose=1)

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] == 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 40 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 20 games.
        # Once you reach 20 games, find the ATOI accross these seasons.
        # If they haven't played 20 games in their past 4 seasons, fill the rest of the 20 games with the -1st z-score of the stat.
        if y1_gp >= 20:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 PK ATOI'].mean() - instance_df_y1['Y4 PK ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 20-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0] + statistics.mean(statline[-4:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0] + statistics.mean(statline[-3:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0] + statistics.mean(statline[-2:]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0] + statistics.mean(statline[-1:]), 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_forward_ev_gper60_projections(stat_df, projection_df, download_file, year=2024):

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(126, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(42, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('Gper60', 'forward', 4, 'EV')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'forward')
    instance_df_y3, _ = create_year_restricted_instance_df('Gper60', 'forward', 3, 'EV')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'forward')
    instance_df_y2, _ = create_year_restricted_instance_df('Gper60', 'forward', 2, 'EV')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'forward')
    instance_df_y1, _ = create_year_restricted_instance_df('Gper60', 'forward', 1, 'EV')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'forward')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=30, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=5, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=1, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

    # permutation_feature_importance(yr4_model, X_4_scaled, y_4, ['Age', 'Height', 'Weight', 'Y1 G/60', 'Y2 G/60', 'Y3 G/60', 'Y4 G/60'])

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] != 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 40 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 50 games.
        # Once you reach 50 games, find the ATOI accross these seasons.
        # If they haven't played 50 games in their past 4 seasons, fill the rest of the 50 games with the -1st z-score of the stat.
        if y1_gp >= 50:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
        elif y1_gp + y2_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV G/60'].mean() - instance_df_y1['Y4 EV G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV ixG/60'].mean() - instance_df_y1['Y4 EV ixG/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        # print(player, [
        #     calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
        #     int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
        #     int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
        #     pseudo_prev_year_stat_1,
        #     pseudo_prev_year_stat_2
        #     ])

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'EV G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0], 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_defence_ev_gper60_projections(stat_df, projection_df, download_file, year=2024):

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(126, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(42, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('Gper60', 'defence', 4, 'EV')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'defence')
    instance_df_y3, _ = create_year_restricted_instance_df('Gper60', 'defence', 3, 'EV')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'defence')
    instance_df_y2, _ = create_year_restricted_instance_df('Gper60', 'defence', 2, 'EV')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'defence')
    instance_df_y1, _ = create_year_restricted_instance_df('Gper60', 'defence', 1, 'EV')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'defence')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=5, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=1, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=5, verbose=1)

    # permutation_feature_importance(yr4_model, X_4_scaled, y_4, ['Age', 'Height', 'Weight', 'Y1 G/60', 'Y2 G/60', 'Y3 G/60', 'Y4 G/60', 'Y1 ixG/60', 'Y2 ixG/60', 'Y3 ixG/60', 'Y4 ixG/60'])

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    gp_adjustment_factor = {
        2023: 1,
        2022: 1,
        2021: 82/56,
        2020: 82/69.5,
        2019: 1,
        2018: 1,
        2017: 1,
        2016: 1,
        2015: 1,
        2014: 1,
        2013: 82/48,
        2012: 1,
        2011: 1,
        2010: 1,
        2009: 1,
        2008: 1
    }

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} GP'] >= 1)]['Player']) and row['Position'] == 'D':
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 40 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 50 games.
        # Once you reach 50 games, find the ATOI accross these seasons.
        # If they haven't played 50 games in their past 4 seasons, fill the rest of the 50 games with the -1st z-score of the stat.
        if y1_gp >= 50:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
        elif y1_gp + y2_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV G/60'].mean() - instance_df_y1['Y4 EV G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV ixG/60'].mean() - instance_df_y1['Y4 EV ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'EV G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0], 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def make_forward_pp_gper60_projections(stat_df, projection_df, download_file, year=2024):

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(19,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(15,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('Gper60', 'forward', 4, 'PP')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'Gper60', 4, 'PP', 'forward')
    instance_df_y3, _ = create_year_restricted_instance_df('Gper60', 'forward', 3, 'PP')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'forward')
    instance_df_y2, _ = create_year_restricted_instance_df('Gper60', 'forward', 2, 'PP')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'Gper60', 2, 'PP', 'forward')
    instance_df_y1, _ = create_year_restricted_instance_df('Gper60', 'forward', 1, 'PP')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'Gper60', 1, 'PP', 'forward')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=5, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=5, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

    # permutation_feature_importance(yr4_model, X_4_scaled, y_4, ['Age', 'Height', 'Weight', 'Y1 PP G/60', 'Y2 PP G/60', 'Y3 PP G/60', 'Y4 PP G/60', 'Y1 PP ixG/60', 'Y2 PP ixG/60', 'Y3 PP ixG/60', 'Y4 PP ixG/60', 'Y1 EV G/60', 'Y2 EV G/60', 'Y3 EV G/60', 'Y4 EV G/60', 'Y1 EV ixG/60', 'Y2 EV ixG/60', 'Y3 EV ixG/60', 'Y4 EV ixG/60'])

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} PP ATOI'] > 0)]['Player']) and row['Position'] != 'D':
            if row[f'{year-4} PP ATOI']*row[f'{year-4} GP'] >= 50 and row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP G/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP G/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ixG/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ixG/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPG/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PP G/60'].mean() - instance_df_y1['Y4 PP G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 PP ixG/60'].mean() - instance_df_y1['Y4 PP ixG/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 EV G/60'].mean() - instance_df_y1['Y4 EV G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 EV ixG/60'].mean() - instance_df_y1['Y4 EV ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    # for index in range(len(yr4_stat_list)):
    #     print(yr4_group[index], yr4_stat_list[index])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'PP G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0], 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'partial_projections'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_gper60_projections(stat_df, projection_df, download_file, year=2024):

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(19,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(15,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('Gper60', 'defence', 4, 'PP')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'Gper60', 4, 'PP', 'defence')
    instance_df_y3, _ = create_year_restricted_instance_df('Gper60', 'defence', 3, 'PP')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'defence')
    instance_df_y2, _ = create_year_restricted_instance_df('Gper60', 'defence', 2, 'PP')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'Gper60', 2, 'PP', 'defence')
    instance_df_y1, _ = create_year_restricted_instance_df('Gper60', 'defence', 1, 'PP')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'Gper60', 1, 'PP', 'defence')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=5, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=5, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=5, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

    # permutation_feature_importance(yr4_model, X_4_scaled, y_4, ['Age', 'Height', 'Weight', 'Y1 PP G/60', 'Y2 PP G/60', 'Y3 PP G/60', 'Y4 PP G/60', 'Y1 PP ixG/60', 'Y2 PP ixG/60', 'Y3 PP ixG/60', 'Y4 PP ixG/60', 'Y1 EV G/60', 'Y2 EV G/60', 'Y3 EV G/60', 'Y4 EV G/60', 'Y1 EV ixG/60', 'Y2 EV ixG/60', 'Y3 EV ixG/60', 'Y4 EV ixG/60'])

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} PP ATOI'] > 0)]['Player']) and row['Position'] == 'D':
            if row[f'{year-4} PP ATOI']*row[f'{year-4} GP'] >= 50 and row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP G/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP G/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ixG/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ixG/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPG/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PP G/60'].mean() - instance_df_y1['Y4 PP G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 PP ixG/60'].mean() - instance_df_y1['Y4 PP ixG/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 EV G/60'].mean() - instance_df_y1['Y4 EV G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 EV ixG/60'].mean() - instance_df_y1['Y4 EV ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'PP G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0], 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'partial_projections'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_gper60_projections(stat_df, projection_df, download_file, year=2024):

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('Gper60', 'forward', 4, 'PK')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'Gper60', 4, 'PK', 'forward')
    instance_df_y3, _ = create_year_restricted_instance_df('Gper60', 'forward', 3, 'PK')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'forward')
    instance_df_y2, _ = create_year_restricted_instance_df('Gper60', 'forward', 2, 'PK')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'Gper60', 2, 'PK', 'forward')
    instance_df_y1, _ = create_year_restricted_instance_df('Gper60', 'forward', 1, 'PK')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'Gper60', 1, 'PK', 'forward')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

    # permutation_feature_importance(yr4_model, X_4_scaled, y_4, ['Age', 'Height', 'Weight', 'Y1 PP G/60', 'Y2 PP G/60', 'Y3 PP G/60', 'Y4 PP G/60', 'Y1 PP ixG/60', 'Y2 PP ixG/60', 'Y3 PP ixG/60', 'Y4 PP ixG/60', 'Y1 EV G/60', 'Y2 EV G/60', 'Y3 EV G/60', 'Y4 EV G/60', 'Y1 EV ixG/60', 'Y2 EV ixG/60', 'Y3 EV ixG/60', 'Y4 EV ixG/60'])

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} PK ATOI'] > 0)]['Player']) and row['Position'] != 'D':
            if row[f'{year-4} PK ATOI']*row[f'{year-4} GP'] >= 50 and row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ixG/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 75 PKTOI.
        # Once you reach 75 PKTOI, find the PPG/60 accross these seasons.
        # If they haven't played 75 PKTOI in their past 4 seasons, fill the rest of the 75 PKTOI with the -1st z-score of the stat.
        if y1_pktoi >= 75:
            pseudo_prev_year_stat = y1_stat
        elif y1_pktoi + y2_pktoi >= 75:
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi)/(y1_pktoi + y2_pktoi)
        elif y1_pktoi + y2_pktoi + y3_pktoi >= 75:
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi)/(y1_pktoi + y2_pktoi + y3_pktoi)
        elif y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi >= 75:
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PK ixG/60'].mean() - instance_df_y1['Y4 PK ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'PK G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0], 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'partial_projections'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_gper60_projections(stat_df, projection_df, download_file, year=2024):

    yr4_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr3_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr2_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr1_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    yr4_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr3_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr2_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])
    yr1_model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

    instance_df_y4, _ = create_year_restricted_instance_df('Gper60', 'defence', 4, 'PK')
    X_4, y_4 = extract_instance_data(instance_df_y4, 'Gper60', 4, 'PK', 'defence')
    instance_df_y3, _ = create_year_restricted_instance_df('Gper60', 'defence', 3, 'PK')
    X_3, y_3 = extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'defence')
    instance_df_y2, _ = create_year_restricted_instance_df('Gper60', 'defence', 2, 'PK')
    X_2, y_2 = extract_instance_data(instance_df_y2, 'Gper60', 2, 'PK', 'defence')
    instance_df_y1, _ = create_year_restricted_instance_df('Gper60', 'defence', 1, 'PK')
    X_1, y_1 = extract_instance_data(instance_df_y1, 'Gper60', 1, 'PK', 'defence')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

    # permutation_feature_importance(yr4_model, X_4_scaled, y_4, ['Age', 'Height', 'Weight', 'Y1 PP G/60', 'Y2 PP G/60', 'Y3 PP G/60', 'Y4 PP G/60', 'Y1 PP ixG/60', 'Y2 PP ixG/60', 'Y3 PP ixG/60', 'Y4 PP ixG/60', 'Y1 EV G/60', 'Y2 EV G/60', 'Y3 EV G/60', 'Y4 EV G/60', 'Y1 EV ixG/60', 'Y2 EV ixG/60', 'Y3 EV ixG/60', 'Y4 EV ixG/60'])

    yr4_group, yr3_group, yr2_group, yr1_group = [], [], [], []

    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{year-1} PK ATOI'] > 0)]['Player']) and row['Position'] == 'D':
            if row[f'{year-4} PK ATOI']*row[f'{year-4} GP'] >= 50 and row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ixG/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 75 PKTOI.
        # Once you reach 75 PKTOI, find the PPG/60 accross these seasons.
        # If they haven't played 75 PKTOI in their past 4 seasons, fill the rest of the 75 PKTOI with the -1st z-score of the stat.
        if y1_pktoi >= 75:
            pseudo_prev_year_stat = y1_stat
        elif y1_pktoi + y2_pktoi >= 75:
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi)/(y1_pktoi + y2_pktoi)
        elif y1_pktoi + y2_pktoi + y3_pktoi >= 75:
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi)/(y1_pktoi + y2_pktoi + y3_pktoi)
        elif y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi >= 75:
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PK ixG/60'].mean() - instance_df_y1['Y4 PK ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    proj_y_4 = yr4_model.predict(yr4_stat_list_scaled, verbose=1)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    proj_y_3 = yr3_model.predict(yr3_stat_list_scaled, verbose=1)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    proj_y_2 = yr2_model.predict(yr2_stat_list_scaled, verbose=1)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    proj_y_1 = yr1_model.predict(yr1_stat_list_scaled, verbose=1)

    column_name = 'PK G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(proj_y_4[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(proj_y_3[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(proj_y_2[index][0], 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index] # watch year yrN
        projection = max(proj_y_1[index][0], 0) # watch year yrN

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    return projection_df

def goal_era_adjustment(stat_df, projection_df, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_goal_df = pd.DataFrame()

    for year in range(2007, 2023):
        col = round(((stat_df[f'{year+1} EV G/60']/60*stat_df[f'{year+1} EV ATOI'] + stat_df[f'{year+1} PP G/60']/60*stat_df[f'{year+1} PP ATOI'] + stat_df[f'{year+1} PK G/60']/60*stat_df[f'{year+1} PK ATOI']) * stat_df[f'{year+1} GP'])).astype(int)
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_goal_df = hist_goal_df.reset_index(drop=True)
        hist_goal_df[year+1] = col
    hist_goal_df.index = hist_goal_df.index + 1

    try:
        hist_goal_df[2021] = round(82/56*hist_goal_df[2021]).astype(int)
    except KeyError:
        pass
    try:
        hist_goal_df[2020] = round(82/70*hist_goal_df[2020]).astype(int)
    except KeyError:
        pass
    try:
        hist_goal_df[2013] = round(82/48*hist_goal_df[2013]).astype(int)
    except KeyError:
        pass

    hist_goal_df['Historical Average'] = hist_goal_df.mean(axis=1)
    hist_goal_df['Projected Average'] = hist_goal_df.iloc[:, -5:-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_goal_df['Adjustment'] = hist_goal_df['Projected Average'] - hist_goal_df['Historical Average']
    hist_goal_df['Smoothed Adjustment'] = savgol_filter(hist_goal_df['Adjustment'], 25, 2)
    # print(hist_goal_df.head(750).to_string())

    projection_df['GOALS'] = (projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP'].astype(int)
    projection_df = projection_df.sort_values('GOALS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    projection_df['Era Adjustment Factor'] = hist_goal_df['Smoothed Adjustment']/((projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
    projection_df['EV G/60'] *= projection_df['Era Adjustment Factor']
    projection_df['PP G/60'] *= projection_df['Era Adjustment Factor']
    projection_df['PK G/60'] *= projection_df['Era Adjustment Factor']
    # print(projection_df.to_string())
    projection_df = projection_df.drop(columns=['GOALS', 'Era Adjustment Factor'])

    # Download file
    if download_file == True:
        filename = f'partial_projections'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def main():
    stat_df = scrape_player_statistics(True)
    projection_df = make_projection_df(stat_df, 2015)
    # projection_df = make_forward_gp_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_defence_gp_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_forward_ev_atoi_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_defence_ev_atoi_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_forward_pp_atoi_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_defence_pp_atoi_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_forward_pk_atoi_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_defence_pk_atoi_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_forward_ev_gper60_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_defence_ev_gper60_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_forward_pp_gper60_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_defence_pp_gper60_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_forward_pk_gper60_projections(stat_df, projection_df, False, 2015)
    # projection_df = make_defence_pk_gper60_projections(stat_df, projection_df, False, 2015)

    # projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/partial_projections.csv")
    # projection_df = projection_df.drop(projection_df.columns[0], axis=1)
    projection_df = pd.DataFrame(columns=['Player'])

    # Do this for all other projections neural nets

    # projection_df = goal_era_adjustment(stat_df, projection_df, False).fillna(0)
    # projection_df['GOALS'] = round((projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']).astype(int)

    # projection_df = projection_df.sort_values('GOALS', ascending=False)
    # projection_df = projection_df.sort_values('GP', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    # print(projection_df)
    print(projection_df.to_string())

main()
