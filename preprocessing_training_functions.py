import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import date
import numpy as np
import statistics

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
            elif dependent_variable == 'forward_EV_A1per60' or dependent_variable == 'forward_EV_A2per60':
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
                        fetch_data(row, year, 1, 'ev', 'A1/60'),
                        fetch_data(row, year, 2, 'ev', 'A1/60'),
                        fetch_data(row, year, 3, 'ev', 'A1/60'),
                        fetch_data(row, year, 4, 'ev', 'A1/60'),
                        fetch_data(row, year, 5, 'ev', 'A1/60'),
                        fetch_data(row, year, 1, 'ev', 'A2/60'),
                        fetch_data(row, year, 2, 'ev', 'A2/60'),
                        fetch_data(row, year, 3, 'ev', 'A2/60'),
                        fetch_data(row, year, 4, 'ev', 'A2/60'),
                        fetch_data(row, year, 5, 'ev', 'A2/60'),
                        fetch_data(row, year, 1, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 2, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 3, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 4, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 5, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 1, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 2, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 3, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 4, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 5, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 1, 'ev', 'oixGF/60'),
                        fetch_data(row, year, 2, 'ev', 'oixGF/60'),
                        fetch_data(row, year, 3, 'ev', 'oixGF/60'),
                        fetch_data(row, year, 4, 'ev', 'oixGF/60'),
                        fetch_data(row, year, 5, 'ev', 'oixGF/60')
                    ]
            elif dependent_variable == 'defence_EV_A1per60' or dependent_variable == 'defence_EV_A2per60':
                # filter out:
                    # defence
                    # players with < 50 GP in Y5
                    # players with < 50 GP in Y4

                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
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
                        fetch_data(row, year, 1, 'ev', 'A1/60'),
                        fetch_data(row, year, 2, 'ev', 'A1/60'),
                        fetch_data(row, year, 3, 'ev', 'A1/60'),
                        fetch_data(row, year, 4, 'ev', 'A1/60'),
                        fetch_data(row, year, 5, 'ev', 'A1/60'),
                        fetch_data(row, year, 1, 'ev', 'A2/60'),
                        fetch_data(row, year, 2, 'ev', 'A2/60'),
                        fetch_data(row, year, 3, 'ev', 'A2/60'),
                        fetch_data(row, year, 4, 'ev', 'A2/60'),
                        fetch_data(row, year, 5, 'ev', 'A2/60'),
                        fetch_data(row, year, 1, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 2, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 3, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 4, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 5, 'ev', 'Rebounds Created/60'),
                        fetch_data(row, year, 1, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 2, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 3, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 4, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 5, 'ev', 'Rush Attempts/60'),
                        fetch_data(row, year, 1, 'ev', 'oixGF/60'),
                        fetch_data(row, year, 2, 'ev', 'oixGF/60'),
                        fetch_data(row, year, 3, 'ev', 'oixGF/60'),
                        fetch_data(row, year, 4, 'ev', 'oixGF/60'),
                        fetch_data(row, year, 5, 'ev', 'oixGF/60')
                    ]

            elif dependent_variable == 'forward_PP_A1per60' or dependent_variable == 'forward_PP_A2per60':
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

                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, 'pp', 'ATOI')*fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, 'pp', 'ATOI')*fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
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
                        fetch_data(row, year, 1, 'ev', 'A1/60'),
                        fetch_data(row, year, 2, 'ev', 'A1/60'),
                        fetch_data(row, year, 3, 'ev', 'A1/60'),
                        fetch_data(row, year, 4, 'ev', 'A1/60'),
                        fetch_data(row, year, 5, 'ev', 'A1/60'),
                        fetch_data(row, year, 1, 'ev', 'A2/60'),
                        fetch_data(row, year, 2, 'ev', 'A2/60'),
                        fetch_data(row, year, 3, 'ev', 'A2/60'),
                        fetch_data(row, year, 4, 'ev', 'A2/60'),
                        fetch_data(row, year, 5, 'ev', 'A2/60'),
                        fetch_data(row, year, 1, 'pp', 'A1/60'),
                        fetch_data(row, year, 2, 'pp', 'A1/60'),
                        fetch_data(row, year, 3, 'pp', 'A1/60'),
                        fetch_data(row, year, 4, 'pp', 'A1/60'),
                        fetch_data(row, year, 5, 'pp', 'A1/60'),
                        fetch_data(row, year, 1, 'pp', 'A2/60'),
                        fetch_data(row, year, 2, 'pp', 'A2/60'),
                        fetch_data(row, year, 3, 'pp', 'A2/60'),
                        fetch_data(row, year, 4, 'pp', 'A2/60'),
                        fetch_data(row, year, 5, 'pp', 'A2/60'),
                        fetch_data(row, year, 1, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 2, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 3, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 4, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 5, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 1, 'pp', 'oixGF/60'),
                        fetch_data(row, year, 2, 'pp', 'oixGF/60'),
                        fetch_data(row, year, 3, 'pp', 'oixGF/60'),
                        fetch_data(row, year, 4, 'pp', 'oixGF/60'),
                        fetch_data(row, year, 5, 'pp', 'oixGF/60')
                    ]
            elif dependent_variable == 'defence_PP_A1per60' or dependent_variable == 'defence_PP_A2per60':
                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, 'pp', 'ATOI')*fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, 'pp', 'ATOI')*fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
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
                        fetch_data(row, year, 1, 'ev', 'A1/60'),
                        fetch_data(row, year, 2, 'ev', 'A1/60'),
                        fetch_data(row, year, 3, 'ev', 'A1/60'),
                        fetch_data(row, year, 4, 'ev', 'A1/60'),
                        fetch_data(row, year, 5, 'ev', 'A1/60'),
                        fetch_data(row, year, 1, 'ev', 'A2/60'),
                        fetch_data(row, year, 2, 'ev', 'A2/60'),
                        fetch_data(row, year, 3, 'ev', 'A2/60'),
                        fetch_data(row, year, 4, 'ev', 'A2/60'),
                        fetch_data(row, year, 5, 'ev', 'A2/60'),
                        fetch_data(row, year, 1, 'pp', 'A1/60'),
                        fetch_data(row, year, 2, 'pp', 'A1/60'),
                        fetch_data(row, year, 3, 'pp', 'A1/60'),
                        fetch_data(row, year, 4, 'pp', 'A1/60'),
                        fetch_data(row, year, 5, 'pp', 'A1/60'),
                        fetch_data(row, year, 1, 'pp', 'A2/60'),
                        fetch_data(row, year, 2, 'pp', 'A2/60'),
                        fetch_data(row, year, 3, 'pp', 'A2/60'),
                        fetch_data(row, year, 4, 'pp', 'A2/60'),
                        fetch_data(row, year, 5, 'pp', 'A2/60'),
                        fetch_data(row, year, 1, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 2, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 3, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 4, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 5, 'pp', 'Rebounds Created/60'),
                        fetch_data(row, year, 1, 'pp', 'oixGF/60'),
                        fetch_data(row, year, 2, 'pp', 'oixGF/60'),
                        fetch_data(row, year, 3, 'pp', 'oixGF/60'),
                        fetch_data(row, year, 4, 'pp', 'oixGF/60'),
                        fetch_data(row, year, 5, 'pp', 'oixGF/60')
                    ]
            elif dependent_variable == 'forward_PK_A1per60' or dependent_variable == 'forward_PK_A2per60':
                if row['Position'] == 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, 'pk', 'ATOI')*fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, 'pk', 'ATOI')*fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
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
                        fetch_data(row, year, 1, 'pk', 'A1/60'),
                        fetch_data(row, year, 2, 'pk', 'A1/60'),
                        fetch_data(row, year, 3, 'pk', 'A1/60'),
                        fetch_data(row, year, 4, 'pk', 'A1/60'),
                        fetch_data(row, year, 5, 'pk', 'A1/60'),
                        fetch_data(row, year, 1, 'pk', 'A2/60'),
                        fetch_data(row, year, 2, 'pk', 'A2/60'),
                        fetch_data(row, year, 3, 'pk', 'A2/60'),
                        fetch_data(row, year, 4, 'pk', 'A2/60'),
                        fetch_data(row, year, 5, 'pk', 'A2/60'),
                    ]
            elif dependent_variable == 'defence_PK_A1per60' or dependent_variable == 'defence_PK_A2per60':
                if row['Position'] != 'D':
                    pass
                elif np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')):
                    pass
                elif fetch_data(row, year, 5, 'pk', 'ATOI')*fetch_data(row, year, 5, None, 'GP') < 50 or fetch_data(row, year, 4, 'pk', 'ATOI')*fetch_data(row, year, 4, None, 'GP') < 50:
                    pass
                else:
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
                        fetch_data(row, year, 1, 'pk', 'A1/60'),
                        fetch_data(row, year, 2, 'pk', 'A1/60'),
                        fetch_data(row, year, 3, 'pk', 'A1/60'),
                        fetch_data(row, year, 4, 'pk', 'A1/60'),
                        fetch_data(row, year, 5, 'pk', 'A1/60'),
                        fetch_data(row, year, 1, 'pk', 'A2/60'),
                        fetch_data(row, year, 2, 'pk', 'A2/60'),
                        fetch_data(row, year, 3, 'pk', 'A2/60'),
                        fetch_data(row, year, 4, 'pk', 'A2/60'),
                        fetch_data(row, year, 5, 'pk', 'A2/60'),
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

    elif proj_stat == 'A1per60' or proj_stat == 'A2per60':
        if situation == 'EV':
            instance_df = create_instance_df(f'{position}_{situation}_A1per60', [
                'Player', 'Year', 'Position', 'Age', 'Height', 'Weight',
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y1 {situation} A1/60', f'Y2 {situation} A1/60', f'Y3 {situation} A1/60', f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y1 {situation} A2/60', f'Y2 {situation} A2/60', f'Y3 {situation} A2/60', f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y1 {situation} Rebounds Created/60', f'Y2 {situation} Rebounds Created/60', f'Y3 {situation} Rebounds Created/60', f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y1 {situation} Rush Attempts/60', f'Y2 {situation} Rush Attempts/60', f'Y3 {situation} Rush Attempts/60', f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y1 {situation} oixGF/60', f'Y2 {situation} oixGF/60', f'Y3 {situation} oixGF/60', f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ], scrape_player_statistics(True), True)      
            if prev_years == 4:
                instance_df = instance_df.loc[(instance_df['Y1 GP'] >= 50) & (instance_df['Y2 GP'] >= 50) & (instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                input_shape = (23,)
            elif prev_years == 3:
                instance_df = instance_df.loc[(instance_df['Y2 GP'] >= 50) & (instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                input_shape = (18,)
            elif prev_years == 2:
                instance_df = instance_df.loc[(instance_df['Y3 GP'] >= 50) & (instance_df['Y4 GP'] >= 50)]
                input_shape = (13,)
            elif prev_years == 1:
                instance_df = instance_df.loc[(instance_df['Y4 GP'] >= 50)]
                input_shape = (8,)
            else:
                print('Invalid prev_years parameter.')
        if situation == 'PP':
            instance_df = create_instance_df(f'{position}_{situation}_A1per60', [
                'Player', 'Year', 'Position', 'Age', 'Height', 'Weight',
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 PP ATOI', f'Y2 PP ATOI', f'Y3 PP ATOI', f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y1 EV A1/60', f'Y2 EV A1/60', f'Y3 EV A1/60', f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y1 EV A2/60', f'Y2 EV A2/60', f'Y3 EV A2/60', f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y1 PP A1/60', f'Y2 PP A1/60', f'Y3 PP A1/60', f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y1 PP A2/60', f'Y2 PP A2/60', f'Y3 PP A2/60', f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y1 PP Rebounds Created/60', f'Y2 PP Rebounds Created/60', f'Y3 PP Rebounds Created/60', f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y1 PP oixGF/60', f'Y2 PP oixGF/60', f'Y3 PP oixGF/60', f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ], scrape_player_statistics(True), True)      
            if prev_years == 4:
                instance_df = instance_df.loc[(instance_df['Y1 PP ATOI']*instance_df['Y1 GP'] >= 50) & (instance_df['Y2 PP ATOI']*instance_df['Y2 GP'] >= 50) & (instance_df['Y3 PP ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PP ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (27,)
            elif prev_years == 3:
                instance_df = instance_df.loc[(instance_df['Y2 PP ATOI']*instance_df['Y2 GP'] >= 50) & (instance_df['Y3 PP ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PP ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (21,)
            elif prev_years == 2:
                instance_df = instance_df.loc[(instance_df['Y3 PP ATOI']*instance_df['Y3 GP'] >= 50) & (instance_df['Y4 PP ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (15,)
            elif prev_years == 1:
                instance_df = instance_df.loc[(instance_df['Y4 PP ATOI']*instance_df['Y4 GP'] >= 50)]
                input_shape = (9,)
            else:
                print('Invalid prev_years parameter.') 
        if situation == 'PK':
            instance_df = create_instance_df(f'{position}_{situation}_A1per60', [
                'Player', 'Year', 'Position', 'Age', 'Height', 'Weight',
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 PK ATOI', f'Y2 PK ATOI', f'Y3 PK ATOI', f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y1 PK A1/60', f'Y2 PK A1/60', f'Y3 PK A1/60', f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y1 PK A2/60', f'Y2 PK A2/60', f'Y3 PK A2/60', f'Y4 PK A2/60', f'Y5 PK A2/60'
                ], scrape_player_statistics(True), True)      
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

    elif proj_stat == 'A1per60' or proj_stat == 'A2per60':
        if situation == 'EV':
            if prev_years == 4:
                instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y1 {situation} A1/60', f'Y2 {situation} A1/60', f'Y3 {situation} A1/60', f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y1 {situation} A2/60', f'Y2 {situation} A2/60', f'Y3 {situation} A2/60', f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y1 {situation} Rebounds Created/60', f'Y2 {situation} Rebounds Created/60', f'Y3 {situation} Rebounds Created/60', f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y1 {situation} Rush Attempts/60', f'Y2 {situation} Rush Attempts/60', f'Y3 {situation} Rush Attempts/60', f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y1 {situation} oixGF/60', f'Y2 {situation} oixGF/60', f'Y3 {situation} oixGF/60', f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ]] = instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 {situation} ATOI', f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y1 {situation} A1/60', f'Y2 {situation} A1/60', f'Y3 {situation} A1/60', f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y1 {situation} A2/60', f'Y2 {situation} A2/60', f'Y3 {situation} A2/60', f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y1 {situation} Rebounds Created/60', f'Y2 {situation} Rebounds Created/60', f'Y3 {situation} Rebounds Created/60', f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y1 {situation} Rush Attempts/60', f'Y2 {situation} Rush Attempts/60', f'Y3 {situation} Rush Attempts/60', f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y1 {situation} oixGF/60', f'Y2 {situation} oixGF/60', f'Y3 {situation} oixGF/60', f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y1 {situation} A1/60'], row[f'Y2 {situation} A1/60'], row[f'Y3 {situation} A1/60'], row[f'Y4 {situation} A1/60'],
                            row[f'Y1 {situation} A2/60'], row[f'Y2 {situation} A2/60'], row[f'Y3 {situation} A2/60'], row[f'Y4 {situation} A2/60'],
                            row[f'Y1 {situation} Rebounds Created/60'], row[f'Y2 {situation} Rebounds Created/60'], row[f'Y3 {situation} Rebounds Created/60'], row[f'Y4 {situation} Rebounds Created/60'],
                            row[f'Y1 {situation} Rush Attempts/60'], row[f'Y2 {situation} Rush Attempts/60'], row[f'Y3 {situation} Rush Attempts/60'], row[f'Y4 {situation} Rush Attempts/60'],
                            row[f'Y1 {situation} oixGF/60'], row[f'Y2 {situation} oixGF/60'], row[f'Y3 {situation} oixGF/60'], row[f'Y4 {situation} oixGF/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 3:
                instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y2 {situation} A1/60', f'Y3 {situation} A1/60', f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y2 {situation} A2/60', f'Y3 {situation} A2/60', f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y2 {situation} Rebounds Created/60', f'Y3 {situation} Rebounds Created/60', f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y2 {situation} Rush Attempts/60', f'Y3 {situation} Rush Attempts/60', f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y2 {situation} oixGF/60', f'Y3 {situation} oixGF/60', f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ]] = instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 {situation} ATOI', f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y2 {situation} A1/60', f'Y3 {situation} A1/60', f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y2 {situation} A2/60', f'Y3 {situation} A2/60', f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y2 {situation} Rebounds Created/60', f'Y3 {situation} Rebounds Created/60', f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y2 {situation} Rush Attempts/60', f'Y3 {situation} Rush Attempts/60', f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y2 {situation} oixGF/60', f'Y3 {situation} oixGF/60', f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y2 {situation} A1/60'], row[f'Y3 {situation} A1/60'], row[f'Y4 {situation} A1/60'],
                            row[f'Y2 {situation} A2/60'], row[f'Y3 {situation} A2/60'], row[f'Y4 {situation} A2/60'],
                            row[f'Y2 {situation} Rebounds Created/60'], row[f'Y3 {situation} Rebounds Created/60'], row[f'Y4 {situation} Rebounds Created/60'],
                            row[f'Y2 {situation} Rush Attempts/60'], row[f'Y3 {situation} Rush Attempts/60'], row[f'Y4 {situation} Rush Attempts/60'],
                            row[f'Y2 {situation} oixGF/60'], row[f'Y3 {situation} oixGF/60'], row[f'Y4 {situation} oixGF/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 2:
                instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y3 {situation} A1/60', f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y3 {situation} A2/60', f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y3 {situation} Rebounds Created/60', f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y3 {situation} Rush Attempts/60', f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y3 {situation} oixGF/60', f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ]] = instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 {situation} ATOI', f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y3 {situation} A1/60', f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y3 {situation} A2/60', f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y3 {situation} Rebounds Created/60', f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y3 {situation} Rush Attempts/60', f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y3 {situation} oixGF/60', f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y3 {situation} A1/60'], row[f'Y4 {situation} A1/60'],
                            row[f'Y3 {situation} A2/60'], row[f'Y4 {situation} A2/60'],
                            row[f'Y3 {situation} Rebounds Created/60'], row[f'Y4 {situation} Rebounds Created/60'],
                            row[f'Y3 {situation} Rush Attempts/60'], row[f'Y4 {situation} Rush Attempts/60'],
                            row[f'Y3 {situation} oixGF/60'], row[f'Y4 {situation} oixGF/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 1:
                instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ]] = instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 {situation} ATOI', f'Y5 {situation} ATOI', 
                f'Y4 {situation} A1/60', f'Y5 {situation} A1/60',
                f'Y4 {situation} A2/60', f'Y5 {situation} A2/60',
                f'Y4 {situation} Rebounds Created/60', f'Y5 {situation} Rebounds Created/60',
                f'Y4 {situation} Rush Attempts/60', f'Y5 {situation} Rush Attempts/60',
                f'Y4 {situation} oixGF/60', f'Y5 {situation} oixGF/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y4 {situation} A1/60'],
                            row[f'Y4 {situation} A2/60'],
                            row[f'Y4 {situation} Rebounds Created/60'],
                            row[f'Y4 {situation} Rush Attempts/60'],
                            row[f'Y4 {situation} oixGF/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
        elif situation == 'PP':
            if prev_years == 4:
                instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 PP ATOI', f'Y2 PP ATOI', f'Y3 PP ATOI', f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y1 EV A1/60', f'Y2 EV A1/60', f'Y3 EV A1/60', f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y1 EV A2/60', f'Y2 EV A2/60', f'Y3 EV A2/60', f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y1 PP A1/60', f'Y2 PP A1/60', f'Y3 PP A1/60', f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y1 PP A2/60', f'Y2 PP A2/60', f'Y3 PP A2/60', f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y1 PP Rebounds Created/60', f'Y2 PP Rebounds Created/60', f'Y3 PP Rebounds Created/60', f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y1 PP oixGF/60', f'Y2 PP oixGF/60', f'Y3 PP oixGF/60', f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ]] = instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 PP ATOI', f'Y2 PP ATOI', f'Y3 PP ATOI', f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y1 EV A1/60', f'Y2 EV A1/60', f'Y3 EV A1/60', f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y1 EV A2/60', f'Y2 EV A2/60', f'Y3 EV A2/60', f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y1 PP A1/60', f'Y2 PP A1/60', f'Y3 PP A1/60', f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y1 PP A2/60', f'Y2 PP A2/60', f'Y3 PP A2/60', f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y1 PP Rebounds Created/60', f'Y2 PP Rebounds Created/60', f'Y3 PP Rebounds Created/60', f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y1 PP oixGF/60', f'Y2 PP oixGF/60', f'Y3 PP oixGF/60', f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y1 EV A1/60'], row[f'Y2 EV A1/60'], row[f'Y3 EV A1/60'], row[f'Y4 EV A1/60'],
                            row[f'Y1 EV A2/60'], row[f'Y2 EV A2/60'], row[f'Y3 EV A2/60'], row[f'Y4 EV A2/60'],
                            row[f'Y1 PP A1/60'], row[f'Y2 PP A1/60'], row[f'Y3 PP A1/60'], row[f'Y4 PP A1/60'],
                            row[f'Y1 PP A2/60'], row[f'Y2 PP A2/60'], row[f'Y3 PP A2/60'], row[f'Y4 PP A2/60'],
                            row[f'Y1 PP Rebounds Created/60'], row[f'Y2 PP Rebounds Created/60'], row[f'Y3 PP Rebounds Created/60'], row[f'Y4 PP Rebounds Created/60'],
                            row[f'Y1 PP oixGF/60'], row[f'Y2 PP oixGF/60'], row[f'Y3 PP oixGF/60'], row[f'Y4 PP oixGF/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 3:
                instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 PP ATOI', f'Y3 PP ATOI', f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y2 EV A1/60', f'Y3 EV A1/60', f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y2 EV A2/60', f'Y3 EV A2/60', f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y2 PP A1/60', f'Y3 PP A1/60', f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y2 PP A2/60', f'Y3 PP A2/60', f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y2 PP Rebounds Created/60', f'Y3 PP Rebounds Created/60', f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y2 PP oixGF/60', f'Y3 PP oixGF/60', f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ]] = instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 PP ATOI', f'Y3 PP ATOI', f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y2 EV A1/60', f'Y3 EV A1/60', f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y2 EV A2/60', f'Y3 EV A2/60', f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y2 PP A1/60', f'Y3 PP A1/60', f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y2 PP A2/60', f'Y3 PP A2/60', f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y2 PP Rebounds Created/60', f'Y3 PP Rebounds Created/60', f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y2 PP oixGF/60', f'Y3 PP oixGF/60', f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y2 EV A1/60'], row[f'Y3 EV A1/60'], row[f'Y4 EV A1/60'],
                            row[f'Y2 EV A2/60'], row[f'Y3 EV A2/60'], row[f'Y4 EV A2/60'],
                            row[f'Y2 PP A1/60'], row[f'Y3 PP A1/60'], row[f'Y4 PP A1/60'],
                            row[f'Y2 PP A2/60'], row[f'Y3 PP A2/60'], row[f'Y4 PP A2/60'],
                            row[f'Y2 PP Rebounds Created/60'], row[f'Y3 PP Rebounds Created/60'], row[f'Y4 PP Rebounds Created/60'],
                            row[f'Y2 PP oixGF/60'], row[f'Y3 PP oixGF/60'], row[f'Y4 PP oixGF/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 2:
                instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 PP ATOI', f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y3 EV A1/60', f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y3 EV A2/60', f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y3 PP A1/60', f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y3 PP A2/60', f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y3 PP Rebounds Created/60', f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y3 PP oixGF/60', f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ]] = instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 PP ATOI', f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y3 EV A1/60', f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y3 EV A2/60', f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y3 PP A1/60', f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y3 PP A2/60', f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y3 PP Rebounds Created/60', f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y3 PP oixGF/60', f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y3 EV A1/60'], row[f'Y4 EV A1/60'],
                            row[f'Y3 EV A2/60'], row[f'Y4 EV A2/60'],
                            row[f'Y3 PP A1/60'], row[f'Y4 PP A1/60'],
                            row[f'Y3 PP A2/60'], row[f'Y4 PP A2/60'],
                            row[f'Y3 PP Rebounds Created/60'], row[f'Y4 PP Rebounds Created/60'],
                            row[f'Y3 PP oixGF/60'], row[f'Y4 PP oixGF/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 1:
                instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ]] = instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 PP ATOI', f'Y5 PP ATOI', 
                f'Y4 EV A1/60', f'Y5 EV A1/60',
                f'Y4 EV A2/60', f'Y5 EV A2/60',
                f'Y4 PP A1/60', f'Y5 PP A1/60',
                f'Y4 PP A2/60', f'Y5 PP A2/60',
                f'Y4 PP Rebounds Created/60', f'Y5 PP Rebounds Created/60',
                f'Y4 PP oixGF/60', f'Y5 PP oixGF/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y4 EV A1/60'],
                            row[f'Y4 EV A2/60'],
                            row[f'Y4 PP A1/60'],
                            row[f'Y4 PP A2/60'],
                            row[f'Y4 PP Rebounds Created/60'],
                            row[f'Y4 PP oixGF/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
        elif situation == 'PK':
            if prev_years == 4:
                instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 PK ATOI', f'Y2 PK ATOI', f'Y3 PK ATOI', f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y1 PK A1/60', f'Y2 PK A1/60', f'Y3 PK A1/60', f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y1 PK A2/60', f'Y2 PK A2/60', f'Y3 PK A2/60', f'Y4 PK A2/60', f'Y5 PK A2/60'
                ]] = instance_df[[
                'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y1 PK ATOI', f'Y2 PK ATOI', f'Y3 PK ATOI', f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y1 PK A1/60', f'Y2 PK A1/60', f'Y3 PK A1/60', f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y1 PK A2/60', f'Y2 PK A2/60', f'Y3 PK A2/60', f'Y4 PK A2/60', f'Y5 PK A2/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y1 PK A1/60'], row[f'Y2 PK A1/60'], row[f'Y3 PK A1/60'], row[f'Y4 PK A1/60'],
                            # row[f'Y1 PK A2/60'], row[f'Y2 PK A2/60'], row[f'Y3 PK A2/60'], row[f'Y4 PK A2/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 3:
                instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 PK ATOI', f'Y3 PK ATOI', f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y2 PK A1/60', f'Y3 PK A1/60', f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y2 PK A2/60', f'Y3 PK A2/60', f'Y4 PK A2/60', f'Y5 PK A2/60'
                ]] = instance_df[[
                'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y2 PK ATOI', f'Y3 PK ATOI', f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y2 PK A1/60', f'Y3 PK A1/60', f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y2 PK A2/60', f'Y3 PK A2/60', f'Y4 PK A2/60', f'Y5 PK A2/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y2 PK A1/60'], row[f'Y3 PK A1/60'], row[f'Y4 PK A1/60'],
                            # row[f'Y2 PK A2/60'], row[f'Y3 PK A2/60'], row[f'Y4 PK A2/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 2:
                instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 PK ATOI', f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y3 PK A1/60', f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y3 PK A2/60', f'Y4 PK A2/60', f'Y5 PK A2/60'
                ]] = instance_df[[
                'Y3 GP', 'Y4 GP', 'Y5 GP', 
                f'Y3 PK ATOI', f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y3 PK A1/60', f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y3 PK A2/60', f'Y4 PK A2/60', f'Y5 PK A2/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y3 PK A1/60'], row[f'Y4 PK A1/60'],
                            # row[f'Y3 PK A2/60'], row[f'Y4 PK A2/60']
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target
            elif prev_years == 1:
                instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y4 PK A2/60', f'Y5 PK A2/60'
                ]] = instance_df[[
                'Y4 GP', 'Y5 GP', 
                f'Y4 PK ATOI', f'Y5 PK ATOI', 
                f'Y4 PK A1/60', f'Y5 PK A1/60',
                f'Y4 PK A2/60', f'Y5 PK A2/60'
                ]].fillna(0)
                for index, row in instance_df.iterrows():
                    X.append([row['Age'], row['Height'], row['Weight'],
                            row[f'Y4 PK A1/60'],
                            # row[f'Y4 PK A2/60'],
                            ]) # features
                    if proj_stat == 'A1per60':
                        y.append(row[f'Y5 {situation} A1/60']) # target 
                    elif proj_stat == 'A2per60':
                        y.append(row[f'Y5 {situation} A2/60']) # target

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
