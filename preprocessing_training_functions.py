import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import date
import numpy as np

def scrape_bios(download_file=False):
    start_year = 2007
    end_year = 2023

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
        player_bio_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/player_bios.csv")
    elif existing_csv == False:
        player_bio_df = scrape_bios(False)
        player_bio_df = player_bio_df.drop(player_bio_df.columns[0], axis=1)

    if existing_csv == True:
        stat_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/historical_player_statistics.csv")
    elif existing_csv == False:
        stat_df = prune_bios(player_bio_df)
        stat_df = scrape_statistics(stat_df, 'ev', 'std', True)
        stat_df = scrape_statistics(stat_df, 'pp', 'std', True)
        stat_df = scrape_statistics(stat_df, 'pk', 'std', True)
        stat_df = scrape_statistics(stat_df, 'ev', 'oi', True)
        stat_df = scrape_statistics(stat_df, 'pp', 'oi', True)
        stat_df = scrape_statistics(stat_df, 'pk', 'oi', True)

    return stat_df

def create_instance_df(dependent_variable, model_features, stat_df, download_file=False):
    # str, list, df, bool

    start_year = 2007
    end_year = 2023

    instance_df = pd.DataFrame(columns=['Player', 'Year', 'Position', 'Age', 'Height', 'Weight', 'Draft Position', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 'Y1 ATOI', 'Y2 ATOI', 'Y3 ATOI', 'Y4 ATOI', 'Y5 ATOI', 'Y1 G/82', 'Y2 G/82', 'Y3 G/82', 'Y4 G/82', 'Y5 G/82'])

    for index, row in stat_df.iterrows():
        for year in range(start_year+4, end_year):
            # filter out:
                # defence
                # players with 0 GP in Y5
                # players with less than 50 GP in either Y4, Y3, or Y2
                # instances where Y5 was 2011 or earlier

            if np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')) or np.isnan(fetch_data(row, year, 3, None, 'GP')) or np.isnan(fetch_data(row, year, 2, None, 'GP')):
                pass
            elif row['Position'] == 'D':
                pass
            elif fetch_data(row, year, 4, None, 'GP') <= 50 or fetch_data(row, year, 3, None, 'GP') <= 50 or fetch_data(row, year, 2, None, 'GP') <= 50 or fetch_data(row, year, 1, None, 'GP') <= 50:
                pass
            else:
                # Age calculation
                # Age calculated as of the October 1st of the season
                dob = date(int(row['Date of Birth'].split('-')[0]), int(row['Date of Birth'].split('-')[1]), int(row['Date of Birth'].split('-')[2])) #year, month, date
                target_date = date(year, 10, 1)
                delta_days = target_date - dob
                age = round(delta_days.days/365.24,3)
                instance_df.loc[f"{row['Player']} {year+1}"] = [
                    row['Player'], 
                    year+1, row['Position'], 
                    age, 
                    row['Height (in)'], 
                    row['Weight (lbs)'], 
                    row['Overall Draft Position'], 
                    fetch_data(row, year, 1, None, 'GP'),
                    fetch_data(row, year, 2, None, 'GP'),
                    fetch_data(row, year, 3, None, 'GP'),
                    fetch_data(row, year, 4, None, 'GP'),
                    fetch_data(row, year, 5, None, 'GP'),
                    fetch_data(row, year, 1, 'ev', 'ATOI') + fetch_data(row, year, 1, 'pp', 'ATOI') + fetch_data(row, year, 1, 'pk', 'ATOI'),
                    fetch_data(row, year, 2, 'ev', 'ATOI') + fetch_data(row, year, 2, 'pp', 'ATOI') + fetch_data(row, year, 1, 'pk', 'ATOI'),
                    fetch_data(row, year, 3, 'ev', 'ATOI') + fetch_data(row, year, 3, 'pp', 'ATOI') + fetch_data(row, year, 1, 'pk', 'ATOI'),
                    fetch_data(row, year, 4, 'ev', 'ATOI') + fetch_data(row, year, 4, 'pp', 'ATOI') + fetch_data(row, year, 1, 'pk', 'ATOI'),
                    fetch_data(row, year, 5, 'ev', 'ATOI') + fetch_data(row, year, 5, 'pp', 'ATOI') + fetch_data(row, year, 5, 'pk', 'ATOI'),
                    fetch_data(row, year, 1, 'ev', 'G/60')*fetch_data(row, year, 1, 'ev', 'ATOI')/60*82 + fetch_data(row, year, 1, 'pp', 'G/60')*fetch_data(row, year, 1, 'pp', 'ATOI')/60*82 +fetch_data(row, year, 1, 'pk', 'G/60')*fetch_data(row, year, 1, 'pk', 'ATOI')/60*82,
                    fetch_data(row, year, 1, 'ev', 'G/60')*fetch_data(row, year, 2, 'ev', 'ATOI')/60*82 + fetch_data(row, year, 2, 'pp', 'G/60')*fetch_data(row, year, 2, 'pp', 'ATOI')/60*82 +fetch_data(row, year, 2, 'pk', 'G/60')*fetch_data(row, year, 2, 'pk', 'ATOI')/60*82,
                    fetch_data(row, year, 1, 'ev', 'G/60')*fetch_data(row, year, 3, 'ev', 'ATOI')/60*82 + fetch_data(row, year, 3, 'pp', 'G/60')*fetch_data(row, year, 3, 'pp', 'ATOI')/60*82 +fetch_data(row, year, 3, 'pk', 'G/60')*fetch_data(row, year, 3, 'pk', 'ATOI')/60*82,
                    fetch_data(row, year, 1, 'ev', 'G/60')*fetch_data(row, year, 4, 'ev', 'ATOI')/60*82 + fetch_data(row, year, 4, 'pp', 'G/60')*fetch_data(row, year, 4, 'pp', 'ATOI')/60*82 +fetch_data(row, year, 4, 'pk', 'G/60')*fetch_data(row, year, 4, 'pk', 'ATOI')/60*82,
                    fetch_data(row, year, 1, 'ev', 'G/60')*fetch_data(row, year, 5, 'ev', 'ATOI')/60*82 + fetch_data(row, year, 5, 'pp', 'G/60')*fetch_data(row, year, 5, 'pp', 'ATOI')/60*82 +fetch_data(row, year, 5, 'pk', 'G/60')*fetch_data(row, year, 5, 'pk', 'ATOI')/60*82
                ]

    instance_df['Y1 GP'] = instance_df['Y1 GP'].fillna(0)
    instance_df['Y2 GP'] = instance_df['Y2 GP'].fillna(0)
    instance_df['Y3 GP'] = instance_df['Y3 GP'].fillna(0)
    instance_df['Y4 GP'] = instance_df['Y4 GP'].fillna(0)
    instance_df['Y5 GP'] = instance_df['Y5 GP'].fillna(0)

    if download_file == True:
        filename = f'{dependent_variable}_instance_training_data'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        instance_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return instance_df

def fetch_data(row, year, yX, situation, stat):
    situation_reassignment = {'ev': 'EV', '5v5': '5v5', 'pp': 'PP', 'pk': 'PK'}
    try:
        if situation == None:
            result = row[f'{year+yX-4} {stat}']
            if stat == 'GP':
                if year+yX-4 == 2021:
                    result = result/56*82
                elif stat == 'GP' and year+yX-4 == 2020:
                    result = result/69.5*82
                try:
                    result = int(result)
                except ValueError:
                    pass
        else:
            result = row[f'{year+yX-4} {situation_reassignment[situation]} {stat}']
    except KeyError:
        result = np.nan
    return result

stat_df = scrape_player_statistics(True)
print(stat_df)
forward_gp_instance_df = create_instance_df('forward_GP', [], stat_df, True)
print(forward_gp_instance_df)
