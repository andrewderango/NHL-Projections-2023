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
                    if not os.path.exists(f'{os.path.dirname(__file__)}/Output CSV Data'):
                        os.makedirs(f'{os.path.dirname(__file__)}/Output CSV Data')
                    running_df.to_csv(f'{os.path.dirname(__file__)}/Output CSV Data/{filename}.csv')
                    print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/Output CSV Data')

                break
            except requests.exceptions.ConnectionError:
                print('Connection failed. Periodic request quota exceeded. Trying again in 5 seconds.')
                time.sleep(5)

    return running_df

def prune_bios(player_bio_df):
    stat_df = player_bio_df.drop(['Team', 'Birth City', 'Birth State/Province', 'Birth Country', 'Nationality', 'Draft Team', 'Draft Round', 'Round Pick'], axis=1)
    return stat_df

def scrape_statistics(stat_df, situation='ev', download_file=False):
    # situation = (ev, 5v5, pp, pk ...)
    start_year = 2007
    end_year = 2023
    stat_df = stat_df.set_index('Player')
    situation_reassignment = {'ev': 'EV', '5v5': '5v5', 'pp': 'PP', 'pk': 'PK'}

    for year in range(start_year, end_year):
        url = f'https://www.naturalstattrick.com/playerteams.php?fromseason={year}{year+1}&thruseason={year}{year+1}&stype=2&sit={situation}&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL'

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

                # The replacement is necessary because Jani Hakanpää is spelled different in the bios and statistics
                for index, row in temp_df.iterrows():
                    stat_df.loc[row['Player'].replace('ä', 'a'), f'{year+1} GP'] = row['GP']
                    stat_df.loc[row['Player'].replace('ä', 'a'), f'{year+1} {situation_reassignment[situation]} ATOI'] = round(float(row['TOI'])/int(row['GP']),2)
                    stat_df.loc[row['Player'].replace('ä', 'a'), f'{year+1} {situation_reassignment[situation]} G/60'] = round(float(row['Goals'])/float(row['TOI']),4)
                    stat_df.loc[row['Player'].replace('ä', 'a'), f'{year+1} {situation_reassignment[situation]} A/60'] = round(float(row['Total Assists'])/float(row['TOI']),4)

                print(f'{year}-{year+1}: Scraped. Dimensions = {stat_df.shape}')
            except requests.exceptions.ConnectionError:
                print('Connection failed. Periodic request quota exceeded. Trying again in 5 seconds.')
                time.sleep(5)

    if download_file == True:
        filename = f'historical_player_statistics'
        if not os.path.exists(f'{os.path.dirname(__file__)}/Output CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/Output CSV Data')
        stat_df.to_csv(f'{os.path.dirname(__file__)}/Output CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/Output CSV Data')

    return stat_df

def create_instance_df(stat_df, download_file=False):
    start_year = 2007
    end_year = 2023

    # Age calculated as of the October 1st of the season
    instance_df = pd.DataFrame(index=['Instance ID'], columns=['Player', 'Year', 'Position', 'Age', 'Height', 'Weight', 'Draft Position', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 'Y1 ev ATOI', 'Y2 ev ATOI', 'Y3 ev ATOI', 'Y4 ev ATOI', 'Y5 ev ATOI'])
    for index, row in stat_df.iterrows():
        for year in range(start_year, end_year):
            print(f'{len(instance_df)} / 45984')

            # Age calculation
            dob = date(int(row['Date of Birth'].split('-')[0]), int(row['Date of Birth'].split('-')[1]), int(row['Date of Birth'].split('-')[2])) #year, month, date
            target_date = date(year, 10, 1)
            delta_days = target_date - dob
            age = round(delta_days.days/365.24,3)

            try: y1_gp = row[f'{year-3} GP']
            except KeyError: y1_gp = np.NaN
            try: y2_gp = row[f'{year-2} GP']
            except KeyError: y2_gp = np.NaN
            try: y3_gp = row[f'{year-1} GP']
            except KeyError: y3_gp = np.NaN
            try: y4_gp = row[f'{year+0} GP']
            except KeyError: y4_gp = np.NaN
            try: y5_gp = row[f'{year+1} GP']
            except KeyError: y5_gp = np.NaN

            instance_df.loc[f"{row['Player']} {year+1}"] = [row['Player'], 
                                                            year+1, row['Position'], 
                                                            age, 
                                                            row['Height (in)'], 
                                                            row['Weight (lbs)'], 
                                                            row['Overall Draft Position'], 
                                                            error_catch_input_data(row, year, 1, None, 'GP'),
                                                            error_catch_input_data(row, year, 2, None, 'GP'),
                                                            error_catch_input_data(row, year, 3, None, 'GP'),
                                                            error_catch_input_data(row, year, 4, None, 'GP'),
                                                            error_catch_input_data(row, year, 5, None, 'GP'),
                                                            error_catch_input_data(row, year, 1, 'ev', 'ATOI'),
                                                            error_catch_input_data(row, year, 2, 'ev', 'ATOI'),
                                                            error_catch_input_data(row, year, 3, 'ev', 'ATOI'),
                                                            error_catch_input_data(row, year, 4, 'ev', 'ATOI'),
                                                            error_catch_input_data(row, year, 5, 'ev', 'ATOI')
            ]

    if download_file == True:
        filename = f'instance_training_data'
        if not os.path.exists(f'{os.path.dirname(__file__)}/Output CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/Output CSV Data')
        instance_df.to_csv(f'{os.path.dirname(__file__)}/Output CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/Output CSV Data')

    return instance_df

def error_catch_input_data(row, year, yX, situation, stat):
    try:
        if situation == None:
            result = row[f'{year+yX-4} {stat}']
        else:
            result = row[f'{year+yX-4} {situation} {stat}']
    except KeyError:
        result = np.nan
    return result


player_bio_df = pd.read_csv(f"{os.path.dirname(__file__)}/Output CSV Data/player_bios.csv")
# player_bio_df = scrape_bios(False)
player_bio_df = player_bio_df.drop(player_bio_df.columns[0], axis=1)

stat_df = pd.read_csv(f"{os.path.dirname(__file__)}/Output CSV Data/historical_player_statistics.csv")
# stat_df = prune_bios(player_bio_df)
# stat_df = scrape_statistics(stat_df)
# if you want to add more statistics for all situations to update stat_df, run:
# stat_df = scrape_statistics(stat_df, 'ev', True)
# stat_df = scrape_statistics(stat_df, 'pp', True)
# stat_df = scrape_statistics(stat_df, 'pk', True)
# watch periodic request quota

print(stat_df)
print(create_instance_df(stat_df, True))
