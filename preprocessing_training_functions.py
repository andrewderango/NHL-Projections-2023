import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from matplotlib import cm
from matplotlib.cm import ScalarMappable, plasma_r
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tensorflow as tf
import statistics

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
        stat_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/historical_player_statistics.csv")
    elif existing_csv == False:
        player_bio_df = scrape_bios(False)
        player_bio_df = player_bio_df.drop(player_bio_df.columns[0], axis=1)

        stat_df = prune_bios(player_bio_df)
        stat_df = scrape_statistics(stat_df, 'ev', 'std', True)
        stat_df = scrape_statistics(stat_df, 'pp', 'std', True)
        stat_df = scrape_statistics(stat_df, 'pk', 'std', True)
        stat_df = scrape_statistics(stat_df, 'ev', 'oi', True)
        stat_df = scrape_statistics(stat_df, 'pp', 'oi', True)
        stat_df = scrape_statistics(stat_df, 'pk', 'oi', True)

    return stat_df

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

    if download_file == True:
        filename = f'{dependent_variable}_instance_training_data'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        instance_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return instance_df

def create_year_restricted_instance_df(proj_stat, position, prev_years, download_file=True):
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

    if download_file == True:
        filename = f'{position}_{proj_stat}_{prev_years}year_instance_training_data'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        instance_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return instance_df, input_shape

def extract_instance_data(instance_df, proj_stat, prev_years):
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
                try:
                    result = int(result)
                except ValueError:
                    pass
        else:
            result = row[f'{year+yX-4} {situation_reassignment[situation]} {stat}']
    except KeyError:
        result = np.nan
    return result

def permutation_feature_importance(model, X_scaled, y, scoring='neg_mean_absolute_error'):
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
    ax.set_yticklabels(np.array(['Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP'])[sorted_idx])
    ax.set_title("Permutation Feature Importance Analysis", weight='bold', fontsize=15, pad=20)
    ax.text(0.5, 1.02, 'Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but how important this feature is for a particular model.', ha='center', va='center', transform=ax.transAxes, fontsize=7, fontstyle='italic')
    ax.set_xlabel("Importance Score", weight='bold')
    ax.set_ylabel("Features", weight='bold')
    ax.tick_params(length=0)
    plt.box(True) # False to hide box
    # plt.tight_layout()
    plt.show()

def make_gp_projections():
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

    instance_df = create_instance_df('forward_GP', ['Player', 'Year', 'Position', 'Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 'Y5 dGP'], scrape_player_statistics(True), True)

def main():
    stat_df = scrape_player_statistics(True)
    print(stat_df)
    forward_gp_instance_df = create_instance_df('forward_GP', ['Player', 'Year', 'Position', 'Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 dGP'], stat_df, True)
    print(forward_gp_instance_df)

make_gp_projections()
