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

    instance_df = pd.DataFrame(columns=['Player', 'Year', 'Position', 'Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 dGP'])

    for index, row in stat_df.iterrows():
        for year in range(start_year+4, end_year):
            # filter out:
                # defence
                # players with 0 GP in Y5
                # players with less than 40 GP in either Y4, Y3, or Y2
                # instances where Y5 was 2011 or earlier

            if np.isnan(fetch_data(row, year, 5, None, 'GP')) or np.isnan(fetch_data(row, year, 4, None, 'GP')) or np.isnan(fetch_data(row, year, 3, None, 'GP')) or np.isnan(fetch_data(row, year, 2, None, 'GP')):
                pass
            elif row['Position'] == 'D':
                pass
            elif fetch_data(row, year, 5, None, 'GP') <= 60 or fetch_data(row, year, 4, None, 'GP') <= 60 or fetch_data(row, year, 3, None, 'GP') <= 60 or fetch_data(row, year, 2, None, 'GP') <= 60 or fetch_data(row, year, 1, None, 'GP') <= 60:
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
                    fetch_data(row, year, 1, None, 'GP'),
                    fetch_data(row, year, 2, None, 'GP'),
                    fetch_data(row, year, 3, None, 'GP'),
                    fetch_data(row, year, 4, None, 'GP'),
                    fetch_data(row, year, 5, None, 'GP') - (fetch_data(row, year, 1, None, 'GP') + fetch_data(row, year, 2, None, 'GP') + fetch_data(row, year, 3, None, 'GP') + fetch_data(row, year, 4, None, 'GP'))/4

                ]

    instance_df['Y1 GP'] = instance_df['Y1 GP'].fillna(0)
    instance_df['Y2 GP'] = instance_df['Y2 GP'].fillna(0)
    instance_df['Y3 GP'] = instance_df['Y3 GP'].fillna(0)
    instance_df['Y4 GP'] = instance_df['Y4 GP'].fillna(0)
    instance_df['Y5 dGP'] = instance_df['Y5 dGP'].fillna(0)

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

def permutation_feature_importance(model, X_test_scaled, y_test, scoring='neg_mean_absolute_error'):
    # Compute permutation importances
    result = permutation_importance(model, X_test_scaled, y_test, scoring=scoring)
    sorted_idx = result.importances_mean.argsort()

    # Define color map and normalization
    cmap = cm.get_cmap('seismic_r')
    normalize = plt.Normalize(result.importances_mean[sorted_idx].min(), result.importances_mean[sorted_idx].max())

    # Create a scalar mappable
    scalar_mappable = ScalarMappable(norm=normalize, cmap=cmap)

    # Plot permutation importances
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_colors = scalar_mappable.to_rgba(result.importances_mean[sorted_idx])
    ax.barh(range(X_test_scaled.shape[1]), result.importances_mean[sorted_idx], color=bar_colors)
    ax.set_yticks(range(X_test_scaled.shape[1]))
    ax.set_yticklabels(np.array(['Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP'])[sorted_idx])
    ax.set_title("Permutation Feature Importance Analysis", weight='bold', fontsize=15, pad=20)
    ax.text(0.5, 1.02, 'Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but how important this feature is for a particular model.', ha='center', va='center', transform=ax.transAxes, fontsize=7, fontstyle='italic')
    ax.set_xlabel("Importance Score", weight='bold')
    ax.set_ylabel("Features", weight='bold')
    ax.tick_params(length=0)
    plt.box(True) # False to hide box
    # plt.tight_layout()
    plt.show()

# Edit this
def mean_decrease_in_impurity_analysis():
    df = pd.read_csv(f'{os.path.dirname(__file__)}/CSV Data/forward_GP_ADV_instance_training_data.csv')
    df = df.dropna()
    print(df)

    X = df[['Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y1 G/82', 'Y2 G/82', 'Y3 G/82', 'Y4 G/82', 'Y1 P/82', 'Y2 P/82', 'Y3 P/82', 'Y4 P/82']] # features
    y = df['Y5 GP'] # target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Train a random forest regressor
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importances using MDI algorithm
    importances = rf.feature_importances_

    # Create a pandas Series object with feature importances
    feat_importances = pd.Series(importances, index=X.columns)

    # Sort the feature importances in descending order
    feat_importances = feat_importances.sort_values(ascending=True)

    # Create a bar chart of the feature importances
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plasma_r(feat_importances.values / max(feat_importances.values))
    ax.barh(y=feat_importances.index, width=feat_importances.values, color=colors)
    ax.set_title("Random Forest Feature Importances (MDI)", weight='bold', fontsize=15, pad=20)
    ax.text(0.5, 1.02, 'Mean Decrease in Impurity', ha='center', va='center', transform=ax.transAxes, fontsize=9, fontstyle='italic')
    ax.set_xlabel("Relative Importance", weight='bold')
    ax.set_ylabel("Feature", weight='bold')
    ax.tick_params(length=0)
    plt.box(False)
    ax.figure.tight_layout()
    plt.show()

def main():
    stat_df = scrape_player_statistics(True)
    print(stat_df)
    forward_gp_instance_df = create_instance_df('forward_GP', [], stat_df, True)
    print(forward_gp_instance_df)
