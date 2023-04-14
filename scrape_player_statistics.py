import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_bios(download_file=False):
    start_year = 2006
    end_year = 2023

    running_df = None
    for year in range(start_year, end_year):
        url = f'https://www.naturalstattrick.com/playerteams.php?fromseason={year}{year+1}&thruseason={year}{year+1}&stype=2&sit=all&score=all&stdoi=bio&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL'

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
            filename = f'player_bios_'
            if not os.path.exists(f'{os.path.dirname(__file__)}/Output CSV Data'):
                os.makedirs(f'{os.path.dirname(__file__)}/Output CSV Data')
            running_df.to_csv(f'{os.path.dirname(__file__)}/Output CSV Data/{filename}.csv')
            print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/Output CSV Data')

    return running_df

def prune_bios(player_bio_df):
    stat_df = player_bio_df.drop(['Team', 'Birth City', 'Birth State/Province', 'Birth Country', 'Nationality', 'Draft Team', 'Draft Round', 'Round Pick'], axis=1)
    return stat_df

def scrape_statistics(stat_df, situation='ev', download_file=False):
    # situation = (ev, 5v5, pp, pk ...)
    start_year = 2006
    end_year = 2023
    stat_df = stat_df.set_index('Player')

    for year in range(start_year, end_year):
        url = f'https://www.naturalstattrick.com/playerteams.php?fromseason={year}{year+1}&thruseason={year}{year+1}&stype=2&sit={situation}&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL'

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
            stat_df.loc[row['Player'].replace('ä', 'a'), f'{year+1} ATOI'] = round(float(row['TOI'])/int(row['GP']),2)
            stat_df.loc[row['Player'].replace('ä', 'a'), f'{year+1} G/60'] = round(float(row['Goals'])/float(row['TOI']),4)
            stat_df.loc[row['Player'].replace('ä', 'a'), f'{year+1} A/60'] = round(float(row['Total Assists'])/float(row['TOI']),4)

        print(f'{year}-{year+1}: Scraped. Dimensions = {stat_df.shape}')

    return stat_df

player_bio_df = pd.read_csv(f"{os.path.dirname(__file__)}/Output CSV Data/player_bios.csv")
# player_bio_df = scrape_bios(False)
player_bio_df = player_bio_df.drop(player_bio_df.columns[0], axis=1)
stat_df = prune_bios(player_bio_df)
stat_df = scrape_statistics(stat_df)

print(stat_df)

filename = f'historical_player_statistics'
if not os.path.exists(f'{os.path.dirname(__file__)}/Output CSV Data'):
    os.makedirs(f'{os.path.dirname(__file__)}/Output CSV Data')
stat_df.to_csv(f'{os.path.dirname(__file__)}/Output CSV Data/{filename}.csv')
print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/Output CSV Data')
