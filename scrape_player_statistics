import requests
from bs4 import BeautifulSoup
import pandas as pd

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

    print(running_df.to_string())
    print(f'{year}-{year+1}\n\n\n')

    running_df.to_csv('player_bios.csv')
