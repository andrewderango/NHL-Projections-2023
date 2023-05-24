import preprocessing_training_functions as ptf
import os
import pandas as pd
import numpy as np

def aggregate_stats(evrate, evatoi, pprate, ppatoi, pkrate, pkatoi, gp, float=False):
    if float == True:
        return round((evrate/60 * evatoi + pprate/60 * ppatoi + pkrate/60 * pkatoi) * gp,2)
    else:
        return int(round((evrate/60 * evatoi + pprate/60 * ppatoi + pkrate/60 * pkatoi) * gp))
    
def aggregate_years(stat_list):
    return sum(stat_list)

stat_df = ptf.scrape_player_statistics(True)
stat_df = stat_df.fillna(0)
shooting_talent_cols = ['Player', 'Position', 'Age']

start_year = 2007
end_year = 2023

for year in range(start_year, end_year):
    shooting_talent_cols.append(f'{year+1} Goals')
    shooting_talent_cols.append(f'{year+1} xGoals')
    shooting_talent_cols.append(f'{year+1} Shots')
    stat_df[f'{year+1} Goals'] = stat_df.apply(lambda row: aggregate_stats(row[f'{year+1} EV G/60'], row[f'{year+1} EV ATOI'], row[f'{year+1} PP G/60'], row[f'{year+1} PP ATOI'], row[f'{year+1} PK G/60'], row[f'{year+1} PK ATOI'], row[f'{year+1} GP']), axis=1)
    stat_df[f'{year+1} xGoals'] = stat_df.apply(lambda row: aggregate_stats(row[f'{year+1} EV ixG/60'], row[f'{year+1} EV ATOI'], row[f'{year+1} PP ixG/60'], row[f'{year+1} PP ATOI'], row[f'{year+1} PK ixG/60'], row[f'{year+1} PK ATOI'], row[f'{year+1} GP'], True), axis=1)
    stat_df[f'{year+1} Shots'] = stat_df.apply(lambda row: aggregate_stats(row[f'{year+1} EV Shots/60'], row[f'{year+1} EV ATOI'], row[f'{year+1} PP Shots/60'], row[f'{year+1} PP ATOI'], row[f'{year+1} PK Shots/60'], row[f'{year+1} PK ATOI'], row[f'{year+1} GP']), axis=1)

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
shooting_talent_df['Shooting Talent'] = (shooting_talent_df['Relevant Goals'] -  shooting_talent_df['Relevant xGoals']) / shooting_talent_df['Relevant xGoals'] * 100

shooting_talent_df = shooting_talent_df.sort_values(by='Shooting Talent', ascending=False)
shooting_talent_df = shooting_talent_df.reset_index(drop=True)
print(shooting_talent_df)

filename = f'shooting_talent'
if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
    os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
shooting_talent_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')