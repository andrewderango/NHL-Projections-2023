import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter


stat_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/historical_player_statistics.csv").fillna(0)
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

print(hist_goal_df.head(750).to_string())

