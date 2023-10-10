import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import preprocessing_training_functions

def make_forward_gp_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 4, None, 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'GP', 4, None, None, 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 3, None, 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 3, None, None, 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 2, None, 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'GP', 2, None, None, 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 1, None, 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'GP', 1, None, None, 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].iloc[0]*gp_adjustment_factor[year-4],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].iloc[0] + stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].iloc[0] + stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].iloc[0]])
    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
    for player in yr1_group:
        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, 82)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, 82)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, 82)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, 82)

    column_name = 'GP'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_gp_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 4, None, 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'GP', 4, None, None, 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 3, None, 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 3, None, None, 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 2, None, 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'GP', 2, None, None, 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 1, None, 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'GP', 1, None, None, 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].iloc[0]*gp_adjustment_factor[year-4],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].iloc[0] + stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].iloc[0] + stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].iloc[0]])
    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
    for player in yr1_group:
        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, 82)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, 82)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, 82)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, 82)

    column_name = 'GP'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_ev_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'EV', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'EV', None, 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'EV', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV', None, 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'EV', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'EV', None, 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'EV', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'EV', None, 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_ev_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'EV', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'EV', None, 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'EV', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV', None, 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'EV', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'EV', None, 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'EV', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'EV', None, 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pp_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'PP', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PP', None, 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'PP', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP', None, 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'PP', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PP', None, 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'PP', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PP', None, 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 20 games.
        # Once you reach 20 games, find the ATOI accross these seasons.
        # If they haven't played 20 games in their past 4 seasons, fill the rest of the 20 games with the -2nd z-score of the stat.
        if y1_gp >= 20:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 PP ATOI'].mean() - instance_df_y1['Y4 PP ATOI'].std()*2, 0) # should not be negative
            games_to_pseudofy = 20-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'PP', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PP', None, 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'PP', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP', None, 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'PP', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PP', None, 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'PP', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PP', None, 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 20 games.
        # Once you reach 20 games, find the ATOI accross these seasons.
        # If they haven't played 20 games in their past 4 seasons, fill the rest of the 20 games with the -2nd z-score of the stat.
        if y1_gp >= 20:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 PP ATOI'].mean() - instance_df_y1['Y4 PP ATOI'].std()*2, 0) # should not be negative
            games_to_pseudofy = 20-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])

    yr4_stat_list = np.where(np.isnan(yr4_stat_list), 0, yr4_stat_list) 
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'PK', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PK', None, 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'PK', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK', None, 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'PK', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PK', None, 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'PK', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PK', None, 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
        # If they haven't played 20 games in their past 4 seasons, fill the rest of the 20 games with the -1.5th z-score of the stat.
        if y1_gp >= 20:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 PK ATOI'].mean() - instance_df_y1['Y4 PK ATOI'].std()*1.5, 0) # should not be negative
            games_to_pseudofy = 20-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'PK', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PK', None, 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'PK', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK', None, 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'PK', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PK', None, 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'PK', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PK', None, 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
        # If they haven't played 20 games in their past 4 seasons, fill the rest of the 20 games with the -1.5th z-score of the stat.
        if y1_gp >= 20:
            pseudo_prev_year_stat = y1_stat
        elif y1_gp + y2_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 20:
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score = max(instance_df_y1['Y4 PK ATOI'].mean() - instance_df_y1['Y4 PK ATOI'].std()*1.5, 0) # should not be negative
            games_to_pseudofy = 20-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_ev_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'EV', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'EV', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'EV', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'EV', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = y4_svr_model.predict(yr4_stat_list_scaled)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_svr_model.predict(yr3_stat_list_scaled)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_svr_model.predict(yr2_stat_list_scaled)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'EV G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(yr4_predictions[index] + np.mean(statline[3:7]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(yr3_predictions[index] + np.mean(statline[3:6]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(yr2_predictions[index] + np.mean(statline[3:5]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_ev_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'EV', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'EV', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'EV', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'EV', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = y4_svr_model.predict(yr4_stat_list_scaled)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_svr_model.predict(yr3_stat_list_scaled)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_svr_model.predict(yr2_stat_list_scaled)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'EV G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(yr4_predictions[index] + np.mean(statline[3:7]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(yr3_predictions[index] + np.mean(statline[3:6]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(yr2_predictions[index] + np.mean(statline[3:5]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pp_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'PP', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PP', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PP', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'PP', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PP', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'PP', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PP', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = y4_svr_model.predict(yr4_stat_list_scaled)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_svr_model.predict(yr3_stat_list_scaled)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_svr_model.predict(yr2_stat_list_scaled)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PP G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(yr4_predictions[index] + np.mean(statline[3:7]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(yr3_predictions[index] + np.mean(statline[3:6]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(yr2_predictions[index] + np.mean(statline[3:5]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'PP', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PP', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'PP', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'PP', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PP', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'PP', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PP', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = y4_svr_model.predict(yr4_stat_list_scaled)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_svr_model.predict(yr3_stat_list_scaled)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_svr_model.predict(yr2_stat_list_scaled)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PP G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(yr4_predictions[index] + np.mean(statline[3:7]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(yr3_predictions[index] + np.mean(statline[3:6]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(yr2_predictions[index] + np.mean(statline[3:5]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'PK', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PK', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PK', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'PK', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PK', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'PK', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PK', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PK G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'PK', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PK', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'PK', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'PK', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PK', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'PK', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PK', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PK G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_ev_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'EV', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'EV', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'EV', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'EV', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'EV', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'EV', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'EV', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'EV', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rebounds Created/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rush Attempts/60'].fillna(0).iloc[0]
        y1_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
        y2_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0]
        y3_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0]
        y4_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV oixGF/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 50 games.
        # Once you reach 50 games, find the ATOI accross these seasons.
        # If they haven't played 50 games in their past 4 seasons, fill the rest of the 50 games with the -1st z-score of the stat.
        if y1_gp >= 50:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
            pseudo_prev_year_stat_5 = y1_stat_5
        elif y1_gp + y2_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp + y4_stat_3*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp + y4_stat_4*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp + y4_stat_5*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV A1/60'].mean() - instance_df_y1['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV A2/60'].mean() - instance_df_y1['Y4 EV A2/60'].std(), 0)
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 EV Rebounds Created/60'].mean() - instance_df_y1['Y4 EV Rebounds Created/60'].std(), 0)
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 EV Rush Attempts/60'].mean() - instance_df_y1['Y4 EV Rush Attempts/60'].std(), 0)
            negative_first_z_score_stat_5 = max(instance_df_y1['Y4 EV oixGF/60'].mean() - instance_df_y1['Y4 EV oixGF/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy) # denominator = 50
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp + y4_stat_3*y4_gp + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp + y4_stat_4*y4_gp + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp + y4_stat_5*y4_gp + negative_first_z_score_stat_5*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4,
            pseudo_prev_year_stat_5
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = y4_svr_model.predict(yr4_stat_list_scaled)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_svr_model.predict(yr3_stat_list_scaled)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_svr_model.predict(yr2_stat_list_scaled)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'EV A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(yr4_predictions[index] + np.mean(statline[3:7]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(yr3_predictions[index] + np.mean(statline[3:6]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(yr2_predictions[index] + np.mean(statline[3:5]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_ev_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'EV', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'EV', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'EV', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'EV', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'EV', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'EV', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'EV', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'EV', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rebounds Created/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rush Attempts/60'].fillna(0).iloc[0]
        y1_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
        y2_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0]
        y3_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0]
        y4_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV oixGF/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 50 games.
        # Once you reach 50 games, find the ATOI accross these seasons.
        # If they haven't played 50 games in their past 4 seasons, fill the rest of the 50 games with the -1st z-score of the stat.
        if y1_gp >= 50:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
            pseudo_prev_year_stat_5 = y1_stat_5
        elif y1_gp + y2_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp + y4_stat_3*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp + y4_stat_4*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp + y4_stat_5*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV A1/60'].mean() - instance_df_y1['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV A2/60'].mean() - instance_df_y1['Y4 EV A2/60'].std(), 0)
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 EV Rebounds Created/60'].mean() - instance_df_y1['Y4 EV Rebounds Created/60'].std(), 0)
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 EV Rush Attempts/60'].mean() - instance_df_y1['Y4 EV Rush Attempts/60'].std(), 0)
            negative_first_z_score_stat_5 = max(instance_df_y1['Y4 EV oixGF/60'].mean() - instance_df_y1['Y4 EV oixGF/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy) # denominator = 50
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp + y4_stat_3*y4_gp + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp + y4_stat_4*y4_gp + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp + y4_stat_5*y4_gp + negative_first_z_score_stat_5*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4,
            pseudo_prev_year_stat_5
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = y4_svr_model.predict(yr4_stat_list_scaled)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_svr_model.predict(yr3_stat_list_scaled)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_svr_model.predict(yr2_stat_list_scaled)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'EV A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(yr4_predictions[index] + np.mean(statline[3:7]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(yr3_predictions[index] + np.mean(statline[3:6]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(yr2_predictions[index] + np.mean(statline[3:5]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pp_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'PP', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PP', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'PP', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PP', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'PP', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PP', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'PP', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PP', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
 
    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0]
        y1_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0]
        y2_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0]
        y3_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0]
        y4_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP Rebounds Created/60'].fillna(0).iloc[0]
        y1_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
        y2_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0]
        y3_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0]
        y4_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP oixGF/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPA1/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
            pseudo_prev_year_stat_5 = y1_stat_5
            pseudo_prev_year_stat_6 = y1_stat_6
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi + y4_stat_5*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi + y4_stat_6*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV A1/60'].mean() - instance_df_y1['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV A2/60'].mean() - instance_df_y1['Y4 EV A2/60'].std(), 0)
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 PP A1/60'].mean() - instance_df_y1['Y4 PP A1/60'].std(), 0)
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 PP A2/60'].mean() - instance_df_y1['Y4 PP A2/60'].std(), 0)
            negative_first_z_score_stat_5 = max(instance_df_y1['Y4 PP Rebounds Created/60'].mean() - instance_df_y1['Y4 PP Rebounds Created/60'].std(), 0)
            negative_first_z_score_stat_6 = max(instance_df_y1['Y4 PP oixGF/60'].mean() - instance_df_y1['Y4 PP oixGF/60'].std(), 0)
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi + y4_stat_5*y4_pptoi + negative_first_z_score_stat_5*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi + y4_stat_6*y4_pptoi + negative_first_z_score_stat_6*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4,
            pseudo_prev_year_stat_5,
            pseudo_prev_year_stat_6
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = y4_svr_model.predict(yr4_stat_list_scaled)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_svr_model.predict(yr3_stat_list_scaled)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_svr_model.predict(yr2_stat_list_scaled)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PP A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(yr4_predictions[index] + np.mean(statline[11:15]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(yr3_predictions[index] + np.mean(statline[9:12]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(yr2_predictions[index] + np.mean(statline[7:9]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'PP', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PP', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'PP', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PP', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'PP', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PP', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'PP', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PP', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, np.nan_to_num(y_4.ravel(), nan=0))
    y3_svr_model.fit(X_3_scaled, np.nan_to_num(y_3.ravel(), nan=0))
    y2_svr_model.fit(X_2_scaled, np.nan_to_num(y_2.ravel(), nan=0))
    y1_svr_model.fit(X_1_scaled, np.nan_to_num(y_1.ravel(), nan=0))

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
 
    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0]
        y1_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0]
        y2_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0]
        y3_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0]
        y4_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP Rebounds Created/60'].fillna(0).iloc[0]
        y1_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
        y2_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0]
        y3_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0]
        y4_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP oixGF/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPA1/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
            pseudo_prev_year_stat_5 = y1_stat_5
            pseudo_prev_year_stat_6 = y1_stat_6
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi + y4_stat_5*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi + y4_stat_6*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV A1/60'].mean() - instance_df_y1['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV A2/60'].mean() - instance_df_y1['Y4 EV A2/60'].std(), 0)
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 PP A1/60'].mean() - instance_df_y1['Y4 PP A1/60'].std(), 0)
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 PP A2/60'].mean() - instance_df_y1['Y4 PP A2/60'].std(), 0)
            negative_first_z_score_stat_5 = max(instance_df_y1['Y4 PP Rebounds Created/60'].mean() - instance_df_y1['Y4 PP Rebounds Created/60'].std(), 0)
            negative_first_z_score_stat_6 = max(instance_df_y1['Y4 PP oixGF/60'].mean() - instance_df_y1['Y4 PP oixGF/60'].std(), 0)
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi + y4_stat_5*y4_pptoi + negative_first_z_score_stat_5*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi + y4_stat_6*y4_pptoi + negative_first_z_score_stat_6*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4,
            pseudo_prev_year_stat_5,
            pseudo_prev_year_stat_6
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = y4_svr_model.predict(yr4_stat_list_scaled)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_svr_model.predict(yr3_stat_list_scaled)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_svr_model.predict(yr2_stat_list_scaled)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PP A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = max(yr4_predictions[index] + np.mean(statline[12:15]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = max(yr3_predictions[index] + np.mean(statline[9:12]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = max(yr2_predictions[index] + np.mean(statline[7:9]), 0)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'PK', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PK', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'PK', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PK', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'PK', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PK', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'PK', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PK', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A1/60'].fillna(0).iloc[0]

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
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PK A1/60'].mean() - instance_df_y1['Y4 PK A1/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PK A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'PK', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PK', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'PK', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PK', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'PK', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PK', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'PK', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PK', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A1/60'].fillna(0).iloc[0]

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
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PK A1/60'].mean() - instance_df_y1['Y4 PK A1/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PK A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_ev_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'EV', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'EV', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'EV', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'EV', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'EV', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'EV', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'EV', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'EV', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rebounds Created/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rush Attempts/60'].fillna(0).iloc[0]
        y1_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
        y2_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0]
        y3_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0]
        y4_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV oixGF/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 50 games.
        # Once you reach 50 games, find the ATOI accross these seasons.
        # If they haven't played 50 games in their past 4 seasons, fill the rest of the 50 games with the -1st z-score of the stat.
        if y1_gp >= 50:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
            pseudo_prev_year_stat_5 = y1_stat_5
        elif y1_gp + y2_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp + y4_stat_3*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp + y4_stat_4*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp + y4_stat_5*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV A1/60'].mean() - instance_df_y1['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV A2/60'].mean() - instance_df_y1['Y4 EV A2/60'].std(), 0)
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 EV Rebounds Created/60'].mean() - instance_df_y1['Y4 EV Rebounds Created/60'].std(), 0)
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 EV Rush Attempts/60'].mean() - instance_df_y1['Y4 EV Rush Attempts/60'].std(), 0)
            negative_first_z_score_stat_5 = max(instance_df_y1['Y4 EV oixGF/60'].mean() - instance_df_y1['Y4 EV oixGF/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy) # denominator = 50
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp + y4_stat_3*y4_gp + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp + y4_stat_4*y4_gp + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp + y4_stat_5*y4_gp + negative_first_z_score_stat_5*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4,
            pseudo_prev_year_stat_5
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'EV A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_ev_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'EV', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'EV', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'EV', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'EV', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'EV', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'EV', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'EV', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'EV', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rebounds Created/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rebounds Created/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rebounds Created/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rebounds Created/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV Rush Attempts/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV Rush Attempts/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV Rush Attempts/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV Rush Attempts/60'].fillna(0).iloc[0]
        y1_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV oixGF/60'].fillna(0).iloc[0]
        y2_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV oixGF/60'].fillna(0).iloc[0]
        y3_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV oixGF/60'].fillna(0).iloc[0]
        y4_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV oixGF/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 50 games.
        # Once you reach 50 games, find the ATOI accross these seasons.
        # If they haven't played 50 games in their past 4 seasons, fill the rest of the 50 games with the -1st z-score of the stat.
        if y1_gp >= 50:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
            pseudo_prev_year_stat_5 = y1_stat_5
        elif y1_gp + y2_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp)/(y1_gp + y2_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp)/(y1_gp + y2_gp)
        elif y1_gp + y2_gp + y3_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp)/(y1_gp + y2_gp + y3_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp)/(y1_gp + y2_gp + y3_gp)
        elif y1_gp + y2_gp + y3_gp + y4_gp >= 50:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp + y4_stat_3*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp + y4_stat_4*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp + y4_stat_5*y4_gp)/(y1_gp + y2_gp + y3_gp + y4_gp)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV A1/60'].mean() - instance_df_y1['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV A2/60'].mean() - instance_df_y1['Y4 EV A2/60'].std(), 0)
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 EV Rebounds Created/60'].mean() - instance_df_y1['Y4 EV Rebounds Created/60'].std(), 0)
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 EV Rush Attempts/60'].mean() - instance_df_y1['Y4 EV Rush Attempts/60'].std(), 0)
            negative_first_z_score_stat_5 = max(instance_df_y1['Y4 EV oixGF/60'].mean() - instance_df_y1['Y4 EV oixGF/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy) # denominator = 50
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_gp + y2_stat_3*y2_gp + y3_stat_3*y3_gp + y4_stat_3*y4_gp + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_gp + y2_stat_4*y2_gp + y3_stat_4*y3_gp + y4_stat_4*y4_gp + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_gp + y2_stat_5*y2_gp + y3_stat_5*y3_gp + y4_stat_5*y4_gp + negative_first_z_score_stat_5*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4,
            pseudo_prev_year_stat_5
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'EV A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pp_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'PP', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PP', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'PP', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PP', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'PP', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PP', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'PP', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PP', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
 
    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0]
        y1_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0]
        y2_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0]
        y3_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0]
        y4_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP Rebounds Created/60'].fillna(0).iloc[0]
        y1_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
        y2_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0]
        y3_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0]
        y4_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP oixGF/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPA1/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
            pseudo_prev_year_stat_5 = y1_stat_5
            pseudo_prev_year_stat_6 = y1_stat_6
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi + y4_stat_5*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi + y4_stat_6*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV A1/60'].mean() - instance_df_y1['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV A2/60'].mean() - instance_df_y1['Y4 EV A2/60'].std(), 0)
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 PP A1/60'].mean() - instance_df_y1['Y4 PP A1/60'].std(), 0)
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 PP A2/60'].mean() - instance_df_y1['Y4 PP A2/60'].std(), 0)
            negative_first_z_score_stat_5 = max(instance_df_y1['Y4 PP Rebounds Created/60'].mean() - instance_df_y1['Y4 PP Rebounds Created/60'].std(), 0)
            negative_first_z_score_stat_6 = max(instance_df_y1['Y4 PP oixGF/60'].mean() - instance_df_y1['Y4 PP oixGF/60'].std(), 0)
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi + y4_stat_5*y4_pptoi + negative_first_z_score_stat_5*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi + y4_stat_6*y4_pptoi + negative_first_z_score_stat_6*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4,
            pseudo_prev_year_stat_5,
            pseudo_prev_year_stat_6
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PP A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection =yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'PP', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PP', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'PP', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PP', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'PP', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PP', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'PP', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PP', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, np.nan_to_num(y_4.ravel(), nan=0))
    y3_svr_model.fit(X_3_scaled, np.nan_to_num(y_3.ravel(), nan=0))
    y2_svr_model.fit(X_2_scaled, np.nan_to_num(y_2.ravel(), nan=0))
    y1_svr_model.fit(X_1_scaled, np.nan_to_num(y_1.ravel(), nan=0))

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
 
    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} EV A2/60'].fillna(0).iloc[0]
        y1_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0]
        y2_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0]
        y3_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0]
        y4_stat_3 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0]
        y1_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
        y2_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0]
        y3_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0]
        y4_stat_4 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0]
        y1_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP Rebounds Created/60'].fillna(0).iloc[0]
        y2_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP Rebounds Created/60'].fillna(0).iloc[0]
        y3_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP Rebounds Created/60'].fillna(0).iloc[0]
        y4_stat_5 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP Rebounds Created/60'].fillna(0).iloc[0]
        y1_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP oixGF/60'].fillna(0).iloc[0]
        y2_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP oixGF/60'].fillna(0).iloc[0]
        y3_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP oixGF/60'].fillna(0).iloc[0]
        y4_stat_6 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP oixGF/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPA1/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
            pseudo_prev_year_stat_3 = y1_stat_3
            pseudo_prev_year_stat_4 = y1_stat_4
            pseudo_prev_year_stat_5 = y1_stat_5
            pseudo_prev_year_stat_6 = y1_stat_6
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi + y4_stat_5*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi + y4_stat_6*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 EV A1/60'].mean() - instance_df_y1['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y1['Y4 EV A2/60'].mean() - instance_df_y1['Y4 EV A2/60'].std(), 0)
            negative_first_z_score_stat_3 = max(instance_df_y1['Y4 PP A1/60'].mean() - instance_df_y1['Y4 PP A1/60'].std(), 0)
            negative_first_z_score_stat_4 = max(instance_df_y1['Y4 PP A2/60'].mean() - instance_df_y1['Y4 PP A2/60'].std(), 0)
            negative_first_z_score_stat_5 = max(instance_df_y1['Y4 PP Rebounds Created/60'].mean() - instance_df_y1['Y4 PP Rebounds Created/60'].std(), 0)
            negative_first_z_score_stat_6 = max(instance_df_y1['Y4 PP oixGF/60'].mean() - instance_df_y1['Y4 PP oixGF/60'].std(), 0)
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_5 = (y1_stat_5*y1_pptoi + y2_stat_5*y2_pptoi + y3_stat_5*y3_pptoi + y4_stat_5*y4_pptoi + negative_first_z_score_stat_5*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_6 = (y1_stat_6*y1_pptoi + y2_stat_6*y2_pptoi + y3_stat_6*y3_pptoi + y4_stat_6*y4_pptoi + negative_first_z_score_stat_6*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4,
            pseudo_prev_year_stat_5,
            pseudo_prev_year_stat_6
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PP A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection =yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'PK', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PK', 'forward', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'PK', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PK', 'forward', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'PK', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PK', 'forward', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'PK', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PK', 'forward', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A1/60'].fillna(0).iloc[0]

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
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PK A1/60'].mean() - instance_df_y1['Y4 PK A1/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PK A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    y4_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y3_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y2_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    y1_svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'PK', 'SVR')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PK', 'defence', 'SVR')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'PK', 'SVR')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PK', 'defence', 'SVR')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'PK', 'SVR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PK', 'defence', 'SVR')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'PK', 'SVR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PK', 'defence', 'SVR')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y4_svr_model.fit(X_4_scaled, y_4.ravel())
    y3_svr_model.fit(X_3_scaled, y_3.ravel())
    y2_svr_model.fit(X_2_scaled, y_2.ravel())
    y1_svr_model.fit(X_1_scaled, y_1.ravel())

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
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
        
    for player in yr1_group:
        y1_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A1/60'].fillna(0).iloc[0]

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
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PK A1/60'].mean() - instance_df_y1['Y4 PK A1/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])
        
    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = np.clip(y4_svr_model.predict(yr4_stat_list_scaled), 0, None)
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = np.clip(y3_svr_model.predict(yr3_stat_list_scaled), 0, None)
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = np.clip(y2_svr_model.predict(yr2_stat_list_scaled), 0, None)
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = np.clip(y1_svr_model.predict(yr1_stat_list_scaled), 0, None)

    column_name = 'PK A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        projection = yr4_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = yr3_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = yr2_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = yr1_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'svr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def goal_era_adjustment(stat_df, projection_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_goal_df = pd.DataFrame()

    for season in range(2007, year-1):
        col = round(((stat_df[f'{season+1} EV G/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP G/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK G/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])) 
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_goal_df = hist_goal_df.reset_index(drop=True)
        hist_goal_df[season+1] = col
    hist_goal_df.index = hist_goal_df.index + 1

    try:
        hist_goal_df[2021] = round(82/56*hist_goal_df[2021]) 
    except KeyError:
        pass
    try:
        hist_goal_df[2020] = round(82/70*hist_goal_df[2020]) 
    except KeyError:
        pass
    try:
        hist_goal_df[2013] = round(82/48*hist_goal_df[2013]) 
    except KeyError:
        pass

    hist_goal_df['Historical Average'] = hist_goal_df.mean(axis=1)
    hist_goal_df['Projected Average'] = hist_goal_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_goal_df['Adjustment'] = hist_goal_df['Projected Average'] - hist_goal_df['Historical Average']
    hist_goal_df['Smoothed Adjustment'] = savgol_filter(hist_goal_df['Adjustment'], 25, 2)
    # print(hist_goal_df.head(750).to_string())

    projection_df['GOALS'] = (projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']
    projection_df = projection_df.sort_values('GOALS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_goal_df['Smoothed Adjustment']/((projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        projection_df['EV G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['GOALS'] = (projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

    # Download file
    if download_file == True:
        filename = f'svr_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df.fillna(0)

def a1_era_adjustment(stat_df, projection_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_a1_df = pd.DataFrame()

    for season in range(2007, year-1):
        col = round(((stat_df[f'{season+1} EV A1/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP A1/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK A1/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])) 
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_a1_df = hist_a1_df.reset_index(drop=True)
        hist_a1_df[season+1] = col
    hist_a1_df.index = hist_a1_df.index + 1

    try:
        hist_a1_df[2021] = round(82/56*hist_a1_df[2021]) 
    except KeyError:
        pass
    try:
        hist_a1_df[2020] = round(82/70*hist_a1_df[2020]) 
    except KeyError:
        pass
    try:
        hist_a1_df[2013] = round(82/48*hist_a1_df[2013]) 
    except KeyError:
        pass

    hist_a1_df['Historical Average'] = hist_a1_df.mean(axis=1)
    hist_a1_df['Projected Average'] = hist_a1_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_a1_df['Adjustment'] = hist_a1_df['Projected Average'] - hist_a1_df['Historical Average']
    hist_a1_df['Smoothed Adjustment'] = savgol_filter(hist_a1_df['Adjustment'], 25, 2)
    # print(hist_a1_df.head(750).to_string())

    projection_df['PRIMARY ASSISTS'] = (projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP']
    projection_df = projection_df.sort_values('PRIMARY ASSISTS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_a1_df['Smoothed Adjustment']/((projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        projection_df['EV A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PRIMARY ASSISTS'] = (projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP']
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

    # Download file
    if download_file == True:
        filename = f'svr_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df.fillna(0)

def a2_era_adjustment(stat_df, projection_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_a2_df = pd.DataFrame()

    for season in range(2007, year-1):
        col = round(((stat_df[f'{season+1} EV A2/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP A2/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK A2/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])) 
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_a2_df = hist_a2_df.reset_index(drop=True)
        hist_a2_df[season+1] = col
    hist_a2_df.index = hist_a2_df.index + 1

    try:
        hist_a2_df[2021] = round(82/56*hist_a2_df[2021]) 
    except KeyError:
        pass
    try:
        hist_a2_df[2020] = round(82/70*hist_a2_df[2020]) 
    except KeyError:
        pass
    try:
        hist_a2_df[2013] = round(82/48*hist_a2_df[2013]) 
    except KeyError:
        pass

    hist_a2_df['Historical Average'] = hist_a2_df.mean(axis=1)
    hist_a2_df['Projected Average'] = hist_a2_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_a2_df['Adjustment'] = hist_a2_df['Projected Average'] - hist_a2_df['Historical Average']
    hist_a2_df['Smoothed Adjustment'] = savgol_filter(hist_a2_df['Adjustment'], 25, 2)
    # print(hist_a2_df.head(750).to_string())

    projection_df['SECONDARY ASSISTS'] = (projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP']
    projection_df = projection_df.sort_values('SECONDARY ASSISTS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_a2_df['Smoothed Adjustment']/((projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        projection_df['EV A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['SECONDARY ASSISTS'] = (projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP']
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

    # Download file
    if download_file == True:
        filename = f'svr_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df.fillna(0)

def make_projections(existing_stat_df=True, existing_partial_projections=True, year=2024, download_csv=False):
    stat_df = preprocessing_training_functions.scrape_player_statistics(existing_stat_df)

    if existing_partial_projections == False:
        projection_df = preprocessing_training_functions.make_projection_df(stat_df, year)
    else:
        projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/svr_partial_projections_{year}.csv")
        projection_df = projection_df.drop(projection_df.columns[0], axis=1)

    projection_df = make_forward_gp_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_gp_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_ev_atoi_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_ev_atoi_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pp_atoi_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_pp_atoi_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pk_atoi_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_pk_atoi_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_ev_gper60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_ev_gper60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pp_gper60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_pp_gper60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pk_gper60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_pk_gper60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_ev_a1per60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_ev_a1per60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pp_a1per60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_pp_a1per60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pk_a1per60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_pk_a1per60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_ev_a2per60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_ev_a2per60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pp_a2per60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_pp_a2per60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pk_a2per60_projections(stat_df, projection_df, True, year)
    projection_df = make_defence_pk_a2per60_projections(stat_df, projection_df, True, year)

    projection_df = goal_era_adjustment(stat_df, projection_df, year, True, False)
    projection_df = a1_era_adjustment(stat_df, projection_df, year, True, False)
    projection_df = a2_era_adjustment(stat_df, projection_df, year, True, False)
    projection_df['POINTS'] = projection_df['GOALS'] + projection_df['PRIMARY ASSISTS'] + projection_df['SECONDARY ASSISTS']

    projection_df = projection_df.sort_values('POINTS', ascending=False)
    # projection_df = projection_df.sort_values('EV ATOI', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    print(projection_df.head(20))

    if download_csv == True:
        filename = f'svr_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

# make_projections(True, False, 2024, True)
