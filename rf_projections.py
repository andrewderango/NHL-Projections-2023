import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import preprocessing_training_functions

def make_forward_gp_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 4, None, 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'GP', 4, None, None, 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 3, None, 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 3, None, None, 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 2, None, 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'GP', 2, None, None, 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 1, None, 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'GP', 1, None, None, 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, 82))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, 82))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, 82))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, 82))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'GP'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        print(player_name, mean, stdev)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        print(player_name, mean, stdev)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        print(player_name, mean, stdev)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        print(player_name, mean, stdev)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_gp_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 4, None, 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'GP', 4, None, None, 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 3, None, 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 3, None, None, 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 2, None, 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'GP', 2, None, None, 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 1, None, 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'GP', 1, None, None, 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, 82))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, 82))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, 82))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, 82))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'GP'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        print(player_name, mean, stdev)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        print(player_name, mean, stdev)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        print(player_name, mean, stdev)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        print(player_name, mean, stdev)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_ev_atoi_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'EV', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'EV', None, 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'EV', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV', None, 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'EV', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'EV', None, 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'EV', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'EV', None, 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_ev_atoi_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'EV', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'EV', None, 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'EV', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV', None, 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'EV', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'EV', None, 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'EV', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'EV', None, 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_pp_atoi_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'PP', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PP', None, 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'PP', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP', None, 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'PP', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PP', None, 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'PP', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PP', None, 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_pp_atoi_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'PP', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PP', None, 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'PP', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP', None, 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'PP', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PP', None, 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'PP', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PP', None, 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_pk_atoi_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'PK', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PK', None, 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'PK', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK', None, 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'PK', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PK', None, 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'PK', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PK', None, 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_pk_atoi_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'PK', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PK', None, 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'PK', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK', None, 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'PK', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PK', None, 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'PK', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PK', None, 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_ev_gper60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'EV', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'EV', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'EV', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'EV', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'EV G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_ev_gper60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'EV', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'EV', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'EV', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'EV', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'EV G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_pp_gper60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'PP', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PP', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PP', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'PP', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PP', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'PP', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PP', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PP G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_pp_gper60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'PP', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PP', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'PP', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'PP', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PP', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'PP', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PP', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PP G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_pk_gper60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'PK', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PK', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PK', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'PK', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PK', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'PK', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PK', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PK G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_pk_gper60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'PK', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PK', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'PK', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'PK', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PK', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'PK', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PK', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PK G/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_ev_a1per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'EV', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'EV', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'EV', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'EV', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'EV', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'EV', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'EV', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'EV', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'EV A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_ev_a1per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'EV', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'EV', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'EV', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'EV', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'EV', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'EV', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'EV', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'EV', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'EV A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_pp_a1per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'PP', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PP', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'PP', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PP', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'PP', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PP', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'PP', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PP', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PP A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_pp_a1per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'PP', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PP', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'PP', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PP', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'PP', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PP', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'PP', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PP', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PP A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_pk_a1per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'PK', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PK', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'PK', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PK', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'PK', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PK', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'PK', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PK', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PK A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_pk_a1per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'PK', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PK', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'PK', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PK', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'PK', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PK', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'PK', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PK', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PK A1/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_ev_a2per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'EV', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'EV', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'EV', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'EV', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'EV', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'EV', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'EV', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'EV', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'EV A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_ev_a2per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'EV', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'EV', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'EV', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'EV', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'EV', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'EV', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'EV', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'EV', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'EV A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_pp_a2per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'PP', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PP', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'PP', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PP', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'PP', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PP', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'PP', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PP', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PP A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_pp_a2per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'PP', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PP', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'PP', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PP', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'PP', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PP', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'PP', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PP', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PP A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_forward_pk_a2per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'PK', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PK', 'forward', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'PK', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PK', 'forward', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'PK', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PK', 'forward', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'PK', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PK', 'forward', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PK A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_defence_pk_a2per60_projections(stat_df, projection_df, stdev_df, download_file, sim_count, year=2024):
    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'PK', 'RF')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PK', 'defence', 'RF')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'PK', 'RF')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PK', 'defence', 'RF')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'PK', 'RF')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PK', 'defence', 'RF')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'PK', 'RF')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PK', 'defence', 'RF')

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
        
    n_bootstrap_models = 100
    yr4_predictions, yr3_predictions, yr2_predictions, yr1_predictions = [], [], [], []
    for _ in range(n_bootstrap_models):
        y4_bootstrap_indices = np.random.choice(len(X_4), size=len(X_4), replace=True)
        y4_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y4_rf_regressor.fit(X_4[y4_bootstrap_indices], y_4[y4_bootstrap_indices])
        yr4_bootstrap_pred = y4_rf_regressor.predict(yr4_stat_list)
        yr4_predictions.append(np.clip(yr4_bootstrap_pred, 0, None))

        y3_bootstrap_indices = np.random.choice(len(X_3), size=len(X_3), replace=True)
        y3_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y3_rf_regressor.fit(X_3[y3_bootstrap_indices], y_3[y3_bootstrap_indices])
        yr3_bootstrap_pred = y3_rf_regressor.predict(yr3_stat_list)
        yr3_predictions.append(np.clip(yr3_bootstrap_pred, 0, None))

        y2_bootstrap_indices = np.random.choice(len(X_2), size=len(X_2), replace=True)
        y2_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y2_rf_regressor.fit(X_2[y2_bootstrap_indices], y_2[y2_bootstrap_indices])
        yr2_bootstrap_pred = y2_rf_regressor.predict(yr2_stat_list)
        yr2_predictions.append(np.clip(yr2_bootstrap_pred, 0, None))

        y1_bootstrap_indices = np.random.choice(len(X_1), size=len(X_1), replace=True)
        y1_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        y1_rf_regressor.fit(X_1[y1_bootstrap_indices], y_1[y1_bootstrap_indices])
        yr1_bootstrap_pred = y1_rf_regressor.predict(yr1_stat_list)
        yr1_predictions.append(np.clip(yr1_bootstrap_pred, 0, None))

    yr4_predictions = np.array(yr4_predictions)
    yr4_mean_predictions = np.mean(yr4_predictions, axis=0)
    yr4_stdev_predictions = np.std(yr4_predictions, axis=0)
    yr3_predictions = np.array(yr3_predictions)
    yr3_mean_predictions = np.mean(yr3_predictions, axis=0)
    yr3_stdev_predictions = np.std(yr3_predictions, axis=0)
    yr2_predictions = np.array(yr2_predictions)
    yr2_mean_predictions = np.mean(yr2_predictions, axis=0)
    yr2_stdev_predictions = np.std(yr2_predictions, axis=0)
    yr1_predictions = np.array(yr1_predictions)
    yr1_mean_predictions = np.mean(yr1_predictions, axis=0)
    yr1_stdev_predictions = np.std(yr1_predictions, axis=0)

    column_name = 'PK A2/60'

    for index, statline in enumerate(yr4_stat_list):
        player_name = yr4_group[index]
        mean = yr4_mean_predictions[index]
        stdev = yr4_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        mean = yr3_mean_predictions[index]
        stdev = yr3_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        mean = yr2_mean_predictions[index]
        stdev = yr2_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        mean = yr1_mean_predictions[index]
        stdev = yr1_stdev_predictions[index]

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = mean
            stdev_df.loc[stdev_df['Player'] == player_name, column_name] = stdev
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [mean]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [stdev]})
            stdev_df = pd.concat([stdev_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rf_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_partial_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, stdev_df

def make_projections(existing_stat_df=True, existing_partial_projections=True, year=2024, download_csv=False):
    stat_df = preprocessing_training_functions.scrape_player_statistics(existing_stat_df)

    if existing_partial_projections == False:
        projection_df = preprocessing_training_functions.make_projection_df(stat_df)
        stdev_df = preprocessing_training_functions.make_projection_df(stat_df)
    else:
        projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/rf_partial_projections_{year}.csv")
        projection_df = projection_df.drop(projection_df.columns[0], axis=1)
        stdev_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/rf_partial_stdeviations_{year}.csv")
        stdev_df = stdev_df.drop(stdev_df.columns[0], axis=1)

    # projection_df, stdev_df = make_forward_gp_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_gp_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_ev_atoi_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_ev_atoi_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_pp_atoi_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_pp_atoi_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_pk_atoi_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_pk_atoi_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_ev_gper60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_ev_gper60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_pp_gper60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_pp_gper60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_pk_gper60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_pk_gper60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_ev_a1per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_ev_a1per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_pp_a1per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_pp_a1per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_pk_a1per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_pk_a1per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_ev_a2per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_ev_a2per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_pp_a2per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_pp_a2per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_forward_pk_a2per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)
    # projection_df, stdev_df = make_defence_pk_a2per60_projections(stat_df, projection_df, stdev_df, True, 1000, year)

    projection_df = projection_df.sort_values('EV G/60', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    print(projection_df)
    print(stdev_df)

    if download_csv == True:
        filename = f'rf_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'rf_final_stdeviations_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stdev_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

make_projections(True, True, 2024, False)

### add more features to GP projections for random forest.
