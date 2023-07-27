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

    projection_df, stdev_df = make_forward_gp_projections(stat_df, projection_df, stdev_df, True, 1000, year)

    projection_df = projection_df.sort_values('GP', ascending=False)
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

make_projections(True, False, 2024, False)
# make_projections(True, True, 2024, False)

### add more features to GP projections for random forest.
