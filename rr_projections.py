import os
import numpy as np
import pandas as pd
from datetime import date
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import preprocessing_training_functions

def calc_age(dob, year):
    dob = date(int(dob.split('-')[0]), int(dob.split('-')[1]), int(dob.split('-')[2])) #year, month, date
    target_date = date(year, 10, 1) # Age calculated as of October 1st of the season.
    delta_days = target_date - dob
    age = round(delta_days.days/365.24,3)
    return age

def fourth_degree_polynomial(x, coef4, coef3, coef2, coef1, coef0):
    return coef4*x**4 + coef3*x**3 + coef2*x**2 + coef1*x + coef0

def make_forward_gp_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 1, None, 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 GP']-agecurve_df['Y4 GP'])
    age_curve_max = np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[np.argmin(np.abs(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]]) - 25))]
    root_index = 0
    while np.iscomplex(age_curve_max) is True:
        if root_index >= 10:
            print('No real roots')
            return
        else:
            age_curve_max = np.real(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[root_index])
            root_index += 1
    age_curve_max = np.real(age_curve_max)
    coef4 = agecurve_parameters[0]/4
    coef3 = agecurve_parameters[1]/3
    coef2 = agecurve_parameters[2]/2
    coef1 = agecurve_parameters[3]/1
    coef0 = -fourth_degree_polynomial(age_curve_max, coef4, coef3, coef2, coef1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 3, None, 'RR')
    instance_df_y3['Y1 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y2 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y3 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y4 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y5 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4, coef3, coef2, coef1, coef0)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 3, None, None, 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 2, None, None, 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 1, None, None, 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr1_group:
        yr1_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'GP'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_gp_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 1, None, 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 GP']-agecurve_df['Y4 GP'])
    age_curve_max = np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[np.argmin(np.abs(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]]) - 25))]
    root_index = 0
    while np.iscomplex(age_curve_max) is True:
        if root_index >= 10:
            print('No real roots')
            return
        else:
            age_curve_max = np.real(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[root_index])
            root_index += 1
    age_curve_max = np.real(age_curve_max)
    coef4 = agecurve_parameters[0]/4
    coef3 = agecurve_parameters[1]/3
    coef2 = agecurve_parameters[2]/2
    coef1 = agecurve_parameters[3]/1
    coef0 = -fourth_degree_polynomial(age_curve_max, coef4, coef3, coef2, coef1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 3, None, 'RR')

    instance_df_y3['Y1 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y2 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y3 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y4 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y5 GP'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4, coef3, coef2, coef1, coef0)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 3, None, None, 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 2, None, None, 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 1, None, None, 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr1_group:
        yr1_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'GP'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_ev_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'EV', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV ATOI']-agecurve_df['Y4 EV ATOI'])
    age_curve_max = np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[np.argmin(np.abs(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]]) - 25))]
    root_index = 0
    while np.iscomplex(age_curve_max) is True:
        if root_index >= 10:
            print('No real roots')
            return
        else:
            age_curve_max = np.real(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[root_index])
            root_index += 1
    age_curve_max = np.real(age_curve_max)
    coef4 = agecurve_parameters[0]/4
    coef3 = agecurve_parameters[1]/3
    coef2 = agecurve_parameters[2]/2
    coef1 = agecurve_parameters[3]/1
    coef0 = -fourth_degree_polynomial(age_curve_max, coef4, coef3, coef2, coef1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'EV', 'RR')
    instance_df_y3['Y1 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y2 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y3 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y4 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y5 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4, coef3, coef2, coef1, coef0)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV', None, 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 2, 'EV', None, 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 1, 'EV', None, 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    # for player in yr1_group:
    #     yr1_stat_list.append([
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
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
            negative_first_z_score = max(instance_df_y3['Y4 EV ATOI'].mean() - instance_df_y3['Y4 EV ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 45-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_ev_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'EV', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV ATOI']-agecurve_df['Y4 EV ATOI'])
    age_curve_max = np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[np.argmin(np.abs(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]]) - 25))]
    root_index = 0
    while np.iscomplex(age_curve_max) is True:
        if root_index >= 10:
            print('No real roots')
            return
        else:
            age_curve_max = np.real(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[root_index])
            root_index += 1
    age_curve_max = np.real(age_curve_max)
    coef4 = agecurve_parameters[0]/4
    coef3 = agecurve_parameters[1]/3
    coef2 = agecurve_parameters[2]/2
    coef1 = agecurve_parameters[3]/1
    coef0 = -fourth_degree_polynomial(age_curve_max, coef4, coef3, coef2, coef1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'EV', 'RR')
    instance_df_y3['Y1 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y2 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y3 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y4 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y5 EV ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4, coef3, coef2, coef1, coef0)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV', None, 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 2, 'EV', None, 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 1, 'EV', None, 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    # for player in yr1_group:
    #     yr1_stat_list.append([
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
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
            negative_first_z_score = max(instance_df_y3['Y4 EV ATOI'].mean() - instance_df_y3['Y4 EV ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 45-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pp_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'PP', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP ATOI'].fillna(0)-agecurve_df['Y4 PP ATOI'].fillna(0))
    age_curve_max = np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[np.argmin(np.abs(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]]) - 25))]
    root_index = 0
    while np.iscomplex(age_curve_max) is True:
        if root_index >= 10:
            print('No real roots')
            return
        else:
            age_curve_max = np.real(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[root_index])
            root_index += 1
    age_curve_max = np.real(age_curve_max)
    coef4 = agecurve_parameters[0]/4
    coef3 = agecurve_parameters[1]/3
    coef2 = agecurve_parameters[2]/2
    coef1 = agecurve_parameters[3]/1
    coef0 = -fourth_degree_polynomial(age_curve_max, coef4, coef3, coef2, coef1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y2 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y3 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y4 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y5 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4, coef3, coef2, coef1, coef0)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP', None, 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 2, 'PP', None, 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 1, 'PP', None, 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]

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
            negative_first_z_score = max(instance_df_y3['Y4 PP ATOI'].mean() - instance_df_y3['Y4 PP ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 45-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat])
        
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'PP', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP ATOI'].fillna(0)-agecurve_df['Y4 PP ATOI'].fillna(0))
    age_curve_max = np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[np.argmin(np.abs(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]]) - 25))]
    root_index = 0
    while np.iscomplex(age_curve_max) is True:
        if root_index >= 10:
            print('No real roots')
            return
        else:
            age_curve_max = np.real(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[root_index])
            root_index += 1
    age_curve_max = np.real(age_curve_max)
    coef4 = agecurve_parameters[0]/4
    coef3 = agecurve_parameters[1]/3
    coef2 = agecurve_parameters[2]/2
    coef1 = agecurve_parameters[3]/1
    coef0 = -fourth_degree_polynomial(age_curve_max, coef4, coef3, coef2, coef1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y2 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y3 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y4 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y5 PP ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4, coef3, coef2, coef1, coef0)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP', None, 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 2, 'PP', None, 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 1, 'PP', None, 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0]

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
            negative_first_z_score = max(instance_df_y3['Y4 PP ATOI'].mean() - instance_df_y3['Y4 PP ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 45-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat])
        
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'PK', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PK ATOI'].fillna(0)-agecurve_df['Y4 PK ATOI'].fillna(0))
    age_curve_max = np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[np.argmin(np.abs(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]]) - 25))]
    root_index = 0
    while np.iscomplex(age_curve_max) is True:
        if root_index >= 10:
            print('No real roots')
            return
        else:
            age_curve_max = np.real(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[root_index])
            root_index += 1
    age_curve_max = np.real(age_curve_max)
    coef4 = agecurve_parameters[0]/4
    coef3 = agecurve_parameters[1]/3
    coef2 = agecurve_parameters[2]/2
    coef1 = agecurve_parameters[3]/1
    coef0 = -fourth_degree_polynomial(age_curve_max, coef4, coef3, coef2, coef1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'PK', 'RR')
    instance_df_y3['Y1 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y2 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y3 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y4 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y5 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4, coef3, coef2, coef1, coef0)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK', None, 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 2, 'PK', None, 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 1, 'PK', None, 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]

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
            negative_first_z_score = max(instance_df_y3['Y4 PK ATOI'].mean() - instance_df_y3['Y4 PK ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 45-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat])
        
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_atoi_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'PK', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PK ATOI'].fillna(0)-agecurve_df['Y4 PK ATOI'].fillna(0))
    age_curve_max = np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[np.argmin(np.abs(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]]) - 25))]
    root_index = 0
    while np.iscomplex(age_curve_max) is True:
        if root_index >= 10:
            print('No real roots')
            return
        else:
            age_curve_max = np.real(np.roots([agecurve_parameters[0], agecurve_parameters[1], agecurve_parameters[2], agecurve_parameters[3]])[root_index])
            root_index += 1
    age_curve_max = np.real(age_curve_max)
    coef4 = agecurve_parameters[0]/4
    coef3 = agecurve_parameters[1]/3
    coef2 = agecurve_parameters[2]/2
    coef1 = agecurve_parameters[3]/1
    coef0 = -fourth_degree_polynomial(age_curve_max, coef4, coef3, coef2, coef1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'PK', 'RR')
    instance_df_y3['Y1 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y2 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y3 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y4 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4, coef3, coef2, coef1, coef0)
    instance_df_y3['Y5 PK ATOI'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4, coef3, coef2, coef1, coef0)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK', None, 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 2, 'PK', None, 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 1, 'PK', None, 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], year-1)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr1_group:
        y1_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0])
        y2_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0])
        y3_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0])
        y4_gp = int(stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0])
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0]

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
            negative_first_z_score = max(instance_df_y3['Y4 PK ATOI'].mean() - instance_df_y3['Y4 PK ATOI'].std(), 0) # should not be negative
            games_to_pseudofy = 45-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat = (y1_stat*y1_gp + y2_stat*y2_gp + y3_stat*y3_gp + y4_stat*y4_gp + negative_first_z_score*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat])
        
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_ev_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'EV', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)
    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV G/60']-agecurve_df['Y4 EV G/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV ixG/60']-agecurve_df['Y4 EV ixG/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'EV', 'RR')
    instance_df_y3['Y1 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 2, 'EV', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 1, 'EV', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 EV G/60'].mean() - instance_df_y3['Y4 EV G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 EV ixG/60'].mean() - instance_df_y3['Y4 EV ixG/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'EV G/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_ev_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'EV', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)
    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV G/60']-agecurve_df['Y4 EV G/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV ixG/60']-agecurve_df['Y4 EV ixG/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'EV', 'RR')
    instance_df_y3['Y1 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 2, 'EV', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 1, 'EV', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 EV G/60'].mean() - instance_df_y3['Y4 EV G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 EV ixG/60'].mean() - instance_df_y3['Y4 EV ixG/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'EV G/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pp_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'PP', 'RR')
    agecurve_df[['Y5 PP G/60', 'Y5 PP ixG/60', 'Y4 PP G/60', 'Y4 PP ixG/60']] = agecurve_df[['Y5 PP G/60', 'Y5 PP ixG/60', 'Y4 PP G/60', 'Y4 PP ixG/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP G/60']-agecurve_df['Y4 PP G/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP ixG/60']-agecurve_df['Y4 PP ixG/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 2, 'PP', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 1, 'PP', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    # X_3_scaled = X_3
    # X_2_scaled = X_2
    # X_1_scaled = X_1

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0]
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PP G/60'].mean() - instance_df_y3['Y4 PP G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 PP ixG/60'].mean() - instance_df_y3['Y4 PP ixG/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_3 = max(instance_df_y3['Y4 EV G/60'].mean() - instance_df_y3['Y4 EV G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_4 = max(instance_df_y3['Y4 EV ixG/60'].mean() - instance_df_y3['Y4 EV ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])
        
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    # yr3_stat_list_scaled = yr3_stat_list
    # yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    # yr2_stat_list_scaled = yr2_stat_list
    # yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    # yr1_stat_list_scaled = yr1_stat_list
    # yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PP G/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'PP', 'RR')
    agecurve_df[['Y5 PP G/60', 'Y5 PP ixG/60', 'Y4 PP G/60', 'Y4 PP ixG/60']] = agecurve_df[['Y5 PP G/60', 'Y5 PP ixG/60', 'Y4 PP G/60', 'Y4 PP ixG/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP G/60']-agecurve_df['Y4 PP G/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP G/60']-agecurve_df['Y4 PP G/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 2, 'PP', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 1, 'PP', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    # X_3_scaled = X_3
    # X_2_scaled = X_2
    # X_1_scaled = X_1

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP G/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ixG/60'].fillna(0).iloc[0]
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PP G/60'].mean() - instance_df_y3['Y4 PP G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 PP ixG/60'].mean() - instance_df_y3['Y4 PP ixG/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_3 = max(instance_df_y3['Y4 EV G/60'].mean() - instance_df_y3['Y4 EV G/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_4 = max(instance_df_y3['Y4 EV ixG/60'].mean() - instance_df_y3['Y4 EV ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_3 = (y1_stat_3*y1_pptoi + y2_stat_3*y2_pptoi + y3_stat_3*y3_pptoi + y4_stat_3*y4_pptoi + negative_first_z_score_stat_3*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_4 = (y1_stat_4*y1_pptoi + y2_stat_4*y2_pptoi + y3_stat_4*y3_pptoi + y4_stat_4*y4_pptoi + negative_first_z_score_stat_4*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])
        
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    # yr3_stat_list_scaled = yr3_stat_list
    # yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    # yr2_stat_list_scaled = yr2_stat_list
    # yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    # yr1_stat_list_scaled = yr1_stat_list
    # yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PP G/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'PK', 'RR')
    agecurve_df[['Y5 PK G/60', 'Y5 PK ixG/60', 'Y4 PK G/60', 'Y4 PK ixG/60']] = agecurve_df[['Y5 PK G/60', 'Y5 PK ixG/60', 'Y4 PK G/60', 'Y4 PK ixG/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PK G/60']-agecurve_df['Y4 PK G/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PK', 'RR')
    instance_df_y3['Y1 PK ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PK ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PK ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PK ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PK G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 2, 'PK', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 1, 'PK', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PK ixG/60'].mean() - instance_df_y3['Y4 PK ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat
            ])
        
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    # yr3_stat_list_scaled = yr3_stat_list
    # yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    # yr2_stat_list_scaled = yr2_stat_list
    # yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    # yr1_stat_list_scaled = yr1_stat_list
    # yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PK G/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_gper60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'PK', 'RR')
    agecurve_df[['Y5 PK G/60', 'Y5 PK ixG/60', 'Y4 PK G/60', 'Y4 PK ixG/60']] = agecurve_df[['Y5 PK G/60', 'Y5 PK ixG/60', 'Y4 PK G/60', 'Y4 PK ixG/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PK G/60']-agecurve_df['Y4 PK G/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'PK', 'RR')
    instance_df_y3['Y1 PK ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PK ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PK ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PK ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PK G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 2, 'PK', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 1, 'PK', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ixG/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ixG/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PK ixG/60'].mean() - instance_df_y3['Y4 PK ixG/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat
            ])
        
    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    # yr3_stat_list_scaled = yr3_stat_list
    # yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    # yr2_stat_list_scaled = yr2_stat_list
    # yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    # yr1_stat_list_scaled = yr1_stat_list
    # yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PK G/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_ev_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'EV', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)
    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV A1/60']-agecurve_df['Y4 EV A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV A2/60']-agecurve_df['Y4 EV A2/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'EV', 'RR')
    instance_df_y3['Y1 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'EV', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 2, 'EV', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 1, 'EV', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 EV A1/60'].mean() - instance_df_y3['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 EV A2/60'].mean() - instance_df_y3['Y4 EV A2/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'EV A1/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_ev_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'EV', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)
    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV A1/60']-agecurve_df['Y4 EV A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV A2/60']-agecurve_df['Y4 EV A2/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'EV', 'RR')
    instance_df_y3['Y1 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'EV', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 2, 'EV', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 1, 'EV', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 EV A1/60'].mean() - instance_df_y3['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 EV A2/60'].mean() - instance_df_y3['Y4 EV A2/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'EV A1/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pp_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'PP', 'RR')
    agecurve_df[['Y5 PP A1/60', 'Y5 PP A2/60', 'Y4 PP A1/60', 'Y4 PP A2/60']] = agecurve_df[['Y5 PP A1/60', 'Y5 PP A2/60', 'Y4 PP A1/60', 'Y4 PP A2/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP A1/60']-agecurve_df['Y4 PP A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP A2/60']-agecurve_df['Y4 PP A2/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PP', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 2, 'PP', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 1, 'PP', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
            ])
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPA1/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PP A1/60'].mean() - instance_df_y3['Y4 PP A1/60'].std(), 0)
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 PP A2/60'].mean() - instance_df_y3['Y4 PP A2/60'].std(), 0)
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PP A1/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'PP', 'RR')
    agecurve_df[['Y5 PP A1/60', 'Y5 PP A2/60', 'Y4 PP A1/60', 'Y4 PP A2/60']] = agecurve_df[['Y5 PP A1/60', 'Y5 PP A2/60', 'Y4 PP A1/60', 'Y4 PP A2/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP A1/60']-agecurve_df['Y4 PP A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP A2/60']-agecurve_df['Y4 PP A2/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PP', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 2, 'PP', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 1, 'PP', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
            ])
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPA1/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PP A1/60'].mean() - instance_df_y3['Y4 PP A1/60'].std(), 0)
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 PP A2/60'].mean() - instance_df_y3['Y4 PP A2/60'].std(), 0)
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PP A1/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'PK', 'RR')
    agecurve_df[['Y5 PK A1/60', 'Y4 PK A1/60']] = agecurve_df[['Y5 PK A1/60', 'Y4 PK A1/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PK A1/60']-agecurve_df['Y4 PK A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'PK', 'RR')
    instance_df_y3['Y1 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PK', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 2, 'PK', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 1, 'PK', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PK A1/60'].mean() - instance_df_y3['Y4 PK A1/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat
            ])

    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PK A1/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_a1per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'PK', 'RR')
    agecurve_df[['Y5 PK A1/60', 'Y5 PK A2/60', 'Y4 PK A1/60', 'Y4 PK A2/60']] = agecurve_df[['Y5 PK A1/60', 'Y5 PK A2/60', 'Y4 PK A1/60', 'Y4 PK A2/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PK A1/60']-agecurve_df['Y4 PK A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'PK', 'RR')
    instance_df_y3['Y1 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PK A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PK', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 2, 'PK', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 1, 'PK', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A1/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PK A1/60'].mean() - instance_df_y3['Y4 PK A1/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat
            ])

    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PK A1/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_ev_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'EV', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)
    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV A1/60']-agecurve_df['Y4 EV A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV A2/60']-agecurve_df['Y4 EV A2/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'EV', 'RR')
    instance_df_y3['Y1 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'EV', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 2, 'EV', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 1, 'EV', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 EV A1/60'].mean() - instance_df_y3['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 EV A2/60'].mean() - instance_df_y3['Y4 EV A2/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'EV A2/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_ev_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'EV', 'RR')
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)
    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV A1/60']-agecurve_df['Y4 EV A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV A2/60']-agecurve_df['Y4 EV A2/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'EV', 'RR')
    instance_df_y3['Y1 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 EV A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 EV A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'EV', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 2, 'EV', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 1, 'EV', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 40 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 40 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 40:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV A2/60'].fillna(0).iloc[0]
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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 EV A1/60'].mean() - instance_df_y3['Y4 EV A1/60'].std(), 0) # should not be negative
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 EV A2/60'].mean() - instance_df_y3['Y4 EV A2/60'].std(), 0)
            games_to_pseudofy = 50-(y1_gp + y2_gp + y3_gp + y4_gp)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_gp + y2_stat_1*y2_gp + y3_stat_1*y3_gp + y4_stat_1*y4_gp + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_gp + y2_stat_2*y2_gp + y3_stat_2*y3_gp + y4_stat_2*y4_gp + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_gp + y2_gp + y3_gp + y4_gp + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'EV A2/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pp_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'PP', 'RR')
    agecurve_df[['Y5 PP A1/60', 'Y5 PP A2/60', 'Y4 PP A1/60', 'Y4 PP A2/60']] = agecurve_df[['Y5 PP A1/60', 'Y5 PP A2/60', 'Y4 PP A1/60', 'Y4 PP A2/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP A1/60']-agecurve_df['Y4 PP A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP A2/60']-agecurve_df['Y4 PP A2/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PP', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 2, 'PP', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 1, 'PP', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
            ])
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPA1/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PP A1/60'].mean() - instance_df_y3['Y4 PP A1/60'].std(), 0)
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 PP A2/60'].mean() - instance_df_y3['Y4 PP A2/60'].std(), 0)
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PP A2/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pp_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'PP', 'RR')
    agecurve_df[['Y5 PP A1/60', 'Y5 PP A2/60', 'Y4 PP A1/60', 'Y4 PP A2/60']] = agecurve_df[['Y5 PP A1/60', 'Y5 PP A2/60', 'Y4 PP A1/60', 'Y4 PP A2/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP A1/60']-agecurve_df['Y4 PP A1/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    agecurve_parameters_2, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PP A2/60']-agecurve_df['Y4 PP A2/60'])
    age_curve_max_2 = np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]]) - 25))]
    root_index_2 = 0
    while np.iscomplex(age_curve_max_2) is True:
        if root_index_2 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_2 = np.real(np.roots([agecurve_parameters_2[0], agecurve_parameters_2[1], agecurve_parameters_2[2], agecurve_parameters_2[3]])[root_index_2])
            root_index_2 += 1
    age_curve_max_2 = np.real(age_curve_max_2)
    coef4_2 = agecurve_parameters_2[0]/4
    coef3_2 = agecurve_parameters_2[1]/3
    coef2_2 = agecurve_parameters_2[2]/2
    coef1_2 = agecurve_parameters_2[3]/1
    coef0_2 = -fourth_degree_polynomial(age_curve_max_2, coef4_2, coef3_2, coef2_2, coef1_2, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PP A1/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 PP A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PP', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 2, 'PP', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 1, 'PP', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PP ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PP ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PP ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
            ])
    for player in yr1_group:
        y1_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pptoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A1/60'].fillna(0).iloc[0]
        y2_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A1/60'].fillna(0).iloc[0]
        y3_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A1/60'].fillna(0).iloc[0]
        y4_stat_1 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A1/60'].fillna(0).iloc[0]
        y1_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-1} PP A2/60'].fillna(0).iloc[0]
        y2_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-2} PP A2/60'].fillna(0).iloc[0]
        y3_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-3} PP A2/60'].fillna(0).iloc[0]
        y4_stat_2 = stat_df.loc[stat_df['Player'] == player, f'{year-4} PP A2/60'].fillna(0).iloc[0]

        # Keep getting games from previous seasons until you reach threshold of 150 PPTOI.
        # Once you reach 150 PPTOI, find the PPA1/60 accross these seasons.
        # If they haven't played 150 PPTOI in their past 4 seasons, fill the rest of the 150 PPTOI with the -1st z-score of the stat.
        if y1_pptoi >= 150:
            pseudo_prev_year_stat_1 = y1_stat_1
            pseudo_prev_year_stat_2 = y1_stat_2
        elif y1_pptoi + y2_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi)/(y1_pptoi + y2_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi)/(y1_pptoi + y2_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi)
        elif y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi >= 150:
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
        else:
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PP A1/60'].mean() - instance_df_y3['Y4 PP A1/60'].std(), 0)
            negative_first_z_score_stat_2 = max(instance_df_y3['Y4 PP A2/60'].mean() - instance_df_y3['Y4 PP A2/60'].std(), 0)
            games_to_pseudofy = 150-(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi)
            pseudo_prev_year_stat_1 = (y1_stat_1*y1_pptoi + y2_stat_1*y2_pptoi + y3_stat_1*y3_pptoi + y4_stat_1*y4_pptoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)
            pseudo_prev_year_stat_2 = (y1_stat_2*y1_pptoi + y2_stat_2*y2_pptoi + y3_stat_2*y3_pptoi + y4_stat_2*y4_pptoi + negative_first_z_score_stat_2*games_to_pseudofy)/(y1_pptoi + y2_pptoi + y3_pptoi + y4_pptoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PP A2/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_forward_pk_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'PK', 'RR')
    agecurve_df[['Y5 PK A2/60', 'Y4 PK A2/60']] = agecurve_df[['Y5 PK A2/60', 'Y4 PK A2/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PK A2/60']-agecurve_df['Y4 PK A2/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'PK', 'RR')
    instance_df_y3['Y1 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PK', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 2, 'PK', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 1, 'PK', 'forward', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
            ])
    for player in yr1_group:
        y1_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A2/60'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A2/60'].fillna(0).iloc[0]

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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PK A2/60'].mean() - instance_df_y3['Y4 PK A2/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat
            ])

    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PK A2/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_defence_pk_a2per60_projections(stat_df, projection_df, download_file=False, year=2024):
    agecurve_df, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'PK', 'RR')
    agecurve_df[['Y5 PK A2/60', 'Y4 PK A2/60']] = agecurve_df[['Y5 PK A2/60', 'Y4 PK A2/60']].fillna(0)
    agecurve_df = agecurve_df.drop(agecurve_df[agecurve_df['Age'] < 18].index)

    agecurve_parameters_1, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 PK A2/60']-agecurve_df['Y4 PK A2/60'])
    age_curve_max_1 = np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]]) - 25))]
    root_index_1 = 0
    while np.iscomplex(age_curve_max_1) is True:
        if root_index_1 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_1 = np.real(np.roots([agecurve_parameters_1[0], agecurve_parameters_1[1], agecurve_parameters_1[2], agecurve_parameters_1[3]])[root_index_1])
            root_index_1 += 1
    age_curve_max_1 = np.real(age_curve_max_1)
    coef4_1 = agecurve_parameters_1[0]/4
    coef3_1 = agecurve_parameters_1[1]/3
    coef2_1 = agecurve_parameters_1[2]/2
    coef1_1 = agecurve_parameters_1[3]/1
    coef0_1 = -fourth_degree_polynomial(age_curve_max_1, coef4_1, coef3_1, coef2_1, coef1_1, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'PK', 'RR')
    instance_df_y3['Y1 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PK A2/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PK', 'defence', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 2, 'PK', 'defence', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 1, 'PK', 'defence', 'RR')

    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)

    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)

    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    y3_ridge_model = Ridge(alpha=0.05)
    y2_ridge_model = Ridge(alpha=0.05)
    y1_ridge_model = Ridge(alpha=0.05)

    y3_ridge_model.fit(X_3_scaled, y_3)
    y2_ridge_model.fit(X_2_scaled, y_2)
    y1_ridge_model.fit(X_1_scaled, y_1)

    yr3_group, yr2_group, yr1_group = [], [], []
        
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
            if row[f'{year-3} PK ATOI']*row[f'{year-3} GP'] >= 50 and row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} PK ATOI']*row[f'{year-2} GP'] >= 50 and row[f'{year-1} PK ATOI']*row[f'{year-1} GP'] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], []

    for player in yr3_group:
        yr3_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
            ])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
            ])
    for player in yr1_group:
        y1_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].fillna(0).iloc[0]
        y2_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].fillna(0).iloc[0]
        y3_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].fillna(0).iloc[0]
        y4_pktoi = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK ATOI'].fillna(0).iloc[0] * stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].fillna(0).iloc[0]
        y1_stat = stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
        y2_stat = stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0]
        y3_stat = stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A2/60'].fillna(0).iloc[0]
        y4_stat = stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A2/60'].fillna(0).iloc[0]

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
            negative_first_z_score_stat_1 = max(instance_df_y3['Y4 PK A2/60'].mean() - instance_df_y3['Y4 PK A2/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            pseudo_prev_year_stat
            ])

    yr3_stat_list = np.where(np.isnan(yr3_stat_list), 0, yr3_stat_list)
    yr2_stat_list = np.where(np.isnan(yr2_stat_list), 0, yr2_stat_list)
    yr1_stat_list = np.where(np.isnan(yr1_stat_list), 0, yr1_stat_list)

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'PK A2/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], year-1), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'rr_partial_projections_{year}'
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
        projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/rr_partial_projections_{year}.csv")
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
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    print(projection_df.head(20))
    # print(projection_df.to_string())

    if download_csv == True:
        filename = f'rr_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

# make_projections(True, False, 2024, True)
