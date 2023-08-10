import os
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr1_group:
        yr1_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'GP'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, 82)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr1_group:
        yr1_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])

    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = y3_ridge_model.predict(yr3_stat_list_scaled)

    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = y2_ridge_model.predict(yr2_stat_list_scaled)

    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = y1_ridge_model.predict(yr1_stat_list_scaled)

    column_name = 'GP'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, 82)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, 82)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    # for player in yr1_group:
    #     yr1_stat_list.append([
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
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
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    # for player in yr1_group:
    #     yr1_stat_list.append([
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
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
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
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
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PP ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PP ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PP ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
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
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
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
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK ATOI'].iloc[0]*gp_adjustment_factor[year-3] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-3, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
    for player in yr2_group:
        yr2_stat_list.append([
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK ATOI'].iloc[0]*gp_adjustment_factor[year-2] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-2, coef4, coef3, coef2, coef1, coef0),
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK ATOI'].iloc[0]*gp_adjustment_factor[year-1] - fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player, f'Date of Birth'].iloc[0], 2023)-1, coef4, coef3, coef2, coef1, coef0)])
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
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4, coef3, coef2, coef1, coef0), 0, None)

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
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

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
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

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

    agecurve_parameters_3, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV G/60']-agecurve_df['Y4 EV G/60'])
    age_curve_max_3 = np.roots([agecurve_parameters_3[0], agecurve_parameters_3[1], agecurve_parameters_3[2], agecurve_parameters_3[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_3[0], agecurve_parameters_3[1], agecurve_parameters_3[2], agecurve_parameters_3[3]]) - 25))]
    root_index_3 = 0
    while np.iscomplex(age_curve_max_3) is True:
        if root_index_3 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_3 = np.real(np.roots([agecurve_parameters_3[0], agecurve_parameters_3[1], agecurve_parameters_3[2], agecurve_parameters_3[3]])[root_index_3])
            root_index_3 += 1
    age_curve_max_3 = np.real(age_curve_max_3)
    coef4_3 = agecurve_parameters_3[0]/4
    coef3_3 = agecurve_parameters_3[1]/3
    coef2_3 = agecurve_parameters_3[2]/2
    coef1_3 = agecurve_parameters_3[3]/1
    coef0_3 = -fourth_degree_polynomial(age_curve_max_3, coef4_3, coef3_3, coef2_3, coef1_3, 0)

    agecurve_parameters_4, _ = curve_fit(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, agecurve_df['Age'], agecurve_df['Y5 EV G/60']-agecurve_df['Y4 EV G/60'])
    age_curve_max_4 = np.roots([agecurve_parameters_4[0], agecurve_parameters_4[1], agecurve_parameters_4[2], agecurve_parameters_4[3]])[np.argmin(np.abs(np.roots([agecurve_parameters_4[0], agecurve_parameters_4[1], agecurve_parameters_4[2], agecurve_parameters_4[3]]) - 25))]
    root_index_4 = 0
    while np.iscomplex(age_curve_max_4) is True:
        if root_index_4 >= 10:
            print('No real roots')
            return
        else:
            age_curve_max_4 = np.real(np.roots([agecurve_parameters_4[0], agecurve_parameters_4[1], agecurve_parameters_4[2], agecurve_parameters_4[3]])[root_index_4])
            root_index_4 += 1
    age_curve_max_4 = np.real(age_curve_max_4)
    coef4_4 = agecurve_parameters_4[0]/4
    coef3_4 = agecurve_parameters_4[1]/3
    coef2_4 = agecurve_parameters_4[2]/2
    coef1_4 = agecurve_parameters_4[3]/1
    coef0_4 = -fourth_degree_polynomial(age_curve_max_4, coef4_4, coef3_4, coef2_4, coef1_4, 0)

    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PP', 'RR')
    instance_df_y3['Y1 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y2 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y3 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y4 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y5 PP G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_1, coef3_1, coef2_1, coef1_1, coef0_1)
    instance_df_y3['Y1 PP ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y2 PP ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y3 PP ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y4 PP ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y5 PP ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_2, coef3_2, coef2_2, coef1_2, coef0_2)
    instance_df_y3['Y1 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_3, coef3_3, coef2_3, coef1_3, coef0_3)
    instance_df_y3['Y2 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_3, coef3_3, coef2_3, coef1_3, coef0_3)
    instance_df_y3['Y3 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_3, coef3_3, coef2_3, coef1_3, coef0_3)
    instance_df_y3['Y4 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_3, coef3_3, coef2_3, coef1_3, coef0_3)
    instance_df_y3['Y5 EV G/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_3, coef3_3, coef2_3, coef1_3, coef0_3)
    instance_df_y3['Y1 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-4, coef4_4, coef3_4, coef2_4, coef1_4, coef0_4)
    instance_df_y3['Y2 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-3, coef4_4, coef3_4, coef2_4, coef1_4, coef0_4)
    instance_df_y3['Y3 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-2, coef4_4, coef3_4, coef2_4, coef1_4, coef0_4)
    instance_df_y3['Y4 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-1, coef4_4, coef3_4, coef2_4, coef1_4, coef0_4)
    instance_df_y3['Y5 EV ixG/60'] -= fourth_degree_polynomial(instance_df_y3['Age']-0, coef4_4, coef3_4, coef2_4, coef1_4, coef0_4)

    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'forward', 'RR')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 2, 'PP', 'forward', 'RR')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 1, 'PP', 'forward', 'RR')

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
            pseudo_prev_year_stat_2,
            pseudo_prev_year_stat_3,
            pseudo_prev_year_stat_4
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

    column_name = 'PP G/60'

    for index, statline in enumerate(yr3_stat_list):
        player_name = yr3_group[index]
        projection = np.clip(yr3_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr2_stat_list):
        player_name = yr2_group[index]
        projection = np.clip(yr2_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    for index, statline in enumerate(yr1_stat_list):
        player_name = yr1_group[index]
        projection = np.clip(yr1_predictions[index] + fourth_degree_polynomial(calc_age(stat_df.loc[stat_df['Player'] == player_name, f'Date of Birth'].iloc[0], 2023), coef4_1, coef3_1, coef2_1, coef1_1, coef0_1), 0, None)

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

def make_projections(existing_stat_df=True, existing_partial_projections=True, year=2024, download_csv=False):
    stat_df = preprocessing_training_functions.scrape_player_statistics(existing_stat_df)

    if existing_partial_projections == False:
        projection_df = preprocessing_training_functions.make_projection_df(stat_df)
    else:
        projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/rr_partial_projections_{year}.csv")
        projection_df = projection_df.drop(projection_df.columns[0], axis=1)

    # projection_df = make_forward_gp_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_gp_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_ev_atoi_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_ev_atoi_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_pp_atoi_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_pp_atoi_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_pk_atoi_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_pk_atoi_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_ev_gper60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_ev_gper60_projections(stat_df, projection_df, True, year)
    projection_df = make_forward_pp_gper60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_pp_gper60_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_pk_gper60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_pk_gper60_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_ev_a1per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_ev_a1per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_pp_a1per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_pp_a1per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_pk_a1per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_pk_a1per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_ev_a2per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_ev_a2per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_pp_a2per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_pp_a2per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_forward_pk_a2per60_projections(stat_df, projection_df, True, year)
    # projection_df = make_defence_pk_a2per60_projections(stat_df, projection_df, True, year)

    projection_df = projection_df.sort_values('PP G/60', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    # print(projection_df.head(20))
    print(projection_df.to_string())

    if download_csv == True:
        filename = f'rr_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

make_projections(True, True, 2024, False)
