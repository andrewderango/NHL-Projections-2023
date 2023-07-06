import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import date
import numpy as np
import statistics
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import statistics
from scipy.signal import savgol_filter
import preprocessing_training_functions
import tensorflow_probability as tfp
tfd = tfp.distributions

def posterior_mean_field(kernel_size, bias_size, dtype):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

def prior_trainable(kernel_size, bias_size, dtype):
    n = kernel_size + bias_size

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=10),
            reinterpreted_batch_ndims=1)),
    ])

def make_forward_gp_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(7,), make_posterior_fn=posterior_mean_field, make_prior_fn=prior_trainable),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=prior_trainable),
        tf.keras.layers.Dense(1),
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    # yr4_model.summary()

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 4, None)
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'GP', 4, None)

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)

    yr4_model.fit(X_4_scaled, y_4, epochs=5, verbose=1)

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
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 50 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 50 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 50 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr2_group.append(row['Player'])
            else:
                yr1_group.append(row['Player'])

    yr4_stat_list = []

    for player in yr4_group:
        yr4_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].iloc[0]*gp_adjustment_factor[year-4],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'GP'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(min(predictions[sim][index][0] + statistics.mean(statline[-4:]), 82))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)
        print(yr4_group[index], round(projection, 1), simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    # Download file
    if download_file == True:
        filename = f'bayesian_nn_partial_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'bayesian_nn_partial_distributions_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        distribution_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df, distribution_df

def make_projections(existing_stat_df=True, existing_partial_projections=True, year=2024, download_csv=False):
    stat_df = preprocessing_training_functions.scrape_player_statistics(existing_stat_df)
    
    if existing_partial_projections == False:
        projection_df = preprocessing_training_functions.make_projection_df(stat_df)
        distribution_df = preprocessing_training_functions.make_projection_df(stat_df)
        projection_df, distribution_df = make_forward_gp_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    else:
        projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/bayesian_nn_partial_projections_{year}.csv")
        projection_df = projection_df.drop(projection_df.columns[0], axis=1)
        distribution_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/bayesian_nn_partial_distributions_{year}.csv")
        distribution_df = distribution_df.drop(distribution_df.columns[0], axis=1)

    projection_df = projection_df.sort_values('GP', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    print(projection_df)
    print(distribution_df.to_string())

    if download_csv == True:
        filename = f'bayesian_nn_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

        filename = f'bayesian_nn_final_distributions_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        distribution_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

make_projections(True, False, 2024, False)
