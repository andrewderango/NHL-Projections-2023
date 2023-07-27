import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import ast
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

def prior_trainable(kernel_size, bias_size, stdev, dtype):
    n = kernel_size + bias_size

    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=stdev),
            reinterpreted_batch_ndims=1)),
    ])

def make_forward_gp_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=10, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    # yr4_model.summary()

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 4, None, 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'GP', 4, None, None, 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 3, None, 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 3, None, None, 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 2, None, 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'GP', 2, None, None, 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'forward', 1, None, 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'GP', 1, None, None, 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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

    yr4_stat_list, yr3_stat_list, yr2_stat_list, yr1_stat_list = [], [], [], []

    for player in yr4_group:
        yr4_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-4} GP'].iloc[0]*gp_adjustment_factor[year-4],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} GP'].iloc[0]*gp_adjustment_factor[year-3],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} GP'].iloc[0]*gp_adjustment_factor[year-2],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'GP'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(min(yr4_predictions[sim][index][0] + statistics.mean(statline[-4:]), 82), 0))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(min(yr3_predictions[sim][index][0] + statistics.mean(statline[-3:]), 82), 0))
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(min(yr2_predictions[sim][index][0] + statistics.mean(statline[-2:]), 82), 0))
        
        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(min(yr1_predictions[sim][index][0] + statistics.mean(statline[-1:]), 82), 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_gp_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=10, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=5, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.50
    l2_alpha = 0.01
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 4, None, 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'GP', 4, None, None, 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 3, None, 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'GP', 3, None, None, 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 2, None, 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'GP', 2, None, None, 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('GP', 'defence', 1, None, 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'GP', 1, None, None, 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
            if row[f'{year-4} GP']*gp_adjustment_factor[year-4] >= 50 and row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 50 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr4_group.append(row['Player'])
            elif row[f'{year-3} GP']*gp_adjustment_factor[year-3] >= 50 and row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
                yr3_group.append(row['Player'])
            elif row[f'{year-2} GP']*gp_adjustment_factor[year-2] >= 50 and row[f'{year-1} GP']*gp_adjustment_factor[year-1] >= 50:
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
            stat_df.loc[stat_df['Player'] == player, f'{year-1} GP'].iloc[0]*gp_adjustment_factor[year-1]])
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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'GP'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(min(yr4_predictions[sim][index][0] + statistics.mean(statline[-4:]), 82), 0))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)
        # print(yr4_group[index], round(projection, 1), simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(min(yr3_predictions[sim][index][0] + statistics.mean(statline[-3:]), 82), 0))
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)
        # print(yr4_group[index], round(projection, 1), simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(min(yr2_predictions[sim][index][0] + statistics.mean(statline[-2:]), 82), 0))
        
        player_name = yr2_group[index]
        projection = statistics.mean(simulations)
        # print(yr4_group[index], round(projection, 1), simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(min(yr1_predictions[sim][index][0] + statistics.mean(statline[-1:]), 82), 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)
        # print(yr4_group[index], round(projection, 1), simulations)

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

def make_forward_ev_atoi_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=1.75, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=2.00, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=2.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=4, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=4, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])


    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'EV', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'EV', None, 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'EV', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV', None, 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'EV', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'EV', None, 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'EV', None, 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[-4:]), 0))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[-3:]), 0))
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[-2:]), 0))
        
        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[-1:]), 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_ev_atoi_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=1.75, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=2.00, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=2.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=4, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=4, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])


    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'EV', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'EV', None, 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'EV', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'EV', None, 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'EV', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'EV', None, 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'EV', None, 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV ATOI'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[-4:]), 0))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[-3:]), 0))
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[-2:]), 0))
        
        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[-1:]), 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_pp_atoi_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.40, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])


    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'PP', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PP', None, 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'PP', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP', None, 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'PP', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PP', None, 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'PP', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PP', None, 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[-4:]), 0))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[-3:]), 0))
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[-2:]), 0))
        
        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[-1:]), 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_pp_atoi_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.40, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])


    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'PP', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PP', None, 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'PP', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PP', None, 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'PP', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PP', None, 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'PP', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PP', None, 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PP ATOI'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[-4:]), 0))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[-3:]), 0))
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[-2:]), 0))
        
        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[-1:]), 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_pk_atoi_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.25, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.40, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])


    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 4, 'PK', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PK', None, 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 3, 'PK', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK', None, 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 2, 'PK', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PK', None, 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'forward', 1, 'PK', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PK', None, 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[-4:]), 0))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[-3:]), 0))
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[-2:]), 0))
        
        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[-1:]), 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_pk_atoi_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.25, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.40, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])


    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 4, 'PK', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'ATOI', 4, 'PK', None, 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 3, 'PK', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'ATOI', 3, 'PK', None, 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 2, 'PK', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'ATOI', 2, 'PK', None, 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('ATOI', 'defence', 1, 'PK', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'ATOI', 1, 'PK', None, 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PK ATOI'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[-4:]), 0))
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[-3:]), 0))
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[-2:]), 0))
        
        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[-1:]), 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_ev_gper60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 11
    input_descent = 2

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'EV', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'EV', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'EV', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'forward', 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV G/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in ['Kirill Kaprizov', 'Jason Robertson', 'Tage Thompson']:
            print(player_name, statline, projection)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[3:4]), 0))
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_ev_gper60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 11
    input_descent = 2

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.40, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 2.00
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'EV', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'EV', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'EV', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'defence', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV G/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]))

        simulations = np.array(simulations)
        temperature = 0.18763678399823020
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]))

        simulations = np.array(simulations)
        temperature = 0.18763678399823020
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]))

        simulations = np.array(simulations)
        temperature = 0.22763678399823020
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr1_predictions[sim][index][0], 0))
            simulations.append(yr1_predictions[sim][index][0])

        simulations = np.array(simulations)
        temperature = 51.04367839982302031
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_pp_gper60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 19
    input_descent = 4

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.10, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 2.00
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'PP', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PP', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PP', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'PP', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PP', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'PP', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PP', 'forward', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PP G/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in ['Kirill Kaprizov', 'Jason Robertson', 'Tage Thompson']:
            print(player_name, statline, projection)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_pp_gper60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 19
    input_descent = 4

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.10, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 2.00
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'PP', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PP', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'PP', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PP', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'PP', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PP', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'PP', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PP', 'defence', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PP G/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]), 0))

        # simulations = np.array(simulations)
        # temperature = 23.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        # simulations = [max(0, num) for num in simulations]
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[3:4]), 0))
            simulations.append(yr1_predictions[sim][index][0] + statistics.mean(statline[3:4]))

        simulations = np.array(simulations)
        temperature = 23.0661224967399795
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_pk_gper60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.10, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'PK', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PK', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'PK', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'PK', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PK', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'PK', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PK', 'forward', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PK G/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in ['Kirill Kaprizov', 'Jason Robertson', 'Tage Thompson']:
            print(player_name, statline, projection)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_pk_gper60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.10, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 4, 'PK', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'PK', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 3, 'PK', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'PK', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 2, 'PK', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'PK', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'defence', 1, 'PK', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'PK', 'defence', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PK G/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in ['Kirill Kaprizov', 'Jason Robertson', 'Tage Thompson']:
            print(player_name, statline, projection)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_ev_a1per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 23
    input_descent = 5

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(3,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*4, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'EV', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'EV', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'EV', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'EV', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'EV', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'EV', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'EV', 'forward', 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=5, verbose=1)

    for item in range(len(X_1_scaled)):
        print(X_1_scaled[item], y_1[item])

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

        # yr1_stat_list.append([
        #     preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
        #     int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
        #     int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
        #     pseudo_prev_year_stat_1,
        #     pseudo_prev_year_stat_2,
        #     pseudo_prev_year_stat_3,
        #     pseudo_prev_year_stat_4,
        #     pseudo_prev_year_stat_5
        #     ])

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV A1/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]), 0))

        # simulations = np.array(simulations)
        # temperature = 0.6661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        # simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_ev_a1per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 23
    input_descent = 5

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(3,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*4, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'EV', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'EV', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'EV', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'EV', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'EV', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'EV', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'EV', 'defence', 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=5, verbose=1)

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

        # yr1_stat_list.append([
        #     preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
        #     int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
        #     int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
        #     pseudo_prev_year_stat_1,
        #     pseudo_prev_year_stat_2,
        #     pseudo_prev_year_stat_3,
        #     pseudo_prev_year_stat_4,
        #     pseudo_prev_year_stat_5
        #     ])
        
        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            pseudo_prev_year_stat_1,
            pseudo_prev_year_stat_2
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV A1/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]), 0))
            simulations.append(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]))

        simulations = np.array(simulations)
        temperature = 0.258712249673997
        simulations = temperature*(simulations-np.mean(simulations)) + statistics.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]))

        simulations = np.array(simulations)
        temperature = 0.587122496739972
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]))

        simulations = np.array(simulations)
        temperature = 0.687122496739977
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_pp_a1per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 27
    input_descent = 6

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'PP', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PP', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'PP', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PP', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'PP', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PP', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'PP', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PP', 'forward', 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PP A1/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]), 0))

        # simulations = np.array(simulations)
        # temperature = 0.6661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        # simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_pp_a1per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 27
    input_descent = 6

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'PP', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PP', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'PP', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PP', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'PP', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PP', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'PP', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PP', 'defence', 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PP A1/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]), 0))

        # simulations = np.array(simulations)
        # temperature = 0.6661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        # simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_pk_a1per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.10, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 4, 'PK', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PK', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 3, 'PK', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PK', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 2, 'PK', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PK', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'forward', 1, 'PK', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PK', 'forward', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PK A1/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_pk_a1per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.10, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 4, 'PK', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A1per60', 4, 'PK', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 3, 'PK', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A1per60', 3, 'PK', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 2, 'PK', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A1per60', 2, 'PK', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A1per60', 'defence', 1, 'PK', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A1per60', 1, 'PK', 'defence', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PK A1/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_ev_a2per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 23
    input_descent = 5

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'EV', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'EV', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'EV', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'EV', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'EV', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'EV', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'EV', 'forward', 'NN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV A2/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[7:11]), 0))

        # simulations = np.array(simulations)
        # temperature = 0.6661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        # simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[6:9]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[5:7]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(yr1_predictions[sim][index][0])

        simulations = np.array(simulations)
        temperature = 26.48224967399360
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_ev_a2per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 23
    input_descent = 5

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'EV', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'EV', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'EV', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'EV', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'EV', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'EV', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'EV', 'defence', 'NN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV A2/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(yr4_predictions[sim][index][0])

        simulations = np.array(simulations)
        temperature = 0.3711224967399795
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[6:9]), 0))

        # simulations = np.array(simulations)
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[5:7]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))

        simulations = np.array(simulations)
        temperature = 26.48224967399360
        simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        simulations = simulations.tolist()
        simulations = [max(0, num) for num in simulations]
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_pp_a2per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 27
    input_descent = 6

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'PP', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PP', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'PP', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PP', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'PP', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PP', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'PP', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PP', 'forward', 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PP A2/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[7:11]), 0))

        # simulations = np.array(simulations)
        # temperature = 0.6661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        # simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[6:9]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[5:7]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_pp_a2per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 27
    input_descent = 6

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'PP', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PP', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'PP', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PP', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'PP', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PP', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'PP', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PP', 'defence', 'BNN')

    X_4_scaler = StandardScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = StandardScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = StandardScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=15, verbose=1)

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
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PP A2/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[7:11]), 0))

        # simulations = np.array(simulations)
        # temperature = 0.6661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        # simulations = [max(0, num) for num in simulations]
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[6:9]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr3_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[5:7]), 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_forward_pk_a2per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.10, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 4, 'PK', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PK', 'forward', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 3, 'PK', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PK', 'forward', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 2, 'PK', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PK', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'forward', 1, 'PK', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PK', 'forward', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
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
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PK A2/60'].mean() - instance_df_y1['Y4 PK A2/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PK A2/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def make_defence_pk_a2per60_projections(stat_df, projection_df, distribution_df, download_file, sim_count, year=2024):
    max_input_shape = 7
    input_descent = 1

    yr4_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.10, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr3_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr2_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    for layer in yr4_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr3_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr2_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 4, 'PK', 'BNN')
    X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'A2per60', 4, 'PK', 'defence', 'BNN')
    instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 3, 'PK', 'BNN')
    X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'A2per60', 3, 'PK', 'defence', 'BNN')
    instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 2, 'PK', 'BNN')
    X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'A2per60', 2, 'PK', 'defence', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('A2per60', 'defence', 1, 'PK', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'A2per60', 1, 'PK', 'defence', 'BNN')

    X_4_scaler = MinMaxScaler().fit(X_4)
    X_4_scaled = X_4_scaler.transform(X_4)
    X_3_scaler = MinMaxScaler().fit(X_3)
    X_3_scaled = X_3_scaler.transform(X_3)
    X_2_scaler = MinMaxScaler().fit(X_2)
    X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = MinMaxScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    yr4_model.fit(X_4_scaled, y_4, epochs=10, verbose=1)
    yr3_model.fit(X_3_scaled, y_3, epochs=10, verbose=1)
    yr2_model.fit(X_2_scaled, y_2, epochs=10, verbose=1)
    yr1_model.fit(X_1_scaled, y_1, epochs=10, verbose=1)

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
            stat_df.loc[stat_df['Player'] == player, f'{year-4} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
            ])

    for player in yr3_group:
        yr3_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            stat_df.loc[stat_df['Player'] == player, f'{year-3} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-2} PK A2/60'].fillna(0).iloc[0],
            stat_df.loc[stat_df['Player'] == player, f'{year-1} PK A2/60'].fillna(0).iloc[0]
            ])
        
    for player in yr2_group:
        yr2_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
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
            negative_first_z_score_stat_1 = max(instance_df_y1['Y4 PK A2/60'].mean() - instance_df_y1['Y4 PK A2/60'].std(), 0) # should not be negative
            games_to_pseudofy = 75-(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi)
            pseudo_prev_year_stat = (y1_stat*y1_pktoi + y2_stat*y2_pktoi + y3_stat*y3_pktoi + y4_stat*y4_pktoi + negative_first_z_score_stat_1*games_to_pseudofy)/(y1_pktoi + y2_pktoi + y3_pktoi + y4_pktoi + games_to_pseudofy)

        yr1_stat_list.append([
            preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
            int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
            int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
            pseudo_prev_year_stat
            ])

    yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'PK A2/60'

    for index, statline in enumerate(yr4_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr4_predictions[sim][index][0], 0))
            simulations.append(max(yr4_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()
        
        player_name = yr4_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr3_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr3_predictions[sim][index][0], 0))
            simulations.append(max(yr3_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr2_stat_list):
        simulations = []
        for sim in range(sim_count):
            # simulations.append(max(yr2_predictions[sim][index][0], 0))
            simulations.append(max(yr2_predictions[sim][index][0], 0))

        # simulations = np.array(simulations)
        # # temperature = 23.0661224967399795
        # temperature = 1.0661224967399795
        # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
        # simulations = simulations.tolist()

        player_name = yr2_group[index]
        projection = statistics.mean(simulations)

        if player_name in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
            distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
        else:
            new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
            new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
            distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    for index, statline in enumerate(yr1_stat_list):
        simulations = []
        for sim in range(sim_count):
            simulations.append(max(yr1_predictions[sim][index][0], 0))
        
        player_name = yr1_group[index]
        projection = statistics.mean(simulations)

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

def goal_era_adjustment(stat_df, projection_df, distribution_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_goal_df = pd.DataFrame()

    for season in range(2007, 2023):
        col = round(((stat_df[f'{season+1} EV G/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP G/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK G/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])).astype(int)
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_goal_df = hist_goal_df.reset_index(drop=True)
        hist_goal_df[season+1] = col
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
    hist_goal_df['Projected Average'] = hist_goal_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_goal_df['Adjustment'] = hist_goal_df['Projected Average'] - hist_goal_df['Historical Average']
    hist_goal_df['Smoothed Adjustment'] = savgol_filter(hist_goal_df['Adjustment'], 25, 2)
    # print(hist_goal_df.head(750).to_string())

    projection_df['GOALS'] = (projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP'].astype(int)
    projection_df = projection_df.sort_values('GOALS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_goal_df['Smoothed Adjustment']/((projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        distribution_df = distribution_df.merge(projection_df[['Player', 'Era Adjustment Factor']], on='Player', how='outer')
        projection_df['EV G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['GOALS'] = round((projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']).astype(int)
        # print(projection_df)
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

        for index_1, player_name in enumerate(distribution_df['Player']):
            try:
                distribution_df.loc[index_1, 'EV G/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'EV G/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            except ValueError:
                distribution_df.loc[index_1, 'EV G/60'] = str([0 for _ in range(1000)])
            try:
                distribution_df.loc[index_1, 'PP G/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'PP G/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            except ValueError:
                distribution_df.loc[index_1, 'PP G/60'] = str([0 for _ in range(1000)])
            try:
                distribution_df.loc[index_1, 'PK G/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'PK G/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            except ValueError:
                distribution_df.loc[index_1, 'PK G/60'] = str([0 for _ in range(1000)])
        distribution_df = distribution_df.drop(columns=['Era Adjustment Factor'])

    for index, simulation in enumerate(distribution_df['Player']):
        ev_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'EV G/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'EV ATOI']))
        try:
            pp_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'PP G/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'PP ATOI']))
        except ValueError:
            pp_stat = [0 for _ in range(1000)]
        try:
            pk_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'PK G/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'PK ATOI']))
        except ValueError:
            pk_stat = [0 for _ in range(1000)]

        stat_total = np.array([x + y + z for x, y, z in zip(ev_stat, pp_stat, pk_stat)]) * np.array(ast.literal_eval(distribution_df.loc[index, 'GP']))
        distribution_df.loc[index, 'GOALS'] = str(stat_total)

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

    return projection_df.fillna(0), distribution_df

def a1_era_adjustment(stat_df, projection_df, distribution_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_goal_df = pd.DataFrame()

    for season in range(2007, 2023):
        col = round(((stat_df[f'{season+1} EV A1/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP A1/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK A1/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])).astype(int)
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_goal_df = hist_goal_df.reset_index(drop=True)
        hist_goal_df[season+1] = col
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
    hist_goal_df['Projected Average'] = hist_goal_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_goal_df['Adjustment'] = hist_goal_df['Projected Average'] - hist_goal_df['Historical Average']
    hist_goal_df['Smoothed Adjustment'] = savgol_filter(hist_goal_df['Adjustment'], 25, 2)
    # print(hist_goal_df.head(750).to_string())

    projection_df['PRIMARY ASSISTS'] = (projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP'].astype(int)
    projection_df = projection_df.sort_values('PRIMARY ASSISTS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_goal_df['Smoothed Adjustment']/((projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        distribution_df = distribution_df.merge(projection_df[['Player', 'Era Adjustment Factor']], on='Player', how='outer')
        projection_df['EV A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PRIMARY ASSISTS'] = round((projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP']).astype(int)
        # print(projection_df)
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

        for index_1, player_name in enumerate(distribution_df['Player']):
            distribution_df.loc[index_1, 'EV A1/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'EV A1/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            try:
                distribution_df.loc[index_1, 'PP A1/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'PP A1/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            except ValueError:
                distribution_df.loc[index_1, 'PP A1/60'] = str([0 for _ in range(1000)])
            try:
                distribution_df.loc[index_1, 'PK A1/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'PK A1/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            except ValueError:
                distribution_df.loc[index_1, 'PK A1/60'] = str([0 for _ in range(1000)])
        distribution_df = distribution_df.drop(columns=['Era Adjustment Factor'])

    for index, simulation in enumerate(distribution_df['Player']):
        ev_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'EV A1/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'EV ATOI']))
        try:
            pp_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'PP A1/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'PP ATOI']))
        except ValueError:
            pp_stat = [0 for _ in range(1000)]
        try:
            pk_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'PK A1/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'PK ATOI']))
        except ValueError:
            pk_stat = [0 for _ in range(1000)]

        stat_total = np.array([x + y + z for x, y, z in zip(ev_stat, pp_stat, pk_stat)]) * np.array(ast.literal_eval(distribution_df.loc[index, 'GP']))
        distribution_df.loc[index, 'PRIMARY ASSISTS'] = str(stat_total)

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

    return projection_df.fillna(0), distribution_df

def a2_era_adjustment(stat_df, projection_df, distribution_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_goal_df = pd.DataFrame()

    for season in range(2007, 2023):
        col = round(((stat_df[f'{season+1} EV A2/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP A2/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK A2/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])).astype(int)
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_goal_df = hist_goal_df.reset_index(drop=True)
        hist_goal_df[season+1] = col
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
    hist_goal_df['Projected Average'] = hist_goal_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_goal_df['Adjustment'] = hist_goal_df['Projected Average'] - hist_goal_df['Historical Average']
    hist_goal_df['Smoothed Adjustment'] = savgol_filter(hist_goal_df['Adjustment'], 25, 2)
    # print(hist_goal_df.head(750).to_string())

    projection_df['SECONDARY ASSISTS'] = (projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP'].astype(int)
    projection_df = projection_df.sort_values('SECONDARY ASSISTS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_goal_df['Smoothed Adjustment']/((projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        distribution_df = distribution_df.merge(projection_df[['Player', 'Era Adjustment Factor']], on='Player', how='outer')
        projection_df['EV A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['SECONDARY ASSISTS'] = round((projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP']).astype(int)
        # print(projection_df)
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

        for index_1, player_name in enumerate(distribution_df['Player']):
            distribution_df.loc[index_1, 'EV A2/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'EV A2/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            try:
                distribution_df.loc[index_1, 'PP A2/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'PP A2/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            except ValueError:
                distribution_df.loc[index_1, 'PP A2/60'] = str([0 for _ in range(1000)])
            try:
                distribution_df.loc[index_1, 'PK A2/60'] = str((np.array(ast.literal_eval(distribution_df.loc[index_1, 'PK A2/60'])) * distribution_df.loc[index_1, 'Era Adjustment Factor']).tolist())
            except ValueError:
                distribution_df.loc[index_1, 'PK A2/60'] = str([0 for _ in range(1000)])
        distribution_df = distribution_df.drop(columns=['Era Adjustment Factor'])

    for index, simulation in enumerate(distribution_df['Player']):
        ev_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'EV A2/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'EV ATOI']))
        try:
            pp_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'PP A2/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'PP ATOI']))
        except ValueError:
            pp_stat = [0 for _ in range(1000)]
        try:
            pk_stat = np.array(ast.literal_eval(distribution_df.loc[index, 'PK A2/60']))/60*np.array(ast.literal_eval(distribution_df.loc[index, 'PK ATOI']))
        except ValueError:
            pk_stat = [0 for _ in range(1000)]

        stat_total = np.array([x + y + z for x, y, z in zip(ev_stat, pp_stat, pk_stat)]) * np.array(ast.literal_eval(distribution_df.loc[index, 'GP']))
        distribution_df.loc[index, 'SECONDARY ASSISTS'] = str(stat_total)

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

    return projection_df.fillna(0), distribution_df

def make_projections(existing_stat_df=True, existing_partial_projections=True, year=2024, download_csv=False):
    stat_df = preprocessing_training_functions.scrape_player_statistics(existing_stat_df)
    
    if existing_partial_projections == False:
        projection_df = preprocessing_training_functions.make_projection_df(stat_df)
        distribution_df = preprocessing_training_functions.make_projection_df(stat_df)
    else:
        projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/bayesian_nn_partial_projections_{year}.csv")
        projection_df = projection_df.drop(projection_df.columns[0], axis=1)
        distribution_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/bayesian_nn_partial_distributions_{year}.csv")
        distribution_df = distribution_df.drop(distribution_df.columns[0], axis=1)

    # projection_df, distribution_df = make_forward_gp_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_gp_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_ev_atoi_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_ev_atoi_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_pp_atoi_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_pp_atoi_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_pk_atoi_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_pk_atoi_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_ev_gper60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_ev_gper60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_pp_gper60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_pp_gper60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_pk_gper60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_pk_gper60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_ev_a1per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_ev_a1per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_pp_a1per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_pp_a1per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_pk_a1per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_pk_a1per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_ev_a2per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_ev_a2per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_pp_a2per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_pp_a2per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_forward_pk_a2per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)
    # projection_df, distribution_df = make_defence_pk_a2per60_projections(stat_df, projection_df, distribution_df, True, 1000, year)

    # projection_df, distribution_df = goal_era_adjustment(stat_df, projection_df, distribution_df, 2024, False, False)
    # projection_df, distribution_df = a1_era_adjustment(stat_df, projection_df, distribution_df, 2024, False, False)
    # projection_df, distribution_df = a2_era_adjustment(stat_df, projection_df, distribution_df, 2024, False, False)
    # projection_df['POINTS'] = projection_df['GOALS'] + projection_df['PRIMARY ASSISTS'] + projection_df['SECONDARY ASSISTS']
    # for index, player in enumerate(distribution_df['Player']):
    #     points = []
    #     goals = [float(x) for x in distribution_df.loc[index, 'GOALS'][1:-1].split()]
    #     p_assists = [float(x) for x in distribution_df.loc[index, 'PRIMARY ASSISTS'][1:-1].split()]
    #     s_assists = [float(x) for x in distribution_df.loc[index, 'SECONDARY ASSISTS'][1:-1].split()]

    #     for simulation in range(1000):
    #         points.append(goals[simulation] + p_assists[simulation] + s_assists[simulation])

    #     distribution_df.loc[index, 'POINTS'] = str(points)
    #     print(index, distribution_df.loc[index, 'Player'], points[:5])

    projection_df = projection_df.sort_values('POINTS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    print(projection_df)
    print(distribution_df)

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

make_projections(True, True, 2024, False)
