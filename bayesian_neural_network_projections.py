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

    # yr4_model = tf.keras.Sequential([
    #     tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 0*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.20, kernel_size=64*((max_input_shape - 0*input_descent)+1), bias_size=0, dtype=np.float64)),
    #     tf.keras.layers.Dense(64, activation = 'relu'),
    #     tf.keras.layers.Dense(1)
    # ])
    # yr3_model = tf.keras.Sequential([
    #     tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 1*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.30, kernel_size=64*((max_input_shape - 1*input_descent)+1), bias_size=0, dtype=np.float64)),
    #     tf.keras.layers.Dense(64, activation = 'relu'),
    #     tf.keras.layers.Dense(1)
    # ])
    # yr2_model = tf.keras.Sequential([
    #     tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 2*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.50, kernel_size=64*((max_input_shape - 2*input_descent)+1), bias_size=0, dtype=np.float64)),
    #     tf.keras.layers.Dense(64, activation = 'relu'),
    #     tf.keras.layers.Dense(1)
    # ])
    yr1_model = tf.keras.Sequential([
        tfp.layers.DenseVariational(64, activation='relu', input_shape=(max_input_shape - 3*input_descent,), make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=64*((max_input_shape - 3*input_descent)+1), bias_size=0, dtype=np.float64)),
        tfp.layers.DenseVariational(64, activation='relu', make_posterior_fn=posterior_mean_field, make_prior_fn=lambda *args, **kwargs: prior_trainable(stdev=0.01, kernel_size=(64+1)*64, bias_size=0, dtype=np.float64)),
        tf.keras.layers.Dense(1)
    ])

    l1_lambda = 0.20
    l2_alpha = 0.02
    # for layer in yr4_model.layers:
    #     if isinstance(layer, tf.keras.layers.Dense):
    #         layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    # for layer in yr3_model.layers:
    #     if isinstance(layer, tf.keras.layers.Dense):
    #         layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    # for layer in yr2_model.layers:
    #     if isinstance(layer, tf.keras.layers.Dense):
    #         layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))
    for layer in yr1_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l1_l2(l1_lambda, l2_alpha)(layer.kernel))

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    # yr4_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    # yr3_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    # yr2_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    yr1_model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    # instance_df_y4, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 4, 'EV', 'BNN')
    # X_4, y_4 = preprocessing_training_functions.extract_instance_data(instance_df_y4, 'Gper60', 4, 'EV', 'forward', 'BNN')
    # instance_df_y3, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 3, 'EV', 'BNN')
    # X_3, y_3 = preprocessing_training_functions.extract_instance_data(instance_df_y3, 'Gper60', 3, 'EV', 'forward', 'BNN')
    # instance_df_y2, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 2, 'EV', 'BNN')
    # X_2, y_2 = preprocessing_training_functions.extract_instance_data(instance_df_y2, 'Gper60', 2, 'EV', 'forward', 'BNN')
    instance_df_y1, _ = preprocessing_training_functions.create_year_restricted_instance_df('Gper60', 'forward', 1, 'EV', 'BNN')
    X_1, y_1 = preprocessing_training_functions.extract_instance_data(instance_df_y1, 'Gper60', 1, 'EV', 'forward', 'BNN')

    # X_4_scaler = StandardScaler().fit(X_4)
    # X_4_scaled = X_4_scaler.transform(X_4)
    # X_3_scaler = StandardScaler().fit(X_3)
    # X_3_scaled = X_3_scaler.transform(X_3)
    # X_2_scaler = StandardScaler().fit(X_2)
    # X_2_scaled = X_2_scaler.transform(X_2)
    X_1_scaler = StandardScaler().fit(X_1)
    X_1_scaled = X_1_scaler.transform(X_1)

    # yr4_model.fit(X_4_scaled, y_4, epochs=15, verbose=1)
    # yr3_model.fit(X_3_scaled, y_3, epochs=15, verbose=1)
    # yr2_model.fit(X_2_scaled, y_2, epochs=15, verbose=1)
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

    # for player in yr4_group:
    #     yr4_stat_list.append([
    #         preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
    #         int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
    #         int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
    #         stat_df.loc[stat_df['Player'] == player, f'{year-4} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-4} EV ixG/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
    #         ])

    # for player in yr3_group:
    #     yr3_stat_list.append([
    #         preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
    #         int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
    #         int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
    #         stat_df.loc[stat_df['Player'] == player, f'{year-3} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-3} EV ixG/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
    #         ])
        
    # for player in yr2_group:
    #     yr2_stat_list.append([
    #         preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], year-1),
    #         int(stat_df.loc[stat_df['Player'] == player, 'Height (in)'].iloc[0]),
    #         int(stat_df.loc[stat_df['Player'] == player, 'Weight (lbs)'].iloc[0]),
    #         stat_df.loc[stat_df['Player'] == player, f'{year-2} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV G/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-2} EV ixG/60'].fillna(0).iloc[0],
    #         stat_df.loc[stat_df['Player'] == player, f'{year-1} EV ixG/60'].fillna(0).iloc[0]
    #         ])
        
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

    # yr4_stat_list_scaled = X_4_scaler.transform(yr4_stat_list)
    # yr4_predictions = [yr4_model.predict(yr4_stat_list_scaled) for _ in range(sim_count)]
    # yr3_stat_list_scaled = X_3_scaler.transform(yr3_stat_list)
    # yr3_predictions = [yr3_model.predict(yr3_stat_list_scaled) for _ in range(sim_count)]
    # yr2_stat_list_scaled = X_2_scaler.transform(yr2_stat_list)
    # yr2_predictions = [yr2_model.predict(yr2_stat_list_scaled) for _ in range(sim_count)]
    yr1_stat_list_scaled = X_1_scaler.transform(yr1_stat_list)
    yr1_predictions = [yr1_model.predict(yr1_stat_list_scaled) for _ in range(sim_count)]

    column_name = 'EV G/60'

    # for index, statline in enumerate(yr4_stat_list):
    #     simulations = []
    #     for sim in range(sim_count):
    #         # simulations.append(max(yr4_predictions[sim][index][0], 0))
    #         simulations.append(max(yr4_predictions[sim][index][0] + statistics.mean(statline[3:7]), 0))

    #     # simulations = np.array(simulations)
    #     # # temperature = 23.0661224967399795
    #     # temperature = 1.0661224967399795
    #     # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
    #     # simulations = simulations.tolist()
        
    #     player_name = yr4_group[index]
    #     projection = statistics.mean(simulations)

    #     if player_name in projection_df['Player'].values:
    #         projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
    #         distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
    #     else:
    #         new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
    #         projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    #         new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
    #         distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    # for index, statline in enumerate(yr3_stat_list):
    #     simulations = []
    #     for sim in range(sim_count):
    #         # simulations.append(max(yr3_predictions[sim][index][0], 0))
    #         simulations.append(max(yr3_predictions[sim][index][0] + statistics.mean(statline[3:6]), 0))

    #     # simulations = np.array(simulations)
    #     # # temperature = 23.0661224967399795
    #     # temperature = 1.0661224967399795
    #     # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
    #     # simulations = simulations.tolist()
        
    #     player_name = yr3_group[index]
    #     projection = statistics.mean(simulations)

    #     if player_name in ['Kirill Kaprizov', 'Jason Robertson', 'Tage Thompson']:
    #         print(player_name, statline, projection)

    #     if player_name in projection_df['Player'].values:
    #         projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
    #         distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
    #     else:
    #         new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
    #         projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    #         new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
    #         distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

    # for index, statline in enumerate(yr2_stat_list):
    #     simulations = []
    #     for sim in range(sim_count):
    #         # simulations.append(max(yr2_predictions[sim][index][0], 0))
    #         simulations.append(max(yr2_predictions[sim][index][0] + statistics.mean(statline[3:5]), 0))

    #     # simulations = np.array(simulations)
    #     # # temperature = 23.0661224967399795
    #     # temperature = 1.0661224967399795
    #     # simulations = temperature*(simulations-np.mean(simulations)) + np.mean(simulations)
    #     # simulations = simulations.tolist()

    #     player_name = yr2_group[index]
    #     projection = statistics.mean(simulations)

    #     if player_name in projection_df['Player'].values:
    #         projection_df.loc[projection_df['Player'] == player_name, column_name] = projection
    #         distribution_df.loc[distribution_df['Player'] == player_name, column_name] = [str(simulations)]
    #     else:
    #         new_row = pd.DataFrame({'Player': [player_name], column_name: [projection]})
    #         projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    #         new_row = pd.DataFrame({'Player': [player_name], column_name: [str(simulations)]})
    #         distribution_df = pd.concat([distribution_df, new_row], ignore_index=True)

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
            simulations.append(max(yr1_predictions[sim][index][0] + statistics.mean(statline[3:4]), 0))

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

def make_projections(existing_stat_df=True, existing_partial_projections=True, year=2024, download_csv=False):
    stat_df = preprocessing_training_functions.scrape_player_statistics(existing_stat_df)
    
    if existing_partial_projections == False:
        projection_df = preprocessing_training_functions.make_projection_df(stat_df)
        distribution_df = preprocessing_training_functions.make_projection_df(stat_df)
        projection_df, distribution_df = make_forward_gp_projections(stat_df, projection_df, distribution_df, True, 1000, year)
        projection_df, distribution_df = make_defence_gp_projections(stat_df, projection_df, distribution_df, True, 1000, year)
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

    projection_df = projection_df.sort_values('PP G/60', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    print(projection_df)
    # print(distribution_df)

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
