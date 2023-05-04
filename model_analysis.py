import time
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import preprocessing_training_functions

def create_models(input_shape):
    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(126, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(42, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(48, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model4 = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model5 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(28, activation='relu'),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model6 = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model7 = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model8 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model9 = tf.keras.Sequential([
        tf.keras.layers.Dense(36, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model10 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model11 = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model12 = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    return [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]

def test_models(proj_stat, position, prev_years, proj_x, situation, download_model_analysis_file=True):
    epoch_list = [1, 5, 10, 30, 50, 100]
    scaler_list = [StandardScaler(), MinMaxScaler()]

    model_performance_df = pd.DataFrame(columns=['Model ID', 'Parent Model ID', 'Epochs', 'Scaler', 'MAE Test', 'MAE Train', 'Proj. 1', 'Proj. 2', 'Proj. 3', 'Proj. 4', 'Proj. 5'])

    instance_df, input_shape = preprocessing_training_functions.create_year_restricted_instance_df(proj_stat, position, prev_years, situation)
    model_list = create_models(input_shape)

    print(f'Models to Test: {len(model_list) * len(epoch_list) * len(scaler_list)}')
    for model_index, model in enumerate(model_list):
        for epoch_index, epochs in enumerate(epoch_list):
            for scaler_index, scaler in enumerate(scaler_list):
                model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

                X, y = preprocessing_training_functions.extract_instance_data(instance_df, proj_stat, prev_years, situation)

                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

                X_scaler = scaler.fit(X_train)
                X_train_scaled = X_scaler.transform(X_train)
                X_test_scaled = X_scaler.transform(X_test)

                model.fit(X_train_scaled, y_train, epochs=epochs, verbose=0)
                test_loss, test_acc, *rest = model.evaluate(X_test_scaled, y_test, verbose=0)
                train_loss, train_acc, *rest = model.evaluate(X_train_scaled, y_train, verbose=0)

                # Make projection
                proj_scaled_x = X_scaler.transform(proj_x)
                proj_y = model.predict(proj_scaled_x, verbose=0)

                print(f'Model {model_index*len(epoch_list)*len(scaler_list) + epoch_index*len(scaler_list) + scaler_index + 1}: {test_loss:.2f} MAE')

                if proj_stat == 'GP' or proj_stat == 'ATOI':
                    model_performance_df.loc[model_index*len(epoch_list)*len(scaler_list) + epoch_index*len(scaler_list) + scaler_index + 1] = [
                        int(model_index*len(epoch_list)*len(scaler_list) + epoch_index*len(scaler_list) + scaler_index + 1), 
                        int(model_index+1), 
                        int(epochs), 
                        scaler, 
                        round(test_loss, 2), 
                        round(train_loss, 2),
                        round(float(proj_y[0] + sum(proj_x[0][-prev_years:])/prev_years), 2),
                        round(float(proj_y[1] + sum(proj_x[1][-prev_years:])/prev_years), 2),
                        round(float(proj_y[2] + sum(proj_x[2][-prev_years:])/prev_years), 2),
                        round(float(proj_y[3] + sum(proj_x[3][-prev_years:])/prev_years), 2),
                        round(float(proj_y[4] + sum(proj_x[4][-prev_years:])/prev_years), 2)
                        ]

    model_performance_df = model_performance_df.sort_values('MAE Test')
    model_performance_df = model_performance_df.reset_index(drop=True)
    model_performance_df.index += 1

    if download_model_analysis_file == True:
        if situation == None:
            filename = f'{position}_{proj_stat}_{prev_years}year_model_analysis'
        else:
            filename = f'{position}_{situation}_{proj_stat}_{prev_years}year_model_analysis'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        model_performance_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return model_performance_df, model_list

def recommend_model(model_performance_df, model_list):
    recommended_model = model_performance_df.iloc[0]
    print('\n\n--RECOMMENDED MODEL--')
    print(f'Model #{recommended_model["Model ID"]}: Parent Model {recommended_model["Parent Model ID"]} with {recommended_model["Epochs"]} epochs and {recommended_model["Scaler"]}')
    print(f'This model gave a MAE of {recommended_model["MAE Test"]} on the test and {recommended_model["MAE Train"]} on the train.')
    print(f'Parent Model {recommended_model["Parent Model ID"]} architecture:')
    model_list[recommended_model["Parent Model ID"] - 1].summary()

# Define test cases
def get_sample_projection(proj_stat, position, prev_years, situation):

    if proj_stat == 'GP':
        if prev_years == 4:
            return [
                [26, 72, 188, 32, 66, 45, 50], 
                [39, 73, 192, 80, 68, 43, 52], 
                [28, 70, 178, 6, 3, 12, 21], 
                [30, 72, 213, 82, 82, 82, 82], 
                [27, 73, 192, 71, 75, 81, 76]]
        elif prev_years == 3:
            return [
                [26, 72, 188, 32, 66, 45], 
                [39, 73, 192, 80, 68, 43], 
                [28, 70, 178, 9, 12, 18], 
                [30, 72, 213, 82, 82, 82], 
                [27, 73, 192, 71, 75, 81]]        
        elif prev_years == 2:
            return [
                [27, 72, 188, 66, 45], 
                [20, 73, 192, 71, 82], 
                [28, 70, 178, 42, 38], 
                [30, 72, 213, 82, 82], 
                [29, 73, 192, 75, 81]]
        elif prev_years == 1:
            return [
                [27, 72, 188, 66], 
                [19, 73, 192, 60], 
                [19, 70, 178, 80], 
                [29, 72, 213, 82], 
                [29, 73, 192, 75]]
    
    elif proj_stat == 'ATOI':
        if situation == 'EV':
            if prev_years == 4:
                return [
                    [26, 72, 188, 20, 20, 20, 20], 
                    [22, 73, 192, 18, 19, 20, 21], 
                    [40, 70, 178, 19, 18, 17, 16], 
                    [30, 72, 213, 16, 19, 15, 17], 
                    [27, 73, 192, 14, 16, 10, 18]]
            elif prev_years == 3:
                return [
                    [26, 72, 188, 20, 20, 20], 
                    [21, 73, 192, 19, 20, 21], 
                    [40, 70, 178, 20, 19, 18], 
                    [30, 72, 213, 19, 15, 17], 
                    [27, 73, 192, 16, 12, 18]]    
            elif prev_years == 2:
                return [
                    [26, 72, 188, 20, 20], 
                    [20, 73, 192, 18, 21], 
                    [40, 70, 178, 20, 17], 
                    [30, 72, 213, 19, 15], 
                    [27, 73, 192, 14, 18]]  
            elif prev_years == 1:
                return [
                    [27, 72, 188, 20], 
                    [19, 73, 192, 19], 
                    [19, 70, 178, 18], 
                    [29, 72, 213, 17], 
                    [29, 73, 192, 16]]
        elif situation == 'PP':
            if prev_years == 4:
                return [
                [26, 72, 188, 0, 0, 0, 0], 
                [26, 73, 192, 1, 1, 1, 1], 
                [26, 70, 178, 2, 2, 2, 2], 
                [30, 72, 213, 0, 0, 1, 2], 
                [23, 73, 192, 0, 0, 1.75, 2]]
            elif prev_years == 3:
                return [
                [26, 72, 188, 0, 0, 0], 
                [26, 73, 192, 1, 1, 1], 
                [26, 70, 178, 2, 2, 2], 
                [30, 72, 213, 0, 1, 2], 
                [22, 73, 192, 0, 1.75, 2]]
            elif prev_years == 2:
                return [
                [26, 72, 188, 0, 0], 
                [26, 73, 192, 1, 1], 
                [26, 70, 178, 2, 2], 
                [30, 72, 213, 0.75, 2], 
                [21, 73, 192, 1.75, 2]]
            elif prev_years == 1:
                return [
                [26, 72, 188, 0], 
                [26, 73, 192, 1], 
                [26, 70, 178, 2], 
                [30, 72, 213, 0.5], 
                [20, 73, 192, 2]]
            
        elif situation == 'PK':
            if prev_years == 4:
                return [
                [26, 72, 188, 0, 0, 0, 0], 
                [26, 73, 192, 1, 1, 1, 1], 
                [26, 70, 178, 2, 2, 2, 2], 
                [30, 72, 213, 0, 0, 1, 2], 
                [23, 73, 192, 0, 0, 1.75, 2]]
            elif prev_years == 3:
                return [
                [26, 72, 188, 0, 0, 0], 
                [26, 73, 192, 1, 1, 1], 
                [26, 70, 178, 2, 2, 2], 
                [30, 72, 213, 0, 1, 2], 
                [22, 73, 192, 0, 1.75, 2]]
            elif prev_years == 2:
                return [
                [26, 72, 188, 0, 0], 
                [26, 73, 192, 1, 1], 
                [26, 70, 178, 2, 2], 
                [30, 72, 213, 0.75, 2], 
                [21, 73, 192, 1.75, 2]]
            elif prev_years == 1:
                return [
                [26, 72, 188, 0], 
                [26, 73, 192, 1], 
                [26, 70, 178, 2], 
                [30, 72, 213, 0.5], 
                [20, 73, 192, 2]]

def main():
    start = time.time()

    # Change these variables to change projection sets
    proj_stat = 'ATOI'
    position = 'defence' # [forward, defence]
    prev_years = 1 # [1, 2, 3, 4]
    situation = 'PK' # [EV, PP, PK, None] use None for projecting GP

    model_performance_df, model_list = test_models(proj_stat, position, prev_years, get_sample_projection(proj_stat, position, prev_years, situation), situation)
    print('\n', model_performance_df.to_string())
    recommend_model(model_performance_df, model_list)

    print(f'Results generated in {time.time()-start:.3f} seconds')

main()

# --- GAMES PLAYED MODEL ---
# Forwards with 4 seasons of > 50 GP: Parent model 1 (126-42-14-6-1), 5 epochs, standard scaler
# Forwards with 3 seasons of > 50 GP: Parent model 12 (8-1), 50 epochs, standard scaler
# Forwards with 2 seasons of > 50 GP: Parent model 6 (32-16-8-1), 50 epochs, minmax scaler
# Forwards with 1 season            : Parent model 6 (32-16-8-1), 100 epochs, minmax scaler

# Defence with 4 seasons of > 50 GP: Parent model 5 (64-28-12-1), 30 epochs, standard scaler
# Defence with 3 seasons of > 50 GP: Parent model 2 (64-32-16-8-1), 30 epochs, minmax scaler
# Defence with 2 seasons of > 50 GP: Parent model 10 (16-4-1), 10 epochs, standard scaler
# Defence with 1 season            : Parent model 7 (128-64-1), 50 epochs, minmax scaler

# --- EV ATOI MODEL ---
# Forwards with 4 seasons of > 40 GP: Parent model 5 (64-28-12-1), 10 epochs, standard scaler
# Forwards with 3 seasons of > 40 GP: Parent model 11 (24-1), 10 epochs, standard scaler
# Forwards with 2 seasons of > 40 GP: Parent model 3 (48-24-12-6-1), 10 epochs, standard scaler
# Forwards with 1 season            : Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler

# Defence with 4 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
# Defence with 3 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
# Defence with 2 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
# Defence with 1 season            : Parent model 11 (24-1), 50 epochs, standard scaler

# --- PP ATOI MODEL ---
# Forwards with 4 seasons of > 40 GP: Parent model 5 (64-28-12-1), 30 epochs, minmax scaler
# Forwards with 3 seasons of > 40 GP: Parent model 5 (64-28-12-1), 30 epochs, minmax scaler
# Forwards with 2 seasons of > 40 GP: Parent model 5 (64-28-12-1), 30 epochs, minmax scaler
# Forwards with 1 season            : Parent model 5 (64-28-12-1), 30 epochs, minmax scaler

# Defence with 4 seasons of > 40 GP: Parent model 5 (64-28-12-1), 30 epochs, minmax scaler
# Defence with 3 seasons of > 40 GP: Parent model 3 (48-24-12-6-1), 100 epochs, minmax scaler
# Defence with 2 seasons of > 40 GP: Parent model 1 (126-42-14-6-1), 30 epochs, standard scaler
# Defence with 1 season            : Parent model 10 (16-4-1), 50 epochs, standard scaler

# --- PK ATOI MODEL ---
# Forwards with 4 seasons of > 40 GP: Parent model 3 (48-24-12-6-1), 10 epochs, minmax scaler
# Forwards with 3 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
# Forwards with 2 seasons of > 40 GP: Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler
# Forwards with 1 season            : Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler

# Defence with 4 seasons of > 40 GP: Parent model 7 (128-64-1), 5 epochs, standard scaler
# Defence with 3 seasons of > 40 GP: Parent model 7 (128-64-1), 5 epochs, minmax scaler
# Defence with 2 seasons of > 40 GP: Parent model 4 (256-64-16-1), 10 epochs, minmax scaler
# Defence with 1 season            : Parent model 9 (36-12-1), 5 epochs, minmax scaler
