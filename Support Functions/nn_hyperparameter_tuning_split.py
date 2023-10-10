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

                # because algorithm for forward and defence is different for Gper60
                if proj_stat == 'Gper60':
                    X, y = preprocessing_training_functions.extract_instance_data(instance_df, proj_stat, prev_years, situation, position)
                else:
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
                        round(float(proj_y[4] + sum(proj_x[4][-prev_years:])/prev_years), 2)]
                else:
                    model_performance_df.loc[model_index*len(epoch_list)*len(scaler_list) + epoch_index*len(scaler_list) + scaler_index + 1] = [
                        int(model_index*len(epoch_list)*len(scaler_list) + epoch_index*len(scaler_list) + scaler_index + 1), 
                        int(model_index+1), 
                        int(epochs), 
                        scaler, 
                        round(test_loss, 2), 
                        round(train_loss, 2),
                        round(float(proj_y[0]), 2),
                        round(float(proj_y[1]), 2),
                        round(float(proj_y[2]), 2),
                        round(float(proj_y[3]), 2),
                        round(float(proj_y[4]), 2)]

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
            
    elif proj_stat == 'Gper60':
        if situation == 'EV':
            if position == 'forward':
                if prev_years == 4:
                    return [
                        [27, 73, 193, 1.22, 1.45, 1.41, 1.62], # connor mcdavid: 1.45
                        [26, 75, 208, 1.70, 1.97, 2.08, 1.32], # auston matthews: 1.59
                        [27, 73, 195, 1.60, 1.65, 1.39, 2.01], # david pastrnak: 1.60
                        [27, 74, 200, 0.66, 0.58, 0.42, 0.46], # jason dickinson: 0.51
                        [29, 71, 186, 0.56, 0.61, 0.71, 0.42]] # alexander kerfoot: 0.56
                elif prev_years == 3:
                    return [
                        [25.43, 75, 200, 0.9705, 1.1698, 0.3984, 0.875, 0.7256, 0.9344], # drake batherson: 0.80
                        [24.195, 75, 200, 1.2897, 1.4714, 1.5589, 0.7451, 1.0287, 1.0204], # jason robertson: 1.55
                        [26.432, 70, 202, 1.3827, 1.5543, 1.2003, 0.8337, 0.9854, 1.0043], # kirill kaprizov: 1.35
                        [25.92, 78, 220, 0.7561, 1.3902, 1.3696, 1.1323, 0.9071, 1.1983], # tage thompson: 1.30
                        [21.709, 72, 193, 0.7019, 0.7089, 1.2029, 0.6068, 0.7761, 0.9998]] # tim stutzle: 1.10
                elif prev_years == 2:
                    return [
                        [26, 73, 193, 0.5, 0.5, 0.5, 0.5],
                        [26, 73, 193, 0.5, 0.3, 0.5, 0.5],
                        [26, 73, 193, 0.5, 0.3, 0.5, 0.3],
                        [26, 73, 193, 0.5, 0.5, 0.3, 0.3],
                        [26, 73, 193, 0.2, 0.2, 0.2, 0.2]]
                elif prev_years == 1:
                    return [
                        [25.487, 67, 140, 0.4192913317937487, 0.5419646957532721], # matthew phillips: 0.60
                        [34.862, 74, 217, 1.5804663043478262, 1.2829826086956522], # max pacioretty: 1.50
                        [20.904, 74, 178, 1.039, 0.8543], # matty beniers: 1.15
                        [27.059, 72, 180, 0.4813, 1.3146], # michael eyssimont: 0.70
                        [24.409, 74, 195, 1.1660837837837839, 0.7956459459459461]] # josh norris: 0.88
            elif position == 'defence':
                    if prev_years == 4:
                        return [
                            [25, 71, 187, 0.49, 0.28, 0.73, 0.61, 0.29, 0.22, 0.41, 0.42], # cale makar
                            [32, 73, 201, 0.51, 0.43, 0.42, 0.43, 0.41, 0.35, 0.36, 0.47], # roman josi
                            [24, 75, 202, 0.13, 0.23, 0.37, 0.33, 0.16, 0.24, 0.23, 0.44], # rasmus dahlin
                            [30, 72, 194, 0.26, 0.58, 0.47, 0.51, 0.30, 0.26, 0.39, 0.43], # brandon montour
                            [36, 75, 209, 0.31, 0.48, 0.29, 0.17, 0.25, 0.28, 0.20, 0.24]] # jeff petry
                    elif prev_years == 3:
                        return [
                            [26, 73, 193, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3],
                            [26, 73, 193, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4],
                            [26, 73, 193, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3]]
                    elif prev_years == 2:
                        return [
                            [26, 73, 193, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.3, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.3, 0.5, 0.3],
                            [26, 73, 193, 0.5, 0.5, 0.3, 0.3],
                            [26, 73, 193, 0.2, 0.2, 0.2, 0.2]]
                    elif prev_years == 1:
                        return [
                            [28, 74, 204, 0.38, 0.45], # mark friedman
                            [31, 72, 204, 0.00, 0.39], # jacob macdonald
                            [21, 78, 218, 0.15, 0.27], # owen power
                            [28, 74, 215, 0.51, 0.33], # jake walman
                            [27, 71, 174, 0.12, 0.25]] # alexandre carrier
        elif situation == 'PP':
            if position == 'forward':
                    if prev_years == 4:
                        return [
                            [27.352, 73, 195, 4.538, 2.1561, 3.5188, 3.4499, 3.356, 2.2985, 3.6623, 3.9301, 1.5971, 1.1623, 1.3946, 2.0056, 1.4408, 1.3076, 1.6164, 1.9446], # david pastrnak: 3.55
                            [27.93, 74, 208, 3.5314, 3.8122, 4.6903, 6.0593, 3.6966, 4.3858, 4.4811, 5.1512, 1.2889, 0.9313, 1.2828, 0.833, 1.3472, 0.9353, 1.3679, 0.976], # leon draisaitl: 4.75
                            [22.383, 71, 175, 1.233, 0.3627, 2.9989, 2.2033, 1.1787, 1.0143, 1.4203, 2.2632, 0.2353, 0.6688, 1.4058, 1.5606, 0.7673, 0.7557, 1.0782, 1.3243], # jack hughes: 2.50
                            [31.404, 69, 186, 0.4993, 2.6608, 2.2452, 0.0, 1.0966, 1.9479, 2.1937, 0.6712, 1.4455, 1.457, 0.1709, 1.0449, 1.3279, 1.1384, 0.9523, 1.0289], # brendan gallagher: 1.25
                            [25.805, 74, 201, 1.5186, 1.338, 2.996, 2.3991, 1.8493, 2.3453, 3.2629, 3.5936, 1.0231, 0.8681, 1.4726, 1.2651, 0.959, 0.8779, 1.195, 1.2851]] # matthew tkachuk: 2.50
                    elif prev_years == 3:
                        return [
                            [28, 74, 210, 3.81, 4.69, 6.06, 4.39, 4.48, 5.15, 0.93, 1.28, 0.83, 0.94, 1.35, 0.97],
                            [26, 73, 193, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
                    elif prev_years == 2:
                        return [
                            [28, 74, 210, 4.69, 6.06, 4.48, 5.15, 1.28, 0.83, 1.35, 0.97],
                            [26, 73, 193, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.3, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.5, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5]]
                    elif prev_years == 1:
                        return [
                            [28, 74, 210, 6.06, 5.15, 0.83, 0.97],
                            [31, 72, 204, 0.00, 0.39, 0.38, 0.45],
                            [21, 78, 218, 0.15, 0.27, 0.38, 0.45],
                            [28, 74, 215, 0.51, 0.33, 0.38, 0.45],
                            [27, 71, 174, 0.12, 0.25, 0.38, 0.45]]
            elif position == 'defence':
                    if prev_years == 4:
                        return [
                            [24.921, 71, 187, 1.1248, 1.3155, 1.8365, 1.2439, 1.1011, 1.0671, 0.8239, 0.9716, 0.4944, 0.2843, 0.7252, 0.6118, 0.2911, 0.2226, 0.4169, 0.4268], # cale makar: 1.20
                            [23.467, 75, 202, 0.609, 0.3516, 0.7326, 1.2758, 0.7896, 0.7115, 0.5438, 0.7935, 0.1287, 0.2325, 0.3743, 0.3319, 0.1604, 0.2465, 0.2363, 0.4447], # rasmus dahlin: 0.88
                            [29.564, 73, 222, 0.0, 0.3729, 0.5027, 0.3059, 0.8279, 0.5468, 0.5097, 0.5738, 0.1935, 0.2155, 0.3067, 0.1564, 0.261, 0.1811, 0.2797, 0.2959], # morgan reilly: 0.40
                            [33.334, 73, 201, 1.0471, 0.4141, 2.426, 2.1189, 1.2427, 0.9322, 1.305, 1.6115, 0.5114, 0.4313, 0.4204, 0.4294, 0.4626, 0.392, 0.4002, 0.5283], # roman josi: 1.80
                            [33.337, 72, 190, 0.3575, 1.1811, 0.0, 1.0814, 0.5936, 0.7478, 1.0192, 0.974, 0.2268, 0.2978, 0.6317, 0.6696, 0.2653, 0.3416, 0.4346, 0.3879]] # erik karlsson: 1.00
                    elif prev_years == 3:
                        return [
                            [23.732, 76, 194, 0.0, 0.8544, 1.1703, 0.6474, 0.8022, 1.1711, 0.273, 0.4148, 0.371, 0.1916, 0.2425, 0.3009], # noah dobson: 0.70
                            [24.08, 70, 185, 1.1465, 0.0, 0.0, 0.6645, 0.2868, 0.2158, 0.1429, 0.0, 0.1171, 0.1932, 0.1979, 0.1966], # erik brannstrom: 0.20
                            [26, 73, 193, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [26, 73, 193, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
                    elif prev_years == 2:
                        return [
                            [23.949, 75, 194, 1.1168, 0.6446, 1.187, 0.8949, 0.4416, 0.2797, 0.3207, 0.2792], # evan bouchard: 0.90
                            [22.487, 76, 204, 0.501, 0.236, 0.8788, 0.6119, 0.2016, 0.1258, 0.2545, 0.1736], # moritz seider: 0.40
                            [23.568, 71, 182, 0.0, 0.0, 0.3356, 0.2306, 0.3802, 0.3435, 0.1965, 0.1701], # rasmus sandin: 0.20
                            [32.193, 71, 214, 1.6194, 0.6087, 1.3864, 0.4806, 0.4107, 0.2875, 0.2021, 0.2865], # dmitry orlov: 0.75
                            [29.474, 72, 194, 0.5348, 0.5302, 0.5383, 1.0133, 0.466, 0.5133, 0.3984, 0.4412]] # brandon montour: 0.70
                    elif prev_years == 1:
                        return [
                            [20.858, 78, 218, 0.0, 0.1918, 0.1469, 0.2782], # owen power: 0.38
                            [22.3, 73, 190, 1.2632411150018934, 0.7874376414530464, 0.3097950938911917, 0.2271180302713942], # bowen byram: 0.50
                            [28.929, 76, 221, 0.6311066023257405, 0.9490117214791752, 0.1312889093424077, 0.13604532143289172], # rasmus ristolainen: 0.30
                            [30.388, 78, 228, 1.21121469647627, 0.7875929272127551, 0.21725032355549223, 0.19505978794684536], # colton parayko: 0.45
                            [36.716, 73, 227, 0.1640974068352275, 0.4598238732131714, 0.09298721421197276, 0.12818205618623077]] # jack johnson: 0.30
    elif proj_stat == 'A1per60' or proj_stat == 'A2per60':
        if situation == 'EV':
            if position == 'forward' or position == 'defence':
                if prev_years == 4:
                    return [
                        [26.714, 73, 193, 1.0096, 2.0484, 1.1602, 1.1498, 0.6376, 0.6025, 0.663, 0.3833, 0.7439, 1.0242, 0.953, 1.0221, 0.797, 0.3615, 0.4144, 0.511, 2.9103, 3.5739, 4.1022, 4.0091], # connor mcdavid: 1.20
                        [26.038, 75, 208, 0.6327, 0.9556, 0.8526, 1.0756, 0.3407, 0.446, 0.7105, 0.3911, 1.2167, 1.083, 1.563, 1.9068, 0.3407, 0.1911, 0.7578, 0.44, 3.2111, 3.3721, 3.7825, 3.7872], # auston matthews: 0.97
                        [26.922, 76, 215, 1.1135, 0.7985, 1.0357, 0.7315, 0.3712, 0.7259, 0.7061, 0.3251, 0.7424, 1.234, 0.9886, 1.1379, 0.2784, 0.2178, 0.4237, 0.447, 2.9991, 3.2027, 3.5448, 3.4312], # mikko rantanen: 0.88
                        [28.083, 72, 200, 1.3881, 1.6173, 1.4402, 1.5858, 0.6169, 0.5391, 0.4801, 0.6063, 1.4395, 1.1552, 1.3869, 1.2127, 0.5655, 0.6161, 0.4801, 0.6996, 3.1906, 3.4641, 3.4944, 3.3951], # nathan mackinnon: 1.50
                        [28.578, 76, 210, 0.4665, 0.3716, 0.9073, 0.9114, 0.622, 0.5574, 0.5833, 0.7595, 0.622, 0.7432, 0.7129, 0.8355, 0.4665, 0.0929, 0.7129, 0.6076, 2.6871, 3.1037, 3.3784, 2.9151]] # valeri nichushkin: 0.85
                elif prev_years == 3:
                    return [
                        [26.432, 70, 202, 1.0188, 1.4601, 0.6784, 0.3639, 0.6123, 0.3653, 0.946, 0.942, 1.096, 0.2911, 0.8007, 0.3131, 2.4589, 2.7746, 3.0311], # kirill kaprizov: 0.80
                        [25.096, 74, 179, 0.8459, 0.6963, 0.7766, 0.0, 0.1741, 0.5695, 0.6579, 0.4062, 0.6212, 0.6579, 0.4642, 1.5531, 2.1279, 2.7033, 3.5721], # brandon hagel: 0.77
                        [24.195, 75, 200, 1.4509, 0.7357, 1.0865, 0.403, 0.8408, 0.5669, 0.8061, 1.524, 1.6534, 0.3224, 0.3679, 0.2834, 2.7285, 3.5151, 3.7569], # jason robertson: 1.05
                        [21.709, 72, 193, 0.6142, 0.6076, 1.0585, 0.4387, 0.2532, 0.5293, 1.0528, 1.0127, 0.9142, 0.2632, 0.3545, 0.6255, 2.3461, 2.71, 3.7828], # tim stutzle: 0.87
                        [25.92, 78, 220, 0.378, 0.6416, 1.1062, 0.252, 0.6416, 0.6321, 1.1341, 1.1229, 1.9491, 0.6301, 0.6416, 0.6848, 2.4951, 2.9013, 3.677]] # tage thompson: 0.85
                elif prev_years == 2:
                    return [
                        [30.29, 71, 181, 1.1928, 1.0457, 0.6362, 0.8183, 0.7952, 2.0913, 2.0675, 2.3186, 3.2707, 3.8053], # nikita kucherov
                        [24.74, 73, 175, 0.9858, 1.2503, 0.6958, 0.6252, 0.5799, 1.5108, 0.4639, 1.5629, 2.9544, 4.075], # nico hischier
                        [24.885, 74, 176, 0.5608, 1.1564, 0.4079, 0.8547, 0.9177, 1.2067, 0.3059, 0.2011, 3.0265, 3.0197], # elias pettersson
                        [22.744, 67, 174, 0.4551, 0.1769, 0.3251, 0.4422, 0.9752, 0.8844, 0.7151, 1.3266, 2.5822, 2.4251], # cole caufield
                        [22.533, 72, 185, 0.894, 0.8815, 0.6836, 0.464, 0.894, 0.9279, 0.4207, 0.232, 2.8103, 2.6595]] # trevor zegras
                elif prev_years == 1:
                    return [
                        [34.862, 74, 217, 0.698145652173913, 0.6566967391304348, 1.0764423913043477, 0.7464489130434783, 3.0651249999999997], # max pacioretty
                        [31.404, 69, 186, 0.7738376344086021, 0.20581505376344084, 2.067366666666667, 0.5165989247311829, 2.8682086021505375], # brendan gallagher
                        [35.856, 73, 202, 0.6610058139534885, 0.4068732558139535, 0.9147267441860465, 0.15246162790697676, 2.6094558139534887], # nicklas backstrom
                        [20.904, 74, 178, 0.7273, 0.6234, 0.4676, 0.6234, 2.729], # matty beniers
                        [19.505, 75, 238, 0.18343551733138838, 0.48495865244833847, 0.9665146689883484, 0.26910138919576726, 1.96986599505593]] # juraj slafkovsky
        if situation == 'PP':
            if position == 'forward' or position == 'defence':
                if prev_years == 4:
                    return [
                        [26.714, 73, 193, 1.0096, 2.0484, 1.1602, 1.1498, 0.6376, 0.6025, 0.663, 0.3833, 4.5808, 4.1055, 4.0286, 5.5921, 3.1342, 3.0791, 2.8201, 3.7281, 1.4466, 1.7962, 3.2229, 2.0504, 8.5781, 8.9756, 10.3335, 11.4694], # connor mcdavid: 4.45
                        [26.038, 75, 208, 0.6327, 0.9556, 0.8526, 1.0756, 0.3407, 0.446, 0.7105, 0.3911, 1.5861, 0.0, 2.1135, 1.5654, 1.8505, 1.0669, 1.3209, 1.789, 1.0574, 2.1338, 1.8493, 2.4599, 7.1931, 8.3857, 9.92, 11.4208], # auston matthews: 1.66
                        [33.03, 73, 216, 0.8705, 1.0366, 0.6063, 0.6351, 0.2902, 0.6911, 1.0611, 0.5821, 2.827, 1.2363, 2.2807, 3.2585, 1.5705, 1.2363, 1.7739, 1.3034, 1.5705, 1.2363, 2.2807, 5.6481, 7.5983, 8.8681, 10.1619, 11.4439], # john tavares: 2.80
                        [28.083, 72, 200, 1.3881, 1.6173, 1.4402, 1.5858, 0.6169, 0.5391, 0.4801, 0.6063, 2.1148, 2.1487, 2.7766, 1.6524, 1.9033, 3.0696, 2.2718, 2.8917, 1.6918, 3.9905, 3.029, 2.0655, 7.203, 9.7367, 9.3395, 9.1647], # nathan mackinnon: 2.20
                        [22.383, 71, 175, 0.549, 0.6688, 1.0358, 1.0557, 0.1569, 0.2675, 0.5919, 0.5049, 0.6165, 1.4507, 2.9989, 2.9378, 0.9247, 0.7254, 0.4284, 2.4481, 0.6165, 1.4507, 0.4284, 1.2241, 7.8911, 6.3216, 7.0518, 9.5478]] # jack hughes: 3.05
                elif prev_years == 3:
                    return [
                        [26.432, 70, 202, 1.0188, 1.4601, 0.6784, 0.3639, 0.6123, 0.3653, 1.0197, 1.7889, 2.9822, 0.6798, 2.0125, 0.4588, 0.6798, 1.7889, 3.441, 7.0022, 7.6007, 10.0019], # kirill kaprizov: 2.35
                        [31.801, 73, 228, 0.8398, 0.8472, 0.6669, 0.168, 0.8472, 0.6063, 1.9934, 3.5928, 1.7871, 1.9934, 1.497, 2.1446, 0.0, 1.1976, 1.4297, 7.8339, 7.0599, 7.5167], # vladimir tarasenko: 1.87
                        [24.195, 75, 200, 1.4509, 0.7357, 1.0865, 0.403, 0.8408, 0.5669, 1.5986, 1.5153, 3.5143, 1.0657, 0.9092, 2.6358, 1.5986, 1.8183, 4.1733, 6.8739, 8.2128, 10.2641], # jason robertson: 2.50
                        [21.709, 72, 193, 0.6142, 0.6076, 1.0585, 0.4387, 0.2532, 0.5293, 0.8039, 2.3438, 1.5329, 1.2059, 2.1094, 1.9161, 1.2059, 1.875, 2.2993, 6.1903, 8.8664, 10.2206], # tim stutzle: 1.50
                        [22.533, 72, 185, 1.5196, 0.894, 0.8815, 0.3799, 0.6836, 0.464, 0.0, 1.2193, 1.3124, 0.0, 1.2193, 2.0999, 0.0, 0.9145, 0.7875, 6.8407, 5.7155, 8.4416]] # trevor zegras: 1.40
                elif prev_years == 2:
                    return [
                        [30.29, 71, 181, 1.1928, 1.0457, 0.6362, 0.8183, 3.9801, 4.569, 2.9851, 3.1069, 1.99, 1.4621, 10.2488, 9.0813], # nikita kucherov: 4.00
                        [28.578, 76, 210, 0.9073, 0.9114, 0.5833, 0.7595, 0.0, 0.3176, 1.6707, 1.5879, 0.8354, 3.1757, 7.1674, 9.3811], # valeri nichushkin: 0.50
                        [21.928, 72, 180, 0.9179, 0.6819, 0.102, 0.3672, 0.7895, 2.7972, 0.7895, 1.6783, 1.1842, 1.6783, 5.325, 8.649], # dawson mercer: 2.80
                        [22.744, 67, 174, 0.4551, 0.1769, 0.3251, 0.4422, 1.5766, 0.0, 0.946, 1.1987, 2.2072, 1.9978, 6.6564, 5.6897], # cole caufield: 1.00
                        [22.489, 74, 201, 0.9831, 0.3567, 0.6881, 0.3567, 2.1654, 1.9514, 1.6241, 1.5611, 1.0827, 2.5369, 7.2162, 9.9777]] # matt boldy: 2.10
                elif prev_years == 1:
                    return [
                        [20.904, 74, 178, 0.7273, 0.6234, 0.958, 0.958, 0.6386, 6.5524], # matty beniers: 1.30
                        [20.953, 72, 175, 0.7617138594628957, 0.2196129906466873, 1.5696898333984095, 1.9620867069942844, 1.9620662705848417, 7.012404932317491], # kent johnson: 2.35
                        [27.355, 71, 179, 0.6223323474261325, 0.25054190696949846, 1.2504267766831358, 0.31263752366036607, 1.2504400511840212, 5.708204219659896], # pius suter: 1.20
                        [24.102, 71, 220, 0.647167562835548, 0.33023935137003824, 1.016688086423377, 0.15751235475510675, 1.0172134680576337, 5.745427775266393], # fabian zetterlund: 1.05
                        [31.856, 75, 208, 0.399877124472204, 0.2651930650348834, 0.38764285779136304, 0.2817472021371184, 0.388582624108337, 5.086026044587531]] # garnet hathaway: 0.35
                
def main():
    start = time.time()

    # Change these variables to change projection sets
    proj_stat = 'Gper60'
    position = 'forward' # [forward, defence]
    prev_years = 3 # [1, 2, 3, 4]
    situation = 'EV' # [EV, PP, PK, None] use None for projecting GP

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

# --- EV G/60 MODEL ---
# Forwards with 4 seasons of > 50 GP: Parent model 6 (32-16-8-1), 30 epochs, minmax scaler
# Forwards with 3 seasons of > 50 GP: Parent model 2 (64-32-16-8-1), 5 epochs, standard scaler
# Forwards with 2 seasons of > 50 GP: Parent model 1 (126-42-14-6-1), 1 epoch, standard scaler
# Forwards with 1 seasons of > 50 GP: Parent model 10 (16-4-1), 10 epoch, standard scaler

# Defence with 4 seasons of > 50 GP: Parent model 2 (64-32-16-8-1), 10 epochs, standard scaler
# Defence with 3 seasons of > 50 GP: Parent model 2 (64-32-16-8-1), 5 epochs, standard scaler
# Defence with 2 seasons of > 50 GP: Parent model 1 (126-42-14-6-1), 1 epoch, standard scaler
# Defence with 1 seasons of > 50 GP: Parent model 4 (256-64-16-1), 5 epochs, minmax scaler

# --- PP G/60 MODEL ---
# Forwards with 4 seasons of > 50 PPTOI: Parent model 2 (64-32-16-8-1), 5 epochs, minmax scaler --> (65, 143, 139)
# Forwards with 3 seasons of > 50 PPTOI: Parent model 4 (256-64-16-1), 5 epochs, standard scaler
# Forwards with 2 seasons of > 50 PPTOI: Parent model 4 (256-64-16-1), 5 epochs, minmax scaler
# Forwards with 1 seasons of > 50 PPTOI: Parent model 6 (32-16-8-1), 10 epochs, minmax scaler

# Defence with 4 seasons of > 50 PPTOI: Parent model 10 (16-4-1), 5 epochs, minmax scaler
# Defence with 3 seasons of > 50 PPTOI: Parent model 6 (32-16-8-1), 5 epochs, minmax scaler
# Defence with 2 seasons of > 50 PPTOI: Parent model 5 (64-28-12-1), 5 epochs, minmax scaler
# Defence with 1 seasons of > 50 PPTOI: Parent model 11 (24-1), 10 epochs, standard scaler

# --- PK G/60 MODEL ---
# Forwards with 4 seasons of > 50 PKTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler
# Forwards with 3 seasons of > 50 PKTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler
# Forwards with 2 seasons of > 50 PKTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler
# Forwards with 1 seasons of > 50 PKTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler

# Defence with 4 seasons of > 50 PKTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler
# Defence with 3 seasons of > 50 PKTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler
# Defence with 2 seasons of > 50 PKTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler
# Defence with 1 seasons of > 50 PKTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler

# --- EV A1/60 MODEL ---
# Forwards with 4 seasons of > 50 GP: Parent model 10 (16-4-1), 50 epochs, minmax scaler
# Forwards with 3 seasons of > 50 GP: Parent model 2 (64-32-16-8-1), 5 epochs, minmax scaler
# Forwards with 2 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler
# Forwards with 1 seasons of > 50 GP: Parent model 9 (36-12-1), 5 epochs, minmax scaler

# Defence with 4 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler
# Defence with 3 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 10 epochs, minmax scaler
# Defence with 2 seasons of > 50 GP: Parent model 6 (32-16-8-1), 5 epochs, minmax scaler (P)
# Defence with 1 seasons of > 50 GP: Parent model 9 (36-12-1), 5 epochs, minmax scaler (P)

# --- PP A1/60 MODEL ---
# Forwards with 4 seasons of > 50 PPTOI: Parent model 7 (128-64-1), 1 epochs, standard scaler
# Forwards with 3 seasons of > 50 PPTOI: Parent model 12 (8-1), 10 epochs, standard scaler
# Forwards with 2 seasons of > 50 PPTOI: Parent model 5 (64-28-12-1), 30 epochs, standard scaler
# Forwards with 2 seasons of > 50 PPTOI: Parent model 4 (256-64-16-1), 5 epochs, minmax scaler

# Defence with 4 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler (P)
# Defence with 3 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 10 epochs, minmax scaler (P)
# Defence with 2 seasons of > 50 GP: Parent model 6 (32-16-8-1), 5 epochs, minmax scaler (P)
# Defence with 1 seasons of > 50 GP: Parent model 9 (36-12-1), 5 epochs, minmax scaler (P)

# --- PK A1/60 MODEL ---
# Forwards with 4 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Forwards with 3 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Forwards with 2 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Forwards with 1 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)

# Defence with 4 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Defence with 3 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Defence with 2 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Defence with 1 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)

# --- EV A2/60 MODEL ---
# Forwards with 4 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 5 epochs, minmax scaler
# Forwards with 3 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 5 epochs, minmax scaler
# Forwards with 2 seasons of > 50 GP: Parent model 9 (36-12-1), 10 epochs, minmax scaler (P)
# Forwards with 1 seasons of > 50 GP: Parent model 9 (36-12-1), 5 epochs, minmax scaler (P)

# Defence with 4 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler (P)
# Defence with 3 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 10 epochs, minmax scaler (P)
# Defence with 2 seasons of > 50 GP: Parent model 6 (32-16-8-1), 5 epochs, minmax scaler (P)
# Defence with 1 seasons of > 50 GP: Parent model 9 (36-12-1), 5 epochs, minmax scaler (P)

# --- PP A2/60 MODEL ---
# Forwards with 4 seasons of > 50 PPTOI: Parent model 7 (128-64-1), 1 epochs, standard scaler (P)
# Forwards with 3 seasons of > 50 PPTOI: Parent model 12 (8-1), 10 epochs, standard scaler (P)
# Forwards with 2 seasons of > 50 PPTOI: Parent model 5 (64-28-12-1), 30 epochs, standard scaler (P)
# Forwards with 2 seasons of > 50 PPTOI: Parent model 4 (256-64-16-1), 5 epochs, minmax scaler (P)

# Defence with 4 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 30 epochs, minmax scaler (P)
# Defence with 3 seasons of > 50 GP: Parent model 3 (48-24-12-6-1), 10 epochs, minmax scaler (P)
# Defence with 2 seasons of > 50 GP: Parent model 6 (32-16-8-1), 5 epochs, minmax scaler (P)
# Defence with 1 seasons of > 50 GP: Parent model 9 (36-12-1), 5 epochs, minmax scaler (P)

# --- PK A2/60 MODEL ---
# Forwards with 4 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Forwards with 3 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Forwards with 2 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Forwards with 1 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)

# Defence with 4 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Defence with 3 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Defence with 2 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
# Defence with 1 seasons of > 50 PKTOI: Parent model 10 (16-4-1), 10 epochs, minmax scaler (P)
