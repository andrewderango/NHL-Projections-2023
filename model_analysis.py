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
                        round(float(proj_y[4] + sum(proj_x[4][-prev_years:])/prev_years), 2)
                        ]
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
                        round(float(proj_y[4]), 2)
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
                        [26, 73, 193, 1.45, 1.41, 1.62], # 1.49
                        [26, 73, 193, 1.97, 2.08, 1.32], # 1.60
                        [26, 73, 193, 1.65, 1.39, 2.01], # 1.60
                        [26, 73, 193, 0.58, 0.42, 0.46], # 0.50
                        [26, 73, 193, 0.61, 0.71, 0.42]] # 0.58
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
                            [22.383, 71, 175, 1.233, 0.3627, 2.9989, 2.2033, 1.1787, 1.0143, 1.4203, 2.2632, 0.2353, 0.6688, 1.4058, 1.5606, 0.7673, 0.7557, 1.0782, 1.3243], # jack hughes: 2.75
                            [31.404, 69, 186, 0.4993, 2.6608, 2.2452, 0.0, 1.0966, 1.9479, 2.1937, 0.6712, 1.4455, 1.457, 0.1709, 1.0449, 1.3279, 1.1384, 0.9523, 1.0289], # brendan gallagher: 1.25
                            [29.134, 73, 205, 2.2843, 1.5553, 2.7517, 1.5492, 2.1043, 1.5967, 2.2656, 2.8676, 0.9547, 0.9064, 1.8681, 1.2561, 1.1555, 1.0381, 1.3305, 1.3137]] # filip forsberg: 2.00
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

def main():
    start = time.time()

    # Change these variables to change projection sets
    proj_stat = 'Gper60'
    position = 'forward' # [forward, defence]
    prev_years = 2 # [1, 2, 3, 4]
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
# Forwards with 4 seasons of > 50 PPTOI: Parent model 6 (32-16-8-1), 10 epochs, standard scaler --> 65, 143, 139
# Forwards with 3 seasons of > 50 PPTOI: Parent model 4 (256-64-16-1), 5 epochs, standard scaler
# Forwards with 2 seasons of > 50 PPTOI: Parent model 4 (256-64-16-1), 5 epochs, minmax scaler
# Forwards with 1 seasons of > 50 PPTOI: Parent model 6 (32-16-8-1), 10 epochs, minmax scaler

# Defence with 4 seasons of > 50 PPTOI: Parent model 10 (16-4-1), 5 epochs, minmax scaler
# Defence with 3 seasons of > 50 PPTOI: Parent model 6 (32-16-8-1), 5 epochs, minmax scaler
# Defence with 2 seasons of > 50 PPTOI: Parent model 5 (64-28-12-1), 5 epochs, minmax scaler
# Defence with 1 seasons of > 50 PPTOI: Parent model 11 (24-1), 10 epochs, standard scaler
