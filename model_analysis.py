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

def test_models(proj_stat, position, prev_years, proj_x, download_instance_file=True, download_model_analysis_file=True):
    epoch_list = [1, 5, 10, 30, 50, 100]
    scaler_list = [StandardScaler(), MinMaxScaler()]

    model_performance_df = pd.DataFrame(columns=['Model ID', 'Parent Model ID', 'Epochs', 'Scaler', 'MAE Test', 'MAE Train', 'Proj. 1', 'Proj. 2', 'Proj. 3', 'Proj. 4', 'Proj. 5'])
    stat_df = preprocessing_training_functions.scrape_player_statistics(True)

    if proj_stat == 'GP':
        instance_df = preprocessing_training_functions.create_instance_df(f'{position}_GP', ['Player', 'Year', 'Position', 'Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP', 'Y5 GP', 'Y5 dGP'], stat_df, True)
        # instance_df = pd.read_csv(f'{os.path.dirname(__file__)}/CSV Data/forward_GP_instance_training_data.csv')
        if prev_years == 4:
            instance_df = instance_df.loc[(instance_df['Y1 GP'] >= 60) & (instance_df['Y2 GP'] >= 60) & (instance_df['Y3 GP'] >= 60) & (instance_df['Y4 GP'] >= 60)]
            input_shape = (7,)
        elif prev_years == 3:
            instance_df = instance_df.loc[(instance_df['Y2 GP'] >= 60) & (instance_df['Y3 GP'] >= 60) & (instance_df['Y4 GP'] >= 60)]
            input_shape = (6,)
        elif prev_years == 2:
            instance_df = instance_df.loc[(instance_df['Y3 GP'] >= 60) & (instance_df['Y4 GP'] >= 60)]
            input_shape = (5,)
        elif prev_years == 1:
            instance_df = instance_df.loc[(instance_df['Y4 GP'] >= 40)]
            input_shape = (4,)
        else:
            print('Invalid prev_years parameter.')
    
    print(instance_df)
    model_list = create_models(input_shape)

    if download_instance_file == True:
        filename = f'{position}_{proj_stat}_{prev_years}year_instance_training_data'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        instance_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    print(f'Models to Test: {len(model_list) * len(epoch_list) * len(scaler_list)}')
    for model_index, model in enumerate(model_list):
        for epoch_index, epochs in enumerate(epoch_list):
            for scaler_index, scaler in enumerate(scaler_list):
                model.compile(optimizer='adam', loss='MeanAbsoluteError', metrics=['mean_squared_error', 'MeanSquaredLogarithmicError'])

                X = []
                y = []

                if proj_stat == 'GP':
                    if prev_years == 4:
                        for index, row in instance_df.iterrows():
                            X.append([row['Age'], row['Height'], row['Weight'], row['Y1 GP'], row['Y2 GP'], row['Y3 GP'], row['Y4 GP']]) # features
                            y.append(row['Y5 dGP']) # target
                    elif prev_years == 3:
                        for index, row in instance_df.iterrows():
                            X.append([row['Age'], row['Height'], row['Weight'], row['Y2 GP'], row['Y3 GP'], row['Y4 GP']]) # features
                            y.append(row['Y5 dGP']) # target
                    elif prev_years == 2:
                        for index, row in instance_df.iterrows():
                            X.append([row['Age'], row['Height'], row['Weight'], row['Y3 GP'], row['Y4 GP']]) # features
                            y.append(row['Y5 dGP']) # target
                    elif prev_years == 1:
                        for index, row in instance_df.iterrows():
                            X.append([row['Age'], row['Height'], row['Weight'], row['Y4 GP']]) # features
                            y.append(row['Y5 dGP']) # target
                    else:
                        print('Invalid prev_years parameter.')
                        
                X = np.array(X)
                y = np.array(y)

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

                if proj_stat == 'GP':
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
    model_performance_df = model_performance_df.reset_index()
    model_performance_df.index += 1

    if download_model_analysis_file == True:
        filename = f'{position}_{proj_stat}_{prev_years}year_model_analysis'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        instance_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return model_performance_df, model_list


def recommend_model(model_performance_df, model_list):
    recommended_model = model_performance_df.iloc[0]
    print('\n\n--RECOMMENDED MODEL--')
    print(f'Model #{recommended_model["Model ID"]}: Parent Model {recommended_model["Parent Model ID"]} with {recommended_model["Epochs"]} epochs and {recommended_model["Scaler"]}')
    print(f'This model gave a MAE of {recommended_model["MAE Test"]} on the test and {recommended_model["MAE Train"]} on the train.')
    print(f'Parent Model {recommended_model["Parent Model ID"]} architecture:')
    model_list[recommended_model["Parent Model ID"] - 1].summary()

def get_sample_projection(proj_stat, position, prev_years):
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

def main():
    start = time.time()

    # Change these variables to change projection sets
    proj_stat = 'GP'
    position = 'forward' # [forward, defence]
    prev_years = 1 # [1, 2, 3, 4]

    model_performance_df, model_list = test_models(proj_stat, position, prev_years, get_sample_projection(proj_stat, position, prev_years))
    print('\n', model_performance_df.to_string())
    recommend_model(model_performance_df, model_list)

    print(f'Results generated in {time.time()-start:.3f} seconds')

main()

# Forwards with 4 seasons of > 50 GP: Parent model 6 (32-16-8-1), 5 epochs, standard scaler
# Forwards with 3 seasons of > 50 GP: Parent model 12 (8-1), 50 epochs, standard scaler
# Forwards with 2 seasons of > 50 GP: Parent model 6 (32-16-8-1), 50 epochs, minmax scaler
# Forwards with 1 seasons of > 50 GP: Parent model 6 (32-16-8-1), 100 epochs, minmax scaler


# To add:
    # Menu?
    # feature importance from function in preprocessing_training_functions.py
    # Graph for a model : epochs vs cost
    # make a custom projection
