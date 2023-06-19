import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import time
import statistics
import preprocessing_training_functions

# R² not directly supported by keras
def r2_metric(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    r2 = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    return r2

start = time.time()
proj_stat = 'Gper60'
position = 'forward'
prev_years = 4
situation = 'PP'
instance_df, input_shape = preprocessing_training_functions.create_year_restricted_instance_df(proj_stat, position, prev_years, situation)

if proj_stat == 'Gper60':
    X, y = preprocessing_training_functions.extract_instance_data(instance_df, proj_stat, prev_years, situation, position)
else:
    X, y = preprocessing_training_functions.extract_instance_data(instance_df, proj_stat, prev_years, situation)


kfold = KFold(n_splits=5, shuffle=True)
fold_no = 1
mse_foldhist = []
mae_foldhist = []
r2_foldhist = []
huber_foldhist = []
logcosh_foldhist = []

for train, test in kfold.split(X, y):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(126, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(42, activation='relu'),
        tf.keras.layers.Dense(14, activation='relu'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', r2_metric, tf.keras.losses.Huber(), tf.keras.losses.LogCosh()])

    print(f'\nFold #{fold_no} - Cross Validation')

    # Fit data to model
    history = model.fit(X[train], y[train], epochs=10, verbose=1)

    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f'Fold #{fold_no} Scoring')
    print(f'MSE     : {scores[0]}')
    print(f'MAE     : {scores[1]}')
    print(f'R²      : {scores[2]}')
    print(f'Huber   : {scores[3]}')
    print(f'Log-Cosh: {scores[4]}')

    mse_foldhist.append(scores[0])
    mae_foldhist.append(scores[1])
    r2_foldhist.append(scores[2])
    huber_foldhist.append(scores[3])
    logcosh_foldhist.append(scores[4])

    fold_no += 1

print('\n--TOTAL SCORING--')
print(f'MSE      | AVG: {statistics.mean(mse_foldhist):.3f} | STDEV: {statistics.stdev(mse_foldhist):.3f} | Raw: {mse_foldhist}')
print(f'MAE      | AVG: {statistics.mean(mae_foldhist):.3f} | STDEV: {statistics.stdev(mae_foldhist):.3f} | Raw: {mae_foldhist}')
print(f'R²       | AVG: {statistics.mean(r2_foldhist):.3f} | STDEV: {statistics.stdev(r2_foldhist):.3f} | Raw: {r2_foldhist}')
print(f'Huber    | AVG: {statistics.mean(huber_foldhist):.3f} | STDEV: {statistics.stdev(huber_foldhist):.3f} | Raw: {huber_foldhist}')
print(f'Log-Cosh | AVG: {statistics.mean(logcosh_foldhist):.3f} | STDEV: {statistics.stdev(logcosh_foldhist):.3f} | Raw: {logcosh_foldhist}')

print(f'\nResults generated in {time.time()-start:.3f} seconds')