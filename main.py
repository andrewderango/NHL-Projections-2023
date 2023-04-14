import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

start = time.time()

input_shape = (5,)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MeanAbsoluteError', 'MeanSquaredLogarithmicError'])

df = pd.read_csv(f'{os.path.dirname(__file__)}/Output CSV Data/instance_training_data.csv')
df = df.dropna()
df = df.reset_index(drop=True)
print(df.to_string())

X = []
y = []

for index, row in df.iterrows():
    X.append([row['Age'], row['Y1 EV ATOI'], row['Y2 EV ATOI'], row['Y3 EV ATOI'], row['Y4 EV ATOI']])
    y.append(row['Y5 EV ATOI'])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

X_scaler = StandardScaler().fit(X_train)
# print(X_scaler.mean_) # mean of each column
# print(X_scaler.scale_) # variance of each column. maybe stdev?
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

model.fit(X_train_scaled, y_train, epochs=30)
test_loss, test_acc, *rest = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'\nMean Absolute Error of test: {test_acc:.4f}')

# Predicting future data given the player's information
x_new = X_scaler.transform([[39, 21, 21, 20, 20]])
y_pred = model.predict(x_new)[0][0]

print(f'Predicted number of goals: {y_pred:.2f}')

print(f'{time.time()-start:.3f} seconds')
