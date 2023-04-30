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

input_shape = (7,)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(126, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(42, activation='relu'),
    tf.keras.layers.Dense(14, activation='relu'),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MeanAbsoluteError', 'MeanSquaredLogarithmicError'])

df = pd.read_csv(f'{os.path.dirname(__file__)}/CSV Data/forward_GP_instance_training_data.csv')
df = df.fillna(0)
df = df.reset_index(drop=True)
print(df)

X = []
y = []

for index, row in df.iterrows():
    X.append([row['Age'], row['Height'], row['Weight'], row['Y1 GP'], row['Y2 GP'], row['Y3 GP'], row['Y4 GP']]) # features
    y.append(row['Y5 dGP']) # target

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

X_scaler = StandardScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

model.fit(X_train_scaled, y_train, epochs=5)
test_loss, test_acc, *rest = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'\nMean Absolute Error of test: {test_acc:.4f}')

# Make Projection
x_new = X_scaler.transform[[26, 72, 188, 82, 82, 82, 82]]
y_pred = model.predict(x_new)[0][0] + (82+82+82+82)/4

print(f'Projected games: {y_pred:.2f}')

print(f'{time.time()-start:.3f} seconds')
