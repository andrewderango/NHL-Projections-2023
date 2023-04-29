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

input_shape = (15,)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MeanAbsoluteError', 'MeanSquaredLogarithmicError'])

df = pd.read_csv(f'{os.path.dirname(__file__)}/CSV Data/forward_GP_instance_training_data.csv')
# df = df.dropna()
df = df.fillna(0)
df = df.reset_index(drop=True)
print(df)

X = []
y = []

for index, row in df.iterrows():
    X.append([row['Age'], row['Height'], row['Weight'], row['Y1 GP'], row['Y2 GP'], row['Y3 GP'], row['Y4 GP'], row['Y1 ATOI'], row['Y2 ATOI'], row['Y3 ATOI'], row['Y4 ATOI'], row['Y1 G/82'], row['Y2 G/82'], row['Y3 G/82'], row['Y4 G/82']]) # features
    y.append(row['Y5 GP']) # target

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

X_scaler = StandardScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

model.fit(X_train_scaled, y_train, epochs=100)
test_loss, test_acc, *rest = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'\nMean Absolute Error of test: {test_acc:.4f}')

# Make Projection
x_new = X_scaler.transform([[26.75, 73, 193, 76, 82, 80, 82, 21.87, 22.15, 22.07, 22.39, 44, 48, 45, 64]])
y_pred = model.predict(x_new)[0][0]

print(f'Projected games: {y_pred:.2f}')

print(f'{time.time()-start:.3f} seconds')
