import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import plasma_r

df = pd.read_csv(f'{os.path.dirname(__file__)}/CSV Data/forward_GP_instance_training_data.csv')
X = df[['Age', 'Height', 'Weight', 'Y1 GP', 'Y2 GP', 'Y3 GP', 'Y4 GP']] # features
y = df['Y5 dGP'] # target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a random forest regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances using MDI algorithm
importances = rf.feature_importances_

# Create a pandas Series object with feature importances
feat_importances = pd.Series(importances, index=X.columns)

# Sort the feature importances in descending order
feat_importances = feat_importances.sort_values(ascending=True)

# Create a bar chart of the feature importances
fig, ax = plt.subplots(figsize=(9, 6))
colors = plasma_r(feat_importances.values / max(feat_importances.values))
ax.barh(y=feat_importances.index, width=feat_importances.values, color=colors)
ax.set_title("Random Forest Feature Importances (MDI)", weight='bold', fontsize=15)
ax.set_xlabel("Relative Importance", weight='bold')
ax.set_ylabel("Features", weight='bold')
ax.tick_params(length=0)
plt.box(False)
ax.figure.tight_layout()
plt.show()
