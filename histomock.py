import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import ast
import pandas as pd
import os

player = 'Andrei Svechnikov'
stat = 'GP'

year = 2024
distribution_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/bayesian_nn_partial_distributions_{year}.csv")
distribution_df = distribution_df.drop(distribution_df.columns[0], axis=1)
distribution = ast.literal_eval(distribution_df.loc[distribution_df['Player'] == player, stat].values[0])
print(distribution)

plt.style.use('ggplot')

# Compute kernel density estimate
kde = gaussian_kde(distribution)

# Generate points on x-axis for plotting
x_vals = np.linspace(min(distribution), max(distribution), 100)

# Compute corresponding y-values for the KDE curve
y_vals = kde(x_vals)

# Plotting
plt.plot(x_vals, y_vals, color='purple', label='KDE')
plt.fill_between(x_vals, y_vals, color='purple', alpha=0.2) # Curve filler
# plt.hist(distribution, bins=10, density=True, alpha=0.5, label='Histogram')
plt.xlabel(stat)
plt.ylabel('Density')
plt.title(f'{player} {stat} Probability Density')
# plt.legend()
plt.show()
