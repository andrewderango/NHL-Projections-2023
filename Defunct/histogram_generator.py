import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import ast
import pandas as pd
import os

player = 'Trevor Zegras'
stat = 'EV G/60'

year = 2024
distribution_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/bayesian_nn_partial_distributions_{year}.csv")
distribution_df = distribution_df.drop(distribution_df.columns[0], axis=1)
distribution = ast.literal_eval(distribution_df.loc[distribution_df['Player'] == player, stat].values[0])
# print(distribution)

x = np.linspace(min(distribution), max(distribution), 1000)
kde = gaussian_kde(distribution)

pdf = kde.pdf(x)
cdf = 1 - (np.cumsum(pdf) * (x[1] - x[0]))

mean = np.mean(distribution)
median = np.median(distribution)
est_mode = x[np.argmax(pdf)]

plt.style.use('ggplot')
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, color='purple', label='PDF')
plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')
plt.axvline(median, color='green', linestyle='--', label=f'Median = {median:.2f}')
plt.axvline(est_mode, color='blue', linestyle='--', label=f'Estimated Mode = {est_mode:.2f}')
plt.fill_between(x, pdf, color='purple', alpha=0.2)
plt.xlabel(stat)
plt.ylabel('Density')
plt.title(f'{player} {stat} PDF')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, cdf, color='purple', label='CDF')
plt.fill_between(x, cdf, color='purple', alpha=0.2)
plt.xlabel(stat)
plt.ylabel('Probability')
plt.title(f'{player} {stat} CDF')
# plt.ylim([0, 1])
# plt.legend()

plt.tight_layout()
plt.show()
