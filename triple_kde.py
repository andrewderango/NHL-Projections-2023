import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import ast
import pandas as pd
import os

player1 = 'Kirill Kaprizov'
player2 = 'Jason Robertson'
player3 = 'Tage Thompson'
stat = 'EV G/60'

year = 2024
distribution_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/bayesian_nn_partial_distributions_{year}.csv")
distribution_df = distribution_df.drop(distribution_df.columns[0], axis=1)
distribution1 = ast.literal_eval(distribution_df.loc[distribution_df['Player'] == player1, stat].values[0])
distribution2 = ast.literal_eval(distribution_df.loc[distribution_df['Player'] == player2, stat].values[0])
distribution3 = ast.literal_eval(distribution_df.loc[distribution_df['Player'] == player3, stat].values[0])
# print(distribution)

x1 = np.linspace(min(distribution1), max(distribution1), 1000)
x2 = np.linspace(min(distribution2), max(distribution2), 1000)
x3 = np.linspace(min(distribution3), max(distribution3), 1000)
kde1 = gaussian_kde(distribution1)
kde2 = gaussian_kde(distribution2)
kde3 = gaussian_kde(distribution3)

pdf1 = kde1.pdf(x1)
pdf2 = kde2.pdf(x2)
pdf3 = kde3.pdf(x3)
cdf1 = 1 - (np.cumsum(pdf1) * (x1[1] - x1[0]))
cdf2 = 1 - (np.cumsum(pdf2) * (x2[1] - x2[0]))
cdf3 = 1 - (np.cumsum(pdf3) * (x3[1] - x3[0]))

mean1 = np.mean(distribution1)
mean2 = np.mean(distribution2)
mean3 = np.mean(distribution3)
median1 = np.median(distribution1)
median2 = np.median(distribution2)
median3 = np.median(distribution3)
est_mode1 = x1[np.argmax(pdf1)]
est_mode2 = x2[np.argmax(pdf2)]
est_mode3 = x3[np.argmax(pdf3)]

plt.style.use('ggplot')
plt.figure(figsize=(10, 8))

plt.subplot(3, 2, 1)
plt.plot(x1, pdf1, color='purple', label='PDF')
plt.axvline(mean1, color='red', linestyle='--', label=f'Mean = {mean1:.2f}')
plt.axvline(median1, color='green', linestyle='--', label=f'Median = {median1:.2f}')
plt.axvline(est_mode1, color='blue', linestyle='--', label=f'Estimated Mode = {est_mode1:.2f}')
plt.fill_between(x1, pdf1, color='purple', alpha=0.2)
plt.xlabel(stat)
plt.ylabel('Density')
plt.title(f'{player1} {stat} PDF')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(x1, cdf1, color='purple', label='CDF')
plt.fill_between(x1, cdf1, color='purple', alpha=0.2)
plt.xlabel(stat)
plt.ylabel('Probability')
plt.title(f'{player1} {stat} CDF')
# plt.ylim([0, 1])
# plt.legend()

plt.subplot(3, 2, 3)
plt.plot(x2, pdf2, color='purple', label='PDF')
plt.axvline(mean2, color='red', linestyle='--', label=f'Mean = {mean2:.2f}')
plt.axvline(median2, color='green', linestyle='--', label=f'Median = {median2:.2f}')
plt.axvline(est_mode2, color='blue', linestyle='--', label=f'Estimated Mode = {est_mode2:.2f}')
plt.fill_between(x2, pdf2, color='purple', alpha=0.2)
plt.xlabel(stat)
plt.ylabel('Density')
plt.title(f'{player2} {stat} PDF')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(x2, cdf2, color='purple', label='CDF')
plt.fill_between(x2, cdf2, color='purple', alpha=0.2)
plt.xlabel(stat)
plt.ylabel('Probability')
plt.title(f'{player2} {stat} CDF')
# plt.ylim([0, 1])
# plt.legend()

plt.subplot(3, 2, 5)
plt.plot(x3, pdf3, color='purple', label='PDF')
plt.axvline(mean3, color='red', linestyle='--', label=f'Mean = {mean3:.2f}')
plt.axvline(median3, color='green', linestyle='--', label=f'Median = {median3:.2f}')
plt.axvline(est_mode3, color='blue', linestyle='--', label=f'Estimated Mode = {est_mode3:.2f}')
plt.fill_between(x3, pdf3, color='purple', alpha=0.2)
plt.xlabel(stat)
plt.ylabel('Density')
plt.title(f'{player3} {stat} PDF')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(x3, cdf3, color='purple', label='CDF')
plt.fill_between(x3, cdf3, color='purple', alpha=0.2)
plt.xlabel(stat)
plt.ylabel('Probability')
plt.title(f'{player3} {stat} CDF')
# plt.ylim([0, 1])
# plt.legend()

plt.tight_layout()
plt.show()