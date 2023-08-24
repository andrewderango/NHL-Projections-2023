import os
import pandas as pd
import xgboost as xgb
import preprocessing_training_functions
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error

projection_year = 2024
years = [2012, 2016, 2019, 2021, 2022, 2023]
year_training_dfs = []
model_types = ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF']
proj_stat = 'EV ATOI'
features = ['EV ATOI', 'PP ATOI', 'EV P/60', 'P/GP']

stat_df = preprocessing_training_functions.scrape_player_statistics(True)
if '2021 GP' in stat_df:
    stat_df['2021 GP'] = stat_df['2021 GP'] * 82/56
if '2020 GP' in stat_df:
    stat_df['2020 GP'] = stat_df['2020 GP'] * 82/69.5
if '2013 GP' in stat_df:
    stat_df['2013 GP'] = stat_df['2013 GP'] * 82/48
for year in range(2008, projection_year):
    stat_df[f'{year} ATOI'] = stat_df[f'{year} EV ATOI'] + stat_df[f'{year} PP ATOI'] + stat_df[f'{year} PK ATOI']
    stat_df[f'{year} EV P/60'] = stat_df[f'{year} EV G/60'] + stat_df[f'{year} EV A1/60'] + stat_df[f'{year} EV A1/60']
    stat_df[f'{year} PP P/60'] = stat_df[f'{year} PP G/60'] + stat_df[f'{year} PP A1/60'] + stat_df[f'{year} PP A1/60']
    stat_df[f'{year} P/GP'] = stat_df[f'{year} EV P/60']*stat_df[f'{year} EV ATOI']/60 + stat_df[f'{year} PP P/60']*stat_df[f'{year} PP ATOI']/60 + (stat_df[f'{year} PK G/60'] + stat_df[f'{year} PK A1/60'] + stat_df[f'{year} PK A1/60'])*stat_df[f'{year} PK ATOI']/60

for year in years:
    training_df = preprocessing_training_functions.make_projection_df(stat_df, year)
    training_df.insert(2, 'FwdBool', training_df['Position'] != 'D')
    training_df.insert(3, 'Year', year)
    for model_type in model_types:
        solo_proj_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/{model_type.lower().replace(' ', '_')}_partial_projections_{year}.csv")
        solo_proj_df = solo_proj_df.drop(solo_proj_df.columns[0], axis=1)
        training_df = training_df.merge(solo_proj_df[['Player', proj_stat]].rename(columns={proj_stat: f'{model_type} {proj_stat}'}), on='Player', how='left')

    for feature in features:
        merge_columns = ['Player', f'{year} {feature}', f'{year-1} {feature}', f'{year-2} {feature}', f'{year-3} {feature}']
        rename_columns = {f'{year} {feature}': f'Y-0 {feature}', f'{year-1} {feature}': f'Y-1 {feature}', f'{year-2} {feature}': f'Y-2 {feature}', f'{year-3} {feature}': f'Y-3 {feature}'}
        training_df = training_df.merge(stat_df[merge_columns].rename(columns=rename_columns), on='Player', how='left')

    training_df = training_df.sort_values(f'RR {proj_stat}', ascending=False).rename(columns={f'Bayesian NN {proj_stat}': f'BNN {proj_stat}'})
    training_df = training_df.reset_index(drop=True)
    training_df.index = training_df.index + 1
    year_training_dfs.append(training_df)

training_df = pd.concat(year_training_dfs, ignore_index=True).dropna(subset=[f'Y-0 {proj_stat}'])
print(training_df.head(10))

feature_list = ['Age', 'FwdBool', f'RR {proj_stat}', f'RF {proj_stat}', f'BNN {proj_stat}', f'NN {proj_stat}', f'SVR {proj_stat}']
for feature in features:
    feature_list.append(f'Y-1 {feature}')
    feature_list.append(f'Y-2 {feature}')
    feature_list.append(f'Y-3 {feature}')
X = training_df[feature_list]
y = training_df[f'Y-0 {proj_stat}']
sample_weights = training_df[f'Y-0 {proj_stat}']

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    objective = 'reg:squarederror',
    subsample = 0.8,
    colsample_bytree = 0.6,
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [5, 6, 7, 8, 9]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1
)

grid_search.fit(X_train, y_train, sample_weight=weights_train)
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f'MSE for parameters {params}: {-mean_score:.2f}')
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Best Model Mean Squared Error: {mse:.2f}")
print(f"Best Model Mean Absolute Error: {mae:.2f}")
print(f'Optimal parameters, as determined by grid search: {grid_search.best_params_}')