import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import preprocessing_training_functions
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error

### bootstrap xgboost model

def train_model(stat_df, projection_df, proj_stat, features, model_types, train_years, model_params, show_feature_importance, download_instance_df, download_partial_projections, projection_year):
    year_training_dfs = []
    for year in train_years:
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

    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train, sample_weight=weights_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse:.2f}")
    print(f"Model Mean Absolute Error: {mae:.2f}")

    player_list = []
    for index, row in stat_df.iterrows():
        if row['Player'] in list(stat_df.loc[(stat_df[f'{projection_year-1} GP'] >= 1)]['Player']):
                player_list.append(row['Player'])

    stat_list = []
    for player in player_list:
        new_list_element = []
        new_list_element.append(preprocessing_training_functions.calc_age(stat_df.loc[stat_df['Player'] == player, 'Date of Birth'].iloc[0], projection_year-1))
        new_list_element.append(stat_df.loc[stat_df['Player'] == player, 'Position'].iloc[0] != 'D'),
        for model_type in model_types:
            solo_proj_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/{model_type.lower().replace(' ', '_')}_partial_projections_{projection_year}.csv")
            solo_proj_df = solo_proj_df.drop(solo_proj_df.columns[0], axis=1)
            try:
                new_list_element.append(solo_proj_df.loc[solo_proj_df['Player'] == player, proj_stat].iloc[0])
            except IndexError:
                new_list_element.append(np.nan)
        for feature in features:
            new_list_element.append(stat_df.loc[stat_df['Player'] == player, f'{projection_year-1} {feature}'].iloc[0])
            new_list_element.append(stat_df.loc[stat_df['Player'] == player, f'{projection_year-2} {feature}'].iloc[0])
            new_list_element.append(stat_df.loc[stat_df['Player'] == player, f'{projection_year-3} {feature}'].iloc[0])
        stat_list.append(new_list_element)

    ### Model is currently trained on X_train and y_train. Fix this before projecting new stats, and replace old stats. Only need to split test/train for hyperparameter tuning.

    print(feature_list)
    stat_prediction_df = pd.DataFrame(stat_list, columns=feature_list)
    stat_prediction_df.insert(0, 'Player', player_list)
    print(stat_prediction_df)

    predictions = np.clip(model.predict(stat_prediction_df[feature_list]), 0, 82)
    stat_prediction_df[f'{proj_stat} Projection'] = predictions

    stat_prediction_df = stat_prediction_df.sort_values(f'{proj_stat} Projection', ascending=False)
    stat_prediction_df = stat_prediction_df.reset_index(drop=True)
    stat_prediction_df.index = stat_prediction_df.index + 1
    print(stat_prediction_df.to_string())

    for index, player in enumerate(player_list):
        projection = predictions[index]

        if player in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player, proj_stat] = projection
        else:
            new_row = pd.DataFrame({'Player': [player], proj_stat: [projection]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    print(projection_df.head(20))

    if show_feature_importance == True:
        importance_scores = model.get_booster().get_score(importance_type='weight')
        sorted_scores = sorted(importance_scores.items(), key=lambda x: x[1], reverse=False)
        feature_names = [score[0] for score in sorted_scores]
        scores = [score[1] for score in sorted_scores]
        norm_scores = np.array(scores) / np.max(scores)
        color_map = plt.get_cmap('viridis').reversed()
        colors = [color_map(score) for score in norm_scores]
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(scores)), scores, color=colors, tick_label=feature_names)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance for XGBoost {proj_stat} Projections')
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Normalized Importance')
        cbar.ax.yaxis.set_ticks_position('left')
        plt.tight_layout()
        plt.show()

    if download_instance_df == True:
        filename = f'xgboost_{proj_stat}_train_{projection_year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        stat_prediction_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    if download_partial_projections == True:
        filename = f'xgboost_partial_projections_{projection_year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df

def make_projections(existing_stat_df, existing_partial_projections, projection_year, download_csv):
    stat_df = preprocessing_training_functions.scrape_player_statistics(existing_stat_df)
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

    if existing_partial_projections == False:
        projection_df = preprocessing_training_functions.make_projection_df(stat_df, projection_year)
    else:
        projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_partial_projections_{projection_year}.csv")
        projection_df = projection_df.drop(projection_df.columns[0], axis=1)

    # projection_df = train_model(stat_df, projection_df, 'GP', ['GP', 'ATOI'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 100, 'colsample_bytree': 0.6, 'random_state': 42}, True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'EV ATOI', ['EV ATOI', 'PP ATOI', 'EV P/60', 'P/GP'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'colsample_bytree': 0.7, 'random_state': 42}, True, True, True, projection_year)

make_projections(True, True, 2024, False)