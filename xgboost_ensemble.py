import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import preprocessing_training_functions

def train_model(stat_df, projection_df, proj_stat, proj_bounds, features, model_types, train_years, model_params, num_bootstraps, samplesize_stat, samplesize_threshold, sample_thresh_columns, show_feature_importance, download_instance_df, download_partial_projections, projection_year):
    print('Generating training dataframe')
    year_training_dfs = []
    for year in train_years:
        training_df = preprocessing_training_functions.make_projection_df(stat_df, year)
        training_df.insert(2, 'FwdBool', training_df['Position'] != 'D')
        training_df.insert(3, 'Year', year)

        if 'GP' not in features:
            training_df = pd.merge(training_df, stat_df[['Player', f'{year} GP']], on='Player', how='left')
            training_df.rename(columns={f'{year} GP': 'Y-0 GP'}, inplace=True)

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
    feature_list = ['Age', 'FwdBool', f'RR {proj_stat}', f'RF {proj_stat}', f'BNN {proj_stat}', f'NN {proj_stat}', f'SVR {proj_stat}']
    for feature in features:
        feature_list.append(f'Y-1 {feature}')
        feature_list.append(f'Y-2 {feature}')
        feature_list.append(f'Y-3 {feature}')
    X = training_df[feature_list]
    y = training_df[f'Y-0 {proj_stat}']
    sample_weights = training_df[f'Y-0 GP']
    predictions = []

    print('Generating projection features')
    player_list = stat_df.loc[stat_df[f'{projection_year-1} GP'] > 0, 'Player'].tolist()
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

    stat_prediction_df = pd.DataFrame(stat_list, columns=feature_list)
    stat_prediction_df.insert(0, 'Player', player_list)

    if samplesize_stat is not None:
        if f'Y-1 {samplesize_stat}' not in stat_prediction_df.columns.to_list():
            stat_prediction_df = stat_prediction_df.merge(stat_df[['Player', f'{projection_year-1} {samplesize_stat}', f'{projection_year-2} {samplesize_stat}', f'{projection_year-3} {samplesize_stat}', ]], on='Player', how='left')
            stat_prediction_df.rename(columns={f'{projection_year-1} {samplesize_stat}': f'Y-1 {samplesize_stat}', f'{projection_year-2} {samplesize_stat}': f'Y-2 {samplesize_stat}', f'{projection_year-3} {samplesize_stat}': f'Y-3 {samplesize_stat}'}, inplace=True)
        for thresh_column in sample_thresh_columns:
            stat_prediction_df.loc[stat_prediction_df[f'Y-1 {samplesize_stat}'] < samplesize_threshold, f'Y-1 {thresh_column}'] = (stat_prediction_df.loc[stat_prediction_df[f'Y-1 {samplesize_stat}'] < samplesize_threshold, f'Y-1 {thresh_column}'] * stat_prediction_df[f'Y-1 {samplesize_stat}'] + stat_prediction_df[stat_prediction_df[f'Y-1 {samplesize_stat}'] >= samplesize_threshold][f'Y-1 {thresh_column}'].quantile(0.30) * (samplesize_threshold - stat_prediction_df[f'Y-1 {samplesize_stat}']))/samplesize_threshold
            stat_prediction_df.loc[stat_prediction_df[f'Y-2 {samplesize_stat}'] < samplesize_threshold, f'Y-2 {thresh_column}'] = (stat_prediction_df.loc[stat_prediction_df[f'Y-2 {samplesize_stat}'] < samplesize_threshold, f'Y-2 {thresh_column}'] * stat_prediction_df[f'Y-2 {samplesize_stat}'] + stat_prediction_df[stat_prediction_df[f'Y-2 {samplesize_stat}'] >= samplesize_threshold][f'Y-2 {thresh_column}'].quantile(0.30) * (samplesize_threshold - stat_prediction_df[f'Y-2 {samplesize_stat}']))/samplesize_threshold
            stat_prediction_df.loc[stat_prediction_df[f'Y-3 {samplesize_stat}'] < samplesize_threshold, f'Y-3 {thresh_column}'] = (stat_prediction_df.loc[stat_prediction_df[f'Y-3 {samplesize_stat}'] < samplesize_threshold, f'Y-3 {thresh_column}'] * stat_prediction_df[f'Y-3 {samplesize_stat}'] + stat_prediction_df[stat_prediction_df[f'Y-3 {samplesize_stat}'] >= samplesize_threshold][f'Y-3 {thresh_column}'].quantile(0.30) * (samplesize_threshold - stat_prediction_df[f'Y-3 {samplesize_stat}']))/samplesize_threshold
        for thresh_column in features:    
            stat_prediction_df.loc[(stat_prediction_df[f'Y-1 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-2 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-3 {samplesize_stat}'].fillna(0)) < samplesize_threshold, f'Y-1 {thresh_column}'] = pd.NA
            stat_prediction_df.loc[(stat_prediction_df[f'Y-1 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-2 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-3 {samplesize_stat}'].fillna(0)) < samplesize_threshold, f'Y-2 {thresh_column}'] = pd.NA
            stat_prediction_df.loc[(stat_prediction_df[f'Y-1 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-2 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-3 {samplesize_stat}'].fillna(0)) < samplesize_threshold, f'Y-3 {thresh_column}'] = pd.NA
        stat_prediction_df.loc[(stat_prediction_df[f'Y-1 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-2 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-3 {samplesize_stat}'].fillna(0)) < samplesize_threshold, f'RF {proj_stat}'] = pd.NA
        stat_prediction_df.loc[(stat_prediction_df[f'Y-1 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-2 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-3 {samplesize_stat}'].fillna(0)) < samplesize_threshold, f'RR {proj_stat}'] = pd.NA
        stat_prediction_df.loc[(stat_prediction_df[f'Y-1 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-2 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-3 {samplesize_stat}'].fillna(0)) < samplesize_threshold, f'NN {proj_stat}'] = pd.NA
        stat_prediction_df.loc[(stat_prediction_df[f'Y-1 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-2 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-3 {samplesize_stat}'].fillna(0)) < samplesize_threshold, f'BNN {proj_stat}'] = pd.NA
        stat_prediction_df.loc[(stat_prediction_df[f'Y-1 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-2 {samplesize_stat}'].fillna(0) + stat_prediction_df[f'Y-3 {samplesize_stat}'].fillna(0)) < samplesize_threshold, f'SVR {proj_stat}'] = pd.NA

    if samplesize_stat not in features and samplesize_stat is not None:
        stat_prediction_df.drop(columns=[f'Y-1 {samplesize_stat}', f'Y-2 {samplesize_stat}', f'Y-3 {samplesize_stat}'], inplace=True)

    average_feature_importance = {}
    print(f'Training XGBoost model using {num_bootstraps} bootstraps')
    for bootstrap in range(num_bootstraps):
        bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
        X_bootstrap = X.iloc[bootstrap_indices]
        y_bootstrap = y.iloc[bootstrap_indices]
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_bootstrap, y_bootstrap, sample_weight=sample_weights)
        predictions.append(np.clip(model.predict(stat_prediction_df[feature_list]), proj_bounds[0], proj_bounds[1]).tolist())
        feature_importance = model.get_booster().get_score(importance_type='weight')
        for feature, importance in feature_importance.items():
            average_feature_importance[feature] = average_feature_importance.get(feature, 0) + importance
        print(str(round((bootstrap + 1)/num_bootstraps*100, 2 ))+ '%')

    average_feature_importance = {feature: importance * (1/num_bootstraps) for feature, importance in average_feature_importance.items()}
    print('Configuring projections')
    predictions = np.transpose(predictions).tolist()
    stat_prediction_df[f'{proj_stat} Projection'] = [np.mean(player_predictions) for player_predictions in predictions]
    stat_prediction_df[f'{proj_stat} Projection Stdev'] = [np.std(player_predictions) for player_predictions in predictions]
    stat_prediction_df[f'{proj_stat} Projection 95% Low CI'] = [np.clip(t.interval(0.95, num_bootstraps-1, loc=avg, scale=stdev)[0], proj_bounds[0], proj_bounds[1]) for avg, stdev in zip([np.mean(player_predictions) for player_predictions in predictions], [np.std(player_predictions) for player_predictions in predictions])]
    stat_prediction_df[f'{proj_stat} Projection 95% High CI'] = [np.clip(t.interval(0.95, num_bootstraps-1, loc=avg, scale=stdev)[1], proj_bounds[0], proj_bounds[1]) for avg, stdev in zip([np.mean(player_predictions) for player_predictions in predictions], [np.std(player_predictions) for player_predictions in predictions])]
    stat_prediction_df[f'{proj_stat} Projection 25th Percentile'] = [np.percentile(player_predictions, 25) for player_predictions in predictions]
    stat_prediction_df[f'{proj_stat} Projection 75th Percentile'] = [np.percentile(player_predictions, 75) for player_predictions in predictions]
    stat_prediction_df[f'{proj_stat} Projection Sample'] = pd.Series(predictions, name=f'{proj_stat} Projection Sample')

    stat_prediction_df = stat_prediction_df.sort_values(f'{proj_stat} Projection', ascending=False)
    stat_prediction_df = stat_prediction_df.reset_index(drop=True)
    stat_prediction_df.index = stat_prediction_df.index + 1
    print(stat_prediction_df[['Player', 'Age', f'Y-3 {proj_stat}', f'Y-2 {proj_stat}', f'Y-1 {proj_stat}', f'{proj_stat} Projection', f'{proj_stat} Projection Stdev', f'{proj_stat} Projection 95% Low CI', f'{proj_stat} Projection 95% High CI']].to_string())

    for index, player in enumerate(player_list):
        projection = predictions[index]

        if player in projection_df['Player'].values:
            projection_df.loc[projection_df['Player'] == player, proj_stat] = np.mean(projection)
        else:
            new_row = pd.DataFrame({'Player': [player], proj_stat: [np.mean(projection)]})
            projection_df = pd.concat([projection_df, new_row], ignore_index=True)
    # print(projection_df.head(20))

    if show_feature_importance == True:
        sorted_scores = sorted(average_feature_importance.items(), key=lambda x: x[1], reverse=False)
        feature_names = [score[0] for score in sorted_scores]
        scores = [score[1] for score in sorted_scores]
        norm_scores = np.array(scores) / np.max(scores)
        color_map = plt.get_cmap('viridis').reversed()
        colors = [color_map(score) for score in norm_scores]
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(scores)), scores, color=colors, tick_label=feature_names)
        plt.xlabel('Average Feature Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Average Feature Importance for Bootstrapped XGBoost {proj_stat} Projections')
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Normalized Importance')
        cbar.ax.yaxis.set_ticks_position('left')
        plt.tight_layout()
        plt.show()

    if download_instance_df == True:
        filename = f'xgboost_{proj_stat.replace("/", "per")}_train_{projection_year}'
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

def goal_era_adjustment(stat_df, projection_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_goal_df = pd.DataFrame()

    for season in range(2007, year-1):
        col = round(((stat_df[f'{season+1} EV G/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP G/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK G/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])) 
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_goal_df = hist_goal_df.reset_index(drop=True)
        hist_goal_df[season+1] = col
    hist_goal_df.index = hist_goal_df.index + 1

    try:
        hist_goal_df[2021] = round(82/56*hist_goal_df[2021]) 
    except KeyError:
        pass
    try:
        hist_goal_df[2020] = round(82/70*hist_goal_df[2020]) 
    except KeyError:
        pass
    try:
        hist_goal_df[2013] = round(82/48*hist_goal_df[2013]) 
    except KeyError:
        pass

    hist_goal_df['Historical Average'] = hist_goal_df.mean(axis=1)
    hist_goal_df['Projected Average'] = hist_goal_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_goal_df['Adjustment'] = hist_goal_df['Projected Average'] - hist_goal_df['Historical Average']
    hist_goal_df['Smoothed Adjustment'] = savgol_filter(hist_goal_df['Adjustment'], 25, 2)
    # print(hist_goal_df.head(750).to_string())

    projection_df['GOALS'] = (projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']
    projection_df = projection_df.sort_values('GOALS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_goal_df['Smoothed Adjustment']/((projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        projection_df['EV G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK G/60'] *= projection_df['Era Adjustment Factor']
        projection_df['GOALS'] = (projection_df['EV G/60']/60*projection_df['EV ATOI'] + projection_df['PP G/60']/60*projection_df['PP ATOI'] + projection_df['PK G/60']/60*projection_df['PK ATOI']) * projection_df['GP']
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

    # Download file
    if download_file == True:
        filename = f'xgboost_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df.fillna(0)

def a1_era_adjustment(stat_df, projection_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_a1_df = pd.DataFrame()

    for season in range(2007, year-1):
        col = round(((stat_df[f'{season+1} EV A1/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP A1/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK A1/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])) 
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_a1_df = hist_a1_df.reset_index(drop=True)
        hist_a1_df[season+1] = col
    hist_a1_df.index = hist_a1_df.index + 1

    try:
        hist_a1_df[2021] = round(82/56*hist_a1_df[2021]) 
    except KeyError:
        pass
    try:
        hist_a1_df[2020] = round(82/70*hist_a1_df[2020]) 
    except KeyError:
        pass
    try:
        hist_a1_df[2013] = round(82/48*hist_a1_df[2013]) 
    except KeyError:
        pass

    hist_a1_df['Historical Average'] = hist_a1_df.mean(axis=1)
    hist_a1_df['Projected Average'] = hist_a1_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_a1_df['Adjustment'] = hist_a1_df['Projected Average'] - hist_a1_df['Historical Average']
    hist_a1_df['Smoothed Adjustment'] = savgol_filter(hist_a1_df['Adjustment'], 25, 2)
    # print(hist_a1_df.head(750).to_string())

    projection_df['PRIMARY ASSISTS'] = (projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP']
    projection_df = projection_df.sort_values('PRIMARY ASSISTS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_a1_df['Smoothed Adjustment']/((projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        projection_df['EV A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK A1/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PRIMARY ASSISTS'] = (projection_df['EV A1/60']/60*projection_df['EV ATOI'] + projection_df['PP A1/60']/60*projection_df['PP ATOI'] + projection_df['PK A1/60']/60*projection_df['PK ATOI']) * projection_df['GP']
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

    # Download file
    if download_file == True:
        filename = f'xgboost_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df.fillna(0)

def a2_era_adjustment(stat_df, projection_df, year=2024, apply_adjustment=True, download_file=False):
    stat_df = stat_df.fillna(0)
    projection_df = projection_df.fillna(0)
    hist_a2_df = pd.DataFrame()

    for season in range(2007, year-1):
        col = round(((stat_df[f'{season+1} EV A2/60']/60*stat_df[f'{season+1} EV ATOI'] + stat_df[f'{season+1} PP A2/60']/60*stat_df[f'{season+1} PP ATOI'] + stat_df[f'{season+1} PK A2/60']/60*stat_df[f'{season+1} PK ATOI']) * stat_df[f'{season+1} GP'])) 
        col = col.sort_values(ascending=False)
        col = col.reset_index(drop=True)
        hist_a2_df = hist_a2_df.reset_index(drop=True)
        hist_a2_df[season+1] = col
    hist_a2_df.index = hist_a2_df.index + 1

    try:
        hist_a2_df[2021] = round(82/56*hist_a2_df[2021]) 
    except KeyError:
        pass
    try:
        hist_a2_df[2020] = round(82/70*hist_a2_df[2020]) 
    except KeyError:
        pass
    try:
        hist_a2_df[2013] = round(82/48*hist_a2_df[2013]) 
    except KeyError:
        pass

    hist_a2_df['Historical Average'] = hist_a2_df.mean(axis=1)
    hist_a2_df['Projected Average'] = hist_a2_df.loc[:, year-4:year-1].mul([0.1, 0.2, 0.3, 0.4]).sum(axis=1)
    hist_a2_df['Adjustment'] = hist_a2_df['Projected Average'] - hist_a2_df['Historical Average']
    hist_a2_df['Smoothed Adjustment'] = savgol_filter(hist_a2_df['Adjustment'], 25, 2)
    # print(hist_a2_df.head(750).to_string())

    projection_df['SECONDARY ASSISTS'] = (projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP']
    projection_df = projection_df.sort_values('SECONDARY ASSISTS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1

    if apply_adjustment == True:
        projection_df['Era Adjustment Factor'] = hist_a2_df['Smoothed Adjustment']/((projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP']) + 1
        projection_df['EV A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PP A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['PK A2/60'] *= projection_df['Era Adjustment Factor']
        projection_df['SECONDARY ASSISTS'] = (projection_df['EV A2/60']/60*projection_df['EV ATOI'] + projection_df['PP A2/60']/60*projection_df['PP ATOI'] + projection_df['PK A2/60']/60*projection_df['PK ATOI']) * projection_df['GP']
        projection_df = projection_df.drop(columns=['Era Adjustment Factor'])

    # Download file
    if download_file == True:
        filename = f'xgboost_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return projection_df.fillna(0)

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
        stat_df[f'{year} EV TOI'] = stat_df[f'{year} EV ATOI'] * stat_df[f'{year} GP']
        stat_df[f'{year} PP TOI'] = stat_df[f'{year} PP ATOI'] * stat_df[f'{year} GP']
        stat_df[f'{year} PK TOI'] = stat_df[f'{year} PK ATOI'] * stat_df[f'{year} GP']
        stat_df[f'{year} EV P/60'] = stat_df[f'{year} EV G/60'] + stat_df[f'{year} EV A1/60'] + stat_df[f'{year} EV A1/60']
        stat_df[f'{year} PP P/60'] = stat_df[f'{year} PP G/60'] + stat_df[f'{year} PP A1/60'] + stat_df[f'{year} PP A1/60']
        stat_df[f'{year} P/GP'] = stat_df[f'{year} EV P/60']*stat_df[f'{year} EV ATOI']/60 + stat_df[f'{year} PP P/60']*stat_df[f'{year} PP ATOI']/60 + (stat_df[f'{year} PK G/60'] + stat_df[f'{year} PK A1/60'] + stat_df[f'{year} PK A1/60'])*stat_df[f'{year} PK ATOI']/60
        stat_df = stat_df.copy()

    if existing_partial_projections == False:
        projection_df = preprocessing_training_functions.make_projection_df(stat_df, projection_year)
    else:
        projection_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_partial_projections_{projection_year}.csv")
        projection_df = projection_df.drop(projection_df.columns[0], axis=1)

    # projection_df = train_model(stat_df, projection_df, 'GP', [0, 82], ['GP', 'ATOI'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 100, 'colsample_bytree': 0.6, 'random_state': 42}, 1000, None, None, None, True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'EV ATOI', [0, 60], ['EV ATOI', 'PP ATOI', 'EV P/60', 'P/GP', 'GP'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 100, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, None, None, None, True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'PP ATOI', [0, 60], ['GP', 'ATOI', 'PP ATOI', 'PP P/60', 'P/GP'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, None, None, None, True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'PK ATOI', [0, 60], ['GP', 'PK ATOI', 'ATOI'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, None, None, None, True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'EV G/60', [0, 60], ['EV G/60', 'EV ixG/60', 'EV Shots/60', 'GP'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 75, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, None, None, None, True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'PP G/60', [0, 60], ['PP G/60', 'PP ixG/60', 'EV ixG/60', 'PP TOI'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 75, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, 'PP TOI', 100, ['PP G/60', 'PP ixG/60'], True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'PK G/60', [0, 60], ['PK G/60', 'PK ixG/60', 'PK TOI'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 75, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, 'PK TOI', 75, ['PK G/60', 'PK ixG/60'], True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'EV A1/60', [0, 60], ['EV A1/60', 'EV A2/60', 'EV TOI'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 100, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, None, None, None, True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'EV A2/60', [0, 60], ['EV A1/60', 'EV A2/60', 'EV TOI'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 150, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, None, None, None, True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'PP A1/60', [0, 60], ['PP TOI', 'PP A1/60', 'PP A2/60', 'PP Rebounds Created/60'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 75, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, 'PP TOI', 100, ['PP A1/60', 'PP A2/60', 'PP Rebounds Created/60'], True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'PP A2/60', [0, 60], ['PP TOI', 'PP A1/60', 'PP A2/60', 'PP Rebounds Created/60'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 75, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, 'PP TOI', 250, ['PP A1/60', 'PP A2/60', 'PP Rebounds Created/60'], True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'PK A1/60', [0, 60], ['PK TOI', 'PK A1/60', 'PK A2/60'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 75, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, 'PK TOI', 100, ['PK A1/60', 'PK A2/60'], True, True, True, projection_year)
    # projection_df = train_model(stat_df, projection_df, 'PK A2/60', [0, 60], ['PK TOI', 'PK A1/60', 'PK A2/60'], ['RR', 'SVR', 'NN', 'Bayesian NN', 'RF'], [2012, 2016, 2019, 2021, 2022, 2023], {'objective': 'reg:squarederror', 'subsample': 0.8, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 75, 'colsample_bytree': 0.7, 'random_state': 42}, 1000, 'PK TOI', 100, ['PK A1/60', 'PK A2/60'], True, True, True, projection_year)

    projection_df = goal_era_adjustment(stat_df, projection_df, year, True, False)
    projection_df = a1_era_adjustment(stat_df, projection_df, year, True, False)
    projection_df = a2_era_adjustment(stat_df, projection_df, year, True, False)
    projection_df['POINTS'] = projection_df['GOALS'] + projection_df['PRIMARY ASSISTS'] + projection_df['SECONDARY ASSISTS']

    projection_df = projection_df.sort_values('POINTS', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    print(projection_df.head(20))
    # print(projection_df.to_string())

    if download_csv == True:
        filename = f'xgboost_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

make_projections(True, True, 2024, True)