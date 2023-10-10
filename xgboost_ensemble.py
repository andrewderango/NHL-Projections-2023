import os
import ast
import scipy
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

def goal_projection_standardizer(stat_df, projection_df, projection_year):
    evg_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_EV Gper60_train_{projection_year+1}.csv")
    evg_per60 = evg_per60.drop(evg_per60.columns[0], axis=1).fillna(0)
    ppg_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PP Gper60_train_{projection_year+1}.csv")
    ppg_per60 = ppg_per60.drop(ppg_per60.columns[0], axis=1).fillna(0)
    pkg_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PK Gper60_train_{projection_year+1}.csv")
    pkg_per60 = pkg_per60.drop(pkg_per60.columns[0], axis=1).fillna(0)

    # model accuracies file
    evg_mse = {
        'RR EV G/60': 0.083,
        'RF EV G/60': 0.056,
        'BNN EV G/60': 0.097,
        'NN EV G/60': 0.082,
        'SVR EV G/60': 0.071,
        'Y-1 EV G/60': 0.131,
    }
    ppg_mse = {
        'RR PP G/60': 2.165,
        'RF PP G/60': 1.893,
        'BNN PP G/60': 2.247,
        'NN PP G/60': 2.074,
        'SVR PP G/60': 2.003,
    }
    pkg_mse = {
        'RR PK G/60': 2.054,
        'RF PK G/60': 1.992,
        'BNN PK G/60': 2.529,
        'NN PK G/60': 2.064,
        'SVR PK G/60': 2.031,
    }

    evg_per60 = evg_per60[['Player', 'RR EV G/60', 'RF EV G/60', 'BNN EV G/60', 'NN EV G/60', 'SVR EV G/60', 'Y-1 EV G/60', 'Y-2 EV G/60', 'Y-3 EV G/60']]
    evg_per60['EV G/60 Stdz. Projection'] = (evg_per60['RR EV G/60']*evg_mse['RR EV G/60'] + evg_per60['RF EV G/60']*evg_mse['RF EV G/60'] + evg_per60['BNN EV G/60']*evg_mse['BNN EV G/60'] + evg_per60['NN EV G/60']*evg_mse['NN EV G/60'] + evg_per60['SVR EV G/60']*evg_mse['SVR EV G/60'] + evg_per60['Y-1 EV G/60']*evg_mse['Y-1 EV G/60']) / (sum(evg_mse.values()))
    ppg_per60 = ppg_per60[['Player', 'RR PP G/60', 'RF PP G/60', 'BNN PP G/60', 'NN PP G/60', 'SVR PP G/60', 'Y-1 PP G/60', 'Y-2 PP G/60', 'Y-3 PP G/60']]
    ppg_per60['PP G/60 Stdz. Projection'] = (ppg_per60['RR PP G/60']*ppg_mse['RR PP G/60'] + ppg_per60['RF PP G/60']*ppg_mse['RF PP G/60'] + ppg_per60['BNN PP G/60']*ppg_mse['BNN PP G/60'] + ppg_per60['NN PP G/60']*ppg_mse['NN PP G/60'] + ppg_per60['SVR PP G/60']*ppg_mse['SVR PP G/60']) / (sum(ppg_mse.values()))
    pkg_per60 = pkg_per60[['Player', 'RR PK G/60', 'RF PK G/60', 'BNN PK G/60', 'NN PK G/60', 'SVR PK G/60', 'Y-1 PK G/60', 'Y-2 PK G/60', 'Y-3 PK G/60']]
    pkg_per60['PK G/60 Stdz. Projection'] = (pkg_per60['RR PK G/60']*pkg_mse['RR PK G/60'] + pkg_per60['RF PK G/60']*pkg_mse['RF PK G/60'] + pkg_per60['BNN PK G/60']*pkg_mse['BNN PK G/60'] + pkg_per60['NN PK G/60']*pkg_mse['NN PK G/60'] + pkg_per60['SVR PK G/60']*pkg_mse['SVR PK G/60']) / (sum(pkg_mse.values()))

    # print(evg_per60)
    # print(ppg_per60)
    # print(pkg_per60)

    standardized_projections = evg_per60[['Player', 'EV G/60 Stdz. Projection']].merge(ppg_per60[['Player', 'PP G/60 Stdz. Projection']], on='Player', how='left').merge(pkg_per60[['Player', 'PK G/60 Stdz. Projection']], on='Player', how='left')
    # print(standardized_projections)

    return standardized_projections

def a1_projection_standardizer(stat_df, projection_df, projection_year):
    eva1_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_EV A1per60_train_{projection_year+1}.csv")
    eva1_per60 = eva1_per60.drop(eva1_per60.columns[0], axis=1).fillna(0)
    ppa1_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PP A1per60_train_{projection_year+1}.csv")
    ppa1_per60 = ppa1_per60.drop(ppa1_per60.columns[0], axis=1).fillna(0)
    pka1_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PK A1per60_train_{projection_year+1}.csv")
    pka1_per60 = pka1_per60.drop(pka1_per60.columns[0], axis=1).fillna(0)

    # model accuracies file
    eva1_mse = {
        'RR EV A1/60': 0.091,
        'RF EV A1/60': 0.066,
        'BNN EV A1/60': 0.104,
        'NN EV A1/60': 0.087,
        'SVR EV A1/60': 0.076,
        'Y-1 EV A1/60': 0.123,
    }
    ppa1_mse = {
        'RR PP A1/60': 3.147,
        'RF PP A1/60': 2.849,
        'BNN PP A1/60': 3.386,
        'NN PP A1/60': 3.118,
        'SVR PP A1/60': 2.858,
    }
    pka1_mse = {
        'RR PK A1/60': 0.253,
        'RF PK A1/60': 0.222,
        'BNN PK A1/60': 0.922,
        'NN PK A1/60': 0.276,
        'SVR PK A1/60': 0.243,
    }

    eva1_per60 = eva1_per60[['Player', 'RR EV A1/60', 'RF EV A1/60', 'BNN EV A1/60', 'NN EV A1/60', 'SVR EV A1/60', 'Y-1 EV A1/60', 'Y-2 EV A1/60', 'Y-3 EV A1/60']]
    eva1_per60['EV A1/60 Stdz. Projection'] = (eva1_per60['RR EV A1/60']*eva1_mse['RR EV A1/60'] + eva1_per60['RF EV A1/60']*eva1_mse['RF EV A1/60'] + eva1_per60['BNN EV A1/60']*eva1_mse['BNN EV A1/60'] + eva1_per60['NN EV A1/60']*eva1_mse['NN EV A1/60'] + eva1_per60['SVR EV A1/60']*eva1_mse['SVR EV A1/60'] + eva1_per60['Y-1 EV A1/60']*eva1_mse['Y-1 EV A1/60']) / (sum(eva1_mse.values()))
    ppa1_per60 = ppa1_per60[['Player', 'RR PP A1/60', 'RF PP A1/60', 'BNN PP A1/60', 'NN PP A1/60', 'SVR PP A1/60', 'Y-1 PP A1/60', 'Y-2 PP A1/60', 'Y-3 PP A1/60']]
    ppa1_per60['PP A1/60 Stdz. Projection'] = (ppa1_per60['RR PP A1/60']*ppa1_mse['RR PP A1/60'] + ppa1_per60['RF PP A1/60']*ppa1_mse['RF PP A1/60'] + ppa1_per60['BNN PP A1/60']*ppa1_mse['BNN PP A1/60'] + ppa1_per60['NN PP A1/60']*ppa1_mse['NN PP A1/60'] + ppa1_per60['SVR PP A1/60']*ppa1_mse['SVR PP A1/60']) / (sum(ppa1_mse.values()))
    pka1_per60 = pka1_per60[['Player', 'RR PK A1/60', 'RF PK A1/60', 'BNN PK A1/60', 'NN PK A1/60', 'SVR PK A1/60', 'Y-1 PK A1/60', 'Y-2 PK A1/60', 'Y-3 PK A1/60']]
    pka1_per60['PK A1/60 Stdz. Projection'] = (pka1_per60['RR PK A1/60']*pka1_mse['RR PK A1/60'] + pka1_per60['RF PK A1/60']*pka1_mse['RF PK A1/60'] + pka1_per60['BNN PK A1/60']*pka1_mse['BNN PK A1/60'] + pka1_per60['NN PK A1/60']*pka1_mse['NN PK A1/60'] + pka1_per60['SVR PK A1/60']*pka1_mse['SVR PK A1/60']) / (sum(pka1_mse.values()))

    # print(eva1_per60)
    # print(ppa1_per60)
    # print(pka1_per60)

    standardized_projections = eva1_per60[['Player', 'EV A1/60 Stdz. Projection']].merge(ppa1_per60[['Player', 'PP A1/60 Stdz. Projection']], on='Player', how='left').merge(pka1_per60[['Player', 'PK A1/60 Stdz. Projection']], on='Player', how='left')
    # print(standardized_projections)

    return standardized_projections

def a2_projection_standardizer(stat_df, projection_df, projection_year):
    eva2_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_EV A2per60_train_{projection_year+1}.csv")
    eva2_per60 = eva2_per60.drop(eva2_per60.columns[0], axis=1).fillna(0)
    ppa2_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PP A2per60_train_{projection_year+1}.csv")
    ppa2_per60 = ppa2_per60.drop(ppa2_per60.columns[0], axis=1).fillna(0)
    pka2_per60 = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PK A2per60_train_{projection_year+1}.csv")
    pka2_per60 = pka2_per60.drop(pka2_per60.columns[0], axis=1).fillna(0)

    # model accuracies file
    eva2_mse = {
        'RR EV A2/60': 0.091,
        'RF EV A2/60': 0.066,
        'BNN EV A2/60': 0.104,
        'NN EV A2/60': 0.087,
        'SVR EV A2/60': 0.076,
        'Y-1 EV A2/60': 0.123,
    }
    ppa2_mse = {
        'RR PP A2/60': 3.147,
        'RF PP A2/60': 2.849,
        'BNN PP A2/60': 3.386,
        'NN PP A2/60': 3.118,
        'SVR PP A2/60': 2.858,
    }
    pka2_mse = {
        'RR PK A2/60': 0.253,
        'RF PK A2/60': 0.222,
        'BNN PK A2/60': 0.922,
        'NN PK A2/60': 0.276,
        'SVR PK A2/60': 0.243,
    }

    eva2_per60 = eva2_per60[['Player', 'RR EV A2/60', 'RF EV A2/60', 'BNN EV A2/60', 'NN EV A2/60', 'SVR EV A2/60', 'Y-1 EV A2/60', 'Y-2 EV A2/60', 'Y-3 EV A2/60']]
    eva2_per60['EV A2/60 Stdz. Projection'] = (eva2_per60['RR EV A2/60']*eva2_mse['RR EV A2/60'] + eva2_per60['RF EV A2/60']*eva2_mse['RF EV A2/60'] + eva2_per60['BNN EV A2/60']*eva2_mse['BNN EV A2/60'] + eva2_per60['NN EV A2/60']*eva2_mse['NN EV A2/60'] + eva2_per60['SVR EV A2/60']*eva2_mse['SVR EV A2/60'] + eva2_per60['Y-1 EV A2/60']*eva2_mse['Y-1 EV A2/60']) / (sum(eva2_mse.values()))
    ppa2_per60 = ppa2_per60[['Player', 'RR PP A2/60', 'RF PP A2/60', 'BNN PP A2/60', 'NN PP A2/60', 'SVR PP A2/60', 'Y-1 PP A2/60', 'Y-2 PP A2/60', 'Y-3 PP A2/60']]
    ppa2_per60['PP A2/60 Stdz. Projection'] = (ppa2_per60['RR PP A2/60']*ppa2_mse['RR PP A2/60'] + ppa2_per60['RF PP A2/60']*ppa2_mse['RF PP A2/60'] + ppa2_per60['BNN PP A2/60']*ppa2_mse['BNN PP A2/60'] + ppa2_per60['NN PP A2/60']*ppa2_mse['NN PP A2/60'] + ppa2_per60['SVR PP A2/60']*ppa2_mse['SVR PP A2/60']) / (sum(ppa2_mse.values()))
    pka2_per60 = pka2_per60[['Player', 'RR PK A2/60', 'RF PK A2/60', 'BNN PK A2/60', 'NN PK A2/60', 'SVR PK A2/60', 'Y-1 PK A2/60', 'Y-2 PK A2/60', 'Y-3 PK A2/60']]
    pka2_per60['PK A2/60 Stdz. Projection'] = (pka2_per60['RR PK A2/60']*pka2_mse['RR PK A2/60'] + pka2_per60['RF PK A2/60']*pka2_mse['RF PK A2/60'] + pka2_per60['BNN PK A2/60']*pka2_mse['BNN PK A2/60'] + pka2_per60['NN PK A2/60']*pka2_mse['NN PK A2/60'] + pka2_per60['SVR PK A2/60']*pka2_mse['SVR PK A2/60']) / (sum(pka2_mse.values()))

    # print(eva2_per60)
    # print(ppa2_per60)
    # print(pka2_per60)

    standardized_projections = eva2_per60[['Player', 'EV A2/60 Stdz. Projection']].merge(ppa2_per60[['Player', 'PP A2/60 Stdz. Projection']], on='Player', how='left').merge(pka2_per60[['Player', 'PK A2/60 Stdz. Projection']], on='Player', how='left')
    # print(standardized_projections)

    return standardized_projections

def goal_distribution_aggregator(stat_df, projection_df, standardized_g_projections, temperature, era_temp, year, download_csv):
    gp_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_GP_train_{year}.csv")
    gp_df = gp_df.drop(gp_df.columns[0], axis=1)
    atoi_evper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_EV ATOI_train_{year}.csv")
    atoi_evper60_df = atoi_evper60_df.drop(atoi_evper60_df.columns[0], axis=1)
    atoi_ppper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PP ATOI_train_{year}.csv")
    atoi_ppper60_df = atoi_ppper60_df.drop(atoi_ppper60_df.columns[0], axis=1)
    atoi_pkper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PK ATOI_train_{year}.csv")
    atoi_pkper60_df = atoi_pkper60_df.drop(atoi_pkper60_df.columns[0], axis=1)

    evper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_EV Gper60_train_{year}.csv")
    evper60_df = evper60_df.drop(evper60_df.columns[0], axis=1)
    ppper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PP Gper60_train_{year}.csv")
    ppper60_df = ppper60_df.drop(ppper60_df.columns[0], axis=1)
    pkper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PK Gper60_train_{year}.csv")
    pkper60_df = pkper60_df.drop(pkper60_df.columns[0], axis=1)

    distribution_df = projection_df[['Player', 'Position', 'Age', 'Height', 'Weight']]
    distribution_df = distribution_df.merge(gp_df[['Player', 'GP Projection', f'GP Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(atoi_evper60_df[['Player', 'EV ATOI Projection', f'EV ATOI Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(atoi_ppper60_df[['Player', 'PP ATOI Projection', f'PP ATOI Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(atoi_pkper60_df[['Player','PK ATOI Projection', f'PK ATOI Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(evper60_df[['Player', 'EV G/60 Projection', f'EV G/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(ppper60_df[['Player', 'PP G/60 Projection', f'PP G/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(pkper60_df[['Player', 'PK G/60 Projection', f'PK G/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(standardized_g_projections, on='Player', how='left')

    distribution_df['GP Projection Sample'] = distribution_df['GP Projection Sample'].apply(ast.literal_eval)
    distribution_df['EV G/60 Projection Sample'] = distribution_df['EV G/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['EV ATOI Projection Sample'] = distribution_df['EV ATOI Projection Sample'].apply(ast.literal_eval)
    distribution_df['PP G/60 Projection Sample'] = distribution_df['PP G/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['PP ATOI Projection Sample'] = distribution_df['PP ATOI Projection Sample'].apply(ast.literal_eval)
    distribution_df['PK G/60 Projection Sample'] = distribution_df['PK G/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['PK ATOI Projection Sample'] = distribution_df['PK ATOI Projection Sample'].apply(ast.literal_eval)

    distribution_df['Goal Projection Sample'] = distribution_df.apply(lambda row: [(a*b/60 + c*d/60 + e*f/60) * g for a, b, c, d, e, f, g in zip(row['EV G/60 Projection Sample'], row['EV ATOI Projection Sample'], row['PP G/60 Projection Sample'], row['PP G/60 Projection Sample'], row['PK G/60 Projection Sample'], row['PK G/60 Projection Sample'], row['GP Projection Sample'])], axis=1)
    distribution_df['Projected Goals I'] = (distribution_df['EV ATOI Projection']*distribution_df['EV G/60 Projection']/60 + distribution_df['PP ATOI Projection']*distribution_df['PP G/60 Projection']/60 + distribution_df['PK ATOI Projection']*distribution_df['PK G/60 Projection']/60) * distribution_df['GP Projection']
    distribution_df['Stdz. Proj. Goals'] = (distribution_df['EV ATOI Projection']*distribution_df['EV G/60 Stdz. Projection']/60 + distribution_df['PP ATOI Projection']*distribution_df['PP G/60 Stdz. Projection']/60 + distribution_df['PK ATOI Projection']*distribution_df['PK G/60 Stdz. Projection']/60) * distribution_df['GP Projection']
    distribution_df = distribution_df.sort_values('Projected Goals I', ascending=False)
    distribution_df = distribution_df.reset_index(drop=True)
    distribution_df.index = distribution_df.index + 1

    distribution_df = distribution_df.drop(columns=[
        'GP Projection', 'GP Projection Sample', 
        'EV ATOI Projection', 'EV ATOI Projection Sample', 
        'PP ATOI Projection', 'PP ATOI Projection Sample', 
        'PK ATOI Projection', 'PK ATOI Projection Sample', 
        'EV G/60 Projection', 'EV G/60 Projection Sample', 
        'PP G/60 Projection', 'PP G/60 Projection Sample', 
        'PK G/60 Projection', 'PK G/60 Projection Sample',
        'EV G/60 Stdz. Projection', 'PP G/60 Stdz. Projection', 'PK G/60 Stdz. Projection'])
    
    distribution_df['Projected Goals II'] = distribution_df['Projected Goals I']*temperature + distribution_df['Stdz. Proj. Goals']*(1-temperature)

    # Era Adjustment
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

    distribution_df = distribution_df.sort_values('Projected Goals II', ascending=False)
    distribution_df = distribution_df.reset_index(drop=True)
    distribution_df.index = distribution_df.index + 1
    distribution_df['Era Adjustment Factor'] = (hist_goal_df['Smoothed Adjustment']/distribution_df['Projected Goals II'])*era_temp + 1
    distribution_df['Projected Goals'] = distribution_df['Projected Goals II']*distribution_df['Era Adjustment Factor']
    distribution_df['Goal Projection Sample'] = distribution_df.apply(lambda row: [x * (row['Projected Goals']/row['Projected Goals I']) for x in row['Goal Projection Sample']], axis=1)
    distribution_df = distribution_df.drop(columns=['Projected Goals I', 'Stdz. Proj. Goals', 'Projected Goals II', 'Era Adjustment Factor'])

    distribution_df['StDev'] = distribution_df['Goal Projection Sample'].apply(lambda x: np.std(x))
    distribution_df['5th Percentile'] = distribution_df['Goal Projection Sample'].apply(lambda x: np.percentile(x, 5))
    distribution_df['95th Percentile'] = distribution_df['Goal Projection Sample'].apply(lambda x: np.percentile(x, 95))
    distribution_df['60G Probability'] = distribution_df['Goal Projection Sample'].apply(lambda x: (sum(value >= 60 for value in x) / len(x)))
    distribution_df['50G Probability'] = distribution_df['Goal Projection Sample'].apply(lambda x: (sum(value >= 50 for value in x) / len(x)))
    distribution_df['40G Probability'] = distribution_df['Goal Projection Sample'].apply(lambda x: (sum(value >= 40 for value in x) / len(x)))
    distribution_df['30G Probability'] = distribution_df['Goal Projection Sample'].apply(lambda x: (sum(value >= 30 for value in x) / len(x)))
    distribution_df['20G Probability'] = distribution_df['Goal Projection Sample'].apply(lambda x: (sum(value >= 20 for value in x) / len(x)))

    if download_csv == True:
        filename = f'xgboost_goal_distributions_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        distribution_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return distribution_df

def assist_distribution_aggregator(stat_df, projection_df, standardized_a1_projections, standardized_a2_projections, temperature, era_temp, year, download_csv):
    gp_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_GP_train_{year}.csv")
    gp_df = gp_df.drop(gp_df.columns[0], axis=1)
    atoi_evper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_EV ATOI_train_{year}.csv")
    atoi_evper60_df = atoi_evper60_df.drop(atoi_evper60_df.columns[0], axis=1)
    atoi_ppper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PP ATOI_train_{year}.csv")
    atoi_ppper60_df = atoi_ppper60_df.drop(atoi_ppper60_df.columns[0], axis=1)
    atoi_pkper60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PK ATOI_train_{year}.csv")
    atoi_pkper60_df = atoi_pkper60_df.drop(atoi_pkper60_df.columns[0], axis=1)

    eva1per60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_EV A1per60_train_{year}.csv")
    eva1per60_df = eva1per60_df.drop(eva1per60_df.columns[0], axis=1)
    ppa1per60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PP A1per60_train_{year}.csv")
    ppa1per60_df = ppa1per60_df.drop(ppa1per60_df.columns[0], axis=1)
    pka1per60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PK A1per60_train_{year}.csv")
    pka1per60_df = pka1per60_df.drop(pka1per60_df.columns[0], axis=1)
    eva2per60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_EV A2per60_train_{year}.csv")
    eva2per60_df = eva2per60_df.drop(eva2per60_df.columns[0], axis=1)
    ppa2per60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PP A2per60_train_{year}.csv")
    ppa2per60_df = ppa2per60_df.drop(ppa2per60_df.columns[0], axis=1)
    pka2per60_df = pd.read_csv(f"{os.path.dirname(__file__)}/CSV Data/xgboost_PK A2per60_train_{year}.csv")
    pka2per60_df = pka2per60_df.drop(pka2per60_df.columns[0], axis=1)

    distribution_df = projection_df[['Player', 'Position', 'Age', 'Height', 'Weight']]
    distribution_df = distribution_df.merge(gp_df[['Player', 'GP Projection', f'GP Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(atoi_evper60_df[['Player', 'EV ATOI Projection', f'EV ATOI Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(atoi_ppper60_df[['Player', 'PP ATOI Projection', f'PP ATOI Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(atoi_pkper60_df[['Player','PK ATOI Projection', f'PK ATOI Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(eva1per60_df[['Player', 'EV A1/60 Projection', f'EV A1/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(ppa1per60_df[['Player', 'PP A1/60 Projection', f'PP A1/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(pka1per60_df[['Player', 'PK A1/60 Projection', f'PK A1/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(eva2per60_df[['Player', 'EV A2/60 Projection', f'EV A2/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(ppa2per60_df[['Player', 'PP A2/60 Projection', f'PP A2/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(pka2per60_df[['Player', 'PK A2/60 Projection', f'PK A2/60 Projection Sample']], on='Player', how='left')
    distribution_df = distribution_df.merge(standardized_a1_projections, on='Player', how='left')
    distribution_df = distribution_df.merge(standardized_a2_projections, on='Player', how='left')

    distribution_df['GP Projection Sample'] = distribution_df['GP Projection Sample'].apply(ast.literal_eval)
    distribution_df['EV A1/60 Projection Sample'] = distribution_df['EV A1/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['EV A2/60 Projection Sample'] = distribution_df['EV A2/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['EV ATOI Projection Sample'] = distribution_df['EV ATOI Projection Sample'].apply(ast.literal_eval)
    distribution_df['PP A1/60 Projection Sample'] = distribution_df['PP A1/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['PP A2/60 Projection Sample'] = distribution_df['PP A2/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['PP ATOI Projection Sample'] = distribution_df['PP ATOI Projection Sample'].apply(ast.literal_eval)
    distribution_df['PK A1/60 Projection Sample'] = distribution_df['PK A1/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['PK A2/60 Projection Sample'] = distribution_df['PK A2/60 Projection Sample'].apply(ast.literal_eval)
    distribution_df['PK ATOI Projection Sample'] = distribution_df['PK ATOI Projection Sample'].apply(ast.literal_eval)

    host_distribution_df = distribution_df.copy()

    distribution_df['A1 Projection Sample'] = distribution_df.apply(lambda row: [(a*b/60 + c*d/60 + e*f/60) * g for a, b, c, d, e, f, g in zip(row['EV A1/60 Projection Sample'], row['EV ATOI Projection Sample'], row['PP A1/60 Projection Sample'], row['PP A1/60 Projection Sample'], row['PK A1/60 Projection Sample'], row['PK A1/60 Projection Sample'], row['GP Projection Sample'])], axis=1)
    distribution_df['Projected A1 I'] = (distribution_df['EV ATOI Projection']*distribution_df['EV A1/60 Projection']/60 + distribution_df['PP ATOI Projection']*distribution_df['PP A1/60 Projection']/60 + distribution_df['PK ATOI Projection']*distribution_df['PK A1/60 Projection']/60) * distribution_df['GP Projection']
    distribution_df['Stdz. Proj. A1'] = (distribution_df['EV ATOI Projection']*distribution_df['EV A1/60 Stdz. Projection']/60 + distribution_df['PP ATOI Projection']*distribution_df['PP A1/60 Stdz. Projection']/60 + distribution_df['PK ATOI Projection']*distribution_df['PK A1/60 Stdz. Projection']/60) * distribution_df['GP Projection']
    distribution_df = distribution_df.sort_values('Projected A1 I', ascending=False)
    distribution_df = distribution_df.reset_index(drop=True)
    distribution_df.index = distribution_df.index + 1

    distribution_df = distribution_df.drop(columns=[
        'GP Projection', 'GP Projection Sample', 
        'EV ATOI Projection', 'EV ATOI Projection Sample', 
        'PP ATOI Projection', 'PP ATOI Projection Sample', 
        'PK ATOI Projection', 'PK ATOI Projection Sample', 
        'EV A1/60 Projection', 'EV A1/60 Projection Sample', 
        'PP A1/60 Projection', 'PP A1/60 Projection Sample', 
        'PK A1/60 Projection', 'PK A1/60 Projection Sample',
        'EV A1/60 Stdz. Projection', 'PP A1/60 Stdz. Projection', 'PK A1/60 Stdz. Projection',
        'EV A2/60 Projection', 'EV A2/60 Projection Sample', 
        'PP A2/60 Projection', 'PP A2/60 Projection Sample', 
        'PK A2/60 Projection', 'PK A2/60 Projection Sample',
        'EV A2/60 Stdz. Projection', 'PP A2/60 Stdz. Projection', 'PK A2/60 Stdz. Projection'])
    
    distribution_df['Projected A1 II'] = distribution_df['Projected A1 I']*temperature + distribution_df['Stdz. Proj. A1']*(1-temperature)

    # Era Adjustment
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

    distribution_df = distribution_df.sort_values('Projected A1 II', ascending=False)
    distribution_df = distribution_df.reset_index(drop=True)
    distribution_df.index = distribution_df.index + 1
    distribution_df['Era Adjustment Factor'] = (hist_a1_df['Smoothed Adjustment']/distribution_df['Projected A1 II'])*era_temp + 1
    distribution_df['Projected A1'] = distribution_df['Projected A1 II']*distribution_df['Era Adjustment Factor']
    distribution_df['A1 Projection Sample'] = distribution_df.apply(lambda row: [x * (row['Projected A1']/row['Projected A1 I']) for x in row['A1 Projection Sample']], axis=1)
    distribution_df = distribution_df.drop(columns=['Projected A1 I', 'Stdz. Proj. A1', 'Projected A1 II', 'Era Adjustment Factor'])

    host_distribution_df['A2 Projection Sample'] = host_distribution_df.apply(lambda row: [(a*b/60 + c*d/60 + e*f/60) * g for a, b, c, d, e, f, g in zip(row['EV A2/60 Projection Sample'], row['EV ATOI Projection Sample'], row['PP A2/60 Projection Sample'], row['PP A2/60 Projection Sample'], row['PK A2/60 Projection Sample'], row['PK A2/60 Projection Sample'], row['GP Projection Sample'])], axis=1)
    host_distribution_df['Projected A2 I'] = (host_distribution_df['EV ATOI Projection']*host_distribution_df['EV A2/60 Projection']/60 + host_distribution_df['PP ATOI Projection']*host_distribution_df['PP A2/60 Projection']/60 + host_distribution_df['PK ATOI Projection']*host_distribution_df['PK A2/60 Projection']/60) * host_distribution_df['GP Projection']
    host_distribution_df['Stdz. Proj. A2'] = (host_distribution_df['EV ATOI Projection']*host_distribution_df['EV A2/60 Stdz. Projection']/60 + host_distribution_df['PP ATOI Projection']*host_distribution_df['PP A2/60 Stdz. Projection']/60 + host_distribution_df['PK ATOI Projection']*host_distribution_df['PK A2/60 Stdz. Projection']/60) * host_distribution_df['GP Projection']
    host_distribution_df = host_distribution_df.sort_values('Projected A2 I', ascending=False)
    host_distribution_df = host_distribution_df.reset_index(drop=True)
    host_distribution_df.index = host_distribution_df.index + 1

    host_distribution_df = host_distribution_df.drop(columns=[
        'GP Projection', 'GP Projection Sample', 
        'EV ATOI Projection', 'EV ATOI Projection Sample', 
        'PP ATOI Projection', 'PP ATOI Projection Sample', 
        'PK ATOI Projection', 'PK ATOI Projection Sample', 
        'EV A2/60 Projection', 'EV A2/60 Projection Sample', 
        'PP A2/60 Projection', 'PP A2/60 Projection Sample', 
        'PK A2/60 Projection', 'PK A2/60 Projection Sample',
        'EV A2/60 Stdz. Projection', 'PP A2/60 Stdz. Projection', 'PK A2/60 Stdz. Projection',
        'EV A2/60 Projection', 'EV A2/60 Projection Sample', 
        'PP A2/60 Projection', 'PP A2/60 Projection Sample', 
        'PK A2/60 Projection', 'PK A2/60 Projection Sample',
        'EV A2/60 Stdz. Projection', 'PP A2/60 Stdz. Projection', 'PK A2/60 Stdz. Projection'])
    
    host_distribution_df['Projected A2 II'] = host_distribution_df['Projected A2 I']*temperature + host_distribution_df['Stdz. Proj. A2']*(1-temperature)

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

    host_distribution_df = host_distribution_df.sort_values('Projected A2 II', ascending=False)
    host_distribution_df = host_distribution_df.reset_index(drop=True)
    host_distribution_df.index = host_distribution_df.index + 1
    host_distribution_df['Era Adjustment Factor'] = (hist_a2_df['Smoothed Adjustment']/host_distribution_df['Projected A2 II'])*era_temp + 1
    host_distribution_df['Projected A2'] = host_distribution_df['Projected A2 II']*host_distribution_df['Era Adjustment Factor']
    host_distribution_df['A2 Projection Sample'] = host_distribution_df.apply(lambda row: [x * (row['Projected A2']/row['Projected A2 I']) for x in row['A2 Projection Sample']], axis=1)
    host_distribution_df = host_distribution_df.drop(columns=['Projected A2 I', 'Stdz. Proj. A2', 'Projected A2 II', 'Era Adjustment Factor'])
    host_distribution_df = host_distribution_df[['Player', 'Position', 'Age', 'Height', 'Weight', 'A2 Projection Sample', 'Projected A2']]

    distribution_df = distribution_df.merge(host_distribution_df[['Player', 'A2 Projection Sample', 'Projected A2']], on='Player', how='left')
    distribution_df['Assist Projection Sample'] = distribution_df.apply(lambda row: [a + b for a, b in zip(row['A1 Projection Sample'], row['A2 Projection Sample'])], axis=1)
    distribution_df['Projected Assists'] = distribution_df['Projected A1'] + distribution_df['Projected A2']
    distribution_df = distribution_df.drop(columns=['A1 Projection Sample', 'A2 Projection Sample'])

    distribution_df['StDev'] = distribution_df['Assist Projection Sample'].apply(lambda x: np.std(x))
    distribution_df['5th Percentile'] = distribution_df['Assist Projection Sample'].apply(lambda x: np.percentile(x, 5))
    distribution_df['95th Percentile'] = distribution_df['Assist Projection Sample'].apply(lambda x: np.percentile(x, 95))
    distribution_df['90A Probability'] = distribution_df['Assist Projection Sample'].apply(lambda x: (sum(value >= 90 for value in x) / len(x)))
    distribution_df['80A Probability'] = distribution_df['Assist Projection Sample'].apply(lambda x: (sum(value >= 80 for value in x) / len(x)))
    distribution_df['70A Probability'] = distribution_df['Assist Projection Sample'].apply(lambda x: (sum(value >= 70 for value in x) / len(x)))
    distribution_df['60A Probability'] = distribution_df['Assist Projection Sample'].apply(lambda x: (sum(value >= 60 for value in x) / len(x)))
    distribution_df['50A Probability'] = distribution_df['Assist Projection Sample'].apply(lambda x: (sum(value >= 50 for value in x) / len(x)))
    distribution_df['40A Probability'] = distribution_df['Assist Projection Sample'].apply(lambda x: (sum(value >= 40 for value in x) / len(x)))
    distribution_df['30A Probability'] = distribution_df['Assist Projection Sample'].apply(lambda x: (sum(value >= 30 for value in x) / len(x)))

    if download_csv == True:
        filename = f'xgboost_assist_distributions_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        distribution_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return distribution_df

def point_distribution_aggregator(g_distribution_df, a_distribution_df, year, download_csv):
    distribution_df = g_distribution_df[['Player', 'Projected Goals', 'Goal Projection Sample']].merge(a_distribution_df[['Player', 'Projected A1', 'Projected A2', 'Projected Assists', 'Assist Projection Sample']], on='Player', how='left')
    distribution_df['Point Projection Sample'] = distribution_df.apply(lambda row: [a + b for a, b in zip(row['Assist Projection Sample'], row['Goal Projection Sample'])], axis=1)
    distribution_df['Projected Points'] = distribution_df['Projected Assists'] + distribution_df['Projected Goals']
    distribution_df = distribution_df.drop(columns=['Goal Projection Sample', 'Assist Projection Sample'])

    distribution_df['StDev'] = distribution_df['Point Projection Sample'].apply(lambda x: np.std(x))
    distribution_df['5th Percentile'] = distribution_df['Point Projection Sample'].apply(lambda x: np.percentile(x, 5))
    distribution_df['95th Percentile'] = distribution_df['Point Projection Sample'].apply(lambda x: np.percentile(x, 95))
    distribution_df['150P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 150 for value in x) / len(x)))
    distribution_df['140P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 140 for value in x) / len(x)))
    distribution_df['130P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 130 for value in x) / len(x)))
    distribution_df['120P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 120 for value in x) / len(x)))
    distribution_df['110P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 110 for value in x) / len(x)))
    distribution_df['100P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 100 for value in x) / len(x)))
    distribution_df['90P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 90 for value in x) / len(x)))
    distribution_df['80P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 80 for value in x) / len(x)))
    distribution_df['70P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 70 for value in x) / len(x)))
    distribution_df['60P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 60 for value in x) / len(x)))
    distribution_df['50P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 50 for value in x) / len(x)))
    distribution_df['40P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 40 for value in x) / len(x)))
    distribution_df['30P Probability'] = distribution_df['Point Projection Sample'].apply(lambda x: (sum(value >= 30 for value in x) / len(x)))

    if download_csv == True:
        filename = f'xgboost_point_distributions_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        distribution_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

    return distribution_df

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

    standardized_g_projections = goal_projection_standardizer(stat_df, projection_df, year)
    g_distribution_df = goal_distribution_aggregator(stat_df, projection_df, standardized_g_projections, 0.11148025, 0.71029331, year+1, True)
    standardized_a1_projections = a1_projection_standardizer(stat_df, projection_df, year)
    standardized_a2_projections = a2_projection_standardizer(stat_df, projection_df, year)
    a_distribution_df = assist_distribution_aggregator(stat_df, projection_df, standardized_a1_projections, standardized_a2_projections, 0.13907949, 0.97590236, year+1, True)
    p_distribution_df = point_distribution_aggregator(g_distribution_df, a_distribution_df, year+1, True)

    projection_df['ATOI'] = projection_df['EV ATOI'] + projection_df['PP ATOI'] + projection_df['PK ATOI']
    projection_df = projection_df.drop(columns=['EV ATOI', 'PP ATOI', 'PK ATOI', 'EV G/60', 'PP G/60', 'PK G/60', 'EV A1/60', 'EV A2/60', 'PP A2/60', 'PK A1/60', 'PK A2/60', 'PP A1/60'])

    projection_df = projection_df.merge(g_distribution_df[['Player', 'Projected Goals']], on='Player', how='left')
    projection_df = projection_df.merge(a_distribution_df[['Player', 'Projected A1', 'Projected A2', 'Projected Assists']], on='Player', how='left')
    projection_df['Points'] = projection_df['Projected Goals'] + projection_df['Projected Assists']
    projection_df.rename(columns={'GP': 'Games', 'ATOI': 'Average TOI', 'Projected Goals': 'Goals', 'Projected A1': 'Primary Assists', 'Projected A2': 'Secondary Assists', 'Projected Assists': 'Total Assists'}, inplace=True)

    projection_df = projection_df.sort_values('Points', ascending=False)
    projection_df = projection_df.reset_index(drop=True)
    projection_df.index = projection_df.index + 1
    print(projection_df.head(20))

    if download_csv == True:
        filename = f'xgboost_final_projections_{year}'
        if not os.path.exists(f'{os.path.dirname(__file__)}/CSV Data'):
            os.makedirs(f'{os.path.dirname(__file__)}/CSV Data')
        projection_df.to_csv(f'{os.path.dirname(__file__)}/CSV Data/{filename}.csv')
        print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/CSV Data')

make_projections(True, True, 2024, True)
