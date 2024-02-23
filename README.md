# NHL Projections 2023

## Overview
Welcome to the NHL Projections 2023 GitHub repository! This repository houses a comprehensive stacked ensemble machine learning pipeline focused on predicting player statistics and generating probability distributions of various stats for 950 NHL players. In this README, I'll provide a detailed overview of the repository, including how the model works, the data pipeline, the machine learning algorithms employed, and the results obtained.

## Website
Explore the final projected statistics on the [NHL Projections 2023 website](https://andrewderango.github.io/NHL-Projections-2023/ "Projections Page")! Deployed using GitHub pages, the website was developed using JavaScript, HTML and CSS to display the projections on a simple website. Although this does not contain uncertainty estimations, it shows the final projected stats as of September 2023. The source can be found in ``/docs``.

## The Process

### Summary
The stacked machine learning model consists of multiple base models trained independently on historical NHL player data. These base models include decision trees, random forests, support vector machines, and neural networks, each capturing different patterns and dependencies in the data. The predictions of these base models, along with additional engineered features, serve as input to a meta-model (blender model), which is an extreme gradient boosting (XGBoost) machine. The meta-model learns to combine the predictions of the base models, resulting in more accurate and robust projections of player statistics. Additionally, bootstrapping techniques and Kernel Density Estimation (KDE) are employed to estimate the uncertainty associated with the predictions and generate probability distributions for various player statistics to provide valuable insights into the range of potential outcomes.

### Data Collection and Preprocessing
The data collection process begins with scraping historical player bios and statistics from [NaturalStatTrick](https://www.naturalstattrick.com "NST Homepage") for every year between 2007 and 2023. Player bios provide crucial information such as date of birth and position, while player statistics encompass individual data, on-ice data, even strength data, powerplay data, and shorthanded data for each season. This extensive data is organized into separate CSV files and then consolidated into "instance dataframes". Each row in these dataframes contains a player's stats within a season, along with raw data from their performance in the previous four seasons. This effectively enables us to analyze correlations between past performance and future projections, which is indubitably important for training the models. This is primarily done in ``preprocessing_training_functions.py``.

### Base Machine Learning Models
With the preprocessed data in hand, the pipeline employs more than 50 different models based on five distinct machine learning algorithms to predict a range of player statistics. This includes games played, time on ice per game, goal rate, primary assist rate, and secondary assist rate for each of the different situations, such as even strength, powerplay, and shorthanded. Using scikit-learn and TensorFlow, the following algorithms formed the basis for the base layer of the stacked model: Ridge regression, random forest regression, support vector regression, feedforward neural networks, and Bayesian neural networks. These algorithms are individually applied to forwards and defensemen, and separate models are trained for players with varying amounts of historical data (i.e., four, three, two, and one prior seasons).

### Hyperparameter Tuning and Feature Selection
To optimize the performance of each machine learning model, an extensive hyperparameter tuning process is conducted. This involves hyperparameters such as hidden layers, layer sizes, learning rates, alpha/gamma values, tree depth, scaling methods, etc. A variety of techniques were used in these processes, including 10-fold cross-validation to assess model performance across various architectures and hyperparameters. Additionally, various feature selection algorithms were undergone to determine the optimal features in cost  minimization via permutation importance, mean decrease of impurity, etc. Some features to predict future statistics used include age, historical goals, primary assists, secondary assists, individual expected goals, Corsi, Fenwick, rush attempts, rebounds created, high-danger scoring chances, on-ice expected goals, shooting percentage, shooting talent, etc.

After careful analysis, we discovered that projecting time on ice, games played, and player statistics in specific game situations (even strength, power play, penalty kill) yields a more accurate prediction of total goals. This approach isolates randomness from individual events, thereby providing a richer source of information for model learning. Moreover, it minimizes bias within the input dataset, enhancing the predictive power of the model for future statistics.

### Stacking the Models
The base models mentioned previously were then used to make predictions for each of the previous 15 years. Aggregating this data, a stacked algorithm approach is adopted to improve prediction accuracy. The outputs of the five tuned machine learning models are combined using an XGBoost model, effectively creating a stacked model. This approach, validated through 10-fold cross-validation, is found to produce the most accurate projections.

### Uncertainty Estimations and Probability Distributions
To provide insight into the uncertainty associated with the projected statistics, the training dataframes for the XGBoost model are bootstrapped. This generates uncertainty estimations for predicted stats, which are then combined with kernel density estimation (KDE) to generate probability distributions for player goals, assists, and points.

## Contributing
Contributions to this project are welcome! If you have any ideas for improvements, feature requests, or bug fixes, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](https://github.com/andrewderango/NHL-Projections-2023/blob/main/LICENSE "Repo License").

## Conclusion
This project leveraged advanced machine learning and data engineering techniques to forecast player statistics with unprecedented historical accuracy. The stacked model approach, coupled with extensive data preprocessing and model tuning, has enabled us to establish a robust framework capable of producing reliable projections and uncertainty estimations for over 900 players for the 2023-24 NHL season.
