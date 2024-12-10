# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:38:19 2024

@author: bryan
"""





# %% imports
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nba_api.stats.endpoints import playergamelogs
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from nba_api.stats.endpoints import leaguegamefinder
import ast
# %% some flags

currentSeasonID = 22024









# %% prepare overall dataset



# Goal is to make streamlit website where the player inputs the teams they want. 
# we do the processing and predictions. and then finally it gives the score below
# will just make the barebones outline with those two inputs in


@st.cache_data
def load_and_aggregate_data():
    # Step 1: Load the overall dataset
    notyet = pd.read_csv('finalized_dataset.csv')
    
    rows1 = notyet.iloc[::2].reset_index(drop=True)  # First of each pair (e.g., "L" rows)
    rows2 = notyet.iloc[1::2].reset_index(drop=True)  # Second of each pair (e.g., "W" rows)

    # Rename columns for rows1 and rows2
    rows1 = rows1.add_suffix('1')
    rows2 = rows2.add_suffix('2')

    # Combine the rows
    combinedshuffled_df = pd.concat([rows1, rows2], axis=1)
    combinedshuffled_df = combinedshuffled_df.drop(columns=['Unnamed: 01', 'Unnamed: 02'])
    combinedshuffled_df['WL1'] = combinedshuffled_df['WL1'].map({'L': 0, 'W': 1})
    combinedshuffled_df['WL2'] = combinedshuffled_df['WL2'].map({'L': 0, 'W': 1})
    combinedshuffled_df = combinedshuffled_df.drop(columns=['Game_ID1', 'Game_ID2'])
    combinedshuffled_df = combinedshuffled_df.dropna()

    # Step 2: Generate a random mask to decide which rows to swap
    group1_cols = combinedshuffled_df.columns[0:10]  # Columns 0-9
    group2_cols = combinedshuffled_df.columns[10:20]  # Columns 10-19
    swap_mask = np.random.choice([True, False], size=len(combinedshuffled_df))

    # Step 3: Swap values between Group 1 and Group 2 for the selected rows
    for i in range(len(combinedshuffled_df)):
        if swap_mask[i]:
            # Swap Group 1 and Group 2 values for row i
            combinedshuffled_df.loc[i, group1_cols], combinedshuffled_df.loc[i, group2_cols] = combinedshuffled_df.loc[i, group2_cols].values, combinedshuffled_df.loc[i, group1_cols].values

    combinedshuffled_df = combinedshuffled_df.drop(columns=['WL2'])

    # Load and filter the player game data
    allPlayerGamesEver = pd.read_csv('allPlayerGamesEver.csv')
    allPlayerGamesEverWhoPlayed = allPlayerGamesEver[allPlayerGamesEver['MIN'] > 0]
    allPlayerGamesEverWhoPlayed['GAME_DATE'] = pd.to_datetime(allPlayerGamesEverWhoPlayed['GAME_DATE'])
    allPlayerGamesEverWhoPlayed = allPlayerGamesEverWhoPlayed.sort_values(by='GAME_DATE')

    # Filter the games based on the date
    filtered_allPlayerGamesEverWhoPlayed = allPlayerGamesEverWhoPlayed[allPlayerGamesEverWhoPlayed['GAME_DATE'] >= "1996-11-01 00:00:00"]
    filtered_allPlayerGamesEverWhoPlayed = filtered_allPlayerGamesEverWhoPlayed.drop("PLUS_MINUS", axis=1)
    filtered_allPlayerGamesEverWhoPlayed = filtered_allPlayerGamesEverWhoPlayed.drop("VIDEO_AVAILABLE", axis=1)

    # Step 4: Aggregate the data for both Win and Loss games
    grouped = filtered_allPlayerGamesEverWhoPlayed.groupby('Game_ID')

    aggregatedL = grouped.apply(lambda x: x[x['WL'] == 'L'].agg({
        'DREB': 'sum',
        'OREB': 'sum',
        'FGA': 'sum',
        'TOV': 'sum',
        'FTA': 'sum',
        'MIN': 'sum',
        'FGM': 'sum',
        'FG3A': 'sum'
    }))

    aggregatedW = grouped.apply(lambda x: x[x['WL'] == 'W'].agg({
        'DREB': 'sum',
        'OREB': 'sum',
        'FGA': 'sum',
        'TOV': 'sum',
        'FTA': 'sum',
        'MIN': 'sum',
        'FGM': 'sum',
        'FG3A': 'sum'
    }))

    aggregatedW['opponent_possessions'] = (
        aggregatedL['FGA'] - aggregatedL['OREB'] + aggregatedL['TOV'] + 0.44 * aggregatedL['FTA']
    )
    aggregatedL['opponent_possessions'] = (
        aggregatedW['FGA'] - aggregatedW['OREB'] + aggregatedW['TOV'] + 0.44 * aggregatedW['FTA']
    )

    aggregatedL['Game_ID'] = aggregatedL.index.get_level_values('Game_ID')
    aggregatedW['Game_ID'] = aggregatedW.index.get_level_values('Game_ID')

    # Continue with the aggregations for other stats
    aggregatedL['opponent_defensive_rebounds'] = aggregatedW['DREB']
    aggregatedW['opponent_defensive_rebounds'] = aggregatedL['DREB']
    
    aggregatedL['opponent_offensive_rebounds'] = aggregatedW['OREB']
    aggregatedW['opponent_offensive_rebounds'] = aggregatedL['OREB']
    
    aggregatedL['opponent_FGA'] = aggregatedW['FGA']
    aggregatedW['opponent_FGA'] = aggregatedL['FGA']
    
    aggregatedL['opponent_FG3A'] = aggregatedW['FG3A']
    aggregatedW['opponent_FG3A'] = aggregatedL['FG3A']

    aggregatedL['WL'] = "L"
    aggregatedW['WL'] = "W"

    return combinedshuffled_df, aggregatedL, aggregatedW, filtered_allPlayerGamesEverWhoPlayed


# Call the function within Streamlit
combinedshuffled_df, aggregatedL, aggregatedW, filtered_allPlayerGamesEverWhoPlayed = load_and_aggregate_data()




# %% functions needed

# ALL ADVANCED METRICS

# 1. True Shooting Percentage (TS%)
def calculate_ts_uw(points, fga, fta):
    # If both FGA and FTA are zero, return None or NaN to signify undefined TS%

    return points / (2 * (fga + 0.44 * fta))  # Returns as decimal (0 to 1)

# 2. Player Efficiency Rating (PER)
# def calculate_per_uw(points, rebounds, assists, steals, blocks, turnovers, fg, fga, ft, fta, min_per, max_per):
#     raw_per = (
#         points
#         + 0.5 * assists
#         + 0.7 * rebounds
#         + 1.5 * steals
#         + 1.7 * blocks
#         - 1.0 * turnovers
#     ) / (fg + fga + ft + fta)
#     return (raw_per - min_per) / (max_per - min_per)  # Returns as decimal (0 to 1)

# 3. 3-Point Attempt Rate (3PAr)
def calculate_3par_uw(three_point_attempts, field_goal_attempts):
    return three_point_attempts / field_goal_attempts  # Returns as decimal (0 to 1)

# 4. Free Throw Attempt Rate (FTr)
def calculate_ftr_uw(free_throw_attempts, field_goal_attempts):
    return free_throw_attempts / field_goal_attempts  # Returns as decimal (0 to 1)

# 5. Offensive Rebound Percentage (ORB%)
def calculate_orb_percentage_uw(offensive_rebounds, player_minutes, team_minutes, team_offensive_rebounds, opponent_defensive_rebounds):
    # Calculate Offensive Rebound Percentage based on the new equation
    numerator = offensive_rebounds * (team_minutes / 5)
    denominator = player_minutes * (team_offensive_rebounds + opponent_defensive_rebounds)

    return numerator / denominator  # Returns as decimal (0 to 1)

# 6. Defensive Rebound Percentage (DRB%)
def calculate_drb_percentage_uw(defensive_rebounds, player_minutes, team_minutes, team_defensive_rebounds, opponent_offensive_rebounds):
    # Calculate Defensive Rebound Percentage based on the new equation
    return (defensive_rebounds * (team_minutes / 5)) / (player_minutes * (team_defensive_rebounds + opponent_offensive_rebounds))

# 7. Assist Percentage (AST%)
def calculate_ast_percentage_uw(player_ast, player_mp, team_mp, team_fg, player_fg):
    # Calculate AST% based on the formula
    return player_ast / (((player_mp / (team_mp / 5)) * team_fg) - player_fg)

# 8. Steal Percentage (STL%)
# def calculate_possessions(fga, oreb, tov, fta):
#     return fga - oreb + tov + 0.44 * fta
def calculate_stl_percentage_uw(steals, player_minutes, team_minutes, opponent_possessions):
    # Calculate Steal Percentage based on the new equation
    return (steals * (team_minutes / 5)) / (player_minutes * opponent_possessions)

# 9. Block Percentage (BLK%)
def calculate_blk_percentage_uw(blocks, player_minutes, team_minutes, opponent_fga, opponent_FG3A):
    # Calculate Block Percentage based on the new equation
    return (blocks * (team_minutes / 5)) / (player_minutes * (opponent_fga - opponent_FG3A))

# 10. Turnover Percentage (TOV%)
def calculate_tov_percentage_uw(turnovers, field_goal_attempts, free_throw_attempts):
    return turnovers / (field_goal_attempts + 0.44 * free_throw_attempts + turnovers)  # Returns as decimal (0 to 1)

def weighted_advanced_metric(metric_percentages):
    # Filter out None or NaN values and corresponding weights
    valid_metrics = []
    valid_weights = []

    for i, metric in enumerate(metric_percentages):
        if metric is not None and not np.isnan(metric):
            valid_metrics.append(metric)
            valid_weights.append(i + 1)  # Weights are just index + 1, i.e., 1, 2, 3, ...

    # If there are no valid metrics left after filtering, return 0 or some default value
    if not valid_metrics:
        print("There is nothing in metric_percentages...", metric_percentages)
        return 0  # You can choose another default value if needed

    # Apply weights to the valid metrics
    weighted_metric = np.multiply(valid_metrics, valid_weights)

    # Compute weighted average
    return weighted_metric.sum() / np.sum(valid_weights)

def getCareerStatsBeforeDateRow(gamedate, playerid):
    # Convert game date to datetime format once
    gamedateTime = pd.to_datetime(gamedate)

    # Filter the data once
    filtered_rows = filtered_allPlayerGamesEverWhoPlayed[
        (filtered_allPlayerGamesEverWhoPlayed['Player_ID'] == playerid) &
        (filtered_allPlayerGamesEverWhoPlayed['GAME_DATE'] < gamedateTime)
    ]
    if len(filtered_rows) == 0:
      # print("Either gamedate or playerid does not exist")
      # or could be rookie
      return np.nan

    # Convert the filtered DataFrame to NumPy arrays for better performance
    player_games = filtered_rows.to_numpy()

    # Extract the relevant columns as NumPy arrays for efficient processing
    MIN = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('MIN')]
    FGM = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('FGM')]
    FGA = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('FGA')]
    FG3A = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('FG3A')]
    FTA = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('FTA')]
    OREB = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('OREB')]
    DREB = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('DREB')]
    AST = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('AST')]
    STL = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('STL')]
    BLK = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('BLK')]
    TOV = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('TOV')]
    PTS = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('PTS')]
    PF = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('PF')]

    game_id = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('Game_ID')]
    wl = player_games[:, filtered_allPlayerGamesEverWhoPlayed.columns.get_loc('WL')]


    # initalize
    # Create dictionaries for each stat in aggregateL and aggregateW
    game_id_to_DREB_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['DREB']))
    game_id_to_OREB_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['OREB']))
    game_id_to_MIN_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['MIN']))
    game_id_to_FGM_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['FGM']))
    game_id_to_possessions_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['opponent_possessions']))
    game_id_to_FGA_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['opponent_FGA']))
    game_id_to_FG3A_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['opponent_FG3A']))

    game_id_to_DREB_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['DREB']))
    game_id_to_OREB_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['OREB']))
    game_id_to_MIN_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['MIN']))
    game_id_to_FGM_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['FGM']))
    game_id_to_possessions_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['opponent_possessions']))
    game_id_to_FGA_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['opponent_FGA']))
    game_id_to_FG3A_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['opponent_FG3A']))

    # Initialize empty lists to store the stats
    team_OREB = np.zeros(game_id.shape)
    team_DREB = np.zeros(game_id.shape)
    team_MIN = np.zeros(game_id.shape)
    team_FGM = np.zeros(game_id.shape)
    opponent_DREB = np.zeros(game_id.shape)
    opponent_OREB = np.zeros(game_id.shape)
    opponent_possessions = np.zeros(game_id.shape)
    opponent_FGA = np.zeros(game_id.shape)
    opponent_FG3A = np.zeros(game_id.shape)

    # Loop through each game and get the appropriate stats based on WL
    for i in range(len(game_id)):
        if wl[i] == 'W':
            # For team (WL = 'W'), use data from aggregateW
            team_OREB[i] = game_id_to_OREB_W.get(game_id[i], np.nan)
            team_DREB[i] = game_id_to_DREB_W.get(game_id[i], np.nan)
            team_MIN[i] = game_id_to_MIN_W.get(game_id[i], np.nan)
            team_FGM[i] = game_id_to_FGM_W.get(game_id[i], np.nan)

            # For opponent (WL = 'W'), use data from aggregateL
            opponent_DREB[i] = game_id_to_DREB_L.get(game_id[i], np.nan)
            opponent_OREB[i] = game_id_to_OREB_L.get(game_id[i], np.nan)
            opponent_possessions[i] = game_id_to_possessions_L.get(game_id[i], np.nan)
            opponent_FGA[i] = game_id_to_FGA_L.get(game_id[i], np.nan)
            opponent_FG3A[i] = game_id_to_FG3A_L.get(game_id[i], np.nan)

        else:
            # For team (WL = 'L'), use data from aggregateL
            team_DREB[i] = game_id_to_DREB_L.get(game_id[i], np.nan)
            team_OREB[i] = game_id_to_OREB_L.get(game_id[i], np.nan)
            team_MIN[i] = game_id_to_MIN_L.get(game_id[i], np.nan)
            team_FGM[i] = game_id_to_FGM_L.get(game_id[i], np.nan)

            # For opponent (WL = 'L'), use data from aggregateW
            opponent_DREB[i] = game_id_to_DREB_W.get(game_id[i], np.nan)
            opponent_OREB[i] = game_id_to_OREB_W.get(game_id[i], np.nan)
            opponent_possessions[i] = game_id_to_possessions_W.get(game_id[i], np.nan)
            opponent_FGA[i] = game_id_to_FGA_W.get(game_id[i], np.nan)
            opponent_FG3A[i] = game_id_to_FG3A_W.get(game_id[i], np.nan)

    # Now the arrays contain the aggregated stats for each game and its opponent

    # Calculate advanced metrics using NumPy for vectorization
    ts_mask_PTS = PTS[2 * (FGA + 0.44 * FTA) > 0]
    ts_mask_FGA = FGA[2 * (FGA + 0.44 * FTA) > 0]
    ts_mask_FTA = FTA[2 * (FGA + 0.44 * FTA) > 0]
    ts_uw = calculate_ts_uw(ts_mask_PTS, ts_mask_FGA, ts_mask_FTA)

    # Similarly for other advanced metrics, e.g., 3par_uw
    par_FGA = FGA[FGA!= 0]
    par_FG3A = FG3A[FGA!= 0]
    par_uw = calculate_3par_uw(par_FG3A, par_FGA)

    # ftr_uw
    ftr_FGA = FGA[FGA!= 0]
    ftr_FTA = FTA[FGA!= 0]
    ftr_uw = calculate_ftr_uw(ftr_FTA, ftr_FGA)

    # orb_percentage_uw
    orb_OREB = OREB[(MIN * (team_OREB + opponent_DREB)) > 0]
    orb_MIN = MIN[(MIN * (team_OREB + opponent_DREB)) > 0]
    orb_team_MIN = team_MIN[(MIN * (team_OREB + opponent_DREB)) > 0]
    orb_team_OREB = team_OREB[(MIN * (team_OREB + opponent_DREB)) > 0]
    orb_opponent_DREB = opponent_DREB[(MIN * (team_OREB + opponent_DREB)) > 0]
    orb_percentage_uw = calculate_orb_percentage_uw(orb_OREB, orb_MIN, orb_team_MIN, orb_team_OREB, orb_opponent_DREB)

    # drb_percentage_uw
    drb_DREB = DREB[(MIN * (team_DREB + opponent_OREB)) > 0]
    drb_MIN = MIN[(MIN * (team_DREB + opponent_OREB)) > 0]
    drb_team_MIN = team_MIN[(MIN * (team_DREB + opponent_OREB)) > 0]
    drb_team_DREB = team_DREB[(MIN * (team_DREB + opponent_OREB)) > 0]
    drb_opponent_OREB = opponent_OREB[(MIN * (team_DREB + opponent_OREB)) > 0]
    drb_percentage_uw = calculate_drb_percentage_uw(drb_DREB, drb_MIN, drb_team_MIN, drb_team_DREB, drb_opponent_OREB)

    # ast_percentage_uw
    ast_AST = AST[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
    ast_MIN = MIN[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
    ast_team_MIN = team_MIN[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
    ast_team_FGM = team_FGM[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
    ast_FGM = FGM[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
    ast_percentage_uw = calculate_ast_percentage_uw(ast_AST, ast_MIN, ast_team_MIN, ast_team_FGM, ast_FGM)

    # stl_percentage_uw
    stl_STL = STL[(MIN * opponent_possessions) > 0]
    stl_MIN = MIN[(MIN * opponent_possessions) > 0]
    stl_team_MIN = team_MIN[(MIN * opponent_possessions) > 0]
    stl_opponent_possessions = opponent_possessions[(MIN * opponent_possessions) > 0]
    stl_percentage_uw = calculate_stl_percentage_uw(stl_STL, stl_MIN, stl_team_MIN, stl_opponent_possessions)

    # blk_percentage_uw
    blk_BLK = BLK[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
    blk_MIN = MIN[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
    blk_team_MIN = team_MIN[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
    blk_opponent_FGA = opponent_FGA[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
    blk_opponent_FG3A = opponent_FG3A[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
    blk_percentage_uw = calculate_blk_percentage_uw(blk_BLK, blk_MIN, blk_team_MIN, blk_opponent_FGA, blk_opponent_FG3A)

    # tov_percentage_uw
    tov_TOV = TOV[(TOV + 0.44 * PF) > 0]
    tov_FGA = FGA[(TOV + 0.44 * PF) > 0]
    tov_PF = PF[(TOV + 0.44 * PF) > 0]
    tov_percentage_uw = calculate_tov_percentage_uw(tov_TOV, tov_FGA, tov_PF)

    return pd.DataFrame([[weighted_advanced_metric(ts_uw), weighted_advanced_metric(par_uw),weighted_advanced_metric(ftr_uw),weighted_advanced_metric(orb_percentage_uw),weighted_advanced_metric(drb_percentage_uw),weighted_advanced_metric(ast_percentage_uw),weighted_advanced_metric(stl_percentage_uw),weighted_advanced_metric(blk_percentage_uw),weighted_advanced_metric(tov_percentage_uw)]], columns=["ts_uw","3par_uw","ftr_uw","orb_percentage_uw","drb_percentage_uw","ast_percentage_uw","stl_percentage_uw","blk_percentage_uw","tov_percentage_uw"]).values[0]

def getCareerStatsBeforeDateRows(gamedates, playerids):
    # Ensure gamedates are in datetime format
    gamedates = pd.to_datetime(gamedates)
    
    # Create NumPy arrays from the full dataset (for filtering)
    # all_game_ids = filtered_allPlayerGamesEverWhoPlayed['Game_ID'].values
    all_player_ids = filtered_allPlayerGamesEverWhoPlayed['Player_ID'].values
    all_game_dates = pd.to_datetime(filtered_allPlayerGamesEverWhoPlayed['GAME_DATE'].values)

    # Prepare the result list
    results = []

    # Loop through each gamedate and playerid pair
    for gamedate, playerid in zip(gamedates, playerids):
        # Convert the gamedate to datetime for each iteration
        gamedateTime = pd.to_datetime(gamedate)

        # Get the matching indices for the current gamedate and playerid
        matching_indices = np.where((all_player_ids == playerid) & (all_game_dates < gamedateTime))[0]
        
        if len(matching_indices) == 0:
            # If no match, append np.nan (or handle as needed)
            results.append(np.nan)
            continue

        # Extract the rows from the matching indices
        filtered_rows = filtered_allPlayerGamesEverWhoPlayed.iloc[matching_indices]

        # Extract necessary columns (e.g., MIN, FGM, FGA, etc.)
        MIN = filtered_rows['MIN'].values
        FGM = filtered_rows['FGM'].values
        FGA = filtered_rows['FGA'].values
        FG3A = filtered_rows['FG3A'].values
        FTA = filtered_rows['FTA'].values
        OREB = filtered_rows['OREB'].values
        DREB = filtered_rows['DREB'].values
        AST = filtered_rows['AST'].values
        STL = filtered_rows['STL'].values
        BLK = filtered_rows['BLK'].values
        TOV = filtered_rows['TOV'].values
        PTS = filtered_rows['PTS'].values
        PF = filtered_rows['PF'].values
        game_id = filtered_rows['Game_ID'].values
        wl = filtered_rows['WL'].values


        # initialize values
        game_id_to_DREB_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['DREB']))
        game_id_to_OREB_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['OREB']))
        game_id_to_MIN_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['MIN']))
        game_id_to_FGM_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['FGM']))
        game_id_to_possessions_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['opponent_possessions']))
        game_id_to_FGA_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['opponent_FGA']))
        game_id_to_FG3A_L = dict(zip(aggregatedL['Game_ID'], aggregatedL['opponent_FG3A']))
    
        game_id_to_DREB_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['DREB']))
        game_id_to_OREB_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['OREB']))
        game_id_to_MIN_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['MIN']))
        game_id_to_FGM_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['FGM']))
        game_id_to_possessions_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['opponent_possessions']))
        game_id_to_FGA_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['opponent_FGA']))
        game_id_to_FG3A_W = dict(zip(aggregatedW['Game_ID'], aggregatedW['opponent_FG3A']))

        # Initialize arrays for team and opponent stats (if needed)
        team_OREB = np.zeros_like(game_id)
        team_DREB = np.zeros_like(game_id)
        team_MIN = np.zeros_like(game_id)
        team_FGM = np.zeros_like(game_id)
        opponent_DREB = np.zeros_like(game_id)
        opponent_OREB = np.zeros_like(game_id)
        opponent_possessions = np.zeros_like(game_id)
        opponent_FGA = np.zeros_like(game_id)
        opponent_FG3A = np.zeros_like(game_id)

        # Populate team and opponent stats based on WL (Winner/Loser)
        team_OREB[wl == 'W'] = [game_id_to_OREB_W.get(game, np.nan) for game in game_id[wl == 'W']]
        team_DREB[wl == 'W'] = [game_id_to_DREB_W.get(game, np.nan) for game in game_id[wl == 'W']]
        team_MIN[wl == 'W'] = [game_id_to_MIN_W.get(game, np.nan) for game in game_id[wl == 'W']]
        team_FGM[wl == 'W'] = [game_id_to_FGM_W.get(game, np.nan) for game in game_id[wl == 'W']]
        opponent_DREB[wl == 'W'] = [game_id_to_DREB_L.get(game, np.nan) for game in game_id[wl == 'W']]
        opponent_OREB[wl == 'W'] = [game_id_to_OREB_L.get(game, np.nan) for game in game_id[wl == 'W']]
        opponent_possessions[wl == 'W'] = [game_id_to_possessions_L.get(game, np.nan) for game in game_id[wl == 'W']]
        opponent_FGA[wl == 'W'] = [game_id_to_FGA_L.get(game, np.nan) for game in game_id[wl == 'W']]
        opponent_FG3A[wl == 'W'] = [game_id_to_FG3A_L.get(game, np.nan) for game in game_id[wl == 'W']]

        # Repeat for 'L' games (opposite data source)
        team_OREB[wl == 'L'] = [game_id_to_OREB_L.get(game, np.nan) for game in game_id[wl == 'L']]
        team_DREB[wl == 'L'] = [game_id_to_DREB_L.get(game, np.nan) for game in game_id[wl == 'L']]
        team_MIN[wl == 'L'] = [game_id_to_MIN_L.get(game, np.nan) for game in game_id[wl == 'L']]
        team_FGM[wl == 'L'] = [game_id_to_FGM_L.get(game, np.nan) for game in game_id[wl == 'L']]
        opponent_DREB[wl == 'L'] = [game_id_to_DREB_W.get(game, np.nan) for game in game_id[wl == 'L']]
        opponent_OREB[wl == 'L'] = [game_id_to_OREB_W.get(game, np.nan) for game in game_id[wl == 'L']]
        opponent_possessions[wl == 'L'] = [game_id_to_possessions_W.get(game, np.nan) for game in game_id[wl == 'L']]
        opponent_FGA[wl == 'L'] = [game_id_to_FGA_W.get(game, np.nan) for game in game_id[wl == 'L']]
        opponent_FG3A[wl == 'L'] = [game_id_to_FG3A_W.get(game, np.nan) for game in game_id[wl == 'L']]

        # Calculate advanced metrics
        ts_mask_PTS = PTS[2 * (FGA + 0.44 * FTA) > 0]
        ts_mask_FGA = FGA[2 * (FGA + 0.44 * FTA) > 0]
        ts_mask_FTA = FTA[2 * (FGA + 0.44 * FTA) > 0]
        ts_uw = calculate_ts_uw(ts_mask_PTS, ts_mask_FGA, ts_mask_FTA)

        # Other advanced metrics
        par_FGA = FGA[FGA != 0]
        par_FG3A = FG3A[FGA != 0]
        par_uw = calculate_3par_uw(par_FG3A, par_FGA)
        
        # ftr_uw
        ftr_FGA = FGA[FGA!= 0]
        ftr_FTA = FTA[FGA!= 0]
        ftr_uw = calculate_ftr_uw(ftr_FTA, ftr_FGA)
        
        # orb_percentage_uw
        orb_OREB = OREB[(MIN * (team_OREB + opponent_DREB)) > 0]
        orb_MIN = MIN[(MIN * (team_OREB + opponent_DREB)) > 0]
        orb_team_MIN = team_MIN[(MIN * (team_OREB + opponent_DREB)) > 0]
        orb_team_OREB = team_OREB[(MIN * (team_OREB + opponent_DREB)) > 0]
        orb_opponent_DREB = opponent_DREB[(MIN * (team_OREB + opponent_DREB)) > 0]
        orb_percentage_uw = calculate_orb_percentage_uw(orb_OREB, orb_MIN, orb_team_MIN, orb_team_OREB, orb_opponent_DREB)
        
        # drb_percentage_uw
        drb_DREB = DREB[(MIN * (team_DREB + opponent_OREB)) > 0]
        drb_MIN = MIN[(MIN * (team_DREB + opponent_OREB)) > 0]
        drb_team_MIN = team_MIN[(MIN * (team_DREB + opponent_OREB)) > 0]
        drb_team_DREB = team_DREB[(MIN * (team_DREB + opponent_OREB)) > 0]
        drb_opponent_OREB = opponent_OREB[(MIN * (team_DREB + opponent_OREB)) > 0]
        drb_percentage_uw = calculate_drb_percentage_uw(drb_DREB, drb_MIN, drb_team_MIN, drb_team_DREB, drb_opponent_OREB)
        
        # ast_percentage_uw
        ast_AST = AST[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
        ast_MIN = MIN[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
        ast_team_MIN = team_MIN[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
        ast_team_FGM = team_FGM[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
        ast_FGM = FGM[((((MIN / (team_MIN / 5)) * team_FGM) - FGM)) > 0]
        ast_percentage_uw = calculate_ast_percentage_uw(ast_AST, ast_MIN, ast_team_MIN, ast_team_FGM, ast_FGM)
        
        # stl_percentage_uw
        stl_STL = STL[(MIN * opponent_possessions) > 0]
        stl_MIN = MIN[(MIN * opponent_possessions) > 0]
        stl_team_MIN = team_MIN[(MIN * opponent_possessions) > 0]
        stl_opponent_possessions = opponent_possessions[(MIN * opponent_possessions) > 0]
        stl_percentage_uw = calculate_stl_percentage_uw(stl_STL, stl_MIN, stl_team_MIN, stl_opponent_possessions)
        
        # blk_percentage_uw
        blk_BLK = BLK[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
        blk_MIN = MIN[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
        blk_team_MIN = team_MIN[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
        blk_opponent_FGA = opponent_FGA[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
        blk_opponent_FG3A = opponent_FG3A[(MIN * (opponent_FGA - opponent_FG3A)) > 0]
        blk_percentage_uw = calculate_blk_percentage_uw(blk_BLK, blk_MIN, blk_team_MIN, blk_opponent_FGA, blk_opponent_FG3A)
        
        # tov_percentage_uw
        tov_TOV = TOV[(TOV + 0.44 * PF) > 0]
        tov_FGA = FGA[(TOV + 0.44 * PF) > 0]
        tov_PF = PF[(TOV + 0.44 * PF) > 0]
        tov_percentage_uw = calculate_tov_percentage_uw(tov_TOV, tov_FGA, tov_PF)

        # Append the results (e.g., metrics) for this pair
        results.append([
            weighted_advanced_metric(ts_uw),
            weighted_advanced_metric(par_uw),
            weighted_advanced_metric(ftr_uw),
            weighted_advanced_metric(orb_percentage_uw),
            weighted_advanced_metric(drb_percentage_uw),
            weighted_advanced_metric(ast_percentage_uw),
            weighted_advanced_metric(stl_percentage_uw),
            weighted_advanced_metric(blk_percentage_uw),
            weighted_advanced_metric(tov_percentage_uw),

        ])
        # for i, num in enumerate(results):
        #     if np.any(np.isnan(num)):
        #         print(i)
        #         print("There is a NAN in")
    return (results)


# %%
# get teams dataset. ONLY NEED THE MOST RECENT ONE
def getTeamDataset(TEAM_ABBREVIATION, teamnumber):
    # get the dataset for all of the games in the current season
    # change this to be for only the ones with the abreviation in team 1
    finalized_data = []
    temp = filtered_allPlayerGamesEverWhoPlayed[
    (filtered_allPlayerGamesEverWhoPlayed["MATCHUP"].str.contains(TEAM_ABBREVIATION)) & 
    (filtered_allPlayerGamesEverWhoPlayed["GAME_DATE"] >= "2024-10-22")
    ]
    
    # Sort by GAME_DATE in descending order
    temp_sorted = temp.sort_values(by="GAME_DATE", ascending=False)
    
    # Group by Game_ID and Player_ID
    temp_grouped = temp_sorted.groupby(["Game_ID"])
    
    # Get the most recent group
    most_recent_group_key = list(temp_grouped.groups.keys())[0]
    most_recent_group = temp_grouped.get_group(most_recent_group_key)
    
    # print("Most Recent Group:")
    # print(most_recent_group[["WL", "MATCHUP", "GAME_DATE"]])
    
    if most_recent_group["MATCHUP"].iloc[0][0:4] == TEAM_ABBREVIATION:
        wl = most_recent_group["WL"].iloc[0]
    else:
        if most_recent_group["WL"].iloc[0] == "L":
            wl = "W"
        else:
            wl = "L"
            
    # get the stats of just our team
    most_recent_group_our_team = most_recent_group[most_recent_group["WL"] == wl]
    aggregatedData = getCareerStatsBeforeDateRows(np.array(most_recent_group_our_team["GAME_DATE"]), np.array(most_recent_group_our_team["Player_ID"]))
    
    cleaned_data = np.array([row for row in aggregatedData if isinstance(row, list) and not np.any(np.isnan(row))])

    # Convert to a proper NumPy array
    cleaned_data = np.array(cleaned_data, dtype=float)
    # print(cleaned_data)

    

    # If aggregatedData has data (non-empty), calculate the mean for this group
    if cleaned_data.size > 0:
        aggregated_array = np.nanmean(cleaned_data, axis=0)
        row_with_wl = np.append(aggregated_array, wl)  # Append win/loss
        row_with_wl = np.append(row_with_wl, most_recent_group["Game_ID"].iloc[0])  # Append game ID

        # Add the aggregated data into the list
        finalized_data.append(row_with_wl)

    else:
        print("THIS SHOULDNT HAPPEN")

    

    # Once all the data has been processed, convert the list to a DataFrame
    finalized_dataset = pd.DataFrame(finalized_data, columns=[
        "ts_uw", "3par_uw", "ftr_uw", "orb_percentage_uw", "drb_percentage_uw", 
        "ast_percentage_uw", "stl_percentage_uw", "blk_percentage_uw", "tov_percentage_uw", 
        "WL", "Game_ID"
    ])
    finalized_dataset = finalized_dataset.add_suffix(teamnumber)

    return finalized_dataset

    
# # %% testing

# getTeamDataset("ATL", 1)

# %% coding the models

# logistic regression 60.5% accuracy
@st.cache_data
def logisticReg(datain):
    # Assuming combinedshuffled_df is your DataFrame
    X = combinedshuffled_df.drop(columns=['WL1'])
    y = combinedshuffled_df['WL1']

    # Train the logistic regression model on the entire dataset
    model = LogisticRegression(C=3792.690190732246, max_iter=1000, random_state=42, solver='saga')
    model.fit(X, y)  # Train on all data (no split into training/test sets)

    # Predict on the new data (datain)
    predicted_class = model.predict(datain.drop(columns=['WL1']))

    # Predict the probability of each class (optional)
    predicted_probabilities = model.predict_proba(datain.drop(columns=['WL1']))

    return [predicted_class, predicted_probabilities]

# RandomForrest 62% accuracy
@st.cache_data
def randomForesttest(datain):
    # Assuming combinedshuffled_df is your DataFrame
    X = combinedshuffled_df.drop(columns=['WL1'])
    y = combinedshuffled_df['WL1']
    
    # Create a Random Forest model with the specified hyperparameters
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=4,
        bootstrap=True,
        random_state=42
    )
    
    # Train the model on the entire dataset
    model.fit(X, y)
    
    # Get predicted probabilities
    
    predicted_proba = model.predict_proba(datain.drop(columns=['WL1']))
    # Get predicted class
    predicted_class = model.predict(datain.drop(columns=['WL1']))
    return [predicted_class, predicted_proba]

@st.cache_data
def XGBoosttest(datain):
    X = combinedshuffled_df.drop(columns=['WL1'])
    y = combinedshuffled_df['WL1']
    
    # Create the XGBoost model with hyperparameters
    model = XGBClassifier(
        n_estimators=200,  # Number of trees
        learning_rate=0.05,  # Step size shrinkage
        max_depth=6,  # Maximum depth of each tree
        subsample=0.8,  # Fraction of samples used per tree
        colsample_bytree=0.8,  # Fraction of features used per tree
        reg_alpha=1.0,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42  # Ensures reproducibility
    )
    
    # Train the model on the entire dataset
    model.fit(X, y)
    
    # Get predicted probabilities
    predicted_proba = model.predict_proba(datain.drop(columns=['WL1']))
    # Get predicted class
    predicted_class = model.predict(datain.drop(columns=['WL1']))
    
    return [predicted_class, predicted_proba]
    # return [0, 1]





# %% ANDYS MODEL

def parse_tuple(val):
    return ast.literal_eval(val)

df = pd.read_csv('kills_list.csv', converters={'kills': parse_tuple})
df = df[~df['kills'].apply(lambda x: x==(None, None))]
print(df)
df['kills_diff'] = df.apply(lambda row: row['kills'][0]-row['kills'][1], axis=1)
df['WL_num'] = df['WL'].map({'W':True, 'L':False})

print(df)



df[['my_kills', 'opp__kills']] = pd.DataFrame(df['kills'].tolist(), index=df.index)
means = df.groupby('TEAM_ABBREVIATION')['my_kills'].mean()
df['kill_mean'] = df.groupby('TEAM_ABBREVIATION')['my_kills'].transform('mean')

every_game = leaguegamefinder.LeagueGameFinder().get_data_frames()[0]
every_game['GAME_ID'] = every_game['GAME_ID'].astype(np.int64)
df = pd.merge(df, every_game, on='GAME_ID')
df['opp'] = df['MATCHUP'].str[-3:]
df['opp_mean'] = df['opp'].map(means)
df['mean_diff'] = df['kill_mean'] - df['opp_mean']
# print(df[df['mean_diff'] != 0])
df['TS'] = df['PTS'] * .5 /(df['FGA'] + 0.474 * df['FTA'])
print(df.columns)


X = df[['kills_diff', 'FG3M', 'TOV', 'TS']]
y = df['WL_num']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Predict probabilities for the positive class
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1


from nba_api.stats.endpoints import teamyearbyyearstats
team_ids = df[['TEAM_ABBREVIATION_y', 'TEAM_ID']].drop_duplicates()
def Andypredict_game(team1, team2):

    team1_id = team_ids.loc[team_ids['TEAM_ABBREVIATION_y'] == team1].iloc[0,1]
    team1_stats = teamyearbyyearstats.TeamYearByYearStats(team1_id).get_data_frames()[0].tail(1)
    team1_stats['TS'] = team1_stats['PTS'] * .5 / (df['FGA'] + 0.474 * df['FTA'])
    team1_3pct =team1_stats['FG3M'].values.tolist()[0]
    team1_tov = team1_stats['TOV'].values.tolist()[0]
    team1_ts = team1_stats['TS'].values.tolist()[0]
    kill_diff = means[team1] - means[team2]
    pred = model.predict([[kill_diff, team1_3pct, team1_tov, team1_ts]])
    proba_pred = model.predict_proba([[kill_diff, team1_3pct, team1_tov, team1_ts]])[0]

    return [pred, proba_pred]


# %% code to reorganize the team data

@st.cache_data
def combineTeamsDataForModeling(team1, team2):
    # TEAM 1 IS THE ONE WE ARE PREDICTING WILL WIN OR NOT
    tempcombinedshuffled_df = pd.concat([team1, team2], axis=1)
    
    tempcombinedshuffled_df['WL1'] = tempcombinedshuffled_df['WL1'].map({'L': 0, 'W': 1})
    tempcombinedshuffled_df['WL2'] = tempcombinedshuffled_df['WL2'].map({'L': 0, 'W': 1})
    tempcombinedshuffled_df = tempcombinedshuffled_df.drop(columns=['Game_ID1', 'Game_ID2'])
    tempcombinedshuffled_df = tempcombinedshuffled_df.dropna()
    
    tempcombinedshuffled_df = tempcombinedshuffled_df.drop(columns=['WL2'])
    
    tempcombinedshuffled_df = tempcombinedshuffled_df.apply(pd.to_numeric, errors='coerce')
    print(tempcombinedshuffled_df.dtypes)
    print(tempcombinedshuffled_df)
    return tempcombinedshuffled_df


# %%
# Streamlit code

# put in the teams
col1, col2 = st.columns(2)
with col1:
    team1pick = st.selectbox(
        "Team1 pick",
        ("ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS", "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"),
        index=None,
        placeholder="Choose your team 1...",
    )

with col2:
    team2pick = st.selectbox(
        "Team2 pick",
        ("ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS", "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"),
        index=None,
        placeholder="Choose your team 2...",
    )




logRegScore = "Not in Yet"
RFScore = "Not in Yet"
XGScore = "Not in Yet"
aScore = "Not in Yet"

wlval1 = ""
wlval2 = ""
wlval3 = ""
wlval4 = ""

output1 = ""
output2 = ""
output3 = ""
output4 = ""

st.write("Press to Predict if Your Team Will Win:")
if st.button("Predict"):
    team1 = getTeamDataset(team1pick, "1")
    team2 = getTeamDataset(team2pick, "2")

    teamInput = combineTeamsDataForModeling(team1, team2)
    
    
   
    
    
    
    # logistic regression
    logRegScore = logisticReg(teamInput)
    if(logRegScore[0] == 1):
        wlval1 = "W"
    else:
        wlval1 = "L"
    output1 = f"Your team is predicted to: {wlval1}. Percentages: {logRegScore[1]}"
    
    # Random Forrest
    RFScore = randomForesttest(teamInput)
    if(RFScore[0] == 1):
        wlval2 = "W"
    else:
        wlval2 = "L"
    output2 = f"Your team is predicted to: {wlval2}. Percentages: {RFScore[1]}"
    
    # XGboost
    XGScore = XGBoosttest(teamInput)
    if(XGScore[0] == 1):
        wlval3 = "W"
    else:
        wlval3 = "L"
    output3 = f"Your team is predicted to: {wlval3}. Percentages: {XGScore[1]}"
    
    # andy
    aScore = Andypredict_game(team1pick, team2pick)
    if(aScore[0] == 1):
        wlval4 = "W"
    else:
        wlval4 = "L"
    output4 = f"Your team is predicted to: {wlval4}. Percentages: {aScore[1]}"

logReg, RF, XG, Andy = st.columns(4)
container1 = st.container(border=True)
container2 = st.container(border=True)
container3 = st.container(border=True)
container4 = st.container(border=True)
with logReg:
    container1.write("Logistic Regression")
    container1.write(output1)

with RF:
    container2.write("Random Forest")
    container2.write(output2)

with XG:
    container3.write("XG Boost")
    container3.write(output3)
    
with XG:
    container4.write("Andy's model")
    container4.write(output4)




# put who we think will win





# give the evaluation metrics grid
