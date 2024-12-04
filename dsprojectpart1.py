# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:16:32 2024

@author: bryan
"""

#%% import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import collections

#%% bring in dataset

allPlayerGamesEver = pd.read_csv('allPlayerGamesEver.csv')

# %% format the dataframe

allPlayerGamesEverWhoPlayed = allPlayerGamesEver[allPlayerGamesEver['MIN'] > 0]
allPlayerGamesEverWhoPlayed['GAME_DATE'] = pd.to_datetime(allPlayerGamesEverWhoPlayed['GAME_DATE'])
allPlayerGamesEverWhoPlayed = allPlayerGamesEverWhoPlayed.sort_values(by='GAME_DATE')
# THe first game to have all data is Game_ID                       29600005    GAME_DATE          1996-11-01 00:00:00

# so we will remove all previous games from the dataset
filtered_allPlayerGamesEverWhoPlayed = allPlayerGamesEverWhoPlayed[allPlayerGamesEverWhoPlayed['GAME_DATE'] >= "1996-11-01 00:00:00"]
filtered_allPlayerGamesEverWhoPlayed = filtered_allPlayerGamesEverWhoPlayed.drop("PLUS_MINUS", axis=1)
filtered_allPlayerGamesEverWhoPlayed = filtered_allPlayerGamesEverWhoPlayed.drop("VIDEO_AVAILABLE", axis=1)

# %% functions for calculating all advanced metrics

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

# %% function for getting weighted_advanced_metrics
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
        "There is nothing in metric_percentages. THIS SHOULDNT HAPPEN"
        return 0  # You can choose another default value if needed

    # Apply weights to the valid metrics
    weighted_metric = np.multiply(valid_metrics, valid_weights)

    # Compute weighted average
    return weighted_metric.sum() / np.sum(valid_weights)

# %% hold aggregated data for faster processing of team data

# predefine the opponent stat dataframe
grouped = filtered_allPlayerGamesEverWhoPlayed.groupby('Game_ID')

# Filter rows where WL == 'L' and aggregate
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


aggregatedL['opponent_defensive_rebounds'] = (
    aggregatedW['DREB']
)
aggregatedW['opponent_defensive_rebounds'] = (
    aggregatedL['DREB']
)


aggregatedL['opponent_offensive_rebounds'] = (
    aggregatedW['OREB']
)
aggregatedW['opponent_offensive_rebounds'] = (
    aggregatedL['OREB']
)


aggregatedL['opponent_FGA'] = (
    aggregatedW['FGA']
)
aggregatedW['opponent_FGA'] = (
    aggregatedL['FGA']
)


aggregatedL['opponent_FG3A'] = (
    aggregatedW['FG3A']
)
aggregatedW['opponent_FG3A'] = (
    aggregatedL['FG3A']
)

aggregatedL['WL'] = "L"
aggregatedW['WL'] = "W"


# %% function optomized for numpy
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



# %% testing
nptest1 = np.array(["2024-10-23", "2024-10-23", "2024-10-23", "2024-10-23"])
nptest2 = np.array([1630533, 1629003, 1627827, 1641730])

print(getCareerStatsBeforeDateRows(nptest1, nptest2))

# print(pd.to_datetime("2024-10-23"))

# temp = filtered_allPlayerGamesEverWhoPlayed[(filtered_allPlayerGamesEverWhoPlayed["MATCHUP"].str.contains("ATL")) & (filtered_allPlayerGamesEverWhoPlayed["GAME_DATE"] >= "2024-10-22")]
# print(len(temp.groupby(("Player_ID"))))
# print(temp[["GAME_DATE", "Player_ID", "MATCHUP"]].head(40))


# %% hold grouped data

grouped_filtered_APGEWP = filtered_allPlayerGamesEverWhoPlayed.groupby(["Game_ID", "WL"])


# %% initialize the finzlized dataset

finalized_dataset = pd.DataFrame(
  [],
  columns=[
    "ts_uw",
    "3par_uw",
    "ftr_uw",
    "orb_percentage_uw",
    "drb_percentage_uw",
    "ast_percentage_uw",
    "stl_percentage_uw",
    "blk_percentage_uw",
    "tov_percentage_uw",
    "WL",
    "Game_ID"
  ],
)



# %% optomized collect final

finalized_data = []  # List to accumulate the results
counter = 0  # Counter for tracking progress

# Iterate through grouped data
for (game_id, wl), group_data in grouped_filtered_APGEWP:
    # aggregatedData = []  # List to store valid career stats

    # Iterate through the rows of the group_data
    # print(len(np.array(group_data["GAME_DATE"])), len(np.array(group_data["Player_ID"])))
    aggregatedData = getCareerStatsBeforeDateRows(np.array(group_data["GAME_DATE"]), np.array(group_data["Player_ID"]))
    
    cleaned_data = np.array([row for row in aggregatedData if isinstance(row, list) and not np.any(np.isnan(row))])

    # Convert to a proper NumPy array
    cleaned_data = np.array(cleaned_data, dtype=float)
    # print(cleaned_data)



    # If aggregatedData has data (non-empty), calculate the mean for this group
    if cleaned_data.size > 0:
        aggregated_array = np.nanmean(cleaned_data, axis=0)
        row_with_wl = np.append(aggregated_array, wl)  # Append win/loss
        row_with_wl = np.append(row_with_wl, game_id)  # Append game ID

        # Add the aggregated data into the list
        finalized_data.append(row_with_wl)

    else:
        print("THIS SHOULDNT HAPPEN")

    # Optionally print the shape to track progress
    counter += 1
    if counter % 10 == 0:
        print(f"Game ID: {game_id}, WL: {wl}, Processed: {counter} groups")

# Once all the data has been processed, convert the list to a DataFrame
finalized_dataset = pd.DataFrame(finalized_data, columns=[
    "ts_uw", "3par_uw", "ftr_uw", "orb_percentage_uw", "drb_percentage_uw", 
    "ast_percentage_uw", "stl_percentage_uw", "blk_percentage_uw", "tov_percentage_uw", 
    "WL", "Game_ID"
])

# Optionally print the shape of the finalized dataset to check progress
print(f"Finalized Dataset Shape: {finalized_dataset.shape}")


# %% MOVE THE FINALIZED DATASET COMMAND HERE RIGHT AFTER IT FINISHES

# %% check to see dataset

finalized_dataset.head()
print(finalized_dataset.shape)

# %% save it

finalized_dataset.to_csv('finalized_dataset.csv', index=True)

