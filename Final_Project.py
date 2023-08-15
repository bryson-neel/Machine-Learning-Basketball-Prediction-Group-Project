import pandas as pd
import csv
from csv import reader
from math import sqrt
from math import exp
import numpy as np
from numpy import genfromtxt
from scipy.stats import norm
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


player_data = np.array(pd.read_csv("players18-19.csv"))
team_data = np.array(pd.read_csv("teams18-19.csv"))
match_data = np.array(pd.read_csv("matches18-19.csv", header = None))


# find the average number of wins
match_teams = []
match_team_wins = []
average_wins = 0
zero_win_teams = []

# training and testing data (match dataset) is split in half
for i in range(int(len(match_data) / 2)):
    if not (match_data[i][1] in match_teams):
        match_teams.append(match_data[i][1])
        match_team_wins.append(1)
    else:
        match_team_wins[match_teams.index(match_data[i][1])] += 1
for i in range(len(match_team_wins)):
    average_wins += match_team_wins[i]

# see if any teams got 0 wins
for i in range(int(len(match_data) / 2)):
    if not (match_data[i][3] in match_teams):
        match_teams.append(match_data[i][3])
        match_team_wins.append(0)
        zero_win_teams.append(match_data[i][3])

average_wins /= len(match_team_wins)

# create lists of teams and players in those teams
teams = []
players_in_teams = []
for i in range(len(team_data)):
    if not team_data[i][0] in teams:
        teams.append(team_data[i][0])
        players_in_teams.append([])
num_teams = len(teams)
teams.sort()

for i in range(len(player_data)):
    if player_data[i][1] in teams:
        players_in_teams[teams.index(player_data[i][1])].append(player_data[i][0])


# (found earlier)
team_feature_maxes = [39.0, 35.0, 123.4, 119.2, 0.9744, 59.0, 59.3, 25.1, 24.7, 38.7, 37.1,
    48.1, 54.0, 61.4, 61.2, 42.4, 41.8, 79.1, 11.2]
team_feature_mins = [26.0, 3.0, 83.7, 85.6, 0.0346, 40.0, 42.5, 13.5, 13.3, 15.9, 21.7,
    21.9, 21.8, 37.7, 40.7, 27.9, 27.9, 60.7, -23.4]


# (found earlier)
player_feature_maxes = [39.0, 96.7, 391.0, 48.9, 150.0, 150.0, 232.3, 231.3, 100.0, 100.0, 239.0, 293.0,
    1.0, 333.0, 678.0, 1.0, 140.0, 380.0, 1.0, 130.6, 29.9, 1000.0, 3.0, 6.77463, 334.921, 160.0, 125.989,
    135.087, 5.34532, 290.363, 34.0071, 34.2667, 57.2375, 154.6, 39.3226, 118.545, 36.055, 4.3226, 9.303,
    12.8438, 9.9394, 3.4444, 3.4194, 30.0909]
player_feature_mins = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.63477, -35.9144, 0.0, 31.7614, 37.0042, -0.0962039, 0.0484208,
    -56.2742, -40.1125, -31.1604, -125.154, 0.0, -95.7764, -42.3436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# data normalization (team features)
for i in range(0, len(team_data)):
    for j in range(1, len(team_data[i])):
        team_data[i][j] = (team_data[i][j] - team_feature_mins[j - 1]) / (team_feature_maxes[j - 1] - team_feature_mins[j - 1])



# data normalization (player features)
for i in range(0, len(player_data)):
    for j in range(2, len(player_data[i])):
        player_data[i][j] = (player_data[i][j] - player_feature_mins[j - 2]) / (player_feature_maxes[j - 2] - player_feature_mins[j - 2])


# initialize team weights to uniform values
team_weights = []
for i in range(1, len(team_data[1])):
    team_weights.append((1 / len(team_data[1])))

# find means of each team feature
team_feature_counts = []
for i in range(0, len(team_data)):
    team_feature_counts.append([])
    for j in range(1, len(team_data[i])):
        team_feature_counts[i].append(team_data[i][j])

team_feature_counts = np.array((team_feature_counts), dtype = np.float32).T

team_feature_means = []
for i in range(len(team_feature_counts)):
    team_feature_means.append(sum(team_feature_counts[i]) / len(team_feature_counts[i]))


# initialize player weights to uniform values
player_weights = []
for i in range(2, len(player_data[1])):
    player_weights.append((1 / len(player_data[1])))

# find means of each player feature
player_feature_counts = []
for i in range(0, len(player_data)):
    player_feature_counts.append([])
    for j in range(2, len(player_data[i])):
        player_feature_counts[i].append(player_data[i][j])

player_feature_counts = np.array((player_feature_counts), dtype = np.float32).T

player_feature_means = []
for i in range(len(player_feature_counts)):
    player_feature_means.append(sum(player_feature_counts[i]) / len(player_feature_counts[i]))


# feature weighting training (team features)
for i in range(0, len(team_data)):
    if team_data[i][0] in match_teams: # else: this team wasn't in the match data, so don't do anything
        win_difference = match_team_wins[match_teams.index(team_data[i][0])] - average_wins
        for j in range(1, len(team_weights) + 1):
            feature_difference = (team_data[i][j] - team_feature_means[j - 1])
            team_weights[j - 1] += (win_difference / average_wins) * (feature_difference / team_feature_means[j - 1])


# feature weighting training (player features)
for i in range(0, len(player_data)):
    if player_data[i][1] in match_teams: # else: this player's team wasn't in the match data, so don't do anything
        team_ind = -1
        for i in range(len(teams)):
            if teams[i] == player_data[i][1]:
                team_ind = i
                i = len(teams)
        if team_ind >= 0:
            win_difference = (match_team_wins[match_teams.index(player_data[i][1])] - average_wins) / len(players_in_teams[team_ind])
            
            # only use first 26 features of players dataset because the others give nan values for the means
            for j in range(2, (len(player_weights) - 18) + 2):
                feature_difference = (player_data[i][j] - player_feature_means[j - 2])
                player_weights[j - 2] += (win_difference / average_wins) * (feature_difference / player_feature_means[j - 2])


# feature weighting testing (team features and player features)
num_correct = 0
num_total = 0
missing = 0

# training and testing data (match dataset) is split in half
for i in range(int(len(match_data) / 2), len(match_data)):
    team1_val = 0
    team2_val = 0
    ind1 = -1
    ind2 = -1
    for k in range(0, len(team_data)):
        if team_data[k][0] == match_data[i][1]:
            ind1 = k
            if (ind1 >= 0) and (ind2 >= 0):
                k = len(team_data)
        elif team_data[k][0] == match_data[i][3]:
            ind2 = k
            if (ind1 >= 0) and (ind2 >= 0):
                k = len(team_data)
    if (ind1 >= 0) and (ind2 >= 0):
        num_total += 1
        for j in range(1, len(team_data[0])):
            team1_val += (team_data[ind1][j] * team_weights[j - 1])
            team2_val += (team_data[ind2][j] * team_weights[j - 1])
    
    if match_data[i][1] in teams:
        team_ind = teams.index(match_data[i][1])
        player_ind = -1
        for k in range(len(player_data)):
            if player_data[k][1] == match_data[i][1]:
                player_ind = k
        if player_ind >= 0 and team_ind >= 0:
            for j in range(2, (len(player_weights) - 18) + 2):
                team1_val += (player_data[player_ind][j] * player_weights[j - 2]) / len(players_in_teams[team_ind])
    
    if match_data[i][3] in teams:
        team_ind = teams.index(match_data[i][3])
        player_ind = -1
        for k in range(len(player_data)):
            if player_data[k][1] == match_data[i][1]:
                player_ind = k
        if player_ind >= 0 and team_ind >= 0:
            # only use first 26 features of players dataset because the others give nan values for the means
            for j in range(2, (len(player_weights) - 18) + 2):
                team2_val += (player_data[player_ind][j] * player_weights[j - 2]) / len(players_in_teams[team_ind])
    
    if (team1_val > team2_val) and (match_data[i][2] > match_data[i][4]):
        num_correct += 1
    elif (team1_val < team2_val) and (match_data[i][2] < match_data[i][4]):
        num_correct += 1
print('Accuracy: ' + str(num_correct / num_total))
