from src.utils import utils
import pandas as pd
import numpy as np
from src import config


def numerical_superiority(match_id: str) -> pd.DataFrame:
    """
    Quantification of different measures of numerical superiority of the defending team:
        - 2 measures in static areas:
            - numerical superiority in own half
            - numerical superiority in own final third
        - 2 measures of variable areas:
            - numerical superiority 15 meters in front of the last defender (rudimental measure for rest defense)
            - numerical superiority around the ball (1. 10m around the ball, 2. 20m around the ball)
    
    Args:
        match_id: DFL match ID

    Returns: dataframe with columns ['match_id', 'half', 'frame', 
                                     'defending_team_id', 
                                     'number_defenders_own_half', 'numerical_superiority_own_half', 
                                     'number_defenders_own_final_third', 'numerical_superiority_own_final_third',
                                     'number_defenders_15m_lastdefender', 'numerical_superiority_15m_lastdefender',
                                     'number_defenders_10m_fromball', 'numerical_superiority_10mfromball'
                                     'number_defenders_20m_fromball', 'numerical_superiority_20mfromball']
    """

    # load data of attackers and defenders
    positions_attackers_defenders = utils.get_positions_attackers_defenders_data([match_id])
    if config.get_reduce_metrics_to_every_fifth_frame():
        positions_attackers_defenders = positions_attackers_defenders[positions_attackers_defenders["frame"] % 5 == 0]
        positions_attackers_defenders.reset_index(inplace=True)
    if positions_attackers_defenders.empty:
        print(f"Warning: No position data, thus skipping metric numerical_superiority, for match {match_id}")
        return pd.DataFrame()

    teams = positions_attackers_defenders.attacking_team_id.unique()
    team1 = teams[0]
    team2 = teams[1]

    defenders_x = ['defender_0_val_x',
                   'defender_1_val_x',
                   'defender_2_val_x',
                   'defender_3_val_x',
                   'defender_4_val_x',
                   'defender_5_val_x',
                   'defender_6_val_x',
                   'defender_7_val_x',
                   'defender_8_val_x',
                   'defender_9_val_x',
                   'defender_10_val_x']
    attackers_x = ['attacker_0_val_x',
                   'attacker_1_val_x',
                   'attacker_2_val_x',
                   'attacker_3_val_x',
                   'attacker_4_val_x',
                   'attacker_5_val_x',
                   'attacker_6_val_x',
                   'attacker_7_val_x',
                   'attacker_8_val_x',
                   'attacker_9_val_x',
                   'attacker_10_val_x']

    # FIRST HALF
    # identififcation of playing direction
    positions_first_half = positions_attackers_defenders[positions_attackers_defenders['half'] == 1]

    defending_team1_first = positions_first_half[positions_first_half['defending_team_id'] == team1]
    defending_team2_first = positions_first_half[positions_first_half['defending_team_id'] == team2]

    mean_team1_first = (
                               defending_team1_first.defender_0_val_x.mean() + defending_team1_first.defender_1_val_x.mean() + defending_team1_first.defender_2_val_x.mean() + defending_team1_first.defender_3_val_x.mean() + defending_team1_first.defender_4_val_x.mean() + defending_team1_first.defender_5_val_x.mean() + defending_team1_first.defender_6_val_x.mean() + defending_team1_first.defender_7_val_x.mean() + defending_team1_first.defender_8_val_x.mean() + defending_team1_first.defender_9_val_x.mean() + defending_team1_first.defender_10_val_x.mean()) / 11
    mean_team2_first = (
                               defending_team2_first.defender_0_val_x.mean() + defending_team2_first.defender_1_val_x.mean() + defending_team2_first.defender_2_val_x.mean() + defending_team2_first.defender_3_val_x.mean() + defending_team2_first.defender_4_val_x.mean() + defending_team2_first.defender_5_val_x.mean() + defending_team2_first.defender_6_val_x.mean() + defending_team2_first.defender_7_val_x.mean() + defending_team2_first.defender_8_val_x.mean() + defending_team2_first.defender_9_val_x.mean() + defending_team2_first.defender_10_val_x.mean()) / 11

    team_first_left = np.where(mean_team1_first < mean_team2_first, team1, team2).item()
    team_first_right = np.where(mean_team1_first < mean_team2_first, team2, team1).item()

    # SECOND HALF
    # identififcation of playing direction
    positions_second_half = positions_attackers_defenders[positions_attackers_defenders['half'] == 2]

    defending_team1_second = positions_second_half[positions_second_half['defending_team_id'] == team1]
    defending_team2_second = positions_second_half[positions_second_half['defending_team_id'] == team2]

    mean_team1_second = (
                                defending_team1_second.defender_0_val_x.mean() + defending_team1_second.defender_1_val_x.mean() + defending_team1_second.defender_2_val_x.mean() + defending_team1_second.defender_3_val_x.mean() + defending_team1_second.defender_4_val_x.mean() + defending_team1_second.defender_5_val_x.mean() + defending_team1_second.defender_6_val_x.mean() + defending_team1_second.defender_7_val_x.mean() + defending_team1_second.defender_8_val_x.mean() + defending_team1_second.defender_9_val_x.mean() + defending_team1_second.defender_10_val_x.mean()) / 11
    mean_team2_second = (
                                defending_team2_second.defender_0_val_x.mean() + defending_team2_second.defender_1_val_x.mean() + defending_team2_second.defender_2_val_x.mean() + defending_team2_second.defender_3_val_x.mean() + defending_team2_second.defender_4_val_x.mean() + defending_team2_second.defender_5_val_x.mean() + defending_team2_second.defender_6_val_x.mean() + defending_team2_second.defender_7_val_x.mean() + defending_team2_second.defender_8_val_x.mean() + defending_team2_second.defender_9_val_x.mean() + defending_team2_second.defender_10_val_x.mean()) / 11

    team_second_left = np.where(mean_team1_second < mean_team2_second, team1, team2).item()
    team_second_right = np.where(mean_team1_second < mean_team2_second, team2, team1).item()
    # numerical superiority in own half
    positions_attackers_defenders['number_defenders_own_half'] = np.where(((positions_attackers_defenders.half == 1) & (
            positions_attackers_defenders.defending_team_id == team_first_left)) | ((
                                                                                            positions_attackers_defenders.half == 2) & (
                                                                                            positions_attackers_defenders.defending_team_id == team_second_left)),
                                                                          (positions_attackers_defenders[
                                                                               positions_attackers_defenders.columns[
                                                                                   positions_attackers_defenders.columns.isin(
                                                                                       defenders_x)]] <= 0).sum(axis=1),
                                                                          (positions_attackers_defenders[
                                                                               positions_attackers_defenders.columns[
                                                                                   positions_attackers_defenders.columns.isin(
                                                                                       defenders_x)]] >= 0).sum(axis=1))
    positions_attackers_defenders['number_attackers_own_half'] = np.where(((positions_attackers_defenders.half == 1) & (
            positions_attackers_defenders.defending_team_id == team_first_left)) | ((
                                                                                            positions_attackers_defenders.half == 2) & (
                                                                                            positions_attackers_defenders.defending_team_id == team_second_left)),
                                                                          (positions_attackers_defenders[
                                                                               positions_attackers_defenders.columns[
                                                                                   positions_attackers_defenders.columns.isin(
                                                                                       attackers_x)]] <= 0).sum(axis=1),
                                                                          (positions_attackers_defenders[
                                                                               positions_attackers_defenders.columns[
                                                                                   positions_attackers_defenders.columns.isin(
                                                                                       attackers_x)]] >= 0).sum(axis=1))
    positions_attackers_defenders['numerical_superiority_own_half'] = positions_attackers_defenders[
                                                                          'number_defenders_own_half'] - \
                                                                      positions_attackers_defenders[
                                                                          'number_attackers_own_half']
    # numerical superiority final third
    positions_attackers_defenders['number_defenders_own_final_third'] = np.where(((
                                                                                          positions_attackers_defenders.half == 1) & (
                                                                                          positions_attackers_defenders.defending_team_id == team_first_left)) | (
                                                                                         (
                                                                                                 positions_attackers_defenders.half == 2) & (
                                                                                                 positions_attackers_defenders.defending_team_id == team_second_left)),
                                                                                 (positions_attackers_defenders[
                                                                                      positions_attackers_defenders.columns[
                                                                                          positions_attackers_defenders.columns.isin(
                                                                                              defenders_x)]] <= (
                                                                                      -17.5)).sum(axis=1),
                                                                                 (positions_attackers_defenders[
                                                                                      positions_attackers_defenders.columns[
                                                                                          positions_attackers_defenders.columns.isin(
                                                                                              defenders_x)]] >= (
                                                                                      17.5)).sum(axis=1))
    positions_attackers_defenders['number_attackers_own_final_third'] = np.where(((
                                                                                          positions_attackers_defenders.half == 1) & (
                                                                                          positions_attackers_defenders.defending_team_id == team_first_left)) | (
                                                                                         (
                                                                                                 positions_attackers_defenders.half == 2) & (
                                                                                                 positions_attackers_defenders.defending_team_id == team_second_left)),
                                                                                 (positions_attackers_defenders[
                                                                                      positions_attackers_defenders.columns[
                                                                                          positions_attackers_defenders.columns.isin(
                                                                                              attackers_x)]] <= (
                                                                                      -17.5)).sum(axis=1),
                                                                                 (positions_attackers_defenders[
                                                                                      positions_attackers_defenders.columns[
                                                                                          positions_attackers_defenders.columns.isin(
                                                                                              attackers_x)]] >= (
                                                                                      17.5)).sum(axis=1))
    positions_attackers_defenders['numerical_superiority_own_final_third'] = positions_attackers_defenders[
                                                                                 'number_defenders_own_final_third'] - \
                                                                             positions_attackers_defenders[
                                                                                 'number_attackers_own_final_third']
    # numerical superiority 15 meters in front of the last defender (rudimental measure for rest defense)
    # identification of x-position of the last defender
    # second smallest distance to the defendeings goal line (because of goalkeeper)
    x_position_last_defender = np.array(positions_attackers_defenders[positions_attackers_defenders.columns[
        positions_attackers_defenders.columns.isin(defenders_x)]])
    x_position_last_defender.sort(axis=1)
    hight_from_last_def = 15
    positions_attackers_defenders['x_position_15m_lastdefender_left'] = x_position_last_defender[:,
                                                                        1] + hight_from_last_def
    positions_attackers_defenders['x_position_15m_lastdefender_right'] = x_position_last_defender[:,
                                                                         9] - hight_from_last_def

    positions_attackers_defenders['defender_0_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_0_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_1_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_1_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_2_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_2_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_3_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_3_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_4_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_4_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_5_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_5_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_6_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_6_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_7_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_7_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_8_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_8_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_9_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'defender_9_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['defender_10_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                              'defender_10_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_left']) < 0

    positions_attackers_defenders['defender_0_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_0_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_1_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_1_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_2_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_2_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_3_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_3_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_4_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_4_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_5_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_5_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_6_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_6_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_7_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_7_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_8_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_8_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_9_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'defender_9_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['defender_10_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                               'defender_10_val_x'] -
                                                                           positions_attackers_defenders[
                                                                               'x_position_15m_lastdefender_right']) > 0

    positions_attackers_defenders['attacker_0_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_0_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_1_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_1_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_2_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_2_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_3_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_3_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_4_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_4_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_5_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_5_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_6_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_6_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_7_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_7_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_8_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_8_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_9_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                             'attacker_9_val_x'] -
                                                                         positions_attackers_defenders[
                                                                             'x_position_15m_lastdefender_left']) < 0
    positions_attackers_defenders['attacker_10_15m_lastdefender_left'] = (positions_attackers_defenders[
                                                                              'attacker_10_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_left']) < 0

    positions_attackers_defenders['attacker_0_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_0_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_1_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_1_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_2_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_2_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_3_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_3_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_4_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_4_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_5_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_5_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_6_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_6_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_7_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_7_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_8_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_8_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_9_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                              'attacker_9_val_x'] -
                                                                          positions_attackers_defenders[
                                                                              'x_position_15m_lastdefender_right']) > 0
    positions_attackers_defenders['attacker_10_15m_lastdefender_right'] = (positions_attackers_defenders[
                                                                               'attacker_10_val_x'] -
                                                                           positions_attackers_defenders[
                                                                               'x_position_15m_lastdefender_right']) > 0

    defenders_15m_lastdefender_left = ['defender_0_15m_lastdefender_left',
                                       'defender_1_15m_lastdefender_left',
                                       'defender_2_15m_lastdefender_left',
                                       'defender_3_15m_lastdefender_left',
                                       'defender_4_15m_lastdefender_left',
                                       'defender_5_15m_lastdefender_left',
                                       'defender_6_15m_lastdefender_left',
                                       'defender_7_15m_lastdefender_left',
                                       'defender_8_15m_lastdefender_left',
                                       'defender_9_15m_lastdefender_left',
                                       'defender_10_15m_lastdefender_left']
    defenders_15m_lastdefender_right = ['defender_0_15m_lastdefender_right',
                                        'defender_1_15m_lastdefender_right',
                                        'defender_2_15m_lastdefender_right',
                                        'defender_3_15m_lastdefender_right',
                                        'defender_4_15m_lastdefender_right',
                                        'defender_5_15m_lastdefender_right',
                                        'defender_6_15m_lastdefender_right',
                                        'defender_7_15m_lastdefender_right',
                                        'defender_8_15m_lastdefender_right',
                                        'defender_9_15m_lastdefender_right',
                                        'defender_10_15m_lastdefender_right']
    attackers_15m_lastdefender_left = ['attacker_0_15m_lastdefender_left',
                                       'attacker_1_15m_lastdefender_left',
                                       'attacker_2_15m_lastdefender_left',
                                       'attacker_3_15m_lastdefender_left',
                                       'attacker_4_15m_lastdefender_left',
                                       'attacker_5_15m_lastdefender_left',
                                       'attacker_6_15m_lastdefender_left',
                                       'attacker_7_15m_lastdefender_left',
                                       'attacker_8_15m_lastdefender_left',
                                       'attacker_9_15m_lastdefender_left',
                                       'attacker_10_15m_lastdefender_left']
    attackers_15m_lastdefender_right = ['attacker_0_15m_lastdefender_right',
                                        'attacker_1_15m_lastdefender_right',
                                        'attacker_2_15m_lastdefender_right',
                                        'attacker_3_15m_lastdefender_right',
                                        'attacker_4_15m_lastdefender_right',
                                        'attacker_5_15m_lastdefender_right',
                                        'attacker_6_15m_lastdefender_right',
                                        'attacker_7_15m_lastdefender_right',
                                        'attacker_8_15m_lastdefender_right',
                                        'attacker_9_15m_lastdefender_right',
                                        'attacker_10_15m_lastdefender_right']

    positions_attackers_defenders['number_defenders_15m_lastdefender'] = np.where(((
                                                                                           positions_attackers_defenders.half == 1) & (
                                                                                           positions_attackers_defenders.defending_team_id == team_first_left)) | (
                                                                                          (
                                                                                                  positions_attackers_defenders.half == 2) & (
                                                                                                  positions_attackers_defenders.defending_team_id == team_second_left)),
                                                                                  positions_attackers_defenders[
                                                                                      positions_attackers_defenders.columns[
                                                                                          positions_attackers_defenders.columns.isin(
                                                                                              defenders_15m_lastdefender_left)]].sum(
                                                                                      axis=1),
                                                                                  positions_attackers_defenders[
                                                                                      positions_attackers_defenders.columns[
                                                                                          positions_attackers_defenders.columns.isin(
                                                                                              defenders_15m_lastdefender_right)]].sum(
                                                                                      axis=1))
    positions_attackers_defenders['number_attackers_15m_lastdefender'] = np.where(((
                                                                                           positions_attackers_defenders.half == 1) & (
                                                                                           positions_attackers_defenders.defending_team_id == team_first_left)) | (
                                                                                          (
                                                                                                  positions_attackers_defenders.half == 2) & (
                                                                                                  positions_attackers_defenders.defending_team_id == team_second_left)),
                                                                                  positions_attackers_defenders[
                                                                                      positions_attackers_defenders.columns[
                                                                                          positions_attackers_defenders.columns.isin(
                                                                                              attackers_15m_lastdefender_left)]].sum(
                                                                                      axis=1),
                                                                                  positions_attackers_defenders[
                                                                                      positions_attackers_defenders.columns[
                                                                                          positions_attackers_defenders.columns.isin(
                                                                                              attackers_15m_lastdefender_right)]].sum(
                                                                                      axis=1))

    positions_attackers_defenders['numerical_superiority_15m_lastdefender'] = positions_attackers_defenders[
                                                                                  'number_defenders_15m_lastdefender'] - \
                                                                              positions_attackers_defenders[
                                                                                  'number_attackers_15m_lastdefender']

    # numerical superiority in 20m around the ball
    positions_attackers_defenders['defender_0_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_0_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_0_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_1_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_1_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_1_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_2_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_2_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_2_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_3_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_3_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_3_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_4_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_4_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_4_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_5_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_5_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_5_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_6_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_6_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_6_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_7_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_7_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_7_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_8_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_8_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_8_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_9_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_9_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['defender_9_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['defender_10_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['defender_10_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + ((
                                                                                                                         positions_attackers_defenders[
                                                                                                                             'defender_10_val_y'] -
                                                                                                                         positions_attackers_defenders[
                                                                                                                             'ball_y']) ** 2))

    positions_attackers_defenders['attacker_0_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_0_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_0_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_1_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_1_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_1_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_2_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_2_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_2_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_3_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_3_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_3_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_4_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_4_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_4_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_5_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_5_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_5_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_6_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_6_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_6_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_7_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_7_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_7_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_8_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_8_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_8_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_9_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_9_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + (
                (positions_attackers_defenders['attacker_9_val_y'] - positions_attackers_defenders['ball_y']) ** 2))
    positions_attackers_defenders['attacker_10_dis_to_ball'] = np.sqrt(
        ((positions_attackers_defenders['attacker_10_val_x'] - positions_attackers_defenders['ball_x']) ** 2) + ((
                                                                                                                         positions_attackers_defenders[
                                                                                                                             'attacker_10_val_y'] -
                                                                                                                         positions_attackers_defenders[
                                                                                                                             'ball_y']) ** 2))
    defenders_dis = ['defender_0_dis_to_ball',
                     'defender_1_dis_to_ball',
                     'defender_2_dis_to_ball',
                     'defender_3_dis_to_ball',
                     'defender_4_dis_to_ball',
                     'defender_5_dis_to_ball',
                     'defender_6_dis_to_ball',
                     'defender_7_dis_to_ball',
                     'defender_8_dis_to_ball',
                     'defender_9_dis_to_ball',
                     'defender_10_dis_to_ball']
    attackers_dis = ['attacker_0_dis_to_ball',
                     'attacker_1_dis_to_ball',
                     'attacker_2_dis_to_ball',
                     'attacker_3_dis_to_ball',
                     'attacker_4_dis_to_ball',
                     'attacker_5_dis_to_ball',
                     'attacker_6_dis_to_ball',
                     'attacker_7_dis_to_ball',
                     'attacker_8_dis_to_ball',
                     'attacker_9_dis_to_ball',
                     'attacker_10_dis_to_ball']
    positions_attackers_defenders['number_defenders_20m_fromball'] = (positions_attackers_defenders[
                                                                          positions_attackers_defenders.columns[
                                                                              positions_attackers_defenders.columns.isin(
                                                                                  defenders_dis)]] < 20).sum(axis=1)
    positions_attackers_defenders['number_attackers_20m_fromball'] = (positions_attackers_defenders[
                                                                          positions_attackers_defenders.columns[
                                                                              positions_attackers_defenders.columns.isin(
                                                                                  attackers_dis)]] < 20).sum(axis=1)
    positions_attackers_defenders['numerical_superiority_20m_fromball'] = positions_attackers_defenders[
                                                                              'number_defenders_20m_fromball'] - \
                                                                          positions_attackers_defenders[
                                                                              'number_attackers_20m_fromball']
    positions_attackers_defenders['number_defenders_10m_fromball'] = (positions_attackers_defenders[
                                                                          positions_attackers_defenders.columns[
                                                                              positions_attackers_defenders.columns.isin(
                                                                                  defenders_dis)]] < 10).sum(axis=1)
    positions_attackers_defenders['number_attackers_10m_fromball'] = (positions_attackers_defenders[
                                                                          positions_attackers_defenders.columns[
                                                                              positions_attackers_defenders.columns.isin(
                                                                                  attackers_dis)]] < 10).sum(axis=1)
    positions_attackers_defenders['numerical_superiority_10m_fromball'] = positions_attackers_defenders[
                                                                              'number_defenders_10m_fromball'] - \
                                                                          positions_attackers_defenders[
                                                                              'number_attackers_10m_fromball']

    final_columns = ['match_id', 'half', 'frame',
                     'defending_team_id',
                     'number_defenders_own_half', 'numerical_superiority_own_half',
                     'number_defenders_own_final_third', 'numerical_superiority_own_final_third',
                     'number_defenders_15m_lastdefender', 'numerical_superiority_15m_lastdefender',
                     'number_defenders_10m_fromball', 'numerical_superiority_10m_fromball',
                     'number_defenders_20m_fromball', 'numerical_superiority_20m_fromball']
    df_final = positions_attackers_defenders[final_columns].rename(
        columns={"defending_team_id": "team_id"})
    return df_final
