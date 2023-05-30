from src.utils import utils
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from src import config


def distances_between_formation_lines(match_id: str) -> pd.DataFrame:
    """
    Quantification of distances between formation lines:
        - distance between defense and midfielder line: 'distance_def_mid'
        - distance between midfielder and attacking line: 'distance_mid_att'
        - distance between defense and attacking line: 'distance_def_att'
    
    Args:
        match_id: DFL match ID

    Returns: dataframe with columns ['match_id', 'half', 'frame', 
                                     'defending_team_id', 'distance_def_mid', 'distance_mid_att', 'distance_def_att']
    """

    # load data of attackers and defenders
    positions_attackers_defenders = utils.get_positions_attackers_defenders_data([match_id])
    if config.get_reduce_metrics_to_every_fifth_frame():
        positions_attackers_defenders = positions_attackers_defenders[positions_attackers_defenders["frame"] % 5 == 0]
        positions_attackers_defenders.reset_index(inplace=True)
    if positions_attackers_defenders.empty:
        print(f"Warning: No position data, thus skipping metric formation_lines, for match {match_id}")
        return pd.DataFrame()
    if positions_attackers_defenders.isna().any().any():
        print(f"Warning: NANs in position data (probably because of red cards)")
        print(f"Thus skipping metric formation_lines, for match {match_id}")
        return pd.DataFrame()

    teams = positions_attackers_defenders.attacking_team_id.unique()
    team1 = teams[0]
    team2 = teams[1]
    # FIRST HALF
    # formation identfification for first half
    positions_first_half = positions_attackers_defenders[positions_attackers_defenders['half'] == 1].copy()

    defending_team1_first = positions_first_half[positions_first_half['defending_team_id'] == team1].copy()
    defending_team2_first = positions_first_half[positions_first_half['defending_team_id'] == team2].copy()

    # Mirroring of data (both teams play from left to right)
    mean_team1_first = (
                               defending_team1_first.defender_0_val_x.mean() + defending_team1_first.defender_1_val_x.mean() + defending_team1_first.defender_2_val_x.mean() + defending_team1_first.defender_3_val_x.mean() + defending_team1_first.defender_4_val_x.mean() + defending_team1_first.defender_5_val_x.mean() + defending_team1_first.defender_6_val_x.mean() + defending_team1_first.defender_7_val_x.mean() + defending_team1_first.defender_8_val_x.mean() + defending_team1_first.defender_9_val_x.mean() + defending_team1_first.defender_10_val_x.mean()) / 11
    mean_team2_first = (
                               defending_team2_first.defender_0_val_x.mean() + defending_team2_first.defender_1_val_x.mean() + defending_team2_first.defender_2_val_x.mean() + defending_team2_first.defender_3_val_x.mean() + defending_team2_first.defender_4_val_x.mean() + defending_team2_first.defender_5_val_x.mean() + defending_team2_first.defender_6_val_x.mean() + defending_team2_first.defender_7_val_x.mean() + defending_team2_first.defender_8_val_x.mean() + defending_team2_first.defender_9_val_x.mean() + defending_team2_first.defender_10_val_x.mean()) / 11

    defending_team1_first['defender_0_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_0_val_x.copy() * (-1),
                                                         defending_team1_first.defender_0_val_x.copy())
    defending_team1_first['defender_1_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_1_val_x.copy() * (-1),
                                                         defending_team1_first.defender_1_val_x.copy())
    defending_team1_first['defender_2_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_2_val_x.copy() * (-1),
                                                         defending_team1_first.defender_2_val_x.copy())
    defending_team1_first['defender_3_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_3_val_x.copy() * (-1),
                                                         defending_team1_first.defender_3_val_x.copy())
    defending_team1_first['defender_4_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_4_val_x.copy() * (-1),
                                                         defending_team1_first.defender_4_val_x.copy())
    defending_team1_first['defender_5_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_5_val_x.copy() * (-1),
                                                         defending_team1_first.defender_5_val_x.copy())
    defending_team1_first['defender_6_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_6_val_x.copy() * (-1),
                                                         defending_team1_first.defender_6_val_x.copy())
    defending_team1_first['defender_7_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_7_val_x.copy() * (-1),
                                                         defending_team1_first.defender_7_val_x.copy())
    defending_team1_first['defender_8_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_8_val_x.copy() * (-1),
                                                         defending_team1_first.defender_8_val_x.copy())
    defending_team1_first['defender_9_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                         defending_team1_first.defender_9_val_x.copy() * (-1),
                                                         defending_team1_first.defender_9_val_x.copy())
    defending_team1_first['defender_10_val_x'] = np.where(mean_team1_first > mean_team2_first,
                                                          defending_team1_first.defender_10_val_x.copy() * (-1),
                                                          defending_team1_first.defender_10_val_x.copy())

    defending_team2_first['defender_0_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_0_val_x.copy() * (-1),
                                                         defending_team2_first.defender_0_val_x.copy())
    defending_team2_first['defender_1_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_1_val_x.copy() * (-1),
                                                         defending_team2_first.defender_1_val_x.copy())
    defending_team2_first['defender_2_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_2_val_x.copy() * (-1),
                                                         defending_team2_first.defender_2_val_x.copy())
    defending_team2_first['defender_3_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_3_val_x.copy() * (-1),
                                                         defending_team2_first.defender_3_val_x.copy())
    defending_team2_first['defender_4_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_4_val_x.copy() * (-1),
                                                         defending_team2_first.defender_4_val_x.copy())
    defending_team2_first['defender_5_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_5_val_x.copy() * (-1),
                                                         defending_team2_first.defender_5_val_x.copy())
    defending_team2_first['defender_6_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_6_val_x.copy() * (-1),
                                                         defending_team2_first.defender_6_val_x.copy())
    defending_team2_first['defender_7_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_7_val_x.copy() * (-1),
                                                         defending_team2_first.defender_7_val_x.copy())
    defending_team2_first['defender_8_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_8_val_x.copy() * (-1),
                                                         defending_team2_first.defender_8_val_x.copy())
    defending_team2_first['defender_9_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                         defending_team2_first.defender_9_val_x.copy() * (-1),
                                                         defending_team2_first.defender_9_val_x.copy())
    defending_team2_first['defender_10_val_x'] = np.where(mean_team1_first < mean_team2_first,
                                                          defending_team2_first.defender_10_val_x.copy() * (-1),
                                                          defending_team2_first.defender_10_val_x.copy())

    # Team1 is defending
    def_team1_first = pd.DataFrame()
    def_team1_first['player_id'] = [defending_team1_first.defender_0_id.value_counts().index[0],
                                    defending_team1_first.defender_1_id.value_counts().index[0],
                                    defending_team1_first.defender_2_id.value_counts().index[0],
                                    defending_team1_first.defender_3_id.value_counts().index[0],
                                    defending_team1_first.defender_4_id.value_counts().index[0],
                                    defending_team1_first.defender_5_id.value_counts().index[0],
                                    defending_team1_first.defender_6_id.value_counts().index[0],
                                    defending_team1_first.defender_7_id.value_counts().index[0],
                                    defending_team1_first.defender_8_id.value_counts().index[0],
                                    defending_team1_first.defender_9_id.value_counts().index[0],
                                    defending_team1_first.defender_10_id.value_counts().index[0]]
    def_team1_first['player_number'] = ['defender_0_val_x',
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
    def_team1_first['X'] = [defending_team1_first[defending_team1_first['defender_0_id'] ==
                                                  defending_team1_first.defender_0_id.value_counts().index[
                                                      0]].defender_0_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_1_id'] ==
                                                  defending_team1_first.defender_1_id.value_counts().index[
                                                      0]].defender_1_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_2_id'] ==
                                                  defending_team1_first.defender_2_id.value_counts().index[
                                                      0]].defender_2_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_3_id'] ==
                                                  defending_team1_first.defender_3_id.value_counts().index[
                                                      0]].defender_3_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_4_id'] ==
                                                  defending_team1_first.defender_4_id.value_counts().index[
                                                      0]].defender_4_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_5_id'] ==
                                                  defending_team1_first.defender_5_id.value_counts().index[
                                                      0]].defender_5_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_6_id'] ==
                                                  defending_team1_first.defender_6_id.value_counts().index[
                                                      0]].defender_6_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_7_id'] ==
                                                  defending_team1_first.defender_7_id.value_counts().index[
                                                      0]].defender_7_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_8_id'] ==
                                                  defending_team1_first.defender_8_id.value_counts().index[
                                                      0]].defender_8_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_9_id'] ==
                                                  defending_team1_first.defender_9_id.value_counts().index[
                                                      0]].defender_9_val_x.mean(),
                            defending_team1_first[defending_team1_first['defender_10_id'] ==
                                                  defending_team1_first.defender_10_id.value_counts().index[
                                                      0]].defender_10_val_x.mean()]

    # Team2 is defending
    def_team2_first = pd.DataFrame()
    def_team2_first['player_id'] = [defending_team2_first.defender_0_id.value_counts().index[0],
                                    defending_team2_first.defender_1_id.value_counts().index[0],
                                    defending_team2_first.defender_2_id.value_counts().index[0],
                                    defending_team2_first.defender_3_id.value_counts().index[0],
                                    defending_team2_first.defender_4_id.value_counts().index[0],
                                    defending_team2_first.defender_5_id.value_counts().index[0],
                                    defending_team2_first.defender_6_id.value_counts().index[0],
                                    defending_team2_first.defender_7_id.value_counts().index[0],
                                    defending_team2_first.defender_8_id.value_counts().index[0],
                                    defending_team2_first.defender_9_id.value_counts().index[0],
                                    defending_team2_first.defender_10_id.value_counts().index[0]]
    def_team2_first['player_number'] = ['defender_0_val_x',
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
    def_team2_first['X'] = [defending_team2_first[defending_team2_first['defender_0_id'] ==
                                                  defending_team2_first.defender_0_id.value_counts().index[
                                                      0]].defender_0_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_1_id'] ==
                                                  defending_team2_first.defender_1_id.value_counts().index[
                                                      0]].defender_1_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_2_id'] ==
                                                  defending_team2_first.defender_2_id.value_counts().index[
                                                      0]].defender_2_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_3_id'] ==
                                                  defending_team2_first.defender_3_id.value_counts().index[
                                                      0]].defender_3_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_4_id'] ==
                                                  defending_team2_first.defender_4_id.value_counts().index[
                                                      0]].defender_4_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_5_id'] ==
                                                  defending_team2_first.defender_5_id.value_counts().index[
                                                      0]].defender_5_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_6_id'] ==
                                                  defending_team2_first.defender_6_id.value_counts().index[
                                                      0]].defender_6_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_7_id'] ==
                                                  defending_team2_first.defender_7_id.value_counts().index[
                                                      0]].defender_7_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_8_id'] ==
                                                  defending_team2_first.defender_8_id.value_counts().index[
                                                      0]].defender_8_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_9_id'] ==
                                                  defending_team2_first.defender_9_id.value_counts().index[
                                                      0]].defender_9_val_x.mean(),
                            defending_team2_first[defending_team2_first['defender_10_id'] ==
                                                  defending_team2_first.defender_10_id.value_counts().index[
                                                      0]].defender_10_val_x.mean()]

    # formation identification
    # team1
    cluster_team1_defending_first = KMeans(n_clusters=4, random_state=0).fit(def_team1_first[['X']])

    clus_def_team1_first = pd.DataFrame()
    clus_def_team1_first['n'] = [0, 1, 2, 3]
    clus_def_team1_first['center'] = cluster_team1_defending_first.cluster_centers_
    clus_def_team1_first = clus_def_team1_first.sort_values(by=['center']).reset_index()

    formation_team1_defending_first = [
        len([x for x in cluster_team1_defending_first.labels_ if x == clus_def_team1_first.loc[0].n]),
        len([x for x in cluster_team1_defending_first.labels_ if x == clus_def_team1_first.loc[1].n]),
        len([x for x in cluster_team1_defending_first.labels_ if x == clus_def_team1_first.loc[2].n]),
        len([x for x in cluster_team1_defending_first.labels_ if x == clus_def_team1_first.loc[3].n])]

    cluster_team2_defending_first = KMeans(n_clusters=4, random_state=0).fit(def_team2_first[['X']])

    clus_def_team2_first = pd.DataFrame()
    clus_def_team2_first['n'] = [0, 1, 2, 3]
    clus_def_team2_first['center'] = cluster_team2_defending_first.cluster_centers_
    clus_def_team2_first = clus_def_team2_first.sort_values(by=['center']).reset_index()

    formation_team2_defending_first = [
        len([x for x in cluster_team2_defending_first.labels_ if x == clus_def_team2_first.loc[0].n]),
        len([x for x in cluster_team2_defending_first.labels_ if x == clus_def_team2_first.loc[1].n]),
        len([x for x in cluster_team2_defending_first.labels_ if x == clus_def_team2_first.loc[2].n]),
        len([x for x in cluster_team2_defending_first.labels_ if x == clus_def_team2_first.loc[3].n])]

    # allocation of players to groups
    def_defenders_team1_first = list(
        def_team1_first.sort_values(by=['X']).reset_index().loc[1:formation_team1_defending_first[1]].player_number)
    def_midfielders_team1_first = list(def_team1_first.sort_values(by=['X']).reset_index().loc[
                                       formation_team1_defending_first[1] + 1:formation_team1_defending_first[1] +
                                                                              formation_team1_defending_first[
                                                                                  2]].player_number)
    def_attackers_team1_first = list(def_team1_first.sort_values(by=['X']).reset_index().loc[
                                     formation_team1_defending_first[1] + formation_team1_defending_first[2] + 1:
                                     formation_team1_defending_first[1] + formation_team1_defending_first[2] +
                                     formation_team1_defending_first[3]].player_number)

    def_defenders_team2_first = list(
        def_team2_first.sort_values(by=['X']).reset_index().loc[1:formation_team2_defending_first[1]].player_number)
    def_midfielders_team2_first = list(def_team2_first.sort_values(by=['X']).reset_index().loc[
                                       formation_team2_defending_first[1] + 1:formation_team2_defending_first[1] +
                                                                              formation_team2_defending_first[
                                                                                  2]].player_number)
    def_attackers_team2_first = list(def_team2_first.sort_values(by=['X']).reset_index().loc[
                                     formation_team2_defending_first[1] + formation_team2_defending_first[2] + 1:
                                     formation_team2_defending_first[1] + formation_team2_defending_first[2] +
                                     formation_team2_defending_first[3]].player_number)

    # distances between formation lines
    positions_first_half['dis_def_mid_defending_team'] = np.where(
        (positions_first_half['half'] == 1) & (positions_first_half['defending_team_id'] == team1),
        abs(positions_first_half[def_defenders_team1_first].mean(axis=1) - positions_first_half[
            def_midfielders_team1_first].mean(axis=1)),
        abs(positions_first_half[def_defenders_team2_first].mean(axis=1) - positions_first_half[
            def_midfielders_team2_first].mean(axis=1)))
    positions_first_half['dis_mid_att_defending_team'] = np.where(
        (positions_first_half['half'] == 1) & (positions_first_half['defending_team_id'] == team1),
        abs(positions_first_half[def_midfielders_team1_first].mean(axis=1) - positions_first_half[
            def_attackers_team1_first].mean(axis=1)),
        abs(positions_first_half[def_midfielders_team2_first].mean(axis=1) - positions_first_half[
            def_attackers_team2_first].mean(axis=1)))
    positions_first_half['dis_def_att_defending_team'] = np.where(
        (positions_first_half['half'] == 1) & (positions_first_half['defending_team_id'] == team1),
        abs(positions_first_half[def_defenders_team1_first].mean(axis=1) - positions_first_half[
            def_attackers_team1_first].mean(axis=1)),
        abs(positions_first_half[def_defenders_team2_first].mean(axis=1) - positions_first_half[
            def_attackers_team2_first].mean(axis=1)))

    # SECOND HALF
    # formation identfification for first half
    positions_second_half = positions_attackers_defenders[positions_attackers_defenders['half'] == 2].copy()

    defending_team1_second = positions_second_half[positions_second_half['defending_team_id'] == team1].copy()
    defending_team2_second = positions_second_half[positions_second_half['defending_team_id'] == team2].copy()

    # Mirroring of data (both teams play from left to right)
    mean_team1_second = (
                                defending_team1_second.defender_0_val_x.mean() + defending_team1_second.defender_1_val_x.mean() + defending_team1_second.defender_2_val_x.mean() + defending_team1_second.defender_3_val_x.mean() + defending_team1_second.defender_4_val_x.mean() + defending_team1_second.defender_5_val_x.mean() + defending_team1_second.defender_6_val_x.mean() + defending_team1_second.defender_7_val_x.mean() + defending_team1_second.defender_8_val_x.mean() + defending_team1_second.defender_9_val_x.mean() + defending_team1_second.defender_10_val_x.mean()) / 11
    mean_team2_second = (
                                defending_team2_second.defender_0_val_x.mean() + defending_team2_second.defender_1_val_x.mean() + defending_team2_second.defender_2_val_x.mean() + defending_team2_second.defender_3_val_x.mean() + defending_team2_second.defender_4_val_x.mean() + defending_team2_second.defender_5_val_x.mean() + defending_team2_second.defender_6_val_x.mean() + defending_team2_second.defender_7_val_x.mean() + defending_team2_second.defender_8_val_x.mean() + defending_team2_second.defender_9_val_x.mean() + defending_team2_second.defender_10_val_x.mean()) / 11

    defending_team1_second['defender_0_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_0_val_x.copy() * (-1),
                                                          defending_team1_second.defender_0_val_x.copy())
    defending_team1_second['defender_1_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_1_val_x.copy() * (-1),
                                                          defending_team1_second.defender_1_val_x.copy())
    defending_team1_second['defender_2_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_2_val_x.copy() * (-1),
                                                          defending_team1_second.defender_2_val_x.copy())
    defending_team1_second['defender_3_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_3_val_x.copy() * (-1),
                                                          defending_team1_second.defender_3_val_x.copy())
    defending_team1_second['defender_4_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_4_val_x.copy() * (-1),
                                                          defending_team1_second.defender_4_val_x.copy())
    defending_team1_second['defender_5_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_5_val_x.copy() * (-1),
                                                          defending_team1_second.defender_5_val_x.copy())
    defending_team1_second['defender_6_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_6_val_x.copy() * (-1),
                                                          defending_team1_second.defender_6_val_x.copy())
    defending_team1_second['defender_7_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_7_val_x.copy() * (-1),
                                                          defending_team1_second.defender_7_val_x.copy())
    defending_team1_second['defender_8_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_8_val_x.copy() * (-1),
                                                          defending_team1_second.defender_8_val_x.copy())
    defending_team1_second['defender_9_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                          defending_team1_second.defender_9_val_x.copy() * (-1),
                                                          defending_team1_second.defender_9_val_x.copy())
    defending_team1_second['defender_10_val_x'] = np.where(mean_team1_second > mean_team2_second,
                                                           defending_team1_second.defender_10_val_x.copy() * (-1),
                                                           defending_team1_second.defender_10_val_x.copy())

    defending_team2_second['defender_0_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_0_val_x.copy() * (-1),
                                                          defending_team2_second.defender_0_val_x.copy())
    defending_team2_second['defender_1_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_1_val_x.copy() * (-1),
                                                          defending_team2_second.defender_1_val_x.copy())
    defending_team2_second['defender_2_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_2_val_x.copy() * (-1),
                                                          defending_team2_second.defender_2_val_x.copy())
    defending_team2_second['defender_3_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_3_val_x.copy() * (-1),
                                                          defending_team2_second.defender_3_val_x.copy())
    defending_team2_second['defender_4_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_4_val_x.copy() * (-1),
                                                          defending_team2_second.defender_4_val_x.copy())
    defending_team2_second['defender_5_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_5_val_x.copy() * (-1),
                                                          defending_team2_second.defender_5_val_x.copy())
    defending_team2_second['defender_6_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_6_val_x.copy() * (-1),
                                                          defending_team2_second.defender_6_val_x.copy())
    defending_team2_second['defender_7_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_7_val_x.copy() * (-1),
                                                          defending_team2_second.defender_7_val_x.copy())
    defending_team2_second['defender_8_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_8_val_x.copy() * (-1),
                                                          defending_team2_second.defender_8_val_x.copy())
    defending_team2_second['defender_9_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                          defending_team2_second.defender_9_val_x.copy() * (-1),
                                                          defending_team2_second.defender_9_val_x.copy())
    defending_team2_second['defender_10_val_x'] = np.where(mean_team1_second < mean_team2_second,
                                                           defending_team2_second.defender_10_val_x.copy() * (-1),
                                                           defending_team2_second.defender_10_val_x.copy())

    # Team1 is defending
    def_team1_second = pd.DataFrame()
    def_team1_second['player_id'] = [defending_team1_second.defender_0_id.value_counts().index[0],
                                     defending_team1_second.defender_1_id.value_counts().index[0],
                                     defending_team1_second.defender_2_id.value_counts().index[0],
                                     defending_team1_second.defender_3_id.value_counts().index[0],
                                     defending_team1_second.defender_4_id.value_counts().index[0],
                                     defending_team1_second.defender_5_id.value_counts().index[0],
                                     defending_team1_second.defender_6_id.value_counts().index[0],
                                     defending_team1_second.defender_7_id.value_counts().index[0],
                                     defending_team1_second.defender_8_id.value_counts().index[0],
                                     defending_team1_second.defender_9_id.value_counts().index[0],
                                     defending_team1_second.defender_10_id.value_counts().index[0]]
    def_team1_second['player_number'] = ['defender_0_val_x',
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
    def_team1_second['X'] = [defending_team1_second[defending_team1_second['defender_0_id'] ==
                                                    defending_team1_second.defender_0_id.value_counts().index[
                                                        0]].defender_0_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_1_id'] ==
                                                    defending_team1_second.defender_1_id.value_counts().index[
                                                        0]].defender_1_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_2_id'] ==
                                                    defending_team1_second.defender_2_id.value_counts().index[
                                                        0]].defender_2_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_3_id'] ==
                                                    defending_team1_second.defender_3_id.value_counts().index[
                                                        0]].defender_3_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_4_id'] ==
                                                    defending_team1_second.defender_4_id.value_counts().index[
                                                        0]].defender_4_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_5_id'] ==
                                                    defending_team1_second.defender_5_id.value_counts().index[
                                                        0]].defender_5_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_6_id'] ==
                                                    defending_team1_second.defender_6_id.value_counts().index[
                                                        0]].defender_6_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_7_id'] ==
                                                    defending_team1_second.defender_7_id.value_counts().index[
                                                        0]].defender_7_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_8_id'] ==
                                                    defending_team1_second.defender_8_id.value_counts().index[
                                                        0]].defender_8_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_9_id'] ==
                                                    defending_team1_second.defender_9_id.value_counts().index[
                                                        0]].defender_9_val_x.mean(),
                             defending_team1_second[defending_team1_second['defender_10_id'] ==
                                                    defending_team1_second.defender_10_id.value_counts().index[
                                                        0]].defender_10_val_x.mean()]

    # Team2 is defending
    def_team2_second = pd.DataFrame()
    def_team2_second['player_id'] = [defending_team2_second.defender_0_id.value_counts().index[0],
                                     defending_team2_second.defender_1_id.value_counts().index[0],
                                     defending_team2_second.defender_2_id.value_counts().index[0],
                                     defending_team2_second.defender_3_id.value_counts().index[0],
                                     defending_team2_second.defender_4_id.value_counts().index[0],
                                     defending_team2_second.defender_5_id.value_counts().index[0],
                                     defending_team2_second.defender_6_id.value_counts().index[0],
                                     defending_team2_second.defender_7_id.value_counts().index[0],
                                     defending_team2_second.defender_8_id.value_counts().index[0],
                                     defending_team2_second.defender_9_id.value_counts().index[0],
                                     defending_team2_second.defender_10_id.value_counts().index[0]]
    def_team2_second['player_number'] = ['defender_0_val_x',
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
    def_team2_second['X'] = [defending_team2_second[defending_team2_second['defender_0_id'] ==
                                                    defending_team2_second.defender_0_id.value_counts().index[
                                                        0]].defender_0_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_1_id'] ==
                                                    defending_team2_second.defender_1_id.value_counts().index[
                                                        0]].defender_1_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_2_id'] ==
                                                    defending_team2_second.defender_2_id.value_counts().index[
                                                        0]].defender_2_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_3_id'] ==
                                                    defending_team2_second.defender_3_id.value_counts().index[
                                                        0]].defender_3_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_4_id'] ==
                                                    defending_team2_second.defender_4_id.value_counts().index[
                                                        0]].defender_4_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_5_id'] ==
                                                    defending_team2_second.defender_5_id.value_counts().index[
                                                        0]].defender_5_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_6_id'] ==
                                                    defending_team2_second.defender_6_id.value_counts().index[
                                                        0]].defender_6_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_7_id'] ==
                                                    defending_team2_second.defender_7_id.value_counts().index[
                                                        0]].defender_7_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_8_id'] ==
                                                    defending_team2_second.defender_8_id.value_counts().index[
                                                        0]].defender_8_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_9_id'] ==
                                                    defending_team2_second.defender_9_id.value_counts().index[
                                                        0]].defender_9_val_x.mean(),
                             defending_team2_second[defending_team2_second['defender_10_id'] ==
                                                    defending_team2_second.defender_10_id.value_counts().index[
                                                        0]].defender_10_val_x.mean()]
    # formation identification
    # team1
    cluster_team1_defending_second = KMeans(n_clusters=4, random_state=0).fit(def_team1_second[['X']])

    clus_def_team1_second = pd.DataFrame()
    clus_def_team1_second['n'] = [0, 1, 2, 3]
    clus_def_team1_second['center'] = cluster_team1_defending_second.cluster_centers_
    clus_def_team1_second = clus_def_team1_second.sort_values(by=['center']).reset_index()

    formation_team1_defending_second = [
        len([x for x in cluster_team1_defending_second.labels_ if x == clus_def_team1_second.loc[0].n]),
        len([x for x in cluster_team1_defending_second.labels_ if x == clus_def_team1_second.loc[1].n]),
        len([x for x in cluster_team1_defending_second.labels_ if x == clus_def_team1_second.loc[2].n]),
        len([x for x in cluster_team1_defending_second.labels_ if x == clus_def_team1_second.loc[3].n])]

    cluster_team2_defending_second = KMeans(n_clusters=4, random_state=0).fit(def_team2_second[['X']])

    clus_def_team2_second = pd.DataFrame()
    clus_def_team2_second['n'] = [0, 1, 2, 3]
    clus_def_team2_second['center'] = cluster_team2_defending_second.cluster_centers_
    clus_def_team2_second = clus_def_team2_second.sort_values(by=['center']).reset_index()

    formation_team2_defending_second = [
        len([x for x in cluster_team2_defending_second.labels_ if x == clus_def_team2_second.loc[0].n]),
        len([x for x in cluster_team2_defending_second.labels_ if x == clus_def_team2_second.loc[1].n]),
        len([x for x in cluster_team2_defending_second.labels_ if x == clus_def_team2_second.loc[2].n]),
        len([x for x in cluster_team2_defending_second.labels_ if x == clus_def_team2_second.loc[3].n])]

    # allocation of players to groups
    def_defenders_team1_second = list(
        def_team1_second.sort_values(by=['X']).reset_index().loc[1:formation_team1_defending_second[1]].player_number)
    def_midfielders_team1_second = list(def_team1_second.sort_values(by=['X']).reset_index().loc[
                                        formation_team1_defending_second[1] + 1:formation_team1_defending_second[1] +
                                                                                formation_team1_defending_second[
                                                                                    2]].player_number)
    def_attackers_team1_second = list(def_team1_second.sort_values(by=['X']).reset_index().loc[
                                      formation_team1_defending_second[1] + formation_team1_defending_second[2] + 1:
                                      formation_team1_defending_second[1] + formation_team1_defending_second[2] +
                                      formation_team1_defending_second[3]].player_number)

    def_defenders_team2_second = list(
        def_team2_second.sort_values(by=['X']).reset_index().loc[1:formation_team2_defending_second[1]].player_number)
    def_midfielders_team2_second = list(def_team2_second.sort_values(by=['X']).reset_index().loc[
                                        formation_team2_defending_second[1] + 1:formation_team2_defending_second[1] +
                                                                                formation_team2_defending_second[
                                                                                    2]].player_number)
    def_attackers_team2_second = list(def_team2_second.sort_values(by=['X']).reset_index().loc[
                                      formation_team2_defending_second[1] + formation_team2_defending_second[2] + 1:
                                      formation_team2_defending_second[1] + formation_team2_defending_second[2] +
                                      formation_team2_defending_second[3]].player_number)

    # distances between formation lines
    positions_second_half['dis_def_mid_defending_team'] = np.where(
        (positions_second_half['half'] == 2) & (positions_second_half['defending_team_id'] == team1),
        abs(positions_second_half[def_defenders_team1_second].mean(axis=1) - positions_second_half[
            def_midfielders_team1_second].mean(axis=1)),
        abs(positions_second_half[def_defenders_team2_second].mean(axis=1) - positions_second_half[
            def_midfielders_team2_second].mean(axis=1)))
    positions_second_half['dis_mid_att_defending_team'] = np.where(
        (positions_second_half['half'] == 2) & (positions_second_half['defending_team_id'] == team1),
        abs(positions_second_half[def_midfielders_team1_second].mean(axis=1) - positions_second_half[
            def_attackers_team1_second].mean(axis=1)),
        abs(positions_second_half[def_midfielders_team2_second].mean(axis=1) - positions_second_half[
            def_attackers_team2_second].mean(axis=1)))
    positions_second_half['dis_def_att_defending_team'] = np.where(
        (positions_second_half['half'] == 2) & (positions_second_half['defending_team_id'] == team1),
        abs(positions_second_half[def_defenders_team1_second].mean(axis=1) - positions_second_half[
            def_attackers_team1_second].mean(axis=1)),
        abs(positions_second_half[def_defenders_team2_second].mean(axis=1) - positions_second_half[
            def_attackers_team2_second].mean(axis=1)))

    # append first and second half:
    df_final = positions_first_half.append(positions_second_half)
    final_columns = ['match_id', 'half', 'frame', 'defending_team_id', 'dis_def_mid_defending_team',
                     'dis_mid_att_defending_team', 'dis_def_att_defending_team']
    df_final = df_final[final_columns].rename(columns={"defending_team_id": "team_id"})

    return df_final
