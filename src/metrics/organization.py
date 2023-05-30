from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
from src.utils import utils
from src import config


def surface_area(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7, x_8, y_8, x_9, y_9, x_10, y_10):
    """
    Quantification of surface area for 10 players  using the surface area of a Convex Hull:
    """
    points = np.array(
        [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5], [x_6, y_6], [x_7, y_7], [x_8, y_8], [x_9, y_9],
         [x_10, y_10]])
    if np.isnan(points).any():
        return np.nan
    hull = ConvexHull(points)
    df = hull.volume
    return df


surface_area_vectorized = np.vectorize(surface_area)


def centroid_x(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7, x_8, y_8, x_9, y_9, x_10, y_10):
    """
    Quantification of y-Centroid for 10 players:
    """
    points = np.array(
        [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5], [x_6, y_6], [x_7, y_7], [x_8, y_8], [x_9, y_9],
         [x_10, y_10]])
    df = np.mean(points[:, 0])
    return df


centroid_x_vectorized = np.vectorize(centroid_x)


def centroid_y(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7, x_8, y_8, x_9, y_9, x_10, y_10):
    """
    Quantification of x-Centroid for 10 players:
    """
    points = np.array(
        [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5], [x_6, y_6], [x_7, y_7], [x_8, y_8], [x_9, y_9],
         [x_10, y_10]])
    df = np.mean(points[:, 1])
    return df


centroid_y_vectorized = np.vectorize(centroid_y)


def spread(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, x_5, y_5, x_6, y_6, x_7, y_7, x_8, y_8, x_9, y_9, x_10, y_10):
    """
    Quantification of spread for 10 players using the squared deviation from the mean:
    """
    points = np.array(
        [[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5], [x_6, y_6], [x_7, y_7], [x_8, y_8], [x_9, y_9],
         [x_10, y_10]])
    df = (np.var(points[:, 0]) + np.var(points[:, 1])) * 10
    return df


spread_vectorized = np.vectorize(spread)


# %%
def organization(match_id: str) -> pd.DataFrame:
    """
    Quantification of organization for attacking and defending team (without the goalkeeper) for every timestamp using:
        - surface area (or convex Hull) 
        - spread 
        - centroid position
    
    Args:
        match_id: DFL match ID

    Returns: dataframe with columns ['match_id', 'half', 'frame', 
                                     'attacking_team_id', 'surface_area_attacking_team', 'spread_attacking_team',
                                     'centroid_x_attacking_team', 'centroid_y_attacking_team',
                                     'defending_team_id', 'surface_area_defending_team', 'spread_defending_team',
                                     'centroid_x_defending_team', 'centroid_y_defending_team']
    """

    # 1 load data of atatckers and defender
    positions_attackers_defenders = utils.get_positions_attackers_defenders_data([match_id])
    if config.get_reduce_metrics_to_every_fifth_frame():
        positions_attackers_defenders = positions_attackers_defenders[
            positions_attackers_defenders["frame"] % 5 == 0].copy()
        positions_attackers_defenders.reset_index(inplace=True)
    if positions_attackers_defenders.empty:
        print(f"Warning: No position data, thus skipping metric organization, for match {match_id}")
        return pd.DataFrame()

    # Identification of goalkeeper to exclude him from team organizational measures
    positions_first = positions_attackers_defenders[positions_attackers_defenders['half'] == 1]
    first_line = positions_attackers_defenders.loc[[0]].reset_index(inplace=False)
    # Team 1
    team_1_id_cols = [col for col in first_line.columns if ('attacker' in col and 'id' in col)]
    team_1_ids = list(first_line[team_1_id_cols].iloc[0])
    if not positions_first["attacker_0_id"].isin(team_1_ids).any():
        print("Error! #10 Attacker_0_id wurde anscheinend ausgewechselt, daher funktioniert die Berechnung nicht!")
        print(f"Error! #10 Aborting {match_id} for metric organization!")
        return pd.DataFrame()

    team_1_attacking_first = positions_first[(positions_first['attacker_0_id'] == team_1_ids[0]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[1]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[2]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[3]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[4]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[5]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[6]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[7]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[8]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[9]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[10])]
    # Team 2
    team_2_id_cols = [col for col in first_line.columns if ('defender' in col and 'id' in col)]
    team_2_ids = list(first_line[team_2_id_cols].iloc[0])
    team_2_defending_first = positions_first[(positions_first['defender_0_id'] == team_2_ids[0]) |
                                             (positions_first['defender_0_id'] == team_2_ids[1]) |
                                             (positions_first['defender_0_id'] == team_2_ids[2]) |
                                             (positions_first['defender_0_id'] == team_2_ids[3]) |
                                             (positions_first['defender_0_id'] == team_2_ids[4]) |
                                             (positions_first['defender_0_id'] == team_2_ids[5]) |
                                             (positions_first['defender_0_id'] == team_2_ids[6]) |
                                             (positions_first['defender_0_id'] == team_2_ids[7]) |
                                             (positions_first['defender_0_id'] == team_2_ids[8]) |
                                             (positions_first['defender_0_id'] == team_2_ids[9]) |
                                             (positions_first['defender_0_id'] == team_2_ids[10])]
    # identification of playing direction
    if team_1_attacking_first.attacked_goal_x.max() == utils.get_position_left_goal_x():
        # attacking team plays from rigth to left
        ascending_att = False
        ascending_def = True
    elif team_1_attacking_first.attacked_goal_x.max() == utils.get_position_right_goal_x():
        # attacking team plays from left to right
        ascending_att = True
        ascending_def = False
    else:
        raise ValueError(
            f"#34 Other values for team_1_attacking_first were not expected"
            f" (value = {team_1_attacking_first.attacked_goal_x.max()} ")

    # Find the goalkeeper id for Team 1 (with x-values closets to the goal)
    team_1_x_cols = [col for col in first_line.columns if ('attacker' in col and '_x' in col)]
    goalkeeper_team1_att = \
        team_1_attacking_first[team_1_x_cols].mean().sort_values(ascending=ascending_att).reset_index().loc[0]['index']

    # Find the goalkeeper id for Team 2 (with x-values closets to the goal)
    team_2_x_cols = [col for col in first_line.columns if ('defender' in col and '_x' in col)]
    goalkeeper_team2_def = \
        team_2_defending_first[team_2_x_cols].mean().sort_values(ascending=ascending_def).reset_index().loc[0]['index']

    # Find goalkeeper when attacking and defending switches
    first_line_2 = positions_attackers_defenders[
        positions_attackers_defenders['attacked_goal_x'] != positions_attackers_defenders.loc[
            [0]].attacked_goal_x.item()].reset_index(inplace=False).loc[[0]]

    # Team 1
    team_1_id_cols = [col for col in first_line_2.columns if ('attacker' in col and 'id' in col)]
    team_1_ids = list(first_line_2[team_1_id_cols].iloc[0])
    team_1_attacking_first = positions_first[(positions_first['attacker_0_id'] == team_1_ids[0]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[1]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[2]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[3]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[4]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[5]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[6]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[7]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[8]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[9]) |
                                             (positions_first['attacker_0_id'] == team_1_ids[10])]
    # Team 2
    team_2_id_cols = [col for col in first_line_2.columns if ('defender' in col and 'id' in col)]
    team_2_ids = list(first_line_2[team_2_id_cols].iloc[0])
    team_2_defending_first = positions_first[(positions_first['defender_0_id'] == team_2_ids[0]) |
                                             (positions_first['defender_0_id'] == team_2_ids[1]) |
                                             (positions_first['defender_0_id'] == team_2_ids[2]) |
                                             (positions_first['defender_0_id'] == team_2_ids[3]) |
                                             (positions_first['defender_0_id'] == team_2_ids[4]) |
                                             (positions_first['defender_0_id'] == team_2_ids[5]) |
                                             (positions_first['defender_0_id'] == team_2_ids[6]) |
                                             (positions_first['defender_0_id'] == team_2_ids[7]) |
                                             (positions_first['defender_0_id'] == team_2_ids[8]) |
                                             (positions_first['defender_0_id'] == team_2_ids[9]) |
                                             (positions_first['defender_0_id'] == team_2_ids[10])]
    # identification of playing direction
    if team_1_attacking_first.attacked_goal_x.max() == utils.get_position_left_goal_x():
        # attacking team plays from rigth to left
        ascending_att = False
        ascending_def = True
    elif team_1_attacking_first.attacked_goal_x.max() == utils.get_position_right_goal_x():
        # attacking team plays from left to right
        ascending_att = True
        ascending_def = False

    # Find the goalkeeper id for Team 1 (with x-values closets to the goal)
    team_1_x_cols = [col for col in first_line.columns if ('attacker' in col and '_x' in col)]
    goalkeeper_team2_att = \
        team_1_attacking_first[team_1_x_cols].mean().sort_values(ascending=ascending_att).reset_index().loc[0]['index']

    # Find the goalkeeper id for Team 2 (with x-values closets to the goal)
    team_2_x_cols = [col for col in first_line.columns if ('defender' in col and '_x' in col)]
    goalkeeper_team1_def = \
        team_2_defending_first[team_2_x_cols].mean().sort_values(ascending=ascending_def).reset_index().loc[0]['index']

    # exclusion of goalkeepers
    team1_att_excluded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    team1_att_excluded.remove(int(goalkeeper_team1_att.split('_')[1]))

    team2_def_excluded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    team2_def_excluded.remove(int(goalkeeper_team2_def.split('_')[1]))

    team1_def_excluded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    team1_def_excluded.remove(int(goalkeeper_team1_def.split('_')[1]))

    team2_att_excluded = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    team2_att_excluded.remove(int(goalkeeper_team2_att.split('_')[1]))

    # calculation of surface area
    positions_attackers_defenders['surface_area_attacking_team'] = np.where(
        positions_attackers_defenders.attacked_goal_x == first_line.attacked_goal_x.item(),
        surface_area_vectorized(positions_attackers_defenders['attacker_' + str(team1_att_excluded[0]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[0]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[1]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[1]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[2]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[2]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[3]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[3]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[4]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[4]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[5]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[5]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[6]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[6]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[7]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[7]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[8]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[8]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[9]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team1_att_excluded[9]) + '_val_y']),
        surface_area_vectorized(positions_attackers_defenders['attacker_' + str(team2_att_excluded[0]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[0]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[1]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[1]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[2]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[2]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[3]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[3]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[4]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[4]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[5]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[5]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[6]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[6]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[7]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[7]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[8]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[8]) + '_val_y'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[9]) + '_val_x'],
                                positions_attackers_defenders['attacker_' + str(team2_att_excluded[9]) + '_val_y']))

    positions_attackers_defenders['surface_area_defending_team'] = np.where(
        positions_attackers_defenders.attacked_goal_x == first_line.attacked_goal_x.item(),
        surface_area_vectorized(positions_attackers_defenders['defender_' + str(team2_def_excluded[0]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[0]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[1]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[1]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[2]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[2]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[3]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[3]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[4]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[4]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[5]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[5]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[6]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[6]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[7]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[7]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[8]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[8]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[9]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team2_def_excluded[9]) + '_val_y']),
        surface_area_vectorized(positions_attackers_defenders['defender_' + str(team1_def_excluded[0]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[0]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[1]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[1]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[2]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[2]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[3]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[3]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[4]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[4]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[5]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[5]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[6]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[6]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[7]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[7]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[8]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[8]) + '_val_y'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[9]) + '_val_x'],
                                positions_attackers_defenders['defender_' + str(team1_def_excluded[9]) + '_val_y']))

    # Centroid Calculation
    positions_attackers_defenders['centroid_x_attacking_team'] = np.where(
        positions_attackers_defenders.attacked_goal_x == first_line.attacked_goal_x.item(),
        centroid_x_vectorized(positions_attackers_defenders['attacker_' + str(team1_att_excluded[0]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[0]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[1]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[1]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[2]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[2]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[3]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[3]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[4]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[4]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[5]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[5]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[6]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[6]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[7]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[7]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[8]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[8]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[9]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[9]) + '_val_y']),
        centroid_x_vectorized(positions_attackers_defenders['attacker_' + str(team2_att_excluded[0]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[0]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[1]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[1]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[2]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[2]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[3]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[3]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[4]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[4]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[5]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[5]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[6]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[6]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[7]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[7]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[8]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[8]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[9]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[9]) + '_val_y']))

    positions_attackers_defenders['centroid_x_defending_team'] = np.where(
        positions_attackers_defenders.attacked_goal_x == first_line.attacked_goal_x.item(),
        centroid_x_vectorized(positions_attackers_defenders['defender_' + str(team2_def_excluded[0]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[0]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[1]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[1]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[2]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[2]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[3]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[3]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[4]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[4]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[5]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[5]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[6]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[6]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[7]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[7]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[8]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[8]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[9]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[9]) + '_val_y']),
        centroid_x_vectorized(positions_attackers_defenders['defender_' + str(team1_def_excluded[0]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[0]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[1]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[1]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[2]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[2]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[3]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[3]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[4]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[4]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[5]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[5]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[6]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[6]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[7]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[7]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[8]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[8]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[9]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[9]) + '_val_y']))

    positions_attackers_defenders['centroid_y_attacking_team'] = np.where(
        positions_attackers_defenders.attacked_goal_x == first_line.attacked_goal_x.item(),
        centroid_y_vectorized(positions_attackers_defenders['attacker_' + str(team1_att_excluded[0]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[0]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[1]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[1]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[2]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[2]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[3]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[3]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[4]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[4]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[5]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[5]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[6]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[6]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[7]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[7]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[8]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[8]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[9]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team1_att_excluded[9]) + '_val_y']),
        centroid_y_vectorized(positions_attackers_defenders['attacker_' + str(team2_att_excluded[0]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[0]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[1]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[1]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[2]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[2]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[3]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[3]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[4]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[4]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[5]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[5]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[6]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[6]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[7]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[7]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[8]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[8]) + '_val_y'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[9]) + '_val_x'],
                              positions_attackers_defenders['attacker_' + str(team2_att_excluded[9]) + '_val_y']))

    positions_attackers_defenders['centroid_y_defending_team'] = np.where(
        positions_attackers_defenders.attacked_goal_x == first_line.attacked_goal_x.item(),
        centroid_y_vectorized(positions_attackers_defenders['defender_' + str(team2_def_excluded[0]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[0]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[1]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[1]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[2]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[2]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[3]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[3]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[4]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[4]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[5]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[5]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[6]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[6]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[7]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[7]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[8]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[8]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[9]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team2_def_excluded[9]) + '_val_y']),
        centroid_y_vectorized(positions_attackers_defenders['defender_' + str(team1_def_excluded[0]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[0]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[1]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[1]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[2]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[2]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[3]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[3]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[4]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[4]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[5]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[5]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[6]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[6]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[7]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[7]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[8]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[8]) + '_val_y'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[9]) + '_val_x'],
                              positions_attackers_defenders['defender_' + str(team1_def_excluded[9]) + '_val_y']))

    positions_attackers_defenders['spread_attacking_team'] = np.where(
        positions_attackers_defenders.attacked_goal_x == first_line.attacked_goal_x.item(),
        spread_vectorized(positions_attackers_defenders['attacker_' + str(team1_att_excluded[0]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[0]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[1]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[1]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[2]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[2]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[3]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[3]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[4]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[4]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[5]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[5]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[6]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[6]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[7]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[7]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[8]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[8]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[9]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team1_att_excluded[9]) + '_val_y']),
        spread_vectorized(positions_attackers_defenders['attacker_' + str(team2_att_excluded[0]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[0]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[1]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[1]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[2]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[2]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[3]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[3]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[4]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[4]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[5]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[5]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[6]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[6]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[7]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[7]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[8]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[8]) + '_val_y'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[9]) + '_val_x'],
                          positions_attackers_defenders['attacker_' + str(team2_att_excluded[9]) + '_val_y']))

    positions_attackers_defenders['spread_defending_team'] = np.where(
        positions_attackers_defenders.attacked_goal_x == first_line.attacked_goal_x.item(),
        spread_vectorized(positions_attackers_defenders['defender_' + str(team2_def_excluded[0]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[0]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[1]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[1]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[2]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[2]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[3]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[3]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[4]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[4]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[5]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[5]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[6]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[6]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[7]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[7]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[8]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[8]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[9]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team2_def_excluded[9]) + '_val_y']),
        spread_vectorized(positions_attackers_defenders['defender_' + str(team1_def_excluded[0]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[0]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[1]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[1]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[2]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[2]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[3]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[3]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[4]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[4]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[5]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[5]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[6]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[6]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[7]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[7]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[8]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[8]) + '_val_y'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[9]) + '_val_x'],
                          positions_attackers_defenders['defender_' + str(team1_def_excluded[9]) + '_val_y']))

    df_defense = positions_attackers_defenders[['match_id', 'half', 'frame',
                                                'defending_team_id', 'surface_area_defending_team',
                                                'spread_defending_team', 'centroid_x_defending_team',
                                                'centroid_y_defending_team']].rename(
        columns={
            "defending_team_id": "team_id", "surface_area_defending_team": "surface_area",
            "spread_defending_team": "spread_team",
            "centroid_x_defending_team": "centroid_x_team", "centroid_y_defending_team": "centroid_y_team"
        })
    df_offense = positions_attackers_defenders[['match_id', 'half', 'frame',
                                                'attacking_team_id', 'surface_area_attacking_team',
                                                'spread_attacking_team', 'centroid_x_attacking_team',
                                                'centroid_y_attacking_team']].rename(
        columns={
            "attacking_team_id": "team_id", "surface_area_attacking_team": "surface_area",
            "spread_attacking_team": "spread_team",
            "centroid_x_attacking_team": "centroid_x_team", "centroid_y_attacking_team": "centroid_y_team"
        })
    df_final = pd.concat((df_defense, df_offense), ignore_index=True)
    return df_final
