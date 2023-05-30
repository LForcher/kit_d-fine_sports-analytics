import pandas as pd
import numpy as np
from src.utils import utils
from src import config

def defensive_pressure(match_id: str) -> pd.DataFrame:
    """
    Quantification of defensive pressure for every timestamp using an advanced and adapted model of Andrienko et al.
    Args:
        match_id: DFL match ID

    Returns: dataframe with columns ["match_id", "half", "frame", "player_id",
                        "defensive_pressure", "other_player_id"]

    """
    positions_attackers_defenders = utils.get_positions_attackers_defenders_data([match_id])
    if config.get_reduce_metrics_to_every_fifth_frame():
        positions_attackers_defenders = positions_attackers_defenders[positions_attackers_defenders["frame"] % 5 == 0]
        positions_attackers_defenders.reset_index(inplace=True)
    if positions_attackers_defenders.empty:
        print(f"Warning: No position data, thus skipping metric defensive_pressure, for match {match_id}")
        return pd.DataFrame()
    df = defensive_pressure_quantification(positions_attackers_defenders)
    return df


def defensive_pressure_quantification(positions_attacker_defenders: pd.DataFrame) -> pd.DataFrame:
    """
    Quantification of defensive pressure for every timestamp using an advanced and adapted model of Andrienko et al.
    Args:
        positions_attacker_defenders: as given by table table_name_positions_of_attackers_and_defenders

    Returns: dataframe with columns ["match_id", "half", "frame", "player_id",
                        "defensive_pressure", "other_player_id"]
    """
    all_results = []
    for atk_idx in range(11):
        df = positions_attacker_defenders.copy()

        dfd_press_list = list()
        for dfd_idx in range(11):
            dfd_press_list.append(pressure_quantification(
                df[f"attacker_{atk_idx}_val_x"].copy(),
                df[f"attacker_{atk_idx}_val_y"].copy(),
                df[f"defender_{dfd_idx}_val_x"].copy(),
                df[f"defender_{dfd_idx}_val_y"].copy(),
                df["attacked_goal_x"].copy(),
                df["attacked_goal_y"].copy()))

        # determine which defender has the highest pressure on the attacker
        dfd_press = pd.DataFrame(np.transpose(dfd_press_list), columns=range(11))
        max_dfd_press = np.max(dfd_press, axis=1)
        arg_max_dfd_press = np.argmax(dfd_press.values, axis=1, keepdims=True)
        defender = np.array([[None]] * len(arg_max_dfd_press))
        for p_idx in range(11):
            dfd_id = np.array(df[f"defender_{p_idx}_id"].values).reshape((-1, 1))
            defender = np.where(arg_max_dfd_press == p_idx, dfd_id, defender)

        df.rename(columns={f"attacker_{atk_idx}_id": "player_id"}, inplace=True)
        df = df[["match_id", "half", "frame", "player_id"]].copy()
        df["defensive_pressure"] = max_dfd_press
        df["other_player_id"] = defender
        all_results.append(df)

    df_final = pd.concat(all_results, axis=0)
    return df_final


def pressure_quantification(attacker_x: pd.Series, attacker_y: pd.Series, defender_x: pd.Series, defender_y: pd.Series,
                            attacked_goal_x: pd.Series, attacked_goal_y: pd.Series) -> np.array:
    """
    Quantification of defensive pressure for every timestamp using an advanced and adapted model of Andrienko et al.
    Warning: Input values are shifted inplace!
    Args:
        attacker_x: series of x values for attacker
        attacker_y: series of y values for attacker
        defender_x: series of x values for defender
        defender_y: series of y values for defender
        attacked_goal_x: series of x values for attacked_goal
        attacked_goal_y: series of y values for attacked_goal

    Returns: Array (same shape as all input series) of pressure values.
    """
    atk_dfd_distance = utils.euclid_metric(attacker_x, attacker_y, defender_x, defender_y)
    attacker_to_goal_distance = utils.euclid_metric(attacker_x, attacker_y, attacked_goal_x, attacked_goal_y)
    defender_to_goal_distance = utils.euclid_metric(defender_x, defender_y, attacked_goal_x, attacked_goal_y)

    # angle between attacker and goal to define threat direction
    threat_direction = np.degrees(np.arcsin((abs(attacker_y - attacked_goal_y)) / attacker_to_goal_distance))

    # definition of angle between defender and attacker dependant on threat direction
    angle_defender = np.degrees(np.arcsin((abs(defender_y - attacked_goal_y)) / defender_to_goal_distance))

    angle_between = abs(threat_direction - angle_defender)

    # definition of distance from attacker to the goal (necessary for the pressure form)
    left_goal_x = utils.get_position_left_goal_x()
    right_goal_x = utils.get_position_right_goal_x()
    goal_y = utils.get_goal_y()

    angle_between = np.where(
        np.logical_or(
            np.logical_and(attacked_goal_x == left_goal_x, defender_x > attacker_x),
            np.logical_and(attacked_goal_x != left_goal_x, defender_x < attacker_x)
        ),
        # Case 1 : Defender is in front of the Attacker:
        angle_between,
        # Case 2 : Defender is behind of the Attacker:
        angle_between + 180,
    )

    # definition of pressure area
    form = (1 - (np.cos(np.radians(angle_between)))) / 2
    # definition of length of the pressure area dependent on the attacker position
    # gets smaller the closer the attacker gets to the goal
    # inside the penalty area GoalDis is divided by 2 to increase the drop of the size of the pressure area
    front = np.where(
        np.logical_or(
            np.logical_and(np.logical_and(attacked_goal_x == left_goal_x, attacker_x <= left_goal_x + 16),
                           np.logical_and(attacker_y >= goal_y - 16, attacker_y <= goal_y + 16)),
            np.logical_and(np.logical_and(attacked_goal_x != left_goal_x, attacker_x >= right_goal_x - 16),
                           np.logical_and(attacker_y >= goal_y - 16, attacker_y <= goal_y + 16))
        ),
        ((attacker_to_goal_distance / 2) * 0.05) + 3.75,
        (attacker_to_goal_distance * 0.05) + 3.75,
    )
    back = front / 3

    length = back + ((front - back) * ((form ** 3) + (form * 0.3)) / 1.3)

    pressure = np.where(
        atk_dfd_distance > length,
        # zero pressure if distance to large
        0,
        (1 - atk_dfd_distance / length) ** 1.75 * 100
    )
    return pressure
