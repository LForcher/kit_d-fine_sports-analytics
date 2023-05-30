import pandas as pd
import numpy as np
from src.utils import utils, db_handler


def main(match_id):
    """
    Returns the information in which lanes and which thirds the ball is at each given frame. See other methods of this
    script for further information
    Args:
        match_id: dfl match id

    Returns: df with columns match_id, half, frame, third, lane.

    """
    positions = utils.get_positions_attackers_defenders_data([match_id])
    thirds = get_thirds(positions["match_id"], positions["ball_y"], positions["attacked_goal_y"])
    lanes = get_lanes(positions["match_id"], positions["ball_x"], positions["attacked_goal_y"])
    df = positions[["match_id", "half", "frame"]].copy()
    df["third"] = thirds
    df["lane"] = lanes
    return df


def get_lanes(match_ids: pd.Series, ball_x: pd.Series, attacked_goal_y: pd.Series) -> np.array:
    """
    Returns the information on which lane the ball is in each frame. The pitch is divided into the three lanes left,
    middle and right.
    Args:
        match_ids: list of match_ids to get the lanes for - used to extract pitch sizes
        ball_x: x coordinates of the ball
        attacked_goal_y: information on playing direction

    Returns: 1d array with values left, middle and right

    """
    pitches = db_handler.get_table(utils.table_name_pitch)
    pos_pitch = pd.merge(match_ids, pitches, on=["match_id"], how="left", validate="many_to_one")
    if (pos_pitch["match_id"] != match_ids).any():
        raise ValueError(f"Order of match_ids changed!")

    x_range_third = (pos_pitch["x_end"] - pos_pitch["x_start"]) / 3
    right_lane = np.where(attacked_goal_y == 0,
                          ball_x >= pos_pitch["x_end"] - x_range_third,
                          ball_x <= pos_pitch["x_start"] + x_range_third)
    left_lane = np.where(attacked_goal_y == 0,
                         ball_x <= pos_pitch["x_start"] + x_range_third,
                         ball_x >= pos_pitch["x_end"] - x_range_third)
    middle_lane = (pos_pitch["x_start"] + x_range_third < ball_x) & (
            ball_x < pos_pitch["x_end"] - x_range_third)
    if ((right_lane * 1 + left_lane * 1 + middle_lane * 1).values != [1] * len(right_lane)).any():
        raise ValueError("#88 wrong implementation!")
    thirds = np.where(left_lane, "left",
                      np.where(middle_lane, "middle",
                               "right"))
    return thirds


def get_thirds(match_ids: pd.Series, ball_y: pd.Series, attacked_goal_y: pd.Series) -> np.array:
    """
    Returns the information in which third the ball is in each frame. The pitch is divided into the three thirds
    attacking, middle and defending with the attacking third being closest to the opponents goal
    Args:
        match_ids: list of match_ids to get the lanes for - used to extract pitch sizes
        ball_y: y coordinates of the ball
        attacked_goal_y: information on playing direction

    Returns: 1d array with values left, middle and right

    """
    pitches = db_handler.get_table(utils.table_name_pitch)
    pos_pitch = pd.merge(match_ids, pitches, on=["match_id"], how="left", validate="many_to_one")
    if (pos_pitch["match_id"].values != match_ids.values).any():
        raise ValueError(f"Order of match_ids changed!")

    y_range_third = (pos_pitch["y_end"] - pos_pitch["y_start"]) / 3
    defending_third = np.where(attacked_goal_y == pos_pitch["y_start"],
                               ball_y <= pos_pitch["y_start"] + y_range_third,
                               ball_y >= pos_pitch["y_end"] - y_range_third)
    attacking_third = np.where(attacked_goal_y != pos_pitch["y_start"],
                               ball_y <= pos_pitch["y_start"] + y_range_third,
                               ball_y >= pos_pitch["y_end"] - y_range_third)
    middle_third = (pos_pitch["y_start"] + y_range_third < ball_y) & (
            ball_y < pos_pitch["y_end"] - y_range_third)
    if ((defending_third * 1 + attacking_third * 1 + middle_third * 1).values != [1] * len(defending_third)).any():
        raise ValueError("#88 wrong implementation!")
    thirds = np.where(attacking_third, "attacking",
                      np.where(middle_third, "middle",
                               "defending"))
    return thirds
