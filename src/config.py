import configparser
import multiprocessing
from src.postprocessing import postprocessing_utils
import pandas as pd

Config = configparser.ConfigParser()
Config.read("src/config.ini", encoding="utf-8")
metrics = {"defensive_pressure": "player", "velocity": "player", "distance_to_the_ball": "player",
           "organization": "team", "numerical_superiority": "team", "formation_lines": "team"}
preprocessing_steps = {"parse_data", "match_positions_events", "rearrange_positions_to_attacker_and_defenders",
                       "rearrange_positions_by_teams",
                       "rearrange_positions_to_attacker_and_defenders_sorted_by_dist_to_ball"}

status_column_names = ['possession',
                       'ballstatus',
                       'match_id',
                       'half',
                       'frame',
                       'Time [s]',
                       'team',
                       'dfl_event_id',
                       'tID',
                       'pID',
                       'outcome',
                       'timestamp',
                       'qualifier',
                       'frame_start',
                       'frame_end',
                       'event_type',
                       'recipient',
                       'player']

tracking_status_column_names = ['possession',
                                'ballstatus',
                                'match_id',
                                'half',
                                'frame']

target_data_columns = ['possession',
                       'ballstatus',
                       'match_id',
                       'half',
                       'frame',
                       'team',
                       'tID',
                       'pID',
                       'recipient',
                       'outcome',
                       'qualifier',
                       'frame_start',
                       'frame_end',
                       'event_type']


def get_matches_to_process() -> list or str:
    """ Returns list of matches to process based on config. Instead of list, also the return value "all" is possible."""
    str_matches = Config.get("general", "matches_to_process")
    if str_matches == "all":
        return str_matches
    match_ids = str_matches.replace(" ", "").replace("[", "").replace("]", "").split(",")
    matches_to_process = [match for match in match_ids if match not in get_blacklist_matches()]
    return matches_to_process


def get_blacklist_matches() -> list:
    str_matches_blacklist = Config.get("general", "blacklist_matches")
    blacklist_match_ids = str_matches_blacklist.replace(" ", "").replace("[", "").replace("]", "").split(",")
    if "" in blacklist_match_ids:
        blacklist_match_ids.remove("")
    return blacklist_match_ids


def get_skip_existing_matches(process="preprocessing") -> bool:
    if process not in ["preprocessing", "postprocessing", "target_dataset"]:
        raise ValueError("#22 invalid argument for process in get_use_parallelization in config.py")
    return Config.get(process, "skip_existing_matches").lower() == "true"


def get_use_parallelization(process: "" or "" = "preprocessing") -> bool:
    """
    Args:
        process: "preprocessing" or "postprocessing"

    Returns:

    """
    if process not in ["preprocessing", "postprocessing", "target_dataset"]:
        raise ValueError("#22 invalid argument for process in get_use_parallelization in config.py")
    if get_available_processes_for_multiprocessing(process) == 1:
        return False
    return Config.get(process, "use_parallelization") == "True"


def get_available_processes_for_multiprocessing(process: "" or "" = "preprocessing") -> int:
    """
    If false, one process is not used
    (which is recommended if you want to use the computer for other purposes simultaneously)
    Args:
        process: "preprocessing" or "postprocessing"

    Returns:

    """
    if process not in ["preprocessing", "postprocessing", "target_dataset"]:
        raise ValueError("#22 invalid argument for process in get_use_parallelization in config.py")
    n_jobs = multiprocessing.cpu_count()
    if not Config.get(process, "use_all_processes_for_parallelization").lower() == "true":
        if n_jobs >= 2:
            n_jobs -= 2
    return n_jobs


def get_execute_metric(metric: str) -> bool:
    """

    Args:
        metric: metric which has to be in metrics.

    Returns:

    """
    if metric not in metrics.keys():
        raise ValueError("#22 invalid argument for metric in get_execute_metric in config.py")
    return Config.get("execute_metrics", metric).lower() == "true"


def get_execute_preprocessing_step(step: str) -> bool:
    """

    Args:
        step: step which has to be in preprocessing_steps.

    Returns:

    """
    if step not in preprocessing_steps:
        raise ValueError(f"#22 invalid argument for step {step} in get_execute_preprocessing_step in config.py")
    return Config.get("preprocessing", step).lower() == "true"


def get_reduce_metrics_to_every_fifth_frame() -> bool:
    return Config.get("execute_metrics", "reduce_metrics_to_every_fifth_frame").lower() == "true"


def get_frames_per_second(reduced_to_every_fifth_frame: bool = False) -> int:
    """Number of frames per second in positions data"""
    frames_per_second = 25  # all this code is written for 25 hertz (except the downsampling parts)
    if reduced_to_every_fifth_frame:
        # this happens in case of downsampling
        frames_per_second = 5
    return frames_per_second


def get_min_frames_ball_in_play(reduced_to_every_fifth_frame: bool = False):
    # add one because we sometimes take the value from the frame before (for ball gains, defenders become attackers)
    return int(Config.get("target_dataset", "min_seconds_ball_in_play")) * get_frames_per_second(
        reduced_to_every_fifth_frame) + 1


def get_min_frames_no_ball_change(reduced_to_every_fifth_frame: bool = False):
    # add one because we sometimes take the value from the frame before (for ball gains, defenders become attackers)
    return int(Config.get("target_dataset", "min_seconds_no_ball_change")) * get_frames_per_second(
        reduced_to_every_fifth_frame) + 1


def get_shift_target_frames_in_seconds():
    return float(Config.get("target_dataset", "shift_target_frames_in_seconds"))
