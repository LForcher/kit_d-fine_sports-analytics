from tqdm import tqdm
import os
from src.utils import utils
from src.preprocessing import preprocessing_utils
from src.preprocessing.parse_data import store_event_data_in_db, store_pitch_info, store_mat_info_links, \
    store_positions_data_in_db
from src import config
import src.utils.db_handler as db_handler
from src.preprocessing.match_positions_and_events import matching_main
from joblib import Parallel, delayed
import time
import numpy as np


def main():
    """ main script which executes all preprocessing steps that are selected in the config.ini.
    This method is executed when running the script as suggested in the readme.
    """
    skip_existing_matches = config.get_skip_existing_matches(process="preprocessing")
    matches_to_process = config.get_matches_to_process()
    if config.get_execute_preprocessing_step("parse_data"):
        print("Start data parsing")
        parse_data(matches_to_process, skip_existing_matches)
    if config.get_execute_preprocessing_step("match_positions_events"):
        print("Start matching positions and events")
        match_positions_and_events(matches_to_process, skip_existing_matches)
    if config.get_execute_preprocessing_step("rearrange_positions_by_teams"):
        print("Start rearrangement of positions by teams (only players on pitch)")
        fill_positions_by_teams_table(matches_to_process, skip_existing_matches=True)
    if config.get_execute_preprocessing_step("rearrange_positions_to_attacker_and_defenders"):
        print("Start rearrangement of positions into attackers and defenders")
        fill_positions_of_attackers_and_defenders_table(matches_to_process, skip_existing_matches=True)
    if config.get_execute_preprocessing_step("rearrange_positions_to_attacker_and_defenders_sorted_by_dist_to_ball"):
        print("Start rearrangement of positions into attackers and defenders sorted by dist to ball")
        fill_positions_of_attackers_and_defenders_dist_to_ball_table(matches_to_process, skip_existing_matches=True)


def match_positions_and_events(matches_to_process: str or list, skip_existing_matches: bool):
    """
    Creates mapping of position and event data. Processes all data available in db table event data.
    Args:
        matches_to_process: "all" or list of match ids
        skip_existing_matches: replace or skip existing matches

    Returns: nothing, data is stored into database
    """
    if isinstance(matches_to_process, str) and matches_to_process == "all":
        all_matches = utils.get_all_matches(utils.table_name_events)
    else:
        if not isinstance(matches_to_process, list):
            raise ValueError(f"# value for matches to process is not correct. given value: {matches_to_process}")
        all_matches = matches_to_process
    if skip_existing_matches and db_handler.has_table(utils.table_pos_to_event_mapping) and \
            not db_handler.is_table_empty(utils.table_pos_to_event_mapping):
        matches_in_pos_to_event_mappings = utils.get_all_matches(utils.table_pos_to_event_mapping)
        all_matches = [match for match in all_matches if match not in matches_in_pos_to_event_mappings]

    # use_parallelization: Use with caution, time advantage not proven yet
    use_parallelization = config.get_use_parallelization(process="preprocessing")
    if use_parallelization and len(all_matches) > 1:
        n_jobs = config.get_available_processes_for_multiprocessing(process="preprocessing")
        print(f"Parallelize matching of pos and events using {n_jobs} different processes.")
        print(f"(Matching starts are tracked, process needs ca. 10 - 15 min after last matching starts).")
        start_time = time.time()
        Parallel(n_jobs=n_jobs)(delayed(matching_main)(match_id) for match_id in tqdm(all_matches))
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")
    else:
        print(f"Matching of pos and events without parallelization.")
        start_time = time.time()
        for match_id in tqdm(all_matches):
            matching_main(match_id)
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")


def parse_data(matches_to_process: str or list, skip_existing_matches: bool):
    """
    Parses all data and reads it into db tables.
    The data has to be in subfolders (typically divided by gamedays) of raw_data_path and match the naming assumptions.
    Args:
        matches_to_process: "all" or list of match ids
        skip_existing_matches: replace or skip existing matches

    Returns: nothing, data is stored into database
    """
    for folder in tqdm(os.listdir(utils.raw_data_path), "Number of folders processed..."):
        full_path = os.path.join(utils.raw_data_path, folder)
        print(f"Processing data in path: {full_path}")
        if os.path.isdir(full_path):
            for file in os.listdir(full_path):
                file = str(file)
                if ".zip" in file:
                    continue
                str_match_id = file[-10:-4]
                if str_match_id not in matches_to_process and matches_to_process != "all":
                    continue
                if "event" in file.lower():
                    event_file = os.path.join(full_path, file)
                    store_event_data_in_db(event_file, str_match_id, skip_existing_matches)
                elif "position" in file.lower():
                    tracking_file = os.path.join(full_path, file)
                    file_match_info = os.path.join(
                        full_path,
                        f"DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-{tracking_file[-10:-4]}.xml")
                    store_positions_data_in_db(tracking_file, file_match_info, str_match_id, skip_existing_matches)
                elif "matchinformation" in file.lower():
                    pitch_file = os.path.join(full_path, file)
                    store_pitch_info(pitch_file, str_match_id, skip_existing_matches)
                    store_mat_info_links(pitch_file, str_match_id, skip_existing_matches)


def fill_positions_of_attackers_and_defenders_table(matches_to_process: str or list, skip_existing_matches: bool):
    """
    Restructures the position data such that there are 22 columns containing attacker positions (x and y) and
    22 columns containing defender positions.
    Pre-condition: Table table_name_positions_of_attackers_and_defenders has to be filled
     (see previous preprocessing step)

    Args:
        matches_to_process: "all" or list of match ids
        skip_existing_matches: replace or skip existing matches

    Returns: nothing, data is stored into database
    """
    if not skip_existing_matches:
        raise not NotImplementedError("at the moment existing matches have to be skipped.")
    if isinstance(matches_to_process, str) and matches_to_process == "all":
        all_matches = utils.get_all_matches(utils.table_name_events)
    else:
        if not isinstance(matches_to_process, list):
            raise ValueError(f"# value for matches to process is not correct. given value: {matches_to_process}")
        all_matches = matches_to_process
    if skip_existing_matches and db_handler.has_table(utils.table_name_positions_of_attackers_and_defenders) and \
            not db_handler.is_table_empty(utils.table_name_positions_of_attackers_and_defenders):
        existing_matches = utils.get_all_matches(utils.table_name_positions_of_attackers_and_defenders)
        all_matches = [match for match in all_matches if match not in existing_matches]

    # use_parallelization: Use with caution, time advantage not proven yet
    use_parallelization = config.get_use_parallelization(process="preprocessing")
    if use_parallelization and len(all_matches) > 1:
        n_jobs = config.get_available_processes_for_multiprocessing(process="preprocessing")
        print(f"Parallelize rearrangement of positions into attackers and defenders into {n_jobs} different processes.")
        print(f"(Starts are tracked, process needs ca. 10 - 15 min after last start).")
        start_time = time.time()
        Parallel(n_jobs=n_jobs)(
            delayed(preprocessing_utils.permutate_positions_into_attacker_and_defender_cols)(match_id) for match_id in
            tqdm(all_matches))
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")
    else:
        print(f"Rearrangement of positions into defenders and attackers without parallelization.")
        start_time = time.time()
        for match_id in tqdm(all_matches):
            preprocessing_utils.permutate_positions_into_attacker_and_defender_cols(match_id)
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")


def fill_positions_of_attackers_and_defenders_dist_to_ball_table(matches_to_process: str or list,
                                                                 skip_existing_matches: bool):
    """
    Restructures the position data such that there are 22 columns containing attacker positions (x and y) and
    22 columns containing defender positions and on top the players of each team are sorted by the distance to the ball
    in the respective frame.
    Pre-condition: Table table_name_positions_players_on_pitch has to be filled (see previous preprocessing step)

    Args:
        matches_to_process: "all" or list of match ids
        skip_existing_matches: replace or skip existing matches

    Returns: nothing, data is stored into database
    """
    if not skip_existing_matches:
        raise not NotImplementedError("at the moment existing matches have to be skipped.")
    if isinstance(matches_to_process, str) and matches_to_process == "all":
        all_matches = utils.get_all_matches(utils.table_name_events)
    else:
        if not isinstance(matches_to_process, list):
            raise ValueError(f"# value for matches to process is not correct. given value: {matches_to_process}")
        all_matches = matches_to_process
    if skip_existing_matches and db_handler.has_table(
            utils.table_name_positions_of_attackers_and_defenders_dist_to_ball) and \
            not db_handler.is_table_empty(utils.table_name_positions_of_attackers_and_defenders_dist_to_ball):
        existing_matches = utils.get_all_matches(utils.table_name_positions_of_attackers_and_defenders_dist_to_ball)
        all_matches = [match for match in all_matches if match not in existing_matches]

    # use_parallelization: Use with caution, time advantage not proven yet
    use_parallelization = config.get_use_parallelization(process="preprocessing")
    if use_parallelization and len(all_matches) > 1:
        n_jobs = config.get_available_processes_for_multiprocessing(process="preprocessing")
        print(f"Parallelize rearrangement of positions into attackers and defenders into {n_jobs} different processes.")
        print(f"(Starts are tracked, process needs ca. 10 - 15 min after last start).")
        start_time = time.time()
        Parallel(n_jobs=n_jobs)(
            delayed(preprocessing_utils.permutate_attacker_and_defender_cols_by_distance_to_ball)(match_id) for
            match_id in
            tqdm(all_matches))
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")
    else:
        print(f"Rearrangement of positions into defenders and attackers without parallelization.")
        start_time = time.time()
        for match_id in tqdm(all_matches):
            preprocessing_utils.permutate_attacker_and_defender_cols_by_distance_to_ball(match_id)
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")


def fill_positions_by_teams_table(matches_to_process: str or list, skip_existing_matches: bool):
    """
    Restructures the position data such that there are only 44 columns containing the players on the pitch
    (x and y data). I.e. there are no nan values in the columns for players on the bench (except for situations
    where players have seen red cards).
    Pre-condition: Table table_name_positions has to be filled (see previous preprocessing step)

    Args:
        matches_to_process: "all" or list of match ids
        skip_existing_matches: replace or skip existing matches

    Returns: nothing, data is stored into database
    """
    if not skip_existing_matches:
        raise not NotImplementedError("at the moment existing matches have to be skipped.")
    if isinstance(matches_to_process, str) and matches_to_process == "all":
        all_matches = utils.get_all_matches(utils.table_name_events)
    else:
        if not isinstance(matches_to_process, list):
            raise ValueError(f"# value for matches to process is not correct. given value: {matches_to_process}")
        all_matches = matches_to_process
    if skip_existing_matches and db_handler.has_table(utils.table_name_positions_players_on_pitch) and \
            not db_handler.is_table_empty(utils.table_name_positions_players_on_pitch):
        existing_matches = utils.get_all_matches(utils.table_name_positions_players_on_pitch)
        all_matches = [match for match in all_matches if match not in existing_matches]

    # use_parallelization: Use with caution, time advantage not proven yet
    use_parallelization = config.get_use_parallelization(process="preprocessing")
    if use_parallelization and len(all_matches) > 1:
        n_jobs = config.get_available_processes_for_multiprocessing(process="preprocessing")
        print(f"Parallelize rearrangement of positions by teams into {n_jobs} different processes.")
        print(f"(Starts are tracked, process needs ca. 10 - 15 min after last start).")
        start_time = time.time()
        Parallel(n_jobs=n_jobs)(
            delayed(preprocessing_utils.sort_positions_by_teams)(match_id) for match_id in
            tqdm(all_matches))
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")
    else:
        print(f"Rearrangement of positions by teams without parallelization.")
        start_time = time.time()
        for match_id in tqdm(all_matches):
            preprocessing_utils.sort_positions_by_teams(match_id)
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")


if __name__ == "__main__":
    main()
