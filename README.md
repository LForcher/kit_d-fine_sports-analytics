
# Tactical Analysis of Defensive Play using Tracking Data and Explainable AI
- Code package used for the investigation in “Prediction of defensive success in elite soccer using machine learning - Tactical analysis of defensive play using tracking data and explainable AI”
- A project to develop advanced models based on DFL positional tracking data for tactical insights
- developed by KIT and d-fine
# Reproduction of this Work
## Data Availability Disclaimer
- The used data is property of the German Football League (Deutsche Fußball Liga, DFL) and is not publicly available. The authors do not have permission to share the data publicly
## floodlight Package
- This work can be reproduced using similar data from professional soccer (e.g. tracking data of other soccer leagues or data providers)
- To process the tracking data in this project, the floodlight package was used (a high-level data-driven sports analytics framework) (Raabe et al. 2022)
- This python package can be used to process similar match data to make use of the developed code of this investigation

# Executing the Code
## Configure Project
1. Create virtualenv venv from requirements.txt
2. Fix floodlight script: 
Replace venv/Lib/site-packages/floodlight/io/dfl.py by docs/dfl.py (important! fixes some bugs and ensures gathering of additional information such as the dfl event id)
3. Install postgres and adapt .env if necessary - (you can use dbeaver as gui for postgres)
4. Put data (positions, events, match_information) into sports-analytics\data\raw_data\xml_data_folder
5. Important! Always execute python scripts with working directory sports-analytics
6. (optional) adapt src/config.ini

## Preprocessing: parse_data
- Run src/preprocessing/execute_preprocessing.py

- This step takes the given raw data from the xml and stores it into postgres databases. The data structure is similar to the structure given by the package floodlight, which was used to read the data from the xml files. Except for some restructurings, no calculations are made on the data in this step.

## Postprocessing: execute_metrics
- Run src/postprocessing/execute_metrics.py

- All the (base) metrics are calculated in this step. All scripts with metric calculations are stored in the folder src/metrics.

## Create Target Dataset
- Run src/postprocessing/create_target_dataset.py

- Calculates the final feature dataset use raw data and the calculated metrics. All timely and spatial calculations/aggregations are made in this step. 

## Prediction
- Run src/postprocessing/ball_gain_prediction.py
- or use method main() from that file through notebooks (see paper_notebook and paper_notebook_lanes)

- In this step the predictions are made based on the target dataset that was created one step before. Then using the predictors, the shapely values are calculated to explain the model and derive practical implications. Instead of running the scripts directly, it is recommended to look into the provided notebooks to see the results and how they were created.
