import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import os

from src.utils import db_handler, utils
from src.models import feature_selection
import optuna
from typing import Union

import sklearn

from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, recall_score, precision_score, \
    RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, roc_auc_score

from src.postprocessing.create_target_dataset import remove_nan_values


def main(clf: str = "XGBoost", use_k_best_features=None, corr_threshold=0.5,
         params=None, name_target_set="target_dataset_shift_2sec_hypothesis_testing",
         target_data_columns: list or None = None, evaluation_set="validation"):
    """
    main script which executes the ball gain prediction as suggested in the README.
    Args:
        clf: which classifier to take, should be "RandomForst", "XGBoost" or "Linear"
        use_k_best_features: option to load k_best_features -> those have to be saved before (see feature_selection)
        corr_threshold: remove features by corr threshold
        params: if given, params for classifier. You can give all, partial or no params.
            Check sklearn or code for details
        name_target_set: which table to load from database
        target_data_columns: optional, if given, target_dataset is limited to given columns
        evaluation_set: evaluate clf on "validation" or "test" set (both 20% of available data)

    Returns: feature importance and classifier. Additionally prints and plots evaluation of the classifier!

    """
    if params is not None:
        print(f"Predicting with {clf} and params {params}")
    if "n_estimators" in params:
        n_estimators = params["n_estimators"]
    else:
        n_estimators = 250
    if "max_features" in params:
        max_features = params["max_features"]
    else:
        max_features = "sqrt"
    if "max_depth" in params:
        max_depth = params["max_depth"]
    else:
        max_depth = 5
    if "min_samples_split" in params:
        min_samples_split = params["min_samples_split"]
    else:
        min_samples_split = 5
    if "min_samples_leaf" in params:
        min_samples_leaf = params["min_samples_leaf"]
    else:
        min_samples_leaf = 5
    if "bootstrap" in params:
        bootstrap = params["bootstrap"]
    else:
        bootstrap = True
    if "class_weight" in params:
        class_weight = params["class_weight"]
    else:
        class_weight = "balanced_subsample"
    if "max_leaves" in params:
        max_leaves = params["max_leaves"]
    else:
        max_leaves = 0
    if "max_bin" in params:
        max_bin = params["max_bin"]
    else:
        max_bin = 5
    if "min_child_weight" in params:
        min_child_weight = params["min_child_weight"]
    else:
        min_child_weight = 3
    if "grow_policy" in params:
        grow_policy = params["grow_policy"]
    else:
        grow_policy = "depthwise"
    if "max_iter" in params:
        max_iter = params["max_iter"]
    else:
        max_iter = 25

    target_dataset = db_handler.get_table(name_target_set)

    # reduce to given columns (if given)
    if target_data_columns is not None:
        target_dataset = target_dataset[target_data_columns]
    # whether to use this feature selection method, see get_best_features_per_group
    if use_k_best_features:
        features_with_time_shift, features_no_time_shift = get_best_features_per_group(name_target_set)
        key_columns = ["match_id", "half", "frame", "target"]
        new_cols = [col for col in target_dataset.columns
                    if col[:63] in features_with_time_shift
                    or col in features_no_time_shift
                    or col in key_columns]
        target_dataset = target_dataset[new_cols]
    if "linear" in clf.lower():
        # classification with linear model
        clf = sklearn.linear_model.SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=max_iter)
        scaler = sklearn.preprocessing.StandardScaler()
        data = split_train_test_validation(target_dataset)
        scaler.fit(data["x_train"])
        x_train_vals = scaler.transform(data["x_train"])
        data["x_train"] = pd.DataFrame(x_train_vals, columns=data["x_train"].columns)
        x_validation_vals = scaler.transform(data["x_validation"])
        data["x_validation"] = pd.DataFrame(x_validation_vals, columns=data["x_validation"].columns)
        x_test_vals = scaler.transform(data["x_test"])
        data["x_test"] = pd.DataFrame(x_test_vals, columns=data["x_test"].columns)
    elif clf == "RandomForest":
        if corr_threshold < 1:
            new_cols = feature_selection.get_uncorrelated_features(target_dataset, threshold=corr_threshold)
            target_dataset = target_dataset[new_cols]
        data = split_train_test_validation(target_dataset)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     bootstrap=bootstrap,
                                     class_weight=class_weight)
    elif clf == "XGBoost":
        if corr_threshold < 1:
            new_cols = feature_selection.get_uncorrelated_features(target_dataset, threshold=corr_threshold)
            target_dataset = target_dataset[new_cols]
        data = split_train_test_validation(target_dataset)
        clf = xgboost.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, max_leaves=max_leaves,
                                    max_bin=max_bin, grow_policy=grow_policy, min_child_weight=min_child_weight)
    else:
        raise ValueError("# Invalid option chosen for clf!")
    clf.fit(data["x_train"].values, data["y_train"].values)

    if evaluation_set == "test":
        x_test = data["x_test"]
        y_test = data["y_test"]
    elif evaluation_set == "validation":
        x_test = data["x_validation"]
        y_test = data["y_validation"]
    else:
        raise NotImplementedError()
    y_pred = clf.predict(x_test.values)

    y_pred = pd.Series([x == 1 for x in y_pred], name="y_pred")

    feat_import_a = evaluate(clf, data["x_train"], data["y_train"], x_test, y_test, y_pred)

    return feat_import_a, clf


def evaluate(clf, x_train, y_train, x_test, y_test, y_pred) -> pd.Series:
    """
    Evaluation of the classifier on the given data. Lots of print messages and plots!
    y_pred can be given (for better performance), but does not have to be given. None is accepted.
    Args:
        clf: classifier from sklearn
        x_train: training data (features)
        y_train: training data (target)
        x_test: validation or test data (depending on which to evaluate) (features)
        y_test: validation or test data (depending on which to evaluate) (target)
        y_pred: predictions or None

    Returns: series with feature importance

    """
    if y_pred is None:
        y_pred = clf.predict(x_test.values)
        y_pred = pd.Series([x == 1 for x in y_pred], name="y_pred")
    print("Score training: %0.4f" % accuracy_score(y_train.values, clf.predict(x_train.values)))
    print(f"f1_score training: {f1_score(y_train.values, clf.predict(x_train.values))}")
    print(f"precision training: {precision_score(y_train.values, clf.predict(x_train.values))}")
    print(f"recall training: {recall_score(y_train.values, clf.predict(x_train.values))}")

    print("Score test: %0.4f" % accuracy_score(y_test.values, y_pred.values))
    print(f"f1_score: {f1_score(y_test.values, y_pred.values)}")
    print(f"precision: {precision_score(y_test.values, y_pred.values)}")
    print(f"recall: {recall_score(y_test.values, y_pred.values)}")

    if hasattr(clf, "steps"):
        classifier = clf.steps[-1][1]
    else:
        classifier = clf
    feature_names = x_train.columns.tolist()
    if hasattr(classifier, "feature_importances_"):
        feature_importances = classifier.feature_importances_
    else:
        feature_importances = classifier.coef_[0]

    plot_feature_importances(feature_names, feature_importances, 15)
    manual_plot_roc_curve(clf, x_test, y_test)
    plot_confusion_matrix(y_test.values * 1, y_pred.values)
    feature_series = pd.Series(feature_importances, index=feature_names, name="feature_importance")
    feature_series.sort_values(inplace=True, ascending=False)
    return feature_series


def plot_feature_importances(feature_names, feature_importances, n_features_to_plot: int):
    _fig = plt.Figure()
    plt.xticks(rotation=45)
    plt.tick_params(axis='x', labelsize=4)

    feat = pd.Series(feature_importances, index=feature_names)
    feat.sort_values(ascending=False, inplace=True)
    feat = feat.iloc[:n_features_to_plot]
    plt.bar(feat.index, feat.values)
    plt.show()
    print(feat)


def manual_plot_roc_curve(clf, x_test, y_test):
    _fig = plt.Figure()
    rfc_y_pred_proba = clf.predict_proba(x_test.values)
    rfc_y_pred_proba_positive = rfc_y_pred_proba[:, 1]
    RocCurveDisplay.from_predictions(y_test.values * 1, rfc_y_pred_proba_positive)
    plt.plot([0, 1], [0, 1], color='navy')
    plt.show()
    _fig = plt.Figure()


def features_best_predictions_per_group(target_set="target_dataset_shift_2sec_hypothesis_testing",
                                        features_per_group=5, use_f_score: bool = False):
    """documentation: see get_best_features_per_group"""
    target_dataset = db_handler.get_table(target_set)
    target_dataset = remove_nan_values(target_dataset)
    data = split_train_test_validation(target_dataset)

    features = target_dataset.columns.tolist()
    key_columns = ["match_id", "half", "frame"]
    for key in key_columns:
        features.remove(key)
    features.remove("target")
    features_with_time_shift = [feat for feat in features if "sec" in feat]
    features_no_time_shift = [feat for feat in features if feat not in features_with_time_shift]
    feature_groups = set([get_feature_name_without_time_info(feat) for feat in features_with_time_shift])

    chosen_features = {"features_no_time_shift": features_no_time_shift}

    result_list = list()
    for group in feature_groups:
        features = [feat for feat in features_with_time_shift if feat.startswith(group)]
        df_training = data["x_train"][features]
        df_validation = data["x_validation"][features]
        group_features, results = forward_feat_selection(df_training, data["y_train"], df_validation,
                                                         data["y_validation"],
                                                         key_columns=key_columns, n_features=features_per_group,
                                                         use_f_score=use_f_score)
        chosen_features[group] = group_features
        results["feature_group"] = group
        result_list.append(results)
    all_results = pd.concat(result_list)
    return chosen_features, all_results


def store_features_per_group(target_set="target_dataset_shift_2sec_hypothesis_testing", features_per_group=5,
                             use_f_score=False):
    """documentation: see get_best_features_per_group"""
    chosen_features, all_results = features_best_predictions_per_group(target_set, features_per_group,
                                                                       use_f_score=use_f_score)
    features_no_time_shift = chosen_features.pop("features_no_time_shift")
    df = pd.DataFrame(chosen_features)
    df["features_no_time_shift"] = ", ".join(features_no_time_shift)
    df.to_excel(feature_selection.path + target_set + "_features_k_best.xlsx")
    if use_f_score:
        all_results.sort_values("f_score", ascending=False, inplace=True)
    else:
        all_results.sort_values("auc_score", ascending=False, inplace=True)
    best_feat = all_results.groupby(["feature_group", "round"], as_index=False).head(1).copy()
    best_feat.rename(
        columns={"feature": "selected_feature", "f_score": "selected_f_score", "auc_score": "selected_auc_score"},
        inplace=True)
    final_results = all_results.merge(best_feat, on=["feature_group", "round"], validate="many_to_one", how="left")
    final_results["selected"] = np.where(final_results["feature"] == final_results["selected_feature"], "X", "")
    final_results.to_excel(feature_selection.path + target_set + "_features_all_results.xlsx")


def get_best_features_per_group(target_set, max_features_per_group=3):
    """
    Reduce features to x features per group.
    A group indicates here all the features belonging to the same base metric,
    e.g. the same base metric for several different players or for different time stamps.

    Note: Those features are usually highly correlated, therefore it is recommended to use only few features per group
    and remove correlated features.

    Stores features to file so that they only have to be calculated once. If they shall be calculated again,
    remove file.
    Args:
        target_set: name of table in database
        max_features_per_group: number of features per group

    Returns: list of features

    """
    file = feature_selection.path + target_set + "_features_k_best.xlsx"
    if not os.path.isfile(file):
        store_features_per_group(target_set, features_per_group=max_features_per_group + 1)
    df = pd.read_excel(file, sheet_name=0, index_col=0)
    features_no_time_shift = df["features_no_time_shift"].iloc[0].split(", ")
    df.drop(columns="features_no_time_shift", inplace=True)
    df = df.iloc[:max_features_per_group]
    features_with_time_shift = df.values.flatten().tolist()
    return features_with_time_shift, features_no_time_shift


def forward_feat_selection(x_train: pd.DataFrame, y_train: pd.Series, x_validation: pd.DataFrame,
                           y_validation: pd.Series, key_columns: list, n_features: int = 3,
                           use_f_score: bool = False):
    """
    Forward selection implemented for this case.
    Args:
        x_train: training data (key_columns + features)
        y_train: training data (target)
        x_validation: validation data (key_columns + features)
        y_validation: validation data (target)
        key_columns: list of identifier columns
        n_features: number features to choose
        use_f_score: whether to use f_score or accuracy for evaluation

    Returns: list of chosen features, dataframe with all results

    """
    n_estimators = 200
    max_features = "sqrt"
    max_depth = n_features * 2
    min_samples_split = 5
    min_samples_leaf = 10
    bootstrap = False
    class_weight = "balanced"
    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 bootstrap=bootstrap,
                                 class_weight=class_weight)

    chosen_features = list()
    remaining_features = [col for col in x_train.columns if col not in key_columns]
    print(f"starting forward feat selection for {remaining_features} with Random Forest, standard params.")
    result_list = list()
    type_score = None
    for idx in range(n_features):
        result = {"round": idx}
        feat_performance = list()
        for feature in remaining_features:
            tested_features = chosen_features + [feature]
            clf.fit(x_train[tested_features].values, y_train.values)

            y_pred = clf.predict(x_validation[tested_features].values)

            y_pred = pd.Series([x == 1 for x in y_pred], name="y_pred")

            f_score = f1_score(y_validation.values, y_pred.values)
            probabilities = clf.predict_proba(x_validation[tested_features].values)
            prob_for_class_1 = probabilities[:, 1]
            auc_score = roc_auc_score(y_validation.values, prob_for_class_1)
            if use_f_score:
                feat_performance.append(f_score)
                type_score = "f_score"
            else:

                feat_performance.append(auc_score)
                type_score = "auc_score"
            result["feature"] = feature
            result["f_score"] = f_score
            result["auc_score"] = auc_score
            result_list.append(result.copy())
        best_feature = remaining_features[np.argmax(feat_performance)]
        chosen_features.append(best_feature)
        remaining_features.remove(best_feature)
        if type_score is None:
            raise ValueError("No remaining features!")
        print(f"Round {idx}: chosen feature is {best_feature} with {type_score} of {max(feat_performance)}")
    all_results = pd.DataFrame(result_list)
    return chosen_features, all_results


def get_feature_name_without_time_info(feature: str) -> str:
    feat_parts = feature.split("_")
    feat_name = "_".join(feat_parts[:-5])
    return feat_name


def plot_confusion_matrix(y_test_values, y_pred_values):
    cfm = confusion_matrix(y_test_values, y_pred_values)
    print(cfm)
    disp = ConfusionMatrixDisplay(cfm)
    disp.plot()
    plt.show()


def split_train_test_validation(target_dataset: pd.DataFrame, validation_size: float = 0.2,
                                test_size: float = 0.2) -> dict:
    """
    Splits data into train, test and validation set. The split validates that samples from the same match always
    end up in the same dataset such that data leakage can not be an issue.
    Args:
        target_dataset: df
        validation_size: proportion of validation set
        test_size: proportion of validation set

    Returns: dictionary containing "x_train",   "y_train",  "x_validation", "y_validation", "x_test", "y_test"

    """
    all_matches = target_dataset["match_id"].unique().tolist()

    test_matches = all_matches[:int(len(all_matches) * test_size)]
    validation_matches = all_matches[len(all_matches) - int(len(all_matches) * validation_size):]
    training_matches = [match for match in all_matches if match not in validation_matches and match not in test_matches]
    if len(test_matches + validation_matches + training_matches) != len(
            set(test_matches + validation_matches + training_matches)):
        raise ValueError("#89 Splitting into training, test and validation went wrong!")
    validation_set = target_dataset[target_dataset["match_id"].isin(validation_matches)].copy()
    training_set = target_dataset[target_dataset["match_id"].isin(training_matches)].copy()
    test_set = target_dataset[target_dataset["match_id"].isin(test_matches)].copy()

    print(f"Training matches are: " + ", ".join(training_set["match_id"].unique()))
    print(f"Validation matches are: " + ", ".join(validation_set["match_id"].unique()))
    print(f"Test matches are: " + ", ".join(test_set["match_id"].unique()))
    training_set = training_set.drop(columns=["match_id", "half", "frame"], inplace=False)
    validation_set = validation_set.drop(columns=["match_id", "half", "frame"], inplace=False)
    test_set = test_set.drop(columns=["match_id", "half", "frame"], inplace=False)

    x_train = training_set.drop(columns="target", inplace=False)
    y_train = training_set["target"].copy()
    x_validation = validation_set.drop(columns="target", inplace=False)
    y_validaton = validation_set["target"].copy()
    x_test = test_set.drop(columns="target", inplace=False)
    y_test = test_set["target"].copy()
    print(f"Training Set: {training_set.shape}")
    print(f"Training pos vs neg: {training_set['target'].sum()} vs {(1 - training_set['target']).sum()}")
    print(f"Training negative: {round((1 - training_set['target']).sum() / training_set.shape[0], 4)} %")
    print(f"Validation Set: {validation_set.shape}")
    print(f"Validation pos vs neg: {validation_set['target'].sum()} vs {(1 - validation_set['target']).sum()}")
    print(f"Validation negative: {round((1 - validation_set['target']).sum() / validation_set.shape[0], 4)} %")
    print(f"Test Set: {test_set.shape}")
    print(f"Test pos vs neg: {test_set['target'].sum()} vs {(1 - test_set['target']).sum()}")
    print(f"Test negative: {round((1 - test_set['target']).sum() / test_set.shape[0], 4)} %")
    print("-----------------------------------------------------------------------------------")
    return {"x_train": x_train,
            "y_train": y_train,
            "x_validation": x_validation,
            "y_validation": y_validaton,
            "x_test": x_test,
            "y_test": y_test}


def hyperparameter_finder(name_target_set_or_shift_seconds: Union[float, str] = 0, n_trials: int = 20,
                          corr_threshold: float = 0.6, use_f_score=False,
                          classifier: str = "RandomForest", inv_training_test=False,
                          use_stored_k_best: Union[None, int] = None, target_data_columns: None or list = None,
                          hyperfinder_method="optuna"):
    """finds the best hyperparameters and prints results."""
    if isinstance(name_target_set_or_shift_seconds, float):
        table_name_target_dataset = utils.get_table_name_target_dataset(name_target_set_or_shift_seconds)
        t_d = db_handler.get_table(table_name_target_dataset)
    else:
        table_name_target_dataset = name_target_set_or_shift_seconds
        t_d = db_handler.get_table(table_name_target_dataset)
    t_d = remove_nan_values(t_d)
    if target_data_columns is not None:
        t_d = t_d[target_data_columns]
    if use_stored_k_best:
        features_with_time_shift, features_no_time_shift = get_best_features_per_group(name_target_set_or_shift_seconds)
        key_columns = ["match_id", "half", "frame", "target"]
        new_cols = [col for col in t_d.columns
                    if col[:63] in features_with_time_shift
                    or col in features_no_time_shift
                    or col in key_columns]
        t_d = t_d[new_cols]
    if corr_threshold < 1:
        new_cols = feature_selection.get_uncorr_features_from_file(corr_threshold, table_name_target_dataset)
        t_d = t_d[new_cols]

    t_t_v_values = split_train_test_validation(t_d)

    global_x_train = t_t_v_values["x_train"]
    global_y_train = t_t_v_values["y_train"]
    global_x_validation = t_t_v_values["x_validation"]
    global_y_validaton = t_t_v_values["y_validation"]

    def objective(trial):
        max_features = trial.suggest_categorical("max_features", ['sqrt'])
        max_depth = trial.suggest_int("max_depth", 2, 20)

        if classifier == "RandomForest":
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])
            n_estimators = trial.suggest_int("n_estimators", 100, 400)  # number of trees in the random forest
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])  # method used to sample data points
            clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         bootstrap=bootstrap, class_weight=class_weight)
        elif classifier == "DecisionTree":
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
            clf = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf,
                                                      max_features=max_features, class_weight=class_weight)
        elif classifier == "XGBoost":
            n_estimators = trial.suggest_int("n_estimators", 100, 500, 10)  # number of trees in the random forest
            max_leaves = trial.suggest_int("max_leaves", 0, 2000)  # 0 indicates no limit
            max_bin = trial.suggest_int("max_bin", 2, 10)
            grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            min_child_weight = trial.suggest_int("child_weight", 1, 6)
            clf = xgboost.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, max_leaves=max_leaves,
                                        max_bin=max_bin, grow_policy=grow_policy, min_child_weight=min_child_weight)
        else:
            raise ValueError(f"Classifier {classifier} not implemented!")
        clf.fit(global_x_train.values, global_y_train.values)
        y_pred = clf.predict(global_x_validation.values)
        f1_accuracy = f1_score(global_y_validaton.values, y_pred)
        y_pred_proba = clf.predict_proba(global_x_validation.values)[:, 1]
        auc_score = roc_auc_score(global_y_validaton.values, y_pred_proba)
        if inv_training_test:
            y_pred_train = clf.predict(global_x_train.values)
            training_score = accuracy_score(global_y_train.values, y_pred_train)
            print(f"Trial number {trial.number}")
            print(f"Training score is: {training_score}")
            test_score = accuracy_score(global_y_validaton.values, y_pred)
            print(f"Test score is: {test_score}")
            f1_train = f1_score(global_y_train.values, y_pred_train)
            print(f"Training f1_score is: {f1_train}")
            print(f"Test f1_score is: {f1_accuracy}")
            y_pred_proba_train = clf.predict_proba(global_x_train.values)[:, 1]
            auc_score_train = roc_auc_score(global_y_train.values, y_pred_proba_train)
            print(f"Training auc_score is: {auc_score_train}")
            print(f"Test auc_score is: {auc_score}")
        if use_f_score:
            return f1_accuracy
        else:
            return auc_score

    if hyperfinder_method == "optuna":
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print(study.best_trial)
        # best parameter combination
        print(study.best_params)

        # score achieved with best parameter combination
        print(study.best_value)
        return study
    elif hyperfinder_method == "halving_grid_search_sv":
        param_grid = {"max_features": ["sqrt"],
                      "max_depth": range(2, 20),
                      }
        if classifier == "RandomForest":
            param_grid["min_samples_split"] = range(2, 10)
            param_grid["min_samples_leaf"] = range(1, 10)
            param_grid["class_weight"] = [None, "balanced", "balanced_subsample"]
            param_grid["n_estimators"] = range(100, 400)  # number of trees in the random forest
            param_grid["bootstrap"] = [True, False]  # method used to sample data points
            clf = RandomForestClassifier()
        elif classifier == "DecisionTree":
            param_grid["min_samples_split"] = range(2, 10)
            param_grid["min_samples_leaf"] = range(1, 10)
            param_grid["class_weight"] = [None, "balanced", "balanced_subsample"]
            clf = sklearn.tree.DecisionTreeClassifier()
        elif classifier == "XGBoost":
            param_grid["n_estimators"] = range(100, 400)  # number of trees in the random forest
            param_grid["max_leaves"] = range(0, 2000)  # 0 indicates no limit
            param_grid["max_bin"] = range(2, 10)
            param_grid["grow_policy"] = ["depthwise", "lossguide"]
            param_grid["min_child_weight"] = range(1, 6)
            clf = xgboost.XGBClassifier()
        else:
            raise ValueError(f"Classifier {classifier} not implemented!")

        final_clf = HalvingGridSearchCV(clf, param_grid)
        final_clf.fit(pd.concat((global_x_train, global_x_validation)), pd.concat((global_y_train, global_y_validaton)))
        return final_clf
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    res = main()
    print(res)
