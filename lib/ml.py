from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from lib.data_processing import ConfoundRegressor
from typing import List
import numpy as np
import shap
import pandas as pd


def get_ml_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    voice_features: List[str],
    covariats: List[str],
    SEED: int,
    ordinal_features: List[str] = [],
    ordinal_categories: List[float] = [],
) -> Pipeline:
    """
    Create a machine learning pipeline for regression tasks with preprocessing, confound regression,
    feature selection, and a Ridge regression model.

    Parameters
    ----------
    numerical_features : list of str
        List of column names corresponding to numerical features.
    categorical_features : list of str
        List of column names corresponding to categorical features.
    voice_features : list of str
        List of column names for features to be unconfounded.
    covariats : list of str
        List of column names to be used as covariats.
    SEED : int
        Random seed for reproducibility in the Ridge regression model.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Configured scikit-learn pipeline with preprocessing, confound regression,
        feature selection, and Ridge regression steps.
    """
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                    categories="auto",  # Let it infer categories from entire dataset
                ),
            )
        ]
    )

    # Preprocessing for ordinal data
    ordinal_transformer = Pipeline(
        steps=[("encoder", OrdinalEncoder(categories=ordinal_categories))]
    )
    if ordinal_features == []:
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
    else:
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("ord", ordinal_transformer, ordinal_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
    # Convert column names to indices for ConfoundRegressor
    all_features = numerical_features + categorical_features + ordinal_features
    # indices of the covariats

    covariats_indices = [all_features.index(col) for col in covariats]
    # Those features that will be unconfounded
    features_to_unconfound = [all_features.index(col) for col in voice_features]

    confound_regressor = ConfoundRegressor(
        confound_indices=covariats_indices,
        features_to_unconfound=features_to_unconfound,
    )

    feature_selector = SelectKBest(score_func=f_regression)  # K = 10 by default

    # Use Ridge for reproducibility with random_state
    model = Ridge(random_state=SEED)

    if voice_features == []:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("selector", feature_selector),
                ("model", model),
            ]
        )
        print("Don't use confound regressor, empty voice features")
    elif covariats == []:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("selector", feature_selector),
                ("model", model),
            ]
        )
        print("Don't use confound regressor, empty covars features")
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("confound_regressor", confound_regressor),
                ("selector", feature_selector),
                ("model", model),
            ]
        )

    print("Defining pipeline was successful.")

    return pipeline


def pad_shap_explanation(explanation, selected_features, all_features):
    """
    Pad all components of a SHAP explanation with zeros for unselected features.

    Parameters:
    -----------
    explanation : shap.Explanation
        SHAP explanation object from explainer()
    selected_features : list
        List of feature names that were selected in this fold
    all_features : list
        List of all possible feature names (complete feature set)

    Returns:
    --------
    padded_explanation : shap.Explanation
        SHAP explanation with padded values, base_values, and data
    """
    # Create mapping from feature name to column index
    feature_to_idx = {feature: idx for idx, feature in enumerate(all_features)}
    n_all_features = len(all_features)

    # Pad values
    orig_values = explanation.values
    if len(orig_values.shape) == 1:
        # Single output case
        padded_values = np.zeros(n_all_features)
        for i, feature in enumerate(selected_features):
            padded_values[feature_to_idx[feature]] = orig_values[i]
    else:
        # Multiple samples or outputs case
        if len(orig_values.shape) == 2:
            # 2D array (n_samples, n_features)
            padded_values = np.zeros((orig_values.shape[0], n_all_features))
            for i, feature in enumerate(selected_features):
                padded_values[:, feature_to_idx[feature]] = orig_values[:, i]
        else:
            # 3D array (n_samples, n_features, n_outputs)
            padded_values = np.zeros(
                (orig_values.shape[0], n_all_features, orig_values.shape[2])
            )
            for i, feature in enumerate(selected_features):
                padded_values[:, feature_to_idx[feature], :] = orig_values[:, i, :]

    # Pad data (feature values)
    orig_data = explanation.data
    if isinstance(orig_data, pd.DataFrame):
        padded_data = pd.DataFrame(
            np.zeros((orig_data.shape[0], n_all_features)), columns=all_features
        )
        for feature in selected_features:
            padded_data[feature] = orig_data[feature]
    else:  # numpy array
        if len(orig_data.shape) == 1:
            padded_data = np.zeros(n_all_features)
            for i, feature in enumerate(selected_features):
                padded_data[feature_to_idx[feature]] = orig_data[i]
        else:
            padded_data = np.zeros((orig_data.shape[0], n_all_features))
            for i, feature in enumerate(selected_features):
                padded_data[:, feature_to_idx[feature]] = orig_data[:, i]

    # Base values don't need padding (they're per-sample, not per-feature)
    base_values = explanation.base_values

    # Create new explanation object
    padded_explanation = shap.Explanation(
        values=padded_values,
        base_values=base_values,
        data=padded_data,
        feature_names=all_features,
    )

    return padded_explanation


def average_shap_components(shap_explanations_list):
    """
    Average SHAP components (values, base_values, data) across cross-validation folds
    when input is a list of lists containing these components.

    Parameters:
    -----------
    shap_explanations_list : list of lists
        Each inner list contains [values, base_values, data] from a fold

    Returns:
    --------
    tuple of (avg_values, avg_base_values, avg_data)
        Averaged components across all folds
    """
    if not shap_explanations_list:
        raise ValueError("Input list of SHAP explanations is empty")

    # Initialize accumulators with zeros matching first fold's shape
    sum_values = np.zeros_like(shap_explanations_list[0][0])
    sum_base_values = np.zeros_like(shap_explanations_list[0][1])
    sum_data = np.zeros_like(shap_explanations_list[0][2])

    # Sum across all folds
    for fold in shap_explanations_list:
        values, base_values, data = fold
        sum_values += values
        sum_base_values += base_values
        sum_data += data

    # Compute averages
    n_folds = len(shap_explanations_list)
    avg_values = sum_values / n_folds
    avg_base_values = sum_base_values / n_folds
    avg_data = sum_data / n_folds

    return avg_values, avg_base_values, avg_data
