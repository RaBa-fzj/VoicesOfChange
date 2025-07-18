import numpy as np
import pandas as pd
from pathlib import Path


def load_and_preprocess_data(
    data_dir: Path, file_name: str, include_voice: bool, include_puberty: bool
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list]:
    """
    Load and preprocess the dataset.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data file.
    file_name : str
        Name of the Excel file to load.
    include_voice : bool
        Whether to include voice features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    groups : pd.Series
        Group labels (pseudosic).
    confounders : list
        List of confounder feature names.
    """
    # Load all voice features to process the data and match the cohorts
    # with the cohort that has voice features
    sdq_vars, voice_features, covariates = get_feature_list(
        True, include_puberty
    )

    data = pd.read_excel(data_dir / file_name)

    # Standardize column names by replacing '.' with '__'
    data = data.rename(columns=lambda x: x.replace(".", "__"))

    # Filter to only include parent-reported data
    data = data.query("reporter == 'parent'")

    all_features = ["pseudosic"] + sdq_vars + voice_features + covariates
    # Remove rows with missing values in any selected feature
    data = data.dropna(subset=all_features)

    data = data[all_features].reset_index(drop=True)

    # Now keep the voice features or not to generate X
    sdq_vars, voice_features, covariates = get_feature_list(
        include_voice, include_puberty
    )
    X = data[voice_features + covariates].copy()
    y = data[sdq_vars[0]].copy()
    groups = data["pseudosic"].astype(str).copy()

    print("Preprocessing successful.")

    return X, y, groups, covariates


def get_feature_list(
    include_voice: bool, include_puberty: bool
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns lists of SDQ features, voice features, and covariates for data analysis.

    Parameters
    ----------
    include_voice : bool
        If True, includes voice features in the output; otherwise, the voice features list will be empty.

    Returns
    -------
    sdq_features : list of str
        List of SDQ (Strengths and Difficulties Questionnaire) feature names.
    voice_features : list of str
        List of voice feature names, or an empty list if `include_voice` is False.
    covariates : list of str
        List of covariate feature names used for analysis.
    """
    sdq_features = ["e_sdq__d00149_hyp_sum"]

    voice_features = [
        "stimme__f0_sprech_1",
        "stimme__f0_sprech_2",
        "stimme__f0_sprech_3",
        "stimme__f0_sprech_4",
        "stimme__f0_sprech_5",
        "stimme__spl_sprech_1",
        "stimme__spl_sprech_2",
        "stimme__spl_sprech_3",
        "stimme__spl_sprech_4",
        "stimme__spl_sprech_5",
        "stimme__mpt",
        "stimme__jitter",
        "stimme__dsi",
    ]

    if not include_voice:
        voice_features = []

    covariates = [
        "age",
        "sex12",
        "soz_winkler_2019__d00408_gesamt_score",
        "c_anthro_kh__d00040_bmi_sds",
    ]

    if include_puberty:
        covariates = covariates + ["c_pub_stat__d00077_pub_status"]

    return sdq_features, voice_features, covariates


def get_num_and_cat_ord_features(
    include_voice: bool, include_puberty: bool
) -> tuple[list[str], list[str], list[str]]:
    """
    Get lists of numerical and categorical feature names for preprocessing.

    Parameters
    ----------
    include_voice : bool
        If True, includes voice features in the numerical features list.

    Returns
    -------
    numerical_features : list of str
        List of numerical (and ordinal) feature names.
    categorical_features : list of str
        List of categorical feature names.
    """
    _, voice_features, _ = get_feature_list(include_voice, include_puberty)

    numerical_features = voice_features + [
        "age",
        "soz_winkler_2019__d00408_gesamt_score",
        "c_anthro_kh__d00040_bmi_sds",
    ]
    categorical_features = ["sex12"]

    ordinal_features = []
    if include_puberty:
        ordinal_features = ["c_pub_stat__d00077_pub_status"]

    # Sanity checks
    assert isinstance(numerical_features, list), "numerical_features must be a list"
    assert isinstance(categorical_features, list), "categorical_features must be a list"
    assert all(isinstance(f, str) for f in numerical_features), (
        "All numerical feature names must be strings"
    )
    assert all(isinstance(f, str) for f in categorical_features), (
        "All categorical feature names must be strings"
    )
    assert "sex12" in categorical_features, "'sex12' must be in categorical_features"
    if not include_voice:
        assert not voice_features, (
            "voice_features should be empty if include_voice is False"
        )

    return numerical_features, categorical_features, ordinal_features


def get_variable_name_mapping():
    variable_name_mapping = {
        "e_sdq__d00149_hyp_sum": "SDQ_HI",
        "num__stimme__f0_sprech_1": "f0_quiet_I",
        "num__stimme__f0_sprech_2": "f0_conversation_II",
        "num__stimme__f0_sprech_3": "f0_presentation_III",
        "num__stimme__f0_sprech_4": "f0_loud_IV",
        "num__stimme__f0_sprech_5": "f0_quiet_V",
        "num__stimme__spl_sprech_1": "spl_quiet_I",
        "num__stimme__spl_sprech_2": "spl_conversation_II",
        "num__stimme__spl_sprech_3": "spl_presentation_III",
        "num__stimme__spl_sprech_4": "spl_loud_IV",
        "num__stimme__spl_sprech_5": "spl_quiet_V",
        "num__stimme__mpt": "MPT",
        "num__stimme__jitter": "Jitter",
        "num__stimme__dsi": "DSI",
        "num__age": "Age",
        "cat__sex12_2": "Sex (male vs. female)",
        "num__soz_winkler_2019__d00408_gesamt_score": "SES",
        "ord__c_pub_stat__d00077_pub_status": "Pubertal status",
        "num__c_anthro_kh__d00040_bmi_sds": "BMI_SDS",
    }

    return variable_name_mapping


def get_used_feature_names(preprocessor) -> list[str]:
    """
    Get the feature names after preprocessing.

    Parameters
    ----------
    preprocessor : object
        The fitted preprocessor (e.g., a sklearn ColumnTransformer or Pipeline).
    numerical_features : list of str
        List of numerical feature names before preprocessing.

    Returns
    -------
    feature_names : list of str
        List of feature names after preprocessing.

    Notes
    -----
    If the preprocessor does not implement `get_feature_names_out`, returns
    the original numerical features plus 'sex_female' as a fallback.
    """

    feature_names = preprocessor.get_feature_names_out()
    assert isinstance(feature_names, (list, tuple, pd.Index, np.ndarray)), (
        "Output of get_feature_names_out must be list-like"
    )
    return list(feature_names)
