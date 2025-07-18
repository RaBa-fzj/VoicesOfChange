# %%
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.model_selection import permutation_test_score
import pandas as pd

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

project_dir = Path().resolve().parents[1]
sys.path.append(str(project_dir / "code/"))

# Project-designed functions
from lib.data_loading import (  # noqa
    load_and_preprocess_data,
    get_num_and_cat_ord_features,
    get_feature_list,
)
from lib.ml import get_ml_pipeline  # noqa
from lib.utils import ensure_dir  # noqa


# %% ############################# Variables and data
data_dir = project_dir / "data/"
file_name = "Paper_PV0775_SDQ_merged_dataset_whole.xlsx"
# Add an experiment name to add in the last part of the
# saved files
exp_name = "include_voice_sex_stratified"

save_dir = project_dir / "output" / (exp_name + "/")
ensure_dir(save_dir)

# include the voice features or not
include_voice = True

# Include the puberty for the "Robustnes" analysis
# If true, several patients will be drop for the lack of data
# and a smaller cohort is generated
include_puberty = False

# Define parameter grid for hyperparameter tuning
param_grid = {
    "selector__k": [5, 10, "all"],  # Number of features to select
    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
}
# For Hyperparameter tuning
n_inner_folds = 3
# For outer loop
n_outer_folds = 5
#######################################################
# %% Data loading
#

# Data is loaded and pre process
X, y, groups, covariats = load_and_preprocess_data(
    data_dir=data_dir,
    file_name=file_name,
    include_voice=include_voice,
    include_puberty=include_puberty,
)
print(f"Number of features: {X.shape[1]}")
print(f"Number of patients: {X.shape[0]}")
print(f"Number of unique patients: {groups.nunique()}")

# %%
# Get features for preprocessing
numerical_features, categorical_features, ordinal_features = (
    get_num_and_cat_ord_features(
        include_voice=include_voice, include_puberty=include_puberty
    )
)

_, voice_features, _ = get_feature_list(
    include_voice=include_voice, include_puberty=include_puberty
)

# For the stratified analysis, don't use sex in the pipeline
covariats.remove("sex12")
categorical_features.remove("sex12")

sex_list = ["male", "female"]

X_original = X.copy()
y_original = y.copy()
groups_original = groups.copy()

for sex in sex_list:
    print(f"\nRunning permutation test for: {sex}")

    if sex == "male":
        X_sex = X_original[X_original["sex12"] == 1].drop(columns="sex12")
    else:
        X_sex = X_original[X_original["sex12"] == 2].drop(columns="sex12")
        
    y_sex = y_original.loc[X_sex.index]
    groups_sex = groups_original.loc[X_sex.index]

    # Get the ML pipeline
    pipeline = get_ml_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        voice_features=voice_features,
        covariats=covariats,
        SEED=SEED,
        ordinal_features=ordinal_features,
    )

    # No need for a special function
    n_permutations = 1000
    inner_cv = GroupKFold(n_splits=n_inner_folds)

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv.split(X_sex, y_sex, groups=groups_sex),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,  # Fixed jobs to align with main model analysis
    )
    grid_search.fit(X_sex, y_sex, groups=groups_sex)  # Train GridSearchCV

    # Get the best estimator from GridSearchCV
    # permutation test only analyses a fix model
    best_estimator = grid_search.best_estimator_

    outer_cv = GroupKFold(n_splits=n_outer_folds)

    # Perform permutation test using the best estimator
    score, perm_scores, p_value = permutation_test_score(
        best_estimator,
        X_sex,
        y_sex,
        scoring="neg_mean_absolute_error",
        cv=outer_cv.split(X_sex, y_sex, groups=groups_sex),  # Use the provided outer_cv
        n_permutations=n_permutations,
        n_jobs=-1,  # Fixed jobs to align with main model analysis
    )

    print(f"Permutation Test ({sex}): Mean MAE = {abs(score):.3f}, p-value = {p_value:.3f}")

    # Save results to individual CSV files
    results_df = pd.DataFrame({
        "perm_scores": perm_scores
    })
    results_df["score"] = score
    results_df["p_value"] = p_value

    results_df.to_csv(save_dir / f"permutation_test_{sex}.csv", index=False)
    print(f"Saved permutation test results for {sex}.")
    print("Experiment done!")

# %%

# %%
