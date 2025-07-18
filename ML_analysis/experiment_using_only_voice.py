# %%
import sys
import shap
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, GridSearchCV

from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import json

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
    get_used_feature_names,
)
from lib.ml import get_ml_pipeline, pad_shap_explanation  # noqa
from lib.utils import ensure_dir  # noqa


# %% ############################# Variables and data
data_dir = project_dir / "data/"
file_name = "Paper_PV0775_SDQ_merged_dataset_whole.xlsx"
# Add an experiment name to add in the last part of the
# saved files
exp_name = "only_voice_features"

save_dir = project_dir / "output" / (exp_name + "/")
ensure_dir(save_dir)

# include the voice features or not
include_voice = True

# Include the puberty for the "Robustnes" analysis
# If true, several patients will be drop for the lack of data
# and a smaller cohort is generated
include_puberty = False

# Set False to avoid print results for each fold
verbose = False

# Define parameter grid for hyperparameter tuning
param_grid = {
    "selector__k": ["all"],  # All retained (4)
    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
}
# For Hyperparameter tuning
n_inner_folds = 3
# For outer loop
n_outer_folds = 5
# Repetitions
n_repeats = 20
#######################################################
# %% Data loading
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

X.drop(
    columns=[
        "age",
        "sex12",
        "soz_winkler_2019__d00408_gesamt_score",
        "c_anthro_kh__d00040_bmi_sds",
    ], inplace=True
)
covariats = []
categorical_features = []

numerical_features.remove("age")
numerical_features.remove("soz_winkler_2019__d00408_gesamt_score")
numerical_features.remove("c_anthro_kh__d00040_bmi_sds")


# Get the ML pipeline
pipeline = get_ml_pipeline(
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    voice_features=voice_features,
    covariats=covariats,
    SEED=SEED,
    ordinal_features=ordinal_features,
)

# Outer cross-validation with GroupKFold
outer_train_mae = []  # NEW training error
outer_train_r2 = []  # NEW training error
outer_train_corr = []
outer_train_corr_p = []
outer_mae = []
outer_r2 = []
outer_corr = []  # NEW outer cross-validation
outer_corr_p = []  # NEW outer cross-validation p-values

cv_predictions = np.zeros(len(y))  # For storing cross-validated predictions
cv_train_predictions = np.zeros(len(y))  # For storing cross-validated predictions

predictions_repetitions = []
train_predictions_repetitions = []

y_loop = []

# for SHAP values
feature_importances_list = []

shap_loop = []
all_feature_names_set = set()  # Collect all features used across folds

# Inner cross-validation for hyperparameter tuning
inner_cv = GroupKFold(n_splits=n_inner_folds)


for repeat in range(n_repeats):
    print(f"Repetition {repeat + 1}/{n_repeats}")
    # Create a different random split for each repetition
    outer_cv = GroupKFold(
        n_splits=n_outer_folds, random_state=SEED + repeat, shuffle=True
    )
    shap_values = np.zeros([X.shape[0], X.shape[1]])
    shap_baseline = np.zeros(X.shape[0])
    shap_data = np.zeros([X.shape[0], X.shape[1]])
    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups=groups)
    ):
        print(f"Fold {fold_idx + 1}")

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]  # not used

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv.split(X_train, y_train, groups=groups_train),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,  # Use all cores for GridSearchCV
        )

        # find the best hyperparameters and fit the final model
        grid_search.fit(X_train, y_train)

        # Evaluate the best model on the test set
        best_model = grid_search.best_estimator_

        # ###### Train Performance
        y_pred_train = best_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        outer_train_mae.append(train_mae)
        outer_train_r2.append(train_r2)

        corr_train, p_value_corr_train = pearsonr(y_train, y_pred_train)
        outer_train_corr.append(corr_train)
        outer_train_corr_p.append(p_value_corr_train)

        # Save training predictions
        cv_train_predictions[train_idx] = y_pred_train

        # ###### Test Performance
        # Test (fold) error
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        outer_mae.append(mae)
        outer_r2.append(r2)

        # Correlation within the loop
        corr, p_value_corr = pearsonr(y_test, y_pred)
        outer_corr.append(corr)
        outer_corr_p.append(p_value_corr)

        # Store predictions for overall metrics
        cv_predictions[test_idx] = (
            y_pred  # this is storing only the last fold, not the repetitions
        )

        if verbose:
            print(f"  MAE: {mae:.3f}")
            print(f"  R²: {r2:.3f}")
            print(f"  Corr: {corr:.3f}, p={p_value_corr:.3f}")  # Correlation
            print(f"  Best Params: {grid_search.best_params_}")

        # #################  Model explanation
        # ######  Model's weights

        # Extract feature coefficients
        importances = best_model.named_steps["model"].coef_

        # Get feature names after preprocessing and selection
        preprocessor = best_model.named_steps["preprocessor"]
        feature_selector = best_model.named_steps["selector"]
        selected_features_mask = feature_selector.get_support()

        feature_names = get_used_feature_names(preprocessor)

        # Apply mask to get selected feature names
        selected_feature_names = [
            feature_names[i]
            for i in range(len(feature_names))
            if selected_features_mask[i]
        ]

        # Update the set of all feature names
        all_feature_names_set.update(selected_feature_names)

        feature_importances_df = pd.DataFrame(
            {"Feature": selected_feature_names, "Importance": importances}
        )

        feature_importances_list.append(feature_importances_df)

        # ######  SHAP
        # Compute SHAP values (Ridge)
        # Prepare your background data from the TRAIN set
        X_train_transformed = best_model.named_steps["preprocessor"].transform(X_train)
        X_train_selected = best_model.named_steps["selector"].transform(
            X_train_transformed
        )

        # Use the universal SHAP explainer with the model’s predict function
        explainer = shap.LinearExplainer(
            best_model.named_steps["model"],
            X_train_selected,
            feature_names=selected_feature_names,
        )

        X_test_transformed = best_model.named_steps["preprocessor"].transform(X_test)
        X_test_selected = best_model.named_steps["selector"].transform(
            X_test_transformed
        )

        shap_values_fold = explainer(X_test_selected)
        shap_values_fold = pad_shap_explanation(
            shap_values_fold, selected_feature_names, feature_names
        )
        shap_values[test_idx, :] = shap_values_fold.values
        shap_baseline[test_idx] = shap_values_fold.base_values
        shap_data[test_idx, :] = shap_values_fold.data

    # End of the repetition, append the results for each repetition
    # Store predictions for each repetition
    predictions_repetitions.append(cv_predictions.copy())
    train_predictions_repetitions.append(cv_train_predictions.copy())
    y_loop.append(y.copy())

    # Get the shaps values of the repetition
    shap_loop.append([shap_values.copy(), shap_baseline.copy(), shap_data.copy()])


### Store model metrics for t-test
# Convert lists to numpy arrays
outer_mae = np.array(outer_mae)
outer_r2 = np.array(outer_r2)
outer_corr = np.array(outer_corr)

outer_train_mae = np.array(outer_train_mae)
outer_train_r2 = np.array(outer_train_r2)
outer_train_corr = np.array(outer_train_corr)

full_predictions = np.concatenate(predictions_repetitions)
train_full_predictions = np.concatenate(train_predictions_repetitions)

full_y = np.concatenate(y_loop)

if verbose:
    print(f"\nAveraged MAE: {outer_mae.mean():.3f}")
    print(f"Average Training MAE (across folds): {outer_train_mae.mean():.3f}")
    print(f"Averaged R²: {outer_r2.mean():.3f}")
    print(f"Average Training R² (across folds): {outer_train_r2.mean():.3f}")
    print(f"Averaged r: {outer_corr.mean():.3f}")
# %% Save results
# Save results (change the filename based on the model type)
np.save(save_dir / f"train_mae_{exp_name}.npy", outer_train_mae)
np.save(save_dir / f"train_r2_{exp_name}.npy", outer_train_r2)
np.save(save_dir / f"train_corr_{exp_name}.npy", outer_train_corr)
np.save(save_dir / f"train_full_predictions_{exp_name}.npy", train_full_predictions)


# Save results (change the filename based on the model type)
np.save(save_dir / f"mae_{exp_name}.npy", outer_mae)
np.save(save_dir / f"r2_{exp_name}.npy", outer_r2)
np.save(save_dir / f"corr_{exp_name}.npy", outer_corr)
np.save(save_dir / f"full_predictions_{exp_name}.npy", full_predictions)
np.save(save_dir / f"full_y_true_{exp_name}.npy", full_y)

with open(save_dir / f"shap_explanations_{exp_name}.pkl", "wb") as f:
    pickle.dump(shap_loop, f)

# Save feature importances
all_feature_importances_df = pd.concat(feature_importances_list, ignore_index=True)
all_feature_importances_df.to_csv(
    save_dir / f"all_feature_importances_{exp_name}.csv", index=False
)

pd.DataFrame({"feature": feature_names}).to_csv(
    save_dir / f"final_feature_names_{exp_name}.csv", index=False
)

summary = {
    "script_name": "experiment_using_voice_features_confound_regressed.py",
    "n_repeats": n_repeats,
    "outer_cv_folds": 5,
    "inner_cv_folds": 3,
    "random_seed": SEED,
    "include_voice": include_voice,
    "include_puberty": include_puberty,
    "exp_name": exp_name,
    "n_features": int(X.shape[1]),
    "n_patients": int(X.shape[0]),
    "n_unique_patients": int(groups.nunique()),
    "param_grid": param_grid,
    "output_dir": str(save_dir),
    "metrics": ["mae", "r2", "corr"],
    "feature_importances_file": str(save_dir / "all_feature_importances_voice.csv"),
    "shap_values_file": str(save_dir / "all_shap_values_df_voice.csv"),
    "shap_data_file": str(save_dir / "all_shap_data_df_voice.csv"),
    "shap_base_value_file": str(save_dir / "all_shap_base_value_df_voice.csv"),
    "final_feature_names_file": str(save_dir / "final_feature_names.csv"),
}

with open(save_dir / f"experiment_summary_{exp_name}.json", "w") as f:
    json.dump(summary, f, indent=4)

print("Experiment done!")

# %%
