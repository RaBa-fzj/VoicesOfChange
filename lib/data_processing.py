from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class ConfoundRegressor(TransformerMixin, BaseEstimator):
    def __init__(self, confound_indices, features_to_unconfound):
        """
        Initialize the confound regressor.

        Parameters:
        confound_indices (list): List of column indices to use as confounders.
        unconfound_indices (list): List of column indices to unconfound.
        """
        self.confound_indices = confound_indices
        self.features_to_unconfound = features_to_unconfound
        self.feature_models = {}  # Store regression models for each feature to unconfound
        self.y_model = None  # Regression model for the target variable

    def fit(self, X, y):
        """
        Fit the confound regression models using the training data.

        Parameters:
        X (np.ndarray): Input features with shape (n_samples, n_features).
        y (np.ndarray): Target variable with shape (n_samples,).
        """
        # Extract confounders from X
        confounders = X[:, self.confound_indices]

        # Fit regression models for each feature to unconfound
        for idx in self.features_to_unconfound:
            reg = LinearRegression().fit(confounders, X[:, idx])
            self.feature_models[idx] = reg

        # Fit the regression model for y
        self.y_model = LinearRegression().fit(confounders, y)

        return self

    def transform(self, X, y=None):
        """
        Transform X and y by removing the effect of the confound variables.

        Parameters:
        X (np.ndarray): Input features with shape (n_samples, n_features).
        y (np.ndarray): Target variable with shape (n_samples,).

        Returns:
        X_unconfounded (np.ndarray): Features after removing the effect of confounders.
        y_unconfounded (np.ndarray): Target after removing the effect of confounders.
        """
        # Extract confounders from X
        confounders = X[:, self.confound_indices]

        # Create a copy of X to store unconfounded features
        X_unconfounded = X.copy()

        # Apply the confound regression to each feature to unconfound
        for idx in self.features_to_unconfound:
            X_unconfounded[:, idx] = X[:, idx] - self.feature_models[idx].predict(
                confounders
            )

        # Apply the confound regression to y if provided
        if y is not None:
            y_unconfounded = y - self.y_model.predict(confounders)
            return X_unconfounded, y_unconfounded
        else:
            return X_unconfounded


