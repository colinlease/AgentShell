

"""Reusable model-spec definitions for the ML Workbench app.

This module contains declarative model definitions only. The registry layer
should import these specs and expose lookup helpers, while the modeling service
should use the registry for validation and orchestration.
"""

from __future__ import annotations

from typing import Any, cast

from app.workspace_apps.ml_workbench.schemas import ModelSpec


PROBLEM_TYPE_CLASSIFICATION = "classification"
PROBLEM_TYPE_REGRESSION = "regression"

MODEL_FAMILY_LINEAR = "linear"
MODEL_FAMILY_TREE = "tree"


DEFAULT_CLASSIFICATION_METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]
DEFAULT_REGRESSION_METRICS = ["rmse", "mae", "r2"]



def _classification_tuning_schema() -> dict[str, Any]:
    """Return a lightweight default tuning schema for classification models."""
    return {
        "supported_search_types": ["grid", "random"],
        "default_scoring": "roc_auc",
        "search_space_defaults": {},
    }



def _regression_tuning_schema() -> dict[str, Any]:
    """Return a lightweight default tuning schema for regression models."""
    return {
        "supported_search_types": ["grid", "random"],
        "default_scoring": "neg_root_mean_squared_error",
        "search_space_defaults": {},
    }



def _build_logistic_regression_spec() -> ModelSpec:
    """Return the logistic regression model spec."""
    return cast(
        ModelSpec,
        {
            "model_id": "logistic_regression",
            "label": "Logistic Regression",
            "problem_types": [PROBLEM_TYPE_CLASSIFICATION],
            "family": MODEL_FAMILY_LINEAR,
            "supports_probability": True,
            "supports_feature_importance": False,
            "default_params": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
            },
            "param_schema": {
                "C": {
                    "label": "Regularization strength",
                    "type": "float",
                    "default": 1.0,
                    "min": 0.0001,
                    "max": 100.0,
                    "step": 0.1,
                    "help_text": "Smaller values increase regularization.",
                },
                "penalty": {
                    "label": "Penalty",
                    "type": "select",
                    "default": "l2",
                    "options": ["l2"],
                    "help_text": "Version 1 supports l2 regularization.",
                },
                "solver": {
                    "label": "Solver",
                    "type": "select",
                    "default": "lbfgs",
                    "options": ["lbfgs", "liblinear"],
                    "help_text": "Optimization algorithm used during fitting.",
                },
                "max_iter": {
                    "label": "Max iterations",
                    "type": "int",
                    "default": 1000,
                    "min": 100,
                    "max": 5000,
                    "step": 100,
                    "help_text": "Upper bound on the number of solver iterations.",
                },
            },
            "tuning_schema": {
                **_classification_tuning_schema(),
                "search_space_defaults": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["lbfgs", "liblinear"],
                },
            },
            "metrics": DEFAULT_CLASSIFICATION_METRICS,
        },
    )



def _build_random_forest_classifier_spec() -> ModelSpec:
    """Return the random forest classifier model spec."""
    return cast(
        ModelSpec,
        {
            "model_id": "random_forest_classifier",
            "label": "Random Forest Classifier",
            "problem_types": [PROBLEM_TYPE_CLASSIFICATION],
            "family": MODEL_FAMILY_TREE,
            "supports_probability": True,
            "supports_feature_importance": True,
            "default_params": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            },
            "param_schema": {
                "n_estimators": {
                    "label": "Number of trees",
                    "type": "int",
                    "default": 200,
                    "min": 50,
                    "max": 1000,
                    "step": 50,
                    "help_text": "More trees can improve stability at the cost of runtime.",
                },
                "max_depth": {
                    "label": "Max depth",
                    "type": "optional_int",
                    "default": None,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "help_text": "Limit tree depth to reduce overfitting.",
                },
                "min_samples_split": {
                    "label": "Min samples to split",
                    "type": "int",
                    "default": 2,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "help_text": "Minimum samples needed to split an internal node.",
                },
                "min_samples_leaf": {
                    "label": "Min samples per leaf",
                    "type": "int",
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "help_text": "Minimum samples required at each leaf node.",
                },
            },
            "tuning_schema": {
                **_classification_tuning_schema(),
                "search_space_defaults": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "metrics": DEFAULT_CLASSIFICATION_METRICS,
        },
    )



def _build_linear_regression_spec() -> ModelSpec:
    """Return the linear regression model spec."""
    return cast(
        ModelSpec,
        {
            "model_id": "linear_regression",
            "label": "Linear Regression",
            "problem_types": [PROBLEM_TYPE_REGRESSION],
            "family": MODEL_FAMILY_LINEAR,
            "supports_probability": False,
            "supports_feature_importance": False,
            "default_params": {
                "fit_intercept": True,
            },
            "param_schema": {
                "fit_intercept": {
                    "label": "Fit intercept",
                    "type": "bool",
                    "default": True,
                    "help_text": "Whether to calculate the intercept for this model.",
                },
            },
            "tuning_schema": {
                **_regression_tuning_schema(),
                "search_space_defaults": {
                    "fit_intercept": [True, False],
                },
            },
            "metrics": DEFAULT_REGRESSION_METRICS,
        },
    )



def _build_random_forest_regressor_spec() -> ModelSpec:
    """Return the random forest regressor model spec."""
    return cast(
        ModelSpec,
        {
            "model_id": "random_forest_regressor",
            "label": "Random Forest Regressor",
            "problem_types": [PROBLEM_TYPE_REGRESSION],
            "family": MODEL_FAMILY_TREE,
            "supports_probability": False,
            "supports_feature_importance": True,
            "default_params": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            },
            "param_schema": {
                "n_estimators": {
                    "label": "Number of trees",
                    "type": "int",
                    "default": 200,
                    "min": 50,
                    "max": 1000,
                    "step": 50,
                    "help_text": "More trees can improve stability at the cost of runtime.",
                },
                "max_depth": {
                    "label": "Max depth",
                    "type": "optional_int",
                    "default": None,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "help_text": "Limit tree depth to reduce overfitting.",
                },
                "min_samples_split": {
                    "label": "Min samples to split",
                    "type": "int",
                    "default": 2,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "help_text": "Minimum samples needed to split an internal node.",
                },
                "min_samples_leaf": {
                    "label": "Min samples per leaf",
                    "type": "int",
                    "default": 1,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "help_text": "Minimum samples required at each leaf node.",
                },
            },
            "tuning_schema": {
                **_regression_tuning_schema(),
                "search_space_defaults": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "metrics": DEFAULT_REGRESSION_METRICS,
        },
    )


LOGISTIC_REGRESSION_SPEC = _build_logistic_regression_spec()
RANDOM_FOREST_CLASSIFIER_SPEC = _build_random_forest_classifier_spec()
LINEAR_REGRESSION_SPEC = _build_linear_regression_spec()
RANDOM_FOREST_REGRESSOR_SPEC = _build_random_forest_regressor_spec()


DEFAULT_MODEL_SPECS = [
    LOGISTIC_REGRESSION_SPEC,
    RANDOM_FOREST_CLASSIFIER_SPEC,
    LINEAR_REGRESSION_SPEC,
    RANDOM_FOREST_REGRESSOR_SPEC,
]