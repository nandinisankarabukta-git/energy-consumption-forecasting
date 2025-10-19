# -*- coding: utf-8 -*-
"""
Model training module.
Trains Random Forest model on processed data.
"""

import pandas as pd
import numpy as np
import click
import logging
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train Random Forest model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training Random Forest with {n_estimators} estimators...")

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    logger.info("Model training completed")
    return rf_model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    logger = logging.getLogger(__name__)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Root Mean Squared Error: {rmse:.4f}")
    logger.info(f"R^2 Score: {r2:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")

    return {"rmse": rmse, "r2": r2, "mae": mae, "predictions": y_pred}


def cross_validate_model(model, X, y, cv=5):
    """Perform k-fold cross-validation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Performing {cv}-fold cross-validation...")

    # Use negative mean squared error for scoring
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")

    # Convert scores to positive RMSE
    rmse_scores = np.sqrt(-cv_scores)

    logger.info(f"RMSE for each fold: {rmse_scores}")
    logger.info(f"Mean RMSE: {rmse_scores.mean():.4f}")
    logger.info(f"Standard Deviation of RMSE: {rmse_scores.std():.4f}")

    return {
        "rmse_scores": rmse_scores,
        "mean_rmse": rmse_scores.mean(),
        "std_rmse": rmse_scores.std(),
    }


def get_feature_importance(model, feature_names):
    """Get feature importance from trained model."""
    feature_importances = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    return feature_importances


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path())
@click.option("--test-size", default=0.3, help="Test set size (default: 0.3)")
@click.option("--n-estimators", default=100, help="Number of trees (default: 100)")
@click.option("--cv", default=5, help="Number of cross-validation folds (default: 5)")
def main(input_filepath, model_filepath, test_size, n_estimators, cv):
    """
    Trains Random Forest model on processed data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Training model on processed data")

    # Load processed data
    logger.info(f"Loading data from {input_filepath}")
    df = pd.read_csv(input_filepath)

    # Load predictor names
    predictor_path = Path(input_filepath).parent / "predictors.txt"
    with open(predictor_path, "r") as f:
        predictors = [line.strip() for line in f.readlines()]

    # Prepare features and target
    X = df[predictors]
    y = df["electricity_usage"]

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {predictors}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Train model
    model = train_random_forest(X_train, y_train, n_estimators=n_estimators)

    # Evaluate model
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test)

    # Get feature importance
    feature_importance = get_feature_importance(model, predictors)
    logger.info("Feature Importance:")
    logger.info(f"\n{feature_importance}")

    # Perform cross-validation
    cv_metrics = cross_validate_model(
        RandomForestRegressor(n_estimators=n_estimators, random_state=42), X, y, cv=cv
    )

    # Save model
    logger.info(f"Saving model to {model_filepath}")
    model_dir = Path(model_filepath).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    metrics_path = model_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("=== Model Performance Metrics ===\n")
        f.write(f"Root Mean Squared Error: {metrics['rmse']:.4f}\n")
        f.write(f"R^2 Score: {metrics['r2']:.4f}\n")
        f.write(f"Mean Absolute Error: {metrics['mae']:.4f}\n")
        f.write(f"\n=== Cross-Validation Results ===\n")
        f.write(f"Mean RMSE: {cv_metrics['mean_rmse']:.4f}\n")
        f.write(f"Standard Deviation of RMSE: {cv_metrics['std_rmse']:.4f}\n")

    # Save feature importance
    feature_importance_path = model_dir / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)

    logger.info("Model training and evaluation completed successfully!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
