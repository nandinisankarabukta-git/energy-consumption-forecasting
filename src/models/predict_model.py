# -*- coding: utf-8 -*-
"""
Model prediction module.
Makes predictions using trained model.
"""
import pandas as pd
import click
import logging
import pickle
from pathlib import Path


def load_model(model_filepath):
    """Load trained model from file."""
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def make_predictions(model, X):
    """Make predictions using trained model."""
    predictions = model.predict(X)
    return predictions


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(model_filepath, input_filepath, output_filepath):
    """
    Makes predictions using trained model.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making predictions with trained model')
    
    # Load model
    logger.info(f'Loading model from {model_filepath}')
    model = load_model(model_filepath)
    
    # Load data
    logger.info(f'Loading data from {input_filepath}')
    df = pd.read_csv(input_filepath)
    
    # Load predictor names
    predictor_path = Path(model_filepath).parent / 'predictors.txt'
    if predictor_path.exists():
        with open(predictor_path, 'r') as f:
            predictors = [line.strip() for line in f.readlines()]
    else:
        # Fallback to default predictors
        predictors = [
            'month', 'day_of_week', 'is_weekend', 'airTemperature',
            'dewTemperature', 'seaLvlPressure', 'windSpeed', 'sqm', 'lag_1', 'lag_7'
        ]
    
    X = df[predictors]
    
    # Make predictions
    logger.info('Making predictions...')
    predictions = make_predictions(model, X)
    
    # Save predictions
    df['predicted_electricity_usage'] = predictions
    logger.info(f'Saving predictions to {output_filepath}')
    df.to_csv(output_filepath, index=False)
    
    logger.info('Predictions completed successfully!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

