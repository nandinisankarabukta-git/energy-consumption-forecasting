# -*- coding: utf-8 -*-
"""
Feature engineering module.
Creates features from the interim dataset for modeling.
"""
import pandas as pd
import click
import logging
from pathlib import Path


def add_time_features(df):
    """Add time-based features."""
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df


def add_lag_features(df):
    """Add lag features for time series."""
    df['lag_1'] = df['electricity_usage'].shift(1)
    df['lag_7'] = df['electricity_usage'].shift(7)
    df.dropna(subset=['lag_1', 'lag_7'], inplace=True)
    
    return df


def remove_outliers(df, column='electricity_usage'):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_cleaned = df[
        (df[column] >= lower_bound) &
        (df[column] <= upper_bound)
    ]
    
    return df_cleaned


def handle_missing_values(df, predictors):
    """Impute missing values for predictors."""
    df[predictors] = df[predictors].fillna(0)
    return df


def build_features(df):
    """Build all features from the interim dataset."""
    # Add time-based features
    df = add_time_features(df)
    
    # Add lag features
    df = add_lag_features(df)
    
    # Remove outliers
    df = remove_outliers(df)
    
    # Handle missing values
    predictors = [
        'month', 'day_of_week', 'is_weekend', 'airTemperature',
        'dewTemperature', 'seaLvlPressure', 'windSpeed', 'sqm', 'lag_1', 'lag_7'
    ]
    df = handle_missing_values(df, predictors)
    
    return df, predictors


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs feature engineering scripts to turn interim data from (../interim) into
    processed data ready for modeling (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Building features from interim data')
    
    # Load interim data
    logger.info(f'Loading data from {input_filepath}')
    df = pd.read_csv(input_filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Build features
    logger.info('Engineering features...')
    df_processed, predictors = build_features(df)
    
    # Save processed data
    logger.info(f'Saving processed dataset to {output_filepath}')
    df_processed.to_csv(output_filepath, index=False)
    
    # Save predictor names
    predictor_path = Path(output_filepath).parent / 'predictors.txt'
    with open(predictor_path, 'w') as f:
        f.write('\n'.join(predictors))
    
    logger.info('Features built successfully!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

