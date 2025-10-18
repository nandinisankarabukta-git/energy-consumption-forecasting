# -*- coding: utf-8 -*-
"""
Visualization module.
Creates visualizations for data exploration and model results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import click
import logging
from pathlib import Path


def plot_aggregated_usage_trend(df, output_path=None):
    """Plot aggregated daily electricity usage trend."""
    # Aggregate electricity usage over time
    aggregated_data = df.groupby('date')['electricity_usage'].sum().reset_index()
    
    plt.figure(figsize=(14, 8))
    plt.plot(aggregated_data['date'], aggregated_data['electricity_usage'], 
             color='purple', linewidth=2, alpha=0.8)
    plt.title('Aggregated Daily Electricity Usage Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Electricity Usage (kWh)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.getLogger(__name__).info(f'Saved plot to {output_path}')
    
    plt.close()


def plot_sample_buildings_trend(df, n_buildings=5, output_path=None):
    """Plot electricity usage trends for sample buildings."""
    # Select a sample of buildings
    sample_buildings = df['building_name'].dropna().unique()[:n_buildings]
    sample_data = df[df['building_name'].isin(sample_buildings)]
    
    plt.figure(figsize=(14, 8))
    for building in sample_buildings:
        building_data = sample_data[sample_data['building_name'] == building]
        plt.plot(building_data['date'], building_data['electricity_usage'], label=building)
    
    plt.title('Electricity Usage Trends for Sample Buildings', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Electricity Usage (kWh)', fontsize=12)
    plt.legend(title="Building Name", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.getLogger(__name__).info(f'Saved plot to {output_path}')
    
    plt.close()


def plot_feature_importance(feature_importance_df, output_path=None):
    """Plot feature importance from model."""
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
             color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Feature Importance - Random Forest', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.getLogger(__name__).info(f'Saved plot to {output_path}')
    
    plt.close()


def plot_actual_vs_predicted(y_test, y_pred, sample_size=500, output_path=None):
    """Plot actual vs predicted values with raw and smoothed versions."""
    rolling_window = 50
    y_test_series = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else pd.Series(y_test)
    y_pred_series = pd.Series(y_pred)
    
    y_test_rolling = y_test_series.rolling(window=rolling_window).mean()
    y_pred_rolling = y_pred_series.rolling(window=rolling_window).mean()
    
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Actual vs Predicted (Raw)
    plt.subplot(2, 1, 1)
    plt.plot(y_test_series[:sample_size], label='Actual (Raw)', 
             color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(y_pred_series[:sample_size], label='Predicted (Raw)', 
             color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.title(f'Actual vs Predicted Electricity Usage (First {sample_size} Samples)', fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Electricity Usage (kWh)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Smoothed Actual vs Predicted
    plt.subplot(2, 1, 2)
    plt.plot(y_test_rolling[:sample_size], label='Actual (Smoothed)', 
             color='blue', linewidth=2, alpha=0.9)
    plt.plot(y_pred_rolling[:sample_size], label='Predicted (Smoothed)', 
             color='orange', linestyle='--', linewidth=2, alpha=0.9)
    plt.title('Smoothed Actual vs Predicted Electricity Usage', fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Electricity Usage (kWh)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.getLogger(__name__).info(f'Saved plot to {output_path}')
    
    plt.close()


def plot_scatter_actual_vs_predicted(y_test, y_pred, output_path=None):
    """Plot scatter plot of predicted vs actual values."""
    y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_array, y_pred, alpha=0.5)
    plt.plot([y_test_array.min(), y_test_array.max()], 
             [y_test_array.min(), y_test_array.max()], 
             'r--', linewidth=2)  # Reference line
    plt.title('Predicted vs Actual Electricity Usage', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Electricity Usage (kWh)', fontsize=12)
    plt.ylabel('Predicted Electricity Usage (kWh)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.getLogger(__name__).info(f'Saved plot to {output_path}')
    
    plt.close()


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--feature-importance', type=click.Path(exists=True), 
              help='Path to feature importance CSV file')
def main(data_filepath, output_dir, feature_importance):
    """
    Creates visualizations for the project.
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating visualizations')
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f'Loading data from {data_filepath}')
    df = pd.read_csv(data_filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create visualizations
    logger.info('Creating aggregated usage trend plot...')
    plot_aggregated_usage_trend(df, output_dir / 'aggregated_usage_trend.png')
    
    logger.info('Creating sample buildings trend plot...')
    plot_sample_buildings_trend(df, output_path=output_dir / 'sample_buildings_trend.png')
    
    if feature_importance:
        logger.info('Creating feature importance plot...')
        feature_imp_df = pd.read_csv(feature_importance)
        plot_feature_importance(feature_imp_df, output_dir / 'feature_importance.png')
    
    logger.info('Visualizations created successfully!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

