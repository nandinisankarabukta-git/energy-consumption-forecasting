# -*- coding: utf-8 -*-
"""
Data loading and preprocessing module.
Loads raw data files and merges them into a single dataset.
"""

import pandas as pd
import click
import logging
from pathlib import Path


def load_metadata(filepath):
    """Load and clean metadata file."""
    metadata_df = pd.read_csv(filepath)

    # Drop unnecessary columns from metadata
    columns_to_drop = [
        "building_id_kaggle",
        "site_id_kaggle",
        "solar",
        "industry",
        "subindustry",
        "heatingtype",
        "date_opened",
        "numberoffloors",
        "occupants",
        "energystarscore",
        "eui",
        "site_eui",
        "source_eui",
        "leed_level",
        "rating",
    ]
    metadata_df_cleaned = metadata_df.drop(columns=columns_to_drop, errors="ignore")

    return metadata_df_cleaned


def load_weather(filepath):
    """Load and preprocess weather data."""
    weather_df = pd.read_csv(filepath)

    # Convert weather timestamp to datetime
    weather_df["timestamp"] = pd.to_datetime(
        weather_df["timestamp"], format="%d-%m-%Y %H:%M"
    )

    return weather_df


def load_electricity(filepath):
    """Load and reshape electricity usage data."""
    electricity_df = pd.read_csv(filepath)

    # Reshape electricity usage dataset into long format
    electricity_long = electricity_df.melt(
        id_vars=["building_name", "site_id"],
        var_name="date",
        value_name="electricity_usage",
    )
    electricity_long["date"] = pd.to_datetime(
        electricity_long["date"], format="%d-%m-%Y"
    )

    return electricity_long


def merge_datasets(electricity_df, metadata_df, weather_df):
    """Merge electricity, metadata, and weather datasets."""
    # Merge electricity usage with metadata
    electricity_metadata_merged = electricity_df.merge(
        metadata_df,
        left_on=["building_name", "site_id"],
        right_on=["building_id", "site_id"],
        how="left",
    )

    # Merge with weather data
    final_merged_dataset = electricity_metadata_merged.merge(
        weather_df,
        left_on=["site_id", "date"],
        right_on=["site_id", "timestamp"],
        how="left",
    )
    final_merged_dataset.drop(columns=["timestamp"], inplace=True)

    # Drop irrelevant columns
    columns_to_drop_final = [
        "hotwater",
        "chilledwater",
        "steam",
        "water",
        "irrigation",
        "gas",
        "unique_space_usages",
    ]
    final_merged_dataset_cleaned = final_merged_dataset.drop(
        columns=columns_to_drop_final, errors="ignore"
    )

    return final_merged_dataset_cleaned


@click.command()
@click.argument("metadata_filepath", type=click.Path(exists=True))
@click.argument("weather_filepath", type=click.Path(exists=True))
@click.argument("electricity_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(metadata_filepath, weather_filepath, electricity_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making interim dataset from raw data")

    # Load datasets
    logger.info("Loading metadata...")
    metadata_df = load_metadata(metadata_filepath)

    logger.info("Loading weather data...")
    weather_df = load_weather(weather_filepath)

    logger.info("Loading electricity usage data...")
    electricity_df = load_electricity(electricity_filepath)

    # Merge datasets
    logger.info("Merging datasets...")
    final_dataset = merge_datasets(electricity_df, metadata_df, weather_df)

    # Save interim data
    logger.info(f"Saving interim dataset to {output_filepath}")
    final_dataset.to_csv(output_filepath, index=False)
    logger.info("Dataset created successfully!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
