Getting Started
===============

This guide will walk you through setting up the Energy Consumption Forecasting project and running the complete machine learning pipeline.

Prerequisites
-------------

Before you begin, ensure you have:

* **Python 3.7 or higher** installed on your system
* **pip** package manager
* **Git** (for cloning the repository)
* At least **3GB of free disk space** for data and models

System Requirements
~~~~~~~~~~~~~~~~~~~

* **Operating System:** Windows, macOS, or Linux
* **RAM:** Minimum 8GB (16GB recommended for faster training)
* **CPU:** Multi-core processor recommended
* **Disk Space:** ~3GB for datasets, models, and outputs


Installation
------------

Step 1: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd energy-consumption-forecasting


Step 2: Create a Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using venv (built-in)
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

   # OR using conda
   conda create --name energy-forecasting python=3.9
   conda activate energy-forecasting


Step 3: Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

This will install:

* pandas >= 1.3.0
* numpy >= 1.21.0
* scikit-learn >= 0.24.0
* matplotlib >= 3.3.0
* click
* python-dotenv


Step 4: Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python test_environment.py

You should see: ``>>> Development environment passes all tests!``


Running the Pipeline
--------------------

The project follows a 4-step data science pipeline:

1. **Data Processing** - Merge raw datasets
2. **Feature Engineering** - Create predictive features
3. **Model Training** - Train Random Forest model
4. **Visualization** - Generate insights and plots


Step 1: Data Processing
~~~~~~~~~~~~~~~~~~~~~~~~

Merge electricity usage, building metadata, and weather data:

.. code-block:: bash

   python src/data/make_dataset.py \
       data/raw/metadata.csv \
       data/raw/weather.csv \
       data/raw/electricity_usage.csv \
       data/interim/merged_data.csv

**Expected Output:**

* Creates ``data/interim/merged_data.csv`` with **494,489 records**
* Execution time: ~4 seconds

**What it does:**

* Loads metadata, weather, and electricity usage datasets
* Reshapes electricity data from wide to long format
* Merges all datasets on building_id, site_id, and timestamp
* Removes unnecessary columns


Step 2: Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create time-series and temporal features:

.. code-block:: bash

   python src/features/build_features.py \
       data/interim/merged_data.csv \
       data/processed/features.csv

**Expected Output:**

* Creates ``data/processed/features.csv`` with **443,075 records**
* Creates ``data/processed/predictors.txt`` listing all features
* Execution time: ~4 seconds

**Features created:**

* **Time-based:** month, day_of_week, is_weekend
* **Lag features:** lag_1 (previous day), lag_7 (previous week)
* **Weather:** airTemperature, dewTemperature, seaLvlPressure, windSpeed
* **Building:** sqm (square meters)

**Data cleaning:**

* Removes outliers using IQR method (~51,414 records)
* Imputes missing values with zeros


Step 3: Model Training
~~~~~~~~~~~~~~~~~~~~~~~

Train Random Forest model with cross-validation:

.. code-block:: bash

   python src/models/train_model.py \
       data/processed/features.csv \
       models/random_forest.pkl \
       --n-estimators 100 \
       --cv 5

**Expected Output:**

* Creates ``models/random_forest.pkl`` (trained model)
* Creates ``models/metrics.txt`` (performance metrics)
* Creates ``models/feature_importance.csv`` (feature rankings)
* Execution time: ~18 minutes (includes 5-fold cross-validation)

**Model Performance:**

* R² Score: 0.9610
* RMSE: 386.70 kWh
* MAE: 167.02 kWh
* Cross-validated RMSE: 700.26 ± 119.56 kWh

**Optional parameters:**

* ``--n-estimators N`` : Number of trees (default: 100)
* ``--cv N`` : Cross-validation folds (default: 5)
* ``--test-size RATIO`` : Test set proportion (default: 0.3)


Step 4: Generate Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create plots and visual insights:

.. code-block:: bash

   python src/visualization/visualize.py \
       data/processed/features.csv \
       reports/figures/ \
       --feature-importance models/feature_importance.csv

**Expected Output:**

* Creates ``reports/figures/aggregated_usage_trend.png``
* Creates ``reports/figures/sample_buildings_trend.png``
* Creates ``reports/figures/feature_importance.png``
* Execution time: ~2 seconds

**Visualizations include:**

* Daily aggregated electricity usage trends
* Individual building consumption patterns (5 sample buildings)
* Feature importance rankings


Making Predictions
------------------

Use the trained model to predict electricity usage on new data:

.. code-block:: bash

   python src/models/predict_model.py \
       models/random_forest.pkl \
       data/processed/features.csv \
       predictions.csv

**Requirements for new data:**

The input data must contain all 10 predictor features:

1. month
2. day_of_week
3. is_weekend
4. airTemperature
5. dewTemperature
6. seaLvlPressure
7. windSpeed
8. sqm
9. lag_1
10. lag_7


Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue:** ``ModuleNotFoundError: No module named 'sklearn'``

**Solution:** Install scikit-learn

.. code-block:: bash

   pip install scikit-learn>=0.24.0


**Issue:** ``SettingWithCopyWarning`` during feature engineering

**Solution:** This is a pandas warning and doesn't affect results. The code functions correctly.


**Issue:** Model training takes too long (>30 minutes)

**Solution:** Reduce the number of estimators:

.. code-block:: bash

   python src/models/train_model.py ... --n-estimators 50


**Issue:** Out of memory error during training

**Solution:** 

* Close other applications
* Reduce dataset size by sampling
* Use fewer cross-validation folds: ``--cv 3``


**Issue:** Missing data files in ``data/raw/``

**Solution:** Ensure you have downloaded the dataset files:

* ``electricity_usage.csv``
* ``metadata.csv``
* ``weather.csv``


Running with Make
-----------------

You can also use the Makefile for common tasks:

.. code-block:: bash

   # Install dependencies
   make requirements

   # Run data processing
   make data

   # Clean compiled Python files
   make clean


Project Output Summary
----------------------

After running the complete pipeline, you should have:

**Data Files:**

* ``data/interim/merged_data.csv`` (97 MB)
* ``data/processed/features.csv`` (97 MB)
* ``data/processed/predictors.txt``

**Model Files:**

* ``models/random_forest.pkl`` (2.4 GB)
* ``models/metrics.txt``
* ``models/feature_importance.csv``

**Visualizations:**

* ``reports/figures/aggregated_usage_trend.png``
* ``reports/figures/sample_buildings_trend.png``
* ``reports/figures/feature_importance.png``


Next Steps
----------

After successfully running the pipeline:

1. **Review model metrics** in ``models/metrics.txt``
2. **Examine visualizations** in ``reports/figures/``
3. **Check feature importance** to understand key drivers
4. **Experiment with hyperparameters** to improve performance
5. **Try predictions** on your own building data


For More Information
--------------------

* **README:** Project overview and business context
* **Source Code:** ``src/`` directory contains all implementation
* **Documentation:** ``docs/`` directory for additional guides
* **Project Presentation:** See project slides for detailed analysis


Support
-------

If you encounter issues:

1. Check this guide for troubleshooting solutions
2. Review error messages carefully
3. Verify all dependencies are installed correctly
4. Ensure data files are in the correct locations


----
