# Exploratory Data Analysis of Online Shopping in Retail
This project analyses retail data based on online shopping experiences. 

## Table of contents:
* db_utils.py - file to run for project tasks 
* customer_activity_data.csv - fetched data from database 
* transformed_customer_activity.csv - transformed data from customer_activity_data.csv 
* eda.ipynb - notebook with EDA instructions - breakdown by task 

###  Installation Instructions:
User requires the following installations: 
    python3
    
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    import numpy as np
    import pandas as pd
    import plotly
    import plotly.express as px
    from scipy import stats
    import seaborn as sns
    import yaml

    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import QueuePool

### Usage:
To run the analysis in python, run: 
python3 db_utils.py

### Functions defined throughout the project:
- load_credentials > loads credentials from YAML file required to connect to RDS database 
- Within class RDSDatabaseConnector:
    - _create_connection_string > creates connection string from credentials
    - initialize_engine > generates connection string and initializes engine 
    - fetch_data > fetches data from the database 
    - save_data_to_csv > saves dataframe to csv file 
- Within class DataTransform: 
    - object_to_category_transform > converts object type columns to category type columns
    - get_transformed_df > returns transformed dataframe
- Within class DataFrameInfo:
    - get_data_types > displays data types and other information in dataframe
    - statistical_summary > displays statistical summary of dataframe
    - unique_values > displays unique values and counts for each column of dataframe
    - df_shape > displays shape of dataframe
- Within class DataFrameTransform: 
    - calc_null_percentage > calculates percentage of null values in each column of dataframe
    - impute_missing_values > imputes missing values based on column skewness
    - log_skew_transform > applies best transformation to reduce skewness (log, sqrt, boxcox) in given column
    - get_transformed_df > gets transformed dataframe
- Within class Plotter: 
    - plot_null_comparison > plots comparison of null values percentage before and after data imputation
    - plot_histogram > plots histograms for given column
- Within class FindSkew: 
    - identify_skew > identifies skewed columns based on threshold of 0.5
    - skew_hist > plots histograms for skewed columns

- Other functions:
    - box_plots > creates boxplot for numeric column in dataframe 
    - detect_outliers_iqr > detects outliers using interquartile range
    - plot_correlation_matrix > plots correlation matrix returning high correlation pairs
    - main > performs all required tasks within the project 

