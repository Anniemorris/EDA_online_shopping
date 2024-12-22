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



def load_credentials(file):
    """
    Function loads credentials from YAML file required to connect to RDS database. 

    Args: 
    - file (str): path to YAML file 
    Returns: 
    - dict: credentials as dictionary 
    """
    with open(file, "r") as f:
        return yaml.safe_load(f) 

credentials = load_credentials('credentials.yaml')
print(credentials)

# Create class to extract data from PostgreSQL RDS database 
class RDSDatabaseConnector:
    """
    Class uses credentials to initialize a database, fetch data and then save to csv. 
    """
    def __init__(self, credentials):
        """
        Initializes database connector with credentials. 

        Args: 
        - credentials (dict): multiple RDS keys 
        """
        self.RDS_HOST = credentials['RDS_HOST']
        self.RDS_PASSWORD = credentials['RDS_PASSWORD']
        self.RDS_USER = credentials['RDS_USER']
        self.RDS_DATABASE = credentials['RDS_DATABASE']
        self.RDS_PORT = credentials['RDS_PORT']
        self.engine = None

    def _create_connection_string(self):
        """
        Creates connection string from credentials. 

        Returns: 
        - str: valid PostgreSQL connection string
        """
        return (
            f"postgresql+psycopg2://{self.RDS_USER}:{self.RDS_PASSWORD}@{self.RDS_HOST}:{self.RDS_PORT}/{self.RDS_DATABASE}"
        )

    def initialize_engine(self):
        """
        Initialize SQLAlchemy engine using credentials. 

        Method generates connection string and initializes engine.

        Raises:
        - ValueError: if fails to initialize.
        """ 
        connection_string = self._create_connection_string()
        try:
            self.engine = create_engine(
                connection_string, 
                pool_pre_ping=True, 
                pool_recycle=3600, 
                poolclass=QueuePool
            )
            print("SQLAlchemy engine initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize engine: {e}")
    
    def fetch_data(self, query = "SELECT * FROM customer_activity LIMIT 100;"):
        """
        Fetches data from the database with defined query. 

        Args: 
        - query (str): default as "SELECT * FROM customer_activity LIMIT 100;"

        Returns: 
        - pd.DataFrame: data from database as pandas dataframe 

        Raises:
        - ValueError: if engine has not been initialized. 
        """
        if self.engine is None:
            raise ValueError("Engine not initialized, call initialize_engine() first.")
        
        with self.engine.connect() as connection:
            df = pd.read_sql(text(query), connection)

        return df 
    
    def save_data_to_csv(self, df, file_name):
        """
        Saves dataframe to csv file. 

        Args: 
        - df(pd.DataFrame): dataframe to save
        - file_name(str): name of file to save data to 

        Raises: 
        - ValueError: where unable to save dataframe to csv
        """
        try:
            df.to_csv(file_name, index=False)
            print(f"Data successfully saved to {file_name}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")



db_connector = RDSDatabaseConnector(credentials) # initializes RDSDatabaseConnector 

try:
    db_connector.initialize_engine()
    print("Engine successfully initialised.")
    
    with db_connector.engine.connect() as connection: # tests connection with simple query 
        result = connection.execute(text("SELECT 1"))
        print("Test query result:", result.fetchone())
    
    df = db_connector.fetch_data()
    print("Fetched data:", df)
    
    db_connector.save_data_to_csv(df, 'customer_activity_data.csv')

except Exception as e:
    print(f"An error occurred: {e}")

customer_df = pd.read_csv("customer_activity_data.csv")
print(customer_df)

shape = customer_df.shape
print(f"This data has {shape[0]} rows and {shape[1]} columns.")

print(customer_df.info())




# converting data to different datatypes as needed 
class DataTransform: 
    """
    Class handles data transformations within the dataframe. 
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes dataframe to transform. 

        Args: 
        - dataframe(pd.DataFrame): dataframe to transform 
        """
        self._dataframe = dataframe

    def object_to_category_transform(self, columns):
        """
        Converts object type columns to category type columns. 

        Args: 
        - columns(List[str]): lists column names to be converted 
        """

        for column in columns:
            self._dataframe[column] = self._dataframe[column].astype('category')

    def get_transformed_df(self) -> pd.DataFrame:
        """
        Function returns transformed dataframe. 
        """
        return self._dataframe

class DataFrameInfo: 
    """
    Class gets basic info about the dataframe like data types, summary stats, unique values and shape. 
    """
    @staticmethod
    def get_data_types(dataframe: pd.DataFrame):
        """
        Displays data types of dataframe. 

        Args: 
        - dataframe(pd.DataFrame): dataframe that will display information 
        """
        print(dataframe.info())

    @staticmethod
    def statistical_summary(dataframe: pd.DataFrame):
        """
        Displays statistical summary of dataframe. 

        Args: 
        - dataframe(pd.DataFrame): dataframe that will display information 
        """    
        print(dataframe.describe())

    @staticmethod
    def unique_values(dataframe: pd.DataFrame):
        """
        Displays unique values and counts for each column of dataframe. 

        Args: 
        - dataframe(pd.DataFrame): dataframe that will display information 
        """
        for column in dataframe.columns:
            print(dataframe[column].value_counts())

    @staticmethod
    def df_shape(dataframe: pd.DataFrame):
        """
        Displays shape of dataframe. 

        Args: 
        - dataframe(pd.DataFrame): dataframe that will display information 
        """
        shape = dataframe.shape
        print(f"This data has {shape[0]} rows and {shape[1]} columns.")

class DataFrameTransform:
    """
    Class performs dataframe transformations. 
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes dataframe. 

        Args:
        - dataframe(pd.DataFrame): dataframe to be transformed 
        """
        self._dataframe = dataframe
   
    def calc_null_percentage(self):
        """
        Calculates percentage of null values in each column of dataframe. 

        Returns: 
        - pd.Series: column names as index and percentage of null values as values
        """
        return (self._dataframe.isna().sum() / len(self._dataframe)) * 100
    
    def impute_missing_values(self):
        """
        Function imputes missing values based on column skewness.
        Uses median to impute in positively skewed columns and mean for negatively skewed. 

        Modifies dataframe in place.
        """
        for column in self._dataframe.columns: 
            if self._dataframe[column].isnull().sum() >0: 
                skewness = self._dataframe[column].skew()
                if skewness >0:
                    print("Imputing column with median due to positive skew")
                    self._dataframe[column].fillna(self._dataframe[column].median(), inplace=True)
                else: 
                    print("Imputing column with mean due to negative skew")
                    self._dataframe[column].fillna(self._dataframe[column].mean(), inplace=True)

    def log_skew_transform(self, skewed_columns):
        """
        Applies best transformation to reduce skewness (log, sqrt, boxcox) in given column.

        Args: 
        - skewed_columns(List[str]): list of skewed column names to transform 
        """
        transformations = {}

        for column in skewed_columns:
            if self._dataframe[column].dtype == "bool":
                print(f"Ignoring boolean column: {column}")
                continue

            original_skew = self._dataframe[column].skew()
            print(f"Original skew for {column}: {original_skew}")
            
            log_transformed = np.log1p(self._dataframe[column]) 
            sqrt_transformed = np.sqrt(self._dataframe[column])
            boxcox_transformed, _ = stats.boxcox(self._dataframe[column] + 1)  

            # Calculates skew for each transformation
            log_skew = log_transformed.skew()
            sqrt_skew = sqrt_transformed.skew()
            boxcox_skew = pd.Series(boxcox_transformed).skew()

            # Stores transformations that reduce skew the most
            skew_values = {
                "original": original_skew,
                "log": log_skew,
                "sqrt": sqrt_skew,
                "boxcox": boxcox_skew
            }
            
            # Selects transformation with minimum skew
            best_transformation = min(skew_values, key=skew_values.get)
            transformations[column] = best_transformation
            
            # Chooses best transformation based on skew reduction
            if best_transformation == "log":
                self._dataframe[column] = log_transformed
            elif best_transformation == "sqrt":
                self._dataframe[column] = sqrt_transformed
            elif best_transformation == "boxcox":
                self._dataframe[column] = boxcox_transformed
            
            print(f"Best transformation for {column}: {best_transformation} with skew {skew_values[best_transformation]}")
        
        return transformations
    
    def get_transformed_df(self):
        """
        Function gets transformed dataframe. 

        Returns: 
        - dataframe(pd.DataFrame): transformed dataframe
        """
        return self._dataframe

class Plotter:
    """
    Class handles plotting data.
    """
    @staticmethod
    def plot_null_comparison(before_null, after_null, title="Null values pre and post imputation"):
        """
        Plots comparison of null values percentage before and after data imputation. 

        Args: 
        - before_null(pd.Series): panda series before data imputation  
        - after_null(pd.Series): panda series after data imputation 
        """
        labels = before_null.index
        x = range(len(labels))

        plt.figure(figsize=(10, 6))
        plt.bar(x, before_null.values, width=0.4, label="Before imputation", color="green")
        plt.bar(x, after_null.values, width=0.4, label="After imputation", color="blue", align="edge")

        plt.xlabel("Columns")
        plt.ylabel("Number of null values")
        plt.title(title)
        plt.xticks(x, labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_histogram(column_data, title):
        """
        Plots histograms for given column. 

        Args: 
        - column_data(pd.Series): data to plot as histogram
        - title(str): title of the histogram
        """
        if column_data.dtype == 'bool':
            column_data = column_data.astype(int) # converts true/false to 1/0
        plt.figure(figsize=(8, 5))
        plt.hist(column_data, bins=30, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

class FindSkew:
    """
    Class identifies skewed columns and plots histograms for them.
    """
    def __init__(self,dataframe):
        """
        Initializes dataframe to handle skewed columns. 

        Args: 
        - dataframe(pd.DataFrame): dataframe to analyse for skewed columns
        """
        self._dataframe = dataframe

    def identify_skew(self, threshold=0.5):
        """
        Identifies skewed columns based on threshold of 0.5.

        Args: 
        - threshold(float): skewness threshold 

        Returns: 
        - skewed_columns(pd.Series): column names as index and skewness values as values 
        """
        skewness = self._dataframe.skew(numeric_only=True)
        skewed_columns = skewness[skewness.abs() > threshold]
        return skewed_columns
    
    def skew_hist(self, skew_columns, plotter):
        """
        Plots histograms for skewed columns. 

        Args: 
        - skew_columns(pd.Series): skewed columns to be plotted 
        - plotter(Plotter): plotter instance to plot histograms 
        """
        for column in skew_columns.index:
            if self._dataframe[column].dtype =='bool':
                continue
            plotter.plot_histogram(self._dataframe[column], title=f"Histogram for {column}")

def box_plots(dataframe):
    """
    Creates boxplot for numeric column in dataframe. 

    Args: 
    - dataframe(pd.DataFrame): dataframe to apply boxplot function 
    """
    for column in dataframe.columns:
        if dataframe[column].dtype in ['int64', 'float64']:
            fig = px.box(dataframe, y=column, title = f"Boxplot for {column}")
            fig.update_layout(yaxis_title="Values", xaxis_title="")
            fig.show()
        else:
            print(f"Skipping non-numeric column: {column}")

def detect_outliers_iqr(df):
    """
    Function detects outliers using interquartile range. 

    Args: 
    - df(pd.DataFrame): dataframe to apply boxplot function 

    Returns: 
    - outliers(dict): column as key and value as outlier value outside of interquartile range 
    """
    outliers = {}
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return outliers

def plot_correlation_matrix(df, correlation_threshold = 0.9):
    """
    Plots correlation matrix returning high correlation pairs. 

    Args: 
    - df(pd.DataFrame): dataframe to calculate correlations on 
    - correlation_threshold(float): threshold which correlatoin is considered high 

    Returns: 
    - high_corr_pairs (list): list of tuples with pairs of high correlated columns 
    """
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    high_correlation = (corr.abs() > correlation_threshold) & ~mask

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, square=True, linewidths=.5, annot=False, cmap=cmap)
    plt.yticks(rotation=0)
    plt.title('Correlation Matrix of all Numerical Variables')
    plt.show()

    high_corr_pairs = [(col, corr.columns[row]) 
                                          for row, col in zip(*np.where(high_correlation))]

    return high_corr_pairs

def main(): 
    """
    Main function to perform all tasks in project.
    """
    transformed_data_types = DataTransform(customer_df)
    transformed_data_types.object_to_category_transform(['operating_systems', 'traffic_type', 'browser', 'visitor_type'])

    updated_data_types = transformed_data_types.get_transformed_df()

    df_info = DataFrameInfo()
    df_info.get_data_types(updated_data_types)
    df_info.statistical_summary(updated_data_types)
    df_info.unique_values(updated_data_types)
    df_info.df_shape(updated_data_types)

    df = updated_data_types
    df_transformer = DataFrameTransform(df)
    before_null = df.isnull().sum()
    print("Before Null:", before_null)

    null_percentage = df_transformer.calc_null_percentage()
    print("Null percentage:", before_null)

    df_transformer.impute_missing_values()
    after_null = df_transformer.get_transformed_df().isnull().sum()
    print("After Null:", after_null)

    plotter = Plotter()
    plotter.plot_null_comparison(before_null, after_null)

    skew_analysis = FindSkew(df_transformer.get_transformed_df())
    skew_columns = skew_analysis.identify_skew(threshold=0.5)
    print(f"Skewed Columns: \n{skew_columns}")
    skew_analysis.skew_hist(skew_columns, plotter) 

    df_transformer.log_skew_transform(skew_columns.index)
    transformed_df = df_transformer.get_transformed_df()

    print("Plotting updated histograms for transformed columns..")
    for column in skew_columns.index:
        plotter.plot_histogram(transformed_df[column], title=f"Updated histogram for {column}")

    transformed_df.to_csv("transformed_customer_activity.csv", index=False)

    box_plots(transformed_df)

    columns_to_exclude = ['month', 'operating_systems', 'browser', 'region', 'traffic_type', 'visitor_type', 'weekend', 'revenue']
    numeric_data = transformed_df.drop(columns=columns_to_exclude)
   
    outliers = detect_outliers_iqr(numeric_data)
    for col, outlier_vals in outliers.items():
        print(f"{col}: {len(outlier_vals)} outliers detected.")

    columns_with_outliers = [col for col, vals in outliers.items() if len(vals) > 0]

    transformer_2 = DataFrameTransform(transformed_df)
    transformations_2 = transformer_2.log_skew_transform(columns_with_outliers)

    transformed_df_2 = transformer_2.get_transformed_df()
    numeric_data_transformed = transformed_df_2.drop(columns=columns_to_exclude)

    print("Transformations applied to reduce outliers and skewness:")
    for col, transformation in transformations_2.items():
        print(f"{col}: {transformation}")

    print(transformed_df_2.describe())
    box_plots(transformed_df_2)

    high_corr_pairs = plot_correlation_matrix(numeric_data_transformed)

    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"High correlation between: {pair[0]} and {pair[1]}")
    else:
        print("No highly correlated columns found.")

    weekend_sales = transformed_df_2.groupby('weekend')['revenue'].mean() * 100
    print(weekend_sales)

    weekend_sales.plot(kind='bar', title="Sales Proportion by Weekend (True/False)", ylabel="Percentage of Sales")
    plt.show()

    region_revenue = transformed_df_2.groupby('region')['revenue'].sum().sort_values(ascending=False)
    print(region_revenue)   

    region_revenue.plot(kind='bar', title="Revenue by Region", ylabel="Total Revenue")
    plt.show()

    traffic_sales = transformed_df_2.groupby('traffic_type')['revenue'].mean() * 100 # N.America generates most revenue 
    print(traffic_sales)

    traffic_sales.plot(kind='bar', title="Sales Proportion by Traffic Type", ylabel="Percentage of Sales") # Instagram pagegets most traffic
    plt.show()

    task_durations = transformed_df_2[['administrative_duration', 'product_related_duration', 'informational_duration']].sum()
    task_percentages = (task_durations / task_durations.sum()) * 100
    print(task_percentages)

    task_percentages.plot(kind='pie', autopct='%1.1f%%', title="Percentage of Time Spent on Tasks") # most time was spent on product related tasks 
    plt.ylabel("")  
    plt.show()

    average_task_durations = transformed_df_2[['informational_duration', 'administrative_duration']].mean()
    print(average_task_durations)

    average_task_durations.plot(kind='bar', title="Average Time Spent on Tasks", ylabel="Average Duration (Seconds)") ## dont think this answers the question but that is because the data transformations have been wrong 
    plt.show()

    monthly_sales = transformed_df_2.groupby('month')['revenue'].sum().sort_values(ascending=False) # May is the biggest month in sales 
    print(monthly_sales)

    monthly_sales.plot(kind='bar', title="Sales by Month", ylabel="Total Sales")
    plt.show()

    os_to_device = {
        'Android': 'Mobile',
        'iOS': 'Mobile',
        'Windows': 'Desktop',
        'MACOS': 'Desktop',
        'ChromeOS': 'Desktop'
    }

    transformed_df_2['device_type'] = transformed_df_2['operating_systems'].map(os_to_device)

    os_counts = transformed_df_2['operating_systems'].value_counts()
    os_percentages = transformed_df_2['operating_systems'].value_counts(normalize=True) * 100

    plt.figure(figsize=(10, 5))
    os_counts.plot(kind='bar', title='Operating Systems Count')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(8, 8))
    os_percentages.plot(kind='pie', autopct='%1.1f%%', title='Operating Systems Percentage', ylabel='') # windows used most 
    plt.show()

    device_counts = transformed_df_2['device_type'].value_counts()

    plt.figure(figsize=(8, 6))
    device_counts.plot(kind='bar', title='Mobile vs Desktop Usage', color=['skyblue', 'orange']) # desktop used most 
    plt.ylabel('Count')
    plt.show()   

    browser_breakdown = transformed_df_2.groupby(['browser', 'device_type']).size().unstack()

    browser_breakdown.plot(kind='bar', stacked=True, figsize=(10, 6), title='Browser Breakdown by Device Type') # chrome used most 
    plt.ylabel('Count')
    plt.show()

    regional_popular_os = transformed_df_2.groupby('region')['operating_systems'].agg(lambda x: x.value_counts().idxmax())
    regional_discrepancies = regional_popular_os[regional_popular_os != regional_popular_os.mode()[0]]

    print("Regions with discrepancies:")
    print(regional_discrepancies)

    plt.figure(figsize=(10, 5))
    regional_discrepancies.value_counts().plot(kind='bar', title="Regions with Discrepant Popular Operating Systems", color='coral')
    plt.ylabel('Count')
    plt.xlabel('Operating Systems')
    plt.show()

    revenue_by_region = transformed_df_2.groupby(['region', 'traffic_type'])['revenue'].sum().unstack()

    plt.figure(figsize=(12, 6))
    revenue_by_region.plot(kind='bar', stacked=True, figsize=(12, 6), title="Revenue by Traffic Type and Region")
    plt.ylabel('Total Revenue')
    plt.xlabel('Region')
    plt.legend(title='Traffic Type')
    plt.show()

    bounce_rate_by_region = transformed_df_2.groupby(['region', 'traffic_type'])['bounce_rates'].mean().unstack()

    plt.figure(figsize=(12, 6))
    bounce_rate_by_region.plot(kind='bar', figsize=(12, 6), title="Bounce Rate by Traffic Type and Region")
    plt.ylabel('Average Bounce Rate (%)')
    plt.xlabel('Region')
    plt.legend(title='Traffic Type')
    plt.show()

    ads_traffic_data = transformed_df_2[transformed_df_2['traffic_type'].str.contains('ads', case=False, na=False)]

    monthly_sales_from_ads = ads_traffic_data.groupby('month')['revenue'].sum()

    plt.figure(figsize=(10, 6))
    monthly_sales_from_ads.plot(kind='line', marker='o', figsize=(10, 6), title="Monthly Sales from Ads Traffic")
    plt.ylabel('Total Revenue')
    plt.xlabel('Month')
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()

    # 1. Which region is currently generating the most/least revenue?
    region_revenue = transformed_df_2.groupby('region')['revenue'].sum().sort_values(ascending=False)
    print(region_revenue)

    # Visualize the results for revenue by region
    plt.figure(figsize=(10, 6))
    region_revenue.plot(kind='bar', title="Revenue by Region", color='lightcoral')
    plt.xlabel('Region')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.show()

    # 2. What percentage of our returning/new customers are making a purchase when they visit the site?
    visitor_purchase = transformed_df_2.groupby('visitor_type')['revenue'].sum() / transformed_df_2.groupby('visitor_type')['revenue'].count() * 100
    print(visitor_purchase)

    visitor_purchase.plot(kind='bar', title="Purchase Percentage by Visitor Type", color='lightblue')
    plt.ylabel('Percentage of Purchases (%)')
    plt.show()

    # 3. Are sales being made more on weekends comparatively to weekdays?
    weekend_sales = transformed_df_2.groupby('weekend')['revenue'].sum()
    print(weekend_sales)

    weekend_sales.plot(kind='bar', title="Sales by Weekend (True/False)", color='lightblue')
    plt.ylabel('Total Revenue')
    plt.show()

    # 4. Which months have been the most effective for generating sales?
    monthly_sales = transformed_df_2.groupby('month')['revenue'].sum().sort_values(ascending=False)
    print(monthly_sales)

    monthly_sales.plot(kind='bar', title="Sales by Month", color='mediumseagreen')
    plt.xlabel('Month')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.show()

    # 5. Is direct/social or advertising traffic contributing heavily to sales?
    ads_traffic_data = transformed_df_2[transformed_df_2['traffic_type'].str.contains('ads', case=False, na=False)]

    ads_traffic_sales = ads_traffic_data.groupby('traffic_type')['revenue'].sum()
    print(ads_traffic_sales)

    ads_traffic_sales.plot(kind='bar', title="Sales by Ads Traffic Type", color='lightcoral')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    """
    Entry point for running the script when executed directly.
    """
    main()




