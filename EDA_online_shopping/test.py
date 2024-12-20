import numpy as np
import pandas as pd
from scipy import stats


# task 3 - import csv to pandas df
customer_df = pd.read_csv("customer_activity_data.csv")
print(customer_df)

shape = customer_df.shape
print(f"This data has {shape[0]} rows and {shape[1]} columns.")

print(customer_df.info())

# converting data to different datatypes as needed 
class DataTransform: 
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def object_to_category_transform(self, columns):
        self.columns = columns 
       
        for column in columns:
            self.dataframe[column] = self.dataframe[column].astype('category')

    def float_to_int_transform(self, columns):
        for column in columns: 
            self.dataframe[column] = self.dataframe[column].astype('int64')

    def get_df(self):
        return self.dataframe

transformed_data_types = DataTransform(customer_df)

### things that need changing - 0 administrator and 4 product_related to int, month 9 to date time?, the durations to time   
transformed_data_types.object_to_category_transform(['operating_systems'])
transformed_data_types.object_to_category_transform(['traffic_type'])
transformed_data_types.object_to_category_transform(['browser'])
transformed_data_types.object_to_category_transform(['visitor_type'])

## dataframe.float_to_int_transform(['administrative'])
## dataframe.float_to_int_transform(['product_related'])

updated_data_types = transformed_data_types.get_df()
print(updated_data_types)

### create dataframeinfo class ###

class DataFrameInfo: 
    def get_data_types(self, dataframe):
        print(dataframe.info())

    def statistical_summary(self, dataframe):
        print(dataframe.describe())

    def unique_values(self, dataframe):
        for column in dataframe.columns:
            print(dataframe[column].value_counts())

    def df_shape(self, dataframe):
        shape = dataframe.shape
        print(f"This data has {shape[0]} rows and {shape[1]} columns.")

#df_info = DataFrameInfo()
#df_info.get_data_types(customer_df)
#df_info.statistical_summary(customer_df)
#df_info.unique_values(customer_df)
#df_info.df_shape(customer_df)

class DataFrameTransform:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def null_values(self, dataframe):
        for column in dataframe.columns:
            print(dataframe[column].isna())
    
    def calc_null_percentage(self, dataframe):
        null_percentage = (dataframe.isna().sum() / len(dataframe)) * 100
        print(null_percentage)
    
    def impute_values(self, dataframe):
        for column in self.dataframe.columns: 
            if self.dataframe[column].isnull().sum() >0: # check for missing values 
                skewness = self.dataframe[column].skew()
                if skewness >0:
                    print("Imputing column with median due to positive skew")
                    self.dataframe[column].fillna(self.dataframe[column].median(), inplace=True)
                else: 
                    print("Imputing column with mean due to negative skew")
                    self.dataframe[column].fillna(self.dataframe[column].mean(), inplace=True)

    def log_skew_transform(self, skewed_columns):
        transformations = {}

        for column in skewed_columns:
            if self.dataframe[column].dtype == "bool":
                print(f"Ignoring boolean column: {column}")
                continue

            original_skew = self.dataframe[column].skew()
            print(f"Original skew for {column}: {original_skew}")

            # Apply transformations and calculate new skew
            log_transformed = np.log1p(self.dataframe[column])  # log(1 + x) to handle zero values
            sqrt_transformed = np.sqrt(self.dataframe[column])
            boxcox_transformed, _ = stats.boxcox(self.dataframe[column] + 1)  # Adding 1 for zero values

            # Calculate skew for each transformation
            log_skew = log_transformed.skew()
            sqrt_skew = sqrt_transformed.skew()
            boxcox_skew = pd.Series(boxcox_transformed).skew()

            # Store the transformations that reduce skew the most
            skew_values = {
                "original": original_skew,
                "log": log_skew,
                "sqrt": sqrt_skew,
                "boxcox": boxcox_skew
            }
            
            # Select the transformation with the minimum skew
            best_transformation = min(skew_values, key=skew_values.get)
            transformations[column] = best_transformation
            
            # Choose the best transformation based on skew reduction
            if best_transformation == "log":
                self.dataframe[column] = log_transformed
            elif best_transformation == "sqrt":
                self.dataframe[column] = sqrt_transformed
            elif best_transformation == "boxcox":
                self.dataframe[column] = boxcox_transformed
            
            print(f"Best transformation for {column}: {best_transformation} with skew {skew_values[best_transformation]}")
        
        return transformations

    def check_nulls(self):
        return self.dataframe.isnull().sum()
    
    def get_df(self):
        return self.dataframe

df = updated_data_types

print(df.info())


df_null = DataFrameTransform(df)

before_null = df.isnull().sum()
print("Before Null",
       before_null)

df_null.calc_null_percentage(df)
df_null.impute_values(df)
transformed_df = df_null.get_df()
print(transformed_df)
print(transformed_df.info())

pd.set_option('display.max_rows', None)
print(transformed_df.isnull())
pd.reset_option('display.max_rows')

transformed_null = DataFrameTransform(transformed_df)
after_null = transformed_df.isnull().sum()
print("After Null", 
      after_null)

## administrative - all but one can be filled in w 0 - drop row 22
## admin _duration - some fill w 0, some take mean 
## both informational_duration can be filled in w 0 
## product_related rows should be dropped (missings have durations so important to EDA)
## product_duration cannot be filled w 0 - maybe imputed w mean 

import matplotlib
matplotlib.use('TkAgg')  # Try using the TkAgg backend for rendering
import matplotlib.pyplot as plt

class FindSkew:
    def __init__(self,dataframe):
        self.dataframe = dataframe

    def identify_skew(self, threshold=0.5):
        skewness = self.dataframe.skew(numeric_only=True)
        skewed_columns = skewness[skewness.abs() > threshold]
        return skewed_columns
    def skew_hist(self, skew_columns, plotter):
        for column in skew_columns.index:
            if self.dataframe[column].dtype =='bool':
                continue
            print(f"Visualising column: {column}")
            plotter.plot_histogram(self.dataframe[column], title=f"Histogram for {column}")



class Plotter:
    @staticmethod
    def plot_null_comparison(before_null, after_null, title="Null values pre and post imputation"):
        labels = before_null.index
        x = range(len(labels))

        # Plotting bar charts
        plt.figure(figsize=(10, 6))
        plt.bar(x, before_null.values, width=0.4, label="Before imputation", color="green")
        plt.bar(x, after_null.values, width=0.4, label="After imputation", color="blue", align="edge")

        # Labels and title
        plt.xlabel("Columns")
        plt.ylabel("Number of null values")
        plt.title(title)
        plt.xticks(x, labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_histogram(column_data, title):
        if column_data.dtype == 'bool':
            column_data = column_data.astype(int) # convert true/false to 1/0
        plt.figure(figsize=(8, 5))
        plt.hist(column_data, bins=30, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

# Create Plotter object and plot comparison
plotter = Plotter()
#plotter.plot_null_comparison(before_null, after_null)

skew_analysis = FindSkew(transformed_df)
skew_columns = skew_analysis.identify_skew(threshold=0.5)
#print(f"Skewed Columns: \n{skew_columns}")
#skew_analysis.skew_hist(skew_columns, plotter) 

df_transformer = DataFrameTransform(transformed_df)
df_transformer.log_skew_transform(skew_columns.index)

transformed_df = df_transformer.get_df()


print("Plotting updated histograms for transformed columns..")
#for column in skew_columns.index:
#    print(f"Visualising transformed column: {column}")
#    plotter.plot_histogram(transformed_df[column], title=f"Updated histogram for {column}")

#transformed_df.to_csv("transformed_customer_activity.csv", index=False)


# boxplot of all columns 
import plotly
import plotly.express as px
# create a boxplot of amount:
def box_plots(dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtype in ['int64', 'float64']:
            fig = px.box(dataframe, y=column, title = f"Boxplot for {column}")
            fig.update_layout(yaxis_title="Values", xaxis_title="")
            fig.show()
        else:
            print(f"Skipping non-numeric column: {column}")

#box_plots(transformed_df)



columns_to_exclude = ['month', 'operating_systems', 'browser', 'region', 'traffic_type', 'visitor_type', 'weekend', 'revenue']
numeric_data = transformed_df.drop(columns=columns_to_exclude)

# Define a function to identify outliers using IQR
def detect_outliers_iqr(df):
    outliers = {}
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Find outliers
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return outliers

# Detect outliers
outliers = detect_outliers_iqr(numeric_data)

# Print summary of outliers
#for col, outlier_vals in outliers.items():
#    print(f"{col}: {len(outlier_vals)} outliers detected.")

# Identify columns with outliers
columns_with_outliers = [col for col, vals in outliers.items() if len(vals) > 0]

# Transform outliers using the existing class
from scipy import stats

# Initialize the transformer
transformer_2 = DataFrameTransform(transformed_df)

# Apply log-skew transform to columns with outliers
transformations_2 = transformer_2.log_skew_transform(columns_with_outliers)

# Get the transformed dataframe
transformed_df_2 = transformer_2.get_df()

# Display summary of transformations
#print("Transformations applied to reduce outliers and skewness:")
#for col, transformation in transformations_2.items():
#    print(f"{col}: {transformation}")

#print(transformed_df_2.describe())
#box_plots(transformed_df_2)

### correlation matrix ### 
import seaborn as sns
numeric_data_transformed = transformed_df_2.drop(columns=columns_to_exclude)

correlation_threshold = 0.9
corr = numeric_data_transformed.corr()
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

high_correlation = (corr.abs() > correlation_threshold) & ~mask

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, 
            square=True, linewidths=.5, annot=False, cmap=cmap)
plt.yticks(rotation=0)
plt.title('Correlation Matrix of all Numerical Variables')

high_corr_pairs = [(col, corr.columns[row]) 
                   for row, col in zip(*np.where(high_correlation))]
#print(f"Highly correlated pairs above the threshold of {correlation_threshold}:")
#for pair in high_corr_pairs:
#    print(pair)

plt.show()

#print(numeric_data_transformed.info()) ## informational and informational duration are correlated above 0.9 threshold 


## milestone 4

weekend_sales = transformed_df_2.groupby('weekend')['revenue'].mean() * 100
print(weekend_sales)

# Visualize the results
weekend_sales.plot(kind='bar', title="Sales Proportion by Weekend (True/False)", ylabel="Percentage of Sales")
plt.show()

## yes more sales on weekends 

region_revenue = transformed_df_2.groupby('region')['revenue'].sum().sort_values(ascending=False)
print(region_revenue)

# Visualize the results
region_revenue.plot(kind='bar', title="Revenue by Region", ylabel="Total Revenue")
plt.show()

## n.amrica generates most revenue 

traffic_sales = transformed_df_2.groupby('traffic_type')['revenue'].mean() * 100
print(traffic_sales)

# Visualize the results
traffic_sales.plot(kind='bar', title="Sales Proportion by Traffic Type", ylabel="Percentage of Sales")
plt.show()


## instagram page has most traffic 

task_durations = transformed_df_2[['administrative_duration', 'product_related_duration', 'informational_duration']].sum()
task_percentages = (task_durations / task_durations.sum()) * 100
print(task_percentages)

# Visualize the results
task_percentages.plot(kind='pie', autopct='%1.1f%%', title="Percentage of Time Spent on Tasks")
plt.ylabel("")  # Optional to remove the ylabel
plt.show()

# most time spent on product related task 

average_task_durations = transformed_df_2[['informational_duration', 'administrative_duration']].mean()
print(average_task_durations)

# Visualize the results
average_task_durations.plot(kind='bar', title="Average Time Spent on Tasks", ylabel="Average Duration (Seconds)")
plt.show()

## dont think this answers the question but that is because the data transformations have been wrong 

monthly_sales = transformed_df_2.groupby('month')['revenue'].sum().sort_values(ascending=False)
print(monthly_sales)

# Visualize the results
monthly_sales.plot(kind='bar', title="Sales by Month", ylabel="Total Sales")
plt.show()

# May is biggest month in sales 









## windows used most 

## desktop used most 

# chrome used most 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping operating systems to device types
os_to_device = {
    'Android': 'Mobile',
    'iOS': 'Mobile',
    'Windows': 'Desktop',
    'MACOS': 'Desktop',
    'ChromeOS': 'Desktop'
}

# Map the operating_systems to device_type
transformed_df_2['device_type'] = transformed_df_2['operating_systems'].map(os_to_device)

# Count of operating systems and percentages
os_counts = transformed_df_2['operating_systems'].value_counts()
os_percentages = transformed_df_2['operating_systems'].value_counts(normalize=True) * 100

# Plot operating systems count
plt.figure(figsize=(10, 5))
os_counts.plot(kind='bar', title='Operating Systems Count')
plt.ylabel('Count')
plt.show()

# Plot operating systems percentage as a pie chart
plt.figure(figsize=(8, 8))
os_percentages.plot(kind='pie', autopct='%1.1f%%', title='Operating Systems Percentage', ylabel='')
plt.show()

# Count of mobile vs. desktop users
device_counts = transformed_df_2['device_type'].value_counts()

# Plot mobile vs. desktop usage
plt.figure(figsize=(8, 6))
device_counts.plot(kind='bar', title='Mobile vs Desktop Usage', color=['skyblue', 'orange'])
plt.ylabel('Count')
plt.show()

# Most commonly used browsers with breakdown by device type
browser_breakdown = transformed_df_2.groupby(['browser', 'device_type']).size().unstack()

# Plot browser breakdown
browser_breakdown.plot(kind='bar', stacked=True, figsize=(10, 6), title='Browser Breakdown by Device Type')
plt.ylabel('Count')
plt.show()

# Identify regions with discrepancies in popular operating systems
regional_popular_os = transformed_df_2.groupby('region')['operating_systems'].agg(lambda x: x.value_counts().idxmax())
regional_discrepancies = regional_popular_os[regional_popular_os != regional_popular_os.mode()[0]]

# Display discrepancies
print("Regions with discrepancies:")
print(regional_discrepancies)

# Fix: Plot discrepancies as a bar chart by converting the values to counts
plt.figure(figsize=(10, 5))
regional_discrepancies.value_counts().plot(kind='bar', title="Regions with Discrepant Popular Operating Systems", color='coral')
plt.ylabel('Count')
plt.xlabel('Operating Systems')
plt.show()






# Grouping revenue by region and traffic type
revenue_by_region = transformed_df_2.groupby(['region', 'traffic_type'])['revenue'].sum().unstack()

# Plot revenue by region and traffic type
plt.figure(figsize=(12, 6))
revenue_by_region.plot(kind='bar', stacked=True, figsize=(12, 6), title="Revenue by Traffic Type and Region")
plt.ylabel('Total Revenue')
plt.xlabel('Region')
plt.legend(title='Traffic Type')
plt.show()

# Grouping average bounce rates by region and traffic type
bounce_rate_by_region = transformed_df_2.groupby(['region', 'traffic_type'])['bounce_rates'].mean().unstack()

# Plot bounce rate by region and traffic type
plt.figure(figsize=(12, 6))
bounce_rate_by_region.plot(kind='bar', figsize=(12, 6), title="Bounce Rate by Traffic Type and Region")
plt.ylabel('Average Bounce Rate (%)')
plt.xlabel('Region')
plt.legend(title='Traffic Type')
plt.show()

# Filter for ads-related traffic (any traffic type containing 'ads')
ads_traffic_data = transformed_df_2[transformed_df_2['traffic_type'].str.contains('ads', case=False, na=False)]

# Group by month and sum revenue
monthly_sales_from_ads = ads_traffic_data.groupby('month')['revenue'].sum()

# Plot monthly sales from ads traffic
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
# Assuming 'visitor_type' column has 'New Visitor' and 'Returning Visitor'
visitor_purchase = transformed_df_2.groupby('visitor_type')['revenue'].sum() / transformed_df_2.groupby('visitor_type')['revenue'].count() * 100
print(visitor_purchase)

# Visualize the purchase percentage by visitor type
visitor_purchase.plot(kind='bar', title="Purchase Percentage by Visitor Type", color='lightblue')
plt.ylabel('Percentage of Purchases (%)')
plt.show()

# 3. Are sales being made more on weekends comparatively to weekdays?
# Assuming 'weekend' is a boolean column (True for weekend, False for weekday)
weekend_sales = transformed_df_2.groupby('weekend')['revenue'].sum()
print(weekend_sales)

# Visualize the results
weekend_sales.plot(kind='bar', title="Sales by Weekend (True/False)", color='lightblue')
plt.ylabel('Total Revenue')
plt.show()

# 4. Which months have been the most effective for generating sales?
monthly_sales = transformed_df_2.groupby('month')['revenue'].sum().sort_values(ascending=False)
print(monthly_sales)

# Visualize the results
monthly_sales.plot(kind='bar', title="Sales by Month", color='mediumseagreen')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()

# 5. Is direct/social or advertising traffic contributing heavily to sales?
# Assuming 'traffic_type' column indicates traffic source (like 'Google search', 'Instagram ads', etc.)
# We will filter traffic types that include 'ads' and compare sales
ads_traffic_data = transformed_df_2[transformed_df_2['traffic_type'].str.contains('ads', case=False, na=False)]

# Sum the revenue from ads traffic
ads_traffic_sales = ads_traffic_data.groupby('traffic_type')['revenue'].sum()
print(ads_traffic_sales)

# Visualize the revenue contribution by ads-related traffic
ads_traffic_sales.plot(kind='bar', title="Sales by Ads Traffic Type", color='lightcoral')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.show()

