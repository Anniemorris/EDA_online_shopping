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
plotter.plot_null_comparison(before_null, after_null)

skew_analysis = FindSkew(transformed_df)
skew_columns = skew_analysis.identify_skew(threshold=0.5)
#print(f"Skewed Columns: \n{skew_columns}")
#skew_analysis.skew_hist(skew_columns, plotter) 

df_transformer = DataFrameTransform(transformed_df)
df_transformer.log_skew_transform(skew_columns.index)

transformed_df = df_transformer.get_df()


print("Plotting updated histograms for transformed columns..")
#for column in skew_columns.index:
    #print(f"Visualising transformed column: {column}")
    #plotter.plot_histogram(transformed_df[column], title=f"Updated histogram for {column}")

transformed_df.to_csv("transformed_customer_activity.csv", index=False)
