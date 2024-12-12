import pandas as pd

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

#print(f"Skew of administrative column is {df['administrative'].skew()}")
#print(f"Skew of administrative_duration column is {df['administrative_duration'].skew()}")
#print(f"Skew of informational_duration column is {df['informational_duration'].skew()}")
#print(f"Skew of product_related column is {df['product_related'].skew()}")
#print(f"Skew of product_related_duration column is {df['product_related_duration'].skew()}")


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




# Simple plot for testing
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]
plt.plot(x, y)
plt.show()

xxx

# Create Plotter object and plot comparison
plotter = Plotter()
plotter.plot_null_comparison(before_null, after_null)