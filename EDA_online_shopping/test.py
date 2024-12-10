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


dataframe = DataTransform(customer_df)

### things that need changing - 0 administrator and 4 product_related to int, month 9 to date time?, the durations to time   

dataframe.object_to_category_transform(['operating_systems'])
dataframe.object_to_category_transform(['traffic_type'])
dataframe.object_to_category_transform(['browser'])
dataframe.object_to_category_transform(['visitor_type'])

## dataframe.float_to_int_transform(['administrative'])
## dataframe.float_to_int_transform(['product_related'])

print(customer_df.info())

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

    

df_info = DataFrameInfo()
#df_info.get_data_types(customer_df)
#df_info.statistical_summary(customer_df)
#df_info.unique_values(customer_df)
#df_info.df_shape(customer_df)

class DataFrameTransform:
    def null_values(self, dataframe):
        for column in dataframe.columns:
            print(dataframe[column].isna())
    
    def calc_null_percentage(self, dataframe):
        null_percentage = (dataframe.isna().sum() / len(dataframe)) * 100
        print(null_percentage)

df_null = DataFrameTransform()
df_null.calc_null_percentage(customer_df)
