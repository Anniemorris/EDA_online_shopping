import yaml
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# Function to load credentials from credentials.yaml file
def load_credentials(file):
    with open(file, "r") as f:
        credentials = yaml.safe_load(f)
    return credentials

# Load credentials from the YAML file
credentials = load_credentials('credentials.yaml')
print(credentials)

# Create class to extract data from PostgreSQL RDS database 
class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.RDS_HOST = credentials['RDS_HOST']
        self.RDS_PASSWORD = credentials['RDS_PASSWORD']
        self.RDS_USER = credentials['RDS_USER']
        self.RDS_DATABASE = credentials['RDS_DATABASE']
        self.RDS_PORT = credentials['RDS_PORT']
        self.engine = None

    def init_engine(self):
        try:
            # Adjust the connection string for PostgreSQL (port 5432)
            connection_string = (
                f"postgresql+psycopg2://{self.RDS_USER}:{self.RDS_PASSWORD}@{self.RDS_HOST}:{self.RDS_PORT}/{self.RDS_DATABASE}"
            )
            # Create SQLAlchemy engine for PostgreSQL
            self.engine = create_engine(
                connection_string, 
                pool_pre_ping=True, 
                pool_recycle=3600, 
                poolclass=QueuePool
            )
            print("SQLAlchemy engine initialised successfully")
            return self.engine
        except Exception as e:
            raise ValueError(f"Failed to initialise: {e}")
    
    def fetch_data(self):
        if self.engine is None:
            raise ValueError("Engine not initialised, call init_engine() first.")
        
        query = "SELECT * FROM customer_activity LIMIT 100;"

        with self.engine.connect() as connection:
            # Use the text() function to make the query executable
            df = pd.read_sql(text(query), connection)

        return df 
    
    def save_data_to_csv(self, df, file_name):
        try:
            # Save the DataFrame to CSV
            df.to_csv(file_name, index=False)
            print(f"Data successfully saved to {file_name}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")

# Initialize the RDSDatabaseConnector
db_connector = RDSDatabaseConnector(credentials)

try:
    # Initialize engine
    engine = db_connector.init_engine()
    print("Engine successfully initialised.")
    
    # Test connection by running a simple query (SELECT 1)
    with engine.connect() as connection: 
        result = connection.execute(text("SELECT 1"))
        print("Test query result:", result.fetchone())
    
    # Fetch data from the database
    df = db_connector.fetch_data()
    print("Fetched data:", df)
    
    # Save data to CSV
    db_connector.save_data_to_csv(df, 'customer_activity_data.csv')

except Exception as e:
    print(f"An error occurred: {e}")

# task 3 - import csv to pandas df

customer_df = pd.read_csv("customer_activity_data.csv")
print(customer_df)