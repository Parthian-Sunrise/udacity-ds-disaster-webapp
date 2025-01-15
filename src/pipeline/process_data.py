## INITIALISATION ##
print("Initialising")

# Set up correct root to directory to keep consist paths on local machine
import os
import sys

## Dynamically set the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

## Import modules from utils
from src.utils import get_root_dir, test_root_dir

## Check that root directory is set up properly
def test_root():
    assert test_root_dir() == "Root directory set up correctly!"

# Import packages
import pandas as pd 
from sqlalchemy import create_engine
import logging
import regex

# Configure logging params
logging.basicConfig(
    level=logging.INFO,  # Set level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{get_root_dir()}/process_data.log"),  # Log to a file
        logging.StreamHandler()  # Also log to the console
    ]
)

## EXTRACT ##
print("Extracting data")

# Set data paths 
msg_path = f'{get_root_dir()}/data/disaster_messages.csv'
cat_path = f'{get_root_dir()}/data/disaster_categories.csv'

paths = [msg_path,cat_path]

# Bring in dataframes 
dfs = []
for path in paths: 
    try:
        a = pd.read_csv(path)
        dfs.append(a)
    except FileNotFoundError as e:
        logging.error("File not found at path: %s", path)
        raise e
    except Exception as e:
        logging.error("An unexpected error occurred while loading file: %s", e)
        raise e

# Assign dataframes
msg_df = dfs[0]
cat_df = dfs[1]

## TRANSFORM ##
print("Transforming data")

# Join dataframes by ID variable
tot_df = pd.merge(left=msg_df,right=cat_df,how="inner",on="id")

# Expand concatenated category column using ';' separator
categories = cat_df["categories"].str.split(';',expand=True)

# Drop the concatenated column from the total dataframe
tot_df.drop(columns=['categories'],inplace=True)

# Extract the column names by using the first row as a template
row = categories.iloc[0]
col_names = [regex.search(r'([a-zA-Z_]+)-(\d+)',item).group(1) for item in row] # search for two items text (with or without a _ before the hyphon and the number afterwards), take the latter
categories.columns = col_names

# Now transform row values to only contain the boolean integer
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x : x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

# Add on transformed expanded columns to total dataframe 
tot_df = pd.concat([tot_df,categories],axis=1,sort=False)

# Print message to conosle
print("Data transformation complete, please check the dataframe below is in the desired format")
print(tot_df.head())

## LOAD ##
print("Loading data")
engine = create_engine(f'sqlite:///{get_root_dir()}/data/Disaster.db')
tot_df.to_sql('DISASTER_MESSAGES', engine, index=False)

## FINISH ##
print('終わり！')