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
import nltk
nltk.download(['punkt', 'wordnet'])

from sqlalchemy import create_engine
import pandas as pd 
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

## LOAD ##
print("Loading Data")
engine = create_engine(f'sqlite:///{get_root_dir()}/data/Disaster.db')

query = 'SELECT * FROM DISASTER_MESSAGES'

df = pd.read_sql(query,engine)

# Get features and predictive variables 
X_cols = "message"
Y_cols = [col for col in df.columns if col not in ['message','id','original','genre']]

# Drop NAs for Y_cols as we cannot predict for NaNs
df.dropna(how="any",subset=Y_cols,inplace=True)

# Assert Nans are gone
for col in [X_cols]+Y_cols:
    assert df[col].isna().sum() == 0

# Assign to X and Y
X = df[X_cols]
Y = df[Y_cols]

# Check there are no columns that are all one value as we cannot classify for these
constant_cols = [col for col in Y.columns if Y[col].nunique() == 1]
print(f"Dropping constant columns: {constant_cols}")

# Drop constant columns if present
Y = Y.drop(columns=constant_cols)

print("Data loaded")

## TRAIN MODEl ##
print("Initialise model objects")

# Define feature extraction process / preprocessing

def tokenize(text):
    """
    Tokenises and lemmatises input text.

    This function takes a string as input, tokenises it into individual words,
    lemmatises each word to its base form, converts them to lowercase, and removes
    any leading or trailing whitespace.

    Args:
        text (str): The input text to be tokenised and lemmatised.

    Returns:
        list: A list of clean, lemmatised tokens (words) from the input text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Initialise a logistic regression model
logreg = LogisticRegression()

# Define pipeline for our classification problem
pipeline = Pipeline([
    ('vect', CountVectorizer(max_features=5000)),  # Reduce complexity for run time
    ('tfidf', TfidfTransformer()),
    ('moc', MultiOutputClassifier(logreg)),
])

# Split data
print("Split data")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=42) # Fix random state for reproducibility

# Make grid search object
print("Generating parameter grid")

# Create parameter grid (had to use minimal combinations due to the hardware constraints of my local machine
param_grid = {
    'moc__estimator__C': [0.1, 1.0],  # Two values for C
    'moc__estimator__penalty': ['l2'],  # Use only 'l2' for simplicity
    'moc__estimator__solver': ['liblinear'],  # Keep only 'liblinear'
}

grid_search = RandomizedSearchCV(pipeline, param_grid,n_iter=4,cv=2,verbose=3) # Perform parallelisation and add verbosity to see progress, use randomised for quicker performance

# Fit grid
print("Fitting Data")
grid_search.fit(X_train,y_train)

# Find best parameters
print("Best Parameters:", grid_search.best_params_)

# Get results for best model
best_pipeline = grid_search.best_estimator_

# Predict on the test set
y_pred = best_pipeline.predict(X_test)

# Generate classification reports for each category with the best parameters
category_reports = {}
for i, category in enumerate(y_test.columns):
    print(f"Category: {category}")
    report = classification_report(y_test.iloc[:, i], y_pred[:, i])
    category_reports[category] = report
    print(report)

## SAVE
print("Saving best model")

# Save the best model using pickle
import pickle
import os

# Ensure the output directory exists
output_dir = os.path.join(ROOT_DIR, 'data/models')
os.makedirs(output_dir, exist_ok=True)

# Save the model
model_path = os.path.join(output_dir, 'best_model.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(best_pipeline, file)

print(f"Model saved to {model_path}")

## FINISH ##
print('終わり！')    