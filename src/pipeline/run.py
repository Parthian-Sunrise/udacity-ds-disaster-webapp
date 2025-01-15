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

## USING UDACITY TEMPLATE AS INSTRUCTED

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine(f'sqlite:///{get_root_dir()}/data/Disaster.db')

query = 'SELECT * FROM DISASTER_MESSAGES'

df = pd.read_sql(query,engine)

# load model
model = joblib.load(f"{get_root_dir()}/data/models/best_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Create category count distribution
    category_counts = df[[col for col in df.columns if col not in ['message','id','original','genre']]].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # Display F1 scores for each category by predicting on test split from train_classifier, we use the same data processing from the train_classifier.py file
    X_cols = "message"
    Y_cols = [col for col in df.columns if col not in ['message','id','original','genre']]
    
    # Drop NAs for Y_cols 
    df.dropna(how="any",subset=Y_cols,inplace=True)
    
    for col in [X_cols]+Y_cols:
        assert df[col].isna().sum() == 0
    
    # Assign to X and Y
    
    X = df[X_cols]
    Y = df[Y_cols]
    
    # Check there are no columns that are all one value
    constant_cols = [col for col in Y.columns if Y[col].nunique() == 1]
    print(f"Dropping constant columns: {constant_cols}")
    
    Y = Y.drop(columns=constant_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

    y_pred = model.predict(X_test)

    f1_scores = {}

    plot2_cats = list(Y.columns)
    for i, category in enumerate(plot2_cats):
        # Generate the classification report as a dictionary
        report = classification_report(y_test.iloc[:, i], y_pred[:, i], output_dict=True)
        
        # Extract the F1 score for the positive class (label '1')
        f1_score_category = report['1.0']['f1-score']
        
        # Store the F1 score in the dictionary
        f1_scores[category] = f1_score_category

    
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(f1_scores.keys()),
                    y=list(f1_scores.values())
                )
            ],

            'layout': {
                'title': 'Distribution of F1 Scores for Categories (Random State 42, frac=0.3)',
                'yaxis': {
                    'title': "F1 Score"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()