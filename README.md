# Udacity Data Science Nanodegree Disaster Project Pipeline

This repository contains the code and analysis for the disaster pipeline project, including a script for the loading of data, training of a model, optimisation by grid search and then displaying in a web app using the Udacity template.

## Project Overview

I load the data and create a SQLite database, I then train a classifier using logistic regression on the tokenised transformation of text data to predict the category of a message (using pre-labelled data to train and test).

## Repository Structure

- `src/`: Python scripts, the pipeline subfolder contains the project code.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `requirements.txt`: Lists the Python package dependencies required for the project.

## Data Overview

Please find sources and acknowledgements for data in the data folder with more information on the data origin and use. 

## Licence

This code may be used however you wish, with or without crediting me. 

## Acknowledgements

All packages are acknowledge in the notebook as well as their current version, data is acknowledged in the readme in the data folder.

## Getting Started

To replicate the analysis or run the code locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Parthian-Sunrise/udacity-ds-project-blog.git
2. **Navigate to the repository:**
   ```bash
   cd udacity-ds-disaster-web-app
3. **Create virtual environment (optrional)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
4. **Install all requirements**
  ```bash
   pip install -r requirements.txt
  ```
5. **Set up pre-commits**
   ```bash
   pre-commit install
   ```
Now launch in any IDE you wish
