# Udacity Data Science Nanodegree Blog Post Project

This repository contains the code and analysis for the blog post project of Udacity's Data Science Nanodegree. The project involves exploring a dataset, performing data analysis, and communicating findings through a blog post.

## Project Overview

The goal of this project is to apply data science techniques to a chosen dataset, derive meaningful answers to a set of 3-5 questions, and present them in a clear and engaging manner. The analysis is documented in a Jupyter Notebook, and the findings are summarised in a blog post on [Medium](https://medium.com/@judeking_21019/a-look-at-the-tokyo-2023-2024-airbnb-datasets-4abe53b6c0eb).

## Repository Structure

- `notebooks/`: Contains Jupyter Notebooks with EDA and the analysis of the questions
- `src/`: Python functions to ease importation of data 
- `.gitignore`: Specifies files and directories to be ignored by git.
- `.pre-commit-config.yaml`: Configuration for pre-commit hooks.
- `requirements.txt`: Lists the Python package dependencies required for the project.

## Data Overview

Please find sources and acknowledgements for data in the data folder with more information on the data origin and use. 

## Licence

This code may be used however you wish, with or without crediting me. 

## Acknowledgements

All packages are acknowledge in the notebook as well as their current version, data is acknowledge in the readme in the data folder.

## Summary of Results 

I had 4 questions I sought to answer, I will provide a short summary for each of these questions based on my findings in the notebook. 

### Questions 

- `Q1:` How much do neighbourhoods affect the price of AirBnbs?

- `Q2:` 花見 (hanami) is the Japanese practice of watching cherry blossoms bloom. In Japan, due to a previous blight among the tree population, around 80% of cherry blossoms are genetically identical, meaning the vast majority will bloom under the same conditions. Forecasts exist and are well known for the date of mass blooming. Does the date of blooming seem to impact the price?

- `Q3:` Which features of the dataset have the best predictive power for the price of a property based on logistic regression?

- `Q4:` What are the features that best predict superhost status?

### Discovery Summary

- `Neighbourhood Impact:` Prices vary dramatically between neighbourhoods, with some, like Arakawa, significantly pricier than others like Higashiyamato. However, within each neighbourhood, there’s substantial variability, suggesting other factors also play a key role in determining price.

- `Hanami’s Influence:` The cherry blossom season does indeed affect Airbnb pricing, with a detectable increase in many neighbourhoods during the blooming period. This aligns with the expected surge in demand as both locals and tourists flock to large cities like Tokyo to experience this idiosyncratic experience.

- `Predicting Price:` Using a combination of features like property type and host characteristics, we achieved a reasonable error margin (~¥10,000 or £50) for price predictions. This suggests that with further refinement, predictive models could be valuable for pricing strategies, particularly if more granular data was available for things such as neighbourhood.

- `Predicting Superhost Status:` Host-related features proved critical for determining superhost status, achieving a 70% accuracy rate with our basic model. This highlights some of the things someone might want to focus on if they wanted to achieve the status themselves. With more features perhaps an even better model could be developed!

## Getting Started

To replicate the analysis or run the code locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Parthian-Sunrise/udacity-ds-project-blog.git
2. **Navigate to the repository:**
   ```bash
   cd udacity-ds-project-blog
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
