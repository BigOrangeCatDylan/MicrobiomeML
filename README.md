#MicrobiomeML

##Project Description
MicrobiomeML is a project that leverages machine learning techniques to analyze microbiome data, with the goal of distinguishing microbial profiles of ulcerative colitis (UC) patients from healthy individuals using data extracted from the GMrepo database. It also aims to explore potential microbial biomarkers associated with UC.
Background
Ulcerative colitis (UC) is a chronic inflammatory bowel disease closely linked to imbalances in the gut microbiome. By analyzing microbiome data from UC patients and healthy controls, this project seeks to identify key microbial genera associated with UC, offering insights for improved diagnosis and treatment.
Key Features

##Data Extraction and Cleaning:

Extracts genus-level abundance data for UC and healthy samples from the GMrepo database.
Filters and matches samples to ensure data quality and comparability.


##Machine Learning Model Training and Evaluation:

Employs a Random Forest model to classify microbiome data into UC and healthy categories.
Evaluates model performance using metrics like AUC, F1 score, and confusion matrix.


##Feature Importance and SHAP Analysis:

Identifies microbial genera with the greatest impact on classification, highlighting potential UC biomarkers.
Uses SHAP analysis to interpret model predictions and enhance explainability.


##Visualization:

Generates ROC curves, feature importance plots, and SHAP summary plots to visualize model performance and feature contributions.



##Usage Instructions

Environment Setup:

Install Python 3.8+ and required libraries: pandas, numpy, scikit-learn, shap, matplotlib, seaborn, joblib.


##Data Preparation:

Download sample_to_run_info.txt.gz, samples_loaded.txt.gz, and species_abundance.txt.gz from the GMrepo database.
Run the data extraction script part1.py to generate uc_healthy_genus_abundances_matched.csv.


Model Training and Analysis:

Execute the machine learning script part2.py to train the model, evaluate performance, and generate visualizations.
Results will be saved as model files and visualization figures.


##Configuration:

Adjust parameters such as filtering criteria and model settings via the config.json file.



##Project Structure

part1.py: Script for data extraction and cleaning.
part2.py: Script for machine learning analysis and visualization.
config.json: Configuration file for parameter adjustments.
gmrepo_data/: Directory for extracted data files.
figures/: Directory for visualization outputs.

Contact
For questions or suggestions, please reach out to the project maintainer at 202300700289@mail.sdu.edu.cn.
