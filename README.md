# Medical Survival Classification

## Project Description
This project implements a machine learning pipeline to classify the survival of coronary artery disease patients.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple regression models are trained and compared to find the best performing one based on various metrics.

## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:

git clone https://github.com/jjwongjun/survival-classification.git
cd survival-classification

2. Create and activate a conda environment:
    ```
    conda create -n hdbenv python=3.11
    conda activate hdbenv
    ```

3. Install the required packages:

pip install -r requirements.txt

## Pipeline Execution
To run the complete pipeline, execute:

python main.py

This will:
1. Load and clean the data
2. Preprocess features
3. Split data into training, validation, and test sets
4. Train baseline models
5. Perform hyperparameter tuning
6. Evaluate models and select the best one

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Train/validation/test split ratios
- Hyperparameter grid for model tuning

### 2. Data Preparation (`DataPreparation` class)
The DataPreparation' class handles:
- Removing duplicates
- Standardizing Survival, Smoke, and Ejection Fraction
- Converting Survival to numerical values
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The ModelTraining' class manages:
- Splitting data into training, validation, and test
sets
- Training baseline models (Logistic Regression, K-Nearest Neighbours (KNN), Decision Trees)
- Hyperparameter tuning for Logistic Regression, KNN, and Decision Trees models
- Evaluating models using multiple metrics (Accuracy, Precision, Recall, F1 Score, Receiver Operating Characteristic (ROC) Curve, Area Under Curve (AUC))
- Selecting the best model based on ROC AUC score

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying and evaluating the best model on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings from the EDA:

**EDA Findings and Explanations**: 
- `Creatinine` has null entries/missing values
- `Age` has negative values
- Duplicate `ID` with different other feature values 
- Mild inbalanced distribution of the two categories in `survive`
- `Survive` has labelling error of 'Yes': 1 and 'No': 0 categories
- `Smoke` has labelling capitalization error of 'NO': 'No' and 'YES':'Yes'
- `Ejection Fraction` has labelling error of 'L': 'Low' and 'N':'Normal'
- Outlier but plausbie values for `Sodium`, `Creatinine`, `Creatine phosphokinase`, `Platelets`, , `Blood Pressure`, , `Hemoglobin` 

### Data Cleaning
- Age converted to absolute values because some values are negative
- Duplicated rows removed
- Smoke converted to same uppercase and lowercase formats because feature is stored as an object (string) with various inconsistent formats
- Survive correctly labelled and converted to integer as part of encoding preparation
- Ejection Fraction correctly labelled because feature is stored with various inconsistent formats 

### Feature Processing
- `Survive` encoded for binary classification
- **Numerical Features**: `Age`, `Sodium`, `Creatinine`, `Platelets`, `Platelets`, `Creatine phosphokinase`, `Blood Pressure`, `Hemoglobin`, `Height`, `Weight`
- Standardized using `StandardScaler` because the numerical features have approximately normal distributions. In addition, the numerical features are on different scales, which can adversely impact the performance of machine learning algorithms if not addressed. Lastly, machine learning algorithms assume normally distributed data and are sensitive to feature scales, particularly those that rely on distance metrics
- **Nominal Features**: `Smoke`, `Gender`- Encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: `Ejection Fraction`, `Diabetes` - Encoded using `OrdinalEncoder` with predefined categories because both features can be considered to have a natural ranking based on their levels

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Logistic Regression**: Starting point for binary classification problems, providing a straightforward approach to predicting probabilities of class membership
2. **K-Nearest Neighbours (KNN)**: Non-parametric algorithm that classifies observations based on the majority class of their nearest neighbours in the feature space; can capture complex decision boundaries
3. **Decision Trees**: Non-parametric algorithm, make predictions by following a series of decisions based on the features; highly interpretable

### Hyperparameter Tuning
- Grid search is performed for Logistic Regression, KNN, and Decision Trees models because tuning model parameters can achieve the best possible performance. Grid Search CV's process is exhaustive but it ensures the best combination
- Parameters tuned because the right combination of these settings ensures balance and should perform well on unseen data. These include:
Logistic Regression:
- C (inverse of regularization strength)
- penalty (norm used in the penalization, e.g., 'l2')
K-Nearest Neighbors (KNN):
- n_neighbors (number of neighbors to use)
- weights (uniform or distance)
Decision Tree:
- max_depth (maximum depth of the tree)
- min_samples_split (minimum number of samples required to split an internal node)
- min_samples_leaf (minimum number of samples required to be at a leaf node)
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions (both true positives and true negatives) among the total number of cases
- **Confusion Matrix**: Detailed breakdown of model's performance, showing the number of correct and incorrect predictions, categorised by actual and predicted classes
- **Precision**: Ratio of correctly predicted positive observations to the total predicted positives
- **Recall**: Ratio of correctly predicted positive observations to all observations
- **F1 Score**: Harmonic mean of Precision and Recall
- **Receiver Operating Characteristic (ROC) Curve**: Graphical representation that illustrates the performance of a binary classifier across different threshold values
- **Area Under Curve (AUC)**: Area under ROC curve

## Deployment Considerations
- Code is structured into modular well-organized componenets making code easier to maintain, more scalable, and ready for real world applications