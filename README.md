# Medical Survival Classification

## Project Description
This project implements a machine learning pipeline to classify the survival of coronary artery disease patients.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple classification models are trained and compared against a trivial baseline to find the best performing one based on various metrics.

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
    conda create -n survival-env python=3.11
    conda activate survival-env
    ```

3. Install the required packages:

pip install -r requirements.txt

## Pipeline Execution
To run the complete pipeline, execute:

python main.py

A different configuration file can be given with `python main.py --config path/to/config.yaml` (default: `./src/config.yaml`).

This will:
1. Load and validate the configuration, then load the data
2. Validate the data schema and clean the data
3. Preprocess features
4. Split data into training, validation, and test sets (stratified, with class distributions logged)
5. Train baseline models (a dummy classifier plus five real models)
6. Perform hyperparameter tuning
7. Evaluate models and select the best one by validation ROC AUC
8. Choose the decision threshold on the validation set and evaluate the best model once on the test set
9. Save the best model, metrics, threshold, config snapshot and plots to a timestamped folder in `artifacts/`, and write a log to `logs/pipeline.log`

The two output locations are set by `artifacts_dir` and `log_file` in the config. Remove either key to switch it off. The versions of Python and the main libraries and the random seed are logged at start-up.

### Checking the helper functions
The cleaning and sanity-check helpers contain doctests that act as unit tests. Run them from the project root with:

python -m doctest -v src/data_preparation.py src/model_training.py

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Random seed (`random_seed`), used for every split and model
- Train/validation/test split ratios (`val_test_size` is the held-out share, `val_size` is the share of that which becomes the test set; 0.2 and 0.5 give 80/10/10)
- Imbalance strategy (`class_weight`), tuning metric (`scoring`, ROC AUC) and cross-validation folds (`cv`)
- Hyperparameter grid for model tuning
- Output locations for artifacts and logs, and optional business costs for threshold selection

The configuration is validated at start-up: missing keys or out-of-range values stop the pipeline with a clear message.

### 2. Data Preparation (`DataPreparation` class)
The DataPreparation' class handles:
- Checking that all required columns exist (clear error message otherwise)
- Taking the absolute value of the negative `Age` values
- Dropping `ID` and `Favorite color`, then removing duplicates (on a copy, so the input DataFrame is not modified)
- Standardizing Survive, Smoke, and Ejection Fraction labels
- Converting Survive to the integers 0 and 1, and stopping with an error if any other value is found
- Sanity checks on the cleaned data: all configured columns are present, the target is only 0 and 1, and the ordinal columns only contain the expected categories (otherwise the encoder would silently turn them into -1)
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The ModelTraining' class manages:
- Splitting data into training, validation, and test sets (stratified, with split-size checks and class distributions logged)
- Training a dummy-classifier baseline and baseline models (Logistic Regression, K-Nearest Neighbours (KNN), Decision Trees, Random Forest, Histogram Gradient Boosting)
- Hyperparameter tuning for Logistic Regression, KNN, and Decision Trees models
- Evaluating models using multiple metrics computed from predicted probabilities (Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, Specificity, ROC AUC, PR AUC, Brier score) with confusion matrix, ROC, precision-recall and calibration plots
- Selecting the best model based on ROC AUC score on the validation set
- Choosing the decision threshold on the validation set and evaluating the best model once on the test set
- Saving the selected model, metrics, threshold, config snapshot and plots

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
- The data has 15,000 rows and 16 columns (7 text, 9 numeric)
- `Creatinine` has 499 missing values (3.3%). All of these patients did not survive (0% survival vs 33% when recorded), so the missingness is informative and the pipeline keeps a "was missing" flag
- `Age` has 430 negative values, treated as sign errors
- 885 `ID`s appear more than once, but the records describe different people (random ID collisions), so `ID` is dropped and no patient can appear in both train and test. Only 2 rows are exact duplicates
- Moderate imbalance in `Survive`: 67.9% did not survive and 32.1% survived (2.1 : 1)
- `Survive` has labelling error of 'Yes': 1 and 'No': 0 categories
- `Smoke` has labelling capitalization error of 'NO': 'No' and 'YES':'Yes'
- `Ejection Fraction` has labelling error of 'L': 'Low' and 'N':'Normal'
- Outlier but plausible values (by the IQR rule) for `Creatinine` (10.4%, skew 4.3), `Creatine phosphokinase` (9.9%, skew 4.4), `Platelets` (6.3%) and `Sodium` (1.2%); the other numeric features have none
- Features most correlated with survival: `Weight`, `Creatinine`, `Age`, `Sodium`. `Diabetes`, `Smoke` and `Gender` show no relationship with survival
- Weight relative to height (BMI) is a strong interaction: survival is 5% below BMI 18.5 and 89% above BMI 35, although `Height` alone is uninformative. BMI is not added as a column because the tree models can build it themselves and the gain for logistic regression was small
- Permutation importance (random forest, validation set) ranks `Creatinine` first, then `Weight`, `Creatine phosphokinase`, `Sodium`, `Age` and `Platelets`; the remaining features are about zero
- The classes are almost perfectly separable by tree-based models (validation ROC AUC close to 1.0). No leakage was found, so this looks like a property of the dataset, but real clinical data would not be this clean 

### Data Cleaning
- Age converted to absolute values because some values are negative
- `ID` and `Favorite color` dropped, then duplicated rows removed
- Smoke converted to same uppercase and lowercase formats because feature is stored as an object (string) with various inconsistent formats
- Survive correctly labelled and converted to integer (0 and 1), and validated, as part of encoding preparation
- Ejection Fraction correctly labelled because feature is stored with various inconsistent formats 

### Feature Processing
- `Survive` encoded for binary classification
- **Numerical Features**: `Age`, `Sodium`, `Creatinine`, `Platelets`, `Creatine phosphokinase`, `Blood Pressure`, `Hemoglobin`, `Height`, `Weight`
- Missing values are imputed with the median **first** (`SimpleImputer`, with `add_indicator=True` so a "was missing" flag is kept for `Creatinine`), and the columns are standardized **afterwards** with `StandardScaler`. Imputing before scaling matters because the scaler must see complete columns. Standardization is used because the numerical features are on very different scales (for example `Platelets` in the hundreds of thousands and `Creatinine` around 1), which hurts distance-based (KNN) and gradient-based (logistic regression) models. Tree-based models are not affected by scaling. Several features (`Creatinine`, `Creatine phosphokinase`) are strongly right-skewed, so they are not normally distributed
- **Nominal Features**: `Smoke`, `Gender` - Encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: `Ejection Fraction`, `Diabetes` - Imputed with the most frequent value (a safeguard: there are currently no missing values) and encoded using `OrdinalEncoder` with predefined categories because both features can be considered to have a natural ranking based on their levels

## Model Choices and Evaluation

### Class Imbalance
The target is moderately imbalanced (about 32% survived, 68% did not). **No resampling is used** (no oversampling, undersampling or SMOTE) because the imbalance is moderate and resampling risks overfitting or discarding data. Instead:
- Splits are stratified, so train, validation, test and every CV fold keep the same class ratio (the distributions are logged after splitting)
- `class_weight='balanced'` is applied to the models that support it (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting). KNN has no such option. It can be switched off with `class_weight: null` in the config
- A `DummyClassifier` baseline (ROC AUC 0.5) shows the minimum bar a real model must beat
- Models are compared on ROC AUC, PR AUC, precision, recall, F1 and balanced accuracy rather than accuracy alone, and the decision threshold is tuned (below)

### Models Implemented and Justifications
0. **Dummy Classifier**: Always predicts from the class prior; the benchmark that every real model must beat
1. **Logistic Regression**: Starting point for binary classification problems, providing a straightforward approach to predicting probabilities of class membership
2. **K-Nearest Neighbours (KNN)**: Non-parametric algorithm that classifies observations based on the majority class of their nearest neighbours in the feature space; can capture complex decision boundaries
3. **Decision Trees**: Non-parametric algorithm, make predictions by following a series of decisions based on the features; highly interpretable
4. **Random Forest**: Ensemble of decision trees; captures non-linear effects and interactions and is robust to outliers
5. **Histogram Gradient Boosting**: Boosted trees that are usually strongest on tabular data and handle missing values natively

### Threshold Selection and Calibration
A probability threshold of 0.5 is rarely optimal in a clinical setting. The threshold is chosen **on the validation set only** (maximising F1 for the positive class, `Survive = 1`), then fixed before the one-time test evaluation. If `cost_false_negative` and `cost_false_positive` are set in the config, the threshold that minimises total cost is chosen instead. Calibration is checked with the Brier score and a calibration curve for every model.

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
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance
- Tuning is scored with ROC AUC (set by `scoring` in the config) because accuracy is misleading on an imbalanced target. The best parameters and CV scores are logged and saved to `metrics.json` in the run's artifacts folder

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions (both true positives and true negatives) among the total number of cases
- **Confusion Matrix**: Detailed breakdown of model's performance, showing the number of correct and incorrect predictions, categorised by actual and predicted classes
- **Precision**: Ratio of correctly predicted positive observations to the total predicted positives
- **Recall**: Ratio of correctly predicted positive observations to all actual positive observations
- **F1 Score**: Harmonic mean of Precision and Recall
- **Balanced Accuracy**: Average of recall on each class, which is not inflated by the larger class
- **Specificity**: Proportion of actual negatives (did not survive) that are correctly predicted as negatives
- **PR AUC**: Area under the precision-recall curve (average precision)
- **Brier Score**: Mean squared error of the predicted probabilities, used as a calibration check (lower is better)
- **Receiver Operating Characteristic (ROC) Curve**: Graphical representation that illustrates the performance of a binary classifier across different threshold values
- **Area Under Curve (AUC)**: Area under ROC curve

## Deployment Considerations
- Code is structured into modular well-organized components making code easier to maintain, more scalable, and ready for real world applications
- The configuration is validated at start-up and the data schema is checked before cleaning; the cleaned data, the split sizes and the preprocessed features are sanity-checked before training
- Console and file logging with timestamps; file, config and data errors are reported with clear messages and a non-zero exit code
- The selected model (`best_model.joblib`), its test metrics, decision threshold, tuning results and a snapshot of the configuration are saved to a timestamped folder in `artifacts/`, so a result can be reproduced and runs can be compared

### Known Limitations and Future Work
- There is no separate `tests/` folder; the helper functions and sanity checks are covered by doctests only. Adding `pytest` unit tests and one end-to-end smoke test is the next step
- There is no batch-prediction command yet. The saved model and threshold are what such a command would load
- Only Logistic Regression, KNN and Decision Tree are hyperparameter tuned; Random Forest and Gradient Boosting use default settings
- Probabilities are checked for calibration but not recalibrated
- The tree-based models separate the classes almost perfectly on this dataset, so test scores will be optimistic compared with real clinical data
- The imputation flag for `Creatinine` assumes the missing value is known to be missing at prediction time; this should be confirmed with the data owner