# Standard library imports
import logging
from typing import Any, Dict, List

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataPreparation:
    """
    A class used to clean and preprocess clinical data for predicting the survival of coronary artery disease patients.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    # Columns that must be present in the raw data before cleaning starts
    REQUIRED_COLUMNS = [
        "ID", "Survive", "Gender", "Smoke", "Diabetes", "Age", "Ejection Fraction",
        "Sodium", "Creatinine", "Platelets", "Creatine phosphokinase", "Blood Pressure",
        "Hemoglobin", "Height", "Weight", "Favorite color",
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        self._validate_schema(df, self.REQUIRED_COLUMNS)
        logging.info(f"Raw data has {df.shape[0]} rows and {df.shape[1]} columns.")
        df = df.copy()  # work on a copy so the caller's DataFrame is not mutated
        df['Age'] = df['Age'].abs()  
        df = df.drop(columns=['Favorite color', 'ID'])
        df = df.drop_duplicates()
        df['Survive'] = self._standardise_target(df['Survive'])
        df['Smoke'] = df['Smoke'].str.strip().str.capitalize()
        df['Ejection Fraction'] = df['Ejection Fraction'].replace({'L': 'Low', 'N': 'Normal'})
        self._check_cleaned_data(df)
        logging.info(
            f"Data cleaning completed. Cleaned data has {df.shape[0]} rows and {df.shape[1]} columns; "
            f"positive class share ({self.config['target_column']}=1): {df[self.config['target_column']].mean():.1%}."
        )
        return df

    @staticmethod
    def _validate_schema(df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Checks that the raw data has the columns the pipeline relies on.

        Args:
        -----
        df (pd.DataFrame): The raw input DataFrame.
        required_columns (List[str]): Column names that must be present.

        Raises:
        -------
        ValueError: If any required column is missing.

        Example:
            >>> DataPreparation._validate_schema(pd.DataFrame({'Age': [1]}), ['Age'])
            >>> DataPreparation._validate_schema(pd.DataFrame({'Age': [1]}), ['Age', 'Sodium'])  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Input data is missing required columns: ['Sodium']
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")

    @staticmethod
    def _standardise_target(target: pd.Series) -> pd.Series:
        """
        Converts the survival target, which mixes 'Yes'/'No' with '1'/'0', into integers 0 and 1.

        Args:
        -----
        target (pd.Series): The raw target column.

        Returns:
        --------
        pd.Series: The target as integers 0 (did not survive) and 1 (survived).

        Raises:
        -------
        ValueError: If the target contains anything that is not Yes/No/1/0.

        Example:
            >>> DataPreparation._standardise_target(pd.Series(['Yes', 'No', '1', '0', 1, 0])).tolist()
            [1, 0, 1, 0, 1, 0]
            >>> DataPreparation._standardise_target(pd.Series(['Yes', 'maybe']))  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Target contains values other than Yes/No/1/0: ['Yes', 'maybe']
        """
        try:
            converted = target.replace({'Yes': 1, 'No': 0}).astype(int)
        except (ValueError, TypeError):
            raise ValueError(f"Target contains values other than Yes/No/1/0: {target.unique().tolist()}")
        if not set(converted.unique()) <= {0, 1}:
            raise ValueError(f"Target contains values other than Yes/No/1/0: {target.unique().tolist()}")
        return converted

    def _check_cleaned_data(self, df: pd.DataFrame) -> None:
        """
        Sanity checks run after cleaning, so problems are caught before splitting and training.

        Checks that every configured feature and the target are present, that the target is only
        0 and 1, and that the ordinal columns only contain the categories the encoder expects
        (otherwise OrdinalEncoder would silently encode them as -1).

        Args:
        -----
        df (pd.DataFrame): The cleaned DataFrame.

        Raises:
        -------
        ValueError: If any check fails.
        """
        features = (
            self.config["numerical_features"]
            + self.config["nominal_features"]
            + self.config["ordinal_features"]
        )
        missing = [col for col in features + [self.config["target_column"]] if col not in df.columns]
        if missing:
            raise ValueError(f"Cleaned data is missing expected columns: {missing}")
        if not set(df[self.config["target_column"]].unique()) <= {0, 1}:
            raise ValueError("Cleaned target must only contain 0 and 1.")
        for col, categories in zip(
            self.config["ordinal_features"],
            [self.config["ejection_fraction_categories"], self.config["diabetes_categories"]],
        ):
            unexpected = set(df[col].dropna().unique()) - set(categories)
            if unexpected:
                raise ValueError(f"Column '{col}' has unexpected categories: {sorted(unexpected)}")

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        # Impute first, then scale: the scaler must see complete columns.
        # add_indicator=True keeps a "was missing" flag: every patient with a missing Creatinine
        # did not survive (see the EDA), so the missingness itself is informative.
        numerical_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler())
        ])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        ordinal_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[
                            self.config["ejection_fraction_categories"],
                            self.config["diabetes_categories"],
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                )
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("ord", ordinal_transformer, self.config["ordinal_features"]),
            ],
            sparse_threshold=0,  # always return a dense array so every model type can consume it
            n_jobs=-1,
        )
        return preprocessor