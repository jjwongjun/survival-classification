# Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataPreparation:
    """
    A class used to clean and preprocess HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

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
        df['Age'] = df['Age'].abs()  
        df.drop(columns=['Favorite color', 'ID'], inplace=True)
        df.drop_duplicates(inplace=True)
        df['Survive'] = df['Survive'].replace({'Yes': '1', 'No': '0'}).astype(int)
        df['Smoke'] = df['Smoke'].str.capitalize()
        df['Ejection Fraction'] = df['Ejection Fraction'].replace({'L': 'Low', 'N': 'Normal'})
        logging.info("Data cleaning completed.")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('impute', SimpleImputer(strategy='median'))
        ])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        ordinal_transformer = Pipeline(
            steps=[
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
            n_jobs=-1,
        )
        return preprocessor