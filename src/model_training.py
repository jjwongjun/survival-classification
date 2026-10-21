# Standard library imports
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Related third-party imports
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class ModelTraining:
    """
    A class used to train and evaluate machine learning models that predict patient survival.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    # Metrics every evaluation must report (checked in _compute_metrics_and_plots)
    REQUIRED_METRICS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocessor
        self.seed = config.get("random_seed", 42)
        # Each run saves its outputs to its own timestamped folder so runs can be compared
        artifacts_dir = config.get("artifacts_dir")
        self.run_dir = (
            os.path.join(artifacts_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
            if artifacts_dir else None
        )
        self.tuning_results: Dict[str, Dict[str, Any]] = {}  # best params and CV score per tuned model

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split the data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            X_train, X_val, X_test, y_train, y_val, y_test.
        """
        logging.info("Starting data splitting.")
        target = self.config["target_column"]
        X = df.drop(columns=target)
        y = df[target]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config["val_test_size"], random_state=self.seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config["val_size"], random_state=self.seed, stratify=y_temp
        )

        self._check_split_sizes(X_train, X_val, X_test)

        # Log class distributions after splitting
        self._log_class_distribution(y, "Full dataset")
        self._log_class_distribution(y_train, "Train")
        self._log_class_distribution(y_val, "Validation")
        self._log_class_distribution(y_test, "Test")
        logging.info(
            f"Imbalance strategy: stratified splits, class_weight={self.config.get('class_weight')}, "
            f"no resampling (positive class share is {y.mean():.1%})."
        )

        logging.info("Data split into train, validation, test sets.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")
        self._check_preprocessing(X_train)
        class_weight = self.config.get("class_weight")  # imbalance handling (None disables it)
        models = {
            "dummy": DummyClassifier(strategy="prior"),  # trivial benchmark every real model must beat
            "logreg": LogisticRegression(max_iter=1000, class_weight=class_weight),
            "knn": KNeighborsClassifier(),  # KNN has no class_weight option
            "dt": DecisionTreeClassifier(criterion='gini', min_samples_split=2, min_samples_leaf=1, class_weight=class_weight, random_state=self.seed),
            "rf": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight=class_weight, n_jobs=-1, random_state=self.seed),
            "hgb": HistGradientBoostingClassifier(class_weight=class_weight, random_state=self.seed),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("classifier", model)]
            )
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline, X_val, y_val, model_name
            )

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning (grid search with cross-validation) for the logistic regression, KNN and decision tree models and evaluate them.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}
        logreg_param_grid = self.config["logreg_param_grid"]
        knn_param_grid = self.config["knn_param_grid"]
        dt_param_grid = self.config["dt_param_grid"]
        param_grids = {
            "logreg_tuned": logreg_param_grid,
            "knn_tuned": knn_param_grid,
            "dt_tuned": dt_param_grid,
        }
        cv = self.config["cv"]
        scoring = self.config["scoring"]

        class_weight = self.config.get("class_weight")
        models = {
            "logreg_tuned": LogisticRegression(max_iter=1000, class_weight=class_weight),
            "knn_tuned": KNeighborsClassifier(),
            "dt_tuned": DecisionTreeClassifier(criterion='gini', min_samples_split=2, min_samples_leaf=1, class_weight=class_weight, random_state=self.seed),
        }

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("classifier", model)]
            ) 
            grid_search = GridSearchCV(
                pipeline, param_grids[model_name], cv=cv, scoring=scoring, n_jobs=-1,
                error_score="raise"  # a failed fit must stop the run, never be scored as NaN silently
            )
            grid_search.fit(X_train, y_train)
            tuned_models[model_name] = grid_search.best_estimator_
            self.tuning_results[model_name] = {
                "best_params": grid_search.best_params_,
                f"best_cv_{scoring}": float(grid_search.best_score_),
            }
            logging.info(
                f"{model_name}: best params {grid_search.best_params_}, "
                f"best CV {scoring} = {grid_search.best_score_:.4f}"
            )
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name + " (tuned)"
            )

        logging.info("Hyperparameter tuning completed.")
        return tuned_models, tuned_metrics

    def find_optimal_threshold(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
    ) -> float:
        """
        Find the decision threshold on the validation set (never on the test set).

        By default the threshold that maximises the F1 score of the positive class (Survive = 1) is chosen.
        If 'cost_false_negative' and 'cost_false_positive' are set in the config, the threshold
        that minimises total business cost is chosen instead.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        float: The selected probability threshold.
        """
        proba = model.predict_proba(X_val)[:, 1]
        cost_fn = self.config.get("cost_false_negative")
        cost_fp = self.config.get("cost_false_positive")

        if cost_fn is not None and cost_fp is not None:
            candidates = np.linspace(0.05, 0.95, 91)
            costs = []
            for t in candidates:
                tn, fp, fn, tp = confusion_matrix(y_val, (proba >= t).astype(int), labels=[0, 1]).ravel()
                costs.append(fn * cost_fn + fp * cost_fp)
            threshold = float(candidates[int(np.argmin(costs))])
            logging.info(f"Optimal threshold (minimum cost) on validation set: {threshold:.2f}")
        else:
            precision, recall, thresholds = precision_recall_curve(y_val, proba)
            f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
            threshold = float(thresholds[int(np.argmax(f1))])
            logging.info(f"Optimal threshold (maximum F1) on validation set: {threshold:.3f}")
        return threshold

    def evaluate_final_model(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.
        threshold (float): Probability threshold above which a patient is predicted to survive.
            Chosen on the validation set beforehand. Defaults to 0.5.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        metrics = self._compute_metrics_and_plots(
            model, X_test, y_test, f"{model_name} (test)", threshold=threshold, save_plots=True
        )
        logging.info(f"Final Test Metrics for {model_name} at threshold {threshold:.3f}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics

    def _evaluate_model(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        metrics = self._compute_metrics_and_plots(model, X_val, y_val, model_name)
        logging.info(f"{model_name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics

    def _compute_metrics_and_plots(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        threshold: float = 0.5,
        save_plots: bool = False,
    ) -> Dict[str, float]:
        """
        Compute classification metrics and draw the confusion matrix and ROC curve.

        Shared by the validation and test evaluations so the metrics and plots stay identical.

        Args:
        -----
        model (Pipeline): The trained model pipeline (must support predict_proba).
        X (pd.DataFrame): The features to evaluate on.
        y (pd.Series): The true target values.
        model_name (str): Name used in plot titles and file names.
        threshold (float): Probability threshold for predicting the positive class (survived). Defaults to 0.5.
        save_plots (bool): Whether to save the plots to the artifacts directory.

        Returns:
        --------
        Dict[str, float]: Accuracy, Balanced Accuracy, Precision, Recall, F1 Score, Specificity,
            ROC AUC, PR AUC, Brier Score (calibration) and, if costs are configured, Total Cost.
        """
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"{model_name} does not support predict_proba, which evaluation requires.")

        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1 Score": f1_score(y, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp),
            "ROC AUC": roc_auc_score(y, y_proba),
            "PR AUC": average_precision_score(y, y_proba),
            "Brier Score": brier_score_loss(y, y_proba),  # lower = better calibrated probabilities
        }
        cost_fn = self.config.get("cost_false_negative")
        cost_fp = self.config.get("cost_false_positive")
        if cost_fn is not None and cost_fp is not None:
            metrics["Total Cost"] = float(fn * cost_fn + fp * cost_fp)
        missing_keys = [k for k in self.REQUIRED_METRICS if k not in metrics]
        if missing_keys:
            raise ValueError(f"Metrics are missing required keys: {missing_keys}")

        # confusion matrix
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.config.get("class_labels", ["0", "1"])
        )
        cm_fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.grid(False)
        self._finish_plot(cm_fig, f"{model_name}_confusion_matrix", save_plots)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        roc_fig = plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (area = {metrics['ROC AUC']:.2f})")
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f"{model_name} - ROC Curve")
        plt.legend(loc='lower right')
        plt.grid()
        self._finish_plot(roc_fig, f"{model_name}_roc_curve", save_plots)

        # Precision-recall curve and calibration curve (are the predicted probabilities trustworthy?)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        frac_positive, mean_predicted = calibration_curve(y, y_proba, n_bins=10, strategy="quantile")
        pr_fig, (ax_pr, ax_cal) = plt.subplots(1, 2, figsize=(12, 5))
        ax_pr.plot(recall, precision, color='green', lw=2, label=f"PR curve (AP = {metrics['PR AUC']:.2f})")
        ax_pr.axhline(float(np.mean(y)), color='red', linestyle='--', label='Positive class share')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f"{model_name} - Precision-Recall Curve")
        ax_pr.legend(loc='upper right')
        ax_cal.plot(mean_predicted, frac_positive, marker='o', color='purple', label='Model')
        ax_cal.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfectly calibrated')
        ax_cal.set_xlabel('Mean predicted probability')
        ax_cal.set_ylabel('Observed fraction of positives')
        ax_cal.set_title(f"{model_name} - Calibration Curve")
        ax_cal.legend(loc='upper left')
        pr_fig.tight_layout()
        self._finish_plot(pr_fig, f"{model_name}_pr_and_calibration", save_plots)

        return metrics

    def _finish_plot(self, fig: plt.Figure, name: str, save: bool) -> None:
        """
        Optionally save a figure to the artifacts directory, then show and close it.

        Args:
        -----
        fig (plt.Figure): The figure to finish.
        name (str): Base file name (without extension).
        save (bool): Whether to save the figure.
        """
        if save and self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            fig.savefig(os.path.join(self.run_dir, f"{safe_name}.png"), bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def save_artifacts(
        self,
        model: Pipeline,
        model_name: str,
        metrics: Dict[str, float],
        threshold: float,
    ) -> None:
        """
        Save the selected model, its test metrics, decision threshold and a config snapshot.

        Files are written to a timestamped folder inside the 'artifacts_dir' set in the config.
        Nothing is saved if it is not set.

        Args:
        -----
        model (Pipeline): The selected trained model pipeline.
        model_name (str): The name of the selected model.
        metrics (Dict[str, float]): The test-set metrics of the selected model.
        threshold (float): The decision threshold chosen on the validation set.
        """
        if not self.run_dir:
            return
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            joblib.dump(model, os.path.join(self.run_dir, "best_model.joblib"))
            summary = {
                "model_name": model_name,
                "threshold": threshold,
                "test_metrics": {k: float(v) for k, v in metrics.items()},
                "tuning_results": self.tuning_results,
            }
            with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
                json.dump(summary, f, indent=2)
            with open(os.path.join(self.run_dir, "config_snapshot.yaml"), "w") as f:
                yaml.safe_dump(self.config, f)
            logging.info(f"Artifacts saved to {self.run_dir}")
        except OSError:
            logging.exception("Failed to save artifacts.")
            raise

    @staticmethod
    def _check_split_sizes(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> None:
        """
        Sanity check that the splits are non-empty and that the training set is the largest.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        X_val (pd.DataFrame): The validation features.
        X_test (pd.DataFrame): The test features.

        Raises:
        -------
        ValueError: If a split is empty or the training set is not larger than validation and test.

        Example:
            >>> big, small = pd.DataFrame({'a': range(8)}), pd.DataFrame({'a': range(1)})
            >>> ModelTraining._check_split_sizes(big, small, small)
            >>> ModelTraining._check_split_sizes(small, big, big)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Training set must be larger than the validation and test sets.
        """
        if min(len(X_train), len(X_val), len(X_test)) == 0:
            raise ValueError("Train, validation and test sets must all be non-empty.")
        if not len(X_train) > max(len(X_val), len(X_test)):
            raise ValueError("Training set must be larger than the validation and test sets.")

    def _check_preprocessing(self, X_train: pd.DataFrame) -> None:
        """
        Sanity check that the preprocessor produces only finite numbers (no NaN left after imputation).

        A copy of the preprocessor is fitted, so the real one is untouched.

        Args:
        -----
        X_train (pd.DataFrame): The training features.

        Raises:
        -------
        ValueError: If the preprocessed training data contains NaN or infinite values.
        """
        transformed = clone(self.preprocessor).fit_transform(X_train)
        if not np.isfinite(np.asarray(transformed, dtype=float)).all():
            raise ValueError("Preprocessed training data still contains NaN or infinite values.")
        logging.info(f"Preprocessing check passed: {transformed.shape[1]} features after encoding.")

    @staticmethod
    def _log_class_distribution(y_split: pd.Series, name: str) -> None:
        counts = y_split.value_counts().sort_index()
        props = y_split.value_counts(normalize=True).sort_index()
        out = pd.DataFrame({"count": counts, "proportion": props.round(4)})
        logging.info(f"--- {name} (n={len(y_split)}) ---\n{out}\n")