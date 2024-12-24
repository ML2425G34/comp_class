import os
import json
import numpy as np
import pandas as pd
import manalake

# Pipeline and transformers
from imblearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold

# Model assessment
from sklearn.metrics import classification_report, confusion_matrix

class Fitter:
    """
    A class to manage model training, evaluation, and result processing using cross-validation.
    """

    def __init__(self, X, y, instructions, logger):
        """
        Initializes the Fitter with data, instructions, and logger.
        
        Args:
            X (DataFrame): Features for training.
            y (Series): Target labels.
            instructions (list): List of model instructions.
            logger (MLLogger): Logger instance to save results.
        """
        self.X = X
        self.y = y
        self.instructions = instructions
        self.ml_logger = logger

        self.outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        self.inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        self.results_buffer = []

    def create_pipeline(self, pipeline_steps):
        """
        Constructs a pipeline based on the provided steps.
        
        Args:
            pipeline_steps (list): List of pipeline steps.
        
        Returns:
            Pipeline: A configured sklearn pipeline.
        """
        return make_pipeline(*[step[1](**step[2]) for step in pipeline_steps])

    def process_and_store_results(self, model_name, df_results, best_fold_index):
        """
        Processes cross-validation results and stores them in a structured format.
        
        Args:
            model_name (str): Name of the model.
            df_results (dict): Cross-validation results.
            best_fold_index (int): Index of the best scoring fold.
        
        Returns:
            dict: Processed results for the model.
        """
        best_fold_estimator = df_results['estimator'][best_fold_index]
        best_score = df_results['test_score'][best_fold_index]
        best_params = best_fold_estimator.best_params_

        mean_train_score = df_results['train_score'].mean()
        mean_test_score = df_results['test_score'].mean()
        fit_time = df_results['fit_time'][best_fold_index]
        score_time = df_results['score_time'][best_fold_index]

        best_fold_estimator.fit(self.X, self.y)
        refit_train_score = best_fold_estimator.score(self.X, self.y)

        model_result = {
            "model": model_name,
            "best_hyperparameters": best_params,
            "best_test_score": best_score,
            "mean_train_score": mean_train_score,
            "mean_test_score": mean_test_score,
            "fit_time": fit_time,
            "score_time": score_time,
            "refit_train_score": refit_train_score,
        }

        self.results_buffer.append(model_result)
        return model_result

    def run(self):
        """
        Executes the fitting process for all instructions and handles results storage.
        """
        try:
            for model_name, pipeline_steps, param_grid in self.instructions:
                print(model_name, pipeline_steps, param_grid)
                try:
                    pipeline = self.create_pipeline(pipeline_steps)

                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        scoring='f1_macro',
                        cv=self.inner_cv,
                        n_jobs=-1,
                        refit=True,
                        verbose=1
                    )

                    cv_results = cross_validate(
                        estimator=grid_search,
                        X=self.X,
                        y=self.y,
                        cv=self.outer_cv,
                        return_train_score=True,
                        return_estimator=True,
                        scoring="f1_macro",
                        n_jobs=-1,
                        verbose=1
                    )

                    best_fold_index = cv_results['test_score'].argmax()
                    self.process_and_store_results(model_name, cv_results, best_fold_index)

                except Exception as e:
                    print(f"Error during model processing for {model_name}: {e}")
                    continue

        except Exception as e:
            print(f"Critical error during fitting process: {e}")
        finally:
            self.ml_logger.results_to_csv(self.results_buffer)
            print("All results saved by reporter.")
