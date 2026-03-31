# -*- coding: utf-8 -*-
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.evaluation.visualization import create_forecast_visualization
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_before


class FixedForecast(ForecastingStrategy):
    """
    Fixed forecast strategy class

    This strategy defines a forecasting task with fixed prediction length.

    The required strategy configs include:

    - horizon (int): The length to predict, i.e. the length of the test series;
    - train_ratio_in_tv (float): The ratio of the training series when performing train-validation split.

    The accepted metrics include all regression metrics.

    The return fields other than the specified metrics are (in order):

    - FieldNames.FILE_NAME: The name of the series;
    - FieldNames.FIT_TIME: The training time;
    - FieldNames.INFERENCE_TIME: The inference time;
    - FieldNames.ACTUAL_DATA: The true test data, encoded as a string.
    - FieldNames.INFERENCE_DATA: The predicted data, encoded as a string.
    - FieldNames.LOG_INFO: Any log returned by the evaluator.
    """

    REQUIRED_CONFIGS = [
        "horizon",
        "train_ratio_in_tv",
        "save_true_pred",
        "save_vis",
    ]

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        model = model_factory()

        horizon = self._get_scalar_config_value("horizon", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        save_vis = self._get_scalar_config_value("save_vis", series_name)

        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = data_len - horizon
        if train_length <= 0:
            raise ValueError("The prediction step exceeds the data length")

        train_valid_data, test_data = split_before(series, train_length)
        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(train_valid_data, train_ratio_in_tv=train_ratio_in_tv)
        end_fit_time = time.time()
        predicted = model.forecast(horizon, train_valid_data)
        end_inference_time = time.time()

        single_series_results, log_info = self.evaluator.evaluate_with_log(
            test_data.to_numpy(),
            predicted,
            # TODO: add configs to control scaling behavior
            self._get_eval_scaler(train_valid_data, train_ratio_in_tv),
            train_valid_data.values,
        )
        inference_data = pd.DataFrame(
            predicted, columns=test_data.columns, index=test_data.index
        )

        # Save visualization if enabled
        visualization_paths = []
        if save_vis:
            visualization_paths = self._save_visualization(
                series_name=series_name,
                actual=test_data,
                predicted=inference_data,
                horizon=horizon
            )

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = self._encode_data(test_data) if save_true_pred else np.nan
        inference_data_encoded = self._encode_data(inference_data) if save_true_pred else np.nan

        # Append visualization paths to log_info
        if visualization_paths:
            vis_info = f"Visualization saved: {visualization_paths}"
            log_info = log_info + "\n" + vis_info if log_info else vis_info

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            end_inference_time - end_fit_time,
            actual_data_encoded,
            inference_data_encoded,
            log_info,
        ]

        return single_series_results

    @staticmethod
    def accepted_metrics():
        return regression_metrics.__all__

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]

    def _save_visualization(
        self,
        series_name: str,
        actual: pd.DataFrame,
        predicted: pd.DataFrame,
        horizon: int
    ) -> dict:
        """
        Save visualization plots for the forecast results.

        :param series_name: The name of the series.
        :param actual: Actual values (DataFrame).
        :param predicted: Predicted values (DataFrame).
        :param horizon: Prediction horizon.
        :return: Dictionary of saved file paths.
        """
        save_path = self.strategy_config.get("save_path", "result")
        vis_dir = os.path.join(save_path, "visualizations", series_name.replace('/', '_'))

        try:
            saved_files = create_forecast_visualization(
                series_name=series_name,
                actual_data=[actual],
                predicted_data=[predicted],
                save_dir=vis_dir,
                horizon=horizon
            )
            return saved_files
        except Exception as e:
            print(f"Warning: Failed to save visualization for {series_name}: {e}")
            return {}
