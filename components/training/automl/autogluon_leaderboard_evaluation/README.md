# Leaderboard Evaluation 📊

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Evaluates fully trained models and generates a comprehensive leaderboard with performance metrics.

The Leaderboard Evaluation component performs final evaluation of the refitted models on test data.
It generates a comprehensive leaderboard ranked by the specified evaluation metric, providing
detailed performance metrics for all models. The component creates multiple Model artifacts
(Predictor1, Predictor2, etc.) representing the evaluated models, and optionally generates
classification metrics artifacts.

This component is the final evaluation stage before optional deployment steps (model registry and KServe deployment).

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_artifacts` | `dsl.Output[dsl.Model]` | `None` | Output Model artifacts containing the evaluated models (Predictor1, Predictor2, etc.). |
| `leaderboard` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the leaderboard CSV with performance metrics. |
| `refit_models` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact containing the refitted models from model-refit. |
| `test_data` | `dsl.Input[dsl.Dataset]` | `None` | Input dataset artifact containing the test data for evaluation. |
| `task_type` | `str` | `None` | Type of ML task. Required: `"classification"`, `"regression"`, or `"time_series"`. |
| `metrics` | `dsl.Output[dsl.Metrics]` | `None` | Optional output metrics artifact for classification metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix). |
| `eval_metric` | `str` | `None` | Optional evaluation metric name used for ranking. If not provided, uses the metric from selection_config. |

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `model_artifacts` | `dsl.Model` | Multiple Model artifacts (Predictor1, Predictor2, etc.) representing the evaluated models. |
| `leaderboard` | `dsl.Artifact` | CSV file containing the model leaderboard with performance metrics ranked by evaluation score. |
| `metrics` | `dsl.Metrics` | Optional metrics artifact containing classification metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix). |
| Return value | `str` | A message indicating the completion status of model evaluation. |

## Usage Examples 💡

### Classification Task

```python
from kfp import dsl
from kfp_components.components.training.automl.model_evaluation.leaderboard_evaluation import (
    leaderboard_evaluation,
)

@dsl.pipeline(name="leaderboard-evaluation-classification-pipeline")
def my_pipeline(refit_models, test_data):
    """Example pipeline for classification evaluation."""
    eval_task = leaderboard_evaluation(
        refit_models=refit_models,
        test_data=test_data,
        task_type="classification",
        eval_metric="accuracy"
    )
    return eval_task
```

### Regression Task

```python
@dsl.pipeline(name="leaderboard-evaluation-regression-pipeline")
def my_pipeline(refit_models, test_data):
    """Example pipeline for regression evaluation."""
    eval_task = leaderboard_evaluation(
        refit_models=refit_models,
        test_data=test_data,
        task_type="regression",
        eval_metric="R2"
    )
    return eval_task
```

### Time-Series Task

```python
@dsl.pipeline(name="leaderboard-evaluation-timeseries-pipeline")
def my_pipeline(refit_models, test_data):
    """Example pipeline for time-series evaluation."""
    eval_task = leaderboard_evaluation(
        refit_models=refit_models,
        test_data=test_data,
        task_type="time_series",
        eval_metric="MAPE"
    )
    return eval_task
```

### With Metrics Artifact

```python
@dsl.pipeline(name="leaderboard-evaluation-with-metrics-pipeline")
def my_pipeline(refit_models, test_data):
    """Example pipeline with metrics artifact."""
    eval_task = leaderboard_evaluation(
        refit_models=refit_models,
        test_data=test_data,
        task_type="classification",
        eval_metric="accuracy"
    )
    # Metrics artifact will contain: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    return eval_task
```

## Leaderboard Structure 📋

The leaderboard CSV contains:

- **Model Identifiers**: Model names/IDs (Predictor1, Predictor2, etc.)
- **Performance Metrics**:
  - Classification: accuracy, precision, recall, F1, ROC-AUC, log_loss
  - Regression: R², RMSE, MAE, MAPE
  - Time-Series: MAPE, MASE, RMSE, MAE
- **Rankings**: Models ranked by the specified evaluation metric
- **Training Information**: Training time, model complexity
- **Feature Importances**: Feature importance scores (for tree-based models)

## Classification Metrics 📊

For classification tasks, the optional metrics artifact includes:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (per-class and weighted average)
- **Recall**: Recall score (per-class and weighted average)
- **F1 Score**: F1 score (per-class and weighted average)
- **ROC-AUC**: Area under the ROC curve (for binary classification)
- **Confusion Matrix**: Confusion matrix visualization
- **Per-Class Metrics**: Detailed metrics for each class (multiclass)

## Model Artifacts 🎯

The component creates multiple Model artifacts:

- **Predictor1**: Best performing model (ranked #1)
- **Predictor2**: Second best model (ranked #2)
- **Predictor3**: Third best model (ranked #3)
- ... (up to top N models)

Each model artifact is a fully trained AutoGluon Predictor ready for deployment.

## Notes 📝

- **Comprehensive Evaluation**: Evaluates all refitted models on test data
- **Multiple Metrics**: Provides comprehensive performance metrics beyond the ranking metric
- **Production-Ready Models**: Model artifacts are AutoGluon Predictors ready for deployment
- **Feature Importances**: Includes feature importance analysis for interpretability
- **Best Model Selection**: The best model (Predictor1) is typically used for deployment

## Metadata 🗂️

- **Name**: leaderboard-evaluation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: AutoGluon, Version: >=1.0.0
    - Name: scikit-learn, Version: >=1.0.0
- **Tags**:
  - automl
  - model-evaluation
  - leaderboard
  - autogluon
- **Last Verified**: 2025-01-27 00:00:00+00:00

## Additional Resources 📚

- **AutoGluon Documentation**: [AutoGluon GitHub](https://github.com/autogluon/autogluon)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
