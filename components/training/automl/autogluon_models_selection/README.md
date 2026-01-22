# Model Building Selection 🏗️

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Builds multiple models using sampled data and selects top performers using AutoGluon.

The Model Building Selection component is a core stage in the AutoML pipeline that builds multiple
machine learning models using the sampled training data (typically 500 samples). It leverages the
AutoGluon library to automatically train various model types (neural networks, tree-based models,
linear models, etc.) and uses an ensembling approach (stacking/bagging) rather than traditional
hyperparameter optimization.

The component evaluates all trained models and promotes the top N performers (default: top 3) to the refit stage, where they will be retrained on the full training dataset. This two-stage approach significantly reduces computational cost while maintaining model quality.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `candidate_models` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the candidate models selected for refitting. |
| `model_leaderboard` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the model leaderboard CSV with performance metrics. |
| `sampled_train_data` | `dsl.Input[dsl.Dataset]` | `None` | Input dataset artifact containing the sampled training data (typically 500 samples). |
| `task_type` | `str` | `None` | Type of ML task. Required: `"classification"`, `"regression"`, or `"time_series"`. |
| `label_column` | `str` | `None` | Name of the label/target column in the dataset. Required. |
| `selection_config` | `dict` | `None` | Optional dictionary with model selection configuration. See [Selection Configuration](#selection-configuration) below. |

### Selection Configuration

The `selection_config` dictionary supports:

```python
{
    "top_n": 3,                    # Number of top models to promote (default: 3)
    "eval_metric": None,           # Evaluation metric (default: AutoGluon chooses based on task_type)
    "time_limit": 3600,            # Time limit in seconds for model building (default: 3600)
    "presets": "best_quality"      # AutoGluon preset: "best_quality", "high_quality", "good_quality_faster_inference", "optimize_for_deployment"
}
```

**Evaluation Metrics:**

- **Classification**: `"accuracy"`, `"f1"`, `"roc_auc"`, `"log_loss"`, etc.
- **Regression**: `"R2"`, `"RMSE"`, `"MAE"`, etc.
- **Time-Series**: `"MAPE"`, `"MASE"`, `"RMSE"`, etc.

**AutoGluon Presets:**

- `"best_quality"` - Best model quality, longer training time
- `"high_quality"` - High quality with faster training
- `"good_quality_faster_inference"` - Good quality optimized for inference speed
- `"optimize_for_deployment"` - Optimized for deployment with smaller model size

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `candidate_models` | `dsl.Artifact` | The candidate models (top N) ready for refitting on full training data. |
| `model_leaderboard` | `dsl.Artifact` | CSV file containing the model leaderboard with performance metrics ranked by evaluation score. |
| Return value | `str` | A message indicating the completion status of model building and selection. |

## Usage Examples 💡

### Classification Task

```python
from kfp import dsl
from kfp_components.components.training.automl.model_training.model_building_selection import (
    model_building_selection,
)

@dsl.pipeline(name="model-building-classification-pipeline")
def my_pipeline(sampled_train_data):
    """Example pipeline for classification task."""
    build_task = model_building_selection(
        sampled_train_data=sampled_train_data,
        task_type="classification",
        label_column="target",
        selection_config={
            "top_n": 3,
            "eval_metric": "accuracy",
            "time_limit": 3600,
            "presets": "best_quality"
        }
    )
    return build_task
```

### Regression Task

```python
@dsl.pipeline(name="model-building-regression-pipeline")
def my_pipeline(sampled_train_data):
    """Example pipeline for regression task."""
    build_task = model_building_selection(
        sampled_train_data=sampled_train_data,
        task_type="regression",
        label_column="price",
        selection_config={
            "top_n": 5,
            "eval_metric": "R2",
            "time_limit": 1800,
            "presets": "high_quality"
        }
    )
    return build_task
```

### Time-Series Task

```python
@dsl.pipeline(name="model-building-timeseries-pipeline")
def my_pipeline(sampled_train_data):
    """Example pipeline for time-series task."""
    build_task = model_building_selection(
        sampled_train_data=sampled_train_data,
        task_type="time_series",
        label_column="value",
        selection_config={
            "top_n": 3,
            "eval_metric": "MAPE",
            "time_limit": 3600,
            "presets": "best_quality"
        }
    )
    return build_task
```

## AutoGluon Ensembling Approach 🎯

AutoGluon uses an **ensembling approach** (stacking/bagging) rather than traditional hyperparameter optimization:

1. **Multiple Model Types**: Trains various model families (neural networks, XGBoost, LightGBM, CatBoost, linear models, etc.)
2. **Stacking**: Combines predictions from multiple models using a meta-learner
3. **Bagging**: Uses bootstrap aggregation to improve model robustness
4. **Automatic Selection**: Selects the best ensemble configuration automatically

This approach is more efficient than traditional HPO and typically produces better results for tabular data.

## Model Leaderboard 📊

The component generates a leaderboard CSV file containing:

- Model names/identifiers
- Performance metrics (accuracy, F1, ROC-AUC for classification; R², RMSE for regression)
- Training time
- Model complexity metrics
- Rankings based on the specified evaluation metric

## Notes 📝

- **Two-Stage Training**: Uses sampled data (500 samples) for initial exploration, then refits top models on full data
- **No Traditional HPO**: AutoGluon uses ensembling instead of hyperparameter optimization (HPO may be explored post-MVP)
- **Automatic Feature Engineering**: AutoGluon automatically handles data preprocessing and feature engineering
- **Production-Ready**: Selected models are AutoGluon Predictors ready for deployment

## Metadata 🗂️

- **Name**: model-building-selection
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: AutoGluon, Version: >=1.0.0
- **Tags**:
  - automl
  - model-training
  - model-selection
  - autogluon
- **Last Verified**: 2025-01-27 00:00:00+00:00

## Additional Resources 📚

- **AutoML Documentation**: [AutoML README](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/README.md)
- **Components Documentation**: [Components Structure](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/components.md)
- **AutoGluon Documentation**: [AutoGluon GitHub](https://github.com/autogluon/autogluon)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
