# Model Registry 📝

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Registers the best model with RHOAI Model Registry for versioning and deployment management.

The Model Registry component is an optional step in the AutoML pipeline (controlled by the `auto_register` parameter) that registers the best model from the evaluation stage with the RHOAI Model Registry. It registers the AutoGluon Predictor with comprehensive model metadata for deployment purposes.

Model registration enables:

- **Version Control**: Track model versions and their performance
- **Metadata Management**: Store model metadata (task type, label column, experiment name, etc.)
- **Deployment Readiness**: Prepare models for deployment workflows
- **Model Lineage**: Track model training history and relationships

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registered_model` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the registration information and model registry metadata. |
| `best_model` | `dsl.Input[dsl.Model]` | `None` | Input Model artifact containing the best model from leaderboard-evaluation (typically Predictor1). |
| `model_metadata` | `dict` | `None` | Dictionary containing model metadata. See [Model Metadata](#model-metadata) below. |
| `auto_register` | `bool` | `False` | Boolean flag to enable automatic model registration. If `False`, the component skips registration. |

### Model Metadata

The `model_metadata` dictionary should contain:

```python
{
    "task_type": "classification",      # Task type: "classification", "regression", "time_series"
    "label_column": "target",           # Name of the label/target column
    "experiment_name": "AutoML Run",    # Name of the AutoML experiment
    "description": "Customer churn prediction model",  # Optional description
    "eval_metric": "accuracy",          # Evaluation metric used
    "model_version": "1.0.0",          # Optional model version
    "author": "data-science-team"       # Optional author information
}
```

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `registered_model` | `dsl.Artifact` | Artifact containing the registration information, including registry ID, model URI, and metadata. |
| Return value | `str` | A message indicating the completion status of model registration. |

## Usage Examples 💡

### Basic Registration

```python
from kfp import dsl
from kfp_components.components.training.automl.model_deployment.model_registry import model_registry

@dsl.pipeline(name="model-registry-pipeline")
def my_pipeline(best_model):
    """Example pipeline for model registration."""
    with dsl.Condition(auto_register == True):
        registry_task = model_registry(
            best_model=best_model,
            model_metadata={
                "task_type": "classification",
                "label_column": "target",
                "experiment_name": "AutoML Run",
                "description": "Customer churn prediction model",
                "eval_metric": "accuracy"
            },
            auto_register=True
        )
    return registry_task
```

### Conditional Registration

```python
@dsl.pipeline(name="conditional-model-registry-pipeline")
def my_pipeline(best_model, auto_register: bool = False):
    """Example pipeline with conditional registration."""
    with dsl.Condition(auto_register == True):
        registry_task = model_registry(
            best_model=best_model,
            model_metadata={
                "task_type": "regression",
                "label_column": "price",
                "experiment_name": "Price Prediction Experiment",
                "eval_metric": "R2"
            },
            auto_register=auto_register
        )
    return registry_task
```

### Time-Series Model Registration

```python
@dsl.pipeline(name="timeseries-model-registry-pipeline")
def my_pipeline(best_model):
    """Example pipeline for time-series model registration."""
    with dsl.Condition(auto_register == True):
        registry_task = model_registry(
            best_model=best_model,
            model_metadata={
                "task_type": "time_series",
                "target": "value",
                "timestamp_column": "timestamp",
                "experiment_name": "Sales Forecasting",
                "eval_metric": "MAPE"
            },
            auto_register=True
        )
    return registry_task
```

## Registration Process 🔄

When `auto_register=True`, the component:

1. **Connects to RHOAI Model Registry**: Uses RHOAI Model Registry API to access the registry
2. **Uploads Model Artifacts**: Uploads the AutoGluon Predictor model files
3. **Registers Model Metadata**: Stores comprehensive metadata including:
   - Task type and configuration
   - Label column and feature information
   - Experiment name and description
   - Evaluation metrics and performance scores
   - Model version and lineage
4. **Returns Registration Info**: Provides registry ID, model URI, and access information

## Model Storage 📦

AutoGluon keeps all models trained under the hood but uses the best one by default (plus required base ones). The Model Registry stores:

- **Best Model**: The top-performing model (Predictor1)
- **Model Artifacts**: All necessary files for model deployment
- **Metadata**: Comprehensive model information for tracking and deployment

## Notes 📝

- **Optional Step**: Controlled by `auto_register` parameter - set to `False` to skip registration
- **AutoGluon Predictor**: Registers AutoGluon Predictor format, which includes all necessary components for inference
- **Storage Location**: Model storage and deployment happens outside the training pipeline
- **Version Control**: Model Registry provides versioning capabilities for tracking model iterations
- **Deployment Ready**: Registered models are ready for deployment via KServe or other serving platforms

## Metadata 🗂️

- **Name**: model-registry
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: RHOAI Model Registry API, Version: >=1.0.0
- **Tags**:
  - automl
  - model-deployment
  - model-registry
- **Last Verified**: 2025-01-27 00:00:00+00:00

## Additional Resources 📚

- **AutoML Documentation**: [AutoML README](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/README.md)
- **Components Documentation**: [Components Structure](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/components.md)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
