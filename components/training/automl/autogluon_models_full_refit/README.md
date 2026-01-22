# Model Refit 🔄

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Refits best candidate models on full training dataset using AutoGluon.

The Model Refit component takes the candidate models selected from the model-building-selection stage
and retrains them on the full training dataset. This produces fully trained models ready for
evaluation. The component leverages AutoGluon's refitting capabilities to ensure models are
trained on the complete dataset while maintaining the model architecture and configuration
selected during the initial building phase.

This two-stage approach (initial building on sampled data, then refitting on full data) significantly reduces computational cost during exploration while ensuring final models are trained on the complete dataset for optimal performance.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `refit_models` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the refitted models ready for evaluation. |
| `train_data` | `dsl.Input[dsl.Dataset]` | `None` | Input dataset artifact containing the full training data. |
| `candidate_models` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact containing the candidate models from model-building-selection. |
| `selection_config` | `dict` | `None` | Optional dictionary with refit configuration. See [Refit Configuration](#refit-configuration) below. |

### Refit Configuration

The `selection_config` dictionary supports:

```python
{
    "time_limit": 7200,            # Time limit in seconds for refitting (default: 7200, 2 hours)
    "presets": "best_quality"      # AutoGluon preset (default: "best_quality")
}
```

**Note**: The refit stage typically uses longer time limits than the initial building stage since it's working with the full dataset and fewer models (top N only).

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `refit_models` | `dsl.Artifact` | The refitted models trained on the full training dataset, ready for evaluation. |
| Return value | `str` | A message indicating the completion status of model refitting. |

## Usage Examples 💡

### Basic Refit

```python
from kfp import dsl
from kfp_components.components.training.automl.model_training.model_refit import model_refit

@dsl.pipeline(name="model-refit-pipeline")
def my_pipeline(train_data, candidate_models):
    """Example pipeline for model refitting."""
    refit_task = model_refit(
        train_data=train_data,
        candidate_models=candidate_models,
        selection_config={
            "time_limit": 7200,
            "presets": "best_quality"
        }
    )
    return refit_task
```

### Custom Time Limit

```python
@dsl.pipeline(name="model-refit-custom-time-pipeline")
def my_pipeline(train_data, candidate_models):
    """Example pipeline with custom time limit."""
    refit_task = model_refit(
        train_data=train_data,
        candidate_models=candidate_models,
        selection_config={
            "time_limit": 10800,  # 3 hours
            "presets": "best_quality"
        }
    )
    return refit_task
```

## Two-Stage Training Strategy 🎯

The AutoML pipeline uses a two-stage training approach:

1. **Stage 1 - Model Building & Selection** (on sampled data):
   - Trains multiple model types on sampled data (500 samples)
   - Evaluates and ranks all models
   - Selects top N models (default: top 3)

2. **Stage 2 - Model Refit** (on full data):
   - Retrains only the top N selected models on full training dataset
   - Produces fully trained models ready for evaluation

**Benefits:**

- **Reduced Computational Cost**: Only explores models on small sample, then refits best ones on full data
- **Maintained Quality**: Final models are trained on complete dataset
- **Faster Iteration**: Faster exploration phase enables more experimentation

## AutoGluon Refitting 🔧

AutoGluon's refitting process:

- Maintains the model architecture and configuration from the initial building stage
- Retrains on the full dataset with potentially longer time limits
- Preserves ensemble structure and model relationships
- Produces production-ready AutoGluon Predictors

## Distributed Training (Future) 🚀

The component explores integration with **Kubeflow Katib** for:

- Distributed computing during refitting
- Experiment and trials logging
- Resource optimization across multiple nodes

This capability will be available in future versions.

## Notes 📝

- **Full Dataset Training**: Models are retrained on the complete training dataset, not just the sampled subset
- **Production-Ready**: Refitted models are AutoGluon Predictors ready for deployment
- **Time Limits**: Refit stage typically uses longer time limits (2+ hours) compared to initial building (1 hour)
- **Model Preservation**: The refit process maintains the model architecture and ensemble structure from the building stage

## Metadata 🗂️

- **Name**: model-refit
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: AutoGluon, Version: >=1.0.0
    - Name: Kubeflow Katib, Version: >=0.15.0 (to be explored)
- **Tags**:
  - automl
  - model-training
  - model-refit
  - autogluon
- **Last Verified**: 2025-01-27 00:00:00+00:00

## Additional Resources 📚

- **AutoGluon Documentation**: [AutoGluon GitHub](https://github.com/autogluon/autogluon)
- **Kubeflow Katib**: [Katib Documentation](https://www.kubeflow.org/docs/components/katib/)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
