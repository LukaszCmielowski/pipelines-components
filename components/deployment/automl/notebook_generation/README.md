# Notebook Generation 📓

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

This component produces a **notebook artifact** whose URI points to a Jupyter notebook. The notebook lets you:

- **Review the experiment leaderboard** — Inspect trained model evaluation quality and compare metrics.
- **Load a chosen AutoGluon model from S3** — Download and load a selected model from pipeline run artifacts (S3-compatible storage).
- **Run predictions** — Use the loaded predictor to score new data (e.g. `predict` or `predict_proba`).

Use the artifact URI to open the notebook in a Jupyter-compatible environment (e.g. workbench) and interact with your AutoML run results.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_param` | `str` | — | TODO: Description of the component parameter. |

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| Notebook artifact | `dsl.Output[dsl.Artifact]` | Artifact whose URI points to the generated Jupyter notebook. Use the URI to open the notebook and review the leaderboard, load an AutoGluon model from S3, and run predictions. |
| Return value | `str` | Optional status or message returned by the component. |

## Usage Examples 💡

### Basic usage

```python
from kfp import dsl
from kfp_components.components.deployment.automl.notebook_generation import (
    notebook_generation,
)


@dsl.pipeline(name="notebook-generation-pipeline")
def my_pipeline():
    """Example pipeline using notebook generation."""
    notebook_task = notebook_generation(input_param="example")
    return notebook_task
```

## Metadata 🗂️

- **Name**: notebook-generation
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - deployment
  - automl
