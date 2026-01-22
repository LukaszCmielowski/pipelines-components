# KServe Deployment 🚀

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Deploys the best model using KServe with AutoGluon runtime for production inference.

The KServe Deployment component is an optional step in the AutoML pipeline (controlled by the
`auto_deploy` parameter) that deploys the best model from the evaluation stage using KServe with
a custom AutoGluon runtime. This enables production-ready model serving with automatic scaling,
health checks, and inference endpoints.

The component uses a custom AutoGluon runtime that understands AutoGluon Predictor format and can serve predictions efficiently. This runtime is designed to work seamlessly with KServe's inference service framework.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference_service` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing the inference service information, endpoint URL, and deployment metadata. |
| `model` | `dsl.Input[dsl.Model]` | `None` | Input Model artifact containing the model to deploy (typically the best model from leaderboard-evaluation). |
| `auto_deploy` | `bool` | `False` | Boolean flag to enable automatic model deployment. If `False`, the component skips deployment. |
| `deployment_config` | `dict` | `None` | Optional dictionary with deployment configuration. See [Deployment Configuration](#deployment-configuration) below. |

### Deployment Configuration

The `deployment_config` dictionary supports:

```python
{
    "service_name": "automl-predictor",     # KServe InferenceService name (default: "automl-predictor")
    "namespace": "default",                  # Kubernetes namespace (default: "default")
    "replicas": 1,                          # Number of replicas (default: 1)
    "resources": {                          # Resource limits and requests
        "cpu": "1",                         # CPU limit (default: "1")
        "memory": "2Gi"                     # Memory limit (default: "2Gi")
    },
    "min_replicas": 0,                      # Minimum replicas for autoscaling (default: 0)
    "max_replicas": 10,                     # Maximum replicas for autoscaling (default: 10)
    "target_utilization": 70                # Target CPU utilization for autoscaling (default: 70)
}
```

## Outputs 📤

| Output | Type | Description |
|--------|------|-------------|
| `inference_service` | `dsl.Artifact` | Artifact containing the inference service information, including endpoint URL, service name, namespace, and deployment status. |
| Return value | `str` | A message indicating the completion status of model deployment. |

## Usage Examples 💡

### Basic Deployment

```python
from kfp import dsl
from kfp_components.components.training.automl.model_deployment.kserve_deployment import (
    kserve_deployment,
)

@dsl.pipeline(name="kserve-deployment-pipeline")
def my_pipeline(model):
    """Example pipeline for KServe deployment."""
    with dsl.Condition(auto_deploy == True):
        deploy_task = kserve_deployment(
            model=model,
            auto_deploy=True,
            deployment_config={
                "service_name": "automl-predictor",
                "namespace": "default",
                "replicas": 1
            }
        )
    return deploy_task
```

### Production Deployment with Autoscaling

```python
@dsl.pipeline(name="kserve-production-deployment-pipeline")
def my_pipeline(model):
    """Example pipeline for production deployment with autoscaling."""
    with dsl.Condition(auto_deploy == True):
        deploy_task = kserve_deployment(
            model=model,
            auto_deploy=True,
            deployment_config={
                "service_name": "automl-predictor-prod",
                "namespace": "production",
                "replicas": 2,
                "min_replicas": 1,
                "max_replicas": 10,
                "target_utilization": 70,
                "resources": {
                    "cpu": "2",
                    "memory": "4Gi"
                }
            }
        )
    return deploy_task
```

### Conditional Deployment

```python
@dsl.pipeline(name="conditional-kserve-deployment-pipeline")
def my_pipeline(model, auto_deploy: bool = False):
    """Example pipeline with conditional deployment."""
    with dsl.Condition(auto_deploy == True):
        deploy_task = kserve_deployment(
            model=model,
            auto_deploy=auto_deploy,
            deployment_config={
                "service_name": "automl-predictor",
                "namespace": "default",
                "replicas": 1
            }
        )
    return deploy_task
```

## KServe Inference Service 🎯

The component creates a KServe InferenceService with:

- **Custom AutoGluon Runtime**: Specialized runtime for AutoGluon Predictor models
- **Automatic Scaling**: Horizontal Pod Autoscaling (HPA) based on CPU utilization
- **Health Checks**: Liveness and readiness probes for service health monitoring
- **REST API Endpoint**: Standard KServe REST API for predictions
- **Model Versioning**: Support for model versioning and A/B testing

## Inference Endpoint 🌐

The deployed service provides:

- **REST API**: Standard KServe REST API endpoint
- **Endpoint URL**: `http://{service_name}.{namespace}.svc.cluster.local/v1/models/automl:predict`
- **Prediction Format**: JSON-based request/response format
- **Batch Predictions**: Support for batch prediction requests

## AutoGluon Runtime 🔧

The custom AutoGluon runtime:

- **Predictor Loading**: Efficiently loads AutoGluon Predictor models
- **Preprocessing**: Handles AutoGluon's automatic preprocessing pipeline
- **Inference**: Fast inference using AutoGluon's optimized prediction engine
- **Postprocessing**: Formats predictions according to task type (classification, regression, time-series)

## Future Contributions 🚀

Contribution to KServe with new AutoGluon runtime to be considered. This would involve:

- Creating a custom KServe runtime image for AutoGluon
- Contributing to the KServe project
- Enabling broader AutoGluon model serving capabilities

## Notes 📝

- **Optional Step**: Controlled by `auto_deploy` parameter - set to `False` to skip deployment
- **Custom Runtime**: Uses custom AutoGluon runtime (not standard KServe runtimes)
- **Production Ready**: Deployed models are ready for production inference workloads
- **Auto Scaling**: Supports automatic scaling based on traffic and CPU utilization
- **Model Format**: Deploys AutoGluon Predictor format, which includes all necessary components

## Metadata 🗂️

- **Name**: kserve-deployment
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: KServe, Version: >=0.11.0
- **Tags**:
  - automl
  - model-deployment
  - kserve
  - inference
- **Last Verified**: 2025-01-27 00:00:00+00:00

## Additional Resources 📚

- **AutoML Documentation**: [AutoML README](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/README.md)
- **Components Documentation**: [Components Structure](https://github.com/LukaszCmielowski/architecture-decision-records/blob/autox_arch_docs/documentation/components/automl/components.md)
- **KServe Documentation**: [KServe Website](https://kserve.github.io/website/)
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
