# Model Deployment Components

This directory contains components in the **Model Deployment** category for AutoML workflows:

- [Model Registry](./model-registry/README.md): Registers the best model with RHOAI Model Registry for versioning and deployment management.
- [KServe Deployment](./kserve-deployment/README.md): Deploys the best model using KServe with AutoGluon runtime for production inference.

## Overview

The Model Deployment components are optional steps in the AutoML pipeline that enable model versioning, registration, and production serving. These components work together to provide a complete model deployment workflow:

1. **Model Registry**: Registers trained models with metadata for tracking and version control
2. **KServe Deployment**: Deploys models as production-ready inference services

Both components are controlled by pipeline parameters (`auto_register` and `auto_deploy`) and can be conditionally executed based on deployment requirements.

## Integration with AutoML Pipeline

These components are typically used at the end of the AutoML pipeline workflow:

```text
Data Processing → Model Training → Model Evaluation → [Model Deployment]
                                                         ├── Model Registry (optional)
                                                         └── KServe Deployment (optional)
```

## Usage

Both deployment components are optional and controlled by boolean flags:

- **Model Registry**: Set `auto_register=True` to enable model registration
- **KServe Deployment**: Set `auto_deploy=True` to enable model deployment

See individual component READMEs for detailed usage examples and configuration options.
