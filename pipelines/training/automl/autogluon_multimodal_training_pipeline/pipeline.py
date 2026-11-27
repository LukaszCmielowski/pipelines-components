"""Build the prototype image classification pipeline."""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import graph_component
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.automl.multimodal_image_data_loader import multimodal_image_data_loader
from kfp_components.components.training.automl.autogluon_multimodal_models_training import (
    autogluon_multimodal_models_training,
)


class MultimodalOutputs(NamedTuple):
    """Artifacts and model location published by the pipeline."""

    model: dsl.Model
    metrics: dsl.Metrics
    data_quality: dsl.Artifact
    model_uri: str


def create_autogluon_multimodal_training_pipeline(
    runtime_image: str, *, use_gpu: bool = True
) -> graph_component.GraphComponent:
    """Build the prototype graph for a prebuilt AutoMM runtime image."""
    if not runtime_image.strip():
        raise ValueError("Provide the prebuilt AutoMM runtime image.")

    @dsl.pipeline(
        name="autogluon-multimodal-training-pipeline",
        pipeline_config=dsl.PipelineConfig(
            workspace=dsl.WorkspaceConfig(
                size="8Gi",
                kubernetes=dsl.KubernetesWorkspaceConfig(pvcSpecPatch={"accessModes": ["ReadWriteOnce"]}),
            )
        ),
    )
    def pipeline(
        train_data_secret_name: str,
        train_data_bucket_name: str,
        train_data_file_key: str,
        label_column: str,
        image_columns: list[str],
        task_type: str,
        task_family: str = "image_prediction",
    ) -> MultimodalOutputs:
        """Train a single AutoMM image classifier and publish an offline-loadable predictor.

        Args:
            train_data_secret_name: Name of the S3 Connection Secret in the run namespace.
            train_data_bucket_name: Bucket containing the manifest and images.
            train_data_file_key: S3 key of the CSV manifest. Image keys are relative to its directory.
            label_column: Label column in the manifest.
            image_columns: Exactly one column containing relative JPEG or PNG image keys.
            task_type: binary or multiclass.
            task_family: image_prediction, the only task family supported by this prototype.

        Returns:
            Standalone model artifact, metrics, aggregate data quality, and model URI.
        """
        loader = multimodal_image_data_loader(
            bucket_name=train_data_bucket_name,
            file_key=train_data_file_key,
            image_columns=image_columns,
            label_column=label_column,
            task_type=task_type,
            task_family=task_family,
            workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        )
        loader.set_container_image(runtime_image).set_caching_options(False)
        loader.set_cpu_request("1").set_memory_request("1Gi").set_memory_limit("2Gi")
        use_secret_as_env(
            loader,
            secret_name=train_data_secret_name,
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
                "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            },
        )
        use_secret_as_env(
            loader,
            secret_name=train_data_secret_name,
            secret_key_to_env={"AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION"},
            optional=True,
        )
        trainer = autogluon_multimodal_models_training(split_manifest_path=loader.outputs["Output"], use_gpu=use_gpu)
        trainer.set_container_image(runtime_image).set_caching_options(False)
        trainer.set_cpu_request("2").set_memory_request("4Gi").set_memory_limit("8Gi")
        if use_gpu:
            trainer.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        return MultimodalOutputs(
            model=trainer.outputs["model"],
            metrics=trainer.outputs["metrics"],
            data_quality=loader.outputs["data_quality"],
            model_uri=trainer.outputs["Output"],
        )

    return pipeline


# A graph for repository validation. Use the factory or CLI to compile a runnable graph
# with the published prototype runtime image.
autogluon_multimodal_training_pipeline = create_autogluon_multimodal_training_pipeline("python:3.12-slim")


if __name__ == "__main__":
    import argparse

    from kfp.compiler import Compiler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Prebuilt AutoMM runtime image, preferably by digest.")
    parser.add_argument("--output", default="autogluon-multimodal-training-pipeline.yaml")
    parser.add_argument("--cpu", action="store_true", help="Compile without a GPU request for development.")
    args = parser.parse_args()
    Compiler().compile(
        create_autogluon_multimodal_training_pipeline(args.image, use_gpu=not args.cpu),
        package_path=args.output,
    )
