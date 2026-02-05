from kfp import dsl


@dsl.component(
    base_image="quay.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9:rhoai-3.2",
    # base_image="localhost:5000/autogluon-py312:v3",
    # packages_to_install=["numpy", "pandas", "boto3"],
)
def automl_data_loader(file_key: str, bucket_name: str, full_dataset: dsl.Output[dsl.Dataset]):  # noqa: D417
    """Automl Data Loader component.

    TODO: Add a detailed description of what this component does.

    Args:
        input_param: Description of the component parameter.
        # Add descriptions for other parameters

    Returns:
        Description of what the component returns.
    """
    import os

    import boto3
    import pandas as pd

    def download_from_s3():
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
        region_name = os.environ.get("AWS_DEFAULT_REGION")

        if (access_key and not secret_key) or (secret_key and not access_key):
            raise ValueError(
                "S3 credentials misconfigured: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must either "
                "both be set and non-empty, or both be unset. Check the 's3-secret' Kubernetes secret."
            )
        if not access_key and not secret_key:
            raise ValueError(
                "S3 credentials missing: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be provided via "
                "the 's3-secret' Kubernetes secret when using s3:// dataset URIs."
            )

        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        s3_client.download_file(bucket_name, file_key, full_dataset.path)

    download_from_s3()
    full_dataset_df = pd.read_csv(full_dataset.path)

    # Sampling
    sampled_dataset = full_dataset_df.sample(frac=0.5, random_state=42)

    sampled_dataset.to_csv(full_dataset.path, index=False)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        automl_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
