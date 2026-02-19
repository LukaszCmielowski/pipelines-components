from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
)
def automl_data_loader(file_key: str, bucket_name: str, full_dataset: dsl.Output[dsl.Dataset]) -> NamedTuple(
    "outputs", sample_config=dict
):
    """Downloads a dataset from S3, samples 50% of the rows, and saves the sample to the output artifact.

    Args:
        file_key (str): The S3 object key (path) for the dataset file.
        bucket_name (str): The S3 bucket containing the dataset file.
        full_dataset (dsl.Output[dsl.Dataset]): Output artifact where the sampled dataset (CSV) will be written.

    Returns:
        NamedTuple: Contains a sample configuration dictionary.
    """
    import os

    import boto3
    import pandas as pd

    # set constants
    DEFAULT_N_SAMPLES = 500
    DEFAULT_RANDOM_STATE = 42

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
    sampled_dataset = full_dataset_df.sample(n=DEFAULT_N_SAMPLES, random_state=DEFAULT_RANDOM_STATE)

    sampled_dataset.to_csv(full_dataset.path, index=False)
    return NamedTuple("outputs", sample_config=dict)(sample_config={"n_samples": DEFAULT_N_SAMPLES})


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        automl_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
