from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["boto3"],
)
def test_data_loader(
    test_data_reference: dict,
    test_data: dsl.Output[dsl.Artifact] = None
) -> str:
    """Test Data Loader component.

    TODO: Add a detailed description of what this component does.

    Args:
        input_param: Description of the component parameter.
        # Add descriptions for other parameters

    Returns:
        Description of what the component returns.
    """
    import os
    import sys
    import json
    import logging
    from dataclasses import dataclass

    import boto3


    logger = logging.getLogger("Test Data Loader component logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    @dataclass
    class DataReference:
        endpoint: str
        region: str
        bucket: str
        path: str

    if not test_data_reference:
        return "No test data reference provided."

    test_data_reference = DataReference(**test_data_reference)

    def get_test_data_s3():
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

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
            endpoint_url=test_data_reference.endpoint,
            region_name=test_data_reference.region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        if test_data_reference.path.endswith(".json"):
            logger.info(f"Fetching test data from S3: bucket={test_data_reference.bucket}, path={test_data_reference.path}")
            try:
                logger.info(f"Starting download to {test_data.path}")
                s3_client.download_file(
                    test_data_reference.bucket,
                    test_data_reference.path,
                    test_data.path
                )
                logger.info("Download completed successfully")
            except Exception as e:
                logger.error("Failed to fetch %s: %s", test_data_reference.path, e)
                raise
        else:
            logger.error("Test data must be a json file: %s", test_data_reference.path)
            raise

    get_test_data_s3()

    return "Test data loading completed."



if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        test_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
