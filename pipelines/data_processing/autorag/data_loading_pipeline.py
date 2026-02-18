from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.autorag.documents_sampling.component import documents_sampling
from kfp_components.components.data_processing.autorag.test_data_loader.component import test_data_loader
from kfp_components.components.data_processing.autorag.text_extraction.component import text_extraction


@dsl.pipeline(
    name="AutoRAG Data Processing Pipeline", description="Pipeline to load test data and documents for AutoRAG."
)
def data_loading_pipeline(
    test_data_secret_name: str = "autorag-input-data-secret",
    input_data_secret_name: str = "autorag-input-data-secret",
    test_data_bucket_name: str = "autorag-test-bucket",
    test_data_key: str = "benchmark.json",
    input_data_bucket_name: str = "autorag-test-bucket",
    sampling_config: dict = {"max_size_gigabytes": 1},
    input_data_key: str = "documents",
):
    """Defines a pipeline to load and sample input data for AutoRAG.

    Args:
        test_data_secret_name : str
            Name of the secret containing environment variables with S3 credentials
            used to access the test data.

        input_data_secret_name : str
            Name of the secret containing environment variables with S3 credentials
            used to access the input data.

        test_data_bucket_name : str
            S3 bucket that contains the test data file.

        test_data_key : str
            S3 object key to the JSON test data file.

        input_data_bucket_name : str
            Name of the S3 bucket containing input data.

        input_data_key : str
            Path to folder with input documents within bucket.

        sampling_config : dict
            Optional sampling configuration dictionary.
    """
    test_data_loader_task = test_data_loader(
        test_data_bucket_name=test_data_bucket_name,
        test_data_path=test_data_key,
    )

    documents_sampling_task = documents_sampling(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_key,
        test_data=test_data_loader_task.outputs["test_data"],
        sampling_config=sampling_config,
    )

    documents_sampling_task.set_caching_options(enable_caching=False)
    test_data_loader_task.set_caching_options(enable_caching=False)

    text_extraction_task = text_extraction(
        sampled_documents_descriptor=documents_sampling_task.outputs["sampled_documents"],
    )

    for task, secret_name in zip(
        [test_data_loader_task, documents_sampling_task, text_extraction_task],
        [test_data_secret_name, input_data_secret_name, input_data_secret_name],
    ):
        use_secret_as_env(
            task,
            secret_name=secret_name,
            secret_key_to_env={
                "aws_access_key_id": "AWS_ACCESS_KEY_ID",
                "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
                "endpoint_url": "AWS_S3_ENDPOINT",
                "aws_region_name": "AWS_DEFAULT_REGION",
            },
        )


if __name__ == "__main__":
    import pathlib

    from kfp.compiler import Compiler

    output_path = pathlib.Path(__file__).with_name("data_loading_pipeline.yaml")
    Compiler().compile(pipeline_func=data_loading_pipeline, package_path=str(output_path))
    print(f"Pipeline compiled to {output_path}")
