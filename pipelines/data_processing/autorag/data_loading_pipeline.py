from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.autorag.document_loader.component import document_loader
from kfp_components.components.data_processing.autorag.test_data_loader.component import test_data_loader
from kfp_components.components.data_processing.autorag.text_extraction.component import text_extraction


@dsl.pipeline(
    name="AutoRAG Data Processing Pipeline", description="Pipeline to load test data and documents for AutoRAG."
)
def data_loading_pipeline(
    secret_name: str,
    test_data_bucket_name: str,
    test_data_path: str,
    input_data_bucket_name: str,
    input_data_path: str,
    sampling_config: dict,
):
    """Defines a pipeline to load and sample input data for AutoRAG.

    Args:
        secret_name : str
            The name of the secret to fetch the environment variables with S3 credentials from.

        test_data_bucket_name : str
            S3 bucket that contains the test data file.

        test_data_path : str
            S3 object key to the JSON test data file.

        input_data_bucket_name : str
            Name of the S3 bucket containing input data.

        input_data_path : str
            Path to folder with input documents within bucket.

        sampling_config : dict
            Optional sampling configuration dictionary.
    """
    test_data_loader_task = test_data_loader(
        test_data_bucket_name=test_data_bucket_name,
        test_data_path=test_data_path,
    )

    document_loader_task = document_loader(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_path,
        test_data=test_data_loader_task.outputs["test_data"],
        sampling_config=sampling_config,
    )

    text_extraction_task = text_extraction(documents=document_loader_task.outputs["sampled_documents"])

    for task in [test_data_loader_task, document_loader_task, text_extraction_task]:
        use_secret_as_env(
            task,
            secret_name=secret_name,
            secret_key_to_env={
                "aws_access_key_id": "AWS_ACCESS_KEY_ID",
                "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
                "endpoint_url": "AWS_ENDPOINT_URL",
                "aws_region_name": "AWS_REGION",
            },
        )


if __name__ == "__main__":
    import pathlib

    from kfp.compiler import Compiler

    output_path = pathlib.Path(__file__).with_name("data_loading_pipeline.yaml")
    Compiler().compile(pipeline_func=data_loading_pipeline, package_path=str(output_path))
    print(f"Pipeline compiled to {output_path}")
