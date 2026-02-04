from kfp import dsl
from kfp.kubernetes import use_secret_as_env

from components.data_processing.autorag.document_loader.component import document_loader
from components.data_processing.autorag.test_data_loader.component import test_data_loader
from components.data_processing.autorag.text_extraction.component import text_extraction


@dsl.pipeline(
    name="AutoRAG Data Processing Pipeline",
    description="Pipeline to load test data and documents for AutoRAG."
)
def autorag_data_loading_pipeline(
    secret_name: str,
    test_data_bucket_name: str,
    test_data_path: str,
    input_data_bucket_name: str,
    input_data_path: str,
    sampling_config: dict,
):
    test_data_loader_task = test_data_loader(
        test_data_bucket_name=test_data_bucket_name,
        test_data_path=test_data_path,
    )
    
    document_loader_task = document_loader(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_path,
        test_data=test_data_loader_task.outputs["test_data"],
        sampling_config=sampling_config
    )

    text_extraction_task = text_extraction(
        documents=document_loader_task.outputs["sampled_documents"]
    )

    for task in [test_data_loader_task, document_loader_task]:
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
    from kfp.compiler import Compiler
    import pathlib

    output_path = pathlib.Path(__file__).with_name("data_loading_pipeline.yaml")
    Compiler().compile(
        pipeline_func=autorag_data_loading_pipeline,
        package_path=str(output_path)
    )
    print(f"Pipeline compiled to {output_path}")
