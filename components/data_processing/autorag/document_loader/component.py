from kfp.dsl import component, Input, Output, Artifact


@component(base_image="quay.io/wnowogorski-org/autorag_data_loading:latest")
def document_loader(
    input_data_bucket_name: str,
    input_data_path: str,
    test_data: Input[Artifact] = None,
    sampling_config: dict = None,
    sampled_documents: Output[Artifact] = None,
):
    """Document Loader component.

    Loads documents from S3 and performs sampling.

    Args:
        input_data_bucket_name: str
            Name of the S3 bucket containing input data.

        input_data_path: str
            Path to folder with input documents within bucket.

        test_data: dict
            Optional artifact containing test data for sampling.

        sampling_config: dict
            Optional sampling configuration dictionary.

        sampled_documents: Output[Artifact]
            artifact containing downloaded documents.
    """
    import os
    import sys
    import json
    import logging

    import boto3

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}
    MAX_SIZE_BYTES = 1024**3  # 1 GB

    logger = logging.getLogger("Document Loader component logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    if sampling_config is None:
        sampling_config = {}

    def get_test_data_docs_names(test_data_artifact: Input[Artifact]) -> list[str]:
        if test_data_artifact is None:
            return []
        with open(test_data.path, "r") as f:
            benchmark = json.load(f)

        docs_names = []
        for question in benchmark:
            docs_names.extend(question["correct_answer_document_ids"])

        return docs_names

    def download_docs_s3():
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        region = os.environ.get("AWS_REGION")

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
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        contents = s3_client.list_objects_v2(
            Bucket=input_data_bucket_name,
            Prefix=input_data_path,
        ).get("Contents", [])

        supported_files = [c for c in contents if c["Key"].endswith(tuple(SUPPORTED_EXTENSIONS))]
        if not supported_files:
            raise Exception("No supported documents found.")

        test_data_docs_names = get_test_data_docs_names(test_data)

        supported_files.sort(key=lambda c: c["Key"] not in test_data_docs_names)

        total_size = 0
        documents_to_download = []

        for file in supported_files:
            if total_size + file["Size"] > MAX_SIZE_BYTES:
                continue
            documents_to_download.append(file)
            total_size += file["Size"]

        os.makedirs(sampled_documents.path, exist_ok=True)
        for file_info in documents_to_download:
            key = file_info["Key"]
            safe_name = key.replace("/", "__")
            local_path = os.path.join(sampled_documents.path, safe_name)

            try:
                logger.info(f"Downloading {key} to {local_path}")
                s3_client.download_file(input_data_bucket_name, key, local_path)
            except Exception as e:
                logger.error("Failed to fetch %s: %s", key, e)
                raise

    download_docs_s3()


if __name__ == "__main__":
    from kfp.compiler import Compiler
    import sys

    Compiler().compile(
        document_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
