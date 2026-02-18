from kfp import dsl


@dsl.component(base_image="quay.io/wnowogorski-org/autorag_data_loading:latest")
def text_extraction(
    sampled_documents_descriptor: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Output[dsl.Artifact],
):
    """Text Extraction component.

    Reads the sampled_documents_descriptor YAML (from document_loader), fetches
    the listed documents from S3, and extracts text using the docling library.

    Args:
        sampled_documents_descriptor: Input artifact containing
            sampled_documents_descriptor.yaml with bucket, prefix, and documents list.
        extracted_text: Output artifact where the extracted text content will be stored.
    """
    import logging
    import os
    import sys
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import boto3
    import yaml
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    SAMPLED_DOCUMENTS_DESCRIPTOR_FILENAME = "sampled_documents_descriptor.yaml"

    logger = logging.getLogger("Text Extraction component logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}

    descriptor_root = Path(sampled_documents_descriptor.path)
    if descriptor_root.is_dir():
        descriptor_path = descriptor_root / SAMPLED_DOCUMENTS_DESCRIPTOR_FILENAME
    else:
        descriptor_path = descriptor_root

    if not descriptor_path.exists():
        raise FileNotFoundError(f"Descriptor not found: {descriptor_path}")

    with open(descriptor_path) as f:
        descriptor = yaml.safe_load(f)

    bucket = descriptor["bucket"]
    documents = descriptor["documents"]

    s3_creds = {
        k: os.environ.get(k)
        for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "AWS_REGION"]
    }
    for k, v in s3_creds.items():
        if v is None:
            raise ValueError(
                f"{k} environment variable not set. Check if kubernetes secret was configured properly."
            )

    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_creds["AWS_ENDPOINT_URL"],
        region_name=s3_creds["AWS_REGION"],
        aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
    )

    with tempfile.TemporaryDirectory() as download_dir:
        download_path = Path(download_dir)
        for doc in documents:
            key = doc["key"]
            local_path = download_path / key
            try:
                logger.info("Downloading %s", key)
                s3_client.download_file(bucket, key, str(local_path))
            except Exception as e:
                logger.error("Failed to fetch %s: %s", key, e)
                raise

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        output_dir = Path(extracted_text.path)
        output_dir.mkdir(parents=True, exist_ok=True)

        files_to_process = [
            f for f in download_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        logger.info("Starting text extraction for %d documents.", len(files_to_process))

        def process_file(file_path: Path):
            try:
                logger.info("Processing document: %s", file_path.name)

                result = converter.convert(file_path)

                markdown_content = result.document.export_to_markdown()

                output_file_name = f"{file_path.stem}.md"
                output_file_path = output_dir / output_file_name

                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)

                logger.info("Successfully extracted text from %s", file_path.name)
                return True
            except Exception as e:
                logger.error("Failed to process %s: %s", file_path.name, e)
                return False

        max_workers = min(len(files_to_process), (os.cpu_count() or 1) * 2) if files_to_process else 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, files_to_process))

    processed_count = sum(1 for r in results if r)
    error_count = len(results) - processed_count

    summary = f"Text extraction completed. Total processed: {processed_count}, Errors: {error_count}."
    logger.info(summary)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        text_extraction,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
