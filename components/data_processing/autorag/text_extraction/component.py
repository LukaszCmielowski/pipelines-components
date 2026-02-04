from kfp.dsl import component, Input, Output, Artifact


@component(base_image="quay.io/wnowogorski-org/autorag_data_loading:latest")
def text_extraction(
    documents: Input[Artifact],
    extracted_text: Output[Artifact],
):
    """Text Extraction component.

    Extracts text from provided documents (PDF, DOCX, PPTX, MD, HTML, TXT) using the docling library.

    Args:
        documents: Input artifact containing the documents to process.
        extracted_text: Output artifact where the extracted text content will be stored.

    Returns:
        A message indicating the completion status and processing statistics.
    """
    import os
    import logging
    import sys
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    logger = logging.getLogger("Text Extraction component logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    input_dir = Path(documents.path)
    output_dir = Path(extracted_text.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}

    if not input_dir.exists():
        msg = f"Input directory {input_dir} does not exist."
        logger.error(msg)
        return msg

    files_to_process = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    logger.info(f"Starting text extraction for {len(files_to_process)} documents.")

    def process_file(file_path: Path):
        try:
            logger.info(f"Processing document: {file_path.name}")

            result = converter.convert(file_path)

            markdown_content = result.document.export_to_markdown()

            output_file_name = f"{file_path.stem}.md"
            output_file_path = output_dir / output_file_name

            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            logger.info(f"Successfully extracted text from {file_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")
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
