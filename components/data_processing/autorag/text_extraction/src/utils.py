from pathlib import Path


def process_document(file_path_str: str, output_dir_str: str) -> bool:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    try:
        path = Path(file_path_str)
        out_dir = Path(output_dir_str)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.accelerator_options = AcceleratorOptions(device="cpu", num_threads=1)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(path)
        markdown_content = result.document.export_to_markdown()
        output_file = out_dir / f"{path.name}.md"
        output_file.write_text(markdown_content, encoding="utf-8")
        return True
    except Exception as e:
        import logging
        logging.getLogger("Text Extraction component logger").error(
            "Failed to process %s: %s", file_path_str, e
        )
        return False