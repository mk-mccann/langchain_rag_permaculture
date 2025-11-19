import os
from pathlib import Path
import pymupdf.layout
import pymupdf4llm


def pdf_to_markdown(pdf_path: str, output_path: str) -> None:
    """Convert a PDF file to markdown while preserving page numbers."""
    doc = pymupdf.open(pdf_path)
    md_text = pymupdf4llm.to_markdown(doc, page_chunks=True, write_images=True, header=False)
    Path(output_path).write_bytes(md_text.encode())
    return


def process_folder(input_folder: str, output_folder: str) -> None:
    """Process all PDF files in a folder and convert them to markdown."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_path.glob('*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return
    
    for pdf_file in pdf_files:
        md_filename = pdf_file.stem + '.md'

        # save markdown file in a directory named after the pdf
        output_path = os.path.join(output_folder, pdf_file.stem)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        md_filepath = os.path.join(output_path, md_filename)
        
        print(f"Converting {pdf_file.name} to markdown...")
        try:
            pdf_to_markdown(str(pdf_file), str(md_filepath))
            print(f"✓ Created {md_filename}")
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {e}")


if __name__ == "__main__":
    INPUT_FOLDER = "../data/raw/pdfs"
    OUTPUT_FOLDER = "../data/raw/markdown_output"
    
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
