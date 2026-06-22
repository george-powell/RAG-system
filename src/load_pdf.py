"""
Load PDF documents and extract page text for ingestion.
"""

from pypdf import PdfReader
import os

def load_pdf(folder):
    """
    Load PDF files from a folder and extract page text.

    Args:
        folder (str): Directory containing PDF files.

    Returns:
        list: Document dictionaries containing text, source file, and page number.
    """
    documents = []

    # Process each PDF in the folder
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            print(f"Loading: {path}")

            reader = PdfReader(path)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if text:
                    # Store page text with source metadata
                    documents.append({
                        "text": text,
                        "source": file,
                        "page": page_num + 1
                    })

    print("\nDONE LOADING ALL PDFs")
    print(f"Total chunks (pages): {len(documents)}")
    return documents