import fitz
from typing import Union

def extract_text_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        return f"Error while opening pdf file: {e}"
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    return text


def extract_text_pdf_bytes(data: bytes) -> Union[str, None]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    return text