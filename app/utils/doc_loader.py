import fitz
import logging
from typing import Union


logger = logging.getLogger(__name__)


def extract_text_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.exception("Error while opening PDF file")
        return f"Error while opening pdf file: {e}"
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    return text


def extract_text_pdf_bytes(data: bytes) -> Union[str, None]:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        logger.exception("Error reading PDF bytes")
        return None
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    return text
