import fitz

def extract_text_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        return f"Error while opening pdf file: {e}"
    text=""
    for page in doc:
        text += page.get_text()
    doc.close()

    return text

text = extract_text_pdf("data/Understanding LSTM Networks -- colah's blog.pdf")

#print(text[:1000])