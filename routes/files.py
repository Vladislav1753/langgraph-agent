from fastapi import APIRouter, File, UploadFile, HTTPException, Request
import logging
import uuid
from doc_loader import extract_text_pdf_bytes
from config import MAX_FILE_SIZE

router = APIRouter()


@router.post("/")
async def upload_text(request: Request, file: UploadFile = File(...)):
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large, max size is 5 MB")

    user_id = str(uuid.uuid4())

    if file.content_type == "application/pdf":
        text = extract_text_pdf_bytes(content)
    else:
        try:
            text = content.decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=415, detail="Unsupported file format")

    request.app.state.user_files[user_id] = text[:3000]
    logging.info(f"User {user_id} uploaded file {file.filename}")

    return {
        "status": "ok",
        "length": len(request.app.state.user_files[user_id]),
        "user_id": user_id
    }