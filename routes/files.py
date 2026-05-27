from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from services.exceptions import (
    DocumentParseError,
    FileTooLargeError,
    UnsupportedFileFormatError,
)

router = APIRouter()


@router.post("/")
async def upload_text(request: Request, file: UploadFile = File(...)):
    content = await file.read()

    try:
        result = request.app.state.file_service.upload(
            filename=file.filename,
            content_type=file.content_type,
            content=content,
        )
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (UnsupportedFileFormatError, DocumentParseError) as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    return {"status": result.status, "length": result.length, "user_id": result.user_id}
