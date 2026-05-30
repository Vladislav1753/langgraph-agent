from fastapi import APIRouter, File, Request, UploadFile


router = APIRouter()


@router.post("/")
async def upload_text(request: Request, file: UploadFile = File(...)):
    content = await file.read()

    result = await request.app.state.file_service.upload(
        filename=file.filename,
        content_type=file.content_type,
        content=content,
    )

    return {"status": result.status, "length": result.length, "user_id": result.user_id}
