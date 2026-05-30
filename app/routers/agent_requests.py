from fastapi import APIRouter, Form, Request

router = APIRouter()


@router.post("/")
async def run_agent(
    request: Request, user_input: str = Form(...), user_id: str = Form(...)
):
    response = await request.app.state.agent_request_service.run(user_input, user_id)

    return {"response": response}
