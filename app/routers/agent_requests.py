from fastapi import APIRouter, Form, HTTPException, Request

from app.services.exceptions import AgentExecutionError, DocumentNotFoundError

router = APIRouter()


@router.post("/")
async def run_agent(
    request: Request, user_input: str = Form(...), user_id: str = Form(...)
):
    try:
        response = await request.app.state.agent_request_service.run(
            user_input, user_id
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AgentExecutionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"response": response}
