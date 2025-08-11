from fastapi import APIRouter, Form, HTTPException, Request
from langchain_core.messages import HumanMessage

router = APIRouter()


@router.post("/")
async def run_agent(request: Request, user_input: str = Form(...), user_id: str = Form(...)):
    if user_id not in request.app.state.user_files:
        raise HTTPException(status_code=404, detail="No document uploaded for this user_id")

    messages = [HumanMessage(content=user_input)]
    try:
        new_state = await request.app.state.agent.ainvoke({
            "messages": messages,
            "text": request.app.state.user_files[user_id],
            "user_id": user_id
        })
    except Exception as e:
        logging.error(f"Agent error: {e}")
        raise HTTPException(status_code=503, detail="Error while using LLM-agent")

    response = new_state["messages"][-1].content
    return {"response": response}