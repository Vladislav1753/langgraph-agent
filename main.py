import logging
from fastapi import FastAPI
from agent import graph

from cachetools import TTLCache
from routes import files, agent_requests


app = FastAPI()

app.include_router(files.router, prefix='/files')
app.include_router(agent_requests.router, prefix='/agent-request')

@app.on_event("startup")
def agent_startup():
    app.state.agent = graph.compile()
    app.state.user_files = TTLCache(maxsize=100, ttl=3600)
    logging.basicConfig(level=logging.INFO)
