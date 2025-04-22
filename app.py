import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from agent_checkpoint import run_agent, extract_name
import uvicorn
from starlette.middleware.sessions import SessionMiddleware
import uuid

# Suppress Pydantic UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Mount the static directory to serve index.html
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Allow all origins (for local HTML frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for conversation history
conversation_store = {}

# Define the expected JSON format
class ReviewRequest(BaseModel):
    name: str
    date: str
    product: str
    review: str
    session_id: str = None

@app.post("/chat")
async def analyze_review(data: ReviewRequest):
    # Generate or retrieve session ID
    session_id = data.session_id or str(uuid.uuid4())
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    input_data = {
        "cust_name": data.name,
        "purch_date": data.date,
        "product": data.product,
        "review": data.review
    }

    try:
        result = run_agent(input_data, session_id, conversation_store[session_id])
        conversation_store[session_id] = result["conversation_history"]
        return {
            "reviewed_response": result["reviewed_response"],
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end_chat")
async def end_chat(session_id: str):
    if session_id in conversation_store:
        del conversation_store[session_id]
    return {"message": "Chat ended successfully"}

# Redirect root to static index.html
@app.get("/")
async def serve_index():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
