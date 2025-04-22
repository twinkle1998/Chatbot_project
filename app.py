import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from agent_checkpoint import run_agent, end_session
import uvicorn
import uuid

# Suppress Pydantic UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

app = FastAPI()

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

# Define the expected JSON format
class ReviewRequest(BaseModel):
    name: str
    date: str
    product: str
    input: str
    session_id: str

@app.post("/chat")
async def analyze_review(data: ReviewRequest):
    input_data = {
        "cust_name": data.name,
        "purch_date": data.date,
        "product": data.product,
        "input": data.input
    }
    try:
        result = run_agent(input_data, data.session_id)
        return {"reviewed_response": result["reviewed_response"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/end_chat")
async def end_chat(data: dict):
    session_id = data.get("session_id", "")
    try:
        end_session(session_id)
        return {"status": "Chat ended successfully"}
    except Exception as e:
        return {"error": str(e)}

# Redirect root to background page
@app.get("/")
async def serve_index():
    return RedirectResponse(url="/static/background.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
