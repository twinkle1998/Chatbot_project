import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from agent_checkpoint import run_agent, end_session
import uvicorn

# Suppress Pydantic UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

app = FastAPI()

# Mount the static directory to serve index.html
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Allow all origins (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected JSON format for /chat
class ReviewRequest(BaseModel):
    name: str
    date: str
    product: str
    input: str  # Changed 'review' to 'input' to match frontend
    session_id: str

# Define the expected JSON format for /end_chat
class EndChatRequest(BaseModel):
    session_id: str

@app.post("/chat")
async def analyze_review(data: ReviewRequest):
    input_data = {
        "cust_name": data.name,
        "purch_date": data.date,
        "product": data.product,
        "input": data.input,
        "session_id": data.session_id
    }
    try:
        result = run_agent(input_data)
        return {"reviewed_response": result["reviewed_response"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/end_chat")
async def end_chat(data: EndChatRequest):
    try:
        result = end_session(data.session_id)
        return result
    except Exception as e:
        return {"error": str(e)}

# Redirect root to static index.html
@app.get("/")
async def serve_index():
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
