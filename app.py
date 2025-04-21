from fastapi import FastAPI
import uvicorn
from chatbot import ChatBot  # Importing the ChatBot class for the /chat endpoint

app = FastAPI()

# Root route for the "frontend" response
@app.get("/")
async def root():
    return {"message": "Welcome to my chatbot project!"}

# Chat endpoint that uses the chatbot module
@app.get("/chat")
async def chat(message: str):
    chatbot = ChatBot()
    response = chatbot.get_response(message)
    return {"response": response}

# Favicon route to handle browser requests
@app.get("/favicon.ico")
async def favicon():
    return {"detail": "No favicon available"}

# Run the app with Uvicorn for Render deployment
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)