import os
import json
from crewai import LLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load credentials from environment variable
vertex_credentials_json = os.getenv("VERTEX_CREDENTIALS")
if not vertex_credentials_json:
    raise ValueError("VERTEX_CREDENTIALS environment variable not set")

# Define class for LLM
class google_model:
    def gemini_2_flash():
        return LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.7,
            vertex_credentials=vertex_credentials_json
        )
        
    def gemini_2_flash_lite():
        return LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature=0.7,
            vertex_credentials=vertex_credentials_json
        )
        
    def gemini_pro():
        return LLM(
            model="gemini/gemini-2.5-pro-exp-03-25",
            temperature=0.7,
            vertex_credentials=vertex_credentials_json
        )

class local_model:
    def mistral():
        return LLM(
            model="ollama/mistral:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
        )
        
    def gemma():
        return LLM(
            model="ollama/gemma3:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
        )
        
    def ollama():
        return LLM(
            model="ollama/llama3.2:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
        )
    
    def cogito():
        return LLM(
            model="ollama/cogito:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
        )