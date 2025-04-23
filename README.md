# Amazon Chatbot

An AI-powered chatbot that analyzes customer reviews and generates tailored responses using Google Gemini models via Vertex AI. Built with FastAPI, containerized with Docker, and deployed on Render.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Docker Setup](#docker-setup)
- [Deployment on Render](#deployment-on-render)
- [API Reference](#api-reference)
- [Validation & Testing](#validation--testing)

## Project Structure

```
amazon-chatbot/
├── static/
│   └── index.html         # Frontend HTML interface
├── app.py                 # FastAPI backend application
├── agent_checkpoint.py    # AI logic and response handling
├── models.py              # LLM integration and model management
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (ignored)
├── .gitignore             # Git ignore rules
└── gen-lang-client-*.json # Vertex AI credentials (ignored)
```

- **FastAPI** serves both the API endpoints and static files.
- Sensitive files (`.env`, credential JSON) are excluded via `.gitignore`.

## Prerequisites

- **Python** 3.11+
- **Docker** (optional, for containerized deployment)
- **Render** account (for deployment)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/amazon-chatbot.git
   cd amazon-chatbot
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\\Scripts\\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your API keys and credentials in a `.env` file:
   ```ini
   SERPER_API_KEY=your_serper_key
   GEMINI_API_KEY=your_gemini_key
   ```

## Running Locally

Start the FastAPI app with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Browse to `http://localhost:8000` to access the chatbot UI.

## Docker Setup

Build and run the Docker container:

```bash
# Build the image
docker build -t amazon-chatbot:latest .

# Run the container
docker run -d \
  --name chatbot \
  -p 8000:8000 \
  amazon-chatbot:latest
```

**Dockerfile highlights**:

- Base image: `python:3.11-slim`
- Non-root user: `appuser`
- Dependencies installed from `requirements.txt`
- Exposes port `8000`
- Entry point uses Uvicorn

## Deployment on Render

1. Connect your GitHub repository to Render as a Docker-based Web Service.
2. Set the deploy region (e.g., Oregon, US).
3. Configure environment variables in the Render dashboard:
   - `SERPER_API_KEY`
   - `GEMINI_API_KEY`
   - Upload `gen-lang-client-*.json` securely for Vertex AI.
4. Trigger a manual or automatic deploy and monitor logs for errors.

## API Reference

### `POST /chat`

Processes customer review data and returns a tailored response.

- **Request Body**: JSON containing review data
- **Response**: JSON with the chatbot's reply

### Static Files

- `GET /static/index.html` serves the frontend UI
- `GET /favicon.ico` serves the site icon
- `/` redirects to `/static/index.html`

## Validation & Testing

- Deployed site was accessed successfully in a browser.
- Verified `/chat` endpoint returns valid responses.
- Checked Render logs: no runtime errors.
- Confirmed favicon loads correctly on the browser tab.

---

_This README provides a concise overview of the Amazon Chatbot project. For questions or contributions, feel free to open an issue or pull request._

