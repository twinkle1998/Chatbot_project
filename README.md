# Technical Architecture
---

## Backend
---

- **Framework:** FastAPI for RESTful API endpoints (`/chat`, `/reply`, `/end_chat`).
- **Server:** Uvicorn for high-performance ASGI server implementation.
- **AI Framework:** CrewAI with four agents (Sentiment Analysis, Sentiment Review, Response Generation, Polishing) powered by Gemini 2.0 Flash and Flash-Lite.
- **Tools:** SerperDevTool for web searches to include contemporary fictional character references in positive/neutral responses (e.g., “Like Miles Morales, you’ve got great taste!”).
- **Deployment:** Docker containerized application hosted on Render for scalability and reliability.

## Frontend
---

- **Technologies:** HTML, CSS, JavaScript (transitioned from React for simplicity).
- **UI Features:** Clean, cheerful interface with a chat widget, tabbed navigation (Home, Messages), and responsive design for mobile and desktop.
- **Enhanced UX:** Delayed message bubbles and WhatsApp-like input behavior for a natural, engaging experience.

## Data
---

- **Dataset:** Amazon US Customer Reviews Dataset (Kaggle, 449,172 records, 148 product categories).
- **Preprocessing:** Sampled 2,000 reviews per rating (1–5 stars), cleaned text (removed HTML tags, stopwords, non-alphabetic characters), normalized (lowercase, standardized formats), and performed tokenization/lemmatization.
- **EDA:** Identified sentiment patterns (1–2 stars: negative, 3: neutral, 4–5: positive) and lexical trends via word clouds (e.g., “great” in positive, “delay” in negative reviews).

# Setup Instructions
---

## Prerequisites

1. Python 3.8+
2. Docker
3. Git
4. Render account (for deployment)
5. API Keys:
   - Serper API (`SERPER_API_KEY`)
   - Gemini API (`GEMINI_API_KEY`)

## Local Setup

### Clone the Repository
```bash
git clone https://github.com/<your-username>/amazon-chatbot.git
cd amazon-chatbot


