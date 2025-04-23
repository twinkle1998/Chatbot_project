# Technical Architecture
---
## Backend
---
**Framework:** FastAPI for RESTful API endpoints (/chat, /reply, /end_chat).

**Server:** Uvicorn for high-performance ASGI server implementation.

**AI Framework:** CrewAI with four agents (Sentiment Analysis, Sentiment Review, Response Generation, Polishing) powered by Gemini 2.0 Flash and Flash-Lite.

**Tools:** SerperDevTool for web searches to include contemporary fictional character references in positive/neutral responses (e.g., “Like Miles Morales, you’ve got great taste!”).

**Deployment:** Docker containerized application hosted on Render for scalability and reliability.

## Frontend

**Technologies:** HTML, CSS, JavaScript (transitioned from React for simplicity).



**UI Features:** Clean, cheerful interface with a chat widget, tabbed navigation (Home, Messages), and responsive design for mobile and desktop.



**Enhanced UX:** Delayed message bubbles and WhatsApp-like input behavior for a natural, engaging experience.

## Data


**Dataset:** Amazon US Customer Reviews Dataset (Kaggle, 449,172 records, 148 product categories).



**Preprocessing:** Sampled 2,000 reviews per rating (1–5 stars), cleaned text (removed HTML tags, stopwords, non-alphabetic characters), normalized (lowercase, standardized formats), and performed tokenization/lemmatization.



**EDA:** Identified sentiment patterns (1–2 stars: negative, 3: neutral, 4–5: positive) and lexical trends via word clouds (e.g., “great” in positive, “delay” in negative reviews).



**#Setup Instructions**

**##Prerequisites
**




1. Python 3.8+



2. Docker



3. Git



4. Render account (for deployment)



6. API Keys:





Serper API (SERPER_API_KEY)



Gemini API (GEMINI_API_KEY)



**# Local Setup**
## Clone the Repository:
```bash
git clone https://github.com/<your-username>/amazon-chatbot.git
cd amazon-chatbot
```bash 
## Install Dependencies:
```bash

pip install -r requirements.txt

```bash


## Set Environment Variables:
Create a .env file or export variables:
```bash
export SERPER_API_KEY=xyz
export GEMINI_API_KEY=xyz
# Place gen-lang-client-0184211067-8d635d347db2.json in the project root.


## Run the Application:
```bash

uvicorn app:app --host 0.0.0.0 --port 8000
## Access Locally:
Open http://localhost:8000 in a browser.

#Deployment on Render





##Create Render Account:





Sign up at render.com.



##Create New Web Service:





Link your GitHub repository (amazon-chatbot).



Configure:





Runtime: Docker



Dockerfile: Use the provided Dockerfile in the repository root.

Environment Variables:
```bash
SERPER_API_KEY=7142a72718
GEMINI_API_KEY=AIzaSyCpHmrgHWrbiv3mow



Secret File: Upload gen-lang-client-0184211067-8d635d347db2.json as a secret file.



#Deploy:





##Trigger a manual deploy from the Render dashboard.



##Monitor logs for “Application startup complete” and no errors.



##Access Deployed App:





Visit https://amazon-chatbot.onrender.com.

##**Project Structure**

amazon-chatbot/
├── app.py                  # FastAPI application
├── agent_checkpoint.py     # CrewAI multi-agent logic
├── requirements.txt        # Python dependencies
├── static/
│   ├── index.html         # Frontend HTML/CSS/JS
│   
├── Dockerfile             # Docker configuration
├── gen-lang-client-*.json # Google Cloud credentials
