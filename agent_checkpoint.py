import os
import json
from crewai import Task, Agent, Crew, Process
from crewai_tools import SerperDevTool
from models import google_model

# Hardcode Serper API key
os.environ["SERPER_API_KEY"] = "7142a72718b003f3142427769de226076a5429ff"

# Assign serper tool to a variable
web_search = SerperDevTool()

# Customer service information
customer_service_contact = {
    "name": "Customer Service Contact",
    "email": "customerservice@amazon.com",
    "phone": "+1-800-123-4567",
    "address": "123 Amazon Way, Seattle, WA 98101",
}

# Sentiment considerations
positive_considerations = [
    "Write a sentence sincerely thanking the customer.",
    "Note that success comes from valued customers like them.",
    "Express hope that the product provides lasting benefits.",
    "Lightly encourage them to shop again with humor."
]

negative_considerations = [
    "Apologize for the issue.",
    "Thank the customer for sharing dissatisfaction and acknowledge feedback.",
    "Elaborate on the issue (e.g., product quality, delivery issues).",
    "Offer a potential solution."
]

neutral_considerations = [
    "Thank the customer for their feedback.",
    "Express appreciation for their loyalty.",
    "Compliment the product they purchased.",
    "Ask what more can be done to improve their experience."
]

# Sentiment-specific expectations
positive_expectations = [
    "A cheerful, empathetic response under 500 words with three paragraphs."
]

negative_expectations = [
    "A polite, solution-oriented response under 500 words with three paragraphs."
]

neutral_expectations = [
    "A friendly, engaging response under 500 words with three paragraphs."
]

warnings = "Do not answer questions that involve offensive language, illegal activities, sensitive information, manipulative intent, or are vague and nonsensical, and politely reject, ask for clarification, or redirect as needed."

# Common response guidelines
common_response_guidelines = [
    f"keep in mind that you need to reply in the same language as the user input {warnings}",
    "Start with 'Dear [Customer's Name]' only in the initial response",
    "Keep it warm, personal, and professional, like assisting a valued customer",
    "If the input is unclear, ask for more details with a polite tone",
    "Offer solutions or help in a clear, enthusiastic, and approachable way",
    "Make it easy to read: use short sentences, simple warm words, and a friendly tone",
    "use maximum 20 words per sentence",
    "use maximum 3 paragraphs",
    "Use emojis to add a warm and friendly touch where relevant",
    "End with a positive note: 'Let us know if you need anything!'",
]

# Session memory to track ongoing chats
session_memory = {}

def run_agent(agent_input, session_id):
    # Extract input data
    name = agent_input.get("cust_name", "")
    purch_date = agent_input.get("purch_date", "")
    product = agent_input.get("product", "")
    user_input = agent_input.get("input", "")

    # Initialize session memory if not exists
    if session_id not in session_memory:
        session_memory[session_id] = {
            "name": name,
            "purch_date": purch_date,
            "product": product,
            "history": []
        }

    # Update session memory with current input
    session_memory[session_id]["history"].append({"user": user_input})

    # Defining Agents
    sentiment_agent = Agent(
        role="Sentiment Analysis Agent",
        goal=(
            "Accurately classify text sentiment as Positive, Negative, or Neutral. "
            "Identify the dominant emotion to guide tailored responses. "
            "Deliver clear and reliable sentiment analysis."
        ),
        backstory=(
            "Trained on vast datasets of human expression, you excel at nuanced text understanding. "
            "You transform raw feedback into empathetic, actionable insights."
        ),
        llm=google_model.gemini_2_flash_lite(),
        verbose=False
    )

    sentiment_review_agent = Agent(
        role="Sentiment Review Agent",
        goal=(
            f"Review the sentiment and emotion analysis for the input: '{user_input}'. "
            "Ensure sentiment analysis is precise and contextually appropriate. "
            "Confirm sentiment as Positive, Negative, or Neutral. "
            "Capture the dominant emotion for response relevance."
        ),
        backstory=(
            "With years of scrutinizing text analysis, your deep understanding of linguistic nuances "
            "ensures reliable evaluations for meaningful customer interactions."
        ),
        llm=google_model.gemini_2_flash(),
        verbose=True,
        max_iterations=10
    )

    response_agent = Agent(
        role="Response Generation Agent",
        goal=(
            "Generate tailored responses for customer inputs based on sentiment and context. "
            "Handle customer service queries like order tracking, cancellation, returns, replacements, and FAQs. "
            "Use session history to maintain context within the same chat. "
            "Strengthen customer trust and satisfaction."
        ),
        backstory=(
            "Experienced in customer interactions, you craft meaningful responses reflecting emotions like joy or frustration. "
            "You uphold business values through compassionate replies."
        ),
        llm=google_model.gemini_2_flash_lite(),
        max_iterations=25
    )

    reviewer_agent = Agent(
        role="Response Reviewer Agent",
        goal=(
            "Generate final tailored responses for customer inputs based on sentiment and context. "
            "Review and adjust responses for empathy, politeness, and conciseness. "
            "Ensure responses address concerns with effective solutions. "
            "Deliver polished replies within 200-350 words."
        ),
        backstory=(
            "Your expertise in evaluating customer communications ensures every response meets high standards of empathy and clarity. "
            "You foster trust through thoughtful refinements."
        ),
        llm=google_model.gemini_2_flash(),
        verbose=True,
        tools=[web_search]
    )

    # Defining Tasks
    sentiment_task = Task(
        description=(
            f"Analyze the sentiment of the text: '{user_input}'. "
            "Classify as Positive, Negative, or Neutral. "
            "Identify the dominant emotion expressed."
        ),
        expected_output=json.dumps({
            "sentiment": "Positive, Negative, or Neutral",
            "emotion": "e.g., happy, sad, angry, excited"
        }, indent=2),
        agent=sentiment_agent
    )

    sentiment_review_task = Task(
        description=(
            "Review the sentiment analysis for accuracy and contextual relevance. "
            "Validate the emotion to ensure it reflects the text’s tone. "
            "Adjust classifications if discrepancies are found."
        ),
        expected_output=json.dumps({
            "sentiment": "Positive, Negative, or Neutral",
            "emotion": "e.g., happy, sad, angry, excited",
            "review": "Explanation of validation or adjustments"
        }, indent=2),
        agent=sentiment_review_agent,
        context=[sentiment_task]
    )

    response_task = Task(
        description=(
            f"Customer information: name:'{name}', product:'{product}', purchase date:'{purch_date}'. "
            f"Current input: '{user_input}'. "
            f"Session history: {json.dumps(session_memory[session_id]['history'], indent=2)}. "
            "Generate a tailored response based on the input and sentiment. "
            "Handle queries like order tracking, cancellation, returns, replacements, and FAQs. "
            "Follow sentiment-specific guidelines:\n"
            f"- Positive: {', '.join(positive_considerations)}\n"
            f"- Negative: {', '.join(negative_considerations)}\n"
            f"- Neutral: {', '.join(neutral_considerations)}\n"
            f"Expectations:\n"
            f"- Positive: {', '.join(positive_expectations)}\n"
            f"- Negative: {', '.join(negative_expectations)}\n"
            f"- Neutral: {', '.join(neutral_expectations)}"
        ),
        expected_output=(
            "A response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            "- Reflects the sentiment (Positive, Negative, or Neutral).\n"
            "- Incorporates empathy and solutions (if negative).\n"
            "- Uses session history to maintain context.\n"
            "- If necessary, search Amazon for relevant details."
        ),
        agent=response_agent,
        context=[sentiment_task, sentiment_review_task]
    )

    reviewer_task = Task(
        description=(
            f"Customer information: name:'{name}', product:'{product}', purchase date:'{purch_date}'. "
            f"Current input: '{user_input}'. "
            "Represent the Amazon Customer Service Team to refine the response. "
            "Ensure empathy, clarity, and alignment with Amazon standards."
        ),
        expected_output=(
            "A polished empathetic response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            "- Addresses sentiment and emotion, within 30-50 words.\n"
            "- For negative sentiment, includes solutions (e.g., new product for faulty items, delivery review for delays).\n"
            "- For positive sentiment, invites repeat shopping with light humor.\n"
            f"- Includes contact details: {customer_service_contact['name']}, "
            f"{customer_service_contact['email']}, {customer_service_contact['phone']}.\n"
            "- If needed, includes a link to product recommendations or solutions from Amazon or web searches."
            " Ends with a warm, positive thank-you note"
        ),
        agent=reviewer_agent,
        context=[sentiment_task, sentiment_review_task, response_task],
        tools=[web_search]
    )

    # Crew Setup and Execution
    crew = Crew(
        agents=[sentiment_agent, sentiment_review_agent, response_agent, reviewer_agent],
        tasks=[sentiment_task, sentiment_review_task, response_task, reviewer_task],
        verbose=True,
        process=Process.sequential
    )

    crew.kickoff()

    # Store bot response in session memory
    bot_response = reviewer_task.output.raw
    session_memory[session_id]["history"].append({"bot": bot_response})

    # Output results
    result = {
        "name": name,
        "purchase_date": purch_date,
        "product": product,
        "input": user_input,
        "sentiment": sentiment_task.output.raw,
        "sentiment_review": sentiment_review_task.output.raw,
        "response": response_task.output.raw,
        "reviewed_response": bot_response,
        "Used_Model": (
            f"for sentiment analysis: {sentiment_agent.llm.model}, "
            f"for sentiment review: {sentiment_review_agent.llm.model}, "
            f"for response generation: {response_agent.llm.model}, "
            f"for reviewer agent: {reviewer_agent.llm.model}"
        )
    }

    return result

def end_session(session_id):
    if session_id in session_memory:
        del session_memory[session_id]
