import os
import json
from crewai import Task, Agent, Crew, Process
from crewai_tools import SerperDevTool
from models import google_model
import re

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
    f"keep in mind that you need to reply in the same language as the input {warnings}",
    "Start with a friendly greeting like 'Dear [Customer's Name], '",
    "Keep it warm, personal, and sweet, like you're chatting with a best friend",
    "If the input is unclear, ask for more details with light humor",
    "Offer solutions or help in a casual, enthusiastic, and approachable way.",
    "Make it easy to read: use short sentences, simple warm words, and a friendly tone",
    "use maximum 20 words per sentence",
    "use maximum 3 paragraphs",
    "Use emojis to add a warm and friendly touch where relevant",
    "End with a positive, open note: 'Let us know if you need anything!'",
]

# Intent-specific responses
intent_responses = {
    "check_order": "Let me check your order status. Please provide your order number or confirm your recent purchase details.",
    "cancel_order": "I can help cancel your order. Please share the order number and reason for cancellation.",
    "talk_to_agent": "I'll connect you to an agent. Please hold, or contact us at {phone} or {email}.",
    "return": "Sorry about the trouble! Please provide the order number and reason for the return to start the process.",
    "replace": "Let’s get that replaced. Share the order number and details of the issue with the product.",
    "track_order": "I’ll track your order. Please provide the order number or purchase details.",
    "faq": "Check out our FAQs at https://www.amazon.com/gp/help/customer/display.html for quick answers, or ask me anything!"
}

def extract_name(input_text):
    # Extract name from inputs like "My name is Abhishek" or "Abhishek"
    match = re.search(r'(?:my name is\s+)?([a-zA-Z]+)', input_text, re.IGNORECASE)
    return match.group(1) if match else input_text.strip()

def run_agent(agent_input, session_id, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    # Extract input data
    name = agent_input.get("cust_name", "")
    if name:
        name = extract_name(name)
    purch_date = agent_input.get("purch_date", "")
    product = agent_input.get("product", "")
    review = agent_input.get("review", "")

    # Append current input to conversation history
    if review:
        conversation_history.append({"role": "user", "content": review})

    # Defining Agents
    intent_agent = Agent(
        role="Intent Detection Agent",
        goal=(
            "Identify the user’s intent from the input text, such as checking orders, canceling orders, returns, or FAQs. "
            "Classify the intent accurately to guide the response."
        ),
        backstory=(
            "Skilled in understanding user queries, you excel at detecting intents from diverse customer inputs. "
            "You ensure responses align with the user’s needs."
        ),
        llm=google_model.gemini_2_flash_lite(),
        verbose=False
    )

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
            f"Review the sentiment and emotion analysis for the input: '{review}'. "
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
            "Generate tailored responses based on detected intent, sentiment, and conversation history. "
            "Ensure empathetic, helpful replies in the same language as the input. "
            "Address concerns appropriately, offering solutions for negative feedback. "
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
            "Review and refine responses for empathy, politeness, and conciseness. "
            "Ensure responses align with Amazon standards and address user intent. "
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
    intent_task = Task(
        description=(
            f"Analyze the input text: '{review}'. "
            "Identify the user’s intent (e.g., check order, cancel order, return, replace, track order, talk to agent, faq, or feedback). "
            "Return the detected intent."
        ),
        expected_output=json.dumps({
            "intent": "e.g., check_order, cancel_order, talk_to_agent, return, replace, track_order, faq, feedback"
        }, indent=2),
        agent=intent_agent
    )

    sentiment_task = Task(
        description=(
            f"Analyze the sentiment of the text: '{review}'. "
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
            f"Customer information: name:'{name}', purchase_date:'{purch_date}', product:'{product}'. "
            f"Conversation history: {json.dumps(conversation_history, indent=2)}. "
            f"Generate a tailored response for the input: '{review}'. "
            "Follow intent-specific guidelines:\n"
            f"- Intent detected: {{intent_task.output}}. "
            f"- For feedback, use sentiment-specific guidelines:\n"
            f"  - Positive: {', '.join(positive_considerations)}\n"
            f"  - Negative: {', '.join(negative_considerations)}\n"
            f"  - Neutral: {', '.join(neutral_considerations)}\n"
            f"Expectations:\n"
            f"- Positive: {', '.join(positive_expectations)}\n"
            f"- Negative: {', '.join(negative_expectations)}\n"
            f"- Neutral: {', '.join(neutral_expectations)}"
        ),
        expected_output=(
            "A response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            "- Reflects the detected intent and sentiment.\n"
            "- Incorporates conversation history for context.\n"
            "- For non-feedback intents, use predefined responses: {json.dumps(intent_responses, indent=2)}.\n"
            "- If necessary, search Amazon for product or order details."
        ),
        agent=response_agent,
        context=[intent_task, sentiment_task, sentiment_review_task]
    )

    reviewer_task = Task(
        description=(
            f"Customer information: name:'{name}', purchase_date:'{purch_date}', product:'{product}'. "
            f"Conversation history: {json.dumps(conversation_history, indent=2)}. "
            "Represent the Amazon Customer Service Team to refine the response. "
            f"Review the response for the input: '{review}'. "
            "Ensure empathy, clarity, and alignment with Amazon standards."
        ),
        expected_output=(
            "A polished empathetic response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            "- Addresses intent, sentiment, and emotion, within 30-50 words.\n"
            "- For negative sentiment, includes solutions (e.g., new product for faulty items, delivery review for delays).\n"
            "- For positive sentiment, invites repeat shopping with light humor.\n"
            f"- Includes contact details: {customer_service_contact['name']}, "
            f"{customer_service_contact['email']}, {customer_service_contact['phone']}.\n"
            "- If needed, includes a link to product recommendations or solutions from Amazon or web searches."
            " Ends with a warm, positive thank-you note"
        ),
        agent=reviewer_agent,
        context=[intent_task, sentiment_task, sentiment_review_task, response_task],
        tools=[web_search]
    )

    # Crew Setup and Execution
    crew = Crew(
        agents=[intent_agent, sentiment_agent, sentiment_review_agent, response_agent, reviewer_agent],
        tasks=[intent_task, sentiment_task, sentiment_review_task, response_task, reviewer_task],
        verbose=True,
        process=Process.sequential
    )

    crew.kickoff()

    # Update conversation history with bot response
    conversation_history.append({"role": "bot", "content": reviewer_task.output.raw})

    # Output results
    result = {
        "name": name,
        "purchase_date": purch_date,
        "product": product,
        "review": review,
        "intent": intent_task.output.raw,
        "sentiment": sentiment_task.output.raw,
        "sentiment_review": sentiment_review_task.output.raw,
        "response": response_task.output.raw,
        "reviewed_response": reviewer_task.output.raw,
        "conversation_history": conversation_history,
        "Used_Model": (
            f"for intent detection: {intent_agent.llm.model}, "
            f"for sentiment analysis: {sentiment_agent.llm.model}, "
            f"for sentiment review: {sentiment_review_agent.llm.model}, "
            f"for response generation: {response_agent.llm.model}, "
            f"for reviewer agent: {reviewer_agent.llm.model}"
        )
    }

    return result
