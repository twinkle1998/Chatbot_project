import os
import json
import datetime as dt
from datetime import datetime
from dateutil.parser import parse
from crewai import Task, Agent, Crew, Process
from crewai_tools import SerperDevTool
from models import google_model

# Hardcode Serper API key for deployment
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
    "Ask for more details to understand the issue better (e.g., product quality, functionality).",
    "Offer a tailored solution (e.g., troubleshooting, replacement, or contact customer service) based on the issue."
]

neutral_considerations = [
    "Thank the customer for their feedback.",
    "Express appreciation for their loyalty.",
    "Compliment the product they purchased.",
    "Ask what more can be done to improve their experience."
]

# Intent-specific considerations
return_considerations = [
    "Provide clear instructions for initiating a return.",
    "Check if the purchase date is within the 30-day return window.",
    "If outside the window, suggest contacting customer service for defective items."
]

cancel_considerations = [
    "Explain that cancellation is possible only for unshipped orders.",
    "Suggest checking the 'Orders' page or contacting customer service."
]

status_considerations = [
    "Guide the customer to check their order status on the 'Orders' page.",
    "Offer to assist further if the order is delayed."
]

# Sentiment-specific expectations
positive_expectations = [
    "A cheerful, empathetic response under 100 words"
]

negative_expectations = [
    "A polite, solution-oriented response under 150 words"
]

neutral_expectations = [
    "A friendly, engaging response under 100 words"
]

intent_expectations = [
    "A concise, action-oriented response under 100 words, addressing the specific intent."
]

# Common response guidelines
common_response_guidelines = [
    "Never say i cannot help you or i am not able to help you",
    "You do not need to address the customer by name",
    "Do not give any greetings or salutations in the response, just start with the response",
    "Keep in mind that you need to reply in the same language as the review",
    "Do not answer questions that involve offensive language, illegal activities, sensitive information, manipulative intent, or are vague and nonsensical, and politely reject, ask for clarification, or redirect as needed.",
    "Keep it warm, personal, and sweet, like you're chatting with a best friend",
    "If the review is unclear, ask for more details with light humor",
    "Offer solutions or help in a casual, enthusiastic, and approachable way.",
    "Make it easy to read: use short sentences, simple warm words, and a friendly tone",
    "Is is mandatory to make your response easy to read: use short sentences, clear separation between paragraphs, and a friendly tone",
    "Use maximum 25 words per sentence",
    "Use maximum 3 paragraphs",
    "Use emojis to add a warm and friendly touch where relevant",
    "End with a positive, open note: 'Let us know if you need anything!'",
]

# Return policy
return_policy = {
    "policy": [
        "customer can return the product within 30 days of purchase for a full refund.",
        "customer cannot return the product after 30 days of purchase."
    ],
    "conditions": [
        "The product must be in its original condition and packaging.",
        "Customer must provide proof of purchase, such as a receipt or order confirmation.",
        "Certain items, such as electronics or personalized products, may have different return policies.",
        "If the product is defective or damaged, you may be eligible for a replacement or repair.",
        "For more information, please visit our return policy page on the Amazon website."
    ]
}

# Order inquiry guidelines
order_inquiry = [
    "If the customer asks about their order status, guide them to check the 'Orders' page or suggest contacting customer service.",
    "If the customer asks to cancel their order, explain cancellation is possible for unshipped orders or suggest contacting customer service.",
    "If the customer asks to talk to an agent, provide customer service contact information."
]

# Get current date
current_date = dt.datetime.now().strftime("%Y-%m-%d")

def validate_purchase_date(purch_date):
    try:
        purchase = parse(purch_date)
        today = datetime.now()
        days_diff = (today - purchase).days
        return days_diff <= 30, days_diff
    except ValueError:
        return False, None

def run_agent(agent_input):
    # Extract input
    name = agent_input.get("cust_name", "")
    purch_date = agent_input.get("purch_date", "")
    product = agent_input.get("product", "")
    review = agent_input.get("review", "")
    intent = agent_input.get("intent", "review")
    last_intent = agent_input.get("last_intent", None)

    # Validate purchase date
    is_within_return_window, days_diff = validate_purchase_date(purch_date)

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
            f"Review the sentiment and emotion analysis from the review: '{review}'. "
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
        max_iterations=2
    )

    response_agent = Agent(
        role="Response Generation Agent",
        goal=(
            "Generate tailored responses for customer reviews or intents based on sentiment and the same language as the input. "
            "Generate empathetic, helpful responses based on sentiment, intent, and context. "
            "Address concerns appropriately, offering solutions for negative feedback or specific intents. "
            "Strengthen customer trust and satisfaction."
        ),
        backstory=(
            "Experienced in customer interactions, you craft meaningful responses reflecting emotions like joy or frustration. "
            "You uphold business values through compassionate replies."
        ),
        llm=google_model.gemini_2_flash_lite(),
        max_iterations=2
    )

    reviewer_agent = Agent(
        role="Response Reviewer Agent",
        goal=(
            "Generate final tailored responses for customer reviews or intents based on sentiment and the same language as the input. "
            "Review and adjust responses for empathy, politeness, and conciseness. "
            "Ensure responses address concerns with effective solutions, considering intent and context. "
            "Deliver polished replies within 30-50 words for intents or 30-150 words for reviews."
        ),
        backstory=(
            "Your expertise in evaluating customer communications ensures every response meets high standards of empathy and clarity. "
            "You foster trust through thoughtful refinements."
        ),
        llm=google_model.gemini_2_flash(),
        verbose=True,
        tools=[web_search],
        max_iterations=2
    )

    # Defining Tasks
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
            f"Customer information provided: name:'{name}', product:'{product}', purchasedate:'{purch_date}', current date:'{current_date}'. "
            f"Generate a tailored response for the input: '{review}' with intent: '{intent}'. "
            f"Previous intent: '{last_intent or 'none'}'. "
            f"Follow guidelines based on sentiment or intent:\n"
            f"- Positive: {', '.join(positive_considerations)}\n"
            f"- Negative: {', '.join(negative_considerations)}\n"
            f"- Neutral: {', '.join(neutral_considerations)}\n"
            f"- Return: {', '.join(return_considerations)}\n"
            f"- Cancel: {', '.join(cancel_considerations)}\n"
            f"- Status: {', '.join(status_considerations)}\n"
            f"Expectations:\n"
            f"- Positive: {', '.join(positive_expectations)}\n"
            f"- Negative: {', '.join(negative_expectations)}\n"
            f"- Neutral: {', '.join(neutral_expectations)}\n"
            f"- Intent-specific: {', '.join(intent_expectations)}\n"
            f"For negative feedback, only mention return policy if intent is 'return' or review explicitly mentions returning. "
            f"Return window is 30 days; purchase is {'within' if is_within_return_window else 'outside'} the window."
        ),
        expected_output=(
            "A response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            f"- Follow return policy: {return_policy['policy']} and conditions: {return_policy['conditions']} only for 'return' intent.\n"
            f"- Follow order inquiry guidelines: {order_inquiry}\n"
            "- Reflects the sentiment (Positive, Negative, or Neutral) or intent (Return, Cancel, Status).\n"
            "- Incorporates empathy and solutions (if negative or intent-specific).\n"
            "- If necessary, search Amazon for product details or alternative solutions."
        ),
        agent=response_agent,
        context=[sentiment_task, sentiment_review_task]
    )

    reviewer_task = Task(
        description=(
            f"Customer information provided: name:'{name}', product:'{product}', purchasedate:'{purch_date}', current date:'{current_date}'. "
            f"Represent the Amazon Customer Service Team to refine the response for the input: '{review}' with intent: '{intent}'. "
            f"Previous intent: '{last_intent or 'none'}'. "
            f"Ensure empathy, clarity, and alignment with Amazon standards. "
            f"For negative feedback, only mention return policy if intent is 'return' or review mentions returning; otherwise, focus on other solutions. "
            f"Return window is 30 days; purchase is {'within' if is_within_return_window else 'outside'} the window."
        ),
        expected_output=(
            "A polished empathetic response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            f"- Follow return policy: {return_policy['policy']} and conditions: {return_policy['conditions']} only for 'return' intent.\n"
            f"- Follow order inquiry guidelines: {order_inquiry}\n"
            "- Addresses sentiment and intent, within 30-50 words for intents or 30-150 words for reviews.\n"
            "- For negative sentiment, includes solutions (e.g., troubleshooting, contact details) unless 'return' intent.\n"
            "- For intents, provides specific actions (e.g., return steps, cancellation info, status check).\n"
            f"- Includes contact details if escalated: {customer_service_contact['name']}, "
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

    # Output results
    result = {
        "name": name,
        "purchase_date": purch_date,
        "product": product,
        "review": review,
        "intent": intent,
        "sentiment": sentiment_task.output.raw,
        "sentiment_review": sentiment_review_task.output.raw,
        "response": response_task.output.raw,
        "reviewed_response": reviewer_task.output.raw,
        "Used_Model": (
            f"for sentiment analysis: {sentiment_agent.llm.model}, "
            f"for sentiment review: {sentiment_review_agent.llm.model}, "
            f"for response generation: {response_agent.llm.model}, "
            f"for reviewer agent: {reviewer_agent.llm.model}"
        )
    }

    return result

def process_reply(agent_input: dict, customer_reply: str, response_result: str):
    name = agent_input.get("cust_name", "")
    purch_date = agent_input.get("purch_date", "")
    product = agent_input.get("product", "")
    review = agent_input.get("review", "")
    intent = agent_input.get("intent", "review")
    last_intent = agent_input.get("last_intent", None)

    general_response_agent = Agent(
        role="General Response Agent",
        goal=(
            "Provide solutions or suggestions to the customer based on their reply and previous context. "
            "Ensure the response is concise, clear, solution-oriented, and friendly. "
            "Respond to inquiries or concerns with a warm, helpful tone, avoiding repetition of prior responses."
        ),
        backstory=(
            "Experienced in customer interactions, you craft meaningful responses reflecting emotions like joy or frustration. "
            "You uphold business values through compassionate replies."
        ),
        llm=google_model.gemini_2_flash_lite(),
        max_iterations=2
    )

    general_response_task = Task(
        description=(
            f"Respond to the customer reply: '{customer_reply}' after the previous response: '{response_result}'. "
            f"Customer information: name:'{name}', product:'{product}', purchasedate:'{purch_date}'. "
            f"Current intent: '{intent}', previous intent: '{last_intent or 'none'}'. "
            f"Follow order inquiry guidelines: {order_inquiry}. "
            "Do not repeat the previous response. "
            "Do not end the conversation unless the customer agrees to the solution."
        ),
        expected_output=(
            "A response string with the following characteristics:\n"
            f"- {', '.join(common_response_guidelines)}\n"
            "- Craft the reply to be short, clear, solution-oriented, and friendly.\n"
            "- If needed, include a link to product recommendations or solutions from Amazon or web searches.\n"
            "- Respond to customer inquiries or concerns with a warm, helpful tone.\n"
            "- Ensure the response is unique and avoids repeating prior responses.\n"
            "- Only end the conversation if the customer agrees to the solution.\n"
            "Ends with a warm, positive thank-you note"
        ),
        agent=general_response_agent,
        tools=[web_search]
    )

    crew = Crew(
        agents=[general_response_agent],
        tasks=[general_response_task],
        verbose=True,
        process=Process.sequential
    )

    crew.kickoff()

    result = {
        "general_response": general_response_task.output.raw,
        "Used_Model": f"for general response: {general_response_agent.llm.model}"
    }

    return result

def end_session(session_id: str):
    return {"status": "success", "message": f"Session {session_id} terminated."}
