# chatbot.py
# This script contains the core logic for the AI Wellness Coach.
# It uses a hybrid approach: our custom ML model for initial intent routing,
# and the Gemini LLM for handling contextual, multi-turn conversations.

import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import requests
import os
from dotenv import load_dotenv

# Load environment variables from a .env file for security
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Load all necessary files at startup
print("--- Starting Chatbot ---")
model = pickle.load(open('chatbot_model.pkl', 'rb'))
intents = json.load(open('intents_expanded.json', 'r', encoding='utf-8'))
lemmatizer = WordNetLemmatizer()
print("All models and data loaded successfully.")

def preprocess_sentence(sentence):
    # Tokenize and lemmatize the input sentence for our ML model
    word_list = nltk.word_tokenize(sentence)
    return " ".join([lemmatizer.lemmatize(w.lower()) for w in word_list])

def predict_intent(processed_sentence):
    # Use our trained scikit-learn model to predict the intent
    return model.predict([processed_sentence])[0]

# --- API Functions for Actionable Follow-ups ---
def get_api_response(intent_tag):
    # Map intents to their respective free APIs
    api_map = {
        'motivation': "https://www.boredapi.com/api/activity/",
        'sadness': "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,religious,political,racist,sexist,explicit,Programming",
        'anxious': "https://catfact.ninja/fact",
        'confusion': "https://api.adviceslip.com/advice",
        'stressed': "https://stoic-quotes.com/api/quote",
        'anger': "https://stoic-quotes.com/api/quote"
    }
    
    if intent_tag not in api_map:
        return ""

    try:
        response = requests.get(api_map[intent_tag])
        response.raise_for_status()
        data = response.json()

        # Format the response based on the API structure
        if intent_tag == 'motivation':
            return f"A new activity can help shift our perspective. How about you try this: **{data.get('activity')}**?"
        elif intent_tag == 'sadness':
            joke = data.get('setup', '') + "\n... " + data.get('delivery', data.get('joke', ''))
            return f"To hopefully bring a smile, here's a little joke:\n{joke.strip()}"
        elif intent_tag == 'anxious':
            return f"As a small distraction, here's a random fact: {data.get('fact')}"
        elif intent_tag == 'confusion':
            return f"Here's a small piece of advice to consider: {data.get('slip', {}).get('advice')}"
        elif intent_tag in ['stressed', 'anger']:
            quote = f"\"{data.get('text')}\" - {data.get('author')}"
            return f"Sometimes a moment of reflection can help. Here is a stoic quote to consider:\n{quote}"
    except Exception:
        return "" # Return empty string if any API fails

# --- LLM Function for Contextual Conversation ---
def get_contextual_llm_response(user_input, chat_history, intent_tag):
    if not API_KEY:
        return "I hear you. (LLM functionality is disabled. Please configure your GEMINI_API_KEY in the .env file.)"

    print(f"Escalating to LLM for contextual response (Intent: {intent_tag})...")
    formatted_history = "\n".join(chat_history)
    
    prompt = f"""
    You are an AI Wellness Coach named Welly. You are having an ongoing conversation with a user.
    The conversation history is:
    ---
    {formatted_history}
    ---
    The user's latest message is: "{user_input}"

    Based on the full conversation, your task is to provide an empathetic and supportive response.
    The user seems to be feeling {intent_tag}. 
    1. First, write 1-2 sentences that acknowledge and validate their feelings in a caring tone.
    2. Then, in a new paragraph, provide a gentle, actionable suggestion of something they could do to feel a little better based on their feeling.
    Keep your entire response concise and supportive.
    """
    
    try:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "I understand. Can you tell me more about that?").strip()
    except Exception as e:
        print(f"API Error (Gemini) or parsing error: {e}")
        return "I hear you, and I want you to know it's okay to feel this way."

# --- Main AI Router Logic ---
def get_response(user_input, chat_history):
    processed_input = preprocess_sentence(user_input)
    intent_tag = predict_intent(processed_input)
    print(f"Predicted Intent: {intent_tag}")

    core_emotions = ['sadness', 'anger', 'anxious', 'stressed', 'confusion', 'motivation']
    simple_intents = ['greeting', 'goodbye', 'gratitude', 'neutral']

    # Use our ML model for the first user message, or if the user just said hello.
    is_first_exchange = len(chat_history) <= 1
    
    if is_first_exchange and intent_tag in core_emotions:
        # If the user starts with an emotion, give the full API + base response
        base_response = random.choice(next(i['responses'] for i in intents['intents'] if i['tag'] == intent_tag))
        api_response = get_api_response(intent_tag)
        return f"{base_response}\n\n{api_response}" if api_response else base_response
    
    elif intent_tag in simple_intents:
        # For simple intents, use our fast, pre-written responses.
        return random.choice(next(i['responses'] for i in intents['intents'] if i['tag'] == intent_tag))
    else:
        # For any follow-up or complex emotion, escalate to the LLM.
        return get_contextual_llm_response(user_input, chat_history, intent_tag)

# This part is for testing the chatbot in the command line
if __name__ == "__main__":
    chat_history = []
    print("\nChatbot is ready! I'm here to listen. Type 'quit' to exit.")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
        
        chat_history.append(f"User: {user_input}")
        response = get_response(user_input, chat_history)
        chat_history.append(f"Bot: {response}")
        print(f"Bot: {response}")
