# chatbot.py
# This script uses a powerful HYBRID approach. It uses our custom-trained ML model
# for the first user message and a powerful LLM for all follow-up conversation.

import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import requests
import os
from dotenv import load_dotenv

# Load environment variables and API key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Load ML model and data at startup
print("--- Starting Chatbot ---")
model = pickle.load(open('chatbot_model.pkl', 'rb'))
intents = json.load(open('intents_expanded.json', 'r', encoding='utf-8'))
lemmatizer = WordNetLemmatizer()
print("All models and data loaded successfully.")

def preprocess_sentence(sentence):
    """Tokenizes and lemmatizes a sentence for prediction."""
    word_list = nltk.word_tokenize(sentence)
    return " ".join([lemmatizer.lemmatize(w.lower()) for w in word_list])

def predict_intent(processed_sentence):
    """Predicts intent using the trained scikit-learn model."""
    return model.predict([processed_sentence])[0]

def get_api_response(intent_tag):
    """Fetches a dynamic response from a relevant free API based on the intent."""
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
        return "" # Return empty string if any API call fails

def get_contextual_llm_response(user_input, chat_history):
    """Handles multi-turn conversation by sending context to the Gemini LLM."""
    if not API_KEY:
        return "I hear you. (LLM functionality is disabled. Please configure your GEMINI_API_KEY in the .env file.)"

    print("Getting contextual response from LLM...")
    formatted_history = "\n".join(chat_history)
    
    prompt = f"""
    You are an AI Wellness Coach named Welly. You are having an ongoing conversation with a user.
    The conversation history is:
    ---
    {formatted_history}
    ---
    The user's latest message is: "{user_input}"

    Your task is to continue the conversation naturally. Based on the full history,
    write a short, empathetic, and supportive response that is relevant to what the user just said.
    Keep your response to 1-3 sentences.
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

def get_response(user_input, chat_history):
    """
    Main AI router. Uses the ML model for the first user message and escalates
    to the LLM for all subsequent, contextual follow-ups.
    """
    # The history contains the bot's initial greeting (1) and the user's first message (2).
    # So, if the length is 2, this is the first real turn.
    if len(chat_history) == 2:
        print("First user message. Using custom ML model for intent prediction.")
        processed_input = preprocess_sentence(user_input)
        intent_tag = predict_intent(processed_input)
        print(f"Predicted Intent: {intent_tag}")
        
        # Get the base response from our JSON file
        base_response = random.choice(next((i['responses'] for i in intents['intents'] if i['tag'] == intent_tag), ["I'm not sure I understand."]))
        
        # Get the API follow-up if it's a core emotion
        api_response = get_api_response(intent_tag)
            
        if api_response:
            return f"{base_response}\n\n{api_response}"
        else:
            return base_response
    else:
        # For all follow-up messages, use the LLM for a contextual response
        return get_contextual_llm_response(user_input, chat_history)

# This part is for testing the chatbot in the command line
if __name__ == "__main__":
    chat_history = []
    
    print("\nChatbot is ready! I'm here to listen. Type 'quit' to exit.")
    
    initial_greeting = random.choice(next(intent['responses'] for intent in intents['intents'] if intent['tag'] == 'greeting'))
    print(f"Bot: {initial_greeting}")
    chat_history.append(f"Bot: {initial_greeting}")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
        
        chat_history.append(f"User: {user_input}")
        response = get_response(user_input, chat_history)
        chat_history.append(f"Bot: {response}")
        print(f"Bot: {response}")
