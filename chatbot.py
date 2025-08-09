# chatbot.py
# This script uses a powerful HYBRID approach. It uses our custom-trained ML model
# as an "AI Router" to decide when to use simple responses and when to escalate to a powerful LLM.

import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import requests # Library for making HTTP requests

# --- IMPORTANT: ADD YOUR API KEY HERE ---
# Get your free API key from Google AI Studio: https://aistudio.google.com/app
API_KEY = "AIzaSyCBfprGBMO1fiwCsoOemRC-M0GX3OGtUwA"


# --- 1. Load the necessary files ---
print("--- Starting Chatbot ---")

try:
    with open('chatbot_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Custom ML model loaded successfully.")
except FileNotFoundError:
    print("ERROR: chatbot_model.pkl not found. Please run train_model.py first.")
    exit()

try:
    with open('intents_expanded.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    print("Intents data loaded successfully.")
except FileNotFoundError:
    print("ERROR: intents_expanded.json not found. Please run the convert_data.py script.")
    exit()

lemmatizer = WordNetLemmatizer()


# --- 2. Define Helper Functions ---

def preprocess_sentence(sentence):
    """
    Takes a sentence, tokenizes it, and lemmatizes the words.
    """
    word_list = nltk.word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(w.lower()) for w in word_list]
    return " ".join(lemmatized_words)

def predict_intent(processed_sentence):
    """
    Uses our trained scikit-learn model to predict the intent of a single sentence.
    """
    prediction = model.predict([processed_sentence])
    return prediction[0]

# --- API Functions for Different Moods ---
def get_joke():
    try:
        url = "https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,religious,political,racist,sexist,explicit,Programming"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data['type'] == 'twopart':
            return f"To hopefully bring a smile, here's a little joke:\n{data['setup']}\n... {data['delivery']}"
        else:
            return f"To hopefully bring a smile, here's a little joke:\n{data['joke']}"
    except requests.exceptions.RequestException: return "I hope you feel better soon."
def get_cat_fact():
    try:
        response = requests.get("https://catfact.ninja/fact")
        response.raise_for_status()
        data = response.json()
        return f"As a small distraction, here's a random fact: {data['fact']}"
    except requests.exceptions.RequestException: return "Focusing on something small can help. Take a deep breath."
def get_advice():
    try:
        response = requests.get("https://api.adviceslip.com/advice")
        response.raise_for_status()
        data = response.json()
        return f"Here's a small piece of advice to consider: {data['slip']['advice']}"
    except requests.exceptions.RequestException: return "It's okay not to have all the answers."
def get_activity_suggestion():
    try:
        response = requests.get("https://www.boredapi.com/api/activity/")
        response.raise_for_status()
        data = response.json()
        activity = data.get("activity", "do something you enjoy")
        return f"A new activity can help shift our perspective. How about you try this: **{activity}**?"
    except requests.exceptions.RequestException: return "Sometimes the best motivation is to just take one small step forward."
def get_stoic_quote():
    try:
        response = requests.get("https://stoic-quotes.com/api/quote")
        response.raise_for_status()
        data = response.json()
        quote = f'"{data["text"]}" - {data["author"]}'
        return f"Sometimes a moment of reflection can help. Here is a stoic quote to consider:\n{quote}"
    except requests.exceptions.RequestException: return "Remember to be kind to yourself. This feeling will pass."


# --- LLM Integration Function for Contextual Conversation ---
def get_contextual_llm_response(user_input, chat_history, intent_tag):
    """
    Calls the Gemini API to generate a response based on the full conversation history.
    """
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        return "I hear you. (LLM functionality is disabled. Please add your API key.)"

    print(f"Escalating to LLM for contextual response (Intent: {intent_tag})...")
    
    formatted_history = "\n".join(chat_history)
    
    # --- UPDATED PROMPT ---
    # Now asks the LLM to provide an actionable suggestion as well.
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
        generated_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "I understand. Can you tell me more about that?")
        return generated_text.strip()
    except Exception as e:
        print(f"API Error (Gemini) or parsing error: {e}")
        return "I hear you, and I want you to know it's okay to feel this way."


# --- Main Response Logic ---
def get_response(user_input, chat_history):
    """
    Acts as an AI Router. Uses our ML model to predict intent, then decides the best
    response strategy.
    """
    processed_input = preprocess_sentence(user_input)
    intent_tag = predict_intent(processed_input)
    print(f"Predicted Intent: {intent_tag}")

    core_emotions = ['sadness', 'anger', 'anxious', 'stressed', 'confusion', 'motivation']
    simple_intents = ['greeting', 'goodbye', 'gratitude', 'neutral']

    # Check if the conversation is just starting.
    is_first_exchange = len(chat_history) <= 1

    if is_first_exchange and intent_tag in core_emotions:
        # If the user starts with an emotion, give the full API + base response
        base_response = random.choice(next(i['responses'] for i in intents['intents'] if i['tag'] == intent_tag))
        api_response = ""
        if intent_tag == 'motivation': api_response = get_activity_suggestion()
        elif intent_tag == 'sadness': api_response = get_joke()
        elif intent_tag == 'anxious': api_response = get_cat_fact()
        elif intent_tag == 'confusion': api_response = get_advice()
        elif intent_tag == 'stressed' or intent_tag == 'anger': api_response = get_stoic_quote()
        return f"{base_response}\n\n{api_response}" if api_response else base_response
    
    elif intent_tag in simple_intents:
         # For simple intents at any point, use our fast, pre-written responses.
        for intent in intents['intents']:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
    else:
        # For any follow-up or complex emotion, escalate to the LLM.
        return get_contextual_llm_response(user_input, chat_history, intent_tag)
    
    # Fallback if the intent is not recognized
    return "I'm not sure I understand. Could you please rephrase?"


# --- Main Chat Loop ---
if __name__ == "__main__":
    chat_history = []
    
    print("\nChatbot is ready! I'm here to listen. Type 'quit' to exit.")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
        
        # Add the user's current message to the history *before* getting a response
        chat_history.append(f"User: {user_input}")

        # Get the response (which will use the history)
        response = get_response(user_input, chat_history)
        
        # Add the bot's response to the history
        chat_history.append(f"Bot: {response}")
        
        print(f"Bot: {response}")
