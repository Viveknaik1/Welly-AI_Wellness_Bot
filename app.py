# app.py
# This script creates the final, polished web interface for our AI Wellness Coach
# using the Streamlit library.

import streamlit as st
import random
# Import the final, sophisticated get_response function from our chatbot logic file
from chatbot import get_response

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Welly",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Custom CSS to hide the header bar ---
# This injects a small piece of CSS code to hide the default colored bar at the top.
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# --- App Title and Description ---
st.title("Welly 🤖")
st.write("Your AI Wellness Coach")
st.write("---")

# --- Initialize Chat History in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add the initial greeting from the bot.
    initial_greeting = "Hello! I'm Welly, your AI Wellness Coach. I'm here to listen. What's on your mind today?"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# --- Display Previous Messages ---
for message in st.session_state.messages:
    # Use the custom avatars you requested
    avatar = "🤖" if message["role"] == "assistant" else "🔵"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("How are you feeling?"):
    # Add user's message to the chat history and display it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🔵"):
        st.markdown(prompt)

    # --- Use the Full, Sophisticated Chatbot Logic ---
    chat_history_for_bot = []
    for msg in st.session_state.messages:
        chat_history_for_bot.append(f"{msg['role']}: {msg['content']}")
        
    with st.spinner("Welly is thinking..."):
        # Call the get_response function from our chatbot.py script
        bot_response = get_response(prompt, chat_history_for_bot)

    # Add the bot's response to the chat history and display it.
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_response)
