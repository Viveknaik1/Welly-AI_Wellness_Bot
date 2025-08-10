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
    layout="centered"
)

# Custom CSS to hide the default Streamlit header/footer for a cleaner look
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# --- App Title ---
st.title("Welly 🤖")
st.write("Your AI Wellness Coach")
st.write("---")

# --- Initialize Chat History ---
# We use Streamlit's session_state to remember the conversation.
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Set the initial greeting message from the bot
    initial_greeting = "Hello! I'm Welly, your AI Wellness Coach. I'm here to listen. What's on your mind today?"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# --- Display Chat History ---
for message in st.session_state.messages:
    # Use the custom avatars you requested
    avatar = "🤖" if message["role"] == "assistant" else "🔵"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("How are you feeling?"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🔵"):
        st.markdown(prompt)

    # Generate and display the bot's response
    with st.spinner("Welly is thinking..."):
        # Format history for the chatbot logic function
        history_for_bot = [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
        bot_response = get_response(prompt, history_for_bot)
    
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_response)
