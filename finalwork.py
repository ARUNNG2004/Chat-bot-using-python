import os
import nltk
import random
import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import ssl
from datetime import datetime, UTC

# Page configuration
st.set_page_config(
    page_title="AI Laptop Guide | ARUNNG2004",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Handle SSL certification for Streamlit cloud deployment
ssl._create_default_https_context = ssl._create_unverified_context

# Ensure the correct tokenizer and stopwords are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load JSON file with error handling
try:
    with open('laptop.json', 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    st.error("The JSON file 'laptop.json' was not found. Please ensure the file exists in the same directory.")
    intents = []
except json.JSONDecodeError as e:
    st.error(f"Error decoding JSON: {e}")
    intents = []

# Extract patterns and tags if intents exist
patterns = []
tags = []
if intents:
    for intent in intents:
        for pattern in intent.get("patterns", []):
            patterns.append(pattern)
            tags.append(intent.get("tag", "unknown"))

# Text vectorization with stopwords
stop_words = 'english'  # Using string instead of set
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words=stop_words)
try:
    if patterns:
        X = vectorizer.fit_transform(patterns)
    else:
        X = None
        st.error("No patterns available for training.")
except Exception as e:
    st.error(f"Vectorization error: {e}")
    X = None

# Encode tags
tag_encoder = LabelEncoder()
try:
    if tags:
        y = tag_encoder.fit_transform(tags)
    else:
        y = []
except Exception as e:
    st.error(f"Tag encoding error: {e}")
    y = []

# Train model with optimized settings
model = LogisticRegression(max_iter=200)
if X is not None and len(y) > 0:
    try:
        model.fit(X, y)
    except Exception as e:
        st.error(f"Model training error: {e}")

def get_response(user_input):
    """Generate chatbot response"""
    if not intents:
        return "The chatbot is not configured correctly. Please check the intents file."
    try:
        user_vector = vectorizer.transform([user_input])
        intent_index = model.predict(user_vector)[0]
        intent_tag = tag_encoder.inverse_transform([intent_index])[0]

        for intent in intents:
            if intent["tag"] == intent_tag:
                return random.choice(intent["responses"])
    except Exception as e:
        return f"I'm sorry, I encountered an error: {str(e)}"

    return "I'm sorry, I didn't understand that."

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Main title and description
st.title("AI Laptop Advisor")
st.subheader("Intelligent Laptop Recommendations & Support")

# Sidebar menu
menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Chat with the Bot")

    # Input for user query
    user_input = st.text_input("You:", placeholder="Type your message here...")
    if user_input:
        # Get the bot's response
        response = get_response(user_input)

        # Save to conversation history with consistent keys
        st.session_state.conversation_history.append({
            "user_message": user_input,
            "bot_response": response
        })

        # Display current exchange
        st.markdown(f"**Bot**: {response}")

elif choice == "Conversation History":
    st.subheader("Chat History")

    # Restart button to reset conversation history
    if st.button("Clear History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")

    # Display conversation history
    if st.session_state.conversation_history:
        for idx, chat in enumerate(st.session_state.conversation_history):
            with st.expander(f"Conversation {idx + 1}"):
                st.write(f"**You**: {chat['user_message']}")
                st.write(f"**Bot**: {chat['bot_response']}")
    else:
        st.info("No conversation history available.")

elif choice == "About":
    st.subheader("About the Chatbot")
    st.write("""
    This chatbot helps you with laptop buying recommendations, budget tips, and technical specifications.
    It uses Natural Language Processing (NLP) and Machine Learning to understand your queries and provide relevant responses.

    Features:
    - Laptop recommendations based on your needs
    - Budget-friendly options
    - Technical specifications explanation
    - Gaming laptop suggestions
    """)
