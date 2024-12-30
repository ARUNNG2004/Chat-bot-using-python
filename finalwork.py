import os
import random
import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timezone
import re

# Page configuration
st.set_page_config(
    page_title="AI Laptop Guide | ARUNNG2004",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom tokenizer function
def simple_tokenize(text):
    """
    Simple tokenizer that splits text into words and handles basic cleaning
    """
    # Convert to lowercase
    text = text.lower()
    # Replace any non-alphanumeric characters with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split on whitespace and filter out empty strings
    return [token for token in text.split() if token.strip()]

# Common English stop words
STOP_WORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                  "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                  'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                  'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                  'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                  'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                  'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                  'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                  'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                  'then', 'once'])

# Default intents
DEFAULT_INTENTS = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "Is anyone there?"],
        "responses": [
            "Hello! I'm here to help you with laptop recommendations.",
            "Hi there! How can I assist you with finding the right laptop?",
            "Hey! What kind of laptop are you looking for?"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you", "Goodbye", "Thanks", "Thank you"],
        "responses": [
            "Goodbye! Feel free to return if you need more help.",
            "Thanks for using our service. Have a great day!",
            "You're welcome! Come back anytime."
        ]
    }
]

# Load JSON file with error handling
@st.cache_data
def load_intents():
    try:
        with open('laptop.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        st.warning("Using default responses as 'laptop.json' was not found.")
        return DEFAULT_INTENTS
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return DEFAULT_INTENTS

intents = load_intents()

# Extract patterns and tags
def prepare_training_data():
    patterns = []
    tags = []
    for intent in intents:
        for pattern in intent.get("patterns", []):
            patterns.append(pattern)
            tags.append(intent.get("tag", "unknown"))
    return patterns, tags

patterns, tags = prepare_training_data()

# Initialize vectorizer with custom tokenizer
@st.cache_resource
def create_vectorizer():
    return TfidfVectorizer(
        tokenizer=simple_tokenize,
        stop_words=STOP_WORDS,
        min_df=1,
        max_features=5000
    )

vectorizer = create_vectorizer()

# Vectorize patterns with error handling
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
        y = None
except Exception as e:
    st.error(f"Tag encoding error: {e}")
    y = None

# Initialize and train model
@st.cache_resource
def train_model(_X, _y):
    if _X is not None and _y is not None and len(set(_y)) > 1:
        try:
            model = LogisticRegression(max_iter=1000)
            model.fit(_X, _y)
            return model
        except Exception as e:
            st.error(f"Model training error: {e}")
    return None

model = train_model(X, y)

def get_response(user_input):
    """Generate chatbot response with comprehensive error handling"""
    if not model:
        return "I'm still learning. Please try again in a moment."
    
    try:
        # Preprocess and vectorize user input
        processed_input = vectorizer.transform([user_input])
        
        # Predict intent
        intent_index = model.predict(processed_input)[0]
        predicted_tag = tag_encoder.inverse_transform([intent_index])[0]
        
        # Find matching intent and return random response
        for intent in intents:
            if intent["tag"] == predicted_tag:
                return random.choice(intent["responses"])
        
        return "I'm not sure how to respond to that. Could you rephrase your question?"
    
    except Exception as e:
        return f"I encountered an error processing your request. Please try again with a different question."

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Main title and description
st.title("AI Laptop Advisor")
st.subheader("Intelligent Laptop Recommendations & Support")

# Sidebar menu
menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# Display current time
st.sidebar.write(f"Current Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.write(f"User: ARUNNG2004")

if choice == "Home":
    st.subheader("Chat with the Bot")
    
    # Input for user query
    user_input = st.text_input("You:", placeholder="Type your message here...")
    
    if user_input:
        user_input = user_input.strip()
        if len(user_input) < 2:
            st.warning("Please enter a longer message.")
        else:
            with st.spinner('Getting response...'):
                response = get_response(user_input)
                
                # Save to conversation history
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "user_message": user_input,
                    "bot_response": response
                })
                
                # Display current exchange
                st.write("**You:**", user_input)
                st.write("**Bot:**", response)

elif choice == "Conversation History":
    st.subheader("Chat History")
    
    if st.button("Clear History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")
    
    if st.session_state.conversation_history:
        for idx, chat in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"Conversation {len(st.session_state.conversation_history) - idx}"):
                st.write(f"**Time:** {chat['timestamp']}")
                st.write(f"**You:** {chat['user_message']}")
                st.write(f"**Bot:** {chat['bot_response']}")
    else:
        st.info("No conversation history available.")

elif choice == "About":
    st.subheader("About the Laptop Advisor Chatbot")
    st.write("""
    Welcome to the AI Laptop Advisor! This intelligent chatbot is designed to help you:
    
    - Find the perfect laptop based on your needs
    - Get recommendations within your budget
    - Understand technical specifications
    - Compare different laptop models
    - Learn about the latest laptop trends
    
    The chatbot uses Natural Language Processing (NLP) and Machine Learning to understand
    your queries and provide relevant, personalized responses.
    """)
    
    st.markdown("### How to Use")
    st.write("""
    1. Type your question about laptops in the chat input
    2. Get instant, relevant recommendations
    3. View your chat history anytime
    4. Clear history when needed
    """)
