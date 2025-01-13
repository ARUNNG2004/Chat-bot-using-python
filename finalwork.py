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
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [token for token in text.split() if token.strip()]

# Extended default intents with laptop-specific patterns
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
    },
    {
        "tag": "laptop_recommendation",
        "patterns": [
            "I need a laptop",
            "What laptop should I buy",
            "Recommend me a laptop",
            "Looking for a new laptop"
        ],
        "responses": [
            "I can help you find a laptop! What's your budget range?",
            "Let me help you find the perfect laptop. What will you primarily use it for?",
            "I'll assist you in choosing a laptop. Do you have any specific requirements?"
        ]
    },
    {
        "tag": "budget",
        "patterns": [
            "My budget is",
            "I can spend",
            "Looking for laptops under",
            "Price range"
        ],
        "responses": [
            "I can suggest several good laptops in that price range. What will be your primary use case?",
            "That's a workable budget. Are you looking for something more focused on performance or portability?",
            "I can recommend several options in that range. Do you have any specific requirements like screen size or battery life?"
        ]
    }
]

# Load intents from JSON file or use defaults
@st.cache_data
def load_intents():
    try:
        # First try to load from file
        if os.path.exists('laptop.json'):
            with open('laptop.json', 'r', encoding='utf-8') as file:
                loaded_intents = json.load(file)
                print("Successfully loaded laptop.json")  # Debug info printed to console
                return loaded_intents
        else:
            print("laptop.json not found, using default intents")  # Debug info printed to console
            return DEFAULT_INTENTS
    except Exception as e:
        print(f"Error loading intents: {str(e)}")  # Debug info printed to console
        return DEFAULT_INTENTS

# Load intents and display debug info
intents = load_intents()
print("Number of intent categories:", len(intents))  # Debug info printed to console


# Extract and display patterns/tags info
def prepare_training_data():
    patterns = []
    tags = []
    for intent in intents:
        for pattern in intent.get("patterns", []):
            patterns.append(pattern)
            tags.append(intent.get("tag", "unknown"))
    return patterns, tags

patterns, tags = prepare_training_data()


# Initialize vectorizer
vectorizer = TfidfVectorizer(tokenizer=lambda text: re.findall(r'\b\w+\b', text.lower()), stop_words='english', min_df=1, max_features=5000)

# Vectorize patterns with debugging
try:
    if patterns:
        X = vectorizer.fit_transform(patterns)
        print(f"Vectorization successful: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        X = None
        print("No patterns available for training")
except Exception as e:
    print(f"Vectorization error: {str(e)}")
    X = None

# Encode tags with debugging
tag_encoder = LabelEncoder()
try:
    if tags:
        y = tag_encoder.fit_transform(tags)
        print(f"Tag encoding successful: {len(set(y))} unique tags")
    else:
        y = None
except Exception as e:
    print(f"Tag encoding error: {str(e)}")
    y = None

# Train model with debugging
@st.cache_resource
def train_model(_X, _y):
    if _X is not None and _y is not None and len(set(_y)) > 1:
        try:
            model = LogisticRegression(max_iter=1000)
            model.fit(_X, _y)
            print("Model training successful")
            return model
        except Exception as e:
            print(f"Model training error: {str(e)}")
    return None

model = train_model(X, y)

# Enhanced response function with debugging
def get_response(user_input):
    if not model:
        return "I'm still learning. Please try again in a moment."
    
    try:
        processed_input = vectorizer.transform([user_input])
        intent_index = model.predict(processed_input)[0]
        predicted_tag = tag_encoder.inverse_transform([intent_index])[0]
        
        # Debug information
        st.sidebar.write("Predicted tag:", predicted_tag)
        
        for intent in intents:
            if intent["tag"] == predicted_tag:
                response = random.choice(intent["responses"])
                st.sidebar.write("Found matching intent")
                return response
        
        return "I'm not sure how to respond to that. Could you rephrase your question?"
    
    except Exception as e:
        st.sidebar.error(f"Response error: {str(e)}")
        return "I encountered an error. Please try again with a different question."

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Main UI layout
st.title("AI Laptop Advisor")
st.subheader("Intelligent Laptop Recommendations & Support")

# Sidebar menu
menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# Display current time and user
st.sidebar.write(f"Current Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.write(f"User: ARUNNG2004")

if choice == "Home":
    st.subheader("Chat with the Bot")
    
    user_input = st.text_input("You:", placeholder="Type your message here...")
    
    if user_input:
        user_input = user_input.strip()
        if len(user_input) < 2:
            st.warning("Please enter a longer message.")
        else:
            with st.spinner('Getting response...'):
                response = get_response(user_input)
                
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "user_message": user_input,
                    "bot_response": response
                })
                
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

else:
    st.subheader("About the Laptop Advisor Chatbot")
    st.write("""
    Welcome to the AI Laptop Advisor! This intelligent chatbot is designed to help you:
    
    - Find the perfect laptop based on your needs
    - Get recommendations within your budget
    - Understand technical specifications
    - Compare different laptop models
    - Learn about the latest laptop trends
    """)
    
    st.markdown("### How to Use")
    st.write("""
    1. Type your question about laptops in the chat input
    2. Get instant, relevant recommendations
    3. View your chat history anytime
    4. Clear history when needed
    """)
