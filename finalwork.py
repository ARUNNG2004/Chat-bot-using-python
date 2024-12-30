import os
import nltk
import random
import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import ssl
from datetime import datetime, timezone
from nltk.tokenize import word_tokenize
import warnings

# Suppress NLTK download messages
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Laptop Guide | ARUNNG2004",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom tokenizer that falls back to basic splitting if NLTK fails
def safe_tokenize(text):
    try:
        return word_tokenize(text)
    except:
        return text.lower().split()

# Initialize NLTK data
@st.cache_resource
def setup_nltk():
    try:
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download required NLTK data
        for resource in ['punkt', 'stopwords']:
            try:
                nltk.download(resource, quiet=True, download_dir=nltk_data_dir)
            except Exception as e:
                st.warning(f"Failed to download {resource}: {str(e)}")
        
        # Verify downloads
        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            word_tokenize("Test sentence")
            stopwords.words('english')
            return True
        except Exception as e:
            st.error(f"NLTK verification failed: {str(e)}")
            return False
            
    except Exception as e:
        st.error(f"NLTK setup failed: {str(e)}")
        return False

# Initialize NLTK
nltk_ready = setup_nltk()

# Default intents remain the same as in your original code
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

# Initialize vectorizer with fallback tokenizer
@st.cache_resource
def create_vectorizer():
    return TfidfVectorizer(
        tokenizer=safe_tokenize,
        stop_words='english',
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

# Rest of the Streamlit UI code remains the same as in your original version
