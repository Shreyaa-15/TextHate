import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
import re
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
stopword =set(stopwords.words("english"))

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load stopwords
stopwords = set(nltk_stopwords.words('english'))

# Load the saved decision tree model
with open('decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the saved CountVectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

# Function to preprocess input text
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove line breaks
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    text = [word for word in text.split() if word not in stopwords]  # Remove stopwords
    text = " ".join(text)
    return text

# Function to make predictions
def predict_sentiment(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    # Transform the preprocessed text into a matrix of token counts
    text_transformed = cv.transform([preprocessed_text])
    # Make prediction using the loaded model
    prediction = loaded_model.predict(text_transformed)[0]
    return prediction

# Streamlit app
def main():
    # Set title
    st.title("Sentiment Analysis with Decision Tree Model")
    
    # Add a text input for user to input their text
    text_input = st.text_area("Enter your text here:")
    
    # When the user clicks the "Predict" button
    if st.button("Predict"):
        # Check if the input text is not empty
        if text_input:
            # Make prediction
            prediction = predict_sentiment(text_input)
            # Display the prediction
            st.write(f"Predicted Sentiment: {prediction}")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
