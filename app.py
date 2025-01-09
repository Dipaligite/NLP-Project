import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import numpy as np

# Importing necessary libraries for machine learning and text processing
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import nltk

# Function to load dataset from CSV files
def load_data(uploaded_file):
    # Check if the file is a CSV and read into pandas
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        raise ValueError("Unsupported file format! Please use CSV files.")

# Load the dataset (Change the path to your file's location)
st.title("Climate Sentiment Analysis")
st.sidebar.title("File Upload")

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a file (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the dataset
        train_df = load_data(uploaded_file)
        st.write("Data Preview:")
        st.write(train_df.head())

        # Check if necessary columns are present
        if 'sentiment' not in train_df.columns or 'message' not in train_df.columns:
            st.error("The dataset must contain 'sentiment' and 'message' columns.")
        else:
            # Preprocess the data: handle missing messages
            train_df['message'] = train_df['message'].fillna('')
            st.subheader("Distribution of Sentiment Classes")
            sentiment_count = train_df.groupby("sentiment").count()["message"].reset_index().sort_values(by="message", ascending=False)
            st.bar_chart(sentiment_count.set_index('sentiment'))

            # Generate and display a word cloud for text analysis
            st.subheader("Word Cloud")
            full_text = " ".join(train_df["message"].dropna())  # Combine all messages into one string
            wc = WordCloud(background_color='white').generate(full_text)
            st.image(wc.to_array(), use_column_width=True)

            # Model Training
            st.subheader("Train Sentiment Classifier")

            # Preprocess the text for training
            train_df['cleaned_message'] = train_df['message'].apply(lambda x: x.lower())
            X = train_df['cleaned_message']
            y = train_df['sentiment']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            # Model selection
            model_option = st.selectbox("Choose a model for sentiment classification:", 
                                        ['Logistic Regression', 'Random Forest', 'SVM'])

            if model_option == 'Logistic Regression':
                model = LogisticRegression(max_iter=1000)  # Increased max_iter to avoid convergence issues
            elif model_option == 'Random Forest':
                model = RandomForestClassifier()
            else:
                model = LinearSVC()

            # Train model
            model.fit(X_train_tfidf, y_train)

            # Model evaluation
            y_pred = model.predict(X_test_tfidf)

            st.subheader("Model Evaluation")
            st.write(f"Accuracy: {model.score(X_test_tfidf, y_test):.4f}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")

else:
    st.write("Please upload a file to begin.")
