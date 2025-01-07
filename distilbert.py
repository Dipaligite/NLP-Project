import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from PIL import Image
import pytesseract
import streamlit as st
import PyPDF2


# Load pre-trained sentiment analysis model and tokenizer
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Save the model and tokenizer for later use
def save_model(model, tokenizer, model_dir="saved_model"):
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    st.success(f"Model and tokenizer saved in '{model_dir}'")

# Uncomment the line below to save the model and tokenizer
save_model(model, tokenizer)


# Function to preprocess text
def preprocess_text(text):
    text = text.strip().lower()
    return text


# Function to predict sentiment
def predict_sentiment(text):
    text = preprocess_text(text)
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    # Map model's predicted class to sentiments
    sentiment_labels = {0: 'Negative', 1: 'Positive'}  # Update mapping based on the pre-trained model
    sentiment = sentiment_labels.get(predicted_class, "Unknown")
    score = probabilities[0][predicted_class].item()
    return sentiment, score


# Streamlit app setup
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="centered")
st.title('Sentiment Analysis with DistilBERT')


# Instructions Section
st.markdown("""
    This app uses a fine-tuned BERT-based model to predict the sentiment of text.
    The possible sentiments are **Positive** or **Negative**.

    You can upload files (CSV, Excel, TXT, JPG, or PDF) or enter text directly.
""")

# File Upload Section
uploaded_file = st.file_uploader("Upload a file (CSV, Excel, TXT, JPG, PDF):", type=['csv', 'xlsx', 'txt', 'jpg', 'pdf'])

# Text Area for Manual Input
user_input = st.text_area("Or enter text for sentiment analysis:")


# Process the uploaded file and extract text
lines = []
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
            lines = df.iloc[:, 0].dropna().astype(str).tolist()  # Assuming first column has the text
        elif file_type == 'xlsx':
            df = pd.read_excel(uploaded_file)
            lines = df.iloc[:, 0].dropna().astype(str).tolist()  # Assuming first column has the text
        elif file_type == 'txt':
            lines = [line.strip() for line in uploaded_file.read().decode('utf-8').splitlines() if line.strip()]
        elif file_type == 'jpg':
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            st.error("Unsupported file type.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Add user input as a line for prediction
if user_input:
    lines.append(user_input)


# Button to trigger prediction
if st.button('Predict'):
    if lines:
        results = []
        for line in lines:
            line = str(line)
            if line.strip():
                sentiment, score = predict_sentiment(line)
                results.append({
                    'Input Text': line,
                    'Predicted Sentiment': sentiment,
                    'Score': f"{score * 100:.2f}%"
                })

        # Display results in a table format
        result_df = pd.DataFrame(results)
        st.write(result_df)

        # Allow users to download the results in CSV format
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Sentiment Analysis Results",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )
    else:
        st.error("No text available for analysis. Please upload a file or enter text.")
