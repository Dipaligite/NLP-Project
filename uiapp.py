import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import emoji
from collections import Counter
import warnings
import streamlit as st
from io import StringIO
import base64
from fpdf import FPDF
import tempfile
import os


warnings.filterwarnings('ignore')


# Function to import data from a .txt file
def import_data(file_path, key):
    return rawToDf(file_path, key)


# Function to convert raw .txt file to DataFrame
def rawToDf(file, key):
    split_formats = {
        '12hr': '\\d{1,2}/\\d{1,2}/\\d{2,4},\\s\\d{1,2}:\\d{2}\\s[APap][mM]\\s-\\s',
        '24hr': '\\d{1,2}/\\d{1,2}/\\d{2,4},\\s\\d{1,2}:\\d{2}\\s-\\s',
    }
    datetime_formats = {
        '12hr': '%d/%m/%Y, %I:%M %p - ',
        '24hr': '%d/%m/%Y, %H:%M - ',
    }

    with open(file, 'r', encoding='utf-8') as raw_data:
        raw_string = ' '.join(raw_data.read().split('\n'))
        user_msg = re.split(split_formats[key], raw_string)[1:]
        date_time = re.findall(split_formats[key], raw_string)

        df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})
        df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[key])

    usernames, msgs = [], []
    for i in df['user_msg']:
        a = re.split('([\\w\\W]+?):\\s', i)
        if a[1:]:
            usernames.append(a[1])
            msgs.append(a[2])
        else:
            usernames.append("group_notification")
            msgs.append(a[0])

    df['user'] = usernames
    df['message'] = msgs
    df.drop('user_msg', axis=1, inplace=True)
    return df

# Processing and Adding Features
def preprocess_data(df):
    df['day'] = df['date_time'].dt.strftime('%a')
    df['month'] = df['date_time'].dt.strftime('%b')
    df['hour'] = df['date_time'].dt.hour
    df['date'] = df['date_time'].dt.date
    return df


# Analysis Functions
def analyze_top10_days(df):
    daily_msgs = df.groupby('date').size().reset_index(name='message_count')
    return daily_msgs.sort_values(by='message_count', ascending=False).head(10)

def analyze_top10_users(df):
    user_msgs = df['user'].value_counts().reset_index(name='message_count')
    user_msgs.rename(columns={'index': 'user'}, inplace=True)
    return user_msgs.head(10)

def analyze_top10_media_users(df):
    media_msgs = df[df['message'] == '<Media omitted>']
    media_users = media_msgs['user'].value_counts().reset_index(name='media_count')
    media_users.rename(columns={'index': 'user'}, inplace=True)
    return media_users.head(10)

def analyze_top10_emojis(df):
    emoji_ctr = Counter()
    for msg in df['message']:
        emojis = [c for c in msg if c in emoji.EMOJI_DATA]
        emoji_ctr.update(emojis)

    emoji_data = pd.DataFrame(emoji_ctr.most_common(10), columns=['emoji', 'count'])
    emoji_data['description'] = emoji_data['emoji'].apply(lambda x: emoji.demojize(x))
    return emoji_data


# Visualization
def plot_activity(df, column, title, xlabel, ylabel):
    activity_count = df[column].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=activity_count.index, y=activity_count.values, palette='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt

def plot_time_series(df, title):
    daily_msgs = df.groupby('date').size().reset_index(name='message_count')
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='date', y='message_count', data=daily_msgs)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Messages Sent')
    return plt


# Function to generate a PDF report
def generate_pdf(df, top10_days, top10_users, top10_media_users, top10_emojis):
    pdf = FPDF()
    pdf.add_page()

    # Set font
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="WhatsApp Chat Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Add DataFrames to PDF
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt="Top 10 Most Active Days", ln=True)
    pdf.multi_cell(0, 10, txt=top10_days.to_string(index=False))
    pdf.ln(10)

    pdf.cell(200, 10, txt="Top 10 Active Users", ln=True)
    pdf.multi_cell(0, 10, txt=top10_users.to_string(index=False))
    pdf.ln(10)

    pdf.cell(200, 10, txt="Top 10 Media Users", ln=True)
    pdf.multi_cell(0, 10, txt=top10_media_users.to_string(index=False))
    pdf.ln(10)

    pdf.cell(200, 10, txt="Top 10 Emojis", ln=True)
    pdf.multi_cell(0, 10, txt=top10_emojis.to_string(index=False))
    
    # Save PDF to temporary file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_pdf.name)
    
    return temp_pdf.name

# Streamlit UI
def main():
    st.title("WhatsApp Chat Analyzer")

    file = st.file_uploader("Upload your .txt file", type=["txt"])
    key = st.selectbox("Select Date-Time Format", options=['12hr', '24hr'])

    if file is not None:
        # Handle file input from Streamlit's file uploader for .txt file only
        df = import_data(file, key)
        df = preprocess_data(df)

        st.write("### Sample Data")
        st.dataframe(df.head())

        st.write("### Top 10 Most Active Days")
        top10_days = analyze_top10_days(df)
        st.dataframe(top10_days)

        st.write("### Top 10 Active Users")
        top10_users = analyze_top10_users(df)
        st.dataframe(top10_users)

        st.write("### Top 10 Media Users")
        top10_media_users = analyze_top10_media_users(df)
        st.dataframe(top10_media_users)

        st.write("### Top 10 Emojis")
        top10_emojis = analyze_top10_emojis(df)
        st.dataframe(top10_emojis)

        st.write("### Visualization")
        fig = plot_time_series(df, "Messages Sent Over Time")
        st.pyplot(fig)

        # Add download button for PDF
        if st.button("Download PDF Report"):
            pdf_path = generate_pdf(df, top10_days, top10_users, top10_media_users, top10_emojis)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_file,
                    file_name="whatsapp_chat_analysis_report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
