import re
import pandas as pd
import emoji
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from fpdf import FPDF  # Add this import for PDF generation
from io import StringIO  # Import StringIO for file handling
import os  # Import os module

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Streamlit App
def main():
    st.title("WhatsApp Chat Analysis")
    
    # File Upload
    st.sidebar.title("Upload Chat File")
    file = st.sidebar.file_uploader("Upload your file", type=['txt'])
    date_format = st.sidebar.radio("Select Date Format", ["12hr", "24hr"])

    if file:
        # Load Data
        df = import_data(file, date_format)
        st.write("Dataframe Preview", df.head())

        # Preprocess Data
        df = preprocess_data(df)
        st.write("Preprocessed Data", df.head())

        # Analysis
        st.sidebar.title("Analysis")
        analysis_options = st.sidebar.multiselect("Select Analysis", [
            "Top 10 Most Active Days",
            "Top 10 Active Users",
            "Top 10 Media Users",
            "Top 10 Emojis"
        ])
        
        top10_days, top10_users, top10_media, top10_emojis = None, None, None, None
        
        if "Top 10 Most Active Days" in analysis_options:
            top10_days = analyze_top10_days(df)
            st.write("Top 10 Most Active Days", top10_days)
            plot_time_series(df, "Messages Sent Over Time")

        if "Top 10 Active Users" in analysis_options:
            top10_users = analyze_top10_users(df)
            st.write("Top 10 Active Users", top10_users)
            plot_activity(top10_users, 'user', "Top 10 Active Users", "Users", "Message Count")

        if "Top 10 Media Users" in analysis_options:
            top10_media = analyze_top10_media_users(df)
            st.write("Top 10 Media Users", top10_media)

        if "Top 10 Emojis" in analysis_options:
            top10_emojis = analyze_top10_emojis(df)
            st.write("Top 10 Emojis", top10_emojis)

        # Download Output (PDF)
        st.sidebar.title("Download Output")
        if st.sidebar.button("Download Analysis Data as PDF"):
            # Create PDF and provide download option
            pdf = create_pdf(df, top10_days, top10_users, top10_media, top10_emojis)
            st.download_button(
                label="Download PDF",
                data=pdf,
                file_name="whatsapp_analysis.pdf",
                mime="application/pdf"
            )

# Helper Functions

def import_data(file, date_format):
    # Define the function that will convert the raw chat data to a pandas dataframe
    split_formats = {
        '12hr': '\\d{1,2}/\\d{1,2}/\\d{2,4},\\s\\d{1,2}:\\d{2}\\s[APap][mM]\\s-\\s',
        '24hr': '\\d{1,2}/\\d{1,2}/\\d{2,4},\\s\\d{1,2}:\\d{2}\\s-\\s',
    }
    datetime_formats = {
        '12hr': '%d/%m/%Y, %I:%M %p - ',
        '24hr': '%d/%m/%Y, %H:%M - ',
    }

    # Read the uploaded file as a string
    raw_string = StringIO(file.getvalue().decode("utf-8")).read()
    
    user_msg = re.split(split_formats[date_format], raw_string)[1:]
    date_time = re.findall(split_formats[date_format], raw_string)

    df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})
    df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[date_format])

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

def preprocess_data(df):
    # Add some features like day, month, hour, etc.
    df['day'] = df['date_time'].dt.strftime('%a')
    df['month'] = df['date_time'].dt.strftime('%b')
    df['hour'] = df['date_time'].dt.hour
    df['date'] = df['date_time'].dt.date
    return df

def analyze_top10_days(df):
    # Analyzing top 10 days with most messages
    daily_msgs = df.groupby('date').size().reset_index(name='message_count')
    return daily_msgs.sort_values(by='message_count', ascending=False).head(10)

def analyze_top10_users(df):
    # Analyzing top 10 users with most messages
    user_msgs = df['user'].value_counts().reset_index(name='message_count')
    user_msgs.rename(columns={'index': 'user'}, inplace=True)
    return user_msgs.head(10)

def analyze_top10_media_users(df):
    # Analyzing top 10 users who sent media (like photos, videos)
    media_msgs = df[df['message'] == '<Media omitted>']
    media_users = media_msgs['user'].value_counts().reset_index(name='media_count')
    media_users.rename(columns={'index': 'user'}, inplace=True)
    return media_users.head(10)

def analyze_top10_emojis(df):
    # Analyzing top 10 emojis used in the chat
    emoji_ctr = Counter()
    for msg in df['message']:
        emojis = [c for c in msg if c in emoji.EMOJI_DATA]
        emoji_ctr.update(emojis)

    emoji_data = pd.DataFrame(emoji_ctr.most_common(10), columns=['emoji', 'count'])
    emoji_data['description'] = emoji_data['emoji'].apply(lambda x: emoji.demojize(x))
    return emoji_data

def plot_time_series(df, title):
    daily_msgs = df.groupby('date').size().reset_index(name='message_count')
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='date', y='message_count', data=daily_msgs)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Messages Sent')
    st.pyplot(plt)  # Display plot in Streamlit

def plot_activity(data, column, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=data[column], y=data['message_count'], palette='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)  # Display plot in Streamlit

def create_pdf(df, top10_days, top10_users, top10_media, top10_emojis):
    # Check if any dataframe is None or empty, and handle it
    if top10_days is None or top10_days.empty:
        top10_days_str = "No data available for Top 10 Most Active Days"
    else:
        top10_days_str = top10_days.to_string(index=False)
        
    if top10_users is None or top10_users.empty:
        top10_users_str = "No data available for Top 10 Active Users"
    else:
        top10_users_str = top10_users.to_string(index=False)
        
    if top10_media is None or top10_media.empty:
        top10_media_str = "No data available for Top 10 Media Users"
    else:
        top10_media_str = top10_media.to_string(index=False)
        
    if top10_emojis is None or top10_emojis.empty:
        top10_emojis_str = "No data available for Top 10 Emojis"
    else:
        top10_emojis_str = top10_emojis.to_string(index=False)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="WhatsApp Chat Analysis", ln=True, align='C')

    # Add a section for data preview
    pdf.ln(10)  # Line break
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Data Preview", ln=True)
    pdf.ln(5)
    
    # Convert dataframe to text and add to PDF
    data_str = df.head().to_string(index=False)
    pdf.multi_cell(0, 10, txt=data_str.encode('latin-1', 'replace').decode('latin-1'))  # Handle non-latin chars

    # Add a section for top 10 active days
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Top 10 Most Active Days", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=top10_days_str.encode('latin-1', 'replace').decode('latin-1'))

    # Add a section for top 10 active users
    pdf.ln(10)
    pdf.cell(200, 10, txt="Top 10 Active Users", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=top10_users_str.encode('latin-1', 'replace').decode('latin-1'))

    # Add a section for top 10 media users
    pdf.ln(10)
    pdf.cell(200, 10, txt="Top 10 Media Users", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=top10_media_str.encode('latin-1', 'replace').decode('latin-1'))

    # Add a section for top 10 emojis
    pdf.ln(10)
    pdf.cell(200, 10, txt="Top 10 Emojis", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=top10_emojis_str.encode('latin-1', 'replace').decode('latin-1'))

    # Add images of the plots
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Analysis Graphs", ln=True)

    # Include the saved images in the PDF
    if os.path.exists("time_series_plot.png"):
        pdf.ln(5)
        pdf.image("time_series_plot.png", x=10, w=180)
    
    if os.path.exists("activity_plot.png"):
        pdf.ln(5)
        pdf.image("activity_plot.png", x=10, w=180)


    # Output the PDF as a byte stream
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output


if __name__ == "__main__":
    main()
