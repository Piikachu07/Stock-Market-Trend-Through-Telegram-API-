import asyncio
import re
import torch
import torch.nn.functional as F
import pandas as pd
import streamlit as st
import spacy
import altair as alt
from transformers import BertTokenizer, BertForSequenceClassification
from telethon import TelegramClient

# ========== CONFIG ==========
api_id = 'Write your telegram api id here'
api_hash = 'Write the hash id here'
channel_username = 't.me/DalalStreetSaga'

# ========== FinBERT SETUP ==========
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
labels = ['positive', 'negative', 'neutral']

# ========== spaCy for Company Name Extraction ==========
nlp = spacy.load("en_core_web_sm")

def extract_company_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return ent.text.strip()
    return "Unknown"

# ========== Text Cleaning & Classification ==========
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def classify_financial_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return {
        "sentiment": labels[predicted_class],
        "confidence": round(probs[0][predicted_class].item(), 3)
    }

# ========== Async Telegram Message Fetch ==========
async def fetch_messages():
    client = TelegramClient('session', api_id, api_hash)
    await client.start()

    messages_data = []
    async for message in client.iter_messages(channel_username, limit=100):
        if message.text:
            clean_msg = clean_text(message.text)
            result = classify_financial_sentiment(clean_msg)
            company = extract_company_name(clean_msg)
            messages_data.append({
                "message": clean_msg,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "company": company
            })

    await client.disconnect()
    return pd.DataFrame(messages_data)

# ========== Run Asyncio Loop Inside Streamlit ==========
@st.cache_data
def run_async_task():
    return asyncio.run(fetch_messages())

# ========== Streamlit UI ==========
st.title("📊 Financial News Sentiment Dashboard")

df = run_async_task()
if df.empty:
    st.warning("No messages found.")
else:
    for sentiment in ['positive', 'negative', 'neutral']:
        st.subheader(f"📍 Top 15 Companies - {sentiment.capitalize()} Sentiment")

        filtered = df[df['sentiment'] == sentiment]
        if not filtered.empty:
            top_companies = (
                filtered.groupby('company')['confidence']
                .max()
                .reset_index()
                .sort_values(by='confidence', ascending=False)
                .head(15)
            )

            # Bar chart using Altair
            chart = alt.Chart(top_companies).mark_bar().encode(
                x=alt.X('company:N', sort='-y', title='Company'),
                y=alt.Y('confidence:Q', title='Confidence'),
                tooltip=['company', 'confidence']
            ).properties(width=700, height=400)

            st.altair_chart(chart)

            # Show top message per company
            st.markdown("**📰 Most Confident News per Company:**")
            for comp in top_companies['company']:
                top_msg = filtered[filtered['company'] == comp].sort_values('confidence', ascending=False).iloc[0]
                st.write(f"📰 {top_msg['message']}")
                st.write(f"🔹 Company: {comp}, Confidence: {top_msg['confidence']}")
                st.markdown("---")
        else:
            st.write("No messages with this sentiment.")
