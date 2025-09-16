import streamlit as st
import pandas as pd
import json
import os
import uuid
import requests
from datetime import datetime, timezone

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("⚠️ python-docx not installed. DOCX files won't be supported.")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ pdfplumber not installed. PDF files won't be supported.")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    st.warning("⚠️ praw not installed. Reddit fetching won't be available.")

# =========================
# Config & Data Path
# =========================
st.set_page_config(page_title="Narrative Nexus", layout="wide")

DATA_DIR = r"D:\narrative nexus\data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "data_store.json")

CONFIG_PATH = r"D:\narrative nexus\config.json"

# Load credentials from config.json
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    st.error(f"Config file not found at {CONFIG_PATH}")
    config = {}

REDDIT_CLIENT_ID = config.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = config.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = config.get("REDDIT_USER_AGENT")
NEWS_API_KEY = config.get("NEWS_API_KEY")

# =========================
# Helper Functions
# =========================
def save_data(source, content):
    """Save content to JSON ensuring consistent structure"""
    record = {
        "id": str(uuid.uuid4()),
        "source": source,
        "author": content.get("author", "Unknown") if isinstance(content, dict) else "Unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": content.get("text", content if isinstance(content, str) else ""),
        "metadata": content.get("metadata", {}) if isinstance(content, dict) else {}
    }

    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    st.success(f"✅ Saved {source} successfully")

def extract_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def fetch_reddit_post(url):
    """Fetch a single Reddit post"""
    if not PRAW_AVAILABLE or not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        st.error("Reddit credentials missing or praw not installed")
        return None
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    try:
        submission = reddit.submission(url=url)
        return {
            "author": submission.author.name if submission.author else "Unknown",
            "text": submission.title + "\n" + submission.selftext,
            "metadata": {
                "url": url,
                "likes": submission.score,
                "language": "en",
                "rating": None
            }
        }
    except Exception as e:
        st.error(f"❌ Error fetching Reddit post: {e}")
        return None

def fetch_news(query):
    """Fetch first news article from NewsAPI"""
    if not NEWS_API_KEY:
        st.error("NewsAPI key missing in config")
        return None
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        response = requests.get(url).json()
        if "articles" not in response or len(response["articles"]) == 0:
            st.warning("No news found for this query.")
            return None
        article = response["articles"][0]
        return {
            "author": article.get("author") or "Unknown",
            "text": (article.get("title") or "") + "\n" + (article.get("description") or ""),
            "metadata": {
                "url": article.get("url"),
                "language": article.get("language", "en"),
                "rating": None,
                "likes": None
            }
        }
    except Exception as e:
        st.error(f"❌ Error fetching news: {e}")
        return None

# =========================
# UI
# =========================
st.title("📊 NarrativeNexus Data Collector")
st.write("Upload files, paste text, or fetch Reddit/News data and save to JSON.")

tab1, tab2, tab3 = st.tabs(["📄 File Upload", "🔗 Reddit Posts", "📰 News Articles"])

# -------------------------
# Tab 1: File Upload
# -------------------------
with tab1:
    st.header("📄 File Upload")
    uploaded_file = st.file_uploader("Upload a file", type=["txt","csv","docx","pdf"])
    text_input = st.text_area("Or paste text here:", height=150)

    content = None
    filename = None

    if uploaded_file:
        filename = uploaded_file.name
        ext = filename.split(".")[-1].lower()
        if ext == "txt":
            content = uploaded_file.read().decode("utf-8")
        elif ext == "csv":
            df = pd.read_csv(uploaded_file)
            content = df.to_string()
        elif ext == "docx" and DOCX_AVAILABLE:
            content = extract_docx(uploaded_file)
        elif ext == "pdf" and PDF_AVAILABLE:
            content = extract_pdf(uploaded_file)
        else:
            st.warning("⚠️ Unsupported file type or missing library")
    elif text_input.strip():
        filename = "pasted_text"
        content = text_input

    if content:
        st.text_area("Preview Content", content[:1000], height=200)
        if st.button("Save Content"):
            save_data(filename, content)

# -------------------------
# Tab 2: Reddit Post
# -------------------------
with tab2:
    st.header("🔗 Fetch Reddit Post")
    reddit_url = st.text_input("Enter Reddit post URL:")
    if st.button("Fetch Reddit Post"):
        if reddit_url.strip():
            record = fetch_reddit_post(reddit_url)
            if record:
                save_data("reddit_post", record)
                st.subheader("Preview")
                st.json(record)

# -------------------------
# Tab 3: News Article
# -------------------------
with tab3:
    st.header("📰 Fetch News Article")
    news_query = st.text_input("Enter News query:")
    if st.button("Fetch News Article"):
        if news_query.strip():
            record = fetch_news(news_query)
            if record:
                save_data("news_article", record)
                st.subheader("Preview")
                st.json(record)

# -------------------------
# View Data
# -------------------------
st.header("📂 Stored Data")
if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
    df = pd.read_json(DATA_FILE)
    expected_cols = ["id","source","author","timestamp","text"]
    available_cols = [c for c in expected_cols if c in df.columns]
    st.dataframe(df[available_cols])
else:
    st.info("No data stored yet.")
