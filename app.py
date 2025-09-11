import streamlit as st
import traceback

# =========================
# App Config
# =========================
st.set_page_config(page_title="Narrative Nexus", layout="wide")

try:
    # =========================
    # Title & Description
    # =========================
    st.title("📊 NarrativeNexus Data Collector")
    st.write("Comprehensive data collection from files, Reddit posts, and news articles.")

    # =========================
    # Tabs
    # =========================
    tab1, tab2, tab3 = st.tabs([
        "📄 File Upload", 
        "🔗 Reddit Posts", 
        "📰 News Articles"
    ])

    # -------------------------
    # Tab 1: File Upload
    # -------------------------
    with tab1:
        st.header("📄 File Upload")
        st.write("Upload text, CSV, DOCX, or PDF files for analysis and storage.")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file:
            st.success(f"✅ You uploaded: {uploaded_file.name}")

    # -------------------------
    # Tab 2: Reddit Posts
    # -------------------------
    with tab2:
        st.header("🔗 Reddit Post Collector")
        reddit_url = st.text_input("Enter Reddit post URL")
        if st.button("Fetch Reddit Post"):
            if reddit_url.strip():
                st.info(f"Fetching post from: {reddit_url}")
            else:
                st.warning("⚠️ Please enter a Reddit URL.")

    # -------------------------
    # Tab 3: News Articles
    # -------------------------
    with tab3:
        st.header("📰 News Article Collector")
        news_query = st.text_input("Enter search keywords")
        if st.button("Fetch News Article"):
            if news_query.strip():
                st.info(f"Searching for news about: {news_query}")
            else:
                st.warning("⚠️ Please enter search keywords.")

except Exception as e:
    st.error(f"❌ Error in app: {e}")
    st.code(traceback.format_exc())
