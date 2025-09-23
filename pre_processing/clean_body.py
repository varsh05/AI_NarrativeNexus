# clean_body.py
import os
import re
import pandas as pd
import email
from email import policy

# ---------------- CONFIG ----------------
DATASET_DIR = r"D:\narrative nexus\20news-18828-20250915T192107Z-1-001"
OUTPUT_CSV = r"D:\narrative nexus\cleaned_data.csv"
MAX_FILES_PER_CATEGORY = 150  # optional limit per category

# ---------------- CLEANING FUNCTION ----------------
def clean_body(raw_text: str) -> str:
    """Clean raw email or text content."""
    if not raw_text:
        return ""

    # Separate headers and body
    parts = re.split(r"\n\s*\n", raw_text, maxsplit=1)
    header = parts[0]
    body = parts[1] if len(parts) > 1 else parts[0]

    # Capture subject if exists
    subject_match = re.search(r"^subject:\s*(.*)", header, re.I | re.M)
    subject = subject_match.group(1).strip() if subject_match else ""

    cleaned_lines = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith((">", "|", "--")):
            continue
        if re.search(r"writes:|wrote:|In article\s*<.*?>", line, re.I):
            continue
        cleaned_lines.append(line)

    body = "\n".join(cleaned_lines)

    # Remove emails, URLs, HTML
    body = re.sub(r"\S+@\S+", " ", body)
    body = re.sub(r"http\S+|www\.\S+", " ", body)
    body = re.sub(r"<[^>]+>", " ", body)

    # Remove unwanted chars, keep punctuation
    body = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\:\;\-\(\)\'\"]", " ", body)

    # Collapse spaces/newlines
    body = re.sub(r"\n{2,}", "\n", body)
    body = re.sub(r"\s{2,}", " ", body)

    body = body.lower().strip()

    # Add subject at the top
    if subject:
        body = f"[subject] {subject.lower()}\n{body}"

    return body

# ---------------- EXTRACT FILE BODY ----------------
def extract_file_body(file_path: str) -> str:
    """Extract text from .eml or plain text files."""
    try:
        if file_path.lower().endswith(".eml"):
            with open(file_path, "r", encoding="latin1") as f:
                msg = email.message_from_file(f, policy=policy.default)

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        charset = part.get_content_charset() or "latin1"
                        body += part.get_payload(decode=True).decode(charset, errors="ignore")
                        break
            else:
                charset = msg.get_content_charset() or "latin1"
                payload = msg.get_payload(decode=True)
                body = payload.decode(charset, errors="ignore") if payload else msg.get_payload()
            return body
        else:
            # plain text file
            with open(file_path, "r", encoding="latin1") as f:
                return f.read()
    except Exception as e:
        print(f"⚠️ Failed to read {file_path}: {e}")
        return ""

# ---------------- PROCESS DATASET ----------------
data_rows = []
category_counts = {}

for root, dirs, files in os.walk(DATASET_DIR):
    category_name = os.path.basename(root)
    if category_name not in category_counts:
        category_counts[category_name] = 0

    for file_name in files:
        if category_counts[category_name] >= MAX_FILES_PER_CATEGORY:
            continue

        if not file_name.lower().endswith((".eml", ".txt")):
            continue

        file_path = os.path.join(root, file_name)
        print(f"Processing: {file_path}")

        raw_text = extract_file_body(file_path)
        cleaned_text = clean_body(raw_text)
        if cleaned_text:
            data_rows.append({
                "file_name": file_name,
                "category": category_name,
                "text": cleaned_text
            })
            category_counts[category_name] += 1

# ---------------- SAVE TO CSV ----------------
df = pd.DataFrame(data_rows, columns=["file_name", "category", "text"])
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\n✅ Cleaned dataset saved at: {OUTPUT_CSV}")
print(f"📊 Total rows: {len(df)}")
print(f"📋 Categories included: {len(df['category'].unique())}")
