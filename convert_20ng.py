import pandas as pd
import re

def clean_body(raw_text):
    """Deep clean the body text only."""
    if pd.isna(raw_text):
        return ""

    # Split headers vs body
    parts = re.split(r"\n\s*\n", str(raw_text), maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    cleaned_lines = []
    for line in body.splitlines():
        if re.match(r"^(Archive-name|From|Subject|Path|Xref|Organization|Lines|Newsgroups|Message-ID|Keywords):", line, re.I):
            continue
        if line.strip().startswith(">"):  # skip quoted lines
            continue
        cleaned_lines.append(line)

    # Remove illegal Excel characters
    body_text = "\n".join(cleaned_lines).strip()
    body_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", body_text)

    return body_text


def sanitize_for_excel(value: str) -> str:
    """Prevent Excel from misinterpreting text as a formula."""
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value


def clean_excel(input_excel, output_excel, max_files=None):
    """Read Excel, clean body text, and save back to another Excel."""
    df = pd.read_excel(input_excel)

    # Limit rows if max_files is set
    if max_files is not None:
        df = df.head(max_files)

    # Apply cleaning to 'text' column
    if "text" in df.columns:
        df["text"] = df["text"].apply(clean_body)

    # Sanitize all cells for Excel safety
    df = df.applymap(sanitize_for_excel)

    # Save to new Excel
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"✅ Saved {len(df)} rows into {output_excel}")


# Run
clean_excel(
    input_excel=r"D:\narrative nexus\meta_data\20news_initial.xlsx",
    output_excel=r"D:\narrative nexus\meta_data\20news_clean.xlsx",
    max_files=50
)
