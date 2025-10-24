# summarize_module.py
from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text(text, max_length=150, min_length=40):
    # same logic as before
    ...

def summarize_file(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    return summarize_text(text)

if __name__ == "__main__":
    summary = summarize_file("train_rf.py")
    print(summary)