import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Load the saved model and tokenizer
model_path = "./saved_xlmroberta"  # Path to your model directory
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

st.title("XLM-RoBERTa NER Model")

# Define a custom label mapping
label_mapping = {
    'eng': 'english',
    'mal': 'malayalam',
    'mix': 'mixed',
    'acr': 'acronyms',
    'univ': 'universal'
}

# Text input
input_text = st.text_area("Enter text:")

if st.button("Predict"):
    # Tokenize input and get predictions
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Decode predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    # Combine tokens into words and filter out special tokens
    words_and_labels = []
    current_word = ""
    current_label = None

    for token, label in zip(tokens, labels):
        if token.startswith("▁"):  # New word
            # If there's a current word, add it to the list (if valid)
            if current_word and current_word not in ['<s>', '</s>']:  # Ignore special tokens
                # Map the label to the custom label
                mapped_label = label_mapping.get(current_label, current_label) if current_label else None
                words_and_labels.append((current_word, mapped_label))
            current_word = token[1:]  # Remove the subword marker (▁)
            current_label = label
        else:
            current_word += token  # Append subword to the current word

    # Add the last word if it's not a special token
    if current_word and current_word not in ['<s>', '</s>', '']:
        # Map the label to the custom label
        mapped_label = label_mapping.get(current_label, current_label) if current_label else None
        words_and_labels.append((current_word, mapped_label))

    # Filter out any occurrences of `</s>` from words_and_labels
    words_and_labels = [(word, label) for word, label in words_and_labels if word != '</s>' and word != '']

    # Display the predictions
    st.write("### Predictions (Word and Label):")
    for word, label in words_and_labels:
        st.write(f"**Word:** {word} | **Label:** {label}")
