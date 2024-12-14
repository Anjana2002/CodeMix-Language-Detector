import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Load the saved model and tokenizer
model_path = "./saved_xlmroberta"  # Path to your model directory
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

st.title("XLM-RoBERTa NER Model")
st.write("Provide text for Named Entity Recognition (NER).")

# Text input
input_text = st.text_area("Enter text:", "njn enn avide poyi, videoil njn kandu comedyu love 123 wait")

if st.button("Predict"):
    # Tokenize input and get predictions
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Decode predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    # Combine tokens into words
    words_and_labels = []
    current_word = ""
    current_label = None

    for token, label in zip(tokens, labels):
        if token.startswith("▁"):  # New word
            if current_word:
                words_and_labels.append((current_word, current_label))
            current_word = token[1:]  # Remove the subword marker
            current_label = label
        else:
            current_word += token  # Append subword to the current word

    # Add the last word
    if current_word:
        words_and_labels.append((current_word, current_label))

    # Display the predictions
    st.write("### Predictions (Word and Label):")
    for word, label in words_and_labels:
        st.write(f"**Word:** {word} | **Label:** {label}")
