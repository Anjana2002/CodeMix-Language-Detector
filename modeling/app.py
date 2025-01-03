import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch


model_path = "./saved_xlmroberta"  
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

st.title("XLM-RoBERTa Model")


label_mapping = {
    'eng': 'english',
    'mal': 'malayalam',
    'mix': 'mixed',
    'acr': 'acronyms',
    'univ': 'universal'
}


input_text = st.text_area("Enter text:")

if st.button("Predict"):
   
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

   
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    
    words_and_labels = []
    current_word = ""
    current_label = None

    for token, label in zip(tokens, labels):
        if token.startswith("▁"):  
            
            if current_word and current_word not in ['<s>', '</s>']:  # Ignore special tokens
               
                mapped_label = label_mapping.get(current_label, current_label) if current_label else None
                words_and_labels.append((current_word, mapped_label))
            current_word = token[1:]  
            current_label = label
        else:
            current_word += token  

  
    if current_word and current_word not in ['<s>', '</s>', '']:
        # Map the label to the custom label
        mapped_label = label_mapping.get(current_label, current_label) if current_label else None
        words_and_labels.append((current_word, mapped_label))

    
    words_and_labels = [(word, label) for word, label in words_and_labels if word != '</s>' and word != '']

    # Display the predictions
    st.write("### Predictions (Word and Label):")
    for word, label in words_and_labels:
        st.write(f"**Word:** {word} | **Label:** {label}")