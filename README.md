# Transformer-Based Language Identification for Malayalam-English Code-Mixed Text

## Overview

This repository tackles the challenge of identifying languages in code-mixed text, with a focus on Malayalam-English data frequently observed on social media platforms like YouTube. Code-mixing, where two or more languages are used within a single sentence, is a common phenomenon in under-resourced languages and presents significant hurdles for natural language processing (NLP).

The goal of this project is to perform Word-Level Language Identification (WLLI) using advanced transformer models like BERT and its variants, including CamemBERT and DistilBERT.

## Dataset

The dataset comprises over 50,000 Malayalam-English code-mixed sentences, collected using the Google API Python client. The data was sourced from YouTube comments spanning various domains, including politics, sports, entertainment, and food.

## Preprocessing

The preprocessing step involves filtering bilingual Malayalam-English sentences from raw YouTube comments, which include Roman scripts, native scripts, and mixed-language content. A function using regular expressions identifies English alphabets, numbers, special characters, and emoticons, discarding sentences with fewer than six words to produce a refined dataset ready for annotation.

## Annotation

After lemmatization and tokenization, words are tagged, dataset is labeled with six tags: 'eng' (English), 'mal' (Malayalam), 'univ' (universal), 'mix' (mixed), 'acr' (acronyms), and 'undef' (undefined).

## Modeling

The project utilizes various BERT-based models, including BERT, CamemBERT, DistilBERT, ELECTRA, and XLM-RoBERTa, for Word-Level Language Identification (WLLI) on the Malayalam-English code-mixed dataset. Models are evaluated and compared using metrics such as F1-score, accuracy, precision, and recall to determine the best-performing approach for language identification.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Anjana2002/Language-Identification-for-Malayalam-English-Code-Mixed-Text.git
   ```

## Acknowledgment

This project draws inspiration from the IEEE paper "Transformer-Based Language Identification for Malayalam-English Code-Mixed Text" by S. Thara and Prabaharan Poornachandran.
