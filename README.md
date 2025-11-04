### 📰 Fake News Detection using LSTM and NLP
## 📘 Overview

This project builds a Deep Learning model using Recurrent Neural Networks (RNN – LSTM) to classify news headlines or articles as real or fake.
It leverages Natural Language Processing (NLP) techniques for text cleaning and tokenization, and trains a neural network capable of learning semantic patterns in text.

## 🧠 Machine Learning & Deep Learning Focus

This project demonstrates:

- Text Preprocessing with NLP libraries
  - Tokenization and stopword removal using spaCy and NLTK
  - Handling missing data and cleaning large text datasets

- Sequence Modeling with LSTM Networks
  - Word embeddings to capture semantic meaning
  - Long Short-Term Memory (LSTM) for sequential pattern recognition

- Model Evaluation & Visualization
    - Accuracy/Loss plots over training epochs
    - Performance metrics on unseen data

## 🧩 Techniques and Methods Used
**🔹 Data Preprocessing**
    - Loaded dataset using Pandas
    - Cleaned and normalized text using spaCy:
      - Removed stopwords and punctuation
      - Tokenized text
    - Applied NLTK stopwords for cross-validation
    - Converted text to numeric sequences using Keras Tokenizer
    - Padded sequences to ensure uniform input length

**🔹 Model Building**
Built an LSTM-based Neural Network using TensorFlow/Keras:

  Embedding → LSTM(128) → Dense(1, Sigmoid)

  - Embedding Layer: Transforms words into dense vector representations
  - LSTM Layer: Captures contextual dependencies in text
  - Dense Output: Binary classification (Fake or Real)

**🔹 Model Compilation**


<img width="267" height="112" alt="image" src="https://github.com/user-attachments/assets/a2a2ad2b-5703-494d-888e-cadf406bcae7" />


🔹 Training

  - Trained for 5 epochs
  - Batch size: 64
  - Validation split: 20%
  - Evaluated accuracy and loss on test data



## ⚙️ Dependencies

Install required libraries:

pip install pandas nltk spacy tensorflow matplotlib scikit-learn
python -m spacy download en_core_web_sm

📈 Results
**Accuracy	Model performance on unseen data	~90%**
<img width="640" height="473" alt="image" src="https://github.com/user-attachments/assets/8fe6d65d-90ee-4f5e-a904-59cc1309f7d2" />

Loss	Binary cross-entropy	~0.25


I

🧑‍💻 Author

Your Name
📧 [your.email@example.com
]
💻 [GitHub Profile Link]
