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

Trained for 5 epochs

Batch size: 64

Validation split: 20%

Evaluated accuracy and loss on test data

🔹 Evaluation & Visualization

Visualized:

Accuracy over epochs

Loss over epochs

📊 Project Structure
fake-news-lstm/
│
├── train.csv                  # Dataset (contains text and label columns)
├── main.py                    # Main script (data prep, model training, testing)
├── README.md                  # Project documentation
└── results/
    ├── accuracy_plot.png
    ├── loss_plot.png
    └── model_summary.txt

⚙️ Dependencies

Install required libraries:

pip install pandas nltk spacy tensorflow matplotlib scikit-learn
python -m spacy download en_core_web_sm

🚀 How to Run

Prepare dataset:
Make sure train.csv has at least two columns:

text → news text

label → 0 for Fake, 1 for Real

Run the script:

python main.py


View training progress:

Accuracy and loss graphs appear after training

Test accuracy and loss printed in console

Predict new text:
Example:

new_text = ["Breaking news: Scientists discover a cure for cancer!"]


Output:

Prediction: Real

📈 Results
Metric	Description	Example Result
Accuracy	Model performance on unseen data	~90%
Loss	Binary cross-entropy	~0.25
🧩 Example Workflow
# Load dataset
df = pd.read_csv('train.csv')

# Clean text using spaCy
df['cleaned_text'] = df['text'].apply(clean_text_spacy)

# Tokenize and pad
tokenizer = Tokenizer(num_words=10000)
X_pad = pad_sequences(tokenizer.texts_to_sequences(df['cleaned_text']), maxlen=100)

# Train LSTM
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

🚧 Future Improvements

Implement Bidirectional LSTM or GRU for enhanced context capture

Use pretrained word embeddings (e.g., GloVe, Word2Vec)

Experiment with Transformer-based models (BERT)

Add hyperparameter tuning (e.g., dropout, embedding size)

Deploy via Streamlit or FastAPI

🧑‍💻 Author

Your Name
📧 [your.email@example.com
]
💻 [GitHub Profile Link]
