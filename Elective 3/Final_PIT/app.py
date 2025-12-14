import re
import pandas as pd
import gradio as gr

from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from autocorrect import spell

# =========================
# Load dataset
# =========================
profanity_words = pd.read_csv("English_profanity_words.csv", nrows=1000)

# =========================
# NLP tools
# =========================
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    words = []

    for token in tokens:
        corrected_token = spell(token)
        stemmed_token = stemmer.stem(corrected_token)
        lemmatized_token = lemmatizer.lemmatize(stemmed_token)
        words.append(lemmatized_token)

    return " ".join(words)

# =========================
# Preprocess dataset
# =========================
profanity_words["processed_text"] = profanity_words["text"].apply(preprocess_text)

# =========================
# Vectorization
# =========================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(profanity_words["processed_text"])
y = profanity_words["is_offensive"]

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Train models
# =========================
models = {
    "Linear SVM": LinearSVC(class_weight="balanced"),
    "SGD (Hinge Loss)": SGDClassifier(
        loss="hinge",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    "Passive Aggressive": PassiveAggressiveClassifier(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# =========================
# Prediction function (Gradio)
# =========================
def predict_profanity(text, model_name):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])

    model = trained_models[model_name]
    prediction = model.predict(vectorized)[0]

    label = "🚫 Offensive" if prediction == 1 else "✅ Not Offensive"
    return label

# =========================
# Gradio Interface
# =========================
interface = gr.Interface(
    fn=predict_profanity,
    inputs=[
        gr.Textbox(
            label="Enter Text",
            placeholder="Type a sentence here...",
            lines=3
        ),
        gr.Dropdown(
            choices=list(trained_models.keys()),
            label="Choose Model",
            value="Linear SVM"
        )
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Profanity Detection using NLP & SVM",
    description=(
        "This app detects whether a text is offensive or not using "
        "TF-IDF and classical machine learning models."
    ),
    examples=[
        ["You are stupid", "Linear SVM"],
        ["Have a nice day", "SGD (Hinge Loss)"]
    ]
)

if __name__ == "__main__":
    interface.launch()
