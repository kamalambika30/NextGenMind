import pandas as pd
import numpy as np 

# Load dataset
data = pd.read_csv(r"C:\Users\kamal\OneDrive\ドキュメント\AI vs HUMAN txt detection\balanced_ai_human_prompts.csv")

# Print columns
print("Columns in dataset:", data.columns)

# Auto detect columns
text_col = data.columns[0]
label_col = data.columns[-1]

print("Using TEXT column:", text_col)
print("Using LABEL column:", label_col)

# Cleaning
data.dropna(subset=[text_col, label_col], inplace=True)

data[text_col] = data[text_col].astype(str)
data[label_col] = data[label_col].astype(str)

data = data[data[text_col].str.strip() != ""]
data = data[data[label_col].str.strip() != ""]

# Features & target
X = data[text_col]
y = data[label_col]

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Text cleaning
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X = X.apply(clean_text)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(X)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

models = {
    "SVC": SVC(kernel='linear', probability=True), 
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

# Train & evaluate
from sklearn.metrics import accuracy_score

for name, model in models.items():
    print(f"\n {name}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Confidence score
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        confidence = np.max(probs, axis=1) * 100
        
        print("Sample Confidence Scores:")
        for i in range(5):   # show first 5
            print(f"{confidence[i]:.2f}%")
    else:
        print("No confidence score available")


import pickle

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Save label encoder
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Files saved successfully!")


import joblib

# Fix the path error here
model = joblib.load(r'C:\Users\kamal\Hackathon 1st\model.pkl')
vectorizer = joblib.load(r'C:\Users\kamal\Hackathon 1st\vectorizer.pkl')
label_encoder = joblib.load(r'C:\Users\kamal\Hackathon 1st\label_encoder.pkl')

def get_prediction(text):
    # Process text
    vectorized = vectorizer.transform([text])
    
    # Get probability (usually returns [prob_human, prob_ai])
    probs = model.predict_proba(vectorized)[0]
    conf_score = max(probs)
    pred_idx = probs.argmax()
    
    # Decision System Logic
    if conf_score >= 0.80:
        decision = " Acceptable (High certainty)"
    elif 0.60 <= conf_score < 0.80:
        decision = " Needs Review (Moderate certainty)"
    else:
        decision = ".0Likely AI-generated / Uncertain"
        
    return {
        "prediction": "AI-generated" if pred_idx == 1 else "Human",
        "confidence": conf_score,
        "decision": decision
    }
