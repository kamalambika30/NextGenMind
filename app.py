from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model files
model = joblib.load(r'C:\Users\kamal\Hackathon 1st\model.pkl')
vectorizer = joblib.load(r'C:\Users\kamal\Hackathon 1st\vectorizer.pkl')
label_encoder = joblib.load(r'C:\Users\kamal\Hackathon 1st\label_encoder.pkl')

# Intelligent function
def analyze_text(input_text):
    vectorized = vectorizer.transform([input_text])
    
    probs = model.predict_proba(vectorized)[0]
    
    prediction_index = np.argmax(probs)
    confidence_score = probs[prediction_index]
    
    raw_label = label_encoder.inverse_transform([prediction_index])[0]
    label_name = "Human" if raw_label == '0' else "AI-generated"

    if confidence_score >= 0.80:
        decision = "Acceptable (High certainty)"
    elif confidence_score >= 0.60:
        decision = " Needs Review (Moderate certainty)"
    else:
        decision = "Likely AI-generated / Uncertain"

    return label_name, confidence_score, decision


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    decision = None

    if request.method == "POST":
        user_text = request.form["text"]

        pred, conf, desc = analyze_text(user_text)

        prediction = pred
        confidence = round(conf * 100, 2)
        decision = desc

    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=confidence, 
                           decision=decision)


if __name__ == "__main__":
    app.run(debug=True)

print("---------------------------------classification report-----------------------------")

import pandas as pd
from sklearn.metrics import classification_report

def generate_live_report():
    test_df = pd.read_csv(r'C:\Users\kamal\OneDrive\ドキュメント\AI vs HUMAN txt detection\balanced_ai_human_prompts.csv') 
    
 
    text_col = test_df.columns[0]
    label_col = test_df.columns[-1]

    print("Using:", text_col, label_col)

    # Transform
    X_test = vectorizer.transform(test_df[text_col])
    y_true = test_df[label_col]

    # Predict
    y_pred = model.predict(X_test)

    print("\n--- Live Classification Report ---")
    print(classification_report(y_true, y_pred))

generate_live_report()