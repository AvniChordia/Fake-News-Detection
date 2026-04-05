from flask import Flask, request, jsonify, render_template
import pandas as pd
import time
from dotenv import load_dotenv
import os
# ML imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Gemini imports (unchanged)
from google import genai
from google.genai.types import GenerationConfig

app = Flask(__name__)

# ---------------------- LOAD OFFLINE MODEL -----------------------
data = pd.read_csv("news_clean.csv")
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=7
)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

offline_model = PassiveAggressiveClassifier(max_iter=50)
offline_model.fit(tfidf_train, y_train)

# ---------------------- GEMINI INIT ------------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
gen_config = GenerationConfig(temperature=0.0, max_output_tokens=10)

# ---------------------- FUNCTIONS ------------------------------
def offline_predict(text):
    vec = vectorizer.transform([text])
    pred = offline_model.predict(vec)[0]
    return pred

def gemini_predict(text):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Is this headline real or fake? Reply Real or Fake only:\n{text}"
    )
    return response.text.strip()

# ---------------------- ROUTES ------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    headline = request.form["headline"]
    offline_result = offline_predict(headline)
    gemini_result = "Not Available"

    return jsonify({
        "offline": offline_result,
        "gemini": gemini_result
    })

if __name__ == "__main__":
    app.run(debug=True)
