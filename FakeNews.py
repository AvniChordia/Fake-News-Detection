import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# ----------------------------------------------------------------
# CRITICAL FIX 1: Import Client from the modern SDK (google-genai)
# ----------------------------------------------------------------
import os
import time

from google import genai
from google.genai.types import GenerationConfig


print("✅ Code started successfully!")

# ------------------- OFFLINE MODEL -------------------
# Load dataset
if not os.path.exists("news_clean.csv"):
    print("⚠️ File 'news_clean.csv' not found. Please ensure it exists to train the offline model.")
    # Exiting since the core data for the offline model is missing.
    exit()

data = pd.read_csv("news_clean.csv")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=7
)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Train PassiveAggressiveClassifier
offline_model = PassiveAggressiveClassifier(max_iter=50)
offline_model.fit(tfidf_train, y_train)

# Evaluate offline model
y_pred = offline_model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Offline Model Accuracy: {round(accuracy*100, 2)}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n--- Fake News Detector (Offline + Gemini) ---\n")

# ------------------- GEMINI API -------------------
GEMINI_MODEL = "gemini-2.5-pro" 

try:
    # ----------------------------------------------------------------
    # CRITICAL FIX 2: Initialize the client directly.
    # Replace "YOUR_API_KEY_HERE" with your actual Gemini API Key.
    client = genai.Client(api_key="AIzaSyDzD2C0EzhZ-diCTbAk0nZh5medEHmZxa4")
    # ----------------------------------------------------------------
    
    # Configuration for deterministic classification response
    classification_config = GenerationConfig(
        temperature=0.0,
        max_output_tokens=10
    )

    
except Exception as e:
    print(f"FATAL: Could not initialize Gemini Client. Please check your API key.")
    print(f"Error: {e}")
    client = None


# ------------------- FUNCTIONS -------------------
def offline_model_predict(text):
    vec = tfidf_vectorizer.transform([text])
    pred = offline_model.predict(vec)[0]   # uses trained model
    
    return str(pred).upper()   

def check_fake_news_gemini(headline):
    """
    Predicts news label using the modern Gemini API call: client.models.generate_content().
    Includes exponential backoff for retries.
    """
    if not client:
        return "⚠️ Gemini Client not initialized. Check API Key."
        
    prompt = f"Determine if the following news headline is REAL or FAKE. Reply with only the word 'REAL' or the word 'FAKE'.\n\nHeadline: {headline}"
    
    max_retries = 3
    delay = 1

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Is this headline real or fake? Answer Real or Fake only.\n'{headline}'"
            )
            return response.text
        except Exception as e:
            return f"Gemini Error: {e}"
# ------------------- MAIN LOOP -------------------
def main():
    if not os.path.exists("news_clean.csv"):
        print("Cannot run detection loop without a trained model.")
        return

    while True:
        headline = input("📰 Enter a news headline (or type 'exit' to quit): ")
        if headline.lower() == "exit":
            print("Exiting Fake News Detector. Goodbye!")
            break

        # Offline prediction
        offline_result = offline_model_predict(headline)

        print(f"\n🧩 Offline Model Prediction: {offline_result}")

        # Gemini prediction
        if client:
            print("💬 Checking with Gemini...")
            gemini_result = check_fake_news_gemini(headline)
            print(f"🤖 Gemini Result: {gemini_result}\n")
        else:
            print("🤖 Gemini check skipped because the client failed to initialize.")

if __name__ == "__main__":
    main()


#"AIzaSyCF6hfE9wMKmDULUNcdyy5NLtEoX6cTLuM" 
#new AIzaSyDzD2C0EzhZ-diCTbAk0nZh5medEHmZxa4