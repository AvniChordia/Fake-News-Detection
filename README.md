# 📰 Fake News Detection System
## 📌 Overview
This project is a web-based application that detects whether a news headline is **REAL or FAKE** using Machine Learning and AI techniques.
It uses a hybrid approach combining:
* **Offline ML model (TF-IDF + Passive Aggressive Classifier)**
* **Gemini API (LLM-based validation with fallback support)**
---
## 🚀 Features
* Real-time fake news detection
* NLP-based text processing using TF-IDF
* Machine Learning classification
* Gemini API integration (with fallback handling)
* Simple and user-friendly web interface
---
## 🛠️ Tech Stack
* Python
* Flask
* Scikit-learn
* Pandas
* HTML, CSS
* Gemini API
---
## ⚙️ How It Works
1. User enters a news headline
2. Text is preprocessed and vectorized using TF-IDF
3. Machine Learning model predicts REAL or FAKE
4. Gemini API (if available) cross-verifies the result
5. Output is displayed on the web interface
---
## 💻 Setup Instructions
### 1. Clone the repository
```bash
git clone https://github.com/AvniChordia/Fake-News-Detection.git
cd Fake-News-Detection
```
### 2. Install dependencies
```bash
pip install flask pandas scikit-learn python-dotenv
```
### 3. Create `.env` file
```text
GEMINI_API_KEY=your_api_key_here
```
### 4. Run the application
```bash
python FAkeNewsNew.py
```
### 5. Open in browser
```
http://127.0.0.1:5000/
```
---
## ⚠️ Note
* Dataset files are not included due to GitHub file size limits
* Gemini API may not work if quota is exceeded (fallback is implemented)
---
## 🎯 Future Improvements
* Add confidence score
* Improve UI/UX
* Use deep learning models
* Deploy on cloud
