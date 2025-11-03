# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np
import google.generativeai as genai
import time  # For retry delay

app = Flask(__name__)

# Gemini Config
GEMINI_API_KEY = "AIzaSyDwIMXR9mu0PQMcNK8B55V7xOcrmnOBtXg"
genai.configure(api_key=GEMINI_API_KEY)
# Primary: gemini-2.5-flash | Fallback: gemini-2.0-flash (higher free quota)
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except:
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')  # Auto-fallback

# Load model/scaler
try:
    model = joblib.load('logistic_heart_model.pkl')
    scaler = joblib.load('scaler (3).pkl')
    print("Loaded model & scaler")
except FileNotFoundError:
    print("Run train_and_save.py first!")
    exit()

def get_gemini_details(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            if '429' in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            return f"AI unavailable (quota/error): {str(e)}. Try again later."

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    details = "Submit a prediction to see AI analysis."
    if request.method == 'POST':
        try:
            features = [float(request.form.get(k)) for k in [
                'age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal'
            ]]
            scaled = scaler.transform([features])
            pred = model.predict(scaled)[0]
            result = "Heart Disease" if pred == 1 else "No Heart Disease"

            prompt = (
                "Detailed heart disease info: causes, symptoms, prevention, treatment. Concise."
                if pred else
                "Heart health tips to prevent disease. Concise."
            )
            details = get_gemini_details(prompt)

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result, details=details)

 
if __name__ == '__main__':
    print("Go to: http://127.0.0.1:5000")
    app.run(debug=True)