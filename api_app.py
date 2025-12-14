from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import sys

# --- Configuration ---
MODEL_FILE = 'trained_spam_classifier_model.pkl'

# --- 1. Model Loading ---
pipeline = None
try:
    if not os.path.exists(MODEL_FILE):
        print(f"CRITICAL ERROR: Model file '{MODEL_FILE}' not found.")
        sys.exit(1)

    # Load the trained scikit-learn Pipeline (Vectorizer + Classifier)
    pipeline = joblib.load(MODEL_FILE)
    print(f"Model '{MODEL_FILE}' loaded successfully.")

except Exception as e:
    print("-" * 50)
    print(f"CRITICAL MODEL LOADING ERROR: {e}")
    print("FastAPI will start, but prediction requests will fail.")
    print("-" * 50)


# --- 2. FastAPI Setup ---
app = FastAPI(
    title="Spam Classification API",
    description="A simple API to classify text messages as SPAM or HAM."
)

# --- 3. Define Input Data Structure (Pydantic Schema) ---
class TextIn(BaseModel):
    """Defines the structure of the incoming request body."""
    text: str
    
# --- 4. Define Output Data Structure (Pydantic Schema) ---
class PredictionOut(BaseModel):
    """Defines the structure of the outgoing response body."""
    text: str
    classification: str

# --- 5. API Endpoint for Prediction ---
@app.post("/predict/", response_model=PredictionOut)
def predict_spam(input_data: TextIn):
    """
    Classifies the input text message using the loaded pipeline.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded.")
        
    # The pipeline expects a list containing the single text string: ['Your text']
    data_to_predict = [input_data.text]

    try:
        # Predict: transform text to features, then classify
        prediction_result = pipeline.predict(data_to_predict)[0]
        
        # Determine the label
        if prediction_result == 1 or str(prediction_result).lower() in ('spam', '1'):
            label = "SPAM"
        else:
            label = "HAM (Not Spam)"
            
        return PredictionOut(
            text=input_data.text,
            classification=label
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# --- 6. Root Endpoint for Health Check ---
@app.get("/")
def home():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": pipeline is not None}
