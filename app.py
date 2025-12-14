import joblib
import os
from flask import Flask, request, render_template_string

# --- Configuration ---
# The name of your uploaded model file
MODEL_FILE = 'trained_spam_classifier_model.pkl'

# --- Load the Model/Pipeline ---
# IMPORTANT NOTE: This file must contain the trained scikit-learn Pipeline 
# object that includes the text vectorizer and the classifier.
try:
    pipeline = joblib.load(MODEL_FILE)
    print(f"Successfully loaded model from {MODEL_FILE}")
except FileNotFoundError:
    pipeline = None
    print(f"ERROR: Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
except Exception as e:
    pipeline = None
    print(f"ERROR: Could not load model. Check if it's a valid scikit-learn object. Details: {e}")

# --- Flask Application Setup ---
app = Flask(__name__)

# Basic HTML template for the web interface
HTML_FORM = """
<!doctype html>
<title>Spam Classifier</title>
<h1 style="color: #333; font-family: Arial, sans-serif;">Spam or Ham SMS Predictor</h1>
<form method="POST" style="padding: 20px; border: 1px solid #ccc; border-radius: 5px; max-width: 500px;">
    <label for="sms_text" style="display: block; margin-bottom: 8px; font-family: Arial, sans-serif;">Enter SMS Text:</label>
    <textarea id="sms_text" name="sms_text" rows="5" cols="50" style="width: 100%; padding: 10px; box-sizing: border-box; border: 1px solid #ddd; border-radius: 4px;" required>{{ text_input }}</textarea><br>
    <input type="submit" value="Classify" style="background-color: #4CAF50; color: white; padding: 10px 15px; margin-top: 10px; border: none; border-radius: 4px; cursor: pointer;">
</form>
{% if prediction_text %}
<div style="margin-top: 20px; padding: 15px; border: 1px solid {% if 'SPAM' in prediction_text %}#f44336{% else %}#4CAF50{% endif %}; border-radius: 5px; max-width: 500px; background-color: {% if 'SPAM' in prediction_text %}#ffe0e0{% else %}#e0ffe0{% endif %}; font-family: Arial, sans-serif;">
    <h2>Prediction Result:</h2>
    <p style="font-size: 1.2em;">{{ prediction_text }}</p>
</div>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    if pipeline is None:
        # If the model failed to load, display an error message
        return render_template_string(
            "<h1>Error</h1><p>The application could not start because the model file 'trained_spam_classifier_model.pkl' failed to load. Check the console for details.</p>"
        ), 500

    prediction_text = None
    text_input = ""

    if request.method == 'POST':
        # Get the text input from the form
        text_input = request.form.get('sms_text', '')
        
        if text_input:
            # The model/pipeline expects a list-like object of texts for prediction
            data_to_predict = [text_input]

            try:
                # Use the loaded pipeline to transform the text and predict
                prediction_result = pipeline.predict(data_to_predict)[0]
                
                # Assuming the model was trained with 'spam' (1) and 'ham' (0) labels
                if prediction_result == 1 or prediction_result in ('spam', '1'):
                    label = "SPAM"
                elif prediction_result == 0 or prediction_result in ('ham', '0'):
                    label = "HAM (Not Spam)"
                else:
                    label = f"Unknown result: {prediction_result}"
                    
                prediction_text = f"The message is: {label}"

            except Exception as e:
                # Catch prediction errors (e.g., wrong input format, missing features)
                prediction_text = f"An error occurred during prediction: {e}"

    # Render the form with the prediction result (if any) and the text input
    return render_template_string(
        HTML_FORM, 
        prediction_text=prediction_text, 
        text_input=text_input
    )

if __name__ == '__main__':
    # Run the application. Use debug=True for local development.
    # Note: For production deployment, set debug=False.
    app.run(debug=True)
