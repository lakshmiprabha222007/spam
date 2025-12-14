import joblib
from flask import Flask, request, render_template_string

# --- Configuration ---
MODEL_FILE = 'trained_spam_classifier_model.pkl'

# --- Load the Model/Pipeline at startup ---
try:
    # Load the entire scikit-learn Pipeline (vectorizer + classifier)
    pipeline = joblib.load(MODEL_FILE)
    print(f"Model '{MODEL_FILE}' loaded successfully.")
except Exception as e:
    # Print error and exit if model cannot be loaded
    print(f"CRITICAL ERROR: Failed to load model. Details: {e}")
    pipeline = None # Set to None so the app will show an error if run

# --- Flask App Setup ---
app = Flask(__name__)

# Very simple HTML template for the web page
SIMPLE_HTML = """
<!doctype html>
<title>Simple Spam Predictor</title>
<h1>Spam or Ham Predictor</h1>

<form method="POST">
    <textarea name="sms_text" rows="4" cols="50" placeholder="Enter SMS text here..." required>{{ text_input or '' }}</textarea><br><br>
    <input type="submit" value="Check Message">
</form>

{% if prediction_text %}
    <hr>
    <h2>Result:</h2>
    <p style="font-size: 1.2em; font-weight: bold; color: {{ 'red' if 'SPAM' in prediction_text else 'green' }};">{{ prediction_text }}</p>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    if pipeline is None:
        return "<h1>Application Error</h1><p>The prediction model failed to load. Please check the console for critical error details.</p>", 500

    prediction_text = None
    text_input = ""

    if request.method == 'POST':
        # Get text from the form
        text_input = request.form.get('sms_text', '')
        
        if text_input:
            try:
                # 1. Prepare data (must be a list)
                data_to_predict = [text_input]
                
                # 2. Make prediction using the loaded pipeline
                prediction = pipeline.predict(data_to_predict)[0]
                
                # 3. Format the result
                # Assumes 1 is SPAM and 0 is HAM
                if prediction == 1 or str(prediction).lower() in ('spam', '1'):
                    label = "SPAM"
                else:
                    label = "HAM (Not Spam)"
                    
                prediction_text = f"This message is classified as: {label}"

            except Exception as e:
                prediction_text = f"An internal prediction error occurred: {e}"

    # Render the HTML template
    return render_template_string(
        SIMPLE_HTML, 
        prediction_text=prediction_text, 
        text_input=text_input
    )

if __name__ == '__main__':
    # Runs the app on http://127.0.0.1:5000/
    app.run(debug=True)
