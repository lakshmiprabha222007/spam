import joblib
import os
import sys

# --- Configuration ---
MODEL_FILE = 'trained_spam_classifier_model.pkl'

def load_model():
    """Loads the trained machine learning pipeline."""
    try:
        # Check if the model file exists
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file '{MODEL_FILE}' not found.")
            print("Please ensure it is in the same directory as this script.")
            sys.exit(1)

        # Load the entire scikit-learn Pipeline (vectorizer + classifier)
        pipeline = joblib.load(MODEL_FILE)
        print(f"Model '{MODEL_FILE}' loaded successfully.")
        return pipeline

    except Exception as e:
        print("-" * 50)
        print(f"CRITICAL ERROR: Failed to load model. Details: {e}")
        print("This often indicates a scikit-learn version mismatch between training and loading.")
        print("-" * 50)
        sys.exit(1)

def predict_spam(pipeline, text_input):
    """Makes a prediction using the loaded pipeline."""
    try:
        # The pipeline expects a list/array of text strings
        data_to_predict = [text_input]
        
        # Make prediction
        prediction = pipeline.predict(data_to_predict)[0]
        
        # Format the result (assuming 1 is SPAM and 0 is HAM)
        if prediction == 1 or str(prediction).lower() in ('spam', '1'):
            label = "SPAM"
            color_code = '\033[91m'  # Red
        else:
            label = "HAM (Not Spam)"
            color_code = '\033[92m'  # Green
        
        # Reset color
        reset_code = '\033[0m'
        
        print("\n" + "=" * 30)
        print(f"Prediction: {color_code}{label}{reset_code}")
        print("=" * 30 + "\n")

    except Exception as e:
        print(f"\n[Prediction Error] Could not classify message. Details: {e}")

def main():
    """Main function to run the CLI tool."""
    pipeline = load_model()

    print("\n--- Spam Classifier CLI Tool ---")
    print("Enter a message to check (or type 'quit' to exit).")
    
    while True:
        try:
            user_input = input("Enter SMS > ")
            
            if user_input.lower() in ('quit', 'exit'):
                print("Exiting tool. Goodbye!")
                break
            
            if user_input:
                predict_spam(pipeline, user_input)
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting tool. Goodbye!")
            break
        except EOFError:
            # Handle Ctrl+D gracefully
            print("\nExiting tool. Goodbye!")
            break

if __name__ == "__main__":
    main()
