import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)

# Load the model
try:
    with open("model.pkl", "rb") as model_file:
        loaded_data = pickle.load(model_file)
        model = loaded_data[0]  # Adjust based on your tuple structure
    print(f"Model loaded successfully. Model type: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", prediction_text="Model not loaded. Please check the server logs.")

    try:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)

        # Interpret the prediction result for better readability
        prediction_text = "has heart disease" if prediction[0] == 1 else "does not have heart disease"
        return render_template("index.html", prediction_text=f"The person {prediction_text}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template("index.html", prediction_text=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
