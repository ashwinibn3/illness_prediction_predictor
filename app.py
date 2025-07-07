from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('illness_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = int(request.form['city'])
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        income = float(request.form['income'])

        # ✅ Print inputs to terminal (for your screenshot and testing)
        print("Received Input:")
        print("City:", city)
        print("Gender:", gender)
        print("Age:", age)
        print("Income:", income)

        input_data = np.array([[city, gender, age, income]])
        prediction = model.predict(input_data)[0]
        result = "Yes" if prediction == 1 else "No"

        return render_template('index.html', prediction_text=f"Illness Prediction: {result}")
    except:
        return render_template('index.html', prediction_text="Error: Invalid input.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    city = int(data['city'])
    gender = int(data['gender'])
    age = int(data['age'])
    income = float(data['income'])

    input_data = np.array([[city, gender, age, income]])
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
