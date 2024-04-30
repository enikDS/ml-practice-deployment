from flask import Flask, render_template, request, jsonify
from utils import model_predict

# Create instance of Flask class
app = Flask(__name__)

# Route / for home route for index.html
# Rendertemplate for dynamic HTML in Flask
@app.route('/')
def home():
    return render_template("index.html")

# ML model for mail prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form.get('email-content')

    prediction = model_predict(email)

    return render_template('index.html', prediction=prediction, email=email)

# API Endpoint for our app
@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    email = data['content']
    prediction = model_predict(email)
    return jsonify({'prediction': prediction, 'email': email})

# Runs the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)