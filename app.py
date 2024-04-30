from flask import Flask, render_template, request
import pickle

# Create instance of Flask class
app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

# Route / for home route for index.html
# Rendertemplate for dynamic HTML in Flask
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form.get('email-content')

    # X data (input)
    tokenized_email = cv.transform([email])

    # y Predictions
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template('index.html', prediction=prediction, email=email)

# Runs the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)