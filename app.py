# import libraries
from flask import Flask, render_template, request
import numpy as np
import pickle


# pickle loading
model = pickle.load(open('cost_rf.pkl', 'rb'))

# creating app
app = Flask(__name__)

 
@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    promotion_name = request.form['promotion_name']
    store_type = request.form['store_type']
    store_city = request.form['store_city']
    store_sqft = request.form['store_sqft']
    frozen_sqft = request.form['frozen_sqft']
    meat_sqft = request.form['meat_sqft']
    coffee_bar = request.form['coffee_bar']
    salad_bar = request.form['salad_bar']
    prepared_food = request.form['prepared_food']
    florist = request.form['florist']
    media_type = request.form['media_type']

    arr = np.array([[promotion_name, store_type, store_city, store_sqft,
       frozen_sqft, meat_sqft, coffee_bar, salad_bar, prepared_food,
       florist, media_type]],dtype=float)

    pred = model.predict(arr)
    return render_template('nextpage.html', data=pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False) 