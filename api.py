from flask import Flask, render_template, request
from model_training import clean_data
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)
pipeline = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    rooms = int(request.form['rooms'])
    city = request.form['city']
    street = request.form['street']
    number_in_street = float(request.form['number_in_street'])
    city_area = request.form['city_area']
    type = request.form['type']
    condition = request.form['condition']
    has_elevator = bool(request.form.get('has_elevator'))
    has_parking = bool(request.form.get('has_parking'))
    has_balcony = bool(request.form.get('has_balcony'))
    has_garden = bool(request.form.get('has_garden'))
    floor = request.form['floor']
    total_floors = request.form['total_floors']
    has_air_conditioning = bool(request.form.get('has_air_conditioning'))
    has_storage = bool(request.form.get('has_storage'))
    has_security_bars = bool(request.form.get('has_security_bars'))
    has_mamad = bool(request.form.get('has_mamad'))
    has_Bars = bool(request.form.get('has_Bars'))
    handicap_friendly = bool(request.form.get('handicap_friendly'))
    furniture= request.form['furniture']
    description= request.form['description']

    input_data = {
        'Area': [area], 
        'room_number': [rooms], 
        'City': [city],
        'Street': [street],
        'number_in_street': [number_in_street],
        'city_area': [city_area],
        'type': [type],
        'condition': [condition],
        'hasElevator': [has_elevator],
        'hasParking': [has_parking],
        'hasBalcony': [has_balcony],
        'hasGarden': [has_garden],
        'floor_out_of': [' '.join(['floor', floor,'from',total_floors])],
        'hasAirCondition': [has_air_conditioning],
        'hasStorage': [has_storage],
        'hasSecurityBars': [has_security_bars],
        'hasMamad': [has_mamad],
        'hasBars': [has_Bars],
        'handicapFriendly': [handicap_friendly],
        'description': [description],
        'num_of_images': [np.nan],
        'entranceDate': [np.nan],
        'publishedDays': [np.nan],
        'index':[np.nan],
        'dis_from_Eilat':[np.nan],
        'furniture': [furniture]
        
    }

    input_df = pd.DataFrame(input_data)

    cleaned_data = clean_data(input_df)
    
    prediction = pipeline.predict(cleaned_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 5000))
    app.run()
