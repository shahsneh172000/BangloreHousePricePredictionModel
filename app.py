from flask import Flask , request , render_template

import ml_model 

app = Flask(__name__)

@app.route("/")
def home():
    loc_list1 = ml_model.loc_list
    return render_template('test.html', loc = loc_list1)

@app.route("/predict",methods=['POST'])
def predict():

    in_data = [x for x in request.form.values()]
    location = in_data[0]
    avalilability = in_data[1]
    area_type = in_data[2]
    total_sqft_int = in_data[3]
    bhk = in_data[4]
    balcony = in_data[5]
    bath = in_data[6]
    loc_list1 = ml_model.loc_list

    predicted_price = ml_model.predict_house_price(bath,balcony,total_sqft_int,bhk,area_type,avalilability,location)
    return render_template('test.html',pred_text = 'Estimated Price = {} Lakh'.format(round(predicted_price)))

app.run(debug=True)
