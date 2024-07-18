import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, static_url_path='', static_folder='.')

model = pickle.load(open("rf_acc_67.pkl", "rb"))
data_normalizer = pickle.load(open("normalizer.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result.html')
def result():
    return app.send_static_file('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        Warehouse_block = int(data["Warehouse_block"])
        Mode_of_Shipment = int(data["Mode_of_Shipment"])
        Customer_care_calls = int(data["Customer_care_calls"])
        Customer_rating = int(data["Customer_rating"])
        Cost_of_the_Product = int(data["Cost_of_the_Product"])
        Prior_purchases = int(data["Prior_purchases"])
        Product_importance = int(data["Product_importance"])
        Gender = int(data["Gender"])
        Discount_offered = int(data["Discount_offered"])
        Weight_in_gms = int(data["Weight_in_gms"])

        preds = [[Warehouse_block, Mode_of_Shipment, Customer_care_calls, Customer_rating, Cost_of_the_Product,
                  Prior_purchases, Product_importance, Gender, Discount_offered, Weight_in_gms]]
        normalized_preds = data_normalizer.transform(preds)

        prediction = model.predict(normalized_preds)
        probability = model.predict_proba(normalized_preds)[0][1]

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': round(probability * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=4000)
