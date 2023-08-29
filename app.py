from flask import Flask, jsonify
from flask_cors import CORS
from Datapipeline import *

app = Flask(__name__)
CORS(app)


@app.route('/')
def send_data():  # put application's code here
    try:
        data = get_data('BTCUSD',datetime(2023, 8, 20))
        if data is None:
            return jsonify({"error": "No data returned from get_data"}), 400

        indicators_df = apply_indicators(data)
        indicators_dict = indicators_df.to_json(orient='records')

        indicated_data = add_required_indicators(data)




        if indicators_dict is None:
            return jsonify({"error": "No data returned from apply_indicators"}), 400

        response_data = {
            'indicators': indicators_dict
        }

        return jsonify(response_data)
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({"error": "An error occurred"}), 500

@app.route('/predict')
def models():
    try:
        data = get_data('BTCUSD')
        highsNlows = get_HnL_Predictions(data)
        highsNlowsList = highsNlows.tolist()

        # predictions_df = pd.read_csv("predictions.csv")
        # predictions = predictions_df.to_json(orient="split")

        response_data = {
            "highsNlows": highsNlowsList,
            # "predictions": predictions
        }
        return jsonify(response_data)
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({"error": "An error occurred"})


if __name__ == '__main__':
    app.run()
