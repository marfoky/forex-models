from flask import Flask, jsonify
from flask_cors import CORS
from Datapipeline import *

app = Flask(__name__)
CORS(app)


@app.route('/')
def send_data():  # put application's code here
    try:
        data = get_data('BTCUSD')
        if data is None:
            return jsonify({"error": "No data returned from get_data"}), 400

        indicators_df = apply_indicators(data)
        indicators_dict = indicators_df.to_json(orient='records')

        indicated_data = add_required_indicators(data)

        scaled_data = scale_data(indicated_data)

        realtime_data = prepare_realtime_data(scaled_data)

        print('realtime_data ', realtime_data)

        if indicators_dict is None:
            return jsonify({"error": "No data returned from apply_indicators"}), 400

        response_data = {
            'indicators': indicators_dict,
            'realtime_data': realtime_data.tolist()  # Convert numpy array to list if it's a numpy array
        }

        return jsonify(response_data)
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({"error": "An error occurred"}), 500



if __name__ == '__main__':
    app.run()
