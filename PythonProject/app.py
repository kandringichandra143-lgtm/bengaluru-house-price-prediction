from flask import Flask, render_template, request, jsonify
import pickle
import json
import os

app = Flask(__name__)

# Base directory for reliable file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
with open(os.path.join(BASE_DIR, 'banglore_home_prices_model.pickle'), 'rb') as f:
    model = pickle.load(f)

# Load the column information
with open(os.path.join(BASE_DIR, 'columns.json'), 'r') as f:
    data_columns = json.load(f)

# Extract locations list (from index 3 onwards)
raw_locations = data_columns['data_columns'][3:]

def format_location_name(loc):
    return ' '.join(word.capitalize() for word in loc.split())

formatted_locations = [
    {'value': loc, 'label': format_location_name(loc)}
    for loc in raw_locations
]
# Sort alphabetically by label for easier selection
formatted_locations = sorted(formatted_locations, key=lambda x: x['label'])


@app.route('/')
def home():
    return render_template('index.html', locations=formatted_locations)


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    return jsonify({
        'status': 'success',
        'locations': formatted_locations
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            location = str(data.get('location', '')).strip().lower()
            sqft = float(data.get('sqft', 0))
            bath = float(data.get('bath', 0))
            bhk = int(data.get('bhk', 0))
        else:
            location = str(request.form.get('location', '')).strip().lower()
            sqft = float(request.form.get('sqft', 0))
            bath = float(request.form.get('bath', 0))
            bhk = int(request.form.get('bhk', 0))

        # Validation
        if sqft <= 0:
            error_msg = 'Please enter a valid area in sq. ft.'
            if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'status': 'error', 'message': error_msg}), 400
            return render_template('index.html', locations=formatted_locations, error=error_msg)

        # Create input array
        x = [0] * len(data_columns['data_columns'])
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        # Find location column
        if location in data_columns['data_columns']:
            loc_index = data_columns['data_columns'].index(location)
            x[loc_index] = 1

        # Predict
        prediction = float(model.predict([x])[0])
        prediction = max(0.0, prediction)  # Ensure non-negative

        formatted_price = f'{prediction:.2f} Lakhs'

        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', ''):
            return jsonify({
                'status': 'success',
                'estimated_price': round(prediction, 2),
                'formatted_price': formatted_price,
                'location': location,
                'sqft': sqft,
                'bhk': bhk,
                'bath': bath
            })

        return render_template(
            'index.html',
            locations=formatted_locations,
            prediction_text=formatted_price,
            selected_location=location,
            selected_sqft=sqft,
            selected_bhk=bhk,
            selected_bath=bath
        )

    except Exception as e:
        error_msg = f'Prediction error: {str(e)}'
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': error_msg}), 500
        return render_template('index.html', locations=formatted_locations, error=error_msg)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

