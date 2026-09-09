import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Safely resolve the path for Vercel's serverless environment
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'linear.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Property Value Predictor</title>
    <style>
        :root {
            --primary: #4F46E5;
            --primary-hover: #4338CA;
            --bg: #F3F4F6;
            --card-bg: #FFFFFF;
            --text: #1F2937;
        }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .container {
            background-color: var(--card-bg);
            padding: 2.5rem;
            border-radius: 16px;
            /* Main card soft shadow */
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1), 0 5px 15px rgba(0, 0, 0, 0.05);
            width: 100%;
            max-width: 700px;
        }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
            color: var(--primary);
            /* Text shadow for depth */
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        label {
            font-weight: 600;
            font-size: 0.9rem;
            color: #4B5563;
        }
        input {
            padding: 0.75rem;
            border: 1px solid #D1D5DB;
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.3s ease;
            /* Inset shadow for input depth */
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.03);
        }
        input:focus {
            outline: none;
            border-color: var(--primary);
            /* Outer glow shadow on focus */
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2), inset 0 2px 4px rgba(0,0,0,0.02);
        }
        button {
            margin-top: 1rem;
            width: 100%;
            padding: 1rem;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            /* Button drop shadow */
            box-shadow: 0 4px 6px rgba(79, 70, 229, 0.3);
        }
        button:hover {
            background-color: var(--primary-hover);
            /* Elevated shadow on hover */
            box-shadow: 0 8px 15px rgba(79, 70, 229, 0.4);
            transform: translateY(-2px);
        }
        .result {
            margin-top: 2rem;
            padding: 1.5rem;
            background: #EEF2FF;
            border-radius: 8px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: bold;
            color: var(--primary);
            /* Result box soft inset shadow */
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Property Value Predictor</h1>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label>Square Footage</label>
                <input type="number" step="any" name="Square_Footage" required>
            </div>
            <div class="form-group">
                <label>Number of Bedrooms</label>
                <input type="number" step="any" name="Num_Bedrooms" required>
            </div>
            <div class="form-group">
                <label>Number of Bathrooms</label>
                <input type="number" step="any" name="Num_Bathrooms" required>
            </div>
            <div class="form-group">
                <label>Year Built</label>
                <input type="number" step="any" name="Year_Built" required>
            </div>
            <div class="form-group">
                <label>Lot Size</label>
                <input type="number" step="any" name="Lot_Size" required>
            </div>
            <div class="form-group">
                <label>Garage Size</label>
                <input type="number" step="any" name="Garage_Size" required>
            </div>
            <div class="form-group">
                <label>Neighborhood Quality (e.g., 1-10)</label>
                <input type="number" step="any" name="Neighborhood_Quality" required>
            </div>
            <div class="form-group full-width">
                <button type="submit">Predict Value</button>
            </div>
        </form>
        
        {% if prediction is not none %}
        <div class="result">
            Estimated Value: ${{ "{:,.2f}".format(prediction) }}
        </div>
        {% endif %}
        
        {% if error %}
        <div class="result" style="color: #DC2626; background: #FEF2F2;">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template_string(HTML_TEMPLATE, prediction=None, error="Model failed to load. Please check linear.pkl.")
    
    try:
        # Extract features in the exact order the model expects
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        # Reshape for a single prediction prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=prediction)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=None, error=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
