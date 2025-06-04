from flask import Flask, render_template, request, redirect, url_for
from model import HousePriceModel
import os

app = Flask(__name__)

# Initialize the house price prediction model handler
model_handler = HousePriceModel()

# Model initialization logic - checks if saved model exists, otherwise trains a new one
if not (os.path.exists("model.pkl") and os.path.exists("scaler.pkl")):
    print("Training model...")
    model_handler.train_model()
    model_handler.save_model()
else:
    print("Loading saved model...")
    model_handler.load_saved_model()

@app.route('/')
def home():
    """
    Render the home page with the input form.
    
    Returns:
        Rendered HTML template: The index.html page with the input form
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the form submission.
    
    Processes the form data, makes a prediction using the trained model,
    and displays the result.
    
    Returns:
        If POST: Rendered result.html with prediction and input data
        Otherwise: Redirect to home page
    """
    if request.method == 'POST':
        try:
            # Collect and convert form data to the required format
            input_data = {
                'bedrooms': int(request.form['bedrooms']),
                'bathrooms': int(request.form['bathrooms']),
                'sqft_living': int(request.form['sqft_living']),
                'sqft_lot': int(request.form['sqft_lot']),
                'floors': int(request.form['floors']),
                'waterfront': int(request.form['waterfront']),
                'view': int(request.form['view']),
                'condition': int(request.form['condition']),
                'yr_built': int(request.form['yr_built']),
                'yr_renovated': int(request.form['yr_renovated'])
            }
            
            # Make prediction using the trained model
            prediction = model_handler.predict(input_data)
            
            # Format the prediction as currency for display
            predicted_price = "${:,.2f}".format(prediction)
            
            # Render results page with prediction and user inputs
            return render_template('result.html', 
                                prediction=predicted_price,
                                input_data=input_data)
        
        except (ValueError, KeyError) as e:
            # Handle potential form input errors gracefully
            print(f"Error processing input: {e}")
            return redirect(url_for('home'))
    
    # If not a POST request, redirect to home page
    return redirect(url_for('home'))

if __name__ == '__main__':
    """
    Main entry point for the Flask application.
    
    Runs the development server with debug mode enabled.
    """
    app.run(debug=True)