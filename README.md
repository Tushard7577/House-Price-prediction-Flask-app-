# House-Price-prediction-Flask-app-

A Flask-based web application that predicts house prices using a machine learning model. Users can input property features through a web form and receive instant price estimates.

## Features

- **User-Friendly Interface**: Clean two-column form layout
- **Machine Learning**: Random Forest regression model
- **Detailed Results**: Shows prediction
- **Validation**: Client-side form validation

## Technologies Used

- **Backend**: Python , Flask 
- **Frontend**: HTML5, CSS3
- **Machine Learning**: Scikit-learn 
- **Data Processing**: Pandas 

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. Clone the repository:

   git clone https://github.com/Tushard7577/House-Price-prediction-Flask-app-.git
   cd House Price prediction Flask app

2. Create and activate virtual environment:

python -m venv venv
On Windows:
.\venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Run the application:
   
python app.py

5. Access the application:

Open http://localhost:5000 in your web browser.

## Usage 

Fill out the form with property details:
Basic features (bedrooms, bathrooms, square ft)
Location and quality metrics
Year built and renovation information

Click "Predict Price" to submit the form

View the estimated price  on the results page

Click "Make Another Prediction" to return to the form
