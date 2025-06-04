import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle

class HousePriceModel:
    """
    A machine learning model for predicting house prices based on various features.
    
    This class handles the entire pipeline including:
    - Data loading and preprocessing
    - Model training (Random Forest Regressor)
    - Model saving and loading
    - Making predictions on new data
    
    Attributes:
        model (RandomForestRegressor): The trained Random Forest model
        scaler (MinMaxScaler): Scaler for normalizing feature values
        features (list): List of feature names used for training
        target (str): Name of the target variable (price)
    """
    
    def __init__(self):
        """Initialize the HousePriceModel with default settings."""
        self.model = None  # Will hold our trained model
        self.scaler = MinMaxScaler()  # For feature scaling/normalization
        self.features = [
                        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                        'waterfront', 'view', 'condition', 'yr_built', 'yr_renovated'
                        ]  # Features used for prediction
        self.target = 'price'  # Target variable to predict
        
    def load_data(self, filename="data.csv"):
        """
        Load and preprocess the housing data from a CSV file.
        
        Args:
            filename (str): Path to the CSV file containing the data
            
        Returns:
            pd.DataFrame: Processed DataFrame with outliers removed
        """
        df = pd.read_csv(filename)
        
        # Outlier Handling (price only) - removes extreme values that could skew predictions
        # Using 5th and 95th percentiles instead of IQR for less aggressive filtering
        Q1 = df[self.target].quantile(0.05)
        Q3 = df[self.target].quantile(0.95)
        df = df[(df[self.target] >= Q1) & (df[self.target] <= Q3)]
        
        return df
        
    def train_model(self):
        """
        Train the Random Forest regression model on the housing data.
        
        Performs:
        - Data splitting into train/test sets
        - Feature scaling
        - Model training with predefined hyperparameters
        
        Returns:
            tuple: (trained model, scaler) for potential external use
        """
        df = self.load_data()
        X = df[self.features]  # Feature matrix
        y = df[self.target]  # Target vector
        
        # Split data into training and testing sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features to [0,1] range to ensure equal consideration of all features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train Random Forest model with optimized hyperparameters
        self.model = RandomForestRegressor(
            n_estimators=100,       # Number of trees in the forest
            max_depth=10,            # Maximum depth of each tree
            min_samples_split=10,    # Minimum samples required to split a node
            min_samples_leaf=4,      # Minimum samples required at each leaf node
            max_features=0.5,        # Fraction of features to consider for splits
            random_state=42,         # Seed for reproducibility
            n_jobs=-1               # Use all available cores for training
        )
        self.model.fit(X_train_scaled, y_train)
        
        return self.model, self.scaler
    
    def save_model(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        """
        Save the trained model and scaler to disk for later use.
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler
        """
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_saved_model(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        """
        Load a previously saved model and scaler from disk.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
            
        Returns:
            tuple: (loaded model, loaded scaler)
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        return self.model, self.scaler
    
    def predict(self, input_data):
        """
        Make a price prediction for a given set of house features.
        
        Args:
            input_data (dict or list): Feature values for prediction. 
                                      Should contain all features in self.features.
                                      
        Returns:
            float: Predicted house price
        """
        # Convert input data to DataFrame for consistent processing
        input_df = pd.DataFrame([input_data], columns=self.features)
        
        # Scale the input features using the previously fitted scaler
        scaled_input = self.scaler.transform(input_df)
        
        # Make prediction and return the single result
        prediction = self.model.predict(scaled_input)
        return prediction[0]