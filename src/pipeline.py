import pandas as pd
from src.preprocessing import preprocess_data
from src.model import train_model

def run_pipeline(data_path, model_path):
    # Preprocess the data
    train_data, test_data = preprocess_data(data_path)

    # Train the model
    train_model(train_data, test_data, model_path)

def retrain_model(data_path, model_path):
    # Load the new data
    new_data = pd.read_csv(data_path)

    # Combine the new data with the existing train and test data
    train_data = pd.concat([train_data, new_data], ignore_index=True)

    # Split the combined data into train and test sets
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Retrain the model
    train_model(train_data, test_data, model_path)