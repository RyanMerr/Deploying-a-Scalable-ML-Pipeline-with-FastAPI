import pytest
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ml.data import process_data
from ml.model import train_model, inference, performance_on_categorical_slice, load_model

# Load the census.csv data
@pytest.fixture
def load_data():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root
    project_root = os.path.dirname(current_dir)

# Construct the path to the data file
    data_path = os.path.join(project_root, 'data', 'census.csv')
    # project_path = "/Users/ryanmerrithew/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/"
    # data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)
    return data

# Test function to check if the data loading is successful
def test_data_loading(load_data):
    assert not load_data.empty, "Failed to load data"

# Test function to check if data splitting into train and test sets is correct
def test_data_splitting(load_data):
    train, test = train_test_split(load_data, test_size=0.2, random_state=42)
    assert len(train) > 0 and len(test) > 0, "Failed to split data into train and test sets"
    assert train.shape[0] > test.shape[0], "Incorrect data splitting ratio"

# Test function to check if model training and inference are successful
def test_model_training_and_inference(load_data):
    # Split data
    train, test = train_test_split(load_data, test_size=0.2, random_state=42)
    
    # Process data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Inference
    preds = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, preds)
    assert accuracy > 0.7, "Model accuracy below threshold"

# Test function to check if performance on categorical slices is correct
def test_performance_on_categorical_slices(load_data):
    # Load data
    data = load_data
    
    # Process data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Test performance on categorical slices
    for col in cat_features:
        for slicevalue in data[col].unique():
            p, r, fb = performance_on_categorical_slice(data, col, slicevalue, cat_features, "salary", encoder, lb, model)
            assert p >= 0 and r >= 0 and fb >= 0, f"Performance metrics below threshold for {col}={slicevalue}"
