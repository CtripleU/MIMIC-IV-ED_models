import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize SpaCy and NLTK components
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    
    # Tokenize and remove stop words
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def extract_features(data):
    # Extract features from chief complaint
    doc = nlp(data['chiefcomplaint'])
    symptoms = [ent.text for ent in doc.ents if ent.label_ == 'SYMPTOM']
    body_parts = [ent.text for ent in doc.ents if ent.label_ == 'BODY_PART']
    
    # Add extracted features to the data
    data['symptoms'] = [', '.join(symptoms)] * len(data)
    data['body_parts'] = [', '.join(body_parts)] * len(data)
    
    return data

def preprocess_data(data_path):
    # Load the MIMIC-IV-ED dataset
    data_path = './raw_data'
    edstays = pd.read_csv(f'{data_path}/edstays.csv')
    diagnosis = pd.read_csv(f'{data_path}/diagnosis.csv')
    medrecon = pd.read_csv(f'{data_path}/medrecon.csv')
    triage = pd.read_csv(f'{data_path}/triage.csv')
    vitalsign = pd.read_csv(f'{data_path}/vitalsign.csv')

    # Merge relevant tables
    data = pd.merge(edstays, triage, on=['subject_id', 'stay_id'], how='left')
    data = pd.merge(data, diagnosis, on=['subject_id', 'stay_id'], how='left')
    data = pd.merge(data, medrecon, on=['subject_id', 'stay_id'], how='left')
    data = pd.merge(data, vitalsign, on=['subject_id', 'stay_id'], how='left')

    # Preprocess chief complaint text
    data['chiefcomplaint'] = data['chiefcomplaint'].apply(preprocess_text)

    # Extract NLP features from chief complaint
    data = extract_features(data)

    # Feature engineering
    # data['age'] = # Calculate age from subject_id (using MIMIC-IV link)
    data['chief_complaint_len'] = data['chiefcomplaint'].str.len()
    data = pd.get_dummies(data, columns=['race', 'arrival_transport', 'disposition'])

    # Check for missing values
    print("Missing values count:")
    print(data.isnull().sum())
    print("\nPercentage of missing values:")
    print(data.isnull().mean() * 100)

    # Handle missing values
    data = data.dropna()  # Replace this with appropriate imputation or dropping strategy

    # Split data into features and target
    X = data[['gender', 'chief_complaint_len', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'symptoms', 'body_parts']]
    y = data['acuity']

    # Detect and remove outliers
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).all(axis=1)
    X = X[outlier_mask]
    y = y[outlier_mask]

    print(f"\nNumber of outliers removed: {sum(~outlier_mask)}")
    print(f"Percentage of outliers removed: {sum(~outlier_mask) / len(X) * 100:.2f}%")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']])
    X_test_num = scaler.transform(X_test[['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']])

    # One-hot encode categorical features
    encoder = OneHotEncoder()
    X_train_cat = encoder.fit_transform(X_train[['gender', 'symptoms', 'body_parts']])
    X_test_cat = encoder.transform(X_test[['gender', 'symptoms', 'body_parts']])

    # Combine numerical and categorical features
    X_train = np.concatenate((X_train_num, X_train_cat.toarray()), axis=1)
    X_test = np.concatenate((X_test_num, X_test_cat.toarray()), axis=1)

    # Create a single train.csv and test.csv file
    train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train, columns=['target'])], axis=1)
    test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test, columns=['target'])], axis=1)

    return train_data, test_data



# Preprocess the data
train_data, test_data = preprocess_data('data')

# Create the data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save the train and test data as CSV files
train_data.to_csv('data/train.csv', index=False)
test_data.to_csv('data/test.csv', index=False)