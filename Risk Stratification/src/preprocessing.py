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
    data['chiefcomplaint'] = data['chiefcomplaint'].apply(nlp)

    # Initialize lists to store symptoms and body parts
    symptoms = []
    body_parts = []

    # Iterate over each row in the 'chiefcomplaint' column
    for doc in data['chiefcomplaint']:
        # Extract symptoms and body parts from the entities in the doc
        symptoms.append([ent.text for ent in doc.ents if ent.label_ == 'SYMPTOM'])
        body_parts.append([ent.text for ent in doc.ents if ent.label_ == 'BODY_PART'])
    
    # Add extracted features to the data
    data['symptoms'] = [', '.join(symptom) for symptom in symptoms]
    data['body_parts'] = [', '.join(symptom) for symptom in body_parts]
    
    return data

def preprocess_data(data_path):
    # Load the MIMIC-IV-ED dataset
    data_path = 'raw_data'
    
    # Get the current working directory (should be the notebook folder)
    current_dir = os.getcwd()

    # Construct the path to the raw_data folder
    project_dir = os.path.dirname(current_dir)  # Move up one level to the project directory
    raw_data_dir = os.path.join(project_dir, 'raw_data')

    # Load the MIMIC-IV-ED dataset
    edstays = pd.read_csv(os.path.join(raw_data_dir, 'edstays.csv'))
    diagnosis = pd.read_csv(os.path.join(raw_data_dir, 'diagnosis.csv'))
    medrecon = pd.read_csv(os.path.join(raw_data_dir, 'medrecon.csv'))
    triage = pd.read_csv(os.path.join(raw_data_dir, 'triage.csv'))
    vitalsign = pd.read_csv(os.path.join(raw_data_dir, 'vitalsign.csv'))

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

    # # Check for missing values
    # print("Missing values count:")
    # print(data.isnull().sum())
    # print("\nPercentage of missing values:")
    # print(data.isnull().mean() * 100)

    # Handle missing values
    # Drop missing values
    data = data.dropna(subset=['pain_x', 'pain_y'])
    data = data.dropna(subset=['acuity'])
    
    cols = ['temperature_x', 'heartrate_x', 'resprate_x', 'o2sat_x', 'sbp_x', 'dbp_x', 'temperature_y', 'heartrate_y', 'resprate_y', 'o2sat_y', 'sbp_y', 'dbp_y']

    for col in cols:
        data[col] = data[col].fillna(data[col].median())
        
    # Define values to drop
    drop_values = ['UA', 'Critical', 'does not scale', 'denies', 'uncooperative']

    # Create mask for 'pain_x' and 'pain_y'
    mask = ~data['pain_x'].isin(drop_values) & ~data['pain_y'].isin(drop_values)

    # Apply mask to data
    data = data[mask]

    # Drop missing values
    data = data.dropna(subset=['pain_x', 'pain_y'])
    
    # Split data into features and target
    X = data[['gender', 'chief_complaint_len', 'temperature_x', 'heartrate_x', 'resprate_x', 'o2sat_x', 'sbp_x', 'dbp_x', 'pain_x', 'temperature_y', 'heartrate_y', 'resprate_y', 'o2sat_y', 'sbp_y', 'dbp_y', 'pain_y','symptoms', 'body_parts']]
    y = data['acuity']

#     # Define numerical columns
#     num_cols = ['chief_complaint_len', 'temperature_x', 'heartrate_x', 'resprate_x', 'o2sat_x', 'sbp_x', 'dbp_x', 
#                 'temperature_y', 'heartrate_y', 'resprate_y', 'o2sat_y', 'sbp_y', 'dbp_y']

#     # Detect and remove outliers in numerical columns
#     Q1 = X[num_cols].quantile(0.25)
#     Q3 = X[num_cols].quantile(0.75)
#     IQR = Q3 - Q1
#     outlier_mask = ~((X[num_cols] < (Q1 - 1.5 * IQR)) | (X[num_cols] > (Q3 + 1.5 * IQR))).all(axis=1)

#     # Apply the mask to the entire DataFrame
#     X = X[outlier_mask]
#     y = y[outlier_mask]

#     print(f"\nNumber of outliers removed: {sum(~outlier_mask)}")
#     print(f"Percentage of outliers removed: {sum(~outlier_mask) / len(X) * 100:.2f}%")
    

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     print("y_train:")
#     print(y_train)

#     print("\ny_test:")
#     print(y_test)
    
#     print("changed")

    # Scale numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[['chief_complaint_len', 'temperature_x', 'heartrate_x', 'resprate_x', 'o2sat_x', 'sbp_x', 'dbp_x', 'pain_x', 'temperature_y', 'heartrate_y', 'resprate_y', 'o2sat_y', 'sbp_y', 'dbp_y', 'pain_y']])
    X_test_num = scaler.transform(X_test[['chief_complaint_len', 'temperature_x', 'heartrate_x', 'resprate_x', 'o2sat_x', 'sbp_x', 'dbp_x', 'pain_x', 'temperature_y', 'heartrate_y', 'resprate_y', 'o2sat_y', 'sbp_y', 'dbp_y', 'pain_y']])

    # One-hot encode categorical features
    encoder = OneHotEncoder()
    X_train_cat = encoder.fit_transform(X_train[['gender', 'symptoms', 'body_parts']])
    X_test_cat = encoder.transform(X_test[['gender', 'symptoms', 'body_parts']])

    # Combine numerical and categorical features
    X_train = np.concatenate((X_train_num, X_train_cat.toarray()), axis=1)
    X_test = np.concatenate((X_test_num, X_test_cat.toarray()), axis=1)
    
    # Reset the indices of y_train and y_test
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Create a single train.csv and test.csv file
    train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train, columns=['acuity'])], axis=1)
    test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test, columns=['acuity'])], axis=1)
    
    print("data preprocessing complete!")
    
    return train_data, test_data

    


    



    