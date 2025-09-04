import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def load_data():
    # Load emoticon data (training)
    emoticon_data = pd.read_csv('train_emoticon.csv')
    emoticon_strings = emoticon_data.drop(columns=['label']).astype(str).apply(lambda x: ''.join(x), axis=1)
    
    # Convert emoticons to categorical integers
    unique_emoticons = list(set(''.join(emoticon_strings)))
    emoticon_to_int = {emoticon: i + 1 for i, emoticon in enumerate(unique_emoticons)}
    
    # Convert emoticon strings into sequences of integers
    emoticon_sequences = [[emoticon_to_int[char] for char in sequence] for sequence in emoticon_strings]
    emoticon_sequences_padded = np.array(emoticon_sequences)
    
    # Extract training labels
    emoticon_labels = emoticon_data['label'].values
    
    # Load deep feature data
    deep_feature_data = np.load('train_feature.npz')
    deep_features = deep_feature_data['features']
    
    # Load text sequence data
    text_seq_data = pd.read_csv('train_text_seq.csv')
    text_seq_features = text_seq_data.drop(columns=['label']).values
    
    return (emoticon_sequences_padded, emoticon_labels, deep_features, text_seq_features, emoticon_to_int)

def preprocess_data(emoticon_sequences_padded, deep_features, text_seq_features):
    # Handle NaN or infinite values
    deep_features = np.nan_to_num(deep_features)
    text_seq_features = np.nan_to_num(text_seq_features)
    
    # Scale the data
    scaler = StandardScaler()
    deep_features_flat = deep_features.reshape(deep_features.shape[0], -1)  # Flatten deep features
    pca = PCA(n_components=100)  # Reduce to 100 components using PCA
    deep_features_reduced = pca.fit_transform(deep_features_flat)
    deep_features_reduced = scaler.fit_transform(deep_features_reduced)
    text_seq_features = scaler.fit_transform(text_seq_features)
    
    # Combine all training features into one matrix
    combined_features = np.concatenate([emoticon_sequences_padded, deep_features_reduced, text_seq_features], axis=1)
    
    return combined_features, pca, scaler

def load_validation_data(emoticon_to_int, pca, scaler):
    # Load validation emoticon data
    validation_emoticon_data = pd.read_csv('valid_emoticon.csv')
    validation_emoticon_strings = validation_emoticon_data.drop(columns=['label']).astype(str).apply(lambda x: ''.join(x), axis=1)
    validation_emoticon_sequences = [[emoticon_to_int[char] for char in sequence] for sequence in validation_emoticon_strings]
    validation_emoticon_sequences_padded = np.array(validation_emoticon_sequences)

    # Load validation deep feature data
    validation_deep_feature_data = np.load('valid_feature.npz')
    validation_deep_features = np.nan_to_num(validation_deep_feature_data['features'])

    # Load validation text sequence data
    validation_text_seq_data = pd.read_csv('valid_text_seq.csv')
    validation_text_seq_features = np.nan_to_num(validation_text_seq_data.drop(columns=['label']).values)

    # Apply PCA transformation to validation deep features
    validation_deep_features_flat = validation_deep_features.reshape(validation_deep_features.shape[0], -1)
    validation_deep_features_reduced = pca.transform(validation_deep_features_flat)
    validation_text_seq_features = scaler.transform(validation_text_seq_features)

    # Combine validation features into one matrix
    X_val = np.concatenate([validation_emoticon_sequences_padded, validation_deep_features_reduced, validation_text_seq_features], axis=1)
    y_val = validation_emoticon_data['label'].values
    
    return X_val, y_val

def load_test_data(emoticon_to_int, pca, scaler):
    # Load test data
    test_emoticon_data = pd.read_csv('test_emoticon.csv')
    test_emoticon_strings = test_emoticon_data.astype(str).apply(lambda x: ''.join(x), axis=1)
    test_emoticon_sequences = [[emoticon_to_int.get(char, 0) for char in sequence] for sequence in test_emoticon_strings]
    test_emoticon_sequences_padded = np.array(test_emoticon_sequences)

    # Load test deep feature data
    test_deep_feature_data = np.load('test_feature.npz')
    test_deep_features = np.nan_to_num(test_deep_feature_data['features'])

    # Load test text sequence data
    test_text_seq_data = pd.read_csv('test_text_seq.csv')
    test_text_seq_features = np.nan_to_num(test_text_seq_data.values)

    # Apply PCA transformation to the test deep features
    test_deep_features_flat = test_deep_features.reshape(test_deep_features.shape[0], -1)
    test_deep_features_reduced = pca.transform(test_deep_features_flat)
    test_text_seq_features = scaler.transform(test_text_seq_features)

    # Combine test features into one matrix
    X_test = np.concatenate([test_emoticon_sequences_padded, test_deep_features_reduced, test_text_seq_features], axis=1)

    return X_test

def train_model(X_val, y_val, combined_features, emoticon_labels):
    # Hyperparameter tuning for Random Forest using validation data
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_val, y_val)

    # Train the final model on the entire training data
    best_rf_model = grid_search.best_estimator_
    best_rf_model.fit(combined_features, emoticon_labels)

    return best_rf_model

def save_predictions(model, X_test):
    # Make predictions on the test data
    predictions = model.predict(X_test)
    
    # Save predictions to a text file
    output_file_path = 'pred_combined.txt'

    # Save the predictions
    np.savetxt(output_file_path, predictions, fmt='%d', comments='')
    print(f"Predictions saved to {output_file_path}")

def main():

    # Load data
    emoticon_sequences_padded, emoticon_labels, deep_features, text_seq_features, emoticon_to_int = load_data()
    
    # Preprocess data
    combined_features, pca, scaler = preprocess_data(emoticon_sequences_padded, deep_features, text_seq_features)

    # Load validation data
    X_val, y_val = load_validation_data(emoticon_to_int, pca, scaler)

    # Load test data
    X_test = load_test_data(emoticon_to_int, pca, scaler)

    # Train model
    best_rf_model = train_model(X_val, y_val, combined_features, emoticon_labels)

    # Evaluate accuracy on validation set
    val_predictions = best_rf_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.2f}")

    # Save predictions
    save_predictions(best_rf_model, X_test)

if __name__ == "__main__":
    main()