import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(train_file, valid_file):
    # Load the deep features dataset (.npz file)
    data = np.load(train_file, allow_pickle=True)
    valid = np.load(valid_file, allow_pickle=True)

    X = data['features']  # Shape should be (N, 13, 768)
    y = data['label']     # Labels for the training set

    X_valid = valid['features']  # Shape should be (N, 13, 768)
    y_valid = valid['label']     # Labels for the validation set

    return X, y, X_valid, y_valid

def preprocess_data(X, y, percent):
    # Use a portion of the dataset (e.g., 80% of the training data)
    num_samples = int(percent * X.shape[0])  # Calculate how many samples to use
    X_train = X[:num_samples]
    y_train = y[:num_samples]
    return X_train, y_train

def build_model():
    # Build the Unidirectional LSTM model
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(13, 768)))  # Correct input shape

    # Unidirectional LSTM layer
    model.add(layers.LSTM(2, return_sequences=False))  # Using 2 units

    # Fully connected layer
    model.add(layers.Dense(64, activation='relu'))

    # Output layer for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def evaluate_model(model, X_valid, y_valid):
    # Make predictions on the separate validation set (X_valid)
    y_pred = (model.predict(X_valid) > 0.5).astype("int32")

    # Evaluate the model
    accuracy = accuracy_score(y_valid, y_pred)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    class_report = classification_report(y_valid, y_pred)

    # Output the results
    print(f"Validation Accuracy: {accuracy}")

def predict_and_save(model, test_file, output_file):
    # Load the test dataset
    test_data = np.load(test_file, allow_pickle=True)
    X_test = test_data['features']  # Shape should be (N, 13, 768)

    # Make predictions on the test set
    y_test_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Save predictions to a text file
    np.savetxt(output_file, y_test_pred, fmt='%d',comments='')

    print(f"Predictions saved to {output_file}")

def main():
    train_file = "train_feature.npz"
    valid_file = "valid_feature.npz"
    test_file = "test_feature.npz"  # Path to the test data
    output_file = "pred_deepfeat.txt"  # Path to save predictions
    
    # Load data
    X, y, X_valid, y_valid = load_data(train_file, valid_file)
    print(X_valid.shape)
    print(y_valid.shape)

    # Preprocess data
    percent = 1  # You can change this to use a different percentage
    X_train, y_train = preprocess_data(X, y, percent)

    # Build and train the model
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid))

    # Evaluate the model
    evaluate_model(model, X_valid, y_valid)

    # Print model summary
    # model.summary()

    # Predict and save results
    predict_and_save(model, test_file, output_file)

if __name__ == "__main__":
    main()