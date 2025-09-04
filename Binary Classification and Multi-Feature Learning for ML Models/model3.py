import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GaussianNoise

def load_data(train_file, valid_file):
    # Load the dataset
    train_text_seq = pd.read_csv(train_file)
    valid_text_seq = pd.read_csv(valid_file)

    # Extract sequences and labels
    X_train_text = train_text_seq['input_str']
    y_train_text = train_text_seq['label']
    X_valid_text = valid_text_seq['input_str']
    y_valid_text = valid_text_seq['label']

    return X_train_text, y_train_text, X_valid_text, y_valid_text

def preprocess_data(X_train_text, X_valid_text):
    # Tokenizer: Each digit is treated as a character
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(X_train_text)

    # Convert text to sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_valid_seq = tokenizer.texts_to_sequences(X_valid_text)

    # Pad sequences to ensure uniform length (50)
    X_train_padded = pad_sequences(X_train_seq, maxlen=50)
    X_valid_padded = pad_sequences(X_valid_seq, maxlen=50)

    return X_train_padded, X_valid_padded, tokenizer

def build_model(tokenizer):
    # Build a CNN+GRU hybrid model with reduced parameters
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=50))  # Reduced embedding dimension
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))  # Reduced number of filters in Conv1D
    model.add(GaussianNoise(0.1))
    model.add(MaxPooling1D(pool_size=2))  # Max pooling
    model.add(Bidirectional(GRU(16)))  # Reduced GRU units
    model.add(Dropout(0.15))  # Dropout
    model.add(Dense(16, activation='relu'))  # Further reduce dense units
    model.add(Dropout(0.15))  # Dropout for regularization
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def load_test_data(test_file):
    # Load the test dataset
    test_text_seq = pd.read_csv(test_file)
    X_test_text = test_text_seq['input_str']  # Extract test sequences
    return X_test_text

def predict_and_save(model, tokenizer, X_test_text, output_file):
    # Preprocess the test data
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    X_test_padded = pad_sequences(X_test_seq, maxlen=50)

    # Make predictions
    predictions = model.predict(X_test_padded)
    predicted_labels = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels

    # Save predictions to a text file
    with open(output_file, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label[0]}\n")  # Write each prediction in a new line

def main():
    train_file = 'train_text_seq.csv'
    valid_file = 'valid_text_seq.csv'
    test_file = 'test_text_seq.csv'  # Path to your test data
    output_file = 'pred_textseq.txt'  # Output file for predictions

    # Load and preprocess data
    X_train_text, y_train_text, X_valid_text, y_valid_text = load_data(train_file, valid_file)
    X_train_padded, X_valid_padded, tokenizer = preprocess_data(X_train_text, X_valid_text)

    # Build and train the model
    model = build_model(tokenizer)
    model.fit(X_train_padded, y_train_text, epochs=50, batch_size=32,
              validation_data=(X_valid_padded, y_valid_text))

    # Evaluate the model on the validation data
    loss, accuracy = model.evaluate(X_valid_padded, y_valid_text)
    print(f"Validation Accuracy (CNN+GRU Model): {accuracy:.4f}")

    # Model summary to check number of parameters
    # model.summary()

    # Load test data and make predictions
    X_test_text = load_test_data(test_file)
    predict_and_save(model, tokenizer, X_test_text, output_file)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()