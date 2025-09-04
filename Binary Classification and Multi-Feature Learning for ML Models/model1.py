import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_data(train_data_path, valid_data_path):
    # Same as before...
    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)
    
    max_length = 20
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(train_data['input_emoticon'])

    X_train_seq = tokenizer.texts_to_sequences(train_data['input_emoticon'])
    X_valid_seq = tokenizer.texts_to_sequences(valid_data['input_emoticon'])

    X_train = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_valid = pad_sequences(X_valid_seq, maxlen=max_length, padding='post')

    y_train = train_data['label'].values
    y_valid = valid_data['label'].values
    
    return X_train, X_valid, y_train, y_valid, tokenizer

def create_model(vocab_size):
    # Same as before...
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=20),
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(train_data_path, valid_data_path):
    # Same as before...
    X_train, X_valid, y_train, y_valid, tokenizer = load_data(train_data_path, valid_data_path)
    
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_valid, y_valid))

    loss, accuracy = model.evaluate(X_valid, y_valid)
    print(f'Validation Accuracy: {accuracy:.4f}')

    # model.summary()
    
    return model, tokenizer

def make_predictions(model, tokenizer, test_data_path, output_file):
    
    # Load the test data
    test_data = pd.read_csv(test_data_path)
    
    # Preprocess the test data
    max_length = 20
    X_test_seq = tokenizer.texts_to_sequences(test_data['input_emoticon'])
    X_test = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

    # Make predictions
    predictions = model.predict(X_test)
    predicted_labels = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary labels

    # Save predictions to a text file
    with open(output_file, 'w') as f:
        for label in predicted_labels:
            f.write(f"{label}\n")

    print(f'Predictions saved to {output_file}')

def main():
    # Paths to data
    train_data_path = 'train_emoticon.csv'
    valid_data_path = 'valid_emoticon.csv'
    test_data_path = 'test_emoticon.csv'  # Path to the test dataset
    output_file = 'pred_emoticon.txt'  # Output file for predictions
    
    # Train the model
    model, tokenizer = train_model(train_data_path, valid_data_path)

    # Make predictions on the test data
    make_predictions(model, tokenizer, test_data_path, output_file)

if __name__ == '__main__':
    main()