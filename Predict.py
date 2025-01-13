import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

# Set global seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Load and preprocess test data
def load_test_data(test_path):
    test_df = pd.read_csv(test_path)
    X_test = test_df['text'].values
    return X_test
from transformers import TFRobertaModel

def load_model_with_custom_objects(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"TFRobertaModel": TFRobertaModel}
    )

# Tokenization and encoding
def prepare_inputs(texts, tokenizer, max_length=None):
    if max_length is None:
        lengths = [len(tokenizer.encode(text)) for text in texts]
        max_length = int(np.percentile(lengths, 95))  # Use 95th percentile as max length
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

# Load the model and tokenizer
def load_model_and_tokenizer(model_path, tokenizer_path):
    model = load_model_with_custom_objects(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

output_path="test-submission.zip"
# Predict on test data
def predict_test_data(test_path, model_path, tokenizer_path, output_csv="test_predictions.csv"):
    # Load test data
    X_test = load_test_data(test_path)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    # Tokenize test data
    test_encodings = prepare_inputs(X_test, tokenizer)

    # Predict
    predictions = model.predict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"]
    })

    # Convert predictions to binary format
    binary_predictions = (predictions > 0.5).astype(int)

    # Save predictions to CSV
    emotion_columns = ['admiration', 'amusement', 'gratitude', 'love', 'pride', 'relief', 'remorse']
    df = pd.DataFrame(binary_predictions, columns=emotion_columns)
    df.insert(0, "text", X_test)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    output_path="test-submission.zip"
    df.to_csv(
    output_path,
    index=False,
    compression=dict(method="zip", archive_name="test-submission.csv")
    )
# Main script
def main():
    test_path = "data/test-in.csv"
    model_path = "best_emotion_classifier8535.h5"  # Path to the .h5 file
    tokenizer_path = "best_emotion_classifier"  # Path to the tokenizer directory

    # Predict test data
    predict_test_data(test_path, model_path, tokenizer_path)
    
# Save the predictions to a zip file



if __name__ == "__main__":
    main()