import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModel, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import zipfile
import tensorflow.keras.backend as K
#import shutil
#from google.colab import files

# Set global seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

#remove the preprocessing- 81.6 dec1
#remove aug_data nov - 81.3

# Load and preprocess data
def load_data(train_path, dev_path):
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    # Emotion columns
    emotion_columns = ['admiration', 'amusement', 'gratitude', 'love', 'pride', 'relief', 'remorse']
    X_train = train_df['text'].values
    y_train = train_df[emotion_columns].values
    X_dev = dev_df['text'].values
    y_dev = dev_df[emotion_columns].values

    return X_train, y_train, X_dev, y_dev, emotion_columns

#inputs-pep- augment on 28nov
def prepare_inputs(texts, tokenizer, max_length=None):
    if max_length is None:
        lengths = [len(tokenizer.encode(text)) for text in texts]
        max_length = int(np.percentile(lengths, 95))
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
#create model- trans->lstm-bi->drop->dense->drop : dont change 28nov
def create_model(transformer_model, num_labels, lstm_units, dropout_rate):
    input_ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    # Transformer output
    transformer_output = transformer_model({"input_ids": input_ids, "attention_mask": attention_mask})[0]
    # Bi-Directional LSTM
    # lstm128 - dropout0.5 - dense64 - dropout0.5 dec4-83.4
    lstm_output = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units, return_sequences=False))(transformer_output)
    # Dropout
    x = keras.layers.Dropout(dropout_rate)(lstm_output)
    # Additional Dense Layer
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    #x = keras.layers.Dense(64, activation="relu")(x) # reducding dec 4
    #x = keras.layers.Dropout(dropout_rate)(x) 
    #Output Layer
    outputs = keras.layers.Dense(num_labels, activation="sigmoid")(x)

    return keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)

#compute and print metrics - all the metrics for labkes needed 
def compute_and_print_metrics(y_true, y_pred, emotion_columns): #class imbalace, 0.3,0.5 - 83.3
    y_pred_binary = (y_pred > 0.5).astype(int)
    print("\n--- Classification Metrics Per Label ---")
    for i, emotion in enumerate(emotion_columns):
        precision = precision_score(y_true[:, i], y_pred_binary[:, i])
        recall = recall_score(y_true[:, i], y_pred_binary[:, i])
        f1 = f1_score(y_true[:, i], y_pred_binary[:, i])
        print(f"{emotion.capitalize()}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

    micro_f1 = f1_score(y_true, y_pred_binary, average="micro")
    macro_f1 = f1_score(y_true, y_pred_binary, average="macro")
    print("\n--- Overall Metrics ---")
    print(f"Micro F1-Score: {micro_f1:.4f}, Macro F1-Score: {macro_f1:.4f}")

#export predictions to CSV and ZIP
def export_predictions_to_csv(texts, predictions, emotion_columns, output_csv="predict.csv", output_zip="predict.zip"):
    binary_predictions = (predictions > 0.5).astype(int)
    df = pd.DataFrame(binary_predictions, columns=emotion_columns)
    df.insert(0, "text", texts)
    df.to_csv(output_csv, index=False)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv)
    os.remove(output_csv)

#best 84.3 - dont use... dec 4

#creating callback for f1 so i can tarck and plan based on it
# class F1Callback(tf.keras.callbacks.Callback):
#     def __init__(self, validation_data):
#         super().__init__()
#         self.validation_data = validation_data

#     def on_epoch_end(self, epoch, logs=None):
#         val_data, val_labels = self.validation_data
#         val_predictions = self.model.predict(val_data)
#         val_pred_binary = (val_predictions > 0.5).astype(int)
#         val_f1 = f1_score(val_labels, val_pred_binary, average="micro")
#         print(f"\nEpoch {epoch + 1}: val_f1_score = {val_f1:.4f}")
#         logs["val_f1_score"] = val_f1
#training and evaluation
# Trying to use val_f1 as the early stoppimng metrics, not working

def f1_metric(y_true, y_pred):
    y_pred_binary = K.cast(K.greater(y_pred, 0.5), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred_binary, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_binary, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_binary), "float"), axis=0)

    f1 = 2 * tp / (2 * tp + fp + fn + K.epsilon())
    return K.mean(f1)

def train_emotion_classifier(train_path, dev_path, config):
    X_train, y_train, X_dev, y_dev, emotion_columns = load_data(train_path, dev_path)
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = TFAutoModel.from_pretrained(model_name)
    train_encodings = prepare_inputs(X_train, tokenizer)
        #prepareing validation data with same max length
    dev_encodings = prepare_inputs(X_dev, tokenizer, max_length=train_encodings["input_ids"].shape[1])

    model = create_model(
        transformer,
        len(emotion_columns),
        config["lstm_units"],
        config["dropout_rate"]
    )
# learning rate scheduler. 
#Source: https://keras.io/api/callbacks/learning_rate_scheduler/ 
#dont change - 84.1 best at 5 epoch - deacy 0.3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["initial_learning_rate"],
        decay_steps=config["decay_steps"],
        decay_rate=config["decay_rate"],
        staircase=True
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay= config["weight_decay"]),
        #optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule),
        #i had missed weight_deacy- set at 0.02 checked 0.2, 0025, 0.03
        loss="binary_crossentropy",
        metrics=["accuracy", f1_metric]
    )
    #dont chnage - final
    #early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6)
# learnign rate sched - 1e-5 84.3 ---- 2e - 5 5 epoch 84.2 --- check 10epoch patcience- 2
        # Early stopping monitoring val_f1_score
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_f1_metric", patience=3, mode="max", restore_best_weights=True
    )
    # Reduce learning rate on plateau
    #next cosine Anneling source: https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay/ 

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_f1_score",
        factor=0.2,
        patience=3,
        mode="max",
        min_lr=1e-6
    )
    model.fit(
        {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"]},
        y_train,
        validation_data=(
            {"input_ids": dev_encodings["input_ids"], "attention_mask": dev_encodings["attention_mask"]},
            y_dev
        ),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stopping, reduce_lr]
    )

    dev_predictions = model.predict({
        "input_ids": dev_encodings["input_ids"],
        "attention_mask": dev_encodings["attention_mask"]
    })
    #chech the metrics 
    compute_and_print_metrics(y_dev, dev_predictions, emotion_columns)
    export_predictions_to_csv(X_dev, dev_predictions, emotion_columns)
    return model, tokenizer

# Main script best yet: dont change dec 10;  0.8535
def main():
    train_path = "train.csv"
    dev_path = "dev.csv"

    config = {
        "initial_learning_rate": 1e-5,
        "decay_steps": 1000,
        "decay_rate": 1.0,
        "batch_size": 16, #32
        "epochs": 20, #typically stpos at 12-13. best model
        "lstm_units": 128, #256 
        "dropout_rate": 0.5, #0.55
        "weight_decay": 0.02
    }
    model, tokenizer = train_emotion_classifier(train_path, dev_path, config)
    model_path = "best_emotion_classifier"
    model.save(model_path, save_format = 'tf')
    tokenizer.save_pretrained("best_emotion_classifier")
    #shutil.make_archive("best_emotion_classifier", 'zip', "best_emotion_classifier")
    #files.download("best_emotion_classifier.zip")
    #files.download("predict.zip")

# dont use 265, 0.02 and 1.2 -83.42
#256, dropot 0.5, deacy 1.2 32 - 84.37

if __name__ == "__main__":
    main()
# best grid serch 
# applied grid search to find my best hyperpara tuning - Best Macro F1-Score: 0.8535 dec 10
#Source : https://stackoverflow.com/questions/64209804/hyperparameter-tuning-with-gridsearch-with-various-parameters
#next check: the optuna- source: https://optuna.readthedocs.io/en/stable/
"""
--- Testing Configuration: {'learning_rate': 1e-05, 'batch_size': 16, 'epochs': 5, 'dropout_rate': 0.3} ---
--- Testing Configuration: {'learning_rate': 1e-05, 'batch_size': 32, 'epochs': 5, 'dropout_rate': 0.5} ---
--- Testing Configuration: {'learning_rate': 1e-05, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.2, 'weight_decay': 0.002, 'patience': 3} ---
--- Testing Configuration: {'learning_rate': 1e-05, 'batch_size': 32, 'epochs': 5, 'dropout_rate': 0.3} ---
--- Testing Configuration: {'learning_rate': 1e-05, 'batch_size': 32, 'epochs': 10, 'dropout_rate': 0.2, 'weight_decay': 0.002, 'patience': 3} ---
--- Starting Grid Search ---
--- Testing Configuration: {'learning_rate': 1e-05, 'batch_size': 16, 'epochs': 5, 'dropout_rate': 0.55, 'weight_decay': 0.002, 'patience': 3} ---
--- Best Configuration: {'learning_rate': 1e-05, 'batch_size': 16, 'epochs': 20, 'dropout_rate': 0.2, 'weight_decay': 0.002, 'patience': 3} ---
Best Macro F1-Score: 0.8535 dec 10
--- Best Configuration: {'learning_rate': 1e-05, 'batch_size': 16, 'epochs': 5, 'dropout_rate': 0.55, 'weight_decay': 0.002, 'patience': 3} ---

"""
# Note to the evaluator: If this source code does not replicate the results, 
# I can provide the .h5 file of the best model. 

# I used it as a seperate .py file for laodimg my saved model. 
#predict 
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