import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


def build_cnn_lstm_model(sequence_length, num_classes):
    """
    CNN-LSTM Hybrid Model
    """

    model = models.Sequential()

    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=5,
            activation="relu",
            input_shape=(sequence_length, 4)
        )
    )

    model.add(
        layers.MaxPooling1D(pool_size=2)
    )

    model.add(
        layers.BatchNormalization()
    )

    model.add(
        layers.Dropout(0.3)
    )

    model.add(
        layers.Bidirectional(
            layers.LSTM(
                128,
                return_sequences=True
            )
        )
    )

    model.add(
        layers.Dropout(0.3)
    )

    model.add(
        layers.Bidirectional(
            layers.LSTM(
                64
            )
        )
    )

    model.add(
        layers.Dropout(0.4)
    )

    model.add(
        layers.Dense(
            128,
            activation="relu"
        )
    )

    model.add(
        layers.Dropout(0.4)
    )

    model.add(
        layers.Dense(
            num_classes,
            activation="softmax"
        )
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


def train_model(

        model,

        X_train,

        y_train,

        X_test,

        y_test,

        epochs=50,

        batch_size=64):

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True

    )

    history = model.fit(

        X_train,

        y_train,

        validation_data=(X_test, y_test),

        epochs=epochs,

        batch_size=batch_size,

        callbacks=[early_stop],

        verbose=1

    )

    return history
    from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(

        model,

        X_test,

        y_test):

    predictions = model.predict(X_test)

    predicted = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predicted
    )

    precision = precision_score(
        y_test,
        predicted,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predicted,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predicted,
        average="weighted",
        zero_division=0
    )

    print("=" * 60)
    print("CNN-LSTM Performance")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print()

    print(
        classification_report(
            y_test,
            predicted,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test,
        predicted
    )

    plt.figure(figsize=(7,6))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()

    return accuracy


if __name__ == "__main__":

    from preprocessing import prepare_data

    (

        X_train,

        X_test,

        y_train,

        y_test,

        sequence_length,

        num_classes,

        encoder

    ) = prepare_data("../dataset/human_dataset.xlsx")

    model = build_cnn_lstm_model(

        sequence_length,

        num_classes

    )

    model.summary()

    train_model(

        model,

        X_train,

        y_train,

        X_test,

        y_test

    )

    evaluate_model(

        model,

        X_test,

        y_test

    )

    model.save("../models/cnn_lstm_model.keras")
