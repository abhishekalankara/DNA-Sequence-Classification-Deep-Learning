import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def build_deepbind_model(sequence_length, num_classes):
    """
    DeepBind-inspired CNN using parallel convolution layers.
    """

    inputs = layers.Input(shape=(sequence_length, 4))

    conv3 = layers.Conv1D(
        filters=128,
        kernel_size=6,
        activation="relu",
        padding="same"
    )(inputs)

    conv5 = layers.Conv1D(
        filters=128,
        kernel_size=9,
        activation="relu",
        padding="same"
    )(inputs)

    conv7 = layers.Conv1D(
        filters=128,
        kernel_size=12,
        activation="relu",
        padding="same"
    )(inputs)

    merged = layers.Concatenate()([
        conv3,
        conv5,
        conv7
    ])

    pooled = layers.MaxPooling1D(pool_size=2)(merged)

    dropout = layers.Dropout(0.3)(pooled)

    flatten = layers.Flatten()(dropout)

    dense = layers.Dense(
        256,
        activation="relu"
    )(flatten)

    dropout2 = layers.Dropout(0.4)(dense)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(dropout2)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer="adam",
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
        epochs=30,
        batch_size=32):

    history = model.fit(

        X_train,

        y_train,

        validation_data=(X_test, y_test),

        epochs=epochs,

        batch_size=batch_size,

        verbose=1

    )

    return history


def evaluate_model(
        model,
        X_test,
        y_test):

    predictions = model.predict(X_test)

    predicted = np.argmax(predictions, axis=1)

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

    print("\nAccuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            predicted,
            zero_division=0
        )
    )

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

    model = build_deepbind_model(
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

    model.save("../models/deepbind_model.keras")
