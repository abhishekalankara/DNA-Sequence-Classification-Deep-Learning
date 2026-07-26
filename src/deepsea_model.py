import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def compute_weights(y_train):
    """
    Compute class weights for imbalanced dataset.
    """
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    return dict(enumerate(class_weights))


def build_deepsea_model(sequence_length, num_classes):
    """
    DeepSEA-inspired CNN model.
    """

    inputs = layers.Input(shape=(sequence_length, 4))

    x = layers.Conv1D(
        filters=320,
        kernel_size=26,
        activation="relu",
        padding="valid"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=13
    )(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(
        filters=480,
        kernel_size=13,
        activation="relu",
        padding="valid"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=7
    )(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(
        filters=960,
        kernel_size=7,
        activation="relu",
        padding="valid"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(
        925,
        activation="relu"
    )(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
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
    batch_size=64
):

    class_weights = compute_weights(y_train)

    early_stop = tf.keras.callbacks.EarlyStopping(
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

        class_weight=class_weights,

        callbacks=[early_stop],

        verbose=1

    )

    return history
  from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predicted_classes
    )

    precision = precision_score(
        y_test,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predicted_classes,
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
            predicted_classes,
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

    model = build_deepsea_model(
        sequence_length,
        num_classes
    )

    model.summary()

    history = train_model(
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

    model.save("../models/deepsea_model.keras")
