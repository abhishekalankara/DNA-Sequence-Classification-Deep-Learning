import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dropout,
    Flatten,
    Dense
)


def build_cnn_model(sequence_length, num_classes):

    model = Sequential([

        Conv1D(
            filters=32,
            kernel_size=8,
            activation='relu',
            input_shape=(sequence_length, 4)
        ),

        MaxPooling1D(pool_size=2),

        Dropout(0.3),

        Conv1D(
            filters=64,
            kernel_size=8,
            activation='relu'
        ),

        MaxPooling1D(pool_size=2),

        Dropout(0.3),

        Flatten(),

        Dense(
            128,
            activation='relu'
        ),

        Dropout(0.3),

        Dense(
            num_classes,
            activation='softmax'
        )

    ])

    model.compile(

        optimizer='adam',

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']

    )

    return model


def train_model(

    model,

    X_train,

    y_train,

    epochs=15,

    batch_size=32

):

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.1,

        epochs=epochs,

        batch_size=batch_size,

        verbose=1

    )

    return history


def evaluate_model(

    model,

    X_test,

    y_test

):

    loss, accuracy = model.evaluate(

        X_test,

        y_test,

        verbose=0

    )

    print()

    print("=" * 50)

    print("Test Accuracy :", round(accuracy * 100, 2), "%")

    print("=" * 50)

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

    model = build_cnn_model(

        sequence_length,

        num_classes

    )

    model.summary()

    train_model(

        model,

        X_train,

        y_train

    )

    evaluate_model(

        model,

        X_test,

        y_test

    )

    model.save("../models/cnn_model.keras")
