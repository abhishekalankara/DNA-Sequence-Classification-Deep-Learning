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
