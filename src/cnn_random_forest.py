import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier

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


def build_feature_extractor(sequence_length):

    inputs = layers.Input(shape=(sequence_length,4))

    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu"
    )(inputs)

    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    )(x)

    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)

    outputs = layers.Dense(
        128,
        activation="relu"
    )(x)

    model = models.Model(inputs, outputs)

    return model


def extract_features(model, X):

    return model.predict(X, verbose=0)


def train_random_forest(features, labels):

    rf = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        n_jobs=-1

    )

    rf.fit(features, labels)

    return rf


def evaluate_model(rf, features, labels):

    predictions = rf.predict(features)

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(
        labels,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        labels,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    print()

    print("="*60)

    print("CNN + Random Forest")

    print("="*60)

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print()

    print(classification_report(labels, predictions))

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(7,6))

    sns.heatmap(

        cm,

        annot=True,

        cmap="Blues",

        fmt="d"

    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("CNN + Random Forest")

    plt.show()


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

    feature_model = build_feature_extractor(sequence_length)

    train_features = extract_features(

        feature_model,

        X_train

    )

    test_features = extract_features(

        feature_model,

        X_test

    )

    rf = train_random_forest(

        train_features,

        y_train

    )

    evaluate_model(

        rf,

        test_features,

        y_test

    )
