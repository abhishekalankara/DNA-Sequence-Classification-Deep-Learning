import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def create_results_folder():
    os.makedirs("results", exist_ok=True)


def evaluate_model(model, X_test, y_test, class_names):

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    create_results_folder()

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("results/confusion_matrix.png")

    plt.show()

    return accuracy, precision, recall, f1


def plot_history(history):

    create_results_folder()

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend(["Train","Validation"])

    plt.subplot(1,2,2)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend(["Train","Validation"])

    plt.tight_layout()

    plt.savefig("results/training_results.png")

    plt.show()
