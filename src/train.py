from preprocessing import prepare_data

from cnn_model import (
    build_cnn_model,
    train_model
)

from evaluation import (
    evaluate_model,
    plot_history


)

DATASET = "dataset/human_dataset.xlsx"

(
    X_train,
    X_test,
    y_train,
    y_test,
    sequence_length,
    num_classes,
    encoder
) = prepare_data(DATASET)

print("Building CNN Model...")

model = build_cnn_model(
    sequence_length,
    num_classes
)

history = train_model(
    model,
    X_train,
    y_train
)

plot_history(history)

evaluate_model(
    model,
    X_test,
    y_test,
    encoder.classes_
)

model.save("models/cnn_model.keras")

print("\nTraining Completed Successfully!")
