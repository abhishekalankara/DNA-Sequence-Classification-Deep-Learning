import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(path):
    """
    Load dataset and remove missing values.
    """
    df = pd.read_excel(path)
    df.dropna(subset=["sequence", "class"], inplace=True)
    return df


def pad_sequences(df):
    """
    Pad DNA sequences to equal length.
    """
    max_length = df["sequence"].apply(len).max()

    df["sequence"] = df["sequence"].apply(
        lambda x: x.ljust(max_length, "N")[:max_length]
    )

    return df, max_length


def one_hot_encode(sequence):
    """
    Convert DNA sequence into one-hot encoded matrix.
    """

    mapping = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "C": [0, 0, 0, 1],
        "N": [0, 0, 0, 0]
    }

    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence])


def prepare_data(path):

    df = load_dataset(path)

    df, sequence_length = pad_sequences(df)

    X = np.array(
        [one_hot_encode(seq) for seq in df["sequence"]]
    )

    encoder = LabelEncoder()

    y = encoder.fit_transform(df["class"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        sequence_length,
        len(np.unique(y)),
        encoder
    )


if __name__ == "__main__":

    dataset = "dataset/human_dataset.xlsx"

    (
        X_train,
        X_test,
        y_train,
        y_test,
        sequence_length,
        num_classes,
        encoder
    ) = prepare_data(dataset)

    print("Training Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))
    print("Sequence Length  :", sequence_length)
    print("Classes          :", num_classes)
