"""Helper functions of the flower classification notebook."""

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.applications import mobilenet_v2
from plotly.subplots import make_subplots
from sklearn.metrics import ConfusionMatrixDisplay


def load_images(split, shuffle, data_dir, image_size):
    """Return a dataset of (images, labels) batches from one folder: "train", "validation" or "test"."""
    return keras.utils.image_dataset_from_directory(
        data_dir / split, image_size=image_size, batch_size=32, label_mode="int", shuffle=shuffle, seed=42
    )


def colour_name(hue):
    """Name the rough colour of a hue in degrees."""
    if hue < 15 or hue >= 345:
        return "red"
    for upper, name in [(40, "orange"), (70, "yellow"), (170, "green"), (260, "blue"), (345, "purple/magenta")]:
        if hue < upper:
            return name


def history_to_frame(history):
    """Return the metrics of a Keras History as a DataFrame with one row per epoch, numbered from 1."""
    curves = pd.DataFrame(history.history)
    curves["epoch"] = np.arange(1, len(curves) + 1)
    return curves


def plot_learning_curves(curves, title=None):
    """Return a figure with the accuracy and the loss of the training and the validation data per epoch."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
    for metric, col in [("accuracy", 1), ("loss", 2)]:
        for prefix, name, color in [("", "training", "#1f77b4"), ("val_", "validation", "#ff7f0e")]:
            fig.add_trace(
                go.Scatter(
                    x=curves["epoch"], y=curves[prefix + metric], name=name, mode="lines+markers",
                    line=dict(color=color), legendgroup=name, showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(title=title, width=1000, height=400)
    return fig


def count_correct_and_wrong(model, dataset):
    """Return how many photos of a dataset the model classifies correctly and how many wrongly.

    The dataset must not be shuffled, so that the predictions line up with the labels."""
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in dataset])
    predictions = model.predict(dataset, verbose=0).argmax(axis=1)
    n_correct = int((predictions == labels).sum())
    return n_correct, len(labels) - n_correct


def summarize_curves(curves, model_name, n_correct, n_wrong):
    """Return a one-row DataFrame with the metrics of the last epoch and the validation photo counts,
    labelled with the name of the model."""
    last = curves.iloc[-1]
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "train_accuracy": last["accuracy"],
                "val_accuracy": last["val_accuracy"],
                "train_loss": last["loss"],
                "val_loss": last["val_loss"],
                "val_correct": n_correct,
                "val_wrong": n_wrong,
            }
        ]
    )


def predict_with_confidence(model, dataset):
    """Return the true labels, the predicted labels and the confidence of every prediction (the probability
    the model gives to its predicted class) for a dataset.

    The dataset must not be shuffled, so that the predictions line up with the labels."""
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in dataset])
    probabilities = model.predict(dataset, verbose=0)
    return labels, probabilities.argmax(axis=1), probabilities.max(axis=1)


def plot_confusion_and_mistakes(model, dataset, class_names, title=None):
    """Plot the confusion matrix of a model on a dataset, then every misclassified photo with its true and
    predicted label and the confidence of the prediction. Return the indices of the misclassified photos in the
    dataset.

    The dataset must not be shuffled, so that the predictions line up with the labels."""
    images = np.concatenate([batch_images.numpy() for batch_images, _ in dataset]).astype("uint8")
    labels, predictions, confidence = predict_with_confidence(model, dataset)

    # rows are the true labels, columns the predicted labels
    ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=class_names, cmap="Blues")
    plt.title(title)
    plt.show()

    wrong = np.flatnonzero(predictions != labels)
    n_cols = 6
    n_rows = max(1, int(np.ceil(len(wrong) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(images[i])
        ax.set_title(
            f"true: {class_names[labels[i]]}\npredicted: {class_names[predictions[i]]} ({confidence[i]:.0%})",
            color="firebrick",
            fontsize=10,
        )
    plt.tight_layout(h_pad=3)  # extra space between the rows, so that the two-line titles stay readable
    plt.show()
    return wrong


def build_model(hp, image_size, n_classes):
    """Return a compiled dense head model on the frozen MobileNetV2 whose augmentation, dropout, size,
    L2 penalty and learning rate are chosen by the Keras Tuner hyperparameters `hp`.

    Like the other models it takes the raw photos (0-255) and scales them itself."""
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=42),
            keras.layers.RandomRotation(hp.Float("rotation", 0.0, 0.2, step=0.05), seed=42),
            keras.layers.RandomZoom(hp.Float("zoom", 0.0, 0.3, step=0.05), seed=42),
            keras.layers.RandomContrast(hp.Float("contrast", 0.0, 0.3, step=0.05), seed=42),
        ],
        name="augmentation",
    )
    base = mobilenet_v2.MobileNetV2(input_shape=image_size + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    l2 = keras.regularizers.L2(hp.Float("l2", 1e-5, 1e-1, sampling="log"))
    layers = [
        keras.Input(shape=image_size + (3,)),
        augmentation,
        keras.layers.Rescaling(1 / 127.5, offset=-1),
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(hp.Float("dropout", 0.2, 0.7, step=0.1)),
    ]
    hidden_units = hp.Choice("hidden_units", [0, 32, 64, 128])  # 0: no hidden layer, straight to the output
    if hidden_units:
        layers += [
            keras.layers.Dense(hidden_units, activation="relu", kernel_regularizer=l2),
            keras.layers.Dropout(hp.Float("hidden_dropout", 0.2, 0.7, step=0.1)),
        ]
    layers.append(keras.layers.Dense(n_classes, activation="softmax", kernel_regularizer=l2))

    model = keras.Sequential(layers)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        metrics=["accuracy"],
    )
    return model


def make_strong_augmentation():
    """Return a new augmentation model with the strong augmentation. Every layer has its own seed, so a new
    model gives the same random photos in every run; use a separate one for each purpose (training, preview)."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=42),
            keras.layers.RandomRotation(0.15, seed=42),  # up to 15 % of a full turn in either direction
            keras.layers.RandomZoom(0.15, seed=42),
            keras.layers.RandomTranslation(0.1, 0.1, seed=42),  # shift by up to 10 % of the height and the width
            keras.layers.RandomBrightness(0.2, seed=42),
            keras.layers.RandomErasing(factor=0.25, scale=(0.02, 0.2), seed=42),  # blank out a patch in a quarter of the photos
        ],
        name="strong_augmentation",
    )
