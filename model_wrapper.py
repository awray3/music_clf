""" Contains my wrapper for keras models. """

import os
import glob

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

class ModelWrapper:
    """
    Wrapper class that manages training, loading, hyperparameter tuning,
    and evaluation of a model.
    """

    def __init__(self, batch_size, model_dir, genre_labels, X_train, y_train, X_valid, y_valid,
                 X_test, y_test):
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.genre_labels = genre_labels
        self.model = None
        self.history = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.model_path = glob.glob(os.path.join(model_dir, "*.h5"))
        # Currently this will fail if there isn't already a file there.

        if self.model_path:
            self.model_path = self.model_path[0]
        else:
            self.model_path = os.path.join(self.model_dir, "model.h5")

        self.input_shape = (1,) + self.X_train.shape[1:]

    def attach_model(self, model):
        """ Attach a Keras model. """
        self.model = model
        self._compile()

    def load_model(self):
        """
        Loads the model from the model directory. Alternative to passing
        a new model.
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            raise ValueError("File Not Found. Please train the model.")

    def fit(self, num_epochs, verbose=1):
        """Fits the model."""
        checkpoint_callback = ModelCheckpoint(
            self.model_path,
            monitor="val_accuracy",
            verbose=verbose,
            save_best_only=True,
            mode="max",
        )
        reducelr_callback = ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_delta=0.01,
            verbose=verbose
        )
        early_stop = EarlyStopping(
            monitor="val_accuracy", patience=15, verbose=verbose, restore_best_weights=True
        )

        callbacks_list = [checkpoint_callback, reducelr_callback, early_stop]

        self.history = self.model.fit(
            x=self.X_train,
            y=self.y_train,
            epochs=num_epochs,
            validation_split=0.1,
            verbose=verbose,
            callbacks=callbacks_list,
        )

    def summary(self):
        """ give model summary """
        return self.model.summary()

    def _compile(self, loss="categorical_crossentropy"):
        """ custom compile method """
        self.model.compile(
            optimizer=Adam(lr=0.001),
            loss=loss,
            metrics=["accuracy"],
        )

    def evaluate(self, verbose=0):
        """ Custom evaluation method """

        print("Training set")
        self.model.evaluate(
            x=self.X_train,
            y=self.y_train,
            verbose=verbose)
        print("Testing set")
        _, accuracy = self.model.evaluate(
            x=self.X_test,
            y=self.y_test,
            verbose=verbose)

        return accuracy

    def plot_history(self):
        """ plots the history of the last training cycle """
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Valid"], loc="upper left")
        plt.show()

        plt.savefig(os.path.join(self.model_dir, "last_training_acc.png"))

        # Plot training & validation loss values
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Valid"], loc="upper left")
        plt.show()
        plt.savefig(os.path.join(self.model_dir, "last_training_loss.png"))

    def confusion_matrix(self):
        """ make a confusion matrix from the model."""

        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        classes = np.argmax(self.y_test, axis=1)

        print(confusion_matrix(classes, y_pred))

    def classification_report(self):
        """ make a classification report for the model."""

        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)
        classes = np.argmax(self.y_test, axis=1)

        print(
            classification_report(
                classes, y_pred,
                target_names=self.genre_labels
            )
        )
