# SMICNet
import numpy
numpy.float = float
numpy.int = numpy.int_
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential, load_model, save_model

import Data_generator
from SMICNet import SMICNet
import time
import tensorflow as tf
import os
from tensorflow.python.client import device_lib

from tensorflow.python.client import device_lib

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(False)


os.environ["KERAS_BACKEND"] = "tensorflow"
tf.debugging.set_log_device_placement(False)
sess = tf.compat.v1.Session()
print(device_lib.list_local_devices())
print(tf.config.list_physical_devices("GPU"))
# test if GPU is enabled on local device: macOS m1 mps, Windows GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Device Name:", tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
tf.config.list_physical_devices("GPU")

 
train_data_gen11 = Data_generator.Data_augmentation_Generator(
    batch_size=4,
    selected_volumes=["0002"],  # train_ids
    steps1=4,
    steps2=4,
    steps3=8,
    alpha_range=10,
    beta_range=10,
    gamma_range=330,
    window_size=81,
    gradual_rotated_padding_size=3,
    rotated_augmentation_size=6,
)
valid_data_gen11 = Data_generator.Data_augmentation_Generator(
    batch_size=4,
    selected_volumes=["0002"],  # valid_ids
    steps1=4,
    steps2=4,
    steps3=8,
    alpha_range=10,
    beta_range=10,
    gamma_range=330,
    window_size=81,
    gradual_rotated_padding_size=3,
    rotated_augmentation_size=6,
)


class PlotTraining(Callback):
    def __init__(
        self, patience=3  # indice of nb of epochs to wait when training goes worse
    ):
        super(PlotTraining, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.wait = 0  # for early stopping

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])
        self.acc.append(logs["accuracy"])
        self.val_acc.append(logs["val_accuracy"])

        if (epoch + 1) % 4 == 0:  # every 2 epochs
            checkpoint_filepath = f"model_checkpoint_epoch_{epoch + 1:02d}.h5"
            save_model(self.model, checkpoint_filepath)

        self.plot(epoch)
        if epoch > 0 and self.val_loss[-1] < min(self.val_loss[:-1]):
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.model.stop_training = True
            print(f"\nEarly stopping was triggered. The training has stopped.")
            self.model.set_weights(self.best_weights)
            print(
                f"\nThe best model weights have been restored from the epoch {self.val_loss.index(min(self.val_loss)) + 1}."
            )

    def plot(self, epoch):
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))
        plt.plot(self.loss, "yo-", label="Training loss", markersize=4)
        plt.plot(self.val_loss, "ro-", label="Validation loss", markersize=4)
        plt.plot(self.acc, "go-", label="Training accuracy", markersize=4)
        plt.plot(self.val_acc, "bo-", label="Validation accuracy", markersize=4)

        plt.title("Training and validation metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.grid(True)

        for i, j in zip(
            range(epoch + 1), self.loss + self.val_loss + self.acc + self.val_acc
        ):
            plt.text(
                i,
                j,
                str(i + 1),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        if (epoch + 1) % 2 == 0:  # every 2 epochs
            plt.savefig(f"epoch_{epoch+1}_plot.png", dpi=400, bbox_inches="tight")
            plt.savefig(
                f"epoch_{epoch+1}_plot.svg", format="svg", dpi=400, bbox_inches="tight"
            )
            plt.savefig(
                f"epoch_{epoch+1}_plot.eps", format="eps", dpi=400, bbox_inches="tight"
            )
        #         plt.show()

plot_training = PlotTraining(patience=16)
# start the training on the fly
historySW12_4classe = SMICNet.fit(
    train_data_gen11,
    epochs=4,
    # steps_per_epoch=len(train_data_gen11)//32, # optional
    callbacks=[plot_training],
    validation_data=valid_data_gen11,
    verbose=1,
    shuffle=False,
)
end_time = time.time()
 
 