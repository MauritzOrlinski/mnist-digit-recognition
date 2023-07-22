import numpy as np
import pathlib


def get_mnist() -> (np.ndarray, np.ndarray):
    """
    ####################################################################################################################
    This function returns the MNIST dataset and labels.
    ####################################################################################################################
    :return: the MNIST dataset and labels as a tuple
    """
    with np.load(f"{pathlib.Path(__file__).parent.parent.parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels