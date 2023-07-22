import numpy as np


def forward_propagation(data, weights, bias) -> np.ndarray:
    """
    ####################################################################################################################
    This function calculates the forward propagation of the neural network.
    ####################################################################################################################
    :param data: the input data to calculate the forward propagation
    :param weights: the weights of the neural network layer
    :param bias: the bias of the neural network layer
    :return: the output after the forward propagation
    """
    return bias + weights @ data


def cost_error_calculation(output) -> (float, int):
    """
    ####################################################################################################################
    calculates the cost error of the neural network output
    ####################################################################################################################
    :param output: the output of the neural network
    :return: the cost error of the neural network output
    """
    error = 1 / len(output) * np.sum((output - 1) ** 2, axis=0)
    nr_correct = int(np.argmax(output) == np.argmax(1))
    return error, nr_correct
