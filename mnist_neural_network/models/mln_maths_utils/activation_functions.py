import numpy as np
from typing import Callable


def derivative(func: Callable, input_value, delta=0.0001) -> float:
    """
    ####################################################################################################################
    This function calculates the derivative of a function.
    ####################################################################################################################
    :param func: the function to calculate the derivative of
    :param input_value: the input value of the function
    :param delta: the delta value to calculate the derivative
    :return: the slope of the function at the given point
    """
    return (func(input_value + delta) - func(input_value)) / delta


def derivative_linear_activation_function(input_value) -> float:
    """
    ####################################################################################################################
    This function calculates the derivative of the linear activation function.
    ####################################################################################################################
    :param input_value: the input value of the linear activation function
    :return: the slope of the linear activation function at the given point
    """
    return 1


def derivative_sigmoid_activation_function(input_value) -> float:
    """
    ####################################################################################################################
    This function calculates the derivative of the sigmoid activation function.
    ####################################################################################################################
    :param input_value: the input value of the sigmoid activation function
    :return: the slope of the sigmoid activation function at the given point
    """
    return input_value * (1 - input_value)


def linear_activation_function(input_value) -> float:
    """
    ####################################################################################################################
    This function calculates the linear activation function.
    ####################################################################################################################
    :param input_value: the input value of the linear activation function
    :return: the output of the linear activation function
    """
    return input_value


def sigmoid_activation_function(input_value) -> float:
    """
    ####################################################################################################################
    This function calculates the sigmoid activation function.
    ####################################################################################################################
    :param input_value: the input value of the sigmoid activation function
    :return: the output of the sigmoid activation function
    """
    return 1 / (1 + np.exp(-input_value))


def relu_activation_function(input_value) -> float:
    """
    ####################################################################################################################
    This function calculates the relu activation function.
    ####################################################################################################################
    :param input_value: the input value of the relu activation function
    :return: the output of the relu activation function
    """
    return np.maximum(0, input_value)


def softmax_activation_function(input_value) -> float:
    """
    ####################################################################################################################
    This function calculates the softmax activation function.
    ####################################################################################################################
    :param input_value: the input value of the softmax activation function
    :return: the output of the softmax activation function
    """
    return np.exp(input_value) / np.sum(np.exp(input_value), axis=0)


def tanh_activation_function(input_value) -> float:
    """
    ####################################################################################################################
    This function calculates the tanh activation function.
    ####################################################################################################################
    :param input_value: the input value of the tanh activation function
    :return: the output of the tanh activation function
    """
    return np.tanh(input_value)
