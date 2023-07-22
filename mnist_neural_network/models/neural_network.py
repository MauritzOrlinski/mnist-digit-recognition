import numpy as np
from mnist_neural_network.models.mln_maths_utils import activation_functions as af
from mnist_neural_network.models.mln_maths_utils import propagation as pr
import matplotlib.pyplot as plt


class mnistNeuralNetwork:
    def __init__(
        self,
        dataset: np.ndarray,
        labels: np.ndarray,
        learn_rate: float,
        train_percentage: float = 0.8,
    ) -> None:
        """
        ###############################################################################################################
        This function initializes the network with random weights and biases. The dataset and labels are given by the
        parameters. The learn rate is also given by the parameter. The train_percentage parameter is optional and
        tells the network how much of the dataset should be used for training. The rest of the dataset will be used
        for testing.
        ###############################################################################################################
        :param dataset: the dataset of the MNIST dataset
        :param labels: the labels of the MNIST dataset
        :param learn_rate: the rate of change of the weights and biases
        :param train_percentage: the percentage of the dataset that should be used for training
        :return: None
        """
        self.weights_input_to_hidden_layer = np.random.uniform(-0.5, 0.5, (20, 784))
        self.weights_hidden_to_output_layer = np.random.uniform(-0.5, 0.5, (10, 20))
        self.bias_input_to_hidden_layer = np.zeros((20, 1))
        self.bias_hidden_to_output_layer = np.zeros((10, 1))

        self.learn_rate = learn_rate

        self.dataset = dataset
        self.labels = labels
        self.dataset_train = dataset[: int(dataset.shape[0] * train_percentage)]
        self.dataset_test = dataset[int(dataset.shape[0] * train_percentage) :]
        self.labels_train = labels[: int(labels.shape[0] * train_percentage)]
        self.labels_test = labels[int(labels.shape[0] * train_percentage) :]

    def train_network(self, learn_iterations_per_data: int) -> list:
        """
        ###############################################################################################################
        This function trains the network with the train dataset. It will print the accuracy of the network in the
        current iteration in the console and returns a list with the accuracy of each iteration. The number of
        iterations per data is given by the learn_iterations_per_data parameter
        ###############################################################################################################
        :param learn_iterations_per_data: The number of iterations per data
        :return: None
        """
        accuracy_list: list = []
        for current_iteration in range(learn_iterations_per_data):
            nr_correct = 0
            for data, lable in zip(self.dataset_train, self.labels_train):
                data.shape += (1,)
                lable.shape += (1,)

                hidden_layer_pre_activation = pr.forward_propagation(
                    data,
                    self.weights_input_to_hidden_layer,
                    self.bias_input_to_hidden_layer,
                )
                hidden_layer_post_activation = af.sigmoid_activation_function(
                    hidden_layer_pre_activation
                )

                output_layer_pre_activation = pr.forward_propagation(
                    hidden_layer_post_activation,
                    self.weights_hidden_to_output_layer,
                    self.bias_hidden_to_output_layer,
                )
                output_layer_post_activation = af.sigmoid_activation_function(
                    output_layer_pre_activation
                )

                nr_correct += int(
                    np.argmax(output_layer_post_activation) == np.argmax(lable)
                )

                delta_output_layer = output_layer_post_activation - lable
                self.weights_hidden_to_output_layer -= (
                    self.learn_rate
                    * delta_output_layer
                    @ hidden_layer_post_activation.T
                )
                self.bias_hidden_to_output_layer -= self.learn_rate * delta_output_layer

                delta_hidden_layer = (
                    np.transpose(self.weights_hidden_to_output_layer)
                    @ delta_output_layer
                    * (
                        hidden_layer_post_activation
                        * (1 - hidden_layer_post_activation)
                    )
                )
                self.weights_input_to_hidden_layer -= (
                    self.learn_rate * delta_hidden_layer @ data.T
                )
                self.bias_input_to_hidden_layer -= self.learn_rate * delta_hidden_layer

            self.learn_rate *= 0.99
            print(
                f"Iterration: {current_iteration} with the accuracy: {round((nr_correct / self.dataset_train.shape[0]) * 100, 5)}%"
            )
            accuracy_list.append(
                round((nr_correct / self.dataset_train.shape[0]) * 100, 5)
            )
            nr_correct = 0
        return accuracy_list

    def test_network(self) -> float:
        """
        ###############################################################################################################
        This function tests the network with the test dataset. It will print the accuracy of the network in the console
        and returns it.
        ###############################################################################################################
        :param: None
        :return: the accuracy of the network in percent
        """
        nr_correct = 0
        for data, lable in zip(self.dataset_test, self.labels_test):
            data.shape += (1,)
            lable.shape += (1,)

            hidden_layer_pre_activation = pr.forward_propagation(
                data,
                self.weights_input_to_hidden_layer,
                self.bias_input_to_hidden_layer,
            )
            hidden_layer_post_activation = af.sigmoid_activation_function(
                hidden_layer_pre_activation
            )

            output_layer_pre_activation = pr.forward_propagation(
                hidden_layer_post_activation,
                self.weights_hidden_to_output_layer,
                self.bias_hidden_to_output_layer,
            )
            output_layer_post_activation = af.sigmoid_activation_function(
                output_layer_pre_activation
            )

            nr_correct += int(
                np.argmax(output_layer_post_activation) == np.argmax(lable)
            )

        print(f"Accuracy: {round((nr_correct / self.dataset_test.shape[0]) * 100, 2)} %")
        return round((nr_correct / self.dataset_test.shape[0]) * 100, 5)

    def test_with_picture(self, index: int, store: bool = False, filetype: str = "pdf") -> None:
        """
        ###############################################################################################################
        This function lets you test the network with a picture index. The picture will be given with matplotlib. The
        title of the window will be the prediction of the network and the actual value of the picture. If you want to
        store the picture in the visualization folder, you can set the store parameter to True, to get a pdf of the
        picture, otherwise the picture will be shown with a matplotlib window.
        ###############################################################################################################
        :param store: Tells the function if it should store the picture in the visualization folder. If none is given,
        the default value is False.
        :param index: The index of the picture you want to test the network with.
        :return: None
        """
        data = self.dataset[index]
        plt.imshow(data.reshape(28, 28), cmap="Greys")

        data.shape += (1,)
        hidden_layer_pre_activation = pr.forward_propagation(
            data, self.weights_input_to_hidden_layer, self.bias_input_to_hidden_layer
        )
        hidden_layer_post_activation = af.sigmoid_activation_function(
            hidden_layer_pre_activation
        )

        output_layer_pre_activation = pr.forward_propagation(
            hidden_layer_post_activation,
            self.weights_hidden_to_output_layer,
            self.bias_hidden_to_output_layer,
        )
        output_layer_post_activation = af.sigmoid_activation_function(
            output_layer_pre_activation
        )

        plt.title(
            f"Prediction: {np.argmax(output_layer_post_activation)}, Actual: {np.argmax(self.labels[index])}"
        )
        if store:
            plt.savefig(
                f"visualizations/picture_{index}.{filetype}",
            )
        else:
            plt.show()
