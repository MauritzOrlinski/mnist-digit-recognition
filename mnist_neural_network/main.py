from models.neural_network import mnistNeuralNetwork
from dataset.get_dataset import get_mnist


def main():
    dataframe, labels = get_mnist()
    nn = mnistNeuralNetwork(dataframe, labels, 0.1)
    print("Training network...")
    number_of_iterations = int(input("Input the number of iterations you want to train the network with: "))
    nn.train_network(number_of_iterations)
    print("Testing network...")
    nn.test_network()
    print("Testing network with pictures choosen by you...")
    store = input("Do you want to store the pictures? (y/n): ") == "y"
    filetype = None
    if store:
        filetype = input("Which filetype do you want to use? (pdf/png): ")
    index = input("Please enter a index of a picture you want to test the network with: ")
    while index != "exit":
        if store:
            nn.test_with_picture(int(index), store, filetype)
        else:
            nn.test_with_picture(int(index), store)
        index = input("Please enter a index of a picture you want to test the network with: ")


if __name__ == "__main__":
    main()
