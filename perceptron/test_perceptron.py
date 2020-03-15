import numpy as np
import matplotlib.pyplot as plt

from perceptron.step_perceptron import StepPerceptron

# Training inputs and desired outputs for AND gate
# see what is initial output is for the 4 possible 2-bit inputs

if __name__ == "__main__":
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets_AND = np.array([0, 0, 0, 1])

    # Tell the perceptron how many inputs its getting.
    perceptron = StepPerceptron(2)
    print(perceptron.feed_forward(inputs))

    # train p to behave like AND gate
    epochs = 30

    # Cost is used for determining how well the algorithm is performing.
    costs = perceptron.train(inputs, targets_AND, epochs, 0.1)

    # check output again after training
    print(perceptron.feed_forward(inputs))

    eps = [e for e in range(epochs)]
    plt.plot(eps, costs)
    plt.title("Step Perceptron")
    plt.ylabel("epochs")
    plt.xlabel("cost")
    plt.show()
