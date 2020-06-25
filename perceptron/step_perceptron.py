import numpy as np

import perceptron.common_functions as cf


# Single Perceptron with n input neurons, n weights and a bias.
# To keep things simple as there is only 1 perceptron, we store weights as a vector of shape (n,)
# rather Nielsen's way as shape (1, n), a single row matrix.
class StepPerceptron:
    def __init__(self, n):
        """

        :param n: Number of inputs
        """
        self.n = n
        np.random.seed(2019)
        self.weights = np.random.randn(n, )  # vector of random weights from Gaussian or Normal distribution
        self.bias = np.random.randn()  # single random value

    # Note that in this Perceptron we are using the step() activation function, not sigmoid()
    def feed_forward(self, xs):
        """

        :param xs: propagates each input vector x in xs through the network (only 1 neuron here)
        :return: an array (list) of outputs, one for each input vector
        """
        res = [self.weights.dot(x) + self.bias for x in xs]
        return np.array([cf.step(r) for r in res])

    def train(self, xs, ys, epochs, eta):
        """
        The following operation illustrates the training algorithm for a Perceptron
        :param xs:
        :param ys:
        :param epochs: Number of iterations.
        :param eta: Learning rate for the algorithms.
        :return:
        """
        # we use cost to record the cost or loss in the Perceptron's current performance
        # over all the training input vectors for the current epoch or training iteration.
        # This is not needed for training, but we record it to plot training progress later on.

        cost = np.zeros((epochs,), dtype=float)

        for ep in range(epochs):
            delta_weight = np.zeros((self.n,), dtype=float)
            delta_bias = 0.0

            for x, y in zip(xs, ys):
                # x is usually an input vector, xs an array of vectors, y is a single value.
                # First compute weighted sum z, neuron's actual output a, and output error e.
                z = self.weights.dot(x) + self.bias
                a = cf.step(z)
                e = y - a  # error e = desired or target output less the actual output.

                # Compute the squared cost (loss) for each input and add them to compute
                # Cost for an entire training set.

                cost[ep] += 0.5 * e ** 2

                # print(z, a, e, cost[ep])
                # Learning formula for single perceptron with step activation
                # note del_w and x are vectors, so all the weight adjustments
                # for a single input are computed in 1 go here

                delta_weight += eta * e * x
                delta_bias += eta * e

            # Strictly should divide the sum by len(xs) to get average cost per training input
            # update the weights and bias with cumulative update based on all the training data
            # Note that in the spreadsheet example the weights are updated after each training
            # input whereas here we update them after each epoch.
            # cost[e] = cost[e] / len(xs)

            self.weights += delta_weight
            self.bias += delta_bias

            # print(cost[ep])
        return cost
