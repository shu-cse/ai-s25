import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNeuralNetwork:
    '''
    A simple feedforward neural network with one hidden layer,
    sigmoid activation, and a single (binary) output neuron.
    Training is done via backpropagation in the `forward` and 
    `backward` methods.
    '''

    # Constructor should set random weights and biases
    def __init__(self, num_features = 2, num_hidden_neurons = 3):
        self.weights_hidden = np.random.rand(
            num_features, num_hidden_neurons)
        self.bias_hidden    = np.zeros(
            (1, num_hidden_neurons))
        self.weights_output = np.random.rand(
            num_hidden_neurons, 1)
        self.bias_output    = np.zeros(1)
    
    # Forward pass version of the prediction
    def forward(self, X):
        self.z_hidden = X @ self.weights_hidden + self.bias_hidden # @ means dot matrix multiply
        self.a_hidden = sigmoid(self.z_hidden)
        self.z_output = self.a_hidden @ self.weights_output + self.bias_output
        self.a_output = sigmoid(self.z_output)
        return self.a_output
    
    # Backward pass for updating weights and biases
    def backward(self, X, y, learning_rate):
        if len(y.shape) <= 1: # y is just a list, not a full vector
            y = y.reshape(-1,1)

        # Compute the output error
        output_error = self.a_output - y
        output_delta = output_error * self.a_output * (1 - self.a_output)

        # Compute the hidden layer error
        hidden_error = output_delta @ self.weights_output.T
        hidden_delta = hidden_error * self.a_hidden * (1 - self.a_hidden)

        # Update weights and biases for the output layer
        self.weights_output -= learning_rate * self.a_hidden.T @ output_delta
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0)

        # Update weights and biases for the hidden layer
        self.weights_hidden -= learning_rate * X.T @ hidden_delta
        self.bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0)

    # Predict method to force binary decision
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int).flatten()

    def plot_decision_surface(self, X, y):
        if X.shape[1] == 2:
            # Create a mesh grid for plotting decision surface
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                np.arange(y_min, y_max, 0.01))

            # Predict class for each point in the mesh grid
            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot decision surface
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o')
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Decision Surface of Neural Network on Scatterplot of Feature Data")
            plt.show()
        else: 
            print("ERROR: Plotting of non-2D features not supported")


    
if __name__ == "__main__":
    # Create a simple dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR problem

    # Initialize the neural network
    nn = SimpleNeuralNetwork(num_features=2, num_hidden_neurons=3)

    # Train the neural network for one epoch
    learning_rate = 1.0
    nn.forward(X) # makes predictions using current weights
    nn.backward(X, y, learning_rate) # updates weights given error

    # Test the neural network
    predictions = nn.predict(X)
    print("Predictions:", predictions)