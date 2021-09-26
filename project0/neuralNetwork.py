import numpy as np

def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    #raise NotImplementedError

    print(inputs.shape)   #(2,1)
    print(weights.shape)  #(2,1)
    print(np.transpose(weights).shape) #transposing the weights matrix before dot product  (1,2)
    print(np.dot(np.transpose(weights),inputs)) #applying the dot product using the transposed weights matrix
    print(np.tanh(np.dot(np.transpose(weights),inputs))) #applying the activation function tanH
    
    return np.tanh(np.dot(np.transpose(weights),inputs))


print(neural_network(np.random.random([2,1]),np.random.random([2,1])))
