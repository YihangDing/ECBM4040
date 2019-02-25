import numpy as np
from ecbm4040.layer_funcs import *

class FullyConnectedLayer(object):
    def reset_layer(self, weight_scale=1e-2):
        """
        Reset weight and bias.
        
        Inputs:
        - weight_scale: (float) define the scale of weights
        """
        input_dim = self.input_dim
        hidden_dim = self.output_dim
        
        W = np.random.rand(input_dim, output_dim)
        b = np.zeros(output_dim)
        self.params = [W, b]
    
    def update_layer(self, params):
        """
        Update weight and bias
        """
        self.params = params

    
class DenseLayer(FullyConnectedLayer):
    """
    A dense hidden layer performs an affine transform followed by ReLU.
    Here we use ReLU as default activation function.
    """
    def __init__(self, input_dim, output_dim=100, weight_scale=1e-2):
        """
        Initialize weight W with random value and 
        bias b with zero.
        
        Inputs:
        - input_dim: (int) the number of input neurons, 
                     like D or D1xD2x...xDn.
        - output_dim: (int) the number of hidden neurons 
                      in this layer
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        W = weight_scale*np.random.rand(input_dim, output_dim)
        b = np.zeros(output_dim)
        self.params = [W, b]
    
    def feedforward(self, X):
        """
        Inputs:
        - X: (float) a tensor of shape (N,D) or 
             (N, D1, D2, ..., Dn).
        Returns:
        - out: output of shape (N, output_dim)
        """
        ################################################
        #TODO: out = ReLU(X*W + b). Use functions in   #
        #layer_funcs.py                                #
        ################################################
        W, b = self.params
        #def affine_forward(x, w, b):
        affine=affine_forward(X, W, b)
        self.A=affine
        #def relu_forward(x):
        out=relu_forward(affine)
        self.X=X

        
        ################################################
        #              END OF YOUR CODE                #
        ################################################
        return out
    
    def backward(self, dout):
        """
        Inputs:
        - dout: (float) a tensor with shape (N, hidden_dim)
        Returns:
        - dX: gradients wrt intput X, shape (N, D)
        - dW: gradients wrt W, shape (D, hidden_dim)
        - db: gradients wrt b, length hidden_dim
        """
        W, b = self.params
        X = self.X # cache input data
        A = self.A # cache intermediate affine result
        
        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        ################################################
        #TODO: derive the gradients wst to X, W, b. Use# 
        #layer_funcs.py                                #
        ################################################
        #relu_backward(dout, x)
        relu=relu_backward(dout, A)

        #def affine_backward(dout, x, w, b):
        dX, dW, db=affine_backward(relu, X, W, b)


        ################################################
        #              END OF YOUR CODE                #
        ################################################
        self.gradients = [dW, db]
        
        return dX

    
class AffineLayer(FullyConnectedLayer):
    """
    A dense hidden layer performs an affine transform followed by ReLU.
    Here we use ReLU as default activation function.
    """
    def __init__(self, input_dim, output_dim=100, weight_scale=1e-2):
        """
        Initialize weight W with random value and 
        bias b with zero.
        
        Inputs:
        - input_dim: (int) the number of input neurons, 
                     like D or D1xD2x...xDn.
        - output_dim: (int) the number of hidden neurons in this layer
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        W = weight_scale*np.random.rand(input_dim, output_dim)
        b = np.zeros(output_dim)
        self.params = [W, b]
    
    def feedforward(self, X):
        """
        Inputs:
        - X: (float) a tensor of shape (N,D) or 
             (N, D1, D2, ..., Dn).
        Returns:
        - out: output of shape (N, hidden_dim)
        """
        W, b = self.params
        ######################################################
        #TODO: out = X*W + b. Use functions in layer_funcs.py#
        ######################################################
        #def affine_forward(x, w, b):
        out=affine_forward(X, W, b)
        self.X=X


        
        #####################################################
        #                   END OF YOUR CODE                #
        #####################################################
        return out
    
    def backward(self, dout):
        """
        Inputs:
        - dout: (float) a tensor with shape (N, hidden_dim)
                Here hidden_dim denotes the number of hidden
                neurons
        Returns:
        - dX: gradients wrt intput X, shape (N, D)
        - dW: gradients wrt W, shape (D, hidden_dim)
        - db: gradients wrt b, length hidden_dim
        """
        W, b = self.params
        X = self.X
        
        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)
        ################################################
        #TODO: derive the gradients wrt to X, W, b. Use# 
        #layer_funcs.py                                #
        ################################################

        #def affine_backward(dout, x, w, b):
        dX, dW, db=affine_backward(dout, X, W, b)

        
        ################################################
        #              END OF YOUR CODE                #
        ################################################
        self.gradients = [dW, db]
        
        return dX
    