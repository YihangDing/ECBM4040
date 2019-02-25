#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


class MyGRUCell(RNNCell):
    """
    Your own basic GRUCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow GRUCell source code. If you're using Anaconda, it's located at
    anaconda_install_path/envs/your_virtual_environment_name/site-packages/tensorflow/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow GRUCell, but with your own language.
    """

    def __init__(self, num_units, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the GRU cell.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyGRUCell, self).__init__(_reuse=tf.AUTO_REUSE)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        params=[]
        params.append(num_units)
        params.append(activation or tf.tanh)
        self.params=params
        

    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units = output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        return self.params[0]
        

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        return self.params[0]

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow GRUCell has a lot of other features. But we will not try them. Focus on the
        very basic GRU functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In GRU there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (new_state, new_state). For details check TensorFlow GRUCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        #raise NotImplementedError('Please edit this function.')
        inputs_shape=inputs.get_shape()
        input_depth = inputs_shape[1].value
        
        self.gate_kernel=tf.get_variable("gate_kernel",[input_depth + self.params[0], 2 * self.params[0]],initializer=tf.glorot_uniform_initializer())
        self.gate_bias=tf.get_variable("gate_bias",[2*self.params[0]],initializer=tf.constant_initializer(1.0,dtype=tf.float32))
        
        gate_inputs = tf.matmul(tf.concat([inputs,state],1),self.gate_kernel)
        #gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
        gate_inputs = tf.nn.bias_add(gate_inputs,self.gate_bias)

        value = tf.sigmoid(gate_inputs)
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
        
        r_state = r * state
        
        #candidate & bias
        self.candidate_kernel=tf.get_variable("candidate_kernel",[input_depth + self.params[0],  self.params[0]],initializer=tf.glorot_uniform_initializer())
        self.candidate_bias=tf.get_variable("candidate_bias",[self.params[0]],initializer=tf.constant_initializer(1.0,dtype=tf.float32))
        
        candidate = tf.matmul(tf.concat([inputs, r_state], 1), self.candidate_kernel)
        #candidate = nn_ops.bias_add(candidate, self._candidate_bias)
        candidate=tf.nn.bias_add(candidate,self.candidate_bias)
        
        act=self.params[1]
        c = act(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h