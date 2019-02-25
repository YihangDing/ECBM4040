from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n=5):
    '''
    choose top_n most possible charactors in predictions
    this can help reduce some noise
    inputs:
    preds
    vocab_size
    top_n
    '''
    p = np.squeeze(preds)
    # set all values other that top_n choices to 0
    p[np.argsort(p)[:-top_n]] = 0
    # normalization
    p = p / np.sum(p)
    # randomly choose one
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


class CharRNN():
    def __init__(self, num_classes, batch_size=64, num_steps=50, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001, 
                 grad_clip=5, train_keep_prob=0.5, sampling=False):
        '''
        Initialize the input parameter to define the network
        inputs:
        :param num_classes: (int) the vocabulary size of your input data
        :param batch_size: (int) number of sequences in one batch
        :param num_steps: (int) length of each seuqence in one batch
        :param cell_type: your rnn cell type, 'LSTM' or 'GRU'
        :param rnn_size: (int) number of units in one rnn layer
        :param num_layers: (int) number of rnn layers
        :param learning_rate: (float)
        :param grad_clip: constraint of gradient to avoid gradient explosion
        :param train_keep_prob: (float) dropout probability for rnn cell training
        :param sampling: (boolean) whether train mode or sample mode
        '''
        # if not training
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.cell_type = cell_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        
        self.inputs_layer()
        self.rnn_layer()
        self.outputs_layer()
        self.my_loss()
        self.my_optimizer()
        self.saver = tf.train.Saver()
    
    
    def inputs_layer(self):
        '''
        build the input layer
        '''
        self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='targets')
        
        # add keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # one_hot encoding
        self.rnn_inputs = tf.one_hot(self.inputs, self.num_classes)
    
    
    def rnn_layer(self):
        '''
        build rnn_cell layer
        we will use the paramters:
        self.cell_type, self.rnn_size, self.keep_prob, self.num_layers,
        self.batch_size, self.rnn_inputs
        we have to define:
        self.rnn_outputs, self.final_state for later use
        '''
        #########################################################################################################
        #           TODO: finish the rnn layer definition, you should enable the selection of cell type         #
        #########################################################################################################
        #raise NotImplementedError('Please edit this function.')
        #self.outputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='outputs')
        #self.rnn_outputs=tf.one_hot(self.outputs,self.num_classes)
        #self.final_state=self.outputs.get_shape()
        
        # create single cell
        def single_cell(rnn_size, keep_prob):
            if self.cell_type=='LSTM':
                sin_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            else:
                sin_cell = tf.contrib.rnn.GRUCell(rnn_size)
            #sin_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            dropped = tf.nn.rnn_cell.DropoutWrapper(sin_cell, output_keep_prob=keep_prob)
            return dropped
        
        cells=[]
        for i in range(int(self.num_layers)):
            cells.append(single_cell(self.rnn_size,self.keep_prob))   
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        #use dynamic_rnn to stack single cell
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.rnn_inputs, initial_state=self.initial_state)

    
    def outputs_layer(self):
        ''' 
        build the output layer
        '''
        # concate the output of rnn_cell，example: [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]
        seq_output = tf.concat(self.rnn_outputs, axis=1) # tf.concat(concat_dim, values)
        # reshape
        x = tf.reshape(seq_output, [-1, self.rnn_size])
        
        # define softmax layer variables:
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))
        
        # calculate logits
        self.logits = tf.matmul(x, softmax_w) + softmax_b
        
        # softmax generate probability predictions
        self.prob_pred = tf.nn.softmax(self.logits, name='predictions')
        
        
    def my_loss(self):
        '''
        calculat loss according to logits and targets
        '''
        # One-hot coding
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
        
        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)
        
        
    def my_optimizer(self):
        '''
        build our optimizer
        Unlike previous worries of gradient vanishing problem,
        for some structures of rnn cells, the calculation of hidden layers' weights 
        may lead to an "exploding gradient" effect where the value keeps growing.
        To mitigate this, we use the gradient clipping trick. Whenever the gradients are updated, 
        they are "clipped" to some reasonable range (like -5 to 5) so they will never get out of this range.
        parameters we will use:
        self.loss, self.grad_clip, self.learning_rate
        we have to define:
        self.optimizer for later use
        '''
        # using clipping gradients
        #######################################################
        # TODO: implement your optimizer with gradient clipping
        #######################################################
        #raise NotImplementedError('Please edit this function.')
        #clipping gradients
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        self.optimizer = optimizer.apply_gradients(zip(gradients,tf.trainable_variables()))
        
        
    def train(self, batches, max_count, save_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            counter = 0
            new_state = sess.run(self.initial_state)
            # Train network
            for x, y in batches:
                counter += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss, 
                                                     self.final_state, 
                                                     self.optimizer], 
                                                     feed_dict=feed)
                    
                end = time.time()
                if counter % 200 == 0:
                    print('step: {} '.format(counter),
                          'loss: {:.4f} '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end-start)))
                    
                if (counter % save_every_n == 0):
                    self.saver.save(sess, "checkpoints/gi{}_l{}.ckpt".format(counter, self.rnn_size))
                    
                if counter >= max_count:
                    break
            
            self.saver.save(sess, "checkpoints/gi{}_l{}.ckpt".format(counter, self.rnn_size))
               
        
    def sample(self, checkpoint, n_samples, vocab_size, vocab_to_ind, ind_to_vocab, prime='You \n'):
        '''
        generate new text given the prime word
        inputs:
        :param n_samples: (int) number of characters you want to generate
        :param vocab_size: (int) number of vocabulary size of your input data
        :param vocab_to_ind, ind_to_vocab: mapping from unique characters to indices
        :param prime: (str) you new text starting word
        outputs:
        -a string of generated characters
        '''
        # change text into character list
        samples = [c for c in prime]
        #####################################################################################
        # TODO: implement sampling function
        # Hint: you should restore from saved checkpoint, start the model in sampling mode on 
        # prime word first and then generate new characters, remember to use pick_top_n to
        # reduce the noise.
        #####################################################################################
        #raise NotImplementedError('Please edit this function.')
        self.session = tf.Session()
        with self.session as sess:
            self.saver.restore(sess,checkpoint)
            #start as initial state
            new_state=sess.run(self.initial_state)

            #feed prime to model
            #pred = np.ones((vocab_size, ))  
            for c in prime:
                x = np.zeros((1, 1))
                x[0, 0] = vocab_to_ind[c]
                preds, new_state = sess.run([self.prob_pred, self.final_state],feed_dict={self.inputs: x,
                                                                    self.keep_prob:1.,
                                                                    self.initial_state: new_state})
            #reduce noise
            picked = pick_top_n(preds, vocab_size)
            # add c to samples
            samples.append(ind_to_vocab[picked])

            #for all num of characters: feed to model
            for i in range(n_samples):
                #x = np.zeros((1, 1))
                x[0, 0] = picked
                preds, new_state = sess.run([self.prob_pred, self.final_state],feed_dict={self.inputs: x,
                                                                      self.keep_prob: 1.,
                                                                      self.initial_state: new_state})
                #reduce noise
                picked = pick_top_n(preds, vocab_size)
                samples.append(ind_to_vocab[picked])

        #return np.array(samples)
        return(''.join(samples))






